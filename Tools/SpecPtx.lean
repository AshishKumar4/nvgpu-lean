import NV.Launch.Load
import NV.Arch.Parts
import NV.PTX.Interp

/-!
`lake exe spec-ptx SPEC.json [--gpu NAME] [--schedules N [--seed S] [--delay D] [--fuel F]]
[--no-mma-values] [--pct] [--save DIR]`: a
launch spec run through the PTX interpreter in place of the SASS simulator, for a bit-exact
comparison of the two on whole kernels. With `--schedules`, each launch runs under `N` seeded
random schedules from seed `S` (`Machine.runRandom`: warps in random order, asynchronous
completions held back, each released with probability `1 / D` a step, at most `F` steps); the
first schedule that deadlocks or does not finish is reported with where every warp is, and its
seed, which reproduces it. `--pct`: priority schedules (`runRandom`'s `pct`). `--no-mma-values`: `tcgen05.mma` writes zeros (`Env.skipMma`), for
kernels whose control flow does not depend on its products. `--save DIR`: each schedule's final
buffers, `DIR/SEED-BUFFER.bin`. Each launch's code
(`X.cubin`, or a listing) has its PTX beside it as `X.ptx`; memory and parameter blocks are built
as `sim` builds them (the parameter layout from the ELF, which the PTX ABI's matches); module
`.global` variables go after the spec's buffers. The spec's `outputs` are written as `FILE.ptx`,
beside `sim`'s `FILE`.
-/

open NV NV.Core NV.Launch

def main (args : List String) : IO UInt32 := do
  let some path := args.find? (!·.startsWith "--") | do
    IO.eprintln "usage: lake exe spec-ptx SPEC.json [--gpu NAME]"; return 2
  let gpu := match args.dropWhile (· != "--gpu") with
    | _ :: g :: _ => g
    | _ => "h200"
  let some p := NV.Arch.parts.lookup gpu | do IO.eprintln s!"no parameter set {gpu}"; return 2
  let path : System.FilePath := path
  let dir := path.parent.getD "."
  let spec ← IO.ofExcept (Spec.parse (← IO.FS.readFile path))
  let work ← IO.FS.createTempDir
  -- Module variables after the buffers, on a fresh page.
  let moduleBase := (globalBase.toNat + (layout spec.buffers).2 + 65535) / 65536 * 65536
  let mut kernels : Array (String × Array SASS.Kernel) := #[]
  let mut progs : Array (String × NV.PTX.Sem.Prog) := #[]
  for l in spec.launches do
    if kernels.any (·.1 == l.code) then continue
    let (sass, elf) ← listing dir l.code work p.target.value
    let names := (spec.launches.filter (·.code == l.code)).toList.map (·.function)
    kernels := kernels.push (l.code, ← IO.ofExcept (SASS.Read.kernels sass elf (some names)))
    let ptx ← IO.FS.readFile (dir / (System.FilePath.mk l.code).withExtension "ptx")
    let prog ← IO.ofExcept (do NV.PTX.Sem.Prog.ofModule (← NV.PTX.read ptx) moduleBase.toUInt64)
    for n in names do progs := progs.push (n, prog)
  let find (code f : String) : Except String SASS.Kernel := do
    let some ks := kernels.toList.lookup code | throw s!"no listing {code}"
    (ks.find? (·.name == f)).elim (throw s!"launch: {f} not in {code}") pure
  let files ← readFiles spec dir
  let opt (name : String) (d : Nat) : Nat := match args.dropWhile (· != name) with
    | _ :: v :: _ => v.toNat?.getD d
    | _ => d
  if args.contains "--schedules" then
    let (n, seed0, delay, fuel) := (opt "--schedules" 1, opt "--seed" 0, opt "--delay" 8, opt "--fuel" 20_000_000)
    let launch (l : Core.Launch) : Except String (NV.PTX.Sem.Prog × NV.PTX.Launch) := do
      let some prog := progs.toList.lookup l.kernel.name | throw s!"no PTX for {l.kernel.name}"
      let pl : NV.PTX.Launch :=
        { entry := l.kernel.name, grid := l.grid, block := l.block, cluster := l.cluster,
          params := l.params, dynShared := l.dynShared,
          global := l.global.cover prog.globalBase prog.globalBytes, textures := l.textures }
      return (prog, pl)
    -- Each schedule replays the whole spec, the schedule's seed fixed for its launches.
    for seed in [seed0:seed0 + n] do
      -- Per launch: thread steps, tensor-core operations and asynchronous copies, so that a run
      -- that exits early (on an empty work range, say) shows as one.
      let run (_ : Unit) (l : Core.Launch) : Except String (Unit × Store × (Nat × Nat × Nat)) := do
        let (prog, pl) ← launch l
        let m ← NV.PTX.Machine.init prog pl
        match { m with skipMma := args.contains "--no-mma-values" }.runRandom seed delay fuel
            (pct := args.contains "--pct") with
        | .ok m => return ((), m.global, (m.steps, m.mmaOps, m.copyOps))
        | .error (e, where_) => throw (String.intercalate "\n  " (s!"{l.kernel.name}: {e}" :: where_))
      match spec.run files find () run with
      | .ok (mem, final, ws) =>
        let (steps, mmas, copies) := ws.foldl (fun (a, b, c) (x, y, z) => (a + x, b + y, c + z)) (0, 0, 0)
        IO.println s!"seed {seed}: completed, {steps} thread steps, {mmas} tensor-core operations, {copies} asynchronous copies"
        -- `--save DIR`: each buffer's final bytes, `DIR/<seed>-<buffer>.bin`.
        if let some dir := (args.dropWhile (· != "--save")).drop 1 |>.head? then
          IO.FS.createDirAll dir
          for b in spec.buffers do
            match mem.bytes final spec b.name with
            | .ok bytes => IO.FS.writeBinFile (System.FilePath.mk dir / s!"{seed}-{b.name}.bin") bytes
            | .error e => IO.eprintln e
      | .error e => IO.println s!"seed {seed}: {e}"; return 1
    return 0
  let run (_ : Unit) (l : Core.Launch) : Except String (Unit × Store × Unit) := do
    let some prog := progs.toList.lookup l.kernel.name | throw s!"no PTX for {l.kernel.name}"
    let pl : NV.PTX.Launch :=
      { entry := l.kernel.name, grid := l.grid, block := l.block, cluster := l.cluster,
        params := l.params, dynShared := l.dynShared,
        global := l.global.cover prog.globalBase prog.globalBytes,
        textures := l.textures }
    return ((), ← NV.PTX.interp prog pl, ())
  match spec.run files find () run with
  | .error e => IO.eprintln s!"PTX interpreter: {e}"; return 1
  | .ok (m, g, _) =>
    for (buffer, file) in spec.outputs do
      IO.FS.writeBinFile (dir / (file ++ ".ptx")) (← IO.ofExcept (m.bytes g spec buffer))
    IO.println s!"{spec.launches.size} launch(es) interpreted"
    return 0
