import NV.Launch.Spec
import NV.SASS.Read
import NV.SASS.Json

/-!
# Running a launch spec from files

What `lake exe sim` and `lake exe validate` share: finding each launch's kernel (a `cuobjdump`
listing, or a `.cubin`, `.ptx` or `.cu` file compiled with the toolchain on `PATH` for the part's
target, `Params.target`),
reading the files the spec names, and running every launch in order on one machine, whose caches
carry from one launch to the next as on the device.
-/

namespace NV.Launch

open NV.Core

/-- Run `cmd`; its standard output. -/
def tool (cmd : String) (args : Array String) : IO String := do
  let out ← IO.Process.output { cmd, args }
  if out.exitCode != 0 then throw (IO.userError s!"{cmd} {" ".intercalate args.toList}: {out.stderr}")
  return out.stdout

/-- The `cuobjdump -sass` and `-elf` listings of a kernel source: `<code>.sass` and `<code>.elf`
beside each other, or a `.cubin`, `.ptx` or `.cu` file (compiled for `target` into `work`). -/
def listing (dir : System.FilePath) (code : String) (work : System.FilePath) (target : String := "sm_90a") :
    IO (String × String) := do
  let path := dir / code
  let cubin ← match path.extension with
    | some "cubin" => pure path
    | some "ptx" =>
      let out := work / (path.fileStem.getD "kernel" ++ ".cubin")
      -- The PTX's `.target` line names the target it was written for; ptxas refuses a lower one,
      -- so it is rewritten to the part's (PTX that needs a higher target fails in ptxas then).
      let text ← IO.FS.readFile path
      -- Only the `.target` line changes; it comes before any code, so the text after it is
      -- copied whole.
      let ls : Array Substring.Raw := NV.SASS.Text.lines text
      let retargeted := match ls.find? (fun l => NV.SASS.Text.starts l.trim ".target ") with
        | some l => String.Pos.Raw.extract text 0 l.startPos ++ s!".target {target}" ++
            String.Pos.Raw.extract text l.stopPos text.rawEndPos
        | none => text
      let ptx := work / (path.fileStem.getD "kernel" ++ ".ptx")
      IO.FS.writeFile ptx retargeted
      discard <| tool "ptxas" #[s!"-arch={target}", "-m64", ptx.toString, "-o", out.toString]
      pure out
    | some "cu" =>
      let out := work / (path.fileStem.getD "kernel" ++ ".cubin")
      discard <| tool "nvcc" #["-cubin", "-O3", s!"-arch={target}", path.toString, "-o", out.toString]
      pure out
    | _ =>
      return (← IO.FS.readFile (dir / s!"{code}.sass"), ← IO.FS.readFile (dir / s!"{code}.elf"))
  return (← tool "cuobjdump" #["-sass", cubin.toString], ← tool "cuobjdump" #["-elf", cubin.toString])

/-- The input and expected-output files a spec names, read from `dir`. -/
def readFiles (s : Spec) (dir : System.FilePath) : IO Files := do
  let paths := s.buffers.filterMap (fun b => match b.init with | .file p => some p | _ => none) ++
    s.expect.map (·.file)
  let mut data : Array (String × ByteArray) := #[]
  for path in paths do
    data := data.push (path, ← IO.FS.readBinFile (dir / path))
  return fun path => (data.toList.lookup path).elim (throw s!"no file {path}") pure

/-- A simulated spec: the spec, its files, the memory before and after, and each launch's kernel
and counters. -/
structure Run where
  spec : Spec
  files : Files
  memory : Memory
  final : Store
  launches : Array (String × Counters)
  /-- Each launch's device `printf` output, rendered, in the order it was produced. -/
  printf : Array (Array String) := #[]
  /-- Each launch's first hazards (`Result.hazards`). -/
  hazards : Array (Array String) := #[]
  /-- Each launch's data races (`Result.races`, with `races`). -/
  races : Array (Array String) := #[]
  /-- Each launch's kernel and warps' traces, when recorded. -/
  traces : Array (SASS.Kernel × Array WarpTrace) := #[]

/-- A store's bytes, `[base, base + size)`. -/
def storeBytes (s : Store) : ByteArray := Id.run do
  let mut b := ByteArray.emptyWithCapacity s.size
  for i in [0:s.size] do b := b.push (s.readByte i)
  return b

/-- `exportLaunch`'s files but the global memory's: `const0.bin`, `kernel.jsonl` and `meta.txt`,
`extra` its further lines (the global memory's segments). -/
def exportLaunchFiles (p : Params) (dir : System.FilePath) (l : Core.Launch) (extra : List String) :
    IO Unit := do
  IO.FS.createDirAll dir
  let k := l.kernel
  IO.FS.writeBinFile (dir / "const0.bin") (storeBytes (Core.constBank0 l))
  let (gx, gy, gz) := l.grid
  let (bx, by_, bz) := l.block
  let (cx, cy, cz) := l.clusterShape
  IO.FS.writeFile (dir / "kernel.jsonl") ("\n".intercalate (SASS.Json.kernelLines k).toList ++ "\n")
  let inits := (List.range (cx * cy * cz)).filterMap fun rank =>
    let ws := Core.reservedInit p (cx * cy * cz) rank
    if ws.isEmpty then none
    else some (s!"sharedInit{rank} " ++ " ".intercalate (ws.toList.map fun (o, w) => s!"{o} {w.toNat}"))
  IO.FS.writeFile (dir / "meta.txt") <| "\n".intercalate (
    [s!"kernel {k.name}", s!"sm {k.sm}", s!"grid {gx} {gy} {gz}", s!"block {bx} {by_} {bz}",
     s!"cluster {cx} {cy} {cz}", s!"shared {k.sharedReserved + k.sharedBytes + l.dynShared}",
     s!"reserved {k.sharedReserved}", s!"local {k.localBytes}", s!"regs {k.regFile}"] ++ extra ++
     inits) ++ "\n"

/-- A launch as the accelerated mode's harness (`tools/lift`) loads it, into `dir`: `const0.bin`
(constant bank 0 at launch, `constBank0`), `global.bin` (the global memory at launch, the parameter
block and the module's data written), `meta.txt` (one `key value` per line: the kernel, its target, grid, block, cluster,
shared and local sizes, the global store's base, and per cluster rank `sharedInit<rank>` the
words its reserved shared memory holds at launch, `Core.reservedInit`), and after the run
`final.bin` (the global memory after it). -/
def exportLaunch (p : Params) (dir : System.FilePath) (l : Core.Launch) : IO Unit := do
  let k := l.kernel
  let mut g := l.global
  -- The parameter block where bank 0 says it is (`Core.paramAddrOf`), as the engine places it.
  let pa := Core.paramAddrOf l
  g := g.cover pa (max 1 l.params.size)
  for i in [0:l.params.size] do g := g.writeByte ((pa - g.base).toNat + i) l.params[i]!
  for (addr, bytes) in k.data do
    g := g.cover addr.toUInt64 bytes.size
    for i in [0:bytes.size] do g := g.writeByte (addr - g.base.toNat + i) bytes[i]!
  exportLaunchFiles p dir l [s!"globalBase {g.base.toNat}", s!"globalSize {g.size}"]
  IO.FS.writeBinFile (dir / "global.bin") (storeBytes g)

/-- Run a spec's steps on `p`: its kernels and files resolved from `dir`. `fresh` gives every
launch a fresh machine instead of the previous launch's. `exportDir`: each launch exported for the
accelerated mode (`exportLaunch`), into `exportDir/<launch number>`. `workers`: `Launch.workers`.
`untilCycle`: `Launch.untilCycle`, for every launch. `spans`: `Launch.spans`. -/
def simulate (p : Params) (s : Spec) (dir : System.FilePath) (fresh : Bool := false)
    (trace : Bool := false) (dumpAt : Option Nat := none)
    (recordTrace : Bool := false) (functional : Bool := false) (races : Bool := false)
    (prepare : Memory → Memory := id) (exportDir : Option System.FilePath := none)
    (workers : Nat := 0) (untilCycle : Option Nat := none) (spans : Bool := false) :
    IO (Except String Run) := do
  let work ← IO.FS.createTempDir
  let mut listings : Array (String × Array SASS.Kernel) := #[]
  for l in s.launches do
    if listings.any (·.1 == l.code) then continue
    let (sass, elf) ← listing dir l.code work p.target.value
    -- Only the kernels the spec launches from this code are read.
    let names := (s.launches.filter (·.code == l.code)).toList.map (·.function)
    match SASS.Read.kernels sass elf (some names) with
    | .ok ks => listings := listings.push (l.code, ks)
    | .error e => return .error s!"SASS reader: {e}"
  let find (code f : String) : Except String SASS.Kernel := do
    let some ks := listings.toList.lookup code | throw s!"no listing {code}"
    (ks.find? (·.name == f)).elim (throw s!"launch: {f} not in {code}") pure
  let files ← readFiles s dir
  let launched ← IO.mkRef 0
  let ran ← s.runM files find (Machine.fresh p) (prepare := prepare) fun m l => do
    let n ← launched.modifyGet fun n => (n, n + 1)
    if let some d := exportDir then exportLaunch p (d / toString n) l
    let r ← runOnIO p (if fresh then Machine.fresh p else m)
      { l with trace, dumpAt, recordTrace, functional, races, workers, untilCycle, spans }
    if let (some d, .ok (_, res)) := (exportDir, r) then
      IO.FS.writeBinFile (d / toString n / "final.bin") (storeBytes res.global)
      IO.FS.writeFile (d / toString n / "final.txt") s!"finalBase {res.global.base.toNat}\nfinalSize {res.global.size}\n"
    return r.mapError (s!"simulator: {·}") |>.map
      fun (m', r) => (m', r.global, (l.kernel.name, r.counters,
        (r.printf.map fun pr => SASS.printfText l.kernel pr.format pr.args, r.hazards, (l.kernel, r.traces),
         r.races)))
  return ran.map
    fun (m, g, cs) => { spec := s, files, memory := m, final := g,
                        launches := cs.map fun (n, c, _) => (n, c), printf := cs.map (·.2.2.1),
                        hazards := cs.map (·.2.2.2.1), traces := cs.map (·.2.2.2.2.1),
                        races := cs.map (·.2.2.2.2.2) }

/-- The bytes of a spec's output buffers no launch wrote. The output buffers are those the spec
names (`expect`, `outputs`) that start as zeros with nothing written into it, or, when it names
none, every such buffer (a captured pipeline's workspaces and zeroed inputs start as zeros too);
the spec is run again, functionally, with each of their bytes `0xA5` instead, and a byte holding
its first value in both runs was never written (or was only rewritten with what it held). Per
buffer with any, its unwritten ranges `[start, end)`. -/
def unwritten (p : Params) (dir : System.FilePath) (r : Run) :
    IO (Except String (Array (String × Array (Nat × Nat)))) := do
  let named := r.spec.expect.map (·.buffer) ++ r.spec.outputs.map (·.1)
  let outs := r.spec.buffers.filter fun b => (match b.init with | .zero => true | _ => false) &&
    b.writes.isEmpty && (named.isEmpty || named.contains b.name)
  if outs.isEmpty then return .ok #[]
  let fill (m : Memory) : Memory := Id.run do
    let mut g := m.global
    for b in outs do
      if let some a := m.addrs.toList.lookup b.name then
        let off := (a - g.base).toNat
        for i in [0:b.bytes] do g := g.writeByte (off + i) 0xA5
    return { m with global := g }
  match ← simulate p r.spec dir (functional := true) (prepare := fill) with
  | .error e => return .error s!"the sentinel run: {e}"
  | .ok r2 =>
    let mut found := #[]
    for b in outs do
      let (.ok x, .ok y) := (r.memory.bytes r.final r.spec b.name, r2.memory.bytes r2.final r.spec b.name)
        | return .error s!"no buffer {b.name}"
      let mut ranges : Array (Nat × Nat) := #[]
      for i in [0:b.bytes] do
        if x[i]! == 0 && y[i]! == 0xA5 then
          ranges := match ranges.back? with
            | some (s, e) => if e == i then ranges.pop.push (s, i + 1) else ranges.push (i, i + 1)
            | none => #[(i, i + 1)]
      unless ranges.isEmpty do found := found.push (b.name, ranges)
    return .ok found

end NV.Launch
