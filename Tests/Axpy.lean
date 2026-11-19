import NV.SASS.Read
import NV.Core.Engine
import NV.Arch.Parts

/-!
`lake exe axpy [n] [--gpu KEY|all]`: the end-to-end path on a real compiler's output.
`tests/kernels/axpy_s32.ptx` is compiled by the pinned `ptxas` for the part's target, read back
from `cuobjdump`, and run on the part (default the H200; `all`: every part) over `n` elements
(default 1000) in CTAs of 256 threads; every output is checked against `y[i] = a * x[i] + y[i]`,
and the counters and the simulator's speed are printed.
-/

open NV NV.SASS NV.Core

def tool (cmd : String) (args : Array String) : IO String := do
  let out ← IO.Process.output { cmd, args }
  if out.exitCode != 0 then throw (IO.userError s!"{cmd} {args}: {out.stderr}")
  return out.stdout

def le (v : Nat) (bytes : Nat) : List UInt8 := (List.range bytes).map fun i => (v >>> (8 * i)).toUInt8

def axpyOn (p : NV.Core.Params) (n : Nat) : IO UInt32 := do
  let dir ← IO.FS.createTempDir
  let cubin := (dir / "axpy.cubin").toString
  -- The PTX names sm_90a; it uses nothing beyond sm_80, so it is retargeted to the part's target.
  let ptx := (dir / "axpy.ptx").toString
  let text ← IO.FS.readFile "tests/kernels/axpy_s32.ptx"
  IO.FS.writeFile ptx (text.replace ".target sm_90a" s!".target {p.target.value}")
  discard <| tool "ptxas" #[s!"-arch={p.target.value}", "-m64", ptx, "-o", cubin]
  let sass ← tool "cuobjdump" #["-sass", cubin]
  let elf ← tool "cuobjdump" #["-elf", cubin]
  let ks ← IO.ofExcept (Read.kernels sass elf)
  let some k := ks.find? (·.name == "axpy_s32") | throw (IO.userError "axpy_s32 not found")
  IO.println s!"axpy on the {p.name} ({p.target.value}): {k.code.size} instructions, {k.regs} registers, params {k.params}"
  let a : Nat := 3
  let base : UInt64 := 0x7f0000000000
  let x := base
  let y := base + (4 * n).toUInt64
  let xs := (Array.range n).map fun i => (7 * i + 1) % 2 ^ 32
  let ys := (Array.range n).map fun i => (3 * i + 2 ^ 32 - 5) % 2 ^ 32
  let mut g := Store.empty base (8 * n)
  for i in [0:n] do
    g := g.writeWord (4 * i) xs[i]!.toUInt32
    g := g.writeWord (4 * n + 4 * i) ys[i]!.toUInt32
  let args : List (List UInt8) := [le x.toNat 8, le y.toNat 8, le a 4, le n 4]
  let mut params := ByteArray.mk (Array.replicate k.paramBytes 0)
  for ((off, sz), bytes) in k.params.toList.zip args do
    for j in [0:sz] do params := params.set! (off + j) bytes[j]!
  let launch : Launch := { kernel := k, grid := ((n + 255) / 256, 1, 1), block := (256, 1, 1), params,
                           global := g }
  -- The reference: the same run with all SMs on one worker thread, which must give the same
  -- outputs and cycles (results do not depend on how SMs are dealt to workers).
  let t1 ← IO.monoMsNow
  let reference ← runIO p { launch with workers := 1 }
  let msSerial := (← IO.monoMsNow) - t1
  let t0 ← IO.monoMsNow
  let result ← runIO p launch
  let ms := (← IO.monoMsNow) - t0
  match result, reference with
  | .error e, _ | _, .error e => IO.eprintln s!"axpy: {e}"; return 1
  | .ok res, .ok ref =>
    IO.println s!"axpy: parallel {res.counters.cycles} cycles in {ms} ms, one worker {ref.counters.cycles} cycles in {msSerial} ms"
    if res.global.read y (4 * n) != ref.global.read y (4 * n) then
      IO.eprintln "axpy: the outputs differ with the SMs grouped differently"; return 1
    if res.counters.cycles != ref.counters.cycles then
      IO.eprintln s!"axpy: {res.counters.cycles} cycles, but {ref.counters.cycles} with the SMs on one worker"; return 1
    let out := res.global.read y (4 * n)
    let mut bad := 0
    for i in [0:n] do
      let want := ((a * xs[i]! + ys[i]!) % 2 ^ 32).toUInt32
      if out[i]! != want then
        bad := bad + 1
        if bad ≤ 5 then IO.eprintln s!"y[{i}] = {out[i]!}, expected {want}"
    IO.println s!"axpy: {repr res.counters}"
    let stalls := Stall.all.map fun r => s!"{repr r} {res.counters.stalls[r.index]!}"
    IO.println s!"axpy: stalls (warp-cycles) {stalls}"
    IO.println s!"axpy: simulated in {ms} ms, {res.counters.instExecuted * 1000 / max ms 1} warp instructions per second"
    if bad > 0 then IO.eprintln s!"axpy: {bad} wrong outputs"; return 1
    if res.counters.hazards > 0 then
      IO.eprintln s!"axpy: {res.counters.hazards} hazards in the compiled code: {res.hazards}"; return 1
    return 0

def main (argv : List String) : IO UInt32 := do
  let n := ((argv.find? (·.all Char.isDigit)) >>= String.toNat?).getD 1000
  let gpu := match argv.dropWhile (· != "--gpu") with
    | _ :: g :: _ => g
    | _ => "h200"
  let parts := if gpu == "all" then NV.Arch.parts else NV.Arch.parts.filter (·.1 == gpu)
  if parts.isEmpty then IO.eprintln s!"axpy: no part {gpu}"; return 2
  let mut code := 0
  for (_, p) in parts do
    if (← axpyOn p n) != 0 then code := 1
  return code
