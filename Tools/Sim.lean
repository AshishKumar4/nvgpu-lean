import NV.Launch.Load
import NV.Launch.Report
import NV.Arch.Parts

/-!
`lake exe sim SPEC.json [--gpu PART] [--clock MHZ] [--fresh] [--functional] [--races] [--unwritten]`: run a launch
spec (`NV/Launch/Spec.lean`; `--functional`: untimed, `Launch.functional`; `--races`: report data
races between the threads of a CTA, `NV.Core.Memory.RaceCheck`; `--unwritten`: report output bytes
no launch wrote, `NV.Launch.unwritten`, in the buffers the spec names as outputs or else every
buffer that starts as zeros; either failing the run when it finds any)
on the simulator and print, per launch, what Nsight Compute would: cycles before, during and after
the kernel, instructions, memory traffic and where warps stalled; and the cycles each resource's
work needs at its rate, the largest named as binding (`binding`). Kernels are the files the spec's
launches name, relative to the spec: a `.cu` (compiled with `nvcc` for `sm_90a`), a `.ptx`
(`ptxas`), a `.cubin`, or a `cuobjdump` listing pair `<name>.sass`/`<name>.elf`. Launches run in
order on one machine, its caches carried from one to the next (`--fresh`: a fresh machine each).
The spec's expected outputs are checked, and its `outputs` written beside it. `--until CYCLE` stops
each launch at that cycle and reports the cycles run (`Launch.untilCycle`): a sample of a launch too
long to simulate whole, whose outputs are not the launch's.
-/

open NV NV.Core NV.Launch NV.Launch.Report

def parts : List (String × Params) := NV.Arch.parts

/-- A byte offset given in decimal or as `0x…` hex (as the listing prints it). -/
def offset? (v : String) : Option Nat :=
  if v.startsWith "0x" then
    (v.drop 2).foldl (fun acc c => acc.bind fun n =>
      if c.isDigit then some (n * 16 + (c.toNat - 48))
      else if 'a' ≤ c.toLower && c.toLower ≤ 'f' then some (n * 16 + (c.toLower.toNat - 87))
      else none) (some 0)
  else v.toNat?

def main (args : List String) : IO UInt32 := do
  let some path := args.find? (!·.startsWith "--") | do
    IO.eprintln ("usage: lake exe sim SPEC.json [--gpu PART] [--fresh] [--functional] [--races] " ++
      "[--unwritten] [--trace] [--dump-at OFF] [--allow-hazards] [--workers N] [--export DIR] [--clock MHZ] [--sms N] [--until CYCLE] [--spans]"); return 2
  let gpu := match args.dropWhile (· != "--gpu") with
    | _ :: g :: _ => g
    | _ => "h200"
  let some p := parts.lookup gpu | do IO.eprintln s!"no parameter set {gpu}"; return 2
  -- `--clock MHZ`: the part at another SM clock.
  let p := match args.dropWhile (· != "--clock") with
    | _ :: m :: _ => match m.toNat? with
      | some v => { p with clockMHz := { p.clockMHz with value := v } }
      | none => p
    | _ => p
  -- `--sms N`: the part with its first N SMs (GPCs filled in order), for a wave of a launch too
  -- large to simulate whole: the CTAs resident on N SMs, with the whole part's memory system.
  let p := match args.dropWhile (· != "--sms") with
    | _ :: m :: _ => match m.toNat? with
      | some n =>
        let g := (p.gpcSMs.value.foldl (fun (acc : List Nat × Nat) k =>
          (acc.1 ++ [min k acc.2], acc.2 - min k acc.2)) ([], n)).1.filter (· > 0)
        { p with sms := { p.sms with value := min n p.sms.value }, gpcSMs := { p.gpcSMs with value := g } }
      | none => p
    | _ => p
  let path : System.FilePath := path
  let dir := path.parent.getD "."
  let spec ← IO.ofExcept (Spec.parse (← IO.FS.readFile path))
  match ← simulate p spec dir (fresh := args.contains "--fresh") (trace := args.contains "--trace")
      (functional := args.contains "--functional") (races := args.contains "--races")
      (exportDir := ((args.dropWhile (· != "--export")).drop 1).head?.map System.FilePath.mk)
      (workers := (((args.dropWhile (· != "--workers")).drop 1).head?.bind String.toNat?).getD 0)
      -- `--until CYCLE`: each launch stopped at that cycle (`Launch.untilCycle`).
      (untilCycle := ((args.dropWhile (· != "--until")).drop 1).head?.bind String.toNat?)
      -- `--spans`: each CTA's span to standard error as it finishes (`Launch.spans`).
      (spans := args.contains "--spans")
      (dumpAt := match args.dropWhile (· != "--dump-at") with
        | _ :: v :: _ => offset? v
        | _ => none) with
  | .error e => IO.eprintln e; return 1
  | .ok r =>
    let mut hazardous := false
    let mut racy := false
    for i in [0:r.launches.size] do
      let (name, c) := r.launches[i]!
      IO.print (report i name c)
      IO.print (binding p c)
      for line in (r.printf[i]?).getD #[] do IO.print line
      if c.hazards > 0 then
        IO.println s!"  hazards     {c.hazards} (control bits under-wait; the first: {(r.hazards[i]?).getD #[]})"
        hazardous := true
      for race in (r.races[i]?).getD #[] do
        IO.println s!"  race        {race}"
        racy := true
    for (buffer, file) in spec.outputs do
      IO.FS.writeBinFile (dir / file) (← IO.ofExcept (r.memory.bytes r.final spec buffer))
    -- Output bytes no launch wrote (`--unwritten`): a second, functional run with a sentinel.
    let mut unwrittenBytes := false
    if args.contains "--unwritten" then
      match ← unwritten p dir r with
      | .error e => IO.eprintln e; return 1
      | .ok found =>
        for (b, ranges) in found do
          let n := ranges.foldl (fun a (s, e) => a + e - s) 0
          IO.println s!"unwritten: {b}, {n} bytes, at {(ranges.extract 0 8).toList}{if ranges.size > 8 then " …" else ""}"
          unwrittenBytes := true
        if found.isEmpty then IO.println "unwritten: none"
    if racy then IO.eprintln "races: threads of a CTA touch a byte between the same barriers, one writing"
    match spec.mismatches r.files r.memory r.final with
    | .error e => IO.eprintln e; return 1
    | .ok #[] =>
      if racy || unwrittenBytes then return 1
      unless spec.expect.isEmpty do IO.println "outputs: as expected"
      -- Compiled code never under-waits: a hazard means a latency in the part's table exceeds the
      -- code's, or a semantics reads outside its footprint.
      if hazardous && !args.contains "--allow-hazards" then
        IO.eprintln "hazards: the run is not one the program semantics allows (--allow-hazards to accept)"
        return 1
      return 0
    | .ok bad =>
      for b in bad do IO.eprintln s!"outputs: {b}"
      return 1
