import NV.SASS.Read
import NV.Core.Engine
import NV.Arch.Parts
import NV.Bounds
import NV.Launch.Load

/-!
`lake exe gap [--gpu KEY] [n]`: gap reports, for one part (`NV.Arch.parts`) or,
without `--gpu`, for every part in turn. Each report reads only its part's parameters.

* **axpy**, run: `tests/kernels/axpy_s32.ptx` compiled by the pinned `ptxas` for the part's target,
  run on the part over `n` elements (default 65536) from a fresh machine. Its bound: the spec's
  compulsory DRAM and L2 traffic, the issue slots its dynamic instruction stream needs, and the
  dependence chain of one warp's path. The report gives the simulated time over each term.
* **GEMM**, run on parts with TMA and warpgroup MMA: the tile kernel
  (`tests/kernels/gemm_tma_wgmma.cu`) at 1024 × 1024 × 1024 BF16 from its launch spec
  (`examples/specs/gemm_tma_wgmma.json`), against the GEMM spec's bound: compulsory traffic,
  multiply-adds, and classical-matmul I/O.
* **GEMM** and **attention** at production sizes: the bounds of a DeepGEMM-style GEMM (FP8 where
  the part has FP8 tensor cores, else BF16) and a FlashAttention-style forward pass, beside no
  simulated time.
* **attention**, run on request: `lake exe gap attention SUITE SPEC HEADS SEQ DIM CAUSAL`
  simulates a forward attention launch spec (`SUITE/build/specs/SPEC.json`), `HEADS` = batch × heads, and reports it against the attention bound.
* **decode-layer floors** on the H200: the stream, issue and chain floors of a decode kernel.

Then the run checks the rate invariants on itself: each resource's work, from the run's
counters, needs no more cycles than the run took. A failure is a defect in the engine (a port
granting more than its rate) and exits 1.
-/

open NV NV.SASS NV.Core NV.Bounds

def tool (cmd : String) (args : Array String) : IO String := do
  let out ← IO.Process.output { cmd, args }
  if out.exitCode != 0 then throw (IO.userError s!"{cmd} {args}: {out.stderr}")
  return out.stdout

def le (v : Nat) (bytes : Nat) : List UInt8 := (List.range bytes).map fun i => (v >>> (8 * i)).toUInt8

/-- The run's work per resource, from its counters, against the cycles the run took. -/
def checkRun (p : Params) (c : Counters) : IO Bool := do
  let t := c.cycles
  let sector := p.mem.sectorBytes.value
  let mm := c.memory
  let rows : List (String × Nat × Nat) := [
    ("DRAM", mm.dramReadBytes + mm.dramWriteBytes, (Part.dram p).capacity.minCycles (mm.dramReadBytes + mm.dramWriteBytes)),
    ("L2 slices", (mm.l2ReadSectors + mm.l2WriteSectors + mm.l2AtomSectors) * sector,
      (Part.l2 p).capacity.minCycles ((mm.l2ReadSectors + mm.l2WriteSectors + mm.l2AtomSectors) * sector)),
    ("issue", c.instExecuted, pipeCycles (Part.issue p).instances 1 c.instExecuted),
    ("busiest sub-partition's issue", c.issueMax, c.issueMax),
    ("busiest tensor-core port", c.tensorBusyMax, c.tensorBusyMax),
    ("busiest tensor operand port", c.memory.tensorPortMax, c.memory.tensorPortMax),
    ("busiest sub-partition pipe", c.pipeBusyMax.foldl max 0, c.pipeBusyMax.foldl max 0)]
  let mut ok := true
  IO.println "\nthe run against the rate invariants (work from its counters, cycles it needs, cycles taken):"
  for (name, work, need) in rows do
    let good := need ≤ t
    ok := ok && good
    IO.println s!"  {name}: {work} → {need} ≤ {t} {if good then "ok" else "VIOLATED"}"
  return ok

def axpyReport (p : Params) (n : Nat) : IO Bool := do
  let dir ← IO.FS.createTempDir
  let cubin := (dir / "axpy.cubin").toString
  -- The kernel's `.target` names the part's.
  let ptx := (← IO.FS.readFile "tests/kernels/axpy_s32.ptx").splitOn "\n" |>.map fun l =>
    if l.startsWith ".target" then s!".target {p.target.value}" else l
  IO.FS.writeFile (dir / "axpy.ptx") ("\n".intercalate ptx)
  discard <| tool "ptxas" #[s!"-arch={p.target.value}", "-m64", (dir / "axpy.ptx").toString, "-o", cubin]
  let sass ← tool "cuobjdump" #["-sass", cubin]
  let elf ← tool "cuobjdump" #["-elf", cubin]
  let ks ← IO.ofExcept (Read.kernels sass elf)
  let some k := ks.find? (·.name == "axpy_s32") | throw (IO.userError "axpy_s32 not found")
  let base : UInt64 := 0x7f0000000000
  let x := base
  let y := base + (4 * n).toUInt64
  let mut g := Store.empty base (8 * n)
  for i in [0:n] do
    g := g.writeWord (4 * i) ((7 * i + 1) % 2 ^ 32).toUInt32
    g := g.writeWord (4 * n + 4 * i) ((3 * i + 2 ^ 32 - 5) % 2 ^ 32).toUInt32
  let args : List (List UInt8) := [le x.toNat 8, le y.toNat 8, le 3 4, le n 4]
  let mut params := ByteArray.mk (Array.replicate k.paramBytes 0)
  for ((off, sz), bytes) in k.params.toList.zip args do
    for j in [0:sz] do params := params.set! (off + j) bytes[j]!
  let launch : Launch := { kernel := k, grid := ((n + 255) / 256, 1, 1), block := (256, 1, 1), params,
                           global := g }
  let res ← IO.ofExcept (run p launch)
  let c := res.counters
  -- The kernel's own terms: its dynamic instruction stream and one warp's chain, with loads at
  -- the idle DRAM latency (a fresh machine: no load hits).
  let stream := pathStream p k p.mem.dramLatency.value
  -- Warps of the launch over the sub-partitions: some sub-partition runs at least this many.
  let warps := (n + 255) / 256 * 8
  let perSmsp := cdiv warps (p.sms.value * p.smsps.value)
  let busiest := stream.copies perSmsp
  let spec := Specs.axpy p n 4
  let bound : Bound := { spec with
    terms := spec.terms ++ [
      (Part.issue p).term c.instExecuted "this SASS kernel's dynamic instruction stream, any schedule"
        "Slots.time_ge: one instruction per sub-partition per cycle; the count is the run's \
        (data-independent for this kernel)",
      { resource := "one warp's dependence chain", unit := "cycles"
        demand := .stream stream []
        scope := "this SASS kernel, any schedule; its control bits correct"
        reason := s!"Stream.bound_le over the path to EXIT ({stream.size} instructions): in-order \
          issue and register flow at the opcode table's latencies, global loads at the idle DRAM \
          latency (a fresh machine: data no earlier access touched misses every cache)"
        facts := pathFacts p k ++ [.of "mem.dramLatency" p.mem.dramLatency] },
      { resource := s!"busiest sub-partition ({perSmsp} warps)", unit := "cycles"
        demand := .stream busiest subPartitionResources
        scope := "this SASS kernel, any schedule; its control bits correct"
        reason := s!"Stream.bound_le over {perSmsp} independent copies of the path: the {warps} \
          warps cover the {p.sms.value * p.smsps.value} sub-partitions, so one runs at least \
          {perSmsp}; each pipe serves one instruction per interval and the issue slot one per \
          cycle, every interval ending within the run"
        facts := pathFacts p k ++ [.of "smsps" p.smsps, .of "sms" p.sms] }] }
  IO.println (Gap.render { bound, simulated := some c.cycles })
  checkRun p c

/-- The GEMM tile kernel's run against the GEMM bound. -/
def gemmReport (p : Params) (spec : System.FilePath) (m n k : Nat) : IO Bool := do
  let s ← IO.ofExcept (NV.Launch.Spec.parse (← IO.FS.readFile spec))
  match ← NV.Launch.simulate p s (spec.parent.getD ".") with
  | .error e => IO.eprintln s!"gemm: {e}"; return false
  | .ok r =>
    let some (_, c) := r.launches[0]? | IO.eprintln "gemm: no launch"; return false
    let g : Specs.Gemm :=
      { name := "TMA + wgmma tile kernel", m, n, k, inBytes := 2, outBytes := 4,
        tensor := p.tensor.fmaPerCycle16 }
    IO.println (Gap.render { bound := Specs.gemm p g, simulated := some c.cycles })
    checkRun p c

/-- A forward attention kernel's run from its launch spec against the attention bound; its rate
invariants checked over every launch. Its kernels are resolved from `dir`. -/
def attentionReport (p : Params) (spec dir : System.FilePath) (a : Specs.Attn) : IO Bool := do
  let s ← IO.ofExcept (NV.Launch.Spec.parse (← IO.FS.readFile spec))
  match ← NV.Launch.simulate p s dir with
  | .error e => IO.eprintln s!"attention: {e}"; return false
  | .ok r =>
    let cycles := r.launches.foldl (fun t (_, c) => t + c.cycles) 0
    IO.println (Gap.render { bound := Specs.attention p a, simulated := some cycles })
    r.launches.foldlM (fun ok (_, c) => do return ok && (← checkRun p c)) true

def otherReports (key : String) (p : Params) : IO Unit := do
  let fp8 := p.tensor.fmaPerCycle8.value > 0
  let gemm : Specs.Gemm :=
    { name := if fp8 then "DeepGEMM-style FP8" else "DeepGEMM-style shape, BF16 (no FP8 tensor cores)"
      m := 4096, n := 7168, k := 2048, inBytes := if fp8 then 1 else 2, outBytes := 2,
      tensor := if fp8 then p.tensor.fmaPerCycle8 else p.tensor.fmaPerCycle16 }
  IO.println (Gap.render { bound := Specs.gemm p gemm, simulated := none })
  let attn : Specs.Attn :=
    { name := "FlashAttention-style forward", heads := 64, seq := 4096, dim := 128, bytes := 2,
      causal := false, tensor := p.tensor.fmaPerCycle16 }
  IO.println (Gap.render { bound := Specs.attention p attn, simulated := none })
  -- A qkv projection at one token: 2304 × 8192 weights at 1.5 bytes, its issue count and its
  -- stream chain, on the H200.
  if key != "h200" then return
  let qkv : Floors.Kernel :=
    { name := "qkv projection, 1 token", bytes := 2304 * 8192 * 3 / 2, fma := 2304 * 8192,
      issue := some (16 * 230), chain := Floors.streamChain p }
  let fixed := Floors.psCycles p 1500000
  IO.println (Gap.render { bound := Floors.kernelFloor p fixed qkv, simulated := none })

/-- Every report for one part. -/
def partReports (key : String) (p : Params) (n : Nat) : IO Bool := do
  IO.println s!"\n# {p.name} ({key}, {p.target.value})\n"
  -- A kernel the simulator cannot run on this part yet is reported, not a failed bound.
  let ok ← try axpyReport p n catch e => do
    IO.println s!"axpy: not run on {key}: {e}"
    pure true
  let okG ← if p.features.value.tma && p.features.value.wgmma then
      gemmReport p "examples/specs/gemm_tma_wgmma.json" 1024 1024 1024
    else do
      IO.println "\nGEMM tile kernel: not run (it needs TMA and warpgroup MMA, which this part lacks)."
      pure true
  otherReports key p
  return ok && okG

def main (argv : List String) : IO UInt32 := do
  let (only, argv) := match argv with
    | "--gpu" :: key :: rest => (some key, rest)
    | _ => (none, argv)
  let chosen : List (String × Params) := match only with
    | some key => NV.Arch.parts.filter fun (k, _) => k == key
    | none => NV.Arch.parts
  if chosen.isEmpty then
    IO.eprintln s!"gap: no part {only.getD ""}; parts: {NV.Arch.parts.map Prod.fst}"; return 2
  -- `gap [--gpu KEY] attention SUITE SPEC BATCH×HEADS SEQ DIM CAUSAL`: one attention run of a
  -- launch spec (`SUITE/build/specs/SPEC.json`), reported alone, on the first chosen part.
  if let "attention" :: suite :: spec :: heads :: seq :: dim :: causal :: _ := argv then
    let suite : System.FilePath := suite
    let some heads := heads.toNat? | return 2
    let some seq := seq.toNat? | return 2
    let some dim := dim.toNat? | return 2
    let some (_, p) := chosen.head? | return 2
    let a : Specs.Attn :=
      { name := s!"forward from {spec}", heads, seq, dim, bytes := 2, causal := causal == "1",
        tensor := p.tensor.fmaPerCycle16 }
    let ok ← attentionReport p (suite / "build" / "specs" / s!"{spec}.json") (suite / "build") a
    return if ok then 0 else 1
  let n := (argv.head? >>= String.toNat?).getD 65536
  let mut ok := true
  for (key, p) in chosen do
    ok := (← partReports key p n) && ok
  return if ok then 0 else 1
