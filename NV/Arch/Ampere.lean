import NV.Arch.Hopper

/-!
# Ampere (GA100, `sm_80`): the facts A100 parts share

The SM of compute capability 8.0: NVIDIA's limits for it, its arithmetic throughput, its tensor
cores, and published A100 costs. A value taken from another part names that part (`onPart`,
`onH200`).

Opcode classes: Hopper's, with compute capability 8.0's documented throughput where it differs
from 9.0, each latency capped by `encoded`.
-/

namespace NV.Arch.Ampere

open NV NV.Core

/-! ## Structure and limits -/

def target : Fact String :=
  doc "sm_80" "ptx-isa-9.4" "§5 (Target ISA notes): sm_80, compute capability 8.0"

def features : Fact Features :=
  doc { cpAsync := true, tma := false, wgmma := false, clusters := false } "cuda-guide"
    "§5.1 Table 32, compute capability 8.0: asynchronous copies (cp.async) and split arrive/wait
    barriers; no tensor memory accelerator, thread block clusters or warpgroup MMA, which start at
    9.0 (ampere-tuning §1.4.1.2–3)"

def smsps : Fact Nat :=
  doc 4 "a100-whitepaper" "Fig. 7: the A100 SM's four processing blocks (sub-partitions)"

def maxThreadsPerSM : Fact Nat := doc 2048 "cuda-guide" "§5.1.3 Table 33, compute capability 8.0"
def maxCtasPerSM : Fact Nat := doc 32 "cuda-guide" "§5.1.3 Table 33, compute capability 8.0"
def maxWarpsPerSM : Fact Nat := doc 64 "cuda-guide" "§5.1.3 Table 33, compute capability 8.0"
def regsPerSM : Fact Nat := doc 65536 "cuda-guide" "§5.1.3 Table 34: 64 K 32-bit registers per SM"

def regAllocUnit : Fact Nat :=
  doc 256 "cuda-occupancy" "register allocation unit size, compute capability 8.0, warp granularity"

def smemPerSM : Fact Nat :=
  measOn 167936 ""
    ""

def smemPerCtaMax : Fact Nat :=
  measOn 166912 ""
    ""

def smemReservedPerCta : Fact Nat :=
  doc 1024 "ampere-tuning" "§1.4.2.3: 1 KB of shared memory is reserved for system use per block"

def maxClusterCtas : Fact Nat :=
  doc 1 "cuda-guide" "§5.1 Table 32: thread block clusters need compute capability 9.0"
    ""

/-! ## Issue, register file and front end -/

def issuePolicy : Fact IssuePolicy :=
  inferred .greedyThenLeastRecent "meas-a100" ""
    ""

def readRelease : Fact Nat :=
  pub 11 "huerta2025" "Table 2: WAR column, 32-bit global load with a regular address" "RTX A6000"

def barrierArrivalCycles : Fact Nat := onH200 Hopper.barrierArrivalCycles
def barrierCycles : Fact Nat :=
  inferred 21 "meas-a100" ""
    "" (some "A100 SXM4 40GB")

/-- The register file's organization (Huerta et al., GA102; on GA100, inferred). -/
def regFile : RegFileParams where
  banks := { Hopper.regFile.banks with kind := .published }
  readPortsPerBank := { Hopper.regFile.readPortsPerBank with kind := .published }
  operandReadCycles := { Hopper.regFile.operandReadCycles with kind := .published }
  reuseEntriesPerBank := { Hopper.regFile.reuseEntriesPerBank with kind := .published }
  reuseSlotsPerEntry := { Hopper.regFile.reuseSlotsPerEntry with kind := .published }

def frontEnd : FrontEndParams :=
  { Hopper.frontEnd with
    codeCaches := onH200 Hopper.frontEnd.codeCaches
    codePrefetchLines := onH200 Hopper.frontEnd.codePrefetchLines
    takenBranchCycles := onH200 Hopper.frontEnd.takenBranchCycles
    constCacheBytes := doc 8192 "cuda-guide"
      "§5.1.3 Table 34: cache working set per SM for constant memory, 8 KB (compute capability 8.0)" }

/-! ## Opcode classes -/

/-- Results per clock per SM where compute capability 8.0 differs from 9.0 (`cuda-best-practices`
§12.1.1 Table 5), indexed by SASS opcode. -/
private def cc80 : List (String × Nat × String) := [
  ("FFMA", 64, "32-bit floating-point add, multiply, multiply-add"),
  ("FADD", 64, "32-bit floating-point add, multiply, multiply-add"),
  ("FMUL", 64, "32-bit floating-point add, multiply, multiply-add"),
  ("DFMA", 32, "64-bit floating-point add, multiply, multiply-add"),
  ("DADD", 32, "64-bit floating-point add, multiply, multiply-add"),
  ("DMUL", 32, "64-bit floating-point add, multiply, multiply-add"),
  ("HFMA2", 128, "16-bit floating-point add, multiply, multiply-add (2-way SIMD)"),
  ("HADD2", 128, "16-bit floating-point add, multiply, multiply-add (2-way SIMD)"),
  ("HMUL2", 128, "16-bit floating-point add, multiply, multiply-add (2-way SIMD)"),
  ("HMNMX2", 64, "16-bit floating-point minimum, maximum"),
  ("FMNMX", 64, "32-bit floating-point minimum, maximum"),
  ("IMNMX", 64, "32-bit integer minimum, maximum"),
  ("VIMNMX", 64, "32-bit integer minimum, maximum"),
  ("I2F", 16, "type conversions between 32-bit integer and 32-bit floating-point"),
  ("F2I", 16, "type conversions between 32-bit integer and 32-bit floating-point"),
  ("I2FP", 16, "type conversions between 32-bit integer and 32-bit floating-point"),
  ("SHFL", 32, "warp shuffle")]

/-- Each fixed-latency opcode's latency cap on `sm_80`. -/
private def encoded : List (String × Nat) := [
  ("CS2R", 5), ("F2FP", 4), ("FADD", 4), ("FFMA", 4), ("FMNMX", 26), ("FMUL", 4), ("FSEL", 4),
  ("FSETP", 4), ("HADD2", 5), ("HFMA2", 5), ("HSETP2", 89), ("IABS", 4), ("IADD3", 4),
  ("IDP", 107), ("IMAD", 4), ("IMNMX", 4), ("ISETP", 4), ("LEA", 4), ("LOP3", 4), ("MOV", 4),
  ("P2R", 4), ("PLOP3", 4), ("PRMT", 4), ("SEL", 4), ("SGXT", 4), ("SHF", 4), ("UIADD3", 4),
  ("UIMAD", 4), ("UISETP", 4), ("ULDC", 2), ("ULEA", 4), ("ULOP3", 4), ("UMOV", 4), ("USEL", 4),
  ("USHF", 4), ("VABSDIFF", 132), ("VABSDIFF4", 7), ("VOTEU", 2)]

/-- Opcodes compute capability 8.0 does not have. -/
private def hopperOnly (opcode : String) : Bool :=
  ["HGMMA", "QGMMA", "IGMMA", "BGMMA", "WARPGROUP", "UTMALDG", "UTMASTG", "UBLKCP", "UBLKRED", "UTMAPF",
   "UBLKPF", "UTMAREDG", "LDGMC", "UTMACMDFLUSH", "UCGABAR_ARV", "UCGABAR_WAIT", "SYNCS", "ELECT", "VIADDMNMX", "VIMNMX3",
   "USETMAXREG", "ACQBULK", "FENCE"].contains opcode

/-- A Hopper class on Ampere: the documented interval where compute capability 8.0 differs, and
the Hopper latency, capped by `encoded`. -/
private def retime (r : OpRule) : OpRule :=
  let h := r.timing
  let lat := match encoded.lookup r.opcode with
    | some d => min h.value.latency d
    | none => h.value.latency
  let base := { h with
    value := { h.value with latency := lat }
    kind := if h.kind == .documented then .inferred else h.kind
    measuredOn := if h.source.ref == "meas-h200" then some "H200" else h.measuredOn
    note := "" }
  match cc80.lookup r.opcode with
  | some (thr, row) =>
    { r with timing := { base with
        value := { base.value with interval := 128 / thr }
        source := { ref := "cuda-best-practices", loc := s!"§12.1.1 Table 5, compute capability 8.0: {row}, {thr} results per clock per SM" } } }
  | none => { r with timing := base }

/-- `sm_80` spells the global reduction RED where `sm_90` spells REDG: the same class. -/
private def renamed : Array OpRule :=
  (Hopper.ops.filter (·.opcode == "REDG")).map fun r => { r with opcode := "RED" }

def ops : Array OpRule :=
  ((Hopper.ops.filter fun r => !hopperOnly r.opcode) ++ renamed).map retime

/-! ## Memory -/

def l1SmemBytes : Fact Nat :=
  doc 196608 "ampere-tuning" "§1.4.2.3: combined L1 and shared memory, 192 KB per SM"

def smemCarveoutsKB : Fact (List Nat) :=
  doc [0, 8, 16, 32, 64, 100, 132, 164] "ampere-tuning"
    "§1.4.2.3: shared memory capacity 0, 8, 16, 32, 64, 100, 132 or 164 KB"
    ""

def smemLatency : Fact Nat :=
  pub 29 "luo2024" "Table IV: shared memory 29.0 cycles per chase step" "A100 PCIe"
    ""

def l1HitLatency : Fact Nat :=
  pub 33 "abdelkhalik2022" "Table IV: L1 hit 33 cycles" "A100"
    ""

def l2NearHitLatency : Fact Nat :=
  pub 262 "luo2024" "Table IV: L2 261.5 cycles" "A100 PCIe"
    ""

def l2FarHitLatency : Fact Nat :=
  inferred 420 "luo2025" "Table 4: L2 near hit 258.0, far hit 414.1 (H800 PCIe)"
    ""
    (some "H800 PCIe")

def l2Partitions : Fact Nat :=
  doc 2 "a100-whitepaper" "p. 35: the L2 cache is divided into two partitions"

def l2SliceBytesPerCycle : Fact Nat :=
  inferred 36 "meas-a100" ""
    "" (some "A100 SXM4 40GB")

def dramLatencyPcie : Fact Nat :=
  pub 466 "luo2024" "Table IV: global memory 466.3 cycles" "A100 PCIe"
    ""

def tensorFma16 : Fact Nat :=
  doc 1024 "a100-whitepaper" "Table 1: peak FP16/BF16 tensor 312 TFLOPS dense at 1410 MHz"
    ""

def tensorFmaTF32 : Fact Nat :=
  doc 512 "a100-whitepaper" "Table 1: peak TF32 tensor 156 TFLOPS dense"

def tensorFmaF64 : Fact Nat :=
  doc 64 "a100-whitepaper" "Table 1: peak FP64 tensor 19.5 TFLOPS"
    ""

def tensorFma8 : Fact Nat :=
  doc 0 "cuda-guide" "§5.1 Table 36: FP8 tensor cores need compute capability 8.9 or 9.0"
    ""

def mmaSyncLatency : Fact Nat :=
  pub 26 "luo2024" "Table VII: mma m16n8k16 f32.f16.f16, 26.0 cycles" "A100 PCIe"

/-! ## Synchronization: the H200's, labelled -/

def sync : SyncParams where
  mbarrierLatency := onH200 Hopper.mbarrierLatency
  clusterMbarrierLatency := doc 0 "cuda-guide" "§5.1 Table 32: no clusters before 9.0"
  clusterBarrierCycles := doc 0 "cuda-guide" "§5.1 Table 32: no clusters before 9.0"
  namedBarrierLatency := onH200 Hopper.namedBarrierLatency
  barSync := onH200 Hopper.barSync
  fenceProxyAsync := onH200 Hopper.fenceProxyAsync
  fenceAcqRelCta := onH200 Hopper.fenceAcqRelCta
  fenceAcqRelGpu := onH200 Hopper.fenceAcqRelGpu
  fenceScGpu := onH200 Hopper.fenceScGpu
  fenceAcqRelSys := onH200 Hopper.fenceAcqRelSys
  fenceCycles := inferred [("MEMBAR.ALL.CTA", 14), ("FENCE.VIEW.ASYNC.S", 0), ("MEMBAR.ALL.GPU", 183),
    ("MEMBAR.SC.GPU", 176), ("MEMBAR.ALL.SYS", 3299)] "meas-a100"
    ""
    "" (some "A100 SXM4 40GB")
  redReleaseGpu := onH200 Hopper.redReleaseGpu
  atomAcqRelGpu := onH200 Hopper.atomAcqRelGpu
  atomRelaxedGpu := onH200 Hopper.atomRelaxedGpu
  ldAcquireGpu := onH200 Hopper.ldAcquireGpu
  flagHopPs := onH200 Hopper.flagHopPs
  printfPs := measOn 36860000 ""
    ""
  printfLanePs := measOn 4260000 ""
    ""
  tryWaitLimit := doc 0 "ptx-isa-9.5" "§9.7.15.16.19 Target ISA notes: mbarrier.try_wait requires
    sm_90 or higher (test_wait, which returns at once, sm_80)"
  tryWaitWake := doc 0 "ptx-isa-9.5" "§9.7.15.16.19 Target ISA notes: mbarrier.try_wait requires
    sm_90 or higher"

end NV.Arch.Ampere
