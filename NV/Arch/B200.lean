import NV.Arch.H200
import NV.Arch.BlackwellRTX

/-!
# B200 (`sm_100a`) and B300 (`sm_103a`)

Blackwell's data-center part: two dies, 148 SMs in eight GPCs, four L2 partitions, eight HBM3e
stacks (jarmusch2025 §IV). The SM is Hopper's as compute capability 10.0 keeps its limits
(cuda-guide Tables 33–34, blackwell-tuning §1.4.1.1) with tensor memory and the 5th-generation
tensor cores in place of warpgroup MMA (`NV.Async.Tcgen05`).

Device facts are NVIDIA's documents, Jarmusch and Chandrasekaran's B200 measurements and our own;
a value taken from the H200 is labelled.
-/

namespace NV.Arch

open NV NV.Core Hopper

namespace Blackwell

def target : Fact String :=
  doc "sm_100a" "ptx-isa-9.5" "§9.7.18 Target ISA notes: tcgen05 requires sm_100a or sm_103a"

def features : Fact Features :=
  doc { cpAsync := true, tma := true, wgmma := false, clusters := true, tcgen05 := true } "cuda-guide"
    "§5.1, compute capability 10.0: asynchronous copies, the tensor memory accelerator, thread
    block clusters; tensor memory and tcgen05 (ptx-isa-9.5 §9.7.18); wgmma is sm_90a's only
    (ptx-isa-9.5 §9.7.17 Target ISA notes)"

/-- Hopper's rule for `opcode` (no modifiers), as the B200's for `as`, labelled. -/
private def asHopper (opcode as : String) : OpRule :=
  let r := (ops.find? fun r => r.opcode == opcode && r.mods.isEmpty).get!
  { r with opcode := as,
           timing := onH200 { r.timing with kind := .inferred,
                                            note := "" } }

/-- The issue of `tcgen05`'s instructions (`NV.SASS.Async`): the MMAs are their async unit's,
the moves and the allocator's atomic go through the MIO, as shared memory's; and sm_100a's other
new forms. -/
def tcgen05Ops : List OpRule :=
  ["UTCHMMA", "UTCQMMA", "UTCIMMA", "UTCOMMA"].map
      (asHopper "HGMMA" ·) ++
    [asHopper "SYNCS" "UTCBAR",
     asHopper "LDS" "LDTM",
     asHopper "STS" "STTM",
     asHopper "ATOMS" "UTCATOMSWS",
     asHopper "WARPGROUP" "UVIRTCOUNT",
     asHopper "UBLKCP" "UTCCP",
     asHopper "UBLKCP" "UTCSHIFT",
     asHopper "NOP" "PMTRIG",
     asHopper "UBLKCP" "UGETNEXTWORKID",
     asHopper "REDUX" "CREDUX",
     asHopper "FMNMX" "FMNMX3"]

/-- Each fixed-latency opcode's latency cap on `sm_100a`. -/
private def encoded : List (String × Nat) := [
  ("BREV", 32), ("CREDUX", 13), ("CS2R", 5), ("DADD", 7), ("DFMA", 8), ("DMUL", 6), ("DSETP", 6),
  ("ELECT", 2), ("F2FP", 9), ("F2I", 17), ("F2IP", 16), ("FADD", 4), ("FADD2", 4), ("FFMA", 4),
  ("FFMA2", 4), ("FMNMX", 4), ("FMNMX3", 4), ("FMUL", 4), ("FMUL2", 4), ("FRND", 75), ("FSEL", 4),
  ("FSETP", 4), ("HADD2", 7), ("HFMA2", 11), ("HMMA", 19), ("HMNMX2", 23), ("HMUL2", 7),
  ("HSET2", 5), ("HSETP2", 56), ("I2F", 18), ("I2FP", 4), ("I2I", 21), ("I2IP", 5), ("IABS", 5),
  ("IADD3", 4), ("IDP", 16), ("IMAD", 3), ("IMMA", 19), ("ISETP", 4), ("LEA", 4), ("LOP3", 4),
  ("MOV", 4), ("MOVM", 6), ("MUFU", 25), ("PLOP3", 2), ("POPC", 62), ("PRMT", 4), ("QSPC", 52),
  ("R2UR", 8), ("SEL", 4), ("SGXT", 4), ("SHF", 4), ("SHFL", 23), ("UFLO", 6), ("UIADD3", 4),
  ("UISETP", 4), ("ULEA", 4), ("ULOP3", 4), ("UMOV", 2), ("UPOPC", 6), ("UPRMT", 7), ("USHF", 4),
  ("VABSDIFF", 102), ("VABSDIFF4", 19), ("VIADD", 4), ("VIADDMNMX", 4), ("VIMNMX", 4),
  ("VIMNMX3", 7), ("VOTE", 4), ("VOTEU", 2)]

/-- A rule's latency held to `encoded`. -/
private def retime (r : OpRule) : OpRule :=
  match encoded.lookup r.opcode with
  | some d =>
    if d ≥ r.timing.value.latency then r else
    { r with timing := { r.timing with
        value := { r.timing.value with latency := d }
        kind := .inferred
        note := "" } }
  | none => r

end Blackwell

open Blackwell in
def b200 : Params := {
  name := "B200"
  target := Blackwell.target
  features := Blackwell.features
  sms := pub 148 "jarmusch2025" "§IV" "B200"
    ""
  gpcSMs := inferred [19, 19, 19, 19, 18, 18, 18, 18] "jarmusch2025" "§IV: 148 SMs in eight GPCs"
    ""
  smsps
  clockMHz := pub 1965 "jarmusch2025" "§IV" "B200"
    ""
  maxThreadsPerSM := doc 2048 "cuda-guide" "§5.1.3 Table 33, compute capability 10.0"
  maxCtasPerSM := doc 32 "cuda-guide" "§5.1.3 Table 33, compute capability 10.0"
    ""
  maxWarpsPerSM := doc 64 "cuda-guide" "§5.1.3 Table 33, compute capability 10.0"
  regsPerSM := doc 65536 "cuda-guide" "§5.1.3 Table 34, compute capability 10.0: 64 K"
  regAllocUnit
  smemPerSM := doc 233472 "cuda-guide" "§5.1.3 Table 34, compute capability 10.0: 228 KB"
  smemPerCtaMax := doc 232448 "cuda-guide" "§5.1.3 Table 34, compute capability 10.0: 227 KB"
  smemReservedPerCta := doc 1024 "blackwell-tuning" "§1.4.2.3: CUDA reserves 1 KB of shared memory per thread block"
  issuePolicy, readRelease
  maxClusterCtas := doc 16 "blackwell-tuning" "§1.4.1.2: B200 allows a nonportable cluster size of 16"
  regFile
  frontEnd := { frontEnd with
    codeCaches := inferred [(24576, 10), (49152, 25), (147456, 16)] "meas" "" "" }
  barrierCycles := onH200 barrierCycles
  barrierArrivalCycles := onH200 barrierArrivalCycles
  ops := ((((ops : Array OpRule) ++ BlackwellRTX.added).map fun (r : OpRule) =>
    { r with timing := onH200 r.timing }) ++ tcgen05Ops.toArray).map retime
  mem := {
    sectorBytes, lineBytes, smemBanks, smemBankBytes, mioQueueDepth, lsuInterval, addressInterval,
    l1BytesPerCycle, l1TagSectorsPerCycle, l1Ways, l1Mshrs, l1StoreInvalidates, l2Ways,
    l2SliceBytesPerCycle, l2SliceWriteBytesPerCycle, dsmemBytesPerCycle,
    smemAtomicInterval, l2FarCopiesNear, l2ClockRefMHz, l2LatencyFixedPermille, l2LatencyRefMHz, dramTurnaroundPs, xbarWriteBytesPerCycle, xbarRedBytesPerCycle
    storeRegCycles := onH200 storeRegCycles
    xbarBytesPerCycle := onH200 xbarBytesPerCycle
    l2FarBytesPerCycle := inferred 2900 "meas" ""
      ""
    l1SmemBytes := doc 262144 "blackwell-tuning"
      "§1.4.2.3: the B200's combined L1, texture cache and shared memory is 256 KB, as Hopper's"
    smemCarveoutsKB := inferred smemCarveoutsKB.value "cuda-guide" "§5.1.3 Table 35, compute capability 9.0"
      ""
    smemLatency := onH200 smemLatency
    l1HitLatency := onH200 l1HitLatency
    dsmemLatency := inferred 207 "meas" ""
      ""
    l2Partitions := pub 4 "jarmusch2025" "§IV: four L2 cache partitions (double those in Hopper)" "B200"
    l2NearHitLatency := inferred 299 "meas" ""
      ""
    l2FarHitLatency := inferred 434 "luo2025" "Table 4: far hit − near hit = 156 cycles"
      "" (some "H800 PCIe")
    l2Bytes := doc 132120576 "blackwell-tuning" "§1.4.2.2: L2 cache capacity 126 MB"
      ""
    l2Slices := measOn 192 ""
      ""
    l2RedF32BytesPerSec := inferred 6927000000000 "meas-h200" ""
      "" (some "H200")
    dramLatency := inferred 781 "jarmusch2025" "§V"
      "" (some "B200")
    dramBusBits := measOn 7680 ""
      ""
    dramClockMHz := measOn 3996 "" ""
    dramBytesPerSec := measOn 7672000000000 ""
      ""
    dramStreamBytesPerSec := measOn 7311000000000 ""
      ""
    dramChannels := inferred 120 "meas" ""
      "" }
  tensor := {
    fmaPerCycle16 := inferred 4096 "hgx-b200" "HGX B200: FP16/BF16 tensor 36 PFLOPS sparse for eight GPUs"
      ""
    fmaPerCycle8 := inferred 8192 "hgx-b200" "HGX B200: FP8/FP6 tensor 72 PFLOPS sparse" ""
    fmaPerCycleTF32 := inferred 2048 "hgx-b200" "HGX B200: TF32 tensor 18 PFLOPS sparse" ""
    fmaPerCycleF64 := inferred 64 "hgx-b200" "HGX B200: FP64/FP64 tensor 296 TFLOPS for eight GPUs"
      ""
    mmaSyncLatency := onH200 mmaSyncLatency
    wgmmaLatency := doc [] "ptx-isa-9.5" "§9.7.17 Target ISA notes: wgmma requires sm_90a"
    wgmmaLatencyRS := doc [] "ptx-isa-9.5" "§9.7.17 Target ISA notes: wgmma requires sm_90a"
    tcgen05Latency := inferred 214 "meas" ""
      ""
    tmemLatency := inferred 3 "meas" ""
      "" }
  sync := {
    mbarrierLatency := measOn 57 ""
    clusterMbarrierLatency := measOn 380 ""
    clusterBarrierCycles
    namedBarrierLatency := measOn 37 ""
    barArriveCycles := inferred 10 "meas" "" ""
    barSync := measOn [(64, 16), (128, 20), (256, 28), (512, 44), (1024, 76)] ""
    fenceProxyAsync := measOn 18 "" ""
    fenceAcqRelCta := measOn 30 "" ""
    fenceAcqRelGpu := measOn 275 "" ""
    fenceScGpu := measOn 274 "" ""
    fenceAcqRelSys := measOn 1710 "" ""
    fenceCycles := inferred [("MEMBAR.ALL.CTA", 22), ("FENCE.VIEW.ASYNC.S", 0), ("MEMBAR.ALL.GPU", 259),
      ("MEMBAR.SC.GPU", 258), ("MEMBAR.ALL.SYS", 1694)] "meas" ""
      ""
    redReleaseGpu := measOn 796 "" ""
    atomAcqRelGpu := measOn 1662 ""
    atomRelaxedGpu := measOn 886 ""
    ldAcquireGpu := measOn 472 ""
    flagHopPs := measOn 793000 ""
    printfPs := measOn 23550000 ""
      ""
    printfLanePs := measOn 116000 ""
      ""
    tryWaitLimit := measOn 1941 ""
    tryWaitWake := inferred 16 "meas" ""
      "" }
  tma := {
    copyDram := onH200 h200.tma.copyDram
    copyL2 := onH200 h200.tma.copyL2
    issueBytesPerSec := onH200 h200.tma.issueBytesPerSec
    storeCreditBytes := onPart "GH200" Hopper.storeCreditBytes }
  launch := {
    emptyGraphPs := measOn 1220000 ""
    kernelGapPs := measOn 768000 "" ""
    kernelGapPdlPs := measOn 512000 ""
    ctaDispatchCycles
    gpcLaunchCycles := onH200 gpcLaunchCycles }
  link := {
    peerGBps := doc 900 "hgx-b200" "HGX B200 and HGX B300: NVLink GPU-to-GPU bandwidth 1.8 TB/s"
      ""
    peerPermille := Hopper.peerPermille } }

theorem b200_sms_gpcs : b200.gpcSMs.value.sum = b200.sms.value := by decide

namespace Blackwell

def ultraTarget : Fact String :=
  doc "sm_103a" "ptx-isa-9.5" "§9.7.18 Target ISA notes: tcgen05 on sm_103a"

/-- `MUFU.EX2` on Blackwell Ultra: twice Blackwell's rate. -/
private def ex2Ultra : OpRule :=
  let r := (b200.ops.find? fun r => r.opcode == "MUFU" && r.mods.isEmpty).get!
  { r with mods := #["EX2"],
           timing := { r.timing with
             value := { r.timing.value with interval := r.timing.value.interval / 2 }
             kind := .inferred
             source := { ref := "nvidia-blackwell-ultra", loc := "Table 2: SFU EX2" }
             note := "" } }

end Blackwell

open Blackwell in
/-- The B300 (Blackwell Ultra): the B200 with Blackwell Ultra's documented differences; its SM
count and clock the B200's (inferred). -/
def b300 : Params := { b200 with
  name := "B300"
  target := ultraTarget
  sms := inferred 148 "hgx-b200" "HGX B300 and HGX B200: FP32 600 TFLOPS, FP16/BF16 36 and FP8 72 PFLOPS sparse alike"
    ""
  -- First match wins (`Params.timingOf`): MUFU.EX2's rule ahead of MUFU's.
  ops := #[ex2Ultra] ++ b200.ops
  tensor := { b200.tensor with
    fmaPerCycleF64 := inferred 2 "hgx-b200" "HGX B300: FP64/FP64 Tensor Core 10 TFLOPS for eight GPUs"
      "" } }

theorem b300_sms_gpcs : b300.gpcSMs.value.sum = b300.sms.value := by decide

end NV.Arch
