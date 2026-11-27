import NV.Arch.H200

/-!
# H100 SXM5 (`sm_90a`)

GH100 with 132 SMs and five HBM3 stacks (80 GB). The SM is the H200's (`NV.Arch.Hopper`); the
memory system is H100's own, from NVIDIA's documentation; a value taken from another part is
labelled.
-/

namespace NV.Arch

open NV NV.Core Hopper

def h100 : Params := {
  name := "H100 SXM5"
  target := Hopper.target
  features := Hopper.features
  sms := measOn 132 ""
    ""
  gpcSMs := inferred h200GpcSMs.value "meas-h200" h200GpcSMs.source.loc
    "" (some "H200")
  smsps
  clockMHz := doc 1980 "hopper-whitepaper"
    "Table 3: GPU boost clock for FP64 tensor, FP32 and FP64 non-tensor operations, 1980 MHz"
    ""
  maxThreadsPerSM, maxCtasPerSM, maxWarpsPerSM, regsPerSM, regAllocUnit, smemPerSM, smemPerCtaMax
  smemReservedPerCta, issuePolicy, readRelease, maxClusterCtas
  regFile
  frontEnd := { frontEnd with codeCaches := onH200 frontEnd.codeCaches }
  barrierCycles := onH200 barrierCycles
  barrierArrivalCycles := onH200 barrierArrivalCycles
  ops := ops.map fun r => { r with timing := onH200 r.timing }
  mem := {
    sectorBytes, lineBytes, l1SmemBytes, smemBanks, smemBankBytes, l2Partitions,
    l2FarHitLatency, mioQueueDepth, lsuInterval, addressInterval, storeRegCycles,
    l1BytesPerCycle, l1TagSectorsPerCycle, l1Ways, l1Mshrs, l1StoreInvalidates, l2Ways,
    l2SliceBytesPerCycle, l2SliceWriteBytesPerCycle, dsmemBytesPerCycle, smemCarveoutsKB, l2FarBytesPerCycle,
    smemAtomicInterval, l2FarCopiesNear, l2ClockRefMHz, l2LatencyFixedPermille, l2LatencyRefMHz, dramTurnaroundPs
    xbarBytesPerCycle := onH200 xbarBytesPerCycle
    xbarWriteBytesPerCycle, xbarRedBytesPerCycle, gpcLinkBytesPerCycle
    l2NearHitLatency := onH200 l2NearHitLatency
    dramChannels := inferred 80 "hopper-whitepaper" "Table 3, H100 SXM5: 5120-bit HBM3"
      ""
    l2RedF32BytesPerSec := inferred 2886250000000 "meas-h200" ""
      "" (some "H200")
    smemLatency := onH200 smemLatency
    l1HitLatency := onH200 l1HitLatency
    dsmemLatency := onH200 dsmemLatency
    l2Bytes := measOn 52428800 ""
      ""
    l2Slices := measOn 80 ""
      ""
    dramLatency := inferred h200.mem.dramLatency.value "meas-h200"
      ""
      "" (some "H200")
    dramBusBits := doc 5120 "hopper-whitepaper" "Table 3, H100 SXM5: memory interface 5120-bit HBM3"
    dramClockMHz := doc 2619 "hopper-whitepaper" "Table 3, H100 SXM5: memory data rate 2619 MHz DDR"
      ""
    dramBytesPerSec := doc 3352000000000 "hopper-whitepaper" "Table 3, H100 SXM5: 3352 GB/sec"
      ""
    dramStreamBytesPerSec := inferred 3210000000000 "meas-h200"
      ""
      "" (some "H200") }
  tensor := {
    fmaPerCycle16, fmaPerCycle8, fmaPerCycleTF32, fmaPerCycleF64, wgmmaLatency, wgmmaLatencyRS, wgmmaOperandDataPermille, wgmmaIssueSharePermille, wgmmaPipeSharePermille, wgmmaQueueCycles,
    wgmmaWaitCycles, mmaSyncLatency := onH200 mmaSyncLatency }
  sync := {
    mbarrierLatency := onH200 mbarrierLatency
    clusterMbarrierLatency := onH200 clusterMbarrierLatency
    clusterBarrierCycles
    namedBarrierLatency := onH200 namedBarrierLatency
    barArriveCycles := onH200 barArriveCycles
    barSync := onH200 barSync
    fenceProxyAsync := onH200 fenceProxyAsync
    fenceAcqRelCta := onH200 fenceAcqRelCta
    fenceAcqRelGpu := onH200 fenceAcqRelGpu
    fenceScGpu := onH200 fenceScGpu
    fenceAcqRelSys := onH200 fenceAcqRelSys
    fenceCycles := onH200 fenceCycles
    fenceDrainCycles := onH200 fenceDrainCycles
    redReleaseGpu := onH200 redReleaseGpu
    atomAcqRelGpu := onH200 atomAcqRelGpu
    atomRelaxedGpu := onH200 atomRelaxedGpu
    ldAcquireGpu := onH200 ldAcquireGpu
    flagHopPs := onH200 flagHopPs
    printfPs, printfLanePs
    tryWaitLimit := onPart "GH200" tryWaitLimit
    tryWaitWake }
  tma := {
    copyDram := onH200 h200.tma.copyDram
    copyL2 := onH200 h200.tma.copyL2
    issueBytesPerSec := onH200 h200.tma.issueBytesPerSec
    storeCreditBytes := Hopper.storeCreditBytes }
  launch := {
    emptyGraphPs := onH200 emptyGraphPs
    kernelGapPs := onH200 kernelGapPs
    kernelGapPdlPs := onH200 kernelGapPdlPs
    ctaDispatchCycles
    gpcLaunchCycles := inferred 3880 "meas-h100" ""
      "" }
  link := {
    peerGBps := doc 450 "h100-product" "Interconnect, H100 SXM: NVIDIA NVLink 900GB/s"
      ""
    peerPermille } }

/-- H100 PCIe, 80 GB HBM2e: GH100 with 114 SMs, five HBM2e stacks, a lower clock. The SM is
Hopper's; the device and memory facts are the board's attributes and NVIDIA's documents, memory
latencies luo2025's H800 PCIe, SM-local costs the H200's. -/
def h100Pcie : Params := { h100 with
  name := "H100 PCIe 80 GB"
  link := { h100.link with
    peerGBps := doc 300 "h100-pcie-brief" "Table 6: total maximum NVLink bandwidth 600 Gbytes per second"
      "" }
  sms := measOn 114 ""
    ""
  gpcSMs := inferred [18, 16, 16, 16, 16, 16, 16] "meas"
    ""
    ""
  clockMHz := measOn 1755 ""
    ""
  mem := { h100.mem with
    l2Bytes := measOn 52428800 ""
      ""
    l2Slices := measOn 80 ""
      ""
    l2NearHitLatency := pub 258 "luo2025" "Table 3 (near L2 hit)" "H800 PCIe"
      ""
    l2FarHitLatency := pub 414 "luo2025" "Table 3 (far L2 hit)" "H800 PCIe"
    dramLatency := pub 650 "luo2025" "Table 4: near 555.5, far 743.7 cycles" "H800 PCIe"
      ""
    dramBusBits := measOn 5120 ""
      ""
    dramClockMHz := measOn 1593 ""
      ""
    dramBytesPerSec := doc 2039000000000 "hopper-whitepaper" "Table 3, H100 PCIe: 2039 GB/sec"
      ""
    dramStreamBytesPerSec := pub 1861500000000 "luo2025" "Table 5 (global memory throughput)" "H800 PCIe"
      ""
    dramChannels := inferred 80 "hopper-whitepaper" "Table 3, H100 PCIe: 5120-bit HBM2e"
      ""
    l2RedF32BytesPerSec := inferred 2886250000000 "meas-h200" ""
      "" (some "H200") } }

/-- H100 NVL, 94 GB HBM3 on a PCIe card: GH100 with six HBM3 stacks on a 6016-bit interface.
Device facts are NVIDIA's documents; the SM count is inferred. -/
def h100Nvl : Params := { h100 with
  name := "H100 NVL 94 GB"
  link := { h100.link with
    peerGBps := doc 300 "h100-product" "Interconnect, H100 NVL: NVIDIA NVLink 600GB/s"
      "" }
  sms := inferred 132 "h100-datasheet" "p.2: FP32 60 TFLOPS at 1785 MHz boost"
    ""
  gpcSMs := inferred h200GpcSMs.value "meas-h200" h200GpcSMs.source.loc
    "" (some "H200")
  clockMHz := doc 1785 "h100-nvl-brief" "Table 1: boost 1785 MHz, base 1080 MHz"
  mem := { h100.mem with
    dramBusBits := doc 6016 "h100-nvl-brief" "Table 2: 6016-bit HBM3"
    dramClockMHz := doc 2619 "h100-nvl-brief" "Table 2: 2619 MHz"
    dramBytesPerSec := doc 3938000000000 "h100-nvl-brief" "Table 2: 3938 GB/s"
      ""
    dramStreamBytesPerSec := inferred 3772000000000 "meas-h200" ""
      ""
      (some "H200")
    dramChannels := inferred 94 "h100-nvl-brief" "Table 2: 6016-bit HBM3"
      ""
    l2Bytes := onPart "H100 SXM5" h100.mem.l2Bytes
    l2Slices := inferred 96 "meas" ""
      ""
    l2RedF32BytesPerSec := inferred 3463500000000 "meas-h200" ""
      "" (some "H200") } }

theorem h100_sms_gpcs : h100.gpcSMs.value.sum = h100.sms.value := by decide
theorem h100Pcie_sms_gpcs : h100Pcie.gpcSMs.value.sum = h100Pcie.sms.value := by decide
theorem h100Nvl_sms_gpcs : h100Nvl.gpcSMs.value.sum = h100Nvl.sms.value := by decide

end NV.Arch
