import NV.Arch.Hopper

/-!
# H200 SXM (`sm_90a`)

GH100 with 132 SMs and six HBM3e stacks (141 GB). The SM facts are Hopper's (`NV.Arch.Hopper`);
this file adds the device's structure, its memory system and the costs measured on it.
-/

namespace NV.Arch

open NV NV.Core Hopper

/-- Per-GPC SM counts: clusters of 1, 2, 4, 8 and 16 CTAs at one CTA per SM fill 132, 66, 30,
15 and 7 clusters (`h200_cluster_occupancy`). -/
def h200GpcSMs : Fact (List Nat) :=
  inferred [18, 18, 18, 18, 18, 18, 16, 8] "meas-h200"
    ""
    ""

def h200 : Params := {
  name := "H200 SXM",
  target := Hopper.target,
  features := Hopper.features,
  sms := meas 132 "" "",
  gpcSMs := h200GpcSMs,
  smsps,
  clockMHz := meas 1980 ""
    "",
  maxThreadsPerSM,
  maxCtasPerSM,
  maxWarpsPerSM,
  regsPerSM,
  regAllocUnit,
  smemPerSM,
  smemPerCtaMax,
  smemReservedPerCta,
  issuePolicy,
  readRelease,
  barrierCycles,
  barrierArrivalCycles,
  maxClusterCtas,
  regFile,
  frontEnd,
  mem := {
    sectorBytes, lineBytes, l1SmemBytes, smemBanks, smemBankBytes, smemLatency, l1HitLatency,
    dsmemLatency, l2Partitions, l2NearHitLatency, l2FarHitLatency, mioQueueDepth, lsuInterval,
    addressInterval, storeRegCycles, l1BytesPerCycle, l1TagSectorsPerCycle, l1Ways, l1Mshrs, l1StoreInvalidates,
    l2Ways, l2SliceBytesPerCycle, l2SliceWriteBytesPerCycle, xbarBytesPerCycle, gpcLinkBytesPerCycle, xbarWriteBytesPerCycle, xbarRedBytesPerCycle, dsmemBytesPerCycle, smemCarveoutsKB,
    l2FarBytesPerCycle, smemAtomicInterval, l2FarCopiesNear, l2ClockRefMHz, l2LatencyFixedPermille, l2LatencyRefMHz, dramTurnaroundPs
    dramChannels := inferred 94 "meas-h200" ""
      ""
    l2RedF32BytesPerSec := meas 3463500000000 ""
      ""
    l2Bytes := meas 62914560 ""
      ""
    l2Slices := meas 96 ""
      ""
    dramLatency := inferred 654 "meas-h200" ""
      ""
    dramBusBits := meas 6016 ""
      ""
    dramClockMHz := meas 3201 ""
    dramBytesPerSec := doc 4800000000000 "h200-product"
      "Specifications, H200 SXM: GPU memory bandwidth 4.8 TB/s"
      ""
    dramStreamBytesPerSec := meas 4610000000000 ""
      "" },
  tensor := {
    fmaPerCycle16, fmaPerCycle8, fmaPerCycleTF32, fmaPerCycleF64, wgmmaLatency, wgmmaLatencyRS,
    wgmmaOperandDataPermille, wgmmaIssueSharePermille, wgmmaPipeSharePermille, mmaSyncLatency },
  sync := {
    mbarrierLatency, clusterMbarrierLatency, clusterBarrierCycles, namedBarrierLatency, barArriveCycles, barSync,
    fenceProxyAsync,
    fenceAcqRelCta, fenceAcqRelGpu, fenceScGpu, fenceAcqRelSys, fenceCycles, redReleaseGpu,
    atomAcqRelGpu,
    atomRelaxedGpu, ldAcquireGpu, flagHopPs, printfPs, printfLanePs,
    tryWaitLimit := onPart "GH200" tryWaitLimit, tryWaitWake },
  tma := {
    copyDram := meas [(1024, 996), (4096, 1244), (16384, 1241), (36864, 1442), (65536, 1696)]
      ""
      ""
    copyL2 := meas [(1024, 326), (4096, 340), (16384, 459), (36864, 650), (65536, 920)]
      ""
    issueBytesPerSec := meas 259600000000 ""
      ""
    storeCreditBytes := Hopper.storeCreditBytes },
  launch := { emptyGraphPs, kernelGapPs, kernelGapPdlPs, ctaDispatchCycles, gpcLaunchCycles },
  link := {
    peerGBps := doc 450 "h200-product" "Interconnect, H200 SXM: NVIDIA NVLink 900GB/s"
      ""
    peerPermille },
  ops }

/-- The layout's cluster capacities. -/
theorem h200_cluster_occupancy :
    [1, 2, 4, 8, 16].map (fun size => (h200.gpcSMs.value.map (· / size)).sum)
      = [132, 66, 30, 15, 7] := by decide

theorem h200_sms_gpcs : h200.gpcSMs.value.sum = h200.sms.value := by decide

end NV.Arch
