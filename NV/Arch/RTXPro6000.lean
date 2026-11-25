import NV.Arch.BlackwellRTX
import NV.Arch.H200

/-!
# RTX PRO 6000 Blackwell Server Edition, the GPU of Google Cloud's G4 (`sm_120a`)

GB202 with 188 SMs and 96 GB of GDDR7 on a 512-bit interface. The SM is Blackwell RTX's
(`NV.Arch.BlackwellRTX`). A value taken from another part is labelled.
-/

namespace NV.Arch

open NV NV.Core BlackwellRTX

/-- 94 TPCs over 12 GPCs (rtxpro-whitepaper Table 1: 12 GPCs; 188 SMs). -/
def rtxPro6000GpcSMs : Fact (List Nat) :=
  inferred [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 14, 14] "rtxpro-whitepaper" "Table 1: 12 GPCs"
    ""

def rtxPro6000 : Params := {
  name := "RTX PRO 6000 Blackwell Server Edition (G4)"
  target, features
  sms := inferred 188 "rtxpro6000-server" "p. 3: 24,064 CUDA cores"
    ""
  gpcSMs := rtxPro6000GpcSMs
  smsps
  clockMHz := inferred assumedMHz "rtxpro-whitepaper" "Table 1: boost clock 2617 MHz (workstation)"
    ""
  maxThreadsPerSM, maxCtasPerSM, maxWarpsPerSM, regsPerSM, regAllocUnit, smemPerSM, smemPerCtaMax
  smemReservedPerCta, maxClusterCtas, ops
  issuePolicy := Ampere.issuePolicy
  readRelease := Ampere.readRelease
  regFile := Ampere.regFile
  frontEnd := { Ampere.frontEnd with
    constCacheBytes := doc 8192 "cuda-guide" "§5.1.3 Table 34: 8 KB constant cache working set"
    stallZeroInterlocks := BlackwellRTX.stallZeroInterlocks }
  barrierCycles := onH200 Hopper.barrierCycles
  barrierArrivalCycles := onH200 Hopper.barrierArrivalCycles
  mem := {
    l2ClockRefMHz := inferred assumedMHz "rtxpro-whitepaper" "Table 1: boost clock 2617 MHz (workstation)"
      ""
    l2LatencyFixedPermille := inferred 0 "meas-gh200" ""
      ""
    l2LatencyRefMHz := inferred assumedMHz "rtxpro-whitepaper" "Table 1: boost clock 2617 MHz (workstation)"
      ""
    dramTurnaroundPs := onPart "H100 SXM5" Hopper.dramTurnaroundPs
    sectorBytes := Hopper.sectorBytes
    lineBytes := Hopper.lineBytes
    l1SmemBytes, smemCarveoutsKB, smemLatency, l1HitLatency
    smemAtomicInterval := Hopper.smemAtomicInterval
    smemBanks := doc 32 "cuda-guide" "§5.1.3 Table 34: 32 shared memory banks"
    smemBankBytes := Hopper.smemBankBytes
    dsmemLatency := onH200 Hopper.dsmemLatency
    mioQueueDepth := { Hopper.mioQueueDepth with kind := .published }
    lsuInterval := inferred 1 "chipsandcheese-blackwell"
      "SM Memory Subsystem: shared 62,038 GB/s on 188 SMs at 2.617 GHz, 126 bytes per cycle per SM"
      "" (some "RTX PRO 6000 Blackwell")
    addressInterval := { Hopper.addressInterval with kind := .published }
    storeRegCycles := onH200 Hopper.storeRegCycles
    l1BytesPerCycle := inferred 128 "chipsandcheese-blackwell"
      "SM Memory Subsystem: L1 63,029 GB/s, shared 62,038 GB/s (RTX PRO 6000 workstation)"
      ""
      (some "RTX PRO 6000 Blackwell")
    l1TagSectorsPerCycle := onPart "H800 PCIe" Hopper.l1TagSectorsPerCycle
    l1Ways := Hopper.l1Ways
    l1Mshrs := Hopper.l1Mshrs
    l1StoreInvalidates := Hopper.l1StoreInvalidates
    xbarBytesPerCycle := onH200 Hopper.xbarBytesPerCycle
    xbarWriteBytesPerCycle := onPart "GH200" Hopper.xbarWriteBytesPerCycle
    gpcLinkBytesPerCycle := onPart "GH200" Hopper.gpcLinkBytesPerCycle
    xbarRedBytesPerCycle := onPart "GH200" Hopper.xbarRedBytesPerCycle
    dsmemBytesPerCycle := onPart "H800 PCIe" Hopper.dsmemBytesPerCycle
    l2Partitions := inferred 1 "chipsandcheese-blackwell" "GPU-Wide Memory Subsystem"
      "" (some "RTX PRO 6000 Blackwell")
    l2NearHitLatency := l2HitLatency
    l2FarHitLatency := l2HitLatency
    l2FarCopiesNear := doc false "rtxpro-whitepaper" "p. 9: one L2"
    l2SliceWriteBytesPerCycle := onPart "GH200" Hopper.l2SliceWriteBytesPerCycle
    l2SliceBytesPerCycle := inferred 26 "chipsandcheese-blackwell" "GPU-Wide Memory Subsystem: L2 about 8.7 TB/s"
      ""
      (some "RTX PRO 6000 Blackwell")
    l2Ways := Hopper.l2Ways
    l2FarBytesPerCycle := doc 0 "rtxpro-whitepaper" "p. 9: one L2, no far partition"
    l2Bytes := doc 134217728 "rtxpro-whitepaper" "p. 9 and Table 4: 128 MB L2 (workstation edition)"
      ""
    l2Slices := inferred 128 "rtxpro-whitepaper" "p. 9: 128 MB L2"
      ""
    l2RedF32BytesPerSec := inferred 1700000000000 "meas-h200" ""
      "" (some "H200")
    dramLatency
    dramBusBits := doc 512 "rtxpro6000-server" "p. 3: 512-bit GDDR7"
    dramClockMHz := inferred 1560 "rtxpro6000-server" "p. 3: 1597 GB/s over 512 bits"
      ""
    dramBytesPerSec := doc 1597000000000 "rtxpro6000-server" "p. 3: 1597 GB/s" ""
    dramStreamBytesPerSec := inferred 1530000000000 "meas-h200" ""
      "" (some "H200")
    dramChannels := inferred 32 "rtxpro6000-server" "p. 3: 512-bit GDDR7"
      "" }
  tensor := {
    fmaPerCycle16 := inferred 512 "rtxpro-whitepaper" "Table 4: dense FP16 tensor 503.8 TFLOPS at 2617 MHz (workstation)"
      "" (some "RTX PRO 6000 Blackwell")
    fmaPerCycle8 := inferred 1024 "rtxpro-whitepaper" "Table 4: dense FP8 tensor" "" (some "RTX PRO 6000 Blackwell")
    fmaPerCycleTF32 := inferred 256 "rtxpro-whitepaper" "Table 4: dense TF32 tensor" "" (some "RTX PRO 6000 Blackwell")
    fmaPerCycleF64 := doc 0 "cuda-guide" "§5.1 Table 36: no FP64 tensor cores on compute capability 12.0"
    mmaSyncLatency
    wgmmaLatency := doc [] "ptx-isa-9.4" "§9.7.15 Target ISA notes: wgmma requires sm_90a"
    wgmmaLatencyRS := doc [] "ptx-isa-9.4" "§9.7.15 Target ISA notes: wgmma requires sm_90a" }
  sync := { Ampere.sync with
    clusterMbarrierLatency := onH200 Hopper.clusterMbarrierLatency
    clusterBarrierCycles := Hopper.clusterBarrierCycles
    tryWaitLimit := inferred 1941 "meas" ""
      "" (some "B200")
    tryWaitWake := inferred 16 "meas" "" ""
      (some "B200") }
  tma := {
    copyDram := onH200 h200.tma.copyDram
    copyL2 := onH200 h200.tma.copyL2
    issueBytesPerSec := inferred 83750000000 "yu2026" "§III-C: TMA about 32 bytes per SM per cycle"
      "" (some "RTX 5090")
    storeCreditBytes := onPart "GH200" Hopper.storeCreditBytes }
  launch := {
    emptyGraphPs := onH200 Hopper.emptyGraphPs
    kernelGapPs := onH200 Hopper.kernelGapPs
    kernelGapPdlPs := onH200 Hopper.kernelGapPdlPs
    ctaDispatchCycles := Hopper.ctaDispatchCycles
    gpcLaunchCycles := onH200 Hopper.gpcLaunchCycles }
  link := {
    peerGBps := inferred 63 "rtxpro6000-server" "p. 3: graphics bus PCI Express 5.0 x16; no NVLink"
      ""
    peerPermille := Hopper.peerPermille } }

theorem rtxPro6000_sms_gpcs : rtxPro6000.gpcSMs.value.sum = rtxPro6000.sms.value := by decide

end NV.Arch
