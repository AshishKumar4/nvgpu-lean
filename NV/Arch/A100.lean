import NV.Arch.Ampere

/-!
# A100 SXM4, 40 GB and 80 GB (`sm_80`)

GA100 with 108 SMs and five HBM2 (40 GB) or HBM2e (80 GB) stacks on a 5120-bit interface. The SM
is Ampere's (`NV.Arch.Ampere`); the device facts are A100 boards' attributes and NVIDIA's
documents. Memory latencies are published measurements of A100 boards; a value taken from
another part is labelled.
-/

namespace NV.Arch

open NV NV.Core Ampere

/-- Per-GPC SMs: 54 TPCs over 7 GPCs of 7 or 8 TPCs (a100-whitepaper Table 4: 7 GPCs, 54 TPCs). -/
def a100GpcSMs : Fact (List Nat) :=
  inferred [16, 16, 16, 16, 16, 14, 14] "meas" ""
    ""

def a100_40 : Params := {
  name := "A100 SXM4 40 GB"
  target, features
  sms := measOn 108 "" ""
  gpcSMs := a100GpcSMs
  smsps
  clockMHz := measOn 1410 "" ""
  maxThreadsPerSM, maxCtasPerSM, maxWarpsPerSM, regsPerSM, regAllocUnit, smemPerSM, smemPerCtaMax
  smemReservedPerCta, issuePolicy, readRelease, maxClusterCtas, regFile, frontEnd
  barrierCycles, barrierArrivalCycles, ops
  mem := {
    l2ClockRefMHz := inferred 1410 "a100-whitepaper" "Table 4: boost clock 1410 MHz"
      ""
    l2LatencyFixedPermille := inferred 0 "meas-gh200" ""
      ""
    l2LatencyRefMHz := inferred 1410 "a100-whitepaper" "Table 4: boost clock 1410 MHz"
      ""
    dramTurnaroundPs := onPart "H100 SXM5" Hopper.dramTurnaroundPs
    sectorBytes := Hopper.sectorBytes
    lineBytes := Hopper.lineBytes
    l1SmemBytes, smemCarveoutsKB
    smemAtomicInterval := Hopper.smemAtomicInterval
    smemBanks := doc 32 "cuda-guide" "§5.1.3 Table 34: 32 shared memory banks (compute capability 8.0)"
    smemBankBytes := Hopper.smemBankBytes
    smemLatency, l1HitLatency
    dsmemLatency := doc 0 "cuda-guide" "§5.1 Table 32: distributed shared memory needs 9.0"
      ""
    mioQueueDepth := { Hopper.mioQueueDepth with kind := .published }
    lsuInterval := pub 1 "luo2024" "Table V: A100 shared memory 128.0 bytes/clk/SM" "A100 PCIe"
      ""
    addressInterval := { Hopper.addressInterval with kind := .published }
    storeRegCycles := inferred 3 "meas-a100" ""
      ""
    l1BytesPerCycle := pub 128 "luo2024" "Table V: shared 128.0, L1 99.5–120.0 bytes/clk/SM"
      "A100 PCIe" ""
    l1TagSectorsPerCycle := onPart "H800 PCIe" Hopper.l1TagSectorsPerCycle
    l1Ways := Hopper.l1Ways
    l1Mshrs := Hopper.l1Mshrs
    l1StoreInvalidates := Hopper.l1StoreInvalidates
    xbarBytesPerCycle := onH200 Hopper.xbarBytesPerCycle
    xbarWriteBytesPerCycle := onPart "GH200" Hopper.xbarWriteBytesPerCycle
    xbarRedBytesPerCycle := onPart "GH200" Hopper.xbarRedBytesPerCycle
    dsmemBytesPerCycle := doc 0 "cuda-guide" "§5.1 Table 32: no distributed shared memory before 9.0"
    l2Partitions, l2NearHitLatency, l2FarHitLatency, l2SliceBytesPerCycle
    l2SliceWriteBytesPerCycle := onPart "GH200" Hopper.l2SliceWriteBytesPerCycle
    l2FarCopiesNear := doc true "a100-whitepaper" "p. 35: each L2 partition localizes and caches
      data for memory accesses from SMs in the GPCs directly connected to the partition"
    l2Ways := Hopper.l2Ways
    l2FarBytesPerCycle := inferred 1280 "a100-whitepaper" "p. 35: L2 read bandwidth 5120 bytes/clk"
      "" none
    l2Bytes := measOn 41943040 "" ""
    l2Slices := measOn 80 "" ""
    l2RedF32BytesPerSec := inferred 2309000000000 "meas-h200" ""
      ""
      (some "H200")
    dramLatency := dramLatencyPcie
    dramBusBits := measOn 5120 "" ""
    dramClockMHz := measOn 1215 "" ""
    dramBytesPerSec := doc 1555000000000 "a100-whitepaper" "Table 4: 1555 GB/sec"
      ""
    dramStreamBytesPerSec := pub 1407200000000 "luo2024" "Table V: global memory 1407.2 GB/s"
      "A100 PCIe" ""
    dramChannels := inferred 80 "a100-whitepaper" "p. 35: ten 512-bit memory controllers"
      "" }
  tensor := {
    fmaPerCycle16 := tensorFma16, fmaPerCycle8 := tensorFma8, fmaPerCycleTF32 := tensorFmaTF32
    fmaPerCycleF64 := tensorFmaF64, mmaSyncLatency
    wgmmaLatency := doc [] "cuda-guide" "§5.1 Table 32: warpgroup MMA needs sm_90a"
    wgmmaLatencyRS := doc [] "cuda-guide" "§5.1 Table 32: warpgroup MMA needs sm_90a" }
  sync := Ampere.sync
  tma := {
    copyDram := doc [] "cuda-guide" "§5.1 Table 32: no tensor memory accelerator before 9.0"
    copyL2 := doc [] "cuda-guide" "§5.1 Table 32: no tensor memory accelerator before 9.0"
    issueBytesPerSec := doc 0 "cuda-guide" "§5.1 Table 32: no tensor memory accelerator before 9.0"
    storeCreditBytes := doc 0 "cuda-guide" "§5.1 Table 32: no tensor memory accelerator before 9.0" }
  launch := {
    emptyGraphPs := onH200 Hopper.emptyGraphPs
    kernelGapPs := onH200 Hopper.kernelGapPs
    kernelGapPdlPs := onH200 Hopper.kernelGapPdlPs
    ctaDispatchCycles := Hopper.ctaDispatchCycles
    gpcLaunchCycles := inferred 2731 "meas-a100" ""
      "" }
  link := {
    peerGBps := inferred 300 "a100-datasheet-80" "p. 1, A100 80GB SXM: NVLink 600GB/s"
      ""
    peerPermille := Hopper.peerPermille } }

/-- The 80 GB part: HBM2e at a higher clock; the rest is the 40 GB part's. -/
def a100_80 : Params := { a100_40 with
  name := "A100 SXM4 80 GB"
  sms := measOn 108 ""
  clockMHz := measOn 1410 ""
  mem := { a100_40.mem with
    l2Bytes := measOn 41943040 ""
    l2Slices := measOn 80 ""
    dramClockMHz := measOn 1593 "" ""
    dramBytesPerSec := doc 2039000000000 "a100-datasheet-80" "p. 1: HBM2e, 2039 GB/s (SXM)"
      ""
    dramStreamBytesPerSec := inferred 1845000000000 "luo2024" "Table V: 1407.2 of 1555 GB/s"
      "" (some "A100 PCIe") } }

theorem a100_sms_gpcs : a100_40.gpcSMs.value.sum = a100_40.sms.value := by decide

end NV.Arch
