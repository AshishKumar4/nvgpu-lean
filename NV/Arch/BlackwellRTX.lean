import NV.Arch.Ampere

/-!
# Blackwell RTX (GB202, `sm_120a`): the RTX PRO 6000 Blackwell Server Edition (Google Cloud G4)

Compute capability 12.0: the consumer and workstation Blackwell SM. It has cp.async, the tensor
memory accelerator, thread block clusters and distributed shared memory, and warp-level tensor
instructions (`mma.sync`) only: no warpgroup MMA and no tcgen05, which belong to `sm_90a` and the
data-center `sm_100` family (ptx-isa-9.4 Target ISA notes).

Its facts are NVIDIA's and Google's documents, the RTX PRO Blackwell architecture whitepaper,
and published measurements of RTX PRO 6000 and RTX 5090 boards, each named; cycle counts from
times in nanoseconds are at `assumedMHz`.
-/

namespace NV.Arch.BlackwellRTX

open NV NV.Core

def target : Fact String :=
  doc "sm_120a" "ptx-isa-9.4" "§5 (Target ISA notes): sm_120a, compute capability 12.0 with its
    architecture-specific features"

def features : Fact Features :=
  doc { cpAsync := true, tma := true, wgmma := false, clusters := true } "cuda-guide"
    "§5.1 Table 32, compute capability 12.0: asynchronous copies, the tensor memory accelerator,
    thread block clusters, distributed shared memory; tensor cores through mma.sync (ptx-isa-9.4
    §9.7.14); wgmma is sm_90a's only"

def smsps : Fact Nat := doc 4 "rtxpro-whitepaper" "Fig. 5: four processing blocks per SM"

def maxThreadsPerSM : Fact Nat := doc 1536 "cuda-guide" "§5.1.3 Table 33, compute capability 12.0"
def maxWarpsPerSM : Fact Nat := doc 48 "cuda-guide" "§5.1.3 Table 33, compute capability 12.0"
def maxCtasPerSM : Fact Nat :=
  doc 24 "cuda-guide" "§5.1.3 Table 33, compute capability 12.0"
    ""
def regsPerSM : Fact Nat := doc 65536 "cuda-guide" "§5.1.3 Table 34: 64 K 32-bit registers per SM"
def regAllocUnit : Fact Nat :=
  doc 256 "cuda-occupancy" "register allocation unit size, warp granularity (unchanged since 5.x)"
def smemPerSM : Fact Nat := doc 102400 "cuda-guide" "§5.1.3 Table 34: 100 KB shared memory per SM"
def smemPerCtaMax : Fact Nat := doc 101376 "cuda-guide" "§5.1.3 Table 34: 99 KB per thread block"
def smemReservedPerCta : Fact Nat :=
  doc 1024 "blackwell-tuning" "§1.4.2.3: 1 KB of shared memory reserved per thread block"
def maxClusterCtas : Fact Nat :=
  doc 8 "blackwell-tuning" "§1.4.1.2: portable cluster size 8"
    ""

/-- Results per clock per SM where compute capability 12.0 differs from 9.0 (`cuda-best-practices`
§12.1.1 Table 5). -/
private def cc120 : List (String × Nat × String) := [
  ("DFMA", 2, "64-bit floating-point add, multiply, multiply-add"),
  ("DADD", 2, "64-bit floating-point add, multiply, multiply-add"),
  ("DMUL", 2, "64-bit floating-point add, multiply, multiply-add"),
  ("HFMA2", 64, "16-bit floating-point add, multiply, multiply-add (2-way SIMD)"),
  ("HADD2", 64, "16-bit floating-point add, multiply, multiply-add (2-way SIMD)"),
  ("HMUL2", 64, "16-bit floating-point add, multiply, multiply-add (2-way SIMD)"),
  ("HMNMX2", 64, "16-bit floating-point minimum, maximum"),
  ("FMNMX", 64, "32-bit floating-point minimum, maximum"),
  ("VOTE", 128, "warp vote"),
  ("VABSDIFF4", 32, "4-way SIMD absolute difference")]

/-- Each fixed-latency opcode's latency cap on `sm_120a`. -/
private def encoded : List (String × Nat) := [
  ("SHFL", 16),
  ("CS2R", 5), ("ELECT", 2), ("F2FP", 4), ("FADD", 4), ("FFMA", 4), ("FMNMX", 5), ("FMUL", 4),
  ("FSEL", 4), ("FSETP", 5), ("HADD2", 7), ("HFMA2", 4), ("HMNMX2", 22), ("HMUL2", 4),
  ("HSETP2", 63), ("I2FP", 5), ("IABS", 4), ("IADD", 4), ("IADD3", 4), ("IDP", 5), ("IMAD", 3),
  ("IMNMX", 4), ("ISETP", 5), ("LEA", 3), ("LOP3", 4), ("MOV", 4), ("P2R", 5), ("PLOP3", 5),
  ("PRMT", 4), ("SEL", 5), ("SGXT", 4), ("SHF", 4), ("UFLO", 6), ("UFMUL", 9), ("UFSEL", 4),
  ("UFSETP", 4), ("UI2F", 4), ("UIADD3", 4), ("UIMAD", 4), ("UIMNMX", 4), ("UISETP", 4),
  ("ULEA", 4), ("ULOP3", 4), ("UMOV", 2), ("UPLOP3", 30), ("UPOPC", 6), ("UPRMT", 13),
  ("USEL", 4), ("USHF", 4), ("UVIMNMX", 4), ("VIADD", 5), ("VIMNMX", 5), ("VOTEU", 2)]

/-- Stall floors on the RTX PRO 6000 Server Edition (`blackwell-isa-stallfix`), by producer: its
latency, and for predicate producers the latency of a guard read. -/
private def stallfix : List (String × Option Nat × Nat) := [
  ("IADD3", some 4, 0), ("UIADD3", some 4, 0), ("R2UR", some 4, 0), ("ISETP", some 5, 13),
  ("UISETP", none, 10)]

/-- On sm_120 a stall count of 0 interlocks. -/
def stallZeroInterlocks : Fact Bool :=
  pub true "blackwell-isa-stallfix" "sm120.json"
    "RTX PRO 6000 Blackwell Server Edition" ""

private def hopperOnly (opcode : String) : Bool :=
  ["HGMMA", "QGMMA", "IGMMA", "WARPGROUP", "USETMAXREG"].contains opcode

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
  let base := match stallfix.find? (·.1 == r.opcode) with
    | some (_, lat, guard) =>
      { base with
        value := { base.value with latency := lat.getD base.value.latency, guardLatency := guard }
        kind := .published
        source := { ref := "blackwell-isa-stallfix", loc := "sm120.json, stallfix" }
        measuredOn := some "RTX PRO 6000 Blackwell Server Edition" }
    | none => base
  match cc120.lookup r.opcode with
  | some (thr, row) =>
    { r with timing := { base with
        value := { base.value with interval := 128 / thr }
        source := { ref := "cuda-best-practices", loc := s!"§12.1.1 Table 5, compute capability 12.0: {row}, {thr} results per clock per SM" } } }
  | none => { r with timing := base }

/-- Opcodes Blackwell (`sm_100a`, `sm_120a`) has and `sm_90a` does not, each in the class of its
`sm_90a` counterpart (the uniform datapath's forms of vector instructions; LDCU, a constant load
into a uniform register, as LDC; QMMA and OMMA, the FP8/FP6 and FP4 warp MMAs, as HMMA), and
issue-only control. -/
private def analogues : List (String × String) :=
  [("LDCU", "LDC"), ("CS2UR", "S2UR"), ("UI2F", "I2F"), ("IADD", "IADD3"), ("UFMUL", "FMUL"),
   ("UFSEL", "USEL"), ("UFSETP", "UISETP"), ("QMMA", "HMMA"), ("OMMA", "HMMA"),
   ("UVIMNMX", "UIMNMX"), ("ENDCOLLECTIVE", "NOP"), ("UP2UR", "ULOP3"),
   ("FADD2", "FADD"), ("FMUL2", "FMUL"), ("FFMA2", "FFMA")]

def added : Array OpRule :=
  (analogues.filterMap fun (new, old) => (Hopper.ops.find? (·.opcode == old)).map fun r =>
    { r with opcode := new, mods := #[],
             timing := { r.timing with kind := .inferred,
                                       note := "" } }).toArray

def ops : Array OpRule :=
  ((Hopper.ops.filter fun r => !hopperOnly r.opcode) ++ added).map retime

def l1SmemBytes : Fact Nat :=
  doc 131072 "cuda-guide" "§5.1.3 Table 35: unified data cache 128 KB per SM, compute capability 12.0"

def smemCarveoutsKB : Fact (List Nat) :=
  doc [0, 8, 16, 32, 64, 100] "cuda-guide" "§5.1.3 Table 35, compute capability 12.0"

/-- The clock nanosecond values are converted at: the workstation edition's boost. -/
def assumedMHz : Nat := 2617

def nsToCycles (ns : Nat) : Nat := (ns * assumedMHz + 500) / 1000

def smemLatency : Fact Nat :=
  pub (nsToCycles 12) "chipsandcheese-blackwell" "SM Memory Subsystem: shared memory 11.96 ns"
    "RTX PRO 6000 Blackwell" ""

def l1HitLatency : Fact Nat :=
  pub (nsToCycles 15) "chipsandcheese-blackwell" "SM Memory Subsystem: L1 vector hit 15.20 ns"
    "RTX PRO 6000 Blackwell" ""

def l2HitLatency : Fact Nat :=
  pub (nsToCycles 130) "chipsandcheese-blackwell" "GPU-Wide Memory Subsystem: L2 130.02 ns (OpenCL)"
    "RTX PRO 6000 Blackwell" ""

def dramLatency : Fact Nat :=
  pub (nsToCycles 329) "chipsandcheese-blackwell" "GPU-Wide Memory Subsystem: DRAM 329 ns"
    "RTX PRO 6000 Blackwell" ""

def mmaSyncLatency : Fact Nat :=
  pub 35 "yu2026" "§III-A, Table III: mma m16n8k16 f32.f16.f16 dependent latency 34.5 cycles"
    "RTX 5090" ""

end NV.Arch.BlackwellRTX
