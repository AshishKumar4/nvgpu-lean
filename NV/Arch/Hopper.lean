import NV.Arch.Table

/-!
# Hopper (GH100, `sm_90a`): the facts H100 and H200 share

H100 SXM5 and H200 SXM are the same GH100 die with the same enabled SMs (132 SMs, 66 TPCs); they
differ in the memory system and its clocks. This file holds what belongs to the SM and the ISA:
NVIDIA's compute capability 9.0 limits, the opcode classes' issue timing, the tensor cores, and
the SM-local costs measured on an H200 (shared memory, barriers, fences). A measured fact is
used for H100 through `onH200`, which labels it with the part it was measured on.

Cycle counts are of the SM clock. Sources are keys of `docs/sources.md`.
-/

namespace NV.Arch.Hopper

open NV NV.Core


/-! ## Structure and limits -/

def target : Fact String :=
  doc "sm_90a" "ptx-isa-9.4" "§5 (Target ISA notes): sm_90a, the architecture-specific Hopper target"
    ""

def features : Fact Features :=
  doc { cpAsync := true, tma := true, wgmma := true, clusters := true } "cuda-guide"
    "compute capability 9.0 feature table: asynchronous copies, the tensor memory accelerator,
    warpgroup MMA (sm_90a), thread block clusters and distributed shared memory"

def smsps : Fact Nat :=
  doc 4 "hopper-whitepaper" "H100 SM key features: four SM sub-partitions (processing blocks)"
    ""

def maxThreadsPerSM : Fact Nat := doc 2048 "cuda-guide" "§5.1.3 Table 33, compute capability 9.0"
def maxCtasPerSM : Fact Nat := doc 32 "cuda-guide" "§5.1.3 Table 33, compute capability 9.0"
def maxWarpsPerSM : Fact Nat := doc 64 "cuda-guide" "§5.1.3 Table 33, compute capability 9.0"
def regsPerSM : Fact Nat := doc 65536 "cuda-guide" "§5.1.3 Table 34: 64 K 32-bit registers per SM"

def regAllocUnit : Fact Nat :=
  doc 256 "cuda-occupancy" "register allocation unit size, compute capability 9.0, warp granularity"

def smemPerSM : Fact Nat := doc 233472 "cuda-guide" "§5.1.3 Table 34: 228 KB shared memory per SM"

def smemPerCtaMax : Fact Nat :=
  doc 232448 "cuda-guide" "§5.1.3 Table 34: 227 KB per thread block"
    ""

def smemReservedPerCta : Fact Nat :=
  doc 1024 "hopper-tuning" "§1.4.2.4: CUDA reserves 1 KB of shared memory per thread block"
    ""

def maxClusterCtas : Fact Nat :=
  doc 16 "hopper-tuning" "§1.4.1.3: nonportable cluster size 16 by opting in; portable maximum 8"

/-! ## Issue -/

def issuePolicy : Fact IssuePolicy :=
  inferred .greedyThenLeastRecent "meas-gh200" ""
    ""

def readRelease : Fact Nat :=
  inferred 11 "huerta2025" "Table 2: WAR column, 32-bit global load with a regular address"
    "" (some "Ampere")

def barrierArrivalCycles : Fact Nat :=
  inferred 2 "meas-h200" "" ""

def barrierCycles : Fact Nat :=
  inferred 12 "meas-h200" ""
    ""

/-! ## Register file and front end -/

def regFile : RegFileParams where
  banks := inferred 2 "huerta2025" "§5.3: the registers of a sub-core are in two banks"
    "" (some "Ampere")
  readPortsPerBank := inferred 1 "huerta2025" "§5.3: one 1024-bit read port per bank"
    "" (some "Ampere")
  operandReadCycles := inferred 3 "huerta2025"
    "§5.3: fixed-latency instructions spend three cycles reading source operands"
    ""
    (some "Ampere")
  reuseEntriesPerBank := inferred 1 "huerta2025" "§5.3.1: one register-file-cache entry per bank"
    "" (some "Ampere")
  reuseSlotsPerEntry := inferred 3 "huerta2025" "§5.3.1: each entry stores three 1024-bit values"
    "" (some "Ampere")

def frontEnd : FrontEndParams where
  instBytes := encoded 16 ""
    ""
  fetchPerCycle := inferred 1 "huerta2025" "§5.2: each sub-core fetches and decodes one
    instruction per cycle" ""
    (some "Ampere")
  instBufferEntries := inferred 3 "huerta2025" "§5.2: an instruction buffer of three entries per
    warp" ""
    (some "Ampere")
  codeCaches := inferred [(24576, 10), (49152, 16), (147456, 16)] "meas-h200"
    ""
    ""
  codePrefetchLines := inferred 6 "meas-h200" ""
    ""
  codeLaunchBytes := inferred 65536 "meas-gh200" ""
    ""
    (some "GH200")
  takenBranchCycles := inferred 4 "meas-gh200" ""
    ""
  forwardBranchCycles := inferred 17 "meas-gh200" ""
    ""
  stallZeroInterlocks := inferred false "huerta2025" "§4: the stall counter is the only wait a
    fixed-latency dependence has" "" (some "Ampere")
  rotateCtaWarps := measOn true ""
    ""
  constCacheBytes := doc 8192 "cuda-guide"
    "§5.1.3 Table 34: cache working set per SM for constant memory, 8 KB"
    ""
  constLineBytes := inferred 64 "jia2018" "Table 3.1: constant L1 cache line, 64 B"
    "" (some "V100")
  constMissCycles := inferred 79 "huerta2025" "§5.4: the issue was delayed 79 cycles, a miss"
    "" (some "Ampere")
  constMissSwitchCycles := inferred 4 "huerta2025" "§5.1.1: after four cycles of an unserved
    constant miss the scheduler switches warps" "" (some "Ampere")

/-! ## Opcode classes

Intervals are the documented results per clock per SM (`cuda-guide-12.6` §5.4.1 Table 4, compute
capability 9.0) over four sub-partitions of 32 lanes, unless the note says otherwise. A
latency is from issue to the earliest dependent issue. -/

private def op (opcode : String) (mods : Array String) (t : Fact OpTiming) : OpRule :=
  { opcode, mods, timing := t }

private def tm (pipe : Pipe) (interval latency : Nat) (readRelease := 0) : OpTiming :=
  { pipe, interval, latency, readRelease }

/-- A memory instruction's timing on the LSU pipe: issued one per cycle, its throughput is the
memory system's (a sub-core issues up to five consecutive memory instructions one per cycle, its
address calculation then takes one every four cycles and the shared structures one every two:
huerta2025 §5.4, measured on Ampere; mem.addressInterval, mem.mioQueueDepth, mem.lsuInterval). -/
private def tmMem (latency : Nat) (readRelease := 0) : OpTiming := tm .lsu 1 latency readRelease

/-- A bulk or tensor copy: issued on the uniform datapath, it hands its operands to the SM's
bulk-copy unit, whose rate (`tma.issueBytesPerSec`) and latency (`tma.copyDram`, `copyL2`) the
memory system times; the issue slot is a placeholder. -/
private def bulk (opcode : String) : OpRule :=
  op opcode #[] (inferred (tm .uniform 1 0) "analogy" ""
    "")

/-- Convergence and memory-ordering instructions: an issue slot here; what they wait for (a
fence's drain, `sync.fence*`) is the memory system's. -/
private def issueOnly (opcode : String) : OpRule :=
  op opcode #[] (inferred (tm .cbu 1 0) "analogy" ""
    "")

private def thr (n : Nat) : String := s!"§5.4.1 Table 4: {n} results per clock per SM"


def ops : Array OpRule := #[
  op "FFMA" #[] (doc (tm .fmaLite 1 4) "cuda-guide-12.6" (thr 128)
    ""),
  op "FADD" #[] (doc (tm .fmaLite 1 4) "cuda-guide-12.6" (thr 128)),
  op "FMUL" #[] (doc (tm .fmaLite 1 4) "cuda-guide-12.6" (thr 128)),
  op "IMAD" #["HI"] (encoded (tm .fmaHeavy 4 4) ""),
  op "IMAD" #["WIDE"] (inferred { tm .fmaHeavy 4 3 with lateRead := 2 } "analogy" ""
    ""),
  op "IMAD" #[] (doc (tm .fmaHeavy 2 4) "cuda-guide-12.6" (thr 64)
    ""),
  op "VIADD" #[] (inferred (tm .fmaHeavy 2 4) "meas-h200" ""
    ""),
  op "IADD3" #[] (doc (tm .alu 2 4) "cuda-guide-12.6" (thr 64)
    ""),
  op "LOP3" #[] (doc (tm .alu 2 4) "cuda-guide-12.6" (thr 64)),
  op "SHF" #[] (doc (tm .alu 2 4) "cuda-guide-12.6" (thr 64)),
  op "PRMT" #[] (encoded (tm .alu 2 4) ""),
  op "LEA" #[] (encoded (tm .alu 2 4) "" ""),
  op "ISETP" #[] (doc (tm .alu 2 4) "cuda-guide-12.6" (thr 64) ""),
  op "FMNMX" #[] (doc (tm .alu 2 4) "cuda-guide-12.6" (thr 64)
    ""),
  op "MOV" #[] (encoded (tm .alu 2 4) ""),
  op "F2FP" #[] (meas (tm .alu 2 4) ""
    ""),
  op "HFMA2" #["MMA"] (encoded (tm .tensor 2 10) ""
    ""),
  op "HFMA2" #[] (encoded (tm .fp16 2 5) ""
    ""),
  op "HMUL2" #[] (encoded (tm .fp16 2 5) ""),
  op "HADD2" #[] (encoded (tm .fp16 2 5) ""),
  op "DFMA" #[] (doc (tm .fp64 2 7) "cuda-guide-12.6" (thr 64)
    ""),
  op "F2I" #[] (inferred (tm .xu 8 13) "cuda-guide-12.6" (thr 16)
    ""),
  op "I2F" #[] (inferred (tm .xu 8 13) "cuda-guide-12.6" (thr 16) ""),
  op "F2F" #[] (inferred (tm .xu 8 13) "cuda-guide-12.6" (thr 16) ""),
  op "FRND" #[] (inferred (tm .xu 8 13) "cuda-guide-12.6" (thr 16) ""),
  op "I2I" #[] (inferred (tm .alu 2 4) "cuda-guide-12.6" (thr 64)
    ""),
  op "HSETP2" #[] (inferred (tm .fp16 2 5) "cuda-guide-12.6" (thr 256) ""),
  op "HSET2" #[] (inferred (tm .fp16 2 5) "cuda-guide-12.6" (thr 256)
    ""),
  op "I2IP" #[] (inferred (tm .alu 2 4) "cuda-guide-12.6" (thr 64) ""),
  op "F2IP" #[] (inferred (tm .alu 2 4) "cuda-guide-12.6" (thr 64)
    ""),
  op "IDP" #[] (inferred (tm .fmaHeavy 2 4) "analogy" ""
    ""),
  op "VABSDIFF4" #[] (doc (tm .alu 2 4) "cuda-guide-12.6" (thr 64) ""),
  op "VABSDIFF" #[] (doc (tm .alu 2 4) "cuda-guide-12.6" (thr 64) ""),
  op "DADD" #[] (encoded (tm .fp64 2 6) ""
    ""),
  op "DMUL" #[] (encoded (tm .fp64 2 6) ""
    ""),
  op "HMNMX2" #[] (inferred (tm .fp16 2 5) "cuda-guide-12.6" (thr 256)
    ""),
  op "FCHK" #[] (inferred (tm .alu 2 4) "analogy" ""
    ""),
  op "MUFU" #["F16"] (encoded (tm .xu 8 10) ""
    ""),
  op "MUFU" #[] (meas (tm .xu 8 17) ""
    ""),
  op "HMMA" #["TF32"] (encoded (tm .tensor 6 16) ""
    ""),
  op "HMMA" #[] (encoded (tm .tensor 6 23) ""
    ""),
  op "S2R" #[] (inferred (tm .cbu 1 20) "analogy" ""
    ""),
  op "S2UR" #[] (inferred (tm .cbu 1 36) "meas-gh200" "" ""),
  op "CS2R" #[] (encoded (tm .alu 2 4) ""),
  op "LDC" #[] (inferred (tm .lsu 2 26 10) "huerta2025" "Table 2: 32-bit constant load, immediate
    address: RAW 26, WAR 10"
    "" (some "Ampere")),
  op "UIMAD" #[] (inferred (tm .uniform 2 4) "meas-h200" ""
    ""),
  op "UIADD3" #[] (inferred (tm .uniform 2 4) "meas-h200" "" ""),
  op "ULOP3" #[] (inferred (tm .uniform 2 4) "meas-h200" "" ""),
  op "USHF" #[] (inferred (tm .uniform 2 4) "meas-h200" "" ""),
  op "ULEA" #[] (encoded (tm .uniform 2 4) ""
    ""),
  op "UMOV" #[] (encoded (tm .uniform 2 2) ""
    ""),
  op "UISETP" #[] (encoded (tm .uniform 2 4) ""
    ""),
  op "UPRMT" #[] (inferred (tm .uniform 2 4) "meas-h200" "" ""),
  op "ULDC" #[] (encoded (tm .uniform 1 2) ""
    ""),
  op "LDG" #[] (inferred (tmMem 654 11) "meas-h200" ""
    ""),
  op "STG" #["128"] (inferred (tmMem 0 20) "huerta2025" "Table 2: 128-bit global store, WAR 20"
    "" (some "Ampere")),
  op "STG" #["64"] (inferred (tmMem 0 16) "huerta2025" "Table 2: 64-bit global store, WAR 16"
    "" (some "Ampere")),
  op "STG" #[] (inferred (tmMem 0 14) "huerta2025" "Table 2: 32-bit global store, WAR 14"
    "" (some "Ampere")),
  op "LDS" #["128"] (inferred (tmMem 25 9) "meas-h200" ""
    ""),
  op "LDS" #[] (inferred (tmMem 23 9) "meas-h200" ""
    ""),
  op "LDGSTS" #[] (inferred (tmMem 39 13) "huerta2025" "Table 2: LDGSTS, any width: WAR 13,
    RAW/WAW 39" "" (some "Ampere")),
  op "LDSM" #[] (inferred (tmMem 23 9) "meas-h200" ""
    ""),
  -- Texture and surface access: texel fetches through the memory pipe, priced as global loads
  -- and stores (inferred).
  op "TEX" #[] (inferred (tm .lsu 4 654 11) "meas-h200" "" ""),
  op "TLD" #[] (inferred (tm .lsu 4 654 11) "meas-h200" "" ""),
  op "TLD4" #[] (inferred (tm .lsu 4 654 11) "meas-h200" "" ""),
  op "TXD" #[] (inferred (tm .lsu 4 654 11) "meas-h200" ""
    ""),
  op "LDGMC" #[] (inferred (tmMem 654 11) "meas-h200" ""
    ""),
  op "QSPC" #[] (inferred (tm .lsu 4 23) "analogy" ""
    ""),
  op "TXQ" #[] (inferred (tm .lsu 4 23 9) "meas-h200" "" ""),
  op "SULD" #[] (inferred (tm .lsu 4 654 11) "meas-h200" "" ""),
  op "SUST" #[] (inferred (tm .lsu 4 0 14) "huerta2025" "Table 2: 32-bit global store, WAR 14" "" (some "Ampere")),
  op "SURED" #[] (inferred (tm .lsu 4 0 14) "huerta2025" "Table 2: 32-bit global store, WAR 14" "" (some "Ampere")),
  op "SUQUERY" #[] (inferred (tm .lsu 4 23 9) "meas-h200" "" ""),
  -- st.async / red.async: a store of register values handed to the async unit, through the
  -- memory pipe; STG's class.
  op "STAS" #[] (inferred (tm .lsu 4 0 14) "huerta2025" "Table 2: 32-bit global store, WAR 14" "" (some "Ampere")),
  op "REDAS" #[] (inferred (tm .lsu 4 0 14) "huerta2025" "Table 2: 32-bit global store, WAR 14" "" (some "Ampere")),
  op "MOVM" #[] (encoded (tm .lsu 4 6 6) ""
    ""),
  op "STSM" #[] (inferred (tmMem 0 12) "huerta2025" "Table 2: 32-bit shared store, WAR 12"
    "" (some "Ampere")),
  op "STS" #["128"] (inferred (tmMem 0 18) "huerta2025" "Table 2: 128-bit shared store, WAR 18"
    "" (some "Ampere")),
  op "STS" #["64"] (inferred (tmMem 0 14) "huerta2025" "Table 2: 64-bit shared store, WAR 14"
    "" (some "Ampere")),
  op "STS" #[] (inferred (tmMem 0 12) "huerta2025" "Table 2: 32-bit shared store, WAR 12"
    "" (some "Ampere")),
  op "VIADDMNMX" #[] (inferred (tm .alu 2 4) "cuda-guide-12.6" (thr 64)
    ""),
  op "PLOP3" #[] (inferred (tm .alu 2 2) "cuda-guide-12.6" (thr 64)
    ""),
  op "VOTE" #[] (doc (tm .alu 2 4) "cuda-guide-12.6" (thr 64)
    ""),
  op "VOTEU" #[] (inferred (tm .uniform 2 1) "cuda-guide-12.6" (thr 64)
    ""),
  op "ELECT" #[] (encoded (tm .uniform 2 4) ""
    ""),
  op "R2UR" #[] (encoded (tm .uniform 2 8) ""
    ""),
  bulk "UTMALDG", bulk "UTMASTG", bulk "UBLKCP", bulk "UBLKRED", bulk "UTMAREDG", bulk "UTMAPF", bulk "UTMACCTL",
  bulk "UTMACMDFLUSH", bulk "UBLKPF",
  op "SYNCS" #[] (inferred (tm .lsu 1 0) "meas-h200" ""
    ""),
  op "ARRIVES" #[] (inferred (tm .lsu 1 0) "meas-h200" ""
    ""),
  op "LDGDEPBAR" #[] (inferred (tm .cbu 1 0) "analogy" ""
    ""),
  op "WARPGROUP" #[] (inferred (tm .cbu 1 0) "analogy" ""
    ""),
  op "FENCE" #[] (meas (tm .cbu 1 0) ""
    ""),
  op "HGMMA" #[] (inferred (tm .gmma 1 0) "luo2024" "Tables VIII–X"
    "" (some "H800 PCIe")),
  op "QGMMA" #[] (inferred (tm .gmma 1 0) "luo2024" "Tables VIII–X" ""
    (some "H800 PCIe")),
  op "BGMMA" #[] (inferred (tm .gmma 1 0) "luo2024" "Tables VIII–X" ""
    (some "H800 PCIe")),
  op "IGMMA" #[] (inferred (tm .gmma 1 0) "luo2024" "Tables VIII–X" ""
    (some "H800 PCIe")),
  op "FSETP" #[] (doc (tm .alu 2 4) "cuda-guide-12.6" (thr 64) ""),
  op "FSEL" #[] (encoded (tm .alu 2 4) ""
    ""),
  op "SEL" #[] (encoded (tm .alu 2 4) ""
    ""),
  op "IMNMX" #[] (doc (tm .alu 2 4) "cuda-guide-12.6" (thr 64) ""),
  op "VIMNMX" #[] (doc (tm .alu 2 4) "cuda-guide-12.6" (thr 64)
    ""),
  op "VIMNMX3" #[] (inferred (tm .alu 2 4) "cuda-guide-12.6" (thr 64)
    ""),
  op "SGXT" #[] (inferred (tm .alu 2 4) "analogy" ""
    ""),
  op "USGXT" #[] (inferred (tm .uniform 2 4) "analogy" "" ""),
  op "BMSK" #[] (inferred (tm .alu 2 4) "analogy" ""
    ""),
  op "IABS" #[] (inferred (tm .alu 2 4) "cuda-guide-12.6" (thr 64)
    ""),
  op "I2FP" #[] (encoded (tm .alu 2 4) ""
    ""),
  op "P2R" #[] (inferred (tm .alu 2 4) "analogy" "" ""),
  op "R2P" #[] (inferred (tm .alu 2 4) "analogy" "" ""),
  op "UFLO" #[] (inferred (tm .uniform 2 4) "analogy" ""
    ""),
  op "UPOPC" #[] (inferred (tm .uniform 2 4) "analogy" ""
    ""),
  op "USEL" #[] (encoded (tm .uniform 2 4) "" ""),
  op "UIMNMX" #[] (inferred (tm .uniform 2 4) "analogy" "" ""),
  op "POPC" #[] (inferred (tm .xu 8 17) "cuda-guide-12.6" (thr 16)
    ""),
  op "FLO" #[] (inferred (tm .xu 8 17) "cuda-guide-12.6" (thr 16) ""),
  op "BREV" #[] (inferred (tm .xu 8 17) "cuda-guide-12.6" (thr 16) ""),
  op "SHFL" #[] (inferred (tm .lsu 4 23) "cuda-guide-12.6" (thr 32)
    ""),
  op "MATCH" #[] (inferred (tm .lsu 8 23) "analogy" ""
    ""),
  op "REDUX" #[] (inferred (tm .lsu 8 23) "cuda-guide-12.6" (thr 16)
    ""),
  op "LDL" #[] (inferred (tmMem 32) "luo2025" "Table 3: L1 32.0"
    "" (some "H800 PCIe")),
  op "STL" #[] (inferred (tmMem 0 14) "huerta2025" "Table 2: 32-bit store, WAR 14"
    "" (some "Ampere")),
  op "ATOM" #[] (inferred (tmMem 654 11) "meas-h200" ""
    ""),
  op "ATOMG" #[] (inferred (tmMem 654 11) "meas-h200" ""
    ""),
  op "REDG" #[] (inferred (tmMem 0 14) "huerta2025" "Table 2: 32-bit global store, WAR 14"
    "" (some "Ampere")),
  op "ATOMS" #[] (inferred (tmMem 23 9) "meas-h200" ""
    ""),
  issueOnly "BREAK", issueOnly "BMOV", issueOnly "CALL", issueOnly "RET", issueOnly "USETMAXREG",
  op "LEPC" #[] (inferred (tm .alu 2 4) "analogy" ""
    ""),
  issueOnly "YIELD", issueOnly "NANOSLEEP",
  op "BAR" #[] (meas (tm .cbu 1 0) ""),
  op "BRX" #[] (encoded (tm .cbu 1 0) "" ""),
  op "BRXU" #[] (inferred (tm .cbu 1 0) "analogy" ""
    ""),
  issueOnly "BSSY", issueOnly "BSYNC", issueOnly "WARPSYNC", issueOnly "MEMBAR",
  issueOnly "ERRBAR", issueOnly "CGAERRBAR", issueOnly "CCTL",
  op "BRA" #[] (encoded (tm .cbu 1 0) ""),
  op "EXIT" #[] (encoded (tm .cbu 1 0) ""),
  op "NOP" #[] (encoded (tm .cbu 1 0) ""),
  op "DEPBAR" #[] (encoded (tm .cbu 1 0) ""),
  -- Forms whose class is a sibling's; the hazard check (`Counters.hazards`) flags any whose
  -- latency exceeds a compiled kernel's distance.
  op "DSETP" #[] (encoded (tm .fp64 2 6) ""
    ""),
  op "FSET" #[] (inferred (tm .alu 2 4) "analogy" "" ""),
  op "B2R" #[] (inferred (tm .cbu 1 20) "analogy" ""
    ""),
  op "UBREV" #[] (inferred (tm .uniform 2 4) "analogy" "" ""),
  op "UBMSK" #[] (inferred (tm .uniform 2 4) "analogy" "" ""),
  op "UFADD" #[] (inferred (tm .uniform 2 4) "analogy" "" ""),
  op "UFFMA" #[] (inferred (tm .uniform 2 4) "analogy" "" ""),
  op "UI2FP" #[] (inferred (tm .uniform 2 4) "analogy" "" ""),
  op "BPT" #[] (inferred (tm .cbu 1 0) "analogy" "" ""),
  -- The FP64 MMA at the documented FP64 tensor rate (`fmaPerCycleF64`, 128 per SM per cycle, 32
  -- per sub-partition): a shape's FMAs over 32 is its issue interval. Latency as HMMA's. The
  -- shapes are DMMA.884 (sm_80) and DMMA.16x8x4/8/16 (sm_90a).
  op "DMMA" #["884"] (inferred (tm .tensor 8 23) "hopper-whitepaper" "Table 1, Table 3"
    ""),
  op "DMMA" #["16x8x4"] (inferred (tm .tensor 16 23) "hopper-whitepaper" "Table 1, Table 3"
    ""),
  op "DMMA" #["16x8x8"] (inferred (tm .tensor 32 23) "hopper-whitepaper" "Table 1, Table 3"
    ""),
  op "DMMA" #["16x8x16"] (inferred (tm .tensor 64 23) "hopper-whitepaper" "Table 1, Table 3"
    ""),
  -- Integer and single-bit MMAs by shape.
  op "BMMA" #["88128"] (encoded (tm .tensor 6 12) ""
    ""),
  op "BMMA" #["168128"] (encoded (tm .tensor 6 15) ""
    ""),
  op "BMMA" #["168256"] (encoded (tm .tensor 6 15) ""
    ""),
  op "IMMA" #["8816"] (encoded (tm .tensor 6 12) ""
    ""),
  op "IMMA" #["8832"] (encoded (tm .tensor 6 12) ""
    ""),
  op "IMMA" #[] (encoded (tm .tensor 6 15) ""
    ""),
  op "DMMA" #[] (inferred (tm .tensor 6 23) "analogy" "" ""),
  -- The cluster barrier's arrive and wait: issue only; the barrier is the engine's (Effect.sync).
  issueOnly "UCGABAR_ARV", issueOnly "UCGABAR_WAIT", issueOnly "ACQBULK", issueOnly "PREEXIT",
  op "UPLOP3" #[] (inferred (tm .uniform 2 4) "analogy" ""
    ""),
  -- Generic loads and stores (an address any state space may hold): the global forms' classes,
  -- the memory system resolving the space.
  op "LD" #[] (inferred (tmMem 654 11) "meas-h200" ""
    ""),
  op "ST" #[] (inferred (tmMem 0 14) "huerta2025" "Table 2: 32-bit global store, WAR 14"
    "" (some "Ampere"))
]

/-! ## Shared memory and the SM's L1 -/

def sectorBytes : Fact Nat :=
  doc 32 "ncu-guide" "Kernel Profiling Guide, memory terminology: a sector is an aligned 32-byte
    chunk; a cache line is four sectors"

def lineBytes : Fact Nat :=
  doc 128 "ncu-guide" "Kernel Profiling Guide, memory terminology: an L1 or L2 cache line is four
    sectors, 128 bytes"

def l1SmemBytes : Fact Nat :=
  doc 262144 "hopper-tuning" "§1.4.2.4: combined L1, texture cache and shared memory, 256 KB per SM"

def smemCarveoutsKB : Fact (List Nat) :=
  doc [0, 8, 16, 32, 64, 100, 132, 164, 196, 228] "cuda-guide"
    "§5.1.3 Table 35, compute capability 9.0: SMEM capacity sizes (KB)"
    ""

def smemAtomicInterval : Fact Nat :=
  inferred 2 "chipsandcheese-mi300x" "Local Memory Atomics Latency: 36.32 ns (H100 PCIe)"
    ""
    (some "H100 PCIe")

def smemBanks : Fact Nat := doc 32 "cuda-guide" "§5.1.3 Table 34: 32 shared memory banks"

def smemBankBytes : Fact Nat :=
  doc 4 "cuda-guide" "shared memory, compute capability 5.x and later: successive 32-bit words map
    to successive banks"

def smemLatency : Fact Nat :=
  inferred 23 "meas-h200" ""
    ""

def l1HitLatency : Fact Nat :=
  pub 32 "luo2025" "Table 3: L1 cache 32.0" "H800 PCIe"
    ""

def dsmemLatency : Fact Nat :=
  inferred 173 "meas-h200" ""
    ""

def mioQueueDepth : Fact Nat :=
  inferred 4 "huerta2025" "§5.4: the memory queue of a sub-core, estimated at 4 entries"
    "" (some "Ampere")

def lsuInterval : Fact Nat :=
  pub 1 "luo2024" "Table V: shared memory 127.4–127.9, L1 125.8 bytes/clk/SM with 4-byte accesses"
    "H800 PCIe" ""

def addressInterval : Fact Nat :=
  inferred 4 "huerta2025" "§5.4: address calculation in each sub-core, one instruction every four
    cycles" "" (some "Ampere")

def storeRegCycles : Fact Nat :=
  inferred 3 "meas-gh200" ""
    ""
    (some "GH200")

def l1BytesPerCycle : Fact Nat :=
  pub 128 "luo2025" "Table 5: L1 125.8 (FP32), 124.1 (FP32.v4); shared 127.4, 127.9 byte/clk/SM"
    "H800 PCIe" ""

def l1TagSectorsPerCycle : Fact Nat :=
  inferred 4 "luo2025" "Table 5: 128 byte/clk/SM from L1"
    "" (some "H800 PCIe")

def l1Ways : Fact Nat :=
  { value := 64, kind := .simulator,
    source := { ref := "accelsim-h100", loc := "gpgpusim.config:100, dl1 S:4:128:64" },
    note := "" }

def l1Mshrs : Fact Nat :=
  { value := 512, kind := .simulator,
    source := { ref := "accelsim-h100", loc := "gpgpusim.config:100, dl1 A:512:64" },
    note := "" }

def l1StoreInvalidates : Fact Bool :=
  { value := false, kind := .simulator,
    source := { ref := "accelsim-h100", loc := "gpgpusim.config:100, dl1 L:T:m:L" },
    note := "" }

def xbarBytesPerCycle : Fact Nat :=
  inferred 107 "meas-h200" ""
    ""

def xbarWriteBytesPerCycle : Fact Nat :=
  inferred 32 "meas-gh200" ""
    ""
    (some "GH200")

def xbarRedBytesPerCycle : Fact Nat :=
  inferred 25 "meas-gh200" ""
    "" (some "GH200")

def dsmemBytesPerCycle : Fact Nat :=
  inferred 16 "luo2024" "§IV-E, Fig. 8: SM-to-SM throughput peaks near 3.27 TB/s at cluster size 2"
    "" (some "H800 PCIe")

/-! ## L2 (the latencies are the die's: the same GH100 crossbar and slices on both parts) -/

def l2Partitions : Fact Nat :=
  pub 2 "luo2025" "§4.1" "H800 PCIe"
    ""

def l2NearHitLatency : Fact Nat :=
  inferred 274 "meas-h200" ""
    ""

def l2FarCopiesNear : Fact Bool :=
  inferred true "a100-whitepaper" "p. 35: each L2 partition localizes and caches data for memory
    accesses from SMs in the GPCs directly connected to the partition"
    "" (some "A100")

def l2Ways : Fact Nat :=
  { value := 20, kind := .simulator,
    source := { ref := "accelsim-h100", loc := "gpgpusim.config:120, dl2 S:256:128:20" },
    note := "" }

def l2FarBytesPerCycle : Fact Nat :=
  inferred 1090 "chipsandcheese-h100" "L2 bandwidth: 5563 GB/s with 16 MB (near partition),
    3827 GB/s with 40 MB (both partitions)"
    "" (some "H100 PCIe")

def l2SliceBytesPerCycle : Fact Nat :=
  inferred 52 "meas-gh200" ""
    "" (some "GH200")

def l2SliceWriteBytesPerCycle : Fact Nat :=
  inferred 21 "meas-gh200" ""
    ""
    (some "GH200")

def l2ClockRefMHz : Fact Nat :=
  inferred 1830 "meas-gh200" ""
    "" (some "GH200")

def l2LatencyFixedPermille : Fact Nat :=
  inferred 489 "meas-gh200" ""
    ""
    (some "GH200")

def dramTurnaroundPs : Fact Nat :=
  inferred 1000 "meas-h100" ""
    ""

def storeCreditBytes : Fact Nat :=
  inferred 16384 "meas-gh200" ""
    ""

def l2LatencyRefMHz : Fact Nat :=
  meas 1980 ""
    ""

def l2FarHitLatency : Fact Nat :=
  inferred 430 "luo2025" "Table 4: L2 cache near hit 258.0, far hit 414.1"
    "" (some "H800 PCIe")

/-! ## Tensor cores. H100 SXM5's dense peaks at the clock NVIDIA quotes them at (1830 MHz) over
132 SMs; the H200's are the same (its product page quotes the same sparse peaks). -/

def fmaPerCycle16 : Fact Nat :=
  doc 2048 "hopper-whitepaper" "Table 3: peak FP16/BF16 tensor 989.4 TFLOPS dense at 1830 MHz"
    ""

def fmaPerCycle8 : Fact Nat :=
  doc 4096 "hopper-whitepaper" "Table 3: peak FP8 tensor 1978.9 TFLOPS dense at 1830 MHz"

def fmaPerCycleTF32 : Fact Nat :=
  doc 1024 "hopper-whitepaper" "Table 3: peak TF32 tensor 494.7 TFLOPS dense at 1830 MHz"

def fmaPerCycleF64 : Fact Nat :=
  doc 128 "hopper-whitepaper" "Table 1: peak FP64 tensor 66.9 TFLOPS; Table 3: at 1980 MHz"
    ""

def wgmmaLatency : Fact (List (Nat × Nat)) :=
  pub [(8, 18), (16, 20), (32, 24), (64, 32), (128, 64), (256, 128)] "luo2024"
    "Table X: wgmma m64nNk16 f32.f16.f16, dense, SS" "H800 PCIe"
    ""

def wgmmaLatencyRS : Fact (List (Nat × Nat)) :=
  pub [(8, 13), (16, 13), (32, 16), (64, 32), (128, 64), (256, 128)] "luo2024"
    "Table X: wgmma m64nNk16 f32.f16.f16, dense, RS" "H800 PCIe"
    ""

def wgmmaOperandDataPermille : Fact Nat :=
  inferred 1000 "meas-gh200" ""
    "" (some "GH200")

def wgmmaIssueSharePermille : Fact Nat :=
  inferred 500 "meas-gh200" ""
    "" (some "GH200")

def wgmmaPipeSharePermille : Fact Nat :=
  inferred 200 "meas-gh200" ""
    "" (some "GH200")

def mmaSyncLatency : Fact Nat :=
  meas 24 ""
    ""

/-! ## Synchronization, measured on an idle H200 -/

def mbarrierLatency : Fact Nat := meas 65 ""
def clusterMbarrierLatency : Fact Nat := meas 395 ""
/-- `barrier.cluster`: one SM-to-SM hop of the cluster network after the last arrival (inferred). -/
def clusterBarrierCycles : Fact Nat :=
  inferred dsmemLatency.value "meas-h200" dsmemLatency.source.loc
    "" (some "H200")

def namedBarrierLatency : Fact Nat := meas 36 ""

def barArriveCycles : Fact Nat := inferred 10 "meas-h200" "" ""

def tryWaitLimit : Fact Nat :=
  measOn 1941 ""
    ""

def tryWaitWake : Fact Nat :=
  inferred 16 "meas-gh200" ""
    ""
    (some "GH200")

def barSync : Fact (List (Nat × Nat)) :=
  meas [(64, 16), (128, 20), (256, 28), (512, 44), (1024, 76)]
    ""

def fenceProxyAsync : Fact Nat :=
  meas 18 "" ""
def fenceAcqRelCta : Fact Nat := meas 23 ""
def fenceAcqRelGpu : Fact Nat := meas 274 ""
def fenceScGpu : Fact Nat := meas 273 ""
def fenceAcqRelSys : Fact Nat := meas 1155 ""

def fenceCycles : Fact (List (String × Nat)) :=
  inferred [("MEMBAR.ALL.CTA", 14), ("FENCE.VIEW.ASYNC.S", 0), ("MEMBAR.ALL.GPU", 257),
    ("MEMBAR.SC.GPU", 256), ("MEMBAR.ALL.SYS", 1138)] "meas-h200"
    ""
    ""

def redReleaseGpu : Fact Nat :=
  meas 426 "" ""

def atomAcqRelGpu : Fact Nat := meas 882 ""
def atomRelaxedGpu : Fact Nat := meas 470 ""
def ldAcquireGpu : Fact Nat := meas 434 ""
def flagHopPs : Fact Nat := meas 443000 ""

/-- Device `printf`, measured on a GH200: a call with three `int` arguments. -/
def printfPs : Fact Nat :=
  measOn 21470000 ""
    ""
def printfLanePs : Fact Nat :=
  measOn 88000 ""
    ""

/-! ## Launch, measured on an H200 -/

def ctaDispatchCycles : Fact (Nat × Nat) :=
  inferred (1, 2) "nvidia-forum-337324" "post 2 (NVIDIA staff, 2026-05-29)"
    ""

def gpcLaunchCycles : Fact Nat :=
  inferred 2775 "meas-gh200" ""
    "" (some "GH200")

def emptyGraphPs : Fact Nat := meas 2940000 ""
def kernelGapPs : Fact Nat :=
  meas 768000 ""
    ""

def kernelGapPdlPs : Fact Nat :=
  meas 512000 "" ""

/-- The share of a link's bandwidth a large collective achieves. -/
def peerPermille : Fact Nat :=
  inferred 702 "meas" ""
    "" (some "H100 SXM5")

end NV.Arch.Hopper
