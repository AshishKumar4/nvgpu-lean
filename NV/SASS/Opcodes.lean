import NV.Fact

/-!
# The opcode catalogues

NVIDIA's lists of the instruction sets (CUDA Binary Utilities v13.5, §4.3): Hopper's (Table 8),
Ampere and Ada's (Table 7) and Blackwell and Rubin's (the second Table 8), each opcode with its
category and NVIDIA's one-line description. It is a list of names, not semantics:
which opcodes `NV.SASS.decode` covers is what its tests show. `Tests/SassReader.lean` checks that
every opcode in the listings it reads is here or in `undocumented`.
-/

namespace NV.SASS

/-- The categories of NVIDIA's table. -/
inductive OpClass where
  | float | integer | conversion | movement | predicate | memory | uniform | warpgroup | tma
  | texture | surface | control | misc
  /-- Blackwell's tensor-memory instructions (`tcgen05`). -/
  | tensorMemory
  deriving DecidableEq, Repr

/-- An opcode of the catalogue. -/
structure OpInfo where
  name : String
  cls : OpClass
  /-- NVIDIA's description. -/
  what : String
  deriving Repr

def opcodes : Fact (List OpInfo) :=
  { kind := .documented, source := { ref := "cuda-binutils", loc := "§4.3 Table 8" }
    value := [
      ⟨"FADD", .float, "FP32 Add"⟩,
      ⟨"FADD32I", .float, "FP32 Add"⟩,
      ⟨"FCHK", .float, "Floating-point Range Check"⟩,
      ⟨"FFMA32I", .float, "FP32 Fused Multiply and Add"⟩,
      ⟨"FFMA", .float, "FP32 Fused Multiply and Add"⟩,
      ⟨"FMNMX", .float, "FP32 Minimum/Maximum"⟩,
      ⟨"FMUL", .float, "FP32 Multiply"⟩,
      ⟨"FMUL32I", .float, "FP32 Multiply"⟩,
      ⟨"FSEL", .float, "Floating Point Select"⟩,
      ⟨"FSET", .float, "FP32 Compare And Set"⟩,
      ⟨"FSETP", .float, "FP32 Compare And Set Predicate"⟩,
      ⟨"FSWZADD", .float, "FP32 Swizzle Add"⟩,
      ⟨"MUFU", .float, "FP32 Multi Function Operation"⟩,
      ⟨"HADD2", .float, "FP16 Add"⟩,
      ⟨"HADD2_32I", .float, "FP16 Add"⟩,
      ⟨"HFMA2", .float, "FP16 Fused Multiply Add"⟩,
      ⟨"HFMA2_32I", .float, "FP16 Fused Multiply Add"⟩,
      ⟨"HMMA", .float, "Matrix Multiply and Accumulate"⟩,
      ⟨"HMNMX2", .float, "FP16 Minimum / Maximum"⟩,
      ⟨"HMUL2", .float, "FP16 Multiply"⟩,
      ⟨"HMUL2_32I", .float, "FP16 Multiply"⟩,
      ⟨"HSET2", .float, "FP16 Compare And Set"⟩,
      ⟨"HSETP2", .float, "FP16 Compare And Set Predicate"⟩,
      ⟨"DADD", .float, "FP64 Add"⟩,
      ⟨"DFMA", .float, "FP64 Fused Multiply Add"⟩,
      ⟨"DMMA", .float, "Matrix Multiply and Accumulate"⟩,
      ⟨"DMUL", .float, "FP64 Multiply"⟩,
      ⟨"DSETP", .float, "FP64 Compare And Set Predicate"⟩,
      ⟨"BMMA", .integer, "Bit Matrix Multiply and Accumulate"⟩,
      ⟨"BMSK", .integer, "Bitfield Mask"⟩,
      ⟨"BREV", .integer, "Bit Reverse"⟩,
      ⟨"FLO", .integer, "Find Leading One"⟩,
      ⟨"IABS", .integer, "Integer Absolute Value"⟩,
      ⟨"IADD", .integer, "Integer Addition"⟩,
      ⟨"IADD3", .integer, "3-input Integer Addition"⟩,
      ⟨"IADD32I", .integer, "Integer Addition"⟩,
      ⟨"IDP", .integer, "Integer Dot Product and Accumulate"⟩,
      ⟨"IDP4A", .integer, "Integer Dot Product and Accumulate"⟩,
      ⟨"IMAD", .integer, "Integer Multiply And Add"⟩,
      ⟨"IMMA", .integer, "Integer Matrix Multiply and Accumulate"⟩,
      ⟨"IMNMX", .integer, "Integer Minimum/Maximum"⟩,
      ⟨"IMUL", .integer, "Integer Multiply"⟩,
      ⟨"IMUL32I", .integer, "Integer Multiply"⟩,
      ⟨"ISCADD", .integer, "Scaled Integer Addition"⟩,
      ⟨"ISCADD32I", .integer, "Scaled Integer Addition"⟩,
      ⟨"ISETP", .integer, "Integer Compare And Set Predicate"⟩,
      ⟨"LEA", .integer, "LOAD Effective Address"⟩,
      ⟨"LOP", .integer, "Logic Operation"⟩,
      ⟨"LOP3", .integer, "Logic Operation"⟩,
      ⟨"LOP32I", .integer, "Logic Operation"⟩,
      ⟨"POPC", .integer, "Population count"⟩,
      ⟨"SHF", .integer, "Funnel Shift"⟩,
      ⟨"SHL", .integer, "Shift Left"⟩,
      ⟨"SHR", .integer, "Shift Right"⟩,
      ⟨"VABSDIFF", .integer, "Absolute Difference"⟩,
      ⟨"VABSDIFF4", .integer, "Absolute Difference"⟩,
      ⟨"VHMNMX", .integer, "SIMD FP16 3-Input Minimum / Maximum"⟩,
      ⟨"VIADD", .integer, "SIMD Integer Addition"⟩,
      ⟨"VIADDMNMX", .integer, "SIMD Integer Addition and Fused Min/Max Comparison"⟩,
      ⟨"VIMNMX", .integer, "SIMD Integer Minimum / Maximum"⟩,
      ⟨"VIMNMX3", .integer, "SIMD Integer 3-Input Minimum / Maximum"⟩,
      ⟨"F2F", .conversion, "Floating Point To Floating Point Conversion"⟩,
      ⟨"F2I", .conversion, "Floating Point To Integer Conversion"⟩,
      ⟨"I2F", .conversion, "Integer To Floating Point Conversion"⟩,
      ⟨"I2I", .conversion, "Integer To Integer Conversion"⟩,
      ⟨"I2IP", .conversion, "Integer To Integer Conversion and Packing"⟩,
      ⟨"I2FP", .conversion, "Integer to FP32 Convert and Pack"⟩,
      ⟨"F2IP", .conversion, "FP32 Down-Convert to Integer and Pack"⟩,
      ⟨"FRND", .conversion, "Round To Integer"⟩,
      ⟨"MOV", .movement, "Move"⟩,
      ⟨"MOV32I", .movement, "Move"⟩,
      ⟨"MOVM", .movement, "Move Matrix with Transposition or Expansion"⟩,
      ⟨"PRMT", .movement, "Permute Register Pair"⟩,
      ⟨"SEL", .movement, "Select Source with Predicate"⟩,
      ⟨"SGXT", .movement, "Sign Extend"⟩,
      ⟨"SHFL", .movement, "Warp Wide Register Shuffle"⟩,
      ⟨"PLOP3", .predicate, "Predicate Logic Operation"⟩,
      ⟨"PSETP", .predicate, "Combine Predicates and Set Predicate"⟩,
      ⟨"P2R", .predicate, "Move Predicate Register To Register"⟩,
      ⟨"R2P", .predicate, "Move Register To Predicate Register"⟩,
      ⟨"FENCE", .memory, "Memory Visibility Guarantee for Shared or Global Memory"⟩,
      ⟨"LD", .memory, "Load from generic Memory"⟩,
      ⟨"LDC", .memory, "Load Constant"⟩,
      ⟨"LDG", .memory, "Load from Global Memory"⟩,
      ⟨"LDGDEPBAR", .memory, "Global Load Dependency Barrier"⟩,
      ⟨"LDGMC", .memory, "Reducing Load"⟩,
      ⟨"LDGSTS", .memory, "Asynchronous Global to Shared Memcopy"⟩,
      ⟨"LDL", .memory, "Load within Local Memory Window"⟩,
      ⟨"LDS", .memory, "Load within Shared Memory Window"⟩,
      ⟨"LDSM", .memory, "Load Matrix from Shared Memory with Element Size Expansion"⟩,
      ⟨"STSM", .memory, "Store Matrix to Shared Memory"⟩,
      ⟨"ST", .memory, "Store to Generic Memory"⟩,
      ⟨"STG", .memory, "Store to Global Memory"⟩,
      ⟨"STL", .memory, "Store to Local Memory"⟩,
      ⟨"STS", .memory, "Store to Shared Memory"⟩,
      ⟨"STAS", .memory, "Asynchronous Store to Distributed Shared Memory With Explicit Synchronization"⟩,
      ⟨"SYNCS", .memory, "Sync Unit"⟩,
      ⟨"MATCH", .memory, "Match Register Values Across Thread Group"⟩,
      ⟨"QSPC", .memory, "Query Space"⟩,
      ⟨"ATOM", .memory, "Atomic Operation on Generic Memory"⟩,
      ⟨"ATOMS", .memory, "Atomic Operation on Shared Memory"⟩,
      ⟨"ATOMG", .memory, "Atomic Operation on Global Memory"⟩,
      ⟨"REDAS", .memory, "Asynchronous Reduction on Distributed Shared Memory With Explicit Synchronization"⟩,
      ⟨"REDG", .memory, "Reduction Operation on Generic Memory"⟩,
      ⟨"CCTL", .memory, "Cache Control"⟩,
      ⟨"CCTLL", .memory, "Cache Control"⟩,
      ⟨"ERRBAR", .memory, "Error Barrier"⟩,
      ⟨"MEMBAR", .memory, "Memory Barrier"⟩,
      ⟨"CCTLT", .memory, "Texture Cache Control"⟩,
      ⟨"R2UR", .uniform, "Move from Vector Register to a Uniform Register"⟩,
      ⟨"REDUX", .uniform, "Reduction of a Vector Register into a Uniform Register"⟩,
      ⟨"S2UR", .uniform, "Move Special Register to Uniform Register"⟩,
      ⟨"UBMSK", .uniform, "Uniform Bitfield Mask"⟩,
      ⟨"UBREV", .uniform, "Uniform Bit Reverse"⟩,
      ⟨"UCGABAR_ARV", .uniform, "CGA Barrier Synchronization"⟩,
      ⟨"UCGABAR_WAIT", .uniform, "CGA Barrier Synchronization"⟩,
      ⟨"UCLEA", .uniform, "Load Effective Address for a Constant"⟩,
      ⟨"UF2FP", .uniform, "Uniform FP32 Down-convert and Pack"⟩,
      ⟨"UFLO", .uniform, "Uniform Find Leading One"⟩,
      ⟨"UIADD3", .uniform, "Uniform Integer Addition"⟩,
      ⟨"UIADD3.64", .uniform, "Uniform Integer Addition"⟩,
      ⟨"UIMAD", .uniform, "Uniform Integer Multiplication"⟩,
      ⟨"UISETP", .uniform, "Integer Compare and Set Uniform Predicate"⟩,
      ⟨"ULDC", .uniform, "Load from Constant Memory into a Uniform Register"⟩,
      ⟨"ULEA", .uniform, "Uniform Load Effective Address"⟩,
      ⟨"ULEPC", .uniform, "Uniform Load Effective PC"⟩,
      ⟨"ULOP", .uniform, "Logic Operation"⟩,
      ⟨"ULOP3", .uniform, "Logic Operation"⟩,
      ⟨"ULOP32I", .uniform, "Logic Operation"⟩,
      ⟨"UMOV", .uniform, "Uniform Move"⟩,
      ⟨"UP2UR", .uniform, "Uniform Predicate to Uniform Register"⟩,
      ⟨"UPLOP3", .uniform, "Uniform Predicate Logic Operation"⟩,
      ⟨"UPOPC", .uniform, "Uniform Population Count"⟩,
      ⟨"UPRMT", .uniform, "Uniform Byte Permute"⟩,
      ⟨"UPSETP", .uniform, "Uniform Predicate Logic Operation"⟩,
      ⟨"UR2UP", .uniform, "Uniform Register to Uniform Predicate"⟩,
      ⟨"USEL", .uniform, "Uniform Select"⟩,
      ⟨"USETMAXREG", .uniform, "Release, Deallocate and Allocate Registers"⟩,
      ⟨"USGXT", .uniform, "Uniform Sign Extend"⟩,
      ⟨"USHF", .uniform, "Uniform Funnel Shift"⟩,
      ⟨"USHL", .uniform, "Uniform Left Shift"⟩,
      ⟨"USHR", .uniform, "Uniform Right Shift"⟩,
      ⟨"VOTEU", .uniform, "Voting across SIMD Thread Group with Results in Uniform Destination"⟩,
      ⟨"BGMMA", .warpgroup, "Bit Matrix Multiply and Accumulate Across Warps"⟩,
      ⟨"HGMMA", .warpgroup, "Matrix Multiply and Accumulate Across a Warpgroup"⟩,
      ⟨"IGMMA", .warpgroup, "Integer Matrix Multiply and Accumulate Across a Warpgroup"⟩,
      ⟨"QGMMA", .warpgroup, "FP8 Matrix Multiply and Accumulate Across a Warpgroup"⟩,
      ⟨"WARPGROUP", .warpgroup, "Warpgroup Synchronization"⟩,
      ⟨"WARPGROUPSET", .warpgroup, "Set Warpgroup Counters"⟩,
      ⟨"UBLKCP", .tma, "Bulk Data Copy"⟩,
      ⟨"UBLKPF", .tma, "Bulk Data Prefetch"⟩,
      ⟨"UBLKRED", .tma, "Bulk Data Copy from Shared Memory with Reduction"⟩,
      ⟨"UTMACCTL", .tma, "TMA Cache Control"⟩,
      ⟨"UTMACMDFLUSH", .tma, "TMA Command Flush"⟩,
      ⟨"UTMALDG", .tma, "Tensor Load from Global to Shared Memory"⟩,
      ⟨"UTMAPF", .tma, "Tensor Prefetch"⟩,
      ⟨"UTMAREDG", .tma, "Tensor Store from Shared to Global Memory with Reduction"⟩,
      ⟨"UTMASTG", .tma, "Tensor Store from Shared to Global Memory"⟩,
      ⟨"TEX", .texture, "Texture Fetch"⟩,
      ⟨"TLD", .texture, "Texture Load"⟩,
      ⟨"TLD4", .texture, "Texture Load 4"⟩,
      ⟨"TMML", .texture, "Texture MipMap Level"⟩,
      ⟨"TXD", .texture, "Texture Fetch With Derivatives"⟩,
      ⟨"TXQ", .texture, "Texture Query"⟩,
      ⟨"SUATOM", .surface, "Atomic Op on Surface Memory"⟩,
      ⟨"SULD", .surface, "Surface Load"⟩,
      ⟨"SURED", .surface, "Reduction Op on Surface Memory"⟩,
      ⟨"SUST", .surface, "Surface Store"⟩,
      ⟨"ACQBULK", .control, "Wait for Bulk Release Status Warp State"⟩,
      ⟨"BMOV", .control, "Move Convergence Barrier State"⟩,
      ⟨"BPT", .control, "BreakPoint/Trap"⟩,
      ⟨"BRA", .control, "Relative Branch"⟩,
      ⟨"BREAK", .control, "Break out of the Specified Convergence Barrier"⟩,
      ⟨"BRX", .control, "Relative Branch Indirect"⟩,
      ⟨"BRXU", .control, "Relative Branch with Uniform Register Based Offset"⟩,
      ⟨"BSSY", .control, "Barrier Set Convergence Synchronization Point"⟩,
      ⟨"BSYNC", .control, "Synchronize Threads on a Convergence Barrier"⟩,
      ⟨"CALL", .control, "Call Function"⟩,
      ⟨"CGAERRBAR", .control, "CGA Error Barrier"⟩,
      ⟨"ELECT", .control, "Elect a Leader Thread"⟩,
      ⟨"ENDCOLLECTIVE", .control, "Reset the MCOLLECTIVE mask"⟩,
      ⟨"EXIT", .control, "Exit Program"⟩,
      ⟨"JMP", .control, "Absolute Jump"⟩,
      ⟨"JMX", .control, "Absolute Jump Indirect"⟩,
      ⟨"JMXU", .control, "Absolute Jump with Uniform Register Based Offset"⟩,
      ⟨"KILL", .control, "Kill Thread"⟩,
      ⟨"NANOSLEEP", .control, "Suspend Execution"⟩,
      ⟨"PREEXIT", .control, "Dependent Task Launch Hint"⟩,
      ⟨"RET", .control, "Return From Subroutine"⟩,
      ⟨"RPCMOV", .control, "PC Register Move"⟩,
      ⟨"WARPSYNC", .control, "Synchronize Threads in Warp"⟩,
      ⟨"YIELD", .control, "Yield Control"⟩,
      ⟨"B2R", .misc, "Move Barrier To Register"⟩,
      ⟨"BAR", .misc, "Barrier Synchronization"⟩,
      ⟨"CS2R", .misc, "Move Special Register to Register"⟩,
      ⟨"DEPBAR", .misc, "Dependency Barrier"⟩,
      ⟨"GETLMEMBASE", .misc, "Get Local Memory Base Address"⟩,
      ⟨"LEPC", .misc, "Load Effective PC"⟩,
      ⟨"NOP", .misc, "No Operation"⟩,
      ⟨"PMTRIG", .misc, "Performance Monitor Trigger"⟩,
      ⟨"S2R", .misc, "Move Special Register to Register"⟩,
      ⟨"SETCTAID", .misc, "Set CTA ID"⟩,
      ⟨"SETLMEMBASE", .misc, "Set Local Memory Base Address"⟩,
      ⟨"VOTE", .misc, "Vote Across SIMT Thread Group"⟩] }

/-- `base` without the opcodes of `drop`, with the entries of `over` in place of theirs, and each
group of `add` after the opcode it names (`""`: first): another family's table as edits of
`opcodes`, in its own order. -/
def derive (base : List OpInfo) (drop : List String) (over : List OpInfo := [])
    (add : List (String × List OpInfo)) : List OpInfo :=
  let kept := (base.filter (!drop.contains ·.name)).map fun o => (over.find? (·.name == o.name)).getD o
  (add.lookup "").getD [] ++ kept.flatMap fun o => o :: (add.lookup o.name).getD []

/-- NVIDIA's list of the Ampere and Ada instruction set (`sm_80`–`sm_89`), as `opcodes`. -/
def opcodesSm80 : Fact (List OpInfo) :=
  { kind := .documented, source := { ref := "cuda-binutils", loc := "§4.3 Table 7, NVIDIA Ampere GPU and Ada Instruction Set" }
    value := derive opcodes.value
      (drop := ["VHMNMX", "VIADD", "VIADDMNMX", "VIMNMX", "VIMNMX3", "FENCE", "LDGMC", "STSM",
        "STAS", "SYNCS", "REDAS", "REDG", "UCGABAR_ARV", "UCGABAR_WAIT", "ULEPC", "USETMAXREG",
        "BGMMA", "HGMMA", "IGMMA", "QGMMA", "WARPGROUP", "WARPGROUPSET", "UBLKCP", "UBLKPF",
        "UBLKRED", "UTMACCTL", "UTMACMDFLUSH", "UTMALDG", "UTMAPF", "UTMAREDG", "UTMASTG",
        "ACQBULK", "CGAERRBAR", "ELECT", "ENDCOLLECTIVE", "PREEXIT"])
      (over := [⟨"VOTE", .misc, "Vote Across SIMD Thread Group"⟩])
      (add := [
        ("ATOMG", [⟨"RED", .memory, "Reduction Operation on Generic Memory"⟩])]) }

/-- NVIDIA's list of the Blackwell and Rubin instruction set (`sm_100` and later, `sm_120` included: the
table is the family's, so it lists data-center opcodes (`tcgen05`'s `UTC…`) that `sm_120` does not have). -/
def opcodesBlackwell : Fact (List OpInfo) :=
  { kind := .documented, source := { ref := "cuda-binutils", loc := "§4.3 Table 8, Blackwell and Rubin Instruction Set" }
    value := derive opcodes.value
      (drop := ["BMMA", "ULDC", "BGMMA", "HGMMA", "IGMMA", "QGMMA", "WARPGROUP", "WARPGROUPSET"])
      (over := [⟨"UISETP", .uniform, "Uniform Integer Compare and Set Uniform Predicate"⟩, ⟨"ULOP", .uniform, "Uniform Logic Operation"⟩, ⟨"ULOP3", .uniform, "Uniform Logic Operation"⟩, ⟨"ULOP32I", .uniform, "Uniform Logic Operation"⟩])
      (add := [
        ("FADD", [⟨"FADD2", .float, "FP32 Add"⟩]),
        ("FFMA", [⟨"FFMA2", .float, "FP32 Fused Multiply and Add"⟩,
          ⟨"FHADD", .float, "FP32 Addition"⟩,
          ⟨"FHADD2", .float, "FP32 Addition"⟩,
          ⟨"FHFMA", .float, "FP32 Fused Multiply and Add"⟩,
          ⟨"FHFMA2", .float, "FP32 Fused Multiply and Add"⟩,
          ⟨"FHMUL2", .float, "FP32 Multiply"⟩]),
        ("FMNMX", [⟨"FMNMX3", .float, "3-Input Floating-point Minimum / Maximum"⟩]),
        ("FMUL", [⟨"FMUL2", .float, "FP32 Multiply"⟩]),
        ("DSETP", [⟨"OMMA", .float, "FP4 Matrix Multiply and Accumulate Across a Warp"⟩,
          ⟨"QMMA", .float, "FP8 Matrix Multiply and Accumulate Across a Warp"⟩]),
        ("BREV", [⟨"DECOMPRESS", .integer, "Decompress a vector"⟩]),
        ("SHR", [⟨"SPARSIFY", .integer, "Generate metadata and compress a vector"⟩]),
        ("MOV32I", [⟨"MOV64IUR", .movement, "Move"⟩]),
        ("SYNCS", [⟨"SYNCSU", .memory, "Sync Unit"⟩,
          ⟨"USYNCS", .memory, "Sync Unit"⟩]),
        ("REDAS", [⟨"REDS", .memory, "Atomic Op on Shared Memory"⟩]),
        ("CCTLT", [⟨"CREDUX", .uniform, "Coupled Reduction of a Vector Register into a Uniform Register"⟩,
          ⟨"CS2UR", .uniform, "Load a Value from Constant Memory into a Uniform Register"⟩,
          ⟨"LDCU", .uniform, "Load a Value from Constant Memory into a Uniform Register"⟩]),
        ("UCLEA", [⟨"UFADD", .uniform, "Uniform FP32 Addition"⟩,
          ⟨"UF2F", .uniform, "Uniform Float-to-Float Conversion"⟩]),
        ("UF2FP", [⟨"UF2I", .uniform, "Uniform Float-to-Integer Conversion"⟩,
          ⟨"UF2IP", .uniform, "Uniform FP32 Down-Convert to Integer and Pack"⟩,
          ⟨"UFFMA", .uniform, "Uniform FP32 Fused Multiply-Add"⟩]),
        ("UFLO", [⟨"UFMNMX", .uniform, "Uniform Floating-point Minimum / Maximum"⟩,
          ⟨"UFMUL", .uniform, "Uniform FP32 Multiply"⟩,
          ⟨"UFRND", .uniform, "Uniform Round to Integer"⟩,
          ⟨"UFSEL", .uniform, "Uniform Floating-Point Select"⟩,
          ⟨"UFSET", .uniform, "Uniform Floating-Point Compare and Set"⟩,
          ⟨"UFSETP", .uniform, "Uniform Floating-Point Compare and Set Predicate"⟩,
          ⟨"UI2F", .uniform, "Uniform Integer to Float conversion"⟩,
          ⟨"UI2FP", .uniform, "Uniform Integer to FP32 Convert and Pack"⟩,
          ⟨"UI2I", .uniform, "Uniform Saturating Integer-to-Integer Conversion"⟩,
          ⟨"UI2IP", .uniform, "Uniform Dual Saturating Integer-to-Integer Conversion and Packing"⟩,
          ⟨"UIABS", .uniform, "Uniform Integer Absolute Value"⟩,
          ⟨"UIMNMX", .uniform, "Uniform Integer Minimum / Maximum"⟩]),
        ("UMOV", [⟨"UMOV.64", .uniform, "Uniform Addition"⟩]),
        ("USHR", [⟨"UTRACEEVENT", .uniform, "Uniform Trace Event"⟩,
          ⟨"UGETNEXTWORKID", .uniform, "Uniform Get Next Work ID"⟩,
          ⟨"UMEMSETS", .uniform, "Initialize Shared Memory"⟩,
          ⟨"UREDGR", .uniform, "Uniform Reduction on Global Memory with Release"⟩,
          ⟨"USTGR", .uniform, "Uniform Store to Global Memory with Release"⟩,
          ⟨"UVIADD", .uniform, "Uniform SIMD Integer Addition"⟩,
          ⟨"UVIMNMX", .uniform, "Uniform SIMD Integer Minimum / Maximum"⟩,
          ⟨"UVIRTCOUNT", .uniform, "Virtual Resource Management"⟩]),
        ("UBLKCP", [⟨"UBLKL2CCTL", .tma, "Bulk Data Prefetch"⟩]),
        ("UTMACMDFLUSH", [⟨"UTMAL2CCTL", .tma, "TMA Cache Control"⟩]),
        ("UTMASTG", [⟨"LDT", .tensorMemory, "Load Matrix from Tensor Memory to Register File"⟩,
          ⟨"LDTM", .tensorMemory, "Load Matrix from Tensor Memory to Register File"⟩,
          ⟨"STT", .tensorMemory, "Store Matrix to Tensor Memory from Register File"⟩,
          ⟨"STTM", .tensorMemory, "Store Matrix to Tensor Memory from Register File"⟩,
          ⟨"UTCATOMSWS", .tensorMemory, "Perform Atomic operation on SW State Register"⟩,
          ⟨"UTCBAR", .tensorMemory, "Tensor Core Barrier"⟩,
          ⟨"UTCCP", .tensorMemory, "Asynchronous data copy from Shared Memory to Tensor Memory"⟩,
          ⟨"UTCHMMA", .tensorMemory, "Uniform Matrix Multiply and Accumulate"⟩,
          ⟨"UTCIMMA", .tensorMemory, "Uniform Matrix Multiply and Accumulate"⟩,
          ⟨"UTCOMMA", .tensorMemory, "Uniform Matrix Multiply and Accumulate"⟩,
          ⟨"UTCQMMA", .tensorMemory, "Uniform Matrix Multiply and Accumulate"⟩,
          ⟨"UTCSHIFT", .tensorMemory, "Shift elements in Tensor Memory"⟩]),
        ("ACQBULK", [⟨"ACQSHMINIT", .control, "Wait for Shared Memory Initialization Release Status Warp State"⟩]),
        ("SETLMEMBASE", [⟨"VISET", .misc, "SIMD Integer Compare and Set"⟩])]) }

/-- The catalogue of the family an SM version belongs to. -/
def catalogue (sm : Nat) : Fact (List OpInfo) :=
  if sm ≥ 100 then opcodesBlackwell else if sm ≥ 90 then opcodes else opcodesSm80

/-- `sm_90a` opcodes the table does not list, with an example. -/
def undocumented : List (String × String) :=
  [("F2FP", "F2FP.BF16.F32.PACK_AB R0, R1, R2: two binary32 values to a packed bfloat16 pair")]

def OpInfo.find? (name : String) : Option OpInfo := opcodes.value.find? (·.name == name)

end NV.SASS
