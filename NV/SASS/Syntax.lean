/-!
# SASS instructions

An instruction as NVIDIA's disassembler prints it, plus its control word.
Operands are kept as the disassembler names them; their meaning is `NV.SASS.Sem`'s. The timing
core reads only `pc`, `opcode`, `mods`, `ctrl` and the register operands.
-/

namespace NV.SASS

/-- Register files of an SM sub-partition (Huerta et al. 2025, §5). Indices are the encoded ones:
`RZ` is `r 255`, `PT` is `p 7`, `UPT` is `up 7`; `URZ` is `ur 255` here on every target (sm_100 on
encode it as `0xff`, earlier targets as `0x3f`, where no `UR63` exists). -/
inductive RegFile where
  | r | ur | p | up
  /-- Convergence barrier registers (`B0`…), used by `BSSY`/`BSYNC`. -/
  | b
  deriving DecidableEq, Repr, Inhabited, Hashable

structure Reg where
  file : RegFile
  idx : Nat
  deriving DecidableEq, Repr, Inhabited, Hashable

namespace Reg
def RZ : Reg := ⟨.r, 255⟩
def URZ : Reg := ⟨.ur, 255⟩
def PT : Reg := ⟨.p, 7⟩
def UPT : Reg := ⟨.up, 7⟩
/-- Writes to the zero registers and the true predicates are discarded. -/
def isSink (r : Reg) : Bool := r == RZ || r == URZ || r == PT || r == UPT
end Reg

/-- Operand modifiers the disassembler prints around a register. -/
structure RegMods where
  neg : Bool := false
  abs : Bool := false
  /-- `!P0`, `~R1`. -/
  not : Bool := false
  /-- `.reuse`: keep the value in the operand reuse cache. -/
  reuse : Bool := false
  /-- Other suffixes, in order (`.H1`, `.64`, `.B2`, `.X4`, …). -/
  suffix : Array String := #[]
  deriving DecidableEq, Repr, Inhabited

inductive Operand where
  | reg (r : Reg) (m : RegMods := {})
  /-- An immediate, as its bit pattern (floats as their IEEE bits). -/
  | imm (bits : UInt64)
  /-- `c[bank][offset]`, `c[bank][R+offset]`, `cx[UR][…]`. -/
  | cbank (bank : Nat) (base : Option Reg) (offset : Int)
  /-- `[R+UR+offset]`, `desc[UR][R.64+offset]`: a memory address. `wide` is the `.64` base. -/
  | mem (base : Option Reg) (wide : Bool) (ureg : Option Reg) (desc : Option Reg) (offset : Int)
  /-- `-c[…]`: a negated constant operand (sm_80's `IADD3 R3, R6, -c[0x0][0x170], RZ`). -/
  | neg (o : Operand)
  /-- `~c[…]` (sm_80): the bitwise complement of a constant. -/
  | not (o : Operand)
  /-- `[R7.X4]`: a memory address whose register base is scaled (sm_80's shared-memory index
  forms); `m` is the address with that register as its base. -/
  | scaled (m : Operand) (scale : Nat)
  /-- A wgmma matrix descriptor: `gdesc[UR4]`, the 64-bit descriptor in `UR4`:`UR5`, with its
  printed suffixes (`gdesc[UR4].negA.tnspA.tnspB`: `negA`, `negB`, `tnspA`, `tnspB`). -/
  | gdesc (r : Reg) (mods : Array String := #[])
  /-- A tensor-memory address or an MMA instruction descriptor in a uniform register (`sm_100a`):
  `tmem[UR6]`, `tmem[UR4+0x8]`, `idesc[UR5]`; `kind` is the prefix, `off` added to the register. -/
  | udesc (kind : String) (r : Reg) (off : Int := 0)
  /-- A branch target, resolved to its address. -/
  | target (pc : Nat)
  /-- Named operands: special registers (`SR_TID.X`), `gsb0`, scoreboard names, … -/
  | name (s : String)
  deriving DecidableEq, Repr, Inhabited

/-- The control word (Huerta et al. 2025 §4). -/
structure Ctrl where
  /-- Cycles before the warp may issue its next instruction (0–15). -/
  stall : Nat
  /-- Yield: prefer another warp next cycle. -/
  yield : Bool
  /-- Dependence counter incremented at issue and decremented at writeback. -/
  wbar : Option (Fin 6)
  /-- Dependence counter incremented at issue and decremented once the sources are read. -/
  rbar : Option (Fin 6)
  /-- Counters that must be zero before issue, one bit each. -/
  wait : Nat
  /-- Reuse flags of the first four source operand slots. -/
  reuse : Nat
  deriving DecidableEq, Repr, Inhabited

/-- Decode the control word from the high 64 bits of an instruction's 128-bit encoding: stall in
bits 41–44, yield 45, write counter 46–48, read counter 49–51, wait mask 52–57, reuse 58–61. A
counter field of 7 means none, and the yield bit is inverted: 0 asks to yield (Huerta et al. 2025
§4); `Tests/SassReader.lean` checks it. -/
def Ctrl.ofHi (hi : UInt64) : Ctrl :=
  let field (lo width : Nat) : Nat := (hi >>> lo.toUInt64).toNat % 2 ^ width
  let counter (v : Nat) : Option (Fin 6) := if h : v < 6 then some ⟨v, h⟩ else none
  { stall := field 41 4, yield := field 45 1 == 0, wbar := counter (field 46 3),
    rbar := counter (field 49 3), wait := field 52 6, reuse := field 58 4 }

structure Instr where
  /-- Byte offset in the function. -/
  pc : Nat
  /-- `@P0`, `@!UP1`: the guard and whether it is negated; `none` is `@PT`. -/
  guard : Option (Reg × Bool) := none
  /-- Base opcode: `FFMA`, `LDG`, `HGMMA`. -/
  opcode : String
  /-- Opcode modifiers in order: `E`, `128`, `CONSTANT` for `LDG.E.128.CONSTANT`. -/
  mods : Array String := #[]
  operands : Array Operand := #[]
  ctrl : Ctrl
  deriving DecidableEq, Repr, Inhabited

def Instr.hasMod (i : Instr) (m : String) : Bool := i.mods.contains m

/-- `USETMAXREG.TRY_ALLOC.CTAPOOL`: opcode and modifiers. -/
def Instr.full (i : Instr) : String := ".".intercalate (i.opcode :: i.mods.toList)

/-- One kernel's code and the attributes the core needs from its ELF. -/
structure Kernel where
  name : String
  /-- The target's SM version, from the ELF header (`sm=90`): 80, 90, 120. -/
  sm : Nat := 90
  code : Array Instr
  /-- Registers per thread (`EIATTR_REGCOUNT`): what a launch allocates. -/
  regs : Nat
  /-- Registers per thread the code can address: `regs`, or the largest count a
  `USETMAXREG.TRY_ALLOC` raises the allocation to while running (warp-specialized kernels hand
  registers between warpgroups); the warp's register file has this many. -/
  regFile : Nat := 0
  /-- Static shared memory bytes per CTA the kernel declares (ptxas's `smem`). -/
  sharedBytes : Nat := 0
  /-- Bytes at the start of each CTA's shared-memory window ahead of the kernel's data, which the
  code addresses from this offset (`NV.SASS.Elf.reservedShared`). -/
  sharedReserved : Nat := 0
  /-- Local memory per thread: the stack the kernel and its callees need
  (`EIATTR_MIN_STACK_SIZE`). The stack pointer starts at its top, `c[0x0][0x28]`, and frames are
  allocated downward from it. -/
  localBytes : Nat := 0
  /-- Parameter block size in bytes; parameters live in constant bank 0 at `paramBase`. -/
  paramBytes : Nat := 0
  paramBase : Nat := 0x210
  /-- Kernel parameters in declaration order: byte offset in the parameter block and size. -/
  params : Array (Nat × Nat) := #[]
  /-- Required cluster shape, if the kernel declares one. -/
  cluster : Option (Nat × Nat × Nat) := none
  /-- Byte offset of the kernel's code within its module's code image: the `.text` sections in
  file order, each at its file offset from the first (128-byte aligned, `textAlignment`). That the
  loader keeps this layout is inferred. -/
  codeOffset : Nat := 0
  /-- Which module (cubin) the kernel comes from: a hash of its listing's text, the same for every
  kernel read from one listing, so that kernels of different modules do not share code addresses. -/
  module : UInt64 := 0
  /-- Constant banks other than 0 that the cubin fills for this kernel (bank 2: jump tables) or
  for its module (bank 3: `__constant__` data; bank 4: relocated addresses), as 32-bit words. -/
  banks : Array (Nat × Array UInt32) := #[]
  /-- The module's device data (`__device__` variables, string literals) at the addresses the
  loader gives them (`NV.SASS.Elf.moduleData`), as bytes: the launch writes them into global
  memory, unless `dataLoaded`. `printf` finds its format strings here. -/
  data : Array (Nat × Array UInt8) := #[]
  /-- The module's data are in global memory already (the emulated device's modules: written once
  when the module loads), so a launch does not write them again over what the program has done to
  them since. -/
  dataLoaded : Bool := false
  /-- External functions the code may call, by address (`vprintf`). -/
  externs : Array (Nat × String) := #[]
  deriving Repr, Inhabited

end NV.SASS
