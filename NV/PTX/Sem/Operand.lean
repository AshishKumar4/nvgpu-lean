import NV.PTX.Sem.State

/-!
# Reading and writing operands

Integer constant expressions are evaluated by PTX ISA 9.4 §4.5.5 (64-bit, signed unless a
literal is too large or marked `U`, with the usual arithmetic conversions). Floating-point
literals are exact bits (`0f…`, `0d…`) used at their own width; other float constants need
rounding and are read through `NV.PTX.Sem.Float`.
-/

namespace NV.PTX.Sem

/-- A 64-bit constant and whether it is unsigned (§4.5.5, Table 5). -/
structure IConst where
  v : UInt64
  unsigned : Bool

namespace IConst

def ofBool (b : Bool) : IConst := ⟨if b then 1 else 0, false⟩

def lt (a b : IConst) : Bool :=
  if a.unsigned || b.unsigned then a.v < b.v else a.v.toInt64 < b.v.toInt64

end IConst

/-- Evaluate an integer constant expression; `none` if it has names or floats. -/
def evalConst : Expr → Option IConst
  | .int n u => some ⟨UInt64.ofNat n, u⟩
  | .un op a => do
    let x ← evalConst a
    match op with
    | .plus => some x
    | .neg => some ⟨0 - x.v, x.unsigned⟩
    | .not => some (IConst.ofBool (x.v == 0))
    | .compl => some ⟨~~~x.v, true⟩
    | .castS64 => some ⟨x.v, false⟩
    | .castU64 => some ⟨x.v, true⟩
  | .bin op a b => do
    let x ← evalConst a
    let y ← evalConst b
    let u := x.unsigned || y.unsigned
    match op with
    | .add => some ⟨x.v + y.v, u⟩
    | .sub => some ⟨x.v - y.v, u⟩
    | .mul => some ⟨x.v * y.v, u⟩
    | .div =>
      if y.v == 0 then none
      else if u then some ⟨x.v / y.v, true⟩ else some ⟨(x.v.toInt64 / y.v.toInt64).toUInt64, false⟩
    | .rem => if y.v == 0 then none else some ⟨x.v % y.v, true⟩
    | .shl => some ⟨x.v <<< y.v, x.unsigned⟩
    | .shr =>
      if x.unsigned then some ⟨x.v >>> y.v, true⟩
      else some ⟨(x.v.toInt64 >>> (min y.v 63).toInt64).toUInt64, false⟩
    | .band => some ⟨x.v &&& y.v, u⟩
    | .bor => some ⟨x.v ||| y.v, u⟩
    | .bxor => some ⟨x.v ^^^ y.v, u⟩
    | .land => some (IConst.ofBool (x.v != 0 && y.v != 0))
    | .lor => some (IConst.ofBool (x.v != 0 || y.v != 0))
    | .eq => some (IConst.ofBool (x.v == y.v))
    | .ne => some (IConst.ofBool (x.v != y.v))
    | .lt => some (IConst.ofBool (x.lt y))
    | .gt => some (IConst.ofBool (y.lt x))
    | .le => some (IConst.ofBool (!(y.lt x)))
    | .ge => some (IConst.ofBool (!(x.lt y)))
  | .cond c a b => do
    let c ← evalConst c
    if c.v != 0 then evalConst a else evalConst b
  | .name "WARP_SZ" => some ⟨32, false⟩
  | _ => none

/-- A special register's value (chapter 10). -/
def sregValue (e : Env) (c : Ctx) : SReg → UInt64
  | .tid d => dim c.tid d | .ntid d => dim c.ntid d
  | .ctaid d => dim c.ctaid d | .nctaid d => dim c.nctaid d
  | .clusterid d => dim c.clusterid d | .nclusterid d => dim c.nclusterid d
  | .clusterCtaid d => dim c.clusterCtaid d | .clusterNctaid d => dim c.clusterNctaid d
  | .laneid => c.laneid.toUInt64
  | .warpid => c.warpid.toUInt64
  | .nwarpid => ((Ctx.flat (c.ntid.1 - 1, c.ntid.2.1 - 1, c.ntid.2.2 - 1) c.ntid) / 32 + 1).toUInt64
  | .smid => c.smid.toUInt64
  | .nsmid => (c.smid + 1).toUInt64
  | .gridid => 0
  | .isExplicitCluster => if c.clusterNctaid == (1, 1, 1) then 0 else 1
  | .clusterCtarank => c.clusterRank.toUInt64
  | .clusterNctarank => (c.clusterNctaid.1 * c.clusterNctaid.2.1 * c.clusterNctaid.2.2).toUInt64
  | .lanemaskEq => 1 <<< c.laneid.toUInt64
  | .lanemaskLe => mask (c.laneid + 1) 0xFFFFFFFF
  | .lanemaskLt => mask c.laneid 0xFFFFFFFF
  | .lanemaskGe => 0xFFFFFFFF &&& ~~~(mask c.laneid 0xFFFFFFFF)
  | .lanemaskGt => 0xFFFFFFFF &&& ~~~(mask (c.laneid + 1) 0xFFFFFFFF)
  | .clock => mask 32 e.clock.toUInt64
  | .clockHi => (e.clock.toUInt64 >>> 32)
  | .clock64 => e.clock.toUInt64
  | .globaltimer => e.clock.toUInt64
  | .globaltimerLo => mask 32 e.clock.toUInt64
  | .globaltimerHi => e.clock.toUInt64 >>> 32
  | .dynamicSmemSize => c.dynShared.toUInt64
  | .totalSmemSize | .aggrSmemSize => (e.prog.sharedBytes + c.dynShared).toUInt64
  | _ => 0
where
  dim (v : Nat × Nat × Nat) (d : Dim) : UInt64 :=
    match d.val with
    | 0 => v.1.toUInt64 | 1 => v.2.1.toUInt64 | 2 => v.2.2.toUInt64 | _ => 0

namespace Thread

def slot (t : Thread) (i : Nat) : UInt64 := t.frame.regs[i]!

/-- The frame taken out of the thread before its register is written, so that, the thread used
linearly, the register file is written in place, not copied. -/
def setSlot (t : Thread) (i : Nat) (v : UInt64) : Thread :=
  let n := t.frames.size - 1
  let f := t.frames[n]!
  let t := { t with frames := t.frames.set! n default }
  { t with frames := t.frames.set! n { f with regs := f.regs.set! i v } }

/-- The value of an address-like symbol: a variable's address in its space. -/
def symAddr (l : Loc) (off : Int) : Except String UInt64 :=
  match l with
  | .mem _ a => pure (a + UInt64.ofInt off)
  | .frame _ => throw "the address of a function parameter is not modelled"

/-- Read `a` as a value of type `ty` (at most 64 bits; predicates read as 0 or 1). -/
def read (e : Env) (t : Thread) (a : Arg) (ty : Ty) : Except String UInt64 :=
  match a with
  | .reg r =>
    let v := t.slot r.slot
    if ty == .pred then pure (if v &&& 1 == 1 then 1 else 0) else pure (mask ty.bits v)
  | .sreg s => pure (mask ty.bits (sregValue e t.ctx s))
  | .imm (.f32 b) =>
    if ty.bits == 32 && ty.basic != .float || ty == .f32 then pure b.toUInt64
    else throw s!"0f constant used as .{Keyword.name ty}"
  | .imm (.f64 b) =>
    if ty.bits == 64 && ty.basic != .float || ty == .f64 then pure b
    else throw s!"0d constant used as .{Keyword.name ty}"
  | .imm x =>
    match evalConst x with
    | some c =>
      if ty == .pred then pure (if c.v != 0 then 1 else 0)
      else if ty.isFloat then throw s!"integer constant used as .{Keyword.name ty}"
      else pure (mask ty.bits c.v)
    | none => throw s!"constant {repr x} needs the numerics library"
  | .sym l off => return mask ty.bits (← symAddr l off)
  | .func f => pure (0xF000_0000 + f.toUInt64)
  | .not a => do pure (if (← read e t a .pred) == 0 then 1 else 0)
  | _ => throw "operand is not a value"

/-- Read a `.b128` operand as (low, high) words. -/
def read128 (e : Env) (t : Thread) (a : Arg) : Except String (UInt64 × UInt64) :=
  match a with
  | .reg r => if r.ty.bits > 64 then pure (t.slot r.slot, t.slot (r.slot + 1))
    else pure (t.slot r.slot, 0)
  | a => return (← read e t a .b64, 0)

/-- Write a `ty`-typed value to a destination register (Table 28); `_` discards it. -/
def write (t : Thread) (a : Arg) (ty : Ty) (v : UInt64) : Except String Thread :=
  match a with
  | .sink => pure t
  | .reg r =>
    if r.ty == .pred || ty == .pred then pure (t.setSlot r.slot (if v &&& 1 == 1 then 1 else 0))
    else pure (t.setSlot r.slot (extendTo ty r.ty v))
  | _ => throw "destination is not a register"

/-- `write`, total, for write-backs that cannot fail: a destination that is not a register (`_`)
leaves the thread as it is. A write-back that kept the thread to return it on an error would
share its register file with the written one, and every write would copy it. -/
def writeOr (t : Thread) (a : Arg) (ty : Ty) (v : UInt64) : Thread :=
  match a with
  | .reg r =>
    if r.ty == .pred || ty == .pred then t.setSlot r.slot (if v &&& 1 == 1 then 1 else 0)
    else t.setSlot r.slot (extendTo ty r.ty v)
  | _ => t

def write128 (t : Thread) (a : Arg) (v : UInt64 × UInt64) : Except String Thread :=
  match a with
  | .sink => pure t
  | .reg r =>
    if r.ty.bits > 64 then pure ((t.setSlot r.slot v.1).setSlot (r.slot + 1) v.2)
    else pure (t.setSlot r.slot v.1)
  | _ => throw "destination is not a register"

/-- `write128`, total (`writeOr`). -/
def writeOr128 (t : Thread) (a : Arg) (v : UInt64 × UInt64) : Thread :=
  match a with
  | .reg r =>
    if r.ty.bits > 64 then (t.setSlot r.slot v.1).setSlot (r.slot + 1) v.2
    else t.setSlot r.slot v.1
  | _ => t

/-- The registers of a vector operand (`{a, b}`) or of a `.vN` register, in order. -/
def elems (a : Arg) (n : Nat) : Except String (Array Arg) :=
  match a with
  | .vec xs => if xs.size == n then pure xs else throw s!"{xs.size} registers where {n} expected"
  | .reg r =>
    if n == 1 then pure #[a]
    else pure ((Array.range n).map fun k => .reg ⟨r.slot + k * (if r.ty.bits > 64 then 2 else 1), r.ty⟩)
  | a => if n == 1 then pure #[a] else throw "not a vector operand"

end Thread

/-- What a memory operand addresses. -/
inductive Target where
  | mem (s : NV.Space) (addr : UInt64)
  /-- The frame's parameter area. -/
  | frame (off : Nat)

/-- The space an instruction's state-space qualifier names; `none` for generic addressing. -/
def spaceOf : Space → Option NV.Space
  | .global => some .global
  | .shared | .sharedCta => some .shared
  | .sharedCluster => some .sharedCluster
  | .local => some .local
  | .const => some (.const 0)
  | .param | .paramEntry | .paramFunc => some .param
  | _ => none

/-- The target of address operand `a` of an instruction with state space `sp` (`none`:
generic, resolved through the windows, §6.4.1.1). -/
def Thread.target (e : Env) (t : Thread) (sp : Option Space) (a : Arg) : Except String Target := do
  let .addr base off := a | throw "not an address operand"
  let raw ← match base with
    | .imm => pure (Sum.inl (UInt64.ofInt off))
    | .reg r => pure (Sum.inl (t.slot r.slot + UInt64.ofInt off))
    | .loc (.frame o) => pure (Sum.inr (Int.toNat (o + off)))
    | .loc (.mem s x) => pure (Sum.inl (if sp.isNone then e.windows.toGeneric s x + UInt64.ofInt off
                                        else x + UInt64.ofInt off))
  match raw with
  | .inr o => return .frame o
  | .inl x =>
    match sp with
    | none => let (s, a) := e.windows.resolve x; return .mem s a
    | some sp =>
      match spaceOf sp with
      | some .param =>
        -- A function's `.param` operand through a register: the frame's parameter area
        -- holds device-function parameters; a kernel's are in the parameter space.
        if (e.prog.funcs[t.frame.fn]!).entry then return .mem .param x else return .frame x.toNat
      | some s => return .mem s x
      | none => throw s!".{Keyword.name sp} is not a memory space"

/-- The value type of an atomic or reduction on PTX type `t`. -/
def atomType (t : Ty) : Option NV.AtomType :=
  match t with
  | .b32 => some .b32 | .b64 => some .b64 | .u32 => some .u32 | .s32 => some .s32
  | .u64 => some .u64 | .s64 => some .s64 | .f32 => some .f32 | .f64 => some .f64
  | .f16 => some .f16 | .bf16 => some .bf16 | .f16x2 => some .f16x2 | .bf16x2 => some .bf16x2
  | _ => none

/-- The memory model's atomic operation for a PTX reduction operator. -/
def atomOp : RedOp → NV.AtomOp
  | .add => .add | .inc => .inc | .dec => .dec | .min => .min | .max => .max
  | .and => .and | .or => .or | .xor => .xor | .exch => .exch | .cas => .cas

end NV.PTX.Sem
