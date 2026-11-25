import NV.PTX.Sem.Operand
import NV.PTX.Sem.Int
import NV.PTX.Sem.Float
import NV.PTX.Sem.Matrix
import NV.PTX.Sem.Video
import NV.Machine.Printf
import NV.PTX.Async.Exec

/-!
# One step of a PTX thread

`Thread.step` executes the thread's next instruction and returns the `Effect` it asks of the
machine: register-only instructions return the next state; memory instructions a
one-lane request whose response is written back by `wb`; barriers a `sync`; warp collectives a
`collect`; `exit` and the end of the kernel `exit`. An instruction whose PTX behaviour is
undefined (division by zero, a misaligned access, an unmapped generic address) or that this
semantics does not define yet is a `fault` naming it, never a guess.
-/

namespace NV.PTX.Sem

open NV (Effect Req Resp SyncOp Collective)
open NV.Num (Format Rounding)

abbrev X := Except String

/-- Bytes `[off, off + n)` of the frame's parameter area as a little-endian number. -/
def frameRead (f : Frame) (off n : Nat) : UInt64 :=
  (List.range (Nat.min n 8)).foldl (fun v k =>
    v ||| ((f.params.get! (off + k)).toUInt64 <<< (8 * k).toUInt64)) 0

def frameWrite (f : Frame) (off n : Nat) (v : UInt64) : Frame :=
  { f with params := (List.range (Nat.min n 8)).foldl (fun ps k =>
      ps.set! (off + k) (v >>> (8 * k).toUInt64).toUInt8) f.params }

/-- `dst` with bytes `[dOff, dOff + n)` replaced by `src`'s from `sOff`. -/
def copyBytes (dst : ByteArray) (dOff : Nat) (src : ByteArray) (sOff n : Nat) : ByteArray :=
  (List.range n).foldl (fun d j => d.set! (dOff + j) (src.get! (sOff + j))) dst

/-- Words of a little-endian value of `bytes` bytes, starting at word-aligned byte 0. -/
def toWords (bytes : Nat) (v : UInt64) : Array UInt32 :=
  if bytes ≤ 4 then #[v.toUInt32] else #[v.toUInt32, (v >>> 32).toUInt32]

/-- Element `k` of `size` bytes from response words. -/
def fromWords (ws : Array UInt32) (size k : Nat) : UInt64 :=
  (List.range size).foldl (fun v j =>
    let b := k * size + j
    let byte := ((ws[b / 4]?.getD 0) >>> (8 * (b % 4)).toUInt32) &&& 0xFF
    v ||| (byte.toUInt64 <<< (8 * j).toUInt64)) 0

/-- Pack `size`-byte elements into words, little-endian. -/
def packWords (size : Nat) (vs : Array UInt64) : Array UInt32 := Id.run do
  let total := size * vs.size
  let mut ws := Array.replicate ((total + 3) / 4) (0 : UInt32)
  for k in [0:vs.size] do
    for j in [0:size] do
      let b := k * size + j
      let byte := ((vs[k]! >>> (8 * j).toUInt64) &&& 0xFF).toUInt32
      ws := ws.set! (b / 4) (ws[b / 4]! ||| (byte <<< (8 * (b % 4)).toUInt32))
  return ws

def memSem (i : Instr) : NV.Sem :=
  match i.sem? with
  | some .relaxed => .relaxed | some .acquire => .acquire | some .release => .release
  | some .acqRel => .acqRel | some .volatile => .volatile
  | _ => .weak

def memScope (i : Instr) : Scope :=
  match i.sem?, i.scope? with
  | some .volatile, _ => .sys
  | _, some s => s
  | some .relaxed, none | some .acquire, none | some .release, none | some .acqRel, none => .gpu
  | _, none => .thread

def hint (i : Instr) : NV.CacheHint :=
  { ops := (i.quals.filter fun
      | .cop _ | .evict _ => true
      | .flag f => [Flag.nc, .l2CacheHint, .l2Size64, .l2Size128, .l2Size256, .unified].contains f
      | _ => false).toArray.map Qual.spelling }

/-- A one-lane request for `bytes` bytes at `addr`, issued with the instruction's
qualifiers. A 32-byte access (`.v8.b32`, sm_100a) is two 16-byte pieces, the lower first: the
memory model's widths stop at 16 bytes (`NV.Memory.Flat`), and its words come back in order. -/
def request (i : Instr) (sp : NV.Space) (addr : UInt64) (bytes : Nat) (op : NV.MemOp)
    (data : Array UInt32 := #[]) : X Req := do
  unless bytes ∈ [1, 2, 4, 8, 16, 32] do throw s!"{bytes}-byte accesses are not modelled"
  unless addr % bytes.toUInt64 == 0 do throw s!"address {addr} is not aligned to {bytes} bytes"
  let (bytes, addrs) := if bytes == 32 then (16, #[some addr, some (addr + 16)]) else (bytes, #[some addr])
  return { space := sp, op, bytes, sem := memSem i, scope := memScope i, hint := hint i, addrs, data }

namespace Step

variable (e : Env) (t : Thread) (pi : PInstr)

def arg (k : Nat) : Arg := pi.args[k]?.getD .sink
def ty (k : Nat := 0) : Ty := pi.i.types.getD k .b32

def rd (k : Nat) (t' : Ty) : X UInt64 := t.read e (arg pi k) t'
def wr (s : Thread) (k : Nat) (t' : Ty) (v : UInt64) : X Thread := s.write (arg pi k) t' v

/-- A register result: write `v` to operand 0 of `p` and go on. -/
def set1 (p : PInstr) (s : Thread) (v : UInt64) (t' : Ty := ty p) : X (Effect Thread) := do
  return .next (← wr p s.advance 0 t' v)

/-- A predicate result to operand 0, which may be a `p|q` pair taking `v` and `w`. -/
def setPreds (s : Thread) (v w : Bool) : X Thread := do
  match arg pi 0 with
  | .pair p q => do
    let s ← s.write p .pred (if v then 1 else 0)
    s.write q .pred (if w then 1 else 0)
  | a => s.write a .pred (if v then 1 else 0)

/-- Integer arithmetic, logic, comparison and selection. -/
def intOp : X (Effect Thread) := do
  let t0 := ty pi
  let w := t0.bits
  let r := rd e t pi
  let flag (f : Flag) := pi.i.has f
  let mode : Flag := if flag .hi then .hi else if flag .wide then .wide else .lo
  match pi.i.op with
  | .add =>
    if flag .cc then
      let s := (← r 1 t0).toNat + (← r 2 t0).toNat
      let s' := (← wr pi t.advance 0 t0 (UInt64.ofNat s))
      return .next (s'.setFrame { s'.frame with carry := s ≥ 2 ^ w })
    set1 pi t (Int.add t0 (flag .sat) (← r 1 t0) (← r 2 t0))
  | .sub =>
    if flag .cc then
      let a := (← r 1 t0).toNat
      let b := (← r 2 t0).toNat
      let s' ← wr pi t.advance 0 t0 (ofInt w (a - b : Int))
      return .next (s'.setFrame { s'.frame with carry := a < b })
    set1 pi t (Int.sub t0 (flag .sat) (← r 1 t0) (← r 2 t0))
  | .addc =>
    let cf := if t.frame.carry then 1 else 0
    let s := (← r 1 t0).toNat + (← r 2 t0).toNat + cf
    let s' ← wr pi t.advance 0 t0 (UInt64.ofNat s)
    return .next (if flag .cc then s'.setFrame { s'.frame with carry := s ≥ 2 ^ w } else s')
  | .subc =>
    let cf := if t.frame.carry then 1 else 0
    let a := (← r 1 t0).toNat
    let b := (← r 2 t0).toNat + cf
    let s' ← wr pi t.advance 0 t0 (ofInt w (a - b : Int))
    return .next (if flag .cc then s'.setFrame { s'.frame with carry := a < b } else s')
  | .mul =>
    let td := if mode == .wide then (Ty.bitsOfWidth (2 * w)).getD t0 else t0
    set1 pi t (Int.mul t0 mode (← r 1 t0) (← r 2 t0)) td
  | .mad | .madc =>
    let wide := mode == .wide
    let tc := if wide then (Ty.bitsOfWidth (2 * w)).getD t0 else t0
    if flag .cc || pi.i.op == .madc then
      let p := Int.prod t0 (← r 1 t0) (← r 2 t0)
      let half := if mode == .hi then p >>> w else p
      let cf := if pi.i.op == .madc && t.frame.carry then 1 else 0
      let s := (ofInt w half).toNat + (← r 3 t0).toNat + cf
      let s' ← wr pi t.advance 0 t0 (UInt64.ofNat s)
      return .next (if flag .cc then s'.setFrame { s'.frame with carry := s ≥ 2 ^ w } else s')
    set1 pi t (Int.mad t0 mode (flag .sat) (← r 1 t0) (← r 2 t0) (← r 3 tc)) tc
  | .mul24 => set1 pi t (Int.mad24 t0 (flag .hi) false (← r 1 t0) (← r 2 t0) 0)
  | .mad24 => set1 pi t (Int.mad24 t0 (flag .hi) (flag .sat) (← r 1 t0) (← r 2 t0) (← r 3 t0))
  | .sad => set1 pi t (Int.sad t0 (← r 1 t0) (← r 2 t0) (← r 3 t0))
  | .div =>
    match Int.div t0 (← r 1 t0) (← r 2 t0) with
    | some v => set1 pi t v
    | none => throw "division by zero (unspecified by the ISA)"
  | .rem =>
    match Int.rem t0 (← r 1 t0) (← r 2 t0) with
    | some v => set1 pi t v
    | none => throw "remainder by zero (unspecified by the ISA)"
  | .abs => set1 pi t (Int.abs t0 (← r 1 t0))
  | .neg => set1 pi t (Int.neg t0 (← r 1 t0))
  | .min => set1 pi t (Int.min t0 (flag .relu) (← r 1 t0) (← r 2 t0))
  | .max => set1 pi t (Int.max t0 (flag .relu) (← r 1 t0) (← r 2 t0))
  | .popc => set1 pi t (Int.popc w (← r 1 t0)) .u32
  | .clz => set1 pi t (Int.clz w (← r 1 t0)) .u32
  | .bfind => set1 pi t (Int.bfind t0 (flag .shiftamt) (← r 1 t0)) .u32
  | .brev => set1 pi t (Int.brev w (← r 1 t0))
  | .bfe => set1 pi t (Int.bfe t0 (← r 1 t0) (← r 2 .u32) (← r 3 .u32))
  | .bfi => set1 pi t (Int.bfi w (← r 1 t0) (← r 2 t0) (← r 3 .u32) (← r 4 .u32))
  | .fns =>
    match Int.fns (← r 1 .b32) (← r 2 .u32) (toInt true 32 (← r 3 .s32)) with
    | some v => set1 pi t v .b32
    | none => throw "fns base above 31 (undefined)"
  | .szext => set1 pi t (Int.szext t0 (flag .clamp) (← r 1 t0) (← r 2 .u32))
  | .bmsk => set1 pi t (Int.bmsk (flag .clamp) (← r 1 .b32) (← r 2 .b32)) .b32
  | .dp4a =>
    let sa := t0.isSigned
    let sb := (ty pi 1).isSigned
    set1 pi t (Int.dp4a sa sb (← r 1 .b32) (← r 2 .b32) (← r 3 .b32)) (if sa || sb then .s32 else .u32)
  | .dp2a =>
    let sa := t0.isSigned
    let sb := (ty pi 1).isSigned
    set1 pi t (Int.dp2a sa sb (flag .hi) (← r 1 .b32) (← r 2 .b32) (← r 3 .b32))
      (if sa || sb then .s32 else .u32)
  | .clmad => set1 pi t (Int.clmad (flag .hi) (← r 1 .u64) (← r 2 .u64) (← r 3 .u64)) .u64
  | .and | .or | .xor =>
    let a ← r 1 t0
    let b ← r 2 t0
    let v := match pi.i.op with | .and => a &&& b | .or => a ||| b | _ => a ^^^ b
    set1 pi t v
  | .not =>
    let a ← r 1 t0
    set1 pi t (if t0 == .pred then (if a == 0 then 1 else 0) else ~~~a)
  | .cnot => set1 pi t (if (← r 1 t0) == 0 then 1 else 0)
  | .lop3 =>
    let d := Int.lop3 (← r 4 .b32) (← r 1 .b32) (← r 2 .b32) (← r 3 .b32)
    match pi.i.bool?, arg pi 0 with
    | some op, .pair dst p =>
      let s ← t.advance.write dst .b32 d
      return .next (← s.write p .pred (if Int.boolOp op (d != 0) ((← r 5 .pred) != 0) then 1 else 0))
    | _, _ => set1 pi t d .b32
  | .shf => set1 pi t (Int.shf (flag .l) (flag .clamp) (← r 1 .b32) (← r 2 .b32) (← r 3 .u32)) .b32
  | .shl => set1 pi t (Int.shl w (← r 1 t0) (← r 2 .u32))
  | .shr => set1 pi t (Int.shr t0 (← r 1 t0) (← r 2 .u32))
  | .prmt =>
    let m := [Flag.f4e, .b4e, .rc8, .ecl, .ecr, .rc16].find? flag
    set1 pi t (Int.prmt m (← r 1 .b32) (← r 2 .b32) (← r 3 .b32)) .b32
  | .setp =>
    let some c := pi.i.cmp? | throw "setp without a comparison"
    let some v := Int.cmp t0 c (← r 1 t0) (← r 2 t0) | throw s!"{repr c} on an integer type"
    match pi.i.bool? with
    | some op =>
      let q := (← r 3 .pred) != 0
      return .next (← setPreds pi t.advance (Int.boolOp op v q) (Int.boolOp op (!v) q))
    | none => return .next (← setPreds pi t.advance v (!v))
  | .set =>
    let s := ty pi 1
    let some c := pi.i.cmp? | throw "set without a comparison"
    let some v := Int.cmp s c (← r 1 s) (← r 2 s) | throw s!"{repr c} on an integer type"
    let v := match pi.i.bool? with
      | some op => (do pure (Int.boolOp op v ((← r 3 .pred) != 0)) : X Bool)
      | none => pure v
    let v ← v
    if t0 == .f32 then set1 pi t (if v then 0x3F800000 else 0) .f32
    else set1 pi t (if v then 0xFFFFFFFF else 0) t0
  | .selp =>
    let a ← r 1 t0
    let b ← r 2 t0
    set1 pi t (if (← r 3 .pred) != 0 then a else b)
  | .slct =>
    let a ← r 1 t0
    let b ← r 2 t0
    set1 pi t (if toInt true 32 (← r 3 .s32) ≥ 0 then a else b)
  | op => throw s!"{op.spelling} is not an integer instruction"

/-- `mov`, including vector pack and unpack (§9.7.10.3, §9.7.10.4). -/
def mov : X (Effect Thread) := do
  let t0 := ty pi
  match arg pi 0, arg pi 1 with
  | .vec ds, a =>
    let n := ds.size
    let w := t0.bits / n
    let some et := Ty.bitsOfWidth w | throw "malformed unpack"
    if t0 == .b128 then
      let (lo, hi) ← t.read128 e a
      let mut s := t.advance
      for k in [0:n] do
        let v := if w * k < 64 then mask w (lo >>> (w * k).toUInt64)
          else mask w (hi >>> (w * k - 64).toUInt64)
        s ← s.write ds[k]! et v
      return .next s
    let v ← t.read e a t0
    let mut s := t.advance
    for k in [0:n] do s ← s.write ds[k]! et (mask w (v >>> (w * k).toUInt64))
    return .next s
  | d, .vec xs =>
    let n := xs.size
    let w := t0.bits / n
    let some et := Ty.bitsOfWidth w | throw "malformed pack"
    let mut lo : UInt64 := 0
    let mut hi : UInt64 := 0
    for k in [0:n] do
      let v ← t.read e xs[k]! et
      if w * k < 64 then lo := lo ||| (v <<< (w * k).toUInt64)
      else hi := hi ||| (v <<< (w * k - 64).toUInt64)
    if t0 == .b128 then return .next (← t.advance.write128 d (lo, hi))
    return .next (← t.advance.write d t0 lo)
  | d, a =>
    if t0 == .b128 then return .next (← t.advance.write128 d (← t.read128 e a))
    return .next (← t.advance.write d t0 (← t.read e a t0))

/-! ## Floating point -/

/-- Floating-point arithmetic, comparison and selection (§9.7.3–§9.7.8). -/
def floatOp : X (Effect Thread) := do
  let t0 := ty pi
  let flag (f : Flag) := pi.i.has f
  let ftz := flag .ftz
  let rnd : Rounding := (pi.i.rnd?.bind Float.rounding).getD .rn
  let rf (k : Nat) (t' : Ty := t0) : X UInt64 := t.readF e (arg pi k) t'
  let fin (f : Format) (x : Nat) : Nat :=
    let x := if flag .relu then NV.Num.relu f x else x
    if flag .sat then NV.Num.satUnit f x else x
  let lanes (op : Format → List Nat → Nat) (args : List UInt64) : X UInt64 :=
    match Float.map t0 op args with
    | some v => pure v
    | none => throw s!".{Keyword.name t0} is not a floating-point type"
  let un (op : Format → Nat → Nat) : X (Effect Thread) := do
    set1 pi t (← lanes (fun f xs => op f (xs.headD 0)) [← rf 1])
  let bin (op : Format → Nat → Nat → Nat) : X (Effect Thread) := do
    set1 pi t (← lanes (fun f xs => fin f (op f (xs.getD 0 0) (xs.getD 1 0))) [← rf 1, ← rf 2])
  -- Mixed precision (§9.7.5): `.f32.f16`/`.f32.bf16` take `a` (and `b`) in the narrow format.
  let mixed : Option Format := match pi.i.types with
    | [.f32, .f16] => some .f16 | [.f32, .bf16] => some .bf16 | _ => none
  let widen (f : Format) (k : Nat) : X Nat := do
    return NV.Num.cvt f .f32 .rn {} (← rd e t pi k (if f == .f16 then .f16 else .bf16)).toNat
  match pi.i.op with
  | .add | .sub =>
    let op := if pi.i.op == .add then NV.Num.add else NV.Num.sub
    match mixed with
    | some f => set1 pi t (fin .f32 (op .f32 rnd false (← widen f 1) (← rf 2 .f32).toNat)).toUInt64 .f32
    | none => bin fun f a b => op f rnd ftz a b
  | .mul => bin fun f a b => NV.Num.mul f rnd ftz a b
  | .fma | .mad =>
    match mixed with
    | some f =>
      set1 pi t (fin .f32 (NV.Num.fma .f32 rnd false (← widen f 1) (← widen f 2)
        (← rf 3 .f32).toNat)).toUInt64 .f32
    | none =>
      set1 pi t (← lanes (fun f xs => fin f (NV.Num.fma f rnd ftz (xs.getD 0 0) (xs.getD 1 0)
        (xs.getD 2 0))) [← rf 1, ← rf 2, ← rf 3])
  | .div =>
    if flag .approx then bin fun _ a b => NV.Num.Approx.divApprox ftz a b
    else if flag .full then bin fun _ a b => NV.Num.Approx.divFull ftz a b
    else bin fun f a b => NV.Num.div f rnd ftz a b
  | .rcp =>
    if flag .approx then
      if t0 == .f64 then set1 pi t (NV.Num.Approx.rcp NV.Num.Approx.f64Hi ftz (← rf 1).toNat).toUInt64
      else un fun f a => NV.Num.Approx.rcp f ftz a
    else un fun f a => NV.Num.rcp f rnd ftz a
  | .sqrt =>
    if flag .approx then un fun _ a => NV.Num.Approx.sqrt ftz a
    else un fun f a => NV.Num.sqrt f rnd ftz a
  | .rsqrt =>
    if t0 == .f64 && ftz then
      set1 pi t (NV.Num.Approx.rsqrt NV.Num.Approx.f64Hi ftz (← rf 1).toNat).toUInt64
    else un fun f a => NV.Num.Approx.rsqrt f ftz a
  | .sin => un fun _ a => NV.Num.Approx.sinCos false false ftz a
  | .cos => un fun _ a => NV.Num.Approx.sinCos true false ftz a
  | .lg2 => un fun _ a => NV.Num.Approx.lg2 ftz a
  | .ex2 => un fun f a => NV.Num.Approx.ex2 f ftz a
  | .tanh => un fun f a => NV.Num.Approx.tanh f a
  | .abs => un fun f a => NV.Num.abs f ftz a
  | .neg => un fun f a => NV.Num.neg f ftz a
  | .copysign => bin fun f a b => NV.Num.copysign f a b
  | .min | .max =>
    let isMax := pi.i.op == .max
    if pi.args.size == 4 then
      -- `min{.ftz}{.NaN}{.abs}.f32 d, a, b, c` (§9.7.3.11): the extremum of the three, of their
      -- absolute values with `.abs`.
      let mm (f : Format) (x y : Nat) := NV.Num.minMax isMax f ftz (flag .nan) false x y
      let ab (f : Format) (x : Nat) := if flag .abs then NV.Num.abs f false x else x
      set1 pi t (← lanes (fun f xs => mm f (mm f (ab f (xs.getD 0 0)) (ab f (xs.getD 1 0)))
        (ab f (xs.getD 2 0))) [← rf 1, ← rf 2, ← rf 3])
    else bin fun f a b => NV.Num.minMax isMax f ftz (flag .nan) (flag .xorsign) a b
  | .testp =>
    let c : NV.Num.TestClass ← if flag .finite then pure .finite else if flag .infinite then
      pure .infinite else if flag .number then pure .number else if flag .notanumber then
      pure .notanumber else if flag .normal then pure .normal else if flag .subnormal then
      pure .subnormal else throw "testp without a class"
    let some (f, _) := Float.lanes? t0 | throw "testp on a non-float type"
    set1 pi t (if NV.Num.testp f c (← rf 1).toNat then 1 else 0) .pred
  | .setp | .set =>
    let st := if pi.i.op == .set then ty pi 1 else t0
    let some c := pi.i.cmp?.bind Float.cmpOp | throw "not a floating-point comparison"
    let some (f, n) := Float.lanes? st | throw "comparison on a non-float type"
    let a ← rf 1 st
    let b ← rf 2 st
    let w := Float.laneBits f st
    let r (k : Nat) := NV.Num.cmp f c ftz (Float.lane w k a) (Float.lane w k b)
    let comb (v : Bool) : X Bool := do
      match pi.i.bool? with
      | some op => pure (Int.boolOp op v ((← rd e t pi 3 .pred) != 0))
      | none => pure v
    if pi.i.op == .setp then
      if n == 1 then
        let v := r 0
        return .next (← setPreds pi t.advance (← comb v) (← comb (!v)))
      return .next (← setPreds pi t.advance (← comb (r 0)) (← comb (r 1)))
    -- `set`: all ones (integers) or 1.0 (floats) per lane where the comparison holds.
    let dt := t0
    let one (g : Format) : Nat := g.pack false g.bias 0
    let v ← match Float.lanes? dt with
      | some (g, m) =>
        let gw := Float.laneBits g dt
        let mut acc : UInt64 := 0
        for k in [0:m] do
          let bit ← comb (r (if n == 1 then 0 else k))
          acc := acc ||| ((if bit then (one g).toUInt64 else 0) <<< (gw * k).toUInt64)
        pure acc
      | none =>
        if n == 1 then pure (if ← comb (r 0) then mask dt.bits 0xFFFFFFFFFFFFFFFF else 0)
        else
          let lo ← comb (r 0)
          let hi ← comb (r 1)
          pure ((if lo then 0xFFFF else 0) ||| (if hi then 0xFFFF0000 else 0))
    set1 pi t v dt
  | .slct =>
    let a ← rd e t pi 1 t0
    let b ← rd e t pi 2 t0
    let c ← rf 3 .f32
    set1 pi t (if NV.Num.cmp .f32 .ge ftz c.toNat 0 then a else b)
  | op => throw s!"{op.spelling} is not a floating-point instruction"

/-- Float conversions (§9.7.10.24): float to float, float to integer, integer to float,
float to an integral float, and the packed forms. -/
def cvtFloat : X (Effect Thread) := do
  let some (d, s) := match pi.i.types with | [d, s] => some (d, s) | _ => none
    | throw "cvt needs two types"
  let flag (f : Flag) := pi.i.has f
  let ftz := flag .ftz
  let o : NV.Num.CvtOpts :=
    { ftz, sat := flag .sat, relu := flag .relu, satfinite := flag .satfinite, pzo := flag .pzo }
  let rnd? := pi.i.rnd?.bind Float.rounding
  if pi.i.rnd? == some .rs then
    -- `cvt.rs{.relu}{.satfinite}.{f16x2,bf16x2}.f32 d, a, b, rbits`: the upper half of `rbits`
    -- rounds `a` (into the upper lane), the lower half `b`; `.f16` uses 13 of the 16 bits.
    -- The four-lane forms say which half of `rbits` feeds `a, b` and which `e, f`, not how a
    -- half divides between its two inputs, so they are not modelled.
    let some (fd, 2) := Float.lanes? d
      | throw s!"cvt.rs to .{Keyword.name d}: the ISA leaves the split of rbits open"
    let n := if fd == .f16 then 13 else 16
    let rb ← rd e t pi 3 .b32
    let one (k : Nat) (half : UInt64) : X Nat := do
      return NV.Num.cvtStochastic .f32 fd (mask n half).toNat n o (← rd e t pi k .f32).toNat
    let hi ← one 1 (rb >>> 16)
    let lo ← one 2 rb
    return ← set1 pi t (Float.pack 16 [lo, hi]) d
  -- `.scaled::n1::ue8m0` (one scale for every lane) and `.scaled::n2::ue8m0` (the upper byte
  -- for the upper lane); `scaleArg` is the operand position, the default 0x7f7f (1.0) when
  -- absent. The conversion itself is `NV.Num.cvtScaled`.
  let scaled := flag .scaledN1Ue8m0 || flag .scaledN2Ue8m0
  let scaleOf (scaleArg : Nat) (lane : Nat) : X Nat := do
    let sf ← if pi.args.size > scaleArg then rd e t pi scaleArg .b16 else pure 0x7F7F
    return (if flag .scaledN1Ue8m0 then sf &&& 0xFF else (sf >>> (8 * lane).toUInt64) &&& 0xFF).toNat
  let conv (fs fd : Format) (r : Rounding) (scaleArg lane : Nat) (x : Nat) : X Nat := do
    if !scaled then return NV.Num.cvt fs fd r o x
    return NV.Num.cvtScaled fs fd r o (← scaleOf scaleArg lane) (fd.width < fs.width)
      (flag .scaledN1Ue8m0) x
  -- `.s2f6x2`, PTX's 8-bit fixed point (§5.2.4), in and out.
  if d == .s2f6x2 || s == .s2f6x2 then
    let some r := rnd? | throw "an .s2f6x2 cvt needs a rounding modifier"
    if d == .s2f6x2 then
      let srcs : List (Format × Nat) ← match Float.lanes? s with
        | some (fs, 1) => do pure [(fs, (← rd e t pi 2 s).toNat), (fs, (← rd e t pi 1 s).toNat)]
        | some (fs, 2) => do
          let a ← rd e t pi 1 s
          pure [(fs, Float.lane 16 0 a), (fs, Float.lane 16 1 a)]
        | _ => throw s!"cvt.s2f6x2.{Keyword.name s} is not a conversion the ISA lists"
      let scaleArg := if (Float.lanes? s).map (·.2) == some 1 then 3 else 2
      let mut vals := []
      for ((fs, x), k) in srcs.zipIdx do
        vals := vals ++ [NV.Num.toS2f6 fs r (flag .relu) (← scaleOf scaleArg k) x]
      return ← set1 pi t (Float.pack 8 vals) d
    let some (fd, 2) := Float.lanes? d | throw "cvt from .s2f6x2 goes to .bf16x2"
    let a ← rd e t pi 1 s
    let mut vals := []
    for k in [0:2] do
      vals := vals ++ [NV.Num.ofS2f6 fd o (← scaleOf 2 k) (Float.lane 8 k a)]
    return ← set1 pi t (Float.pack 16 vals) d
  match Float.lanes? d, Float.lanes? s with
  -- integer to float
  | some (fd, 1), none =>
    let some r := rnd? | throw "integer-to-float cvt needs a rounding modifier"
    let v := NV.Num.ofInt fd r s.isSigned s.bits (← rd e t pi 1 s).toNat
    set1 pi t (o.finish fd v).toUInt64 d
  -- float to integer: rounding is mandatory, the result saturates (§9.7.10.24)
  | none, some (fs, 1) =>
    let some ir := pi.i.irnd? | throw "float-to-integer cvt needs an integer rounding modifier"
    let v := NV.Num.toInt fs (Float.intRounding ir) d.isSigned d.bits ftz (← t.readF e (arg pi 1) s).toNat
    set1 pi t v.toUInt64 d
  -- float to float, one element
  | some (fd, 1), some (fs, 1) =>
    let a := (← t.readF e (arg pi 1) s).toNat
    match pi.i.irnd? with
    | some ir =>
      if fd != fs then throw "integer rounding between different float formats"
      set1 pi t (o.finish fd (NV.Num.roundInt fd (Float.intRounding ir) ftz a)).toUInt64 d
    | none =>
      let r ← match rnd? with
        | some r => pure r
        | none => if fd.width ≥ fs.width then pure .rn else throw "narrowing cvt needs a rounding modifier"
      set1 pi t (NV.Num.cvt fs fd r o a).toUInt64 d
  -- two (or four) scalars packed into a vector: `a` lands in the upper lane (§9.7.10.24)
  | some (fd, n), some (fs, 1) =>
    let some r := rnd? | throw "packing cvt needs a rounding modifier"
    let srcs ← if n == 4 then do
        let xs ← Thread.elems (arg pi 1) 4
        xs.toList.mapM (t.readF e · s)
      else (List.range n).mapM fun k => rd e t pi (k + 1) s
    let w := Float.laneBits fd d
    let mut vals := []
    for (x, k) in srcs.reverse.zipIdx do
      vals := vals ++ [← conv fs fd r (n + 1) k x.toNat]
    set1 pi t (Float.pack w vals) d
  -- a packed vector converted lane by lane, upper to upper
  | some (fd, n), some (fs, m) =>
    if n != m then throw "lane counts differ"
    let some r := rnd? | throw "packed cvt needs a rounding modifier"
    let a ← rd e t pi 1 s
    let ws := Float.laneBits fs s
    let wd := Float.laneBits fd d
    let mut vals := []
    for k in [0:n] do
      vals := vals ++ [← conv fs fd r 2 k (Float.lane ws k a)]
    set1 pi t (Float.pack wd vals) d
  | _, _ => throw s!"cvt.{Keyword.name d}.{Keyword.name s} is not modelled"

/-- `cvt.pack.sat.convertType.s32[.b32] d, a, b[, c]` (§9.7.10.25): a and b saturated to
`convertType` and packed into d's low bits, b lowest; with c (the sub-16-bit types), d's
remaining bits are c's low bits. -/
def cvtPack : X (Effect Thread) := do
  unless pi.i.has .sat do throw "cvt.pack without .sat"
  let ct ← match pi.i.types with
    | [ct, .s32] | [ct, .s32, .b32] => pure ct
    | _ => throw "cvt.pack needs .convertType.s32 (and .b32 for the sub-16-bit types)"
  let n := ct.bits
  let mask : UInt64 := (1 <<< n.toUInt64) - 1
  let a := Int.cvt ct .s32 true (← rd e t pi 1 .s32) &&& mask
  let b := Int.cvt ct .s32 true (← rd e t pi 2 .s32) &&& mask
  let low := (a <<< n.toUInt64) ||| b
  let d ← if n == 16 then pure low
    else do pure (low ||| ((← rd e t pi 3 .b32) <<< (2 * n).toUInt64))
  set1 pi t (d &&& 0xffffffff) .b32

/-- Integer-to-integer and bit-identity conversions (`cvt`, §9.7.10.24). -/
def cvt : X (Effect Thread) := do
  match pi.i.types with
  | [d, s] =>
    if (d.isInt || d.basic == .bits) && (s.isInt || s.basic == .bits) && pi.i.rnd?.isNone &&
        pi.i.irnd?.isNone then
      set1 pi t (Int.cvt d s (pi.i.has .sat) (← rd e t pi 1 s)) d
    else cvtFloat e t pi
  | _ => throw "cvt needs two types"



/-! ## Video instructions (§9.7.20) -/

/-- An operand without its selector and negation: the register, the selector, negated. -/
def unwrap : Arg → Arg × Option String × Bool
  | .neg a => let (r, s, _) := unwrap a; (r, s, true)
  | .sel a s => (a, some s, false)
  | a => (a, none, false)

def videoOp : X (Effect Thread) := do
  let flag (f : Flag) := pi.i.has f
  let raw (k : Nat) : X (UInt64 × Option String × Bool) := do
    let (a, s, n) := unwrap (arg pi k)
    return (← t.read e a .b32, s, n)
  let (dArg, dsel, _) := unwrap (arg pi 0)
  let out (v : UInt64) : X (Effect Thread) := do return .next (← t.advance.write dArg .b32 v)
  let c ← if pi.args.size > 3 then (do pure (← raw 3).1) else pure 0
  let secop := pi.i.red?
  let simd2 := pi.i.op matches .vadd2 | .vsub2 | .vavrg2 | .vabsdiff2 | .vmin2 | .vmax2 | .vset2
  let simd4 := pi.i.op matches .vadd4 | .vsub4 | .vavrg4 | .vabsdiff4 | .vmin4 | .vmax4 | .vset4
  if simd2 || simd4 then
    let lanes := if simd2 then 2 else 4
    let (dty, aty, bty) := match pi.i.types with
      | [a, b] => (Ty.u32, a, b) | [d, a, b] => (d, a, b) | _ => (.u32, .u32, .u32)
    let (a, asel, _) ← raw 1
    let (b, bsel, _) ← raw 2
    let some asel := Video.simdSel lanes asel (List.range lanes) | throw "malformed selector"
    let some bsel := Video.simdSel lanes bsel ((List.range lanes).map (· + lanes))
      | throw "malformed selector"
    let some msk := Video.simdMask lanes dsel | throw "malformed mask"
    let some v := Video.simd pi.i.op lanes dty.isSigned aty.isSigned bty.isSigned (flag .sat)
        (secop == some .add) pi.i.cmp? asel bsel msk a b c | throw "not a SIMD video instruction"
    return ← out v
  let dsel' ← match Video.scalarSel dsel with
    | some s => pure (if dsel.isSome then some s else none)
    | none => throw "malformed destination selector"
  let (a, asel, na) ← raw 1
  let (b, bsel, nb) ← raw 2
  let some as := Video.scalarSel asel | throw "malformed selector"
  let some bs := Video.scalarSel bsel | throw "malformed selector"
  match pi.i.op with
  | .vmad =>
    let [_, ta, tb] := pi.i.types | throw "vmad needs three types"
    let (_, _, nc) := unwrap (arg pi 3)
    let shift := if flag .shr7 then 7 else if flag .shr15 then 15 else 0
    out (Video.vmad ta.isSigned tb.isSigned (Video.extract ta.isSigned as a)
      (Video.extract tb.isSigned bs b) (na != nb) nc (flag .po) shift (flag .sat) c)
  | op =>
    let (dty, aty, bty) := match op, pi.i.types with
      | .vset, [a, b] => (Ty.u32, a, b)
      | _, [d, a, b] => (d, a, b)
      | _, _ => (.u32, .u32, .u32)
    let bsigned := if op matches .vshl | .vshr then false else bty.isSigned
    let some tmp := Video.primary op pi.i.cmp? (flag .clamp)
        (Video.extract aty.isSigned as a) (Video.extract bsigned bs b)
      | throw s!"{op.spelling} is not a scalar video instruction"
    out (Video.finish tmp (flag .sat) (dty.isSigned && op != .vset) dsel' secop c)

/-! ## Memory -/

def space (i : Instr) : Option Space := i.space?

/-- `ld`, `ldu` (§9.7.10.8, §9.7.10.10). -/
def ld : X (Effect Thread) := do
  let t0 := ty pi
  let n := (pi.i.vec?.map Vec.length).getD 1
  let size := t0.bytes
  let ds ← Thread.elems (arg pi 0) n
  match ← t.target e (space pi.i) (arg pi 1) with
  | .frame off =>
    let mut s := t.advance
    for k in [0:n] do s ← s.write ds[k]! t0 (frameRead t.frame (off + k * size) size)
    return .next s
  | .mem sp addr =>
    let req ← request pi.i sp addr (n * size) .ld
    let wb (resp : Resp) (s : Thread) : Thread := Id.run do
      let mut s := s
      for k in [0:n] do
        s := if t0 == .b128 then s.writeOr128 ds[k]! (fromWords resp 8 0, fromWords resp 8 1)
          else s.writeOr ds[k]! t0 (fromWords resp size k)
      return s
    return .mem (fun _ => req) t.advance wb

/-- `st` (§9.7.10.11). -/
def st : X (Effect Thread) := do
  let t0 := ty pi
  let n := (pi.i.vec?.map Vec.length).getD 1
  let size := t0.bytes
  let xs ← Thread.elems (arg pi 1) n
  let vals ← xs.mapM fun x =>
    if t0 == .b128 then do let (lo, hi) ← t.read128 e x; pure #[lo, hi]
    else do pure #[← t.read e x t0]
  match ← t.target e (space pi.i) (arg pi 0) with
  | .frame off =>
    let f := (List.range n).foldl (fun f k =>
      frameWrite f (off + k * size) size (vals[k]!)[0]!) t.frame
    return .next (t.setFrame f).advance
  | .mem sp addr =>
    let data := packWords (if t0 == .b128 then 8 else size) (vals.flatMap id)
    let req ← request pi.i sp addr (n * size) .st data
    return .mem (fun _ => req) t.advance fun _ s => s

/-- `atom` and `red` (§9.7.15.5, §9.7.15.6): scalars (`.b16` `cas` on its two bytes); `.b128`
`exch`/`cas`; and the vector forms
(`.v2`/`.v4` of `.f32`, `.f16x2`, `.bf16x2`, `.v2`/`.v4`/`.v8` of `.f16`, `.bf16`), whose elements
are atomics of their own (`AtomType.vec`), 16-bit ones travelling in pairs. -/
def atom (returns : Bool) : X (Effect Thread) := do
  let t0 := ty pi
  let some o := pi.i.red? | throw "atomic without an operation"
  let n := (pi.i.vec?.map Vec.length).getD 1
  let some ety := (if n > 1 && t0 == .f16 then some .f16x2 else if n > 1 && t0 == .bf16 then some .bf16x2
    else if t0 == .b128 then some .b128 else if t0 == .b16 && o == .cas then some .b32
    else atomType t0)
    | throw s!"atomics on .{Keyword.name t0} are not modelled"
  let first := if returns then 1 else 0
  let size := t0.bytes
  let operand (k : Nat) : X (Array UInt32) := do
    let xs ← Thread.elems (arg pi k) n
    if t0 == .b128 then
      let (lo, hi) ← t.read128 e xs[0]!
      return toWords 8 lo ++ toWords 8 hi
    return packWords size (← xs.mapM fun x => t.read e x t0)
  let data ← if o == .cas then do pure ((← operand (first + 1)) ++ (← operand (first + 2)))
    else operand (first + 1)
  let .mem sp addr ← t.target e (space pi.i) (arg pi first) | throw "atomic on a parameter"
  -- `atom.add.f32` without `.noftz` flushes subnormals on global memory, not on shared.
  let ety := if ety == .f32 && sp == .global && !pi.i.has .noftz then .f32ftz else ety
  -- A vector: 32-bit elements, 16-bit ones in pairs.
  let aty := if n > 1 then NV.AtomType.vec ety (n * size / 4) else ety
  let op := atomOp o
  let req ← request pi.i sp addr (n * size) (if returns then .atom op aty else .red op aty) data
  if returns then
    return .mem (fun _ => req) t.advance fun resp s => Id.run do
      if t0 == .b128 then
        return s.writeOr128 (arg pi 0) (fromWords resp 8 0, fromWords resp 8 1)
      let ds := match Thread.elems (arg pi 0) n with | .ok ds => ds | .error _ => #[]
      let mut s := s
      for k in [0:ds.size] do
        s := s.writeOr ds[k]! t0 (fromWords resp size k)
      return s
  return .mem (fun _ => req) t.advance fun _ s => s

/-- `cvta` (§9.7.10.23), `isspacep` (§9.7.10.22), `mapa` (§9.7.10.26), `getctarank`
(§9.7.10.27). -/
def addressOp : X (Effect Thread) := do
  let t0 := ty pi
  let w := e.windows
  let sp := (space pi.i).bind spaceOf
  match pi.i.op with
  | .cvta =>
    let some sp := sp | throw "cvta without a space"
    let a ← rd e t pi 1 .u64
    if pi.i.has .to then
      let (s, off) := w.resolve a
      unless s == sp || (sp == .sharedCluster && s == .shared) do
        throw s!"cvta.to: {a} is not in the .{repr sp} window"
      set1 pi t off t0
    else set1 pi t (w.toGeneric sp a) t0
  | .isspacep =>
    let some sp := sp | throw "isspacep without a space"
    set1 pi t (if w.inWindow sp (← rd e t pi 1 .u64) then 1 else 0) .pred
  | .mapa =>
    let rank := (← rd e t pi 2 .u32).toNat
    let self := t.ctx.clusterRank
    match sp with
    | some _ =>
      let (_, off) := w.sharedTarget (← rd e t pi 1 t0)
      set1 pi t (w.clusterAddr self rank off) t0
    | none =>
      let (_, a) := w.resolve (← rd e t pi 1 .u64)
      let (_, off) := w.sharedTarget a
      set1 pi t (w.toGeneric .sharedCluster (w.clusterAddr self rank off)) t0
  | .getctarank =>
    let a ← rd e t pi 1 t0
    let a := if sp.isSome then a else (w.resolve a).2
    set1 pi t ((w.sharedTarget a).1.getD t.ctx.clusterRank).toUInt64 .u32
  | op => throw s!"{op.spelling} is not an address instruction"

/-! ## Control -/

/-- A call of `vprintf(fmt, buf)` (device `printf`, the CUDA runtime's external function): the
format string is read from memory byte by byte, then the argument buffer as `NV.Printf.layout`
sizes it, one step each (the thread stays at the call meanwhile, its progress in the frame's
`scratch`: format address, buffer address, the format's length once read, then the bytes); then
a record is appended to the thread's `printf` log and the call returns the number of arguments. -/
def vprintf (rets params : Array Arg) : X (Effect Thread) := do
  let f := t.frame
  let param (k : Nat) : X UInt64 := match (params[k]? : Option Arg) with
    | some (.sym (.frame off) o) => pure (frameRead f (Int.toNat (off + o)) 8)
    | some a => t.read e a .u64
    | none => throw "vprintf takes a format and an argument buffer"
  let (fmt, buf) ← if f.micro == 0 then do pure (← param 0, ← param 1)
    else pure (f.scratch[0]!, f.scratch[1]!)
  let st := if f.micro == 0 then #[fmt, buf, 0] else f.scratch
  let strLen := st[2]!.toNat
  let bytes := st.extract 3 st.size
  let resume (st : Array UInt64) (s : Thread) : Thread :=
    s.setFrame { s.frame with micro := s.frame.micro + 1, scratch := st }
  let byteAt (a : UInt64) : X (NV.Space × UInt64) := pure (e.windows.resolve a)
  if strLen == 0 then
    -- The next byte of the format.
    let (sp, a) ← byteAt (fmt + bytes.size.toUInt64)
    let req ← request pi.i sp a 1 .ld
    return .mem (fun _ => req) t fun resp s =>
      let b := (resp[0]?.getD 0).toUInt64 &&& 0xFF
      resume (if b == 0 then st.set! 2 (bytes.size + 1).toUInt64 else st.push b) s
  let fmtStr := String.ofList ((bytes.extract 0 (strLen - 1)).toList.map (Char.ofNat ·.toNat))
  let (slots, n) := NV.Printf.layout fmtStr
  let args := bytes.extract (strLen - 1) bytes.size
  if args.size < n then
    let (sp, a) ← byteAt (buf + args.size.toUInt64)
    let req ← request pi.i sp a 1 .ld
    return .mem (fun _ => req) t fun resp s => resume (st.push ((resp[0]?.getD 0).toUInt64 &&& 0xFF)) s
  let record := (fmtStr, ByteArray.mk (args.map (·.toUInt8)))
  let s := { t with printf := t.printf.push record }
  let s := s.setFrame { s.frame with micro := 0, scratch := #[] }
  let s ← match (rets[0]? : Option Arg) with
    | some (.sym (.frame off) o) =>
      pure (s.setFrame (frameWrite s.frame (Int.toNat (off + o)) 4 slots.length.toUInt64))
    | some a => s.write a .s32 slots.length.toUInt64
    | none => pure s
  return .next s.advance

/-- `call` (§9.7.14.5): push a frame with the actual arguments copied into the formals. -/
def call : X (Effect Thread) := do
  let (rets, target, params) ← match pi.args.toList with
    | [.args rs, f, .args ps] | [.args rs, f, .args ps, _] => pure (rs, f, ps)
    | [.args rs, f] => pure (rs, f, #[])
    | [f, .args ps] | [f, .args ps, _] => pure (#[], f, ps)
    | [f] => pure (#[], f, #[])
    | _ => throw "malformed call"
  let fn ← match target with
    | .func f => pure f
    | a => do
      let v ← t.read e a .u64
      if v ≥ 0xF000_0000 && (v - 0xF000_0000).toNat < e.prog.funcs.size then
        pure (v - 0xF000_0000).toNat
      else throw s!"indirect call to {v}, not a function"
  let some f := e.prog.funcs[fn]? | throw "call to an unknown function"
  if f.code.isEmpty && f.name == "vprintf" then return ← vprintf e t pi rets params
  if f.code.isEmpty then throw s!"call to {f.name}, which has no body in this module"
  let caller := t.advance
  let mut callee : Frame := { fn, pc := 0, regs := Array.replicate f.slots 0,
                              params := ByteArray.mk (Array.replicate f.frameBytes 0),
                              retTo := rets, sp := t.frame.sp }
  for (formal, k) in f.params.zipIdx do
    let some a := params[k]? | continue
    let d := formal.decl
    let v : UInt64 ← match a with
      | .sym (.frame off) o => pure (frameRead caller.frame (Int.toNat (off + o)) d.bytes)
      | a => t.read e a (if d.ty.bits > 64 then .b64 else d.ty)
    callee := match formal.reg with
      | some r => { callee with regs := callee.regs.set! r.slot (extendTo d.ty r.ty v) }
      | none =>
        if d.bytes > 8 then
          match a with
          | .sym (.frame off) o =>
            let ps := copyBytes callee.params formal.off caller.frame.params (Int.toNat (off + o))
              d.bytes
            { callee with params := ps }
          | _ => frameWrite callee formal.off d.bytes v
        else frameWrite callee formal.off d.bytes v
  return .next { caller with frames := caller.frames.push callee }

/-- `ret` (§9.7.14.6): copy the return formals to the caller's operands and pop the frame. -/
def ret : X (Effect Thread) := do
  if t.frames.size ≤ 1 then return .exit
  let callee := t.frame
  let f := e.prog.funcs[callee.fn]!
  let mut s : Thread := { t with frames := t.frames.pop }
  for (formal, k) in f.rets.zipIdx do
    let some dst := callee.retTo[k]? | continue
    let d := formal.decl
    let v := match formal.reg with
      | some r => callee.regs[r.slot]!
      | none => frameRead callee formal.off d.bytes
    match dst with
    | .sym (.frame off) o =>
      if d.bytes > 8 then
        let fr := s.frame
        let ps := copyBytes fr.params (Int.toNat (off + o)) callee.params formal.off d.bytes
        s := s.setFrame { fr with params := ps }
      else s := s.setFrame (frameWrite s.frame (Int.toNat (off + o)) d.bytes v)
    | a => s ← s.write a d.ty v
  return .next s

/-! ## Synchronization and collectives -/

def barCount (k : Nat) : X (Option Nat) := do
  match arg pi k with
  | .sink => return none
  | _ => return some (← rd e t pi k .u32).toNat

def collect (op : Collective) (maskArg : Option Nat) (contrib : Array UInt32)
    (wb : Resp → Thread → X Thread) : X (Effect Thread) := do
  let m ← match maskArg with
    | some k => pure (← rd e t pi k .b32).toUInt32
    | none => pure 0
  return .collect op m contrib t.advance fun resp s =>
    match wb resp s with
    | .ok s' => s'
    | .error _ => s

def syncOp : X (Effect Thread) := do
  let r := rd e t pi
  let w (k : Nat) (t' : Ty) (resp : Resp) (j : Nat) (s : Thread) : X Thread :=
    s.write (arg pi k) t' (resp[j]?.getD 0).toUInt64
  match pi.i.op with
  | .barSync | .barrierSync =>
    let cnt ← if pi.args.size > 1 then barCount e t pi 1 else pure none
    return .sync (.bar (← r 0 .u32).toNat cnt false) t.advance
  | .barArrive | .barrierArrive =>
    return .sync (.bar (← r 0 .u32).toNat (← barCount e t pi 1) true) t.advance
  | .barRed | .barrierRed =>
    -- `bar.red.popc.u32 d, a{, b}, {!}c` / `bar.red.op.pred p, a{, b}, {!}c`
    let n := pi.args.size
    let cnt ← if n == 4 then barCount e t pi 2 else pure none
    let c ← r (n - 1) .pred
    let op : NV.BarRedOp ← if pi.i.has .popc then pure .popc else
      match pi.i.bool? with
      | some .and => pure .and | some .or => pure .or | _ => throw "bar.red needs .popc, .and or .or"
    let dty := if op == .popc then Ty.u32 else .pred
    collect e t pi (.barRed (← r 1 .u32).toNat cnt op) none #[c.toUInt32] (w 0 dty · 0)
  | .barWarpSync => collect e t pi .warpSync (some 0) #[] fun _ s => pure s
  | .barrierClusterArrive => return .sync .clusterArrive t.advance
  | .barrierClusterWait => return .sync .clusterWait t.advance
  | .membar =>
    if pi.i.has .proxy then return .sync .aliasFence t.advance
    let sc : Scope := match pi.i.scope? with | some s => s | none => if pi.i.has .gl then .gpu else .sys
    return .sync (.fence true sc) t.advance
  | .fence =>
    if pi.i.has .proxy then
      if pi.i.has .aliasProxy then return .sync .aliasFence t.advance
      if pi.i.has .async || pi.i.has .asyncGeneric then return .sync (.proxyFence .async) t.advance
      -- Tensor maps are read in the generic proxy here (`NV.PTX.Async.tensorOperand`): the
      -- tensormap proxy fence is the release or acquire fence at its scope.
      if pi.i.has .tensormapGeneric then return .sync (.fence false ((pi.i.scope?).getD .sys)) t.advance
      throw "this proxy fence belongs to the asynchronous-operation semantics"
    let sc := (pi.i.scope?).getD .sys
    return .sync (.fence (pi.i.sem? == some .sc) sc) t.advance
  | .shflSync =>
    let contrib := #[(← r 1 .b32).toUInt32, (← r 2 .b32).toUInt32, (← r 3 .b32).toUInt32]
    let mode : NV.ShflMode := if pi.i.has .up then .up else if pi.i.has .down then .down
      else if pi.i.has .bfly then .bfly else .idx
    collect e t pi (.shfl mode) (some 4) contrib fun resp s => do
      match arg pi 0 with
      | .pair d p =>
        let s ← s.write d .b32 (resp[0]?.getD 0).toUInt64
        s.write p .pred (resp[1]?.getD 0).toUInt64
      | d => s.write d .b32 (resp[0]?.getD 0).toUInt64
  | .voteSync =>
    let mode : NV.VoteMode := if pi.i.has .all then .all else if pi.i.has .any then .any
      else if pi.i.has .uni then .uni else .ballot
    let dty := if mode == .ballot then Ty.b32 else .pred
    collect e t pi (.vote mode) (some 2) #[(← r 1 .pred).toUInt32] (w 0 dty · 0)
  | .matchSync =>
    let t0 := ty pi
    let a ← r 1 t0
    let contrib := if t0.bits > 32 then #[a.toUInt32, (a >>> 32).toUInt32] else #[a.toUInt32]
    if pi.i.has .all then
      collect e t pi .matchAll (some 2) contrib fun resp s => do
        match arg pi 0 with
        | .pair d p =>
          let s ← s.write d .b32 (resp[0]?.getD 0).toUInt64
          s.write p .pred (resp[1]?.getD 0).toUInt64
        | d => s.write d .b32 (resp[0]?.getD 0).toUInt64
    else collect e t pi .matchAny (some 2) contrib (w 0 .b32 · 0)
  | .reduxSync =>
    let t0 := ty pi
    if t0 == .f32 then
      let isMax ← match pi.i.red? with
        | some .min => pure false | some .max => pure true
        | _ => throw "redux.sync.f32 takes .min or .max"
      return ← collect e t pi (.reduxF32 isMax (pi.i.has .abs) (pi.i.has .nan)) (some 2)
        #[(← r 1 .b32).toUInt32] (w 0 .b32 · 0)
    let op : NV.ReduxOp ← match pi.i.red? with
      | some .add => pure .add | some .min => pure .min | some .max => pure .max
      | some .and => pure .and | some .or => pure .or | some .xor => pure .xor
      | _ => throw "redux.sync without an operation"
    collect e t pi (.redux op t0.isSigned) (some 2) #[(← r 1 t0).toUInt32] (w 0 t0 · 0)
  | .electSync =>
    collect e t pi .elect (some 1) #[] fun resp s => do
      match arg pi 0 with
      | .pair d p =>
        let s ← s.write d .b32 (resp[0]?.getD 0).toUInt64
        s.write p .pred (resp[1]?.getD 0).toUInt64
      | d => s.write d .b32 (resp[0]?.getD 0).toUInt64
  | .activemask => collect e t pi .activemask none #[] (w 0 .b32 · 0)
  | op => throw s!"{op.spelling} is not a synchronization instruction"


/-! ## Warp-level matrix instructions (§9.7.16) -/

/-- The form of an `ldmatrix`/`stmatrix` (§9.7.16.5.15–16): `.m8n8` with `.b16`; `ldmatrix`
`.m16n16.trans` with `.b8` or `.b8x16` from `.b4x16_p64`/`.b6x16_p32`, `.m8n16` with `.b8x16`
from those; `stmatrix` `.m16n8.trans.b8`. -/
def matMove (i : Instr) : Except String Matrix.MatMove := do
  let num := i.num?.getD 1
  let trans := i.has .trans
  let some (m, n, _) := i.shape? | throw "a matrix move without a shape"
  let src ← match i.types with
    | [.b16] => pure 16 | [.b8] => pure 8 | [.b8x16, .b4x16p64] => pure 4
    | [.b8x16, .b6x16p32] => pure 6
    | _ => throw "the matrix move's types are not modelled (.s8.s4 needs PTX ISA 9.4)"
  let load := i.op == .ldmatrix
  let ok := match m, n with
    | 8, 8 => src == 16
    | 16, 16 => load && trans && src != 16 && num ≤ 2
    | 8, 16 => load && !trans && (src == 4 || src == 6)
    | 16, 8 => !load && trans && src == 8
    | _, _ => false
  unless ok do throw s!"{if load then "ldmatrix" else "stmatrix"}.m{m}n{n} of that type is not a form of the ISA"
  return { m, n, trans, src, num }

/-- A gather of every lane's words that leaves the program counter in place, for instructions
that continue with further sub-steps. -/
def gatherHere (contrib : Array UInt32) (wb : Resp → Thread → X Thread) : Effect Thread :=
  .collect .gather 0xFFFFFFFF contrib t fun resp s =>
    match wb resp s with
    | .ok s' => s'
    | .error _ => s

def setMicro (s : Thread) (micro : Nat) (scratch : Array UInt64) : Thread :=
  s.setFrame { s.frame with micro, scratch }

/-- The 32-bit words of a vector of registers, 64-bit registers as two words. -/
def fragWords (a : Arg) (n : Nat) (ty : Ty) : X (Array UInt32) := do
  let xs ← Thread.elems a n
  let mut out := #[]
  for x in xs do
    let v ← t.read e x ty
    out := if ty.bits == 64 then out ++ #[v.toUInt32, (v >>> 32).toUInt32] else out.push v.toUInt32
  return out

/-- Half `e` (16 bits) of register `j` of lane `l` in gathered words, `n` words per lane. -/
def half (ws : Array UInt64) (n l j e : Nat) : UInt64 := (ws[l * n + j]! >>> (16 * e).toUInt64) &&& 0xFFFF

def matrixOp : X (Effect Thread) := do
  let lane := t.ctx.laneid
  match pi.i.op with
  | .mma =>
    let some (m, n, k) := pi.i.shape? | throw "mma without a shape"
    unless pi.i.has .row && pi.i.has .col do throw "only the .row.col layouts are modelled"
    if pi.i.has .blockScale then
      -- `mma.sync…kind.block_scale{.scale_vec}.f32.atype.btype.f32.stype d, a, b, c, scale-a,
      -- {byte-id-a, thread-id-a}, scale-b, {byte-id-b, thread-id-b}` (§9.7.16.3): the warp's A, B,
      -- C and scale registers gathered.
      let [.f32, a, b, .f32, st] := pi.i.types | throw "a block-scaled mma is .f32.atype.btype.f32.stype"
      if pi.i.has .sp || pi.i.has .spOrderedMetadata then throw "sparse block-scaled mma is not modelled"
      let some x := Matrix.shape? m n k a .f32 | throw s!"mma.m{m}n{n}k{k} with .{Keyword.name a}"
      let vec := if pi.i.has .scaleVec4X then 4 else if pi.i.has .scaleVec2X then 2
        else if pi.i.has .scaleVec1X then 1 else if pi.i.has .kindMxf8f6f4 then 1 else 2
      let some (format, _) := Float.lanes? st | throw s!"scale type .{Keyword.name st}"
      let sel ← fragWords e t (arg pi 5) 2 .u32
      let selB ← fragWords e t (arg pi 7) 2 .u32
      let s : Matrix.Scaling := { format, vec, byteA := sel[0]!.toNat, threadA := sel[1]!.toNat,
                                  byteB := selB[0]!.toNat, threadB := selB[1]!.toNat }
      let _ ← Matrix.model e.prog.sm (if Matrix.narrow a then a else b) .f32
      let contrib := (← fragWords e t (arg pi 1) x.aWords .b32) ++ (← fragWords e t (arg pi 2) x.bWords .b32) ++
        (← fragWords e t (arg pi 3) x.cWords .b32) ++ #[(← rd e t pi 4 .b32).toUInt32, (← rd e t pi 6 .b32).toUInt32]
      return .collect .gather 0xFFFFFFFF contrib t.advance fun resp s' =>
        let r : X Thread := do
          let outs ← Matrix.computeScaled e.prog.sm x a b s resp
          let mut s' := s'
          for (dst, j) in (← Thread.elems (arg pi 0) x.cWords).zipIdx do
            s' ← s'.write dst .b32 outs[lane]![j]!.toUInt64
          return s'
        match r with | .ok s'' => s'' | .error _ => s'
    let [d, a, b, c] := pi.i.types | throw "mma needs four types"
    if pi.i.has .sp || pi.i.has .spOrderedMetadata then
      -- `mma.sp` (§9.7.16.6): the warp's compressed A, B, C and metadata gathered, then each
      -- lane computes D, faulting on metadata the ISA leaves undefined.
      let some (x, xa) := Matrix.sparseShape? m n k a d
        | throw s!"mma.sp.m{m}n{n}k{k} with .{Keyword.name a} is not modelled (the sparse tensor-core datapath of floats is unpublished)"
      let f := t.frame
      if f.micro == 0 then
        let contrib := (← fragWords e t (arg pi 1) xa.aWords .b32) ++
          (← fragWords e t (arg pi 2) x.bWords .b32) ++ (← fragWords e t (arg pi 3) x.cWords .b32) ++
          #[(← rd e t pi 4 .b32).toUInt32]
        return gatherHere t contrib fun resp s => pure (setMicro s 1 (resp.map (·.toUInt64)))
      let outs ← Matrix.computeSparse x xa a b (pi.i.has .satfinite) (← rd e t pi 5 .u32).toNat
        (pi.i.has .spOrderedMetadata) (f.scratch.map (·.toUInt32))
      let mut s := setMicro t 0 #[]
      for (dst, j) in (← Thread.elems (arg pi 0) x.cWords).zipIdx do
        s ← s.write dst .b32 outs[lane]![j]!.toUInt64
      return .next s.advance
    let some x := Matrix.shape? m n k a d
      | throw s!"mma.m{m}n{n}k{k} with .{Keyword.name a} is not modelled"
    let x := { x with xor := pi.i.bool? == some .xor }
    Matrix.available e.prog.sm a d
    let regTy (t' : Ty) : Ty := if t' == .f64 then .f64 else .b32
    let regs (words : Nat) (t' : Ty) := if t' == .f64 then words / 2 else words
    let contrib := (← fragWords e t (arg pi 1) (regs x.aWords a) (regTy a)) ++
      (← fragWords e t (arg pi 2) (regs x.bWords b) (regTy b)) ++
      (← fragWords e t (arg pi 3) (regs x.cWords c) (regTy c))
    return .collect .gather 0xFFFFFFFF contrib t.advance fun resp s =>
      let r : X Thread := do
        let dw ← Matrix.compute e.prog.sm x a b c d (pi.i.has .satfinite) resp lane
        let ds ← Thread.elems (arg pi 0) (regs x.cWords d)
        let mut s := s
        for (dst, j) in ds.zipIdx do
          s ← if d == .f64 then
              s.write dst .b64 (dw[2 * j]!.toUInt64 ||| (dw[2 * j + 1]!.toUInt64 <<< 32))
            else s.write dst .b32 dw[j]!.toUInt64
        return s
      match r with | .ok s' => s' | .error _ => s
  | .movmatrix =>
    -- `movmatrix.sync.aligned.m8n8.trans.b16 d, a`: lane `l`'s half `h` of `d` is element
    -- (l/4, 2(l%4)+h) of Aᵀ, A's element (r, c) being in lane 4r + c/2, half c%2.
    return .collect .gather 0xFFFFFFFF #[(← rd e t pi 1 .b32).toUInt32] t.advance fun resp s =>
      let ws := resp.map (·.toUInt64)
      let v := (List.range 2).foldl (fun acc h =>
        let r := 2 * (lane % 4) + h
        let c := lane / 4
        acc ||| (half ws 1 (4 * r + c / 2) 0 (c % 2) <<< (16 * h).toUInt64)) 0
      s.writeOr (arg pi 0) .b32 v
  | .ldmatrix | .stmatrix =>
    let x ← matMove pi.i
    let regs := x.num * x.regs
    let addrArg := if pi.i.op == .ldmatrix then arg pi 1 else arg pi 0
    let .mem sp own ← t.target e (space pi.i) addrArg | throw "not a memory address"
    let f := t.frame
    let rowLane := lane < x.rows * x.num
    if pi.i.op == .ldmatrix then
      -- Each lane giving a row's address loads its 16 bytes; then the warp exchanges the rows
      -- and each lane keeps its fragment (`MatMove.frag`).
      if f.micro == 0 then
        if !rowLane then return .next (setMicro t 1 #[0, 0, 0, 0])
        let req ← request pi.i sp own 16 .ld
        return .mem (fun _ => req) t fun resp s => setMicro s 1 (resp.map (·.toUInt64))
      let ds ← Thread.elems (arg pi 0) regs
      return gatherHere t (f.scratch.map (·.toUInt32)) fun resp s => do
        let mut s := s
        for (dst, j) in ds.zipIdx do
          s ← s.write dst .b32 (x.frag (fun τ k => resp[4 * τ + k]!) lane j).toUInt64
        return (setMicro s 0 #[]).advance
    -- stmatrix: gather every lane's registers, then each lane giving a row's address stores the
    -- row (`MatMove.rowsOf`).
    if f.micro == 0 then
      let ws ← fragWords e t (arg pi 1) regs .b32
      return gatherHere t ws fun resp s => pure (setMicro s 1 (resp.map (·.toUInt64)))
    if !rowLane then return .next (setMicro t 0 #[]).advance
    let data := (x.rowsOf fun l j => f.scratch[regs * l + j]!.toUInt32).extract (4 * lane) (4 * lane + 4)
    let req ← request pi.i sp own 16 .st data
    return .mem (fun _ => req) t fun _ s => (setMicro s 0 #[]).advance
  | op => throw s!"{op.spelling} is not a warp-level matrix instruction"

/-- The fragment a `wmma` instruction names: matrix `kind` of the instruction's shape, of type `ty`. -/
def wfragOf (kind : Char) (ty : Ty) : X Matrix.WFrag := do
  let some (m, n, k) := pi.i.shape? | throw "wmma without a shape"
  let some f := Matrix.wfrag kind m n k ty
    | throw s!"wmma.m{m}n{n}k{k} with .{Keyword.name ty} is not modelled"
  return f

/-- `wmma.load`/`wmma.store` (§9.7.16.4.2–3): each lane moves its fragment's distinct elements one
by one, element (r, c) of the matrix at `p + (r·stride + c)` elements (row-major) or
`p + (c·stride + r)` (column-major); the stride defaults to the leading dimension. -/
def wmmaMove : X (Effect Thread) := do
  let lane := t.ctx.laneid
  let load := pi.i.op == .wmmaLoad
  let kind := if !load then 'c' else if pi.i.has .a then 'a' else if pi.i.has .b then 'b' else 'c'
  let some (m, n, k) := pi.i.shape? | throw "wmma without a shape"
  let ty := pi.i.types.headD .b32
  let f ← wfragOf pi kind ty
  let (rows, cols) := match kind with | 'a' => (m, k) | 'b' => (k, n) | _ => (m, n)
  let row := pi.i.has .row
  let (pk, rk, sk) := if load then (1, 0, 2) else (0, 1, 2)
  let stride ← if pi.args.size > sk then do pure (← rd e t pi sk .u32).toNat
    else pure (if row then cols else rows)
  let .mem sp base ← t.target e (space pi.i) (arg pi pk) | throw "not a memory address"
  let bytes := f.w / 8
  let distinct := f.e * f.blocks.size
  let regs := (f.count * f.w + 31) / 32 / (if f.w == 64 then 2 else 1)
  let rty : Ty := if f.w == 64 then .b64 else .b32
  let dst ← Thread.elems (arg pi rk) regs
  let fr := t.frame
  let i := fr.micro
  let (r, c) := f.coord lane i
  -- The offset in elements: the contiguous coordinate counts `sub` elements a unit.
  let off := if row then r * stride + c * f.sub else c * stride + r * f.sub
  let addr := base + (off * f.w / f.sub / 8).toUInt64
  let last := i + 1 == distinct
  if load then
    let req ← request pi.i sp addr bytes .ld
    return .mem (fun _ => req) t fun resp s =>
      let v := if bytes == 8 then resp[0]!.toUInt64 ||| (resp[1]!.toUInt64 <<< 32)
        else resp[0]!.toUInt64 &&& ((1 <<< (8 * bytes).toUInt64) - 1)
      let scratch := s.frame.scratch.push v
      if !last then setMicro s (i + 1) scratch else
      let vals := (Array.range f.count).map fun j => (scratch[j % distinct]!).toNat
      let ws := f.pack vals
      let s := (dst.zipIdx.foldl (fun s (x, j) =>
        let v := if f.w == 64 then ws[2 * j]!.toUInt64 ||| (ws[2 * j + 1]!.toUInt64 <<< 32)
          else ws[j]!.toUInt64
        s.writeOr x rty v) s)
      (setMicro s 0 #[]).advance
  else
    let ws ← fragWords e t (arg pi rk) regs rty
    let v := f.elem ws i
    let data := if bytes == 8 then #[v.toUInt32, (v >>> 32).toUInt32] else
      packWords bytes #[v.toUInt64]
    let req ← request pi.i sp addr bytes .st data
    return .mem (fun _ => req) t fun _ s =>
      if last then (setMicro s 0 #[]).advance else setMicro s (i + 1) #[]

/-- `wmma.mma` (§9.7.16.4.4): the warp's fragments gathered, D = A·B + C computed once with the
family's tensor-core numerics (as `mma`), each lane keeping its D fragment. -/
def wmmaMma : X (Effect Thread) := do
  let lane := t.ctx.laneid
  let some (m, n, k) := pi.i.shape? | throw "wmma without a shape"
  let (d, a, b, c) ← match pi.i.types with
    | [d, c] => pure (d, Ty.f16, Ty.f16, c)
    | [d, a, b, c] => pure (d, a, b, c)
    | _ => throw "wmma.mma types"
  if a == .f64 && pi.i.rnd?.any (· != .rn) then throw "wmma.mma.f64 is modelled with .rn only"
  Matrix.available e.prog.sm a d
  let fa ← wfragOf pi 'a' a
  let fb ← wfragOf pi 'b' b
  let fc ← wfragOf pi 'c' c
  let fd ← wfragOf pi 'c' d
  let rty (f : Matrix.WFrag) : Ty := if f.w == 64 then .b64 else .b32
  let regs (f : Matrix.WFrag) := f.words / (if f.w == 64 then 2 else 1)
  let contrib := (← fragWords e t (arg pi 1) (regs fa) (rty fa)) ++
    (← fragWords e t (arg pi 2) (regs fb) (rty fb)) ++ (← fragWords e t (arg pi 3) (regs fc) (rty fc))
  let dst ← Thread.elems (arg pi 0) (regs fd)
  let sat := pi.i.has .satfinite
  return .collect .gather 0xFFFFFFFF contrib t.advance fun resp s =>
    match Matrix.wmmaAll e.prog.sm m n k a b d sat fa fb fc fd resp
        (pi.i.bool? == some .xor) with
    | .error _ => s
    | .ok all =>
      let ws := all[lane]!
      dst.zipIdx.foldl (fun s (x, j) =>
        let v := if fd.w == 64 then ws[2 * j]!.toUInt64 ||| (ws[2 * j + 1]!.toUInt64 <<< 32)
          else ws[j]!.toUInt64
        s.writeOr x (rty fd) v) s

/-- `st.bulk{.weak}{.shared::cta} [a], size, 0` (§9.7.10.14): `size` bytes of shared memory
zeroed by weak stores, 256 bytes (32 lanes of 8) a step. -/
def stBulk : X (Effect Thread) := do
  unless (← rd e t pi 2 .u64) == 0 do throw "st.bulk's initval must be 0"
  let size := (← rd e t pi 1 .u64).toNat
  unless size % 8 == 0 do throw s!"st.bulk of {size} bytes: not a multiple of 8"
  let .mem sp a ← t.target e (some (pi.i.space?.getD .shared)) (arg pi 0) | throw "not a memory address"
  unless sp == .shared do throw "st.bulk initializes shared memory"
  unless a % 8 == 0 do throw s!"st.bulk at {a}: not 8-byte aligned"
  let done := t.frame.micro * 256
  if done ≥ size then return .next (setMicro t 0 #[]).advance
  let lanes := min 32 ((size - done) / 8)
  let addrs := (Array.range lanes).map fun j => some (a + (done + 8 * j).toUInt64)
  let req : Req := { space := sp, op := .st, bytes := 8, addrs, data := Array.replicate (2 * lanes) 0 }
  return .mem (fun _ => req) t fun _ s =>
    if done + 256 ≥ size then (setMicro s 0 #[]).advance else setMicro s (s.frame.micro + 1) #[]

/-- A `multimem.*` instruction as the plain memory instruction `op` on its one location: the
reduction and accumulation qualifiers dropped for a load, and FP8 element types as the bit type of
their width (a one-value reduction does not convert). -/
def multimemAs (pi : PInstr) (op : Op) : PInstr :=
  let bitsTy (t : Ty) : Ty :=
    if [Ty.e4m3, .e5m2, .e4m3x2, .e5m2x2, .e4m3x4, .e5m2x4].contains t then
      match t.bits with | 8 => .b8 | 16 => .b16 | _ => .b32
    else t
  let quals := if op == .ld then pi.i.quals.filter fun q => !(q matches .red _) && q != .flag .accF32 &&
      q != .flag .accF16 else pi.i.quals
  { pi with i := { pi.i with op, quals, types := pi.i.types.map bitsTy } }

/-- The texture or surface object a handle names (`NV.Tex`: handle `i + 1` is entry `i`). -/
def texObject (h : UInt64) (surface : Bool) : X NV.Tex.Texture := do
  let some tx := (if h == 0 then none else e.textures[h.toNat - 1]?)
    | throw s!"handle {h} names no {if surface then "surface" else "texture"} object of the launch"
  if tx.surface != surface then
    throw s!"handle {h} names a {if tx.surface then "surface" else "texture"} object"
  if let some why := tx.invalid? then throw s!"object {h}: {why}"
  return tx

/-- The object and the coordinates of a `[a, c]` operand (`c` a scalar or vector). -/
def texOperand (k : Nat) : X (UInt64 × Array Arg) := do
  let .list xs := arg pi k | throw "expected [object, coordinates]"
  unless xs.size == 2 do throw "explicit samplers are not modelled"
  let h ← t.read e xs[0]! .b64
  let cs := match xs[1]! with | .vec cs => cs | c => #[c]
  return (h, cs)

/-- `tex.{1d,2d,3d}.v4.dtype.ctype d, [a, c]` and `tld4.comp.2d.v4.dtype.f32 d, [a, c]`
(§9.7.12.3–4): one load of every texel the fetch combines, then the combination
(`NV.Tex.combine`). Integer coordinates (`.s32`) index texels directly with nearest-point
sampling (`tex1Dfetch`; inferred for 2d and 3d). The launch's textures have one level, so the
mipmap forms (`.base`, `.level` with its LOD, `.grad` with its gradients) fetch from it: the LOD
clamps to the one level there is (inferred; the LOD and gradients are read, not used); a
mipmapped object's `.level` fetch reads the levels its LOD names (`Texture.lodOf`), and its other
fetches are not modelled. The array geometries (`.a1d`, `.a2d`) fetch from the layer their
first coordinate names (`Texture.layer`). `.cube`/`.acube` fetch from the face their direction
points at (`Texture.cubeFace`), the direction first scaled by the reciprocal of its largest
magnitude, and
`.acube`'s cubemap index clamped to 65535 (`SEL`). Not modelled, and faulting: samplers, offsets,
depth compare, multisampling, `.f16` results, the residency predicate. The offset operand moves the
coordinates by whole texels (`Texture.footprint`). -/
def texFetch : X (Effect Thread) := do
  let i := pi.i
  if i.has .geom2dms || i.has .a2dms then throw "this texture geometry is not modelled"
  let cube := i.has .cube || i.has .acube
  let array := i.has .a1d || i.has .a2d || i.has .acube
  let extra := if i.has .level then 1 else if i.has .grad then 2 else 0
  if pi.args.size > 3 + extra || (arg pi 0) matches .pair .. then
    throw "depth compare and the residency predicate are not modelled"
  for k in [2:2 + extra] do
    match arg pi k with
    | .vec cs => for c in cs do let _ ← t.read e c .f32
    | c => let _ ← t.read e c .f32
  -- The offset operand: texels, each its low four bits as a signed number (the ISA's range is -8
  -- to 7; outside it, inferred).
  let offs ← if pi.args.size == 3 + extra then
      match arg pi (2 + extra) with
      | .vec xs => xs.mapM fun x => do
          let v := (← t.read e x .s32).toNat % 16
          pure (if v ≥ 8 then (v : Int) - 16 else v)
      | _ => throw "depth compare is not modelled"
    else pure #[]
  let some dims := if i.has .a1d then some 1 else if i.has .a2d || cube then some 2 else i.dim?
    | throw "no geometry"
  let gather := i.op == .tld4
  let (dty, cty) ← match i.types with
    | [d, c] => pure (d, c)
    | _ => throw "tex needs .dtype.ctype"
  let (h, cargs) ← texOperand e t pi 1
  let tx ← texObject e h false
  unless tx.dims == dims do throw s!"a {dims}d fetch from a {tx.dims}d texture"
  unless tx.cube == cube do throw s!"a {if cube then "cubemap" else "plain"} fetch from a {if tx.cube then "cubemap" else "plain"} texture"
  if cube && cty != .f32 then throw "cubemap fetches take .f32 coordinates"
  -- An array fetch: the layer the first coordinate names, the rest its coordinates.
  let (idx, cargs) ← if array then do
      let k ← t.read e (cargs[0]?.getD .sink) .u32
      pure (if cube then min k.toNat 65535 else k.toNat, cargs.extract 1 cargs.size)
    else pure (0, cargs)
  let (tx, cargs, faceUV) ← if cube then do
      let x ← (cargs.extract 0 3).mapM fun c => do pure (← t.read e c .f32).toUInt32
      let mx := NV.Num.F32.minMax true false false
      let m := mx (mx (x[0]! &&& 0x7fffffff) (x[1]! &&& 0x7fffffff)) (x[2]! &&& 0x7fffffff)
      let r := (NV.Num.Approx.rcp .f32 true m.toNat).toUInt32
      let x := x.map (NV.Num.F32.mul .rn false · r)
      let (face, u, v) := NV.Tex.Texture.cubeFace x[0]! x[1]! x[2]!
      pure ({ (← NV.Tex.Texture.layer tx (6 * idx + face)) with normalizedCoords := true },
            #[], some (u, v))
    else if array then pure (← NV.Tex.Texture.layer tx idx, cargs, none)
    else pure (tx, cargs, none)
  let floats := tx.kind == .float || tx.readNormalized
  unless (dty == .f32) == floats do
    throw s!".{Keyword.name dty} results from a texture of {repr tx.kind} texels{if tx.readNormalized then " read normalized" else ""}"
  let n := if dims == 3 || (array && dims == 2) then (if array then 3 else 4) else dims
  unless cube || cargs.size == n do throw s!"{cargs.size} coordinates for a {dims}d fetch"
  let (tx, coords) ← if let some (u, v) := faceUV then do
      pure (tx, #[← NV.Tex.coordOf u, ← NV.Tex.coordOf v])
    else if cty == .s32 then do
      -- Integer coordinates: texel indices, nearest point, unnormalized.
      let cs ← (cargs.extract 0 dims).mapM fun c => do pure (toInt true 32 (← t.read e c .s32))
      pure ({ tx with filter := .point, normalizedCoords := false },
            cs.map fun c => ({ n := c, e := 0 } : NV.Tex.Dy))
    else do
      let cs ← (cargs.extract 0 dims).mapM fun c => do NV.Tex.coordOf (← t.read e c .f32).toUInt32
      pure (tx, cs)
  if gather then
    unless dims == 2 do throw "tld4 is modelled for 2d textures"
  if cube && !offs.isEmpty then throw "cubemap fetches take no offsets"
  let tx := if gather then { tx with filter := .linear } else tx
  -- A mipmapped object: `.level` reads the levels its LOD names (`Texture.mipFootprint`).
  let fp ← if tx.levels > 1 then do
      unless i.has .level && !gather && cty == .f32 do
        throw "a fetch from a mipmapped texture other than tex.level of float coordinates (not modelled)"
      let lod ← t.read e (match arg pi 2 with | .vec cs => cs[0]?.getD .sink | c => c) .f32
      pure (tx.mipFootprint (tx.lodOf lod.toUInt32) fun tl s => NV.Tex.footprint tl coords offs s)
    else pure (NV.Tex.footprint tx coords offs)
  let bytes := tx.texelBytes
  let req : Req := { space := .global, op := .ld, bytes, addrs := fp.toArray.map (·.1) }
  let ds ← Thread.elems (arg pi 0) 4
  let comp := if i.has .r then 0 else if i.has .g then 1 else if i.has .b then 2 else 3
  return .mem (fun _ => req) t.advance fun resp s =>
    let out := if !gather then NV.Tex.combine tx fp resp else
      let wpt := (bytes + 3) / 4
      let one (k : Nat) : UInt32 :=
        let tx1 := { tx with filter := .point }
        (NV.Tex.combine tx1 [(fp[k]!.1, 1)] (resp.extract (k * wpt) (k * wpt + wpt)))[comp]!
      (NV.Tex.gatherOrder.map one).toArray
    ds.zipIdx.foldl (fun s (x, j) =>
      s.writeOr x .b32 (out[j]?.getD 0).toUInt64) s

/-- `txq` and `suq` (§9.7.12.5, §9.7.13.4): the object's dimensions, coordinate mode, filter and
addressing modes (as the CUDA runtime's enumerations: `cudaFilterModePoint` 0, `Linear` 1;
`cudaAddressModeWrap` 0, `Clamp` 1, `Mirror` 2, `Border` 3). Channel data type and order, array
sizes, mipmap levels and samples are not modelled. -/
def texQuery : X (Effect Thread) := do
  let i := pi.i
  let surface := i.op == .suq
  let h ← match arg pi 1 with
    | .list xs => t.read e (xs[0]?.getD .sink) .b64
    | .addr (.reg r) 0 => t.read e (.reg r) .b64
    | a => t.read e a .b64
  let tx ← texObject e h surface
  let addr (d : Nat) : Nat := match tx.address[d]?.getD .clamp with
    | .wrap => 0 | .clamp => 1 | .mirror => 2 | .border => 3
  let v ← if i.has .width then pure tx.width else if i.has .height then pure tx.height
    else if i.has .depth then pure tx.depth
    else if i.has .normalizedCoords then pure (if tx.normalizedCoords then 1 else 0)
    else if i.has .filterMode then pure (if tx.filter == .linear then 1 else 0)
    else if i.has .addrMode0 then pure (addr 0) else if i.has .addrMode1 then pure (addr 1)
    else if i.has .addrMode2 then pure (addr 2)
    else throw "this query is not modelled"
  set1 pi t v.toUInt64 .b32

/-- `suld.b`, `sust.b`, `sured.b` (§9.7.13.1–3): an unformatted access at a byte offset (the lowest
coordinate) of the surface, `.trap`, `.clamp` or `.zero` out of bounds (`NV.Tex.surfAddr`).
The array geometries (`.a1d`, `.a2d`) access the layer their first coordinate names.
`sust.p`/`sured.p` (formatted) are not modelled. -/
def surfAccess : X (Effect Thread) := do
  let i := pi.i
  unless i.has .b do throw "formatted (.p) surface accesses are not modelled"
  let array := i.has .a1d || i.has .a2d
  let some dims := if i.has .a1d then some 1 else if i.has .a2d then some 2 else i.dim?
    | throw "no geometry"
  let mode : NV.Tex.Clamp := if i.has .trap then .trap else if i.has .clamp then .clamp else .zero
  let load := i.op == .suld
  let opArg := if load then 1 else 0
  let (h, cargs) ← texOperand e t pi opArg
  let tx ← texObject e h true
  unless tx.dims == dims do throw s!"a {dims}d access to a {tx.dims}d surface"
  -- An array access: the layer the first coordinate names (`.a2d`'s fourth is ignored).
  let (tx, cargs) ← if array then do
      let k ← t.read e (cargs[0]?.getD .sink) .u32
      pure (← NV.Tex.Texture.layer tx k.toNat, cargs.extract 1 (1 + dims))
    else pure (tx, cargs)
  let c (k : Nat) : X Int := do
    match cargs[k]? with
    | some a => pure (toInt true 32 (← t.read e a .s32))
    | none => pure 0
  let ty := i.types.headD .b32
  let n := (i.vec?.map Vec.length).getD 1
  let bytes := ty.bytes * n
  let some a ← NV.Tex.surfAddr tx mode bytes (← c 0) (← c 1) (← c 2)
    | -- Out of bounds under `.zero`: loads read zeros, stores and reductions do nothing.
      if load then do
        let ds ← Thread.elems (arg pi 0) n
        return .next (← ds.foldlM (fun s x => s.write x ty 0) t.advance)
      else return .next t.advance
  match i.op with
  | .suld =>
    let ds ← Thread.elems (arg pi 0) n
    let req ← request i .global a bytes .ld
    return .mem (fun _ => req) t.advance fun resp s =>
      let ws := resp.map (·.toUInt64)
      ds.zipIdx.foldl (fun s (x, j) =>
        let v := if ty.bytes == 8 then ws[2 * j]! ||| (ws[2 * j + 1]! <<< 32)
          else (ws[(j * ty.bytes) / 4]! >>> (8 * ((j * ty.bytes) % 4)).toUInt64) &&&
            ((1 <<< (8 * ty.bytes).toUInt64) - 1)
        s.writeOr x ty v) s
  | .sust =>
    let vs ← (← Thread.elems (arg pi 1) n).mapM fun x => t.read e x ty
    let req ← request i .global a bytes .st (packWords ty.bytes vs)
    return .mem (fun _ => req) t.advance fun _ s => s
  | _ =>
    let some o := i.red? | throw "no reduction operator"
    let some aty := atomType ty | throw s!"reductions on .{Keyword.name ty} are not modelled"
    let v ← rd e t pi 1 ty
    let data := if ty.bytes == 8 then #[v.toUInt32, (v >>> 32).toUInt32] else #[v.toUInt32]
    let req ← request i .global a bytes (.red (atomOp o) aty) data
    return .mem (fun _ => req) t.advance fun _ s => s

/-- The instruction's effect, after its guard held. -/
def exec : X (Effect Thread) :=
  match pi.i.op with
  | .add | .sub | .mul | .mad | .min | .max | .abs | .neg | .div | .rem =>
    if Float.isFloatTy (ty pi) then floatOp e t pi else intOp e t pi
  | .fma | .rcp | .sqrt | .rsqrt | .sin | .cos | .lg2 | .ex2 | .tanh | .copysign | .testp =>
    floatOp e t pi
  | .addc | .subc | .madc | .mul24 | .mad24 | .sad | .popc | .clz | .bfind | .fns | .brev
  | .bfe | .bfi | .szext | .bmsk | .dp4a | .dp2a | .clmad | .and | .or | .xor | .not | .cnot
  | .slct => if ty pi 1 == .f32 then floatOp e t pi else intOp e t pi
  | .lop3 | .shf | .shl | .shr | .prmt | .selp => intOp e t pi
  | .vadd | .vsub | .vabsdiff | .vmin | .vmax | .vshl | .vshr | .vmad | .vset
  | .vadd2 | .vsub2 | .vavrg2 | .vabsdiff2 | .vmin2 | .vmax2 | .vset2
  | .vadd4 | .vsub4 | .vavrg4 | .vabsdiff4 | .vmin4 | .vmax4 | .vset4 => videoOp e t pi
  | .setp | .set =>
    let st := if pi.i.op == .set then ty pi 1 else ty pi
    if Float.isFloatTy st then floatOp e t pi else intOp e t pi
  | .mov => mov e t pi
  | .cvt => cvt e t pi
  | .cvtPack => cvtPack e t pi
  | .ld | .ldu => ld e t pi
  | .st => st e t pi
  | .atom => atom e t pi true
  | .red => atom e t pi false
  | .cvta | .isspacep | .mapa | .getctarank => addressOp e t pi
  | .bra =>
    match arg pi 0 with
    | .label pc => pure (.next (t.jump pc))
    | _ => throw "branch to a non-label"
  | .brxIdx => do
    let .args ts := arg pi 1 | throw "brx.idx without a .branchtargets list"
    let k := (← rd e t pi 0 .u32).toNat
    match ts[k]? with
    | some (Arg.label pc) => pure (.next (t.jump pc))
    | _ => throw s!"brx.idx index {k} outside its {ts.size} targets (undefined)"
  | .call => call e t pi
  | .ret => ret e t
  | .exit => pure .exit
  | .trap | .brkpt => throw "trap"
  | .nanosleep | .pmevent | .setmaxnreg | .griddepcontrol | .prefetch | .prefetchu
  | .applypriority | .discard => pure (.next t.advance)
  -- `createpolicy` (§9.7.10.21): an opaque cache-eviction policy, read only by cache hints;
  -- caches are not part of the thread semantics, so the policy is a constant.
  | .createpolicy => set1 pi t 0
  -- Stack manipulation (§9.7.19): the frame's stack pointer.
  | .stacksave => set1 pi t t.frame.sp
  | .stackrestore => do
    let sp ← rd e t pi 0 (ty pi)
    pure (.next (t.setFrame { t.frame with sp }).advance)
  | .alloca => do
    let size ← rd e t pi 1 (ty pi)
    -- The alignment: at least the guaranteed 8 bytes, a power of 2.
    let align ← if pi.args.size > 2 then rd e t pi 2 .u32 else pure 8
    let a := max align 8
    if a &&& (a - 1) != 0 then throw s!"alloca alignment {a} is not a power of 2"
    let sp := t.frame.sp
    if size > sp then throw "alloca: stack overflow"
    let p := (sp - size) &&& ~~~(a - 1)
    if p.toNat < e.prog.localBytes then
      throw s!"alloca: stack overflow (the {stackBytes}-byte stack)"
    set1 pi (t.setFrame { t.frame with sp := p }) p
  | .barSync | .barrierSync | .barArrive | .barrierArrive | .barRed | .barrierRed
  | .barWarpSync | .barrierClusterArrive | .barrierClusterWait | .membar | .fence | .shflSync
  | .voteSync | .matchSync | .reduxSync | .electSync | .activemask => syncOp e t pi
  | .mma | .movmatrix | .ldmatrix | .stmatrix => matrixOp e t pi
  | .wmmaLoad | .wmmaStore => wmmaMove e t pi
  | .wmmaMma => wmmaMma e t pi
  | .stBulk => stBulk e t pi
  | .tex | .tld4 => texFetch e t pi
  | .txq | .suq => texQuery e t pi
  -- `istypep.type p, a`: whether handle `a` names a texture (`.texref`) or surface (`.surfref`)
  -- object of the launch; the model has no separate sampler objects.
  | .istypep => do
    let h ← rd e t pi 1 .u64
    let surf := pi.i.types.contains .surfref
    let is := pi.i.types.contains .texref || surf
    let ok := is && h != 0 && (e.textures[h.toNat - 1]?.any (·.surface == surf))
    set1 pi t (if ok then 1 else 0) .pred
  | .suld | .sust | .sured => surfAccess e t pi
  -- Outside the model: fabric memory (another node's, PTX ISA 9.3 and later).
  | .fabricTryGet | .fabricTryPut | .fabricTryRed | .fabricTryPullred | .fabricTryAtom
  | .fabricSubmit | .fabricWait => throw "fabric memory (other nodes) is outside the model"
  -- multimem (§9.7.10.13, §9.7.15.8): a multicast object of one device, the model's: the
  -- multimem address names the device's own global location, so `ld_reduce` reduces one value
  -- (the load), `st` and `red` act on the one location; the `.async` forms complete at once.
  | .multimemLdReduce => ld e t (multimemAs pi .ld)
  | .multimemSt | .multimemStAsync => st e t (multimemAs pi .st)
  | .multimemRed | .multimemRedAsync => atom e t (multimemAs pi .red) false
  | op => throw s!"{op.spelling} is not modelled by the thread semantics"

end Step

/-- One step of thread `t`: its next instruction's effect. -/
def Thread.step (e : Env) (t : Thread) : Effect Thread :=
  match t.current? e with
  | none => if t.done then .exit else .fault "the program counter left the function"
  | some pi =>
    let on : Except String Bool := match pi.guard with
      | none => pure true
      | some (neg, r) => pure ((t.slot r.slot &&& 1 == 1) != neg)
    match on with
    | .error m => .fault m
    | .ok false => .next t.advance
    | .ok true =>
      -- The asynchronous instructions are `NV.PTX.Async`'s.
      match NV.PTX.Async.exec? e t pi with
      | some eff => eff
      | none =>
      match Step.exec e t pi with
      | .ok eff => eff
      | .error m => .fault s!"line {pi.line}: {pi.i.op.spelling}: {m}"

end NV.PTX.Sem
