import NV.PTX.Sem.State

/-!
# Integer and bit instructions

The value functions of PTX ISA 9.4 §9.7.1 (integer arithmetic), §9.7.2 (extended precision),
§9.7.7 (integer comparison) and §9.7.9 (logic and shift), on the low `w` bits of 64-bit words.
Each follows the ISA's semantics pseudocode; packed types (`.u16x2`, `.s8x4`, …) apply the scalar
operation lane by lane.
-/

namespace NV.PTX.Sem.Int

def maxS (w : Nat) : Int := 2 ^ (w - 1) - 1
def minS (w : Nat) : Int := -(2 ^ (w - 1))

def clampS (w : Nat) (x : Int) : UInt64 := ofInt w (max (minS w) (min (maxS w) x))
def clampU (w : Nat) (x : Int) : UInt64 := ofInt w (max 0 (min (2 ^ w - 1) x))

/-- Lane width and count of a packed integer type, or the scalar width. -/
def lanes (t : Ty) : Nat × Nat :=
  match t with
  | .u16x2 | .s16x2 => (16, 2)
  | .u8x4 | .s8x4 => (8, 4)
  | t => (t.bits, 1)

def signed (t : Ty) : Bool := t matches .s16x2 | .s8x4 || t.isSigned

/-- Apply `f` (on lane values as integers) lane by lane, with each result truncated to the lane. -/
def lanewise (t : Ty) (f : Int → Int → UInt64) (a b : UInt64) : UInt64 :=
  let (w, n) := lanes t
  (List.range n).foldl (fun acc k =>
    let x := toInt (signed t) w (a >>> (k * w).toUInt64)
    let y := toInt (signed t) w (b >>> (k * w).toUInt64)
    acc ||| (mask w (f x y) <<< (k * w).toUInt64)) (0 : UInt64)

def add (t : Ty) (sat : Bool) (a b : UInt64) : UInt64 :=
  let (w, _) := lanes t
  lanewise t (fun x y =>
    if sat then (if signed t then clampS w (x + y) else clampU w (x + y)) else ofInt w (x + y)) a b

def sub (t : Ty) (sat : Bool) (a b : UInt64) : UInt64 :=
  let (w, _) := lanes t
  lanewise t (fun x y =>
    if sat then (if signed t then clampS w (x - y) else clampU w (x - y)) else ofInt w (x - y)) a b

def min (t : Ty) (relu : Bool) (a b : UInt64) : UInt64 :=
  let (w, _) := lanes t
  lanewise t (fun x y => ofInt w (if relu then max 0 (Min.min x y) else Min.min x y)) a b

def max (t : Ty) (relu : Bool) (a b : UInt64) : UInt64 :=
  let (w, _) := lanes t
  lanewise t (fun x y => ofInt w (if relu then Max.max 0 (Max.max x y) else Max.max x y)) a b

/-- The full product of two `t` values as an integer. -/
def prod (t : Ty) (a b : UInt64) : Int :=
  toInt t.isSigned t.bits a * toInt t.isSigned t.bits b

/-- `mul.lo`/`.hi`/`.wide` (§9.7.1.3): the low, high or full `2w` bits of the product. -/
def mul (t : Ty) (mode : Flag) (a b : UInt64) : UInt64 :=
  let w := t.bits
  let p := prod t a b
  match mode with
  | .hi => ofInt w (p >>> w)
  | .wide => ofInt (2 * w) p
  | _ => ofInt w p

/-- `mad` (§9.7.1.4); `.sat` only for `.hi.s32`. -/
def mad (t : Ty) (mode : Flag) (sat : Bool) (a b c : UInt64) : UInt64 :=
  let w := t.bits
  let p := prod t a b
  match mode with
  | .hi =>
    let s := (p >>> w) + toInt t.isSigned w c
    if sat then clampS w s else ofInt w s
  | .wide => ofInt (2 * w) (p + toInt t.isSigned (2 * w) c)
  | _ => ofInt w (p + toInt t.isSigned w c)

/-- `mul24`/`mad24` (§9.7.1.6, §9.7.1.7): 24-bit operands, bits 47..16 (`.hi`) or 31..0. -/
def mad24 (t : Ty) (hi sat : Bool) (a b c : UInt64) : UInt64 :=
  let s := t.isSigned
  let p := toInt s 24 a * toInt s 24 b
  let r := (if hi then ofInt 32 (p >>> 16) else ofInt 32 p)
  let sum := toInt s 32 r + toInt s 32 c
  if sat then clampS 32 sum else ofInt 32 sum

/-- `sad` (§9.7.1.8): `c + |a - b|`. -/
def sad (t : Ty) (a b c : UInt64) : UInt64 :=
  let x := toInt t.isSigned t.bits a
  let y := toInt t.isSigned t.bits b
  ofInt t.bits (toInt t.isSigned t.bits c + (x - y).natAbs)

/-- `div` and `rem` (§9.7.1.9, §9.7.1.10): truncating, the remainder takes the dividend's sign.
Division by zero is "an unspecified, machine-specific value": `none`. -/
def div (t : Ty) (a b : UInt64) : Option UInt64 :=
  let x := toInt t.isSigned t.bits a
  let y := toInt t.isSigned t.bits b
  if y == 0 then none else some (ofInt t.bits (x.tdiv y))

def rem (t : Ty) (a b : UInt64) : Option UInt64 :=
  let x := toInt t.isSigned t.bits a
  let y := toInt t.isSigned t.bits b
  if y == 0 then none else some (ofInt t.bits (x.tmod y))

def abs (t : Ty) (a : UInt64) : UInt64 :=
  let (w, _) := lanes t
  lanewise t (fun x _ => ofInt w x.natAbs) a 0

def neg (t : Ty) (a : UInt64) : UInt64 :=
  let (w, _) := lanes t
  lanewise t (fun x _ => ofInt w (-x)) a 0

def popc (w : Nat) (a : UInt64) : UInt64 :=
  ((List.range w).filter (fun i => (a >>> i.toUInt64) &&& 1 == 1)).length.toUInt64

def clz (w : Nat) (a : UInt64) : UInt64 :=
  (((List.range w).reverse.takeWhile (fun i => (a >>> i.toUInt64) &&& 1 == 0)).length).toUInt64

/-- `bfind` (§9.7.1.17). -/
def bfind (t : Ty) (shiftamt : Bool) (a : UInt64) : UInt64 :=
  let msb := t.bits - 1
  let a := if t.isSigned && (a >>> msb.toUInt64) &&& 1 == 1 then ~~~a else a
  match (List.range t.bits).reverse.find? (fun i => (a >>> i.toUInt64) &&& 1 == 1) with
  | some i => (if shiftamt then msb - i else i).toUInt64
  | none => 0xFFFFFFFF

/-- `fns` (§9.7.1.18); `base > 31` is undefined: `none`. -/
def fns (m base : UInt64) (offset : Int) : Option UInt64 :=
  if base > 31 then none else
  let bit (p : Nat) := (m >>> p.toUInt64) &&& 1 == 1
  if offset == 0 then some (if bit base.toNat then base else 0xFFFFFFFF)
  else
    let ps := if offset > 0 then (List.range 32).drop base.toNat
      else (List.range (base.toNat + 1)).reverse
    some (match (ps.filter bit)[offset.natAbs - 1]? with
      | some p => p.toUInt64
      | none => 0xFFFFFFFF)

def brev (w : Nat) (a : UInt64) : UInt64 :=
  (List.range w).foldl (fun acc i =>
    if (a >>> i.toUInt64) &&& 1 == 1 then acc ||| ((1 : UInt64) <<< (w - 1 - i).toUInt64) else acc) (0 : UInt64)

/-- `bfe` (§9.7.1.20). -/
def bfe (t : Ty) (a b c : UInt64) : UInt64 :=
  let msb := t.bits - 1
  let pos := (b &&& 0xff).toNat
  let len := (c &&& 0xff).toNat
  let bit (i : Nat) := (a >>> i.toUInt64) &&& 1 == 1
  let sbit := if !t.isSigned || len == 0 then false else bit (Nat.min (pos + len - 1) msb)
  (List.range t.bits).foldl (fun d i =>
    let v := if i < len && pos + i ≤ msb then bit (pos + i) else sbit
    if v then d ||| ((1 : UInt64) <<< i.toUInt64) else d) (0 : UInt64)

/-- `bfi` (§9.7.1.21). -/
def bfi (w : Nat) (a b c d : UInt64) : UInt64 :=
  let pos := (c &&& 0xff).toNat
  let len := (d &&& 0xff).toNat
  (List.range len).foldl (fun f i =>
    if pos + i ≤ w - 1 then
      let m := (1 : UInt64) <<< (pos + i).toUInt64
      if (a >>> i.toUInt64) &&& 1 == 1 then f ||| m else f &&& ~~~m
    else f) b

/-- `szext` (§9.7.1.22). -/
def szext (t : Ty) (clamp : Bool) (a b : UInt64) : UInt64 :=
  let b1 := b &&& 0x1f
  let tooLarge := b ≥ 32 && clamp
  let m : UInt64 := if tooLarge then 0 else mask 32 (0xFFFFFFFF <<< b1)
  let signBit := !(b1 == 0 || tooLarge || !t.isSigned) && (a >>> ((b1 - 1) &&& 0x1f)) &&& 1 == 1
  mask 32 ((a &&& ~~~m) ||| (if signBit then m else 0))

/-- `bmsk` (§9.7.1.23). -/
def bmsk (clamp : Bool) (a b : UInt64) : UInt64 :=
  let a1 := a &&& 0x1f
  let b1 := b &&& 0x1f
  let sum := a1 + b1
  let mask0 : UInt64 := if clamp && a ≥ 32 then 0 else mask 32 (0xFFFFFFFF <<< a1)
  let over := sum ≥ 32 || (clamp && (a ≥ 32 || b ≥ 32))
  let mask1 : UInt64 := if over then 0 else if b1 == 0 then 0xFFFFFFFF
    else mask 32 (0xFFFFFFFF <<< sum)
  mask 32 (mask0 &&& ~~~mask1)

/-- `dp4a` (§9.7.1.24) and `dp2a` (§9.7.1.25). -/
def dp4a (sa sb : Bool) (a b c : UInt64) : UInt64 :=
  let c := toInt (sa || sb) 32 c
  ofInt 32 ((List.range 4).foldl (fun acc i =>
    acc + toInt sa 8 (a >>> (8 * i).toUInt64) * toInt sb 8 (b >>> (8 * i).toUInt64)) c)

def dp2a (sa sb hi : Bool) (a b c : UInt64) : UInt64 :=
  let c := toInt (sa || sb) 32 c
  let sel := if hi then 2 else 0
  ofInt 32 ((List.range 2).foldl (fun acc i =>
    acc + toInt sa 16 (a >>> (16 * i).toUInt64) *
      toInt sb 8 (b >>> (8 * (sel + i)).toUInt64)) c)

/-- `clmad` (§9.7.1.5): carry-less product of 64-bit `a` and `b`, low or high half, xor `c`. -/
def clmad (hi : Bool) (a b c : UInt64) : UInt64 :=
  let p := (List.range 64).foldl (fun acc i =>
    if (b >>> i.toUInt64) &&& 1 == 1 then acc ^^^ (a.toNat <<< i) else acc) (0 : Nat)
  (if hi then (p >>> 64).toUInt64 else p.toUInt64) ^^^ c

/-- `shl`/`shr` (§9.7.9.8, §9.7.9.9): amounts beyond the width clamp to the width. -/
def shl (w : Nat) (a b : UInt64) : UInt64 :=
  if b.toNat ≥ w then 0 else mask w (a <<< b)

def shr (t : Ty) (a b : UInt64) : UInt64 :=
  let w := t.bits
  let n := Nat.min b.toNat w
  if t.isSigned then ofInt w ((toInt true w a) >>> n)
  else if n ≥ w then 0 else mask w a >>> n.toUInt64

/-- `shf.l`/`shf.r` (§9.7.9.7): funnel shift of `{b, a}` by `c`, clamped to 32 or wrapped. -/
def shf (left clamp : Bool) (a b c : UInt64) : UInt64 :=
  let n := if clamp then Nat.min c.toNat 32 else c.toNat % 32
  let v := (b.toNat <<< 32) ||| a.toNat
  if left then ((v <<< n) >>> 32).toUInt64 &&& 0xFFFFFFFF
  else (v >>> n).toUInt64 &&& 0xFFFFFFFF

/-- `lop3` (§9.7.9.6): bit `i` of the result is bit `(a_i, b_i, c_i)` of `lut`. -/
def lop3 (lut a b c : UInt64) : UInt64 :=
  (List.range 32).foldl (fun d i =>
    let k : UInt64 := ((a >>> i.toUInt64) &&& 1) <<< 2 ||| ((b >>> i.toUInt64) &&& 1) <<< 1 |||
      ((c >>> i.toUInt64) &&& 1)
    if (lut >>> k) &&& 1 == 1 then d ||| ((1 : UInt64) <<< i.toUInt64) else d) (0 : UInt64)

/-- `prmt` (§9.7.10.7). -/
def prmt (mode : Option Flag) (a b c : UInt64) : UInt64 :=
  let lo32 : UInt64 := 0xFFFFFFFF
  let bytes : UInt64 := (a &&& lo32) ||| ((b &&& lo32) <<< 32)
  let byte (i : Nat) : UInt64 := (bytes >>> (8 * (i % 8)).toUInt64) &&& 0xFF
  let pick (sel : Nat) : UInt64 :=
    let v := byte (sel &&& 7)
    if sel &&& 8 != 0 then (if v &&& 0x80 != 0 then 0xFF else 0) else v
  let assemble (sels : List Nat) : UInt64 :=
    sels.zipIdx.foldl (fun d (s, i) => d ||| (pick s <<< (8 * i).toUInt64)) (0 : UInt64)
  let assembleRaw (sels : List Nat) : UInt64 :=
    sels.zipIdx.foldl (fun d (s, i) => d ||| (byte s <<< (8 * i).toUInt64)) (0 : UInt64)
  let n := (c &&& 3).toNat
  match mode with
  | none => assemble ((List.range 4).map fun i => ((c >>> (4 * i).toUInt64) &&& 0xF).toNat)
  | some .f4e => assembleRaw ((List.range 4).map (· + n))
  | some .b4e => assembleRaw ((List.range 4).map fun i => (8 + n - i) % 8)
  | some .rc8 => assembleRaw [n, n, n, n]
  | some .ecl => assembleRaw ((List.range 4).map fun i => Nat.max n i)
  | some .ecr => assembleRaw ((List.range 4).map fun i => Nat.min n i)
  | some .rc16 => assembleRaw (if n &&& 1 == 0 then [0, 1, 0, 1] else [2, 3, 2, 3])
  | some _ => 0

/-- Integer comparison of `setp`/`set` (§9.7.7.1): `lo ls hi hs` and bit types compare
unsigned. -/
def cmp (t : Ty) (op : CmpOp) (a b : UInt64) : Option Bool :=
  let s := t.isSigned
  let x := toInt s t.bits a
  let y := toInt s t.bits b
  let ux := mask t.bits a
  let uy := mask t.bits b
  match op with
  | .eq => some (x == y) | .ne => some (x != y)
  | .lt => some (x < y) | .le => some (x ≤ y) | .gt => some (x > y) | .ge => some (x ≥ y)
  | .lo => some (ux < uy) | .ls => some (ux ≤ uy) | .hi => some (ux > uy) | .hs => some (ux ≥ uy)
  | _ => none

def boolOp : BoolOp → Bool → Bool → Bool
  | .and, x, y => x && y
  | .or, x, y => x || y
  | .xor, x, y => x != y

/-- Integer-to-integer `cvt` (§9.7.10.24): extend or truncate; `.sat` clamps to the
destination's range. -/
def cvt (d s : Ty) (sat : Bool) (a : UInt64) : UInt64 :=
  let x := toInt s.isSigned s.bits a
  if sat then (if d.isSigned then clampS d.bits x else clampU d.bits x) else ofInt d.bits x

end NV.PTX.Sem.Int
