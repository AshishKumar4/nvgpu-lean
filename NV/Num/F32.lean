import NV.Num.Arith

/-!
# Binary32 arithmetic on machine words

`add`, `mul` and `fma` of `NV.Num` (`Arith`) for binary32, specialised: operands decoded with
bit operations, exact intermediate values as small naturals (products of 24-bit significands;
sums aligned within 64 bits, the far smaller operand of a sum kept as a sticky stand-in that
rounds identically), and one rounding written out for binary32. The SASS fast path uses these
(`NV.SASS.Fast`); `NV.Num`'s general definitions stay the specification, and
`lake exe num-tests` checks the two agree on every edge-value combination and on random operands
of every class, under every rounding mode with and without `.ftz`.
-/

namespace NV.Num.F32

/-- A decoded binary32: `(-1)^neg · m · 2^e` when finite (`m < 2^24`; zero when `m = 0`). -/
inductive V where
  | nan
  | inf (neg : Bool)
  | fin (neg : Bool) (m : Nat) (e : Int)

@[inline] def decode (ftz : Bool) (x : UInt32) : V :=
  let neg := x >>> 31 == 1
  let ef := (x >>> 23) &&& 0xff
  let fr := x &&& 0x7fffff
  if ef == 0xff then (if fr == 0 then .inf neg else .nan)
  else if ef == 0 then (if ftz || fr == 0 then .fin neg 0 0 else .fin neg fr.toNat (-149 : Int))
  else .fin neg (fr ||| 0x800000).toNat ((ef.toNat : Int) - 150)

def nan : UInt32 := 0x7fffffff

@[inline] def inf (neg : Bool) : UInt32 := if neg then 0xff800000 else 0x7f800000

@[inline] def zero (neg : Bool) : UInt32 := if neg then 0x80000000 else 0

/-- `Format.round .f32 r .ieee neg m e sticky`, then the `.ftz` flush of a subnormal result. -/
def round (r : Rounding) (ftz : Bool) (neg : Bool) (m : Nat) (e : Int) (sticky : Bool) : UInt32 :=
  if m == 0 then zero neg else
  let q : Int := Max.max (e + (Nat.log2 m : Int) - 23) (-149)
  let (hi, q) : Nat × Int :=
    if q ≤ e then (m <<< (e - q).toNat, q) else
      let k := (q - e).toNat
      let t := m >>> k
      let t := if r.up neg (t % 2 == 1) (m % 2 ^ k) k sticky then t + 1 else t
      if t == 2 ^ 24 then (2 ^ 23, q + 1) else (t, q)
  if hi == 0 then zero neg else
  let ef : Nat := if hi < 2 ^ 23 then 0 else (q + 150).toNat
  let frac := hi % 2 ^ 23
  let s : UInt32 := if neg then 0x80000000 else 0
  if ef ≥ 255 then (if r.away neg then inf neg else s ||| 0x7f7fffff)
  else if ftz && ef == 0 then s
  else s ||| (ef.toUInt32 <<< 23) ||| frac.toUInt32

/-- Stand in for a value far below another: a value of the same sign below `2^(top - 62)`, where
`top` is the larger one's leading bit. The exact sum then rounds as with the true value (the
stand-in and it both lie strictly inside an interval no rounding boundary of a 24-bit result
crosses), and the aligned sum stays within 64 bits. -/
@[inline] def far (top : Int) (m : Nat) (e : Int) : Nat × Int :=
  if m == 0 then (0, e) else
  if top - (e + (Nat.log2 m : Int)) > 62 then (1, top - 64) else (m, e)

/-- The exact sum of two finite values (`NV.Num.addExact`), with a far smaller operand replaced by
its stand-in (`far`): the sticky information it carries is all the rounding reads. -/
def addExact (r : Rounding) (n1 : Bool) (m1 : Nat) (e1 : Int) (n2 : Bool) (m2 : Nat) (e2 : Int) :
    Bool × Nat × Int :=
  let t1 : Int := e1 + (Nat.log2 m1 : Int)
  let t2 : Int := e2 + (Nat.log2 m2 : Int)
  let (m1, e1) := if m2 != 0 then far t2 m1 e1 else (m1, e1)
  let (m2, e2) := if m1 != 0 then far t1 m2 e2 else (m2, e2)
  let e := Min.min e1 e2
  let a : Int := (m1 <<< (e1 - e).toNat : Nat)
  let b : Int := (m2 <<< (e2 - e).toNat : Nat)
  let s := (if n1 then -a else a) + (if n2 then -b else b)
  if s = 0 then (if n1 == n2 then n1 else r == .rm, 0, e) else (s < 0, s.natAbs, e)

def add (r : Rounding) (ftz : Bool) (a b : UInt32) : UInt32 :=
  match decode ftz a, decode ftz b with
  | .nan, _ | _, .nan => nan
  | .inf n1, .inf n2 => if n1 == n2 then inf n1 else nan
  | .inf n, .fin .. | .fin .., .inf n => inf n
  | .fin n1 m1 e1, .fin n2 m2 e2 =>
    let (n, m, e) := addExact r n1 m1 e1 n2 m2 e2
    round r ftz n m e false

def mul (r : Rounding) (ftz : Bool) (a b : UInt32) : UInt32 :=
  match decode ftz a, decode ftz b with
  | .nan, _ | _, .nan => nan
  | .inf n1, .inf n2 => inf (n1 != n2)
  | .inf n1, .fin n2 m _ | .fin n2 m _, .inf n1 => if m == 0 then nan else inf (n1 != n2)
  | .fin n1 m1 e1, .fin n2 m2 e2 => round r ftz (n1 != n2) (m1 * m2) (e1 + e2) false

def fma (r : Rounding) (ftz : Bool) (a b c : UInt32) : UInt32 :=
  let addInf (n : Bool) : V → UInt32
    | .inf n3 => if n == n3 then inf n else nan
    | _ => inf n
  match decode ftz a, decode ftz b, decode ftz c with
  | .nan, _, _ | _, .nan, _ | _, _, .nan => nan
  | .inf n1, .inf n2, c' => addInf (n1 != n2) c'
  | .inf n1, .fin n2 m _, c' | .fin n2 m _, .inf n1, c' =>
    if m == 0 then nan else addInf (n1 != n2) c'
  | .fin _ _ _, .fin _ _ _, .inf n => inf n
  | .fin n1 m1 e1, .fin n2 m2 e2, .fin n3 m3 e3 =>
    let (n, m, e) := addExact r (n1 != n2) (m1 * m2) (e1 + e2) n3 m3 e3
    round r ftz n m e false

/-- `NV.Num.minMax` for binary32 without `.xorsign.abs`: a NaN operand loses unless `nanProp`; two
NaNs give the canonical NaN; `-0 < +0`. -/
def minMax (isMax ftz nanProp : Bool) (a b : UInt32) : UInt32 :=
  let flush (x : UInt32) : UInt32 :=
    if ftz && (x >>> 23) &&& 0xff == 0 && x &&& 0x7fffff != 0 then x &&& 0x80000000 else x
  let a := flush a
  let b := flush b
  let isNaN (x : UInt32) : Bool := (x >>> 23) &&& 0xff == 0xff && x &&& 0x7fffff != 0
  if nanProp && (isNaN a || isNaN b) then nan else
  if isNaN a && isNaN b then nan else
  if isNaN a then b else if isNaN b then a else
  -- The magnitude negated below `-0`'s key for negatives: an order on the encodings.
  let key (x : UInt32) : Int :=
    let m : Int := (x &&& 0x7fffffff).toNat
    if x >>> 31 == 1 then -m - 1 else m
  if (key a < key b) != isMax then a else b

end NV.Num.F32
