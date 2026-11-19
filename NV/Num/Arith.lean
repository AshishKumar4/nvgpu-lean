import NV.Num.Round

/-!
# Floating-point arithmetic

The IEEE 754 operations of PTX's floating-point instructions (ISA 9.5 §9.7.3–9.7.6) on encodings:
each result is the exact result rounded once (`Format.round`), with NVIDIA's rules for what IEEE
leaves open:

* `ftz` flushes subnormal inputs and results to sign-preserving zero (§9.7.3).
* NaN results: binary64 instructions carry a NaN operand's payload (§9.7.3), an invalid operation
  `CUDART_NAN`; every other format returns the canonical NaN (`Format.nan`), which is what PTX
  calls "an unspecified NaN".
* `min`/`max` order `-0 < +0` (§9.7.3.11–12); comparisons treat them as equal.
-/

namespace NV.Num

namespace Format

def one (f : Format) : Nat := f.pack false f.bias 0

/-- Flip the sign bit. -/
def negate (f : Format) (b : Nat) : Nat :=
  let (neg, e, frac) := f.fields b
  f.pack (!neg) e frac

def decodeIn (f : Format) (ftz : Bool) (b : Nat) : Val := f.decode (if ftz then f.flush b else b)

/-- The NaN an instruction returns, given its NaN operands in operand order. Binary64 returns the
first one quieted (which operand wins is inferred), and an invalid operation without NaN operands
`0xfff8000000000000` (`CUDART_NAN`, `math_constants.h`: the default NaN of the fp64 unit, which the
SASS semantics' `DFMA` gives too); every other format gives the canonical NaN. -/
def nanOf (f : Format) (ops : List Nat) : Nat :=
  if f == .f64 then
    match ops.find? f.isNaN with
    | some b => b % 2 ^ 64 ||| 2 ^ 51
    | none => 0xfff8000000000000
  else f.nan

end Format

/-- The exact sum of two finite values. An exact zero sum has the sign IEEE 754 §6.3 gives it:
the operands' common sign, otherwise `+0` (`-0` when rounding toward `-∞`). -/
def addExact (r : Rounding) (n1 : Bool) (m1 : Nat) (e1 : Int) (n2 : Bool) (m2 : Nat) (e2 : Int) :
    Bool × Nat × Int :=
  let e := min e1 e2
  let a : Int := (m1 * pow2 (e1 - e).toNat : Nat)
  let b : Int := (m2 * pow2 (e2 - e).toNat : Nat)
  let s := (if n1 then -a else a) + (if n2 then -b else b)
  if s = 0 then (if n1 == n2 then n1 else r == .rm, 0, e) else (s < 0, s.natAbs, e)

open Format in
def add (f : Format) (r : Rounding) (ftz : Bool) (a b : Nat) : Nat :=
  match f.decodeIn ftz a, f.decodeIn ftz b with
  | .nan, _ | _, .nan => f.nanOf [a, b]
  | .inf n1, .inf n2 => if n1 == n2 then f.inf n1 else f.nanOf []
  | .inf n, .fin .. | .fin .., .inf n => f.inf n
  | .fin n1 m1 e1, .fin n2 m2 e2 =>
    let (n, m, e) := addExact r n1 m1 e1 n2 m2 e2
    f.flushResult ftz (f.round r .ieee n m e)

def sub (f : Format) (r : Rounding) (ftz : Bool) (a b : Nat) : Nat :=
  if f.isNaN b then add f r ftz a b else add f r ftz a (f.negate b)

def mul (f : Format) (r : Rounding) (ftz : Bool) (a b : Nat) : Nat :=
  match f.decodeIn ftz a, f.decodeIn ftz b with
  | .nan, _ | _, .nan => f.nanOf [a, b]
  | .inf n1, .inf n2 => f.inf (n1 != n2)
  | .inf n1, .fin n2 m _ | .fin n2 m _, .inf n1 => if m = 0 then f.nanOf [] else f.inf (n1 != n2)
  | .fin n1 m1 e1, .fin n2 m2 e2 =>
    f.flushResult ftz (f.round r .ieee (n1 != n2) (m1 * m2) (e1 + e2))

/-- Fused multiply-add `a · b + c` with one rounding (PTX `fma`, and `mad` for `sm_20`+). -/
def fma (f : Format) (r : Rounding) (ftz : Bool) (a b c : Nat) : Nat :=
  match f.decodeIn ftz a, f.decodeIn ftz b, f.decodeIn ftz c with
  | .nan, _, _ | _, .nan, _ | _, _, .nan => f.nanOf [a, b, c]
  | .inf n1, .inf n2, c' => addInf (n1 != n2) c'
  | .inf n1, .fin n2 m _, c' | .fin n2 m _, .inf n1, c' =>
    if m = 0 then f.nanOf [] else addInf (n1 != n2) c'
  | .fin _ _ _, .fin _ _ _, .inf n => f.inf n
  | .fin n1 m1 e1, .fin n2 m2 e2, .fin n3 m3 e3 =>
    let (n, m, e) := addExact r (n1 != n2) (m1 * m2) (e1 + e2) n3 m3 e3
    f.flushResult ftz (f.round r .ieee n m e)
where
  /-- An infinite product plus `c`. -/
  addInf (n : Bool) : Val → Nat
    | .inf n3 => if n == n3 then f.inf n else f.nanOf []
    | _ => f.inf n

/-- The quotient `m1 · 2^e1 / (m2 · 2^e2)` (with `m2 > 0`) as a truncated value with at least
`p + 2` bits and a sticky remainder, ready for rounding to precision `p`. -/
def divExact (p m1 : Nat) (e1 : Int) (m2 : Nat) (e2 : Int) : Nat × Int × Bool :=
  let s := p + 2 + Nat.log2 m2 + 1 - Nat.log2 m1
  ((m1 * pow2 s) / m2, e1 - e2 - s, (m1 * pow2 s) % m2 != 0)

def div (f : Format) (r : Rounding) (ftz : Bool) (a b : Nat) : Nat :=
  match f.decodeIn ftz a, f.decodeIn ftz b with
  | .nan, _ | _, .nan => f.nanOf [a, b]
  | .inf _, .inf _ => f.nanOf []
  | .inf n1, .fin n2 _ _ => f.inf (n1 != n2)
  | .fin n1 _ _, .inf n2 => f.zero (n1 != n2)
  | .fin n1 m1 e1, .fin n2 m2 e2 =>
    if m2 = 0 then (if m1 = 0 then f.nanOf [] else f.inf (n1 != n2))
    else if m1 = 0 then f.zero (n1 != n2)
    else
      let (q, e, sticky) := divExact (f.man + 1) m1 e1 m2 e2
      f.flushResult ftz (f.round r .ieee (n1 != n2) q e sticky)

def rcp (f : Format) (r : Rounding) (ftz : Bool) (a : Nat) : Nat := div f r ftz f.one a

def sqrt (f : Format) (r : Rounding) (ftz : Bool) (a : Nat) : Nat :=
  match f.decodeIn ftz a with
  | .nan => f.nanOf [a]
  | .inf n => if n then f.nanOf [] else f.inf false
  | .fin n m e =>
    if m = 0 then f.zero n
    else if n then f.nanOf []
    else
      -- An even exponent and at least 2(p + 2) bits, so the root has p + 2.
      let s0 := 2 * (f.man + 3) - Nat.log2 m
      let s := s0 + ((e - s0) % 2).toNat
      let x := m * pow2 s
      let root := Nat.sqrt x
      f.flushResult ftz (f.round r .ieee false root ((e - s) / 2) (root * root != x))

/-- Clamp to `[0, 1]`, NaN to `+0` (`.sat`, PTX §9.7.3). A negative zero becomes `+0` (inferred). -/
def satUnit (f : Format) (b : Nat) : Nat :=
  if f.isNaN b || f.isNeg b then f.zero false
  else if f.magnitude b > f.one then f.one
  else b

/-- Clamp negative results to `+0`, NaN to the canonical NaN (`.relu`, PTX §9.7.4.4). A negative
zero becomes `+0` (inferred). -/
def relu (f : Format) (b : Nat) : Nat :=
  if f.isNaN b then f.nan else if f.isNeg b then f.zero false else b

/-- A key ordering the non-NaN encodings by value, `-0` below `+0` when `signedZero`. -/
def Format.orderKey (f : Format) (signedZero : Bool) (b : Nat) : Int :=
  let m : Int := f.magnitude b
  if f.isNeg b then (if signedZero then -m - 1 else -m) else m

/-- `min` and `max` (PTX §9.7.3.11–12, §9.7.4.7–8): a NaN operand loses unless `nanProp`
(`.NaN`, canonical NaN); two NaNs give NaN (binary64: a payload, `Format.nanOf`); `-0 < +0`; and
`.xorsign.abs` takes the larger or smaller magnitude with the sign the XOR of the operands'
signs. -/
def minMax (isMax : Bool) (f : Format) (ftz nanProp xorsignAbs : Bool) (a b : Nat) : Nat :=
  let a := if ftz then f.flush a else a
  let b := if ftz then f.flush b else b
  let s := f.isNeg a != f.isNeg b
  let (a, b) := if xorsignAbs then (f.magnitude a, f.magnitude b) else (a, b)
  if nanProp && (f.isNaN a || f.isNaN b) then f.nan else
  if f.isNaN a && f.isNaN b then f.nanOf [a, b] else
  let d :=
    if f.isNaN a then b else if f.isNaN b then a
    else if (f.orderKey true a < f.orderKey true b) != isMax then a else b
  if xorsignAbs then (if s then f.negate d else d) else d

def min (f : Format) (ftz nanProp xorsignAbs : Bool) (a b : Nat) : Nat :=
  minMax false f ftz nanProp xorsignAbs a b

def max (f : Format) (ftz nanProp xorsignAbs : Bool) (a b : Nat) : Nat :=
  minMax true f ftz nanProp xorsignAbs a b

/-- `abs` (PTX §9.7.3.9): binary64 passes a NaN through unchanged; the other formats return the
canonical NaN (PTX: unspecified). -/
def abs (f : Format) (ftz : Bool) (a : Nat) : Nat :=
  if f.isNaN a then (if f == .f64 then a else f.nan)
  else f.magnitude (if ftz then f.flush a else a)

/-- `neg` (PTX §9.7.3.10): a NaN gives the canonical NaN, binary64 its operand with the sign
flipped (PTX: unspecified; inferred). -/
def neg (f : Format) (ftz : Bool) (a : Nat) : Nat :=
  if f.isNaN a then (if f == .f64 then f.negate a else f.nan)
  else f.negate (if ftz then f.flush a else a)

/-- `copysign a b`: `b` with the sign of `a` (PTX §9.7.3.2). -/
def copysign (f : Format) (a b : Nat) : Nat :=
  let (_, e, frac) := f.fields b
  f.pack (f.isNeg a) e frac

inductive TestClass where
  | finite | infinite | number | notanumber | normal | subnormal
  deriving DecidableEq, Repr, Inhabited

/-- `testp` (PTX §9.7.3.1); zeros count as normal. -/
def testp (f : Format) (c : TestClass) (a : Nat) : Bool :=
  match c with
  | .finite => !f.isNaN a && !f.isInf a
  | .infinite => f.isInf a
  | .number => !f.isNaN a
  | .notanumber => f.isNaN a
  | .normal => !f.isNaN a && !f.isInf a && !f.isSubnormal a
  | .subnormal => f.isSubnormal a

/-- The comparisons of `setp`/`set` (PTX §9.7.7.1, Table 30); the `u` forms are true also when
an operand is NaN. -/
inductive CmpOp where
  | eq | ne | lt | le | gt | ge | equ | neu | ltu | leu | gtu | geu | num | nan
  deriving DecidableEq, Repr, Inhabited

def cmp (f : Format) (op : CmpOp) (ftz : Bool) (a b : Nat) : Bool :=
  let a := if ftz then f.flush a else a
  let b := if ftz then f.flush b else b
  let un := f.isNaN a || f.isNaN b
  let x := f.orderKey false a
  let y := f.orderKey false b
  match op with
  | .eq => !un && x == y
  | .ne => !un && x != y
  | .lt => !un && decide (x < y)
  | .le => !un && decide (x ≤ y)
  | .gt => !un && decide (x > y)
  | .ge => !un && decide (x ≥ y)
  | .equ => un || x == y
  | .neu => un || x != y
  | .ltu => un || decide (x < y)
  | .leu => un || decide (x ≤ y)
  | .gtu => un || decide (x > y)
  | .geu => un || decide (x ≥ y)
  | .num => !un
  | .nan => un

end NV.Num
