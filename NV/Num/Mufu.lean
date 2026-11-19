import NV.Num.Convert
import NV.Num.MufuTables
import NV.Fact

/-!
# The special-function unit (MUFU)

`rcp`, `sqrt`, `rsqrt`, `lg2`, `ex2`, `sin` and `cos` with `.approx.f32` run on the SM's
special-function unit, whose results are not correctly rounded. This is Carleton's bit-accurate
model of it (PTX Math, `ptxmath` in `docs/sources.md`): a table lookup on the input's leading
significand bits, a quadratic `c₀ + c₁·x_l ± c₂·x_l²` with a truncated squarer, a bias, and
truncation to 23 fraction bits; around it, the scaling for subnormal inputs and
results of the non-`.ftz` forms. `sin`/`cos` are the `sm_70` variant. Hopper and later keep the
unit (inferred).

The `.ftz` forms flush subnormal inputs and results (PTX §9.7.3); their unscaled path is the same
interpolation (inferred).

Everything is integer arithmetic on encodings, the model's few binary32/binary64 steps included.
-/

namespace NV.Num.Mufu

/-- The functions the model covers. -/
def validated : Fact String :=
  { value := "rcp sqrt rsqrt lg2 ex2 sin cos (.approx.f32)", kind := .published
    source := { ref := "ptxmath", loc := "README" }
    measuredOn := some "sm_50, sm_60, sm_70, sm_75"
    note := "" }

private def mask (n : Nat) : Nat := pow2 n - 1

/-- The partial-product squarer: `x²` less the bits of the 9 low partial-product rows and the
antidiagonal that fall in the 19 truncated columns (`common/squarer.c`). On machine words for
the significand pieces the unit squares (below `2^32`), on `Nat` beyond. -/
def squareApprox (x : Nat) : Nat :=
  if x < 2 ^ 32 then
    let w := x.toUInt64
    Id.run do
      -- the antidiagonal x_i·x_i sits at column 2i
      let mut diag : UInt64 := 0
      for i in [0:16] do
        if (w >>> i.toUInt64) &&& 1 == 1 then diag := diag + ((1 : UInt64) <<< (2 * i).toUInt64)
      let m19 : UInt64 := 0x7ffff
      let mut err := diag &&& m19
      for i in [0:9] do
        if (w >>> i.toUInt64) &&& 1 == 1 then
          let row := (w <<< (i + 1).toUInt64) &&& 0xffffffff
          err := err + ((row - (row &&& (((1 : UInt64) <<< (2 * i + 2).toUInt64) - 1))) &&& m19)
      let sq := w * w
      return if err ≤ sq then (sq - err).toNat else 0
  else Id.run do
    let mut diag := 0
    for i in [0:16] do
      if x.testBit i then diag := diag + pow2 (2 * i)
    let mut err := diag % 2 ^ 19
    for i in [0:9] do
      if x.testBit i then
        let row := (x * pow2 (i + 1)) % 2 ^ 32
        err := err + (row - row % pow2 (2 * i + 2)) % 2 ^ 19
    return x * x - err

/-- One interpolation: `x_h` indexes the table, `x_l` is the remaining significand. The signs of
the linear and quadratic terms and the alignments are each function's; the sum wraps at 64 bits
as in hardware. -/
structure Poly where
  table : Array (Nat × Nat × Nat)
  a0 : Nat
  a1 : Nat
  /-- Subtract the linear term. -/
  neg1 : Bool
  /-- Subtract the quadratic term. -/
  neg2 : Bool
  /-- The sum's weight: the result's units bit. -/
  weight : Nat
  bias : Nat

def Poly.eval (p : Poly) (index xl sq : Nat) : Nat :=
  let (c0, c1, c2) := p.table[index]!
  -- Modulo 2^64 throughout, as the sum: machine words.
  let sh (v : UInt64) (k : Nat) : UInt64 := if k < 64 then v <<< k.toUInt64 else 0
  let t1 := sh (c1.toUInt64 * xl.toUInt64) p.a1
  let t2 := c2.toUInt64 * sq.toUInt64
  let s := sh c0.toUInt64 p.a0 + (if p.neg1 then 0 - t1 else t1) + (if p.neg2 then 0 - t2 else t2)
  (s + (p.bias / pow2 (64 - p.weight + 23)).toUInt64).toNat

/-- The 23 fraction bits below the sum's units bit. -/
private def frac (p : Poly) (sum : Nat) : Nat := (sum / pow2 (p.weight - 23)) % 2 ^ 23

def ex2Poly : Poly :=
  { table := ex2Table, a0 := 32, a1 := 19, neg1 := false, neg2 := false, weight := 57,
    bias := 0x6fc4000000000000 }
def lg2Poly : Poly :=
  { table := lg2Table, a0 := 30, a1 := 18, neg1 := false, neg2 := true, weight := 56,
    bias := 0x868b000000000000 }
def rcpPoly : Poly :=
  { table := rcpTable, a0 := 30, a1 := 17, neg1 := true, neg2 := false, weight := 56,
    bias := 0x67e7000000000000 }
def rsqrtPoly : Poly :=
  { table := rsqrtTable, a0 := 31, a1 := 17, neg1 := true, neg2 := false, weight := 57,
    bias := 0x7fff800000000000 }
def sqrtPoly : Poly :=
  { table := sqrtTable, a0 := 33, a1 := 18, neg1 := false, neg2 := true, weight := 58,
    bias := 0x7fff800000000000 }
def sinPoly : Poly :=
  { table := sinTable, a0 := 31, a1 := 19, neg1 := false, neg2 := true, weight := 56, bias := 0 }

/-! ## Binary32 steps -/

private def f := Format.f32

private def bits (neg : Bool) (e fr : Nat) : Nat := f.pack neg e fr

/-- `x · 2^k` rounded to nearest (C's `ldexpf`). -/
private def ldexp (x : Nat) (k : Int) : Nat :=
  match f.decode x with
  | .fin n m e => f.round .rn .ieee n m (e + k)
  | _ => x

/-- The exponent of a normal encoding (C's `ilogbf`). -/
private def ilogb (x : Nat) : Int := ((f.fields x).2.1 : Int) - 127

/-- A subnormal input scaled by `2^24` into the normal range, and the scale's exponent. -/
private def normalize (x : Nat) : Nat × Int :=
  if f.isSubnormal x then (mul f .rn false x (bits false (127 + 24) 0), -24) else (x, 0)

/-- `x` with its significand split for a table of `2^m` segments. -/
private def split (x m : Nat) : Nat × Nat := ((x / pow2 (23 - m)) % pow2 m, x % pow2 (23 - m))

/-! ## The functions (PTX semantics of the `.approx.f32` instructions) -/

def rcp (x : Nat) : Nat :=
  if f.isNaN x then f.nan
  else if f.isInf x then f.zero (f.isNeg x)
  else if f.isZero x then f.inf (f.isNeg x)
  else
    let (x, s) := normalize x
    let (xh, xl) := split x 7
    let sum := rcpPoly.eval xh xl (squareApprox (2 * xl) / 4)
    let (re, sum) := if sum / 2 ^ 56 == 0 then (126, 2 * sum) else (127, sum)
    ldexp (bits (f.isNeg x) re (frac rcpPoly sum)) (-(ilogb x + s))

def rsqrt (x : Nat) : Nat :=
  if f.isNaN x then f.nan
  else if f.isInf x then (if f.isNeg x then f.nan else f.zero false)
  else if f.isZero x then f.inf (f.isNeg x)
  else if f.isNeg x then f.nan
  else
    let (x, s) := normalize x
    let l := ilogb x + s
    let (xh, xl) := split x 6
    if l % 2 == 0 && x % 2 ^ 23 == 0 then ldexp f.one (-(l / 2))
    else
      let (index, l) := if l % 2 == 0 then (xh, l) else (xh + 64, l - 1)
      let sum := 2 * rsqrtPoly.eval index xl (squareApprox xl)
      ldexp (bits false 126 (frac rsqrtPoly sum)) (-(l / 2))

def sqrt (x : Nat) : Nat :=
  if f.isNaN x then f.nan
  else if f.isInf x then (if f.isNeg x then f.nan else f.inf false)
  else if f.isZero x then x
  else if f.isNeg x then f.nan
  else
    let (x, s) := normalize x
    let l := ilogb x + s
    let (xh, xl) := split x 6
    let (index, l) := if l % 2 == 0 then (xh, l) else (xh + 64, l - 1)
    let sum := sqrtPoly.eval index xl (squareApprox xl)
    ldexp (bits false 127 (frac sqrtPoly sum)) (l / 2)

def lg2 (x : Nat) : Nat :=
  if f.isNaN x then f.nan
  else if f.isInf x then (if f.isNeg x then f.nan else f.inf false)
  else if f.isZero x then f.inf true
  else if f.isNeg x then f.nan
  else if x == f.one then f.zero false
  else
    let (y, s) := normalize x
    let (xh, xl) := split y 6
    let sum : Int := lg2Poly.eval xh xl (squareApprox xl) + ilogb y * 2 ^ 56
    let sum := sum.natAbs / 2 ^ 20 * 2 ^ 20
    -- normalize to the units bit 56
    let (re, sum) := Id.run do
      let mut re := 127
      let mut v := sum
      for _ in [0:64] do
        if v / 2 ^ 56 == 0 && v != 0 then re := re - 1; v := 2 * v
        else if v / 2 ^ 57 != 0 then re := re + 1; v := v / 2
        else break
      return (re, v)
    let r := bits (decide (f.magnitude y < f.one) && !f.isNeg y) re (frac lg2Poly sum)
    if s == 0 then r else sub f .rn false r (bits false (127 + 4) (2 ^ 23 / 2))

/-- `2^n` for an integral binary32 `n` (`fexp2i`): infinity above 127, zero below -126. -/
private def exp2i (n : Int) : Nat :=
  if n > 127 then f.inf false else if n < -126 then f.zero false else ldexp f.one n

/-- `ex2.approx.f32`; `ftz` drops the halving and squaring that reach subnormal
results. -/
def ex2 (ftz : Bool) (x : Nat) : Nat :=
  if f.isNaN x then f.nan
  else if f.isInf x then (if f.isNeg x then f.zero false else f.inf false)
  else if f.isZero x || f.isSubnormal x then f.one
  else
    let half := !ftz && cmp f .lt false x (bits true (127 + 6) (2 ^ 23 - 2 ^ 18))
    let x := if half then mul f .rn false x (bits false 126 0) else x
    let (n, m, e) := match f.decode x with
      | .fin n m e => (n, m, e)
      | _ => (false, 0, 0)
    -- integral and fractional parts, truncated toward zero
    let ip : Nat := if e ≥ 0 then m * pow2 e.toNat else m / pow2 (-e).toNat
    let fp : Nat := if e ≥ 0 then 0 else m % pow2 (-e).toNat
    let k := (-e).toNat
    -- the fraction's magnitude in fixed point with 23 fraction bits, truncated
    let xb : Nat := fp * 2 ^ 23 / pow2 k
    let (xb, i) : Nat × Int :=
      if n && xb != 0 then (2 ^ 32 - 1 - xb, -(ip : Int) - 1) else (xb, if n then -(ip : Int) else ip)
    let (xh, xl) := split (xb % 2 ^ 23) 6
    let sum := ex2Poly.eval xh xl (squareApprox xl)
    let rec_ := mul f .rn false (exp2i i) (bits false 127 (frac ex2Poly sum))
    let r := if half then mul f .rn false rec_ rec_ else rec_
    if ftz then f.flush r else r

/-! ### `sin` and `cos`: range reduction then the unit -/

private def d := Format.f64
private def piOver2Rcp : Nat := 0x3fe45f3060000000
private def piOver2 : Nat := 0x3ff921fb6546df20
private def q0Max : Nat := 0x3ff921fb60000000

/-- The exponent of a nonzero binary64 (C's `ilogb`). -/
private def ilogbD (x : Nat) : Int :=
  match d.decode x with
  | .fin _ m e => e + Nat.log2 m
  | _ => 0

/-- The reduced argument: sign, quadrant and a 23-bit fraction of a quarter turn, as the
`sm_70` unit receives it (`reduction/rro_sincos.c`; NaN and infinity marked in the quadrant). -/
def reduce (x : Nat) : Nat :=
  let sign := f.isNeg x
  let pk (q r : Nat) : Nat := (if sign then 2 ^ 31 else 0) ||| (q * 2 ^ 23) ||| r
  if f.isNaN x then pk 0x80 0
  else if f.isInf x then pk 0x81 0
  else if f.isZero x || f.isSubnormal x then pk 0 0
  else
    let x := f.magnitude x
    if x ≥ bits false (127 + 33) 0 then pk 0 0
    else
      let xd := cvt f d .rn {} x
      let kd := mul d .rn false xd piOver2Rcp
      let k := toInt d .rz false 64 false kd
      let reduced := sub d .rn false xd (mul d .rn false (ofInt d .rn false 64 k) piOver2)
      let clamped := if cmp d .lt false reduced 0 then 0
        else if cmp d .gt false reduced q0Max then q0Max else reduced
      let t := mul d .rn false clamped piOver2Rcp
      let r := if d.isZero t then 0 else
        let tb := (t / 2 ^ 29) % 2 ^ 32
        let tb := (tb ||| 2 ^ 23) % 2 ^ 24
        tb / 2 ^ (Min.min (-(ilogbD t)) 24).toNat
      let packed := pk (k % 4) r
      let packed := packed - packed % pow2 (Max.max (ilogb x - 8) 0).toNat
      -- sm_70: the fraction loses the bits the quadrant count occupies
      let kl := Min.min (Max.max (ilogbD kd) 0) 25
      packed - packed % pow2 kl.toNat

private def poly (reduced : Nat) (sign : Bool) : Nat := Id.run do
  let (xh, xl) := split (reduced % 2 ^ 23) 6
  let mut sum := sinPoly.eval xh xl (squareApprox xl)
  let mut re := 0
  if sum != 0 then
    re := 127
    for _ in [0:64] do
      if sum / 2 ^ 56 == 0 then re := re - 1; sum := 2 * sum
  return bits sign re (frac sinPoly sum)

/-- The reduced argument of an angle in turns (`.2pi`, and `MUFU.SIN`/`MUFU.COS`'s own input on
`sm_70`+): `4|x|` exactly, its quadrant and the 23 bits of its fraction (inferred;
`num-tests turns` compares it with `reduce`). -/
def reduceTurns (x : Nat) : Nat :=
  let sign := f.isNeg x
  let pk (q r : Nat) : Nat := (if sign then 2 ^ 31 else 0) ||| (q * 2 ^ 23) ||| r
  match f.decode x with
  | .nan => pk 0x80 0
  | .inf _ => pk 0x81 0
  | .fin _ m e =>
    if f.isSubnormal x then pk 0 0
    else
      let w := e + 2
      if w ≥ 0 then pk ((m * pow2 w.toNat) % 4) 0
      else pk ((m / pow2 (-w).toNat) % 4) ((m % pow2 (-w).toNat) * 2 ^ 23 / pow2 (-w).toNat)

/-- The unit's sine of a reduced argument. -/
def sinReduced (r : Nat) : Nat :=
  let q := (r / 2 ^ 23) % 2 ^ 8
  if q ≥ 0x80 then f.nan
  else
    let r' := if q % 2 == 1 then 2 ^ 32 - 1 - r else r
    poly r' ((r / 2 ^ 31 == 1) != (q / 2 % 2 == 1))

/-- The unit's cosine of a reduced argument. -/
def cosReduced (r : Nat) : Nat :=
  let q := (r / 2 ^ 23) % 2 ^ 8
  if q ≥ 0x80 then f.nan
  else
    let r' := if q % 2 == 0 then 2 ^ 32 - 1 - r else r
    poly r' ((q + 1) / 2 % 2 == 1)

/-- `sin.approx.f32`. -/
def sin (x : Nat) : Nat := sinReduced (reduce x)

/-- `MUFU.SIN` on an angle in turns. -/
def sinTurns (u : Nat) : Nat := sinReduced (reduceTurns u)

/-- `MUFU.COS` on an angle in turns. -/
def cosTurns (u : Nat) : Nat := cosReduced (reduceTurns u)

/-- `cos.approx.f32`. -/
def cos (x : Nat) : Nat := cosReduced (reduce x)

end NV.Num.Mufu
