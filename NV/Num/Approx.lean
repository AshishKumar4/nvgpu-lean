import NV.Num.Mufu

/-!
# The `.approx` functions

PTX documents the accuracy of `ex2`, `lg2`, `rsqrt`, `rcp`, `sqrt`, `div`, `sin`, `cos` and `tanh`
with `.approx` (ISA 9.5 §9.7.3.8–22, §9.7.4.9–10), and the documented bounds are the facts below.

* binary32 `rcp`, `sqrt`, `rsqrt`, `lg2`, `ex2`, `sin`, `cos`: the special-function unit's own
  results (`NV.Num.Mufu`, bit-accurate on `sm_50`–`sm_75`).
* `div.full.f32`: a scaled `a · MUFU.RCP(b)`.
* binary16/bfloat16 `ex2`, every `tanh`, binary64 `rcp`/`rsqrt`: no public model of
  the bits; the result here is the correctly rounded one (nearest-even), which lies inside every
  documented bound. It is computed with integer interval arithmetic: an enclosure
  `lo · 2^e ≤ y ≤ hi · 2^e` at a working precision that doubles until both ends round alike
  (Ziv's strategy); these results are never a rounding boundary except in the exact cases handled
  first (`2^x` and `tanh x` of a nonzero binary fraction are irrational), so the loop ends.

Corner cases follow PTX's tables.
-/

namespace NV.Num.Approx

/-! ## Documented accuracy -/

/-- A documented error bound: absolute or relative as a power of two (the exponent in
thousandths, so `2^-22.9` is `-22900`), or in ulps of the correctly rounded result. -/
inductive Bound where
  | abs (log2Milli : Int)
  | rel (log2Milli : Int)
  | ulp (n : Nat)
  deriving DecidableEq, Repr, Inhabited

structure Accuracy where
  bound : Bound
  /-- Where the bound holds, as the source states it. -/
  domain : String
  deriving DecidableEq, Repr, Inhabited

private def ptx (loc : String) (bound : Bound) (domain : String) : Fact Accuracy :=
  { value := { bound, domain }, kind := .documented, source := { ref := "ptx-isa-9.5", loc } }

def ex2F32 := ptx "§9.7.3.21" (.ulp 2) "full range"
def ex2F16 := ptx "§9.7.4.10" (.rel (-9900)) "all inputs"
def ex2BF16 := ptx "§9.7.4.10" (.rel (-7000)) "all inputs"
def lg2F32Near1 := ptx "§9.7.3.20" (.abs (-22000)) "(0.5, 2)"
def lg2F32 := ptx "§9.7.3.20" (.rel (-22000)) "positive finite outside (0.5, 2)"
def rsqrtF32 := ptx "§9.7.3.16" (.rel (-22900)) "positive finite"
def sqrtF32 := ptx "§9.7.3.15" (.rel (-23000)) "positive finite"
def rcpF32 := ptx "§9.7.3.13" (.ulp 1) "full range"
def divF32 := ptx "§9.7.3.8" (.ulp 2) "|b| in [2^-126, 2^126]"
def divFullF32 := ptx "§9.7.3.8" (.ulp 2) "full range"
def sinF32 := ptx "§9.7.3.18" (.abs (-20500)) "[-2π, 2π]"
def sinF32Wide := ptx "§9.7.3.18" (.abs (-14700)) "[-100π, 100π]"
def sinTurnsF32 := ptx "§9.7.3.18" (.abs (-20000)) "[-1, 1] turns (.2pi)"
def sinTurnsF32Wide := ptx "§9.7.3.18" (.abs (-14200)) "[-50, 50] turns (.2pi)"
def cosF32 := ptx "§9.7.3.19" (.abs (-20400)) "[-2π, 2π]"
def cosF32Wide := ptx "§9.7.3.19" (.abs (-14700)) "[-100π, 100π]"
def cosTurnsF32 := ptx "§9.7.3.19" (.abs (-19900)) "[-1, 1] turns (.2pi)"
def cosTurnsF32Wide := ptx "§9.7.3.19" (.abs (-14200)) "[-50, 50] turns (.2pi)"
def tanhF32 := ptx "§9.7.3.22" (.rel (-11000)) "full range"
def tanhF16 := ptx "§9.7.4.9" (.abs (-10987)) "all inputs"
def tanhBF16 := ptx "§9.7.4.9" (.abs (-8000)) "all inputs"

/-! ## Enclosures -/

private def cdiv (a b : Nat) : Nat := (a + b - 1) / b

/-- Round the real in `[lo · 2^e, hi · 2^e]`, where `encl p` gives the enclosure at working
precision `p`: the precision doubles from 64 until both ends round alike (at most 2048 bits;
then the lower end's rounding, an error no larger than the enclosure's width). -/
def zivRound (f : Format) (encl : Nat → Int × Int × Int) : Nat :=
  go 6 64
where
  rnd (v e : Int) : Nat := f.round .rn .ieee (v < 0) v.natAbs e
  go : Nat → Nat → Nat
    | 0, p => let (lo, _, e) := encl p; rnd lo e
    | k + 1, p =>
      let (lo, hi, e) := encl p
      let a := rnd lo e
      if a == rnd hi e then a else go k (2 * p)

/-- `2^(t / 2^k)` for `0 ≤ t < 2^k`, enclosed at scale `2^-p` by the square roots `2^(2^-i)`. -/
def exp2Frac (t k p : Nat) : Nat × Nat := Id.run do
  let one := pow2 p
  let mut rl := 2 * one
  let mut rh := 2 * one
  let mut lo := one
  let mut hi := one
  for i in [1:k + 1] do
    rl := Nat.sqrt (rl * one)
    rh := Nat.sqrt (rh * one) + 1
    if (t / pow2 (k - i)) % 2 == 1 then
      lo := lo * rl / one
      hi := cdiv (hi * rh) one
  return (lo, hi)

/-- `e^(z · 2^-p)` for `0 ≤ z ≤ 2^(p-1)`, enclosed at scale `2^-p`: Taylor terms floored for the
lower end, ceiled for the upper end with the tail (at most the last term) added. -/
def expSmall (zl zh p : Nat) : Nat × Nat := Id.run do
  let one := pow2 p
  let mut tl := one
  let mut th := one
  let mut sl := 0
  let mut sh := 0
  let mut n := 0
  for _ in [0:p + 1] do
    if th == 0 then break
    sl := sl + tl
    sh := sh + th
    n := n + 1
    tl := tl * zl / (one * n)
    th := cdiv (th * zh) (one * n)
  return (sl, sh + 1)

/-! ## The functions -/

private def flushIn (f : Format) (ftz : Bool) (a : Nat) : Nat := if ftz then f.flush a else a

/-- `ex2.approx` (PTX §9.7.3.21, §9.7.4.10): `2^a`. -/
def ex2 (f : Format) (ftz : Bool) (a : Nat) : Nat :=
  if f == .f32 then Mufu.ex2 ftz a else
  match f.decodeIn ftz a with
  | .nan => f.nan
  | .inf n => if n then f.zero false else f.inf false
  | .fin n m e =>
    if m = 0 then f.one else
    let x : Int := if n then -(m : Int) else m
    -- An integer: exact. Beyond ±2^16 every format here has long overflowed or underflowed.
    if e ≥ 0 then
      let x := Max.max (-2 ^ 16) (Min.min (2 ^ 16) (x * pow2 e.toNat))
      f.flushResult ftz (f.round .rn .ieee false 1 x)
    else
      let k := (-e).toNat
      let i := x / pow2 k
      let t := (x % pow2 k).toNat
      f.flushResult ftz <| zivRound f fun p =>
        let (lo, hi) := exp2Frac t k p
        (lo, hi, i - p)

/-- `lg2.approx.f32` (PTX §9.7.3.20): `log₂ a`. -/
def lg2 (ftz : Bool) (a : Nat) : Nat := Mufu.lg2 (flushIn .f32 ftz a)

/-- `rsqrt.approx` (PTX §9.7.3.16–17): `1/√a`; binary64 correctly rounded (it is algebraic). -/
def rsqrt (f : Format) (ftz : Bool) (a : Nat) : Nat :=
  if f == .f32 then Format.f32.flushResult ftz (Mufu.rsqrt (flushIn f ftz a)) else
  match f.decodeIn ftz a with
  | .nan => f.nanOf [a]
  | .inf n => if n then f.nan else f.zero false
  | .fin n m e =>
    if m = 0 then f.inf n
    else if n then f.nan
    else
      let (m, e) := if e % 2 == 0 then (m, e) else (2 * m, e - 1)
      -- ⌊2^s / √m⌋ with at least p + 2 bits.
      let s := f.man + 4 + Nat.log2 m / 2 + 1
      let q := Nat.sqrt (pow2 (2 * s) / m)
      f.flushResult ftz (f.round .rn .ieee false q (-(s : Int) - e / 2) (q * q * m != pow2 (2 * s)))

/-- `rcp.approx` (PTX §9.7.3.13–14); binary64 correctly rounded. -/
def rcp (f : Format) (ftz : Bool) (a : Nat) : Nat :=
  if f == .f32 then Format.f32.flushResult ftz (Mufu.rcp (flushIn f ftz a))
  else NV.Num.rcp f .rn ftz a

/-- `sqrt.approx.f32` (PTX §9.7.3.15). -/
def sqrt (ftz : Bool) (a : Nat) : Nat :=
  Format.f32.flushResult ftz (Mufu.sqrt (flushIn .f32 ftz a))

/-- The upper word of a binary64 as PTX's `rcp.approx.ftz.f64` and `rsqrt.approx.ftz.f64` read
and write it (1.11.20, the low 32 bits zero); its NaN is their canonical `0x7fffffff00000000`. -/
def f64Hi : Format := { exp := 11, man := 20, special := .ieee, pad := 32 }

/-- `div.full.f32` (PTX §9.7.3.8, at most 2 ulp off) as modeled here: both operands scaled by `1/4` when `|b| > 2^126`, by `2^24` when
`|b| < 2^-126` (NaN compares unordered and is not scaled), then `a' · MUFU.RCP(b')` rounded to
nearest; `.ftz` flushes in the multiplications. -/
def divFull (ftz : Bool) (a b : Nat) : Nat :=
  let f := Format.f32
  let mag := f.magnitude b
  let scale (k a b : Nat) : Nat × Nat := (mul f .rn ftz a k, mul f .rn ftz b k)
  let (a, b) :=
    if f.isNaN b then (a, b)
    else if mag > f.pack false (127 + 126) 0 then scale (f.pack false (127 - 2) 0) a b
    else if mag < f.pack false 1 0 then scale (f.pack false (127 + 24) 0) a b
    else (a, b)
  mul f .rn ftz a (Mufu.rcp (f.flush b))

/-- `div.approx.f32` (PTX §9.7.3.8): `a · MUFU.RCP(b)`, the unit flushing its subnormal input and
result, so for `2^126 < |b| < 2^128` the result is a signed zero, or NaN for infinite `a`. Without
`.ftz`, both operands are first scaled by `2^24` when `|b| < 2^-126`. -/
def divApprox (ftz : Bool) (a b : Nat) : Nat :=
  let f := Format.f32
  let (a, b) :=
    if !ftz && !(cmp f .geu false (f.magnitude b) (f.pack false 1 0)) then
      (mul f .rn false a (f.pack false (127 + 24) 0), mul f .rn false b (f.pack false (127 + 24) 0))
    else (a, b)
  mul f .rn ftz a (rcp f true b)

/-- `sin.approx.f32`/`cos.approx.f32` (PTX §9.7.3.18–19), the angle in radians or, with `turns`
(`.2pi`), in turns. -/
def sinCos (cos turns ftz : Bool) (a : Nat) : Nat :=
  let a := flushIn .f32 ftz a
  let r := if turns then Mufu.reduceTurns a else Mufu.reduce a
  Format.f32.flushResult ftz (if cos then Mufu.cosReduced r else Mufu.sinReduced r)

/-- `tanh.approx` (PTX §9.7.3.22, §9.7.4.9); subnormals are supported. -/
def tanh (f : Format) (a : Nat) : Nat :=
  match f.decode a with
  | .nan => f.nan
  | .inf n => if n then f.negate f.one else f.one
  | .fin n m e =>
    if m = 0 then f.zero n
    -- |a| ≥ 16: 1 - tanh |a| < 2^-45, which rounds to 1 in every format here.
    else if Nat.log2 m + e ≥ 4 then (if n then f.negate f.one else f.one)
    else
      let r := zivRound f fun p =>
        -- e^(2|a|) = (e^z)^(2^6), z = 2|a| / 2^6 ≤ 1/2.
        let z : Int := e + 1 - 6 + p
        let zl := if z ≥ 0 then m * pow2 z.toNat else m / pow2 (-z).toNat
        let zh := if z ≥ 0 then zl else zl + 1
        let one := pow2 p
        let (el, eh) := (List.range 6).foldl (fun (l, h) _ => (l * l / one, cdiv (h * h) one))
          (expSmall zl zh p)
        ((((el - one) * one) / (el + one) : Nat), (cdiv ((eh - one) * one) (eh + one) : Nat),
          -(p : Int))
      if n then f.negate r else r

end NV.Num.Approx
