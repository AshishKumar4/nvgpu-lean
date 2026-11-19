import NV.Num.Arith

/-!
# Conversions

The conversions of PTX's `cvt` (ISA 9.5 §9.7.10.24, Tables 15–16), one element at a time (the
PTX semantics packs and unpacks `x2`/`x4` registers): float to float, with and without UE8M0
scale factors, float to integer, integer to float, float to an integral float, to and from the
fixed-point `s2f6`, plus decimal literals. Each is the exact source value rounded once into the
destination.

NaN rules: a NaN becomes the destination's canonical NaN (`cuda_fp8.hpp`, `cuda_fp16.hpp`,
`cuda_bf16.hpp` canonicalize; PTX leaves `cvt.f32.bf16`'s NaN unspecified), so for the formats
without NaN it becomes `+MAX_NORM` under `.satfinite` (PTX's rule for E2M1/E2M3/E3M2). Float to
integer gives 0 for NaN, or `1 << (width - 1)` from binary64 or into 64 bits (PTX's Integer Notes).
-/

namespace NV.Num

/-- The modifiers of a float-to-float `cvt`. `ftz` flushes binary32 subnormal inputs and results
only (PTX: ".ftz applies only to single precision inputs and results"). -/
structure CvtOpts where
  ftz : Bool := false
  /-- Clamp to `[0, 1]`, NaN to `+0`. -/
  sat : Bool := false
  /-- Negative to `+0`, NaN to the canonical NaN. -/
  relu : Bool := false
  /-- Beyond the largest finite value (infinity included) to it, sign preserved. -/
  satfinite : Bool := false
  /-- `-0` to `+0` (`.pzo`). -/
  pzo : Bool := false
  deriving DecidableEq, Repr, Inhabited

/-- The clamps that follow the rounding, in PTX's order. -/
def CvtOpts.finish (o : CvtOpts) (dst : Format) (d : Nat) : Nat :=
  let d := dst.flushResult (o.ftz && dst == .f32) d
  let d := if o.relu then NV.Num.relu dst d else d
  let d := if o.sat then satUnit dst d else d
  if o.pzo && dst.isZero d then dst.zero false else d

/-- Round a decoded value into `dst`. -/
def cvtVal (dst : Format) (encodeRound : Bool → Nat → Int → Nat) (o : CvtOpts) : Val → Nat
  | .nan => o.finish dst dst.nan
  | .inf n => o.finish dst (dst.encode .rn (if o.satfinite then .sat else .ieee) (.inf n))
  | .fin n m e => o.finish dst (encodeRound n m e)

/-- Float to float. -/
def cvt (src dst : Format) (r : Rounding) (o : CvtOpts := {}) (a : Nat) : Nat :=
  let ovf : Overflow := if o.satfinite then .sat else .ieee
  cvtVal dst (fun n m e => dst.round r ovf n m e) o (src.decodeIn (o.ftz && src == .f32) a)

/-- A decoded value times the UE8M0 scale factor `sf` (`2^(sf - 127)`; `0xff` is NaN), or divided
by it when `down`. Exact: hardware applying the scale to the exponent is inferred (PTX writes
`convert(a / scale-factor)`). -/
def Val.scaleBy (v : Val) (sf : Nat) (down : Bool) : Val :=
  if sf % 256 == 0xff then .nan else
  let k : Int := (sf % 256 : Nat) - 127
  match v with
  | .fin n m e => .fin n m (if down then e - k else e + k)
  | v => v

/-- `cvt` with `.scaled::n1::ue8m0` or `.scaled::n2::ue8m0` (PTX §9.7.10.24): a conversion into a
narrower format divides the source by the scale factor `sf` first, flushing subnormal sources
under `.scaled::n1::ue8m0` (`n1`); a conversion out of one (into bf16) multiplies by it. -/
def cvtScaled (src dst : Format) (r : Rounding) (o : CvtOpts) (sf : Nat) (down n1 : Bool)
    (a : Nat) : Nat :=
  let ovf : Overflow := if o.satfinite then .sat else .ieee
  cvtVal dst (fun n m e => dst.round r ovf n m e) o
    ((src.decodeIn (down && n1) a).scaleBy sf down)

/-- Float to float with stochastic rounding (`.rs`): `rbits` holds the `n` random bits for this
element (PTX Figures 39–42 give each destination's layout; the PTX semantics extracts them). -/
def cvtStochastic (src dst : Format) (rbits n : Nat) (o : CvtOpts := {}) (a : Nat) : Nat :=
  let ovf : Overflow := if o.satfinite then .sat else .ieee
  cvtVal dst (fun neg m e => dst.roundStochastic rbits n ovf neg m e) o
    (src.decodeIn (o.ftz && src == .f32) a)

/-- Round a float to an integral value in its own format (`cvt.rni.f32.f32` and the like). -/
def roundInt (f : Format) (r : Rounding) (ftz : Bool) (a : Nat) : Nat :=
  match f.decodeIn ftz a with
  | .nan => f.nanOf [a]
  | .inf n => f.inf n
  | .fin n m e =>
    if e ≥ 0 then f.flushResult ftz a
    else f.round .rz .ieee n (r.shift n m (-e).toNat) 0

/-- The integer an `n`-bit two's complement or unsigned encoding denotes. -/
def intOfBits (signed : Bool) (width bits : Nat) : Int :=
  let b := bits % pow2 width
  if signed && b ≥ pow2 (width - 1) then (b : Int) - pow2 width else b

/-- Float to integer (`cvt.{rni,rzi,rmi,rpi}.{s,u}N.fN`): rounded, then clamped to the
destination's range (PTX: "the result is clamped to the destination range by default"); the
result is the destination's `width`-bit encoding. -/
def toInt (src : Format) (r : Rounding) (signed : Bool) (width : Nat) (ftz : Bool) (a : Nat) :
    Nat :=
  let lo : Int := if signed then -pow2 (width - 1) else 0
  let hi : Int := if signed then pow2 (width - 1) - 1 else pow2 width - 1
  let clamp (i : Int) : Nat := (Max.max lo (Min.min hi i) % pow2 width).toNat
  match src.decodeIn (ftz && src == .f32) a with
  | .nan => if src == .f64 || width == 64 then pow2 (width - 1) else 0
  | .inf n => clamp (if n then lo else hi)
  | .fin n m e =>
    let i : Nat := if e ≥ 0 then m * pow2 e.toNat else r.shift n m (-e).toNat
    clamp (if n then -(i : Int) else i)

/-- Integer to float (`cvt.{rn,rz,rm,rp}.fN.{s,u}N`). -/
def ofInt (dst : Format) (r : Rounding) (signed : Bool) (width bits : Nat) : Nat :=
  let i := intOfBits signed width bits
  dst.round r .ieee (i < 0) i.natAbs 0

/-! ## The fixed-point `s2f6`

An 8-bit two's complement integer times `2^-6` (PTX §5.2.4): `[-2, 127/64]`, no infinity or NaN.
Conversions into it are `.satfinite` and round with `.rn` or `.rz` (`.frnd2`). -/

/-- Into `s2f6` (`cvt.frnd2.satfinite{.relu}{.scaled::n2::ue8m0}.s2f6x2`), one element: the source
divided by the scale factor `sf`; NaN to `+MAX_NORM`; a magnitude above `MAX_NORM` (127/64) to the
sign-preserved `MAX_NORM` (PTX's `.satfinite`, so `-2` is never produced); `.relu` takes negative
results to 0. -/
def toS2f6 (src : Format) (r : Rounding) (relu : Bool) (sf : Nat) (a : Nat) : Nat :=
  let i : Int := match (src.decode a).scaleBy sf true with
    | .nan => 127
    | .inf n => if n then -127 else 127
    | .fin n m e =>
      -- |x| · 2^6 rounded; above 127 saturates
      let q : Nat := if e + 6 ≥ 0 then m * pow2 (e + 6).toNat else r.shift n m (-(e + 6)).toNat
      let q : Int := Min.min q 127
      if n then -q else q
  let i := if relu && i < 0 then 0 else i
  (i % 256).toNat

/-- Out of `s2f6` into `dst` (`cvt.rn{.satfinite}{.relu}{.scaled::n2::ue8m0}.bf16x2.s2f6x2`), one
element: its value times the scale factor `sf`, rounded once. -/
def ofS2f6 (dst : Format) (o : CvtOpts) (sf : Nat) (a : Nat) : Nat :=
  let i := intOfBits true 8 a
  let ovf : Overflow := if o.satfinite then .sat else .ieee
  cvtVal dst (fun n m e => dst.round .rn ovf n m e) o
    ((Val.fin (i < 0) i.natAbs (-6)).scaleBy sf false)

/-- The decimal `m · 10^e` rounded into `f`: PTX's decimal floating-point literals (binary64,
round to nearest even, PTX §4.5.2). -/
def ofDecimal (f : Format) (m : Nat) (e : Int) (r : Rounding := .rn) : Nat :=
  if m = 0 then f.zero false
  -- Past 10^400 every format here overflows; below 2^-1100 (`10^e ≤ 2^(3e)`) each rounds alike.
  else if e ≥ 0 then f.round r .ieee false (m * 10 ^ (Min.min e 400).toNat) 0
  else
    let e := Max.max e (-((1101 + Nat.log2 m) / 3 + 1 : Nat))
    -- m / 10^k = m / 5^k · 2^-k
    let (q, e2, sticky) := divExact (f.man + 1) m 0 (5 ^ (-e).toNat) 0
    f.round r .ieee false q (e2 + e) sticky

end NV.Num
