import NV.Num.Format

/-!
# Rounding

Every result in `NV.Num` is computed exactly and rounded once, here. The exact value arrives as
`(-1)^neg · m · 2^e`, optionally with `sticky`: a nonzero amount below `m`'s last bit, for the
quotients and roots whose exact value is not a binary fraction. Callers that set `sticky` give
`m` at least two bits more than the format's precision, so the bit at the rounding position is
always known.

The modes are PTX's (ISA 9.5 §6.5.2, Table 17): `.rn` nearest-even, `.rna` nearest-away, `.rz`,
`.rm`, `.rp`; the integer modes `.rni`/`.rzi`/`.rmi`/`.rpi` are the same directions to an integer.
Stochastic rounding (`.rs`) is `Format.roundStochastic`.
-/

namespace NV.Num

inductive Rounding where
  | rn | rna | rz | rm | rp
  deriving DecidableEq, Repr, Inhabited

namespace Rounding

/-- Whether to increment the magnitude truncated to a multiple of `2^k`: `odd` is its last bit,
`rem` the `k` discarded bits, `sticky` a nonzero amount below them, `neg` the value's sign. -/
def up (r : Rounding) (neg odd : Bool) (rem k : Nat) (sticky : Bool) : Bool :=
  match r with
  | .rz => false
  | .rm => neg && (rem != 0 || sticky)
  | .rp => !neg && (rem != 0 || sticky)
  | .rn => k > 0 && (rem > pow2 (k - 1) || (rem == pow2 (k - 1) && (sticky || odd)))
  | .rna => k > 0 && rem ≥ pow2 (k - 1)

/-- Whether an overflowing result goes to infinity (IEEE 754 §7.4) rather than to the largest
finite value. -/
def away (r : Rounding) (neg : Bool) : Bool :=
  match r with
  | .rn | .rna => true
  | .rz => false
  | .rm => neg
  | .rp => !neg

/-- `m / 2^k` rounded to an integer in direction `r`, for a value of sign `neg`. -/
def shift (r : Rounding) (neg : Bool) (m k : Nat) (sticky : Bool := false) : Nat :=
  let t := m / pow2 k
  if r.up neg (t % 2 == 1) (m % pow2 k) k sticky then t + 1 else t

end Rounding

/-- What a result beyond the largest finite value becomes. -/
inductive Overflow where
  /-- IEEE 754: infinity or the largest finite value by rounding direction. A format without
  infinity gives NaN where IEEE gives infinity (E4M3, UE8M0: `cuda_fp8.hpp`'s no-saturation
  path), or its largest value if it has no NaN either. -/
  | ieee
  /-- `.satfinite`: the largest finite value, sign preserved (PTX §9.7.10.24). -/
  | sat
  /-- Infinity whatever the direction: the tensor cores' output stage. -/
  | inf
  deriving DecidableEq, Repr, Inhabited

namespace Format

/-- The overflow result, given whether the rounding direction points away from zero. -/
def overflow (f : Format) (ovf : Overflow) (away neg : Bool) : Nat :=
  match ovf, f.special with
  | .sat, _ => f.maxFinite neg
  | _, .finite => f.maxFinite neg
  | .inf, .ieee => f.inf neg
  | .inf, .nanOnly => f.pack neg f.expMax (pow2 f.man - 1)
  | .ieee, .ieee => if away then f.inf neg else f.maxFinite neg
  | .ieee, .nanOnly => if away then f.pack neg f.expMax (pow2 f.man - 1) else f.maxFinite neg

/-- The significand step: `(m + δ) · 2^e` (`m > 0`) rounded to a multiple `hi · 2^q` of its
binade's quantum (the subnormals' below the normal range), with `hi < 2^(man + 1)`. `up` decides
the direction (see `Rounding.up`). -/
def quantize (f : Format) (up : Bool → Bool → Nat → Nat → Bool → Bool) (neg : Bool) (m : Nat)
    (e : Int) (sticky : Bool) : Nat × Int :=
  let q : Int := max (e + Nat.log2 m - f.man) f.qMin
  if q ≤ e then (m * pow2 (e - q).toNat, q) else
    let k := (q - e).toNat
    let t := m / pow2 k
    let t := if up neg (t % 2 == 1) (m % pow2 k) k sticky then t + 1 else t
    if t = pow2 (f.man + 1) then (pow2 f.man, q + 1) else (t, q)

/-- Round `(-1)^neg · (m + δ) · 2^e` into `f`, where `δ ∈ (0, 1)` if `sticky` and `δ = 0`
otherwise. `up` decides the direction (see `Rounding.up`) and `away` the overflow result. Unsigned
formats take the magnitude. A format without subnormals and zero (UE8M0) gives its smallest value
for anything smaller. -/
def roundWith (f : Format) (up : Bool → Bool → Nat → Nat → Bool → Bool) (away : Bool → Bool)
    (ovf : Overflow) (neg : Bool) (m : Nat) (e : Int) (sticky : Bool) : Nat :=
  let neg := neg && f.signed
  if m = 0 then f.zero neg else
  let (hi, q) := f.quantize up neg m e sticky
  let hi := if !f.subnormals && hi < pow2 f.man then pow2 f.man else hi
  if hi = 0 then f.zero neg else
  let ef : Nat := if hi < pow2 f.man then 0 else (q - f.qMin).toNat + (if f.subnormals then 1 else 0)
  let frac := hi % pow2 f.man
  if ef > f.topFinite.1 ∨ (ef = f.topFinite.1 ∧ frac > f.topFinite.2) then
    f.overflow ovf (away neg) neg
  else f.pack neg ef frac

/-- Round an exact value in direction `r`. -/
def round (f : Format) (r : Rounding) (ovf : Overflow) (neg : Bool) (m : Nat) (e : Int)
    (sticky : Bool := false) : Nat :=
  f.roundWith r.up r.away ovf neg m e sticky

/-- Stochastic rounding (`.rs`, PTX §9.7.10.24): the magnitude moves away from zero when adding
the `n` random bits `rbits` to the top `n` discarded bits carries out. Discarded bits beyond the
top `n` take no part (inferred). Overflow goes to infinity unless saturating (inferred). -/
def roundStochastic (f : Format) (rbits n : Nat) (ovf : Overflow) (neg : Bool) (m : Nat)
    (e : Int) : Nat :=
  let up (_ _ : Bool) (rem k : Nat) (_ : Bool) : Bool :=
    let top := if k ≥ n then rem / pow2 (k - n) else rem * pow2 (n - k)
    top + rbits % pow2 n ≥ pow2 n
  f.roundWith up (fun _ => true) ovf neg m e false

/-- Encode a value: NaN becomes the canonical NaN, infinity the overflow result of a format
without it. -/
def encode (f : Format) (r : Rounding) (ovf : Overflow) : Val → Nat
  | .nan => f.nan
  | .inf neg => if f.special = .ieee && ovf != .sat then f.inf neg else f.overflow ovf true neg
  | .fin neg m e => f.round r ovf neg m e

/-- A subnormal result becomes sign-preserving zero (the `.ftz` rule for results). The flush
looks at the rounded result (inferred). -/
def flushResult (f : Format) (ftz : Bool) (b : Nat) : Nat :=
  if ftz then f.flush b else b

end Format

end NV.Num
