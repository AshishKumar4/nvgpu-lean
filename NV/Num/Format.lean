/-!
# Floating-point formats

A format is a bit layout: an optional sign bit, an exponent field, a fraction field, and a rule
for what the top of the exponent range encodes. Every format NVIDIA's instructions read or write
is an instance (PTX ISA 9.5 §5.2.1, §5.2.3): IEEE binary64, binary32 and binary16, bfloat16,
TF32, the 8-bit E4M3 and E5M2, the 6-bit E2M3 and E3M2, the 4-bit E2M1, and the unsigned scale
formats UE8M0, UE4M3 and UE5M3.

Values are decoded exactly: a finite encoding is `(-1)^neg · m · 2^e` with natural `m`, so every
operation in `NV.Num` works on integers, never on host floating point.
-/

namespace NV.Num

/-- `2 ^ n`. The runtime's `Nat.pow` builds every power through GMP; below `2^1100` (every shift
in the formats here) this looks it up. -/
def pow2Table : Array Nat := Array.ofFn (n := 1100) fun i => 2 ^ i.val

def pow2 (n : Nat) : Nat :=
  if h : n < 1100 then pow2Table[n]'(by simp [pow2Table, h]) else 2 ^ n

@[simp] theorem pow2_eq (n : Nat) : pow2 n = 2 ^ n := by
  unfold pow2; split <;> simp [pow2Table]

/-- What the all-ones exponent field encodes. -/
inductive Special where
  /-- IEEE 754: infinity (fraction zero) and NaN (fraction nonzero). Binary64/32/16, bfloat16,
  TF32, E5M2. -/
  | ieee
  /-- No infinity; only the all-ones exponent with the all-ones fraction is NaN, the rest of the
  top binade is finite. E4M3 (PTX §5.2.3: NaN only `0x7f`/`0xff`), UE4M3, UE5M3, UE8M0. -/
  | nanOnly
  /-- No infinity and no NaN: E2M1, E2M3, E3M2. -/
  | finite
  deriving DecidableEq, Repr, Inhabited

structure Format where
  /-- Exponent field width. -/
  exp : Nat
  /-- Fraction field width. -/
  man : Nat
  signed : Bool := true
  special : Special
  /-- Whether exponent field 0 encodes zero and the subnormals. UE8M0 has neither: field 0 is
  the normal value `2^-127`. -/
  subnormals : Bool := true
  /-- Zero bits below the fraction in the container: TF32 is the top 19 bits of a `.b32`. -/
  pad : Nat := 0
  deriving DecidableEq, Repr, Inhabited

namespace Format

def f64 : Format := { exp := 11, man := 52, special := .ieee }
def f32 : Format := { exp := 8, man := 23, special := .ieee }
def f16 : Format := { exp := 5, man := 10, special := .ieee }
def bf16 : Format := { exp := 8, man := 7, special := .ieee }
/-- TF32: binary32's range with a 10-bit fraction, held in the top 19 bits of a `.b32` register
(PTX §5.2.3: layout implementation defined; `cvt.tf32.f32` and the tensor cores use this one,
and wgmma ignores the low 13 bits, PTX §9.7.15.5.1). -/
def tf32 : Format := { exp := 8, man := 10, special := .ieee, pad := 13 }
def e4m3 : Format := { exp := 4, man := 3, special := .nanOnly }
def e5m2 : Format := { exp := 5, man := 2, special := .ieee }
def e2m3 : Format := { exp := 2, man := 3, special := .finite }
def e3m2 : Format := { exp := 3, man := 2, special := .finite }
def e2m1 : Format := { exp := 2, man := 1, special := .finite }
def ue8m0 : Format :=
  { exp := 8, man := 0, signed := false, special := .nanOnly, subnormals := false }
def ue4m3 : Format := { exp := 4, man := 3, signed := false, special := .nanOnly }
def ue5m3 : Format := { exp := 5, man := 3, signed := false, special := .nanOnly }

/-- Bits of the encoding, padding included. -/
def width (f : Format) : Nat := (if f.signed then 1 else 0) + f.exp + f.man + f.pad

def bias (f : Format) : Nat := pow2 (f.exp - 1) - 1

/-- The all-ones exponent field. -/
def expMax (f : Format) : Nat := pow2 f.exp - 1

/-- The exponent of the last fraction bit at exponent field 0 (and of every subnormal): the
smallest quantum of the format. -/
def qMin (f : Format) : Int := (if f.subnormals then 1 else 0) - (f.bias : Int) - f.man

/-- The largest exponent field and the largest fraction there that encode finite values. -/
def topFinite (f : Format) : Nat × Nat :=
  match f.special with
  | .ieee => (f.expMax - 1, pow2 f.man - 1)
  | .finite => (f.expMax, pow2 f.man - 1)
  | .nanOnly => if f.man = 0 then (f.expMax - 1, 0) else (f.expMax, pow2 f.man - 2)

def signBit (f : Format) : Nat := if f.signed then pow2 (f.width - 1) else 0

/-- Assemble an encoding from its fields. -/
def pack (f : Format) (neg : Bool) (e frac : Nat) : Nat :=
  (if neg && f.signed then f.signBit else 0) + (e * pow2 f.man + frac) * pow2 f.pad

/-- The exponent field, fraction field and sign of an encoding. Bits above the width and the
padding are ignored. -/
def fields (f : Format) (b : Nat) : Bool × Nat × Nat :=
  let x := b / pow2 f.pad
  (f.signed && (b / pow2 (f.width - 1)) % 2 == 1, (x / pow2 f.man) % pow2 f.exp, x % pow2 f.man)

/-- Sign-preserving zero (UE8M0 has no zero; its smallest value stands in). -/
def zero (f : Format) (neg : Bool) : Nat := f.pack neg 0 0

def inf (f : Format) (neg : Bool) : Nat := f.pack neg f.expMax 0

def maxFinite (f : Format) (neg : Bool) : Nat := f.pack neg f.topFinite.1 f.topFinite.2

/-- NVIDIA's canonical NaN: positive, all exponent and fraction bits set. Binary32 `0x7fffffff`
(`CUDART_NAN_F`), binary16 and bfloat16 `0x7fff` (`CUDART_NAN_FP16`, `CUDART_NAN_BF16`), E4M3 and
E5M2 `0x7f` (`cuda_fp8.hpp`). Formats without NaN have none; the all-ones pattern stands in. -/
def nan (f : Format) : Nat := f.pack false f.expMax (pow2 f.man - 1)

end Format

/-- A decoded value: `(-1)^neg · m · 2^e` when finite (zero when `m = 0`). -/
inductive Val where
  | nan
  | inf (neg : Bool)
  | fin (neg : Bool) (m : Nat) (e : Int)
  deriving DecidableEq, Repr, Inhabited

namespace Format

def decode (f : Format) (b : Nat) : Val :=
  let (neg, e, frac) := f.fields b
  if e = f.expMax ∧ f.special = .ieee then
    if frac = 0 then .inf neg else .nan
  else if e = f.expMax ∧ f.special = .nanOnly ∧ frac = pow2 f.man - 1 then .nan
  else if e = 0 ∧ f.subnormals then .fin neg frac f.qMin
  else .fin neg (pow2 f.man + frac) ((e : Int) - f.bias - f.man)

def isNaN (f : Format) (b : Nat) : Bool := f.decode b == .nan

def isInf (f : Format) (b : Nat) : Bool :=
  match f.decode b with
  | .inf _ => true
  | _ => false

/-- Zero or subnormal: exponent field 0 in a format that has subnormals. -/
def isTiny (f : Format) (b : Nat) : Bool := f.subnormals && (f.fields b).2.1 == 0

def isSubnormal (f : Format) (b : Nat) : Bool := f.isTiny b && (f.fields b).2.2 != 0

def isZero (f : Format) (b : Nat) : Bool := f.isTiny b && (f.fields b).2.2 == 0

def isNeg (f : Format) (b : Nat) : Bool := (f.fields b).1

/-- The encoding with the sign bit cleared, padding and higher bits dropped. -/
def magnitude (f : Format) (b : Nat) : Nat :=
  let (_, e, frac) := f.fields b
  f.pack false e frac

/-- Flush a subnormal input to sign-preserving zero (the `.ftz` rule for inputs). -/
def flush (f : Format) (b : Nat) : Nat :=
  if f.isSubnormal b then f.zero (f.isNeg b) else b

end Format

end NV.Num
