import NV.Num.Mufu
import NV.Num.F32

/-!
# The special-function unit on machine words

`MUFU.EX2`, `MUFU.RCP` and `MUFU.RSQ` as the SASS instructions compute them (the unit's `.ftz`
behaviour: subnormal inputs and results flushed), on `UInt32` encodings with `UInt64`
interpolation: `NV.Num.Mufu`'s model step for step, its binary32 steps through `NV.Num.F32`. The
SASS fast path uses these (`NV.SASS.Fast`); `NV.Num.Mufu` stays the specification, and
`lake exe num-tests` checks the two agree on edge values and on a spread of every exponent.
-/

namespace NV.Num.MufuWord

open NV.Num.Mufu

/-- `squareApprox` below `2^32`. -/
@[inline] def square (w : UInt64) : UInt64 := Id.run do
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
  return if err ≤ sq then sq - err else 0

/-- `Poly.eval`, modulo `2^64`. -/
@[inline] def eval (p : Poly) (index : Nat) (xl sq : UInt64) : UInt64 :=
  let (c0, c1, c2) := p.table[index]!
  let sh (v : UInt64) (k : Nat) : UInt64 := if k < 64 then v <<< k.toUInt64 else 0
  let t1 := sh (c1.toUInt64 * xl) p.a1
  let t2 := c2.toUInt64 * sq
  sh c0.toUInt64 p.a0 + (if p.neg1 then 0 - t1 else t1) + (if p.neg2 then 0 - t2 else t2) +
    (p.bias / pow2 (64 - p.weight + 23)).toUInt64

/-- The 23 fraction bits below the sum's units bit. -/
@[inline] def frac (p : Poly) (sum : UInt64) : UInt32 :=
  ((sum >>> (p.weight - 23).toUInt64) &&& 0x7fffff).toUInt32

/-- `x · 2^k` for a normal `x`, rounded to nearest; a subnormal result flushed. -/
@[inline] def ldexp (x : UInt32) (k : Int) : UInt32 :=
  let ef := ((x >>> 23) &&& 0xff).toNat
  F32.round .rn true (x >>> 31 == 1) ((x &&& 0x7fffff) ||| 0x800000).toNat ((ef : Int) - 150 + k)
    false

@[inline] private def flushIn (x : UInt32) : UInt32 :=
  if (x >>> 23) &&& 0xff == 0 then x &&& 0x80000000 else x

@[inline] private def isNaN (x : UInt32) : Bool := x &&& 0x7fffffff > 0x7f800000

/-- `MUFU.EX2`: `Mufu.ex2 true`. -/
def ex2 (x : UInt32) : UInt32 :=
  let neg := x >>> 31 == 1
  let ef := (x >>> 23) &&& 0xff
  if isNaN x then F32.nan
  else if ef == 0xff then (if neg then 0 else F32.inf false)
  else if ef == 0 then 0x3f800000
  -- |x| ≥ 2^23: an integer whose power of two overflows or underflows.
  else if ef ≥ 150 then (if neg then 0 else F32.inf false)
  else
    -- x = m · 2^-k: integral part `ip`, fraction `fp`, and the fraction with 23 bits, truncated.
    let m := ((x &&& 0x7fffff) ||| 0x800000).toUInt64
    let k := (150 - ef).toUInt64
    let (ip, fp) := if k < 24 then (m >>> k, m &&& (((1 : UInt64) <<< k) - 1)) else (0, m)
    let xb := if k ≤ 23 then fp <<< (23 - k) else if k < 47 then fp >>> (k - 23) else 0
    let (xb, i) : UInt64 × Int :=
      if neg && xb != 0 then (0xffffffff - xb, -(ip.toNat : Int) - 1)
      else (xb, if neg then -(ip.toNat : Int) else ip.toNat)
    let xb := xb &&& 0x7fffff
    let (xh, xl) := ((xb >>> 17).toNat, xb &&& 0x1ffff)
    let sum := eval ex2Poly xh xl (square xl)
    let scale : UInt32 :=
      if i > 127 then F32.inf false else if i < -126 then 0 else ((i + 127).toNat.toUInt32 <<< 23)
    F32.mul .rn true scale (0x3f800000 ||| frac ex2Poly sum)

/-- `MUFU.RCP`: `Approx.rcp .f32 true`. -/
def rcp (x : UInt32) : UInt32 :=
  let x := flushIn x
  let neg := x >>> 31 == 1
  let ef := (x >>> 23) &&& 0xff
  if isNaN x then F32.nan
  else if ef == 0xff then F32.zero neg
  else if ef == 0 then F32.inf neg
  else
    let xh := ((x >>> 16) &&& 0x7f).toNat
    let xl := (x &&& 0xffff).toUInt64
    let sum := eval rcpPoly xh xl (square (2 * xl) / 4)
    let (re, sum) : UInt32 × UInt64 := if sum >>> 56 == 0 then (126, 2 * sum) else (127, sum)
    let s : UInt32 := if neg then 0x80000000 else 0
    ldexp (s ||| (re <<< 23) ||| frac rcpPoly sum) (127 - (ef.toNat : Int))

/-- `MUFU.RSQ`: `Approx.rsqrt .f32 true`. -/
def rsqrt (x : UInt32) : UInt32 :=
  let x := flushIn x
  let neg := x >>> 31 == 1
  let ef := (x >>> 23) &&& 0xff
  if isNaN x then F32.nan
  else if ef == 0xff then (if neg then F32.nan else 0)
  else if ef == 0 then F32.inf neg
  else if neg then F32.nan
  else
    let l : Int := (ef.toNat : Int) - 127
    let fr := x &&& 0x7fffff
    if l % 2 == 0 && fr == 0 then ldexp 0x3f800000 (-(l / 2))
    else
      let xh := (fr >>> 17).toNat
      let xl := (fr &&& 0x1ffff).toUInt64
      let (index, l) := if l % 2 == 0 then (xh, l) else (xh + 64, l - 1)
      let sum := 2 * eval rsqrtPoly index xl (square xl)
      ldexp (((126 : UInt32) <<< 23) ||| frac rsqrtPoly sum) (-(l / 2))

end NV.Num.MufuWord
