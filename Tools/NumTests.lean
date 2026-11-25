import NV.Num
import NV.PTX.Sem.Matrix

/-!
`lake exe num-tests`: behavioural tests of `NV.Num` against an independent exact reference.

The reference (`Ref`) reads values from the bits by the formats' definitions, as rationals, and
rounds an exact rational by searching the format's encodings, which are ordered like their values,
for its neighbours: no shifting, no sticky bits, nothing shared with `Format.round`. Expected
special values (canonical NaNs, maxima) are constants from NVIDIA's documents.

* `num-tests`: the suite. Every encoding of every format of 16 bits or fewer is decoded, re-encoded
  and converted to every other format in every rounding mode; `f32` inputs at every rounding
  boundary of each narrow format; arithmetic in every mode on random and special operands; integer
  conversions; decimal literals (Python's correctly rounded `float`); the `.approx` functions
  against mpmath (`tools/num/approx-vectors.txt`); the tensor-core model on the cases its authors
  published and on every single FP8 product.
* `num-tests tensor DIR`: the H100/H200 tensor-core outputs Khattak and Mikaitis measured
  (`model_validation/` of their repository) against `NV.Num.Hopper`.
* `num-tests exhaustive FORMAT SHARD SHARDS`: hashes of add, sub, mul and div over one shard of
  every binary16 or bfloat16 operand pair in each mode, for `tools/num/exhaustive.sh` to compare
  with a host reference.

Exits non-zero on any mismatch.
-/

open NV.Num

/-! ## The reference -/

namespace Ref

/-- What an encoding denotes. `fin neg x` is `±x` with `x ≥ 0`. -/
inductive Kind where
  | nan
  | inf (neg : Bool)
  | fin (neg : Bool) (x : Rat)
  deriving Inhabited

def pow2 (e : Int) : Rat :=
  if e ≥ 0 then ((2 ^ e.toNat : Nat) : Rat) else 1 / ((2 ^ (-e).toNat : Nat) : Rat)

/-- IEEE 754 §3.4's reading of the fields, with each format's special-value rule. -/
def kind (f : Format) (b : Nat) : Kind :=
  let neg := f.signed && b.testBit (f.width - 1)
  let ef := (b >>> (f.man + f.pad)) % 2 ^ f.exp
  let fr := (b >>> f.pad) % 2 ^ f.man
  let bias : Int := 2 ^ (f.exp - 1) - 1
  let x : Rat :=
    if ef = 0 ∧ f.subnormals then (fr : Rat) * pow2 (1 - bias - f.man)
    else ((2 ^ f.man + fr : Nat) : Rat) * pow2 ((ef : Int) - bias - f.man)
  let top := ef = 2 ^ f.exp - 1
  match f.special with
  | .ieee => if top then (if fr = 0 then .inf neg else .nan) else .fin neg x
  | .nanOnly => if top ∧ fr = 2 ^ f.man - 1 then .nan else .fin neg x
  | .finite => .fin neg x

def isNaN (f : Format) (b : Nat) : Bool := match kind f b with | .nan => true | _ => false

/-- The non-negative finite encodings in order: index `i` is the encoding `i · 2^pad`. -/
def count (f : Format) : Nat :=
  match f.special with
  | .ieee => (2 ^ f.exp - 1) * 2 ^ f.man
  | .nanOnly => 2 ^ (f.exp + f.man) - 1
  | .finite => 2 ^ (f.exp + f.man)

def valueAt (f : Format) (i : Nat) : Rat := match kind f (i * 2 ^ f.pad) with | .fin _ x => x | _ => 0

/-- The values of the formats of 16 bits or fewer, computed once. -/
def tables : List (Format × Array Rat) :=
  [Format.f16, .bf16, .e4m3, .e5m2, .e2m3, .e3m2, .e2m1, .ue8m0, .ue4m3, .ue5m3].map fun f =>
    (f, (Array.range (count f)).map (valueAt f))

def at_ (f : Format) (i : Nat) : Rat :=
  match tables.find? (·.1 == f) with
  | some (_, t) => t[i]!
  | none => valueAt f i

def enc (f : Format) (neg : Bool) (i : Nat) : Nat :=
  (if neg && f.signed then 2 ^ (f.width - 1) else 0) + i * 2 ^ f.pad

/-- NVIDIA's canonical NaN, or what a format without NaN gives instead. -/
def nan (f : Format) : Nat := enc f false (2 ^ (f.exp + f.man) - 1)

def maxIdx (f : Format) : Nat := count f - 1

/-- What a result past the largest finite value becomes, rounding away from zero. -/
def over (f : Format) (ovf : Overflow) (neg : Bool) : Nat :=
  match ovf, f.special with
  | .sat, _ | _, .finite => enc f neg (maxIdx f)
  | _, .ieee => enc f neg (count f)
  | _, .nanOnly => enc f neg (2 ^ (f.exp + f.man) - 1)

/-- The largest index whose value is at most `a` (`a` at least the smallest value). -/
def floorIdx (f : Format) (sq : Bool) (a : Rat) : Nat := Id.run do
  let v (i : Nat) := if sq then at_ f i * at_ f i else at_ f i
  let mut lo := 0
  let mut hi := maxIdx f
  for _ in [0:70] do
    if lo ≥ hi then break
    let mid := (lo + hi + 1) / 2
    if v mid ≤ a then lo := mid else hi := mid - 1
  return lo

/-- `±a` (or `±√a` when `sq`) rounded into `f` by the definition of each mode. -/
def round (f : Format) (r : Rounding) (ovf : Overflow) (neg : Bool) (a : Rat) (sq := false) :
    Nat :=
  let neg := neg && f.signed
  let v (i : Nat) := if sq then at_ f i * at_ f i else at_ f i
  if a < v 0 then enc f neg 0 else
  let i := floorIdx f sq a
  if v i == a then enc f neg i else
  let top := i == maxIdx f
  let lo := at_ f i
  let hi := if !top then at_ f (i + 1) else if f.man = 0 then 2 * lo else 2 * lo - at_ f (i - 1)
  -- distances, compared through squares for a root
  let below := if sq then a - ((lo + hi) / 2) * ((lo + hi) / 2) else a - (lo + hi) / 2
  let up := match r with
    | .rz => false
    | .rp => !neg
    | .rm => neg
    | .rn => decide (below > 0) || (below == 0 && i % 2 == 1)
    | .rna => decide (below ≥ 0)
  if !up then enc f neg i else if top then over f ovf neg else enc f neg (i + 1)

def zero (f : Format) (neg : Bool) : Nat := enc f neg 0

def inf (f : Format) (neg : Bool) : Nat := enc f neg (count f)

/-- A finite result of an arithmetic operation, with IEEE 754 §6.3's sign for an exact zero. -/
def result (f : Format) (r : Rounding) (x : Rat) (zeroNeg : Bool) : Nat :=
  if x == 0 then zero f zeroNeg else round f r .ieee (decide (x < 0)) (if x < 0 then -x else x)

def val (neg : Bool) (x : Rat) : Rat := if neg then -x else x

def add (f : Format) (r : Rounding) (a b : Nat) : Nat :=
  match kind f a, kind f b with
  | .nan, _ | _, .nan => nan f
  | .inf s, .inf t => if s == t then inf f s else nan f
  | .inf s, _ | _, .inf s => inf f s
  | .fin s x, .fin t y =>
    result f r (val s x + val t y) (if x == 0 && y == 0 && s == t then s else r == .rm)

def mul (f : Format) (r : Rounding) (a b : Nat) : Nat :=
  match kind f a, kind f b with
  | .nan, _ | _, .nan => nan f
  | .inf s, .inf t => inf f (s != t)
  | .inf s, .fin t x | .fin t x, .inf s => if x == 0 then nan f else inf f (s != t)
  | .fin s x, .fin t y => result f r (val (s != t) (x * y)) (s != t)

def div (f : Format) (r : Rounding) (a b : Nat) : Nat :=
  match kind f a, kind f b with
  | .nan, _ | _, .nan => nan f
  | .inf _, .inf _ => nan f
  | .inf s, .fin t _ => inf f (s != t)
  | .fin s _, .inf t => zero f (s != t)
  | .fin s x, .fin t y =>
    if y == 0 then (if x == 0 then nan f else inf f (s != t))
    else result f r (val (s != t) (x / y)) (s != t)

def fma (f : Format) (r : Rounding) (a b c : Nat) : Nat :=
  match kind f a, kind f b, kind f c with
  | .nan, _, _ | _, .nan, _ | _, _, .nan => nan f
  | ka, kb, kc =>
    let prodInf : Option Bool := match ka, kb with
      | .inf s, .inf t => some (s != t)
      | .inf s, .fin t x | .fin t x, .inf s => if x == 0 then none else some (s != t)
      | _, _ => none
    let invalid := match ka, kb with
      | .inf _, .fin _ x | .fin _ x, .inf _ => x == 0
      | _, _ => false
    if invalid then nan f else
    match prodInf, kc with
    | some s, .inf t => if s == t then inf f s else nan f
    | some s, _ => inf f s
    | none, .inf t => inf f t
    | none, .fin u z =>
      match ka, kb with
      | .fin s x, .fin t y =>
        let p := val (s != t) (x * y)
        result f r (p + val u z) (if p == 0 && z == 0 && (s != t) == u then u else r == .rm)
      | _, _ => nan f
    | none, .nan => nan f

def sqrt (f : Format) (r : Rounding) (a : Nat) : Nat :=
  match kind f a with
  | .nan => nan f
  | .inf s => if s then nan f else inf f false
  | .fin s x => if x == 0 then zero f s else if s then nan f else round f r .ieee false x true

end Ref

/-! ## Harness -/

structure Tally where
  name : String
  run : Nat := 0
  bad : Nat := 0
  shown : Array String := #[]

def hex (n : Nat) : String := "0x" ++ String.ofList (Nat.toDigits 16 n)

def parseHex (s : String) : Nat :=
  s.foldl (fun n c => 16 * n + (if c.isDigit then c.toNat - 48 else (c.toLower.toNat - 87))) 0

/-- The whitespace-separated words of a line. -/
def words (l : String) : List String :=
  let (ws, w) := l.foldl (init := (([] : List String), "")) fun (ws, w) c =>
    if c.isWhitespace then (if w.isEmpty then (ws, w) else (w :: ws, "")) else (ws, w.push c)
  (if w.isEmpty then ws else w :: ws).reverse


def Tally.check (t : Tally) (ok : Bool) (what : Unit → String) : Tally :=
  if ok then { t with run := t.run + 1 }
  else { t with run := t.run + 1, bad := t.bad + 1,
                shown := if t.shown.size < 8 then t.shown.push (what ()) else t.shown }

def Tally.report (t : Tally) : IO Bool := do
  IO.println s!"{if t.bad == 0 then "ok  " else "FAIL"} {t.name}: {t.run} cases, {t.bad} wrong"
  for s in t.shown do IO.println s!"       {s}"
  return t.bad == 0

/-- A deterministic stream of pseudo-random numbers (SplitMix64). -/
structure Rng where
  s : UInt64

def Rng.next (g : Rng) : UInt64 × Rng :=
  let s := g.s + 0x9e3779b97f4a7c15
  let z := (s ^^^ (s >>> 30)) * 0xbf58476d1ce4e5b9
  let z := (z ^^^ (z >>> 27)) * 0x94d049bb133111eb
  (z ^^^ (z >>> 31), ⟨s⟩)

/-- `n` random bits. -/
def Rng.bits (g : Rng) (n : Nat) : Nat × Rng := Id.run do
  let mut g := g
  let mut x := 0
  for i in [0:(n + 63) / 64] do
    let (v, g') := g.next
    g := g'
    x := x + v.toNat * 2 ^ (64 * i)
  return (x % 2 ^ n, g)

def modes : List Rounding := [.rn, .rna, .rz, .rm, .rp]
def ieeeModes : List Rounding := [.rn, .rz, .rm, .rp]

/-- The modes tested into `f`: PTX converts into UE8M0 only with `.rz` and `.rp`, and nearest-even
is not defined for a format without fraction bits. -/
def modesInto (f : Format) : List Rounding := if f.man == 0 then [.rz, .rm, .rp] else modes

def small : List (String × Format) :=
  [("f16", .f16), ("bf16", .bf16), ("e4m3", .e4m3), ("e5m2", .e5m2), ("e2m3", .e2m3),
   ("e3m2", .e3m2), ("e2m1", .e2m1), ("ue8m0", .ue8m0), ("ue4m3", .ue4m3), ("ue5m3", .ue5m3)]

def valOf : Val → Option Rat
  | .fin n m e => some (Ref.val n (m * Ref.pow2 e))
  | _ => none

/-- A random operand of `f`: a quarter of the time a special or boundary value, otherwise random
bits, with the exponent often close to `near`'s so that sums cancel. -/
def operand (f : Format) (g : Rng) (near : Nat) : Nat × Rng :=
  let (k, g) := g.bits 3
  let (b, g) := g.bits f.width
  let specials := #[0, f.signBit, f.inf false, f.inf true, f.nan, f.maxFinite false,
    f.maxFinite true, 1 * 2 ^ f.pad, f.one, f.pack false 1 0, f.pack false 0 (2 ^ f.man - 1)]
  if k == 0 then (specials[b % specials.size]!, g)
  else if k ≤ 3 then
    let (_, e, _) := f.fields near
    let (d, g) := g.bits 3
    let e' := if e + d ≥ 4 then e + d - 4 else e
    (f.pack (b % 2 == 1) (Min.min e' (2 ^ f.exp - 1)) ((b / 2) % 2 ^ f.man), g)
  else (b * 2 ^ f.pad % 2 ^ f.width, g)

/-! ## The suite -/

def decodeAll (_scale : Nat) : Tally := Id.run do
  let mut t : Tally := { name := "decode: every encoding of every format of 16 bits or fewer" }
  for (n, f) in small ++ [("tf32", .tf32)] do
    for i in [0:2 ^ (f.width - f.pad)] do
      let b := i * 2 ^ f.pad
      let ok := match Ref.kind f b, f.decode b with
        | .nan, .nan => true
        | .inf s, .inf s' => s == s'
        | .fin s x, v@(.fin s' _ _) => s == s' && valOf v == some (Ref.val s x)
        | _, _ => false
      t := t.check ok fun _ => s!"{n} {hex b}"
  return t

/-- Values the formats' definitions fix (OCP MX v1.0 Tables 1–2, PTX §5.2.3, IEEE 754). -/
def known (_scale : Nat) : Tally := Id.run do
  let mut t : Tally := { name := "decode: published values" }
  let q (n d : Nat) : Rat := (n : Rat) / (d : Rat)
  let cases : List (String × Format × Nat × Rat) := [
    ("e2m1", .e2m1, 0x1, q 1 2), ("e2m1", .e2m1, 0x2, 1), ("e2m1", .e2m1, 0x3, q 3 2),
    ("e2m1", .e2m1, 0x4, 2), ("e2m1", .e2m1, 0x5, 3), ("e2m1", .e2m1, 0x6, 4),
    ("e2m1", .e2m1, 0x7, 6), ("e2m1", .e2m1, 0xf, -6),
    ("e2m3", .e2m3, 0x1f, q 15 2), ("e2m3", .e2m3, 0x01, q 1 8),
    ("e3m2", .e3m2, 0x1f, 28), ("e3m2", .e3m2, 0x01, q 1 16),
    ("e4m3", .e4m3, 0x7e, 448), ("e4m3", .e4m3, 0x08, q 1 64), ("e4m3", .e4m3, 0x01, q 1 512),
    ("e5m2", .e5m2, 0x7b, 57344), ("e5m2", .e5m2, 0x04, q 1 16384),
    ("e5m2", .e5m2, 0x01, q 1 65536),
    ("ue8m0", .ue8m0, 0x7f, 1), ("ue8m0", .ue8m0, 0x80, 2), ("ue8m0", .ue8m0, 0x00, Ref.pow2 (-127)),
    ("ue8m0", .ue8m0, 0xfe, Ref.pow2 127), ("ue4m3", .ue4m3, 0x7e, 448),
    ("f16", .f16, 0x7bff, 65504), ("f16", .f16, 0x0001, Ref.pow2 (-24)),
    ("bf16", .bf16, 0x7f7f, 255 * Ref.pow2 120), ("tf32", .tf32, 0x7f7fe000, 2047 * Ref.pow2 117)]
  for (n, f, b, x) in cases do
    t := t.check (valOf (f.decode b) == some x) fun _ => s!"{n} {hex b}"
  for (n, f, b) in [("f32", Format.f32, 0x7fffffff), ("f16", .f16, 0x7fff), ("bf16", .bf16, 0x7fff),
                    ("e4m3", .e4m3, 0x7f), ("e5m2", .e5m2, 0x7f), ("ue8m0", .ue8m0, 0xff)] do
    t := t.check (f.nan == b) fun _ => s!"canonical NaN {n}: {hex f.nan}"
  return t

/-- Every finite encoding rounds to itself in every mode; widening and narrowing back is the
identity, NaN becoming the canonical NaN (binary32 through binary64 the same NaN, quieted). -/
def roundTrips (_scale : Nat) : Tally := Id.run do
  let mut t : Tally := { name := "round trips: encode ∘ decode, and F → G → F for F ⊂ G" }
  for (n, f) in small do
    for b in [0:2 ^ f.width] do
      if let .fin s m e := f.decode b then
        for r in modes do
          t := t.check (f.round r .ieee s m e == b) fun _ => s!"{n} {hex b} {repr r}"
  let pairs : List (String × Format × Format) := [
    ("f16→f32", .f16, .f32), ("f16→f64", .f16, .f64), ("bf16→f32", .bf16, .f32),
    ("e4m3→f16", .e4m3, .f16), ("e4m3→bf16", .e4m3, .bf16), ("e5m2→f16", .e5m2, .f16),
    ("e5m2→bf16", .e5m2, .bf16), ("e2m3→f16", .e2m3, .f16), ("e3m2→f16", .e3m2, .f16),
    ("e2m1→f16", .e2m1, .f16), ("e2m1→e4m3", .e2m1, .e4m3), ("ue8m0→bf16", .ue8m0, .bf16),
    ("ue4m3→f16", .ue4m3, .f16), ("ue5m3→bf16", .ue5m3, .bf16), ("f32→f64", .f32, .f64)]
  for (n, f, g) in pairs do
    let total := if f.width ≤ 16 then 2 ^ f.width else 2 ^ 16
    for i in [0:total] do
      -- binary32 sampled at every 2^16-th encoding plus its low bits
      let b := if f.width ≤ 16 then i else i * 65521 % 2 ^ 32
      let back := cvt g f .rn {} (cvt f g .rn {} b)
      let want := if !f.isNaN b then b else if f == .f32 && g == .f64 then b ||| 0x400000 else f.nan
      t := t.check (back == want) fun _ => s!"{n} {hex b} → {hex back}"
  return t

/-- Conversions from every `f16` and `bf16` encoding into every format, in every mode, with and
without `.satfinite`. -/
def narrowing (_scale : Nat) : Tally := Id.run do
  let mut t : Tally := { name := "cvt: every f16/bf16 input into every format, every mode" }
  for (sn, src) in [("f16", Format.f16), ("bf16", .bf16)] do
    for (dn, dst) in small do
      if dst != src then
        for b in [0:2 ^ 16] do
          for r in modesInto dst do
            for sat in [false, true] do
              let got := cvt src dst r { satfinite := sat } b
              let ovf : Overflow := if sat then .sat else .ieee
              let want := match Ref.kind src b with
                | .nan => Ref.nan dst
                | .inf s => if sat || dst.special != .ieee then Ref.over dst ovf s else Ref.inf dst s
                | .fin s x => Ref.round dst r ovf s x
              t := t.check (got == want) fun _ =>
                s!"{sn}→{dn} {repr r} sat={sat} {hex b}: {hex got}, want {hex want}"
  return t

/-- `f32` inputs at and around every rounding boundary of each narrow format: each pair of
neighbours, their midpoint, and one `f32` ulp either side of it. -/
def boundaries (_scale : Nat) : Tally := Id.run do
  let mut t : Tally := { name := "cvt: f32 at every rounding boundary of each narrow format" }
  for (dn, dst) in small do
    for i in [0:Ref.count dst - 1] do
      let lo := Ref.at_ dst i
      let hi := Ref.at_ dst (i + 1)
      -- the f32 encodings of lo, hi and the midpoint (exact: narrow formats sit inside f32)
      -- the f32 encoding of a value every narrow format shares with f32 (checked below)
      let enc (x : Rat) : Nat := Format.f32.round .rn .ieee false x.num.natAbs (-(Nat.log2 x.den : Int))
      for (v, x) in [(lo, enc lo), ((lo + hi) / 2, enc ((lo + hi) / 2)), (hi, enc hi)] do
        t := t.check (match Ref.kind .f32 x with | .fin false y => y == v | _ => false)
          fun _ => s!"f32 encoding of a {dn} boundary {hex x}"
        for d in [x - 1, x, x + 1] do
          for s in [false, true] do
            let b := if s then d + 2 ^ 31 else d
            for r in modesInto dst do
              let got := cvt .f32 dst r { satfinite := true } b
              let want := match Ref.kind .f32 b with
                | .fin s' y => Ref.round dst r .sat s' y
                | _ => got
              t := t.check (got == want) fun _ => s!"f32→{dn} {repr r} {hex b}: {hex got}, want {hex want}"
  return t

/-- The 16-bit arithmetic on random operand pairs and on every encoding against the specials. -/
def arith16 (scale : Nat) : Tally := Id.run do
  let mut t : Tally := { name := "arith f16/bf16: add sub mul div fma sqrt rcp, every mode" }
  let mut g : Rng := ⟨7⟩
  for (n, f) in [("f16", Format.f16), ("bf16", .bf16)] do
    for r in ieeeModes do
      for b in [0:2 ^ 16] do
        t := t.check (sqrt f r false b == Ref.sqrt f r b) fun _ => s!"{n} sqrt {repr r} {hex b}"
        t := t.check (rcp f r false b == Ref.div f r f.one b) fun _ => s!"{n} rcp {repr r} {hex b}"
      for _ in [0:40000 * scale] do
        let (a, g1) := operand f g 0x3c00
        let (b, g2) := operand f g1 a
        let (c, g3) := operand f g2 a
        g := g3
        let nb := f.negate b
        t := t.check (add f r false a b == Ref.add f r a b) fun _ => s!"{n} add {repr r} {hex a} {hex b}"
        t := t.check (sub f r false a b == (if f.isNaN b then Ref.nan f else Ref.add f r a nb))
          fun _ => s!"{n} sub {repr r} {hex a} {hex b}"
        t := t.check (mul f r false a b == Ref.mul f r a b) fun _ => s!"{n} mul {repr r} {hex a} {hex b}"
        t := t.check (div f r false a b == Ref.div f r a b) fun _ => s!"{n} div {repr r} {hex a} {hex b}"
        t := t.check (fma f r false a b c == Ref.fma f r a b c)
          fun _ => s!"{n} fma {repr r} {hex a} {hex b} {hex c}"
  return t

/-- Binary32 and binary64 arithmetic on random operands (NaN results compared as NaN: binary64
carries payloads). -/
def arithWide (scale : Nat) : Tally := Id.run do
  let mut t : Tally := { name := "arith f32/f64: add mul div fma sqrt, every mode" }
  let mut g : Rng := ⟨11⟩
  for (n, f) in [("f32", Format.f32), ("f64", .f64)] do
    let same (x y : Nat) := x == y || (f.isNaN x && f.isNaN y)
    for r in ieeeModes do
      for _ in [0:4000 * scale] do
        let (a, g1) := operand f g f.one
        let (b, g2) := operand f g1 a
        let (c, g3) := operand f g2 a
        g := g3
        t := t.check (same (add f r false a b) (Ref.add f r a b)) fun _ => s!"{n} add {repr r} {hex a} {hex b}"
        t := t.check (same (mul f r false a b) (Ref.mul f r a b)) fun _ => s!"{n} mul {repr r} {hex a} {hex b}"
        t := t.check (same (div f r false a b) (Ref.div f r a b)) fun _ => s!"{n} div {repr r} {hex a} {hex b}"
        t := t.check (same (fma f r false a b c) (Ref.fma f r a b c))
          fun _ => s!"{n} fma {repr r} {hex a} {hex b} {hex c}"
        t := t.check (same (sqrt f r false a) (Ref.sqrt f r a)) fun _ => s!"{n} sqrt {repr r} {hex a}"
  return t

/-- The binary32 word arithmetic (`NV.Num.F32`, the SASS fast path's) against the general
definitions, bit for bit (NaNs included): every pair and triple of edge values, and random
operands near each other and far apart, under every IEEE mode with and without `.ftz`. -/
def f32Words (scale : Nat) : Tally := Id.run do
  let mut t : Tally := { name := "f32 word arithmetic = NV.Num: add mul fma, every mode, ftz" }
  let f := Format.f32
  let edges : List Nat := [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x80800000,
    0x00ffffff, 0x3f800000, 0xbf800000, 0x3f800001, 0x3fffffff, 0x4b000000, 0x33800000,
    0x34000000, 0x7f7fffff, 0xff7fffff, 0x7f000000, 0x7f800000, 0xff800000, 0x7fc00000,
    0x7f800001, 0x0d800000, 0x72800000, 0x3eaaaaab, 0x00000003]
  for r in ieeeModes do
    for ftz in [false, true] do
      for a in edges do
        for b in edges do
          t := t.check (F32.add r ftz a.toUInt32 b.toUInt32 == (add f r ftz a b).toUInt32)
            fun _ => s!"add {repr r} ftz={ftz} {hex a} {hex b}"
          t := t.check (F32.mul r ftz a.toUInt32 b.toUInt32 == (mul f r ftz a b).toUInt32)
            fun _ => s!"mul {repr r} ftz={ftz} {hex a} {hex b}"
          for c in edges do
            t := t.check (F32.fma r ftz a.toUInt32 b.toUInt32 c.toUInt32 == (fma f r ftz a b c).toUInt32)
              fun _ => s!"fma {repr r} ftz={ftz} {hex a} {hex b} {hex c}"
  let mut g : Rng := ⟨23⟩
  for r in ieeeModes do
    for ftz in [false, true] do
      for k in [0:6000 * scale] do
        let (a, g1) := operand f g f.one
        -- Near `a` (cancellation, ties) or anywhere (far exponents).
        let (b, g2) := operand f g1 (if k % 2 == 0 then a else f.one)
        let (c, g3) := operand f g2 (if k % 3 == 0 then a else f.one)
        let (raw, g4) := g3.bits 32
        g := g4
        let c := if k % 5 == 0 then raw else c
        t := t.check (F32.add r ftz a.toUInt32 b.toUInt32 == (add f r ftz a b).toUInt32)
          fun _ => s!"add {repr r} ftz={ftz} {hex a} {hex b}"
        t := t.check (F32.mul r ftz a.toUInt32 b.toUInt32 == (mul f r ftz a b).toUInt32)
          fun _ => s!"mul {repr r} ftz={ftz} {hex a} {hex b}"
        t := t.check (F32.fma r ftz a.toUInt32 b.toUInt32 c.toUInt32 == (fma f r ftz a b c).toUInt32)
          fun _ => s!"fma {repr r} ftz={ftz} {hex a} {hex b} {hex c}"
  return t

/-- The SASS MUFU units on machine words (`NV.Num.MufuWord`) against `NV.Num.Approx`'s `.ftz`
binary32 forms, bit for bit: edge values, inputs spread over every encoding, and inputs dense in
the exponents where the functions are finite and nonzero. -/
def mufuWords (scale : Nat) : Tally := Id.run do
  let mut t : Tally := { name := "MUFU on machine words = NV.Num.Approx: ex2 rcp rsqrt (.ftz)" }
  let fns : List (String × (UInt32 → UInt32) × (Nat → Nat)) :=
    [("ex2", MufuWord.ex2, Approx.ex2 .f32 true), ("rcp", MufuWord.rcp, Approx.rcp .f32 true),
     ("rsqrt", MufuWord.rsqrt, Approx.rsqrt .f32 true)]
  let edges : List Nat := [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000,
    0x80800000, 0x3f800000, 0xbf800000, 0x3f800001, 0x3f7fffff, 0x3f000000, 0xbf000000,
    0x42fe0000, 0xc2fc0000, 0x42fe0001, 0xc2fc0001, 0x43000000, 0xc3000000, 0xc3160000,
    0x4b000000, 0xcb000000, 0x4affffff, 0xcaffffff, 0x7f7fffff, 0xff7fffff, 0x7e800000,
    0x7f000000, 0x40000000, 0x40800000, 0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001,
    0xffffffff]
  for (name, fast, spec) in fns do
    let check (t : Tally) (x : Nat) : Tally :=
      t.check (fast x.toUInt32 == (spec x).toUInt32) fun _ =>
        s!"{name} {hex x}: words {hex (fast x.toUInt32).toNat}, NV.Num.Approx {hex (spec x)}"
    for x in edges do t := check t x
    for k in [0:20000 * scale] do
      -- Anywhere (a multiplicative hash of k), and |x| in [2^-8, 2^8) with either sign.
      t := check t (k * 2654435761 % 2 ^ 32)
      t := check t ((k % 2) * 0x80000000 + 0x3b800000 + k * 40503 % 0x08000000)
  return t

/-- NVIDIA's rules beyond IEEE 754: flush to zero, NaN results, min/max, comparisons. -/
def nvidiaRules (_scale : Nat) : Tally := Id.run do
  let mut t : Tally := { name := "ftz, NaN results, min/max, setp, testp, sat, relu" }
  let f := Format.f32
  let sub1 : Nat := 0x00000001
  let cases : List (String × Nat × Nat) := [
    -- .ftz: subnormal inputs and results become sign-preserving zero
    ("add.ftz sub+0", add f .rn true sub1 0, 0),
    ("add.ftz -sub+0", add f .rn true 0x80000001 0x80000000, 0x80000000),
    ("mul.ftz → subnormal", mul f .rn true 0x00800000 0x3f000000, 0),
    ("mul → subnormal", mul f .rn false 0x00800000 0x3f000000, 0x00400000),
    ("mul.ftz rounds up to normal", mul f .rn true 0x00ffffff 0x3f000000, 0x00800000),
    ("fma.ftz sub*big", fma f .rn true sub1 0x7f000000 0, 0),
    -- f32 NaN results are 0x7fffffff whatever the operands
    ("add NaN payload", add f .rn false 0xffc00001 0x3f800000, 0x7fffffff),
    ("inf - inf", add f .rn false 0x7f800000 0xff800000, 0x7fffffff),
    ("0 * inf", mul f .rn false 0 0x7f800000, 0x7fffffff),
    ("sqrt -1", sqrt f .rn false 0xbf800000, 0x7fffffff),
    ("sqrt -0", sqrt f .rn false 0x80000000, 0x80000000),
    -- binary64 carries the first NaN operand's payload, quieted
    ("f64 NaN payload", add .f64 .rn false 0x7ff0000000000001 0x7ff8000000000002, 0x7ff8000000000001),
    ("f64 NaN second", mul .f64 .rn false 0x3ff0000000000000 0xfff4000000000000, 0xfffc000000000000),
    -- min/max: -0 < +0, NaN loses unless .NaN
    ("min ±0", NV.Num.min f false false false 0 0x80000000, 0x80000000),
    ("max ±0", NV.Num.max f false false false 0x80000000 0, 0),
    ("min NaN,1", NV.Num.min f false false false 0x7fc00000 0x3f800000, 0x3f800000),
    ("min.NaN NaN,1", NV.Num.min f false true false 0x7fc00000 0x3f800000, 0x7fffffff),
    ("min NaN,NaN", NV.Num.min f false false false 0x7fc00000 0xffc00000, 0x7fffffff),
    ("min.xorsign.abs -2,3", NV.Num.min f false false true 0xc0000000 0x40400000, 0xc0000000),
    ("max.xorsign.abs -2,-3", NV.Num.max f false false true 0xc0000000 0xc0400000, 0x40400000),
    ("min.ftz sub,0", NV.Num.min f true false false 0x80000001 0, 0x80000000),
    -- sat, relu
    ("add.sat", satUnit f (add f .rn false 0x3f800000 0x3f800000), 0x3f800000),
    ("sat NaN", satUnit f 0x7fffffff, 0), ("sat -2", satUnit f 0xc0000000, 0),
    ("sat 0.5", satUnit f 0x3f000000, 0x3f000000),
    ("relu NaN", relu .f16 0xfe00, 0x7fff), ("relu -1", relu .f16 0xbc00, 0),
    -- abs/neg: binary32 NaN canonical, binary64 NaN passes abs unchanged
    ("abs NaN f32", NV.Num.abs f false 0xffc00001, 0x7fffffff),
    ("abs NaN f64", NV.Num.abs .f64 false 0xfff0000000000001, 0xfff0000000000001),
    ("abs.ftz sub", NV.Num.abs f true 0x80000001, 0),
    ("neg 0", NV.Num.neg f false 0, 0x80000000),
    ("copysign", copysign f 0x80000000 0x3f800000, 0xbf800000),
    -- the rounding of an exact zero sum
    ("x - x rn", add f .rn false 0x3f800000 0xbf800000, 0),
    ("x - x rm", add f .rm false 0x3f800000 0xbf800000, 0x80000000),
    ("-0 + -0", add f .rn false 0x80000000 0x80000000, 0x80000000),
    -- div.approx: |b| in (2^126, 2^128) gives a signed zero, or NaN for infinite a
    ("div.approx big b", Approx.divApprox false 0x3f800000 0x7f000001, 0),
    ("div.approx 0/subnormal", Approx.divApprox false 0 1, 0),
    ("div.approx.ftz 0/subnormal", Approx.divApprox true 0 1, 0x7fffffff),
    ("div.approx 1/subnormal", Approx.divApprox false 0x3f800000 0x00400000, 0x7f000000),
    ("div.approx inf/big", Approx.divApprox false 0x7f800000 0xff000001, 0x7fffffff),
    -- rcp.approx.ftz.f64: the upper word, canonical NaN 0x7fffffff00000000
    ("rcp.approx.ftz.f64 2", Approx.rcp Approx.f64Hi true 0x4000000000000000, 0x3fe0000000000000),
    ("rcp.approx.ftz.f64 NaN", Approx.rcp Approx.f64Hi true 0x7ff8000000000001, 0x7fffffff00000000),
    ("rsqrt.approx.ftz.f64 4", Approx.rsqrt Approx.f64Hi true 0x4010000000000000, 0x3fe0000000000000),
    -- binary64 keeps NaN payloads outside the two .approx.ftz.f64 instructions
    ("rsqrt.approx.f64 NaN", Approx.rsqrt .f64 false 0x7ff8000000000001, 0x7ff8000000000001),
    ("min.f64 NaN NaN", NV.Num.min .f64 false false false 0x7ff8000000000001 0xfff8000000000002,
      0x7ff8000000000001),
    -- scaled conversions: n1 flushes subnormal sources; 0xff is a NaN scale
    ("cvt.scaled::n1 subnormal", cvtScaled .f32 .e4m3 .rn { satfinite := true } 0x00 true true
      0x007fffff, 0),
    ("cvt.scaled::n1 4/2", cvtScaled .f32 .e4m3 .rn { satfinite := true } 0x80 true true
      0x40800000, 0x40),
    ("cvt.scaled NaN scale", cvtScaled .f32 .e4m3 .rn { satfinite := true } 0xff true true
      0x3f800000, 0x7f),
    ("cvt.scaled::n2 e4m3 1 × 4 → bf16", cvtScaled .e4m3 .bf16 .rn {} 0x81 false false 0x38, 0x4080),
    -- s2f6: 1/64 steps in [-2, 127/64]
    ("s2f6 1", toS2f6 .f32 .rn false 0x7f 0x3f800000, 0x40),
    ("s2f6 bf16 -1", toS2f6 .bf16 .rn false 0x7f 0xbf80, 0xc0),
    ("s2f6 2 saturates", toS2f6 .f32 .rn false 0x7f 0x40000000, 0x7f),
    ("s2f6 -2 saturates sign-preserved", toS2f6 .f32 .rn false 0x7f 0xc0000000, 0x81),
    ("s2f6 NaN", toS2f6 .f32 .rn false 0x7f 0x7fc00000, 0x7f),
    ("s2f6 relu -1", toS2f6 .f32 .rn true 0x7f 0xbf800000, 0),
    ("s2f6 1/128 ties to even", toS2f6 .f32 .rn false 0x7f 0x3c000000, 0),
    ("s2f6 3/128 ties to even", toS2f6 .f32 .rn false 0x7f 0x3cc00000, 0x02),
    ("s2f6 3/128 rz", toS2f6 .f32 .rz false 0x7f 0x3cc00000, 0x01),
    ("s2f6 2 / scale 2", toS2f6 .f32 .rn false 0x80 0x40000000, 0x40),
    ("s2f6 → bf16 1", ofS2f6 .bf16 {} 0x7f 0x40, 0x3f80),
    ("s2f6 → bf16 -2", ofS2f6 .bf16 {} 0x7f 0x80, 0xc000),
    ("s2f6 → bf16 1 × 4", ofS2f6 .bf16 {} 0x81 0x40, 0x4080),
    -- exponents far outside every format
    ("ex2 -max", Approx.ex2 f false 0xff7fffff, 0), ("ex2 max", Approx.ex2 f false 0x7f7fffff, 0x7f800000),
    ("1e100000", ofDecimal .f64 1 100000, 0x7ff0000000000000),
    ("1e-100000 rp", ofDecimal .f64 1 (-100000) .rp, 1), ("1e-100000", ofDecimal .f64 1 (-100000), 0)]
  for (n, got, want) in cases do
    t := t.check (got == want) fun _ => s!"{n}: {hex got}, want {hex want}"
  let bcases : List (String × Bool × Bool) := [
    ("setp.eq ±0", cmp f .eq false 0 0x80000000, true),
    ("setp.lt NaN", cmp f .lt false 0x7fc00000 0, false),
    ("setp.ltu NaN", cmp f .ltu false 0x7fc00000 0, true),
    ("setp.ne NaN", cmp f .ne false 0x7fc00000 0, false),
    ("setp.neu NaN", cmp f .neu false 0x7fc00000 0, true),
    ("setp.lt -1 -0", cmp f .lt false 0xbf800000 0x80000000, true),
    ("setp.gt.ftz sub 0", cmp f .gt true 1 0, false),
    ("setp.gt sub 0", cmp f .gt false 1 0, true),
    ("setp.num", cmp f .num false 0 0x3f800000, true),
    ("testp.normal 0", testp f .normal 0, true),
    ("testp.subnormal", testp f .subnormal 1, true),
    ("testp.normal sub", testp f .normal 1, false),
    ("testp.finite inf", testp f .finite 0x7f800000, false),
    ("testp.infinite", testp f .infinite 0xff800000, true),
    ("testp.notanumber", testp f .notanumber 0x7f800001, true)]
  for (n, got, want) in bcases do
    t := t.check (got == want) fun _ => s!"{n}: {got}, want {want}"
  -- div.full: within 2 ulp of the quotient (PTX §9.7.3.8), over the whole range
  let mut g : Rng := ⟨17⟩
  for _ in [0:20000] do
    let (a, g1) := operand f g f.one
    let (b, g2) := operand f g1 a
    g := g2
    let got := Approx.divFull false a b
    let cr := div f .rn false a b
    let ok := if f.isNaN cr || f.isNaN got then f.isNaN cr && f.isNaN got
      else (f.orderKey true got - f.orderKey true cr).natAbs ≤ 2
    t := t.check ok fun _ => s!"div.full {hex a}/{hex b}: {hex got}, quotient {hex cr}"
  return t

/-- Float to integer and back, against the definitions of the integer rounding modes. -/
def integers (scale : Nat) : Tally := Id.run do
  let mut t : Tally := { name := "cvt: float ↔ integer, every mode and width" }
  let toward (r : Rounding) (x : Rat) : Int :=
    match r with
    | .rz => if x < 0 then x.ceil else x.floor
    | .rm => x.floor
    | .rp => x.ceil
    | .rn | .rna =>
      let d := x - x.floor
      if d > 1 / 2 then x.floor + 1 else if d < 1 / 2 then x.floor
      else if r == .rna then (if x < 0 then x.floor else x.floor + 1)
      else if x.floor % 2 == 0 then x.floor else x.floor + 1
  for (n, f) in [("f16", Format.f16), ("bf16", .bf16)] do
    for b in [0:2 ^ 16] do
      for r in ieeeModes do
        for (signed, w) in [(true, 8), (false, 8), (true, 16), (false, 16), (true, 32), (false, 32),
                            (true, 64), (false, 64)] do
          let lo : Int := if signed then -2 ^ (w - 1) else 0
          let hi : Int := if signed then 2 ^ (w - 1) - 1 else 2 ^ w - 1
          let want : Nat := match Ref.kind f b with
            | .nan => if w == 64 then 2 ^ (w - 1) else 0
            | .inf s => ((if s then lo else hi) % 2 ^ w).toNat
            | .fin s x => ((Max.max lo (Min.min hi (toward r (Ref.val s x)))) % 2 ^ w).toNat
          let got := toInt f r signed w false b
          t := t.check (got == want) fun _ => s!"{n}→{if signed then "s" else "u"}{w} {repr r} {hex b}"
        let ri := roundInt f r false b
        let want := match Ref.kind f b with
          | .fin s x => Ref.result f .rn (toward r (Ref.val s x)) s
          | .nan => f.nan
          | .inf _ => b
        t := t.check (ri == want) fun _ => s!"{n} roundInt {repr r} {hex b}"
  for i in [0:2 ^ 16] do
    for r in ieeeModes do
      for (n, f) in [("f16", Format.f16), ("bf16", .bf16), ("f32", .f32)] do
        for signed in [false, true] do
          let x := intOfBits signed 16 i
          t := t.check (ofInt f r signed 16 i == Ref.result f r x false)
            fun _ => s!"{if signed then "s" else "u"}16→{n} {repr r} {hex i}"
  let mut g : Rng := ⟨5⟩
  for _ in [0:20000 * scale] do
    let (i, g') := g.bits 64
    g := g'
    for r in ieeeModes do
      for (n, f) in [("f32", Format.f32), ("f64", .f64), ("f16", .f16)] do
        t := t.check (ofInt f r true 64 i == Ref.result f r (intOfBits true 64 i) false)
          fun _ => s!"s64→{n} {repr r} {hex i}"
      t := t.check (toInt .f64 r true 32 false i ==
          match Ref.kind .f64 i with
          | .nan => 2 ^ 31
          | .inf s => if s then 2 ^ 31 else 2 ^ 31 - 1
          | .fin s x => ((Max.max (-2 ^ 31) (Min.min (2 ^ 31 - 1) (toward r (Ref.val s x)))) % 2 ^ 32).toNat)
        fun _ => s!"f64→s32 {repr r} {hex i}"
  return t

/-- Decimal literals against Python's correctly rounded `float` (`tools/num/approx_vectors.py`'s
companion list). -/
def decimals (_scale : Nat) : Tally := Id.run do
  let mut t : Tally := { name := "decimal literals → f64" }
  let cases : List (Nat × Int × Nat) := [
    (1, -1, 0x3fb999999999999a),
    (2, -1, 0x3fc999999999999a),
    (3, -1, 0x3fd3333333333333),
    (1, -300, 0x01a56e1fc2f8f359),
    (22250738585072014, -324, 0x0010000000000000),
    (22250738585072011, -324, 0x000fffffffffffff),
    (49, -325, 0x0000000000000001),
    (24703282292062327, -340, 0x0000000000000000),
    (24703282292062328, -340, 0x0000000000000001),
    (17976931348623157, 292, 0x7fefffffffffffff),
    (17976931348623158, 292, 0x7fefffffffffffff),
    (17976931348623159, 292, 0x7ff0000000000000),
    (1, 309, 0x7ff0000000000000),
    (1, -400, 0x0000000000000000),
    (123456789012345678901234567890, 0, 0x45f8ee90ff6c373e),
    (3141592653589793238462643383279, -30, 0x400921fb54442d18),
    (9007199254740993, 0, 0x4340000000000000),
    (9007199254740995, 0, 0x4340000000000002),
    (1, -6, 0x3eb0c6f7a0b5ed8d),
    (1, 23, 0x44b52d02c7e14af6),
    (898846567431158, 293, 0x7fe0000000000000),
    (5, -324, 0x0000000000000001),
    (1, 22, 0x4480f0cf064dd592),
    (0, 0, 0x0000000000000000),
    (21250919909, 64, 0x4f5e11a9122399a0),
    (960566905443639394257, 208, 0x6f7957b398ce64dd),
    (5992, 256, 0x75df2d5586f2d341),
    (65, -121, 0x2730c8dd21cf2425),
    (12, 104, 0x55c0be33e5481279),
    (33870412123024, -248, 0x0f413b19fc02eba9),
    (68149772622318119, 239, 0x750d0c51ca3df82f),
    (3658, 305, 0x7ff0000000000000),
    (72822376571655375090, 250, 0x77f60e19fbdd8510),
    (914761360679426581, -114, 0x2bff433ff2ab7c44),
    (72, -204, 0x15f692f538e956f1),
    (1800188483, 213, 0x6e13eba8ac69530f),
    (9354, -25, 0x3b91ab511cdd05e9),
    (786295579237787696, -155, 0x237768b7ad89704f),
    (9529, 244, 0x736b41d65e114c11),
    (117549986606898237638, 220, 0x71c690a62bccb200),
    (9012421865441972121656, 293, 0x7ff0000000000000),
    (8328454, 204, 0x6bb955427e415eb7),
    (44211436549913, 136, 0x5f0149bf9ccff83f),
    (8359776232105486216, 30, 0x4a16e1455305c8c6),
    (9362026907, -91, 0x2f1c6aed68863606),
    (589, -33, 0x39a7e486943b9d13),
    (41496206415154235, 283, 0x7e23d4031fd4a31d),
    (121, 184, 0x6691cc4136f738a0),
    (21390406252591, 160, 0x63ebacdd002077a6),
    (10924971798993, 231, 0x729999a87d24c2db),
    (5787620081580088011, 8, 0x457debd9f5471014),
    (75052319080847610563196, 253, 0x7935ad6cf67014e5),
    (945717734175442, -245, 0x102d5d68a2823ee1),
    (509059211, -274, 0x08da4354764900b9),
    (94, -23, 0x3b91c18fa30dbaf4),
    (811529381761590249652, 116, 0x5c5be9b90958277d),
    (7372860243, 15, 0x4518650e37f4c30f),
    (8, 23, 0x44e52d02c7e14af6),
    (640596, -221, 0x1341aaa14f32f50d),
    (3930878876558081, -46, 0x399fe41db347233b),
    (96779, -87, 0x2ee7804e4181e1b8),
    (8735411241649, -258, 0x0d0e89e4d36e6b55),
    (471008, 71, 0x4fda0886eaa7deec),
    (944565344516544383, 100, 0x586df734c2e6e089),
    (814419959085439566, 85, 0x554d16f9552692fc),
    (973594882085, 49, 0x4c983c1ac22e86c3),
    (20256262, -256, 0x0c4d017fd921bb7a),
    (158648, -103, 0x2ba159a346be6170),
    (9149807584800820423147, 263, 0x7b189ccdf9cee5de),
    (275510, -52, 0x36442206c52bd1b8),
    (3, 89, 0x5282d9d5d24c02ab),
    (703065871400527598, 239, 0x7542baca71d13f89),
    (21273393601, 187, 0x68e6c46722a68c86),
    (62578345457050480329, 68, 0x52292a83430d340d),
    (8470120196141, 309, 0x7ff0000000000000),
    (3350341843241, -272, 0x0a107bee4a70c4a8),
    (7392493, -174, 0x1d3be62442f6b7c2),
    (5572, 275, 0x79cf6ed59e8175a7),
    (14, -340, 0x0000000000000000),
    (9898867670432435328, -237, 0x12abf45cc3c229db),
    (28405785249, -268, 0x0a75d666e6ffe15d),
    (6312082, -188, 0x1a50c34f90b667a4),
    (423453643803100182239, 276, 0x7d84b833d4ed3f14),
    (135180451219, -222, 0x1426c110b92fdaba),
    (8654030390367704, 155, 0x636ca9e2383665b4),
    (368871839, -236, 0x10b65ecd525ab557),
    (320083113492482671180671, 150, 0x63f4b4d8a44abae9),
    (3495959320664934189679, -130, 0x296a45f7a29bcc38),
    (21127650057375542, 216, 0x702b379ed6acd76d),
    (9, -35, 0x38dde85a6ef78951),
    (813335587545698983571, -73, 0x351f292c1759cbfe),
    (51261217079247943, -112, 0x2c25e60e84942d6b),
    (898189135500813453, 174, 0x67c932e7270d265b),
    (32798268350, 287, 0x7db40f6908a75179),
    (7750377724659338043753126, -108, 0x2eae1ce2c82b8518),
    (8684537, 164, 0x636cc3c02d0ca6ee),
    (33204409334, -312, 0x0156c53b77b95e5b),
    (507063908, -75, 0x322b5740c148510b),
    (5776076, 117, 0x59a179ad52b7e248),
    (97368641565910965538891, -115, 0x2cd44f7432f01904),
    (3717, 141, 0x5df30d05464b9e48),
    (5666295, -131, 0x26232d9d2769a6ca),
    (34380527880300, 150, 0x61e31adb0a4ff017),
    (771067330553936139791, -254, 0x0f88843b071bc43b),
    (7340139823268990419210, -136, 0x283cebef84315491),
    (3215963660368319, 104, 0x58bfe1edc0a7bd38),
    (942384200086309558522, 65, 0x51c841401fc7b8a0),
    (836940526078287, -254, 0x0e4be75619af2b74),
    (182715508236710085760565, 264, 0x7b9332f454fb0c5a),
    (738432896133621, -191, 0x1b5dec58d4d760f4),
    (76663052900741959315, 221, 0x71f26539d68a6d87),
    (2805, -326, 0x0000000000000006),
    (636599033793523800413254, -198, 0x1bc42754cd09c687),
    (29703452253198, -312, 0x01f3e466983ba7c9),
    (228470564, -41, 0x3927b9ca15f69259),
    (46980012659752718, -75, 0x33d2df9d32eea7db),
    (961761355399194631, -206, 0x18e56cf94d9b017f),
    (95, 22, 0x44e925734d5b8905),
    (656793539179346, 189, 0x6a40c23ddd411493),
    (70604444117837, -207, 0x17d49dc134f275d8),
    (603562000888283100, 182, 0x69693b700b99dbce),
    (8, -153, 0x205ad0cc33744e4b),
    (21625979880187983016, 144, 0x61d808e0007b36d6),
    (76006795016352973871, -277, 0x0a8d3742b117e70d),
    (71650067633, 203, 0x6c554888982dc07e),
    (904204602017334482, -232, 0x137f2ba5225462d4),
    (286494449398967445, -145, 0x2573dbde6030b56f),
    (45310713, -240, 0x0fb2022d083f81f7),
    (80953735970409773, -312, 0x02aa78c47f5db46f),
    (1225747635899110334003642, 113, 0x5c6514828e887695),
    (83775712775, 184, 0x6866f3da3d22d73f),
    (4650402, 123, 0x5adad5ee11a664ed),
    (73174755739473540, -87, 0x3160293c3b168f6c),
    (39234892961509526388526, 232, 0x74cac1ede16cd3fa),
    (7508278, -200, 0x17d5ec83d1dc1907),
    (55216621919697, 112, 0x5a0a1a3b8c843105),
    (90505896326, -94, 0x2eb195118b04b1cb),
    (29931941137134, -30, 0x3c81412cc3c661f9),
    (2531, 34, 0x47b30a87582da521),
    (33176, -200, 0x1758cca586f2e362),
    (840689251706982, -244, 0x10605097c69f5d30),
    (8572260513285, -174, 0x1e7eda49d12aad20),
    (3047839533333860784150, 101, 0x595d81f1ddc3bfe8),
    (48871786055471976, 91, 0x564aa2d3da01bd52)
  ]
  for (m, e, want) in cases do
    let got := ofDecimal .f64 m e
    t := t.check (got == want) fun _ => s!"{m}e{e}: {hex got}, want {hex want}"
  return t

/-- The tensor-core model on the cases Khattak and Mikaitis use to establish each feature
(§4.1.6), and on every single FP8 product, which it must return exactly. -/
def tensor (_scale : Nat) : Tally := Id.run do
  let mut t : Tally := { name := "tensor cores: published feature cases, every FP8 product (E4M3, E5M2, mixed)" }
  let m := Hopper.f16f32.value
  let b16 (x : Rat) : Nat := Ref.round .bf16 .rn .ieee (decide (x < 0)) (if x < 0 then -x else x)
  let f32 (x : Rat) : Nat := Ref.round .f32 .rn .ieee (decide (x < 0)) (if x < 0 then -x else x)
  let p := Ref.pow2
  -- products kept unnormalized: 1.5·1.5 + (2^-23 + 2^-24 + 2^-25) + 2^-25 = 2.25 + 2^-22
  let c1 := m.block .bf16 .bf16 .f32 #[b16 (3 / 2), b16 (p (-23) + p (-24) + p (-25)), b16 (p (-25))]
    #[b16 (3 / 2), b16 1, b16 1] 0
  t := t.check (c1 == f32 (9 / 4 + p (-22))) fun _ => s!"unnormalized products: {hex c1}"
  -- the alignment exponent never drops below -133
  let c2 := m.block .bf16 .bf16 .f32 #[b16 ((127 / 64) * p (-75)), b16 (p (-79)), b16 (p (-79)), b16 (p (-79))]
    #[b16 (p (-75)), b16 (p (-78)), b16 (p (-79)), b16 (p (-79))] 0
  t := t.check (c2 == f32 (p (-149))) fun _ => s!"min alignment exponent, kept: {hex c2}"
  let c3 := m.block .bf16 .bf16 .f32 #[b16 ((127 / 64) * p (-75)), b16 ((3 / 2) * p (-79)), b16 (p (-80)),
    b16 (p (-80))] #[b16 (p (-75)), b16 (p (-78)), b16 (p (-79)), b16 (p (-79))] 0
  t := t.check (c3 == 0) fun _ => s!"min alignment exponent, truncated: {hex c3}"
  -- a subnormal accumulator aligns at -126
  let c4 := m.block .bf16 .bf16 .f32 #[b16 (p (-75)), b16 (p (-76)), b16 (p (-76))]
    #[b16 (p (-76)), b16 (p (-76)), b16 (p (-76))] (f32 (p (-127)))
  t := t.check (c4 == f32 (p (-127))) fun _ => s!"subnormal c: {hex c4}"
  -- 13 fraction bits through wgmma with FP8: two 2^-13 products are kept, two 2^-14 vanish
  let f8 := Hopper.f8f32.value
  let e4 (x : Rat) : Nat := Ref.round .e4m3 .rn .sat false x
  let c5 := f8.block .e4m3 .e4m3 .f32 #[e4 1, e4 (p (-6)), e4 (p (-6)), e4 (p (-7)), e4 (p (-7))]
    #[e4 1, e4 (p (-7)), e4 (p (-7)), e4 (p (-7)), e4 (p (-7))] 0
  t := t.check (c5 == f32 (1 + p (-12))) fun _ => s!"fp8 13 bits: {hex c5}"
  -- every single product of two FP8 values is exact
  for (n, f) in [("e4m3", Format.e4m3), ("e5m2", .e5m2)] do
    for a in [0:256] do
      for b in [0:256] do
        let got := f8.block f f .f32 #[a] #[b] 0
        let want := match Ref.kind f a, Ref.kind f b with
          | .fin s x, .fin u y => if x * y == 0 then 0 else f32 (Ref.val (s != u) (x * y))
          | .inf s, .fin u y | .fin u y, .inf s => if y == 0 then Format.f32.nan else Ref.inf .f32 (s != u)
          | .inf s, .inf u => Ref.inf .f32 (s != u)
          | _, _ => Format.f32.nan
        t := t.check (got == want) fun _ => s!"{n} {hex a}·{hex b}: {hex got}, want {hex want}"
  for a in [0:256] do
    for b in [0:256] do
      let got := f8.block .e4m3 .e5m2 .f32 #[a] #[b] 0
      let want := match Ref.kind .e4m3 a, Ref.kind .e5m2 b with
        | .fin s x, .fin u y => if x * y == 0 then 0 else f32 (Ref.val (s != u) (x * y))
        | .fin s x, .inf u => if x == 0 then Format.f32.nan else Ref.inf .f32 (s != u)
        | _, _ => Format.f32.nan
      t := t.check (got == want) fun _ => s!"e4m3·e5m2 {hex a}·{hex b}: {hex got}, want {hex want}"
  return t

/-- The fast tensor datapath against the model: every single product of the 8-bit formats with
accumulators of every class, and random blocks of every Hopper datapath. -/
def fastTensor (scale : Nat) : Tally := Id.run do
  let mut t : Tally := { name := "tensor fast path = model: operands, every FP8 product, random blocks and tiles" }
  let same (m : TensorModel) (aF bF outF : Format) (a b : Array Nat) (c : Nat) : Bool :=
    ((m.prepare aF.man bF.man outF).block (a.map (TensorModel.operand aF))
      (b.map (TensorModel.operand bF)) c.toUInt64).toNat ==
      m.blockVals aF.man bF.man outF (a.map aF.decode) (b.map bF.decode) c
  -- operands against their definition through `Format.decode`
  for f in [Format.f16, .bf16, .e4m3, .e5m2, .e2m3, .e3m2, .e2m1, .ue8m0, .ue4m3, .ue5m3] do
    for x in [0:2 ^ f.width] do
      let want : UInt32 := match f.decode x with
        | .nan => 0x1000000
        | .inf n => 0x800000 ||| (if n then 0x400000 else 0)
        | .fin n m e =>
          ((if n then 0x400000 else 0) : UInt32) ||| (if m = 0 then (0x1800000 : UInt32)
            else ((e + f.man + 512).toNat.toUInt32 <<< 12) ||| m.toUInt32)
      t := t.check (TensorModel.operand f x == want) fun _ => s!"operand {repr f} {hex x}"
  for x in [0:2 ^ 19] do
    let x := x * 2 ^ 13
    let want : UInt32 := match Format.tf32.decode x with
      | .nan => 0x1000000
      | .inf n => 0x800000 ||| (if n then 0x400000 else 0)
      | .fin n m e =>
        ((if n then 0x400000 else 0) : UInt32) ||| (if m = 0 then (0x1800000 : UInt32)
          else ((e + 10 + 512).toNat.toUInt32 <<< 12) ||| m.toUInt32)
    t := t.check (TensorModel.operand .tf32 x == want) fun _ => s!"operand tf32 {hex x}"
  let f8 := Hopper.f8f32.value
  for (aF, bF) in [(Format.e4m3, Format.e4m3), (.e5m2, .e5m2), (.e4m3, .e5m2)] do
    for x in [0:256] do
      for y in [0:256] do
        for c in [0, 0x3f800000, 0xbf000000, 0x00000003, 0x7f800000, 0xff800000, 0x7fffffff] do
          t := t.check (same f8 aF bF .f32 #[x] #[y] c) fun _ => s!"fp8 {hex x}·{hex y} + {hex c}"
  let mut g : Rng := ⟨23⟩
  let cases : List (TensorModel × Format × Format) := [(Hopper.f16f32.value, .f16, .f32),
    (Hopper.f16f32.value, .bf16, .f32), (Hopper.f16f16.value, .f16, .f16),
    (Hopper.tf32f32.value, .tf32, .f32), (Hopper.f8f32.value, .e4m3, .f32),
    (Hopper.f8f32.value, .e5m2, .f32), (Hopper.f8f16.value, .e4m3, .f16)]
  for (m, inF, outF) in cases do
    for _ in [0:20000 * scale] do
      let (len, g1) := g.bits 5
      g := g1
      let len := len % m.blockSize + 1
      let mut a : Array Nat := #[]
      let mut b : Array Nat := #[]
      for _ in [0:len] do
        let (x, g2) := operand inF g (inF.one)
        let (y, g3) := operand inF g2 x
        g := g3
        a := a.push x
        b := b.push y
      let (c, g4) := operand outF g (outF.one)
      g := g4
      t := t.check (same m inF inF outF a b c) fun _ => s!"{repr inF} block {a.map hex} {b.map hex} {hex c}"
  -- tiles against `dot` elementwise: 3 × 2 outputs, K over two and a half blocks
  for (m, inF, outF) in cases do
    for _ in [0:200 * scale] do
      let k := 2 * m.blockSize + m.blockSize / 2
      let mut rows : Array (Array Nat) := #[]
      for _ in [0:5] do
        let mut v : Array Nat := #[]
        for _ in [0:k] do
          let (x, g1) := operand inF g inF.one
          g := g1
          v := v.push x
        rows := rows.push v
      let (as, bs) := (rows.extract 0 3, rows.extract 3 5)
      let mut cs : Array Nat := #[]
      for _ in [0:6] do
        let (c, g1) := operand outF g outF.one
        g := g1
        cs := cs.push c
      for m in [m, { m with outRound := .rp }] do
        let got := m.tile inF inF outF as bs cs
        let want := (List.range 6).toArray.map fun ij => m.dot inF inF outF as[ij / 2]! bs[ij % 2]! cs[ij]!
        t := t.check (got == want) fun _ => s!"{repr inF} tile: {got.map hex}, dot {want.map hex}"
  -- empty and unequal lengths
  let e8 := Hopper.f8f32.value
  for (as, bs) in [(#[#[0]], #[#[]]), (#[#[]], #[#[]]), (#[#[0x38, 0x38]], #[#[0x38]])] do
    let got := e8.tile .e4m3 .e4m3 .f32 as bs #[0x3f800001]
    let want := #[e8.dot .e4m3 .e4m3 .f32 as[0]! bs[0]! 0x3f800001]
    t := t.check (got == want) fun _ => s!"tile lengths {as} {bs}: {got.map hex}, dot {want.map hex}"
  return t

/-- Block-scaled values against the product of the parts' reference values, and OCP MX §6.3's
conversion on blocks with known results. -/
def blocks (_scale : Nat) : Tally := Id.run do
  let mut t : Tally := { name := "block-scaled formats: MXFP4/NVFP4 values, MX conversion" }
  for (n, b) in [("mxfp4", BlockFormat.mxfp4), ("nvfp4", .nvfp4), ("mxfp8e4m3", .mxfp8e4m3)] do
    for s in [0:2 ^ b.scale.width] do
      for x in [0:2 ^ b.elem.width] do
        let want : Option Rat := match Ref.kind b.scale s, Ref.kind b.elem x with
          | .fin _ y, .fin u z => some (Ref.val u (y * z))
          | _, _ => none
        let got := valOf (b.value s x)
        t := t.check (got == want || (got == none && want == none)) fun _ => s!"{n} {hex s} {hex x}"
  let q (n d : Nat) : Rat := (n : Rat) / (d : Rat)
  let v (x : Rat) : Val := match Ref.kind .f32 (Ref.round .f32 .rn .ieee (decide (x < 0)) (if x < 0 then -x else x)) with
    | .fin s y => .fin s y.num.natAbs (-(Nat.log2 y.den : Int))
    | _ => .nan
  let cases : List (Array Val × Nat × Array Nat) := [
    (#[v 6, v (-3), v (q 1 2), v 0], 0x7f, #[0x7, 0xd, 0x1, 0x0]),
    (#[v 100, v 1], 0x83, #[0x7, 0x0]),
    (#[v 0, v 0], 0x7f, #[0x0, 0x0])]
  for (vs, s, xs) in cases do
    let (gs, gx) := BlockFormat.mxfp4.quantize vs
    t := t.check (gs == s && gx == xs) fun _ => s!"mxfp4 quantize: scale {hex gs}, elements {gx.map hex}"
  return t

/-- Whether the special-function unit's `got` for input `x` is within PTX's documented accuracy of
the true value, `cr` being the true value correctly rounded (so half an ulp of slack). -/
def withinBound (fn : String) (x got cr : Nat) : Bool :=
  let f := Format.f32
  if f.isNaN cr || f.isInf cr || f.isNaN got || f.isInf got then got == cr else
  let v (b : Nat) : Rat := match Ref.kind f b with | .fin s y => Ref.val s y | _ => 0
  let absR (q : Rat) : Rat := if q < 0 then -q else q
  let ulps := (f.orderKey true got - f.orderKey true cr).natAbs
  let err := absR (v got - v cr)
  let slack := absR (v (cr + 1) - v cr) / 2
  let ax := absR (v x)
  let p := Ref.pow2
  match fn with
  | "ex2" => ulps ≤ 2
  | "rcp" => ulps ≤ 1
  | "lg2" => if ax > 1 / 2 && ax < 2 then err ≤ p (-22) + slack else err ≤ p (-22) * absR (v cr) + slack
  | "rsqrt" => err ≤ (10718 / 10000) * p (-23) * absR (v cr) + slack
  | "sqrt" => err ≤ p (-23) * absR (v cr) + slack
  | "sin" | "cos" =>
    let tight : Rat := if fn == "sin" then 7072 / 10000 else 7579 / 10000
    if ax ≤ 628318 / 100000 then err ≤ tight * p (-20) + slack
    else if ax ≤ 314159 / 1000 then err ≤ (6157 / 10000) * p (-14) + slack else true
  | _ =>
    let tight : Rat := if fn == "sin2pi" then 1 else 10718 / 10000
    if ax ≤ 1 then err ≤ tight * p (-20) + slack
    else if ax ≤ 50 then err ≤ (8706 / 10000) * p (-14) + slack else true

/-- The `.approx` functions against mpmath: the special-function unit's binary32 results within
PTX's documented accuracy, the correctly rounded ones equal. -/
def approx (lines : List String) (_scale : Nat) : Tally := Id.run do
  let mut t : Tally := { name := "approx: MUFU within PTX's bounds, the rest correctly rounded" }
  for l in lines do
    match words l with
    | [fn, fmt, x, y] =>
      let f : Format := if fmt == "f16" then .f16 else if fmt == "bf16" then .bf16 else .f32
      let x := parseHex x
      let y := parseHex y
      let got := match fn with
        | "ex2" => Approx.ex2 f false x
        | "lg2" => Approx.lg2 false x
        | "rsqrt" => Approx.rsqrt f false x
        | "rcp" => Approx.rcp f false x
        | "sqrt" => Approx.sqrt false x
        | "sin" => Approx.sinCos false false false x
        | "cos" => Approx.sinCos true false false x
        | "sin2pi" => Approx.sinCos false true false x
        | "cos2pi" => Approx.sinCos true true false x
        | _ => Approx.tanh f x
      let ok := if fn == "tanh" || fmt != "f32" then got == y else withinBound fn x got y
      t := t.check ok fun _ => s!"{fn} {fmt} {hex x}: {hex got}, correctly rounded {hex y}"
    | _ => pure ()
  return t

/-! ## Entry points -/

/-- The model of `gpu`'s datapath for inputs `inF` into `outF`, if there is one. -/
def modelFor (gpu : String) (inF outF : Format) : Option TensorModel :=
  let fp8 := inF == .e4m3 || inF == .e5m2
  let f16out := outF == .f16
  match gpu with
  | "H100" | "H200" => some <|
    if fp8 then (if f16out then Hopper.f8f16.value else Hopper.f8f32.value)
    else if inF == .tf32 then Hopper.tf32f32.value
    else if f16out then Hopper.f16f16.value else Hopper.f16f32.value
  | "A100" =>
    if fp8 then none else some <|
    if inF == .tf32 then Ampere.tf32f32.value
    else if f16out then Ampere.f16f16.value else Ampere.f16f32.value
  | "B200" =>
    -- The model implements FP8 `mma.sync` with `Matrix.loweredF8`, not a datapath of its own.
    if fp8 then (if f16out then none else some Blackwell.f16f32.value) else some <|
    if inF == .tf32 then Blackwell.tf32f32.value
    else if f16out then Blackwell.f16f16.value else Blackwell.f16f32.value
  | _ => none

/-- Khattak and Mikaitis's measured H100/H200 outputs: per format, `a_*`/`b_*` hold the inputs as
binary32 hex (`K` per line), `c_*_fp32` the accumulators and `d_*_{fp32,fp16}` the outputs as
binary32 bit strings. Their FP8 outputs into binary32 are compared with the `wgmma` datapath with
the accumulator ignored (as `wgmma` with `scale-d = 0`); their FP8 → binary16 files are not read. -/
def tensorData (dir : String) : IO Bool := do
  let mut ok := true
  for (gpu, n) in [("H100", 16), ("H200", 16), ("A100", 8), ("B200", 16)] do
    for (fmt, inF, k, outs) in [("fp16", Format.f16, n, ["fp32", "fp16"]), ("bf16", .bf16, n, ["fp32"]),
        ("tf32", .tf32, 4, ["fp32"]), ("E4M3", .e4m3, 32, ["fp32"]), ("E5M2", .e5m2, 32, ["fp32"])] do
      let base := s!"{dir}/{gpu}/{fmt}"
      if !(← System.FilePath.pathExists s!"{base}/a_{gpu}_{fmt}.txt") then continue
      let rows (p : String) : IO (Array (Array Nat)) := do
        let text ← IO.FS.readFile p
        return (text.splitOn "\n").toArray.filterMap fun l =>
          let ws := words l
          if ws.isEmpty then none else some (ws.toArray.map parseHex)
      let bits (p : String) : IO (Array Nat) := do
        return ((← IO.FS.readFile p).splitOn "\n").toArray.filterMap fun l =>
          match words l with
          | [w] => some (w.foldl (fun n c => 2 * n + (if c == '1' then 1 else 0)) 0)
          | _ => none
      let a ← rows s!"{base}/a_{gpu}_{fmt}.txt"
      let b ← rows s!"{base}/b_{gpu}_{fmt}.txt"
      let cs ← bits s!"{base}/c_{gpu}_fp32.txt"
      for out in outs do
        let path := s!"{base}/d_{gpu}_{out}.txt"
        if !(← System.FilePath.pathExists path) then continue
        let d ← bits path
        let outF : Format := if out == "fp16" then .f16 else .f32
        let fp8 := inF == .e4m3 || inF == .e5m2
        let some model := modelFor gpu inF outF | continue
        let mut t : Tally := { name := s!"tensor data {gpu} {fmt} → {out}" }
        let known : List Nat := []
        for i in [0:d.size] do
          -- inputs are binary32 encodings of the narrow values
          let conv (x : Nat) : Nat := cvt .f32 inF .rn {} x
          let ai := (a[i]!.extract 0 k).map conv
          let bi := (b[i]!.extract 0 k).map conv
          -- accumulators: binary32, rounded to binary16 for a binary16 output; ignored by
          -- Hopper's FP8 runs into binary32 (see `Hopper.f8f32`)
          let c := if out == "fp16" then cvt .f32 .f16 .rn {} cs[i]!
            else if fp8 && gpu.startsWith "H" then 0 else cs[i]!
          -- outputs are recorded as binary32, a binary16 result widened exactly
          -- The B200's FP8 outputs compared with `Matrix.loweredF8`.
          let ty : NV.PTX.Ty := if inF == .e4m3 then .e4m3 else .e5m2
          let got := if gpu == "B200" && fp8 then
              match NV.PTX.Sem.Matrix.loweredF8 100 ty ty .f32 1 1 k ai bi #[c] with
              | .ok ds => ds[0]!
              | .error _ => Format.f32.nan + 1
            else cvt outF .f32 .rn {} (model.block inF inF outF ai bi c)
          let want := d[i]!
          let agrees := got == want || (Format.f32.isNaN got && Format.f32.isNaN want)
          t := t.check (agrees != known.contains i) fun _ =>
            if known.contains i then s!"row {i}: now agrees; drop it from the known misses"
            else s!"row {i}: {hex got}, want {hex want}"
        ok := (← t.report) && ok
  return ok

/-- SF Tensor's model of the B200's `tcgen05.mma` datapaths (`sf-tensor`) on the cases
`tools/num/sftensor_vectors.py` drew (`tools/num/sftensor-data.sh`): each line's family, K, A, B,
scale factors, accumulator and the model's result, against `Blackwell`'s and `Scaled`'s. -/
def sfTensor (file : String) : IO Bool := do
  let fmt : String → Option Format
    | "e4m3" => some .e4m3 | "e5m2" => some .e5m2 | "e2m3" => some .e2m3
    | "e3m2" => some .e3m2 | "e2m1" => some .e2m1 | _ => none
  let mut tallies : Std.HashMap String Tally := {}
  let mut order : Array String := #[]
  for line in (← IO.FS.readFile file).splitOn "\n" do
    let name :: kk :: rest := words line | continue
    let k := kk.toNat!
    let ws := rest.toArray.map parseHex
    let (a, b) := (ws.extract 0 k, ws.extract k (2 * k))
    let c := ws[ws.size - 2]!
    let want := ws[ws.size - 1]!
    let nsc := (ws.size - 2 - 2 * k) / 2
    let (sa, sb) := (ws.extract (2 * k) (2 * k + nsc), ws.extract (2 * k + nsc) (2 * k + 2 * nsc))
    let fmts := (name.splitOn ".").filterMap fmt
    let got : Option Nat := match name, fmts with
      | "f16", _ => some (Blackwell.f16f32.value.block .f16 .f16 .f32 a b c)
      | "bf16", _ => some (Blackwell.f16f32.value.block .bf16 .bf16 .f32 (a.map (· >>> 16)) (b.map (· >>> 16)) c)
      | "tf32", _ => some (Blackwell.tf32f32.value.block .tf32 .tf32 .f32 a b c)
      | "mxf4", _ | "nvf4.2x", _ | "nvf4.4x", _ | "nvfp4", _ =>
        let sF := if name == "nvfp4" then Format.ue4m3 else .ue8m0
        let kd : Scaled.Kind := { scaleBlock := k / nsc, nvfp4 := name == "nvfp4" }
        some (Scaled.dot kd sF (a.map Format.e2m1.decode) (b.map Format.e2m1.decode) (sa.map sF.decode)
          (sb.map sF.decode) c)
      | _, [aF, bF] =>
        if name.startsWith "mx." then
          let fa : BlockFormat := { elem := aF, scale := .ue8m0, size := 32 }
          let fb : BlockFormat := { elem := bF, scale := .ue8m0, size := 32 }
          some (Blackwell.f8f32.value.dotVals aF.man bF.man .f32
            ((Array.range k).map fun i => fa.value sa[i / 32]! a[i]!)
            ((Array.range k).map fun i => fb.value sb[i / 32]! b[i]!) c)
        else some (Blackwell.f8f32.value.block aF bF .f32 a b c)
      | _, _ => none
    let some got := got | continue
    let family := if name.startsWith "mx." then "mxf8f6f4" else if fmts.length == 2 then "f8f6f4" else name
    unless tallies.contains family do order := order.push family
    let t := (tallies.getD family { name := s!"sf-tensor {family}" }).check (got == want) fun _ =>
      s!"{name} c {hex c}: {hex got}, want {hex want}"
    tallies := tallies.insert family t
  let mut ok := !order.isEmpty
  for f in order do ok := (← (tallies.getD f {name := f}).report) && ok
  return ok

/-- `tools/num/exhaustive.sh`'s counterpart of `tools/num/exhaustive_ref.c`: for one shard of the
first operand and every second operand, an FNV-1a hash of each operation's results per mode. -/
def exhaustive (fmt : String) (shard shards : Nat) : IO Unit := do
  let f : Format := if fmt == "bf16" then .bf16 else .f16
  let per := 2 ^ 16 / shards
  let ops : List (String × (Rounding → Nat → Nat → Nat)) := [("add", fun r a b => add f r false a b),
    ("sub", fun r a b => sub f r false a b), ("mul", fun r a b => mul f r false a b),
    ("div", fun r a b => div f r false a b)]
  for (name, op) in ops do
    for (mn, r) in [("rn", Rounding.rn), ("rz", .rz), ("rm", .rm), ("rp", .rp)] do
      let mut h : UInt64 := 0xcbf29ce484222325
      for a in [shard * per:(shard + 1) * per] do
        for b in [0:2 ^ 16] do
          h := (h ^^^ (op r a b).toUInt64) * 0x100000001b3
      let hs := String.ofList (Nat.toDigits 16 h.toNat)
      IO.println s!"{fmt} {name} {mn} {String.ofList (List.replicate (16 - hs.length) '0')}{hs}"

/-- `NV.Num.Mufu`'s outputs for the inputs of `tools/num/mufu_ref.c`, in its format. -/
def mufu : IO Unit := do
  let out ← IO.getStdout
  let fns : List (String × (Nat → Nat)) := [("rcp", Mufu.rcp), ("sqrt", Mufu.sqrt),
    ("rsqrt", Mufu.rsqrt), ("lg2", Mufu.lg2), ("ex2", Mufu.ex2 false), ("sin", Mufu.sin),
    ("cos", Mufu.cos)]
  let h (n : Nat) : String := String.ofList (Nat.toDigits 16 n)
  for (name, fn) in fns do
    for i in [0:65536] do
      for x in [i * 65537 % 2 ^ 32, 0x3f800000 + i, 0x3f000000 + i * 128] do
        out.putStrLn s!"{name} {h x} {h (fn x)}"

/-- Nanoseconds per operation of the hot paths the simulator uses, on random operands. -/
def bench : IO Unit := do
  let n := 200000
  let mut g : Rng := ⟨3⟩
  let mut xs : Array Nat := #[]
  -- values of magnitude 2^-10 to 2^10 with random sign and significand, as in a GEMM
  for _ in [0:n + 64] do
    let (x, g') := g.bits 32
    g := g'
    xs := xs.push ((x % 2 ^ 23) + (117 + x / 2 ^ 23 % 21) * 2 ^ 23 + (x / 2 ^ 31) * 2 ^ 31)
  let run (k : Nat → Nat) : IO (Nat × Nat) := do
    let t0 ← IO.monoNanosNow
    let mut acc := 0
    for i in [0:n] do acc := acc + k i
    return ((← IO.monoNanosNow) - t0, acc)
  -- the loop's own cost, subtracted below
  let (base, _) ← run fun i => i
  let time (name : String) (ops : Nat) (k : Nat → Nat) : IO Unit := do
    let (t, acc) ← run k
    let tenths := (t - Min.min t base) * 10 / (n * ops)
    IO.println s!"{name}: {tenths / 10}.{tenths % 10} ns per op (checksum {acc % 1000})"
  let h (x : Nat) : Nat := x / 2 ^ 16
  let e (x : Nat) : Nat := cvt .f32 .e4m3 .rn { satfinite := true } x
  time "fma.rn.f32" 1 fun i => fma .f32 .rn false xs[i]! xs[i + 1]! xs[i + 2]!
  time "add.rn.bf16" 1 fun i => add .bf16 .rn false (h xs[i]!) (h xs[i + 1]!)
  time "cvt.rn.bf16.f32" 1 fun i => cvt .f32 .bf16 .rn {} xs[i]!
  time "ex2.approx.f32" 1 fun i => Approx.ex2 .f32 false xs[i]!
  time "MUFU.EX2 on words" 1 fun i => (MufuWord.ex2 xs[i]!.toUInt32).toNat
  time "MUFU.RCP on words" 1 fun i => (MufuWord.rcp xs[i]!.toUInt32).toNat
  let m16 := Hopper.f16f32.value
  time "HGMMA bf16 k16 (per MAC)" 16 fun i =>
    m16.block .bf16 .bf16 .f32 ((xs.extract i (i + 16)).map h) ((xs.extract (i + 16) (i + 32)).map h) xs[i]!
  let va := (xs.extract 0 16).map (Format.bf16.decode ∘ h)
  time "HGMMA bf16 k16, decoded operands (per MAC)" 16 fun i =>
    m16.blockVals 7 7 .f32 va ((xs.extract (i + 16) (i + 32)).map (Format.bf16.decode ∘ h)) xs[i]!
  let vb := (xs.extract 16 32).map (Format.bf16.decode ∘ h)
  time "HGMMA bf16 k16, both decoded (per MAC)" 16 fun i => m16.blockVals 7 7 .f32 va vb xs[i]!
  let oa := (xs.extract 0 16).map (TensorModel.operand .bf16 ∘ h)
  let ob := (xs.extract 16 32).map (TensorModel.operand .bf16 ∘ h)
  let dp := m16.prepare 7 7 .f32
  let cs := xs.map (·.toUInt64)
  time "HGMMA bf16 k16, fast datapath (per MAC)" 16 fun i => (dp.block oa ob cs[i]!).toNat
  time "fast, empty block, c = 0 (per call)" 1 fun i => (dp.block #[] #[] (i % 2).toUInt64).toNat
  time "fast, empty block (per call)" 1 fun i => (dp.block #[] #[] cs[i]!).toNat
  time "fast, 16 products, c = 0 (per call)" 1 fun _ => (dp.block oa ob 0).toNat
  -- a warp's tile: 16 rows of A by 8 columns of B, K = 512
  let ta := (Array.range 16).map fun r => (xs.extract (r * 512 % n) (r * 512 % n + 512)).map h
  let tb := (Array.range 8).map fun c => (xs.extract (c * 777 % n) (c * 777 % n + 512)).map h
  let tc := (xs.extract 0 128)
  let t0 ← IO.monoNanosNow
  let out := m16.tile .bf16 .bf16 .f32 ta tb tc
  let t1 ← IO.monoNanosNow
  IO.println s!"tile 16×8×512 bf16: {(t1 - t0) / (16 * 8 * 512)} ns per MAC (checksum {out.foldl (· + ·) 0 % 1000})"
  let m8 := Hopper.f8f32.value
  let e8 := (xs.extract 0 64).map (TensorModel.operand .e4m3 ∘ e)
  let dp8 := m8.prepare 3 3 .f32
  let (e8a, e8b) := (e8.extract 0 32, e8.extract 32 64)
  time "QGMMA e4m3 k32, fast datapath (per MAC)" 32 fun i => (dp8.block e8a e8b cs[i]!).toNat
  time "QGMMA e4m3 k32 (per MAC)" 32 fun i =>
    m8.block .e4m3 .e4m3 .f32 ((xs.extract i (i + 32)).map e) ((xs.extract (i + 32) (i + 64)).map e) 0

/-- Does the radians model equal the turns model after `FMUL.RZ` by `1/2π`
(`sin.approx.f32` on `sm_70`+)? -/
def turnsCheck : IO Unit := do
  let c : Nat := 0x3e22f983
  let mut bad := 0
  let mut shown := 0
  for i in [0:2 ^ 22] do
    let x := i * 1021 % 2 ^ 32
    let u := mul .f32 .rz false x c
    for (name, viaRad, viaTurns) in [("sin", Mufu.sin x, Mufu.sinTurns u),
        ("cos", Mufu.cos x, Mufu.cosTurns u)] do
      if viaRad != viaTurns then
        bad := bad + 1
        if shown < 10 then
          shown := shown + 1
          IO.println s!"{name} {hex x}: radians {hex viaRad}, turns {hex viaTurns} (u {hex u})"
  IO.println s!"turns vs radians: {bad} differences in {2 ^ 23}"

def main (args : List String) : IO UInt32 := do
  match args with
  | ["turns"] => turnsCheck; return 0
  | ["bench"] => bench; return 0
  | ["mufu"] => mufu; return 0
  | ["tensor", dir] => return if ← tensorData dir then 0 else 1
  | ["sftensor", file] => return if ← sfTensor file then 0 else 1
  | ["exhaustive", fmt, s, n] => exhaustive fmt s.toNat! n.toNat!; return 0
  | _ =>
    let scale := match args with | ["--scale", n] => n.toNat! | _ => 1
    let vectors ← IO.FS.readFile "tools/num/approx-vectors.txt"
    let suites := [decodeAll, known, roundTrips, narrowing, boundaries, arith16, arithWide,
      f32Words, mufuWords, nvidiaRules, integers, decimals, tensor, fastTensor, blocks, approx (vectors.splitOn "\n")]
    let mut ok := true
    for s in suites do
      let t0 ← IO.monoMsNow
      let t := s scale
      ok := (← t.report) && ok
      IO.println s!"       {(← IO.monoMsNow) - t0} ms"
    return if ok then 0 else 1
