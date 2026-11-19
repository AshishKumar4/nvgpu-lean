import NV.Num.Arith
import NV.Fact

/-!
# Tensor-core accumulation

PTX leaves the tensor cores' accumulation order, rounding and subnormal handling unspecified
(ISA 9.5 §9.7.16.5.14, §9.7.17.5.2). The model here is Khattak and Mikaitis's (`khattak2025`, §4.1.6 and
Table 3):

1. Each product `a·b` is exact, kept unnormalized: significand in `[0, 4)` at the sum of the
   operands' exponents (subnormal operands at the format's minimum exponent).
2. The products of a block and the accumulator `c` are aligned to the largest exponent among them
   (a zero `c` takes no part), but never below `minAlignExp`; each aligned significand keeps
   `alignFrac` fraction bits and is truncated.
3. The aligned terms are summed exactly and the sum is rounded once into the output: truncated
   into binary32 (to `outFrac` fraction bits), nearest-even into binary16. Overflow is infinite.

A dot product longer than a block chains blocks, each block's result the next one's `c`, as the
hardware chains instructions through the accumulator registers.

Everything is integer arithmetic on significands. `blockVals` is the model; `Datapath.block` is
the same computation on machine words, the simulator's fast path for MACs, and
`tile` runs a warp's tile of dot products on it.
-/

namespace NV.Num

/-- The numerical features of one tensor-core datapath (`khattak2025` Table 3). -/
structure TensorModel where
  /-- Products summed in one fused step together with the accumulator (N_FMA). -/
  blockSize : Nat
  /-- Fraction bits each aligned term keeps: 23 + n_eab. -/
  alignFrac : Nat
  /-- The smallest exponent the alignment uses (e_min^lim). -/
  minAlignExp : Int
  /-- Fraction bits of the result. -/
  outFrac : Nat
  /-- Rounding of the sum into the result. -/
  outRound : Rounding
  /-- A nonzero sum the rounding takes to zero gives `+0` (else the sum's sign). -/
  posZero : Bool := false
  deriving DecidableEq, Repr, Inhabited

namespace TensorModel

/-- `m · 2^k`, truncated when `k < 0`. -/
private def scale (m : Nat) (k : Int) : Nat :=
  if k ≥ 0 then m * pow2 k.toNat else m / pow2 (-k).toNat

/-- The infinite product's sign, if `x · y` is infinite. -/
private def infSign : Val → Val → Option Bool
  | .inf n1, .inf n2 => some (n1 != n2)
  | .inf n1, .fin n2 m _ | .fin n2 m _, .inf n1 => if m = 0 then none else some (n1 != n2)
  | _, _ => none

private def invalid : Val → Val → Bool
  | .nan, _ | _, .nan => true
  | .inf _, .fin _ m _ | .fin _ m _, .inf _ => m = 0
  | _, _ => false

/-- The output format at the result's precision: `outF` with the low fraction bits fixed at 0. -/
def outFormat (t : TensorModel) (outF : Format) : Format :=
  { outF with man := t.outFrac, pad := outF.pad + (outF.man - t.outFrac) }

/-- One block over decoded operands (`Format.decode`): `Σ a[i]·b[i] + c`, the operands' formats
having `aMan` and `bMan` fraction bits and `c` encoded in `outF`. An operand used in many blocks
(a row of A against every column of B) is decoded once. -/
def blockVals (t : TensorModel) (aMan bMan : Nat) (outF : Format) (a b : Array Val) (c : Nat) :
    Nat := Id.run do
  let vc := outF.decode c
  let n := Min.min a.size b.size
  let shift : Int := (t.alignFrac : Int) - aMan - bMan
  -- First pass: special values, and the largest exponent.
  let mut nan := vc == .nan
  let mut posInf := false
  let mut negInf := false
  let mut any := false
  let mut emax := t.minAlignExp
  match vc with
  | .inf neg => if neg then negInf := true else posInf := true
  | _ => pure ()
  for i in [0:n] do
    match a[i]!, b[i]! with
    | .fin _ m1 e1, .fin _ m2 e2 =>
      if m1 != 0 && m2 != 0 then
        any := true
        emax := Max.max emax (e1 + e2 + aMan + bMan)
    | x, y =>
      if invalid x y then nan := true
      else match infSign x y with
        | some true => negInf := true
        | some false => posInf := true
        | none => pure ()
  if nan || (posInf && negInf) then return outF.nan
  if posInf then return outF.inf false
  if negInf then return outF.inf true
  -- The accumulator as binary32 sees it: exponent at least -126, 23 fraction bits.
  let cterm : Option (Bool × Nat × Int) := match vc with
    | .fin neg m e =>
      if m == 0 then none
      else
        let ce := Max.max (e + Nat.log2 m) (-126)
        some (neg, scale m (e - ce + t.alignFrac), ce)
    | _ => none
  if !any && cterm.isNone then return outF.zero false
  if let some (_, _, ce) := cterm then emax := Max.max emax ce
  -- Second pass: the aligned, truncated terms summed exactly.
  let mut s : Int := 0
  for i in [0:n] do
    match a[i]!, b[i]! with
    | .fin n1 m1 e1, .fin n2 m2 e2 =>
      if m1 != 0 && m2 != 0 then
        let v : Int := scale (m1 * m2) shift >>> (emax - (e1 + e2 + aMan + bMan)).toNat
        s := if n1 != n2 then s - v else s + v
    | _, _ => pure ()
  if let some (neg, m, ce) := cterm then
    let v : Int := m >>> (emax - ce).toNat
    s := if neg then s - v else s + v
  if s == 0 then return outF.zero false
  let r := (t.outFormat outF).round t.outRound .inf (s < 0) s.natAbs (emax - t.alignFrac)
  return if t.posZero && r == outF.zero true then outF.zero false else r

/-- One block: `Σ a[i]·b[i] + c`, with `a` encoded in `aF`, `b` in `bF` (FP8 operands may mix
E4M3 and E5M2) and `c` in `outF`. -/
def block (t : TensorModel) (aF bF outF : Format) (a b : Array Nat) (c : Nat) : Nat :=
  t.blockVals aF.man bF.man outF (a.map aF.decode) (b.map bF.decode) c

/-! ### The fast datapath

The simulator's MACs: `Datapath.block` computes `blockVals` with machine words, on operands decoded
once (`operand`) and a datapath prepared once per instruction (`prepare`), in loops with unboxed
state and no allocation. Equal to `blockVals` on every single product of the 8-bit formats with
accumulators of each class, and on random blocks of every Hopper datapath with special and
subnormal values mixed in (`num-tests`'s fast-path suite); not proved. -/

/-- An operand for `Datapath.block`: significand in bits 0–11, units exponent + 512 in bits 12–21,
sign in bit 22, and in bits 23–24 its class: 0 nonzero finite, 1 infinity, 2 NaN, 3 zero. For formats with at most 11 fraction bits
and exponents within ±511: f16, bf16, tf32 and the 4-, 6- and 8-bit formats. -/
def operand (f : Format) (x : Nat) : UInt32 :=
  let w := x.toUInt64 >>> f.pad.toUInt64
  let man := f.man.toUInt64
  let ex := f.exp.toUInt64
  let frac := w &&& ((1 <<< man) - 1)
  let ef := (w >>> man) &&& ((1 <<< ex) - 1)
  let sign : UInt32 := if f.signed && (w >>> (man + ex)) &&& 1 == 1 then 0x400000 else 0
  let top := ef == (1 <<< ex) - 1
  if top && f.special == .ieee then (if frac == 0 then 0x800000 ||| sign else 0x1000000)
  else if top && f.special == .nanOnly && frac == (1 <<< man) - 1 then 0x1000000
  else if ef == 0 && f.subnormals then
    -- subnormals at the smallest exponent, 1 - bias
    if frac == 0 then 0x1800000 ||| sign
    else sign ||| ((513 - f.bias.toUInt64).toUInt32 <<< 12) ||| frac.toUInt32
  else sign ||| ((ef + 512 - f.bias.toUInt64).toUInt32 <<< 12) ||| (frac ||| (1 <<< man)).toUInt32

/-- `operand` for one format with its parameters as machine words, prepared once per instruction:
the same encoding without arithmetic on `Nat` per element. -/
structure Codec where
  pad : UInt64
  man : UInt64
  ex : UInt64
  bias : UInt64
  signed : Bool
  ieee : Bool
  nanOnly : Bool
  subnormals : Bool

def Codec.of (f : Format) : Codec :=
  { pad := f.pad.toUInt64, man := f.man.toUInt64, ex := f.exp.toUInt64, bias := f.bias.toUInt64,
    signed := f.signed, ieee := f.special == .ieee, nanOnly := f.special == .nanOnly,
    subnormals := f.subnormals }

/-- `operand f x` (`Codec.of f`). -/
@[inline] def Codec.encode (c : Codec) (x : UInt32) : UInt32 :=
  let w := x.toUInt64 >>> c.pad
  let frac := w &&& ((1 <<< c.man) - 1)
  let ef := (w >>> c.man) &&& ((1 <<< c.ex) - 1)
  let sign : UInt32 := if c.signed && (w >>> (c.man + c.ex)) &&& 1 == 1 then 0x400000 else 0
  let top := ef == (1 <<< c.ex) - 1
  if top && c.ieee then (if frac == 0 then 0x800000 ||| sign else 0x1000000)
  else if top && c.nanOnly && frac == (1 <<< c.man) - 1 then 0x1000000
  else if ef == 0 && c.subnormals then
    if frac == 0 then 0x1800000 ||| sign
    else sign ||| ((513 - c.bias).toUInt32 <<< 12) ||| frac.toUInt32
  else sign ||| ((ef + 512 - c.bias).toUInt32 <<< 12) ||| (frac ||| (1 <<< c.man)).toUInt32

/-- An `Int` known to be small, as a word (`Int64.ofInt` goes through a bignum modulus). -/
private def int64 (i : Int) : Int64 :=
  if i ≥ 0 then i.toNat.toInt64 else -(-i).toNat.toInt64

/-- A datapath with its input and output formats, every parameter a machine word, for the
datapaths `TensorModel.fast` accepts. -/
structure Datapath where
  shl : Bool
  sh : UInt64
  alignFrac : Int64
  minAlignExp : Int64
  rn : Bool
  /-- The accumulator's format. -/
  cMan : UInt64
  cExp : UInt64
  cPad : UInt64
  cBias : Int64
  cQMin : Int64
  cExpMax : UInt64
  /-- The result's precision: `outFormat`. -/
  oMan : UInt64
  oPad : UInt64
  oQMin : Int64
  signBit : UInt64
  nan : UInt64
  posZero : Bool
  deriving Inhabited

def prepare (t : TensorModel) (aMan bMan : Nat) (outF : Format) : Datapath :=
  let o := t.outFormat outF
  let sl : Int := (t.alignFrac : Int) - aMan - bMan
  { shl := sl ≥ 0, sh := sl.natAbs.toUInt64, alignFrac := t.alignFrac.toInt64,
    minAlignExp := int64 t.minAlignExp, rn := t.outRound == .rn,
    cMan := outF.man.toUInt64, cExp := outF.exp.toUInt64, cPad := outF.pad.toUInt64,
    cBias := outF.bias.toInt64, cQMin := int64 outF.qMin, cExpMax := outF.expMax.toUInt64,
    oMan := o.man.toUInt64, oPad := o.pad.toUInt64, oQMin := int64 o.qMin,
    signBit := outF.signBit.toUInt64, nan := outF.nan.toUInt64, posZero := t.posZero }

namespace Datapath

private def sig (x : UInt32) : UInt64 := (x &&& 0xfff).toUInt64

/-- The sum of two operands' exponent fields: their product's units exponent + 1024. -/
private def expSum (x y : UInt32) : UInt32 :=
  ((x >>> (12 : UInt32)) &&& (0x3ff : UInt32)) + ((y >>> (12 : UInt32)) &&& (0x3ff : UInt32))

/-- The first pass over `[i, n)`: the largest `expSum` of a nonzero product, and flags (1 NaN,
2 `+∞`, 4 `-∞`, 8 some nonzero product), packed as `emax · 2^8 + flags`. -/
private def scan (a b : Array UInt32) (n i : Nat) (ha : n ≤ a.size) (hb : n ≤ b.size)
    (emax : UInt32) (flags : UInt32) : UInt32 :=
  if h : i < n then
    let x := a[i]'(by omega)
    let y := b[i]'(by omega)
    if (x ||| y) >>> 23 == 0 then
      let e := expSum x y
      scan a b n (i + 1) ha hb (if e > emax then e else emax) (flags ||| 8)
    else
      let cx := x >>> 23
      let cy := y >>> 23
      -- a zero times a finite value is no term; times infinity, or any NaN, is NaN
      if cx == 2 || cy == 2 || (cx == 3 && cy == 1) || (cx == 1 && cy == 3) then
        scan a b n (i + 1) ha hb emax (flags ||| 1)
      else if cx == 3 || cy == 3 then scan a b n (i + 1) ha hb emax flags
      else if ((x ^^^ y) >>> 22) &&& 1 == 1 then scan a b n (i + 1) ha hb emax (flags ||| 4)
      else scan a b n (i + 1) ha hb emax (flags ||| 2)
  else (emax <<< 8) ||| flags
termination_by n - i

/-- The second pass over `[i, n)`: the aligned, truncated products summed. -/
private def sum (shl : Bool) (sh : UInt64) (a b : Array UInt32) (n i : Nat) (ha : n ≤ a.size)
    (hb : n ≤ b.size) (emax : UInt32) (s : Int64) : Int64 :=
  if h : i < n then
    let x := a[i]'(by omega)
    let y := b[i]'(by omega)
    if (x ||| y) >>> 23 == 0 then
      let p := if shl then (sig x * sig y) <<< sh else (sig x * sig y) >>> sh
      let d := emax - expSum x y
      let v : Int64 := if d ≥ 64 then 0 else (p >>> d.toUInt64).toInt64
      sum shl sh a b n (i + 1) ha hb emax (if ((x ^^^ y) >>> 22) &&& 1 == 1 then s - v else s + v)
    else sum shl sh a b n (i + 1) ha hb emax s
  else s
termination_by n - i

/-- The position of the highest set bit of a nonzero word. -/
private def log2Word (x : UInt64) : Int64 :=
  let s5 : UInt64 := if x >>> 32 != 0 then 32 else 0
  let x := x >>> s5
  let s4 : UInt64 := if x >>> 16 != 0 then 16 else 0
  let x := x >>> s4
  let s3 : UInt64 := if x >>> 8 != 0 then 8 else 0
  let x := x >>> s3
  let s2 : UInt64 := if x >>> 4 != 0 then 4 else 0
  let x := x >>> s2
  let s1 : UInt64 := if x >>> 2 != 0 then 2 else 0
  let x := x >>> s1
  let s0 : UInt64 := if x >>> 1 != 0 then 1 else 0
  (s5 + s4 + s3 + s2 + s1 + s0).toInt64

/-- `Format.round` into the result's precision on words: `±M · 2^E` (`0 < M < 2^62`), toward zero
or to nearest even, overflowing to infinity. -/
private def round (d : Datapath) (neg : Bool) (M : UInt64) (E : Int64) : UInt64 :=
  let one : UInt64 := 1
  let sign := if neg then d.signBit else 0
  let man := d.oMan
  let q0 := E + log2Word M - man.toInt64
  let q := if q0 < d.oQMin then d.oQMin else q0
  let k := q - E
  let t : UInt64 :=
    if k ≤ 0 then M <<< (-k).toUInt64
    else if k ≥ 64 then 0
    else
      let t := M >>> k.toUInt64
      let rem := M &&& ((one <<< k.toUInt64) - 1)
      let half := one <<< (k - 1).toUInt64
      if d.rn && (rem > half || (rem == half && t &&& 1 == 1)) then t + 1 else t
  let carry := t == one <<< (man + 1)
  let t := if carry then one <<< man else t
  let q := if carry then q + 1 else q
  if t == 0 then sign
  else
    let ef : UInt64 := if t < one <<< man then 0 else (q - d.oQMin + 1).toUInt64
    let ef := if ef ≥ d.cExpMax then d.cExpMax else ef
    let frac := if ef == d.cExpMax then 0 else t &&& ((one <<< man) - 1)
    sign ||| (((ef <<< man) ||| frac) <<< d.oPad)

/-- `Σ a[i]·b[i] + c` with `blockVals`'s result, on `operand`s and the accumulator's encoding. -/
def blockAt (d : Datapath) (a b : Array UInt32) (start stop : Nat) (c : UInt64) : UInt64 :=
  let n := Min.min stop (Min.min a.size b.size)
  have ha : n ≤ a.size := Nat.le_trans (Nat.min_le_right _ _) (Nat.min_le_left _ _)
  have hb : n ≤ b.size := Nat.le_trans (Nat.min_le_right _ _) (Nat.min_le_right _ _)
  let r := scan a b n start ha hb (d.minAlignExp + 1024).toUInt64.toUInt32 0
  -- the accumulator, as binary32 sees it: exponent at least -126, 23 fraction bits
  let w := c >>> d.cPad
  let frac := w &&& ((1 <<< d.cMan) - 1)
  let ef := (w >>> d.cMan) &&& ((1 <<< d.cExp) - 1)
  let cneg := (w >>> (d.cMan + d.cExp)) &&& 1 == 1
  let cspecial := ef == d.cExpMax
  let cnonzero := !cspecial && (ef != 0 || frac != 0)
  let flags := (r &&& 0xff) |||
    (if cspecial then (if frac != 0 then 1 else if cneg then 4 else 2) else if cnonzero then 8 else 0)
  if flags &&& 1 != 0 || flags &&& 6 == 6 then d.nan
  else if flags &&& 2 != 0 then d.cExpMax <<< (d.cMan + d.cPad)
  else if flags &&& 4 != 0 then d.signBit ||| (d.cExpMax <<< (d.cMan + d.cPad))
  else if flags &&& 8 == 0 then 0
  else
    let m : UInt64 := if ef == 0 then frac else frac ||| (1 <<< d.cMan)
    let e : Int64 := if ef == 0 then d.cQMin else ef.toInt64 - d.cBias - d.cMan.toInt64
    let ce0 := e + log2Word m
    let ce : Int64 := if !cnonzero then d.minAlignExp else if ce0 < -126 then -126 else ce0
    let k := e - ce + d.alignFrac
    let cm : UInt64 := if !cnonzero then 0 else if k ≥ 0 then m <<< k.toUInt64 else m >>> (-k).toUInt64
    -- exponents as field sums (units exponent + 1024)
    let cf := (ce + 1024).toUInt64.toUInt32
    let emax := if cf > r >>> 8 then cf else r >>> 8
    let s := sum d.shl d.sh a b n start ha hb emax 0
    let dc := emax - cf
    let cv : Int64 := if dc ≥ 64 then 0 else (cm >>> dc.toUInt64).toInt64
    let s := if cneg then s - cv else s + cv
    if s == 0 then 0
    else
      let r := d.round (s < 0) (if s < 0 then (-s).toUInt64 else s.toUInt64)
        (emax.toUInt64.toInt64 - 1024 - d.alignFrac)
      if d.posZero && r == d.signBit then 0 else r

/-- `Σ a[i]·b[i] + c` with `blockVals`'s result, on `operand`s and the accumulator's encoding. -/
def block (d : Datapath) (a b : Array UInt32) (c : UInt64) : UInt64 :=
  d.blockAt a b 0 (Min.min a.size b.size) c

/-- `dot` on words: blocks of `size` products from `start`, chained through the accumulator. -/
def dot (d : Datapath) (size : Nat) (a b : Array UInt32) (c : UInt64) : UInt64 :=
  go 0 c
where
  go (start : Nat) (acc : UInt64) : UInt64 :=
    if start < Min.min a.size b.size && size > 0 then
      go (start + size) (d.blockAt a b start (start + size) acc)
    else acc
  termination_by Min.min a.size b.size - start
  decreasing_by
    simp only [Bool.and_eq_true, decide_eq_true_eq] at *
    omega

end Datapath

/-- `dot` on decoded operands (`blockVals`): `aMan` and `bMan` fraction bits. -/
def dotVals (t : TensorModel) (aMan bMan : Nat) (outF : Format) (a b : Array Val) (c : Nat) : Nat :=
  let n := t.blockSize
  (List.range ((Min.min a.size b.size + n - 1) / n)).foldl (init := c) fun acc i =>
    t.blockVals aMan bMan outF (a.extract (i * n) ((i + 1) * n)) (b.extract (i * n) ((i + 1) * n)) acc

/-- `Σ a[i]·b[i] + c` over any length, one block after another through the accumulator. -/
def dot (t : TensorModel) (aF bF outF : Format) (a b : Array Nat) (c : Nat) : Nat :=
  t.dotVals aF.man bF.man outF (a.map aF.decode) (b.map bF.decode) c

/-- Whether `Datapath.block` covers this datapath: inputs with at most 11 fraction bits and 8
exponent bits, a signed IEEE output with subnormals of at most binary32's size, rounded toward
zero or to nearest even. -/
def fast (t : TensorModel) (aF bF outF : Format) : Bool :=
  aF.man ≤ 11 && bF.man ≤ 11 && aF.exp ≤ 8 && bF.exp ≤ 8 && outF.special == .ieee &&
    outF.subnormals && outF.signed && outF.exp ≤ 8 && outF.man ≤ 23 &&
    (t.outRound == .rz || t.outRound == .rn)

/-- `tile` on raw element bits in words, the fast path only: A's `m` rows and B's `n` columns of
`k` elements each (`a[i·k + kk]`, `b[j·k + kk]`) and the accumulator `c[i·n + j]` (its
encoding), each operand encoded once (`Codec`); `none` where `fast` does not cover the formats.
The one tile computation of the warp-level (`mma.sync`, `HMMA`) and warpgroup (`wgmma`, `HGMMA`)
fast paths. -/
def tileWords (t : TensorModel) (aF bF outF : Format) (m n k : Nat) (a b : Array UInt32)
    (c : Array UInt64) : Option (Array UInt64) :=
  if !t.fast aF bF outF then none else
  let d := t.prepare aF.man bF.man outF
  let ea := a.map (Codec.of aF).encode
  let eb := b.map (Codec.of bF).encode
  let rows := (Array.range m).map fun i => ea.extract (i * k) (i * k + k)
  let cols := (Array.range n).map fun j => eb.extract (j * k) (j * k + k)
  some <| Id.run do
    let mut out := Array.mkEmpty (m * n)
    for i in [0:m] do
      for j in [0:n] do
        out := out.push (d.dot t.blockSize rows[i]! cols[j]! c[i * n + j]!)
    return out

/-- A tile of dot products: `out[i·n + j] = dot a[i] b[j] c[i·n + j]` for the `m` rows of A and
`n` columns of B, row-major. Each row and column is decoded once and the fast path computes every
block (`dot` where it does not apply). Tested equal to `dot` elementwise. -/
def tile (t : TensorModel) (aF bF outF : Format) (a b : Array (Array Nat)) (c : Array Nat) :
    Array Nat := Id.run do
  let n := b.size
  let mut out := Array.mkEmpty (a.size * n)
  if t.fast aF bF outF then
    let d := t.prepare aF.man bF.man outF
    let ra := a.map (·.map (operand aF))
    let rb := b.map (·.map (operand bF))
    for i in [0:a.size] do
      for j in [0:n] do
        out := out.push (d.dot t.blockSize ra[i]! rb[j]! (c[i * n + j]!).toUInt64).toNat
  else
    for i in [0:a.size] do
      for j in [0:n] do
        out := out.push (t.dot aF bF outF a[i]! b[j]! c[i * n + j]!)
  return out

end TensorModel

/-! ## Hopper (`sm_90a`) datapaths

Every value below is `khattak2025`'s, for the H100 and H200. -/

namespace Hopper

private def km (loc : String) : Source := { ref := "khattak2025", loc }

/-- FP16 or BF16 inputs, FP32 accumulator: `mma.sync` (`HMMA.16816.F32`) and `wgmma` (`HGMMA`). -/
def f16f32 : Fact TensorModel :=
  { value :=
      { blockSize := 16, alignFrac := 25, minAlignExp := -133, outFrac := 23, outRound := .rz }
    kind := .published, source := km "§4.1.6, Table 3, Fig. 5a; models/H100TC.m" }

/-- FP16 inputs, FP16 accumulator. -/
def f16f16 : Fact TensorModel :=
  { value :=
      { blockSize := 16, alignFrac := 25, minAlignExp := -21, outFrac := 10, outRound := .rn }
    kind := .published, source := km "Table 3; models/H100TC.m (min_exp_limit -21 for fp16 out)" }

/-- TF32 inputs (the low 13 bits of each `.b32` ignored, PTX §9.7.17.5.2), FP32 accumulator. -/
def tf32f32 : Fact TensorModel :=
  { value :=
      { blockSize := 8, alignFrac := 25, minAlignExp := -133, outFrac := 23, outRound := .rz }
    kind := .published, source := km "§4.1.6, Table 3, Fig. 5b"
    note := "" }

/-- E4M3 or E5M2 inputs through `wgmma` (`QGMMA`), FP32 accumulator: 13 fraction bits kept in
alignment and in the result, so a product sum keeps about 14 significant bits. -/
def f8f32 : Fact TensorModel :=
  { value :=
      { blockSize := 32, alignFrac := 13, minAlignExp := -133, outFrac := 13, outRound := .rz }
    kind := .published, source := km "§4.1.6, Table 3, Fig. 5d; models/H100TC.m (neab -10)"
    note := "" }

/-- E4M3 or E5M2 inputs through `wgmma`, FP16 accumulator. -/
def f8f16 : Fact TensorModel :=
  { value :=
      { blockSize := 32, alignFrac := 13, minAlignExp := -133, outFrac := 10, outRound := .rn }
    kind := .inferred, source := km "Table 3 (fp8 → fp16: alignment (2,13), RNE), Fig. 5d"
    note := "" }

/-- Binary64 (`DMMA`): one product per fused step, so IEEE fused multiply-adds in order,
`((c + p₁) + p₂) + …` (`dotF64`). -/
def f64BlockSize : Fact Nat :=
  { value := 1, kind := .published, source := km "Table 3; §4.1.2 (order measured on A100)" }

end Hopper

/-! ## Ampere (`sm_80`) datapaths

`khattak2025`'s A100 model. Eight products per fused step (four for TF32), one extra alignment bit. Ampere has
no FP8 datapath. -/

namespace Ampere

private def km (loc : String) : Source := { ref := "khattak2025", loc }

def f16f32 : Fact TensorModel :=
  { value :=
      { blockSize := 8, alignFrac := 24, minAlignExp := -132, outFrac := 23, outRound := .rz }
    kind := .published, source := km "Table 3; models/A100TC.m (fma 8, neab 1, min_exp_limit -132)"
    note := "" }

def f16f16 : Fact TensorModel :=
  { value :=
      { blockSize := 8, alignFrac := 24, minAlignExp := -20, outFrac := 10, outRound := .rn }
    kind := .published, source := km "Table 3; models/A100TC.m (fp16 out: rne, min_exp_limit -20)" }

def tf32f32 : Fact TensorModel :=
  { value :=
      { blockSize := 4, alignFrac := 24, minAlignExp := -132, outFrac := 23, outRound := .rz }
    kind := .published, source := km "Table 3; models/A100TC.m (tf32: fma 4)" }

end Ampere

/-! ## Blackwell datapaths (`sm_100a`, and `sm_120a` of the RTX PRO 6000)

`khattak2025`'s B200 model, also taken for the RTX PRO 6000 (`rtxPro6000`, inferred). FP8 is
accumulated at full binary32 alignment,
unlike Hopper's. FP6 and FP4 (`kind::f8f6f4`, `mxf4`) are `f8f32`'s and `NV.Num.Scaled`'s. -/

namespace Blackwell

private def km (loc : String) : Source := { ref := "khattak2025", loc }

/-- The RTX PRO 6000 (`sm_120a`) has the B200's tensor-core datapaths (inferred). -/
def rtxPro6000 : Fact Bool :=
  { value := true, kind := .inferred
    source := km ""
    note := "" }

def f16f32 : Fact TensorModel :=
  { value :=
      { blockSize := 16, alignFrac := 25, minAlignExp := -133, outFrac := 23, outRound := .rz }
    kind := .published, source := km "Table 3; models/B200TC.m, models/RTXPRO6000TC.m" }

def f16f16 : Fact TensorModel :=
  { value :=
      { blockSize := 16, alignFrac := 25, minAlignExp := -21, outFrac := 10, outRound := .rn }
    kind := .published, source := km "Table 3; models/B200TC.m (fp16 out: rne, min_exp_limit -21)" }

def tf32f32 : Fact TensorModel :=
  { value :=
      { blockSize := 8, alignFrac := 25, minAlignExp := -133, outFrac := 23, outRound := .rz }
    kind := .published, source := km "Table 3; models/B200TC.m (tf32: fma 8)" }

/-- FP8, FP6 and FP4 inputs (`tcgen05.mma .kind::f8f6f4`, in any mix, and `.kind::mxf8f6f4`'s
scaled elements, `BlockFormat.value`), FP32 accumulator: the 32 products of an instruction and the
accumulator in one window of 25 fraction bits below the largest raw exponent, each truncated, the
sum truncated into binary32 (`sf-tensor`). A sum truncated to zero is `+0`. -/
def f8f32 : Fact TensorModel :=
  { value :=
      { blockSize := 32, alignFrac := 25, minAlignExp := -133, outFrac := 23, outRound := .rz,
        posZero := true }
    kind := .published
    source := { ref := "sf-tensor", loc := "tcgen05_model/MODEL.md §3, §5 (UE8M0 MXF8F6F4); model.py Tcgen05RawWindowFp8E4M3MmaModel" }
    note := "" }

end Blackwell

/-- Binary64 tensor-core dot product (`Hopper.f64`). -/
def dotF64 (a b : Array Nat) (c : Nat) : Nat :=
  (a.zip b).foldl (fun acc (x, y) => fma .f64 .rn false x y acc) c

/-- Integer tensor-core dot product on `s32` (PTX §9.7.16.5.14): the exact sum, wrapped to 32
bits, or clamped to the `s32` range under `.satfinite` (where the clamp applies is inferred: to
the exact result). -/
def dotInt (a b : Array Int) (c : Int) (satfinite : Bool) : Int :=
  let s := (a.zip b).foldl (fun acc (x, y) => acc + x * y) c
  if satfinite then Max.max (-2 ^ 31) (Min.min (2 ^ 31 - 1) s)
  else let w := s % 2 ^ 32; if w ≥ 2 ^ 31 then w - 2 ^ 32 else w

end NV.Num
