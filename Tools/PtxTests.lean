import NV.PTX.Interp

/-!
`lake exe ptx-tests`: the PTX semantics on nvcc-compiled kernels (`tests/ptx/kernels/*.cu`,
their `sm_90a` PTX committed beside them), run by the interpreter on fixed inputs and compared
word for word with references written here from the CUDA functions' documented meaning, not from
the PTX semantics.
-/

open NV NV.PTX

def base : UInt64 := 0x10_0000

/-- A test: kernel file, entry, launch shape, input words, parameter values, and the check of
the final global memory. -/
structure Case where
  name : String
  file : String
  entry : String
  grid : Nat × Nat × Nat
  block : Nat × Nat × Nat
  /-- Initial global memory: (byte offset from `base`, 32-bit words). -/
  inputs : List (Nat × Array UInt32)
  params : List UInt64
  /-- The words expected at (byte offset, words). -/
  expect : List (Nat × Array UInt32)
  /-- Instead of `expect`, a check of the words read at an offset (count, predicate). -/
  holds : List (Nat × Nat × (Array UInt32 → Option String)) := []
  /-- Expected words where `none` accepts any NaN encoding of an `.f32`. -/
  expectF : List (Nat × Array (Option UInt32)) := []

def u32 (i : Int) : UInt32 := UInt32.ofInt i
def s32 (x : UInt32) : Int := x.toInt32.toInt
def lo64 (v : UInt64) : UInt32 := v.toUInt32
def hi64 (v : UInt64) : UInt32 := (v >>> 32).toUInt32
def wordsOf64 (vs : Array UInt64) : Array UInt32 := vs.flatMap fun v => #[lo64 v, hi64 v]

/-! ## References (CUDA Math API and Programming Guide meanings) -/

def popcount (x : UInt32) : Nat := (List.range 32).filter (fun i => (x >>> i.toUInt32) &&& 1 == 1) |>.length
def clz32 (x : UInt32) : Nat := ((List.range 32).reverse.takeWhile (fun i => (x >>> i.toUInt32) &&& 1 == 0)).length
def brev32 (x : UInt32) : UInt32 :=
  (List.range 32).foldl (fun a i => if (x >>> i.toUInt32) &&& 1 == 1 then a ||| ((1 : UInt32) <<< (31 - i).toUInt32) else a) (0 : UInt32)
def bytePerm (x y s : UInt32) : UInt32 :=
  let v : UInt64 := x.toUInt64 ||| (y.toUInt64 <<< 32)
  (List.range 4).foldl (fun r n =>
    let sel := ((s >>> (4 * n).toUInt32) &&& 7).toUInt64
    r ||| (((v >>> (8 * sel)) &&& 0xFF).toUInt32 <<< (8 * n).toUInt32)) 0
def funnelL (lo hi s : UInt32) : UInt32 :=
  let v := (hi.toNat <<< 32) ||| lo.toNat
  ((v <<< (s.toNat % 32)) >>> 32).toUInt32
def funnelRc (lo hi s : UInt32) : UInt32 :=
  let v := (hi.toNat <<< 32) ||| lo.toNat
  (v >>> (Nat.min s.toNat 32)).toUInt32
def sext24 (x : UInt32) : Int := let v := (x &&& 0xFFFFFF).toNat; if v ≥ 2 ^ 23 then v - 2 ^ 24 else v
def ffs (x : UInt32) : UInt32 :=
  match (List.range 32).find? (fun i => (x >>> i.toUInt32) &&& 1 == 1) with
  | some i => (i + 1).toUInt32 | none => 0

/-- `intops` thread `i` on operands `x`, `y`, `z`: the 24 results of intops.cu. -/
def intopsRef (x y z : UInt32) : Array UInt32 :=
  let sx := s32 x
  let sy := s32 y
  let defined := sy != 0 && !(sx == -2147483648 && sy == -1)
  #[ ((x.toNat * y.toNat) >>> 32).toUInt32,
     u32 ((sx * sy).fdiv (2 ^ 32)),
     if y == 0 then 0 else x / y,
     if y == 0 then 0 else x % y,
     if defined then u32 (sx.tdiv sy) else 0,
     if defined then u32 (sx.tmod sy) else 0,
     (popcount x).toUInt32, (clz32 x).toUInt32, brev32 x, bytePerm x y z,
     funnelL x y z, funnelRc x y z,
     u32 (max sx sy), min x y, u32 sx.natAbs, u32 ((sx - sy).natAbs + (z.toNat : Int)),
     u32 ((List.range 4).foldl (fun acc k =>
        let b (v : UInt32) : Int := let w := ((v >>> (8 * k).toUInt32) &&& 0xFF).toNat
          if w ≥ 128 then w - 256 else w
        acc + b x * b y) (s32 z)),
     u32 (sext24 x * sext24 y),
     x * y + z,
     (x >>> (y &&& 31)) ||| (x <<< ((32 - (y &&& 31)) &&& 31)),
     u32 (sx >>> (y &&& 31).toNat),
     ffs x,
     ((x.toNat * y.toNat) >>> 7).toUInt32,
     (List.range 4).foldl (fun r k =>
        let a := ((x >>> (8 * k).toUInt32) &&& 0xFF).toNat
        let b := ((y >>> (8 * k).toUInt32) &&& 0xFF).toNat
        r ||| ((if a ≥ b then a - b else b - a).toUInt32 <<< (8 * k).toUInt32)) 0 ]

def intInputs : Array (UInt32 × UInt32 × UInt32) :=
  let edge : List UInt32 := [0, 1, 2, 7, 31, 32, 33, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF,
    0xFFFFFFFE, 0x12345678, 0x9ABCDEF0, 0x00FF00FF, 0x80808080, 0x0F0F0F0F]
  (Array.range 48).map fun i =>
    let x := edge[i % 16]!
    let y := edge[(i * 7 + 3) % 16]! + (i / 16).toUInt32
    let z := edge[(i * 5 + 1) % 16]! ^^^ (i.toUInt32 * 0x01010101)
    (x, y, z)

def warpInput : Array UInt32 :=
  (Array.range 32).map fun i =>
    let t : UInt32 := i.toUInt32
    let hi : UInt32 := if t % 3 == 0 then 0x40000000 else 0
    if t % 5 == 0 then 7 else (t * (0x9E3779B9 : UInt32)) ^^^ (t >>> 2) ||| hi

def warpRef (t : Nat) : Array UInt32 :=
  let x := warpInput
  let me := x[t]!
  let up := if t ≥ 3 then x[t - 3]! else me
  let down := if t % 16 + 5 ≤ 15 then x[t + 5]! else me
  let bit (j : Nat) : UInt32 := (1 : UInt32) <<< j.toUInt32
  let ballot : UInt32 :=
    (List.range 32).foldl (fun m j => if x[j]! &&& 1 == 1 then m ||| bit j else m) 0
  let same (f : UInt32 → UInt32) : UInt32 :=
    (List.range 32).foldl (fun m j => if f x[j]! == f me then m ||| bit j else m) 0
  let allSame := (List.range 32).all fun j => x[j]! >>> 30 == x[0]! >>> 30
  #[x[(t * 7) % 32]!, up, down, x[t ^^^ 9]!, ballot,
    if (List.range 32).all (x[·]! != 0) then 1 else 0,
    if (List.range 32).any (x[·]! == 7) then 1 else 0,
    same (· &&& 3),
    (if allSame then 0xFFFFFFFF else 0) ||| (if allSame then 0x80000000 else 0),
    x.foldl (· + ·) 0, x.foldl max 0]


/-! ## An independent floating-point oracle

Binary formats by precision `p` (significand bits with the hidden one) and exponent range;
values as exact rationals; rounding as IEEE 754 §4.3 states it. Written for these tests apart
from `NV.Num`, so that the two check each other. -/

structure Bin where
  p : Nat
  emin : Int
  emax : Int
  bits : Nat

def B64 : Bin := ⟨53, -1022, 1023, 64⟩
def B32 : Bin := ⟨24, -126, 127, 32⟩
def B16 : Bin := ⟨11, -14, 15, 16⟩
def BB16 : Bin := ⟨8, -126, 127, 16⟩

inductive Mode where
  | rn | rz | rm | rp

/-- A finite value as sign and the rational `num / den`; or NaN or an infinity. -/
inductive Q where
  | fin (neg : Bool) (num den : Nat)
  | inf (neg : Bool)
  | nan

namespace Bin

def expBits (b : Bin) : Nat := b.bits - b.p
def bias (b : Bin) : Int := b.emax

def decode (b : Bin) (x : Nat) : Q :=
  let neg := x >>> (b.bits - 1) == 1
  let e := (x >>> (b.p - 1)) % 2 ^ b.expBits
  let f := x % 2 ^ (b.p - 1)
  if e == 2 ^ b.expBits - 1 then (if f == 0 then .inf neg else .nan)
  else
    let m := if e == 0 then f else f + 2 ^ (b.p - 1)
    let ex : Int := (if e == 0 then b.emin else (e : Int) - b.bias) - (b.p - 1)
    if ex ≥ 0 then .fin neg (m * 2 ^ ex.toNat) 1 else .fin neg m (2 ^ (-ex).toNat)

/-- `⌊log₂ (n / d)⌋` for positive `n`, `d`. -/
def ilog2 (n d : Nat) : Int :=
  let e : Int := (Nat.log2 n : Int) - Nat.log2 d
  let atLeast := if e ≥ 0 then n ≥ d * 2 ^ e.toNat else n * 2 ^ (-e).toNat ≥ d
  if atLeast then e else e - 1

/-- The encoding of `±n / d` rounded by `mode`. -/
def round (b : Bin) (mode : Mode) (neg : Bool) (n d : Nat) : Nat :=
  let sign := if neg then 2 ^ (b.bits - 1) else 0
  if n == 0 then sign else
  let ulp : Int := max (ilog2 n d) b.emin - (b.p - 1)
  let (m, r, den) : Nat × Nat × Nat :=
    if ulp ≥ 0 then (n / (d * 2 ^ ulp.toNat), n % (d * 2 ^ ulp.toNat), d * 2 ^ ulp.toNat)
    else (n * 2 ^ (-ulp).toNat / d, n * 2 ^ (-ulp).toNat % d, d)
  let up := match mode with
    | .rn => 2 * r > den || (2 * r == den && m % 2 == 1)
    | .rz => false
    | .rm => neg && r != 0
    | .rp => !neg && r != 0
  let m := if up then m + 1 else m
  let (m, ulp) := if m == 2 ^ b.p then (2 ^ (b.p - 1), ulp + 1) else (m, ulp)
  let top := 2 ^ b.expBits - 1
  if ulp + (b.p - 1) > b.emax then
    let toInf := match mode with | .rn => true | .rz => false | .rm => neg | .rp => !neg
    if toInf then sign + top * 2 ^ (b.p - 1) else sign + top * 2 ^ (b.p - 1) - 1
  else if m < 2 ^ (b.p - 1) then sign + m
  else sign + (ulp + (b.p - 1) + b.bias).toNat * 2 ^ (b.p - 1) + (m - 2 ^ (b.p - 1))

end Bin

def qadd (a b : Bool × Nat × Nat) : Bool × Nat × Nat :=
  let (sa, na, da) := a
  let (sb, nb, db) := b
  let z : Int := (if sa then -((na * db : Nat) : Int) else (na * db : Nat)) +
    (if sb then -((nb * da : Nat) : Int) else (nb * da : Nat))
  (z < 0, z.natAbs, da * db)

def isNaN32 (x : UInt32) : Bool := (x >>> 23) &&& 0xFF == 0xFF && x &&& 0x7FFFFF != 0
def f32Of (x : UInt32) : Bool × Nat × Nat :=
  match B32.decode x.toNat with | .fin s n d => (s, n, d) | _ => (false, 0, 1)
def r32 (mode : Mode) (q : Bool × Nat × Nat) (zeroNeg : Bool := false) : UInt32 :=
  if q.2.1 == 0 then (if zeroNeg then 0x80000000 else 0) else (B32.round mode q.1 q.2.1 q.2.2).toUInt32

def f32Bits (x : Float) : UInt32 := x.toFloat32.toBits
def f32Val (x : UInt32) : Float := (Float32.ofBits x).toFloat

/-- Exact rational comparison of finite f32 values. -/
def ltQ (a b : Bool × Nat × Nat) : Bool :=
  let (s, n, _) := qadd a (!b.1, b.2.1, b.2.2)
  s && n != 0

/-- Round an f32 to an integer by `mode` (exact rational in, integer out). -/
def roundToInt (mode : Mode) (q : Bool × Nat × Nat) : Int :=
  let (neg, n, d) := q
  let m := n / d
  let r := n % d
  let up := match mode with
    | .rn => 2 * r > d || (2 * r == d && m % 2 == 1)
    | .rz => false
    | .rm => neg && r != 0
    | .rp => !neg && r != 0
  let v : Int := (if up then m + 1 else m)
  if neg then -v else v

def satInt (lo hi : Int) (v : Int) : Int := max lo (min hi v)

/-- `floatops` thread on finite `x`, `y`, `z`: expected words (`none`: any NaN). -/
def floatopsRef (x y z : UInt32) : Array (Option UInt32) :=
  let qx := f32Of x
  let qy := f32Of y
  let qz := f32Of z
  let mulq (a b : Bool × Nat × Nat) := (a.1 != b.1, a.2.1 * b.2.1, a.2.2 * b.2.2)
  let divq (a b : Bool × Nat × Nat) := (a.1 != b.1, a.2.1 * b.2.2, a.2.2 * b.2.1)
  let toF32 (i : Int) : UInt32 := r32 .rn (i < 0, i.natAbs, 1)
  let intF (mode : Mode) := roundToInt mode qx
  let negY := (!qy.1, qy.2.1, qy.2.2)
  let absX := (false, qx.2.1, qx.2.2)
  let sqrtX : Option UInt32 := if qx.1 && qx.2.1 != 0 then none else some (f32Bits (f32Val x).sqrt)
  let d := qadd (qx.1 != qy.1, qx.2.1 * qy.2.1, qx.2.2 * qy.2.2) qz
  let d64 : UInt64 := (if d.2.1 == 0 then 0 else B64.round .rn d.1 d.2.1 d.2.2).toUInt64
  #[ some (r32 .rn (qadd (mulq qx qy) qz)),
     some (r32 .rn (divq qx qy)),
     sqrtX,
     some (r32 .rn (divq (false, 1, 1) qx)),
     some (if ltQ qy qx then y else x),
     some (if ltQ qx qy then y else x),
     some (UInt32.ofInt (satInt (-2 ^ 31) (2 ^ 31 - 1) (intF .rn))),
     some (UInt32.ofInt (satInt (-2 ^ 31) (2 ^ 31 - 1) (intF .rz))),
     some (UInt32.ofInt (satInt 0 (2 ^ 32 - 1) (intF .rm))),
     some (toF32 (s32 y)),
     some (B16.round .rn qx.1 qx.2.1 qx.2.2).toUInt32,
     some (match B16.decode (y.toNat % 65536) with
       | .fin s n dd => r32 .rn (s, n, dd) s
       | .inf s => if s then 0xFF800000 else 0x7F800000
       | .nan => 0x7FFFFFFF),
     some (BB16.round .rn qx.1 qx.2.1 qx.2.2).toUInt32,
     some (r32 .rz (qadd qx qy)),
     some (r32 .rp (mulq qx qy)),
     some (r32 .rm (divq qx qy)),
     some (if qx.2.1 ≥ qx.2.2 * 2 ^ 23 then x else
       let i := intF .rn; if i == 0 then (if qx.1 then 0x80000000 else 0) else toF32 i),
     some (if qx.2.1 ≥ qx.2.2 * 2 ^ 23 then x else
       let i := intF .rm; if i == 0 then (if qx.1 then 0x80000000 else 0) else toF32 i),
     some (r32 .rn (qadd absX negY)),
     some ((if ltQ qx qy then 1 else 0) ||| (if isNaN32 x then 2 else 0)),
     some ((x &&& 0x7FFFFFFF) ||| (y &&& 0x80000000)),
     some (if qx.1 then 0 else if ltQ (false, 1, 1) qx then 0x3F800000 else x),
     some (hi64 d64),
     some (lo64 d64) ]

def floatInputs : Array (UInt32 × UInt32 × UInt32) :=
  let v : List UInt32 := [0x3F800000, 0xC0200000, 0x00200000, 0x7149F2CA, 0x3DCCCCCD, 0x40490FDB,
    0xBF000000, 0x4B000001, 0x3F000001, 0xC1A00000, 0x00000001, 0x7F7FFFFF, 0x42F6E979,
    0xBEAAAAAB, 0x3FC00000, 0x40200000]
  (Array.range 48).map fun i => (v[i % 16]!, v[(i * 7 + 3) % 16]!, v[(i * 5 + 1) % 16]!)

/-! ## mma.sync, ldmatrix, stmatrix, movmatrix on small integers (every sum exact) -/

def h16 (i : Int) : Nat := B16.round .rn (i < 0) i.natAbs 1
def mmaA (i : Nat) : Int := ((i * 5 + 1) % 7 : Nat) - 3
def mmaBt (i : Nat) : Int := ((i / 16 * 3 + i % 16 * 2) % 5 : Nat) - 2
def mmaC (i : Nat) : Int := (i / 8 : Int) - (i % 8 : Int)
def pack16 (xs : Array Nat) : Array UInt32 :=
  (Array.range (xs.size / 2)).map fun j => (xs[2 * j]! ||| (xs[2 * j + 1]! <<< 16)).toUInt32
def mmaD : Array UInt32 :=
  (Array.range 128).map fun i =>
    let r := i / 8
    let col := i % 8
    let v := (List.range 16).foldl (fun acc k => acc + mmaA (r * 16 + k) * mmaBt (col * 16 + k))
      (mmaC i)
    r32 .rn (v < 0, v.natAbs, 1)
def mmaAT : Array UInt32 :=
  (Array.range 32).map fun l =>
    (h16 (mmaA ((2 * (l % 4)) * 16 + l / 4)) ||| (h16 (mmaA ((2 * (l % 4) + 1) * 16 + l / 4)) <<< 16)).toUInt32

def cases : List Case :=
  let n := 100
  let a : Array UInt64 := (Array.range n).map fun i => (i.toUInt64 * 0x1_0000_0001) - 7
  let b : Array UInt64 := (Array.range n).map fun i => 0xFFFF_FFFF_0000_0000 + i.toUInt64 * 977
  let ints := intInputs
  let m := ints.size
  let rin : Array UInt32 := (Array.range 1000).map fun (i : Nat) => u32 ((i : Int) * 31 - 5000)
  let (w, h) := (40, 24)
  let fins := floatInputs
  let fm := fins.size
  let tin : Array UInt32 := (Array.range (w * h)).map fun i => (i * 2654435761).toUInt32
  [ { name := "vecadd_i64: grid-stride loop, 64-bit integer add"
      file := "tests/ptx/kernels/vecadd.sm_90a.ptx", entry := "vecadd_i64"
      grid := (2, 1, 1), block := (32, 1, 1)
      inputs := [(0, wordsOf64 a), (8 * n, wordsOf64 b)]
      params := [base, base + (8 * n).toUInt64, base + (16 * n).toUInt64, n.toUInt64]
      expect := [(16 * n, wordsOf64 ((Array.range n).map fun i => a[i]! + b[i]!))] },
    { name := "intops: integer and bit instructions on edge values"
      file := "tests/ptx/kernels/intops.sm_90a.ptx", entry := "intops"
      grid := (1, 1, 1), block := (64, 1, 1)
      inputs := [(0, ints.map (·.1)), (4 * m, ints.map (·.2.1)), (8 * m, ints.map (·.2.2))]
      params := [base, base + (4 * m).toUInt64, base + (8 * m).toUInt64,
                 base + (12 * m).toUInt64, m.toUInt64]
      expect := [(12 * m, ints.flatMap fun (x, y, z) => intopsRef x y z)] },
    { name := "reduce_sum_i32: shuffles, shared memory, bar.sync, atomic add"
      file := "tests/ptx/kernels/reduce.sm_90a.ptx", entry := "reduce_sum_i32"
      grid := (4, 1, 1), block := (256, 1, 1)
      inputs := [(0, rin), (4096, #[0])]
      params := [base, base + 4096, 1000]
      expect := [(4096, #[rin.foldl (· + ·) 0])] },
    { name := "transpose_f32: tiled transpose through padded shared memory"
      file := "tests/ptx/kernels/transpose.sm_90a.ptx", entry := "transpose_f32"
      grid := (2, 1, 1), block := (32, 8, 1)
      inputs := [(0, tin)]
      params := [base + 8192, base, w.toUInt64, h.toUInt64]
      expect := [(8192, (Array.range (w * h)).map fun k => tin[(k % h) * w + k / h]!)] },
    { name := "warpops: shfl, vote, match, redux and activemask"
      file := "tests/ptx/kernels/warp.sm_90a.ptx", entry := "warpops"
      grid := (1, 1, 1), block := (32, 1, 1)
      inputs := [(0, warpInput)]
      params := [base, base + 128]
      expect := (List.range 32).map fun t => (128 + 48 * t, warpRef t)
      -- `(x & 1) ? __activemask() : 0`: which of the lanes taking the branch execute it
      -- together is the scheduler's choice; it holds this lane and only such lanes.
      holds := (List.range 32).map fun t => (128 + 48 * t + 44, 1, fun ws =>
        let msk := ws[0]!
        let taking : UInt32 := (List.range 32).foldl (fun (m : UInt32) (j : Nat) =>
          if warpInput[j]! &&& 1 == 1 then m ||| ((1 : UInt32) <<< j.toUInt32) else m) 0
        if warpInput[t]! &&& 1 == 0 then
          (if msk == 0 then none else some s!"lane {t}: activemask {msk}, expected 0")
        else if (msk >>> t.toUInt32) &&& 1 == 1 && msk &&& ~~~taking == 0 then none
        else some s!"lane {t}: activemask {msk} is not a set of branch-taking lanes with {t}") },
    { name := "floatops: rounding, conversions, min/max, fma and f64 arithmetic"
      file := "tests/ptx/kernels/floatops.sm_90a.ptx", entry := "floatops"
      grid := (1, 1, 1), block := (64, 1, 1)
      inputs := [(0, fins.map (·.1)), (4 * fm, fins.map (·.2.1)), (8 * fm, fins.map (·.2.2))]
      params := [base, base + (4 * fm).toUInt64, base + (8 * fm).toUInt64,
                 base + (12 * fm).toUInt64, fm.toUInt64]
      expect := []
      expectF := [(12 * fm, fins.flatMap fun (x, y, z) => floatopsRef x y z)] },
    { name := "vecadd_f32 and saxpy: add.f32, fma.rn.f32"
      file := "tests/ptx/kernels/vecadd.sm_90a.ptx", entry := "saxpy"
      grid := (2, 1, 1), block := (32, 1, 1)
      inputs := [(0, fins.map (·.1)), (4 * fm, fins.map (·.2.1))]
      params := [0x40490FDB, base, base + (4 * fm).toUInt64, fm.toUInt64]
      expect := [(4 * fm, fins.map fun (x, y, _) =>
        r32 .rn (qadd (let a := f32Of 0x40490FDB; let b := f32Of x; (a.1 != b.1, a.2.1 * b.2.1, a.2.2 * b.2.2)) (f32Of y)))] },
    { name := "reduce_max_f32: max.f32 through shared memory"
      file := "tests/ptx/kernels/reduce.sm_90a.ptx", entry := "reduce_max_f32"
      grid := (2, 1, 1), block := (256, 1, 1)
      inputs := [(0, (Array.range 700).map fun i => fins[i % fm]!.2.2 ^^^ (if i % 3 == 0 then 0x80000000 else 0))]
      params := [base, base + 4096, 700]
      expect := [(4096, #[0, 1].map fun b =>
        let xs := (List.range 700).filter (fun i => (i / 256) % 2 == b) |>.map fun i =>
          fins[i % fm]!.2.2 ^^^ (if i % 3 == 0 then 0x80000000 else 0)
        xs.foldl (fun m v => if ltQ (f32Of m) (f32Of v) then v else m) 0xFF800000)] },
    { name := "mma: ldmatrix, mma.sync m16n8k16 f16→f32, stmatrix round trip, movmatrix"
      file := "tests/ptx/kernels/mma.sm_90a.ptx", entry := "mma_m16n8k16"
      grid := (1, 1, 1), block := (32, 1, 1)
      inputs := [(0, pack16 ((Array.range 256).map (h16 ∘ mmaA))),
                 (512, pack16 ((Array.range 128).map (h16 ∘ mmaBt))),
                 (1024, (Array.range 128).map fun i => r32 .rn (mmaC i < 0, (mmaC i).natAbs, 1))]
      params := [base, base + 512, base + 1024, base + 2048, base + 3072, base + 4096]
      expect := [(2048, mmaD), (3072, pack16 ((Array.range 256).map (h16 ∘ mmaA))), (4096, mmaAT)] },
    -- Block 1 adds only subnormals: the global `atom.add.f32` flushes them, the shared one keeps
    -- them; every sum is exact, so the order of the atomics does not matter.
    { name := "atomics: atom.add f32 (global flushes, shared keeps subnormals), f16x2, f64"
      file := "tests/ptx/kernels/atomics.sm_90a.ptx", entry := "atomics"
      grid := (2, 1, 1), block := (32, 1, 1)
      inputs := [(64, (Array.range 64).map fun i => if i < 32 then r32 .rn (false, i + 1, 1) else 0x200),
                 (576, (Array.range 64).map fun i => (h16 (i % 5 : Nat) ||| (h16 2 <<< 16)).toUInt32),
                 (1152, wordsOf64 ((Array.range 64).map fun i => (B64.round .rn false i 2).toUInt64))]
      params := [base, base + 64, base + 512, base + 576, base + 1024, base + 1152, 64]
      expect := [(0, #[r32 .rn (false, 528, 1), r32 .rn (false, 528, 1), 0x4000, 0,
                   r32 .rn (false, 528, 1), 0]),
                 (512, #[(h16 126 ||| (h16 128 <<< 16)).toUInt32]),
                 (1024, wordsOf64 #[(B64.round .rn false 1008 1).toUInt64])] } ]

/-! ## Interleavings

Two-thread programs explored over every interleaving of their memory steps
(`Machine.explore`): the interpreter's memory is sequentially consistent, so the outcomes are
exactly the SC ones. -/

def litmusSrc (body : String) : String :=
  ".version 9.0\n.target sm_90a\n.address_size 64\n" ++
  ".visible .entry litmus(.param .u64 p)\n{\n  .reg .pred %p;\n  .reg .b32 %r<4>;\n" ++
  "  .reg .b64 %rd<4>;\n  ld.param.u64 %rd0, [p];\n  mov.u32 %r0, %tid.x;\n" ++
  "  setp.ne.u32 %p, %r0, 0;\n  @%p bra T1;\n" ++ body ++ "\n}\n"

/-- Message passing: data then flag; the reader sees the flag and then reads the data. -/
def mp : String := litmusSrc "
  mov.u32 %r1, 1;
  st.global.u32 [%rd0], %r1;
  st.global.u32 [%rd0+4], %r1;
  exit;
T1:
  ld.global.u32 %r2, [%rd0+4];
  ld.global.u32 %r3, [%rd0];
  st.global.u32 [%rd0+8], %r2;
  st.global.u32 [%rd0+12], %r3;
  exit;"

/-- Store buffering: each thread stores its flag, then reads the other's. -/
def sb : String := litmusSrc "
  mov.u32 %r1, 1;
  st.global.u32 [%rd0], %r1;
  ld.global.u32 %r2, [%rd0+4];
  st.global.u32 [%rd0+8], %r2;
  exit;
T1:
  mov.u32 %r1, 1;
  st.global.u32 [%rd0+4], %r1;
  ld.global.u32 %r3, [%rd0];
  st.global.u32 [%rd0+12], %r3;
  exit;"

/-- The sorted, deduplicated outcomes (the two words at byte 8) over every interleaving. -/
def outcomes (src : String) : Except String (List (UInt32 × UInt32)) := do
  let m ← read src
  let prog ← Sem.Prog.ofModule m
  let params ← paramBlock m "litmus" [base]
  let g := Store.empty base 4096
  let mach ← Machine.init prog { entry := "litmus", grid := (1, 1, 1), block := (2, 1, 1), params,
                                 global := g }
  let finals ← mach.explore 100
  let outs := finals.map fun f => let w := f.global.read (base + 8) 8; (w[0]!, w[1]!)
  return (outs.foldl (fun acc o => if acc.contains o then acc else acc ++ [o]) []).mergeSort
    (fun a b => a.1 < b.1 || (a.1 == b.1 && a.2 ≤ b.2))

/-- `brx.idx` through a `.branchtargets` list: index 2 takes the third label. -/
def brx : String := litmusSrc "
  ts: .branchtargets L0, L1, L2;
  mov.u32 %r1, 2;
  brx.idx %r1, ts;
L0:
  mov.u32 %r2, 10;
  bra.uni D;
L1:
  mov.u32 %r2, 11;
  bra.uni D;
L2:
  mov.u32 %r2, 12;
D:
  st.global.u32 [%rd0+8], %r2;
  st.global.u32 [%rd0+12], %r2;
  exit;
T1:
  exit;"

def litmus : List (String × String × List (UInt32 × UInt32)) :=
  [("MP: flag seen implies data seen (SC interleavings)", mp, [(0, 0), (0, 1), (1, 1)]),
   ("SB: some thread sees the other's store (SC interleavings)", sb, [(0, 1), (1, 0), (1, 1)]),
   ("brx.idx: the indexed branch target", brx, [(12, 12)])]

def runLitmus (name src : String) (want : List (UInt32 × UInt32)) : IO Bool := do
  match outcomes src with
  | .ok got =>
    if got == want then IO.println s!"ok   {name}"; return true
    IO.println s!"FAIL {name}: outcomes {got}, expected {want}"; return false
  | .error e => IO.println s!"FAIL {name}: {e}"; return false

/-! ## Single instructions -/

/-- (mnemonic, source types, destination types, sources, expected destinations). -/
def units : List (String × List Ty × List Ty × List UInt64 × List UInt64) :=
  [ ("min.f32", [.f32, .f32, .f32], [.f32], [0x3F800000, 0xC0000000, 0x3F000000], [0xC0000000]),
    ("max.abs.f32", [.f32, .f32, .f32], [.f32], [0xC0400000, 0x40000000, 0x3F800000], [0x40400000]),
    -- 1 + 2^-8 lies halfway between two bf16 values; stochastic rounding adds the random bits
    -- below bf16's last place and truncates.
    ("cvt.rs.bf16x2.f32", [.f32, .f32, .b32], [.b32], [0x3F808000, 0x3F808000, 0x7FFF8000],
      [0x3F803F81]),
    ("cvt.rn.f16x2.f32", [.f32, .f32], [.b32], [0x3F800000, 0xC0000000], [0x3C00C000]),
    ("cvt.rn.satfinite.e4m3x2.f32", [.f32, .f32], [.b16], [0x43E00000, 0x7F800000], [0x7E7E]),
    -- 448 / 2^2 = 112 = 1.75·2^6, e4m3 0 1101 110 = 0x6E; 1.0 / 4 = 0.25 = 0x28; the scale
    -- byte 0x81 is 2^2.
    ("cvt.rn.satfinite.scaled::n1::ue8m0.e4m3x2.f32", [.f32, .f32, .b16], [.b16],
      [0x43E00000, 0x3F800000, 0x81], [0x6E28]),
    -- e4m3 0x38 = 1.0 (upper) and 0x40 = 2.0 (lower), times 2^1 and 2^-1: bf16 2.0, 1.0.
    ("cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2", [.b16, .b16], [.b32], [0x3840, 0x807E],
      [0x40003F80]),
    -- s2f6 is value·64 in 8 bits: 1.5 → 96 = 0x60 (upper, from a); -3.0 saturates to -127 = 0x81.
    ("cvt.rn.satfinite.s2f6x2.f32", [.f32, .f32], [.b16], [0x3FC00000, 0xC0400000], [0x6081]),
    -- and back: 1.5 = bf16 0x3FC0; -127/64 = -1.111111b = bf16 0xBFFE.
    ("cvt.rn.bf16x2.s2f6x2", [.b16], [.b32], [0x6081], [0x3FC0BFFE]) ]

def runUnits : IO Nat := do
  let mut bad := 0
  for (m, ins, outs, args, want) in units do
    match (do (← OneInstr.make m ins outs).run args) with
    | .ok got =>
      if got == want then IO.println s!"ok   {m}"
      else bad := bad + 1; IO.println s!"FAIL {m}: got {got}, expected {want}"
    | .error e => bad := bad + 1; IO.println s!"FAIL {m}: {e}"
  return bad


/-! ## The checker rejects -/

def badSrc (target body : String) : String :=
  s!".version 9.0\n.target {target}\n.address_size 64\n.visible .entry k()\n\{\n" ++
  "  .reg .pred %p;\n  .reg .b32 %r<4>;\n  .reg .f32 %f;\n  .reg .b64 %rd;\n" ++ body ++ "\n  ret;\n}\n"

/-- (what, module, a substring of the expected diagnostic). -/
def rejected : List (String × String × String) :=
  [ ("an undeclared register", badSrc "sm_90a" "  add.s32 %r1, %r2, %q;", "%q is not declared"),
    ("a float register where .s32 is expected", badSrc "sm_90a" "  add.s32 %r1, %f, %r2;",
      "register of type .f32 where .s32 is expected"),
    ("a guard that is not a predicate", badSrc "sm_90a" "  @%r1 bra L;\nL:", "is not a predicate"),
    ("a type the instruction does not take", badSrc "sm_90a" "  popc.u16 %r1, %r2;",
      "does not take types .u16"),
    ("an instruction the target lacks", badSrc "sm_80" "  elect.sync %r1|%p, -1;",
      "elect.sync requires sm_90"),
    ("wgmma off sm_90a", badSrc "sm_100a" "  wgmma.fence.sync.aligned;", "requires sm_90a"),
    ("bf16x2 add before sm_90", badSrc "sm_80" "  add.rn.bf16x2 %r1, %r2, %r3;", "requires sm_90"),
    ("tcgen05 on sm_120a", badSrc "sm_120a" "  tcgen05.fence::before_thread_sync;", "requires sm_100a"),
    ("a vector of the wrong length", badSrc "sm_90a" "  ld.global.v4.u32 {%r0, %r1}, [%rd];",
      "2 registers where 4 are expected") ]

def runRejected : IO Nat := do
  let mut bad := 0
  for (what, src, want) in rejected do
    match read src with
    | .error e =>
      if (e.splitOn want).length > 1 then IO.println s!"ok   rejects {what}"
      else bad := bad + 1; IO.println s!"FAIL rejects {what}: {e}"
    | .ok m =>
      let ds := check m
      if ds.any (fun d => (d.msg.splitOn want).length > 1) then IO.println s!"ok   rejects {what}"
      else bad := bad + 1; IO.println s!"FAIL rejects {what}: {ds.toList.map (·.msg)}"
  return bad

/-! ## Programs of one thread -/

/-- Run `body` in one thread with `%a`, `%b`, `%c` holding the given words; the body stores its
results from `[%rd0]` on, and the first `n` words there are returned. -/
def runOne (body : String) (a b c : UInt32) (n : Nat) : Except String (Array UInt32) := do
  let src := ".version 9.0\n.target sm_90a\n.address_size 64\n" ++
    ".visible .entry one(.param .u64 p)\n{\n  .reg .b32 %a, %b, %c, %r<16>;\n  .reg .b64 %rd0;\n" ++
    s!"  ld.param.u64 %rd0, [p];\n  mov.b32 %a, {a};\n  mov.b32 %b, {b};\n  mov.b32 %c, {c};\n" ++
    body ++ "\n  ret;\n}\n"
  let m ← read src
  let prog ← Sem.Prog.ofModule m
  let params ← paramBlock m "one" [base]
  let out ← interp prog
    { entry := "one", grid := (1, 1, 1), block := (1, 1, 1), params, global := Store.empty base 4096 }
  return out.read base (4 * n)

/-- Video instructions with selectors, saturation, secondary operations, merges and masks; the
expected words follow §9.7.20's pseudocode by hand for `a = 0x80FF7F01`, `b = 0x00020003`,
`c = 0x11223344`. -/
def videoBody : List (String × UInt32) :=
  [ ("vadd.s32.s32.s32.sat %r1, %a.b1, %b.b0;", 0x82),
    ("vadd.u32.u32.u32.sat %r1.b0, %a.b1, %b.b0, %c;", 0x11223382),
    ("vsub.s32.s32.s32.sat %r1.b2, %a.b3, %b.b0, %c;", 0x11803344),
    ("vabsdiff.u32.s32.s32.min %r1, %a.h1, %b.h0, %c;", 0x7F04),
    ("vshr.s32.s32.u32.clamp %r1, %a, %b.h1;", 0xE03FDFC0),
    ("vset.u32.u32.lt.add %r1, %a.b2, %b.b2, %c;", 0x11223344),
    ("vmad.s32.u32.u32.shr7 %r1, %a.h0, -%b.h0, %c;", 0x22416C),
    ("vadd2.s32.s32.s32 %r1, %a.h11, %b, %c;", 0x81018102),
    ("vsub2.s32.s32.s32.sat %r1, %a.h11, %c, %b;", 0x80008000),
    ("vavrg4.u32.u32.u32 %r1.b10, %a, %b, %c;", 0x11224002),
    ("vmax4.s32.s32.s32.add %r1, %a, %b, %c;", 0x112233C8),
    ("vset2.u32.u32.ne %r1, %a.h01, %b, %c;", 0x00010001) ]

def runVideo : IO Nat := do
  let body := "\n".intercalate (videoBody.zipIdx.map fun ((i, _), k) =>
    s!"  {i}\n  st.global.u32 [%rd0+{4 * k}], %r1;")
  match runOne body 0x80FF7F01 0x00020003 0x11223344 videoBody.length with
  | .error e => IO.println s!"FAIL video: {e}"; return 1
  | .ok got =>
    let mut bad := 0
    for ((i, want), k) in videoBody.zipIdx do
      if got[k]! == want then IO.println s!"ok   {i}"
      else bad := bad + 1; IO.println s!"FAIL {i}: got {got[k]!}, expected {want}"
    return bad

def runCase (c : Case) : IO Bool := do
  let src ← IO.FS.readFile c.file
  let result : Except String (List String) := do
    let m ← read src
    let prog ← Sem.Prog.ofModule m
    let params ← paramBlock m c.entry c.params
    let mut g := Store.empty base 65536
    for (off, ws) in c.inputs do
      g := g.write (base + off.toUInt64) (4 * ws.size) ws 0
    let out ← interp prog { entry := c.entry, grid := c.grid, block := c.block, params, global := g }
    let mut errs := []
    for (off, ws) in c.expect do
      let got := out.read (base + off.toUInt64) (4 * ws.size)
      for k in [0:ws.size] do
        -- `holds` overrides the word it checks.
        if c.holds.any (fun (o, _, _) => o == off + 4 * k) then continue
        if got[k]! != ws[k]! then
          errs := errs ++ [s!"word {off / 4 + k}: got {got[k]!}, expected {ws[k]!}"]
    for (off, ws) in c.expectF do
      let got := out.read (base + off.toUInt64) (4 * ws.size)
      for k in [0:ws.size] do
        let ok := match ws[k]! with
          | some w => got[k]! == w
          | none => isNaN32 got[k]!
        unless ok do
          errs := errs ++ [s!"word {off / 4 + k}: got {got[k]!}, expected {ws[k]!}"]
    for (off, cnt, p) in c.holds do
      if let some e := p (out.read (base + off.toUInt64) (4 * cnt)) then errs := errs ++ [e]
    return errs
  match result with
  | .ok [] => IO.println s!"ok   {c.name}"; return true
  | .ok errs =>
    IO.println s!"FAIL {c.name}: {errs.length} mismatches"
    for e in errs.take 8 do IO.println s!"       {e}"
    return false
  | .error e => IO.println s!"FAIL {c.name}: {e}"; return false

def main : IO UInt32 := do
  let mut bad := 0
  for c in cases do
    unless ← runCase c do bad := bad + 1
  for (n, src, want) in litmus do
    unless ← runLitmus n src want do bad := bad + 1
  bad := bad + (← runUnits)
  bad := bad + (← runRejected)
  bad := bad + (← runVideo)
  IO.println s!"{cases.length} kernels, {litmus.length} interleaving tests, {units.length} \
    single instructions, {videoBody.length} video instructions and {rejected.length} rejected \
    modules, {bad} failing"
  return if bad == 0 then 0 else 1
