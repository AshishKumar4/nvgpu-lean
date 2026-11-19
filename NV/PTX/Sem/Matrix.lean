import NV.PTX.Sem.Float

/-!
# Warp-level matrix multiply-accumulate (`mma.sync`)

PTX ISA 9.4 §9.7.16.5: the fragments each lane holds (§9.7.16.5.1–.10) and the product computed
with the tensor-core numerics of `NV.Num.Tensor` for the target's family (`model`, `khattak2025`).
Shapes and types modelled: `m16n8k8`/`m16n8k16` with `.f16`/`.bf16` inputs and `.f32` or `.f16`
accumulators, `m16n8k4`/`m16n8k8` with `.tf32`, the integer forms (`m8n8k16`, `m16n8k16`,
`m16n8k32` with `.s8`/`.u8`; `m8n8k32`, `m16n8k32`, `m16n8k64` with `.s4`/`.u4`; `m8n8k128`,
`m16n8k128`, `m16n8k256` with `.b1` and `.and.popc` or `.xor.popc`), `m8n8k4`, `m16n8k4`,
`m16n8k8`, `m16n8k16` with `.f64`, and `m16n8k32` with `.e4m3`/`.e5m2`: on a family whose
`mma.sync` has an FP8 datapath (sm_120a) by its model; on `sm_90a` and `sm_100a` as FP16 MMAs
(`loweredF8`). On sm_120a also `.kind::f8f6f4`'s
`m16n8k32` with FP6 and FP4 inputs, each in a byte (an `.e2m1` in bits 2–5, an `.e3m2`/`.e2m3` in
bits 0–5, §9.7.16.5.14: `unbox`), and the block-scaled kinds (§9.7.16.3): `.kind::mxf8f6f4`
(`m16n8k32`) and `.kind::mxf4`/`mxf4nvf4` (`m16n8k64`, `.e2m1` eight to a register), each
element times its block's scale factor (`computeScaled`), through the B200's datapaths (the
RTX PRO 6000's taken as the B200's, `Blackwell.rtxPro6000`, inferred).
-/

namespace NV.PTX.Sem.Matrix

open NV.Num (Format)

/-- The shape and element formats of an `mma.sync`, and its fragment sizes in 32-bit words per
lane. -/
structure Mma where
  m : Nat
  n : Nat
  k : Nat
  /-- Bits per A/B element. -/
  w : Nat
  aWords : Nat
  bWords : Nat
  cWords : Nat
  /-- `.b1` only: the bit operation is `.xor` (else `.and`). -/
  xor : Bool := false
  deriving Repr

/-- Elements per 32-bit fragment register (an `.f64` element has a register of its own). -/
def Mma.per (x : Mma) : Nat := max 1 (32 / x.w)

/-- The input types whose products are exact: integers and binary64 (`product`). -/
def exact (a : Ty) : Bool := [Ty.s8, .u8, .s4, .u4, .b1, .f64].contains a

/-- FP6 and FP4 (sm_120a's `.kind::f8f6f4` and block-scaled kinds). -/
def narrow (a : Ty) : Bool := [Ty.e2m1, .e3m2, .e2m3].contains a

/-- An element of `w` bits as its format reads it: in an 8-bit container an `.e2m1` in bits 2–5
and an `.e3m2`/`.e2m3` in bits 0–5 (§9.7.16.5.14); every other element as it is. -/
def unbox (a : Ty) (w v : Nat) : Nat :=
  if w == 8 && a == .e2m1 then (v >>> 2) % 16 else if w == 8 && narrow a then v % 64 else v

def shape? (m n k : Nat) (a d : Ty) : Option Mma :=
  let mk (w : Nat) (cw : Nat) : Mma :=
    { m, n, k, w, aWords := m * k * w / 32 / 32, bWords := k * n * w / 32 / 32, cWords := cw }
  match m, n, k, a, d with
  | 16, 8, 8, .f16, .f32 | 16, 8, 16, .f16, .f32 | 16, 8, 8, .bf16, .f32 | 16, 8, 16, .bf16, .f32 =>
    some (mk 16 4)
  | 16, 8, 8, .f16, .f16 | 16, 8, 16, .f16, .f16 => some (mk 16 2)
  | 16, 8, 4, .tf32, .f32 | 16, 8, 8, .tf32, .f32 => some (mk 32 4)
  | 16, 8, 16, .s8, .s32 | 16, 8, 32, .s8, .s32 | 16, 8, 16, .u8, .s32 | 16, 8, 32, .u8, .s32 =>
    some (mk 8 4)
  | 8, 8, 16, .s8, .s32 | 8, 8, 16, .u8, .s32 => some (mk 8 2)
  | 16, 8, 32, .s4, .s32 | 16, 8, 64, .s4, .s32 | 16, 8, 32, .u4, .s32 | 16, 8, 64, .u4, .s32 =>
    some (mk 4 4)
  | 8, 8, 32, .s4, .s32 | 8, 8, 32, .u4, .s32 => some (mk 4 2)
  | 16, 8, 128, .b1, .s32 | 16, 8, 256, .b1, .s32 => some (mk 1 4)
  | 8, 8, 128, .b1, .s32 => some (mk 1 2)
  | 8, 8, 4, .f64, .f64 => some { m, n, k, w := 64, aWords := 2, bWords := 2, cWords := 4 }
  | 16, 8, 4, .f64, .f64 | 16, 8, 8, .f64, .f64 | 16, 8, 16, .f64, .f64 => some (mk 64 8)
  | 16, 8, 32, .e4m3, .f32 | 16, 8, 32, .e5m2, .f32 => some (mk 8 4)
  | 16, 8, 32, .e4m3, .f16 | 16, 8, 32, .e5m2, .f16 => some (mk 8 2)
  | 16, 8, 32, .e2m1, .f32 | 16, 8, 32, .e3m2, .f32 | 16, 8, 32, .e2m3, .f32 => some (mk 8 4)
  | 16, 8, 64, .e2m1, .f32 => some (mk 4 4)
  | _, _, _, _, _ => none

/-- Whether the model takes FP8 `mma.sync` on SM version `sm` as two FP16 MMAs (`loweredF8`):
sm_90a and sm_100a; on sm_120a it has its own datapath. -/
def f8Lowered (sm : Nat) : Bool := sm == 90 || sm == 100 || sm == 103

/-- The tensor-core datapath of SM version `sm` for inputs `a` into `d` (`NV.Num.Tensor`'s
models per family; `sm_120` has the B200's, `Blackwell.rtxPro6000`); an error names a
combination without a published model. -/
def model (sm : Nat) (a d : Ty) : Except String NV.Num.TensorModel :=
  let fp8 := a == .e4m3 || a == .e5m2
  let f16out := d == .f16
  if narrow a then
    if sm ≥ 120 && sm < 130 && NV.Num.Blackwell.rtxPro6000.value && !f16out then
      pure NV.Num.Blackwell.f8f32.value
    else .error s!"sm_{sm} has no FP6/FP4 mma.sync into .{Keyword.name d}"
  else if sm ≥ 80 && sm < 90 then
    if fp8 then .error "sm_80 has no FP8 tensor-core datapath"
    else pure <| if a == .tf32 then NV.Num.Ampere.tf32f32.value
      else if f16out then NV.Num.Ampere.f16f16.value else NV.Num.Ampere.f16f32.value
  else if sm == 90 then
    if fp8 then .error "sm_90a has no FP8 tensor-core datapath (mma.sync runs as `loweredF8`)"
    else pure <| if a == .tf32 then NV.Num.Hopper.tf32f32.value
      else if f16out then NV.Num.Hopper.f16f16.value else NV.Num.Hopper.f16f32.value
  else if (sm ≥ 100 && sm < 120) || (sm ≥ 120 && sm < 130 && NV.Num.Blackwell.rtxPro6000.value) then
    if fp8 && f16out then .error s!"no published measurement of sm_{sm}'s FP8 datapath into FP16"
    else pure <| if fp8 then NV.Num.Blackwell.f8f32.value else if a == .tf32 then NV.Num.Blackwell.tf32f32.value
      else if f16out then NV.Num.Blackwell.f16f16.value else NV.Num.Blackwell.f16f32.value
  else .error s!"no tensor-core model for sm_{sm}"

/-- Row and column of element `i` of a lane's fragment of A (§9.7.16.5.2–.13): lane (g, q) holds
`per` consecutive elements of row `g` from column `q·per` per register; in the `m16` shapes the
registers alternate between rows `g` and `g + 8`, and the second pair of registers holds the
upper half of K. (§9.7.16.5.13 gives `.m16n8k256`'s A columns as `q·32 + i` for `i < 64`, which
would give lanes overlapping columns; the masked `q·32 + (i & 31)` of its B, and of every other
shape, is the one taken.) -/
def aCoord (x : Mma) (lane i : Nat) : Nat × Nat :=
  let g := lane / 4
  let q := lane % 4
  if x.m == 8 then (g, q * x.per + i) else
  let r := i / x.per
  (g + 8 * (r % 2), q * x.per + i % x.per + 4 * x.per * (r / 2))

def bCoord (x : Mma) (lane i : Nat) : Nat × Nat :=
  let g := lane / 4
  let q := lane % 4
  if x.m == 8 then (q * x.per + i, g) else
  let r := i / x.per
  (q * x.per + i % x.per + 4 * x.per * r, g)

def cCoord (x : Mma) (lane i : Nat) : Nat × Nat :=
  let g := lane / 4
  let q := lane % 4
  if x.m == 8 then (g, q * 2 + i) else (g + 8 * (i / 2), q * 2 + i % 2)

/-- Where every fragment element lives, for lane `l` and element `i` (lane-major): the row-major
index in A (`m×k`), the column-major one in B (`col·k + row`: the products read B's columns) and
the row-major one in C and D (`m×n`). Computed once per instruction form. -/
structure Layout where
  nA : Nat
  nB : Nat
  nC : Nat
  /-- Bits of a C (and D) element. -/
  cw : Nat
  a : Array Nat
  b : Array Nat
  c : Array Nat
  deriving Inhabited

def layout (x : Mma) (cw : Nat) : Layout :=
  let nA := x.aWords * 32 / x.w
  let nB := x.bWords * 32 / x.w
  let nC := x.cWords * 32 / cw
  { nA, nB, nC, cw,
    a := (Array.range (32 * nA)).map fun p => let (r, c) := aCoord x (p / nA) (p % nA); r * x.k + c
    b := (Array.range (32 * nB)).map fun p => let (r, c) := bCoord x (p / nB) (p % nB); c * x.k + r
    c := (Array.range (32 * nC)).map fun p => let (r, c) := cCoord x (p / nC) (p % nC); r * x.n + c }

/-- Element `i` of `w`-bit elements (`w ≤ 32`) held in the words `word 0`, `word 1`, …. -/
@[inline] def elemOf (word : Nat → UInt32) (w i : Nat) : UInt32 :=
  if w == 32 then word i
  else (word (i * w / 32) >>> ((i * w) % 32).toUInt32) &&& ((1 <<< w.toUInt32) - 1)

/-- `computeAll`'s result on the fast datapath (`NV.Num.TensorModel.fast`), when the inputs are
floats it covers: machine words throughout, from the warp's fragments read through `word l j`
(lane `l`'s `j`-th word of A, then B, then C) into each D word's 32 lanes (`[word][lane]`).
`none` where the fast datapath does not apply. -/
def computeFast (sm : Nat) (x : Mma) (lay : Layout) (a b d : Ty)
    (word : Nat → Nat → UInt32) : Except String (Option (Array (Array UInt32))) := do
  let fmt : Ty → Option Format := fun t => (Float.lanes? t).map (·.1)
  match fmt a, fmt b, fmt d with
  | some aF, some bF, some dF =>
    if exact a || narrow a || narrow b then return none
    let t ← model sm a d
    let (A, B, C) := Id.run do
      let mut A : Array UInt32 := Array.replicate (x.m * x.k) 0
      let mut B : Array UInt32 := Array.replicate (x.k * x.n) 0
      let mut C : Array UInt64 := Array.replicate (x.m * x.n) 0
      for l in [0:32] do
        for i in [0:lay.nA] do
          A := A.set! lay.a[l * lay.nA + i]! (elemOf (word l) x.w i)
        for i in [0:lay.nB] do
          B := B.set! lay.b[l * lay.nB + i]! (elemOf (fun j => word l (x.aWords + j)) x.w i)
        for i in [0:lay.nC] do
          C := C.set! lay.c[l * lay.nC + i]!
            (elemOf (fun j => word l (x.aWords + x.bWords + j)) lay.cw i).toUInt64
      return (A, B, C)
    let some D := t.tileWords aF bF dF x.m x.n x.k A B C | return none
    return some <| (Array.range x.cWords).map fun j => (Array.range 32).map fun l =>
      if lay.cw == 32 then D[lay.c[l * lay.nC + j]!]!.toUInt32
      else (D[lay.c[l * lay.nC + 2 * j]!]! ||| (D[lay.c[l * lay.nC + 2 * j + 1]!]! <<< 16)).toUInt32
  | _, _, _ => return none

/-- Element `i` of a fragment of `w`-bit elements held in 32-bit words. -/
@[inline] def elem (ws : Array UInt32) (w i : Nat) : Nat :=
  if w == 64 then (ws[2 * i]!.toNat ||| (ws[2 * i + 1]!.toNat <<< 32))
  else if w == 32 then ws[i]!.toNat
  else ((ws[i * w / 32]! >>> ((i * w) % 32).toUInt32) &&& ((1 <<< w.toUInt32) - 1)).toNat

/-- Whether `product` can compute inputs `a` into `d` on SM version `sm`: the error it would give,
found when the instruction issues (its result is written by a collective's writeback, which
cannot fail). -/
def available (sm : Nat) (a d : Ty) : Except String Unit :=
  if exact a then pure ()
  else if f8Lowered sm && (a == .e4m3 || a == .e5m2) then
    if d == .f32 then pure () else .error s!"FP8 mma.sync is into .f32 only on sm_{sm}"
  else do let _ ← model sm a d

/-- `mma.sync.m16n8k32` with FP8 inputs into `.f32` on sm_90a and sm_100a (`f8Lowered`), as the
model defines it on the family's FP16 datapath: each FP8 element widened exactly to binary16, an
FP16 MMA from zero over the K indices `k % 4 ∈ {0, 1}` (an A register's low half), a second over
`k % 4 ∈ {2, 3}` accumulating onto the first, its K index `j` taking `k = 4⌊j/2⌋ + j % 2` (+ 2),
then `d = c + P` (round to nearest, subnormals kept). -/
def loweredF8 (sm : Nat) (a b : Ty) (m n k : Nat) (A B C : Array Nat) :
    Except String (Array Nat) := do
  let some aF := (Float.lanes? a).map (·.1) | throw "unknown A format"
  let some bF := (Float.lanes? b).map (·.1) | throw "unknown B format"
  let wide (f : Format) (v : Nat) : Nat := NV.Num.cvt f .f16 .rn {} v
  let kh := k / 2
  let half (off : Nat) : Array Nat × Array Nat :=
    let kk (j : Nat) : Nat := 4 * (j / 2) + j % 2 + off
    ((Array.range (m * kh)).map fun i => wide aF A[(i / kh) * k + kk (i % kh)]!,
     (Array.range (kh * n)).map fun i => wide bF B[kk (i / n) * n + i % n]!)
  let t ← model sm .f16 .f32
  let tile (A' B' acc : Array Nat) : Array Nat :=
    t.tile .f16 .f16 .f32 ((Array.range m).map fun r => A'.extract (r * kh) (r * kh + kh))
      ((Array.range n).map fun c => (Array.range kh).map fun j => B'[j * n + c]!) acc
  let (A1, B1) := half 0
  let (A2, B2) := half 2
  let P := tile A2 B2 (tile A1 B1 (Array.replicate (m * n) 0))
  return (Array.range (m * n)).map fun rc => NV.Num.add .f32 .rn false C[rc]! P[rc]!

/-- D = A·B + C by (row, col) for element bits `A` (M×K), `B` (K×N) and `C` (M×N), row-major:
integers exactly (`.b1`: the population count of `A ∧ B`, or of `A ⊕ B` when `xor`, plus C;
§9.7.16.5.14), binary64 by `dotF64`, the other formats by the family's tensor-core model. -/
def product (sm : Nat) (a b d : Ty) (satfinite : Bool) (m n k : Nat)
    (A B C : Array Nat) (xor : Bool := false) : Except String (Array Nat) := do
  let rows := (Array.range m).map fun r => A.extract (r * k) (r * k + k)
  let cols := (Array.range n).map fun col => (Array.range k).map fun kk => B[kk * n + col]!
  let fmt : Ty → Option Format := fun t => (Float.lanes? t).map (·.1)
  match a, d with
  | .s8, _ | .u8, _ | .s4, _ | .u4, _ | .b1, _ =>
    let toI (t : Ty) (v : Nat) : Int :=
      if t == .s8 && v ≥ 128 then (v : Int) - 256 else if t == .s4 && v ≥ 8 then (v : Int) - 16
      else v
    pure <| (Array.range (m * n)).map fun rc =>
      let c0 : Nat := C[rc]!
      let cv : Int := if c0 ≥ 2 ^ 31 then (c0 : Int) - 2 ^ 32 else c0
      let row := rows[rc / n]!
      let col := cols[rc % n]!
      let s := if xor then
          NV.Num.dotInt ((row.zip col).map fun (u, v) => ((u ^^^ v : Nat) : Int))
            (Array.replicate k 1) cv false
        else NV.Num.dotInt (row.map (toI a)) (col.map (toI b)) cv satfinite
      if s < 0 then (s + 2 ^ 32).toNat else s.toNat
  | .f64, _ => pure <| (Array.range (m * n)).map fun rc =>
      NV.Num.dotF64 rows[rc / n]! cols[rc % n]! C[rc]!
  | .e4m3, .f32 | .e5m2, .f32 =>
    if f8Lowered sm then loweredF8 sm a b m n k A B C else
    let some aF := fmt a | throw "unknown A format"
    let some bF := fmt b | throw "unknown B format"
    pure ((← model sm a d).tile aF bF .f32 rows cols C)
  | _, _ =>
    let some aF := fmt a | throw "unknown A format"
    let some bF := fmt b | throw "unknown B format"
    let some dF := fmt d | throw "unknown D format"
    pure ((← model sm a d).tile aF bF dF rows cols C)

/-- The `rows×cols` matrix (row-major element bits) of the fragments the lanes hold: lane `l`'s
element `i` of `w` bits, in its gathered words `[l·per + off, l·per + off + words)`, at
`coord l i`. -/
def unfrag (gathered : Array UInt32) (per off words w cols : Nat) (size : Nat)
    (coord : Nat → Nat → Nat × Nat) : Array Nat := Id.run do
  let mut M := Array.replicate size 0
  for l in [0:32] do
    let f := gathered.extract (l * per + off) (l * per + off + words)
    for i in [0:words * 32 / w] do
      let (r, col) := coord l i
      M := M.set! (r * cols + col) (elem f w i)
  return M

/-- Each lane's D fragment as words, from D (row-major, `cw`-bit elements; packed like C). -/
def refrag (x : Mma) (cw : Nat) (D : Array Nat) : Array (Array UInt32) :=
  let nC := x.cWords * 32 / cw
  (Array.range 32).map fun lane =>
    let out := (Array.range nC).map fun i => let (r, col) := cCoord x lane i; D[r * x.n + col]!
    if cw == 64 then out.flatMap fun v => #[v.toUInt32, (v >>> 32).toUInt32]
    else if cw == 16 then
      (Array.range (nC / 2)).map fun j => (out[2 * j]! ||| (out[2 * j + 1]! <<< 16)).toUInt32
    else out.map (·.toUInt32)

/-- Every lane's D fragment (as 32-bit words, indexed by lane) from every lane's A, B, C words,
gathered lane by lane (`aWords + bWords + cWords` per lane). `a`, `b`, `c`, `d` are the element
types. The tile is computed once (`NV.Num.TensorModel.tile` for the float types). -/
def computeAll (sm : Nat) (x : Mma) (a b c d : Ty) (satfinite : Bool)
    (gathered : Array UInt32) : Except String (Array (Array UInt32)) := do
  let per := x.aWords + x.bWords + x.cWords
  let cw := if c == .f64 then 64 else if c == .f16 then 16 else 32
  let A := (unfrag gathered per 0 x.aWords x.w x.k (x.m * x.k) (aCoord x)).map (unbox a x.w)
  let B := (unfrag gathered per x.aWords x.bWords x.w x.n (x.k * x.n) (bCoord x)).map (unbox b x.w)
  let C := unfrag gathered per (x.aWords + x.bWords) x.cWords cw x.n (x.m * x.n) (cCoord x)
  return refrag x cw (← product sm a b d satfinite x.m x.n x.k A B C x.xor)

/-- A block-scaled `mma.sync`'s scaling (§9.7.16.3, Tables 44–46): the scale factors' format, how
many a row of A and a column of B have (`.scale_vec::NX`), and the selectors
`{byte-id-a, thread-id-a}` and `{byte-id-b, thread-id-b}`. -/
structure Scaling where
  format : Format
  vec : Nat
  byteA : Nat
  threadA : Nat
  byteB : Nat
  threadB : Nat
  deriving Inhabited

/-- The lane whose scale register holds row `r`'s factors (Figure 46): in quad `r % 8`, the
thread pair `thread-id-a` selects, its lower thread rows 0–7 and its upper rows 8–15. -/
def Scaling.laneA (s : Scaling) (r : Nat) : Nat := 4 * (r % 8) + 2 * s.threadA + r / 8

/-- The lane whose scale register holds column `n`'s factors (Figure 47): thread `thread-id-b` of
quad `n`. -/
def Scaling.laneB (s : Scaling) (n : Nat) : Nat := 4 * n + s.threadB

/-- Factor `j` of a scale register (Figure 48): its byte `byte-id + j`. -/
def Scaling.factor (byte j : Nat) (w : UInt32) : Nat := (w.toNat >>> (8 * (byte + j))) % 256

/-- `computeAll` with block scaling: the gathered words per lane are A's, B's, C's, then the
lane's `scale-a-data` and `scale-b-data`. An element `k` of a row of A is the element's value
times the row's factor `k / (K / vec)`, exact (`BlockFormat.value`), likewise B's columns; the
products go through the datapath (`model`) from those values. Into `.f32` only. -/
def computeScaled (sm : Nat) (x : Mma) (a b : Ty) (s : Scaling) (gathered : Array UInt32) :
    Except String (Array (Array UInt32)) := do
  let per := x.aWords + x.bWords + x.cWords + 2
  let A := (unfrag gathered per 0 x.aWords x.w x.k (x.m * x.k) (aCoord x)).map (unbox a x.w)
  let B := (unfrag gathered per x.aWords x.bWords x.w x.n (x.k * x.n) (bCoord x)).map (unbox b x.w)
  let C := unfrag gathered per (x.aWords + x.bWords) x.cWords 32 x.n (x.m * x.n) (cCoord x)
  let some aF := (Float.lanes? a).map (·.1) | throw "unknown A format"
  let some bF := (Float.lanes? b).map (·.1) | throw "unknown B format"
  let t ← model sm (if narrow a then a else b) .f32
  let blk := x.k / s.vec
  let reg (l j : Nat) : UInt32 := gathered[l * per + x.aWords + x.bWords + x.cWords + j]!
  let fa (r : Nat) : Array Nat := (Array.range s.vec).map fun j => Scaling.factor s.byteA j (reg (s.laneA r) 0)
  let fb (n : Nat) : Array Nat := (Array.range s.vec).map fun j => Scaling.factor s.byteB j (reg (s.laneB n) 1)
  -- The FP4 kinds (E2M1 eight to a register): the B200's E2M1 block-scaled datapath
  -- (`NV.Num.Scaled`); the others' scaled elements on the FP8 one.
  let D := if x.w == 4 then
      let kd : NV.Num.Scaled.Kind := { scaleBlock := blk, nvfp4 := s.format == .ue4m3 }
      let rows := (Array.range x.m).map fun r => (Array.range x.k).map fun kk => aF.decode A[r * x.k + kk]!
      let cols := (Array.range x.n).map fun n => (Array.range x.k).map fun kk => bF.decode B[kk * x.n + n]!
      let fas := (Array.range x.m).map fun r => (fa r).map s.format.decode
      let fbs := (Array.range x.n).map fun n => (fb n).map s.format.decode
      (Array.range (x.m * x.n)).map fun rc =>
        NV.Num.Scaled.dot kd s.format rows[rc / x.n]! cols[rc % x.n]! fas[rc / x.n]! fbs[rc % x.n]! C[rc]!
    else
      let fA : NV.Num.BlockFormat := { elem := aF, scale := s.format, size := blk }
      let fB : NV.Num.BlockFormat := { elem := bF, scale := s.format, size := blk }
      let rows := (Array.range x.m).map fun r => (Array.range x.k).map fun kk =>
        fA.value (fa r)[kk / blk]! A[r * x.k + kk]!
      let cols := (Array.range x.n).map fun n => (Array.range x.k).map fun kk =>
        fB.value (fb n)[kk / blk]! B[kk * x.n + n]!
      (Array.range (x.m * x.n)).map fun rc =>
        t.dotVals (aF.man + s.format.man) (bF.man + s.format.man) .f32 rows[rc / x.n]! cols[rc % x.n]! C[rc]!
  return refrag x 32 D

/-! ## Sparse `mma.sp` (§9.7.16.6)

A is stored compressed: of every chunk of four granules of a row (a granule is an element, or two
for `.s4`/`.u4`), two, in the fragment of the dense `mma` of half the K; the metadata gives their
places. A row's 2-bit index pairs (low pair: the first stored granule; high: the second) come from
one lane's metadata register, four bits a chunk (Figures 132, 139–140, 142, 149–150): row `r` of
lane `4(r % 8) + 2h + r / 8`, where `h` is the half of K for `m16n8k64` (`.s8`/`.u8`) and
`m16n8k128` (`.s4`/`.u4`), whose rows have sixteen chunks, and the sparsity selector otherwise.
Only the integer types are modelled: their products are exact. -/

/-- The shapes of an integer `mma.sp` (`m16n8k32`/`k64` `.s8`/`.u8`, `m16n8k64`/`k128`
`.s4`/`.u4`): the dense one's, and the one of half the K whose A fragment holds the compressed A. -/
def sparseShape? (m n k : Nat) (a d : Ty) : Option (Mma × Mma) := do
  let xa ← shape? m n (k / 2) a d
  unless m == 16 && n == 8 && [Ty.s8, .u8, .s4, .u4].contains a do none
  unless k == (if xa.w == 4 then 64 else 32) || k == (if xa.w == 4 then 128 else 64) do none
  return ({ xa with k, bWords := k * n * xa.w / 32 / 32 }, xa)

/-- The dense A (row-major element bits) of a sparse `mma.sp` of dense shape `x`, from the
compressed `A'` (`m × k/2`), every lane's metadata word and the selector; an error for metadata
the ISA leaves undefined (two equal indices; with `ordered`, decreasing ones). -/
def sparseA (x : Mma) (A' : Array Nat) (md : Array UInt32) (sel : Nat) (ordered : Bool) :
    Except String (Array Nat) := do
  let g := if x.w == 4 then 2 else 1
  let big := x.k == 64 * g
  let kh := x.k / 2
  let mut A := Array.replicate (x.m * x.k) 0
  for r in [0:x.m] do
    for cc in [0:x.k / (4 * g)] do
      let h := if big then cc / 8 else sel
      let f := (md[4 * (r % 8) + 2 * h + r / 8]! >>> (4 * (cc % 8)).toUInt32) &&& 0xf
      let i0 := (f &&& 3).toNat
      let i1 := (f >>> 2).toNat
      if i0 == i1 || (ordered && i0 > i1) then
        throw s!"mma.sp: metadata {f} for row {r}, chunk {cc} is undefined"
      for j in [0:2 * g] do
        let slot := if j < g then i0 else i1
        A := A.set! (r * x.k + 4 * g * cc + g * slot + j % g) A'[r * kh + 2 * g * cc + j]!
  return A

/-- `computeAll` for `mma.sp` of dense shape `x` on integers: per lane the compressed A's words
(`xa`: the dense `mma` of half the K), then B's, C's and the metadata word. -/
def computeSparse (x xa : Mma) (a b : Ty) (satfinite : Bool) (sel : Nat) (ordered : Bool)
    (gathered : Array UInt32) : Except String (Array (Array UInt32)) := do
  unless [Ty.s8, .u8, .s4, .u4].contains a do
    throw "mma.sp: no published numerics for the sparse tensor-core datapath of floats"
  let big := x.k == (if x.w == 4 then 128 else 64)
  if sel > (if big then 0 else 1) then throw s!"mma.sp: sparsity selector {sel} is undefined"
  let per := xa.aWords + x.bWords + x.cWords + 1
  let A' := unfrag gathered per 0 xa.aWords x.w xa.k (x.m * xa.k) (aCoord xa)
  let B := unfrag gathered per xa.aWords x.bWords x.w x.n (x.k * x.n) (bCoord x)
  let C := unfrag gathered per (xa.aWords + x.bWords) x.cWords 32 x.n (x.m * x.n) (cCoord x)
  let md := (Array.range 32).map fun l => gathered[l * per + per - 1]!
  let A ← sparseA x A' md sel ordered
  return refrag x 32 (← product 0 a b .s32 satfinite x.m x.n x.k A B C)

/-- The lane's D fragment (`computeAll`'s entry for `lane`). -/
def compute (sm : Nat) (x : Mma) (a b c d : Ty) (satfinite : Bool)
    (gathered : Array UInt32) (lane : Nat) : Except String (Array UInt32) := do
  return (← computeAll sm x a b c d satfinite gathered)[lane]!

/-! ## `ldmatrix` and `stmatrix` (§9.7.16.5.15–16)

Lanes give the addresses of the matrices' rows of 16 bytes (`rows` a matrix: 16 for `m16n16`,
else 8; lane `rows·m + i` row `i` of matrix `m`); each lane's registers hold its fragment.
`MatMove.at` says which row and which element of it each register element is: the figures of
§9.7.16.5.15–16 (`.m8n8`, `.m16n16`, `.m8n16`, `.m16n8`) with `.trans` reading the rows as
columns. For `.m16n16`, the lane of a row's address and the order of a register's bytes are
those of CUTLASS's `SM100_U8x8_LDSM_T` and `SM100_U8x16_LDSM_T` (`cute/atom/copy_traits_sm100.hpp`). The padded formats (`.b4x16_p64`: sixteen 4-bit elements then 64 bits of
padding; `.b6x16_p32`: sixteen 6-bit ones then 32) widen each element to a byte: a 4-bit one
into bits 5:2 and a 6-bit one into bits 5:0, the container of `mma`'s `.kind::f8f6f4` operands
(§9.7.16.5.14), which these loads feed (inferred); element `i` of a row at bits `w·i`
(inferred). -/

/-- A form of `ldmatrix`/`stmatrix`: its shape, `.trans`, the bits of an element in memory
(16, 8, or 4 and 6 for the padded formats) and the number of matrices. -/
structure MatMove where
  m : Nat
  n : Nat
  trans : Bool
  src : Nat
  num : Nat
  deriving Repr

/-- Rows a matrix has in memory: 16 for `.m16n16`, else 8. -/
def MatMove.rows (x : MatMove) : Nat := if x.m == 16 && x.n == 16 then 16 else 8

/-- Registers a matrix takes per lane. -/
def MatMove.regs (x : MatMove) : Nat := if x.m == 16 && x.n == 16 then 2 else 1

/-- Elements a register holds: two of 16 bits, or four bytes. -/
def MatMove.per (x : MatMove) : Nat := if x.src == 16 then 2 else 4

/-- The row (by the lane giving its address) and the element of the row that lane `l`'s element
`e` of register `j` is. -/
def MatMove.at (x : MatMove) (l j e : Nat) : Nat × Nat :=
  let base := x.rows * (j / x.regs)
  let a := l % 4
  let g := l / 4
  if x.src == 16 then
    if x.trans then (base + 2 * a + e, g) else (base + g, 2 * a + e)
  else if x.m == 16 && x.n == 16 then (base + 4 * a + e, 8 * (j % x.regs) + g)
  else if x.m == 16 then (base + 2 * a + e % 2, 8 * (e / 2) + g)
  else (base + g, 4 * a + e)

/-- Element `p` of a row whose bytes are `byte 0`, `byte 1`, …, widened as the form loads it. -/
def MatMove.elemOf (x : MatMove) (byte : Nat → UInt32) (p : Nat) : UInt32 :=
  let bit := x.src * p
  let raw := ((byte (bit / 8) ||| (byte (bit / 8 + 1) <<< 8)) >>> (bit % 8).toUInt32) &&&
    ((1 <<< x.src.toUInt32) - 1)
  if x.src == 4 then raw <<< 2 else raw

/-- Lane `l`'s register `j`, from the rows (`word τ k`: word `k` of the row lane `τ` gives). -/
def MatMove.frag (x : MatMove) (word : Nat → Nat → UInt32) (l j : Nat) : UInt32 :=
  let w := 32 / x.per
  (List.range x.per).foldl (fun acc e =>
    let (τ, p) := x.at l j e
    let byte (b : Nat) : UInt32 :=
      if b < 16 then (word τ (b / 4) >>> (8 * (b % 4)).toUInt32) &&& 0xff else 0
    acc ||| (x.elemOf byte p <<< (w * e).toUInt32)) 0

/-- The rows the lanes give, four words each (lane-major; zero for a lane giving none),
stored from every lane's registers (`reg l j`: lane `l`'s register `j`; elements of 16 or 8 bits,
unwidened). -/
def MatMove.rowsOf (x : MatMove) (reg : Nat → Nat → UInt32) : Array UInt32 :=
  let w := 32 / x.per
  let mask : UInt32 := (1 <<< w.toUInt32) - 1
  (List.range 32).foldl (fun rows l => (List.range (x.num * x.regs)).foldl (fun rows j =>
    let v := reg l j
    (List.range x.per).foldl (fun rows e =>
      let (τ, p) := x.at l j e
      let bit := w * p
      rows.modify (4 * τ + bit / 32)
        (· ||| (((v >>> (w * e).toUInt32) &&& mask) <<< (bit % 32).toUInt32))) rows) rows)
    (Array.replicate (4 * 32) 0)

/-! ## WMMA (§9.7.16.4)

The distribution of a WMMA fragment over the warp is unspecified and target dependent
(§9.7.16.4.1). The model's is the same on `sm_80`, `sm_90a` and `sm_120a`: every
fragment is a list of blocks, and lane (g, t) = (lane / 4, lane % 4) holds `e` consecutive
elements (g, e·t + j) of each row-wise block, or (e·t + j, g) of each column-wise one. An `.f16`
multiplicand holds its blocks twice (eight `.f16x2` registers for eight distinct pairs). The
fragment does not depend on the matrix's layout in memory. `tests/diff/wmma.ptx` checks every
form. -/

/-- A WMMA fragment: the element bits `w`, `e` elements per block, their orientation, the
blocks' origins in fragment order, and the fragment's element count. -/
structure WFrag where
  w : Nat
  e : Nat
  rowwise : Bool
  blocks : Array (Nat × Nat)
  count : Nat
  /-- Matrix elements per `w`-bit element of the fragment: the sub-byte and single-bit
  multiplicands move whole 32-bit words of packed elements, and their coordinates count words
  along K. -/
  sub : Nat := 1
  deriving Repr

/-- Row and column of element `i` of `lane`'s fragment. -/
def WFrag.coord (f : WFrag) (lane i : Nat) : Nat × Nat :=
  let i := i % (f.e * f.blocks.size)
  let (r0, c0) := f.blocks[i / f.e]!
  let g := lane / 4
  let t := lane % 4
  if f.rowwise then (r0 + g, c0 + f.e * t + i % f.e) else (r0 + f.e * t + i % f.e, c0 + g)

/-- The shapes WMMA has for each multiplicand type. -/
def wmmaShapes (a : Ty) : List (Nat × Nat × Nat) :=
  match a with
  | .f16 | .bf16 | .s8 | .u8 => [(16, 16, 16), (32, 8, 16), (8, 32, 16)]
  | .tf32 => [(16, 16, 8)]
  | .f64 => [(8, 8, 4)]
  | _ => []

/-- The fragment of matrix `kind` (`'a'`, `'b'`, or `'c'` for C and D) of an `m×n×k` WMMA with
element type `ty`, if modelled. -/
def wfrag (kind : Char) (m n k : Nat) (ty : Ty) : Option WFrag :=
  let range (lo hi step : Nat) : List Nat := (List.range ((hi - lo + step - 1) / step)).map (lo + step * ·)
  -- `m8n8k32` (`.s4`/`.u4`) and `m8n8k128` (`.b1`): `mma.sync`'s fragments (§9.7.16.5.4–.5), as
  -- `IMMA.8832`/`BMMA.88128` take them: a word of A's row `g`, B's column `g`,
  -- and C's elements (g, 2t + j).
  if (m, n, k) == (8, 8, 32) || (m, n, k) == (8, 8, 128) then
    let sub := k / 4
    match kind, ty with
    | 'a', t | 'b', t =>
      if (k == 32 && (t == .s4 || t == .u4)) || (k == 128 && t == .b1) then
        some { w := 32, e := 1, rowwise := kind == 'a', blocks := #[(0, 0)], count := 1, sub }
      else none
    | _, .s32 => some { w := 32, e := 2, rowwise := true, blocks := #[(0, 0)], count := 2 }
    | _, _ => none
  else if ty == .f64 then
    if (m, n, k) != (8, 8, 4) then none else
    some <| match kind with
      | 'a' => { w := 64, e := 1, rowwise := true, blocks := #[(0, 0)], count := 1 }
      | 'b' => { w := 64, e := 1, rowwise := false, blocks := #[(0, 0)], count := 1 }
      | _ => { w := 64, e := 2, rowwise := true, blocks := #[(0, 0)], count := 2 }
  else if kind == 'c' then
    let w := if ty == .f16 then 16 else 32
    if !([Ty.f16, .f32, .s32].contains ty) then none
    else if m == 8 then
      some { w, e := 2, rowwise := false, blocks := (range 0 n 8).toArray.map (0, ·), count := 8 }
    else if m == 16 then
      some { w, e := 2, rowwise := true, count := 8,
             blocks := ((range 0 n 8).flatMap fun j => [(0, j), (8, j)]).toArray }
    else some { w, e := 2, rowwise := true, blocks := (range 0 m 8).toArray.map (·, 0), count := 8 }
  else
    let e := match ty with | .f16 | .bf16 => 2 | .tf32 => 1 | .s8 | .u8 => 4 | _ => 0
    if e == 0 || !(wmmaShapes ty).contains (m, n, k) then none else
    let blocks : List (Nat × Nat) :=
      if kind == 'a' then
        (range 0 m 16).flatMap fun ib => (range 0 k (4 * e)).flatMap fun j =>
          (range ib (ib + min m 16) 8).map (·, j)
      else if m == 8 then
        (range 0 n 16).flatMap fun jb => (range 0 k (4 * e)).flatMap fun i =>
          (range jb (jb + 16) 8).map (i, ·)
      else
        (range 0 n 16).flatMap fun jb => (range jb (jb + min n 16) 8).flatMap fun j =>
          (range 0 k (4 * e)).map (·, j)
    some { w := 32 / e, e, rowwise := kind == 'a', blocks := blocks.toArray,
           count := if ty == .f16 then 16 else e * blocks.length }

/-- The fragment's 32-bit words (64-bit elements as two words) from its elements. -/
def WFrag.pack (f : WFrag) (vs : Array Nat) : Array UInt32 :=
  if f.w == 64 then vs.flatMap fun v => #[v.toUInt32, (v >>> 32).toUInt32]
  else
    let per := 32 / f.w
    (Array.range ((vs.size + per - 1) / per)).map fun r =>
      (List.range per).foldl (fun acc j => acc ||| ((vs[r * per + j]?.getD 0) <<< (f.w * j))) 0 |>.toUInt32

/-- Element `i` of a fragment held in `ws` (32-bit words). -/
def WFrag.elem (f : WFrag) (ws : Array UInt32) (i : Nat) : Nat := Matrix.elem ws f.w i

/-- Words per lane of the fragment. -/
def WFrag.words (f : WFrag) : Nat := (f.count * f.w + 31) / 32

/-- Every lane's D fragment of a WMMA (lane-major words, `fd.words` per lane) from every lane's
A, B and C fragment words gathered lane by lane. -/
def wmmaAll (sm : Nat) (m n k : Nat) (a b d : Ty) (satfinite : Bool)
    (fa fb fc fd : WFrag) (gathered : Array UInt32) (xor : Bool := false) :
    Except String (Array (Array UInt32)) := do
  if fa.sub > 1 then
    let some x := shape? m n k a d | throw s!"wmma.m{m}n{n}k{k}: no mma.sync of its types"
    return ← computeAll sm { x with xor } a b d d satfinite gathered
  let per := fa.words + fb.words + fc.words
  let mut A := Array.replicate (m * k) 0
  let mut B := Array.replicate (k * n) 0
  let mut C := Array.replicate (m * n) 0
  for l in [0:32] do
    let ws := gathered.extract (l * per) (l * per + per)
    let wa := ws.extract 0 fa.words
    let wb := ws.extract fa.words (fa.words + fb.words)
    let wc := ws.extract (fa.words + fb.words) per
    for i in [0:fa.count] do
      let (r, col) := fa.coord l i
      A := A.set! (r * k + col) (fa.elem wa i)
    for i in [0:fb.count] do
      let (r, col) := fb.coord l i
      B := B.set! (r * n + col) (fb.elem wb i)
    for i in [0:fc.count] do
      let (r, col) := fc.coord l i
      C := C.set! (r * n + col) (fc.elem wc i)
  -- The tf32 shape (`m16n16k8`) runs as `HMMA.1684` pairs chained through the accumulator:
  -- four products at a time, as the tensor cores then round.
  let D ← if a == .tf32 && k > 4 then
      (List.range (k / 4)).foldlM (init := C) fun acc q =>
        product sm a b d satfinite m n 4
          ((Array.range (m * 4)).map fun i => A[(i / 4) * k + 4 * q + i % 4]!)
          ((Array.range (4 * n)).map fun i => B[(4 * q + i / n) * n + i % n]!) acc
    else product sm a b d satfinite m n k A B C
  return (Array.range 32).map fun l =>
    fd.pack ((Array.range fd.count).map fun i => let (r, col) := fd.coord l i; D[r * n + col]!)

end NV.PTX.Sem.Matrix
