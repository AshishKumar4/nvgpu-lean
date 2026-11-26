/-!
# Tensor maps

A TMA descriptor (`CUtensorMap`) as the host builds it: the arguments of
`cuTensorMapEncodeTiled`, or of `cuTensorMapEncodeIm2col` (`im2col`) (CUDA Driver API 13.4,
§6.42 "Tensor Map Object Management"). NVIDIA's
128-byte encoding is opaque ("should only be accessed through CUDA APIs and PTX"), so the model
uses its own: `encode` writes every argument into 128 bytes, and `decode` reads them back
(`decode_encode`). A launch writes `encode m` wherever the host would have put the descriptor
(the parameter block for a `__grid_constant__` parameter, or a global buffer); the TMA model
decodes the 128 bytes at the descriptor address an instruction names. Descriptors in parameter,
global or shared memory, copies of them, and `tensormap.replace` (a field update) then follow
from data flow alone.

Every array has five entries, the most dimensions a map has; entries at and beyond `rank` are
carried but not used.
-/

namespace NV.Async

/-- `CUtensorMapDataType`, in the enum's order. -/
inductive TmDType where
  | u8 | u16 | u32 | s32 | u64 | s64 | f16 | f32 | f64 | bf16 | f32Ftz | tf32 | tf32Ftz
  /-- Packed 4- and 6-bit types: 16 values per 8 or 16 bytes. -/
  | u4x16Align8 | u4x16Align16 | u6x16Align16
  deriving DecidableEq, Repr, Inhabited

/-- `CUtensorMapInterleave`. -/
inductive Interleave where
  | none | b16 | b32
  deriving DecidableEq, Repr, Inhabited

/-- `CUtensorMapSwizzle`: 16-byte chunks swizzled within a 32-, 64- or 128-byte span, or 32- or
64-byte chunks within 128 bytes (`b128Atom32Flip8` also swaps 8-byte halves on alternate rows). -/
inductive Swizzle where
  | none | b32 | b64 | b128 | b128Atom32 | b128Atom32Flip8 | b128Atom64
  deriving DecidableEq, Repr, Inhabited

/-- `CUtensorMapL2promotion`: the granularity at which L2 fills from DRAM. -/
inductive L2Promotion where
  | none | b64 | b128 | b256
  deriving DecidableEq, Repr, Inhabited

/-- `CUtensorMapFloatOOBfill`: out-of-bound elements read as zero, or as a NaN that requests a
zero result from the consuming FMA. -/
inductive OobFill where
  | none | nanRequestZeroFma
  deriving DecidableEq, Repr, Inhabited

/-- Each enum's values in the driver's numbering. -/
def TmDType.all : List TmDType :=
  [.u8, .u16, .u32, .s32, .u64, .s64, .f16, .f32, .f64, .bf16, .f32Ftz, .tf32, .tf32Ftz,
   .u4x16Align8, .u4x16Align16, .u6x16Align16]
def Interleave.all : List Interleave := [.none, .b16, .b32]
def Swizzle.all : List Swizzle :=
  [.none, .b32, .b64, .b128, .b128Atom32, .b128Atom32Flip8, .b128Atom64]
def L2Promotion.all : List L2Promotion := [.none, .b64, .b128, .b256]
def OobFill.all : List OobFill := [.none, .nanRequestZeroFma]

/-- The driver's number of an enum value: its index in `all`. -/
def index {α : Type} [DecidableEq α] (all : List α) (a : α) : Nat := all.idxOf a

/-- Bytes per element; the packed types report their 16-value group over 16 (`none`). -/
def TmDType.bytes? : TmDType → Option Nat
  | .u8 => some 1
  | .u16 | .f16 | .bf16 => some 2
  | .u32 | .s32 | .f32 | .f32Ftz | .tf32 | .tf32Ftz => some 4
  | .u64 | .s64 | .f64 => some 8
  | .u4x16Align8 | .u4x16Align16 | .u6x16Align16 => none

def TmDType.isFloat : TmDType → Bool
  | .f16 | .f32 | .f64 | .bf16 | .f32Ftz | .tf32 | .tf32Ftz => true
  | _ => false

/-- Bytes of the span a swizzle mode permutes within. -/
def Swizzle.span : Swizzle → Nat
  | .none => 0
  | .b32 => 32
  | .b64 => 64
  | _ => 128

theorem Swizzle.span_le (s : Swizzle) : s.span ≤ 128 := by cases s <;> decide

/-- `cuTensorMapEncodeIm2col`'s own arguments (CUDA Driver API 13.4 §6.42; PTX ISA §5.5.4): the
image tensor is `NWC`, `NHWC` or `NDHWC` (rank 3 to 5, `C` dimension 0, `N` the last). -/
structure Im2col where
  /-- The bounding box's corners per spatial dimension, `W` first: the filter bases a copy walks
  along spatial dimension `i` run from `lower[i]` to `globalDim[i + 1] − 1 + upper[i]`. -/
  lower : Vector Int 3
  upper : Vector Int 3
  /-- Elements a copy moves along `C` per pixel, and pixels it moves. -/
  channelsPerPixel : Nat
  pixelsPerColumn : Nat
  deriving DecidableEq, Repr, Inhabited

structure TensorMap where
  dtype : TmDType
  /-- Dimensions, 1 to 5; dimension 0 is the contiguous one. -/
  rank : Nat
  globalAddress : UInt64
  /-- Elements per dimension. -/
  globalDim : Vector Nat 5
  /-- `globalStrides[i]`: bytes between consecutive indices of dimension `i + 1`. -/
  globalStrides : Vector Nat 4
  /-- Elements per dimension of the box one copy moves. -/
  boxDim : Vector Nat 5
  /-- Traversal step per dimension (dimension 0's is ignored without interleave). -/
  elementStrides : Vector Nat 5
  interleave : Interleave := .none
  swizzle : Swizzle := .none
  l2Promotion : L2Promotion := .none
  oobFill : OobFill := .none
  /-- An im2col map (`boxDim` unused), or `none` for a tiled one. -/
  im2col : Option Im2col := none
  deriving DecidableEq, Repr, Inhabited

namespace TensorMap

/-- The checks `cuTensorMapEncodeTiled` documents (CUDA Driver API 13.4 §6.42.1); the first one
that fails, or `none`. -/
def invalid? (m : TensorMap) : Option String := Id.run do
  let r := m.rank
  if r == 0 || r > 5 then return some s!"rank {r}: must be 1 to 5"
  if m.interleave != .none && r < 3 then return some "an interleaved map has rank 3 to 5"
  let align := if m.interleave == .b32 || m.dtype == .u4x16Align16 || m.dtype == .u6x16Align16
    then 32 else 16
  if m.globalAddress.toNat % align != 0 then
    return some s!"globalAddress {m.globalAddress}: must be {align}-byte aligned"
  let packed16 := m.dtype == .u4x16Align16 || m.dtype == .u6x16Align16
  if let some x := m.im2col then
    if r < 3 then return some s!"rank {r}: an im2col map has rank 3 to 5"
    -- The corners' ranges by rank (`cuTensorMapEncodeIm2col`).
    let lim : Int := if r == 3 then 32768 else if r == 4 then 128 else 16
    for i in [0:r - 2] do
      for c in [x.lower[i]!, x.upper[i]!] do
        if c < -lim || c ≥ lim then return some s!"bounding-box corner {c}: must be in [{-lim}, {lim - 1}]"
    if x.channelsPerPixel == 0 || x.channelsPerPixel > 256 then
      return some s!"channelsPerPixel {x.channelsPerPixel}: must be 1 to 256"
    if x.pixelsPerColumn == 0 || x.pixelsPerColumn > 1024 then
      return some s!"pixelsPerColumn {x.pixelsPerColumn}: must be 1 to 1024"
  for i in [0:r] do
    let d := m.globalDim[i]!
    if d == 0 || d > 2 ^ 32 then return some s!"globalDim[{i}] = {d}: must be 1 to 2^32"
    let b := m.boxDim[i]!
    if m.im2col.isNone && (b == 0 || b > 256) then return some s!"boxDim[{i}] = {b}: must be 1 to 256"
    let e := m.elementStrides[i]!
    if e == 0 || e > 8 then return some s!"elementStrides[{i}] = {e}: must be 1 to 8"
  if packed16 && m.globalDim[0]! % 128 != 0 then return some "globalDim[0]: a multiple of 128"
  if m.dtype == .u4x16Align8 && m.globalDim[0]! % 2 != 0 then return some "globalDim[0]: even"
  let strideAlign := if m.interleave == .b32 || packed16 then 32 else 16
  for i in [0:r - 1] do
    let s := m.globalStrides[i]!
    if s % strideAlign != 0 || s ≥ 2 ^ 40 then
      return some s!"globalStrides[{i}] = {s}: a multiple of {strideAlign} below 2^40"
  let inner0 := match m.im2col with | some x => x.channelsPerPixel | none => m.boxDim[0]!
  if packed16 && inner0 != 128 then return some "the box's inner dimension must be 128 for this type"
  if m.interleave == .none then
    if let some eb := m.dtype.bytes? then
      let inner := inner0 * eb
      if inner % 16 != 0 then return some s!"boxDim[0] spans {inner} bytes: a multiple of 16"
      if m.swizzle != .none && inner > m.swizzle.span then
        return some s!"boxDim[0] spans {inner} bytes, more than the swizzle's {m.swizzle.span}"
  if m.interleave == .b32 && m.swizzle != .b32 then return some "32B interleave needs 32B swizzle"
  if m.oobFill == .nanRequestZeroFma && !m.dtype.isFloat then
    return some "NaN fill needs a floating-point type"
  return none

/-! ## The 128-byte encoding

The descriptor's layout (sm_90a, sm_100a, sm_103a): bit fields of the 1024 bits read as one
little-endian number. Fields `tensormap.replace` never writes (L2 promotion, the im2col mode and
its corners) are placed as `fields` lists them (inferred).
-/

/-- `n` as `k` little-endian bytes (its low `8k` bits). -/
def le : Nat → Nat → List UInt8
  | 0, _ => []
  | k + 1, n => (n % 256).toUInt8 :: le k (n / 256)

/-- The number little-endian bytes spell. -/
def unle : List UInt8 → Nat
  | [] => 0
  | b :: bs => b.toNat + 256 * unle bs

/-- Values as bit fields of the given widths, from bit 0 up. -/
def packBits : List (Nat × Nat) → Nat
  | [] => 0
  | (w, v) :: fs => v % 2 ^ w + 2 ^ w * packBits fs

/-- Read bit fields of the given widths, from bit 0 up. -/
def unpackBits : List Nat → Nat → List Nat
  | [], _ => []
  | w :: ws, n => n % 2 ^ w :: unpackBits ws (n / 2 ^ w)

/-- The driver's element types (`TmDType.all`'s order) as the descriptor's 4-bit code and its
tf32 flag (bit 16): PTX's `.elemtype` numbers (Table 36), tf32 the f32 codes with the flag, the
packed types after `.bf16` (`.elemtype` 11 and 12 as 7 and 8 with the flag, 13–15 as 11–13). -/
def dtypeCode : TmDType → Nat × Nat
  | .u8 => (0, 0) | .u16 => (1, 0) | .u32 => (2, 0) | .s32 => (3, 0) | .u64 => (4, 0)
  | .s64 => (5, 0) | .f16 => (6, 0) | .f32 => (7, 0) | .f32Ftz => (8, 0) | .f64 => (9, 0)
  | .bf16 => (10, 0) | .tf32 => (7, 1) | .tf32Ftz => (8, 1) | .u4x16Align8 => (11, 0)
  | .u4x16Align16 => (12, 0) | .u6x16Align16 => (13, 0)

/-- The element type a code and tf32 flag name: the flag counts with the f32 codes alone. -/
def dtypeOf? (code tf32 : Nat) : Option TmDType :=
  match code, tf32 with
  | 7, 1 => some .tf32 | 8, 1 => some .tf32Ftz
  | 0, _ => some .u8 | 1, _ => some .u16 | 2, _ => some .u32 | 3, _ => some .s32 | 4, _ => some .u64
  | 5, _ => some .s64 | 6, _ => some .f16 | 7, _ => some .f32 | 8, _ => some .f32Ftz | 9, _ => some .f64
  | 10, _ => some .bf16 | 11, _ => some .u4x16Align8 | 12, _ => some .u4x16Align16
  | 13, _ => some .u6x16Align16 | _, _ => none

/-- A swizzle as the mode (bits 13–14) and the atomicity (bits 19–21). -/
def swizzleCode : Swizzle → Nat × Nat
  | .none => (0, 0) | .b32 => (1, 0) | .b64 => (2, 0) | .b128 => (3, 0)
  | .b128Atom32 => (3, 1) | .b128Atom32Flip8 => (3, 2) | .b128Atom64 => (3, 3)

/-- The swizzle a mode and atomicity name; another atomicity than 16 bytes goes with 128-byte
swizzling alone, and 96-byte swizzling (bit 22, sm_103a) is not modelled. -/
def swizzleOf? (mode atom b96 : Nat) : Option Swizzle :=
  if b96 != 0 then none else
  match mode, atom with
  | 0, 0 => some .none | 1, 0 => some .b32 | 2, 0 => some .b64 | 3, 0 => some .b128
  | 3, 1 => some .b128Atom32 | 3, 2 => some .b128Atom32Flip8 | 3, 3 => some .b128Atom64
  | _, _ => none

/-- Bits per corner of an im2col map of rank `r`: its `r − 2` lower corners share 16 bits, and so
do its upper ones (16, 8 and 5). -/
def cornerBits (r : Nat) : Nat := 16 / max (r - 2) 1

/-- An im2col map's corners as the 32 bits where a tiled map keeps box dimensions 1–4: the lower ones from bit 0, the upper ones from bit 16, `cornerBits` each, two's complement. -/
def cornerWord (r : Nat) (x : Im2col) : Nat :=
  let w := cornerBits r
  let enc (v : Int) : Nat := (v % (2 ^ w : Int)).toNat
  (List.range (r - 2)).foldl (fun acc i =>
    acc + enc x.lower[i]! * 2 ^ (w * i) + enc x.upper[i]! * 2 ^ (16 + w * i)) 0

/-- The corners back from their 32 bits. -/
def cornersOf (r n : Nat) : Vector Int 3 × Vector Int 3 :=
  let w := cornerBits r
  let dec (k : Nat) : Int :=
    let v := n / 2 ^ k % 2 ^ w
    if v ≥ 2 ^ (w - 1) then (v : Int) - 2 ^ w else v
  let get (base i : Nat) : Int := if i < r - 2 then dec (base + w * i) else 0
  (Vector.ofFn fun i => get 0 i, Vector.ofFn fun i => get 16 i)

/-- The pixels per column (inferred width: 10 bits, the PTX ISA's limit of 1024) after the box
fields; zero for a tiled map. -/
def im2colFields : Option Im2col → List (Nat × Nat)
  | none => [(10, 0)]
  | some x => [(10, x.pixelsPerColumn - 1)]

/-- A stride in 16-byte units, split as the descriptor keeps it: its low word, its high four bits. -/
def strideLo (s : Nat) : Nat := s / 16 % 2 ^ 32
def strideHi (s : Nat) : Nat := s / 16 / 2 ^ 32

/-- The descriptor's fields from bit 0 up, with their widths (gaps as zero fields):
the address (0x00); the control word (0x08): the im2col flag in bit 0, rank − 1 in bits 4–6,
element type 7–10, interleave 11–12, swizzle mode 13–14, fill mode 15, tf32 16, L2 promotion
17–18, swizzle atomicity 19–21, 96-byte swizzle 22; the four strides' low words in 16-byte units
(0x0c) and their high four bits (0x1c); the five dimensions − 1 (0x20); the element strides − 1,
three bits each, and box dimension 0 − 1 in bits 24–31 (0x34); box dimensions 1–4 − 1 (0x38); the
pixels per column − 1 (0x3c). An im2col map keeps its channels per pixel − 1 where box dimension
0 is and its corners where 1–4 are (`cornerWord`). The words from 0x40 are left zero. -/
def fields (m : TensorMap) : List (Nat × Nat) :=
  let dc := (dtypeCode m.dtype).1
  let tf := (dtypeCode m.dtype).2
  let sm := (swizzleCode m.swizzle).1
  let sa := (swizzleCode m.swizzle).2
  let g := m.globalDim.toList
  let st := m.globalStrides.toList
  let b := m.boxDim.toList
  let e := m.elementStrides.toList
  let ic := if m.im2col.isSome then 1 else 0
  let boxes : List (Nat × Nat) := match m.im2col with
    | none => (8, b[0]! - 1) :: (b.drop 1).map (fun x => (8, x - 1))
    | some x =>
      let cw := cornerWord m.rank x
      (8, x.channelsPerPixel - 1) :: (List.range 4).map fun k => (8, cw / 2 ^ (8 * k) % 256)
  [(64, m.globalAddress.toNat), (1, ic), (3, 0), (3, m.rank - 1), (4, dc),
   (2, index Interleave.all m.interleave), (2, sm), (1, index OobFill.all m.oobFill), (1, tf),
   (2, index L2Promotion.all m.l2Promotion), (3, sa), (1, 0), (9, 0)] ++
  st.map (fun x => (32, strideLo x)) ++ st.map (fun x => (4, strideHi x)) ++ [(16, 0)] ++
  g.map (fun x => (32, x - 1)) ++ e.map (fun x => (3, x - 1)) ++ (9, 0) :: boxes ++ im2colFields m.im2col

/-- The fields' widths. -/
def widths : List Nat :=
  [64, 1, 3, 3, 4, 2, 2, 1, 1, 2, 3, 1, 9] ++ List.replicate 4 32 ++ List.replicate 4 4 ++ [16] ++
  List.replicate 5 32 ++ List.replicate 5 3 ++ [9, 8] ++ List.replicate 4 8 ++ [10]

/-- Bytes in a descriptor. -/
def size : Nat := 128

def encode (m : TensorMap) : ByteArray := ⟨(le size (packBits m.fields)).toArray⟩

/-- The enum value a driver number names. -/
def ofIndex? {α : Type} (all : List α) (n : Nat) : Option α := all[n]?

def decode (b : ByteArray) : Option TensorMap :=
  match unpackBits widths (unle b.data.toList) with
  | addr :: fl :: _ :: rank :: dc :: il :: sm :: oob :: tf :: l2 :: sa :: b96 :: _ ::
    sl0 :: sl1 :: sl2 :: sl3 :: sh0 :: sh1 :: sh2 :: sh3 :: _ ::
    d0 :: d1 :: d2 :: d3 :: d4 :: e0 :: e1 :: e2 :: e3 :: e4 :: _ :: b0 :: b1 :: b2 :: b3 :: b4 ::
    ppc :: _ => do
    let stride (lo hi : Nat) : Nat := 16 * (lo + 2 ^ 32 * hi)
    return { dtype := ← dtypeOf? dc tf, rank := rank + 1, globalAddress := addr.toUInt64
             globalDim := #v[d0 + 1, d1 + 1, d2 + 1, d3 + 1, d4 + 1]
             globalStrides := #v[stride sl0 sh0, stride sl1 sh1, stride sl2 sh2, stride sl3 sh3]
             boxDim := if fl == 1 then #v[1, 1, 1, 1, 1] else #v[b0 + 1, b1 + 1, b2 + 1, b3 + 1, b4 + 1]
             elementStrides := #v[e0 + 1, e1 + 1, e2 + 1, e3 + 1, e4 + 1]
             interleave := ← ofIndex? Interleave.all il, swizzle := ← swizzleOf? sm sa b96
             l2Promotion := ← ofIndex? L2Promotion.all l2, oobFill := ← ofIndex? OobFill.all oob
             im2col := if fl == 1 then
               let cs := cornersOf (rank + 1) (b1 + 256 * b2 + 65536 * b3 + 16777216 * b4)
               some { lower := cs.1, upper := cs.2, channelsPerPixel := b0 + 1,
                      pixelsPerColumn := ppc + 1 } else none }
  | _ => none

/-- The fields `tensormap.replace` writes (PTX ISA 9.5 §9.7.10.29). -/
inductive Field where
  | globalAddress | rank | boxDim | globalDim | globalStride | elementStride | elemtype
  | interleaveLayout | swizzleMode | swizzleAtomicity | fillMode
  deriving DecidableEq, Repr, Inhabited

/-- `n` with its `w` bits from bit `off` set to `v`'s low `w` bits. -/
def setBits (n off w v : Nat) : Nat :=
  n - (n / 2 ^ off % 2 ^ w) * 2 ^ off + (v % 2 ^ w) * 2 ^ off

/-- `tensormap.replace` on a descriptor read as one number (its bytes little-endian): the bits
stored for the field at ordinal `ord`, from the instruction's `v` (Table 36's numbers for
the enumerations, the rank less one): an element type keeps the tf32 flag but for the tf32 types,
which set it; a swizzle mode clears the 96-byte flag but for 96 bytes, which sets it alone. -/
def replace (f : Field) (ord v n : Nat) : Except String Nat := do
  let ordinal (k : Nat) : Except String Unit :=
    if ord < k then pure () else throw s!"ordinal {ord} is not in [0..{k - 1}]"
  let bad : Except String Nat := throw s!"value {v} is not one Table 36 allows for this field"
  match f with
  | .globalAddress => return setBits n 0 64 v
  | .rank => return setBits n 68 3 v
  | .globalStride => do ordinal 4; return setBits (setBits n (96 + 32 * ord) 32 (v / 16)) (224 + 4 * ord) 4 (v / 16 / 2 ^ 32)
  | .globalDim => do ordinal 5; return setBits n (256 + 32 * ord) 32 (v - 1)
  | .elementStride => do ordinal 5; return setBits n (416 + 3 * ord) 3 (v - 1)
  | .boxDim => do ordinal 5; return setBits n (if ord == 0 then 440 else 448 + 8 * (ord - 1)) 8 (v - 1)
  | .elemtype =>
    if v ≤ 10 then return setBits n 71 4 v
    else if v ≤ 12 then return setBits (setBits n 71 4 (v - 4)) 80 1 1
    else if v ≤ 15 then return setBits n 71 4 (v - 2)
    else bad
  | .interleaveLayout => if v ≤ 2 then return setBits n 75 2 v else bad
  | .swizzleMode =>
    if v ≤ 3 then return setBits (setBits n 77 2 v) 86 1 0
    else if v == 4 then return setBits (setBits n 77 2 0) 86 1 1
    else bad
  | .swizzleAtomicity => if v ≤ 3 then return setBits n 83 3 v else bad
  | .fillMode => if v ≤ 1 then return setBits n 79 1 v else bad

/-- A tiled map whose every field fits its bits: rank 1 to 8, dimensions 1 to `2^32`, strides
multiples of 16 below `2^40`, boxes 1 to 256, element strides 1 to 8. A tiled map the driver
accepts fits (`invalid? m = none` bounds each used entry, the unused ones 1 and 0), and so does any `tensormap.replace` result within the PTX ISA's ranges. (An im2col
map's round trip is checked by the differential tests, not proved.) -/
def Fits (m : TensorMap) : Prop :=
  1 ≤ m.rank ∧ m.rank ≤ 8 ∧ (∀ x ∈ m.globalDim.toList, 1 ≤ x ∧ x ≤ 2 ^ 32) ∧
  (∀ x ∈ m.globalStrides.toList, x % 16 = 0 ∧ x < 2 ^ 40) ∧ (∀ x ∈ m.boxDim.toList, 1 ≤ x ∧ x ≤ 256) ∧
  (∀ x ∈ m.elementStrides.toList, 1 ≤ x ∧ x ≤ 8) ∧ m.im2col = none

/-! ## Round trip -/

theorem length_le (k n : Nat) : (le k n).length = k := by
  induction k generalizing n <;> simp [le, *]

theorem unle_le (k n : Nat) : unle (le k n) = n % 256 ^ k := by
  induction k generalizing n with
  | zero => simp [le, unle, Nat.mod_one]
  | succ k ih =>
    simp only [le, unle, ih, Nat.pow_succ']
    have : (n % 256).toUInt8.toNat = n % 256 := by simp
    rw [this, Nat.mod_mul]

theorem unpackBits_packBits (fs : List (Nat × Nat)) :
    unpackBits (fs.map (·.1)) (packBits fs) = fs.map fun (w, v) => v % 2 ^ w := by
  induction fs with
  | nil => simp [unpackBits]
  | cons f fs ih =>
    obtain ⟨w, v⟩ := f
    have hp : 0 < 2 ^ w := Nat.two_pow_pos w
    have hlt : v % 2 ^ w < 2 ^ w := Nat.mod_lt _ hp
    simp only [List.map, unpackBits, packBits]
    rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hlt, Nat.add_mul_div_left _ _ hp,
      Nat.div_eq_of_lt hlt, Nat.zero_add, ih]

theorem packBits_lt (fs : List (Nat × Nat)) : packBits fs < 2 ^ (fs.map (·.1)).sum := by
  induction fs with
  | nil => simp [packBits]
  | cons f fs ih =>
    obtain ⟨w, v⟩ := f
    have hlt : v % 2 ^ w < 2 ^ w := Nat.mod_lt _ (Nat.two_pow_pos w)
    simp only [packBits, List.map, List.sum_cons, Nat.pow_add]
    have h2 : 2 ^ w * (packBits fs + 1) ≤ 2 ^ w * 2 ^ (List.map (·.1) fs).sum := Nat.mul_le_mul_left _ ih
    rw [Nat.mul_add, Nat.mul_one] at h2
    omega

theorem vec3 {α : Type} (v : Vector α 3) : ∃ a b c, v = #v[a, b, c] := by
  refine ⟨v[0], v[1], v[2], Vector.ext fun i hi => ?_⟩
  match i, hi with
  | 0, _ | 1, _ | 2, _ => simp

theorem vec5 {α : Type} (v : Vector α 5) : ∃ a b c d e, v = #v[a, b, c, d, e] := by
  refine ⟨v[0], v[1], v[2], v[3], v[4], Vector.ext fun i hi => ?_⟩
  match i, hi with
  | 0, _ | 1, _ | 2, _ | 3, _ | 4, _ => simp

theorem vec4 {α : Type} (v : Vector α 4) : ∃ a b c d, v = #v[a, b, c, d] := by
  refine ⟨v[0], v[1], v[2], v[3], Vector.ext fun i hi => ?_⟩
  match i, hi with
  | 0, _ | 1, _ | 2, _ | 3, _ => simp

theorem widths_eq (m : TensorMap) (h : m.im2col = none) : m.fields.map (·.1) = widths := by
  rcases m with ⟨dt, rank, addr, gd, gs, bd, es, il, sw, l2, oob, i2c⟩
  cases h
  obtain ⟨g0, g1, g2, g3, g4, rfl⟩ := vec5 gd
  obtain ⟨s0, s1, s2, s3, rfl⟩ := vec4 gs
  obtain ⟨b0, b1, b2, b3, b4, rfl⟩ := vec5 bd
  obtain ⟨e0, e1, e2, e3, e4, rfl⟩ := vec5 es
  simp [fields, widths, im2colFields, List.replicate]

theorem dtypeOf_code (a : TmDType) : dtypeOf? ((dtypeCode a).1 % 16) ((dtypeCode a).2 % 2) = some a := by
  cases a <;> rfl
theorem swizzleOf_code (a : Swizzle) : swizzleOf? ((swizzleCode a).1 % 4) ((swizzleCode a).2 % 8) 0 = some a := by
  cases a <;> rfl
theorem ofIndex_interleave (a : Interleave) : ofIndex? Interleave.all (index Interleave.all a % 4) = some a := by
  cases a <;> rfl
theorem ofIndex_l2 (a : L2Promotion) : ofIndex? L2Promotion.all (index L2Promotion.all a % 4) = some a := by
  cases a <;> rfl
theorem ofIndex_oob (a : OobFill) : ofIndex? OobFill.all (index OobFill.all a % 2) = some a := by
  cases a <;> rfl

/-- A field of `w` bits holding `x - 1` gives `x` back, for `1 ≤ x ≤ 2^w`. -/
theorem pred_round (w x : Nat) (h1 : 1 ≤ x) (h2 : x ≤ 2 ^ w) : (x - 1) % 2 ^ w + 1 = x := by
  rw [Nat.mod_eq_of_lt (by omega)]; omega

/-- A stride split into its low word and high bits gives it back. -/
theorem stride_round (x : Nat) (h16 : x % 16 = 0) (h40 : x < 2 ^ 40) :
    16 * (strideLo x % 2 ^ 32 + 2 ^ 32 * (strideHi x % 2 ^ 4)) = x := by
  unfold strideLo strideHi
  have hq : x / 16 < 2 ^ 36 := by omega
  rw [Nat.mod_mod, Nat.mod_eq_of_lt (Nat.div_lt_of_lt_mul (by omega) : x / 16 / 2 ^ 32 < 2 ^ 4)]
  have := Nat.mod_add_div (x / 16) (2 ^ 32)
  have := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h16)
  omega

/-- A map whose fields fit their bits decodes from its encoding. -/
theorem decode_encode (m : TensorMap) (h : m.Fits) : decode (encode m) = some m := by
  have hw := widths_eq m h.2.2.2.2.2.2
  have hlt : packBits m.fields < 2 ^ (8 * size) := by
    have := packBits_lt m.fields
    rw [hw] at this
    exact Nat.lt_of_lt_of_le this (Nat.pow_le_pow_right (by decide) (by decide))
  have hd : unpackBits widths (unle (encode m).data.toList) = m.fields.map fun (w, v) => v % 2 ^ w := by
    have : unle (encode m).data.toList = packBits m.fields := by
      have h28 : (2 : Nat) ^ 8 = 256 := by decide
      have h256 : (256 : Nat) ^ size = 2 ^ (8 * size) := by rw [Nat.pow_mul, h28]
      simp only [encode, List.toList_toArray, unle_le]
      rw [h256]; exact Nat.mod_eq_of_lt hlt
    rw [this, ← hw]; exact unpackBits_packBits _
  rcases m with ⟨dt, rank, addr, gd, gs, bd, es, il, sw, l2, oob, i2c⟩
  obtain ⟨hr1, hr8, hg, hs, hb, he, hn⟩ := h
  simp only at hr1 hr8 hg hs hb he hn
  cases hn
  obtain ⟨g0, g1, g2, g3, g4, rfl⟩ := vec5 gd
  obtain ⟨s0, s1, s2, s3, rfl⟩ := vec4 gs
  obtain ⟨b0, b1, b2, b3, b4, rfl⟩ := vec5 bd
  obtain ⟨e0, e1, e2, e3, e4, rfl⟩ := vec5 es
  simp only [Vector.toList_mk, List.mem_cons, List.mem_nil_iff, or_false,
    forall_eq_or_imp, forall_eq] at hg hs hb he
  have ha : addr.toNat % 18446744073709551616 = addr.toNat := Nat.mod_eq_of_lt addr.toNat_lt_size
  unfold decode
  rw [hd]
  simp [fields, im2colFields, strideLo, strideHi, dtypeOf_code, swizzleOf_code, ofIndex_interleave,
    ofIndex_l2, ofIndex_oob, ha]
  omega

/-- Encodings are `size` bytes. -/
theorem size_encode (m : TensorMap) : (encode m).size = size := by
  simp [encode, ByteArray.size, length_le]

end TensorMap

end NV.Async
