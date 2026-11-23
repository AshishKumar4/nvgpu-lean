import NV.Async.WgmmaNum
import NV.Num.Scaled
import NV.Memory.Atomic
import NV.Async.MBarrierWord

/-!
# 5th-generation tensor cores: tensor memory and `tcgen05.mma`

What the data-center Blackwell SM (`sm_100a`) adds (PTX ISA 9.5 §9.7.18):

* **Tensor memory** (§9.7.18.1): per SM, 128 lanes of 512 columns of 32 bits. A `tcgen05`
  address is `lane << 16 | column`; the model keeps it in the `.tmem` space as one store per SM,
  column `c` of lane `l` at byte `4 (512 l + c)` (`byteAt`), followed by the word of its column
  allocator (`allocAddr`), one bit per 32-column granule (`UTCATOMSWS`, below).
* **The instruction descriptor** (§9.7.18.4.2, Table 51; CUTLASS `cute/arch/mma_sm100_desc.hpp`
  `InstrDescriptor`, v4.2.1): `IDesc`.
* **The shared-memory descriptor** (§9.7.18.4.1, Table 49): the wgmma descriptor's fields at the
  same bits, with a version field (bits 46–48, `0b001`) and a 3-bit swizzle field (bits 61–63)
  whose values 0, 2, 4 and 6 read as the wgmma field's 0–3 (`Wgmma.Desc.decode`); 1 (128-byte
  swizzling with 32-byte atomicity) is not modelled.
* **`tcgen05.mma`** of `.cta_group::1` (§9.7.18.10): `D = A·B + D` (or `A·B` without
  `enable-input-d`) for `M` of 64 or 128, A from shared memory or tensor memory, B from shared
  memory, D in tensor memory: row `i` of D in lane `i` (`M = 128`, Layout D, Figure 217) or lane
  `32 (i / 16) + i mod 16` plus the address's lane offset of 0 or 16 (`M = 64`, Layout F, Figures
  221–222), column `j` in the address's column plus `j`; a 16-bit D element in the low half of
  its word (§9.7.18.10.4.1). One `MmaT` per instruction, its products those of the B200's
  datapath (`NV.Num.Blackwell`, Khattak and Mikaitis).

`.cta_group::2` (a CTA pair, §9.7.18.2.1) with M = 256 (Layout A) or 128 (Layout B, dense A):
A, B and D split between the pair, the peer's shared memory through the cluster window and its
tensor memory past `2 ^ 24` (`peerByte`).

FP6 and FP4 inputs (`.kind::f8f6f4`) and block scaling (`.kind::mxf8f6f4`, `mxf4`, `mxf4nvf4`,
§9.7.18.10.7) on SF Tensor's measured datapaths (`Blackwell.f8f32`, `NV.Num.Scaled`), and
`tcgen05.cp` and `tcgen05.shift` (§9.7.18.9). Where the ISA draws no layout (`CpShape.lanes`,
`sfaByte`, the FP6/FP4 bit orders, the shift), the model's is the B200's, measured on one.

`tcgen05.mma.ws` (§9.7.18.10.10.3, weight stationary) of M = 32, 64 or 128 and N = 64, 128 or 256
(Table 48): D in Layout G, E or D (§9.7.18.10.5): with M = 64 a row `r` in lane `r` for D's first
N/2 columns and lane `64 + r` for the rest, with M = 32 in lane `r + 32 q` for the q-th quarter of
the columns, each part from D's column. The collector-buffer qualifiers change no result.

Not modelled, and a fault: `.cta_group::2` with A in tensor memory or with M = 128 and scale
factors, `.ws` with A in tensor memory, a zero-column mask that zeroes columns or shifts them, or
B's shifts, sparse A,
MN-major FP6 and FP4 operands, A's scale factors in 128 lanes, K of 96 or 128, `.kind::ti16`,
`tcgen05.cp` to a lane other than 0, the collector-buffer hints (which change no result).
-/

namespace NV.Async.Tcgen05

open NV NV.Num NV.Async.Wgmma

/-- Tensor memory's lanes and 32-bit columns per SM (§9.7.18.1). -/
def lanes : Nat := 128
def columns : Nat := 512

/-- Columns per allocation unit (§9.7.18.1.2: "The unit of allocation is 32 columns"). -/
def granule : Nat := 32

/-- The byte of the `.tmem` store holding column `c` of lane `l`. -/
def byteAt (l c : Nat) : Nat := 4 * (columns * l + c)

/-- The column allocator's word, after the data: bit `g` set when granule `g` is allocated. -/
def allocAddr : Nat := byteAt lanes 0

/-- The `.tmem` store's size. -/
def storeBytes : Nat := allocAddr + 4

theorem storeBytes_eq : storeBytes = Space.tmemBytes := by decide

/-- A `tcgen05` address's lane and column (§9.7.18.1.1). -/
def laneOf (a : Nat) : Nat := (a >>> 16) % 2 ^ 16
def colOf (a : Nat) : Nat := a % 2 ^ 16

/-- The instruction descriptor of `.kind::tf32`, `::f16`, `::f8f6f4` and `::i8` (Table 51). -/
structure IDesc where
  sparse : Bool
  saturate : Bool
  /-- D's type: 0 F16, 1 F32, 2 S32. -/
  dtype : Nat
  /-- A's and B's type codes, read by the instruction's kind (`inType`). -/
  atype : Nat
  btype : Nat
  negA : Bool
  negB : Bool
  /-- Transpose: the operand is MN-major. -/
  transA : Bool
  transB : Bool
  n : Nat
  m : Nat
  /-- `.ws`'s maximum shift (bits 30–31). -/
  maxShift : Nat
  deriving DecidableEq, Repr, Inhabited

def IDesc.decode (d : Nat) : IDesc :=
  let f (lo w : Nat) : Nat := (d >>> lo) % 2 ^ w
  { sparse := f 2 1 == 1, saturate := f 3 1 == 1, dtype := f 4 2, atype := f 7 3, btype := f 10 3,
    negA := f 13 1 == 1, negB := f 14 1 == 1, transA := f 15 1 == 1, transB := f 16 1 == 1,
    n := f 17 6 <<< 3, m := f 24 5 <<< 4, maxShift := f 30 2 }

/-- The kinds of `tcgen05.mma` the model has (§9.7.18.10): SASS `UTCHMMA` is `::f16` or `::tf32`
(the type codes tell them apart), `UTCQMMA` `::f8f6f4`, or `::mxf8f6f4` with scale factors,
`UTCIMMA` `::i8`, `UTCOMMA` `::mxf4` or `::mxf4nvf4`, the two encoded alike (`sm_100a`). -/
inductive Kind where
  | f16 | tf32 | f8f6f4 | i8 | mxf8f6f4 | mxf4 | mxf4nvf4
  deriving DecidableEq, Repr, Inhabited

/-- Whether the kind scales its operands by blocks (§9.7.18.10.7). -/
def Kind.scaled (k : Kind) : Bool := k == .mxf8f6f4 || k == .mxf4 || k == .mxf4nvf4

/-- Whether the kind's FP4 elements sit two to a byte (`mxf4`, `mxf4nvf4`, §9.7.18.10.4.5–6),
not one to an 8-bit container. -/
def Kind.packed (k : Kind) : Bool := k == .mxf4 || k == .mxf4nvf4

/-- An operand's element type, from its type code under the kind (Tables 51–53). -/
def inType : Kind → Nat → Except String InType
  | .f16, 0 => pure .f16
  | .f16, 1 => pure .bf16
  | .f16, 2 | .tf32, 2 => pure .tf32
  | .f8f6f4, 0 | .mxf8f6f4, 0 => pure .e4m3
  | .f8f6f4, 1 | .mxf8f6f4, 1 => pure .e5m2
  | .f8f6f4, 3 | .mxf8f6f4, 3 => pure .e2m3
  | .f8f6f4, 4 | .mxf8f6f4, 4 => pure .e3m2
  | .f8f6f4, 5 | .mxf8f6f4, 5 => pure .e2m1
  | .mxf4, 1 | .mxf4nvf4, 1 => pure .e2m1
  | .i8, 0 => pure .u8
  | .i8, 1 => pure .s8
  | k, c => throw s!"tcgen05.mma: type code {c} under .kind::{repr k}"

def accOf : Nat → Except String Acc
  | 0 => pure .f16
  | 1 => pure .f32
  | 2 => pure .s32
  | c => throw s!"tcgen05.mma: D type code {c}"

/-- The B200 datapath for an input and accumulator type (`NV.Num.Blackwell`). -/
def model : InType → Acc → Option TensorModel
  | .f16, .f32 | .bf16, .f32 => some Blackwell.f16f32.value
  | .f16, .f16 => some Blackwell.f16f16.value
  | .tf32, .f32 => some Blackwell.tf32f32.value
  | .e4m3, .f32 | .e5m2, .f32 => some Blackwell.f8f32.value
  | .e2m3, .f32 | .e3m2, .f32 | .e2m1, .f32 => some Blackwell.f8f32.value
  | _, _ => none

/-- How a block-scaled MMA scales (Tables 52–53, §9.7.18.10.7): its scale factors' format, how
many a row of A and a column of B have in one instruction (`.scale_vec::NX`), and the byte of a
tensor-memory word the first sits at (`SFA_ID`, `SFB_ID`). -/
structure ScaleForm where
  format : Format
  vec : Nat
  idA : Nat
  idB : Nat
  deriving Inhabited

/-- The instruction descriptor of a block-scaled kind (Tables 52–53) with `vec` scale factors a
row: `IDesc`, D in F32, and the scaling. Not modelled, and a fault: A's scale factors in 128
lanes (sm_107f), K other than the dense 32 (`mxf8f6f4`) or 64, UE5M3 scale factors. -/
def IDesc.decodeScaled (k : Kind) (vec d : Nat) : Except String (IDesc × ScaleForm) := do
  let f (lo w : Nat) : Nat := (d >>> lo) % 2 ^ w
  if f 26 1 == 1 then throw "tcgen05.mma: A's scale factors in 128 lanes (sm_107f) are not modelled"
  if f 31 1 == 1 || (k != .mxf8f6f4 && f 3 1 == 1) then throw "tcgen05.mma: this K is not modelled"
  let format ← match k, (if k == .mxf8f6f4 then f 23 1 else f 23 2) with
    | .mxf4nvf4, 0 => pure Format.ue4m3
    | _, 1 => pure Format.ue8m0
    | _, c => throw s!"tcgen05.mma: scale-factor type {c} under .kind::{repr k}"
  let (idA, idB) := (f 29 2, f 4 2)
  unless (idA + vec ≤ 4 && idB + vec ≤ 4 && idA % vec == 0 && idB % vec == 0) do
    throw s!"tcgen05.mma: scale-factor IDs {idA} and {idB} with {vec} factors a row"
  let idesc : IDesc :=
    { sparse := f 2 1 == 1, saturate := false, dtype := 1, atype := f 7 3,
      btype := if k == .mxf8f6f4 then f 10 3 else f 10 2,
      negA := f 13 1 == 1, negB := f 14 1 == 1, transA := f 15 1 == 1, transB := f 16 1 == 1,
      n := f 17 6 <<< 3, m := f 27 2 <<< 7, maxShift := 0 }
  return (idesc, { format, vec, idA, idB })

/-- A shared-memory matrix descriptor (Table 49). -/
def sdesc (d : UInt64) : Except String Desc := do
  -- Bits 46–48 are "fixed constant 0b001" (PTX ISA 9.5 §9.7.18.4.1); the hardware takes 0 as 1
  -- (inferred).
  if field d 46 3 > 1 then throw s!"tcgen05: descriptor version {field d 46 3}, not 0b001"
  if field d 52 1 != 0 then throw "tcgen05: absolute leading-dimension addresses (sm_103a) are not modelled"
  if field d 61 1 != 0 then throw "tcgen05: 128-byte swizzling with 32-byte atomicity is not modelled"
  return Desc.decode d

/-- Where an operand of `M` rows sits in tensor memory: the lane of row `i` for the address's
lane `l0` (Layouts D and F). -/
def rowLane (m l0 i : Nat) : Nat := if m == 128 then l0 + i else l0 + 32 * (i / 16) + i % 16

/-- Where A comes from. -/
inductive ASrc where
  | smem (d : Desc)
  /-- Tensor memory at a `tcgen05` address: row `i` in its lanes as D's, `32 / bits` elements a
  column from the address's column on, low element first (§9.7.18.10.4.2; inferred). -/
  | tmem (addr : Nat)
  deriving Inhabited

/-- What a CTA of a cluster holds at launch in its reserved shared memory for
`tcgen05.alloc.cta_group::2` (inferred): an mbarrier at 0x48 and its phase bit at 0x50; the even CTA
of the pair allocates once its wait on parity `[0x50]` passes, and the odd one waits on the same
for the column the even one sends. The state that runs that code as written: an mbarrier
expecting one arrival, its phase bit 1 in the even CTA (its first wait passes at once) and 0 in
the odd one (its first wait is for the even one's column). Words, in our mbarrier encoding
(`NV.Async.MBarrierWord`). -/
def reservedInit (rank : Nat) : Array (Nat × UInt32) :=
  let w := NV.Async.MBarrierWord.encode { phase := 0, pending := 1, expected := 1, tx := 0 }
  #[(0x48, w.toUInt32), (0x4c, (w >>> 32).toUInt32), (0x50, if rank % 2 == 0 then 1 else 0)]

/-- A byte of the tensor memory of the CTA of cluster rank `rank` (`.cta_group::2`'s peer): the
`.tmem` space's addresses past `2 ^ 24` are a peer's, rank plus one in bits 24 and up. -/
def peerByte (rank byte : Nat) : Nat := (rank + 1) * 2 ^ 24 + byte

/-- An element's shared address of the CTA pair, before it names a CTA: past `2 ^ 24`, the peer's. -/
def peerShared : Nat := 2 ^ 24

/-- A block-scaled MMA's scale factors: their form and the `tcgen05` addresses of A's and B's. -/
structure Scales where
  form : ScaleForm
  a : Nat
  b : Nat
  deriving Inhabited

/-- One `tcgen05.mma`. -/
structure Inst where
  kind : Kind
  idesc : IDesc
  a : ASrc
  b : Desc
  /-- D's `tcgen05` address. -/
  d : Nat
  /-- `enable-input-d`: accumulate into D. -/
  accumulate : Bool
  /-- `.cta_group::2`: the issuing CTA's cluster rank and its peer's (§9.7.18.2.1, the CTA pair).
  The model has M = 256 (Layout A, §9.7.18.10.5.1, Figure 211): rows 0–127 of A and D in the
  issuer's shared and tensor memory (A in either), rows 128–255 in the peer's, at the same addresses; B's
  columns split the same way, its first N/2 the issuer's (CUTLASS `SM100_MMA_F16BF16_2x1SM_SS`,
  which loads each CTA's half of B). And M = 128 (Layout B, §9.7.18.10.5.2, Figures 213–214): A's
  rows 0–63 and D's in the issuer, 64–127 in the peer, B split as for 256; in each CTA a D row `r`
  (of its 64) is in lane `r` for D's first N/2 columns and lane `64 + r` for the rest, at
  columns from D's own (the second half's from D's column again). -/
  pair : Option (Nat × Nat) := none
  /-- `tcgen05.mma.ws`. -/
  ws : Bool := false
  /-- `.ws`'s zero-column mask descriptor (§9.7.18.4.3); 0, none. -/
  zeroMask : Nat := 0
  /-- The scale factors of a block-scaled kind. -/
  scales : Option Scales := none
  deriving Inhabited

/-- 16-byte chunks of tensor memory: each address four columns of one lane. -/
def tmemReqs (op : MemOp) (addrs : Array Nat) (data : Array UInt32 := #[]) : Array Req :=
  (Array.range ((addrs.size + 31) / 32)).map fun i =>
    let part := addrs.extract (32 * i) (32 * i + 32)
    { asyncReq .tmem op 16 (part.map fun a => some a.toUInt64) with
      data := data.extract (128 * i) (128 * i + 4 * part.size) }

/-- The `width`-bit field at bit `off` of the 16-byte chunk at `a` of an image. -/
def chunkBits (img : Std.HashMap Nat (Array UInt32)) (a off width : Nat) : Nat :=
  let ws := img.getD (a / 16 * 16) #[0, 0, 0, 0]
  let v := (List.range 4).foldl (fun acc i => acc ||| ((ws[i]?.getD 0).toNat <<< (32 * i))) 0
  (v >>> off) % 2 ^ width

/-- The tensor-memory byte of a scale factor word (§9.7.18.10.7.2–3, Figures 237–259): row `r` of
A (of the CTA's 128) has its factors in lane `r` of the column `r / 32` past A's address; column
`n` of B in lane `32 q + n % 32` of the column `n / 32` past B's, for the rows of D in lane
quarter `q`. The ISA asks B's factors (and on `sm_100a` A's) copied into all four quarters, and
each quarter reads its own copy: so on the B200. -/
def sfaByte (a r : Nat) : Nat := byteAt (laneOf a + r) (colOf a + r / 32)
def sfbByte (b r n : Nat) : Nat := byteAt (laneOf b + 32 * (r / 32) + n % 32) (colOf b + n / 32)

/-- The SASS front end's `.sharedCluster` address of offset `a` of the CTA of cluster rank `rank`:
the rank in bits 24 and up (`NV.SASS.sharedAt`). -/
def rankAddr (rank a : Nat) : UInt64 := (rank * 2 ^ 24 + a % 2 ^ 24).toUInt64

/-- The operation of an instruction: its loads (A's and B's shared chunks through `addr`, A's
tensor-memory chunks, D's when accumulating, then the scale factors' chunks), its result (D's
`M × N` words, row-major) and its stores of the result.

FP6 and FP4 operands are K-major: in shared memory 16 to a 16-byte chunk, element `k % 16` at
bit `width · (k % 16)` (§9.7.18.10.4.4, Figures 207–209; the bit order the B200's), or under
`mxf4`/`mxf4nvf4` 32 to a chunk, two to a byte, low first (§9.7.18.10.4.6); in tensor memory each
in its own byte, an FP4 one in bits 2–5 (§9.7.18.10.4.3, Figures 202–204), or under `mxf4` two to
a byte, low first (Figure 210; the order the B200's). A block-scaled kind's element `k` of a row of
A takes the row's scale factor `k / (K / vec)` (the byte past `SFA_ID`), likewise B's columns. -/
def Inst.op (x : Inst) (addr : Nat → UInt64) (cluster : Nat → Nat → UInt64 := rankAddr) :
    Except String MmaT := do
  let idc := x.idesc
  if idc.sparse then throw "tcgen05.mma: sparse A is not modelled"
  if idc.maxShift != 0 then throw "tcgen05.mma: .ws shifts are not modelled"
  -- A zero-column mask with bit 39 clear generates no zero column; one shifting B is not either.
  if (x.zeroMask >>> 39) % 2 != 0 || (x.zeroMask >>> 56) % 64 != 0 then
    throw "tcgen05.mma.ws: a zero-column mask that zeroes or shifts columns is not modelled"
  if x.kind.scaled != x.scales.isSome then throw "tcgen05.mma: scale factors with a kind that does not take them"
  let ta ← inType x.kind idc.atype
  let tb ← inType x.kind idc.btype
  let acc ← accOf idc.dtype
  let float := x.kind != .i8
  if float && (model ta acc).isNone then throw s!"tcgen05.mma: {repr ta} into {repr acc} is not modelled"
  if !float && acc != .s32 then throw "tcgen05.mma: .kind::i8 accumulates into S32"
  let M := idc.m
  let N := idc.n
  if x.ws then
    if x.pair.isSome then throw "tcgen05.mma.ws: .cta_group::2"
    if x.kind.scaled then throw "tcgen05.mma.ws: a block-scaled kind"
    unless M == 32 || M == 64 || M == 128 do throw s!"tcgen05.mma.ws: M = {M}"
    unless N == 64 || N == 128 || N == 256 do throw s!"tcgen05.mma.ws: N = {N}"
    if (match x.a with | .tmem _ => true | .smem _ => false) then
      throw "tcgen05.mma.ws: A in tensor memory is not modelled"
  else if x.pair.isSome then
    unless M == 256 || M == 128 do throw s!"tcgen05.mma.cta_group::2: M = {M}"
    if M == 128 && x.scales.isSome then throw "tcgen05.mma.cta_group::2: block scaling with M = 128"
    if M == 128 && (match x.a with | .tmem _ => true | .smem _ => false) then
      throw "tcgen05.mma.cta_group::2: A in tensor memory with M = 128"
  else
    unless M == 64 || M == 128 do throw s!"tcgen05.mma.cta_group::1: M = {M}"
    if x.scales.isSome && M != 128 then throw s!"tcgen05.mma: block scaling with M = {M}"
  unless N % (if M == 64 then 8 else 16) == 0 && 8 ≤ N && N ≤ 256 do throw s!"tcgen05.mma: N = {N} for M = {M}"
  let K := if x.kind.packed then 64 else ta.k
  let sub (t : InType) : Bool := t.width < 8
  if (sub ta && idc.transA) || (sub tb && idc.transB) then throw "tcgen05.mma: MN-major FP6 and FP4 operands are not modelled"
  -- Elements of an FP6 or FP4 operand per 16-byte chunk of shared memory.
  let perChunk := if x.kind.packed then 32 else 16
  let (dl, dc) := (laneOf x.d, colOf x.d)
  unless (if M == 64 && !x.ws then dl == 0 || dl == 16 else dl == 0) do throw s!"tcgen05.mma: D at lane {dl} for M = {M}"
  -- Layout B folds D's second half of columns into lanes 64–127.
  -- Layouts B and E fold D's second half of columns into lanes 64–127, G its quarters into the
  -- lanes' quarters.
  let parts := if (x.pair.isSome && M == 128) || (x.ws && M == 64) then 2 else if x.ws && M == 32 then 4 else 1
  let dCols := N / parts
  unless dc + dCols ≤ columns do throw s!"tcgen05.mma: D's columns {dc}…{dc + dCols} pass column {columns}"
  -- A pair's second half is the peer's, at the same addresses (`peerShared` on).
  let half (n i : Nat) : Nat × Nat :=
    if x.pair.isSome && i ≥ n / 2 then (peerShared, i - n / 2) else (0, i)
  -- The shared address of element `k` of row `mn`: an FP6 or FP4 element's chunk.
  let at_ (d : Desc) (t : InType) (kMajor : Bool) (mn k : Nat) : Except String Nat :=
    if sub t then d.addr .u8 true mn (16 * (k / perChunk)) else d.addr t kMajor mn k
  let aAddrs ← match x.a with
    | .smem d => ((Array.range M).flatMap fun r => (Array.range K).map fun k =>
        let (o, r) := half M r; (at_ d ta (!idc.transA) r k).map (o + ·)).mapM id
    | .tmem _ => pure #[]
  let bAddrs ← ((Array.range N).flatMap fun c => (Array.range K).map fun k =>
    let (o, c) := half N c; (at_ x.b tb (!idc.transB) c k).map (o + ·)).mapM id
  let cs := chunks (aAddrs ++ bAddrs)
  -- A in tensor memory: per row, the columns holding its K elements, 16 bytes a chunk; FP4 two to
  -- a byte under `mxf4`.
  let aBits := if x.kind.packed then 4 else ta.bits
  let per := 32 / aBits
  let aCols := K / per
  let aT : Array Nat ← match x.a with
    | .tmem a =>
      let (al, ac) := (laneOf a, colOf a)
      unless al == dl do throw s!"tcgen05.mma: A at lane {al}, D at lane {dl}"
      -- A pair's A as its D (Layout A): rows 0–127 in the issuer's lanes, 128–255 in the peer's,
      -- at the same address (inferred).
      let aByte (r q : Nat) : Nat := match x.pair with
        | some (_, peer) =>
          if r ≥ 128 then peerByte peer (byteAt (al + r - 128) (ac + 4 * q)) else byteAt (al + r) (ac + 4 * q)
        | none => byteAt (rowLane M al r) (ac + 4 * q)
      pure ((Array.range M).flatMap fun r => (Array.range (aCols / 4)).map fun q => aByte r q)
    | .smem _ => pure #[]
  -- D: per row, its N columns, 16 bytes a chunk; a pair's rows 128 on in the peer's.
  let dByte (r q : Nat) : Nat := match x.pair with
    | some (_, peer) =>
      if M == 128 then
        let r' := r % 64
        let (lane, col) := if 4 * q ≥ N / 2 then (64 + r', dc + 4 * q - N / 2) else (r', dc + 4 * q)
        if r ≥ 64 then peerByte peer (byteAt lane col) else byteAt lane col
      else if r ≥ 128 then peerByte peer (byteAt (r - 128) (dc + 4 * q)) else byteAt r (dc + 4 * q)
    | none =>
      if parts == 1 then byteAt (rowLane M dl r) (dc + 4 * q)
      else
        let part := 4 * q / dCols
        byteAt (r + (128 / parts) * part) (dc + 4 * q - part * dCols)
  let dChunks := (Array.range M).flatMap fun r => (Array.range (N / 4)).map fun q => dByte r q
  let cT := if x.accumulate then dChunks else #[]
  -- Scale factors: the words of A's rows and of B's columns for each lane quarter, a pair's rows
  -- 128 on in the peer's.
  let ctaByte (r : Nat) (b : Nat) : Nat := match x.pair with
    | some (_, peer) => if r ≥ 128 then peerByte peer b else b
    | none => b
  let quarters := (Array.range (M / 32)).map (32 * ·)
  let sT : Array Nat := match x.scales with
    | some sc => chunks ((Array.range M).map (fun r => ctaByte r (sfaByte sc.a (r % 128))) ++
        quarters.flatMap fun r => (Array.range N).map fun n => ctaByte r (sfbByte sc.b (r % 128) n))
    | none => #[]
  -- A pair's shared loads name a CTA through the front end's cluster window (`cluster`).
  let sLoads := match x.pair with
    | some (own, peer) => (loadReqs cs fun c =>
        if c ≥ peerShared then cluster peer (addr (c - peerShared)).toNat
        else cluster own (addr c).toNat).map ({ · with space := .sharedCluster })
    | none => loadReqs cs addr
  let loads := sLoads ++ tmemReqs .ld aT ++ tmemReqs .ld cT ++ tmemReqs .ld sT
  let neg (t : InType) (on : Bool) (v : Nat) : Nat := t.negate (float && on) v
  let get (img : Std.HashMap Nat (Array UInt32)) (t : InType) (a k : Nat) : Nat :=
    if sub t then chunkBits img a (t.width * (k % perChunk)) t.width else elem img t.bits a
  let result (data : Resp) : Array UInt32 :=
    let img := image cs data
    let at0 := 4 * cs.size
    let ct0 := at0 + 4 * aT.size
    let st0 := ct0 + 4 * cT.size
    let aRow : Array (Array Nat) := match x.a with
      | .smem _ => (Array.range M).map fun r => (Array.range K).map fun k =>
          neg ta idc.negA (get img ta aAddrs[r * K + k]! k)
      | .tmem _ => (Array.range M).map fun r => (Array.range K).map fun k =>
          let w := (data[at0 + r * aCols + k / per]?.getD 0).toNat
          let v := (w >>> (aBits * (k % per))) % 2 ^ aBits
          -- An FP4 element in its own byte sits in bits 2–5, an FP6 one in bits 0–5.
          neg ta idc.negA (if aBits == 8 && sub ta then (v >>> (if ta == .e2m1 then 2 else 0)) % 2 ^ ta.width else v)
    let bs : Array (Array Nat) := (Array.range N).map fun c => (Array.range K).map fun k =>
      neg tb idc.negB (get img tb bAddrs[c * K + k]! k)
    let cm : Array Nat := (Array.range (M * N)).map fun e =>
      if x.accumulate then
        let w := (data[ct0 + e]?.getD 0).toNat
        if acc == .f16 then w % 2 ^ 16 else w
      else 0
    let dm : Array Nat := match model ta acc, x.scales with
      | some md, some sc =>
        let simg := image sT (data.extract st0 (st0 + 4 * sT.size))
        let byte (b j : Nat) : Nat := ((simg.getD (b / 16 * 16) #[0, 0, 0, 0])[(b % 16) / 4]!.toNat >>> (8 * j)) % 256
        let sF := sc.form.format
        -- Row `r`'s factors, and column `n`'s as the rows of `r`'s lane quarter see them.
        let fa (r : Nat) : Array Nat := (Array.range sc.form.vec).map fun j =>
          byte (ctaByte r (sfaByte sc.a (r % 128))) (sc.form.idA + j)
        let fb (r n : Nat) : Array Nat := (Array.range sc.form.vec).map fun j =>
          byte (ctaByte r (sfbByte sc.b (r % 128) n)) (sc.form.idB + j)
        let blk := K / sc.form.vec
        if x.kind.packed then
          let kd : Scaled.Kind := { scaleBlock := blk, nvfp4 := sF == Format.ue4m3 }
          let av := aRow.map (·.map Format.e2m1.decode)
          let bv := bs.map (·.map Format.e2m1.decode)
          let fav := (Array.range M).map fun r => (fa r).map sF.decode
          let fbv := quarters.map fun r => (Array.range N).map fun n => (fb r n).map sF.decode
          (Array.range (M * N)).map fun e =>
            Scaled.dot kd sF av[e / N]! bv[e % N]! fav[e / N]! fbv[e / N / 32]![e % N]! cm[e]!
        else
          let fA : BlockFormat := { elem := ta.format, scale := sF, size := blk }
          let fB : BlockFormat := { elem := tb.format, scale := sF, size := blk }
          let av := (Array.range M).map fun r => (Array.range K).map fun k =>
            fA.value (fa r)[k / blk]! aRow[r]![k]!
          let bv := quarters.map fun r => (Array.range N).map fun n => (Array.range K).map fun k =>
            fB.value (fb r n)[k / blk]! bs[n]![k]!
          (Array.range (M * N)).map fun e =>
            md.dotVals (ta.format.man + sF.man) (tb.format.man + sF.man) Format.f32 av[e / N]!
              bv[e / N / 32]![e % N]! cm[e]!
      | some md, none => md.tile ta.format tb.format (if acc == .f16 then Format.f16 else Format.f32) aRow bs cm
      | none, _ =>
        let sg (t : InType) (v : Nat) : Int := if t == .s8 then signed8 v else v
        (Array.range (M * N)).map fun e =>
          let cv : Nat := cm[e]!
          let cInt : Int := if cv ≥ 2 ^ 31 then (cv : Int) - 2 ^ 32 else (cv : Int)
          let d := dotInt (aRow[e / N]!.map (sg ta)) (bs[e % N]!.map (sg tb)) cInt idc.saturate
          (if d < 0 then d + 2 ^ 32 else d).toNat
    dm.map fun v => (if acc == .f16 then v % 2 ^ 16 else v).toUInt32
  return { loads, result, stores := fun d => tmemReqs .st dChunks d, macs := M * N * K, n := N,
           inBits := ta.width, dKey := x.d
           accIn := if x.accumulate then some (4 * cs.size + 4 * aT.size, 4 * cT.size) else none }

/-! ## Copies into tensor memory and row shifts (§9.7.18.9)

Both are asynchronous `tcgen05` operations of their issuer (`MmaT`, without multiply-adds), in
issue order with its MMAs and tracked by its `tcgen05.commit`. -/

/-- `tcgen05.cp`'s shapes with their multicast (§9.7.18.2.3, §9.7.18.9.2). -/
inductive CpShape where
  | s128x256b | s4x256b | s128x128b
  /-- `.64x128b.warpx2::02_13` and `.64x128b.warpx2::01_23`. -/
  | s64x128b02_13 | s64x128b01_23
  /-- `.32x128b.warpx4`. -/
  | s32x128bx4
  deriving DecidableEq, Repr, Inhabited

/-- Rows of the source. -/
def CpShape.rows : CpShape → Nat
  | .s128x256b | .s128x128b => 128
  | .s4x256b => 4
  | .s64x128b02_13 | .s64x128b01_23 => 64
  | .s32x128bx4 => 32

/-- Bytes of a source row. -/
def CpShape.rowBytes : CpShape → Nat
  | .s128x256b | .s4x256b => 32
  | _ => 16

/-- The lanes source row `r` lands in (CUTLASS's traits of the `SM100_UTCCP_*` atoms,
`cute/atom/copy_traits_sm100.hpp` v4.2.1, and the B200's): `.4x256b` refreshes the first
lane of each warp's quarter; `.warpx2::02_13` sends rows 0–31 to warps 0 and 2 and rows 32–63 to
warps 1 and 3, `::01_23` rows 0–31 to warps 0 and 1 and rows 32–63 to warps 2 and 3; `.warpx4`
sends every row to all four. -/
def CpShape.lanes : CpShape → Nat → List Nat
  | .s128x256b, r | .s128x128b, r => [r]
  | .s4x256b, r => [32 * r]
  | .s64x128b02_13, r => [r, r + 64]
  | .s64x128b01_23, r => [64 * (r / 32) + r % 32, 64 * (r / 32) + 32 + r % 32]
  | .s32x128bx4, r => [r, r + 32, r + 64, r + 96]

/-- `tcgen05.cp`'s optional decompression (§9.7.18.9.1): each 16-byte chunk of the source holds
16 elements of 6 bits (`.b6x16_p32`, in its first 12 bytes) or of 4 bits (`.b4x16_p64`, its first
8), and becomes 16 bytes, a 6-bit element in bits 0–5 of its byte (Figures 199–201), a 4-bit one
in bits 2–5 (Figures 197–198). Element `i` sits at bit `6 i` or `4 i` of the chunk (the figures
draw the elements in order, not their bits; the B200 places them so). -/
inductive Unpack where
  | none | b6 | b4
  deriving DecidableEq, Repr, Inhabited

/-- A 16-byte chunk (four words, low first) as `u` leaves it. -/
def Unpack.chunk (u : Unpack) (ws : Array UInt32) : Array UInt32 :=
  let (bits, sh) : Nat × Nat := match u with
    | .none => (8, 0)
    | .b6 => (6, 0)
    | .b4 => (4, 2)
  if u == .none then ws else
  let v := (List.range 4).foldl (fun acc i => acc ||| ((ws[i]?.getD 0).toNat <<< (32 * i))) 0
  (Array.range 4).map fun w => ((List.range 4).foldl (fun acc j =>
    acc ||| ((((v >>> (bits * (4 * w + j))) % 2 ^ bits) <<< sh) <<< (8 * j))) 0).toUInt32

/-- The CTAs an operation of `.cta_group::2` touches, each by the offset its shared and tensor
addresses take (`none` the issuer's own). -/
private def pairCtas (pair : Option (Nat × Nat)) : Array (Option Nat) :=
  match pair with
  | some (_, peer) => #[none, some peer]
  | none => #[none]

/-- A tensor-memory byte of the issuer's CTA (`none`) or of its peer. -/
private def tmemOf (cta : Option Nat) (b : Nat) : Nat :=
  match cta with
  | some peer => peerByte peer b
  | none => b

/-- Shared loads of the chunks `cs` in each CTA of `ctas`, one after another (`Inst.op`'s
addressing: a pair's name their CTA through the cluster window). -/
private def ctaLoads (pair : Option (Nat × Nat)) (cs : Array Nat) (addr : Nat → UInt64)
    (cluster : Nat → Nat → UInt64) : Array Req :=
  (pairCtas pair).flatMap fun cta => match pair, cta with
    | some (own, _), c =>
      let rank := c.getD own
      (loadReqs cs fun a => cluster rank (addr a).toNat).map ({ · with space := .sharedCluster })
    | none, _ => loadReqs cs addr

/-- One `tcgen05.cp` (§9.7.18.9.2): `.cta_group::2` also copies the peer's shared memory into
its tensor memory, at the same addresses. -/
structure Cp where
  shape : CpShape
  unpack : Unpack
  src : Desc
  /-- The destination's `tcgen05` address. -/
  dst : Nat
  pair : Option (Nat × Nat) := none
  deriving Inhabited

/-- The operation of a copy: source row `r`'s bytes are those of row `r` of a K-major operand of
8-bit elements through the descriptor (`Desc.addr`), stored in each of its lanes
(`CpShape.lanes`) from the address's column on. The address's lane is not used: on the B200 a
`.4x256b` copy to lane 32 lands where one to lane 0 does. -/
def Cp.op (x : Cp) (addr : Nat → UInt64) (cluster : Nat → Nat → UInt64 := rankAddr) :
    Except String MmaT := do
  let c0 := colOf x.dst
  let n := x.shape.rowBytes / 16
  unless c0 + 4 * n ≤ columns do throw s!"tcgen05.cp: columns {c0}…{c0 + 4 * n} pass column {columns}"
  let src ← ((Array.range x.shape.rows).flatMap fun r =>
    (Array.range n).map fun j => x.src.addr .u8 true r (16 * j)).mapM id
  let cs := chunks src
  -- Per CTA, per row, per lane of the row, per chunk: the chunk's destination.
  let place := (pairCtas x.pair).flatMap fun cta => (Array.range x.shape.rows).flatMap fun r =>
    (x.shape.lanes r).toArray.flatMap fun l => (Array.range n).map fun j =>
      (cta, r * n + j, tmemOf cta (byteAt l (c0 + 4 * j)))
  let ctas := pairCtas x.pair
  let result (data : Resp) : Array UInt32 :=
    let imgs := (Array.range ctas.size).map fun i =>
      image cs (data.extract (4 * cs.size * i) (4 * cs.size * (i + 1)))
    place.flatMap fun (cta, s, _) =>
      let img := imgs[if cta.isSome then 1 else 0]!
      x.unpack.chunk (img.getD (src[s]! / 16 * 16) #[0, 0, 0, 0])
  return { loads := ctaLoads x.pair cs addr cluster, result, stores := fun d => tmemReqs .st (place.map (·.2.2)) d,
           macs := 0, n := 0, inBits := 8 }

/-- One `tcgen05.shift.down` (§9.7.18.9.3: "shifting of 32-byte elements downwards across all the
rows, except the last, by one row"), as the B200 does it: at the
address's column and the seven after, in each lane quarter, lane `32 q + i` takes lane
`32 q + i + 1`'s element for `i` < 31 and the quarter's last lane keeps its own. Every quarter
shifts whatever the address's lane (a multiple of 32). `.cta_group::2` shifts the peer's tensor
memory too. -/
def shiftOp (dst : Nat) (pair : Option (Nat × Nat)) : Except String MmaT := do
  let (l0, c0) := (laneOf dst, colOf dst)
  unless l0 % 32 == 0 && l0 < lanes do throw s!"tcgen05.shift: lane {l0} is not a multiple of 32 below {lanes}"
  unless c0 + 8 ≤ columns do throw s!"tcgen05.shift: columns {c0}…{c0 + 8} pass column {columns}"
  let at_ (d : Nat) : Array Nat := (pairCtas pair).flatMap fun cta =>
    (Array.range 4).flatMap fun q => (Array.range 31).flatMap fun i =>
      #[0, 4].map fun j => tmemOf cta (byteAt (32 * q + i + d) (c0 + j))
  return { loads := tmemReqs .ld (at_ 1), result := id, stores := fun d => tmemReqs .st (at_ 0) d,
           macs := 0, n := 0, inBits := 8 }

/-! ## Loads and stores between registers and tensor memory (§9.7.18.8) -/

/-- The shapes of `tcgen05.ld`/`tcgen05.st` (§9.7.18.2.3). -/
inductive Shape where
  | s32x32b | s16x64b | s16x128b | s16x256b | s16x32bx2
  deriving DecidableEq, Repr, Inhabited

/-- Registers per thread for `.num = x1` (Table 59). -/
def Shape.regs : Shape → Nat
  | .s16x128b => 2
  | .s16x256b => 4
  | _ => 1

/-- Where register `i` of thread `t` sits for a load or store of `num` repeats at lane `l0`,
column `c0` (Figures 186–190): its lane and column. `half` is `.16x32bx2`'s second-half column
offset. -/
def Shape.pos (s : Shape) (half t i : Nat) (l0 c0 : Nat) : Nat × Nat :=
  match s with
  | .s32x32b => (l0 + t, c0 + i)
  -- 64 bits a row: threads 4j and 4j + 2 in row j, 4j + 1 and 4j + 3 in row 8 + j.
  | .s16x64b => (l0 + t / 4 + 8 * (t % 2), c0 + 2 * i + (t / 2) % 2)
  -- 128 bits a row: thread 4j + q in column q of row j (r0) and of row 8 + j (r1).
  | .s16x128b => (l0 + t / 4 + 8 * (i % 2), c0 + 4 * (i / 2) + t % 4)
  -- 256 bits a row: thread 4j + q in columns 2q, 2q + 1 of row j (r0, r1) and of row 8 + j (r2, r3).
  | .s16x256b => (l0 + t / 4 + 8 * ((i % 4) / 2), c0 + 8 * (i / 4) + 2 * (t % 4) + i % 2)
  -- Threads 0–15 in lanes 0–15 from the base, threads 16–31 in lanes 16–31 from the base plus
  -- `half` columns.
  | .s16x32bx2 => (l0 + t, c0 + i + (if t < 16 then 0 else half))

/-- The first granule `UTCATOMSWS.FIND_AND_SET.ALIGN` claims for `n` granules from the allocator
word it read (`AtomOp.findSetAlign`). -/
def findSetAlign? (old : UInt32) (n : Nat) : Option Nat := NV.findSetAlign old n

/-- The lanes of tensor memory warp `w` of a warpgroup may access (§9.7.18.8.1). -/
def warpLanes (w : Nat) : Nat × Nat := (32 * (w % 4), 32 * (w % 4) + 32)

/-- The request of a load or store of `num` repeats of shape `s` at `addr` by warp `warp`: one
slot per thread and register (thread-major), 4 bytes each, those of threads outside `threads`
inactive; with `.pack::16b`/`.unpack::16b` (`pack`) two 2-byte slots a register, the low halves of
two adjacent columns, low first (Figure 196; for the other shapes, inferred). `data` is what a store
writes, the same order. Also the slots a thread has. -/
def moveReq (op : MemOp) (s : Shape) (num half : Nat) (addr : Nat) (warp : Nat) (pack : Bool)
    (data : Array UInt32 := #[]) (threads : Nat → Bool := fun _ => true) :
    Except String (Req × Nat) := do
  let (lo, hi) := warpLanes warp
  -- An address in lanes 0–31 names the warp's own quarter; one in another quarter must be the
  -- warp's (inferred).
  let (l0, c0) := (if laneOf addr < 32 then lo + laneOf addr else laneOf addr, colOf addr)
  let regs := s.regs * num
  -- Packed, a register's place in the shape is a pair of adjacent 16-bit columns: the shape laid
  -- over column pairs, its column `c0 + k` the columns `c0 + 2k` (low half) and `c0 + 2k + 1`.
  let pos (t i : Nat) : Array (Nat × Nat) :=
    let (l, c) := s.pos half t i l0 c0
    if pack then #[(l, c0 + 2 * (c - c0)), (l, c0 + 2 * (c - c0) + 1)] else #[(l, c)]
  let slots := (Array.range 32).flatMap fun t => (Array.range regs).flatMap fun i =>
    (pos t i).map fun x => if threads t then some x else none
  for x in slots do
    if let some (l, c) := x then
      unless lo ≤ l && l < hi do throw s!"tcgen05.ld/st: lane {l} is outside warp {warp % 4}'s lanes {lo}…{hi - 1}"
      unless c < columns do throw s!"tcgen05.ld/st: column {c}"
  let words := if pack then data.flatMap fun (w : UInt32) => #[w &&& (0xFFFF : UInt32), w >>> (16 : UInt32)] else data
  return ({ space := .tmem, op, bytes := if pack then 2 else 4,
            addrs := slots.map (·.map fun (l, c) => (byteAt l c).toUInt64),
            data := if op == .st then words else #[] }, regs)

/-- A load's registers from its response (`moveReq`'s slots, one word each): thread `t`'s
`regs` words, packing pairs when `pack`. -/
def unpackLoad (resp : Resp) (t regs : Nat) (pack : Bool) : Array UInt32 :=
  if pack then (Array.range regs).map fun i =>
      let lo : UInt32 := resp[2 * (t * regs + i)]?.getD 0
      let hi : UInt32 := resp[2 * (t * regs + i) + 1]?.getD 0
      (lo &&& (0xFFFF : UInt32)) ||| (hi <<< (16 : UInt32))
  else (Array.range regs).map fun i => resp[t * regs + i]?.getD 0

end NV.Async.Tcgen05
