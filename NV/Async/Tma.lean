import NV.Async.TensorMap
import NV.Async.Op

/-!
# Tensor and bulk copies

Which global bytes a TMA copy moves to which shared bytes (PTX ISA 9.5 §5.5): the tiled mode's
bounding box at the instruction's coordinates, traversed with the element strides, laid out
densely in shared memory with dimension 0 fastest, then swizzled (§5.5.7); elements outside the
tensor read as zero (§5.5.3.3) and are not written by stores. A bulk copy (`cp.async.bulk`) is
the one-dimensional case without a tensor map.

A copy is cut into `Piece`s: 16-byte chunks of a box row, or single elements where a chunk
straddles the tensor's boundary. `loadCopy` and `storeCopy` pack the pieces into requests of up
to 32 lanes, one global 128-byte line per request, as the TMA unit issues them.

The swizzle is a function of the shared address (the tables of §5.5.7 start at 256-, 512- and
1024-byte boundaries, and their "base offset" is the address's position within that pattern),
the same pattern wgmma's matrix descriptors name (CuTe's `Swizzle<B,4,3>`, §9.7.17.5.1.2).
-/

namespace NV.Async

/-- The shared byte address at which the swizzle mode stores the byte the unswizzled layout
puts at `a`: 16-byte chunks (`b32`, `b64`, `b128`) or 32- and 64-byte ones (the `Atom` modes)
are permuted within a 128-byte line by XOR with the line's index in the repeating pattern
(§5.5.7's tables); `b128Atom32Flip8` also swaps the 8-byte halves of chunks on odd lines. -/
def Swizzle.apply (s : Swizzle) (a : Nat) : Nat :=
  let line := a >>> 7
  match s with
  | .none => a
  | .b32 => a ^^^ ((line &&& 1) <<< 4)
  | .b64 => a ^^^ ((line &&& 3) <<< 4)
  | .b128 => a ^^^ ((line &&& 7) <<< 4)
  | .b128Atom32 => a ^^^ ((line &&& 3) <<< 5)
  | .b128Atom32Flip8 => a ^^^ ((line &&& 3) <<< 5) ^^^ ((line &&& 1) <<< 3)
  | .b128Atom64 => a ^^^ ((line &&& 1) <<< 6)

/-- A piece of a copy: `bytes` bytes at global address `global` (`none`: outside the tensor)
and at shared address `smem`. -/
structure Piece where
  global : Option UInt64
  smem : Nat
  bytes : Nat
  deriving Repr, Inhabited, DecidableEq

namespace TensorMap

/-- Elements the box moves along dimension `i`: `⌈boxDim / elementStride⌉` (CUDA Driver API
13.4 §6.42, `cuTensorMapEncodeTiled`), 1 beyond the rank. -/
def boxCount (m : TensorMap) (i : Nat) : Nat :=
  if i < m.rank then (m.boxDim[i]! + m.elementStrides[i]! - 1) / max m.elementStrides[i]! 1 else 1

/-- Bytes one copy moves, the complete-tx count of a load: every element of the box, in bounds
or not. -/
def boxBytes (m : TensorMap) : Nat :=
  match m.im2col with
  | some x => x.pixelsPerColumn * x.channelsPerPixel * m.dtype.bytes?.getD 0
  | none => (List.range m.rank).foldl (fun acc i => acc * m.boxCount i) (m.dtype.bytes?.getD 0)

/-- The pieces of one row of a box: `rowBytes` bytes at shared offset `off` (swizzled), holding
dimension 0's elements from `x0` on of the row at global address `base` (`inb`: the row's other
coordinates are inside the tensor); 16-byte pieces, split into elements where the row crosses
the tensor's edge. (`tile` keeps this loop inline: `NV.Formal.TileLayout.tile_box2` follows its
form.) -/
def rowPieces (m : TensorMap) (esz : Nat) (inb : Bool) (base x0 : Int) (off rowBytes : Nat) :
    Array Piece := Id.run do
  let per := 16 / esz
  let inDim (x : Int) : Bool := 0 ≤ x && x < m.globalDim[0]!
  let mut out : Array Piece := #[]
  for q in [0:rowBytes / 16] do
    let off := off + q * 16
    let x0 := x0 + q * per
    let allIn := inb && inDim x0 && inDim (x0 + per - 1)
    let anyIn := inb && (List.range per).any (fun k => inDim (x0 + k))
    if allIn then
      out := out.push { global := some (base + x0 * esz).toNat.toUInt64,
                        smem := m.swizzle.apply off, bytes := 16 }
    else if !anyIn then
      out := out.push { global := none, smem := m.swizzle.apply off, bytes := 16 }
    else
      for k in [0:per] do
        let x := x0 + k
        out := out.push { global := if inDim x then some (base + x * esz).toNat.toUInt64 else none,
                          smem := m.swizzle.apply (off + k * esz), bytes := esz }
  return out

/-- Shared-memory bytes from one box row to the next: the row's bytes, or the swizzle's span
when that is wider ("if the swizzle width exceeds the inner dimension, ensure that the shared
memory is allocated to accommodate the full swizzle width", `cuda-guide-async` §4.13.2.2.5). -/
def rowPitch (m : TensorMap) (rowBytes : Nat) : Nat := max rowBytes m.swizzle.span

/-- The pieces of the tiled-mode box at `coords` (`.s32` each, dimension 0 first) whose shared
image starts at `smem`. Interleaved layouts, packed types and the OOB-NaN fill are not modelled
yet and are errors, as are maps the driver would reject. -/
def tile (m : TensorMap) (coords : Array Int) (smem : Nat) : Except String (Array Piece) := do
  if let some e := m.invalid? then throw s!"tensor map: {e}"
  if m.interleave != .none then throw "interleaved tensor maps are not modelled yet"
  let some esz := m.dtype.bytes? | throw s!"tensor-map type {repr m.dtype} is not modelled yet"
  let r := m.rank
  if coords.size != r then throw s!"{coords.size} coordinates for a rank-{r} tensor map"
  if m.elementStrides[0]! != 1 then
    throw "dimension 0's traversal stride must be 1 without interleave (§5.5.3.2)"
  if smem % 16 != 0 then throw s!"shared address {smem} is not 16-byte aligned"
  let rowBytes := m.boxCount 0 * esz
  let pitch := m.rowPitch rowBytes
  let per := 16 / esz
  let rows := (List.range r).foldl (fun acc i => if i == 0 then acc else acc * m.boxCount i) 1
  let inDim (i : Nat) (x : Int) : Bool := 0 ≤ x && x < m.globalDim[i]!
  let mut out : Array Piece := #[]
  for row in [0:rows] do
    -- The row's coordinates in dimensions 1 and up, and its global address.
    let mut rest := row
    let mut inb := true
    let mut base : Int := m.globalAddress.toNat
    for i in [1:r] do
      let j := rest % m.boxCount i
      rest := rest / m.boxCount i
      let x := coords[i]! + j * m.elementStrides[i]!
      inb := inb && inDim i x
      base := base + x * m.globalStrides[i - 1]!
    for q in [0:rowBytes / 16] do
      let off := smem + row * pitch + q * 16
      let x0 := coords[0]! + q * per
      let allIn := inb && inDim 0 x0 && inDim 0 (x0 + per - 1)
      let anyIn := inb && (List.range per).any (fun k => inDim 0 (x0 + k))
      if allIn then
        out := out.push { global := some (base + x0 * esz).toNat.toUInt64,
                          smem := m.swizzle.apply off, bytes := 16 }
      else if !anyIn then
        out := out.push { global := none, smem := m.swizzle.apply off, bytes := 16 }
      else
        for k in [0:per] do
          let x := x0 + k
          out := out.push { global := if inDim 0 x then some (base + x * esz).toNat.toUInt64 else none,
                            smem := m.swizzle.apply (off + k * esz), bytes := esz }
  if m.oobFill != .none && out.any (·.global.isNone) then
    throw "the OOB-NaN fill is not modelled yet"
  return out

/-- `.tile::gather4` (PTX ISA §5.5.3.4): four rows of a 2D tensor, row `ys[j]` from column `x` on
for the box's dimension-0 length, as rows 0 to 3 of one tile at `smem` (each a one-row box, at
the tile's row pitch and swizzled at its shared offset, as a tiled copy's rows); the map's box is
one row in dimension 1. -/
def gather4 (m : TensorMap) (x : Int) (ys : Array Int) (smem : Nat) : Except String (Array Piece) := do
  if m.rank != 2 then throw s!"gather4 on a rank-{m.rank} tensor map (§5.5.3.4: 2D only)"
  if m.boxDim[1]! != 1 then throw s!"gather4 with a box of {m.boxDim[1]!} rows in dimension 1 (§5.5.3.4.1: one)"
  if ys.size != 4 then throw s!"gather4 with {ys.size} row coordinates"
  let some esz := m.dtype.bytes? | throw s!"tensor-map type {repr m.dtype} is not modelled yet"
  let pitch := m.rowPitch (m.boxCount 0 * esz)
  let mut out := #[]
  for j in [0:4] do
    out := out ++ (← m.tile #[x, ys[j]!] (smem + j * pitch))
  return out

/-- The pieces of the im2col-mode copy at `coords` (`C`, then the filter base in `W`, `H`, `D`,
then `N`) with im2col `offsets` (`W` first) whose shared image starts at `smem` (PTX ISA §5.5.4):
`pixelsPerColumn` rows of `channelsPerPixel` elements. The pixels are the filter bases from
`coords` on, in order `W`, `H`, `D`, `N` within the bounding box, each spatial one stepping by
its traversal stride and wrapping to the box's lower corner past its upper one (Figures 13–15);
a pixel's elements are those at its base plus the offsets, out-of-tensor ones zero-filled. -/
def im2colTile (m : TensorMap) (coords : Array Int) (offsets : Array Nat) (smem : Nat) :
    Except String (Array Piece) := do
  if let some e := m.invalid? then throw s!"tensor map: {e}"
  let some x := m.im2col | throw "not an im2col tensor map"
  if m.interleave != .none then throw "interleaved tensor maps are not modelled yet"
  let some esz := m.dtype.bytes? | throw s!"tensor-map type {repr m.dtype} is not modelled yet"
  let r := m.rank
  if coords.size != r then throw s!"{coords.size} coordinates for a rank-{r} tensor map"
  if smem % 16 != 0 then throw s!"shared address {smem} is not 16-byte aligned"
  let rowBytes := x.channelsPerPixel * esz
  if rowBytes % 16 != 0 then throw s!"{rowBytes}-byte pixels: a multiple of 16"
  let sp := r - 2
  let lo (i : Nat) : Int := x.lower[i]!
  let hi (i : Nat) : Int := (m.globalDim[i + 1]! : Int) - 1 + x.upper[i]!
  let inDim (i : Nat) (v : Int) : Bool := 0 ≤ v && v < m.globalDim[i]!
  let mut pos := (Array.range sp).map fun i => coords[i + 1]!
  let mut n := coords[r - 1]!
  let mut out : Array Piece := #[]
  for p in [0:x.pixelsPerColumn] do
    let mut inb := inDim (r - 1) n
    let mut base : Int := m.globalAddress.toNat + n * m.globalStrides[r - 2]!
    for i in [0:sp] do
      let v := pos[i]! + offsets[i]!
      inb := inb && inDim (i + 1) v
      base := base + v * m.globalStrides[i]!
    out := out ++ m.rowPieces esz inb base coords[0]! (smem + p * m.rowPitch rowBytes) rowBytes
    -- The next pixel.
    let mut carry := true
    for i in [0:sp] do
      if carry then
        let v := pos[i]! + m.elementStrides[i + 1]!
        if v > hi i then
          pos := pos.set! i (lo i)
        else
          pos := pos.set! i v
          carry := false
    if carry then n := n + 1
  if m.oobFill != .none && out.any (·.global.isNone) then
    throw "the OOB-NaN fill is not modelled yet"
  return out

end TensorMap

/-- The pieces of a bulk copy of `size` bytes between global `global` and shared `smem`; both
16-byte aligned and `size` a multiple of 16 (§9.7.10.28.4.1). -/
def bulkPieces (global : UInt64) (smem size : Nat) : Except String (Array Piece) := do
  if global.toNat % 16 != 0 || smem % 16 != 0 || size % 16 != 0 then
    throw s!"bulk copy of {size} bytes between {global} and shared {smem}: not 16-byte aligned"
  return (Array.range (size / 16)).map fun i =>
    { global := some (global + (16 * i).toUInt64), smem := smem + 16 * i, bytes := 16 }

/-- Pieces split into elements of `elem` bytes: a bulk reduction acts element by element. -/
def elementPieces (elem : Nat) (ps : Array Piece) : Array Piece :=
  ps.flatMap fun p => (Array.range (p.bytes / elem)).map fun j =>
    { global := p.global.map (· + (elem * j).toUInt64), smem := p.smem + elem * j, bytes := elem }

/-- Group pieces into lanes of requests: consecutive pieces of one size, one global line (or all
out of bounds), at most 32 per request. -/
def batches (ps : Array Piece) : Array (Array Piece) := Id.run do
  let key (p : Piece) : Nat × Option UInt64 := (p.bytes, p.global.map (· / 128))
  let mut out : Array (Array Piece) := #[]
  let mut cur : Array Piece := #[]
  for p in ps do
    if let some q := cur[0]? then
      if key q != key p || cur.size == 32 then
        out := out.push cur
        cur := #[]
    cur := cur.push p
  if !cur.isEmpty then out := out.push cur
  return out

/-- A request of the async proxy. -/
def asyncReq (space : Space) (op : MemOp) (bytes : Nat) (addrs : Array (Option UInt64)) : Req :=
  { space, op, bytes, addrs, proxy := .async }

/-- The element type a tensor reduction `o` acts on for tensor-map type `d`, if the ISA's table
of `.redOp` and element types allows the pair (§9.7.10.28.5.4). -/
def tensorRedType (d : TmDType) (o : AtomOp) : Option AtomType :=
  let bitwise := o == .and || o == .or || o == .xor
  match d, o with
  | .u32, _ => some (if bitwise then .b32 else .u32)
  | .s32, .add | .s32, .min | .s32, .max => some .s32
  | .u64, .add | .u64, .min | .u64, .max => some .u64
  | .u64, _ => if bitwise then some .b64 else none
  | .s64, .min | .s64, .max => some .s64
  | .f32, .add => some .f32
  | .f16, .add | .f16, .min | .f16, .max => some .f16
  | .bf16, .add | .bf16, .min | .bf16, .max => some .bf16
  | _, _ => none

/-- A bulk copy (`op` a store) or reduction (`op` a reduction) between shared memories of
`size` bytes, `eb` bytes at a time, from the source (space and address) into the destination,
signalling `signals`: both addresses and the size 16-byte aligned (§9.7.10.28.4.1–2), the size
within shared memory (no target has more than 228 KiB an SM). -/
def elementCopy (src dst : Space × UInt64) (size eb : Nat) (op : MemOp) (signals : Array Req) :
    Except String Copy := do
  if src.2.toNat % 16 != 0 || dst.2.toNat % 16 != 0 || size % 16 != 0 || size > 228 * 1024 then
    throw s!"bulk copy of {size} bytes from {src.2} into {dst.2}: not 16-byte aligned or past shared memory"
  let move (j : Nat) : Move :=
    { src := asyncReq src.1 .ld eb #[some (src.2 + (j * eb).toUInt64)]
      dsts := #[asyncReq dst.1 op eb #[some (dst.2 + (j * eb).toUInt64)]] }
  return { moves := (Array.range (size / eb)).map move, signals }

/-- A load from global memory into shared memory: each destination maps a piece's shared address
to the space and address it is stored at (several for a multicast); `signals` are the copy's
completion RMWs. Out-of-bound pieces store zeros. -/
def loadCopy (ps : Array Piece) (dsts : Array (Nat → Space × UInt64)) (signals : Array Req)
    (group : Option Group := none) : Copy :=
  let moves := (batches ps).map fun b =>
    let bytes := b[0]!.bytes
    { src := asyncReq .global .ld bytes (b.map (·.global))
      dsts := dsts.map fun d =>
        let space := (d b[0]!.smem).1
        asyncReq space .st bytes (b.map fun p => some (d p.smem).2) : Move }
  { moves, signals, group }

/-- A store from shared memory at the pieces' shared addresses (space `space`, address map
`addr`) to global memory; out-of-bound pieces are not written. `op` is `.st`, or `.red` for
`cp.reduce.async.bulk`. -/
def storeCopy (ps : Array Piece) (space : Space) (addr : Nat → UInt64) (op : MemOp := .st)
    (group : Option Group := some .bulk) : Copy :=
  let inb := ps.filter (·.global.isSome)
  let moves := (batches inb).map fun b =>
    let bytes := b[0]!.bytes
    { src := asyncReq space .ld bytes (b.map fun p => some (addr p.smem))
      dsts := #[asyncReq .global op bytes (b.map (·.global))] : Move }
  { moves, group }

end NV.Async
