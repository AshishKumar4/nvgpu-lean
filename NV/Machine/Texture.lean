import NV.Num.Round
import NV.Num.Convert
import NV.Num.Arith
import NV.Num.F32

/-!
# Textures and surfaces

A launch's texture and surface objects (`Texture`), which its kernels name by handle: the handle of
entry `i` of the launch's table is `i + 1`, and 0 names none. The table is read-only launch data;
texel accesses are ordinary global loads, stores and reductions at the addresses a texture's
layout gives (`Texture.texelAddr`), so the memory model sees nothing new.

A fetch (`sample`) is the CUDA C++ Programming Guide's (§3.2.14.1 "Texture Memory", §15 "Texture
Fetching"): coordinates in `[0, N)` or normalized to `[0, 1)`, the addressing mode per coordinate
(clamp, border, and for normalized coordinates wrap and mirror), nearest-point or linear filtering
with weights in 9-bit fixed point with 8 fractional bits, and integer texels read as they are or
mapped to `[0, 1]` (`[-1, 1]` signed). What the guide leaves open is marked inferred where it is
decided: how a weight is rounded to fixed point, the precision of the weighted sum, linear
filtering's neighbour under wrap and mirror, and the components a texel lacks.

Array layouts (block-linear in hardware) are pitch-linear rows here: a texture or surface
instruction cannot observe the difference.
-/

namespace NV.Tex

open NV.Num

/-- The kind of a texel's channels (`cudaChannelFormatKind`). -/
inductive ChanKind where
  | unsigned | signed | float
  deriving DecidableEq, Repr, Inhabited

/-- `cudaTextureAddressMode`. -/
inductive Address where
  | wrap | clamp | mirror | border
  deriving DecidableEq, Repr, Inhabited

/-- `cudaTextureFilterMode`. -/
inductive Filter where
  | point | linear
  deriving DecidableEq, Repr, Inhabited

/-- A texel a fetch reads: the address it reads, and of a block-compressed object, whose texels a
fetch reads by their 4×4 block, the texel's place in the block (`4 · row + column`). -/
structure Texel where
  addr : UInt64
  sub : Nat := 0
  deriving Repr, Inhabited, BEq

structure Texture where
  /-- Global address of texel (0, 0, 0). -/
  base : UInt64
  /-- Dimensions addressed: 1, 2 or 3. -/
  dims : Nat
  width : Nat
  height : Nat := 1
  depth : Nat := 1
  /-- Bytes from a row to the next, and from a 2D slice to the next. -/
  pitch : Nat := 0
  slice : Nat := 0
  /-- Bits of each channel, 1, 2 or 4 channels of one kind (`cudaChannelFormatDesc`). -/
  chans : Array Nat
  kind : ChanKind
  /-- `cudaReadModeNormalizedFloat`: 8- and 16-bit integer texels read as floats. -/
  readNormalized : Bool := false
  normalizedCoords : Bool := false
  filter : Filter := .point
  /-- Per coordinate. -/
  address : Array Address := #[.clamp, .clamp, .clamp]
  /-- A surface object (`cudaCreateSurfaceObject`) rather than a texture object. -/
  surface : Bool := false
  /-- Layers of a layered object (`cudaArrayLayered`), 0 if not layered: `dims`-dimensional
  layers, one `pitch` (1D) or `slice` (2D) apart. -/
  layers : Nat := 0
  /-- A cubemap (`cudaArrayCubemap`): square 2D layers, six a cubemap (`layers` counts faces). -/
  cube : Bool := false
  /-- A mipmapped array's levels (`Texture.level`), 1 for any other object. -/
  levels : Nat := 1
  /-- Between levels, linear filtering rather than the nearest level (`cudaTextureDesc
  ::mipmapFilterMode`). -/
  mipLinear : Bool := false
  /-- The level bias and the level clamps (`mipmapLevelBias`, `min`/`maxMipmapLevelClamp`), binary32. -/
  lodBias : UInt32 := 0
  lodMin : UInt32 := 0
  lodMax : UInt32 := 0
  /-- A block-compressed array's format (`CUarray_format`, `CU_AD_FORMAT_BC1_UNORM` 0x91 to
  `BC7_UNORM_SRGB` 0x9e), 0 for any other object: its texels in 4×4 blocks of 8 or 16 bytes. -/
  bcFormat : Nat := 0
  deriving Repr, Inhabited

namespace Texture

/-- A block-compressed object's BC number (1 to 7), 0 for any other. -/
def bc (t : Texture) : Nat :=
  if 0x91 ≤ t.bcFormat && t.bcFormat ≤ 0x9e then [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7][t.bcFormat - 0x91]!
  else 0

/-- The bytes a fetch reads for a texel: its own, or a block-compressed object's block's. -/
def texelBytes (t : Texture) : Nat :=
  if t.bc == 0 then t.chans.foldl (· + ·) 0 / 8 else if t.bc ∈ [1, 4] then 8 else 16

def size (t : Texture) (d : Nat) : Nat := match d with | 0 => t.width | 1 => t.height | _ => t.depth

/-- The texture's own inconsistencies, if any. -/
def invalid? (t : Texture) : Option String :=
  if !(t.dims ∈ [1, 2, 3]) then some s!"{t.dims} dimensions"
  else if !(t.chans.size ∈ [1, 2, 4]) then some s!"{t.chans.size} channels"
  else if t.chans.any (· != t.chans[0]!) then some "channels of different widths"
  else if !(t.texelBytes ∈ [1, 2, 4, 8, 16]) then some s!"{t.texelBytes}-byte texels"
  else if t.bc != 0 && (t.dims != 2 || t.layers != 0 || t.levels != 1 || t.surface) then
    some "a block-compressed object other than a 2D array"
  else if t.bc ∈ [6, 7] || t.bcFormat ∈ [0x92, 0x94, 0x96, 0x98, 0x9a] then
    some s!"block-compressed format {t.bcFormat} (BC6H, BC7, sRGB and signed are not modelled)"
  else if t.bc != 0 && !t.readNormalized then some "block-compressed texels read as integers"
  else if t.kind == .float && !(t.chans[0]! ∈ [16, 32]) then some "float channels of 16 or 32 bits"
  else if t.filter == .linear && t.kind != .float && !t.readNormalized then
    some "linear filtering of texels read as integers (the guide allows it for floats only)"
  else if !t.normalizedCoords && t.address.any (· ∈ [.wrap, .mirror]) then
    some "wrap and mirror need normalized coordinates"
  else if t.layers > 0 && t.dims == 3 then some "layers of three dimensions"
  else if t.cube && (t.dims != 2 || t.width != t.height || t.layers % 6 != 0 || t.layers == 0) then
    some "a cubemap has square 2D layers, six a cubemap"
  else none

/-- The face of a cubemap a direction `(x, y, z)` (binary32 bits) points at, and the face's
normalized coordinates `((s/m + 1)/2, (t/m + 1)/2)` (CUDA C++ Programming Guide 13.4, §6.2.14.1.4,
Table 6: the face of the coordinate of largest magnitude `m`, and `s`, `t` by face). Ties between
magnitudes, which the table leaves out, go to `x`, then `y`; the arithmetic is binary32 rounded to
nearest (both inferred). -/
def cubeFace (x y z : UInt32) : Nat × UInt32 × UInt32 :=
  let mag (v : UInt32) : UInt32 := v &&& 0x7fffffff
  let neg (v : UInt32) : UInt32 := v ^^^ 0x80000000
  let pos (v : UInt32) : Bool := v >>> 31 == 0
  let (ax, ay, az) := (mag x, mag y, mag z)
  let (face, m, s, t) :=
    if ax ≥ ay && ax ≥ az then
      if pos x then (0, ax, neg z, neg y) else (1, ax, z, neg y)
    else if ay ≥ az then
      if pos y then (2, ay, x, z) else (3, ay, x, neg z)
    else if pos z then (4, az, x, neg y) else (5, az, neg x, neg y)
  let unit (c : UInt32) : UInt32 :=
    let q := (NV.Num.div .f32 .rn false c.toNat m.toNat).toUInt32
    NV.Num.F32.mul .rn false (NV.Num.F32.add .rn false q 0x3f800000) 0x3f000000
  (face, unit s, unit t)

/-- Layer `k` of a layered object, which must be one of its layers (`layer`). -/
def layerAt (t : Texture) (k : Nat) : Texture :=
  { t with base := t.base + (k * (if t.dims == 1 then t.pitch else t.slice)).toUInt64, layers := 0 }

/-- Layer `k` of a layered object, as an object of its own (filtering stays within a layer, CUDA
Programming Guide, layered textures). A layer index past the last is left undefined by the
ISA (§9.7.12.3: an unsigned index into the array), so it is an error. -/
def layer (t : Texture) (k : Nat) : Except String Texture :=
  if t.layers == 0 then .error "an array access to an object that is not layered"
  else if k ≥ t.layers then .error s!"layer {k} of {t.layers}"
  else .ok (t.layerAt k)

/-- Level `k` of a mipmapped object, as an object of its own: each dimension half the last
level's (at least 1), rows 128-byte aligned, the levels one after another from level 0's base,
a layered level's layers (a cubemap's faces) together (the driver's layout, `tools/cudadev/tex.h`). -/
def level (t : Texture) (k : Nat) : Texture := Id.run do
  let mut base := t.base
  let mut lv := { t with levels := 1 }
  for j in [0:k + 1] do
    let w := max 1 (t.width >>> j)
    let h := if t.dims ≥ 2 then max 1 (t.height >>> j) else 1
    let d := if t.dims ≥ 3 then max 1 (t.depth >>> j) else 1
    let pitch := (w * t.texelBytes + 127) / 128 * 128
    lv := { lv with base, width := w, height := h, depth := d, pitch, slice := pitch * h }
    base := base + (pitch * h * d * max 1 t.layers).toUInt64
  return lv

/-- A binary32 value in 1/256 units, truncated toward zero (non-finite values read as 0). -/
def fixed8 (bits : UInt32) : Int :=
  match Format.f32.decode bits.toNat with
  | .fin neg m e =>
    let v : Nat := if e + 8 ≥ 0 then m * 2 ^ (e + 8).toNat else m / 2 ^ (-(e + 8)).toNat
    if neg then -(v : Int) else v
  | _ => 0

/-- The level of detail of an explicit-LOD fetch (`lod`, binary32), in 1/256 of a level: the
LOD and the bias each truncated toward zero to 1/256, summed, clamped to the level clamps (so
truncated) and to the levels there are. A GH200 and an A100 agree with this on 172,032 fetches
of 7 levels (LODs from -1/2 to 6.5 in steps of 1/4096; biases 0.3 and 1/3; clamps 0.3 and 5.7). -/
def lodOf (t : Texture) (lod : UInt32) : Int :=
  let l := fixed8 lod + fixed8 t.lodBias
  let hi := min (fixed8 t.lodMax) (256 * ((t.levels : Int) - 1))
  max 0 (max (fixed8 t.lodMin) (min l hi))

/-- The footprint of a fetch at LOD `l` (`lodOf`) over the levels it reads, `fpAt` a level's own
footprint for a share of the result out of 256: nearest-level filtering the level nearest `l`
(halves up), all of it; linear filtering the two levels about `l` (the last level twice past it),
the upper level the LOD's fraction `f` and the lower the rest. -/
def mipFootprint (t : Texture) (l : Int)
    (fpAt : Texture → Nat → List (Option Texel × Nat)) : List (Option Texel × Nat) :=
  let last := t.levels - 1
  if !t.mipLinear then fpAt (t.level (min last ((l + 128) / 256).toNat)) 256 else
  let k := min last (l / 256).toNat
  let f := if k == last then 0 else (l % 256).toNat
  fpAt (t.level k) (256 - f) ++ fpAt (t.level (min last (k + 1))) f

/-- The address of texel `(i, j, k)`, which must be in range; of a block-compressed object's,
its 4×4 block's (rows of blocks `pitch` apart). -/
def texelAddr (t : Texture) (i j k : Nat) : UInt64 :=
  if t.bc == 0 then t.base + (k * t.slice + j * t.pitch + i * t.texelBytes).toUInt64
  else t.base + (k * t.slice + j / 4 * t.pitch + i / 4 * t.texelBytes).toUInt64

/-- Texel `(i, j, k)` as a fetch reads it (`Texel`). -/
def texelAt (t : Texture) (i j k : Nat) : Texel :=
  { addr := t.texelAddr i j k, sub := if t.bc == 0 then 0 else j % 4 * 4 + i % 4 }

end Texture

/-- A dyadic rational `n · 2^e`: coordinates are binary32, and every step of addressing and
filtering stays exact in these. -/
structure Dy where
  n : Int
  e : Int
  deriving Repr, Inhabited

namespace Dy

def ofNat (k : Nat) : Dy := ⟨k, 0⟩

/-- `a / 2^p` as an integer, rounded down. -/
def shiftFloor (n : Int) (p : Nat) : Int := n / (2 ^ p : Nat)

def floor (x : Dy) : Int := if x.e ≥ 0 then x.n * (2 ^ x.e.toNat : Nat) else shiftFloor x.n (-x.e).toNat

/-- The fractional part `x − floor x` in units of `2^-p` (exact when `p ≥ -e`). -/
def fracUnits (x : Dy) (p : Nat) : Int :=
  if x.e ≥ 0 then 0 else
  let q := (-x.e).toNat
  let r := x.n - shiftFloor x.n q * (2 ^ q : Nat)
  if q ≤ p then r * (2 ^ (p - q) : Nat) else shiftFloor r (q - p)

def mulNat (x : Dy) (k : Nat) : Dy := ⟨x.n * k, x.e⟩

/-- `x − 1/2`. -/
def subHalf (x : Dy) : Dy :=
  let e := min x.e (-1)
  ⟨x.n * (2 ^ (x.e - e).toNat : Nat) - (2 ^ (-1 - e).toNat : Nat), e⟩

end Dy

/-- A binary32 coordinate as a dyadic rational; non-finite coordinates are not modelled. -/
def coordOf (bits : UInt32) : Except String Dy :=
  match Format.f32.decode bits.toNat with
  | .fin neg m e => pure ⟨if neg then -(m : Int) else m, e⟩
  | _ => throw "a non-finite texture coordinate (not modelled)"

/-- The texel index along one dimension for index `i` (any integer), or `none` (a border texel). -/
def resolve (mode : Address) (n : Nat) (i : Int) : Option Nat :=
  match mode with
  | .border => if 0 ≤ i ∧ i < n then some i.toNat else none
  -- Wrap and mirror index the neighbour of a linear filter modulo N, mirrored for mirror
  -- (inferred).
  | .wrap => some (i % (n : Int)).toNat
  | .mirror =>
    let p := i % (2 * n : Int)
    some (if p < n then p.toNat else (2 * n - 1 - p).toNat)
  | .clamp => some (max 0 (min i (n - 1 : Int))).toNat

/-- A coordinate in texels: normalized ones wrapped or mirrored first (§3.2.14.1), then scaled
by the size. -/
def texelCoord (t : Texture) (d : Nat) (c : Dy) : Dy :=
  let n := t.size d
  if !t.normalizedCoords then c else
  let mode := t.address[d]?.getD .clamp
  -- frac(c) and 1 − frac(c) in units of 2^-p.
  let p := (max 0 (-c.e)).toNat
  let f := c.fracUnits p
  let c' : Dy := match mode with
    | .wrap => ⟨f, -(p : Int)⟩
    | .mirror => if c.floor % 2 == 0 then ⟨f, -(p : Int)⟩ else ⟨(2 ^ p : Nat) - f, -(p : Int)⟩
    | _ => c
  c'.mulNat n

/-- The texel indices (with their 8-bit-fraction weights out of 256) a fetch combines along one
dimension: one with weight 256 for nearest-point sampling, two for linear filtering. A weight is
`frac(x − 1/2)` rounded to the nearest 1/256, half up; clamped addressing clamps `x − 1/2` to the
texels first, so that past either edge the first texel weighs all (a GH200 agrees: `footprint`). -/
def taps (t : Texture) (d : Nat) (c : Dy) (off : Int := 0) : List (Option Nat × Nat) :=
  let x := texelCoord t d c
  let x : Dy := if off == 0 then x
    else if x.e ≥ 0 then ⟨x.n * (2 ^ x.e.toNat : Nat) + off, 0⟩
    else ⟨x.n + off * (2 ^ (-x.e).toNat : Nat), x.e⟩
  let n := t.size d
  let mode := t.address[d]?.getD .clamp
  match t.filter with
  | .point => [(resolve mode n x.floor, 256)]
  | .linear =>
    let xb := x.subHalf
    let i := xb.floor
    -- Rounded to nearest, half up, from 2^-9 units.
    let a := if mode == .clamp && (i < 0 || i ≥ n - 1) then 0 else ((xb.fracUnits 9) + 1) / 2
    [(resolve mode n i, (256 - a).toNat), (resolve mode n (i + 1), a.toNat)]

/-- The footprint of a fetch at coordinates `cs` (one per dimension), each moved by `offs` texels
(PTX's offset operand, §9.7.12.3: "applied to coordinates before doing texture lookup", from -8
to 7; taken in texel units, after normalized coordinates are scaled and wrapped: inferred): each
texel's address (`none`: a border texel, zero) and its weight, for a fetch whose share of the result
is `share` out of 256 (a mipmap level's, `Texture.mipFootprint`). Linear filtering splits the share
along each dimension in turn by its weight (`taps`), each part rounded to 1/256: in three
dimensions first along the third, the upper part rounded half down; then along the first, the
upper part `x1` rounded half up; then each part along the second, the upper part of `x1` rounded
half up and of `x0` half down. A GH200 and an A100 agree on 4,096 fetches over random coordinates
and texels in each of one, two and three dimensions, with clamped, wrapped and bordered
addressing, and with linear filtering between mipmap levels in each. -/
def footprint (t : Texture) (cs : Array Dy) (offs : Array Int := #[]) (share : Nat := 256) :
    List (Option Texel × Nat) :=
  let along (d : Nat) : List (Option Nat × Nat) :=
    if d < t.dims then taps t d (cs[d]?.getD ⟨0, 0⟩) (offs[d]?.getD 0) else [(some 0, 256)]
  let at_ (i j k : Option Nat) : Option Texel := do pure (t.texelAt (← i) (← j) (← k))
  match t.dims, t.filter, along 0, along 1, along 2 with
  | 1, .linear, [(i0, _), (i1, a)], _, _ =>
    let x1 := (share * a + 128) / 256
    [(at_ i0 (some 0) (some 0), share - x1), (at_ i1 (some 0) (some 0), x1)]
  | d, .linear, [(i0, _), (i1, a)], [(j0, _), (j1, b)], zs =>
    let plane (k : Option Nat) (s : Nat) : List (Option Texel × Nat) :=
      let x1 := (s * a + 128) / 256
      let x0 := s - x1
      let q1 := (x1 * b + 128) / 256
      let q0 := (x0 * b + 127) / 256
      [(at_ i0 j0 k, x0 - q0), (at_ i1 j0 k, x1 - q1), (at_ i0 j1 k, q0), (at_ i1 j1 k, q1)]
    match d, zs with
    | 3, [(k0, _), (k1, c)] =>
      let z1 := (share * c + 127) / 256
      plane k0 (share - z1) ++ plane k1 z1
    | _, _ => plane (some 0) share
  | _, _, _, _, _ =>
  (along 2).flatMap fun (k, wk) => (along 1).flatMap fun (j, wj) => (along 0).map fun (i, wi) =>
    (at_ i j k, wi * wj * wk * share)

/-- Channel `c` of a texel's little-endian bytes (as words). -/
def channel (t : Texture) (ws : Array UInt32) (c : Nat) : Nat :=
  let w := t.chans[0]!
  let bit := c * w
  let all := ws.toList.zipIdx.foldl (fun acc (x, k) => acc ||| (x.toNat <<< (32 * k))) 0
  (all >>> bit) % 2 ^ w

/-- A channel's value as an exact rational for filtering: floats decoded, integers read as they
are or normalized (`v / (2^w − 1)` unsigned, `max(v / (2^(w−1) − 1), −1)` signed; the clamp of
the most negative value is inferred). `none` for NaN and infinities (not combined). -/
def channelVal (t : Texture) (v : Nat) : Option (Int × Nat × Int) :=
  -- (numerator, denominator, binary exponent): numerator / denominator · 2^e.
  let w := t.chans[0]!
  match t.kind with
  | .float =>
    match (if w == 16 then Format.f16 else Format.f32).decode v with
    | .fin neg m e => some (if neg then -(m : Int) else m, 1, e)
    | _ => none
  | .unsigned => some (v, 2 ^ w - 1, 0)
  | .signed =>
    let s : Int := if v ≥ 2 ^ (w - 1) then (v : Int) - 2 ^ w else v
    some (max s (-(2 ^ (w - 1) - 1 : Int)), 2 ^ (w - 1) - 1, 0)

/-! ## Block-compressed texels

A 4×4 block's 8 or 16 bytes hold its texels as endpoints and per-texel indices into a palette
between them (BC1 to BC5: Microsoft's "Texture Block Compression in Direct3D 11"), each texel's
channels read as unsigned normalized; the palette's
in-between entries are the GPU's own: an A100's over every pair of endpoints, which a GH200's
agree with on random blocks. -/

/-- A BC4 block's (the alpha half of a BC3 block's and either half of a BC5 block's) value at
`sub`, as `numerator / denominator`, from its 8 bytes `blk` (bit `8i + b` of byte `i`): two
endpoint bytes, then a 3-bit index per texel. With the first endpoint above the second, indices
2 to 7 blend them; otherwise 2 to 5 do, and 6 and 7 are the format's least and greatest values.
Unsigned, a blend is the 16-bit unsigned normalized `(257 − w)·a0 + w·a1` (`w` by index: 36, 72,
113, 144, 185, 221; or 48, 96, 161, 209); BC3's alpha (`alpha8`) is 8-bit, `(A·a0 + B·a1 + 1024)
/ 2048` rounded down (`B` by index: 289, 578, 892, 1156, 1470, 1759; or 385, 770, 1278, 1663;
`A = 2048 − B`). -/
def bc4Value (blk : Nat) (sub : Nat) (alpha8 : Bool := false) : Int × Nat :=
  let a0 := blk % 256
  let a1 := blk / 256 % 256
  let idx := blk / 2 ^ (16 + 3 * sub) % 8
  let eight := a0 > a1
  if idx == 0 then (if alpha8 then (a0, 255) else (a0 * 257, 65535))
  else if idx == 1 then (if alpha8 then (a1, 255) else (a1 * 257, 65535))
  else if !eight && idx ≥ 6 then (if idx == 6 then 0 else 1, 1)
  else if !alpha8 then
    let w := (if eight then #[36, 72, 113, 144, 185, 221] else #[48, 96, 161, 209])[idx - 2]!
    ((257 - w) * a0 + w * a1, 65535)
  else
    let b := (if eight then #[289, 578, 892, 1156, 1470, 1759] else #[385, 770, 1278, 1663])[idx - 2]!
    (((2048 - b) * a0 + b * a1 + 1024) / 2048, 255)

/-- A BC1 colour block's (the colour half of a BC2 or BC3 block's) texel at `sub` from its 8 bytes
`blk`: two RGB 5:6:5 endpoints, then a 2-bit index per texel; its red, green and blue as 8-bit
values, and whether it is transparent. With the first endpoint above the second as 16-bit words
(always, for BC2 and BC3: `fourAlways`), indices 2 and 3 blend them; otherwise index 2 is their
mean and 3 transparent black. Red and blue blend the 5-bit endpoints exactly and scale by 33/4,
rounded down (`(2·e0 + e1)·11/4`, `(e0 + e1)·33/8`; an endpoint alone is its bits replicated);
green blends the replicated 8-bit endpoints `a`, `b`: `(703·a + 321·b + 512) / 1024` and its
mirror, or `(255·a + 257·b + 256) / 512`, rounded down. -/
def bcColour (blk sub : Nat) (fourAlways : Bool) : Array Nat × Bool :=
  let c0 := blk % 65536
  let c1 := blk / 65536 % 65536
  let idx := blk / 2 ^ (32 + 2 * sub) % 4
  let four := fourAlways || c0 > c1
  let fiveBit (a b : Nat) : Nat :=
    if idx == 0 then a * 33 / 4 else if idx == 1 then b * 33 / 4
    else if four then (if idx == 2 then 2 * a + b else a + 2 * b) * 11 / 4
    else if idx == 2 then (a + b) * 33 / 8 else 0
  let green (g0 g1 : Nat) : Nat :=
    let a := g0 * 4 + g0 / 16
    let b := g1 * 4 + g1 / 16
    if idx == 0 then a else if idx == 1 then b
    else if four then (if idx == 2 then 703 * a + 321 * b + 512 else 321 * a + 703 * b + 512) / 1024
    else if idx == 2 then (255 * a + 257 * b + 256) / 512 else 0
  (#[fiveBit (c0 / 2048) (c1 / 2048), green (c0 / 32 % 64) (c1 / 32 % 64), fiveBit (c0 % 32) (c1 % 32)],
   !four && idx == 3)

/-- A block-compressed texel's channels (red, green, blue, alpha) as `numerator / denominator`,
from its block's words `ws` and its place `sub`: BC1's colour, alpha one or (transparent) zero;
BC2's colour from its second half and alpha from its first, 4 bits a texel; BC3's colour from its
second half and alpha (8-bit) from its first; BC4 red (green and blue zero, alpha one), BC5 red
and green from its two halves (blue zero, alpha one). The colour and BC3's alpha match an A100 and
a GH200 on every texel of random blocks, the colour on every pair of endpoints per channel. -/
def bcTexel (t : Texture) (ws : Array UInt32) (sub : Nat) : Array (Int × Nat) :=
  let half (k : Nat) : Nat := ws[2 * k]!.toNat + ws[2 * k + 1]!.toNat * 2 ^ 32
  let rgb (c : Array Nat) : List (Int × Nat) := c.toList.map (fun (v : Nat) => (Int.ofNat v, 255))
  match t.bc with
  | 1 =>
    let (c, clear) := bcColour (half 0) sub false
    (rgb c ++ [(if clear then 0 else 1, 1)]).toArray
  | 2 =>
    let (c, _) := bcColour (half 1) sub true
    (rgb c ++ [(Int.ofNat (half 0 / 2 ^ (4 * sub) % 16), 15)]).toArray
  | 3 =>
    let (c, _) := bcColour (half 1) sub true
    (rgb c ++ [bc4Value (half 0) sub true]).toArray
  | 4 => #[bc4Value (half 0) sub, (0, 1), (0, 1), (1, 1)]
  | 5 => #[bc4Value (half 0) sub, bc4Value (half 1) sub, (0, 1), (1, 1)]
  | _ => #[(0, 1), (0, 1), (0, 1), (1, 1)]

/-- `num / den · 2^e` rounded to nearest binary32, ties away from zero. -/
def toF32 (num : Int) (den : Nat) (e : Int) : UInt32 :=
  if num == 0 || den == 0 then 0 else
  -- Enough quotient bits for a correctly rounded binary32, the remainder as sticky.
  let neg := num < 0
  let a := num.natAbs * 2 ^ 60
  let q := a / den
  let sticky := a % den != 0
  (Format.f32.round .rna .ieee neg q (e - 60) sticky).toUInt32

/-- The four result words of a fetch from the texels of its footprint (`resp`: each texel's
words, `texelBytes` rounded up to words, in footprint order) and their weights. Floats and
normalized integers: each channel the weighted sum computed exactly and rounded once to binary32,
ties away from zero (a GH200's linear filtering in one and two dimensions and between levels of
linearly filtered mipmaps, and its and an A100's blends of point-sampled levels). Integers read as
they are (point sampling only): the texel's channels, sign-extended. A texel's missing
channels read as zero (inferred). -/
def combine (t : Texture) (fp : List (Option Texel × Nat)) (resp : Array UInt32) : Array UInt32 :=
  let wpt := (t.texelBytes + 3) / 4
  let texel (k : Nat) : Array UInt32 := resp.extract (k * wpt) (k * wpt + wpt)
  let w := t.chans[0]!
  (Array.range 4).map fun c =>
    if c ≥ t.chans.size && t.bc == 0 then 0 else
    if t.kind != .float && !t.readNormalized then
      match fp.zipIdx.find? (·.1.1.isSome) with
      | some (_, k) =>
        let v := channel t (texel k) c
        if t.kind == .signed && v ≥ 2 ^ (w - 1) then (v + 2 ^ 32 - 2 ^ w).toUInt32 else v.toUInt32
      | none => 0
    else
      let wsum := fp.foldl (· + ·.2) 0
      -- Point sampling of a float returns the texel unchanged (NaN included).
      if t.kind == .float && fp.length == 1 then
        match fp with
        | [(some _, _)] =>
          let v := channel t (texel 0) c
          if w == 16 then (cvt Format.f16 Format.f32 .rn {} v).toUInt32 else v.toUInt32
        | _ => 0
      else
        -- Σ weight_k · v_k over a common denominator.
        let terms := fp.zipIdx.filterMap fun ((a, wt), k) =>
          match a with
          | none => none
          | some x =>
            if wt == 0 then none
            else if t.bc != 0 then
              let (n, d) := (bcTexel t (texel k) x.sub)[c]!
              some (wt, n, d, (0 : Int))
            else channelVal t (channel t (texel k) c) |>.map (wt, ·)
        let emin := terms.foldl (fun m (_, _, _, e) => min m e) 0
        let den := terms.foldl (fun d (_, _, dn, _) => Nat.lcm d dn) 1
        let num := terms.foldl (fun s (wt, n, dn, e) =>
          s + (wt : Int) * n * (den / dn : Nat) * (2 ^ (e - emin).toNat : Nat)) 0
        toF32 num (den * wsum) emin

/-- The result of a `tld4` (gather) of component `c`: that component of the four texels of the
bilinear footprint at `(x, y)`, in the order (i, j+1), (i+1, j+1), (i+1, j), (i, j) (inferred). -/
def gatherOrder : List Nat := [2, 3, 1, 0]

/-- How a surface instruction treats coordinates out of bounds (PTX `.trap`, `.clamp`, `.zero`). -/
inductive Clamp where
  | trap | clamp | zero
  deriving DecidableEq, Repr, Inhabited

/-- The address of a `bytes`-byte surface access at byte offset `x` of row `j` of slice `k`
(the lowest coordinate is a byte offset, §9.7.13.1): `none` when out of bounds under `.zero`, an
error under `.trap`. `.clamp` moves each coordinate to the nearest in-bounds one (inferred). -/
def surfAddr (t : Texture) (mode : Clamp) (bytes : Nat) (x j k : Int) : Except String (Option UInt64) := do
  let row := t.width * t.texelBytes
  let hi := [ (row - bytes : Int), (t.height - 1 : Int), (t.depth - 1 : Int) ]
  let cs := [x, if t.dims ≥ 2 then j else 0, if t.dims ≥ 3 then k else 0]
  let inb := (cs.zip hi).all fun (c, h) => 0 ≤ c ∧ c ≤ h
  if !inb then
    match mode with
    | .trap => throw s!"surface access at ({x}, {j}, {k}) out of bounds (.trap)"
    | .zero => return none
    | .clamp => pure ()
  let cl := (cs.zip hi).map fun (c, h) => (max 0 (min c h)).toNat
  match cl with
  | [x, j, k] =>
    unless x % bytes == 0 do throw s!"surface byte offset {x} is not aligned to {bytes} bytes"
    return some (t.base + (k * t.slice + j * t.pitch + x).toUInt64)
  | _ => return none

end NV.Tex
