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
  /-- sRGB texels (`cudaTextureDesc::sRGB`): red, green and blue read through `srgbTable`. -/
  srgb : Bool := false
  /-- The border colour (`cudaTextureDesc::borderColor`, binary32 bits), which texels past the edge
  read under border addressing. -/
  border : Array UInt32 := #[0, 0, 0, 0]
  /-- A view of some layers of a layered object: the first of them, and the layers each level of
  the object holds (0: `layers`), which a mipmapped object's levels lie apart by. -/
  firstLayer : Nat := 0
  levelLayers : Nat := 0
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
  else if t.bcFormat ∈ [0x98, 0x9a] then
    some s!"block-compressed format {t.bcFormat} (signed BC4 and BC5 are not modelled)"
  else if (t.bcFormat ∈ [0x92, 0x94, 0x96, 0x9e]) != t.srgb && t.bc != 0 then
    some "an sRGB block-compressed format read without sRGB, or the reverse"
  else if t.srgb && t.bc == 0 && (t.kind != .unsigned || t.chans[0]! != 8 || !t.readNormalized) then
    some "sRGB texels other than 8-bit unsigned read normalized"
  else if t.srgb && (t.filter == .linear || t.mipLinear) then
    some "linear filtering of sRGB texels (its precision is the GPU's own, not modelled)"
  else if t.kind == .signed && t.readNormalized && (t.filter == .linear || t.mipLinear) then
    some "linear filtering of signed normalized texels (its rounding is the GPU's own, not modelled)"
  else if t.border.any (· != 0) && (t.srgb || t.bc != 0 || (t.kind != .float && !t.readNormalized)) then
    some "a border colour for sRGB, block-compressed or integer-read texels"
  else if t.bc != 0 && t.bc != 6 && !t.readNormalized then some "block-compressed texels read as integers"
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
  { t with base := t.base + ((t.firstLayer + k) * (if t.dims == 1 then t.pitch else t.slice)).toUInt64,
           layers := 0, firstLayer := 0 }

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
  let held := if t.levelLayers > 0 then t.levelLayers else max 1 t.layers
  for j in [0:k + 1] do
    let w := max 1 (t.width >>> j)
    let h := if t.dims ≥ 2 then max 1 (t.height >>> j) else 1
    let d := if t.dims ≥ 3 then max 1 (t.depth >>> j) else 1
    let pitch := (w * t.texelBytes + 127) / 128 * 128
    lv := { lv with base, width := w, height := h, depth := d, pitch, slice := pitch * h }
    base := base + (pitch * h * d * held).toUInt64
  -- A view of some layers: its first layer's place in the level.
  return { lv with base := lv.base + (t.firstLayer * (if t.dims == 1 then lv.pitch else lv.slice)).toUInt64,
                   firstLayer := 0, levelLayers := 0 }

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
texel's address (`none`: a border texel, the border colour) and its weight, for a fetch whose share of the result
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

/-- Channel `c` of the border colour as a texel of the object would read it (`channelVal`'s form):
a float exactly; for normalized integers clamped to the format's range and rounded half up to its
steps (a GH200 and an A100 for 8-bit unsigned; the rest inferred). `none` for NaN and infinities. -/
def borderVal (t : Texture) (c : Nat) : Option (Int × Nat × Int) :=
  match Format.f32.decode (t.border[c]?.getD 0).toNat with
  | .fin neg m e =>
    let x : Int := if neg then -(m : Int) else m
    if t.kind == .float then some (x, 1, e) else
    let w := t.chans[0]!
    let top : Nat := if t.kind == .signed then 2 ^ (w - 1) - 1 else 2 ^ w - 1
    -- x · 2^e · top, rounded half up and clamped to the format's range.
    let scaled : Int := if e ≥ 0 then x * top * 2 ^ e.toNat * 2 + 1 else x * top * 2 + 2 ^ (-e).toNat
    let den : Nat := if e ≥ 0 then 2 else 2 * 2 ^ (-e).toNat
    let q : Int := Int.fdiv scaled den
    let lo : Int := if t.kind == .signed then -(top : Int) else 0
    some (max lo (min top q), top, 0)
  | _ => none

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

/-- BC7's partitions (Direct3D 11's BC7 format: "Partition patterns"): per partition of two
subsets, each texel's subset as a bit (texel `i`, bit `i`); of three, as two bits; and the anchor
texel of subset 1 of two, and of subsets 1 and 2 of three, whose index has one bit fewer. -/
def bc7Two : Array Nat := #[
  0xcccc, 0x8888, 0xeeee, 0xecc8, 0xc880, 0xfeec, 0xfec8, 0xec80,
  0xc800, 0xffec, 0xfe80, 0xe800, 0xffe8, 0xff00, 0xfff0, 0xf000,
  0xf710, 0x8e, 0x7100, 0x8ce, 0x8c, 0x7310, 0x3100, 0x8cce,
  0x88c, 0x3110, 0x6666, 0x366c, 0x17e8, 0xff0, 0x718e, 0x399c,
  0xaaaa, 0xf0f0, 0x5a5a, 0x33cc, 0x3c3c, 0x55aa, 0x9696, 0xa55a,
  0x73ce, 0x13c8, 0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x660,
  0x272, 0x4e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c,
  0x9336, 0x9cc6, 0x817e, 0xe718, 0xccf0, 0xfcc, 0x7744, 0xee22]
def bc7TwoAnchor : Array Nat := #[
  15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
  15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2,
  15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6,
  6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15]
def bc7Three : Array Nat := #[
  0xaa685050, 0x6a5a5040, 0x5a5a4200, 0x5450a0a8, 0xa5a50000, 0xa0a05050, 0x5555a0a0, 0x5a5a5050,
  0xaa550000, 0xaa555500, 0xaaaa5500, 0x90909090, 0x94949494, 0xa4a4a4a4, 0xa9a59450, 0x2a0a4250,
  0xa5945040, 0xa425054, 0xa5a5a500, 0x55a0a0a0, 0xa8a85454, 0x6a6a4040, 0xa4a45000, 0x1a1a0500,
  0x50a4a4, 0xaaa59090, 0x14696914, 0x69691400, 0xa08585a0, 0xaa821414, 0x50a4a450, 0x6a5a0200,
  0xa9a58000, 0x5090a0a8, 0xa8a09050, 0x24242424, 0xaa5500, 0x24924924, 0x24499224, 0x50a50a50,
  0x500aa550, 0xaaaa4444, 0x66660000, 0xa5a0a5a0, 0x50a050a0, 0x69286928, 0x44aaaa44, 0x66666600,
  0xaa444444, 0x54a854a8, 0x95809580, 0x96969600, 0xa85454a8, 0x80959580, 0xaa141414, 0x96960000,
  0xaaaa1414, 0xa05050a0, 0xa0a5a5a0, 0x96000000, 0x40804080, 0xa9a8a9a8, 0xaaaaaa44, 0x2a4a5254]
def bc7ThreeAnchor : Array (Array Nat) := #[#[
  3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3,
  3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15,
  8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15,
  3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3], #[
  15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8,
  15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8,
  15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8,
  15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8]]

/-- A BC7 mode (Direct3D 11's BC7 format, "BC7 Format Mode Reference"): its subsets, partition,
rotation and index-selection bits, colour and alpha bits per endpoint, per-endpoint and shared
P-bits, and its two index sets' bits (0: none). -/
structure Bc7Mode where
  subsets : Nat
  partBits : Nat
  rotBits : Nat
  selBits : Nat
  colorBits : Nat
  alphaBits : Nat
  endpointP : Nat
  sharedP : Nat
  index0 : Nat
  index1 : Nat
  deriving Inhabited

def bc7Modes : Array Bc7Mode := #[
  ⟨3, 4, 0, 0, 4, 0, 1, 0, 3, 0⟩, ⟨2, 6, 0, 0, 6, 0, 0, 1, 3, 0⟩, ⟨3, 6, 0, 0, 5, 0, 0, 0, 2, 0⟩,
  ⟨2, 6, 0, 0, 7, 0, 1, 0, 2, 0⟩, ⟨1, 0, 2, 1, 5, 6, 0, 0, 2, 3⟩, ⟨1, 0, 2, 0, 7, 8, 0, 0, 2, 2⟩,
  ⟨1, 0, 0, 0, 7, 7, 1, 0, 4, 0⟩, ⟨2, 6, 0, 0, 5, 5, 1, 0, 2, 0⟩]

/-- The interpolation weights out of 64 for indices of 2, 3 and 4 bits. -/
def bc7Weights : Array (Array Nat) := #[#[0, 21, 43, 64], #[0, 9, 18, 27, 37, 46, 55, 64],
  #[0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64]]

/-- A BC7 block's (`blk`, its 128 bits) texel at `sub` as 8-bit red, green, blue and alpha, as the
format defines it, which an A100 and a GH200 match on every texel of random blocks: the mode
(the first set bit), its partition, rotation and index selection, the endpoints (with their
P-bits, widened by replicating their top bits), then each texel's index or indices weighting its
subset's endpoints, `((64 − w)·e0 + w·e1 + 32) / 64` rounded down; a block with no mode reads
zero. -/
def bc7Texel (blk sub : Nat) : Array Nat := Id.run do
  let bit (pos n : Nat) : Nat := blk / 2 ^ pos % 2 ^ n
  let some m := (List.range 8).find? (bit · 1 == 1) | return #[0, 0, 0, 0]
  let mi := bc7Modes[m]!
  let mut pos := m + 1
  let part := bit pos mi.partBits
  pos := pos + mi.partBits
  let rot := bit pos mi.rotBits
  pos := pos + mi.rotBits
  let sel := bit pos mi.selBits
  pos := pos + mi.selBits
  let pb := if mi.endpointP > 0 then mi.endpointP else mi.sharedP
  -- Endpoints per channel: subset s's two at 2s and 2s + 1.
  let mut ep : Array (Array Nat) := #[#[], #[], #[], #[]]
  for c in [0:4] do
    let bits := if c < 3 then mi.colorBits else mi.alphaBits
    for _ in [0:2 * mi.subsets] do
      ep := ep.modify c (·.push (if bits == 0 then 255 else bit pos bits * 2 ^ pb))
      pos := pos + bits
  if pb > 0 then
    for s in [0:mi.subsets] do
      let pa := bit pos pb
      pos := pos + pb
      let pbit := if mi.sharedP == 0 then bit pos pb else pa
      if mi.sharedP == 0 then pos := pos + pb
      ep := ep.map fun e => (e.modify (2 * s) (· ||| pa)).modify (2 * s + 1) (· ||| pbit)
  let widen (v bits : Nat) : Nat := let x := v * 2 ^ (8 - bits) % 256; x ||| x / 2 ^ bits
  ep := ep.mapIdx fun c e =>
    let bits := if c < 3 then mi.colorBits else mi.alphaBits
    if bits == 0 then e else e.map (widen · (bits + pb))
  -- The texel's subset, and the texels before it in each index set (anchors one bit fewer).
  let subsetOf (i : Nat) : Nat := match mi.subsets with
    | 2 => bc7Two[part]! / 2 ^ i % 2
    | 3 => bc7Three[part]! / 2 ^ (2 * i) % 4
    | _ => 0
  let anchor (i : Nat) : Bool := match mi.subsets, subsetOf i with
    | _, 0 => i == 0
    | 2, _ => i == bc7TwoAnchor[part]!
    | _, s => i == bc7ThreeAnchor[s - 1]![part]!
  let width (n i : Nat) : Nat := if anchor i then n - 1 else n
  let at_ (start n : Nat) : Nat :=
    let off := (List.range sub).foldl (fun o i => o + width n i) 0
    bit (start + off) (width n sub)
  let i0 := at_ pos mi.index0
  let i1 := if mi.index1 > 0 then at_ (pos + mi.subsets * (16 * mi.index0 - 1)) mi.index1 else i0
  let w0 := bc7Weights[mi.index0 - 2]!
  let w1 := if mi.index1 > 0 then bc7Weights[mi.index1 - 2]! else w0
  let (fc, fa) := if sel == 0 then (w0[i0]!, w1[i1]!) else (w1[i1]!, w0[i0]!)
  let s := subsetOf sub
  let blend (c f : Nat) : Nat := ((64 - f) * ep[c]![2 * s]! + f * ep[c]![2 * s + 1]! + 32) / 64
  let mut rgba := #[blend 0 fc, blend 1 fc, blend 2 fc, blend 3 fa]
  if rot > 0 then rgba := (rgba.set! 3 rgba[rot - 1]!).set! (rot - 1) rgba[3]!
  return rgba

/-- BC6H's modes (Direct3D 11's BC6H format, "BC6H Implementation"): per mode number (its 2 or 5
leading bits), whether its endpoints after the first are deltas, its partition bits, its first
endpoint's bits and each channel's delta bits, and how its header's bits fill the endpoints in
order: a run of `n` bits into endpoint `i` of channel `c` (red, green, blue) from its bit `s`. -/
def bc6hModes : List (Nat × (Bool × Nat × Nat × Array Nat) × List (Nat × Nat × Nat × Nat)) := [
  (0, (true, 5, 10, #[5, 5, 5]),
    [(1, 2, 1, 4), (2, 2, 1, 4), (2, 3, 1, 4), (0, 0, 10, 0), (1, 0, 10, 0), (2, 0, 10, 0),
     (0, 1, 5, 0), (1, 3, 1, 4), (1, 2, 4, 0), (1, 1, 5, 0), (2, 3, 1, 0), (1, 3, 4, 0),
     (2, 1, 5, 0), (2, 3, 1, 1), (2, 2, 4, 0), (0, 2, 5, 0), (2, 3, 1, 2), (0, 3, 5, 0),
     (2, 3, 1, 3)]),
  (1, (true, 5, 7, #[6, 6, 6]),
    [(1, 2, 1, 5), (1, 3, 1, 4), (1, 3, 1, 5), (0, 0, 7, 0), (2, 3, 1, 0), (2, 3, 1, 1),
     (2, 2, 1, 4), (1, 0, 7, 0), (2, 2, 1, 5), (2, 3, 1, 2), (1, 2, 1, 4), (2, 0, 7, 0),
     (2, 3, 1, 3), (2, 3, 1, 5), (2, 3, 1, 4), (0, 1, 6, 0), (1, 2, 4, 0), (1, 1, 6, 0),
     (1, 3, 4, 0), (2, 1, 6, 0), (2, 2, 4, 0), (0, 2, 6, 0), (0, 3, 6, 0)]),
  (2, (true, 5, 11, #[5, 4, 4]),
    [(0, 0, 10, 0), (1, 0, 10, 0), (2, 0, 10, 0), (0, 1, 5, 0), (0, 0, 1, 10), (1, 2, 4, 0),
     (1, 1, 4, 0), (1, 0, 1, 10), (2, 3, 1, 0), (1, 3, 4, 0), (2, 1, 4, 0), (2, 0, 1, 10),
     (2, 3, 1, 1), (2, 2, 4, 0), (0, 2, 5, 0), (2, 3, 1, 2), (0, 3, 5, 0), (2, 3, 1, 3)]),
  (3, (false, 0, 10, #[10, 10, 10]),
    [(0, 0, 10, 0), (1, 0, 10, 0), (2, 0, 10, 0), (0, 1, 10, 0), (1, 1, 10, 0), (2, 1, 10, 0)]),
  (6, (true, 5, 11, #[4, 5, 4]),
    [(0, 0, 10, 0), (1, 0, 10, 0), (2, 0, 10, 0), (0, 1, 4, 0), (0, 0, 1, 10), (1, 3, 1, 4),
     (1, 2, 4, 0), (1, 1, 5, 0), (1, 0, 1, 10), (1, 3, 4, 0), (2, 1, 4, 0), (2, 0, 1, 10),
     (2, 3, 1, 1), (2, 2, 4, 0), (0, 2, 4, 0), (2, 3, 1, 0), (2, 3, 1, 2), (0, 3, 4, 0),
     (1, 2, 1, 4), (2, 3, 1, 3)]),
  (7, (true, 0, 11, #[9, 9, 9]),
    [(0, 0, 10, 0), (1, 0, 10, 0), (2, 0, 10, 0), (0, 1, 9, 0), (0, 0, 1, 10), (1, 1, 9, 0),
     (1, 0, 1, 10), (2, 1, 9, 0), (2, 0, 1, 10)]),
  (10, (true, 5, 11, #[4, 4, 5]),
    [(0, 0, 10, 0), (1, 0, 10, 0), (2, 0, 10, 0), (0, 1, 4, 0), (0, 0, 1, 10), (2, 2, 1, 4),
     (1, 2, 4, 0), (1, 1, 4, 0), (1, 0, 1, 10), (2, 3, 1, 0), (1, 3, 4, 0), (2, 1, 5, 0),
     (2, 0, 1, 10), (2, 2, 4, 0), (0, 2, 4, 0), (2, 3, 1, 1), (2, 3, 1, 2), (0, 3, 4, 0),
     (2, 3, 1, 4), (2, 3, 1, 3)]),
  (11, (true, 0, 12, #[8, 8, 8]),
    [(0, 0, 10, 0), (1, 0, 10, 0), (2, 0, 10, 0), (0, 1, 8, 0), (0, 0, 1, 11), (0, 0, 1, 10),
     (1, 1, 8, 0), (1, 0, 1, 11), (1, 0, 1, 10), (2, 1, 8, 0), (2, 0, 1, 11), (2, 0, 1, 10)]),
  (14, (true, 5, 9, #[5, 5, 5]),
    [(0, 0, 9, 0), (2, 2, 1, 4), (1, 0, 9, 0), (1, 2, 1, 4), (2, 0, 9, 0), (2, 3, 1, 4),
     (0, 1, 5, 0), (1, 3, 1, 4), (1, 2, 4, 0), (1, 1, 5, 0), (2, 3, 1, 0), (1, 3, 4, 0),
     (2, 1, 5, 0), (2, 3, 1, 1), (2, 2, 4, 0), (0, 2, 5, 0), (2, 3, 1, 2), (0, 3, 5, 0),
     (2, 3, 1, 3)]),
  (15, (true, 0, 16, #[4, 4, 4]),
    [(0, 0, 10, 0), (1, 0, 10, 0), (2, 0, 10, 0), (0, 1, 4, 0), (0, 0, 1, 15), (0, 0, 1, 14),
     (0, 0, 1, 13), (0, 0, 1, 12), (0, 0, 1, 11), (0, 0, 1, 10), (1, 1, 4, 0), (1, 0, 1, 15),
     (1, 0, 1, 14), (1, 0, 1, 13), (1, 0, 1, 12), (1, 0, 1, 11), (1, 0, 1, 10), (2, 1, 4, 0),
     (2, 0, 1, 15), (2, 0, 1, 14), (2, 0, 1, 13), (2, 0, 1, 12), (2, 0, 1, 11), (2, 0, 1, 10)]),
  (18, (true, 5, 8, #[6, 5, 5]),
    [(0, 0, 8, 0), (1, 3, 1, 4), (2, 2, 1, 4), (1, 0, 8, 0), (2, 3, 1, 2), (1, 2, 1, 4),
     (2, 0, 8, 0), (2, 3, 1, 3), (2, 3, 1, 4), (0, 1, 6, 0), (1, 2, 4, 0), (1, 1, 5, 0),
     (2, 3, 1, 0), (1, 3, 4, 0), (2, 1, 5, 0), (2, 3, 1, 1), (2, 2, 4, 0), (0, 2, 6, 0),
     (0, 3, 6, 0)]),
  (22, (true, 5, 8, #[5, 6, 5]),
    [(0, 0, 8, 0), (2, 3, 1, 0), (2, 2, 1, 4), (1, 0, 8, 0), (1, 2, 1, 5), (1, 2, 1, 4),
     (2, 0, 8, 0), (1, 3, 1, 5), (2, 3, 1, 4), (0, 1, 5, 0), (1, 3, 1, 4), (1, 2, 4, 0),
     (1, 1, 6, 0), (1, 3, 4, 0), (2, 1, 5, 0), (2, 3, 1, 1), (2, 2, 4, 0), (0, 2, 5, 0),
     (2, 3, 1, 2), (0, 3, 5, 0), (2, 3, 1, 3)]),
  (26, (true, 5, 8, #[5, 5, 6]),
    [(0, 0, 8, 0), (2, 3, 1, 1), (2, 2, 1, 4), (1, 0, 8, 0), (2, 2, 1, 5), (1, 2, 1, 4),
     (2, 0, 8, 0), (2, 3, 1, 5), (2, 3, 1, 4), (0, 1, 5, 0), (1, 3, 1, 4), (1, 2, 4, 0),
     (1, 1, 5, 0), (2, 3, 1, 0), (1, 3, 4, 0), (2, 1, 6, 0), (2, 2, 4, 0), (0, 2, 5, 0),
     (2, 3, 1, 2), (0, 3, 5, 0), (2, 3, 1, 3)]),
  (30, (false, 5, 6, #[6, 6, 6]),
    [(0, 0, 6, 0), (1, 3, 1, 4), (2, 3, 1, 0), (2, 3, 1, 1), (2, 2, 1, 4), (1, 0, 6, 0),
     (1, 2, 1, 5), (2, 2, 1, 5), (2, 3, 1, 2), (1, 2, 1, 4), (2, 0, 6, 0), (1, 3, 1, 5),
     (2, 3, 1, 3), (2, 3, 1, 5), (2, 3, 1, 4), (0, 1, 6, 0), (1, 2, 4, 0), (1, 1, 6, 0),
     (1, 3, 4, 0), (2, 1, 6, 0), (2, 2, 4, 0), (0, 2, 6, 0), (0, 3, 6, 0)])]

/-- A BC6H block's (`blk`, its 128 bits) texel at `sub` as red, green and blue binary16 bits, as
the format defines it, which an A100 and a GH200 match on every texel of random blocks, unsigned
and signed (`signed`): the mode, its endpoints (a delta mode's later ones added to the first),
each widened to 16 bits (`unq`), the partition and the texel's index blending its subset's
endpoints, `((64 − w)·e0 + w·e1 + 32) / 64` rounded down, then scaled to binary16's range
(`·31/64`, signed `·31/32` of the magnitude); a block of a reserved mode reads zero. -/
def bc6hTexel (blk sub : Nat) (signed : Bool) : Array Nat := Id.run do
  let bit (pos n : Nat) : Nat := blk / 2 ^ pos % 2 ^ n
  let mode := if bit 0 2 < 2 then bit 0 2 else bit 0 5
  let some (_, (delta, pbits, eb, db), steps) := bc6hModes.find? (·.1 == mode) | return #[0, 0, 0]
  let mut pos := if bit 0 2 < 2 then 2 else 5
  let mut ep : Array (Array Int) := Array.replicate 3 (Array.replicate 4 0)
  for (c, i, n, s) in steps do
    ep := ep.modify c (·.modify i (· + (bit pos n * 2 ^ s : Nat)))
    pos := pos + n
  -- `v`'s low `n` bits (two's complement) as a signed number.
  let sx (v : Int) (n : Nat) : Int := let u := (v % (2 ^ n : Nat)).toNat; if u ≥ 2 ^ (n - 1) then u - 2 ^ n else u
  let subsets := if pbits > 0 then 2 else 1
  let unq (v : Int) : Int :=
    if signed then
      if eb ≥ 16 then v else
      let a := v.natAbs
      let u : Nat := if a == 0 then 0 else if a ≥ 2 ^ (eb - 1) - 1 then 0x7fff
        else (a * 2 ^ 15 + 0x4000) / 2 ^ (eb - 1)
      if v < 0 then -(u : Int) else u
    else if eb ≥ 15 then v
    else if v == 0 then 0 else if v == 2 ^ eb - 1 then 0xffff
    else (v.toNat * 2 ^ 16 + 0x8000) / 2 ^ eb
  for c in [0:3] do
    let mut e := ep[c]!
    if signed then e := e.set! 0 (sx e[0]! eb)
    for i in [1:2 * subsets] do
      if signed || delta then e := e.set! i (sx e[i]! db[c]!)
      if delta then
        e := e.set! i ((e[i]! + e[0]!) % (2 ^ eb : Nat))
        if signed then e := e.set! i (sx e[i]! eb)
    ep := ep.set! c (e.map unq)
  let part := if pbits > 0 then bit pos 5 else 0
  if pbits > 0 then pos := pos + 5
  let ib := if pbits > 0 then 3 else 4
  let subsetOf (i : Nat) : Nat := if pbits > 0 then bc7Two[part]! / 2 ^ i % 2 else 0
  let width (i : Nat) : Nat :=
    if i == (if subsetOf i == 0 then 0 else bc7TwoAnchor[part]!) then ib - 1 else ib
  let off := (List.range sub).foldl (fun o i => o + width i) 0
  let f := bc7Weights[ib - 2]![bit (pos + off) (width sub)]!
  let s := subsetOf sub
  let finish (v : Int) : Nat :=
    if signed then
      if v < 0 then 0x8000 + v.natAbs * 31 / 32 else v.toNat * 31 / 32
    else v.toNat * 31 / 64
  return (Array.range 3).map fun c =>
    finish (Int.fdiv (ep[c]![2 * s]! * (64 - f : Nat) + ep[c]![2 * s + 1]! * (f : Nat) + 32) 64)

/-- A binary16's value as `numerator / denominator` (finite: BC6H's are). -/
def halfValue (h : Nat) : Int × Nat :=
  let e := h / 1024 % 32
  let m := h % 1024
  let mag : Nat × Nat :=
    if e == 0 then (m, 2 ^ 24) else if e ≥ 25 then ((1024 + m) * 2 ^ (e - 25), 1) else (1024 + m, 2 ^ (25 - e))
  (if h ≥ 0x8000 then -(mag.1 : Int) else mag.1, mag.2)

/-- A block-compressed texel's channels (red, green, blue, alpha) as `numerator / denominator`, from
its block's words `ws` and its place `sub`: BC1's colour, alpha one or (transparent) zero; BC2's
colour from its second half and alpha from its first, 4 bits a texel; BC3's colour from its second
half and alpha (8-bit) from its first; BC6H's binary16 as `bc6hTexel` (alpha one); BC7's as
`bc7Texel`; BC4 red (green and blue zero, alpha one), BC5 red and green from its two halves (blue
zero, alpha one). The colour and BC3's alpha match an A100 and a GH200 on every texel of random
blocks, the colour on every pair of endpoints per channel. -/
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
  | 7 => (rgb (bc7Texel (half 0 + half 1 * 2 ^ 64) sub)).toArray
  | 6 => (bc6hTexel (half 0 + half 1 * 2 ^ 64) sub (t.bcFormat == 0x9c)).map halfValue |>.push (1, 1)
  | 4 => #[bc4Value (half 0) sub, (0, 1), (0, 1), (1, 1)]
  | 5 => #[bc4Value (half 0) sub, bc4Value (half 1) sub, (0, 1), (1, 1)]
  | _ => #[(0, 1), (0, 1), (0, 1), (1, 1)]

/-- An sRGB texel's 8-bit colour value read as linear (binary32's upper 16 bits): an A100's and a
GH200's own table, which rounds to 8 significant bits and departs from IEC 61966-2-1's curve by
up to 0.004 (`srgb.cu`). -/
def srgbTable : Array UInt16 := #[
  0x0000, 0x39a0, 0x3a20, 0x3a70, 0x3aa0, 0x3ac8, 0x3af0, 0x3b0c, 0x3b20, 0x3b34, 0x3b48, 0x3b5c, 0x3b70, 0x3b84, 0x3b90, 0x3b9c,
  0x3baa, 0x3bb8, 0x3bc6, 0x3bd6, 0x3be6, 0x3bf6, 0x3c03, 0x3c0c, 0x3c16, 0x3c1f, 0x3c29, 0x3c34, 0x3c3e, 0x3c49, 0x3c55, 0x3c60,
  0x3c6c, 0x3c7a, 0x3c83, 0x3c8a, 0x3c91, 0x3c98, 0x3c9f, 0x3ca6, 0x3cae, 0x3cb6, 0x3cbe, 0x3cc6, 0x3cce, 0x3cd7, 0x3ce0, 0x3ce9,
  0x3cf2, 0x3cfc, 0x3d03, 0x3d08, 0x3d0d, 0x3d12, 0x3d17, 0x3d1c, 0x3d22, 0x3d28, 0x3d2d, 0x3d33, 0x3d39, 0x3d3f, 0x3d45, 0x3d4c,
  0x3d52, 0x3d58, 0x3d60, 0x3d66, 0x3d6c, 0x3d74, 0x3d7a, 0x3d81, 0x3d85, 0x3d88, 0x3d8c, 0x3d90, 0x3d94, 0x3d98, 0x3d9c, 0x3da0,
  0x3da4, 0x3da9, 0x3dad, 0x3db1, 0x3db6, 0x3dba, 0x3dbf, 0x3dc3, 0x3dc8, 0x3dcd, 0x3dd1, 0x3dd6, 0x3ddb, 0x3de0, 0x3de5, 0x3dea,
  0x3df0, 0x3df4, 0x3dfa, 0x3e00, 0x3e02, 0x3e05, 0x3e08, 0x3e0b, 0x3e0e, 0x3e11, 0x3e14, 0x3e17, 0x3e1a, 0x3e1d, 0x3e20, 0x3e23,
  0x3e26, 0x3e29, 0x3e2c, 0x3e30, 0x3e33, 0x3e36, 0x3e3a, 0x3e3d, 0x3e40, 0x3e44, 0x3e47, 0x3e4b, 0x3e4e, 0x3e52, 0x3e56, 0x3e59,
  0x3e5d, 0x3e61, 0x3e65, 0x3e68, 0x3e6c, 0x3e70, 0x3e74, 0x3e78, 0x3e7c, 0x3e80, 0x3e82, 0x3e84, 0x3e86, 0x3e88, 0x3e8a, 0x3e8d,
  0x3e8f, 0x3e91, 0x3e93, 0x3e95, 0x3e98, 0x3e9a, 0x3e9c, 0x3e9e, 0x3ea1, 0x3ea3, 0x3ea5, 0x3ea8, 0x3eaa, 0x3ead, 0x3eaf, 0x3eb2,
  0x3eb4, 0x3eb6, 0x3eb9, 0x3ebc, 0x3ebe, 0x3ec1, 0x3ec3, 0x3ec6, 0x3ec8, 0x3ecb, 0x3ece, 0x3ed1, 0x3ed3, 0x3ed6, 0x3ed9, 0x3edb,
  0x3ede, 0x3ee1, 0x3ee4, 0x3ee7, 0x3eea, 0x3eed, 0x3ef0, 0x3ef2, 0x3ef6, 0x3ef8, 0x3efc, 0x3efe, 0x3f01, 0x3f02, 0x3f04, 0x3f05,
  0x3f07, 0x3f09, 0x3f0a, 0x3f0c, 0x3f0d, 0x3f0f, 0x3f11, 0x3f12, 0x3f14, 0x3f16, 0x3f17, 0x3f19, 0x3f1b, 0x3f1c, 0x3f1e, 0x3f20,
  0x3f21, 0x3f23, 0x3f25, 0x3f27, 0x3f29, 0x3f2a, 0x3f2c, 0x3f2e, 0x3f30, 0x3f32, 0x3f33, 0x3f35, 0x3f37, 0x3f39, 0x3f3b, 0x3f3d,
  0x3f3f, 0x3f41, 0x3f43, 0x3f45, 0x3f47, 0x3f49, 0x3f4b, 0x3f4d, 0x3f4f, 0x3f51, 0x3f53, 0x3f55, 0x3f57, 0x3f59, 0x3f5b, 0x3f5d,
  0x3f5f, 0x3f61, 0x3f63, 0x3f65, 0x3f68, 0x3f6a, 0x3f6c, 0x3f6e, 0x3f70, 0x3f72, 0x3f74, 0x3f76, 0x3f7a, 0x3f7c, 0x3f7e, 0x3f80]

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
ties away from zero, an unsigned normalized sum first rounded half up to 16-bit unsigned
normalized (an A100 and a GH200 on every one of 4,096 fetches of 8- and 16-bit texels and of
blends of levels) (a GH200's linear filtering in one and two dimensions and between levels of
linearly filtered mipmaps, and its and an A100's blends of point-sampled levels); a border texel
reads the border colour (`borderVal`). Integers read as they are (point sampling only): the texel's
channels, sign-extended. A texel's missing channels read as zero (inferred). -/
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
    else if t.srgb && c < 3 then
      -- sRGB (point sampling only, `invalid?`): the texel's 8-bit value through the table.
      match fp.zipIdx.find? (·.1.1.isSome) with
      | some ((some x, _), k) =>
        let v : Nat := if t.bc != 0 then (bcTexel t (texel k) x.sub)[c]!.1.toNat else channel t (texel k) c
        (srgbTable[v]?.getD 0).toUInt32 <<< 16
      | _ => 0
    else
      let wsum := fp.foldl (· + ·.2) 0
      -- Point sampling of a float returns the texel unchanged (NaN included).
      if t.kind == .float && t.bc == 0 && fp.length == 1 then
        match fp with
        | [(some _, _)] =>
          let v := channel t (texel 0) c
          if w == 16 then (cvt Format.f16 Format.f32 .rn {} v).toUInt32 else v.toUInt32
        -- A border texel reads the border colour as given; no texel (weight 0) reads zero.
        | [(none, w)] => if w == 0 then 0 else t.border[c]?.getD 0
        | _ => 0
      else
        -- Σ weight_k · v_k over a common denominator.
        let terms := fp.zipIdx.filterMap fun ((a, wt), k) =>
          match a with
          | none => if wt == 0 then none else borderVal t c |>.map (wt, ·)
          | some x =>
            if wt == 0 then none
            else if t.bc != 0 then
              let (n, d) := (bcTexel t (texel k) x.sub)[c]!
              some (wt, n, d, (0 : Int))
            else channelVal t (channel t (texel k) c) |>.map (wt, ·)
        let emin := terms.foldl (fun m (_, _, _, e) => min m e) 0
        let den := terms.foldl (fun d (_, _, dn, _) => Nat.lcm d dn) 1
        let num : Int := terms.foldl (fun s (wt, n, dn, e) =>
          s + (wt : Int) * n * (den / dn : Nat) * (2 ^ (e - emin).toNat : Nat)) 0
        -- Unsigned normalized: the sum rounded half up to 16-bit unsigned normalized first.
        if t.kind == .unsigned then
          let all : Int := (den * wsum : Nat)
          let q := Int.fdiv (2 * num * 65535 + all) (2 * all)
          toF32 q 65535 0
        else toF32 num (den * wsum) emin

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
