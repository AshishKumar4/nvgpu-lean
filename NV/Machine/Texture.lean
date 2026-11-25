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
  deriving Repr, Inhabited

namespace Texture

def texelBytes (t : Texture) : Nat := t.chans.foldl (· + ·) 0 / 8

def size (t : Texture) (d : Nat) : Nat := match d with | 0 => t.width | 1 => t.height | _ => t.depth

/-- The texture's own inconsistencies, if any. -/
def invalid? (t : Texture) : Option String :=
  if !(t.dims ∈ [1, 2, 3]) then some s!"{t.dims} dimensions"
  else if !(t.chans.size ∈ [1, 2, 4]) then some s!"{t.chans.size} channels"
  else if t.chans.any (· != t.chans[0]!) then some "channels of different widths"
  else if !(t.texelBytes ∈ [1, 2, 4, 8, 16]) then some s!"{t.texelBytes}-byte texels"
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

/-- Layer `k` of a layered object, as an object of its own (filtering stays within a layer, CUDA
Programming Guide, layered textures). A layer index past the last is left undefined by the
ISA (§9.7.12.3: an unsigned index into the array), so it is an error. -/
def layer (t : Texture) (k : Nat) : Except String Texture :=
  if t.layers == 0 then .error "an array access to an object that is not layered"
  else if k ≥ t.layers then .error s!"layer {k} of {t.layers}"
  else .ok { t with base := t.base + (k * (if t.dims == 1 then t.pitch else t.slice)).toUInt64,
                    layers := 0 }

/-- Level `k` of a mipmapped object, as an object of its own: each dimension half the last
level's (at least 1), rows 128-byte aligned, the levels one after another from level 0's base
(the driver's layout, `tools/cudadev/tex.h`). -/
def level (t : Texture) (k : Nat) : Texture := Id.run do
  let mut base := t.base
  let mut lv := { t with levels := 1 }
  for j in [0:k + 1] do
    let w := max 1 (t.width >>> j)
    let h := if t.dims ≥ 2 then max 1 (t.height >>> j) else 1
    let d := if t.dims ≥ 3 then max 1 (t.depth >>> j) else 1
    let pitch := (w * t.texelBytes + 127) / 128 * 128
    lv := { lv with base, width := w, height := h, depth := d, pitch, slice := pitch * h }
    base := base + (pitch * h * d).toUInt64
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
footprint: nearest-level filtering the level nearest `l` (halves up); linear filtering the two
levels about `l` (the last level twice past it), each footprint's weights scaled by its share
out of 256, so that every fetch of the object reads as many texels. -/
def mipFootprint (t : Texture) (l : Int)
    (fpAt : Texture → List (Option UInt64 × Nat)) : List (Option UInt64 × Nat) :=
  let last := t.levels - 1
  if !t.mipLinear then fpAt (t.level (min last ((l + 128) / 256).toNat)) else
  let k := min last (l / 256).toNat
  let f := if k == last then 0 else (l % 256).toNat
  (fpAt (t.level k)).map (fun (a, w) => (a, w * (256 - f))) ++ (fpAt (t.level (min last (k + 1)))).map (fun (a, w) => (a, w * f))

/-- The address of texel `(i, j, k)`, which must be in range. -/
def texelAddr (t : Texture) (i j k : Nat) : UInt64 :=
  t.base + (k * t.slice + j * t.pitch + i * t.texelBytes).toUInt64

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
`frac(x − 1/2)` rounded to the nearest 1/256 (inferred). -/
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
    let a := ((xb.fracUnits 9) + 1) / 2
    [(resolve mode n i, (256 - a).toNat), (resolve mode n (i + 1), a.toNat)]

/-- The footprint of a fetch at coordinates `cs` (one per dimension), each moved by `offs` texels
(PTX's offset operand, §9.7.12.3: "applied to coordinates before doing texture lookup", from -8
to 7; taken in texel units, after normalized coordinates are scaled and wrapped: inferred): each
texel's address (`none`: a border texel, zero) and its weight, out of `256^dims` or, linearly
filtering two dimensions, out of 256: the far corner's weight `p`, the product of the two axes'
weights `a`, `b` in 1/256 rounded half up, and the others `a - p`, `b - p` and `256 - a - b + p`
(a GH200 agrees on 4,096 fetches; three dimensions take the product, inferred). -/
def footprint (t : Texture) (cs : Array Dy) (offs : Array Int := #[]) : List (Option UInt64 × Nat) :=
  let along (d : Nat) : List (Option Nat × Nat) :=
    if d < t.dims then taps t d (cs[d]?.getD ⟨0, 0⟩) (offs[d]?.getD 0) else [(some 0, 256)]
  match t.dims, t.filter, along 0, along 1 with
  | 2, .linear, [(i0, _), (i1, a)], [(j0, _), (j1, b)] =>
    let p := (a * b + 128) / 256
    let at_ (i j : Option Nat) : Option UInt64 := do pure (t.texelAddr (← i) (← j) 0)
    [(at_ i0 j0, 256 + p - a - b), (at_ i1 j0, a - p), (at_ i0 j1, b - p), (at_ i1 j1, p)]
  | _, _, _, _ =>
  (along 2).flatMap fun (k, wk) => (along 1).flatMap fun (j, wj) => (along 0).map fun (i, wi) =>
    let a := do let i ← i; let j ← j; let k ← k; pure (t.texelAddr i j k)
    (a, wi * wj * wk)

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
ties away from zero (a GH200's linear filtering in one and two dimensions, and its and an A100's
blends of point-sampled mipmap levels). Integers
read as they are (point sampling only): the texel's channels, sign-extended. A texel's missing
channels read as zero (inferred). -/
def combine (t : Texture) (fp : List (Option UInt64 × Nat)) (resp : Array UInt32) : Array UInt32 :=
  let wpt := (t.texelBytes + 3) / 4
  let texel (k : Nat) : Array UInt32 := resp.extract (k * wpt) (k * wpt + wpt)
  let w := t.chans[0]!
  (Array.range 4).map fun c =>
    if c ≥ t.chans.size then 0 else
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
          if a.isNone || wt == 0 then none else channelVal t (channel t (texel k) c) |>.map (wt, ·)
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
