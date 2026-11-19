import Lean.Data.Json
import NV.Async.TensorMap
import NV.Core.Engine
import NV.Num

/-!
# Launch specs

Everything a run of kernels needs besides their code, as JSON: buffers in global memory with
their contents, tensor maps by their `cuTensorMapEncodeTiled` arguments, and steps in order:
launches (which kernel, the grid, the parameter values) and host writes between them. Launches
share global memory, so a sequence (an application's iterations, a model layer's kernels) runs as
it did on the device. The parameter block's layout is the kernel's ELF's (`EIATTR_KPARAM_INFO`,
read into `SASS.Kernel.params`), so a spec names parameters by ordinal and the build places them.
A spec with one launch may give it at the top level, as below.

```json
{ "code": "gemm", "function": "fp8_gemm",
  "grid": [132, 1, 1], "block": [384, 1, 1], "cluster": [2, 1, 1], "dynamicShared": 232448,
  "buffers": [
    { "name": "a", "bytes": 1048576, "init": { "gen": "uniform", "dtype": "e4m3", "lo": -64,
      "hi": 64, "scale": -4, "seed": 1 } },
    { "name": "d", "bytes": 2097152 } ],
  "tensorMaps": [
    { "name": "tma_a", "dtype": "u8", "buffer": "a", "dims": [1024, 1024], "strides": [1024],
      "box": [128, 128], "swizzle": "128B" } ],
  "params": [
    { "ordinal": 0, "ptr": "d" }, { "ordinal": 1, "u32": 1024 },
    { "ordinal": 2, "tensorMap": "tma_a" },
    { "ordinal": 3, "fields": [ { "offset": 0, "ptr": "a", "offset_in": 256 },
                                { "offset": 8, "s32": -1 } ] } ] }
```

An im2col tensor map (`cuTensorMapEncodeIm2col`) has `"im2col": { "lower": [W, H, D…], "upper":
[…], "channelsPerPixel", "pixelsPerColumn" }` (rank − 2 corners each) in place of `box`.

Steps: `{ "launch": { "code", "function", "grid", "block", "cluster", "dynamicShared",
"params" } }` and `{ "write": { "buffer", "offset", <value> } }`. `code` names the listing the
function is in (the caller resolves it).

Values (a parameter, a struct field, or a `writes` entry of a buffer) are one of: `ptr` (a
buffer's address plus `offset_in` bytes), `u8`/`u16`/`u32`/`u64`/`s32`/`s64`, `f16`/`bf16`/`f32`/
`f64` (a decimal, rounded to nearest even), `tensorMap` (the 128-byte encoding of
`NV.Async.TensorMap`), `texture` (a texture or surface object's 64-bit handle), `hex` (bytes
in order), or `fields` (values at byte offsets). Buffer contents come from `init` (default zero),
then `writes`.

Texture and surface objects (`NV.Tex.Texture`, every launch gets the whole table; handle `i + 1`
names entry `i`): `"textures": [ { "name", "buffer", "offset", "dims": [w, h, d], "pitch",
"slice", "channels": [bits, …], "kind": "unsigned"|"signed"|"float", "readMode":
"element"|"normalized", "normalizedCoords", "filter": "point"|"linear", "address":
["clamp"|"border"|"wrap"|"mirror", …], "surface" } ]`, `pitch` and `slice` defaulting to packed
rows and slices.

Generators are deterministic: element `i` of a buffer seeded `s` draws `splitmix64(s, i)`, so a
buffer's contents depend on the spec alone, never on generation order. `uniform` draws an integer
`k` in `[lo, hi]` (modulo reduction: a bias below `(hi - lo + 1) / 2^64`) and stores `k · 2^scale`,
rounded to nearest even into a floating-point `dtype`, or `k` itself into an integer one.
-/

namespace NV.Launch

open Lean (Json JsonNumber)
open NV.Async (TensorMap)

/-- Element types of buffers and scalars. -/
inductive DType where
  | u8 | s8 | u16 | s16 | u32 | s32 | u64 | s64 | f16 | bf16 | tf32 | f32 | f64 | e4m3 | e5m2
  deriving DecidableEq, Repr, Inhabited

namespace DType

def all : List (String × DType) :=
  [("u8", .u8), ("s8", .s8), ("u16", .u16), ("s16", .s16), ("u32", .u32), ("s32", .s32),
   ("u64", .u64), ("s64", .s64), ("f16", .f16), ("bf16", .bf16), ("tf32", .tf32), ("f32", .f32),
   ("f64", .f64), ("e4m3", .e4m3), ("e5m2", .e5m2)]

def ofString? (s : String) : Option DType := all.lookup s

def bytes : DType → Nat
  | .u8 | .s8 | .e4m3 | .e5m2 => 1
  | .u16 | .s16 | .f16 | .bf16 => 2
  | .u32 | .s32 | .f32 | .tf32 => 4
  | .u64 | .s64 | .f64 => 8

def format? : DType → Option Num.Format
  | .f16 => some .f16
  | .bf16 => some .bf16
  | .tf32 => some .tf32
  | .f32 => some .f32
  | .f64 => some .f64
  | .e4m3 => some .e4m3
  | .e5m2 => some .e5m2
  | _ => none

/-- The bits of `k · 2^scale`: two's complement for an integer type (`scale` ignored), rounded to
nearest even for a floating-point one. -/
def bits (d : DType) (k : Int) (scale : Int := 0) : Nat :=
  match d.format? with
  | some f => f.round .rn .ieee (k < 0) k.natAbs scale
  | none => (k % (2 ^ (8 * d.bytes) : Int)).toNat

end DType

/-- SplitMix64 (Steele, Lea and Flood, OOPSLA 2014): the `i`-th output of the stream seeded
`seed`, computed directly. -/
def splitmix64 (seed i : UInt64) : UInt64 :=
  let z := seed + (i + 1) * 0x9E3779B97F4A7C15
  let z := (z ^^^ (z >>> 30)) * 0xBF58476D1CE4E5B9
  let z := (z ^^^ (z >>> 27)) * 0x94D049BB133111EB
  z ^^^ (z >>> 31)

inductive Gen where
  | zero
  /-- Every element `k · 2^scale`. -/
  | fill (d : DType) (k : Int) (scale : Int)
  /-- Element `i` is `start + (i / period) · step` (JSON `repeat`, default 1). -/
  | iota (d : DType) (start step : Int) (period : Nat)
  | uniform (d : DType) (lo hi : Int) (scale : Int) (seed : UInt64)
  /-- A file's bytes, from the start of the buffer (the caller resolves the path). -/
  | file (path : String)
  deriving Repr, Inhabited

/-- One piece of a value. -/
inductive Leaf where
  /-- A buffer's address plus an offset. -/
  | ptr (buffer : String) (offset : Nat)
  /-- An integer or float of a scalar type, as its bits. -/
  | scalar (d : DType) (bits : Nat)
  | tensorMap (name : String)
  /-- A texture or surface object's handle (`NV.Tex`: entry `i` of the table is handle `i + 1`). -/
  | texture (name : String)
  | bytes (b : Array UInt8)
  deriving Repr, Inhabited

/-- A value placed at a byte offset of the parameter block or of a buffer: leaves at offsets
within it, written in order (nested `fields` are flattened when read). -/
abbrev Value := Array (Nat × Leaf)

structure Buffer where
  name : String
  bytes : Nat
  /-- Alignment of its address (bytes, a power of two). -/
  align : Nat := 256
  init : Gen := .zero
  writes : Array (Nat × Value) := #[]
  deriving Repr, Inhabited

/-- A tensor map whose global address is a buffer's. -/
structure MapSpec where
  name : String
  buffer : String
  offset : Nat := 0
  /-- The map with `globalAddress` zero; the build adds the buffer's address. -/
  map : TensorMap
  deriving Repr, Inhabited

/-- A texture or surface object over a buffer (`NV.Tex.Texture` with its base the buffer's address
plus `offset`). -/
structure TexSpec where
  name : String
  buffer : String
  offset : Nat := 0
  /-- The object with `base` zero; the build adds the buffer's address. -/
  tex : NV.Tex.Texture
  deriving Repr, Inhabited

structure LaunchStep where
  /-- The listing the kernel is in, and its name there. -/
  code : String
  function : String
  grid : Nat × Nat × Nat
  block : Nat × Nat × Nat
  /-- `none`: the kernel's required cluster shape, else no clusters. -/
  cluster : Option (Nat × Nat × Nat) := none
  dynamicShared : Nat := 0
  params : Array (Nat × Value) := #[]
  deriving Repr, Inhabited

inductive Step where
  | launch (l : LaunchStep)
  /-- The host writes a value into a buffer. -/
  | write (buffer : String) (offset : Nat) (v : Value)
  deriving Repr, Inhabited

/-- After the last step, a buffer's bytes from `offset` must equal a file's: the outputs the
kernels produced on the device. -/
structure Expect where
  buffer : String
  offset : Nat := 0
  file : String
  deriving Repr, Inhabited

structure Spec where
  buffers : Array Buffer := #[]
  tensorMaps : Array MapSpec := #[]
  /-- Texture and surface objects; every launch gets the whole table. -/
  textures : Array TexSpec := #[]
  steps : Array Step := #[]
  expect : Array Expect := #[]
  /-- After the last step, write these buffers to these files. -/
  outputs : Array (String × String) := #[]
  deriving Repr, Inhabited

def Spec.launches (s : Spec) : Array LaunchStep :=
  s.steps.filterMap fun | .launch l => some l | .write .. => none

/-! ## Reading the JSON -/

section Json

private def field (j : Json) (k : String) : Except String Json :=
  j.getObjVal? k |>.mapError fun _ => s!"missing \"{k}\""

private def opt (j : Json) (k : String) : Option Json :=
  match j.getObjVal? k with
  | .ok v => if v.isNull then none else some v
  | .error _ => none

private def nat (j : Json) (k : String) : Except String Nat := do
  (← field j k).getNat? |>.mapError fun _ => s!"\"{k}\": not a natural number"

private def natD (j : Json) (k : String) (d : Nat) : Except String Nat :=
  if (opt j k).isSome then nat j k else pure d

private def int (j : Json) (k : String) : Except String Int := do
  (← field j k).getInt? |>.mapError fun _ => s!"\"{k}\": not an integer"

private def intD (j : Json) (k : String) (d : Int) : Except String Int :=
  if (opt j k).isSome then int j k else pure d

private def str (j : Json) (k : String) : Except String String := do
  (← field j k).getStr? |>.mapError fun _ => s!"\"{k}\": not a string"

private def nats (j : Json) : Except String (Array Nat) := do
  (← j.getArr?).mapM (·.getNat?)

private def dims3 (j : Json) (k : String) : Except String (Nat × Nat × Nat) := do
  match ← nats (← field j k) with
  | #[x, y, z] => pure (x, y, z)
  | #[x, y] => pure (x, y, 1)
  | #[x] => pure (x, 1, 1)
  | _ => throw s!"\"{k}\": one to three dimensions"

private def dtype (s : String) : Except String DType :=
  (DType.ofString? s).elim (throw s!"unknown dtype \"{s}\"") pure

private def hexDigit? (c : Char) : Option Nat :=
  if '0' ≤ c && c ≤ '9' then some (c.toNat - '0'.toNat)
  else if 'a' ≤ c && c ≤ 'f' then some (c.toNat - 'a'.toNat + 10)
  else if 'A' ≤ c && c ≤ 'F' then some (c.toNat - 'A'.toNat + 10)
  else none

private def hexBytes (s : String) : Except String (Array UInt8) := do
  let ds ← s.toList.mapM fun c => (hexDigit? c).elim (throw s!"\"{s}\": not hexadecimal") pure
  if ds.length % 2 != 0 then throw s!"\"{s}\": odd number of hex digits"
  let rec go : List Nat → List UInt8
    | a :: b :: rest => (a * 16 + b).toUInt8 :: go rest
    | _ => []
  return (go ds).toArray

/-- A decimal JSON number `m · 10^-e` rounded into a floating-point type. -/
private def decimalBits (d : DType) (n : JsonNumber) : Except String Nat := do
  let some f := d.format? | throw "not a floating-point type"
  let mag := Num.ofDecimal f n.mantissa.natAbs (-(n.exponent : Int))
  return if n.mantissa < 0 then mag ||| f.signBit else mag

def scalarKeys : List String := ["u8", "u16", "u32", "u64", "s32", "s64", "f16", "bf16", "f32", "f64"]

/-- How deeply `fields` may nest. -/
def maxDepth : Nat := 16

/-- A value: its leaves at offsets. `{"fields": [...], "size": n}` is `n` zero bytes overwritten
by its fields. -/
def value : Nat → Json → Except String Value
  | 0, _ => throw s!"fields nest deeper than {maxDepth}"
  | fuel + 1, j => do
    if let some b := opt j "ptr" then
      return #[(0, .ptr (← b.getStr?) (← natD j "offset_in" 0))]
    if let some t := opt j "tensorMap" then return #[(0, .tensorMap (← t.getStr?))]
    if let some t := opt j "texture" then return #[(0, .texture (← t.getStr?))]
    if let some h := opt j "hex" then return #[(0, .bytes (← hexBytes (← h.getStr?)))]
    if let some fs := opt j "fields" then
      let mut out : Value := #[]
      if let some n := opt j "size" then out := out.push (0, .bytes (Array.replicate (← n.getNat?) 0))
      for f in ← fs.getArr? do
        let off ← nat f "offset"
        out := out ++ (← value fuel f).map fun (o, l) => (off + o, l)
      return out
    for k in scalarKeys do
      if let some v := opt j k then
        let d ← dtype k
        if d.format?.isSome then return #[(0, .scalar d (← decimalBits d (← v.getNum?)))]
        let i ← v.getInt?
        let w := 8 * d.bytes
        let signed := k.startsWith "s"
        if (signed && (i < -(2 ^ (w - 1) : Int) || i ≥ 2 ^ (w - 1))) ||
           (!signed && (i < 0 || i ≥ 2 ^ w)) then
          throw s!"{i} does not fit {k}"
        return #[(0, .scalar d (d.bits i))]
    throw s!"not a value: {j.compress}"

private def gen (j : Json) : Except String Gen := do
  match ← str j "gen" with
  | "zero" => pure .zero
  | "fill" => pure (.fill (← dtype (← str j "dtype")) (← int j "value") (← intD j "scale" 0))
  | "iota" =>
    let r ← natD j "repeat" 1
    if r == 0 then throw "iota: repeat 0"
    pure (.iota (← dtype (← str j "dtype")) (← intD j "start" 0) (← intD j "step" 1) r)
  | "uniform" =>
    let lo ← int j "lo"
    let hi ← int j "hi"
    if hi < lo then throw s!"uniform: hi {hi} below lo {lo}"
    pure (.uniform (← dtype (← str j "dtype")) lo hi (← intD j "scale" 0) (← nat j "seed").toUInt64)
  | "file" => pure (.file (← str j "path"))
  | g => throw s!"unknown generator \"{g}\""

private def writes (j : Json) (k : String) : Except String (Array (Nat × Value)) := do
  let some ws := opt j k | return #[]
  (← ws.getArr?).mapM fun w => do pure (← nat w "offset", ← value maxDepth w)

private def buffer (j : Json) : Except String Buffer := do
  let name ← str j "name"
  let align ← natD j "align" 256
  if align == 0 || align &&& (align - 1) != 0 then throw s!"{name}: align {align}"
  let init ← match opt j "init" with
    | some g => gen g
    | none => pure .zero
  return { name, bytes := ← nat j "bytes", align, init, writes := ← writes j "writes" }

private def enumOf {α : Type} [Inhabited α] (what : String) (names : List String) (all : List α) (j : Json)
    (k : String) : Except String α := do
  let some v := opt j k | return all[0]!
  let s ← v.getStr?
  match names.idxOf? s with
  | some i => pure all[i]!
  | none => throw s!"{what} \"{s}\": one of {names}"

private def tmDType (s : String) : Except String Async.TmDType :=
  match s with
  | "u8" => pure .u8 | "u16" => pure .u16 | "u32" => pure .u32 | "s32" => pure .s32
  | "u64" => pure .u64 | "s64" => pure .s64 | "f16" => pure .f16 | "f32" => pure .f32
  | "f64" => pure .f64 | "bf16" => pure .bf16 | "f32_ftz" => pure .f32Ftz | "tf32" => pure .tf32
  | "tf32_ftz" => pure .tf32Ftz | "u4x16_align8" => pure .u4x16Align8
  | "u4x16_align16" => pure .u4x16Align16 | "u6x16_align16" => pure .u6x16Align16
  | _ => throw s!"unknown tensor map dtype \"{s}\""

/-- `xs` padded with `d` to `n` entries. -/
private def pad (n : Nat) (d : Nat) (xs : Array Nat) : Vector Nat n :=
  Vector.ofFn fun i => xs.getD i d

private def mapSpec (j : Json) : Except String MapSpec := do
  let name ← str j "name"
  let dims ← nats (← field j "dims")
  let rank := dims.size
  let strides ← nats (← field j "strides")
  -- An im2col map (`cuTensorMapEncodeIm2col`) gives `im2col` in place of `box`.
  let im2col ← match opt j "im2col" with
    | none => pure none
    | some x => do
      let corners (k : String) : Except String (Vector Int 3) := do
        let cs ← (← (← field x k).getArr?).mapM (·.getInt?)
        unless cs.size + 2 == rank do throw s!"{name}: \"{k}\" needs {rank - 2} corners"
        pure (Vector.ofFn fun i => cs.getD i 0)
      pure (some ({ lower := ← corners "lower", upper := ← corners "upper"
                    channelsPerPixel := ← nat x "channelsPerPixel"
                    pixelsPerColumn := ← nat x "pixelsPerColumn" } : Async.Im2col))
  let box ← if im2col.isSome then pure (Array.replicate rank 1) else nats (← field j "box")
  let es ← match opt j "elementStrides" with
    | some e => nats e
    | none => pure (Array.replicate rank 1)
  if strides.size + 1 != rank || box.size != rank || es.size != rank then
    throw s!"{name}: {rank} dims need {rank - 1} strides and {rank} box and element strides"
  let map : TensorMap := {
    dtype := ← tmDType (← str j "dtype"), rank, globalAddress := 0
    globalDim := pad 5 1 dims, globalStrides := pad 4 0 strides, boxDim := pad 5 1 box
    elementStrides := pad 5 1 es
    interleave := ← enumOf "interleave" ["none", "16B", "32B"] Async.Interleave.all j "interleave"
    swizzle := ← enumOf "swizzle" ["none", "32B", "64B", "128B", "128B_atom_32B",
      "128B_atom_32B_flip_8B", "128B_atom_64B"] Async.Swizzle.all j "swizzle"
    l2Promotion := ← enumOf "l2Promotion" ["none", "64B", "128B", "256B"] Async.L2Promotion.all j
      "l2Promotion"
    oobFill := ← enumOf "oobFill" ["none", "nan_request_zero_fma"] Async.OobFill.all j "oobFill"
    im2col }
  return { name, buffer := ← str j "buffer", offset := ← natD j "offset" 0, map }

/-- `{name, buffer, offset?, dims: [w, h?, d?], pitch?, slice?, channels: [bits…], kind:
"unsigned"|"signed"|"float", readMode: "element"|"normalized", normalizedCoords, filter:
"point"|"linear", address: ["clamp"|"border"|"wrap"|"mirror", …], surface, layers?, cubemap?}`:
pitch and slice default to packed rows and slices; `layers` makes a layered object of 1D or 2D
layers, `cubemap` its 2D layers a cubemap's faces (six a cubemap). -/
private def texSpec (j : Json) : Except String TexSpec := do
  let name ← str j "name"
  let dims ← nats (← field j "dims")
  unless dims.size ∈ [1, 2, 3] do throw s!"texture {name}: one to three dims"
  let chans ← nats (← field j "channels")
  let kind ← match ← str j "kind" with
    | "unsigned" => pure NV.Tex.ChanKind.unsigned | "signed" => pure .signed | "float" => pure .float
    | k => throw s!"texture {name}: kind \"{k}\""
  let flag (k : String) : Except String Bool := match opt j k with
    | some b => b.getBool?
    | none => pure false
  let addrMode (s : String) : Except String NV.Tex.Address := match s with
    | "clamp" => pure .clamp | "border" => pure .border | "wrap" => pure .wrap
    | "mirror" => pure .mirror | m => throw s!"texture {name}: address mode \"{m}\""
  let address ← match opt j "address" with
    | some a => (← a.getArr?).mapM fun m => do addrMode (← m.getStr?)
    | none => pure #[.clamp, .clamp, .clamp]
  let width := dims[0]!
  let height := dims.getD 1 1
  let texel := chans.foldl (· + ·) 0 / 8
  let pitch ← natD j "pitch" (width * texel)
  let tex : NV.Tex.Texture := {
    base := 0, dims := dims.size, width, height, depth := dims.getD 2 1, pitch
    slice := ← natD j "slice" (pitch * height), chans, kind
    readNormalized := (← match opt j "readMode" with
      | some m => do pure ((← m.getStr?) == "normalized") | none => pure false)
    normalizedCoords := ← flag "normalizedCoords"
    filter := ← match opt j "filter" with
      | some f => do
        match ← f.getStr? with
        | "point" => pure .point | "linear" => pure .linear
        | m => throw s!"texture {name}: filter \"{m}\""
      | none => pure .point
    address, surface := ← flag "surface", layers := ← natD j "layers" 0, cube := ← flag "cubemap" }
  if let some e := tex.invalid? then throw s!"texture {name}: {e}"
  return { name, buffer := ← str j "buffer", offset := ← natD j "offset" 0, tex }

private def launchStep (j : Json) : Except String LaunchStep := do
  return {
    code := ← str j "code", function := ← str j "function"
    grid := ← dims3 j "grid", block := ← dims3 j "block"
    cluster := ← if (opt j "cluster").isSome then some <$> dims3 j "cluster" else pure none
    dynamicShared := ← natD j "dynamicShared" 0
    params := ← match opt j "params" with
      | some ps => (← ps.getArr?).mapM fun p => do pure (← nat p "ordinal", ← value maxDepth p)
      | none => pure #[] }

private def step (j : Json) : Except String Step := do
  if let some l := opt j "launch" then return .launch (← launchStep l)
  if let some w := opt j "write" then
    return .write (← str w "buffer") (← natD w "offset" 0) (← value maxDepth w)
  throw s!"a step is a launch or a write: {j.compress}"

/-- Read a launch spec. -/
def Spec.ofJson (j : Json) : Except String Spec := do
  let arr (k : String) : Except String (Array Json) :=
    match opt j k with
    | some a => a.getArr?
    | none => pure #[]
  let steps ← if (opt j "steps").isSome then (← arr "steps").mapM step
    else pure #[.launch (← launchStep j)]
  let expect ← (← arr "expect").mapM fun e => do
    pure { buffer := ← str e "buffer", offset := ← natD e "offset" 0, file := ← str e "file" }
  let outputs ← (← arr "outputs").mapM fun o => do pure (← str o "buffer", ← str o "file")
  return { buffers := ← (← arr "buffers").mapM buffer
           tensorMaps := ← (← arr "tensorMaps").mapM mapSpec
           textures := ← (← arr "textures").mapM texSpec, steps, expect, outputs }

def Spec.parse (text : String) : Except String Spec := do
  Spec.ofJson (← Json.parse text)

end Json

/-! ## Building the launch -/

/-- Where buffers start: a 64-bit address of the shape the CUDA driver hands out, so code that
relies on the upper half being nonzero sees a realistic value. -/
def globalBase : UInt64 := 0x7f0000000000

/-- Buffer addresses: each at the next multiple of its alignment. -/
def layout (bs : Array Buffer) : Array (String × UInt64) × Nat := Id.run do
  let mut out := #[]
  let mut off := 0
  for b in bs do
    off := (off + b.align - 1) / b.align * b.align
    out := out.push (b.name, globalBase + off.toUInt64)
    off := off + b.bytes
  return (out, off)

/-- `n` as `k` little-endian bytes. -/
def leBytes (k n : Nat) : Array UInt8 := (Async.TensorMap.le k n).toArray

/-- The resolved tensor maps, by name. -/
def maps (s : Spec) (addrs : Array (String × UInt64)) :
    Except String (Array (String × TensorMap)) :=
  s.tensorMaps.mapM fun t => do
    let some a := addrs.toList.lookup t.buffer | throw s!"tensor map {t.name}: no buffer {t.buffer}"
    let m := { t.map with globalAddress := a + t.offset.toUInt64 }
    if let some e := m.invalid? then throw s!"tensor map {t.name}: {e}"
    return (t.name, m)

def Leaf.encode (addrs : Array (String × UInt64)) (tms : Array (String × TensorMap))
    (texNames : Array String) : Leaf → Except String (Array UInt8)
  | .ptr b off => do
    let some a := addrs.toList.lookup b | throw s!"no buffer {b}"
    pure (leBytes 8 (a.toNat + off))
  | .scalar d bits => pure (leBytes d.bytes bits)
  | .tensorMap n => do
    let some m := tms.toList.lookup n | throw s!"no tensor map {n}"
    pure m.encode.data
  | .texture n => do
    let some i := texNames.idxOf? n | throw s!"no texture {n}"
    pure (leBytes 8 (i + 1))
  | .bytes b => pure b

/-- A value's bytes: as long as its furthest leaf reaches. -/
def encodeValue (addrs : Array (String × UInt64)) (tms : Array (String × TensorMap))
    (texNames : Array String) (v : Value) :
    Except String (Array UInt8) := do
  let mut out : Array UInt8 := #[]
  for (off, l) in v do
    let b ← l.encode addrs tms texNames
    if out.size < off + b.size then out := out ++ Array.replicate (off + b.size - out.size) 0
    for i in [0:b.size] do out := out.set! (off + i) b[i]!
  return out

/-- Element `i`'s draw from a range of `n` values: `splitmix64(seed, i) mod n`, in machine words
(a draw is often past `2^63`, where a `Nat` would be a bignum). -/
def draw (seed : UInt64) (i n : Nat) : Nat :=
  if n < 2 ^ 64 then (splitmix64 seed i.toUInt64 % n.toUInt64).toNat
  else (splitmix64 seed i.toUInt64).toNat % n

#guard draw 7 12345 129 == (splitmix64 7 12345).toNat % 129
#guard (List.range 64).all fun i => draw 3 i 65536 == (splitmix64 3 i.toUInt64).toNat % 65536

/-- One element's bits under a generator. -/
def Gen.element (g : Gen) (i : Nat) : Option (DType × Nat) :=
  match g with
  | .zero => none
  | .fill d k s => some (d, d.bits k s)
  | .iota d start step r => some (d, d.bits (start + (i / r : Nat) * step))
  | .uniform d lo hi s seed => some (d, d.bits (lo + draw seed i (hi - lo + 1).toNat) s)
  | .file _ => none

/-- Reads the files a spec names (paths as the spec gives them). -/
abbrev Files := String → Except String ByteArray

/-- Write a buffer's contents into the global store at `addr`. -/
def fillBuffer (files : Files) (st : Store) (b : Buffer) (addr : UInt64)
    (addrs : Array (String × UInt64)) (tms : Array (String × TensorMap)) (texNames : Array String) :
    Except String Store := do
  let mut st := st
  let off := (addr - st.base).toNat
  match b.init with
  | .zero => pure ()
  | .file path =>
    let data ← files path
    if data.size > b.bytes then throw s!"{b.name}: {path} has {data.size} bytes, the buffer {b.bytes}"
    for j in [0:data.size] do st := st.writeByte (off + j) data[j]!
  | .uniform dt lo hi sc seed =>
    -- Inputs run to hundreds of millions of elements: round each of the range's values once.
    let n := (hi - lo + 1).toNat
    let table := if n ≤ 65536 then (Array.range n).map fun (r : Nat) => dt.bits (lo + r) sc else #[]
    let d := dt.bytes
    for i in [0:b.bytes / d] do
      let r := draw seed i n
      let bits := (table[r]?.getD (dt.bits (lo + r) sc)).toUInt64
      for j in [0:d] do st := st.writeByte (off + i * d + j) (bits >>> (8 * j).toUInt64).toUInt8
  | g =>
    let d := (g.element 0).elim 1 (·.1.bytes)
    for i in [0:b.bytes / d] do
      if let some (_, bits) := g.element i then
        let bits := bits.toUInt64
        for j in [0:d] do st := st.writeByte (off + i * d + j) (bits >>> (8 * j).toUInt64).toUInt8
  for (wo, v) in b.writes do
    let bytes ← encodeValue addrs tms texNames v
    if wo + bytes.size > b.bytes then throw s!"{b.name}: a write at {wo} runs past its end"
    for j in [0:bytes.size] do st := st.writeByte (off + wo + j) bytes[j]!
  return st

/-- Global memory before the first step, with the buffers' addresses and the tensor maps. -/
structure Memory where
  global : Store
  addrs : Array (String × UInt64)
  maps : Array (String × TensorMap)
  /-- The texture and surface objects, in the spec's order (handle `i + 1` is entry `i`). -/
  textures : Array (String × NV.Tex.Texture) := #[]

def Memory.texNames (m : Memory) : Array String := m.textures.map (·.1)

def Spec.memory (s : Spec) (files : Files) : Except String Memory := do
  let (addrs, total) := layout s.buffers
  let tms ← maps s addrs
  let texs ← s.textures.mapM fun t => do
    let some a := addrs.toList.lookup t.buffer | throw s!"texture {t.name}: no buffer {t.buffer}"
    pure (t.name, { t.tex with base := a + t.offset.toUInt64 })
  let mut global := Store.empty globalBase total
  for (b, (_, a)) in s.buffers.zip addrs do
    global ← fillBuffer files global b a addrs tms (texs.map (·.1))
  return { global, addrs, maps := tms, textures := texs }

/-- A host write into the store. -/
def Memory.write (m : Memory) (global : Store) (buffer : String) (offset : Nat) (v : Value) :
    Except String Store := do
  let some a := m.addrs.toList.lookup buffer | throw s!"write: no buffer {buffer}"
  let bytes ← encodeValue m.addrs m.maps m.texNames v
  let off := (a - global.base).toNat + offset
  let mut g := global
  for j in [0:bytes.size] do g := g.writeByte (off + j) bytes[j]!
  return g

/-- One launch over the current global memory. Every parameter the kernel declares
(`Kernel.params`, from its ELF) must be given, at its size. -/
def LaunchStep.launch (l : LaunchStep) (m : Memory) (k : SASS.Kernel) (global : Store) :
    Except String Core.Launch := do
  let mut block := Array.replicate k.paramBytes (0 : UInt8)
  for h : ord in [0:k.params.size] do
    let (off, size) := k.params[ord]
    let some (_, v) := l.params.find? (·.1 == ord) | throw s!"parameter {ord} is not given"
    let bytes ← encodeValue m.addrs m.maps m.texNames v
    if bytes.size != size then throw s!"parameter {ord}: {bytes.size} bytes given, the ELF says {size}"
    if off + size > block.size then block := block ++ Array.replicate (off + size - block.size) 0
    for i in [0:size] do block := block.set! (off + i) bytes[i]!
  for (ord, _) in l.params do
    unless ord < k.params.size do throw s!"parameter {ord}: the kernel has {k.params.size}"
  let cluster := l.cluster.getD (k.cluster.getD (1, 1, 1))
  return { kernel := k, grid := l.grid, block := l.block, cluster, params := ⟨block⟩, global,
           dynShared := l.dynamicShared, textures := m.textures.map (·.2) }

/-- Run every step in order: `kernel` finds a launch's kernel (by listing and name), `run` runs
a launch from a state `σ` that the launches thread (a machine whose caches persist), starting at
`init`, in a monad `n` (the simulator runs SMs on threads in `IO`). The final memory and each
launch's result. -/
def Spec.runM {n : Type → Type} [Monad n] {σ ρ : Type} (s : Spec) (files : Files)
    (kernel : String → String → Except String SASS.Kernel) (init : σ)
    (run : σ → Core.Launch → n (Except String (σ × Store × ρ))) (prepare : Memory → Memory := id) :
    n (Except String (Memory × Store × Array ρ)) := do
  match s.memory files with
  | .error e => return .error e
  | .ok m =>
    -- The memory as the caller wants it before the first step (`Launch.unwritten`'s sentinel).
    let m := prepare m
    let mut global := m.global
    let mut state := init
    let mut out := #[]
    for st in s.steps do
      match st with
      | .write b off v =>
        match m.write global b off v with
        | .ok g => global := g
        | .error e => return .error e
      | .launch l =>
        match kernel l.code l.function >>= fun k => l.launch m k global with
        | .error e => return .error e
        | .ok launch =>
          match ← run state launch with
          | .ok (st', g, r) =>
            state := st'
            global := g
            out := out.push r
          | .error e => return .error e
    return .ok (m, global, out)

/-- `Spec.runM` without effects. -/
def Spec.run {σ ρ : Type} (s : Spec) (files : Files)
    (kernel : String → String → Except String SASS.Kernel) (init : σ)
    (run : σ → Core.Launch → Except String (σ × Store × ρ)) :
    Except String (Memory × Store × Array ρ) :=
  Id.run (s.runM files kernel init (fun st l => pure (run st l)))

/-- A buffer's bytes in the final memory, pushed one by one into a byte array of the buffer's size (no
array of boxed indices or bytes on the way: a 200 MB output stays 200 MB). -/
def Memory.bytes (m : Memory) (global : Store) (s : Spec) (buffer : String) : Except String ByteArray := do
  let some a := m.addrs.toList.lookup buffer | throw s!"no buffer {buffer}"
  let some b := s.buffers.find? (·.name == buffer) | throw s!"no buffer {buffer}"
  let off := (a - global.base).toNat
  let mut out := ByteArray.emptyWithCapacity b.bytes
  for i in [0:b.bytes] do
    out := out.push (global.readByte (off + i))
  return out

/-- The expectations the final memory fails, each with its first differing byte. -/
def Spec.mismatches (s : Spec) (files : Files) (m : Memory) (global : Store) :
    Except String (Array String) := do
  let mut out := #[]
  for e in s.expect do
    let some a := m.addrs.toList.lookup e.buffer | throw s!"expect: no buffer {e.buffer}"
    let want ← files e.file
    let base := (a - global.base).toNat + e.offset
    -- The first differing byte, by index (no list of every index of a large file).
    let mut first : Option Nat := none
    for i in [0:want.size] do
      if global.readByte (base + i) != want[i]! then
        first := some i
        break
    if let some i := first then
      out := out.push s!"{e.buffer}[{e.offset + i}] = {global.readByte (base + i)}, {e.file} has {want[i]!}"
  return out

end NV.Launch
