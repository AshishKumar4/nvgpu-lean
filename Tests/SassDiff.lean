import Tests.Kernels
import NV.PTX.Interp
import NV.PTX.Syntax

/-!
`lake exe sass-diff [kernel…]`: the PTX↔SASS differential test. Each kernel of
`tests/diff/` runs twice on the same inputs: in the PTX interpreter (`NV.PTX.interp`, the PTX
ISA's semantics) and, compiled by the pinned `ptxas`, in the SASS simulator on the H200
parameters. Every output word must agree. Inputs mix edge values (signed zeros, subnormals,
infinities, NaNs, the extremes of each format) with pseudo-random words and moderate values.

A disagreement is either a SASS semantics defect, a PTX semantics defect, or a compiled form the
model does not capture; the report names the kernel, thread and output.
-/

open NV NV.SASS NV.Core Tests

/-- A deterministic word sequence (a 64-bit linear congruential generator, high word). -/
def words (seed n : Nat) : Array UInt32 := Id.run do
  let mut x := seed
  let mut out := #[]
  for _ in [0:n] do
    x := (x * 6364136223846793005 + 1442695040888963407) % 2 ^ 64
    out := out.push (x / 2 ^ 32).toUInt32
  return out

/-- binary32 edge values. -/
def edges32 : Array UInt32 := #[0, 0x80000000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000,
  0x7fc00000, 0x7f800001, 0x00000001, 0x807fffff, 0x00800000, 0x7f7fffff, 0x3f000000, 0x40000000,
  0x40400000, 0x4b000000, 0xcf000000, 0x4f800000, 0x3eaaaaab, 0x0d800000, 0x72800000, 0x33800000]

/-- binary16 pairs of edge values. -/
def edges16 : Array UInt32 := #[0, 0x80008000, 0x3c00bc00, 0x7c00fc00, 0x7e000001, 0x03ff8001,
  0x7bff0400, 0x38004000, 0x3f804080, 0x7f80ff80]

/-- Binary32 values of moderate magnitude (exponents 2^-4 to 2^3, either sign): finite texels. -/
def smallFloats (n : Nat) : Array UInt32 :=
  (words 5 n).map fun (w : UInt32) =>
    (w &&& (0x807fffff : UInt32)) ||| (((0x7b : UInt32) + (w >>> 24) % 8) <<< 23)

/-- Input words: edge values first, then moderate values (exponents near 1) and raw words. -/
def inputs (edges : Array UInt32) (n : Nat) : Array UInt32 :=
  let raw := words 12345 n
  let moderate := (words 777 n).map fun (w : UInt32) =>
    let e : UInt32 := (0x7c : UInt32) + (w >>> 24) % 8
    (w &&& (0x807fffff : UInt32)) ||| (e <<< 23)
  (List.range n).toArray.map fun k =>
    if k < edges.size then edges[k]! else if k % 2 == 0 then moderate[k]! else raw[k]!

structure Case where
  file : String
  kernel : String
  grid : Nat := 1
  /-- CTAs per cluster (along x). -/
  cluster : Nat := 1
  /-- Texture and surface objects over buffers (`Tests.run`); their handles follow `extra`. -/
  textures : List (Nat × NV.Tex.Texture) := []
  threads : Nat
  /-- Buffers: initial words; each buffer's address is a 64-bit parameter, in order. -/
  bufs : List (Array UInt32)
  /-- Parameters after the buffer addresses, as bytes. -/
  extra : List (List UInt8) := []
  /-- Output words per thread of the first buffer, for reports; 0 to report by word. -/
  per : Nat := 0
  /-- Outputs (indices modulo `per`) whose disagreement is understood and assigned, with the
  reason. -/
  known : List (Nat × String) := []
  /-- Opcodes (with modifiers, as `Instr.full` prints them) ptxas must emit for the kernel to
  test what it is meant to, on the targets of `formTargets`. -/
  forms : List String := []
  formTargets : List String := ["sm_90a"]
  /-- Targets the case does not run on, with the reason. -/
  skipOn : List (String × String) := []
  /-- Pairs of outputs (indices modulo `per`) each thread must have equal in both runs: one value
  reached by two routes the kernel computes side by side. -/
  same : List (Nat × Nat) := []

def f32Ops : Nat := 49

def cases : List Case := [
  { file := "tests/diff/f32.ptx", kernel := "f32ops", threads := 64, per := f32Ops,
    bufs := [Array.replicate (f32Ops * 64) 0, inputs edges32 (3 * 64)] },
  { file := "tests/diff/packed.ptx", kernel := "packed", threads := 64, per := 8,
    bufs := [Array.replicate (8 * 64) 0, inputs edges16 (3 * 64)] },
  { file := "tests/diff/vmnmx.ptx", kernel := "vmnmx", threads := 64, per := 10,
    bufs := [Array.replicate (10 * 64) 0, words 23 128],
    forms := ["VIMNMX.U16x2", "VIMNMX.S16x2", "VIMNMX.S16x2.RELU"],
    formTargets := ["sm_90a", "sm_100a", "sm_120a"],
    skipOn := [("sm_80", "min.relu needs sm_90")] },
  { file := "tests/diff/paramaddr.ptx", kernel := "paramaddr", threads := 64, per := 2,
    bufs := [Array.replicate (2 * 64) 0, words 29 (4 * 64)], forms := ["CALL.REL.NOINC", "STL.64"],
    formTargets := ["sm_80", "sm_90a", "sm_100a", "sm_120a"] },
  { file := "tests/diff/ldcbyte.ptx", kernel := "ldcbyte", threads := 64, per := 4,
    bufs := [Array.replicate (4 * 64) 0],
    extra := [(List.range 64).map fun i => (i * 37 + 0x5b).toUInt8],
    forms := ["LDC.U8", "LDC.S8", "LDC.U16", "LDC.S16"], formTargets := ["sm_90a", "sm_80"] },
  { file := "tests/diff/tf32cvt.ptx", kernel := "tf32cvt", threads := 64, per := 2,
    bufs := [Array.replicate (2 * 64) 0, inputs edges32 64],
    forms := ["F2FP.TF32.F32.PACK_B", "F2FP.TF32.F32.PACK_B.RZ"], formTargets := ["sm_90a"],
    skipOn := [("sm_80", "cvt.rn.tf32.f32 needs sm_90")] },
  { file := "tests/diff/fmulscale.ptx", kernel := "fmulscale", threads := 64, per := 6,
    bufs := [Array.replicate (6 * 64) 0, inputs edges32 (7 * 64)],
    forms := ["FMUL.D2", "FMUL.D4", "FMUL.D8", "FMUL.M2", "FMUL.M4", "FMUL.M8"],
    formTargets := ["sm_90a", "sm_80"] },
  { file := "tests/diff/f64.ptx", kernel := "f64ops", threads := 64, per := 18,
    bufs := [Array.replicate (18 * 64) 0, inputs #[0, 0x80000000, 0, 0x3ff00000, 0, 0x7ff00000, 1,
      0x7ff00000, 1, 0, 0xffffffff, 0x7fefffff, 0, 0x00100000, 0x55555555, 0x40091eb8, 0, 0xbff80000]
      (6 * 64)],
    forms := ["DSETP.MIN", "DSETP.MAX", "MUFU.RCP64H", "MUFU.RSQ64H", "DADD.RP"],
    formTargets := ["sm_80", "sm_90a", "sm_120a"] },
  -- Binary64 division and reciprocal over 16 × 16 edge operand pairs (subnormal quotients,
  -- near-cancelling denominators, the slow path).
  { file := "tests/diff/f64div.ptx", kernel := "f64div", threads := 256, per := 4,
    bufs := [Array.replicate (4 * 256) 0, #[0x0, 0x0, 0x0, 0x80000000, 0x1, 0x0, 0xffffffff, 0xfffff, 0x0, 0x1, 0x0, 0x100000, 0x0, 0x1a700000, 0x0, 0x3ff00000, 0x1, 0x3ff00000, 0x0, 0x40080000, 0x0, 0x65700000, 0xffffffff, 0x7fefffff, 0x0, 0x7fe00000, 0x0, 0x7ff00000, 0x0, 0x7ff80000, 0x0, 0x81700000, 0x0, 0x0, 0x0, 0x80000000, 0x1, 0x0, 0xffffffff, 0xfffff, 0x0, 0x1, 0x0, 0x100000, 0x0, 0x1a700000, 0x0, 0x3ff00000, 0xffffffff, 0x3fefffff, 0x0, 0xc0080000, 0x0, 0x65700000, 0xffffffff, 0x7fefffff, 0x0, 0x80000, 0x0, 0x7ff00000, 0x0, 0x7ff80000, 0x0, 0x3cb00000]],
    formTargets := ["sm_80", "sm_90a", "sm_120a"] },
  -- The same, relocatable: the slow path a function of its own, called by absolute address.
  { file := "tests/diff/f64divrel.ptx", kernel := "f64divrel", threads := 256, per := 4,
    bufs := [Array.replicate (4 * 256) 0, #[0x0, 0x0, 0x0, 0x80000000, 0x1, 0x0, 0xffffffff, 0xfffff, 0x0, 0x1, 0x0, 0x100000, 0x0, 0x1a700000, 0x0, 0x3ff00000, 0x1, 0x3ff00000, 0x0, 0x40080000, 0x0, 0x65700000, 0xffffffff, 0x7fefffff, 0x0, 0x7fe00000, 0x0, 0x7ff00000, 0x0, 0x7ff80000, 0x0, 0x81700000, 0x0, 0x0, 0x0, 0x80000000, 0x1, 0x0, 0xffffffff, 0xfffff, 0x0, 0x1, 0x0, 0x100000, 0x0, 0x1a700000, 0x0, 0x3ff00000, 0xffffffff, 0x3fefffff, 0x0, 0xc0080000, 0x0, 0x65700000, 0xffffffff, 0x7fefffff, 0x0, 0x80000, 0x0, 0x7ff00000, 0x0, 0x7ff80000, 0x0, 0x3cb00000]],
    formTargets := ["sm_80", "sm_90a", "sm_120a"] },
  -- Binary32 to 16-bit integers, widened back: edge values (saturation, NaN, signed zeros) and
  -- values of either sign in range.
  { file := "tests/diff/f2i16.ptx", kernel := "f2i16", threads := 256, per := 4,
    bufs := [Array.replicate (4 * 256) 0, (inputs edges32 256).map fun w =>
      if w % 3 == 0 then (w &&& 0x80ffffff) ||| 0x46000000 else w] },
  -- mma.sync fragments: moderate values (exact products are not the point; the datapath's
  -- rounding is) with edge values in some lanes.
  { file := "tests/diff/mma.ptx", kernel := "mmas", threads := 32, per := 16,
    bufs := [Array.replicate (16 * 32) 0, (inputs edges16 (16 * 32)).map fun w =>
      -- binary16 pairs of moderate exponent in most words
      if w &&& 0x100 == 0 then (w &&& 0x83ff83ff) ||| 0x38003800 else w] },
  -- WMMA: the fragment layouts (NV.PTX.Sem.Matrix.wfrag).
  { file := "tests/diff/wmma.ptx", kernel := "wmma0", threads := 32,
    bufs := [Array.replicate 256 0, inputs edges32 512],
    forms := ["HMMA.16816.F32"], formTargets := ["sm_80", "sm_90a"] },
  { file := "tests/diff/wmma.ptx", kernel := "wmma1", threads := 32,
    bufs := [Array.replicate 128 0, inputs edges32 384],
    forms := ["HMMA.16816.F16"], formTargets := ["sm_80", "sm_90a"] },
  { file := "tests/diff/wmma.ptx", kernel := "wmma2", threads := 32,
    bufs := [Array.replicate 256 0, inputs edges32 576],
    forms := ["HMMA.16816.F32"], formTargets := ["sm_80", "sm_90a"] },
  { file := "tests/diff/wmma.ptx", kernel := "wmma3", threads := 32,
    bufs := [Array.replicate 256 0, inputs edges32 576],
    forms := ["HMMA.16816.F32"], formTargets := ["sm_80", "sm_90a"] },
  { file := "tests/diff/wmma.ptx", kernel := "wmma4", threads := 32,
    bufs := [Array.replicate 256 0, inputs edges32 512],
    forms := ["HMMA.16816.F32.BF16"], formTargets := ["sm_80", "sm_90a"] },
  { file := "tests/diff/wmma.ptx", kernel := "wmma5", threads := 32,
    bufs := [Array.replicate 256 0, inputs edges32 512],
    forms := ["HMMA.1684.F32.TF32"], formTargets := ["sm_80", "sm_90a"] },
  { file := "tests/diff/wmma.ptx", kernel := "wmma6", threads := 32,
    bufs := [Array.replicate 256 0, inputs edges32 384],
    forms := ["IMMA.16816.S8.S8"], formTargets := ["sm_80", "sm_90a"] },
  { file := "tests/diff/wmma.ptx", kernel := "wmma7", threads := 32,
    bufs := [Array.replicate 256 0, inputs edges32 416],
    forms := ["IMMA.16816.U8.U8"], formTargets := ["sm_80", "sm_90a"] },
  { file := "tests/diff/wmma.ptx", kernel := "wmma9", threads := 32,
    bufs := [Array.replicate 256 0, inputs edges32 192],
    forms := ["IMMA.8832.S4.S4", "IMMA.8832.U4.U4.SAT", "BMMA.88128.XOR.POPC", "BMMA.88128.AND.POPC"],
    formTargets := ["sm_80"] },
  { file := "tests/diff/wmma.ptx", kernel := "wmma8", threads := 32,
    bufs := [Array.replicate 128 0, inputs edges32 256],
    forms := ["DMMA"], formTargets := ["sm_80", "sm_90a"] },
  -- Asynchronous stores and reductions into the peer CTA's shared memory.
  { file := "tests/diff/stasync.ptx", kernel := "stasync", grid := 2, cluster := 2, threads := 32,
    per := 8, bufs := [Array.replicate (2 * 32 * 8) 0, inputs edges32 (2 * 32 * 2)],
    forms := ["STAS", "STAS.64", "REDAS.ADD", "REDAS.MIN.S32", "REDAS.INC", "REDAS.XOR"],
    skipOn := [("sm_80", "no st.async before sm_90")] },
  -- Buffer 1 sits after buffer 0's 2 KB (`Tests.run`'s layout): the map's global address.
  -- Buffer 1 (after buffer 0's 512 bytes): the 4D tensor, then at 1 KB the 3D one.
  { file := "tests/diff/im2col.ptx", kernel := "im2col", threads := 32, per := 4,
    bufs := [Array.replicate 128 0, words 29 416],
    extra := [(NV.Async.TensorMap.encode
        { dtype := .f32, rank := 4, globalAddress := Tests.base + 512,
          globalDim := #v[4, 6, 5, 2, 1], globalStrides := #v[16, 96, 480, 0],
          boxDim := #v[1, 1, 1, 1, 1], elementStrides := #v[1, 1, 1, 1, 1],
          im2col := some { lower := #v[-1, -1, 0], upper := #v[-1, -1, 0],
                           channelsPerPixel := 4, pixelsPerColumn := 16 } }).data.toList,
      (NV.Async.TensorMap.encode
        { dtype := .f32, rank := 3, globalAddress := Tests.base + 512 + 1024,
          globalDim := #v[4, 20, 2, 1, 1], globalStrides := #v[16, 320, 0, 0],
          boxDim := #v[1, 1, 1, 1, 1], elementStrides := #v[1, 2, 1, 1, 1],
          im2col := some { lower := #v[-2, 0, 0], upper := #v[1, 0, 0],
                           channelsPerPixel := 4, pixelsPerColumn := 16 } }).data.toList],
    forms := ["UTMALDG.4D.IM2COL", "UTMALDG.3D.IM2COL"], formTargets := ["sm_90a"],
    skipOn := [("sm_80", "no TMA before sm_90")] },
  -- The tensor is buffer 0 (1 KB) at the base; buffer 1 the box's values.
  { file := "tests/diff/im2col.ptx", kernel := "im2colmc", grid := 2, cluster := 2, threads := 32, per := 2,
    bufs := [Array.replicate 128 0, words 29 416],
    extra := [(NV.Async.TensorMap.encode
        { dtype := .f32, rank := 4, globalAddress := Tests.base + 512,
          globalDim := #v[4, 6, 5, 2, 1], globalStrides := #v[16, 96, 480, 0],
          boxDim := #v[1, 1, 1, 1, 1], elementStrides := #v[1, 1, 1, 1, 1],
          im2col := some { lower := #v[-1, -1, 0], upper := #v[-1, -1, 0],
                           channelsPerPixel := 4, pixelsPerColumn := 16 } }).data.toList,
      (NV.Async.TensorMap.encode
        { dtype := .f32, rank := 3, globalAddress := Tests.base + 512 + 1024,
          globalDim := #v[4, 20, 2, 1, 1], globalStrides := #v[16, 320, 0, 0],
          boxDim := #v[1, 1, 1, 1, 1], elementStrides := #v[1, 2, 1, 1, 1],
          im2col := some { lower := #v[-2, 0, 0], upper := #v[1, 0, 0],
                           channelsPerPixel := 4, pixelsPerColumn := 16 } }).data.toList],
    forms := ["UTMALDG.4D.IM2COL.MULTICAST"], formTargets := ["sm_90a"],
    skipOn := [("sm_80", "no TMA before sm_90"), ("sm_120a", "no multicast TMA load in sm_120a SASS")] },
  -- The tensor is buffer 0 (1 KB) at the base; buffer 1 the box's values.
  { file := "tests/diff/tmared.ptx", kernel := "tmared", threads := 32, per := 8,
    bufs := [smallFloats 256, smallFloats 128],
    extra := [(NV.Async.TensorMap.encode
      { dtype := .f32, rank := 2, globalAddress := Tests.base, globalDim := #v[16, 16, 1, 1, 1],
        globalStrides := #v[64, 0, 0, 0], boxDim := #v[16, 8, 1, 1, 1],
        elementStrides := #v[1, 1, 1, 1, 1] }).data.toList],
    forms := ["UTMAREDG.2D.ADD"], formTargets := ["sm_90a"],
    skipOn := [("sm_80", "no TMA before sm_90")] },
  -- A map the kernel builds on the device (tensormap.replace of a copy in global memory), then
  -- loads through.
  { file := "tests/diff/tmreplace.ptx", kernel := "tmreplace", threads := 32, per := 5,
    bufs := [Array.replicate 160 0, smallFloats 512, Array.replicate 32 0],
    extra := [(NV.Async.TensorMap.encode
      { dtype := .f32, rank := 2, globalAddress := Tests.base, globalDim := #v[16, 16, 1, 1, 1],
        globalStrides := #v[64, 0, 0, 0], boxDim := #v[16, 8, 1, 1, 1],
        elementStrides := #v[1, 1, 1, 1, 1] }).data.toList],
    forms := ["UTMALDG.2D"], formTargets := ["sm_90a", "sm_100a"],
    skipOn := [("sm_80", "no TMA before sm_90")] },
  { file := "tests/diff/gather4.ptx", kernel := "gather4", threads := 32,
    bufs := [Array.replicate 128 0, words 13 256],
    extra := [(NV.Async.TensorMap.encode
      { dtype := .f32, rank := 2, globalAddress := Tests.base + 512, globalDim := #v[16, 16, 1, 1, 1],
        globalStrides := #v[64, 0, 0, 0], boxDim := #v[16, 1, 1, 1, 1],
        elementStrides := #v[1, 1, 1, 1, 1], swizzle := .b128 }).data.toList],
    forms := ["UTMALDG.2D.GATHER4"], formTargets := ["sm_100a"],
    skipOn := [("sm_80", "no TMA before sm_90"), ("sm_90a", ".tile::gather4 requires sm_100a"),
               ("sm_120a", ".tile::gather4 requires sm_100a")] },
  { file := "tests/diff/gather4.ptx", kernel := "scatter4", threads := 32,
    bufs := [words 17 256, words 13 128],
    extra := [(NV.Async.TensorMap.encode
      { dtype := .f32, rank := 2, globalAddress := Tests.base, globalDim := #v[16, 16, 1, 1, 1],
        globalStrides := #v[64, 0, 0, 0], boxDim := #v[16, 1, 1, 1, 1],
        elementStrides := #v[1, 1, 1, 1, 1], swizzle := .b128 }).data.toList],
    forms := ["UTMASTG.2D.SCATTER4"], formTargets := ["sm_100a"],
    skipOn := [("sm_80", "no TMA before sm_90"), ("sm_90a", ".tile::scatter4 requires sm_100a"),
               ("sm_120a", ".tile::scatter4 requires sm_100a")] },
  { file := "tests/diff/tmamc.ptx", kernel := "tmamc", grid := 2, cluster := 2, threads := 32,
    bufs := [Array.replicate 512 0, words 11 256],
    extra := [(NV.Async.TensorMap.encode
      { dtype := .f32, rank := 2, globalAddress := Tests.base + 2048, globalDim := #v[16, 16, 1, 1, 1],
        globalStrides := #v[64, 0, 0, 0], boxDim := #v[16, 8, 1, 1, 1],
        elementStrides := #v[1, 1, 1, 1, 1] }).data.toList],
    forms := ["UTMALDG.2D.MULTICAST"], formTargets := ["sm_90a"],
    skipOn := [("sm_80", "no TMA before sm_90"), ("sm_120a",
      "no multicast TMA load in sm_120a SASS")] },
  { file := "tests/diff/tmamc.ptx", kernel := "tmaplain", grid := 2, cluster := 2, threads := 32,
    bufs := [Array.replicate 512 0, words 11 256],
    extra := [(NV.Async.TensorMap.encode
      { dtype := .f32, rank := 2, globalAddress := Tests.base + 2048, globalDim := #v[16, 16, 1, 1, 1],
        globalStrides := #v[64, 0, 0, 0], boxDim := #v[16, 8, 1, 1, 1],
        elementStrides := #v[1, 1, 1, 1, 1] }).data.toList],
    skipOn := [("sm_80", "no TMA before sm_90")] },
  { file := "tests/diff/cparrive.ptx", kernel := "cparrive", threads := 32,
    bufs := [Array.replicate 64 0, words 17 64],
    forms := ["ARRIVES.LDGSTSBAR.64.TRANSCNT", "ARRIVES.LDGSTSBAR.64.ARVCNT"],
    formTargets := ["sm_90a", "sm_120a"], skipOn := [("sm_80", "no mbarrier.try_wait before sm_90")] },
  { file := "tests/diff/rarrive.ptx", kernel := "bulkcp", grid := 2, cluster := 2, threads := 32,
    bufs := [Array.replicate 64 0, words 19 128], forms := ["UBLKCP.S.S"],
    skipOn := [("sm_80", "no clusters before sm_90")] },
  { file := "tests/diff/rarrive.ptx", kernel := "bulkred", grid := 2, cluster := 2, threads := 32,
    bufs := [Array.replicate 64 0, words 19 128], forms := ["UBLKRED.S.S.ADD"],
    skipOn := [("sm_80", "no clusters before sm_90")] },
  { file := "tests/diff/rarrive.ptx", kernel := "rarrive", grid := 2, cluster := 2, threads := 32,
    bufs := [Array.replicate 64 0, words 7 64],
    skipOn := [("sm_80", "no clusters before sm_90")] },
  -- One arrival whose lanes reach the CTA's own barrier and the peer's.
  { file := "tests/diff/rarrive.ptx", kernel := "mixedarrive", grid := 2, cluster := 2, threads := 32,
    bufs := [Array.replicate 64 0, words 7 64], forms := ["SYNCS.ARRIVE.TRANS64.RED.A1T0"],
    formTargets := ["sm_90a", "sm_100a"], skipOn := [("sm_80", "no clusters before sm_90")] },
  -- Tensor memory and tcgen05.mma (sm_100a): A and B from binary16 pairs, D and the fragments of
  -- every load and store shape.
  -- A predicated-off named-barrier arrival does not count.
  { file := "tests/diff/namedbar.ptx", kernel := "predbar", threads := 96, per := 1,
    bufs := [Array.replicate 96 0, #[1, 2, 3, 4]], forms := ["BAR.ARV"],
    formTargets := ["sm_80", "sm_90a", "sm_120a"] },
  -- 256-bit global loads and stores (sm_100a): both quads of LDG/STG.E.ENL2.256.
  { file := "tests/diff/v8.ptx", kernel := "v8", threads := 64, per := 16,
    bufs := [Array.replicate (64 * 16) 0, inputs edges32 (64 * 16)],
    forms := ["LDG.E.ENL2.256.CONSTANT", "LDG.E.ENL2.256", "STG.E.ENL2.256"], formTargets := ["sm_100a"] },
  -- Masked bulk stores (sm_100a): only the bytes each mask selects, as one 16-byte pattern.
  { file := "tests/diff/cpmask.ptx", kernel := "cpmask", threads := 64,
    bufs := [Array.replicate 80 0xdeadbeef, inputs edges32 64],
    forms := ["UBLKCP.G.S.DST_G_BYTE_MASK"], formTargets := ["sm_100a"] },
  -- The cluster's special registers, clusters of 3 CTAs (sm_100a: %clusterid through a table of
  -- reciprocals in constant bank 0).
  { file := "tests/diff/clusterid.ptx", kernel := "clusterid", grid := 12, cluster := 3, threads := 32,
    per := 16, bufs := [Array.replicate (12 * 16) 0], skipOn := [("sm_80", "no clusters before sm_90")] },
  -- Packed binary32 pairs (sm_100a): FADD2/FMUL2/FFMA2 under each rounding and flushing, with
  -- swapped (`.LO_HI`), single (`UR.F32`), negated and immediate sources; `s` is 1.1.
  { file := "tests/diff/f32x2.ptx", kernel := "f32x2", threads := 64, per := 28,
    bufs := [Array.replicate (64 * 28) 0, inputs edges32 (64 * 6)], extra := [[0xcd, 0xcc, 0x8c, 0x3f]],
    forms := ["FADD2", "FMUL2", "FFMA2", "FADD2.RZ", "FMUL2.RM", "FFMA2.FTZ.RP", "FADD2.FTZ"],
    formTargets := ["sm_100a"] },
  -- Named barriers by register, one per pair of warps: kept apart, each reader sees its writer.
  { file := "tests/diff/namedbar.ptx", kernel := "namedbar", threads := 128, per := 1,
    bufs := [Array.replicate 128 0, #[1, 2, 3, 4]], forms := ["BAR.SYNC.DEFER_BLOCKING", "BAR.ARV"],
    formTargets := ["sm_80", "sm_90a", "sm_120a"] },
  { file := "tests/diff/tcgen05.ptx", kernel := "tcgen05", threads := 128, per := 72,
    bufs := [Array.replicate (72 * 128) 0, inputs edges16 1536],
    forms := ["UTCATOMSWS.FIND_AND_SET.ALIGN", "UTCHMMA", "UTCBAR", "LDTM.x64", "STTM.16dp256bit",
      "LDTM.16dp64bit.x4", "STTM.x2.EXPAND16BIT", "LDTM.x2.PACK16BIT", "LDTM.16dp32bit_t0_t15.x2",
      "LDTM.16dp32bit_t16_t31.x2"],
    formTargets := ["sm_100a"],
    skipOn := [("sm_80", "tcgen05 needs sm_100a"), ("sm_90a", "tcgen05 needs sm_100a"),
      ("sm_120a", "tcgen05 needs sm_100a")] },
  -- tcgen05.cp in every shape, decompressing and from a swizzled source, and tcgen05.shift.
  -- One CTA, then a cluster of two (BROADCAST's response and complete-tx reach both CTAs).
  { file := "tests/diff/clc.ptx", kernel := "clc", grid := 2, threads := 32, per := 2,
    bufs := [Array.replicate 4 7],
    forms := ["UGETNEXTWORKID.SELFCAST", "UGETNEXTWORKID.BROADCAST"], formTargets := ["sm_100a"] },
  { file := "tests/diff/clc.ptx", kernel := "clc", grid := 2, cluster := 2, threads := 32, per := 2,
    bufs := [Array.replicate 4 7] },
  { file := "tests/diff/tc5cp.ptx", kernel := "tc5cp", threads := 128, per := 64,
    bufs := [Array.replicate (64 * 128) 0, words 11 2048],
    forms := ["UTCCP.T.S", "UTCCP.T.S.4dp256bit", "UTCCP.T.S.128dp128bit",
      "UTCCP.T.S.2x64dp128bit_lw02_lw13", "UTCCP.T.S.2x64dp128bit_lw01_lw23", "UTCCP.T.S.4x32dp128bit",
      "UTCCP.T.S.U6x16P32", "UTCCP.T.S.128dp128bit.U4x16P64", "UTCSHIFT.DOWN"],
    formTargets := ["sm_100a"],
    skipOn := [("sm_80", "tcgen05 needs sm_100a"), ("sm_90a", "tcgen05 needs sm_100a"),
      ("sm_120a", "tcgen05 needs sm_100a")] },
  -- A CTA pair's tcgen05 (.cta_group::2): allocation in both CTAs, the pair's copy and M = 256 MMA,
  -- the commit multicast to both mbarriers.
  { file := "tests/diff/tc5pair.ptx", kernel := "tc5pair", grid := 2, cluster := 2, threads := 128,
    per := 68, bufs := [Array.replicate (68 * 256) 0, inputs edges16 2560],
    forms := ["UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN", "UTCCP.T.S.2CTA.128dp128bit", "UTCHMMA.2CTA",
      "UTCBAR.2CTA.MULTICAST"],
    formTargets := ["sm_100a"],
    skipOn := [("sm_80", "tcgen05 needs sm_100a"), ("sm_90a", "tcgen05 needs sm_100a"),
      ("sm_120a", "tcgen05 needs sm_100a")] },
  -- A CTA pair's M = 256 MMA with A in tensor memory (UTCHMMA.2CTA, A from tmem), equal to the same
  -- product with A in shared memory.
  { file := "tests/diff/tc5pairts.ptx", kernel := "tc5pairts", grid := 2, cluster := 2, threads := 128,
    per := 128, bufs := [Array.replicate (128 * 256) 0, inputs edges16 2560],
    same := (List.range 64).map fun j => (j, j + 64),
    forms := ["UTCCP.T.S.2CTA.128dp128bit", "UTCHMMA.2CTA"],
    formTargets := ["sm_100a"],
    skipOn := [("sm_80", "tcgen05 needs sm_100a"), ("sm_90a", "tcgen05 needs sm_100a"),
      ("sm_120a", "tcgen05 needs sm_100a")] },
  -- Block-scaled tcgen05.mma of each kind, FP6 and FP4 operands, A from tensor memory.
  { file := "tests/diff/tc5mx.ptx", kernel := "tc5mx", threads := 128, per := 384,
    bufs := [Array.replicate (384 * 128) 0, inputs edges32 (1536 + 384 * 128)],
    forms := ["UTCQMMA", "UTCOMMA.4X", "UTCOMMA"],
    formTargets := ["sm_100a"],
    skipOn := [("sm_80", "tcgen05 needs sm_100a"), ("sm_90a", "tcgen05 needs sm_100a"),
      ("sm_120a", "tcgen05 needs sm_100a")] },
  { file := "tests/diff/rarrive.ptx", kernel := "carrive", threads := 64,
    bufs := [Array.replicate 64 0, words 7 64], forms := ["SYNCS.ARRIVE.TRANS64.ART0"],
    skipOn := [("sm_80", "no mbarrier arrive with a count before sm_90")] },
  -- Textures and surfaces (handle 1 over buffer 1). Texels: small floats or bytes.
  { file := "tests/diff/tex.ptx", kernel := "tex2d", threads := 64, per := 4,
    bufs := [Array.replicate (4 * 64) 0, smallFloats (8 * 6 * 4)], extra := [le 1 8],
    textures := [(1, { base := 0, dims := 2, width := 8, height := 6, pitch := 128,
                       chans := #[32, 32, 32, 32], kind := .float, filter := .linear,
                       address := #[.clamp, .border] })],
    forms := ["TEX"], formTargets := ["sm_80", "sm_90a", "sm_120a"] },
  { file := "tests/diff/tex.ptx", kernel := "tex2d", threads := 64, per := 4,
    bufs := [Array.replicate (4 * 64) 0, words 7 (6 * 4)], extra := [le 1 8],
    textures := [(1, { base := 0, dims := 2, width := 5, height := 4, pitch := 24,
                       chans := #[8, 8, 8, 8], kind := .unsigned, readNormalized := true,
                       filter := .linear, normalizedCoords := true,
                       address := #[.wrap, .mirror] })] },
  -- Offsets over the two tex2d textures (clamp/border unnormalized, wrap/mirror normalized), with
  -- a 16-texel 1D texture after the 2D one.
  { file := "tests/diff/texoff.ptx", kernel := "texoff", threads := 64, per := 8,
    bufs := [Array.replicate (8 * 64) 0, smallFloats (8 * 6 * 4 + 16)], extra := [le 1 8, le 2 8],
    textures := [(1, { base := 0, dims := 2, width := 8, height := 6, pitch := 128,
                       chans := #[32, 32, 32, 32], kind := .float, filter := .linear,
                       address := #[.clamp, .border] }),
                 (1, { base := 768, dims := 1, width := 16, chans := #[32], kind := .float,
                       filter := .linear, address := #[.border] })],
    forms := ["TEX"], formTargets := ["sm_80", "sm_90a", "sm_120a"] },
  { file := "tests/diff/texoff.ptx", kernel := "texoff", threads := 64, per := 8,
    bufs := [Array.replicate (8 * 64) 0, words 7 (6 * 4 + 16)], extra := [le 1 8, le 2 8],
    textures := [(1, { base := 0, dims := 2, width := 5, height := 4, pitch := 24,
                       chans := #[8, 8, 8, 8], kind := .unsigned, readNormalized := true,
                       filter := .linear, normalizedCoords := true, address := #[.wrap, .mirror] }),
                 (1, { base := 96, dims := 1, width := 16, chans := #[8, 8, 8, 8], kind := .unsigned,
                       readNormalized := true, filter := .linear, normalizedCoords := true,
                       address := #[.mirror] })] },
  { file := "tests/diff/tex.ptx", kernel := "tex1di", threads := 32, per := 4,
    bufs := [Array.replicate (4 * 32) 0, words 9 16], extra := [le 1 8],
    textures := [(1, { base := 0, dims := 1, width := 16, chans := #[32], kind := .unsigned,
                       address := #[.border] })] },
  { file := "tests/diff/tex.ptx", kernel := "gather", threads := 64, per := 4,
    bufs := [Array.replicate (4 * 64) 0, smallFloats (8 * 6 * 2)], extra := [le 1 8],
    textures := [(1, { base := 0, dims := 2, width := 8, height := 6, pitch := 64,
                       chans := #[32, 32], kind := .float, address := #[.clamp, .clamp] })] },
  { file := "tests/diff/tex.ptx", kernel := "query", threads := 32,
    bufs := [Array.replicate 8 0, Array.replicate 64 0], extra := [le 1 8],
    textures := [(1, { base := 0, dims := 2, width := 8, height := 6, pitch := 64,
                       chans := #[32, 32], kind := .float })] },
  { file := "tests/diff/tex.ptx", kernel := "surf", threads := 64, per := 2,
    bufs := [Array.replicate (2 * 64) 0, words 11 (16 * 4 * 2)], extra := [le 1 8],
    textures := [(1, { base := 0, dims := 2, width := 16, height := 4, pitch := 128,
                       chans := #[32, 32], kind := .unsigned, surface := true })],
    forms := ["SULD", "SUST", "SURED"], formTargets := ["sm_80", "sm_90a", "sm_120a"] },
  -- Layered objects over one buffer: three 8 x 6 float layers (texture), three 16-texel layers
  -- of a 1D float texture, three 16 x 4 two-word layers (surface).
  { file := "tests/diff/vatom.ptx", kernel := "vatom", threads := 32, per := 32,
    bufs := [words 23 (32 * 32), smallFloats (16 * 32)],
    forms := ["REDG.E.ADD.F32x4.FTZ.RN.STRONG.GPU", "ATOMG.E.ADD.F32x2.FTZ.RN.STRONG.GPU",
      "REDG.E.ADD.F16x8.RN.STRONG.GPU", "ATOMG.E.MAX.BF16x8.RN.STRONG.GPU",
      "ATOMG.E.EXCH.128.STRONG.GPU", "ATOMG.E.CAS.128.STRONG.GPU"],
    formTargets := ["sm_90a"], skipOn := [("sm_80", "vector and 128-bit atomics need sm_90")] },
  { file := "tests/diff/wgmma.ptx", kernel := "wgb1", threads := 128, per := 8,
    bufs := [Array.replicate (8 * 128) 0, words 31 (512 + 128)], formTargets := ["sm_90a"],
    forms := ["BGMMA.64x16x256.AND.POPC"],
    skipOn := [("sm_80", "no wgmma before sm_90a"), ("sm_120a", "no wgmma on sm_120a")] },
  { file := "tests/diff/wgmma.ptx", kernel := "wgrs", threads := 128, per := 8,
    bufs := [Array.replicate (8 * 128) 0, smallFloats (512 + 128)], formTargets := ["sm_90a"],
    forms := ["HGMMA.64x16x16.F32.BF16"],
    skipOn := [("sm_80", "no wgmma before sm_90a"), ("sm_120a", "no wgmma on sm_120a")] },
  -- Two 4 x 4 float cubemaps (twelve faces of 64 bytes) in buffer 1.
  { file := "tests/diff/cube.ptx", kernel := "cube", threads := 64, per := 8,
    bufs := [Array.replicate (8 * 64) 0, smallFloats (12 * 16)],
    extra := [le 1 8],
    textures := [(1, { base := 0, dims := 2, width := 4, height := 4, pitch := 16, slice := 64,
                       layers := 12, cube := true, chans := #[32], kind := .float,
                       filter := .linear, normalizedCoords := true })],
    forms := ["TEX"], formTargets := ["sm_80", "sm_90a", "sm_120a"] },
  { file := "tests/diff/layered.ptx", kernel := "layered", threads := 64, per := 12,
    bufs := [Array.replicate (12 * 64) 0, smallFloats (3 * 8 * 6 + 3 * 16) ++ words 13 (3 * 16 * 4 * 2)],
    extra := [le 1 8, le 2 8, le 3 8],
    textures := [(1, { base := 0, dims := 2, width := 8, height := 6, pitch := 32, slice := 192,
                       layers := 3, chans := #[32], kind := .float, filter := .linear,
                       address := #[.clamp, .border] }),
                 (1, { base := 576, dims := 1, width := 16, pitch := 64, layers := 3, chans := #[32],
                       kind := .float, filter := .linear, address := #[.clamp] }),
                 (1, { base := 768, dims := 2, width := 16, height := 4, pitch := 128, slice := 512,
                       layers := 3, chans := #[32, 32], kind := .unsigned, surface := true })],
    forms := ["SULD", "SUST"], formTargets := ["sm_80", "sm_90a", "sm_120a"] },
  -- The opcode gaps: edge words (signed and unsigned extremes, float specials) as operands.
  { file := "tests/diff/gaps.ptx", kernel := "gaps", threads := 64, per := 16,
    bufs := [Array.replicate (16 * 64 + 64) 0, inputs edges32 (4 * 64)],
    forms := ["I2IP", "F2IP", "HSET2", "QSPC", "UBLKPF"], formTargets := ["sm_90a"],
    skipOn := [("sm_80", "cp.async.bulk.prefetch needs sm_90")] },
  -- tex.grad (TXD): the gradients read, not used, on the launch's one-level textures.
  { file := "tests/diff/texgrad.ptx", kernel := "texgrad", threads := 64, per := 4,
    bufs := [Array.replicate (4 * 64) 0, smallFloats (8 * 6 * 4)], extra := [le 1 8],
    textures := [(1, { base := 0, dims := 2, width := 8, height := 6, pitch := 128,
                       chans := #[32, 32, 32, 32], kind := .float, filter := .linear,
                       address := #[.clamp, .border] })],
    forms := ["TXD"], formTargets := ["sm_80", "sm_90a", "sm_120a"] },
  -- multimem.ld_reduce (LDGMC) on the device's own multicast object.
  { file := "tests/diff/ldreduce.ptx", kernel := "ldreduce", threads := 32, per := 4,
    bufs := [Array.replicate (4 * 32) 0, inputs edges32 (4 * 32)],
    forms := ["LDGMC"], formTargets := ["sm_90a"],
    skipOn := [("sm_80", "multimem needs sm_90")] },
  -- Integer, sub-byte and single-bit mma.sync: any bits.
  { file := "tests/diff/mmaint.ptx", kernel := "mmaint", threads := 32, per := 24,
    bufs := [Array.replicate (24 * 32) 0, inputs edges32 (8 * 32)],
    forms := ["IMMA.8816.S8.U8", "IMMA.8832.U4.U4", "IMMA.16832.S4.U4", "IMMA.16864.S4.S4.SAT",
      "BMMA.88128.AND.POPC", "BMMA.168128.AND.POPC", "BMMA.168256.XOR.POPC"],
    formTargets := ["sm_80"] },
  -- Sparse integer mma: any bits, metadata words of ordered index pairs.
  { file := "tests/diff/mmasp.ptx", kernel := "mmasp", threads := 32, per := 20,
    bufs := [Array.replicate (20 * 32) 0, (inputs edges32 (16 * 32)).mapIdx fun j w =>
      if j % 16 != 12 then w else
      let pairs : Array UInt32 := #[0x4, 0x8, 0x9, 0xc, 0xd, 0xe]
      (List.range 8).foldl (fun acc i => acc ||| (pairs[((w >>> (4 * i).toUInt32) % 6).toNat]! <<< (4 * i).toUInt32)) 0],
    forms := ["IMMA.SP.16832.S8.S8", "IMMA.SP.16832.U8.S8.SAT", "IMMA.SP.16864.U8.U8",
      "IMMA.SP.16864.S4.U4", "IMMA.SP.168128.S4.S4.SAT"],
    formTargets := ["sm_80"] },
  -- The sm_90 binary64 shapes: doubles of moderate exponent (high words), edge values in some.
  { file := "tests/diff/mmaf64.ptx", kernel := "mmaf64", threads := 32, per := 24,
    bufs := [Array.replicate (24 * 32) 0, (inputs edges32 (32 * 32)).mapIdx fun j w =>
      if j % 2 == 1 && w &&& 0x100 == 0 then (w &&& 0x800fffff) ||| 0x3fc00000 else w],
    forms := ["DMMA.16x8x4", "DMMA.16x8x8", "DMMA.16x8x16"], formTargets := ["sm_90a"],
    skipOn := [("sm_80", "the m16n8 binary64 shapes need sm_90")] },
  { file := "tests/diff/mma8.ptx", kernel := "mmas8", threads := 32, per := 16,
    -- E4M3/E5M2 bytes of moderate exponent, with edge values in some lanes.
    bufs := [Array.replicate (16 * 32) 0, (inputs edges32 (16 * 32)).map fun w =>
      if w &&& 0x100 == 0 then (w &&& 0x87878787) ||| 0x30303030 else w],
    forms := ["QMMA.16832.F32.E4M3.E4M3", "QMMA.16832.F32.E5M2.E4M3"], formTargets := ["sm_120a"],
    skipOn := [("sm_80", "no FP8 mma.sync before sm_89")] },
  -- Into f16 accumulators: ptxas's FP16 lowering on sm_90a and sm_100a (`Matrix.loweredF8`).
  { file := "tests/diff/mma8.ptx", kernel := "mmas8h", threads := 32, per := 16,
    bufs := [Array.replicate (16 * 32) 0, (inputs edges32 (16 * 32)).map fun w =>
      if w &&& 0x100 == 0 then (w &&& 0x87878787) ||| 0x30303030 else w],
    forms := ["HMMA.16816.F16"], formTargets := ["sm_90a"],
    skipOn := [("sm_80", "no FP8 mma.sync before sm_89"),
               ("sm_120a", "no published numerics of sm_120's FP8 datapath into FP16")] },
  -- FP6/FP4 and block-scaled mma.sync (sm_120a): moderate elements as for FP8, scale words of bytes
  -- 0x78-0x87 (UE8M0 2^-7 to 2^8).
  { file := "tests/diff/mma120.ptx", kernel := "mma120", threads := 32, per := 24,
    bufs := [Array.replicate (24 * 32) 0, ((inputs edges32 (16 * 32)).zipIdx).map fun (w, i) =>
      if i % 16 ≥ 12 then (w &&& 0x0f0f0f0f) ||| 0x78787878
      else if i % 16 < 8 && w &&& 0x100 == 0 then (w &&& 0x87878787) ||| 0x30303030 else w],
    forms := ["QMMA.16832.F32.E2M1.E3M2", "QMMA.16832.F32.E4M3.E2M3", "QMMA.SF.16832.F32.E4M3.E2M3.E8",
      "OMMA.SF.16864.F32.E2M1.E2M1.E8", "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X"],
    formTargets := ["sm_120a"],
    skipOn := [("sm_80", "kind::f8f6f4 needs sm_120a"), ("sm_90a", "kind::f8f6f4 needs sm_120a"),
      ("sm_100a", "kind::f8f6f4 needs sm_120a")] },
  { file := "tests/diff/nw.ptx", kernel := "nwchain", threads := 64, per := 3,
    bufs := [Array.replicate (3 * 64) 0, (words 31 (4 * 64)).map (· % 64 - 32)],
    extra := [le 10 4] },
  { file := "tests/diff/nw.ptx", kernel := "min3", threads := 64, per := 4,
    bufs := [Array.replicate (4 * 64) 0, inputs edges32 (4 * 64)], forms := ["VIMNMX3", "LEA.HI"] },
  { file := "tests/diff/ampere.ptx", kernel := "ampere", threads := 64, per := 16,
    -- The partial copies' source is all ones, so a copied byte shows as 0xff, a filled one as 0.
    bufs := [Array.replicate (16 * 64) 0,
      (inputs edges32 2048).mapIdx fun j w => if j ≥ 320 && j < 448 then 0xffffffff else w],
    forms := ["LDGSTS", "LDGDEPBAR", "DEPBAR", "LDSM"],
    formTargets := ["sm_80", "sm_90a", "sm_120a"] },
  { file := "tests/diff/misc.ptx", kernel := "sad4", threads := 64, per := 2,
    bufs := [Array.replicate (2 * 64) 0, inputs #[0, 0xffffffff, 7, 0x80808080, 0x7f7f7f7f, 0xffff0000] (3 * 64)],
    forms := ["VABSDIFF4.U8.ACC"], formTargets := ["sm_80", "sm_90a"] },
  -- Two warps: the barrier reductions span them. redux.sync needs sm_80.
  { file := "tests/diff/barred.ptx", kernel := "barred", threads := 64, per := 8,
    bufs := [Array.replicate (8 * 64) 0, inputs #[0, 0xffffffff, 7, 0xf0000001] 64],
    forms := ["BAR.RED.POPC", "BAR.RED.AND", "BAR.RED.OR", "B2R.RESULT", "REDUX"],
    formTargets := ["sm_80", "sm_90a", "sm_120a"] },
  { file := "tests/diff/int64.ptx", kernel := "int64", threads := 64, per := 14,
    bufs := [Array.replicate (14 * 64) 0,
      inputs #[0, 0x80000000, 0xffffffff, 0x7fffffff, 1, 0, 0xffffffff, 0xffffffff] (4 * 64)],
    forms := ["IMNMX.S64", "IMNMX.U64", "SEL.64"], formTargets := ["sm_120a"] },
  { file := "tests/diff/misc.ptx", kernel := "misc", threads := 64, per := 16,
    bufs := [Array.replicate (16 * 64) 0, inputs edges32 (3 * 64)] },
  { file := "tests/diff/ldsm.ptx", kernel := "ldsm", threads := 32,
    bufs := [Array.replicate (16 * 32 + 16 * 32) 0, words 99 (16 * 32)] },
  { file := "tests/diff/ldsm8.ptx", kernel := "ldsm8", threads := 32,
    bufs := [Array.replicate (16 * 32 + 16 * 32) 0, words 99 (16 * 32)],
    forms := ["LDSM.8.MT1616.2", "LDSM.U4x16P64TO8.MT1616", "LDSM.U6x16P32TO8.M816.4",
      "LDSM.U4x16P64TO8.M816", "STSM.8.MT168.4", "STSM.8.MT168", "STSM.8.MT168.2"],
    formTargets := ["sm_120a"],
    skipOn := [("sm_80", "the .m16n16/.m8n16/.m16n8 forms need sm_100a or sm_120a"),
      ("sm_90a", "the .m16n16/.m8n16/.m16n8 forms need sm_100a or sm_120a")] },
  -- The PTX part's atomics kernel: f32 adds on global (flushing) and shared (keeping subnormals),
  -- f16x2 and f64, over two CTAs. Every sum is exact, so the order of the atomics does not matter.
  { file := "tests/ptx/kernels/atomics.sm_90a.ptx", kernel := "atomics", grid := 2, threads := 32,
    bufs := [Array.replicate 16 0,
      (Array.range 64).map fun i =>
        if i < 32 then (0x3f800000 + 0x800000 * (i % 3)).toUInt32 else 0x200,
      Array.replicate 1 0, (Array.range 64).map fun i => 0x3c004000 + (i % 5).toUInt32,
      Array.replicate 2 0, (Array.range 128).map fun i => if i % 2 == 1 then 0x3ff00000 else 0],
    extra := [le 64 4] } ]

/-- Byte offsets of the buffers in global memory, as `Tests.run` lays them out. -/
def offsets (bufs : List (Array UInt32)) : List Nat :=
  (bufs.foldl (fun (acc : List Nat × Nat) b =>
    (acc.1 ++ [acc.2], acc.2 + (4 * b.size + 255) / 256 * 256)) ([], 0)).1

/-- The PTX interpreter's final buffers. -/
def interp (c : Case) (src : String) : Except String (List (Array UInt32)) := do
  let m ← NV.PTX.read src
  let prog ← NV.PTX.Sem.Prog.ofModule m
  let offs := offsets c.bufs
  let size := (c.bufs.zip offs).foldl
    (fun acc (b, o) => max acc (o + (4 * b.size + 255) / 256 * 256)) 256
  let g := (c.bufs.zip offs).foldl (fun g (b, o) => g.write (base + o.toUInt64) (4 * b.size) b 0)
    (Store.empty base size)
  let addr (o : Nat) : List UInt8 := (List.range 8).map fun j => ((base.toNat + o) >>> (8 * j)).toUInt8
  let params ← NV.PTX.paramBytes m c.kernel (offs.map addr ++ c.extra)
  let out ← NV.PTX.interp prog { entry := c.kernel, grid := (c.grid, 1, 1),
                                 block := (c.threads, 1, 1), cluster := (c.cluster, 1, 1),
                                 params, global := g,
                                 textures := (c.textures.map fun (b, t) =>
                                   { t with base := base + (offs.getD b 0).toUInt64 + t.base }).toArray }
  return (c.bufs.zip offs).map fun (b, o) => out.read (base + o.toUInt64) (4 * b.size)

def check (c : Case) : IO (List String) := do
  let k ← kernel (← compile c.file) c.kernel
  let forms := if c.formTargets.contains (← arch) then c.forms else []
  let lat := latencyProblems ((NV.Arch.forSm k.sm).getD NV.Arch.h200) k ++ forms.filterMap fun f =>
    if k.code.any (·.full.startsWith f) then none else some s!"{c.kernel}: ptxas emitted no {f}"
  let ptx := interp c (← ptxFor c.file)
  let sass ← run k c.grid c.threads c.bufs c.extra (cluster := (c.cluster, 1, 1))
    (textures := c.textures)
  match ptx, sass with
  | .error e, _ => return lat ++ [s!"{c.kernel}: PTX interpreter: {e}"]
  | _, .error e => return lat ++ [s!"{c.kernel}: SASS: {e}"]
  | .ok as, .ok bs =>
    let unequal (run : String) (out : Array UInt32) : List String :=
      ((List.range (if c.per > 0 then out.size / c.per else 0)).flatMap fun th =>
        c.same.filterMap fun (i, j) =>
          let (x, y) := (out[th * c.per + i]!, out[th * c.per + j]!)
          if x == y then none else some s!"{c.kernel} {run} thread {th}: output {i} {x}, output {j} {y}").take 12
    let sameFails := unequal "PTX" (as.headD #[]) ++ unequal "SASS" (bs.headD #[])
    return lat ++ sameFails ++ ((as.zip bs).zipIdx.flatMap fun ((a, b), n) =>
      (List.range a.size).filterMap fun j =>
        let where_ := if n == 0 && c.per > 0 then s!"thread {j / c.per} output {j % c.per}"
          else s!"buffer {n} word {j}"
        if a[j]! == b[j]! || (n == 0 && c.per > 0 && c.known.any (·.1 == j % c.per)) then none
        else some s!"{c.kernel} {where_}: PTX {a[j]!}, SASS {b[j]!}").take 12

def main (args : List String) : IO UInt32 := do
  let mut failures := 0
  for c in cases.filter fun c => args.isEmpty || args.contains c.kernel do
    if let some why := c.skipOn.lookup (← arch) then
      IO.println s!"{c.kernel}: skipped on {← arch}: {why}"; continue
    let fs ← try check c catch e =>
      match unsupported e with
      | some why => do IO.println s!"{c.kernel}: skipped on {← arch}: {why}"; continue
      | none => pure [s!"{c.kernel}: {e}"]
    IO.println s!"{c.kernel}: {if fs.isEmpty then "agree" else "DIFFER"}"
    for (j, why) in c.known do IO.println s!"  output {j} not compared: {why}"
    for f in fs do IO.eprintln s!"  {f}"
    failures := failures + fs.length
  return if failures == 0 then 0 else 1
