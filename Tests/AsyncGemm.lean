import NV.SASS.Read
import NV.Core.Engine
import NV.Arch.H200
import NV.Async.Tma
import NV.Num

/-!
`lake exe async-gemm`: a DeepGEMM-style TMA + mbarrier + wgmma GEMM, compiled by the pinned `nvcc`
for `sm_90a` (`tests/kernels/gemm_tma_wgmma.cu`), read back from `cuobjdump`, and run end to end
on the H200 parameters: `C = A · Bᵀ` with A (M×K) and B (N×K) BF16 in K-major order, C FP32.
The kernel's producer warp streams 64-wide K blocks of A and B through a four-stage ring of
shared memory with `cp.async.bulk.tensor` (128-byte swizzle) and full/empty mbarriers; its
consumer warpgroup runs `wgmma.mma_async m64n128k16` on each stage.

Two checks, both exact:
* integer-valued inputs, whose products and sums are exact in FP32 in any order, against integer
  arithmetic: an oracle independent of the model's layouts, swizzle and numerics;
* random BF16 inputs against the tensor-core accumulation model (`NV.Num.TensorModel.block`,
  one block per k16 instruction, chained in the kernel's K order), bit for bit.
-/

open NV NV.SASS NV.Core NV.Async NV.Num

def tool (cmd : String) (args : Array String) : IO String := do
  let out ← IO.Process.output { cmd, args }
  if out.exitCode != 0 then throw (IO.userError s!"{cmd} {args}: {out.stderr}")
  return out.stdout

def le (v : Nat) (bytes : Nat) : List UInt8 := (List.range bytes).map fun i => (v >>> (8 * i)).toUInt8

/-- The binary32 bits of a small integer. -/
def f32OfInt (x : Int) : Nat :=
  if x == 0 then 0 else
  let m := x.natAbs
  let e := Nat.log2 m
  (if x < 0 then 2 ^ 31 else 0) + (e + 127) * 2 ^ 23 + (m * 2 ^ 23 / 2 ^ e) % 2 ^ 23

/-- The bfloat16 bits of an integer of magnitude below 256 (exact). -/
def bf16OfInt (x : Int) : Nat := f32OfInt x / 2 ^ 16

/-- A row-major `rows × cols` BF16 matrix, K contiguous, as a 2-D tensor map with a `box` of
`64 × boxRows` and 128-byte swizzle. -/
def mapOf (addr : UInt64) (rows cols boxRows : Nat) : TensorMap :=
  { dtype := .bf16, rank := 2, globalAddress := addr, globalDim := #v[cols, rows, 1, 1, 1],
    globalStrides := #v[2 * cols, 0, 0, 0], boxDim := #v[64, boxRows, 1, 1, 1],
    elementStrides := #v[1, 1, 1, 1, 1], swizzle := .b128 }

structure Case where
  name : String
  a : Array Nat
  b : Array Nat
  want : Array Nat

def run1 (k : Kernel) (mDim nDim kDim : Nat) (c : Case) : IO Bool := do
  let aBase : UInt64 := 0x7f0000000000
  let bBase := aBase + (2 * mDim * kDim).toUInt64
  let cBase := bBase + (2 * nDim * kDim).toUInt64
  let mut g := Store.empty aBase (2 * mDim * kDim + 2 * nDim * kDim + 4 * mDim * nDim)
  for i in [0:c.a.size] do
    g := g.writeByte (2 * i) c.a[i]!.toUInt8 |>.writeByte (2 * i + 1) (c.a[i]! / 256).toUInt8
  for i in [0:c.b.size] do
    let off := 2 * mDim * kDim + 2 * i
    g := g.writeByte off c.b[i]!.toUInt8 |>.writeByte (off + 1) (c.b[i]! / 256).toUInt8
  let mapA := mapOf aBase mDim kDim 64
  let mapB := mapOf bBase nDim kDim 128
  let args : List (List UInt8) :=
    [(TensorMap.encode mapA).data.toList, (TensorMap.encode mapB).data.toList, le cBase.toNat 8,
     le nDim 4, le kDim 4]
  let mut params := ByteArray.mk (Array.replicate k.paramBytes 0)
  for ((off, sz), bytes) in k.params.toList.zip args do
    for j in [0:sz] do params := params.set! (off + j) (bytes[j]?.getD 0)
  -- Four stages of a 64×64 A tile and a 128×64 B tile, then eight mbarriers.
  let launch : Launch := { kernel := k, grid := (nDim / 128, mDim / 64, 1), block := (256, 1, 1),
                           params, global := g, dynShared := 4 * (8192 + 16384) + 64 }
  let t0 ← IO.monoMsNow
  match run NV.Arch.h200 launch with
  | .error e => IO.eprintln s!"{c.name}: {e}"; return false
  | .ok res =>
    let ms := (← IO.monoMsNow) - t0
    let out := res.global.read cBase (4 * mDim * nDim)
    let mut bad := 0
    for i in [0:mDim * nDim] do
      if out[i]!.toNat != c.want[i]! then
        bad := bad + 1
        if bad ≤ 5 then
          IO.eprintln s!"{c.name}: C[{i / nDim}][{i % nDim}] = {out[i]!.toNat}, expected {c.want[i]!}"
    IO.println s!"{c.name}: {res.counters.cycles} cycles, {res.counters.instExecuted} warp \
      instructions, {res.counters.tmaBytes} TMA bytes, {res.counters.gmmaInsts} HGMMA; \
      simulated in {ms} ms"
    if bad > 0 then IO.eprintln s!"{c.name}: {bad} of {mDim * nDim} outputs wrong"
    return bad == 0

def main : IO UInt32 := do
  let dir ← IO.FS.createTempDir
  let cubin := (dir / "gemm.cubin").toString
  discard <| tool "nvcc" #["-arch=sm_90a", "-cubin", "-o", cubin, "tests/kernels/gemm_tma_wgmma.cu"]
  let ks ← IO.ofExcept (Read.kernels (← tool "cuobjdump" #["-sass", cubin]) (← tool "cuobjdump" #["-elf", cubin]))
  let some k := ks.find? (·.name == "gemm") | throw (IO.userError "gemm not found")
  IO.println s!"gemm: {k.code.size} instructions, {k.regs} registers, params {k.params}"
  let (mDim, nDim, kDim) := (128, 256, 512)
  -- Integer-valued inputs in [-3, 3]: every partial sum is an integer below 2^24 in magnitude.
  let ai (i kk : Nat) : Int := ((7 * i + 3 * kk) % 5 : Nat) - 2
  let bi (j kk : Nat) : Int := ((5 * j + 11 * kk) % 7 : Nat) - 3
  let exact : Case :=
    { name := "gemm integer-valued"
      a := (Array.range (mDim * kDim)).map fun x => bf16OfInt (ai (x / kDim) (x % kDim))
      b := (Array.range (nDim * kDim)).map fun x => bf16OfInt (bi (x / kDim) (x % kDim))
      want := (Array.range (mDim * nDim)).map fun x =>
        f32OfInt ((List.range kDim).foldl (fun s kk => s + ai (x / nDim) kk * bi (x % nDim) kk) 0) }
  -- Random BF16 with exponents spanning 2^-7 … 2^7, so alignment and truncation matter.
  let rnd (seed x : Nat) : Nat :=
    let h := (x * 2654435761 + seed * 40503) % 2 ^ 32
    let h := (h ^^^ (h >>> 13)) * 1274126177 % 2 ^ 32
    (h % 2) * 2 ^ 15 + (120 + h / 2 % 15) * 2 ^ 7 + (h / 64) % 2 ^ 7
  let ra := (Array.range (mDim * kDim)).map (rnd 1)
  let rb := (Array.range (nDim * kDim)).map (rnd 2)
  let tm := Hopper.f16f32.value
  let ref (i j : Nat) : Nat := (List.range (kDim / 16)).foldl (init := 0) fun acc q =>
    tm.block Format.bf16 Format.bf16 Format.f32 (ra.extract (i * kDim + 16 * q) (i * kDim + 16 * q + 16))
      (rb.extract (j * kDim + 16 * q) (j * kDim + 16 * q + 16)) acc
  let random : Case :=
    { name := "gemm random bf16", a := ra, b := rb,
      want := (Array.range (mDim * nDim)).map fun x => ref (x / nDim) (x % nDim) }
  let ok1 ← run1 k mDim nDim kDim exact
  let ok2 ← run1 k mDim nDim kDim random
  -- A warp ready in a cycle its sub-partition's tensor core takes waits for the next free one: at
  -- most a cycle an issue goes to the dispatch stall, the rest of its waits (here a chain of
  -- shared-memory loads beside another warpgroup's wgmma) to their own reasons.
  let cubin2 := (dir / "beside.cubin").toString
  discard <| tool "nvcc" #["-arch=sm_90a", "-cubin", "-o", cubin2, "tests/kernels/wgmma_beside.cu"]
  let ks2 ← IO.ofExcept (Read.kernels (← tool "cuobjdump" #["-sass", cubin2]) (← tool "cuobjdump" #["-elf", cubin2]))
  let some kb := ks2.find? (·.name == "beside") | throw (IO.userError "beside not found")
  let mut params := ByteArray.mk (Array.replicate kb.paramBytes 0)
  for ((off, sz), v) in kb.params.toList.zip [0x7f0000000000, 64, 2000] do
    for (j, byte) in (List.range sz).zip (le v sz) do params := params.set! (off + j) byte
  let beside : Launch := { kernel := kb, grid := (1, 1, 1), block := (256, 1, 1), params,
                            global := Store.empty 0x7f0000000000 64, dynShared := 8192 + 16384 + 4096 }
  let ok3 ← match run NV.Arch.h200 beside with
    | .error e => do IO.eprintln s!"beside: {e}"; pure false
    | .ok res => do
      let lost := res.counters.stalls[Stall.dispatch.index]!
      IO.println s!"beside: {res.counters.cycles} cycles, {res.counters.instExecuted} issues, {lost} dispatch-stall cycles"
      if lost > res.counters.instExecuted then
        IO.eprintln s!"beside: {lost} dispatch-stall cycles over {res.counters.instExecuted} issues"
      pure (lost ≤ res.counters.instExecuted)
  -- An empty commit group (`gdesc[URZ]`: zero descriptors) runs and the warpgroup goes on.
  let some ke := ks2.find? (·.name == "empty_group") | throw (IO.userError "empty_group not found")
  let mut pe := ByteArray.mk (Array.replicate ke.paramBytes 0)
  for ((off, sz), v) in ke.params.toList.zip [0x7f0000000000] do
    for (j, byte) in (List.range sz).zip (le v sz) do pe := pe.set! (off + j) byte
  let empty : Launch := { kernel := ke, grid := (1, 1, 1), block := (128, 1, 1), params := pe,
                          global := Store.empty 0x7f0000000000 512 }
  let ok4 ← match run NV.Arch.h200 empty with
    | .error e => do IO.eprintln s!"empty_group: {e}"; pure false
    | .ok res =>
      let out := res.global.read 0x7f0000000000 512
      let got := (List.range 128).map fun i => out[i]!.toNat
      if got == List.range 128 then pure true
      else do IO.eprintln s!"empty_group: wrote {got.take 8}…"; pure false
  return if ok1 && ok2 && ok3 && ok4 then 0 else 1
