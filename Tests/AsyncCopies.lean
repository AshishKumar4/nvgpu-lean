import NV.SASS.Read
import NV.Core.Engine
import NV.Arch.H200
import NV.Async.Tma

/-!
`lake exe async-copies`: the kernels of `tests/kernels/async_copies.cu`, compiled by the pinned
`nvcc` for `sm_90a` and run end to end on the H200 parameters, every output byte checked
against what the PTX ISA defines:

* `cpasync`: `cp.async` with `src-size` and `ignore-src` zero fill (§9.7.10.28.3.1), completed
  through `wait_group` and through `cp.async.mbarrier.arrive.noinc` (`LDGSTS`, `LDGDEPBAR`,
  `DEPBAR`, `ARRIVES`);
* `bulk`: `cp.async.bulk` both ways with mbarrier and bulk-group completion (`UBLKCP`);
* `bulk_red`: `cp.reduce.async.bulk` max.s32 and add.u32 into global memory (`UBLKRED`);
* `tma_oob`: tiled `cp.async.bulk.tensor` loads whose boxes cross the tensor's edges, zero
  filled (§5.5.3.3), and the stores back, which skip the elements outside (`UTMALDG`,
  `UTMASTG`).
-/

open NV NV.SASS NV.Core NV.Async

def tool (cmd : String) (args : Array String) : IO String := do
  let out ← IO.Process.output { cmd, args }
  if out.exitCode != 0 then throw (IO.userError s!"{cmd} {args}: {out.stderr}")
  return out.stdout

def le (v : Nat) (bytes : Nat) : List UInt8 := (List.range bytes).map fun i => (v >>> (8 * i)).toUInt8

def paramsOf (k : Kernel) (args : List (List UInt8)) : ByteArray := Id.run do
  let mut ps := ByteArray.mk (Array.replicate k.paramBytes 0)
  for ((off, sz), bytes) in k.params.toList.zip args do
    for j in [0:sz] do ps := ps.set! (off + j) (bytes[j]?.getD 0)
  return ps

def base : UInt64 := 0x7f0000000000

/-- Run `k` and compare the bytes at `out` with `want`. -/
def check (name : String) (k : Kernel) (grid block : Nat × Nat × Nat) (args : List (List UInt8))
    (g : Store) (out : UInt64) (want : Array UInt8) : IO Bool := do
  let launch : Launch := { kernel := k, grid, block, params := paramsOf k args, global := g }
  match run NV.Arch.h200 launch with
  | .error e => IO.eprintln s!"{name}: {e}"; return false
  | .ok res =>
    let mut bad := 0
    for i in [0:want.size] do
      let got := res.global.readByte ((out - res.global.base).toNat + i)
      if got != want[i]! then
        bad := bad + 1
        if bad ≤ 5 then IO.eprintln s!"{name}: byte {i} = {got}, expected {want[i]!}"
    IO.println s!"{name}: {res.counters.cycles} cycles, {res.counters.instExecuted} warp instructions, \
      {res.counters.tmaBytes} TMA bytes; {if bad == 0 then "ok" else s!"{bad} bytes wrong"}"
    return bad == 0

def f32OfNat (m : Nat) : Nat :=
  if m == 0 then 0 else
  let e := Nat.log2 m
  (e + 127) * 2 ^ 23 + (m * 2 ^ 23 / 2 ^ e) % 2 ^ 23

def bytesOfWords (ws : Array Nat) : Array UInt8 :=
  ws.flatMap fun w => #[w.toUInt8, (w >>> 8).toUInt8, (w >>> 16).toUInt8, (w >>> 24).toUInt8]

def main : IO UInt32 := do
  let dir ← IO.FS.createTempDir
  let cubin := (dir / "copies.cubin").toString
  discard <| tool "nvcc" #["-arch=sm_90a", "-cubin", "-o", cubin, "tests/kernels/async_copies.cu"]
  let ks ← IO.ofExcept (Read.kernels (← tool "cuobjdump" #["-sass", cubin]) (← tool "cuobjdump" #["-elf", cubin]))
  let find (n : String) : IO Kernel := do
    let some k := ks.find? (·.name == n) | throw (IO.userError s!"{n} not found")
    pure k
  let mut ok := true
  -- cp.async
  let src := (Array.range 4096).map fun i => ((7 * i + 3) % 256).toUInt8
  let mut g := Store.empty base 8192
  for i in [0:4096] do g := g.writeByte i src[i]!
  let want := (Array.range 3584).map fun i =>
    if i < 2048 then src[i]!
    else if i < 3072 then (let t := (i - 2048) / 8; if (i - 2048) % 8 < t % 9 then src[i]! else 0)
    else (let t := (i - 3072) / 4; if t % 3 == 1 then 0 else src[i]!)
  ok := (← check "cpasync" (← find "cpasync") (1, 1, 1) (128, 1, 1)
    [le base.toNat 8, le (base.toNat + 4096) 8, le 0 4] g (base + 4096) want) && ok
  -- Bulk copies.
  let words := (Array.range 1024).map fun i => (i * 2654435761) % 2 ^ 32
  g := Store.empty base 8192
  for i in [0:1024] do g := g.writeWord (4 * i) words[i]!.toUInt32
  ok := (← check "bulk" (← find "bulk") (1, 1, 1) (128, 1, 1) [le base.toNat 8, le (base.toNat + 4096) 8] g
    (base + 4096) (bytesOfWords ((Array.range 1024).map fun i => (words[i]! * 3 + i) % 2 ^ 32))) && ok
  -- Bulk reductions: max as s32 into `words`, then add as u32 into `words` again.
  let sv := (Array.range 1024).map fun i => (i * 2654435761) % 2 ^ 32
  let s32 (x : Nat) : Int := if x ≥ 2 ^ 31 then (x : Int) - 2 ^ 32 else x
  g := Store.empty base 8192
  for i in [0:1024] do
    g := g.writeWord (4 * i) words[i]!.toUInt32
    g := g.writeWord (4096 + 4 * i) words[i]!.toUInt32
  let redWant := (Array.range 1024).map (fun i =>
      if s32 sv[i]! > s32 words[i]! then sv[i]! else words[i]!) ++
    (Array.range 1024).map fun i => (words[i]! + sv[i]!) % 2 ^ 32
  ok := (← check "bulk_red" (← find "bulk_red") (1, 1, 1) (128, 1, 1) [le base.toNat 8] g base
    (bytesOfWords redWant)) && ok
  -- TMA boxes over the edges of a 20 × 10 f32 tensor.
  let (w, h) := (20, 10)
  let v (x y : Nat) : Nat := f32OfNat (y * w + x + 1)
  let inA := base
  let outA := base + 1024
  let boxA := base + 2048
  g := Store.empty base 4096
  for y in [0:h] do
    for x in [0:w] do g := g.writeWord (4 * (y * w + x)) (v x y).toUInt32
  let map (a : UInt64) : TensorMap :=
    { dtype := .f32, rank := 2, globalAddress := a, globalDim := #v[w, h, 1, 1, 1],
      globalStrides := #v[4 * w, 0, 0, 0], boxDim := #v[16, 8, 1, 1, 1],
      elementStrides := #v[1, 1, 1, 1, 1] }
  let boxes := (Array.range 512).map fun i =>
    let cta := i / 128
    let t := i % 128
    let x : Int := ((16 * (cta % 2) + t % 16 : Nat) : Int) - 4
    let y : Int := ((8 * (cta / 2) + t / 16 : Nat) : Int) - 3
    if 0 ≤ x && x < w && 0 ≤ y && y < h then v x.toNat y.toNat else 0
  let k ← find "tma_oob"
  let args := [(TensorMap.encode (map inA)).data.toList, (TensorMap.encode (map outA)).data.toList,
               le boxA.toNat 8]
  ok := (← check "tma_oob boxes" k (2, 2, 1) (128, 1, 1) args g boxA (bytesOfWords boxes)) && ok
  -- The stores cover the tensor exactly once.
  ok := (← check "tma_oob store" k (2, 2, 1) (128, 1, 1) args g outA
    (bytesOfWords ((Array.range (w * h)).map fun i => v (i % w) (i / w)))) && ok
  return if ok then 0 else 1
