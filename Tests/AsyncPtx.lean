import NV.PTX.Interp
import NV.Async.Tma

/-!
`lake exe async-ptx`: the PTX semantics of the asynchronous instructions on the same CUDA kernels
the SASS tests run (`tests/kernels/*.cu`, compiled to PTX by the pinned `nvcc`), executed by the
interpreter (`NV.PTX.Interp`, blocking: every asynchronous operation completes at issue) and
checked against the same oracles: the PTX side of the PTX↔SASS agreement on TMA, mbarrier,
`cp.async`, bulk copies and wgmma.
-/

open NV NV.PTX NV.Async

def tool (cmd : String) (args : Array String) : IO String := do
  let out ← IO.Process.output { cmd, args }
  if out.exitCode != 0 then throw (IO.userError s!"{cmd} {args}: {out.stderr}")
  return out.stdout

def le (v : Nat) (bytes : Nat) : List UInt8 := (List.range bytes).map fun i => (v >>> (8 * i)).toUInt8

/-- A kernel's parameter block from each parameter's bytes, at `Sem.paramLayout`'s offsets. -/
def params (m : Module Instr) (entry : String) (vals : List (List UInt8)) : Except String ByteArray := do
  let kernel? : Item Instr → Option (Function Instr)
    | .func f => if f.name == entry && f.entry then some f else none
    | _ => none
  let some f := m.items.findSome? kernel? | throw s!"no kernel {entry}"
  let (layout, size) := Sem.paramLayout f.params
  let mut b := ByteArray.mk (Array.replicate size 0)
  for ((_, off), v) in layout.zip vals do
    for j in [0:v.length] do b := b.set! (off + j) v[j]!
  return b

def base : UInt64 := 0x7f0000000000

/-- Run a kernel thread by thread, round robin: the kernels wait on mbarriers in spin loops, which
need every thread to make progress. -/
def run (prog : Sem.Prog) (entry : String) (grid : Nat × Nat × Nat) (threads : Nat) (ps : ByteArray)
    (g : Store) (dynShared : Nat := 0) : Except String Store := do
  let m ← Machine.init prog { entry, grid, block := (threads, 1, 1), params := ps, global := g, dynShared }
  return (← m.run (fun m rs => rs[m.steps % rs.length]!) 200_000_000).global

def f32OfInt (x : Int) : Nat :=
  if x == 0 then 0 else
  let m := x.natAbs
  let e := Nat.log2 m
  (if x < 0 then 2 ^ 31 else 0) + (e + 127) * 2 ^ 23 + (m * 2 ^ 23 / 2 ^ e) % 2 ^ 23

def bytesOfWords (ws : Array Nat) : Array UInt8 :=
  ws.flatMap fun w => #[w.toUInt8, (w >>> 8).toUInt8, (w >>> 16).toUInt8, (w >>> 24).toUInt8]

def check (name : String) (got : Except String Store) (out : UInt64) (want : Array UInt8) : IO Bool := do
  IO.eprintln s!"{name}: checking"
  match got with
  | .error e => IO.eprintln s!"{name}: {e}"; return false
  | .ok g =>
    let mut bad := 0
    for i in [0:want.size] do
      let b := g.readByte ((out - g.base).toNat + i)
      if b != want[i]! then
        bad := bad + 1
        if bad ≤ 5 then IO.eprintln s!"{name}: byte {i} = {b}, expected {want[i]!}"
    IO.println s!"{name}: {if bad == 0 then "ok" else s!"{bad} bytes wrong"}"
    return bad == 0

def main : IO UInt32 := do
  let dir ← IO.FS.createTempDir
  let ptxOf (src : String) : IO (Module Instr) := do
    let out := (dir / "k.ptx").toString
    discard <| tool "nvcc" #["-arch=sm_90a", "-ptx", "-o", out, src]
    IO.ofExcept (read (← IO.FS.readFile out))
  let mut ok := true
  -- cp.async, bulk copies and TMA boxes.
  let m ← ptxOf "tests/kernels/async_copies.cu"
  let prog ← IO.ofExcept (Sem.Prog.ofModule m)
  let src := (Array.range 4096).map fun i => ((7 * i + 3) % 256).toUInt8
  let mut g := Store.empty base 8192
  for i in [0:4096] do g := g.writeByte i src[i]!
  let want := (Array.range 3584).map fun i =>
    if i < 2048 then src[i]!
    else if i < 3072 then (let t := (i - 2048) / 8; if (i - 2048) % 8 < t % 9 then src[i]! else 0)
    else (let t := (i - 3072) / 4; if t % 3 == 1 then 0 else src[i]!)
  let ps ← IO.ofExcept (params m "cpasync" [le base.toNat 8, le (base.toNat + 4096) 8, le 0 4])
  ok := (← check "cpasync" (run prog "cpasync" (1, 1, 1) 128 ps g) (base + 4096) want) && ok
  let words := (Array.range 1024).map fun i => (i * 2654435761) % 2 ^ 32
  g := Store.empty base 8192
  for i in [0:1024] do g := g.writeWord (4 * i) words[i]!.toUInt32
  let ps ← IO.ofExcept (params m "bulk" [le base.toNat 8, le (base.toNat + 4096) 8])
  ok := (← check "bulk" (run prog "bulk" (1, 1, 1) 128 ps g) (base + 4096)
    (bytesOfWords ((Array.range 1024).map fun i => (words[i]! * 3 + i) % 2 ^ 32))) && ok
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
  let ps ← IO.ofExcept (params m "bulk_red" [le base.toNat 8])
  ok := (← check "bulk_red" (run prog "bulk_red" (1, 1, 1) 128 ps g) base (bytesOfWords redWant)) && ok
  let (w, h) : Nat × Nat := (20, 10)
  let v (x y : Nat) : Nat := f32OfInt (y * w + x + 1)
  g := Store.empty base 4096
  for y in [0:h] do
    for x in [0:w] do g := g.writeWord (4 * (y * w + x)) (v x y).toUInt32
  let map (a : UInt64) : TensorMap :=
    { dtype := .f32, rank := 2, globalAddress := a, globalDim := #v[w, h, 1, 1, 1],
      globalStrides := #v[4 * w, 0, 0, 0], boxDim := #v[16, 8, 1, 1, 1],
      elementStrides := #v[1, 1, 1, 1, 1] }
  let ps ← IO.ofExcept (params m "tma_oob" [(TensorMap.encode (map base)).data.toList,
    (TensorMap.encode (map (base + 1024))).data.toList, le (base.toNat + 2048) 8])
  let res := run prog "tma_oob" (2, 2, 1) 128 ps g
  let boxes := (Array.range 512).map fun i =>
    let cta := i / 128
    let t := i % 128
    let x : Int := ((16 * (cta % 2) + t % 16 : Nat) : Int) - 4
    let y : Int := ((8 * (cta / 2) + t / 16 : Nat) : Int) - 3
    if 0 ≤ x && x < w && 0 ≤ y && y < h then v x.toNat y.toNat else 0
  ok := (← check "tma_oob boxes" res (base + 2048) (bytesOfWords boxes)) && ok
  ok := (← check "tma_oob store" res (base + 1024)
    (bytesOfWords ((Array.range (w * h)).map fun i => v (i % w) (i / w)))) && ok
  -- The GEMM, at one CTA: integer-valued inputs against integer arithmetic.
  let m ← ptxOf "tests/kernels/gemm_tma_wgmma.cu"
  let prog ← IO.ofExcept (Sem.Prog.ofModule m)
  let (mD, nD, kD) := (64, 128, 128)
  let ai (i kk : Nat) : Int := (((7 * i + 3 * kk) % 5 : Nat) : Int) - 2
  let bi (j kk : Nat) : Int := (((5 * j + 11 * kk) % 7 : Nat) : Int) - 3
  let bf (x : Int) : Nat := f32OfInt x / 2 ^ 16
  let aB := base
  let bB := base + (2 * mD * kD).toUInt64
  let cB := bB + (2 * nD * kD).toUInt64
  g := Store.empty base (2 * mD * kD + 2 * nD * kD + 4 * mD * nD)
  for i in [0:mD * kD] do
    let x := bf (ai (i / kD) (i % kD))
    g := g.writeByte (2 * i) x.toUInt8 |>.writeByte (2 * i + 1) (x / 256).toUInt8
  for i in [0:nD * kD] do
    let x := bf (bi (i / kD) (i % kD))
    let off := 2 * mD * kD + 2 * i
    g := g.writeByte off x.toUInt8 |>.writeByte (off + 1) (x / 256).toUInt8
  let tm (a : UInt64) (rows boxRows : Nat) : TensorMap :=
    { dtype := .bf16, rank := 2, globalAddress := a, globalDim := #v[kD, rows, 1, 1, 1],
      globalStrides := #v[2 * kD, 0, 0, 0], boxDim := #v[64, boxRows, 1, 1, 1],
      elementStrides := #v[1, 1, 1, 1, 1], swizzle := .b128 }
  let ps ← IO.ofExcept (params m "gemm" [(TensorMap.encode (tm aB mD 64)).data.toList,
    (TensorMap.encode (tm bB nD 128)).data.toList, le cB.toNat 8, le nD 4, le kD 4])
  let want := (Array.range (mD * nD)).map fun x =>
    f32OfInt ((List.range kD).foldl (fun s kk => s + ai (x / nD) kk * bi (x % nD) kk) 0)
  ok := (← check "gemm integer-valued" (run prog "gemm" (1, 1, 1) 256 ps g (4 * (8192 + 16384) + 64))
    cB (bytesOfWords want)) && ok
  return if ok then 0 else 1
