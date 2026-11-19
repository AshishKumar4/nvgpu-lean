import Tests.Kernels
import NV.PTX.Interp

/-!
`lake exe sass-printf`: device `printf`.

* The formatter (`NV.Printf.render`) against what C's `printf` prints for the same format and
  arguments (written here from the C standard's definitions).
* `tests/sem/printf.cu`, compiled by the pinned `nvcc`, read back (module data, the relocated
  bank 4, `vprintf` as an external) and run for one warp, effects applied in order: every lane's
  record, and lane 3's second one, render as the host would print them.
* The same kernel's PTX in the PTX interpreter prints the same lines.
-/

open NV NV.SASS NV.Core Tests

def le64 (v : UInt64) : List UInt8 := (List.range 8).map fun i => (v >>> (8 * i).toUInt64).toUInt8
def le32 (v : UInt32) : List UInt8 := (List.range 4).map fun i => (v >>> (8 * i).toUInt32).toUInt8

/-- (format, argument bytes, expected text). -/
def renders : List (String × List UInt8 × String) :=
  [ ("%d|%5d|%-5d|%05d|%+d", le32 42 ++ le32 42 ++ le32 42 ++ le32 0xFFFFFFF9 ++ le32 7,
      "42|   42|42   |-0007|+7"),
    ("%u %x %X %#x %o", le32 0xFFFFFFFF ++ le32 255 ++ le32 255 ++ le32 255 ++ le32 8,
      "4294967295 ff FF 0xff 10"),
    ("%ld %lu", le64 0xFFFFFFFFFFFFFFFF ++ le64 18446744073709551615, "-1 18446744073709551615"),
    -- 2.5 rounds to even at precision 0; 0.125 to three places; 1e-5 in %e and %g.
    ("%.0f %.3f %f %e %g %g", le64 0x4004000000000000 ++ le64 0x3FC0000000000000 ++
      le64 0x400921FB54442D18 ++ le64 0x3EE4F8B588E368F1 ++ le64 0x3EE4F8B588E368F1 ++
      le64 0x40FE240000000000, "2 0.125 3.141593 1.000000e-05 1e-05 123456"),
    ("%g %g %.3g %E", le64 0x4120000000000000 ++ le64 0x3FB999999999999A ++
      le64 0x4058FD70A3D70A3D ++ le64 0xC0C81C8000000000, "524288 0.1 100 -1.234500E+04"),
    ("%f %f %5.1f", le64 0x7FF0000000000000 ++ le64 0xFFF8000000000000 ++ le64 0xBFF0000000000000,
      "inf -nan  -1.0"),
    ("%c%c %%", le32 0x41 ++ le32 0x7A, "Az %"),
    ("%*d|%-*d|%.*f", le32 4 ++ le32 9 ++ le32 3 ++ le32 9 ++ le32 2 ++ [0, 0, 0, 0] ++
      le64 0x3FF8000000000000, "   9|9  |1.50") ]

/-- One warp of `k`, effects applied as they come (a blocking reading of the semantics): its final
state, or the fault. -/
def runWarp (k : Kernel) (env : Env) (fuel : Nat := 10000) : Except String Warp := do
  let ops := decode k
  let mut w := Warp.init (max k.regs 1) 32
  let mut locals := Array.replicate 32 (Store.empty 0 (max k.localBytes 16))
  for _ in [0:fuel] do
    if w.done then return w
    let some (pc, m) := w.current | throw "no lane can run"
    match (ops[pc]!).run env w m with
    | .next s | .sync _ s => w := s
    | .reg s wb => w := wb s
    | .mem r s wb =>
      let req := r s
      unless req.space == .local do throw s!"pc {pc}: only local memory is modelled here"
      let mut resp := Array.replicate (32 * words req.bytes) 0
      for l in [0:32] do
        if (req.addrs[l]?.join).isNone then continue
        let (st, out) ← locals[l]!.perform req l
        locals := locals.set! l st
        for j in [0:out.size] do resp := resp.set! (l * words req.bytes + j) out[j]!
      w := wb resp s
    | .exit => return w
    | .fault msg => throw msg
    | _ => throw s!"pc {pc}: an effect this test does not run"
  throw "no completion"

def main : IO UInt32 := do
  let mut bad := 0
  for (fmt, args, want) in renders do
    let got := NV.Printf.render fmt ⟨args.toArray⟩ fun _ => none
    if got == want then IO.println s!"ok   {fmt}"
    else bad := bad + 1; IO.println s!"FAIL {fmt}: {repr got}, expected {repr want}"
  let dir ← IO.FS.createTempDir
  let cubin := (dir / "p.cubin").toString
  discard <| tool "nvcc" #["-arch=sm_90a", "-cubin", "-o", cubin, "tests/sem/printf.cu"]
  let ks ← IO.ofExcept (Read.kernels (← tool "cuobjdump" #["-sass", cubin])
    (← tool "cuobjdump" #["-elf", cubin]))
  let k ← kernel ks "hello"
  unless k.externs.any (·.2 == "vprintf") do bad := bad + 1; IO.println "FAIL vprintf not an extern"
  let mut params := ByteArray.mk (Array.replicate k.paramBytes 0)
  for (b, j) in (le32 7 ++ le32 0x40200000).zipIdx do params := params.set! j b
  let launch : Launch := { kernel := k, grid := (1, 1, 1), block := (32, 1, 1), params,
                           global := Store.empty base 256 }
  let env : Env := { ctx := { ntid := (32, 1, 1), nctaid := (1, 1, 1), ctaid := (0, 0, 0), warp := 0,
                              threads := 32 }, const0 := constBank0 launch }
  match runWarp k env with
  | .error e => bad := bad + 1; IO.println s!"FAIL hello: {e}"
  | .ok w =>
    let texts := w.printf.toList.map fun (f, a) => printfText k f a
    let want := (List.range 32).map (s!"t{·}\n") ++ ["a=7 b=2.500\n"]
    if texts == want then IO.println "ok   hello: 33 records"
    else bad := bad + 1; IO.println s!"FAIL hello: {repr texts}"
    -- The PTX interpreter on the same kernel's PTX: the same lines (the order between threads
    -- is the hardware's, so compared sorted).
    let ptx ← tool "nvcc" #["-arch=sm_90a", "-ptx", "-o", "-", "tests/sem/printf.cu"]
    let lines : Except String (List String) := do
      let m ← NV.PTX.read ptx
      let prog ← NV.PTX.Sem.Prog.ofModule m
      let params ← NV.PTX.paramBlock m "hello" [7, 0x40200000]
      let run ← NV.PTX.interpRun prog { entry := "hello", grid := (1, 1, 1), block := (32, 1, 1),
                                        params, global := Store.empty 0x1000 0x10000 }
      return run.printfText
    let sorted (xs : List String) := xs.mergeSort (· ≤ ·)
    match lines with
    | .error e => bad := bad + 1; IO.println s!"FAIL hello (PTX): {e}"
    | .ok ls =>
      if sorted ls == sorted want then IO.println "ok   hello (PTX): the same 33 lines"
      else bad := bad + 1; IO.println s!"FAIL hello (PTX): {repr ls}"
  IO.println s!"{renders.length + 2} checks, {bad} failing"
  return if bad == 0 then 0 else 1
