import Tests.Kernels

/-!
`lake exe sass-sem`: SASS semantics tests. PTX kernels in `tests/sem/` are compiled by the pinned
`ptxas`, run on the H200 parameters, and every output word is checked against a reference written
here from the PTX ISA's definition of the instructions the kernel uses. Then two hand-assembled
kernels show register writeback timing: with the control bits compiled code would carry they compute
the right values, and with a stall cut short or a wait dropped the consumer reads the register's
old value, as on silicon.
-/

open NV NV.SASS NV.Core Tests

/-! ## References -/

/-- A deterministic word sequence (a 64-bit linear congruential generator, high words). -/
def words32 (seed n : Nat) : Array UInt32 := Id.run do
  let mut x : UInt64 := seed.toUInt64 * 0x9e3779b97f4a7c15 + 1
  let mut out := Array.mkEmpty n
  for _ in [0:n] do
    x := x * 6364136223846793005 + 1442695040888963407
    out := out.push (x >>> 32).toUInt32
  return out

def u32 (n : Int) : Nat := (n % 2 ^ 32).toNat
def s32 (x : Nat) : Int := if x ≥ 2 ^ 31 then (x : Int) - 2 ^ 32 else (x : Int)
def bit (v i : Nat) : Bool := v / 2 ^ i % 2 == 1
def popc (v : Nat) : Nat := (List.range 32).countP (bit v)
def msb (v : Nat) : Option Nat := (List.range 32).reverse.find? (bit v)
def brev (v : Nat) : Nat := (List.range 32).foldl (fun acc i => if bit v i then acc + 2 ^ (31 - i) else acc) 0
def band (a b : Nat) : Nat := (List.range 32).foldl (fun acc i => if bit a i && bit b i then acc + 2 ^ i else acc) 0
def bor (a b : Nat) : Nat := (List.range 32).foldl (fun acc i => if bit a i || bit b i then acc + 2 ^ i else acc) 0
def bxor (a b : Nat) : Nat := (List.range 32).foldl (fun acc i => if bit a i != bit b i then acc + 2 ^ i else acc) 0

/-- PTX `prmt.b32` default mode: byte k is selector nibble k of `c` into the bytes of `b:a`. -/
def prmt (a b c : Nat) : Nat :=
  let x : Nat := a + 2 ^ 32 * b
  (List.range 4).foldl (fun acc k =>
    let nib := c / 16 ^ k % 16
    let byte := x / 256 ^ (nib % 8) % 256
    let byte := if nib ≥ 8 then (if byte ≥ 128 then 255 else 0) else byte
    acc + byte * 256 ^ k) 0

/-- The 32 results of kernel `ints` for inputs `x`, `y` in thread `t`. -/
def intsRef (t x y : Nat) : List Nat :=
  let sx := s32 x; let sy := s32 y
  let bigX : Nat := x + 2 ^ 32 * y; let bigY : Nat := y + 2 ^ 32 * x
  let w64 (v : Int) : Nat × Nat := let n := (v % 2 ^ 64).toNat; (n % 2 ^ 32, n / 2 ^ 32)
  let (w0l, w0h) := w64 (bigX + bigY)
  let (w1l, w1h) := w64 ((bigX : Int) - bigY)
  let (w2l, w2h) := w64 (bigX * bigY)
  let (w3l, w3h) := w64 (sx * sy)
  let s := y % 32
  let p0 := sx < sy
  [x * y % 2 ^ 32, x * y / 2 ^ 32, ((sx * sy) % (2 ^ 64 : Int)).toNat / 2 ^ 32, (x * y + 7) % 2 ^ 32,
   w0l, w0h, w1l, w1h, w2l, w2h,
   x * 2 ^ s % 2 ^ 32, x / 2 ^ s, u32 (Int.fdiv sx (2 ^ s)), (bigX * 2 ^ s) / 2 ^ 32 % 2 ^ 32,
   bor (bor (band x y) (band x t)) (band y t), prmt x y 0x5410, prmt x y y, popc x,
   ((msb x).map (31 - ·)).getD 32, brev x, (msb x).getD (2 ^ 32 - 1),
   u32 (min sx sy), max x y, u32 (min (s32 ((x + y) % 2 ^ 32)) t), u32 sx.natAbs,
   if p0 then x else y, if p0 && x > y then 1 else 0, x / 8 % 512,
   y - y / 2 ^ 5 % 128 * 2 ^ 5 + x % 128 * 2 ^ 5,
   w3l, w3h, if bigX < bigY then 1 else 0]

def edges : List Nat := [0, 1, 0xffffffff, 0x80000000, 0x7fffffff, 2, 0xfffffffe, 0x80000001, 13, 7,
  0x10000, 0xffff, 31, 32, 0x55555555, 0xaaaaaaaa]

/-- Inputs of thread `t`: edge values for the first 16 threads, hashes after. -/
def intsIn (t : Nat) : Nat × Nat :=
  if t < 16 then (edges[t]!, edges[(5 * t + 3) % 16]!)
  else ((t * 2654435761 + 12345) % 2 ^ 32, (t * 40503 * 2654435761 + 777) % 2 ^ 32)

def accBranchy (t : Nat) : Nat :=
  if t % 2 == 1 then ((List.range (t % 8)).foldl (fun a k => a + k * t) 0) % 2 ^ 32
  else bxor (3 * t) (if t / 4 % 2 == 1 then 0x55 else 0)

def mix (a b : Nat) : Nat := (bxor (a * b % 2 ^ 32) a + b) % 2 ^ 32

def v (l : Nat) : Nat := (7 * l + 3) / 4 % 8

/-- The 16 results of kernel `collectives` in lane `l`. -/
def collRef (l : Nat) : List Nat :=
  let lanes := List.range 32
  let ballot (p : Nat → Bool) : Nat := lanes.foldl (fun acc k => if p k then acc + 2 ^ k else acc) 0
  let seg := l / 8 * 8
  [if l ≥ 3 then v (l - 3) else v l, if l + 5 ≤ 31 then v (l + 5) else v l, v (bxor l 6),
   v ((2 ^ 32 + 29 - l) % 32), ballot (v · > 5), if lanes.any (v · == 7) then 1 else 0, 1,
   ballot (v · == v l), (lanes.map v).foldl (· + ·) 0,
   u32 ((lanes.map fun k => (v k : Int) - 5).foldl min 2), (lanes.map v).foldl max 0,
   (lanes.map v).foldl bxor 0,
   if l + 1 ≤ seg + 7 then v (l + 1) else v l, v (seg + 2), if l ≥ seg + 1 then v (l - 1) else v l,
   0]

/-! ## Checks -/

/-- Mismatches between `got` and `want` at the given word offsets, as messages. -/
def compare (what : String) (got : Array UInt32) (want : List (Nat × Nat)) : List String :=
  (want.filterMap fun (i, w) =>
    if got[i]!.toNat == w % 2 ^ 32 then none else some s!"{what}[{i}] = {got[i]!}, expected {w % 2 ^ 32}").take 8

def sentinel : Nat := 0xdeadbeef

def checkInts (ks : Array Kernel) : IO (List String) := do
  let k ← kernel ks "ints"
  let n := 64
  let inp := (List.range n).foldl (fun acc t => let (x, y) := intsIn t; acc.push x.toUInt32 |>.push y.toUInt32) #[]
  match ← run k 1 n [Array.replicate (32 * n) 0, inp] with
  | .error e => return [s!"ints: {e}"]
  | .ok outs =>
    return compare "ints" outs[0]! ((List.range n).flatMap fun t =>
      let (x, y) := intsIn t
      (intsRef t x y).zipIdx.map fun (w, j) => (32 * t + j, w))

def checkCtrl (ks : Array Kernel) : IO (List String) := do
  let n := 64
  let mut out := []
  match ← run (← kernel ks "branchy") 1 n [Array.replicate n 0] with
  | .error e => out := out ++ [s!"branchy: {e}"]
  | .ok bs => out := out ++ compare "branchy" bs[0]! ((List.range n).map fun t =>
      (t, accBranchy t + 1000 * accBranchy (bxor t 1)))
  match ← run (← kernel ks "exits") 1 n [Array.replicate n sentinel.toUInt32] with
  | .error e => out := out ++ [s!"exits: {e}"]
  | .ok bs => out := out ++ compare "exits" bs[0]! ((List.range n).map fun t =>
      (t, if t % 4 == 3 then sentinel else (List.range (t % 16)).foldl (· + ·) 0))
  let calls ← kernel ks "calls"
  unless calls.code.any (·.opcode == "CALL") do out := out ++ ["calls: expected a CALL instruction"]
  match ← run calls 1 n [Array.replicate n 0] with
  | .error e => out := out ++ [s!"calls: {e}"]
  | .ok bs => out := out ++ compare "calls" bs[0]! ((List.range n).map fun t =>
      (t, mix t (t + 3) + mix (t + 1) 5))
  return out

/-- The jump table of `tests/sass/jumps.ptx` (`brx.idx`, `BRX`). -/
def checkJumps : IO (List String) := do
  let k ← kernel (← compile "tests/sass/jumps.ptx") "jumps"
  let n := 64
  match ← run k 1 n [Array.replicate n 0] with
  | .error e => return [s!"jumps: {e}"]
  | .ok bs => return compare "jumps" bs[0]! ((List.range n).map fun t =>
      (t, match t % 4 with | 0 => 10 | 1 => 3 * t | 2 => bxor t 0x5a | _ => t + 1000))

/-- A generic load through a pointer to a thread's local array (`tests/sass/genlocal.ptx`): its
address is the stack pointer over the local window's base in constant bank 0. -/
def checkGenericLocal : IO (List String) := do
  let k ← kernel (← compile "tests/sass/genlocal.ptx") "genlocal"
  let n := 64
  match ← run k 1 n [Array.replicate n 0] with
  | .error e => return [s!"genlocal: {e}"]
  | .ok bs =>
    let ks ← compile "tests/sass/genlocal.ptx"
    let mixed ← match ← run (← kernel ks "genmixed") 1 n [Array.replicate n 0] with
      | .error e => pure [s!"genmixed: {e}"]
      | .ok cs => pure (compare "genmixed" cs[0]! ((List.range n).map fun t =>
          (t, if t % 2 == 1 then 3 * t + 7 else 5 * t + 1)))
    return compare "genlocal" bs[0]! ((List.range n).map fun t => (t, 3 * t + 7)) ++ mixed

/-- The same table on a warp-uniform index (`tests/sass/jumpsu.ptx`): `BRXU`, timed. -/
def checkJumpsU : IO (List String) := do
  let k ← kernel (← compile "tests/sass/jumpsu.ptx") "jumpsu"
  let n := 64
  let mut out := if k.code.any (·.opcode == "BRXU") then [] else ["jumpsu: ptxas emitted no BRXU"]
  for sel in [1, 2] do
    match ← run k 1 n [Array.replicate n 0] [[sel.toUInt8, 0, 0, 0]] with
    | .error e => out := out ++ [s!"jumpsu {sel}: {e}"]
    | .ok bs => out := out ++ compare s!"jumpsu {sel}" bs[0]! ((List.range n).map fun t =>
        (t, if sel == 1 then 3 * t else bxor t 0x5a))
  return out

def checkWarp (ks : Array Kernel) : IO (List String) := do
  let mut out := []
  match ← run (← kernel ks "collectives") 1 32 [Array.replicate (16 * 32) 0] with
  | .error e => out := out ++ [s!"collectives: {e}"]
  | .ok bs => out := out ++ compare "collectives" bs[0]! ((List.range 32).flatMap fun l =>
      (collRef l).zipIdx.map fun (w, j) => (16 * l + j, w))
  match ← run (← kernel ks "halfwarp") 1 32 [Array.replicate (4 * 32) 0] with
  | .error e => out := out ++ [s!"halfwarp: {e}"]
  | .ok bs =>
    let odd := (List.range 32).filter (· % 2 == 1)
    let ballot := odd.foldl (fun acc l => if band (l * l) 6 != 0 then acc + 2 ^ l else acc) 0
    out := out ++ compare "halfwarp" bs[0]! ((List.range 32).flatMap fun l =>
      (if l % 2 == 0 then [0, 0, 0, 0] else [ballot, if l == 1 then 1 else 0, 256, 1]).zipIdx.map
        fun (w, j) => (4 * l + j, w))
  return out

def checkAtomics (ks : Array Kernel) : IO (List String) := do
  let k ← kernel ks "tickets"
  let n := 256
  match ← run k 4 64 [Array.replicate n sentinel.toUInt32, Array.replicate 8 0] with
  | .error e => return [s!"tickets: {e}"]
  | .ok bs =>
    let seen := (bs[0]!.toList.map (·.toNat)).mergeSort (· ≤ ·)
    let maxV := ((List.range n).map fun t => t * 37 % 128).foldl max 0
    let sum := (List.range n).foldl (· + ·) 0
    return (if seen == List.range n then [] else ["tickets: out is not a permutation of the threads"]) ++
      compare "tickets ctr" bs[1]! [(0, n), (1, maxV), (2, n), (4, sum % 2 ^ 32), (5, sum / 2 ^ 32)]

/-! ## Writeback timing -/

def r (i : Nat) : Operand := .reg ⟨.r, i⟩

def ctrl (stall : Nat) (wbar : Option (Fin 6) := none) (wait : Nat := 0) (rbar : Option (Fin 6) := none) : Ctrl :=
  { stall, yield := false, wbar, rbar, wait, reuse := 0 }

def ins (pc : Nat) (opcode : String) (mods : Array String) (ops : Array Operand) (c : Ctrl) : Instr :=
  { pc, opcode, mods, operands := ops, ctrl := c }

def gaddr (i : Nat) : Operand := .mem (some ⟨.r, i⟩) true none (some ⟨.ur, 4⟩) 0

/-- A fused predicate must preserve the election mask when its destination aliases its input. -/
def checkLop3Pred : IO (List String) := do
  let mut failures := []
  let env : Env :=
    { ctx := { ntid := (32, 1, 1), nctaid := (1, 1, 1), ctaid := (0, 0, 0), warp := 0, threads := 32 },
      const0 := Store.empty 0 0 }
  for mode in ["PAND", "POR"] do
    for value in [0, 1] do
      for negate in [false, true] do
        let instr := ins 0 "LOP3" #["LUT", mode]
          #[.reg ⟨.p, 0⟩, r 2, .imm value, .reg Reg.RZ, .reg Reg.RZ, .imm 0xf0,
            .reg ⟨.p, 0⟩ { not := negate }] (ctrl 4)
        let kernel : Kernel := { name := "lop3-pred", regs := 3, code := #[instr] }
        let warp := (Warp.init 3 32).setP 0 0 true
        match (decodeGeneral kernel 0).run env warp fullMask with
        | .reg s wb =>
          let result := wb s
          for lane in [0:32] do
            let input := (lane == 0) != negate
            let want := if mode == "PAND" then value != 0 && input else value != 0 || input
            if result.getP 0 lane != want || result.getR 2 lane != value.toUInt32 then
              failures := failures ++ [s!"LOP3 {mode} value{value} negate{negate} lane{lane}"]
        | _ => failures := failures ++ ["LOP3 predicate: expected register effect"]
  return failures

/-- LEA's carry out with a negated source (`LEA R2, P0, -R1, R3, 0x3`): `-a` is negated after the
shift, so `-0` carries as `IADD3`'s does; Triton's 64-bit bounds take their high words from it. -/
def checkLeaNeg : IO (List String) := do
  let env : Env :=
    { ctx := { ntid := (32, 1, 1), nctaid := (1, 1, 1), ctaid := (0, 0, 0), warp := 0, threads := 32 },
      const0 := Store.empty 0 0 }
  let mut failures := []
  for (a, b, sum, carry) in [(0, 0x3c, 0x3c, true), (1, 0x3c, 0x34, true), (8, 0x3c, 0xfffffffc, false),
      (0x20000000, 5, 5, true)] do
    let instr := ins 0 "LEA" #[] #[r 2, .reg ⟨.p, 0⟩, .reg ⟨.r, 1⟩ { neg := true }, r 3, .imm 3] (ctrl 4)
    let warp := ((Warp.init 4 32).setR 1 0 a.toUInt32).setR 3 0 b.toUInt32
    match (decodeGeneral { name := "lea-neg", regs := 4, code := #[instr] } 0).run env warp fullMask with
    | .reg s wb =>
      let w := wb s
      if w.getR 2 0 != sum.toUInt32 || w.getP 0 0 != carry then
        failures := failures ++ [s!"LEA -{a} << 3 + {b}: {w.getR 2 0} carry {w.getP 0 0}, expected {sum} carry {carry}"]
    | _ => failures := failures ++ ["LEA: expected a register effect"]
  return failures

/-- `out[t] = 3t + 5 + 1`, where the add consumes the IMAD after `stall` cycles (IMAD's latency
is 4, and no counter guards it). -/
def fixedLatency (stall : Nat) : Kernel :=
  { name := "fixed", regs := 8, params := #[(0, 8)], paramBytes := 8, code := #[
    ins 0x00 "S2R" #[] #[r 0, .name "SR_TID.X"] (ctrl 1 (some 0)),
    ins 0x10 "LDC" #["64"] #[r 4, .cbank 0 none 0x210] (ctrl 1 (some 1)),
    ins 0x20 "IMAD" #[] #[r 2, r 0, .imm 3, .imm 5] (ctrl stall none 0b01),
    ins 0x30 "IADD3" #[] #[r 3, r 2, .imm 1, .reg Reg.RZ] (ctrl 4),
    ins 0x40 "IMAD" #["WIDE", "U32"] #[r 4, r 0, .imm 4, r 4] (ctrl 4 none 0b10),
    ins 0x50 "STG" #["E"] #[gaddr 4, r 3] (ctrl 1),
    ins 0x60 "EXIT" #[] #[] (ctrl 5)] }

/-- `out[t] = in[t] + 1`, where the add waits on the load's counter only if `wait`. -/
def variableLatency (wait : Bool) : Kernel :=
  { name := "variable", regs := 8, params := #[(0, 8), (8, 8)], paramBytes := 16, code := #[
    ins 0x00 "S2R" #[] #[r 0, .name "SR_TID.X"] (ctrl 1 (some 0)),
    ins 0x10 "LDC" #["64"] #[r 4, .cbank 0 none 0x218] (ctrl 1 (some 1)),
    ins 0x20 "LDC" #["64"] #[r 6, .cbank 0 none 0x210] (ctrl 1 (some 2)),
    ins 0x30 "IMAD" #["WIDE", "U32"] #[r 4, r 0, .imm 4, r 4] (ctrl 4 none 0b11),
    ins 0x40 "LDG" #["E"] #[r 2, gaddr 4] (ctrl 1 (some 3)),
    ins 0x50 "IADD3" #[] #[r 3, r 2, .imm 1, .reg Reg.RZ] (ctrl 4 none (if wait then 0b1000 else 0)),
    ins 0x60 "IMAD" #["WIDE", "U32"] #[r 6, r 0, .imm 4, r 6] (ctrl 4 none 0b100),
    ins 0x70 "STG" #["E"] #[gaddr 6, r 3] (ctrl 1),
    ins 0x80 "EXIT" #[] #[] (ctrl 5)] }

/-- `out[t] = t`, stored through `[R4.64+UR6]` with UR6 = 0, which the next instruction sets to
0x100 without waiting: the store's address uses UR6 as at issue. -/
def uniformAtIssue : Kernel :=
  { name := "uniform", regs := 8, params := #[(0, 8)], paramBytes := 8, code := #[
    ins 0x00 "S2R" #[] #[r 0, .name "SR_TID.X"] (ctrl 1 (some 0)),
    ins 0x10 "LDC" #["64"] #[r 4, .cbank 0 none 0x210] (ctrl 1 (some 1)),
    ins 0x20 "UMOV" #[] #[.reg ⟨.ur, 6⟩, .imm 0] (ctrl 2),
    ins 0x30 "IMAD" #["WIDE", "U32"] #[r 4, r 0, .imm 4, r 4] (ctrl 4 none 0b11),
    ins 0x40 "STG" #["E"] #[.mem (some ⟨.r, 4⟩) true (some ⟨.ur, 6⟩) none 0, r 0] (ctrl 1),
    ins 0x50 "UMOV" #[] #[.reg ⟨.ur, 6⟩, .imm 0x100] (ctrl 1),
    ins 0x60 "EXIT" #[] #[] (ctrl 5)] }

/-- `out[t] = 0xffffff03`: `UP2UR UR5, UPR, UR6, 0x7` with UP0 and UP1 set (UP1 the second, negated
result of a comparison that fails), UP2 clear and UR6
`0xffffff00` (sm_100a). -/
def uniformPredicates : Kernel :=
  let ur (i : Nat) : Operand := .reg ⟨.ur, i⟩
  let up (i : Nat) : Operand := .reg ⟨.up, i⟩
  { name := "up2ur", sm := 100, regs := 8, params := #[(0, 8)], paramBytes := 8, code := #[
    ins 0x00 "S2R" #[] #[r 0, .name "SR_TID.X"] (ctrl 1 (some 0)),
    ins 0x10 "LDC" #["64"] #[r 4, .cbank 0 none 0x210] (ctrl 1 (some 1)),
    ins 0x20 "UISETP" #["EQ", "U32", "AND"] #[up 0, up 7, ur 255, ur 255, up 7] (ctrl 1),
    ins 0x30 "UISETP" #["NE", "U32", "AND"] #[up 2, up 1, ur 255, ur 255, up 7] (ctrl 1),
    ins 0x40 "UMOV" #[] #[ur 6, .imm 0xffffff00] (ctrl 1),
    ins 0x50 "UP2UR" #[] #[ur 5, .name "UPR", ur 6, .imm 0x7] (ctrl 4),
    ins 0x60 "MOV" #[] #[r 3, ur 5] (ctrl 4),
    ins 0x70 "IMAD" #["WIDE", "U32"] #[r 4, r 0, .imm 4, r 4] (ctrl 4 none 0b11),
    ins 0x80 "STG" #["E"] #[gaddr 4, r 3] (ctrl 1),
    ins 0x90 "EXIT" #[] #[] (ctrl 5)] }

/-- `out[t] = 7 + 4 · 256`: the most significant set bit (UFLO) and the population count (UPOPC)
of UR6 = 0xf0, each read by a MOV 4 cycles on with no counter between, on `sm`. -/
def uniformCounts (sm : Nat) : Kernel :=
  let ur (i : Nat) : Operand := .reg ⟨.ur, i⟩
  { name := "ucounts", sm, regs := 8, params := #[(0, 8)], paramBytes := 8, code := #[
    ins 0x00 "S2R" #[] #[r 0, .name "SR_TID.X"] (ctrl 1 (some 0)),
    ins 0x10 "LDC" #["64"] #[r 4, .cbank 0 none 0x210] (ctrl 1 (some 1)),
    ins 0x20 "UMOV" #[] #[ur 6, .imm 0xf0] (ctrl 2),
    ins 0x30 "UFLO" #["U32"] #[ur 5, ur 6] (ctrl 1),
    ins 0x40 "UPOPC" #[] #[ur 7, ur 6] (ctrl 3),
    ins 0x50 "MOV" #[] #[r 3, ur 5] (ctrl 1),
    ins 0x60 "MOV" #[] #[r 2, ur 7] (ctrl 4),
    ins 0x70 "LEA" #[] #[r 3, r 2, r 3, .imm 8] (ctrl 4),
    ins 0x80 "IMAD" #["WIDE", "U32"] #[r 4, r 0, .imm 4, r 4] (ctrl 4 none 0b11),
    ins 0x90 "STG" #["E"] #[gaddr 4, r 3] (ctrl 1),
    ins 0xa0 "EXIT" #[] #[] (ctrl 5)] }

/-- `out[t] = 100 + t`: lane `t` stores `100 + t` to shared memory at `128 t` (every lane in one
bank: a 32-way conflict, a slow load), loads it back with an `LDS.128` no counter covers, then
shuffles (`SHFL.BFLY` by 0, its own value) on counter 2, and once an add has waited on that counter
reads the loaded register with no wait: the shuffle returns through the MIO after the earlier load,
so its counter covers it (`Slot.mioHeld`). -/
def shuffleBehindLoad : Kernel :=
  let sh (i : Nat) : Operand := .mem (some ⟨.r, i⟩) false none none 0
  { name := "shflbehind", regs := 16, sharedBytes := 4096, params := #[(0, 8)], paramBytes := 8, code := #[
    ins 0x00 "S2R" #[] #[r 0, .name "SR_TID.X"] (ctrl 1 (some 0)),
    ins 0x10 "LDC" #["64"] #[r 4, .cbank 0 none 0x210] (ctrl 1 (some 1)),
    ins 0x20 "IADD3" #[] #[r 6, r 0, .imm 100, .reg Reg.RZ] (ctrl 1 none 0b01),
    ins 0x30 "IMAD" #[] #[r 7, r 0, .imm 128, .reg Reg.RZ] (ctrl 4),
    ins 0x40 "STS" #[] #[sh 7, r 6] (ctrl 1),
    ins 0x50 "LDS" #["128"] #[r 8, sh 7] (ctrl 1),
    ins 0x60 "SHFL" #["BFLY"] #[.reg ⟨.p, 7⟩, r 12, r 0, .imm 0, .imm 0x1f] (ctrl 1 (some 2)),
    ins 0x70 "IADD3" #[] #[r 13, r 12, .imm 0, .reg Reg.RZ] (ctrl 1 none 0b100),
    ins 0x80 "IADD3" #[] #[r 3, r 8, .imm 0, .reg Reg.RZ] (ctrl 4),
    ins 0x90 "IMAD" #["WIDE", "U32"] #[r 4, r 0, .imm 4, r 4] (ctrl 4 none 0b10),
    ins 0xa0 "STG" #["E"] #[gaddr 4, r 3] (ctrl 1),
    ins 0xb0 "EXIT" #[] #[] (ctrl 5)] }

/-- `out[t] = 100 + t`: lane `t` stores `100 + t` to shared memory at `128 t` with an `STS` that
reads its address on a read counter, and the next instruction, a `SHFL`, overwrites the address
register without waiting on it: the shuffle leaves the MIO after the store, so the store reads its
old value; then the word is loaded back from a copy of the address and stored out. No hazard. -/
def shuffleAfterStore : Kernel :=
  let sh (i : Nat) : Operand := .mem (some ⟨.r, i⟩) false none none 0
  { name := "shflafterst", regs := 16, sharedBytes := 4096, params := #[(0, 8)], paramBytes := 8, code := #[
    ins 0x00 "S2R" #[] #[r 0, .name "SR_TID.X"] (ctrl 1 (some 0)),
    ins 0x10 "LDC" #["64"] #[r 4, .cbank 0 none 0x210] (ctrl 1 (some 1)),
    ins 0x20 "IADD3" #[] #[r 6, r 0, .imm 100, .reg Reg.RZ] (ctrl 1 none 0b01),
    ins 0x30 "IMAD" #[] #[r 7, r 0, .imm 128, .reg Reg.RZ] (ctrl 1),
    ins 0x40 "IMAD" #[] #[r 9, r 0, .imm 128, .reg Reg.RZ] (ctrl 4),
    ins 0x50 "STS" #[] #[sh 7, r 6] (ctrl 1 none 0 (some 3)),
    ins 0x60 "SHFL" #["BFLY"] #[.reg ⟨.p, 7⟩, r 7, r 0, .imm 0, .imm 0x1f] (ctrl 1 (some 2)),
    ins 0x70 "LDS" #[] #[r 8, sh 9] (ctrl 1 (some 4) 0b1000),
    ins 0x80 "IADD3" #[] #[r 3, r 8, .imm 0, .reg Reg.RZ] (ctrl 4 none 0b10100),
    ins 0x90 "IMAD" #["WIDE", "U32"] #[r 4, r 0, .imm 4, r 4] (ctrl 4 none 0b10),
    ins 0xa0 "STG" #["E"] #[gaddr 4, r 3] (ctrl 1),
    ins 0xb0 "EXIT" #[] #[] (ctrl 5)] }

/-- `out[t] = 100 + t` on sm_100: an LEA writes R3, and an IMAD.MOV one cycle later overwrites it
with `100 + t`; the B200's IMAD lands one cycle sooner than its LEA, so both land in one cycle, in
issue order (a pattern compiled sm_100 code has). -/
def sameCycleWrites : Kernel :=
  { name := "samecycle", sm := 100, regs := 8, params := #[(0, 8)], paramBytes := 8, code := #[
    ins 0x00 "S2R" #[] #[r 0, .name "SR_TID.X"] (ctrl 1 (some 0)),
    ins 0x10 "LDC" #["64"] #[r 4, .cbank 0 none 0x210] (ctrl 1 (some 1)),
    ins 0x20 "IADD3" #[] #[r 6, r 0, .imm 100, .reg Reg.RZ] (ctrl 4 none 0b01),
    ins 0x30 "LEA" #[] #[r 3, r 0, r 0, .imm 1] (ctrl 1),
    ins 0x40 "IMAD" #["MOV"] #[r 3, .reg Reg.RZ, .reg Reg.RZ, r 6] (ctrl 4),
    ins 0x50 "IMAD" #["WIDE", "U32"] #[r 4, r 0, .imm 4, r 4] (ctrl 4 none 0b10),
    ins 0x60 "STG" #["E"] #[gaddr 4, r 3] (ctrl 1),
    ins 0x70 "EXIT" #[] #[] (ctrl 5)] }

/-- `out[0] = 100`, every other thread exited first: `@P0 EXIT P1` (P0 holding everywhere, P1 for
threads past the first) leaves thread 0 to store, as ptxas's reduction epilogues do. -/
def exitPred : Kernel :=
  let pr (i : Nat) : Operand := .reg ⟨.p, i⟩
  { name := "exitpred", sm := 90, regs := 8, params := #[(0, 8)], paramBytes := 8, code := #[
    ins 0x00 "S2R" #[] #[r 0, .name "SR_TID.X"] (ctrl 1 (some 0)),
    ins 0x10 "LDC" #["64"] #[r 4, .cbank 0 none 0x210] (ctrl 1 (some 1)),
    ins 0x20 "ISETP" #["NE", "AND"] #[pr 1, .reg Reg.PT, r 0, .reg Reg.RZ, .reg Reg.PT] (ctrl 4 none 0b01),
    ins 0x30 "ISETP" #["EQ", "AND"] #[pr 0, .reg Reg.PT, .reg Reg.RZ, .reg Reg.RZ, .reg Reg.PT] (ctrl 4),
    { (ins 0x40 "EXIT" #[] #[pr 1] (ctrl 5 none 0b10)) with guard := some (⟨.p, 0⟩, false) },
    ins 0x50 "IMAD" #["WIDE", "U32"] #[r 4, r 0, .imm 4, r 4] (ctrl 4),
    ins 0x60 "IADD3" #[] #[r 6, .reg Reg.RZ, .imm 100, .reg Reg.RZ] (ctrl 4),
    ins 0x70 "STG" #["E"] #[gaddr 4, r 6] (ctrl 1),
    ins 0x80 "EXIT" #[] #[] (ctrl 5)] }

/-- `out[t] = t` after `NANOSLEEP ns`: the warp sleeps about `ns` ns before its store. -/
def sleeper (ns : UInt64) : Kernel :=
  { name := "sleeper", sm := 90, regs := 8, params := #[(0, 8)], paramBytes := 8, code := #[
    ins 0x00 "S2R" #[] #[r 0, .name "SR_TID.X"] (ctrl 1 (some 0)),
    ins 0x10 "LDC" #["64"] #[r 4, .cbank 0 none 0x210] (ctrl 1 (some 1)),
    ins 0x20 "NANOSLEEP" #[] #[.imm ns] (ctrl 1),
    ins 0x30 "IMAD" #["WIDE", "U32"] #[r 4, r 0, .imm 4, r 4] (ctrl 4 none 0b11),
    ins 0x40 "STG" #["E"] #[gaddr 4, r 0] (ctrl 1),
    ins 0x50 "EXIT" #[] #[] (ctrl 5)] }

def checkTiming : IO (List String) := do
  let n := 32
  let mut out := []
  for (stall, want) in ([(4, fun t => 3 * t + 6), (1, fun _ => 1)] : List (Nat × (Nat → Nat))) do
    match ← run (fixedLatency stall) 1 n [Array.replicate n 0] with
    | .error e => out := out ++ [s!"fixed latency, stall {stall}: {e}"]
    | .ok bs =>
      out := out ++ compare s!"fixed latency, stall {stall}" bs[0]! ((List.range n).map fun t => (t, want t))
  match ← run uniformAtIssue 1 n [Array.replicate n 0] with
  | .error e => out := out ++ [s!"uniform source at issue: {e}"]
  | .ok bs =>
    out := out ++ compare "uniform source at issue" bs[0]! ((List.range n).map fun t => (t, t))
  match ← run uniformPredicates 1 n [Array.replicate n 0] with
  | .error e => out := out ++ [s!"UP2UR: {e}"]
  | .ok bs => out := out ++ compare "UP2UR" bs[0]! ((List.range n).map fun t => (t, 0xffffff03))
  for sm in [90, 100] do
    match ← run (uniformCounts sm) 1 n [Array.replicate n 0] with
    | .error e => out := out ++ [s!"UFLO, UPOPC sm_{sm}: {e}"]
    | .ok bs => out := out ++ compare s!"UFLO, UPOPC sm_{sm}" bs[0]! ((List.range n).map fun t => (t, 7 + 4 * 256))
  match ← runWith shuffleAfterStore 1 n [Array.replicate n 0] with
  | .error e => out := out ++ [s!"shuffle after a store: {e}"]
  | .ok (bs, c) =>
    out := out ++ compare "shuffle after a store" bs[0]! ((List.range n).map fun t => (t, 100 + t))
    if c.hazards != 0 then out := out ++ [s!"shuffle after a store: {c.hazards} hazards"]
  match ← runWith (sleeper 0) 1 n [Array.replicate n 0], ← runWith (sleeper 1000) 1 n [Array.replicate n 0] with
  | .ok (_, c0), .ok (bs, c) =>
    out := out ++ compare "NANOSLEEP" bs[0]! ((List.range n).map fun t => (t, t))
    -- About 1000 ns at the part's clock (the work the 0 ns run waits for overlapping the
    -- sleep), against the same kernel sleeping 0 ns.
    let want := 1000 * NV.Arch.h200.clockMHz.value / 1000 * 95 / 100
    if c.cycles < c0.cycles + want then
      out := out ++ [s!"NANOSLEEP 1000 ns: {c.cycles} cycles against {c0.cycles} for 0 ns, under {want} more"]
  | .error e, _ | _, .error e => out := out ++ [s!"NANOSLEEP: {e}"]
  match ← run exitPred 1 n [Array.replicate n 0] with
  | .error e => out := out ++ [s!"EXIT with a predicate operand: {e}"]
  | .ok bs =>
    let want := (List.range n).map fun t => (t, if t == 0 then 100 else 0)
    out := out ++ compare "EXIT with a predicate operand" bs[0]! want
  match ← runWith sameCycleWrites 1 n [Array.replicate n 0] with
  | .error e => out := out ++ [s!"writes landing in one cycle: {e}"]
  | .ok (bs, c) =>
    out := out ++ compare "writes landing in one cycle" bs[0]! ((List.range n).map fun t => (t, 100 + t))
    if c.hazards != 0 then out := out ++ [s!"writes landing in one cycle: {c.hazards} hazards"]
  match ← run shuffleBehindLoad 1 n [Array.replicate n 0] with
  | .error e => out := out ++ [s!"shuffle behind a shared load: {e}"]
  | .ok bs => out := out ++ compare "shuffle behind a shared load" bs[0]! ((List.range n).map fun t => (t, 100 + t))
  let inp := (List.range n).toArray.map fun t => (100 + t).toUInt32
  for (wait, want) in ([(true, fun t => 101 + t), (false, fun _ => 1)] : List (Bool × (Nat → Nat))) do
    match ← run (variableLatency wait) 1 n [Array.replicate n 0, inp] with
    | .error e => out := out ++ [s!"variable latency, wait {wait}: {e}"]
    | .ok bs =>
      out := out ++ compare s!"variable latency, wait {wait}" bs[0]! ((List.range n).map fun t => (t, want t))
  return out

/-! ## Compiled forms against the general definition

`NV.SASS.Fast` compiles the hot instruction forms; each must mean what the general definition
means. For every instruction of the reader's listings (`tests/sass/`) that has a compiled form,
both run from random warp states (registers, predicates, lanes at the instruction, guards) and
their effects are compared: the state after issue and after writeback, and for memory the
request and the state after a random response. -/

def warpOf (seed : Nat) (pc : Nat) : Warp :=
  -- Eight registers past R254: a footprint's span may reach past the last register named.
  let n := 263
  let ws := words32 seed (n * 32 + 255 + 7 + 7)
  { (Warp.init n 32) with
    r := (ws.extract 0 (n * 32)), ur := ws.extract (n * 32) (n * 32 + 255),
    p := ws.extract (n * 32 + 255) (n * 32 + 262),
    up := (ws.extract (n * 32 + 262) (n * 32 + 269)).map (· % 2 == 1),
    conv := some pc }

def sameWarp (a b : Warp) : Bool :=
  a.r == b.r && a.ur == b.ur && a.p == b.p && a.up == b.up && a.live == b.live &&
  a.b == b.b && a.waiting == b.waiting && (List.range 32).all (fun l => a.pc l == b.pc l)

def sameReq (a b : Req) : Bool :=
  a.space == b.space && a.op == b.op && a.bytes == b.bytes && a.sem == b.sem &&
  a.scope == b.scope && a.addrs == b.addrs && a.data == b.data

/-- Whether two effects agree; `full` gives the warp an issued state continues as (the
semantics continue from the control state alone, `Warp.blank`), which writebacks apply to. -/
def compareEffects (env : Env) (seed : Nat) (full : Warp → Warp) (fast general : Effect Warp) :
    Option String :=
  match fast, general with
  | .reg s wb, .reg s' wb' =>
    if !sameWarp s s' then some "after issue"
    else if !sameWarp (wb (full s)) (wb' (full s')) then some "after writeback"
    else none
  | .mem r s wb, .mem r' s' wb' =>
    if !sameWarp s s' then some "after issue" else
    if !sameReq (r (full s)) (r' (full s')) then some "request" else
    let resp := words32 (seed + 1) (32 * words (r (full s)).bytes)
    if !sameWarp (wb resp (full s)) (wb' resp (full s')) then some "after the response" else none
  | .next s, .next s' => if sameWarp s s' then none else some "next"
  | .fault _, .fault _ => none
  | _, _ => some "different effect kinds"

def checkCompiled : IO (List String) := do
  let mut out := []
  let mut count := 0
  let a ← arch
  let dir := if a == "sm_90a" then "tests/sass" else s!"tests/sass/{a}"
  for f in ["basic", "misc", "wgmma", "tma", "elect", "ops", "jumps"] do
    -- A fixture a target cannot compile is absent from its directory (`gen.sh`'s SKIPPED).
    unless ← System.FilePath.pathExists s!"{dir}/{f}.sass" do continue
    let ks ← IO.ofExcept (Read.kernels (← IO.FS.readFile s!"{dir}/{f}.sass")
      (← IO.FS.readFile s!"{dir}/{f}.elf"))
    for k in ks do
      for pc in [0:k.code.size] do
        let fp := footprint k.code[pc]!
        let some fast := Fast.decodeFast fp.reads k.code[pc]! pc | continue
        count := count + 1
        let general := (decodeGeneral k pc).out
        for t in [0:4] do
          let seed := 7919 * pc + 104729 * t + k.name.length
          let w := warpOf seed pc
          let ms := words32 (seed + 2) 1
          let m := if t == 0 then fullMask else ms[0]!
          let const0 := (List.range 256).foldl (fun st j => st.writeWord (4 * j) (words32 (seed + 3) 256)[j]!)
            (Store.empty 0 1024)
          let env : Env := { ctx := { ntid := (128, 1, 1), nctaid := (4, 1, 1), ctaid := (1, 0, 0),
                                      warp := 2, threads := 128 }, const0 }
          let v := View.of fp.reads w
          if let some why := compareEffects env seed (fp.issue w) ((fast env v w.blank m).plain fp.reads w)
              ((general env v w.blank m).plain fp.reads w) then
            out := out ++ [s!"{f} {k.name} {k.code[pc]!.pc} {k.code[pc]!.full}: {why}"]
  IO.println s!"compiled forms: {count} instructions compared"
  return out.take 20

/-- `lake exe sass-sem [suite…]`: every suite, or the named ones. -/
def main (args : List String) : IO UInt32 := do
  let mut failures : List String := []
  let fs ← if args.isEmpty || args.contains "lop3-pred" then checkLop3Pred else pure []
  let fs := fs ++ (← if args.isEmpty || args.contains "lea-neg" then checkLeaNeg else pure [])
  IO.println s!"LOP3 predicates, LEA carries: {if fs.isEmpty then "ok" else s!"{fs.length} failures"}"
  failures := failures ++ fs
  let suites := [("ints", checkInts), ("ctrl", checkCtrl), ("warp", checkWarp),
    ("atomics", checkAtomics)]
  for (file, check) in suites.filter fun (n, _) => args.isEmpty || args.contains n do
    let fs ← try
        let ks ← compile s!"tests/sem/{file}.ptx"
        pure (ks.toList.flatMap (fun k => latencyProblems ((NV.Arch.forSm k.sm).getD NV.Arch.h200) k) ++
          (← check ks))
      catch e =>
        match unsupported e with
        | some why => do IO.println s!"{file}: skipped on {← arch}: {why}"; continue
        | none => pure [s!"{file}: {e}"]
    IO.println s!"{file}: {if fs.isEmpty then "ok" else s!"{fs.length} failures"}"
    failures := failures ++ fs
  let fs ← if args.isEmpty || args.contains "compiled" then checkCompiled else pure []
  IO.println s!"compiled forms: {if fs.isEmpty then "ok" else s!"{fs.length} failures"}"
  failures := failures ++ fs
  let fs ← if args.isEmpty || args.contains "jumps" then (do pure ((← checkJumps) ++ (← checkJumpsU) ++ (← checkGenericLocal))) else pure []
  IO.println s!"jumps: {if fs.isEmpty then "ok" else s!"{fs.length} failures"}"
  failures := failures ++ fs
  let fs ← if args.isEmpty || args.contains "timing" then checkTiming else pure []
  IO.println s!"writeback timing: {if fs.isEmpty then "ok" else s!"{fs.length} failures"}"
  failures := failures ++ fs
  for f in failures.take 60 do IO.eprintln f
  return if failures.isEmpty then 0 else 1
