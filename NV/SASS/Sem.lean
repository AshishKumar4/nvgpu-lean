import NV.SASS.Async
import NV.SASS.Float
import NV.SASS.Fast
import NV.SASS.Footprint
import NV.PTX.Sem.Matrix
import NV.PTX.Sem.Int
import NV.Machine.Printf

/-!
# SASS semantics (`sm_90`)

`decode` resolves each instruction of a kernel once, opcode, modifiers and operand forms, into a
`Run`: the function from the warp's environment, its state and the lanes at the instruction to the
instruction's `Effect`. `exec` is decode then run. The semantics of an opcode is
NVIDIA-undocumented; each definition here states the form it covers, and the PTX↔SASS
differential tests check it against the PTX ISA's semantics. Forms not covered return `Effect.fault`, never a guess.

Register results land at writeback, not at issue: an instruction that writes registers returns
`Effect.reg` with the state after issue (its sources read, its lanes advanced) and the writes,
which the timing core applies when the result is ready (the opcode's latency, or the variable
latency that releases its write counter). The hardware does not check dependences on
fixed-latency results (Huerta et al. 2025 §4), so code whose control bits are too optimistic
reads the old register values here as on silicon. Memory loads write back the same way
(`Effect.mem`).

Divergence follows the convergence-barrier model of `sm_70` and later: `BSSY Bx` records the
lanes that will meet at the matching `BSYNC Bx`; a lane reaching `BSYNC` (or `WARPSYNC`) waits
until every live lane it waits for has arrived, and the warp meanwhile runs its other lanes
(`Warp.current`).
-/

namespace NV.SASS

open NV Exec

/-- One instruction being executed: what its semantics reads. -/
structure Issue where
  k : Kernel
  env : Env
  /-- What the instruction reads of the warp (its footprint's reads). -/
  v : View
  /-- The warp it continues from: the control state (`Warp.blank`). -/
  w : Warp
  /-- The instruction's index in the code. -/
  pc : Nat
  /-- The lanes at this instruction. -/
  m : UInt32
  i : Instr
  /-- The lanes of `m` whose guard holds. -/
  on : UInt32

namespace Issue

def o (x : Issue) (n : Nat) : Operand := x.i.operands[n]?.getD (.name "?")

/-- The warp with the lanes at this instruction moved past it. -/
def fin (x : Issue) (w : Warp) : Warp := w.advance x.m x.pc

def bad (x : Issue) : String := s!"{x.i.full} at {x.i.pc}: unsupported form"

/-- The executing lanes: the guarded lanes, or for a uniform-datapath instruction the first of
them (its operands are warp-wide). -/
def ex (x : Issue) (uniform : Bool) : UInt32 :=
  if uniform then (match lanes x.on with | l :: _ => (1 : UInt32) <<< l.toUInt32 | [] => 0) else x.on

/-- A register-only result: per executing lane, the writes computed now, applied at writeback. -/
def result (x : Issue) (uniform : Bool) (f : Nat → Option (List Write)) : Out := Id.run do
  let on := x.ex uniform
  let mut ws : Array Write := #[]
  for l in [0:32] do
    if (on >>> l.toUInt32) &&& 1 == 1 then
      match f l with
      | some w => for v in w do ws := ws.push v
      | none => return .fault x.bad
  return .reg (x.fin x.w) ws.toList

def s (x : Issue) (l n : Nat) : Option UInt32 := src x.env x.v l (x.o n)
def sN (x : Issue) (l n : Nat) : Option Nat := srcN x.env x.v l (x.o n)
def p (x : Issue) (l n : Nat) : Option Bool := pred x.v l (x.o n)

/-- The code index of a branch target. -/
def target (x : Issue) (t : Nat) : Option Nat := x.k.code.findIdx? (·.pc == t)

end Issue

/-- An instruction being decoded. -/
structure Dec where
  k : Kernel
  pc : Nat
  i : Instr
  /-- A uniform-datapath form (`U…`) of a vector opcode. -/
  uniform : Bool := false

namespace Dec

def o (d : Dec) (n : Nat) : Operand := d.i.operands[n]?.getD (.name "?")

def bad (d : Dec) : String := s!"{d.i.full} at {d.i.pc}: unsupported form"

def issue (d : Dec) (env : Env) (v : View) (w : Warp) (m : UInt32) : Issue :=
  { k := d.k, env, v, w, pc := d.pc, m, i := d.i, on := guarded v m d.i }

/-- The semantics, given the instruction in execution. -/
def run (d : Dec) (f : Issue → Out) : Run := fun env v w m => f (d.issue env v w m)

/-- `run` for an instruction the whole warp must execute (a warp-level matrix operation): a fault
otherwise. -/
def runWarp (d : Dec) (f : Issue → Out) : Run :=
  d.run fun x => if x.on != fullMask then .fault s!"{d.i.full} at {d.i.pc}: not executed by the whole warp" else f x

/-- A register-only result computed per executing lane. -/
def res (d : Dec) (uniform : Bool) (f : Issue → Nat → Option (List Write)) : Run :=
  d.run fun x => x.result uniform (f x)

def fault (d : Dec) : Run := fun _ _ _ _ => .fault d.bad

end Dec

/-- `ISETP.cmp.EX`: the high-word step of a 64-bit comparison, `low` the low words' result. -/
def cmpEx (op : String) (signed : Bool) (a b : UInt32) (low : Bool) : Option Bool := do
  let lt ← cmp signed "LT" a b
  let eq ← cmp signed "EQ" a b
  let gt ← cmp signed "GT" a b
  match op with
  | "LT" | "LE" => some (lt || eq && low)
  | "GT" | "GE" => some (gt || eq && low)
  | "EQ" => some (eq && low)
  | "NE" => some (!eq || low)
  | _ => none

/-- `PRMT` in its default mode: byte `k` of the result is byte `sel[4k+2:4k]` of `b:a`, replicated
from its sign bit when `sel[4k+3]` is set (PTX ISA `prmt`). -/
def prmt (a b sel : UInt32) : UInt32 := Id.run do
  let x := toU64 a b
  let mut r : UInt32 := 0
  for k in [0:4] do
    let s := (sel >>> (4 * k).toUInt32) &&& 0xf
    let byte := ((x >>> (8 * (s &&& 7)).toUInt64) &&& 0xff).toUInt32
    let byte := if s &&& 8 != 0 then (if byte &&& 0x80 != 0 then 0xff else 0) else byte
    r := r ||| (byte <<< (8 * k).toUInt32)
  return r

/-- Bits of `v` most significant first: the index of the highest set bit. -/
def msb? (v : UInt32) : Option Nat := (List.range 32).reverse.find? fun b => (v >>> b.toUInt32) &&& 1 == 1

/-- Integer, logic, predicate and move instructions on the vector and uniform datapaths. -/
def decodeInt (d : Dec) : Option Run :=
  let i := d.i
  -- Uniform-datapath forms share their vector form's semantics.
  let (u, op) := if i.opcode.startsWith "U" then (true, (i.opcode.drop 1).toString) else (false, i.opcode)
  let res := d.res u
  let nP (from_ : Nat) := ((i.operands.toList.drop from_).takeWhile isPred).length
  match op with
  | "MOV" =>
    if i.operands.size > 2 && d.o 2 != .imm 0xf then some d.fault else
    -- `MOV.64 Rd, a` (sm_100 on): a register pair, or a 64-bit immediate as written.
    if i.hasMod "64" then
      some (res fun x l => do
        wr64 (x.o 0) l (← match x.o 1 with | .imm b => some b | o => src64 x.env x.v l o))
    else
    some (res fun x l => do wr (x.o 0) l (← x.s l 1))
  | "IMAD" =>
    let signed := !i.hasMod "U32"
    if i.hasMod "WIDE" then
      -- `IMAD.WIDE[.U32][.X] Rd, [Pc,] a, b, c[, Pin]`: the 64-bit a·b + c (+ carry in) to the pair
      -- at Rd, with the carry out of bit 63.
      let n := nP 1
      let carryIn := i.hasMod "X"
      some (res fun x l => do
        let a ← x.s l (1 + n); let b ← x.s l (2 + n)
        let c := toU64 (← x.s l (3 + n)) (← srcHi x.env x.v l (x.o (3 + n)))
        let p : UInt64 := if signed then (a.toInt32.toInt64 * b.toInt32.toInt64).toUInt64
          else a.toUInt64 * b.toUInt64
        let cin := if carryIn && (x.p l (4 + n)).getD false then 1 else 0
        let t := p.toNat + c.toNat + cin
        let ws ← wr64 (x.o 0) l t.toUInt64
        if n ≥ 1 then (wrP (x.o 1) l (t ≥ 2 ^ 64)).map (ws ++ ·) else some ws)
    else if i.hasMod "HI" then
      -- `IMAD.HI[.U32][.X] Rd, [Pc,] a, b, c[, Pin]`: the high word of the 64-bit sum of the product
      -- and the register pair `c:c+1` (`c` the low word; a constant or immediate c is the low word
      -- alone), plus the carry in, with the carry out of the high word (the remainder of `t % 9` checks it,
      -- tests/diff/ampere.ptx).
      let n := nP 1
      let carryIn := i.hasMod "X"
      some (res fun x l => do
        let a ← x.s l (1 + n); let b ← x.s l (2 + n)
        let c : Nat ← match x.o (3 + n) with
          | .reg ⟨.r, r⟩ m => do
            let v := (toU64 (← x.s l (3 + n)) (if r == 255 then 0 else x.v.getR (r + 1) l)).toNat
            pure (if m.neg then (2 ^ 64 - v) % 2 ^ 64 else v)
          | _ => do pure (← x.s l (3 + n)).toNat
        let p : Int := if signed then (a.toInt32.toInt * b.toInt32.toInt) else (a.toNat * b.toNat : Nat)
        let cin : Nat := if carryIn && (x.p l (4 + n)).getD false then 1 else 0
        let t := (p + c + (cin * 2 ^ 32 : Nat)) % 2 ^ 64
        let carry := (p % 2 ^ 64).toNat + c + cin * 2 ^ 32 ≥ 2 ^ 64
        let ws ← wr (x.o 0) l (t / 2 ^ 32).toNat.toUInt32
        if n ≥ 1 then (wrP (x.o 1) l carry).map (ws ++ ·) else some ws)
    else if i.hasMod "X" then some (res fun x l => do
      -- `IMAD.X Rd, a, b, c, Pin`: the high-word step of a 64-bit multiply-add.
      let cin : UInt32 := if (← x.p l 4) then 1 else 0
      wr (x.o 0) l ((← x.s l 1) * (← x.s l 2) + (← x.s l 3) + cin))
    else some (res fun x l => do wr (x.o 0) l ((← x.s l 1) * (← x.s l 2) + (← x.s l 3)))
  | "IADD3" =>
    -- `IADD3[.X] Rd, [Pc0, [Pc1,]] a, b, c[, Pin0, Pin1]`: the sum's carry out of the low word (0
    -- to 2) as two predicates of weight one each, Pc0 when it is at least 1 and Pc1 when 2, so
    -- `IADD3.X` adding Pin0 + Pin1 takes it whole;
    -- `UIADD3.64 URd, a, b, c` adds register pairs.
    let n := nP 1
    let carryIn := i.hasMod "X"
    if i.hasMod "64" then some (res fun x l => do
      let v (k : Nat) : Option UInt64 := src64 x.env x.v l (x.o (k + n))
      wr64 (x.o 0) l ((← v 1) + (← v 2) + (← v 3)))
    else some (res fun x l => do
      let a ← x.sN l (1 + n); let b ← x.sN l (2 + n); let c ← x.sN l (3 + n)
      let cin := if carryIn then
        [x.o (4 + n), x.o (5 + n)].foldl (fun acc q => acc + (if (pred x.v l q).getD false then 1 else 0)) 0
        else 0
      let t := a + b + c + cin
      let ws ← wr (x.o 0) l (t % 2 ^ 32).toUInt32
      let ws ← if n ≥ 1 then (wrP (x.o 1) l (t / 2 ^ 32 ≥ 1)).map (ws ++ ·) else some ws
      if n ≥ 2 then (wrP (x.o 2) l (t / 2 ^ 32 ≥ 2)).map (ws ++ ·) else some ws)
  | "VIADD" => some (res fun x l => do wr (x.o 0) l ((← x.s l 1) + (← x.s l 2)))
  | "IADD" =>
    -- `IADD.64 Rd, a, b` (sm_100 on): the 64-bit sum of register pairs or a pair and a
    -- sign-extended immediate; `-b` negates the pair.
    if !i.hasMod "64" then none else
    some (res fun x l => do wr64 (x.o 0) l ((← src64 x.env x.v l (x.o 1)) + (← src64 x.env x.v l (x.o 2))))
  | "LEA" =>
    if i.hasMod "HI" then
      -- `LEA.HI[.X][.SX32] Rd, [Pc,] a, b, c, s[, Pin]`: b plus the high word of (c:a) << s,
      -- where `.SX32` takes the sign extension of a for c; `.X` adds the carry in; Pc is the
      -- carry out of the sum (as `LEA`'s). `~a` inverts the pair (c:a).
      let sx := i.hasMod "SX32"
      let carryIn := i.hasMod "X"
      let n := nP 1
      let inv := match d.o (1 + n) with | .reg _ m => m.not | .not _ => true | _ => false
      some (res fun x l => do
        let a ← x.s l (1 + n); let b ← x.s l (2 + n)
        let c ← if sx then some (if a.toInt32 < 0 then 0xffffffff else 0) else x.s l (3 + n)
        let c := if inv && !sx then ~~~c else c
        let sh ← x.s l (if sx then 3 + n else 4 + n)
        let hi := shf true false true true a c sh
        let cin : Nat :=
          if carryIn && (x.p l (if sx then 4 + n else 5 + n)).getD false then 1 else 0
        let t := b.toNat + hi.toNat + cin
        let ws ← wr (x.o 0) l (t % 2 ^ 32).toUInt32
        if n ≥ 1 then (wrP (x.o 1) l (t ≥ 2 ^ 32)).map (ws ++ ·) else some ws)
    else
      -- `LEA Rd, [Pc,] a, b, s`: (a << s) + b, with the carry out. `-a` is negated after the
      -- shift, as `IADD3`'s operands are (`Exec.srcN`): when a << s is 0 it carries out (Triton's
      -- 64-bit row bound `ULEA UR14, UP0, -UR8, 0x3c, 0x3` takes its high word's carry from it).
      let n := nP 1
      let neg := match d.o (1 + n) with | .reg _ m => m.neg && !m.not | .neg _ => true | _ => false
      some (res fun x l => do
        let a ← x.sN l (1 + n); let b ← x.sN l (2 + n); let sh ← x.s l (3 + n)
        let s := (a <<< sh.toNat) % 2 ^ 32
        let t := (if neg && s == 0 then 2 ^ 32 else s) + b
        let ws ← wr (x.o 0) l (t % 2 ^ 32).toUInt32
        if n ≥ 1 then (wrP (x.o 1) l (t ≥ 2 ^ 32)).map (ws ++ ·) else some ws)
  | "SHF" =>
    let signed := i.hasMod "S32" || i.hasMod "S64"
    let wide := i.hasMod "U64" || i.hasMod "S64"
    let left := i.hasMod "L"; let hi := i.hasMod "HI"
    some (res fun x l => do
      wr (x.o 0) l (shf left signed wide hi (← x.s l 1) (← x.s l 3) (← x.s l 2)))
  | "LOP3" =>
    -- `LOP3.LUT [Pd,] Rd, a, b, c, lut, Pq`: `.PAND`/`.POR` combines
    -- result ≠ 0 with Pq.
    let pOut := isPred (d.o 0)
    let b0 := if pOut then 2 else 1
    some (res fun x l => do
      let r := lop3 (← x.s l b0) (← x.s l (b0 + 1)) (← x.s l (b0 + 2)) (← x.s l (b0 + 3))
      if pOut then do
        let p ← if i.hasMod "PAND" then (x.p l (b0 + 4)).map (r != 0 && ·)
          else if i.hasMod "POR" then (x.p l (b0 + 4)).map (r != 0 || ·)
          else some (r != 0)
        let ws ← wrP (x.o 0) l p
        (wr (x.o 1) l r).map (ws ++ ·)
      else wr (x.o 0) l r)
  | "PLOP3" =>
    -- `PLOP3.LUT Pd, Pq, a, b, c, lut, lut'`: Pd = lut(a, b, c), Pq = lut'(a, b, c).
    some (res fun x l => do
      let bit (n : Nat) : Option UInt32 := (x.p l n).map fun v => if v then 1 else 0
      let a ← bit 2; let b ← bit 3; let c ← bit 4
      let r1 := lop3 a b c (← x.s l 5) &&& 1; let r2 := lop3 a b c (← x.s l 6) &&& 1
      let ws ← wrP (x.o 0) l (r1 == 1)
      (wrP (x.o 1) l (r2 == 1)).map (ws ++ ·))
  | "ISETP" =>
    -- `ISETP.cmp[.U32].bop[.EX] Pd, Pq, a, b, Pc[, Plow]`: Pd = (a cmp b) bop Pc, Pq = ¬(a cmp b)
    -- bop Pc; `.EX` compares the high words of a 64-bit comparison whose low result is Plow.
    -- `.U64`/`.S64` (sm_100 on): one comparison of register pairs.
    let wide := i.hasMod "U64" || i.hasMod "S64"
    let signed := if wide then i.hasMod "S64" else !i.hasMod "U32"
    let ex := i.hasMod "EX"
    match i.mods[0]?, i.mods.find? (["AND", "OR", "XOR"].contains ·) with
    | some c, some bop => if wide then some (res fun x l => do
        let a ← src64 x.env x.v l (x.o 2); let b ← src64 x.env x.v l (x.o 3)
        let (u, v) : Int × Int := if signed then (a.toInt64.toInt, b.toInt64.toInt) else (a.toNat, b.toNat)
        let r ← match c with
          | "EQ" => some (u == v) | "NE" => some (u != v) | "LT" => some (u < v)
          | "LE" => some (u ≤ v) | "GT" => some (u > v) | "GE" => some (u ≥ v) | _ => none
        let pc ← x.p l 4
        let ws ← wrP (x.o 0) l (← boolOp bop r pc)
        (wrP (x.o 1) l (← boolOp bop (!r) pc)).map (ws ++ ·))
      else some (res fun x l => do
        let a ← x.s l 2; let b ← x.s l 3
        let r ← if ex then cmpEx c signed a b (← x.p l 5) else cmp signed c a b
        let pc ← x.p l 4
        let ws ← wrP (x.o 0) l (← boolOp bop r pc)
        (wrP (x.o 1) l (← boolOp bop (!r) pc)).map (ws ++ ·))
    | _, _ => some d.fault
  | "VIMNMX3" =>
    -- `VIMNMX3[.U32] Rd, a, b, c, P`: the minimum of a, b and c when P holds, else the maximum
    -- (as `VIMNMX`).
    let signed := !i.hasMod "U32"
    some (res fun x l => do
      let a ← x.s l 1; let b ← x.s l 2; let c ← x.s l 3
      let isMin ← x.p l 4
      let pick (u v : UInt32) : Option UInt32 := do
        let lt ← cmp signed "LT" u v
        pure (if isMin == lt then u else v)
      wr (x.o 0) l (← pick (← pick a b) c))
  | "IMNMX" | "VIMNMX" | "VIADDMNMX" =>
    -- `[V]IMNMX[.U32] Rd, a, b, P` and `VIADDMNMX[.U32] Rd, a, b, c, P`: the minimum when P holds,
    -- else the maximum, of a and b (of a + b and c).
    let signed := !i.hasMod "U32"
    let three := op == "VIADDMNMX"
    -- `IMNMX.S64|U64 Pu, Pv, Rd, a, b, P, Q` (sm_100 on): of register pairs, the minimum when P
    -- holds. Pu, Pv and Q are PT and !PT in every listing seen; what they mean otherwise is unknown, so other forms fault.
    if i.hasMod "S64" || i.hasMod "U64" then
      let sink (n : Nat) (neg : Bool) : Bool := match d.o n with
        | .reg ⟨f, 7⟩ m => (f == .p || f == .up) && m.not == neg | _ => false
      if !(sink 0 false && sink 1 false && sink 6 true) then some d.fault else
      let s64 := i.hasMod "S64"
      some (res fun x l => do
        let a ← src64 x.env x.v l (x.o 3); let b ← src64 x.env x.v l (x.o 4)
        let lt := if s64 then a.toInt64 < b.toInt64 else a < b
        wr64 (x.o 2) l (if (← x.p l 5) == lt then a else b))
    else
    -- Packed forms (`VIMNMX.U8x4`, sm_120a on): lane by lane, `w` bits each.
    let packed : Option (Nat × Bool) :=
      if i.hasMod "U8x4" then some (8, false) else if i.hasMod "S8x4" then some (8, true)
      else if i.hasMod "U16x2" then some (16, false) else if i.hasMod "S16x2" then some (16, true)
      else none
    if let some (w, sgn) := packed then
      if three then none else
      some (res fun x l => do
        let a ← x.s l 1; let b ← x.s l 2
        let isMin ← x.p l 3
        let lane (v : UInt32) (k : Nat) : Int :=
          let u := ((v >>> (w * k).toUInt32).toNat) % 2 ^ w
          if sgn && u ≥ 2 ^ (w - 1) then (u : Int) - 2 ^ w else u
        let r := (List.range (32 / w)).foldl (fun acc k =>
          let x := lane a k; let y := lane b k
          let m := if isMin then min x y else max x y
          acc ||| (((m % 2 ^ w).toNat).toUInt32 <<< (w * k).toUInt32)) (0 : UInt32)
        wr (x.o 0) l r)
    else
    -- `VIMNMX[.U32] Rd, PT, PT, a, b, P` (sm_100a): two predicate operands before
    -- the sources, PT in every listing seen; what they mean otherwise is unknown, so other forms
    -- fault.
    let preds := !three && (match d.o 1, d.o 2 with
      | .reg ⟨.p, _⟩ _, .reg ⟨.p, _⟩ _ => true | _, _ => false)
    let sinkPT (n : Nat) : Bool := match d.o n with
      | .reg ⟨.p, 7⟩ m => !m.not | _ => false
    if preds && !(sinkPT 1 && sinkPT 2) then some d.fault else
    let k := if preds then 2 else 0
    some (res fun x l => do
      let (a, b, pi) ← if three then (do pure ((← x.s l 1) + (← x.s l 2), ← x.s l 3, 4))
        else (do pure (← x.s l (1 + k), ← x.s l (2 + k), 3 + k))
      let lt ← cmp signed "LT" a b
      wr (x.o 0) l (if (← x.p l pi) == lt then a else b))
  | "SGXT" =>
    -- `SGXT[.U32] Rd, a, n`: the low n bits of a, sign- (or zero-) extended; n ≥ 32 keeps a.
    let signed := !i.hasMod "U32"
    some (res fun x l => do
      let a ← x.s l 1; let n := (← x.s l 2).toNat % 256
      if n ≥ 32 then wr (x.o 0) l a else
      let low := a.toNat % 2 ^ n
      let neg := signed && n > 0 && low / 2 ^ (n - 1) == 1
      wr (x.o 0) l (if neg then low + (2 ^ 32 - 2 ^ n) else low).toUInt32)
  | "BMSK" =>
    -- `BMSK[.W] Rd, pos, width` (PTX `bmsk`): `width` ones from bit `pos`; `.W` wraps both to
    -- 0–31, otherwise they clamp at 32.
    let wrap := i.hasMod "W"
    some (res fun x l => do
      let a := (← x.s l 1).toNat; let b := (← x.s l 2).toNat
      let (a, b) := if wrap then (a % 32, b % 32) else (min a 32, min b 32)
      wr (x.o 0) l ((2 ^ (min (a + b) 32) - 2 ^ a) % 2 ^ 32).toUInt32)
  | "IDP" =>
    -- `IDP.4A.{S8|U8}.{S8|U8} Rd, a, b, c` and `IDP.2A.{LO|HI}.{S16|U16}.{S8|U8} Rd, a, b, c`:
    -- PTX `dp4a` and `dp2a` (§9.7.1.24–25).
    let signed (n : Nat) := i.mods[n]?.any (·.startsWith "S")
    if i.hasMod "4A" then some (res fun x l => do
      wr (x.o 0) l (NV.PTX.Sem.Int.dp4a (signed 1) (signed 2) (← x.s l 1).toUInt64
        (← x.s l 2).toUInt64 (← x.s l 3).toUInt64).toUInt32)
    else if i.hasMod "2A" then some (res fun x l => do
      wr (x.o 0) l (NV.PTX.Sem.Int.dp2a (signed 2) (signed 3) (i.hasMod "HI") (← x.s l 1).toUInt64
        (← x.s l 2).toUInt64 (← x.s l 3).toUInt64).toUInt32)
    else none
  | "VABSDIFF" =>
    -- `VABSDIFF.{U32|S32} Rd, a, b, c`: |a − b| + c (PTX `sad`).
    let signed := i.hasMod "S32"
    some (res fun x l => do
      let a ← x.s l 1; let b ← x.s l 2; let c ← x.s l 3
      let d : UInt32 := if signed then (a.toInt32.toInt - b.toInt32.toInt).natAbs.toUInt32
        else if a ≥ b then a - b else b - a
      wr (x.o 0) l (d + c))
  | "VABSDIFF4" =>
    -- `VABSDIFF4.U8 Rd, a, b, c`: per byte |a − b| (PTX `vabsdiff4` with every byte selected and
    -- no accumulation; c is not read in this form — inferred, checked by the differential test).
    -- `.ACC`: the four differences summed and added to c (PTX `vabsdiff4….add`, the `__sad`-like
    -- sum of absolute differences; inferred, checked by tests/diff misc).
    if !i.hasMod "U8" then none else
    let acc := i.hasMod "ACC"
    some (res fun x l => do
      let a ← x.s l 1; let b ← x.s l 2
      let diff (k : Nat) : UInt32 :=
        let sh := (8 * k).toUInt32
        let p := (a >>> sh) &&& 0xff; let q := (b >>> sh) &&& 0xff
        if p ≥ q then p - q else q - p
      if acc then wr (x.o 0) l ((List.range 4).foldl (fun s k => s + diff k) (← x.s l 3))
      else wr (x.o 0) l ((List.range 4).foldl (fun r k => r ||| (diff k <<< (8 * k).toUInt32)) 0))
  | "I2IP" =>
    -- `I2IP.{S8|U8|S16|U16}.S32.SAT Rd, a, b, c` (PTX `cvt.pack.sat`, §9.7.10.24): a and b clamped
    -- to the narrow type, b in the low bits and a above it, the rest of the word c's low bits (no
    -- c for 16-bit types).
    match i.mods[0]?.bind Float.intType with
    | some (ds, dw) =>
      if !i.hasMod "SAT" || dw > 16 then none else
      some (res fun x l => do
        let lo : Int := if ds then -(2 ^ (dw - 1) : Nat) else 0
        let hi : Int := if ds then (2 ^ (dw - 1) - 1 : Nat) else (2 ^ dw - 1 : Nat)
        let cv (v : UInt32) : UInt32 :=
          let w := max lo (min hi v.toInt32.toInt)
          if w < 0 then (0 : UInt32) - (-w).toNat.toUInt32 else w.toNat.toUInt32
        let a ← x.s l 1; let b ← x.s l 2
        let c ← if dw == 16 then pure 0 else x.s l 3
        let mask : UInt32 := (1 <<< dw.toUInt32) - 1
        wr (x.o 0) l ((cv b &&& mask) ||| ((cv a &&& mask) <<< dw.toUInt32) |||
          (if dw == 16 then 0 else c <<< (2 * dw).toUInt32)))
    | none => none
  | "I2I" =>
    -- `I2I.{S8|U8|S16|U16}.{S32|U32}.SAT Rd, a`: a clamped to the narrow type, then extended to
    -- 32 bits by its sign (PTX `cvt.sat`).
    match i.mods[0]?.bind Float.intType, i.mods[1]?.bind Float.intType with
    | some (ds, dw), some (ss, 32) =>
      if !i.hasMod "SAT" then none else
      let lo : Int := if ds then -(2 ^ (dw - 1) : Nat) else 0
      let hi : Int := if ds then (2 ^ (dw - 1) - 1 : Nat) else (2 ^ dw - 1 : Nat)
      some (res fun x l => do
        let a ← x.s l 1
        let v : Int := if ss then a.toInt32.toInt else a.toNat
        let v := max lo (min hi v)
        wr (x.o 0) l (if v < 0 then (0 : UInt32) - (-v).toNat.toUInt32 else v.toNat.toUInt32))
    | _, _ => none
  | "SEL" =>
    -- `SEL.64` (sm_100 on): of register pairs (an immediate as written).
    if i.hasMod "64" then
      let s64 (x : Issue) (l n : Nat) : Option UInt64 :=
        match x.o n with | .imm b => some b | o => src64 x.env x.v l o
      some (res fun x l => do
        let a ← s64 x l 1; let b ← s64 x l 2
        wr64 (x.o 0) l (if ← x.p l 3 then a else b))
    else
    some (res fun x l => do
      let a ← x.s l 1; let b ← x.s l 2
      wr (x.o 0) l (if ← x.p l 3 then a else b))
  | "PRMT" =>
    if i.mods.size > 0 then none else
    some (res fun x l => do wr (x.o 0) l (prmt (← x.s l 1) (← x.s l 3) (← x.s l 2)))
  | "POPC" => some (res fun x l => do
      let v ← x.s l 1
      wr (x.o 0) l ((List.range 32).countP (fun b => (v >>> b.toUInt32) &&& 1 == 1)).toUInt32)
  | "BREV" => some (res fun x l => do
      let v ← x.s l 1
      wr (x.o 0) l ((List.range 32).foldl (fun acc b =>
        if (v >>> b.toUInt32) &&& 1 == 1 then acc ||| ((1 : UInt32) <<< (31 - b).toUInt32) else acc) 0))
  | "FLO" =>
    -- `FLO.U32`: index of the highest set bit; `.SH`: its distance from bit 31; none: all ones.
    if i.hasMod "S32" then none else
    let sh := i.hasMod "SH"
    some (res fun x l => do
      let v ← x.s l 1
      wr (x.o 0) l (match msb? v with
        | some b => (if sh then 31 - b else b).toUInt32
        | none => 0xffffffff))
  | "IABS" => some (res fun x l => do
      let v ← x.s l 1
      wr (x.o 0) l (if v.toInt32 < 0 then 0 - v else v))
  | "S2R" | "CS2R" | "CS2UR" =>
    -- `CS2R[.32]` reads a counter pair (or SRZ); `CS2UR[.32]` (sm_100 on) into uniform registers.
    let res := if op == "CS2UR" then d.res true else res
    if op != "S2R" && d.o 1 == .name "SRZ" then
      if i.hasMod "32" then some (res fun x l => wr (x.o 0) l 0) else some (res fun x l => wr64 (x.o 0) l 0)
    else if op != "S2R" then some (res fun x l => do
      let lo ← x.s l 1
      if i.hasMod "32" then wr (x.o 0) l lo else
      let hi ← match x.o 1 with
        | .name "SR_CLOCKLO" => specialReg x.env l "SR_CLOCKHI"
        | .name "SR_GLOBALTIMERLO" => specialReg x.env l "SR_GLOBALTIMERHI"
        | _ => none
      wr64 (x.o 0) l (toU64 lo hi))
    else some (res fun x l => do wr (x.o 0) l (← x.s l 1))
  | _ =>
    match i.opcode with
    | "S2UR" => some (d.res true fun x l => do wr (x.o 0) l (← x.s l 1))
    | "USETMAXREG" =>
      -- Registers are not a limited resource of the functional model: every allocation succeeds
      -- (the CTA pool that could make it fail is not modelled; success is a legal outcome). The warp's file holds the largest count from launch
      -- (`Kernel.regFile`), so registers past the old allocation keep their values.
      if i.hasMod "TRY_ALLOC" then some (d.res true fun x l => wrP (x.o 0) l true)
      else if i.hasMod "DEALLOC" then some (d.run fun x => .next (x.fin x.w)) else some d.fault
    | "R2UR" =>
      -- The value of the first executing lane.
      match d.o 0 with
      | .reg ⟨.p, _⟩ _ =>
        -- `R2UR[.OR] Pd, URd, Ra`: also Pd, per lane, whether its Ra differs from the value taken
        -- (with `.OR`, or Pd).
        let orIn := i.hasMod "OR"
        some <| d.run fun x => match lanes x.on with
          | [] => .next (x.fin x.w)
          | first :: _ => match x.s first 2 with
            | none => .fault x.bad
            | some v => x.result false fun l => do
              let differs := (← x.s l 2) != v
              let p ← if orIn then (do pure (differs || (← x.p l 0))) else pure differs
              pure ((← wrP (x.o 0) l p) ++ (← wr (x.o 1) l v))
      | _ => some (d.res true fun x l => do wr (x.o 0) l (← x.s l 1))
    | "ELECT" =>
      -- `ELECT Pd, URd, Pmask`: the lowest executing lane is the leader; Pd marks it, URd gets
      -- its index.
      some <| d.run fun x => match lanes x.on with
      | [] => .next (x.fin x.w)
      | leader :: _ => .reg (x.fin x.w) (((lanes x.on).filterMap fun l =>
          wrP (x.o 0) l (l == leader)).flatten ++ (wr (x.o 1) leader leader.toUInt32).getD [])
    | "VOTE" | "VOTEU" =>
      -- `VOTE.ANY|ALL|EQ Rd, Pd, Ps`, `VOTE.ANY Pd, Ps`, `VOTEU.ANY URd, UPd, Ps`: Rd is the
      -- ballot of Ps over the executing lanes, Pd its reduction.
      let n := i.operands.size
      let ps := d.o (n - 1); let pd := d.o (n - 2)
      let any := i.hasMod "ANY"; let all := i.hasMod "ALL"; let eq := i.hasMod "EQ"
      if !(any || all || eq) then some d.fault else
      some <| d.run fun x =>
        let ballot : UInt32 := (lanes x.on).foldl (fun acc l =>
          if (pred x.v l ps).getD false then acc ||| ((1 : UInt32) <<< l.toUInt32) else acc) 0
        let red := if any then ballot != 0 else if all then ballot == x.on
          else ballot == 0 || ballot == x.on
        let ws := (lanes x.on).flatMap fun l =>
          (if n == 3 then (wr (x.o 0) l ballot).getD [] else []) ++ (wrP pd l red).getD []
        .reg (x.fin x.w) ws
    | "SHFL" =>
      -- `SHFL.mode Pd, Rd, Ra, b, c` (PTX `shfl.sync`): lane j's Ra, j from the mode, b, the clamp
      -- c[4:0] and the segment mask c[12:8]; Pd says whether j was in range (else j is the lane
      -- itself). A source lane that does not execute gives its register's current contents
      -- (PTX: undefined value; inferred).
      -- The source lane and whether it is in range, from the lane, b, the clamp and the mask.
      let lane? : Option (Nat → Nat → Nat → Nat → Int × Bool) := match i.mods[0]? with
        | some "UP" => some fun l bv maxLane _ => ((l : Int) - bv, (l : Int) - bv ≥ maxLane)
        | some "DOWN" => some fun l bv maxLane _ => (l + bv, l + bv ≤ maxLane)
        | some "BFLY" => some fun l bv maxLane _ => ((l ^^^ bv : Nat), (l ^^^ bv) ≤ maxLane)
        | some "IDX" => some fun l bv maxLane mask =>
          let j := (l &&& mask) ||| (bv &&& (0x1f ^^^ mask)); ((j : Nat), j ≤ maxLane)
        | _ => none
      lane?.map fun lane => res fun x l => do
        let b ← x.s l 3; let c ← x.s l 4
        let bv := (b &&& 0x1f).toNat; let cv := (c &&& 0x1f).toNat; let mask := ((c >>> 8) &&& 0x1f).toNat
        let (j, ok) := lane l bv ((l &&& mask) ||| (cv &&& (0x1f ^^^ mask))) mask
        let jl := if ok then j.toNat else l
        let v ← src x.env x.v jl (x.o 2)
        let ws ← wrP (x.o 0) l ok
        (wr (x.o 1) l v).map (ws ++ ·)
    | "MATCH" =>
      -- `MATCH.ANY Rd, Ra`: the executing lanes holding the same value; `MATCH.ALL [Pd,] Rd, Ra`:
      -- all of them when every executing lane agrees, else 0.
      let wide := i.hasMod "U64"
      let a := d.o (i.operands.size - 1)
      let pOut := isPred (d.o 0)
      let any := i.hasMod "ANY"
      if !(any || i.hasMod "ALL") then some d.fault else
      some <| d.run fun x =>
        let val (l : Nat) : Option UInt64 := do
          let lo ← src x.env x.v l a
          if wide then some (toU64 lo (← srcHi x.env x.v l a)) else some lo.toUInt64
        match (lanes x.on).mapM fun l => (val l).map (l, ·) with
        | none => .fault x.bad
        | some vs =>
          let same (v : UInt64) : UInt32 := vs.foldl (fun acc (l', v') =>
            if v' == v then acc ||| ((1 : UInt32) <<< l'.toUInt32) else acc) 0
          let all := vs.all fun (_, v) => some v == (vs.head?.map (·.2))
          .reg (x.fin x.w) (vs.flatMap fun (l, v) =>
            if any then (wr (x.o 0) l (same v)).getD []
            else (if pOut then (wrP (x.o 0) l all).getD [] else []) ++
              (wr (x.o (if pOut then 1 else 0)) l (if all then x.on else 0)).getD [])
    | "REDUX" | "CREDUX" =>
      -- `REDUX[.op][.S32] URd, Ra`: the reduction of Ra over the executing lanes. `CREDUX` is
      -- sm_100a's form of the minimum and maximum.
      let signed := i.hasMod "S32"
      let f? : Option (UInt32 → UInt32 → UInt32) :=
        if i.hasMod "SUM" then some (· + ·) else if i.hasMod "AND" then some (· &&& ·)
        else if i.hasMod "OR" then some (· ||| ·) else if i.hasMod "XOR" then some (· ^^^ ·)
        else if i.hasMod "MIN" then some fun a b => if (cmp signed "LT" a b).getD false then a else b
        else if i.hasMod "MAX" then some fun a b => if (cmp signed "GT" a b).getD false then a else b
        -- No operation named: AND.
        else some (· &&& ·)
      match f? with
      | none => some d.fault
      | some f => some <| d.run fun x =>
        match (lanes x.on).mapM (fun l => x.s l 1) with
        | none | some [] => .fault x.bad
        | some (v :: vs) => x.result true fun l => wr (x.o 0) l (vs.foldl f v)
    | "B2R" =>
      -- `B2R.RESULT Rd[, Pd]`: the warp's last barrier reduction's answer (`BAR.RED`), Pd
      -- whether it is nonzero.
      if !i.hasMod "RESULT" then some d.fault else
      some (res fun x l => do
        let v := x.v.barResult
        let pw ← if i.operands.size > 1 then wrP (x.o 1) l (v != 0) else pure []
        pure ((← wr (x.o 0) l v) ++ pw))
    | "P2R" | "UP2UR" =>
      -- `P2R Rd, PR, Ra, mask`: Rd = (Ra & ~mask) | (predicates P0–P6 as bits & mask);
      -- `UP2UR URd, UPR, URa, mask` (sm_100a): the uniform predicates UP0–UP6. `.B1`–`.B3`
      -- place the predicates and the mask in that byte of Rd instead.
      let sh : UInt32 := if i.hasMod "B1" then 8 else if i.hasMod "B2" then 16 else if i.hasMod "B3" then 24 else 0
      some (res fun x l => do
        let a ← x.s l 2; let mask := (← x.s l 3) <<< sh
        let bits := (List.range 7).foldl (fun acc b =>
          if (if u then x.v.getUP b else x.v.getP b l) then acc ||| ((1 : UInt32) <<< b.toUInt32) else acc) 0
        wr (x.o 0) l ((a &&& ~~~mask) ||| ((bits <<< sh) &&& mask)))
    | "R2P" =>
      -- `R2P PR, Ra, mask`: predicates P0–P6 under mask take Ra's bits; `Ra.B1`–`.B3`, those of
      -- that byte of Ra.
      let sh : UInt32 := match d.o 1 with
        | .reg _ m => if m.suffix.contains "B1" then 8 else if m.suffix.contains "B2" then 16
            else if m.suffix.contains "B3" then 24 else 0
        | _ => 0
      some (res fun x l => do
        let a := (← x.s l 1) >>> sh; let mask ← x.s l 2
        some ((List.range 7).filterMap fun b =>
          if (mask >>> b.toUInt32) &&& 1 == 1 then some (Write.p b (lane32 l) ((a >>> b.toUInt32) &&& 1 == 1))
          else none))
    | _ => none

namespace Issue

/-- Move lanes `j` to the instruction at byte offset `t`, and the other lanes here past this one. -/
def jumpTo (x : Issue) (t : Operand) (j : UInt32) : Out :=
  match t with
  | .target pc => match x.target pc with
    | some n => .next ((x.w.advance (x.m &&& ~~~j) x.pc).jump j n)
    | none => .fault s!"{x.i.full} to {pc}: no instruction there"
  | _ => .fault x.bad

/-- The lanes here reach a barrier that waits for `expect`: everyone at this instruction goes on
when all live lanes of `expect` are here, else they wait. -/
def barrier (x : Issue) (expect : UInt32) : Out :=
  let w := x.w
  let here := (lanes (w.live &&& expect)).all fun l => Warp.laneOn x.m l || w.pc l == x.pc
  let atPc := (lanes w.live).foldl (fun acc l =>
    if w.pc l == x.pc then acc ||| ((1 : UInt32) <<< l.toUInt32) else acc) 0
  if here then .next ({ w with waiting := w.waiting &&& ~~~atPc }.advance atPc x.pc)
  else if x.m &&& w.waiting == x.m && (w.live &&& ~~~w.waiting) == 0 then
    .fault s!"{x.i.full} at {x.i.pc}: convergence deadlock"
  else .next { w with waiting := w.waiting ||| x.m }

/-- The NUL-terminated string at `a` in the module's data, if it lies there. -/
def cString (k : Kernel) (a : UInt64) : Option String := do
  let (base, bytes) ← k.data.find? fun (b, bs) => b ≤ a.toNat && a.toNat < b + bs.size
  let cs := ((bytes.toList.drop (a.toNat - base)).takeWhile (· != 0)).map (Char.ofNat ·.toNat)
  return String.ofList cs

/-- `vprintf(fmt, buf)` (device `printf`): per executing lane, the format at `R4:R5` (in the
module's data) and the argument buffer at `R6:R7` (a generic address of the lane's stack; the
local window's base is `localWindow`). The arguments are loaded from local
memory, and the writeback appends a record per lane to the warp's `printf` log and returns the
number of arguments in `R4`. Arguments of more than 16 bytes are not modelled yet. -/
def vprintf (x : Issue) : Out := Id.run do
  let ls := lanes x.on
  let x := { x with w := { x.w with printing := ls.length } }
  let window := localWindow x.env x.k.sm
  let mut calls : Array (Nat × UInt64 × Nat × Nat) := #[]
  for l in ls do
    let fmt := toU64 (x.v.getR 4 l) (x.v.getR 5 l)
    let some f := cString x.k fmt
      | return .fault s!"{x.i.full} at {x.i.pc}: printf's format at {fmt} is not in the module's data"
    let (slots, n) := NV.Printf.layout f
    if n > 16 then
      return .fault s!"{x.i.full} at {x.i.pc}: printf with {n} bytes of arguments (more than 16 is not modelled yet)"
    calls := calls.push (l, fmt, n, slots.length)
  let n : Nat := calls.foldl (fun (m : Nat) (c : Nat × UInt64 × Nat × Nat) => max m c.2.2.1) 0
  let records (args : Nat → Nat → ByteArray) : List Write :=
    calls.toList.flatMap fun (l, fmt, k, nargs) =>
      [.printf fmt (args l k), .r 4 (lane32 l) nargs.toUInt32]
  if n == 0 then return .reg (x.fin x.w) (records fun _ _ => ByteArray.empty)
  let bytes := if n ≤ 4 then 4 else if n ≤ 8 then 8 else 16
  let w := words bytes
  let req : Req :=
    { space := .local, op := .ld, bytes, sem := .weak, scope := .thread,
      addrs := (Array.range 32).map fun l => if calls.any (·.1 == l) then
        some (toU64 (x.v.getR 6 l) (x.v.getR 7 l) - window) else none }
  return .mem (fun _ => req) (x.fin x.w) fun resp => records fun l k =>
    ⟨(Array.range k).map fun j => ((resp[l * w + j / 4]?.getD 0) >>> (8 * (j % 4)).toUInt32).toUInt8⟩

/-- `cudaGraphSetConditional(handle, value)` (the device runtime's, which steers a CUDA graph's
conditional nodes; CUDA Runtime API §6.30 and the Programming Guide's conditional nodes): per
executing lane, the value in `R6` stored as 4 bytes to the handle in `R4:R5`, the registers the
call ABI passes them in. A handle is the global address of the word the condition lives in
(inferred). -/
def setConditional (x : Issue) : Out :=
  let on := lanes x.on
  let req : Req :=
    { space := .global, op := .st, bytes := 4,
      addrs := (Array.range 32).map fun l =>
        if on.contains l then some (toU64 (x.v.getR 4 l) (x.v.getR 5 l)) else none,
      data := (Array.range 32).map fun l => x.v.getR 6 l }
  .mem (fun _ => req) (x.fin x.w) fun _ => []

end Issue

/-- The text of a `printf` record of kernel `k`, as the host prints it. -/
def printfText (k : Kernel) (fmt : UInt64) (args : ByteArray) : String :=
  match Issue.cString k fmt with
  | some f => NV.Printf.render f args (Issue.cString k)
  | none => s!"(printf: format at {fmt} not in the module's data)"

/-- Branches, calls, convergence barriers and warp synchronization. -/
def decodeCtrl (d : Dec) : Option Run :=
  let i := d.i
  let barReg : Option Nat := match d.o 0 with | .reg ⟨.b, n⟩ _ => some n | _ => none
  match i.opcode with
  | "EXIT" =>
    -- `@P EXIT [Q]`: a predicate operand also has to hold for a lane to exit (ptxas's
    -- `@P0 EXIT P1`: every lane but those where `P1` fails).
    let preds := i.operands.toList.filter isPred
    some <| d.run fun x =>
      let out := if preds.isEmpty then x.on else (lanes x.on).foldl (fun acc l =>
        if preds.all (fun o => (pred x.v l o).getD false) then acc ||| ((1 : UInt32) <<< l.toUInt32) else acc) 0
      let w := x.w.advance x.m x.pc
      .next { w with live := w.live &&& ~~~out, waiting := w.waiting &&& ~~~out }
  | "BRA" =>
    let t := i.operands.toList.getLast?.getD (.name "?")
    if i.hasMod "DIV" then
      -- `BRA.DIV mask, target`: taken when lanes of the mask are live but not here (the warp is
      -- diverged).
      some <| d.run fun x => match src x.env x.v 0 (x.o 0) with
        | some mask => x.jumpTo t (if (mask &&& x.w.live &&& ~~~x.m) != 0 then x.m else 0)
        | none => .fault x.bad
    else if i.hasMod "ANY" then
      -- `@P BRA.U.ANY target`: every lane here goes when the guard holds in any of them.
      some <| d.run fun x => x.jumpTo t (if x.on != 0 then x.m else 0)
    else
      -- `@P BRA [Q,] target`: a predicate operand also has to hold.
      let preds := i.operands.toList.filter isPred
      if preds.isEmpty then some <| d.run fun x => x.jumpTo t x.on
      else some <| d.run fun x => x.jumpTo t ((lanes x.on).foldl (fun acc l =>
        if preds.all (fun o => (pred x.v l o).getD false)
        then acc ||| ((1 : UInt32) <<< l.toUInt32) else acc) 0)
  | "BSSY" => some <| match barReg with
    | some n => d.run fun x => .next (x.fin { x.w with b := x.w.b.set! n x.on })
    | none => d.fault
  | "BSYNC" => some <| match barReg with
    | some n => d.run fun x => if x.on != x.m then .fault x.bad else x.barrier (x.v.getB n)
    | none => d.fault
  | "BREAK" => some <| match barReg with
    | some n => d.run fun x => .next (x.fin { x.w with b := x.w.b.set! n ((x.v.getB n) &&& ~~~x.on) })
    | none => d.fault
  | "BMOV" =>
    -- `BMOV.32 Bx, Ra` and `BMOV.32[.CLEAR] Rd, Bx`.
    let clear := i.hasMod "CLEAR"
    some <| match d.o 0, d.o 1 with
    | .reg ⟨.b, n⟩ _, a => d.run fun x => match lanes x.on with
      | l :: _ => match src x.env x.v l a with
        | some v => .next (x.fin { x.w with b := x.w.b.set! n v })
        | none => .fault x.bad
      | [] => .next (x.fin x.w)
    | dst, .reg ⟨.b, n⟩ _ => d.run fun x =>
      let w' := if clear then { x.w with b := x.w.b.set! n 0 } else x.w
      match (lanes x.on).foldlM (fun acc l => (wr dst l (x.v.getB n)).map (acc ++ ·)) [] with
      | some ws => .reg (x.fin w') ws
      | none => .fault x.bad
    | _, _ => d.fault
  | "UCGABAR_ARV" | "UCGABAR_WAIT" =>
    -- The cluster barrier (PTX `barrier.cluster.arrive`/`.wait`, rule `clusterBarrier`): the
    -- warp's live threads arrive, or wait for every thread of the cluster to have arrived; the
    -- machine counts arrivals.
    let op := if i.opcode == "UCGABAR_ARV" then SyncOp.clusterArrive else .clusterWait
    some <| d.run fun x => .sync op (x.fin x.w)
  | "ENDCOLLECTIVE" =>
    -- Closes the region a `WARPSYNC.COLLECTIVE mask, end` opens (sm_120a, around an `ELECT` on
    -- a divergent path); the region's lanes are already together, so nothing to do (inferred).
    some <| d.run fun x => .next (x.fin x.w)
  | "WARPSYNC" =>
    if i.hasMod "ALL" then some <| d.run fun x => x.barrier fullMask
    else some <| d.run fun x => match lanes x.on with
      | l :: _ => match src x.env x.v l (x.o 0) with
        | some mask => x.barrier mask
        | none => .fault x.bad
      | [] => .next (x.fin x.w)
  | "ACQBULK" | "PREEXIT" =>
    -- PTX `griddepcontrol.wait` and `griddepcontrol.launch_dependents` (sm_120a; inferred): ordering against other grids, which a
    -- single launch does not have, so nothing to do for the warp.
    some <| d.run fun x => .next (x.fin x.w)
  | "BRXU" =>
    -- `BRXU URa off` (sm_120a): `BRX` through a uniform register pair, the same target for every
    -- lane.
    match d.o 0, d.o 1 with
    | .reg ⟨.ur, ra⟩ _, .imm off => some <| d.run fun x =>
      let t := (x.i.pc + 16 : Nat).toUInt64 + toU64 (x.v.getUR ra) (x.v.getUR (ra + 1)) + off
      match x.target t.toNat with
      | some pc => .next ((x.w.advance (x.m &&& ~~~x.on) x.pc).jump x.on pc)
      | none => .fault s!"{x.i.full} at {x.i.pc}: branch to no instruction"
    | _, _ => some d.fault
  | "BRX" =>
    -- `BRX Ra off`: each lane to the byte offset next pc + Ra:Ra+1 + off.
    match d.o 0, d.o 1 with
    | .reg ⟨.r, ra⟩ _, .imm off => some <| d.run fun x =>
      let w := x.w.advance (x.m &&& ~~~x.on) x.pc
      match (lanes x.on).foldlM (fun w l =>
          let t := (x.i.pc + 16 : Nat).toUInt64 + toU64 (x.v.getR ra l) (x.v.getR (ra + 1) l) + off
          (x.target t.toNat).map (w.jump ((1 : UInt32) <<< l.toUInt32) ·)) w with
      | some w => .next w
      | none => .fault s!"{x.i.full} at {x.i.pc}: branch to no instruction"
    | _, _ => some d.fault
  | "CALL" =>
    match d.o 0 with
    | .reg ⟨.r, rn⟩ _ =>
      -- `CALL.REL.NOINC Rn base`: each lane to the code offset Rn:Rn+1 from `base`.
      if i.hasMod "REL" then
        match d.o 1 with
        | .target base => some <| d.run fun x =>
          let w := x.w.advance (x.m &&& ~~~x.on) x.pc
          match (lanes x.on).foldlM (fun w l =>
              let t := base.toUInt64 + toU64 (x.v.getR rn l) (x.v.getR (rn + 1) l)
              (x.target t.toNat).map (w.jump ((1 : UInt32) <<< l.toUInt32) ·)) w with
          | some w => .next w
          | none => .fault s!"{x.i.full} at {x.i.pc}: call to no instruction"
        | _ => some d.fault
      -- `CALL.ABS.NOINC Rn`: to the absolute address in Rn:Rn+1. The targets modelled are the
      -- module's external functions; of those, `vprintf` (device `printf`) and the device
      -- runtime's `cudaGraphSetConditional`.
      else if !i.hasMod "ABS" then none else some <| d.run fun x => match lanes x.on with
        | [] => .next (x.fin x.w)
        | l :: _ =>
          let a := toU64 (x.v.getR rn l) (x.v.getR (rn + 1) l)
          match x.k.externs.find? (·.1 == a.toNat) with
          | some (_, "vprintf") => x.vprintf
          | some (_, "cudaGraphSetConditional") => x.setConditional
          | some (_, name) => .fault s!"{i.full} at {i.pc}: a call to {name}, which is not modelled"
          | none => .fault s!"{i.full} at {i.pc}: a call to {a}, not a function of the module"
    -- `CALL.REL.NOINC target` and `CALL.REL target`: the same jump; the callee
    -- returns with `RET.ABS Ra offset`. The call-depth count it keeps for convergence is not
    -- modelled.
    | _ => if !i.hasMod "NOINC" && !i.hasMod "REL" then none else
      some <| d.run fun x => x.jumpTo (x.o 0) x.on
  | "LEPC" =>
    -- `LEPC Rd, target`: the code address of `target` (the module's code offset plus the byte
    -- offset).
    match d.o 1 with
    | .imm t => some (d.res false fun x l => wr64 (x.o 0) l (d.k.codeOffset + t.toNat).toUInt64)
    | .target t => some (d.res false fun x l => wr64 (x.o 0) l (d.k.codeOffset + t).toUInt64)
    -- `LEPC Rd` (sm_80): its own address.
    | _ => if i.operands.size != 1 then some d.fault else
      some (d.res false fun x l => wr64 (x.o 0) l (d.k.codeOffset + i.pc).toUInt64)
  | "RET" =>
    -- `RET.REL.NODEC Ra target`: to the address in Ra, relative to the function at `target`.
    -- `RET.ABS Ra offset`: to the code address in Ra:Ra+1 (an `LEPC`'s) plus `offset`.
    some <| match d.o 1 with
    | .target base => d.run fun x => match lanes x.on with
      | l :: _ =>
        if i.hasMod "ABS" then
          let a := toU64 (x.v.getR (match x.o 0 with | .reg ⟨.r, r⟩ _ => r | _ => 0) l)
            (x.v.getR (match x.o 0 with | .reg ⟨.r, r⟩ _ => r + 1 | _ => 0) l)
          if a.toNat < x.k.codeOffset then .fault s!"{x.i.full} at {x.i.pc}: a return to {a}, below the kernel's code"
          else x.jumpTo (.target (a.toNat - x.k.codeOffset + base)) x.on
        else match src x.env x.v l (x.o 0) with
        | some a => x.jumpTo (.target (base + a.toNat)) x.on
        | none => .fault x.bad
      | [] => .next (x.fin x.w)
    | _ => d.fault
  -- `PMTRIG` (a performance-monitor trigger): nothing the model observes.
  | "NOP" | "DEPBAR" | "NANOSLEEP" | "YIELD" | "PMTRIG" => some <| d.run fun x => .next (x.fin x.w)
  | "BPT" => some fun _ _ _ _ => .fault s!"{i.full} at {i.pc}: trap"
  | _ => none

/-- Floating-point arithmetic, comparison, conversion and the special-function unit, on
`NV.Num`. Binary32 unless the opcode says otherwise (`H…2`: packed binary16 or bfloat16 pairs;
`D…`: binary64 register pairs). -/
def decodeFloat (d : Dec) : Option Run :=
  let i := d.i
  let ftz := i.hasMod "FTZ"
  let r := Float.rounding i.mods
  let sat := i.hasMod "SAT"
  let f32 := NV.Num.Format.f32
  let fin32 (v : Nat) : UInt32 := (if sat then NV.Num.satUnit f32 v else v).toUInt32
  -- A binary32 source.
  let fs (x : Issue) (l n : Nat) : Option Nat := (Float.src x.env x.v l 0x80000000 (x.o n)).map (·.toNat)
  let f64 (x : Issue) (l n : Nat) : Option Nat := (Float.src64 x.env x.v l (x.o n)).map (·.toNat)
  -- Packed pairs: the format of `HADD2`-like opcodes.
  let hf := if i.hasMod "BF16_V2" then NV.Num.Format.bf16 else NV.Num.Format.f16
  let pair (x : Issue) (l n : Nat) : Option (Nat × Nat) :=
    (Float.src x.env x.v l 0x80008000 (x.o n)).map fun v =>
      let (h, lo) := Float.halves (x.o n) v; (h.toNat, lo.toNat)
  let packed (hi lo : Nat) : UInt32 := ((hi % 2 ^ 16) * 2 ^ 16 + lo % 2 ^ 16).toUInt32
  let hsat (v : Nat) : Nat := if sat then NV.Num.satUnit hf v else v
  -- Uniform-datapath forms (`UFMUL`, `UFSETP`, `UI2F`, sm_100 on) share their vector forms'.
  let (u, op) := if i.opcode.startsWith "U" then (true, (i.opcode.drop 1).toString)
    else (false, i.opcode)
  let d := { d with uniform := u }
  match op with
  | "FADD" => some <| d.res d.uniform fun x l => do wr (x.o 0) l (fin32 (NV.Num.add f32 r ftz (← fs x l 1) (← fs x l 2)))
  | "FMUL" =>
    -- `.D2`/`.D4`/`.D8` and `.M2`/`.M4`/`.M8`: the product scaled by 2^-1..2^-3 or 2^1..2^3 and
    -- rounded again (tests/diff/fmulscale.ptx).
    let scale : Option Nat := [("D2", 126), ("D4", 125), ("D8", 124), ("M2", 128), ("M4", 129),
      ("M8", 130)].findSome? fun (m, e) => if i.hasMod m then some (e * 2 ^ 23) else none
    some <| d.res d.uniform fun x l => do
      let v := NV.Num.mul f32 r ftz (← fs x l 1) (← fs x l 2)
      wr (x.o 0) l (fin32 (match scale with | some s => NV.Num.mul f32 r ftz v s | none => v))
  | "FFMA" => some <| d.res d.uniform fun x l => do
      wr (x.o 0) l (fin32 (NV.Num.fma f32 r ftz (← fs x l 1) (← fs x l 2) (← fs x l 3)))
  | "FMNMX" =>
    -- `FMNMX Rd, a, b, P`: the minimum when P holds, else the maximum (as `VIMNMX`).
    let nanProp := i.hasMod "NAN"
    some <| d.res d.uniform fun x l => do
      let a ← fs x l 1; let b ← fs x l 2
      wr (x.o 0) l (if ← x.p l 3 then NV.Num.min f32 ftz nanProp false a b
        else NV.Num.max f32 ftz nanProp false a b).toUInt32
  | "FMNMX3" =>
    -- `FMNMX3 Rd, a, b, c, P` (sm_100a): FMNMX of the three, pairwise (inferred; tests/sass diff-cube).
    let nanProp := i.hasMod "NAN"
    some <| d.res d.uniform fun x l => do
      let a ← fs x l 1; let b ← fs x l 2; let c ← fs x l 3
      let f := if ← x.p l 4 then NV.Num.min f32 ftz nanProp false else NV.Num.max f32 ftz nanProp false
      wr (x.o 0) l (f (f a b) c).toUInt32
  | "FCHK" =>
    -- `FCHK Pd, a, b`: whether a fast division of a / b may be inexact. NVIDIA does not document the condition; this is a conservative
    -- one (inferred): the fast path only for normal (or zero) a and normal b whose exponents and
    -- their difference stay well inside the normal range. The differential tests check the
    -- quotients, which are exact either way.
    some <| d.res d.uniform fun x l => do
      let a ← fs x l 1; let b ← fs x l 2
      let ex (v : Nat) : Int := ((v / 2 ^ 23) % 256 : Nat) - 127
      let normal (v : Nat) : Bool := let e := (v / 2 ^ 23) % 256; e != 0 && e != 255
      let zero (v : Nat) : Bool := v % 2 ^ 31 == 0
      let fast := normal b && (zero a || normal a && ex a ≥ -100 && ex a ≤ 100 &&
        ex b ≥ -100 && ex b ≤ 100 && (ex a - ex b) ≤ 100 && (ex b - ex a) ≤ 100)
      wrP (x.o 0) l !fast
  | "FRND" =>
    -- `FRND[.TRUNC|.FLOOR|.CEIL][.FTZ][.F64] Rd, a`: a rounded to an integral value (PTX
    -- `cvt.rXi.f32.f32`), to nearest even without a mode.
    if i.hasMod "F64" then some <| d.res d.uniform fun x l => do
      wr64 (x.o 0) l (NV.Num.roundInt .f64 r false (← f64 x l 1)).toUInt64
    else some <| d.res d.uniform fun x l => do wr (x.o 0) l (NV.Num.roundInt f32 r ftz (← fs x l 1)).toUInt32
  | "HSETP2" =>
    -- `HSETP2.cmp.bop Pd, Pq, a, b, Pc`: Pd from the low halves, Pq from the high halves, each
    -- combined with Pc (PTX `setp.f16x2 p|q`, p for the low half).
    match i.mods[0]?.bind Float.cmpOp, i.mods.find? (["AND", "OR", "XOR"].contains ·) with
    | some (some c), some bop =>
      if i.hasMod "H_AND" then none else
      some <| d.res d.uniform fun x l => do
        let (ah, al) ← pair x l 2; let (bh, bl) ← pair x l 3
        let pc ← x.p l 4
        let ws ← wrP (x.o 0) l (← boolOp bop (NV.Num.cmp hf c ftz al bl) pc)
        (wrP (x.o 1) l (← boolOp bop (NV.Num.cmp hf c ftz ah bh) pc)).map (ws ++ ·)
    | _, _ => some d.fault
  | "HSET2" =>
    -- `HSET2[.BF].cmp[.FTZ].bop Rd, a, b, Pc` (PTX `set.cmp{.ftz}.{u32|f16x2}.f16x2`): per half the
    -- comparison combined with Pc, as all ones, or with `.BF` as 1.0 of the halves' format, or 0.
    match i.mods.findSome? Float.cmpOp, i.mods.find? (["AND", "OR", "XOR"].contains ·) with
    | some (some c), some bop =>
      let one : Nat := if i.hasMod "BF" then (if hf == .bf16 then 0x3f80 else 0x3c00) else 0xffff
      some <| d.res d.uniform fun x l => do
        let (ah, al) ← pair x l 1; let (bh, bl) ← pair x l 2
        let pc ← x.p l 3
        let h (u v : Nat) : Option Nat := do pure (if ← boolOp bop (NV.Num.cmp hf c ftz u v) pc then one else 0)
        wr (x.o 0) l (packed (← h ah bh) (← h al bl))
    | _, _ => some d.fault
  | "HMNMX2" =>
    -- `HMNMX2 Rd, a, b, P`: per half, the minimum when P holds, else the maximum.
    let nanProp := i.hasMod "NAN"
    some <| d.res d.uniform fun x l => do
      let (ah, al) ← pair x l 1; let (bh, bl) ← pair x l 2
      let m (a b : Nat) (isMin : Bool) : Nat :=
        if isMin then NV.Num.min hf false nanProp false a b else NV.Num.max hf false nanProp false a b
      let isMin ← x.p l 3
      wr (x.o 0) l (packed (m ah bh isMin) (m al bl isMin))
  | "FSEL" => some <| d.res d.uniform fun x l => do
      let a ← Float.src x.env x.v l 0x80000000 (x.o 1); let b ← Float.src x.env x.v l 0x80000000 (x.o 2)
      wr (x.o 0) l (if ← x.p l 3 then a else b)
  | "FSETP" | "DSETP" =>
    -- `FSETP.cmp.bop[.FTZ] Pd, Pq, a, b, Pc` (as `ISETP`).
    let wide := i.opcode == "DSETP"
    if wide && (i.mods[0]? == some "MIN" || i.mods[0]? == some "MAX") then
      -- `DSETP.MIN|MAX.AND Pd, Pq, a, b, PT` (`min`/`max.f64` before sm_100): Pd selects a (the
      -- minimum or maximum; a NaN loses to a number; -0 is below +0), Pq = both are NaN (checked
      -- by tests/diff f64ops).
      let isMax := i.mods[0]? == some "MAX"
      some <| d.res d.uniform fun x l => do
        let a ← f64 x l 2; let b ← f64 x l 3
        let nan (v : Nat) := NV.Num.Format.f64.isNaN v
        let neg (v : Nat) := v ≥ 2 ^ 63
        let pickA : Bool :=
          if nan b then true else if nan a then false
          else if NV.Num.cmp .f64 .eq false a b then (if isMax then !neg a else neg a)
          else NV.Num.cmp .f64 (if isMax then .gt else .lt) false a b
        let ws ← wrP (x.o 0) l pickA
        (wrP (x.o 1) l (nan a && nan b)).map (ws ++ ·)
    else
    match i.mods[0]?.bind Float.cmpOp, i.mods.find? (["AND", "OR", "XOR"].contains ·) with
    | some c, some bop => some <| d.res d.uniform fun x l => do
        let rr ← match c with
          | some c =>
            if wide then (do let a ← f64 x l 2; let b ← f64 x l 3; pure (NV.Num.cmp .f64 c false a b))
            else (do let a ← fs x l 2; let b ← fs x l 3; pure (NV.Num.cmp f32 c ftz a b))
          | none => some (i.mods[0]? == some "T")
        let pc ← x.p l 4
        let ws ← wrP (x.o 0) l (← boolOp bop rr pc)
        (wrP (x.o 1) l (← boolOp bop (!rr) pc)).map (ws ++ ·)
    | _, _ => some d.fault
  | "FSET" =>
    -- `FSET[.BF].cmp.bop[.FTZ] Rd, a, b, Pc` (PTX `set.cmp.f32`): (a cmp b) bop Pc as 1.0 with
    -- `.BF`, as all ones without; 0 otherwise.
    let bf := i.hasMod "BF"
    let ms := i.mods.filter (· != "BF")
    match ms[0]?.bind Float.cmpOp, ms.find? (["AND", "OR", "XOR"].contains ·) with
    | some c, some bop => some <| d.res d.uniform fun x l => do
        let rr ← match c with
          | some c => (do let a ← fs x l 1; let b ← fs x l 2; pure (NV.Num.cmp f32 c ftz a b))
          | none => some (ms[0]? == some "T")
        let v ← boolOp bop rr (← x.p l 3)
        wr (x.o 0) l (if v then (if bf then 0x3F800000 else 0xFFFFFFFF) else 0)
    | _, _ => some d.fault
  | "DADD" | "DMUL" | "DFMA" =>
    let op := i.opcode
    some <| d.res d.uniform fun x l => do
      let a ← f64 x l 1; let b ← f64 x l 2
      let v ← if op == "DADD" then some (NV.Num.add .f64 r false a b)
        else if op == "DMUL" then some (NV.Num.mul .f64 r false a b)
        else (f64 x l 3).map (NV.Num.fma .f64 r false a b)
      wr64 (x.o 0) l v.toUInt64
  | "FADD2" | "FMUL2" | "FFMA2" =>
    -- `FADD2 Rd, a, b` (sm_100a; PTX `add.f32x2`): two binary32 lanes, the pair at Rd low first.
    -- A source `.F32x2.HI_LO` is its register pair as is, `.F32x2.LO_HI` the pair's halves swapped,
    -- `.F32` one binary32 for both, as an immediate is.
    let op := i.opcode
    -- `-R`/`|R|` flip/clear each lane's sign; an immediate is one binary32 for both lanes.
    let sign (m : RegMods) (v : Nat) : Nat :=
      let v := if m.abs then v % 2 ^ 31 else v
      if m.neg then v ^^^ 2 ^ 31 else v
    let lanes (x : Issue) (l n : Nat) : Option (Nat × Nat) :=
      match x.o n with
      | .reg g m =>
        let o := Operand.reg g {}
        if m.suffix.contains "F32x2" then do
          let lo ← Exec.src x.env x.v l o; let hi ← srcHi x.env x.v l o
          let (a, b) := if m.suffix.contains "LO_HI" then (hi.toNat, lo.toNat) else (lo.toNat, hi.toNat)
          some (sign m a, sign m b)
        else if m.suffix.contains "F32" then (Exec.src x.env x.v l o).map fun v => (sign m v.toNat, sign m v.toNat)
        else none
      | .imm bits => some ((bits % 2 ^ 32).toNat, (bits % 2 ^ 32).toNat)
      | _ => none
    let f (a b : Nat) (c : Nat) : UInt64 := (fin32 (match op with
      | "FADD2" => NV.Num.add f32 r ftz a b
      | "FMUL2" => NV.Num.mul f32 r ftz a b
      | _ => NV.Num.fma f32 r ftz a b c)).toUInt64
    some <| d.res d.uniform fun x l => do
      let (al, ah) ← lanes x l 1; let (bl, bh) ← lanes x l 2
      let (cl, ch) ← if op == "FFMA2" then lanes x l 3 else pure (0, 0)
      wr64 (x.o 0) l (f al bl cl ||| (f ah bh ch <<< 32))
  | "HADD2" | "HMUL2" | "HFMA2" =>
    -- Per half; `.F32` (`HADD2.F32 Rd, -RZ, Ra.H0_H0`) gives the low half's sum as binary32.
    let op := i.opcode
    let toF32 := i.hasMod "F32"
    let ftzH := ftz && hf == .f16
    let f (a b : Nat) (c : Option Nat) : Nat := match op, c with
      | "HADD2", _ => hsat (NV.Num.add hf r ftzH a b)
      | "HMUL2", _ => hsat (NV.Num.mul hf r ftzH a b)
      | _, some c => hsat (NV.Num.fma hf r ftzH a b c)
      | _, none => 0
    if toF32 then
      if op != "HADD2" then none else
      some <| d.res d.uniform fun x l => do
        let (_, a) ← pair x l 1; let (_, b) ← pair x l 2
        wr (x.o 0) l (fin32 (NV.Num.add f32 r ftz (NV.Num.cvt hf f32 .rn {} a) (NV.Num.cvt hf f32 .rn {} b)))
    else some <| d.res d.uniform fun x l => do
      let (ah, al) ← pair x l 1; let (bh, bl) ← pair x l 2
      let c ← if op == "HFMA2" then (pair x l 3).map some else pure none
      wr (x.o 0) l (packed (f ah bh (c.map (·.1))) (f al bl (c.map (·.2))))
  | "I2FP" | "I2F" =>
    -- `I2FP.F32.S32`, `I2F[.F64|.F16][.S64|.U32…][.RP…] Rd, Ra`: integer to float.
    let dst := (i.mods.findSome? Float.format).getD f32
    let (signed, width) := (i.mods.findSome? Float.intType).getD (true, 32)
    some <| d.res d.uniform fun x l => do
      let a ← x.s l 1
      let bits ← if width == 64 then (srcHi x.env x.v l (x.o 1)).map (fun h => toU64 a h |>.toNat)
        else pure a.toNat
      let v := NV.Num.ofInt dst r signed width bits
      if dst == .f64 then wr64 (x.o 0) l v.toUInt64 else wr (x.o 0) l v.toUInt32
  | "F2I" =>
    -- `F2I[.FTZ][.U32|.S64…][.TRUNC|.FLOOR|.CEIL][.NTZ] Rd, Ra`: rounded, clamped (`cvt.r?i`); a
    -- signed result narrower than the register sign-extended into it (nvcc stores `(int)(short)`
    -- of an `F2I.S16` as it is).
    let srcF := if i.hasMod "F64" then NV.Num.Format.f64 else f32
    let (signed, width) := (i.mods.findSome? Float.intType).getD (true, 32)
    some <| d.res d.uniform fun x l => do
      let a ← if srcF == .f64 then f64 x l 1 else fs x l 1
      let v := NV.Num.toInt srcF r signed width ftz a
      let v := if signed && width < 32 && v ≥ 2 ^ (width - 1) then v + 2 ^ 32 - 2 ^ width else v
      if width == 64 then wr64 (x.o 0) l v.toUInt64 else wr (x.o 0) l v.toUInt32
  | "F2IP" =>
    -- `F2IP.{S8|U8|S16|U16}.F32[.TRUNC|.FLOOR|.CEIL].NTZ Rd, a, b, c`: a and b converted as
    -- `F2I` does (saturating to the narrow type), packed as `I2IP` packs: b in the low bits, a
    -- above it, c's low bits above both (checked by tests/diff/gaps.ptx).
    match i.mods[0]?.bind Float.intType with
    | some (signed, width) =>
      if width > 16 then none else
      some <| d.res d.uniform fun x l => do
        let cv (n : Nat) : Option UInt32 := do
          pure (NV.Num.toInt f32 r signed width ftz (← fs x l n)).toUInt32
        let mask : UInt32 := (1 <<< width.toUInt32) - 1
        let c ← x.s l 3
        wr (x.o 0) l (((← cv 2) &&& mask) ||| (((← cv 1) &&& mask) <<< width.toUInt32) |||
          (if width == 16 then 0 else c <<< (2 * width).toUInt32))
    | none => none
  | "F2F" =>
    -- `F2F.dst.src Rd, Ra`: float to float, rounded.
    match (i.mods.filterMap Float.format).toList with
    | [dst, srcF] => some <| d.res d.uniform fun x l => do
        let a ← if srcF == .f64 then f64 x l 1 else if srcF == .f32 then fs x l 1
          else (pair x l 1).map (·.2)
        let v := NV.Num.cvt srcF dst r { ftz } a
        if dst == .f64 then wr64 (x.o 0) l v.toUInt64 else wr (x.o 0) l v.toUInt32
    | _ => none
  | "F2FP" =>
    -- `F2FP[.RELU].{BF16|F16}.F32.PACK_AB Rd, Ra, Rb`: Ra converted to the high half, Rb to the low
    -- (PTX `cvt.rn.f16x2.f32 d, a, b`); `F2FP.F16.{E4M3|E5M2}.UNPACK_B Rd, Ra[.H1]`: the two 8-bit
    -- values of Ra's selected half widened to a binary16 pair.
    let relu := i.hasMod "RELU"
    -- sm_80 prints the formats only when not the defaults: `F2FP.PACK_AB` is f16 from f32,
    -- `F2FP.BF16.PACK_AB` bf16 from f32.
    let fmts : List NV.Num.Format := match (i.mods.filterMap Float.format).toList with
      | [] => [.f16, .f32]
      | [.bf16] => [.bf16, .f32]
      | fs => fs
    match fmts with
    -- `F2FP.TF32.F32.PACK_B[.RZ] Rd, Rb`: one binary32 rounded to tf32 in its 32-bit container
    -- (PTX `cvt.{rn,rz}.tf32.f32` on sm_90 and sm_100a).
    | [.tf32, .f32] =>
      if !i.hasMod "PACK_B" then none else
      some <| d.res d.uniform fun x l => do
        wr (x.o 0) l (NV.Num.cvt .f32 .tf32 (if i.hasMod "RZ" then .rz else .rn) { relu } (← fs x l 1)).toUInt32
    | [dst, .f32] =>
      if !i.hasMod "PACK_AB" then none else
      some <| d.res d.uniform fun x l => do
        let c (n : Nat) : Option Nat := (fs x l n).map (NV.Num.cvt .f32 dst .rn { relu })
        wr (x.o 0) l (packed (← c 1) (← c 2))
    | [.f16, src8] =>
      if !i.hasMod "UNPACK_B" then none else
      some <| d.res d.uniform fun x l => do
        let v ← x.s l 1
        let (_, h) := Float.halves (x.o 1) v
        wr (x.o 0) l (packed (NV.Num.cvt src8 .f16 .rn {} (h.toNat / 256)) (NV.Num.cvt src8 .f16 .rn {} (h.toNat % 256)))
    | _ => none
  | "MUFU" =>
    -- The special-function unit: `NV.Num.Mufu`'s bit-accurate model (inputs and results
    -- flushed, as the `.ftz` forms). `SIN`/`COS` take the angle in turns.
    let f16 := i.hasMod "F16"
    let unit : Option (Nat → Nat) := match i.mods[0]? with
      | some "EX2" => some (if f16 then NV.Num.Approx.ex2 .f16 false else NV.Num.Approx.ex2 f32 true)
      | some "RCP" => some (NV.Num.Approx.rcp f32 true)
      | some "RSQ" => some (NV.Num.Approx.rsqrt f32 true)
      -- `RCP64H`/`RSQ64H`: on the high word of a binary64 (sign, exponent, 20 fraction bits),
      -- a seed for `div`/`rcp`/`rsqrt.f64`; `NV.Num`'s f64Hi
      -- model (correctly rounded on those bits, unmeasured), whose encodings are the binary64's
      -- with the low word zero.
      | some "RCP64H" => some fun a => NV.Num.Approx.rcp NV.Num.Approx.f64Hi true (a <<< 32) >>> 32
      | some "RSQ64H" => some fun a => NV.Num.Approx.rsqrt NV.Num.Approx.f64Hi true (a <<< 32) >>> 32
      | some "LG2" => some (NV.Num.Approx.lg2 true)
      | some "SQRT" => some (NV.Num.Approx.sqrt true)
      | some "SIN" => some (NV.Num.Approx.sinCos false true true)
      | some "COS" => some (NV.Num.Approx.sinCos true true true)
      | some "TANH" => some (NV.Num.Approx.tanh (if f16 then .f16 else f32))
      | _ => none
    unit.map fun g => d.res d.uniform fun x l => do
      if f16 then
        let (h, lo) ← pair x l 1
        wr (x.o 0) l (packed (g h) (g lo))
      else wr (x.o 0) l (g (← fs x l 1)).toUInt32
  | _ => none

/-- Warp-level matrix multiply-accumulate: the SASS forms of `mma.sync` (`mmaForm`). Operands
`Rd, Ra, Rb, Rc` hold the PTX fragments in order (the differential test checks it), so the product is the PTX part's (`NV.PTX.Sem.Matrix`), on
`NV.Num`'s tensor-core numerics. The whole warp must execute it. -/
def decodeMma (d : Dec) : Option Run := do
  let i := d.i
  if i.hasMod "SP" && i.opcode != "IMMA" then
    return d.run fun _ => .fault s!"{i.full} at {i.pc}: no published numerics for the sparse \
      tensor-core datapath of floats"
  let f ← mmaForm i
  let (x, a, b, c, dt) := (f.x, f.a, f.b, f.c, f.d)
  let regIdx (o : Operand) : Option (Option Nat) := match o with
    | .reg ⟨.r, 255⟩ _ => some none | .reg ⟨.r, r⟩ _ => some (some r) | _ => none
  let rd ← (regIdx (d.o 0)).join
  let ra ← (regIdx (d.o 1)).join
  let rb ← (regIdx (d.o 2)).join
  let rc ← regIdx (d.o 3)
  let sat := i.hasMod "SAT"
  let regs (v : View) (r : Option Nat) (cnt l : Nat) : Array UInt32 := (Array.range cnt).map fun j =>
    match r with | some r => v.getR (r + j) l | none => 0
  if let some xa := f.sparse then
    -- `IMMA.SP.{16832|16864}… Rd, Ra, Rb, Rc, Re, sel`: the compressed A, Re the metadata.
    let re ← regIdx (d.o 4)
    let .imm sel := d.o 5 | none
    return d.runWarp fun x' =>
      let gathered := (Array.range 32).foldl (fun acc l => acc ++ regs x'.v (some ra) xa.aWords l ++
        regs x'.v (some rb) x.bWords l ++ regs x'.v rc x.cWords l ++ regs x'.v re 1 l) #[]
      match NV.PTX.Sem.Matrix.computeSparse x xa a b sat sel.toNat false gathered with
      | .error e => .fault s!"{i.full} at {i.pc}: {e}"
      | .ok outs => .reg (x'.fin x'.w) ((List.range x.cWords).map fun j =>
          .lanes (rd + j) fullMask (outs.map (·[j]!)))
  if let some (format, vec) := f.scaled then
    -- `QMMA.SF`/`OMMA.SF … Rd, Ra, Rb, Rc, Rsa, Rsb, URsel`: Rsa and Rsb `scale-a-data` and
    -- `scale-b-data`, URsel the selectors, byte-id-a in bits 0–1, thread-id-a bit 2, byte-id-b
    -- bits 3–4, thread-id-b bits 5–6 (tests/diff/mma120.ptx; inferred).
    let rsa ← (regIdx (d.o 4)).join
    let rsb ← (regIdx (d.o 5)).join
    let sel : View → Nat := match d.o 6 with
      | .reg ⟨.ur, u⟩ _ => fun v => (v.getUR u).toNat
      | .imm n => fun _ => n.toNat
      | _ => fun _ => 0
    return d.runWarp fun x' =>
      let sv := sel x'.v
      let s : NV.PTX.Sem.Matrix.Scaling := { format, vec, byteA := sv % 4, threadA := (sv >>> 2) % 2,
                                             byteB := (sv >>> 3) % 4, threadB := (sv >>> 5) % 4 }
      let gathered := (Array.range 32).foldl (fun acc l => acc ++ regs x'.v (some ra) x.aWords l ++
        regs x'.v (some rb) x.bWords l ++ regs x'.v rc x.cWords l ++ regs x'.v (some rsa) 1 l ++
        regs x'.v (some rsb) 1 l) #[]
      match NV.PTX.Sem.Matrix.computeScaled d.k.sm x a b s gathered with
      | .error e => .fault s!"{i.full} at {i.pc}: {e}"
      | .ok outs => .reg (x'.fin x'.w) ((List.range x.cWords).map fun j =>
          .lanes (rd + j) fullMask (outs.map (·[j]!)))
  let cw := if c == .f64 then 64 else if c == .f16 then 16 else 32
  let lay := NV.PTX.Sem.Matrix.layout x cw
  some <| d.runWarp fun x' =>
    -- Lane `l`'s `j`-th fragment word: A's registers, then B's, then C's (`RZ`: zero).
    let word (l j : Nat) : UInt32 :=
      if j < x.aWords then x'.v.getR (ra + j) l
      else if j < x.aWords + x.bWords then x'.v.getR (rb + (j - x.aWords)) l
      else match rc with | some r => x'.v.getR (r + (j - x.aWords - x.bWords)) l | none => 0
    match NV.PTX.Sem.Matrix.computeFast d.k.sm x lay a b dt word with
    | .error e => .fault s!"{i.full} at {i.pc}: {e}"
    | .ok (some outs) => .reg (x'.fin x'.w) ((List.range outs.size).map fun j =>
        .lanes (rd + j) fullMask outs[j]!)
    | .ok none =>
    let gathered := (Array.range 32).foldl (fun acc l => acc ++ regs x'.v (some ra) x.aWords l ++
      regs x'.v (some rb) x.bWords l ++ regs x'.v rc x.cWords l) #[]
    match NV.PTX.Sem.Matrix.computeAll d.k.sm x a b c dt sat gathered with
    | .error e => .fault s!"{i.full} at {i.pc}: {e}"
    | .ok outs => .reg (x'.fin x'.w) ((List.range (outs[0]?.map (·.size) |>.getD 0)).map fun j =>
        .lanes (rd + j) fullMask (outs.map (·[j]!)))

/-- `MOVM.16.MT88 Rd, Ra` (PTX `movmatrix.sync.aligned.m8n8.trans.b16`, and column-major
`wmma.load`): the 8×8 matrix of 16-bit elements the warp holds (element (r, c)
in lane 4r + c/2, half c%2) transposed, by the whole warp.
`MOVM.U4TO8.M832 Rd, Ra` (sm_120a; sub-byte and single-bit `mma.sync`): the 8×32 matrix of 4-bit elements the warp holds as an
`m8n8k32` `.u4` fragment (element (r, c) in lane 4r + c/8, bits 4(c%8)) widened to bytes in the
8-bit `m16n8k32` fragment's rows (element (r, c) in lane 4r + (c%16)/4, register Rd + c/16,
byte c%4) (inferred; the differential test checks it against the PTX semantics). -/
def decodeMovm (d : Dec) : Option Run :=
  let i := d.i
  if i.opcode != "MOVM" then none else do
  if i.hasMod "U4TO8" && i.hasMod "M832" then
    let .reg ⟨.r, rd⟩ _ := d.o 0 | return d.fault
    let .reg ⟨.r, ra⟩ _ := d.o 1 | return d.fault
    return d.runWarp fun x =>
      let out (j : Nat) : Array UInt32 := (Array.range 32).map fun l =>
        (List.range 4).foldl (fun acc b =>
          let c := 16 * j + 4 * (l % 4) + b
          let v := (x.v.getR ra (4 * (l / 4) + c / 8) >>> (4 * (c % 8)).toUInt32) &&& 0xf
          acc ||| (v <<< (8 * b).toUInt32)) 0
      .reg (x.fin x.w) [.lanes rd fullMask (out 0), .lanes (rd + 1) fullMask (out 1)]
  if !(i.hasMod "16" && i.hasMod "MT88") then return d.fault
  let .reg ⟨.r, rd⟩ _ := d.o 0 | return d.fault
  let .reg ⟨.r, ra⟩ _ := d.o 1 | return d.fault
  return d.runWarp fun x =>
    let half (l h : Nat) : UInt32 := (x.v.getR ra l >>> (16 * h).toUInt32) &&& 0xffff
    let vals := (Array.range 32).map fun l =>
      (List.range 2).foldl (fun acc h =>
        let r := 2 * (l % 4) + h
        let c := l / 4
        acc ||| (half (4 * r + c / 2) (c % 2) <<< (16 * h).toUInt32)) 0
    .reg (x.fin x.w) (if rd == 255 then [] else [.lanes rd fullMask vals])

/-- `TEX`, `TLD`, `TLD4`, `TXQ` and `SULD`, `SUST`, `SURED` on the launch's texture and surface
objects (`NV.Tex`), in the SASS forms of PTX `tex`, `tld4`, `txq.width|height|depth`, `suld.b`, `sust.b`,
`sured.b` (inferred):
* the object's handle is `URh` (sm_90 on: `ULDC URh, c[0x0][param]`, the handle's low word) or,
  on sm_80 (no uniform operand), the word at constant bank 0 offset `4 · imm`, `imm` the second
  immediate (the handle parameter's word index);
* `TEX[.LL|.LZ] Rzw, Rxy, Ra, Rb, URh|…, slot, nD[, mask]`: float coordinates x, y, z in Ra,
  Ra+1, Ra+2 (sm_90 on), or split between Ra and Rb (sm_80); `TLD[.LZ]` the same with integer
  coordinates; `TLD4.c`
  a gather of component c. The results of the components the mask selects (default 0xf) fill
  Rxy, Rxy+1, then Rzw, Rzw+1 (`RZ`: dropped). A 1D object fetched as 2D ignores y;
* `TXQ Rzw, Rxy, Rlod, TEX_HEADER_DIMENSION, …, mask`: width, height, depth as components 0–2.
  Sampler queries are not modelled;
* `SULD.D.BA.nD[.64|.128] Rd, [Rc], …`, `SUST.D.BA.nD[.64|.128] [Rc], Rv, …`, `SURED.D.BA.nD.op
  [Rc], Rv, …`: byte coordinate x in Rc, y in Rc+1, z in Rc+2; `.IGN` reads zero and drops stores
  out of bounds, `.TRAP` faults, `.CLAMP` clamps. -/
def decodeTex (d : Dec) : Option Run :=
  let i := d.i
  let op := i.opcode
  if !["TEX", "TLD", "TLD4", "TXD", "TXQ", "SULD", "SUST", "SURED"].contains op then none else
  let ⟨dims, array, cube, mask, words⟩ := texForm i
  let handle (env : Env) (w : View) : Option UInt64 :=
    (i.operands.findSome? fun | .reg ⟨.ur, u⟩ _ => some (w.getUR u).toUInt64 | _ => none) <|>
    ((i.operands.filterMap fun | .imm v => some v | _ => none)[1]? |>.map fun v =>
      (env.const0.readWord (4 * v.toNat)).toUInt64)
  let reg (n : Nat) : Option Nat := match d.o n with | .reg ⟨.r, r⟩ _ => some r | _ => none
  let surface := op.startsWith "SU"
  let object (env : Env) (w : View) : Except String NV.Tex.Texture := do
    let some h := handle env w | throw "no object handle"
    let some tx := (if h == 0 then none else env.textures[h.toNat - 1]?)
      | throw s!"handle {h} names no object of the launch"
    if tx.surface != surface then throw s!"handle {h} names the wrong kind of object"
    if let some why := tx.invalid? then throw why
    return tx
  -- Writes of each lane's four result words to the registers the mask selects.
  let results (on : UInt32) (vals : Nat → Array UInt32) : List Write :=
    let comps := (List.range 4).filter fun c => (mask >>> c) % 2 == 1
    let dsts := [reg 1, (reg 1).map (· + 1), reg 0, (reg 0).map (· + 1)]
    comps.zipIdx.filterMap fun (c, k) => match dsts[k]?.join with
      | some r => if r ≥ 255 || (k < 2 && reg 1 == some 255) || (k ≥ 2 && reg 0 == some 255) then none
        else some (.lanes r on ((Array.range 32).map fun l => if Warp.laneOn on l then (vals l)[c]! else 0))
      | none => none
  match op with
  | "TXQ" => do
    if !(i.operands.any (· == .name "TEX_HEADER_DIMENSION")) then return d.fault
    return d.run fun x => match object x.env x.v with
      | .error e => .fault s!"{i.full} at {i.pc}: {e}"
      | .ok tx => .reg (x.fin x.w) (results x.on fun _ => #[tx.width.toUInt32, tx.height.toUInt32,
          tx.depth.toUInt32, 1])
  -- `TXD Rd2, Rd0, Ra, Rb, URh, …` (PTX `tex.grad`): `TEX`'s operands, Rb the gradients, which
  -- a texture of one mipmap level does not use (`PTX.Sem.texFetch`).
  | "TEX" | "TLD" | "TLD4" | "TXD" => do
    let some n := dims | some d.fault
    -- Depth compare (`.DC`) is not modelled: the compare function is no state of the model's.
    if i.hasMod "DC" then return d.fault
    let some ra := reg 2 | some d.fault
    let rb := reg 3
    let ints := op == "TLD"
    let gather := op == "TLD4"
    let comp := if i.hasMod "R" then 0 else if i.hasMod "G" then 1 else if i.hasMod "B" then 2 else 3
    return d.run fun x => match object x.env x.v with
      | .error e => .fault s!"{i.full} at {i.pc}: {e}"
      | .ok tx0 =>
        let lanes := Exec.lanes x.on
        -- sm_90 on: the coordinates in Ra, Ra+1, Ra+2 (an explicit LOD in Rb). sm_80 (`.SCR`):
        -- the coordinates then the LOD (`.LL`) split between Ra and Rb, Ra taking the first half
        -- rounded up (`TLD4.SCR.G Rzw, Rxy, Rx, Ry, …, 2D`; `TEX.SCR.LL …, Rxy, Rz_lod, …, 3D`).
        -- An array fetch's layer comes first (`TEX.SCR.LL …, Rlx, Ry_lod, …, ARRAY_2D`). Without `.SCR` (sm_80's `ARRAY_CUBE`) they are Ra, Ra+1, ….
        -- A cubemap fetch takes the scaled direction (`PTX.Sem.texFetch`), the face
        -- `Texture.cubeFace` chooses within the cubemap the layer names.
        -- After the coordinates come an explicit LOD (`.LL`) and the offsets (`.AOFFI`: the
        -- coordinates' signed 4-bit texel offsets in bits 0–3, 4–7, 8–11), in Rb, or split as the coordinates are.
        let lead := if array then 1 else 0
        let ll := i.hasMod "LL"
        let nvals := lead + n
        let extras := (if ll then 1 else 0) + (if i.hasMod "AOFFI" then 1 else 0)
        let all : List Nat :=
          if d.k.sm ≥ 90 || !i.hasMod "SCR" then
            (List.range nvals).map (ra + ·) ++ (List.range extras).map (rb.getD 255 + ·)
          else
            let vals := nvals + extras
            let ka := (vals + 1) / 2
            (List.range vals).map fun j => if j < ka then ra + j else rb.getD 255 + (j - ka)
        let regs := all.take nvals
        let offReg := if i.hasMod "AOFFI" then all[nvals + (if ll then 1 else 0)]? else none
        let coordRegs := regs.drop lead
        let tx := if ints then { tx0 with filter := .point, normalizedCoords := false } else
          if gather then { tx0 with filter := .linear } else tx0
        let fpOf (l : Nat) : Except String (List (Option UInt64 × Nat)) := do
          if tx.cube != cube then throw "a cubemap fetch from a plain texture, or the reverse"
          if cube then
            let (face, u, v) := NV.Tex.Texture.cubeFace (x.v.getR coordRegs[0]! l)
              (x.v.getR coordRegs[1]! l) (x.v.getR coordRegs[2]! l)
            let idx := if array then (x.v.getR regs[0]! l).toNat else 0
            let tf := { (← NV.Tex.Texture.layer tx (6 * idx + face)) with normalizedCoords := true }
            return NV.Tex.footprint tf #[← NV.Tex.coordOf u, ← NV.Tex.coordOf v]
          let tx ← if array then NV.Tex.Texture.layer tx (x.v.getR regs[0]! l).toNat else pure tx
          let cs ← (coordRegs.take tx.dims).toArray.mapM fun r =>
            if ints then pure ({ n := (x.v.getR r l).toInt32.toInt, e := 0 } : NV.Tex.Dy)
            else NV.Tex.coordOf (x.v.getR r l)
          let offs : Array Int := match offReg with
            | some r => (Array.range 3).map fun k =>
                let v := ((x.v.getR r l).toNat >>> (4 * k)) % 16
                if v ≥ 8 then (v : Int) - 16 else v
            | none => #[]
          return NV.Tex.footprint tx cs offs
        match lanes.mapM fun l => do pure (l, ← fpOf l) with
        | .error e => .fault s!"{i.full} at {i.pc}: {e}"
        | .ok fps =>
          let k := (fps.head?.map (·.2.length)).getD 1
          let addrs := (Array.range 32).flatMap fun l => match fps.lookup l with
            | some fp => fp.toArray.map (·.1)
            | none => Array.replicate k none
          let req : Req := { space := .global, op := .ld, bytes := tx.texelBytes, addrs }
          let wpt := (tx.texelBytes + 3) / 4
          .mem (fun _ => req) (x.fin x.w) fun resp =>
            results x.on fun l =>
              let mine := resp.extract (l * k * wpt) ((l + 1) * k * wpt)
              let fp := (fps.lookup l).getD []
              if !gather then NV.Tex.combine tx fp mine else
                (NV.Tex.gatherOrder.map fun j =>
                  (NV.Tex.combine { tx with filter := .point } [(fp[j]!.1, 1)]
                    (mine.extract (j * wpt) (j * wpt + wpt)))[comp]!).toArray
  | _ => do
    -- Surfaces.
    let some n := dims | some d.fault
    if !(i.hasMod "D" && i.hasMod "BA") then return d.fault
    if i.hasMod "CUBE" then return d.fault
    -- `.U8`/`.S8`, `.U16`/`.S16`: one 1- or 2-byte texel (`sust.b.b8`, `.b16`).
    let bytes := if i.hasMod "U8" || i.hasMod "S8" then 1
      else if i.hasMod "U16" || i.hasMod "S16" then 2 else 4 * words
    let mode : NV.Tex.Clamp := if i.hasMod "TRAP" then .trap else if i.hasMod "CLAMP" then .clamp else .zero
    let (co, vo) := if op == "SULD" then (d.o 1, d.o 0) else (d.o 0, d.o 1)
    let some rc := (match co with | .mem (some ⟨.r, r⟩) _ none none 0 => some r | _ => none) | some d.fault
    let some rv := (match vo with | .reg ⟨.r, r⟩ _ => some r | _ => none) | some d.fault
    let red : Option AtomOp := if i.hasMod "ADD" then some .add else if i.hasMod "MIN" then some .min
      else if i.hasMod "MAX" then some .max else if i.hasMod "AND" then some .and
      else if i.hasMod "OR" then some .or else none
    let rty : AtomType := if words == 2 then (if i.hasMod "S64" then .s64 else .u64)
      else if i.hasMod "S32" then .s32 else .u32
    let wop : MemOp := match red with | some o => .red o rty | none => .st
    if op == "SURED" && red.isNone then return d.fault
    return d.run fun x => match object x.env x.v with
      | .error e => .fault s!"{i.full} at {i.pc}: {e}"
      | .ok tx =>
        let lanes := Exec.lanes x.on
        -- An array access's layer follows the coordinates (`SULD.D.BA.2D_ARRAY … [Rxyl]`).
        let at_ (l : Nat) : Except String (Option UInt64) := do
          let tx ← if array then NV.Tex.Texture.layer tx (x.v.getR (rc + n) l).toNat else pure tx
          let c (j : Nat) : Int := if j < n then (x.v.getR (rc + j) l).toInt32.toInt else 0
          NV.Tex.surfAddr tx mode bytes (c 0) (c 1) (c 2)
        match lanes.mapM fun l => do pure (l, ← at_ l) with
        | .error e => .fault s!"{i.full} at {i.pc}: {e}"
        | .ok as =>
          let addrs := (Array.range 32).map fun l => (as.lookup l).join
          let data := (Array.range 32).flatMap fun l => (Array.range words).map fun j =>
            let v := x.v.getR (rv + j) l
            if bytes == 1 then v &&& 0xff else if bytes == 2 then v &&& 0xffff else v
          match op with
          | "SULD" =>
            let req : Req := { space := .global, op := .ld, bytes, addrs }
            .mem (fun _ => req) (x.fin x.w) fun resp =>
              if rv == 255 then [] else (List.range words).map fun j =>
                .lanes (rv + j) x.on ((Array.range 32).map fun l =>
                  let v := resp[l * words + j]!
                  -- A narrow load is zero- or sign-extended.
                  if bytes == 1 then (if i.hasMod "S8" then (v.toUInt8.toInt8.toInt32.toUInt32) else v &&& 0xff)
                  else if bytes == 2 then (if i.hasMod "S16" then (v.toUInt16.toInt16.toInt32.toUInt32) else v &&& 0xffff)
                  else v)
          | _ =>
            let req : Req := { space := .global, op := wop, bytes, addrs, data }
            .mem (fun _ => req) (x.fin x.w) fun _ => []

/-- Loads, stores and atomics on global, shared and local memory, and constant loads. A memory
instruction reads its sources after issue (when its read counter is released), so its request is
a function of the warp's state at that time. `LDS`/`STS`/`ATOMS` address the executing CTA's
shared memory: the low 24 bits of the window address (`Exec.sharedAddr`). -/
def decodeMem (d : Dec) : Option Run :=
  let i := d.i
  let (bytes, sx) := width i
  let n := words bytes
  -- With `.E` a global address register is a 64-bit pair even where the listing omits `.64`
  -- (`ATOMG.E.CAS … [R8+0x4]`).
  let wideE (shared : Bool) : Operand → Operand
    | .mem b w u dsc off => .mem b (w || !shared && i.hasMod "E") u dsc off
    | a => a
  -- Per-lane addresses of address operand `a` in warp state `w'`, for the executing lanes `on`.
  let addrs (shared : Bool) (a : Operand) (on : UInt32) (w' : View) : Array (Option UInt64) :=
    (List.range 32).toArray.map fun l =>
      if Warp.laneOn on l then (addr w' l a).map fun v => if shared then v % (1 <<< 24) else v
      else none
  -- Lane-major data words: per lane, `count` words of each register operand in `ds` (inactive
  -- lanes zero).
  let data (ds : List Operand) (count : Nat) (on : UInt32) (w' : View) : Array UInt32 :=
    (List.range 32).foldl (fun acc l => ds.foldl (fun acc o => (List.range count).foldl (fun acc j =>
      acc.push (if Warp.laneOn on l then (match o with
        | .reg ⟨.r, r⟩ _ => w'.getR (r + j) l
        | _ => 0) else 0)) acc) acc) #[]
  -- Writes of each executing lane's `count` response words to the registers from `o`.
  let load (o : Operand) (count : Nat) (ext : UInt32 → UInt32) (on : UInt32) (resp : Resp) :
      List Write :=
    match o with
    | .reg ⟨.r, 255⟩ _ => []
    | .reg ⟨.r, r⟩ _ => (List.range count).map fun j =>
        .lanes (r + j) on ((Array.range 32).map fun l => ext resp[l * count + j]!)
    | _ => []
  let extend (v : UInt32) : UInt32 :=
    if bytes == 1 && sx then v.toUInt8.toInt8.toInt32.toUInt32
    else if bytes == 2 && sx then v.toUInt16.toInt16.toInt32.toUInt32 else v
  let isAddr : Operand → Bool := fun | .mem .. | .scaled (.mem ..) _ => true | _ => false
  let isReg : Operand → Bool := fun | .reg ⟨.r, _⟩ _ => true | _ => false
  let space := if i.opcode.endsWith "G" then some Space.global
    else if i.opcode.endsWith "S" then some .shared else if i.opcode.endsWith "L" then some .local
    else none
  let strong := if i.hasMod "STRONG" then
      (if i.hasMod "GPU" then some NV.PTX.Scope.gpu else if i.hasMod "SYS" then some .sys
       else if i.hasMod "SM" then some .cta else none) else none
  let sem := if strong.isSome then Sem.relaxed else .weak
  let scope := strong.getD .thread
  -- `LDG.E.ENL2.256 Rhi, Rlo, [a]` and `STG.E.ENL2.256 [a], Rlo, Rhi` (sm_100a, PTX `.v8.b32`):
  -- 32 bytes a lane in two register quads, bytes 0–15 in the load's second operand and the
  -- store's first. The request is 64 pieces of 16 bytes, each lane's two
  -- in order (the memory model's widths stop at 16 bytes), so its words are lane-major as ever.
  let enl2 := i.hasMod "ENL2"
  let halves (a : Operand) (on : UInt32) (w' : View) : Array (Option UInt64) :=
    (addrs false a on w').flatMap fun x => #[x, x.map (· + 16)]
  match i.opcode with
  | "LDG" =>
    if enl2 then
      if bytes != 32 || !isAddr (d.o 2) || !isReg (d.o 0) || !isReg (d.o 1) then some d.fault else
      let a := wideE false (d.o 2)
      let req (on : UInt32) (w' : View) : Req :=
        { space := .global, op := .ld, bytes := 16, sem, scope, addrs := halves a on w' }
      let quad (o : Operand) (from_ : Nat) (on : UInt32) (resp : Resp) : List Write :=
        match o with
        | .reg ⟨.r, 255⟩ _ => []
        | .reg ⟨.r, r⟩ _ => (List.range 4).map fun j =>
            .lanes (r + j) on ((Array.range 32).map fun l => resp[l * 8 + from_ + j]!)
        | _ => []
      some <| d.run fun x =>
        .mem (req x.on) (x.fin x.w) fun resp => quad (d.o 1) 0 x.on resp ++ quad (d.o 0) 4 x.on resp
    else
    let a := wideE false (d.o 1)
    if !isAddr (d.o 1) || !isReg (d.o 0) then some d.fault else
    let req (on : UInt32) (w' : View) : Req :=
      { space := .global, op := .ld, bytes, sem, scope, addrs := addrs false a on w' }
    some <| d.run fun x =>
      .mem (req x.on) (x.fin x.w) (load (d.o 0) n extend x.on)
  | "STG" =>
    if enl2 then
      if bytes != 32 || !isAddr (d.o 0) || !isReg (d.o 1) || !isReg (d.o 2) then some d.fault else
      let a := wideE false (d.o 0)
      let req (on : UInt32) (w' : View) : Req :=
        { space := .global, op := .st, bytes := 16, sem, scope, addrs := halves a on w',
          data := data [d.o 1, d.o 2] 4 on w' }
      some <| d.run fun x => .mem (req x.on) (x.fin x.w) fun _ => []
    else
    if !isAddr (d.o 0) || !isReg (d.o 1) then some d.fault else
    let a := wideE false (d.o 0)
    let req (on : UInt32) (w' : View) : Req :=
      { space := .global, op := .st, bytes, sem, scope, addrs := addrs false a on w',
        data := data [d.o 1] n on w' }
    some <| d.run fun x =>
      .mem (req x.on) (x.fin x.w) fun _ => []
  | "LDS" | "LDL" => do
    let sp ← space
    if !isAddr (d.o 1) || !isReg (d.o 0) then some d.fault else
    let a := wideE (sp == .shared) (d.o 1)
    let req (on : UInt32) (w' : View) : Req :=
      { space := sp, op := .ld, bytes, sem, scope, addrs := addrs (sp == .shared) a on w' }
    some <| d.run fun x =>
      .mem (req x.on) (x.fin x.w) (load (d.o 0) n extend x.on)
  | "STS" | "STL" => do
    let sp ← space
    if !isAddr (d.o 0) || !isReg (d.o 1) then some d.fault else
    let a := wideE (sp == .shared) (d.o 0)
    let req (on : UInt32) (w' : View) : Req :=
      { space := sp, op := .st, bytes, sem, scope, addrs := addrs (sp == .shared) a on w',
        data := data [d.o 1] n on w' }
    some <| d.run fun x =>
      .mem (req x.on) (x.fin x.w) fun _ => []
  | "LD" | "ST" =>
    -- Generic loads and stores (`LD.E`, `ST.E`): each lane's address resolved through the
    -- windows (`genericTarget`) at issue, from the address registers the read counter keeps
    -- unchanged until they are read. Lanes into the CTA's shared memory and one other space
    -- (a select between a shared and a stack object) are one request (`Req.sharedPart`); other
    -- mixes are not modelled.
    let ld := i.opcode == "LD"
    let (ao, vo) := if ld then (d.o 1, d.o 0) else (d.o 0, d.o 1)
    if !isAddr ao || !isReg vo then some d.fault else
    let a := wideE false ao
    some <| d.run fun x =>
      let targets := (List.range 32).filterMap fun l =>
        if Warp.laneOn x.on l then (addr x.v l a).map (genericTarget x.env d.k.sm) else none
      let others := targets.filter (·.1 != .shared)
      match targets, others with
      | [], _ => .next (x.fin x.w)
      | (sp0, _) :: _, rest =>
        let sp := (rest.head?.map (·.1)).getD sp0
        if rest.any (·.1 != sp) then
          .fault s!"{i.full} at {i.pc}: lanes address different state spaces (not modelled)"
        else
          let mixed := sp != .shared && targets.any (·.1 == .shared)
          let at_ (w' : View) (l : Nat) : Option (Space × UInt64) :=
            if Warp.laneOn x.on l then (addr w' l a).map (genericTarget x.env d.k.sm) else none
          let req (w' : View) : Req :=
            { space := sp, op := if ld then .ld else .st, bytes, sem, scope,
              addrs := (List.range 32).toArray.map fun l => match at_ w' l with
                | some (s, b) => if s == sp then some b else none
                | none => none,
              sharedPart := if !mixed then #[] else (List.range 32).toArray.map fun l => match at_ w' l with
                | some (.shared, b) => some b
                | _ => none,
              data := if ld then #[] else data [vo] n x.on w' }
          .mem req (x.fin x.w)
            (if ld then load vo n extend x.on else fun _ => [])
  | "LDGMC" => do
    -- `LDGMC.E.op.type[.STRONG.scope] Rd, [addr]` (PTX `multimem.ld_reduce`): the reduction over
    -- the multicast object's copies, which for the one device the model has is its own global
    -- location's value (as `PTX.Sem` reads it): a load.
    if !isAddr (d.o 1) || !isReg (d.o 0) then some d.fault else
    let a := wideE false (d.o 1)
    let req (on : UInt32) (w' : View) : Req :=
      { space := .global, op := .ld, bytes, sem, scope, addrs := addrs false a on w' }
    some <| d.run fun x => .mem (req x.on) (x.fin x.w) (load (d.o 0) n extend x.on)
  | "QSPC" =>
    -- `QSPC.E.{G|S|L} Pd, RZ, [addr]` (PTX `isspacep.{global|shared|local}`): whether the
    -- generic address lies in the space's window (`genericTarget`; shared: the executing CTA's
    -- own). The second destination is RZ in every listing seen; others fault.
    let target := if i.hasMod "G" then some Space.global else if i.hasMod "S" then some .shared
      else if i.hasMod "L" then some .local else none
    match target, d.o 1 with
    | some sp, .reg ⟨.r, 255⟩ _ =>
      if !isAddr (d.o 2) then some d.fault else
      let a := wideE false (d.o 2)
      some <| d.res d.uniform fun x l => do
        let (s', _) := genericTarget x.env d.k.sm (← addr x.v l a)
        wrP (x.o 0) l (s' == sp)
    | _, _ => some d.fault
  | "ATOMG" | "ATOMS" | "RED" | "REDG" | "ATOM" => do
    -- `ATOMG.E.op[.type][.STRONG.scope] Pd, Rd, [addr], Rb[, Rc]`, `ATOMS.op Rd, [addr], Rb`,
    -- `REDG.E.op[.type] [addr], Rb`.
    let red := i.opcode.startsWith "RED"
    let sp := if i.opcode == "ATOMS" then Space.shared else .global
    -- The address is the first memory operand: `ATOMS.CAST.SPIN P1, [R8], R2, R3` (sm_120a)
    -- has a predicate destination where sm_90a's form has a register.
    let a0 := ((List.range i.operands.size).find? (isAddr ∘ d.o)).getD (if red then 0 else 1)
    if !isAddr (d.o a0) then some d.fault else
    -- `ATOMS.CAST.SPIN Rd, [addr], cmp, new`: compare and store, Rd = 1 when it stored, else 0.
    let spin := i.hasMod "CAST"
    let aop ← if spin then some AtomOp.cas else
      match i.mods.find? (["ADD", "MIN", "MAX", "INC", "DEC", "AND", "OR", "XOR", "EXCH",
      "CAS"].contains ·) with
      | some "ADD" => some AtomOp.add | some "MIN" => some .min | some "MAX" => some .max
      | some "INC" => some .inc | some "DEC" => some .dec | some "AND" => some .and
      | some "OR" => some .or | some "XOR" => some .xor | some "EXCH" => some .exch
      | some "CAS" => some .cas | _ => none
    let (ty, bytes) := atomForm i
    let k := words bytes
    let ds := if aop == .cas then [d.o (a0 + 1), d.o (a0 + 2)] else [d.o (a0 + 1)]
    let mop := if red then MemOp.red aop ty else .atom aop ty
    let scope := strong.getD (if sp == .global then .gpu else .cta)
    let a := wideE (sp == .shared) (d.o a0)
    let req (on : UInt32) (w' : View) : Req :=
      { space := sp, op := mop, bytes, sem := .relaxed, scope, addrs := addrs (sp == .shared) a on w',
        data := data ds k on w' }
    some <| d.run fun x =>
      let wb : Resp → List Write :=
        if red then fun _ => []
        else if spin then
          -- The comparison values as read at issue.
          let cmpv := data [d.o (a0 + 1)] 1 x.on x.v
          let stored (resp : Resp) : UInt32 := (List.range 32).foldl (fun m l =>
            if resp[l]! == cmpv[l]! then m ||| ((1 : UInt32) <<< l.toUInt32) else m) 0
          match d.o (a0 - 1) with
          | .reg ⟨.p, pi⟩ _ => fun resp => [.pbits pi x.on (stored resp)]
          | o => fun resp => load o 1 id x.on
            ((Array.range 32).map fun l => if resp[l]! == cmpv[l]! then 1 else 0)
        else load (d.o (a0 - 1)) k id x.on
      .mem (req x.on) (x.fin x.w) wb
  | "LDC" | "ULDC" | "LDCU" =>
    -- Constant reads are register results: `LDC` releases a counter like a load, and so does
    -- `LDCU` (sm_100 on: into uniform registers through the load path). Bank 0 is the launch's
    -- (`Env.const0`); other banks are the kernel's own (`Kernel.banks`). `.64` and `.128` read two
    -- and four words.
    let u := i.opcode != "LDC"
    let n := if i.hasMod "128" then 4 else if i.hasMod "64" then 2 else 1
    -- `.U8`/`.S8`/`.U16`/`.S16`: the byte or halfword at the address, zero- or sign-extended.
    let sub : Option (Nat × Bool) :=
      if i.hasMod "U8" then some (8, false) else if i.hasMod "S8" then some (8, true)
      else if i.hasMod "U16" then some (16, false) else if i.hasMod "S16" then some (16, true)
      else none
    let narrow (word : Nat → UInt32) (a : Nat) : UInt32 :=
      match sub with
      | none => word a
      | some (bits, signed) =>
        let v := (word (a - a % 4) >>> (8 * (a % 4)).toUInt32) &&& ((1 <<< bits.toUInt32) - 1)
        if signed && v >>> (bits - 1).toUInt32 == 1 then v ||| ~~~((1 <<< bits.toUInt32) - 1) else v
    let dest (x : Issue) (l : Nat) (ws : List UInt32) : Option (List Write) :=
      match x.o 0 with
      | .reg ⟨.ur, 255⟩ _ | .reg ⟨.r, 255⟩ _ => some []
      | .reg ⟨.ur, r⟩ _ => some (ws.zipIdx.map fun (v, j) => .ur (r + j) v)
      | .reg ⟨.r, r⟩ _ => some (ws.zipIdx.map fun (v, j) => .r (r + j) (lane32 l) v)
      | _ => none
    match d.o 1 with
    | .cbank b base off =>
      let addr (x : Issue) (l : Nat) : Int :=
        (match base with
          | some r => ((src x.env x.v l (.reg r {})).map UInt32.toNat).getD 0
          | none => 0) + off
      if b == 0 then
        some <| d.res u fun x l =>
          let a := (addr x l).toNat
          dest x l ((List.range n).map fun j => narrow x.env.const0.readWord (a + 4 * j))
      else match d.k.banks.find? (·.1 == b) with
        | none => some d.fault
        | some (_, ws) =>
          let word (a : Nat) : UInt32 := if a % 4 == 0 then ws[a / 4]?.getD 0 else 0
          some <| d.res u fun x l =>
            dest x l ((List.range n).map fun (j : Nat) => narrow word ((addr x l).toNat + 4 * j))
    | _ => some d.fault
  | "LDSM" | "STSM" =>
    -- PTX `ldmatrix`/`stmatrix` (`matMove`, §9.7.16.5.15–16): each lane giving a row's address
    -- moves that row's 16 bytes, as one request; the exchange between lanes is the semantics'
    -- (`MatMove.frag`, `MatMove.rowsOf`, shared with the PTX part).
    let load := i.opcode == "LDSM"
    let (a, rv) := if load then (d.o 1, d.o 0) else (d.o 0, d.o 1)
    match rv, matMove i with
    | .reg ⟨.r, r⟩ _, some mv =>
      if !isAddr a then some d.fault else
      let regs := mv.num * mv.regs
      let rowAddrs (w' : View) : Array (Option UInt64) := (List.range 32).toArray.map fun l =>
        if l < mv.rows * mv.num then (addr w' l a).map (· % (1 <<< 24)) else none
      some <| d.runWarp fun x =>
        if load then
          .mem (fun w' => { space := .shared, op := .ld, bytes := 16, sem := .weak,
                            scope := .thread, addrs := rowAddrs w' }) (x.fin x.w) fun rows =>
            (List.range regs).map fun j => .lanes (r + j) fullMask
              ((Array.range 32).map fun l => mv.frag (fun τ k => rows[4 * τ + k]!) l j)
        else
          .mem (fun w' =>
              { space := .shared, op := .st, bytes := 16, sem := .weak, scope := .thread,
                addrs := rowAddrs w',
                data := mv.rowsOf fun l j => w'.getR (r + j) l })
            (x.fin x.w) fun _ => []
    | _, _ => some d.fault
  | "MEMBAR" =>
    let scope := if i.hasMod "GPU" then NV.PTX.Scope.gpu else if i.hasMod "SYS" then .sys else .cta
    some <| d.run fun x => .sync (.fence true scope) (x.fin x.w)
  -- `FENCE.VIEW.ASYNC.S` (`fence.proxy.async`): a proxy fence, which waits for the warp's earlier
  -- accesses as a fence does (`SyncParams.fenceCycles` its own cost).
  | "FENCE" =>
    if i.hasMod "VIEW" && i.hasMod "ASYNC" && i.hasMod "S" then
      some <| d.run fun x => .sync (.proxyFence .async) (x.fin x.w)
    else some <| d.run fun x => .next (x.fin x.w)
  | "ERRBAR" | "CGAERRBAR" | "CCTL" => some <| d.run fun x => .next (x.fin x.w)
  | _ => none

/-- A decoded instruction. A structure rather than a bare `Run` so that decoding happens once:
a definition whose type is a function type is compiled to take the run's arguments too, and
would decode again at every execution. -/
structure Op where
  out : Run
  /-- What it reads, writes back and sets at issue (`NV.SASS.footprint`). -/
  fp : Footprint
  /-- Only convergence barriers are set at issue (`footprint_setsB`). -/
  setsB : fp.SetsB

instance : Inhabited Op := ⟨⟨fun _ _ _ _ => .fault "no instruction", {}, Footprint.setsB_empty⟩⟩

namespace Op

/-- The instruction's outcome from warp `w`: what its semantics makes of its reads
(`View.of fp.reads w`) and its control state (`Warp.blank w`). -/
def outOf (op : Op) (env : Env) (w : Warp) (m : UInt32) : Out :=
  op.out env (View.of op.fp.reads w) w.blank m

/-- The instruction's effect: its outcome confined to its footprint (`Footprint.effect`). -/
def run (op : Op) (env : Env) (w : Warp) (m : UInt32) : Effect Warp :=
  op.fp.effect w (op.outOf env w m)

/-- From any two warp states of one shape that agree on the reads and the control state, the same
outcome
(`Footprint.SameOutcome`). A theorem for every instruction (`Op.readsWithin`): the semantics
sees only the reads and the control state. -/
def ReadsWithin (op : Op) : Prop :=
  ∀ env w w' m, w.ctrl = w'.ctrl → w.shape = w'.shape → Warp.AgreeOn op.fp.reads w w' →
    op.fp.SameOutcome w' (op.run env w m) (op.run env w' m)

theorem readsWithin (op : Op) : op.ReadsWithin := by
  intro env w w' m hc hs ha
  have hv : View.of op.fp.reads w = View.of op.fp.reads w' := View.of_congr ha
  have hb : w.blank = w'.blank := Warp.blank_congr hc
  unfold run outOf
  rw [hv, hb]
  exact op.fp.effect_same op.setsB w w' hs ha _

/-- Every effect changes at issue only the control state and the sets, and writes back only the
writes (`Footprint.Within`). -/
def Confined (op : Op) : Prop := ∀ env w m, op.fp.Within w (op.run env w m)

theorem confined (op : Op) : op.Confined := fun _ w _ => op.fp.effect_within w _

/-- Every writeback keeps the control state and the register files' sizes, and on the write set
depends only on the write set (`WbLocal`). Every outcome's writebacks are data, so it holds for every
instruction (`Op.wbLocal`). -/
def WbLocal (op : Op) : Prop := ∀ env w m, effectWbLocal op.fp.writes (op.run env w m)

/-- The state an instruction continues with keeps the issuer's register-file sizes. -/
def Shape (op : Op) : Prop := ∀ env w m, effectShape w.shape (op.run env w m)

theorem shape (op : Op) : op.Shape := fun _ w _ => op.fp.effect_shape w _

theorem wbLocal (op : Op) : op.WbLocal := fun _ w _ => op.fp.effect_wbLocal w _

end Op

/-- The general semantics of instruction index `pc` of `k`, its form resolved once. -/
def decodeGeneral (k : Kernel) (pc : Nat) : Op :=
  match k.code[pc]? with
  | none => ⟨fun _ _ _ _ => .fault s!"pc {pc} outside the code", {}, Footprint.setsB_empty⟩
  | some i =>
    let d : Dec := { k, pc, i }
    let fp := footprint i
    match i.opcode with
    | "BAR" =>
      if i.hasMod "SYNC" || i.hasMod "ARV" then
        -- `BAR.SYNC a[, b]`: barrier `a` and thread count `b`, immediates or registers (`BAR.SYNC
        -- R2, 0x80`), whose value the executing threads share
        -- (PTX ISA §9.7.13.4: `a` and `b` must be the same in every thread of a warp); the lowest
        -- executing lane's is taken.
        let arrive := i.hasMod "ARV"
        ⟨d.run fun x =>
          -- A warp whose guard holds in no lane does not arrive (FA3's `@P1 BAR.ARV 0x9`, taken by
          -- the warpgroups after the first only).
          if x.on == 0 then .next (x.fin x.w) else
          let l := Exec.lowLane x.on
          let val (n : Nat) : Option Nat := match x.o n with
            | .imm b => some b.toNat
            | _ => (x.s l n).map (·.toNat)
          match val 0, (if i.operands.size > 1 then (val 1).map some else some none) with
          | some id, some count =>
            if id ≥ 16 then .fault s!"{x.bad}: barrier {id}" else
            .sync (.bar id count arrive) (x.fin x.w)
          | _, _ => .fault x.bad, fp, footprint_setsB i⟩
      else if i.hasMod "RED" then
        -- `BAR.RED.{POPC,AND,OR}.DEFER_BLOCKING id[, count], Pc` (PTX `bar.red`): a barrier whose
        -- answer, over the participating threads' Pc, `B2R.RESULT` reads. Modelled as blocking at the barrier: what `DEFER_BLOCKING` lets the warp do
        -- before `B2R` is not documented.
        let op : Option BarRedOp :=
          if i.hasMod "POPC" then some .popc else if i.hasMod "AND" then some .and
          else if i.hasMod "OR" then some .or else none
        let n := i.operands.size
        match op, d.o 0, (if n == 3 then some (d.o 1) else none) with
        | some op, .imm id, cnt =>
          let count := match cnt with | some (.imm c) => some (some c.toNat) | none => some none | _ => none
          match count with
          | some count =>
            ⟨d.run fun x =>
              let contrib := (Array.range 32).map fun l =>
                if Warp.laneOn x.on l && (x.p l (n - 1)).getD false then (1 : UInt32) else 0
              .collect (.barRed id.toNat count op) x.on contrib (x.fin x.w)
                fun resp => [.bar (resp[0]?.getD 0)], fp, footprint_setsB i⟩
          | none => ⟨d.fault, fp, footprint_setsB i⟩
        | _, _, _ => ⟨d.fault, fp, footprint_setsB i⟩
      else ⟨d.fault, fp, footprint_setsB i⟩
    | _ =>
      match decodeInt d <|> decodeCtrl d <|> decodeMem d <|> decodeFloat d <|> decodeMma d <|>
          decodeTex d <|> decodeMovm d with
      | some r => ⟨r, fp, footprint_setsB i⟩
      | none => ⟨fun env v w m => match execAsync k env v w pc m i with
          | some o => o
          | none => .fault s!"{i.full}: no semantics yet (pc {i.pc})", fp, footprint_setsB i⟩

/-- `i` with its operands from the module's constant banks (`c[0x2][…]`, `c[0x3][…]`: fixed for
the kernel, `Kernel.banks`) read as immediates holding the word and the next one above it; bank
0 is the launch's and stays. Constant loads (`LDC`, `ULDC`, `LDCU`) keep theirs: they read by
address. -/
def resolveBanks (k : Kernel) (i : Instr) : Instr :=
  if ["LDC", "ULDC", "LDCU"].contains i.opcode then i else
  let word (b : Nat) (off : Int) : Option UInt32 := do
    let (_, ws) ← k.banks.find? (·.1 == b)
    if off % 4 != 0 || off < 0 then none else ws[(off / 4).toNat]?
  let rec resolve : Operand → Operand
    | .cbank b none off =>
      if b == 0 then .cbank b none off else
      match word b off with
      | some lo => .imm (lo.toUInt64 ||| ((word b (off + 4)).getD 0).toUInt64 <<< 32)
      | none => .cbank b none off
    | .neg o => .neg (resolve o)
    | .not o => .not (resolve o)
    | o => o
  { i with operands := i.operands.map resolve }

/-- The semantics of instruction index `pc` of `k`: its hot form compiled (`NV.SASS.Fast`), or
the general definition. -/
def decodeAt (k : Kernel) (pc : Nat) : Op :=
  match k.code[pc]? with
  | some i =>
    -- A register past the warp's file is a reading error, not a value: say so.
    let file := max k.regs k.regFile
    let past := i.operands.findSome? fun
      | .reg ⟨.r, n⟩ _ => if n != 255 && n ≥ file then some n else none
      | _ => none
    if let some n := past then
      ⟨fun _ _ _ _ => .fault s!"{i.full} at {i.pc}: R{n} is past the warp's {file} registers",
        footprint i, footprint_setsB i⟩
    else
    let i' := resolveBanks k i
    let k' := if i' == i then k else { k with code := k.code.set! pc i' }
    let fp := footprint i
    match Fast.decodeFast fp.reads i' pc with
    | some r => ⟨r, fp, footprint_setsB i⟩
    | none => decodeGeneral k' pc
  | none => decodeGeneral k pc

/-- Every instruction of `k` decoded: the engine runs instruction `pc` as `(decode k)[pc]`. -/
def decode (k : Kernel) : Array Op := (Array.range k.code.size).map fun pc => decodeAt k pc

/-- Execute instruction index `pc` of `k` for the lanes `m` (the lanes at `pc`). -/
def exec (k : Kernel) (env : Env) (w : Warp) (pc : Nat) (m : UInt32) : Effect Warp :=
  (decodeAt k pc).run env w m

end NV.SASS
