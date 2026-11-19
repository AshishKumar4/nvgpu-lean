import NV.SASS.Float
import NV.Num.F32
import NV.Num.MufuWord

/-!
# The hot instruction forms, compiled at decode

The integer, move, compare, float and memory forms that make up most executed code, compiled when
the kernel is decoded into loops over lane masks: operands are read by functions resolved once
(`Exec.val?`, `Exec.addr?`), results land in arrays and bit masks, and predicates combine
word-wide. Each form means exactly what the general definition in `NV.SASS.Sem` means for it;
`decodeFast` returns `none` for every other form, which then takes the general definition.
`lake exe sass-sem` checks the two agree on every instruction of the test listings, from random
warp states.
-/

namespace NV.SASS.Fast

open NV Exec

/-- What compiling a form needs: the instruction, its code index and its footprint's reads
(where each source sits in the `View`). -/
structure Form where
  i : Instr
  pc : Nat
  reads : Array Loc

namespace Form

def o (f : Form) (n : Nat) : Operand := f.i.operands[n]?.getD (.name "?")

/-- A run: the executing lanes (guarded; one lane for the uniform datapath) to an effect. -/
def run (f : Form) (u : Bool) (g : Env → View → UInt32 → Warp → Out) : Run :=
  let i := f.i; let pc := f.pc
  fun env v w m =>
    let on := guarded v m i
    g env v (if u then firstLane on else on) (w.advance m pc)

/-- One 32-bit result in operand 0. -/
def out32 (f : Form) (u : Bool) (v : Val) : Option Run := do
  let wb ← write32? (f.o 0)
  some <| f.run u fun env w on s => .reg s (wb on (valsOf (v env w) on))

/-- One 64-bit result in the register pair at operand 0. -/
def out64 (f : Form) (u : Bool) (v : Env → View → Nat → UInt64) : Option Run := do
  let wb ← write64? (f.o 0)
  some <| f.run u fun env w on s =>
    let vs := Id.run do
      let mut a : Array UInt64 := Array.mkEmpty 32
      for l in [0:32] do a := a.push (if (on >>> l.toUInt32) &&& 1 == 1 then v env w l else 0)
      return a
    .reg s (wb on (vs.map (·.toUInt32)) (vs.map fun (x : UInt64) => (x >>> 32).toUInt32))

end Form

/-- An integer source as a carry-arithmetic term (`Exec.srcN`): `-x` is `2^32 - x`. -/
def term? (reads : Array Loc) (o : Operand) : Option (Env → View → Nat → UInt64) :=
  match o with
  | .reg r m =>
    (val? reads (.reg r { m with neg := false })).map fun v =>
      if m.neg && !m.not then fun env w l => 0x100000000 - (v env w l).toUInt64
      else fun env w l => (v env w l).toUInt64
  | o => (val? reads o).map fun v env w l => (v env w l).toUInt64

/-- A float source's bits with its sign modifiers (`Float.src`), resolved. -/
def fval? (reads : Array Loc) (signs : UInt32) (o : Operand) : Option Val :=
  match o with
  | .reg r m =>
    (val? reads (.reg r {})).map fun v =>
      if m.abs || m.neg then fun env w l =>
        let x := v env w l
        let x := if m.abs then x &&& ~~~signs else x
        if m.neg then x ^^^ signs else x
      else v
  | o => val? reads o

/-- A 64-bit source from a register pair or constant words. -/
def pair? (reads : Array Loc) (o : Operand) : Option (Env → View → Nat → UInt64) := do
  let lo ← val? reads o; let hi ← hi? reads o
  some fun env w l => toU64 (lo env w l) (hi env w l)

/-- Comparison of two 32-bit integers, resolved. -/
def icmp? (op : String) (signed : Bool) : Option (UInt32 → UInt32 → Bool) :=
  let lt (a b : UInt32) : Bool := if signed then a.toInt32 < b.toInt32 else a < b
  match op with
  | "EQ" => some (· == ·) | "NE" => some (· != ·)
  | "LT" => some lt | "LE" => some fun a b => !lt b a
  | "GT" => some fun a b => lt b a | "GE" => some fun a b => !lt a b
  | "T" => some fun _ _ => true | "F" => some fun _ _ => false
  | _ => none

/-- A predicate combination, word-wide. -/
def bop? : String → Option (UInt32 → UInt32 → UInt32)
  | "AND" => some (· &&& ·) | "OR" => some (· ||| ·) | "XOR" => some (· ^^^ ·)
  | _ => none

def nPreds (i : Instr) (from_ : Nat) : Nat :=
  ((i.operands.toList.drop from_).takeWhile isPred).length

/-- Integer, move and compare forms. -/
def int? (f : Form) : Option Run := do
  let i := f.i
  let (u, op) := if i.opcode.startsWith "U" then (true, (i.opcode.drop 1).toString)
    else (false, i.opcode)
  let v (n : Nat) : Option Val := val? f.reads (f.o n)
  match op with
  | "MOV" =>
    if i.operands.size > 2 && f.o 2 != .imm 0xf || i.hasMod "64" then none
    let a ← v 1
    f.out32 u a
  | "S2R" | "S2UR" => let a ← v 1; f.out32 (u || op == "S2UR") a
  | "CS2R" =>
    if f.o 1 != .name "SRZ" then none
    if i.hasMod "32" then f.out32 u fun _ _ _ => 0 else f.out64 u fun _ _ _ => 0
  | "R2UR" => let a ← v 1; f.out32 true a
  | "LDC" =>
    -- bank 0 only (`ULDC` is `LDC` on the uniform datapath), whole words only
    if ["U8", "S8", "U16", "S16"].any i.hasMod then none
    match f.o 1 with
    | .cbank 0 _ _ =>
      if i.hasMod "64" then f.out64 u (← pair? f.reads (f.o 1)) else f.out32 u (← v 1)
    | _ => none
  | "IMAD" =>
    if i.hasMod "X" || i.hasMod "HI" then none
    let signed := !i.hasMod "U32"
    if i.hasMod "WIDE" then
      if nPreds i 1 > 0 then none
      let a ← v 1; let b ← v 2; let c ← pair? f.reads (f.o 3)
      f.out64 u fun env w l =>
        let x := a env w l; let y := b env w l
        let p : UInt64 := if signed then (x.toInt32.toInt64 * y.toInt32.toInt64).toUInt64
          else x.toUInt64 * y.toUInt64
        p + c env w l
    else
      let a ← v 1; let b ← v 2; let c ← v 3
      f.out32 u fun env w l => a env w l * b env w l + c env w l
  | "VIADD" => let a ← v 1; let b ← v 2; f.out32 u fun env w l => a env w l + b env w l
  | "IADD3" =>
    if i.hasMod "64" then none
    let n := nPreds i 1
    let a ← term? f.reads (f.o (1 + n)); let b ← term? f.reads (f.o (2 + n)); let c ← term? f.reads (f.o (3 + n))
    let cin : View → UInt32 → Nat → UInt64 ← if i.hasMod "X" then do
        let p ← pmask? f.reads (f.o (4 + n)); let q ← pmask? f.reads (f.o (5 + n))
        some fun w _ l => ((p w >>> l.toUInt32) &&& 1).toUInt64 + ((q w >>> l.toUInt32) &&& 1).toUInt64
      else some fun _ _ _ => 0
    let wb ← write32? (f.o 0)
    let c0 ← if n ≥ 1 then writeP? (f.o 1) else some fun _ _ => []
    let c1 ← if n ≥ 2 then writeP? (f.o 2) else some fun _ _ => []
    some <| f.run u fun env w on s =>
      let (vs, b32, b33) := Id.run do
        let mut vs := Array.mkEmpty 32
        let mut b32 : UInt32 := 0
        let mut b33 : UInt32 := 0
        for l in [0:32] do
          if (on >>> l.toUInt32) &&& 1 == 1 then
            let t := a env w l + b env w l + c env w l + cin w on l
            vs := vs.push t.toUInt32
            let bit := (1 : UInt32) <<< l.toUInt32
            if t >>> 32 ≥ 1 then b32 := b32 ||| bit
            if t >>> 32 ≥ 2 then b33 := b33 ||| bit
          else vs := vs.push 0
        return (vs, b32, b33)
      .reg s (wb on vs ++ c0 on b32 ++ c1 on b33)
  | "LEA" =>
    if i.hasMod "HI" then
      let sx := i.hasMod "SX32"
      if i.hasMod "X" then none
      let a ← v 1; let b ← v 2
      let c ← if sx then some fun env w l => if (a env w l).toInt32 < 0 then 0xffffffff else 0 else v 3
      let sh ← v (if sx then 3 else 4)
      f.out32 u fun env w l => b env w l + shf true false true true (a env w l) (c env w l) (sh env w l)
    else
      if nPreds i 1 > 0 then none
      let a ← v 1; let b ← v 2; let sh ← v 3
      f.out32 u fun env w l =>
        let k := sh env w l
        (if k ≥ 32 then 0 else a env w l <<< k) + b env w l
  | "SHF" =>
    let signed := i.hasMod "S32" || i.hasMod "S64"
    let wide := i.hasMod "U64" || i.hasMod "S64"
    let left := i.hasMod "L"; let hi := i.hasMod "HI"
    let a ← v 1; let b ← v 2; let c ← v 3
    f.out32 u fun env w l => shf left signed wide hi (a env w l) (c env w l) (b env w l)
  | "LOP3" =>
    if isPred (f.o 0) then none
    let a ← v 1; let b ← v 2; let c ← v 3; let lut ← v 4
    f.out32 u fun env w l => lop3 (a env w l) (b env w l) (c env w l) (lut env w l)
  | "SEL" =>
    if i.hasMod "64" then none
    let a ← v 1; let b ← v 2; let p ← pmask? f.reads (f.o 3)
    f.out32 u fun env w l => if (p w >>> l.toUInt32) &&& 1 == 1 then a env w l else b env w l
  | "ISETP" =>
    if i.hasMod "U64" || i.hasMod "S64" then none else
    let some c := i.mods[0]? | none
    let some bopName := i.mods.find? (["AND", "OR", "XOR"].contains ·) | none
    let signed := !i.hasMod "U32"
    let bop ← bop? bopName
    let a ← v 2; let b ← v 3; let pc ← pmask? f.reads (f.o 4)
    let pd ← writeP? (f.o 0); let pq ← writeP? (f.o 1)
    let r ← if i.hasMod "EX" then do
        -- the high-word step: lt || eq ∧ low, as `cmpEx`
        let low ← pmask? f.reads (f.o 5)
        let lt ← icmp? "LT" signed; let gt ← icmp? "GT" signed
        let comb : (UInt32 → UInt32 → UInt32 → UInt32) ← match c with
          | "LT" | "LE" => some fun l e _ => l ||| e
          | "GT" | "GE" => some fun _ e g => g ||| e
          | "EQ" => some fun _ e _ => e
          | "NE" => some fun _ e _ => ~~~e
          | _ => none
        some fun (env : Env) (w : View) (on : UInt32) =>
          let lo := low w
          let ltm := bitsOf (fun l => lt (a env w l) (b env w l)) on
          let eqm := bitsOf (fun l => a env w l == b env w l) on
          let gtm := bitsOf (fun l => gt (a env w l) (b env w l)) on
          let e := eqm &&& lo
          if c == "NE" then ~~~eqm ||| lo else comb ltm e gtm
      else do
        let cmpf ← icmp? c signed
        some fun (env : Env) (w : View) (on : UInt32) => bitsOf (fun l => cmpf (a env w l) (b env w l)) on
    some <| f.run u fun env w on s =>
      let rm := r env w on
      let p := pc w
      let d := bop rm p; let q := bop (~~~rm) p
      .reg s (pd on d ++ pq on q)
  | _ => none

/-- Binary32 arithmetic forms. -/
def float? (f : Form) : Option Run := do
  let i := f.i
  if i.hasMod "SAT" then none
  let ftz := i.hasMod "FTZ"
  let r := Float.rounding i.mods
  let s32 : UInt32 := 0x80000000
  let v (n : Nat) : Option Val := fval? f.reads s32 (f.o n)
  match i.opcode with
  | "FADD" =>
    let a ← v 1; let b ← v 2
    f.out32 false fun env w l => NV.Num.F32.add r ftz (a env w l) (b env w l)
  | "FMUL" =>
    if i.mods.any (["D2", "D4", "D8", "M2", "M4", "M8"].contains ·) then none
    let a ← v 1; let b ← v 2
    f.out32 false fun env w l => NV.Num.F32.mul r ftz (a env w l) (b env w l)
  | "FFMA" =>
    let a ← v 1; let b ← v 2; let c ← v 3
    f.out32 false fun env w l =>
      NV.Num.F32.fma r ftz (a env w l) (b env w l) (c env w l)
  | "FMNMX" =>
    if i.mods.any (· ∉ ["FTZ", "NAN"]) || i.operands.size != 4 then none
    let nanProp := i.hasMod "NAN"
    let a ← v 1; let b ← v 2; let p ← pmask? f.reads (f.o 3)
    f.out32 false fun env w l =>
      let isMin := (p w >>> l.toUInt32) &&& 1 == 1
      NV.Num.F32.minMax (!isMin) ftz nanProp (a env w l) (b env w l)
  | "FSEL" =>
    if !i.mods.isEmpty || i.operands.size != 4 then none
    let a ← v 1; let b ← v 2; let p ← pmask? f.reads (f.o 3)
    f.out32 false fun env w l => if (p w >>> l.toUInt32) &&& 1 == 1 then a env w l else b env w l
  | "MUFU" =>
    -- The binary32 forms of hot loops, on machine words (`NV.Num.MufuWord`); LG2
    -- takes the general definition's unit.
    if i.mods != #["EX2"] && i.mods != #["RCP"] && i.mods != #["LG2"] && i.mods != #["RSQ"] then none
    let unit : UInt32 → UInt32 := match i.mods[0]! with
      | "EX2" => NV.Num.MufuWord.ex2
      | "RCP" => NV.Num.MufuWord.rcp
      | "RSQ" => NV.Num.MufuWord.rsqrt
      | _ => fun x => (NV.Num.Approx.lg2 true x.toNat).toUInt32
    let a ← v 1
    f.out32 false fun env w l => unit (a env w l)
  | _ => none

/-- Loads and stores on global and shared memory. -/
def mem? (f : Form) : Option Run := do
  let i := f.i
  -- `.ENL2.256`'s two register quads are `decodeMem`'s.
  if i.hasMod "STRONG" || i.hasMod "ENL2" then none
  let (bytes, sx) := width i
  if sx then none
  let n := words bytes
  let (load, sp) ← match i.opcode with
    | "LDG" => some (true, Space.global) | "STG" => some (false, Space.global)
    | "LDS" => some (true, Space.shared) | "STS" => some (false, Space.shared)
    | _ => none
  let shared := sp == .shared
  let (ao, ro) := if load then (f.o 1, f.o 0) else (f.o 0, f.o 1)
  let ao := match ao with
    | .mem b w u d off => .mem b (w || !shared && i.hasMod "E") u d off
    | a => a
  let ad ← addr? f.reads ao
  let reg ← match ro with | .reg ⟨.r, r⟩ _ => some r | _ => none
  let mask : UInt64 := if shared then 0xffffff else 0xffffffffffffffff
  let addrs (on : UInt32) (w : View) : Array (Option UInt64) := Id.run do
    let mut a := Array.mkEmpty 32
    for l in [0:32] do
      a := a.push (if (on >>> l.toUInt32) &&& 1 == 1 then some (ad w l &&& mask) else none)
    return a
  -- The data registers, read with the sources (the request's `View`).
  let dat : View → Nat → Nat → UInt32 ← if load then some (fun _ _ _ => 0) else
    (List.range n).foldlM (fun (acc : View → Nat → Nat → UInt32) j => do
      let g ← if reg == 255 then some (fun (_ : View) (_ : Nat) => (0 : UInt32)) else slotOf f.reads (.r (reg + j))
      some fun w l k => if k == j then g w l else acc w l k) (fun _ _ _ => 0)
  some <| f.run false fun _ _ on s =>
    if load then
      .mem (fun w' => { space := sp, op := .ld, bytes, sem := .weak, scope := .thread,
                        addrs := addrs on w' }) s fun resp =>
        if reg == 255 then [] else
        (List.range n).map fun j => .lanes (reg + j) on ((Array.range 32).map fun l => resp[l * n + j]!)
    else
      .mem (fun w' =>
          let data := Id.run do
            let mut d := Array.mkEmpty (32 * n)
            for l in [0:32] do
              for j in [0:n] do
                d := d.push (if (on >>> l.toUInt32) &&& 1 == 1 then dat w' l j else 0)
            return d
          { space := sp, op := .st, bytes, sem := .weak, scope := .thread, addrs := addrs on w',
            data }) s fun _ => []

/-- The compiled form of instruction `i` at code index `pc`, if it is a hot form. -/
def decodeFast (reads : Array Loc) (i : Instr) (pc : Nat) : Option Run :=
  let f : Form := { i, pc, reads }
  int? f <|> float? f <|> mem? f

end NV.SASS.Fast
