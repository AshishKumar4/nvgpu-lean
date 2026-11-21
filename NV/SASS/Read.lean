import NV.SASS.Syntax
import NV.SASS.Elf
import NV.SASS.Text

/-!
# Reading disassembled SASS

`kernels` turns the listing that `cuobjdump -sass` or `nvdisasm -hex` prints for a cubin, plus that
cubin's `cuobjdump -elf` attributes, into `Kernel`s. An instruction line is

```
        /*0060*/              @!P0 BRA `(.L_x_1) ;            /* 0x0000000000088947 */
                                                              /* 0x000fea0003800000 */
```

the byte offset, the instruction as the disassembler prints it, and the two 64-bit halves of its
encoding. The high half carries the control word (`Ctrl.ofHi`). Operands are taken as printed,
with two exceptions where the text alone does not give the bits: a floating-point immediate is
printed in decimal (`0.5`, `1.00000001504746621988e+30`, `+QNAN`), so its bits are read from the
encoding (bits 32–63, where every ALU form keeps its 32-bit immediate) and the printed text is
checked against them; and `cuobjdump` prints branch targets as offsets where `nvdisasm` prints
labels, which both become `Operand.target`.
-/

namespace NV.SASS.Read

open Text

/-- `0x1f` or `-0x1`, as a two's-complement 64-bit pattern. -/
def intImm? : List Char → Option UInt64
  | '-' :: cs => (hex? cs).map fun n => 0 - n.toUInt64
  | cs => (hex? cs).map (·.toUInt64)

/-! ## Floating-point immediates -/

/-- A printed floating-point literal: `-1.5e+30`, `16777216`, `+INF`, `-QNAN`. -/
inductive Lit where
  | fin (neg : Bool) (digits : Nat) (exp10 : Int)
  | inf (neg : Bool)
  | nan
  deriving Repr

def lit? (cs : List Char) : Option Lit :=
  let (neg, cs) := match cs with
    | '-' :: r => (true, r) | '+' :: r => (false, r) | r => (false, r)
  match str cs with
  | "INF" => some (.inf neg)
  | "QNAN" | "NAN" => some .nan
  | _ =>
    let (mant, ex) := match splitWhen (fun c => c == 'e' || c == 'E') cs with
      | [m, e] => (m, some e)
      | _ => (cs, none)
    let (ip, fp) := match splitWhen (· == '.') mant with
      | [i, f] => (i, f)
      | _ => (mant, [])
    let e10 : Option Int := match ex with
      | none => some 0
      | some ('-' :: d) => (digitsVal 10 d).map fun n => -(n : Int)
      | some ('+' :: d) => (digitsVal 10 d).map (·)
      | some d => (digitsVal 10 d).map (·)
    match digitsVal 10 (ip ++ fp), e10 with
    | some d, some e => some (.fin neg d (e - fp.length))
    | _, _ => none

/-- An IEEE binary format by its exponent and fraction widths. -/
structure Fmt where
  ebits : Nat
  mbits : Nat

def f16 : Fmt := ⟨5, 10⟩
def bf16 : Fmt := ⟨8, 7⟩
def f32 : Fmt := ⟨8, 23⟩
def f64 : Fmt := ⟨11, 52⟩

/-- The magnitude of finite bits `b` as `m · 2^x`, or `none` for infinities and NaNs. -/
def Fmt.mag (f : Fmt) (b : Nat) : Option (Nat × Int) :=
  let e := (b / 2 ^ f.mbits) % 2 ^ f.ebits
  let m := b % 2 ^ f.mbits
  let bias : Int := 2 ^ (f.ebits - 1) - 1
  if e == 2 ^ f.ebits - 1 then none
  else if e == 0 then some (m, 1 - bias - f.mbits)
  else some (m + 2 ^ f.mbits, (e : Int) - bias - f.mbits)

/-- Whether the bits `b` of format `f` are what the disassembler means by `l`: the same special
value, or the representable value nearest to the decimal (ties either way). -/
def Fmt.matches (f : Fmt) (b : Nat) (l : Lit) : Bool :=
  let sign := (b / 2 ^ (f.ebits + f.mbits)) % 2 == 1
  let mbits := b % 2 ^ (f.ebits + f.mbits)
  match l, f.mag mbits with
  | .inf neg, none => sign == neg && mbits % 2 ^ f.mbits == 0
  | .nan, none => mbits % 2 ^ f.mbits != 0
  | .fin neg d k, some (m, x) =>
    if d == 0 then mbits == 0 && sign == neg else
    -- The finite neighbouring magnitudes; scale every value by 2^a · 10^c to compare integers.
    let nbrs := [mbits - 1, mbits + 1].filter (· != mbits) |>.filterMap f.mag
    let a := (-(nbrs.foldl (fun acc (_, x') => min acc x') x)).toNat
    let c := (-k).toNat
    let dec : Int := (d * 10 ^ (k + c).toNat * 2 ^ a : Nat)
    let dist (m : Nat) (x : Int) : Nat := (dec - (m * 2 ^ (x + a).toNat * 10 ^ c : Nat)).natAbs
    sign == neg && nbrs.all fun (m', x') => dist m x ≤ dist m' x'
  | _, _ => false

/-- How an opcode reads a 32-bit immediate: one f32, two f16 or bf16 halves (high half printed
first), or the high word of an f64. -/
inductive FloatImm where
  | single (f : Fmt)
  | halves (f : Fmt)
  | doubleHi

def floatImm (opcode : String) (mods : Array String) : FloatImm :=
  if ["HADD2", "HFMA2", "HMUL2", "HSET2", "HSETP2", "HMNMX2"].contains opcode then
    .halves (if mods.contains "BF16_V2" then bf16 else f16)
  -- `MUFU.RCP64H`/`RSQ64H` take a binary64's high word too (`MUFU.RCP64H R7, 2.147…e+09`).
  else if ["DADD", "DFMA", "DMUL", "DSETP", "DMNMX"].contains opcode ||
      (opcode == "MUFU" && mods.any (·.endsWith "64H")) then .doubleHi
  else .single f32

/-! ## Operands -/

/-- An operand as printed, before float immediates and labels are resolved. -/
inductive Tok where
  | op (o : Operand)
  | flt (l : Lit) (text : String)
  | label (name : String)

def reg? (name : String) : Option Reg :=
  let num (pre : String) : Option Nat := digitsVal 10 (name.toList.drop pre.length)
  match name with
  | "RZ" => some Reg.RZ | "URZ" => some Reg.URZ | "PT" => some Reg.PT | "UPT" => some Reg.UPT
  | _ =>
    if name.startsWith "UR" then (num "UR").map (⟨.ur, ·⟩)
    else if name.startsWith "UP" then (num "UP").map (⟨.up, ·⟩)
    else if name.startsWith "R" then (num "R").map (⟨.r, ·⟩)
    else if name.startsWith "P" then (num "P").map (⟨.p, ·⟩)
    else if name.startsWith "B" then (num "B").map (⟨.b, ·⟩)
    else none

/-- A register with its printed modifiers: `-|R2|.reuse`, `!P0`, `R0.H1`. -/
def regOperand? (cs : List Char) : Option Operand :=
  let (neg, cs) := match cs with | '-' :: r => (true, r) | r => (false, r)
  let (not, cs) := match cs with | '!' :: r => (true, r) | '~' :: r => (true, r) | r => (false, r)
  let (abs, cs) := match cs with
    | '|' :: r => (true, r.takeWhile (· != '|') ++ (r.dropWhile (· != '|')).drop 1)
    | r => (false, r)
  match splitWhen (· == '.') cs with
  | [] => none
  | base :: sufs =>
    (reg? (str base)).map fun r =>
      let sufs := sufs.map str
      .reg r { neg, abs, not, reuse := sufs.contains "reuse",
               suffix := (sufs.filter (· != "reuse")).toArray }

/-- Split at top-level occurrences of `sep`, outside brackets, parentheses and braces. -/
def splitTop (sep : Char → Bool) (cs : List Char) : List (List Char) :=
  let step := fun (acc : List (List Char) × List Char × Nat) (c : Char) =>
    let (out, cur, depth) := acc
    if sep c && depth == 0 then (cur.reverse :: out, [], depth)
    else
      let depth := if c == '[' || c == '(' || c == '{' then depth + 1
        else if (c == ']' || c == ')' || c == '}') && depth > 0 then depth - 1 else depth
      (out, c :: cur, depth)
  let (out, cur, _) := cs.foldl step ([], [], 0)
  (cur.reverse :: out).reverse

/-- The inside of `[R2.64+UR4+0x10]`: base register, whether it is 64-bit, uniform register,
offset, and the base's scale (`[R7.X4]`; 1 otherwise). `[RZ.U32+UR4]` (sm_90's `QSPC` of a
uniform generic address, with an offset): the 64-bit uniform pair `UR4` as the base; a non-zero `.U32` base is not read. -/
def address? (inner : List Char) : Option (Option Reg × Bool × Option Reg × Int × Nat) :=
  let terms := (splitWhen (· == '+') inner).map trimL
  match terms.map str with
  | ["RZ.U32", u] => (reg? u).bind fun r => if r.file == .ur then some (some r, true, none, 0, 1) else none
  | ["RZ.U32", u, o] => (reg? u).bind fun r => (intImm? (o.toList)).bind fun v =>
      if r.file == .ur then some (some r, true, none, v.toInt64.toInt, 1) else none
  | _ =>
  terms.foldlM (fun (base, wide, ureg, off, sc) term =>
    let term := trimL term
    match intImm? term with
    | some v => some (base, wide, ureg, off + v.toInt64.toInt, sc)
    | none =>
      let (name, wide', sc') := match (splitWhen (· == '.') term).map str with
        | [name] => (some name, false, 1)
        | [name, "64"] => (some name, true, 1)
        | [name, "X4"] => (some name, false, 4)
        | [name, "X8"] => (some name, false, 8)
        | [name, "X16"] => (some name, false, 16)
        | _ => (none, false, 1)
      match name.bind reg? with
      | some r =>
        if r.file == .ur && base.isSome then some (base, wide, some r, off, sc)
        else if base.isNone then some (some r, wide', ureg, off, sc')
        else none
      | none => none) (none, false, none, 0, 1)

/-- `c[0x0][0x28]`, `c[0x0][R2+0x10]`. -/
def cbank? (cs : List Char) : Option Operand :=
  match cs with
  | 'c' :: '[' :: rest =>
    match splitWhen (· == ']') rest with
    | [bank, '[' :: inner, []] =>
      (hex? bank).bind fun b => (address? inner).map fun (base, _, _, off, _) => .cbank b base off
    | _ => none
  | _ => none

/-- `[R2+0x10]`, `desc[UR4][R2.64]`, `gdesc[UR4]`, `tmem[UR6]`, `idesc[UR5]`. -/
def mem? (cs : List Char) : Option Operand :=
  let (pre, rest) := (cs.takeWhile (· != '['), cs.dropWhile (· != '['))
  match pre, (splitWhen (· == ']') rest).filter (!·.isEmpty) with
  | [], ['[' :: inner] => (address? inner).map fun (b, w, u, o, sc) =>
    if sc == 1 then .mem b w u none o else .scaled (.mem b w u none o) sc
  | ['d', 'e', 's', 'c'], ['[' :: d, '[' :: inner] =>
    (reg? (str d)).bind fun d => (address? inner).map fun (b, w, u, o, _) => .mem b w u (some d) o
  | ['g', 'd', 'e', 's', 'c'], ['[' :: d] => (reg? (str d)).map (.gdesc · #[])
  | ['g', 'd', 'e', 's', 'c'], ['[' :: d, '.' :: mods] =>
    (reg? (str d)).map (.gdesc · ((splitWhen (· == '.') mods).map str).toArray)
  | ['t', 'm', 'e', 'm'], ['[' :: inner] => (address? inner).bind fun
    | (some b, _, none, o, _) => some (.udesc "tmem" b o)
    | _ => none
  | ['i', 'd', 'e', 's', 'c'], ['[' :: d] => (reg? (str d)).map (.udesc "idesc" ·)
  | _, _ => none

def tok (cs : List Char) : Tok :=
  let cs := trimL cs
  match cs with
  | '`' :: '(' :: rest => .label (str (rest.takeWhile (· != ')')))
  | _ =>
    match intImm? cs, lit? cs with
    | some v, _ => .op (.imm v)
    | none, some l => .flt l (str cs)
    | none, none =>
      match cs with
      | '-' :: rest@('c' :: '[' :: _) =>
        .op ((cbank? rest).map .neg |>.getD (.name (str cs)))
      | '~' :: rest@('c' :: '[' :: _) =>
        .op ((cbank? rest).map .not |>.getD (.name (str cs)))
      | _ => .op ((cbank? cs <|> mem? cs <|> regOperand? cs).getD (.name (str cs)))

/-- Opcodes whose immediate last operand is a code offset in `cuobjdump`'s listing. -/
def branchOps : List String := ["BRA", "BSSY", "CALL", "JMP", "RET"]

/-! ## Instructions -/

/-- The text before an `nvdisasm` annotation `(*…*)`. -/
def dropAnnotation : List Char → List Char
  | '(' :: '*' :: _ => []
  | c :: r => c :: dropAnnotation r
  | [] => []

/-- One instruction line of a listing. -/
structure Line where
  pc : Nat
  text : String
  lo : UInt64
  hi : UInt64 := 0

/-- Parse an instruction line's text, given the function's labels. -/
def instr (labels : List (String × Nat)) (l : Line) : Except String Instr := do
  let err {α} (m : String) : Except String α := .error s!"{l.pc}: {l.text}: {m}"
  -- `nvdisasm` annotates some instructions in `(*…*)` (`BRX R2 -0x90 (*"BRANCH_TARGETS …"*)`).
  let cs := trimL (dropAnnotation l.text.toList)
  let (guard, cs) ← match cs with
    | '@' :: r =>
      let g := r.takeWhile (!isWs ·)
      let (neg, g) := match g with | '!' :: g => (true, g) | g => (false, g)
      match reg? (str g) with
      | some p => pure (some (p, neg), trimL (r.dropWhile (!isWs ·)))
      | none => err "guard"
    | _ => pure (none, cs)
  let opTok := cs.takeWhile (!isWs ·)
  let (opcode, mods) ← match (splitWhen (· == '.') opTok).map str with
    | o :: ms => pure (o, ms.toArray)
    | [] => err "no opcode"
  let rest := trimL (cs.dropWhile (!isWs ·))
  -- Operands are comma-separated; `RET.REL.NODEC R2 0x0` also separates by a space.
  let pieces := (splitTop (· == ',') rest).flatMap fun p =>
    (splitTop isWs (trimL p)).filter (!·.isEmpty)
  let toks := pieces.map tok
  -- The 32-bit immediate field, for float immediates.
  let imm32 : Nat := (l.lo >>> 32).toNat
  let flts := toks.filterMap fun | .flt lt s => some (lt, s) | _ => none
  let fimm : Option Operand ← match flts, floatImm opcode mods with
    | [], _ => pure none
    | [(lt, s)], .single f =>
      if f.matches imm32 lt then pure (some (.imm imm32.toUInt64))
      else err s!"{s} is not the encoded {imm32}"
    | [(lt, s)], .doubleHi =>
      -- A binary64 operand (`DFMA`) is the double; `MUFU.RCP64H`'s source is the high word.
      if f64.matches (imm32 * 2 ^ 32) lt then
        pure (some (.imm (if opcode == "MUFU" then imm32.toUInt64 else imm32.toUInt64 <<< 32)))
      else err s!"{s} is not the encoded high word {imm32}"
    | [(h, sh), (lo, sl)], .halves f =>
      if f.matches (imm32 / 2 ^ 16) h && f.matches (imm32 % 2 ^ 16) lo then
        pure (some (.imm imm32.toUInt64))
      else err s!"{sh}, {sl} are not the encoded halves {imm32}"
    | _, _ => err "float immediates of an unknown form"
  let isBranch := branchOps.contains opcode
  let last := toks.length - 1
  let ops ← (toks.zipIdx.filterMap fun (t, k) =>
    match t with
    | .op (.imm v) =>
      some (if isBranch && k == last then .ok (.target v.toNat) else .ok (.imm v))
    | .op o => some (.ok o)
    | .label name => some (match labels.lookup name with
      | some pc => .ok (.target pc)
      | none => .error s!"{l.pc}: {l.text}: unknown label {name}")
    | .flt .. =>
      -- the first float token stands for the merged immediate
      if (toks.take k).any (fun | .flt .. => true | _ => false) then none
      else fimm.map .ok).mapM id
  return { pc := l.pc, guard, opcode, mods, operands := ops.toArray, ctrl := Ctrl.ofHi l.hi }

/-! ## Listings -/

/-- A function of a listing: its name, instruction lines and labels. -/
structure Fn where
  name : String
  lines : Array Line := #[]
  labels : List (String × Nat) := []

/-- `/*0060*/`: the offset at the start of an instruction line, and the rest of the line. -/
def offset? (cs : List Char) : Option (Nat × List Char) :=
  match cs with
  | '/' :: '*' :: rest =>
    let ds := rest.takeWhile (· != '*')
    match digitsVal 16 ds, rest.drop (ds.length) with
    | some pc, '*' :: '/' :: r => some (pc, r)
    | _, _ => none
  | _ => none

/-- `/* 0x000fe20000000800 */`. -/
def word? (cs : List Char) : Option UInt64 :=
  match trimL cs with
  | '/' :: '*' :: r => (hex? (trimL (r.takeWhile (· != '*')))).map (·.toUInt64)
  | _ => none

/-- Split a `cuobjdump -sass` or `nvdisasm -hex` listing into functions; only the functions `want`
accepts keep their lines (the others are skipped cheaply: a module may hold many kernels). -/
def functions (text : String) (want : String → Bool := fun _ => true) : Except String (Array Fn) := do
  let mut fns : Array Fn := #[]
  let mut cur : Option Fn := none
  -- Whether the current function is wanted.
  let mut keep := false
  -- Labels seen since the last instruction; they name the next one.
  let mut pending : List String := []
  for line in lines text do
    -- Instruction and encoding lines of functions not wanted are most of a module's listing, and
    -- the only lines that start with a space (headers and `.section` lines start with a tab,
    -- labels at the margin): skipped before any trimming.
    if !keep && starts line " " then continue
    let t := line.trim
    if starts t "Function : " then
      let name := (t.drop "Function : ".length).trim.toString
      fns := fns ++ cur.toArray
      keep := want name
      cur := if keep then some { name } else none
      continue
    if starts t ".text." && ends t ":" then
      let name := ((t.drop ".text.".length).dropRight 1).toString
      fns := fns ++ cur.toArray
      keep := want name
      cur := if keep then some { name } else none
      continue
    -- A label line; `nvdisasm` prints a function's own label before its `.text.` line, so labels
    -- are collected whichever function is current.
    if ends t ":" && (!starts t "." || starts t ".L_") then
      pending := (t.dropRight 1).toString :: pending
      continue
    if !keep then
      if starts t ".section" then pending := []
      continue
    let cs := t.toString.toList
    let s := t.toString
    if s.startsWith ".section" then
      -- The end of a function: labels after its last instruction name its end.
      if let some f := cur then
        let endPc := (f.lines.back?.map (·.pc + 16)).getD 0
        cur := some { f with labels := pending.map (·, endPc) ++ f.labels }
      pending := []
    else if let some (pc, rest) := offset? cs then
      let text := rest.takeWhile (· != ';')
      let some lo := word? (rest.drop (text.length + 1))
        | throw s!"{s}: no encoding; disassemble with `nvdisasm -hex` or `cuobjdump -sass`"
      -- `cur` is emptied before the update so the lines array is not shared and grows in place.
      let some f := cur | throw s!"{s}: instruction outside a function"
      cur := none
      cur := some { f with lines := f.lines.push { pc, text := str text, lo },
                           labels := pending.map (·, pc) ++ f.labels }
      pending := []
    else if let some hi := word? cs then
      if let some f := cur then
        cur := none
        cur := some { f with lines := f.lines.modify (f.lines.size - 1) fun l => { l with hi } }
  return fns ++ cur.toArray

/-- The instructions of the functions of a listing `want` accepts, by function name. -/
def listing (text : String) (want : String → Bool := fun _ => true) :
    Except String (Array (String × Array Instr)) := do
  (← functions text want).mapM fun f => do
    let code ← f.lines.mapM (instr f.labels)
    return (f.name, code)

/-- A cubin read once: its listing, its kernels' ELF attributes and its module data, from which
kernels are made by name as they are needed (`Module.kernels`): a library's cubin holds hundreds
of kernels, of which a program runs a few. -/
structure Module where
  sass : String
  attrs : List (String × Elf.Attrs)
  textBase : Nat
  /-- A hash of the listing: kernels of one module share it (`Kernel.module`). -/
  hash : UInt64
  data : Elf.ModuleData
  sm : Nat

/-- A cubin's listing and `cuobjdump -elf` text, its data placed from `base`. -/
def Module.of (sass elf : String) (base : Nat := Elf.dataBase) : Module :=
  let ls := lines elf
  let attrs := Elf.parse elf ls
  let texts := attrs.filterMap (·.2.textOffset)
  { sass, attrs, textBase := texts.foldl min (texts.head?.getD 0), hash := Hashable.hash sass,
    data := Elf.moduleData elf ls base, sm := (Elf.sm? elf ls).getD 90 }

/-- A kernel's code with the module's device functions it calls appended (a relocatable module's
`CALL.ABS.NOINC` to a function in its own section: `__cuda_sm20_div_rn_f64_full`, the binary64
division's slow path), their code addresses relocated as the driver's linker does: each callee
placed after the code before it, its branch targets moved with it, a `CALL.ABS`'s target
(`R_CUDA_ABS55_16_34`, sm_80's `R_CUDA_ABS47_34`) its place, and the return address a caller loads into a register
(`R_CUDA_ABS32_LO_32`, `_HI_32`) the code address of its place (`codeOffset` plus it, which
`RET.ABS` reads back). -/
def withCallees (m : Module) (name : String) (code : Array Instr) (codeOffset : Nat) :
    Except String (Array Instr) := do
  let relas := m.data.textRelas
  if !relas.any (·.1 == name) then return code
  -- The callees, transitively, in the order first referenced.
  let mut order : Array String := #[name]
  let mut i := 0
  while i < order.size do
    for (f, _, sym, _, _) in relas do
      if f == order[i]! && !order.contains sym && !m.data.externs.any (·.2 == sym) then
        order := order.push sym
    i := i + 1
  let callees := order.extract 1 order.size
  let fns ← if callees.isEmpty then pure #[] else listing m.sass (callees.contains ·)
  -- Each function's place: the kernel at 0, each callee after the code before it.
  let mut base : Array (String × Nat) := #[(name, 0)]
  let mut out := code
  let mut next := (code.back?.map (·.pc + 16)).getD 0
  for c in callees do
    let some (_, cc) := fns.find? (·.1 == c) | throw s!"{name}: calls {c}, which the listing lacks"
    base := base.push (c, next)
    let b := next
    let moved := cc.map fun ins =>
      let absRet := ins.opcode == "RET" && ins.mods.contains "ABS"
      { ins with pc := ins.pc + b,
                 operands := if absRet then ins.operands else ins.operands.map fun
                   | .target t => .target (t + b)
                   | o => o }
    out := out ++ moved
    next := (moved.back?.map (·.pc + 16)).getD next
  -- The relocations of every placed function, applied to its instructions.
  for (f, off, sym, ty, add) in relas do
    let some (_, fb) := base.find? (·.1 == f) | continue
    let some (_, sb) := base.find? (·.1 == sym) | continue
    let some k := out.findIdx? (·.pc == fb + off) | throw s!"{f}: a relocation at {off}, no instruction"
    let ins := out[k]!
    let addr := codeOffset + sb + add
    let isImm : Operand → Bool := fun | .imm _ => true | _ => false
    let lastImm : Option Nat := (ins.operands.size - 1 - ·) <$> ins.operands.reverse.findIdx? isImm
    let ops ← match ty with
      | "R_CUDA_ABS55_16_34" | "R_CUDA_ABS47_34" =>
        if ins.operands.isEmpty then throw s!"{f}: {ins.opcode} at {off}: a call with no target"
        pure (ins.operands.set! 0 (Operand.target (sb + add)))
      | "R_CUDA_ABS32_LO_32" | "R_CUDA_ABS32_HI_32" =>
        let some j := lastImm | throw s!"{f}: {ins.opcode} at {off}: no immediate to relocate"
        let v := if ty == "R_CUDA_ABS32_LO_32" then addr % 2 ^ 32 else addr / 2 ^ 32
        pure (ins.operands.set! j (Operand.imm v.toUInt64))
      | _ => throw s!"{f}: relocation {ty} at {off} is not modelled"
    out := out.set! k { ins with operands := ops }
  return out

/-- The module's kernels `want` accepts. -/
def Module.kernels (m : Module) (want : String → Bool := fun _ => true) : Except String (Array Kernel) := do
  (← listing m.sass want).mapM fun (name, code) => do
    let some a := m.attrs.lookup name | throw s!"{name}: not in the ELF attributes"
    let codeOffset := (a.textOffset.getD m.textBase) - m.textBase
    let own := code.size
    let code ← withCallees m name code codeOffset
    -- A callee's registers count with the kernel's (the kernel's `EIATTR_REGCOUNT` in a
    -- relocatable module is its own): the highest register each callee names, and its pair.
    let calleeRegs := (code.extract own code.size).foldl (fun n i => i.operands.foldl (fun n o => match o with
      | .reg ⟨.r, r⟩ _ => if r < 255 then max n (r + 2) else n
      | _ => n) n) 0
    let params := (a.params.qsort (·.1 < ·.1)).map fun (_, off, n) => (off, n)
    -- The parameter block covers every parameter, `EIATTR_CBANK_PARAM_SIZE` or not (an sm_90
    -- cubin in PyTorch 2.9's wheel has only `EIATTR_KPARAM_INFO`).
    let paramBytes := params.foldl (fun m (off, n) => max m (off + n)) a.paramBytes
    -- `USETMAXREG.TRY_ALLOC UPd, n` raises the warp's allocation to n registers.
    let raised := code.foldl (fun m i => if i.opcode != "USETMAXREG" then m else
      i.operands.foldl (fun m o => match o with | .imm n => max m n.toNat | _ => m) m) 0
    let regs := max a.regs calleeRegs
    return { name, code, regs, regFile := max regs raised, params, paramBytes,
             paramBase := a.paramBase, cluster := a.cluster,
             sharedBytes := a.sharedSection - Elf.reservedShared.value,
             sharedReserved := Elf.reservedShared.value, localBytes := a.minStackBytes,
             sm := m.sm, banks := a.banks.toArray ++ m.data.banks, data := m.data.data,
             externs := m.data.externs, codeOffset,
             module := m.hash }

/-- Kernels from a listing and the same cubin's `cuobjdump -elf` text: those named in `names`, or
all of them. -/
def kernels (sass elf : String) (names : Option (List String) := none) :
    Except String (Array Kernel) :=
  (Module.of sass elf).kernels fun n => names.all (·.contains n)

end NV.SASS.Read
