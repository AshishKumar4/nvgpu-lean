import NV.Fact
import NV.SASS.Text

/-!
# Kernel attributes from `cuobjdump -elf`

A cubin's per-kernel facts that the disassembly does not print: registers per thread, the
parameter layout in constant bank 0, static shared memory, required thread and cluster shapes.
They come from the section table and from the `.nv.info` attribute sections (`EIATTR_*`), which
`cuobjdump -elf` prints as text, e.g.

```
.nv.info.vadd
	<0x2>
	Attribute:	EIATTR_KPARAM_INFO
	Format:	EIFMT_SVAL
	Value:	Index : 0x0	Ordinal : 0x3	Offset  : 0x18	Size    : 0x4
		Pointee's logAlignment : 0x0	Space : 0x0	cbank : 0x1f	Parameter Space : CBANK
```
-/

namespace NV.SASS.Elf

open Text

/-- Bytes at the start of each CTA's shared-memory window that are not the kernel's: the
`.nv.shared.<kernel>` section is ptxas's `smem` plus these, and user data start after them
(the code addresses it from `0x400`). -/
def reservedShared : Fact Nat :=
  { value := 1024, kind := .encoded, source := { ref := "fixtures", loc := "tests/sass ELF listings" }
    note := "" }

/-- One kernel's attributes. Absent attributes keep their defaults. -/
structure Attrs where
  /-- `EIATTR_REGCOUNT`. -/
  regs : Nat := 0
  /-- `EIATTR_KPARAM_INFO`: parameter ordinal, byte offset, bytes. -/
  params : Array (Nat × Nat × Nat) := #[]
  /-- `EIATTR_CBANK_PARAM_SIZE`. -/
  paramBytes : Nat := 0
  /-- `EIATTR_PARAM_CBANK`: where the parameter block starts in constant bank 0. -/
  paramBase : Nat := 0x210
  /-- Size of the `.nv.shared.<kernel>` section (the reserved bytes included). -/
  sharedSection : Nat := 0
  /-- `EIATTR_MAX_THREADS` (`.maxntid`). -/
  maxThreads : Option (Nat × Nat × Nat) := none
  /-- `EIATTR_REQNTID` (`.reqntid`). -/
  reqThreads : Option (Nat × Nat × Nat) := none
  /-- `EIATTR_CTA_PER_CLUSTER` (`.reqnctapercluster`, `__cluster_dims__`). -/
  cluster : Option (Nat × Nat × Nat) := none
  /-- `EIATTR_NUM_BARRIERS`: named barriers the kernel uses. -/
  barriers : Nat := 0
  /-- `EIATTR_FRAME_SIZE`: bytes of local memory per thread for the kernel's own stack frame. -/
  frameBytes : Nat := 0
  /-- `EIATTR_MIN_STACK_SIZE`: the stack the kernel and its callees need, per thread. -/
  minStackBytes : Nat := 0
  /-- Alignment of the kernel's `.text` section. -/
  textAlign : Nat := 0
  /-- File offset of the kernel's `.text` section. -/
  textOffset : Option Nat := none
  /-- Contents of the kernel's constant banks other than 0 (`.nv.constant2.<kernel>`: jump
  tables), as 32-bit words. -/
  banks : List (Nat × Array UInt32) := []
  deriving Repr, Inhabited

/-- The numbers of a value list such as `0x2 0x1 0x1`. -/
def nums (s : String) : List Nat :=
  (s.splitOn " ").filterMap fun w => if (trim w).isEmpty then none else num? w

def triple? (s : String) : Option (Nat × Nat × Nat) :=
  match nums s with
  | [x, y, z] => some (x, y, z)
  | _ => none

/-- `key : value` fields of a tab-separated attribute value (`Ordinal : 0x3	Offset : 0x18`). -/
def fields (s : String) : List (String × String) :=
  (s.splitOn "\t").filterMap fun f =>
    match f.splitOn ":" with
    | [k, v] => some (trim k, trim v)
    | _ => none

/-- The function name of a global `.nv.info` value: `function: vadd(0xc)	register count: 12`. -/
def fnOf (v : String) : Option String :=
  match v.splitOn "function: " with
  | [_, rest] =>
    let s := (rest.splitOn "\t").headD ""
    -- the name ends at its last `(`: `$f$_Z1gi(0xd)`
    let cs := s.toList.reverse.dropWhile (· != '(')
    if cs.isEmpty then none else some (str (cs.drop 1).reverse)
  | _ => none

/-- The number after the last `: ` of a global `.nv.info` value. -/
def lastNum (v : String) : Option Nat :=
  (v.splitOn ": ").getLast?.bind num?

/-- An attribute as printed: its name and value text (continuation lines joined by tabs). -/
structure Attr where
  name : String
  value : String := ""

def setAttr (a : Attrs) (att : Attr) : Attrs :=
  let v := att.value
  match att.name with
  | "EIATTR_KPARAM_INFO" =>
    let fs := fields v
    let get (k : String) : Option Nat := (fs.lookup k).bind num?
    match get "Ordinal", get "Offset", get "Size" with
    | some o, some off, some n => { a with params := a.params.push (o, off, n) }
    | _, _, _ => a
  | "EIATTR_CBANK_PARAM_SIZE" => { a with paramBytes := (num? v).getD a.paramBytes }
  | "EIATTR_PARAM_CBANK" =>
    match nums v with
    | [_, x] => { a with paramBase := x % 0x10000 }
    | _ => a
  | "EIATTR_MAX_THREADS" => { a with maxThreads := triple? v }
  | "EIATTR_REQNTID" => { a with reqThreads := triple? v }
  | "EIATTR_CTA_PER_CLUSTER" => { a with cluster := triple? v }
  | "EIATTR_NUM_BARRIERS" => { a with barriers := (num? v).getD 0 }
  | _ => a

/-- Section table rows: `Index Offset Size ES Align Type Flags Link Info Name`, as name, file
offset, size, alignment and `sh_info`. -/
def sectionRowInfo? (line : String) : Option (String × Nat × Nat × Nat × Nat) :=
  -- Cheap rejection first: a row starts with an index and ends with a section name.
  let t := line.toRawSubstring.trim
  if t.isEmpty || !(t.front.isDigit || ('a' ≤ t.front && t.front ≤ 'f')) || starts t "0x" then none else
  if (t.takeRightWhile (· != ' ')).front != '.' then none else
  match (line.splitOn " ").filter (!·.isEmpty) with
  | [_, off, size, _, align, _, _, _, info, name] =>
    if name.startsWith "." then
      match num? ("0x" ++ off), num? ("0x" ++ size), num? ("0x" ++ align), num? ("0x" ++ info) with
      | some o, some n, some a, some i => some (name, o, n, a, i)
      | _, _, _, _ => none
    else none
  | _ => none

/-- `sectionRowInfo?` without the `sh_info`. -/
def sectionRow? (line : String) : Option (String × Nat × Nat × Nat) :=
  (sectionRowInfo? line).map fun (name, o, n, a, _) => (name, o, n, a)

/-- Where the section table (`Sections:` up to the first blank line) ends: lines before it are
rows (`sectionRow?`), lines after are sections' contents. -/
def tableEnd (ls : Array Substring.Raw) : Nat :=
  let start := (ls.findIdx? (fun l => starts l.trim "Sections:")).getD 0
  ((ls.extract start ls.size).findIdx? (·.trim.isEmpty)).map (start + ·) |>.getD 0

/-- `.nv.constant2.vadd`: bank 2 of kernel `vadd` (bank 0, the parameters, is the launch's). -/
def bankOf (sect : String) : Option (Nat × String) :=
  match sect.splitOn "." with
  | ["", "nv", c, fn] =>
    if c.startsWith "constant" then
      (num? (str (c.toList.drop "constant".length))).bind fun b =>
        if b == 0 then none else some (b, fn)
    else none
  | _ => none

/-- Append words to bank `b`. -/
def addWords (b : Nat) (ws : Array UInt32) (a : Attrs) : Attrs :=
  if a.banks.any (·.1 == b) then
    { a with banks := a.banks.map fun (b', v) => (b', if b' == b then v ++ ws else v) }
  else { a with banks := a.banks ++ [(b, ws)] }

/-- The function a per-function section belongs to (`.nv.info.vadd`, `.nv.constant2.vadd`). -/
def sectionFn? (sect : String) : Option String :=
  if sect.startsWith ".nv.info." then some (sect.drop ".nv.info.".length).toString
  else (bankOf sect).map (·.2)

/-- Parse `cuobjdump -elf` text into each kernel's attributes, by kernel name: the functions `want`
accepts (sections of others are skipped). -/
def parse (text : String) (ls : Array Substring.Raw := lines text)
    (want : String → Bool := fun _ => true) : List (String × Attrs) := Id.run do
  let mut out : List (String × Attrs) := []
  let upd (out : List (String × Attrs)) (fn : String) (f : Attrs → Attrs) :=
    match out.lookup fn with
    | some a => out.map fun (n, b) => if n == fn then (n, f a) else (n, b)
    | none => out ++ [(fn, f {})]
  -- The section whose attributes we are reading, and the attribute being read.
  let mut sect : String := ""
  let mut cur : Option Attr := none
  let flush (out : List (String × Attrs)) (sect : String) (cur : Option Attr) :=
    match cur with
    | none => out
    | some att =>
      if sect == ".nv.info" then
        match fnOf att.value with
        | some fn =>
          match att.name with
          | "EIATTR_REGCOUNT" => upd out fn fun a => { a with regs := (lastNum att.value).getD 0 }
          | "EIATTR_FRAME_SIZE" =>
            upd out fn fun a => { a with frameBytes := (lastNum att.value).getD 0 }
          | "EIATTR_MIN_STACK_SIZE" =>
            upd out fn fun a => { a with minStackBytes := (lastNum att.value).getD 0 }
          | _ => out
        | none => out
      else if sect.startsWith ".nv.info." then
        upd out (str (sect.toList.drop ".nv.info.".length)) (setAttr · att)
      else out
  let mut bank : Option (Nat × String) := none
  -- Inside a section of a function not wanted.
  let mut skip := false
  let te := tableEnd ls
  for ln in ls.extract 0 te do
    if let some (name, off, size, align, info) := sectionRowInfo? ln.toString then
      if name.startsWith ".nv.shared." then
        let fn := str (name.toList.drop ".nv.shared.".length)
        if fn != "reserved.0" then out := upd out fn fun a => { a with sharedSection := size }
      else if name.startsWith ".text." then
        -- A `.text` section's `sh_info` carries the function's register count in its top byte
        -- when the kernel has no `EIATTR_REGCOUNT` (inferred).
        out := upd out (str (name.toList.drop ".text.".length)) fun a =>
          { a with textAlign := align, textOffset := some off,
                   regs := if a.regs == 0 then info >>> 24 else a.regs }
  for ln in ls.extract te ls.size do
    -- Hex dump lines matter only inside a constant bank's section.
    if bank.isNone && starts (dropBlank ln) "0x" then continue
    if skip && (starts ln " " || starts ln "\t") then continue
    let line := ln.toString
    let t := ln.trim.toString
    if let some (b, fn) := bank then
      -- The words of a constant bank, until the next section.
      if t.startsWith "0x" then
        out := upd out fn (addWords b ((nums t).toArray.map (·.toUInt32)))
      else if !t.isEmpty then
        sect := t
        skip := !((sectionFn? t).all want)
        bank := if skip then none else bankOf t
    else if !line.startsWith "\t" && !line.startsWith " " && !t.isEmpty then
      -- A new section (`.nv.info.vadd`, `.text.vadd`, …).
      out := flush out sect cur
      cur := none
      sect := t
      skip := !((sectionFn? t).all want)
      bank := if skip then none else bankOf t
    else if t.startsWith "Attribute:" then
      out := flush out sect cur
      cur := some { name := trim ((t.splitOn "\t").getLast?.getD "") }
    else if t.startsWith "Value:" then
      cur := cur.map fun att => { att with value := str (trimL (t.toList.drop 6)) }
    else if t.startsWith "<0x" then
      out := flush out sect cur
      cur := none
    else if line.startsWith "\t\t" then
      cur := cur.map fun att => { att with value := att.value ++ "\t" ++ t }
  return flush out sect cur

/-! ## Module data

What a cubin holds beside its kernels: initialized and zeroed device data (`.nv.global.init`,
`.nv.global`), the module's constant banks (`.nv.constant3` for `__constant__` data,
`.nv.constant4` for addresses the loader fills in), and the external functions the code calls
(`vprintf`). The loader places the data and relocates the bank words that hold addresses
(`.rela.nv.constantN`, `R_CUDA_64`). Where it places them is not documented; the addresses here
are ours (`dataBase`, `externBase`), and nothing a correct program computes depends on them. -/

/-- Where module data starts in the global address space (ours): 256 MB below the launch specs'
global base, so one store covers both without a page table spanning a terabyte. -/
def dataBase : Nat := 0x7eff_f000_0000

/-- Addresses given to external functions (ours): not code the simulator runs, so calls to them
are recognized by address (`vprintf`). -/
def externBase : Nat := 0x7d00_0000_0000

/-- A symbol of `.symtab`: its value, size and section index. -/
structure Symbol where
  value : Nat
  size : Nat
  shndx : Nat
  deriving Repr, Inhabited

structure ModuleData where
  /-- Placed data: address and bytes (zeros for `.nv.global`). -/
  data : Array (Nat × Array UInt8) := #[]
  /-- Module constant banks after relocation, as words. -/
  banks : Array (Nat × Array UInt32) := #[]
  /-- External functions by address. -/
  externs : Array (Nat × String) := #[]
  /-- The data symbols (`__device__` and `__constant__` variables) as name, address and bytes:
  what `cuModuleGetGlobal` returns. -/
  symbols : Array (String × Nat × Nat) := #[]
  /-- Where bank 3 (`__constant__` data) is placed among the data, if the module has one: a host
  copy to a `__constant__` variable writes there, and the device reads the bank from it. -/
  bank3 : Option Nat := none
  deriving Inhabited

/-- The bytes a section dump shows: words, then a short tail byte by byte (`.nv.global.init` of
23 bytes is five words and three bytes). -/
def dumpBytes (toks : List Nat) (size : Nat) : ByteArray := Id.run do
  let nw := size / 4
  let mut out := ByteArray.empty
  for (t, k) in toks.zipIdx do
    if k < nw then
      for j in [0:4] do out := out.push (t >>> (8 * j)).toUInt8
    else out := out.push t.toUInt8
  return out.extract 0 size

def wordsOf (b : ByteArray) : Array UInt32 :=
  (Array.range ((b.size + 3) / 4)).map fun w =>
    (List.range 4).foldl (fun v j => v ||| ((b.get! (4 * w + j)).toUInt32 <<< (8 * j).toUInt32))
      (0 : UInt32)

/-- The SM version in the ELF header line (`64-bit ELF: type=ET_EXEC, ABI=8, sm=80, …`). -/
def sm? (text : String) (ls : Array Substring.Raw := lines text) : Option Nat :=
  -- The header is among the first lines.
  let head := (ls.extract 0 8).toList.map (·.toString)
  (head.findSome? fun l => (l.splitOn "sm=")[1]?).bind fun rest => (rest.takeWhile Char.isDigit).toString.toNat?

/-- The module data of a cubin's `cuobjdump -elf` text, placed from `base` (the emulated device
gives each module its own place in device memory). -/
def moduleData (text : String) (ls : Array Substring.Raw := lines text) (base : Nat := dataBase) :
    ModuleData := Id.run do
  -- Section table: index → name and size.
  let mut sects : Array (Nat × String × Nat) := #[]
  let mut syms : Array (String × Symbol) := #[]
  let mut dumps : Array (String × Array Nat) := #[]
  -- Relocations: section, offset, symbol, and the addend (`none` for a `.rel` section's, whose
  -- addend is the word already at the offset).
  let mut relas : Array (String × Nat × String × Option Nat) := #[]
  let mut mode : String := ""
  let mut cur : String := ""
  let te := tableEnd ls
  for ln in ls.extract 0 te do
    let line := ln.toString
    if let some (name, _, size, _) := sectionRow? line then
      let cols := (line.splitOn " ").filter (!·.isEmpty)
      if let some idx := cols.head?.bind (fun c => num? ("0x" ++ c)) then
        sects := sects.push (idx, name, size)
  for ln in ls.extract te ls.size do
    -- Hex dump lines of sections other than the module's data are most of the text: skip them
    -- before any copying.
    if mode == "" && (starts (dropBlank ln) "0x" || starts ln " " || starts ln "\t") then continue
    let line := ln.toString
    let t := ln.trim.toString
    let cols := (t.splitOn " ").filter (!·.isEmpty) |>.flatMap (·.splitOn "\t") |>.filter (!·.isEmpty)
    if t.startsWith ".section .symtab" then mode := "symtab"; continue
    if t.startsWith ".section .rela." || t.startsWith ".section .rel." then
      mode := "rela"
      let lead := if t.startsWith ".section .rela." then ".section .rela" else ".section .rel"
      cur := trim (((t.drop lead.length).toString.splitOn "\t").headD "")
      continue
    if t.startsWith ".section" then mode := ""; continue
    if !line.startsWith " " && !line.startsWith "\t" && t.startsWith "." then
      -- A section's dump starts: the module's data (`.nv.global.init`, the module-wide banks
      -- `.nv.constant3`, `.nv.constant4`) are kept, others skipped.
      let bankNo := (t.drop ".nv.constant".length).toString
      let data := t == ".nv.global.init" ||
        (t.startsWith ".nv.constant" && !bankNo.isEmpty && bankNo.all Char.isDigit && bankNo != "0")
      if data then mode := "dump"; cur := t else mode := ""
      continue
    match mode, cols with
    | "symtab", [_, v, sz, _, _, sh, name] =>
      match num? v, num? sz, num? sh with
      | some v, some sz, some sh => syms := syms.push (name, { value := v, size := sz, shndx := sh })
      | _, _, _ => pure ()
    | "rela", [off, sym, "R_CUDA_64", add] =>
      if let (some o, some a) := (num? off, num? add) then relas := relas.push (cur, o, sym, some a)
    | "rela", [off, sym, "R_CUDA_64"] =>
      if let some o := num? off then relas := relas.push (cur, o, sym, none)
    | "dump", toks =>
      if toks.all (·.startsWith "0x") && !toks.isEmpty then
        let ns := (toks.filterMap num?).toArray
        dumps := match dumps.findIdx? (·.1 == cur) with
          | some k => dumps.modify k fun (n, l) => (n, l ++ ns)
          | none => dumps.push (cur, ns)
    | _, _ => pure ()
  let sizeOf (n : String) : Nat := (sects.find? (·.2.1 == n)).map (·.2.2) |>.getD 0
  let bytesOf (n : String) : ByteArray :=
    dumpBytes ((dumps.find? (·.1 == n)).map (·.2.toList) |>.getD []) (sizeOf n)
  -- Place the data sections, 256-byte aligned: the device data, and bank 3's contents (its
  -- bytes, relocated, are written below), whose `__constant__` variables a host copy can write.
  let mut placed : Array (String × Nat) := #[]
  let mut data : Array (Nat × Array UInt8) := #[]
  let mut at_ := base
  for name in [".nv.global.init", ".nv.global", ".nv.constant3"] do
    let size := sizeOf name
    if size == 0 then continue
    placed := placed.push (name, at_)
    if name != ".nv.constant3" then
      let b := if name == ".nv.global" then Array.replicate size 0 else (bytesOf name).data
      data := data.push (at_, b)
    at_ := at_ + (size + 255) / 256 * 256
  -- Symbols: data symbols at their section's place; undefined functions as externs.
  let mut externs : Array (Nat × String) := #[]
  let mut addrOf : Array (String × Nat) := #[]
  let mut symbols : Array (String × Nat × Nat) := #[]
  for (name, sym) in syms do
    if sym.shndx == 0 then
      if name != "(null)" && !name.startsWith "." then
        let a := externBase + 16 * externs.size
        externs := externs.push (a, name)
        addrOf := addrOf.push (name, a)
    else if let some (_, sname, _) := sects.find? (·.1 == sym.shndx) then
      if let some (_, sbase) := placed.find? (·.1 == sname) then
        addrOf := addrOf.push (name, sbase + sym.value)
        if !name.startsWith "." then symbols := symbols.push (name, sbase + sym.value, sym.size)
  -- Module banks, relocated.
  let mut banks : Array (Nat × Array UInt32) := #[]
  for (_, name, size) in sects do
    let suffix := (name.drop ".nv.constant".length).toString
    let some b := (if name.startsWith ".nv.constant" && !suffix.contains '.' then num? suffix
        else none) | continue
    if b == 0 then continue
    let mut ws := wordsOf (bytesOf name)
    ws := ws ++ Array.replicate ((size + 3) / 4 - ws.size) 0
    for r in relas do
      let (sect, off, sym, add) := r
      if sect == name then
        let implicit := (ws.getD (off / 4) 0).toNat + (ws.getD (off / 4 + 1) 0).toNat * 2 ^ 32
        let a := ((addrOf.find? (·.1 == sym)).map (·.2)).getD 0 + add.getD implicit
        ws := (ws.set! (off / 4) a.toUInt32).set! (off / 4 + 1) (a >>> 32).toUInt32
    banks := banks.push (b, ws)
  let bank3 := (placed.find? (·.1 == ".nv.constant3")).map (·.2)
  if let (some b3, some (_, ws)) := (bank3, banks.find? (·.1 == 3)) then
    data := data.push (b3, (ws.toList.flatMap fun w =>
      (List.range 4).map fun j => (w >>> (8 * j).toUInt32).toUInt8).toArray.extract 0 (sizeOf ".nv.constant3"))
  return { data, banks, externs, symbols, bank3 }

end NV.SASS.Elf
