import Tests.Kernels
import NV.SASS.Opcodes

/-!
# Reader tests

Reads the listings in `tests/sass/` (made by `tests/sass/gen.sh` with the pinned CUDA tools) and
checks, for every kernel of every source:

* the `cuobjdump -sass` and `nvdisasm -hex` listings of the same cubin read to the same
  instructions (one prints branch targets as offsets, the other as labels);
* the reuse flags decoded from the control word are as many as the `.reuse` operands printed;
* every dependence counter an instruction waits on is set by some instruction of the kernel;

and then specific instructions and attributes against values read off the listings by hand.

Last, the H200's fixed latencies against the fixtures' spacing between a producer and its first
consumer. Memory instructions are left out as consumers: they read some operands after issue.
-/

open NV NV.SASS

def fixtures : List String := ["basic", "misc", "wgmma", "tma", "elect", "ops", "jumps"]

def popcount (n : Nat) : Nat := (List.range 8).countP fun i => n / 2 ^ i % 2 == 1

def printedReuse (i : Instr) : Nat :=
  i.operands.toList.countP fun | .reg _ m => m.reuse | _ => false

/-- Problems with one kernel's two readings and its control words. -/
def kernelProblems (k k' : Kernel) : List String := Id.run do
  let mut out := []
  if k.code.size != k'.code.size then
    out := s!"{k.name}: {k.code.size} instructions from cuobjdump, {k'.code.size} from nvdisasm" :: out
  for (i, i') in k.code.toList.zip k'.code.toList do
    if i != i' then out := s!"{k.name} {i.pc}: cuobjdump {repr i} ≠ nvdisasm {repr i'}" :: out
    if popcount i.ctrl.reuse != printedReuse i then
      out := s!"{k.name} {i.pc} {i.opcode}: reuse bits {i.ctrl.reuse}, printed {printedReuse i}" :: out
  let set := k.code.toList.flatMap fun i => i.ctrl.wbar.toList ++ i.ctrl.rbar.toList
  for i in k.code do
    for b in List.range 6 do
      if i.ctrl.wait / 2 ^ b % 2 == 1 && !set.any (·.val == b) then
        out := s!"{k.name} {i.pc} {i.opcode}: waits on counter {b}, which nothing sets" :: out
  return out.reverse

def find (ks : Array Kernel) (name : String) : Option Kernel := ks.find? (·.name == name)

def at? (k : Kernel) (pc : Nat) : Option Instr := k.code.find? (·.pc == pc)

def r (i : Nat) : Operand := .reg ⟨.r, i⟩
def ur (i : Nat) : Operand := .reg ⟨.ur, i⟩

/-- An sm_80 fixture's ELF with its `EIATTR_REGCOUNT` records taken out (as some code
has none): each kernel's register count still read, from its `.text` section's `sh_info`. -/
def regsWithoutAttribute : IO (List (String × Bool)) := do
  let dir := "tests/sass/sm_80"
  let sass ← IO.FS.readFile s!"{dir}/diff-f32.sass"
  let elf ← IO.FS.readFile s!"{dir}/diff-f32.elf"
  -- Drop each three-line REGCOUNT record (attribute, format, value).
  let ls := elf.splitOn "\n"
  let keep := (List.range ls.length).filter fun i =>
    !((ls[i]!.splitOn "EIATTR_REGCOUNT").length > 1 ||
      (i ≥ 1 && (ls[i - 1]!.splitOn "EIATTR_REGCOUNT").length > 1) ||
      (i ≥ 2 && (ls[i - 2]!.splitOn "EIATTR_REGCOUNT").length > 1))
  let stripped := "\n".intercalate (keep.map (ls[·]!))
  match Read.kernels sass elf, Read.kernels sass stripped with
  | .ok ks, .ok ks' =>
    return ks.toList.map fun k =>
      (s!"{k.name} registers without EIATTR_REGCOUNT", k.regs > 0 && (find ks' k.name).map (·.regs) == some k.regs)
  | _, _ => return [("diff-f32 read", false)]

/-- The same fixture without its `EIATTR_CBANK_PARAM_SIZE` records (as an sm_90 cubin in
PyTorch 2.9's wheel has none): each kernel's parameter block still covers its parameters. -/
def paramBytesWithoutAttribute : IO (List (String × Bool)) := do
  let dir := "tests/sass/sm_80"
  let sass ← IO.FS.readFile s!"{dir}/diff-f32.sass"
  let elf ← IO.FS.readFile s!"{dir}/diff-f32.elf"
  let ls := elf.splitOn "\n"
  let keep := (List.range ls.length).filter fun i =>
    !((ls[i]!.splitOn "EIATTR_CBANK_PARAM_SIZE").length > 1 ||
      (i ≥ 1 && (ls[i - 1]!.splitOn "EIATTR_CBANK_PARAM_SIZE").length > 1) ||
      (i ≥ 2 && (ls[i - 2]!.splitOn "EIATTR_CBANK_PARAM_SIZE").length > 1))
  let stripped := "\n".intercalate (keep.map (ls[·]!))
  match Read.kernels sass elf, Read.kernels sass stripped with
  | .ok ks, .ok ks' =>
    return ks.toList.map fun k =>
      (s!"{k.name} parameter block without EIATTR_CBANK_PARAM_SIZE",
       k.paramBytes > 0 && (find ks' k.name).map (·.paramBytes) == some k.paramBytes)
  | _, _ => return [("diff-f32 read", false)]

/-- Expectations read off the listings: (description, holds). -/
def golden (all : List (String × Array Kernel)) : List (String × Bool) :=
  let kernel (src name : String) : Option Kernel := (all.lookup src).bind (find · name)
  let instr (src name : String) (pc : Nat) : Option Instr := (kernel src name).bind (at? · pc)
  let ops (src name : String) (pc : Nat) : Option (Array Operand) :=
    (instr src name pc).map (·.operands)
  [ ("vadd attributes", (kernel "basic" "vadd").map (fun k =>
      k.regs == 12 && k.params == #[(0, 8), (8, 8), (16, 8), (24, 4)] && k.paramBytes == 0x1c &&
      k.paramBase == 0x210 && k.sharedBytes == 0 && k.cluster == none) == some true),
    -- `LDG.E R2, desc[UR4][R2.64]`, high word 0x000ee2000c1e1900: sets counter 3.
    ("vadd LDG control", (instr "basic" "vadd" 0xd0).map (fun i =>
      i.opcode == "LDG" && i.mods == #["E"] &&
      i.ctrl == { stall := 1, yield := false, wbar := some 3, rbar := none, wait := 0, reuse := 0 } &&
      i.operands == #[r 2, .mem (some ⟨.r, 2⟩) true none (some ⟨.ur, 4⟩) 0]) == some true),
    -- `FFMA R9, R2, R5, R2`, high word 0x008fca0000000002: waits on counter 3 and yields.
    ("vadd FFMA control", (instr "basic" "vadd" 0x110).map (fun i =>
      i.ctrl == { stall := 5, yield := true, wbar := none, rbar := none, wait := 8, reuse := 0 })
      == some true),
    ("red shared memory", (kernel "basic" "red").map (fun k => k.sharedBytes == 1024 && k.regs == 14)
      == some true),
    -- `LOP3.LUT P0, RZ, R9.reuse, 0x1f, RZ, 0xc0, !PT`: reuse of the first source slot.
    ("red LOP3 reuse", (instr "basic" "red" 0x80).map (fun i => i.ctrl.reuse == 1 &&
      i.operands == #[.reg ⟨.p, 0⟩, .reg Reg.RZ, .reg ⟨.r, 9⟩ { reuse := true }, .imm 0x1f,
        .reg Reg.RZ, .imm 0xc0, .reg Reg.PT { not := true }]) == some true),
    ("red SHFL operands", ops "basic" "red" 0xf0 ==
      some #[.reg Reg.PT, r 5, r 2, .imm 0x10, .imm 0x1f]),
    -- The mbarrier retry block at the end: `@!P0 BRA` back to it.
    ("spin branch target", (instr "misc" "spin" 0x100).map (fun i =>
      i.guard == some (⟨.p, 0⟩, true) && i.operands == #[.target 0x190]) == some true),
    ("cluster dims", (kernel "misc" "clus").map (·.cluster) == some (some (2, 1, 1))),
    ("HGMMA operands", (instr "wgmma" "wg" 0x70).map (fun i =>
      i.opcode == "HGMMA" && i.mods == #["64x64x16", "F32", "BF16"] &&
      i.operands == #[r 24, .gdesc ⟨.ur, 4⟩ #[], .reg Reg.RZ, .reg Reg.UPT { not := true }])
      == some true),
    ("HGMMA descriptor suffixes", (instr "wgmma" "wg_neg" 0x60).map (fun i =>
      i.operands[1]? == some (.gdesc ⟨.ur, 4⟩ #["negA", "tnspA", "tnspB"])) == some true),
    ("f16x2 immediate", (kernel "ops" "fimm").map (fun k => k.code.any fun i =>
      i.opcode == "HFMA2" && i.operands.contains (.imm 0x3400c000)) == some true),
    ("NaN immediate", (kernel "ops" "fimm").map (fun k => k.code.any fun i =>
      i.opcode == "FSEL" && i.operands.contains (.imm 0x7fffffff)) == some true),
    ("f64 immediate", (kernel "ops" "fimm").map (fun k => k.code.any fun i =>
      i.opcode == "DFMA" && i.operands.contains (.imm 0x4000000000000000)) == some true),
    -- `MUFU.RCP64H R3, 2.14748262400000000000e+09`: the binary64 2^31 - 2^10's high word.
    ("RCP64H high-word immediate", (kernel "ops" "fimm").map (fun k => k.code.any fun i =>
      i.opcode == "MUFU" && i.mods.contains "RCP64H" && i.operands.contains (.imm 0x41dfffff))
      == some true),
    -- 1e30f is 0x7149f2ca; nvdisasm prints it rounded to 21 digits.
    ("rounded f32 immediate", (kernel "ops" "fimm").map (fun k => k.code.any fun i =>
      i.opcode == "FADD" && i.operands.contains (.imm 0x7149f2ca)) == some true),
    ("cluster kernel", (kernel "ops" "clusterk").map (·.cluster) == some (some (2, 1, 1))),
    ("shared bytes", ((kernel "ops" "mmak").map (·.sharedBytes), (kernel "ops" "atomk").map
      (·.sharedBytes)) == (some 2048, some 256)),
    -- `LDC R2, c[0x2][R0]; …; BRX R2 -0x90`: a jump table in bank 2 of absolute offsets.
    ("jump table", (kernel "jumps" "jumps").map (fun k =>
      k.banks == #[(2, #[0x90, 0xb0, 0xd0, 0xf0])] &&
      (at? k 0x80).map (·.operands) == some #[r 2, .imm (0 - 0x90)]) == some true),
    -- `.text.red` at file offset 0x880, `.text.vadd` at 0xb80 (tests/sass/basic.elf)
    ("code offsets", ((kernel "basic" "vadd").map (·.codeOffset), (kernel "basic" "red").map
      (·.codeOffset)) == (some 0x300, some 0)),
    ("modules", ((kernel "basic" "vadd").map (·.module)) == (kernel "basic" "red").map (·.module) &&
      ((kernel "basic" "vadd").map (·.module)) != (kernel "misc" "spin").map (·.module)),
    ("local bytes", ((kernel "ops" "spillk").map (·.localBytes), (kernel "ops" "mmak").map
      (·.localBytes)) == (some 0x100, some 0)) ]

def main : IO UInt32 := do
  let dir := "tests/sass"
  let mut all : List (String × Array Kernel) := []
  let mut failures : List String := []
  let mut count := 0
  for f in fixtures do
    let sass ← IO.FS.readFile s!"{dir}/{f}.sass"
    let dis ← IO.FS.readFile s!"{dir}/{f}.nvdisasm"
    let elf ← IO.FS.readFile s!"{dir}/{f}.elf"
    match Read.kernels sass elf, Read.kernels dis elf with
    | .ok ks, .ok ks' =>
      all := all ++ [(f, ks)]
      for k in ks do
        count := count + k.code.size
        match find ks' k.name with
        | some k' => failures := failures ++ kernelProblems k k'
        | none => failures := failures ++ [s!"{f}: {k.name} missing from the nvdisasm listing"]
    | .error e, _ | _, .error e => failures := failures ++ [s!"{f}: {e}"]
  for (what, ok) in golden all ++ (← regsWithoutAttribute) ++ (← paramBytesWithoutAttribute) do
    if !ok then failures := failures ++ [s!"golden: {what}"]
  for (_, ks) in all do
    for k in ks do
      failures := failures ++ Tests.latencyProblems NV.Arch.h200 k
      for i in k.code do
        if (OpInfo.find? i.opcode).isNone && (undocumented.lookup i.opcode).isNone then
          failures := failures ++ [s!"{k.name} {i.pc}: {i.opcode} is not in the opcode catalogue"]
  for m in failures do IO.eprintln m
  IO.println s!"sass-tests: {count} instructions in {all.length} listings, {failures.length} failures"
  return if failures.isEmpty then 0 else 1
