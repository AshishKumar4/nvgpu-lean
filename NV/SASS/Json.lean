import NV.SASS.Syntax

/-!
# A kernel as JSON lines

A kernel as the simulator reads it, one JSON object per line, for the accelerated mode's lifter
(`tools/lift/lift.py`): first the kernel's attributes, then each instruction with its guard,
opcode, modifiers and typed operands. The lifter consumes the simulator's own parse, so it cannot
read an instruction differently from it. `lake exe sass-dump` and `sim --export` write it.
-/

namespace NV.SASS.Json

def jstr (s : String) : String := "\"" ++ (s.replace "\\" "\\\\" |>.replace "\"" "\\\"") ++ "\""

def jreg (r : Reg) : String :=
  let f := match r.file with | .r => "r" | .ur => "ur" | .p => "p" | .up => "up" | .b => "b"
  s!"\{\"f\":{jstr f},\"i\":{r.idx}}"

def jopt (o : Option Reg) : String := (o.map jreg).getD "null"

def jop : Operand → String
  | .reg r m =>
    s!"\{\"k\":\"reg\",\"r\":{jreg r},\"neg\":{m.neg},\"abs\":{m.abs},\"not\":{m.not}," ++
    s!"\"sfx\":[{",".intercalate (m.suffix.toList.map jstr)}]}"
  | .imm b => s!"\{\"k\":\"imm\",\"v\":{b.toNat}}"
  | .cbank bank base off => s!"\{\"k\":\"c\",\"bank\":{bank},\"base\":{jopt base},\"off\":{off}}"
  | .mem base wide ureg desc off =>
    s!"\{\"k\":\"mem\",\"base\":{jopt base},\"wide\":{wide},\"ureg\":{jopt ureg},\"desc\":{jopt desc},\"off\":{off}}"
  | .neg o => s!"\{\"k\":\"neg\",\"o\":{jop o}}"
  | .not o => s!"\{\"k\":\"not\",\"o\":{jop o}}"
  | .scaled m s => s!"\{\"k\":\"scaled\",\"m\":{jop m},\"scale\":{s}}"
  | .gdesc r ms => s!"\{\"k\":\"gdesc\",\"r\":{jreg r},\"mods\":[{",".intercalate (ms.toList.map jstr)}]}"
  | .udesc kind r off => s!"\{\"k\":\"udesc\",\"kind\":{jstr kind},\"r\":{jreg r},\"off\":{off}}"
  | .target pc => s!"\{\"k\":\"target\",\"pc\":{pc}}"
  | .name s => s!"\{\"k\":\"name\",\"s\":{jstr s}}"

/-- The kernel's lines. -/
def kernelLines (k : Kernel) : Array String :=
  let (cx, cy, cz) := k.cluster.getD (1, 1, 1)
  #[s!"\{\"kernel\":{jstr k.name},\"sm\":{k.sm},\"regs\":{k.regs},\"sharedBytes\":{k.sharedBytes}," ++
    s!"\"sharedReserved\":{k.sharedReserved},\"localBytes\":{k.localBytes},\"paramBase\":{k.paramBase}," ++
    s!"\"paramBytes\":{k.paramBytes},\"cluster\":[{cx},{cy},{cz}],\"instructions\":{k.code.size}," ++
    -- The kernel's own constant banks (bank 0 is the launch's), as words.
    s!"\"banks\":\{{",".intercalate (k.banks.toList.filter (·.1 != 0) |>.map fun (b, ws) =>
      s!"\"{b}\":[{",".intercalate (ws.toList.map toString)}]")}}}"] ++
  k.code.map fun i =>
    let g := match i.guard with | some (r, neg) => s!"\{\"r\":{jreg r},\"not\":{neg}}" | none => "null"
    s!"\{\"pc\":{i.pc},\"guard\":{g},\"op\":{jstr i.opcode}," ++
      s!"\"mods\":[{",".intercalate (i.mods.toList.map jstr)}],\"ops\":[{",".intercalate (i.operands.toList.map jop)}]}"

end NV.SASS.Json
