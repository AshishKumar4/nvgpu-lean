import NV.SASS.Read

/-!
# Reading compiled code

Queries over a kernel's SASS as `NV.SASS.Read` parses it: an instruction's registers read, counts,
first occurrences, loops and definitions before a point.
-/

namespace NV.SASS

def Instr.is (i : Instr) (full : String) : Bool := i.full == full

/-- The registers an instruction reads: register operands after the destinations, and the
registers inside its address and constant-bank operands. -/
def Instr.reads (i : Instr) (dsts : Nat) : List Reg :=
  (i.operands.toList.drop dsts).flatMap fun
    | .reg r _ => [r]
    | .mem b _ u d _ => b.toList ++ u.toList ++ d.toList
    | .cbank _ b _ => b.toList
    | .gdesc r _ | .udesc _ r _ => [r]
    | _ => []

def Kernel.count (k : Kernel) (p : Instr → Bool) : Nat := k.code.toList.countP p

def Kernel.has (k : Kernel) (full : String) : Bool := k.code.any (·.is full)

/-- The highest vector register a kernel names (RZ aside), from its operands. -/
def Kernel.topReg (k : Kernel) : Nat :=
  let ofReg (r : Reg) : Nat := if r.file == .r && r.idx != 255 then r.idx else 0
  let rec ofOp : Operand → Nat
    | .reg r _ => ofReg r
    | .mem b _ _ _ _ => (b.map ofReg).getD 0
    | .cbank _ b _ => (b.map ofReg).getD 0
    | .neg o | .not o => ofOp o
    | _ => 0
  k.code.foldl (fun m i => i.operands.foldl (fun m o => max m (ofOp o)) m) 0

/-- The index of the first instruction satisfying `p`. -/
def Kernel.first (k : Kernel) (p : Instr → Bool) : Option Nat := k.code.findIdx? p

/-- Loops: the instruction indices `(head, branch)` of each backward branch. -/
def Kernel.loops (k : Kernel) : List (Nat × Nat) :=
  k.code.toList.zipIdx.filterMap fun (i, n) =>
    match i.opcode, i.operands.toList.getLast? with
    | "BRA", some (.target t) =>
      if t ≤ i.pc then (k.code.findIdx? (·.pc == t)).map (·, n) else none
    | _, _ => none

/-- The instructions of a loop, head to branch. -/
def Kernel.body (k : Kernel) (loop : Nat × Nat) : List Instr :=
  (k.code.toList.drop loop.1).take (loop.2 + 1 - loop.1)

/-- Consecutive instructions. -/
def Kernel.adjacent (k : Kernel) : List (Instr × Instr) := k.code.toList.zip (k.code.toList.drop 1)

/-- The register an instruction writes first, if its first operand is a register. -/
def Instr.dst? (i : Instr) : Option Reg :=
  match (i.operands[0]? : Option Operand) with
  | some (.reg r _) => some r
  | _ => none

/-- The last instruction before index `j` whose first operand is register `r`. -/
def Kernel.defBefore (k : Kernel) (j : Nat) (r : Reg) : Option Instr :=
  ((k.code.toList.take j).filter (·.dst? == some r)).getLast?

end NV.SASS
