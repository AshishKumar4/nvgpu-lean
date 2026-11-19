import NV.Bounds.Chain
import NV.Bounds.Machine
import NV.SASS.Query

/-!
# The dependence chain of a SASS path

The stream one warp issues along a straight path through a kernel, as a `Stream` whose
dependences are true ones, so that `Stream.critical` bounds every run in which a warp takes the
path:

* program order: a warp issues in order, at most one instruction per cycle;
* register flow: an instruction that reads a register waits for the latest earlier writer of it
  on the path, by the writer's latency. Only writers whose first operand is certainly a
  destination are tracked (`writesFirst`; compares write their first two), and a guarded writer
  is not (it may not write); a dependence left out only weakens the bound.

Latencies are the opcode table's; a load's is the caller's, with the assumption it rests on (on a
fresh machine a load of data no earlier access touched misses every cache, so it takes at least
the idle DRAM latency; an L1 hit latency is the least any load takes).

The path is the code from the first instruction to the first unguarded `EXIT`; it stops early at an
unguarded branch, so it is a prefix of what every warp that reaches that `EXIT` issues.

Each instruction also holds its pipe for the pipe's interval and its sub-partition's issue slot
for a cycle, as the engine's ports do. `Stream.copies g w` is `w` warps on one sub-partition:
with `w` the warps of the launch over the sub-partitions, rounded up, some sub-partition runs at
least that many, so `Stream.bound_le` bounds the run by the busiest pipe's work too.
-/

namespace NV.Bounds

open NV NV.SASS NV.Core

/-- Opcodes whose first operand is the register they write. -/
def writesFirst : List String :=
  ["FFMA", "FADD", "FMUL", "FMNMX", "IMAD", "IADD3", "VIADD", "LOP3", "SHF", "LEA", "MOV", "PRMT",
   "SEL", "FSEL", "S2R", "S2UR", "CS2R", "LDC", "ULDC", "LDG", "LDS", "LDSM", "UMOV", "UIADD3",
   "UIMAD", "ULOP3", "USHF", "ULEA", "MUFU", "F2F", "F2I", "I2F", "HFMA2", "HMUL2", "HADD2",
   "DFMA", "DADD", "DMUL", "I2I", "F2FP", "IABS", "POPC", "FLO", "BREV"]

/-- Compares that write two predicates. -/
def writesTwo : List String := ["ISETP", "FSETP", "DSETP", "UISETP", "PLOP3", "HSETP2"]

def instrDefs (i : Instr) : Nat :=
  if writesTwo.contains i.opcode then 2 else if writesFirst.contains i.opcode then 1 else 0

/-- The registers it writes that the chain tracks: none if it is guarded. -/
def tracked (i : Instr) : List Reg :=
  if i.guard.isSome then [] else
    (i.operands.toList.take (instrDefs i)).filterMap fun | .reg r _ => if r.isSink then none else some r | _ => none

/-- The registers it reads: its sources, address registers and guard. -/
def sources (i : Instr) : List Reg :=
  (i.reads (instrDefs i)).filter (!·.isSink) ++ (i.guard.map (·.1)).toList

/-- The path: from the first instruction to the first unguarded `EXIT`, stopping before an
unguarded branch. -/
def path (k : Kernel) : List Instr :=
  let rec go : List Instr → List Instr
    | [] => []
    | i :: rest =>
      if i.opcode == "EXIT" && i.guard.isNone then [i]
      else if (i.opcode == "BRA" || i.opcode == "JMP") && i.guard.isNone then []
      else i :: go rest
  go k.code.toList

/-- Resource identifiers in a path's stream: the sub-partition's pipes by `Pipe.index`, and its
issue slot. -/
def issueSlot : Nat := Pipe.count

def subPartitionResources : List Nat := List.range (Pipe.count + 1)

/-- The stream of the path, with `loadLatency` cycles for global loads. Each instruction holds
its pipe for the pipe's interval and the issue slot for a cycle. -/
def pathStream (p : Params) (k : Kernel) (loadLatency : Nat) : Stream := Id.run do
  let lat (i : Instr) : Nat :=
    if i.opcode == "LDG" then loadLatency
    else if i.opcode == "LDS" || i.opcode == "LDSM" then p.mem.smemLatency.value
    else ((p.timing i).map (·.latency)).getD 1
  let mut last : List (Reg × Nat × Nat) := []
  let mut out : Array Node := #[]
  for i in path k do
    let j := out.size
    let order := if j = 0 then [] else [(j - 1, 1)]
    let flow := (sources i).filterMap fun r => (last.find? (·.1 == r)).map (·.2)
    let uses := match p.timing i with
      | some tm => [(tm.pipe.index, tm.interval), (issueSlot, 1)]
      | none => [(issueSlot, 1)]
    out := out.push { preds := order ++ flow, done := 1, uses }
    for r in tracked i do
      last := (r, j, lat i) :: last.filter (·.1 != r)
  return out

/-- `w` copies of a stream, independent of each other: the warps one sub-partition runs, each
along the path. -/
def Stream.copies (g : Stream) (w : Nat) : Stream := Id.run do
  let mut out : Array Node := #[]
  for c in [0:w] do
    let off := c * g.size
    for nd in g do
      out := out.push { nd with preds := nd.preds.map fun q => (q.1 + off, q.2) }
  return out

/-- The timing facts of the opcodes on the path. -/
def pathFacts (p : Params) (k : Kernel) : List FactRef := Id.run do
  let mut seen : List String := []
  let mut out : List FactRef := []
  for i in path k do
    if seen.contains i.opcode then continue
    seen := i.opcode :: seen
    if let some r := p.ops.find? (·.matches i) then
      let fact (v : Nat) : Fact Nat :=
        { value := v, kind := r.timing.kind, source := r.timing.source,
          measuredOn := r.timing.measuredOn, note := r.timing.note }
      out := out ++ [FactRef.of s!"ops.{i.opcode}.latency" (fact r.timing.value.latency),
        FactRef.of s!"ops.{i.opcode}.interval" (fact r.timing.value.interval)]
  return out

end NV.Bounds
