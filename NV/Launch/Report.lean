import NV.Launch.Load
import NV.Bounds

/-!
A launch's report as `lake exe sim` prints it and the emulated device under `NVGPU_LEAN_UNTIL`: what
Nsight Compute would (cycles before, during and after the kernel, instructions, memory traffic,
where warps stalled), and the cycles each resource's work needs at its rate (`binding`).
-/

namespace NV.Launch.Report

open NV NV.Core

def pct (part whole : Nat) : String :=
  if whole == 0 then "0%" else s!"{(1000 * part / whole + 5) / 10}%"

def report (i : Nat) (name : String) (c : Counters) : String := Id.run do
  let m := c.memory
  let stalls := c.stalls.foldl (· + ·) 0
  let named := (Stall.all.map fun s => (((reprStr s).splitOn ".").getLast!, c.stalls[s.index]!)).toArray
    |>.qsort (·.2 > ·.2) |>.filter (·.2 > 0)
  return s!"launch {i}: {name}\n" ++
    s!"  cycles        {c.startCycles} start + {c.cycles} active + {c.endCycles} end = " ++
      s!"{c.startCycles + c.cycles + c.endCycles}\n" ++
    s!"  instructions  {c.instExecuted} warp instructions; global {c.globalLoads} ld, " ++
      s!"{c.globalStores} st; shared {c.sharedLoads} ld, {c.sharedStores} st\n" ++
    s!"  global        {m.globalLdRequests} load requests, {m.globalLdSectors} sectors " ++
      s!"({pct m.globalLdHitSectors m.globalLdSectors} L1 hits); {m.globalStRequests} store requests, " ++
      s!"{m.globalStSectors} sectors\n" ++
    s!"  L2            {m.l2ReadSectors} read, {m.l2WriteSectors} write, {m.l2AtomSectors} atomic sectors; " ++
      s!"{m.l2HitSectors} hit, {m.l2MissSectors} miss, {m.l2FarSectors} across the partitions\n" ++
    s!"  DRAM          {m.dramReadBytes} bytes read, {m.dramWriteBytes} written, {m.dramTurns} turns\n" ++
    s!"  shared        {m.sharedWavefronts} wavefronts, {m.sharedBankConflicts} bank conflicts\n" ++
    s!"  stalls        " ++ (if stalls == 0 then "none" else
      ", ".intercalate (named.map fun (n, v) => s!"{n} {pct v stalls}").toList) ++ "\n" ++
    String.join (c.unqualified.map fun why => s!"  unqualified   {why}\n")

/-- The busiest sub-partition pipe, named, and its cycles. -/
def pipeRow (c : Counters) : String × Nat :=
  let (i, v) := (List.range Pipe.count).foldl (fun (b : Nat × Nat) i =>
    if c.pipeBusyMax[i]! > b.2 then (i, c.pipeBusyMax[i]!) else b) (0, 0)
  let name := (Pipe.all.find? (·.index == i)).map (fun q => ((reprStr q).splitOn ".").getLast!) |>.getD "?"
  (s!"{name} pipe", v)

/-- The run's work per resource (its counters) as the cycles that resource needs at its rate
(`NV.Bounds.Part`): DRAM, the L2 slices, the busiest sub-partition's issue (one instruction a
cycle), the busiest tensor core's multiply-adds and its operand reads from shared memory, and the
busiest sub-partition pipe (its input port's cycles: `MUFU` on `xu`, say); the binding one is the
largest. A run far above every term is
bound by latency (its dependent chains), not by a resource. -/
def binding (p : Params) (c : Counters) : String := Id.run do
  let m := c.memory
  let dram := m.dramReadBytes + m.dramWriteBytes
  let l2 := (m.l2ReadSectors + m.l2WriteSectors + m.l2AtomSectors) * p.mem.sectorBytes.value
  let rows : List (String × Nat) := [
    ("DRAM", (NV.Bounds.Part.dram p).capacity.minCycles dram),
    ("L2", (NV.Bounds.Part.l2 p).capacity.minCycles l2),
    ("issue", c.issueMax),
    ("tensor", c.tensorBusyMax),
    ("tensor operands", c.memory.tensorPortMax),
    pipeRow c]
  let (top, need) := rows.foldl (fun b r => if r.2 > b.2 then r else b) ("none", 0)
  let t := max c.cycles 1
  return s!"  binding       {top}: {need} of {c.cycles} active cycles ({pct need t}); " ++
    ", ".intercalate (rows.map fun (n, v) => s!"{n} {v}") ++
    (if 2 * need < t then " (under half: latency-bound)" else "") ++ "\n"

end NV.Launch.Report
