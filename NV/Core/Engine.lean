import NV.Core.Params
import NV.Core.Heap
import NV.Core.Memory
import NV.Core.Async
import NV.Core.Port
import NV.SASS.Sem
import NV.SASS.Footprint
import NV.Core.Pending
import NV.Memory.Atomic
import NV.Memory.Events
import NV.Machine.Trace
import Std.Data.HashMap
import Std.Data.HashSet

/-!
# The engine

Runs a SASS kernel's grid on a `Params` machine: CTAs are dispatched to SMs as occupancy allows;
each cycle every sub-partition's scheduler issues at most one instruction from an eligible warp.
A warp is eligible when its stall count has elapsed, the dependence counters in the instruction's
wait mask are zero (and a `DEPBAR.LE`'s counter is low enough), and the instruction's pipe accepts
work (Huerta et al. 2025 §4–5). The scheduler keeps issuing from the warp it issued last while that
warp is eligible, and otherwise takes the oldest or youngest eligible warp (`IssuePolicy`).

Data are computed only by `NV.SASS.exec`; the engine decides when. Register results are written
back after their class's latency. A memory instruction takes a slot of its sub-partition's MIO
queue at issue (it cannot issue while the queue is full) and reads its sources when its read
counter is released; its request then goes to the memory system (`NV.Core.Memory`), which
performs it and delivers it at a cycle of its timing: the delivery writes the response and
releases the instruction's write counter. A fence holds its warp until the memory system
delivers it, after every earlier request of the warp.

Time advances by events: every warp's next possible issue cycle is known when it is set (stall
counts, dependence-counter releases, pipe occupancy, barrier releases), so the engine visits only
the cycles at which some SM can issue, and an SM only when it can.
-/

namespace NV.Core

open NV NV.SASS

structure Launch where
  kernel : Kernel
  grid : Nat × Nat × Nat
  block : Nat × Nat × Nat
  /-- The parameter block, laid out as the kernel's ELF says. -/
  params : ByteArray
  global : Store
  dynShared : Nat := 0
  /-- Workers the SMs are dealt to (0: 8); one runs them all in turn. Timed results do not depend
  on it; functional ones only where the program's outcome depends on how its threads interleave. -/
  workers : Nat := 0
  /-- The functional mode (no timing): each instruction completes at issue — results written,
  memory requests performed on the stores at once (`MemSys.performNow`), asynchronous operations
  run to completion (`Op.run`) — and a warp is ready whenever it is not at a barrier. Warps still
  interleave, so the run is one of the program's outcomes; the cycle count is not a time. Over
  several workers, an SM sees another worker's global writes from the next quantum (the memory
  model promises no sooner), and a global atomic that returns a value is performed by the master
  at the quantum's end, its warp waiting for it. -/
  functional : Bool := false
  /-- Print every issued instruction (cycle, SM, CTA, warp, pc, opcode) to standard error, to find
where a run goes wrong. -/
  trace : Bool := false
  /-- Print each warp of CTA 0's registers (per register, its 32 lanes in hex) to standard error
  when it issues the instruction at this byte offset, to inspect values mid-kernel. -/
  dumpAt : Option Nat := none
  /-- CTAs per cluster from the launch attribute; (1, 1, 1) takes the kernel's required shape
  (`Kernel.cluster`), if it declares one (`Launch.clusterShape`). -/
  cluster : Nat × Nat × Nat := (1, 1, 1)
  /-- The launch's texture and surface objects; handle `i + 1` names entry `i` (`NV.Tex`). -/
  textures : Array NV.Tex.Texture := #[]
  /-- Record each warp's trace (`Result.traces`), which `NV.Formal.Trace.check` replays against the
  program semantics. -/
  recordTrace : Bool := false
  /-- Check the threads of each CTA for data races on shared and global memory
  (`NV.Core.Memory.RaceCheck`; `Result.races`). -/
  races : Bool := false
  /-- Stop the launch at this cycle, its counters those of the cycles run (a sample of a launch too
  long to simulate whole; its memory is then not the launch's result). -/
  untilCycle : Option Nat := none
  /-- Each CTA's span to standard error as it finishes, `cta ID sm SM start S end E` (its linear
  index, the cycle it was placed at and its last warp's exit): what a launch's per-CTA work costs,
  for scheduling the same CTAs over a larger grid. -/
  spans : Bool := false
  /-- Where the kernel's code sits in device memory: each module's code image in its own 16 MB slot
  (named by `Kernel.module`), its kernels in file order (`Kernel.codeOffset`), so prefetching runs
  across kernel boundaries as on silicon and modules don't alias in a `Machine`'s caches. -/
  codeBase : Nat := kernel.module.toNat % 65536 * 16777216 + kernel.codeOffset

/-- Why a warp did not issue, after Nsight Compute's warp states
(`smsp__pcsamp_warps_issue_stalled_*`). -/
inductive Stall where
  /-- The previous instruction's stall count, yield, or a taken branch's redirect (`wait`,
  `branch_resolving`). -/
  | wait
  /-- A dependence counter set by a memory instruction (`long_scoreboard`). -/
  | longScoreboard
  /-- A dependence counter set by another variable-latency instruction (`short_scoreboard`). -/
  | shortScoreboard
  /-- The instruction's pipe busy (`math_pipe_throttle`). -/
  | pipe
  /-- The register file's read stage busy (bank conflicts). -/
  | regRead
  /-- The memory pipeline's queue full (`mio_throttle`, `lg_throttle`). -/
  | mio
  /-- The next instruction not yet fetched (`no_instruction`). -/
  | fetch
  /-- Waiting at a CTA barrier (`barrier`). -/
  | barrier
  /-- Waiting for a fence (`membar`). -/
  | fence
  /-- Eligible, but the scheduler issued another warp (`not_selected`). -/
  | notSelected
  /-- A cycle the sub-partition lost to its tensor core (`dispatch_stall`;
  `TensorParams.wgmmaIssueSharePermille`). -/
  | dispatch
  /-- Asleep after a `NANOSLEEP` (`sleeping`). -/
  | sleeping
  deriving DecidableEq, Repr, Inhabited

def Stall.index : Stall → Nat
  | .wait => 0 | .longScoreboard => 1 | .shortScoreboard => 2 | .pipe => 3 | .regRead => 4
  | .mio => 5 | .fetch => 6 | .barrier => 7 | .fence => 8 | .notSelected => 9 | .dispatch => 10
  | .sleeping => 11

def Stall.all : List Stall :=
  [.wait, .longScoreboard, .shortScoreboard, .pipe, .regRead, .mio, .fetch, .barrier, .fence,
   .notSelected, .dispatch, .sleeping]

/-- The launch's cluster shape: the attribute's, else the kernel's required one. -/
def Launch.clusterShape (l : Launch) : Nat × Nat × Nat :=
  if l.cluster != (1, 1, 1) then l.cluster else l.kernel.cluster.getD (1, 1, 1)

/-- Counters, named after the Nsight Compute metrics they correspond to. -/
structure Counters where
  /-- Instructions that issued while a register they read or write had a writeback pending, or
  wrote a register a pending request had yet to read: control bits that under-wait (a stale
  read or a lost write on silicon). The bridge to the program semantics assumes none. -/
  hazards : Nat := 0
  /-- Warp-cycles each live warp spent not issuing, by reason (`Stall.index`): the sum over warps
  of the cycles between its issues, attributed to the reason that held it last. -/
  stalls : Array Nat := Array.replicate 12 0
  /-- `gpc__cycles_elapsed.max`: cycles from the first CTA's start to the last warp's exit or the
  last memory request's delivery, whichever is later. -/
  cycles : Nat := 0
  /-- Cycles from the launch to the first CTA issuing (`LaunchParams`: the plain kernel gap less
  the PDL gap), and from the last warp's exit to the kernel's completion being visible to its
  dependents (the PDL gap). A source that times a kernel from launch to completion compares
  with `startCycles + cycles + endCycles`. -/
  startCycles : Nat := 0
  endCycles : Nat := 0
  /-- `smsp__inst_executed.sum`: warp instructions issued. -/
  instExecuted : Nat := 0
  /-- `smsp__inst_executed_op_global_ld.sum`, `…_st.sum`. -/
  globalLoads : Nat := 0
  globalStores : Nat := 0
  /-- `smsp__inst_executed_op_shared_ld.sum`, `…_st.sum`. -/
  sharedLoads : Nat := 0
  sharedStores : Nat := 0
  /-- CTAs of the launch one SM holds at once (its occupancy: the least of the CTA, thread,
  register and shared-memory limits). -/
  ctasPerSM : Nat := 0
  /-- Bytes the SMs' TMA units requested, and warpgroup MMA instructions (one per warp). -/
  tmaBytes : Nat := 0
  gmmaInsts : Nat := 0
  /-- The busiest tensor-core port's cycles of multiply-adds (`NV.Core.Async` `tensorBusy`). -/
  tensorBusyMax : Nat := 0
  /-- Per pipe (`Pipe.index`): the busiest sub-partition's cycles of that pipe's input port taken
  by the instructions it issued (each its opcode's `interval`). -/
  pipeBusyMax : Array Nat := Array.replicate Pipe.count 0
  /-- The busiest sub-partition's warp instructions issued (one a cycle at most). -/
  issueMax : Nat := 0
  memory : Memory.Counters := {}
  deriving Repr, Inhabited

/-- Two SMs' counters as one: sums, and the later end. -/
private def Counters.merge (a b : Counters) : Counters :=
  { a with
    hazards := a.hazards + b.hazards
    stalls := (a.stalls.zip b.stalls).map fun (x, y) => x + y
    cycles := max a.cycles b.cycles
    instExecuted := a.instExecuted + b.instExecuted
    globalLoads := a.globalLoads + b.globalLoads, globalStores := a.globalStores + b.globalStores
    sharedLoads := a.sharedLoads + b.sharedLoads, sharedStores := a.sharedStores + b.sharedStores }

/-- One device `printf` record: when and where it was produced, its format string's address and
its argument bytes (`NV.SASS.printfText` renders it). -/
structure PrintfRecord where
  cycle : Nat
  sm : Nat
  cta : Nat
  warp : Nat
  format : UInt64
  args : ByteArray

/-- What the engine did with one warp (`Launch.recordTrace`): its CTA (linear index) and warp,
its environment and state at the start, and its trace. -/
structure WarpTrace where
  cta : Nat
  warp : Nat
  env : Env
  init : Warp
  evs : Array TraceEv := #[]
  deriving Inhabited

structure Result where
  global : Store
  counters : Counters
  /-- The launch's `printf` output, in the order the records were produced. -/
  printf : Array PrintfRecord := #[]
  /-- The first hazards (`Counters.hazards`), described. -/
  hazards : Array String := #[]
  /-- The data races found (`Launch.races`), up to 16 per SM, described. -/
  races : Array String := #[]
  /-- Each warp's trace, CTA by CTA, when the launch records them. -/
  traces : Array WarpTrace := #[]

/-- Work in flight for warp `slot` of SM `sm`. -/
private inductive Ev where
  /-- A result: written into the warp. -/
  | wb (f : Warp → Warp) (pc : Option Nat := none) (land : Option Nat := none)
  /-- A fixed-latency instruction that reads its sources after issue: run it on the warp as it is
  now, and write its result at `at_`. -/
  | late (env : Env) (m : UInt32) (pc : Nat) (at_ : Nat) (land : Nat)
  /-- A memory instruction reading its sources: its request is built from the warp and goes to
  the memory system; its delivery writes `wb` and releases dependence counter `bar`. `load` numbers
  it among the warp's loads (0: it writes no register); it reads registers loads `after` write. -/
  | mem (req : Warp → Req) (wb : Resp → Warp → Warp) (bar : Option (Fin 6)) (issued : Nat) (pc : Nat)
    (readsPending : Bool) (tix land : Nat) (load : Nat) (after : Array Nat)
  /-- A fence reaching the memory system behind the warp's earlier requests. -/
  | fence (cost : Nat)
  /-- An asynchronous instruction reading its sources: its operation is built from the warp and
  handed to the async units (`NV.Core.Async`). -/
  | async (op : Warp → NV.Async.Op) (wb : Resp → Warp → Warp) (pc : Nat) (tix land : Nat)
  /-- A commit of the warp's open async operations of a kind, after the operations issued before
  it have reached the async units; `bar`'s pending entry is released when the group completes. -/
  | commit (g : NV.Async.Group) (bar : Option (Fin 6)) (onRead : Bool)
  deriving Inhabited

private structure Slot where
  w : Warp
  env : Env
  cta : Nat
  smsp : Nat
  age : Nat
  /-- The cycle the warp last issued (`IssuePolicy.leastRecent`). -/
  issuedAt : Nat := 0
  readyAt : Nat := 0
  /-- Per dependence counter: release cycles of its outstanding events. -/
  sb : Array (Array Nat) := Array.replicate 6 #[]
  atBar : Option Nat := none
  /-- Registers a pending writeback will write, and registers a pending request has yet to read,
  with their counts (the hazard check). -/
  pw : Pending := {}
  pr : Pending := {}
  /-- The warp's loads (memory instructions writing registers), numbered from 1: the next number,
  per location the latest load writing it, and those not yet delivered. A request reading a
  register an earlier load has yet to write is held until that load's writeback, with the
  requests behind it (`readsBehindLoad`). -/
  loads : Nat := 1
  loadOf : Std.HashMap Nat Nat := {}
  inflight : Std.HashSet Nat := {}
  held : Array Ev := #[]
  /-- The warp's shared-memory loads (`LDS`, `LDSM`) not yet delivered, and its shuffles issued
  behind them: a shuffle's result returns through the MIO after theirs, in issue order (inferred).
  Each held shuffle: the shared loads it follows, the cycle its latency alone would land it, its
  writeback and its counter. -/
  sharedLoads : Std.HashSet Nat := {}
  mioHeld : Array (Array Nat × Nat × Ev × Option (Fin 6)) := #[]
  /-- The cluster barrier phase the warp last arrived in, and whether it waits for that phase. -/
  clusterPhase : Nat := 0
  atClusterWait : Bool := false
  live : Bool := true
  /-- The cycle the warp's latest memory request reads its sources; requests reach the memory
  system in issue order. -/
  lastMem : Nat := 0
  /-- Waiting for a fence to deliver. -/
  atFence : Bool := false
  /-- Suspended in a `try_wait` (`SYNCS.PHASECHK.TRANS64.TRYWAIT`) until it delivers: the warp
  issues nothing meanwhile (on the GH200, a try_wait on a barrier that never completes takes
  1941 cycles). -/
  atTryWait : Bool := false
  /-- Dependence counters whose latest event came from a memory instruction, one bit each. -/
  sbMem : Nat := 0
  /-- Whether the warp's last instruction had a stall count of 0 on a part where that interlocks
  (`FrontEndParams.stallZeroInterlocks`): the next waits for the fixed-latency results it reads. -/
  interlock : Bool := false
  /-- Per predicate: the cycle a pending write of it can first guard an instruction
  (`OpTiming.guardLatency`), for the hazard check. -/
  guardAt : Std.HashMap NV.SASS.Loc Nat := {}
  /-- Why the warp is waiting now, and the cycle of its last issue (for `Counters.stalls`). -/
  why : Stall := .wait
  lastIssue : Option Nat := none
  /-- Cycles since its last issue it was ready in but lost to its sub-partition's tensor core
  (`Stall.dispatch`), apart from the reason it waited for the rest. -/
  lost : Nat := 0
  /-- The functional mode over several workers: the writeback of the global atomic the warp waits
  on, which the master performs as the quantum ends. -/
  atomWb : Option (Resp → Warp → Warp) := none
  /-- The code line the warp last fetched from. -/
  codeLine : Option Nat := none
  /-- The id the warp's next pending writeback takes in its trace. -/
  nextLand : Nat := 0
  deriving Inhabited

/-- A warp that has exited and has nothing in flight keeps no registers or pending tables: a
launch's retired warps would otherwise hold their register files to its end. -/
private def Slot.settle (s : Slot) : Slot :=
  if s.live || !s.held.isEmpty || !s.mioHeld.isEmpty || !s.pw.isEmpty || !s.pr.isEmpty then s
  else { s with w := Warp.init 0 0, pw := {}, pr := {}, loadOf := {} }

/-- A CTA's barriers: the 16 named ones (`BAR`), then one per warpgroup (`WARPGROUP.WAIT`, 8 for
1024 threads). -/
def ctaBarriers : Nat := 24

private structure Cta where
  warps : Array Nat
  /-- Per barrier (`ctaBarriers`): warps arrived in the current instance, and the warps expected (0:
  every live warp of the CTA). -/
  arrived : Array Nat := Array.replicate ctaBarriers 0
  expected : Array Nat := Array.replicate ctaBarriers 0
  /-- Per named barrier: the cycle its latest arrival was counted. -/
  lastArrival : Array Nat := Array.replicate ctaBarriers 0
  /-- Per named barrier, for `bar.red`: the reduction, its threads' contributions so far, and the
  waiting warps' writebacks with their instruction. -/
  redOp : Array (Option Collective) := Array.replicate ctaBarriers none
  redContrib : Array (Array (Option (Array UInt32))) := Array.replicate ctaBarriers #[]
  redWaiters : Array (Array (Nat × (Resp → Warp → Warp) × Nat × Nat × Nat)) := Array.replicate ctaBarriers #[]
  /-- Per named barrier: the instances completed. -/
  gen : Array Nat := Array.replicate ctaBarriers 0
  live : Nat
  /-- Its linear index in the grid and the cycle it was placed at (`Launch.spans`). -/
  id : Nat := 0
  placedAt : Nat := 0
  deriving Inhabited

/-- An address in hexadecimal. -/
private def hexAddr (a : Nat) : String := "0x" ++ String.ofList (Nat.toDigits 16 a)

/-- A cluster's barrier (`barrier.cluster`, PTX ISA §9.7.13.2), kept by the master: the warps that
arrived in the current phase, the cluster's live warps, the phase, and the cycle of the latest
arrival or exit counted. -/
private structure ClusterBar where
  arrived : Nat := 0
  live : Nat := 0
  phase : Nat := 0
  last : Nat := 0
  deriving Inhabited

/-- One level of the SM's instruction caches: the lines it holds with their last use, and the
port it is refilled through. -/
structure CodeLevel where
  lines : Std.HashMap Nat Nat := {}
  capacity : Nat
  /-- Bytes per cycle it is refilled at from the level below. -/
  rate : Nat
  fill : Port := {}
  deriving Inhabited

/-- Record a use of `line` at `t`, inserting it and evicting the least recently used line if the
level is full, the lowest such line on a tie (whatever order the map holds its lines in, so a
machine restored from its lines evicts as the original). -/
def CodeLevel.touch (lv : CodeLevel) (line t : Nat) : CodeLevel := Id.run do
  let mut lines := lv.lines
  if !lines.contains line && lines.size ≥ lv.capacity then
    let victim := lines.fold (fun (best : Option (Nat × Nat)) l u => match best with
      | some (bl, bu) => if u < bu || (u == bu && l < bl) then some (l, u) else best
      | none => some (l, u)) none
    if let some (l, _) := victim then lines := lines.erase l
  return { lv with lines := lines.insert line t }

/-- Bring `line`, at level `j` from `t` (`levels.size`: arrived from the L2), into the nearest
level: it is refilled level by level down to the nearest through each level's port. The cycle it
is in the nearest level, and the levels after. -/
private def refill (levels : Array CodeLevel) (lineBytes line j t : Nat) :
    Array CodeLevel × Nat := Id.run do
  let mut lv := levels
  let mut at_ := t
  for k in [0:j] do
    let i := j - 1 - k
    let cost := (lineBytes + lv[i]!.rate - 1) / max lv[i]!.rate 1
    let (start, port) := lv[i]!.fill.reserve at_ cost
    at_ := start + cost
    lv := lv.modify i fun l => { l with fill := port }
  for i in [0:min (j + 1) lv.size] do
    lv := lv.modify i (·.touch line t)
  return (lv, at_)

private structure Sm where
  slots : Array Slot := #[]
  ctas : Array Cta := #[]
  /-- Slots of live warps, per sub-partition. -/
  bySmsp : Array (Array Nat)
  /-- Per sub-partition and pipe: the pipe's input port, and the cycles it has been taken. -/
  pipes : Array Port
  pipeWork : Array Nat := #[]
  /-- Per sub-partition: warp instructions issued. -/
  issued : Array Nat := #[]
  /-- Per sub-partition: the register file's read stage, and the operand reuse cache (per source
  position: the warp and register it holds). -/
  rfRead : Array Port
  reuse : Array (Array (Option (Nat × Nat)))
  /-- Per sub-partition: the fraction of a cycle its stretched work has carried
  (`TensorParams.wgmmaPipeSharePermille`), in `1000 - share`ths. -/
  stealCarry : Array Nat := #[]
  last : Array (Option Nat)
  resident : Nat := 0
  /-- CTAs this SM has placed this launch (`FrontEndParams.rotateCtaWarps`). -/
  placed : Nat := 0
  /-- The instruction caches, nearest first, and the cycle each line being filled arrives. -/
  code : Array CodeLevel
  codeReady : Std.HashMap Nat Nat := {}
  /-- Code lines on their way from the L2, by the memory system's request id. -/
  codeReqs : Std.HashMap Nat Nat := {}
  /-- The constant cache: the lines it holds (bank and line) with their last use, and the cycle
  each line being filled arrives. -/
  constLines : Std.HashMap (Nat × Nat) Nat := {}
  constReady : Std.HashMap (Nat × Nat) Nat := {}
  /-- Per sub-partition: the cycle its scheduler may issue again after holding on a constant
  miss. -/
  smspHold : Array Nat := #[]
  /-- The SM counts one barrier arrival per `barrierArrivalCycles`. -/
  barPort : Port := {}
  /-- The earliest cycle at which this SM may issue. -/
  next : Nat := 0
  deriving Inhabited

/-- A memory request in flight: its response goes to warp `slot` of SM `sm`. -/
private structure MemWb where
  sm : Nat
  slot : Nat
  wb : Resp → Warp → Warp
  bar : Option (Fin 6)
  pc : Nat
  /-- The request, and its issue's index and writeback id in the warp's trace. -/
  req : Req := default
  tix : Nat := 0
  land : Nat := 0
  /-- The warp's number for the load (`Ev.mem`), 0 for none. -/
  load : Nat := 0
  deriving Inhabited

/-- The vector registers among `ls`: what a read counter covers (uniform registers and predicates
are read at issue). -/
private def vectorRegs (ls : Array NV.SASS.Loc) : Array NV.SASS.Loc :=
  ls.filter fun | .r _ => true | _ => false

/-- Whether memory event `ev` reads a vector register an earlier load of the warp has yet to
write. The warp's memory instructions pass the MIO queue in order: a request reads its data
after an earlier load's writeback (inferred). -/
private def readsBehindLoad (s : Slot) : Ev → Bool
  | .mem _ _ _ _ _ _ _ _ _ after => after.any s.inflight.contains
  | _ => false

/-- Memory event `ev` of warp `slot` of SM `si` reaches the memory system at `t`: a request is
built from the warp and issued, a fence is entered. -/
private def toMemory (fps : Array NV.SASS.Footprint) (t si slot : Nat) (s : Slot) (ev : Ev)
    (mem : MemSys) (memWbs : Std.HashMap Nat MemWb) (c : Counters) :
    Slot × MemSys × Std.HashMap Nat MemWb × Counters :=
  match ev with
  | .mem req wb bar issued pc readsPending tix land load _ =>
    let r := req s.w
    let s := if readsPending then { s with pr := s.pr.remove (vectorRegs fps[pc]!.reads) } else s
    let (mem, id) := mem.issue t { sm := si, smsp := s.smsp, cta := s.cta, slot } r issued
    let memWbs := memWbs.insert id { sm := si, slot, wb, bar, pc, req := r, tix, land, load }
    let c := match r.space, r.op.returns with
      | .global, true => { c with globalLoads := c.globalLoads + 1 }
      | .global, false => { c with globalStores := c.globalStores + 1 }
      | .shared, true => { c with sharedLoads := c.sharedLoads + 1 }
      | .shared, false => { c with sharedStores := c.sharedStores + 1 }
      | _, _ => c
    (s, mem, memWbs, c)
  | .fence cost =>
    let (mem, _) := mem.fence t { sm := si, smsp := s.smsp, cta := s.cta, slot } cost
    (s, mem, memWbs, c)
  | _ => (s, mem, memWbs, c)

/-- Asynchronous event `op` of warp `slot` of SM `si` reaches the async units at `t`: its operation
is built from the warp and started. -/
private def toAsync (p : Params) (fps : Array NV.SASS.Footprint) (t si slot : Nat) (s : Slot)
    (op : Warp → NV.Async.Op) (wb : Resp → Warp → Warp) (pc : Nat) (trace : Option (Nat × Nat))
    (u : AsyncUnit.State) (mem : MemSys) :
    Except String (Slot × AsyncUnit.State × MemSys × Array AsyncUnit.Out) := do
  let x := op s.w
  let s := { s with pr := s.pr.remove (vectorRegs fps[pc]!.reads),
                    pw := if x.writesRegisters then (s.pw.add fps[pc]!.writes).setMma fps[pc]!.writes
                          else s.pw }
  match AsyncUnit.start p u mem t { sm := si, smsp := s.smsp, cta := s.cta, slot } x wb pc (trace := trace) with
  | .error e => throw s!"cycle {t}: {e}"
  | .ok (u, mem, o) => return (s, u, mem, o)

/-- What a writeback left in its write set, for the trace: each location's 32 lanes (the `printf`
records, which the engine hands to the run's output, are not compared). -/
private def landed (writes : Array NV.SASS.Loc) (w : Warp) : Array (Option UInt64) :=
  writes.map fun l => if l == .log then none else some (digest (w.get l))

private structure Wb where
  sm : Nat
  slot : Nat
  ev : Ev
  deriving Inhabited

private def count (sb : Array Nat) (t : Nat) : Nat := (sb.filter (· > t)).size

private def cdiv (a b : Nat) : Nat := (a + b - 1) / b

/-- What a CTA of cluster rank `rank` holds at launch in its reserved shared memory: on a part with
tensor memory, in a cluster, the pair allocator's state (`NV.Async.Tcgen05.reservedInit`). -/
def reservedInit (p : Params) (clusterCtas rank : Nat) : Array (Nat × UInt32) :=
  if p.features.value.tcgen05 && clusterCtas > 1 then NV.Async.Tcgen05.reservedInit rank else #[]

/-- Where the launch's values sit in constant bank 0, per target:
`%ntid`, `%nctaid`, the local stack's top, `%dynamic_smem_size`, the reserved shared memory
(`%total_smem` is `SR_SMEMSZ` less it) and the cluster's geometry. `sm_100a`, `sm_103a` and
`sm_120a` share one layout; there `%dynamic_smem_size`, `%cluster_nctaid`,
`%nclusterid` and `%cluster_nctarank` are tables indexed by `SR_CgaSize`, whose value the model
takes as 0 (inferred), so the launch's values sit at the tables' first entries. The table at
0x2b0 holds `%cluster_nctaid`'s reciprocals in binary32, as sm_90a's layout does (inferred). -/
structure ConstLayout where
  ntid : Nat
  nctaid : Nat
  stack : Nat
  dynSmem : Nat
  reservedSmem : Nat
  /-- `sm_90a`: is-explicit-cluster, `%cluster_nctaid`, its reciprocals, `%nclusterid`, and
  `%cluster_nctarank` at +0x48. -/
  cluster : Option Nat := none
  /-- `sm_100a`, `sm_120a`: the tables of `%cluster_nctaid` (x, y, z, and its reciprocals 0x10 on),
  `%nclusterid` (x, y, z, then `%cluster_nctarank`) and `%is_explicit_cluster`'s word. -/
  clusterTables : Option (Nat × Nat × Nat) := none
  /-- 64-bit words holding the parameter block's address in device memory (`cvta.param`: the
  word at 0x198 on sm_90a, 0x348 on sm_100a and sm_120a), and words holding constant bank 0's address (the parameter block's less `paramBase`:
  sm_80's `cvta.param` adds the offset in the bank to the word at 0x40). -/
  paramAddr : List Nat := []
  bankAddr : List Nat := []
  /-- A 64-bit word holding the shared window's base (`cvta.shared` on sm_80, at 0x18; later
  targets read `SR_SWINHI`). -/
  sharedWindow : Option Nat := none

def ConstLayout.ofSm : Nat → ConstLayout
  | 100 | 103 | 120 =>
    { ntid := 0x360, nctaid := 0x370, stack := 0x37c, dynSmem := 0x2ac, reservedSmem := 0x16c,
      clusterTables := some (0x2a0, 0x2c0, 0x36c), paramAddr := [0x348] }
  | 80 => { ntid := 0x0, nctaid := 0xc, stack := 0x28, dynSmem := 0x2c, reservedSmem := 0x114,
            bankAddr := [0x40], sharedWindow := some 0x18 }
  | _ => { ntid := 0x0, nctaid := 0xc, stack := 0x28, dynSmem := 0x2c, reservedSmem := 0x114,
           cluster := some 0x140, paramAddr := [0x198] }

/-- The parameter block's address in device memory: just past the launch's global memory
(`Env.paramAddr`). -/
def paramAddrOf (l : Launch) : UInt64 :=
  ((l.global.base.toNat + l.global.size + 4095) / 4096 * 4096).toUInt64

/-- Constant bank 0 at launch: the launch's values where the kernel's target reads them
(`ConstLayout`), the parameter block's and the bank's addresses in device memory, and the
parameters at `paramBase`. -/
def constBank0 (l : Launch) : Store := Id.run do
  let k := l.kernel
  let c := ConstLayout.ofSm k.sm
  let mut s := Store.empty 0 (max 0x400 (k.paramBase + max k.paramBytes l.params.size + 16))
  let (bx, by_, bz) := l.block
  let (gx, gy, gz) := l.grid
  let (cx, cy, cz) := l.clusterShape
  let explicit := l.cluster != (1, 1, 1) || k.cluster.isSome
  let mut words : List (Nat × Nat) :=
    [(c.ntid, bx), (c.ntid + 4, by_), (c.ntid + 8, bz), (c.nctaid, gx), (c.nctaid + 4, gy),
     (c.nctaid + 8, gz), (c.stack, k.localBytes), (c.dynSmem, l.dynShared),
     (c.reservedSmem, k.sharedReserved)]
  if let some b := c.cluster then
    words := words ++ [(b, if explicit then 1 else 0), (b + 0x4, cx), (b + 0x8, cy), (b + 0xc, cz),
      (b + 0x1c, gx / cx), (b + 0x20, gy / cy), (b + 0x24, gz / cz), (b + 0x48, cx * cy * cz)]
    for (off, n) in [(b + 0x10, cx), (b + 0x14, cy), (b + 0x18, cz)] do
      s := s.writeWord off (Float32.ofNat 1 / Float32.ofNat n).toBits
  if let some (nctaid, ncluster, expl) := c.clusterTables then
    words := words ++ [(nctaid, cx), (nctaid + 4, cy), (nctaid + 8, cz), (ncluster, gx / cx),
      (ncluster + 4, gy / cy), (ncluster + 8, gz / cz), (ncluster + 0xc, cx * cy * cz),
      (expl, if explicit then 1 else 0)]
    for (off, n) in [(nctaid + 0x10, cx), (nctaid + 0x14, cy), (nctaid + 0x18, cz)] do
      s := s.writeWord off (Float32.ofNat 1 / Float32.ofNat n).toBits
  -- The local window's base, where `cvta.local` reads it (a generic address of a thread's
  -- local memory: the stack pointer over it).
  let lw := ({} : NV.Windows).local.toNat
  let lo := NV.SASS.Exec.localWindowAt k.sm
  words := words ++ [(lo, lw % 2 ^ 32), (lo + 4, lw / 2 ^ 32)]
  if let some sw := c.sharedWindow then
    let b := ({} : NV.Windows).shared.toNat
    words := words ++ [(sw, b % 2 ^ 32), (sw + 4, b / 2 ^ 32)]
  let pa := (paramAddrOf l).toNat
  for off in c.paramAddr do words := words ++ [(off, pa % 2 ^ 32), (off + 4, pa / 2 ^ 32)]
  for off in c.bankAddr do
    words := words ++ [(off, (pa - k.paramBase) % 2 ^ 32), (off + 4, (pa - k.paramBase) / 2 ^ 32)]
  for (off, v) in words do
    s := s.writeWord off v.toUInt32
  for i in [0:l.params.size] do
    s := s.writeByte (k.paramBase + i) l.params[i]!
  return s

/-- The lanes of `m` whose guard holds (all of `m` without one). -/
private def guarded (i : Instr) (w : Warp) (m : UInt32) : UInt32 :=
  match i.guard with
  | none => m
  | some (g, neg) =>
    if g.file == .up then (if w.getUP g.idx != neg then m else 0)
    else (List.range 32).foldl (fun acc l =>
      if (m >>> l.toUInt32) &&& 1 == 1 && w.getP g.idx l != neg then acc ||| ((1 : UInt32) <<< l.toUInt32) else acc) 0

/-- A CTA's index in its grid, x fastest. -/
private def ctaLinear (c : Ctx) : Nat :=
  let (x, y, z) := c.ctaid
  let (gx, gy, _) := c.nctaid
  x + gx * (y + gy * z)

/-- A CTA's cluster's index in its grid, x fastest. -/
private def clusterOfDims (c : Ctx) (cl : Nat × Nat × Nat) : Nat :=
  let (x, y, z) := c.ctaid
  let (gx, gy, _) := c.nctaid
  let (cx, cy, cz) := cl
  x / cx + (gx / cx) * (y / cy + (gy / cy) * (z / cz))

/-- Cycles beyond which an SM with nothing in flight is deadlocked. -/
private def never : Nat := 1000000000

/-- The release cycle a dependence counter holds for a memory request not yet delivered. -/
private def pendingAt : Nat := 1000000000000000

/-- Whether sub-partition `q`'s tensor core is running a warpgroup MMA at `t`: the
sub-partition's other work slows (`TensorParams.wgmmaIssueSharePermille`, `wgmmaPipeSharePermille`). -/
private def tensorBusy (p : Params) (au : AsyncUnit.State) (si q t : Nat) : Bool :=
  (au.gmma[si * p.smsps.value + q]?.any (·.free > t))

/-- Cycle `t` is one sub-partition `q` loses to its tensor core: `wgmmaIssueSharePermille` of the
cycles it is busy, spread evenly. -/
private def lostCycle (p : Params) (au : AsyncUnit.State) (si q t : Nat) : Bool :=
  let share := p.tensor.wgmmaIssueSharePermille.value
  share > 0 && (t + 1) * share / 1000 > t * share / 1000 && tensorBusy p au si q t

/-- Whether slot `s` can issue at `t`, and if not, the earliest cycle it might and the reason
holding it longest. -/
private def readiness (k : Kernel) (timings : Array (Option OpTiming))
    (fps : Array NV.SASS.Footprint) (mem : MemSys)
    (au : AsyncUnit.State) (si : Nat) (sm : Sm) (q : Nat) (i : Nat) (s : Slot) (t : Nat)
    (functional : Bool) (p : Params) : Bool × Nat × Stall := Id.run do
  if s.readyAt > t then return (false, s.readyAt, s.why)
  let some (pc, _) := s.w.current | return (false, 0, s.why)
  let some ins := k.code[pc]? | return (true, t, s.why)
  -- Functionally everything an instruction waits on has completed.
  if functional then return (true, t, s.why)
  let mut wait := t
  let mut why := Stall.wait
  -- After a stall count of 0 on an interlocking part, the fixed-latency results this
  -- instruction reads (variable-latency ones are the dependence counters' business).
  if s.interlock then
    for l in fps[pc]!.reads do
      let at_ := s.pw.landing l pendingAt
      if at_ < pendingAt && at_ > wait then wait := at_; why := .shortScoreboard
  let counterWhy (b : Nat) : Stall :=
    if (s.sbMem >>> b) % 2 == 1 then .longScoreboard else .shortScoreboard
  for b in [0:6] do
    if (ins.ctrl.wait >>> b) % 2 == 1 then
      for x in s.sb[b]! do
        if x > wait then wait := x; why := counterWhy b
  -- `DEPBAR.LE SBk, n`: wait until counter k holds at most n events.
  if ins.opcode == "DEPBAR" then
    match ins.operands[0]?, ins.operands[1]? with
    | some (Operand.name sbk), some (Operand.imm n) =>
      let b := (sbk.drop 2).toNat?.getD 0
      let pend := (s.sb[b]!.filter (· > t)).qsort (· < ·)
      if pend.size > n.toNat && pend[pend.size - 1 - n.toNat]! > wait then
        wait := pend[pend.size - 1 - n.toNat]!; why := counterWhy b
    | _, _ => pure ()
  -- `WARPGROUP.DEPBAR.LE gsb0, n`: wait until at most n of the warp's wgmma groups are pending.
  if ins.opcode == "WARPGROUP" && ins.hasMod "DEPBAR" then
    if let some (Operand.imm n) := ins.operands[1]? then
      -- A group whose completion is not known yet holds the warp until the async unit wakes it.
      let pend := (AsyncUnit.wgmmaPending au si i t).map (·.getD pendingAt) |>.qsort (· < ·)
      if pend.size > n.toNat && pend[pend.size - 1 - n.toNat]! > wait then
        wait := pend[pend.size - 1 - n.toNat]!; why := .shortScoreboard
  -- A barrier orders the warp's earlier memory accesses before the accesses other warps make after
  -- it (PTX ISA §9.7.13.1), so the warp arrives once its requests have reached the memory system
  -- and performed.
  if ins.opcode == "BAR" then
    let drained := if s.lastMem > t then s.lastMem
      else if !s.held.isEmpty || !mem.performed si i then max (t + 1) ((mem.nextSm si).getD (t + 1)) else t
    if drained > wait then wait := drained; why := .barrier
  if let some tm := timings[pc]?.join then
    let free := sm.pipes[q * Pipe.count + tm.pipe.index]!.free
    if free > wait then wait := free; why := .pipe
    -- A full MIO queue frees when the memory system next dispatches.
    if tm.pipe == .lsu && !mem.accepts si q then
      let next := max (t + 1) ((mem.nextSm si).getD (t + 1))
      if next > wait then wait := next; why := .mio
  if sm.rfRead[q]!.free > wait then wait := sm.rfRead[q]!.free; why := .regRead
  -- A cycle the sub-partition's tensor core takes: a warp ready in it waits for the next.
  if wait ≤ t && lostCycle p au si q t then return (false, t + 1, .dispatch)
  return (wait ≤ t, wait, why)

/-- The regular-register sources of an instruction by operand position (a, b, c): the register
operands after the destination. -/
def regSources (i : Instr) : Array Nat :=
  let regs := i.operands.filterMap fun
    | .reg ⟨.r, n⟩ _ => some n
    | _ => none
  let srcs := match i.operands[0]? with
    | some (.reg ⟨.r, _⟩ _) => regs.extract 1 regs.size
    | _ => regs
  srcs.extract 0 3

/-- Cycles a fixed-latency instruction holds the register file's read stage: the most reads any
bank must serve, over its read ports (Huerta et al. 2025 §5.3: a conflict costs a bubble). Sources
found in the reuse cache are not read, and a register named twice is read once. -/
def readCycles (p : Params) (srcs : Array Nat) (hit : Nat → Bool) : Nat := Id.run do
  let banks := max p.regFile.banks.value 1
  let mut per := Array.replicate banks 0
  let mut seen : Array Nat := #[]
  for i in [0:srcs.size] do
    let r := srcs[i]!
    if r != 255 && !hit i && !seen.contains r then
      seen := seen.push r
      per := per.modify (r % banks) (· + 1)
  let ports := max p.regFile.readPortsPerBank.value 1
  return max 1 (per.foldl (fun m c => max m ((c + ports - 1) / ports)) 0)

/-- The candidate the policy takes when the greedy warp is not eligible. -/
private def byAge (p : Params) (cands : Array Nat) (sm : Sm) : Nat :=
  let older (a b : Nat) : Bool := sm.slots[a]!.age < sm.slots[b]!.age
  -- Issued longer ago, or as long ago and younger.
  let staler (a b : Nat) : Bool :=
    let (x, y) := (sm.slots[a]!, sm.slots[b]!)
    x.issuedAt < y.issuedAt || (x.issuedAt == y.issuedAt && y.age < x.age)
  cands.foldl (fun a i => match p.issuePolicy.value with
    | .greedyThenOldest => if older i a then i else a
    | .greedyThenYoungest => if older a i then i else a
    | .leastRecent | .greedyThenLeastRecent => if staler i a then i else a) cands[0]!

/-- Opcodes that transfer control. -/
private def transfers : List String := ["BRA", "BRX", "JMP", "JMX", "CALL", "RET"]



/-- What every SM of a launch reads and none changes. -/
private structure Kit where
  p : Params
  k : Kernel
  l : Launch
  ops : Array NV.SASS.Op
  fps : Array NV.SASS.Footprint
  timings : Array (Option OpTiming)
  sources : Array (Array Nat)
  /-- Per instruction, the constant-cache lines its immediate-offset constant operands read. -/
  constRefs : Array (Array (Nat × Nat))
  constCapacity : Nat
  lineBytes : Nat
  prefetch : Nat
  codeLines : Nat
  smsps : Nat
  nSms : Nat
  /-- For placing CTAs: constant bank 0, threads and warps per CTA, the cluster shape. -/
  const0 : Store
  threads : Nat
  nWarps : Nat
  cluster : Nat × Nat × Nat
  /-- The parameter block's address in device memory (`Env.paramAddr`). -/
  paramAddr : UInt64 := 0

/-- The launch's changing state. -/
private structure St where
  sms : Array Sm
  mem : MemSys
  memWbs : Std.HashMap Nat MemWb
  wbs : Heap Wb
  printfOut : Array PrintfRecord
  asyncU : AsyncUnit.State
  outs : Array AsyncUnit.Out
  doneCtas : Nat
  c : Counters
  t : Nat
  hazards : Array String
  /-- Each cluster's barrier phase as the quantum began (the master completes phases between
  quanta), and the arrivals (`true`) and exits (`false`) of this quantum: cluster, cycle. -/
  phases : Array Nat
  clusterLog : Array (Nat × Nat × Bool) := #[]
  /-- With a focus SM: the CTAs reserved for it this quantum, each started as soon as one of its
  CTAs completes (the dispatcher fills a freed slot; `ctaDispatchCycles`' replacement cost). -/
  reserved : Array Nat := #[]
  /-- With a focus SM: the CTAs it placed this quantum (SM, CTA slot, cluster, rank, shared
  bytes), which the quantum's end registers with the memory system's L2 side. -/
  placed : Array (Nat × Nat × Nat × Nat × Nat) := #[]
  /-- Warp traces (`Launch.recordTrace`), by CTA and warp (`ctaLinear * warps + warp`), and the
  writebacks whose landing a trace has, by warp key and id: an MMA's lands early, before an MMA
  chained onto it, and not again when the engine applies it. -/
  traces : Std.HashMap Nat WarpTrace := {}
  landedIds : Std.HashSet (Nat × Nat) := {}
  /-- The functional mode over several workers: the global atomics its warps issued this quantum
  (warp slot, CTA slot, request), for the master. -/
  atoms : Array (Nat × Nat × Req) := #[]
  deriving Inhabited

/-- Place CTA `id` on SM `si`, its warps ready at `start`: the SM and the memory system after. A
warp's age (the scheduler's oldest/youngest order) is its CTA's id and warp index. -/
private def placeCta (kit : Kit) (sm : Sm) (mem : MemSys) (si id start : Nat)
    (traces : Std.HashMap Nat WarpTrace) : Sm × MemSys × Std.HashMap Nat WarpTrace := Id.run do
  let ⟨_, k, l, _, _, _, _, _, _, _, _, _, smsps, _, const0, threads, nWarps, (cx, cy, cz), _⟩ := kit
  let (gx, gy, _) := l.grid
  let ctaid := (id % gx, (id / gx) % gy, id / (gx * gy))
  let (x, y, z) := ctaid
  let rank := x % cx + cx * (y % cy + cy * (z % cz))
  let cluster := x / cx + (gx / cx) * (y / cy + (gy / cy) * (z / cz))
  let mut sm := sm
  let mut mem := mem
  let mut traces := traces
  let ci := sm.ctas.size
  let mut ws := #[]
  for wi in [0:nWarps] do
    let lanes := min 32 (threads - wi * 32)
    -- The warp's slot and sub-partition: the SM's k-th CTA takes the slots after the k before it,
    -- rotated within each row of sub-partitions by k when its warps fill whole rows
    -- (`FrontEndParams.rotateCtaWarps`).
    let base := sm.placed * nWarps
    let slot := if kit.p.frontEnd.rotateCtaWarps.value && nWarps % smsps == 0 then
        base + wi / smsps * smsps + (wi + sm.placed) % smsps
      else if kit.p.frontEnd.rotateCtaWarps.value then base + wi
      else wi
    let q := slot % smsps
    let env : Env := { ctx := { ntid := l.block, nctaid := l.grid, ctaid, warp := wi,
                                threads, smid := si, clusterRank := rank,
                                clusterSize := l.clusterShape.1 * l.clusterShape.2.1 * l.clusterShape.2.2,
                                warpSlot := slot % kit.p.maxWarpsPerSM.value,
                                smemSize := k.sharedReserved + k.sharedBytes + l.dynShared }, const0,
                       textures := l.textures, paramAddr := kit.paramAddr }
    ws := ws.push sm.slots.size
    sm := { sm with bySmsp := sm.bySmsp.modify q (·.push sm.slots.size) }
    mem := mem.addWarp si ci sm.slots.size k.localBytes
    let w0 := Warp.init (max k.regs k.regFile) lanes
    if l.recordTrace then
      traces := traces.insert (id * nWarps + wi) { cta := id, warp := wi, env, init := w0 }
    sm := { sm with slots := sm.slots.push { w := w0, env, cta := ci,
                                             smsp := q, age := id * nWarps + wi, readyAt := start } }
  mem := mem.addCta si ci cluster rank (k.sharedReserved + k.sharedBytes + l.dynShared)
    (reservedInit kit.p (cx * cy * cz) rank) k.sharedReserved
  sm := { sm with ctas := sm.ctas.push { warps := ws, live := nWarps, id, placedAt := start },
                  resident := sm.resident + 1, placed := sm.placed + 1, next := min sm.next start }
  return (sm, mem, traces)

/-- One step of the launch: to the next cycle at which anything can happen, and everything that
happens then. With `focus`, the step is SM `focus`'s alone (phase A of a quantum: its warps, its
side of the memory system, its async unit), and nothing at or after `stop` happens; whether it
stepped. Without, every SM steps and the memory system runs its quanta itself. -/
private def pass (kit : Kit) (focus : Nat) (stop : Nat) (st : St) :
    Except String (St × Bool) := Id.run do
  let ⟨p, k, l, ops, fps, timings, sources, constRefs, constCapacity, lineBytes, prefetch,
    codeLines, smsps, _, _, _, nWarps, _, _⟩ := kit
  let banks := max 1 p.regFile.banks.value
  let slots := max 1 (p.regFile.reuseEntriesPerBank.value * p.regFile.reuseSlotsPerEntry.value)
  let clusterOf (c : Ctx) : Nat := clusterOfDims c l.clusterShape
  let mut sms := st.sms
  let mut mem := st.mem
  let mut memWbs := st.memWbs
  let mut wbs := st.wbs
  let mut printfOut := st.printfOut
  let mut asyncU := st.asyncU
  let mut outs := st.outs
  let mut doneCtas := st.doneCtas
  let mut c := st.c
  let mut t := st.t
  let mut hazards := st.hazards
  let phases := st.phases
  let mut clusterLog := st.clusterLog
  let mut reserved := st.reserved
  let mut placed := st.placed
  let mut traces := st.traces
  let mut landedIds := st.landedIds
  let mut atoms := st.atoms
  let tkey (c : Ctx) : Nat := ctaLinear c * nWarps + c.warp
  -- The next cycle at which anything can happen.
  let mut tn : Option Nat := wbs.minKey?
  if let some x := mem.nextSm focus then tn := some (match tn with | some y => min x y | none => x)
  if !outs.isEmpty then tn := some t
  if sms[focus]!.resident > 0 then
    let n := sms[focus]!.next
    tn := some (match tn with | some x => min x n | none => n)
  let unchanged : St :=
    { st with sms := sms, mem := mem, memWbs := memWbs, wbs := wbs, printfOut := printfOut, asyncU := asyncU, outs := outs, doneCtas := doneCtas, c := c, t := t, hazards := hazards, clusterLog := clusterLog, reserved := reserved, placed := placed, traces := traces, landedIds := landedIds, atoms := atoms }
  let some tNext := tn | return .ok (unchanged, false)
  if tNext ≥ stop then return .ok (unchanged, false)
  if tNext ≥ t + never then
    return .error s!"deadlock at cycle {t}: no warp can issue and nothing is in flight"
  t := max t tNext
  -- Writebacks due.
  for _ in [0:wbs.size] do
    match wbs.minKey? with
    | some at_ => if at_ > t then break
    | none => break
    let h0 := wbs
    wbs := {}
    let some (_, d, h) := h0.pop? | break
    wbs := h
    let mut sm := sms[d.sm]!
    sms := sms.set! d.sm default
    let s := sm.slots[d.slot]!
    sm := { sm with slots := sm.slots.set! d.slot default }
    match d.ev with
    | .late env m pc at_ land =>
      match (ops[pc]!).run env s.w m with
      | .reg _ wb =>
        wbs := wbs.push at_ { sm := d.sm, slot := d.slot, ev := .wb wb (some pc) (some land) }
        sm := { sm with slots := sm.slots.set! d.slot s }
      | _ => return .error s!"cycle {t}: a late-reading instruction at {pc} changed its effect"
    | .wb f pc land =>
      let w := s.w
      let s := { s with w := default }
      let pw := match pc with | some pc => s.pw.remove fps[pc]!.writes | none => s.pw
      let w := f w
      if l.recordTrace then
        if let (some pc, some id) := (pc, land) then
          if !landedIds.contains (tkey s.env.ctx, id) then
            let ev : TraceEv := .land id (landed fps[pc]!.writes w)
            traces := traces.modify (tkey s.env.ctx) fun wt => { wt with evs := wt.evs.push ev }
            landedIds := landedIds.insert (tkey s.env.ctx, id)
      sm := { sm with slots := sm.slots.set! d.slot { s with w, pw }.settle }
    | .mem .. | .fence .. =>
      -- Behind a held request, or reading what an earlier load has yet to write: held.
      if !s.held.isEmpty || readsBehindLoad s d.ev then
        sm := { sm with slots := sm.slots.set! d.slot { s with held := s.held.push d.ev } }
      else
        let (s, m', mw', c') := toMemory fps t d.sm d.slot s d.ev mem memWbs c
        mem := m'; memWbs := mw'; c := c'
        sm := { sm with slots := sm.slots.set! d.slot s.settle }
    | .commit g bar onRead =>
      let (u, released, done) := AsyncUnit.commit asyncU t d.sm d.slot g bar onRead
      asyncU := u
      mem := mem.raceAgentsDone d.sm d.slot done
      if let some b := bar then
        if let some r := released then outs := outs.push (.release d.sm d.slot b (max t r))
      sm := { sm with slots := sm.slots.set! d.slot s }
    | .async op wb pc tix land =>
      -- Behind held requests it waits its turn in MIO order too.
      if !s.held.isEmpty then
        sm := { sm with slots := sm.slots.set! d.slot { s with held := s.held.push d.ev } }
      else
        match toAsync p fps t d.sm d.slot s op wb pc (if l.recordTrace then some (tix, land) else none) asyncU mem with
        | .error e => return .error e
        | .ok (s, u, m', o') =>
          asyncU := u; mem := m'; outs := outs ++ o'
          sm := { sm with slots := sm.slots.set! d.slot s }
    sms := sms.set! d.sm sm
  -- Memory deliveries due: responses written, write counters released, fences passed.
  let m0 := mem
  mem := default
  match m0.advanceSm focus t with
  | .error e => return .error s!"cycle {t}: {e}"
  | .ok (m', ds) =>
    mem := m'
    -- The async units' deliveries first; the rest are the warps'.
    let ds ← match AsyncUnit.deliver p asyncU mem ds with
      | .error e => return .error s!"cycle {t}: async unit: {e}"
      | .ok (u, m', o, rest) => asyncU := u; mem := m'; outs := outs ++ o; pure rest
    for d in ds do
      let o := d.origin
      -- A code line from the L2: into the SM's caches, and the warps waiting to fetch look again.
      if let some line := sms[o.sm]!.codeReqs.get? d.id then
        let sm := sms[o.sm]!
        sms := sms.set! o.sm default
        let (code, r) := refill sm.code lineBytes line sm.code.size d.at_
        let slots := sm.slots.map fun w =>
          if w.why == .fetch && w.readyAt ≥ pendingAt then { w with readyAt := r } else w
        sms := sms.set! o.sm { sm with code, slots, codeReady := sm.codeReady.insert line r,
                                       codeReqs := sm.codeReqs.erase d.id, next := min sm.next r }
        continue
      let mut sm := sms[o.sm]!
      sms := sms.set! o.sm default
      let mut s := sm.slots[o.slot]!
      sm := { sm with slots := sm.slots.set! o.slot default }
      if d.fence then s := { s with atFence := false, readyAt := max s.readyAt d.at_ }
      else if let some mw := memWbs.get? d.id then
        memWbs := memWbs.erase d.id
        if s.atTryWait && (k.code[mw.pc]?.any fun i => i.opcode == "SYNCS" && i.hasMod "TRYWAIT") then
          s := { s with atTryWait := false, readyAt := max s.readyAt d.at_ }
        let w := s.w
        s := { s with w := default }
        s := { s with w := mw.wb d.resp w, pw := s.pw.remove fps[mw.pc]!.writes,
                      inflight := s.inflight.erase mw.load, sharedLoads := s.sharedLoads.erase mw.load }
        -- Shuffles held behind shared loads land, in order, once those have: no earlier than
        -- this delivery, nor than their own latency; their counters release then.
        while !s.mioHeld.isEmpty && !(s.mioHeld[0]!.1.any s.inflight.contains) do
          let (_, due, ev, bar) := s.mioHeld[0]!
          s := { s with mioHeld := s.mioHeld.extract 1 s.mioHeld.size }
          let at_ := max due (d.at_ + 1)
          wbs := wbs.push at_ { sm := o.sm, slot := o.slot, ev }
          if let some b := bar then
            if let some i := s.sb[b.val]!.findIdx? (· ≥ pendingAt) then
              s := { s with sb := s.sb.modify b.val (·.set! i at_) }
            if s.readyAt ≥ pendingAt then s := { s with readyAt := at_ }
        -- Requests held behind the load go on, in order, as far as the next still waiting.
        while !s.held.isEmpty && !readsBehindLoad s s.held[0]! do
          let ev := s.held[0]!
          s := { s with held := s.held.extract 1 s.held.size }
          match ev with
          | .async op wb pc tix land =>
            match toAsync p fps t o.sm o.slot s op wb pc (if l.recordTrace then some (tix, land) else none) asyncU mem with
            | .error e => return .error e
            | .ok (s', u, m', o') => s := s'; asyncU := u; mem := m'; outs := outs ++ o'
          | _ =>
            let (s', m', mw', c') := toMemory fps t o.sm o.slot s ev mem memWbs c
            s := s'; mem := m'; memWbs := mw'; c := c'
        -- The trace: the request's reads, now known, and the landing.
        if l.recordTrace then
          let issue : TraceEv := .issue mw.pc (.mem (readBytes mw.req d.resp)) (some mw.land)
          let land : TraceEv := .land mw.land (landed fps[mw.pc]!.writes s.w)
          traces := traces.modify (tkey s.env.ctx) fun wt => { wt with evs := (wt.evs.set! mw.tix issue).push land }
        -- A writeback may complete a `printf` call: its record leaves the warp for the run's output.
        if !s.w.printf.isEmpty then
          for (format, args) in s.w.printf do
            let r : PrintfRecord := { cycle := d.at_, sm := o.sm, cta := ctaLinear s.env.ctx,
                                      warp := s.env.ctx.warp, format := format, args := args }
            printfOut := printfOut.push r
          s := { s with w := { s.w with printf := #[] } }
        if let some b := mw.bar then
          -- One of the counter's pending entries becomes this delivery's cycle; a readiness
          -- cached behind a pending entry is recomputed from it.
          let mut sb := s.sb
          s := { s with sb := #[] }
          if let some i := sb[b.val]!.findIdx? (· ≥ pendingAt) then
            sb := sb.modify b.val (·.set! i d.at_)
          s := { s with sb }
          if s.readyAt ≥ pendingAt then s := { s with readyAt := d.at_ }
      sm := { sm with slots := sm.slots.set! o.slot s.settle, next := min sm.next d.at_ }
      sms := sms.set! o.sm sm
      c := { c with cycles := max c.cycles (d.at_ + 1) }
  -- What the async units' progress asks of the warps: results to write, dependence counters
  -- to release, waits to recheck.
  for o in outs do
    match o with
    | .wb si slot at_ f pc land => wbs := wbs.push at_ { sm := si, slot, ev := .wb f (some pc) land }
    | .trace si slot tix pc land reads res =>
      let issue : TraceEv := .issue pc (.async reads res) (some land)
      traces := traces.modify (tkey sms[si]!.slots[slot]!.env.ctx) fun wt => { wt with evs := wt.evs.set! tix issue }
    | .earlyLand si slot ix pc land f =>
      let key := tkey sms[si]!.slots[slot]!.env.ctx
      let ev : TraceEv := .land land (landed fps[pc]!.writes (f sms[si]!.slots[slot]!.w))
      traces := traces.modify key fun wt => { wt with evs := wt.evs.set! ix ev }
      landedIds := landedIds.insert (key, land)
    | .release si slot b at_ =>
      let mut sm := sms[si]!
      sms := sms.set! si default
      let mut s := sm.slots[slot]!
      sm := { sm with slots := sm.slots.set! slot default }
      if let some i := s.sb[b.val]!.findIdx? (· ≥ pendingAt) then
        s := { s with sb := s.sb.modify b.val (·.set! i at_) }
      if s.readyAt ≥ pendingAt then s := { s with readyAt := at_ }
      sms := sms.set! si { sm with slots := sm.slots.set! slot s, next := min sm.next at_ }
    | .wake si slot at_ =>
      let sm := sms[si]!
      let sm := if sm.slots[slot]!.readyAt ≥ pendingAt then
          { sm with slots := sm.slots.modify slot fun (w : Slot) => { w with readyAt := at_ } } else sm
      sms := sms.set! si { sm with next := min sm.next at_ }
  outs := #[]
  -- Issue on every SM that can.
  for si in #[focus] do
    if sms[si]!.resident == 0 || sms[si]!.next > t then continue
    let mut sm := sms[si]!
    sms := sms.set! si default
    let mut issued := false
    let mut wake : Nat := t + never
    for q in [0:smsps] do
      -- A scheduler that met an unserved constant miss holds before switching warps
      -- (`constMissSwitchCycles`).
      if sm.smspHold[q]! > t then
        wake := min wake sm.smspHold[q]!
        continue
      -- Eligible warps, and the earliest cycle an ineligible one may become eligible.
      let mut cands : Array Nat := #[]
      for i in sm.bySmsp[q]! do
        let s := sm.slots[i]!
        if s.atBar.isSome || s.atFence || s.atTryWait || s.atClusterWait || s.readyAt > t then
          if s.atBar.isNone && !s.atFence && !s.atTryWait && !s.atClusterWait then wake := min wake s.readyAt
          continue
        -- Instruction delivery: a warp entering a code line the nearest cache lacks waits for it.
        if let some (pc, _) := s.w.current then
          let line := (l.codeBase + (k.code[pc]?.map (·.pc)).getD 0) / lineBytes
          if s.codeLine != some line then
            -- The line, then the stream prefetcher's window after it (`codePrefetchLines`):
            -- each line neither held nearest nor in flight is requested now, from the farthest
            -- level holding it, or from the L2 like any line (contending with data there, from
            -- DRAM on a miss).
            for ahead in [0:prefetch + 1] do
              let l := line + ahead
              if l ≥ codeLines then break
              if sm.codeReady.contains l || (sm.code[0]?.map (·.lines.contains l)).getD true then
                continue
              match sm.code.findIdx? (·.lines.contains l) with
              | some j =>
                let (code, r) := refill sm.code lineBytes l j t
                sm := { sm with code, codeReady := sm.codeReady.insert l r }
              | none =>
                let (m', id) := mem.issueCode t { sm := si, smsp := q, cta := s.cta, slot := i } (l * lineBytes)
                mem := m'
                sm := { sm with codeReady := sm.codeReady.insert l pendingAt, codeReqs := sm.codeReqs.insert id l }
            let ready := (sm.codeReady[line]?).getD t
            if ready > t then
              wake := min wake ready
              sm := { sm with slots := sm.slots.modify i fun (w : Slot) =>
                { w with readyAt := ready, why := .fetch } }
              continue
            sm := { sm with code := sm.code.modify 0 (·.touch line t),
                            codeReady := sm.codeReady.erase line,
                            slots := sm.slots.modify i fun (w : Slot) => { w with codeLine := some line } }
        -- Constant operands: an instruction reading a constant line the SM's constant cache lacks
        -- waits for it (`constMissCycles`; the cache holds `constCacheBytes`, LRU).
        if let some (pc, _) := s.w.current then
          let mut late := t
          for (bank, line) in constRefs[pc]! do
            match sm.constReady[(bank, line)]? with
            | some r => late := max late r
            | none =>
              if sm.constLines.contains (bank, line) then
                sm := { sm with constLines := sm.constLines.insert (bank, line) t }
              else
                let r := t + p.frontEnd.constMissCycles.value
                sm := { sm with constReady := sm.constReady.insert (bank, line) r,
                                smspHold := sm.smspHold.set! q (t + p.frontEnd.constMissSwitchCycles.value) }
                late := max late r
          if late > t then
            wake := min wake late
            sm := { sm with slots := sm.slots.modify i fun (w : Slot) =>
              { w with readyAt := late, why := .fetch } }
            continue
          -- Lines that arrived join the cache, evicting the least recently used beyond capacity.
          for (bank, line) in constRefs[pc]! do
            if sm.constReady.contains (bank, line) then
              let mut lines := sm.constLines
              if lines.size ≥ constCapacity then
                let victim := lines.fold (fun (b : Option ((Nat × Nat) × Nat)) key u => match b with
                  | some (_, bu) => if u < bu then some (key, u) else b
                  | none => some (key, u)) none
                if let some (key, _) := victim then lines := lines.erase key
              sm := { sm with constLines := lines.insert (bank, line) t,
                              constReady := sm.constReady.erase (bank, line) }
        let s := sm.slots[i]!
        let (ok, at_, why) := readiness k timings fps mem asyncU si sm q i s t l.functional p
        if ok then cands := cands.push i
        else if at_ > t then
          wake := min wake at_
          -- No warp can become eligible before the cycle computed here (counter releases are
          -- fixed when set, ports only move later), so remember it and skip the warp until then.
          if at_ > s.readyAt then
            sm := { sm with slots := sm.slots.modify i fun (w : Slot) =>
              if why == .dispatch then { w with readyAt := at_, lost := w.lost + 1 } else { w with readyAt := at_, why } }
      if cands.isEmpty then continue
      let pick := match sm.last[q]!, p.issuePolicy.value with
        | _, .leastRecent => byAge p cands sm
        | some i, _ => if cands.contains i then i else byAge p cands sm
        | none, _ => byAge p cands sm
      for i in cands do
        if i != pick then
          sm := { sm with slots := sm.slots.modify i fun (w : Slot) => { w with why := .notSelected } }
      let mut s := sm.slots[pick]!
      sm := { sm with slots := sm.slots.set! pick default }
      -- The cycles since the warp's last issue go to the reason that held it last.
      let held := s.why
      if let some li := s.lastIssue then
        if t > li + 1 then
          let lost := min s.lost (t - li - 1)
          c := { c with stalls := (c.stalls.modify held.index (· + (t - li - 1 - lost))).modify
                                    Stall.dispatch.index (· + lost) }
      s := { s with lastIssue := some t, why := .wait, lost := 0 }
      let some (pc, m) := s.w.current | return .error "an eligible warp with no live lane"
      let some ins := k.code[pc]? | return .error s!"pc {pc} outside the code"
      let some tm := timings[pc]?.join
        | return .error s!"{ins.opcode}{ins.mods.toList} at pc {ins.pc}: no timing class"
      let env := { s.env with clock := t, timeNs := t * 1000 / p.clockMHz.value }
      s := match l.dumpAt with
        | some at_ =>
          if ins.pc == at_ && ctaLinear s.env.ctx == 0 then
            let rows := (List.range (max k.regs k.regFile)).map fun r =>
              s!"R{r}: " ++ " ".intercalate ((List.range 32).map fun ln =>
                let v := s.w.getR r ln
                String.ofList (Nat.toDigits 16 v.toNat))
            dbgTrace (s!"dump cycle {t} warp {s.env.ctx.warp} pc {ins.pc} (registers before it)\n" ++ "\n".intercalate rows) fun _ => s
          else s
        | none => s
      s := if l.trace then
          dbgTrace (s!"issue cycle {t} SM {si} CTA {ctaLinear s.env.ctx} warp {s.env.ctx.warp} pc {ins.pc} " ++
            s!"{ins.opcode}{ins.mods.toList} after {repr held}") fun _ => s
        else s
      -- The race check counts a proxy fence as it issues: a `MEMBAR.ALL.CTA` precedes
      -- `FENCE.VIEW.ASYNC.S`, so the warp's earlier requests have reached memory by then.
      if l.races && ins.opcode == "FENCE" && ins.mods.contains "ASYNC" then mem := mem.raceProxyFence si pick
      let w := s.w
      s := { s with w := default }
      let on := guarded ins w m
      let ex := (ops[pc]!).run env w m
      -- The functional mode: results, requests and asynchronous operations complete now.
      let mut ex := ex
      let mut atomWb : Option (Resp → Warp → Warp) := none
      if l.functional then
        let perform (r : Req) : StateT MemSys (Except String) Resp := fun m =>
          (m.performNow si s.cta pick r).map fun (m', resp) =>
            -- With the trace, each mbarrier operation and the word it found.
            if l.trace && (match r.op with | .mbar _ => true | _ => false) then
              dbgTrace (s!"mbar cycle {t} SM {si} CTA {ctaLinear s.env.ctx} warp {s.env.ctx.warp} pc {ins.pc} " ++
                s!"{repr r.op} {repr r.space} at {(r.addrs.filterMap id).toList.take 2} found " ++
                s!"{(resp.toList.take 2).map fun (x : UInt32) => String.ofList (Nat.toDigits 16 x.toNat)}")
                fun _ => (resp, m')
            else (resp, m')
        match ex with
        | .reg w wb => ex := .next (wb w)
        | .mem req w wb =>
          let r := req w
          -- Over several workers, a global atomic that returns a value is the master's to perform,
          -- at the quantum's end, in SM order; the warp waits for its answer.
          if mem.logGlobal && r.space == .global && (match r.op with | .atom .. => true | _ => false) then
            atoms := atoms.push (pick, s.cta, r)
            atomWb := some wb
            ex := .next w
          else
            match (perform r).run mem with
            | .ok (resp, m') =>
              mem := m'
              ex := .next (wb resp w)
            | .error e => return .error s!"{ins.opcode} at pc {ins.pc}: {e}"
        | .async op w wb =>
          match ((op w).run perform).run mem with
          | .ok (resp, m') =>
            mem := m'
            ex := .next (wb resp w)
          | .error e => return .error s!"{ins.opcode} at pc {ins.pc}: {e}"
        | .sync (.fence ..) w | .sync (.proxyFence _) w | .sync .aliasFence w
        | .sync (.asyncCommit _) w => ex := .next w
        | _ => pure ()
      let isMem := match ex with | .mem .. => true | _ => false
      let viaMio := match ex with | .mem .. | .async .. => true | _ => false
      -- Asynchronous MMAs chain on their accumulators in the tensor unit, in issue order: one
      -- may read or write what an earlier one has pending (wgmma's accumulate).
      let chained (l : NV.SASS.Loc) : Bool :=
        (match ex with | .async .. => true | _ => false) && s.pw.isMma l
      -- The hazard check: what the instruction touches against what is pending.
      let fp := fps[pc]!
      -- A memory instruction may overwrite what an earlier one has yet to read: the warp's
      -- memory instructions pass the MIO queue in order, so the earlier reads first.
      -- A later write to a register with a write pending is safe when it lands after it.
      -- A shuffle behind the warp's shared loads still in flight lands after them (`mioHeld`).
      let shfl := ins.opcode == "SHFL" && tm.lateRead == 0
      let behindShared : Array Nat :=
        if shfl then s.sharedLoads.toArray.filter s.inflight.contains else #[]
      -- A shuffle also leaves the MIO after the warp's earlier requests: its result lands after
      -- they read their sources (`lastMem`), so it may overwrite one a store has yet to read.
      let shflLands := if shfl then max (t + tm.latency) (s.lastMem + 1) else t + tm.latency
      let lands := match ex with
        | .reg .. => if behindShared.isEmpty then shflLands else pendingAt
        | _ => pendingAt
      -- Pending entries are per lane: an instruction conflicts only on lanes it runs.
      -- Two writes landing in one cycle land in issue order (an LEA's result overwritten one cycle
      -- later by an IMAD, whose latency is one less, as compiled sm_100 code does).
      let waw (l : NV.SASS.Loc) : Bool :=
        s.pw.hasFor on l && lands < s.pw.landing l pendingAt && !chained l
      -- A late-reading instruction reads after its pending writebacks land by then; a request
      -- reading its vector registers when its read counter releases reads them after the
      -- warp's earlier loads have written them (`readsBehindLoad`).
      let behindLoad (l : NV.SASS.Loc) : Bool :=
        isMem && ins.ctrl.rbar.isSome && (match l with | .r _ => true | _ => false) &&
          (s.loadOf.get? l.index).any s.inflight.contains
      let raw (l : NV.SASS.Loc) : Bool :=
        s.pw.hasFor on l && !chained l && !behindLoad l &&
          !(tm.lateRead > 0 && s.pw.landing l pendingAt ≤ t + tm.lateRead)
      -- A guard reads its predicate later in the pipeline than a data operand does: on parts
      -- that measure it, a predicate can guard only `guardLatency` after its producer issued.
      let guardLoc : Option NV.SASS.Loc := ins.guard.map fun (g, _) =>
        if g.file == .up then .up g.idx else .p g.idx
      let guardEarly := match guardLoc with
        | some gl => (s.guardAt.get? gl).any (· > t)
        | none => false
      let clash := ((fp.reads ++ fp.sets).find? raw).map (·, "writeback") <|>
        (fp.writes.find? waw).map (·, "writeback") <|>
        (if viaMio || shfl then none else ((fp.writes ++ fp.sets).find? (s.pr.hasFor on)).map (·, "source read")) <|>
        (if guardEarly then guardLoc.map (·, "guard latency") else none)
      -- Functionally nothing is pending: timing hazards do not apply.
      if let some (loc, what) := if l.functional then none else clash then
        c := { c with hazards := c.hazards + 1 }
        if hazards.size < 8 then
          let src := if what == "source read" then s.pr else s.pw
          let by_ := match k.code[src.by_[loc.index]!]? with
            | some i => s!"{i.opcode}{i.mods.toList} at {i.pc}"
            | none => "?"
          hazards := hazards.push s!"cycle {t}, SM {si}, CTA {ctaLinear s.env.ctx}, warp {s.env.ctx.warp}: {ins.opcode}{ins.mods.toList} at {ins.pc} uses {repr loc} while its {what} by {by_} is pending"
      s := { s with pw := s.pw.landsBy fp.writes lands pc, pr := s.pr.markBy fp.reads pc }
      if tm.guardLatency > 0 then
        for l in fp.writes do
          match l with
          | .p _ | .up _ => s := { s with guardAt := s.guardAt.insert l (t + tm.guardLatency) }
          | _ => pure ()
      -- The trace: this issue, and the id a writeback it leaves pending takes. A request's reads
      -- and a reduction's answer are filled in when they arrive.
      let land := s.nextLand
      let tix := if l.recordTrace then ((traces.get? (tkey s.env.ctx)).map (·.evs.size)).getD 0 else 0
      if l.recordTrace then
        let pends := match ex with | .reg .. | .mem .. | .collect .. | .async .. => true | _ => false
        let choice : Choice := match ex with
          | .mem .. => .mem #[]
          | .sync (.bar id _ _) _ => .sync (sm.ctas[s.cta]!.gen[id]!)
          | .sync .clusterArrive _ => .sync phases[clusterOf s.env.ctx]!
          | .sync .clusterWait _ => .sync s.clusterPhase
          | .sync .. => .sync 0
          | .collect (.barRed id _ _) _ _ _ _ => .collect (sm.ctas[s.cta]!.gen[id]!) (.ok #[])
          | .async .. => .async #[] (.ok #[])
          | _ => .none
        let issue : TraceEv := .issue pc choice (if pends then some land else none)
        traces := traces.modify (tkey s.env.ctx) fun wt => { wt with evs := wt.evs.push issue }
        if pends then s := { s with nextLand := land + 1 }
      let mut commitBar : Option (Fin 6 × Nat) := none
      match ex with
      | .next w => s := { s with w }
      | .reg w wb =>
        if !behindShared.isEmpty then
          let held := (behindShared, t + tm.latency, Ev.wb wb (some pc) (some land), ins.ctrl.wbar)
          s := { s with mioHeld := s.mioHeld.push held }
        else if tm.lateRead == 0 then
          wbs := wbs.push shflLands { sm := si, slot := pick, ev := .wb wb (some pc) (some land) }
        else
          -- Sources read `lateRead` after issue: the instruction's result is computed then, from
          -- the warp as it is then, and written at its latency.
          wbs := wbs.push (t + tm.lateRead)
            { sm := si, slot := pick, ev := .late env m pc (t + tm.latency) land }
        s := { s with w, pw := s.pw.add fp.writes on }
      | .mem req w wb =>
        -- The request takes its MIO slot now and reaches the memory system no earlier than the
        -- warp's previous request, in issue order. With a read counter it reads its sources when
        -- that releases; without one it reads them now.
        let rr := if tm.readRelease == 0 then p.readRelease.value else tm.readRelease
        let at_ := max (t + rr) s.lastMem
        mem := mem.reserve si q s.cta
        -- The earlier loads still to write what it reads; then its own number, if it loads.
        let after := (vectorRegs fp.reads).filterMap fun l =>
          (s.loadOf.get? l.index).filter s.inflight.contains
        let load := if fp.writes.isEmpty then 0 else s.loads
        if load != 0 then
          s := { s with loads := load + 1, inflight := s.inflight.insert load,
                        loadOf := fp.writes.foldl (fun h l => h.insert l.index load) s.loadOf }
          if ins.opcode == "LDS" || ins.opcode == "LDSM" then
            s := { s with sharedLoads := s.sharedLoads.insert load }
        if ins.ctrl.rbar.isSome then
          -- The read counter covers the vector registers; uniform registers and predicates are
          -- read now.
          let (ur, up, pr) := (w.ur, w.up, w.p)
          let req' (x : Warp) : Req := req { x with ur, up, p := pr }
          wbs := wbs.push at_ { sm := si, slot := pick, ev := .mem req' wb ins.ctrl.wbar t pc true tix land load after }
          s := { s with w, lastMem := at_, pr := s.pr.add (vectorRegs fp.reads) on,
                        pw := s.pw.add fp.writes on }
        else
          let r := req w
          wbs := wbs.push at_ { sm := si, slot := pick, ev := .mem (fun _ => r) wb ins.ctrl.wbar t pc false tix land load after }
          s := { s with w, lastMem := at_, pw := s.pw.add fp.writes on }
        if ins.opcode == "SYNCS" && ins.hasMod "TRYWAIT" then s := { s with atTryWait := true, why := .longScoreboard }
      | .sync (.bar id cnt arrive) w =>
        let ct := sm.ctas[s.cta]!
        let expected := match cnt with | some n => cdiv n 32 | none => 0
        -- Arrivals are counted one per `barrierArrivalCycles` per SM (`facts-h200.md`).
        -- An arrival counts at the end of its slot of the SM's barrier unit.
        let (slot, port) := sm.barPort.reserve t p.barrierArrivalCycles.value
        let counted := slot + p.barrierArrivalCycles.value
        sm := { sm with barPort := port }
        let ct := { ct with arrived := ct.arrived.set! id (ct.arrived[id]! + 1),
                            expected := ct.expected.set! id expected,
                            lastArrival := ct.lastArrival.set! id counted }
        sm := { sm with ctas := sm.ctas.set! s.cta ct }
        mem := mem.barArrive si s.cta id pick
        s := { s with w, atBar := if arrive then none else some id, why := if arrive then .wait else .barrier }
      -- A fence holds the warp until the memory system delivers it, behind the warp's earlier
      -- requests, and then for the fence's own cost (`SyncParams.fenceCycles`, by SASS form; a
      -- form not listed costs nothing more). No proxy but the generic one reaches memory yet, so
      -- a proxy fence orders the same requests.
      | .sync (.fence ..) w | .sync (.proxyFence _) w | .sync .aliasFence w =>
        let form := ".".intercalate (ins.opcode :: ins.mods.toList)
        let own := ((p.sync.fenceCycles.value.find? (·.1 == form)).map (·.2)).getD 0
        wbs := wbs.push (max t s.lastMem) { sm := si, slot := pick, ev := .fence own }
        s := { s with w, atFence := true, why := .fence }
      -- A commit's group releases the instruction's dependence counter when it completes.
      -- An async instruction issued before the commit may not have reached the async units yet
      -- (it reads its sources at its read release): the commit follows it there.
      -- On its read counter alone, the counter is released once the group's operations have read
      -- their sources, their data sent on (`UTMACMDFLUSH`).
      | .sync (.asyncCommit g) w =>
        let bar := ins.ctrl.wbar <|> ins.ctrl.rbar
        let onRead := ins.ctrl.wbar.isNone && ins.ctrl.rbar.isSome
        if s.lastMem > t then
          wbs := wbs.push s.lastMem { sm := si, slot := pick, ev := .commit g bar onRead }
          if let some b := bar then commitBar := some (b, pendingAt)
        else
          let (u, released, done) := AsyncUnit.commit asyncU t si pick g bar onRead
          asyncU := u
          mem := mem.raceAgentsDone si pick done
          if let some b := bar then commitBar := some (b, match released with | some r => max (t + 1) r | none => pendingAt)
        s := { s with w }
      -- SASS waits on async groups through dependence counters and `readiness`.
      | .sync (.asyncWait ..) w => s := { s with w }
      -- `barrier.cluster.arrive`: the warp's arrival in the cluster's current phase, which the
      -- master counts between quanta; the last live warp's completes the phase, and the waiters go
      -- on `clusterBarrierCycles` later.
      | .sync .clusterArrive w =>
        let cid := clusterOf s.env.ctx
        s := { s with w, clusterPhase := phases[cid]! }
        clusterLog := clusterLog.push (cid, t, true)
        mem := mem.barArrive si s.cta Memory.RaceCheck.clusterBarId pick
      -- `barrier.cluster.wait`: on once the phase the warp last arrived in has completed.
      | .sync .clusterWait w =>
        let done := phases[clusterOf s.env.ctx]! > s.clusterPhase
        s := { s with w, atClusterWait := !done, why := if done then .wait else .barrier }
      -- `bar.red` (BAR.RED): a CTA barrier arrival whose threads' predicates are reduced; every
      -- participating warp gets the reduction when the barrier releases.
      | .collect op@(.barRed id cnt _) mask contrib w wb =>
        let ct := sm.ctas[s.cta]!
        let expected := match cnt with | some n => cdiv n 32 | none => 0
        let (slot, port) := sm.barPort.reserve t p.barrierArrivalCycles.value
        let counted := slot + p.barrierArrivalCycles.value
        sm := { sm with barPort := port }
        let mine := (List.range 32).filterMap fun l =>
          if Collective.bit mask l then some (some #[contrib[l]!]) else none
        let ct := { ct with arrived := ct.arrived.set! id (ct.arrived[id]! + 1),
                            expected := ct.expected.set! id expected,
                            lastArrival := ct.lastArrival.set! id counted,
                            redOp := ct.redOp.set! id (some op),
                            redContrib := ct.redContrib.modify id (· ++ mine.toArray),
                            redWaiters := ct.redWaiters.modify id (·.push (pick, wb, pc, tix, land)) }
        sm := { sm with ctas := sm.ctas.set! s.cta ct }
        s := { s with w, atBar := some id, why := .barrier, pw := s.pw.add fp.writes on }
      | .collect .. => return .error s!"{ins.opcode}: thread-level collectives are PTX's"
      -- An asynchronous instruction with a read counter reads its sources when that releases,
      -- like a memory instruction; one without (`HGMMA`) reads them at issue.
      | .async op w wb =>
        -- The race check: a copy's shared-memory sources, read by the async proxy, as the warp's
        -- reads now (its `wait_group.read` releases them before the copy completes); an MMA's are
        -- its own (`AsyncUnit.issue`).
        if l.races then
          let reads : Array Req := match op w with
            | .copy c => (c.moves.map (·.src)).filter fun r => r.space == .shared && r.addrs.any (·.isSome)
            | _ => #[]
          mem := mem.raceAsyncReads t si s.cta pick reads
        -- An arrive-on that tracks the warp's earlier cp.async operations (`ARRIVES.LDGSTSBAR`,
        -- `Copy.afterGroup`) starts after those still to read their sources, in MIO order.
        if ins.opcode == "ARRIVES" && (s.lastMem > t || !s.held.isEmpty) then
          let (ur, up, pr) := (w.ur, w.up, w.p)
          let op' (x : Warp) := op { x with ur, up, p := pr }
          wbs := wbs.push (max t s.lastMem) { sm := si, slot := pick, ev := .async op' wb pc tix land }
          s := { s with w, pr := s.pr.add (vectorRegs fp.reads) on }
        else if ins.ctrl.rbar.isSome then
          let rr := if tm.readRelease == 0 then p.readRelease.value else tm.readRelease
          let at_ := max (t + rr) s.lastMem
          let (ur, up, pr) := (w.ur, w.up, w.p)
          let op' (x : Warp) := op { x with ur, up, p := pr }
          wbs := wbs.push at_ { sm := si, slot := pick, ev := .async op' wb pc tix land }
          s := { s with w, lastMem := at_, pr := s.pr.add (vectorRegs fp.reads) on }
        else
          let o : Origin := { sm := si, smsp := q, cta := s.cta, slot := pick }
          let x := op w
          match AsyncUnit.start p asyncU mem t o x wb pc (trace := if l.recordTrace then some (tix, land) else none) with
          | .error e => return .error s!"{ins.opcode} at pc {ins.pc}: {e}"
          | .ok (u, m', o') => asyncU := u; mem := m'; outs := outs ++ o'
          -- The trace: an MMA chained onto one whose result has not landed reads that result, so
          -- the trace lands it first (now if known, else when it is).
          if l.recordTrace && x.writesRegisters then
            let a := asyncU.agents.size - 1
            let g := asyncU.agents[a]!
            if let (true, some pr) := (g.chained, g.source) then
              let pg := asyncU.agents[pr]!
              let key := tkey s.env.ctx
              if !landedIds.contains (key, pg.land) then
                let early : TraceEv := .land pg.land #[]
                traces := traces.modify key fun wt => { wt with evs := wt.evs.insertIdxIfInBounds tix early }
                asyncU := { asyncU with agents := asyncU.agents.modify a ({ · with tix := some (tix + 1) }) }
                match pg.task with
                | some tk =>
                  let ev : TraceEv := .land pg.land (landed fps[pg.pc]!.writes (pg.wb tk.get w))
                  traces := traces.modify key fun wt => { wt with evs := wt.evs.set! tix ev }
                  landedIds := landedIds.insert (key, pg.land)
                | none => asyncU := { asyncU with agents := asyncU.agents.modify pr ({ · with early := some tix }) }
          s := { s with w, pw := if x.writesRegisters then (s.pw.add fp.writes on).setMma fp.writes else s.pw }
      | .exit => s := { s with live := false }
      | .fault msg => return .error msg
      -- A counter gaining an event drops its released ones.
      let mut sb := s.sb
      s := { s with sb := #[] }
      if let some b := ins.ctrl.wbar then
        -- A memory request's counter is released by its delivery; until then it is pending.
        let release := if isMem || !behindShared.isEmpty then pendingAt else max (t + max tm.latency 1) shflLands
        let release := match commitBar with | some (b', r) => if b' == b then r else release | none => release
        sb := sb.modify b fun es => (es.filter (· > t)).push release
        s := { s with sbMem := if isMem then s.sbMem ||| (1 <<< b.val) else s.sbMem - (s.sbMem &&& (1 <<< b.val)) }
      if let some b := ins.ctrl.rbar then
        let rr := if tm.readRelease == 0 then p.readRelease.value else tm.readRelease
        let release := match commitBar with
          | some (b', r) => if b' == b && ins.ctrl.wbar != some b then r else t + rr
          | none => t + rr
        sb := sb.modify b fun es => (es.filter (· > t)).push release
      -- Yield: the warp does not issue in the next cycle, and the scheduler's greedy choice
      -- moves on (Huerta et al. 2025 §4; measured on Ampere, inferred for Hopper). A taken
      -- control transfer redirects the warp's fetch (`takenBranchCycles`).
      let taken := match s.w.current with
        | some (next, _) => next != pc + 1 && transfers.contains ins.opcode
        | none => false
      -- Functionally a warp waits for every other warp's turn (none is ever stalled, so the
      -- greedy choice would keep a spinning warp).
      let hold := if l.functional then sm.slots.size else
        (if ins.ctrl.yield then max ins.ctrl.stall 2 else max ins.ctrl.stall 1) +
        (if taken then p.frontEnd.takenBranchCycles.value else 0)
      -- While the sub-partition's tensor core runs a warpgroup MMA, its pipes lose a share of
      -- their time: the instruction's occupancy stretches (`TensorParams.wgmmaPipeSharePermille`).
      let share := p.tensor.wgmmaPipeSharePermille.value
      let mut interval := tm.interval
      if share > 0 && !l.functional && tm.pipe != .gmma && tensorBusy p asyncU si q t then
        let den := 1000 - share
        let b := interval * share + sm.stealCarry[q]!
        sm := { sm with stealCarry := sm.stealCarry.set! q (b % den) }
        interval := interval + b / den
      s := { s with readyAt := max s.readyAt (t + hold), sb, live := s.live && !s.w.done,
                    interlock := ins.ctrl.stall == 0 && p.frontEnd.stallZeroInterlocks.value }
      -- `NANOSLEEP t` suspends the warp for about `t` ns, at most 1 ms (PTX `nanosleep`: between 0
      -- and twice `t`; the model takes `t`). `NANOSLEEP.SYNCS`, a barrier wait's backoff, wakes on
      -- the barrier and is not held here.
      if ins.opcode == "NANOSLEEP" && !ins.mods.contains "SYNCS" && !l.functional then
        let lane := (List.range 32).find? (fun i => (s.w.live >>> i.toUInt32) &&& 1 == 1) |>.getD 0
        let ns : Nat := match ins.operands[0]? with
          | some (NV.SASS.Operand.imm v) => v.toNat
          | some (NV.SASS.Operand.reg ⟨.r, r⟩ _) => (s.w.getR r lane).toNat
          | _ => 0
        let sleep := min ns 1000000 * p.clockMHz.value / 1000
        if sleep > hold then s := { s with readyAt := max s.readyAt (t + sleep), why := .sleeping }
      if atomWb.isSome then s := { s with readyAt := pendingAt, atomWb }
      -- A `printf` call holds the warp until it returns (`SyncParams.printfPs`), timed alone.
      if s.w.printing > 0 then
        if !l.functional then
          let ps := p.sync.printfPs.value + p.sync.printfLanePs.value * (s.w.printing - 1)
          s := { s with readyAt := max s.readyAt (t + ps * p.clockMHz.value / 1000000), why := .fence }
        s := { s with w := { s.w with printing := 0 } }
      let ci := s.cta
      let exited := !s.live
      -- Fixed-latency instructions read their sources through the register banks; the reuse
      -- cache serves a source the same warp flagged at the same position.
      let srcs := sources[pc]!
      -- The reuse cache (Huerta et al. 2025 §5.3.1): per bank, one entry of a slot per operand
      -- position. A source hits the slot of its register's bank and its position holding the
      -- same warp and register; any read of that bank at that position replaces the slot, kept
      -- only if the operand's reuse bit is set.
      let cache := sm.reuse[q]!
      let at_ (i : Nat) : Nat := (srcs[i]! % banks) * slots + i
      let hit (i : Nat) : Bool := i < slots && cache[at_ i]! == some (pick, srcs[i]!)
      if ins.ctrl.wbar.isNone then
        let rc := readCycles p srcs hit
        sm := { sm with rfRead := sm.rfRead.modify q (fun (port : Port) => (port.reserve t rc).2) }
      if !srcs.isEmpty then
        let mut cache := cache
        for i in [0:min srcs.size slots] do
          if srcs[i]! != 255 then
            cache := cache.set! (at_ i) (if (ins.ctrl.reuse >>> i) % 2 == 1 then some (pick, srcs[i]!) else none)
        sm := { sm with reuse := sm.reuse.set! q cache }
      sm := { sm with slots := sm.slots.set! pick { s.settle with issuedAt := t },
                      pipes := sm.pipes.modify (q * Pipe.count + tm.pipe.index)
                        (fun (port : Port) => (port.reserve t interval).2),
                      pipeWork := sm.pipeWork.modify (q * Pipe.count + tm.pipe.index) (· + interval),
                      issued := sm.issued.modify q (· + 1),
                      last := sm.last.set! q (if ins.ctrl.yield then none else some pick) }
      c := { c with instExecuted := c.instExecuted + 1 }
      issued := true
      if exited then
        -- An exited warp no longer counts toward its cluster's barrier.
        clusterLog := clusterLog.push (clusterOf s.env.ctx, t, false)
        sm := { sm with bySmsp := sm.bySmsp.modify q (·.filter (· != pick)) }
        let ct := sm.ctas[ci]!
        sm := { sm with ctas := sm.ctas.set! ci { ct with live := ct.live - 1 } }
        if ct.live == 1 then
          sm := if l.spans then dbgTrace s!"cta {ct.id} sm {si} start {ct.placedAt} end {t + 1}" fun _ => sm else sm
          sm := { sm with resident := sm.resident - 1 }
          doneCtas := doneCtas + 1
          mem := mem.removeCta si ci
          c := { c with cycles := max c.cycles (t + 1) }
          -- A CTA reserved for this SM starts in the freed slot.
          if !reserved.isEmpty then
            let id := reserved[0]!
            reserved := reserved.extract 1 reserved.size
            let ci := sm.ctas.size
            let (sm', m', tr) := placeCta kit sm mem si id (t + p.launch.ctaDispatchCycles.value.2) traces
            sm := sm'; mem := m'; traces := tr
            placed := placed.push (si, ci, id, 0, k.sharedReserved + k.sharedBytes + l.dynShared)
      -- A barrier whose expected warps have all arrived releases its waiters `barrierCycles`
      -- after the last arrival was counted (only an arrival or an exit can complete one).
      let ct := sm.ctas[ci]!
      let barrierMoved := exited || ins.opcode == "BAR" || ins.opcode == "WARPGROUP"
      for id in [0:if barrierMoved then ctaBarriers else 0] do
        let expected := if ct.expected[id]! == 0 then ct.live else ct.expected[id]!
        if ct.arrived[id]! > 0 && ct.arrived[id]! ≥ expected then
          let release := ct.lastArrival[id]! + p.barrierCycles.value
          -- A reduction's answer, the same for every participant, written at the release.
          if let some op := ct.redOp[id]! then
            match Collective.answer op 0 ct.redContrib[id]! with
            | .error e => return .error s!"bar.red: {e}"
            | .ok answers =>
              let resp := (answers.find? (!·.isEmpty)).getD #[0]
              for (slot, wb, wpc, wtix, wland) in ct.redWaiters[id]! do
                if l.recordTrace then
                  let issue : TraceEv := .issue wpc (.collect ct.gen[id]! (.ok resp)) (some wland)
                  traces := traces.modify (tkey sm.slots[slot]!.env.ctx) fun wt => { wt with evs := wt.evs.set! wtix issue }
                wbs := wbs.push release { sm := si, slot, ev := .wb (wb resp) (some wpc) (some wland) }
            let c0 := sm.ctas[ci]!
            let c1 := { c0 with redOp := c0.redOp.set! id none, redContrib := c0.redContrib.set! id #[],
                                redWaiters := c0.redWaiters.set! id #[] }
            sm := { sm with ctas := sm.ctas.set! ci c1 }
          mem := mem.barRelease si ci id (ct.warps.filter fun i => sm.slots[i]!.atBar == some id)
          for i in ct.warps do
            let w := sm.slots[i]!
            if w.atBar == some id then
              let w := { w with atBar := none, readyAt := release }
              sm := { sm with slots := sm.slots.set! i w }
          let ct := sm.ctas[ci]!
          let ct := { ct with arrived := ct.arrived.set! id 0, gen := ct.gen.modify id (· + 1) }
          sm := { sm with ctas := sm.ctas.set! ci ct }
          wake := min wake release
    sm := { sm with next := if issued then t + 1 else wake }
    sms := sms.set! si sm
  let final : St :=
    { st with sms := sms, mem := mem, memWbs := memWbs, wbs := wbs, printfOut := printfOut, asyncU := asyncU, outs := outs, doneCtas := doneCtas, c := c, t := t, hazards := hazards, clusterLog := clusterLog, reserved := reserved, placed := placed, traces := traces, landedIds := landedIds, atoms := atoms }
  return .ok (final, true)

/-- Phase A of a quantum for SMs `ids`, one after another on one copy of the SMs and the memory
system: each runs its own events up to `stop`. The SMs, the memory system and each SM's state
after (whose own `sms` and `mem` are dropped: the group's copy holds them). -/
private def runGroup (kit : Kit) (ids : Array Nat) (stop : Nat) (sms : Array Sm) (mem : MemSys)
    (states : Array St) : Except String (Array Sm × MemSys × Array St) := do
  let mut sms := sms
  let mut mem := mem
  let mut out := #[]
  for j in [0:ids.size] do
    let mut st := { states[j]! with sms, mem }
    sms := #[]; mem := default
    -- Each step moves the SM's time forward; the bound only guards a defect.
    for _ in [0:4000000000] do
      let (st', moved) ← pass kit ids[j]! stop st
      st := st'
      if !moved then break
    sms := st.sms; mem := st.mem
    out := out.push { st with sms := #[], mem := default }
  return (sms, mem, out)

/-! ## Workers

When SMs run in parallel, each belongs to one worker for the whole launch: SM `i` to worker
`i % n`. A worker holds its SMs' cores, their halves of the memory system and their run states, and
runs phase A of each quantum on them; the master keeps the L2 side, dispatches CTAs and runs phase
B. Only small messages pass between them, so a worker's state never leaves its thread. -/

/-- What the master asks of a worker for one quantum. -/
private structure Cmd where
  /-- The quantum `[t, stop)`. -/
  t : Nat
  stop : Nat
  /-- Global memory, the directory and the quantum's start, as the quantum begins. -/
  global : Store
  dir : Memory.Directory
  start : Nat
  /-- The L2 side's answers for the worker's SMs. -/
  answers : Array (Memory.Stamped Memory.Msg) := #[]
  /-- CTAs to place: SM, CTA id, the cycle its warps may start. -/
  placements : Array (Nat × Nat × Nat) := #[]
  /-- Per SM: waiting CTAs it may start in slots its CTAs free this quantum. -/
  reserved : Array (Nat × Array Nat) := #[]
  /-- Cluster barrier phases the master completed: cluster, its phase now, the cycle its waiters
  go on. -/
  releases : Array (Nat × Nat × Nat) := #[]
  /-- The functional mode over several workers: the answers to its warps' global atomics (SM,
  warp slot, response). -/
  atomAnswers : Array (Nat × Nat × Resp) := #[]
  /-- Hand back the SMs' state instead: the launch is over. -/
  finish : Bool := false

private instance : Inhabited Cmd := ⟨{ t := 0, stop := 0, global := Store.empty 0 0, dir := {}, start := 0 }⟩

/-- What a worker reports of one SM after a quantum. -/
private structure SmReport where
  sm : Nat
  resident : Nat
  /-- The earliest cycle it has something to do. -/
  earliest : Option Nat
  doneCtas : Nat
  /-- Reserved CTAs it did not start. -/
  returned : Array Nat
  /-- CTAs it placed in slots its CTAs freed (`Cmd.reserved`), for the L2 side and the directory:
  SM, CTA slot, cluster, rank, shared bytes. -/
  placed : Array (Nat × Nat × Nat × Nat × Nat)
  /-- Its messages and records for the L2 side. -/
  msgs : Array (Memory.Stamped Memory.Msg)
  logs : Array (Memory.Stamped Memory.Log)
  /-- Its warps' cluster barrier arrivals and exits (`St.clusterLog`). -/
  clusterLog : Array (Nat × Nat × Bool)
  /-- Its warps' global atomics, for the master (`St.atoms`). -/
  atoms : Array (Nat × Nat × Req) := #[]
  deriving Inhabited

private structure Rep where
  error : Option String := none
  /-- The functional mode on one worker: the global store, handed back (`Cmd.global` is the store
  itself); on several, the global writes its SMs made (`MemSys.globalLog`). -/
  global : Option Store := none
  globalLog : Array (Nat × Req) := #[]
  sms : Array SmReport := #[]
  /-- At the end: each SM's core, memory-system half and run state. -/
  final : Array (Nat × Sm × Memory.SmMem × St) := #[]
  deriving Inhabited

private structure Worker where
  ids : Array Nat
  sms : Array Sm
  mem : MemSys
  runs : Array St
  /-- Each cluster's barrier phase, as the master last sent it. -/
  phases : Array Nat
  deriving Inhabited

/-- One quantum on a worker's SMs: deliver the L2's answers and the cluster barriers' releases,
place the CTAs it is given, run phase A on each SM to the quantum's end, and report. -/
private def Worker.step (kit : Kit) (w : Worker) (cmd : Cmd) : Except String (Worker × Rep) := do
  if cmd.finish then
    return (w, { final := (Array.range w.ids.size).map fun j =>
                   let si := w.ids[j]!
                   (si, w.sms[si]!, w.mem.sms[si]!, w.runs[j]!) })
  let mut mem := (w.mem.atQuantum cmd.global cmd.dir cmd.start).deliverAnswers cmd.answers
  let mut sms := w.sms
  let mut runs := w.runs
  let mut phases := w.phases
  -- Global atomics the master performed: their warps go on.
  for (si, slot, resp) in cmd.atomAnswers do
    let sm := sms[si]!
    let x := sm.slots[slot]!
    if let some wb := x.atomWb then
      let x' : Slot := { x with w := wb resp x.w, atomWb := none, readyAt := cmd.t }
      sms := sms.set! si { sm with slots := sm.slots.set! slot x', next := min sm.next cmd.t }
  -- A completed cluster phase lets its waiting warps on.
  for (cid, phase, release) in cmd.releases do
    phases := phases.set! cid phase
    for si in w.ids do
      let sm := sms[si]!
      let mut slots := sm.slots
      let mut woke := false
      for i in [0:slots.size] do
        let x := slots[i]!
        if x.atClusterWait && clusterOfDims x.env.ctx kit.cluster == cid then
          slots := slots.set! i { x with atClusterWait := false, readyAt := max x.readyAt release }
          woke := true
      -- The race check: each of the cluster's CTAs here completes the phase as a CTA barrier.
      let inCluster := (List.range slots.size).filter fun i =>
        slots[i]!.live && clusterOfDims slots[i]!.env.ctx kit.cluster == cid
      for cta in (inCluster.map (slots[·]!.cta)).eraseDups do
        mem := mem.barRelease si cta Memory.RaceCheck.clusterBarId
          (inCluster.filter (slots[·]!.cta == cta)).toArray
      if woke then sms := sms.set! si { sm with slots, next := min sm.next release }
  -- The master's placements (it registered them with the L2 side and the directory already).
  for (si, id, start) in cmd.placements do
    let j := w.ids.idxOf si
    let sm := sms[si]!
    sms := sms.set! si default
    let (sm, m', tr) := placeCta kit sm mem si id start runs[j]!.traces
    sms := sms.set! si sm
    mem := m'
    runs := runs.modify j fun r => { r with traces := tr }
  let states := (Array.range w.ids.size).map fun j =>
    { runs[j]! with t := max runs[j]!.t cmd.t, doneCtas := 0, phases, clusterLog := #[],
                    reserved := ((cmd.reserved.find? (·.1 == w.ids[j]!)).map (·.2)).getD #[] }
  let (sms', mem', states) ← runGroup kit w.ids cmd.stop sms mem states
  -- The snapshot is dropped, so the master's global memory stays unshared; functionally it is the
  -- store itself, handed back.
  let global := if kit.l.functional && !mem'.logGlobal then some mem'.global else none
  let globalLog := mem'.globalLog
  mem := { mem'.atQuantum (Store.empty 0 0) mem'.dir mem'.start with globalLog := #[] }
  let mut reports : Array SmReport := #[]
  for j in [0:w.ids.size] do
    let si := w.ids[j]!
    let st := states[j]!
    let (m2, o, lg) := mem.takeOutSm si
    mem := m2
    let sm := sms'[si]!
    let cands := [st.wbs.minKey?, mem.nextSm si, if sm.resident > 0 then some sm.next else none,
                  if st.outs.isEmpty && st.atoms.isEmpty then none else some cmd.stop]
    let earliest := cands.foldl (fun acc x => match acc, x with
      | some a, some b => some (min a b) | none, x => x | a, none => a) none
    reports := reports.push { sm := si, resident := sm.resident, earliest, doneCtas := st.doneCtas,
                              returned := st.reserved, placed := st.placed,
                              msgs := o, logs := lg, clusterLog := st.clusterLog, atoms := st.atoms }
  let runs' := states.map fun st => { st with reserved := #[], placed := #[], doneCtas := 0, atoms := #[] }
  return ({ w with sms := sms', mem, runs := runs', phases }, { sms := reports, global, globalLog })

/-- How the workers run: in place, one after another, or each on its own thread. `σ` is what the
master holds of them. -/
private structure Pool (m : Type → Type) (σ : Type) where
  start : Kit → Array Worker → m σ
  /-- One command per worker; their reports, in worker order. -/
  round : σ → Array Cmd → m (σ × Array (Except String Rep))

private def inlinePool : Pool Id (Kit × Array Worker) where
  start kit ws := (kit, ws)
  round := fun (kit, ws) cmds => Id.run do
    let mut ws := ws
    let mut reps := #[]
    for i in [0:ws.size] do
      let w := ws[i]!
      ws := ws.set! i default
      match Worker.step kit w cmds[i]! with
      | .error e => reps := reps.push (.error e)
      | .ok (w', rep) => ws := ws.set! i w'; reps := reps.push (.ok rep)
    return ((kit, ws), reps)

/-- A worker thread's mailbox: the command, the promise its reply resolves, and the promise that
wakes it for the next command. -/
private abbrev Box := IO.Ref (Cmd × IO.Promise Rep × IO.Promise Unit)

private structure Handle where
  box : Box
  wake : IO.Promise Unit

/-- A worker thread: wait to be woken, run the command in the mailbox, reply; until the last. -/
private def workerLoop (kit : Kit) (w0 : Worker) (box : Box) (wake0 : IO.Promise Unit) : IO Unit := do
  let mut w := w0
  let mut wake := wake0
  -- Each pass is one quantum's command; the bound only guards a defect.
  for _ in [0:4000000000] do
    let _ ← IO.wait wake.result?
    let (cmd, reply, next) ← box.get
    box.set (default, reply, next)
    wake := next
    match Worker.step kit w cmd with
    -- The master stops the launch and sends every worker its finish, this one included.
    | .error e => reply.resolve { error := some e }
    | .ok (w', rep) =>
      w := w'
      reply.resolve rep
      if cmd.finish then return

private def threadPool : Pool IO (Array Handle) where
  start kit ws := ws.mapM fun w => do
    let wake ← IO.Promise.new
    let reply ← IO.Promise.new
    let box ← IO.mkRef ((default : Cmd), reply, wake)
    let _ ← IO.asTask (workerLoop kit w box wake) (prio := .dedicated)
    return { box, wake }
  round hs cmds := do
    let mut next : Array Handle := #[]
    let mut replies : Array (IO.Promise Rep) := #[]
    for (h, cmd) in hs.zip cmds do
      let reply ← IO.Promise.new
      let wake ← IO.Promise.new
      h.box.set (cmd, reply, wake)
      h.wake.resolve ()
      next := next.push { h with wake }
      replies := replies.push reply
    let mut reps := #[]
    for r in replies do
      match (← IO.wait r.result? : Option Rep) with
      | some rep => reps := reps.push (match rep.error with | some e => .error e | none => .ok rep)
      | none => reps := reps.push (.error "a worker thread ended without replying")
    return (next, reps)

/-- The microarchitectural state that persists from one launch to the next on a GPU: what its
caches hold, and the time elapsed. Memory contents flow through `Launch.global` and
`Result.global`; this is everything else. -/
structure Machine where
  /-- Per SM: its instruction caches, nearest first. -/
  code : Array (Array CodeLevel)
  /-- What the L2 holds (`NV.Core.Memory`); empty when fresh. -/
  l2 : Array Memory.Tags := #[]
  /-- Cycles elapsed since the machine was fresh. -/
  clock : Nat := 0
  /-- The modules whose code has been loaded onto it (`Kernel.module`). -/
  modules : List UInt64 := []
  deriving Inhabited

/-- A machine with every cache empty (after a reset, or caches flushed). -/
def Machine.fresh (p : Params) : Machine where
  code := Array.replicate p.sms.value <|
    (p.frontEnd.codeCaches.value.map fun (bytes, rate) =>
      ({ capacity := bytes / p.mem.lineBytes.value, rate } : CodeLevel)).toArray

/-- Run the launch on a machine in a given state, to completion, the SMs' quanta run by `pool`:
the machine after it, and the launch's result. -/
private def runOnWith {m : Type → Type} {σ : Type} [Monad m] (pool : Pool m σ) (p : Params)
    (machine : Machine) (l : Launch) : m (Except String (Machine × Result)) := do
  let k := l.kernel
  let (bx, by_, bz) := l.block
  let (gx, gy, gz) := l.grid
  let threads := bx * by_ * bz
  let nWarps := cdiv threads 32
  let ctaCount := gx * gy * gz
  let regsPerWarp := cdiv (max k.regs 1 * 32) p.regAllocUnit.value * p.regAllocUnit.value
  let smemPerCta := k.sharedBytes + l.dynShared + p.smemReservedPerCta.value
  if threads == 0 || threads > 1024 then return .error s!"block of {threads} threads"
  -- A kernel using a unit its part lacks was built for another target.
  for i in k.code do
    if let some what := p.features.value.lacks i.opcode then
      return .error s!"{i.opcode} at pc {i.pc} needs {what}, which the {p.name} lacks; build the kernel for {p.target.value}"
  if l.clusterShape != (1, 1, 1) && !p.features.value.clusters then
    return .error s!"a launch with clusters on the {p.name}, which has none"
  if smemPerCta > p.smemPerCtaMax.value + p.smemReservedPerCta.value then
    return .error s!"{smemPerCta} bytes of shared memory per CTA"
  let ctasPerSM := min p.maxCtasPerSM.value <| min (p.maxWarpsPerSM.value / nWarps) <|
    min (p.maxThreadsPerSM.value / (nWarps * 32)) <|
    min (p.regsPerSM.value / (regsPerWarp * nWarps)) (p.smemPerSM.value / max smemPerCta 1)
  if ctasPerSM == 0 then return .error "the CTA does not fit on an SM"
  let const0 := constBank0 l
  -- Each instruction's timing class, looked up once.
  let timings := k.code.map p.timing
  let sources := k.code.map regSources
  -- Each instruction's semantics, decoded once.
  let ops := decode k
  let fps := k.code.map NV.SASS.footprint
  let lineBytes := p.mem.lineBytes.value
  let prefetch := p.frontEnd.codePrefetchLines.value
  let codeLines := (l.codeBase + k.code.foldl (fun m i => max m i.pc) 0) / lineBytes + 1
  let smsps := p.smsps.value
  let banks := max 1 p.regFile.banks.value
  let slots := max 1 (p.regFile.reuseEntriesPerBank.value * p.regFile.reuseSlotsPerEntry.value)
  let nSms := p.sms.value
  let (cx, cy, cz) := l.clusterShape
  let csize := cx * cy * cz
  if gx % cx != 0 || gy % cy != 0 || gz % cz != 0 then
    return .error s!"grid {l.grid} is not a whole number of {l.clusterShape} clusters"
  if csize > p.maxClusterCtas.value then
    return .error s!"clusters of {csize} CTAs on the {p.name}, whose limit is {p.maxClusterCtas.value}"
  -- The GPCs' SMs, in index order (inferred: SM indices follow GPCs).
  let gpcSize := p.gpcSMs.value.toArray
  let gpcStart := (gpcSize.foldl (fun (acc : Array Nat × Nat) n => (acc.1.push acc.2, acc.2 + n)) (#[], 0)).1
  if csize > 1 && gpcSize.all (· * ctasPerSM < csize) then
    return .error s!"a cluster of {csize} CTAs fits on no GPC of the {p.name}"
  let constLine := p.frontEnd.constLineBytes.value
  let constRefs := k.code.map fun i => (i.operands.filterMap fun
    | .cbank b none off => some (b, off.toNat / constLine)
    | _ => none).toList.eraseDups.toArray
  let constCapacity := p.frontEnd.constCacheBytes.value / constLine
  -- The parameter block in device memory (`paramAddrOf`), where the async unit fetches a
  -- parameter's tensor map.
  let paramAddr := paramAddrOf l
  let kit : Kit := { p, k, l, ops, fps, timings, sources, constRefs, constCapacity, lineBytes,
                     prefetch, codeLines, smsps, nSms, const0, threads, nWarps,
                     cluster := l.clusterShape, paramAddr }
  -- Each cluster's barrier, kept by the master (`ClusterBar`).
  let mut clusterBars : Array ClusterBar := Array.replicate (ctaCount / csize) {}
  let clustered := csize > 1 || k.code.any (·.opcode.startsWith "UCGABAR")
  let clusterOfId (id : Nat) : Nat :=
    let (x, y, z) := (id % gx, (id / gx) % gy, id / (gx * gy))
    x / cx + (gx / cx) * (y / cy + (gy / cy) * (z / cz))
  -- The module's data (`__device__` variables, `printf` format strings) at the addresses its
  -- relocations name; the global store grows to cover them.
  let mut global := l.global
  -- Covered 128 bytes past its end: a 128-byte tensor-map fetch from the block's tail reads
  -- zeros past it.
  global := global.cover paramAddr (l.params.size + 128)
  for i in [0:l.params.size] do
    global := global.writeByte ((paramAddr - global.base).toNat + i) l.params[i]!
  for (addr, bytes) in if k.dataLoaded then #[] else k.data do
    global := global.cover addr.toUInt64 bytes.size
    for i in [0:bytes.size] do
      global := global.writeByte (addr - global.base.toNat + i) bytes[i]!
  let mut mem := MemSys.init { Memory.Config.ofParams p (ctasPerSM * smemPerCta) with races := l.races }
    nSms global machine.l2
  -- A module's first launch on the machine loads its code, through the L2 (inferred).
  let firstLoad := !machine.modules.contains k.module
  if firstLoad then
    mem := mem.loadCode (List.range' (l.codeBase / lineBytes) (codeLines - l.codeBase / lineBytes))
  let sm0 : Sm :=
    { bySmsp := Array.replicate smsps #[], pipes := Array.replicate (smsps * Pipe.count) {},
      pipeWork := Array.replicate (smsps * Pipe.count) 0, issued := Array.replicate smsps 0,
      rfRead := Array.replicate smsps {}, stealCarry := Array.replicate smsps 0,
      reuse := Array.replicate smsps (Array.replicate (banks * slots) none), smspHold := Array.replicate smsps 0,
      code := #[], last := Array.replicate smsps none }
  let quantum := mem.quantum
  -- The kernel's entry line and the prefetcher's window after it are in every SM's instruction
  -- caches when its CTAs start: they are fetched during the launch.
  let entry := l.codeBase / lineBytes
  let warm (levels : Array CodeLevel) : Array CodeLevel :=
    (List.range' entry (min (prefetch + 1) (codeLines - entry))).foldl
      (fun lv line => lv.map (·.touch line 0)) levels
  let run0 : St := { sms := #[], mem := default, memWbs := {}, wbs := {}, printfOut := #[],
                     asyncU := AsyncUnit.init nSms smsps, outs := #[], doneCtas := 0, c := {}, t := 0,
                     hazards := #[], phases := Array.replicate clusterBars.size 0 }
  -- The SMs move to `nW` workers for the whole launch, SM `i` to worker `i % nW`; timed results do
  -- not depend on how they are grouped (`Launch.workers`). Each quantum a worker runs its SMs'
  -- phase A; the master keeps the L2 side, dispatches CTAs, counts the cluster barriers and runs
  -- phase B, and keeps per SM its resident CTAs and the earliest cycle it has something to do.
  -- Functionally a CTA reaches its cluster peers' shared memory at once, so with clusters the SMs
  -- are dealt by GPC (a cluster's CTAs are on one GPC); over several workers each works on its
  -- own copy of global memory, whose writes the master makes at the quantum's end
  -- (`MemSys.globalLog`).
  let byGpc := l.functional && csize > 1
  let gpcOf (i : Nat) : Nat :=
    ((List.range gpcSize.size).find? fun g => gpcStart[g]! ≤ i && i < gpcStart[g]! + gpcSize[g]!).getD i
  let nW := min (if byGpc then gpcSize.size else nSms) (if l.workers == 0 then 8 else l.workers)
  let wOf (i : Nat) : Nat := (if byGpc then gpcOf i else i) % nW
  let ws := (Array.range nW).map fun w =>
    let ids := (Array.range nSms).filter (wOf · == w)
    ({ ids, sms := ids.foldl (fun a i => a.set! i { sm0 with code := warm (machine.code[i]?.getD #[]) })
                    (Array.replicate nSms default),
       mem := { mem.forWorker ids with logGlobal := l.functional && nW > 1 }, runs := ids.map fun _ => run0,
       phases := Array.replicate clusterBars.size 0 } : Worker)
  mem := mem.withoutSms (Array.range nSms)
  let mut ps ← pool.start kit ws
  -- On an error the workers are finished first: a worker thread left waiting for its next
  -- command keeps the process from exiting.
  let abort (ps : σ) (e : String) : m (Except String (Machine × Result)) := do
    let _ ← pool.round ps (Array.replicate nW { (default : Cmd) with finish := true })
    return .error e
  let mut nextCta := 0
  -- The first CTA dispatches once the kernel has started on the GPCs.
  let mut dispatcher : Port := { free := p.launch.gpcLaunchCycles.value }
  let mut doneCtas := 0
  -- CTA ids handed back unused by a quantum's reservations, dispatched before fresh ones.
  let mut returned : Array Nat := #[]
  let mut t := 0
  let mut resident : Array Nat := Array.replicate nSms 0
  let mut earliest : Array (Option Nat) := Array.replicate nSms none
  let mut answersFor : Array (Array (Memory.Stamped Memory.Msg)) := Array.replicate nW #[]
  let mut placeFor : Array (Array (Nat × Nat × Nat)) := Array.replicate nW #[]
  let mut atomsFor : Array (Array (Nat × Nat × Resp)) := Array.replicate nW #[]
  let mut releases : Array (Nat × Nat × Nat) := #[]
  -- Per SM: CTAs placed on it so far (the next one's slot), and each CTA's shared bytes.
  let mut ctasOn : Array Nat := Array.replicate nSms 0
  let ctaBytes := k.sharedReserved + k.sharedBytes + l.dynShared
  -- The cycle the launch was stopped at (`Launch.untilCycle`), if it was.
  let mut stopped : Option Nat := none
  -- Each pass is one quantum; the bound only guards a defect.
  for _ in [0:4000000000] do
    -- Dispatch breadth first: each CTA to the SM with the fewest resident CTAs, lowest index among
    -- equals, so a small grid gets an SM per CTA (inferred: the hardware spreads CTAs over SMs
    -- rather than filling one). A cluster's CTAs are
    -- dispatched together, all on SMs of one GPC (CUDA Programming Guide §"Thread Block
    -- Clusters"), on the GPC with the most free CTA slots; the GPCs' SMs are taken in index order.
    let mut placements : Array (Nat × Nat) := #[]
    for _ in [0:ctaCount] do
      if nextCta ≥ ctaCount && returned.isEmpty then break
      -- Free CTA slots, and the SM of `is` with the most, given the placements so far (passed in:
      -- a closure would keep the placements as they were when it was made).
      let freeIn (ps : Array (Nat × Nat)) (i : Nat) : Nat :=
        ctasPerSM - min ctasPerSM (resident[i]! + (ps.filter (·.1 == i)).size)
      let most (ps : Array (Nat × Nat)) (is : List Nat) : Option Nat := is.foldl (fun best i =>
          if freeIn ps i == 0 then best
          else match best with
            | some b => if freeIn ps i > freeIn ps b then some i else best
            | none => some i) none
      let free := freeIn placements
      if csize == 1 then
        let some si := most placements (List.range nSms) | break
        if let some id := returned[0]? then
          placements := placements.push (si, id)
          returned := returned.extract 1 returned.size
        else
          placements := placements.push (si, nextCta)
          nextCta := nextCta + 1
      else
        let room (g : Nat) : Nat := (List.range' gpcStart[g]! gpcSize[g]!).foldl (· + free ·) 0
        let some g := (List.range gpcSize.size).foldl (fun best g =>
            if room g < csize then best
            else match best with
              | some b => if room g > room b then some g else best
              | none => some g) none
          | break
        -- The cluster's CTAs in rank order, each to the GPC's SM with the most free slots.
        let cl := nextCta / csize
        let (ncx, ncy) := (gx / cx, gy / cy)
        let (ox, oy, oz) := ((cl % ncx) * cx, ((cl / ncx) % ncy) * cy, (cl / (ncx * ncy)) * cz)
        for r in [0:csize] do
          let some si := most placements (List.range' gpcStart[g]! gpcSize[g]!) | break
          let (x, y, z) := (ox + r % cx, oy + (r / cx) % cy, oz + r / (cx * cy))
          placements := placements.push (si, x + gx * (y + gy * z))
        nextCta := nextCta + csize
    -- The GPU's dispatcher hands out one CTA at a time (`LaunchParams.ctaDispatchCycles`); each
    -- goes to the worker that owns its SM.
    for (si, id) in placements do
      let (firstWave, later) := p.launch.ctaDispatchCycles.value
      let (start, port) := dispatcher.reserve t (if id < nSms * ctasPerSM then firstWave else later)
      dispatcher := port
      resident := resident.modify si (· + 1)
      placeFor := placeFor.modify (wOf si) (·.push (si, id, start))
      -- Registered with the L2 side and the cluster directory now, so its cluster's peers reach its
      -- shared memory from the quantum it starts in.
      let (x, y, z) := (id % gx, (id / gx) % gy, id / (gx * gy))
      mem := mem.addCtaL2 si ctasOn[si]! (clusterOfId id) (x % cx + cx * (y % cy + cy * (z % cz))) ctaBytes
        (reservedInit p (cx * cy * cz) (x % cx + cx * (y % cy + cy * (z % cz))))
      ctasOn := ctasOn.modify si (· + 1)
      earliest := earliest.modify si fun e => some (min start (e.getD start))
      clusterBars := clusterBars.modify (clusterOfId id) fun cb => { cb with live := cb.live + nWarps }
    let busy := earliest.any (·.isSome) || answersFor.any (!·.isEmpty) || mem.l2Waiting ||
      !releases.isEmpty
    if doneCtas == ctaCount && !busy then break
    if l.untilCycle.any (t ≥ ·) then
      stopped := some t
      break
    let stop := t + quantum
    -- Each SM with work this quantum reserves as many waiting CTAs as it has resident (each may
    -- complete), round-robin over the SMs, at most as many as the dispatcher hands out in a
    -- quantum (`ctaDispatchCycles`' replacement rate); what is unused comes back. A cluster's CTAs
    -- start together, and the master counts each cluster's warps for its barrier, so a launch with
    -- clusters or cluster barriers is dispatched only between quanta.
    let mut res : Array (Array Nat) := Array.replicate nSms #[]
    let mut budget := if clustered then 0 else max 1 (quantum / max 1 p.launch.ctaDispatchCycles.value.2)
    for round in [0:ctasPerSM] do
      for i in [0:nSms] do
        if resident[i]! > round && earliest[i]!.any (· < stop) && budget > 0 then
          if let some id := returned[0]? then
            budget := budget - 1
            res := res.modify i (·.push id); returned := returned.extract 1 returned.size
          else if nextCta < ctaCount then
            budget := budget - 1
            res := res.modify i (·.push nextCta); nextCta := nextCta + 1
    let cmds := (Array.range nW).map fun w =>
      ({ t, stop, global := mem.global, dir := mem.dir, start := mem.start,
         answers := answersFor[w]!, placements := placeFor[w]!, releases, atomAnswers := atomsFor[w]!,
         reserved := ((Array.range nSms).filter (wOf · == w)).filterMap fun i =>
           if res[i]!.isEmpty then none else some (i, res[i]!) } : Cmd)
    answersFor := Array.replicate nW #[]
    placeFor := Array.replicate nW #[]
    atomsFor := Array.replicate nW #[]
    releases := #[]
    -- Functionally one worker takes the global store itself: the master lets go of it.
    if l.functional && nW == 1 then mem := mem.atQuantum (Store.empty 0 0) mem.dir mem.start
    let (ps', reps) ← pool.round ps cmds
    ps := ps'
    -- The SMs' reports, in SM order.
    let mut reports : Array SmReport := Array.replicate nSms default
    for r in reps do
      match r with
      | .error e => return ← abort ps e
      | .ok rep =>
        for x in rep.sms do reports := reports.set! x.sm x
        if let some g := rep.global then mem := mem.atQuantum g mem.dir mem.start
        -- Over several workers, each one's global writes, in worker order.
        for (cta, r) in rep.globalLog do
          match mem.performNow 0 cta 0 r with
          | .ok (m', _) => mem := m'
          | .error e => return ← abort ps e
    -- Then the quantum's global atomics, in SM order; their warps go on in the next quantum.
    for x in reports do
      for (slot, cta, r) in x.atoms do
        match mem.performNow x.sm cta slot r with
        | .ok (m', resp) =>
          mem := m'
          atomsFor := atomsFor.modify (wOf x.sm) (·.push (x.sm, slot, resp))
        | .error e => return ← abort ps e
    let mut msgs := #[]
    let mut logs := #[]
    let mut touched : Array Nat := #[]
    for x in reports do
      resident := resident.set! x.sm x.resident
      earliest := earliest.set! x.sm x.earliest
      doneCtas := doneCtas + x.doneCtas
      returned := returned ++ x.returned
      for (psm, pci, cl, rk, bytes) in x.placed do
        mem := mem.addCtaL2 psm pci cl rk bytes (reservedInit p (cx * cy * cz) rk)
        ctasOn := ctasOn.modify psm (max (pci + 1))
      msgs := msgs ++ x.msgs
      logs := logs ++ x.logs
      for (cid, at_, arrive) in x.clusterLog do
        clusterBars := clusterBars.modify cid fun cb =>
          if arrive then { cb with arrived := cb.arrived + 1, last := max cb.last at_ }
          else { cb with live := cb.live - 1, last := max cb.last at_ }
        touched := touched.push cid
    -- A cluster whose live warps have all arrived completes its phase: its waiters go on
    -- `clusterBarrierCycles` after the last arrival or exit, no earlier than the next quantum
    -- (the cost exceeds the lookahead on every part with clusters, so never later than serially).
    for cid in touched do
      let cb := clusterBars[cid]!
      if cb.arrived > 0 && cb.arrived ≥ cb.live then
        let release := max (cb.last + p.sync.clusterBarrierCycles.value) stop
        clusterBars := clusterBars.set! cid { cb with arrived := 0, phase := cb.phase + 1 }
        releases := releases.push (cid, cb.phase + 1, release)
    match mem.endQuantumWith msgs logs stop with
    | .error e => return ← abort ps e
    | .ok (m', answers) =>
      mem := m'
      for a in answers do
        answersFor := answersFor.modify (wOf a.sm) (·.push a)
        earliest := earliest.modify a.sm fun e => some (min a.stamp (e.getD a.stamp))
    t := stop
    -- Nothing before a later quantum: skip to the quantum it starts in.
    let next := (earliest ++ releases.map (some ·.2.2)).foldl (fun acc x => match acc, x with
      | some a, some b => some (min a b) | none, x => x | a, none => a) none
    if !mem.l2Waiting && doneCtas < ctaCount then
      match next with
      | none =>
        if nextCta ≥ ctaCount && returned.isEmpty then
          return ← abort ps s!"deadlock at cycle {t}: no warp can issue and nothing is in flight"
      | some e =>
        -- Not past a stop (`Launch.untilCycle`): the launch ends at its quantum.
        let e := match l.untilCycle with
          | some u => min e (max u (t + quantum))
          | none => e
        if e > t then
          t := t + (e - t) / quantum * quantum
          mem := { mem with start := t }
  -- The workers hand back their SMs.
  let mut sms : Array Sm := Array.replicate nSms default
  let mut runs : Array St := #[]
  let (_, reps) ← pool.round ps (Array.replicate nW { (default : Cmd) with finish := true })
  for r in reps do
    match r with
    | .error e => return .error e
    | .ok rep =>
      for (si, sm, smm, st) in rep.final do
        sms := sms.set! si sm
        mem := { mem with sms := mem.sms.set! si smm }
        runs := runs.push st
  let mut c : Counters := {}
  let mut printfOut : Array PrintfRecord := #[]
  let mut hazards : Array String := #[]
  let mut traces : Std.HashMap Nat WarpTrace := {}
  for r in runs do
    c := c.merge r.c
    printfOut := printfOut ++ r.printfOut
    hazards := hazards ++ r.hazards
    traces := r.traces.fold (fun acc key wt => acc.insert key wt) traces
  printfOut := printfOut.qsort fun a b => a.cycle < b.cycle || (a.cycle == b.cycle && a.sm < b.sm)
  hazards := hazards.extract 0 8
  let toCycles (ps : Nat) : Nat := ps * p.clockMHz.value / 1000000
  -- The launch ends once the L2 side has taken in what its warps sent: stores the SMs let go of
  -- (a bulk store waited for by its reads) still queue at the slices.
  c := { c with cycles := max c.cycles (mem.l2BusyUntil + 1) }
  -- A launch stopped early ran until then: what was still queued is not counted.
  if let some at_ := stopped then c := { c with cycles := at_ }
  c := { c with startCycles := toCycles (p.launch.kernelGapPs.value - p.launch.kernelGapPdlPs.value),
                endCycles := toCycles p.launch.kernelGapPdlPs.value, memory := mem.counters,
                ctasPerSM, tmaBytes := runs.foldl (· + ·.asyncU.copyBytes) 0,
                gmmaInsts := runs.foldl (· + ·.asyncU.mmas) 0
                tensorBusyMax := runs.foldl (fun m r => r.asyncU.tensorBusy.fold (fun m _ v => max m v) m) 0
                pipeBusyMax := sms.foldl (fun acc sm =>
                  (List.range sm.pipeWork.size).foldl (fun acc i =>
                    acc.modify (i % Pipe.count) (max · sm.pipeWork[i]!)) acc) (Array.replicate Pipe.count 0)
                issueMax := sms.foldl (fun m sm => sm.issued.foldl max m) 0 }
  let machine := { machine with code := sms.map (·.code.map ({ · with fill := {} })), l2 := mem.l2Tags,
                                clock := machine.clock + c.startCycles + c.cycles + c.endCycles,
                                modules := if firstLoad then k.module :: machine.modules else machine.modules }
  -- An access's thread: a warp's, or an async operation's, named by its issuing warp and pc.
  let who (si slot : Nat) : String :=
    if slot < Memory.RaceCheck.agentSlot 0 then s!"warp slot {slot}" else
    let a := slot - Memory.RaceCheck.agentSlot 0
    match runs.findSome? fun r => (r.asyncU.agents[a]?).filter (·.origin.sm == si) with
    | some g => s!"the async operation of warp slot {g.origin.slot} from {(k.code[g.pc]?.map fun i => s!"pc {i.pc} {i.full}").getD s!"instruction {g.pc}"}"
    | none => s!"async operation {a}"
  let races := (Array.range nSms).flatMap fun si => (mem.races si).map fun r =>
    s!"{r.kind} on {if r.shared then "shared" else "global"} byte {hexAddr r.addr} at cycle {r.at_}, SM {si}, " ++
    s!"CTA index {r.cta}: {who si r.first.1} lane {r.first.2}, then {who si r.second.1} lane {r.second.2}"
  -- Stores of two CTAs to one global byte: each SM's, then across SMs (`RaceCheck.CrossWrite`).
  let races := Id.run do
    if !l.races then return races
    let mut seen : Std.HashMap Nat (Nat × Nat × Nat × UInt8) := {}
    let mut count := 0
    let mut differ := 0
    let mut examples : Array Memory.CrossWrite := #[]
    for si in [0:nSms] do
      let rc := mem.sms[si]!.race
      count := count + rc.crossCount
      differ := differ + rc.crossDiffer
      examples := examples ++ rc.crossWrites.map fun x =>
        { x with first := (si, x.first.2), second := (si, x.second.2) }
      for (addr, w) in rc.global do
        if let some (sj, c, g, v) := seen.get? addr then
          count := count + 1
          if v != w.value then differ := differ + 1
          if examples.size < Memory.RaceCheck.cap then
            examples := examples.push { addr, first := (sj, c, g), second := (si, w.cta, w.gen), differ := v != w.value }
        seen := seen.insert addr (si, w.cta, w.gen, w.value)
    -- Bytes one CTA read and another stored: each SM's, then a read against another SM's store.
    let mut reads := 0
    let mut readEx : Array Memory.CrossWrite := #[]
    for si in [0:nSms] do
      let rc := mem.sms[si]!.race
      reads := reads + rc.crossReadCount
      readEx := readEx ++ rc.crossReads.map fun x => { x with first := (si, x.first.2), second := (si, x.second.2) }
      for (addr, (c, g)) in rc.globalReads do
        if let some (sj, c', g', _) := seen.get? addr then
          if sj != si then
            reads := reads + 1
            if readEx.size < Memory.RaceCheck.cap then
              readEx := readEx.push { addr, first := (sj, c', g'), second := (si, c, g), differ := false }
    -- A CTA: its SM, its index there and which of that index's CTAs it was (from 0).
    let cta (c : Nat × Nat × Nat) : String := s!"SM {c.1} CTA index {c.2.1} #{c.2.2}"
    let mut races := races
    if count > 0 then
      let ex := examples.toList.take 4 |>.map fun x =>
        s!"{hexAddr x.addr} by {cta x.first} and {cta x.second}{if x.differ then ", values differ" else ""}"
      races := races.push (s!"stores of two CTAs to one global byte: {count} bytes, {differ} with different values; " ++
        String.intercalate "; " ex)
    if reads > 0 then
      let ex := readEx.toList.take 4 |>.map fun x => s!"{hexAddr x.addr} by {cta x.first} and {cta x.second}"
      races := races.push (s!"a global byte one CTA read and another stored, in either order: {reads} bytes; " ++
        String.intercalate "; " ex)
    -- Floating-point atomic additions to one address by several warps: each SM's, then across SMs.
    let mut adders : Std.HashMap Nat Nat := {}
    let mut adds := 0
    let mut addEx : Array Nat := #[]
    for si in [0:nSms] do
      let rc := mem.sms[si]!.race
      adds := adds + rc.floatAddCount
      addEx := addEx ++ rc.floatAddClashes
      for (addr, _) in rc.floatAdds do
        if adders.contains addr && !rc.floatAddClashes.contains addr then
          adds := adds + 1
          if addEx.size < Memory.RaceCheck.cap then addEx := addEx.push addr
        adders := adders.insert addr si
    if adds > 0 then
      races := races.push (s!"floating-point atomic additions to one address from several warps, whose sum " ++
        s!"depends on their order: {adds} addresses; {String.intercalate ", " ((addEx.toList.take 4).map hexAddr)}")
    return races
  return .ok (machine, { global := mem.global, counters := c, printf := printfOut, hazards, races,
                         traces := (traces.toArray.qsort (·.1 < ·.1)).map (·.2) })

/-- Run the launch on a machine in a given state, to completion: the machine after it, and the
launch's result. SMs run their quanta one after another (`runOnIO` runs them in parallel; the
results are the same). -/
def runOn (p : Params) (machine : Machine) (l : Launch) : Except String (Machine × Result) :=
  Id.run (runOnWith inlinePool p machine l)

/-- `runOn`, with the SMs' quanta on worker threads. -/
def runOnIO (p : Params) (machine : Machine) (l : Launch) : IO (Except String (Machine × Result)) :=
  runOnWith threadPool p machine l

/-- Run the launch on a fresh machine. -/
def run (p : Params) (l : Launch) : Except String Result :=
  (runOn p (Machine.fresh p) l).map (·.2)

/-- `run`, with the SMs' quanta on worker threads. -/
def runIO (p : Params) (l : Launch) : IO (Except String Result) :=
  return (← runOnIO p (Machine.fresh p) l).map (·.2)

end NV.Core
