import NV.Core.Memory
import NV.Memory.Events
import NV.SASS.Warp

/-!
# The asynchronous units

The agents that run `Effect.async` operations (`NV.Async.Op`) beside the warps: the SM's TMA
unit (tensor and bulk copies), `LDGSTS`'s copy path, and the warpgroup MMA unit. An agent's data
move only through the memory system (`MemSys`); this module decides when each of its requests
is issued and what follows each delivery:

* A copy issues every move's load, then on each load's delivery the move's stores (the loaded
  words, `Move.store`), and once every store has delivered, its signals (complete-tx or arrive-on
  RMWs) in order: the agent's program `Copy.program`, with the implicit proxy fence satisfied
  because every move has delivered. A copy marked `afterGroup` starts when the issuer's earlier
  operations of its group have completed. Tensor and bulk copies issue their loads through the
  SM's TMA port at `TmaParams.issueBytesPerSec` (`Port.take`, so `Rate.served_le` bounds the
  bytes the unit generates); `LDGSTS`'s loads take the warp's normal LSU path, as the instruction
  does, and its shared stores go to the data port.
* A warpgroup MMA (one warp's share) occupies its sub-partition's GMMA port for
  `macs / (fmaPerCycle / smsps)` cycles (the documented peak, `TensorParams.fmaPerCycle*`) and
  completes `wgmmaLatency(N)` (or `wgmmaLatencyRS`) after it starts, or when its shared loads
  have delivered if that is later, and never before the warp's previous MMA; its result is
  written then. A wait (`wgmma.wait_group`) sees its group done once the matching groups of the
  warpgroup's four warps all are (`groupDone`): the MMA is one operation. The result is computed on a `Task` as soon as the operands are in (from the
  result of the latest earlier MMA writing the registers it accumulates from, when it issued
  before that result reached them, else from the accumulator it captured), and forced only by
  the write, so the tensor arithmetic of all warps runs in
  parallel; it is a pure function of its inputs, so runs stay deterministic.

Groups: operations join their issuer's open group of their kind until a commit closes it. A
closed group completes when its last operation does. The engine learns of completions through
`Out`: a dependence counter entry to release (`LDGDEPBAR`, `UTMACMDFLUSH`) or a register result
to write.
-/

namespace NV.Core

open NV NV.Async NV.SASS

namespace AsyncUnit

/-- What a memory request of an agent is for. -/
inductive Role where
  | load (move : Nat)
  | store
  | signal (i : Nat)
  | desc
  | mmaLoad (i : Nat)
  | mmaStore
  deriving Inhabited, Repr

inductive Body where
  | copy (c : Copy)
  | tensor (t : TensorCopy)
  | mma (m : Mma)
  | mmaT (m : MmaT)
  deriving Inhabited

/-- A finished operation's body, as later operations still read it: its kind, a warpgroup MMA's
warp in its group. Its requests, the accumulator an MMA captured at issue, and the closures that
computed and wrote its result (which hold what it read: a copy's moves) are let go. -/
def Body.spent : Body → Body
  | .copy c => .copy { c with moves := #[], signals := #[] }
  | .tensor t => .tensor { t with build := fun _ => .error "spent" }
  | .mma m => .mma { m with loads := #[], c0 := #[], result := fun _ a => a, fast := fun _ _ => none }
  | .mmaT m => .mmaT { m with loads := #[], result := fun _ => #[], stores := fun _ => #[] }

structure Agent where
  origin : Origin
  body : Body
  wb : Resp → Warp → Warp
  /-- The issuing instruction's index, so the engine can retire its pending writes. -/
  pc : Nat := 0
  group : Option Group
  /-- Requests in flight. -/
  inflight : Nat := 0
  /-- Signals not yet issued (a copy's, after its moves). -/
  signalsLeft : Bool := true
  /-- The copy's next signal to issue. -/
  nextSignal : Nat := 0
  /-- Earlier operations it waits for (`afterGroup`), by agent index. -/
  deps : Array Nat := #[]
  started : Bool := false
  /-- An MMA's responses, by load, and the earliest cycle its result exists. -/
  resps : Array Resp := #[]
  readyAt : Nat := 0
  /-- The committed group it belongs to. -/
  groupId : Option Nat := none
  /-- An MMA's predecessor among its issuer's MMAs: it completes first. -/
  prev : Option Nat := none
  /-- For an MMA: the earlier MMA whose result it accumulates from, if any. -/
  source : Option Nat := none
  /-- Whether the MMA starts from its predecessor's result rather than from the accumulator it
  captured at issue: the predecessor had not completed when it issued. -/
  chained : Bool := false
  /-- An MMA's result, computed on a task once its operands are in, so the multiply-adds of many
  MMAs run in parallel with the engine; it is forced only when the result is written. -/
  task : Option (Task (Array UInt32)) := none
  /-- A later MMA chained onto this one and has yet to take its result. -/
  held : Bool := false
  doneAt : Option Nat := none
  /-- A tensor-memory MMA: whether it has computed its result and sent its stores. -/
  stored : Bool := false
  /-- Source loads still to deliver, and when the last did: the operation has read its sources
  (`cp.async.bulk.wait_group.read`, PTX ISA §9.7.9.25.5.2), though its writes may still be on their
  way. -/
  loadsLeft : Nat := 0
  readAt : Option Nat := none
  /-- The latest its stores' data left the SM: its sources are read no earlier. -/
  readBy : Nat := 0
  /-- Its SM's writes (`State.storeLog`) up to these cumulative bytes must be answered before its
  sources count as read (`TmaParams.storeCreditBytes`). -/
  creditTo : Nat := 0
  /-- For the issuer's trace (`Launch.recordTrace`): its issue's index there and its writeback's
  id; the responses of a copy's moves and signals and of a tensor copy's descriptor; and, for an
  MMA a later one chained onto, where its result lands early in the trace. -/
  tix : Option Nat := none
  land : Nat := 0
  moveResps : Array Resp := #[]
  sigResps : Array Resp := #[]
  desc : Option (Req × Resp) := none
  early : Option Nat := none
  deriving Inhabited

structure GroupRec where
  sm : Nat
  slot : Nat
  kind : Group
  /-- The dependence counter whose pending entry the group's completion releases, or, when
  `onRead`, the group's operations having read their sources (SASS: the commit's read counter). -/
  bar : Option (Fin 6)
  onRead : Bool := false
  remaining : Nat
  /-- Operations still to read their sources, and when the latest of the others did. -/
  unread : Nat := 0
  readLatest : Nat := 0
  doneAt : Option Nat := none
  /-- Its operations (agent indices). -/
  ops : Array Nat := #[]
  /-- The issuer's previous group of this kind, unreleased when this one committed, and the next:
  groups release in commit order (`cp.async.wait_group N`, `cp.async.bulk.wait_group[.read] N`:
  "only N or fewer of the most recent … groups are pending and all the prior … groups committed by
  the executing threads are complete", PTX ISA 9.5; `wgmma.wait_group` alike), so an empty or
  finished group waits for the ones before it. -/
  prev : Option Nat := none
  next : Option Nat := none
  /-- Its release done (its dependence counter's entry, or a warpgroup MMA group's completion). -/
  released : Bool := false
  /-- How many groups of this kind the issuer committed before it. -/
  seq : Nat := 0
  deriving Inhabited

/-- What the engine must do. -/
inductive Out where
  /-- Write a register result into warp `slot` of SM `sm` at `at_`; `land`, its id in the warp's
  trace. -/
  | wb (sm slot at_ : Nat) (f : Warp → Warp) (pc : Nat := 0) (land : Option Nat := none)
  /-- Release one pending entry of dependence counter `bar` at `at_`. -/
  | release (sm slot : Nat) (bar : Fin 6) (at_ : Nat)
  /-- Something warp `slot` may wait on changed at `at_` (a wgmma group completed). -/
  | wake (sm slot at_ : Nat)
  /-- For the issuer's trace: the operation issued at `pc` (event `tix`) read `reads` and ended
  with `result`; its writeback is pending under `land`. -/
  | trace (sm slot tix pc land : Nat) (reads : Array (Array ByteArray)) (result : Except String Resp)
  /-- For the issuer's trace: the writeback `f` of the instruction at `pc`, pending under `land`,
  lands at event `ix`, before an MMA that chained onto it. -/
  | earlyLand (sm slot ix pc land : Nat) (f : Warp → Warp)
  deriving Inhabited

/-- What each lane's reads returned, from a request's response: a returning lane's bytes; a lane
whose reads the issuer does not see (a reduction) zeros of the reads' length. -/
def _root_.NV.Core.readBytes (r : Req) (resp : Resp) : Array ByteArray :=
  let n := words r.bytes
  (Array.range r.addrs.size).map fun l =>
    if !r.active l then .empty
    else if r.op.returns then ((toBytes (resp.extract (l * n) (l * n + n)).toList).take r.bytes).toByteArray
    else (List.replicate r.readLen 0).toByteArray

def _root_.NV.Async.Group.index : Group → Nat
  | .cpAsync => 0 | .bulk => 1 | .wgmma => 2 | .tcgen05 => 3

/-- Points `(x, y)` sorted by `x`, read piecewise linearly (constant beyond the ends). -/
def curve : List (Nat × Nat) → Nat → Nat
  | [], _ => 0
  | [(_, y)], _ => y
  | (x0, y0) :: (x1, y1) :: rest, x =>
    if x ≤ x0 then y0
    else if x ≤ x1 then
      let d := max (x1 - x0) 1
      if y1 ≥ y0 then y0 + (y1 - y0) * (x - x0) / d else y0 - (y0 - y1) * (x - x0) / d
    else curve ((x1, y1) :: rest) x

structure State where
  agents : Array Agent := #[]
  groups : Array GroupRec := #[]
  /-- Memory request id ↦ agent and role. -/
  reqs : Std.HashMap Nat (Nat × Role) := {}
  /-- `(sm, slot, group kind)` ↦ agents not yet in a committed group. -/
  open_ : Std.HashMap (Nat × Nat × Nat) (Array Nat) := {}
  /-- `(sm, slot)` ↦ committed wgmma groups not yet complete. -/
  wgmma : Std.HashMap (Nat × Nat) (Array Nat) := {}
  /-- Agents waiting for earlier ones. -/
  waiting : Array Nat := #[]
  /-- `(sm, slot)` ↦ the warp's latest MMA. -/
  lastMma : Std.HashMap (Nat × Nat) Nat := {}
  /-- `(sm, slot)` ↦ the warp's latest tensor-memory MMA, and per D address the latest on it. -/
  lastMmaT : Std.HashMap (Nat × Nat) Nat := {}
  lastMmaTByD : Std.HashMap (Nat × Nat × Nat) Nat := {}
  /-- Per issuer and first destination register: its latest MMA writing there. -/
  lastByDst : Std.HashMap (Nat × Nat × Nat) Nat := {}
  /-- Per agent: how many entries of `lastByDst` and `lastMmaTByD` name it (absent: none), so
  whether a later MMA may start from its result is one lookup. -/
  latestOf : Std.HashMap Nat Nat := {}
  /-- `(sm, slot, group kind)` ↦ the issuer's operations of that kind, in issue order, among them
  every one not yet done (`earlier`): those done are dropped as they are read. -/
  byGroup : Std.HashMap (Nat × Nat × Nat) (Array Nat) := {}
  /-- Per SM, the TMA unit's request generation. -/
  tma : Array Port
  /-- Per sub-partition, the GMMA pipe. -/
  gmma : Array Port
  /-- Bytes of copies issued, MMA instructions, for the counters. -/
  copyBytes : Nat := 0
  mmas : Nat := 0
  /-- Per tensor-core port (an SM sub-partition's for wgmma, sub-partition 0's for tcgen05): the
  cycles its multiply-adds held it. -/
  tensorBusy : Std.HashMap Nat Nat := {}
  /-- `(sm, slot, group kind)` ↦ the issuer's latest committed group, and how many it committed. -/
  lastGroup : Std.HashMap (Nat × Nat × Nat) Nat := {}
  /-- Per SM, the tensor-map addresses its copies have fetched this launch: the copy engine's
  descriptor cache (inferred), modeled as unbounded with no hit cost. A hit still takes the map's
  current bytes, so a map rewritten on the device reads as written. -/
  descCached : Std.HashSet (Nat × UInt64) := {}
  /-- Per SM, its async-unit writes to global memory in the order sent: the cumulative bytes after
  each and when the L2 answered it; and the entries before `storeHead` dropped. -/
  storeLog : Std.HashMap Nat (Array (Nat × Option Nat)) := {}
  storeHead : Std.HashMap Nat Nat := {}
  storeSent : Std.HashMap Nat Nat := {}
  /-- Per SM, the latest answer of the entries dropped from `storeLog`'s front. -/
  storeFloor : Std.HashMap Nat Nat := {}
  /-- A write request in flight ↦ its SM and its entry in `storeLog`. -/
  storeOf : Std.HashMap Nat (Nat × Nat) := {}
  /-- Per SM, agents whose sources are read but whose credit is not yet in. -/
  creditWait : Std.HashMap Nat (Array (Nat × Nat)) := {}
  groupCount : Std.HashMap (Nat × Nat × Nat) Nat := {}
  /-- `(sm, slot, group kind, seq)` ↦ the group (`GroupRec.seq`). -/
  bySeq : Std.HashMap (Nat × Nat × Nat × Nat) Nat := {}
  deriving Inhabited

def init (sms smsps : Nat) : State :=
  { tma := Array.replicate sms {}, gmma := Array.replicate (sms * smsps) {} }

def owns (st : State) (id : Nat) : Bool := st.reqs.contains id

/-- Cycles the GMMA port is busy for an MMA: its multiply-adds at the sub-partition's share of
the SM's peak for its input width. -/
def mmaInterval (p : Params) (m : Mma) : Nat :=
  let perSM := match m.inBits with
    | 16 => p.tensor.fmaPerCycle16.value
    | 32 => p.tensor.fmaPerCycleTF32.value
    | _ => p.tensor.fmaPerCycle8.value
  let perSmsp := max (perSM / max p.smsps.value 1) 1
  (m.macs + perSmsp - 1) / perSmsp

def mmaLatency (p : Params) (m : Mma) : Nat :=
  curve (if m.aRegs then p.tensor.wgmmaLatencyRS.value else p.tensor.wgmmaLatency.value) m.n

/-- Issue request `r` of agent `a` for `role` at `t`: through the LSU (`viaLsu`) or the async
path. -/
def send (st : State) (mem : MemSys) (t : Nat) (a : Nat) (role : Role) (r : Req)
    (viaLsu : Bool := false) (credit : Nat := 0) (tensor : Bool := false) (tensorShares : Nat := 1)
    (wgmma : Bool := false) :
    Except String (State × MemSys) := do
  let o := st.agents[a]!.origin
  let (mem, id) ← if viaLsu then pure (mem.issue t o r) else mem.issueAsync t o r a tensor tensorShares wgmma
  -- The bulk engine reads its source only as fast as it sends it on: a source is read once the
  -- data stored from it has left the SM (`readDone`), and once the L2 has answered the SM's writes
  -- sent `credit` bytes before it.
  let write := r.space == .global && !r.op.returns && !viaLsu
  let depart := if write then mem.departAt o.sm id else 0
  let mut st := { st with reqs := st.reqs.insert id (a, role),
                          agents := st.agents.modify a fun g =>
                            { g with inflight := g.inflight + 1, readBy := max g.readBy depart } }
  if write && credit > 0 then
    let bytes := (r.addrs.filter (·.isSome)).size * r.bytes
    let sent := st.storeSent.getD o.sm 0 + bytes
    let log := st.storeLog.getD o.sm #[]
    st := { st with storeSent := st.storeSent.insert o.sm sent,
                    storeLog := st.storeLog.insert o.sm (log.push (sent, none)),
                    storeOf := st.storeOf.insert id (o.sm, log.size),
                    agents := st.agents.modify a fun g =>
                      { g with creditTo := max g.creditTo (sent - min sent credit) } }
  return (st, mem)

/-- A warpgroup MMA group's family: the groups at the same place in the commit sequences of its
warpgroup's four warps (itself included), which one `wgmma.mma_async.sync.aligned` per warp fills
together; `none` while one of them is not committed yet. Any other group is its own family. -/
def family (st : State) (gi : Nat) : Option (Array Nat) :=
  let gr := st.groups[gi]!
  let w? : Option Nat := gr.ops[0]?.bind fun o => match st.agents[o]!.body with
    | .mma m => m.warpInGroup
    | _ => none
  match w? with
  | some w => if w > gr.slot then some #[gi] else
    (Array.range 4).mapM fun j =>
      if j == w then some gi else st.bySeq.get? (gr.sm, gr.slot - w + j, gr.kind.index, gr.seq)
  | none => some #[gi]

/-- Group `gi`'s operations are done: all of them have completed, and its family's too (the
warpgroup's MMA is one operation, which `wgmma.wait_group` in any of its warps waits for whole). -/
def groupDone (st : State) (gi : Nat) : Bool :=
  st.groups[gi]!.remaining == 0 &&
    ((family st gi).map (·.all (st.groups[·]!.remaining == 0))).getD false

/-- Group `gi` releases at `t` if its operations have read their sources (`onRead`) or are done
(`groupDone`) and the group before it has released, and so on down the groups committed after it. -/
def releaseFrom (st : State) (gi t : Nat) (outs : Array Out) : State × Array Out := Id.run do
  let mut st := st
  let mut outs := outs
  let mut cur := some gi
  for _ in [0:st.groups.size] do
    let some g := cur | break
    let gr := st.groups[g]!
    let met := if gr.onRead then gr.unread == 0 else groupDone st g
    if gr.released || !met || !gr.prev.all (st.groups[·]!.released) then break
    let gr' := { gr with released := true, doneAt := if gr.onRead then gr.doneAt else some t }
    st := { st with groups := st.groups.set! g gr' }
    if let some b := gr.bar then outs := outs.push (.release gr.sm gr.slot b (if gr.onRead then max t gr.readLatest else t))
    if gr.kind == .wgmma then outs := outs.push (.wake gr.sm gr.slot t)
    cur := gr.next
  return (st, outs)

/-- When the writes of SM `sm` up to cumulative bytes `upTo` were all answered, if they were. -/
def creditAt (st : State) (sm upTo : Nat) : Option Nat := Id.run do
  let log := st.storeLog.getD sm #[]
  let mut at_ := st.storeFloor.getD sm 0
  for i in [st.storeHead.getD sm 0:log.size] do
    let (cum, done) := log[i]!
    if cum > upTo then break
    match done with
    | some d => at_ := max at_ d
    | none => return none
  return some at_

/-- Agent `a` has read its sources at `t`: its group's read release, once all its operations have,
and once its SM's writes `storeCreditBytes` before its own are answered (`creditTo`). -/
def readDone (st : State) (t a : Nat) (outs : Array Out) : State × Array Out := Id.run do
  if st.agents[a]!.readAt.isSome then return (st, outs)
  let g := st.agents[a]!
  let t := max t g.readBy
  let t ← if g.creditTo == 0 then pure t else
    match creditAt st g.origin.sm g.creditTo with
    | some c => pure (max t c)
    | none =>
      let w := st.creditWait.getD g.origin.sm #[]
      if !w.any (·.1 == a) then
        return ({ st with creditWait := st.creditWait.insert g.origin.sm (w.push (a, t)) }, outs)
      return (st, outs)
  let mut st := { st with agents := st.agents.modify a ({ · with readAt := some t }) }
  let mut outs := outs
  if let some gi := st.agents[a]!.groupId then
    let gr := st.groups[gi]!
    let gr := { gr with unread := gr.unread - 1, readLatest := max gr.readLatest t }
    st := { st with groups := st.groups.set! gi gr }
    if gr.onRead then (st, outs) := releaseFrom st gi t outs
  return (st, outs)

/-- A done operation no later one starts from any more: its result and the writeback that lands
it are let go (each holds what the operation computed; a run keeps every agent, by index, and an
accumulating chain of MMAs kept every link's until the launch ended). -/
def Agent.letGo (p : Agent) : Agent :=
  if p.doneAt.isSome && !p.held then { p with task := none, wb := fun _ w => w } else p

/-- Agent `a` completed at `t`: its group's bookkeeping, and the waiting agents now free to
start. -/
def finish (st : State) (t a : Nat) (outs : Array Out) : State × Array Out × Array Nat := Id.run do
  let (st, outs) := readDone st t a outs
  -- Done, its data delivered: what it read and moved is let go (a run keeps every agent, by index,
  -- and their data grew with the run's work). An MMA's result, and the writeback that lands it,
  -- stay while a later MMA may start from it: the latest to its accumulator, or one a pending MMA
  -- chained onto.
  let source := st.agents[a]!.held || st.latestOf.contains a
  let mut st := { st with agents := st.agents.modify a fun g =>
    let spent := { g with doneAt := some t, resps := #[], moveResps := #[], sigResps := #[], desc := none,
                          task := if source then g.task else none, body := g.body.spent }
    -- A later MMA chained onto its result lands it early in the trace (`Engine`) by its `wb`.
    -- Chosen between whole records: `wb := if source then g.wb else …` compiles to a closure
    -- over `g` as it was, all of its data.
    if source then spent else { spent with wb := fun _ w => w } }
  let mut outs := outs
  if let some gi := st.agents[a]!.groupId then
    let gr := st.groups[gi]!
    let gr := { gr with remaining := gr.remaining - 1 }
    st := { st with groups := st.groups.set! gi gr }
    if gr.remaining == 0 then
      if gr.onRead then st := { st with groups := st.groups.set! gi { gr with doneAt := some t } }
      else
        -- The family's groups release together, once its last operation completes.
        for f in (family st gi).getD #[gi] do (st, outs) := releaseFrom st f t outs
  let ready := st.waiting.filter fun w => st.agents[w]!.deps.all (st.agents[·]!.doneAt.isSome)
  st := { st with waiting := st.waiting.filter (!ready.contains ·) }
  -- The next MMA of the warp, if its data are in, completes now too.
  let g := st.agents[a]!
  let next := match g.body with
    | .mma _ => match st.lastMma.get? (g.origin.sm, g.origin.slot) with
      | some l => ((List.range (l + 1 - a)).map (a + ·)).filter fun (j : Nat) =>
          (st.agents[j]!).prev == some a && (st.agents[j]!).started
      | none => []
    | .mmaT _ => match st.lastMmaT.get? (g.origin.sm, g.origin.slot) with
      | some l => ((List.range (l + 1 - a)).map (a + ·)).filter fun (j : Nat) =>
          (st.agents[j]!).prev == some a && (st.agents[j]!).started
      | none => []
    | _ => []
  return (st, outs, ready ++ next.toArray)

/-- The race check, once agent `a` has finished: if its group's operations have all completed, their
accesses (a copy's writes, an MMA's operand reads) are its issuer's (`cp.async.wait_group`,
`cp.async.bulk.wait_group`, `wgmma.wait_group`); a warpgroup MMA's family's, once all of it has
(`groupDone`), to each of its warps. -/
private def raceGroupDone (st : State) (mem : MemSys) (a : Nat) : MemSys :=
  match st.agents[a]!.groupId with
  | some gi =>
    if !groupDone st gi then mem else
    let fam := (family st gi).getD #[gi]
    let ops := fam.flatMap (st.groups[·]!.ops)
    fam.foldl (fun m f => m.raceAgentsDone st.groups[f]!.sm st.groups[f]!.slot ops) mem
  | none => mem

/-- After a delivery (or the start) of agent `a` at `t`: issue a copy's signals once its moves
are done, one after another (each a release after the copy's earlier requests, the previous
signal included: `Copy.program`); finish it once nothing is left. Returns the agents to start
next. -/
def progress (st : State) (mem : MemSys) (t a : Nat) :
    Except String (State × MemSys × Array Out × Array Nat) := do
  let g := st.agents[a]!
  if g.inflight > 0 || g.doneAt.isSome then return (st, mem, #[], #[])
  match g.body with
  | .copy c =>
    if g.signalsLeft && g.nextSignal < c.signals.size then
      let i := g.nextSignal
      let st := { st with agents := st.agents.modify a fun g =>
        { g with nextSignal := i + 1, signalsLeft := i + 1 < c.signals.size } }
      let (st, mem) ← send st mem t a (.signal i) c.signals[i]!
      return (st, mem, #[], #[])
    let tr := match g.tix with
      | some ix =>
        let reads := (g.desc.map fun (r, d) => #[readBytes r d]).getD #[] ++
          (c.moves.mapIdx fun i mv => readBytes mv.src (g.moveResps[i]?.getD #[])) ++
          c.signals.mapIdx fun i r => readBytes r (g.sigResps[i]?.getD #[])
        #[Out.trace g.origin.sm g.origin.slot ix g.pc g.land reads (.ok #[])]
      | none => #[]
    let (st, outs, ready) := finish st t a tr
    return (st, raceGroupDone st mem a, outs, ready)
  | .mma m =>
    -- In order: after the warp's previous MMA, whose result it may start from.
    let prevDone := g.prev.map (st.agents[·]!.doneAt)
    if prevDone matches some none then return (st, mem, #[], #[])
    let at_ := max (max t g.readyAt) ((prevDone.join).getD 0)
    let data := g.resps.foldl (· ++ ·) #[]
    let task := match g.chained, g.source.bind (st.agents[·]!.task) with
      | true, some c => c.bind fun c => Task.spawn fun _ => (m.fast data c).getD (m.result data c)
      | _, _ => Task.spawn fun _ => (m.fast data m.c0).getD (m.result data m.c0)
    let st := { st with agents := st.agents.modify a ({ · with task := some task }) }
    -- The predecessor's result is this task's to hold now; done, it lets it go.
    let st := match g.chained, g.source with
      | true, some src =>
        { st with agents := st.agents.modify src (fun p => Agent.letGo { p with held := false }) }
      | _, _ => st
    let o := g.origin
    let tr := match g.tix with
      | some ix =>
        let reads := m.loads.mapIdx fun i r => readBytes r (g.resps[i]?.getD #[])
        #[Out.trace o.sm o.slot ix g.pc g.land reads (.ok task.get)] ++
          (g.early.map fun e => Out.earlyLand o.sm o.slot e g.pc g.land (fun w => g.wb task.get w)).toArray
      | none => #[]
    let wbLand := if g.tix.isSome && g.early.isNone then some g.land else none
    let (st, outs, ready) := finish st at_ a
      (tr.push (.wb o.sm o.slot at_ (fun w => g.wb task.get w) g.pc wbLand))
    return (st, raceGroupDone st mem a, outs, ready)
  | .mmaT m =>
    -- Its result once its operands are in, its time has passed and the warp's previous MMA is
    -- done, then stored; done once the stores have delivered. One accumulating onto a D an MMA
    -- still in flight wrote when it issued takes that MMA's result for D (`chained`): the tensor
    -- core's accumulator, not tensor memory.
    if !g.stored then
      let prevDone := g.prev.map (st.agents[·]!.doneAt)
      if prevDone matches some none then return (st, mem, #[], #[])
      let at_ := max (max t g.readyAt) ((prevDone.join).getD 0)
      let data := g.resps.foldl (· ++ ·) #[]
      let data := match g.chained, g.source.bind (st.agents[·]!.task), m.accIn with
        | true, some tk, some (o, n) =>
          let d := tk.get
          (Array.range data.size).map fun i => if o ≤ i && i < o + n then d[i - o]?.getD 0 else data[i]!
        | _, _, _ => data
      let res := m.result data
      let rs := m.stores res
      -- The predecessor's accumulator has been consumed, as for a register MMA above.
      let st := match g.chained, g.source with
        | true, some src =>
          { st with agents := st.agents.modify src (fun p => Agent.letGo { p with held := false }) }
        | _, _ => st
      let mut st := { st with agents := st.agents.modify a ({ · with stored := true, task := some (Task.pure res) }) }
      let mut mem := mem
      for r in rs do
        (st, mem) ← send st mem at_ a .mmaStore r
      if rs.isEmpty then
        let (st', outs, ready) := finish st at_ a #[]
        return (st', mem, outs, ready)
      return (st, mem, #[], #[])
    let (st, outs, ready) := finish st t a #[]
    return (st, mem, outs, ready)
  | .tensor _ => return (st, mem, #[], #[])

/-- Start agent `a`'s work at `t`. Returns the agents to start next (when it finished at once). -/
def startOne (p : Params) (st : State) (mem : MemSys) (t a : Nat) :
    Except String (State × MemSys × Array Out × Array Nat) := do
  -- An agent already started is resumed: its predecessor finished.
  if st.agents[a]!.started then return ← progress st mem t a
  let mut st := { st with agents := st.agents.modify a ({ · with started := true }) }
  let mut mem := mem
  -- A tensor copy whose map the SM's descriptor cache holds (`State.descCached`): the map's bytes
  -- now, and the copy starts at once.
  if let .tensor tc := st.agents[a]!.body then
    let addr := (tc.desc.addrs[0]?.join).getD 0
    let gm := mem.global
    if st.descCached.contains (st.agents[a]!.origin.sm, addr) && gm.contains addr 128 then
      let off := addr.toNat - gm.base.toNat
      let words := (Array.range 32).map fun w => gm.readWord (off + 4 * w)
      let c ← tc.build words
      st := { st with agents := st.agents.modify a ({ · with body := .copy c, desc := some (tc.desc, words) }) }
  let g := st.agents[a]!
  let o := g.origin
  match g.body with
  | .copy c =>
    let lsu := c.group == some .cpAsync
    let rate := Rate.perSecond p.tma.issueBytesPerSec.value p.clockMHz.value
    let loads := (c.moves.filter fun mv => mv.imm.isEmpty && mv.src.addrs.any (·.isSome)).size
    st := { st with agents := st.agents.modify a ({ · with loadsLeft := loads }) }
    let mut outs : Array Out := #[]
    if loads == 0 then (st, outs) := readDone st t a outs
    for i in [0:c.moves.size] do
      let mv := c.moves[i]!
      let r := mv.src
      let mut at_ := t
      if !lsu then
        let bytes := r.bytes * (r.addrs.filter (·.isSome)).size
        let (s, _, port) := st.tma[o.sm]!.take rate t bytes
        st := { st with tma := st.tma.set! o.sm port, copyBytes := st.copyBytes + bytes }
        at_ := s
      if !mv.imm.isEmpty then
        -- Register values: straight to the destinations, as a load's response would.
        for d in mv.dsts do
          (st, mem) ← send st mem at_ a .store (mv.store d mv.imm)
      else if r.addrs.any (·.isSome) then
        (st, mem) ← send st mem at_ a (.load i) r (viaLsu := lsu)
      else
        -- Wholly out of bounds: the fill goes straight to the destinations.
        for d in mv.dsts do
          (st, mem) ← send st mem at_ a .store (mv.store d #[])
    let (st', mem', o, ready) ← progress st mem t a
    return (st', mem', outs ++ o, ready)
  | .tensor tc =>
    (st, mem) ← send st mem t a .desc tc.desc
    return (st, mem, #[], #[])
  | .mma m =>
    let q := o.sm * p.smsps.value + o.smsp
    let (s, port) := st.gmma[q]!.reserve t (mmaInterval p m)
    st := { st with gmma := st.gmma.set! q port, mmas := st.mmas + 1,
                    tensorBusy := st.tensorBusy.insert q (st.tensorBusy.getD q 0 + mmaInterval p m),
                    agents := st.agents.modify a ({ · with readyAt := s + mmaLatency p m,
                                                           resps := Array.replicate m.loads.size #[],
                                                           loadsLeft := m.loads.size }) }
    -- A and B from shared memory through the tensor core's port (`SmRes.tensor`), B once a
    -- warpgroup: a quarter of its port time to each share (`Mma.bFrom`).
    for i in [0:m.loads.size] do
      (st, mem) ← send st mem s a (.mmaLoad i) m.loads[i]! (tensor := m.loads[i]!.space == .shared)
        (tensorShares := if m.bFrom.any (· ≤ i) then 4 else 1) (wgmma := true)
    progress st mem t a
  | .mmaT m =>
    -- The SM's tensor core (sub-partition 0's port): its multiply-adds at the SM's peak for its
    -- input width, its result `tcgen05Latency` after it starts.
    let q := o.sm * p.smsps.value
    let perSM := match m.inBits with
      | 16 => p.tensor.fmaPerCycle16.value
      | 32 => p.tensor.fmaPerCycleTF32.value
      | _ => p.tensor.fmaPerCycle8.value
    let busy := (m.macs + max perSM 1 - 1) / max perSM 1
    let (s, port) := st.gmma[q]!.reserve t busy
    st := { st with gmma := st.gmma.set! q port, mmas := st.mmas + 1,
                    tensorBusy := st.tensorBusy.insert q (st.tensorBusy.getD q 0 + busy),
                    agents := st.agents.modify a ({ · with readyAt := s + p.tensor.tcgen05Latency.value,
                                                           resps := Array.replicate m.loads.size #[],
                                                           loadsLeft := m.loads.size }) }
    -- A and B from the CTA's own shared memory through the tensor core's port (`SmRes.tensor`).
    for i in [0:m.loads.size] do
      (st, mem) ← send st mem s a (.mmaLoad i) m.loads[i]! (tensor := m.loads[i]!.space == .shared)
    progress st mem t a

/-- Start agents `as` at `t`, and those their completion frees, until none is left. -/
def startAll (p : Params) (st : State) (mem : MemSys) (t : Nat) (as : Array Nat) :
    Except String (State × MemSys × Array Out) := do
  let mut st := st
  let mut mem := mem
  let mut outs := #[]
  let mut queue := as
  -- Each agent starts once, so the queue empties within as many rounds as there are agents.
  for _ in [0:st.agents.size + 1] do
    let some a := queue[0]? | break
    queue := queue.eraseIdxIfInBounds 0
    let (st', mem', o, ready) ← startOne p st mem t a
    st := st'; mem := mem'; outs := outs ++ o; queue := queue ++ ready
  return (st, mem, outs)

/-- Close warp `(sm, slot)`'s open operations of kind `g` into a group, whose completion (with
`onRead`, its operations' having read their sources) releases a pending entry of `bar`, at `t`. Returns
when the entry is released if that is known now (the group is already complete, or read), or
`none` when the group's completion will release it; and, released now, the operations already
complete, whose accesses the issuer acquires (the race check: `MemSys.raceAgentsDone`). A pending
group keeps its completed operations among its `ops` (not its `remaining`), so its completion
acquires them. -/
def commit (st : State) (t sm slot : Nat) (g : Group) (bar : Option (Fin 6)) (onRead := false) :
    State × Option Nat × Array Nat :=
  let key := (sm, slot, g.index)
  let all := st.open_.getD key #[]
  let ops := all.filter (st.agents[·]!.doneAt.isNone)
  let st := { st with open_ := st.open_.erase key }
  let unread := (ops.filter (st.agents[·]!.readAt.isNone)).size
  -- A read is known once its data has left, which may be later than now (`readDone`).
  let readLatest := ops.foldl (fun m a => max m ((st.agents[a]!.readAt).getD 0)) 0
  let prev := (st.lastGroup.get? key).filter (!st.groups[·]!.released)
  if prev.isNone && ops.isEmpty then (st, some 0, all)
  else if prev.isNone && onRead && unread == 0 then (st, some readLatest, all)
  else
    -- Pending, or done but behind a group still pending: released in commit order (`releaseFrom`).
    let gi := st.groups.size
    let seq := st.groupCount.getD key 0
    let st := { st with groups := st.groups.push { sm, slot, kind := g, bar, onRead, unread, readLatest,
                                                   remaining := ops.size, ops := all, prev, seq },
                        agents := ops.foldl (fun as a => as.modify a ({ · with groupId := some gi })) st.agents,
                        lastGroup := st.lastGroup.insert key gi, groupCount := st.groupCount.insert key (seq + 1),
                        bySeq := st.bySeq.insert (sm, slot, g.index, seq) gi }
    let st := match prev with
      | some p => { st with groups := st.groups.modify p ({ · with next := some gi }) }
      | none => st
    -- The warp's wgmma groups not complete by `t`: one whose completion is known but later is
    -- still pending to `WARPGROUP.DEPBAR` (`wgmmaPending`) until then.
    let live := (st.wgmma.getD (sm, slot) #[]).filter fun gi => match st.groups[gi]!.doneAt with
      | some d => d > t
      | none => true
    let st := if g == .wgmma then { st with wgmma := st.wgmma.insert (sm, slot) (live.push gi) } else st
    (st, none, #[])

/-- `latestOf` once an entry naming `old` (if any) names `new` instead. -/
def latestSwap (m : Std.HashMap Nat Nat) (old : Option Nat) (new : Nat) : Std.HashMap Nat Nat :=
  let m := match old with
    | some p => match m.get? p with
      | some 1 | none => m.erase p
      | some k => m.insert p (k - 1)
    | none => m
  m.insert new (m.getD new 0 + 1)

/-- Hand operation `op` from warp `o` to its agent at `t`. -/
def start (p : Params) (st : State) (mem : MemSys) (t : Nat) (o : Origin) (op : Op)
    (wb : Resp → Warp → Warp) (pc : Nat := 0) (trace : Option (Nat × Nat) := none) :
    Except String (State × MemSys × Array Out) := do
  let body ← match op with
    | .copy c => pure (Body.copy c)
    | .tensor tc => pure (Body.tensor tc)
    | .mma m => pure (Body.mma m)
    | .mmaT m => pure (Body.mmaT m)
    | .prefetch _ => return (st, mem, #[])
    | .fault msg => throw msg
  let a := st.agents.size
  let group := op.group
  -- The issuer's operations of kind `g` not yet done, in issue order.
  let earlier (g : Group) : Array Nat :=
    (st.byGroup.getD (o.sm, o.slot, g.index) #[]).filter (st.agents[·]!.doneAt.isNone)
  let deps := match op, group with
    | .copy c, some g => if c.afterGroup then earlier g else #[]
    -- A warp's tensor-memory MMAs start at once, in issue order on its tensor core (`prev`), after
    -- its earlier copies and shifts (tcgen05.cp and .shift before .mma are pipelined, §9.7.18.6.2.1;
    -- theirs are the `MmaT`s without multiply-adds, whose tensor-memory writes an MMA may read)
    -- but not its commits: a commit tracks the operations before it, and orders nothing after it.
    | .mmaT _, some g => (earlier g).filter fun i => match st.agents[i]!.body with
      | .mmaT m => m.macs == 0
      | .copy c => !c.afterGroup
      | _ => true
    | _, _ => #[]
  let prev := match op with
    | .mma _ => st.lastMma.get? (o.sm, o.slot)
    | .mmaT _ => st.lastMmaT.get? (o.sm, o.slot)
    | _ => none
  -- The MMA whose result it accumulates from: the latest of the issuer writing the registers it
  -- reads its accumulator from. If that result had not reached the registers when this one issued
  -- (or reaches them in this very cycle, possibly after this read), it accumulates onto it.
  let source := match op with
    | .mma m => m.src.bind fun r => st.lastByDst.get? (o.sm, o.slot, r)
    | .mmaT m => if m.accIn.isSome then st.lastMmaTByD.get? (o.sm, o.slot, m.dKey) else none
    | _ => none
  let chained := match source.map (st.agents[·]!.doneAt) with
    | some none => true
    | some (some d) => d ≥ t
    | none => false
  -- The source's result: held for this one, or, done and passed over, let go.
  let mut st := match source with
    | some src => { st with agents := st.agents.modify src fun p =>
        if chained == true then { p with held := true } else p.letGo }
    | none => st
  st := { st with agents := st.agents.push { origin := o, body, wb, pc, group, deps, prev,
                                             source, chained, tix := trace.map (·.1),
                                             land := (trace.map (·.2)).getD 0 } }
  if let some g := group then
    let key := (o.sm, o.slot, g.index)
    st := { st with byGroup := st.byGroup.insert key (earlier g |>.push a) }
  if let .mma m := op then
    -- Writing D afresh retires its former result too, when this MMA does not accumulate from it.
    if let some prev := st.lastByDst.get? (o.sm, o.slot, m.dst) then
      if source != some prev then st := { st with agents := st.agents.modify prev Agent.letGo }
    st := { st with lastMma := st.lastMma.insert (o.sm, o.slot) a,
                    lastByDst := st.lastByDst.insert (o.sm, o.slot, m.dst) a,
                    latestOf := latestSwap st.latestOf (st.lastByDst.get? (o.sm, o.slot, m.dst)) a }
  if let .mmaT m := op then
    -- Replacing D also retires its former result when this MMA does not accumulate from it.
    if let some prev := st.lastMmaTByD.get? (o.sm, o.slot, m.dKey) then
      st := { st with agents := st.agents.modify prev Agent.letGo }
    st := { st with lastMmaT := st.lastMmaT.insert (o.sm, o.slot) a,
                    lastMmaTByD := st.lastMmaTByD.insert (o.sm, o.slot, m.dKey) a,
                    latestOf := latestSwap st.latestOf (st.lastMmaTByD.get? (o.sm, o.slot, m.dKey)) a }
  -- Operations join their warp's open group until a commit closes it (`tcgen05` operations have
  -- no commit-group: `tcgen05.commit` tracks them as their dependents, `deps`).
  if let some g := group then
    if g != .tcgen05 then
      let key := (o.sm, o.slot, g.index)
      st := { st with open_ := st.open_.insert key ((st.open_.getD key #[]).push a) }
  let mut outs := #[]
  -- The race check: the operation's clock, and an MMA's shared-memory reads as its own (ordered
  -- before the issuer's later writes only by the MMA's completion: wgmma's wait_group, or the
  -- mbarrier of the tcgen05.commit that tracks it).
  let mut mem := mem.raceAgentStart o a deps
  let mmaReads : Array Req := match op with
    | .mma m => m.loads.filter (·.space == .shared)
    | .mmaT m => m.loads.filter (·.space == .shared)
    | _ => #[]
  if !mmaReads.isEmpty then mem := mem.raceAgentReads t o a mmaReads
  if deps.isEmpty then
    (st, mem, outs) ← startAll p st mem t #[a]
  else st := { st with waiting := st.waiting.push a }
  if let .mma m := op then
    if m.commit then
      let (st', _, done) := commit st t o.sm o.slot .wgmma none
      st := st'; mem := mem.raceAgentsDone o.sm o.slot done
  return (st, mem, outs)

/-- A delivery of request `d` of agent `a`, issued for `role`. -/
def deliverAs (p : Params) (st : State) (mem : MemSys) (d : Delivery) (a : Nat) (role : Role) :
    Except String (State × MemSys × Array Out) := do
  let mut st := { st with agents := st.agents.modify a fun g => { g with inflight := g.inflight - 1 } }
  let mut mem := mem
  let mut readOuts : Array Out := #[]
  let t := d.at_
  -- A write answered: its credit returns, and the agents waiting on its SM's credit may count
  -- their sources as read (`readDone`).
  if let some (sm, i) := st.storeOf.get? d.id then
    let log := (st.storeLog.getD sm #[]).modify i fun (c, _) => (c, some t)
    st := { st with storeOf := st.storeOf.erase d.id, storeLog := st.storeLog.insert sm log }
    -- Answered entries leave the front, their latest answer kept as the floor.
    let mut head := st.storeHead.getD sm 0
    let mut floor := st.storeFloor.getD sm 0
    while h : head < log.size do
      match log[head].2 with
      | some at_ => floor := max floor at_; head := head + 1
      | none => break
    let log := if head > 4096 then log.extract head log.size else log
    let head' := if head > 4096 then 0 else head
    let rebase (m : Std.HashMap Nat (Nat × Nat)) (drop : Nat) : Std.HashMap Nat (Nat × Nat) :=
      if drop == 0 then m else m.fold (fun acc k (s', j) => acc.insert k (s', if s' == sm then j - drop else j)) {}
    let dropped := (st.storeLog.getD sm #[]).size - log.size
    st := { st with storeLog := st.storeLog.insert sm log, storeHead := st.storeHead.insert sm head',
                    storeFloor := st.storeFloor.insert sm floor, storeOf := rebase st.storeOf dropped }
    let waiting := st.creditWait.getD sm #[]
    st := { st with creditWait := st.creditWait.erase sm }
    for (w, tw) in waiting do
      (st, readOuts) := readDone st (max tw t) w readOuts
  match role, st.agents[a]!.body with
  | .load i, .copy c =>
    let mv := c.moves[i]!
    st := { st with agents := st.agents.modify a fun g => { g with loadsLeft := g.loadsLeft - 1 } }
    if st.agents[a]!.tix.isSome then
      st := { st with agents := st.agents.modify a fun g =>
        { g with moveResps := (if g.moveResps.isEmpty then Array.replicate c.moves.size #[] else g.moveResps).set! i d.resp } }
    for dst in mv.dsts do
      (st, mem) ← send st mem t a .store (mv.store dst d.resp) (credit := p.tma.storeCreditBytes.value)
    -- Read once the last source is in and the stores made from it have left the SM (`readBy`).
    if st.agents[a]!.loadsLeft == 0 then (st, readOuts) := readDone st t a readOuts
  | .desc, .tensor tc =>
    st := { st with descCached := st.descCached.insert (st.agents[a]!.origin.sm, (tc.desc.addrs[0]?.join).getD 0) }
    let c ← tc.build d.resp
    -- The copy starts now (the agent's start only fetched the descriptor).
    let restart (g : Agent) : Agent := { g with body := .copy c, desc := some (tc.desc, d.resp), started := false }
    st := { st with agents := st.agents.modify a restart }
    return ← startAll p st mem t #[a]
  | .mmaLoad i, .mma _ | .mmaLoad i, .mmaT _ =>
    st := { st with agents := st.agents.modify a fun g =>
      { g with resps := g.resps.set! i d.resp, loadsLeft := g.loadsLeft - 1 } }
    if st.agents[a]!.loadsLeft == 0 then (st, readOuts) := readDone st t a readOuts
  | .signal i, .copy c =>
    if st.agents[a]!.tix.isSome then
      st := { st with agents := st.agents.modify a fun g =>
        { g with sigResps := (if g.sigResps.isEmpty then Array.replicate c.signals.size #[] else g.sigResps).set! i d.resp } }
  | _, _ => pure ()
  let (st', mem', outs, ready) ← progress st mem t a
  let (st', mem', outs') ← startAll p st' mem' t ready
  return (st', mem', readOuts ++ outs ++ outs')

/-- A batch of deliveries: the agents' requests are answered, the others returned for the engine.
Every delivery is matched to its request before any is answered, since answering one may issue
requests that reuse the ids of others in the batch. -/
def deliver (p : Params) (st : State) (mem : MemSys) (ds : Array Delivery) :
    Except String (State × MemSys × Array Out × Array Delivery) := do
  let tagged := ds.map fun d => (d, st.reqs.get? d.id)
  let mut st := { st with reqs := ds.foldl (fun r d => r.erase d.id) st.reqs }
  let mut mem := mem
  let mut outs := #[]
  let mut others := #[]
  for (d, tag) in tagged do
    match tag with
    | some (a, role) =>
      let (st', mem', o) ← deliverAs p st mem d a role
      st := st'; mem := mem'; outs := outs ++ o
    | none => others := others.push d
  return (st, mem, outs, others)

/-- The cycles at which warp `(sm, slot)`'s committed wgmma groups complete, for those not
complete by `t` (`none`: not known yet). -/
def wgmmaPending (st : State) (sm slot t : Nat) : Array (Option Nat) :=
  ((st.wgmma.getD (sm, slot) #[]).map (st.groups[·]!.doneAt)).filter fun
    | some d => d > t
    | none => true

end AsyncUnit

end NV.Core
