import Std.Data.HashMap
import NV.Core.Heap
import NV.Core.Memory.Config
import NV.Core.Memory.Tags

/-!
# The L2 side of the memory system

The crossbar's far side, the L2 slices and their two partitions, the DRAM channels, the slices'
atomic units and the SM-to-SM network, and the request machine (`NV.Memory.Flat`) that holds
every byte of global and shared memory. It runs sequentially (phase B of a quantum): first it
replays what the SMs did to the request machine in the quantum (`Log`), checking that the machine
gives every lane the bytes the SM gave it; then it takes the SMs' messages, in stamp order, and
runs its events up to the quantum's end. Its answers reach the SMs at least a quantum later.

A global access performs at the slice that serves it: its home slice, or for a read of a line
homed in the other partition, the copy slice of the SM's partition when that holds the line.
Every request of an SM to a far line passes that copy slice (writes drop the copy there and go
on), so an SM's requests to a line reach the slice that performs them in the order the SM sent
them. A copy slice keeps one miss in flight per line, as a slice does for its DRAM fills: a read
missing sectors already on their way over the link waits for them there rather than crossing too.
-/

namespace NV.Core.Memory

open NV

/-- The L2 side's rate-limited resources, each one `Port` of `L2Mem.ports`. -/
inductive L2Res where
  | slice (s : Nat)
  /-- The slice's atomic unit. -/
  | atom (s : Nat)
  | chan (c : Nat)
  /-- The link between the L2 partitions, each way. -/
  | farOut | farIn
  deriving Repr, Inhabited, DecidableEq

def L2Res.index (c : Config) : L2Res → Nat
  | .slice s => s
  | .atom s => c.slices + s
  | .chan ch => 2 * c.slices + ch
  | .farOut => 2 * c.slices + c.channels
  | .farIn => 2 * c.slices + c.channels + 1

def L2Res.count (c : Config) : Nat := 2 * c.slices + c.channels + 2

/-- What a resource serves per cycle: bytes, and for the atomic unit operations. -/
def L2Res.rate (c : Config) : L2Res → Rate
  | .slice _ => c.slice
  | .atom _ => c.atom
  | .chan _ => c.channel
  | .farOut | .farIn => c.far

/-- A line request on the L2 side. -/
structure L2Txn where
  sm : Nat
  x : LineReq
  /-- Its line's home slice is in the other partition from the SM. -/
  far : Bool
  /-- A far request that went on from its copy slice to the home slice. -/
  forwarded : Bool := false
  /-- The answers of the lanes performed. -/
  resp : Array (Nat × Array UInt32) := #[]
  deriving Inhabited

/-- Whether a request crosses the partition link: a far line not served by its copy slice. -/
def L2Txn.crosses (x : L2Txn) : Bool := x.far && (x.x.copy.isNone || x.forwarded)

inductive L2Ev where
  /-- A line request leaving its SM's crossbar port. -/
  | enter (sm : Nat) (x : LineReq)
  | slice (txn : Nat)
  /-- An atomic reaching its slice's atomic unit, its line present. -/
  | atom (txn : Nat)
  /-- A DRAM fill of `line` in `slice` arriving at the cycle it was recorded with. -/
  | filled (slice line ready : Nat)
  | ret (txn : Nat)
  deriving Inhabited

structure L2Mem where
  cfg : Config
  flat : Flat
  /-- The request machine's slot of each agent, its CTA index of each `(sm, cta)`, and its
  `(slot, id)` of each request part in flight. -/
  slots : Std.HashMap Agent Nat := {}
  ctas : Std.HashMap (Nat × Nat) Nat := {}
  keys : Std.HashMap Key (Nat × Nat) := {}
  ports : Array Port
  l2 : Array Tags
  /-- Per slice: line ↦ the sectors being filled from DRAM and the cycle they arrive. -/
  l2Fill : Array (Std.HashMap Nat (Nat × Nat))
  /-- Per `(copy slice, line)`: the sectors reads forwarded from there are bringing, how many such
  reads are in flight, and the reads waiting for them. -/
  copyFill : Std.HashMap (Nat × Nat) (Nat × Nat × Array Nat) := {}
  txns : Array L2Txn := #[]
  freeTxns : Array Nat := #[]
  events : Heap L2Ev := {}
  outbox : Array (Stamped Msg) := #[]
  seq : Nat := 0
  counters : Counters := {}
  /-- Per DRAM channel, whether its last transfer wrote (absent: read). -/
  chanWrite : Array Bool := #[]
  deriving Inhabited

namespace L2Mem

/-- The request machine, taken out so that it (and the stores in it) update in place. -/
@[inline] def takeFlat (m : L2Mem) : L2Mem × Flat := ({ m with flat := default }, m.flat)

/-- The L2 side over `global`, its tags holding `l2` from an earlier launch on the same machine
(empty tags if they do not fit this configuration). -/
def init (c : Config) (global : Store) (l2 : Array Tags := #[]) : L2Mem :=
  { cfg := c, flat := Flat.init global #[] 0, ports := Array.replicate (L2Res.count c) {},
    l2 := if l2.size == c.slices then l2 else Array.replicate c.slices (Tags.empty c.l2Sets c.l2Ways),
    l2Fill := Array.replicate c.slices {} }

/-- CTA `cta` of `sm` starts with `bytes` of shared memory. -/
def addCta (m : L2Mem) (sm cta bytes : Nat) (init : Array (Nat × UInt32) := #[]) : L2Mem :=
  let (m, f) := m.takeFlat
  let (flat, i) := f.addCta bytes init
  { m with flat, ctas := m.ctas.insert (sm, cta) i }

def next (m : L2Mem) : Option Nat := m.events.minKey?

/-- The last cycle any of the L2 side's resources (slices, their atomic units, the DRAM channels,
the link between partitions) is busy with work already taken. -/
def busyUntil (m : L2Mem) : Nat := Id.run do
  let c := m.cfg
  let mut last := 0
  for i in [0:m.ports.size] do
    let r : Rate := if i < c.slices then c.slice else if i < 2 * c.slices then c.atom
      else if i < 2 * c.slices + c.channels then c.channel else c.far
    let f := m.ports[i]!.free
    if f > 0 then last := max last ((f + r.units - 1) / max r.units 1 - 1)
  return last

private def take (m : L2Mem) (r : L2Res) (t n : Nat) : Nat × Nat × L2Mem :=
  let i := r.index m.cfg
  let (s, l, p) := m.ports[i]!.take (r.rate m.cfg) t n
  (s, l, { m with ports := m.ports.set! i p })

private def push (m : L2Mem) (t : Nat) (e : L2Ev) : L2Mem := { m with events := m.events.push t e }

private def send (m : L2Mem) (t sm : Nat) (msg : Msg) : L2Mem :=
  { m with outbox := m.outbox.push { stamp := t, sm, seq := m.seq, body := msg }, seq := m.seq + 1 }

/-- The request machine's slot for `a`, made on its first request. -/
private def slotOf (m : L2Mem) (a : Agent) : L2Mem × Nat :=
  match m.slots.get? a with
  | some s => (m, s)
  | none =>
    let (m, f) := m.takeFlat
    let (flat, s) := f.addSlot
    ({ m with flat, slots := m.slots.insert a s }, s)

/-- The agent's own CTA, which its issuer names. -/
private def ownCta : Agent → Nat × Nat
  | .warp sm _ cta => (sm, cta)
  | .async sm cta _ => (sm, cta)

/-- Deliver lane `lane` of part `key` to the request machine: the bytes its reads returned. -/
private def deliver (m : L2Mem) (key : Key) (lane : Nat) : Except String (L2Mem × List UInt8) := do
  let some (slot, id) := m.keys.get? key | throw s!"request {repr key} is not in flight"
  let (m, f) := m.takeFlat
  let (flat, read) ← f.deliver slot id lane
  let m := { m with flat }
  -- Forget the part once its last lane has performed.
  return (if (flat.find slot id).isNone then { m with keys := m.keys.erase key } else m, read)

/-- Replay one record of an SM's phase A. -/
private def replay (m : L2Mem) (e : Log) : Except String L2Mem := do
  match e with
  | .issue key a cta r =>
    let (m, slot) := m.slotOf a
    let some own := m.ctas.get? (ownCta a) | throw s!"{repr a}: its CTA is unknown"
    let some ci := m.ctas.get? cta | throw s!"{repr key}: CTA {cta} is unknown"
    let (m, f) := m.takeFlat
    let (flat, id) ← f.issue { slot, tid := slot * 32, cta := own } r (cta := ci)
    return { m with flat, keys := m.keys.insert key (slot, id) }
  | .perform key lane read =>
    let (m, read') ← m.deliver key lane
    if read' != read then
      throw s!"{repr key} lane {lane}: the SM read {read}, the request machine {read'}"
    return m
  | .fence a =>
    let (m, slot) := m.slotOf a
    let (m, f) := m.takeFlat
    return { m with flat := ← f.fence slot }

private def allocTxn (m : L2Mem) (x : L2Txn) : L2Mem × Nat :=
  match m.freeTxns.back? with
  | some i => ({ m with freeTxns := m.freeTxns.pop, txns := m.txns.set! i x }, i)
  | none => let i := m.txns.size; ({ m with txns := m.txns.push x }, i)

/-- A line request leaves its SM at `t`: to its copy slice (far lines, with copies), across the
link to its home slice (far lines, without), or to its home slice. -/
private def enterStage (m : L2Mem) (t sm : Nat) (x : LineReq) : L2Mem :=
  let c := m.cfg
  let far := x.slice * c.partitions / c.slices != c.nearPartition[sm]!
  let (m, id) := m.allocTxn { sm, x, far }
  if far && x.copy.isNone then
    let (_, last, m) := m.take .farOut t x.bytes
    let m := { m with counters := { m.counters with l2FarSectors := m.counters.l2FarSectors + popcount x.sectors } }
    m.push (last + c.out[1]!) (.slice id)
  else m.push (t + c.out[0]!) (.slice id)

/-- DRAM channel of a line: lines interleave over channels in groups of four (inferred). -/
private def channelOf (c : Config) (line : Nat) : Nat := spread (line >>> 2) c.channels

/-- A transfer of `bytes` on DRAM channel `ch` from `t`, writing or reading: a turn from the other
direction costs the channel `turnBytes` more (`MemParams.dramTurnaroundPs`). -/
private def takeChan (m : L2Mem) (ch t bytes : Nat) (write : Bool) : Nat × Nat × L2Mem :=
  let turn := m.chanWrite.getD ch false != write
  let m := if turn then { m with counters := { m.counters with dramTurns := m.counters.dramTurns + 1 } } else m
  let m := { m with chanWrite := (if m.chanWrite.size > ch then m.chanWrite
      else m.chanWrite ++ Array.replicate (ch + 1 - m.chanWrite.size) false).set! ch write }
  m.take (.chan ch) t (bytes + if turn then m.cfg.turnBytes else 0)

/-- Write a dirty line evicted from the L2 back to DRAM. -/
private def writeBack (m : L2Mem) (t : Nat) (victim : Option (Nat × Nat)) : L2Mem :=
  match victim with
  | some (line, dirty) =>
    if dirty == 0 then m else
    let bytes := popcount dirty * m.cfg.sectorBytes
    let (_, _, m) := m.takeChan (channelOf m.cfg line) t bytes true
    { m with counters := { m.counters with dramWriteBytes := m.counters.dramWriteBytes + bytes } }
  | none => m

/-- Fill sectors of a line in slice `s` (dirty for a write), writing back what it evicts. -/
private def fillL2 (m : L2Mem) (t s line mask : Nat) (write : Bool) : L2Mem :=
  let tags := m.l2[s]!
  let m := { m with l2 := m.l2.set! s default }
  let (tags, victim) := tags.fill line mask write
  { m with l2 := m.l2.set! s tags }.writeBack t victim

/-- A write at the home slice drops the line's copies in the other partitions. -/
private def dropCopies (m : L2Mem) (x : LineReq) : L2Mem := Id.run do
  let c := m.cfg
  if !c.farCopies then return m
  let home := x.slice * c.partitions / c.slices
  let mut m := m
  for q in [0:c.partitions] do
    if q != home then
      let cs := copySlice c x.line q
      m := { m with l2 := m.l2.modify cs (·.invalidate x.line x.sectors) }
  return m

/-- Perform the request's lanes on the request machine, keeping their answers. -/
private def performLanes (m : L2Mem) (id : Nat) : Except String L2Mem := do
  let y := m.txns[id]!
  if y.x.lanes.isEmpty then return { m with txns := m.txns.modify id ({ · with resp := #[] }) }
  let some (slot, rid) := m.keys.get? y.x.key | throw s!"request {repr y.x.key} is not in flight"
  -- Its lanes to the request machine together (`Flat.deliverLanes`: one after another).
  let (m, f) := m.takeFlat
  let (flat, reads) ← f.deliverLanes slot rid y.x.lanes.toList
  let m := { m with flat }
  -- Forget the part once its last lane has performed.
  let m := if (flat.find slot rid).isNone then { m with keys := m.keys.erase y.x.key } else m
  let resp := if y.x.req.op.returns then
      (y.x.lanes.zip reads.toArray).map fun (lane, read) => (lane, laneWords y.x.req read)
    else #[]
  return { m with txns := m.txns.modify id ({ · with resp }) }

/-- A far read at its copy slice: served there if every sector is present, held there if the
missing ones are on their way from reads forwarded before it, else sent on over the partition link
to the home slice. -/
private def copyStage (m : L2Mem) (t id cs : Nat) : Except String L2Mem := do
  let c := m.cfg
  let y := m.txns[id]!
  let n := popcount y.x.sectors
  let present := m.l2[cs]!.present y.x.line y.x.sectors
  -- The copy slice's data port serves the reads it holds; one it sends on, or holds until the line
  -- arrives, passes it as an idle port would (as at the home slice, `sliceStage`).
  let (sLast, m) := if present == y.x.sectors then
      let (_, l, m) := m.take (.slice cs) t (n * c.sectorBytes)
      (l, m)
    else ((({} : Port).take c.slice t (n * c.sectorBytes)).2.1, m)
  let mut m := m
  let key := (cs, y.x.line)
  let (coming, leaders, waiting) := (m.copyFill.get? key).getD (0, 0, #[])
  if present != y.x.sectors && (y.x.sectors ^^^ present) &&& coming == y.x.sectors ^^^ present then
    return { m with copyFill := m.copyFill.insert key (coming, leaders, waiting.push id) }
  if present == y.x.sectors then
    m := { m with l2 := m.l2.modify cs (·.touch y.x.line),
                  counters := { m.counters with l2HitSectors := m.counters.l2HitSectors + n,
                                                l2ReadSectors := m.counters.l2ReadSectors + n } }
    m ← m.performLanes id
    return m.push sLast (.ret id)
  let (_, last, m') := m.take .farOut sLast y.x.bytes
  m := { m' with txns := m'.txns.modify id ({ · with forwarded := true }),
                 copyFill := m'.copyFill.insert key (coming ||| y.x.sectors, leaders + 1, waiting),
                 counters := { m'.counters with l2FarSectors := m'.counters.l2FarSectors + n } }
  return m.push (last + c.out[1]! - c.out[0]!) (.slice id)

private def sliceStage (m : L2Mem) (t id : Nat) : Except String L2Mem := do
  let c := m.cfg
  let y := m.txns[id]!
  if let some cs := y.x.copy then
    if !y.forwarded then return ← m.copyStage t id cs
  let x := y.x
  let n := popcount x.sectors
  -- Writes go into a slice at `sliceWrite` bytes per cycle, against reads' `slice` rate.
  let bytes := n * c.sectorBytes
  let cost := if x.req.op == .ld then bytes else (bytes * c.sliceRead + c.sliceWrite - 1) / max c.sliceWrite 1
  -- A read's sectors the slice lacks pass it as DRAM brings them, their channel's pace: they hold
  -- its data port only as an idle one would, so hits behind a burst of misses are not queued
  -- behind DRAM's backlog (a slice serves hits under misses).
  let missing := if x.req.op == .ld then x.sectors ^^^ m.l2[x.slice]!.present x.line x.sectors else 0
  let (_, sLast, m) := m.take (.slice x.slice) t (cost - popcount missing * c.sectorBytes)
  let sLast := if missing == 0 then sLast else max sLast (({} : Port).take c.slice t cost).2.1
  let mut m ← m.performLanes id
  -- Sectors still arriving from DRAM.
  let (pmask, pready) := (m.l2Fill[x.slice]!.get? x.line).getD (0, 0)
  let (pmask, pready) := if pready ≤ t then (0, 0) else (pmask, pready)
  let mut ready := sLast
  match x.req.op with
  | .st =>
    m := { m with counters := { m.counters with l2WriteSectors := m.counters.l2WriteSectors + n } }
    m := (m.fillL2 sLast x.slice x.line x.sectors true).dropCopies x
  | _ =>
    let miss := x.sectors ^^^ m.l2[x.slice]!.present x.line x.sectors
    if x.sectors &&& pmask != 0 then ready := max ready pready
    m := { m with counters :=
      let k := { m.counters with l2HitSectors := m.counters.l2HitSectors + (n - popcount miss),
                                 l2MissSectors := m.counters.l2MissSectors + popcount miss }
      if x.req.op == .ld then { k with l2ReadSectors := k.l2ReadSectors + n }
      else { k with l2AtomSectors := k.l2AtomSectors + n } }
    let write := x.req.op != .ld
    m := m.fillL2 sLast x.slice x.line x.sectors write
    if miss != 0 then
      let bytes := popcount miss * c.sectorBytes
      let (_, cLast, m') := m.takeChan (channelOf c x.line) sLast bytes false
      let fill := max pready (cLast + c.dramExtra)
      ready := max ready fill
      m := { m' with l2Fill := m'.l2Fill.modify x.slice (·.insert x.line (pmask ||| miss, fill)),
                     counters := { m'.counters with dramReadBytes := m'.counters.dramReadBytes + bytes } }
      m := m.push fill (.filled x.slice x.line fill)
    if write then return (m.dropCopies x).push ready (.atom id)
  return m.push ready (.ret id)

private def atomStage (m : L2Mem) (t id : Nat) : L2Mem :=
  let y := m.txns[id]!
  let (_, last, m) := m.take (.atom y.x.slice) t y.x.lanes.size
  m.push last (.ret id)

/-- A recorded fill has arrived: forget it, unless a later one for the line replaced it. -/
private def filledStage (m : L2Mem) (s line ready : Nat) : L2Mem :=
  match m.l2Fill[s]!.get? line with
  | some (_, r) => if r == ready then { m with l2Fill := m.l2Fill.modify s (·.erase line) } else m
  | none => m

/-- The answer leaves the slice at `t` for its SM. -/
private def retStage (m : L2Mem) (t id : Nat) : L2Mem := Id.run do
  let c := m.cfg
  let y := m.txns[id]!
  let x := y.x
  let mut m := { m with freeTxns := m.freeTxns.push id }
  let bytes := match x.req.op with
    | .ld => popcount x.sectors * c.sectorBytes
    | .atom .. => x.lanes.size * x.req.bytes
    | _ => 0
  let mut last := t
  if y.crosses then
    let (_, l, m') := m.take .farIn t bytes
    m := m'
    last := l
  let answer := last + c.back[if y.crosses then 1 else 0]!
  -- A read forwarded from its copy slice leaves the line there on its way back, and the reads
  -- waiting there look again: answered as it is, not before.
  if let some cs := x.copy then
    if y.forwarded && x.req.op == .ld then
      m := m.fillL2 t cs x.line x.sectors false
      if let some (coming, leaders, waiting) := m.copyFill.get? (cs, x.line) then
        m := { m with copyFill := if leaders ≤ 1 then m.copyFill.erase (cs, x.line)
                                  else m.copyFill.insert (cs, x.line) (coming, leaders - 1, #[]) }
        for w in waiting do m := m.push (answer - c.back[0]!) (.slice w)
  return m.send answer y.sm (.answer x.txn y.resp)

private def step (m : L2Mem) (t : Nat) : L2Ev → Except String L2Mem
  | .enter sm x => pure (m.enterStage t sm x)
  | .slice id => m.sliceStage t id
  | .atom id => pure (m.atomStage t id)
  | .filled s line ready => pure (m.filledStage s line ready)
  | .ret id => pure (m.retStage t id)

/-- Phase B of the quantum ending at `stop`: replay the SMs' records, take their messages, run
the events before `stop`. The messages for the SMs, stamped with their arrival. -/
def run (m : L2Mem) (logs : Array (Stamped Log)) (msgs : Array (Stamped Msg)) (stop : Nat) :
    Except String (L2Mem × Array (Stamped Msg)) := do
  let c := m.cfg
  let mut m := m
  for e in logs.qsort (·.before ·) do m ← m.replay e.body
  for msg in msgs.qsort (·.before ·) do
    match msg.body with
    | .line x => m := m.push msg.stamp (.enter msg.sm x)
    | .peer x => m := m.send (msg.stamp + c.dsmemOut) x.to (.peer x)
    | .peerAnswer key resp => m := m.send (msg.stamp + c.dsmemBack) key.sm (.peerAnswer key resp)
    | .answer .. => throw "an answer sent to the L2"
  -- Each event is taken once; the bound only guards a defect.
  for _ in [0:1000000000] do
    match m.events.minKey? with
    | some at_ => if at_ ≥ stop then break
    | none => break
    let h := m.events
    m := { m with events := {} }
    let some (at_, e, h) := h.pop? | break
    m ← { m with events := h }.step at_ e
  return ({ m with outbox := #[] }, m.outbox)

end L2Mem

end NV.Core.Memory
