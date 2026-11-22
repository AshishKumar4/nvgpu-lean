import Std.Data.HashMap
import NV.Core.Heap
import NV.Core.Memory.Config
import NV.Core.Memory.Tags
import NV.Core.Memory.Race

/-!
# The SM's side of the memory system

One SM's memory pipeline: its sub-partitions' MIO queues, the address and LSU ports, the
coalescer, the L1 tag stage with its MSHRs, the data port where shared memory and L1 hits are
served, its CTAs' shared memory and its warps' local memory, and its crossbar and SM-to-SM ports.
An SM runs alone for a quantum (phase A): what it sends off the SM leaves stamped in `outbox`,
what it does to the request machine is recorded in `log`, and what reaches it was stamped by the
L2 half at least a quantum earlier.

Data. A shared or local access performs at the data port, on the SM's own copy of the CTA's
shared memory (the request machine replays it in phase B and must agree). An L1 hit performs at
the tag stage on the global memory as it was when the quantum began; it is refused (sent on as a
miss) while the SM has a request to the same sector unanswered, so an SM's accesses to a sector
stay in issue order, and anything that could causally precede the hit crossed between SMs, which
takes at least a quantum, so is in that memory. Every other global access performs at its L2
slice (`NV.Core.Memory.L2`), reached by every request of the SM to a line along one path, in
order.
-/

namespace NV.Core.Memory

open NV

/-- A request on the SM. -/
structure ReqState where
  origin : Origin
  req : Req
  resp : Resp
  /-- Parts (lines, shared requests, cluster peers) not yet back. -/
  parts : Nat := 0
  ready : Nat := 0
  fence : Bool := false
  /-- From the SM's async unit: no MIO queue, no L1, not ordered by the warp's fences. -/
  async : Bool := false
  /-- A tensor-core operand read (`wgmma`'s and `tcgen05.mma`'s shared-memory A and B): through the
  tensor core's own port into shared memory (`SmRes.tensor`), not the data stage. -/
  tensor : Bool := false
  /-- The warpgroup MMA's shares that read this operand together (B: the four warps'), each
  charged its share of the port time. -/
  tensorShares : Nat := 1
  /-- A warpgroup MMA's operand read (not `tcgen05.mma`'s). -/
  wgmma : Bool := false
  /-- For an async request, the async unit's operation it belongs to. -/
  op : Nat := 0
  /-- The cycle its instruction issued, which the pipeline's latencies count from. -/
  issued : Nat := 0
  /-- Parts not yet past the SM's ordering point. -/
  toOrder : Nat := 0
  /-- An instruction-cache refill (`issueCode`): timed through the L2, no lane performs. -/
  code : Bool := false
  /-- The cycle its last data left the SM for the L2 (stores, reductions). -/
  departAt : Nat := 0
  /-- A `try_wait` held at the data port (`SmMem.parked`): the cycle it returns false at. -/
  heldUntil : Option Nat := none
  deriving Inhabited

/-- A `try_wait` held at the data port while its barrier stays in the phase it waits past, with
what its data-port event needs to look again. -/
structure Held where
  req : Nat
  cta : Nat
  lanes : Array Nat
  key : Key
  r : Req
  /-- The cycle it returns false at (`Config.tryWaitLimit` after it first arrived). -/
  deadline : Nat
  deriving Inhabited

/-- A line request of the SM waiting for its answer. -/
structure Txn where
  req : Nat
  line : Nat
  sectors : Nat
  /-- The sectors whose L1 miss it fills. -/
  fills : Nat := 0
  write : Bool
  /-- Loads merged into it: their requests and the lanes that perform when it arrives. -/
  waiters : Array (Nat × Array Nat) := #[]
  deriving Inhabited

/-- A warp's requests on the SM, for fences, release/acquire and barriers. -/
structure WarpMem where
  /-- Requests not yet past the SM's ordering point (tag stage, data port, network port). -/
  unordered : Nat := 0
  queued : Nat := 0
  inflight : Nat := 0
  acquiring : Nat := 0
  /-- Fences waiting for the warp to drain: their ids and own costs. -/
  fences : Array (Nat × Nat) := #[]
  deriving Inhabited

/-- The SM's rate-limited resources, each one `Port` of `SmMem.ports`. -/
inductive SmRes where
  /-- A sub-partition's address calculation, one instruction per `addressInterval`. -/
  | addr (q : Nat)
  /-- The LSU: one instruction per `lsuInterval` from all sub-partitions. -/
  | lsu
  /-- The L1 tag stage: one line per cycle. -/
  | tag
  /-- The L1/shared data stage: one wavefront per `wavefrontCycles`. -/
  | data
  /-- The crossbar port, towards the L2 and back. -/
  | xout | xin
  /-- The port into the SM-to-SM network, and back. -/
  | dsOut | dsIn
  /-- The tensor core's operand port into shared memory (`tcgen05.mma`'s A and B): one 128-byte
  wavefront a cycle, apart from the data stage the warps' loads and stores take. -/
  | tensor
  deriving Repr, Inhabited, DecidableEq

/-- Cycles the tag stage spends on one line's access: its sectors at `l1TagSectors` per cycle. -/
def tagCycles (c : Config) (g : LineAccess) : Nat :=
  max 1 ((popcount g.sectors + c.l1TagSectors - 1) / max 1 c.l1TagSectors)

def SmRes.index (c : Config) : SmRes → Nat
  | .addr q => q
  | .lsu => c.smsps
  | .tag => c.smsps + 1
  | .data => c.smsps + 2
  | .xout => c.smsps + 3
  | .xin => c.smsps + 4
  | .dsOut => c.smsps + 5
  | .dsIn => c.smsps + 6
  | .tensor => c.smsps + 7

def SmRes.count (c : Config) : Nat := c.smsps + 8

/-- What a resource serves per cycle: instructions, lines and wavefronts are whole cycles of work;
the network ports move bytes. -/
def SmRes.rate (c : Config) : SmRes → Rate
  | .addr _ | .lsu | .tag | .data | .tensor => { units := 1 }
  | .xout => c.xbarWrite
  | .xin => c.xbar
  | .dsOut | .dsIn => c.dsmem

inductive SmEv where
  /-- The LSU takes the next request from the sub-partitions' MIO queues. -/
  | dispatch
  /-- One line of a global or local request at the tag stage. -/
  | tag (req : Nat) (g : LineAccess)
  /-- A shared request (or the part `key` of a cluster request for this SM's own CTA) at the data
  port, for CTA `cta`. -/
  | bank (req cta : Nat) (lanes : Array Nat) (key : Key) (r : Req)
  /-- Another SM's cluster request part at the data port. -/
  | remote (x : PeerReq)
  /-- A tensor-memory request's lanes, at the SM's tensor memory. -/
  | tmem (req : Nat) (lanes : Array Nat)
  /-- A warpgroup MMA's operand read takes the data stage for `cycles`
  (`Config.wgmmaOperandData`). -/
  | dataShare (cycles : Nat)
  /-- Held `try_wait` `req` on CTA `cta`'s barrier at `addr` reaches its limit `deadline`. -/
  | tryWaitEnd (cta addr req deadline : Nat)
  /-- A cluster request part leaving for another SM. -/
  | send (x : PeerReq)
  /-- A message from the other half arriving. -/
  | arrive (m : Msg)
  | done (req : Nat)
  deriving Inhabited

/-- Where the CTAs of clusters run: `(cluster, rank)` ↦ `(sm, cta)`, and back. The engine places
CTAs; the SMs only read it. -/
structure Directory where
  slot : Std.HashMap (Nat × Nat) (Nat × Nat) := {}
  rank : Std.HashMap (Nat × Nat) (Nat × Nat) := {}
  deriving Inhabited

structure SmMem where
  cfg : Config
  sm : Nat
  /-- Per CTA index on this SM: its shared memory. -/
  shared : Array Store := #[]
  /-- The SM's tensor memory (`Space.tmem`), allocated on its first write. -/
  tmem : Store := Store.empty 0 Space.tmemBytes
  /-- Per warp slot: each lane's local memory; per CTA its warps. -/
  locals : Std.HashMap Nat (Array Store) := {}
  ctaWarps : Std.HashMap Nat (Array Nat) := {}
  /-- CTAs removed while requests of theirs were in flight; freed when those deliver. -/
  retired : Std.HashMap Nat Unit := {}
  ctaInflight : Std.HashMap Nat Nat := {}
  reqs : Array ReqState := #[]
  freeReqs : Array Nat := #[]
  /-- Ids delivered by the current `advance`, reusable from the next one, so the ids of one batch
  of deliveries stay distinct while the caller answers them. -/
  delivered : Array Nat := #[]
  txns : Array Txn := #[]
  freeTxns : Array Nat := #[]
  events : Heap SmEv := {}
  /-- Per sub-partition and queue (`queueOf`: shared memory's MIO queue 0, apart from the global
  and local accesses' LG queue 1; index `2 * q + queue`): accepted requests not yet dispatched,
  and slots taken by issued instructions whose requests have not arrived yet. -/
  mio : Array (Array Nat)
  reserved : Array Nat
  armed : Bool := false
  rr : Nat := 0
  /-- When the shared stores dispatched so far leave the data stage: a store waits in its queue
  behind them, not behind the loads' and the tensor core's backlog there. -/
  storesFree : Nat := 0
  warps : Std.HashMap Nat WarpMem := {}
  ports : Array Port
  l1 : Tags
  /-- Line ↦ the transactions filling it, each with the sectors it fills. -/
  mshr : Std.HashMap Nat (Array (Nat × Nat)) := {}
  /-- Lines with a load waiting for an MSHR, and that load. -/
  mshrBlocked : Std.HashMap Nat Nat := {}
  /-- Sector ↦ the SM's requests to it sent to the L2 and not answered, and of those the writes. -/
  pending : Std.HashMap Nat Nat := {}
  pendingWrites : Std.HashMap Nat Nat := {}
  outbox : Array (Stamped Msg) := #[]
  /-- Per line, the last cycle a write of this SM to it occupies the crossbar port: a later request
  to the line leaves no earlier, so it reaches the L2 behind the write (`sendLine`). -/
  writeOut : Std.HashMap Nat Nat := {}
  /-- The data-race check (`Config.races`): its state, kept only when on. -/
  race : RaceCheck := {}
  /-- Per CTA and barrier address, the `try_wait`s held on it (`bankStage`). -/
  held : Std.HashMap (Nat × Nat) (Array Held) := {}
  log : Array (Stamped Log) := #[]
  seq : Nat := 0
  counters : Counters := {}
  deriving Inhabited

namespace SmMem

def init (c : Config) (sm : Nat) : SmMem :=
  { cfg := c, sm, mio := Array.replicate (2 * c.smsps) #[], reserved := Array.replicate (2 * c.smsps) 0,
    ports := Array.replicate (SmRes.count c) {}, l1 := Tags.empty c.l1Sets c.l1Ways }

/-- CTA `cta` starts with `bytes` of shared memory. -/
def addCta (m : SmMem) (cta bytes : Nat) (init : Array (Nat × UInt32) := #[]) (reserved := 0) : SmMem :=
  let sh := m.shared ++ Array.replicate (cta + 1 - m.shared.size) (Store.empty 0 0)
  let s := init.foldl (fun s (off, w) => if off + 4 ≤ bytes then s.writeWord off w else s) (Store.empty 0 bytes)
  { m with shared := sh.set! cta s,
           race := if m.cfg.races then m.race.newCta cta reserved else m.race }

/-- Warp `slot` of CTA `cta` starts, each lane with `bytes` of local memory. -/
def addWarp (m : SmMem) (cta slot bytes : Nat) : SmMem :=
  { m with race := if m.cfg.races then m.race.start slot else m.race,
           locals := m.locals.insert slot (Array.replicate 32 (Store.empty 0 bytes)),
           ctaWarps := m.ctaWarps.insert cta ((m.ctaWarps.getD cta #[]).push slot) }

private def free (m : SmMem) (cta : Nat) : SmMem :=
  let ws := m.ctaWarps.getD cta #[]
  { m with shared := m.shared.set! cta (Store.empty 0 0),
           locals := ws.foldl (fun h w => h.erase w) m.locals,
           ctaWarps := m.ctaWarps.erase cta, retired := m.retired.erase cta }

/-- CTA `cta` has finished: its memory is freed once its requests in flight have delivered. -/
def removeCta (m : SmMem) (cta : Nat) : SmMem :=
  if m.ctaInflight.getD cta 0 == 0 then m.free cta else { m with retired := m.retired.insert cta () }

/-- The queue a request of `space` waits in: shared memory's (MIO, 0) or the global and local
accesses' (LG, 1), apart, so a shared store held for the data stage does not hold a global load
(wgmio on the GH200: an LDG chain beside an STS.128 stream runs 13% slower). -/
def queueOf (space : Space) : Nat :=
  match space with
  | .shared | .sharedCluster => 0
  | _ => 1

/-- Whether sub-partition `q`'s queue `queue` has room. -/
def accepts (m : SmMem) (q : Nat) (queue : Nat) : Bool :=
  m.mio[2 * q + queue]!.size + m.reserved[2 * q + queue]! < m.cfg.mioDepth

/-- A memory instruction of CTA `cta` issued on sub-partition `q` takes a slot in its queue; its
request, which `issue` brings later, fills it. The CTA's memory is kept from then on. -/
def reserve (m : SmMem) (q cta queue : Nat) : SmMem :=
  { m with reserved := m.reserved.modify (2 * q + queue) (· + 1),
           ctaInflight := m.ctaInflight.insert cta (m.ctaInflight.getD cta 0 + 1) }

/-- Whether every request warp `slot` has issued is past the SM's ordering point, so the SM's later
accesses, by any warp, follow it: a barrier may count the warp's arrival. -/
def ordered (m : SmMem) (slot : Nat) : Bool := (m.warps.get? slot).all (·.unordered == 0)

def next (m : SmMem) : Option Nat := m.events.minKey?

private def push (m : SmMem) (t : Nat) (e : SmEv) : SmMem := { m with events := m.events.push t e }

private def take (m : SmMem) (r : SmRes) (t n : Nat) : Nat × Nat × SmMem :=
  let i := r.index m.cfg
  let (s, l, p) := m.ports[i]!.take (r.rate m.cfg) t n
  (s, l, { m with ports := m.ports.set! i p })

private def alloc (m : SmMem) (r : ReqState) : SmMem × Nat :=
  match m.freeReqs.back? with
  | some i => ({ m with freeReqs := m.freeReqs.pop, reqs := m.reqs.set! i r }, i)
  | none => let i := m.reqs.size; ({ m with reqs := m.reqs.push r }, i)

private def allocTxn (m : SmMem) (x : Txn) : SmMem × Nat :=
  match m.freeTxns.back? with
  | some i => ({ m with freeTxns := m.freeTxns.pop, txns := m.txns.set! i x }, i)
  | none => let i := m.txns.size; ({ m with txns := m.txns.push x }, i)

private def modReq (m : SmMem) (id : Nat) (f : ReqState → ReqState) : SmMem :=
  { m with reqs := m.reqs.modify id f }

private def warp (m : SmMem) (slot : Nat) : WarpMem := m.warps.getD slot {}

private def setWarp (m : SmMem) (slot : Nat) (w : WarpMem) : SmMem :=
  if w.unordered == 0 && w.queued == 0 && w.inflight == 0 && w.acquiring == 0 && w.fences.isEmpty then
    { m with warps := m.warps.erase slot }
  else { m with warps := m.warps.insert slot w }

private def record (m : SmMem) (t : Nat) (e : Log) : SmMem :=
  { m with log := m.log.push { stamp := t, sm := m.sm, seq := m.seq, body := e }, seq := m.seq + 1 }

private def send (m : SmMem) (t : Nat) (msg : Msg) : SmMem :=
  { m with outbox := m.outbox.push { stamp := t, sm := m.sm, seq := m.seq, body := msg }, seq := m.seq + 1 }

/-- Make sure the LSU will look at the queues at `t` or later. -/
private def arm (m : SmMem) (t : Nat) : SmMem :=
  if m.armed then m
  else { (m.push (max t m.ports[SmRes.lsu.index m.cfg]!.free) .dispatch) with armed := true }

private def agent (m : SmMem) (r : ReqState) : Agent :=
  if r.async then .async m.sm r.origin.cta r.op else .warp m.sm r.origin.slot r.origin.cta

/-- Warp `slot`'s held `try_wait`s answer now (`bankStage`), ahead of a request of the warp
passing the LSU after its wait (a cp.async's copy starting late): the request machine performs a
thread's accesses after its earlier acquire. -/
private def releaseHeld (m : SmMem) (t slot : Nat) : SmMem := Id.run do
  if m.held.isEmpty then return m
  let mut m := m
  for (k, hs) in m.held.toList do
    let (mine, rest) := hs.partition fun h => m.reqs[h.req]!.origin.slot == slot
    if mine.isEmpty then continue
    m := { m with held := if rest.isEmpty then m.held.erase k else m.held.insert k rest }
    for h in mine do
      m := (m.modReq h.req fun q => { q with heldUntil := some t }).push t (.bank h.req h.cta h.lanes h.key h.r)
  return m

/-- Accept a request from warp `o` at cycle `t` into its sub-partition's MIO queue. Its
instruction issued at `issued`; the pipeline's latencies count from there. -/
def issue (m : SmMem) (t : Nat) (o : Origin) (r : Req) (issued : Nat := t) : SmMem × Nat :=
  let m := if m.cfg.races then { m with race := m.race.access issued o.cta o.slot r } else m
  let (m, id) := m.alloc { origin := o, req := r, resp := Array.replicate (r.addrs.size * words r.bytes) 0,
                           issued }
  let m := match r.op with
    | .mbar (.arrive ..) => if m.cfg.races then { m with race := m.race.mbarArrive o.slot id } else m
    | _ => m
  let qi := 2 * o.smsp + queueOf r.space
  let m := { m with mio := m.mio.modify qi (·.push id) }
  -- The engine reserved the slot (and counted the request against its CTA) at issue.
  let m := if m.reserved[qi]! > 0 then { m with reserved := m.reserved.modify qi (· - 1) }
    else { m with ctaInflight := m.ctaInflight.insert o.cta (m.ctaInflight.getD o.cta 0 + 1) }
  let w := m.warp o.slot
  let m := m.setWarp o.slot { w with queued := w.queued + 1, unordered := w.unordered + 1 }
  (m.arm t, id)

/-- A fence from warp `o` at `t` costing `cost` cycles of its own (`SyncParams.fenceCycles`): it
delivers `cost` cycles after every earlier request of the warp has delivered. -/
def fence (m : SmMem) (t : Nat) (o : Origin) (cost : Nat) : SmMem × Nat :=
  let (m, id) := m.alloc { origin := o, req := default, resp := #[], fence := true }
  let w := m.warp o.slot
  if w.queued == 0 && w.inflight == 0 then (m.push (t + cost) (.done id), id)
  else (m.setWarp o.slot { w with fences := w.fences.push (id, cost) }, id)

/-- One part of request `id` is back, ready at `ready`. -/
private def complete (m : SmMem) (id ready : Nat) : SmMem :=
  let m := m.modReq id fun r => { r with parts := r.parts - 1, ready := max r.ready ready }
  let r := m.reqs[id]!
  if r.parts == 0 then m.push r.ready (.done id) else m

/-- Write lane `lane`'s response words into request `id`. -/
private def answerLane (m : SmMem) (id lane : Nat) (ws : Array UInt32) : SmMem :=
  let n := words m.reqs[id]!.req.bytes
  m.modReq id fun r =>
    let resp := (List.range ws.size).foldl (fun resp j => resp.set! (lane * n + j) ws[j]!) r.resp
    { r with resp }

/-- One of request `id`'s parts has passed the SM's ordering point (the tag stage, the data port,
the network port); once all have, the warp's barrier arrivals no longer wait for it. -/
private def passed (m : SmMem) (id : Nat) : SmMem :=
  let m := m.modReq id fun r => { r with toOrder := r.toOrder - 1 }
  let r := m.reqs[id]!
  if r.toOrder == 0 && !r.async then
    let w := m.warp r.origin.slot
    m.setWarp r.origin.slot { w with unordered := w.unordered - 1 }
  else m

/-- Perform `lanes` of `key`'s request `r` on store `s` as the request machine does, recording
them; the request `id`'s response gets their words. -/
private def performOn (m : SmMem) (t : Nat) (s : Store) (cta id : Nat) (key : Key) (r : Req)
    (lanes : Array Nat) (writes : Bool) (answer := true) :
    Except String (SmMem × Store × Array (Nat × Array UInt32)) := do
  let mut m := m
  let mut s := s
  let mut out := #[]
  for lane in lanes do
    let some l := Flat.loc cta r lane | continue
    -- Writing, the store is handed over (its only reference), so it is written in place.
    let mut read := []
    if writes then
      let (s', rd) ← Flat.perform r lane l s
      s := s'; read := rd
    else
      let (_, rd) ← Flat.perform r lane l s
      read := rd
    m := m.record t (.perform key lane read)
    -- The race check's synchronization: an mbarrier arrive's release, an acquire read's acquire.
    if m.cfg.races && answer && r.space == .shared then
      if let some (a : UInt64) := r.addrs[lane]! then
        m := match r.op with
          | .mbar (.arrive ..) | .mbar (.completeTx _) => { m with race := m.race.mbarPerformed cta a.toNat id }
          | .ld => if r.sem == Sem.acquire || r.sem == Sem.acqRel then
              { m with race := m.race.acquire cta m.reqs[id]!.origin.slot a.toNat } else m
          | _ => m
    let ws := laneWords r read
    if answer then m := m.answerLane id lane ws
    out := out.push (lane, ws)
  return (m, s, out)

/-- Perform lanes of a local request on the warp's lanes' stores, in issue order per thread
(local memory is thread-private; the request machine does not see it). -/
private def performLocal (m : SmMem) (id : Nat) (r : Req) (lanes : Array Nat) : Except String SmMem := do
  let slot := m.reqs[id]!.origin.slot
  let some ls := m.locals.get? slot | throw "local memory of a warp that is gone"
  let mut m := { m with locals := m.locals.erase slot }
  let mut ls := ls
  let n := words r.bytes
  for lane in lanes do
    let st := ls[lane]!
    ls := ls.set! lane default
    let (st, out) ← st.perform r lane
    ls := ls.set! lane st
    m := m.answerLane id lane (out.extract 0 n)
  return { m with locals := m.locals.insert slot ls }

/-- Whether the head of a warp's queue must wait: a release behind the warp's requests in flight,
or anything behind an acquire in flight. -/
private def gated (m : SmMem) (id : Nat) : Bool :=
  let r := m.reqs[id]!
  let w := m.warp r.origin.slot
  w.acquiring > 0 || ((r.req.sem == .release || r.req.sem == .acqRel) && w.inflight > 0)

/-- The sector index of sector `s` of `line`. -/
private def sectorKey (c : Config) (line s : Nat) : Nat := line * (c.lineBytes / c.sectorBytes) + s

private def countSectors (m : SmMem) (line mask : Nat) (write : Bool) (d : Int) : SmMem := Id.run do
  let c := m.cfg
  let mut m := m
  for s in [0:c.lineBytes / c.sectorBytes] do
    if (mask >>> s) % 2 == 1 then
      let k := sectorKey c line s
      let bump (h : Std.HashMap Nat Nat) :=
        let n := ((h.getD k 0 : Int) + d).toNat
        if n == 0 then h.erase k else h.insert k n
      m := { m with pending := bump m.pending }
      if write then m := { m with pendingWrites := bump m.pendingWrites }
  return m

/-- The sectors of `mask` of `line` that the SM has requests to unanswered (`writes`: writes). -/
private def busy (m : SmMem) (line mask : Nat) (writes : Bool) (atLeast : Nat := 1) : Nat := Id.run do
  let c := m.cfg
  let h := if writes then m.pendingWrites else m.pending
  let mut out := 0
  for s in [0:c.lineBytes / c.sectorBytes] do
    if (mask >>> s) % 2 == 1 && h.getD (sectorKey c line s) 0 ≥ atLeast then out := out ||| (1 <<< s)
  return out

/-- Send a line of request `id` to the L2 at `t`: the crossbar port, then the message. -/
private def sendLine (m : SmMem) (t id : Nat) (g : LineAccess) (fills : Nat) (waiters : Array (Nat × Array Nat) := #[]) :
    SmMem := Id.run do
  let c := m.cfg
  let r := m.reqs[id]!
  let req := r.req
  let write := req.op.writes
  let n := popcount g.sectors
  let bytes := match req.op with
    | .ld => 0
    | .st => n * c.sectorBytes
    | _ => g.lanes.size * req.bytes
  -- Reduction and atomic operands leave at `xbarRed` bytes per cycle, slower than stores' data.
  let cost := match req.op with
    | .ld | .st => bytes
    | _ => (bytes * c.xbarWriteBytes + c.xbarRed - 1) / max c.xbarRed 1
  -- A request to a line leaves behind the SM's earlier writes to it, whose data may still be on the
  -- port: an access may not reach a byte before its thread's earlier write to it does (the request
  -- machine's per-thread order, `Flat.blocker`); a load moves no data, so the port alone would let
  -- it pass.
  let t := max t (m.writeOut.getD g.line 0)
  let (_, last, m) := m.take .xout t cost
  let m := if !write then m else
    let wo := if m.writeOut.size < 4096 then m.writeOut else m.writeOut.filter fun _ v => v > t
    { m with writeOut := wo.insert g.line last }
  let m := m.modReq id fun r => { r with departAt := max r.departAt last }
  let (m, txn) := m.allocTxn { req := id, line := g.line, sectors := g.sectors, fills, write, waiters }
  let m := m.countSectors g.line g.sectors write 1
  let slice := spread g.line c.slices
  let home := slice * c.partitions / c.slices
  let near := c.nearPartition[m.sm]!
  -- A far line's copy serves reads; a write goes to its home slice, which drops the copies. A read
  -- behind the SM's own write to the line in flight goes to the home slice too, after the write.
  let copy := if home != near && c.farCopies && !write && busy m g.line g.sectors true == 0 then
    some (copySlice c g.line near) else none
  let x : LineReq := { key := { sm := m.sm, req := id }, txn := txn, req := req, line := g.line,
                       sectors := g.sectors, lanes := g.lanes, slice := slice, copy := copy, bytes := bytes }
  return m.send last (.line x)

/-- A line of the async unit's write (bulk or tensor store, reduction) at `t`: it passes the tag
stage without looking up the L1 and leaves for the L2 at once, so when its data leaves the SM is
known as it is issued (`ReqState.departAt`, which the async unit's read release waits for). -/
private def asyncWriteLine (m : SmMem) (t id : Nat) (g : LineAccess) : SmMem :=
  let n := popcount g.sectors
  let m := m.passed id
  let m := match m.reqs[id]!.req.op with
    | .st => { m with counters := { m.counters with globalStRequests := m.counters.globalStRequests + 1,
                                                    globalStSectors := m.counters.globalStSectors + n } }
    | _ => { m with counters := { m.counters with globalAtomSectors := m.counters.globalAtomSectors + n },
                    l1 := m.l1.invalidate g.line g.sectors }
  m.sendLine t id g 0

/-- The race check (`Config.races`): warp `slot` of CTA index `cta` starts async-proxy reads `rs`
at `t` (`RaceCheck.access`), or issues a proxy fence. -/
def raceAsyncReads (m : SmMem) (t cta slot : Nat) (rs : Array Req) : SmMem :=
  if m.cfg.races then { m with race := rs.foldl (fun rc r => rc.access t cta slot r (async := true)) m.race }
  else m

/-- The race check: warp `slot` issues async operation `a`, or acquires the writes of the
completed group `ops` it issued. -/
def raceAgentStart (m : SmMem) (slot a : Nat) (deps : Array Nat := #[]) : SmMem :=
  if m.cfg.races then { m with race := m.race.agentStart slot a deps } else m

/-- The race check: async operation `a` of warp `slot` of CTA index `cta` reads `rs` at `t`. -/
def raceAgentReads (m : SmMem) (t cta slot a : Nat) (rs : Array Req) : SmMem :=
  if m.cfg.races then
    { m with race := rs.foldl (fun rc r => rc.access t cta slot r (async := true) (agent := some a)) m.race }
  else m

def raceAgentsDone (m : SmMem) (slot : Nat) (ops : Array Nat) : SmMem :=
  if m.cfg.races then { m with race := m.race.agentsDone slot ops } else m

def raceProxyFence (m : SmMem) (slot : Nat) : SmMem :=
  if m.cfg.races then { m with race := m.race.proxyFence slot } else m

/-- A code line's refill for the SM's instruction caches at `t`, for warp `o`: the line at `addr`
from the L2 like a load of all its sectors, which no lane performs (code is not in the modelled
memory) and no warp or CTA waits on as a request; its delivery says when the line is here. -/
def issueCode (m : SmMem) (t : Nat) (o : Origin) (addr : Nat) : SmMem × Nat :=
  let c := m.cfg
  let req : Req := { space := .global, op := .ld, bytes := 4, addrs := #[some addr.toUInt64] }
  let (m, id) := m.alloc { origin := o, req, resp := #[], parts := 1, code := true }
  let all := (1 <<< (c.lineBytes / c.sectorBytes)) - 1
  let g : LineAccess := { line := addr / c.lineBytes, sectors := all, lanes := #[] }
  (m.sendLine t id g 0, id)

/-- A `shared::cluster` request of CTA `cta` on SM `sm`, split by the CTA each lane's address names
(`dir`): per part, that CTA's SM and slot, its lanes, and the part as a `.shared` request at
offsets in that CTA's window. -/
def clusterParts (dir : Directory) (sm cta : Nat) (req : Req) (space := Space.shared) :
    Except String (Array (Nat × Nat × Array Nat × Req)) := do
  let some (cl, _) := dir.rank.get? (sm, cta)
    | throw s!"shared::cluster access from a CTA with no cluster rank"
  let mut groups : Array (Nat × Array Nat) := #[]
  for lane in [0:req.addrs.size] do
    if let some a := req.addrs[lane]! then
      let k := clusterRank a
      match groups.findIdx? (·.1 == k) with
      | some i => groups := groups.modify i fun (k, ls) => (k, ls.push lane)
      | none => groups := groups.push (k, #[lane])
  groups.mapM fun (k, lanes) => do
    let some (psm, pcta) := dir.slot.get? (cl, k)
      | throw s!"shared::cluster access to rank {k}, not in the cluster"
    let addrs := (List.range req.addrs.size).toArray.map fun l =>
      if lanes.contains l then req.addrs[l]!.map clusterOffset else none
    return (psm, pcta, lanes, { req with space, addrs })

/-- Send request `id`, past the LSU at `t`, to the stage its space starts at. -/
private def route (m : SmMem) (t id : Nat) (dir : Directory) : Except String SmMem := do
  let c := m.cfg
  let r := m.reqs[id]!
  let req := r.req
  let key : Key := { sm := m.sm, req := id }
  let active := req.addrs.any (·.isSome)
  let mut m := if r.async then m else m.releaseHeld t r.origin.slot
  match req.space with
  | .global =>
    let groups := coalesce c.sectorBytes c.lineBytes req.bytes req.addrs
    if active then m := m.record t (.issue key (m.agent r) (m.sm, r.origin.cta) req)
    m := (m.modReq id ({ · with parts := groups.size, toOrder := groups.size + 1 })).passed id
    if groups.isEmpty then m := m.push t (.done id)
    for g in groups do
      if r.async && req.op.writes then m := m.asyncWriteLine t id g
      else if r.async then m := m.push t (.tag id g)
      else
        let i := SmRes.tag.index c
        let (tt, p) := m.ports[i]!.reserve t (tagCycles c g)
        m := { m with ports := m.ports.set! i p }.push tt (.tag id g)
  | .«local» =>
    -- Local memory lives in device memory, interleaved so that consecutive 32-bit words belong to
    -- consecutive lanes (CUDA Programming Guide, "Local Memory"): the lines it occupies there. An
    -- 8- or 16-byte access touches that many interleaved words; a line performs the lanes whose
    -- first word it holds.
    let bytes := ((m.locals.getD r.origin.slot #[]).getD 0 default).size
    let window := (m.sm * 65536 + r.origin.slot) * 32 * ((bytes + 3) / 4 * 4)
    let at_ (lane : Nat) (a : UInt64) : UInt64 :=
      (localBase + window + a.toNat / 4 * 128 + lane * 4 + a.toNat % 4).toUInt64
    let wordsPer := max 1 (req.bytes / 4)
    let mut taddrs : Array (Option UInt64) := #[]
    for l in [0:req.addrs.size] do
      for k in [0:wordsPer] do taddrs := taddrs.push (req.addrs[l]!.map fun a => at_ l (a + (4 * k).toUInt64))
    let groups := (coalesce c.sectorBytes c.lineBytes (min req.bytes 4) taddrs).map fun g =>
      { g with lanes := (g.lanes.filter (· % wordsPer == 0)).map (· / wordsPer) }
    m := (m.modReq id ({ · with parts := groups.size, toOrder := groups.size + 1 })).passed id
    if groups.isEmpty then m := m.push t (.done id)
    for g in groups do
      let i := SmRes.tag.index c
      let (tt, p) := m.ports[i]!.reserve t (tagCycles c g)
      m := { m with ports := m.ports.set! i p }.push tt (.tag id g)
  | .shared =>
    if active then m := m.record t (.issue key (m.agent r) (m.sm, r.origin.cta) req)
    m := m.modReq id ({ · with parts := 1, toOrder := 1 })
    let lanes := (List.range req.addrs.size).toArray.filter (req.addrs[·]!.isSome)
    m := m.push t (.bank id r.origin.cta lanes key req)
  | .sharedCluster =>
    let parts ← clusterParts dir m.sm r.origin.cta req
    m := (m.modReq id ({ · with req := { req with addrs := req.addrs.map (·.map clusterOffset) },
                                  parts := parts.size, toOrder := parts.size + 1 })).passed id
    if parts.isEmpty then m := m.push t (.done id)
    for i in [0:parts.size] do
      let (psm, pcta, lanes, part) := parts[i]!
      let pkey := { key with part := i }
      m := m.record t (.issue pkey (m.agent r) (psm, pcta) part)
      -- A CTA on this SM, the issuer's own or a peer's, is reached through the SM's own data port.
      if psm == m.sm && m.cfg.races && pcta != r.origin.cta then
        m := { m with race := m.race.peerWrite pcta part lanes }
      if psm == m.sm then m := m.push t (.bank id pcta lanes pkey part)
      else m := m.push t (.send { key := pkey, req := part, to := psm, cta := pcta, lanes })
  | .tmem =>
    -- The SM's own tensor memory, and a CTA pair's peer's (`Tcgen05.peerByte`: past `2 ^ 24`),
    -- reached over the network as the peer's shared memory is.
    let lanes := (List.range req.addrs.size).toArray.filter fun l => match req.addrs[l]! with
      | some a => a < 2 ^ 24 | none => false
    let peer : Req := { req with addrs := req.addrs.map (·.bind fun a => if a < 2 ^ 24 then none else some (a - 2 ^ 24)) }
    let parts ← if peer.addrs.any (·.isSome) then clusterParts dir m.sm r.origin.cta peer .tmem else pure #[]
    let own := if lanes.isEmpty then 0 else 1
    m := (m.modReq id ({ · with parts := parts.size + own, toOrder := parts.size + own + 1 })).passed id
    if parts.size + own == 0 then m := m.push t (.done id)
    if own == 1 then m := m.push (t + c.tmemLatency) (.tmem id lanes)
    for i in [0:parts.size] do
      let (psm, pcta, plns, part) := parts[i]!
      if psm == m.sm then throw "tcgen05: a CTA pair on one SM"
      m := m.push t (.send { key := { key with part := i }, req := part, to := psm, cta := pcta, lanes := plns })
  | s => throw s!"{repr s} memory is not modelled yet"
  -- A generic access's lanes into the CTA's shared memory (`Req.sharedPart`): one more part, at
  -- the shared stage.
  if !req.sharedPart.isEmpty then
    let part : Req := { req with space := .shared, addrs := req.sharedPart, sharedPart := #[] }
    let lanes := (List.range part.addrs.size).toArray.filter (part.addrs[·]!.isSome)
    let pkey := { key with part := 1000 }
    if !lanes.isEmpty then m := m.record t (.issue pkey (m.agent r) (m.sm, r.origin.cta) part)
    -- The other space's path counted its parts (and finished the request if it had none).
    let r' := m.reqs[id]!
    m := m.modReq id ({ · with parts := r'.parts + 1, toOrder := r'.toOrder + 1 })
    m := m.push t (.bank id r.origin.cta lanes pkey part)
  return m

private def dispatch (m : SmMem) (t : Nat) (dir : Directory) : Except String SmMem := do
  let c := m.cfg
  let mut m := { m with armed := false }
  -- The shared stage takes the next request whose sub-partition has its address calculated, in
  -- round-robin order among those ready first (Huerta et al. 2025 §5.4: each sub-core computes
  -- addresses at its own rate, the shared structures accept from any of them), so a sub-partition
  -- still calculating does not hold the others back.
  let mut pick : Option Nat := none
  let mut best := t
  for k in [0:2 * c.smsps] do
    let qi := (m.rr + k) % (2 * c.smsps)
    let q := qi / 2
    if let some id := m.mio[qi]![0]? then
      if !m.gated id then
        -- A shared-memory store leaves its queue once the stores before it would have left the
        -- data stage as it arrives (`front` later): their backlog holds the queue, and through it
        -- the warps' issue (four warps streaming STS.128 run at the stage's rate). A store beside
        -- a load stream or the tensor core's operand reads is not held, nor are loads.
        let r := m.reqs[id]!.req
        let data := if r.space == .shared && r.op == .st then m.storesFree - min c.front m.storesFree else t
        let ready := max (max t data) m.ports[(SmRes.addr q).index c]!.free
        if pick.isNone || ready < best then
          pick := some qi; best := ready
  -- Every head waits on its warp: a delivery re-arms the LSU.
  let some qi := pick | return m
  if best > t then return m.arm best
  let q := qi / 2
  let id := m.mio[qi]![0]!
  m := { m with mio := m.mio.modify qi (·.eraseIdxIfInBounds 0), rr := qi + 1 }
  let r := m.reqs[id]!
  let w := m.warp r.origin.slot
  let acq := r.req.sem == .acquire || r.req.sem == .acqRel
  m := m.setWarp r.origin.slot { w with queued := w.queued - 1, inflight := w.inflight + 1,
                                        acquiring := w.acquiring + (if acq then 1 else 0) }
  let ai := (SmRes.addr q).index c
  -- A global store also moves its data through it (`storeRegCycles` per 32-bit register).
  let store := r.req.space == .global && (match r.req.op with | .st => true | _ => false)
  let (a, ap) := m.ports[ai]!.reserve t
    (c.addressInterval + if store then c.storeRegCycles * words r.req.bytes else 0)
  let li := SmRes.lsu.index c
  let (s0, lp) := m.ports[li]!.reserve a c.lsuInterval
  m := { m with ports := (m.ports.set! ai ap).set! li lp }
  m ← m.route (max s0 (r.issued + c.front)) id dir
  if m.mio.any (!·.isEmpty) then m := m.arm (t + 1)
  return m

/-- Accept a request from the SM's async unit (TMA, bulk copies, `LDGSTS`'s shared write) at `t`:
it skips the MIO queue and the LSU, a global request skips the L1 and goes to its slice, a shared
request goes to the data port, contending with the LSU's wavefronts. It delivers like any
request; the warp's fences do not wait for it (the async proxy is ordered by mbarriers and proxy
fences, PTX ISA §8.6). -/
def issueAsync (m : SmMem) (t : Nat) (o : Origin) (r : Req) (dir : Directory) (op : Nat)
    (tensor : Bool := false) (tensorShares : Nat := 1) (wgmma : Bool := false) : Except String (SmMem × Nat) := do
  let (m, id) := m.alloc { origin := o, req := r, resp := Array.replicate (r.addrs.size * words r.bytes) 0,
                           async := true, op, tensor, tensorShares, wgmma }
  -- The race check: the operation's writes, and the release its signals carry.
  let m := if !m.cfg.races then m else match r.op with
    | .mbar (.arrive ..) | .mbar (.completeTx _) => { m with race := m.race.agentSignal op id }
    | .mbar _ => m
    | .ld => { m with race := m.race.globalLoad o.cta r }
    | _ => { m with race := m.race.access t o.cta o.slot r (agent := some op) }
  let m := { m with ctaInflight := m.ctaInflight.insert o.cta (m.ctaInflight.getD o.cta 0 + 1) }
  return (← m.route t id dir, id)

/-- The lanes of `g` whose access starts in a sector of `mask`. -/
private def lanesIn (c : Config) (r : Req) (lanes : Array Nat) (mask : Nat) : Array Nat :=
  lanes.filter fun l => match r.addrs[l]! with
    | some a => (mask >>> (a.toNat % c.lineBytes / c.sectorBytes)) % 2 == 1
    | none => false

private def tagStage (m : SmMem) (t id : Nat) (g : LineAccess) (snapshot : Store) :
    Except String SmMem := do
  let c := m.cfg
  let r := m.reqs[id]!
  let req := r.req
  let n := popcount g.sectors
  let isLocal := req.space == .«local»
  -- A load that would miss on a new line while every MSHR is taken waits at the tag stage and
  -- tries again (`l1Mshrs`); while it waits, later accesses to that line wait behind it, so the
  -- SM's accesses to a line keep their order.
  if m.mshrBlocked.contains g.line && m.mshrBlocked.get? g.line != some id then
    return m.push (t + c.wavefrontCycles) (.tag id g)
  if req.op == .ld && !(bypassL1 req || r.async) && !m.mshr.contains g.line &&
      m.l1.present g.line g.sectors != g.sectors && m.mshr.size ≥ c.l1Mshrs then
    let m := { m with mshrBlocked := m.mshrBlocked.insert g.line id }
    return m.push (t + c.wavefrontCycles) (.tag id g)
  let m := if m.mshrBlocked.get? g.line == some id then { m with mshrBlocked := m.mshrBlocked.erase g.line } else m
  -- A write to a line with loads merged into a fill still in flight waits until they have
  -- performed: they perform when the fill returns, and must stay ahead of it.
  if req.op.writes && !r.async &&
      ((m.mshr.get? g.line).getD #[]).any (fun (tx, _) => !m.txns[tx]!.waiters.isEmpty) then
    return m.push (t + c.wavefrontCycles) (.tag id g)
  -- Local memory performs here, in issue order; its lines still cost the caches' time.
  let mut m ← if isLocal then m.performLocal id req g.lanes else pure m
  m := m.passed id
  match req.op with
  | .ld =>
    m := { m with counters := if isLocal then { m.counters with localLdSectors := m.counters.localLdSectors + n }
      else { m.counters with globalLdRequests := m.counters.globalLdRequests + 1,
                             globalLdSectors := m.counters.globalLdSectors + n } }
    if (bypassL1 req || r.async) && !isLocal then return m.sendLine t id g 0
    -- Sectors with a write of the SM unanswered go on behind it; sectors being filled for a
    -- single earlier miss wait for that fill; valid sectors with nothing unanswered hit.
    let written := m.busy g.line g.sectors true
    let valid := m.l1.present g.line g.sectors
    let pending := (m.mshr.get? g.line).getD #[]
    let mut merged := 0
    let mut waitOn : Array (Nat × Nat) := #[]
    for (tx, mask) in pending do
      -- The fill's sectors for which it is the SM's only request unanswered.
      let alone := mask ^^^ m.busy g.line mask false 2
      let want := g.sectors &&& alone
      let want := want ^^^ (want &&& (written ||| merged))
      if want != 0 then
        waitOn := waitOn.push (tx, want)
        merged := merged ||| want
    let unanswered := m.busy g.line g.sectors false
    let hit := valid ^^^ (valid &&& (unanswered ||| merged ||| written))
    let fresh := g.sectors ^^^ (hit ||| merged)
    m := { m with l1 := m.l1.touch g.line,
                  counters := { m.counters with globalLdHitSectors := m.counters.globalLdHitSectors + popcount hit } }
    let parts := (if hit != 0 then 1 else 0) + waitOn.size + (if fresh != 0 then 1 else 0)
    m := m.modReq id fun q => { q with parts := q.parts + parts - 1 }
    if hit != 0 then
      if !isLocal then
        let (m', _, _) ← m.performOn t snapshot r.origin.cta id { sm := m.sm, req := id } req
          (lanesIn c req g.lanes hit) false
        m := m'
      let (_, last, m') := m.take .data t c.wavefrontCycles
      m := m'.complete id (last + c.backL1)
    for (tx, want) in waitOn do
      m := { m with txns := m.txns.modify tx fun y =>
        { y with waiters := y.waiters.push (id, if isLocal then #[] else lanesIn c req g.lanes want) } }
    if fresh != 0 then
      let tid := m.freeTxns.back?.getD m.txns.size
      m := { m with mshr := m.mshr.insert g.line (pending.push (tid, fresh)) }
      m := m.sendLine t id { g with sectors := fresh, lanes := if isLocal then #[] else lanesIn c req g.lanes fresh } fresh
    return m
  | .st =>
    m := { m with counters := if isLocal then { m.counters with localStSectors := m.counters.localStSectors + n }
      else { m.counters with globalStRequests := m.counters.globalStRequests + 1,
                             globalStSectors := m.counters.globalStSectors + n } }
    if !r.async then
      m := { m with l1 := if c.storeInvalidates then m.l1.invalidate g.line g.sectors
                          else (m.l1.fill g.line g.sectors false).1 }
    return m.sendLine t id (if isLocal then { g with lanes := #[] } else g) 0
  | .atom .. | .red .. =>
    m := { m with counters := { m.counters with globalAtomSectors := m.counters.globalAtomSectors + n },
                  l1 := m.l1.invalidate g.line g.sectors }
    return m.sendLine t id g 0
  | .mbar _ => throw "an mbarrier object lives in shared memory (PTX ISA 9.5 §9.7.15.16.1)"

/-- The data port's wavefronts for `lanes` of `r`, the cycles they take, and the bank conflicts. -/
private def bankCost (c : Config) (r : Req) (lanes : Array Nat) : Nat × Nat × Nat :=
  let addrs := (List.range r.addrs.size).toArray.map fun l => if lanes.contains l then r.addrs[l]! else none
  let (w, ideal) := wavefronts c.banks c.bankBytes r.bytes addrs
  let per := match r.op with | .atom .. | .red .. | .mbar _ => c.smemAtomInterval | _ => c.wavefrontCycles
  (w, w * per, w - ideal)

private def countBank (m : SmMem) (w conflicts : Nat) : SmMem :=
  { m with counters := { m.counters with sharedWavefronts := m.counters.sharedWavefronts + w,
                                         sharedBankConflicts := m.counters.sharedBankConflicts + conflicts } }

/-- Perform `lanes` of `r` on CTA `cta`'s shared memory, as part `key`: the SM and the answers. -/
private def performShared (m : SmMem) (t cta id : Nat) (key : Key) (r : Req) (lanes : Array Nat)
    (answer := true) : Except String (SmMem × Array (Nat × Array UInt32)) := do
  let s := m.shared[cta]!
  let m := { m with shared := m.shared.set! cta default }
  let (m, s, out) ← m.performOn t s cta id key r lanes true answer
  let mut m := { m with shared := m.shared.set! cta s }
  -- A write to a barrier a `try_wait` is held on: the wait looks again (`bankStage`), its wake
  -- (`Config.tryWaitWake`) later.
  if r.op.writes && !m.held.isEmpty then
    for lane in lanes do
      if let some a := (r.addrs[lane]?).join then
        if let some hs := m.held.get? (cta, a.toNat) then
          m := { m with held := m.held.erase (cta, a.toNat) }
          for h in hs do m := m.push (t + m.cfg.tryWaitWake) (.bank h.req h.cta h.lanes h.key h.r)
  return (m, out)

/-- `r` is a `try_wait` (`SASS.phaseChk`: hint "TRYWAIT" and "want" the phase bit) whose lanes
all read its barrier still in the phase they wait past. -/
private def tryWaitHeld (m : SmMem) (cta : Nat) (r : Req) (lanes : Array Nat) : Bool := Id.run do
  let some want := (r.hint.ops.find? (·.startsWith "want")).bind fun o => (o.drop 4).toNat? | return false
  let some s := m.shared[cta]? | return false
  for lane in lanes do
    let some l := Flat.loc cta r lane | return false
    match Flat.perform r lane l s with
    | .ok (_, read) =>
      let ws := laneWords r read
      if (ws[1]?.getD 0 >>> 31).toNat != want then return false
    | .error _ => return false
  return true

private def bankStage (m : SmMem) (t id cta : Nat) (lanes : Array Nat) (key : Key) (r : Req) :
    Except String SmMem := do
  let c := m.cfg
  let (w, cost, conflicts) := bankCost c r lanes
  -- A tensor-core operand read takes its own port, at its conflict-free wavefronts.
  if m.reqs[id]!.tensor then
    let busy := ((w - conflicts) * c.wavefrontCycles + m.reqs[id]!.tensorShares - 1) / max m.reqs[id]!.tensorShares 1
    let m := { m with counters := { m.counters with tensorPortMax := m.counters.tensorPortMax + busy } }
    let (start, last, m) := m.take .tensor t busy
    -- On a part where a warpgroup MMA's reads share the data stage with the warps' accesses
    -- (`Config.wgmmaOperandData`), they take it too, when the tensor core starts on them: the
    -- warps' loads and stores then wait for the MMA being read, not for every queued MMA's reads
    -- (a burst of four HGMMAs left FA3's shared loads 400 cycles behind it). They wait for nothing
    -- themselves (wgmma beside a TMA stream runs at its rate).
    let mut m := m
    if c.wgmmaOperandData > 0 && m.reqs[id]!.wgmma then
      m := m.push start (.dataShare ((busy * c.wgmmaOperandData + 999) / 1000))
    let (m', _) ← m.performShared t cta id key r lanes
    return (m'.passed id).complete id (last + c.backShared)
  let (_, last, m) := m.take .data t cost
  let m := if r.op == .st then { m with storesFree := max m.storesFree last } else m
  -- `try_wait` (SASS `SYNCS.PHASECHK.TRANS64.TRYWAIT`, marked by `SASS.phaseChk`): while its
  -- barrier is in the phase it waits past, held until a write to the barrier or its time limit
  -- (GH200, B200: a waiter resumes 93 cycles after a late arrival, an unfinished wait returns
  -- false after 1941).
  if c.tryWaitLimit > 0 && r.hint.ops.contains "TRYWAIT" then
    let first := m.reqs[id]!.heldUntil.isNone
    let deadline := m.reqs[id]!.heldUntil.getD (t + c.tryWaitLimit)
    -- Not while another request of the warp is in flight: the request machine would see it
    -- perform behind the wait's acquire (`releaseHeld` for those that pass the LSU later).
    let alone := (m.warp m.reqs[id]!.origin.slot).inflight ≤ 1
    if t < deadline && alone && m.tryWaitHeld cta r lanes then
      if let some a := (lanes[0]?.bind fun l => (r.addrs[l]?).join) then
        let k := (cta, a.toNat)
        let m := m.modReq id fun q => { q with heldUntil := some deadline }
        let m := { m with held := m.held.insert k ((m.held.getD k #[]).push { req := id, cta, lanes, key, r, deadline }) }
        return if first then m.push deadline (.tryWaitEnd cta a.toNat id deadline) else m
  let (m, _) ← (m.countBank w conflicts).performShared t cta id key r lanes
  return (m.passed id).complete id (last + c.backShared)

/-- A held `try_wait` reaching its limit: its data-port event runs again, and answers. -/
private def tryWaitEnd (m : SmMem) (t cta addr id deadline : Nat) : SmMem :=
  match (m.held.get? (cta, addr)).bind fun hs => hs.find? fun h => h.req == id && h.deadline == deadline with
  | none => m
  | some h =>
    let rest := (m.held.getD (cta, addr) #[]).filter fun h => !(h.req == id && h.deadline == deadline)
    let m := { m with held := if rest.isEmpty then m.held.erase (cta, addr) else m.held.insert (cta, addr) rest }
    m.push t (.bank h.req h.cta h.lanes h.key h.r)

/-- A tensor-memory request performs, lane by lane in lane order, and delivers. -/
private def tmemStage (m : SmMem) (t id : Nat) (lanes : Array Nat) : Except String SmMem := do
  let r := m.reqs[id]!.req
  let mut s := m.tmem
  let mut m := { m with tmem := default }
  for lane in lanes do
    let (s', ws) ← s.perform r lane
    s := s'
    if r.op.returns then m := m.answerLane id lane ws
  return ({ m with tmem := s }.passed id).complete id t

/-- Another SM's cluster request part reaches this SM's data port: performed, and answered over
the network. -/
private def remoteStage (m : SmMem) (t : Nat) (x : PeerReq) : Except String SmMem := do
  -- A CTA pair's tensor-memory access, at this SM's tensor memory.
  if x.req.space == .tmem then
    let mut s := m.tmem
    let mut out : Array (Nat × Array UInt32) := #[]
    for lane in x.lanes do
      let (s', ws) ← s.perform x.req lane
      s := s'
      out := out.push (lane, ws)
    let m := { m with tmem := s }
    return m.send (t + m.cfg.tmemLatency) (.peerAnswer x.key (if x.req.op.returns then out else #[]))
  let (w, cost, conflicts) := bankCost m.cfg x.req x.lanes
  let (_, last, m) := m.take .data t cost
  let m := if m.cfg.races then { m with race := m.race.peerWrite x.cta x.req x.lanes } else m
  -- The answer goes back over the network, even to a peer CTA on this SM.
  let (m, out) ← (m.countBank w conflicts).performShared t x.cta 0 x.key x.req x.lanes (answer := false)
  return m.send last (.peerAnswer x.key (if x.req.op.returns then out else #[]))

/-- A cluster request part leaves for its SM over the network port. -/
private def sendStage (m : SmMem) (t : Nat) (x : PeerReq) : SmMem :=
  let bytes := x.lanes.size * x.req.bytes
  let (_, last, m) := m.take .dsOut t bytes
  let m := { m with counters := { m.counters with dsmemBytes := m.counters.dsmemBytes + bytes } }
  (m.passed x.key.req).send last (.peer x)

/-- A message from the other half arrives at `t`. -/
private def arriveStage (m : SmMem) (t : Nat) (msg : Msg) (snapshot : Store) : Except String SmMem := do
  let c := m.cfg
  match msg with
  | .answer txn resp =>
    let x := m.txns[txn]!
    let r := m.reqs[x.req]!
    let mut m := { m with freeTxns := m.freeTxns.push txn }.countSectors x.line x.sectors x.write (-1)
    let bytes := match r.req.op with
      | .ld => popcount x.sectors * c.sectorBytes
      | .atom .. => resp.size * r.req.bytes
      | _ => 0
    let (_, l, m') := m.take .xin t bytes
    m := m'
    if r.code then return m.complete x.req l
    for (lane, ws) in resp do m := m.answerLane x.req lane ws
    if !r.req.op.returns then return m.complete x.req l
    if x.fills != 0 then
      let rest := ((m.mshr.get? x.line).getD #[]).filter (·.1 != txn)
      m := { m with l1 := (m.l1.fill x.line x.fills false).1,
                    mshr := if rest.isEmpty then m.mshr.erase x.line else m.mshr.insert x.line rest }
    let (_, last, m') := m.take .data l c.wavefrontCycles
    let ready := last + c.backL1
    m := m'.complete x.req ready
    -- Loads merged into it perform now, on the memory as the quantum began, after the fill.
    for (w, lanes) in x.waiters do
      let q := m.reqs[w]!
      let (m', _, _) ← m.performOn t snapshot q.origin.cta w { sm := m.sm, req := w } q.req lanes false
      m := m'.complete w ready
    return m
  | .peer x => return m.push t (.remote x)
  | .peerAnswer key resp =>
    let r := m.reqs[key.req]!
    let (_, l, m) := m.take .dsIn t (resp.size * r.req.bytes)
    let m := resp.foldl (fun m (lane, ws) => m.answerLane key.req lane ws) m
    return m.complete key.req (l + c.backShared)
  | .line _ => throw "a line request delivered to an SM"

/-- Request `id` reaches its issuer at `t`. -/
private def finish (m : SmMem) (t id : Nat) : SmMem × Delivery := Id.run do
  let r := m.reqs[id]!
  let o := r.origin
  let d : Delivery := { id, origin := o, at_ := t, resp := r.resp, fence := r.fence }
  let mut m := { m with reqs := m.reqs.set! id default, delivered := m.delivered.push id }
  if r.fence then return (m.record t (.fence (.warp m.sm o.slot o.cta)), d)
  if r.code then return (m, d)
  let n := m.ctaInflight.getD o.cta 1 - 1
  m := { m with ctaInflight := if n == 0 then m.ctaInflight.erase o.cta else m.ctaInflight.insert o.cta n }
  if n == 0 && m.retired.contains o.cta then m := m.free o.cta
  if r.async then return (m, d)
  let w := m.warp o.slot
  let acq := r.req.sem == .acquire || r.req.sem == .acqRel
  let mut w := { w with inflight := w.inflight - 1, acquiring := w.acquiring - (if acq then 1 else 0) }
  if w.queued == 0 && w.inflight == 0 then
    for (f, cost) in w.fences do m := m.push (t + cost) (.done f)
    w := { w with fences := #[] }
  m := m.setWarp o.slot w
  return (m.arm t, d)

private def step (m : SmMem) (t : Nat) (snapshot : Store) (dir : Directory) :
    SmEv → Except String (SmMem × Option Delivery)
  | .dispatch => do return (← m.dispatch t dir, none)
  | .tag id g => do return (← m.tagStage t id g snapshot, none)
  | .bank id cta lanes key r => do return (← m.bankStage t id cta lanes key r, none)
  | .remote x => do return (← m.remoteStage t x, none)
  | .tmem id lanes => do return (← m.tmemStage t id lanes, none)
  | .dataShare n => pure ((m.take .data t n).2.2, none)
  | .tryWaitEnd cta addr id deadline => pure (m.tryWaitEnd t cta addr id deadline, none)
  | .send x => pure (m.sendStage t x, none)
  | .arrive msg => do return (← m.arriveStage t msg snapshot, none)
  | .done id => let (m, d) := m.finish t id; pure (m, some d)

/-- Run the SM's events up to cycle `t`, global memory being `snapshot` (as the current quantum
began): the requests delivered by then, in delivery order. -/
def advance (m : SmMem) (t : Nat) (snapshot : Store) (dir : Directory) :
    Except String (SmMem × Array Delivery) := do
  let mut m := { m with freeReqs := m.freeReqs ++ m.delivered, delivered := #[] }
  let mut out := #[]
  -- Each event is taken once; the bound only guards a defect.
  for _ in [0:1000000000] do
    match m.events.minKey? with
    | some at_ => if at_ > t then break
    | none => break
    let h := m.events
    m := { m with events := {} }
    let some (at_, e, h) := h.pop? | break
    let (m', d) ← { m with events := h }.step at_ snapshot dir e
    m := m'
    if let some d := d then out := out.push d
  return (m, out)

/-- What left the SM, and what it did to the request machine, since last asked. -/
def takeOut (m : SmMem) : SmMem × Array (Stamped Msg) × Array (Stamped Log) :=
  ({ m with outbox := #[], log := #[] }, m.outbox, m.log)

/-- A message for this SM, taken when its stamp comes. -/
def deliverIn (m : SmMem) (msg : Stamped Msg) : SmMem := m.push msg.stamp (.arrive msg.body)

end SmMem

end NV.Core.Memory
