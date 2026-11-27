import NV.Core.Memory.Sm
import NV.Core.Memory.L2

/-!
# The memory system

What happens to a memory request between the cycle a warp issues it and the cycle its response
reaches the warp's registers. The engine sees a few operations: `reserve` an MIO slot when a
memory instruction issues and `issue` its request when it reads its sources, `fence`, ask for
the `next` cycle anything happens, and `advance` to a cycle, which returns the requests delivered
by then. Data are computed only by the request machine (`NV.Memory.Flat`) and the stores; this
module decides when each lane performs and when each request delivers.

## The path of a request

```
issue ─► MIO queue (per sub-partition, mioQueueDepth) ─► address port (per sub-partition)
      ─► LSU port (per SM) ─┬─ global: coalescer ─► tag port ─► L1 tags ─┬─ hit ─► data port ─► registers
                            │                                            └─ miss/store/atomic ─► crossbar port
                            │      ─► (far line: copy slice ─► partition link) ─► L2 slice ─┬─ hit
                            │                                                              └─ miss ─► DRAM channel
                            │      ─► crossbar back ─► data port (L1 fill) ─► registers
                            ├─ shared: data port, one cycle per bank wavefront ─► registers
                            └─ shared::cluster: SM-to-SM port ─► peer's data port ─► back
```

Every rate is a `Port` (`NV.Core.Port`, through `Port.take` for fractional rates), so the bound
invariants (`Port.served_le`, `Rate.served_le`) hold of every resource. Ports are reserved when
work arrives at them, events taken in cycle order, so each port serves in arrival order.

## Two halves, in quanta

The SMs' side (`NV.Core.Memory.Sm`) and the L2 side (`NV.Core.Memory.L2`) meet only through
stamped messages that take at least `Config.lookahead` cycles to cross. Time advances in quanta of
`quantum ≤ lookahead` cycles: in phase A every SM runs the quantum on its own (in any order, or in
parallel, with the same result); in phase B the L2 side replays what the SMs did to the request
machine, takes their messages in stamp order and runs the quantum. Its answers arrive in later
quanta.

## When data move

A lane performs at one instant between its issue and its delivery: a global access at the L2
slice that serves it, or at the L1 tag stage for a hit; a shared access at its CTA's SM's data
port; a local access at the tag stage. An SM's requests to a sector reach the place they perform
in the order the SM dispatched them, and a warp's requests are dispatched in issue order, so two
accesses of one thread to one address perform in program order. A fence delivers only after
every earlier request of its warp has delivered, a release request dispatches only after them,
and an acquire request holds back its warp's later requests until it delivers. The request
machine checks all of this as it replays the run, and its soundness theorem (`Mem.sound`) makes
every run an execution the PTX memory model allows.

## Latencies

The parameters give end-to-end latencies on an idle machine; the segments between stages are
set from them (inferred): instruction issue to tag stage `front = smemLatency / 2` (a request
reaches the MIO queue when its instruction reads its sources, often about `front` after issue,
and passes the tag stage no earlier than `front` after issue); tag stage to registers
`l1HitLatency - front`; shared data port to registers `smemLatency - front`; the L1 miss adds
`l2NearHitLatency - l1HitLatency` (or far) split between the request and the response legs; an L2
miss adds `dramLatency - (l2NearHitLatency + l2FarHitLatency) / 2` (DRAM latency is the near/far
average); a cluster access adds `dsmemLatency - smemLatency`, split between the legs.

Local memory goes through the L1 and L2 at the device addresses its interleaving gives it
(consecutive 32-bit words to consecutive lanes, CUDA Programming Guide, "Local Memory").

Not modelled yet: L1 MSHR exhaustion (512 entries), DRAM row-buffer timing, texture and constant
paths; instruction fetches do not reach the L2 here (the engine's code caches time them).
-/

namespace NV.Core

open NV Memory

structure MemSys where
  cfg : Config
  sms : Array SmMem
  l2 : L2Mem
  dir : Directory := {}
  /-- Cycles per quantum, and the cycle the current quantum began. -/
  quantum : Nat
  start : Nat := 0
  /-- The functional mode over several workers: each worker's global writes (`performNow`), with
  their CTA, for the master to make on global memory as the quantum ends. -/
  logGlobal : Bool := false
  globalLog : Array (Nat × Req) := #[]
  deriving Inhabited

namespace MemSys

/-- A memory system of `sms` SMs over `global`, its L2 holding `l2` from an earlier launch on the
same machine. L1s start empty: a launch invalidates them. -/
def init (c : Config) (sms : Nat) (global : Store) (l2 : Array Tags := #[]) : MemSys :=
  { cfg := c, sms := (Array.range sms).map (SmMem.init c), l2 := L2Mem.init c global l2,
    quantum := max 1 c.lookahead }

/-- Global memory. -/
def global (m : MemSys) : Store := m.l2.flat.global

/-- What the L2 holds, for the next launch on the machine. -/
def l2Tags (m : MemSys) : Array Tags := m.l2.l2

/-- The last cycle the L2 side is busy with work already taken (`L2Mem.busyUntil`). -/
def l2BusyUntil (m : MemSys) : Nat := m.l2.busyUntil

/-- Bytes the launch writes to device memory (a module's code image, the parameter block): lines
`lines` pass through their L2 slices on the way, so they are there (clean) until evicted. -/
def loadCode (m : MemSys) (lines : List Nat) : MemSys :=
  let c := m.l2.cfg
  let all := (1 <<< (c.lineBytes / c.sectorBytes)) - 1
  let l2 := lines.foldl (fun l2 line => l2.modify (spread line c.slices) fun t => (t.fill line all false).1) m.l2.l2
  { m with l2 := { m.l2 with l2 } }

/-- The shared memory of CTA `cta` on `sm`. -/
def sharedOf (m : MemSys) (sm cta : Nat) : Store := m.sms[sm]!.shared[cta]!

def counters (m : MemSys) : Counters := m.sms.foldl (fun k s => k.add s.counters) m.l2.counters

/-- CTA `cta` starts on `sm` as rank `rank` of cluster `cluster`, with `bytes` of shared memory. -/
def addCta (m : MemSys) (sm cta cluster rank bytes : Nat) (init : Array (Nat × UInt32) := #[])
    (reserved := 0) : MemSys :=
  { m with sms := m.sms.modify sm (·.addCta cta bytes init reserved), l2 := m.l2.addCta sm cta bytes init,
           dir := { slot := m.dir.slot.insert (cluster, rank) (sm, cta),
                    rank := m.dir.rank.insert (sm, cta) (cluster, rank) } }

/-- The L2 side's and the directory's part of `addCta`, for a CTA an SM placed on its own copy of
the memory system during a quantum. -/
def addCtaL2 (m : MemSys) (sm cta cluster rank bytes : Nat) (init : Array (Nat × UInt32) := #[]) :
    MemSys :=
  { m with l2 := m.l2.addCta sm cta bytes init,
           dir := { slot := m.dir.slot.insert (cluster, rank) (sm, cta),
                    rank := m.dir.rank.insert (sm, cta) (cluster, rank) } }

/-- Warp `slot` of CTA `cta` starts on `sm`, each lane with `bytes` of local memory. -/
def addWarp (m : MemSys) (sm cta slot bytes : Nat) : MemSys :=
  { m with sms := m.sms.modify sm (·.addWarp cta slot bytes) }

/-- A CTA has finished: its memory on the SM is freed once its requests in flight have
delivered. Its cluster rank stays known for those requests (a `shared::cluster` request it issued
before its warps exited, routed after), while no other CTA reaches it any more; a CTA's index on its SM is not reused. -/
def removeCta (m : MemSys) (sm cta : Nat) : MemSys :=
  let dir := match m.dir.rank.get? (sm, cta) with
    | some k => { m.dir with slot := m.dir.slot.erase k }
    | none => m.dir
  { m with sms := m.sms.modify sm (·.removeCta cta), dir }

def accepts (m : MemSys) (sm smsp queue : Nat) : Bool := m.sms[sm]!.accepts smsp queue

def reserve (m : MemSys) (sm smsp cta queue : Nat) : MemSys :=
  { m with sms := m.sms.modify sm (·.reserve smsp cta queue) }

/-- Take SM `sm`'s side out, so it is updated in place, and put it back. -/
@[inline] private def withSm {α : Type} (m : MemSys) (sm : Nat) (f : SmMem → SmMem × α) : MemSys × α :=
  let s := m.sms[sm]!
  let m := { m with sms := m.sms.set! sm default }
  let (s, a) := f s
  ({ m with sms := m.sms.set! sm s }, a)

/-- Whether warp `slot`'s requests are all past its SM's ordering point: its barrier arrival may
count. -/
def performed (m : MemSys) (sm slot : Nat) : Bool := m.sms[sm]!.ordered slot

/-- A request's id across the GPU: each SM numbers its own requests, so the SM is folded in. -/
private def globalId (m : MemSys) (sm id : Nat) : Nat := id * m.sms.size + sm

private def globalIds (m : MemSys) (ds : Array Delivery) : Array Delivery :=
  ds.map fun d => { d with id := m.globalId d.origin.sm d.id }

def issue (m : MemSys) (t : Nat) (o : Origin) (r : Req) (issued : Nat := t) : MemSys × Nat :=
  let (m', id) := m.withSm o.sm (·.issue t o r issued)
  (m', m.globalId o.sm id)

def issueAsync (m : MemSys) (t : Nat) (o : Origin) (r : Req) (op : Nat) (tensor : Bool := false)
    (tensorShares : Nat := 1) (wgmma : Bool := false) : Except String (MemSys × Nat) := do
  let s := m.sms[o.sm]!
  let m := { m with sms := m.sms.set! o.sm default }
  let (s, id) ← s.issueAsync t o r m.dir op tensor tensorShares wgmma
  return ({ m with sms := m.sms.set! o.sm s }, m.globalId o.sm id)

/-- When request `id` (as `issue` and `issueAsync` number it) of SM `sm` sent its last data to
the L2: known once it is issued, since a request takes the SM's output port as it is routed. -/
def departAt (m : MemSys) (sm id : Nat) : Nat :=
  ((m.sms[sm]!).reqs[id / m.sms.size]?.map (·.departAt)).getD 0

def fence (m : MemSys) (t : Nat) (o : Origin) (cost : Nat) (drain : Option (Nat × Nat) := none) :
    MemSys × Nat :=
  let (m', id) := m.withSm o.sm (·.fence t o cost drain)
  (m', m.globalId o.sm id)

/-- A code line's refill for SM `o.sm`'s instruction caches (`SmMem.issueCode`). -/
def issueCode (m : MemSys) (t : Nat) (o : Origin) (addr : Nat) : MemSys × Nat :=
  let (m', id) := m.withSm o.sm (·.issueCode t o addr)
  (m', m.globalId o.sm id)

/-- Whether anything has left the SMs that phase B has not taken. -/
private def waiting (m : MemSys) : Bool := m.l2.next.isSome || m.sms.any (fun s => !s.outbox.isEmpty || !s.log.isEmpty)

/-- The earliest cycle at which anything happens: an SM's event, or the end of the quantum when
the L2 side has work. -/
def next (m : MemSys) : Option Nat :=
  let q := if m.waiting then some (m.start + m.quantum) else none
  m.sms.foldl (fun n s => match s.next, n with
    | some a, some b => some (min a b)
    | some a, none => some a
    | none, n => n) q

/-- Run every SM's events up to `t`: the deliveries. -/
private def runSms (m : MemSys) (t : Nat) : Except String (MemSys × Array Delivery) := do
  let snapshot := m.global
  let mut sms := m.sms
  let mut out := #[]
  for i in [0:sms.size] do
    if sms[i]!.next.all (· > t) then continue
    let s := sms[i]!
    sms := sms.set! i default
    let (s, ds) ← s.advance t snapshot m.dir
    sms := sms.set! i s
    out := out ++ ds
  return ({ m with sms }, out)

/-- The memory system with only SMs `ids`' sides (the others default): what one task of a quantum
needs. -/
def onlySms (m : MemSys) (ids : Array Nat) : MemSys :=
  { m with sms := ids.foldl (fun a i => a.set! i m.sms[i]!) (Array.replicate m.sms.size default) }

/-- The memory system without SMs `ids`' sides (they moved to a quantum's tasks). -/
def withoutSms (m : MemSys) (ids : Array Nat) : MemSys :=
  { m with sms := ids.foldl (fun a i => a.set! i default) m.sms }

/-- SM `si`'s side as `from` has it. -/
def setSm (m : MemSys) (si : Nat) (from_ : MemSys) : MemSys := { m with sms := m.sms.set! si from_.sms[si]! }

/-- The race check's async-proxy reads and proxy fences (`SmMem.raceAsyncReads`). -/
def raceAsyncReads (m : MemSys) (t si cta slot : Nat) (rs : Array Req) : MemSys :=
  if m.cfg.races then { m with sms := m.sms.modify si (·.raceAsyncReads t cta slot rs) } else m

/-- The race check's async operations (`SmMem.raceAgentStart`, `SmMem.raceAgentsDone`). -/
def raceAgentStart (m : MemSys) (o : Origin) (a : Nat) (deps : Array Nat := #[]) : MemSys :=
  if m.cfg.races then { m with sms := m.sms.modify o.sm (·.raceAgentStart o.slot a deps) } else m

def raceAgentReads (m : MemSys) (t : Nat) (o : Origin) (a : Nat) (rs : Array Req) : MemSys :=
  if m.cfg.races then { m with sms := m.sms.modify o.sm (·.raceAgentReads t o.cta o.slot a rs) } else m

def raceAgentsDone (m : MemSys) (si slot : Nat) (ops : Array Nat) : MemSys :=
  if m.cfg.races then { m with sms := m.sms.modify si (·.raceAgentsDone slot ops) } else m

def raceProxyFence (m : MemSys) (si slot : Nat) : MemSys :=
  if m.cfg.races then { m with sms := m.sms.modify si (·.raceProxyFence slot) } else m

/-- Warp `slot` of CTA index `cta` of SM `si` arrives at barrier `id` (the race check's release). -/
def barArrive (m : MemSys) (si cta id slot : Nat) : MemSys :=
  if m.cfg.races then { m with sms := m.sms.modify si fun s => { s with race := s.race.barArrive cta id slot } } else m

/-- Barrier `id` of CTA index `cta` of SM `si` completes, its waiting warps `waiters` going on. -/
def barRelease (m : MemSys) (si cta id : Nat) (waiters : Array Nat) : MemSys :=
  if m.cfg.races then { m with sms := m.sms.modify si fun s => { s with race := s.race.barRelease cta id waiters } } else m

/-- The races SM `si` found. -/
def races (m : MemSys) (si : Nat) : Array Race := m.sms[si]!.race.races

/-- SM `si`'s next event. -/
def nextSm (m : MemSys) (si : Nat) : Option Nat := m.sms[si]!.next

/-- Whether the L2 side has work for the quantum's end. -/
def l2Waiting (m : MemSys) : Bool := m.waiting

/-- Phase A for one SM: run SM `si`'s events up to `t`, against the global memory as the quantum
began. The deliveries, with ids across the GPU. -/
def advanceSm (m : MemSys) (si t : Nat) : Except String (MemSys × Array Delivery) := do
  let snapshot := m.global
  let s := m.sms[si]!
  let m := { m with sms := m.sms.set! si default }
  let (s, ds) ← s.advance t snapshot m.dir
  let m := { m with sms := m.sms.set! si s }
  return (m, m.globalIds ds)

/-- What a worker running SMs `ids` needs of the memory system: their halves, global memory as
the quantum begins, the directory and the quantum's start. The L2 side is the master's alone (an
empty one stands in, so the worker shares none of it). -/
def forWorker (m : MemSys) (ids : Array Nat) : MemSys :=
  { m with sms := ids.foldl (fun a i => a.set! i m.sms[i]!) (Array.replicate m.sms.size default),
           l2 := { (default : L2Mem) with flat := { (default : Flat) with global := m.global } } }

/-- A worker's memory system at a quantum's start: global memory as it begins, the directory, and
the quantum's start. -/
def atQuantum (m : MemSys) (global : Store) (dir : Directory) (start : Nat) : MemSys :=
  { m with l2 := { m.l2 with flat := { m.l2.flat with global } }, dir, start }

/-- SM `si`'s messages and records for the L2 side, taken out of it. -/
def takeOutSm (m : MemSys) (si : Nat) : MemSys × Array (Stamped Msg) × Array (Stamped Log) :=
  m.withSm si fun s => let (s, o, l) := s.takeOut; (s, (o, l))

/-- The L2 side's answers, into the SMs they are for. -/
def deliverAnswers (m : MemSys) (answers : Array (Stamped Msg)) : MemSys :=
  answers.foldl (fun m a => { m with sms := m.sms.modify a.sm (·.deliverIn a) }) m

/-- Phase B at the quantum's end `stop`, on the SMs' messages and records taken out by their
workers: the L2 side runs to `stop`; its answers, for later quanta. -/
def endQuantumWith (m : MemSys) (msgs : Array (Stamped Msg)) (logs : Array (Stamped Log))
    (stop : Nat) : Except String (MemSys × Array (Stamped Msg)) := do
  let l2 := m.l2
  let m := { m with l2 := default }
  let (l2, answers) ← l2.run logs msgs stop
  for a in answers do
    if a.stamp < stop then
      throw s!"an L2 answer stamped {a.stamp} for SM {a.sm}, before the quantum's end {stop}"
  return ({ m with l2, start := stop }, answers)

/-- Phase B at the quantum's end `stop`: the L2 side takes every SM's messages and records in
stamp order, runs to `stop`, and its answers go to the SMs for later quanta. -/
def endQuantum (m : MemSys) (stop : Nat) : Except String MemSys := do
  let mut m := m
  let mut msgs := #[]
  let mut logs := #[]
  for i in [0:m.sms.size] do
    let (m', o, l) := m.takeOutSm i
    m := m'
    msgs := msgs ++ o
    logs := logs ++ l
  let (m', answers) ← m.endQuantumWith msgs logs stop
  return m'.deliverAnswers answers

/-- Run to cycle `t`: every quantum that ends by `t` in full (phase A, then phase B), then the SMs
up to `t`. The requests delivered by then. -/
def advance (m : MemSys) (t : Nat) : Except String (MemSys × Array Delivery) := do
  let mut m := m
  let mut out := #[]
  -- Each pass ends one quantum; the bound only guards a defect.
  for _ in [0:1000000000] do
    let stop := m.start + m.quantum
    if stop > t then break
    -- Nothing before `t` anywhere: skip to the last quantum boundary before it.
    if !m.waiting && m.sms.all (·.next.all (· > t)) then
      m := { m with start := m.start + (t - m.start) / m.quantum * m.quantum }
      break
    let (m', ds) ← m.runSms (stop - 1)
    m := m'
    out := out ++ ds
    let mut msgs := #[]
    let mut logs := #[]
    for i in [0:m.sms.size] do
      let (m', (o, l)) := m.withSm i fun s => let (s, o, l) := s.takeOut; (s, (o, l))
      m := m'
      msgs := msgs ++ o
      logs := logs ++ l
    let l2 := m.l2
    m := { m with l2 := default }
    let (l2, answers) ← l2.run logs msgs stop
    m := { m with l2, start := stop }
    -- The SMs have run to `stop`; an answer stamped earlier would change their past, so the
    -- quantum would not be a valid lookahead.
    for a in answers do
      if a.stamp < stop then
        throw s!"an L2 answer stamped {a.stamp} for SM {a.sm}, before the quantum's end {stop}"
      m := { m with sms := m.sms.modify a.sm (·.deliverIn a) }
  let (m', ds) ← m.runSms t
  return (m', m'.globalIds (out ++ ds))

/-- Lane `lane` of `r` performed at once on store `st` at location `l`: the new store and the lane's
response words. -/
private def performLane (st : Store) (r : Req) (lane : Nat) (l : Loc) :
    Except String (Store × Array UInt32) := do
  let (st, rd) ← Flat.perform r lane l st
  return (st, laneWords r rd)

/-- The functional mode's memory (`Launch.functional`): request `r` of the warp in `slot` of CTA
`cta` on SM `sm` performed at once, lane by lane in order, on the stores themselves (no caches,
no queues, no timing), and its response (`words bytes` per lane, lane-major, as `Req.resp`). A
`shared::cluster` lane performs on the shared memory of the CTA its address names (the cluster
`Directory`), a local lane on its own lane's store. Each store is taken out while written, so it
is written in place. -/
private def performOne (m : MemSys) (sm cta slot : Nat) (r : Req) : Except String (MemSys × Resp) := do
  let n := words r.bytes
  let mut m := m
  let mut out : Array UInt32 := Array.replicate (if r.op.returns then r.addrs.size * n else 0) 0
  let put (out : Array UInt32) (lane : Nat) (ws : Array UInt32) : Array UInt32 :=
    if r.op.returns then (List.range n).foldl (fun o j => o.set! (lane * n + j) (ws[j]?.getD 0)) out
    else out
  let lanes := (List.range r.addrs.size).filter fun l => (r.addrs[l]?.join).isSome
  match r.space with
  | .global =>
    if m.logGlobal && r.op.writes then m := { m with globalLog := m.globalLog.push (cta, r) }
    let mut g := m.l2.flat.global
    m := { m with l2 := { m.l2 with flat := { m.l2.flat with global := default } } }
    for lane in lanes do
      let some l := Flat.loc cta r lane | continue
      let (g', ws) ← performLane g r lane l
      g := g'; out := put out lane ws
    m := { m with l2 := { m.l2 with flat := { m.l2.flat with global := g } } }
  | .shared =>
    let mut st := m.sms[sm]!.shared[cta]!
    m := { m with sms := m.sms.modify sm fun x => { x with shared := x.shared.set! cta default } }
    for lane in lanes do
      let some l := Flat.loc cta r lane | continue
      let (st', ws) ← performLane st r lane l
      st := st'; out := put out lane ws
    m := { m with sms := m.sms.modify sm fun x => { x with shared := x.shared.set! cta st } }
  | .sharedCluster =>
    for (psm, pcta, plns, part) in ← SmMem.clusterParts m.dir sm cta r do
      let mut st := m.sms[psm]!.shared[pcta]!
      m := { m with sms := m.sms.modify psm fun x => { x with shared := x.shared.set! pcta default } }
      for lane in plns do
        let some l := Flat.loc pcta part lane | continue
        let (st', ws) ← performLane st part lane l
        st := st'; out := put out lane ws
      m := { m with sms := m.sms.modify psm fun x => { x with shared := x.shared.set! pcta st } }
  | .local =>
    let some ls := m.sms[sm]!.locals.get? slot | throw "local memory of a warp that is gone"
    let mut ls := ls
    for lane in lanes do
      let (st, ws) ← ls[lane]!.perform r lane
      ls := ls.set! lane st
      out := put out lane (ws.extract 0 n)
    m := { m with sms := m.sms.modify sm fun x => { x with locals := x.locals.insert slot ls } }
  | .tmem =>
    -- The SM's own tensor memory, and a CTA pair's peer's (past `2 ^ 24`).
    let own := lanes.filter fun l => match (r.addrs[l]? : Option (Option UInt64)) with
      | some (some a) => a < 2 ^ 24 | _ => false
    let peer : Req := { r with addrs := r.addrs.map (·.bind fun a => if a < 2 ^ 24 then none else some (a - 2 ^ 24)) }
    let parts ← if peer.addrs.any (·.isSome) then SmMem.clusterParts m.dir sm cta peer .tmem else pure #[]
    for (psm, ls, q) in #[(sm, own, r)] ++ parts.map fun (psm, _, pl, part) => (psm, pl.toList, part) do
      let mut st := m.sms[psm]!.tmem
      m := { m with sms := m.sms.modify psm fun x => { x with tmem := default } }
      for lane in ls do
        let (st', ws) ← st.perform q lane
        st := st'; out := put out lane (ws.extract 0 n)
      m := { m with sms := m.sms.modify psm fun x => { x with tmem := st } }
  | sp => throw s!"{repr sp} memory is not modelled yet"
  return (m, out)

@[inherit_doc performOne]
def performNow (m : MemSys) (sm cta slot : Nat) (r : Req) : Except String (MemSys × Resp) := do
  if r.sharedPart.isEmpty then return ← m.performOne sm cta slot r
  -- A generic access into shared memory and another space: each part on its own.
  let (m, a) ← m.performOne sm cta slot { r with sharedPart := #[] }
  let (m, b) ← m.performOne sm cta slot { r with space := .shared, addrs := r.sharedPart, sharedPart := #[] }
  let n := words r.bytes
  return (m, (List.range r.sharedPart.size).foldl (fun out l =>
    if (r.sharedPart[l]?.join).isSome && r.op.returns then
      (List.range n).foldl (fun o j => o.set! (l * n + j) (b[l * n + j]?.getD 0)) out
    else out) a)

end MemSys

end NV.Core
