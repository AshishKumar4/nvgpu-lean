import NV.Memory.Events

/-!
# The request machine

The operational memory of the simulator, and of the soundness theorem: one flat copy of each
state space, and the requests issued and not yet performed. An issued request is in flight; the
timing core decides when each lane of it is delivered, and delivery performs that lane on the
flat memory (reads, then writes, as one step). The machine refuses a delivery the ISA does not
allow, so a timing model that breaks the memory model is caught, not hidden:

* a thread's accesses to overlapping bytes perform in the order it issued them;
* a release performs only once every earlier request of its thread has;
* nothing of a thread performs while an earlier acquire of that thread is in flight.

Fences and barrier arrivals proceed only once the issuer has nothing in flight (`drained`).
Barriers are counted here too: an arrival joins the barrier's current instance, which completes
when the expected number of threads have arrived, and a wait passes once its instance has
completed.

Lane `l` of an issuer is memory-model thread `tid + l`. `Mem.sound` (`NV.Formal.Sound`) proves
that every sequence of steps this machine accepts performs a consistent execution.

Shared memory across the cluster (`shared::cluster`) and local memory are not modelled yet:
delivering them is an error.
-/

namespace NV

/-- Who issues: a thread (PTX) or a warp (SASS). `slot` names its in-flight queue. -/
structure Issuer where
  slot : Nat
  /-- Memory-model thread of lane 0. -/
  tid : Nat
  /-- The CTA whose shared memory `.shared` addresses name. -/
  cta : Nat
  deriving DecidableEq, Repr, Inhabited

structure InFlight where
  id : Nat
  who : Issuer
  /-- The CTA whose shared memory the request's `.shared` addresses name: the issuer's, or a peer's
  for a `shared::cluster` access the timing core resolved to it. -/
  cta : Nat
  req : Req
  /-- Active lanes not yet delivered. -/
  lanes : List Nat
  deriving Inhabited

/-- A barrier: its completed instances and the threads arrived in the current one. -/
structure Barrier where
  done : Nat := 0
  arrived : Nat := 0
  /-- Threads the current instance expects, set by its first arrival. -/
  expected : Nat := 0
  deriving Inhabited

structure Flat where
  global : Store
  /-- Shared memory, per CTA. -/
  shared : Array Store
  /-- In-flight requests, per issuer slot, oldest first. -/
  queues : Array (List InFlight)
  next : Nat := 0
  /-- Barriers, by the index the caller gives them (CTA barrier or cluster barrier). -/
  bars : Array Barrier := #[]
  deriving Inhabited

namespace Flat

def init (global : Store) (shared : Array Store) (slots : Nat) : Flat :=
  { global, shared, queues := Array.replicate slots [] }

/-- The location lane `lane` of `r` accesses, its shared addresses naming CTA `cta`. -/
def loc (cta : Nat) (r : Req) (lane : Nat) : Option Loc :=
  (r.addrs[lane]?).join.map fun a =>
    { space := r.space, owner := if r.space == .shared then cta else 0, addr := a.toNat }

def queue (m : Flat) (slot : Nat) : List InFlight := m.queues[slot]?.getD []

/-- Issue `r`, its shared addresses naming CTA `cta`; its active lanes are in flight. The
request's id. -/
def issue (m : Flat) (who : Issuer) (r : Req) (cta : Nat := who.cta) : Except String (Flat × Nat) :=
  if !(who.slot < m.queues.size) then .error s!"issuer slot {who.slot} outside the machine's {m.queues.size}"
  else if !([1, 2, 4, 8, 16].contains r.bytes) then .error s!"a {r.bytes}-byte access"
  else
    let lanes := (List.range r.addrs.size).filter fun l => (r.addrs[l]?).join.isSome
    let q : InFlight := { id := m.next, who, cta, req := r, lanes }
    .ok ({ m with queues := m.queues.modify who.slot (· ++ [q]), next := m.next + 1 }, m.next)

/-- The issuer has no request in flight. -/
def drained (m : Flat) (slot : Nat) : Bool := (m.queue slot).isEmpty

/-- Lane `lane` of `a` and of `b` touch overlapping bytes. -/
def overlap (a b : InFlight) (lane : Nat) : Bool :=
  match loc a.cta a.req lane, loc b.cta b.req lane with
  | some x, some y => x.space == y.space && x.owner == y.owner &&
      x.addr < y.addr + b.req.bytes && y.addr < x.addr + a.req.bytes
  | _, _ => false

/-- Why lane `lane` of request `q` may not perform while `p`, issued earlier by the same issuer,
is in flight; `none` when it may. -/
def blocker (p q : InFlight) (lane : Nat) : Option String :=
  if !p.lanes.contains lane then none
  else if overlap p q lane then some "an earlier access to the same bytes is in flight"
  else if q.req.releases then some "a release before its thread's earlier requests"
  else if p.req.acquires then some "an access before its thread's earlier acquire"
  else none

/-- The store of a space instance. -/
def store? (m : Flat) (l : Loc) : Option Store :=
  match l.space with
  | .global => if l.owner = 0 then some m.global else none
  | .shared => m.shared[l.owner]?
  | _ => none

def setStore (m : Flat) (l : Loc) (s : Store) : Flat :=
  match l.space with
  | .shared => { m with shared := m.shared.set! l.owner s }
  | _ => { m with global := s }

/-- `size` bytes at offset `off`. -/
def readBytes (s : Store) (off size : Nat) : List UInt8 :=
  (List.range size).map fun i => s.readByte (off + i)

/-- Bytes `bs` at offset `off`. -/
def writeBytes (s : Store) (off : Nat) : List UInt8 → Store
  | [] => s
  | b :: bs => writeBytes (s.writeByte off b) (off + 1) bs

/-- Write pieces `ps` of lane `lane` of `r` at store offset `off`, its reads having returned `read`. -/
def performWrites (r : Req) (lane off : Nat) (read : List UInt8) : List Piece → Store →
    Except String Store
  | [], s => .ok s
  | p :: ps, s =>
    match r.written lane p read with
    | some v =>
      if v.length = p.size then performWrites r lane off read ps (writeBytes s (off + p.off) v)
      else .error s!"lane {lane}: {v.length} bytes of operands for a {p.size}-byte write"
    | none => .error s!"{repr r.op}: not supported yet"

/-- The bytes lane `lane`'s reads return from store `s` at offset `off`: all of the access's bytes
when it reads (its read pieces cover them). -/
def readPart (r : Req) (off : Nat) (s : Store) : List UInt8 :=
  if r.pieces.any (!·.write) then readBytes s off r.bytes else []

/-- Perform lane `lane` of `r` at location `l` on store `s`: its reads (which all come before its
writes), then its writes. The new store and the bytes the reads returned. -/
def perform (r : Req) (lane : Nat) (l : Loc) (s : Store) : Except String (Store × List UInt8) :=
  if !s.contains (UInt64.ofNat l.addr) r.bytes then
    .error s!"lane {lane}: {r.bytes}-byte access at {l.addr} outside {repr r.space}"
  else
    let off := l.addr - s.base.toNat
    let read := readPart r off s
    (performWrites r lane off read (r.pieces.filter (·.write)) s).map (·, read)

/-- The in-flight request `id` of issuer slot `slot`, and its index in the slot's queue. -/
def find (m : Flat) (slot id : Nat) : Option (Nat × InFlight) :=
  ((m.queue slot).findIdx? (·.id == id)).map fun i => (i, (m.queue slot)[i]!)

/-- Request `p` once lane `lane` of request `id` has performed. -/
def retireOne (id lane : Nat) (p : InFlight) : Option InFlight :=
  if p.id == id then
    if (p.lanes.erase lane).isEmpty then none else some { p with lanes := p.lanes.erase lane }
  else some p

/-- The queue once lane `lane` of request `id` has performed: the request without that lane, or
without the request when it was its last. -/
def retire (qs : List InFlight) (id lane : Nat) : List InFlight :=
  qs.filterMap (retireOne id lane)

/-- Lane `lane` of `q` performs, with `q` at index `i` of the queue it belongs to. -/
def deliverAt (m : Flat) (slot i : Nat) (q : InFlight) (lane : Nat) :
    Except String (Flat × List UInt8) :=
  let qs := m.queue slot
  if !q.lanes.contains lane then .error s!"lane {lane} of request {q.id} is not in flight" else
  match (qs.take i).findSome? (blocker · q lane) with
  | some why =>
    let earlier := (qs.take i).filter (·.lanes.contains lane) |>.map fun p =>
      s!"{p.id} {repr p.req.op} {repr p.req.space} {repr p.req.sem}"
    .error s!"request {q.id} ({repr q.req.op} {repr q.req.space} {repr q.req.sem}) lane {lane}: {why}; in flight before it: {earlier}"
  | none =>
    match loc q.cta q.req lane with
    | none => .error s!"lane {lane} of request {q.id} is inactive"
    | some l =>
      match m.store? l with
      | none => .error s!"{repr l.space} memory is not modelled yet"
      | some s =>
        -- Detach the store and the queue from the machine, so both are updated in place.
        let m := m.setStore l default
        let m := { m with queues := m.queues.set! slot [] }
        match perform q.req lane l s with
        | .error e => .error e
        | .ok (s', bytes) =>
          .ok ({ m.setStore l s' with queues := m.queues.set! slot (retire qs q.id lane) }, bytes)

/-- Deliver lane `lane` of request `id` from issuer slot `slot`: the bytes its reads returned
(`Req.laneResp` makes them response words). -/
def deliver (m : Flat) (slot id lane : Nat) : Except String (Flat × List UInt8) :=
  match find m slot id with
  | none => .error s!"request {id} is not in flight"
  | some (i, q) => deliverAt m slot i q lane

/-! ## A request's lanes delivered together

The timing core delivers the lanes of a request part one after another (`deliverSeq`).
`deliverLanes` does the same at once: the request found once, the earlier requests that may hold
one of its lanes back picked once (`mayBlock`), and its issuer's queue rebuilt once at the end.
`NV.Memory.SoundLanes.deliverLanes_eq` proves it equal to `deliverSeq`, so a run that uses it is
still a sequence of `deliver` steps (`Mem.sound`). -/

/-- Lanes `ls` of request `id` of issuer slot `slot` delivered one after another (`deliver`): the
bytes each lane's reads returned, in order. -/
def deliverSeq (m : Flat) (slot id : Nat) : List Nat → Except String (Flat × List (List UInt8))
  | [] => .ok (m, [])
  | l :: ls => match m.deliver slot id l with
    | .error e => .error e
    | .ok (m', b) => match deliverSeq m' slot id ls with
      | .error e => .error e
      | .ok (m'', bs) => .ok (m'', b :: bs)

/-- Request `p` once lanes `ls` of request `id` have performed, one after another (`retireOne`). -/
def retireLanes (id : Nat) (ls : List Nat) (p : InFlight) : Option InFlight :=
  if p.id == id then ls.foldlM (fun p l => retireOne id l p) p else some p

/-- Whether `p`, issued before `q` by the same issuer, may hold back one of lanes `ls` of `q`
(`blocker`). -/
def mayBlock (ls : List Nat) (p q : InFlight) : Bool :=
  q.req.releases || p.req.acquires || ls.any fun l => p.lanes.contains l && overlap p q l

/-- Lane `lane` of `q` performed on `m`'s stores as `deliverAt` performs it, none of `cand` (the
requests before `q` that may hold one of its lanes back) holding it back; `none` where it would
not perform. -/
def performLane (m : Flat) (cand : List InFlight) (q : InFlight) (lane : Nat) :
    Option (Flat × List UInt8) :=
  if !q.lanes.contains lane then none else
  match cand.findSome? (blocker · q lane), loc q.cta q.req lane with
  | none, some l =>
    match m.store? l with
    | none => none
    | some s =>
      -- Detached from the machine, so the store is updated in place.
      let m := m.setStore l default
      match perform q.req lane l s with
      | .ok (s', bytes) => some (m.setStore l s', bytes)
      | .error _ => none
  | _, _ => none

/-- `m` once lanes `done` of request `id` of slot `slot`, whose queue was `qs`, have performed. -/
def withRetired (m : Flat) (slot id : Nat) (qs : List InFlight) (done : List Nat) : Flat :=
  if done.isEmpty then m else { m with queues := m.queues.set! slot (qs.filterMap (retireLanes id done)) }

/-- `deliverLanes`' loop: `m` the machine with the stores the lanes `done` wrote and slot `slot`'s
queue still `qs`, `q` the request without those lanes, `acc` their bytes, latest first. -/
def deliverLanesGo (slot id : Nat) (qs cand : List InFlight) (m : Flat) (q : InFlight) (done : List Nat)
    (acc : List (List UInt8)) : List Nat → Except String (Flat × List (List UInt8))
  | [] => .ok (m.withRetired slot id qs done, acc.reverse)
  | l :: ls =>
    match m.performLane cand q l with
    | none => (deliverSeq (m.withRetired slot id qs done) slot id (l :: ls)).map
        fun (m, bs) => (m, acc.reverse ++ bs)
    | some (m', b) =>
      match retireOne id l q with
      | some q' => deliverLanesGo slot id qs cand m' q' (done ++ [l]) (b :: acc) ls
      | none => (deliverSeq (m'.withRetired slot id qs (done ++ [l])) slot id ls).map
          fun (m, bs) => (m, acc.reverse ++ b :: bs)

/-- `deliverSeq` at once (`deliverLanes_eq`); a lane that would not perform, or a request left with
no lanes, hands the rest to `deliverSeq`. -/
def deliverLanes (m : Flat) (slot id : Nat) (lanes : List Nat) : Except String (Flat × List (List UInt8)) :=
  match m.find slot id with
  | none => deliverSeq m slot id lanes
  | some (i, q) =>
    let qs := m.queue slot
    deliverLanesGo slot id qs ((qs.take i).filter (mayBlock lanes · q)) m q [] [] lanes

/-- A new issuer slot, with nothing in flight. -/
def addSlot (m : Flat) : Flat × Nat := ({ m with queues := m.queues.push [] }, m.queues.size)

/-- A new CTA's shared memory: `size` zero bytes. -/
def addCta (m : Flat) (size : Nat) (init : Array (Nat × UInt32) := #[]) : Flat × Nat :=
  let s := init.foldl (fun s (off, w) => if off + 4 ≤ size then s.writeWord off w else s) (Store.empty 0 size)
  ({ m with shared := m.shared.push s }, m.shared.size)

/-- A fence: the issuer must have nothing in flight. -/
def fence (m : Flat) (slot : Nat) : Except String Flat :=
  if m.drained slot then .ok m else .error "a fence with requests in flight"

def bar (m : Flat) (b : Nat) : Barrier := m.bars[b]?.getD default

/-- `n` threads of a drained issuer arrive on barrier `b`, whose instance completes at `expected`
threads. The instance they joined. More arrivals than an instance expects, or arrivals expecting
different counts, are errors (PTX ISA §9.7.13.1 leaves them undefined). -/
def arrive (m : Flat) (slot b n expected : Nat) : Except String (Flat × Nat) :=
  if !m.drained slot then .error "a barrier arrival with requests in flight" else
  let x := m.bar b
  if x.arrived ≠ 0 ∧ x.expected ≠ expected then .error "barrier arrivals expecting different counts"
  else if x.arrived + n > expected then .error "more barrier arrivals than expected" else
  let x' : Barrier := if x.arrived + n = expected then { done := x.done + 1, arrived := 0, expected }
    else { done := x.done, arrived := x.arrived + n, expected }
  let bars := if b < m.bars.size then m.bars else m.bars ++ Array.replicate (b + 1 - m.bars.size) default
  .ok ({ m with bars := bars.set! b x' }, x.done)

/-- A wait on instance `inst` of barrier `b` passes once the instance has completed. -/
def passed (m : Flat) (b inst : Nat) : Bool := inst < (m.bar b).done

/-- A drained issuer's wait on instance `inst` of barrier `b`. -/
def wait (m : Flat) (slot b inst : Nat) : Except String Flat :=
  if !m.drained slot then .error "a barrier wait with requests in flight"
  else if m.passed b inst then .ok m
  else .error s!"barrier {b} instance {inst} has not completed"

end Flat

end NV
