import Std.Data.HashMap
import Std.Data.HashSet
import NV.Machine.Access

/-!
# Data races between the threads of a CTA

The check `Launch.races` turns on. Two accesses by threads of one CTA to the same byte of shared or
global memory race when at least one writes, not both are atomic, and neither happens before the
other. Happens-before is a warp's program order and the synchronization PTX's memory model gives
(ISA 9.5 §8.9, §9.7.13): a barrier (`bar.sync`/`barrier.sync`, named or not, any thread count)
orders each participant's accesses before its arrival before every waiting participant's accesses
after it; an `mbarrier` arrive releases and an acquire read of the mbarrier word (`try_wait`,
`test_wait`) synchronizes with every arrive performed before it (the arrives are read-modify-writes
of the word, so a read of its latest value reads the release sequence of all of them). Lanes of one
warp are ordered across its instructions; two lanes of one instruction touching the same byte, one
writing, race.

The check is FastTrack's (Flanagan and Freund, PLDI 2009) with a warp as the thread of its vector
clocks: each warp's clock advances at each release it performs, each byte keeps its last write and
each warp's last read as (warp, clock). Accesses are taken in each warp's issue order; a release is
taken at issue, an acquire when the mbarrier read performs (the warp's later accesses depend on it).

The async proxy's reads of shared memory a warp's instructions start (`wgmma` operands, a bulk or
tensor copy's shared source) are checked as that warp's reads at the instruction's issue, with one
more rule (PTX ISA §8.6, "Proxies"): a generic-proxy write they read must be followed, in its
writer's program order, by a `fence.proxy.async` before it is released to them.

The async proxy's writes (tensor and bulk copies, `cp.async`'s shared stores, `st.async`) and
the warpgroup and tensor-memory MMAs' shared-memory reads are checked as the accesses of a thread
of their own, one per operation, whose clock is its issuing warp's
at issue (a release: the warp's earlier accesses happen before the copy's). Its signals carry that
clock: a complete-tx or arrive-on it performs on an mbarrier releases its writes to the warps that
acquire the mbarrier, and its group's completion (`cp.async.wait_group`,
`cp.async.bulk.wait_group`) to the warp that issued it, taken when the group completes (an access
the warp makes between that and its wait is taken as ordered: missed, not invented).

The cluster barrier (`barrier.cluster.arrive`/`.wait`) orders a CTA's threads as a CTA barrier does
(`clusterBarId`); a warp acquires its phase when the phase completes, though it may not wait for
it until later (its accesses in between are taken as ordered: missed, not invented).

A read of a shared byte no access of its CTA wrote is reported too: its value is whatever the
SM's previous CTA there left (the simulator's is zero), so results may differ from run to run.

Across CTAs, global memory's plain accesses are seen: two CTAs of a launch storing the same byte,
or one reading (the async proxy's copies included) a byte another stores, in either order
(`CrossWrite`; the engine joins the SMs'). Nothing within a launch orders them but flags or atomics
with fences, which the check does not follow; whether two stores' values differ says whether the
byte's final value depends on which comes last. Floating-point atomic additions to one global
address from two warps are reported too: rounding makes their sum depend on their order, which
the hardware does not fix (`floatAdds`).

Not seen: the async proxy's reads of global memory, ordering
through fences, flags or atomics other than mbarriers, cluster-scope mbarriers and peer CTAs'
shared memory, other accesses of threads of different CTAs, and races between lanes of one warp across its
instructions (the check takes a warp's lanes as ordered by its program order, as they execute
here; PTX orders them only through `bar.warp.sync`, so such races are missed, not invented).
-/

namespace NV.Core.Memory

open NV

/-- A vector clock: a warp slot's clock, by slot (absent: 0). -/
abbrev Clock := Std.HashMap Nat Nat

def Clock.join (a b : Clock) : Clock := b.fold (fun acc k v => acc.insert k (max v (acc.getD k 0))) a

/-- An access as a byte remembers it: the warp's slot, its lane, its clock and the warp's
instruction count then. -/
structure Stamp where
  slot : Nat
  lane : Nat
  clock : Nat
  seq : Nat
  /-- How many proxy fences the warp had issued: a later one covers this access. -/
  fences : Nat := 0
  /-- An access of an async operation (the async proxy's own), which the async proxy reads with no
  proxy fence. -/
  asyncProxy : Bool := false
  deriving Inhabited

structure RaceByte where
  write : Option Stamp := none
  atomicWrite : Bool := false
  /-- An mbarrier operation wrote it (`mbarrier.init` and kin, which the check does not order). -/
  mbarWritten : Bool := false
  /-- Each warp's last read. -/
  reads : Std.HashMap Nat Stamp := {}
  deriving Inhabited

/-- A race found: what, where, when and between which threads. -/
structure Race where
  kind : String
  shared : Bool
  addr : Nat
  /-- The CTA's index on its SM. -/
  cta : Nat
  first : Nat × Nat
  second : Nat × Nat
  at_ : Nat
  deriving Inhabited, Repr

/-- A global byte a plain store wrote: by which CTA (its index on the SM and that index's
generation, `RaceCheck.ctaGen`) and what. -/
structure GlobalWrite where
  cta : Nat
  gen : Nat
  value : UInt8
  deriving Inhabited

/-- Two CTAs of a launch storing the same global byte, neither atomically: no order between them
is given (`RaceCheck.global`). Each CTA by SM, index on the SM and generation. -/
structure CrossWrite where
  addr : Nat
  first : Nat × Nat × Nat
  second : Nat × Nat × Nat
  /-- The two stores wrote different values: the byte's final value depends on their order. -/
  differ : Bool
  deriving Inhabited, Repr

/-- The check's state on one SM. -/
structure RaceCheck where
  /-- Per warp slot: its vector clock, and how many instructions it issued. -/
  clocks : Std.HashMap Nat Clock := {}
  seqs : Std.HashMap Nat Nat := {}
  /-- Per warp slot: its proxy fences so far. -/
  fences : Std.HashMap Nat Nat := {}
  /-- Per CTA index and barrier: what the arrivals of its current generation release. -/
  bars : Std.HashMap (Nat × Nat) Clock := {}
  /-- Per mbarrier (CTA index, shared offset): what the arrives performed so far release. -/
  mbars : Std.HashMap (Nat × Nat) Clock := {}
  /-- Per arrive request (the memory system's id): its warp's clock at issue. -/
  arrives : Std.HashMap Nat Clock := {}
  /-- Per async operation (the async unit's agent index): its clock, its issuing warp's at issue. -/
  agents : Std.HashMap Nat Clock := {}
  /-- Per CTA index, the CTAs it has held: a new CTA's bytes start unaccessed. -/
  ctaGen : Std.HashMap Nat Nat := {}
  /-- Per CTA index: its shared memory's reserved bytes (the driver's, `Kernel.sharedReserved`),
  which it reads as the driver set them. -/
  reserved : Std.HashMap Nat Nat := {}
  bytes : Std.HashMap (Nat × Nat × Bool × Nat) RaceByte := {}
  races : Array Race := #[]
  reported : Std.HashSet (Nat × Nat × Bool × Nat) := {}
  /-- Per global byte a plain store wrote, the last such store's CTA and value; and the bytes two
  of this SM's CTAs both stored (`CrossWrite`, SM fields 0: the engine fills them). -/
  global : Std.HashMap Nat GlobalWrite := {}
  crossWrites : Array CrossWrite := #[]
  /-- How many such bytes, and with different values. -/
  crossCount : Nat := 0
  crossDiffer : Nat := 0
  /-- Per global byte a plain load read (the async proxy's too), the last such load's CTA (index,
  generation); and the bytes one of this SM's CTAs read and another stored, in either order
  (`CrossWrite` with `differ` false). -/
  globalReads : Std.HashMap Nat (Nat × Nat) := {}
  crossReads : Array CrossWrite := #[]
  crossReadCount : Nat := 0
  /-- Per global address a floating-point atomic addition reached, the first adder (CTA index,
  generation, warp slot); and the addresses a second adder reached (another warp's or CTA's):
  rounding makes the sum depend on the additions' order, which the hardware does not fix. -/
  floatAdds : Std.HashMap Nat (Nat × Nat × Nat) := {}
  floatAddClashes : Array Nat := #[]
  floatAddCount : Nat := 0
  deriving Inhabited

namespace RaceCheck

/-- The most races kept. -/
def cap : Nat := 16

private def clockOf (rc : RaceCheck) (slot : Nat) : Clock :=
  (rc.clocks.getD slot {}).insert slot ((rc.clocks.getD slot {}).getD slot 1)

/-- A new CTA takes index `cta`: nothing of the index's earlier CTA (its barriers, its accesses)
carries over. -/
def newCta (rc : RaceCheck) (cta : Nat) (reserved := 0) : RaceCheck :=
  { rc with ctaGen := rc.ctaGen.insert cta (rc.ctaGen.getD cta 0 + 1),
            reserved := rc.reserved.insert cta reserved,
            bars := rc.bars.filter fun (c, _) _ => c != cta,
            mbars := rc.mbars.filter fun (c, _) _ => c != cta }

/-- Warp `slot` starts (a fresh warp in the slot: nothing of the slot's earlier warp orders it). -/
def start (rc : RaceCheck) (slot : Nat) : RaceCheck :=
  let c := ((rc.clocks.getD slot {}).getD slot 0) + 1
  { rc with clocks := rc.clocks.insert slot (({} : Clock).insert slot c) }

/-- Warp `slot` issues a proxy fence (`fence.proxy.async`). -/
def proxyFence (rc : RaceCheck) (slot : Nat) : RaceCheck :=
  { rc with fences := rc.fences.insert slot (rc.fences.getD slot 0 + 1) }

/-- Warp `slot` releases: its accesses so far are covered by what it hands on; its clock moves on. -/
private def release (rc : RaceCheck) (slot : Nat) : RaceCheck × Clock :=
  let c := rc.clockOf slot
  ({ rc with clocks := rc.clocks.insert slot (c.insert slot (c.getD slot 1 + 1)) }, c)

/-- The slot an async operation's accesses are stamped with: past every warp slot. -/
def agentSlot (a : Nat) : Nat := 2 ^ 32 + a

/-- Warp `slot` issues async operation `a`, which runs after operations `deps` complete: the
operation's clock is the warp's now joined with theirs (a `tcgen05.commit`'s arrival carries the
completion of the MMAs it tracks). -/
def agentStart (rc : RaceCheck) (slot a : Nat) (deps : Array Nat := #[]) : RaceCheck :=
  let (rc, c) := rc.release slot
  let c := deps.foldl (fun (c : Clock) d => c.join (rc.agents.getD d {})) c
  { rc with agents := rc.agents.insert a (c.insert (agentSlot a) 1) }

/-- Request `id`, a signal (complete-tx or arrive-on) of async operation `a`, is issued: it
releases the operation's writes. -/
def agentSignal (rc : RaceCheck) (a id : Nat) : RaceCheck :=
  { rc with arrives := rc.arrives.insert id (rc.agents.getD a {}) }

/-- The group of async operations `ops` that warp `slot` issued has completed: the warp acquires
their writes. -/
def agentsDone (rc : RaceCheck) (slot : Nat) (ops : Array Nat) : RaceCheck :=
  let joined := ops.foldl (fun (c : Clock) a => c.join (rc.agents.getD a {})) (rc.clockOf slot)
  { rc with clocks := rc.clocks.insert slot joined }

/-- The barrier id the cluster barrier (`barrier.cluster`) takes here, past the CTA's 16 named
barriers: it orders the threads of each of the cluster's CTAs as a CTA barrier would. -/
def clusterBarId : Nat := 16

/-- Warp `slot` of CTA index `cta` arrives at barrier `id`. -/
def barArrive (rc : RaceCheck) (cta id slot : Nat) : RaceCheck :=
  let (rc, c) := rc.release slot
  { rc with bars := rc.bars.insert (cta, id) ((rc.bars.getD (cta, id) {}).join c) }

/-- Barrier `id` of CTA index `cta` completes: its waiting warps `waiters` acquire what the
generation's arrivals released, and the next generation starts empty. -/
def barRelease (rc : RaceCheck) (cta id : Nat) (waiters : Array Nat) : RaceCheck :=
  let g := rc.bars.getD (cta, id) {}
  let rc := waiters.foldl (fun rc w => { rc with clocks := rc.clocks.insert w ((rc.clockOf w).join g) }) rc
  { rc with bars := rc.bars.erase (cta, id) }

/-- Request `id`, an mbarrier arrive of warp `slot`, is issued: its release is its clock now. -/
def mbarArrive (rc : RaceCheck) (slot id : Nat) : RaceCheck :=
  let (rc, c) := rc.release slot
  { rc with arrives := rc.arrives.insert id c }

/-- An arrive (request `id`) on the mbarrier at shared offset `addr` of CTA index `cta` performs. -/
def mbarPerformed (rc : RaceCheck) (cta addr id : Nat) : RaceCheck :=
  match rc.arrives.get? id with
  | some c => { rc with arrives := rc.arrives.erase id,
                        mbars := rc.mbars.insert (cta, addr) ((rc.mbars.getD (cta, addr) {}).join c) }
  | none => rc

/-- Warp `slot` of CTA index `cta` performs an acquire read of shared offset `addr`: if an mbarrier
lives there, it acquires what its arrives released. -/
def acquire (rc : RaceCheck) (cta slot addr : Nat) : RaceCheck :=
  match rc.mbars.get? (cta, addr) with
  | some m => { rc with clocks := rc.clocks.insert slot ((rc.clockOf slot).join m) }
  | none => rc

/-- Whether an atomic type adds floating-point values (whose sum depends on the order). -/
def floatType : AtomType → Bool
  | .f32 | .f32ftz | .f64 | .f16 | .bf16 | .f16x2 | .bf16x2 => true
  | .vec t _ => floatType t
  | _ => false

/-- Warp `slot` of CTA index `cta` adds floating-point values atomically at global addresses
`addrs` (`floatAdds`). -/
def floatAdd (rc : RaceCheck) (cta slot : Nat) (addrs : Array (Option UInt64)) : RaceCheck := Id.run do
  let gen := rc.ctaGen.getD cta 0
  let mut rc := rc
  for a? in addrs do
    let some a := a? | continue
    match rc.floatAdds.get? a.toNat with
    | some first =>
      if first != (cta, gen, slot) && !rc.floatAddClashes.contains a.toNat then
        rc := { rc with floatAddCount := rc.floatAddCount + 1,
                        floatAddClashes := if rc.floatAddClashes.size < cap then rc.floatAddClashes.push a.toNat
                                           else rc.floatAddClashes }
    | none => rc := { rc with floatAdds := rc.floatAdds.insert a.toNat (cta, gen, slot) }
  return rc

/-- Request `r` from another CTA of the cluster (a peer's store or multicast copy) writes lanes
`lanes` of CTA index `cta`'s shared memory: its bytes count as written for the unwritten-read
report (`mbarWritten`); a peer's order with this CTA's threads is not seen (`RaceCheck`). -/
def peerWrite (rc : RaceCheck) (cta : Nat) (r : Req) (lanes : Array Nat) : RaceCheck := Id.run do
  if !r.op.writes then return rc
  let gen := rc.ctaGen.getD cta 0
  let mut rc := rc
  for lane in lanes do
    let some a := r.addrs[lane]?.join | continue
    for b in [0:r.bytes] do
      let k := (cta, gen, true, a.toNat + b)
      rc := { rc with bytes := rc.bytes.insert k { (rc.bytes.get? k).getD {} with mbarWritten := true } }
  return rc

/-- A byte CTA `first` read (or stored) and CTA `second` stored (or read), on this SM. -/
def crossRead (rc : RaceCheck) (addr : Nat) (first second : Nat × Nat) : RaceCheck :=
  let rc := { rc with crossReadCount := rc.crossReadCount + 1 }
  if rc.crossReads.size < cap then
    let x : CrossWrite := { addr, first := (0, first), second := (0, second), differ := false }
    { rc with crossReads := rc.crossReads.push x }
  else rc

/-- CTA index `cta` reads global byte `addr` with a plain load (or the async proxy's): another of
this SM's CTAs' earlier store there is reported (`crossRead`). -/
def globalRead (rc : RaceCheck) (cta addr : Nat) : RaceCheck :=
  let gen := rc.ctaGen.getD cta 0
  let rc := match rc.global.get? addr with
    | some o => if (o.cta, o.gen) != (cta, gen) then rc.crossRead addr (o.cta, o.gen) (cta, gen) else rc
    | none => rc
  { rc with globalReads := rc.globalReads.insert addr (cta, gen) }

/-- The global bytes of request `r`, a load of CTA index `cta` (the async proxy's: a tensor or bulk
copy's source), for the checks across CTAs alone. -/
def globalLoad (rc : RaceCheck) (cta : Nat) (r : Req) : RaceCheck := Id.run do
  if r.space != .global then return rc
  let mut rc := rc
  for lane in [0:r.addrs.size] do
    let some a := r.addrs[lane]! | continue
    for b in [0:r.bytes] do rc := rc.globalRead cta (a.toNat + b)
  return rc

/-- Request `r` of warp `slot` of CTA index `cta`, issued at `t`: each active lane's bytes; `async`,
a read by the async proxy the warp's instruction starts; `agent`, a write of that async operation
(stamped with its own slot and clock). -/
def access (rc : RaceCheck) (t cta slot : Nat) (r : Req) (async := false) (agent : Option Nat := none) :
    RaceCheck := Id.run do
  let some shared := (match r.space with
    | .shared => some true
    | .global => some false
    | _ => none) | return rc
  -- An mbarrier operation: its bytes are written (a later phase check reads them), no more.
  if let .mbar _ := r.op then
    if !shared then return rc
    let mut rc := rc
    for lane in [0:r.addrs.size] do
      let some a := r.addrs[lane]! | continue
      for b in [0:r.bytes] do
        let k := (cta, rc.ctaGen.getD cta 0, true, a.toNat + b)
        rc := { rc with bytes := rc.bytes.insert k { (rc.bytes.get? k).getD {} with mbarWritten := true } }
    return rc
  let (writes, atomic) := match r.op with
    | .ld => (false, false)
    | .st => (true, false)
    | _ => (true, true)
  let rc := match r.op, shared with
    | .atom .add t, false | .red .add t, false => if floatType t then rc.floatAdd cta slot r.addrs else rc
    | _, _ => rc
  let (slot, vc) := match agent with
    | some a => (agentSlot a, rc.agents.getD a {})
    | none => (slot, rc.clockOf slot)
  let seq := rc.seqs.getD slot 0
  let mut rc := { rc with seqs := rc.seqs.insert slot (seq + 1) }
  for lane in [0:r.addrs.size] do
    let some a := r.addrs[lane]! | continue
    let me : Stamp := { slot, lane, clock := vc.getD slot 1, seq, fences := rc.fences.getD slot 0,
                        asyncProxy := agent.isSome }
    -- An earlier access is ordered before this one by program order (another instruction of this
    -- warp, or this lane) or by synchronization (its clock is within what this warp knows).
    let ordered (s : Stamp) : Bool :=
      if s.slot == slot then s.seq != seq || s.lane == lane else s.clock ≤ vc.getD s.slot 0
    for b in [0:r.bytes] do
      let k := (cta, rc.ctaGen.getD cta 0, shared, a.toNat + b)
      let x : RaceByte := (rc.bytes.get? k).getD {}
      let found : Option (String × Stamp) :=
        match x.write with
        | some w =>
          if !ordered w && !(atomic && x.atomicWrite) then
            some (if writes then "write after write" else "read after write", w)
          else if async && shared && !w.asyncProxy && rc.fences.getD w.slot 0 == w.fences then
            some ("async-proxy read of a generic write with no proxy fence after it", w)
          else none
        | none => none
      let found := found <|> if writes && !atomic then
          (x.reads.fold (fun acc _ s => acc <|> if ordered s then none else some s) none).map ("write after read", ·)
        else none
      -- A read of shared memory no access of this CTA wrote (atomics included): on hardware it
      -- holds what the SM's previous CTA there, or nothing, left.
      let found := found <|> if shared && !writes && x.write.isNone && !x.mbarWritten &&
          a.toNat + b ≥ rc.reserved.getD cta 0 then
          some ("read of shared memory its CTA never wrote", me)
        else none
      if let some (kind, o) := found then
        if rc.races.size < cap && !rc.reported.contains k then
          rc := { rc with races := rc.races.push { kind, shared, addr := a.toNat + b, cta,
                                                   first := (o.slot, o.lane), second := (slot, lane), at_ := t },
                          reported := rc.reported.insert k }
      let x : RaceByte := if writes then { write := some me, atomicWrite := atomic, reads := {} }
        else { x with reads := x.reads.insert slot me }
      rc := { rc with bytes := rc.bytes.insert k x }
      -- Across CTAs, global memory: a plain store's byte, and another CTA's earlier one there.
      if !shared && writes && !atomic then
        let w := words r.bytes
        let word := r.data[lane * w + b / 4]?.getD 0
        let value := (word >>> (8 * (b % 4)).toUInt32).toUInt8
        let gen := rc.ctaGen.getD cta 0
        let addr := a.toNat + b
        if let some (c, g) := rc.globalReads.get? addr then
          if (c, g) != (cta, gen) then rc := rc.crossRead addr (c, g) (cta, gen)
        if let some o := rc.global.get? addr then
          if (o.cta, o.gen) != (cta, gen) then
            rc := { rc with crossCount := rc.crossCount + 1,
                            crossDiffer := rc.crossDiffer + (if o.value != value then 1 else 0) }
            if rc.crossWrites.size < cap then
              let x : CrossWrite := { addr, first := (0, o.cta, o.gen), second := (0, cta, gen), differ := o.value != value }
              rc := { rc with crossWrites := rc.crossWrites.push x }
        rc := { rc with global := rc.global.insert addr { cta, gen, value } }
      if !shared && !writes then rc := rc.globalRead cta (a.toNat + b)
  return rc

end RaceCheck

end NV.Core.Memory
