import NV.PTX.Async.MBarrier
import NV.Formal.System
import NV.Formal.Count

/-!
# The TMA ring: producers, consumers, full and empty mbarriers

A producer/consumer ring as warp-specialized Hopper kernels build it:
`S` shared-memory slots, each with a full and an empty mbarrier. For stage `q` the slot is
`q % S` and the parity bit `(q / S) & 1`.

* Producer `i` (one elected lane): waits on the slot's empty barrier for the phase
  before round `q / S` (parity `wph ^ 1`; the first round passes on a fresh barrier), performs
  `mbarrier.arrive.expect_tx` of its `bytes i` on the full barrier and issues
  `cp.async.bulk ... mbarrier::complete_tx::bytes` into its region of the slot.
* The copy's bytes land in pieces; each piece performs complete-tx on the full barrier.
* Consumer warp `c`: `try_wait.parity` on the full barrier for round `q / S`, reads
  the slot, then its lane 0 arrives once on the empty barrier.

With `P` producers joined on one full barrier (e.g. weights and activations completing on one
slot's barrier) each full barrier must be initialized with count `P`, each empty barrier with the
consumer count `C`.

Every fault is a property violation: an mbarrier condition the ISA leaves undefined
(over-arrival, a stale parity wait), a consumer starting to read a slot whose bytes have not
all landed for its stage, or a producer issuing into a slot a consumer is still reading.
-/

namespace Examples.Ring

open NV NV.Formal

open NV.PTX

structure Config where
  /-- Ring slots `S`. -/
  slots : Nat
  /-- Consumer warps `C`, each arriving once per stage on the empty barrier. -/
  consumers : Nat
  /-- Producers `P` joined on each full barrier. -/
  producers : Nat
  /-- Bytes producer `i` copies per stage. -/
  bytes : Nat → Nat
  /-- Stages `N` streamed through the ring. -/
  stages : Nat
  /-- `mbarrier.init` counts of the full and empty barriers. -/
  fullCount : Nat
  emptyCount : Nat

inductive Fault where
  | mbarrier (f : MBarrier.Fault)
  /-- A consumer acquired a slot whose bytes for its stage have not all landed. -/
  | readBeforeLanded
  /-- A producer issued into a slot a consumer was still reading. -/
  | overwrite
  deriving DecidableEq, Repr

structure State where
  full : Nat → MBarrier
  empty : Nat → MBarrier
  /-- Stages producer `i` has issued. -/
  issued : Nat → Nat
  /-- Bytes of producer `i`'s copy into slot `w` not yet landed. -/
  pending : Nat → Nat → Nat
  /-- The stage whose data producer `i` last wrote into slot `w`, if any. -/
  tag : Nat → Nat → Option Nat
  /-- Consumer `c`'s current stage, and whether it holds the slot (acquired, not released). -/
  stage : Nat → Nat
  reading : Nat → Bool
  fault : Option Fault := none

inductive Action where
  | issue (i : Nat)
  | land (i w bytes : Nat)
  | acquire (c : Nat)
  | release (c : Nat)
  deriving DecidableEq, Repr

variable (cfg : Config)

def update {β : Type} (f : Nat → β) (i : Nat) (v : β) : Nat → β := fun j => if j = i then v else f j

def update₂ {β : Type} (f : Nat → Nat → β) (i w : Nat) (v : β) : Nat → Nat → β :=
  fun j x => if j = i ∧ x = w then v else f j x

/-- Every barrier as `mbarrier.init` leaves it (`MBarrier.init`; `init_full`, `init_empty`). -/
def init : State where
  full _ := ⟨0, cfg.fullCount, cfg.fullCount, 0, true⟩
  empty _ := ⟨0, cfg.emptyCount, cfg.emptyCount, 0, true⟩
  issued _ := 0
  pending _ _ := 0
  tag _ _ := none
  stage _ := 0
  reading _ := false

def faulted (s : State) (f : Fault) : Option State := some { s with fault := some f }

/-- Continue with a barrier operation's result, or fault. -/
def withBarrier (s : State) (r : Except MBarrier.Fault MBarrier) (k : MBarrier → Option State) :
    Option State :=
  match r with
  | .ok b => k b
  | .error f => faulted s (.mbarrier f)

/-- A wait that spins disables the action; a stale parity faults. -/
def afterWait (s : State) (r : Except MBarrier.Fault Bool) (k : Unit → Option State) :
    Option State :=
  match r with
  | .ok true => k ()
  | .ok false => none
  | .error f => faulted s (.mbarrier f)

def step (s : State) (a : Action) : Option State :=
  if s.fault.isSome then none else
  match a with
  | .issue i =>
    let q := s.issued i
    let w := q % cfg.slots
    if i < cfg.producers ∧ q < cfg.stages then
      afterWait s ((s.empty w).waitPhase ((q / cfg.slots : Nat) - 1 : Int)) fun _ =>
        if (List.range cfg.consumers).any fun c => s.reading c && s.stage c % cfg.slots == w then
          faulted s .overwrite
        else
          withBarrier s ((s.full w).arriveExpectTx (cfg.bytes i)) fun b =>
            some { s with
              full := update s.full w b
              empty := update s.empty w (s.empty w).observe
              issued := update s.issued i (q + 1)
              pending := update₂ s.pending i w (cfg.bytes i)
              tag := update₂ s.tag i w (some q) }
    else none
  | .land i w k =>
    if i < cfg.producers ∧ w < cfg.slots ∧ 0 < k ∧ k ≤ s.pending i w then
      withBarrier s ((s.full w).completeTx k) fun b =>
        some { s with
          full := update s.full w b
          pending := update₂ s.pending i w (s.pending i w - k) }
    else none
  | .acquire c =>
    let q := s.stage c
    let w := q % cfg.slots
    if c < cfg.consumers ∧ q < cfg.stages ∧ s.reading c = false then
      afterWait s ((s.full w).waitPhase (q / cfg.slots : Nat)) fun _ =>
        if (List.range cfg.producers).all fun i => s.tag i w == some q && s.pending i w == 0 then
          some { s with
            full := update s.full w (s.full w).observe
            reading := update s.reading c true }
        else faulted s .readBeforeLanded
    else none
  | .release c =>
    let w := s.stage c % cfg.slots
    if c < cfg.consumers ∧ s.reading c = true then
      withBarrier s ((s.empty w).arrive 1) fun b =>
        some { s with
          empty := update s.empty w b
          stage := update s.stage c (s.stage c + 1)
          reading := update s.reading c false }
    else none

def system : System State Action := ⟨init cfg, step cfg⟩

/-- Every producer has issued every stage, every consumer has released every stage, every byte
has landed. -/
def Done (s : State) : Prop :=
  (∀ i < cfg.producers, s.issued i = cfg.stages) ∧
  (∀ c < cfg.consumers, s.stage c = cfg.stages ∧ s.reading c = false) ∧
  (∀ i < cfg.producers, ∀ w < cfg.slots, s.pending i w = 0)

/-- The configurations the code uses: barrier counts match the arrivals, and one stage's bytes fit
the tx-count. -/
structure Config.Valid : Prop where
  slots_pos : 0 < cfg.slots
  consumers_pos : 0 < cfg.consumers
  producers_pos : 0 < cfg.producers
  bytes_pos : ∀ i < cfg.producers, 0 < cfg.bytes i
  bytes_fit : sumBelow cfg.producers cfg.bytes ≤ MBarrier.countMax
  full_count : cfg.fullCount = cfg.producers
  empty_count : cfg.emptyCount = cfg.consumers
  /-- Arrival counts fit `mbarrier.init` (Table 43). -/
  counts_fit : cfg.producers ≤ MBarrier.countMax ∧ cfg.consumers ≤ MBarrier.countMax

theorem init_full {cfg : Config} (hv : cfg.Valid) (w : Nat) :
    MBarrier.init cfg.fullCount = .ok ((init cfg).full w) := by
  simp [MBarrier.init, init, hv.full_count, hv.counts_fit.1, Nat.one_le_iff_ne_zero,
    Nat.pos_iff_ne_zero.1 hv.producers_pos]

theorem init_empty {cfg : Config} (hv : cfg.Valid) (w : Nat) :
    MBarrier.init cfg.emptyCount = .ok ((init cfg).empty w) := by
  simp [MBarrier.init, init, hv.empty_count, hv.counts_fit.2, Nat.one_le_iff_ne_zero,
    Nat.pos_iff_ne_zero.1 hv.consumers_pos]

end Examples.Ring
