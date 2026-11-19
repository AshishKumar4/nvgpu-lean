/-!
# mbarrier

The `.layout::v0` mbarrier object of PTX ISA 9.4 §9.7.15.16: its contents (§9.7.15.16.3), phase
completion (§9.7.15.16.8), arrive-on (§9.7.15.16.9), expect-tx and complete-tx
(§9.7.15.16.6), and the parity test of `mbarrier.test_wait/try_wait.parity` (§9.7.15.16.19).

The ISA counts phases from zero and exposes only their parity. `phase` here is the full count, a
ghost the hardware does not keep, so that a protocol can state which phase a waiter means and the
model can tell a valid parity test from one the ISA leaves undefined.
-/

namespace NV.PTX

/-- Out-of-range counts and other conditions the ISA leaves undefined. -/
inductive MBarrier.Fault where
  /-- `mbarrier.init` count outside `[1, 2²⁰ - 1]` (Table 43). -/
  | initCount
  /-- An arrive-on whose count exceeds the pending arrivals: the pending count would leave its
  range `[0, 2²⁰ - 1]`. -/
  | overArrival
  /-- tx-count outside `[-(2²⁰ - 1), 2²⁰ - 1]`. -/
  | txRange
  /-- `.noComplete` arrival that completes the phase (§9.7.15.16.16). -/
  | noComplete
  /-- A parity wait that names neither the current phase nor the one before it. -/
  | staleParity
  /-- An arrive-on in a phase whose predecessor no wait has yet observed complete
  (§9.7.15.16.5.1: each phase needs a `test_wait`/`try_wait` returning true before any arrive-on
  of the next). -/
  | unobserved
  deriving DecidableEq, Repr, Hashable

/-- Largest arrival count and tx-count magnitude, `2²⁰ - 1` (Table 43, `.layout::v0`). -/
def MBarrier.countMax : Nat := 2 ^ 20 - 1

structure MBarrier where
  /-- Number of completed phases; the current phase is `phase` and its parity `phase % 2`. -/
  phase : Nat
  pending : Nat
  expected : Nat
  tx : Int
  /-- Ghost: whether a wait has observed the previous phase complete (vacuous in phase 0). -/
  observed : Bool := true
  deriving DecidableEq, Repr, Hashable

namespace MBarrier

/-- `mbarrier.init [addr], count`. -/
def init (count : Nat) : Except Fault MBarrier :=
  if 1 ≤ count ∧ count ≤ countMax then .ok ⟨0, count, count, 0, true⟩ else .error .initCount

/-- Phase completion: with no pending arrivals and no pending transactions the barrier moves to
the next phase and reloads the pending count from the expected count. -/
def settle (b : MBarrier) : MBarrier :=
  if b.pending = 0 ∧ b.tx = 0 then { b with phase := b.phase + 1, pending := b.expected, observed := false }
  else b

def txOk (t : Int) : Bool := -(countMax : Int) ≤ t ∧ t ≤ countMax

/-- The arrive-on operation with `count` arrivals. -/
def arrive (b : MBarrier) (count : Nat := 1) : Except Fault MBarrier :=
  if !b.observed then .error .unobserved
  else if count ≤ b.pending then .ok (settle { b with pending := b.pending - count })
  else .error .overArrival

/-- `mbarrier.arrive.noComplete`: an arrival that must not complete the phase. -/
def arriveNoComplete (b : MBarrier) (count : Nat) : Except Fault MBarrier := do
  let b' ← b.arrive count
  if b'.phase = b.phase then .ok b' else .error .noComplete

/-- The expect-tx operation. It never completes a phase: completion is checked by arrive-on and
complete-tx only (§9.7.15.16.8–9). -/
def expectTx (b : MBarrier) (bytes : Nat) : Except Fault MBarrier :=
  if txOk (b.tx + bytes) then .ok { b with tx := b.tx + bytes } else .error .txRange

/-- `mbarrier.arrive.expect_tx`: expect-tx, then one arrival. -/
def arriveExpectTx (b : MBarrier) (bytes : Nat) : Except Fault MBarrier := do
  (← b.expectTx bytes).arrive 1

/-- The complete-tx operation an asynchronous copy performs when `bytes` of its data have landed. -/
def completeTx (b : MBarrier) (bytes : Nat) : Except Fault MBarrier :=
  if txOk (b.tx - bytes) then .ok (settle { b with tx := b.tx - bytes }) else .error .txRange

/-- What `try_wait.parity` returns for `parity`: true when the phase of that parity just before the
current one has completed, that is, when `parity` differs from the current phase's parity. -/
def parityDone (b : MBarrier) (parity : Nat) : Bool := b.phase % 2 != parity % 2

/-- A parity wait by a thread that waits for phase `k` to complete: valid only when `k` is the
current phase (`false`) or the one immediately before it (`true`). `k = -1` is the phase before
the first, which a fresh barrier reports complete: producers start their empty-slot waits there. -/
def waitPhase (b : MBarrier) (k : Int) : Except Fault Bool :=
  if (b.phase : Int) = k then .ok false
  else if (b.phase : Int) = k + 1 then .ok true
  else .error .staleParity

/-- A wait that returned true has observed the previous phase. -/
def observe (b : MBarrier) : MBarrier := { b with observed := true }

/-- A valid wait returns what the hardware's parity test returns for the parity of `k`. -/
theorem waitPhase_parity {b : MBarrier} {k : Int} {r : Bool} (h : b.waitPhase k = .ok r) :
    b.parityDone (k % 2).toNat = r := by
  unfold waitPhase at h
  unfold parityDone
  split at h
  · cases h
    have : b.phase % 2 = (k % 2).toNat % 2 := by omega
    simp [this]
  · split at h
    · cases h
      have : b.phase % 2 ≠ (k % 2).toNat % 2 := by omega
      simp [this]
    · cases h

theorem waitPhase_eq_error {b : MBarrier} {k : Int} {f : Fault} (h : b.waitPhase k = .error f) :
    (b.phase : Int) ≠ k ∧ (b.phase : Int) ≠ k + 1 := by
  unfold waitPhase at h
  split at h
  · cases h
  · split at h
    · cases h
    · exact ⟨‹_›, ‹_›⟩

theorem waitPhase_eq_true {b : MBarrier} {k : Int} (h : b.waitPhase k = .ok true) :
    (b.phase : Int) = k + 1 := by
  unfold waitPhase at h
  split at h
  · cases h
  · split at h
    · assumption
    · cases h

theorem arrive_eq {b : MBarrier} {n : Nat} (ho : b.observed = true) (h : n ≤ b.pending) :
    b.arrive n = .ok (settle { b with pending := b.pending - n }) := by
  simp [arrive, ho, h]

theorem expectTx_eq {b : MBarrier} {n : Nat} (h : txOk (b.tx + n) = true) :
    b.expectTx n = .ok { b with tx := b.tx + n } := by
  simp [expectTx, h]

theorem completeTx_eq {b : MBarrier} {n : Nat} (h : txOk (b.tx - n) = true) :
    b.completeTx n = .ok (settle { b with tx := b.tx - n }) := by
  simp [completeTx, h]

/-- An `arrive.expect_tx` with bytes still to land never completes the phase. -/
theorem arriveExpectTx_eq {b : MBarrier} {n : Nat} (ho : b.observed = true)
    (h₁ : txOk (b.tx + n) = true) (h₂ : 1 ≤ b.pending) (h₃ : b.tx + n ≠ 0) :
    b.arriveExpectTx n = .ok { b with tx := b.tx + n, pending := b.pending - 1 } := by
  simp [arriveExpectTx, expectTx, h₁, bind, Except.bind, arrive, ho, h₂, settle, h₃]

theorem txOk_iff (t : Int) : txOk t = true ↔ -(countMax : Int) ≤ t ∧ t ≤ countMax := by
  simp [txOk]

theorem settle_of_pending {b : MBarrier} (h : b.pending ≠ 0) : settle b = b := by
  simp [settle, h]

theorem settle_of_tx {b : MBarrier} (h : b.tx ≠ 0) : settle b = b := by
  simp [settle, h]

theorem settle_of_not {b : MBarrier} (h : ¬(b.pending = 0 ∧ b.tx = 0)) : settle b = b := by
  simp only [settle]; split
  · exact absurd ‹_› h
  · rfl

theorem settle_of_done {b : MBarrier} (h₁ : b.pending = 0) (h₂ : b.tx = 0) :
    settle b = { b with phase := b.phase + 1, pending := b.expected, observed := false } := by
  simp [settle, h₁, h₂]

end MBarrier
end NV.PTX
