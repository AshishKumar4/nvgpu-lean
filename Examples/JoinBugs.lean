import Examples.Warps

/-!
# Three join and hand-off bugs, rejected; their fixes accepted

Small instances (two compute warps, so `NT = 64`) of a warp-specialized kernel's role protocols:
compute warps, a weight producer, an activation producer and a helper warp, joined by named
barriers and mbarriers. Each bug was found in a real kernel's development; the fixes are the ones
that shipped.

* **Mismatched join counts.** One change moves the weight producer's start onto barrier 8 with the
  activation producer (`bar.sync 8, 64` / `bar.arrive 8, 64`) and leaves the consumers and the
  activation producer on barrier 2 with `NT + 32` threads. Another replaces the weight producer's
  body with an unconditional join `bar.sync 2, NT + 64`. Composed, barrier 2's participants name
  different thread counts, which §9.7.15.1 leaves undefined (`mismatched_join_rejected`). The
  first change alone runs to completion in every interleaving (`producer_join_ok`).
* **A warm-up warp aliasing the helper.** An extra warp's rotated logical thread index equals the
  helper's, so it runs the helper's hand-off too, and arrives a second time on the partial-empty
  mbarrier (init count 1) after the real helper completed its phase and before the compute warps
  observed it: an arrive-on the ISA forbids (§9.7.15.16.5.1; `dry_alias_rejected`). Without the
  alias the hand-off completes (`handoff_completes`).
* **A start join on the end join's barrier.** Compute warps `bar.arrive 2` at start and
  `bar.sync 2` at the end, both with `NT + 64` threads, while the helper starts with `bar.sync 2`.
  Compute warps that reach the end first complete instance 0 with `2 × NT ≥ NT + 64` arrivals;
  the helper's start lands in instance 1 and nothing completes it
  (`shared_join_barrier_deadlocks`). With the start join on barrier 15, every interleaving
  completes (`separate_join_barrier_ok`).
-/

namespace Examples.Warps

open NV NV.Formal

/-- `NT`: two compute warps. -/
def NT : Nat := 64

/-! ## Bug 2: awready + producerJoin -/

/-- Warps 0 and 1 consume, 2 is the activation producer, 3 the weight producer. -/
def producerJoin (weightProducer : List Instr) : Program where
  warps := [[.barSync 2 (NT + 32)], [.barSync 2 (NT + 32)],
            [.barArrive 8 64, .barSync 2 (NT + 32)], weightProducer]
  mbarriers := []

theorem producer_join_ok :
    (system (producerJoin [.barSync 8 64])).exhaust (List.range 4) (healthy (producerJoin [.barSync 8 64])) 12
      (system (producerJoin [.barSync 8 64])).init = true := by
  decide +kernel

/-- The weight producer's `WS_JOIN()` meets the consumers on barrier 2 with another count. -/
theorem mismatched_join_rejected :
    ((system (producerJoin [.barSync 2 (NT + 64)])).run (init (producerJoin [.barSync 2 (NT + 64)])) [3, 0]).map (·.fault) =
      some (some (.barrier .countMismatch)) := by
  decide +kernel

/-! ## Bug 3: the dry warm-up warp aliasing the helper -/

/-- Mbarriers: 0, 1 partial-ready (`S.wsrf`, count 1); 2, 3 partial-empty (`S.wsre`, count 1). Two
units through two partial buffers. Compute warp `c` waits for its buffer's release, stores its
partial, joins the compute barrier, and warp 0 signals the helper; the helper waits, reads both
partials and releases the buffer. -/
def compute (c : Nat) : List Instr :=
  [0, 1].flatMap fun u =>
    [.wait (2 + u) (-1), .store (c * 2 + u) u, .barSync 1 NT] ++ if c = 0 then [.arrive u] else []

def helper : List Instr :=
  [0, 1].flatMap fun u => [.wait u 0, .load u u, .load (2 + u) u, .arrive (2 + u)]

def handoff (dryAliasesHelper : Bool) : Program where
  warps := [compute 0, compute 1, helper] ++ if dryAliasesHelper then [helper] else []
  mbarriers := [1, 1, 1, 1]

theorem handoff_completes :
    ((system (handoff false)).run (init (handoff false))
      [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2]).map
        (done (handoff false)) = some true := by
  decide +kernel

theorem dry_alias_rejected :
    ((system (handoff true)).run (init (handoff true))
      [0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3]).map (·.fault) =
        some (some (.mbarrier .unobserved)) := by
  decide +kernel

/-! ## warmupJoin v1: the start join on the end join's barrier -/

def warmupJoin (startBarrier : Nat) : Program where
  warps := [[.barArrive startBarrier (NT + 32), .barSync 2 (NT + 32)],
            [.barArrive startBarrier (NT + 32), .barSync 2 (NT + 32)],
            [.barSync startBarrier (NT + 32), .barSync 2 (NT + 32)]]
  mbarriers := []

theorem shared_join_barrier_deadlocks :
    ((system (warmupJoin 2)).run (init (warmupJoin 2)) [0, 1, 0, 1, 2]).map (deadlocked (warmupJoin 2)) = some true := by
  decide +kernel

theorem separate_join_barrier_ok :
    (system (warmupJoin 15)).exhaust (List.range 3) (healthy (warmupJoin 15)) 10 (system (warmupJoin 15)).init
      = true := by
  decide +kernel

end Examples.Warps
