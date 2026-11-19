import Examples.RingSafety

/-!
# The ring theorem is not vacuous, and it rejects the joined-ring count bug

* `weight_ring_valid`: a weight ring of 3 slots of 36 864 bytes, 16 consumer warps and one
  producer satisfies `Config.Valid`, so `ring_safe` applies to it.
* `small_ring_completes`: a small valid ring streams all its stages to completion: the safe
  transition system does make progress.
* `joined_ring_count1_rejected`: two producers (weights and activations) complete on one slot's
  full barrier, which must therefore be initialized with count 2 (one `expect_tx` per producer).
  Initialized with count 1, the weight producer's arrival and bytes complete the phase alone, and
  a consumer acquires the slot before the activations have landed. With count 2 the same ring is
  valid (`joined_ring_valid`).
-/

namespace Examples.Ring

open NV NV.Formal

/-- A weight ring: 3 slots of 36 864 bytes, 16 consumer warps, one producer. -/
def weightRing (stages : Nat) : Config where
  slots := 3
  consumers := 16
  producers := 1
  bytes _ := 36864
  stages
  fullCount := 1
  emptyCount := 16

theorem weight_ring_valid (stages : Nat) : (weightRing stages).Valid where
  slots_pos := by simp [weightRing]
  consumers_pos := by simp [weightRing]
  producers_pos := by simp [weightRing]
  bytes_pos _ _ := by simp [weightRing]
  bytes_fit := by simp [weightRing, sumBelow, PTX.MBarrier.countMax]
  full_count := rfl
  empty_count := rfl
  counts_fit := by simp [weightRing, PTX.MBarrier.countMax]

def small : Config where
  slots := 2
  consumers := 2
  producers := 1
  bytes _ := 8
  stages := 3
  fullCount := 1
  emptyCount := 2

theorem small_valid : small.Valid where
  slots_pos := by decide
  consumers_pos := by decide
  producers_pos := by decide
  bytes_pos _ _ := by simp [small]
  bytes_fit := by decide
  full_count := rfl
  empty_count := rfl
  counts_fit := by decide

/-- Three stages through two slots, the copies landing in two pieces each. -/
def smallTrace : List Action :=
  [.issue 0, .issue 0, .land 0 0 3, .land 0 0 5, .acquire 0, .acquire 1, .release 0, .release 1,
   .land 0 1 8, .acquire 0, .release 0, .issue 0, .acquire 1, .release 1, .land 0 0 8,
   .acquire 0, .acquire 1, .release 1, .release 0]

theorem small_ring_completes :
    ∃ s, (system small).run (init small) smallTrace = some s ∧ Done small s := by
  refine ⟨_, rfl, ?_⟩
  refine ⟨fun i hi => ?_, fun c hc => ?_, fun i hi w hw => ?_⟩
  · have : i = 0 := by simp [small] at hi; omega
    subst this; decide
  · have : c = 0 ∨ c = 1 := by simp [small] at hc; omega
    rcases this with rfl | rfl <;> decide
  · have hi0 : i = 0 := by simp [small] at hi; omega
    have : w = 0 ∨ w = 1 := by simp [small] at hw; omega
    subst hi0; rcases this with rfl | rfl <;> decide

/-- Weights and activations joined on one full barrier, initialized with `fullCount`. -/
def joinedRing (fullCount : Nat) : Config where
  slots := 3
  consumers := 16
  producers := 2
  bytes i := if i = 0 then 36864 else 32768
  stages := 8
  fullCount
  emptyCount := 16

theorem joined_ring_valid : (joinedRing 2).Valid where
  slots_pos := by decide
  consumers_pos := by decide
  producers_pos := by decide
  bytes_pos i _ := by simp [joinedRing]; split <;> decide
  bytes_fit := by decide
  full_count := rfl
  empty_count := rfl
  counts_fit := by decide

theorem joined_ring_count1_rejected :
    ((system (joinedRing 1)).run (init (joinedRing 1))
      [.issue 0, .land 0 0 36864, .acquire 0]).map (·.fault) = some (some .readBeforeLanded) := by
  decide +kernel

end Examples.Ring
