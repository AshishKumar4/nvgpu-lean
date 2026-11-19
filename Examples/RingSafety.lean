import Examples.Ring
import NV.Formal.Count

/-!
# The ring is safe for every valid configuration

`ring_safe`: for any slot count, consumer count, producer count, byte counts and stage count
whose barrier counts match (`Config.Valid`), no reachable state faults. No mbarrier is over-arrived
or waited on with a stale parity, no consumer acquires a slot before all of its stage's bytes have
landed, and no producer overwrites a slot a consumer still holds.

The invariant ties each barrier to thread progress. For slot `w`, `rounds S p w` counts the
stages among the first `p` that use it, so producer `i` has issued `rounds S (issued i) w` rounds
into `w` and consumer `c` has released `rounds S (stage c) w`. The full barrier's phase is the
number of completed rounds; every producer is in that round or one ahead; the pending count is the
producers not yet ahead and the tx-count the bytes still in flight from those ahead. The empty
barrier is the same with consumers and releases.
-/

namespace Examples.Ring

open NV NV.Formal

open NV.PTX

variable {cfg : Config}

section
variable (cfg)

/-- Producers that have arrived on slot `w`'s full barrier in its current phase. -/
def arrivedP (s : State) (w : Nat) : Nat :=
  countBelow cfg.producers fun i => rounds cfg.slots (s.issued i) w == (s.full w).phase + 1

/-- Bytes of the current phase's copies into slot `w` that have not landed. -/
def inflight (s : State) (w : Nat) : Nat :=
  sumBelow cfg.producers fun i =>
    if rounds cfg.slots (s.issued i) w = (s.full w).phase + 1 then s.pending i w else 0

/-- Consumers that have arrived on slot `w`'s empty barrier in its current phase. -/
def arrivedC (s : State) (w : Nat) : Nat :=
  countBelow cfg.consumers fun c => rounds cfg.slots (s.stage c) w == (s.empty w).phase + 1

structure Inv (s : State) : Prop where
  fault : s.fault = none
  issued_le : ∀ i < cfg.producers, s.issued i ≤ cfg.stages
  stage_le : ∀ c < cfg.consumers, s.stage c ≤ cfg.stages
  reading_lt : ∀ c < cfg.consumers, s.reading c = true → s.stage c < cfg.stages
  full_expected : ∀ w < cfg.slots, (s.full w).expected = cfg.producers
  full_rounds : ∀ w < cfg.slots, ∀ i < cfg.producers,
    rounds cfg.slots (s.issued i) w = (s.full w).phase ∨
    rounds cfg.slots (s.issued i) w = (s.full w).phase + 1
  full_pending : ∀ w < cfg.slots, (s.full w).pending + arrivedP cfg s w = cfg.producers
  full_tx : ∀ w < cfg.slots, (s.full w).tx = inflight cfg s w
  landed : ∀ w < cfg.slots, ∀ i < cfg.producers,
    rounds cfg.slots (s.issued i) w = (s.full w).phase → s.pending i w = 0
  pending_le : ∀ w < cfg.slots, ∀ i < cfg.producers, s.pending i w ≤ cfg.bytes i
  tag : ∀ w < cfg.slots, ∀ i < cfg.producers, 0 < rounds cfg.slots (s.issued i) w →
    ∃ q, s.tag i w = some q ∧ q % cfg.slots = w ∧ q / cfg.slots + 1 = rounds cfg.slots (s.issued i) w
  consumed : ∀ w < cfg.slots, ∀ c < cfg.consumers, rounds cfg.slots (s.stage c) w ≤ (s.full w).phase
  held : ∀ c < cfg.consumers, s.reading c = true →
    s.stage c / cfg.slots + 1 ≤ (s.full (s.stage c % cfg.slots)).phase
  empty_expected : ∀ w < cfg.slots, (s.empty w).expected = cfg.consumers
  empty_tx : ∀ w < cfg.slots, (s.empty w).tx = 0
  empty_rounds : ∀ w < cfg.slots, ∀ c < cfg.consumers,
    rounds cfg.slots (s.stage c) w = (s.empty w).phase ∨
    rounds cfg.slots (s.stage c) w = (s.empty w).phase + 1
  empty_pending : ∀ w < cfg.slots, (s.empty w).pending + arrivedC cfg s w = cfg.consumers
  ahead : ∀ w < cfg.slots, ∀ i < cfg.producers,
    rounds cfg.slots (s.issued i) w ≤ (s.empty w).phase + 1
  /-- A full barrier's last completion is unobserved only while no consumer has acquired it. -/
  full_seen : ∀ w < cfg.slots, (s.full w).observed = false → 0 < (s.full w).phase ∧
    ∀ c < cfg.consumers, rounds cfg.slots (s.stage c) w < (s.full w).phase ∧
      ¬(s.reading c = true ∧ s.stage c % cfg.slots = w ∧ s.stage c / cfg.slots + 1 = (s.full w).phase)
  /-- An empty barrier's last completion is unobserved only while no producer has issued past it. -/
  empty_seen : ∀ w < cfg.slots, (s.empty w).observed = false → 0 < (s.empty w).phase ∧
    ∀ i < cfg.producers, rounds cfg.slots (s.issued i) w ≤ (s.empty w).phase

end

@[simp] theorem update_same {β : Type} (f : Nat → β) (i : Nat) (v : β) : update f i v i = v := by
  simp [update]

@[simp] theorem update_other {β : Type} (f : Nat → β) {i j : Nat} (v : β) (h : j ≠ i) :
    update f i v j = f j := by
  simp [update, h]

@[simp] theorem update₂_same {β : Type} (f : Nat → Nat → β) (i w : Nat) (v : β) :
    update₂ f i w v i w = v := by
  simp [update₂]

theorem update₂_other {β : Type} (f : Nat → Nat → β) {i w j x : Nat} (v : β) (h : ¬(j = i ∧ x = w)) :
    update₂ f i w v j x = f j x := by
  simp [update₂, h]

theorem inv_init (hv : cfg.Valid) : Inv cfg (init cfg) := by
  have hr : ∀ w, rounds cfg.slots 0 w = 0 := fun _ => rfl
  refine
    { fault := rfl
      issued_le := fun _ _ => Nat.zero_le _
      stage_le := fun _ _ => Nat.zero_le _
      reading_lt := fun _ _ h => by simp [init] at h
      full_expected := fun _ _ => by simp [init, hv.full_count]
      full_rounds := fun w _ _ _ => by simp [init, hr]
      full_pending := fun w _ => ?_
      full_tx := fun w _ => ?_
      landed := fun _ _ _ _ _ => rfl
      pending_le := fun _ _ _ _ => Nat.zero_le _
      tag := fun w _ _ _ h => by simp [init, hr] at h
      consumed := fun w _ _ _ => by simp [init, hr]
      held := fun _ _ h => by simp [init] at h
      empty_expected := fun _ _ => by simp [init, hv.empty_count]
      empty_tx := fun _ _ => rfl
      empty_rounds := fun w _ _ _ => by simp [init, hr]
      empty_pending := fun w _ => ?_
      ahead := fun w _ _ _ => by simp [init, hr]
      full_seen := fun _ _ h => by simp [init] at h
      empty_seen := fun _ _ h => by simp [init] at h }
  · simp only [init, arrivedP, hr, hv.full_count]
    rw [(countBelow_zero_iff _ _).2 (fun _ _ => by simp)]; rfl
  · simp only [init, inflight, hr]
    rw [(sumBelow_zero_iff _ _).2 (fun _ _ => by simp)]; rfl
  · simp only [init, arrivedC, hr, hv.empty_count]
    rw [(countBelow_zero_iff _ _).2 (fun _ _ => by simp)]; rfl

/-! ## Facts the invariant implies -/

section
variable {s : State} (h : Inv cfg s)
include h

/-- A phase cannot run ahead of the round a producer issues next: the full barrier's phase is at
most every producer's round count, the empty barrier's at most every consumer's. -/
theorem full_le_issued {w i : Nat} (hw : w < cfg.slots) (hi : i < cfg.producers) :
    (s.full w).phase ≤ rounds cfg.slots (s.issued i) w := by
  rcases h.full_rounds w hw i hi with e | e <;> omega

theorem empty_le_stage {w c : Nat} (hw : w < cfg.slots) (hc : c < cfg.consumers) :
    (s.empty w).phase ≤ rounds cfg.slots (s.stage c) w := by
  rcases h.empty_rounds w hw c hc with e | e <;> omega

theorem inflight_le_bytes (w : Nat) (hw : w < cfg.slots) :
    inflight cfg s w ≤ sumBelow cfg.producers cfg.bytes := by
  apply sumBelow_le
  intro j hj
  split
  · exact h.pending_le w hw j hj
  · exact Nat.zero_le _

variable (hv : cfg.Valid)
include hv

/-- The empty phase trails the full phase, which trails the producers. -/
theorem empty_le_full {w : Nat} (hw : w < cfg.slots) : (s.empty w).phase ≤ (s.full w).phase :=
  Nat.le_trans (empty_le_stage h hw hv.consumers_pos) (h.consumed w hw 0 hv.consumers_pos)

theorem full_le_empty {w : Nat} (hw : w < cfg.slots) : (s.full w).phase ≤ (s.empty w).phase + 1 :=
  Nat.le_trans (full_le_issued h hw hv.producers_pos) (h.ahead w hw 0 hv.producers_pos)

end

theorem afterWait_eq_some {s t : State} {r : Except MBarrier.Fault Bool} {k : Unit → Option State}
    (hs : afterWait s r k = some t) : (r = .ok true ∧ k () = some t) ∨ ∃ f, r = .error f := by
  unfold afterWait at hs
  split at hs
  · exact .inl ⟨rfl, hs⟩
  · cases hs
  · exact .inr ⟨_, rfl⟩

theorem withBarrier_eq_some {s t : State} {r : Except MBarrier.Fault MBarrier}
    {k : MBarrier → Option State} (hs : withBarrier s r k = some t) :
    (∃ b, r = .ok b ∧ k b = some t) ∨ ∃ f, r = .error f := by
  unfold withBarrier at hs
  split at hs
  · exact .inl ⟨_, rfl, hs⟩
  · exact .inr ⟨_, rfl⟩

theorem rounds_update {S : Nat} {f : Nat → Nat} {i j q : Nat} (w : Nat) :
    rounds S (update f i (q + 1) j) w =
      if j = i then rounds S q w + (if q % S = w then 1 else 0) else rounds S (f j) w := by
  by_cases hj : j = i
  · subst hj; simp [rounds_succ]
  · simp [update, hj]

/-! ## Each action preserves the invariant -/

section
variable {s t : State} (hv : cfg.Valid) (h : Inv cfg s)
include hv h

theorem inv_issue {i : Nat} (hs : step cfg s (.issue i) = some t) : Inv cfg t := by
  have hS := hv.slots_pos
  simp only [step, h.fault, Option.isSome_none, Bool.false_eq_true, ite_false] at hs
  split at hs
  case isFalse => cases hs
  case isTrue hc =>
  obtain ⟨hi, hq⟩ := hc
  generalize hqd : s.issued i = q at hs hq
  have hw : q % cfg.slots < cfg.slots := Nat.mod_lt _ hS
  have hR : rounds cfg.slots q (q % cfg.slots) = q / cfg.slots := rounds_self hS q
  have hRi : rounds cfg.slots (s.issued i) (q % cfg.slots) = q / cfg.slots := hqd ▸ hR
  have hFi := full_le_issued h hw hi
  have hEF := empty_le_full h hv hw
  have hE1 := h.ahead _ hw i hi
  rw [hRi] at hFi hE1
  rcases afterWait_eq_some hs with ⟨hwait, hk⟩ | ⟨f, hf⟩
  case inr =>
    have := MBarrier.waitPhase_eq_error hf
    omega
  have hE : (s.empty (q % cfg.slots)).phase = q / cfg.slots := by
    have := MBarrier.waitPhase_eq_true hwait; omega
  have hF : (s.full (q % cfg.slots)).phase = q / cfg.slots := by omega
  -- No consumer holds the slot: one that did would be in a round both before and after `q / S`.
  have hfree : ((List.range cfg.consumers).any fun c =>
      s.reading c && s.stage c % cfg.slots == q % cfg.slots) = false := by
    rw [List.any_eq_false]
    intro c hc
    simp only [List.mem_range] at hc
    simp only [Bool.and_eq_true, beq_iff_eq, not_and]
    intro hr hcw
    have h₁ := h.held c hc hr
    have h₂ := empty_le_stage h hw hc
    rw [hcw, hF] at h₁
    rw [← hcw, rounds_self hS, hcw, hE] at h₂
    omega
  rw [hfree, ite_eq_right (by decide)] at hk
  -- Producer `i` has not arrived in this phase, so the barrier has room for its arrival and bytes.
  have hnot : (rounds cfg.slots (s.issued i) (q % cfg.slots) == (s.full (q % cfg.slots)).phase + 1)
      = false := by rw [hRi, hF]; simp
  have hpend : 1 ≤ (s.full (q % cfg.slots)).pending := by
    have hlt : arrivedP cfg s (q % cfg.slots) ≠ cfg.producers := fun he =>
      absurd ((countBelow_eq_iff _ _).1 he i hi) (by rw [hnot]; decide)
    have := h.full_pending _ hw
    have : arrivedP cfg s (q % cfg.slots) ≤ cfg.producers := countBelow_le _ _
    omega
  have hzero : (fun j => if rounds cfg.slots (s.issued j) (q % cfg.slots) =
      (s.full (q % cfg.slots)).phase + 1 then s.pending j (q % cfg.slots) else 0) i = 0 := by
    simp only [hRi, hF]; simp
  have hroom : inflight cfg s (q % cfg.slots) + cfg.bytes i ≤ MBarrier.countMax := by
    let others := fun j => if j = i then 0 else cfg.bytes j
    have hle : inflight cfg s (q % cfg.slots) ≤ sumBelow cfg.producers others := by
      apply sumBelow_le
      intro j hj
      by_cases hji : j = i
      · subst hji; simp only [others, ite_eq_left rfl]; exact Nat.le_of_eq hzero
      · simp only [others, ite_eq_right hji]; split
        · exact h.pending_le _ hw j hj
        · exact Nat.zero_le _
    have hsum := sumBelow_update (f := cfg.bytes) (g := others) hi (fun j _ hj => by simp [others, hj])
    have hoi : others i = 0 := by simp [others]
    have := hv.bytes_fit
    omega
  have hbytes := hv.bytes_pos i hi
  have htx := h.full_tx _ hw
  -- Every consumer has acquired the previous round of the slot, so the full barrier's last
  -- completion has been observed.
  have hseen : (s.full (q % cfg.slots)).observed = true := by
    cases hob : (s.full (q % cfg.slots)).observed
    · have := ((h.full_seen _ hw hob).2 0 hv.consumers_pos).1
      have := empty_le_stage h hw hv.consumers_pos (c := 0)
      omega
    · rfl
  have harr : (s.full (q % cfg.slots)).arriveExpectTx (cfg.bytes i) =
      .ok { s.full (q % cfg.slots) with
        tx := (s.full (q % cfg.slots)).tx + cfg.bytes i
        pending := (s.full (q % cfg.slots)).pending - 1 } := by
    exact MBarrier.arriveExpectTx_eq hseen (by rw [MBarrier.txOk_iff]; omega) hpend (by omega)
  rw [harr] at hk
  simp only [withBarrier, Option.some.injEq] at hk
  -- What changed: slot `q % S`'s full barrier, producer `i`'s count, its bytes in flight and tag.
  have tF : ∀ w, w ≠ q % cfg.slots → t.full w = s.full w := by intro w hw'; rw [← hk]; simp [hw']
  have tF₀ : t.full (q % cfg.slots) = { s.full (q % cfg.slots) with
      tx := (s.full (q % cfg.slots)).tx + cfg.bytes i
      pending := (s.full (q % cfg.slots)).pending - 1 } := by rw [← hk]; simp
  have tPh : ∀ w, (t.full w).phase = (s.full w).phase := by
    intro w; by_cases hw' : w = q % cfg.slots
    · subst hw'; rw [tF₀]
    · rw [tF w hw']
  have tEx : ∀ w, (t.full w).expected = (s.full w).expected := by
    intro w; by_cases hw' : w = q % cfg.slots
    · subst hw'; rw [tF₀]
    · rw [tF w hw']
  have tE : ∀ x, t.empty x = if x = q % cfg.slots then (s.empty x).observe else s.empty x := by
    intro x; rw [← hk]; by_cases hx : x = q % cfg.slots
    · subst hx; simp
    · simp [hx]
  have tEp : ∀ x, (t.empty x).phase = (s.empty x).phase ∧ (t.empty x).pending = (s.empty x).pending ∧
      (t.empty x).expected = (s.empty x).expected ∧ (t.empty x).tx = (s.empty x).tx := by
    intro x; rw [tE x]; split <;> exact ⟨rfl, rfl, rfl, rfl⟩
  have tS : t.stage = s.stage := by rw [← hk]
  have tRd : t.reading = s.reading := by rw [← hk]
  have tR : ∀ j w, rounds cfg.slots (t.issued j) w =
      rounds cfg.slots (s.issued j) w + if j = i ∧ w = q % cfg.slots then 1 else 0 := by
    intro j w
    rw [← hk]; simp only
    rw [rounds_update]
    by_cases hj : j = i
    · subst hj
      rw [ite_eq_left rfl, hqd]
      by_cases hww : w = q % cfg.slots
      · subst hww; simp
      · simp [hww, Ne.symm hww]
    · simp [hj]
  have tI : ∀ j, j ≠ i → t.issued j = s.issued j := by intro j hj; rw [← hk]; simp [hj]
  have tI₀ : t.issued i = q + 1 := by rw [← hk]; simp
  have tP : ∀ j w, ¬(j = i ∧ w = q % cfg.slots) → t.pending j w = s.pending j w := by
    intro j w hjw; rw [← hk]; exact update₂_other _ _ hjw
  have tP₀ : t.pending i (q % cfg.slots) = cfg.bytes i := by rw [← hk]; simp
  have tT : ∀ j w, ¬(j = i ∧ w = q % cfg.slots) → t.tag j w = s.tag j w := by
    intro j w hjw; rw [← hk]; exact update₂_other _ _ hjw
  have tT₀ : t.tag i (q % cfg.slots) = some q := by rw [← hk]; simp
  have tf : t.fault = none := by rw [← hk]
  clear hk hs
  refine
    { fault := tf
      issued_le := fun j hj => ?_
      stage_le := tS ▸ h.stage_le
      reading_lt := tS ▸ tRd ▸ h.reading_lt
      full_expected := fun w hw' => tEx w ▸ h.full_expected w hw'
      full_rounds := fun w hw' j hj => ?_
      full_pending := fun w hw' => ?_
      full_tx := fun w hw' => ?_
      landed := fun w hw' j hj => ?_
      pending_le := fun w hw' j hj => ?_
      tag := fun w hw' j hj => ?_
      consumed := fun w hw' c hc => by rw [tPh, tS]; exact h.consumed w hw' c hc
      held := fun c hc hr => by rw [tPh, tS]; rw [tRd] at hr; exact h.held c hc hr
      empty_expected := fun x hx => (tEp x).2.2.1 ▸ h.empty_expected x hx
      empty_tx := fun x hx => (tEp x).2.2.2 ▸ h.empty_tx x hx
      empty_rounds := fun x hx c hc => by rw [tS, (tEp x).1]; exact h.empty_rounds x hx c hc
      empty_pending := fun w hw' => ?_
      ahead := fun w hw' j hj => ?_
      full_seen := fun x hx hob => ?_
      empty_seen := fun x hx hob => ?_ }
  · by_cases hji : j = i
    · subst hji; rw [tI₀]; omega
    · rw [tI j hji]; exact h.issued_le j hj
  · rw [tR, tPh]
    by_cases hc : j = i ∧ w = q % cfg.slots
    · obtain ⟨rfl, rfl⟩ := hc; rw [ite_eq_left ⟨rfl, rfl⟩, hRi, hF]; omega
    · rw [ite_eq_right hc, Nat.add_zero]; exact h.full_rounds w hw' j hj
  · have hfp := h.full_pending w hw'
    unfold arrivedP at hfp ⊢
    by_cases hww : w = q % cfg.slots
    · subst hww
      have hc : countBelow cfg.producers (fun j => rounds cfg.slots (t.issued j) (q % cfg.slots) ==
            (t.full (q % cfg.slots)).phase + 1) =
          countBelow cfg.producers (fun j => rounds cfg.slots (s.issued j) (q % cfg.slots) ==
            (s.full (q % cfg.slots)).phase + 1) + 1 := by
        apply countBelow_bump hi
        · intro j _ hj; simp [tR, tPh, hj]
        · exact hnot
        · simp [tR, tPh, hRi, hF]
      rw [hc, tF₀]; simp only; omega
    · have hsame : ∀ j < cfg.producers, (rounds cfg.slots (t.issued j) w == (t.full w).phase + 1) =
          (rounds cfg.slots (s.issued j) w == (s.full w).phase + 1) := by
        intro j _; simp [tR, tPh, hww]
      rw [countBelow_congr hsame, tF w hww]
      exact hfp
  · have hft := h.full_tx w hw'
    unfold inflight at hft ⊢
    by_cases hww : w = q % cfg.slots
    · subst hww
      have hsum := sumBelow_update (n := cfg.producers) hi
        (f := fun j => if rounds cfg.slots (s.issued j) (q % cfg.slots) = (s.full (q % cfg.slots)).phase + 1
          then s.pending j (q % cfg.slots) else 0)
        (g := fun j => if rounds cfg.slots (t.issued j) (q % cfg.slots) = (t.full (q % cfg.slots)).phase + 1
          then t.pending j (q % cfg.slots) else 0)
        (fun j _ hj => by simp [tR, tPh, hj, tP j (q % cfg.slots) (fun hc => hj hc.1)])
      have hfi : (if rounds cfg.slots (s.issued i) (q % cfg.slots) = (s.full (q % cfg.slots)).phase + 1
          then s.pending i (q % cfg.slots) else 0) = 0 := by simp [hRi, hF]
      have hgi : (if rounds cfg.slots (t.issued i) (q % cfg.slots) = (t.full (q % cfg.slots)).phase + 1
          then t.pending i (q % cfg.slots) else 0) = cfg.bytes i := by simp [tR, tPh, hRi, hF, tP₀]
      rw [hfi, hgi] at hsum
      have htx' : (t.full (q % cfg.slots)).tx = (s.full (q % cfg.slots)).tx + cfg.bytes i := by rw [tF₀]
      rw [htx']
      omega
    · have hsame : ∀ j < cfg.producers,
          (if rounds cfg.slots (t.issued j) w = (t.full w).phase + 1 then t.pending j w else 0) =
          (if rounds cfg.slots (s.issued j) w = (s.full w).phase + 1 then s.pending j w else 0) := by
        intro j _; simp [tR, tPh, hww, tP j w (fun hc => hww hc.2)]
      rw [sumBelow_congr hsame, tF w hww]
      exact hft
  · intro hr
    rw [tR, tPh] at hr
    by_cases hc : j = i ∧ w = q % cfg.slots
    · obtain ⟨rfl, rfl⟩ := hc; rw [ite_eq_left ⟨rfl, rfl⟩, hRi, hF] at hr; omega
    · rw [ite_eq_right hc, Nat.add_zero] at hr; rw [tP j w hc]; exact h.landed w hw' j hj hr
  · by_cases hc : j = i ∧ w = q % cfg.slots
    · obtain ⟨rfl, rfl⟩ := hc; rw [tP₀]; exact Nat.le_refl _
    · rw [tP j w hc]; exact h.pending_le w hw' j hj
  · intro hpos
    by_cases hc : j = i ∧ w = q % cfg.slots
    · obtain ⟨rfl, rfl⟩ := hc
      refine ⟨q, tT₀, rfl, ?_⟩
      rw [tR, ite_eq_left ⟨rfl, rfl⟩, hRi]
    · rw [tR, ite_eq_right hc, Nat.add_zero] at hpos ⊢
      rw [tT j w hc]; exact h.tag w hw' j hj hpos
  · unfold arrivedC; rw [tS, (tEp w).1, (tEp w).2.1]; exact h.empty_pending w hw'
  · rw [tR, (tEp w).1]
    by_cases hc : j = i ∧ w = q % cfg.slots
    · obtain ⟨rfl, rfl⟩ := hc; rw [ite_eq_left ⟨rfl, rfl⟩, hRi, hE]; exact Nat.le_refl _
    · rw [ite_eq_right hc, Nat.add_zero]; exact h.ahead w hw' j hj
  · have hob' : (s.full x).observed = false := by
      by_cases hxw : x = q % cfg.slots
      · subst hxw; rw [tF₀] at hob; exact hob
      · rw [tF x hxw] at hob; exact hob
    rw [tPh, tS, tRd]; exact h.full_seen x hx hob'
  · by_cases hxw : x = q % cfg.slots
    · subst hxw; rw [tE, ite_eq_left rfl] at hob; cases hob
    · rw [tE, ite_eq_right hxw] at hob
      refine ⟨(tEp x).1 ▸ (h.empty_seen x hx hob).1, fun j hj => ?_⟩
      rw [tR, (tEp x).1, ite_eq_right (fun hc => hxw hc.2), Nat.add_zero]
      exact (h.empty_seen x hx hob).2 j hj

theorem inv_land {i w k : Nat} (hs : step cfg s (.land i w k) = some t) : Inv cfg t := by
  simp only [step, h.fault, Option.isSome_none, Bool.false_eq_true, ite_false] at hs
  split at hs
  case isFalse => cases hs
  case isTrue hc =>
  obtain ⟨hi, hw, hk, hkp⟩ := hc
  -- Producer `i`'s copy is in flight, so it is ahead in the phase and its bytes are counted.
  have hahead : rounds cfg.slots (s.issued i) w = (s.full w).phase + 1 := by
    rcases h.full_rounds w hw i hi with e | e
    · have := h.landed w hw i hi e; omega
    · exact e
  have hfi : (if rounds cfg.slots (s.issued i) w = (s.full w).phase + 1 then s.pending i w else 0)
      = s.pending i w := by simp [hahead]
  have hge : s.pending i w ≤ inflight cfg s w := by
    have := le_sumBelow (fun j => if rounds cfg.slots (s.issued j) w = (s.full w).phase + 1
      then s.pending j w else 0) hi
    rw [hfi] at this; exact this
  have hle := inflight_le_bytes h w hw
  have hfit := hv.bytes_fit
  have htx := h.full_tx w hw
  have hok : MBarrier.txOk ((s.full w).tx - k) = true := by rw [MBarrier.txOk_iff]; omega
  rw [MBarrier.completeTx_eq hok] at hs
  simp only [withBarrier, Option.some.injEq] at hs
  have tE : t.empty = s.empty := by rw [← hs]
  have tS : t.stage = s.stage := by rw [← hs]
  have tRd : t.reading = s.reading := by rw [← hs]
  have tI : t.issued = s.issued := by rw [← hs]
  have tT : t.tag = s.tag := by rw [← hs]
  have tf : t.fault = none := by rw [← hs]
  have tF : ∀ x, x ≠ w → t.full x = s.full x := by intro x hx; rw [← hs]; simp [hx]
  have tF₀ : t.full w = MBarrier.settle { s.full w with tx := (s.full w).tx - k } := by rw [← hs]; simp
  have tP : ∀ j x, ¬(j = i ∧ x = w) → t.pending j x = s.pending j x := by
    intro j x hjx; rw [← hs]; exact update₂_other _ _ hjx
  have tP₀ : t.pending i w = s.pending i w - k := by rw [← hs]; simp
  have hsum := sumBelow_update (n := cfg.producers) hi
    (f := fun j => if rounds cfg.slots (s.issued j) w = (s.full w).phase + 1 then s.pending j w else 0)
    (g := fun j => if rounds cfg.slots (s.issued j) w = (s.full w).phase + 1 then t.pending j w else 0)
    (fun j _ hj => by simp [tP j w (fun hc => hj hc.1)])
  have hgi : (if rounds cfg.slots (s.issued i) w = (s.full w).phase + 1 then t.pending i w else 0)
      = s.pending i w - k := by simp [hahead, tP₀]
  rw [hgi, hfi] at hsum
  -- Unchanged slots and the parts of the invariant that do not mention slot `w`'s full barrier.
  have common : ∀ x < cfg.slots, x ≠ w →
      (t.full x).pending + arrivedP cfg t x = cfg.producers ∧ (t.full x).tx = inflight cfg t x := by
    intro x hx hxw
    have e₁ : arrivedP cfg t x = arrivedP cfg s x := by unfold arrivedP; rw [tI, tF x hxw]
    have e₂ : inflight cfg t x = inflight cfg s x := by
      unfold inflight; rw [tI, tF x hxw]
      exact sumBelow_congr fun j _ => by simp [tP j x (fun hc => hxw hc.2)]
    rw [e₁, e₂, tF x hxw]; exact ⟨h.full_pending x hx, h.full_tx x hx⟩
  by_cases hdone : (s.full w).pending = 0 ∧ (s.full w).tx - k = 0
  · -- The last bytes of the phase landed: every producer is ahead and nothing is in flight.
    obtain ⟨hp0, htx0⟩ := hdone
    rw [MBarrier.settle_of_done (by simp only; exact hp0) (by simp only; exact htx0)] at tF₀
    have hall : ∀ j < cfg.producers, rounds cfg.slots (s.issued j) w = (s.full w).phase + 1 := by
      have := h.full_pending w hw
      have := (countBelow_eq_iff _ _).1 (show arrivedP cfg s w = cfg.producers by omega)
      intro j hj; simpa using this j hj
    have hnone : ∀ j < cfg.producers, t.pending j w = 0 := by
      have h0 : sumBelow cfg.producers (fun j => if rounds cfg.slots (s.issued j) w =
          (s.full w).phase + 1 then t.pending j w else 0) = 0 := by unfold inflight at htx; omega
      intro j hj
      have := (sumBelow_zero_iff _ _).1 h0 j hj
      simpa [hall j hj] using this
    have tPh : (t.full w).phase = (s.full w).phase + 1 := by rw [tF₀]
    refine
      { fault := tf
        issued_le := tI ▸ h.issued_le
        stage_le := tS ▸ h.stage_le
        reading_lt := tS ▸ tRd ▸ h.reading_lt
        full_expected := fun x hx => ?_
        full_rounds := fun x hx j hj => ?_
        full_pending := fun x hx => ?_
        full_tx := fun x hx => ?_
        landed := fun x hx j hj => ?_
        pending_le := fun x hx j hj => ?_
        tag := tT ▸ tI ▸ fun x hx j hj hp => ?_
        consumed := fun x hx c hc => ?_
        held := fun c hc hr => ?_
        empty_expected := tE ▸ h.empty_expected
        empty_tx := tE ▸ h.empty_tx
        empty_rounds := tE ▸ tS ▸ h.empty_rounds
        empty_pending := fun x hx => by unfold arrivedC; rw [tE, tS]; exact h.empty_pending x hx
        ahead := tE ▸ tI ▸ h.ahead
        full_seen := fun x hx hob => ?_
        empty_seen := tE ▸ tI ▸ h.empty_seen }
    · by_cases hxw : x = w
      · subst hxw; rw [tF₀]; exact h.full_expected _ hw
      · rw [tF x hxw]; exact h.full_expected x hx
    · rw [tI]
      by_cases hxw : x = w
      · subst hxw; rw [tPh]; exact .inl (hall j hj)
      · rw [tF x hxw]; exact h.full_rounds x hx j hj
    · by_cases hxw : x = w
      · subst hxw
        have e : arrivedP cfg t x = 0 := by
          unfold arrivedP; rw [tI, tPh]
          exact (countBelow_zero_iff _ _).2 fun j hj => by simp [hall j hj]
        rw [e, tF₀]; exact h.full_expected _ hw
      · exact (common x hx hxw).1
    · by_cases hxw : x = w
      · subst hxw
        have e : inflight cfg t x = 0 := by
          unfold inflight; rw [tI, tPh]
          exact (sumBelow_zero_iff _ _).2 fun j hj => by simp [hall j hj]
        rw [e, tF₀]
        show (s.full x).tx - (k : Int) = ((0 : Nat) : Int)
        omega
      · exact (common x hx hxw).2
    · intro _
      by_cases hxw : x = w
      · subst hxw; exact hnone j hj
      · rw [tP j x (fun hc => hxw hc.2)]; rw [tI, tF x hxw] at *; exact h.landed x hx j hj ‹_›
    · by_cases hc : j = i ∧ x = w
      · obtain ⟨rfl, rfl⟩ := hc; rw [tP₀]; have := h.pending_le _ hw _ hi; omega
      · rw [tP j x hc]; exact h.pending_le x hx j hj
    · exact h.tag x hx j hj hp
    · rw [tS]
      by_cases hxw : x = w
      · subst hxw; rw [tPh]; have := h.consumed _ hw c hc; omega
      · rw [tF x hxw]; exact h.consumed x hx c hc
    · rw [tS]; rw [tRd] at hr
      by_cases hxw : s.stage c % cfg.slots = w
      · rw [hxw, tPh]; have := h.held c hc hr; rw [hxw] at this; omega
      · rw [tF _ hxw]; exact h.held c hc hr
    · rw [tS, tRd]
      by_cases hxw : x = w
      · -- Just completed: no consumer can have acquired it yet.
        subst hxw
        rw [tPh]
        refine ⟨Nat.succ_pos _, fun c hc => ⟨?_, ?_⟩⟩
        · have := h.consumed x hx c hc; omega
        · rintro ⟨hr, hcw, hround⟩
          have := h.held c hc hr; rw [hcw] at this; omega
      · rw [tF x hxw] at hob ⊢; exact h.full_seen x hx hob
  · -- The phase continues: only the tx-count and the producer's bytes in flight change.
    rw [MBarrier.settle_of_not (by simp only; exact hdone)] at tF₀
    have tPh : ∀ x, (t.full x).phase = (s.full x).phase := by
      intro x; by_cases hxw : x = w
      · subst hxw; rw [tF₀]
      · rw [tF x hxw]
    refine
      { fault := tf
        issued_le := tI ▸ h.issued_le
        stage_le := tS ▸ h.stage_le
        reading_lt := tS ▸ tRd ▸ h.reading_lt
        full_expected := fun x hx => ?_
        full_rounds := fun x hx j hj => by rw [tI, tPh]; exact h.full_rounds x hx j hj
        full_pending := fun x hx => ?_
        full_tx := fun x hx => ?_
        landed := fun x hx j hj hr => ?_
        pending_le := fun x hx j hj => ?_
        tag := tT ▸ tI ▸ h.tag
        consumed := fun x hx c hc => by rw [tS, tPh]; exact h.consumed x hx c hc
        held := fun c hc hr => by rw [tS, tPh]; rw [tRd] at hr; exact h.held c hc hr
        empty_expected := tE ▸ h.empty_expected
        empty_tx := tE ▸ h.empty_tx
        empty_rounds := tE ▸ tS ▸ h.empty_rounds
        empty_pending := fun x hx => by unfold arrivedC; rw [tE, tS]; exact h.empty_pending x hx
        ahead := tE ▸ tI ▸ h.ahead
        full_seen := fun x hx hob => ?_
        empty_seen := tE ▸ tI ▸ h.empty_seen }
    · by_cases hxw : x = w
      · subst hxw; rw [tF₀]; exact h.full_expected _ hw
      · rw [tF x hxw]; exact h.full_expected x hx
    · by_cases hxw : x = w
      · subst hxw
        have e : arrivedP cfg t x = arrivedP cfg s x := by unfold arrivedP; rw [tI, tPh]
        rw [e, tF₀]; exact h.full_pending _ hw
      · exact (common x hx hxw).1
    · by_cases hxw : x = w
      · subst hxw
        have e : inflight cfg t x = sumBelow cfg.producers (fun j => if rounds cfg.slots (s.issued j) x =
            (s.full x).phase + 1 then t.pending j x else 0) := by unfold inflight; rw [tI, tPh]
        rw [e, tF₀]; show (s.full x).tx - (k : Int) = _
        unfold inflight at htx
        omega
      · exact (common x hx hxw).2
    · rw [tI, tPh] at hr
      by_cases hc : j = i ∧ x = w
      · obtain ⟨rfl, rfl⟩ := hc; omega
      · rw [tP j x hc]; exact h.landed x hx j hj hr
    · by_cases hc : j = i ∧ x = w
      · obtain ⟨rfl, rfl⟩ := hc; rw [tP₀]; have := h.pending_le _ hw _ hi; omega
      · rw [tP j x hc]; exact h.pending_le x hx j hj
    · have hob' : (s.full x).observed = false := by
        by_cases hxw : x = w
        · subst hxw; rw [tF₀] at hob; exact hob
        · rw [tF x hxw] at hob; exact hob
      rw [tPh, tS, tRd]; exact h.full_seen x hx hob'

theorem inv_acquire {c : Nat} (hs : step cfg s (.acquire c) = some t) : Inv cfg t := by
  have hS := hv.slots_pos
  simp only [step, h.fault, Option.isSome_none, Bool.false_eq_true, ite_false] at hs
  split at hs
  case isFalse => cases hs
  case isTrue hcond =>
  obtain ⟨hc, hq, hnr⟩ := hcond
  generalize hqd : s.stage c = q at hs hq
  have hw : q % cfg.slots < cfg.slots := Nat.mod_lt _ hS
  have hRc : rounds cfg.slots (s.stage c) (q % cfg.slots) = q / cfg.slots := hqd ▸ rounds_self hS q
  have hF₁ := h.consumed _ hw c hc
  have hE₁ := empty_le_stage h hw hc
  have hF₂ := full_le_empty h hv hw
  rw [hRc] at hF₁ hE₁
  rcases afterWait_eq_some hs with ⟨hwait, hk⟩ | ⟨f, hf⟩
  case inr => have := MBarrier.waitPhase_eq_error hf; omega
  have hF : (s.full (q % cfg.slots)).phase = q / cfg.slots + 1 := by
    have := MBarrier.waitPhase_eq_true hwait; omega
  -- Every producer issued round `q / S` of the slot and all of its bytes landed.
  have hall : ((List.range cfg.producers).all fun i =>
      s.tag i (q % cfg.slots) == some q && s.pending i (q % cfg.slots) == 0) = true := by
    rw [List.all_eq_true]
    intro i hi
    simp only [List.mem_range] at hi
    have hRi : rounds cfg.slots (s.issued i) (q % cfg.slots) = (s.full (q % cfg.slots)).phase := by
      have := h.full_rounds _ hw i hi; have := h.ahead _ hw i hi; omega
    obtain ⟨q', hq', hmod, hdiv⟩ := h.tag _ hw i hi (by omega)
    have : q' = q := by
      have e₁ := Nat.mod_add_div q' cfg.slots
      have e₂ := Nat.mod_add_div q cfg.slots
      have : q' / cfg.slots = q / cfg.slots := by omega
      rw [hmod, this] at e₁; omega
    simp [hq', this, h.landed _ hw i hi hRi]
  rw [ite_eq_left hall] at hk
  simp only [Option.some.injEq] at hk
  have tF : ∀ x, t.full x = if x = q % cfg.slots then (s.full x).observe else s.full x := by
    intro x; rw [← hk]; by_cases hx : x = q % cfg.slots
    · subst hx; simp
    · simp [hx]
  have tFp : ∀ x, (t.full x).phase = (s.full x).phase ∧ (t.full x).pending = (s.full x).pending ∧
      (t.full x).expected = (s.full x).expected ∧ (t.full x).tx = (s.full x).tx := by
    intro x; rw [tF x]; split <;> exact ⟨rfl, rfl, rfl, rfl⟩
  have tRd : ∀ d, t.reading d = if d = c then true else s.reading d := by
    intro d; rw [← hk]; by_cases hd : d = c
    · subst hd; simp
    · simp [hd]
  have tE : t.empty = s.empty := by rw [← hk]
  have tS : t.stage = s.stage := by rw [← hk]
  have tI : t.issued = s.issued := by rw [← hk]
  have tP : t.pending = s.pending := by rw [← hk]
  have tT : t.tag = s.tag := by rw [← hk]
  have tf : t.fault = none := by rw [← hk]
  clear hk hs
  refine
    { fault := tf
      issued_le := tI ▸ h.issued_le
      stage_le := tS ▸ h.stage_le
      reading_lt := fun d hd hr => ?_
      full_expected := fun x hx => (tFp x).2.2.1 ▸ h.full_expected x hx
      full_rounds := fun x hx i hi => by rw [tI, (tFp x).1]; exact h.full_rounds x hx i hi
      full_pending := fun x hx => ?_
      full_tx := fun x hx => ?_
      landed := fun x hx i hi hr => by rw [tI, (tFp x).1] at hr; rw [tP]; exact h.landed x hx i hi hr
      pending_le := tP ▸ h.pending_le
      tag := tT ▸ tI ▸ h.tag
      consumed := fun x hx d hd => by rw [tS, (tFp x).1]; exact h.consumed x hx d hd
      held := fun d hd hr => ?_
      empty_expected := tE ▸ h.empty_expected
      empty_tx := tE ▸ h.empty_tx
      empty_rounds := tE ▸ tS ▸ h.empty_rounds
      empty_pending := fun x hx => by unfold arrivedC; rw [tE, tS]; exact h.empty_pending x hx
      ahead := tE ▸ tI ▸ h.ahead
      full_seen := fun x hx hob => ?_
      empty_seen := tE ▸ tI ▸ h.empty_seen }
  · rw [tS]; rw [tRd] at hr
    by_cases hdc : d = c
    · subst hdc; rw [hqd]; exact hq
    · rw [ite_eq_right hdc] at hr; exact h.reading_lt d hd hr
  · unfold arrivedP; rw [tI, (tFp x).1, (tFp x).2.1]; exact h.full_pending x hx
  · unfold inflight; rw [tI, tP, (tFp x).1, (tFp x).2.2.2]; exact h.full_tx x hx
  · rw [tS, (tFp _).1]; rw [tRd] at hr
    by_cases hdc : d = c
    · subst hdc; rw [hqd, hF]; exact Nat.le_refl _
    · rw [ite_eq_right hdc] at hr; exact h.held d hd hr
  · by_cases hxw : x = q % cfg.slots
    · subst hxw; rw [tF, ite_eq_left rfl] at hob; cases hob
    · rw [tF, ite_eq_right hxw] at hob
      obtain ⟨hpos, hcons⟩ := h.full_seen x hx hob
      refine ⟨(tFp x).1 ▸ hpos, fun d hd => ?_⟩
      rw [tS, (tFp x).1, tRd]
      obtain ⟨h₁, h₂⟩ := hcons d hd
      refine ⟨h₁, fun ⟨hr, hdx, hround⟩ => ?_⟩
      by_cases hdc : d = c
      · subst hdc; rw [hqd] at hdx; exact hxw hdx.symm
      · rw [ite_eq_right hdc] at hr; exact h₂ ⟨hr, hdx, hround⟩

theorem inv_release {c : Nat} (hs : step cfg s (.release c) = some t) : Inv cfg t := by
  have hS := hv.slots_pos
  simp only [step, h.fault, Option.isSome_none, Bool.false_eq_true, ite_false] at hs
  split at hs
  case isFalse => cases hs
  case isTrue hcond =>
  obtain ⟨hc, hr⟩ := hcond
  have hq := h.reading_lt c hc hr
  generalize hqd : s.stage c = q at hs hq
  have hw : q % cfg.slots < cfg.slots := Nat.mod_lt _ hS
  have hRc : rounds cfg.slots (s.stage c) (q % cfg.slots) = q / cfg.slots := hqd ▸ rounds_self hS q
  have hheld := h.held c hc hr
  rw [hqd] at hheld
  have hE₁ := empty_le_stage h hw hc
  have hF₂ := full_le_empty h hv hw
  rw [hRc] at hE₁
  have hE : (s.empty (q % cfg.slots)).phase = q / cfg.slots := by omega
  -- The producers have issued the round, so they observed the empty barrier's last completion.
  have hobs : (s.empty (q % cfg.slots)).observed = true := by
    cases hob : (s.empty (q % cfg.slots)).observed
    · have := (h.empty_seen _ hw hob).2 0 hv.producers_pos
      have := full_le_issued h hw hv.producers_pos (i := 0)
      omega
    · rfl
  have hnot : (rounds cfg.slots (s.stage c) (q % cfg.slots) == (s.empty (q % cfg.slots)).phase + 1)
      = false := by rw [hRc, hE]; simp
  have hpend : 1 ≤ (s.empty (q % cfg.slots)).pending := by
    have hlt : arrivedC cfg s (q % cfg.slots) ≠ cfg.consumers := fun he =>
      absurd ((countBelow_eq_iff _ _).1 he c hc) (by rw [hnot]; decide)
    have := h.empty_pending _ hw
    have : arrivedC cfg s (q % cfg.slots) ≤ cfg.consumers := countBelow_le _ _
    omega
  rw [MBarrier.arrive_eq hobs hpend] at hs
  simp only [withBarrier, Option.some.injEq] at hs
  have tE : ∀ x, x ≠ q % cfg.slots → t.empty x = s.empty x := by intro x hx; rw [← hs]; simp [hx]
  have tE₀ : t.empty (q % cfg.slots) =
      MBarrier.settle { s.empty (q % cfg.slots) with pending := (s.empty (q % cfg.slots)).pending - 1 } := by
    rw [← hs]; simp
  have tR : ∀ d x, rounds cfg.slots (t.stage d) x =
      rounds cfg.slots (s.stage d) x + if d = c ∧ x = q % cfg.slots then 1 else 0 := by
    intro d x
    rw [← hs]; simp only
    rw [rounds_update]
    by_cases hd : d = c
    · subst hd
      rw [ite_eq_left rfl, hqd]
      by_cases hxw : x = q % cfg.slots
      · subst hxw; simp
      · simp [hxw, Ne.symm hxw]
    · simp [hd]
  have tSc : t.stage c = q + 1 := by rw [← hs]; simp
  have tS : ∀ d, d ≠ c → t.stage d = s.stage d := by intro d hd; rw [← hs]; simp [hd]
  have tRd : ∀ d, t.reading d = if d = c then false else s.reading d := by
    intro d; rw [← hs]; by_cases hd : d = c
    · subst hd; simp
    · simp [hd]
  have tF : t.full = s.full := by rw [← hs]
  have tI : t.issued = s.issued := by rw [← hs]
  have tP : t.pending = s.pending := by rw [← hs]
  have tT : t.tag = s.tag := by rw [← hs]
  have tf : t.fault = none := by rw [← hs]
  clear hs
  -- Consumer `c` joins the consumers that released round `q / S` of the slot.
  have hcount : countBelow cfg.consumers (fun d => rounds cfg.slots (t.stage d) (q % cfg.slots) ==
      (s.empty (q % cfg.slots)).phase + 1) = arrivedC cfg s (q % cfg.slots) + 1 := by
    unfold arrivedC
    apply countBelow_bump hc
    · intro d _ hd; simp [tR, hd]
    · exact hnot
    · simp [tR, hRc, hE]
  have hpe := h.empty_pending _ hw
  have htx0 := h.empty_tx _ hw
  have common : ∀ x < cfg.slots, x ≠ q % cfg.slots →
      (t.empty x).pending + arrivedC cfg t x = cfg.consumers := by
    intro x hx hxw
    have e : arrivedC cfg t x = arrivedC cfg s x := by
      unfold arrivedC; rw [tE x hxw]
      exact countBelow_congr fun d _ => by simp [tR, hxw]
    rw [e, tE x hxw]; exact h.empty_pending x hx
  have hEle : ∀ x, (s.empty x).phase ≤ (t.empty x).phase := by
    intro x; by_cases hxw : x = q % cfg.slots
    · subst hxw; rw [tE₀]; unfold MBarrier.settle; split <;> simp
    · rw [tE x hxw]; exact Nat.le_refl _
  have shared : ∀ x < cfg.slots, ∀ d < cfg.consumers,
      rounds cfg.slots (t.stage d) x ≤ (t.full x).phase := by
    intro x hx d hd
    rw [tR, tF]
    by_cases hc' : d = c ∧ x = q % cfg.slots
    · obtain ⟨rfl, rfl⟩ := hc'; rw [ite_eq_left ⟨rfl, rfl⟩, hRc]; exact hheld
    · rw [ite_eq_right hc', Nat.add_zero]; exact h.consumed x hx d hd
  have fseen : ∀ x < cfg.slots, (t.full x).observed = false → 0 < (t.full x).phase ∧
      ∀ d < cfg.consumers, rounds cfg.slots (t.stage d) x < (t.full x).phase ∧
        ¬(t.reading d = true ∧ t.stage d % cfg.slots = x ∧ t.stage d / cfg.slots + 1 = (t.full x).phase) := by
    intro x hx hob
    rw [tF] at hob ⊢
    obtain ⟨hpos, hcons⟩ := h.full_seen x hx hob
    refine ⟨hpos, fun d hd => ⟨?_, fun ⟨hrd, _⟩ => ?_⟩⟩
    · rw [tR]
      by_cases hc' : d = c ∧ x = q % cfg.slots
      · obtain ⟨rfl, rfl⟩ := hc'
        rw [ite_eq_left ⟨rfl, rfl⟩, hRc]
        have := (hcons d hd).2
        rw [hqd] at this
        have : ¬(q / cfg.slots + 1 = (s.full (q % cfg.slots)).phase) := fun he => this ⟨hr, rfl, he⟩
        omega
      · rw [ite_eq_right hc', Nat.add_zero]; exact (hcons d hd).1
    · rw [tRd] at hrd
      by_cases hdc : d = c
      · rw [ite_eq_left hdc] at hrd; cases hrd
      · rw [ite_eq_right hdc] at hrd
        rename_i hrest
        rw [tS d hdc] at hrest
        exact (hcons d hd).2 ⟨hrd, hrest⟩
  by_cases hdone : (s.empty (q % cfg.slots)).pending - 1 = 0 ∧ (s.empty (q % cfg.slots)).tx = 0
  · -- The last consumer released the round: every consumer is one round further.
    rw [MBarrier.settle_of_done (b := { s.empty (q % cfg.slots) with
      pending := (s.empty (q % cfg.slots)).pending - 1 }) hdone.1 hdone.2] at tE₀
    have hall : ∀ d < cfg.consumers, rounds cfg.slots (t.stage d) (q % cfg.slots) =
        (s.empty (q % cfg.slots)).phase + 1 := by
      have := (countBelow_eq_iff _ _).1 (show countBelow cfg.consumers (fun d => rounds cfg.slots
        (t.stage d) (q % cfg.slots) == (s.empty (q % cfg.slots)).phase + 1) = cfg.consumers by omega)
      intro d hd; simpa using this d hd
    have tEph : (t.empty (q % cfg.slots)).phase = (s.empty (q % cfg.slots)).phase + 1 := by rw [tE₀]
    refine
      { fault := tf
        issued_le := tI ▸ h.issued_le
        stage_le := fun d hd => ?_
        reading_lt := fun d hd hrd => ?_
        full_expected := tF ▸ h.full_expected
        full_rounds := tF ▸ tI ▸ h.full_rounds
        full_pending := fun x hx => by unfold arrivedP; rw [tF, tI]; exact h.full_pending x hx
        full_tx := fun x hx => by unfold inflight; rw [tF, tI, tP]; exact h.full_tx x hx
        landed := tF ▸ tI ▸ tP ▸ h.landed
        pending_le := tP ▸ h.pending_le
        tag := tT ▸ tI ▸ h.tag
        consumed := shared
        held := fun d hd hrd => ?_
        empty_expected := fun x hx => ?_
        empty_tx := fun x hx => ?_
        empty_rounds := fun x hx d hd => ?_
        empty_pending := fun x hx => ?_
        ahead := fun x hx i hi => by rw [tI]; have := h.ahead x hx i hi; have := hEle x; omega
        full_seen := fseen
        empty_seen := fun x hx hob => ?_ }
    · by_cases hdc : d = c
      · subst hdc; rw [tSc]; omega
      · rw [tS d hdc]; exact h.stage_le d hd
    · rw [tRd] at hrd
      by_cases hdc : d = c
      · rw [ite_eq_left hdc] at hrd; cases hrd
      · rw [ite_eq_right hdc] at hrd; rw [tS d hdc]; exact h.reading_lt d hd hrd
    · rw [tRd] at hrd
      by_cases hdc : d = c
      · rw [ite_eq_left hdc] at hrd; cases hrd
      · rw [ite_eq_right hdc] at hrd; rw [tS d hdc, tF]; exact h.held d hd hrd
    · by_cases hxw : x = q % cfg.slots
      · subst hxw; rw [tE₀]; exact h.empty_expected _ hw
      · rw [tE x hxw]; exact h.empty_expected x hx
    · by_cases hxw : x = q % cfg.slots
      · subst hxw; rw [tE₀]; exact htx0
      · rw [tE x hxw]; exact h.empty_tx x hx
    · by_cases hxw : x = q % cfg.slots
      · subst hxw; rw [tEph]; exact .inl (hall d hd)
      · rw [tE x hxw, tR, ite_eq_right (fun hc' => hxw hc'.2), Nat.add_zero]; exact h.empty_rounds x hx d hd
    · by_cases hxw : x = q % cfg.slots
      · subst hxw
        have e : arrivedC cfg t (q % cfg.slots) = 0 := by
          unfold arrivedC; rw [tEph]
          exact (countBelow_zero_iff _ _).2 fun d hd => by simp [hall d hd]
        rw [e, tE₀]; exact h.empty_expected _ hw
      · exact common x hx hxw
    · by_cases hxw : x = q % cfg.slots
      · subst hxw
        refine ⟨by rw [tEph]; exact Nat.succ_pos _, fun i hi => ?_⟩
        rw [tI, tEph]; exact h.ahead _ hw i hi
      · rw [tE x hxw] at hob ⊢; rw [tI]; exact h.empty_seen x hx hob
  · -- The phase continues with one more consumer arrived.
    rw [MBarrier.settle_of_not (by simp only; exact hdone)] at tE₀
    have tEph : ∀ x, (t.empty x).phase = (s.empty x).phase := by
      intro x; by_cases hxw : x = q % cfg.slots
      · subst hxw; rw [tE₀]
      · rw [tE x hxw]
    refine
      { fault := tf
        issued_le := tI ▸ h.issued_le
        stage_le := fun d hd => ?_
        reading_lt := fun d hd hrd => ?_
        full_expected := tF ▸ h.full_expected
        full_rounds := tF ▸ tI ▸ h.full_rounds
        full_pending := fun x hx => by unfold arrivedP; rw [tF, tI]; exact h.full_pending x hx
        full_tx := fun x hx => by unfold inflight; rw [tF, tI, tP]; exact h.full_tx x hx
        landed := tF ▸ tI ▸ tP ▸ h.landed
        pending_le := tP ▸ h.pending_le
        tag := tT ▸ tI ▸ h.tag
        consumed := shared
        held := fun d hd hrd => ?_
        empty_expected := fun x hx => ?_
        empty_tx := fun x hx => ?_
        empty_rounds := fun x hx d hd => ?_
        empty_pending := fun x hx => ?_
        ahead := fun x hx i hi => by rw [tI, tEph]; exact h.ahead x hx i hi
        full_seen := fseen
        empty_seen := fun x hx hob => ?_ }
    · by_cases hdc : d = c
      · subst hdc; rw [tSc]; omega
      · rw [tS d hdc]; exact h.stage_le d hd
    · rw [tRd] at hrd
      by_cases hdc : d = c
      · rw [ite_eq_left hdc] at hrd; cases hrd
      · rw [ite_eq_right hdc] at hrd; rw [tS d hdc]; exact h.reading_lt d hd hrd
    · rw [tRd] at hrd
      by_cases hdc : d = c
      · rw [ite_eq_left hdc] at hrd; cases hrd
      · rw [ite_eq_right hdc] at hrd; rw [tS d hdc, tF]; exact h.held d hd hrd
    · by_cases hxw : x = q % cfg.slots
      · subst hxw; rw [tE₀]; exact h.empty_expected _ hw
      · rw [tE x hxw]; exact h.empty_expected x hx
    · by_cases hxw : x = q % cfg.slots
      · subst hxw; rw [tE₀]; exact htx0
      · rw [tE x hxw]; exact h.empty_tx x hx
    · rw [tEph, tR]
      by_cases hc' : d = c ∧ x = q % cfg.slots
      · obtain ⟨rfl, rfl⟩ := hc'; rw [ite_eq_left ⟨rfl, rfl⟩, hRc, hE]; exact .inr rfl
      · rw [ite_eq_right hc', Nat.add_zero]; exact h.empty_rounds x hx d hd
    · by_cases hxw : x = q % cfg.slots
      · subst hxw
        have e : arrivedC cfg t (q % cfg.slots) = arrivedC cfg s (q % cfg.slots) + 1 := by
          unfold arrivedC; rw [tEph]; exact hcount
        rw [e, tE₀]; show (s.empty (q % cfg.slots)).pending - 1 + _ = _; omega
      · exact common x hx hxw
    · by_cases hxw : x = q % cfg.slots
      · subst hxw; rw [tE₀] at hob; rw [hobs] at hob; cases hob
      · rw [tE x hxw] at hob ⊢; rw [tI]; exact h.empty_seen x hx hob

theorem inv_step {a : Action} (hs : step cfg s a = some t) : Inv cfg t := by
  cases a with
  | issue i => exact inv_issue hv h hs
  | land i w k => exact inv_land hv h hs
  | acquire c => exact inv_acquire hv h hs
  | release c => exact inv_release hv h hs

end

/-- **The ring is safe.** For every valid configuration, no reachable state faults: no mbarrier
is over-arrived, re-armed before its last phase was observed or waited on with a stale parity; no
consumer acquires a slot before every byte of its stage has landed; no producer overwrites a slot
a consumer holds. -/
theorem ring_safe {cfg : Config} (hv : cfg.Valid) {s : State} (hr : (system cfg).Reachable s) :
    s.fault = none :=
  (NV.Formal.System.Reachable.induct (system cfg) (inv_init hv)
    (fun _ _ _ _ hs hstep => inv_step hv hs hstep) hr).fault


end Examples.Ring
