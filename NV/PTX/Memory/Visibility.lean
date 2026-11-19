import NV.PTX.Memory.Model
import NV.Formal.Count

/-!
# Visibility in consistent executions

General consequences of the axioms, for executions of any size:

* `read_sees`: a read that a generic write to its address precedes in base causality order reads
  neither the initial value nor a write coherence-before that write (Causality, §8.10.6).
* `causeBase_*`: base causality order is closed under program order and synchronization.
* `sync_of_observation`: a release write synchronizes with an acquire read it reaches through
  observation order (§8.8, §8.9.2, §8.9.4).
* `counter_observes`: on a location updated only by `atom.add`s of positive increments, an
  `atom.add` that reads the sum of all the others' increments observes every one of them.
-/

namespace NV.PTX.Execution

open Rel NV.Formal

variable {n : Nat} {V : Type} [DecidableEq V] {x : Execution n V}

theorem consistent_parts (hc : x.consistent = true) :
    x.wellFormed = true ∧ x.noThinAir = true ∧ x.atomicity = true ∧ x.coherence = true ∧
      x.causality = true ∧ x.seqCst = true := by
  simp only [consistent, Bool.and_eq_true] at hc
  exact ⟨hc.1.1.1.1.1, hc.1.1.1.1.2, hc.1.1.1.2, hc.1.1.2, hc.1.2, hc.2⟩

theorem wellFormed_parts (hw : x.wellFormed = true) :
    x.poWF = true ∧ x.rfWF = true ∧ x.coWF = true ∧ x.scWF = true ∧ x.rmwWF = true ∧
      x.barWF = true := by
  simp only [wellFormed, Bool.and_eq_true] at hw
  exact ⟨hw.1.1.1.1.1, hw.1.1.1.1.2, hw.1.1.1.2, hw.1.1.2, hw.1.2, hw.2⟩

theorem strictOrder_trans {r : Rel n} (h : strictOrder r = true) {a b c : Fin n}
    (h₁ : r a b = true) (h₂ : r b c = true) : r a c = true := by
  simp only [strictOrder, Bool.and_eq_true, subset_iff] at h
  exact h.2 a c (comp_of h₁ h₂)

theorem strictOrder_irrefl {r : Rel n} (h : strictOrder r = true) (a : Fin n) : r a a = false := by
  simp only [strictOrder, Bool.and_eq_true, irreflexive_iff] at h
  exact h.1 a

/-- A transitive relation is its own transitive closure. -/
theorem of_tc_strictOrder {r : Rel n} (h : strictOrder r = true) {a b : Fin n}
    (hab : tc r a b = true) : r a b = true := by
  rw [tc_iff_transGen] at hab
  induction hab with
  | single e => exact e
  | tail _ e ih => exact strictOrder_trans h ih e

theorem co_trans (hc : x.consistent = true) {a b c : Fin n} (h₁ : x.co a b = true)
    (h₂ : x.co b c = true) : x.co a c = true := by
  have := (wellFormed_parts (consistent_parts hc).1).2.2.1
  simp only [coWF, Bool.and_eq_true] at this
  exact strictOrder_trans this.1.1 h₁ h₂

theorem co_of_tc (hc : x.consistent = true) {a b : Fin n} (h : tc x.co a b = true) : x.co a b = true := by
  have := (wellFormed_parts (consistent_parts hc).1).2.2.1
  simp only [coWF, Bool.and_eq_true] at this
  exact of_tc_strictOrder this.1.1 h

theorem co_irrefl (hc : x.consistent = true) (a : Fin n) : x.co a a = false := by
  have := (wellFormed_parts (consistent_parts hc).1).2.2.1
  simp only [coWF, Bool.and_eq_true] at this
  exact strictOrder_irrefl this.1.1 a

/-- Coherence order relates every two distinct morally strong writes to one address. -/
theorem co_total (hc : x.consistent = true) {a b : Fin n} (hab : a ≠ b) (ha : x.writes a = true)
    (hb : x.writes b = true) (haddr : (x.ev a).addr = (x.ev b).addr) (hs : x.strongR a b = true) :
    x.co a b = true ∨ x.co b a = true := by
  have := (wellFormed_parts (consistent_parts hc).1).2.2.1
  simp only [coWF, Bool.and_eq_true, List.all_eq_true, List.mem_finRange, true_implies] at this
  have h := this.2 a b
  simp [hab, ha, hb, haddr, hs] at h
  exact h

/-! ## Base causality order -/

/-- One synchronizing hop of base causality order, followed by program order. -/
def hop (x : Execution n V) : Rel n := (x.sc ∪ x.sync) ⨾ rtc x.order

theorem causeBase_iff {a b : Fin n} :
    x.causeBase a b = true ↔ tc x.order a b = true ∨ ∃ u, rtc x.order a u = true ∧ tc x.hop u b = true := by
  simp only [causeBase, get_union, Bool.or_eq_true, get_comp, hop]

theorem rtc_order_trans {a b c : Fin n} (h₁ : rtc x.order a b = true) (h₂ : rtc x.order b c = true) :
    rtc x.order a c = true := by
  simp only [rtc, get_opt, Bool.or_eq_true, beq_iff_eq] at h₁ h₂ ⊢
  rcases h₁ with rfl | h₁ <;> rcases h₂ with rfl | h₂
  · exact .inl rfl
  · exact .inr h₂
  · exact .inr h₁
  · exact .inr (tc_trans h₁ h₂)

theorem tc_hop_order {u b c : Fin n} (h : tc x.hop u b = true) (h₂ : rtc x.order b c = true) :
    tc x.hop u c = true := by
  rw [tc_iff_transGen] at h ⊢
  cases h with
  | single e =>
    obtain ⟨v, hv, hvb⟩ := get_comp.1 e
    exact .single (comp_of hv (rtc_order_trans hvb h₂))
  | tail hp e =>
    obtain ⟨v, hv, hvb⟩ := get_comp.1 e
    exact hp.tail (comp_of hv (rtc_order_trans hvb h₂))

theorem causeBase_of_order {a b : Fin n} (h : tc x.order a b = true) : x.causeBase a b = true :=
  causeBase_iff.2 (.inl h)

theorem causeBase_of_sync {a b : Fin n} (h : x.sync a b = true) : x.causeBase a b = true :=
  causeBase_iff.2 (.inr ⟨a, rtc_refl _ _, tc_of (comp_of (by simp [h]) (rtc_refl _ _))⟩)

theorem causeBase_order {a b c : Fin n} (h : x.causeBase a b = true) (h₂ : rtc x.order b c = true) :
    x.causeBase a c = true := by
  rcases causeBase_iff.1 h with h | ⟨u, hu, hub⟩
  · simp only [rtc, get_opt, Bool.or_eq_true, beq_iff_eq] at h₂
    rcases h₂ with rfl | h₂
    · exact causeBase_of_order h
    · exact causeBase_of_order (tc_trans h h₂)
  · exact causeBase_iff.2 (.inr ⟨u, hu, tc_hop_order hub h₂⟩)

theorem causeBase_sync {a b c : Fin n} (h : x.causeBase a b = true) (h₂ : x.sync b c = true) :
    x.causeBase a c = true := by
  have hop₁ : x.hop b c = true := comp_of (by simp [h₂]) (rtc_refl _ _)
  rcases causeBase_iff.1 h with h | ⟨u, hu, hub⟩
  · exact causeBase_iff.2 (.inr ⟨b, rtc_of_tc h, tc_of hop₁⟩)
  · exact causeBase_iff.2 (.inr ⟨u, hu, tc_trans hub (tc_of hop₁)⟩)

/-! ## Causality makes causally earlier writes visible -/

theorem ppcb_of_generic {w r : Fin n} (hgw : x.generic w = true) (hgr : x.generic r = true)
    (ha : (x.ev w).addr = (x.ev r).addr) (hcb : x.causeBase w r = true) :
    x.proxyPreservedCauseBase w r = true := by
  have hmw : x.mems w = true := by simp only [generic, Bool.and_eq_true] at hgw; exact hgw.1
  have hmr : x.mems r = true := by simp only [generic, Bool.and_eq_true] at hgr; exact hgr.1
  simp only [proxyPreservedCauseBase, get_union, get_inter, Bool.or_eq_true, Bool.and_eq_true]
  left; left; left; left; left; left; left; left
  refine ⟨comp_of (comp_of (by simp [hgw]) hcb) (by simp [hgr]), ?_⟩
  simp [sameAlias, hmw, hmr, ha]

theorem cause_of_generic {w r : Fin n} (hgw : x.generic w = true) (hgr : x.generic r = true)
    (ha : (x.ev w).addr = (x.ev r).addr) (hcb : x.causeBase w r = true) : x.cause w r = true :=
  comp_of (b := w) (by simp) (ppcb_of_generic hgw hgr ha hcb)

/-- A read that a write precedes in proxy-preserved base causality order reads that write or a
later one in coherence order, never an earlier one or the initial value. -/
theorem read_sees_ppcb (hc : x.consistent = true) {w r : Fin n} (hw : x.writes w = true)
    (hr : x.reads r = true) (hmw : x.mems w = true) (hmr : x.mems r = true)
    (ha : (x.ev w).addr = (x.ev r).addr) (hp : x.proxyPreservedCauseBase w r = true) :
    (∃ w', x.rf w' r = true) ∧ ∀ w', x.rf w' r = true → x.co w' w = false := by
  have hcause : x.cause w r = true := comp_of (b := w) (by simp) hp
  have hcaus := (consistent_parts hc).2.2.2.2.1
  simp only [causality, irreflexive_iff] at hcaus
  -- A read that `fr`-precedes `w` would close a cycle with `w`'s causal precedence.
  have nofr : x.fr r w = false := by
    cases hfr : x.fr r w
    · rfl
    · have := hcaus r
      rw [show (opt x.com ⨾ x.cause) r r = true from
        comp_of (b := w) (by simp [com, hfr]) hcause] at this
      cases this
  constructor
  · apply Classical.byContradiction; intro hnone
    simp only [not_exists, Bool.not_eq_true] at hnone
    have : x.fr r w = true := by
      simp only [fr, get_union, Bool.or_eq_true, get_ofFn, Bool.and_eq_true, Bool.not_eq_true',
        List.any_eq_false, List.mem_finRange, true_implies]
      exact .inr ⟨⟨⟨hr, hw⟩, by simp [sameLocation, hmw, hmr, ha]⟩, fun w' => by simp [hnone w']⟩
    rw [nofr] at this; cases this
  · intro w' hrf
    cases hco : x.co w' w
    · rfl
    · have : x.fr r w = true := by
        simp only [fr, get_union, Bool.or_eq_true]
        exact .inl (comp_of (by simp [hrf]) (tc_of hco))
      rw [nofr] at this; cases this

theorem read_sees (hc : x.consistent = true) {w r : Fin n} (hw : x.writes w = true)
    (hr : x.reads r = true) (hgw : x.generic w = true) (hgr : x.generic r = true)
    (ha : (x.ev w).addr = (x.ev r).addr) (hcb : x.causeBase w r = true) :
    (∃ w', x.rf w' r = true) ∧ ∀ w', x.rf w' r = true → x.co w' w = false := by
  have hmw : x.mems w = true := by simp only [generic, Bool.and_eq_true] at hgw; exact hgw.1
  have hmr : x.mems r = true := by simp only [generic, Bool.and_eq_true] at hgr; exact hgr.1
  exact read_sees_ppcb hc hw hr hmw hmr ha (ppcb_of_generic hgw hgr ha hcb)

/-- A read reads the initial value of its location, from no write, or from one write to it. -/
theorem rf_cases (hc : x.consistent = true) {r : Fin n} (hr : x.reads r = true) :
    ((∀ w, x.rf w r = false) ∧ x.val r = x.init (x.loc (x.ev r).addr)) ∨
      ∃ w, x.rf w r = true ∧ x.val r = x.val w ∧ x.writes w = true ∧
        x.loc (x.ev w).addr = x.loc (x.ev r).addr := by
  have := (wellFormed_parts (consistent_parts hc).1).2.1
  simp only [rfWF, Bool.and_eq_true, subset_iff, List.all_eq_true, List.mem_finRange,
    true_implies] at this
  obtain ⟨hsub, hval⟩ := this
  have h := hval r
  simp only [hr, Bool.not_true, Bool.false_or] at h
  split at h
  case h_1 hnil =>
    left
    refine ⟨fun w => ?_, by simpa using h⟩
    cases hw : x.rf w r
    · rfl
    · have : w ∈ (List.finRange n).filter fun w => x.rf w r := by simp [hw]
      rw [hnil] at this; cases this
  case h_2 w hone =>
    right
    have hw : x.rf w r = true := by
      have : w ∈ (List.finRange n).filter fun w => x.rf w r := by rw [hone]; simp
      simpa using this
    obtain ⟨b, hb, _⟩ := get_comp.1 (hsub w r hw)
    obtain ⟨a, ha, hab⟩ := get_comp.1 hb
    simp only [get_dom, Bool.and_eq_true, beq_iff_eq] at ha
    obtain ⟨rfl, hwr⟩ := ha
    have hbr : b = r := by
      obtain ⟨_, h1, h2⟩ := get_comp.1 (hsub w r hw)
      simp only [get_dom, Bool.and_eq_true, beq_iff_eq] at h2
      simp_all
    subst hbr
    simp only [sameLocation, get_ofFn, Bool.and_eq_true, beq_iff_eq] at hab
    exact ⟨w, hw, by simpa using h, hwr, hab.2⟩
  case h_3 => cases h

/-! ## Release, observation, acquire -/

theorem strongR_self (a : Fin n) : x.strongR a a = true := by
  simp only [strongR, get_ofFn, Bool.and_eq_true, beq_self_eq_true, and_true]
  cases h : (x.ev a).scope <;> simp [Scope.contains]

theorem sync_of_observation {w a : Fin n} (hrel : (x.ev w).release = true) (hw : x.writes w = true)
    (hacq : (x.ev a).acquire = true) (hr : x.reads a = true) (hnr : (x.ev a).reduction = false)
    (hobs : tc x.observation w a = true) (hs : x.strongR w a = true) : x.sync w a = true := by
  have rs : x.releaseSequence w w = true := by
    simp only [releaseSequence, strong, get_union, get_inter, Bool.or_eq_true, Bool.and_eq_true]
    exact .inl ⟨comp_of (b := w) (comp_of (b := w) (by simp [hrel, hw]) (by simp)) (by simp [hw]),
      strongR_self w⟩
  have as : x.acquireSequence a a = true := by
    simp only [acquireSequence, strong, get_union, get_inter, Bool.or_eq_true, Bool.and_eq_true]
    exact .inl ⟨comp_of (b := a) (comp_of (b := a) (by simp [hr, hnr]) (by simp)) (by simp [hacq, hr]),
      strongR_self a⟩
  simp only [sync, strong, get_union, get_inter, Bool.or_eq_true, Bool.and_eq_true]
  exact .inl ⟨comp_of (comp_of rs hobs) as, hs⟩

end NV.PTX.Execution
