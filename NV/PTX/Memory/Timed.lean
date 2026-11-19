import NV.PTX.Memory.DRF

/-!
# Executions performed in a total order

The operational request machine (`NV.Memory.Flat`) performs every event at one instant of a total
order `t`: a request reads the flat memory and writes it when it is delivered, a fence or barrier
acts when it executes. `Timed` lists what such a run guarantees of the order, all of them local
facts about the machine's rules:

* each read reads the latest write to its location performed before it (or the initial value
  when there is none), and coherence order is the order writes were performed;
* a read-modify-write performs its read and its write with no write to the location between;
* a thread's accesses to one location perform in program order;
* a release write, release fence or barrier arrival waits until everything before it in program
  order has performed (`drains`), and nothing after an acquire read, acquire fence or barrier wait
  performs before it (`blocks`);
* barrier arrivals perform before the waits they release, `fence.sc` order is the order the fences
  executed, and dependencies point forward.

`Timed.consistent`: an execution with such an order, well formed in the program-given relations,
satisfies every axiom of §8.10. The proof shows that causality order runs forward in `t`.
-/

namespace NV.PTX

/-- Events after which program-order-earlier events have performed. -/
def Event.drains (e : Event) : Bool :=
  (e.isWrite && e.release) || e.isReleaseFence ||
    (match e.kind with | .barArrive _ | .clusterArrive => true | _ => false)

/-- Events before which program-order-later events do not perform. -/
def Event.blocks (e : Event) : Bool :=
  (e.isRead && e.acquire) || e.isAcquireFence ||
    (match e.kind with | .barWait _ | .clusterWait => true | _ => false)

namespace Execution

open Rel

variable {n : Nat} {V : Type} [DecidableEq V]

structure Timed (x : Execution n V) where
  t : Fin n → Nat
  inj : ∀ a b, t a = t b → a = b
  rf_lt : ∀ w r, x.rf w r = true → t w < t r
  rf_latest : ∀ w r w', x.rf w r = true → x.writes w' = true → x.sameLocation w' r = true →
    t w < t w' → t r < t w'
  init_first : ∀ r w', x.reads r = true → (∀ w, x.rf w r = false) → x.writes w' = true →
    x.sameLocation w' r = true → t r < t w'
  co_iff : ∀ a b, x.co a b = true ↔
    x.writes a = true ∧ x.writes b = true ∧ x.sameLocation a b = true ∧ t a < t b
  sc_iff : ∀ a b, x.sc a b = true ↔
    (x.ev a).isFenceSC = true ∧ (x.ev b).isFenceSC = true ∧ t a < t b
  rmw_lt : ∀ r w, x.rmw r w = true → t r < t w
  rmw_adjacent : ∀ r w w', x.rmw r w = true → x.writes w' = true → x.sameLocation w' w = true →
    w' ≠ w → t r < t w' → t w < t w'
  fifo : ∀ a b, tc x.order a b = true → x.sameLocation a b = true → t a < t b
  drains : ∀ a u, tc x.order a u = true → (x.ev u).drains = true → t a < t u
  blocks : ∀ v b, tc x.order v b = true → (x.ev v).blocks = true → t v < t b
  bar_lt : ∀ a b, x.bar a b = true → t a < t b
  dep_lt : ∀ a b, x.dep a b = true → t a < t b

omit [DecidableEq V] in
theorem sameLocation_of_sameAlias {x : Execution n V} {a b : Fin n} (h : x.sameAlias a b = true) :
    x.sameLocation a b = true := by
  simp only [sameAlias, sameLocation, get_ofFn, Bool.and_eq_true, beq_iff_eq] at h ⊢
  exact ⟨h.1, by rw [h.2]⟩

theorem rf_parts' {x : Execution n V} (hwf : x.rfWF = true) {w r : Fin n} (h : x.rf w r = true) :
    x.writes w = true ∧ x.reads r = true ∧ x.sameLocation w r = true := by
  simp only [rfWF, Bool.and_eq_true, subset_iff] at hwf
  obtain ⟨b, hb, hbr⟩ := get_comp.1 (hwf.1 w r h)
  obtain ⟨c, hc, hcb⟩ := get_comp.1 hb
  simp only [get_dom, Bool.and_eq_true, beq_iff_eq] at hc hbr
  obtain ⟨rfl, hw⟩ := hc
  obtain ⟨rfl, hr⟩ := hbr
  exact ⟨hw, hr, hcb⟩

theorem ppcb_sameLocation {x : Execution n V} {a b : Fin n} (h : x.proxyPreservedCauseBase a b = true) :
    x.sameLocation a b = true := by
  simp only [proxyPreservedCauseBase, get_union, get_inter, Bool.or_eq_true, Bool.and_eq_true] at h
  rcases h with ((((((((h | h) | h) | h) | h) | h) | h) | h) | h) <;>
    first | exact sameLocation_of_sameAlias h.2 | exact h.2

namespace Timed

variable {x : Execution n V} (T : x.Timed)
include T

/-- A relation whose edges run forward in `t` is acyclic. -/
theorem acyclic_of_lt {r : Rel n} (h : ∀ a b, r a b = true → T.t a < T.t b) : acyclic r = true := by
  have key : ∀ a b, tc r a b = true → T.t a < T.t b := fun a b hab => by
    rw [tc_iff_transGen] at hab
    induction hab with
    | single e => exact h _ _ e
    | tail _ e ih => exact Nat.lt_trans ih (h _ _ e)
  simp only [acyclic, irreflexive_iff]
  intro a
  cases hc : tc r a a
  · rfl
  · exact absurd (key a a hc) (Nat.lt_irrefl _)

theorem co_lt {a b : Fin n} (h : x.co a b = true) : T.t a < T.t b := ((T.co_iff a b).1 h).2.2.2

theorem tc_co_lt {a b : Fin n} (h : tc x.co a b = true) : T.t a < T.t b := by
  rw [tc_iff_transGen] at h
  induction h with
  | single e => exact T.co_lt e
  | tail _ e ih => exact Nat.lt_trans ih (T.co_lt e)

theorem tc_co_sameLocation {a b : Fin n} (h : tc x.co a b = true) : x.sameLocation a b = true ∧
    x.writes b = true := by
  rw [tc_iff_transGen] at h
  induction h with
  | single e => have := (T.co_iff _ _).1 e; exact ⟨this.2.2.1, this.2.1⟩
  | tail _ e ih =>
    have := (T.co_iff _ _).1 e
    exact ⟨sameLocation_trans ih.1 this.2.2.1, this.2.1⟩

theorem fr_lt (hwf : x.rfWF = true) {r w : Fin n} (h : x.fr r w = true) : T.t r < T.t w := by
  simp only [fr, get_union, Bool.or_eq_true] at h
  rcases h with h | h
  · obtain ⟨w₀, h₁, h₂⟩ := get_comp.1 h
    rw [get_inv] at h₁
    obtain ⟨_, _, hl₁⟩ := rf_parts' hwf h₁
    obtain ⟨hl₂, hw⟩ := T.tc_co_sameLocation h₂
    exact T.rf_latest w₀ r w h₁ hw (sameLocation_trans (sameLocation_symm hl₂) hl₁) (T.tc_co_lt h₂)
  · simp only [get_ofFn, Bool.and_eq_true, Bool.not_eq_true', List.any_eq_false,
      List.mem_finRange, true_implies] at h
    exact T.init_first r w h.1.1.1 (fun w' => by simpa using h.2 w') h.1.1.2
      (sameLocation_symm h.1.2)

theorem com_lt (hwf : x.rfWF = true) {a b : Fin n} (h : x.com a b = true) : T.t a < T.t b := by
  simp only [com, get_union, Bool.or_eq_true] at h
  rcases h with (h | h) | h
  · exact T.rf_lt a b h
  · exact T.tc_co_lt h
  · exact T.fr_lt hwf h

theorem observation_lt {a b : Fin n} (h : tc x.observation a b = true) : T.t a < T.t b := by
  rw [tc_iff_transGen] at h
  have one : ∀ a b, x.observation a b = true → T.t a < T.t b := fun a b h => by
    simp only [observation, strong, get_union, get_inter, Bool.or_eq_true, Bool.and_eq_true] at h
    rcases h with ⟨h, _⟩ | h
    · exact T.rf_lt a b h
    · exact T.rmw_lt a b h
  induction h with
  | single e => exact one _ _ e
  | tail _ e ih => exact Nat.lt_trans ih (one _ _ e)

theorem le_of_rtc_order {a b : Fin n} (h : rtc x.order a b = true) (hb : (x.ev b).drains = true) :
    T.t a ≤ T.t b := by
  simp only [rtc, get_opt, Bool.or_eq_true, beq_iff_eq] at h
  rcases h with rfl | h
  · exact Nat.le_refl _
  · exact Nat.le_of_lt (T.drains a b h hb)

theorem le_of_rtc_order' {a b : Fin n} (h : rtc x.order a b = true) (ha : (x.ev a).blocks = true) :
    T.t a ≤ T.t b := by
  simp only [rtc, get_opt, Bool.or_eq_true, beq_iff_eq] at h
  rcases h with rfl | h
  · exact Nat.le_refl _
  · exact Nat.le_of_lt (T.blocks a b h ha)

theorem le_of_opt_poLoc {a b : Fin n} (h : opt x.poLoc a b = true) : T.t a ≤ T.t b := by
  simp only [get_opt, Bool.or_eq_true, beq_iff_eq, poLoc, get_inter, Bool.and_eq_true] at h
  rcases h with rfl | ⟨h, hl⟩
  · exact Nat.le_refl _
  · have hl' : x.sameLocation a b = true := by
      simp only [sameAlias, sameLocation, get_ofFn, Bool.and_eq_true, beq_iff_eq] at hl ⊢
      exact ⟨hl.1, by rw [hl.2]⟩
    exact Nat.le_of_lt (T.fifo a b h hl')

theorem releaseSequence_le {u w : Fin n} (h : x.releaseSequence u w = true) :
    (x.ev u).drains = true ∧ T.t u ≤ T.t w := by
  simp only [releaseSequence, strong, get_union, get_inter, Bool.or_eq_true, Bool.and_eq_true,
    dom_comp_iff, comp_dom_iff] at h
  rcases h with ⟨⟨⟨⟨hrel, hwu⟩, hpo⟩, _⟩, _⟩ | ⟨⟨⟨hf, hpo⟩, _⟩, _⟩
  · refine ⟨?_, T.le_of_opt_poLoc hpo⟩
    simp only [writes] at hwu
    simp [Event.drains, hrel, hwu]
  · have hd : (x.ev u).drains = true := by simp [Event.drains, hf]
    have hb : (x.ev u).blocks = true := by
      simp only [Event.isReleaseFence] at hf
      simp [Event.blocks, Event.isAcquireFence, Event.isReleaseFence, hf]
    exact ⟨hd, Nat.le_of_lt (T.blocks u w hpo hb)⟩

theorem acquireSequence_le {z v : Fin n} (h : x.acquireSequence z v = true) :
    (x.ev v).blocks = true ∧ T.t z ≤ T.t v := by
  simp only [acquireSequence, strong, get_union, get_inter, Bool.or_eq_true, Bool.and_eq_true,
    dom_comp_iff, comp_dom_iff] at h
  rcases h with ⟨⟨⟨_, hpo⟩, hacq, hrv⟩, _⟩ | ⟨⟨⟨_, hpo⟩, hf⟩, _⟩
  · refine ⟨?_, T.le_of_opt_poLoc hpo⟩
    simp only [reads] at hrv
    simp [Event.blocks, hacq, hrv]
  · have hb : (x.ev v).blocks = true := by simp [Event.blocks, hf]
    have hd : (x.ev v).drains = true := by
      simp only [Event.isAcquireFence, Event.isReleaseFence] at hf
      simp [Event.drains, Event.isReleaseFence, hf]
    exact ⟨hb, Nat.le_of_lt (T.drains z v hpo hd)⟩

/-- One synchronizing step: it starts at an event that drains, ends at one that blocks, and runs
forward in `t`. -/
theorem hop_parts (hwf : x.wellFormed = true) {u v : Fin n} (h : (x.sc ∪ x.sync) u v = true) :
    (x.ev u).drains = true ∧ (x.ev v).blocks = true ∧ T.t u < T.t v := by
  simp only [get_union, Bool.or_eq_true] at h
  rcases h with h | h
  · have := (T.sc_iff u v).1 h
    simp only [Event.isFenceSC, beq_iff_eq] at this
    simp [Event.drains, Event.blocks, Event.isReleaseFence, Event.isAcquireFence, this.1, this.2.1,
      this.2.2]
  · simp only [sync, strong, get_union, get_inter, Bool.or_eq_true, Bool.and_eq_true] at h
    rcases h with ⟨h, _⟩ | h
    · obtain ⟨z, h, hzv⟩ := get_comp.1 h
      obtain ⟨w, huw, hwz⟩ := get_comp.1 h
      obtain ⟨hd, hle₁⟩ := T.releaseSequence_le huw
      obtain ⟨hb, hle₂⟩ := T.acquireSequence_le hzv
      exact ⟨hd, hb, Nat.lt_of_le_of_lt hle₁ (Nat.lt_of_lt_of_le (T.observation_lt hwz) hle₂)⟩
    · have := (wellFormed_parts hwf).2.2.2.2.2
      simp only [barWF, List.all_eq_true, List.mem_finRange, true_implies] at this
      have := this u v
      simp only [h, Bool.not_true, Bool.false_or] at this
      refine ⟨?_, ?_, T.bar_lt u v h⟩ <;> split at this <;>
        first | cases this | simp [Event.drains, Event.blocks, *]

/-- Base causality order is program order, or runs forward in `t`. -/
theorem causeBase_lt (hwf : x.wellFormed = true) {a b : Fin n} (h : x.causeBase a b = true) :
    tc x.order a b = true ∨ T.t a < T.t b := by
  rcases causeBase_iff.1 h with h | ⟨u, hu, hub⟩
  · exact .inl h
  · right
    -- Each hop leaves a draining event, reaches a blocking one, and runs forward.
    have key : ∀ b, tc x.hop u b = true → (x.ev u).drains = true ∧ T.t u < T.t b := by
      intro b hub
      rw [tc_iff_transGen] at hub
      induction hub with
      | single e =>
        obtain ⟨v, h₁, h₂⟩ := get_comp.1 e
        obtain ⟨hd, hb, hlt⟩ := T.hop_parts hwf h₁
        exact ⟨hd, Nat.lt_of_lt_of_le hlt (T.le_of_rtc_order' h₂ hb)⟩
      | tail _ e ih =>
        obtain ⟨v, h₁, h₂⟩ := get_comp.1 e
        obtain ⟨_, hb, hlt⟩ := T.hop_parts hwf h₁
        exact ⟨ih.1, Nat.lt_trans ih.2 (Nat.lt_of_lt_of_le hlt (T.le_of_rtc_order' h₂ hb))⟩
    obtain ⟨hd, hlt⟩ := key b hub
    exact Nat.lt_of_le_of_lt (T.le_of_rtc_order hu hd) hlt

theorem ppcb_lt (hwf : x.wellFormed = true) {a b : Fin n}
    (h : x.proxyPreservedCauseBase a b = true) : T.t a < T.t b := by
  rcases T.causeBase_lt hwf (causeBase_of_ppcb h) with h' | h'
  · exact T.fifo a b h' (ppcb_sameLocation h)
  · exact h'

theorem cause_lt (hwf : x.wellFormed = true) {a b : Fin n} (h : x.cause a b = true) :
    T.t a < T.t b := by
  obtain ⟨z, h₁, h₂⟩ := get_comp.1 h
  refine Nat.lt_of_le_of_lt ?_ (T.ppcb_lt hwf h₂)
  simp only [get_opt, Bool.or_eq_true, beq_iff_eq] at h₁
  rcases h₁ with rfl | h₁
  · exact Nat.le_refl _
  · exact Nat.le_of_lt (T.observation_lt (tc_of h₁))

theorem coWF : x.coWF = true := by
  simp only [Execution.coWF, strictOrder, Bool.and_eq_true, irreflexive_iff, subset_iff,
    List.all_eq_true, List.mem_finRange, true_implies]
  refine ⟨⟨⟨fun a => ?_, fun a c h => ?_⟩, fun a b h => ?_⟩, fun a b => ?_⟩
  · cases h : x.co a a
    · rfl
    · exact absurd (T.co_lt h) (Nat.lt_irrefl _)
  · obtain ⟨b, h₁, h₂⟩ := get_comp.1 h
    obtain ⟨hwa, _, hl₁, lt₁⟩ := (T.co_iff a b).1 h₁
    obtain ⟨_, hwc, hl₂, lt₂⟩ := (T.co_iff b c).1 h₂
    exact (T.co_iff a c).2 ⟨hwa, hwc, sameLocation_trans hl₁ hl₂, Nat.lt_trans lt₁ lt₂⟩
  · obtain ⟨hwa, hwb, hl, _⟩ := (T.co_iff a b).1 h
    exact comp_dom_iff.2 ⟨dom_comp_iff.2 ⟨hwa, hl⟩, hwb⟩
  · by_cases hab : a = b
    · simp [hab]
    have hlt := Nat.lt_or_gt_of_ne (fun e => hab (T.inj a b e))
    cases hwa : x.writes a <;> cases hwb : x.writes b <;>
      try simp only [Bool.false_and, Bool.and_false, Bool.not_false, Bool.or_true, Bool.true_or]
    by_cases had : (x.ev a).addr = (x.ev b).addr
    · have hl : x.sameLocation a b = true := by
        simp [sameLocation, mems_of_writes hwa, mems_of_writes hwb, had]
      rcases hlt with h | h
      · simp [(T.co_iff a b).2 ⟨hwa, hwb, hl, h⟩]
      · simp [(T.co_iff b a).2 ⟨hwb, hwa, sameLocation_symm hl, h⟩]
    · simp [had]

theorem scWF : x.scWF = true := by
  simp only [Execution.scWF, strictOrder, Bool.and_eq_true, irreflexive_iff, subset_iff,
    List.all_eq_true, List.mem_finRange, true_implies]
  refine ⟨⟨⟨fun a => ?_, fun a c h => ?_⟩, fun a b h => ?_⟩, fun a b => ?_⟩
  · cases h : x.sc a a
    · rfl
    · exact absurd ((T.sc_iff a a).1 h).2.2 (Nat.lt_irrefl _)
  · obtain ⟨b, h₁, h₂⟩ := get_comp.1 h
    obtain ⟨hfa, _, lt₁⟩ := (T.sc_iff a b).1 h₁
    obtain ⟨_, hfc, lt₂⟩ := (T.sc_iff b c).1 h₂
    exact (T.sc_iff a c).2 ⟨hfa, hfc, Nat.lt_trans lt₁ lt₂⟩
  · obtain ⟨hfa, hfb, _⟩ := (T.sc_iff a b).1 h
    simp [hfa, hfb]
  · by_cases hab : a = b
    · simp [hab]
    have hlt := Nat.lt_or_gt_of_ne (fun e => hab (T.inj a b e))
    cases hfa : (x.ev a).isFenceSC <;> cases hfb : (x.ev b).isFenceSC <;>
      try simp only [Bool.false_and, Bool.and_false, Bool.not_false, Bool.or_true, Bool.true_or]
    rcases hlt with h | h
    · simp [(T.sc_iff a b).2 ⟨hfa, hfb, h⟩]
    · simp [(T.sc_iff b a).2 ⟨hfb, hfa, h⟩]

/-- **An execution performed in a total order is consistent.** -/
theorem consistent (hpo : x.poWF = true) (hrf : x.rfWF = true) (hrmw : x.rmwWF = true)
    (hbar : x.barWF = true) : x.consistent = true := by
  have hwf : x.wellFormed = true := by
    simp [wellFormed, hpo, hrf, T.coWF, T.scWF, hrmw, hbar]
  simp only [Execution.consistent, hwf, Bool.true_and, Bool.and_eq_true]
  refine ⟨⟨⟨⟨?_, ?_⟩, ?_⟩, ?_⟩, ?_⟩
  · exact T.acyclic_of_lt fun a b h => by
      simp only [get_union, Bool.or_eq_true] at h
      exact h.elim (T.rf_lt a b) (T.dep_lt a b)
  · simp only [atomicity, isEmpty_iff]
    intro a b
    cases h : ((x.strong x.fr ⨾ x.strong x.co) ∩ x.rmw) a b
    · rfl
    · exfalso
      simp only [get_inter, Bool.and_eq_true] at h
      obtain ⟨⟨w, h₁, h₂⟩, hrmw⟩ := And.intro (get_comp.1 h.1) h.2
      simp only [strong, get_inter, Bool.and_eq_true] at h₁ h₂
      obtain ⟨hw, _, hl, lt₂⟩ := (T.co_iff w b).1 h₂.1
      have lt₁ := T.fr_lt hrf h₁.1
      have hne : w ≠ b := fun e => by subst e; exact Nat.lt_irrefl _ lt₂
      exact absurd (T.rmw_adjacent a b w hrmw hw hl hne lt₁) (Nat.not_lt.2 (Nat.le_of_lt lt₂))
  · simp only [coherence, subset_iff, get_inter, Bool.and_eq_true]
    rintro a b ⟨h, hl⟩
    obtain ⟨⟨ha, h⟩, hb⟩ := comp_dom_iff.1 h |>.imp_left dom_comp_iff.1
    exact tc_of ((T.co_iff a b).2 ⟨ha, hb, hl, T.cause_lt hwf h⟩)
  · simp only [causality, irreflexive_iff]
    intro a
    cases h : (opt x.com ⨾ x.cause) a a
    · rfl
    · exfalso
      obtain ⟨z, h₁, h₂⟩ := get_comp.1 h
      have lt := T.cause_lt hwf h₂
      simp only [get_opt, Bool.or_eq_true, beq_iff_eq] at h₁
      rcases h₁ with rfl | h₁
      · exact Nat.lt_irrefl _ lt
      · exact Nat.lt_irrefl _ (Nat.lt_trans (T.com_lt hrf h₁) lt)
  · simp only [seqCst, subset_iff, get_inter, Bool.and_eq_true]
    rintro a b ⟨h, _⟩
    obtain ⟨⟨ha, h⟩, hb⟩ := comp_dom_iff.1 h |>.imp_left dom_comp_iff.1
    refine (T.sc_iff a b).2 ⟨ha, hb, ?_⟩
    rcases T.causeBase_lt hwf h with h | h
    · refine T.drains a b h ?_
      simp only [Event.isFenceSC, beq_iff_eq] at hb
      simp [Event.drains, Event.isReleaseFence, hb]
    · exact h

/-- `x` with more dependencies, `D`. -/
def withDep (x : Execution n V) (D : Rel n) : Execution n V := { x with dep := x.dep ∪ D }

/-- **More dependencies, still consistent**: dependencies `D` within program order that run forward
in the order keep a timed execution consistent. -/
theorem consistent_withDep {D : Rel n} (hD : ∀ a b, D a b = true → T.t a < T.t b)
    (hpo' : ∀ a b, D a b = true → x.po a b = true) (hpo : x.poWF = true) (hrf : x.rfWF = true)
    (hrmw : x.rmwWF = true) (hbar : x.barWF = true) : (withDep x D).consistent = true := by
  let T' : (withDep x D).Timed :=
    { t := T.t, inj := T.inj, rf_lt := T.rf_lt, rf_latest := T.rf_latest, init_first := T.init_first,
      co_iff := T.co_iff, sc_iff := T.sc_iff, rmw_lt := T.rmw_lt, rmw_adjacent := T.rmw_adjacent,
      fifo := T.fifo, drains := T.drains, blocks := T.blocks, bar_lt := T.bar_lt
      dep_lt := fun a b h => by
        simp only [withDep, get_union, Bool.or_eq_true] at h
        exact h.elim (T.dep_lt a b) (hD a b) }
  refine T'.consistent hpo hrf ?_ hbar
  simp only [rmwWF, Bool.and_eq_true, subset_iff, List.all_eq_true, List.mem_finRange, true_implies]
    at hrmw ⊢
  obtain ⟨⟨h₁, h₂⟩, h₃⟩ := hrmw
  refine ⟨⟨fun a b h => ?_, h₂⟩, fun a b h => ?_⟩
  · have := h₁ a b h
    simp only [withDep, get_comp, get_inter, get_union, Bool.and_eq_true, Bool.or_eq_true] at this ⊢
    obtain ⟨c, ⟨d, hd, ⟨⟨hp, hdep⟩, hs⟩⟩, hc⟩ := this
    exact ⟨c, ⟨d, hd, ⟨⟨hp, .inl hdep⟩, hs⟩⟩, hc⟩
  · simp only [withDep, get_union, Bool.or_eq_true] at h
    exact h.elim (h₃ a b) (hpo' a b)

end Timed
end Execution
end NV.PTX
