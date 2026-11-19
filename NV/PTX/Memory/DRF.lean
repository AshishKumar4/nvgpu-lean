import NV.PTX.Memory.Visibility

/-!
# Race freedom and sequential consistency

A candidate execution is *sequentially consistent* when program order and communication order
together are acyclic (`isSC`): its memory events then have a total order, extending program
order, in which every read reads the latest write to its location (`linear`). This is the
standard axiomatic statement of Lamport's SC (Shasha and Snir; Alglave et al., "Herding cats",
`sc.cat`).

Two memory events *conflict* when they are distinct, access one location and one of them writes
(§8.7.1). An execution is *race-free* when every conflicting pair is ordered by proxy-preserved
base causality order (§8.9.5), the order the Causality axiom constrains. This is causality order
without its observation prefix: a pair of morally strong accesses that is ordered only by one
reading the other (two relaxed accesses to a flag) is a race here, as it must be, since relaxed
and release/acquire accesses alone do not give SC (`SB` with release stores and acquire loads
allows both reads to return the initial value, §8.10.6).

* `drf_sc`: a consistent race-free execution is SC. The proxies take part through the
  proxy-preserved order: generic accesses to one address are ordered by plain causality order
  (`ppcb_of_generic`); an async-proxy write and a generic access, by causality through a
  `fence.proxy.async` of their CTA; aliases, by an alias fence. mbarrier operations are atomic
  operations on the barrier object, so their completion synchronizes through release and acquire
  patterns like any other atomic.
* `rmw_atomic_of_adjacent`: in such an execution, a read-modify-write whose write follows its read
  immediately in program order has no write to its location between them in communication order.
* `reads_latest`: the per-location form, for programs that synchronize through racing atomics
  (flags, counters, mbarriers) but whose data are race-free: a read of a race-free location reads
  the write causally latest before it, or the initial value when no write precedes it.
-/

namespace NV.PTX.Execution

open Rel

variable {n : Nat} {V : Type} [DecidableEq V] {x : Execution n V}

/-! ## Base causality order is transitive -/

theorem causeBase_hop {a u c : Fin n} (h : x.causeBase a u = true) (h₂ : x.hop u c = true) :
    x.causeBase a c = true := by
  rcases causeBase_iff.1 h with h | ⟨v, hv, hvu⟩
  · exact causeBase_iff.2 (.inr ⟨u, rtc_of_tc h, tc_of h₂⟩)
  · exact causeBase_iff.2 (.inr ⟨v, hv, tc_trans hvu (tc_of h₂)⟩)

theorem causeBase_trans {a b c : Fin n} (h₁ : x.causeBase a b = true) (h₂ : x.causeBase b c = true) :
    x.causeBase a c = true := by
  rcases causeBase_iff.1 h₂ with h | ⟨u, hu, hub⟩
  · exact causeBase_order h₁ (rtc_of_tc h)
  · have h₃ := causeBase_order h₁ hu
    rw [tc_iff_transGen] at hub
    clear h₂
    induction hub with
    | single e => exact causeBase_hop h₃ e
    | tail _ e ih => exact causeBase_hop ih e

/-! ## Proxy-preserved base causality order lies in base causality order -/

/-- `r` is contained in `s`. -/
def Sub (r s : Rel n) : Prop := ∀ a b, r a b = true → s a b = true

theorem Sub.union {r s t : Rel n} (h₁ : Sub r t) (h₂ : Sub s t) : Sub (r ∪ s) t := fun a b h => by
  simp only [get_union, Bool.or_eq_true] at h
  exact h.elim (h₁ a b) (h₂ a b)

theorem Sub.inter {r s t : Rel n} (h : Sub r t) : Sub (r ∩ s) t := fun a b hab => by
  simp only [get_inter, Bool.and_eq_true] at hab
  exact h a b hab.1

theorem Sub.domComp {p : Fin n → Bool} {r t : Rel n} (h : Sub r t) : Sub (dom p ⨾ r) t :=
  fun a c hac => by
    obtain ⟨b, hab, hbc⟩ := get_comp.1 hac
    simp only [get_dom, Bool.and_eq_true, beq_iff_eq] at hab
    exact hab.1 ▸ h b c hbc

theorem Sub.compDom {p : Fin n → Bool} {r t : Rel n} (h : Sub r t) : Sub (r ⨾ dom p) t :=
  fun a c hac => by
    obtain ⟨b, hab, hbc⟩ := get_comp.1 hac
    simp only [get_dom, Bool.and_eq_true, beq_iff_eq] at hbc
    exact hbc.1 ▸ h a b hab

theorem Sub.comp {r s : Rel n} (h₁ : Sub r x.causeBase) (h₂ : Sub s x.causeBase) :
    Sub (r ⨾ s) x.causeBase := fun a c hac => by
  obtain ⟨b, hab, hbc⟩ := get_comp.1 hac
  exact causeBase_trans (h₁ a b hab) (h₂ b c hbc)

theorem causeBase_of_ppcb {a b : Fin n} (h : x.proxyPreservedCauseBase a b = true) :
    x.causeBase a b = true := by
  have cb : Sub x.causeBase x.causeBase := fun _ _ h => h
  have f : Sub (x.causeBase ∩ inv x.proxyFenceOps) x.causeBase := cb.inter
  have g : Sub (x.causeBase ∩ x.proxyFenceOps) x.causeBase := cb.inter
  refine (Sub.union (Sub.union (Sub.union (Sub.union (Sub.union (Sub.union (Sub.union (Sub.union
    ?_ ?_) ?_) ?_) ?_) ?_) ?_) ?_) ?_) a b h
  · exact cb.domComp.compDom.inter
  · exact cb.inter.inter.inter
  · exact (Sub.comp f cb).compDom.inter
  · exact (Sub.comp cb.domComp g).inter
  · exact (Sub.comp (Sub.comp f cb) g).inter
  · exact (Sub.comp cb.domComp.compDom cb).compDom.inter
  · exact (Sub.comp (Sub.comp f cb).compDom cb).compDom.inter
  · exact (Sub.comp (Sub.comp cb.domComp.compDom cb) g).inter
  · exact (Sub.comp (Sub.comp (Sub.comp f cb).compDom cb) g).inter

/-! ## Conflicts and races -/

theorem sameLocation_iff {a b : Fin n} :
    x.sameLocation a b = true ↔ x.mems a = true ∧ x.mems b = true ∧ x.loc (x.ev a).addr = x.loc (x.ev b).addr := by
  simp [sameLocation, Bool.and_eq_true, and_assoc]

theorem sameLocation_symm {a b : Fin n} (h : x.sameLocation a b = true) : x.sameLocation b a = true := by
  rw [sameLocation_iff] at *; exact ⟨h.2.1, h.1, h.2.2.symm⟩

theorem sameLocation_trans {a b c : Fin n} (h₁ : x.sameLocation a b = true)
    (h₂ : x.sameLocation b c = true) : x.sameLocation a c = true := by
  rw [sameLocation_iff] at *; exact ⟨h₁.1, h₂.2.1, h₁.2.2.trans h₂.2.2⟩

theorem mems_of_writes {a : Fin n} (h : x.writes a = true) : x.mems a = true := by
  simp only [writes, mems, Event.isMem, Bool.or_eq_true] at *; exact .inr h

theorem mems_of_reads {a : Fin n} (h : x.reads a = true) : x.mems a = true := by
  simp only [reads, mems, Event.isMem, Bool.or_eq_true] at *; exact .inl h

theorem ne_of_reads_writes {a b : Fin n} (ha : x.reads a = true) (hb : x.writes b = true) : a ≠ b := by
  rintro rfl
  simp only [reads, writes, Event.isRead, Event.isWrite, beq_iff_eq] at ha hb
  rw [ha] at hb; cases hb

/-- What `rfWF` says of one `rf` edge. -/
theorem rf_parts (hc : x.consistent = true) {w r : Fin n} (h : x.rf w r = true) :
    x.writes w = true ∧ x.reads r = true ∧ x.sameLocation w r = true := by
  have := (wellFormed_parts (consistent_parts hc).1).2.1
  simp only [rfWF, Bool.and_eq_true, subset_iff] at this
  obtain ⟨b, hb, hbr⟩ := get_comp.1 (this.1 w r h)
  obtain ⟨c, hc, hcb⟩ := get_comp.1 hb
  simp only [get_dom, Bool.and_eq_true, beq_iff_eq] at hc hbr
  obtain ⟨rfl, hw⟩ := hc
  obtain ⟨rfl, hr⟩ := hbr
  exact ⟨hw, hr, hcb⟩

/-- What `coWF` says of one `co` edge. -/
theorem co_parts (hc : x.consistent = true) {a b : Fin n} (h : x.co a b = true) :
    x.writes a = true ∧ x.writes b = true ∧ x.sameLocation a b = true := by
  have := (wellFormed_parts (consistent_parts hc).1).2.2.1
  simp only [coWF, Bool.and_eq_true, subset_iff] at this
  obtain ⟨c, hc', hcb⟩ := get_comp.1 (this.1.2 a b h)
  obtain ⟨d, hd, hdc⟩ := get_comp.1 hc'
  simp only [get_dom, Bool.and_eq_true, beq_iff_eq] at hd hcb
  obtain ⟨rfl, hw⟩ := hd
  obtain ⟨rfl, hw'⟩ := hcb
  exact ⟨hw, hw', hdc⟩

theorem fr_parts (hc : x.consistent = true) {r w : Fin n} (h : x.fr r w = true) :
    x.reads r = true ∧ x.writes w = true ∧ x.sameLocation r w = true := by
  simp only [fr, get_union, Bool.or_eq_true] at h
  rcases h with h | h
  · obtain ⟨w₀, h₁, h₂⟩ := get_comp.1 h
    rw [get_inv] at h₁
    obtain ⟨hw₀, hr, hl₁⟩ := rf_parts hc h₁
    obtain ⟨_, hw, hl₂⟩ := co_parts hc (co_of_tc hc h₂)
    exact ⟨hr, hw, sameLocation_trans (sameLocation_symm hl₁) hl₂⟩
  · simp only [get_ofFn, Bool.and_eq_true] at h
    exact ⟨h.1.1.1, h.1.1.2, h.1.2⟩

/-- Two distinct memory events conflict when they access one location and one of them writes
(§8.7.1). -/
def conflict (x : Execution n V) (a b : Fin n) : Prop :=
  a ≠ b ∧ x.sameLocation a b = true ∧ (x.writes a = true ∨ x.writes b = true)

/-- Every conflicting pair on locations satisfying `at_` is ordered by proxy-preserved base
causality order. -/
def RaceFreeOn (x : Execution n V) (at_ : Nat → Prop) : Prop :=
  ∀ a b, x.conflict a b → at_ (x.loc (x.ev a).addr) →
    x.proxyPreservedCauseBase a b = true ∨ x.proxyPreservedCauseBase b a = true

def RaceFree (x : Execution n V) : Prop := x.RaceFreeOn fun _ => True

theorem com_conflict (hc : x.consistent = true) {a b : Fin n} (h : x.com a b = true) :
    x.conflict a b := by
  simp only [com, get_union, Bool.or_eq_true] at h
  rcases h with (h | h) | h
  · obtain ⟨hw, hr, hl⟩ := rf_parts hc h
    exact ⟨(ne_of_reads_writes hr hw).symm, hl, .inl hw⟩
  · have h := co_of_tc hc h
    obtain ⟨hw, _, hl⟩ := co_parts hc h
    exact ⟨fun e => (by subst e; rw [co_irrefl hc] at h; cases h), hl, .inl hw⟩
  · obtain ⟨hr, hw, hl⟩ := fr_parts hc h
    exact ⟨ne_of_reads_writes hr hw, hl, .inr hw⟩

theorem cause_of_ppcb {a b : Fin n} (h : x.proxyPreservedCauseBase a b = true) : x.cause a b = true :=
  comp_of (b := a) (by simp) h

/-- Causality (§8.10.6): communication never runs against proxy-preserved causality. -/
theorem not_com_of_ppcb (hc : x.consistent = true) {a b : Fin n}
    (h : x.proxyPreservedCauseBase b a = true) (hcom : x.com a b = true) : False := by
  have hcaus := (consistent_parts hc).2.2.2.2.1
  simp only [causality, irreflexive_iff] at hcaus
  have := hcaus a
  rw [comp_of (b := b) (by simp [hcom]) (cause_of_ppcb h)] at this
  cases this

/-- No memory event precedes itself in base causality order. -/
theorem causeBase_irrefl (hc : x.consistent = true) {a : Fin n} (hm : x.mems a = true)
    (h : x.causeBase a a = true) : False := by
  have hp : x.proxyPreservedCauseBase a a = true := by
    simp only [proxyPreservedCauseBase, get_union, get_inter, Bool.or_eq_true, Bool.and_eq_true]
    left; left; left; left; left; left; left
    right
    simp [h, sameBlock, sameProxy, sameAlias, hm]
  have hcaus := (consistent_parts hc).2.2.2.2.1
  simp only [causality, irreflexive_iff] at hcaus
  have := hcaus a
  rw [comp_of (b := a) (by simp) (cause_of_ppcb hp)] at this
  cases this

theorem causeBase_of_com (hc : x.consistent = true) (hr : x.RaceFree) {a b : Fin n}
    (h : x.com a b = true) : x.causeBase a b = true := by
  rcases hr a b (com_conflict hc h) trivial with h' | h'
  · exact causeBase_of_ppcb h'
  · exact (not_com_of_ppcb hc h' h).elim

/-! ## DRF-SC -/

/-- Sequential consistency (Shasha and Snir): program order and communication order are acyclic,
and the writes to each location are totally ordered by coherence. -/
def IsSC (x : Execution n V) : Prop :=
  acyclic (x.order ∪ x.com) = true ∧
    ∀ a b, a ≠ b → x.writes a = true → x.writes b = true → x.sameLocation a b = true →
      x.co a b = true ∨ x.co b a = true

theorem drf_sc (hc : x.consistent = true) (hr : x.RaceFree) : x.IsSC := by
  constructor
  · -- Every edge of a cycle is base causality order, and some edge leaves a memory event.
    have key : ∀ {a b}, Relation.TransGen (fun a b => (x.order ∪ x.com) a b = true) a b →
        tc x.order a b = true ∨
          ∃ m, x.mems m = true ∧ (m = a ∨ x.causeBase a m = true) ∧ x.causeBase m b = true := by
      intro a b h
      induction h with
      | single e =>
        simp only [get_union, Bool.or_eq_true] at e
        rcases e with e | e
        · exact .inl (tc_of e)
        · obtain ⟨_, hl, _⟩ := com_conflict hc e
          exact .inr ⟨_, ((sameLocation_iff).1 hl).1, .inl rfl, causeBase_of_com hc hr e⟩
      | @tail c d _ e ih =>
        simp only [get_union, Bool.or_eq_true] at e
        rcases ih with ih | ⟨m, hm, ham, hmc⟩ <;> rcases e with e | e
        · exact .inl (tc_trans ih (tc_of e))
        · obtain ⟨_, hl, _⟩ := com_conflict hc e
          exact .inr ⟨c, ((sameLocation_iff).1 hl).1, .inr (causeBase_of_order ih),
            causeBase_of_com hc hr e⟩
        · exact .inr ⟨m, hm, ham, causeBase_order hmc (rtc_of_tc (tc_of e))⟩
        · exact .inr ⟨m, hm, ham, causeBase_trans hmc (causeBase_of_com hc hr e)⟩
    simp only [acyclic, irreflexive_iff]
    intro a
    cases h : tc (x.order ∪ x.com) a a
    · rfl
    · exfalso
      rcases key ((tc_iff_transGen _ _ _).1 h) with h | ⟨m, hm, ham, hma⟩
      · have := (wellFormed_parts (consistent_parts hc).1).1
        simp only [poWF, Bool.and_eq_true, acyclic, irreflexive_iff] at this
        rw [this.2 a] at h; cases h
      · rcases ham with rfl | ham
        · exact causeBase_irrefl hc hm hma
        · exact causeBase_irrefl hc hm (causeBase_trans hma ham)
  · intro a b hab ha hb hl
    have hcoh := (consistent_parts hc).2.2.2.1
    simp only [coherence, subset_iff] at hcoh
    have co_of {a b : Fin n} (ha : x.writes a = true) (hb : x.writes b = true)
        (hl : x.sameLocation a b = true) (h : x.proxyPreservedCauseBase a b = true) :
        x.co a b = true :=
      co_of_tc hc (hcoh a b (by
        simp only [get_inter, Bool.and_eq_true]
        exact ⟨comp_of (comp_of (by simp [ha]) (cause_of_ppcb h)) (by simp [hb]), hl⟩))
    rcases hr a b (show x.conflict a b from ⟨hab, hl, .inl ha⟩) trivial with h | h
    · exact .inl (co_of ha hb hl h)
    · exact .inr (co_of hb ha (sameLocation_symm hl) h)

/-! ## An SC execution is an interleaving -/

/-- In an SC execution, the memory events have a total order (`rank`) extending program order and
communication order, in which each read reads the latest write to its location before it, or the
initial value when no write to its location precedes it. -/
theorem IsSC.reads_latest (hc : x.consistent = true) (hsc : x.IsSC) :
    (∀ a b, (x.order ∪ x.com) a b = true → rank (x.order ∪ x.com) a < rank (x.order ∪ x.com) b) ∧
    (∀ w r w', x.rf w r = true → x.writes w' = true → x.sameLocation w' r = true → w' ≠ w →
      rank (x.order ∪ x.com) w' < rank (x.order ∪ x.com) w ∨
        rank (x.order ∪ x.com) r < rank (x.order ∪ x.com) w') ∧
    (∀ r w', x.reads r = true → (∀ w, x.rf w r = false) → x.writes w' = true →
      x.sameLocation w' r = true → rank (x.order ∪ x.com) r < rank (x.order ∪ x.com) w') := by
  have mono : ∀ a b, (x.order ∪ x.com) a b = true →
      rank (x.order ∪ x.com) a < rank (x.order ∪ x.com) b := fun a b h => rank_lt hsc.1 (tc_of h)
  have ofCom : ∀ {a b}, x.com a b = true → rank (x.order ∪ x.com) a < rank (x.order ∪ x.com) b :=
    fun h => mono _ _ (by simp [h])
  refine ⟨mono, fun w r w' hrf hw' hl hne => ?_, fun r w' hr hnone hw' hl => ?_⟩
  · obtain ⟨hw, _, hwr⟩ := rf_parts hc hrf
    rcases hsc.2 w' w hne hw' hw (sameLocation_trans hl (sameLocation_symm hwr)) with h | h
    · exact .inl (ofCom (by simp [com, h, tc_of h]))
    · refine .inr (ofCom (a := r) ?_)
      simp only [com, get_union, Bool.or_eq_true]
      exact .inr (by simp only [fr, get_union, Bool.or_eq_true]; exact .inl (comp_of (by simp [hrf]) (tc_of h)))
  · refine ofCom (a := r) ?_
    simp only [com, get_union, Bool.or_eq_true]
    refine .inr ?_
    simp only [fr, get_union, Bool.or_eq_true, get_ofFn, Bool.and_eq_true, Bool.not_eq_true',
      List.any_eq_false, List.mem_finRange, true_implies]
    exact .inr ⟨⟨⟨hr, hw'⟩, sameLocation_symm hl⟩, fun w => by simp [hnone w]⟩

/-! ## Read-modify-writes stay atomic -/

/-- Synchronization starts at a release write or fence, a fence.sc, or a barrier arrival, never at
a read. -/
theorem not_reads_of_hop (hc : x.consistent = true) {u v : Fin n}
    (h : (x.sc ∪ x.sync) u v = true) : x.reads u = false := by
  have hw := wellFormed_parts (consistent_parts hc).1
  simp only [get_union, Bool.or_eq_true] at h
  have kind_ne : ∀ k, (x.ev u).kind = k → k ≠ .read → x.reads u = false := fun k hk hne => by
    simp only [reads, Event.isRead, hk]; simpa using hne
  rcases h with h | h
  · have := hw.2.2.2.1
    simp only [scWF, Bool.and_eq_true, subset_iff] at this
    have := this.1.2 u v h
    simp only [get_prod, Bool.and_eq_true, Event.isFenceSC, beq_iff_eq] at this
    exact kind_ne _ this.1 (by simp)
  · simp only [sync, strong, get_union, get_inter, Bool.or_eq_true, Bool.and_eq_true] at h
    rcases h with ⟨h, _⟩ | h
    · obtain ⟨_, h, _⟩ := get_comp.1 h
      obtain ⟨_, h, _⟩ := get_comp.1 h
      simp only [releaseSequence, strong, get_union, get_inter, Bool.or_eq_true,
        Bool.and_eq_true] at h
      rcases h with ⟨h, _⟩ | ⟨h, _⟩ <;>
        obtain ⟨_, h, _⟩ := get_comp.1 h <;> obtain ⟨_, h, _⟩ := get_comp.1 h <;>
        simp only [get_dom, Bool.and_eq_true, beq_iff_eq] at h
      · exact (Bool.eq_false_iff.2 fun hr => ne_of_reads_writes hr h.2.2 rfl)
      · simp only [Event.isReleaseFence, Bool.or_eq_true, beq_iff_eq] at h
        rcases h.2 with h | h <;> exact kind_ne _ h (by simp)
    · have := hw.2.2.2.2.2
      simp only [barWF, List.all_eq_true, List.mem_finRange, true_implies] at this
      have := this u v
      simp only [h, Bool.not_true, Bool.false_or] at this
      split at this <;> first | cases this | exact kind_ne _ ‹_› (by simp)

/-- In a consistent race-free execution, no write to its location comes between the read and the
write of a read-modify-write that program order keeps adjacent. -/
theorem rmw_atomic_of_adjacent (hc : x.consistent = true) (hr : x.RaceFree) {r w w' : Fin n}
    (hrmw : x.rmw r w = true) (hadj : ∀ u, tc x.order r u = true → u = w ∨ tc x.order w u = true)
    (hfr : x.fr r w' = true) (hco : x.co w' w = true) : False := by
  obtain ⟨hrd, hw', _⟩ := fr_parts hc hfr
  obtain ⟨_, hw, hl⟩ := co_parts hc hco
  have hne : w' ≠ w := fun e => by subst e; rw [co_irrefl hc] at hco; cases hco
  -- `r` precedes `w'` causally; since `r` starts no synchronization, so does `w`.
  have hrw' : x.causeBase r w' = true :=
    causeBase_of_com hc hr (by simp [com, hfr])
  have hww' : w = w' ∨ x.causeBase w w' = true := by
    rcases causeBase_iff.1 hrw' with h | ⟨u, hu, hub⟩
    · rcases hadj w' h with rfl | h
      · exact .inl rfl
      · exact .inr (causeBase_of_order h)
    · have hu' : u = w ∨ tc x.order w u = true := by
        simp only [rtc, get_opt, Bool.or_eq_true, beq_iff_eq] at hu
        rcases hu with rfl | hu
        · exfalso
          obtain ⟨v, hv⟩ := tc_first hub
          obtain ⟨v', hv', _⟩ := get_comp.1 hv
          rw [not_reads_of_hop hc hv'] at hrd; cases hrd
        · exact hadj u hu
      exact .inr (causeBase_iff.2 (.inr ⟨u, by
        rcases hu' with rfl | h
        · exact rtc_refl _ _
        · exact rtc_of_tc h, hub⟩))
  rcases hww' with rfl | hww'
  · exact hne rfl
  · rcases hr w' w ⟨hne, hl, .inl hw'⟩ trivial with h | h
    · -- `w'` before `w` causally as well: a cycle through a memory event.
      exact causeBase_irrefl hc (mems_of_writes hw) (causeBase_trans hww' (causeBase_of_ppcb h))
    · have hcoh := (consistent_parts hc).2.2.2.1
      simp only [coherence, subset_iff] at hcoh
      have := co_of_tc hc (hcoh w w' (by
        simp only [get_inter, Bool.and_eq_true]
        exact ⟨comp_of (comp_of (by simp [hw]) (cause_of_ppcb h)) (by simp [hw']),
          sameLocation_symm hl⟩))
      have := co_trans hc this hco
      rw [co_irrefl hc w] at this
      cases this

/-! ## Asynchronous completion publishes across proxies -/

/-- **An async-proxy access is ordered before a generic one through a proxy fence.** An access `w`
in the async proxy (a TMA copy's write into shared memory), followed in its thread's order by a
`fence.proxy.async` of its CTA (the implicit generic-async proxy fence that ends a
`cp.async.bulk`, §9.7.9.25), which comes causally before a generic access `r` to the same address
(the copy's `complete_tx` releasing an mbarrier, a consumer's acquiring wait, then its read): `w`
is ordered before `r` in proxy-preserved base causality order, so the two do not race. -/
theorem ppcb_async_completion {w f r : Fin n} (hw : x.mems w = true)
    (hf : (x.ev f).proxyFenced = some (x.ev w).proxy) (hblock : x.sameBlock f w = true)
    (hwf : tc x.order w f = true) (hfr : x.causeBase f r = true) (hr : x.generic r = true)
    (hal : x.sameAlias w r = true) : x.proxyPreservedCauseBase w r = true := by
  have hfence : x.proxyFenceOps f w = true := by
    simp [proxyFenceOps, hf, hw, hblock]
  have hsrc : (x.causeBase ∩ inv x.proxyFenceOps) w f = true := by
    simp only [get_inter, get_inv, Bool.and_eq_true]
    exact ⟨causeBase_of_order hwf, hfence⟩
  have hc : (((x.causeBase ∩ inv x.proxyFenceOps) ⨾ x.causeBase ⨾ dom x.generic) ∩ x.sameAlias) w r =
      true := by
    simp only [get_inter, Bool.and_eq_true]
    exact ⟨comp_dom_iff.2 ⟨comp_of hsrc hfr, hr⟩, hal⟩
  simp only [proxyPreservedCauseBase, get_union, hc, Bool.or_true, Bool.true_or]

/-- **A generic access is ordered before an async-proxy one through a proxy fence.** The other
direction: a generic access `r`, causally before a `fence.proxy.async` of the CTA of an async
access `w`, which comes before `w` in causality order. -/
theorem ppcb_async_issue {r f w : Fin n} (hr : x.generic r = true) (hrf : x.causeBase r f = true)
    (hw : x.mems w = true) (hf : (x.ev f).proxyFenced = some (x.ev w).proxy)
    (hblock : x.sameBlock f w = true) (hfw : x.causeBase f w = true)
    (hal : x.sameAlias r w = true) : x.proxyPreservedCauseBase r w = true := by
  have hfence : x.proxyFenceOps f w = true := by
    simp [proxyFenceOps, hf, hw, hblock]
  have hdst : (x.causeBase ∩ x.proxyFenceOps) f w = true := by
    simp only [get_inter, Bool.and_eq_true]
    exact ⟨hfw, hfence⟩
  have hc : ((dom x.generic ⨾ x.causeBase ⨾ (x.causeBase ∩ x.proxyFenceOps)) ∩ x.sameAlias) r w =
      true := by
    simp only [get_inter, Bool.and_eq_true]
    exact ⟨comp_of (dom_comp_iff.2 ⟨hr, hrf⟩) hdst, hal⟩
  simp only [proxyPreservedCauseBase, get_union, hc, Bool.or_true, Bool.true_or]

theorem ppcb_async_both {w f g w' : Fin n} (hw : x.mems w = true) (hw' : x.mems w' = true)
    (hf : (x.ev f).proxyFenced = some (x.ev w).proxy) (hfb : x.sameBlock f w = true)
    (hwf : tc x.order w f = true) (hfg : x.causeBase f g = true)
    (hg : (x.ev g).proxyFenced = some (x.ev w').proxy) (hgb : x.sameBlock g w' = true)
    (hgw : x.causeBase g w' = true) (hal : x.sameAlias w w' = true) :
    x.proxyPreservedCauseBase w w' = true := by
  have hsrc : (x.causeBase ∩ inv x.proxyFenceOps) w f = true := by
    simp only [get_inter, get_inv, Bool.and_eq_true]
    exact ⟨causeBase_of_order hwf, by simp [proxyFenceOps, hf, hw, hfb]⟩
  have hdst : (x.causeBase ∩ x.proxyFenceOps) g w' = true := by
    simp only [get_inter, Bool.and_eq_true]
    exact ⟨hgw, by simp [proxyFenceOps, hg, hw', hgb]⟩
  have hc : (((x.causeBase ∩ inv x.proxyFenceOps) ⨾ x.causeBase ⨾ (x.causeBase ∩ x.proxyFenceOps)) ∩
      x.sameAlias) w w' = true := by
    simp only [get_inter, Bool.and_eq_true]
    exact ⟨comp_of (comp_of hsrc hfg) hdst, hal⟩
  simp only [proxyPreservedCauseBase, get_union, hc, Bool.or_true, Bool.true_or]

/-- Coherence (§8.10.1): writes to one location ordered by proxy-preserved causality are in
coherence order. -/
theorem co_of_ppcb (hc : x.consistent = true) {a b : Fin n} (ha : x.writes a = true)
    (hb : x.writes b = true) (hl : x.sameLocation a b = true)
    (h : x.proxyPreservedCauseBase a b = true) : x.co a b = true := by
  have hcoh := (consistent_parts hc).2.2.2.1
  simp only [coherence, subset_iff] at hcoh
  exact co_of_tc hc (hcoh a b (by
    simp only [get_inter, Bool.and_eq_true]
    exact ⟨comp_of (comp_of (by simp [ha]) (cause_of_ppcb h)) (by simp [hb]), hl⟩))

/-! ## A read-modify-write reads its coherence predecessor -/

/-- **A read-modify-write reads the write just before its own in coherence order.** Let `w` be
the write of read-modify-write `(r, w)`, `w'` the write coherence-before `w` with no write of the
location between them, and every access of the location morally strong with every other (one
proxy, scopes covering each other: an mbarrier's operations within a CTA). If `r` precedes `w` in
proxy-preserved base causality (program order does it for one thread's generic accesses), `r`
reads from `w'`: reading the initial value or a write coherence-before `w'` breaks atomicity
(§8.10.3); reading `w` itself or a write after it breaks causality (§8.10.6) or coherence (§8.10.1). -/
theorem rmw_reads_prev (hc : x.consistent = true) {r w w' : Fin n} (hrmw : x.rmw r w = true)
    (hrw : x.proxyPreservedCauseBase r w = true) (hco : x.co w' w = true)
    (hadj : ∀ v, x.co w' v = true → x.co v w = false)
    (hs : ∀ a b, x.sameLocation a w = true → x.sameLocation b w = true → x.strongR a b = true)
    (hal : ∀ a, x.sameLocation a w = true → (x.ev a).addr = (x.ev w).addr) :
    x.rf w' r = true := by
  have hww : x.sameLocation w w = true := sameLocation_iff.2 ⟨mems_of_writes (co_parts hc hco).2.1,
    mems_of_writes (co_parts hc hco).2.1, rfl⟩
  have hat := (consistent_parts hc).2.2.1
  simp only [atomicity, isEmpty_iff] at hat
  obtain ⟨hw', hw, hl'⟩ := co_parts hc hco
  have hrl : x.sameLocation r w = true := by
    have := (wellFormed_parts (consistent_parts hc).1).2.2.2.2.1
    simp only [rmwWF, Bool.and_eq_true, subset_iff] at this
    obtain ⟨b, hb, hbw⟩ := get_comp.1 (this.1.1 r w hrmw)
    obtain ⟨a, ha, hab⟩ := get_comp.1 hb
    simp only [get_dom, Bool.and_eq_true, beq_iff_eq] at ha hbw
    obtain ⟨rfl, hr⟩ := ha
    obtain ⟨rfl, -⟩ := hbw
    simp only [get_inter, Bool.and_eq_true, sameAlias, get_ofFn, beq_iff_eq] at hab
    simp [sameLocation, hab.2.1.1, hab.2.1.2, hab.2.2]
  have hr : x.reads r = true := by
    have := (wellFormed_parts (consistent_parts hc).1).2.2.2.2.1
    simp only [rmwWF, Bool.and_eq_true, subset_iff] at this
    obtain ⟨b, hb, _⟩ := get_comp.1 (this.1.1 r w hrmw)
    obtain ⟨a, ha, _⟩ := get_comp.1 hb
    simp only [get_dom, Bool.and_eq_true, beq_iff_eq] at ha
    obtain ⟨rfl, hr⟩ := ha
    exact hr
  have hl'w : x.sameLocation w' w = true := hl'
  -- Reading before `w'`: `r` comes before `w'` in `fr`, `w'` before `w` in `co`.
  have notFr : x.fr r w' = false := by
    cases hfr : x.fr r w'
    · rfl
    · have h := hat r w
      rw [get_inter, hrmw, Bool.and_true,
        comp_of (b := w') (by simp [strong, hfr, hs r w' hrl hl'w])
          (by simp [strong, hco, hs w' w hl'w hww])] at h
      cases h
  rcases rf_cases hc hr with ⟨hnone, -⟩ | ⟨u, hrf, -, hu, -⟩
  · exfalso
    have : x.fr r w' = true := by
      simp only [fr, get_union, Bool.or_eq_true, get_ofFn, Bool.and_eq_true, Bool.not_eq_true',
        List.any_eq_false, List.mem_finRange, true_implies]
      exact .inr ⟨⟨⟨hr, hw'⟩, sameLocation_trans hrl (sameLocation_symm hl'w)⟩, fun v => by simp [hnone v]⟩
    rw [notFr] at this; cases this
  · have hul : x.sameLocation u w = true := sameLocation_trans (rf_parts hc hrf).2.2 hrl
    by_cases hu' : u = w'
    · exact hu' ▸ hrf
    exfalso
    rcases co_total hc hu' hu hw' ((hal u hul).trans (hal w' hl'w).symm) (hs u w' hul hl'w) with h | h
    · -- `u` before `w'`: `r` reads before `w'`.
      have : x.fr r w' = true := by
        simp only [fr, get_union, Bool.or_eq_true]
        exact .inl (comp_of (by simp [hrf]) (tc_of h))
      rw [notFr] at this; cases this
    · by_cases huw : u = w
      · subst huw; exact not_com_of_ppcb hc hrw (by simp [com, hrf])
      rcases co_total hc huw hu hw (hal u hul) (hs u w hul hww) with h' | h'
      · rw [hadj u h] at h'; cases h'
      · -- `w` before `u`, yet `u` is observed by `r`, which precedes `w`: coherence orders `u` first.
        have hcoh := (consistent_parts hc).2.2.2.1
        simp only [coherence, subset_iff] at hcoh
        have := co_of_tc hc (hcoh u w (by
          simp only [get_inter, Bool.and_eq_true]
          have hcause : x.cause u w = true :=
            comp_of (b := r) (by simp [observation, strong, hrf, hs u r hul hrl]) hrw
          exact ⟨comp_of (b := w) (comp_of (b := u) (by simp [hu]) hcause) (by simp [hw]), hul⟩))
        have := co_trans hc this h'
        rw [co_irrefl hc u] at this; cases this

end NV.PTX.Execution
