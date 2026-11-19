import NV.PTX.Memory.DRF
import NV.PTX.Memory.Timed
import Examples.SplitMerge

/-!
# Ring safety as race freedom

The producer/consumer ring of `Examples.Ring` in an execution of the general memory model. Stage
`q < N` uses slot `q % S`: a TMA copy writes the slot through the async proxy (`W q`), its
completion is followed by the implicit generic-async proxy fence (`F q`, PTX ISA §9.7.9.25), and
consumer `c < C` reads the slot through the generic proxy (`R q c`). Before the copy of stage `q`
reuses a slot (`S ≤ q`), the producer has a `fence.proxy.async` (`G q`).

`Ring` takes the two orders the barriers give as hypotheses: the full barrier orders the
completion's fence before every consumer's read (`raw`; a `complete_tx` release, the consumer's
acquiring wait), the empty barrier orders every consumer's read of the previous round before the
producer's fence (`war`; each consumer's release arrival, the producer's acquiring wait). From
them:

* `race_free`: every two conflicting accesses to the ring's slots are ordered by proxy-preserved
  base causality order, so the ring has no data race (`Execution.RaceFreeOn`);
* `reads_stage`: every consumer read of stage `q` reads the copy of stage `q` (none from an
  earlier round, none from a later one).

The fence `G` is needed: without it a consumer's generic read and the next round's async write are
ordered by causality but not by a proxy-preserved order, which the model (§8.6, §8.9.5; ptx.als)
counts as a race.
-/

namespace Examples.Ring.Race

open NV NV.PTX NV.PTX.Execution NV.Rel

variable {n : Nat} {V : Type} [DecidableEq V]

structure Ring (x : Execution n V) where
  /-- Stages, slots and consumers. -/
  N : Nat
  S : Nat
  C : Nat
  slots_pos : 0 < S
  consumers_pos : 0 < C
  W : Nat → Fin n
  F : Nat → Fin n
  R : Nat → Nat → Fin n
  G : Nat → Fin n
  /-- Address of each slot. -/
  A : Nat → Nat
  /-- Distinct slots are distinct locations. -/
  slots_apart : ∀ i < S, ∀ j < S, i < N → j < N → x.loc (A i) = x.loc (A j) → i = j
  write : ∀ q < N, x.writes (W q) = true ∧ (x.ev (W q)).addr = A (q % S)
  fence : ∀ q < N, (x.ev (F q)).proxyFenced = some (x.ev (W q)).proxy ∧ x.sameBlock (F q) (W q) = true ∧
    tc x.order (W q) (F q) = true
  read : ∀ q < N, ∀ c < C, x.reads (R q c) = true ∧ x.generic (R q c) = true ∧
    (x.ev (R q c)).addr = A (q % S)
  reuse : ∀ q < N, S ≤ q → (x.ev (G q)).proxyFenced = some (x.ev (W q)).proxy ∧
    x.sameBlock (G q) (W q) = true ∧ x.causeBase (G q) (W q) = true
  raw : ∀ q < N, ∀ c < C, x.causeBase (F q) (R q c) = true
  war : ∀ q < N, S ≤ q → ∀ c < C, x.causeBase (R (q - S) c) (G q) = true

namespace Ring

variable {x : Execution n V} (K : Ring x)

theorem mems_W {q : Nat} (hq : q < K.N) : x.mems (K.W q) = true := mems_of_writes (K.write q hq).1

theorem mems_R {q c : Nat} (hq : q < K.N) (hc : c < K.C) : x.mems (K.R q c) = true :=
  mems_of_reads (K.read q hq c hc).1

/-- Each round's fence comes causally before every later round's reuse fence and copy fence of the
same slot. -/
theorem chain : ∀ k q, q + (k + 1) * K.S < K.N →
    x.causeBase (K.F q) (K.G (q + (k + 1) * K.S)) = true ∧
      x.causeBase (K.F q) (K.F (q + (k + 1) * K.S)) = true
  | 0, q, h => by
    have hq' : q + K.S < K.N := by simpa using h
    have hS : K.S ≤ q + K.S := Nat.le_add_left _ _
    have hg : x.causeBase (K.F q) (K.G (q + K.S)) = true := by
      have := K.war (q + K.S) hq' hS 0 K.consumers_pos
      rw [Nat.add_sub_cancel] at this
      exact causeBase_trans (K.raw q (by omega) 0 K.consumers_pos) this
    simp only [Nat.zero_add, Nat.one_mul]
    refine ⟨hg, ?_⟩
    exact causeBase_order (causeBase_trans hg (K.reuse _ hq' hS).2.2)
      (rtc_of_tc (K.fence _ hq').2.2)
  | k + 1, q, h => by
    have h₁ : q + (k + 1) * K.S < K.N := by
      have : (k + 1 + 1) * K.S = (k + 1) * K.S + K.S := Nat.succ_mul (k + 1) K.S
      omega
    obtain ⟨-, ih⟩ := chain k q h₁
    have e : q + (k + 1 + 1) * K.S = q + (k + 1) * K.S + K.S := by rw [Nat.succ_mul]; omega
    obtain ⟨h₂, h₃⟩ := chain 0 (q + (k + 1) * K.S) (by rw [Nat.zero_add, Nat.one_mul, ← e]; exact h)
    simp only [Nat.zero_add, Nat.one_mul] at h₂ h₃
    rw [e]
    exact ⟨causeBase_trans ih h₂, causeBase_trans ih h₃⟩

omit [DecidableEq V] in
/-- Two stages of one slot, the earlier `q`, the later `q + (k + 1) * S`. -/
theorem same_slot {q q' : Nat} (h : q % K.S = q' % K.S) (hlt : q < q') :
    ∃ k, q' = q + (k + 1) * K.S := by
  have h0 : (q' - q) % K.S = 0 := Nat.sub_mod_eq_zero_of_mod_eq h.symm
  obtain ⟨m, hm⟩ := Nat.dvd_of_mod_eq_zero h0
  have hm0 : m ≠ 0 := by
    rintro rfl
    rw [Nat.mul_zero] at hm
    omega
  refine ⟨m - 1, ?_⟩
  rw [Nat.sub_add_cancel (Nat.pos_of_ne_zero hm0), Nat.mul_comm, ← hm]
  omega


omit [DecidableEq V] in
theorem slot_lt {q : Nat} : q % K.S < K.S := Nat.mod_lt _ K.slots_pos

/-- Two accesses of the ring at one location belong to one slot. -/
theorem slot_eq {a b : Fin n} {q q' : Nat} (hq : q < K.N) (hq' : q' < K.N)
    (ha : (x.ev a).addr = K.A (q % K.S)) (hb : (x.ev b).addr = K.A (q' % K.S))
    (hl : x.sameLocation a b = true) : q % K.S = q' % K.S := by
  have := ((sameLocation_iff).1 hl).2.2
  rw [ha, hb] at this
  exact K.slots_apart _ K.slot_lt _ K.slot_lt (Nat.lt_of_le_of_lt (Nat.mod_le _ _) hq)
    (Nat.lt_of_le_of_lt (Nat.mod_le _ _) hq') this

omit [DecidableEq V] in
theorem alias_of {a b : Fin n} {q q' : Nat} (ha : (x.ev a).addr = K.A (q % K.S))
    (hb : (x.ev b).addr = K.A (q' % K.S)) (hma : x.mems a = true) (hmb : x.mems b = true)
    (h : q % K.S = q' % K.S) : x.sameAlias a b = true := by
  simp [sameAlias, hma, hmb, ha, hb, h]

/-- A later round's reuse fence comes after every consumer's read of an earlier round of the
slot. -/
theorem read_before_reuse {q q' c : Nat} (hc : c < K.C) (h : q % K.S = q' % K.S) (hlt : q < q')
    (hq' : q' < K.N) : x.causeBase (K.R q c) (K.G q') = true := by
  obtain ⟨k, rfl⟩ := K.same_slot h hlt
  have hS := K.slots_pos
  have hw := K.war (q + K.S) (by have : (k + 1) * K.S = k * K.S + K.S := Nat.succ_mul k K.S; omega) (Nat.le_add_left _ _) c hc
  rw [Nat.add_sub_cancel] at hw
  cases k with
  | zero => simpa using hw
  | succ k =>
    have e : q + (k + 1 + 1) * K.S = q + K.S + (k + 1) * K.S := by
      have : (k + 1 + 1) * K.S = (k + 1) * K.S + K.S := Nat.succ_mul (k + 1) K.S
      omega
    rw [e] at hq' ⊢
    have h₁ : q + K.S < K.N := by omega
    have hfg := (K.chain k (q + K.S) hq').1
    exact causeBase_trans hw (causeBase_trans (causeBase_order (K.reuse _ h₁ (Nat.le_add_left _ _)).2.2
      (rtc_of_tc (K.fence _ h₁).2.2)) hfg)

/-- A copy comes before every consumer read of the same or a later round of its slot. -/
theorem copy_before_read {q q' c : Nat} (hc : c < K.C) (h : q % K.S = q' % K.S) (hle : q ≤ q')
    (hq' : q' < K.N) : x.causeBase (K.F q) (K.R q' c) = true := by
  rcases Nat.lt_or_eq_of_le hle with hlt | rfl
  · obtain ⟨k, rfl⟩ := K.same_slot h hlt
    exact causeBase_trans (K.chain k q hq').2 (K.raw _ hq' c hc)
  · exact K.raw q hq' c hc

theorem ppcb_W_W {q q' : Nat} (hq' : q' < K.N) (h : q % K.S = q' % K.S) (hlt : q < q') :
    x.proxyPreservedCauseBase (K.W q) (K.W q') = true := by
  have hq : q < K.N := by omega
  obtain ⟨k, rfl⟩ := K.same_slot h hlt
  have hS : K.S ≤ q + (k + 1) * K.S := by
    have : (k + 1) * K.S = k * K.S + K.S := Nat.succ_mul k K.S; omega
  obtain ⟨hf, hfb, hwf⟩ := K.fence q hq
  obtain ⟨hg, hgb, hgw⟩ := K.reuse _ hq' hS
  exact ppcb_async_both (K.mems_W hq) (K.mems_W hq') hf hfb hwf (K.chain k q hq').1 hg hgb hgw
    (K.alias_of (K.write q hq).2 (K.write _ hq').2 (K.mems_W hq) (K.mems_W hq') h)

theorem ppcb_W_R {q q' c : Nat} (hc : c < K.C) (hq' : q' < K.N) (h : q % K.S = q' % K.S)
    (hle : q ≤ q') : x.proxyPreservedCauseBase (K.W q) (K.R q' c) = true := by
  have hq : q < K.N := by omega
  obtain ⟨hf, hfb, hwf⟩ := K.fence q hq
  exact ppcb_async_completion (K.mems_W hq) hf hfb hwf (K.copy_before_read hc h hle hq')
    (K.read q' hq' c hc).2.1
    (K.alias_of (K.write q hq).2 (K.read q' hq' c hc).2.2 (K.mems_W hq) (K.mems_R hq' hc) h)

theorem ppcb_R_W {q q' c : Nat} (hc : c < K.C) (hq' : q' < K.N) (h : q % K.S = q' % K.S)
    (hlt : q < q') : x.proxyPreservedCauseBase (K.R q c) (K.W q') = true := by
  have hq : q < K.N := by omega
  have hS : K.S ≤ q' := by
    obtain ⟨k, rfl⟩ := K.same_slot h hlt
    have : (k + 1) * K.S = k * K.S + K.S := Nat.succ_mul k K.S; omega
  obtain ⟨hg, hgb, hgw⟩ := K.reuse q' hq' hS
  exact ppcb_async_issue (K.read q hq c hc).2.1 (K.read_before_reuse hc h hlt hq') (K.mems_W hq')
    hg hgb hgw
    (K.alias_of (K.read q hq c hc).2.2 (K.write q' hq').2 (K.mems_R hq hc) (K.mems_W hq') h)

/-- The ring's accesses: its copies' writes and its consumers' reads. -/
def Access (e : Fin n) : Prop := (∃ q < K.N, e = K.W q) ∨ ∃ q < K.N, ∃ c < K.C, e = K.R q c

theorem not_reads_writes {e : Fin n} (hr : x.reads e = true) : x.writes e = false := by
  cases hw : x.writes e
  · rfl
  · exact absurd rfl (ne_of_reads_writes hr hw)

/-- **The ring is race-free**: any two conflicting accesses of the ring are ordered by
proxy-preserved base causality order. -/
theorem race_free {a b : Fin n} (ha : K.Access a) (hb : K.Access b) (hab : x.conflict a b) :
    x.proxyPreservedCauseBase a b = true ∨ x.proxyPreservedCauseBase b a = true := by
  obtain ⟨hne, hl, hw⟩ := hab
  rcases ha with ⟨q, hq, rfl⟩ | ⟨q, hq, c, hc, rfl⟩ <;> rcases hb with ⟨q', hq', rfl⟩ | ⟨q', hq', c', hc', rfl⟩
  · have h := K.slot_eq hq hq' (K.write q hq).2 (K.write q' hq').2 hl
    rcases Nat.lt_trichotomy q q' with c | rfl | c
    · exact .inl (K.ppcb_W_W hq' h c)
    · exact absurd rfl hne
    · exact .inr (K.ppcb_W_W hq h.symm c)
  · have h := K.slot_eq hq hq' (K.write q hq).2 (K.read q' hq' c' hc').2.2 hl
    rcases Nat.lt_or_ge q' q with c | c
    · exact .inr (K.ppcb_R_W hc' hq h.symm c)
    · exact .inl (K.ppcb_W_R hc' hq' h c)
  · have h := K.slot_eq hq hq' (K.read q hq c hc).2.2 (K.write q' hq').2 hl
    rcases Nat.lt_or_ge q q' with c' | c'
    · exact .inl (K.ppcb_R_W hc hq' h c')
    · exact .inr (K.ppcb_W_R hc hq h.symm c')
  · exfalso
    rw [not_reads_writes (K.read q hq c hc).1, not_reads_writes (K.read q' hq' c' hc').1] at hw
    simp at hw

/-- **Every consumer read of a stage reads that stage's copy**, when the copies are the only writes
to the slots. -/
theorem reads_stage (hcons : x.consistent = true)
    (only : ∀ q < K.N, ∀ c < K.C, ∀ e, x.writes e = true → x.sameLocation e (K.R q c) = true →
      ∃ q' < K.N, e = K.W q') {q c : Nat} (hq : q < K.N) (hc : c < K.C) :
    x.rf (K.W q) (K.R q c) = true := by
  have hr := (K.read q hq c hc).1
  have hWR := K.ppcb_W_R hc hq rfl (Nat.le_refl q)
  rcases rf_cases hcons hr with ⟨hn, -⟩ | ⟨w, hrf, -, hw, hloc⟩
  · -- Reading the initial value would come before the copy in communication order.
    exfalso
    have hfr : x.fr (K.R q c) (K.W q) = true := by
      simp only [fr, get_union, Bool.or_eq_true, get_ofFn, Bool.and_eq_true, Bool.not_eq_true',
        List.any_eq_false, List.mem_finRange, true_implies]
      refine .inr ⟨⟨⟨hr, (K.write q hq).1⟩, sameLocation_symm (ppcb_sameLocation hWR)⟩,
        fun w => by simp [hn w]⟩
    exact not_com_of_ppcb hcons hWR (by simp [com, hfr])
  · have hlw : x.sameLocation w (K.R q c) = true := (rf_parts hcons hrf).2.2
    obtain ⟨q', hq', rfl⟩ := only q hq c hc w hw hlw
    have h := K.slot_eq hq' hq (K.write q' hq').2 (K.read q hq c hc).2.2 hlw
    rcases Nat.lt_trichotomy q' q with c' | rfl | c'
    · -- An earlier round's copy is overwritten by this round's before the read.
      exfalso
      have hco := co_of_ppcb hcons (K.write q' hq').1 (K.write q hq).1
        (sameLocation_trans hlw (sameLocation_symm (ppcb_sameLocation hWR))) (K.ppcb_W_W hq h c')
      have hfr : x.fr (K.R q c) (K.W q) = true := by
        simp only [fr, get_union, Bool.or_eq_true]
        exact .inl (comp_of (by simp [hrf]) (tc_of hco))
      exact not_com_of_ppcb hcons hWR (by simp [com, hfr])
    · exact hrf
    · -- A later round's copy comes after the read.
      exact absurd (by simp [com, hrf]) fun hcom => not_com_of_ppcb hcons (K.ppcb_R_W hc hq' h.symm c') hcom

end Ring

end Examples.Ring.Race
