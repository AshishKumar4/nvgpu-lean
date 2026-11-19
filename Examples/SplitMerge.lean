import NV.PTX.Memory.Visibility

/-!
# The split-KV attention merge

The merge of a split-KV decode attention: each of `K` split CTAs reduces its partial softmax sums
into workspace copies with `red.global.add.f32`, synchronizes the CTA, and its thread 0 performs
`atom.acq_rel.gpu.global.add.s32 old, [counter], mine` with `mine` its key count. The split whose
`old + mine` equals the total key count is the last; it reads every copy and finalizes.

`merge_sees`, for executions of any size, any `K`, any key partition with positive parts and any
threads: in a consistent execution, every read the last split makes of a workspace location reads
a write, and not one coherence-before any split's partial write to that location. The visibility
comes from Causality (§8.10.6) through the counter: each split's release reaches the last split's
acquire through the observation order of the atomic chain (`counter_observes`, §8.9.2).

`Chain` states what the program guarantees about a location updated only by `K` atomic adds
(each field is what `atom.gpu.global.add` or `red.gpu.global.add` is, §9.7.13.3 and §8.4), with
values read through `dec` and added modulo `M` (`M = 2³²` for `.u32`); `chain`: each update reads
the sum of the updates coherence-before it. `Counter` adds the acquire and release of
`atom.acq_rel`; `published` is how a split's partial write comes before its counter update (a CTA
barrier, §8.9.4, then program order).
-/

namespace Examples.SplitMerge

open NV NV.Formal

open NV.PTX NV.PTX.Execution Rel

variable {n : Nat} {V : Type} [DecidableEq V] {x : Execution n V}

/-- A location updated only by `K` atomic adds: update `j < K` is the read `rd j` and the write
`wr j` of one read-modify-write adding `mine j` modulo `M` to the value `dec` reads. -/
structure Chain (x : Execution n V) (K : Nat) (dec : V → Nat) (M : Nat) where
  rd : Nat → Fin n
  wr : Nat → Fin n
  mine : Nat → Nat
  addr : Nat
  rmw : ∀ j < K, x.rmw (rd j) (wr j) = true
  reads : ∀ j < K, x.reads (rd j) = true
  writes : ∀ j < K, x.writes (wr j) = true
  address : ∀ j < K, (x.ev (rd j)).addr = addr ∧ (x.ev (wr j)).addr = addr
  generic : ∀ j < K, x.generic (rd j) = true ∧ x.generic (wr j) = true
  /-- `.gpu` atomics of one grid are morally strong with each other (§8.7). -/
  strong : ∀ j < K, ∀ l < K, x.strongR (rd j) (wr l) = true ∧ x.strongR (wr j) (wr l) = true ∧
    x.strongR (wr l) (rd j) = true
  distinct : ∀ j < K, ∀ l < K, wr j = wr l → j = l
  /-- Nothing else writes the counter's location. -/
  only : ∀ e, x.writes e = true → x.loc (x.ev e).addr = x.loc addr → ∃ j < K, e = wr j
  /-- The location starts at zero. -/
  zero : dec (x.init (x.loc addr)) = 0
  add : ∀ j < K, dec (x.val (wr j)) = (dec (x.val (rd j)) + mine j) % M

/-- The counter of the merge: a chain of `atom.acq_rel` adds of positive key counts. -/
structure Counter (x : Execution n V) (K : Nat) (dec : V → Nat) (M : Nat)
    extends Chain x K dec M where
  acquire : ∀ j < K, (x.ev (rd j)).acquire = true ∧ (x.ev (rd j)).reduction = false
  release : ∀ j < K, (x.ev (wr j)).release = true
  pos : ∀ j < K, 0 < mine j

section
variable {K : Nat} {dec : V → Nat} {M : Nat} (hc : x.consistent = true) (C : Chain x K dec M)
include hc C

/-- Splits whose counter update coherence-precedes split `j`'s. -/
def before (j l : Nat) : Bool := x.co (C.wr l) (C.wr j)

theorem rmw_order {j : Nat} (hj : j < K) : tc x.order (C.rd j) (C.wr j) = true := by
  have := (wellFormed_parts (consistent_parts hc).1).2.2.2.2.1
  simp only [rmwWF, Bool.and_eq_true, subset_iff] at this
  obtain ⟨b, hb, _⟩ := get_comp.1 (this.1.1 _ _ (C.rmw j hj))
  obtain ⟨a, ha, hab⟩ := get_comp.1 hb
  simp only [get_dom, Bool.and_eq_true, beq_iff_eq] at ha
  obtain ⟨rfl, _⟩ := ha
  simp only [get_inter, Bool.and_eq_true] at hab
  have hb' : b = C.wr j := by
    obtain ⟨b', hb'⟩ := get_comp.1 (this.1.1 _ _ (C.rmw j hj))
    simp_all
  subst hb'
  exact tc_of (by simp [order, hab.1.1])

theorem atomic {j l : Nat} (hj : j < K) (hl : l < K) (hfr : x.fr (C.rd j) (C.wr l) = true)
    (hco : x.co (C.wr l) (C.wr j) = true) : False := by
  have := (consistent_parts hc).2.2.1
  simp only [atomicity, isEmpty_iff] at this
  have h := this (C.rd j) (C.wr j)
  simp only [get_inter, Bool.and_eq_false_iff] at h
  rcases h with h | h
  · rw [comp_of (b := C.wr l) (by simp [strong, hfr, (C.strong j hj l hl).1])
      (by simp [strong, hco, (C.strong l hl j hj).2.1])] at h
    cases h
  · rw [C.rmw j hj] at h; cases h

/-- Along the coherence order of the counter's writes, each update reads the sum of the updates
before it and observes every one of them. -/
theorem chain : ∀ m j, j < K → countBelow K (before C j) = m →
    dec (x.val (C.rd j)) = sumBelow K (fun l => if before C j l then C.mine l else 0) % M ∧
      ∀ l < K, before C j l = true → tc x.observation (C.wr l) (C.rd j) = true := by
  intro m
  induction m using Nat.strongRecOn with
  | ind m ih =>
  intro j hj hm
  have hloc : ∀ l < K, x.loc (x.ev (C.wr l)).addr = x.loc (x.ev (C.rd j)).addr := fun l hl => by
    rw [(C.address l hl).2, (C.address j hj).1]
  rcases rf_cases hc (C.reads j hj) with ⟨hnone, hval⟩ | ⟨w, hrf, hval, hw, hwloc⟩
  · -- Reads the initial value: no update can precede it.
    have hnb : ∀ l < K, before C j l = false := by
      intro l hl
      cases hb : before C j l
      · rfl
      · exfalso
        apply atomic hc C hj hl _ hb
        have hmr : x.mems (C.rd j) = true := by
          simp only [mems, Event.isMem, Bool.or_eq_true]; left; exact C.reads j hj
        have hmw : x.mems (C.wr l) = true := by
          simp only [mems, Event.isMem, Bool.or_eq_true]; right; exact C.writes l hl
        simp only [fr, get_union, Bool.or_eq_true, get_ofFn, Bool.and_eq_true, Bool.not_eq_true',
          List.any_eq_false, List.mem_finRange, true_implies]
        exact .inr ⟨⟨⟨C.reads j hj, C.writes l hl⟩, by simp [sameLocation, hmr, hmw, hloc l hl]⟩,
          fun w => by simp [hnone w]⟩
    refine ⟨?_, fun l hl hb => by rw [hnb l hl] at hb; cases hb⟩
    rw [hval, (C.address j hj).1, C.zero]
    rw [sumBelow_congr (g := fun _ => 0) (fun l hl => by simp [hnb l hl])]
    simp [(sumBelow_zero_iff _ _).2 (fun _ _ => rfl)]
  · -- Reads some update `p`: the one immediately before it.
    obtain ⟨p, hp, rfl⟩ := C.only w hw (by rw [hwloc, (C.address j hj).1])
    have hstrong : x.strong x.rf (C.wr p) (C.rd j) = true := by
      simp [strong, hrf, (C.strong j hj p hp).2.2]
    have hobs : x.observation (C.wr p) (C.rd j) = true := by simp [observation, hstrong]
    have hgen := C.generic
    have hpj : before C j p = true := by
      have hcoh := (consistent_parts hc).2.2.2.1
      simp only [coherence, subset_iff] at hcoh
      have hcause : x.cause (C.wr p) (C.wr j) = true := by
        exact comp_of (b := C.rd j) (by simp [hobs]) (ppcb_of_generic (hgen j hj).1 (hgen j hj).2
          (by rw [(C.address j hj).1, (C.address j hj).2]) (causeBase_of_order (rmw_order hc C hj)))
      have := hcoh (C.wr p) (C.wr j) (by
        simp only [get_inter, Bool.and_eq_true]
        refine ⟨comp_of (comp_of (by simp [C.writes p hp]) hcause) (by simp [C.writes j hj]), ?_⟩
        have hmw : ∀ l < K, x.mems (C.wr l) = true := fun l hl => by
          simp only [mems, Event.isMem, Bool.or_eq_true]; right; exact C.writes l hl
        simp [sameLocation, hmw p hp, hmw j hj, (C.address p hp).2, (C.address j hj).2])
      exact co_of_tc hc this
    -- Nothing lies strictly between `p` and `j`.
    have hgap : ∀ l < K, x.co (C.wr p) (C.wr l) = true → x.co (C.wr l) (C.wr j) = true → False :=
      fun l hl h₁ h₂ => atomic hc C hj hl (by
        simp only [fr, get_union, Bool.or_eq_true]
        exact .inl (comp_of (by simp [hrf]) (tc_of h₁))) h₂
    have hsplit : ∀ l < K, l ≠ p → before C j l = before C p l := by
      intro l hl hlp
      cases hb : before C p l
      · cases hb' : before C j l
        · rfl
        · exfalso
          have hne : C.wr l ≠ C.wr p := fun e => hlp (C.distinct l hl p hp e)
          rcases co_total hc hne (C.writes l hl) (C.writes p hp)
            (by rw [(C.address l hl).2, (C.address p hp).2]) (C.strong l hl p hp).2.1 with h | h
          · simp [before, h] at hb
          · exact hgap l hl h hb'
      · exact co_trans hc hb hpj
    have hpp : before C p p = false := co_irrefl hc _
    have hcount : countBelow K (before C j) = countBelow K (before C p) + 1 :=
      countBelow_bump hp (fun l hl hlp => (hsplit l hl hlp).symm) hpp hpj
    obtain ⟨ihv, iho⟩ := ih (countBelow K (before C p)) (by omega) p hp rfl
    constructor
    · rw [hval, C.add p hp, ihv]
      have := sumBelow_update (n := K) hp
        (f := fun l => if before C p l then C.mine l else 0)
        (g := fun l => if before C j l then C.mine l else 0)
        (fun l hl hlp => by simp [hsplit l hl hlp])
      simp only [hpp, hpj, Bool.false_eq_true, ite_false, ite_true] at this
      rw [Nat.mod_add_mod, ← this, Nat.add_zero]
    · intro l hl hb
      by_cases hlp : l = p
      · subst hlp; exact tc_of hobs
      · rw [hsplit l hl hlp] at hb
        exact tc_trans (iho l hl hb) (tc_trans (tc_of (by simp [observation, C.rmw p hp]))
          (tc_of hobs))

omit hc C in
theorem counter_observes {C : Counter x K dec M} (hc : x.consistent = true) {m : Nat} (hm : m < K)
    (hbound : sumBelow K C.mine < M)
    (hlast : dec (x.val (C.rd m)) + C.mine m = sumBelow K C.mine) :
    ∀ l < K, l ≠ m → tc x.observation (C.wr l) (C.rd m) = true := by
  obtain ⟨hv, ho⟩ := chain hc C.toChain _ m hm rfl
  intro l hl hlm
  cases hb : before C.toChain m l
  · exfalso
    have hmm : before C.toChain m m = false := co_irrefl hc _
    -- Leaving out `l` and `m` would make the sum short by `mine l > 0`.
    have hle : sumBelow K (fun i => if before C.toChain m i then C.mine i else 0) ≤
        sumBelow K (fun i => if i = l ∨ i = m then 0 else C.mine i) :=
      sumBelow_le fun i _ => by
        by_cases hil : i = l
        · subst hil; simp [hb]
        · by_cases him : i = m
          · subst him; simp [hmm]
          · simp only [hil, him, or_self, ite_false]; split <;> omega
    have e₁ := sumBelow_update (n := K) hm (f := C.mine) (g := fun i => if i = m then 0 else C.mine i)
      (fun i _ hi => by simp [hi])
    have e₂ := sumBelow_update (n := K) hl (f := fun i => if i = m then 0 else C.mine i)
      (g := fun i => if i = l ∨ i = m then 0 else C.mine i) (fun i _ hi => by simp [hi])
    simp only [ite_true, true_or, hlm, ite_false] at e₁ e₂
    have := C.pos l hl
    have hle' : sumBelow K (fun i => if before C.toChain m i then C.mine i else 0) ≤ sumBelow K C.mine :=
      sumBelow_le fun i _ => by split <;> omega
    rw [hv, Nat.mod_eq_of_lt (by omega)] at hlast
    omega
  · exact ho l hl hb

end

/-- **The merge sees every partial.** A read `r` the last split makes after its counter update,
of a location some split `j` wrote with partial `d` before its own update, reads a write, and not
one coherence-before `d`. -/
theorem merge_sees {K : Nat} {dec : V → Nat} {M : Nat} (hc : x.consistent = true)
    (C : Counter x K dec M) {m : Nat} (hm : m < K) (hbound : sumBelow K C.mine < M)
    (hlast : dec (x.val (C.rd m)) + C.mine m = sumBelow K C.mine) {j : Nat} (hj : j < K) {d r : Fin n}
    (hd : x.writes d = true) (hgd : x.generic d = true) (hpub : x.causeBase d (C.wr j) = true)
    (hr : x.reads r = true) (hgr : x.generic r = true) (haddr : (x.ev d).addr = (x.ev r).addr)
    (hafter : rtc x.order (C.wr m) r = true) :
    (∃ w, x.rf w r = true) ∧ ∀ w, x.rf w r = true → x.co w d = false := by
  apply read_sees hc hd hr hgd hgr haddr
  by_cases hjm : j = m
  · subst hjm; exact causeBase_order hpub hafter
  · have hsync := sync_of_observation (C.release j hj) (C.writes j hj) (C.acquire m hm).1
      (C.reads m hm) (C.acquire m hm).2 (counter_observes hc hm hbound hlast j hj hjm)
      (C.strong m hm j hj).2.2
    exact causeBase_order (causeBase_order (causeBase_sync hpub hsync)
      (rtc_of_tc (rmw_order hc C.toChain hm))) hafter

/-- A partial written before the split's CTA barrier, whose wait precedes the split's counter
update in thread 0, is published by that update. -/
theorem published {d e w0 : Fin n} {K j : Nat} {dec : V → Nat} {M : Nat} (hc : x.consistent = true)
    (C : Chain x K dec M)
    (hj : j < K) (hde : rtc x.order d e = true) (hbar : x.bar e w0 = true)
    (hw0 : rtc x.order w0 (C.rd j) = true) : x.causeBase d (C.wr j) = true := by
  have hsync : x.sync e w0 = true := by simp [sync, hbar]
  have hcb : x.causeBase d w0 = true :=
    causeBase_iff.2 (.inr ⟨e, hde, tc_of (comp_of (by simp [hsync]) (rtc_refl _ _))⟩)
  exact causeBase_order (causeBase_order hcb hw0) (rtc_of_tc (rmw_order hc C hj))

end Examples.SplitMerge
