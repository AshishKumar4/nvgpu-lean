import NV.PTX.Memory.DRF

/-!
# Locations updated by read-modify-writes

A synchronization object such as an mbarrier is a location whose every access is morally strong
with every other (`OneLoc`: one proxy, scopes covering the accessing threads, one address), and
whose writes after the first are read-modify-writes. In a consistent execution, coherence order
totally orders its writes, and:

* `rf_co`: a read that precedes a write in proxy-preserved base causality reads only writes
  coherence-before it (coherence, causality);
* `rmw_reads_pred`: the read of a read-modify-write reads the write just before its own in
  coherence order (`rmw_reads_prev`), or no write if its own is the first; so the location's
  values follow coherence order, operation by operation (`co_induction`);
* `sync_of_rmws`: a release write synchronizes with an acquire read of any write after it that
  only read-modify-writes separate from it (the release sequence, §8.9.2).
-/

namespace NV.PTX.Execution

open Rel

variable {n : Nat} {V : Type} [DecidableEq V] {x : Execution n V}

/-- The location initialized by write `o`: every access other than `o` is morally strong with every
other, every access uses `o`'s address, and `o` comes first in coherence order. The initialization
may be weak, as `mbarrier.init` is (PTX ISA §9.7.15.16.2: it takes no `.sem`, and a
`fence.mbarrier_init` or the barrier's own synchronization publishes it). -/
structure OneLoc (x : Execution n V) (o : Fin n) : Prop where
  strong : ∀ a b, x.sameLocation a o = true → x.sameLocation b o = true →
    (x.ev a).scope ≠ .thread → (x.ev b).scope ≠ .thread → x.strongR a b = true
  nonweak : ∀ w, x.writes w = true → x.sameLocation w o = true → w ≠ o → (x.ev w).scope ≠ .thread
  addr : ∀ a, x.sameLocation a o = true → (x.ev a).addr = (x.ev o).addr
  write : x.writes o = true
  first : ∀ v, x.writes v = true → x.sameLocation v o = true → v ≠ o → x.co o v = true

/-- The number of writes coherence-before `v`. -/
def rank (x : Execution n V) (v : Fin n) : Nat := (List.finRange n).countP fun u => x.co u v

theorem countP_lt {α : Type} {p q : α → Bool} :
    ∀ {l : List α}, (∀ a ∈ l, p a = true → q a = true) → (∃ a ∈ l, q a = true ∧ p a = false) →
      l.countP p < l.countP q
  | [], _, ⟨_, h, _⟩ => absurd h (List.not_mem_nil)
  | a :: l, h, ⟨b, hb, hqb, hpb⟩ => by
    simp only [List.countP_cons]
    have hl : ∀ c ∈ l, p c = true → q c = true := fun c hc => h c (List.mem_cons_of_mem _ hc)
    rcases List.mem_cons.1 hb with rfl | hb
    · rw [hpb, hqb]; simpa using Nat.lt_succ_of_le (List.countP_mono_left hl)
    · have := countP_lt hl ⟨b, hb, hqb, hpb⟩
      cases hp : p a <;> cases hq : q a <;> simp_all <;> omega

variable (hc : x.consistent = true)
include hc

theorem rank_lt {u v : Fin n} (h : x.co u v = true) : x.rank u < x.rank v :=
  countP_lt (fun _ _ ha => co_trans hc ha h)
    ⟨u, List.mem_finRange u, h, co_irrefl hc u⟩

/-- **Coherence order is well-founded**: a property of every write that holds once it holds of
every write coherence-before, holds of every write. -/
theorem co_induction {P : Fin n → Prop} (h : ∀ v, (∀ u, x.co u v = true → P u) → P v) (v : Fin n) :
    P v := by
  induction hr : x.rank v using Nat.strongRecOn generalizing v with
  | ind m ih => exact h v fun u hu => ih _ (hr ▸ rank_lt hc hu) u rfl

/-- Coherence orders a write that an observed write causally precedes. -/
theorem co_of_cause {u v : Fin n} (hu : x.writes u = true) (hv : x.writes v = true)
    (hl : x.sameLocation u v = true) (h : x.cause u v = true) : x.co u v = true := by
  have hcoh := (consistent_parts hc).2.2.2.1
  simp only [coherence, subset_iff] at hcoh
  exact co_of_tc hc (hcoh u v (by
    simp only [get_inter, Bool.and_eq_true]
    exact ⟨comp_of (b := v) (comp_of (b := u) (by simp [hu]) h) (by simp [hv]), hl⟩))


/-- A read returns the value of the write it reads from. -/
theorem rf_val {w r : Fin n} (h : x.rf w r = true) : x.val r = x.val w := by
  have hw := (wellFormed_parts (consistent_parts hc).1).2.1
  simp only [rfWF, Bool.and_eq_true, List.all_eq_true, List.mem_finRange, true_implies] at hw
  have h₂ := hw.2 r
  simp only [(rf_parts hc h).2.1, Bool.not_true, Bool.false_or] at h₂
  have hmem : w ∈ (List.finRange n).filter (fun w => x.rf w r) := by simp [h]
  split at h₂
  · rename_i heq; rw [heq] at hmem; cases hmem
  · rename_i w' heq
    rw [heq] at hmem
    simp only [List.mem_singleton] at hmem
    subst hmem
    simpa using h₂
  · cases h₂

variable {o : Fin n} (L : OneLoc x o)
include L

theorem co_total_loc {u v : Fin n} (hne : u ≠ v) (hu : x.writes u = true) (hv : x.writes v = true)
    (hul : x.sameLocation u o = true) (hvl : x.sameLocation v o = true) :
    x.co u v = true ∨ x.co v u = true := by
  by_cases huo : u = o
  · subst huo; exact .inl (L.first v hv hvl (Ne.symm hne))
  by_cases hvo : v = o
  · subst hvo; exact .inr (L.first u hu hul hne)
  exact co_total hc hne hu hv ((L.addr u hul).trans (L.addr v hvl).symm) (L.strong u v hul hvl (L.nonweak u hu hul huo) (L.nonweak v hv hvl hvo))

/-- A write coherence-after another is not the initialization. -/
theorem ne_first {u v : Fin n} (h : x.co u v = true) : v ≠ o := by
  intro he
  obtain ⟨hu, -, hul⟩ := co_parts hc h
  by_cases huo : u = o
  · rw [huo, he, co_irrefl hc] at h; cases h
  · have h₁ := L.first u hu (he ▸ hul) huo
    rw [he] at h
    have := co_trans hc h h₁
    rw [co_irrefl hc] at this; cases this

/-- **A read before a write reads only writes coherence-before it.** -/
theorem rf_co {u r v : Fin n} (hrf : x.rf u r = true) (hrl : x.sameLocation r o = true)
    (hrs : (x.ev r).scope ≠ .thread) (hrv : x.proxyPreservedCauseBase r v = true)
    (hv : x.writes v = true) (hvl : x.sameLocation v o = true) : x.co u v = true := by
  obtain ⟨hu, hr, hur⟩ := rf_parts hc hrf
  have hul := sameLocation_trans hur hrl
  by_cases huv : u = v
  · subst huv; exact absurd (by simp [com, hrf]) fun h => not_com_of_ppcb hc hrv h
  by_cases huo : u = o
  · subst huo; exact L.first v hv hvl (Ne.symm huv)
  rcases co_total_loc hc L huv hu hv hul hvl with h | h
  · exact h
  · have hcause : x.cause u v = true :=
      comp_of (b := r) (by simp [observation, strong, hrf, L.strong u r hul hrl (L.nonweak u hu hul huo)
        hrs]) hrv
    have := co_trans hc (co_of_cause hc hu hv (sameLocation_trans hul (sameLocation_symm hvl)) hcause) h
    rw [co_irrefl hc] at this; cases this

/-- Among the writes coherence-before a write of the location, if any, one comes last. -/
theorem exists_pred {v : Fin n} (hvl : x.sameLocation v o = true)
    {u : Fin n} (hu : x.co u v = true) :
    ∃ p, x.co p v = true ∧ ∀ u', x.co u' v = true → u' = p ∨ x.co u' p = true := by
  -- The coherence-predecessor of greatest rank.
  have key : ∀ l : List (Fin n), (∃ a ∈ l, x.co a v = true) →
      ∃ p ∈ l, x.co p v = true ∧ ∀ a ∈ l, x.co a v = true → x.rank a ≤ x.rank p := by
    intro l
    induction l with
    | nil => intro ⟨_, h, _⟩; cases h
    | cons a l ih =>
      intro ⟨b, hb, hbv⟩
      by_cases hl : ∃ c ∈ l, x.co c v = true
      · obtain ⟨p, hp, hpv, hmax⟩ := ih hl
        by_cases ha : x.co a v = true ∧ x.rank p < x.rank a
        · exact ⟨a, by simp, ha.1, fun c hc hcv => by
            rcases List.mem_cons.1 hc with rfl | hc
            · exact Nat.le_refl _
            · exact Nat.le_of_lt (Nat.lt_of_le_of_lt (hmax c hc hcv) ha.2)⟩
        · exact ⟨p, List.mem_cons_of_mem _ hp, hpv, fun c hc hcv => by
            rcases List.mem_cons.1 hc with rfl | hc
            · exact Nat.le_of_not_lt fun h => ha ⟨hcv, h⟩
            · exact hmax c hc hcv⟩
      · have hb' : b = a := by
          rcases List.mem_cons.1 hb with rfl | hb
          · rfl
          · exact absurd ⟨b, hb, hbv⟩ hl
        subst hb'
        exact ⟨b, by simp, hbv, fun c hc hcv => by
          rcases List.mem_cons.1 hc with rfl | hc
          · exact Nat.le_refl _
          · exact absurd ⟨c, hc, hcv⟩ hl⟩
  obtain ⟨p, -, hpv, hmax⟩ := key (List.finRange n) ⟨u, List.mem_finRange u, hu⟩
  refine ⟨p, hpv, fun u' hu' => ?_⟩
  obtain ⟨hu'w, -, hu'l⟩ := co_parts hc hu'
  obtain ⟨hpw, -, hpl⟩ := co_parts hc hpv
  by_cases he : u' = p
  · exact .inl he
  rcases co_total_loc hc L he hu'w hpw (sameLocation_trans hu'l hvl) (sameLocation_trans hpl hvl)
    with h | h
  · exact .inr h
  · have := hmax u' (List.mem_finRange u') hu'
    have := rank_lt hc h
    omega

omit hc L in
/-- A read-modify-write's read reads the location of its write. -/
theorem rmw_parts (hc : x.consistent = true) {r v : Fin n} (hrmw : x.rmw r v = true) :
    x.reads r = true ∧ x.writes v = true ∧ x.sameLocation r v = true ∧
      (x.ev r).scope = (x.ev v).scope := by
  have := (wellFormed_parts (consistent_parts hc).1).2.2.2.2.1
  simp only [rmwWF, Bool.and_eq_true, subset_iff, List.all_eq_true, List.mem_finRange,
    true_implies] at this
  have hsc := this.1.2 r v
  simp only [hrmw, Bool.not_true, Bool.false_or, beq_iff_eq] at hsc
  obtain ⟨b, hb, hbw⟩ := get_comp.1 (this.1.1 r v hrmw)
  obtain ⟨a, ha, hab⟩ := get_comp.1 hb
  simp only [get_dom, Bool.and_eq_true, beq_iff_eq] at ha hbw
  obtain ⟨rfl, hr⟩ := ha
  obtain ⟨rfl, hw⟩ := hbw
  simp only [get_inter, Bool.and_eq_true, sameAlias, get_ofFn, beq_iff_eq] at hab
  exact ⟨hr, hw, by simp [sameLocation, hab.2.1.1, hab.2.1.2, hab.2.2], hsc⟩

/-- **A read-modify-write reads the write just before its own** in coherence order, when that write
is not the initialization: reading anything earlier breaks atomicity (§8.10.3), reading its own
write or a later one causality (§8.10.6) or coherence (§8.10.1). -/
theorem rmw_reads_pred {r v p : Fin n} (hrmw : x.rmw r v = true)
    (hrv : x.proxyPreservedCauseBase r v = true) (hvl : x.sameLocation v o = true)
    (hpv : x.co p v = true) (hlast : ∀ u', x.co u' v = true → u' = p ∨ x.co u' p = true)
    (hpo : p ≠ o) : x.rf p r = true := by
  have hat := (consistent_parts hc).2.2.1
  simp only [atomicity, isEmpty_iff] at hat
  obtain ⟨hp, hv, hpvl⟩ := co_parts hc hpv
  have hpl := sameLocation_trans hpvl hvl
  have hvo : v ≠ o := ne_first hc L hpv
  obtain ⟨hr, -, hrv', hrsc⟩ := rmw_parts hc hrmw
  have hrl := sameLocation_trans hrv' hvl
  have hrs : (x.ev r).scope ≠ .thread := hrsc ▸ L.nonweak v hv hvl hvo
  have notFr : x.fr r p = false := by
    cases hfr : x.fr r p
    · rfl
    · have h := hat r v
      rw [get_inter, hrmw, Bool.and_true,
        comp_of (b := p) (by simp [strong, hfr, L.strong r p hrl hpl hrs (L.nonweak p hp hpl hpo)])
          (by simp [strong, hpv, L.strong p v hpl hvl (L.nonweak p hp hpl hpo) (L.nonweak v hv hvl hvo)])] at h
      cases h
  rcases rf_cases hc hr with ⟨hnone, -⟩ | ⟨u, hrf, -, hu, -⟩
  · exfalso
    have : x.fr r p = true := by
      simp only [fr, get_union, Bool.or_eq_true, get_ofFn, Bool.and_eq_true, Bool.not_eq_true',
        List.any_eq_false, List.mem_finRange, true_implies]
      exact .inr ⟨⟨⟨hr, hp⟩, sameLocation_trans hrl (sameLocation_symm hpl)⟩, fun w => by simp [hnone w]⟩
    rw [notFr] at this; cases this
  by_cases hup : u = p
  · exact hup ▸ hrf
  exfalso
  have hul : x.sameLocation u o = true := sameLocation_trans (rf_parts hc hrf).2.2 hrl
  rcases co_total_loc hc L hup hu hp hul hpl with h | h
  · have : x.fr r p = true := by
      simp only [fr, get_union, Bool.or_eq_true]
      exact .inl (comp_of (by simp [hrf]) (tc_of h))
    rw [notFr] at this; cases this
  · have huo : u ≠ o := ne_first hc L h
    by_cases huv : u = v
    · subst huv; exact not_com_of_ppcb hc hrv (by simp [com, hrf])
    rcases co_total_loc hc L huv hu hv hul hvl with h' | h'
    · rcases hlast u h' with rfl | h''
      · exact hup rfl
      · have := co_trans hc h h''; rw [co_irrefl hc] at this; cases this
    · have hcause : x.cause u v = true :=
        comp_of (b := r) (by simp [observation, strong, hrf, L.strong u r hul hrl (L.nonweak u hu hul huo)
          hrs]) hrv
      have := co_trans hc (co_of_cause hc hu hv (sameLocation_trans hul (sameLocation_symm hvl)) hcause) h'
      rw [co_irrefl hc] at this; cases this

/-- The read of a read-modify-write whose write comes first after the initialization reads the
initialization or the location's initial value. -/
theorem rmw_reads_init {r v : Fin n} (hrmw : x.rmw r v = true)
    (hrv : x.proxyPreservedCauseBase r v = true) (hvl : x.sameLocation v o = true) (hvo : v ≠ o)
    (hlast : ∀ u', x.co u' v = true → u' = o ∨ x.co u' o = true) :
    x.rf o r = true ∨ ∀ u, x.rf u r = false := by
  obtain ⟨hr, hv, hrv', hrsc⟩ := rmw_parts hc hrmw
  have hrl := sameLocation_trans hrv' hvl
  rcases rf_cases hc hr with ⟨hn, -⟩ | ⟨u, hrf, -, -, -⟩
  · exact .inr hn
  · by_cases huo : u = o
    · exact .inl (huo ▸ hrf)
    exfalso
    rcases hlast u (rf_co hc L hrf hrl (hrsc ▸ L.nonweak v hv hvl hvo) hrv hv hvl) with h | h
    · exact huo h
    · exact ne_first hc L h rfl

/-- **What a read-modify-write's read returns**: the value of the write before its own, or, when
that write is the initialization, possibly the location's initial value. -/
theorem rmw_src {r v p : Fin n} (hrmw : x.rmw r v = true)
    (hrv : x.proxyPreservedCauseBase r v = true) (hvl : x.sameLocation v o = true)
    (hpv : x.co p v = true) (hlast : ∀ u', x.co u' v = true → u' = p ∨ x.co u' p = true) :
    x.val r = x.val p ∨ (p = o ∧ x.val r = x.init (x.loc (x.ev o).addr)) := by
  by_cases hpo : p = o
  · subst hpo
    rcases rmw_reads_init hc L hrmw hrv hvl (ne_first hc L hpv) hlast with h | h
    · exact .inl (rf_val hc h)
    · obtain ⟨hr, -, hrv', -⟩ := rmw_parts hc hrmw
      rcases rf_cases hc hr with ⟨-, hv⟩ | ⟨u, hu, -⟩
      · exact .inr ⟨rfl, hv.trans (by rw [(sameLocation_iff.1 (sameLocation_trans hrv' hvl)).2.2])⟩
      · rw [h u] at hu; cases hu
  · exact .inl (rf_val hc (rmw_reads_pred hc L hrmw hrv hvl hpv hlast hpo))

/-- **The release sequence**: a release write `u` of the location synchronizes with an acquire
read of any write `v` at or after it, when every write after `u` up to `v` is a read-modify-write
whose read precedes its write in proxy-preserved base causality. -/
theorem sync_of_rmws {u v R : Fin n} (hu : x.writes u = true) (hul : x.sameLocation u o = true)
    (huo : u ≠ o) (hrel : (x.ev u).release = true) (hrf : x.rf v R = true)
    (hRl : x.sameLocation R o = true) (hRs : (x.ev R).scope ≠ .thread)
    (hacq : (x.ev R).acquire = true) (hnr : (x.ev R).reduction = false)
    (hle : u = v ∨ x.co u v = true)
    (hrmw : ∀ w, x.co u w = true → (w = v ∨ x.co w v = true) →
      ∃ r, x.rmw r w = true ∧ x.proxyPreservedCauseBase r w = true ∧ x.sameLocation r o = true) :
    x.sync u R = true := by
  obtain ⟨-, hR, -⟩ := rf_parts hc hrf
  refine sync_of_observation hrel hu hacq hR hnr ?_ (L.strong u R hul hRl (L.nonweak u hu hul huo) hRs)
  -- Every read of a write at or after `u` (up to `v`) is reached from `u` in observation order.
  suffices h : ∀ v', (u = v' ∨ x.co u v' = true) → (v' = v ∨ x.co v' v = true) →
      ∀ R', x.rf v' R' = true → x.sameLocation R' o = true → (x.ev R').scope ≠ .thread →
        tc x.observation u R' = true from
    h v hle (.inl rfl) R hrf hRl hRs
  intro v'
  induction v' using co_induction hc with
  | h v' ih =>
  intro hle' hv' R' hrf' hR'l hR's
  have hv'l : x.sameLocation v' o = true :=
    sameLocation_trans (rf_parts hc hrf').2.2 hR'l
  have hv'o : v' ≠ o := by rcases hle' with rfl | h; exact huo; exact ne_first hc L h
  have hobs : x.observation v' R' = true := by
    simp [observation, strong, hrf', L.strong v' R' hv'l hR'l
      (L.nonweak v' (rf_parts hc hrf').1 hv'l hv'o) hR's]
  rcases hle' with rfl | hco
  · exact tc_of hobs
  obtain ⟨r, hrmw', hrv', hrl⟩ := hrmw v' hco hv'
  obtain ⟨p, hpv, hlast⟩ := exists_pred hc L hv'l hco
  have hup : u = p ∨ x.co u p = true := by
    rcases hlast u hco with h | h
    · exact .inl h
    · exact .inr h
  have hrp := rmw_reads_pred hc L hrmw' hrv' hv'l hpv hlast
    (by rcases hup with rfl | h; exact huo; exact ne_first hc L h)
  have hpv' : p = v ∨ x.co p v = true := by
    rcases hv' with rfl | h
    · exact .inr hpv
    · exact .inr (co_trans hc hpv h)
  have h₁ := ih p hpv hup hpv' r hrp hrl
    ((rmw_parts hc hrmw').2.2.2 ▸ L.nonweak v' (rf_parts hc hrf').1 hv'l hv'o)
  exact tc_trans (tc_trans h₁ (tc_of (by simp [observation, hrmw']))) (tc_of hobs)

end NV.PTX.Execution
