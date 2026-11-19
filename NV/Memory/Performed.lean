import NV.Memory.Events
import NV.PTX.Memory.Timed

/-!
# Event lists in perform order

`Performed evs init` says that `evs` lists a run's events in the order they were performed, by
a machine that keeps the ISA's ordering rules: positions in a thread are distinct; a pair the
machine performed out of program order touches different locations, the later-in-program-order
one does not drain and the earlier one does not block; a read-modify-write's read and write are
performed together; a barrier's arrivals come before its waits; and each read returned the bytes
of the last write to its location performed before it, or the initial bytes.

`Performed.consistent`: the execution of such a list, with reads-from the last earlier write,
coherence the perform order of writes and Fence-SC the perform order of `fence.sc`, is consistent.
The request machine produces such lists (`NV.Memory.Sound`).
-/

namespace NV

open NV.PTX (Event Place Execution)
open NV.Rel

section
variable (evs : List Ev)

def Ev.isWriteTo (v : Ev) (k : Loc × Nat) : Bool := v.e.isWrite && v.key == k

/-- Reads-from: the last write to the read's location performed before it. -/
def rfOf : Rel evs.length := Rel.ofFn fun w r =>
  evs[w].e.isWrite && evs[r].e.isRead && evs[w].key == evs[r].key && decide (w < r) &&
    (List.finRange evs.length).all fun w' => !(decide (w < w') && decide (w' < r) &&
      evs[w'].isWriteTo evs[r].key)

/-- Coherence: the order writes to a location were performed. -/
def coOf : Rel evs.length := Rel.ofFn fun a b =>
  evs[a].e.isWrite && evs[b].e.isWrite && evs[a].key == evs[b].key && decide (a < b)

/-- Fence-SC order: the order the `fence.sc` operations executed. -/
def scOf : Rel evs.length := Rel.ofFn fun a b =>
  evs[a].e.isFenceSC && evs[b].e.isFenceSC && decide (a < b)

end

structure Performed (evs : List Ev) (init : Loc → UInt8) : Prop where
  /-- The machine performs no asynchronous operation yet: no event has an issue order. -/
  sync_only : ∀ i : Fin evs.length, evs[i].after = []
  dist : ∀ i j : Fin evs.length, i ≠ j → evs[i].e.thread = evs[j].e.thread → evs[i].seq ≠ evs[j].seq
  po : ∀ i j : Fin evs.length, i < j → evs[i].e.thread = evs[j].e.thread → evs[j].seq < evs[i].seq →
    ¬(evs[i].e.isMem = true ∧ evs[j].e.isMem = true ∧ evs[i].key = evs[j].key) ∧
      evs[i].e.drains = false ∧ evs[j].e.blocks = false
  rmw : ∀ i : Fin evs.length, evs[i].rmwRead = true → ∃ h : i.val + 1 < evs.length,
    evs[i].e.isRead = true ∧ evs[i.val + 1].e.isWrite = true ∧
    evs[i.val + 1].e.thread = evs[i].e.thread ∧ evs[i.val + 1].seq = evs[i].seq + 1 ∧
    evs[i.val + 1].key = evs[i].key ∧ evs[i.val + 1].e.scope = evs[i].e.scope
  bar : ∀ i j : Fin evs.length, evs[i].arrives = true → evs[j].waits = true →
    evs[i].obj = evs[j].obj → evs[i].inst = evs[j].inst → i < j
  val : ∀ i : Fin evs.length, evs[i].e.isRead = true →
    (∃ w : Fin evs.length, w < i ∧ evs[w].isWriteTo evs[i].key = true ∧
      (∀ w' : Fin evs.length, w < w' → w' < i → evs[w'].isWriteTo evs[i].key = false) ∧
      evs[i].val = evs[w].val) ∨
    ((∀ w : Fin evs.length, w < i → evs[w].isWriteTo evs[i].key = false) ∧
      evs[i].val = initBytes init evs[i].key)

namespace Performed

variable {evs : List Ev} {init : Loc → UInt8} (place : Nat → Place)

/-- The execution of a performed list. -/
abbrev exec (evs : List Ev) (init : Loc → UInt8) (place : Nat → Place) :
    Execution evs.length (List UInt8) :=
  toExec evs place init (rfOf evs) (coOf evs) (scOf evs)

theorem code_eq {a b : Ev} (ha : a ∈ evs) (hb : b ∈ evs) : code evs a = code evs b ↔ a.key = b.key := by
  constructor
  · intro h
    have ha' : (evs.map Ev.key).idxOf a.key < (evs.map Ev.key).length :=
      List.idxOf_lt_length_of_mem (List.mem_map_of_mem ha)
    have hb' : (evs.map Ev.key).idxOf b.key < (evs.map Ev.key).length :=
      List.idxOf_lt_length_of_mem (List.mem_map_of_mem hb)
    have e₁ := List.getElem_idxOf ha'
    have e₂ := List.getElem_idxOf hb'
    unfold code at h
    simp only [h] at e₁
    exact e₁.symm.trans e₂
  · intro h; unfold code; rw [h]

theorem mem_get (i : Fin evs.length) : evs[i] ∈ evs := List.getElem_mem _

theorem sameLocation_iff {i j : Fin evs.length} :
    (exec evs init place).sameLocation i j = true ↔
      evs[i].e.isMem = true ∧ evs[j].e.isMem = true ∧ evs[i].key = evs[j].key := by
  simp only [Execution.sameLocation, Execution.mems, exec, toExec, Rel.get_ofFn, Bool.and_eq_true,
    beq_iff_eq, id]
  rw [code_eq (mem_get i) (mem_get j)]
  exact and_assoc


variable {place}

@[simp] theorem ev_kind (i : Fin evs.length) : ((exec evs init place).ev i).kind = evs[i].e.kind := rfl
@[simp] theorem ev_thread (i : Fin evs.length) :
    ((exec evs init place).ev i).thread = evs[i].e.thread := rfl

@[simp] theorem ev_drains (i : Fin evs.length) :
    ((exec evs init place).ev i).drains = evs[i].e.drains := rfl
@[simp] theorem ev_blocks (i : Fin evs.length) :
    ((exec evs init place).ev i).blocks = evs[i].e.blocks := rfl
@[simp] theorem ev_isFenceSC (i : Fin evs.length) :
    ((exec evs init place).ev i).isFenceSC = evs[i].e.isFenceSC := rfl

theorem writes_iff (i : Fin evs.length) : (exec evs init place).writes i = evs[i].e.isWrite := rfl
theorem reads_iff (i : Fin evs.length) : (exec evs init place).reads i = evs[i].e.isRead := rfl
theorem mems_iff (i : Fin evs.length) : (exec evs init place).mems i = evs[i].e.isMem := rfl

theorem isMem_of_isWrite {v : Ev} (h : v.e.isWrite = true) : v.e.isMem = true := by
  simp [Event.isMem, h]

theorem isMem_of_isRead {v : Ev} (h : v.e.isRead = true) : v.e.isMem = true := by
  simp [Event.isMem, h]

theorem not_read_write {v : Ev} (h₁ : v.e.isRead = true) (h₂ : v.e.isWrite = true) : False := by
  simp only [Event.isRead, Event.isWrite, beq_iff_eq] at h₁ h₂; rw [h₁] at h₂; cases h₂

/-- Program order and its closure: a thread's positions in increasing order. -/
theorem order_iff (hp : Performed evs init) {a b : Fin evs.length} :
    tc (exec evs init place).order a b = true →
      evs[a].e.thread = evs[b].e.thread ∧ evs[a].seq < evs[b].seq := by
  have one : ∀ a b, (exec evs init place).order a b = true →
      evs[a].e.thread = evs[b].e.thread ∧ evs[a].seq < evs[b].seq := fun a b h => by
    simp only [Execution.order, get_union, Bool.or_eq_true, exec, toExec, get_ofFn,
      Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
    rcases h with h | h
    · exact h
    · rw [hp.sync_only b] at h; simp at h
  intro h
  rw [tc_iff_transGen] at h
  induction h with
  | single e => exact one _ _ e
  | tail _ e ih =>
    obtain ⟨h₁, h₂⟩ := one _ _ e
    exact ⟨ih.1.trans h₁, Nat.lt_trans ih.2 h₂⟩

theorem rf_iff {w r : Fin evs.length} : rfOf evs w r = true ↔
    evs[w].e.isWrite = true ∧ evs[r].e.isRead = true ∧ evs[w].key = evs[r].key ∧ w < r ∧
      ∀ w' : Fin evs.length, w < w' → w' < r → evs[w'].isWriteTo evs[r].key = false := by
  simp only [rfOf, get_ofFn, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, List.all_eq_true,
    List.mem_finRange, true_implies]
  constructor
  · rintro ⟨⟨⟨⟨h₁, h₂⟩, h₃⟩, h₄⟩, h₅⟩
    refine ⟨h₁, h₂, h₃, h₄, fun w' a b => ?_⟩
    have := h₅ w'
    simp only [a, b, decide_true, Bool.true_and, Bool.not_eq_true'] at this
    exact this
  · rintro ⟨h₁, h₂, h₃, h₄, h₅⟩
    refine ⟨⟨⟨⟨h₁, h₂⟩, h₃⟩, h₄⟩, fun w' => ?_⟩
    by_cases a : w < w'
    · by_cases b : w' < r
      · simp only [a, b, decide_true, Bool.true_and, h₅ w' a b, Bool.not_false]
      · simp only [b, decide_false, Bool.and_false, Bool.false_and, Bool.not_false]
    · simp only [a, decide_false, Bool.false_and, Bool.not_false]

/-- The last write to a read's location before it. -/
theorem exists_rf (hp : Performed evs init) {r w : Fin evs.length} (hr : evs[r].e.isRead = true)
    (hw : evs[w].isWriteTo evs[r].key = true) (hwr : w < r) : ∃ w', rfOf evs w' r = true := by
  -- Descend on the distance to `r`.
  have key : ∀ d (w : Fin evs.length), r.val - w.val = d → evs[w].isWriteTo evs[r].key = true →
      w < r → ∃ w', rfOf evs w' r = true := by
    intro d
    induction d using Nat.strongRecOn with
    | ind d ih =>
      intro w hd hw hwr
      by_cases hlast : ∀ w' : Fin evs.length, w < w' → w' < r → evs[w'].isWriteTo evs[r].key = false
      · simp only [Ev.isWriteTo, Bool.and_eq_true, beq_iff_eq] at hw
        exact ⟨w, rf_iff.2 ⟨hw.1, hr, hw.2, hwr, hlast⟩⟩
      · have : ∃ w' : Fin evs.length, w < w' ∧ w' < r ∧ evs[w'].isWriteTo evs[r].key = true :=
          Classical.byContradiction fun hn => hlast fun w' h₁ h₂ => by
            cases hx : evs[w'].isWriteTo evs[r].key
            · rfl
            · exact absurd ⟨w', h₁, h₂, hx⟩ hn
        obtain ⟨w', h₁, h₂, h₃⟩ := this
        have : r.val - w'.val < d := by
          have : w.val < w'.val := h₁
          omega
        exact ih (r.val - w'.val) this w' rfl h₃ h₂
  exact key _ w rfl hw hwr

theorem rf_unique {w w' r : Fin evs.length} (h : rfOf evs w r = true) (h' : rfOf evs w' r = true) :
    w = w' := by
  obtain ⟨hw, -, hk, hlt, hlast⟩ := rf_iff.1 h
  obtain ⟨hw', -, hk', hlt', hlast'⟩ := rf_iff.1 h'
  rcases Nat.lt_trichotomy w.val w'.val with c | c | c
  · have := hlast w' c hlt'
    simp only [Ev.isWriteTo, hw', hk', beq_self_eq_true, Bool.and_true] at this; cases this
  · exact Fin.ext c
  · have := hlast' w c hlt
    simp only [Ev.isWriteTo, hw, hk, beq_self_eq_true, Bool.and_true] at this; cases this

theorem rmw_iff {a b : Fin evs.length} : (exec evs init place).rmw a b = true ↔
    evs[a].rmwRead = true ∧ evs[a].e.thread = evs[b].e.thread ∧ evs[b].seq = evs[a].seq + 1 := by
  simp [exec, toExec, and_assoc]

theorem rmw_next (hp : Performed evs init) {a b : Fin evs.length}
    (h : (exec evs init place).rmw a b = true) : b.val = a.val + 1 := by
  obtain ⟨hr, ht, hs⟩ := rmw_iff.1 h
  obtain ⟨hlt, -, -, ht', hs', -, -⟩ := hp.rmw a hr
  apply Classical.byContradiction
  intro hne
  have ht'' : evs[(⟨a.val + 1, hlt⟩ : Fin evs.length)].e.thread = evs[a].e.thread := ht'
  have hs'' : evs[(⟨a.val + 1, hlt⟩ : Fin evs.length)].seq = evs[a].seq + 1 := hs'
  exact hp.dist b ⟨a.val + 1, hlt⟩ (fun e => hne (congrArg Fin.val e)) (by rw [← ht, ht''])
    (by rw [hs, hs''])

def timed (hp : Performed evs init) (place : Nat → Place) : (exec evs init place).Timed where
  t := Fin.val
  inj a b h := Fin.ext h
  rf_lt w r h := (rf_iff.1 h).2.2.2.1
  rf_latest w r w' h hw' hl hlt := by
    obtain ⟨_, hr, _, hwr, hlast⟩ := rf_iff.1 h
    obtain ⟨-, -, hk⟩ := (sameLocation_iff place).1 hl
    rcases Nat.lt_or_ge r.val w'.val with c | c
    · exact c
    · rcases Nat.lt_or_eq_of_le c with c | c
      · have hw'' : evs[w'].e.isWrite = true := hw'
        have := hlast w' hlt c
        simp only [Ev.isWriteTo, hw'', hk, beq_self_eq_true, Bool.and_true] at this; cases this
      · have : w' = r := Fin.ext c
        subst this
        exact absurd hw' (fun hw => not_read_write hr hw)
  init_first r w' hr hnone hw' hl := by
    obtain ⟨-, -, hk⟩ := (sameLocation_iff place).1 hl
    rcases Nat.lt_or_ge r.val w'.val with c | c
    · exact c
    · exfalso
      rcases Nat.lt_or_eq_of_le c with c | c
      · have hw'' : evs[w'].e.isWrite = true := hw'
        obtain ⟨w₀, hw₀⟩ := exists_rf hp hr
          (by simp only [Ev.isWriteTo, Bool.and_eq_true, beq_iff_eq]; exact ⟨hw'', hk⟩) c
        have := hnone w₀; simp only [exec, toExec] at this; rw [this] at hw₀; cases hw₀
      · have : w' = r := Fin.ext c
        subst this
        exact not_read_write hr hw'
  co_iff a b := by
    simp only [exec, toExec, coOf, get_ofFn, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq]
    constructor
    · rintro ⟨⟨⟨h₁, h₂⟩, h₃⟩, h₄⟩
      exact ⟨h₁, h₂, (sameLocation_iff place).2 ⟨isMem_of_isWrite h₁, isMem_of_isWrite h₂, h₃⟩, h₄⟩
    · rintro ⟨h₁, h₂, h₃, h₄⟩
      exact ⟨⟨⟨h₁, h₂⟩, ((sameLocation_iff place).1 h₃).2.2⟩, h₄⟩
  sc_iff a b := by
    simp only [exec, toExec, scOf, get_ofFn, Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨fun ⟨⟨h₁, h₂⟩, h₃⟩ => ⟨h₁, h₂, h₃⟩, fun ⟨h₁, h₂, h₃⟩ => ⟨⟨h₁, h₂⟩, h₃⟩⟩
  rmw_lt a b h := by show a.val < b.val; rw [rmw_next hp h]; omega
  rmw_adjacent a b w' h _ _ hne hlt := by
    show b.val < w'.val
    have := rmw_next hp h
    have : w'.val ≠ b.val := fun e => hne (Fin.ext e)
    have hlt' : a.val < w'.val := hlt
    omega
  fifo a b h hl := by
    obtain ⟨ht, hs⟩ := order_iff hp h
    obtain ⟨hma, hmb, hk⟩ := (sameLocation_iff place).1 hl
    show a.val < b.val
    rcases Nat.lt_or_ge a.val b.val with c | c
    · exact c
    · have hne : a ≠ b := fun e => by subst e; exact Nat.lt_irrefl _ hs
      have c : b < a := Nat.lt_of_le_of_ne c (fun e => hne (Fin.ext e).symm)
      exact absurd ⟨hmb, hma, hk.symm⟩ (hp.po b a c ht.symm hs).1
  drains a u h hd := by
    obtain ⟨ht, hs⟩ := order_iff hp h
    show a.val < u.val
    rcases Nat.lt_or_ge a.val u.val with c | c
    · exact c
    · have hne : a ≠ u := fun e => by subst e; exact Nat.lt_irrefl _ hs
      have c : u < a := Nat.lt_of_le_of_ne c (fun e => hne (Fin.ext e).symm)
      have := (hp.po u a c ht.symm hs).2.1
      rw [ev_drains, this] at hd; cases hd
  blocks v b h hb := by
    obtain ⟨ht, hs⟩ := order_iff hp h
    show v.val < b.val
    rcases Nat.lt_or_ge v.val b.val with c | c
    · exact c
    · have hne : v ≠ b := fun e => by subst e; exact Nat.lt_irrefl _ hs
      have c : b < v := Nat.lt_of_le_of_ne c (fun e => hne (Fin.ext e).symm)
      have := (hp.po b v c ht.symm hs).2.2
      rw [ev_blocks, this] at hb; cases hb
  bar_lt a b h := by
    simp only [exec, toExec, get_ofFn, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨ho, hi⟩, hk⟩ := h
    apply hp.bar a b _ _ ho hi
    · simp only [Ev.arrives]; split at hk <;> simp_all
    · simp only [Ev.waits]; split at hk <;> simp_all
  dep_lt a b h := by
    show a.val < b.val
    have : (exec evs init place).rmw a b = true := h
    rw [rmw_next hp this]; omega


theorem po_iff {a b : Fin evs.length} : (exec evs init place).po a b = true ↔
    evs[a].e.thread = evs[b].e.thread ∧ evs[a].seq < evs[b].seq := by
  simp [exec, toExec]

theorem poWF (hp : Performed evs init) : (exec evs init place).poWF = true := by
  simp only [Execution.poWF, Execution.strictOrder, Bool.and_eq_true, irreflexive_iff, subset_iff,
    List.all_eq_true, List.mem_finRange, true_implies]
  refine ⟨⟨⟨⟨fun a => ?_, fun a c h => ?_⟩, fun a b h => ?_⟩, fun a b => ?_⟩, ?_⟩
  · cases h : (exec evs init place).po a a
    · rfl
    · exact absurd (po_iff.1 h).2 (Nat.lt_irrefl _)
  · obtain ⟨b, h₁, h₂⟩ := get_comp.1 h
    obtain ⟨t₁, s₁⟩ := po_iff.1 h₁
    obtain ⟨t₂, s₂⟩ := po_iff.1 h₂
    exact po_iff.2 ⟨t₁.trans t₂, Nat.lt_trans s₁ s₂⟩
  · simp only [Execution.sameThread, Execution.thr, get_ofFn, beq_iff_eq, ev_thread]
    exact (po_iff.1 h).1
  · by_cases hab : a = b
    · simp [hab]
    by_cases ht : evs[a].e.thread = evs[b].e.thread
    · have hs := hp.dist a b hab ht
      rcases Nat.lt_or_gt_of_ne hs with c | c
      · simp [po_iff.2 ⟨ht, c⟩]
      · simp [po_iff.2 ⟨ht.symm, c⟩]
    · simp only [Execution.sameThread, Execution.thr, get_ofFn, beq_iff_eq, ev_thread,
        Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not]
      exact .inl (.inl (.inr (by simpa using ht)))
  · -- Program order runs forward in `seq`.
    have key : ∀ a b, tc (exec evs init place).order a b = true → evs[a].seq < evs[b].seq :=
      fun a b h => (order_iff hp h).2
    simp only [acyclic, irreflexive_iff]
    intro a
    cases h : tc (exec evs init place).order a a
    · rfl
    · exact absurd (key a a h) (Nat.lt_irrefl _)

theorem rfWF (hp : Performed evs init) : (exec evs init place).rfWF = true := by
  simp only [Execution.rfWF, Bool.and_eq_true, subset_iff, List.all_eq_true, List.mem_finRange,
    true_implies]
  refine ⟨fun w r h => ?_, fun r => ?_⟩
  · obtain ⟨hw, hr, hk, -, -⟩ := rf_iff.1 (show rfOf evs w r = true from h)
    exact comp_dom_iff.2 ⟨dom_comp_iff.2 ⟨hw, (sameLocation_iff place).2
      ⟨isMem_of_isWrite hw, isMem_of_isRead hr, hk⟩⟩, hr⟩
  · cases hr : evs[r].e.isRead
    · simp only [reads_iff, Bool.or_eq_true, Bool.not_eq_true']; exact .inl hr
    · simp only [reads_iff, hr, Bool.not_true, Bool.false_or]
      have hrf : ∀ w, (exec evs init place).rf w r = rfOf evs w r := fun _ => rfl
      rcases hp.val r hr with ⟨w, hwr, hw, hlast, hv⟩ | ⟨hnone, hv⟩
      · -- The read reads `w`, its only source.
        have hrfw : rfOf evs w r = true := by
          simp only [Ev.isWriteTo, Bool.and_eq_true, beq_iff_eq] at hw
          exact rf_iff.2 ⟨hw.1, hr, hw.2, hwr, hlast⟩
        have hfil : (List.finRange evs.length).filter (fun w' => (exec evs init place).rf w' r) =
            [w] := by
          have hp' := (List.pairwise_lt_finRange evs.length).filter
            (fun w' => (exec evs init place).rf w' r)
          have hmem : ∀ w', w' ∈ (List.finRange evs.length).filter
              (fun w' => (exec evs init place).rf w' r) ↔ w' = w := fun w' => by
            simp only [List.mem_filter, List.mem_finRange, true_and, hrf]
            exact ⟨fun h => rf_unique h hrfw, fun e => e ▸ hrfw⟩
          generalize (List.finRange evs.length).filter (fun w' => (exec evs init place).rf w' r) = l
            at hp' hmem
          match l, hp', hmem with
          | [], _, hmem => exact absurd ((hmem w).2 rfl) (by simp)
          | [a], _, hmem => rw [(hmem a).1 (by simp)]
          | a :: b :: _, hp', hmem =>
            have ha := (hmem a).1 (by simp)
            have hb := (hmem b).1 (by simp)
            have : a < b := List.rel_of_pairwise_cons hp' (by simp)
            exact absurd (ha.trans hb.symm) (Fin.ne_of_lt this)
        rw [hfil]
        simp only [exec, toExec, beq_iff_eq]
        exact hv
      · have hfil : (List.finRange evs.length).filter (fun w' => (exec evs init place).rf w' r) =
            [] := by
          rw [List.filter_eq_nil_iff]
          intro w' _ h
          obtain ⟨hw, -, hk, hlt, -⟩ := rf_iff.1 (show rfOf evs w' r = true from h)
          have := hnone w' hlt
          simp only [Ev.isWriteTo, hw, hk, beq_self_eq_true, Bool.and_true] at this
          cases this
        rw [hfil]
        have hlt : (evs.map Ev.key).idxOf evs[r].key < (evs.map Ev.key).length :=
          List.idxOf_lt_length_of_mem (List.mem_map_of_mem (mem_get r))
        have e : (evs.map Ev.key)[code evs evs[r]]? = some evs[r].key := by
          unfold code; rw [List.getElem?_eq_getElem hlt, List.getElem_idxOf hlt]
        simp only [exec, toExec, id, beq_iff_eq]
        rw [e, hv]

theorem rmwWF (hp : Performed evs init) : (exec evs init place).rmwWF = true := by
  simp only [Execution.rmwWF, Bool.and_eq_true, subset_iff, List.all_eq_true, List.mem_finRange,
    true_implies]
  have parts : ∀ a b, (exec evs init place).rmw a b = true → b.val = a.val + 1 ∧
      evs[a].e.isRead = true ∧ evs[b].e.isWrite = true ∧ evs[a].e.thread = evs[b].e.thread ∧
      evs[b].seq = evs[a].seq + 1 ∧ evs[b].key = evs[a].key ∧ evs[b].e.scope = evs[a].e.scope := by
    intro a b h
    have hb := rmw_next hp h
    obtain ⟨hr, -, -⟩ := rmw_iff.1 h
    obtain ⟨hlt, h₁, h₂, h₃, h₄, h₅, h₆⟩ := hp.rmw a hr
    have : b = ⟨a.val + 1, hlt⟩ := Fin.ext hb
    subst this
    exact ⟨rfl, h₁, h₂, h₃.symm, h₄, h₅, h₆⟩
  refine ⟨⟨fun a b h => ?_, fun a b => ?_⟩, fun a b h => ?_⟩
  · obtain ⟨-, hr, hw, ht, hs, hk, -⟩ := parts a b h
    refine comp_dom_iff.2 ⟨dom_comp_iff.2 ⟨hr, ?_⟩, hw⟩
    simp only [get_inter, Bool.and_eq_true]
    refine ⟨⟨po_iff.2 ⟨ht, by omega⟩, h⟩, ?_⟩
    simp only [Execution.sameAlias, Execution.mems, get_ofFn, Bool.and_eq_true, beq_iff_eq]
    exact ⟨⟨isMem_of_isRead hr, isMem_of_isWrite hw⟩,
      show code evs evs[a] = code evs evs[b] by unfold code; rw [hk]⟩
  · cases h : (exec evs init place).rmw a b
    · simp
    · obtain ⟨-, -, -, -, -, -, hs⟩ := parts a b h
      simp only [Bool.not_true, Bool.false_or, beq_iff_eq]
      exact hs.symm
  · obtain ⟨-, -, -, ht, hs, -, -⟩ := parts a b h
    exact po_iff.2 ⟨ht, by omega⟩

theorem barWF : (exec evs init place).barWF = true := by
  simp only [Execution.barWF, List.all_eq_true, List.mem_finRange, true_implies]
  intro a b
  cases h : (exec evs init place).bar a b
  · rfl
  · simp only [exec, toExec, get_ofFn, Bool.and_eq_true] at h
    obtain ⟨_, hk⟩ := h
    simp only [Bool.not_true, Bool.false_or, ev_kind, Execution.placeOf, Execution.thr, ev_thread]
    exact hk

/-- **A performed list's execution is consistent.** -/
theorem consistent (hp : Performed evs init) (place : Nat → Place) :
    (exec evs init place).consistent = true :=
  (hp.timed place).consistent hp.poWF hp.rfWF hp.rmwWF barWF


/-! ## Appending events -/

theorem fin_append {α : Type} {xs ys : List α} (k : Fin (xs ++ ys).length) :
    (∃ h : k.val < xs.length, (xs ++ ys)[k] = xs[k.val]) ∨
      (∃ h : k.val - xs.length < ys.length, xs.length ≤ k.val ∧ (xs ++ ys)[k] = ys[k.val - xs.length]) := by
  by_cases h : k.val < xs.length
  · exact .inl ⟨h, List.getElem_append_left h⟩
  · have hk := k.isLt
    simp only [List.length_append] at hk
    exact .inr ⟨by omega, by omega, List.getElem_append_right (by omega)⟩

/-- The conditions under which events `es`, performed after `evs`, keep the list performed. -/
structure Append (evs es : List Ev) (init : Loc → UInt8) : Prop where
  sync_new : ∀ f ∈ es, f.after = []
  dist_new : ∀ a b : Fin es.length, a ≠ b → es[a].e.thread = es[b].e.thread → es[a].seq ≠ es[b].seq
  dist_cross : ∀ e ∈ evs, ∀ f ∈ es, e.e.thread = f.e.thread → e.seq ≠ f.seq
  po_new : ∀ a b : Fin es.length, a < b → es[a].e.thread = es[b].e.thread → es[a].seq < es[b].seq
  po_cross : ∀ e ∈ evs, ∀ f ∈ es, e.e.thread = f.e.thread → f.seq < e.seq →
    ¬(e.e.isMem = true ∧ f.e.isMem = true ∧ e.key = f.key) ∧ e.e.drains = false ∧ f.e.blocks = false
  rmw_new : ∀ a : Fin es.length, es[a].rmwRead = true → ∃ h : a.val + 1 < es.length,
    es[a].e.isRead = true ∧ es[a.val + 1].e.isWrite = true ∧
    es[a.val + 1].e.thread = es[a].e.thread ∧ es[a.val + 1].seq = es[a].seq + 1 ∧
    es[a.val + 1].key = es[a].key ∧ es[a.val + 1].e.scope = es[a].e.scope
  bar_cross : ∀ a ∈ es, ∀ e ∈ evs, a.arrives = true → e.waits = true → a.obj = e.obj → a.inst ≠ e.inst
  bar_new : ∀ a b : Fin es.length, es[a].arrives = true → es[b].waits = true →
    es[a].obj = es[b].obj → es[a].inst = es[b].inst → a < b
  val_new : ∀ a : Fin es.length, es[a].e.isRead = true →
    (∀ b : Fin es.length, b < a → es[b].e.isWrite = false) ∧
    ((∃ w : Fin evs.length, evs[w].isWriteTo es[a].key = true ∧
        (∀ w' : Fin evs.length, w < w' → evs[w'].isWriteTo es[a].key = false) ∧ es[a].val = evs[w].val) ∨
      ((∀ w : Fin evs.length, evs[w].isWriteTo es[a].key = false) ∧ es[a].val = initBytes init es[a].key))

theorem append (hp : Performed evs init) {es : List Ev} (ha : Append evs es init) :
    Performed (evs ++ es) init := by
  have mem_l : ∀ (i : Nat) (h : i < evs.length), evs[i] ∈ evs := fun i h => List.getElem_mem h
  have mem_r : ∀ (i : Nat) (h : i < es.length), es[i] ∈ es := fun i h => List.getElem_mem h
  constructor
  · intro i
    rcases fin_append i with ⟨hi, ei⟩ | ⟨hi, li, ei⟩ <;> rw [ei]
    · exact hp.sync_only ⟨i, hi⟩
    · exact ha.sync_new _ (mem_r _ hi)
  · intro i j hij ht
    rcases fin_append i with ⟨hi, ei⟩ | ⟨hi, li, ei⟩ <;> rcases fin_append j with ⟨hj, ej⟩ | ⟨hj, lj, ej⟩ <;>
      rw [ei, ej] at ht ⊢
    · exact hp.dist ⟨i, hi⟩ ⟨j, hj⟩ (fun e => hij (Fin.ext (by
        have := congrArg Fin.val e; simpa using this))) ht
    · exact ha.dist_cross _ (mem_l _ hi) _ (mem_r _ hj) ht
    · exact fun e => ha.dist_cross _ (mem_l _ hj) _ (mem_r _ hi) ht.symm e.symm
    · exact ha.dist_new ⟨_, hi⟩ ⟨_, hj⟩ (fun e => hij (Fin.ext (by
        have := congrArg Fin.val e; simp only at this; omega))) ht
  · intro i j hij ht hs
    rcases fin_append i with ⟨hi, ei⟩ | ⟨hi, li, ei⟩ <;> rcases fin_append j with ⟨hj, ej⟩ | ⟨hj, lj, ej⟩ <;>
      rw [ei, ej] at ht hs ⊢
    · exact hp.po ⟨i, hi⟩ ⟨j, hj⟩ hij ht hs
    · exact ha.po_cross _ (mem_l _ hi) _ (mem_r _ hj) ht hs
    · exact absurd hij (by have : (i : Nat) < j := hij; omega)
    · exact absurd hs (Nat.not_lt.2 (Nat.le_of_lt (ha.po_new ⟨_, hi⟩ ⟨_, hj⟩
        (by show i.val - evs.length < j.val - evs.length; have : (i : Nat) < j := hij; omega) ht)))
  · intro i hr
    rcases fin_append i with ⟨hi, ei⟩ | ⟨hi, li, ei⟩ <;> rw [ei] at hr
    · obtain ⟨h, h₁, h₂, h₃, h₄, h₅, h₆⟩ := hp.rmw ⟨i, hi⟩ hr
      have h : i.val + 1 < evs.length := h
      have e₁ : (evs ++ es)[i.val + 1]'(by simp; omega) = evs[i.val + 1] := List.getElem_append_left h
      refine ⟨by simp; omega, ?_⟩
      rw [ei, e₁]
      exact ⟨h₁, h₂, h₃, h₄, h₅, h₆⟩
    · obtain ⟨h, h₁, h₂, h₃, h₄, h₅, h₆⟩ := ha.rmw_new ⟨_, hi⟩ hr
      have h : i.val - evs.length + 1 < es.length := h
      have e₁ : (evs ++ es)[i.val + 1]'(by simp; omega) = es[i.val - evs.length + 1] := by
        rw [List.getElem_append_right (by omega)]; congr 1; omega
      refine ⟨by simp; omega, ?_⟩
      rw [ei, e₁]
      exact ⟨h₁, h₂, h₃, h₄, h₅, h₆⟩
  · intro i j hai hwj ho hin
    rcases fin_append i with ⟨hi, ei⟩ | ⟨hi, li, ei⟩ <;> rcases fin_append j with ⟨hj, ej⟩ | ⟨hj, lj, ej⟩ <;>
      rw [ei] at hai ho hin <;> rw [ej] at hwj ho hin
    · exact hp.bar ⟨i, hi⟩ ⟨j, hj⟩ hai hwj ho hin
    · show i.val < j.val; omega
    · exact absurd hin (ha.bar_cross _ (mem_r _ hi) _ (mem_l _ hj) hai hwj ho)
    · have := ha.bar_new ⟨_, hi⟩ ⟨_, hj⟩ hai hwj ho hin
      show i.val < j.val
      have : i.val - evs.length < j.val - evs.length := this
      omega
  · intro i hr
    have notw : ∀ v : Ev, v.e.isWrite = false → ∀ k, v.isWriteTo k = false := fun v h k => by
      simp [Ev.isWriteTo, h]
    rcases fin_append i with ⟨hi, ei⟩ | ⟨hi, li, ei⟩
    · rw [ei] at hr ⊢
      rcases hp.val ⟨i, hi⟩ hr with ⟨w, hw, h₁, h₂, h₃⟩ | ⟨h₁, h₂⟩
      · have ew : (evs ++ es)[w.val]'(by simp; omega) = evs[w] := List.getElem_append_left w.isLt
        refine .inl ⟨⟨w, by simp; omega⟩, hw, ?_, fun w' hw₁ hw₂ => ?_, ?_⟩
        · show (evs ++ es)[w.val].isWriteTo _ = true; rw [ew]; exact h₁
        · have hw' : w'.val < evs.length := by have : (w' : Nat) < i := hw₂; omega
          show (evs ++ es)[w'.val].isWriteTo _ = false
          rw [List.getElem_append_left hw']
          exact h₂ ⟨w', hw'⟩ hw₁ hw₂
        · show _ = (evs ++ es)[w.val].val; rw [ew]; exact h₃
      · refine .inr ⟨fun w' hw₂ => ?_, h₂⟩
        have hw' : w'.val < evs.length := by have : (w' : Nat) < i := hw₂; omega
        show (evs ++ es)[w'.val].isWriteTo _ = false
        rw [List.getElem_append_left hw']
        exact h₁ ⟨w', hw'⟩ hw₂
    · rw [ei] at hr ⊢
      obtain ⟨hpre, hlast⟩ := ha.val_new ⟨_, hi⟩ hr
      -- A write before `i` among the new events would come before a read of theirs.
      have newer : ∀ w' : Fin (evs ++ es).length, evs.length ≤ w'.val → w' < i →
          (evs ++ es)[w'].isWriteTo es[i.val - evs.length].key = false := fun w' h₁ h₂ => by
        have hw' : w'.val - evs.length < es.length := by
          have := w'.isLt; simp at this; omega
        show (evs ++ es)[w'.val].isWriteTo _ = false
        rw [List.getElem_append_right h₁]
        exact notw _ (hpre ⟨_, hw'⟩ (by show w'.val - evs.length < i.val - evs.length
                                        have : (w' : Nat) < i := h₂; omega)) _
      rcases hlast with ⟨w, h₁, h₂, h₃⟩ | ⟨h₁, h₂⟩
      · have ew : (evs ++ es)[w.val]'(by simp; omega) = evs[w] := List.getElem_append_left w.isLt
        refine .inl ⟨⟨w, by simp; omega⟩, by show w.val < i.val; omega, ?_, fun w' hw₁ hw₂ => ?_, ?_⟩
        · show (evs ++ es)[w.val].isWriteTo _ = true; rw [ew]; exact h₁
        · by_cases hw' : w'.val < evs.length
          · show (evs ++ es)[w'.val].isWriteTo _ = false
            rw [List.getElem_append_left hw']
            exact h₂ ⟨w', hw'⟩ hw₁
          · exact newer w' (by omega) hw₂
        · show _ = (evs ++ es)[w.val].val; rw [ew]; exact h₃
      · refine .inr ⟨fun w' hw₂ => ?_, h₂⟩
        by_cases hw' : w'.val < evs.length
        · show (evs ++ es)[w'.val].isWriteTo _ = false
          rw [List.getElem_append_left hw']
          exact h₁ ⟨w', hw'⟩
        · exact newer w' (by omega) hw₂

end Performed

end NV
