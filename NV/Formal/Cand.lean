import NV.Formal.Outcomes
import NV.PTX.Memory.Chain
import NV.Memory.Performed

/-!
# A candidate execution at its events

What a proof about a program's outcomes reads off a candidate `c`'s execution: its relations and
event fields at the events of `c.evs` (`po_iff`, `issue_iff`, `addr_iff`, …), program and issue
order from threads and positions (`order_of_po`, `order_of_issue`), proxy-preserved causality
between generic accesses of one location (`ppcb_gen`), and where a read's value comes from
(`source`).
-/

namespace NV.Formal.Candidate

open NV NV.PTX NV.PTX.Execution NV.Rel

variable {σ : Type} {P : Program σ} (c : Candidate P)

theorem po_iff {i j : Fin c.evs.length} :
    c.exec.po i j = true ↔ c.evs[i].e.thread = c.evs[j].e.thread ∧ c.evs[i].seq < c.evs[j].seq := by
  simp [Candidate.exec, toExec]

theorem issue_iff {i j : Fin c.evs.length} :
    c.exec.issue i j = true ↔ ∃ p ∈ c.evs[j].after, c.evs[i].e.thread = p.1 ∧ c.evs[i].seq < p.2 := by
  simp [Candidate.exec, toExec, List.any_eq_true]

theorem addr_iff {i j : Fin c.evs.length} :
    (c.exec.ev i).addr = (c.exec.ev j).addr ↔ c.evs[i].key = c.evs[j].key :=
  Performed.code_eq (List.getElem_mem _) (List.getElem_mem _)

theorem writes_iff {i : Fin c.evs.length} : c.exec.writes i = c.evs[i].e.isWrite := rfl
theorem reads_iff {i : Fin c.evs.length} : c.exec.reads i = c.evs[i].e.isRead := rfl
theorem mems_iff {i : Fin c.evs.length} : c.exec.mems i = c.evs[i].e.isMem := rfl
theorem val_iff {i : Fin c.evs.length} : c.exec.val i = c.evs[i].val := rfl

/-- An order of one thread at increasing positions. -/
theorem order_of_po {i j : Fin c.evs.length} (ht : c.evs[i].e.thread = c.evs[j].e.thread)
    (hs : c.evs[i].seq < c.evs[j].seq) : tc c.exec.order i j = true :=
  tc_of (by simp [Execution.order, (po_iff c).2 ⟨ht, hs⟩])

/-- An asynchronous event after its issuer's events below the issue position. -/
theorem order_of_issue {i j : Fin c.evs.length} {p : Nat × Nat} (hp : p ∈ c.evs[j].after)
    (ht : c.evs[i].e.thread = p.1) (hs : c.evs[i].seq < p.2) : tc c.exec.order i j = true :=
  tc_of (by simp [Execution.order, (issue_iff c).2 ⟨p, hp, ht, hs⟩])

/-- An index of a member. -/
theorem index {v : Ev} (h : v ∈ c.evs) : ∃ i : Fin c.evs.length, c.evs[i] = v := by
  obtain ⟨i, hi, e⟩ := List.mem_iff_getElem.1 h
  exact ⟨⟨i, hi⟩, e⟩

theorem sameLoc_of {i j : Fin c.evs.length} {A B : Ev} (hi : c.evs[i] = A) (hj : c.evs[j] = B)
    (hA : A.e.isMem = true) (hB : B.e.isMem = true) (hk : A.key = B.key) :
    c.exec.sameLocation i j = true := by
  simp only [Execution.sameLocation, Rel.get_ofFn, Bool.and_eq_true, beq_iff_eq, mems_iff, hi, hj,
    hA, hB, true_and]
  exact congrArg _ ((addr_iff c).2 (by rw [hi, hj]; exact hk))

/-- Two generic accesses of one location, ordered: proxy-preserved base causality. -/
theorem ppcb_gen {i j : Fin c.evs.length} {A B : Ev} (hi : c.evs[i] = A) (hj : c.evs[j] = B)
    (hA : A.e.isMem = true ∧ A.e.proxy = .generic) (hB : B.e.isMem = true ∧ B.e.proxy = .generic)
    (hk : A.key = B.key) (ho : c.exec.causeBase i j = true) :
    c.exec.proxyPreservedCauseBase i j = true := by
  have hg : ∀ {k : Fin c.evs.length} {C : Ev}, c.evs[k] = C → C.e.isMem = true ∧ C.e.proxy = .generic →
      c.exec.generic k = true := fun {k} {C} hk hC => by
    show (c.evs[k].e.isMem && c.evs[k].e.proxy == .generic) = true
    rw [hk, hC.1, hC.2]; rfl
  exact ppcb_of_generic (hg hi hA) (hg hj hB) ((addr_iff c).2 (by rw [hi, hj]; exact hk)) ho

/-- A read's source: none (the initial bytes), or a write to its location. -/
theorem source (hc : c.exec.consistent = true) {r : Fin c.evs.length} (hr : c.evs[r].e.isRead = true) :
    ((∀ w, c.exec.rf w r = false) ∧ c.evs[r].val = initBytes P.mem c.evs[r].key) ∨
      ∃ w, c.exec.rf w r = true ∧ c.evs[r].val = c.evs[w].val ∧ c.evs[w].e.isWrite = true ∧
        c.evs[w].key = c.evs[r].key := by
  rcases rf_cases hc (r := r) hr with ⟨hn, hv⟩ | ⟨w, h₁, h₂, h₃, h₄⟩
  · left
    refine ⟨hn, ?_⟩
    rw [val_iff] at hv
    rw [hv]
    have hm : c.evs[r] ∈ c.evs := List.getElem_mem _
    have hlt := List.idxOf_lt_length_of_mem (List.mem_map_of_mem (f := Ev.key) hm)
    have hget : (c.evs.map Ev.key)[code c.evs c.evs[r]]? = some c.evs[r].key := by
      unfold code; rw [List.getElem?_eq_getElem hlt, List.getElem_idxOf hlt]
    show (match (c.evs.map Ev.key)[code c.evs c.evs[r]]? with
      | some key => initBytes P.mem key | none => []) = _
    rw [hget]
  · exact .inr ⟨w, h₁, h₂, h₃, (addr_iff c).1 h₄⟩

/-- Two accesses that are not weak, of one proxy, by threads all placed alike, are morally strong. -/
theorem strongR_of {pl : Place} (hpl : ∀ t, P.place t = pl) {i j : Fin c.evs.length}
    (hi : c.evs[i].e.scope ≠ .thread) (hj : c.evs[j].e.scope ≠ .thread)
    (hp : c.evs[i].e.proxy = c.evs[j].e.proxy) : c.exec.strongR i j = true := by
  have hc : ∀ (s : Scope) (t u : Nat), s ≠ .thread → s.contains t u pl pl = true := by
    intro s t u hs; cases s <;> simp_all [Scope.contains]
  simp only [Execution.strongR, Rel.get_ofFn, Execution.placeOf, Bool.and_eq_true, beq_iff_eq]
  show (Scope.contains c.evs[i].e.scope _ _ (P.place _) (P.place _) = true ∧
    Scope.contains c.evs[j].e.scope _ _ (P.place _) (P.place _) = true) ∧ c.evs[i].e.proxy = c.evs[j].e.proxy
  rw [hpl, hpl]
  exact ⟨⟨hc _ _ _ hi, hc _ _ _ hj⟩, hp⟩

theorem ev_get (i : Fin c.evs.length) : c.exec.ev i = { c.evs[i].e with addr := code c.evs c.evs[i] } :=
  rfl

theorem rmw_of {i j : Fin c.evs.length} {A B : Ev} (hi : c.evs[i] = A) (hj : c.evs[j] = B)
    (hr : A.rmwRead = true) (ht : A.e.thread = B.e.thread) (hs : B.seq = A.seq + 1) :
    c.exec.rmw i j = true := by
  have hi' : c.evs[(i : Nat)] = A := hi
  have hj' : c.evs[(j : Nat)] = B := hj
  simp [Candidate.exec, toExec, hi', hj', hr, ht, hs]

/-- The initial bytes an event's location holds. -/
theorem init_at (i : Fin c.evs.length) :
    c.exec.init (c.exec.loc (c.exec.ev i).addr) = initBytes P.mem c.evs[i].key := by
  have hm : c.evs[i] ∈ c.evs := List.getElem_mem _
  have hlt := List.idxOf_lt_length_of_mem (List.mem_map_of_mem (f := Ev.key) hm)
  have hget : (c.evs.map Ev.key)[code c.evs c.evs[i]]? = some c.evs[i].key := by
    unfold code; rw [List.getElem?_eq_getElem hlt, List.getElem_idxOf hlt]
  show (match (c.evs.map Ev.key)[code c.evs c.evs[i]]? with
    | some key => initBytes P.mem key | none => []) = _
  rw [hget]

end NV.Formal.Candidate
