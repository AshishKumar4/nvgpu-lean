import NV.Formal.Outcomes

/-!
# Compulsory traffic

A non-interference argument on the axiomatic semantics (`NV.Formal.Outcomes`), so it holds of
every program, whatever its algorithm: if a program computes `f`, and an output byte of `f`
depends on an input byte `b` (changing `b` alone changes it), then every execution of the
program that finishes reads `b`.

Why: take a consistent candidate execution that reads no event's worth of `b`. Changing the
initial value of `b` changes nothing the execution's consistency looks at, since the memory model
consults initial values only for reads (`rfWF`), so the same candidate is an execution of the
program started from the changed memory. Its final memory is the same except at bytes nothing
writes; an output byte other than `b` is then unchanged, though `f` says it must change.

* `compulsory`: an execution of a program computing `f` reads every byte an output depends on,
  unless that output is the byte itself and `f` leaves it unchanged.
* `compulsory_traffic`: its reads move at least as many bytes as there are such bytes.

The class is every program: the theorem is about the semantics, not an algorithm. What it needs
of `f` is the dependence, at the initial memory the run starts from (`DependsAt`). An in-place
output byte that `f` leaves equal to its input is excluded: a program may leave it alone.
* `compulsory_write`, `compulsory_write_traffic`: every output byte `f` changes is written.
-/

namespace NV.Bounds

open NV NV.Formal NV.PTX

variable {σ : Type}

/-- Memory `m` with byte `b` set to `v`. -/
def upd (m : Loc → UInt8) (b : Loc) (v : UInt8) : Loc → UInt8 := fun l => if l = b then v else m l

/-- `P`, started from memory `m`. -/
def withMem (P : Program σ) (m : Loc → UInt8) : Program σ := { P with mem := m }

/-- `P` computes `f` at the output bytes `out`, from every initial memory `pre` admits: each
`done` outcome agrees with `f` there. -/
def Computes (P : Program σ) (pre : (Loc → UInt8) → Prop) (out : Loc → Prop)
    (f : (Loc → UInt8) → Loc → UInt8) : Prop :=
  ∀ m, pre m → ∀ mf, Outcomes (withMem P m) (.done mf) → ∀ o, out o → mf o = f m o

/-- Output byte `o` of `f` depends on byte `b` at memory `m`: changing `b` alone, to a memory
`pre` admits, changes it. -/
def DependsAt (pre : (Loc → UInt8) → Prop) (f : (Loc → UInt8) → Loc → UInt8) (m : Loc → UInt8)
    (b o : Loc) : Prop :=
  ∃ v, pre (upd m b v) ∧ f (upd m b v) o ≠ f m o

/-- Event `v` reads byte `b`. -/
def readsByte (v : Ev) (b : Loc) : Prop :=
  v.e.isRead = true ∧ v.loc.space = b.space ∧ v.loc.owner = b.owner ∧ v.loc.addr ≤ b.addr ∧
    b.addr < v.loc.addr + v.size

/-- Some event of the candidate reads byte `b`. -/
def Reads {P : Program σ} (c : Candidate P) (b : Loc) : Prop := ∃ v ∈ c.evs, readsByte v b

/-! ## The candidate does not depend on the initial memory -/

theorem actEvents_withMem (P : Program σ) (m : Loc → UInt8) (a : Agent σ) (ctr : Nat → Nat)
    (act : Act) : (withMem P m).actEvents a ctr act = P.actEvents a ctr act := rfl

theorem agentEvents_withMem (P : Program σ) (m : Loc → UInt8) (a : Agent σ) :
    ∀ (ctr : Nat → Nat) (acts : List Act),
      (withMem P m).agentEvents a ctr acts = P.agentEvents a ctr acts
  | _, [] => rfl
  | ctr, act :: acts => by
    simp only [Program.agentEvents, actEvents_withMem, agentEvents_withMem P m a _ acts]

theorem asyncEvents_withMem (P : Program σ) (m : Loc → UInt8) (a : Agent σ) :
    ∀ (ctr : Nat → Nat) (st : AsyncState) (acts : List Act),
      (withMem P m).asyncEvents a ctr st acts = P.asyncEvents a ctr st acts
  | _, _, [] => rfl
  | ctr, st, act :: acts => by
    have ih := fun c st => asyncEvents_withMem P m a c st acts
    simp only [Program.asyncEvents, ih]
    rfl

theorem events_withMem (P : Program σ) (m : Loc → UInt8) (runs : List (List Act × End)) :
    (withMem P m).events runs = P.events runs := by
  simp only [Program.events, agentEvents_withMem, asyncEvents_withMem]
  rfl

/-- Consistency consults the initial values only of the locations reads read. -/
theorem consistent_init_congr {n : Nat} {V : Type} [DecidableEq V] (x : Execution n V)
    (i' : Nat → V)
    (h : ∀ r : Fin n, x.reads r = true → x.init (x.loc (x.ev r).addr) = i' (x.loc (x.ev r).addr)) :
    ({ x with init := i' } : Execution n V).consistent = x.consistent := by
  have hrf : ({ x with init := i' } : Execution n V).rfWF = x.rfWF := by
    simp only [Execution.rfWF, Execution.reads, Execution.sameLocation]
    congr 1
    congr 1
    funext r
    cases hr : (x.ev r).isRead
    · simp
    · simp [h r hr]
  unfold Execution.consistent Execution.wellFormed
  rw [hrf]
  rfl

/-- What a read of the candidate execution finds as its location's initial value: the initial
bytes of its own location. -/
theorem toExec_init_read (evs : List Ev) (place : Nat → Place) (init : Loc → UInt8)
    (rf co sc : NV.Rel evs.length) (r : Fin evs.length) :
    let x := toExec evs place init rf co sc
    x.init (x.loc (x.ev r).addr) = initBytes init evs[r].key := by
  have hmem : evs[r].key ∈ evs.map Ev.key := List.mem_map.2 ⟨evs[r], by simp, rfl⟩
  have hlt := List.idxOf_lt_length_of_mem hmem
  show (match (evs.map Ev.key)[(evs.map Ev.key).idxOf evs[r].key]? with
    | some key => initBytes init key | none => []) = _
  simp only [List.getElem?_eq_getElem hlt, List.getElem_idxOf hlt]

theorem initBytes_upd (m : Loc → UInt8) (b : Loc) (v : UInt8) (key : Loc × Nat)
    (h : ¬ (key.1.space = b.space ∧ key.1.owner = b.owner ∧ key.1.addr ≤ b.addr ∧
      b.addr < key.1.addr + key.2)) :
    initBytes (upd m b v) key = initBytes m key := by
  unfold initBytes
  apply List.map_congr_left
  intro i hi
  have hi := List.mem_range.1 hi
  unfold upd
  refine ite_eq_right fun heq => h ?_
  subst heq
  exact ⟨rfl, rfl, by show key.1.addr ≤ key.1.addr + i; omega,
    by show key.1.addr + i < key.1.addr + key.2; omega⟩

/-! ## Non-interference -/

/-- **Compulsory reads.** If `P` computes `f`, and output byte `o` depends on byte `b` at memory
`m`, then every consistent candidate execution of `P` from `m` that finishes reads `b`, provided
`o` is another byte or one `f` changes (an output equal to its own unchanged input need not be
read). -/
theorem compulsory {P : Program σ} {pre : (Loc → UInt8) → Prop} {out : Loc → Prop}
    {f : (Loc → UInt8) → Loc → UInt8} {m : Loc → UInt8} {b o : Loc} {mf : Loc → UInt8}
    (hP : Computes P pre out f) (hm : pre m) (ho : out o) (hob : o ≠ b ∨ f m o ≠ m o)
    (hd : DependsAt pre f m b o) (c : Candidate (withMem P m)) (hv : c.Valid)
    (hc : c.exec.consistent = true) (hy : c.Yields (.done mf)) : Reads c b := by
  classical
  apply Classical.byContradiction
  intro hnr
  obtain ⟨v, hpre', hne⟩ := hd
  let c' : Candidate (withMem P (upd m b v)) :=
    { runs := c.runs, length := c.length, runs_ok := c.runs_ok, evs := c.evs,
      perm := by
        have := c.perm
        rw [events_withMem] at this
        rw [events_withMem]
        exact this,
      rf := c.rf, co := c.co, sc := c.sc, collects := c.collects, dep := c.dep, dep_ok := c.dep_ok }
  have hv' : c'.Valid := hv
  have hc' : c'.exec.consistent = true := by
    have hx : c'.exec = { c.exec with init := c'.exec.init } := rfl
    have key : ∀ r : Fin c.evs.length, c.exec.reads r = true →
        c.exec.init (c.exec.loc (c.exec.ev r).addr) = c'.exec.init (c.exec.loc (c.exec.ev r).addr) := by
      intro r hr
      have h₁ : c.exec.init (c.exec.loc (c.exec.ev r).addr) = initBytes m c.evs[r].key :=
        toExec_init_read c.evs (withMem P m).place m c.rf c.co c.sc r
      have h₂ : c'.exec.init (c'.exec.loc (c'.exec.ev r).addr) = initBytes (upd m b v) c.evs[r].key :=
        toExec_init_read c.evs (withMem P (upd m b v)).place (upd m b v) c.rf c.co c.sc r
      show c.exec.init _ = c'.exec.init (c'.exec.loc (c'.exec.ev r).addr)
      rw [h₁, h₂, initBytes_upd]
      intro hk
      exact hnr ⟨c.evs[r], by simp, hr, hk⟩
    rw [hx, consistent_init_congr c.exec c'.exec.init key]
    exact hc
  -- The final memory of the changed run: `mf` where something writes, the changed memory elsewhere.
  let mf' : Loc → UInt8 := fun l =>
    if ∀ i : Fin c.evs.length, c.evs[i].covers l = false then upd m b v l else mf l
  have hy' : c'.Yields (.done mf') := by
    refine ⟨hy.1, hy.2.1, ?_⟩
    intro l
    by_cases hl : ∀ i : Fin c.evs.length, c.evs[i].covers l = false
    · exact .inr ⟨hl, ite_eq_left hl⟩
    · rcases hy.2.2 l with ⟨i, h₁, h₂, h₃⟩ | ⟨h₁, _⟩
      · exact .inl ⟨i, h₁, h₂, (ite_eq_right hl).trans h₃⟩
      · exact absurd h₁ hl
  have hf' := hP (upd m b v) hpre' mf' ⟨c', hv', hc', hy'⟩ o ho
  have hf := hP m hm mf ⟨c, hv, hc, hy⟩ o ho
  apply hne
  rw [← hf', ← hf]
  by_cases hl : ∀ i : Fin c.evs.length, c.evs[i].covers o = false
  · rw [show mf' o = upd m b v o from ite_eq_left hl]
    rcases hy.2.2 o with ⟨i, h₁, _, _⟩ | ⟨_, h₂⟩
    · rw [hl i] at h₁; cases h₁
    · rcases hob with hob | hfo
      · rw [h₂]
        show (if o = b then v else m o) = m o
        rw [ite_eq_right hob]
      · exact absurd (hf.symm.trans h₂) hfo
  · exact ite_eq_right hl

/-! ## Counting -/

/-- The bytes event `v` reads. -/
def bytesRead (v : Ev) : List Loc :=
  if v.e.isRead then (List.range v.size).map fun k => { v.loc with addr := v.loc.addr + k } else []

theorem mem_bytesRead {v : Ev} {b : Loc} (h : readsByte v b) : b ∈ bytesRead v := by
  obtain ⟨hr, hs, ho, h₁, h₂⟩ := h
  rw [bytesRead, ite_eq_left hr]
  refine List.mem_map.2 ⟨b.addr - v.loc.addr, List.mem_range.2 (by omega), ?_⟩
  obtain ⟨sp, ow, ad⟩ := b
  show Loc.mk v.loc.space v.loc.owner (v.loc.addr + (ad - v.loc.addr)) = Loc.mk sp ow ad
  have hs' : v.loc.space = sp := hs
  have ho' : v.loc.owner = ow := ho
  have h₁' : v.loc.addr ≤ ad := h₁
  rw [hs', ho']
  congr 1
  omega

/-- Bytes a candidate's reads move, counted per read. -/
def readBytes {P : Program σ} (c : Candidate P) : Nat :=
  (c.evs.map fun v => if v.e.isRead then v.size else 0).sum

theorem readBytes_ge {P : Program σ} (c : Candidate P) (D : List Loc) (hD : D.Nodup)
    (h : ∀ b ∈ D, Reads c b) : D.length ≤ readBytes c := by
  have hsub : D ⊆ c.evs.flatMap bytesRead := by
    intro b hb
    obtain ⟨v, hv, hr⟩ := h b hb
    exact List.mem_flatMap.2 ⟨v, hv, mem_bytesRead hr⟩
  have := hD.length_le_of_subset hsub
  rw [List.length_flatMap] at this
  unfold readBytes
  have he : (c.evs.map fun v => (bytesRead v).length) = c.evs.map fun v => if v.e.isRead then v.size else 0 := by
    apply List.map_congr_left
    intro v _
    unfold bytesRead
    by_cases hr : v.e.isRead = true <;> simp [hr]
  rw [he] at this
  exact this

/-- **Compulsory traffic.** If `P` computes `f` and each byte of the duplicate-free list `D`
is depended on, at `m`, by an output at another location or one `f` changes, then every
consistent candidate execution of `P` from `m` that finishes reads at least `D.length` bytes. -/
theorem compulsory_traffic {P : Program σ} {pre : (Loc → UInt8) → Prop} {out : Loc → Prop}
    {f : (Loc → UInt8) → Loc → UInt8} {m : Loc → UInt8} {mf : Loc → UInt8}
    (hP : Computes P pre out f) (hm : pre m) (D : List Loc) (hD : D.Nodup)
    (hdep : ∀ b ∈ D, ∃ o, out o ∧ (o ≠ b ∨ f m o ≠ m o) ∧ DependsAt pre f m b o)
    (c : Candidate (withMem P m)) (hv : c.Valid) (hc : c.exec.consistent = true)
    (hy : c.Yields (.done mf)) : D.length ≤ readBytes c :=
  readBytes_ge c D hD fun b hb => by
    obtain ⟨_, ho, hob, hd⟩ := hdep b hb
    exact compulsory hP hm ho hob hd c hv hc hy

/-! ## Compulsory writes -/

/-- **Compulsory writes.** If `P` computes `f` and output byte `o` must change (`f m o ≠ m o`),
then every consistent candidate execution of `P` from `m` that finishes writes `o`: a byte no
write covers keeps its initial value. -/
theorem compulsory_write {P : Program σ} {pre : (Loc → UInt8) → Prop} {out : Loc → Prop}
    {f : (Loc → UInt8) → Loc → UInt8} {m : Loc → UInt8} {o : Loc} {mf : Loc → UInt8}
    (hP : Computes P pre out f) (hm : pre m) (ho : out o) (hne : f m o ≠ m o)
    (c : Candidate (withMem P m)) (hv : c.Valid) (hc : c.exec.consistent = true)
    (hy : c.Yields (.done mf)) : ∃ v ∈ c.evs, v.covers o = true := by
  have hf := hP m hm mf ⟨c, hv, hc, hy⟩ o ho
  rcases hy.2.2 o with ⟨i, h₁, _, _⟩ | ⟨_, h₂⟩
  · exact ⟨c.evs[i], by simp, h₁⟩
  · exact absurd (hf.symm.trans h₂) hne

/-- Bytes a candidate's writes move, counted per write. -/
def writeBytes {P : Program σ} (c : Candidate P) : Nat :=
  (c.evs.map fun v => if v.e.isWrite then v.size else 0).sum

/-- The bytes event `v` writes. -/
def bytesWritten (v : Ev) : List Loc :=
  if v.e.isWrite then (List.range v.size).map fun k => { v.loc with addr := v.loc.addr + k } else []

theorem mem_bytesWritten {v : Ev} {o : Loc} (h : v.covers o = true) : o ∈ bytesWritten v := by
  obtain ⟨hw, hs, ho, h₁, h₂⟩ := NV.covers_iff.1 h
  rw [bytesWritten, ite_eq_left hw]
  refine List.mem_map.2 ⟨o.addr - v.loc.addr, List.mem_range.2 (by omega), ?_⟩
  obtain ⟨sp, ow, ad⟩ := o
  show Loc.mk v.loc.space v.loc.owner (v.loc.addr + (ad - v.loc.addr)) = Loc.mk sp ow ad
  have hs' : v.loc.space = sp := hs
  have ho' : v.loc.owner = ow := ho
  have h₁' : v.loc.addr ≤ ad := h₁
  rw [hs', ho']
  congr 1
  omega

/-- **Compulsory write traffic.** If `P` computes `f` and every byte of the duplicate-free list
`D` is an output `f` changes at `m`, every consistent candidate execution from `m` that finishes
writes at least `D.length` bytes. -/
theorem compulsory_write_traffic {P : Program σ} {pre : (Loc → UInt8) → Prop} {out : Loc → Prop}
    {f : (Loc → UInt8) → Loc → UInt8} {m : Loc → UInt8} {mf : Loc → UInt8}
    (hP : Computes P pre out f) (hm : pre m) (D : List Loc) (hD : D.Nodup)
    (hout : ∀ o ∈ D, out o ∧ f m o ≠ m o)
    (c : Candidate (withMem P m)) (hv : c.Valid) (hc : c.exec.consistent = true)
    (hy : c.Yields (.done mf)) : D.length ≤ writeBytes c := by
  have hsub : D ⊆ c.evs.flatMap bytesWritten := by
    intro o ho
    obtain ⟨v, hv', hcv⟩ := compulsory_write hP hm (hout o ho).1 (hout o ho).2 c hv hc hy
    exact List.mem_flatMap.2 ⟨v, hv', mem_bytesWritten hcv⟩
  have := hD.length_le_of_subset hsub
  rw [List.length_flatMap] at this
  unfold writeBytes
  have he : (c.evs.map fun v => (bytesWritten v).length) =
      c.evs.map fun v => if v.e.isWrite then v.size else 0 := by
    apply List.map_congr_left
    intro v _
    unfold bytesWritten
    by_cases hw : v.e.isWrite = true <;> simp [hw]
  rw [he] at this
  exact this

end NV.Bounds
