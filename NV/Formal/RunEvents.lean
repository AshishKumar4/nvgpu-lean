import NV.Formal.Outcomes

/-!
# The events of a run, act by act

`Program.agentEvents` numbers each thread's events from a counter that every act advances by its
width. Splitting a run splits its events (`agentEvents_app`, `asyncEvents_append`), and each
event of an act lies within the act's positions of its thread (`agentEvents_seq`), so an event of
an earlier act precedes one of a later act in program order (`seq_lt`). These are what a proof
needs to find a program's events and order them without computing their positions.
-/

namespace NV.Formal

variable {σ : Type} (P : Program σ) (a : Agent σ)

omit P in
theorem span_nil (t : Nat) : a.span t [] = 0 := rfl

omit P in
theorem span_cons (t : Nat) (act : Act) (acts : List Act) :
    a.span t (act :: acts) = act.width a t + a.span t acts := by
  simp [Agent.span]

omit P in
theorem span_append (t : Nat) (xs ys : List Act) :
    a.span t (xs ++ ys) = a.span t xs + a.span t ys := by
  simp [Agent.span]

omit P in
@[simp] theorem after_nil (ctr : Nat → Nat) : a.after ctr [] = ctr :=
  funext fun t => by simp [Agent.after, Agent.span]

omit P in
theorem after_cons (ctr : Nat → Nat) (act : Act) (acts : List Act) :
    a.after ctr (act :: acts) = a.after (fun t => ctr t + act.width a t) acts :=
  funext fun t => by simp [Agent.after, span_cons, Nat.add_assoc]

omit P in
theorem after_append (ctr : Nat → Nat) (xs ys : List Act) :
    a.after ctr (xs ++ ys) = a.after (a.after ctr xs) ys :=
  funext fun t => by simp [Agent.after, span_append, Nat.add_assoc]

theorem agentEvents_app : ∀ (ctr : Nat → Nat) (xs ys : List Act),
    P.agentEvents a ctr (xs ++ ys) = P.agentEvents a ctr xs ++ P.agentEvents a (a.after ctr xs) ys
  | ctr, [], ys => by
    rw [show a.after ctr [] = ctr from funext fun t => by simp [Agent.after, Agent.span]]; rfl
  | ctr, act :: xs, ys => by
    simp only [List.cons_append, Program.agentEvents, List.append_assoc]
    rw [agentEvents_app _ xs ys, show a.after (fun t => ctr t + act.width a t) xs =
      a.after ctr (act :: xs) from funext fun t => by simp [Agent.after, span_cons, Nat.add_assoc]]

/-- Every event of an act lies within the act's positions of its thread. -/
theorem actEvents_seq (ctr : Nat → Nat) (act : Act) {e : Ev} (he : e ∈ P.actEvents a ctr act) :
    ctr e.e.thread ≤ e.seq ∧ e.seq < ctr e.e.thread + act.width a e.e.thread := by
  cases act with
  | mem r read =>
    simp only [Program.actEvents, List.mem_flatMap, List.mem_range] at he
    obtain ⟨l, hl, he⟩ := he
    split at he
    · rename_i loc hloc
      obtain ⟨⟨p, i⟩, hi, rfl⟩ := List.mem_map.1 he
      have hil : i < r.pieces.length := by simpa using List.snd_lt_of_mem_zipIdx hi
      have hact : r.active l = true := by
        simp only [Agent.loc, Option.map_eq_some_iff] at hloc
        obtain ⟨_, h, _⟩ := hloc
        simp [Req.active, h]
      simp [Act.width, Req.event, hact]
      omega
    · simp at he
  | fail r read =>
    simp only [Program.actEvents, List.mem_flatMap, List.mem_range] at he
    obtain ⟨l, hl, he⟩ := he
    split at he
    · rename_i loc hloc
      simp only [Req.failEvents, List.mem_filterMap] at he
      obtain ⟨v, hv, he⟩ := he
      split at he
      · cases he
      · cases he
        obtain ⟨⟨p, i⟩, hi, rfl⟩ := List.mem_map.1 hv
        have hil : i < r.pieces.length := by simpa using List.snd_lt_of_mem_zipIdx hi
        have hact : r.active l = true := by
          simp only [Agent.loc, Option.map_eq_some_iff] at hloc
          obtain ⟨_, h, _⟩ := hloc
          simp [Req.active, h]
        simp [Act.width, Req.event, hact]
        omega
    · simp at he
  | sync op inst =>
    simp only [Program.actEvents, List.mem_flatMap, List.mem_range] at he
    obtain ⟨l, hl, he⟩ := he
    cases op with
    | bar id cnt arv =>
      cases arv <;> simp only [SyncOp.events, List.mem_cons, ite_true, ite_false,
        Bool.false_eq_true, List.not_mem_nil, or_false] at he <;>
        (try rcases he with rfl | rfl) <;> (try subst he) <;> simp [Act.width, barEvent, hl]
    | clusterArrive | clusterWait =>
      simp only [SyncOp.events, List.mem_singleton] at he; subst he; simp [Act.width, barEvent, hl]
    | fence sc sco | proxyFence p | aliasFence =>
      simp only [SyncOp.events, List.mem_singleton] at he; subst he; simp [Act.width, fenceEvent, hl]
    | asyncCommit g | asyncWait g k rd => simp [SyncOp.events] at he
  | async op agent => simp [Program.actEvents] at he
  | collect => simp [Program.actEvents] at he

/-- Every event of a run lies within the run's positions of its thread. -/
theorem agentEvents_seq : ∀ (ctr : Nat → Nat) (acts : List Act) {e : Ev},
    e ∈ P.agentEvents a ctr acts →
      ctr e.e.thread ≤ e.seq ∧ e.seq < ctr e.e.thread + a.span e.e.thread acts
  | ctr, [], e, he => by simp [Program.agentEvents] at he
  | ctr, act :: acts, e, he => by
    simp only [Program.agentEvents, List.mem_append] at he
    rw [span_cons]
    rcases he with he | he
    · have := actEvents_seq P a ctr act he; omega
    · have := agentEvents_seq (fun t => ctr t + act.width a t) acts he; omega

/-- **An event of an earlier act precedes one of a later act on the same thread.** -/
theorem seq_lt (ctr : Nat → Nat) (xs ys : List Act) {e₁ e₂ : Ev}
    (h₁ : e₁ ∈ P.agentEvents a ctr xs) (h₂ : e₂ ∈ P.agentEvents a (a.after ctr xs) ys)
    (ht : e₁.e.thread = e₂.e.thread) : e₁.seq < e₂.seq := by
  have := agentEvents_seq P a ctr xs h₁
  have := agentEvents_seq P a _ ys h₂
  simp only [Agent.after] at this
  rw [ht] at *; omega

theorem zipIdx_lt {α : Type} : ∀ (l : List α) (k : Nat), (l.zipIdx k).Pairwise fun x y => x.2 < y.2
  | [], _ => List.Pairwise.nil
  | _ :: l, k => by
    simp only [List.zipIdx_cons, List.pairwise_cons]
    exact ⟨fun y hy => by have := List.le_snd_of_mem_zipIdx hy; omega, zipIdx_lt l (k + 1)⟩

theorem nodup_of_map {α β : Type} (f : α → β) {l : List α} (h : (l.map f).Nodup) : l.Nodup := by
  unfold List.Nodup at *
  rw [List.pairwise_map] at h
  exact h.imp fun h e => h (congrArg f e)

theorem nodup_flatMap {α β : Type} {f : α → List β} :
    ∀ {l : List α}, (∀ a ∈ l, (f a).Nodup) →
      l.Pairwise (fun a b => ∀ x ∈ f a, ∀ y ∈ f b, x ≠ y) → (l.flatMap f).Nodup
  | [], _, _ => List.nodup_nil
  | a :: l, h₁, h₂ => by
    rw [List.flatMap_cons]
    rw [List.pairwise_cons] at h₂
    refine List.nodup_append.2 ⟨h₁ a (List.mem_cons_self ..),
      nodup_flatMap (fun b hb => h₁ b (List.mem_cons_of_mem _ hb)) h₂.2, fun x hx y hy => ?_⟩
    obtain ⟨b, hb, hy⟩ := List.mem_flatMap.1 hy
    exact h₂.1 b hb x hx y hy

/-- A lane's events are distinct: they take consecutive positions. -/
theorem laneEvents_nodup (r : Req) (lane t k : Nat) (l : Loc) (bs : List UInt8) :
    (r.laneEvents lane t k l bs).Nodup := by
  unfold Req.laneEvents List.Nodup
  rw [List.pairwise_map]
  exact (zipIdx_lt r.pieces 0).imp fun h e => by
    have := congrArg Ev.seq e; simp only at this; omega

/-- **A run's events are distinct** when each act's are: acts take disjoint positions. -/
theorem agentEvents_nodup : ∀ (ctr : Nat → Nat) (acts : List Act),
    (∀ act ∈ acts, ∀ ctr, (P.actEvents a ctr act).Nodup) → (P.agentEvents a ctr acts).Nodup
  | _, [], _ => List.nodup_nil
  | ctr, act :: acts, h => by
    simp only [Program.agentEvents]
    refine List.nodup_append.2 ⟨h act (List.mem_cons_self ..) ctr,
      agentEvents_nodup _ acts fun act' h' => h act' (List.mem_cons_of_mem _ h'), ?_⟩
    intro e h₁ e' h₂ he
    subst he
    have := actEvents_seq P a ctr act h₁
    have := agentEvents_seq P a (fun t => ctr t + act.width a t) acts h₂
    omega

/-- What an act leaves of the asynchronous operations' state (`Program.asyncEvents`). -/
def Program.asyncStep (st : AsyncState) : Act → AsyncState
  | .async op agent =>
    let t := P.asyncThread a st.k
    let st' := { st with k := st.k + 1, open_ := st.open_ ++ [(op.group, t, aactWidth agent)] }
    match op with
    | .mma mm => if mm.commit then
        { st' with open_ := st'.open_.filter (·.1 != some .wgmma),
                   committed := st'.committed ++
                     [(.wgmma, (st'.open_.filter (·.1 == some .wgmma)).map (·.2))] }
      else st'
    | _ => st'
  | .sync (.asyncCommit g) _ =>
    { st with open_ := st.open_.filter (·.1 != some g),
              committed := st.committed ++ [(g, (st.open_.filter (·.1 == some g)).map (·.2))] }
  | _ => st

def Program.asyncAfter (st : AsyncState) (acts : List Act) : AsyncState :=
  acts.foldl (P.asyncStep a) st

theorem asyncEvents_append : ∀ (ctr : Nat → Nat) (st : AsyncState) (xs ys : List Act),
    P.asyncEvents a ctr st (xs ++ ys) =
      P.asyncEvents a ctr st xs ++ P.asyncEvents a (a.after ctr xs) (P.asyncAfter a st xs) ys
  | ctr, st, [], ys => by
    rw [show a.after ctr [] = ctr from funext fun t => by simp [Agent.after, Agent.span]]; rfl
  | ctr, st, act :: xs, ys => by
    have hafter : a.after (fun t => ctr t + act.width a t) xs = a.after ctr (act :: xs) :=
      funext fun t => by simp [Agent.after, span_cons, Nat.add_assoc]
    have hst : P.asyncAfter a (P.asyncStep a st act) xs = P.asyncAfter a st (act :: xs) := rfl
    cases act with
    | async op agent =>
      simp only [List.cons_append, Program.asyncEvents, List.append_assoc]
      rw [asyncEvents_append, hafter, ← hst]
      cases op <;> simp [Program.asyncStep]
    | sync op inst =>
      cases op <;> simp only [List.cons_append, Program.asyncEvents, List.append_assoc] <;>
        rw [asyncEvents_append, hafter, ← hst] <;> rfl
    | mem r read =>
      simp only [List.cons_append, Program.asyncEvents]
      rw [asyncEvents_append, hafter, ← hst]; rfl
    | fail r read =>
      simp only [List.cons_append, Program.asyncEvents]
      rw [asyncEvents_append, hafter, ← hst]; rfl
    | collect =>
      simp only [List.cons_append, Program.asyncEvents]
      rw [asyncEvents_append, hafter, ← hst]; rfl

/-- Plain acts leave the asynchronous operations' state as it was. -/
theorem asyncAfter_plain : ∀ (st : AsyncState) (acts : List Act), (∀ act ∈ acts, act.plain = true) →
    P.asyncAfter a st acts = st
  | _, [], _ => rfl
  | st, act :: acts, h => by
    have ha := h act (List.mem_cons_self ..)
    have hst : P.asyncStep a st act = st := by
      cases act with
      | mem r read => rfl
      | fail r read => rfl
      | sync op k => cases op <;> first | rfl | simp [Act.plain] at ha
      | async op agent => simp [Act.plain] at ha
      | collect => rfl
    show P.asyncAfter a (P.asyncStep a st act) acts = st
    rw [hst]; exact asyncAfter_plain st acts fun x hx => h x (List.mem_cons_of_mem _ hx)

/-! ## One-lane accesses -/

/-- A location of the CTA's shared memory, or of global memory. -/
def sloc (a : Nat) : Loc := { space := .shared, owner := 0, addr := a }
def gloc (a : Nat) : Loc := { space := .global, owner := 0, addr := a }

/-- A one-piece access of `size` bytes by thread `t` at position `k`. -/
def oneEv (r : Req) (t k : Nat) (l : Loc) (write : Bool) (size : Nat) (bs : List UInt8) : Ev :=
  { e := r.event t ⟨write, 0, size, false⟩, seq := k, loc := l, size,
    val := if write then (r.written 0 ⟨write, 0, size, false⟩ bs).getD [] else bs.extract 0 size }

theorem one_events {r : Req} {w : Bool} {size : Nat} (hp : r.pieces = [⟨w, 0, size, false⟩])
    (t k : Nat) (l : Loc) (bs : List UInt8) :
    r.laneEvents 0 t k l bs = [oneEv r t k l w size bs] := by
  simp [Req.laneEvents, hp, oneEv]

theorem ofNat_toNat {a : Nat} (h : a < 2 ^ 64) : (UInt64.ofNat a).toNat = a := by
  simp [Nat.mod_eq_of_lt h]

/-- The events of a one-lane request by a one-lane agent of CTA 0. -/
theorem actEvents_one {σ : Type} (P : Program σ) {ag : Agent σ} (hp : ag.place.cta = 0) {r : Req}
    {a : Nat} (ha : r.addrs = #[some (UInt64.ofNat a)]) (hlt : a < 2 ^ 64)
    (hs : r.space = .shared ∨ r.space = .global) (c : Nat → Nat) (read : Nat → List UInt8) :
    P.actEvents ag c (.mem r read) =
      r.laneEvents 0 ag.tid (c ag.tid) { space := r.space, owner := 0, addr := a } (read 0) := by
  simp only [Program.actEvents, ha, Array.size_singleton, List.range_one, List.flatMap_cons,
    List.flatMap_nil, List.append_nil, Nat.add_zero]
  have : ag.loc r 0 = some { space := r.space, owner := 0, addr := a } := by
    simp only [Agent.loc, ha]
    rcases hs with h | h <;> simp [h, hp, ofNat_toNat hlt]
  rw [this]

theorem width_one {σ : Type} {ag : Agent σ} {r : Req} {a : Nat} (ha : r.addrs = #[some (UInt64.ofNat a)])
    (read : Nat → List UInt8) : Act.width ag ag.tid (.mem r read) = r.pieces.length := by
  simp [Act.width, Req.active, ha]

/-- A read of all of a word's bytes returns them. -/
theorem take_all {l : List UInt8} {n : Nat} (h : l.length = n) : l.extract 0 n = l := by
  simp [List.extract, ← h]

/-! ## Dependencies run forward -/

omit P in
theorem span_take_le (t : Nat) (acts : List Act) {n m : Nat} (h : n ≤ m) :
    a.span t (acts.take n) ≤ a.span t (acts.take m) := by
  have : acts.take m = acts.take n ++ (acts.take m).drop n := by
    conv => lhs; rw [← List.take_append_drop n (acts.take m)]
    rw [List.take_take, Nat.min_eq_left h]
  rw [this, span_append]; omega

omit P in
theorem span_take_succ (t : Nat) (acts : List Act) {j : Nat} (hj : j < acts.length) :
    a.span t (acts.take (j + 1)) = a.span t (acts.take j) + acts[j].width a t := by
  rw [List.take_add_one, List.getElem?_eq_getElem hj, Option.toList_some, span_append]
  simp [Agent.span]

/-- **A dependency goes from a read to a later event of its thread.** -/
theorem dep_forward {acts : List Act} {d e : Ev} (h : P.DepIn a acts d e) :
    d.e.isRead = true ∧ d.e.thread = e.e.thread ∧ d.seq < e.seq := by
  obtain ⟨ja, jb, hab, hb, hd, hr, he, ht, -⟩ := h
  refine ⟨hr, ht, ?_⟩
  have h₁ := (actEvents_seq P a _ _ hd).2
  have h₂ := (actEvents_seq P a _ _ he).1
  rw [ht] at h₁
  simp only [Agent.after] at h₁ h₂
  have := span_take_succ a e.e.thread acts (Nat.lt_trans hab hb)
  have := span_take_le a e.e.thread acts (show ja + 1 ≤ jb from hab)
  omega

/-- Every dependency of a candidate's runs is in `readPo`. -/
theorem semDep_readPo {runs : List (List Act × End)} {evs : List Ev} {i j : Fin evs.length}
    (h : P.SemDep runs evs[i] evs[j]) : readPo evs i j = true := by
  obtain ⟨a, acts, en, -, hd⟩ := h
  obtain ⟨h₁, h₂, h₃⟩ := dep_forward P a hd
  simp only [readPo, NV.Rel.get_ofFn, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq]
  exact ⟨⟨h₁, h₂⟩, h₃⟩

end NV.Formal
