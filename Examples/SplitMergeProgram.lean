import Examples.SplitMerge
import NV.Formal.RunEvents
import NV.Memory.Performed

/-!
# The split merge as a program

The merge of `Examples.SplitMerge` written as threads of the general semantics: `K` split threads,
each in its own CTA. Split `j` reduces its partial `part j` into a workspace word with
`red.relaxed.gpu.global.add.u32`, then adds one to a counter with
`atom.acq_rel.gpu.global.add.u32`; the split that reads `K - 1` is the last, loads the workspace
and stores it to the output word.

`merge_correct`: for every `K` and every partial, every outcome the PTX memory model allows is a
completed run whose output word is the sum of the partials (modulo 2³², as `.u32` adds). No
outcome faults or leaves the model. The proof reads each split's run off its code, finds the
counter's and the workspace's atomic chains among the candidate's events (`Chain`, `Counter`),
and uses `merge_sees`: the last split's load is ordered after every reduction by the counter's
release/acquire chain, so it reads the last of them, which holds the whole sum.
-/

namespace Examples.SplitMerge.Program

open NV NV.Formal NV.PTX

/-- Where the merge keeps its three words in global memory. -/
structure Layout where
  ws : UInt64
  ctr : UInt64
  out : UInt64
  apart : ws.toNat + 4 ≤ ctr.toNat ∧ ctr.toNat + 4 ≤ out.toNat

def wordReq (op : MemOp) (sem : NV.Sem) (a : UInt64) (data : Array UInt32) : Req :=
  { space := .global, op, bytes := 4, sem, scope := .gpu, addrs := #[some a], data }

def redReq (L : Layout) (p : UInt32) : Req := wordReq (.red .add .u32) .relaxed L.ws #[p]
def atomReq (L : Layout) : Req := wordReq (.atom .add .u32) .acqRel L.ctr #[1]
def ldReq (L : Layout) : Req := wordReq .ld .weak L.ws #[]
def stReq (L : Layout) (v : UInt32) : Req := wordReq .st .weak L.out #[v]

/-- A split thread: the instruction it is at, its partial, and its two registers. -/
structure S where
  pc : Nat
  part : UInt32
  old : UInt32 := 0
  v : UInt32 := 0

def step (K : Nat) (L : Layout) (s : S) : Effect S :=
  match s.pc with
  | 0 => .mem (fun _ => redReq L s.part) { s with pc := 1 } fun _ t => t
  | 1 => .mem (fun _ => atomReq L) { s with pc := 2 } fun r t => { t with old := r[0]! }
  | 2 => if s.old.toNat + 1 = K then .next { s with pc := 3 } else .exit
  | 3 => .mem (fun _ => ldReq L) { s with pc := 4 } fun r t => { t with v := r[0]! }
  | 4 => .mem (fun _ => stReq L s.v) { s with pc := 5 } fun _ t => t
  | _ => .exit

def agent (part : Nat → UInt32) (j : Nat) : Agent S :=
  { init := { pc := 0, part := part j }, tid := j, place := ⟨j, j, 0⟩ }

def program (K : Nat) (L : Layout) (part : Nat → UInt32) : Program S where
  step := step K L
  agents := (List.range K).map (agent part)
  mem _ := 0

/-! ## What a split does -/

/-- The word a lane's four bytes spell. -/
def word (bs : List UInt8) : UInt32 := UInt32.ofNat (leNat bs)

theorem resp0 {r : Req} {a : UInt64} (hr : r.op.returns = true) (ha : r.addrs = #[some a])
    (hb : r.bytes = 4) {read : Nat → List UInt8} (hl : (read 0).length = 4) :
    (r.resp read)[0]! = word (read 0) := by
  simp [Req.resp, hr, ha, hb, Req.laneResp, words]
  match h : read 0, hl with
  | [b₀, b₁, b₂, b₃], _ => simp [toWords, word]

theorem not_unsupported_red (L : Layout) (p : UInt32) (read : Nat → List UInt8) :
    (redReq L p).unsupported read = false := by
  simp [Req.unsupported, redReq, wordReq, MemOp.writes, Req.active, Req.atomNew, AtomOp.apply]

theorem not_unsupported_atom (L : Layout) (read : Nat → List UInt8) :
    (atomReq L).unsupported read = false := by
  simp [Req.unsupported, atomReq, wordReq, MemOp.writes, Req.active, Req.atomNew, AtomOp.apply]

theorem active0 {op : MemOp} {sem : NV.Sem} {a : UInt64} {d : Array UInt32} :
    (wordReq op sem a d).active 0 = true := rfl

/-- The shape of a split's run: cut short somewhere, or the reduction and the counter update and,
for the split whose update read `K - 1`, the load and the store of what it loaded. -/
theorem run_shape {K : Nat} {L : Layout} {p : UInt32} {acts : List Act} {e : End}
    (h : Run (step K L) { pc := 0, part := p } acts e) :
    e = .stop ∨
      (e = .exit ∧ ∃ r₁ r₂, (r₁ 0).length = 4 ∧ (r₂ 0).length = 4 ∧
        (((word (r₂ 0)).toNat + 1 = K ∧ ∃ r₃ r₄, (r₃ 0).length = 4 ∧
            acts = [.mem (redReq L p) r₁, .mem (atomReq L) r₂, .mem (ldReq L) r₃,
              .mem (stReq L (word (r₃ 0))) r₄]) ∨
          ((word (r₂ 0)).toNat + 1 ≠ K ∧ acts = [.mem (redReq L p) r₁, .mem (atomReq L) r₂]))) := by
  rcases Run.of_mem (s' := { pc := 1, part := p }) rfl h with ⟨-, rfl⟩ | ⟨read, -, -, -, hu⟩ |
    ⟨r₁, rest, rfl, hl₁, -, h⟩
  · exact .inl rfl
  · rw [not_unsupported_red] at hu; cases hu
  rcases Run.of_mem (s' := { pc := 2, part := p }) rfl h with ⟨-, rfl⟩ | ⟨read, -, -, -, hu⟩ |
    ⟨r₂, rest, rfl, hl₂, -, h⟩
  · exact .inl rfl
  · rw [not_unsupported_atom] at hu; cases hu
  have hl₁' := hl₁ 0 active0
  have hl₂' := hl₂ 0 active0
  have hold : ((atomReq L).resp r₂)[0]! = word (r₂ 0) := resp0 rfl rfl rfl hl₂'
  simp only [hold] at h
  by_cases hk : (word (r₂ 0)).toNat + 1 = K
  · rcases Run.of_next (s' := { pc := 3, part := p, old := word (r₂ 0) })
      (by simp [step, hk]) h with ⟨-, rfl⟩ | h
    · exact .inl rfl
    rcases Run.of_mem (s' := { pc := 4, part := p, old := word (r₂ 0) }) rfl h with ⟨-, rfl⟩ |
      ⟨read, -, -, -, hu⟩ | ⟨r₃, rest, rfl, hl₃, -, h⟩
    · exact .inl rfl
    · simp [Req.unsupported, ldReq, wordReq, MemOp.writes] at hu
    have hl₃' : (r₃ 0).length = 4 :=
      (hl₃ 0 active0).trans (by simp [Req.readLen, Req.pieces, Req.granule, Req.strong, ldReq, wordReq])
    have hv : ((ldReq L).resp r₃)[0]! = word (r₃ 0) := resp0 rfl rfl rfl hl₃'
    simp only [hv] at h
    rcases Run.of_mem (s' := { pc := 5, part := p, old := word (r₂ 0), v := word (r₃ 0) }) rfl h
      with ⟨-, rfl⟩ | ⟨read, -, -, -, hu⟩ | ⟨r₄, rest, rfl, -, -, h⟩
    · exact .inl rfl
    · simp [Req.unsupported, stReq, wordReq, MemOp.writes] at hu
    obtain ⟨rfl, he⟩ := Run.of_exit rfl h
    rcases he with rfl | rfl
    · exact .inr ⟨rfl, r₁, r₂, hl₁', hl₂', .inl ⟨hk, r₃, r₄, hl₃', rfl⟩⟩
    · exact .inl rfl
  · obtain ⟨rfl, he⟩ := Run.of_exit (by simp [step, hk]) h
    rcases he with rfl | rfl
    · exact .inr ⟨rfl, r₁, r₂, hl₁', hl₂', .inr ⟨hk, rfl⟩⟩
    · exact .inl rfl

/-! ## The events of a split -/

def wsLoc (L : Layout) : Loc := { space := .global, owner := 0, addr := L.ws.toNat }
def ctrLoc (L : Layout) : Loc := { space := .global, owner := 0, addr := L.ctr.toNat }
def outLoc (L : Layout) : Loc := { space := .global, owner := 0, addr := L.out.toNat }

/-- The read and the write of a four-byte read-modify-write `r` by thread `j`, at positions `s`
and `s + 1`, its read having returned `bs`. -/
def rmwR (r : Req) (l : Loc) (j s : Nat) (bs : List UInt8) : Ev :=
  { e := r.event j ⟨false, 0, 4, true⟩, seq := s, loc := l, size := 4, rmwRead := true,
    val := bs.extract 0 4 }

def rmwW (r : Req) (l : Loc) (j s : Nat) (bs : List UInt8) : Ev :=
  { e := r.event j ⟨true, 0, 4, false⟩, seq := s + 1, loc := l, size := 4,
    val := (r.written 0 ⟨true, 0, 4, false⟩ bs).getD [] }

def rmwEvents (r : Req) (l : Loc) (j s : Nat) (bs : List UInt8) : List Ev :=
  [rmwR r l j s bs, rmwW r l j s bs]

/-- The single event of a four-byte weak `ld` or `st`. -/
def oneEvent (r : Req) (l : Loc) (write : Bool) (j s : Nat) (bs : List UInt8) : Ev :=
  { e := r.event j ⟨write, 0, 4, false⟩, seq := s, loc := l, size := 4,
    val := if write then (r.written 0 ⟨write, 0, 4, false⟩ bs).getD [] else bs.extract 0 4 }

theorem pieces_rmw (L : Layout) (p : UInt32) :
    (redReq L p).pieces = [⟨false, 0, 4, true⟩, ⟨true, 0, 4, false⟩] := rfl
theorem pieces_atom (L : Layout) : (atomReq L).pieces = [⟨false, 0, 4, true⟩, ⟨true, 0, 4, false⟩] := rfl
theorem pieces_ld (L : Layout) : (ldReq L).pieces = [⟨false, 0, 4, false⟩] := by
  simp [Req.pieces, Req.granule, Req.strong, ldReq, wordReq]
theorem pieces_st (L : Layout) (v : UInt32) : (stReq L v).pieces = [⟨true, 0, 4, false⟩] := by
  simp [Req.pieces, Req.granule, Req.strong, stReq, wordReq]

/-- The events of a split that is not the last. -/
theorem events_short {K : Nat} {L : Layout} {part : Nat → UInt32} (j : Nat) (r₁ r₂ : Nat → List UInt8) :
    (program K L part).agentEvents (agent part j) (fun _ => 0)
      [.mem (redReq L (part j)) r₁, .mem (atomReq L) r₂] =
      rmwEvents (redReq L (part j)) (wsLoc L) j 0 (r₁ 0) ++
        rmwEvents (atomReq L) (ctrLoc L) j 2 (r₂ 0) := by
  simp [Program.agentEvents, Program.actEvents, Act.width, Agent.loc, agent, redReq, atomReq,
    wordReq, Req.laneEvents, Req.pieces, rmwEvents, rmwR, rmwW, wsLoc, ctrLoc, Req.active]

/-- The events of the last split. -/
theorem events_last {K : Nat} {L : Layout} {part : Nat → UInt32} (j : Nat)
    (r₁ r₂ r₃ r₄ : Nat → List UInt8) :
    (program K L part).agentEvents (agent part j) (fun _ => 0)
      [.mem (redReq L (part j)) r₁, .mem (atomReq L) r₂, .mem (ldReq L) r₃,
        .mem (stReq L (word (r₃ 0))) r₄] =
      rmwEvents (redReq L (part j)) (wsLoc L) j 0 (r₁ 0) ++
        rmwEvents (atomReq L) (ctrLoc L) j 2 (r₂ 0) ++
        [oneEvent (ldReq L) (wsLoc L) false j 4 (r₃ 0),
         oneEvent (stReq L (word (r₃ 0))) (outLoc L) true j 5 (r₄ 0)] := by
  simp [Program.agentEvents, Program.actEvents, Act.width, Agent.loc, agent, redReq, atomReq,
    ldReq, stReq, wordReq, Req.laneEvents, Req.pieces, Req.granule, Req.strong, rmwEvents,
    rmwR, rmwW, oneEvent, wsLoc, ctrLoc, outLoc, Req.active]
  all_goals rfl

/-! ## Every candidate's runs -/

/-- What one split's run read: the reduction, the counter update and, for the last split, the
load and the store. -/
structure Split where
  r₁ : Nat → List UInt8
  r₂ : Nat → List UInt8
  r₃ : Nat → List UInt8
  r₄ : Nat → List UInt8

instance : Inhabited Split := ⟨⟨fun _ => [], fun _ => [], fun _ => [], fun _ => []⟩⟩

/-- The split read `K - 1` from the counter: it is the last. -/
def Split.last (K : Nat) (x : Split) : Prop := (word (x.r₂ 0)).toNat + 1 = K

instance (K : Nat) (x : Split) : Decidable (x.last K) := inferInstanceAs (Decidable (_ = _))

def Split.acts (K : Nat) (L : Layout) (p : UInt32) (x : Split) : List Act :=
  if x.last K then
    [.mem (redReq L p) x.r₁, .mem (atomReq L) x.r₂, .mem (ldReq L) x.r₃,
      .mem (stReq L (word (x.r₃ 0))) x.r₄]
  else [.mem (redReq L p) x.r₁, .mem (atomReq L) x.r₂]

/-- The events of split `j`'s run. -/
def Split.events (K : Nat) (L : Layout) (part : Nat → UInt32) (j : Nat) (x : Split) : List Ev :=
  rmwEvents (redReq L (part j)) (wsLoc L) j 0 (x.r₁ 0) ++
    rmwEvents (atomReq L) (ctrLoc L) j 2 (x.r₂ 0) ++
    if x.last K then
      [oneEvent (ldReq L) (wsLoc L) false j 4 (x.r₃ 0),
       oneEvent (stReq L (word (x.r₃ 0))) (outLoc L) true j 5 (x.r₄ 0)]
    else []

theorem agentEvents_split {K : Nat} {L : Layout} {part : Nat → UInt32} (j : Nat) (x : Split) :
    (program K L part).agentEvents (agent part j) (fun _ => 0) (x.acts K L (part j)) =
      x.events K L part j := by
  unfold Split.acts Split.events
  split
  · rw [events_last]
  · rw [events_short]; simp

section
variable {K : Nat} {L : Layout} {part : Nat → UInt32} (c : Candidate (program K L part))

theorem runs_length : c.runs.length = K := by
  rw [c.length]; simp [program]

theorem run_of {j : Nat} (hj : j < K) :
    Run (step K L) { pc := 0, part := part j } (c.runs[j]'(by rw [runs_length]; exact hj)).1
      (c.runs[j]'(by rw [runs_length]; exact hj)).2 := by
  have hz : ((program K L part).agents.zip c.runs)[j]'(by simp [runs_length, program]; exact hj) =
      ((program K L part).agents[j]'(by simp [program]; exact hj),
        c.runs[j]'(by rw [runs_length]; exact hj)) := List.getElem_zip
  have := c.runs_ok _ (hz ▸ List.getElem_mem _)
  simpa [program, agent] using this

/-- Every run ends by exiting, having done what its code says. -/
theorem splits (hv : c.Valid) {j : Nat} (hj : j < K) :
    ∃ x : Split, c.runs[j]'(by rw [runs_length]; exact hj) = (x.acts K L (part j), .exit) ∧
      (x.r₁ 0).length = 4 ∧ (x.r₂ 0).length = 4 ∧ (x.last K → (x.r₃ 0).length = 4) := by
  -- No run is cut short: that needs one that faulted or left the model.
  have hend : ∀ re ∈ c.runs, re.2 = .stop ∨ re.2 = .exit := by
    intro re hre
    obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hre
    rcases run_shape (run_of c (by rw [runs_length] at hi; exact hi)) with h | ⟨h, -⟩
    · exact .inl h
    · exact .inr h
  have hnostop : ∀ re ∈ c.runs, re.2 ≠ .stop := fun re hre hs => by
    obtain ⟨re', hre', h₁, h₂⟩ := hv.2 ⟨re, hre, hs⟩
    rcases hend re' hre' with h | h
    · exact h₂ h
    · exact h₁ h
  rcases run_shape (run_of c hj) with h | ⟨h, r₁, r₂, h₁, h₂, h₃⟩
  · exact absurd h (hnostop _ (List.getElem_mem _))
  · rcases h₃ with ⟨hk, r₃, r₄, h₄, ha⟩ | ⟨hk, ha⟩
    · refine ⟨⟨r₁, r₂, r₃, r₄⟩, ?_, h₁, h₂, fun _ => h₄⟩
      ext1 <;> simp [Split.acts, Split.last, hk, ha, h]
    · refine ⟨⟨r₁, r₂, fun _ => [], fun _ => []⟩, ?_, h₁, h₂, fun hl => absurd hl hk⟩
      ext1 <;> simp [Split.acts, Split.last, hk, ha, h]

variable (hv : c.Valid)

/-- What split `j` read, in the candidate. -/
noncomputable def spl (j : Nat) : Split :=
  if h : j < K then Classical.choose (splits c hv h) else default

theorem spl_spec {j : Nat} (hj : j < K) :
    c.runs[j]'(by rw [runs_length]; exact hj) = ((spl c hv j).acts K L (part j), .exit) ∧
      ((spl c hv j).r₁ 0).length = 4 ∧ ((spl c hv j).r₂ 0).length = 4 ∧
      ((spl c hv j).last K → ((spl c hv j).r₃ 0).length = 4) := by
  unfold spl; rw [dite_eq_left hj]; exact Classical.choose_spec (splits c hv hj)

omit c in
theorem thread_of {j : Nat} {x : Split} {e : Ev} (h : e ∈ x.events K L part j) : e.e.thread = j := by
  have : ∀ e ∈ x.events K L part j, e.e.thread = j := by
    by_cases hl : x.last K <;> simp [Split.events, hl, rmwEvents, rmwR, rmwW, oneEvent, Req.event]
  exact this e h

theorem events_eq :
    (program K L part).events c.runs = (List.range K).flatMap fun j => (spl c hv j).events K L part j := by
  have hz : (program K L part).agents.zip c.runs =
      (List.range K).map fun j => (agent part j, ((spl c hv j).acts K L (part j), End.exit)) := by
    apply List.ext_getElem
    · simp [runs_length, program]
    · intro i h₁ h₂
      have hi : i < K := by simpa using h₂
      rw [List.getElem_zip, List.getElem_map, (spl_spec c hv hi).1]
      simp [program, List.getElem_range]
  unfold Program.events
  rw [hz, List.flatMap_map]
  congr 1
  funext j
  dsimp only
  rw [Program.asyncEvents_plain _ _ _ _ _ (fun act ha => by
    unfold Split.acts at ha; split at ha <;> simp at ha <;>
      rcases ha with rfl | rfl | rfl | rfl <;> rfl), List.append_nil]
  exact agentEvents_split j _

theorem mem_evs {e : Ev} : e ∈ c.evs ↔ ∃ j < K, e ∈ (spl c hv j).events K L part j := by
  rw [c.perm.mem_iff, events_eq c hv]
  simp

omit c in
theorem nodup_flatMap {α β : Type} {f : α → List β} :
    ∀ {l : List α}, (∀ a ∈ l, (f a).Nodup) → l.Pairwise (fun a b => ∀ e ∈ f a, e ∉ f b) →
      (l.flatMap f).Nodup
  | [], _, _ => by simp
  | a :: l, h₁, h₂ => by
    rw [List.flatMap_cons, List.nodup_append]
    refine ⟨h₁ a (by simp), nodup_flatMap (fun b hb => h₁ b (by simp [hb])) (List.pairwise_cons.1 h₂).2,
      fun e he e' he' hee => ?_⟩
    subst hee
    obtain ⟨b, hb, hb'⟩ := List.mem_flatMap.1 he'
    exact (List.pairwise_cons.1 h₂).1 b hb e he hb'

include hv in
theorem nodup_evs : c.evs.Nodup := by
  rw [c.perm.nodup_iff, events_eq c hv]
  refine nodup_flatMap (fun j _ => ?_) ?_
  · by_cases hl : (spl c hv j).last K <;> simp [Split.events, hl, rmwEvents, rmwR, rmwW, oneEvent]
  · apply List.pairwise_lt_range.imp
    intro a b hab e ha hb
    exact absurd ((thread_of ha).symm.trans (thread_of hb)) (Nat.ne_of_lt hab)

end

/-! ## The counter's and the workspace's chains -/

section
variable {K : Nat} {L : Layout} {part : Nat → UInt32} (c : Candidate (program K L part))
  (hv : c.Valid)

theorem place_gpu (t : Nat) : ((program K L part).place t).gpu = 0 :=
  Program.place_all (f := fun p => p.gpu = 0) (fun a ha => by
    simp only [program, List.mem_map, List.mem_range] at ha
    obtain ⟨j, _, rfl⟩ := ha
    rfl) rfl t

/-- The position of an event of the candidate. -/
def ix {e : Ev} (h : e ∈ c.evs) : Fin c.evs.length := ⟨c.evs.idxOf e, List.idxOf_lt_length_of_mem h⟩

theorem ix_get {e : Ev} (h : e ∈ c.evs) : c.evs[ix c h] = e := List.getElem_idxOf _

theorem ix_get' {e : Ev} (h : e ∈ c.evs) : c.evs[(ix c h).val] = e := List.getElem_idxOf _

include hv in
theorem ix_of {e : Ev} (h : e ∈ c.evs) {i : Fin c.evs.length} (hi : c.evs[i] = e) : ix c h = i := by
  apply Fin.ext
  have h₁ : c.evs[(ix c h).val] = c.evs[i.val] := (ix_get c h).trans hi.symm
  exact (nodup_evs c hv).getElem_inj.1 h₁

noncomputable def redR (j : Nat) : Ev := rmwR (redReq L (part j)) (wsLoc L) j 0 ((spl c hv j).r₁ 0)
noncomputable def redW (j : Nat) : Ev := rmwW (redReq L (part j)) (wsLoc L) j 0 ((spl c hv j).r₁ 0)
noncomputable def ctrR (j : Nat) : Ev := rmwR (atomReq L) (ctrLoc L) j 2 ((spl c hv j).r₂ 0)
noncomputable def ctrW (j : Nat) : Ev := rmwW (atomReq L) (ctrLoc L) j 2 ((spl c hv j).r₂ 0)
noncomputable def ldE (j : Nat) : Ev := oneEvent (ldReq L) (wsLoc L) false j 4 ((spl c hv j).r₃ 0)
noncomputable def stE (j : Nat) : Ev :=
  oneEvent (stReq L (word ((spl c hv j).r₃ 0))) (outLoc L) true j 5 ((spl c hv j).r₄ 0)

theorem mem_split {j : Nat} (hj : j < K) {e : Ev} (he : e ∈ (spl c hv j).events K L part j) :
    e ∈ c.evs := (mem_evs c hv).2 ⟨j, hj, he⟩

theorem mem_redR {j : Nat} (hj : j < K) : redR c hv j ∈ c.evs :=
  mem_split c hv hj (by simp [Split.events, rmwEvents, redR])
theorem mem_redW {j : Nat} (hj : j < K) : redW c hv j ∈ c.evs :=
  mem_split c hv hj (by simp [Split.events, rmwEvents, redW])
theorem mem_ctrR {j : Nat} (hj : j < K) : ctrR c hv j ∈ c.evs :=
  mem_split c hv hj (by simp [Split.events, rmwEvents, ctrR])
theorem mem_ctrW {j : Nat} (hj : j < K) : ctrW c hv j ∈ c.evs :=
  mem_split c hv hj (by simp [Split.events, rmwEvents, ctrW])
theorem mem_ldE {j : Nat} (hj : j < K) (hl : (spl c hv j).last K) : ldE c hv j ∈ c.evs :=
  mem_split c hv hj (by simp [Split.events, hl, ldE])
theorem mem_stE {j : Nat} (hj : j < K) (hl : (spl c hv j).last K) : stE c hv j ∈ c.evs :=
  mem_split c hv hj (by simp [Split.events, hl, stE])

variable (hK : 0 < K)

/-- The position of split `j`'s event `ev j` (split 0's for `j ≥ K`). -/
noncomputable def at_ (ev : Nat → Ev) (hm : ∀ {j}, j < K → ev j ∈ c.evs) (j : Nat) : Fin c.evs.length :=
  if hj : j < K then ix c (hm hj) else ix c (hm hK)

theorem at_get {ev : Nat → Ev} {hm : ∀ {j}, j < K → ev j ∈ c.evs} {j : Nat} (hj : j < K) :
    c.evs[(at_ c hK ev hm j).val] = ev j := by
  unfold at_
  split
  · exact ix_get c _
  · contradiction

/-- An atomic add of `p` to four bytes `bs` writes the bytes of their word plus `p`. -/
theorem red_written {p : UInt32} {bs : List UInt8} (hl : bs.length = 4) :
    (redReq L p).written 0 ⟨true, 0, 4, false⟩ bs = some (wordBytes (word bs + p)) := by
  match bs, hl with
  | [b₀, b₁, b₂, b₃], _ =>
    simp [Req.written, Req.atomNew, Req.operands, redReq, wordReq, AtomOp.apply, toWords, word,
      toBytes, words, wordBytes]

theorem atom_written {bs : List UInt8} (hl : bs.length = 4) :
    (atomReq L).written 0 ⟨true, 0, 4, false⟩ bs = some (wordBytes (word bs + 1)) := by
  match bs, hl with
  | [b₀, b₁, b₂, b₃], _ =>
    simp [Req.written, Req.atomNew, Req.operands, atomReq, wordReq, AtomOp.apply, toWords, word,
      toBytes, words, wordBytes]

theorem extract_four {bs : List UInt8} (hl : bs.length = 4) : bs.extract 0 4 = bs := by
  simp [List.take_of_length_le (Nat.le_of_eq hl)]

theorem leNat_four {bs : List UInt8} (hl : bs.length = 4) : leNat bs < 2 ^ 32 := by
  have := leNat_lt (bs := bs); rw [hl] at this; exact this

theorem word_toNat {bs : List UInt8} (hl : bs.length = 4) : (word bs).toNat = leNat bs :=
  UInt32.toNat_ofNat_of_lt' (by have := leNat_four hl; exact this)

theorem add_dec {bs : List UInt8} (hl : bs.length = 4) (p : UInt32) :
    leNat (wordBytes (word bs + p)) = (leNat bs + p.toNat) % 2 ^ 32 := by
  rw [leNat_wordBytes, UInt32.toNat_add, word_toNat hl]

theorem wsLoc_ne_ctrLoc : wsLoc L ≠ ctrLoc L := fun h => by
  have := congrArg Loc.addr h; simp [wsLoc, ctrLoc] at this; have := L.apart; omega
theorem wsLoc_ne_outLoc : wsLoc L ≠ outLoc L := fun h => by
  have := congrArg Loc.addr h; simp [wsLoc, outLoc] at this; have := L.apart; omega
theorem ctrLoc_ne_outLoc : ctrLoc L ≠ outLoc L := fun h => by
  have := congrArg Loc.addr h; simp [ctrLoc, outLoc] at this; have := L.apart; omega

/-- The workspace: split `j`'s reduction adds `part j`. -/
noncomputable def wsChain : Chain c.exec K leNat (2 ^ 32) where
  rd := at_ c hK (redR c hv) (mem_redR c hv)
  wr := at_ c hK (redW c hv) (mem_redW c hv)
  mine j := (part j).toNat
  addr := code c.evs (redR c hv 0)
  rmw j hj := by
    simp [Candidate.exec, toExec, at_get c hK hj, redR, redW, rmwR, rmwW, Req.event]
  reads j hj := by
    simp [Candidate.exec, toExec, Execution.reads, at_get c hK hj, redR, rmwR, Req.event,
      Event.isRead]
  writes j hj := by
    simp [Candidate.exec, toExec, Execution.writes, at_get c hK hj, redW, rmwW, Req.event,
      Event.isWrite]
  address j hj := by
    simp [Candidate.exec, toExec, at_get c hK hj, redR, redW, rmwR, rmwW, code, Ev.key]
  generic j hj := by
    simp [Candidate.exec, toExec, Execution.generic, at_get c hK hj, redR, redW, rmwR, rmwW,
      Req.event, Event.isMem, Event.isRead, Event.isWrite, redReq, wordReq]
  strong j hj l hl := by
    simp [Execution.strongR, Candidate.exec, toExec, at_get c hK hj, at_get c hK hl, redR, redW,
      rmwR, rmwW, Req.event, Req.eventScope, redReq, wordReq, Scope.contains, Execution.placeOf,
      Execution.thr, place_gpu]
  distinct j hj l hl h := by
    have := congrArg (fun i : Fin c.evs.length => c.evs[i].e.thread) h
    simpa [at_get c hK hj, at_get c hK hl, redW, rmwW, Req.event] using this
  only e he hloc := by
    have hk : c.evs[e].key = (wsLoc L, 4) := by
      have := (Performed.code_eq (List.getElem_mem _) (mem_redR c hv hK)).1 hloc
      simpa [redR, rmwR, Ev.key] using this
    have hw : c.evs[e].e.isWrite = true := he
    obtain ⟨j, hj, hej⟩ := (mem_evs c hv).1 (List.getElem_mem e.isLt)
    have key_of : ∀ {v : Ev}, c.evs[e] = v → v.key = (wsLoc L, 4) := fun h => h ▸ hk
    have write_of : ∀ {v : Ev}, c.evs[e] = v → v.e.isWrite = true := fun h => h ▸ hw
    refine ⟨j, hj, ?_⟩
    have hall : ∀ v ∈ (spl c hv j).events K L part j, v.e.isWrite = true → v.key = (wsLoc L, 4) →
        v = redW c hv j := by
      have h₁ := wsLoc_ne_ctrLoc (L := L)
      have h₂ := wsLoc_ne_outLoc (L := L)
      by_cases hl : (spl c hv j).last K <;>
        simp [Split.events, hl, rmwEvents, rmwR, rmwW, oneEvent, Req.event, Event.isWrite, Ev.key,
          redW, Ne.symm h₁, Ne.symm h₂]
    have hred : c.evs[e] = redW c hv j := hall _ hej hw hk
    show e = at_ c hK (redW c hv) (mem_redW c hv) j
    unfold at_
    rw [dite_eq_left hj]
    exact (ix_of c hv _ hred).symm
  zero := by
    have hm := mem_redR c hv hK
    have hlt := List.idxOf_lt_length_of_mem (List.mem_map_of_mem (f := Ev.key) hm)
    have hget : (c.evs.map Ev.key)[code c.evs (redR c hv 0)]? = some (wsLoc L, 4) := by
      unfold code
      rw [List.getElem?_eq_getElem hlt, List.getElem_idxOf hlt]
      rfl
    show leNat (match (c.evs.map Ev.key)[code c.evs (redR c hv 0)]? with
      | some key => initBytes (program K L part).mem key | none => []) = 0
    rw [hget]
    rfl
  add j hj := by
    obtain ⟨-, h₁, -⟩ := spl_spec c hv hj
    simp only [Candidate.exec, toExec, Fin.getElem_fin, at_get c hK hj, redR, redW, rmwR, rmwW]
    rw [red_written h₁, Option.getD_some, add_dec h₁, extract_four h₁]

/-- The counter: split `j`'s `atom.acq_rel` adds one. -/
noncomputable def ctrCounter : Counter c.exec K leNat (2 ^ 32) where
  rd := at_ c hK (ctrR c hv) (mem_ctrR c hv)
  wr := at_ c hK (ctrW c hv) (mem_ctrW c hv)
  mine _ := 1
  addr := code c.evs (ctrR c hv 0)
  rmw j hj := by
    simp [Candidate.exec, toExec, at_get c hK hj, ctrR, ctrW, rmwR, rmwW, Req.event]
  reads j hj := by
    simp [Candidate.exec, toExec, Execution.reads, at_get c hK hj, ctrR, rmwR, Req.event,
      Event.isRead]
  writes j hj := by
    simp [Candidate.exec, toExec, Execution.writes, at_get c hK hj, ctrW, rmwW, Req.event,
      Event.isWrite]
  address j hj := by
    simp [Candidate.exec, toExec, at_get c hK hj, ctrR, ctrW, rmwR, rmwW, code, Ev.key]
  generic j hj := by
    simp [Candidate.exec, toExec, Execution.generic, at_get c hK hj, ctrR, ctrW, rmwR, rmwW,
      Req.event, Event.isMem, Event.isRead, Event.isWrite, atomReq, wordReq]
  strong j hj l hl := by
    simp [Execution.strongR, Candidate.exec, toExec, at_get c hK hj, at_get c hK hl, ctrR, ctrW,
      rmwR, rmwW, Req.event, Req.eventScope, atomReq, wordReq, Scope.contains, Execution.placeOf,
      Execution.thr, place_gpu]
  distinct j hj l hl h := by
    have := congrArg (fun i : Fin c.evs.length => c.evs[i].e.thread) h
    simpa [at_get c hK hj, at_get c hK hl, ctrW, rmwW, Req.event] using this
  only e he hloc := by
    have hk : c.evs[e].key = (ctrLoc L, 4) := by
      have := (Performed.code_eq (List.getElem_mem _) (mem_ctrR c hv hK)).1 hloc
      simpa [ctrR, rmwR, Ev.key] using this
    have hw : c.evs[e].e.isWrite = true := he
    obtain ⟨j, hj, hej⟩ := (mem_evs c hv).1 (List.getElem_mem e.isLt)
    have key_of : ∀ {v : Ev}, c.evs[e] = v → v.key = (ctrLoc L, 4) := fun h => h ▸ hk
    have write_of : ∀ {v : Ev}, c.evs[e] = v → v.e.isWrite = true := fun h => h ▸ hw
    refine ⟨j, hj, ?_⟩
    have hall : ∀ v ∈ (spl c hv j).events K L part j, v.e.isWrite = true → v.key = (ctrLoc L, 4) →
        v = ctrW c hv j := by
      have h₁ := wsLoc_ne_ctrLoc (L := L)
      have h₂ := ctrLoc_ne_outLoc (L := L)
      by_cases hl : (spl c hv j).last K <;>
        simp [Split.events, hl, rmwEvents, rmwR, rmwW, oneEvent, Req.event, Event.isWrite, Ev.key,
          ctrW, h₁, Ne.symm h₂]
    have hred : c.evs[e] = ctrW c hv j := hall _ hej hw hk
    show e = at_ c hK (ctrW c hv) (mem_ctrW c hv) j
    unfold at_
    rw [dite_eq_left hj]
    exact (ix_of c hv _ hred).symm
  zero := by
    have hm := mem_ctrR c hv hK
    have hlt := List.idxOf_lt_length_of_mem (List.mem_map_of_mem (f := Ev.key) hm)
    have hget : (c.evs.map Ev.key)[code c.evs (ctrR c hv 0)]? = some (ctrLoc L, 4) := by
      unfold code
      rw [List.getElem?_eq_getElem hlt, List.getElem_idxOf hlt]
      rfl
    show leNat (match (c.evs.map Ev.key)[code c.evs (ctrR c hv 0)]? with
      | some key => initBytes (program K L part).mem key | none => []) = 0
    rw [hget]
    rfl
  add j hj := by
    obtain ⟨-, -, h₁, -⟩ := spl_spec c hv hj
    simp only [Candidate.exec, toExec, Fin.getElem_fin, at_get c hK hj, ctrR, ctrW, rmwR, rmwW]
    rw [atom_written h₁, Option.getD_some, add_dec h₁, extract_four h₁]
    rfl
  acquire j hj := by
    simp [Candidate.exec, toExec, at_get c hK hj, ctrR, rmwR, Req.event, Req.acquires, Req.isRed,
      atomReq, wordReq]
  release j hj := by
    simp [Candidate.exec, toExec, at_get c hK hj, ctrW, rmwW, Req.event, Req.releases, atomReq,
      wordReq]
  pos _ _ := Nat.one_pos

end

/-- The sum of the partials, as `.u32` adds compute it. -/
def total (K : Nat) (part : Nat → UInt32) : UInt32 := UInt32.ofNat (sumBelow K fun j => (part j).toNat)

theorem st_written {L : Layout} (v : UInt32) (bs : List UInt8) :
    (stReq L v).written 0 ⟨true, 0, 4, false⟩ bs = some (wordBytes v) := by
  simp [Req.written, Req.storeBytes, Req.operands, stReq, wordReq, toBytes, words, wordBytes]

theorem word_wordBytes (w : UInt32) : word (wordBytes w) = w := by
  rw [word, leNat_wordBytes, UInt32.ofNat_toNat]

/-! ## The last split loads the sum -/

theorem exists_max (f : Nat → Nat) : ∀ {K : Nat}, 0 < K → ∃ j < K, ∀ l < K, f l ≤ f j
  | 0, h => absurd h (Nat.lt_irrefl _)
  | K + 1, _ => by
    by_cases hK : K = 0
    · subst hK; exact ⟨0, by omega, fun l hl => by rw [show l = 0 by omega]; exact Nat.le_refl _⟩
    · obtain ⟨j, hj, hmax⟩ := exists_max f (Nat.pos_of_ne_zero hK)
      by_cases hle : f K ≤ f j
      · exact ⟨j, by omega, fun l hl => by
          rcases Nat.lt_or_ge l K with h | h
          · exact hmax l h
          · rw [show l = K by omega]; exact hle⟩
      · exact ⟨K, by omega, fun l hl => by
          rcases Nat.lt_or_ge l K with h | h
          · exact Nat.le_trans (hmax l h) (by omega)
          · rw [show l = K by omega]; exact Nat.le_refl _⟩

theorem sum_ind (n : Nat) (p : Nat → Bool) :
    sumBelow n (fun l => if p l then 1 else 0) = countBelow n p := by
  induction n with
  | zero => rfl
  | succ n ih => rw [sumBelow_succ, countBelow_succ, ih]

theorem countBelow_lt {n : Nat} {p q : Nat → Bool} (h : ∀ j < n, p j = true → q j = true)
    {i : Nat} (hi : i < n) (hq : q i = true) (hp : p i = false) : countBelow n p < countBelow n q :=
  NV.Rel.countP_lt_of_imp (fun a ha hpa => h a (List.mem_range.1 ha) hpa) (List.mem_range.2 hi) hq hp

theorem countBelow_all_but {n i : Nat} (hi : i < n) {p : Nat → Bool}
    (h : ∀ j < n, p j = (j != i)) : countBelow n p = n - 1 := by
  have h₁ : countBelow n (fun _ => true) = n := (countBelow_eq_iff _ _).2 fun _ _ => rfl
  have := countBelow_update (p := fun _ => true) (q := p) hi (fun j hj hji => by simp [h j hj, hji])
  rw [h₁, h i hi] at this
  simp at this
  omega

/-- In a chain where every update but `j₀`'s comes before `j₀`'s, `j₀`'s counts all others. -/
theorem before_all {n : Nat} {V : Type} [DecidableEq V] {x : Execution n V} {K : Nat} {dec : V → Nat}
    {M : Nat} (C : Chain x K dec M) (hc : x.consistent = true) {j₀ : Nat} (hj₀ : j₀ < K)
    (h : ∀ l < K, l ≠ j₀ → before C j₀ l = true) : ∀ l < K, before C j₀ l = (l != j₀) := by
  intro l hl
  by_cases e : l = j₀
  · subst e; simp [before, Execution.co_irrefl hc]
  · simp [h l hl e, e]

section
variable {K : Nat} {L : Layout} {part : Nat → UInt32} (c : Candidate (program K L part))
  (hv : c.Valid) (hK : 0 < K) (hc : c.exec.consistent = true)
include hK hc

/-- Some split reads `K - 1` from the counter: the one whose update comes after all others. -/
theorem last_exists (hK' : K < 2 ^ 32) : ∃ m < K, (spl c hv m).last K := by
  let C := ctrCounter c hv hK
  obtain ⟨jm, hjm, hmax⟩ := exists_max (fun j => countBelow K (before C.toChain j)) hK
  have hall : ∀ l < K, l ≠ jm → before C.toChain jm l = true := by
    intro l hl hne
    cases hb : before C.toChain jm l
    · exfalso
      have hne' : C.wr l ≠ C.wr jm := fun e => hne (C.distinct l hl jm hjm e)
      rcases Execution.co_total hc hne' (C.writes l hl) (C.writes jm hjm)
        (by rw [(C.address l hl).2, (C.address jm hjm).2]) (C.strong l hl jm hjm).2.1 with h | h
      · simp [before, h] at hb
      · -- `jm` before `l`: then `l` counts strictly more.
        have := countBelow_lt (p := before C.toChain jm) (q := before C.toChain l)
          (fun l' _ h' => Execution.co_trans hc h' h) hjm h (Execution.co_irrefl hc _)
        exact absurd (hmax l hl) (Nat.not_le.2 this)
    · rfl
  have hcount : countBelow K (before C.toChain jm) = K - 1 :=
    countBelow_all_but hjm (before_all C.toChain hc hjm hall)
  obtain ⟨hval, -⟩ := chain hc C.toChain _ jm hjm rfl
  rw [show (fun l => if before C.toChain jm l = true then C.mine l else 0) =
    (fun l => if before C.toChain jm l = true then 1 else 0) from rfl, sum_ind, hcount,
    Nat.mod_eq_of_lt (by omega)] at hval
  refine ⟨jm, hjm, ?_⟩
  obtain ⟨-, -, h₂, -⟩ := spl_spec c hv hjm
  have : c.exec.val (C.rd jm) = (spl c hv jm).r₂ 0 := by
    show c.evs[(C.rd jm).val].val = _
    simp only [C, ctrCounter, at_get c hK hjm, ctrR, rmwR]
    exact extract_four h₂
  rw [this] at hval
  show (word ((spl c hv jm).r₂ 0)).toNat + 1 = K
  rw [word_toNat h₂, hval]
  omega

omit hK hc in
theorem ofNat_mod (n : Nat) : UInt32.ofNat (n % 2 ^ 32) = UInt32.ofNat n := by
  apply UInt32.toNat_inj.1
  simp [UInt32.toNat_ofNat']

/-- **A last split stores the sum.** Its load follows every reduction through the counter's
release/acquire chain (`merge_sees`), so it reads the reduction coherence-last, which adds every
partial. -/
theorem store_value (hK' : K < 2 ^ 32) {m : Nat} (hm : m < K) (hl : (spl c hv m).last K) :
    (stE c hv m).val = wordBytes (total K part) := by
  let C := ctrCounter c hv hK
  let W := wsChain c hv hK
  have hm₁ := mem_ldE c hv hm hl
  have hr : c.evs[(ix c hm₁).val] = ldE c hv m := ix_get' c hm₁
  obtain ⟨-, -, h₂m, h₃m⟩ := spl_spec c hv hm
  have hone : sumBelow K C.mine = K := by
    rw [show C.mine = fun l => if true then 1 else 0 from rfl, sum_ind,
      (countBelow_eq_iff _ _).2 (fun _ _ => rfl)]
  have hlast : leNat (c.exec.val (C.rd m)) + C.mine m = sumBelow K C.mine := by
    rw [hone]
    show leNat c.evs[(C.rd m).val].val + 1 = K
    simp only [C, ctrCounter, at_get c hK hm, ctrR, rmwR]
    rw [extract_four h₂m, ← word_toNat h₂m]
    exact hl
  have hreads : c.exec.reads (ix c hm₁) = true := by
    simp [Candidate.exec, toExec, Execution.reads, ix_get', ldE, oneEvent, Req.event, Event.isRead]
  have haddr : ∀ j < K, (c.exec.ev (W.wr j)).addr = (c.exec.ev (ix c hm₁)).addr := fun j hj => by
    simp [Candidate.exec, toExec, W, wsChain, at_get c hK hj, ix_get', redW, rmwW, ldE, oneEvent,
      code, Ev.key]
  -- What every reduction's publication gives.
  have hsees : ∀ j < K, (∃ w, c.exec.rf w (ix c hm₁) = true) ∧
      ∀ w, c.exec.rf w (ix c hm₁) = true → c.exec.co w (W.wr j) = false := by
    intro j hj
    refine merge_sees hc C hm (by rw [hone]; exact hK') hlast hj (W.writes j hj) (W.generic j hj).2
      ?_ hreads ?_ (haddr j hj) ?_
    · apply Execution.causeBase_of_order
      apply Rel.tc_of
      simp [Execution.order, Candidate.exec, toExec, W, C, wsChain, ctrCounter, at_get c hK hj,
        redW, ctrW, rmwW, Req.event]
    · simp [Candidate.exec, toExec, Execution.generic, ix_get', ldE, oneEvent, Req.event,
        Event.isMem, Event.isRead, ldReq, wordReq]
    · apply Rel.rtc_of_tc
      apply Rel.tc_of
      simp [Execution.order, Candidate.exec, toExec, C, ctrCounter, at_get c hK hm, ix_get', ctrW,
        rmwW, ldE, oneEvent, Req.event]
  obtain ⟨⟨w₀, hw₀⟩, -⟩ := hsees 0 hK
  rcases NV.PTX.Execution.rf_cases hc hreads with ⟨hn, -⟩ | ⟨w, hrf, hval, hw, hloc⟩
  · rw [hn w₀] at hw₀; cases hw₀
  obtain ⟨j₀, hj₀, rfl⟩ := W.only w hw (by rw [hloc, ← haddr 0 hK, (W.address 0 hK).2])
  -- Every other reduction comes before `j₀`'s.
  have hall : ∀ l < K, l ≠ j₀ → before W j₀ l = true := by
    intro l hl hne
    have hne' : W.wr l ≠ W.wr j₀ := fun e => hne (W.distinct l hl j₀ hj₀ e)
    rcases Execution.co_total hc hne' (W.writes l hl) (W.writes j₀ hj₀)
      (by rw [(W.address l hl).2, (W.address j₀ hj₀).2]) (W.strong l hl j₀ hj₀).2.1 with h | h
    · exact h
    · rw [(hsees l hl).2 _ hrf] at h; cases h
  have hb := before_all W hc hj₀ hall
  obtain ⟨hrd, -⟩ := chain hc W _ j₀ hj₀ rfl
  have hwr := W.add j₀ hj₀
  have hsum : sumBelow K (fun l => if before W j₀ l = true then W.mine l else 0) + W.mine j₀ =
      sumBelow K W.mine := by
    have := sumBelow_update (n := K) hj₀ (f := W.mine)
      (g := fun l => if before W j₀ l = true then W.mine l else 0)
      (fun l hl hne => by simp [hb l hl, hne])
    simp [hb j₀ hj₀] at this
    omega
  have hwv : leNat (c.exec.val (W.wr j₀)) = (sumBelow K fun j => (part j).toNat) % 2 ^ 32 := by
    rw [hwr, hrd, Nat.mod_add_mod, hsum]; rfl
  -- The load returned the last reduction's bytes, which the store writes back.
  have hld : c.exec.val (ix c hm₁) = (spl c hv m).r₃ 0 := by
    show c.evs[(ix c hm₁).val].val = _
    rw [hr]
    simp [ldE, oneEvent, extract_four (h₃m hl)]
  have hlen : (c.exec.val (W.wr j₀)).length = 4 := by
    rw [← hval, hld]; exact h₃m hl
  simp only [stE, oneEvent, ite_true, st_written, Option.getD_some]
  rw [← hld, hval]
  congr 1
  rw [word, hwv, ofNat_mod]
  rfl

omit hK hc in
include hv in
/-- Every event is a four-byte access to one of the three words. -/
theorem locs {v : Ev} (h : v ∈ c.evs) :
    (v.loc = wsLoc L ∨ v.loc = ctrLoc L ∨ v.loc = outLoc L) ∧ v.size = 4 := by
  obtain ⟨j, -, hj⟩ := (mem_evs c hv).1 h
  have : ∀ v ∈ (spl c hv j).events K L part j,
      (v.loc = wsLoc L ∨ v.loc = ctrLoc L ∨ v.loc = outLoc L) ∧ v.size = 4 := by
    by_cases hl : (spl c hv j).last K <;> simp [Split.events, hl, rmwEvents, rmwR, rmwW, oneEvent]
  exact this v hj

omit hK hc in
include hv in
/-- No two accesses overlap partially. -/
theorem uniform : Uniform c.evs := by
  intro a ha b hb hov
  rw [NV.overlaps_iff] at hov
  obtain ⟨-, -, -, -, h₁, h₂⟩ := hov
  obtain ⟨la, sa⟩ := locs c hv ha
  obtain ⟨lb, sb⟩ := locs c hv hb
  have := L.apart
  simp only [Ev.key, sa, sb, Prod.mk.injEq, and_true]
  rw [sa] at h₂; rw [sb] at h₁
  rcases la with e | e | e <;> rcases lb with e' | e' | e' <;> rw [e, e'] <;>
    first | rfl | (rw [e, e'] at h₁ h₂; simp [wsLoc, ctrLoc, outLoc] at h₁ h₂; omega)

omit hK hc in
/-- A write covering a byte of the output word is a last split's store. -/
theorem cover_out {k : Fin c.evs.length} {i : Nat} (hi : i < 4)
    (h : c.evs[k].covers { outLoc L with addr := L.out.toNat + i } = true) :
    ∃ m < K, (spl c hv m).last K ∧ c.evs[k] = stE c hv m := by
  rw [NV.covers_iff] at h
  obtain ⟨hw, -, -, h₃, h₄⟩ := h
  obtain ⟨hl, hs⟩ : (c.evs[k].loc = wsLoc L ∨ c.evs[k].loc = ctrLoc L ∨ c.evs[k].loc = outLoc L) ∧
      c.evs[k].size = 4 := locs c hv (List.getElem_mem k.isLt)
  have := L.apart
  have hout : c.evs[k].loc = outLoc L := by
    rcases hl with e | e | e
    · rw [e] at h₃ h₄; rw [hs] at h₄; simp [wsLoc] at h₃ h₄; omega
    · rw [e] at h₃ h₄; rw [hs] at h₄; simp [ctrLoc] at h₃ h₄; omega
    · exact e
  obtain ⟨j, hj, hmem⟩ := (mem_evs c hv).1 (List.getElem_mem k.isLt)
  have hall : ∀ v ∈ (spl c hv j).events K L part j, v.e.isWrite = true → v.loc = outLoc L →
      (spl c hv j).last K ∧ v = stE c hv j := by
    have h₁ := wsLoc_ne_outLoc (L := L)
    have h₂ := ctrLoc_ne_outLoc (L := L)
    by_cases hl : (spl c hv j).last K <;>
      simp [Split.events, hl, rmwEvents, rmwR, rmwW, oneEvent, Req.event, Event.isWrite, stE, h₁, h₂]
  obtain ⟨hl', he⟩ := hall _ hmem hw hout
  exact ⟨j, hj, hl', he⟩

end

/-- **The merge is correct.** For any number of splits `K` (`0 < K < 2³²`), any partials and any
layout of the three words, every outcome the PTX memory model allows completes, and its output
word holds the sum of the partials. -/
theorem merge_correct {K : Nat} (hK : 0 < K) (hK' : K < 2 ^ 32) (L : Layout) (part : Nat → UInt32)
    {o : Outcome} (h : Outcomes (program K L part) o) :
    ∃ m, o = .done m ∧ ∀ i < 4,
      m { outLoc L with addr := L.out.toNat + i } = (wordBytes (total K part)).getD i 0 := by
  obtain ⟨c, hv, hc, hy⟩ := h
  have hexit : ∀ re ∈ c.runs, re.2 = .exit := fun re hre => by
    obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 hre
    rw [runs_length] at hj
    rw [(spl_spec c hv hj).1]
  cases o with
  | fault msg =>
    obtain ⟨re, hre, h⟩ := hy; rw [hexit re hre] at h; cases h
  | unsupported msg =>
    rcases hy with ⟨re, hre, h⟩ | ⟨hn, -⟩
    · rw [hexit re hre] at h; cases h
    · exact absurd (uniform c hv) hn
  | done m =>
    obtain ⟨-, -, hfin⟩ := hy
    refine ⟨m, rfl, fun i hi => ?_⟩
    obtain ⟨m₀, hm₀, hl₀⟩ := last_exists c hv hK hc hK'
    rcases hfin { outLoc L with addr := L.out.toNat + i } with ⟨k, hcov, -, hmk⟩ | ⟨hnone, -⟩
    · obtain ⟨m', hm', hl', hk⟩ := cover_out c hv hi hcov
      rw [hmk, hk, store_value c hv hK hc hK' hm' hl']
      simp [stE, oneEvent, outLoc]
    · exfalso
      have := hnone (ix c (mem_stE c hv hm₀ hl₀))
      rw [ix_get, NV.covers_iff.2 (by
        simp [stE, oneEvent, Req.event, Event.isWrite, outLoc]; omega)] at this
      cases this

/-! ## Non-vacuity

`merge_correct` quantifies over the program's outcomes; a program with none would satisfy it
vacuously. One split adding 7 has an outcome. -/

namespace Witness

def L₀ : Layout := { ws := 0, ctr := 4, out := 8, apart := by decide }

def part₀ (_ : Nat) : UInt32 := 7

def P₀ : Program S := program 1 L₀ part₀

def r₁ : Nat → List UInt8 := fun _ => [0, 0, 0, 0]
def r₂ : Nat → List UInt8 := fun _ => [0, 0, 0, 0]
def r₃ : Nat → List UInt8 := fun _ => [7, 0, 0, 0]
def r₄ : Nat → List UInt8 := fun _ => []

def acts₀ : List Act :=
  [.mem (redReq L₀ 7) r₁, .mem (atomReq L₀) r₂, .mem (ldReq L₀) r₃, .mem (stReq L₀ 7) r₄]

theorem run₀ : Run P₀.step { pc := 0, part := 7 } acts₀ .exit := by
  show Run (step 1 L₀) _ [.mem (redReq L₀ 7) r₁, .mem (atomReq L₀) r₂, .mem (ldReq L₀) r₃,
    .mem (stReq L₀ 7) r₄] _
  refine Run.mem (rf := fun _ => redReq L₀ 7) (read := r₁) rfl (fun _ _ => rfl) (by decide) ?_
  refine Run.mem (rf := fun _ => atomReq L₀) (read := r₂) rfl (fun _ _ => rfl) (by decide) ?_
  refine Run.next (s' := { pc := 3, part := 7, old := ((atomReq L₀).resp r₂)[0]! }) ?_ ?_
  · show (if (((atomReq L₀).resp r₂)[0]!).toNat + 1 = 1 then _ else _) = _
    rw [ite_eq_left (by decide +kernel)]
  refine Run.mem (rf := fun _ => ldReq L₀) (read := r₃)
    (s' := { pc := 4, part := 7, old := ((atomReq L₀).resp r₂)[0]! })
    (wb := fun r t => { t with v := r[0]! }) rfl (fun _ _ => rfl) (by decide) ?_
  have hv : ((ldReq L₀).resp r₃)[0]! = 7 := by decide +kernel
  refine Run.mem (rf := fun _ => stReq L₀ 7) (read := r₄)
    (s' := { pc := 5, part := 7, old := ((atomReq L₀).resp r₂)[0]!, v := ((ldReq L₀).resp r₃)[0]! })
    (wb := fun _ t => t) ?_ (fun _ _ => rfl) (by decide) ?_
  · show Effect.mem (fun _ => stReq L₀ (((ldReq L₀).resp r₃)[0]!)) _ _ = _
    rw [hv]
  exact Run.exit rfl

def evs₀ : List Ev := P₀.events [(acts₀, .exit)]

def c₀ : Candidate P₀ where
  runs := [(acts₀, .exit)]
  length := rfl
  runs_ok := by
    intro p hp
    simp only [P₀, program, List.range_one, List.map_cons, List.map_nil, List.zip_cons_cons,
      List.zip_nil_right, List.mem_singleton] at hp
    subst hp
    exact run₀
  evs := evs₀
  perm := List.Perm.refl _
  rf := rfOf evs₀
  co := coOf evs₀
  sc := scOf evs₀
  collects := P₀.collectsOk_nil rfl
  dep := readPo evs₀
  dep_ok _ _ h := semDep_readPo P₀ h

/-- The memory the run leaves: the workspace and the output hold 7, the counter 1. -/
def m₀ (L : Loc) : UInt8 :=
  if L.space = .global ∧ L.owner = 0 ∧ (L.addr = 0 ∨ L.addr = 8) then 7
  else if L.space = .global ∧ L.owner = 0 ∧ L.addr = 4 then 1 else 0

theorem consistent₀ : c₀.exec.consistent = true := by decide +kernel

theorem final₀ : c₀.IsFinal m₀ := by
  intro L
  obtain ⟨sp, ow, a⟩ := L
  by_cases h : sp = .global ∧ ow = 0 ∧ a < 12
  · obtain ⟨rfl, rfl, ha⟩ := h
    left
    match a, ha with
    | 0, _ | 1, _ | 2, _ | 3, _ | 4, _ | 5, _ | 6, _ | 7, _ | 8, _ | 9, _ | 10, _ | 11, _ =>
      decide +kernel
  · right
    -- Every event is a four-byte global access below address 12.
    have hin : ∀ i : Fin evs₀.length, evs₀[i].loc.space = .global ∧ evs₀[i].loc.owner = 0 ∧
        evs₀[i].loc.addr + evs₀[i].size ≤ 12 := by decide +kernel
    refine ⟨fun i => ?_, ?_⟩
    · change evs₀[i].covers ⟨sp, ow, a⟩ = false
      cases hc : evs₀[i].covers ⟨sp, ow, a⟩
      · rfl
      · obtain ⟨-, h₁, h₂, -, h₄⟩ := NV.covers_iff.1 hc
        obtain ⟨e₁, e₂, e₃⟩ := hin i
        exact absurd ⟨h₁.symm.trans e₁, h₂.symm.trans e₂, by simp only at h₄; omega⟩ h
    · show m₀ ⟨sp, ow, a⟩ = 0
      unfold m₀
      rw [ite_eq_right (fun h' => h ⟨h'.1, h'.2.1, by rcases h'.2.2 with e | e <;> simp only at e <;> omega⟩),
        ite_eq_right (fun h' => h ⟨h'.1, h'.2.1, by have e := h'.2.2; simp only at e; omega⟩)]
end Witness

/-- **The merge's outcomes are not vacuous**: one split adding 7 has an outcome. -/
theorem merge_nonvacuous : ∃ o, Outcomes Witness.P₀ o := by
  have hv : Witness.c₀.Valid :=
    ⟨by show BarriersOk Witness.evs₀; unfold BarriersOk; decide +kernel,
     fun ⟨re, hre, h⟩ => by simp [Witness.c₀] at hre; subst hre; cases h⟩
  exact ⟨.done Witness.m₀, Witness.c₀, hv, Witness.consistent₀, by simp [Witness.c₀],
    uniform Witness.c₀ hv, Witness.final₀⟩

end Examples.SplitMerge.Program
