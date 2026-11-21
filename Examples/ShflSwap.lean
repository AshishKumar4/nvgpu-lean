import NV.Formal.RunEvents
import NV.Formal.Cand

/-!
# Two threads swap words through `shfl.sync`

Threads 0 and 1 of a warp each load word `t` of `X`, exchange it with `shfl.sync.bfly.b32` (lane
`t` reads lane `t ^ 1`, member mask `0b11`), and store what they got to word `t` of `O`.
`swap_correct`: every completed outcome holds word `1 - t` of `X` at word `t` of `O`. The
answers come from the collective's instance (`Program.CollectsOk`), which both threads must join.
-/

namespace Examples.ShflSwap

open NV NV.Formal NV.PTX NV.Rel NV.PTX.Execution

structure Layout where
  X : Nat
  O : Nat
  apart : X + 8 ≤ O
  fits : O + 8 < 2 ^ 64

inductive Pc where
  | ld | shfl | st | done
  deriving DecidableEq, Repr

structure St where
  pc : Pc
  tid : Nat
  v : UInt32 := 0

section
variable (L : Layout)

def ldReq (t : Nat) : Req :=
  { space := .global, op := .ld, bytes := 4, addrs := #[some (UInt64.ofNat (L.X + 4 * t))] }

def stReq (t : Nat) (v : UInt32) : Req :=
  { space := .global, op := .st, bytes := 4, addrs := #[some (UInt64.ofNat (L.O + 4 * t))],
    data := #[v] }

/-- `shfl.sync.bfly.b32 d, v, 1, 0x1f, 0b11`. -/
def shflOp : Collective := .shfl .bfly

def step (s : St) : Effect St :=
  match s.pc with
  | .ld => .mem (fun _ => ldReq L s.tid) { s with pc := .shfl } fun r t => { t with v := r[0]! }
  | .shfl => .collect shflOp 3 #[s.v, 1, 31] { s with pc := .st } fun r t => { t with v := r[0]! }
  | .st => .mem (fun _ => stReq L s.tid s.v) { s with pc := .done } fun _ t => t
  | .done => .exit

def agent (t : Nat) : Agent St := { init := { pc := .ld, tid := t }, tid := t, place := ⟨0, 0, 0⟩ }

/-- `x t` is word `t` of `X`'s initial bytes; the rest of memory starts zero. -/
def program (x : Fin 2 → Fin 4 → UInt8) : Program St where
  step := step L
  agents := [agent 0, agent 1]
  mem l := if h : l.space = .global ∧ l.owner = 0 ∧ L.X ≤ l.addr ∧ l.addr < L.X + 8 then
    x ⟨(l.addr - L.X) / 4, by omega⟩ ⟨(l.addr - L.X) % 4, Nat.mod_lt _ (by decide)⟩ else 0

end

/-! ## A thread's runs -/

section runs
variable {L : Layout}

/-- What a thread that exits did: its load, its shuffle and its store. -/
def acts (L : Layout) (t : Nat) (rd : Nat → List UInt8) (inst : Nat) (resp : Resp)
    (ro : Nat → List UInt8) : List Act :=
  [.mem (ldReq L t) rd, .collect shflOp 3 #[((ldReq L t).resp rd)[0]!, 1, 31] inst (.ok resp),
   .mem (stReq L t resp[0]!) ro]

theorem collect_inv {s : St} {op mask contrib s' wb} (hs : step L s = .collect op mask contrib s' wb)
    {acts : List Act} {e : End} (h : Run (step L) s acts e) :
    (acts = [] ∧ e = .stop) ∨ (∃ inst, acts = [.collect op mask contrib inst (.error e)]) ∨
      ∃ inst resp rest, acts = .collect op mask contrib inst (.ok resp) :: rest ∧
        Run (step L) (wb resp s') rest e := by
  cases h with
  | stop => exact .inl ⟨rfl, rfl⟩
  | collect h₁ h₂ => rw [hs] at h₁; cases h₁; exact .inr (.inr ⟨_, _, _, rfl, h₂⟩)
  | collectFail h₁ => rw [hs] at h₁; cases h₁; exact .inr (.inl ⟨_, rfl⟩)
  | _ => simp_all

/-- **A thread's run that exits**: its load, its shuffle with some answer, its store; or a
shuffle that ended the run (`.exit` included, which `CollectsOk` will rule out). -/
theorem run_exit {t : Nat} {as : List Act} (h : Run (step L) { pc := .ld, tid := t } as .exit) :
    (∃ rd inst resp ro, as = acts L t rd inst resp ro ∧ (rd 0).length = 4) ∨
      ∃ rd inst, as = [.mem (ldReq L t) rd,
        .collect shflOp 3 #[((ldReq L t).resp rd)[0]!, 1, 31] inst (.error .exit)] := by
  rcases Run.of_mem (r := ldReq L t) (wb := fun r t => { t with v := r[0]! })
      (s' := { pc := .shfl, tid := t }) (by simp [step]) h with
    ⟨-, h⟩ | ⟨read, -, h, -, -⟩ | ⟨rd, rest, rfl, hrd, -, h⟩
  · cases h
  · simp [failureOf, ldReq] at h
  have hl : (rd 0).length = 4 := (hrd 0 (by simp [Req.active, ldReq])).trans (by
    simp [Req.readLen, Req.pieces, Req.granule, Req.strong, ldReq])
  rcases collect_inv (op := shflOp) (mask := 3) (contrib := #[((ldReq L t).resp rd)[0]!, 1, 31])
      (s' := { pc := .st, tid := t, v := ((ldReq L t).resp rd)[0]! })
      (wb := fun r t => { t with v := r[0]! }) (by simp [step]) h with
    ⟨-, h₁⟩ | ⟨inst, rfl⟩ | ⟨inst, resp, rest, rfl, h₂⟩
  · cases h₁
  · exact .inr ⟨rd, inst, rfl⟩
  rcases Run.of_mem (s := ({ pc := .st, tid := t, v := resp[0]! } : St)) (r := stReq L t resp[0]!)
      (wb := fun _ t => t) (s' := ({ pc := .done, tid := t, v := resp[0]! } : St)) rfl h₂ with
    ⟨-, h⟩ | ⟨read, -, h, -, -⟩ | ⟨ro, rest, rfl, -, -, h⟩
  · cases h
  · simp [failureOf, stReq] at h
  obtain ⟨rfl, -⟩ := Run.of_exit (by simp [step]) h
  exact .inl ⟨rd, inst, resp, ro, rfl, hl⟩

end runs

/-! ## The shuffle's answer -/

/-- Two lanes' contributions, the rest absent. -/
def two (c₀ c₁ : Array UInt32) : Array (Option (Array UInt32)) :=
  (List.range 32).toArray.map fun i => if i = 0 then some c₀ else if i = 1 then some c₁ else none

theorem present_two (c₀ c₁ : Array UInt32) : Collective.present (two c₀ c₁) = 3 := by
  simp only [Collective.present, Id.run, Std.Legacy.Range.forIn_eq_forIn_range']
  simp [two, List.range_succ, Std.Legacy.Range.size]
  rfl

/-- **The butterfly shuffle swaps the two lanes' words.** -/
theorem answer_swap (v₀ v₁ : UInt32) :
    Collective.answer shflOp 3 (two #[v₀, 1, 31] #[v₁, 1, 31]) =
      .ok ((List.range 32).toArray.map fun i => if i = 0 then #[v₁, 1] else if i = 1 then #[v₀, 1]
        else #[]) := by
  unfold Collective.answer
  rw [present_two]
  simp [two, shflOp, List.range_succ, Collective.respond, Collective.shflSource, Collective.word,
    Collective.ofBool]
  rfl

/-! ## The run's events -/

section events
variable (L : Layout) (x : Fin 2 → Fin 4 → UInt8)

/-- Thread `t`'s load and store events. -/
def ldE (t : Nat) (rd : Nat → List UInt8) : Ev := oneEv (ldReq L t) t 0 (gloc (L.X + 4 * t)) false 4 (rd 0)
def stE (t : Nat) (v : UInt32) (ro : Nat → List UInt8) : Ev :=
  oneEv (stReq L t v) t 1 (gloc (L.O + 4 * t)) true 4 (ro 0)

theorem pieces_ld (t : Nat) : (ldReq L t).pieces = [⟨false, 0, 4, false⟩] := by
  simp [Req.pieces, Req.granule, Req.strong, ldReq]
theorem pieces_st (t : Nat) (v : UInt32) : (stReq L t v).pieces = [⟨true, 0, 4, false⟩] := by
  simp [Req.pieces, Req.granule, Req.strong, stReq]

theorem events_acts {t : Nat} (ht : t < 2) (rd : Nat → List UInt8) (inst : Nat) (resp : Resp)
    (ro : Nat → List UInt8) :
    (program L x).agentEvents (agent t) (fun _ => 0) (acts L t rd inst resp ro) ++
      (program L x).asyncEvents (agent t) (fun _ => 0) {} (acts L t rd inst resp ro) =
      [ldE L t rd, stE L t resp[0]! ro] := by
  have hX : L.X + 4 * t < 2 ^ 64 := by have := L.fits; have := L.apart; omega
  have hO : L.O + 4 * t < 2 ^ 64 := by have := L.fits; omega
  have w₁ : Act.width (agent t) t (.mem (ldReq L t) rd) = 1 :=
    (width_one (ag := agent t) (r := ldReq L t) (a := L.X + 4 * t) rfl rd).trans (by rw [pieces_ld]; rfl)
  simp only [acts, Program.agentEvents, Program.asyncEvents, List.append_nil]
  rw [actEvents_one (program L x) (ag := agent t) rfl rfl hX (.inr rfl),
    actEvents_one (program L x) (ag := agent t) rfl rfl hO (.inr rfl), one_events (pieces_ld L t),
    one_events (pieces_st L t _)]
  simp only [show (agent t).tid = t from rfl, w₁, Program.actEvents, List.nil_append, 
    Nat.zero_add]
  rfl

end events

/-! ## Every completed outcome -/

section final
variable {L : Layout} {x : Fin 2 → Fin 4 → UInt8}

/-- Thread `t`'s shuffle, as `Program.joined` lists it. -/
def joinOf (L : Layout) (t : Nat) (rd : Nat → List UInt8) (inst : Nat) (ans : Except End Resp) : Joined St :=
  ⟨agent t, shflOp, 3, #[((ldReq L t).resp rd)[0]!, 1, 31], inst, ans⟩

theorem contribs_two (rd₀ rd₁ : Nat → List UInt8) (i₀ i₁ : Nat) (a₀ a₁ : Except End Resp) :
    Joined.contribs [joinOf L 0 rd₀ i₀ a₀, joinOf L 1 rd₁ i₁ a₁] shflOp =
      two #[((ldReq L 0).resp rd₀)[0]!, 1, 31] #[((ldReq L 1).resp rd₁)[0]!, 1, 31] := by
  simp only [Joined.contribs, two]
  congr 1
  funext i
  simp only [List.find?, Joined.index, joinOf, agent, shflOp]
  by_cases h0 : i = 0
  · subst h0; rfl
  · by_cases h1 : i = 1
    · subst h1; rfl
    · rw [show (0 == i) = false by simpa using Ne.symm h0, show (1 == i) = false by simpa using Ne.symm h1]
      simp [h0, h1]

/-- **The shuffle's answers**: both threads join one instance, and each gets the other's word. -/
theorem answers {runs : List (List Act × End)} {rd₀ rd₁ : Nat → List UInt8} {i₀ i₁ : Nat}
    {a₀ a₁ : Except End Resp}
    (h : (program L x).CollectsOk runs)
    (hj : (program L x).joined runs = [joinOf L 0 rd₀ i₀ a₀, joinOf L 1 rd₁ i₁ a₁]) :
    a₀ = .ok #[((ldReq L 1).resp rd₁)[0]!, 1] ∧ a₁ = .ok #[((ldReq L 0).resp rd₀)[0]!, 1] := by
  have m₀ : joinOf L 0 rd₀ i₀ a₀ ∈ (program L x).joined runs := by rw [hj]; simp
  have m₁ : joinOf L 1 rd₁ i₁ a₁ ∈ (program L x).joined runs := by rw [hj]; simp
  -- Thread 1 is in thread 0's instance: it is in the member mask.
  obtain ⟨-, -, hmem, -⟩ := h _ m₀
  obtain ⟨k, hk, hkt⟩ := hmem (agent 1) (by simp [program]) rfl (by simp [joinOf, agent]; decide)
  rw [hj] at hk
  simp only [List.mem_filter, List.mem_cons, List.not_mem_nil, or_false, beq_iff_eq] at hk
  obtain ⟨hk | hk, hki⟩ := hk
  · subst hk; simp [joinOf, agent] at hkt
  subst hk
  simp only [joinOf] at hki
  subst hki
  have hgrp : ((program L x).joined runs).filter (·.inst == i₁) =
      [joinOf L 0 rd₀ i₁ a₀, joinOf L 1 rd₁ i₁ a₁] := by rw [hj]; simp [joinOf]
  obtain ⟨-, -, -, ha₀⟩ := h _ m₀
  obtain ⟨-, -, -, ha₁⟩ := h _ m₁
  simp only [joinOf] at ha₀ ha₁ hgrp
  rw [hgrp] at ha₀ ha₁
  have hc := contribs_two (L := L) rd₀ rd₁ i₁ i₁ a₀ a₁
  simp only [joinOf] at hc
  rw [hc, answer_swap] at ha₀ ha₁
  simp [ha₀, ha₁, agent, Joined.index, shflOp]


theorem joined_eq {rd₀ rd₁ : Nat → List UInt8} {i₀ i₁ : Nat} {r₀ r₁ : Resp} {ro₀ ro₁ : Nat → List UInt8} :
    (program L x).joined [(acts L 0 rd₀ i₀ r₀ ro₀, .exit), (acts L 1 rd₁ i₁ r₁ ro₁, .exit)] =
      [joinOf L 0 rd₀ i₀ (.ok r₀), joinOf L 1 rd₁ i₁ (.ok r₁)] := rfl

theorem st_val {t : Nat} {rd ro : Nat → List UInt8} (h : (rd 0).length = 4) (t' : Nat) :
    (stE L t ((ldReq L t').resp rd)[0]! ro).val = rd 0 := by
  match hb : rd 0, h with
  | [b₀, b₁, b₂, b₃], _ =>
    simp [stE, oneEv, Req.written, stReq, Req.storeSlice_eq, Req.storeBytes, Req.operands, Req.resp, Req.laneResp, ldReq,
      words, MemOp.returns, hb, toWords, toBytes]
    rw [wordBytes_leNat rfl]; rfl

theorem init_x {t i : Nat} (ht : t < 2) (hi : i < 4) :
    (initBytes (program L x).mem (gloc (L.X + 4 * t), 4)).getD i 0 = x ⟨t, ht⟩ ⟨i, hi⟩ := by
  simp only [initBytes, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hi,
    Option.map_some, Option.getD_some, program, gloc]
  split
  · congr 1
    · ext; simp; omega
    · ext; simp; omega
  · rename_i h; exact absurd ⟨trivial, trivial, by omega, by omega⟩ h

/-- **The threads swap**: every completed outcome holds word `1 - t` of `X` at word `t` of `O`. -/
theorem swap_correct {m : Loc → UInt8} (h : Outcomes (program L x) (.done m)) {t i : Nat}
    (ht : t < 2) (hi : i < 4) : m (gloc (L.O + 4 * t + i)) = x ⟨1 - t, by omega⟩ ⟨i, hi⟩ := by
  obtain ⟨c, -, hc, hex, -, hf⟩ := h
  obtain ⟨⟨a₀, e₀⟩, ⟨a₁, e₁⟩, hruns⟩ : ∃ p₁ p₂, c.runs = [p₁, p₂] := by
    match hr : c.runs, c.length with
    | [p₁, p₂], _ => exact ⟨p₁, p₂, rfl⟩
  have he₀ : e₀ = .exit := hex (a₀, e₀) (by simp [hruns])
  have he₁ : e₁ = .exit := hex (a₁, e₁) (by simp [hruns])
  subst he₀ he₁
  have hr₀ : Run (step L) { pc := .ld, tid := 0 } a₀ .exit :=
    c.runs_ok (agent 0, a₀, .exit) (by simp [hruns, program, agent])
  have hr₁ : Run (step L) { pc := .ld, tid := 1 } a₁ .exit :=
    c.runs_ok (agent 1, a₁, .exit) (by simp [hruns, program, agent])
  have hco := c.collects
  rw [hruns] at hco
  -- Neither shuffle ended its run.
  obtain ⟨rd₀, i₀, r₀, ro₀, rfl, hl₀⟩ :
      ∃ rd inst resp ro, a₀ = acts L 0 rd inst resp ro ∧ (rd 0).length = 4 := by
    rcases run_exit hr₀ with h | ⟨rd₀, i₀, rfl⟩
    · exact h
    · exfalso
      exact hco.not_exit (j := ⟨agent 0, shflOp, 3, #[((ldReq L 0).resp rd₀)[0]!, 1, 31], i₀, .error .exit⟩)
        (by simp [Program.joined, program]) rfl
  obtain ⟨rd₁, i₁, r₁, ro₁, rfl, hl₁⟩ :
      ∃ rd inst resp ro, a₁ = acts L 1 rd inst resp ro ∧ (rd 0).length = 4 := by
    rcases run_exit hr₁ with h | ⟨rd₁, i₁, rfl⟩
    · exact h
    · exfalso
      exact hco.not_exit (j := ⟨agent 1, shflOp, 3, #[((ldReq L 1).resp rd₁)[0]!, 1, 31], i₁, .error .exit⟩)
        (by simp [Program.joined, program]) rfl
  obtain ⟨hr₀, hr₁⟩ := answers hco joined_eq
  cases hr₀; cases hr₁
  -- The events.
  have hE : ∀ v, v ∈ c.evs ↔ v ∈ [ldE L 0 rd₀, stE L 0 ((ldReq L 1).resp rd₁)[0]! ro₀,
      ldE L 1 rd₁, stE L 1 ((ldReq L 0).resp rd₀)[0]! ro₁] := fun v => by
    rw [c.perm.mem_iff, hruns]
    simp only [Program.events, show (program L x).agents = [agent 0, agent 1] from rfl,
      List.zip_cons_cons, List.zip_nil_left, List.flatMap_cons, List.flatMap_nil, List.append_nil]
    rw [events_acts L x (by decide), events_acts L x (by decide)]
    simp
  -- A load reads `X`'s initial bytes: only the stores write, and to `O`.
  have hread : ∀ (u : Nat) (rd : Nat → List UInt8), u < 2 → (rd 0).length = 4 → ldE L u rd ∈ c.evs →
      rd 0 = initBytes (program L x).mem (gloc (L.X + 4 * u), 4) := by
    intro u rd hu hl hm
    obtain ⟨k, hk⟩ := Candidate.index c hm
    rcases Candidate.source c hc (r := k) (by rw [hk]; simp [ldE, oneEv, Req.event, Event.isRead])
      with ⟨-, hv⟩ | ⟨w, -, -, hw, hkey⟩
    · rw [hk] at hv
      simpa [ldE, oneEv, take_all hl, Ev.key] using hv
    · exfalso
      have hwm := (hE _).1 (List.getElem_mem w.isLt)
      have hw' : c.evs[w.1].e.isWrite = true := hw
      have hkey' : c.evs[w.1].key = (ldE L u rd).key := by rw [← hk]; exact hkey
      have := L.apart
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hwm
      rcases hwm with h | h | h | h <;> rw [h] at hw' hkey' <;>
        simp [ldE, stE, oneEv, Ev.key, gloc, Req.event, Event.isWrite] at hw' hkey' <;> omega
  -- The final byte is the store's.
  have hcov : (stE L t (if t = 0 then ((ldReq L 1).resp rd₁)[0]! else ((ldReq L 0).resp rd₀)[0]!)
      (if t = 0 then ro₀ else ro₁)).covers (gloc (L.O + 4 * t + i)) = true := by
    rw [NV.covers_iff]
    refine ⟨by simp [stE, oneEv, Req.event, Event.isWrite], rfl, rfl, by simp [stE, oneEv, gloc],
      by simp [stE, oneEv, gloc]; omega⟩
  have hSt : stE L t (if t = 0 then ((ldReq L 1).resp rd₁)[0]! else ((ldReq L 0).resp rd₀)[0]!)
      (if t = 0 then ro₀ else ro₁) ∈ c.evs := by
    rw [hE]; rcases (by omega : t = 0 ∨ t = 1) with rfl | rfl <;> simp
  rcases hf (gloc (L.O + 4 * t + i)) with ⟨j, hj, -, hm⟩ | ⟨hn, -⟩
  · obtain ⟨hw, hsp, -, hlo, hhi⟩ := NV.covers_iff.1 hj
    have hjm := (hE _).1 (List.getElem_mem j.isLt)
    have := L.apart
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hjm
    simp only [Fin.getElem_fin] at hw hlo hhi hm
    rw [hm]
    rcases (by omega : t = 0 ∨ t = 1) with rfl | rfl
    · rcases hjm with h | h | h | h <;> rw [h] at hw hlo hhi ⊢ <;>
        simp [ldE, stE, oneEv, gloc, Req.event, Event.isWrite] at hw hlo hhi <;> try omega
      rw [st_val hl₁, hread 1 rd₁ (by decide) hl₁ ((hE _).2 (by simp))]
      simp only [stE, oneEv, gloc, Nat.mul_zero, Nat.add_zero, Nat.add_sub_cancel_left]
      exact init_x (by decide) hi
    · rcases hjm with h | h | h | h <;> rw [h] at hw hlo hhi ⊢ <;>
        simp [ldE, stE, oneEv, gloc, Req.event, Event.isWrite] at hw hlo hhi <;> try omega
      rw [st_val hl₀, hread 0 rd₀ (by decide) hl₀ ((hE _).2 (by simp))]
      simp only [stE, oneEv, gloc, Nat.mul_one, Nat.add_sub_cancel_left]
      exact init_x (by decide) hi
  · obtain ⟨j, hj⟩ := Candidate.index c hSt
    rw [← hj, hn j] at hcov; cases hcov

end final

end Examples.ShflSwap
