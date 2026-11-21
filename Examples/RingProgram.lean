import NV.Formal.Pipeline
import NV.Formal.RunEvents
import NV.Formal.Cand
import NV.Async.Tma
import Examples.RingRace

/-!
# The TMA ring as a program

A producer thread and a consumer thread of one CTA pass `N` words through a ring of `S` shared
slots. The producer initializes a full and an empty mbarrier per slot, then for stage `q`: from the
second round of a slot on, it waits on the slot's empty barrier for the consumer's arrive of the
round before and fences the async proxy (`fence.proxy.async`); it arrives on the full barrier
expecting 4 bytes (`mbarrier.arrive.expect_tx`) and issues a bulk copy of word `q` of `X` into the
slot, which completes 4 bytes on the full barrier. The consumer, for stage `q`, spins on the full
barrier for the round's parity, loads the slot, stores the word to `O`, and arrives on the empty
barrier.

`ring_correct`: every outcome in which both threads finish holds, at word `q` of `O`, word `q` of
`X` for every stage. The barriers' orders come from `NV.Formal.Pipeline` (`Full.raw`,
`Empty.war`), the ring's from `Examples.Ring.Race` (`Ring.reads_stage`).
-/

namespace Examples.RingProgram

open NV NV.Formal NV.Formal.Pipeline NV.PTX NV.Async NV.Rel NV.PTX.Execution

/-- Where the ring keeps its words: `N` words of `X` and `O` in global memory; `S` slots, `S` full
and `S` empty mbarriers in the CTA's shared memory. -/
structure Layout where
  N : Nat
  S : Nat
  X : Nat
  O : Nat
  Y : Nat
  F : Nat
  E : Nat
  S_pos : 0 < S
  global : X + 4 * N ≤ O
  shared : Y + 4 * S ≤ F ∧ F + 8 * S ≤ E
  fits : O + 4 * N < 2 ^ 64 ∧ E + 8 * S < 2 ^ 64

def mbarReq (a : Nat) (op : MBarOp) (sem : NV.Sem) (scope : Scope) : Req :=
  { space := .shared, op := .mbar op, bytes := 8, sem, scope, addrs := #[some (UInt64.ofNat a)] }

/-- `mbarrier.try_wait.parity`: an acquire load of the barrier word. -/
def waitReq (a : Nat) : Req :=
  { space := .shared, op := .ld, bytes := 8, sem := .acquire, scope := .cta,
    addrs := #[some (UInt64.ofNat a)] }

namespace Layout

variable (L : Layout)

def slot (q : Nat) : Nat := L.Y + 4 * (q % L.S)
def full (q : Nat) : Nat := L.F + 8 * (q % L.S)
def empty (q : Nat) : Nat := L.E + 8 * (q % L.S)

def initF (k : Nat) : Req := mbarReq (L.F + 8 * k) (.init 1) .weak .cta
def initE (k : Nat) : Req := mbarReq (L.E + 8 * k) (.init 1) .weak .cta
def arriveF (q : Nat) : Req := mbarReq (L.full q) (.arrive 1 4 false) .release .cta
def completeF (q : Nat) : Req := mbarReq (L.full q) (.completeTx 4) .release .cluster
def arriveE (q : Nat) : Req := mbarReq (L.empty q) (.arrive 1 0 false) .release .cta

def ldSlot (q : Nat) : Req :=
  { space := .shared, op := .ld, bytes := 4, addrs := #[some (UInt64.ofNat (L.slot q))] }
def stOut (q : Nat) (v : UInt32) : Req :=
  { space := .global, op := .st, bytes := 4, addrs := #[some (UInt64.ofNat (L.O + 4 * q))],
    data := #[v] }

/-- Stage `q`'s copy: word `q` of `X` into the slot, then complete-tx on the full barrier. -/
def copy (q : Nat) : Copy :=
  { moves := #[{ src := asyncReq .global .ld 4 #[some (UInt64.ofNat (L.X + 4 * q))],
                 dsts := #[asyncReq .shared .st 4 #[some (UInt64.ofNat (L.slot q))]] }]
    signals := #[L.completeF q] }

end Layout

/-- Where each thread is. The producer: initializing barriers, at a stage, waiting on the empty
barrier, fencing, arriving, copying. The consumer: at a stage, waiting on the full barrier,
loading, storing, arriving. -/
inductive Pc where
  | initF (k : Nat) | initE (k : Nat) | stage (q : Nat) | wait (q : Nat) | check (q : Nat)
  | fence (q : Nat) | arrive (q : Nat) | copy (q : Nat)
  | cstage (q : Nat) | cwait (q : Nat) | ccheck (q : Nat) | load (q : Nat) | store (q : Nat)
  | carrive (q : Nat)
  deriving DecidableEq, Repr

structure St where
  pc : Pc
  done : Bool := false
  v : UInt32 := 0

def step (L : Layout) (s : St) : Effect St :=
  match s.pc with
  | .initF k => .mem (fun _ => L.initF k)
      { s with pc := if k + 1 < L.S then .initF (k + 1) else .initE 0 } fun _ t => t
  | .initE k => .mem (fun _ => L.initE k)
      { s with pc := if k + 1 < L.S then .initE (k + 1) else .stage 0 } fun _ t => t
  | .stage q => if q < L.N then .next { s with pc := if q < L.S then .arrive q else .wait q }
      else .exit
  | .wait q => .mem (fun _ => waitReq (L.empty q)) { s with pc := .check q } fun r t =>
      { t with done := MBarrierWord.parityDone (word64 r) (q / L.S - 1) }
  | .check q => .next { s with pc := if s.done then .fence q else .wait q }
  | .fence q => .sync (.proxyFence .async) { s with pc := .arrive q }
  | .arrive q => .mem (fun _ => L.arriveF q) { s with pc := .copy q } fun _ t => t
  | .copy q => .async (fun _ => .copy (L.copy q)) { s with pc := .stage (q + 1) } fun _ t => t
  | .cstage q => if q < L.N then .next { s with pc := .cwait q } else .exit
  | .cwait q => .mem (fun _ => waitReq (L.full q)) { s with pc := .ccheck q } fun r t =>
      { t with done := MBarrierWord.parityDone (word64 r) (q / L.S) }
  | .ccheck q => .next { s with pc := if s.done then .load q else .cwait q }
  | .load q => .mem (fun _ => L.ldSlot q) { s with pc := .store q } fun r t => { t with v := r[0]! }
  | .store q => .mem (fun _ => L.stOut q s.v) { s with pc := .carrive q } fun _ t => t
  | .carrive q => .mem (fun _ => L.arriveE q) { s with pc := .cstage (q + 1) } fun _ t => t

/-- `x q` is word `q` of `X`'s initial bytes; the rest of memory starts zero. -/
def program (L : Layout) (x : Nat → Fin 4 → UInt8) : Program St where
  step := step L
  agents := [{ init := { pc := .initF 0 }, tid := 0, place := ⟨0, 0, 0⟩ },
             { init := { pc := .cstage 0 }, tid := 1, place := ⟨0, 0, 0⟩ }]
  mem l := if h : l.space = .global ∧ l.owner = 0 ∧ L.X ≤ l.addr ∧ l.addr < L.X + 4 * L.N then
    x ((l.addr - L.X) / 4) ⟨(l.addr - L.X) % 4, Nat.mod_lt _ (by decide)⟩ else 0

/-! ## The threads' runs -/

section runs

variable {L : Layout}

/-- How a read of request `r` may return: its bytes for every active lane. -/
def ReadOk (r : Req) (read : Nat → List UInt8) : Prop :=
  ∀ l, r.active l → (read l).length = r.readLen

theorem wait_ok (a : Nat) (read : Nat → List UInt8) : (waitReq a).unsupported read = false := by
  simp [Req.unsupported, waitReq, MemOp.writes]

theorem ld_ok (q : Nat) (read : Nat → List UInt8) : (L.ldSlot q).unsupported read = false := by
  simp [Req.unsupported, Layout.ldSlot, MemOp.writes]

theorem st_ok (q : Nat) (v : UInt32) (read : Nat → List UInt8) :
    (L.stOut q v).unsupported read = false := by
  simp [Req.unsupported, Layout.stOut, MemOp.writes]

/-- A spin on a barrier in a run that exits: waits that returned false, then one that returned
true, then the run from where the spin leads. -/
theorem spin {w c A : Pc} {a p : Nat}
    (hw : ∀ s : St, s.pc = w → step L s = .mem (fun _ => waitReq a) { s with pc := c }
      fun r t => { t with done := MBarrierWord.parityDone (word64 r) p })
    (hc : ∀ s : St, s.pc = c → step L s = .next { s with pc := if s.done then A else w })
    {d : Bool} {v : UInt32} {acts : List Act}
    (h : Run (step L) { pc := w, done := d, v } acts .exit) :
    ∃ (ws : List (Nat → List UInt8)) (rw : Nat → List UInt8) (rest : List Act),
      acts = ws.map (Act.mem (waitReq a)) ++ .mem (waitReq a) rw :: rest ∧
      (∀ r ∈ ws, ReadOk (waitReq a) r) ∧ ReadOk (waitReq a) rw ∧
      MBarrierWord.parityDone (word64 ((waitReq a).resp rw)) p = true ∧
      Run (step L) { pc := A, done := true, v } rest .exit := by
  induction hn : acts.length using Nat.strongRecOn generalizing acts d with
  | ind n ih =>
    rcases Run.of_mem (hw _ rfl) h with ⟨-, h⟩ | ⟨read, -, h, -, -⟩ | ⟨rw, rest, rfl, hl, -, h⟩
    · cases h
    · simp [failureOf, waitReq] at h
    cases hd : MBarrierWord.parityDone (word64 ((waitReq a).resp rw)) p
    · rcases Run.of_next (hc _ rfl) h with ⟨-, h⟩ | h
      · cases h
      simp only [hd, Bool.false_eq_true, ite_false] at h
      obtain ⟨ws, rw', rest', rfl, hws, hl', hd', h'⟩ := ih rest.length (by rw [← hn]; simp) h rfl
      exact ⟨rw :: ws, rw', rest', by simp, fun r hr => by
        rcases List.mem_cons.1 hr with rfl | hr
        · exact hl
        · exact hws r hr, hl', hd', h'⟩
    · rcases Run.of_next (hc _ rfl) h with ⟨-, h⟩ | h
      · cases h
      simp only [hd, ite_true] at h
      exact ⟨[], rw, rest, by simp, by simp, hl, hd, h⟩

/-- What the consumer's stage `q` read: its failed waits, the wait that returned true, the slot,
the store, and the arrive. -/
structure CStage where
  ws : List (Nat → List UInt8)
  rw : Nat → List UInt8
  ry : Nat → List UInt8
  ro : Nat → List UInt8
  re : Nat → List UInt8

def CStage.acts (L : Layout) (q : Nat) (c : CStage) : List Act :=
  c.ws.map (Act.mem (waitReq (L.full q))) ++
    [.mem (waitReq (L.full q)) c.rw, .mem (L.ldSlot q) c.ry,
     .mem (L.stOut q ((L.ldSlot q).resp c.ry)[0]!) c.ro, .mem (L.arriveE q) c.re]

def CStage.Ok (L : Layout) (q : Nat) (c : CStage) : Prop :=
  (∀ r ∈ c.ws, ReadOk (waitReq (L.full q)) r) ∧ ReadOk (waitReq (L.full q)) c.rw ∧
  MBarrierWord.parityDone (word64 ((waitReq (L.full q)).resp c.rw)) (q / L.S) = true ∧
  ReadOk (L.ldSlot q) c.ry ∧ ReadOk (L.arriveE q) c.re ∧ (L.arriveE q).unsupported c.re = false

/-- What the producer's stage `q` read: from the second round of its slot on, its failed waits,
the wait that returned true and the fence's instance; then the arrive, and what the copy's agent
read (its load `sr`, its complete `gr`). -/
structure PStage where
  ws : List (Nat → List UInt8) := []
  rw : Nat → List UInt8 := fun _ => []
  inst : Nat := 0
  ra : Nat → List UInt8
  sr : Nat → Nat → List UInt8
  gr : Nat → Nat → List UInt8

def PStage.acts (L : Layout) (q : Nat) (c : PStage) : List Act :=
  (if q < L.S then [] else c.ws.map (Act.mem (waitReq (L.empty q))) ++
      [.mem (waitReq (L.empty q)) c.rw, .sync (.proxyFence .async) c.inst]) ++
    [.mem (L.arriveF q) c.ra, .async (.copy (L.copy q)) (copyActs (L.copy q) c.sr c.gr)]

def PStage.Ok (L : Layout) (q : Nat) (c : PStage) : Prop :=
  (L.S ≤ q → (∀ r ∈ c.ws, ReadOk (waitReq (L.empty q)) r) ∧ ReadOk (waitReq (L.empty q)) c.rw ∧
    MBarrierWord.parityDone (word64 ((waitReq (L.empty q)).resp c.rw)) (q / L.S - 1) = true) ∧
  ReadOk (L.arriveF q) c.ra ∧ (L.arriveF q).unsupported c.ra = false ∧
  ∀ rq read, AAct.mem rq read ∈ copyActs (L.copy q) c.sr c.gr → ReqOk rq read

theorem flatMap_update {α β : Type} {q : Nat} {c : α} {f : Nat → α} (g : Nat → α → List β) :
    ∀ (l : List Nat), (∀ q' ∈ l, q' ≠ q) →
      l.flatMap (fun q' => g q' (if q' = q then c else f q')) = l.flatMap (fun q' => g q' (f q'))
  | [], _ => rfl
  | a :: l, h => by
    have ha : a ≠ q := h a (by simp)
    simp only [List.flatMap_cons, ha, ite_false,
      flatMap_update g l (fun b hb => h b (List.mem_cons_of_mem _ hb))]

/-- The consumer's runs that exit: every stage from `q` on. -/
theorem consumer_run : ∀ (k q : Nat), L.N - q = k → ∀ {d : Bool} {v : UInt32} {acts : List Act},
    Run (step L) { pc := .cstage q, done := d, v } acts .exit →
    ∃ cs : Nat → CStage, acts = (List.range' q (L.N - q)).flatMap (fun q' => (cs q').acts L q') ∧
      ∀ q', q ≤ q' → q' < L.N → (cs q').Ok L q' := by
  intro k
  induction k with
  | zero =>
    intro q hq d v acts h
    have hs : step L { pc := .cstage q, done := d, v } = .exit := by
      simp [step, show ¬ q < L.N by omega]
    obtain ⟨rfl, -⟩ := Run.of_exit hs h
    exact ⟨fun _ => ⟨[], fun _ => [], fun _ => [], fun _ => [], fun _ => []⟩,
      by simp [show L.N - q = 0 by omega], fun q' h₁ h₂ => by omega⟩
  | succ k ih =>
    intro q hq d v acts h
    have hqN : q < L.N := by omega
    rcases Run.of_next (s' := { pc := .cwait q, done := d, v }) (by simp [step, hqN]) h with
      ⟨-, h⟩ | h
    · cases h
    obtain ⟨ws, rw, rest, rfl, hws, hrw, hd, h⟩ := spin (L := L) (w := .cwait q) (c := .ccheck q)
      (A := .load q) (a := L.full q) (p := q / L.S)
      (fun s hs => by simp [step, hs]) (fun s hs => by simp [step, hs]) h
    rcases Run.of_mem (r := L.ldSlot q) (wb := fun r t => { t with v := r[0]! })
      (s' := { pc := .store q, done := true, v }) (by simp [step]) h with
      ⟨-, h⟩ | ⟨read, -, h, -, -⟩ | ⟨ry, rest, rfl, hry, -, h⟩
    · cases h
    · simp [failureOf, Layout.ldSlot] at h
    rcases Run.of_mem (r := L.stOut q ((L.ldSlot q).resp ry)[0]!) (wb := fun _ t => t)
      (s' := { pc := .carrive q, done := true, v := ((L.ldSlot q).resp ry)[0]! })
      (by simp [step]) h with ⟨-, h⟩ | ⟨read, -, h, -, -⟩ | ⟨ro, rest, rfl, -, -, h⟩
    · cases h
    · simp [failureOf, Layout.stOut] at h
    rcases Run.of_mem (r := L.arriveE q) (wb := fun _ t => t)
      (s' := { pc := .cstage (q + 1), done := true, v := ((L.ldSlot q).resp ry)[0]! })
      (by simp [step]) h with ⟨-, h⟩ | ⟨read, -, h, -, -⟩ | ⟨re, rest, rfl, hre, hu, h⟩
    · cases h
    · simp [failureOf, Layout.arriveE, mbarReq] at h
    obtain ⟨cs, hrest, hok⟩ := ih (q + 1) (by omega) h
    refine ⟨fun q' => if q' = q then ⟨ws, rw, ry, ro, re⟩ else cs q', ?_, fun q' h₁ h₂ => ?_⟩
    · rw [show L.N - q = (L.N - (q + 1)) + 1 by omega, List.range'_succ, List.flatMap_cons,
        flatMap_update (fun q' c => CStage.acts L q' c) _
          (fun q' hq' => by simp only [List.mem_range'] at hq'; omega), ← hrest]
      simp [CStage.acts]
    · by_cases he : q' = q
      · subst he; simp only [ite_true]; exact ⟨hws, hrw, hd, hry, hre, hu⟩
      · simp only [he, ite_false]; exact hok q' (by omega) h₂

/-- The producer's stages from `q` on, in a run that exits. -/
theorem producer_stages : ∀ (k q : Nat), L.N - q = k → ∀ {d : Bool} {v : UInt32} {acts : List Act},
    Run (step L) { pc := .stage q, done := d, v } acts .exit →
    ∃ cs : Nat → PStage, acts = (List.range' q (L.N - q)).flatMap (fun q' => (cs q').acts L q') ∧
      ∀ q', q ≤ q' → q' < L.N → (cs q').Ok L q' := by
  intro k
  induction k with
  | zero =>
    intro q hq d v acts h
    have hs : step L { pc := .stage q, done := d, v } = .exit := by
      simp [step, show ¬ q < L.N by omega]
    obtain ⟨rfl, -⟩ := Run.of_exit hs h
    exact ⟨fun _ => ⟨[], fun _ => [], 0, fun _ => [], fun _ _ => [], fun _ _ => []⟩,
      by simp [show L.N - q = 0 by omega], fun q' h₁ h₂ => by omega⟩
  | succ k ih =>
    intro q hq d v acts h
    have hqN : q < L.N := by omega
    -- From the arrive on: the arrive, the copy, the next stages.
    have tail : ∀ {d : Bool} {acts : List Act}, Run (step L) { pc := .arrive q, done := d, v } acts .exit →
        ∃ ra sr gr rest, ∃ cs : Nat → PStage,
          acts = [.mem (L.arriveF q) ra, .async (.copy (L.copy q)) (copyActs (L.copy q) sr gr)] ++ rest ∧
          ReadOk (L.arriveF q) ra ∧ (L.arriveF q).unsupported ra = false ∧
          (∀ rq read, AAct.mem rq read ∈ copyActs (L.copy q) sr gr → ReqOk rq read) ∧
          rest = (List.range' (q + 1) (L.N - (q + 1))).flatMap (fun q' => (cs q').acts L q') ∧
          ∀ q', q + 1 ≤ q' → q' < L.N → (cs q').Ok L q' := by
      intro d acts h
      rcases Run.of_mem (r := L.arriveF q) (wb := fun _ t => t) (s' := { pc := .copy q, done := d, v })
        (by simp [step]) h with ⟨-, h⟩ | ⟨read, -, h, -, -⟩ | ⟨ra, rest, rfl, hra, hu, h⟩
      · cases h
      · simp [failureOf, Layout.arriveF, mbarReq] at h
      rcases Run.of_async (op := .copy (L.copy q)) (wb := fun _ t => t)
        (s' := { pc := .stage (q + 1), done := d, v }) (by simp [step]) h with
        ⟨-, h⟩ | ⟨agent, -, ha⟩ | ⟨agent, res, rest, rfl, ha, h⟩
      · cases h
      · rcases AgentRun.copy_inv ha with ⟨-, -, -, -, h⟩ | ⟨-, -, -, rq, -, -, -, -, -, -, -, h⟩
        · cases h
        · simp only [Except.error.injEq, failureOf] at h; split at h <;> cases h
      obtain ⟨cs, hrest, hok⟩ := ih (q + 1) (by omega) h
      rcases AgentRun.copy_inv ha with ⟨sr, gr, rfl, hok', -⟩ | ⟨_, _, _, _, _, _, _, _, _, _, _, hres⟩
      · exact ⟨ra, sr, gr, rest, cs, rfl, hra, hu, hok', hrest, hok⟩
      · exact absurd hres (by intro h; injection h)
    by_cases hqS : q < L.S
    · rcases Run.of_next (s' := { pc := .arrive q, done := d, v }) (by simp [step, hqN, hqS]) h with
        ⟨-, h⟩ | h
      · cases h
      obtain ⟨ra, sr, gr, rest, cs, rfl, hra, hu, ha, hrest, hok⟩ := tail h
      refine ⟨fun q' => if q' = q then ⟨[], fun _ => [], 0, ra, sr, gr⟩ else cs q', ?_,
        fun q' h₁ h₂ => ?_⟩
      · rw [show L.N - q = (L.N - (q + 1)) + 1 by omega, List.range'_succ, List.flatMap_cons,
          flatMap_update (fun q' c => PStage.acts L q' c) _
            (fun q' hq' => by simp only [List.mem_range'] at hq'; omega), ← hrest]
        simp [PStage.acts, hqS]
      · by_cases he : q' = q
        · subst he; simp only [ite_true]; exact ⟨fun h => absurd h (by omega), hra, hu, ha⟩
        · simp only [he, ite_false]; exact hok q' (by omega) h₂
    · rcases Run.of_next (s' := { pc := .wait q, done := d, v }) (by simp [step, hqN, hqS]) h with
        ⟨-, h⟩ | h
      · cases h
      obtain ⟨ws, rw, rest, rfl, hws, hrw, hd, h⟩ := spin (L := L) (w := .wait q) (c := .check q)
        (A := .fence q) (a := L.empty q) (p := q / L.S - 1)
        (fun s hs => by simp [step, hs]) (fun s hs => by simp [step, hs]) h
      rcases Run.of_sync (op := .proxyFence .async) (s' := { pc := .arrive q, done := true, v })
        (by simp [step]) h with
        ⟨-, h⟩ | ⟨inst, rest', hr', h'⟩
      · cases h
      subst hr'
      obtain ⟨ra, sr, gr, rest, cs, rfl, hra, hu, ha, hrest, hok⟩ := tail h'
      refine ⟨fun q' => if q' = q then ⟨ws, rw, inst, ra, sr, gr⟩ else cs q', ?_,
        fun q' h₁ h₂ => ?_⟩
      · rw [show L.N - q = (L.N - (q + 1)) + 1 by omega, List.range'_succ, List.flatMap_cons,
          flatMap_update (fun q' c => PStage.acts L q' c) _
            (fun q' hq' => by simp only [List.mem_range'] at hq'; omega), ← hrest]
        simp [PStage.acts, hqS]
      · by_cases he : q' = q
        · subst he; simp only [ite_true]; exact ⟨fun _ => ⟨hws, hrw, hd⟩, hra, hu, ha⟩
        · simp only [he, ite_false]; exact hok q' (by omega) h₂

/-- The producer's initializations, from barrier `j` of a kind on, in a run that exits. -/
theorem producer_initE : ∀ (k j : Nat), L.S - j = k → j < L.S → ∀ {d : Bool} {v : UInt32} {acts : List Act},
    Run (step L) { pc := .initE j, done := d, v } acts .exit →
    ∃ (re : Nat → Nat → List UInt8) (cs : Nat → PStage),
      acts = (List.range' j (L.S - j)).map (fun k => .mem (L.initE k) (re k)) ++
        (List.range' 0 L.N).flatMap (fun q' => (cs q').acts L q') ∧
      ∀ q' < L.N, (cs q').Ok L q' := by
  intro k
  induction k with
  | zero => intro j hj hjS; omega
  | succ k ih =>
    intro j hj hjS d v acts h
    rcases Run.of_mem (r := L.initE j) (wb := fun _ t => t)
      (s' := { pc := if j + 1 < L.S then .initE (j + 1) else .stage 0, done := d, v })
      (by simp [step]) h with ⟨-, h⟩ | ⟨read, -, h, -, -⟩ | ⟨r, rest, rfl, -, -, h⟩
    · cases h
    · simp [failureOf, Layout.initE, mbarReq] at h
    by_cases hj1 : j + 1 < L.S
    · simp only [hj1, ite_true] at h
      obtain ⟨re, cs, rfl, hok⟩ := ih (j + 1) (by omega) hj1 h
      refine ⟨fun k => if k = j then r else re k, cs, ?_, hok⟩
      rw [show L.S - j = (L.S - (j + 1)) + 1 by omega, List.range'_succ]
      simp only [List.map_cons, ite_true, List.cons_append]
      congr 2
      apply List.map_congr_left
      intro k hk
      simp only [List.mem_range'] at hk
      simp [show k ≠ j by omega]
    · simp only [hj1, ite_false] at h
      obtain ⟨cs, rfl, hok⟩ := producer_stages _ 0 rfl h
      refine ⟨fun _ => r, cs, ?_, fun q' hq' => hok q' (Nat.zero_le _) hq'⟩
      rw [show L.S - j = 1 by omega]
      simp

theorem producer_initF : ∀ (k j : Nat), L.S - j = k → j < L.S → ∀ {d : Bool} {v : UInt32} {acts : List Act},
    Run (step L) { pc := .initF j, done := d, v } acts .exit →
    ∃ (rf re : Nat → Nat → List UInt8) (cs : Nat → PStage),
      acts = (List.range' j (L.S - j)).map (fun k => .mem (L.initF k) (rf k)) ++
        (List.range' 0 L.S).map (fun k => .mem (L.initE k) (re k)) ++
        (List.range' 0 L.N).flatMap (fun q' => (cs q').acts L q') ∧
      ∀ q' < L.N, (cs q').Ok L q' := by
  intro k
  induction k with
  | zero => intro j hj hjS; omega
  | succ k ih =>
    intro j hj hjS d v acts h
    rcases Run.of_mem (r := L.initF j) (wb := fun _ t => t)
      (s' := { pc := if j + 1 < L.S then .initF (j + 1) else .initE 0, done := d, v })
      (by simp [step]) h with ⟨-, h⟩ | ⟨read, -, h, -, -⟩ | ⟨r, rest, rfl, -, -, h⟩
    · cases h
    · simp [failureOf, Layout.initF, mbarReq] at h
    by_cases hj1 : j + 1 < L.S
    · simp only [hj1, ite_true] at h
      obtain ⟨rf, re, cs, rfl, hok⟩ := ih (j + 1) (by omega) hj1 h
      refine ⟨fun k => if k = j then r else rf k, re, cs, ?_, hok⟩
      rw [show L.S - j = (L.S - (j + 1)) + 1 by omega, List.range'_succ]
      simp only [List.map_cons, ite_true, List.cons_append, List.append_assoc]
      congr 2
      apply List.map_congr_left
      intro k hk
      simp only [List.mem_range'] at hk
      simp [show k ≠ j by omega]
    · simp only [hj1, ite_false] at h
      obtain ⟨re, cs, rfl, hok⟩ := producer_initE _ 0 rfl L.S_pos h
      refine ⟨fun _ => r, re, cs, ?_, hok⟩
      rw [show L.S - j = 1 by omega]
      simp

end runs

/-! ## The run's events -/

section events
variable (L : Layout) (x : Nat → Fin 4 → UInt8)

def pAgent : Agent St := { init := { pc := .initF 0 }, tid := 0, place := ⟨0, 0, 0⟩ }
def cAgent : Agent St := { init := { pc := .cstage 0 }, tid := 1, place := ⟨0, 0, 0⟩ }

theorem agents_eq : (program L x).agents = [pAgent, cAgent] := rfl

theorem threads_eq : (program L x).threads = 2 := rfl

/-- The read and the write of an 8-byte read-modify-write `r` by thread `t` at position `k`, its
read having returned `bs`. -/
def rmwR (r : Req) (t k : Nat) (l : Loc) (bs : List UInt8) : Ev :=
  { e := r.event t ⟨false, 0, 8, true⟩, seq := k, loc := l, size := 8, rmwRead := true,
    val := bs.extract 0 8 }
def rmwW (r : Req) (t k : Nat) (l : Loc) (bs : List UInt8) : Ev :=
  { e := r.event t ⟨true, 0, 8, false⟩, seq := k + 1, loc := l, size := 8,
    val := (r.written 0 ⟨true, 0, 8, false⟩ bs).getD [] }

theorem rmw_events {r : Req} (hp : r.pieces = [⟨false, 0, 8, true⟩, ⟨true, 0, 8, false⟩])
    (t k : Nat) (l : Loc) (bs : List UInt8) :
    r.laneEvents 0 t k l bs = [rmwR r t k l bs, rmwW r t k l bs] := by
  simp [Req.laneEvents, hp, rmwR, rmwW]

theorem pieces_mbar {a : Nat} {op : MBarOp} {sem : NV.Sem} {scope : Scope} :
    (mbarReq a op sem scope).pieces = [⟨false, 0, 8, true⟩, ⟨true, 0, 8, false⟩] := rfl

theorem pieces_wait (a : Nat) : (waitReq a).pieces = [⟨false, 0, 8, false⟩] := by
  simp [Req.pieces, Req.granule, Req.strong, Req.elemBytes, waitReq]

theorem pieces_ld (q : Nat) : (L.ldSlot q).pieces = [⟨false, 0, 4, false⟩] := by
  simp [Req.pieces, Req.granule, Req.strong, Layout.ldSlot]

theorem pieces_st (q : Nat) (v : UInt32) : (L.stOut q v).pieces = [⟨true, 0, 4, false⟩] := by
  simp [Req.pieces, Req.granule, Req.strong, Layout.stOut]

/-- A wait on the barrier word at `a` by thread `t` at position `k` that read `bs`. -/
def waitEv (a t k : Nat) (bs : List UInt8) : Ev := oneEv (waitReq a) t k (sloc a) false 8 bs

/-- Consecutive waits from position `k`. -/
def waitEvs (a t : Nat) : Nat → List (Nat → List UInt8) → List Ev
  | _, [] => []
  | k, w :: ws => waitEv a t k (w 0) :: waitEvs a t (k + 1) ws

theorem wait_mem {a t : Nat} : ∀ {k : Nat} {ws : List (Nat → List UInt8)} {v : Ev},
    v ∈ waitEvs a t k ws → ∃ pos bs, v = waitEv a t pos bs
  | _, [], _, h => by simp [waitEvs] at h
  | k, w :: ws, v, h => by
    simp only [waitEvs, List.mem_cons] at h
    rcases h with rfl | h
    · exact ⟨_, _, rfl⟩
    · exact wait_mem h

theorem fits_full (q : Nat) : L.full q < 2 ^ 64 := by
  have := L.fits; have := L.shared; have := Nat.mod_lt q L.S_pos; unfold Layout.full; omega
theorem fits_empty (q : Nat) : L.empty q < 2 ^ 64 := by
  have := L.fits; have := L.shared; have := Nat.mod_lt q L.S_pos; unfold Layout.empty; omega
theorem fits_slot (q : Nat) : L.slot q < 2 ^ 64 := by
  have := L.fits; have := L.shared; have := Nat.mod_lt q L.S_pos; unfold Layout.slot; omega

theorem agentEvents_waits {ag : Agent St} (hp : ag.place.cta = 0) {a : Nat} (ha : a < 2 ^ 64) :
    ∀ (ws : List (Nat → List UInt8)) (c : Nat → Nat) (rest : List Act),
      (program L x).agentEvents ag c (ws.map (Act.mem (waitReq a)) ++ rest) =
        waitEvs a ag.tid (c ag.tid) ws ++
          (program L x).agentEvents ag (ag.after c (ws.map (Act.mem (waitReq a)))) rest := by
  intro ws
  induction ws with
  | nil => intro c rest; simp [waitEvs]
  | cons w ws ih =>
    intro c rest
    have hw : Act.width ag ag.tid (.mem (waitReq a) w) = 1 := by
      rw [width_one (r := waitReq a) (a := a) rfl, pieces_wait]; rfl
    have haft := (after_cons ag c (.mem (waitReq a) w) (ws.map (Act.mem (waitReq a)))).symm
    simp only [List.map_cons, List.cons_append, Program.agentEvents]
    rw [actEvents_one (program L x) hp rfl ha (.inl rfl), one_events (pieces_wait a), ih, haft, hw]
    rfl

/-- Where the producer's arrive of stage `q` sits in the stage. -/
def pOff (q : Nat) (c : PStage) : Nat := if q < L.S then 0 else c.ws.length + 2

/-- The producer's events of stage `q` from position `k`. -/
def pStageEvs (q : Nat) (c : PStage) (k : Nat) : List Ev :=
  (if q < L.S then [] else
    waitEvs (L.empty q) 0 k c.ws ++
      [waitEv (L.empty q) 0 (k + c.ws.length) (c.rw 0),
       fenceEvent (.proxyFence .async) 0 (k + c.ws.length + 1)]) ++
  [rmwR (L.arriveF q) 0 (k + pOff L q c) (sloc (L.full q)) (c.ra 0),
   rmwW (L.arriveF q) 0 (k + pOff L q c) (sloc (L.full q)) (c.ra 0)]

theorem actEvents_arrive (q : Nat) (c : Nat → Nat) (ra : Nat → List UInt8) :
    (program L x).actEvents pAgent c (.mem (L.arriveF q) ra) =
      [rmwR (L.arriveF q) 0 (c 0) (sloc (L.full q)) (ra 0),
       rmwW (L.arriveF q) 0 (c 0) (sloc (L.full q)) (ra 0)] := by
  rw [actEvents_one (program L x) (r := L.arriveF q) rfl rfl (fits_full L q) (.inl rfl),
    rmw_events (r := L.arriveF q) pieces_mbar]; rfl

theorem actEvents_wait {ag : Agent St} (hp : ag.place.cta = 0) {a : Nat} (ha : a < 2 ^ 64)
    (c : Nat → Nat) (w : Nat → List UInt8) :
    (program L x).actEvents ag c (.mem (waitReq a) w) = [waitEv a ag.tid (c ag.tid) (w 0)] := by
  rw [actEvents_one (program L x) (r := waitReq a) hp rfl ha (.inl rfl), one_events (pieces_wait a)]; rfl

theorem width_wait (ag : Agent St) (a : Nat) (w : Nat → List UInt8) :
    Act.width ag ag.tid (.mem (waitReq a) w) = 1 := by
  rw [width_one (r := waitReq a) (a := a) rfl, pieces_wait]; rfl

theorem span_waits (ag : Agent St) (a : Nat) (ws : List (Nat → List UInt8)) :
    (ws.map ((fun act => Act.width ag ag.tid act) ∘ Act.mem (waitReq a))).sum = ws.length := by
  induction ws with
  | nil => rfl
  | cons w ws ih => simp only [List.map_cons, List.sum_cons, List.length_cons, Function.comp, width_wait] at ih ⊢; omega

theorem actEvents_fence (c : Nat → Nat) (inst : Nat) :
    (program L x).actEvents pAgent c (.sync (.proxyFence .async) inst) =
      [fenceEvent (.proxyFence .async) 0 (c 0)] := by
  simp [Program.actEvents, SyncOp.events, pAgent]

theorem pStage_events (q : Nat) (c : PStage) (ctr : Nat → Nat) :
    (program L x).agentEvents pAgent ctr (c.acts L q) = pStageEvs L q c (ctr 0) := by
  by_cases hq : q < L.S
  · simp only [PStage.acts, hq, ite_true, List.nil_append, Program.agentEvents, actEvents_arrive]
    simp [Program.actEvents, pStageEvs, pOff, hq]
  · simp only [PStage.acts, hq, ite_false, List.append_assoc, List.cons_append, List.nil_append]
    rw [agentEvents_waits L x rfl (fits_empty L q)]
    have hs := span_waits pAgent (L.empty q) c.ws
    simp only [show pAgent.tid = 0 from rfl] at hs
    have hf : Act.width pAgent 0 (.sync (.proxyFence .async) c.inst) = 1 := by simp [Act.width, pAgent]
    have hw := width_wait pAgent (L.empty q) c.rw
    simp only [show pAgent.tid = 0 from rfl] at hw
    simp only [Program.agentEvents, actEvents_wait L x (ag := pAgent) rfl (fits_empty L q),
      actEvents_fence, actEvents_arrive]
    simp [Program.actEvents, pStageEvs, pOff, hq, Agent.after, Agent.span, hs, hw, hf,
      show pAgent.tid = 0 from rfl, Nat.add_assoc]

theorem width_arrive (q : Nat) (ra : Nat → List UInt8) :
    Act.width pAgent 0 (.mem (L.arriveF q) ra) = 2 :=
  width_one (ag := pAgent) (r := L.arriveF q) (a := L.full q) rfl ra

theorem pStage_span (q : Nat) (c : PStage) : pAgent.span 0 (c.acts L q) = pOff L q c + 2 := by
  have hs := span_waits pAgent (L.empty q) c.ws
  simp only [show pAgent.tid = 0 from rfl] at hs
  have hw := width_wait pAgent (L.empty q) c.rw
  simp only [show pAgent.tid = 0 from rfl] at hw
  have hf : Act.width pAgent 0 (.sync (.proxyFence .async) c.inst) = 1 := by simp [Act.width, pAgent]
  have has : Act.width pAgent 0 (.async (.copy (L.copy q)) (copyActs (L.copy q) c.sr c.gr)) = 0 := rfl
  by_cases hq : q < L.S
  · simp only [PStage.acts, hq, ite_true, Agent.span, List.nil_append, List.map_cons, List.map_nil,
      List.sum_cons, List.sum_nil, width_arrive, has, pOff]
    omega
  · simp only [PStage.acts, hq, ite_false, Agent.span, List.map_append, List.map_map, List.sum_append,
      hs, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, hw, hf, width_arrive, has, pOff]
    omega

/-- The consumer's events of stage `q` from position `k`. -/
def cStageEvs (q : Nat) (c : CStage) (k : Nat) : List Ev :=
  waitEvs (L.full q) 1 k c.ws ++
    [waitEv (L.full q) 1 (k + c.ws.length) (c.rw 0),
     oneEv (L.ldSlot q) 1 (k + c.ws.length + 1) (sloc (L.slot q)) false 4 (c.ry 0),
     oneEv (L.stOut q ((L.ldSlot q).resp c.ry)[0]!) 1 (k + c.ws.length + 2)
       (gloc (L.O + 4 * q)) true 4 (c.ro 0),
     rmwR (L.arriveE q) 1 (k + c.ws.length + 3) (sloc (L.empty q)) (c.re 0),
     rmwW (L.arriveE q) 1 (k + c.ws.length + 3) (sloc (L.empty q)) (c.re 0)]

theorem fits_out (q : Nat) (hq : q < L.N) : L.O + 4 * q < 2 ^ 64 := by
  have := L.fits; omega

theorem cStage_events (q : Nat) (hq : q < L.N) (c : CStage) (ctr : Nat → Nat) :
    (program L x).agentEvents cAgent ctr (c.acts L q) = cStageEvs L q c (ctr 1) := by
  simp only [CStage.acts]
  rw [agentEvents_waits L x rfl (fits_full L q)]
  have hs := span_waits cAgent (L.full q) c.ws
  simp only [show cAgent.tid = 1 from rfl] at hs
  have hw := width_wait cAgent (L.full q) c.rw
  simp only [show cAgent.tid = 1 from rfl] at hw
  have hl : Act.width cAgent 1 (.mem (L.ldSlot q) c.ry) = 1 :=
    (width_one (ag := cAgent) (r := L.ldSlot q) (a := L.slot q) rfl c.ry).trans (by rw [pieces_ld]; rfl)
  have hst : Act.width cAgent 1 (.mem (L.stOut q ((L.ldSlot q).resp c.ry)[0]!) c.ro) = 1 :=
    (width_one (ag := cAgent) (r := L.stOut q _) (a := L.O + 4 * q) rfl c.ro).trans
      (by rw [pieces_st]; rfl)
  simp only [Program.agentEvents, actEvents_wait L x (ag := cAgent) rfl (fits_full L q),
    actEvents_one (program L x) (ag := cAgent) (r := L.ldSlot q) rfl rfl (fits_slot L q) (.inl rfl),
    actEvents_one (program L x) (ag := cAgent) (r := L.stOut q _) rfl rfl (fits_out L q hq) (.inr rfl),
    actEvents_one (program L x) (ag := cAgent) (r := L.arriveE q) rfl rfl (fits_empty L q) (.inl rfl),
    one_events (pieces_ld L q), one_events (pieces_st L q _), rmw_events (r := L.arriveE q) pieces_mbar]
  simp only [cStageEvs, Agent.after, Agent.span, List.map_map, hs,
    hw, hl, hst, show cAgent.tid = 1 from rfl,
    waitEv, sloc, gloc]
  simp [Nat.add_assoc, Layout.ldSlot, Layout.stOut, Layout.arriveE, mbarReq]

theorem cStage_span (q : Nat) (c : CStage) : cAgent.span 1 (c.acts L q) = c.ws.length + 5 := by
  have hs := span_waits cAgent (L.full q) c.ws
  simp only [show cAgent.tid = 1 from rfl] at hs
  have hw := width_wait cAgent (L.full q) c.rw
  simp only [show cAgent.tid = 1 from rfl] at hw
  have hl : Act.width cAgent 1 (.mem (L.ldSlot q) c.ry) = 1 :=
    (width_one (ag := cAgent) (r := L.ldSlot q) (a := L.slot q) rfl c.ry).trans (by rw [pieces_ld]; rfl)
  have hst : Act.width cAgent 1 (.mem (L.stOut q ((L.ldSlot q).resp c.ry)[0]!) c.ro) = 1 :=
    (width_one (ag := cAgent) (r := L.stOut q _) (a := L.O + 4 * q) rfl c.ro).trans
      (by rw [pieces_st]; rfl)
  have ha : Act.width cAgent 1 (.mem (L.arriveE q) c.re) = 2 :=
    width_one (ag := cAgent) (r := L.arriveE q) (a := L.empty q) rfl c.re
  simp only [CStage.acts, Agent.span, List.map_append, List.map_map, List.sum_append, hs,
    List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, hw, hl, hst, ha]
  omega

theorem lanes_one {r : Req} {u : UInt64} (ha : r.addrs = #[some u]) :
    (List.range r.addrs.size).filter (r.active ·) = [0] := by
  simp [Req.active, ha]

/-- Stage `q`'s copy: its load of `X`, its store into the slot. -/
def srcReq (q : Nat) : Req := asyncReq .global .ld 4 #[some (UInt64.ofNat (L.X + 4 * q))]
def dstReq (q : Nat) : Req := asyncReq .shared .st 4 #[some (UInt64.ofNat (L.slot q))]
def theMove (q : Nat) : Move := { src := srcReq L q, dsts := #[dstReq L q] }
def storeReq (q : Nat) (bs : Nat → List UInt8) : Req :=
  (theMove L q).store (dstReq L q) ((srcReq L q).resp bs)

theorem copyActs_eq (q : Nat) (sr gr : Nat → Nat → List UInt8) :
    copyActs (L.copy q) sr gr =
      [.mem (srcReq L q) (sr 0), .mem (storeReq L q (sr 0)) (fun _ => []), .proxyFence,
        .mem (L.completeF q) (gr 0)] := by
  simp [copyActs, Layout.copy, srcReq, dstReq, storeReq, theMove]

theorem fits_x (q : Nat) (hq : q < L.N) : L.X + 4 * q < 2 ^ 64 := by
  obtain ⟨h₁, -⟩ := L.fits; have := L.global; omega

/-- Stage `q`'s copy's events: on thread `T`, after its issuer's first `k`. -/
def copyEvs (q : Nat) (c : PStage) (T k : Nat) : List Ev :=
  [oneEv (srcReq L q) T 0 (gloc (L.X + 4 * q)) false 4 (c.sr 0 0),
   oneEv (storeReq L q (c.sr 0)) T 1 (sloc (L.slot q)) true 4 [],
   fenceEvent (.proxyFence .async) T 2,
   rmwR (L.completeF q) T 3 (sloc (L.full q)) (c.gr 0 0),
   rmwW (L.completeF q) T 3 (sloc (L.full q)) (c.gr 0 0)].map ({ · with after := [(0, k)] })

theorem aactEvents_copy (q : Nat) (hq : q < L.N) (c : PStage) (T k : Nat) :
    pAgent.aactEvents T [(0, k)] 0 (copyActs (L.copy q) c.sr c.gr) = copyEvs L q c T k := by
  rw [copyActs_eq]
  have h₁ : pAgent.loc (srcReq L q) 0 = some (gloc (L.X + 4 * q)) := by
    show some ({ space := .global, owner := 0, addr := (UInt64.ofNat (L.X + 4 * q)).toNat } : Loc) = _
    rw [ofNat_toNat (fits_x L q hq)]; rfl
  have h₂ : pAgent.loc (storeReq L q (c.sr 0)) 0 = some (sloc (L.slot q)) := by
    simp [Agent.loc, storeReq, Move.store, theMove, dstReq, asyncReq, sloc, pAgent,
      ofNat_toNat (fits_slot L q)]
  have h₃ : pAgent.loc (L.completeF q) 0 = some (sloc (L.full q)) := by
    simp [Agent.loc, Layout.completeF, mbarReq, sloc, pAgent, ofNat_toNat (fits_full L q)]
  have l₁ := lanes_one (r := srcReq L q) rfl
  have l₂ := lanes_one (r := storeReq L q (c.sr 0)) rfl
  have l₃ := lanes_one (r := L.completeF q) rfl
  have p₁ : (srcReq L q).pieces = [⟨false, 0, 4, false⟩] := by
    simp [Req.pieces, Req.granule, Req.strong, srcReq, asyncReq]
  have p₂ : (storeReq L q (c.sr 0)).pieces = [⟨true, 0, 4, false⟩] := by
    simp [Req.pieces, Req.granule, Req.strong, storeReq, theMove, Move.store, dstReq, asyncReq]
  simp only [Agent.aactEvents, l₁, l₂, l₃, List.zipIdx_cons, List.zipIdx_nil, List.flatMap_cons,
    List.flatMap_nil, List.append_nil, h₁, h₂, h₃, one_events p₁, one_events p₂,
    rmw_events (r := L.completeF q) pieces_mbar, p₁, p₂, List.length_cons, List.length_nil]
  simp [copyEvs]

end events

/-! ## The whole run's events -/

section whole
variable (L : Layout) (x : Nat → Fin 4 → UInt8)

/-- Where the producer's stage `q` starts: after its `4 S` initialization events and the stages
before. -/
def posP (cs : Nat → PStage) (q : Nat) : Nat := 4 * L.S + sumBelow q fun i => pOff L i (cs i) + 2

/-- Where the consumer's stage `q` starts. -/
def posC (cs : Nat → CStage) (q : Nat) : Nat := sumBelow q fun i => (cs i).ws.length + 5

theorem pStages_mem (cs : Nat → PStage) : ∀ (n q : Nat) (ctr : Nat → Nat), ctr 0 = posP L cs q →
    ∀ e, e ∈ (program L x).agentEvents pAgent ctr ((List.range' q n).flatMap fun i => (cs i).acts L i) ↔
      ∃ i, q ≤ i ∧ i < q + n ∧ e ∈ pStageEvs L i (cs i) (posP L cs i) := by
  intro n
  induction n with
  | zero => intro q ctr _ e; simp only [List.range'_zero, List.flatMap_nil, Program.agentEvents]; simp; omega
  | succ n ih =>
    intro q ctr hctr e
    rw [List.range'_succ, List.flatMap_cons, agentEvents_app, pStage_events, List.mem_append,
      ih (q + 1) _ (by simp [Agent.after, pStage_span, hctr, posP, sumBelow_succ]; omega)]
    constructor
    · rintro (h | ⟨i, h₁, h₂, h⟩)
      · exact ⟨q, Nat.le_refl _, by omega, hctr ▸ h⟩
      · exact ⟨i, by omega, by omega, h⟩
    · rintro ⟨i, h₁, h₂, h⟩
      by_cases hi : i = q
      · subst hi; exact .inl (hctr ▸ h)
      · exact .inr ⟨i, by omega, by omega, h⟩

theorem cStages_mem (cs : Nat → CStage) : ∀ (n q : Nat) (ctr : Nat → Nat), q + n ≤ L.N →
    ctr 1 = posC cs q →
    ∀ e, e ∈ (program L x).agentEvents cAgent ctr ((List.range' q n).flatMap fun i => (cs i).acts L i) ↔
      ∃ i, q ≤ i ∧ i < q + n ∧ e ∈ cStageEvs L i (cs i) (posC cs i) := by
  intro n
  induction n with
  | zero => intro q ctr _ _ e; simp only [List.range'_zero, List.flatMap_nil, Program.agentEvents]; simp; omega
  | succ n ih =>
    intro q ctr hn hctr e
    rw [List.range'_succ, List.flatMap_cons, agentEvents_app, cStage_events L x q (by omega),
      List.mem_append,
      ih (q + 1) _ (by omega) (by simp [Agent.after, cStage_span, hctr, posC, sumBelow_succ])]
    constructor
    · rintro (h | ⟨i, h₁, h₂, h⟩)
      · exact ⟨q, Nat.le_refl _, by omega, hctr ▸ h⟩
      · exact ⟨i, by omega, by omega, h⟩
    · rintro ⟨i, h₁, h₂, h⟩
      by_cases hi : i = q
      · subst hi; exact .inl (hctr ▸ h)
      · exact .inr ⟨i, by omega, by omega, h⟩

theorem initF_events (rf : Nat → Nat → List UInt8) : ∀ (n j : Nat) (ctr : Nat → Nat), j + n ≤ L.S →
    ctr 0 = 2 * j → ∀ e,
      e ∈ (program L x).agentEvents pAgent ctr ((List.range' j n).map fun k => .mem (L.initF k) (rf k)) ↔
        ∃ k, j ≤ k ∧ k < j + n ∧ (e = rmwR (L.initF k) 0 (2 * k) (sloc (L.F + 8 * k)) (rf k 0) ∨
          e = rmwW (L.initF k) 0 (2 * k) (sloc (L.F + 8 * k)) (rf k 0)) := by
  intro n
  induction n with
  | zero => intro j ctr _ _ e; simp only [List.range'_zero, List.map_nil, Program.agentEvents]; simp; omega
  | succ n ih =>
    intro j ctr hn hctr e
    have hfit : L.F + 8 * j < 2 ^ 64 := by have := L.fits; have := L.shared; omega
    rw [List.range'_succ, List.map_cons]
    simp only [Program.agentEvents, List.mem_append]
    rw [actEvents_one (program L x) (r := L.initF j) rfl rfl hfit (.inl rfl), rmw_events (r := L.initF j) pieces_mbar,
      ih (j + 1) _ (by omega) (by
        show ctr 0 + Act.width pAgent 0 (.mem (L.initF j) (rf j)) = 2 * (j + 1)
        rw [hctr, show Act.width pAgent 0 (.mem (L.initF j) (rf j)) = 2 from
          width_one (ag := pAgent) (r := L.initF j) (a := L.F + 8 * j) rfl (rf j)]; omega)]
    simp only [show pAgent.tid = 0 from rfl, hctr, List.mem_cons, List.not_mem_nil, or_false]
    constructor
    · rintro ((h | h) | ⟨k, h₁, h₂, h⟩)
      · exact ⟨j, Nat.le_refl _, by omega, .inl (by rw [h]; rfl)⟩
      · exact ⟨j, Nat.le_refl _, by omega, .inr (by rw [h]; rfl)⟩
      · exact ⟨k, by omega, by omega, h⟩
    · rintro ⟨k, h₁, h₂, h⟩
      by_cases hk : k = j
      · subst hk; rcases h with h | h
        · exact .inl (.inl (by rw [h]; rfl))
        · exact .inl (.inr (by rw [h]; rfl))
      · exact .inr ⟨k, by omega, by omega, h⟩

theorem initE_events (re : Nat → Nat → List UInt8) : ∀ (n j : Nat) (ctr : Nat → Nat), j + n ≤ L.S →
    ctr 0 = 2 * L.S + 2 * j → ∀ e,
      e ∈ (program L x).agentEvents pAgent ctr ((List.range' j n).map fun k => .mem (L.initE k) (re k)) ↔
        ∃ k, j ≤ k ∧ k < j + n ∧ (e = rmwR (L.initE k) 0 (2 * L.S + 2 * k) (sloc (L.E + 8 * k)) (re k 0) ∨
          e = rmwW (L.initE k) 0 (2 * L.S + 2 * k) (sloc (L.E + 8 * k)) (re k 0)) := by
  intro n
  induction n with
  | zero => intro j ctr _ _ e; simp only [List.range'_zero, List.map_nil, Program.agentEvents]; simp; omega
  | succ n ih =>
    intro j ctr hn hctr e
    have hfit : L.E + 8 * j < 2 ^ 64 := by have := L.fits; have := L.shared; omega
    rw [List.range'_succ, List.map_cons]
    simp only [Program.agentEvents, List.mem_append]
    rw [actEvents_one (program L x) (r := L.initE j) rfl rfl hfit (.inl rfl), rmw_events (r := L.initE j) pieces_mbar,
      ih (j + 1) _ (by omega) (by
        show ctr 0 + Act.width pAgent 0 (.mem (L.initE j) (re j)) = 2 * L.S + 2 * (j + 1)
        rw [hctr, show Act.width pAgent 0 (.mem (L.initE j) (re j)) = 2 from
          width_one (ag := pAgent) (r := L.initE j) (a := L.E + 8 * j) rfl (re j)]; omega)]
    simp only [show pAgent.tid = 0 from rfl, hctr, List.mem_cons, List.not_mem_nil, or_false]
    constructor
    · rintro ((h | h) | ⟨k, h₁, h₂, h⟩)
      · exact ⟨j, Nat.le_refl _, by omega, .inl (by rw [h]; rfl)⟩
      · exact ⟨j, Nat.le_refl _, by omega, .inr (by rw [h]; rfl)⟩
      · exact ⟨k, by omega, by omega, h⟩
    · rintro ⟨k, h₁, h₂, h⟩
      by_cases hk : k = j
      · subst hk; rcases h with h | h
        · exact .inl (.inl (by rw [h]; rfl))
        · exact .inl (.inr (by rw [h]; rfl))
      · exact .inr ⟨k, by omega, by omega, h⟩

theorem copy_async (q : Nat) (hq : q < L.N) (c : PStage) (ctr : Nat → Nat) (st : AsyncState) :
    (program L x).asyncEvents pAgent ctr st (c.acts L q) =
      copyEvs L q c (2 * (st.k + 1)) (ctr 0 + pOff L q c + 2) ∧
    ((program L x).asyncAfter pAgent st (c.acts L q)).k = st.k + 1 := by
  have hsplit : c.acts L q = ((if q < L.S then [] else c.ws.map (Act.mem (waitReq (L.empty q))) ++
      [.mem (waitReq (L.empty q)) c.rw, .sync (.proxyFence .async) c.inst]) ++
      [.mem (L.arriveF q) c.ra]) ++ [.async (.copy (L.copy q)) (copyActs (L.copy q) c.sr c.gr)] := by
    simp [PStage.acts]
  have hplain : ∀ act ∈ ((if q < L.S then [] else c.ws.map (Act.mem (waitReq (L.empty q))) ++
      [.mem (waitReq (L.empty q)) c.rw, .sync (.proxyFence .async) c.inst]) ++
      [.mem (L.arriveF q) c.ra]), act.plain = true := by
    split
    · simp [Act.plain]
    · simp only [List.mem_append, List.mem_map, List.mem_cons, List.not_mem_nil, or_false]
      rintro act ((⟨a, -, rfl⟩ | rfl | rfl) | rfl) <;> rfl
  have hspan := pStage_span L q c
  rw [hsplit, span_append] at hspan
  have h0 : pAgent.span 0 [.async (.copy (L.copy q)) (copyActs (L.copy q) c.sr c.gr)] = 0 := rfl
  have hk : ∀ st' : AsyncState, ((program L x).asyncAfter pAgent st'
      [.async (.copy (L.copy q)) (copyActs (L.copy q) c.sr c.gr)]).k = st'.k + 1 := fun _ => rfl
  rw [hsplit, asyncEvents_append, Program.asyncEvents_plain _ _ _ _ _ hplain]
  refine ⟨?_, by
    rw [show (program L x).asyncAfter pAgent st (_ ++ _) =
      (program L x).asyncAfter pAgent ((program L x).asyncAfter pAgent st _) _ from List.foldl_append,
      asyncAfter_plain _ _ _ _ hplain, hk]⟩
  rw [asyncAfter_plain _ _ _ _ hplain]
  simp only [Program.asyncEvents, List.nil_append, List.append_nil, List.range_one, List.map_cons,
    List.map_nil, show pAgent.lanes = 1 from rfl, show pAgent.tid = 0 from rfl, Nat.add_zero,
    Program.asyncThread, threads_eq]
  rw [show (pAgent.after ctr _) 0 = ctr 0 + pOff L q c + 2 by
    simp only [Agent.after]; rw [h0] at hspan; omega]
  rw [aactEvents_copy L q hq]

theorem copies_mem (cs : Nat → PStage) : ∀ (n q : Nat) (ctr : Nat → Nat) (st : AsyncState),
    q + n ≤ L.N → ctr 0 = posP L cs q → st.k = q → ∀ e,
      e ∈ (program L x).asyncEvents pAgent ctr st ((List.range' q n).flatMap fun i => (cs i).acts L i) ↔
        ∃ i, q ≤ i ∧ i < q + n ∧
          e ∈ copyEvs L i (cs i) (2 * (i + 1)) (posP L cs i + pOff L i (cs i) + 2) := by
  intro n
  induction n with
  | zero => intro q ctr st _ _ _ e; simp only [List.range'_zero, List.flatMap_nil, Program.asyncEvents]; simp; omega
  | succ n ih =>
    intro q ctr st hn hctr hst e
    obtain ⟨hev, hk⟩ := copy_async L x q (by omega) (cs q) ctr st
    rw [List.range'_succ, List.flatMap_cons, asyncEvents_append, hev, List.mem_append,
      ih (q + 1) _ _ (by omega) (by simp [Agent.after, pStage_span, hctr, posP, sumBelow_succ]; omega)
        (by rw [hk, hst])]
    rw [hst, hctr]
    constructor
    · rintro (h | ⟨i, h₁, h₂, h⟩)
      · exact ⟨q, Nat.le_refl _, by omega, h⟩
      · exact ⟨i, by omega, by omega, h⟩
    · rintro ⟨i, h₁, h₂, h⟩
      by_cases hi : i = q
      · subst hi; exact .inl h
      · exact .inr ⟨i, by omega, by omega, h⟩

theorem copies_eq (cs : Nat → PStage) : ∀ (n q : Nat) (ctr : Nat → Nat) (st : AsyncState),
    q + n ≤ L.N → ctr 0 = posP L cs q → st.k = q →
      (program L x).asyncEvents pAgent ctr st ((List.range' q n).flatMap fun i => (cs i).acts L i) =
        (List.range' q n).flatMap fun i =>
          copyEvs L i (cs i) (2 * (i + 1)) (posP L cs i + pOff L i (cs i) + 2) := by
  intro n
  induction n with
  | zero => intro q ctr st _ _ _; rfl
  | succ n ih =>
    intro q ctr st hn hctr hst
    obtain ⟨hev, hk⟩ := copy_async L x q (by omega) (cs q) ctr st
    rw [List.range'_succ, List.flatMap_cons, List.flatMap_cons, asyncEvents_append, hev,
      ih (q + 1) _ _ (by omega) (by simp [Agent.after, pStage_span, hctr, posP, sumBelow_succ]; omega)
        (by rw [hk, hst]), hst, hctr]

/-- The acts of a one-lane request have distinct events. -/
theorem actEvents_nodup_mem (ag : Agent St) {r : Req} {u : UInt64} (ha : r.addrs = #[some u])
    (c : Nat → Nat) (read : Nat → List UInt8) : ((program L x).actEvents ag c (.mem r read)).Nodup := by
  simp only [Program.actEvents, ha, Array.size_singleton, List.range_one, List.flatMap_cons,
    List.flatMap_nil, List.append_nil]
  split
  · exact laneEvents_nodup _ _ _ _ _ _
  · exact List.nodup_nil

theorem copyEvs_nodup (q : Nat) (c : PStage) (T k : Nat) : (copyEvs L q c T k).Nodup := by
  apply nodup_of_map Ev.seq
  have : (copyEvs L q c T k).map Ev.seq = [0, 1, 2, 3, 4] := by
    simp [copyEvs, oneEv, fenceEvent, rmwR, rmwW]
  rw [this]; decide

theorem copyEvs_thread {q : Nat} {c : PStage} {T k : Nat} {e : Ev} (h : e ∈ copyEvs L q c T k) :
    e.e.thread = T := by
  simp only [copyEvs, List.map_cons, List.map_nil, List.mem_cons, List.not_mem_nil, or_false] at h
  rcases h with rfl | rfl | rfl | rfl | rfl <;> simp [oneEv, fenceEvent, rmwR, rmwW, Req.event]

theorem span_inits (rf : Nat → Nat → List UInt8) (R : Nat → Req) (A : Nat → Nat)
    (hR : ∀ k, R k = mbarReq (A k) (.init 1) .weak .cta) :
    ∀ (n j : Nat), pAgent.span 0 ((List.range' j n).map fun k => .mem (R k) (rf k)) = 2 * n := by
  intro n
  induction n with
  | zero => intro j; rfl
  | succ n ih =>
    intro j
    rw [List.range'_succ, List.map_cons, span_cons, ih]
    rw [show Act.width pAgent 0 (.mem (R j) (rf j)) = 2 by
      rw [hR]; exact width_one (ag := pAgent) (r := mbarReq (A j) _ _ _) (a := A j) rfl (rf j)]
    omega

/-- The producer's acts in a run that exits. -/
def pActs (rf re : Nat → Nat → List UInt8) (cs : Nat → PStage) : List Act :=
  (List.range' 0 L.S).map (fun k => .mem (L.initF k) (rf k)) ++
    (List.range' 0 L.S).map (fun k => .mem (L.initE k) (re k)) ++
    (List.range' 0 L.N).flatMap fun q => (cs q).acts L q

/-- The consumer's acts in a run that exits. -/
def cActs (cs : Nat → CStage) : List Act := (List.range' 0 L.N).flatMap fun q => (cs q).acts L q

/-- **The events of a run in which both threads exit.** -/
theorem events_mem (rf re : Nat → Nat → List UInt8) (pcs : Nat → PStage) (ccs : Nat → CStage)
    (e : Ev) :
    e ∈ (program L x).events [(pActs L rf re pcs, .exit), (cActs L ccs, .exit)] ↔
      (∃ k < L.S, e = rmwR (L.initF k) 0 (2 * k) (sloc (L.F + 8 * k)) (rf k 0) ∨
        e = rmwW (L.initF k) 0 (2 * k) (sloc (L.F + 8 * k)) (rf k 0)) ∨
      (∃ k < L.S, e = rmwR (L.initE k) 0 (2 * L.S + 2 * k) (sloc (L.E + 8 * k)) (re k 0) ∨
        e = rmwW (L.initE k) 0 (2 * L.S + 2 * k) (sloc (L.E + 8 * k)) (re k 0)) ∨
      (∃ q < L.N, e ∈ pStageEvs L q (pcs q) (posP L pcs q)) ∨
      (∃ q < L.N, e ∈ copyEvs L q (pcs q) (2 * (q + 1)) (posP L pcs q + pOff L q (pcs q) + 2)) ∨
      (∃ q < L.N, e ∈ cStageEvs L q (ccs q) (posC ccs q)) := by
  have hplainI : ∀ act ∈ (List.range' 0 L.S).map (fun k => Act.mem (L.initF k) (rf k)) ++
      (List.range' 0 L.S).map (fun k => Act.mem (L.initE k) (re k)), act.plain = true := by
    simp only [List.mem_append, List.mem_map]
    rintro act (⟨a, -, rfl⟩ | ⟨a, -, rfl⟩) <;> rfl
  have hplainC : ∀ act ∈ (List.range' 0 L.N).flatMap (fun q => (ccs q).acts L q), act.plain = true := by
    simp only [CStage.acts, List.mem_flatMap, List.mem_append, List.mem_map, List.mem_cons,
      List.not_mem_nil, or_false]
    rintro act ⟨q, -, (⟨a, -, rfl⟩ | rfl | rfl | rfl | rfl | rfl)⟩ <;> rfl
  have hF := span_inits rf L.initF (fun k => L.F + 8 * k) (fun _ => rfl) L.S 0
  have hE := span_inits re L.initE (fun k => L.E + 8 * k) (fun _ => rfl) L.S 0
  simp only [Program.events, agents_eq, List.zip_cons_cons, List.zip_nil_left, List.flatMap_cons,
    List.flatMap_nil, List.append_nil, List.mem_append]
  simp only [pActs, cActs, agentEvents_app, asyncEvents_append, List.mem_append,
    Program.asyncEvents_plain _ _ _ _ _ hplainI, Program.asyncEvents_plain _ _ _ _ _ hplainC,
    asyncAfter_plain _ _ _ _ hplainI]
  rw [initF_events L x rf L.S 0 _ (by omega) (by rfl), initE_events L x re L.S 0 _ (by omega)
      (by simp [Agent.after, hF]),
    pStages_mem L x pcs L.N 0 _ (by simp [Agent.after, span_append, hF, hE, posP, sumBelow]; omega),
    copies_mem L x pcs L.N 0 _ _ (by omega)
      (by simp [Agent.after, span_append, hF, hE, posP, sumBelow]; omega) (by rfl),
    cStages_mem L x ccs L.N 0 _ (by omega) (by simp [posC, sumBelow])]
  simp only [Nat.zero_le, true_and, Nat.zero_add, List.not_mem_nil, or_false, false_or, or_assoc]

theorem pStageEvs_thread {q : Nat} {c : PStage} {k : Nat} {e : Ev} (h : e ∈ pStageEvs L q c k) :
    e.e.thread = 0 := by
  by_cases hq : q < L.S
  · simp only [pStageEvs, hq, ite_true, List.nil_append, List.mem_cons, List.not_mem_nil,
      or_false] at h
    rcases h with rfl | rfl <;> simp [rmwR, rmwW, Req.event]
  · simp only [pStageEvs, hq, ite_false, List.mem_append, List.mem_cons, List.not_mem_nil,
      or_false] at h
    rcases h with ((h | rfl | rfl) | rfl | rfl)
    · obtain ⟨pos, bs, rfl⟩ := wait_mem h; simp [waitEv, oneEv, Req.event]
    all_goals simp [waitEv, oneEv, fenceEvent, rmwR, rmwW, Req.event]

theorem cStageEvs_thread {q : Nat} {c : CStage} {k : Nat} {e : Ev} (h : e ∈ cStageEvs L q c k) :
    e.e.thread = 1 := by
  simp only [cStageEvs, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at h
  rcases h with h | rfl | rfl | rfl | rfl | rfl
  · obtain ⟨pos, bs, rfl⟩ := wait_mem h; simp [waitEv, oneEv, Req.event]
  all_goals simp [waitEv, oneEv, rmwR, rmwW, Req.event]

theorem p_acts_nodup (rf re : Nat → Nat → List UInt8) (pcs : Nat → PStage) :
    ∀ act ∈ pActs L rf re pcs, ∀ ctr, ((program L x).actEvents pAgent ctr act).Nodup := by
  intro act h ctr
  simp only [pActs, List.mem_append, List.mem_map, List.mem_flatMap] at h
  rcases h with (⟨k, -, rfl⟩ | ⟨k, -, rfl⟩) | ⟨q, -, h⟩
  · exact actEvents_nodup_mem L x pAgent rfl _ _
  · exact actEvents_nodup_mem L x pAgent rfl _ _
  · by_cases hq : q < L.S
    · simp only [PStage.acts, hq, ite_true, List.nil_append, List.mem_cons, List.not_mem_nil,
        or_false] at h
      rcases h with rfl | rfl
      · exact actEvents_nodup_mem L x pAgent rfl _ _
      · simp [Program.actEvents]
    · simp only [PStage.acts, hq, ite_false, List.mem_append, List.mem_map, List.mem_cons,
        List.not_mem_nil, or_false] at h
      rcases h with ((⟨w, -, rfl⟩ | rfl | rfl) | rfl | rfl)
      · exact actEvents_nodup_mem L x pAgent rfl _ _
      · exact actEvents_nodup_mem L x pAgent rfl _ _
      · rw [actEvents_fence]; simp
      · exact actEvents_nodup_mem L x pAgent rfl _ _
      · simp [Program.actEvents]

theorem c_acts_nodup (ccs : Nat → CStage) :
    ∀ act ∈ cActs L ccs, ∀ ctr, ((program L x).actEvents cAgent ctr act).Nodup := by
  intro act h ctr
  simp only [cActs, CStage.acts, List.mem_flatMap, List.mem_append, List.mem_map, List.mem_cons,
    List.not_mem_nil, or_false] at h
  rcases h with ⟨q, -, (⟨w, -, rfl⟩ | rfl | rfl | rfl | rfl)⟩ <;>
    exact actEvents_nodup_mem L x cAgent rfl _ _

/-- **The run's events are distinct.** -/
theorem events_nodup (rf re : Nat → Nat → List UInt8) (pcs : Nat → PStage) (ccs : Nat → CStage) :
    ((program L x).events [(pActs L rf re pcs, .exit), (cActs L ccs, .exit)]).Nodup := by
  have hplainI : ∀ act ∈ (List.range' 0 L.S).map (fun k => Act.mem (L.initF k) (rf k)) ++
      (List.range' 0 L.S).map (fun k => Act.mem (L.initE k) (re k)), act.plain = true := by
    simp only [List.mem_append, List.mem_map]
    rintro act (⟨a, -, rfl⟩ | ⟨a, -, rfl⟩) <;> rfl
  have hplainC : ∀ act ∈ cActs L ccs, act.plain = true := by
    simp only [cActs, CStage.acts, List.mem_flatMap, List.mem_append, List.mem_map, List.mem_cons,
      List.not_mem_nil, or_false]
    rintro act ⟨q, -, (⟨a, -, rfl⟩ | rfl | rfl | rfl | rfl | rfl)⟩ <;> rfl
  have hF := span_inits rf L.initF (fun k => L.F + 8 * k) (fun _ => rfl) L.S 0
  have hE := span_inits re L.initE (fun k => L.E + 8 * k) (fun _ => rfl) L.S 0
  have hB : (program L x).asyncEvents pAgent (fun _ => 0) {} (pActs L rf re pcs) =
      (List.range' 0 L.N).flatMap fun i =>
        copyEvs L i (pcs i) (2 * (i + 1)) (posP L pcs i + pOff L i (pcs i) + 2) := by
    rw [pActs, asyncEvents_append, Program.asyncEvents_plain _ _ _ _ _ hplainI,
      asyncAfter_plain _ _ _ _ hplainI, List.nil_append]
    exact copies_eq L x pcs L.N 0 _ _ (by omega)
      (by simp [Agent.after, span_append, hF, hE, posP, sumBelow]; omega) rfl
  have hAt : ∀ e ∈ (program L x).agentEvents pAgent (fun _ => 0) (pActs L rf re pcs),
      e.e.thread = 0 := by
    intro e he
    simp only [pActs, agentEvents_app, List.mem_append] at he
    rw [initF_events L x rf L.S 0 _ (by omega) (by rfl), initE_events L x re L.S 0 _ (by omega)
        (by simp [Agent.after, hF]),
      pStages_mem L x pcs L.N 0 _ (by simp [Agent.after, span_append, hF, hE, posP, sumBelow]; omega)]
      at he
    rcases he with (⟨k, -, -, rfl | rfl⟩ | ⟨k, -, -, rfl | rfl⟩) | ⟨q, -, -, h⟩
    all_goals first | exact pStageEvs_thread L h | simp [rmwR, rmwW, Req.event]
  have hBt : ∀ e ∈ (program L x).asyncEvents pAgent (fun _ => 0) {} (pActs L rf re pcs),
      2 ≤ e.e.thread := by
    intro e he
    rw [hB] at he
    obtain ⟨i, -, h⟩ := List.mem_flatMap.1 he
    rw [copyEvs_thread L h]; omega
  have hCt : ∀ e ∈ (program L x).agentEvents cAgent (fun _ => 0) (cActs L ccs), e.e.thread = 1 := by
    intro e he
    rw [cActs, cStages_mem L x ccs L.N 0 _ (by omega) (by simp [posC, sumBelow])] at he
    obtain ⟨q, -, -, h⟩ := he
    exact cStageEvs_thread L h
  simp only [Program.events, agents_eq, List.zip_cons_cons, List.zip_nil_left, List.flatMap_cons,
    List.flatMap_nil, List.append_nil, Program.asyncEvents_plain _ _ _ _ _ hplainC]
  refine List.nodup_append.2 ⟨List.nodup_append.2 ⟨agentEvents_nodup _ _ _ _ (p_acts_nodup L x rf re pcs),
    ?_, fun a ha b hb e => by have := hAt a ha; have := hBt b hb; subst e; omega⟩,
    agentEvents_nodup _ _ _ _ (c_acts_nodup L x ccs), fun a ha b hb e => ?_⟩
  · rw [hB]
    refine nodup_flatMap (fun i _ => copyEvs_nodup L i _ _ _) ?_
    exact (List.nodup_range' (s := 0) (n := L.N)).imp fun hne a ha b hb e => by
      have h₁ := copyEvs_thread L ha; have h₂ := copyEvs_thread L hb
      subst e; rw [h₁] at h₂; omega
  · subst e
    have := hCt _ hb
    rcases List.mem_append.1 ha with h | h
    · have := hAt _ h; omega
    · have := hBt _ h; omega

end whole

/-! ## The run's named events -/

section named
variable (L : Layout) (rf re : Nat → Nat → List UInt8) (pcs : Nat → PStage) (ccs : Nat → CStage)

/-- An event of stage `q`'s copy: on its thread, after the issuer's events of the stage's arrive. -/
def aft (k : Nat) (v : Ev) : Ev := { v with after := [(0, k)] }

/-- Stage `q`'s copy's thread and its issue position. -/
def cpT (q : Nat) : Nat := 2 * (q + 1)
def cpK (q : Nat) : Nat := posP L pcs q + pOff L q (pcs q) + 2

def aRe (q : Nat) : Ev :=
  rmwR (L.arriveF q) 0 (posP L pcs q + pOff L q (pcs q)) (sloc (L.full q)) ((pcs q).ra 0)
def aWe (q : Nat) : Ev :=
  rmwW (L.arriveF q) 0 (posP L pcs q + pOff L q (pcs q)) (sloc (L.full q)) ((pcs q).ra 0)
def peE (q : Nat) : Ev := waitEv (L.empty q) 0 (posP L pcs q + (pcs q).ws.length) ((pcs q).rw 0)
def gE (q : Nat) : Ev := fenceEvent (.proxyFence .async) 0 (posP L pcs q + (pcs q).ws.length + 1)
def xRe (q : Nat) : Ev :=
  aft (cpK L pcs q) (oneEv (srcReq L q) (cpT q) 0 (gloc (L.X + 4 * q)) false 4 ((pcs q).sr 0 0))
def wYe (q : Nat) : Ev :=
  aft (cpK L pcs q) (oneEv (storeReq L q ((pcs q).sr 0)) (cpT q) 1 (sloc (L.slot q)) true 4 [])
def fCe (q : Nat) : Ev := aft (cpK L pcs q) (fenceEvent (.proxyFence .async) (cpT q) 2)
def cRe (q : Nat) : Ev :=
  aft (cpK L pcs q) (rmwR (L.completeF q) (cpT q) 3 (sloc (L.full q)) ((pcs q).gr 0 0))
def cWe (q : Nat) : Ev :=
  aft (cpK L pcs q) (rmwW (L.completeF q) (cpT q) 3 (sloc (L.full q)) ((pcs q).gr 0 0))
def rfE (q : Nat) : Ev := waitEv (L.full q) 1 (posC ccs q + (ccs q).ws.length) ((ccs q).rw 0)
def ldE (q : Nat) : Ev :=
  oneEv (L.ldSlot q) 1 (posC ccs q + (ccs q).ws.length + 1) (sloc (L.slot q)) false 4 ((ccs q).ry 0)
def stE (q : Nat) : Ev :=
  oneEv (L.stOut q ((L.ldSlot q).resp (ccs q).ry)[0]!) 1 (posC ccs q + (ccs q).ws.length + 2)
    (gloc (L.O + 4 * q)) true 4 ((ccs q).ro 0)
def eRe (q : Nat) : Ev :=
  rmwR (L.arriveE q) 1 (posC ccs q + (ccs q).ws.length + 3) (sloc (L.empty q)) ((ccs q).re 0)
def eWe (q : Nat) : Ev :=
  rmwW (L.arriveE q) 1 (posC ccs q + (ccs q).ws.length + 3) (sloc (L.empty q)) ((ccs q).re 0)
def iFe (k : Nat) : Ev := rmwW (L.initF k) 0 (2 * k) (sloc (L.F + 8 * k)) (rf k 0)
def iEe (k : Nat) : Ev := rmwW (L.initE k) 0 (2 * L.S + 2 * k) (sloc (L.E + 8 * k)) (re k 0)

theorem aWe_mem {q : Nat} : aRe L pcs q ∈ pStageEvs L q (pcs q) (posP L pcs q) ∧
    aWe L pcs q ∈ pStageEvs L q (pcs q) (posP L pcs q) := by
  by_cases hq : q < L.S <;> simp [pStageEvs, hq, aRe, aWe]

theorem peE_mem {q : Nat} (hq : L.S ≤ q) : peE L pcs q ∈ pStageEvs L q (pcs q) (posP L pcs q) ∧
    gE L pcs q ∈ pStageEvs L q (pcs q) (posP L pcs q) := by
  simp [pStageEvs, show ¬ q < L.S by omega, peE, gE]

theorem copy_mem {q : Nat} :
    ∀ v ∈ [xRe L pcs q, wYe L pcs q, fCe L pcs q, cRe L pcs q, cWe L pcs q],
      v ∈ copyEvs L q (pcs q) (2 * (q + 1)) (posP L pcs q + pOff L q (pcs q) + 2) := by
  simp [copyEvs, xRe, wYe, fCe, cRe, cWe, aft, cpT, cpK]

theorem cons_mem {q : Nat} :
    ∀ v ∈ [rfE L ccs q, ldE L ccs q, stE L ccs q, eRe L ccs q, eWe L ccs q],
      v ∈ cStageEvs L q (ccs q) (posC ccs q) := by
  simp [cStageEvs, rfE, ldE, stE, eRe, eWe]

end named

/-! ## The run's execution -/

section exec
variable {L : Layout} {x : Nat → Fin 4 → UInt8} (c : Candidate (program L x))
  {rf re : Nat → Nat → List UInt8} {pcs : Nat → PStage} {ccs : Nat → CStage}
  (hE : ∀ v, v ∈ c.evs ↔
    v ∈ (program L x).events [(pActs L rf re pcs, .exit), (cActs L ccs, .exit)])
include hE

omit hE in
theorem place_eq (t : Nat) : (program L x).place t = ⟨0, 0, 0⟩ :=
  Program.place_all (f := fun p => p = ⟨0, 0, 0⟩) (fun a ha => by
    simp only [program, List.mem_cons, List.not_mem_nil, or_false] at ha
    rcases ha with rfl | rfl <;> rfl) rfl t

theorem mem_p {q : Nat} (hq : q < L.N) {v : Ev} (h : v ∈ pStageEvs L q (pcs q) (posP L pcs q)) :
    v ∈ c.evs := (hE v).2 ((events_mem L x rf re pcs ccs v).2 (.inr (.inr (.inl ⟨q, hq, h⟩))))

theorem mem_cp {q : Nat} (hq : q < L.N) {v : Ev}
    (h : v ∈ copyEvs L q (pcs q) (2 * (q + 1)) (posP L pcs q + pOff L q (pcs q) + 2)) : v ∈ c.evs :=
  (hE v).2 ((events_mem L x rf re pcs ccs v).2 (.inr (.inr (.inr (.inl ⟨q, hq, h⟩)))))

theorem mem_c {q : Nat} (hq : q < L.N) {v : Ev} (h : v ∈ cStageEvs L q (ccs q) (posC ccs q)) :
    v ∈ c.evs := (hE v).2 ((events_mem L x rf re pcs ccs v).2 (.inr (.inr (.inr (.inr ⟨q, hq, h⟩)))))

theorem mem_iF {k : Nat} (hk : k < L.S) : iFe L rf k ∈ c.evs :=
  (hE _).2 ((events_mem L x rf re pcs ccs _).2 (.inl ⟨k, hk, .inr rfl⟩))

theorem mem_iE {k : Nat} (hk : k < L.S) : iEe L re k ∈ c.evs :=
  (hE _).2 ((events_mem L x rf re pcs ccs _).2 (.inr (.inl ⟨k, hk, .inr rfl⟩)))

omit hE in
theorem full_eq {q s : Nat} : L.F + 8 * s = L.full q ↔ q % L.S = s := by
  unfold Layout.full; omega

/-- **The events at a full barrier**: its initialization, the arrives and completes of its slot's
stages, and the consumer's waits on it. -/
theorem on_full {v : Ev} (hv : v ∈ c.evs) {s : Nat} (hs : s < L.S)
    (hk : v.key = (sloc (L.F + 8 * s), 8)) :
    v = iFe L rf s ∨ v = rmwR (L.initF s) 0 (2 * s) (sloc (L.F + 8 * s)) (rf s 0) ∨
      ∃ q < L.N, q % L.S = s ∧ (v = aRe L pcs q ∨ v = aWe L pcs q ∨ v = cRe L pcs q ∨
        v = cWe L pcs q ∨ ∃ pos bs, v = waitEv (L.full q) 1 pos bs) := by
  have hsh := L.shared
  have hS := L.S_pos
  rcases (events_mem L x rf re pcs ccs v).1 ((hE v).1 hv) with
    ⟨k, hk', h | h⟩ | ⟨k, hk', h | h⟩ | ⟨q, hq, h⟩ | ⟨q, hq, h⟩ | ⟨q, hq, h⟩ <;>
    (try subst h)
  · have : k = s := by simp [rmwR, Ev.key, sloc] at hk; omega
    subst this; exact .inr (.inl rfl)
  · have : k = s := by simp [rmwW, Ev.key, sloc] at hk; omega
    subst this; exact .inl rfl
  · simp [rmwR, Ev.key, sloc] at hk; omega
  · simp [rmwW, Ev.key, sloc] at hk; omega
  · have hfq := (full_eq (L := L) (q := q) (s := s))
    have hlt := Nat.mod_lt q hS
    by_cases hqS : q < L.S
    · simp only [pStageEvs, hqS, ite_true, List.nil_append, List.mem_cons, List.not_mem_nil,
        or_false] at h
      rcases h with rfl | rfl
      · exact .inr (.inr ⟨q, hq, hfq.1 (by simp [rmwR, Ev.key, sloc] at hk; omega), .inl rfl⟩)
      · exact .inr (.inr ⟨q, hq, hfq.1 (by simp [rmwW, Ev.key, sloc] at hk; omega), .inr (.inl rfl)⟩)
    · simp only [pStageEvs, hqS, ite_false, List.mem_append, List.mem_cons, List.not_mem_nil,
        or_false] at h
      rcases h with ((h | rfl | rfl) | rfl | rfl)
      · obtain ⟨pos, bs, rfl⟩ := wait_mem h
        simp [waitEv, oneEv, Ev.key, sloc, Layout.empty] at hk; omega
      · simp [waitEv, oneEv, Ev.key, sloc, Layout.empty] at hk; omega
      · simp [fenceEvent, Ev.key] at hk
      · exact .inr (.inr ⟨q, hq, hfq.1 (by simp [rmwR, Ev.key, sloc] at hk; omega), .inl rfl⟩)
      · exact .inr (.inr ⟨q, hq, hfq.1 (by simp [rmwW, Ev.key, sloc] at hk; omega), .inr (.inl rfl)⟩)
  · have hfq := (full_eq (L := L) (q := q) (s := s))
    simp only [copyEvs, List.map_cons, List.map_nil, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl | rfl
    · simp [oneEv, Ev.key, gloc, sloc] at hk
    · simp [oneEv, Ev.key] at hk
    · simp [fenceEvent, Ev.key] at hk
    · exact .inr (.inr ⟨q, hq, hfq.1 (by simp [rmwR, Ev.key, sloc] at hk; omega),
        .inr (.inr (.inl rfl))⟩)
    · exact .inr (.inr ⟨q, hq, hfq.1 (by simp [rmwW, Ev.key, sloc] at hk; omega),
        .inr (.inr (.inr (.inl rfl)))⟩)
  · have hfq := (full_eq (L := L) (q := q) (s := s))
    have hlt := Nat.mod_lt q hS
    simp only [cStageEvs, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with h | rfl | rfl | rfl | rfl | rfl
    · obtain ⟨pos, bs, rfl⟩ := wait_mem h
      exact .inr (.inr ⟨q, hq, hfq.1 (by simp [waitEv, oneEv, Ev.key, sloc] at hk; omega),
        .inr (.inr (.inr (.inr ⟨pos, bs, rfl⟩)))⟩)
    · exact .inr (.inr ⟨q, hq, hfq.1 (by simp [waitEv, oneEv, Ev.key, sloc] at hk; omega),
        .inr (.inr (.inr (.inr ⟨_, _, rfl⟩)))⟩)
    · simp [oneEv, Ev.key] at hk
    · simp [oneEv, Ev.key, gloc, sloc] at hk
    · simp [rmwR, Ev.key, sloc, Layout.empty] at hk; omega
    · simp [rmwW, Ev.key, sloc, Layout.empty] at hk; omega

omit hE in
theorem empty_eq {q s : Nat} : L.E + 8 * s = L.empty q ↔ q % L.S = s := by
  unfold Layout.empty; omega

omit hE in
theorem slot_eq {q s : Nat} : L.Y + 4 * s = L.slot q ↔ q % L.S = s := by
  unfold Layout.slot; omega

/-- **The events at an empty barrier**: its initialization, the consumer's arrives of its slot's
stages, and the producer's waits on it. -/
theorem on_empty {v : Ev} (hv : v ∈ c.evs) {s : Nat} (hs : s < L.S)
    (hk : v.key = (sloc (L.E + 8 * s), 8)) :
    v = iEe L re s ∨ v = rmwR (L.initE s) 0 (2 * L.S + 2 * s) (sloc (L.E + 8 * s)) (re s 0) ∨
      ∃ q < L.N, q % L.S = s ∧ (v = eRe L ccs q ∨ v = eWe L ccs q ∨
        ∃ pos bs, v = waitEv (L.empty q) 0 pos bs) := by
  have hsh := L.shared
  have hS := L.S_pos
  rcases (events_mem L x rf re pcs ccs v).1 ((hE v).1 hv) with
    ⟨k, hk', h | h⟩ | ⟨k, hk', h | h⟩ | ⟨q, hq, h⟩ | ⟨q, hq, h⟩ | ⟨q, hq, h⟩ <;>
    (try subst h)
  · simp [rmwR, Ev.key, sloc] at hk; omega
  · simp [rmwW, Ev.key, sloc] at hk; omega
  · have : k = s := by simp [rmwR, Ev.key, sloc] at hk; omega
    subst this; exact .inr (.inl rfl)
  · have : k = s := by simp [rmwW, Ev.key, sloc] at hk; omega
    subst this; exact .inl rfl
  · have hfq := (empty_eq (L := L) (q := q) (s := s))
    have hlt := Nat.mod_lt q hS
    by_cases hqS : q < L.S
    · simp only [pStageEvs, hqS, ite_true, List.nil_append, List.mem_cons, List.not_mem_nil,
        or_false] at h
      rcases h with rfl | rfl
      · simp [rmwR, Ev.key, sloc, Layout.full] at hk; omega
      · simp [rmwW, Ev.key, sloc, Layout.full] at hk; omega
    · simp only [pStageEvs, hqS, ite_false, List.mem_append, List.mem_cons, List.not_mem_nil,
        or_false] at h
      rcases h with ((h | rfl | rfl) | rfl | rfl)
      · obtain ⟨pos, bs, rfl⟩ := wait_mem h
        exact .inr (.inr ⟨q, hq, hfq.1 (by simp [waitEv, oneEv, Ev.key, sloc] at hk; omega),
          .inr (.inr ⟨pos, bs, rfl⟩)⟩)
      · exact .inr (.inr ⟨q, hq, hfq.1 (by simp [waitEv, oneEv, Ev.key, sloc] at hk; omega),
          .inr (.inr ⟨_, _, rfl⟩)⟩)
      · simp [fenceEvent, Ev.key] at hk
      · simp [rmwR, Ev.key, sloc, Layout.full] at hk; omega
      · simp [rmwW, Ev.key, sloc, Layout.full] at hk; omega
  · simp only [copyEvs, List.map_cons, List.map_nil, List.mem_cons, List.not_mem_nil, or_false] at h
    have hlt := Nat.mod_lt q hS
    rcases h with rfl | rfl | rfl | rfl | rfl
    · simp [oneEv, Ev.key, gloc, sloc] at hk
    · simp [oneEv, Ev.key] at hk
    · simp [fenceEvent, Ev.key] at hk
    · simp [rmwR, Ev.key, sloc, Layout.full] at hk; omega
    · simp [rmwW, Ev.key, sloc, Layout.full] at hk; omega
  · have hfq := (empty_eq (L := L) (q := q) (s := s))
    have hlt := Nat.mod_lt q hS
    simp only [cStageEvs, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with h | rfl | rfl | rfl | rfl | rfl
    · obtain ⟨pos, bs, rfl⟩ := wait_mem h
      simp [waitEv, oneEv, Ev.key, sloc, Layout.full] at hk; omega
    · simp [waitEv, oneEv, Ev.key, sloc, Layout.full] at hk; omega
    · simp [oneEv, Ev.key] at hk
    · simp [oneEv, Ev.key, gloc, sloc] at hk
    · exact .inr (.inr ⟨q, hq, hfq.1 (by simp [rmwR, Ev.key, sloc] at hk; omega), .inl rfl⟩)
    · exact .inr (.inr ⟨q, hq, hfq.1 (by simp [rmwW, Ev.key, sloc] at hk; omega), .inr (.inl rfl)⟩)

/-- **The events at a slot**: the copies' stores into it and the consumer's loads of it. -/
theorem on_slot {v : Ev} (hv : v ∈ c.evs) {s : Nat}
    (hk : v.key = (sloc (L.Y + 4 * s), 4)) :
    ∃ q < L.N, q % L.S = s ∧ (v = wYe L pcs q ∨ v = ldE L ccs q) := by
  have hsh := L.shared
  have hS := L.S_pos
  rcases (events_mem L x rf re pcs ccs v).1 ((hE v).1 hv) with
    ⟨k, hk', h | h⟩ | ⟨k, hk', h | h⟩ | ⟨q, hq, h⟩ | ⟨q, hq, h⟩ | ⟨q, hq, h⟩ <;>
    (try subst h) <;> (try (simp [rmwR, rmwW, Ev.key, sloc] at hk; done))
  · by_cases hqS : q < L.S
    · simp only [pStageEvs, hqS, ite_true, List.nil_append, List.mem_cons, List.not_mem_nil,
        or_false] at h
      rcases h with rfl | rfl <;> simp [rmwR, rmwW, Ev.key] at hk
    · simp only [pStageEvs, hqS, ite_false, List.mem_append, List.mem_cons, List.not_mem_nil,
        or_false] at h
      rcases h with ((h | rfl | rfl) | rfl | rfl)
      · obtain ⟨pos, bs, rfl⟩ := wait_mem h
        simp [waitEv, oneEv, Ev.key] at hk
      all_goals simp [waitEv, oneEv, fenceEvent, rmwR, rmwW, Ev.key] at hk
  · have hfq := (slot_eq (L := L) (q := q) (s := s))
    simp only [copyEvs, List.map_cons, List.map_nil, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl | rfl
    · simp [oneEv, Ev.key, gloc, sloc] at hk
    · exact ⟨q, hq, hfq.1 (by simp [oneEv, Ev.key, sloc] at hk; omega), .inl rfl⟩
    all_goals simp [fenceEvent, rmwR, rmwW, Ev.key] at hk
  · have hfq := (slot_eq (L := L) (q := q) (s := s))
    simp only [cStageEvs, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with h | rfl | rfl | rfl | rfl | rfl
    · obtain ⟨pos, bs, rfl⟩ := wait_mem h
      simp [waitEv, oneEv, Ev.key] at hk
    · simp [waitEv, oneEv, Ev.key] at hk
    · exact ⟨q, hq, hfq.1 (by simp [oneEv, Ev.key, sloc] at hk; omega), .inr rfl⟩
    · simp [oneEv, Ev.key, gloc, sloc] at hk
    all_goals simp [rmwR, rmwW, Ev.key] at hk

/-- **The events at a word of `O`**: the consumer's store of its stage. -/
theorem on_out {v : Ev} (hv : v ∈ c.evs) {q : Nat} (hq : q < L.N)
    (hk : v.loc = gloc (L.O + 4 * q) ∧ v.size = 4) : v = stE L ccs q := by
  have hsh := L.shared
  have hS := L.S_pos
  rcases (events_mem L x rf re pcs ccs v).1 ((hE v).1 hv) with
    ⟨k, hk', h | h⟩ | ⟨k, hk', h | h⟩ | ⟨q', hq', h⟩ | ⟨q', hq', h⟩ | ⟨q', hq', h⟩ <;>
    (try subst h) <;> (try (simp [rmwR, rmwW, gloc, sloc] at hk; done))
  · by_cases hqS : q' < L.S
    · simp only [pStageEvs, hqS, ite_true, List.nil_append, List.mem_cons, List.not_mem_nil,
        or_false] at h
      rcases h with rfl | rfl <;> simp [rmwR, rmwW, gloc, sloc] at hk
    · simp only [pStageEvs, hqS, ite_false, List.mem_append, List.mem_cons, List.not_mem_nil,
        or_false] at h
      rcases h with ((h | rfl | rfl) | rfl | rfl)
      · obtain ⟨pos, bs, rfl⟩ := wait_mem h
        simp [waitEv, oneEv, gloc, sloc] at hk
      all_goals simp [waitEv, oneEv, fenceEvent, rmwR, rmwW, gloc, sloc] at hk
  · simp only [copyEvs, List.map_cons, List.map_nil, List.mem_cons, List.not_mem_nil, or_false] at h
    have := L.global
    rcases h with rfl | rfl | rfl | rfl | rfl
    · simp [oneEv, gloc] at hk; omega
    all_goals simp [oneEv, fenceEvent, rmwR, rmwW, gloc, sloc] at hk
  · simp only [cStageEvs, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with h | rfl | rfl | rfl | rfl | rfl
    · obtain ⟨pos, bs, rfl⟩ := wait_mem h
      simp [waitEv, oneEv, gloc, sloc] at hk
    · simp [waitEv, oneEv, gloc, sloc] at hk
    · simp [oneEv, gloc, sloc] at hk
    · have : q' = q := by simp [oneEv, gloc] at hk; omega
      subst this; rfl
    all_goals simp [rmwR, rmwW, gloc, sloc] at hk

/-- **The writes to global memory**: the consumer's stores. -/
theorem global_write {v : Ev} (hv : v ∈ c.evs) (hw : v.e.isWrite = true) (hg : v.loc.space = .global) :
    ∃ q < L.N, v = stE L ccs q := by
  rcases (events_mem L x rf re pcs ccs v).1 ((hE v).1 hv) with
    ⟨k, hk', h | h⟩ | ⟨k, hk', h | h⟩ | ⟨q', hq', h⟩ | ⟨q', hq', h⟩ | ⟨q', hq', h⟩ <;>
    (try subst h) <;> (try (simp [rmwR, rmwW, sloc] at hg; done))
  · by_cases hqS : q' < L.S
    · simp only [pStageEvs, hqS, ite_true, List.nil_append, List.mem_cons, List.not_mem_nil,
        or_false] at h
      rcases h with rfl | rfl <;> simp [rmwR, rmwW, sloc] at hg
    · simp only [pStageEvs, hqS, ite_false, List.mem_append, List.mem_cons, List.not_mem_nil,
        or_false] at h
      rcases h with ((h | rfl | rfl) | rfl | rfl)
      · obtain ⟨pos, bs, rfl⟩ := wait_mem h
        simp [waitEv, oneEv, sloc] at hg
      all_goals simp [waitEv, oneEv, fenceEvent, rmwR, rmwW, sloc, Req.event, Event.isWrite] at hg hw
  · simp only [copyEvs, List.map_cons, List.map_nil, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl | rfl <;>
      simp [oneEv, fenceEvent, rmwR, rmwW, gloc, sloc, Req.event, Event.isWrite] at hg hw
  · simp only [cStageEvs, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with h | rfl | rfl | rfl | rfl | rfl
    · obtain ⟨pos, bs, rfl⟩ := wait_mem h
      simp [waitEv, oneEv, sloc] at hg
    · simp [waitEv, oneEv, sloc] at hg
    · simp [oneEv, sloc] at hg
    · exact ⟨q', hq', rfl⟩
    all_goals simp [rmwR, rmwW, sloc] at hg

/-- **The events at a word of `X`**: the copy's load of its stage. -/
theorem on_x {v : Ev} (hv : v ∈ c.evs) {q : Nat} (hq : q < L.N)
    (hk : v.loc = gloc (L.X + 4 * q) ∧ v.size = 4) : v = xRe L pcs q := by
  have hsh := L.shared
  have hS := L.S_pos
  rcases (events_mem L x rf re pcs ccs v).1 ((hE v).1 hv) with
    ⟨k, hk', h | h⟩ | ⟨k, hk', h | h⟩ | ⟨q', hq', h⟩ | ⟨q', hq', h⟩ | ⟨q', hq', h⟩ <;>
    (try subst h) <;> (try (simp [rmwR, rmwW, gloc, sloc] at hk; done))
  · by_cases hqS : q' < L.S
    · simp only [pStageEvs, hqS, ite_true, List.nil_append, List.mem_cons, List.not_mem_nil,
        or_false] at h
      rcases h with rfl | rfl <;> simp [rmwR, rmwW, gloc, sloc] at hk
    · simp only [pStageEvs, hqS, ite_false, List.mem_append, List.mem_cons, List.not_mem_nil,
        or_false] at h
      rcases h with ((h | rfl | rfl) | rfl | rfl)
      · obtain ⟨pos, bs, rfl⟩ := wait_mem h
        simp [waitEv, oneEv, gloc, sloc] at hk
      all_goals simp [waitEv, oneEv, fenceEvent, rmwR, rmwW, gloc, sloc] at hk
  · simp only [copyEvs, List.map_cons, List.map_nil, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl | rfl
    · have : q' = q := by simp [oneEv, gloc] at hk; omega
      subst this; rfl
    all_goals simp [oneEv, fenceEvent, rmwR, rmwW, gloc, sloc] at hk
  · simp only [cStageEvs, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with h | rfl | rfl | rfl | rfl | rfl
    · obtain ⟨pos, bs, rfl⟩ := wait_mem h
      simp [waitEv, oneEv, gloc, sloc] at hk
    · simp [waitEv, oneEv, gloc, sloc] at hk
    · simp [oneEv, gloc, sloc] at hk
    · have := L.global; simp [oneEv, gloc] at hk; omega
    all_goals simp [rmwR, rmwW, gloc, sloc] at hk

theorem mem_aR {q : Nat} (hq : q < L.N) : aRe L pcs q ∈ c.evs := mem_p c hE hq (aWe_mem L pcs).1
theorem mem_aW {q : Nat} (hq : q < L.N) : aWe L pcs q ∈ c.evs := mem_p c hE hq (aWe_mem L pcs).2
theorem mem_cR {q : Nat} (hq : q < L.N) : cRe L pcs q ∈ c.evs :=
  mem_cp c hE hq (copy_mem L pcs _ (by simp))
theorem mem_cW {q : Nat} (hq : q < L.N) : cWe L pcs q ∈ c.evs :=
  mem_cp c hE hq (copy_mem L pcs _ (by simp))
theorem mem_wY {q : Nat} (hq : q < L.N) : wYe L pcs q ∈ c.evs :=
  mem_cp c hE hq (copy_mem L pcs _ (by simp))
theorem mem_fC {q : Nat} (hq : q < L.N) : fCe L pcs q ∈ c.evs :=
  mem_cp c hE hq (copy_mem L pcs _ (by simp))
theorem mem_xR {q : Nat} (hq : q < L.N) : xRe L pcs q ∈ c.evs :=
  mem_cp c hE hq (copy_mem L pcs _ (by simp))
theorem mem_pe {q : Nat} (hq : q < L.N) (hqS : L.S ≤ q) : peE L pcs q ∈ c.evs :=
  mem_p c hE hq (peE_mem L pcs hqS).1
theorem mem_g {q : Nat} (hq : q < L.N) (hqS : L.S ≤ q) : gE L pcs q ∈ c.evs :=
  mem_p c hE hq (peE_mem L pcs hqS).2
theorem mem_rfE {q : Nat} (hq : q < L.N) : rfE L ccs q ∈ c.evs := mem_c c hE hq (cons_mem L ccs _ (by simp))
theorem mem_ld {q : Nat} (hq : q < L.N) : ldE L ccs q ∈ c.evs := mem_c c hE hq (cons_mem L ccs _ (by simp))
theorem mem_st {q : Nat} (hq : q < L.N) : stE L ccs q ∈ c.evs := mem_c c hE hq (cons_mem L ccs _ (by simp))
theorem mem_eR {q : Nat} (hq : q < L.N) : eRe L ccs q ∈ c.evs := mem_c c hE hq (cons_mem L ccs _ (by simp))
theorem mem_eW {q : Nat} (hq : q < L.N) : eWe L ccs q ∈ c.evs := mem_c c hE hq (cons_mem L ccs _ (by simp))

end exec

/-- The index of a member. -/
def ix {v : Ev} {evs : List Ev} (h : v ∈ evs) : Fin evs.length :=
  ⟨evs.idxOf v, List.idxOf_lt_length_of_mem h⟩

theorem ix_get {v : Ev} {evs : List Ev} (h : v ∈ evs) : evs[ix h] = v := List.getElem_idxOf _

theorem ix_inj {v w : Ev} {evs : List Ev} (hv : v ∈ evs) (hw : w ∈ evs) (h : ix hv = ix hw) : v = w := by
  rw [← ix_get hv, ← ix_get hw]; exact congrArg (evs[·]) h

/-! ## Rounds of a slot -/

/-- Round `r` of slot `s` is stage `r S + s`. -/
def stg (L : Layout) (s r : Nat) : Nat := r * L.S + s

theorem stg_mod {L : Layout} {s : Nat} (hs : s < L.S) (r : Nat) : stg L s r % L.S = s := by
  simp [stg, Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hs]

theorem stg_div {L : Layout} {s : Nat} (hs : s < L.S) (r : Nat) : stg L s r / L.S = r := by
  rw [stg, Nat.add_comm, Nat.add_mul_div_right _ _ L.S_pos, Nat.div_eq_of_lt hs, Nat.zero_add]

theorem stg_of (L : Layout) (q : Nat) : stg L (q % L.S) (q / L.S) = q := by
  rw [stg, Nat.mul_comm, Nat.add_comm]; exact Nat.mod_add_div q L.S

theorem stg_lt {L : Layout} {s r : Nat} (hs : s < L.S) :
    r < rounds L.S L.N s ↔ stg L s r < L.N := by
  rw [rounds_eq L.S_pos L.N hs, stg]
  have h₁ := Nat.div_add_mod L.N L.S
  have hm := Nat.mod_lt L.N L.S_pos
  generalize L.N / L.S = d at *
  generalize L.N % L.S = m at *
  rcases Nat.lt_trichotomy r d with h | rfl | h
  · have : (r + 1) * L.S ≤ d * L.S := Nat.mul_le_mul_right _ h
    rw [Nat.succ_mul, Nat.mul_comm d] at this
    by_cases hsm : s < m <;> simp only [hsm, ite_true, ite_false] <;> constructor <;> intro <;> omega
  · rw [Nat.mul_comm r]
    by_cases hsm : s < m <;> simp only [hsm, ite_true, ite_false] <;> constructor <;> intro <;> omega
  · have : (d + 1) * L.S ≤ r * L.S := Nat.mul_le_mul_right _ h
    rw [Nat.succ_mul, Nat.mul_comm d] at this
    by_cases hsm : s < m <;> simp only [hsm, ite_true, ite_false] <;> constructor <;> intro <;> omega

theorem stg_succ {L : Layout} (s r : Nat) : stg L s (r + 1) = stg L s r + L.S := by
  simp [stg, Nat.succ_mul]; omega

/-! ## Positions -/

theorem posP_succ (L : Layout) (cs : Nat → PStage) (q : Nat) :
    posP L cs (q + 1) = posP L cs q + pOff L q (cs q) + 2 := by
  simp [posP, sumBelow_succ]; omega

theorem posP_lt (L : Layout) (cs : Nat → PStage) {q q' : Nat} (h : q < q') :
    posP L cs q + pOff L q (cs q) + 2 ≤ posP L cs q' := by
  induction q' with
  | zero => omega
  | succ q' ih =>
    rw [posP_succ]
    rcases Nat.lt_succ_iff_lt_or_eq.1 h with h | rfl
    · have := ih h; omega
    · omega

theorem posP_ge (L : Layout) (cs : Nat → PStage) (q : Nat) : 4 * L.S ≤ posP L cs q := by
  simp [posP]

theorem posC_succ (cs : Nat → CStage) (q : Nat) : posC cs (q + 1) = posC cs q + (cs q).ws.length + 5 := by
  simp [posC, sumBelow_succ]; omega

theorem posC_lt (cs : Nat → CStage) {q q' : Nat} (h : q < q') :
    posC cs q + (cs q).ws.length + 5 ≤ posC cs q' := by
  induction q' with
  | zero => omega
  | succ q' ih =>
    rw [posC_succ]
    rcases Nat.lt_succ_iff_lt_or_eq.1 h with h | rfl
    · have := ih h; omega
    · omega

theorem stg_inj {L : Layout} {s r r' : Nat} (h : stg L s r = stg L s r') : r = r' := by
  simp only [stg] at h
  exact Nat.eq_of_mul_eq_mul_right L.S_pos (by omega)

theorem stg_lt_stg {L : Layout} {s r r' : Nat} (h : r < r') : stg L s r < stg L s r' := by
  simp only [stg]
  have := Nat.mul_lt_mul_of_pos_right h L.S_pos
  omega

/-! ## The barriers in the run's execution -/

section barriers
variable {L : Layout} {x : Nat → Fin 4 → UInt8} (c : Candidate (program L x))
  {rf re : Nat → Nat → List UInt8} {pcs : Nat → PStage} {ccs : Nat → CStage}
  (hE : ∀ v, v ∈ c.evs ↔
    v ∈ (program L x).events [(pActs L rf re pcs, .exit), (cActs L ccs, .exit)])
  (hnd : c.evs.Nodup)

/-- The index of event `v`, or `d` if it is not one. -/
noncomputable def at_ (d : Fin c.evs.length) (v : Ev) : Fin c.evs.length := by
  classical exact if h : v ∈ c.evs then ix h else d

theorem at_get {d : Fin c.evs.length} {v : Ev} (h : v ∈ c.evs) : c.evs[at_ c d v] = v := by
  have e : at_ c d v = ix h := by unfold at_; simp [h]
  simp only [e]; exact ix_get h

include hnd in
/-- Distinct events have distinct indices, and an event has one. -/
theorem at_eq {d : Fin c.evs.length} {v : Ev} {j : Fin c.evs.length} (h : c.evs[j] = v) :
    j = at_ c d v := by
  have hm : v ∈ c.evs := h ▸ List.getElem_mem _
  have h₂ := at_get c (d := d) hm
  exact Fin.ext (hnd.eq_of_getElem_eq j.isLt (at_ c d v).isLt (h.trans h₂.symm))

theorem at_ne {d : Fin c.evs.length} {v w : Ev} (hv : v ∈ c.evs) (hw : w ∈ c.evs) (hne : v ≠ w) :
    at_ c d v ≠ at_ c d w := fun e => hne (by
  have := congrArg (fun i : Fin c.evs.length => c.evs[i]) e
  rwa [at_get c hv, at_get c hw] at this)

theorem key_of {i j : Fin c.evs.length} (h : c.exec.sameLocation i j = true) :
    c.evs[i].key = c.evs[j].key :=
  (Candidate.addr_iff c).1 (sameLocation_iff.1 h).2.2

theorem full_at {q s : Nat} (hq : q % L.S = s) : L.full q = L.F + 8 * s := by
  unfold Layout.full; rw [hq]

theorem gen_mbar {r : Req} {t k : Nat} {l : Loc} {bs : List UInt8} (h : r.proxy = .generic) :
    ((rmwR r t k l bs).e.isMem = true ∧ (rmwR r t k l bs).e.proxy = .generic) ∧
      ((rmwW r t k l bs).e.isMem = true ∧ (rmwW r t k l bs).e.proxy = .generic) := by
  simp [rmwR, rmwW, Req.event, Event.isMem, Event.isRead, Event.isWrite, h]

/-- A read-modify-write's two events, generic, in order on one thread. -/
theorem rmw_pair {A B : Ev} {i j : Fin c.evs.length} (hi : c.evs[i] = A) (hj : c.evs[j] = B)
    (hr : A.rmwRead = true) (ht : A.e.thread = B.e.thread) (hsq : B.seq = A.seq + 1)
    (hA : A.e.isMem = true ∧ A.e.proxy = .generic) (hB : B.e.isMem = true ∧ B.e.proxy = .generic)
    (hk : A.key = B.key) :
    c.exec.rmw i j = true ∧ c.exec.proxyPreservedCauseBase i j = true ∧
      c.exec.sameLocation i j = true :=
  ⟨Candidate.rmw_of c hi hj hr ht hsq,
    Candidate.ppcb_gen c hi hj hA hB hk
      (causeBase_of_order (Candidate.order_of_po c (by rw [hi, hj]; exact ht) (by rw [hi, hj]; omega))),
    Candidate.sameLoc_of c hi hj hA.1 hB.1 hk⟩

/-- An 8-byte read-modify-write's read, defined, reads its whole word. -/
theorem fb_val {r : Req} {op : MBarOp} (hop : r.op = .mbar op) {u : UInt64} (ha : r.addrs = #[some u])
    {read : Nat → List UInt8} (hl : (read 0).length = 8) (hu : r.unsupported read = false)
    {t k : Nat} {l : Loc} :
    MBarrierWord.apply op (Pipeline.mword (rmwR r t k l (read 0)).val) =
      .ok (Pipeline.mword (rmwW r t k l (read 0)).val) := by
  obtain ⟨w, hw⟩ := Pipeline.mbar_ok hop ha hu
  simp only [rmwR, rmwW]
  rw [take_all hl, Pipeline.written_mbar hop hw, Option.getD_some, Pipeline.mword_bytes64]
  exact hw

/-- A wait's response spells the word it read. -/
theorem word64_wait {a : Nat} {rw : Nat → List UInt8} (hl : (rw 0).length = 8) :
    Pipeline.word64 ((waitReq a).resp rw) = Pipeline.mword (rw 0) := by
  rw [← Pipeline.word64_resp hl]
  simp [Req.resp, waitReq, Req.laneResp, MemOp.returns, words]

end barriers

/-! ## A slot's full barrier -/

section fullB
variable {L : Layout} {x : Nat → Fin 4 → UInt8} (c : Candidate (program L x))
  {rf re : Nat → Nat → List UInt8} {pcs : Nat → PStage} {ccs : Nat → CStage}
  (hE : ∀ v, v ∈ c.evs ↔
    v ∈ (program L x).events [(pActs L rf re pcs, .exit), (cActs L ccs, .exit)])
  (hnd : c.evs.Nodup) (hc : c.exec.consistent = true) (d : Fin c.evs.length)
  {s : Nat} (hs : s < L.S)

include hE hs

theorem full_gen {v : Ev} (hv : v ∈ c.evs) (hk : v.key = (sloc (L.F + 8 * s), 8)) :
    v.e.isMem = true ∧ v.e.proxy = .generic := by
  rcases on_full c hE hv hs hk with rfl | rfl | ⟨q, -, -, rfl | rfl | rfl | rfl | ⟨pos, bs, rfl⟩⟩ <;>
    simp [iFe, rmwR, rmwW, aRe, aWe, cRe, cWe, aft, waitEv, oneEv, Req.event, Event.isMem,
      Event.isRead, Event.isWrite, mbarReq, waitReq, Layout.initF, Layout.arriveF, Layout.completeF]

theorem full_writes {v : Ev} (hv : v ∈ c.evs) (hk : v.key = (sloc (L.F + 8 * s), 8))
    (hw : v.e.isWrite = true) :
    v = iFe L rf s ∨ ∃ q < L.N, q % L.S = s ∧ (v = aWe L pcs q ∨ v = cWe L pcs q) := by
  rcases on_full c hE hv hs hk with h | rfl | ⟨q, hq, hqs, h | h | h | h | ⟨pos, bs, rfl⟩⟩
  · exact .inl h
  · simp [rmwR, Req.event, Event.isWrite] at hw
  · subst h; simp [aRe, rmwR, Req.event, Event.isWrite] at hw
  · exact .inr ⟨q, hq, hqs, .inl h⟩
  · subst h; simp [cRe, aft, rmwR, Req.event, Event.isWrite] at hw
  · exact .inr ⟨q, hq, hqs, .inr h⟩
  · simp [waitEv, oneEv, Req.event, Event.isWrite] at hw

theorem iF_mem : iFe L rf s ∈ c.evs := mem_iF c hE hs

omit hE hs in
theorem key_iF : (iFe L rf s).key = (sloc (L.F + 8 * s), 8) := rfl

theorem fb_key {v : Fin c.evs.length} (h : c.exec.sameLocation v (at_ c d (iFe L rf s)) = true) :
    c.evs[v].key = (sloc (L.F + 8 * s), 8) := by
  have h₁ := key_of c h
  rw [at_get c (d := d) (iF_mem c hE hs)] at h₁
  exact h₁

theorem fb_generic {v : Fin c.evs.length} (h : c.exec.sameLocation v (at_ c d (iFe L rf s)) = true) :
    c.exec.generic v = true := by
  have hk := fb_key c hE d hs h
  simp only [Fin.getElem_fin] at hk
  obtain ⟨h₁, h₂⟩ := full_gen c hE hs (List.getElem_mem _) hk
  show (c.evs[v.1].e.isMem && c.evs[v.1].e.proxy == .generic) = true
  rw [h₁, h₂]; rfl

include hnd hc in
theorem fb_loc : OneLoc c.exec (at_ c d (iFe L rf s)) where
  strong a b ha hb hsa hsb := Candidate.strongR_of c (pl := ⟨0, 0, 0⟩) place_eq hsa hsb (by
    have ka := fb_key c hE d hs ha
    have kb := fb_key c hE d hs hb
    simp only [Fin.getElem_fin] at ka kb ⊢
    rw [(full_gen c hE hs (List.getElem_mem _) ka).2, (full_gen c hE hs (List.getElem_mem _) kb).2])
  addr a ha := (sameLocation_iff.1 ha).2.2
  write := by
    show c.evs[at_ c d (iFe L rf s)].e.isWrite = true
    rw [at_get c (iF_mem c hE hs)]; rfl
  nonweak w hw hwl hne := by
    have hk := fb_key c hE d hs hwl
    simp only [Fin.getElem_fin] at hk
    rcases full_writes c hE hs (List.getElem_mem _) hk hw with h | ⟨q, -, -, h | h⟩
    · exact absurd (at_eq c hnd h) hne
    · show c.evs[w.1].e.scope ≠ .thread
      rw [h]; simp [aWe, rmwW, Req.event, Req.eventScope, Layout.arriveF, mbarReq]
    · show c.evs[w.1].e.scope ≠ .thread
      rw [h]; simp [cWe, aft, rmwW, Req.event, Req.eventScope, Layout.completeF, mbarReq]
  first v hv hvl hne := by
    have hk := fb_key c hE d hs hvl
    simp only [Fin.getElem_fin] at hk
    have hm := iF_mem c hE hs
    have hI := at_get c (d := d) hm
    have hIw : c.exec.writes (at_ c d (iFe L rf s)) = true := by
      show c.evs[at_ c d (iFe L rf s)].e.isWrite = true
      rw [hI]; rfl
    have hIg := (gen_mbar (r := L.initF s) (t := 0) (k := 2 * s) (l := sloc (L.F + 8 * s))
      (bs := rf s 0) rfl).2
    rcases full_writes c hE hs (List.getElem_mem _) hk hv with h | ⟨q, hq, hqs, h | h⟩
    · exact absurd (at_eq c hnd h) hne
    · refine co_of_ppcb hc hIw hv (sameLocation_symm hvl) (Candidate.ppcb_gen c hI h hIg
        (gen_mbar rfl).2 (by simp [iFe, aWe, rmwW, Ev.key, full_at hqs]) (causeBase_of_order ?_))
      have h' : c.evs[v] = aWe L pcs q := h
      refine Candidate.order_of_po c (by rw [hI, h']; simp [iFe, aWe, rmwW, Req.event]) ?_
      rw [hI, h']; simp only [iFe, aWe, rmwW]; have := posP_ge L pcs q; omega
    · refine co_of_ppcb hc hIw hv (sameLocation_symm hvl) (Candidate.ppcb_gen c hI h hIg
        (by simp [cWe, aft, rmwW, Req.event, Event.isMem, Event.isWrite, Layout.completeF, mbarReq])
        (by simp [iFe, cWe, aft, rmwW, Ev.key, full_at hqs]) (causeBase_of_order ?_))
      have h' : c.evs[v] = cWe L pcs q := h
      refine Candidate.order_of_issue c (p := (0, cpK L pcs q)) (by rw [h']; simp [cWe, aft])
        (by rw [hI]; simp [iFe, rmwW, Req.event]) ?_
      rw [hI]; simp only [iFe, rmwW, cpK]; have := posP_ge L pcs q; omega

theorem fb_iv : Pipeline.mword (c.exec.val (at_ c d (iFe L rf s))) = Pipeline.fresh 0 := by
  show Pipeline.mword c.evs[at_ c d (iFe L rf s)].val = _
  rw [at_get c (iF_mem c hE hs)]
  have h : MBarrierWord.apply (.init 1) (Pipeline.mword (rf s 0)) = .ok (Pipeline.fresh 0) := by
    simp [MBarrierWord.apply, NV.PTX.MBarrier.init, NV.PTX.MBarrier.countMax, Pipeline.fresh]
  simp only [iFe, rmwW]
  rw [Pipeline.written_mbar (r := L.initF s) rfl h, Option.getD_some, Pipeline.mword_bytes64]

omit hE hs in
theorem fb_zero : initBytes (program L x).mem (iFe L rf s).key = List.replicate 8 0 := by
  simp [initBytes, program, iFe, rmwW, Ev.key, sloc, List.range_succ]

theorem fb_mem0 (w : UInt64) : MBarrierWord.apply (.arrive 1 4 false)
    (Pipeline.mword (c.exec.init (c.exec.loc (c.exec.ev (at_ c d (iFe L rf s))).addr))) ≠ .ok w := by
  rw [Candidate.init_at c, at_get c (iF_mem c hE hs), fb_zero]
  intro h
  have h0 : (MBarrierWord.apply (.arrive 1 4 false) (Pipeline.mword (List.replicate 8 0))).toBool =
    false := by decide +kernel
  rw [h] at h0; cases h0

theorem fb_aW {r : Nat} (hr : r < rounds L.S L.N s) :
    c.exec.writes (at_ c d (aWe L pcs (stg L s r))) = true ∧
      c.exec.sameLocation (at_ c d (aWe L pcs (stg L s r))) (at_ c d (iFe L rf s)) = true := by
  have hm := mem_aW c hE ((stg_lt hs).1 hr)
  have hg := at_get c (d := d) hm
  refine ⟨by show c.evs[at_ c d (aWe L pcs (stg L s r))].e.isWrite = true; rw [hg]; simp [aWe, rmwW, Req.event, Event.isWrite],
    ?_⟩
  exact Candidate.sameLoc_of c hg (at_get c (iF_mem c hE hs)) (gen_mbar rfl).2.1 (gen_mbar rfl).2.1
    (by simp [aWe, iFe, rmwW, Ev.key, full_at (stg_mod hs r)])

theorem fb_cW {r : Nat} (hr : r < rounds L.S L.N s) :
    c.exec.writes (at_ c d (cWe L pcs (stg L s r))) = true ∧
      c.exec.sameLocation (at_ c d (cWe L pcs (stg L s r))) (at_ c d (iFe L rf s)) = true := by
  have hm := mem_cW c hE ((stg_lt hs).1 hr)
  have hg := at_get c (d := d) hm
  refine ⟨by show c.evs[at_ c d (cWe L pcs (stg L s r))].e.isWrite = true; rw [hg]; simp [cWe, aft, rmwW, Req.event, Event.isWrite],
    ?_⟩
  exact Candidate.sameLoc_of c hg (at_get c (iF_mem c hE hs))
    (by simp [cWe, aft, rmwW, Req.event, Event.isMem, Event.isWrite]) (gen_mbar rfl).2.1
    (by simp [cWe, aft, iFe, rmwW, Ev.key, full_at (stg_mod hs r)])

theorem fb_ine {r : Nat} (hr : r < rounds L.S L.N s) :
    at_ c d (aWe L pcs (stg L s r)) ≠ at_ c d (iFe L rf s) ∧
      at_ c d (cWe L pcs (stg L s r)) ≠ at_ c d (iFe L rf s) := by
  have hq := (stg_lt hs).1 hr
  refine ⟨at_ne c (mem_aW c hE hq) (iF_mem c hE hs) fun h => ?_,
    at_ne c (mem_cW c hE hq) (iF_mem c hE hs) fun h => ?_⟩
  · have := congrArg Ev.seq h; simp only [aWe, iFe, rmwW] at this
    have := posP_ge L pcs (stg L s r); omega
  · have := congrArg (·.e.thread) h; simp [cWe, aft, iFe, rmwW, Req.event, cpT] at this

theorem fb_acne {r r' : Nat} (hr : r < rounds L.S L.N s) (hr' : r' < rounds L.S L.N s) :
    at_ c d (aWe L pcs (stg L s r)) ≠ at_ c d (cWe L pcs (stg L s r')) :=
  at_ne c (mem_aW c hE ((stg_lt hs).1 hr)) (mem_cW c hE ((stg_lt hs).1 hr')) fun h => by
    have := congrArg (·.e.thread) h; simp [cWe, aft, aWe, rmwW, Req.event, cpT] at this

include hnd in
theorem fb_all (v : Fin c.evs.length) (hv : c.exec.writes v = true)
    (hvl : c.exec.sameLocation v (at_ c d (iFe L rf s)) = true) :
    v = at_ c d (iFe L rf s) ∨ (∃ r < rounds L.S L.N s, v = at_ c d (aWe L pcs (stg L s r))) ∨
      ∃ r < rounds L.S L.N s, ∃ j < 1, v = at_ c d (cWe L pcs (stg L s r)) := by
  have hk := fb_key c hE d hs hvl
  simp only [Fin.getElem_fin] at hk
  rcases full_writes c hE hs (List.getElem_mem _) hk hv with h | ⟨q, hq, hqs, h | h⟩
  · exact .inl (at_eq c hnd h)
  · have he : stg L s (q / L.S) = q := hqs ▸ stg_of L q
    exact .inr (.inl ⟨q / L.S, (stg_lt hs).2 (by rw [he]; exact hq), at_eq c hnd (by rw [he]; exact h)⟩)
  · have he : stg L s (q / L.S) = q := hqs ▸ stg_of L q
    exact .inr (.inr ⟨q / L.S, (stg_lt hs).2 (by rw [he]; exact hq), 0, Nat.one_pos,
      at_eq c hnd (by rw [he]; exact h)⟩)

theorem fb_armw {r : Nat} (hr : r < rounds L.S L.N s) :
    c.exec.rmw (at_ c d (aRe L pcs (stg L s r))) (at_ c d (aWe L pcs (stg L s r))) = true ∧
      c.exec.proxyPreservedCauseBase (at_ c d (aRe L pcs (stg L s r)))
        (at_ c d (aWe L pcs (stg L s r))) = true ∧
      c.exec.sameLocation (at_ c d (aRe L pcs (stg L s r))) (at_ c d (iFe L rf s)) = true := by
  have hq := (stg_lt hs).1 hr
  have hR := at_get c (d := d) (mem_aR c hE hq)
  have hW := at_get c (d := d) (mem_aW c hE hq)
  obtain ⟨h₁, h₂, h₃⟩ := rmw_pair c hR hW rfl rfl rfl (gen_mbar rfl).1 (gen_mbar rfl).2 rfl
  exact ⟨h₁, h₂, sameLocation_trans h₃ (fb_aW c hE d hs hr).2⟩

theorem fb_crmw {r : Nat} (hr : r < rounds L.S L.N s) :
    c.exec.rmw (at_ c d (cRe L pcs (stg L s r))) (at_ c d (cWe L pcs (stg L s r))) = true ∧
      c.exec.proxyPreservedCauseBase (at_ c d (cRe L pcs (stg L s r)))
        (at_ c d (cWe L pcs (stg L s r))) = true ∧
      c.exec.sameLocation (at_ c d (cRe L pcs (stg L s r))) (at_ c d (iFe L rf s)) = true := by
  have hq := (stg_lt hs).1 hr
  have hR := at_get c (d := d) (mem_cR c hE hq)
  have hW := at_get c (d := d) (mem_cW c hE hq)
  obtain ⟨h₁, h₂, h₃⟩ := rmw_pair c hR hW rfl rfl rfl
    (by simp [cRe, aft, rmwR, Req.event, Event.isMem, Event.isRead, Layout.completeF, mbarReq])
    (by simp [cWe, aft, rmwW, Req.event, Event.isMem, Event.isWrite, Layout.completeF, mbarReq]) rfl
  exact ⟨h₁, h₂, sameLocation_trans h₃ (fb_cW c hE d hs hr).2⟩

include hc in
theorem fb_next {r : Nat} (hr : r + 1 < rounds L.S L.N s) :
    c.exec.co (at_ c d (aWe L pcs (stg L s r))) (at_ c d (aWe L pcs (stg L s (r + 1)))) = true := by
  have hq := (stg_lt hs).1 hr
  have h₁ := at_get c (d := d) (mem_aW c hE ((stg_lt hs).1 (Nat.lt_of_succ_lt hr)))
  have h₂ := at_get c (d := d) (mem_aW c hE hq)
  refine co_of_ppcb hc (fb_aW c hE d hs (Nat.lt_of_succ_lt hr)).1 (fb_aW c hE d hs hr).1
    (sameLocation_trans (fb_aW c hE d hs (Nat.lt_of_succ_lt hr)).2
      (sameLocation_symm (fb_aW c hE d hs hr).2))
    (Candidate.ppcb_gen c h₁ h₂ (gen_mbar rfl).2 (gen_mbar rfl).2
      (by simp [aWe, rmwW, Ev.key, full_at (stg_mod hs r), full_at (stg_mod hs (r + 1))])
      (causeBase_of_order (Candidate.order_of_po c (by rw [h₁, h₂]; rfl) ?_)))
  rw [h₁, h₂]; simp only [aWe, rmwW]
  have := posP_lt L pcs (stg_lt_stg (s := s) (L := L) (show r < r + 1 by omega))
  omega

include hc in
theorem fb_ac {r : Nat} (hr : r < rounds L.S L.N s) :
    c.exec.co (at_ c d (aWe L pcs (stg L s r))) (at_ c d (cWe L pcs (stg L s r))) = true := by
  have hq := (stg_lt hs).1 hr
  have h₁ := at_get c (d := d) (mem_aW c hE hq)
  have h₂ := at_get c (d := d) (mem_cW c hE hq)
  refine co_of_ppcb hc (fb_aW c hE d hs hr).1 (fb_cW c hE d hs hr).1
    (sameLocation_trans (fb_aW c hE d hs hr).2 (sameLocation_symm (fb_cW c hE d hs hr).2))
    (Candidate.ppcb_gen c h₁ h₂ (gen_mbar rfl).2
      (by simp [cWe, aft, rmwW, Req.event, Event.isMem, Event.isWrite, Layout.completeF, mbarReq])
      (by simp [aWe, cWe, aft, rmwW, Ev.key])
      (causeBase_of_order (Candidate.order_of_issue c (p := (0, cpK L pcs (stg L s r)))
        (by rw [h₂]; simp [cWe, aft]) (by rw [h₁]; simp [aWe, rmwW, Req.event])
        (by rw [h₁]; simp only [aWe, rmwW, cpK]; omega))))

theorem fb_cinj {r r' : Nat} (hr : r < rounds L.S L.N s) (hr' : r' < rounds L.S L.N s)
    (h : at_ c d (cWe L pcs (stg L s r)) = at_ c d (cWe L pcs (stg L s r'))) : r = r' := by
  have := congrArg (fun i : Fin c.evs.length => c.evs[i]) h
  rw [at_get c (mem_cW c hE ((stg_lt hs).1 hr)), at_get c (mem_cW c hE ((stg_lt hs).1 hr'))] at this
  have := congrArg (·.e.thread) this
  simp [cWe, aft, rmwW, Req.event, cpT] at this
  exact stg_inj (L := L) (s := s) (by omega)

theorem fb_av (hp : ∀ q < L.N, (pcs q).Ok L q) {r : Nat} (hr : r < rounds L.S L.N s) :
    MBarrierWord.apply (.arrive 1 4 false) (Pipeline.mword (c.exec.val (at_ c d (aRe L pcs (stg L s r))))) =
      .ok (Pipeline.mword (c.exec.val (at_ c d (aWe L pcs (stg L s r))))) := by
  have hq := (stg_lt hs).1 hr
  obtain ⟨-, hra, hu, -⟩ := hp _ hq
  show MBarrierWord.apply _ (Pipeline.mword c.evs[at_ c d (aRe L pcs (stg L s r))].val) =
    .ok (Pipeline.mword c.evs[at_ c d (aWe L pcs (stg L s r))].val)
  rw [at_get c (mem_aR c hE hq), at_get c (mem_aW c hE hq)]
  exact fb_val rfl rfl (hra 0 (by simp [Req.active, Layout.arriveF, mbarReq])) hu

theorem fb_cv (hp : ∀ q < L.N, (pcs q).Ok L q) {r : Nat} (hr : r < rounds L.S L.N s) :
    MBarrierWord.apply (.completeTx 4) (Pipeline.mword (c.exec.val (at_ c d (cRe L pcs (stg L s r))))) =
      .ok (Pipeline.mword (c.exec.val (at_ c d (cWe L pcs (stg L s r))))) := by
  have hq := (stg_lt hs).1 hr
  obtain ⟨-, -, -, hok⟩ := hp _ hq
  obtain ⟨hl, hu⟩ := hok (L.completeF (stg L s r)) ((pcs (stg L s r)).gr 0)
    (by rw [copyActs_eq]; simp)
  show MBarrierWord.apply _ (Pipeline.mword c.evs[at_ c d (cRe L pcs (stg L s r))].val) =
    .ok (Pipeline.mword c.evs[at_ c d (cWe L pcs (stg L s r))].val)
  rw [at_get c (mem_cR c hE hq), at_get c (mem_cW c hE hq)]
  simp only [cRe, cWe, aft]
  exact fb_val rfl rfl (hl 0 (by simp [Req.active, Layout.completeF, mbarReq])) hu

include hnd hc in
/-- **A slot's full barrier**, round `r` being stage `stg s r`. -/
noncomputable def fullB (hp : ∀ q < L.N, (pcs q).Ok L q) : Pipeline.Full c.exec Pipeline.mword where
  R := rounds L.S L.N s
  K := 1
  T := 4
  amt := fun _ => 4
  K_pos := Nat.one_pos
  amt_pos := fun _ _ => by decide
  amt_sum := rfl
  T_lt := by decide
  i := at_ c d (iFe L rf s)
  aR r := at_ c d (aRe L pcs (stg L s r))
  aW r := at_ c d (aWe L pcs (stg L s r))
  cR r _ := at_ c d (cRe L pcs (stg L s r))
  cW r _ := at_ c d (cWe L pcs (stg L s r))
  loc := fb_loc c hE hnd hc d hs
  i_v := fb_iv c hE d hs
  mem0 := fb_mem0 c hE d hs
  aW_w _ hr := fb_aW c hE d hs hr
  cW_w _ hr _ _ := fb_cW c hE d hs hr
  i_ne _ hr := ⟨(fb_ine c hE d hs hr).1, fun _ _ => (fb_ine c hE d hs hr).2⟩
  ac_ne _ hr _ hr' _ _ := fb_acne c hE d hs hr hr'
  all := fb_all c hE hnd d hs
  a_rmw _ hr := fb_armw c hE d hs hr
  c_rmw _ hr _ _ := fb_crmw c hE d hs hr
  a_next _ hr := fb_next c hE hc d hs hr
  a_c _ hr _ _ := fb_ac c hE hc d hs hr
  c_inj _ hr _ hr' _ hj _ hj' h := ⟨fb_cinj c hE d hs hr hr' h, by omega⟩
  a_v _ hr := fb_av c hE d hs hp hr
  c_v _ hr _ _ := fb_cv c hE d hs hp hr

theorem fw_read {r : Nat} (hr : r < rounds L.S L.N s) :
    c.exec.reads (at_ c d (rfE L ccs (stg L s r))) = true ∧
      c.exec.sameLocation (at_ c d (rfE L ccs (stg L s r))) (at_ c d (iFe L rf s)) = true ∧
      (c.exec.ev (at_ c d (rfE L ccs (stg L s r)))).acquire = true ∧
      (c.exec.ev (at_ c d (rfE L ccs (stg L s r)))).reduction = false ∧
      (c.exec.ev (at_ c d (rfE L ccs (stg L s r)))).scope ≠ .thread := by
  have hq := (stg_lt hs).1 hr
  have hg := at_get c (d := d) (mem_rfE c hE hq)
  have hI := at_get c (d := d) (iF_mem c hE hs)
  refine ⟨?_, Candidate.sameLoc_of c hg hI (by simp [rfE, waitEv, oneEv, Req.event, Event.isMem,
    Event.isRead]) (gen_mbar rfl).2.1 (by simp [rfE, waitEv, oneEv, iFe, rmwW, Ev.key,
    full_at (stg_mod hs r)]), ?_, ?_, ?_⟩
  · show c.evs[at_ c d (rfE L ccs (stg L s r))].e.isRead = true
    rw [hg]; simp [rfE, waitEv, oneEv, Req.event, Event.isRead]
  · show c.evs[at_ c d (rfE L ccs (stg L s r))].e.acquire = true
    rw [hg]; simp [rfE, waitEv, oneEv, Req.event, Req.acquires, waitReq]
  · show c.evs[at_ c d (rfE L ccs (stg L s r))].e.reduction = false
    rw [hg]; simp [rfE, waitEv, oneEv, Req.event, Req.isRed, waitReq]
  · show c.evs[at_ c d (rfE L ccs (stg L s r))].e.scope ≠ .thread
    rw [hg]; simp [rfE, waitEv, oneEv, Req.event, Req.eventScope, waitReq]

theorem fw_done (hcs : ∀ q < L.N, (ccs q).Ok L q) {r : Nat} (hr : r < rounds L.S L.N s) :
    MBarrierWord.parityDone (Pipeline.mword (c.exec.val (at_ c d (rfE L ccs (stg L s r))))) r = true := by
  have hq := (stg_lt hs).1 hr
  obtain ⟨-, hrw, hd, -⟩ := hcs _ hq
  have hl : ((ccs (stg L s r)).rw 0).length = 8 :=
    (hrw 0 (by simp [Req.active, waitReq])).trans (by rw [Req.readLen, pieces_wait]; rfl)
  show MBarrierWord.parityDone (Pipeline.mword c.evs[at_ c d (rfE L ccs (stg L s r))].val) r = true
  rw [at_get c (mem_rfE c hE hq)]
  simp only [rfE, waitEv, oneEv, Bool.false_eq_true, ite_false]
  rw [take_all hl, ← word64_wait hl]
  rw [stg_div hs r] at hd
  exact hd

theorem fw_mem0 : MBarrierWord.parityDone
    (Pipeline.mword (c.exec.init (c.exec.loc (c.exec.ev (at_ c d (iFe L rf s))).addr))) 0 = false := by
  rw [Candidate.init_at c, at_get c (iF_mem c hE hs), fb_zero]
  decide +kernel

theorem fw_next {r : Nat} (hr : r + 1 < rounds L.S L.N s) :
    c.exec.causeBase (at_ c d (rfE L ccs (stg L s r))) (at_ c d (rfE L ccs (stg L s (r + 1)))) = true := by
  have h₁ := at_get c (d := d) (mem_rfE c hE ((stg_lt hs).1 (Nat.lt_of_succ_lt hr)))
  have h₂ := at_get c (d := d) (mem_rfE c hE ((stg_lt hs).1 hr))
  refine causeBase_of_order (Candidate.order_of_po c (by rw [h₁, h₂]; rfl) ?_)
  rw [h₁, h₂]; simp only [rfE, waitEv, oneEv]
  have := posC_lt ccs (stg_lt_stg (s := s) (L := L) (show r < r + 1 by omega))
  omega

theorem fw_rel {r : Nat} (hr : r < rounds L.S L.N s) :
    (c.exec.ev (at_ c d (cWe L pcs (stg L s r)))).release = true := by
  show c.evs[at_ c d (cWe L pcs (stg L s r))].e.release = true
  rw [at_get c (mem_cW c hE ((stg_lt hs).1 hr))]
  simp [cWe, aft, rmwW, Req.event, Req.releases, Layout.completeF, mbarReq]

include hnd hc in
/-- **The consumer's waits on a slot's full barrier.** -/
noncomputable def fullW (hp : ∀ q < L.N, (pcs q).Ok L q) (hcs : ∀ q < L.N, (ccs q).Ok L q) :
    Pipeline.Full.Waits (fullB c hE hnd hc d hs hp) where
  Rf r := at_ c d (rfE L ccs (stg L s r))
  read _ hr := fw_read c hE d hs hr
  done _ hr := fw_done c hE d hs hcs hr
  mem0 := fw_mem0 c hE d hs
  next _ hr := fw_next c hE d hs hr
  release _ hr _ _ := fw_rel c hE d hs hr
  generic _ ha := fb_generic c hE d hs ha

include hnd hc in
/-- **The full barrier's order**: the consumer's wait for round `r` of a slot synchronizes with the
copy's complete of that round. -/
theorem fb_raw (hp : ∀ q < L.N, (pcs q).Ok L q) (hcs : ∀ q < L.N, (ccs q).Ok L q) {r : Nat}
    (hr : r < rounds L.S L.N s) :
    c.exec.sync (at_ c d (cWe L pcs (stg L s r))) (at_ c d (rfE L ccs (stg L s r))) = true :=
  Pipeline.Full.raw _ hc (fullW c hE hnd hc d hs hp hcs) r hr 0 Nat.one_pos

end fullB

/-! ## A slot's empty barrier -/

section emptyB
variable {L : Layout} {x : Nat → Fin 4 → UInt8} (c : Candidate (program L x))
  {rf re : Nat → Nat → List UInt8} {pcs : Nat → PStage} {ccs : Nat → CStage}
  (hE : ∀ v, v ∈ c.evs ↔
    v ∈ (program L x).events [(pActs L rf re pcs, .exit), (cActs L ccs, .exit)])
  (hnd : c.evs.Nodup) (hc : c.exec.consistent = true) (d : Fin c.evs.length)
  {s : Nat} (hs : s < L.S)

theorem empty_at {q s : Nat} (hq : q % L.S = s) : L.empty q = L.E + 8 * s := by
  unfold Layout.empty; rw [hq]

include hE hs

theorem iE_mem : iEe L re s ∈ c.evs := mem_iE c hE hs

theorem empty_gen {v : Ev} (hv : v ∈ c.evs) (hk : v.key = (sloc (L.E + 8 * s), 8)) :
    v.e.isMem = true ∧ v.e.proxy = .generic := by
  rcases on_empty c hE hv hs hk with rfl | rfl | ⟨q, -, -, rfl | rfl | ⟨pos, bs, rfl⟩⟩ <;>
    simp [iEe, rmwR, rmwW, eRe, eWe, waitEv, oneEv, Req.event, Event.isMem, Event.isRead,
      Event.isWrite, mbarReq, waitReq, Layout.initE, Layout.arriveE]

theorem empty_writes {v : Ev} (hv : v ∈ c.evs) (hk : v.key = (sloc (L.E + 8 * s), 8))
    (hw : v.e.isWrite = true) : v = iEe L re s ∨ ∃ q < L.N, q % L.S = s ∧ v = eWe L ccs q := by
  rcases on_empty c hE hv hs hk with h | rfl | ⟨q, hq, hqs, h | h | ⟨pos, bs, rfl⟩⟩
  · exact .inl h
  · simp [rmwR, Req.event, Event.isWrite] at hw
  · subst h; simp [eRe, rmwR, Req.event, Event.isWrite] at hw
  · exact .inr ⟨q, hq, hqs, h⟩
  · simp [waitEv, oneEv, Req.event, Event.isWrite] at hw

theorem eb_key {v : Fin c.evs.length} (h : c.exec.sameLocation v (at_ c d (iEe L re s)) = true) :
    c.evs[v].key = (sloc (L.E + 8 * s), 8) := by
  have h₁ := key_of c h
  rw [at_get c (d := d) (iE_mem c hE hs)] at h₁
  exact h₁

theorem eb_generic {v : Fin c.evs.length} (h : c.exec.sameLocation v (at_ c d (iEe L re s)) = true) :
    c.exec.generic v = true := by
  have hk := eb_key c hE d hs h
  simp only [Fin.getElem_fin] at hk
  obtain ⟨h₁, h₂⟩ := empty_gen c hE hs (List.getElem_mem _) hk
  show (c.evs[v.1].e.isMem && c.evs[v.1].e.proxy == .generic) = true
  rw [h₁, h₂]; rfl

/-- The producer's initialization of a slot's empty barrier precedes the consumer's arrive on it:
through the full barrier, whose complete the consumer waited for. -/
theorem eb_before (hp : ∀ q < L.N, (pcs q).Ok L q) (hcs : ∀ q < L.N, (ccs q).Ok L q)
    (hnd : c.evs.Nodup) (hc : c.exec.consistent = true) {r : Nat} (hr : r < rounds L.S L.N s) :
    c.exec.causeBase (at_ c d (iEe L re s)) (at_ c d (eWe L ccs (stg L s r))) = true := by
  have hq := (stg_lt hs).1 hr
  have hsync := fb_raw c hE hnd hc d hs hp hcs hr
  have hI := at_get c (d := d) (iE_mem c hE hs)
  have hC := at_get c (d := d) (mem_cW c hE hq)
  have hR := at_get c (d := d) (mem_rfE c hE hq)
  have hW := at_get c (d := d) (mem_eW c hE hq)
  refine causeBase_trans (causeBase_of_order (Candidate.order_of_issue c (p := (0, cpK L pcs (stg L s r)))
    (by rw [hC]; simp [cWe, aft]) (by rw [hI]; simp [iEe, rmwW, Req.event]) ?_))
    (causeBase_trans (causeBase_of_sync hsync) (causeBase_of_order (Candidate.order_of_po c
      (by rw [hR, hW]; simp [rfE, waitEv, oneEv, eWe, rmwW, Req.event]) ?_)))
  · rw [hI]; simp only [iEe, rmwW, cpK]; have := posP_ge L pcs (stg L s r); omega
  · rw [hR, hW]; simp only [rfE, waitEv, oneEv, eWe, rmwW]; omega

include hnd hc in
theorem eb_loc (hp : ∀ q < L.N, (pcs q).Ok L q) (hcs : ∀ q < L.N, (ccs q).Ok L q) :
    OneLoc c.exec (at_ c d (iEe L re s)) where
  strong a b ha hb hsa hsb := Candidate.strongR_of c (pl := ⟨0, 0, 0⟩) place_eq hsa hsb (by
    have ka := eb_key c hE d hs ha
    have kb := eb_key c hE d hs hb
    simp only [Fin.getElem_fin] at ka kb ⊢
    rw [(empty_gen c hE hs (List.getElem_mem _) ka).2, (empty_gen c hE hs (List.getElem_mem _) kb).2])
  addr a ha := (sameLocation_iff.1 ha).2.2
  write := by
    show c.evs[at_ c d (iEe L re s)].e.isWrite = true
    rw [at_get c (iE_mem c hE hs)]; rfl
  nonweak w hw hwl hne := by
    have hk := eb_key c hE d hs hwl
    simp only [Fin.getElem_fin] at hk
    rcases empty_writes c hE hs (List.getElem_mem _) hk hw with h | ⟨q, -, -, h⟩
    · exact absurd (at_eq c hnd h) hne
    · show c.evs[w.1].e.scope ≠ .thread
      rw [h]; simp [eWe, rmwW, Req.event, Req.eventScope, Layout.arriveE, mbarReq]
  first v hv hvl hne := by
    have hk := eb_key c hE d hs hvl
    simp only [Fin.getElem_fin] at hk
    have hI := at_get c (d := d) (iE_mem c hE hs)
    have hIw : c.exec.writes (at_ c d (iEe L re s)) = true := by
      show c.evs[at_ c d (iEe L re s)].e.isWrite = true
      rw [hI]; rfl
    rcases empty_writes c hE hs (List.getElem_mem _) hk hv with h | ⟨q, hq, hqs, h⟩
    · exact absurd (at_eq c hnd h) hne
    · have he : stg L s (q / L.S) = q := hqs ▸ stg_of L q
      have hr : q / L.S < rounds L.S L.N s := (stg_lt hs).2 (by rw [he]; exact hq)
      have hv' : v = at_ c d (eWe L ccs (stg L s (q / L.S))) := at_eq c hnd (by rw [he]; exact h)
      subst hv'
      refine co_of_ppcb hc hIw hv (sameLocation_symm hvl) (Candidate.ppcb_gen c hI
        (at_get c (mem_eW c hE (by rw [he]; exact hq)))
        (gen_mbar rfl).2 (gen_mbar rfl).2
        (by simp [iEe, eWe, rmwW, Ev.key, empty_at (stg_mod hs (q / L.S))])
        (eb_before c hE d hs hp hcs hnd hc hr))

theorem eb_iv : Pipeline.mword (c.exec.val (at_ c d (iEe L re s))) = Pipeline.fresh 0 := by
  show Pipeline.mword c.evs[at_ c d (iEe L re s)].val = _
  rw [at_get c (iE_mem c hE hs)]
  have h : MBarrierWord.apply (.init 1) (Pipeline.mword (re s 0)) = .ok (Pipeline.fresh 0) := by
    simp [MBarrierWord.apply, NV.PTX.MBarrier.init, NV.PTX.MBarrier.countMax, Pipeline.fresh]
  simp only [iEe, rmwW]
  rw [Pipeline.written_mbar (r := L.initE s) rfl h, Option.getD_some, Pipeline.mword_bytes64]

omit hE hs in
theorem eb_zero : initBytes (program L x).mem (iEe L re s).key = List.replicate 8 0 := by
  simp [initBytes, program, iEe, rmwW, Ev.key, sloc, List.range_succ]

theorem eb_mem0 (w : UInt64) : MBarrierWord.apply (.arrive 1 0 false)
    (Pipeline.mword (c.exec.init (c.exec.loc (c.exec.ev (at_ c d (iEe L re s))).addr))) ≠ .ok w := by
  rw [Candidate.init_at c, at_get c (iE_mem c hE hs), eb_zero]
  intro h
  have h0 : (MBarrierWord.apply (.arrive 1 0 false) (Pipeline.mword (List.replicate 8 0))).toBool =
    false := by decide +kernel
  rw [h] at h0; cases h0

theorem eb_eW {r : Nat} (hr : r < rounds L.S L.N s) :
    c.exec.writes (at_ c d (eWe L ccs (stg L s r))) = true ∧
      c.exec.sameLocation (at_ c d (eWe L ccs (stg L s r))) (at_ c d (iEe L re s)) = true := by
  have hg := at_get c (d := d) (mem_eW c hE ((stg_lt hs).1 hr))
  have hw : c.exec.writes (at_ c d (eWe L ccs (stg L s r))) = true := by
    show c.evs[at_ c d (eWe L ccs (stg L s r))].e.isWrite = true
    rw [hg]; simp [eWe, rmwW, Req.event, Event.isWrite]
  refine ⟨hw, ?_⟩
  exact Candidate.sameLoc_of c hg (at_get c (iE_mem c hE hs)) (gen_mbar rfl).2.1 (gen_mbar rfl).2.1
    (by simp [eWe, iEe, rmwW, Ev.key, empty_at (stg_mod hs r)])

theorem eb_ine {r : Nat} (hr : r < rounds L.S L.N s) :
    at_ c d (eWe L ccs (stg L s r)) ≠ at_ c d (iEe L re s) :=
  at_ne c (mem_eW c hE ((stg_lt hs).1 hr)) (iE_mem c hE hs) fun h => by
    have := congrArg (·.e.thread) h; simp [eWe, iEe, rmwW, Req.event] at this

include hnd in
theorem eb_all (v : Fin c.evs.length) (hv : c.exec.writes v = true)
    (hvl : c.exec.sameLocation v (at_ c d (iEe L re s)) = true) :
    v = at_ c d (iEe L re s) ∨ ∃ r < rounds L.S L.N s, v = at_ c d (eWe L ccs (stg L s r)) := by
  have hk := eb_key c hE d hs hvl
  simp only [Fin.getElem_fin] at hk
  rcases empty_writes c hE hs (List.getElem_mem _) hk hv with h | ⟨q, hq, hqs, h⟩
  · exact .inl (at_eq c hnd h)
  · have he : stg L s (q / L.S) = q := hqs ▸ stg_of L q
    exact .inr ⟨q / L.S, (stg_lt hs).2 (by rw [he]; exact hq), at_eq c hnd (by rw [he]; exact h)⟩

theorem eb_rmw {r : Nat} (hr : r < rounds L.S L.N s) :
    c.exec.rmw (at_ c d (eRe L ccs (stg L s r))) (at_ c d (eWe L ccs (stg L s r))) = true ∧
      c.exec.proxyPreservedCauseBase (at_ c d (eRe L ccs (stg L s r)))
        (at_ c d (eWe L ccs (stg L s r))) = true ∧
      c.exec.sameLocation (at_ c d (eRe L ccs (stg L s r))) (at_ c d (iEe L re s)) = true := by
  have hq := (stg_lt hs).1 hr
  have hR := at_get c (d := d) (mem_eR c hE hq)
  have hW := at_get c (d := d) (mem_eW c hE hq)
  obtain ⟨h₁, h₂, h₃⟩ := rmw_pair c hR hW rfl rfl rfl (gen_mbar rfl).1 (gen_mbar rfl).2 rfl
  exact ⟨h₁, h₂, sameLocation_trans h₃ (eb_eW c hE d hs hr).2⟩

include hc in
theorem eb_next {r : Nat} (hr : r + 1 < rounds L.S L.N s) :
    c.exec.co (at_ c d (eWe L ccs (stg L s r))) (at_ c d (eWe L ccs (stg L s (r + 1)))) = true := by
  have h₁ := at_get c (d := d) (mem_eW c hE ((stg_lt hs).1 (Nat.lt_of_succ_lt hr)))
  have h₂ := at_get c (d := d) (mem_eW c hE ((stg_lt hs).1 hr))
  refine co_of_ppcb hc (eb_eW c hE d hs (Nat.lt_of_succ_lt hr)).1 (eb_eW c hE d hs hr).1
    (sameLocation_trans (eb_eW c hE d hs (Nat.lt_of_succ_lt hr)).2
      (sameLocation_symm (eb_eW c hE d hs hr).2))
    (Candidate.ppcb_gen c h₁ h₂ (gen_mbar rfl).2 (gen_mbar rfl).2
      (by simp [eWe, rmwW, Ev.key, empty_at (stg_mod hs r), empty_at (stg_mod hs (r + 1))])
      (causeBase_of_order (Candidate.order_of_po c (by rw [h₁, h₂]; rfl) ?_)))
  rw [h₁, h₂]; simp only [eWe, rmwW]
  have := posC_lt ccs (stg_lt_stg (s := s) (L := L) (show r < r + 1 by omega))
  omega

theorem eb_v (hcs : ∀ q < L.N, (ccs q).Ok L q) {r : Nat} (hr : r < rounds L.S L.N s) :
    MBarrierWord.apply (.arrive 1 0 false) (Pipeline.mword (c.exec.val (at_ c d (eRe L ccs (stg L s r))))) =
      .ok (Pipeline.mword (c.exec.val (at_ c d (eWe L ccs (stg L s r))))) := by
  have hq := (stg_lt hs).1 hr
  obtain ⟨-, -, -, -, hre, hu⟩ := hcs _ hq
  show MBarrierWord.apply _ (Pipeline.mword c.evs[at_ c d (eRe L ccs (stg L s r))].val) =
    .ok (Pipeline.mword c.evs[at_ c d (eWe L ccs (stg L s r))].val)
  rw [at_get c (mem_eR c hE hq), at_get c (mem_eW c hE hq)]
  exact fb_val rfl rfl (hre 0 (by simp [Req.active, Layout.arriveE, mbarReq])) hu

include hnd hc in
/-- **A slot's empty barrier**, round `r` being stage `stg s r`. -/
noncomputable def emptyB (hp : ∀ q < L.N, (pcs q).Ok L q) (hcs : ∀ q < L.N, (ccs q).Ok L q) :
    Pipeline.Empty c.exec Pipeline.mword where
  R := rounds L.S L.N s
  i := at_ c d (iEe L re s)
  eR r := at_ c d (eRe L ccs (stg L s r))
  eW r := at_ c d (eWe L ccs (stg L s r))
  loc := eb_loc c hE hnd hc d hs hp hcs
  i_v := eb_iv c hE d hs
  mem0 := eb_mem0 c hE d hs
  eW_w _ hr := eb_eW c hE d hs hr
  i_ne _ hr := eb_ine c hE d hs hr
  all := eb_all c hE hnd d hs
  e_rmw _ hr := eb_rmw c hE d hs hr
  e_next _ hr := eb_next c hE hc d hs hr
  e_v _ hr := eb_v c hE d hs hcs hr


omit hE in
theorem pe_stage {r : Nat} (hr : r < rounds L.S L.N s - 1) :
    stg L s (r + 1) < L.N ∧ L.S ≤ stg L s (r + 1) := by
  refine ⟨(stg_lt hs).1 (by omega), ?_⟩
  simp only [stg, Nat.succ_mul]; omega

theorem ew_read {r : Nat} (hr : r < rounds L.S L.N s - 1) :
    c.exec.reads (at_ c d (peE L pcs (stg L s (r + 1)))) = true ∧
      c.exec.sameLocation (at_ c d (peE L pcs (stg L s (r + 1)))) (at_ c d (iEe L re s)) = true ∧
      (c.exec.ev (at_ c d (peE L pcs (stg L s (r + 1))))).acquire = true ∧
      (c.exec.ev (at_ c d (peE L pcs (stg L s (r + 1))))).reduction = false ∧
      (c.exec.ev (at_ c d (peE L pcs (stg L s (r + 1))))).scope ≠ .thread := by
  obtain ⟨hq, hqS⟩ := pe_stage hs hr
  have hg := at_get c (d := d) (mem_pe c hE hq hqS)
  have hI := at_get c (d := d) (iE_mem c hE hs)
  refine ⟨?_, Candidate.sameLoc_of c hg hI (by simp [peE, waitEv, oneEv, Req.event, Event.isMem,
    Event.isRead]) (gen_mbar rfl).2.1 (by simp [peE, waitEv, oneEv, iEe, rmwW, Ev.key,
    empty_at (stg_mod hs (r + 1))]), ?_, ?_, ?_⟩
  · show c.evs[at_ c d (peE L pcs (stg L s (r + 1)))].e.isRead = true
    rw [hg]; simp [peE, waitEv, oneEv, Req.event, Event.isRead]
  · show c.evs[at_ c d (peE L pcs (stg L s (r + 1)))].e.acquire = true
    rw [hg]; simp [peE, waitEv, oneEv, Req.event, Req.acquires, waitReq]
  · show c.evs[at_ c d (peE L pcs (stg L s (r + 1)))].e.reduction = false
    rw [hg]; simp [peE, waitEv, oneEv, Req.event, Req.isRed, waitReq]
  · show c.evs[at_ c d (peE L pcs (stg L s (r + 1)))].e.scope ≠ .thread
    rw [hg]; simp [peE, waitEv, oneEv, Req.event, Req.eventScope, waitReq]

theorem ew_done (hp : ∀ q < L.N, (pcs q).Ok L q) {r : Nat} (hr : r < rounds L.S L.N s - 1) :
    MBarrierWord.parityDone (Pipeline.mword (c.exec.val (at_ c d (peE L pcs (stg L s (r + 1)))))) r =
      true := by
  obtain ⟨hq, hqS⟩ := pe_stage hs hr
  obtain ⟨-, hrw, hd⟩ := (hp _ hq).1 hqS
  have hl : ((pcs (stg L s (r + 1))).rw 0).length = 8 :=
    (hrw 0 (by simp [Req.active, waitReq])).trans (by rw [Req.readLen, pieces_wait]; rfl)
  show MBarrierWord.parityDone (Pipeline.mword c.evs[at_ c d (peE L pcs (stg L s (r + 1)))].val) r = true
  rw [at_get c (mem_pe c hE hq hqS)]
  simp only [peE, waitEv, oneEv, Bool.false_eq_true, ite_false]
  rw [take_all hl, ← word64_wait hl]
  rw [stg_div hs (r + 1), Nat.add_sub_cancel] at hd
  exact hd

theorem ew_init {r : Nat} (hr : r < rounds L.S L.N s - 1) :
    c.exec.proxyPreservedCauseBase (at_ c d (iEe L re s)) (at_ c d (peE L pcs (stg L s (r + 1)))) =
      true := by
  obtain ⟨hq, hqS⟩ := pe_stage hs hr
  have hI := at_get c (d := d) (iE_mem c hE hs)
  have hg := at_get c (d := d) (mem_pe c hE hq hqS)
  refine Candidate.ppcb_gen c hI hg (gen_mbar rfl).2
    (by simp [peE, waitEv, oneEv, Req.event, Event.isMem, Event.isRead, waitReq])
    (by simp [peE, waitEv, oneEv, iEe, rmwW, Ev.key, empty_at (stg_mod hs (r + 1))])
    (causeBase_of_order (Candidate.order_of_po c (by rw [hI, hg]; simp [iEe, rmwW, peE, waitEv, oneEv,
      Req.event]) ?_))
  rw [hI, hg]; simp only [iEe, rmwW, peE, waitEv, oneEv]
  have := posP_ge L pcs (stg L s (r + 1)); omega

theorem ew_next {r : Nat} (hr : r + 1 < rounds L.S L.N s - 1) :
    c.exec.causeBase (at_ c d (peE L pcs (stg L s (r + 1))))
      (at_ c d (peE L pcs (stg L s (r + 1 + 1)))) = true := by
  obtain ⟨hq₁, hqS₁⟩ := pe_stage hs (show r < rounds L.S L.N s - 1 by omega)
  obtain ⟨hq₂, hqS₂⟩ := pe_stage hs hr
  have h₁ := at_get c (d := d) (mem_pe c hE hq₁ hqS₁)
  have h₂ := at_get c (d := d) (mem_pe c hE hq₂ hqS₂)
  refine causeBase_of_order (Candidate.order_of_po c (by rw [h₁, h₂]; rfl) ?_)
  rw [h₁, h₂]; simp only [peE, waitEv, oneEv]
  have := posP_lt L pcs (stg_lt_stg (s := s) (L := L) (show r + 1 < r + 1 + 1 by omega))
  have hoff : pOff L (stg L s (r + 1)) (pcs (stg L s (r + 1))) = (pcs (stg L s (r + 1))).ws.length + 2 := by
    simp [pOff, show ¬ stg L s (r + 1) < L.S by omega]
  omega

theorem ew_rel {r : Nat} (hr : r < rounds L.S L.N s) :
    (c.exec.ev (at_ c d (eWe L ccs (stg L s r)))).release = true := by
  show c.evs[at_ c d (eWe L ccs (stg L s r))].e.release = true
  rw [at_get c (mem_eW c hE ((stg_lt hs).1 hr))]
  simp [eWe, rmwW, Req.event, Req.releases, Layout.arriveE, mbarReq]

include hnd hc in
/-- **The producer's waits on a slot's empty barrier**: from the slot's second round on, the wait
for the consumer's arrive of the round before. -/
noncomputable def emptyW (hp : ∀ q < L.N, (pcs q).Ok L q) (hcs : ∀ q < L.N, (ccs q).Ok L q) :
    Pipeline.Empty.Waits (emptyB c hE hnd hc d hs hp hcs) where
  M := rounds L.S L.N s - 1
  M_le := Nat.sub_le _ _
  Pe r := at_ c d (peE L pcs (stg L s (r + 1)))
  read _ hr := ew_read c hE d hs hr
  done _ hr := ew_done c hE d hs hp hr
  init _ hr := ew_init c hE d hs hr
  next _ hr := ew_next c hE d hs hr
  release _ hr := ew_rel c hE d hs hr
  generic _ ha := eb_generic c hE d hs ha

include hnd hc in
/-- **The empty barrier's order**: the producer's wait before reusing a slot for round `r + 1`
synchronizes with the consumer's arrive of round `r`. -/
theorem eb_war (hp : ∀ q < L.N, (pcs q).Ok L q) (hcs : ∀ q < L.N, (ccs q).Ok L q) {r : Nat}
    (hr : r < rounds L.S L.N s - 1) :
    c.exec.sync (at_ c d (eWe L ccs (stg L s r))) (at_ c d (peE L pcs (stg L s (r + 1)))) = true :=
  Pipeline.Empty.war _ hc (emptyW c hE hnd hc d hs hp hcs) r hr

end emptyB

/-! ## The ring in the run's execution -/

section ringI
variable {L : Layout} {x : Nat → Fin 4 → UInt8} (c : Candidate (program L x))
  {rf re : Nat → Nat → List UInt8} {pcs : Nat → PStage} {ccs : Nat → CStage}
  (hE : ∀ v, v ∈ c.evs ↔
    v ∈ (program L x).events [(pActs L rf re pcs, .exit), (cActs L ccs, .exit)])
  (hnd : c.evs.Nodup) (hc : c.exec.consistent = true) (d : Fin c.evs.length)
  (hp : ∀ q < L.N, (pcs q).Ok L q) (hcs : ∀ q < L.N, (ccs q).Ok L q)

theorem slot_mod (q : Nat) : L.slot (q % L.S) = L.slot q := by simp [Layout.slot]

theorem sameBlock_all (a b : Fin c.evs.length) : c.exec.sameBlock a b = true := by
  simp [Execution.sameBlock, Execution.placeOf, Candidate.exec, toExec, place_eq]

include hE

theorem ri_write {q : Nat} (hq : q < L.N) :
    c.exec.writes (at_ c d (wYe L pcs q)) = true ∧
      (c.exec.ev (at_ c d (wYe L pcs q))).addr = (c.exec.ev (at_ c d (wYe L pcs (q % L.S)))).addr := by
  have hg := at_get c (d := d) (mem_wY c hE hq)
  have hg' := at_get c (d := d) (mem_wY c hE (Nat.lt_of_le_of_lt (Nat.mod_le q L.S) hq))
  refine ⟨?_, (Candidate.addr_iff c).2 (by rw [hg, hg']; simp [wYe, aft, oneEv, Ev.key, slot_mod])⟩
  show c.evs[at_ c d (wYe L pcs q)].e.isWrite = true
  rw [hg]; simp [wYe, aft, oneEv, Req.event, Event.isWrite]

theorem ri_apart {i j : Nat} (hi : i < L.S) (hj : j < L.S) (hiN : i < L.N) (hjN : j < L.N)
    (h : c.exec.loc (c.exec.ev (at_ c d (wYe L pcs i))).addr =
      c.exec.loc (c.exec.ev (at_ c d (wYe L pcs j))).addr) : i = j := by
  have hk := (Candidate.addr_iff c).1 h
  rw [at_get c (mem_wY c hE hiN), at_get c (mem_wY c hE hjN)] at hk
  simp [wYe, aft, oneEv, Ev.key, sloc, Layout.slot, Nat.mod_eq_of_lt hi, Nat.mod_eq_of_lt hj] at hk
  omega

theorem ri_fence {q : Nat} (hq : q < L.N) :
    (c.exec.ev (at_ c d (fCe L pcs q))).proxyFenced = some (c.exec.ev (at_ c d (wYe L pcs q))).proxy ∧
      c.exec.sameBlock (at_ c d (fCe L pcs q)) (at_ c d (wYe L pcs q)) = true ∧
      tc c.exec.order (at_ c d (wYe L pcs q)) (at_ c d (fCe L pcs q)) = true := by
  have hF := at_get c (d := d) (mem_fC c hE hq)
  have hW := at_get c (d := d) (mem_wY c hE hq)
  refine ⟨?_, sameBlock_all c _ _, Candidate.order_of_po c (by rw [hW, hF]; simp [wYe, fCe, aft, oneEv,
    fenceEvent, Req.event]) (by rw [hW, hF]; simp [wYe, fCe, aft, oneEv, fenceEvent])⟩
  show Event.proxyFenced c.evs[at_ c d (fCe L pcs q)].e = some c.evs[at_ c d (wYe L pcs q)].e.proxy
  rw [hF, hW]
  simp [fCe, wYe, aft, oneEv, fenceEvent, Event.proxyFenced, Req.event, storeReq, Move.store, dstReq,
    asyncReq]

theorem ri_read {q : Nat} (hq : q < L.N) :
    c.exec.reads (at_ c d (ldE L ccs q)) = true ∧ c.exec.generic (at_ c d (ldE L ccs q)) = true ∧
      (c.exec.ev (at_ c d (ldE L ccs q))).addr = (c.exec.ev (at_ c d (wYe L pcs (q % L.S)))).addr := by
  have hR := at_get c (d := d) (mem_ld c hE hq)
  have hW := at_get c (d := d) (mem_wY c hE (Nat.lt_of_le_of_lt (Nat.mod_le q L.S) hq))
  refine ⟨?_, ?_, (Candidate.addr_iff c).2 (by rw [hR, hW]; simp [ldE, wYe, aft, oneEv, Ev.key, slot_mod])⟩
  · show c.evs[at_ c d (ldE L ccs q)].e.isRead = true
    rw [hR]; simp [ldE, oneEv, Req.event, Event.isRead]
  · show (c.evs[at_ c d (ldE L ccs q)].e.isMem && c.evs[at_ c d (ldE L ccs q)].e.proxy == .generic) = true
    rw [hR]; simp [ldE, oneEv, Req.event, Event.isMem, Event.isRead, Layout.ldSlot]

theorem ri_reuse {q : Nat} (hq : q < L.N) (hqS : L.S ≤ q) :
    (c.exec.ev (at_ c d (gE L pcs q))).proxyFenced = some (c.exec.ev (at_ c d (wYe L pcs q))).proxy ∧
      c.exec.sameBlock (at_ c d (gE L pcs q)) (at_ c d (wYe L pcs q)) = true ∧
      c.exec.causeBase (at_ c d (gE L pcs q)) (at_ c d (wYe L pcs q)) = true := by
  have hG := at_get c (d := d) (mem_g c hE hq hqS)
  have hW := at_get c (d := d) (mem_wY c hE hq)
  refine ⟨?_, sameBlock_all c _ _, causeBase_of_order (Candidate.order_of_issue c (p := (0, cpK L pcs q))
    (by rw [hW]; simp [wYe, aft]) (by rw [hG]; simp [gE, fenceEvent]) ?_)⟩
  · show Event.proxyFenced c.evs[at_ c d (gE L pcs q)].e = some c.evs[at_ c d (wYe L pcs q)].e.proxy
    rw [hG, hW]
    simp [gE, wYe, aft, oneEv, fenceEvent, Event.proxyFenced, Req.event, storeReq, Move.store, dstReq,
      asyncReq]
  · rw [hG]; simp [gE, fenceEvent, cpK, pOff, show ¬ q < L.S by omega]; omega

include hnd hc hp hcs in
theorem ri_raw {q : Nat} (hq : q < L.N) :
    c.exec.causeBase (at_ c d (fCe L pcs q)) (at_ c d (ldE L ccs q)) = true := by
  have hs := Nat.mod_lt q L.S_pos
  have hr : q / L.S < rounds L.S L.N (q % L.S) := (stg_lt hs).2 (by rw [stg_of]; exact hq)
  have hsync := fb_raw c hE hnd hc d hs hp hcs hr
  rw [stg_of] at hsync
  have hF := at_get c (d := d) (mem_fC c hE hq)
  have hC := at_get c (d := d) (mem_cW c hE hq)
  have hRf := at_get c (d := d) (mem_rfE c hE hq)
  have hL := at_get c (d := d) (mem_ld c hE hq)
  refine causeBase_trans (causeBase_of_order (Candidate.order_of_po c
    (by rw [hF, hC]; simp [fCe, cWe, aft, fenceEvent, rmwW, Req.event]) (by rw [hF, hC]; simp [fCe, cWe,
      aft, fenceEvent, rmwW])))
    (causeBase_trans (causeBase_of_sync hsync) (causeBase_of_order (Candidate.order_of_po c
      (by rw [hRf, hL]; simp [rfE, ldE, waitEv, oneEv, Req.event]) (by rw [hRf, hL]; simp [rfE, ldE,
        waitEv, oneEv]))))

omit hE in
/-- Stage `q` of the slot's round `r + 1` follows stage `q - S` of round `r`. -/
theorem stg_back {q : Nat} (hqS : L.S ≤ q) :
    stg L (q % L.S) (q / L.S - 1) = q - L.S ∧ stg L (q % L.S) (q / L.S - 1 + 1) = q := by
  have h := Nat.mod_add_div q L.S
  have h1 : 1 ≤ q / L.S := (Nat.le_div_iff_mul_le L.S_pos).2 (by omega)
  have e : q / L.S - 1 + 1 = q / L.S := by omega
  refine ⟨?_, ?_⟩
  · rw [stg, Nat.sub_one_mul, Nat.mul_comm (q / L.S)]
    have : L.S ≤ L.S * (q / L.S) := Nat.le_mul_of_pos_right _ h1
    omega
  · rw [e, stg, Nat.mul_comm (q / L.S)]; omega

include hnd hc hp hcs in
theorem ri_war {q : Nat} (hq : q < L.N) (hqS : L.S ≤ q) :
    c.exec.causeBase (at_ c d (ldE L ccs (q - L.S))) (at_ c d (gE L pcs q)) = true := by
  have hs := Nat.mod_lt q L.S_pos
  obtain ⟨h₁, h₂⟩ := stg_back (L := L) hqS
  have hr : q / L.S - 1 < rounds L.S L.N (q % L.S) - 1 := by
    have := (stg_lt hs (r := q / L.S - 1 + 1)).2 (by rw [h₂]; exact hq)
    omega
  have hsync := eb_war c hE hnd hc d hs hp hcs hr
  rw [h₁, h₂] at hsync
  have hq' : q - L.S < L.N := by omega
  have hL := at_get c (d := d) (mem_ld c hE hq')
  have hEw := at_get c (d := d) (mem_eW c hE hq')
  have hP := at_get c (d := d) (mem_pe c hE hq hqS)
  have hG := at_get c (d := d) (mem_g c hE hq hqS)
  have o₁ : tc c.exec.order (at_ c d (ldE L ccs (q - L.S))) (at_ c d (eWe L ccs (q - L.S))) = true := by
    refine Candidate.order_of_po c ?_ ?_ <;> rw [hL, hEw]
    · simp [ldE, eWe, oneEv, rmwW, Req.event]
    · simp [ldE, eWe, oneEv, rmwW]
  have o₂ : tc c.exec.order (at_ c d (peE L pcs q)) (at_ c d (gE L pcs q)) = true := by
    refine Candidate.order_of_po c ?_ ?_ <;> rw [hP, hG]
    · simp [peE, gE, waitEv, oneEv, fenceEvent, Req.event]
    · simp [peE, gE, waitEv, oneEv, fenceEvent]
  exact causeBase_trans (causeBase_of_order o₁)
    (causeBase_trans (causeBase_of_sync hsync) (causeBase_of_order o₂))

include hnd hc hp hcs in
/-- **The run's ring**: `Examples.Ring.Race.Ring`'s orders, from the barriers. -/
noncomputable def ring : Examples.Ring.Race.Ring c.exec where
  N := L.N
  S := L.S
  C := 1
  slots_pos := L.S_pos
  consumers_pos := Nat.one_pos
  W q := at_ c d (wYe L pcs q)
  F q := at_ c d (fCe L pcs q)
  R q _ := at_ c d (ldE L ccs q)
  G q := at_ c d (gE L pcs q)
  A k := (c.exec.ev (at_ c d (wYe L pcs k))).addr
  slots_apart _ hi _ hj hiN hjN h := ri_apart c hE d hi hj hiN hjN h
  write _ hq := ri_write c hE d hq
  fence _ hq := ri_fence c hE d hq
  read _ hq _ _ := ri_read c hE d hq
  reuse _ hq hqS := ri_reuse c hE d hq hqS
  raw _ hq _ _ := ri_raw c hE hnd hc d hp hcs hq
  war _ hq hqS _ _ := ri_war c hE hnd hc d hp hcs hq hqS

include hnd hc hp hcs in
/-- **Each consumer load reads its stage's copy.** -/
theorem reads_copy {q : Nat} (hq : q < L.N) :
    c.exec.rf (at_ c d (wYe L pcs q)) (at_ c d (ldE L ccs q)) = true := by
  refine (ring c hE hnd hc d hp hcs).reads_stage hc (fun q hq _ _ e hw hl => ?_) hq Nat.one_pos
  have hq : q < L.N := hq
  have hl : c.exec.sameLocation e (at_ c d (ldE L ccs q)) = true := hl
  show ∃ q' < L.N, e = at_ c d (wYe L pcs q')
  have hR := at_get c (d := d) (mem_ld c hE hq)
  have hk := key_of c hl
  rw [hR] at hk
  simp only [Fin.getElem_fin] at hk
  have hs := Nat.mod_lt q L.S_pos
  obtain ⟨q', hq', -, h | h⟩ := on_slot c hE (List.getElem_mem _) (s := q % L.S)
    (by rw [hk]; simp [ldE, oneEv, Ev.key, Layout.slot])
  · exact ⟨q', hq', at_eq c hnd h⟩
  · exfalso
    have : c.evs[e.1].e.isWrite = true := hw
    rw [h] at this; simp [ldE, oneEv, Req.event, Event.isWrite] at this

end ringI

/-! ## The values -/

section values
variable {L : Layout} {pcs : Nat → PStage} {ccs : Nat → CStage}

theorem wY_val {q : Nat} (h : ((pcs q).sr 0 0).length = 4) : (wYe L pcs q).val = (pcs q).sr 0 0 := by
  match hb : (pcs q).sr 0 0, h with
  | [b₀, b₁, b₂, b₃], _ =>
    simp [wYe, aft, oneEv, Req.written, storeReq, Move.store, Move.data, theMove, dstReq, srcReq,
      asyncReq, Req.storeSlice_eq, Req.storeBytes, Req.operands, Req.resp, Req.laneResp, words, MemOp.returns, hb, toWords,
      toBytes]
    rw [wordBytes_leNat rfl]; rfl

theorem st_val {q : Nat} (h : ((ccs q).ry 0).length = 4) : (stE L ccs q).val = (ccs q).ry 0 := by
  match hb : (ccs q).ry 0, h with
  | [b₀, b₁, b₂, b₃], _ =>
    simp [stE, oneEv, Req.written, Layout.stOut, Req.storeSlice_eq, Req.storeBytes, Req.operands, Req.resp, Req.laneResp,
      Layout.ldSlot, words, MemOp.returns, hb, toWords, toBytes]
    rw [wordBytes_leNat rfl]; rfl

end values

/-! ## Every completed outcome -/

section final
variable {L : Layout} {x : Nat → Fin 4 → UInt8} (c : Candidate (program L x))
  {rf re : Nat → Nat → List UInt8} {pcs : Nat → PStage} {ccs : Nat → CStage}
  (hE : ∀ v, v ∈ c.evs ↔
    v ∈ (program L x).events [(pActs L rf re pcs, .exit), (cActs L ccs, .exit)])
  (hnd : c.evs.Nodup) (hc : c.exec.consistent = true) (d : Fin c.evs.length)
  (hp : ∀ q < L.N, (pcs q).Ok L q) (hcs : ∀ q < L.N, (ccs q).Ok L q)

theorem init_x {q i : Nat} (hq : q < L.N) (hi : i < 4) :
    (initBytes (program L x).mem (gloc (L.X + 4 * q), 4)).getD i 0 = x q ⟨i, hi⟩ := by
  simp only [initBytes, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hi,
    Option.map_some, Option.getD_some, program, gloc]
  split
  · congr 1
    · omega
    · ext; simp; omega
  · rename_i h; exact absurd ⟨trivial, trivial, by omega, by omega⟩ h

include hE hc in
/-- The copy's load reads `X`'s initial bytes: nothing writes `X`. -/
theorem x_read {q : Nat} (hq : q < L.N) :
    c.evs[at_ c d (xRe L pcs q)].val = initBytes (program L x).mem (gloc (L.X + 4 * q), 4) := by
  have hg := at_get c (d := d) (mem_xR c hE hq)
  rcases Candidate.source c hc (r := at_ c d (xRe L pcs q))
      (by rw [hg]; simp [xRe, aft, oneEv, Req.event, Event.isRead]) with ⟨-, hv⟩ | ⟨w, -, -, hw, hk⟩
  · rw [hv, hg]; rfl
  · exfalso
    rw [hg] at hk
    have := on_x c hE (List.getElem_mem w.isLt) hq
      ⟨by have := congrArg Prod.fst hk; simpa [xRe, aft, oneEv, Ev.key] using this,
       by have := congrArg Prod.snd hk; simpa [xRe, aft, oneEv, Ev.key] using this⟩
    have hw' : c.evs[w.1].e.isWrite = true := hw
    rw [this] at hw'; simp [xRe, aft, oneEv, Req.event, Event.isWrite] at hw'

include hE hnd hc hp hcs in
/-- **Stage `q`'s load returns word `q` of `X`.** -/
theorem stage_value {q : Nat} (hq : q < L.N) :
    (ccs q).ry 0 = initBytes (program L x).mem (gloc (L.X + 4 * q), 4) := by
  have d : Fin c.evs.length := ix (mem_iF c hE L.S_pos)
  obtain ⟨-, -, -, hok⟩ := hp q hq
  have hsr : ((pcs q).sr 0 0).length = 4 :=
    ((hok (srcReq L q) ((pcs q).sr 0) (by rw [copyActs_eq]; simp)).1 0
      (by simp [Req.active, srcReq, asyncReq])).trans
      (by simp [Req.readLen, Req.pieces, Req.granule, Req.strong, srcReq, asyncReq])
  have hry : ((ccs q).ry 0).length = 4 :=
    ((hcs q hq).2.2.2.1 0 (by simp [Req.active, Layout.ldSlot])).trans (by rw [Req.readLen, pieces_ld]; rfl)
  have hrf := rf_val hc (reads_copy c hE hnd hc d hp hcs hq)
  have hL : c.evs[at_ c d (ldE L ccs q)].val = c.evs[at_ c d (wYe L pcs q)].val := hrf
  rw [at_get c (mem_ld c hE hq), at_get c (mem_wY c hE hq), wY_val hsr] at hL
  have hX := x_read c hE hc d hq
  rw [at_get c (mem_xR c hE hq)] at hX
  simp only [ldE, oneEv, Bool.false_eq_true, ite_false, take_all hry] at hL
  simp only [xRe, aft, oneEv, Bool.false_eq_true, ite_false, take_all hsr] at hX
  rw [hL, hX]

include hE hnd hc hp hcs in
/-- **The output's word `q` is `X`'s word `q`** in every final memory. -/
theorem final_out {m : Loc → UInt8} (hf : c.IsFinal m) {q i : Nat} (hq : q < L.N) (hi : i < 4) :
    m (gloc (L.O + 4 * q + i)) = x q ⟨i, hi⟩ := by
  have d : Fin c.evs.length := ix (mem_iF c hE L.S_pos)
  have hry : ((ccs q).ry 0).length = 4 :=
    ((hcs q hq).2.2.2.1 0 (by simp [Req.active, Layout.ldSlot])).trans (by rw [Req.readLen, pieces_ld]; rfl)
  have hSt := mem_st c hE hq
  have hcov : (stE L ccs q).covers (gloc (L.O + 4 * q + i)) = true := by
    rw [NV.covers_iff]
    refine ⟨by simp [stE, oneEv, Req.event, Event.isWrite], rfl, rfl, by simp [stE, oneEv, gloc],
      by simp [stE, oneEv, gloc]; omega⟩
  rcases hf (gloc (L.O + 4 * q + i)) with ⟨j, hj, -, hm⟩ | ⟨hn, -⟩
  · obtain ⟨hw, hsp, -, hlo, hhi⟩ := NV.covers_iff.1 hj
    obtain ⟨q', hq', h⟩ := global_write c hE (List.getElem_mem j.isLt) hw (by simpa [gloc] using hsp)
    have h' : c.evs[j] = stE L ccs q' := h
    rw [h'] at hlo hhi
    simp [stE, oneEv, gloc] at hlo hhi
    have : q' = q := by omega
    subst this
    rw [hm, h', st_val hry, stage_value c hE hnd hc hp hcs hq]
    simp only [stE, oneEv, gloc]
    rw [show L.O + 4 * q' + i - (L.O + 4 * q') = i by omega]
    exact init_x hq hi
  · obtain ⟨j, hj⟩ := Candidate.index c hSt
    rw [← hj, hn j] at hcov; cases hcov

end final

/-- **The ring is correct**: every completed outcome of the ring program holds, at word `q` of `O`,
word `q` of `X`, for every layout (`N` words through `S` slots) and every `X`. -/
theorem ring_correct {L : Layout} {x : Nat → Fin 4 → UInt8} {m : Loc → UInt8}
    (h : Outcomes (program L x) (.done m)) {q i : Nat} (hq : q < L.N) (hi : i < 4) :
    m (gloc (L.O + 4 * q + i)) = x q ⟨i, hi⟩ := by
  obtain ⟨c, -, hc, hex, -, hf⟩ := h
  obtain ⟨⟨pa, pe⟩, ⟨ca, ce⟩, hruns⟩ : ∃ p₁ p₂, c.runs = [p₁, p₂] := by
    match hr : c.runs, c.length with
    | [p₁, p₂], _ => exact ⟨p₁, p₂, rfl⟩
  have hpe : pe = .exit := hex (pa, pe) (by simp [hruns])
  have hce : ce = .exit := hex (ca, ce) (by simp [hruns])
  subst hpe hce
  have hrp : Run (step L) { pc := .initF 0 } pa .exit :=
    c.runs_ok (pAgent, pa, .exit) (by simp [hruns, agents_eq, pAgent])
  have hrc : Run (step L) { pc := .cstage 0 } ca .exit :=
    c.runs_ok (cAgent, ca, .exit) (by simp [hruns, agents_eq, cAgent])
  obtain ⟨rf, re, pcs, rfl, hp⟩ := producer_initF _ 0 rfl L.S_pos hrp
  obtain ⟨ccs, rfl, hcs⟩ := consumer_run _ 0 rfl hrc
  simp only [Nat.sub_zero] at *
  have hE : ∀ v, v ∈ c.evs ↔
      v ∈ (program L x).events [(pActs L rf re pcs, .exit), (cActs L ccs, .exit)] := fun v => by
    rw [c.perm.mem_iff, hruns]; rfl
  have hnd : c.evs.Nodup := c.perm.nodup_iff.2 (by rw [hruns]; exact events_nodup L x rf re pcs ccs)
  exact final_out c hE hnd hc hp (fun q hq => hcs q (Nat.zero_le _) hq) hf hq hi

end Examples.RingProgram
