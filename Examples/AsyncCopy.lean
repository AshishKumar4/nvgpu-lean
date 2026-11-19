import Examples.SplitMergeProgram
import NV.Async.Op
import NV.Async.Tma

/-!
# An asynchronous copy, completed through an mbarrier

One thread of a CTA initializes an mbarrier for one arrival, arrives on it expecting 4 bytes of
transactions (`mbarrier.arrive.expect_tx`), and issues an asynchronous copy of a global word `X`
into a shared word `Y` (`cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes`): the
copy's agent loads `X` and stores `Y` through the async proxy, then, after the implicit
generic-async proxy fence, performs complete-tx of 4 bytes on the mbarrier (release, cluster
scope). The thread spins on `mbarrier.try_wait.parity` for phase 0 (an acquire load of the
barrier word), then loads `Y` and stores it to the global word `O`.

`copy_correct`: every outcome the PTX memory model allows is a completed run whose `O` holds
`X`'s initial bytes, and none faults or leaves the model (the arrive reads init's word, the
complete-tx reads the arrive's, and both are defined on those). The proof finds the wait that returned true: only complete-tx's write has
phase 1, so the wait read it; the release/acquire pair and the copy's proxy fence order the
async store of `Y` before the generic load of `Y` (`ppcb_async_completion`), which therefore
reads it.
-/

namespace Examples.AsyncCopy

open NV NV.Formal NV.PTX NV.Async NV.Rel NV.PTX.Execution

/-- Where the example keeps its words: `X` and `O` in global memory, the mbarrier `B` and `Y` in
the CTA's shared memory. -/
structure Layout where
  X : UInt64
  O : UInt64
  B : UInt64
  Y : UInt64
  global : X.toNat + 4 ≤ O.toNat
  shared : Y.toNat + 4 ≤ B.toNat

def mbarReq (L : Layout) (op : MBarOp) (sem : NV.Sem) (scope : Scope) : Req :=
  { space := .shared, op := .mbar op, bytes := 8, sem, scope, addrs := #[some L.B] }

def initReq (L : Layout) : Req := mbarReq L (.init 1) .weak .cta
def arriveReq (L : Layout) : Req := mbarReq L (.arrive 1 4) .release .cta
def completeReq (L : Layout) : Req := mbarReq L (.completeTx 4) .release .cluster
def waitReq (L : Layout) : Req :=
  { space := .shared, op := .ld, bytes := 8, sem := .acquire, scope := .cta, addrs := #[some L.B] }
def ldY (L : Layout) : Req := { space := .shared, op := .ld, bytes := 4, addrs := #[some L.Y] }
def stO (L : Layout) (v : UInt32) : Req :=
  { space := .global, op := .st, bytes := 4, addrs := #[some L.O], data := #[v] }

/-- The copy: one move of 4 bytes, then complete-tx. -/
def copy (L : Layout) : Copy :=
  { moves := #[{ src := asyncReq .global .ld 4 #[some L.X], dsts := #[asyncReq .shared .st 4 #[some L.Y]] }]
    signals := #[completeReq L] }

/-- The 64-bit word a response's two words spell. -/
def word64 (resp : Resp) : UInt64 :=
  (resp[0]?.getD 0).toUInt64 ||| ((resp[1]?.getD 0).toUInt64 <<< (32 : UInt64))

structure S where
  pc : Nat
  done : Bool := false
  v : UInt32 := 0

def step (L : Layout) (s : S) : Effect S :=
  match s.pc with
  | 0 => .mem (fun _ => initReq L) { s with pc := 1 } fun _ t => t
  | 1 => .mem (fun _ => arriveReq L) { s with pc := 2 } fun _ t => t
  | 2 => .async (fun _ => .copy (copy L)) { s with pc := 3 } fun _ t => t
  | 3 => .mem (fun _ => waitReq L) { s with pc := 4 } fun r t =>
      { t with done := MBarrierWord.parityDone (word64 r) 0 }
  | 4 => if s.done then .next { s with pc := 5 } else .next { s with pc := 3 }
  | 5 => .mem (fun _ => ldY L) { s with pc := 6 } fun r t => { t with v := r[0]! }
  | 6 => .mem (fun _ => stO L s.v) { s with pc := 7 } fun _ t => t
  | _ => .exit

/-- `x` is `X`'s initial bytes. -/
def program (L : Layout) (x : Fin 4 → UInt8) : Program S where
  step := step L
  agents := [{ init := { pc := 0 }, tid := 0, place := ⟨0, 0, 0⟩ }]
  mem l := if h : l.space = .global ∧ l.owner = 0 ∧ L.X.toNat ≤ l.addr ∧ l.addr < L.X.toNat + 4
    then x ⟨l.addr - L.X.toNat, by omega⟩ else 0

/-! ## The thread's runs -/

/-- The bytes of a 64-bit word, low first. -/
def bytes64 (w : UInt64) : List UInt8 := wordBytes w.toUInt32 ++ wordBytes (w >>> 32).toUInt32

section shape
variable {L : Layout}

theorem active0 {r : Req} {a : UInt64} (h : r.addrs = #[some a]) : r.active 0 = true := by
  simp [Req.active, h]

theorem init_ok (L : Layout) (read : Nat → List UInt8) : (initReq L).unsupported read = false := by
  simp [Req.unsupported, initReq, mbarReq, MemOp.writes, Req.active, Req.atomNew,
    MBarrierWord.apply, MBarrier.init, MBarrier.countMax]

theorem wait_ok (L : Layout) (read : Nat → List UInt8) : (waitReq L).unsupported read = false := by
  simp [Req.unsupported, waitReq, MemOp.writes]

theorem ld_ok (L : Layout) (read : Nat → List UInt8) : (ldY L).unsupported read = false := by
  simp [Req.unsupported, ldY, MemOp.writes]

theorem st_ok (L : Layout) (v : UInt32) (read : Nat → List UInt8) :
    (stO L v).unsupported read = false := by
  simp [Req.unsupported, stO, MemOp.writes]

/-- The waits of a run: some that returned false, then one that returned true. -/
theorem waits {d : Bool} {v : UInt32} {acts : List Act} {e : End}
    (h : Run (step L) { pc := 3, done := d, v } acts e) :
    e = .stop ∨ ∃ (ws : List (Nat → List UInt8)), ∃ rw rest,
      acts = ws.map (Act.mem (waitReq L)) ++ .mem (waitReq L) rw :: rest ∧
      (∀ l, (waitReq L).active l → (rw l).length = (waitReq L).readLen) ∧
      MBarrierWord.parityDone (word64 ((waitReq L).resp rw)) 0 = true ∧
      Run (step L) { pc := 5, done := true, v } rest e := by
  induction hn : acts.length using Nat.strongRecOn generalizing acts d with
  | ind n ih =>
    rcases Run.of_mem (s' := { pc := 4, done := d, v }) rfl h with ⟨-, rfl⟩ | ⟨read, -, -, -, hu⟩ |
      ⟨rw, rest, rfl, hl, -, h⟩
    · exact .inl rfl
    · rw [wait_ok] at hu; cases hu
    cases hd : MBarrierWord.parityDone (word64 ((waitReq L).resp rw)) 0
    · rcases Run.of_next (s' := { pc := 3, done := false, v }) (by simp [step, hd]) h with
        ⟨-, rfl⟩ | h
      · exact .inl rfl
      rcases ih rest.length (by rw [← hn]; simp) h rfl with he | ⟨ws, rw', rest', rfl, hl', hd', h'⟩
      · exact .inl he
      exact .inr ⟨rw :: ws, rw', rest', by simp, hl', hd', h'⟩
    · rcases Run.of_next (s' := { pc := 5, done := true, v }) (by simp [step, hd]) h with
        ⟨-, rfl⟩ | h
      · exact .inl rfl
      exact .inr ⟨[], rw, rest, by simp, hl, hd, h⟩

/-- The thread's runs: cut short; the arrive left undefined; the copy's agent failing; or the
whole program, its last wait returning true. -/
theorem run_shape {acts : List Act} {e : End} (h : Run (step L) { pc := 0 } acts e) :
    e = .stop ∨
    (∃ r₀ r₁, acts = [.mem (initReq L) r₀, .fail (arriveReq L) r₁] ∧ e = failureOf (arriveReq L) ∧
      (∀ l, (arriveReq L).active l → (r₁ l).length = (arriveReq L).readLen) ∧
      (arriveReq L).unsupported r₁ = true) ∨
    (∃ r₀ r₁ agent, acts = [.mem (initReq L) r₀, .mem (arriveReq L) r₁, .async (.copy (copy L)) agent] ∧
      (∀ l, (arriveReq L).active l → (r₁ l).length = (arriveReq L).readLen) ∧
      AgentRun (.copy (copy L)) agent (.error e)) ∨
    (∃ r₀ r₁ agent res, ∃ (ws : List (Nat → List UInt8)), ∃ rw ry ro,
      acts = [.mem (initReq L) r₀, .mem (arriveReq L) r₁, .async (.copy (copy L)) agent] ++
        ws.map (Act.mem (waitReq L)) ++
        [.mem (waitReq L) rw, .mem (ldY L) ry, .mem (stO L ((ldY L).resp ry)[0]!) ro] ∧
      (∀ l, (arriveReq L).active l → (r₁ l).length = (arriveReq L).readLen) ∧
      AgentRun (.copy (copy L)) agent (.ok res) ∧ e = .exit ∧
      (∀ l, (waitReq L).active l → (rw l).length = (waitReq L).readLen) ∧
      MBarrierWord.parityDone (word64 ((waitReq L).resp rw)) 0 = true ∧
      (∀ l, (ldY L).active l → (ry l).length = (ldY L).readLen)) := by
  rcases Run.of_mem (s' := { pc := 1 }) rfl h with ⟨-, rfl⟩ | ⟨read, -, -, -, hu⟩ |
    ⟨r₀, rest, rfl, -, -, h⟩
  · exact .inl rfl
  · rw [init_ok] at hu; cases hu
  rcases Run.of_mem (s' := { pc := 2 }) rfl h with ⟨-, rfl⟩ | ⟨r₁, rfl, rfl, hl, hu⟩ |
    ⟨r₁, rest, rfl, hl₁, -, h⟩
  · exact .inl rfl
  · exact .inr (.inl ⟨r₀, r₁, rfl, rfl, hl, hu⟩)
  rcases Run.of_async (s' := { pc := 3 }) rfl h with ⟨-, rfl⟩ | ⟨agent, rfl, ha⟩ |
    ⟨agent, res, rest, rfl, ha, h⟩
  · exact .inl rfl
  · exact .inr (.inr (.inl ⟨r₀, r₁, agent, rfl, hl₁, ha⟩))
  rcases waits h with he | ⟨ws, rw, rest, rfl, hlw, hd, h⟩
  · exact .inl he
  rcases Run.of_mem (s' := { pc := 6, done := true }) rfl h with ⟨-, rfl⟩ | ⟨read, -, -, -, hu⟩ |
    ⟨ry, rest, rfl, hly, -, h⟩
  · exact .inl rfl
  · rw [ld_ok] at hu; cases hu
  rcases Run.of_mem (s' := { pc := 7, done := true, v := ((ldY L).resp ry)[0]! }) rfl h with
    ⟨-, rfl⟩ | ⟨read, -, -, -, hu⟩ | ⟨ro, rest, rfl, -, -, h⟩
  · exact .inl rfl
  · rw [st_ok] at hu; cases hu
  obtain ⟨rfl, he⟩ := Run.of_exit rfl h
  rcases he with rfl | rfl
  · exact .inr (.inr (.inr ⟨r₀, r₁, agent, res, ws, rw, ry, ro, by simp, hl₁, ha, rfl, hlw, hd, hly⟩))
  · exact .inl rfl

end shape

/-! ## The run's events -/

section events
variable (L : Layout) (x : Fin 4 → UInt8)

def locB : Loc := { space := .shared, owner := 0, addr := L.B.toNat }
def locY : Loc := { space := .shared, owner := 0, addr := L.Y.toNat }
def locX : Loc := { space := .global, owner := 0, addr := L.X.toNat }
def locO : Loc := { space := .global, owner := 0, addr := L.O.toNat }

def theAgent : Agent S := { init := { pc := 0 }, tid := 0, place := ⟨0, 0, 0⟩ }

theorem agents_eq : (program L x).agents = [theAgent] := rfl

theorem threads_eq : (program L x).threads = 1 := rfl

theorem loc_one {r : Req} {a : UInt64} (ha : r.addrs = #[some a]) :
    theAgent.loc r 0 = some { space := r.space, owner := 0, addr := a.toNat } := by
  simp only [Agent.loc, ha]
  cases r.space <;> rfl

/-- The events of a one-lane request by thread 0. -/
theorem actEvents_mem {r : Req} {a : UInt64} (ha : r.addrs = #[some a]) (c : Nat → Nat)
    (read : Nat → List UInt8) :
    (program L x).actEvents theAgent c (.mem r read) =
      r.laneEvents 0 0 (c 0) { space := r.space, owner := 0, addr := a.toNat } (read 0) := by
  simp only [Program.actEvents, ha, Array.size_singleton, List.range_one, List.flatMap_cons,
    List.flatMap_nil, List.append_nil]
  rw [loc_one ha]
  rfl

theorem width_mem {r : Req} {a : UInt64} (ha : r.addrs = #[some a]) (read : Nat → List UInt8)
    (t : Nat) : Act.width theAgent t (.mem r read) = if t = 0 then r.pieces.length else 0 := by
  simp only [Act.width, theAgent, Nat.zero_le, true_and, Nat.sub_zero, Req.active, ha]
  by_cases h : t = 0
  · subst h; simp
  · simp only [h, ite_false]
    refine ite_eq_right_iff.mpr fun hc => ?_
    have : [some a][t]? = none := List.getElem?_eq_none (by simp; omega)
    simp [this] at hc


/-- Thread 0's event at position `pos` of a wait that returned `w`. -/
def waitEv (pos : Nat) (w : List UInt8) : Ev :=
  { e := (waitReq L).event 0 ⟨false, 0, 8, false⟩, seq := pos, loc := locB L, size := 8,
    val := w.extract 0 8 }

theorem waitReq_events (pos : Nat) (w : List UInt8) :
    (waitReq L).laneEvents 0 0 pos (locB L) w = [waitEv L pos w] := by
  simp [Req.laneEvents, Req.pieces, Req.granule, Req.strong, Req.elemBytes, waitReq, waitEv, locB]
  rfl

/-- The events of consecutive waits from position `pos`. -/
def waitEvs : Nat → List (Nat → List UInt8) → List Ev
  | _, [] => []
  | pos, w :: ws => waitEv L pos (w 0) :: waitEvs (pos + 1) ws

theorem width_wait (w : Nat → List UInt8) (t : Nat) :
    Act.width theAgent t (.mem (waitReq L) w) = if t = 0 then 1 else 0 := by
  rw [width_mem (r := waitReq L) (a := L.B) rfl,
    show (waitReq L).pieces.length = 1 by simp [Req.pieces, Req.granule, Req.strong, Req.elemBytes, waitReq]]

theorem agentEvents_waits (ws : List (Nat → List UInt8)) :
    ∀ (c : Nat → Nat) (rest : List Act),
      (program L x).agentEvents theAgent c (ws.map (Act.mem (waitReq L)) ++ rest) =
        waitEvs L (c 0) ws ++
          (program L x).agentEvents theAgent (fun t => c t + if t = 0 then ws.length else 0) rest := by
  induction ws with
  | nil => intro c rest; simp [waitEvs]
  | cons w ws ih =>
    intro c rest
    have hc : (fun t => c t + Act.width theAgent t (.mem (waitReq L) w) +
        if t = 0 then ws.length else 0) = (fun t => c t + if t = 0 then ws.length + 1 else 0) := by
      funext t; rw [width_wait]; by_cases h : t = 0 <;> simp [h]; omega
    show (program L x).actEvents theAgent c (.mem (waitReq L) w) ++
      (program L x).agentEvents theAgent (fun t => c t + Act.width theAgent t (.mem (waitReq L) w))
        (ws.map (Act.mem (waitReq L)) ++ rest) = _
    rw [ih, hc, actEvents_mem L x (r := waitReq L) (a := L.B) rfl, width_wait]
    rw [show ({ space := (waitReq L).space, owner := 0, addr := L.B.toNat } : Loc) = locB L from rfl,
      waitReq_events]
    simp [waitEvs]

end events

/-! ## The copy's agent -/

section agent
variable (L : Layout) (x : Fin 4 → UInt8)

def srcReq : Req := asyncReq .global .ld 4 #[some L.X]
def dstReq : Req := asyncReq .shared .st 4 #[some L.Y]
def theMove : Move := { src := srcReq L, dsts := #[dstReq L] }

/-- The store of the copy, once its load returned `bs`. -/
def storeReq (bs : Nat → List UInt8) : Req := (theMove L).store (dstReq L) ((srcReq L).resp bs)

theorem copyActs_eq (sr gr : Nat → Nat → List UInt8) :
    copyActs (copy L) sr gr =
      [.mem (srcReq L) (sr 0), .mem (storeReq L (sr 0)) (fun _ => []), .proxyFence,
        .mem (completeReq L) (gr 0)] := by
  simp [copyActs, copy, srcReq, dstReq, storeReq, theMove]

/-- The copy's agent's events: on thread 1, after thread 0's first four. -/
def asyncEvs (sr gr : Nat → Nat → List UInt8) : List Ev :=
  ((srcReq L).laneEvents 0 1 0 (locX L) (sr 0 0) ++
    (storeReq L (sr 0)).laneEvents 0 1 1 (locY L) [] ++
    [fenceEvent (.proxyFence .async) 1 2] ++
    (completeReq L).laneEvents 0 1 3 (locB L) (gr 0 0)).map ({ · with after := [(0, 4)] })

theorem aactEvents_copy (sr gr : Nat → Nat → List UInt8) :
    theAgent.aactEvents 1 [(0, 4)] 0 (copyActs (copy L) sr gr) = asyncEvs L sr gr := by
  rw [copyActs_eq]
  simp [Agent.aactEvents, asyncEvs, Req.active, srcReq, storeReq, theMove, dstReq, completeReq,
    mbarReq, asyncReq, Agent.loc, theAgent, locX, locY, locB, Move.store, Req.pieces, Req.granule,
    Req.strong]

end agent

/-! ## A finished run's events -/

section full
variable (L : Layout) (x : Fin 4 → UInt8)

/-- The acts of a finished run. -/
def fullActs (r₀ r₁ : Nat → List UInt8) (agent : List AAct) (ws : List (Nat → List UInt8))
    (rw ry ro : Nat → List UInt8) : List Act :=
  [.mem (initReq L) r₀, .mem (arriveReq L) r₁, .async (.copy (copy L)) agent] ++
    ws.map (Act.mem (waitReq L)) ++
    [.mem (waitReq L) rw, .mem (ldY L) ry, .mem (stO L ((ldY L).resp ry)[0]!) ro]

/-- Thread 0's events of a finished run. -/
def threadEvs (r₀ r₁ : Nat → List UInt8) (ws : List (Nat → List UInt8)) (rw ry ro : Nat → List UInt8) :
    List Ev :=
  (initReq L).laneEvents 0 0 0 (locB L) (r₀ 0) ++ (arriveReq L).laneEvents 0 0 2 (locB L) (r₁ 0) ++
    waitEvs L 4 ws ++ [waitEv L (4 + ws.length) (rw 0)] ++
    (ldY L).laneEvents 0 0 (5 + ws.length) (locY L) (ry 0) ++
    (stO L ((ldY L).resp ry)[0]!).laneEvents 0 0 (6 + ws.length) (locO L) (ro 0)

theorem width_one {r : Req} {a : UInt64} {k : Nat} (ha : r.addrs = #[some a]) (hp : r.pieces.length = k)
    (read : Nat → List UInt8) (t : Nat) :
    Act.width theAgent t (.mem r read) = if t = 0 then k else 0 := by
  rw [width_mem ha, hp]

theorem agentEvents_full (r₀ r₁ : Nat → List UInt8) (agent : List AAct) (ws : List (Nat → List UInt8))
    (rw ry ro : Nat → List UInt8) :
    (program L x).agentEvents theAgent (fun _ => 0) (fullActs L r₀ r₁ agent ws rw ry ro) =
      threadEvs L r₀ r₁ ws rw ry ro := by
  have e₁ : ∀ c read, (program L x).actEvents theAgent c (.mem (initReq L) read) =
      (initReq L).laneEvents 0 0 (c 0) (locB L) (read 0) := fun c read => actEvents_mem L x rfl c read
  have e₂ : ∀ c read, (program L x).actEvents theAgent c (.mem (arriveReq L) read) =
      (arriveReq L).laneEvents 0 0 (c 0) (locB L) (read 0) := fun c read => actEvents_mem L x rfl c read
  have e₃ : ∀ c read, (program L x).actEvents theAgent c (.mem (waitReq L) read) =
      [waitEv L (c 0) (read 0)] := fun c read => by
    rw [actEvents_mem L x (r := waitReq L) (a := L.B) rfl]; exact waitReq_events L _ _
  have e₄ : ∀ c read, (program L x).actEvents theAgent c (.mem (ldY L) read) =
      (ldY L).laneEvents 0 0 (c 0) (locY L) (read 0) := fun c read => actEvents_mem L x rfl c read
  have e₅ : ∀ c v read, (program L x).actEvents theAgent c (.mem (stO L v) read) =
      (stO L v).laneEvents 0 0 (c 0) (locO L) (read 0) := fun c v read => actEvents_mem L x rfl c read
  have e₆ : ∀ c op ag, (program L x).actEvents theAgent c (.async op ag) = [] := fun _ _ _ => rfl
  have w₁ : ∀ t read, Act.width theAgent t (.mem (initReq L) read) = if t = 0 then 2 else 0 :=
    fun t read => width_one (r := initReq L) rfl (by simp [Req.pieces, initReq, mbarReq]) read t
  have w₂ : ∀ t read, Act.width theAgent t (.mem (arriveReq L) read) = if t = 0 then 2 else 0 :=
    fun t read => width_one (r := arriveReq L) rfl (by simp [Req.pieces, arriveReq, mbarReq]) read t
  have w₆ : ∀ t op ag, Act.width theAgent t (.async op ag) = 0 := fun _ _ _ => rfl
  simp only [fullActs, List.cons_append, List.nil_append, Program.agentEvents, e₁, e₂, e₆, w₁, w₂,
    w₆, List.nil_append]
  rw [agentEvents_waits]
  have w₃ : Act.width theAgent 0 (.mem (waitReq L) rw) = 1 := by rw [width_wait]; rfl
  have w₄ : Act.width theAgent 0 (.mem (ldY L) ry) = 1 := by
    rw [width_one (r := ldY L) (k := 1) rfl (by simp [Req.pieces, Req.granule, Req.strong, ldY])]; rfl
  simp only [Program.agentEvents, e₃, e₄, e₅, threadEvs, List.append_assoc, List.cons_append,
    List.nil_append, List.append_nil, w₃, w₄]
  simp only [ite_true, Nat.zero_add, Nat.add_zero]
  rw [show 2 + 2 + ws.length + 1 + 1 = 6 + ws.length by omega,
    show 2 + 2 + ws.length + 1 = 5 + ws.length by omega, show 2 + 2 + ws.length = 4 + ws.length by omega]

theorem asyncEvents_full (r₀ r₁ : Nat → List UInt8) (sr gr : Nat → Nat → List UInt8)
    (ws : List (Nat → List UInt8)) (rw ry ro : Nat → List UInt8) :
    (program L x).asyncEvents theAgent (fun _ => 0) {}
      (fullActs L r₀ r₁ (copyActs (copy L) sr gr) ws rw ry ro) = asyncEvs L sr gr := by
  have hplain : ∀ act ∈ ws.map (Act.mem (waitReq L)) ++
      [.mem (waitReq L) rw, .mem (ldY L) ry, .mem (stO L ((ldY L).resp ry)[0]!) ro], act.plain = true := by
    intro act h
    simp only [List.mem_append, List.mem_map, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with ⟨_, _, rfl⟩ | rfl | rfl | rfl <;> rfl
  simp only [fullActs, List.cons_append, List.nil_append]
  simp only [Program.asyncEvents]
  rw [Program.asyncEvents_plain _ _ _ _ _ hplain, List.append_nil]
  exact aactEvents_copy L sr gr

end full

/-! ## The events, one by one -/

section named
variable (L : Layout)

/-- The read and the write of an 8-byte read-modify-write `r` on `B` by thread `t` at position
`s`, its read having returned `bs`. -/
def rmwR (r : Req) (t s : Nat) (bs : List UInt8) : Ev :=
  { e := r.event t ⟨false, 0, 8, true⟩, seq := s, loc := locB L, size := 8, rmwRead := true,
    val := bs.extract 0 8 }
def rmwW (r : Req) (t s : Nat) (bs : List UInt8) : Ev :=
  { e := r.event t ⟨true, 0, 8, false⟩, seq := s + 1, loc := locB L, size := 8,
    val := (r.written 0 ⟨true, 0, 8, false⟩ bs).getD [] }

theorem rmw_events {r : Req} (hp : r.pieces = [⟨false, 0, 8, true⟩, ⟨true, 0, 8, false⟩])
    (t s : Nat) (bs : List UInt8) :
    r.laneEvents 0 t s (locB L) bs = [rmwR L r t s bs, rmwW L r t s bs] := by
  simp [Req.laneEvents, hp, rmwR, rmwW, locB]

theorem pieces_mbar {op : MBarOp} {sem : NV.Sem} {scope : Scope} :
    (mbarReq L op sem scope).pieces = [⟨false, 0, 8, true⟩, ⟨true, 0, 8, false⟩] := rfl

/-- A one-piece 4-byte access by thread `t` at position `s` of location `l`. -/
def oneEv (r : Req) (t s : Nat) (l : Loc) (write : Bool) (bs : List UInt8) : Ev :=
  { e := r.event t ⟨write, 0, 4, false⟩, seq := s, loc := l, size := 4,
    val := if write then (r.written 0 ⟨write, 0, 4, false⟩ bs).getD [] else bs.extract 0 4 }

theorem one_events {r : Req} {w : Bool} (hp : r.pieces = [⟨w, 0, 4, false⟩]) (t s : Nat) (l : Loc)
    (bs : List UInt8) : r.laneEvents 0 t s l bs = [oneEv r t s l w bs] := by
  simp [Req.laneEvents, hp, oneEv]

end named

/-! ## The execution at its events -/

section exec
variable {L : Layout} {x : Fin 4 → UInt8} (c : Candidate (program L x))

theorem po_iff {i j : Fin c.evs.length} :
    c.exec.po i j = true ↔ c.evs[i].e.thread = c.evs[j].e.thread ∧ c.evs[i].seq < c.evs[j].seq := by
  simp [Candidate.exec, toExec]

theorem issue_iff {i j : Fin c.evs.length} :
    c.exec.issue i j = true ↔ ∃ p ∈ c.evs[j].after, c.evs[i].e.thread = p.1 ∧ c.evs[i].seq < p.2 := by
  simp [Candidate.exec, toExec, List.any_eq_true]

theorem addr_iff {i j : Fin c.evs.length} :
    (c.exec.ev i).addr = (c.exec.ev j).addr ↔ c.evs[i].key = c.evs[j].key :=
  Performed.code_eq (List.getElem_mem _) (List.getElem_mem _)

theorem place_eq (t : Nat) : (program L x).place t = ⟨0, 0, 0⟩ :=
  Program.place_all (f := fun p => p = ⟨0, 0, 0⟩) (fun a ha => by
    simp only [program, List.mem_singleton] at ha; subst ha; rfl) rfl t

/-- An order of the same thread at increasing positions. -/
theorem order_of_po {i j : Fin c.evs.length} (ht : c.evs[i].e.thread = c.evs[j].e.thread)
    (hs : c.evs[i].seq < c.evs[j].seq) : tc c.exec.order i j = true :=
  tc_of (by simp [Execution.order, (po_iff c).2 ⟨ht, hs⟩])

theorem order_of_issue {i j : Fin c.evs.length} {p : Nat × Nat} (hp : p ∈ c.evs[j].after)
    (ht : c.evs[i].e.thread = p.1) (hs : c.evs[i].seq < p.2) : tc c.exec.order i j = true :=
  tc_of (by simp [Execution.order, (issue_iff c).2 ⟨p, hp, ht, hs⟩])

/-- An index of a member. -/
theorem index {v : Ev} (h : v ∈ c.evs) : ∃ i : Fin c.evs.length, c.evs[i] = v := by
  obtain ⟨i, hi, e⟩ := List.mem_iff_getElem.1 h
  exact ⟨⟨i, hi⟩, e⟩

end exec

/-! ## The barrier word's values -/

section values
variable (L : Layout)

def wInit : UInt64 := MBarrierWord.encode ⟨0, 1, 1, 0, true⟩
def wArr : UInt64 := MBarrierWord.encode ⟨0, 0, 1, 4, true⟩
def wCt : UInt64 := MBarrierWord.encode ⟨1, 1, 1, 0, false⟩

theorem init_written (bs : List UInt8) :
    (initReq L).written 0 ⟨true, 0, 8, false⟩ bs = some (bytes64 (wInit)) := by
  simp only [Req.written, Req.atomNew, initReq, mbarReq]
  rfl

theorem arrive_written :
    (arriveReq L).written 0 ⟨true, 0, 8, false⟩ (bytes64 wInit) = some (bytes64 wArr) := by
  simp only [Req.written, Req.atomNew, arriveReq, mbarReq]
  decide +kernel

theorem complete_written :
    (completeReq L).written 0 ⟨true, 0, 8, false⟩ (bytes64 wArr) = some (bytes64 wCt) := by
  simp only [Req.written, Req.atomNew, completeReq, mbarReq]
  decide +kernel

theorem bytes64_length (w : UInt64) : (bytes64 w).length = 8 := by simp [bytes64, wordBytes]

/-- The word a wait's response spells, from the bytes it read. -/
def waitWord (bs : List UInt8) : UInt64 :=
  word64 ((toWords bs).take 2 ++ List.replicate (2 - (toWords bs).length) 0).toArray

theorem resp_wait (rw : Nat → List UInt8) : word64 ((waitReq L).resp rw) = waitWord (rw 0) := by
  simp [Req.resp, waitReq, Req.laneResp, waitWord, MemOp.returns, words]

theorem done_init : MBarrierWord.parityDone (waitWord (bytes64 wInit)) 0 = false := by decide +kernel
theorem done_arr : MBarrierWord.parityDone (waitWord (bytes64 wArr)) 0 = false := by decide +kernel
theorem done_zero : MBarrierWord.parityDone (waitWord [0, 0, 0, 0, 0, 0, 0, 0]) 0 = false := by
  decide +kernel

end values

/-! ## A finished run, event by event -/

section finished
variable (L : Layout) (r₀ r₁ : Nat → List UInt8) (sr gr : Nat → Nat → List UInt8)
  (ws : List (Nat → List UInt8)) (rw ry ro : Nat → List UInt8)

def aft (v : Ev) : Ev := { v with after := [(0, 4)] }

def Ri : Ev := rmwR L (initReq L) 0 0 (r₀ 0)
def Wi : Ev := rmwW L (initReq L) 0 0 (r₀ 0)
def Ra : Ev := rmwR L (arriveReq L) 0 2 (r₁ 0)
def Wa : Ev := rmwW L (arriveReq L) 0 2 (r₁ 0)
def Rw : Ev := waitEv L (4 + ws.length) (rw 0)
def RY : Ev := oneEv (ldY L) 0 (5 + ws.length) (locY L) false (ry 0)
def WO : Ev := oneEv (stO L ((ldY L).resp ry)[0]!) 0 (6 + ws.length) (locO L) true (ro 0)
def RX : Ev := aft (oneEv (srcReq L) 1 0 (locX L) false (sr 0 0))
def WY : Ev := aft (oneEv (storeReq L (sr 0)) 1 1 (locY L) true [])
def Fe : Ev := aft (fenceEvent (.proxyFence .async) 1 2)
def Rc : Ev := aft (rmwR L (completeReq L) 1 3 (gr 0 0))
def Wc : Ev := aft (rmwW L (completeReq L) 1 3 (gr 0 0))

/-- The events of a finished run. -/
def evsOf : List Ev :=
  [Ri L r₀, Wi L r₀, Ra L r₁, Wa L r₁] ++ waitEvs L 4 ws ++
    [Rw L ws rw, RY L ws ry, WO L ws ry ro, RX L sr, WY L sr, Fe, Rc L gr, Wc L gr]

theorem evs_eq :
    threadEvs L r₀ r₁ ws rw ry ro ++ asyncEvs L sr gr = evsOf L r₀ r₁ sr gr ws rw ry ro := by
  simp only [threadEvs, asyncEvs, evsOf, initReq, arriveReq, completeReq, rmw_events L (pieces_mbar L),
    one_events (r := ldY L) (w := false) (by simp [Req.pieces, Req.granule, Req.strong, ldY]),
    one_events (r := stO L _) (w := true) (by simp [Req.pieces, Req.granule, Req.strong, stO]),
    one_events (r := srcReq L) (w := false)
      (by simp [Req.pieces, Req.granule, Req.strong, srcReq, asyncReq]),
    one_events (r := storeReq L (sr 0)) (w := true)
      (by simp [Req.pieces, Req.granule, Req.strong, storeReq, Move.store, dstReq, asyncReq]),
    List.map_cons, List.map_nil, List.append_assoc, List.cons_append,
    List.nil_append]
  rfl

theorem wait_mem {v : Ev} (h : v ∈ waitEvs L 4 ws) : ∃ pos w, v = waitEv L pos w := by
  revert h
  suffices ∀ p, v ∈ waitEvs L p ws → ∃ pos w, v = waitEv L pos w from this 4
  induction ws with
  | nil => intro p h; simp [waitEvs] at h
  | cons w ws ih =>
    intro p h
    simp only [waitEvs, List.mem_cons] at h
    rcases h with rfl | h
    · exact ⟨_, _, rfl⟩
    · exact ih (p + 1) h

end finished

/-! ## The barrier word's chain -/

section chain
variable {L : Layout} {x : Fin 4 → UInt8} (c : Candidate (program L x))
  {r₀ r₁ : Nat → List UInt8} {sr gr : Nat → Nat → List UInt8}
  {ws : List (Nat → List UInt8)} {rw ry ro : Nat → List UInt8}

/-- An order between two events of the run, from their threads and positions. -/
theorem ord_po {i j : Fin c.evs.length} {A B : Ev} (hi : c.evs[i] = A) (hj : c.evs[j] = B)
    (ht : A.e.thread = B.e.thread) (hs : A.seq < B.seq) : tc c.exec.order i j = true :=
  order_of_po c (by rw [hi, hj]; exact ht) (by rw [hi, hj]; exact hs)

theorem ord_issue {i j : Fin c.evs.length} {A B : Ev} (hi : c.evs[i] = A) (hj : c.evs[j] = B)
    (hp : (0, 4) ∈ B.after) (ht : A.e.thread = 0) (hs : A.seq < 4) : tc c.exec.order i j = true :=
  order_of_issue c (p := (0, 4)) (by rw [hj]; exact hp) (by rw [hi]; exact ht) (by rw [hi]; exact hs)

/-- Two generic accesses of one location, ordered: proxy-preserved base causality. -/
theorem ppcb_gen {i j : Fin c.evs.length} {A B : Ev} (hi : c.evs[i] = A) (hj : c.evs[j] = B)
    (hA : A.e.isMem = true ∧ A.e.proxy = .generic) (hB : B.e.isMem = true ∧ B.e.proxy = .generic)
    (hk : A.key = B.key) (ho : tc c.exec.order i j = true) :
    c.exec.proxyPreservedCauseBase i j = true := by
  have hg : ∀ {k : Fin c.evs.length} {C : Ev}, c.evs[k] = C → C.e.isMem = true ∧ C.e.proxy = .generic →
      c.exec.generic k = true := fun {k} {C} hk hC => by
    show (c.evs[k].e.isMem && c.evs[k].e.proxy == .generic) = true
    rw [hk, hC.1, hC.2]; rfl
  exact ppcb_of_generic (hg hi hA) (hg hj hB) ((addr_iff c).2 (by rw [hi, hj]; exact hk))
    (causeBase_of_order ho)

theorem writes_iff' {i : Fin c.evs.length} : c.exec.writes i = c.evs[i].e.isWrite := rfl
theorem reads_iff' {i : Fin c.evs.length} : c.exec.reads i = c.evs[i].e.isRead := rfl
theorem mems_iff' {i : Fin c.evs.length} : c.exec.mems i = c.evs[i].e.isMem := rfl
theorem val_iff' {i : Fin c.evs.length} : c.exec.val i = c.evs[i].val := rfl

theorem sameLoc_of {i j : Fin c.evs.length} {A B : Ev} (hi : c.evs[i] = A) (hj : c.evs[j] = B)
    (hA : A.e.isMem = true) (hB : B.e.isMem = true) (hk : A.key = B.key) :
    c.exec.sameLocation i j = true := by
  simp only [Execution.sameLocation, Rel.get_ofFn, Bool.and_eq_true, beq_iff_eq, mems_iff', hi, hj,
    hA, hB, true_and]
  exact congrArg _ ((addr_iff c).2 (by rw [hi, hj]; exact hk))

/-- A read's source: none (the initial bytes), or a write to its location. -/
theorem source (hc : c.exec.consistent = true) {r : Fin c.evs.length} (hr : c.evs[r].e.isRead = true) :
    ((∀ w, c.exec.rf w r = false) ∧ c.evs[r].val = initBytes (program L x).mem c.evs[r].key) ∨
      ∃ w, c.exec.rf w r = true ∧ c.evs[r].val = c.evs[w].val ∧ c.evs[w].e.isWrite = true ∧
        c.evs[w].key = c.evs[r].key := by
  rcases rf_cases hc (r := r) hr with ⟨hn, hv⟩ | ⟨w, h₁, h₂, h₃, h₄⟩
  · left
    refine ⟨hn, ?_⟩
    rw [val_iff'] at hv
    rw [hv]
    have hm : c.evs[r] ∈ c.evs := List.getElem_mem _
    have hlt := List.idxOf_lt_length_of_mem (List.mem_map_of_mem (f := Ev.key) hm)
    have hget : (c.evs.map Ev.key)[code c.evs c.evs[r]]? = some c.evs[r].key := by
      unfold code; rw [List.getElem?_eq_getElem hlt, List.getElem_idxOf hlt]
    show (match (c.evs.map Ev.key)[code c.evs c.evs[r]]? with
      | some key => initBytes (program L x).mem key | none => []) = _
    rw [hget]
  · exact .inr ⟨w, h₁, h₂, h₃, (addr_iff c).1 h₄⟩

end chain

section chain2
variable {L : Layout} {x : Fin 4 → UInt8} (c : Candidate (program L x))
  {r₀ r₁ : Nat → List UInt8} {sr gr : Nat → Nat → List UInt8}
  {ws : List (Nat → List UInt8)} {rw ry ro : Nat → List UInt8}

theorem locB_ne_locY : (locB L, 8) ≠ (locY L, 4) := by simp [locB, locY]
theorem locB_ne_locO : (locB L, 8) ≠ (locO L, 4) := by simp [locB, locO]
theorem locB_ne_locX : (locB L, 8) ≠ (locX L, 4) := by simp [locB, locX]

/-- The writes to the barrier word: init's, arrive's and complete-tx's. -/
theorem writers_B (hE : ∀ v, v ∈ c.evs ↔ v ∈ evsOf L r₀ r₁ sr gr ws rw ry ro) {j : Fin c.evs.length}
    (hw : c.evs[j].e.isWrite = true) (hk : c.evs[j].key = (locB L, 8)) :
    c.evs[j] = Wi L r₀ ∨ c.evs[j] = Wa L r₁ ∨ c.evs[j] = Wc L gr := by
  have hm := (hE _).1 (List.getElem_mem j.isLt)
  generalize c.evs[j] = v at hw hk hm ⊢
  have : v ∈ [Ri L r₀, Wi L r₀, Ra L r₁, Wa L r₁] ∨ v ∈ waitEvs L 4 ws ∨
      v ∈ [Rw L ws rw, RY L ws ry, WO L ws ry ro, RX L sr, WY L sr, Fe, Rc L gr, Wc L gr] := by
    simpa only [evsOf, List.mem_append, or_assoc] using hm
  rcases this with h | h | h
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl
    · exfalso; simp [Ri, rmwR, Req.event, Event.isWrite] at hw
    · exact .inl rfl
    · exfalso; simp [Ra, rmwR, Req.event, Event.isWrite] at hw
    · exact .inr (.inl rfl)
  · obtain ⟨pos, w, rfl⟩ := wait_mem L ws h
    exfalso; simp [waitEv, Req.event, Event.isWrite] at hw
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · exfalso; simp [Rw, waitEv, Req.event, Event.isWrite] at hw
    · exfalso; simp [RY, oneEv, Ev.key, locY, locB] at hk
    · exfalso; simp [WO, oneEv, Ev.key, locO, locB] at hk
    · exfalso; simp [RX, aft, oneEv, Ev.key, locX, locB] at hk
    · exfalso; simp [WY, aft, oneEv, Ev.key, locY, locB] at hk
    · exfalso; simp [Fe, aft, fenceEvent, Event.isWrite] at hw
    · exfalso; simp [Rc, aft, rmwR, Req.event, Event.isWrite] at hw
    · exact .inr (.inr rfl)

end chain2

section chain3
variable {L : Layout} {r₀ r₁ : Nat → List UInt8} {gr : Nat → Nat → List UInt8}
  {ws : List (Nat → List UInt8)} {rw : Nat → List UInt8}

/-- What each event of the barrier word is: a read or a write, generic, its thread and position. -/
structure Facts (v : Ev) (rd : Bool) (t s : Nat) : Prop where
  read : v.e.isRead = rd
  write : v.e.isWrite = !rd
  mem : v.e.isMem = true
  generic : v.e.proxy = .generic
  thread : v.e.thread = t
  seq : v.seq = s
  key : v.key = (locB L, 8)

theorem Ri_f : Facts (L := L) (Ri L r₀) true 0 0 := ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
theorem Wi_f : Facts (L := L) (Wi L r₀) false 0 1 := ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
theorem Ra_f : Facts (L := L) (Ra L r₁) true 0 2 := ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
theorem Wa_f : Facts (L := L) (Wa L r₁) false 0 3 := ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
theorem Rc_f : Facts (L := L) (Rc L gr) true 1 3 := ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
theorem Wc_f : Facts (L := L) (Wc L gr) false 1 4 := ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
theorem Rw_f : Facts (L := L) (Rw L ws rw) true 0 (4 + ws.length) :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem Rc_after : (0, 4) ∈ (Rc L gr).after := by simp [Rc, aft]
theorem Wc_after : (0, 4) ∈ (Wc L gr).after := by simp [Wc, aft]

end chain3

section chain4
variable {L : Layout} {x : Fin 4 → UInt8} (c : Candidate (program L x))
  {r₀ r₁ : Nat → List UInt8} {sr gr : Nat → Nat → List UInt8}
  {ws : List (Nat → List UInt8)} {rw ry ro : Nat → List UInt8}

theorem extract8 {bs : List UInt8} (h : bs.length = 8) : bs.extract 0 8 = bs := by
  simp [List.take_of_length_le (Nat.le_of_eq h)]

theorem extract4 {bs : List UInt8} (h : bs.length = 4) : bs.extract 0 4 = bs := by
  simp [List.take_of_length_le (Nat.le_of_eq h)]

theorem com_of_rf {i j : Fin c.evs.length} (h : c.exec.rf i j = true) : c.exec.com i j = true := by
  simp [Execution.com, h]

/-- Two events of the barrier word, the first before the second in their threads' order. -/
theorem ppcb_B {i j : Fin c.evs.length} {A B : Ev} {ra rb ta tb sa sb}
    (hi : c.evs[i] = A) (hj : c.evs[j] = B) (fa : Facts (L := L) A ra ta sa)
    (fb : Facts (L := L) B rb tb sb) (ho : tc c.exec.order i j = true) :
    c.exec.proxyPreservedCauseBase i j = true :=
  ppcb_gen c hi hj ⟨fa.mem, fa.generic⟩ ⟨fb.mem, fb.generic⟩ (fa.key.trans fb.key.symm) ho

/-- The writes to the barrier word are among init's, arrive's and complete-tx's (a run that stops
early has fewer). -/
def WritersB (c : Candidate (program L x)) (r₀ r₁ : Nat → List UInt8) (gr : Nat → Nat → List UInt8) :
    Prop :=
  ∀ j : Fin c.evs.length, c.evs[j].e.isWrite = true → c.evs[j].key = (locB L, 8) →
    c.evs[j] = Wi L r₀ ∨ c.evs[j] = Wa L r₁ ∨ c.evs[j] = Wc L gr

variable (hW : WritersB c r₀ r₁ gr) (hc : c.exec.consistent = true)
include hW hc

/-- **Arrive read init's word**: arrive's read `R` (of a run that completes arrive or not). -/
theorem arrive_reads {R : Ev} (fR : Facts (L := L) R true 0 2) (hRv : R.val = (r₁ 0).extract 0 8)
    (hWim : Wi L r₀ ∈ c.evs) (hRm : R ∈ c.evs) (hl : (r₁ 0).length = 8) : r₁ 0 = bytes64 wInit := by
  obtain ⟨iWi, hWi⟩ := index c hWim
  obtain ⟨iRa, hRa⟩ := index c hRm
  have pWiRa := ppcb_B c hWi hRa Wi_f fR
    (ord_po c hWi hRa (by rw [Wi_f.thread, fR.thread]) (by rw [Wi_f.seq, fR.seq]; decide))
  rcases source c hc (r := iRa) (by rw [hRa]; exact fR.read) with ⟨hn, -⟩ | ⟨w, hrf, hv, hw, hk⟩
  · exfalso
    obtain ⟨w', hw'⟩ := (read_sees_ppcb hc (by rw [writes_iff', hWi]; rfl)
      (by rw [reads_iff', hRa]; exact fR.read) (by rw [mems_iff', hWi]; rfl)
      (by rw [mems_iff', hRa]; exact fR.mem) ((addr_iff c).2 (by rw [hWi, hRa, fR.key]; rfl)) pWiRa).1
    rw [hn w'] at hw'; cases hw'
  · rw [hRa] at hk hv
    rcases hW w hw (hk.trans fR.key) with h | h | h
    · rw [h, hRv] at hv
      rw [← extract8 hl, hv]
      simp [Wi, rmwW, init_written]
    · exfalso
      exact not_com_of_ppcb hc (ppcb_B c hRa h fR Wa_f
        (ord_po c hRa h (by rw [fR.thread, Wa_f.thread]) (by rw [fR.seq, Wa_f.seq]; decide)))
        (com_of_rf c hrf)
    · exfalso
      exact not_com_of_ppcb hc (ppcb_B c hRa h fR Wc_f
        (ord_issue c hRa h Wc_after fR.thread (by rw [fR.seq]; decide)))
        (com_of_rf c hrf)

omit hW in
theorem co_B {i j : Fin c.evs.length} {A B : Ev} {sa sb ta tb}
    (hi : c.evs[i] = A) (hj : c.evs[j] = B) (fa : Facts (L := L) A false ta sa)
    (fb : Facts (L := L) B false tb sb) (ho : tc c.exec.order i j = true) : c.exec.co i j = true :=
  co_of_ppcb hc (by rw [writes_iff', hi]; exact fa.write) (by rw [writes_iff', hj]; exact fb.write)
    (sameLoc_of c hi hj fa.mem fb.mem (fa.key.trans fb.key.symm)) (ppcb_B c hi hj fa fb ho)

/-- **Complete-tx read arrive's word**: complete-tx's read `R` (of a copy that completes it or
not), arrive having read init's. -/
theorem complete_reads {R : Ev} (fR : Facts (L := L) R true 1 3) (hRa : (0, 4) ∈ R.after)
    (hRv : R.val = (gr 0 0).extract 0 8) (hWam : Wa L r₁ ∈ c.evs)
    (hRm : R ∈ c.evs) (h₁ : r₁ 0 = bytes64 wInit) (hl : (gr 0 0).length = 8) :
    gr 0 0 = bytes64 wArr := by
  obtain ⟨iWa, hWa⟩ := index c hWam
  obtain ⟨iRc, hRc⟩ := index c hRm
  have pWaRc := ppcb_B c hWa hRc Wa_f fR (ord_issue c hWa hRc hRa rfl (by rw [Wa_f.seq]; decide))
  have sees := read_sees_ppcb hc (by rw [writes_iff', hWa]; rfl) (by rw [reads_iff', hRc]; exact fR.read)
      (by rw [mems_iff', hWa]; rfl) (by rw [mems_iff', hRc]; exact fR.mem)
      ((addr_iff c).2 (by rw [hWa, hRc, fR.key]; rfl)) pWaRc
  rcases source c hc (r := iRc) (by rw [hRc]; exact fR.read) with ⟨hn, -⟩ | ⟨w, hrf, hv, hw, hk⟩
  · exfalso
    obtain ⟨w', hw'⟩ := sees.1
    rw [hn w'] at hw'; cases hw'
  · rw [hRc] at hk hv
    rcases hW w hw (hk.trans fR.key) with h | h | h
    · exfalso
      have hco := co_B c hc h hWa Wi_f Wa_f (ord_po c h hWa rfl (by rw [Wi_f.seq, Wa_f.seq]; decide))
      rw [sees.2 w hrf] at hco; cases hco
    · rw [h, hRv] at hv
      rw [← extract8 hl, hv]
      simp [Wa, rmwW, h₁, arrive_written]
    · exfalso
      exact not_com_of_ppcb hc (ppcb_B c hRc h fR Wc_f
        (ord_po c hRc h (by rw [fR.thread, Wc_f.thread]) (by rw [fR.seq, Wc_f.seq]; decide)))
        (com_of_rf c hrf)

omit hW hc in
theorem zeros_B (hw : (rw 0).length = 8) (h : (Rw L ws rw).val = initBytes (program L x).mem (Rw L ws rw).key) :
    rw 0 = [0, 0, 0, 0, 0, 0, 0, 0] := by
  rw [← extract8 hw]
  have : (Rw L ws rw).val = (rw 0).extract 0 8 := rfl
  rw [← this, h, Rw_f.key]
  simp [initBytes, program, locB]
  rfl

/-- **The wait that returned true read complete-tx's word.** -/
theorem wait_reads (h₁ : r₁ 0 = bytes64 wInit) (hRwm : Rw L ws rw ∈ c.evs) (hlw : (rw 0).length = 8)
    (hd : MBarrierWord.parityDone (waitWord (rw 0)) 0 = true) :
    ∃ iw ir : Fin c.evs.length, c.evs[iw] = Wc L gr ∧ c.evs[ir] = Rw L ws rw ∧ c.exec.rf iw ir = true := by
  obtain ⟨iRw, hRw⟩ := index c hRwm
  have hval : ∀ w : UInt64, (Rw L ws rw).val = bytes64 w → rw 0 = bytes64 w := fun w h => by
    rw [← extract8 hlw]; exact h
  rcases source c hc (r := iRw) (by rw [hRw]; rfl) with ⟨-, hv⟩ | ⟨w, hrf, hv, hw, hk⟩
  · rw [hRw] at hv
    rw [zeros_B hlw hv, done_zero] at hd; cases hd
  · rw [hRw] at hk hv
    rcases hW w hw (hk.trans Rw_f.key) with h | h | h
    · rw [h, show (Wi L r₀).val = bytes64 wInit by simp [Wi, rmwW, init_written]] at hv
      rw [hval _ hv, done_init] at hd; cases hd
    · rw [h, show (Wa L r₁).val = bytes64 wArr by simp [Wa, rmwW, h₁, arrive_written]] at hv
      rw [hval _ hv, done_arr] at hd; cases hd
    · exact ⟨w, iRw, h, hRw, hrf⟩

end chain4

/-! ## The data's path -/

section data
variable {L : Layout} {r₀ r₁ : Nat → List UInt8} {sr gr : Nat → Nat → List UInt8}
  {ws : List (Nat → List UInt8)} {rw ry ro : Nat → List UInt8}

theorem wordBytes_word {bs : List UInt8} (h : bs.length = 4) :
    wordBytes (Examples.SplitMerge.Program.word bs) = bs := by
  match bs, h with
  | [b₀, b₁, b₂, b₃], _ =>
    simp only [Examples.SplitMerge.Program.word]
    exact wordBytes_leNat rfl

/-- The copy stores what it loaded. -/
theorem WY_val (h : (sr 0 0).length = 4) : (WY L sr).val = sr 0 0 := by
  match hb : sr 0 0, h with
  | [b₀, b₁, b₂, b₃], _ =>
    simp [WY, aft, oneEv, Req.written, storeReq, Move.store, Move.data, theMove, dstReq, srcReq,
      asyncReq, Req.storeBytes, Req.operands, Req.resp, Req.laneResp, words, MemOp.returns, hb, toWords,
      toBytes]
    rw [wordBytes_leNat rfl]; rfl

theorem WO_val (h : (ry 0).length = 4) : (WO L ws ry ro).val = ry 0 := by
  match hb : ry 0, h with
  | [b₀, b₁, b₂, b₃], _ =>
    simp [WO, oneEv, Req.written, stO, Req.storeBytes, Req.operands, Req.resp, Req.laneResp, ldY,
      words, MemOp.returns, hb, toWords, toBytes]
    rw [wordBytes_leNat rfl]; rfl

section paths
variable {x : Fin 4 → UInt8} (c : Candidate (program L x))
variable (hE : ∀ v, v ∈ c.evs ↔ v ∈ evsOf L r₀ r₁ sr gr ws rw ry ro)
include hE

/-- The only write to a location other than the barrier word's is the one the program makes:
the copy's store to `Y`, the thread's store to `O`; nothing writes `X`. -/
theorem writers_other {j : Fin c.evs.length} (hw : c.evs[j].e.isWrite = true)
    (hk : c.evs[j].key ≠ (locB L, 8)) :
    (c.evs[j] = WY L sr ∧ c.evs[j].key = (locY L, 4)) ∨
      (c.evs[j] = WO L ws ry ro ∧ c.evs[j].key = (locO L, 4)) := by
  have hm := (hE _).1 (List.getElem_mem j.isLt)
  generalize c.evs[j] = v at hw hk hm ⊢
  have : v ∈ [Ri L r₀, Wi L r₀, Ra L r₁, Wa L r₁] ∨ v ∈ waitEvs L 4 ws ∨
      v ∈ [Rw L ws rw, RY L ws ry, WO L ws ry ro, RX L sr, WY L sr, Fe, Rc L gr, Wc L gr] := by
    simpa only [evsOf, List.mem_append, or_assoc] using hm
  rcases this with h | h | h
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl <;> exfalso <;> first
      | exact hk Wi_f.key | exact hk Wa_f.key
      | simp [Ri, Ra, rmwR, Req.event, Event.isWrite] at hw
  · obtain ⟨pos, w, rfl⟩ := wait_mem L ws h
    exfalso; simp [waitEv, Req.event, Event.isWrite] at hw
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · exfalso; simp [Rw, waitEv, Req.event, Event.isWrite] at hw
    · exfalso; simp [RY, oneEv, Req.event, Event.isWrite] at hw
    · exact .inr ⟨rfl, rfl⟩
    · exfalso; simp [RX, aft, oneEv, Req.event, Event.isWrite] at hw
    · exact .inl ⟨rfl, rfl⟩
    · exfalso; simp [Fe, aft, fenceEvent, Event.isWrite] at hw
    · exfalso; simp [Rc, aft, rmwR, Req.event, Event.isWrite] at hw
    · exfalso; exact hk Wc_f.key

omit hE in
theorem key_ne {a b : Loc} {m n : Nat} (h : m ≠ n) : (a, m) ≠ (b, n) := fun e => h (congrArg Prod.snd e)

/-- **The load of `Y` returns what the copy loaded from `X`, which is `X`'s initial bytes.** -/
theorem data (hc : c.exec.consistent = true) (hl₁ : (r₁ 0).length = 8)
    (hlw : (rw 0).length = 8) (hd : MBarrierWord.parityDone (waitWord (rw 0)) 0 = true)
    (hsr : (sr 0 0).length = 4) (hry : (ry 0).length = 4) :
    ry 0 = initBytes (program L x).mem (locX L, 4) := by
  have hW : WritersB c r₀ r₁ gr := fun j hw hk => writers_B c hE hw hk
  have mem : ∀ {v}, v ∈ evsOf L r₀ r₁ sr gr ws rw ry ro → v ∈ c.evs := fun h => (hE _).2 h
  have h₁ := arrive_reads c hW hc Ra_f rfl (mem (by simp [evsOf])) (mem (by simp [evsOf])) hl₁
  obtain ⟨iWc, iRw, hWc, hRw, hrf⟩ := wait_reads c hW hc (ws := ws) h₁ (mem (by simp [evsOf])) hlw hd
  obtain ⟨iFe, hFe⟩ := index c (v := Fe) (mem (by simp [evsOf]))
  obtain ⟨iWY, hWY⟩ := index c (v := WY L sr) (mem (by simp [evsOf]))
  obtain ⟨iRY, hRY⟩ := index c (v := RY L ws ry) (mem (by simp [evsOf]))
  obtain ⟨iRX, hRX⟩ := index c (v := RX L sr) (mem (by simp [evsOf]))
  -- Complete-tx releases, the wait acquires: they synchronize.
  have hs : c.exec.strongR iWc iRw = true := by
    rw [Fin.getElem_fin] at hWc hRw
    simp [Execution.strongR, Candidate.exec, toExec, hWc, hRw, Wc, Rw, aft, rmwW, waitEv, Req.event,
      Req.eventScope, completeReq, waitReq, mbarReq, Scope.contains, Execution.thr, Execution.placeOf,
      place_eq]
  have hsync : c.exec.sync iWc iRw = true :=
    sync_of_observation (by show c.evs[iWc].e.release = true; rw [hWc]; rfl)
      (by rw [writes_iff', hWc]; rfl) (by show c.evs[iRw].e.acquire = true; rw [hRw]; rfl)
      (by rw [reads_iff', hRw]; rfl) (by show c.evs[iRw].e.reduction = false; rw [hRw]; rfl)
      (tc_of (by simp [Execution.observation, Execution.strong, get_union, get_inter, hrf, hs])) hs
  have hcb : c.exec.causeBase iFe iRY = true :=
    causeBase_order (causeBase_sync (causeBase_of_order
      (ord_po c hFe hWc rfl (by show 2 < 4; decide))) hsync)
      (rtc_of_tc (ord_po c hRw hRY rfl (by show 4 + ws.length < 5 + ws.length; omega)))
  have hp : c.exec.proxyPreservedCauseBase iWY iRY = true :=
    ppcb_async_completion (by rw [mems_iff', hWY]; rfl)
      (by show (c.exec.ev iFe).proxyFenced = some (c.exec.ev iWY).proxy
          show (c.evs[iFe].e.proxyFenced) = some c.evs[iWY].e.proxy
          rw [hFe, hWY]; rfl)
      (by simp [Execution.sameBlock, Execution.placeOf, Candidate.exec, toExec, place_eq])
      (ord_po c hWY hFe rfl (by show 1 < 2; decide)) hcb
      (by show (c.evs[iRY].e.isMem && c.evs[iRY].e.proxy == .generic) = true; rw [hRY]; rfl)
      (by simp only [Execution.sameAlias, Rel.get_ofFn, Bool.and_eq_true, beq_iff_eq]
          exact ⟨⟨by rw [mems_iff', hWY]; rfl, by rw [mems_iff', hRY]; rfl⟩,
            (addr_iff c).2 (by rw [hWY, hRY]; rfl)⟩)
  have sees := read_sees_ppcb hc (by rw [writes_iff', hWY]; rfl) (by rw [reads_iff', hRY]; rfl)
    (by rw [mems_iff', hWY]; rfl) (by rw [mems_iff', hRY]; rfl) ((addr_iff c).2 (by rw [hWY, hRY]; rfl)) hp
  -- The load of `Y` read the copy's store.
  have hY : ry 0 = sr 0 0 := by
    rcases source c hc (r := iRY) (by rw [hRY]; rfl) with ⟨hn, -⟩ | ⟨w, hrf', hv, hw, hk⟩
    · obtain ⟨w', hw'⟩ := sees.1; rw [hn w'] at hw'; cases hw'
    · rw [hRY] at hk hv
      have hkY : c.evs[w].key = (locY L, 4) := hk
      have hne : c.evs[w].key ≠ (locB L, 8) := by
        rw [hkY]; exact key_ne (m := 4) (n := 8) (by decide)
      rcases writers_other c hE hw hne with ⟨h, -⟩ | ⟨-, h⟩
      · rw [h, WY_val hsr] at hv
        rw [← extract4 hry]; exact hv
      · rw [hkY] at h; simp [locY, locO] at h
  -- The copy's load read `X`'s initial bytes: nothing writes `X`.
  have hX : sr 0 0 = initBytes (program L x).mem (locX L, 4) := by
    rcases source c hc (r := iRX) (by rw [hRX]; rfl) with ⟨-, hv⟩ | ⟨w, -, -, hw, hk⟩
    · rw [hRX] at hv
      rw [← extract4 hsr]; exact hv
    · rw [hRX] at hk
      have hkX : c.evs[w].key = (locX L, 4) := hk
      have hne : c.evs[w].key ≠ (locB L, 8) := by
        rw [hkX]; exact key_ne (m := 4) (n := 8) (by decide)
      have := L.global
      rcases writers_other c hE hw hne with ⟨-, h⟩ | ⟨-, h⟩ <;>
        rw [hkX] at h <;> simp [locX, locY, locO] at h <;> omega
  rw [hY, hX]

/-- The output word's bytes in a final memory of a finished run: what the load of `Y` returned. -/
theorem final_O {m : Loc → UInt8} (hf : c.IsFinal m) (hry : (ry 0).length = 4) {i : Nat} (hi : i < 4) :
    m { space := .global, owner := 0, addr := L.O.toNat + i } = (ry 0).getD i 0 := by
  have hWOm : WO L ws ry ro ∈ c.evs := (hE _).2 (by simp [evsOf])
  have hcov : (WO L ws ry ro).covers { space := .global, owner := 0, addr := L.O.toNat + i } = true := by
    rw [NV.covers_iff]
    refine ⟨rfl, rfl, rfl, by simp [WO, oneEv, locO], by simp [WO, oneEv, locO]; omega⟩
  rcases hf { space := .global, owner := 0, addr := L.O.toNat + i } with ⟨j, hj, -, hm⟩ | ⟨hn, -⟩
  · obtain ⟨hw, hsp, -, hlo, -⟩ := NV.covers_iff.1 hj
    by_cases hkB : c.evs[j].key = (locB L, 8)
    · exfalso
      have hloc : c.evs[j].loc = locB L := congrArg Prod.fst hkB
      rw [hloc] at hsp; simp [locB] at hsp
    · rcases writers_other c hE hw hkB with ⟨h, -⟩ | ⟨h, -⟩
      · exfalso; rw [h] at hsp; simp [WY, aft, oneEv, locY] at hsp
      · rw [hm, h, WO_val hry]
        simp [WO, oneEv, locO]
  · obtain ⟨j, hj⟩ := index c hWOm
    rw [← hj, hn j] at hcov; cases hcov

end paths

end data

/-! ## No partial overlaps -/

section uniform
variable {L : Layout}

/-- The four locations the program accesses. -/
def Located (L : Layout) (v : Ev) : Prop :=
  v.key = (locB L, 8) ∨ v.key = (locY L, 4) ∨ v.key = (locX L, 4) ∨ v.key = (locO L, 4)

theorem key_parts {v : Ev} {l : Loc} {n : Nat} (h : v.key = (l, n)) : v.loc = l ∧ v.size = n := by
  simp only [Ev.key, Prod.mk.injEq] at h; exact h

theorem uniform_of (l : List Ev) (h : ∀ v ∈ l, v.e.isMem = true → Located L v) : Uniform l := by
  intro a ha b hb hov
  obtain ⟨hma, hmb, hsp, how, h₁, h₂⟩ := NV.overlaps_iff.1 hov
  have := L.global; have := L.shared
  rcases h a ha hma with ka | ka | ka | ka <;> rcases h b hb hmb with kb | kb | kb | kb <;>
    first
    | exact ka.trans kb.symm
    | (exfalso
       obtain ⟨la, sa⟩ := key_parts ka
       obtain ⟨lb, sb⟩ := key_parts kb
       rw [la, lb] at hsp
       rw [la, lb, sb] at h₁
       rw [la, lb, sa] at h₂
       simp [locB, locY, locX, locO] at hsp h₁ h₂ <;> omega)

theorem located_evsOf (r₀ r₁ : Nat → List UInt8) (sr gr : Nat → Nat → List UInt8)
    (ws : List (Nat → List UInt8)) (rw ry ro : Nat → List UInt8) :
    ∀ v ∈ evsOf L r₀ r₁ sr gr ws rw ry ro, v.e.isMem = true → Located L v := by
  intro v hv hm
  have : v ∈ [Ri L r₀, Wi L r₀, Ra L r₁, Wa L r₁] ∨ v ∈ waitEvs L 4 ws ∨
      v ∈ [Rw L ws rw, RY L ws ry, WO L ws ry ro, RX L sr, WY L sr, Fe, Rc L gr, Wc L gr] := by
    simpa only [evsOf, List.mem_append, or_assoc] using hv
  rcases this with h | h | h
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl <;> exact .inl rfl
  · obtain ⟨pos, w, rfl⟩ := wait_mem L ws h; exact .inl rfl
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · exact .inl rfl
    · exact .inr (.inl rfl)
    · exact .inr (.inr (.inr rfl))
    · exact .inr (.inr (.inl rfl))
    · exact .inr (.inl rfl)
    · exfalso; simp [Fe, aft, fenceEvent, Event.isMem, Event.isRead, Event.isWrite] at hm
    · exact .inl rfl
    · exact .inl rfl

end uniform

/-! ## Runs that fail -/

section failing
variable (L : Layout) (x : Fin 4 → UInt8) (r₀ r₁ : Nat → List UInt8) (sr gr : Nat → Nat → List UInt8)

def RaF : Ev := { Ra L r₁ with rmwRead := false }
def RcF : Ev := aft { rmwR L (completeReq L) 1 3 (gr 0 0) with rmwRead := false }

theorem RaF_f : Facts (L := L) (RaF L r₁) true 0 2 := ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
theorem RcF_f : Facts (L := L) (RcF L gr) true 1 3 := ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem failEvents_rmw {r : Req} (hp : r.pieces = [⟨false, 0, 8, true⟩, ⟨true, 0, 8, false⟩])
    (t s : Nat) (bs : List UInt8) :
    r.failEvents 0 t s (locB L) bs = [{ rmwR L r t s bs with rmwRead := false }] := by
  simp [Req.failEvents, rmw_events L hp, rmwR, rmwW, Req.event, Event.isWrite]

/-- The events of a run whose arrive is left undefined. -/
theorem events_arriveFail :
    (program L x).events [([.mem (initReq L) r₀, .fail (arriveReq L) r₁], failureOf (arriveReq L))] =
      [Ri L r₀, Wi L r₀, RaF L r₁] := by
  have hp : ∀ act ∈ [Act.mem (initReq L) r₀, .fail (arriveReq L) r₁], act.plain = true := by
    intro act h; simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl <;> rfl
  simp only [Program.events, program, List.zip_cons_cons, List.zip_nil_left, List.flatMap_cons,
    List.flatMap_nil, List.append_nil]
  rw [Program.asyncEvents_plain _ _ _ _ _ hp, List.append_nil]
  show (program L x).agentEvents theAgent (fun _ => 0) _ = _
  simp only [Program.agentEvents]
  rw [actEvents_mem L x (r := initReq L) (a := L.B) rfl]
  have hf : ∀ c, (program L x).actEvents theAgent c (.fail (arriveReq L) r₁) =
      (arriveReq L).failEvents 0 0 (c 0) (locB L) (r₁ 0) := fun c => by
    have hl := loc_one (r := arriveReq L) (a := L.B) rfl
    simp only [Program.actEvents, show (arriveReq L).addrs.size = 1 from rfl, List.range_one,
      List.flatMap_cons, List.flatMap_nil, List.append_nil, hl]
    rfl
  rw [hf, width_one (r := initReq L) (k := 2) rfl (by simp [Req.pieces, initReq, mbarReq])]
  simp only [ite_true, Nat.zero_add, List.append_nil]
  rw [show (initReq L).laneEvents 0 0 0 { space := (initReq L).space, owner := 0, addr := L.B.toNat }
      (r₀ 0) = [Ri L r₀, Wi L r₀] from rmw_events L (pieces_mbar L) 0 0 (r₀ 0),
    show (arriveReq L).failEvents 0 0 2 (locB L) (r₁ 0) = [RaF L r₁] from
      failEvents_rmw L (pieces_mbar L) 0 2 (r₁ 0)]
  rfl

/-- A copy's agent fails only at complete-tx: its load and store always have a value. -/
theorem copyFail_at {pre rest : List AAct} {rq : Req} {read : Nat → List UInt8}
    (h : copyActs (copy L) sr gr = pre ++ AAct.mem rq read :: rest) (hu : rq.unsupported read = true) :
    pre = [.mem (srcReq L) (sr 0), .mem (storeReq L (sr 0)) (fun _ => []), .proxyFence] ∧
      rq = completeReq L ∧ read = gr 0 := by
  rw [copyActs_eq] at h
  rcases pre with _ | ⟨a, _ | ⟨b, _ | ⟨d, _ | ⟨e, pre⟩⟩⟩⟩ <;>
    simp only [List.nil_append, List.cons_append, List.cons.injEq, AAct.mem.injEq] at h
  · obtain ⟨⟨rfl, rfl⟩, -⟩ := h
    simp [Req.unsupported, srcReq, asyncReq, MemOp.writes] at hu
  · obtain ⟨-, ⟨rfl, rfl⟩, -⟩ := h
    simp [Req.unsupported, storeReq, Move.store, dstReq, asyncReq, MemOp.writes] at hu
  · obtain ⟨-, -, h, -⟩ := h; cases h
  · obtain ⟨rfl, rfl, rfl, ⟨rfl, rfl⟩, -⟩ := h
    exact ⟨rfl, rfl, rfl⟩
  · obtain ⟨-, -, -, -, h⟩ := h
    simp at h

/-- The events of a run whose copy's complete-tx is left undefined. -/
theorem events_copyFail (e : End) :
    (program L x).events [([.mem (initReq L) r₀, .mem (arriveReq L) r₁, .async (.copy (copy L))
      [.mem (srcReq L) (sr 0), .mem (storeReq L (sr 0)) (fun _ => []), .proxyFence,
        .fail (completeReq L) (gr 0)]], e)] =
      [Ri L r₀, Wi L r₀, Ra L r₁, Wa L r₁, RX L sr, WY L sr, Fe, RcF L gr] := by
  simp only [Program.events, program, List.zip_cons_cons, List.zip_nil_left, List.flatMap_cons,
    List.flatMap_nil, List.append_nil]
  show (program L x).agentEvents theAgent (fun _ => 0) _ ++
    (program L x).asyncEvents theAgent (fun _ => 0) {} _ = _
  have w₁ : ∀ t read, Act.width theAgent t (.mem (initReq L) read) = if t = 0 then 2 else 0 :=
    fun t read => width_one (r := initReq L) rfl (by simp [Req.pieces, initReq, mbarReq]) read t
  have w₂ : ∀ t read, Act.width theAgent t (.mem (arriveReq L) read) = if t = 0 then 2 else 0 :=
    fun t read => width_one (r := arriveReq L) rfl (by simp [Req.pieces, arriveReq, mbarReq]) read t
  simp only [Program.agentEvents, Program.asyncEvents, w₁, w₂, List.append_nil]
  rw [actEvents_mem L x (r := initReq L) (a := L.B) rfl, actEvents_mem L x (r := arriveReq L) (a := L.B) rfl]
  simp only [ite_true, Nat.zero_add, show (program L x).actEvents theAgent _
    (.async (.copy (copy L)) _) = [] from rfl, List.append_nil]
  rw [show (initReq L).laneEvents 0 0 0 { space := (initReq L).space, owner := 0, addr := L.B.toNat }
      (r₀ 0) = [Ri L r₀, Wi L r₀] from rmw_events L (pieces_mbar L) 0 0 (r₀ 0),
    show (arriveReq L).laneEvents 0 0 2 { space := (arriveReq L).space, owner := 0, addr := L.B.toNat }
      (r₁ 0) = [Ra L r₁, Wa L r₁] from rmw_events L (pieces_mbar L) 0 2 (r₁ 0)]
  simp [Agent.aactEvents, Program.asyncThread, threads_eq, Req.active, srcReq, storeReq, theMove,
    dstReq, completeReq, mbarReq, asyncReq, Agent.loc, theAgent, locX, locY, locB, Move.store,
    Req.pieces, Req.granule, Req.strong, RX, WY, Fe, RcF, aft, oneEv,
    Req.laneEvents, rmwR, Req.failEvents, Req.event, Event.isWrite]
  done

end failing

/-! ## Every outcome -/

section main
variable {L : Layout} {x : Fin 4 → UInt8}

theorem init_x {i : Nat} (hi : i < 4) :
    (initBytes (program L x).mem (locX L, 4)).getD i 0 = x ⟨i, hi⟩ := by
  simp only [initBytes, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hi,
    Option.map_some, Option.getD_some, program, locX]
  split
  · congr 1; ext; simp
  · rename_i h; exact absurd ⟨trivial, trivial, by omega, by omega⟩ h

theorem arrive_ok {r₁ : Nat → List UInt8} (h : r₁ 0 = bytes64 wInit) :
    (arriveReq L).unsupported r₁ = false := by
  have := arrive_written L
  simp only [Req.written] at this
  simp [Req.unsupported, arriveReq, mbarReq, MemOp.writes, Req.active, h] at this ⊢
  simp [this]

theorem complete_ok {read : Nat → List UInt8} (h : read 0 = bytes64 wArr) :
    (completeReq L).unsupported read = false := by
  have := complete_written L
  simp only [Req.written] at this
  simp [Req.unsupported, completeReq, mbarReq, MemOp.writes, Req.active, h] at this ⊢
  simp [this]

theorem writersB_of {c : Candidate (program L x)} {r₀ r₁ : Nat → List UInt8}
    {gr : Nat → Nat → List UInt8} {l : List Ev} (hE : ∀ v, v ∈ c.evs ↔ v ∈ l)
    (hl : ∀ v ∈ l, v.e.isWrite = true → v.key = (locB L, 8) →
      v = Wi L r₀ ∨ v = Wa L r₁ ∨ v = Wc L gr) : WritersB c r₀ r₁ gr :=
  fun j hw hk => hl _ ((hE _).1 (List.getElem_mem j.isLt)) hw hk

/-- **The copy is correct**: for every layout and every initial `X`, every outcome the PTX
memory model allows is a completed run whose output word holds `X`'s initial bytes. No outcome
faults or leaves the model. -/
theorem copy_correct {o : Outcome} (h : Outcomes (program L x) o) :
    ∃ m, o = .done m ∧ ∀ i (hi : i < 4),
      m { space := .global, owner := 0, addr := L.O.toNat + i } = x ⟨i, hi⟩ := by
  obtain ⟨c, hv, hc, hy⟩ := h
  have hlen := c.length
  obtain ⟨⟨acts, e⟩, hrun⟩ : ∃ re, c.runs = [re] := by
    match hr : c.runs, hlen with
    | [re], _ => exact ⟨re, rfl⟩
  have hr : Run (step L) { pc := 0 } acts e :=
    c.runs_ok (theAgent, acts, e) (by simp [hrun, program, theAgent])
  have hperm : ∀ v, v ∈ c.evs ↔ v ∈ (program L x).events [(acts, e)] := fun v => by
    rw [← hrun]; exact c.perm.mem_iff
  rcases run_shape hr with rfl | ⟨r₀, r₁, rfl, rfl, hl, hu⟩ | ⟨r₀, r₁, agent, rfl, hl₁, ha⟩ |
    ⟨r₀, r₁, agent, res, ws, rw, ry, ro, rfl, hl₁, ha, rfl, hlw, hd, hly⟩
  · -- A run cut short needs another that faulted: there is none.
    exfalso
    obtain ⟨re, hre, h₁, h₂⟩ := hv.2 ⟨(acts, .stop), by simp [hrun], rfl⟩
    simp [hrun] at hre; subst hre; exact h₂ rfl
  · -- Arrive left undefined: it read init's word, which it is defined on.
    exfalso
    rw [events_arriveFail] at hperm
    have hW : WritersB c r₀ r₁ (fun _ _ => []) := writersB_of hperm (by
      intro v hv hw hk
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hv
      rcases hv with rfl | rfl | rfl
      · simp [Ri, rmwR, Req.event, Event.isWrite] at hw
      · exact .inl rfl
      · simp [RaF, Ra, rmwR, Req.event, Event.isWrite] at hw)
    have h₁ := arrive_reads c hW hc (RaF_f L r₁) rfl ((hperm _).2 (by simp)) ((hperm _).2 (by simp))
      (hl 0 rfl)
    rw [arrive_ok h₁] at hu; cases hu
  · -- The copy's agent failed: only at complete-tx, which read arrive's word.
    exfalso
    rcases AgentRun.copy_inv ha with h | ⟨sr, gr, pre, rq, read, rest, hca, -, hl, hu, rfl, -⟩
    · obtain ⟨_, _, _, _, h⟩ := h
      cases h
    obtain ⟨rfl, rfl, rfl⟩ := copyFail_at L sr gr hca hu
    simp only [List.cons_append, List.nil_append] at hperm
    rw [events_copyFail] at hperm
    have hW : WritersB c r₀ r₁ gr := writersB_of hperm (by
      intro v hv hw hk
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hv
      rcases hv with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
      · simp [Ri, rmwR, Req.event, Event.isWrite] at hw
      · exact .inl rfl
      · simp [Ra, rmwR, Req.event, Event.isWrite] at hw
      · exact .inr (.inl rfl)
      · simp [RX, aft, oneEv, Req.event, Event.isWrite] at hw
      · simp [WY, aft, oneEv, Ev.key, locY, locB] at hk
      · simp [Fe, aft, fenceEvent, Event.isWrite] at hw
      · simp [RcF, aft, rmwR, Req.event, Event.isWrite] at hw)
    have h₁ := arrive_reads c hW hc Ra_f rfl ((hperm _).2 (by simp)) ((hperm _).2 (by simp)) (hl₁ 0 rfl)
    have h₂ := complete_reads c hW hc (RcF_f L gr) (by simp [RcF, aft]) rfl ((hperm _).2 (by simp))
      ((hperm _).2 (by simp)) h₁ (hl 0 rfl)
    rw [complete_ok h₂] at hu; cases hu
  · -- A finished run.
    obtain ⟨sr, gr, rfl, hok, -⟩ : ∃ sr gr, agent = copyActs (copy L) sr gr ∧
        (∀ rq read, AAct.mem rq read ∈ copyActs (copy L) sr gr → ReqOk rq read) ∧ True := by
      rcases AgentRun.copy_inv ha with ⟨sr, gr, h₁, h₂, -⟩ | h
      · exact ⟨sr, gr, h₁, h₂, trivial⟩
      · obtain ⟨_, _, _, _, _, _, _, _, _, _, _, h⟩ := h
        cases h
    have hev : (program L x).events [(fullActs L r₀ r₁ (copyActs (copy L) sr gr) ws rw ry ro, .exit)] =
        evsOf L r₀ r₁ sr gr ws rw ry ro := by
      rw [← evs_eq L r₀ r₁ sr gr ws rw ry ro,
        ← agentEvents_full L x r₀ r₁ (copyActs (copy L) sr gr) ws rw ry ro,
        ← asyncEvents_full L x r₀ r₁ sr gr ws rw ry ro]
      simp only [Program.events, program, List.zip_cons_cons, List.zip_nil_left, List.flatMap_cons,
        List.flatMap_nil, List.append_nil]
      rfl
    have hE : ∀ v, v ∈ c.evs ↔ v ∈ evsOf L r₀ r₁ sr gr ws rw ry ro := fun v =>
      (hperm v).trans (Iff.of_eq (congrArg (v ∈ ·) hev))
    have len : ∀ {rq : Req} {n : Nat} {b : List UInt8}, b.length = rq.readLen →
        rq.readLen = n → b.length = n := fun h h' => h.trans h'
    have hsr : (sr 0 0).length = 4 := len ((hok (srcReq L) (sr 0) (by rw [copyActs_eq]; simp)).1 0 rfl)
      (by simp [Req.readLen, Req.pieces, Req.granule, Req.strong, srcReq, asyncReq])
    have hry : (ry 0).length = 4 := len (hly 0 rfl)
      (by simp [Req.readLen, Req.pieces, Req.granule, Req.strong, ldY])
    have hy₀ := data c hE hc (hl₁ 0 rfl)
      (len (hlw 0 rfl) (by simp (config := { decide := true }) [Req.readLen, Req.pieces, Req.granule, Req.strong, waitReq,
        Req.elemBytes]))
      (by rw [← resp_wait]; exact hd) hsr hry
    cases o with
    | done m =>
      obtain ⟨-, -, hf⟩ := hy
      exact ⟨m, rfl, fun i hi => by rw [final_O c hE hf hry hi, hy₀, init_x hi]⟩
    | fault msg =>
      obtain ⟨re, hre, h⟩ := hy
      simp [hrun] at hre; subst hre; cases h
    | unsupported msg =>
      rcases hy with ⟨re, hre, h⟩ | ⟨hn, -⟩
      · simp [hrun] at hre; subst hre; cases h
      · exact absurd (uniform_of c.evs fun v hv => located_evsOf r₀ r₁ sr gr ws rw ry ro v ((hE v).1 hv)) hn

end main

end Examples.AsyncCopy
