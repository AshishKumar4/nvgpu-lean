import NV.Memory.Events

/-!
# The axiomatic program semantics

A program's outcomes are the final memories of the candidate executions the PTX memory model
accepts (PTX ISA §8). Three steps:

1. **Runs.** An agent (a thread, or a warp read lane by lane) runs its step function; a memory
   request's reads may return *any* bytes, and a barrier arrival may join *any* instance. `Run`
   records what it did: each request with the bytes its lanes read, each synchronizing operation
   with the barrier instance it joined. Reads block: the request is built from the state after
   issue and the next step sees the response (the blocking reading of `Effect.mem`).
2. **Candidates.** The events of every agent's run (`Req.laneEvents`, `SyncOp.events`), in any
   arrangement, with a chosen reads-from, coherence and Fence-SC order, and barrier instances that
   each gather at most the expected number of threads, exactly that many when anyone waits on them,
   and each thread at most once.
3. **Outcomes.** A candidate whose execution is `consistent` yields its final memory: each byte is
   the value of a coherence-last write to it, or its initial value if nothing writes it. A
   candidate in which an agent faulted yields the fault, whatever the others did; one that leaves
   the model (an operation the library does not model yet, or partially overlapping accesses,
   `Uniform`) yields `unsupported`. A proof that a program is correct shows every outcome is a
   `done` memory with the property.

Dependencies are not tracked through the step function, so the No-Thin-Air axiom (§8.10.4) sees
only read-modify-write dependencies: `Outcomes` is a superset of the ISA's, which keeps every
theorem about all outcomes true of the ISA's. Termination is not part of an outcome: a program
whose agents cannot all finish has no `done` outcome from such runs.
-/

namespace NV.Formal

open NV.PTX (Place Execution)

/-- One agent: a PTX thread (`lanes = 1`) or a warp. Lane `l` is memory-model thread `tid + l`. -/
structure Agent (σ : Type) where
  init : σ
  tid : Nat
  lanes : Nat := 1
  place : Place
  /-- The warp a thread agent belongs to, and its lane there: who joins its warp collectives. -/
  warp : Nat := tid / 32
  lane : Nat := tid % 32

structure Program (σ : Type) where
  step : σ → Effect σ
  agents : List (Agent σ)
  /-- Initial memory, per byte. -/
  mem : Loc → UInt8

/-- What an asynchronous agent did, in its program order (`Async.Copy.program`). -/
inductive AAct where
  /-- A request; `read l` is what lane `l`'s reads returned. -/
  | mem (r : Req) (read : Nat → List UInt8)
  /-- A request whose reads returned `read` and that wrote nothing: what they returned leaves its
  result undefined (`failureOf`). -/
  | fail (r : Req) (read : Nat → List UInt8)
  /-- The implicit generic-async proxy fence of completion (PTX ISA §9.7.9.25). -/
  | proxyFence

/-- How a run ended. `stop` cuts a run short; it is allowed only beside a fault. -/
inductive End where
  | exit
  | fault (msg : String)
  | unsupported (msg : String)
  | stop
  deriving DecidableEq, Repr

/-- What an agent did. -/
inductive Act where
  /-- A memory request; `read l` is what lane `l`'s reads returned. -/
  | mem (r : Req) (read : Nat → List UInt8)
  /-- A request whose reads returned `read` and that wrote nothing: what they returned leaves its
  result undefined (`failureOf`). Its reads are events, so a run ends this way only where some
  consistent execution lets them return such bytes. -/
  | fail (r : Req) (read : Nat → List UInt8)
  /-- A synchronizing operation; a barrier arrival or wait names the instance it joined. -/
  | sync (op : SyncOp) (inst : Nat)
  /-- An asynchronous operation and what its agent did. -/
  | async (op : Async.Op) (agent : List AAct)
  /-- A collective: this thread's contribution, the instance it joined, and the answer it got, or
  how the run ends there; `Program.CollectsOk` relates the answers of an instance's members. -/
  | collect (op : Collective) (mask : UInt32) (contrib : Array UInt32) (inst : Nat)
      (answer : Except End Resp)


/-- How a run ends at a request with no value for its reads: an mbarrier operation the ISA leaves
undefined is a fault; an atomic the library does not define yet is outside the model. -/
def failureOf (r : Req) : End :=
  match r.op with
  | .mbar _ => .fault "an mbarrier operation the ISA leaves undefined"
  | _ => .unsupported "an atomic operation on a type the library does not define yet"

/-- The reads of request `r` returned `read`: the request's bytes for every active lane, and a
value the library defines for every write. -/
def ReqOk (r : Req) (read : Nat → List UInt8) : Prop :=
  (∀ l, r.active l → (read l).length = r.readLen) ∧ r.unsupported read = false

/-- A copy's agent: per move its load (which returned `srcRead i`) then its stores of what it
loaded; then, when it signals, the implicit proxy fence and its signals (which returned
`sigRead i`). The moves are performed in order: a copy whose moves overlap is ordered here. -/
def copyActs (c : Async.Copy) (srcRead sigRead : Nat → Nat → List UInt8) : List AAct :=
  (c.moves.toList.zipIdx.flatMap fun (mv, i) =>
    AAct.mem mv.src (srcRead i) ::
      mv.dsts.toList.map fun d => AAct.mem (mv.store d (mv.src.resp (srcRead i))) fun _ => []) ++
  if c.signals.isEmpty then [] else
    .proxyFence :: c.signals.toList.zipIdx.map fun (r, i) => AAct.mem r (sigRead i)

/-- A run of an asynchronous operation's agent: what it did, and its result for the issuer (an
MMA's accumulator words, nothing for a copy) or how it ended (a fault, or outside the model). Its
reads may return any bytes, as an agent's do in `Run`. -/
inductive AgentRun : Async.Op → List AAct → Except End Resp → Prop
  | copy {c sr gr} : (∀ r read, AAct.mem r read ∈ copyActs c sr gr → ReqOk r read) →
      AgentRun (.copy c) (copyActs c sr gr) (.ok #[])
  | copyFail {c sr gr pre r read rest} : copyActs c sr gr = pre ++ AAct.mem r read :: rest →
      (∀ r' read', AAct.mem r' read' ∈ pre → ReqOk r' read') →
      (∀ l, r.active l → (read l).length = r.readLen) → r.unsupported read = true →
      AgentRun (.copy c) (pre ++ [.fail r read]) (.error (failureOf r))
  | tensor {t rd c acts res} : ReqOk t.desc rd → t.build (t.desc.resp rd) = .ok c →
      AgentRun (.copy c) acts res → AgentRun (.tensor t) (.mem t.desc rd :: acts) res
  | tensorFail {t rd e} : ReqOk t.desc rd → t.build (t.desc.resp rd) = .error e →
      AgentRun (.tensor t) [.mem t.desc rd] (.error (.fault e))
  | mma {mm} {rd : Nat → Nat → List UInt8} : (∀ i (h : i < mm.loads.size), ReqOk mm.loads[i] (rd i)) →
      AgentRun (.mma mm) (mm.loads.toList.zipIdx.map fun (r, i) => .mem r (rd i))
        (.ok (mm.result ((mm.loads.toList.zipIdx.map fun (r, i) => r.resp (rd i)).foldl (· ++ ·) #[])
          mm.c0))
  | prefetch {ls} : AgentRun (.prefetch ls) [] (.ok #[])
  | fault {msg} : AgentRun (.fault msg) [] (.error (.fault msg))

section
variable {σ : Type} (step : σ → Effect σ)

inductive Run : σ → List Act → End → Prop
  | exit {s} : step s = .exit → Run s [] .exit
  | fault {s msg} : step s = .fault msg → Run s [] (.fault msg)
  | stop {s} : Run s [] .stop
  | next {s s' acts e} : step s = .next s' → Run s' acts e → Run s acts e
  | reg {s s' wb acts e} : step s = .reg s' wb → Run (wb s') acts e → Run s acts e
  | mem {s s' rf wb read acts e} : step s = .mem rf s' wb →
      (∀ l, (rf s').active l → (read l).length = (rf s').readLen) → (rf s').unsupported read = false →
      Run (wb ((rf s').resp read) s') acts e → Run s (.mem (rf s') read :: acts) e
  | memUnsupported {s s' rf wb read} : step s = .mem rf s' wb →
      (∀ l, (rf s').active l → (read l).length = (rf s').readLen) → (rf s').unsupported read = true →
      Run s [.fail (rf s') read] (failureOf (rf s'))
  | sync {s s' op inst acts e} : step s = .sync op s' → Run s' acts e → Run s (.sync op inst :: acts) e
  /-- A collective joins some instance and gets some answer, written back; which instance and
  answer is the candidate's choice (`CollectsOk`), as a read's bytes are. -/
  | collect {s op mask contrib s' wb inst resp acts e} : step s = .collect op mask contrib s' wb →
      Run (wb resp s') acts e → Run s (.collect op mask contrib inst (.ok resp) :: acts) e
  | collectFail {s op mask contrib s' wb inst e} : step s = .collect op mask contrib s' wb →
      Run s [.collect op mask contrib inst (.error e)] e
  /-- An asynchronous operation runs on its own agent; the blocking reading continues with its
  result written back. -/
  | async {s op s' wb agent res acts e} : step s = .async op s' wb → AgentRun (op s') agent (.ok res) →
      Run (wb res s') acts e → Run s (.async (op s') agent :: acts) e
  | asyncFail {s op s' wb agent e} : step s = .async op s' wb → AgentRun (op s') agent (.error e) →
      Run s [.async (op s') agent] e

namespace Run

variable {step} {s : σ} {acts : List Act} {e : End}

/-- A run of a state that exits did nothing more. -/
theorem of_exit (hs : step s = .exit) (h : Run step s acts e) :
    acts = [] ∧ (e = .exit ∨ e = .stop) := by
  cases h <;> simp_all

theorem of_next {s' : σ} (hs : step s = .next s') (h : Run step s acts e) :
    (acts = [] ∧ e = .stop) ∨ Run step s' acts e := by
  cases h <;> simp_all

/-- A run of a state that issues a request: cut short, out of the model, or the request answered
with some bytes and the run from the resulting state. -/
theorem of_mem {r : Req} {s' : σ} {wb : Resp → σ → σ} (hs : step s = .mem (fun _ => r) s' wb)
    (h : Run step s acts e) :
    (acts = [] ∧ e = .stop) ∨ (∃ read, acts = [.fail r read] ∧ e = failureOf r ∧
      (∀ l, r.active l → (read l).length = r.readLen) ∧ r.unsupported read = true) ∨
      ∃ read rest, acts = .mem r read :: rest ∧ (∀ l, r.active l → (read l).length = r.readLen) ∧
        r.unsupported read = false ∧ Run step (wb (r.resp read) s') rest e := by
  cases h with
  | stop => exact .inl ⟨rfl, rfl⟩
  | exit h₁ => rw [hs] at h₁; cases h₁
  | fault h₁ => rw [hs] at h₁; cases h₁
  | next h₁ => rw [hs] at h₁; cases h₁
  | reg h₁ => rw [hs] at h₁; cases h₁
  | mem h₁ h₂ h₃ h₄ =>
    rw [hs] at h₁; cases h₁
    exact .inr (.inr ⟨_, _, rfl, h₂, h₃, h₄⟩)
  | memUnsupported h₁ h₂ h₃ =>
    rw [hs] at h₁; cases h₁
    exact .inr (.inl ⟨_, rfl, rfl, h₂, h₃⟩)
  | sync h₁ => rw [hs] at h₁; cases h₁
  | collect h₁ => rw [hs] at h₁; cases h₁
  | collectFail h₁ => rw [hs] at h₁; cases h₁
  | async h₁ => rw [hs] at h₁; cases h₁
  | asyncFail h₁ => rw [hs] at h₁; cases h₁

/-- A run of a state that synchronizes: cut short, or the operation and the run from the
continuation. -/
theorem of_sync {op : SyncOp} {s' : σ} (hs : step s = .sync op s') (h : Run step s acts e) :
    (acts = [] ∧ e = .stop) ∨ ∃ inst rest, acts = .sync op inst :: rest ∧ Run step s' rest e := by
  cases h with
  | stop => exact .inl ⟨rfl, rfl⟩
  | sync h₁ h₂ => rw [hs] at h₁; cases h₁; exact .inr ⟨_, _, rfl, h₂⟩
  | _ => simp_all

/-- A run of a state that issues an asynchronous operation: cut short, its agent failing, or
its agent running and the run continuing from the written-back state. -/
theorem of_async {op : Async.Op} {s' : σ} {wb : Resp → σ → σ}
    (hs : step s = .async (fun _ => op) s' wb) (h : Run step s acts e) :
    (acts = [] ∧ e = .stop) ∨ (∃ agent, acts = [.async op agent] ∧ AgentRun op agent (.error e)) ∨
      ∃ agent res rest, acts = .async op agent :: rest ∧ AgentRun op agent (.ok res) ∧
        Run step (wb res s') rest e := by
  cases h with
  | stop => exact .inl ⟨rfl, rfl⟩
  | exit h₁ => rw [hs] at h₁; cases h₁
  | fault h₁ => rw [hs] at h₁; cases h₁
  | next h₁ => rw [hs] at h₁; cases h₁
  | reg h₁ => rw [hs] at h₁; cases h₁
  | mem h₁ => rw [hs] at h₁; cases h₁
  | memUnsupported h₁ => rw [hs] at h₁; cases h₁
  | sync h₁ => rw [hs] at h₁; cases h₁
  | collect h₁ => rw [hs] at h₁; cases h₁
  | collectFail h₁ => rw [hs] at h₁; cases h₁
  | async h₁ h₂ h₃ =>
    rw [hs] at h₁; cases h₁
    exact .inr (.inr ⟨_, _, _, rfl, h₂, h₃⟩)
  | asyncFail h₁ h₂ =>
    rw [hs] at h₁; cases h₁
    exact .inr (.inl ⟨_, rfl, h₂⟩)

end Run

/-- What a copy's agent did: its whole program, every request answered, or its program up to a
request whose reads left the result undefined. -/
theorem AgentRun.copy_inv {c : Async.Copy} {agent : List AAct} {r : Except End Resp}
    (h : AgentRun (.copy c) agent r) :
    (∃ sr gr, agent = copyActs c sr gr ∧ (∀ rq read, AAct.mem rq read ∈ copyActs c sr gr → ReqOk rq read) ∧
      r = .ok #[]) ∨
    (∃ sr gr pre rq read rest, copyActs c sr gr = pre ++ AAct.mem rq read :: rest ∧
      (∀ r' read', AAct.mem r' read' ∈ pre → ReqOk r' read') ∧
      (∀ l, rq.active l → (read l).length = rq.readLen) ∧ rq.unsupported read = true ∧
      agent = pre ++ [.fail rq read] ∧ r = .error (failureOf rq)) := by
  cases h with
  | copy h₁ => exact .inl ⟨_, _, rfl, h₁, rfl⟩
  | copyFail h₁ h₂ h₃ h₄ => exact .inr ⟨_, _, _, _, _, _, h₁, h₂, h₃, h₄, rfl, rfl⟩

end

/-! ## Events of runs -/

/-- Where location `l` names a byte range of the issuing agent's state space instance. -/
def Agent.loc {σ : Type} (a : Agent σ) (r : Req) (lane : Nat) : Option Loc :=
  (r.addrs[lane]?).join.map fun x =>
    { space := r.space, addr := x.toNat,
      owner := match r.space with
        | .shared => a.place.cta
        | .«local» => a.tid + lane
        | _ => 0 }

/-- Threads below this belong to agents; above it, to asynchronous operations' agents. -/
def Program.threads {σ : Type} (P : Program σ) : Nat :=
  P.agents.foldl (fun n a => max n (a.tid + a.lanes)) 0

/-- The memory-model thread of agent `a`'s `k`th asynchronous operation. -/
def Program.asyncThread {σ : Type} (P : Program σ) (a : Agent σ) (k : Nat) : Nat :=
  P.threads * (k + 1) + a.tid

def Program.agentPlace {σ : Type} (P : Program σ) (t : Nat) : Option Place :=
  (P.agents.find? fun a => a.tid ≤ t && t < a.tid + a.lanes).map (·.place)

/-- The memory-model place of a thread: its agent's, or for an asynchronous operation's thread,
its issuer's (§8.9.1.1: the async agent is in the issuer's CTA). -/
def Program.place {σ : Type} (P : Program σ) (t : Nat) : Place :=
  (P.agentPlace t).getD ((P.agentPlace (t % P.threads)).getD default)

/-- The threads a barrier operation of thread `t` expects: its count, else every thread of the
CTA (`bar`) or cluster (`barrier.cluster`). -/
def Program.expected {σ : Type} (P : Program σ) (t : Nat) : SyncOp → Nat
  | .bar _ (some n) _ => n
  | .bar _ none _ =>
    ((P.agents.filter (·.place.cta == (P.place t).cta)).map (·.lanes)).sum
  | _ => ((P.agents.filter (·.place.cluster == (P.place t).cluster)).map (·.lanes)).sum

/-- The events of a lane of a request that wrote nothing: its read pieces, none of them joined to
a write. -/
def _root_.NV.Req.failEvents (r : Req) (lane t pos : Nat) (loc : Loc) (read : List UInt8) : List Ev :=
  (r.laneEvents lane t pos loc read).filterMap fun v =>
    if v.e.isWrite then none else some { v with rmwRead := false }

/-- The events of one action of agent `a`, thread `t`'s first at position `ctr t`. -/
def Program.actEvents {σ : Type} (P : Program σ) (a : Agent σ) (ctr : Nat → Nat) : Act → List Ev
  | .mem r read => (List.range r.addrs.size).flatMap fun l =>
      match a.loc r l with
      | some loc => r.laneEvents l (a.tid + l) (ctr (a.tid + l)) loc (read l)
      | none => []
  | .sync op inst => (List.range a.lanes).flatMap fun l =>
      op.events (a.tid + l) (ctr (a.tid + l)) (op.barObj (P.place (a.tid + l))) inst
        (P.expected (a.tid + l) op)
  | .fail r read => (List.range r.addrs.size).flatMap fun l =>
      match a.loc r l with
      | some loc => r.failEvents l (a.tid + l) (ctr (a.tid + l)) loc (read l)
      | none => []
  | .async .. | .collect .. => []

/-- How many program-order positions an action takes in thread `t`. -/
def Act.width {σ : Type} (a : Agent σ) (t : Nat) : Act → Nat
  | .mem r _ | .fail r _ => if a.tid ≤ t ∧ r.active (t - a.tid) then r.pieces.length else 0
  | .sync op _ => if a.tid ≤ t ∧ t < a.tid + a.lanes then
      (match op with | .bar _ _ false => 2 | _ => 1) else 0
  | .async .. | .collect .. => 0

/-- The events of an asynchronous agent's actions on thread `t`, from position `pos`, each after
`issue` (its issuer's events before the operation). -/
def Agent.aactEvents {σ : Type} (a : Agent σ) (t : Nat) (issue : List (Nat × Nat)) :
    Nat → List AAct → List Ev
  | _, [] => []
  | pos, .proxyFence :: rest =>
    { fenceEvent (.proxyFence .async) t pos with after := issue } :: a.aactEvents t issue (pos + 1) rest
  | pos, .mem r read :: rest =>
    let lanes := (List.range r.addrs.size).filter (r.active ·)
    let evs := lanes.zipIdx.flatMap fun (l, i) => match a.loc r l with
      | some loc => r.laneEvents l t (pos + i * r.pieces.length) loc (read l)
      | none => []
    evs.map ({ · with after := issue }) ++
      a.aactEvents t issue (pos + lanes.length * r.pieces.length) rest
  | pos, .fail r read :: rest =>
    let lanes := (List.range r.addrs.size).filter (r.active ·)
    let evs := lanes.zipIdx.flatMap fun (l, i) => match a.loc r l with
      | some loc => r.failEvents l t (pos + i * r.pieces.length) loc (read l)
      | none => []
    evs.map ({ · with after := issue }) ++
      a.aactEvents t issue (pos + lanes.length * r.pieces.length) rest

/-- The positions an asynchronous agent's actions take. -/
def aactWidth : List AAct → Nat
  | [] => 0
  | .proxyFence :: rest => 1 + aactWidth rest
  | .mem r _ :: rest | .fail r _ :: rest =>
    ((List.range r.addrs.size).filter (r.active ·)).length * r.pieces.length +
      aactWidth rest

/-- What the events of asynchronous operations need from the run so far: how many operations the
agent issued, its operations not yet committed (group, thread, positions), and its committed
groups in order. -/
structure AsyncState where
  k : Nat := 0
  open_ : List (Option Async.Group × Nat × Nat) := []
  committed : List (Async.Group × List (Nat × Nat)) := []

/-- The events of agent `a`'s asynchronous operations and group waits: each operation's agent's
events, after the issuer's events before it; each `wait_group`'s event, at the position the wait
takes in the issuer, after the waited groups' events (none for `.read`, which waits for source
reads only: ordering nothing more is the sound reading). -/
def Program.asyncEvents {σ : Type} (P : Program σ) (a : Agent σ) :
    (Nat → Nat) → AsyncState → List Act → List Ev
  | _, _, [] => []
  | ctr, st, act :: acts =>
    let ctr' := fun t => ctr t + act.width a t
    let issue := (List.range a.lanes).map fun l => (a.tid + l, ctr (a.tid + l))
    match act with
    | .async op agent =>
      let t := P.asyncThread a st.k
      let st' := { st with k := st.k + 1, open_ := st.open_ ++ [(op.group, t, aactWidth agent)] }
      -- A warpgroup MMA with `commit` closes the open wgmma operations into a group.
      let st' := match op with
        | .mma mm => if mm.commit then
            { st' with open_ := st'.open_.filter (·.1 != some .wgmma),
                       committed := st'.committed ++
                         [(.wgmma, (st'.open_.filter (·.1 == some .wgmma)).map (·.2))] }
          else st'
        | _ => st'
      a.aactEvents t issue 0 agent ++ P.asyncEvents a ctr' st' acts
    | .sync (.asyncCommit g) _ =>
      P.asyncEvents a ctr' { st with
        open_ := st.open_.filter (·.1 != some g),
        committed := st.committed ++ [(g, (st.open_.filter (·.1 == some g)).map (·.2))] } acts
    | .sync (.asyncWait g n rd) _ =>
      let gs := (st.committed.filter (·.1 == g)).map (·.2)
      let waited := if rd then [] else (gs.take (gs.length - n)).flatten
      ((List.range a.lanes).map fun l =>
        ({ e := { thread := a.tid + l, kind := .asyncWait }, seq := ctr (a.tid + l), after := waited } : Ev)) ++
        P.asyncEvents a ctr' st acts
    | _ => P.asyncEvents a ctr' st acts

/-- The events of an agent's actions, numbering thread `t`'s positions from `ctr t`. -/
def Program.agentEvents {σ : Type} (P : Program σ) (a : Agent σ) : (Nat → Nat) → List Act → List Ev
  | _, [] => []
  | ctr, act :: acts =>
    P.actEvents a ctr act ++ P.agentEvents a (fun t => ctr t + act.width a t) acts

/-- The positions acts take on thread `t`. -/
def Agent.span {σ : Type} (a : Agent σ) (t : Nat) (acts : List Act) : Nat := (acts.map (·.width a t)).sum

/-- The counter after acts `acts`. -/
def Agent.after {σ : Type} (a : Agent σ) (ctr : Nat → Nat) (acts : List Act) : Nat → Nat :=
  fun t => ctr t + a.span t acts

/-- An action with no asynchronous part: a request, or synchronization other than group commits
and waits. -/
def Act.plain : Act → Bool
  | .mem .. | .fail .. => true
  | .sync (.asyncCommit _) _ | .sync (.asyncWait ..) _ => false
  | .sync .. => true
  | .async .. | .collect .. => false

theorem Program.asyncEvents_plain {σ : Type} (P : Program σ) (a : Agent σ) :
    ∀ (ctr : Nat → Nat) (st : AsyncState) (acts : List Act), (∀ act ∈ acts, act.plain = true) →
      P.asyncEvents a ctr st acts = []
  | _, _, [], _ => rfl
  | ctr, st, act :: acts, h => by
    have hrest := P.asyncEvents_plain a (fun t => ctr t + act.width a t) st acts
      fun x hx => h x (List.mem_cons_of_mem _ hx)
    have ha := h act (List.mem_cons_self ..)
    unfold Program.asyncEvents
    cases act with
    | mem r read => exact hrest
    | fail r read => exact hrest
    | sync op k => cases op <;> first | exact hrest | simp [Act.plain] at ha
    | async op ag => simp [Act.plain] at ha
    | collect => simp [Act.plain] at ha

theorem Program.place_of {σ : Type} {P : Program σ} {t : Nat} {p : Place}
    (h : P.agentPlace t = some p) : P.place t = p := by
  simp [Program.place, h]

/-- A property of every agent's place and of the default place holds of every thread's place. -/
theorem Program.place_all {σ : Type} {P : Program σ} {f : Place → Prop}
    (h : ∀ a ∈ P.agents, f a.place) (hd : f default) (t : Nat) : f (P.place t) := by
  have g : ∀ u, ∀ p, P.agentPlace u = some p → f p := fun u p hu => by
    simp only [Program.agentPlace, Option.map_eq_some_iff] at hu
    obtain ⟨a, ha, rfl⟩ := hu
    exact h a (List.mem_of_find?_eq_some ha)
  unfold Program.place
  cases h₁ : P.agentPlace t with
  | some p => exact g _ _ h₁
  | none =>
    cases h₂ : P.agentPlace (t % P.threads) with
    | some p => exact g _ _ h₂
    | none => exact hd

/-! ## Candidates and outcomes -/

/-- Every barrier instance gathers at most as many arrivals as it expects, and exactly that many if
anyone waits on it. -/
def BarriersOk (evs : List Ev) : Prop :=
  ∀ v ∈ evs, v.barKey ≠ none →
    let arrivals := evs.filter fun a => a.arrives && a.barKey == v.barKey
    arrivals.length ≤ v.count ∧ (v.waits = true → arrivals.length = v.count)

/-! ## Collectives

A collective's instance gathers the acts that joined it. Its members share a warp (a CTA for
`bar.red`); a warp collective is joined by every thread of the warp in its member mask, `bar.red`
by its thread count or the whole CTA; each member's answer is `Collective.answer` on the members'
contributions, indexed by lane (by thread order for `bar.red`), and an error there (behaviour the
ISA leaves undefined) is a fault. Which threads `activemask` finds together is the hardware's
choice: any members are allowed. Collectives order no memory accesses here, `bar.warp.sync` and
`bar.red` included (the ISA orders them): `Outcomes` is a superset of the ISA's there, as for
dependencies. Collectives are defined for thread agents (`lanes = 1`); a warp agent's leaves the
model (`unsupported`). -/

/-- A collective act of a run, with its agent. -/
structure Joined (σ : Type) where
  agent : Agent σ
  op : Collective
  mask : UInt32
  contrib : Array UInt32
  inst : Nat
  answer : Except End Resp

def Program.joined {σ : Type} (P : Program σ) (runs : List (List Act × End)) : List (Joined σ) :=
  (P.agents.zip runs).flatMap fun (a, acts, _) => acts.filterMap fun
    | .collect op mask contrib inst ans => some ⟨a, op, mask, contrib, inst, ans⟩
    | _ => none

/-- A member's index among an instance's members: its lane, or for `bar.red` its rank by thread. -/
def Joined.index {σ : Type} (grp : List (Joined σ)) (j : Joined σ) : Nat :=
  match j.op with
  | .barRed .. => (grp.filter (·.agent.tid < j.agent.tid)).length
  | _ => j.agent.lane

/-- The members' contributions, by index. -/
def Joined.contribs {σ : Type} (grp : List (Joined σ)) (op : Collective) : Array (Option (Array UInt32)) :=
  let width := match op with | .barRed .. => grp.length | _ => 32
  (List.range width).toArray.map fun i => (grp.find? (Joined.index grp · == i)).map (·.contrib)

/-- The instances' members agree and get their answers (see above). -/
def Program.CollectsOk {σ : Type} (P : Program σ) (runs : List (List Act × End)) : Prop :=
  let js := P.joined runs
  ∀ j ∈ js,
    let grp := js.filter (·.inst == j.inst)
    (∀ k ∈ grp, k.op = j.op ∧ k.mask = j.mask ∧ match j.op with
      | .barRed .. => P.place k.agent.tid = P.place j.agent.tid
      | _ => k.agent.warp = j.agent.warp) ∧
    (grp.map (·.agent.tid)).Nodup ∧
    (match j.op with
      | .activemask => True
      | .barRed _ (some n) _ => grp.length = n
      | .barRed _ none _ => grp.length = (P.agents.filter (P.place ·.tid == P.place j.agent.tid)).length
      | _ => ∀ a ∈ P.agents, a.warp = j.agent.warp → Collective.bit j.mask a.lane = true →
          ∃ k ∈ grp, k.agent.tid = a.tid) ∧
    j.answer = if j.agent.lanes ≠ 1 then .error (.unsupported "a collective of a warp agent") else
      match Collective.answer j.op j.mask (Joined.contribs grp j.op) with
      | .ok answers => .ok answers[Joined.index grp j]!
      | .error msg => .error (.fault msg)

/-- An instance's answer is an answer, a fault, or leaving the model: never an exit. -/
theorem Program.CollectsOk.not_exit {σ : Type} {P : Program σ} {runs : List (List Act × End)}
    (h : P.CollectsOk runs) {j : Joined σ} (hj : j ∈ P.joined runs) : j.answer ≠ .error .exit := by
  obtain ⟨-, -, -, ha⟩ := h j hj
  rw [ha]
  split
  · simp
  · split <;> simp

theorem Program.collectsOk_nil {σ : Type} (P : Program σ) {runs : List (List Act × End)}
    (h : P.joined runs = []) : P.CollectsOk runs := by
  intro j hj; rw [h] at hj; cases hj

/-! ## Dependencies

No-Thin-Air (§8.10.4) forbids cycles of reads-from and dependencies. Here a dependency is
*semantic*: event `e` of an agent's act `jb` depends on read `d` of an earlier act `ja` of the
same thread when some run of the agent, making the same choices except act `ja`'s read bytes,
does not produce `e` at act `jb` (its address, data or presence changed). An instruction whose
request does not read a value cannot change with it, so each such dependency is one of the ISA's:
`Outcomes` stays a superset of the ISA's outcomes, now with the dependencies the step functions
make. Dependencies through asynchronous agents and across a warp agent's lanes are left out. -/

/-- Two acts made the same choices: the same read bytes, instance, answer or agent run. -/
def Act.sameChoice : Act → Act → Prop
  | .mem _ rd, .mem _ rd' | .fail _ rd, .fail _ rd' => rd = rd'
  | .sync _ i, .sync _ i' => i = i'
  | .collect _ _ _ i a, .collect _ _ _ i' a' => i = i' ∧ a = a'
  | .async _ ag, .async _ ag' => ag = ag'
  | _, _ => False

/-- Event `e` depends on read `d` in run `acts` of agent `a`. -/
def Program.DepIn {σ : Type} (P : Program σ) (a : Agent σ) (acts : List Act) (d e : Ev) : Prop :=
  ∃ ja jb, ∃ (hab : ja < jb) (hb : jb < acts.length),
    d ∈ P.actEvents a (a.after (fun _ => 0) (acts.take ja)) (acts[ja]'(Nat.lt_trans hab hb)) ∧
    d.e.isRead = true ∧
    e ∈ P.actEvents a (a.after (fun _ => 0) (acts.take jb)) acts[jb] ∧ d.e.thread = e.e.thread ∧
    ∃ acts' e', Run P.step a.init acts' e' ∧
      (∀ k (hk : k < jb), k ≠ ja → ∃ h : k < acts'.length,
        Act.sameChoice (acts[k]'(Nat.lt_trans hk hb)) acts'[k]) ∧
      ∀ h : jb < acts'.length, e ∉ P.actEvents a (a.after (fun _ => 0) (acts.take jb)) acts'[jb]

/-- The dependencies of a candidate's runs. -/
def Program.SemDep {σ : Type} (P : Program σ) (runs : List (List Act × End)) (d e : Ev) : Prop :=
  ∃ a acts en, (a, acts, en) ∈ P.agents.zip runs ∧ P.DepIn a acts d e

/-- Reads related to their thread's later events: a relation containing every dependency. -/
def readPo (evs : List Ev) : NV.Rel evs.length :=
  NV.Rel.ofFn fun a b => evs[a].e.isRead && evs[a].e.thread == evs[b].e.thread && decide (evs[a].seq < evs[b].seq)

inductive Outcome where
  /-- Every agent exited, leaving this memory. -/
  | done (mem : Loc → UInt8)
  | fault (msg : String)
  /-- Outside what the model covers. -/
  | unsupported (msg : String)

variable {σ : Type} (P : Program σ)

def Program.events (runs : List (List Act × End)) : List Ev :=
  (P.agents.zip runs).flatMap fun (a, acts, _) =>
    P.agentEvents a (fun _ => 0) acts ++ P.asyncEvents a (fun _ => 0) {} acts

structure Candidate where
  /-- Each agent's run: what it did and how it ended. -/
  runs : List (List Act × End)
  length : runs.length = P.agents.length
  runs_ok : ∀ p ∈ P.agents.zip runs, Run P.step p.1.init p.2.1 p.2.2
  /-- The events, in any arrangement. -/
  evs : List Ev
  perm : evs.Perm (P.events runs)
  rf : NV.Rel evs.length
  co : NV.Rel evs.length
  sc : NV.Rel evs.length
  /-- The collectives' instances and answers. -/
  collects : P.CollectsOk runs
  /-- The dependencies No-Thin-Air sees: at least the runs' (`Program.SemDep`). -/
  dep : NV.Rel evs.length
  dep_ok : ∀ i j : Fin evs.length, P.SemDep runs evs[i] evs[j] → dep i j = true

namespace Candidate

variable {P} (c : Candidate P)

def exec : Execution c.evs.length (List UInt8) :=
  let x := toExec c.evs P.place P.mem c.rf c.co c.sc
  { x with dep := x.dep ∪ c.dep }

/-- A run is cut short only beside one that faulted or left the model. -/
def Valid : Prop :=
  BarriersOk c.evs ∧
    ((∃ re ∈ c.runs, re.2 = .stop) → ∃ re ∈ c.runs, re.2 ≠ .exit ∧ re.2 ≠ .stop)

/-- Memory `m` is a final memory of the candidate: each byte is that of a write to it which no
write follows in coherence order, or the initial byte when nothing writes it. -/
def IsFinal (m : Loc → UInt8) : Prop :=
  ∀ l, (∃ i : Fin c.evs.length, c.evs[i].covers l = true ∧ (∀ j, c.co i j = false) ∧
      m l = c.evs[i].val.getD (l.addr - c.evs[i].loc.addr) 0) ∨
    ((∀ i : Fin c.evs.length, c.evs[i].covers l = false) ∧ m l = P.mem l)

def Yields : Outcome → Prop
  | .done m => (∀ re ∈ c.runs, re.2 = .exit) ∧ Uniform c.evs ∧ c.IsFinal m
  | .fault msg => ∃ re ∈ c.runs, re.2 = .fault msg
  | .unsupported msg => (∃ re ∈ c.runs, re.2 = .unsupported msg) ∨
      (¬ Uniform c.evs ∧ msg = "accesses that overlap partially")

end Candidate

/-- **The outcomes of a program**: what the consistent candidate executions yield. -/
def Outcomes (o : Outcome) : Prop :=
  ∃ c : Candidate P, c.Valid ∧ c.exec.consistent = true ∧ c.Yields o

end NV.Formal
