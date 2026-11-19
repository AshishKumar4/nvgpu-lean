import NV.Formal.RunEvents
import NV.Memory.Sound

/-!
# The reference machine: simulator outputs are outcomes

An untimed machine that runs a program's agents on the request machine (`NV.Memory.Flat`): an
agent steps its semantics; a memory request is issued to the machine and the agent waits until
every lane of it has been delivered, in whatever order, interleaved with every other agent's
steps; fences and barriers act on the machine's drained issuer and barrier counts. Any timing
model that schedules these steps is one of its runs.

`Sim.done_outcome`: when every agent of a run has exited, the machine's memory is an outcome of the
program (`Outcomes`), as long as the accesses did not overlap partially (the memory model's own
limit) and the agents occupy disjoint threads. `Sim.fault_outcome`: a fault an agent reaches with
no request in flight is an outcome. So every output of the simulator is allowed by the PTX memory
model, and a theorem about all outcomes holds of every simulated run.

The agents block on their requests, as `Outcomes`'s runs do (the blocking reading of
`Effect.mem`). A cluster wait waits on an instance its own thread arrived at.
-/

namespace NV.Formal

open NV.Flat

/-- What an agent of the machine is doing. -/
inductive Status (σ : Type) where
  | run (s : σ)
  /-- Waiting for request `id`; `got l` is what lane `l`'s reads returned. -/
  | mem (id : Nat) (r : Req) (s : σ) (wb : Resp → σ → σ) (got : Nat → List UInt8)
  /-- Arrived at instance `inst` of a `bar.sync` barrier, waiting for it to complete. -/
  | bar (op : SyncOp) (inst : Nat) (s : σ)
  | done
  | fault (msg : String)

structure Sim (σ : Type) where
  g : Ghost
  /-- Agent `i`'s status and what it has done, oldest first. -/
  st : Nat → Status σ
  acts : Nat → List Act

def upd {β : Type} (f : Nat → β) (i : Nat) (v : β) : Nat → β := fun j => if j = i then v else f j

variable {σ : Type} (P : Program σ)

/-- Agent `a`'s issuer as the `i`th: its own queue slot, its threads, its CTA. -/
def Agent.issuer (a : Agent σ) (i : Nat) : Issuer := { slot := i, tid := a.tid, cta := a.place.cta }

def Sim.init (m : Flat) : Sim σ :=
  { g := Ghost.init m, st := fun i => match P.agents[i]? with | some a => .run a.init | none => .done
    acts := fun _ => [] }

namespace Sim

/-- Agent `i` moves to status `y`, having done `new`, the machine to `g`. -/
def set (S : Sim σ) (i : Nat) (y : Status σ) (new : List Act) (g : Ghost) : Sim σ :=
  { g, st := upd S.st i y, acts := upd S.acts i (S.acts i ++ new) }

def barId : SyncOp → Option Nat
  | .bar id _ _ => some id
  | _ => none

variable {P}

inductive Step : Sim σ → Sim σ → Prop
  | next {S i a s s'} : P.agents[i]? = some a → S.st i = .run s → P.step s = .next s' →
      Step S (S.set i (.run s') [] S.g)
  | reg {S i a s s' wb} : P.agents[i]? = some a → S.st i = .run s → P.step s = .reg s' wb →
      Step S (S.set i (.run (wb s')) [] S.g)
  | issue {S i a s rf s' wb g'} : P.agents[i]? = some a → S.st i = .run s →
      P.step s = .mem rf s' wb → (∀ l, (rf s').active l = true → l < a.lanes) →
      S.g.step (.issue (a.issuer i) (rf s') a.place.cta) = some g' →
      Step S (S.set i (.mem S.g.m.next (rf s') s' wb fun _ => []) [] g')
  | deliver {S i id r s wb got l m bytes g'} : S.st i = .mem id r s wb got →
      S.g.m.deliver i id l = .ok (m, bytes) → S.g.step (.deliver i id l) = some g' →
      Step S (S.set i (.mem id r s wb (upd got l bytes)) [] g')
  | complete {S i id r s wb got} : S.st i = .mem id r s wb got → S.g.m.find i id = none →
      Step S (S.set i (.run (wb (r.resp got) s)) [.mem r got] S.g)
  | fence {S i a s op s' g'} : P.agents[i]? = some a → S.st i = .run s → P.step s = .sync op s' →
      isFenceOp op = true → S.g.step (.fence (a.issuer i) (List.range a.lanes) op) = some g' →
      Step S (S.set i (.run s') [.sync op 0] g')
  /-- `bar.arrive` or `barrier.cluster.arrive`. -/
  | arrive {S i a s op s' g'} : P.agents[i]? = some a → S.st i = .run s → P.step s = .sync op s' →
      (op = .clusterArrive ∨ ∃ id n, op = .bar id n true) →
      S.g.step (.arrive (a.issuer i) (List.range a.lanes) (barId op) (op.barObj a.place)
        (P.expected a.tid op)) = some g' →
      Step S (S.set i (.run s') [.sync op (S.g.m.bar (op.barObj a.place)).done] g')
  /-- `bar.sync`'s arrival. -/
  | barArrive {S i a s id n s' g'} : P.agents[i]? = some a → S.st i = .run s →
      P.step s = .sync (.bar id n false) s' →
      S.g.step (.arrive (a.issuer i) (List.range a.lanes) (some id)
        ((SyncOp.bar id n false).barObj a.place) (P.expected a.tid (.bar id n false))) = some g' →
      Step S (S.set i (.bar (.bar id n false)
        (S.g.m.bar ((SyncOp.bar id n false).barObj a.place)).done s') [] g')
  /-- `bar.sync`'s wait, once its instance has completed. -/
  | barWait {S i a id n k s g'} : P.agents[i]? = some a → S.st i = .bar (.bar id n false) k s →
      S.g.step (.wait (a.issuer i) (List.range a.lanes) (some id)
        ((SyncOp.bar id n false).barObj a.place) k (P.expected a.tid (.bar id n false))) = some g' →
      Step S (S.set i (.run s) [.sync (.bar id n false) k] g')
  /-- `barrier.cluster.wait`, on an instance its thread arrived at. -/
  | clusterWait {S i a s s' k g'} : P.agents[i]? = some a → S.st i = .run s →
      P.step s = .sync .clusterWait s' →
      (∃ v ∈ S.g.evs, v.arrives = true ∧ v.e.thread = a.tid ∧
        v.obj = SyncOp.clusterWait.barObj a.place ∧ v.inst = k ∧
        v.count = P.expected a.tid .clusterWait) →
      S.g.step (.wait (a.issuer i) (List.range a.lanes) none (SyncOp.clusterWait.barObj a.place) k
        (P.expected a.tid .clusterWait)) = some g' →
      Step S (S.set i (.run s') [.sync .clusterWait k] g')
  | exit {S i a s} : P.agents[i]? = some a → S.st i = .run s → P.step s = .exit →
      Step S (S.set i .done [] S.g)
  | fault {S i a s msg} : P.agents[i]? = some a → S.st i = .run s → P.step s = .fault msg →
      Step S (S.set i (.fault msg) [] S.g)

variable (P)

/-- The states a run from machine `m` reaches. -/
inductive Reachable (m : Flat) : Sim σ → Prop
  | init : Reachable m (Sim.init P m)
  | step {S S'} : Reachable m S → Step (P := P) S S' → Reachable m S'

open Classical in
/-- The memory a run leaves: the machine's bytes where it has them, the program's elsewhere. -/
noncomputable def final (S : Sim σ) (L : Loc) : UInt8 :=
  if InBounds S.g.m L then byteAt S.g.m L else P.mem L

end Sim

/-! ## Bookkeeping -/

theorem upd_same {β : Type} (f : Nat → β) (i : Nat) (v : β) : upd f i v i = v := by simp [upd]
theorem upd_other {β : Type} (f : Nat → β) {i j : Nat} (v : β) (h : j ≠ i) : upd f i v j = f j := by
  simp [upd, h]

/-- Positions thread `t` of agent `a` has used in `acts`. -/
def wsum (a : Agent σ) (acts : List Act) (t : Nat) : Nat := (acts.map (Act.width a t)).sum

theorem wsum_append (a : Agent σ) (acts : List Act) (act : Act) (t : Nat) :
    wsum a (acts ++ [act]) t = wsum a acts t + act.width a t := by
  simp [wsum]

theorem agentEvents_append (a : Agent σ) (act : Act) :
    ∀ (acts : List Act) (c : Nat → Nat), P.agentEvents a c (acts ++ [act]) =
      P.agentEvents a c acts ++ P.actEvents a (fun t => c t + wsum a acts t) act
  | [], c => by simp [Program.agentEvents, wsum]
  | x :: acts, c => by
    simp only [List.cons_append, Program.agentEvents, List.append_assoc]
    rw [agentEvents_append a act acts]
    congr 3
    funext t
    simp [wsum, Nat.add_assoc]

/-- The agents occupy disjoint thread ranges. -/
def Program.Disjoint : Prop :=
  P.agents.Pairwise fun a b => a.tid + a.lanes ≤ b.tid ∨ b.tid + b.lanes ≤ a.tid

/-- The agent a thread belongs to. -/
def Program.slotOf (t : Nat) : Nat :=
  (P.agents.findIdx? fun a => a.tid ≤ t && t < a.tid + a.lanes).getD 0

/-- Agent `i`'s issuer, every agent's own. -/
def Program.issuerOf (i : Nat) : Issuer :=
  match P.agents[i]? with
  | some a => a.issuer i
  | none => default

section disjoint
variable {P} (hd : P.Disjoint)
include hd

theorem agents_disjoint {i j : Nat} {a b : Agent σ} (hi : P.agents[i]? = some a)
    (hj : P.agents[j]? = some b) (hij : i ≠ j) : a.tid + a.lanes ≤ b.tid ∨ b.tid + b.lanes ≤ a.tid := by
  obtain ⟨hi', rfl⟩ := List.getElem?_eq_some_iff.1 hi
  obtain ⟨hj', rfl⟩ := List.getElem?_eq_some_iff.1 hj
  rcases Nat.lt_or_gt_of_ne hij with c | c
  · exact List.pairwise_iff_getElem.1 hd i j hi' hj' c
  · exact (List.pairwise_iff_getElem.1 hd j i hj' hi' c).symm

/-- The first agent holding a thread of agent `i` is agent `i`. -/
theorem findIdx_lane {i : Nat} {a : Agent σ} (hi : P.agents[i]? = some a) {l : Nat} (hl : l < a.lanes) :
    P.agents.findIdx? (fun b => b.tid ≤ a.tid + l && a.tid + l < b.tid + b.lanes) = some i := by
  obtain ⟨hi', rfl⟩ := List.getElem?_eq_some_iff.1 hi
  rw [List.findIdx?_eq_some_iff_getElem]
  refine ⟨hi', by simp; omega, fun j hj => ?_⟩
  have := agents_disjoint hd (List.getElem?_eq_getElem (Nat.lt_trans hj hi')) hi
    (Nat.ne_of_lt hj)
  cases h : (decide (P.agents[j].tid ≤ P.agents[i].tid + l) &&
      decide (P.agents[i].tid + l < P.agents[j].tid + P.agents[j].lanes))
  · simp
  · simp only [Bool.and_eq_true, decide_eq_true_eq] at h
    omega

theorem slotOf_lane {i : Nat} {a : Agent σ} (hi : P.agents[i]? = some a) {l : Nat} (hl : l < a.lanes) :
    P.slotOf (a.tid + l) = i := by
  simp [Program.slotOf, findIdx_lane hd hi hl]

theorem place_lane {i : Nat} {a : Agent σ} (hi : P.agents[i]? = some a) {l : Nat} (hl : l < a.lanes) :
    P.place (a.tid + l) = a.place := by
  have hf := findIdx_lane hd hi hl
  apply Program.place_of
  unfold Program.agentPlace
  rw [List.find?_eq_some_iff_getElem.2 ?_]
  · rfl
  obtain ⟨hi', rfl⟩ := List.getElem?_eq_some_iff.1 hi
  obtain ⟨h₁, h₂, h₃⟩ := List.findIdx?_eq_some_iff_getElem.1 hf
  refine ⟨by simpa using h₂, i, hi', rfl, fun j hj => ?_⟩
  have := h₃ j hj
  by_cases h₄ : P.agents[j].tid ≤ P.agents[i].tid + l
  · by_cases h₅ : P.agents[i].tid + l < P.agents[j].tid + P.agents[j].lanes
    · simp [h₄, h₅] at this
    · simp [h₄, h₅]
  · simp [h₄]

theorem expected_lane {i : Nat} {a : Agent σ} (hi : P.agents[i]? = some a) {l : Nat} (hl : l < a.lanes)
    (op : SyncOp) : P.expected (a.tid + l) op = P.expected a.tid op := by
  have h₁ := place_lane hd hi hl
  have h₀ : a.tid = a.tid + 0 := rfl
  by_cases hz : 0 < a.lanes
  · have h₂ := place_lane hd hi hz
    rw [← h₀] at h₂
    cases op <;> simp only [Program.expected, h₁, h₂] <;> split <;> rfl
  · omega

end disjoint

/-! ## Lists -/

theorem flatMap_ext {α β : Type} {f g : α → List β} :
    ∀ {l : List α}, (∀ x ∈ l, f x = g x) → l.flatMap f = l.flatMap g
  | [], _ => rfl
  | x :: l, h => by
    rw [List.flatMap_cons, List.flatMap_cons, h x (by simp), flatMap_ext fun y hy => h y (by simp [hy])]

theorem perm_flatMap_upd {β : Type} {F F' : Nat → List β} {new : List β} {i : Nat}
    (hF : (F' i).Perm (F i ++ new)) (hother : ∀ j, j ≠ i → F' j = F j) :
    ∀ {n : Nat}, i < n → ((List.range n).flatMap F').Perm ((List.range n).flatMap F ++ new)
  | 0, h => absurd h (Nat.not_lt_zero _)
  | n + 1, h => by
    rw [List.range_succ, List.flatMap_append, List.flatMap_append, List.flatMap_singleton,
      List.flatMap_singleton]
    rcases Nat.lt_or_ge i n with c | c
    · rw [hother n (by omega)]
      refine ((perm_flatMap_upd hF hother c).append_right _).trans ?_
      rw [List.append_assoc, List.append_assoc]
      exact List.Perm.append_left _ List.perm_append_comm
    · have hi : i = n := by omega
      subst hi
      have : (List.range i).flatMap F' = (List.range i).flatMap F :=
        flatMap_ext fun j hj => hother j (by simp at hj; omega)
      rw [this, List.append_assoc]
      exact List.Perm.append_left _ hF

theorem flatMap_pair {α β : Type} (l : List α) (f g : α → β) :
    (l.flatMap fun x => [f x, g x]).Perm (l.map f ++ l.map g) := by
  induction l with
  | nil => simp
  | cons x l ih =>
    simp only [List.flatMap_cons, List.map_cons, List.cons_append]
    refine List.Perm.cons _ ?_
    refine (List.Perm.cons _ ih).trans ?_
    exact List.perm_middle.symm

theorem flatMap_single {α β : Type} (l : List α) (f : α → β) : (l.flatMap fun x => [f x]) = l.map f := by
  induction l with
  | nil => rfl
  | cons x l ih => simp [ih]

theorem events_fence {op : SyncOp} (h : isFenceOp op = true) (t s o k c : Nat) :
    op.events t s o k c = [fenceEvent op t s] := by
  cases op <;> simp_all [isFenceOp, SyncOp.events]

theorem events_arrive {op : SyncOp} (h : op = .clusterArrive ∨ ∃ id n, op = .bar id n true)
    (t s o k c : Nat) : op.events t s o k c = [barEvent t s (Sim.barId op) false o k c] := by
  rcases h with rfl | ⟨id, n, rfl⟩ <;> rfl

theorem events_barSync (id : Nat) (n : Option Nat) (t s o k c : Nat) :
    (SyncOp.bar id n false).events t s o k c =
      [barEvent t s (some id) false o k c, barEvent t (s + 1) (some id) true o k c] := rfl

theorem events_clusterWait (t s o k c : Nat) :
    SyncOp.clusterWait.events t s o k c = [barEvent t s none true o k c] := rfl

/-! ## The invariant -/

/-- Lane `l` of request `id` is still in flight in `qs`. -/
def inflight (qs : List InFlight) (id l : Nat) : Bool := qs.any fun q => q.id == id && q.lanes.contains l

/-- The events of agent `a`'s current operation already performed: the delivered lanes of its
request, or the arrivals of its `bar.sync`. -/
def pending (a : Agent σ) (acts : List Act) (qs : List InFlight) : Status σ → List Ev
  | .mem id r _ _ got => (List.range r.addrs.size).flatMap fun l =>
      if inflight qs id l then [] else
        match a.loc r l with
        | some loc => r.laneEvents l (a.tid + l) (wsum a acts (a.tid + l)) loc (got l)
        | none => []
  | .bar op k _ => (List.range a.lanes).map fun l =>
      barEvent (a.tid + l) (wsum a acts (a.tid + l)) (Sim.barId op) false (op.barObj a.place) k
        (P.expected a.tid op)
  | _ => []

/-- Positions agent `a`'s current operation has taken in thread `t` beyond its recorded acts. -/
def extra (a : Agent σ) (t : Nat) : Status σ → Nat
  | .mem _ r _ _ got => Act.width a t (.mem r got)
  | .bar .. => 1
  | _ => 0

/-- A run of agent `a` so far: `acts`, and the status it is in, continue to runs of its code. -/
def Tracks (a : Agent σ) (acts : List Act) : Status σ → Prop
  | .run s => ∀ rest e, Run P.step s rest e → Run P.step a.init (acts ++ rest) e
  | .mem _ r s wb _ => ∀ read rest e, (∀ l, r.active l = true → (read l).length = r.readLen) →
      r.unsupported read = false → Run P.step (wb (r.resp read) s) rest e →
      Run P.step a.init (acts ++ .mem r read :: rest) e
  | .bar op k s => ∀ rest e, Run P.step s rest e → Run P.step a.init (acts ++ .sync op k :: rest) e
  | .done => Run P.step a.init acts .exit
  | .fault msg => Run P.step a.init acts (.fault msg)

/-- What the machine holds for agent `i`: nothing in flight unless it waits on a request, and
then only that request, issued as its own. -/
def Holds (g : Ghost) (acts : List Act) (i : Nat) (a : Agent σ) : Status σ → Prop
  | .mem id r _ _ got =>
    id < g.m.next ∧ (∀ l, r.active l = true → l < a.lanes) ∧
    (∀ q ∈ g.m.queue i, q.id = id ∧ q.who = a.issuer i ∧ q.cta = a.place.cta ∧ q.req = r) ∧
    (∀ l, r.active l = true → g.base id l = wsum a acts (a.tid + l)) ∧
    (∀ l, r.active l = true → inflight (g.m.queue i) id l = false →
      (got l).length = r.readLen ∧ (r.op.writes && r.op != .st → (r.atomNew l (got l)).isSome) ∧
      (∃ x, a.loc r l = some x ∧ loc a.place.cta r l = some x))
  | .bar op _ _ => g.m.queue i = [] ∧ ∃ id n, op = .bar id n false
  | _ => g.m.queue i = []

/-- Arrivals at the instance of a barrier event. -/
def arrivals (evs : List Ev) (key : Option (Nat × Nat)) : Nat :=
  (evs.filter fun v => v.arrives && v.barKey == key).length

/-- Barrier accounting: events of one instance expect one count; a completed instance has as many
arrivals as it expects; the current one has the machine's count, and no waits. -/
structure BarInv (g : Ghost) : Prop where
  same : ∀ v ∈ g.evs, ∀ w ∈ g.evs, v.barKey ≠ none → v.barKey = w.barKey → v.count = w.count
  done : ∀ v ∈ g.evs, v.barKey ≠ none → v.inst < (g.m.bar v.obj).done →
    arrivals g.evs v.barKey = v.count
  cur : ∀ v ∈ g.evs, v.barKey ≠ none → v.inst = (g.m.bar v.obj).done →
    v.arrives = true ∧ v.count = (g.m.bar v.obj).expected ∧
      (g.m.bar v.obj).arrived < (g.m.bar v.obj).expected
  count : ∀ b, arrivals g.evs (some (b, (g.m.bar b).done)) = (g.m.bar b).arrived

structure SimInv (S : Sim σ) : Prop where
  ghost : Inv P.mem P.issuerOf P.slotOf S.g
  perm : S.g.evs.Perm ((List.range P.agents.length).flatMap fun i =>
    match P.agents[i]? with
    | some a => P.agentEvents a (fun _ => 0) (S.acts i) ++ pending P a (S.acts i) (S.g.m.queue i) (S.st i)
    | none => [])
  agents : ∀ i a, P.agents[i]? = some a →
    Tracks P a (S.acts i) (S.st i) ∧ Holds S.g (S.acts i) i a (S.st i) ∧
      ∀ l < a.lanes, S.g.ctr (a.tid + l) = wsum a (S.acts i) (a.tid + l) + extra a (a.tid + l) (S.st i)
  beyond : ∀ i, P.agents.length ≤ i → S.st i = .done ∧ S.acts i = []
  bars : BarInv S.g
  bounds : ∀ v ∈ S.g.evs, ∀ L, v.covers L = true → InBounds S.g.m L
  /-- The machine has no asynchronous operations: every recorded action is plain. -/
  plain : ∀ i, ∀ act ∈ S.acts i, act.plain = true

theorem plain_set {S : Sim σ} (h : ∀ i, ∀ act ∈ S.acts i, act.plain = true) {i : Nat} {y : Status σ}
    {new : List Act} {g : Ghost} (hn : ∀ act ∈ new, act.plain = true) :
    ∀ j, ∀ act ∈ (S.set i y new g).acts j, act.plain = true := by
  intro j act ha
  simp only [Sim.set, upd] at ha
  split at ha
  · rcases List.mem_append.1 ha with ha | ha
    · subst_vars; exact h _ act ha
    · exact hn act ha
  · exact h j act ha

/-! ## The invariant holds initially -/

theorem simInv_init {m : Flat} (hq : ∀ s, m.queue s = []) (hs : StoresOk m) (hbars : m.bars = #[])
    (hm : ∀ L, InBounds m L → byteAt m L = P.mem L)
    (hf : ∀ L : Loc, L.space = .shared → m.shared.size ≤ L.owner → P.mem L = 0) :
    SimInv P (Sim.init P m) where
  ghost := inv_init_of hq hs hm hf
  perm := by
    have : ∀ i ∈ List.range P.agents.length, (match P.agents[i]? with
        | some a => P.agentEvents a (fun _ => 0) ((Sim.init P m).acts i) ++
            pending P a ((Sim.init P m).acts i) ((Sim.init P m).g.m.queue i) ((Sim.init P m).st i)
        | none => []) = [] := by
      intro i _
      cases h : P.agents[i]? with
      | none => rfl
      | some a => simp [Sim.init, h, Program.agentEvents, pending]
    rw [flatMap_ext this]
    simp [Sim.init, Ghost.init]
  agents i a hi := by
    simp only [Sim.init, hi]
    refine ⟨fun rest e h => by simpa using h, hq i, fun l _ => ?_⟩
    simp [Ghost.init, wsum, extra]
  beyond i hi := by
    simp [Sim.init, List.getElem?_eq_none hi]
  bars := {
    same := by simp [Sim.init, Ghost.init]
    done := by simp [Sim.init, Ghost.init]
    cur := by simp [Sim.init, Ghost.init]
    count := fun b => by simp [Sim.init, Ghost.init, arrivals, Flat.bar, hbars]; rfl }
  bounds := by simp [Sim.init, Ghost.init]
  plain := by simp [Sim.init]

/-! ## Steps that only move one agent's code -/

/-- An agent running, done or faulted holds nothing. -/
def Status.idle : Status σ → Bool
  | .run _ | .done | .fault _ => true
  | _ => false

theorem SimInv.local {S : Sim σ} (h : SimInv P S) {i : Nat} {a : Agent σ} (hi : P.agents[i]? = some a)
    {s : σ} (hs : S.st i = .run s) {y : Status σ} (hy : y.idle = true) (htr : Tracks P a (S.acts i) y) :
    SimInv P (S.set i y [] S.g) := by
  have hpy : ∀ acts qs, pending P a acts qs y = [] := fun _ _ => by
    cases y <;> simp_all [Status.idle, pending]
  have hxy : ∀ t, extra a t y = 0 := fun _ => by cases y <;> simp_all [Status.idle, extra]
  obtain ⟨htr₀, hho, hctr⟩ := h.agents i a hi
  rw [hs] at hho hctr
  have hq : S.g.m.queue i = [] := hho
  refine { ghost := h.ghost, perm := ?_, agents := ?_, beyond := ?_, bars := h.bars
           bounds := h.bounds, plain := plain_set h.plain (by simp) }
  · refine h.perm.trans (List.Perm.of_eq (flatMap_ext fun j _ => ?_))
    by_cases hj : j = i
    · subst hj
      simp only [Sim.set, upd_same, hi, List.append_nil, hs, hpy]
      simp [pending]
    · simp only [Sim.set, upd_other _ _ hj]
  · intro j b hj
    by_cases hji : j = i
    · subst hji
      rw [hi] at hj; cases hj
      simp only [Sim.set, upd_same, List.append_nil]
      refine ⟨htr, ?_, fun l hl => by rw [hctr l hl, hxy]; simp [extra]⟩
      cases y <;> simp_all [Status.idle, Holds]
    · simp only [Sim.set, upd_other _ _ hji]
      exact h.agents j b hj
  · intro j hj
    have : j ≠ i := fun e => by subst e; simp [List.getElem?_eq_none hj] at hi
    simp only [Sim.set, upd_other _ _ this]
    exact h.beyond j hj

/-! ## Completing a request -/

theorem unsupported_false {r : Req} {got : Nat → List UInt8}
    (h : ∀ l, r.active l = true → (r.op.writes && r.op != .st → (r.atomNew l (got l)).isSome)) :
    r.unsupported got = false := by
  unfold Req.unsupported
  cases hw : (r.op.writes && r.op != .st)
  · simp
  · simp only [Bool.true_and, List.any_eq_false, List.mem_range, Bool.and_eq_true, not_and,
      Option.isNone_iff_eq_none]
    intro l _ ha hn
    have := h l ha hw
    rw [hn] at this; cases this

theorem agent_of_st {S : Sim σ} (h : SimInv P S) {i : Nat} (hne : (S.st i).idle = false ∨
    ∃ id r s wb got, S.st i = .mem id r s wb got) : ∃ a, P.agents[i]? = some a := by
  cases ha : P.agents[i]? with
  | some a => exact ⟨a, rfl⟩
  | none =>
    exfalso
    have hi : P.agents.length ≤ i := by
      rcases Nat.lt_or_ge i P.agents.length with c | c
      · rw [List.getElem?_eq_getElem c] at ha; cases ha
      · exact c
    rw [(h.beyond i hi).1] at hne
    rcases hne with hne | ⟨_, _, _, _, _, hne⟩ <;> cases hne

theorem SimInv.complete {S : Sim σ} (h : SimInv P S) {i id : Nat} {r : Req} {s : σ}
    {wb : Resp → σ → σ} {got : Nat → List UInt8} (hst : S.st i = .mem id r s wb got)
    (hf : S.g.m.find i id = none) :
    SimInv P (S.set i (.run (wb (r.resp got) s)) [.mem r got] S.g) := by
  obtain ⟨a, hi⟩ := agent_of_st P h (.inr ⟨_, _, _, _, _, hst⟩)
  obtain ⟨htr, hho, hctr⟩ := h.agents i a hi
  rw [hst] at htr hho hctr
  obtain ⟨-, -, hqs, -, hdel⟩ := hho
  -- Nothing of the request is in flight.
  have hnone : ∀ q ∈ S.g.m.queue i, q.id ≠ id := by
    intro q hq e
    simp only [Flat.find, Option.map_eq_none_iff, List.findIdx?_eq_none_iff] at hf
    exact absurd e (by simpa using hf q hq)
  have hq : S.g.m.queue i = [] := by
    cases hq : S.g.m.queue i with
    | nil => rfl
    | cons q _ => exact absurd (hqs q (by simp [hq])).1 (hnone q (by simp [hq]))
  have hinf : ∀ l, inflight (S.g.m.queue i) id l = false := fun l => by simp [inflight, hq]
  refine { ghost := h.ghost, perm := ?_, agents := ?_, beyond := ?_, bars := h.bars
           bounds := h.bounds, plain := plain_set h.plain (by simp [Act.plain]) }
  · refine h.perm.trans (List.Perm.of_eq (flatMap_ext fun j _ => ?_))
    by_cases hj : j = i
    · subst hj
      simp only [Sim.set, upd_same, hi, hst, agentEvents_append, pending, hinf,
        Program.actEvents, Nat.zero_add, List.append_nil, Bool.false_eq_true,
        ↓reduceIte]
      rfl
    · simp only [Sim.set, upd_other _ _ hj]
  · intro j b hj
    by_cases hji : j = i
    · subst hji
      rw [hi] at hj; cases hj
      simp only [Sim.set, upd_same]
      refine ⟨fun rest e hr => ?_, hq, fun l hl => ?_⟩
      · have := htr got rest e (fun l ha => (hdel l ha (hinf l)).1)
          (unsupported_false fun l ha => (hdel l ha (hinf l)).2.1) hr
        simpa using this
      · rw [hctr l hl, wsum_append]; simp [extra]
    · simp only [Sim.set, upd_other _ _ hji]
      exact h.agents j b hj
  · intro j hj
    have : j ≠ i := fun e => by subst e; simp [List.getElem?_eq_none hj] at hi
    simp only [Sim.set, upd_other _ _ this]
    exact h.beyond j hj

/-! ## Issuing a request -/

theorem active_lt {r : Req} {l : Nat} (h : r.active l = true) : l < r.addrs.size := by
  simp only [Req.active] at h
  rcases Nat.lt_or_ge l r.addrs.size with c | c
  · exact c
  · rw [Array.getElem?_eq_none c] at h; simp at h

theorem issued_contains (m : Flat) (who : Issuer) (r : Req) (cta l : Nat) :
    (issued m who r cta).lanes.contains l = r.active l := by
  rw [Bool.eq_iff_iff, List.contains_iff_mem]
  simp only [issued, List.mem_filter, List.mem_range]
  exact ⟨fun h => h.2, fun h => ⟨active_lt h, h⟩⟩

theorem issue_step {g g' : Ghost} {who : Issuer} {r : Req} {cta : Nat}
    (h : g.step (.issue who r cta) = some g') :
    who.slot < g.m.queues.size ∧ g'.m = afterIssue g.m who r cta ∧ g'.evs = g.evs ∧
      g'.ctr = (fun t => if who.tid ≤ t ∧ r.active (t - who.tid) = true then
        g.ctr t + r.pieces.length else g.ctr t) ∧
      g'.base = fun i l => if i = g.m.next then g.ctr (who.tid + l) else g.base i l := by
  simp only [Ghost.step] at h
  split at h
  · next m id heq =>
    obtain ⟨hsl, -, rfl, rfl⟩ := issue_ok heq
    simp only [Option.some.injEq] at h
    subst h
    exact ⟨hsl, rfl, rfl, rfl, rfl⟩
  · cases h

theorem afterIssue_bar (m : Flat) (who : Issuer) (r : Req) (cta b : Nat) :
    (afterIssue m who r cta).bar b = m.bar b := rfl

theorem afterIssue_store (m : Flat) (who : Issuer) (r : Req) (cta : Nat) (L : Loc) :
    (afterIssue m who r cta).store? L = m.store? L := by
  simp only [afterIssue, store?]

theorem inBounds_iff {m m' : Flat} (h : ∀ L, m'.store? L = m.store? L) (L : Loc) :
    InBounds m' L ↔ InBounds m L := by
  simp only [InBounds, h]

theorem SimInv.issue (hd : P.Disjoint) {S : Sim σ} (h : SimInv P S) {i : Nat} {a : Agent σ}
    (hi : P.agents[i]? = some a) {s s' : σ} {rf : σ → Req} {wb : Resp → σ → σ} {g' : Ghost}
    (hs : S.st i = .run s) (hstep : P.step s = .mem rf s' wb)
    (hl : ∀ l, (rf s').active l = true → l < a.lanes)
    (hg : S.g.step (.issue (a.issuer i) (rf s') a.place.cta) = some g') :
    SimInv P (S.set i (.mem S.g.m.next (rf s') s' wb fun _ => []) [] g') := by
  obtain ⟨hsl, hm, hevs, hctr', hbase'⟩ := issue_step hg
  obtain ⟨htr, hho, hctr⟩ := h.agents i a hi
  rw [hs] at htr hho hctr
  have hq : S.g.m.queue i = [] := hho
  have hqs : ∀ j, g'.m.queue j = if j = i then [issued S.g.m (a.issuer i) (rf s') a.place.cta]
      else S.g.m.queue j := fun j => by
    rw [hm, queue_issue hsl]
    by_cases hj : j = i
    · subst hj; simp [Agent.issuer, hq]
    · simp [Agent.issuer, hj]
  have hinf : ∀ l, inflight (g'.m.queue i) S.g.m.next l = (rf s').active l := fun l => by
    rw [hqs, ite_eq_left rfl]
    simp only [inflight, List.any_cons, List.any_nil, Bool.or_false, issued_contains]
    simp [issued]
  -- Other agents' threads are not this request's.
  have hother : ∀ j b, P.agents[j]? = some b → j ≠ i → ∀ l < b.lanes,
      ¬(a.tid ≤ b.tid + l ∧ (rf s').active (b.tid + l - a.tid) = true) := by
    intro j b hj hji l hlb ⟨h₁, h₂⟩
    have := hl _ h₂
    rcases agents_disjoint hd hj hi hji with c | c <;> omega
  have hp : pending P a (S.acts i) (g'.m.queue i)
      (.mem S.g.m.next (rf s') s' wb fun _ => []) = [] := by
    apply List.flatMap_eq_nil_iff.2
    intro l _
    rw [hinf]
    cases ha : (rf s').active l
    · have : a.loc (rf s') l = none := by
        simp only [Agent.loc, Option.map_eq_none_iff]
        simpa [Req.active] using ha
      simp [this]
    · simp
  refine { ghost := ?_, perm := ?_, agents := ?_, beyond := ?_, bars := ?_
           bounds := ?_, plain := plain_set h.plain (by simp) }
  · refine h.ghost.issue ⟨by simp [Program.issuerOf, hi, Agent.issuer], fun l ha => ?_⟩ hg
    exact slotOf_lane hd hi (hl l ha)
  · show g'.evs.Perm _
    rw [hevs]
    refine h.perm.trans (List.Perm.of_eq (flatMap_ext fun j _ => ?_))
    by_cases hj : j = i
    · subst hj
      simp only [Sim.set, upd_same, hi, List.append_nil, hs, hp]
      simp [pending]
    · simp only [Sim.set, upd_other _ _ hj, hqs j, hj, ite_false]
  · intro j b hj
    by_cases hji : j = i
    · subst hji
      rw [hi] at hj; cases hj
      simp only [Sim.set, upd_same, List.append_nil]
      refine ⟨fun read rest e h₁ h₂ hr => ?_, ?_, fun l hl' => ?_⟩
      · have := htr _ _ (Run.mem hstep h₁ h₂ hr)
        simpa using this
      · refine ⟨by rw [hm]; exact Nat.lt_succ_self _, hl, fun q hq' => ?_, fun l ha => ?_,
          fun l ha hn => ?_⟩
        · rw [hqs, ite_eq_left rfl, List.mem_singleton] at hq'
          subst hq'
          exact ⟨rfl, rfl, rfl, rfl⟩
        · rw [hbase']
          dsimp only
          rw [ite_eq_left rfl]
          simp only [Agent.issuer]
          rw [hctr l (hl l ha)]
          simp [extra]
        · rw [hinf, ha] at hn; cases hn
      · rw [hctr']
        dsimp only
        simp only [Agent.issuer, Nat.le_add_right, true_and, Nat.add_sub_cancel_left]
        rw [hctr l hl']
        simp only [extra, Act.width, Nat.le_add_right, true_and, Nat.add_sub_cancel_left, Nat.add_zero]
        split <;> rfl
    · obtain ⟨htr₁, hho₁, hctr₁⟩ := h.agents j b hj
      simp only [Sim.set, upd_other _ _ hji]
      refine ⟨htr₁, ?_, fun l hl' => ?_⟩
      · cases hsj : S.st j with
        | mem id₁ r₁ s₁ wb₁ got₁ =>
          rw [hsj] at hho₁
          obtain ⟨h₁, h₂, h₃, h₄, h₅⟩ := hho₁
          have hne : id₁ ≠ S.g.m.next := Nat.ne_of_lt h₁
          refine ⟨by rw [hm]; exact Nat.lt_succ_of_lt h₁, h₂, fun q hq' => ?_, fun l ha => ?_,
            fun l ha hn => ?_⟩
          · rw [hqs, ite_eq_right hji] at hq'; exact h₃ q hq'
          · rw [hbase']; dsimp only; rw [ite_eq_right hne]; exact h₄ l ha
          · rw [hqs, ite_eq_right hji] at hn; exact h₅ l ha hn
        | bar op k s₁ => rw [hsj] at hho₁; exact ⟨by rw [hqs, ite_eq_right hji]; exact hho₁.1, hho₁.2⟩
        | run _ | done | fault _ => rw [hsj] at hho₁; show g'.m.queue j = []; rw [hqs, ite_eq_right hji]; exact hho₁
      · rw [hctr']
        exact (ite_eq_right (hother j b hj hji l hl')).trans (hctr₁ l hl')
  · intro j hj
    have : j ≠ i := fun e => by subst e; simp [List.getElem?_eq_none hj] at hi
    simp only [Sim.set, upd_other _ _ this]
    exact h.beyond j hj
  · show BarInv g'
    have hb : ∀ b, g'.m.bar b = S.g.m.bar b := fun b => by rw [hm]; rfl
    obtain ⟨b₁, b₂, b₃, b₄⟩ := h.bars
    exact ⟨by rw [hevs]; exact b₁, by rw [hevs]; simpa [hb] using b₂, by rw [hevs]; simpa [hb] using b₃,
      fun b => by rw [hevs, hb]; exact b₄ b⟩
  · intro v hv L hc
    show InBounds g'.m L
    rw [hm, inBounds_iff (afterIssue_store _ _ _ _)]
    exact h.bounds v (by rw [← hevs]; exact hv) L hc

/-! ## Delivering a lane -/

theorem inflight_append (x y : List InFlight) (id l : Nat) :
    inflight (x ++ y) id l = (inflight x id l || inflight y id l) := by
  simp [inflight, List.any_append]

theorem retire_cons (q : InFlight) (qs : List InFlight) (id lane : Nat) :
    retire (q :: qs) id lane = (retireOne id lane q).toList ++ retire qs id lane := by
  cases h : retireOne id lane q <;> simp [retire, h]

theorem inflight_retireOne {q : InFlight} (hnd : q.lanes.Nodup) (id lane l : Nat) :
    inflight (retireOne id lane q).toList id l = (q.id == id && q.lanes.contains l && l != lane) := by
  have hmem : (q.lanes.erase lane).contains l = (q.lanes.contains l && l != lane) := by
    rw [Bool.eq_iff_iff]
    simp only [List.contains_iff_mem, Bool.and_eq_true, bne_iff_ne, ne_eq]
    rw [hnd.mem_erase_iff]
    exact and_comm
  unfold retireOne
  by_cases hq : q.id = id
  · simp only [hq, beq_self_eq_true, ↓reduceIte, Bool.true_and]
    split
    · next he =>
      rw [List.isEmpty_iff] at he
      rw [he] at hmem
      simp only [Option.toList_none, inflight, List.any_nil]
      rw [← hmem]; rfl
    · simp only [Option.toList_some, inflight, List.any_cons, List.any_nil, Bool.or_false, 
        beq_self_eq_true, Bool.true_and]
      exact hmem
  · have : (q.id == id) = false := by simp [hq]
    simp only [this, Bool.false_and, Bool.false_eq_true, ↓reduceIte, Option.toList_some, inflight,
      List.any_cons, List.any_nil, Bool.or_false]

theorem inflight_retire {qs : List InFlight} (hnd : ∀ q ∈ qs, q.lanes.Nodup) (id lane l : Nat) :
    inflight (retire qs id lane) id l = (inflight qs id l && l != lane) := by
  induction qs with
  | nil => simp [retire, inflight]
  | cons q qs ih =>
    rw [retire_cons, inflight_append, inflight_retireOne (hnd q (by simp)),
      ih fun q hq => hnd q (by simp [hq])]
    simp only [inflight, List.any_cons]
    cases (q.id == id && q.lanes.contains l) <;> cases (qs.any fun q => q.id == id && q.lanes.contains l) <;>
      cases (l != lane) <;> rfl

theorem readPart_length (r : Req) (off : Nat) (st : Store) : (readPart r off st).length = r.readLen := by
  unfold readPart Req.readLen
  split <;> simp [readBytes]

theorem store?_loc {m : Flat} {cta : Nat} {r : Req} {lane : Nat} {x : Loc} {st : Store}
    (hl : loc cta r lane = some x) (hs : m.store? x = some st) {a : Agent σ} (hc : a.place.cta = cta) :
    a.loc r lane = some x := by
  simp only [loc, Option.map_eq_some_iff] at hl
  obtain ⟨y, hy, rfl⟩ := hl
  simp only [Agent.loc, hy, Option.map_some, Option.some.injEq]
  rcases store?_some hs with ⟨hsp, -, -⟩ | ⟨hsp, -, -⟩ <;> simp only at hsp <;> simp [hsp, hc]

theorem atomNew_some {r : Req} {lane off : Nat} {read : List UInt8} {st st' : Store}
    (hw : performWrites r lane off read (r.pieces.filter (·.write)) st = .ok st')
    (hok : st.WF ∧ st.Fits) (hin : ∀ p ∈ r.pieces.filter (·.write), off + p.off + p.size ≤ st.size)
    (hpw : (r.pieces.filter (·.write)).Pairwise fun a b => a.off + a.size ≤ b.off) :
    r.op.writes && r.op != .st → (r.atomNew lane read).isSome := by
  intro hop
  obtain ⟨-, -, -, -, he, -⟩ := performWrites_ok _ st st' hw hok.1 hok.2 hin hpw
  cases hr : r.op with
  | ld => simp [hr, MemOp.writes] at hop
  | st => simp [hr] at hop
  | atom op ty =>
    obtain ⟨v, hv, -⟩ := he ⟨true, 0, r.bytes, false⟩ (by simp [Req.pieces, hr])
    simp only [Req.written, hr] at hv
    simp [hv]
  | red op ty =>
    obtain ⟨v, hv, -⟩ := he ⟨true, 0, r.bytes, false⟩ (by simp [Req.pieces, hr])
    simp only [Req.written, hr] at hv
    simp [hv]
  | mbar op =>
    obtain ⟨v, hv, -⟩ := he ⟨true, 0, r.bytes, false⟩ (by simp [Req.pieces, hr])
    simp only [Req.written, hr] at hv
    simp [hv]

theorem BarInv.append_plain {g g' : Ghost} (hb : BarInv g) (hm : ∀ b, g'.m.bar b = g.m.bar b)
    {E : List Ev} (he : g'.evs = g.evs ++ E) (hE : ∀ v ∈ E, v.arrives = false ∧ v.waits = false) :
    BarInv g' := by
  have hkey : ∀ v ∈ E, v.barKey = none := fun v hv => by
    simp [Ev.barKey, (hE v hv).1, (hE v hv).2]
  have harr : ∀ key, arrivals g'.evs key = arrivals g.evs key := fun key => by
    rw [he, arrivals, List.filter_append, List.length_append, arrivals]
    have : E.filter (fun v => v.arrives && v.barKey == key) = [] :=
      List.filter_eq_nil_iff.2 fun v hv => by simp [(hE v hv).1]
    rw [this]; rfl
  have hold : ∀ v ∈ g'.evs, v.barKey ≠ none → v ∈ g.evs := fun v hv hk => by
    rw [he] at hv
    rcases List.mem_append.1 hv with hv | hv
    · exact hv
    · exact absurd (hkey v hv) hk
  refine ⟨fun v hv w hw hk e => ?_, fun v hv hk hlt => ?_, fun v hv hk heq => ?_, fun b => ?_⟩
  · exact hb.same v (hold v hv hk) w (hold w hw (e ▸ hk)) hk e
  · rw [harr]; exact hb.done v (hold v hv hk) hk (by rw [← hm]; exact hlt)
  · rw [hm]; exact hb.cur v (hold v hv hk) hk (by rw [← hm]; exact heq)
  · rw [harr, hm]; exact hb.count b

theorem queue_deliver {m : Flat} {x : Loc} {st' : Store} {i : Nat} (X : List InFlight)
    (hs : i < m.queues.size) (j : Nat) :
    ({ m.setStore x st' with queues := m.queues.set! i X } : Flat).queue j =
      if j = i then X else m.queue j := by
  have := queue_set (m := m) X hs j
  simp only [queue] at this ⊢
  exact this

theorem SimInv.deliver {S : Sim σ} (h : SimInv P S) {i id : Nat} {r : Req} {s : σ}
    {wb : Resp → σ → σ} {got : Nat → List UInt8} {l : Nat} {m : Flat} {bytes : List UInt8}
    {g' : Ghost} (hst : S.st i = .mem id r s wb got) (hdl : S.g.m.deliver i id l = .ok (m, bytes))
    (hg : S.g.step (.deliver i id l) = some g') (hu : Uniform g'.evs) :
    SimInv P (S.set i (.mem id r s wb (upd got l bytes)) [] g') := by
  obtain ⟨a, hi⟩ := agent_of_st P h (.inr ⟨_, _, _, _, _, hst⟩)
  obtain ⟨htr, hho, hctr⟩ := h.agents i a hi
  rw [hst] at htr hho hctr
  obtain ⟨hid_lt, hlanes, hqs, hbase, hdel⟩ := hho
  have hg₀ := hg
  obtain ⟨k, hk, x, st, st', bytes', hkid, hlane, -, hx, hstx, hc, hb, hw, hd', rfl⟩ := deliver_facts hg
  have hbytes : bytes = bytes' := by
    rw [hdl] at hd'; simp only [Except.ok.injEq, Prod.mk.injEq] at hd'; exact hd'.2
  subst hbytes
  generalize hq : (S.g.m.queue i)[k] = q at *
  have hqin : q ∈ S.g.m.queue i := hq ▸ List.getElem_mem hk
  obtain ⟨-, hqwho, hqcta, hqreq⟩ := hqs q hqin
  subst hkid
  obtain ⟨-, -, hnd, hbytesok, hql⟩ := h.ghost.queued i q hqin
  have hact : r.active l = true := hqreq ▸ (hql l hlane).1
  have hlt := active_lt hact
  have hP := pieces_ok (hqreq ▸ hbytesok)
  rw [hqreq, hqcta] at hx
  rw [hqreq] at hc hb hw
  have hax : a.loc r l = some x := store?_loc hx hstx rfl
  obtain ⟨swf, sfit⟩ := store?_wf h.ghost.stores hstx
  obtain ⟨hbase0, hfit⟩ := Store.contains_nat (loc_addr_lt hx) hc
  have hin : ∀ p ∈ r.pieces.filter (·.write), x.addr - st.base.toNat + p.off + p.size ≤ st.size :=
    fun p hp => by have := hP.inside p (List.mem_filter.1 hp).1; omega
  obtain ⟨w₁, f₁, b₁, z₁, -, -, -⟩ := performWrites_ok _ st st' hw swf sfit hin hP.writes
  -- The new events.
  have hbl : S.g.base q.id l = wsum a (S.acts i) (a.tid + l) := hbase l hact
  let E := r.laneEvents l (a.tid + l) (wsum a (S.acts i) (a.tid + l)) x bytes
  have hevs : (S.g.evs ++ q.req.laneEvents l (q.who.tid + l) (S.g.base q.id l) x bytes) = S.g.evs ++ E := by
    rw [hqreq, hqwho, hbl]; rfl
  -- The queue after the delivery.
  have hslt : i < S.g.m.queues.size := slot_lt hqin
  have hq' := queue_deliver (m := S.g.m) (x := x) (st' := st') (retire (S.g.m.queue i) q.id l) hslt
  have hndq : ∀ p ∈ S.g.m.queue i, p.lanes.Nodup := fun p hp => (h.ghost.queued i p hp).2.2.1
  have hinf' : ∀ l', inflight (retire (S.g.m.queue i) q.id l) q.id l' =
      (inflight (S.g.m.queue i) q.id l' && l' != l) := inflight_retire hndq _ _
  have hinfl : inflight (S.g.m.queue i) q.id l = true := by
    simp only [inflight, List.any_eq_true, Bool.and_eq_true, beq_iff_eq, List.contains_iff_mem]
    exact ⟨q, hqin, rfl, hlane⟩
  have hothq : ∀ j, j ≠ i → ({ S.g.m.setStore x st' with
      queues := S.g.m.queues.set! i (retire (S.g.m.queue i) q.id l) } : Flat).queue j = S.g.m.queue j :=
    fun j hj => by rw [hq', ite_eq_right hj]
  have hown : ({ S.g.m.setStore x st' with
      queues := S.g.m.queues.set! i (retire (S.g.m.queue i) q.id l) } : Flat).queue i =
      retire (S.g.m.queue i) q.id l := by rw [hq', ite_eq_left rfl]
  have hnext : ({ S.g.m.setStore x st' with
      queues := S.g.m.queues.set! i (retire (S.g.m.queue i) q.id l) } : Flat).next = S.g.m.next :=
    (setStore_fields S.g.m x st').1
  -- The agent's pending events gain the lane's.
  have hpend : (pending P a (S.acts i) (retire (S.g.m.queue i) q.id l) (.mem q.id r s wb (upd got l bytes))).Perm
      (pending P a (S.acts i) (S.g.m.queue i) (.mem q.id r s wb got) ++ E) := by
    simp only [pending]
    refine perm_flatMap_upd (i := l) ?_ (fun l' hl' => ?_) hlt
    · rw [hinf' l, hinfl]
      simp only [bne_self_eq_false, Bool.and_false, Bool.false_eq_true, ↓reduceIte, hax, upd_same,
        List.nil_append]
      exact List.Perm.refl _
    · rw [hinf' l', upd_other _ _ hl']
      simp [hl']
  refine ⟨h.ghost.deliver hg₀ hu, ?_, ?_, ?_, ?_, ?_, plain_set h.plain (by simp)⟩
  · show (S.g.evs ++ _).Perm _
    rw [hevs]
    refine (h.perm.append_right E).trans (List.Perm.symm (perm_flatMap_upd (i := i) ?_ (fun j hj => ?_)
      (List.getElem?_eq_some_iff.1 hi).1))
    · simp only [Sim.set, upd_same, hi, List.append_nil, hst, hown]
      exact (List.Perm.append_left _ hpend).trans (List.Perm.of_eq (List.append_assoc ..).symm)
    · simp only [Sim.set, upd_other _ _ hj, hothq j hj]
  · intro j b hj
    by_cases hji : j = i
    · subst hji
      rw [hi] at hj; cases hj
      simp only [Sim.set, upd_same, List.append_nil]
      refine ⟨htr, ⟨by rw [hnext]; exact hid_lt, hlanes, fun q' hq'' => ?_, hbase, fun l' ha hn => ?_⟩,
        fun l' hl' => by rw [hctr l' hl']; rfl⟩
      · rw [hown] at hq''
        obtain ⟨p₀, hp₀, e₁, e₂, e₃, -, -, -, e₄⟩ := mem_retire hq''
        obtain ⟨f₁, f₂, f₃, f₄⟩ := hqs p₀ hp₀
        exact ⟨e₁.trans f₁, e₂.trans f₂, e₄.trans f₃, e₃.trans f₄⟩
      · rw [hown, hinf'] at hn
        by_cases hll : l' = l
        · subst hll
          rw [upd_same]
          refine ⟨by rw [hb]; exact readPart_length _ _ _, ?_, ⟨x, hax, hx⟩⟩
          exact atomNew_some hw ⟨swf, sfit⟩ hin hP.writes
        · rw [upd_other _ _ hll]
          have : inflight (S.g.m.queue j) q.id l' = false := by simpa [hll] using hn
          exact hdel l' ha this
    · obtain ⟨htr₁, hho₁, hctr₁⟩ := h.agents j b hj
      simp only [Sim.set, upd_other _ _ hji]
      refine ⟨htr₁, ?_, hctr₁⟩
      cases hsj : S.st j with
      | mem id₁ r₁ s₁ wb₁ got₁ =>
        rw [hsj] at hho₁
        obtain ⟨h₁, h₂, h₃, h₄, h₅⟩ := hho₁
        exact ⟨by rw [hnext]; exact h₁, h₂, fun q' hq'' => h₃ q' (by rw [hothq j hji] at hq''; exact hq''),
          h₄, fun l' ha hn => h₅ l' ha (by rw [hothq j hji] at hn; exact hn)⟩
      | bar op k s₁ => rw [hsj] at hho₁; exact ⟨by rw [hothq j hji]; exact hho₁.1, hho₁.2⟩
      | run _ | done | fault _ => rw [hsj] at hho₁; show Flat.queue _ j = []; rw [hothq j hji]; exact hho₁
  · intro j hj
    have : j ≠ i := fun e => by subst e; simp [List.getElem?_eq_none hj] at hi
    simp only [Sim.set, upd_other _ _ this]
    exact h.beyond j hj
  · refine BarInv.append_plain h.bars (fun b => ?_) hevs (fun v hv => ?_)
    · show Flat.bar { S.g.m.setStore x st' with queues := _ } b = _
      simp only [Flat.bar]; rw [(setStore_fields S.g.m x st').2.1]
    · obtain ⟨a', ha', rfl⟩ := List.mem_iff_getElem.1 hv
      obtain ⟨-, -, -, -, -, -, -, -, -, -, -, h₁, h₂, -⟩ := laneEvent_props hP l (a.tid + l)
        (wsum a (S.acts i) (a.tid + l)) x bytes a' ha'
      exact ⟨h₁, h₂⟩
  · -- Bounds: the stores keep their extents.
    have hext : ∀ L, InBounds S.g.m L → InBounds (S.g.m.setStore x st') L := by
      intro L ⟨sL, h₁, h₂, h₃⟩
      show ∃ s, (S.g.m.setStore x st').store? L = some s ∧ _
      rw [store?_setStore hstx]
      split
      · next hs =>
        rw [store?_congr S.g.m hs.1 hs.2, hstx] at h₁
        simp only [Option.some.injEq] at h₁
        subst h₁
        exact ⟨st', rfl, by rw [b₁]; exact h₂, by rw [b₁, z₁]; exact h₃⟩
      · exact ⟨sL, h₁, h₂, h₃⟩
    intro v hv L hcov
    show InBounds { S.g.m.setStore x st' with queues := _ } L
    rw [inBounds_iff (fun L => rfl)]
    rw [hevs] at hv
    rcases List.mem_append.1 hv with hv | hv
    · exact hext L (h.bounds v hv L hcov)
    · obtain ⟨a', ha', rfl⟩ := List.mem_iff_getElem.1 hv
      obtain ⟨-, -, -, -, hloc, hsz, -⟩ := laneEvent_props hP l (a.tid + l)
        (wsum a (S.acts i) (a.tid + l)) x bytes a' ha'
      have hpin := hP.inside _ (List.getElem_mem (l := r.pieces) (by
        rw [laneEvents_length] at ha'; exact ha'))
      rw [covers_iff, hloc, hsz] at hcov
      obtain ⟨-, h₁, h₂, h₃, h₄⟩ := hcov
      refine hext L ⟨st, by rw [store?_congr S.g.m h₁.symm h₂.symm]; exact hstx, ?_, ?_⟩ <;>
        simp only at h₃ h₄ <;> omega

/-! ## Synchronizing -/

theorem Holds.congr {g g' : Ghost} (hq : ∀ j, g'.m.queue j = g.m.queue j) (hn : g'.m.next = g.m.next)
    (hb : g'.base = g.base) {acts : List Act} {i : Nat} {a : Agent σ} {st : Status σ}
    (h : Holds g acts i a st) : Holds g' acts i a st := by
  cases st with
  | mem id r s wb got =>
    obtain ⟨h₁, h₂, h₃, h₄, h₅⟩ := h
    exact ⟨by rw [hn]; exact h₁, h₂, fun q hq' => h₃ q (by rw [← hq]; exact hq'),
      fun l ha => by rw [hb]; exact h₄ l ha, fun l ha hn' => h₅ l ha (by rw [← hq]; exact hn')⟩
  | bar op k s => exact ⟨by rw [hq]; exact h.1, h.2⟩
  | run _ | done | fault _ => show g'.m.queue i = []; rw [hq]; exact h

theorem bump_own {i : Nat} {a : Agent σ} (ctr : Nat → Nat) {l : Nat} (hl : l < a.lanes) :
    bump ctr (a.issuer i) (List.range a.lanes) (a.tid + l) = ctr (a.tid + l) + 1 := by
  have : isLane (a.issuer i) (List.range a.lanes) (a.tid + l) = true :=
    List.any_eq_true.2 ⟨l, List.mem_range.2 hl, by simp [Agent.issuer]⟩
  simp [bump, this]

theorem bump_other (hd : P.Disjoint) {i j : Nat} {a b : Agent σ} (hi : P.agents[i]? = some a)
    (hj : P.agents[j]? = some b) (hji : j ≠ i) (ctr : Nat → Nat) {l : Nat} (hl : l < b.lanes) :
    bump ctr (a.issuer i) (List.range a.lanes) (b.tid + l) = ctr (b.tid + l) := by
  have : isLane (a.issuer i) (List.range a.lanes) (b.tid + l) = false := by
    simp only [isLane, List.any_eq_false, List.mem_range, beq_iff_eq, Agent.issuer]
    intro l' hl' e
    rcases agents_disjoint hd hj hi hji with c | c <;> omega
  simp [bump, this]

theorem syncEvents_eq (g : Ghost) (i : Nat) (a : Agent σ) (mk : Nat → Nat → Ev) :
    g.syncEvents (a.issuer i) (List.range a.lanes) mk =
      (List.range a.lanes).map fun l => mk (a.tid + l) (g.ctr (a.tid + l)) := rfl

theorem bounds_sync {g g' : Ghost} (hb : ∀ v ∈ g.evs, ∀ L, v.covers L = true → InBounds g.m L)
    (hm : ∀ L, g'.m.store? L = g.m.store? L) {E : List Ev} (he : g'.evs = g.evs ++ E)
    (hE : ∀ v ∈ E, v.e.isMem = false) : ∀ v ∈ g'.evs, ∀ L, v.covers L = true → InBounds g'.m L := by
  intro v hv L hc
  rw [inBounds_iff hm]
  rw [he] at hv
  rcases List.mem_append.1 hv with hv | hv
  · exact hb v hv L hc
  · have := hE v hv
    rw [covers_iff] at hc
    simp [PTX.Event.isMem, hc.1] at this

theorem fenceEvent_mem (op : SyncOp) (t s : Nat) : (fenceEvent op t s).e.isMem = false ∧
    (fenceEvent op t s).arrives = false ∧ (fenceEvent op t s).waits = false := by
  obtain ⟨-, -, h₁, -, h₂, h₃⟩ := fenceEvent_props op t s
  exact ⟨h₁, h₂, h₃⟩

theorem actEvents_fence {a : Agent σ} {op : SyncOp} (hf : isFenceOp op = true) (k : Nat)
    (c : Nat → Nat) : P.actEvents a c (.sync op k) =
      (List.range a.lanes).map fun l => fenceEvent op (a.tid + l) (c (a.tid + l)) := by
  simp only [Program.actEvents, events_fence hf, flatMap_single]

theorem width_sync_own {a : Agent σ} (op : SyncOp) (k : Nat) {l : Nat} (hl : l < a.lanes) :
    Act.width a (a.tid + l) (.sync op k) = (match op with | .bar _ _ false => 2 | _ => 1) := by
  simp only [Act.width]
  rw [ite_eq_left ⟨Nat.le_add_right _ _, by omega⟩]
  cases op with
  | bar id c ar => cases ar <;> rfl
  | _ => rfl

theorem arrivals_append (x y : List Ev) (key : Option (Nat × Nat)) :
    arrivals (x ++ y) key = arrivals x key + arrivals y key := by
  simp [arrivals, List.filter_append]

theorem barKey_barEvent (t s : Nat) (id : Option Nat) (w : Bool) (b k e : Nat) :
    (barEvent t s id w b k e).barKey = some (b, k) := by
  cases id <;> cases w <;> simp [barEvent, Ev.barKey, Ev.arrives, Ev.waits]

theorem arrivals_barEvents (n : Nat) (f : Nat → Nat) (g : Nat → Nat) (id : Option Nat) (b k e : Nat)
    (key : Option (Nat × Nat)) :
    arrivals ((List.range n).map fun l => barEvent (f l) (g l) id false b k e) key =
      if key = some (b, k) then n else 0 := by
  unfold arrivals
  split
  · next h =>
    subst h
    rw [List.filter_eq_self.2]
    · simp
    · intro v hv
      simp only [List.mem_map] at hv
      obtain ⟨l, _, rfl⟩ := hv
      simp [barKey_barEvent, (barEvent_props _ _ id false b k e).2.2.2.2.2.2.1]
  · next h =>
    rw [List.filter_eq_nil_iff.2]
    · rfl
    · intro v hv
      simp only [List.mem_map] at hv
      obtain ⟨l, _, rfl⟩ := hv
      simp [barKey_barEvent, Ne.symm h]

theorem arrivals_waits (n : Nat) (f : Nat → Nat) (g : Nat → Nat) (id : Option Nat) (b k e : Nat)
    (key : Option (Nat × Nat)) :
    arrivals ((List.range n).map fun l => barEvent (f l) (g l) id true b k e) key = 0 := by
  unfold arrivals
  rw [List.filter_eq_nil_iff.2]
  · rfl
  · intro v hv
    simp only [List.mem_map] at hv
    obtain ⟨l, _, rfl⟩ := hv
    simp [(barEvent_props _ _ id true b k e).2.2.2.2.2.2.1]

theorem arrive_bar {m m' : Flat} {slot b n e k : Nat} (h : m.arrive slot b n e = .ok (m', k)) :
    k = (m.bar b).done ∧ ((m.bar b).arrived ≠ 0 → (m.bar b).expected = e) ∧
      (m.bar b).arrived + n ≤ e ∧
      m'.bar b = (if (m.bar b).arrived + n = e then
        ({ done := (m.bar b).done + 1, arrived := 0, expected := e } : Barrier)
        else { done := (m.bar b).done, arrived := (m.bar b).arrived + n, expected := e }) ∧
      ∀ b', b' ≠ b → m'.bar b' = m.bar b' := by
  unfold Flat.arrive at h
  split at h
  · cases h
  · dsimp only at h
    split at h
    · cases h
    · next h₁ =>
      split at h
      · cases h
      · next h₂ =>
        simp only [Except.ok.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        refine ⟨rfl, fun ha => ?_, by omega, ?_, fun b' hb' => ?_⟩
        · exact Classical.byContradiction fun hne => h₁ ⟨ha, hne⟩
        · rw [bar_set, ite_eq_left rfl]
        · rw [bar_set, ite_eq_right hb']

theorem BarInv.arrive {g g' : Ghost} (hb : BarInv g)
    (hbars : ∀ v ∈ g.evs, (v.arrives = true → v.inst ≤ (g.m.bar v.obj).done) ∧
      (v.waits = true → v.inst < (g.m.bar v.obj).done))
    {slot b e k : Nat} {n : Nat} {f₁ f₂ : Nat → Nat} {id : Option Nat}
    (ha : g.m.arrive slot b n e = .ok (g'.m, k))
    (he : g'.evs = g.evs ++ (List.range n).map fun l => barEvent (f₁ l) (f₂ l) id false b k e) :
    BarInv g' := by
  obtain ⟨rfl, hexp, hle, hbar, hother⟩ := arrive_bar ha
  let E := (List.range n).map fun l => barEvent (f₁ l) (f₂ l) id false b (g.m.bar b).done e
  have hE : ∀ v ∈ E, v.barKey = some (b, (g.m.bar b).done) ∧ v.arrives = true ∧ v.count = e ∧ v.obj = b ∧
      v.inst = (g.m.bar b).done := fun v hv => by
    simp only [E, List.mem_map] at hv
    obtain ⟨l, _, rfl⟩ := hv
    obtain ⟨-, -, -, -, h₁, h₂, h₃, -⟩ := barEvent_props (f₁ l) (f₂ l) id false b (g.m.bar b).done e
    exact ⟨barKey_barEvent .., h₃, rfl, h₁, h₂⟩
  have harr : ∀ key, arrivals g'.evs key = arrivals g.evs key + if key = some (b, (g.m.bar b).done) then n else 0 :=
    fun key => by rw [he, arrivals_append, arrivals_barEvents]
  have hkey : ∀ v : Ev, v.barKey ≠ none → v.barKey = some (v.obj, v.inst) := fun v hv => by
    unfold Ev.barKey at hv ⊢; split <;> simp_all
  have hbarK : ∀ v : Ev, v.barKey ≠ none → (v.arrives = true ∨ v.waits = true) := fun v hv => by
    unfold Ev.barKey at hv; split at hv <;> simp_all
  -- An old event of the current instance: an arrival expecting `e`.
  have hcurold : ∀ v ∈ g.evs, v.barKey = some (b, (g.m.bar b).done) → v.arrives = true ∧ v.count = e := by
    intro v hv hk
    have hk' := hkey v (by rw [hk]; simp)
    rw [hk] at hk'
    simp only [Option.some.injEq, Prod.mk.injEq] at hk'
    obtain ⟨hob, hin⟩ := hk'
    obtain ⟨h₁, h₂, -⟩ := hb.cur v hv (by rw [hk]; simp) (by rw [← hob, ← hin])
    refine ⟨h₁, ?_⟩
    rw [h₂, ← hob, hexp]
    -- The instance has an arrival, so its count is positive.
    have := hb.count b
    have hpos : 0 < arrivals g.evs (some (b, (g.m.bar b).done)) := by
      unfold arrivals
      exact List.length_pos_of_mem (List.mem_filter.2 ⟨hv, by simp [h₁, hk]⟩)
    omega
  have hevs : ∀ v ∈ g'.evs, v ∈ g.evs ∨ v ∈ E := fun v hv => by
    rw [he] at hv; exact List.mem_append.1 hv
  have hold_le : ∀ v ∈ g.evs, v.barKey ≠ none → v.inst ≤ (g.m.bar v.obj).done := fun v hv hk => by
    rcases hbarK v hk with h | h
    · exact (hbars v hv).1 h
    · exact Nat.le_of_lt ((hbars v hv).2 h)
  refine ⟨fun v hv w hw hk hvw => ?_, fun v hv hk hlt => ?_, fun v hv hk heq => ?_, fun b' => ?_⟩
  · -- same
    rcases hevs v hv with hv | hv <;> rcases hevs w hw with hw | hw
    · exact hb.same v hv w hw hk hvw
    · rw [(hcurold v hv (hvw.trans (hE w hw).1)).2, (hE w hw).2.2.1]
    · rw [(hcurold w hw (hvw.symm.trans (hE v hv).1)).2, (hE v hv).2.2.1]
    · rw [(hE v hv).2.2.1, (hE w hw).2.2.1]
  · -- done
    rw [harr]
    by_cases hvb : v.obj = b
    · rw [hvb, hbar] at hlt
      by_cases hcomp : (g.m.bar b).arrived + n = e
      · rw [ite_eq_left hcomp] at hlt
        simp only at hlt
        rcases Nat.lt_or_eq_of_le (Nat.le_of_lt_succ hlt) with c | c
        · have hold : v ∈ g.evs := (hevs v hv).resolve_right fun hv' => by
            rw [(hE v hv').2.2.2.2] at c; exact Nat.lt_irrefl _ c
          rw [ite_eq_right (by rw [hkey v hk, hvb]; simp; omega)]
          exact hb.done v hold hk (by rw [hvb]; exact c)
        · rw [ite_eq_left (by rw [hkey v hk, hvb, c])]
          have hc : v.count = e := by
            rcases hevs v hv with hv' | hv'
            · exact (hcurold v hv' (by rw [hkey v hk, hvb, c])).2
            · exact (hE v hv').2.2.1
          rw [hc, ← hcomp]
          have := hb.count b
          rw [show v.barKey = some (b, (g.m.bar b).done) by rw [hkey v hk, hvb, c], this]
      · rw [ite_eq_right hcomp] at hlt
        simp only at hlt
        have hold : v ∈ g.evs := (hevs v hv).resolve_right fun hv' => by
          rw [(hE v hv').2.2.2.2] at hlt; exact Nat.lt_irrefl _ hlt
        rw [ite_eq_right (by rw [hkey v hk, hvb]; simp; omega)]
        exact hb.done v hold hk (by rw [hvb]; exact hlt)
    · have hold : v ∈ g.evs := (hevs v hv).resolve_right fun hv' => hvb (hE v hv').2.2.2.1
      rw [hother _ hvb] at hlt
      rw [ite_eq_right (by rw [hkey v hk]; simp [hvb])]
      exact hb.done v hold hk hlt
  · -- cur
    by_cases hvb : v.obj = b
    · rw [hvb, hbar] at heq ⊢
      by_cases hcomp : (g.m.bar b).arrived + n = e
      · exfalso
        rw [ite_eq_left hcomp] at heq
        simp only at heq
        rcases hevs v hv with hv' | hv'
        · have := hold_le v hv' hk
          rw [hvb] at this; omega
        · rw [(hE v hv').2.2.2.2] at heq; omega
      · rw [ite_eq_right hcomp] at heq ⊢
        simp only at heq ⊢
        have hc : v.arrives = true ∧ v.count = e := by
          rcases hevs v hv with hv' | hv'
          · exact hcurold v hv' (by rw [hkey v hk, hvb, heq])
          · exact ⟨(hE v hv').2.1, (hE v hv').2.2.1⟩
        exact ⟨hc.1, hc.2, by omega⟩
    · have hold : v ∈ g.evs := (hevs v hv).resolve_right fun hv' => hvb (hE v hv').2.2.2.1
      rw [hother _ hvb] at heq ⊢
      exact hb.cur v hold hk heq
  · -- count
    rw [harr]
    by_cases hbb : b' = b
    · subst b'
      rw [hbar]
      by_cases hcomp : (g.m.bar b).arrived + n = e
      · rw [ite_eq_left hcomp]
        simp only [Option.some.injEq, Prod.mk.injEq, true_and, Nat.succ_ne_self, ↓reduceIte,
          Nat.add_zero]
        -- No old arrival is at the next instance.
        have : arrivals g.evs (some (b, (g.m.bar b).done + 1)) = 0 := by
          unfold arrivals
          rw [List.length_eq_zero_iff, List.filter_eq_nil_iff]
          intro v hv hc
          simp only [Bool.and_eq_true, beq_iff_eq] at hc
          have hk' := hkey v (by rw [hc.2]; simp)
          rw [hc.2] at hk'
          simp only [Option.some.injEq, Prod.mk.injEq] at hk'
          have := hold_le v hv (by rw [hc.2]; simp)
          rw [← hk'.1] at this
          omega
        rw [this]
      · rw [ite_eq_right hcomp]
        simp only [↓reduceIte]
        rw [hb.count b]
    · rw [hother _ hbb, ite_eq_right (by simp [hbb])]
      exact hb.count b'

theorem BarInv.wait {g g' : Ghost} (hb : BarInv g) (hm : g'.m = g.m) {b k e : Nat} {n : Nat}
    {f₁ f₂ : Nat → Nat} {id : Option Nat} (hk : k < (g.m.bar b).done)
    (hcnt : 0 < n → ∀ v ∈ g.evs, v.barKey = some (b, k) → v.count = e)
    (hex : 0 < n → ∃ v ∈ g.evs, v.barKey = some (b, k))
    (he : g'.evs = g.evs ++ (List.range n).map fun l => barEvent (f₁ l) (f₂ l) id true b k e) :
    BarInv g' := by
  let E := (List.range n).map fun l => barEvent (f₁ l) (f₂ l) id true b k e
  have hE : ∀ v ∈ E, v.barKey = some (b, k) ∧ v.count = e ∧ v.obj = b ∧ v.inst = k := fun v hv => by
    simp only [E, List.mem_map] at hv
    obtain ⟨l, _, rfl⟩ := hv
    obtain ⟨-, -, -, -, h₁, h₂, -, -⟩ := barEvent_props (f₁ l) (f₂ l) id true b k e
    exact ⟨barKey_barEvent .., rfl, h₁, h₂⟩
  have hnpos : ∀ v ∈ E, 0 < n := fun v hv => by
    simp only [E, List.mem_map, List.mem_range] at hv
    obtain ⟨l, hl, -⟩ := hv; omega
  have harr : ∀ key, arrivals g'.evs key = arrivals g.evs key := fun key => by
    rw [he, arrivals_append, arrivals_waits]; rfl
  have hevs : ∀ v ∈ g'.evs, v ∈ g.evs ∨ v ∈ E := fun v hv => by
    rw [he] at hv; exact List.mem_append.1 hv
  refine ⟨fun v hv w hw hkv hvw => ?_, fun v hv hkv hlt => ?_, fun v hv hkv heq => ?_, fun b' => ?_⟩
  · rcases hevs v hv with hv | hv <;> rcases hevs w hw with hw | hw
    · exact hb.same v hv w hw hkv hvw
    · rw [(hE w hw).2.1]; exact hcnt (hnpos w hw) v hv (hvw.trans (hE w hw).1)
    · rw [(hE v hv).2.1]; exact (hcnt (hnpos v hv) w hw (hvw.symm.trans (hE v hv).1)).symm
    · rw [(hE v hv).2.1, (hE w hw).2.1]
  · rw [harr, hm] at *
    rcases hevs v hv with hv' | hv'
    · exact hb.done v hv' hkv hlt
    · obtain ⟨w, hw', hwk⟩ := hex (hnpos v hv')
      rw [(hE v hv').1, (hE v hv').2.1, ← hwk, ← hcnt (hnpos v hv') w hw' hwk]
      have hwk' : w.barKey = some (w.obj, w.inst) := by
        unfold Ev.barKey at hwk ⊢; split <;> simp_all
      rw [hwk'] at hwk
      simp only [Option.some.injEq, Prod.mk.injEq] at hwk
      exact hb.done w hw' (by rw [hwk', hwk.1, hwk.2]; simp) (by rw [hwk.1, hwk.2]; exact hk)
  · rw [hm] at heq ⊢
    rcases hevs v hv with hv' | hv'
    · exact hb.cur v hv' hkv heq
    · exfalso; rw [(hE v hv').2.2.1, (hE v hv').2.2.2] at heq; omega
  · rw [harr, hm]; exact hb.count b'

/-- A synchronizing step of agent `i`: the shape every barrier and fence step shares. -/
theorem SimInv.sync (hd : P.Disjoint) {S : Sim σ} (h : SimInv P S) {i : Nat} {a : Agent σ}
    (hi : P.agents[i]? = some a) {y : Status σ} {new : List Act} {E : List Ev} {g' : Ghost}
    (hevs : g'.evs = S.g.evs ++ E)
    (hctr' : g'.ctr = bump S.g.ctr (a.issuer i) (List.range a.lanes)) (hbase : g'.base = S.g.base)
    (hq : ∀ j, g'.m.queue j = S.g.m.queue j) (hnext : g'.m.next = S.g.m.next)
    (hstore : ∀ L, g'.m.store? L = S.g.m.store? L)
    (hghost : Inv P.mem P.issuerOf P.slotOf g') (hbars : BarInv g') (hE : ∀ v ∈ E, v.e.isMem = false)
    (hperm : (P.agentEvents a (fun _ => 0) (S.acts i ++ new) ++
        pending P a (S.acts i ++ new) (S.g.m.queue i) y).Perm
      (P.agentEvents a (fun _ => 0) (S.acts i) ++ pending P a (S.acts i) (S.g.m.queue i) (S.st i) ++ E))
    (htr : Tracks P a (S.acts i ++ new) y) (hholds : Holds g' (S.acts i ++ new) i a y)
    (hown : ∀ l < a.lanes, wsum a (S.acts i) (a.tid + l) + extra a (a.tid + l) (S.st i) + 1 =
      wsum a (S.acts i ++ new) (a.tid + l) + extra a (a.tid + l) y)
    (hnew : ∀ act ∈ new, act.plain = true) :
    SimInv P (S.set i y new g') := by
  refine ⟨hghost, ?_, fun j b hj => ?_, fun j hj => ?_, hbars, ?_, plain_set h.plain hnew⟩
  · show g'.evs.Perm _
    rw [hevs]
    refine (h.perm.append_right E).trans (List.Perm.symm (perm_flatMap_upd (i := i) ?_ (fun j hj => ?_)
      (List.getElem?_eq_some_iff.1 hi).1))
    · simp only [Sim.set, upd_same, hi, hq]
      exact hperm
    · simp only [Sim.set, upd_other _ _ hj, hq]
  · by_cases hji : j = i
    · subst hji
      rw [hi] at hj; cases hj
      simp only [Sim.set, upd_same]
      refine ⟨htr, hholds, fun l hl => ?_⟩
      rw [hctr', bump_own _ hl, (h.agents j a hi).2.2 l hl]
      exact hown l hl
    · obtain ⟨htr₁, hho₁, hctr₁⟩ := h.agents j b hj
      simp only [Sim.set, upd_other _ _ hji]
      refine ⟨htr₁, Holds.congr hq hnext hbase hho₁, fun l hl => ?_⟩
      rw [hctr', bump_other P hd hi hj hji _ hl]
      exact hctr₁ l hl
  · have : j ≠ i := fun e => by subst e; simp [List.getElem?_eq_none hj] at hi
    simp only [Sim.set, upd_other _ _ this]
    exact h.beyond j hj
  · exact bounds_sync h.bounds hstore hevs hE

theorem fence_step {g g' : Ghost} {who : Issuer} {lanes : List Nat} {op : SyncOp}
    (h : g.step (.fence who lanes op) = some g') :
    g.m.queue who.slot = [] ∧
      g' = { g with evs := g.evs ++ g.syncEvents who lanes (fenceEvent op), ctr := bump g.ctr who lanes } := by
  simp only [Ghost.step] at h
  split at h
  · cases h
  · split at h
    · next m hf =>
      obtain ⟨hdr, rfl⟩ := fence_ok hf
      simp only [Option.some.injEq] at h
      exact ⟨hdr, h.symm⟩
    · cases h

theorem arrive_step {g g' : Ghost} {who : Issuer} {lanes : List Nat} {id : Option Nat} {b e : Nat}
    (h : g.step (.arrive who lanes id b e) = some g') :
    g.m.arrive who.slot b lanes.length e = .ok (g'.m, (g.m.bar b).done) ∧
      g'.evs = g.evs ++ g.syncEvents who lanes (fun t s => barEvent t s id false b (g.m.bar b).done e) ∧
      g'.ctr = bump g.ctr who lanes ∧ g'.base = g.base := by
  simp only [Ghost.step] at h
  split at h
  · next m k ha =>
    obtain ⟨-, rfl, -⟩ := arrive_ok ha
    simp only [Option.some.injEq] at h
    subst h
    exact ⟨ha, rfl, rfl, rfl⟩
  · cases h

theorem wait_step {g g' : Ghost} {who : Issuer} {lanes : List Nat} {id : Option Nat} {b k e : Nat}
    (h : g.step (.wait who lanes id b k e) = some g') :
    g.m.queue who.slot = [] ∧ k < (g.m.bar b).done ∧
      g' = { g with
        evs := g.evs ++ g.syncEvents who lanes (fun t s => barEvent t s id true b k e)
        ctr := bump g.ctr who lanes } := by
  simp only [Ghost.step] at h
  split at h
  · next m hw =>
    obtain ⟨hdr, hlt, rfl⟩ := wait_ok hw
    simp only [Option.some.injEq] at h
    exact ⟨hdr, hlt, h.symm⟩
  · cases h

theorem arrive_store {m m' : Flat} {slot b n e k : Nat} (h : m.arrive slot b n e = .ok (m', k)) (L : Loc) :
    m'.store? L = m.store? L := by
  obtain ⟨-, -, -, -, hg, hs, -⟩ := arrive_ok h
  exact store?_eq hg hs L

theorem barEvent_mem (t s : Nat) (id : Option Nat) (w : Bool) (b k e : Nat) :
    (barEvent t s id w b k e).e.isMem = false := (barEvent_props t s id w b k e).2.2.1

/-- A lane's arrival events are those `Outcomes` gives the operation. -/
theorem actEvents_arrive (hd : P.Disjoint) {i : Nat} {a : Agent σ} (hi : P.agents[i]? = some a)
    {op : SyncOp} (hop : op = .clusterArrive ∨ ∃ id n, op = .bar id n true) (k : Nat) (c : Nat → Nat) :
    P.actEvents a c (.sync op k) = (List.range a.lanes).map fun l =>
      barEvent (a.tid + l) (c (a.tid + l)) (Sim.barId op) false (op.barObj a.place) k (P.expected a.tid op) := by
  simp only [Program.actEvents, events_arrive hop, ← flatMap_single]
  exact flatMap_ext fun l hl => by
    rw [place_lane hd hi (List.mem_range.1 hl), expected_lane hd hi (List.mem_range.1 hl)]

theorem actEvents_barSync (hd : P.Disjoint) {i : Nat} {a : Agent σ} (hi : P.agents[i]? = some a)
    (id : Nat) (n : Option Nat) (k : Nat) (c : Nat → Nat) :
    P.actEvents a c (.sync (.bar id n false) k) = (List.range a.lanes).flatMap fun l =>
      [barEvent (a.tid + l) (c (a.tid + l)) (some id) false ((SyncOp.bar id n false).barObj a.place) k
          (P.expected a.tid (.bar id n false)),
        barEvent (a.tid + l) (c (a.tid + l) + 1) (some id) true ((SyncOp.bar id n false).barObj a.place) k
          (P.expected a.tid (.bar id n false))] := by
  simp only [Program.actEvents, events_barSync]
  exact flatMap_ext fun l hl => by
    rw [place_lane hd hi (List.mem_range.1 hl), expected_lane hd hi (List.mem_range.1 hl)]

theorem actEvents_clusterWait (hd : P.Disjoint) {i : Nat} {a : Agent σ} (hi : P.agents[i]? = some a)
    (k : Nat) (c : Nat → Nat) :
    P.actEvents a c (.sync .clusterWait k) = (List.range a.lanes).map fun l =>
      barEvent (a.tid + l) (c (a.tid + l)) none true (SyncOp.clusterWait.barObj a.place) k
        (P.expected a.tid .clusterWait) := by
  simp only [Program.actEvents, events_clusterWait, ← flatMap_single]
  exact flatMap_ext fun l hl => by
    rw [place_lane hd hi (List.mem_range.1 hl), expected_lane hd hi (List.mem_range.1 hl)]

theorem issuerOf_eq {i : Nat} {a : Agent σ} (hi : P.agents[i]? = some a) : P.issuerOf i = a.issuer i := by
  simp [Program.issuerOf, hi]

theorem syncOk (hd : P.Disjoint) {i : Nat} {a : Agent σ} (hi : P.agents[i]? = some a) :
    a.issuer i = P.issuerOf (a.issuer i).slot ∧ (List.range a.lanes).Nodup ∧
      ∀ l ∈ List.range a.lanes, P.slotOf ((a.issuer i).tid + l) = (a.issuer i).slot :=
  ⟨(issuerOf_eq P hi).symm, List.nodup_range, fun _ hl => slotOf_lane hd hi (List.mem_range.1 hl)⟩

theorem ctr_run {S : Sim σ} (h : SimInv P S) {i : Nat} {a : Agent σ} (hi : P.agents[i]? = some a)
    {s : σ} (hs : S.st i = .run s) {l : Nat} (hl : l < a.lanes) :
    S.g.ctr (a.tid + l) = wsum a (S.acts i) (a.tid + l) := by
  rw [(h.agents i a hi).2.2 l hl, hs]; rfl

theorem SimInv.fence (hd : P.Disjoint) {S : Sim σ} (h : SimInv P S) {i : Nat} {a : Agent σ}
    (hi : P.agents[i]? = some a) {s s' : σ} {op : SyncOp} {g' : Ghost} (hs : S.st i = .run s)
    (hstep : P.step s = .sync op s') (hf : isFenceOp op = true)
    (hg : S.g.step (.fence (a.issuer i) (List.range a.lanes) op) = some g') :
    SimInv P (S.set i (.run s') [.sync op 0] g') := by
  have hg₀ := hg
  obtain ⟨hdr, rfl⟩ := fence_step hg
  have htr := (h.agents i a hi).1
  rw [hs] at htr
  refine SimInv.sync P hd h hi (E := S.g.syncEvents (a.issuer i) (List.range a.lanes) (fenceEvent op))
    rfl rfl rfl (fun _ => rfl) rfl (fun _ => rfl) (h.ghost.fence (syncOk P hd hi) hg₀)
    (BarInv.append_plain h.bars (fun _ => rfl) rfl fun v hv => ?_) (fun v hv => ?_) ?_
    (fun rest e hr => by simpa using htr _ e (Run.sync hstep hr)) hdr (fun l hl => ?_)
    (by intro act ha; simp only [List.mem_cons, List.not_mem_nil, or_false] at ha <;> subst_vars <;> first | rfl | (cases op <;> simp_all [isFenceOp, Act.plain]) | (rcases hop with rfl | ⟨_, _, rfl⟩ <;> rfl))
  · simp only [syncEvents_eq, List.mem_map] at hv
    obtain ⟨l, _, rfl⟩ := hv
    exact (fenceEvent_mem op _ _).2
  · simp only [syncEvents_eq, List.mem_map] at hv
    obtain ⟨l, _, rfl⟩ := hv
    exact (fenceEvent_mem op _ _).1
  · rw [agentEvents_append, actEvents_fence P hf, hs, syncEvents_eq]
    simp only [pending, List.append_nil]
    refine List.Perm.of_eq ?_
    congr 1
    exact List.map_congr_left fun l hl => by rw [ctr_run P h hi hs (List.mem_range.1 hl), Nat.zero_add]
  · rw [hs, wsum_append, width_sync_own op 0 hl]
    cases op <;> simp_all [isFenceOp, extra]

theorem SimInv.arrive (hd : P.Disjoint) {S : Sim σ} (h : SimInv P S) {i : Nat} {a : Agent σ}
    (hi : P.agents[i]? = some a) {s s' : σ} {op : SyncOp} {g' : Ghost} (hs : S.st i = .run s)
    (hstep : P.step s = .sync op s') (hop : op = .clusterArrive ∨ ∃ id n, op = .bar id n true)
    (hg : S.g.step (.arrive (a.issuer i) (List.range a.lanes) (Sim.barId op) (op.barObj a.place)
      (P.expected a.tid op)) = some g') :
    SimInv P (S.set i (.run s') [.sync op (S.g.m.bar (op.barObj a.place)).done] g') := by
  obtain ⟨ha, hevs, hctr', hbase⟩ := arrive_step hg
  obtain ⟨hdr, -, hq, hn, hgl, hsh, -⟩ := arrive_ok ha
  have htr := (h.agents i a hi).1
  rw [hs] at htr
  rw [List.length_range] at ha
  refine SimInv.sync P hd h hi hevs hctr' hbase (queue_eq hq) hn (store?_eq hgl hsh)
    (h.ghost.arrive (syncOk P hd hi) hg)
    (BarInv.arrive h.bars h.ghost.bars ha (by rw [hevs, syncEvents_eq])) (fun v hv => ?_) ?_
    (fun rest e hr => by simpa using htr _ e (Run.sync hstep hr))
    (show g'.m.queue i = [] by rw [queue_eq hq]; exact hdr) (fun l hl => ?_)
    (by intro act ha; simp only [List.mem_cons, List.not_mem_nil, or_false] at ha <;> subst_vars <;> first | rfl | (cases op <;> simp_all [Act.plain]) | (rcases hop with rfl | ⟨_, _, rfl⟩ <;> rfl))
  · simp only [syncEvents_eq, List.mem_map] at hv
    obtain ⟨l, _, rfl⟩ := hv
    exact barEvent_mem ..
  · rw [agentEvents_append, actEvents_arrive P hd hi hop, hs, syncEvents_eq]
    simp only [pending, List.append_nil]
    refine List.Perm.of_eq ?_
    congr 1
    exact List.map_congr_left fun l hl => by rw [ctr_run P h hi hs (List.mem_range.1 hl), Nat.zero_add]
  · rw [hs, wsum_append, width_sync_own _ _ hl]
    rcases hop with rfl | ⟨id, n, rfl⟩ <;> simp [extra]

theorem SimInv.barArrive (hd : P.Disjoint) {S : Sim σ} (h : SimInv P S) {i : Nat} {a : Agent σ}
    (hi : P.agents[i]? = some a) {s s' : σ} {id : Nat} {n : Option Nat} {g' : Ghost}
    (hs : S.st i = .run s) (hstep : P.step s = .sync (.bar id n false) s')
    (hg : S.g.step (.arrive (a.issuer i) (List.range a.lanes) (some id)
      ((SyncOp.bar id n false).barObj a.place) (P.expected a.tid (.bar id n false))) = some g') :
    SimInv P (S.set i (.bar (.bar id n false)
      (S.g.m.bar ((SyncOp.bar id n false).barObj a.place)).done s') [] g') := by
  obtain ⟨ha, hevs, hctr', hbase⟩ := arrive_step hg
  obtain ⟨hdr, -, hq, hn, hgl, hsh, -⟩ := arrive_ok ha
  have htr := (h.agents i a hi).1
  rw [hs] at htr
  rw [List.length_range] at ha
  refine SimInv.sync P hd h hi hevs hctr' hbase (queue_eq hq) hn (store?_eq hgl hsh)
    (h.ghost.arrive (syncOk P hd hi) hg)
    (BarInv.arrive h.bars h.ghost.bars ha (by rw [hevs, syncEvents_eq])) (fun v hv => ?_) ?_
    (fun rest e hr => by simpa using htr _ e (Run.sync hstep hr))
    ⟨show g'.m.queue i = [] by rw [queue_eq hq]; exact hdr, id, n, rfl⟩ (fun l hl => ?_)
    (by intro act ha; simp only [List.not_mem_nil] at ha <;> subst_vars <;> first | rfl | (cases op <;> simp_all [isFenceOp, Act.plain]) | (rcases hop with rfl | ⟨_, _, rfl⟩ <;> rfl))
  · simp only [syncEvents_eq, List.mem_map] at hv
    obtain ⟨l, _, rfl⟩ := hv
    exact barEvent_mem ..
  · rw [hs, syncEvents_eq, List.append_nil]
    simp only [pending, List.append_nil, Sim.barId]
    refine List.Perm.of_eq ?_
    congr 1
    exact List.map_congr_left fun l hl => by rw [ctr_run P h hi hs (List.mem_range.1 hl)]
  · rw [hs, List.append_nil]; simp [extra]

theorem SimInv.barWait (hd : P.Disjoint) {S : Sim σ} (h : SimInv P S) {i : Nat} {a : Agent σ}
    (hi : P.agents[i]? = some a) {id : Nat} {n : Option Nat} {k : Nat} {s : σ} {g' : Ghost}
    (hs : S.st i = .bar (.bar id n false) k s)
    (hg : S.g.step (.wait (a.issuer i) (List.range a.lanes) (some id)
      ((SyncOp.bar id n false).barObj a.place) k (P.expected a.tid (.bar id n false))) = some g') :
    SimInv P (S.set i (.run s) [.sync (.bar id n false) k] g') := by
  have hg₀ := hg
  obtain ⟨hdr, hlt, rfl⟩ := wait_step hg
  have htr := (h.agents i a hi).1
  rw [hs] at htr
  have hctr : ∀ l < a.lanes, S.g.ctr (a.tid + l) = wsum a (S.acts i) (a.tid + l) + 1 := fun l hl => by
    rw [(h.agents i a hi).2.2 l hl, hs]; rfl
  -- The agent's own arrivals, among the events.
  let b := (SyncOp.bar id n false).barObj a.place
  let e := P.expected a.tid (.bar id n false)
  have hown : ∀ l < a.lanes, barEvent (a.tid + l) (wsum a (S.acts i) (a.tid + l)) (some id) false b k e ∈
      S.g.evs := by
    intro l hl
    rw [h.perm.mem_iff, List.mem_flatMap]
    refine ⟨i, List.mem_range.2 (List.getElem?_eq_some_iff.1 hi).1, ?_⟩
    simp only [hi, hs, pending, Sim.barId]
    exact List.mem_append_right _ (List.mem_map.2 ⟨l, List.mem_range.2 hl, rfl⟩)
  have hbars := BarInv.wait (g' := { S.g with
      evs := S.g.evs ++ S.g.syncEvents (a.issuer i) (List.range a.lanes)
        (fun t s => barEvent t s (some id) true b k e),
      ctr := bump S.g.ctr (a.issuer i) (List.range a.lanes) }) (n := a.lanes)
    (f₁ := fun l => a.tid + l) (f₂ := fun l => S.g.ctr (a.tid + l)) h.bars rfl hlt
    (fun hn v hv hk => by
      have := h.bars.same v hv _ (hown 0 hn) (by rw [hk]; simp) (by rw [hk, barKey_barEvent])
      rw [this]; rfl)
    (fun hn => ⟨_, hown 0 hn, barKey_barEvent ..⟩) rfl
  refine SimInv.sync P hd h hi (E := S.g.syncEvents (a.issuer i) (List.range a.lanes)
      (fun t s => barEvent t s (some id) true b k e))
    rfl rfl rfl (fun _ => rfl) rfl (fun _ => rfl) (h.ghost.wait (syncOk P hd hi) hg₀) hbars
    (fun v hv => ?_) ?_ (fun rest e' hr => by simpa using htr rest e' hr) hdr (fun l hl => ?_)
    (by intro act ha; simp only [List.mem_cons, List.not_mem_nil, or_false] at ha <;> subst_vars <;> first | rfl | (cases op <;> simp_all [isFenceOp, Act.plain]) | (rcases hop with rfl | ⟨_, _, rfl⟩ <;> rfl))
  · simp only [syncEvents_eq, List.mem_map] at hv
    obtain ⟨l, _, rfl⟩ := hv
    exact barEvent_mem ..
  · rw [agentEvents_append, actEvents_barSync P hd hi, hs, syncEvents_eq]
    simp only [pending, List.append_nil, Sim.barId, Nat.zero_add]
    rw [List.append_assoc]
    refine List.Perm.append_left _ ((flatMap_pair _ _ _).trans (List.Perm.of_eq ?_))
    congr 1
    exact List.map_congr_left fun l hl => by rw [hctr l (List.mem_range.1 hl)]
  · rw [hs, wsum_append, width_sync_own _ _ hl]
    simp [extra]

theorem SimInv.clusterWait (hd : P.Disjoint) {S : Sim σ} (h : SimInv P S) {i : Nat} {a : Agent σ}
    (hi : P.agents[i]? = some a) {s s' : σ} {k : Nat} {g' : Ghost} (hs : S.st i = .run s)
    (hstep : P.step s = .sync .clusterWait s')
    (hv : ∃ v ∈ S.g.evs, v.arrives = true ∧ v.e.thread = a.tid ∧
      v.obj = SyncOp.clusterWait.barObj a.place ∧ v.inst = k ∧ v.count = P.expected a.tid .clusterWait)
    (hg : S.g.step (.wait (a.issuer i) (List.range a.lanes) none (SyncOp.clusterWait.barObj a.place) k
      (P.expected a.tid .clusterWait)) = some g') :
    SimInv P (S.set i (.run s') [.sync .clusterWait k] g') := by
  have hg₀ := hg
  obtain ⟨hdr, hlt, rfl⟩ := wait_step hg
  have htr := (h.agents i a hi).1
  rw [hs] at htr
  obtain ⟨v, hv, hva, -, hvo, hvi, hvc⟩ := hv
  have hvk : v.barKey = some (SyncOp.clusterWait.barObj a.place, k) := by
    simp [Ev.barKey, hva, hvo, hvi]
  refine SimInv.sync P hd h hi rfl rfl rfl (fun _ => rfl) rfl (fun _ => rfl)
    (h.ghost.wait (syncOk P hd hi) hg₀)
    (BarInv.wait (n := a.lanes) (f₁ := fun l => a.tid + l)
      (f₂ := fun l => S.g.ctr (a.tid + l)) (id := none) h.bars rfl hlt
      (fun _ w hw hk => by rw [h.bars.same w hw v hv (by rw [hk]; simp) (hk.trans hvk.symm), hvc])
      (fun _ => ⟨v, hv, hvk⟩) rfl) (fun w hw => ?_) ?_
    (fun rest e hr => by simpa using htr _ e (Run.sync hstep hr)) hdr (fun l hl => ?_)
    (by intro act ha; simp only [List.mem_cons, List.not_mem_nil, or_false] at ha <;> subst_vars <;> first | rfl | (cases op <;> simp_all [isFenceOp, Act.plain]) | (rcases hop with rfl | ⟨_, _, rfl⟩ <;> rfl))
  · simp only [syncEvents_eq, List.mem_map] at hw
    obtain ⟨l, _, rfl⟩ := hw
    exact barEvent_mem ..
  · rw [agentEvents_append, actEvents_clusterWait P hd hi, hs, syncEvents_eq]
    simp only [pending, List.append_nil]
    refine List.Perm.of_eq ?_
    congr 1
    exact List.map_congr_left fun l hl => by rw [ctr_run P h hi hs (List.mem_range.1 hl), Nat.zero_add]
  · rw [hs, wsum_append, width_sync_own _ _ hl]
    simp [extra]

/-! ## Every step keeps the invariant -/

theorem SimInv.step (hd : P.Disjoint) {S S' : Sim σ} (h : SimInv P S) (hs : Sim.Step (P := P) S S')
    (hu : Uniform S'.g.evs) : SimInv P S' := by
  have tr : ∀ {i a s}, P.agents[i]? = some a → S.st i = .run s →
      ∀ rest e, Run P.step s rest e → Run P.step a.init (S.acts i ++ rest) e := fun hi hst => by
    have := (h.agents _ _ hi).1; rw [hst] at this; exact this
  cases hs with
  | next hi hst hstep =>
    exact h.local P hi hst rfl fun rest e hr => tr hi hst rest e (Run.next hstep hr)
  | reg hi hst hstep =>
    exact h.local P hi hst rfl fun rest e hr => tr hi hst rest e (Run.reg hstep hr)
  | issue hi hst hstep hl hg => exact h.issue P hd hi hst hstep hl hg
  | deliver hst hdl hg => exact h.deliver P hst hdl hg hu
  | complete hst hf => exact h.complete P hst hf
  | fence hi hst hstep hf hg => exact h.fence P hd hi hst hstep hf hg
  | arrive hi hst hstep hop hg => exact h.arrive P hd hi hst hstep hop hg
  | barArrive hi hst hstep hg => exact h.barArrive P hd hi hst hstep hg
  | barWait hi hst hg => exact h.barWait P hd hi hst hg
  | clusterWait hi hst hstep hv hg => exact h.clusterWait P hd hi hst hstep hv hg
  | exit hi hst hstep =>
    have := tr hi hst [] .exit (Run.exit hstep)
    rw [List.append_nil] at this
    exact h.local P hi hst rfl this
  | fault hi hst hstep =>
    have := tr hi hst [] (.fault _) (Run.fault hstep)
    rw [List.append_nil] at this
    exact h.local P hi hst rfl this

theorem Sim.step_evs {S S' : Sim σ} (hs : Sim.Step (P := P) S S') : ∃ es, S'.g.evs = S.g.evs ++ es := by
  cases hs with
  | issue _ _ _ _ hg | deliver _ _ hg | fence _ _ _ _ hg | arrive _ _ _ _ hg | barArrive _ _ _ hg
  | barWait _ _ hg | clusterWait _ _ _ _ hg => exact Flat.step_evs hg
  | _ => exact ⟨[], by simp [Sim.set]⟩

theorem uniform_prefix {x y : List Ev} (h : Uniform (x ++ y)) : Uniform x :=
  fun a ha b hb => h a (List.mem_append_left _ ha) b (List.mem_append_left _ hb)

theorem reach_inv (hd : P.Disjoint) {m : Flat} (h₀ : SimInv P (Sim.init P m)) {S : Sim σ}
    (hr : Sim.Reachable P m S) (hu : Uniform S.g.evs) : SimInv P S := by
  induction hr with
  | init => exact h₀
  | step _ hs ih =>
    obtain ⟨es, hes⟩ := Sim.step_evs P hs
    exact (ih (uniform_prefix (hes ▸ hu))).step P hd hs hu

/-! ## Each thread's events in program order

Agents block on their requests, so a thread's events are performed in its program order: a
request's events after everything its thread did before, and nothing of its thread after them
until it completes. Dependencies (`Program.SemDep`) therefore run forward in the performed order. -/

/-- Each thread's events in program order. -/
def Ordered (evs : List Ev) : Prop :=
  ∀ i j : Fin evs.length, evs[i].e.thread = evs[j].e.thread → evs[i].seq < evs[j].seq → i < j

theorem ordered_append {evs es : List Ev} (h : Ordered evs)
    (hnew : ∀ a b : Fin es.length, es[a].e.thread = es[b].e.thread → es[a].seq < es[b].seq → a < b)
    (hcross : ∀ e ∈ evs, ∀ f ∈ es, e.e.thread = f.e.thread → e.seq < f.seq) : Ordered (evs ++ es) := by
  intro i j ht hs
  rcases Performed.fin_append i with ⟨hi, ei⟩ | ⟨hi, li, ei⟩ <;>
    rcases Performed.fin_append j with ⟨hj, ej⟩ | ⟨hj, lj, ej⟩ <;> rw [ei, ej] at ht hs
  · exact h ⟨i, hi⟩ ⟨j, hj⟩ ht hs
  · show i.val < j.val; omega
  · exfalso
    have := hcross _ (List.getElem_mem hj) _ (List.getElem_mem hi) ht.symm
    omega
  · have := hnew ⟨_, hi⟩ ⟨_, hj⟩ ht hs
    show i.val < j.val
    have : i.val - evs.length < j.val - evs.length := this
    omega

theorem ordered_sync {g : Ghost} (hi : Inv P.mem P.issuerOf P.slotOf g) {evs : List Ev}
    (h : Ordered evs) (he : evs = g.evs) (who : Issuer) (n : Nat) (mk : Nat → Nat → Ev)
    (ht : ∀ t s, (mk t s).e.thread = t) (hseq : ∀ t s, (mk t s).seq = s) :
    Ordered (evs ++ g.syncEvents who (List.range n) mk) := by
  subst he
  refine ordered_append h (fun a b hab hs => ?_) (fun e he f hf het => ?_)
  · exfalso
    obtain ⟨a, ha⟩ := a; obtain ⟨b, hb⟩ := b
    have hl : (g.syncEvents who (List.range n) mk).length = n := by simp [Ghost.syncEvents]
    have get : ∀ k (hk : k < (g.syncEvents who (List.range n) mk).length),
        (g.syncEvents who (List.range n) mk)[k] = mk (who.tid + k) (g.ctr (who.tid + k)) := by
      intro k hk; simp [Ghost.syncEvents]
    simp only [Fin.getElem_fin, get, ht, hseq] at hab hs
    have : a = b := by omega
    subst this
    exact Nat.lt_irrefl _ hs
  · simp only [Ghost.syncEvents, List.mem_map, List.mem_range] at hf
    obtain ⟨l, -, rfl⟩ := hf
    rw [ht] at het
    rw [hseq, ← het]
    exact hi.seq_lt e he

theorem laneEvents_seq (r : Req) (lane t base : Nat) (x : Loc) (bytes : List UInt8) :
    ∀ f ∈ r.laneEvents lane t base x bytes, f.e.thread = t ∧ base ≤ f.seq := by
  intro f hf
  obtain ⟨⟨p, k⟩, hk, rfl⟩ := List.mem_map.1 hf
  simp [Req.event]

theorem laneEvents_sorted (r : Req) (lane t base : Nat) (x : Loc) (bytes : List UInt8) :
    ∀ a b : Fin (r.laneEvents lane t base x bytes).length,
      (r.laneEvents lane t base x bytes)[a].seq < (r.laneEvents lane t base x bytes)[b].seq → a < b := by
  intro ⟨a, ha⟩ ⟨b, hb⟩ h
  have get : ∀ k (hk : k < (r.laneEvents lane t base x bytes).length),
      (r.laneEvents lane t base x bytes)[k].seq = base + k := by
    intro k hk; simp [Req.laneEvents]
  simp only [Fin.getElem_fin, get] at h
  show a < b
  omega

/-- **Reachable states perform each thread's events in program order.** -/
theorem reach_ordered (hd : P.Disjoint) {m : Flat} (h₀ : SimInv P (Sim.init P m)) {S : Sim σ}
    (hr : Sim.Reachable P m S) (hu : Uniform S.g.evs) : Ordered S.g.evs := by
  induction hr with
  | init => intro i; exact absurd i.isLt (by simp [Sim.init, Ghost.init])
  | @step S S' hreach hs ih =>
    obtain ⟨es, hes⟩ := Sim.step_evs P hs
    have hu₀ := uniform_prefix (hes ▸ hu)
    have hS := reach_inv P hd h₀ hreach hu₀
    have ho := ih hu₀
    cases hs with
    | next | reg | complete | exit | fault => simpa [Sim.set] using ho
    | issue _ _ _ _ hg =>
      obtain ⟨-, -, he, -⟩ := issue_step hg
      show Ordered (S.set _ _ _ _).g.evs
      simpa only [Sim.set, he] using ho
    | @deliver i id r s wb got l m bytes g' hst hdl hg =>
      obtain ⟨a, hi⟩ := agent_of_st P hS (.inr ⟨_, _, _, _, _, hst⟩)
      obtain ⟨-, hho, hctr⟩ := hS.agents i a hi
      rw [hst] at hho hctr
      obtain ⟨-, hlanes, hqs, hbase, -⟩ := hho
      obtain ⟨k, hk, x, -, -, -, hkid, hlane, -, -, -, -, -, -, -, rfl⟩ := deliver_facts hg
      show Ordered (S.g.evs ++ _)
      generalize hq : (S.g.m.queue i)[k] = q at *
      have hqin : q ∈ S.g.m.queue i := hq ▸ List.getElem_mem hk
      obtain ⟨-, hqwho, -, hqreq⟩ := hqs q hqin
      obtain ⟨-, -, -, -, hql⟩ := hS.ghost.queued i q hqin
      have hact : r.active l = true := hqreq ▸ (hql l hlane).1
      have hsep := (hql l hlane).2.2.2
      have htid : q.who.tid = a.tid := by rw [hqwho]; rfl
      subst hkid
      have hb : S.g.base q.id l = wsum a (S.acts i) (a.tid + l) := hbase l hact
      have hc := hctr l (hlanes l hact)
      have hw : Act.width a (a.tid + l) (.mem r got) = r.pieces.length := by
        simp [Act.width, hact]
      refine ordered_append ho (fun a b _ h => laneEvents_sorted _ _ _ _ _ _ a b h) (fun e he f hf het => ?_)
      obtain ⟨hft, hfs⟩ := laneEvents_seq _ _ _ _ _ _ f hf
      rw [hft] at het
      have hlt := hS.ghost.seq_lt e he
      rw [het, htid, hc] at hlt
      rcases hsep e he het with h' | h'
      · omega
      · simp only [extra, hw] at hlt
        rw [hb, hqreq] at h'
        omega
    | fence _ _ _ _ hg =>
      obtain ⟨-, rfl⟩ := fence_step hg
      exact ordered_sync P hS.ghost ho rfl _ _ _ (fun _ _ => rfl) (fun _ _ => rfl)
    | arrive _ _ _ _ hg =>
      obtain ⟨-, he, -, -⟩ := arrive_step hg
      show Ordered (S.set _ _ _ _).g.evs
      simp only [Sim.set, he]
      exact ordered_sync P hS.ghost ho rfl _ _ _ (fun _ _ => rfl) (fun _ _ => rfl)
    | barArrive _ _ _ hg =>
      obtain ⟨-, he, -, -⟩ := arrive_step hg
      show Ordered (S.set _ _ _ _).g.evs
      simp only [Sim.set, he]
      exact ordered_sync P hS.ghost ho rfl _ _ _ (fun _ _ => rfl) (fun _ _ => rfl)
    | barWait _ _ hg =>
      obtain ⟨-, -, rfl⟩ := wait_step hg
      exact ordered_sync P hS.ghost ho rfl _ _ _ (fun _ _ => rfl) (fun _ _ => rfl)
    | clusterWait _ _ _ _ hg =>
      obtain ⟨-, -, rfl⟩ := wait_step hg
      exact ordered_sync P hS.ghost ho rfl _ _ _ (fun _ _ => rfl) (fun _ _ => rfl)

/-! ## Outcomes -/

theorem barKey_some {v : Ev} (h : v.barKey ≠ none) : v.barKey = some (v.obj, v.inst) ∧
    (v.arrives = true ∨ v.waits = true) := by
  unfold Ev.barKey at h ⊢
  split <;> simp_all

theorem barriersOk {g : Ghost} (hb : BarInv g)
    (hbars : ∀ v ∈ g.evs, (v.arrives = true → v.inst ≤ (g.m.bar v.obj).done) ∧
      (v.waits = true → v.inst < (g.m.bar v.obj).done)) : BarriersOk g.evs := by
  intro v hv hk
  show arrivals g.evs v.barKey ≤ v.count ∧ (v.waits = true → arrivals g.evs v.barKey = v.count)
  obtain ⟨hkey, haw⟩ := barKey_some hk
  have hle : v.inst ≤ (g.m.bar v.obj).done := by
    rcases haw with h | h
    · exact (hbars v hv).1 h
    · exact Nat.le_of_lt ((hbars v hv).2 h)
  rcases Nat.lt_or_eq_of_le hle with c | c
  · have := hb.done v hv hk c
    exact ⟨Nat.le_of_eq this, fun _ => this⟩
  · obtain ⟨ha, hc, hlt⟩ := hb.cur v hv hk c
    have hcnt := hb.count v.obj
    rw [hkey, c, hcnt, hc]
    refine ⟨Nat.le_of_lt hlt, fun hw => ?_⟩
    simp [Ev.arrives, Ev.waits] at ha hw
    split at ha <;> simp_all

theorem mem_zip_range {α β : Type} {l : List α} {f : Nat → β} {p : α × β}
    (h : p ∈ l.zip ((List.range l.length).map f)) : ∃ i, ∃ hi : i < l.length, p = (l[i], f i) := by
  obtain ⟨k, hk, rfl⟩ := List.mem_iff_getElem.1 h
  simp only [List.length_zip, List.length_map, List.length_range, Nat.min_self] at hk
  exact ⟨k, hk, by simp [List.getElem_zip]⟩

theorem flatMap_zip_range {α β γ : Type} (g : α × β → List γ) :
    ∀ (l : List α) (f : Nat → β), (l.zip ((List.range l.length).map f)).flatMap g =
      (List.range l.length).flatMap fun i => match l[i]? with
        | some a => g (a, f i)
        | none => []
  | [], _ => rfl
  | a :: l, f => by
    rw [List.length_cons, List.range_succ_eq_map, List.map_cons, List.map_map, List.zip_cons_cons,
      List.flatMap_cons, List.flatMap_cons, List.flatMap_map]
    rw [flatMap_zip_range g l (f ∘ Nat.succ)]
    rfl

/-- The events of the runs a final state records. -/
theorem events_runs {S : Sim σ} (h : SimInv P S) (E : Status σ → End)
    (hidle : ∀ i, (S.st i).idle = true) :
    S.g.evs.Perm (P.events ((List.range P.agents.length).map fun i => (S.acts i, E (S.st i)))) := by
  refine h.perm.trans (List.Perm.of_eq ?_)
  unfold Program.events
  rw [flatMap_zip_range]
  exact flatMap_ext fun i hi => by
    have hi' := List.mem_range.1 hi
    rw [List.getElem?_eq_getElem hi']
    have : pending P P.agents[i] (S.acts i) (S.g.m.queue i) (S.st i) = [] := by
      have := hidle i
      cases hst : S.st i <;> simp_all [Status.idle, pending]
    dsimp only
    rw [Program.asyncEvents_plain _ _ _ _ _ (h.plain i)]
    simp [this]

theorem covers_of_key {v w : Ev} (hv : v.e.isWrite = true) (hw : w.e.isWrite = true)
    (hk : v.key = w.key) (L : Loc) : v.covers L = w.covers L := by
  simp only [Ev.key, Prod.mk.injEq] at hk
  simp [Ev.covers, hv, hw, hk.1, hk.2]

/-- The memory a run leaves is the last write to each byte, or the program's initial byte. -/
theorem final_spec {S : Sim σ} (h : SimInv P S) (L : Loc) :
    (∃ i : Fin S.g.evs.length, S.g.evs[i].covers L = true ∧ (∀ j, coOf S.g.evs i j = false) ∧
      Sim.final P S L = S.g.evs[i].val.getD (L.addr - S.g.evs[i].loc.addr) 0) ∨
    ((∀ i : Fin S.g.evs.length, S.g.evs[i].covers L = false) ∧ Sim.final P S L = P.mem L) := by
  by_cases hin : InBounds S.g.m L
  · have hfin : Sim.final P S L = byteAt S.g.m L := by simp [Sim.final, hin]
    rcases h.ghost.store L hin with ⟨j, h₁, h₂, h₃⟩ | ⟨h₁, h₂⟩
    · refine .inl ⟨j, h₁, fun j' => ?_, hfin.trans h₃⟩
      cases hc : coOf S.g.evs j j'
      · rfl
      · exfalso
        simp only [coOf, Rel.get_ofFn, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hc
        obtain ⟨⟨⟨hw, hw'⟩, hk⟩, hlt⟩ := hc
        have := h₂ j' hlt
        rw [← covers_of_key hw hw' hk L, h₁] at this
        cases this
    · exact .inr ⟨h₁, hfin.trans h₂⟩
  · refine .inr ⟨fun i => ?_, by simp [Sim.final, hin]⟩
    cases hc : S.g.evs[i].covers L
    · rfl
    · exact absurd (h.bounds _ (List.getElem_mem _) L hc) hin

theorem Status.idle_done : (Status.done : Status σ).idle = true := rfl

/-- The reference machine's runs join no collective. -/
theorem collectsOk_plain {S : Sim σ} (hp : ∀ i, ∀ act ∈ S.acts i, act.plain = true) (e : Nat → End) :
    P.CollectsOk ((List.range P.agents.length).map fun i => (S.acts i, e i)) := by
  have hnil : P.joined ((List.range P.agents.length).map fun i => (S.acts i, e i)) = [] := by
    simp only [Program.joined, List.flatMap_eq_nil_iff, List.filterMap_eq_nil_iff]
    intro x hx act hact
    obtain ⟨⟨a, acts, en⟩, rfl⟩ : ∃ y, x = y := ⟨x, rfl⟩
    obtain ⟨i, -, hi⟩ := List.mem_map.1 (List.of_mem_zip hx).2
    simp only [Prod.mk.injEq] at hi
    obtain ⟨rfl, -⟩ := hi
    have := hp i act hact
    cases act <;> simp_all [Act.plain]
  intro j hj
  rw [hnil] at hj; cases hj

/-- A performed list whose threads' events are in program order stays consistent with every read
ordered before its thread's later events as a dependency. -/
theorem consistent_readPo {evs : List Ev} {init : Loc → UInt8} (hp : Performed evs init)
    (ho : Ordered evs) (place : Nat → NV.PTX.Place) :
    (NV.PTX.Execution.Timed.withDep (Performed.exec evs init place) (readPo evs)).consistent = true := by
  have hr : ∀ a b, readPo evs a b = true → evs[a].e.thread = evs[b].e.thread ∧ evs[a].seq < evs[b].seq := by
    intro a b h
    simp only [readPo, NV.Rel.get_ofFn, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
    exact ⟨h.1.2, h.2⟩
  refine (hp.timed place).consistent_withDep (fun a b h => ?_) (fun a b h => Performed.po_iff.2 (hr a b h))
    hp.poWF hp.rfWF hp.rmwWF Performed.barWF
  exact ho a b (hr a b h).1 (hr a b h).2

/-- **A finished run's memory is an outcome.** From a fresh machine whose bytes are the program's
initial memory, every run of the reference machine in which every agent exits leaves a memory that
is one of the program's outcomes, as long as its accesses did not overlap partially. -/
theorem done_outcome (hd : P.Disjoint) {m : Flat} (hq : ∀ s, m.queue s = []) (hs : StoresOk m)
    (hbars : m.bars = #[]) (hm : ∀ L, InBounds m L → byteAt m L = P.mem L)
    (hf : ∀ L : Loc, L.space = .shared → m.shared.size ≤ L.owner → P.mem L = 0) {S : Sim σ}
    (hr : Sim.Reachable P m S) (hdone : ∀ i < P.agents.length, S.st i = .done)
    (hu : Uniform S.g.evs) : Outcomes P (.done (Sim.final P S)) := by
  have h := reach_inv P hd (simInv_init P hq hs hbars hm hf) hr hu
  have hdone' : ∀ i, S.st i = .done := fun i => by
    rcases Nat.lt_or_ge i P.agents.length with c | c
    · exact hdone i c
    · exact (h.beyond i c).1
  have hidle : ∀ i, (S.st i).idle = true := fun i => by rw [hdone' i]; rfl
  refine ⟨{ runs := (List.range P.agents.length).map fun i => (S.acts i, End.exit),
            length := by simp
            runs_ok := ?_
            evs := S.g.evs
            perm := ?_
            rf := rfOf S.g.evs, co := coOf S.g.evs, sc := scOf S.g.evs
            collects := collectsOk_plain P h.plain _
            dep := readPo S.g.evs, dep_ok := fun _ _ hd => semDep_readPo P hd }, ⟨?_, ?_⟩, ?_, ?_⟩
  · intro p hp
    obtain ⟨i, hi', rfl⟩ := mem_zip_range hp
    have := (h.agents i _ (List.getElem?_eq_getElem hi')).1
    rw [hdone' i] at this
    exact this
  · have := events_runs P h (fun _ => End.exit) hidle
    simpa using this
  · exact barriersOk h.bars h.ghost.bars
  · intro ⟨re, hre, hstop⟩
    obtain ⟨i, _, rfl⟩ := List.mem_map.1 hre
    cases hstop
  · exact consistent_readPo h.ghost.perf (reach_ordered P hd (simInv_init P hq hs hbars hm hf) hr hu) P.place
  · refine ⟨fun re hre => ?_, hu, fun L => final_spec P h L⟩
    obtain ⟨i, _, rfl⟩ := List.mem_map.1 hre
    rfl

/-- How a run that has stopped ends, by its agent's status. -/
def Status.ending : Status σ → End
  | .done => .exit
  | .fault msg => .fault msg
  | _ => .stop

/-- **A fault is an outcome.** A run in which an agent faulted, with no agent waiting on a request
or a barrier, shows a fault outcome of the program. -/
theorem fault_outcome (hd : P.Disjoint) {m : Flat} (hq : ∀ s, m.queue s = []) (hs : StoresOk m)
    (hbars : m.bars = #[]) (hm : ∀ L, InBounds m L → byteAt m L = P.mem L)
    (hf : ∀ L : Loc, L.space = .shared → m.shared.size ≤ L.owner → P.mem L = 0) {S : Sim σ}
    (hr : Sim.Reachable P m S) (hu : Uniform S.g.evs) (hidle : ∀ i, (S.st i).idle = true)
    {i : Nat} {msg : String} (hi : S.st i = .fault msg) : Outcomes P (.fault msg) := by
  have h := reach_inv P hd (simInv_init P hq hs hbars hm hf) hr hu
  have hin : i < P.agents.length := by
    rcases Nat.lt_or_ge i P.agents.length with c | c
    · exact c
    · rw [(h.beyond i c).1] at hi; cases hi
  have hmem : (S.acts i, End.fault msg) ∈
      (List.range P.agents.length).map fun j => (S.acts j, (S.st j).ending) :=
    List.mem_map.2 ⟨i, List.mem_range.2 hin, by rw [hi]; rfl⟩
  refine ⟨{ runs := (List.range P.agents.length).map fun j => (S.acts j, (S.st j).ending),
            length := by simp
            runs_ok := ?_
            evs := S.g.evs
            perm := ?_
            rf := rfOf S.g.evs, co := coOf S.g.evs, sc := scOf S.g.evs
            collects := collectsOk_plain P h.plain _
            dep := readPo S.g.evs, dep_ok := fun _ _ hd => semDep_readPo P hd }, ⟨?_, ?_⟩, ?_, ?_⟩
  · intro p hp
    obtain ⟨j, hj', rfl⟩ := mem_zip_range hp
    have htr := (h.agents j _ (List.getElem?_eq_getElem hj')).1
    have := hidle j
    show Run P.step P.agents[j].init (S.acts j) (S.st j).ending
    cases hst : S.st j with
    | run s => rw [hst] at htr; simpa [Status.ending] using htr [] .stop Run.stop
    | done => rw [hst] at htr; exact htr
    | fault msg' => rw [hst] at htr; exact htr
    | mem id r s wb got => rw [hst] at this; cases this
    | bar op k s => rw [hst] at this; cases this
  · exact events_runs P h _ hidle
  · exact barriersOk h.bars h.ghost.bars
  · intro _
    exact ⟨_, hmem, by simp, by simp⟩
  · exact consistent_readPo h.ghost.perf (reach_ordered P hd (simInv_init P hq hs hbars hm hf) hr hu) P.place
  · exact ⟨_, hmem, rfl⟩


end NV.Formal
