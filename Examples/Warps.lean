import NV.PTX.Async.MBarrier
import NV.PTX.NamedBarrier
import NV.Formal.System
import NV.Formal.Explore

/-!
# Warps running straight-line synchronization code

A CTA whose warps each run a fixed instruction list against the CTA's named barriers
(`NamedBarrier`), its mbarriers (`MBarrier`) and tagged shared-memory slots. A step lets one warp
execute its next instruction, or leave a wait that has been released. Faults are the barriers'
undefined conditions and loads that find another tag than the one the program expects (a read
before the intended write is visible, or after it was overwritten).

The model fits the join and hand-off protocols whose control flow does not depend on data
(kernel prologues, helper hand-offs); a warp is one participant (its lanes act together, as
`.aligned` barriers require).
-/

namespace Examples.Warps

open NV NV.Formal

open NV.PTX

inductive Instr where
  /-- `bar.sync id, n`. -/
  | barSync (id n : Nat)
  /-- `bar.arrive id, n`. -/
  | barArrive (id n : Nat)
  /-- `mbarrier.arrive` (count 1, `.release.cta`) on mbarrier `b`. -/
  | arrive (b : Nat)
  /-- `mbarrier.try_wait.parity` until phase `k` of mbarrier `b` has completed. -/
  | wait (b : Nat) (k : Int)
  /-- A store of data tagged `tag` into shared slot `slot`. -/
  | store (slot tag : Nat)
  /-- A load of shared slot `slot` that needs the data tagged `tag`. -/
  | load (slot tag : Nat)
  deriving DecidableEq, Repr, Hashable

inductive Fault where
  | barrier (f : NamedBarrier.Fault)
  | mbarrier (f : MBarrier.Fault)
  /-- A load found tag `found` (or nothing) where the program needs `tag`. -/
  | stale (slot tag : Nat) (found : Option Nat)
  deriving DecidableEq, Repr, Hashable

structure Program where
  warps : List (List Instr)
  /-- `mbarrier.init` counts. -/
  mbarriers : List Nat

structure State where
  pc : List Nat
  /-- A warp in `bar.sync`: the barrier and the instance it arrived in. -/
  blocked : List (Option (Nat × Nat))
  bars : NamedBarriers
  mbars : List MBarrier
  slots : Nat → Option Nat
  fault : Option Fault := none

variable (prog : Program)

def init : State where
  pc := prog.warps.map fun _ => 0
  blocked := prog.warps.map fun _ => none
  bars := {}
  mbars := prog.mbarriers.map fun c => ⟨0, c, c, 0, true⟩
  slots _ := none

def faulted (s : State) (f : Fault) : Option State := some { s with fault := some f }

/-- Warp `w` executes its next instruction, or leaves `bar.sync` once its instance completed. -/
def step (s : State) (w : Nat) : Option State :=
  if s.fault.isSome ∨ prog.warps.length ≤ w then none else
  let advance (t : State) : State := { t with pc := t.pc.set w (t.pc.getD w 0 + 1) }
  match s.blocked.getD w none with
  | some (id, g) =>
    if h : id < 16 then
      if (s.bars.bars ⟨id, h⟩).released g then advance { s with blocked := s.blocked.set w none }
      else none
    else none
  | none =>
    match (prog.warps.getD w [])[s.pc.getD w 0]? with
    | none => none
    | some (.barSync id n) =>
      match s.bars.arrive id w n with
      | .error f => faulted s (.barrier f)
      | .ok (bars, a) =>
        if a.completed.isSome then advance { s with bars }
        else some { s with bars, blocked := s.blocked.set w (some (id, a.instance_)) }
    | some (.barArrive id n) =>
      match s.bars.arrive id w n with
      | .error f => faulted s (.barrier f)
      | .ok (bars, _) => advance { s with bars }
    | some (.arrive b) =>
      match (s.mbars.getD b ⟨0, 0, 0, 0, true⟩).arrive 1 with
      | .error f => faulted s (.mbarrier f)
      | .ok m => advance { s with mbars := s.mbars.set b m }
    | some (.wait b k) =>
      let m := s.mbars.getD b ⟨0, 0, 0, 0, true⟩
      match m.waitPhase k with
      | .error f => faulted s (.mbarrier f)
      | .ok false => none
      | .ok true => advance { s with mbars := s.mbars.set b m.observe }
    | some (.store slot tag) =>
      advance { s with slots := fun x => if x = slot then some tag else s.slots x }
    | some (.load slot tag) =>
      if s.slots slot = some tag then advance s else faulted s (.stale slot tag (s.slots slot))

def system : System State Nat := ⟨init prog, step prog⟩

theorem step_cover {s t : State} {w : Nat} (h : step prog s w = some t) :
    w ∈ List.range prog.warps.length := by
  simp only [List.mem_range]
  unfold step at h
  split at h
  · cases h
  · rename_i hw; simp only [not_or] at hw; omega

def done (s : State) : Bool :=
  (List.range prog.warps.length).all fun w => s.pc.getD w 0 == (prog.warps.getD w []).length

/-- Every state is fault-free and either finished or able to move. -/
def healthy (s : State) : Bool := s.fault.isNone && (done prog s ||
    (List.range prog.warps.length).any fun w => (step prog s w).isSome)

/-- No warp can move, yet some has not finished: the CTA hangs. -/
def deadlocked (s : State) : Bool :=
  s.fault.isNone && !done prog s &&
    (List.range prog.warps.length).all fun w => (step prog s w).isNone

def key (s : State) : List Nat × List (Option (Nat × Nat)) × List MBarrier ×
    List (Option Nat × Option Nat × Nat) × List (Option Nat) × Option Fault :=
  (s.pc, s.blocked, s.mbars,
   (List.range 16).map (fun i => if h : i < 16 then
     let b := s.bars.bars ⟨i, h⟩; (b.count, some b.arrived, b.generation) else (none, none, 0)),
   (List.range 64).map s.slots, s.fault)

/-- The shortest interleaving that faults or deadlocks, if one exists within `limit` states.
The key reads slots below 64, so programs use slots below 64. -/
def counterexample (limit : Nat := 1000000) : Option (List Nat) :=
  (system prog).search (fun _ => List.range prog.warps.length) (key) (fun s => s.fault.isSome || deadlocked prog s) limit

end Examples.Warps
