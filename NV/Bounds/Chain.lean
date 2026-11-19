import NV.Bounds.Rate

/-!
# Schedules of a fixed instruction stream

A stream is a list of instructions (a straight-line SASS region, or the dynamic instructions of a
kernel's warps), each with the producers it waits for and the resources it holds. A schedule
gives each instruction an issue cycle. The machine admits a schedule only if

* it respects the dependences: an instruction issues at least its producer's latency after the
  producer (`Respects`). This is the dependent-latency chain: a value does not exist before its
  producer's latency has elapsed; the engine writes a result back `latency` cycles after issue,
  and the stall counts and dependence counters hold the consumer until then;
* each resource serves its instructions one after another (`Serves`): a sub-partition's issue
  slot one per cycle, a pipe one per interval, as the engine's ports do (`serveAll_chained`).

`Stream.bound_le`: every such schedule that finishes by `T` has `T ≥ Stream.bound`, the larger of
the dependence DAG's critical path (`critical`) and the busiest resource's work (`load`). So no
issue order of the stream, under any scheduler policy, beats the bound; that is the class
optimality statement for a fixed SASS stream, its class being "this stream, these latencies and
intervals, any schedule".
-/

namespace NV.Bounds

open NV.Core

/-- One instruction of a stream. -/
structure Node where
  /-- The instructions it waits for, by index, each with the cycles it must issue after them. -/
  preds : List (Nat × Nat) := []
  /-- Cycles from its issue until the stream is done with it: its result written, or its effect
  complete. -/
  done : Nat := 1
  /-- The resources it holds, by identifier, each with the cycles it holds it (its
  sub-partition's issue slot for one cycle, its pipe for the pipe's interval). -/
  uses : List (Nat × Nat) := []
  deriving Repr, Inhabited

abbrev Stream := Array Node

namespace Stream

def preds (g : Stream) (j : Nat) : List (Nat × Nat) := (g[j]?.map (·.preds)).getD []

def done (g : Stream) (j : Nat) : Nat := (g[j]?.map (·.done)).getD 0

/-- Cycles instruction `j` holds resource `r`. -/
def cost (g : Stream) (r j : Nat) : Nat :=
  (((g[j]?.map (·.uses)).getD []).filter (·.1 == r)).foldl (fun a u => a + u.2) 0

/-- The instructions that hold resource `r`. -/
def users (g : Stream) (r : Nat) : List Nat := (List.range g.size).filter fun j => g.cost r j != 0

/-- The cycles of work the stream puts on resource `r`. -/
def load (g : Stream) (r : Nat) : Nat := ((g.users r).map (g.cost r)).sum

/-- The longest path into the next instruction, over its producers among the first `d.size`. -/
def pathStep (d : Array Nat) (ps : List (Nat × Nat)) : Nat :=
  ps.foldl (fun m p => if p.1 < d.size then max m (d[p.1]?.getD 0 + p.2) else m) 0

def depthsFrom (g : Stream) : Nat → Array Nat → Array Nat
  | 0, d => d
  | k + 1, d => depthsFrom g k (d.push (pathStep d (g.preds d.size)))

/-- Per instruction, the earliest cycle it can issue when the first issues at 0: the longest
weighted path into it through earlier instructions. -/
def depths (g : Stream) : Array Nat := depthsFrom g g.size #[]

/-- The critical path: the latest any instruction can be done, following dependences only. -/
def critical (g : Stream) : Nat :=
  let d := g.depths
  (List.range g.size).foldl (fun m j => max m (d[j]?.getD 0 + g.done j)) 0

/-- The bound on any schedule: the critical path, or the work of the busiest of resources `rs`. -/
def bound (g : Stream) (rs : List Nat) : Nat :=
  max g.critical (rs.foldl (fun m r => max m (g.load r)) 0)

/-- A schedule `s` respects the stream's dependences. -/
def Respects (g : Stream) (s : Nat → Nat) : Prop := ∀ j, ∀ p ∈ g.preds j, s p.1 + p.2 ≤ s j

/-- Every instruction is done by `T`. -/
def Finishes (g : Stream) (s : Nat → Nat) (T : Nat) : Prop := ∀ j < g.size, s j + g.done j ≤ T

/-- Resource `r` serves its instructions one after another, in some order, each done with it by
`T`. -/
def Serves (g : Stream) (s : Nat → Nat) (T r : Nat) : Prop :=
  ∃ order : List Nat, order.Perm (g.users r) ∧
    Chained 0 (order.map fun j => (s j, g.cost r j)) ∧ ∀ j ∈ order, s j + g.cost r j ≤ T

theorem foldl_max_le (d : Array Nat) (B : Nat) : ∀ (ps : List (Nat × Nat)) (m : Nat), m ≤ B →
    (∀ p ∈ ps, p.1 < d.size → d[p.1]?.getD 0 + p.2 ≤ B) →
    ps.foldl (fun m p => if p.1 < d.size then max m (d[p.1]?.getD 0 + p.2) else m) m ≤ B
  | [], _, hm, _ => hm
  | p :: ps, m, hm, h => by
    simp only [List.foldl_cons]
    apply foldl_max_le d B ps
    · split
      · exact Nat.max_le.2 ⟨hm, h p List.mem_cons_self ‹_›⟩
      · exact hm
    · exact fun q hq => h q (List.mem_cons_of_mem _ hq)

theorem depthsFrom_le {g : Stream} {s : Nat → Nat} (hs : g.Respects s) :
    ∀ (k : Nat) (d : Array Nat), (∀ j, d[j]?.getD 0 ≤ s j) → ∀ j, (g.depthsFrom k d)[j]?.getD 0 ≤ s j
  | 0, _, hd => hd
  | k + 1, d, hd => by
    apply depthsFrom_le hs k
    intro j
    rw [Array.getElem?_push]
    split
    · next hj =>
      subst hj
      simp only [Option.getD_some, pathStep]
      apply foldl_max_le d (s d.size) _ 0 (Nat.zero_le _)
      intro p hp _
      have h₁ := hs d.size p hp
      have h₂ := hd p.1
      omega
    · exact hd j

/-- **The dependent-latency chain**: in a schedule that respects the dependences, no instruction
issues before its depth. -/
theorem depths_le {g : Stream} {s : Nat → Nat} (hs : g.Respects s) (j : Nat) :
    g.depths[j]?.getD 0 ≤ s j :=
  depthsFrom_le hs g.size #[] (fun j => by simp) j

theorem critical_le {g : Stream} {s : Nat → Nat} {T : Nat} (hs : g.Respects s) (hf : g.Finishes s T) :
    g.critical ≤ T := by
  show (List.range g.size).foldl (fun m j => max m (g.depths[j]?.getD 0 + g.done j)) 0 ≤ T
  suffices ∀ (l : List Nat) (m : Nat), m ≤ T → (∀ j ∈ l, j < g.size) →
      l.foldl (fun m j => max m (g.depths[j]?.getD 0 + g.done j)) m ≤ T from
    this _ 0 (Nat.zero_le _) (fun j hj => List.mem_range.1 hj)
  intro l
  induction l with
  | nil => intro m hm _; exact hm
  | cons j rest ih =>
    intro m hm hl
    simp only [List.foldl_cons]
    apply ih _ _ (fun i hi => hl i (List.mem_cons_of_mem _ hi))
    have := depths_le hs j
    have := hf j (hl j List.mem_cons_self)
    exact Nat.max_le.2 ⟨hm, by omega⟩

theorem work_map_pairs (l : List Nat) (f g : Nat → Nat) :
    Port.work (l.map fun j => (f j, g j)) = (l.map g).sum := by
  induction l with
  | nil => rfl
  | cons j rest ih => simp only [List.map_cons, Port.work, List.sum_cons, ih]

/-- **Per-resource work**: a resource that serves the stream's instructions one after another
and is done by `T` has `load ≤ T`. -/
theorem load_le {g : Stream} {s : Nat → Nat} {T r : Nat} (h : g.Serves s T r) : g.load r ≤ T := by
  obtain ⟨order, hp, hc, he⟩ := h
  have hw := Chained.work_le_of_ends hc (b := T) (by
    intro x hx
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hx
    exact he j hj)
  rw [work_map_pairs] at hw
  unfold load
  rw [← (hp.map (g.cost r)).sum_nat]
  omega

theorem foldl_max_le_of {α : Type} (f : α → Nat) (T : Nat) :
    ∀ (l : List α) (m : Nat), m ≤ T → (∀ x ∈ l, f x ≤ T) → l.foldl (fun m x => max m (f x)) m ≤ T
  | [], _, hm, _ => hm
  | x :: rest, m, hm, h => by
    simp only [List.foldl_cons]
    exact foldl_max_le_of f T rest _ (Nat.max_le.2 ⟨hm, h x List.mem_cons_self⟩)
      (fun y hy => h y (List.mem_cons_of_mem _ hy))

/-- **No schedule of the stream beats its bound**: every schedule that respects the dependences,
in which each resource of `rs` serves its instructions one after another, and which is done by
`T`, has `T ≥ bound g rs`. -/
theorem bound_le {g : Stream} {s : Nat → Nat} {T : Nat} {rs : List Nat} (hs : g.Respects s)
    (hf : g.Finishes s T) (hr : ∀ r ∈ rs, g.Serves s T r) : g.bound rs ≤ T :=
  Nat.max_le.2 ⟨critical_le hs hf, foldl_max_le_of _ T rs 0 (Nat.zero_le _)
    (fun r hr' => load_le (hr r hr'))⟩

end Stream

end NV.Bounds
