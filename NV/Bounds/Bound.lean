import NV.Bounds.Machine
import NV.Bounds.Chain

/-!
# Bounds and gap reports

A bound is a list of terms, each a resource with the work a spec (or a fixed kernel) forces on it;
`Bound.cycles` is the largest term's time, and `Bound.cycles_le` says no run that does the work
takes less. A term carries its class (the programs the work is forced in, and the theorem that
forces it) and the facts its capacity rests on, so a report says why each number holds.

A `Gap` sets a bound beside a simulated time: per term, time over the term's cycles, with the
binding term named. Simulated time is an achieved time, so the gap is at most the ratio by which
the kernel could improve within the model and the assumptions stated.
-/

namespace NV.Bounds

open NV NV.Core

/-- What a term asks of a run. -/
inductive Demand where
  /-- `work` units through a rated resource. -/
  | rate (c : Capacity) (work : Nat)
  /-- `work` requests on `instances` copies of a pipe-like resource, one per `interval` each. -/
  | pipe (instances interval work : Nat)
  /-- A fixed instruction stream: every schedule is bounded by its critical path and the work on
  each resource of `rs` (`Stream.bound`). -/
  | stream (g : Stream) (rs : List Nat)
  /-- Dependent steps, each waiting the previous one's latency: the dependent-latency chain. -/
  | chain (lats : List Nat)
  /-- Two demands met one after the other, in disjoint stretches of the run. -/
  | seq (a b : Demand)
  /-- Two demands met by the same run. -/
  | both (a b : Demand)

/-- Cycles `s` at which dependent steps start, from step `j`: each starts at least the previous
one's latency after it. -/
def ChainOk (s : Nat → Nat) : Nat → List Nat → Prop
  | _, [] => True
  | j, l :: rest => s j + l ≤ s (j + 1) ∧ ChainOk s (j + 1) rest

theorem ChainOk.le {s : Nat → Nat} : ∀ {j : Nat} {lats : List Nat}, ChainOk s j lats →
    s j + lats.sum ≤ s (j + lats.length)
  | _, [], _ => by simp
  | j, l :: rest, ⟨h₁, h₂⟩ => by
    have := ChainOk.le h₂
    simp only [List.sum_cons, List.length_cons]
    have e : j + 1 + rest.length = j + (rest.length + 1) := by omega
    rw [e] at this
    omega

namespace Demand

def cycles : Demand → Nat
  | .rate c w => c.minCycles w
  | .pipe n i w => pipeCycles n i w
  | .stream g rs => g.bound rs
  | .chain lats => lats.sum
  | .seq a b => a.cycles + b.cycles
  | .both a b => max a.cycles b.cycles

/-- A run of `T` cycles does what the term asks: it moves the work through the resource, it is a
schedule of the stream, or its steps are dependent. -/
def Met (T : Nat) : Demand → Prop
  | .rate c w => 0 < c.instances * c.rate.units ∧ ∃ u : Use c T, w ≤ u.units
  | .pipe n i w => 0 < n ∧ 0 < i ∧ ∃ u : Slots n i T, w ≤ u.count
  | .stream g rs => ∃ s, g.Respects s ∧ g.Finishes s T ∧ ∀ r ∈ rs, g.Serves s T r
  | .chain lats => ∃ s : Nat → Nat, ChainOk s 0 lats ∧ s lats.length ≤ T
  | .seq a b => ∃ T₁ T₂, T₁ + T₂ ≤ T ∧ a.Met T₁ ∧ b.Met T₂
  | .both a b => a.Met T ∧ b.Met T

theorem cycles_le : ∀ {T : Nat} {d : Demand}, d.Met T → d.cycles ≤ T
  | _, .rate _ _, ⟨hpos, u, hw⟩ => u.time_ge hpos hw
  | _, .pipe _ _ _, ⟨hn, hi, u, hw⟩ => u.time_ge hn hi hw
  | _, .stream _ _, ⟨_, hs, hf, hr⟩ => Stream.bound_le hs hf hr
  | _, .chain _, ⟨_, hc, hT⟩ => by
    have := hc.le
    simp only [Nat.zero_add] at this
    show List.sum _ ≤ _
    omega
  | _, .seq _ _, ⟨_, _, hT, ha, hb⟩ => by
    have := cycles_le ha
    have := cycles_le hb
    show _ + _ ≤ _
    omega
  | _, .both _ _, ⟨ha, hb⟩ => Nat.max_le.2 ⟨cycles_le ha, cycles_le hb⟩

end Demand

/-- One term of a bound. -/
structure Term where
  resource : String
  unit : String
  demand : Demand
  /-- The programs the demand holds for. -/
  scope : String
  /-- The theorem that forces the work, and what it assumes beyond the term's facts. -/
  reason : String
  facts : List FactRef := []

structure Bound where
  spec : String
  gpu : String
  terms : List Term

def Bound.cycles (b : Bound) : Nat := b.terms.foldl (fun m t => max m t.demand.cycles) 0

/-- The term with the most cycles. -/
def Bound.binding (b : Bound) : Option Term :=
  b.terms.foldl (fun best t => match best with
    | some u => if t.demand.cycles > u.demand.cycles then some t else best
    | none => some t) none

/-- **No run beats the bound**: a run of `T` cycles that does what every term asks takes at least
the largest term's cycles. -/
theorem Bound.cycles_le (b : Bound) {T : Nat} (h : ∀ t ∈ b.terms, t.demand.Met T) : b.cycles ≤ T := by
  have := Stream.foldl_max_le_of (fun t : Term => t.demand.cycles) T b.terms 0 (Nat.zero_le _)
    fun t ht => Demand.cycles_le (h t ht)
  exact this

/-! ## Terms from resources -/

def Resource.term (r : Resource) (work : Nat) (scope reason : String) : Term :=
  { resource := r.name, unit := r.unit, demand := .rate r.capacity work, scope, reason,
    facts := r.facts }

def PipeResource.term (r : PipeResource) (work : Nat) (scope reason : String) : Term :=
  { resource := r.name, unit := r.unit, demand := .pipe r.instances r.interval work, scope, reason,
    facts := r.facts }

/-! ## Reports -/

/-- `a / b` to two decimals. -/
def ratio (a b : Nat) : String :=
  if b == 0 then "-" else
    let h := a * 100 / b
    let frac := h % 100
    s!"{h / 100}.{if frac < 10 then "0" else ""}{frac}"

def Demand.describe : Demand → String
  | .rate c w => s!"{w} over {c.instances} × {c.rate.units}/{c.rate.cycles} per cycle"
  | .pipe n i w => s!"{w} over {n} × 1/{i} per cycle"
  | .stream g rs => s!"{g.size} instructions: critical path {g.critical}, busiest of {rs.length} resources {(rs.map g.load).foldl max 0}"
  | .chain lats => s!"dependent steps {lats}"
  | .seq a b => s!"({a.describe}) then ({b.describe})"
  | .both a b => s!"({a.describe}) and ({b.describe})"

/-- A bound set beside a simulated time. -/
structure Gap where
  bound : Bound
  /-- Simulated cycles, when a run exists. -/
  simulated : Option Nat

def Gap.render (g : Gap) : String := Id.run do
  let b := g.bound
  let total := b.cycles
  let mut out := s!"bound: {b.spec} on {b.gpu}\n"
  match g.simulated with
  | some t => out := out ++ s!"simulated {t} cycles; bound {total} cycles; gap {ratio t total}\n"
  | none => out := out ++ s!"bound {total} cycles (no simulated run)\n"
  match b.binding with
  | some t => out := out ++ s!"binding: {t.resource}\n"
  | none => pure ()
  out := out ++ "\n| resource | work | cycles | simulated / cycles | scope |\n|---|---|---|---|---|\n"
  for t in b.terms do
    let c := t.demand.cycles
    let gap := match g.simulated with | some s => ratio s c | none => "-"
    out := out ++ s!"| {t.resource} | {t.demand.describe} {t.unit} | {c} | {gap} | {t.scope} |\n"
  out := out ++ "\nwhy each term holds, and the facts it used:\n"
  for t in b.terms do
    out := out ++ s!"\n* {t.resource}: {t.reason}\n"
    for f in t.facts do
      out := out ++ s!"  - {f.render}\n"
  return out

end NV.Bounds
