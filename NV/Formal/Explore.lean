import Std.Data.HashSet
import NV.Formal.System

/-!
# Explicit-state search

Breadth-first search over a finite instance of a protocol, for the shortest action trace that
reaches a bad state. A trace it returns is replayed by `System.run` in a theorem, so a
counterexample stands on evaluation alone and not on the search being correct.
-/

namespace NV.Formal.System

variable {σ α κ : Type} [BEq κ] [Hashable κ] (S : System σ α)

/-- The shortest trace from the start to a state satisfying `bad`, exploring at most `limit`
distinct states (by `key`) built from the candidate actions `moves`. -/
def search (moves : σ → List α) (key : σ → κ) (bad : σ → Bool) (limit : Nat) :
    Option (List α) := Id.run do
  let mut seen : Std.HashSet κ := Std.HashSet.emptyWithCapacity
  let mut frontier : Array (σ × List α) := #[(S.init, [])]
  seen := seen.insert (key S.init)
  while !frontier.isEmpty && seen.size < limit do
    let mut next := #[]
    for (s, trace) in frontier do
      if bad s then return some trace.reverse
      for a in moves s do
        if let some t := S.step s a then
          let k := key t
          if !seen.contains k then
            seen := seen.insert k
            next := next.push (t, a :: trace)
    frontier := next
  return none

end NV.Formal.System
