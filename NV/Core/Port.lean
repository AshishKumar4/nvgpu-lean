/-!
# Ports

Every rate-limited resource of the core is a port: a pipe's input, the barrier unit's arrival
counter, a memory channel. A port serves one request at a time; a request asking at cycle `t` for
`cost` cycles starts at the later of `t` and the cycle the port frees, and holds the port until
it ends. The core grants work through ports only, so a rate bound on the machine is a property
of this one structure: `served_le` says a port free at cycle `a` and free again by `b` served at
most `b - a` cycles of work — the invariant lower bounds rest on.
-/

namespace NV.Core

structure Port where
  /-- The cycle the port accepts its next request. -/
  free : Nat := 0
  deriving Inhabited, Repr, DecidableEq

namespace Port

/-- Serve a request arriving at `t` for `cost` cycles: its start, and the port after it. -/
def reserve (p : Port) (t cost : Nat) : Nat × Port :=
  let start := max t p.free
  (start, { free := start + cost })

theorem reserve_start_ge (p : Port) (t cost : Nat) : (p.reserve t cost).1 ≥ t ∧ (p.reserve t cost).1 ≥ p.free := by
  simp [reserve]; omega

theorem reserve_free (p : Port) (t cost : Nat) : (p.reserve t cost).2.free = (p.reserve t cost).1 + cost := by
  simp [reserve]

/-- Serve a sequence of requests `(arrival, cost)` in order: their starts, and the final port. -/
def serveAll (p : Port) : List (Nat × Nat) → List Nat × Port
  | [] => ([], p)
  | (t, c) :: rest =>
    let r := p.reserve t c
    let q := r.2.serveAll rest
    (r.1 :: q.1, q.2)

/-- Total cost of a list of requests. -/
def work : List (Nat × Nat) → Nat
  | [] => 0
  | (_, c) :: rest => c + work rest

/-- A port that served requests starting no earlier than `p.free` ends at least their total cost
later: the requests occupy disjoint intervals. -/
theorem serveAll_free_ge (p : Port) (rs : List (Nat × Nat)) :
    (p.serveAll rs).2.free ≥ p.free + work rs := by
  induction rs generalizing p with
  | nil => simp [serveAll, work]
  | cons r rest ih =>
    obtain ⟨t, c⟩ := r
    have h := ih (p.reserve t c).2
    have hs := reserve_start_ge p t c
    have hf := reserve_free p t c
    show ((p.reserve t c).2.serveAll rest).2.free ≥ p.free + (c + work rest)
    omega

/-- The rate bound: if a port, free at `a`, serves requests and ends free by `b`, it served at most
`b - a` cycles of work. -/
theorem served_le (p : Port) (rs : List (Nat × Nat)) (b : Nat) (h : (p.serveAll rs).2.free ≤ b) :
    work rs ≤ b - p.free := by
  have := serveAll_free_ge p rs; omega

end Port

end NV.Core
