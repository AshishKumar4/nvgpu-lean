import NV.Core.Port

/-!
# Ports with a rate

Most memory resources move bytes at a rate that is not a whole number per cycle (a 32-byte sector
through a 64-byte-per-cycle slice takes half a cycle; a DRAM channel moves about 26 bytes per
cycle). Such a resource is a `Port` whose clock ticks `units` times per cycle and whose unit of
work costs `cycles` ticks, so it serves exactly `units` per `cycles` cycles. `Port.served_le`
applies unchanged, in ticks: a rated port free at cycle `a` and free again by cycle `b` served at
most `(b - a) * units / cycles` units (`Rate.served_le`).
-/

namespace NV.Core

/-- `units` (bytes, operations, …) per `cycles` cycles. -/
structure Rate where
  units : Nat
  cycles : Nat := 1
  deriving Repr, Inhabited, DecidableEq

namespace Rate

/-- `units` per cycle of a clock of `refMHz`, as a rate per cycle of an SM clock of `clockMHz` (the
same units per second), reduced so ticks stay small. -/
def atClock (units refMHz clockMHz : Nat) : Rate :=
  let u := units * refMHz
  let c := max clockMHz 1
  let g := Nat.gcd u c
  { units := u / max g 1, cycles := c / max g 1 }

/-- A rate from a per-second figure and the SM clock, reduced so ticks stay small. -/
def perSecond (unitsPerSec clockMHz : Nat) : Rate :=
  let c := clockMHz * 1000000
  let g := Nat.gcd unitsPerSec c
  { units := unitsPerSec / max g 1, cycles := c / max g 1 }

end Rate

namespace Port

/-- Serve `n` units arriving at cycle `t` at rate `r`: the cycle the work starts, the last cycle it
occupies, and the port after it. No work (`n = 0`) passes without waiting. -/
def take (p : Port) (r : Rate) (t n : Nat) : Nat × Nat × Port :=
  if n == 0 then (t, t, p)
  else
    let (s, q) := p.reserve (t * r.units) (n * r.cycles)
    let start := s / r.units
    (start, max start ((q.free + r.units - 1) / r.units - 1), q)

theorem take_start_ge (p : Port) (r : Rate) (t n : Nat) (hu : 0 < r.units) :
    (p.take r t n).1 ≥ t := by
  unfold take
  split
  · exact Nat.le_refl t
  · have h := (reserve_start_ge p (t * r.units) (n * r.cycles)).1
    exact (Nat.le_div_iff_mul_le hu).2 h

/-- `take` is a reservation in ticks: the port after it is the port after `reserve`, so every
rated port's history is a `serveAll` history and `served_le` bounds it. -/
theorem take_port (p : Port) (r : Rate) (t n : Nat) :
    (p.take r t n).2.2 = if n == 0 then p else (p.reserve (t * r.units) (n * r.cycles)).2 := by
  unfold take
  split <;> simp_all

theorem take_le_last (p : Port) (r : Rate) (t n : Nat) : (p.take r t n).1 ≤ (p.take r t n).2.1 := by
  unfold take
  split
  · exact Nat.le_refl t
  · exact Nat.le_max_left _ _

end Port

/-- Work in ticks of requests sized in units. -/
theorem Rate.work_map (r : Rate) (rs : List (Nat × Nat)) :
    Port.work (rs.map fun x => (x.1, x.2 * r.cycles)) = (rs.map (·.2)).sum * r.cycles := by
  induction rs with
  | nil => simp [Port.work]
  | cons x rest ih =>
    simp only [List.map_cons, List.sum_cons, Port.work, ih, Nat.add_mul]

/-- The rate bound in units: a sequence of requests `(arrival tick, units)` served on a port at
rate `r`, starting free at tick `a` and ending free by tick `b`, moved at most `(b - a) / r.cycles`
units. With `a = a₀ * r.units` and `b = b₀ * r.units` that is `(b₀ - a₀) * r.units / r.cycles`. -/
theorem Rate.served_le (r : Rate) (p : Port) (rs : List (Nat × Nat)) (b : Nat)
    (h : (p.serveAll (rs.map fun x => (x.1, x.2 * r.cycles))).2.free ≤ b) :
    (rs.map (·.2)).sum * r.cycles ≤ b - p.free := by
  rw [← Rate.work_map]
  exact Port.served_le p _ b h

end NV.Core
