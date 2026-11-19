import NV.Core.Memory.Rate

/-!
# Rate invariants

Every rate-limited resource of the core is granted through a port (`NV.Core.Port`): a
sub-partition's issue slot, a pipe's input, an L1 data stage, an L2 slice, a DRAM channel. This
module says what a resource can serve in a given time, for any order of service, and ties that to
the engine's ports.

* `Chained a rs`: requests `(start, cost)` served one after another from cycle `a`, each starting
  no earlier than the previous one ended. Every `Port.serveAll` history is chained
  (`serveAll_chained`), and so is every schedule that respects the resource, whatever its
  policy, so the theorems below hold of the engine's runs and of any other schedule alike.
* `Chained.work_le_of_ends`: chained requests that end by `b` did at most `b - a` cycles of work.
  `Chained.span_lt_of_starts`: `k + 1` requests of cost `c` that start before `b` have
  `a + k * c < b`.
* A `Capacity` is `instances` copies of a resource, each at a `Rate`. `Use` is what a run of `T`
  cycles did on one: per copy, its requests in service order. `Use.units_le`: it moved at most
  `instances * T * rate.units / rate.cycles` units, so `Use.time_ge`: `T ≥ minCycles w` for any `w`
  it moved. `Slots` is the same for a pipe, whose requests are counted by their starts (a pipe is
  busy for its interval after the run's last issue); `Slots.time_ge`: `T ≥ pipeCycles`.
* `perSecond_exact`: the engine's per-second rates (DRAM channels, L2 atomics) are exact,
  not rounded: `units / cycles` is the per-second figure over the clock.
-/

namespace NV.Bounds

open NV.Core

/-! ## Chained service -/

/-- Requests `(start, cost)` served one at a time from cycle `a`: each starts no earlier than the
previous one ended (or `a`, for the first). -/
def Chained : Nat → List (Nat × Nat) → Prop
  | _, [] => True
  | a, (s, c) :: rest => a ≤ s ∧ Chained (s + c) rest

/-- Chained requests that all end by `b` did at most `b - a` cycles of work: they occupy disjoint
intervals of `[a, b)`. -/
theorem Chained.work_le_of_ends : ∀ {a b : Nat} {rs : List (Nat × Nat)}, Chained a rs →
    (∀ x ∈ rs, x.1 + x.2 ≤ b) → Port.work rs ≤ b - a
  | _, _, [], _, _ => by simp [Port.work]
  | a, b, (s, c) :: rest, ⟨h₁, h₂⟩, hb => by
    have ih := Chained.work_le_of_ends h₂ (fun x hx => hb x (List.mem_cons_of_mem _ hx))
    have hsc : s + c ≤ b := hb (s, c) List.mem_cons_self
    simp only [Port.work]
    omega

/-- `k + 1` chained requests of cost `c` that all start before `b`: the last starts at least
`k * c` after `a`, so `a + k * c < b`. -/
theorem Chained.span_lt_of_starts : ∀ {a b c : Nat} {r : Nat × Nat} {rest : List (Nat × Nat)},
    Chained a (r :: rest) → (∀ x ∈ r :: rest, x.2 = c ∧ x.1 < b) → a + rest.length * c < b
  | a, b, c, (s, c₀), [], ⟨h₁, _⟩, hb => by
    have : s < b := (hb (s, c₀) List.mem_cons_self).2
    simp only [List.length_nil, Nat.zero_mul, Nat.add_zero]
    omega
  | a, b, c, (s, c₀), r' :: rest', ⟨h₁, h₂⟩, hb => by
    have hc : c₀ = c := (hb (s, c₀) List.mem_cons_self).1
    have ih := Chained.span_lt_of_starts h₂ (fun x hx => hb x (List.mem_cons_of_mem _ hx))
    simp only [List.length_cons, Nat.add_mul, Nat.one_mul] at ih ⊢
    omega

/-- The engine's ports serve chained: a `Port.serveAll` history, its starts paired with its costs,
is chained from the port's free cycle, and every request ends by the port's final free cycle. -/
theorem serveAll_chained (p : Port) (rs : List (Nat × Nat)) :
    Chained p.free ((p.serveAll rs).1.zip (rs.map (·.2))) ∧
      ∀ x ∈ (p.serveAll rs).1.zip (rs.map (·.2)), x.1 + x.2 ≤ (p.serveAll rs).2.free := by
  induction rs generalizing p with
  | nil => simp [Port.serveAll, Chained]
  | cons r rest ih =>
    obtain ⟨t, c⟩ := r
    have hs := Port.reserve_start_ge p t c
    have hf := Port.reserve_free p t c
    have hg := Port.serveAll_free_ge (p.reserve t c).2 rest
    obtain ⟨ih₁, ih₂⟩ := ih (p.reserve t c).2
    show Chained p.free (((p.reserve t c).1 :: ((p.reserve t c).2.serveAll rest).1).zip
        (c :: rest.map (·.2))) ∧ ∀ x ∈ (((p.reserve t c).1 :: ((p.reserve t c).2.serveAll rest).1).zip
        (c :: rest.map (·.2))), x.1 + x.2 ≤ ((p.reserve t c).2.serveAll rest).2.free
    rw [List.zip_cons_cons]
    refine ⟨⟨hs.2, by rw [← hf]; exact ih₁⟩, ?_⟩
    intro x hx
    rcases List.mem_cons.1 hx with rfl | hx
    · show (p.reserve t c).1 + c ≤ _
      omega
    · exact ih₂ x hx

/-! ## Sums -/

theorem sum_le_mul {α : Type} (l : List α) (f : α → Nat) (B : Nat) (h : ∀ x ∈ l, f x ≤ B) :
    (l.map f).sum ≤ l.length * B := by
  induction l with
  | nil => simp
  | cons x rest ih =>
    have hx := h x List.mem_cons_self
    have := ih (fun y hy => h y (List.mem_cons_of_mem _ hy))
    simp only [List.map_cons, List.sum_cons, List.length_cons, Nat.add_mul, Nat.one_mul]
    omega

theorem sum_map_mul {α : Type} (l : List α) (f : α → Nat) (k : Nat) :
    (l.map f).sum * k = (l.map fun x => f x * k).sum := by
  induction l with
  | nil => simp
  | cons x rest ih => simp only [List.map_cons, List.sum_cons, Nat.add_mul, ih]

/-! ## Capacities and the time they force -/

/-- `⌈a / b⌉`. -/
def cdiv (a b : Nat) : Nat := (a + b - 1) / b

theorem cdiv_le {a b T : Nat} (hb : 0 < b) (h : a ≤ b * T) : cdiv a b ≤ T := by
  unfold cdiv
  have : (a + b - 1) / b < T + 1 := by
    rw [Nat.div_lt_iff_lt_mul hb, Nat.add_mul, Nat.one_mul, Nat.mul_comm T b]
    omega
  omega

theorem le_mul_of_cdiv_le {a b T : Nat} (hb : 0 < b) (h : cdiv a b ≤ T) : a ≤ b * T := by
  unfold cdiv at h
  have h' : (a + b - 1) / b < T + 1 := by omega
  rw [Nat.div_lt_iff_lt_mul hb, Nat.add_mul, Nat.one_mul, Nat.mul_comm T b] at h'
  omega

/-- `instances` copies of a resource, each serving `rate.units` units per `rate.cycles` cycles. -/
structure Capacity where
  instances : Nat
  rate : Rate
  deriving Repr, Inhabited, DecidableEq

/-- The fewest cycles in which the resource can move `w` units. -/
def Capacity.minCycles (c : Capacity) (w : Nat) : Nat :=
  cdiv (w * c.rate.cycles) (c.instances * c.rate.units)

/-- What a run of `T` cycles did on a resource: per copy, the requests it served `(start tick,
units)` in service order, chained, each ending by the run's end. A tick is `1 / rate.units` of a
cycle and a unit costs `rate.cycles` ticks, as in `Port.take`. -/
structure Use (c : Capacity) (T : Nat) where
  hists : List (List (Nat × Nat))
  length : hists.length = c.instances
  chained : ∀ h ∈ hists, Chained 0 (h.map fun x => (x.1, x.2 * c.rate.cycles))
  ends : ∀ h ∈ hists, ∀ x ∈ h, x.1 + x.2 * c.rate.cycles ≤ T * c.rate.units

/-- Units the run moved through the resource. -/
def Use.units {c : Capacity} {T : Nat} (u : Use c T) : Nat :=
  (u.hists.map fun h => (h.map (·.2)).sum).sum

/-- **The rate invariant**: in `T` cycles a resource moves at most `instances * T * units /
cycles` units. -/
theorem Use.units_le {c : Capacity} {T : Nat} (u : Use c T) :
    u.units * c.rate.cycles ≤ c.instances * (T * c.rate.units) := by
  unfold Use.units
  rw [sum_map_mul, ← u.length]
  apply sum_le_mul
  intro h hh
  have hw := Chained.work_le_of_ends (u.chained h hh) (b := T * c.rate.units) (by
    intro x hx
    obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hx
    exact u.ends h hh y hy)
  rw [Rate.work_map] at hw
  omega

/-- A run that moved `w` units through a resource took at least `minCycles w` cycles. -/
theorem Use.time_ge {c : Capacity} {T w : Nat} (u : Use c T) (hpos : 0 < c.instances * c.rate.units)
    (hw : w ≤ u.units) : c.minCycles w ≤ T := by
  apply cdiv_le hpos
  have := u.units_le
  have : w * c.rate.cycles ≤ u.units * c.rate.cycles := Nat.mul_le_mul_right _ hw
  rw [Nat.mul_assoc, Nat.mul_comm c.rate.units T]
  omega

/-! ## Pipes and issue slots -/

/-- What a run of `T` cycles did on `instances` copies of a pipe-like resource that each request
holds for `interval` cycles: per copy, the cycles its requests started, in service order, chained,
each before the run's end. An issue slot is the case `interval = 1`. -/
structure Slots (instances interval T : Nat) where
  hists : List (List Nat)
  length : hists.length = instances
  chained : ∀ h ∈ hists, Chained 0 (h.map fun s => (s, interval))
  starts : ∀ h ∈ hists, ∀ s ∈ h, s < T

/-- Requests the run served on the resource. -/
def Slots.count {n c T : Nat} (u : Slots n c T) : Nat := (u.hists.map List.length).sum

/-- One copy of a pipe takes at most `(T - 1) / interval + 1` requests in `T` cycles. -/
theorem Slots.length_le {n c T : Nat} (u : Slots n c T) (hc : 0 < c) :
    ∀ h ∈ u.hists, h.length ≤ (T - 1) / c + 1 := by
  intro h hh
  match h, hh with
  | [], _ => simp
  | s :: rest, hh =>
    have hch : Chained 0 ((s, c) :: rest.map fun s => (s, c)) := u.chained _ hh
    have hspan := Chained.span_lt_of_starts hch (b := T) (c := c) (by
      intro x hx
      have hx' : x ∈ (s :: rest).map fun s => (s, c) := hx
      obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hx'
      exact ⟨rfl, u.starts _ hh y hy⟩)
    simp only [List.length_map] at hspan
    have : rest.length ≤ (T - 1) / c := (Nat.le_div_iff_mul_le hc).2 (by omega)
    simp only [List.length_cons]
    omega

theorem Slots.count_le {n c T : Nat} (u : Slots n c T) (hc : 0 < c) :
    u.count ≤ n * ((T - 1) / c + 1) := by
  have := sum_le_mul u.hists List.length _ (u.length_le hc)
  rw [u.length] at this
  exact this

theorem Slots.pos_of_count {n c T : Nat} (u : Slots n c T) (h : 0 < u.count) : 0 < T := by
  apply Nat.pos_of_ne_zero
  intro hT
  have : u.count ≤ u.hists.length * 0 := by
    unfold Slots.count
    apply sum_le_mul
    intro l hl
    have : l = [] := List.eq_nil_iff_forall_not_mem.2 fun s hs => by
      have := u.starts l hl s hs
      omega
    simp [this]
  omega

/-- The fewest cycles in which `instances` copies of a pipe of the given interval take `k`
requests: the busiest copy takes `⌈k / instances⌉`, one every `interval` cycles. -/
def pipeCycles (instances interval k : Nat) : Nat :=
  if k = 0 then 0 else (cdiv k instances - 1) * interval + 1

/-- A run that served `k` requests on a pipe took at least `pipeCycles` cycles. With
`interval = 1` this is the issue bound: one instruction per sub-partition per cycle. -/
theorem Slots.time_ge {n c T k : Nat} (u : Slots n c T) (hn : 0 < n) (hc : 0 < c)
    (hk : k ≤ u.count) : pipeCycles n c k ≤ T := by
  unfold pipeCycles
  split
  · exact Nat.zero_le _
  · have hT := u.pos_of_count (by omega)
    have hle := u.count_le hc
    have hq : cdiv k n ≤ (T - 1) / c + 1 := cdiv_le hn (Nat.le_trans hk hle)
    have h₁ : cdiv k n - 1 ≤ (T - 1) / c := Nat.sub_le_of_le_add hq
    have h₂ := (Nat.le_div_iff_mul_le hc).1 h₁
    generalize (cdiv k n - 1) * c = x at h₂ ⊢
    omega

/-! ## The engine's rates -/

/-- `Rate.perSecond` is exact: `units` per `cycles` cycles is `unitsPerSec` per second at
`clockMHz`. So the engine's DRAM channels together serve exactly `dramBytesPerSec`, and its
L2 atomic units exactly the documented reduction rate (`Memory.Config.ofParams`). -/
theorem perSecond_exact (u c : Nat) :
    (Rate.perSecond u c).units * (c * 1000000) = u * (Rate.perSecond u c).cycles := by
  simp only [Rate.perSecond]
  generalize c * 1000000 = C
  have hu := Nat.gcd_dvd_left u C
  have hc := Nat.gcd_dvd_right u C
  generalize Nat.gcd u C = g at hu hc ⊢
  obtain ⟨a, rfl⟩ := hu
  obtain ⟨b, rfl⟩ := hc
  rcases Nat.eq_zero_or_pos g with rfl | hg
  · simp
  · have hm : max g 1 = g := by omega
    rw [hm, Nat.mul_div_cancel_left _ hg, Nat.mul_div_cancel_left _ hg, Nat.mul_left_comm,
      Nat.mul_assoc]

end NV.Bounds
