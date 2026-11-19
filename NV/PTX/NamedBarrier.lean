/-!
# CTA named barriers

`bar{.cta}.sync`, `bar{.cta}.arrive` and their `barrier{.cta}` spellings, PTX ISA 9.4 §9.7.15.1.
A CTA has sixteen barriers. Arrivals are counted per warp (a warp's arrival marks all its
non-exited threads), so an arrival adds 32 threads; the barrier completes when the count named by
the instruction is reached, releases its waiters and resets for immediate reuse.

The ISA requires every participant of one barrier instance to name the same thread count, a
multiple of 32, and warns that a warp must not arrive twice before the barrier resets
("Care must be taken..."). The first two are faults here. The third is not a hardware fault: the
extra arrival is simply counted, possibly completing the barrier early, so protocols state which
warps they expect to complete each instance (`Generation.arrivals`) and prove it.
-/

namespace NV.PTX

inductive NamedBarrier.Fault where
  | badId
  | badCount
  /-- Two participants of one instance named different thread counts. -/
  | countMismatch
  deriving DecidableEq, Repr, Hashable

/-- One barrier resource: the instance in progress and how many instances have completed. -/
structure NamedBarrier where
  /-- Thread count of the instance in progress, set by its first arrival. -/
  count : Option Nat := none
  arrived : Nat := 0
  /-- Completed instances; a waiter released when this passes the instance it arrived in. -/
  generation : Nat := 0
  /-- Warps that arrived in the instance in progress, oldest first. -/
  arrivals : List Nat := []
  deriving DecidableEq, Repr, Inhabited

namespace NamedBarrier

def warpSize : Nat := 32

/-- The outcome of one warp's arrival. -/
structure Arrival where
  barrier : NamedBarrier
  /-- The instance the warp arrived in; `bar.sync` waits until `generation` exceeds it. -/
  instance_ : Nat
  /-- The warps that completed the instance, if this arrival completed it. -/
  completed : Option (List Nat)

/-- Warp `w` arrives with thread count `n`. -/
def arrive (b : NamedBarrier) (w n : Nat) : Except Fault Arrival :=
  if n = 0 ∨ n % warpSize ≠ 0 then .error .badCount
  else if b.count.any (· ≠ n) then .error .countMismatch
  else
    let arrived := b.arrived + warpSize
    let arrivals := b.arrivals ++ [w]
    if arrived = n then .ok ⟨{ generation := b.generation + 1 }, b.generation, some arrivals⟩
    else .ok ⟨{ b with count := some n, arrived, arrivals }, b.generation, none⟩

/-- Whether a warp that arrived in instance `g` has been released. -/
def released (b : NamedBarrier) (g : Nat) : Bool := g < b.generation

end NamedBarrier

/-- A CTA's sixteen barriers. -/
structure NamedBarriers where
  bars : Fin 16 → NamedBarrier := fun _ => {}

namespace NamedBarriers

def arrive (bs : NamedBarriers) (id : Nat) (w n : Nat) :
    Except NamedBarrier.Fault (NamedBarriers × NamedBarrier.Arrival) :=
  if h : id < 16 then do
    let a ← (bs.bars ⟨id, h⟩).arrive w n
    return ({ bars := fun i => if i = ⟨id, h⟩ then a.barrier else bs.bars i }, a)
  else .error .badId

end NamedBarriers
end NV.PTX
