/-!
# A binary min-heap

The engine's event queue: values ordered by a key, the least taken first; ties leave in insertion
order (the key includes a sequence number), so runs are deterministic.
-/

namespace NV.Core

structure Heap (α : Type) where
  data : Array (Nat × Nat × α) := #[]
  seq : Nat := 0
  deriving Inhabited

namespace Heap

variable {α : Type} [Inhabited α]

private def less (a b : Nat × Nat × α) : Bool := a.1 < b.1 || (a.1 == b.1 && a.2.1 < b.2.1)

def size (h : Heap α) : Nat := h.data.size
def isEmpty (h : Heap α) : Bool := h.data.isEmpty

/-- The least key, if any. -/
def minKey? (h : Heap α) : Option Nat := h.data[0]?.map (·.1)

def push (h : Heap α) (key : Nat) (v : α) : Heap α := Id.run do
  -- Take the fields first, so `h` dies and the array is updated in place.
  let seq := h.seq
  let mut d := h.data.push (key, seq, v)
  let mut i := d.size - 1
  for _ in [0:d.size] do
    if i == 0 then break
    let p := (i - 1) / 2
    if less d[i]! d[p]! then
      d := d.swapIfInBounds i p
      i := p
    else break
  return { data := d, seq := seq + 1 }

/-- Remove the least element. -/
def pop? (h : Heap α) : Option (Nat × α × Heap α) := Id.run do
  let seq := h.seq
  let data := h.data
  let some top := data[0]? | return none
  let last := data.back!
  let mut d := data.pop
  if d.isEmpty then return some (top.1, top.2.2, { data := d, seq })
  d := d.set! 0 last
  let mut i := 0
  for _ in [0:d.size] do
    let l := 2 * i + 1
    let r := l + 1
    let mut m := i
    if l < d.size && less d[l]! d[m]! then m := l
    if r < d.size && less d[r]! d[m]! then m := r
    if m == i then break
    d := d.swapIfInBounds i m
    i := m
  return some (top.1, top.2.2, { data := d, seq })

end Heap

end NV.Core
