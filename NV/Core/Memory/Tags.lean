/-!
# Sectored cache tags

The tag array of an L1 or an L2 slice: `sets × ways` lines, each with a valid and a dirty bit per
sector, replaced least recently used. Tags only: the data stay in the flat stores, which is why the
simulator never returns a stale cached value.

A line lives in set `line % sets`. A set holds only the ways that have been filled, so an empty
cache costs one shared empty array per set and fills as it is used.
-/

namespace NV.Core.Memory

/-- A filled way: its line, its valid and dirty sectors (masks), and when it was last used. -/
structure Way where
  line : Nat
  valid : Nat
  dirty : Nat
  used : Nat
  deriving Inhabited

structure Tags where
  sets : Nat
  ways : Nat
  /-- Per set, its filled ways, at most `ways`. -/
  set : Array (Array Way)
  clock : Nat := 0
  deriving Inhabited

namespace Tags

def empty (sets ways : Nat) : Tags :=
  { sets := max sets 1, ways := max ways 1, set := Array.replicate (max sets 1) #[] }

/-- The set of `line`, and the position of `line` in it. -/
def find? (t : Tags) (line : Nat) : Nat × Option Nat :=
  let s := line % t.sets
  (s, t.set[s]!.findIdx? (·.line == line))

/-- The sectors of `mask` that are valid. -/
def present (t : Tags) (line mask : Nat) : Nat :=
  match t.find? line with
  | (s, some i) => t.set[s]![i]!.valid &&& mask
  | (_, none) => 0

/-- Mark `line` used now. -/
def touch (t : Tags) (line : Nat) : Tags :=
  match t.find? line with
  | (s, some i) =>
    let clock := t.clock + 1
    { t with set := t.set.modify s (·.modify i ({ · with used := clock })), clock }
  | (_, none) => t

/-- Make the sectors `mask` of `line` valid (and dirty, for a write), allocating the line in its
set's least recently used way if it is absent: the tags, and the evicted line with its dirty
sectors when one was evicted. -/
def fill (t : Tags) (line mask : Nat) (write : Bool) : Tags × Option (Nat × Nat) :=
  let clock := t.clock + 1
  let d := if write then mask else 0
  match t.find? line with
  | (s, some i) =>
    ({ t with set := t.set.modify s (·.modify i fun w =>
        { w with valid := w.valid ||| mask, dirty := w.dirty ||| d, used := clock }), clock }, none)
  | (s, none) =>
    let ws := t.set[s]!
    let way : Way := { line, valid := mask, dirty := d, used := clock }
    if ws.size < t.ways then ({ t with set := t.set.modify s (·.push way), clock }, none)
    else
      let v := (List.range ws.size).foldl (fun v i => if ws[i]!.used < ws[v]!.used then i else v) 0
      let old := ws[v]!
      ({ t with set := t.set.modify s (·.set! v way), clock }, some (old.line, old.dirty))

/-- Drop the sectors `mask` of `line`. -/
def invalidate (t : Tags) (line mask : Nat) : Tags :=
  match t.find? line with
  | (s, some i) =>
    { t with set := t.set.modify s (·.modify i fun w =>
        { w with valid := w.valid ^^^ (w.valid &&& mask), dirty := w.dirty ^^^ (w.dirty &&& mask) }) }
  | (_, none) => t

end Tags

end NV.Core.Memory
