import NV.SASS.Footprint

/-!
# Pending register accesses

Per warp, what its in-flight instructions will still write or have yet to read, by location: the
engine's hazard check reads it on every issue, so it is a flat table indexed by location rather
than a map.
-/

namespace NV.Core

open NV.SASS

/-- Locations, numbered: vector registers, uniform registers, predicates, uniform predicates,
convergence barriers, the `printf` log, the barrier-reduction answer. -/
def _root_.NV.SASS.Loc.index : Loc → Nat
  | .r n => n % 256
  | .ur n => 256 + n % 256
  | .p n => 512 + n % 8
  | .up n => 520 + n % 8
  | .b n => 528 + n % 16
  | .log => 544
  | .bar => 545

def locCount : Nat := 546

/-- For each location: how many pending entries, the lanes they cover, the latest cycle a
pending write lands, the instruction that registered it last, and whether the pending write is an
asynchronous MMA's. -/
structure Pending where
  count : Array Nat := Array.replicate locCount 0
  lanes : Array UInt32 := Array.replicate locCount 0
  landsAt : Array Nat := Array.replicate locCount 0
  by_ : Array Nat := Array.replicate locCount 0
  mma : Array Bool := Array.replicate locCount false
  deriving Inhabited

namespace Pending

/-- Count `ls` in, for the lanes `lanes`. -/
def add (h : Pending) (ls : Array Loc) (lanes : UInt32 := 0xFFFFFFFF) : Pending :=
  ls.foldl (fun h l =>
    let i := l.index
    { h with count := h.count.modify i (· + 1), lanes := h.lanes.modify i (· ||| lanes) }) h

/-- Count `ls` out; a location with no entry left forgets its lanes and MMA mark. -/
def remove (h : Pending) (ls : Array Loc) : Pending :=
  ls.foldl (fun h l =>
    let i := l.index
    let n := h.count[i]!
    if n ≤ 1 then
      { h with count := h.count.set! i 0, lanes := h.lanes.set! i 0, landsAt := h.landsAt.set! i 0,
               mma := h.mma.set! i false }
    else { h with count := h.count.set! i (n - 1) }) h

def has (h : Pending) (l : Loc) : Bool := h.count[l.index]! > 0

/-- Whether nothing is pending. -/
def isEmpty (h : Pending) : Bool := h.count.all (· == 0)

/-- Whether `l` is pending for any of `lanes`. -/
def hasFor (h : Pending) (lanes : UInt32) (l : Loc) : Bool :=
  let i := l.index
  h.count[i]! > 0 && h.lanes[i]! &&& lanes != 0

def landing (h : Pending) (l : Loc) (default : Nat) : Nat :=
  if h.has l then h.landsAt[l.index]! else default

/-- Record that `ls`' pending writes land by `at_` (the latest kept), by instruction `pc`. -/
def landsBy (h : Pending) (ls : Array Loc) (at_ pc : Nat) : Pending :=
  ls.foldl (fun h l =>
    let i := l.index
    let prev := if h.count[i]! > 0 then h.landsAt[i]! else 0
    { h with landsAt := h.landsAt.set! i (max at_ prev), by_ := h.by_.set! i pc }) h

def markBy (h : Pending) (ls : Array Loc) (pc : Nat) : Pending :=
  ls.foldl (fun h l => { h with by_ := h.by_.set! l.index pc }) h

def setMma (h : Pending) (ls : Array Loc) : Pending :=
  ls.foldl (fun h l => { h with mma := h.mma.set! l.index true }) h

def isMma (h : Pending) (l : Loc) : Bool := h.mma[l.index]!

end Pending

end NV.Core
