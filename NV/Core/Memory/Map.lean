import NV.Machine.Access

/-!
# Where a request's bytes go

The pure part of the memory system: which 128-byte lines and 32-byte sectors a warp's request
touches (the coalescer), which L2 slice and DRAM channel a line lives in, and how many cycles
(wavefronts) a shared-memory request takes through 32 banks.

NVIDIA does not publish its slice and channel hashes. `spread` stands in for both: an XOR fold of
the line address, so that consecutive lines, and strides of a power of two, spread over every
target. It is model structure, not a measured fact: the bandwidths it divides are facts, and a
kernel whose traffic the real hash would concentrate on few slices is the case it gets wrong.
-/

namespace NV.Core.Memory

/-- One line's share of a request: the line, its sectors as a 4-bit mask, and the active lanes
whose access starts in it. -/
structure LineAccess where
  line : Nat
  sectors : Nat
  lanes : Array Nat
  deriving Repr, Inhabited

/-- Spread a line address over `n` targets. -/
def spread (line n : Nat) : Nat :=
  (line ^^^ (line >>> 7) ^^^ (line >>> 13) ^^^ (line >>> 19)) % n

theorem spread_lt (line n : Nat) (h : 0 < n) : spread line n < n := Nat.mod_lt _ h

/-- Sectors `lo` to `hi` of a line, as a mask. -/
def sectorRange (lo hi : Nat) : Nat := 2 ^ (hi + 1) - 2 ^ lo

/-- The coalescer: group the active lanes by the line their access starts in, in order of first
use. A lane's access lies in one sector when it is naturally aligned, as the ISA requires; it is
counted in the sectors of its line its bytes cover. -/
def coalesce (sectorBytes lineBytes bytes : Nat) (addrs : Array (Option UInt64)) :
    Array LineAccess := Id.run do
  let mut out : Array LineAccess := #[]
  for lane in [0:addrs.size] do
    let some a := addrs[lane]! | continue
    let a := a.toNat
    let line := a / lineBytes
    let off := a % lineBytes
    let mask := sectorRange (off / sectorBytes) ((min (off + max bytes 1) lineBytes - 1) / sectorBytes)
    -- Lanes usually walk addresses in order, so the line is most often the last one opened.
    let i := if out.back?.any (·.line == line) then some (out.size - 1) else out.findIdx? (·.line == line)
    match i with
    | some i => out := out.modify i fun g => { g with sectors := g.sectors ||| mask,
                                                      lanes := g.lanes.push lane }
    | none => out := out.push { line, sectors := mask, lanes := #[lane] }
  return out

/-- Number of set bits of a sector mask. -/
def popcount (m : Nat) : Nat := Id.run do
  let mut n := 0
  let mut m := m
  for _ in [0:64] do
    if m == 0 then break
    n := n + m % 2
    m := m / 2
  return n

/-- Shared-memory wavefronts of a request, and the wavefronts it would take without bank
conflicts. The banks serve `banks * bankBytes` bytes per wavefront, so lanes are taken in phases
of `banks * bankBytes / max bytes bankBytes` lanes (32 for 4-byte accesses, 16 for 8-byte, 8 for
16-byte); within a phase, lanes reading the same word share it (broadcast) and a bank holding `k`
distinct words takes `k` wavefronts (CUDA Programming Guide, shared memory of compute capability
5.x and later; the phases are our reading of its 64- and 128-bit rules). -/
def wavefronts (banks bankBytes bytes : Nat) (addrs : Array (Option UInt64)) : Nat × Nat := Id.run do
  let wordsPer := max 1 ((bytes + bankBytes - 1) / bankBytes)
  let phase := max 1 (banks / wordsPer)
  let mut total := 0
  let mut ideal := 0
  for p in [0:(addrs.size + phase - 1) / phase] do
    let mut ws : Array Nat := #[]
    for lane in [p * phase:min addrs.size ((p + 1) * phase)] do
      if let some a := addrs[lane]! then
        for k in [0:wordsPer] do ws := ws.push (a.toNat / bankBytes + k)
    if ws.isEmpty then continue
    -- Sorted by bank then word, so each bank's distinct words are adjacent.
    let sorted := ws.qsort fun x y => x % banks < y % banks || (x % banks == y % banks && x < y)
    let mut worst := 0
    let mut run := 0
    for i in [0:sorted.size] do
      let x := sorted[i]!
      if i == 0 || sorted[i - 1]! % banks != x % banks then run := 1
      else if sorted[i - 1]! != x then run := run + 1
      worst := max worst run
    total := total + worst
    ideal := ideal + 1
  return (total, ideal)

end NV.Core.Memory
