import NV.Machine.Access

/-!
# Byte stores

The data of one state space: a contiguous byte range starting at `base`, held in 64 KB pages that
are allocated on first write (unwritten bytes read as zero). Global memory, each CTA's shared
memory and each thread's local memory are stores. Updates are in place as long as the store is
used linearly, which the simulator does; a write to a store that is not (a functional worker's view
of global memory) copies the page table's chunk index and the chunk it writes, `chunkPages`
entries, not an entry for every page (an emulated B200's 196 GB is 3M pages).
-/

namespace NV

def pageBits : Nat := 16
def pageSize : Nat := 1 <<< pageBits

def chunkBits : Nat := 10
def chunkPages : Nat := 1 <<< chunkBits

structure Store where
  base : UInt64
  size : Nat
  /-- Pages: `npages`, page `i` at `chunks[i / chunkPages][i % chunkPages]` (`none`: unwritten). -/
  npages : Nat
  chunks : Array (Array (Option ByteArray))
  deriving Inhabited

/-- The zero page; never mutated in place, since every store that uses it shares it. -/
def zeroPage : ByteArray := ⟨Array.replicate pageSize 0⟩

namespace Store

/-- Page `i`, if written. -/
def page (s : Store) (i : Nat) : Option ByteArray :=
  if i < s.npages then ((s.chunks[i >>> chunkBits]?).bind (·[i % chunkPages]?)).join else none

/-- Page `i` set to `p` (taken out of its chunk first, so a store used linearly updates in place). -/
def setPage (s : Store) (i : Nat) (p : Option ByteArray) : Store :=
  if i < s.npages then
    match s.chunks[i >>> chunkBits]? with
    | some ch =>
      let s' := { s with chunks := s.chunks.set! (i >>> chunkBits) #[] }
      { s' with chunks := s'.chunks.set! (i >>> chunkBits) (ch.set! (i % chunkPages) p) }
    | none => s
  else s

/-- The page table of `n` pages: whole chunks, and a last one of the pages left (a small store, a
CTA's shared memory, has a table as small as its pages, which a copy of it copies). -/
def emptyChunks (n : Nat) : Array (Array (Option ByteArray)) :=
  (Array.range ((n + chunkPages - 1) / chunkPages)).map fun c => Array.replicate (min chunkPages (n - c * chunkPages)) none

def empty (base : UInt64) (size : Nat) : Store :=
  let n := (size + pageSize - 1) / pageSize
  { base, size, npages := n, chunks := emptyChunks n }

def contains (s : Store) (addr : UInt64) (bytes : Nat) : Bool :=
  s.base ≤ addr && (addr - s.base).toNat + bytes ≤ s.size

def readByte (s : Store) (off : Nat) : UInt8 :=
  match s.page (off >>> pageBits) with
  | some p => p.get! (off % pageSize)
  | none => 0

def writeByte (s : Store) (off : Nat) (b : UInt8) : Store :=
  let i := off >>> pageBits
  if i < s.npages then
    let page := s.page i
    -- The page's slot cleared before it is written, so the page is unshared and written in place.
    -- An unwritten page starts as a copy of the shared zero page (copied once, on its first write).
    (s.setPage i none).setPage i (some ((page.getD zeroPage).set! (off % pageSize) b))
  else s

/-- The 32-bit little-endian word at byte offset `off` (any alignment). A word inside one page
is read from that page directly. -/
def readWord (s : Store) (off : Nat) : UInt32 :=
  let o := off % pageSize
  if o + 4 ≤ pageSize then
    match s.page (off >>> pageBits) with
    | some p =>
      (p.get! o).toUInt32 ||| ((p.get! (o + 1)).toUInt32 <<< 8) |||
        ((p.get! (o + 2)).toUInt32 <<< 16) ||| ((p.get! (o + 3)).toUInt32 <<< 24)
    | none => 0
  else
    (s.readByte off).toUInt32 ||| ((s.readByte (off + 1)).toUInt32 <<< 8) |||
      ((s.readByte (off + 2)).toUInt32 <<< 16) ||| ((s.readByte (off + 3)).toUInt32 <<< 24)

def writeWord (s : Store) (off : Nat) (w : UInt32) : Store :=
  ((((s.writeByte off w.toUInt8).writeByte (off + 1) (w >>> 8).toUInt8).writeByte (off + 2)
    (w >>> 16).toUInt8).writeByte (off + 3) (w >>> 24).toUInt8)

/-- `bytes` bytes at `addr` as words, little-endian; the last word holds a sub-word access in
its low bits. -/
def read (s : Store) (addr : UInt64) (bytes : Nat) : Array UInt32 := Id.run do
  let off := (addr - s.base).toNat
  let mut out := Array.mkEmpty (words bytes)
  for w in [0:words bytes] do
    if bytes ≥ 4 * w + 4 then out := out.push (s.readWord (off + 4 * w))
    else
      let mut v : UInt32 := 0
      for b in [0:bytes - 4 * w] do
        v := v ||| ((s.readByte (off + 4 * w + b)).toUInt32 <<< (8 * b).toUInt32)
      out := out.push v
  return out

/-- Write `bytes` bytes at `addr` from `ws[at]`, `ws[at+1]`, … -/
def write (s : Store) (addr : UInt64) (bytes : Nat) (ws : Array UInt32) (at_ : Nat) : Store := Id.run do
  let off := (addr - s.base).toNat
  let mut s := s
  for w in [0:words bytes] do
    let v := ws[at_ + w]!
    if bytes ≥ 4 * w + 4 then s := s.writeWord (off + 4 * w) v
    else
      for b in [0:bytes - 4 * w] do
        s := s.writeByte (off + 4 * w + b) (v >>> (8 * b).toUInt32).toUInt8
  return s

/-- `data` written at byte offset `off`, a page at a time (the host's copies into device memory:
megabytes per call, which `writeByte` would take one at a time). Bytes past the store's end are
dropped. -/
def writeBytes (s : Store) (off : Nat) (data : ByteArray) : Store := Id.run do
  let mut s := s
  let mut done := 0
  while done < data.size do
    let o := off + done
    let i := o >>> pageBits
    if i < s.npages then
      let po := o % pageSize
      let n := min (pageSize - po) (data.size - done)
      let page := s.page i
      s := s.setPage i none
      let p := data.copySlice done (page.getD zeroPage) po n
      s := s.setPage i (some p)
      done := done + n
    else break
  return s

/-- `n` bytes from byte offset `off`, a page at a time; unwritten pages read as zero. -/
def readBytes (s : Store) (off n : Nat) : ByteArray := Id.run do
  let mut out := ByteArray.mk (Array.replicate n 0)
  let mut done := 0
  while done < n do
    let o := off + done
    let po := o % pageSize
    let k := min (pageSize - po) (n - done)
    if let some p := s.page (o >>> pageBits) then
      out := p.copySlice po out done k
    done := done + k
  return out

/-- `n` bytes from byte offset `off` set to `v`, a page at a time; zero over unwritten pages
leaves them unwritten. -/
def fill (s : Store) (off n : Nat) (v : UInt8) : Store := Id.run do
  let mut s := s
  let mut done := 0
  while done < n do
    let o := off + done
    let i := o >>> pageBits
    let po := o % pageSize
    let k := min (pageSize - po) (n - done)
    if i < s.npages then
      let page := s.page i
      if !(v == 0 && page.isNone) then
        s := s.setPage i none
        let mut p := page.getD zeroPage
        for j in [po:po + k] do p := p.set! j v
        s := s.setPage i (some p)
    else break
    done := done + k
  return s

/-- The whole pages within `n` bytes from byte offset `off` returned unwritten (freed device
memory gives its host memory back; it reads as zero until written again). -/
def release (s : Store) (off n : Nat) : Store := Id.run do
  let first := (off + pageSize - 1) / pageSize
  let last := (off + n) / pageSize
  let mut s := s
  for i in [first:min last s.npages] do
    s := s.setPage i none
  return s

/-- Stores laid end to end as one, each starting where the one before ends, page tables moved
(not copied) where every store but the last fills whole chunks of the table (a mesh's devices'
memories, `Device.launchMesh`); `none` if they are not so laid out. -/
def concat (ss : Array Store) : Option Store := Id.run do
  let some s0 := ss[0]? | return none
  let mut chunks : Array (Array (Option ByteArray)) := #[]
  let mut size := 0
  for s in ss do
    if s.base.toNat != s0.base.toNat + size then return none
    if size % (pageSize * chunkPages) != 0 then return none
    chunks := chunks ++ s.chunks
    size := size + s.size
  let npages := (size + pageSize - 1) / pageSize
  return some { base := s0.base, size, npages, chunks }

/-- `concat`'s inverse: the stores of `sizes` bytes each, from the store's base, their page tables
taken from it. -/
def split (s : Store) (sizes : Array Nat) : Array Store := Id.run do
  let mut out := #[]
  let mut base := s.base.toNat
  let mut c := 0
  for size in sizes do
    let n := (size + pageSize - 1) / pageSize
    let nc := (n + chunkPages - 1) / chunkPages
    out := out.push { base := base.toUInt64, size, npages := n, chunks := s.chunks.extract c (c + nc) }
    base := base + size
    c := c + nc
  return out

/-- The store extended, if needed, to hold `bytes` bytes at `addr`: the range grows to whole pages
around both, and the existing contents stay. -/
def cover (s : Store) (addr : UInt64) (bytes : Nat) : Store := Id.run do
  let lo := min (s.base.toNat / pageSize * pageSize) (addr.toNat / pageSize * pageSize)
  let hi := max (s.base.toNat + s.size) (addr.toNat + bytes)
  if lo == s.base.toNat && hi == s.base.toNat + s.size then return s
  -- Pages move by whole pages when the old base is page-aligned; otherwise the bytes are copied.
  if s.base.toNat % pageSize == 0 then
    let shift := (s.base.toNat - lo) / pageSize
    let mut t := empty lo.toUInt64 (hi - lo)
    for i in [0:s.npages] do
      if let some p := s.page i then t := t.setPage (i + shift) (some p)
    return t
  let mut t := empty lo.toUInt64 (hi - lo)
  for i in [0:s.size] do
    let b := s.readByte i
    if b != 0 then t := t.writeByte (s.base.toNat - lo + i) b
  return t

end Store

end NV
