import NV.Core.Engine
import NV.Memory.Store

/-!
# The emulated device's launch-result cache

A timed launch can take hours to simulate, and a program of many launches (a training step through
XLA) loses all of them when its container is recalled. `NVGPU_LEAN_LAUNCH_CACHE` names a directory
each launch's result is written to as it finishes: its cycles, the machine after it (caches, clock)
and the device-memory pages it changed. `NVGPU_LEAN_LAUNCH_CACHE_READ` names more directories,
`:`-separated, read but not written (an earlier attempt's, unpacked by armada at `$ARMADA_RESUME`;
an earlier job's, passed with `--inputs`). A launch whose key is found is not simulated: its result
is applied as recorded.

The key is a 128-bit digest (`Digest`: Murmur's `ByteArray.hash` and FNV-1a, independently) of
everything a result depends on: the model's code (the bytes of the loaded libraries, `version`),
the model's whole configuration (`reprStr` of the part's `Params`: every fact, override and clock,
the link model), the kernel (its module's listings and placement, its name, its constant banks),
the launch (shapes, parameter bytes, dynamic shared memory, functional or timed, workers), the
machine state it starts from, and the device memory it reads. -/

namespace NV.Device.Cache

open NV NV.Core

/-- FNV-1a, 64 bits, continued from `h` over `b`. -/
def fnv (h : UInt64) (b : ByteArray) : UInt64 := Id.run do
  let mut h := h
  for i in [0:b.size] do
    h := (h ^^^ b[i]!.toUInt64) * 0x100000001b3
  return h

/-- A base-128 natural number appended to `b`. -/
def putNat (b : ByteArray) (n : Nat) : ByteArray := Id.run do
  let mut b := b
  let mut n := n
  while n ≥ 128 do
    b := b.push (n % 128 + 128).toUInt8
    n := n / 128
  return b.push n.toUInt8

/-- Bytes appended to `b`, their length first. -/
def putBytes (b x : ByteArray) : ByteArray := putNat b x.size ++ x

/-- Two independent 64-bit hashes of a sequence of byte strings, each length-prefixed. -/
structure Digest where
  a : UInt64 := 7
  b : UInt64 := 0xcbf29ce484222325
  deriving BEq, Inhabited

def Digest.add (d : Digest) (x : ByteArray) : Digest :=
  let n := putNat .empty x.size
  { a := mixHash (mixHash d.a (ByteArray.hash n)) (ByteArray.hash x), b := fnv (fnv d.b n) x }

def Digest.nat (d : Digest) (n : Nat) : Digest := d.add (putNat .empty n)

def Digest.str (d : Digest) (s : String) : Digest := d.add s.toUTF8

def Digest.hex (d : Digest) : String :=
  let h (x : UInt64) : String :=
    let s := String.mk (Nat.toDigits 16 x.toNat)
    String.mk (List.replicate (16 - s.length) '0') ++ s
  h d.a ++ h d.b

/-! ## The machine's state, encoded -/

private def putLevel (b : ByteArray) (lv : CodeLevel) : ByteArray :=
  let ls := lv.lines.toArray.qsort (·.1 < ·.1)
  let b := putNat (putNat (putNat b lv.fill.free) lv.capacity) lv.lineCycles
  ls.foldl (fun b (l, u) => putNat (putNat b l) u) (putNat b ls.size)

private def putTags (b : ByteArray) (t : Memory.Tags) : ByteArray :=
  let b := putNat (putNat (putNat (putNat b t.sets) t.ways) t.clock) t.set.size
  t.set.foldl (fun b ws => ws.foldl (fun b w =>
    putNat (putNat (putNat (putNat b w.line) w.valid) w.dirty) w.used) (putNat b ws.size)) b

/-- The machine's whole state as bytes, its code caches' lines in line order (the order their
eviction does not depend on, `CodeLevel.touch`). -/
def encodeMachine (m : Machine) : ByteArray :=
  let b := m.code.foldl (fun b lvs => lvs.foldl putLevel (putNat b lvs.size)) (putNat .empty m.code.size)
  let b := putNat (m.l2.foldl putTags (putNat b m.l2.size)) m.clock
  m.modules.foldl (fun b x => putNat b x.toNat) (putNat b m.modules.length)

/-- A reader of what `putNat` and `putBytes` wrote. -/
abbrev Dec := StateT Nat (ReaderT ByteArray (Except String))

def getNat : Dec Nat := do
  let buf ← read
  let mut n := 0
  let mut shift := 0
  for _ in [0:buf.size] do
    let p ← get
    let some x := buf[p]? | throw "a launch-cache entry ends early"
    set (p + 1)
    n := n + (x.toNat % 128) * 2 ^ shift
    if x < 128 then return n
    shift := shift + 7
  throw "a launch-cache entry ends early"

def getBytes : Dec ByteArray := do
  let n ← getNat
  let buf ← read
  let p ← get
  if p + n > buf.size then throw "a launch-cache entry ends early"
  set (p + n)
  return buf.extract p (p + n)

private def getArray {α : Type} (f : Dec α) : Dec (Array α) := do
  let n ← getNat
  let mut out := #[]
  for _ in [0:n] do out := out.push (← f)
  return out

private def getLevel : Dec CodeLevel := do
  let free ← getNat
  let capacity ← getNat
  let lineCycles ← getNat
  let ls ← getArray (do return ((← getNat), (← getNat)))
  return { lines := ls.foldl (fun m (l, u) => m.insert l u) {}, capacity, lineCycles, fill := { free } }

private def getTags : Dec Memory.Tags := do
  let sets ← getNat
  let ways ← getNat
  let clock ← getNat
  let set ← getArray (getArray (do
    return { line := ← getNat, valid := ← getNat, dirty := ← getNat, used := ← getNat }))
  return { sets, ways, set, clock }

def getMachine : Dec Machine := do
  let code ← getArray (getArray getLevel)
  let l2 ← getArray getTags
  let clock ← getNat
  let modules ← getArray (do return (← getNat).toUInt64)
  return { code, l2, clock, modules := modules.toList }

/-! ## Device memory -/

/-- Each page's hash (0 unwritten) and the digest `d` continued over the store's pages. -/
def storeDigest (d : Digest) (s : Store) : Digest × Array UInt64 := Id.run do
  let mut d := (d.nat s.base.toNat).nat s.size |>.nat s.npages
  let mut hs := Array.replicate s.npages 0
  for i in [0:s.npages] do
    if let some p := s.page i then
      d := (d.nat (i + 1)).add p
      hs := hs.set! i (mixHash (ByteArray.hash p) 1)
  return (d, hs)

/-- The pages of `out` that differ from the store whose page hashes are `before`: index and
contents (`none`: unwritten). -/
def changedPages (before : Array UInt64) (out : Store) : Array (Nat × Option ByteArray) := Id.run do
  let mut ch := #[]
  for i in [0:out.npages] do
    let h := match out.page i with | some p => mixHash (ByteArray.hash p) 1 | none => 0
    if h != before.getD i 0 then ch := ch.push (i, out.page i)
  return ch

/-- `s` grown to the output's range (a launch covers its parameter block, `Engine`), then `pages`
(indexed from the output's base) set. -/
def applyPages (s : Store) (base size : Nat) (pages : Array (Nat × Option ByteArray)) : Store := Id.run do
  let mut s := s.cover base.toUInt64 size
  for (i, p) in pages do
    s := s.setPage i p
  return s

/-! ## Entries -/

/-- What a launch left: its cycles, the machine after it, the device memory's range after it and
the pages it changed. -/
structure Entry where
  cycles : Nat
  machine : Machine
  base : Nat
  size : Nat
  pages : Array (Nat × Option ByteArray)

def magic : ByteArray := "nvgpu-lean launch cache 2\n".toUTF8

def Entry.encode (e : Entry) (key : String) : ByteArray := Id.run do
  let mut b := putBytes (putNat (putBytes .empty (magic ++ key.toUTF8)) e.cycles) (encodeMachine e.machine)
  b := putNat (putNat (putNat b e.base) e.size) e.pages.size
  for (i, p) in e.pages do
    b := match p with
      | some p => putBytes (putNat (putNat b i) 1) p
      | none => putNat (putNat b i) 0
  return b

def Entry.decode (key : String) (buf : ByteArray) : Except String Entry := do
  let go : Dec Entry := do
    let head ← getBytes
    if head != magic ++ key.toUTF8 then throw "a launch-cache entry for another key"
    let cycles ← getNat
    let mb ← getBytes
    let machine ← (getMachine.run 0).run mb |>.map (·.1)
    let base ← getNat
    let size ← getNat
    let pages ← getArray (do
      let i ← getNat
      if (← getNat) == 1 then return (i, some (← getBytes)) else return (i, none))
    return { cycles, machine, base, size, pages }
  (go.run 0).run buf |>.map (·.1)

/-! ## Where entries live -/

/-- The directory entries are written to, and the directories they are also read from. -/
structure Stores where
  dir : System.FilePath
  reads : List System.FilePath := []

/-- From `NVGPU_LEAN_LAUNCH_CACHE` and `NVGPU_LEAN_LAUNCH_CACHE_READ`. -/
def Stores.fromEnv : IO (Option Stores) := do
  let some dir ← IO.getEnv "NVGPU_LEAN_LAUNCH_CACHE" | return none
  let reads := ((← IO.getEnv "NVGPU_LEAN_LAUNCH_CACHE_READ").getD "").splitOn ":"
    |>.filter (· != "") |>.map System.FilePath.mk
  return some { dir, reads }

def Stores.get (s : Stores) (key : String) : IO (Option ByteArray) := do
  for d in s.dir :: s.reads do
    let f := d / (key ++ ".nvlc")
    if ← f.pathExists then return some (← IO.FS.readBinFile f)
  return none

/-- Written whole: to a temporary name, then renamed, so an archive of the directory never holds
half an entry. -/
def Stores.put (s : Stores) (key : String) (bytes : ByteArray) : IO Unit := do
  IO.FS.createDirAll s.dir
  let tmp := s.dir / (key ++ ".nvlc.part")
  IO.FS.writeBinFile tmp bytes
  IO.FS.rename tmp (s.dir / (key ++ ".nvlc"))

/-! ## The model's own code -/

/-- The model's code: the bytes of the loaded libraries it runs in (the driver, the Lean model's
libraries, the Lean runtime), from `/proc/self/maps`. -/
def libraries : IO Digest := do
  let maps ← IO.FS.readFile "/proc/self/maps"
  let paths := maps.splitOn "\n" |>.filterMap fun (l : String) =>
    let p := (l.splitOn " ").getLast!
    let base := (p.splitOn "/").getLast!
    if p.startsWith "/" && (base.startsWith "libcuda.so" || base.startsWith "libleanshared"
      || ((base.endsWith "NV.so" || base.endsWith "Device.so") && base.startsWith "lib"))
    then some p else none
  let paths := paths.eraseDups.toArray.qsort (fun a b => a < b)
  if paths.isEmpty then throw (IO.userError "no model library found in /proc/self/maps")
  let mut d : Digest := {}
  for p in paths do
    d := d.add (← IO.FS.readBinFile p)
  return d

initialize versionRef : IO.Ref (Option Digest) ← IO.mkRef none

/-- `libraries`, read once a process. -/
def version : IO Digest := do
  if let some d ← versionRef.get then return d
  let d ← libraries
  versionRef.set (some d)
  return d

end NV.Device.Cache
