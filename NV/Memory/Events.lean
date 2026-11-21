import NV.Machine.Effect
import NV.Memory.Atomic
import NV.PTX.Memory.Model
import NV.Async.MBarrierWord

/-!
# Requests as memory-model events

How one lane of a request (`NV.Req`) becomes events of the PTX memory model, and how a list of
events becomes a candidate execution. Both the axiomatic program semantics (`NV.Formal.Outcomes`)
and the operational request machine (`NV.Memory.Flat`) build their executions here, so they agree
on what an access is.

The model relates operations on locations of one size (PTX ISA §8.2, §8.7.2; Lustig et al.), so
an access is split into events of a common granule where the ISA allows its parts to be observed
separately, and the granule is the 32-bit word, GPU memory's own:

* a weak `ld`/`st` into one event per word (or the whole access when smaller): weak accesses are
  not single-copy atomic (§8.10.3), and data written with one width and read with another (a TMA
  copy's words read by vector loads, a quad stored and read back as words, a reduction's
  workspace read by a plain load) stay within the model;
* a strong `ld`/`st` into one event per element (§8.2.4): the whole access when it is at most
  8 bytes, else words (a request does not say whether 16 bytes are two 8-byte or four 4-byte
  elements; words allow every tearing either does);
* an atomic or reduction into a read and a write of the whole access, joined by `rmw`.

Two events whose byte ranges overlap without coinciding are outside the model (`Uniform`): a
location accessed both as words and as bytes, halves or 64-bit atomics.

A location is a byte range of a state space instance: `owner` is the CTA of a shared address and
the thread of a local one.
-/

namespace NV

open NV.PTX (Event Kind Place Execution)

structure Loc where
  space : Space
  owner : Nat
  addr : Nat
  deriving DecidableEq, Repr, Inhabited

/-- One event of a program's or a machine's execution. `seq` is its position in its thread's
program order; `inst` numbers the instance of the barrier object `obj` an arrival or wait belongs
to, and `count` is the number of threads the instance expects. -/
structure Ev where
  e : Event
  seq : Nat
  loc : Loc := default
  size : Nat := 0
  val : List UInt8 := []
  /-- The read of a read-modify-write; its write is the thread's next event. -/
  rmwRead : Bool := false
  obj : Nat := 0
  inst : Nat := 0
  count : Nat := 0
  /-- Asynchronous issue order (PTX ISA §8.9.1.1): `(t, k)` orders this event after thread `t`'s
  events at positions below `k`. An asynchronous operation's events carry their issuer's position. -/
  after : List (Nat × Nat) := []
  deriving DecidableEq, Repr, Inhabited

namespace Ev

def key (v : Ev) : Loc × Nat := (v.loc, v.size)

/-- The byte ranges of two memory events overlap. -/
def overlaps (a b : Ev) : Bool :=
  a.e.isMem && b.e.isMem && a.loc.space == b.loc.space && a.loc.owner == b.loc.owner &&
    a.loc.addr < b.loc.addr + b.size && b.loc.addr < a.loc.addr + a.size

end Ev

/-! ## Bytes and words -/

/-- The number little-endian bytes spell. -/
def leNat (bs : List UInt8) : Nat := bs.foldr (fun b acc => b.toNat + 256 * acc) 0

/-- Little-endian bytes of a word. -/
def wordBytes (w : UInt32) : List UInt8 :=
  [w.toNat.toUInt8, (w.toNat / 2 ^ 8).toUInt8, (w.toNat / 2 ^ 16).toUInt8, (w.toNat / 2 ^ 24).toUInt8]

/-- Little-endian bytes of words. -/
def toBytes (ws : List UInt32) : List UInt8 := ws.flatMap wordBytes

/-- Words of little-endian bytes; a short last word is zero-extended. -/
def toWords : List UInt8 → List UInt32
  | [] => []
  | b :: bs => UInt32.ofNat (leNat ((b :: bs).take 4)) :: toWords ((b :: bs).drop 4)
termination_by bs => bs.length
decreasing_by simp; omega

theorem leNat_wordBytes (w : UInt32) : leNat (wordBytes w) = w.toNat := by
  have := w.toNat_lt
  simp only [leNat, wordBytes, List.foldr, Nat.toUInt8, UInt8.toNat_ofNat']
  omega

theorem length_toBytes (ws : List UInt32) : (toBytes ws).length = 4 * ws.length := by
  induction ws with
  | nil => rfl
  | cons w ws ih =>
    simp only [toBytes, List.flatMap_cons, List.length_append, List.length_cons] at ih ⊢
    rw [ih]; simp [wordBytes]; omega

/-- Byte `k` of `toBytes ws`: byte `k % 4` of word `k / 4`. -/
theorem getElem_toBytes (ws : List UInt32) (k : Nat) (h : k < (toBytes ws).length) :
    (toBytes ws)[k] = ((ws[k / 4]'(by rw [length_toBytes] at h; omega)).toNat / 2 ^ (8 * (k % 4))).toUInt8 := by
  induction ws generalizing k with
  | nil => simp [toBytes] at h
  | cons w ws ih =>
    have hw : (wordBytes w).length = 4 := rfl
    by_cases hk : k < 4
    · simp only [toBytes, List.flatMap_cons]
      rw [List.getElem_append_left (by rw [hw]; exact hk)]
      have h0 : k / 4 = 0 := by omega
      have h1 : k % 4 = k := by omega
      simp only [h0, h1, List.getElem_cons_zero]
      match k, hk with
      | 0, _ => simp [wordBytes]
      | 1, _ => rfl
      | 2, _ => rfl
      | 3, _ => rfl
    · have hl : (toBytes (w :: ws)).length = 4 + (toBytes ws).length := by
        rw [length_toBytes, length_toBytes, List.length_cons]; omega
      have h' : k - 4 < (toBytes ws).length := by omega
      have e := ih (k - 4) h'
      simp only [toBytes, List.flatMap_cons] at e ⊢
      rw [List.getElem_append_right (by rw [hw]; omega)]
      simp only [hw]
      rw [e]
      have h2 : k / 4 = (k - 4) / 4 + 1 := by omega
      have h3 : k % 4 = (k - 4) % 4 := by omega
      simp only [h2, h3, List.getElem_cons_succ]

theorem toWords_wordBytes (w : UInt32) : toWords (wordBytes w) = [w] := by
  obtain ⟨b₀, b₁, b₂, b₃, hb⟩ : ∃ b₀ b₁ b₂ b₃, wordBytes w = [b₀, b₁, b₂, b₃] := ⟨_, _, _, _, rfl⟩
  have hl := leNat_wordBytes w
  rw [hb] at hl ⊢
  simp only [toWords, List.take, List.drop, hl, UInt32.ofNat_toNat]

theorem leNat_lt {bs : List UInt8} : leNat bs < 256 ^ bs.length := by
  induction bs with
  | nil => simp [leNat]
  | cons b bs ih =>
    simp only [leNat, List.foldr, List.length_cons, Nat.pow_succ] at ih ⊢
    have := b.toNat_lt
    have : 256 * List.foldr (fun b acc => b.toNat + 256 * acc) 0 bs ≤ 256 * (256 ^ bs.length - 1) :=
      Nat.mul_le_mul_left _ (by omega)
    rw [Nat.mul_sub, Nat.mul_one] at this
    have : 256 ≤ 256 * 256 ^ bs.length := Nat.le_mul_of_pos_right _ (Nat.pow_pos (by decide))
    rw [Nat.mul_comm (256 ^ bs.length)]
    omega

/-- Four bytes are the bytes of the word they spell. -/
theorem wordBytes_leNat {bs : List UInt8} (h : bs.length = 4) :
    wordBytes (UInt32.ofNat (leNat bs)) = bs := by
  match bs, h with
  | [b₀, b₁, b₂, b₃], _ =>
    have := b₀.toNat_lt; have := b₁.toNat_lt; have := b₂.toNat_lt; have := b₃.toNat_lt
    have hl : leNat [b₀, b₁, b₂, b₃] =
        b₀.toNat + 256 * (b₁.toNat + 256 * (b₂.toNat + 256 * b₃.toNat)) := by
      simp [leNat]
    have hs : UInt32.size = 4294967296 := rfl
    have hof : (UInt32.ofNat (b₀.toNat + 256 * (b₁.toNat + 256 * (b₂.toNat + 256 * b₃.toNat)))).toNat =
        b₀.toNat + 256 * (b₁.toNat + 256 * (b₂.toNat + 256 * b₃.toNat)) :=
      UInt32.toNat_ofNat_of_lt' (by rw [hs]; omega)
    rw [hl]
    simp only [wordBytes, hof, List.cons.injEq, and_true]
    refine ⟨?_, ?_, ?_, ?_⟩ <;> apply UInt8.toNat_inj.1 <;>
      simp only [Nat.toUInt8, UInt8.toNat_ofNat'] <;> omega

/-! ## The events of one lane -/

/-- One event of a lane's access: a read or a write of `size` bytes at offset `off`. -/
structure Piece where
  write : Bool
  off : Nat
  size : Nat
  rmwRead : Bool := false
  deriving DecidableEq, Repr

namespace Req

variable (r : Req)

def strong : Bool := r.sem != .weak

/-- The access's elements when it is strong. -/
def elemBytes : Nat := if r.bytes ≤ 8 then r.bytes else 4

/-- The size of each event of an `ld`/`st` lane. -/
def granule : Nat := if r.strong then r.elemBytes else min 4 r.bytes

def pieces : List Piece :=
  match r.op with
  | .ld | .st =>
    let k := r.granule
    (List.range (r.bytes / k)).map fun j => { write := r.op == .st, off := j * k, size := k }
  | .atom .. | .red .. | .mbar _ => [{ write := false, off := 0, size := r.bytes, rmwRead := true },
                                     { write := true, off := 0, size := r.bytes }]

/-- `volatile` is `relaxed` at system scope (§8.4.2); a weak access is strong only within its
thread. -/
def eventScope : PTX.Scope :=
  match r.sem with
  | .weak => .thread
  | .volatile => .sys
  | _ => r.scope

def acquires : Bool := r.sem == .acquire || r.sem == .acqRel
def releases : Bool := r.sem == .release || r.sem == .acqRel
def isRed : Bool := match r.op with | .red .. => true | _ => false

/-- How many bytes a lane's reads return: the whole access when it reads. -/
def readLen : Nat := if r.pieces.any (!·.write) then r.bytes else 0

/-- The memory-model event of piece `p`, by thread `thread`. -/
def event (thread : Nat) (p : Piece) : Event :=
  { thread, kind := if p.write then .write else .read, scope := r.eventScope,
    acquire := !p.write && r.acquires, release := p.write && r.releases,
    reduction := !p.write && r.isRed, proxy := r.proxy }

/-- Words of lane `lane`'s operands: `words bytes` per lane, twice that for `cas`. -/
def operands (lane : Nat) : List UInt32 :=
  let n := words r.bytes
  match r.op with
  | .st => (r.data.extract (lane * n) (lane * n + n)).toList
  | .atom op _ | .red op _ =>
    let per := if op == .cas then 2 * n else n
    (r.data.extract (lane * per) (lane * per + per)).toList
  | .ld | .mbar _ => []

/-- The bytes an `st` lane writes. -/
def storeBytes (lane : Nat) : List UInt8 := (toBytes (r.operands lane)).take r.bytes

/-- The value an atomic or reduction lane writes, given the value it read; `none` for a type the
library does not define yet. -/
def atomNew (lane : Nat) (old : List UInt8) : Option (List UInt8) :=
  match r.op with
  | .atom op ty | .red op ty =>
    let n := words r.bytes
    let ops := (r.operands lane).toArray
    (op.apply ty (toWords old).toArray (ops.extract 0 n) (ops.extract n (2 * n))).map
      fun ws => (toBytes ws.toList).take r.bytes
  -- An mbarrier operation on its 8-byte word; behaviour the ISA leaves undefined has no value.
  | .mbar op =>
    match Async.MBarrierWord.apply op (UInt64.ofNat (leNat old)) with
    | .ok w => some (wordBytes w.toUInt32 ++ wordBytes (w >>> 32).toUInt32)
    | .error _ => none
  | _ => none

/-- Bytes `off` to `off + size` of `storeBytes lane` (`storeSlice_eq`), each read off its operand
word: a piece's bytes without the lane's whole byte list (a 16-byte store's four pieces built it
four times). -/
def storeSlice (lane off size : Nat) : List UInt8 :=
  let ws := r.operands lane
  let len := min r.bytes (4 * ws.length)
  (List.range (min size (len - off))).map fun j =>
    let k := off + j
    (ws[k / 4]!.toNat / 2 ^ (8 * (k % 4))).toUInt8

/-- The value piece `p` of lane `lane` writes, given the bytes the lane's reads returned. -/
def written (lane : Nat) (p : Piece) (read : List UInt8) : Option (List UInt8) :=
  match r.op with
  | .st => some (r.storeSlice lane p.off p.size)
  | _ => r.atomNew lane read

/-- The lane's response words to the issuer: the bytes its reads returned, for `ld` and `atom`. -/
def laneResp (read : List UInt8) : List UInt32 :=
  if r.op.returns then (toWords read).take (words r.bytes) ++
    List.replicate (words r.bytes - (toWords read).length) 0 else []

/-- The whole response: per lane `words bytes` words, inactive lanes zero. -/
def resp (read : Nat → List UInt8) : Resp :=
  if r.op.returns then
    ((List.range r.addrs.size).flatMap fun l =>
      if (r.addrs[l]?).join.isSome then r.laneResp (read l)
      else List.replicate (words r.bytes) 0).toArray
  else #[]

/-- The events lane `lane` of `r` performs as thread `thread` at location `l`, the first at
position `seq` of the thread's program order, its reads having returned `read`. -/
def laneEvents (lane thread seq : Nat) (l : Loc) (read : List UInt8) : List Ev :=
  r.pieces.zipIdx.map fun (p, i) =>
    { e := r.event thread p, seq := seq + i, loc := { l with addr := l.addr + p.off },
      size := p.size, rmwRead := p.rmwRead,
      val := if p.write then (r.written lane p read).getD [] else read.extract p.off (p.off + p.size) }


/-- Lane `l` of `r` is active. -/
def active (l : Nat) : Bool := (r.addrs[l]?).join.isSome

/-- The request writes no value for what its reads returned: an atomic on a type the library does
not define yet (a floating-point atomic), or an mbarrier operation the ISA leaves undefined. -/
def unsupported (read : Nat → List UInt8) : Bool :=
  r.op.writes && r.op != .st &&
    (List.range r.addrs.size).any fun l => r.active l && (r.atomNew l (read l)).isNone

theorem storeSlice_eq (lane off size : Nat) :
    r.storeSlice lane off size = (r.storeBytes lane).extract off (off + size) := by
  apply List.ext_getElem
  · simp [storeSlice, storeBytes, length_toBytes]
  · intro j h₁ h₂
    simp only [storeSlice, List.getElem_map, List.getElem_range]
    simp only [storeBytes, List.extract_eq_take_drop, List.getElem_take, List.getElem_drop]
    rw [getElem_toBytes]
    congr 3
    have hb : (off + j) / 4 < (r.operands lane).length := by simp [storeSlice] at h₁; omega
    exact getElem!_pos (r.operands lane) ((off + j) / 4) hb

end Req

/-- A barrier event: an arrival (`wait = false`) or a wait, on named barrier `id` of the CTA or
(`none`) the cluster barrier, joining instance `inst` of barrier object `obj`, which expects
`count` threads. -/
def barEvent (thread seq : Nat) (id : Option Nat) (wait : Bool) (obj inst count : Nat) : Ev :=
  { e := { thread, scope := if id.isSome then .cta else .cluster,
           kind := (match id, wait with
             | some i, false => .barArrive i | some i, true => .barWait i
             | none, false => .clusterArrive | none, true => .clusterWait) },
    seq, obj, inst, count }

def isFenceOp : SyncOp → Bool
  | .fence .. | .proxyFence _ | .aliasFence => true
  | _ => false

/-- The event of a fence. -/
def fenceEvent (op : SyncOp) (thread seq : Nat) : Ev :=
  { e := { thread,
           kind := (match op with
             | .fence true _ => .fenceSC | .proxyFence p => .proxyFence p | .aliasFence => .aliasFence
             | _ => .fenceAcqRel),
           scope := (match op with | .fence _ s => s | _ => .thread) },
    seq }

/-- The events of a synchronizing operation by thread `thread`, the first at position `seq` of its
program order; a barrier's arrival and wait join instance `inst` of barrier object `obj`, which
expects `count` threads. -/
def SyncOp.events (op : SyncOp) (thread seq obj inst count : Nat) : List Ev :=
  match op with
  | .bar id _ arrive =>
    barEvent thread seq (some id) false obj inst count ::
      if arrive then [] else [barEvent thread (seq + 1) (some id) true obj inst count]
  | .clusterArrive => [barEvent thread seq none false obj inst count]
  | .clusterWait => [barEvent thread seq none true obj inst count]
  -- Group commits and waits order the issuer against its asynchronous operations; their events
  -- belong to the asynchronous operations' model, not yet given.
  | .asyncCommit _ | .asyncWait .. => []
  | op => [fenceEvent op thread seq]

/-- The barrier object a barrier operation of a thread at `place` names: its CTA's named barrier
`id` (PTX ISA §9.7.13.1: `id < 16`), or its cluster's barrier. Distinct barriers get distinct
objects. -/
def SyncOp.barObj (place : PTX.Place) : SyncOp → Nat
  | .bar id _ _ => 2 * (place.cta * 16 + id)
  | _ => 2 * place.cluster + 1

namespace Ev

/-- Barrier instance of a barrier event: kind of barrier, CTA or cluster, barrier id, instance. -/
def arrives (v : Ev) : Bool :=
  match v.e.kind with
  | .barArrive _ | .clusterArrive => true
  | _ => false

def waits (v : Ev) : Bool :=
  match v.e.kind with
  | .barWait _ | .clusterWait => true
  | _ => false

/-- The barrier instance a barrier event joins: its object and instance. -/
def barKey (v : Ev) : Option (Nat × Nat) :=
  if v.arrives || v.waits then some (v.obj, v.inst) else none

/-- Event `v` writes byte `l`. -/
def covers (v : Ev) (l : Loc) : Bool :=
  v.e.isWrite && v.loc.space == l.space && v.loc.owner == l.owner && v.loc.addr ≤ l.addr &&
    l.addr < v.loc.addr + v.size

end Ev

theorem covers_iff {v : Ev} {L : Loc} : v.covers L = true ↔ v.e.isWrite = true ∧
    v.loc.space = L.space ∧ v.loc.owner = L.owner ∧ v.loc.addr ≤ L.addr ∧ L.addr < v.loc.addr + v.size := by
  simp [Ev.covers, and_assoc]

theorem overlaps_iff {a b : Ev} : a.overlaps b = true ↔ a.e.isMem = true ∧ b.e.isMem = true ∧
    a.loc.space = b.loc.space ∧ a.loc.owner = b.loc.owner ∧ a.loc.addr < b.loc.addr + b.size ∧
    b.loc.addr < a.loc.addr + a.size := by
  simp [Ev.overlaps, and_assoc]

/-! ## Executions of event lists -/

section
variable (evs : List Ev)

/-- No two memory events overlap without accessing the same bytes. -/
def Uniform : Prop :=
  ∀ a ∈ evs, ∀ b ∈ evs, a.overlaps b = true → a.key = b.key

/-- The initial bytes of the location of `key`. -/
def initBytes (init : Loc → UInt8) (key : Loc × Nat) : List UInt8 :=
  (List.range key.2).map fun i => init { key.1 with addr := key.1.addr + i }

/-- Locations are numbered by their key's first occurrence. -/
def code (v : Ev) : Nat := (evs.map Ev.key).idxOf v.key

/-- The candidate execution of `evs` with the chosen reads-from, coherence and Fence-SC orders.
Program order is `seq` order within a thread; a read-modify-write's read and write are adjacent;
an arrival relates to the waits of its barrier instance. -/
def toExec (place : Nat → Place) (init : Loc → UInt8) (rf co sc : Rel evs.length) :
    Execution evs.length (List UInt8) where
  ev i := { evs[i].e with addr := code evs evs[i] }
  place := place
  init c := match (evs.map Ev.key)[c]? with
    | some key => initBytes init key
    | none => []
  val i := evs[i].val
  po := Rel.ofFn fun a b => evs[a].e.thread == evs[b].e.thread && evs[a].seq < evs[b].seq
  rf := rf
  co := co
  sc := sc
  rmw := Rel.ofFn fun a b => evs[a].rmwRead && evs[a].e.thread == evs[b].e.thread &&
    evs[b].seq == evs[a].seq + 1
  issue := Rel.ofFn fun a b => evs[b].after.any fun (t, k) => evs[a].e.thread == t && decide (evs[a].seq < k)
  dep := Rel.ofFn fun a b => evs[a].rmwRead && evs[a].e.thread == evs[b].e.thread &&
    evs[b].seq == evs[a].seq + 1
  bar := Rel.ofFn fun a b =>
    evs[a].obj == evs[b].obj && evs[a].inst == evs[b].inst && match evs[a].e.kind, evs[b].e.kind with
    | .barArrive i, .barWait j =>
      i == j && (place evs[a].e.thread).cta == (place evs[b].e.thread).cta
    | .clusterArrive, .clusterWait =>
      (place evs[a].e.thread).cluster == (place evs[b].e.thread).cluster
    | _, _ => false

end

end NV
