import NV.PTX.Memory.Execution

/-!
# Memory requests

The vocabulary PTX and SASS semantics use to touch memory, and the memory machine and the
timing core consume. One request is what one instruction issues: an access per active lane, all
of one kind. A thread-level (PTX) access is a request with one lane. Scopes and proxies are the
memory model's (`NV.PTX.Scope`, `NV.PTX.Proxy`), so every request maps to memory-model events.

Data are 32-bit words, little-endian, lane-major: a lane's access of `bytes` bytes occupies
`words bytes` consecutive words; sub-word data sit in the low bits.
-/

namespace NV

open NV.PTX (Scope Proxy)

/-- State spaces (PTX ISA §5.1), after generic addresses are resolved. -/
inductive Space where
  | global
  /-- The issuing CTA's shared memory. -/
  | shared
  /-- Shared memory of a CTA in the cluster (`shared::cluster`): the address names the CTA. -/
  | sharedCluster
  | «local»
  /-- The SM's tensor memory (`sm_100a`, PTX ISA 9.5 §9.7.18.1): 128 lanes of 512 32-bit columns,
  column `c` of lane `l` at byte `4 (512 l + c)`, then the word of its column allocator
  (`NV.Async.Tcgen05`). The CTAs on an SM share it; allocation keeps them apart. -/
  | tmem
  /-- A constant bank: `c[bank][…]`. -/
  | const (bank : Nat)
  | param
  deriving DecidableEq, Repr, Inhabited, Hashable

/-- The bytes of an SM's `.tmem` store: 128 lanes of 512 4-byte columns, then the column
allocator's word (`NV.Async.Tcgen05.storeBytes`). -/
def Space.tmemBytes : Nat := 4 * 128 * 512 + 4

/-- Memory ordering semantics (PTX ISA §8.4). `weak` is a plain `ld`/`st`. -/
inductive Sem where
  | weak | relaxed | acquire | release | acqRel | volatile
  deriving DecidableEq, Repr, Inhabited

/-- The value type an atomic operation interprets its words as. `f32ftz` is binary32 with
subnormal inputs and results flushed to zero (PTX `atom.add.f32` without `.noftz` on global
memory, SASS `.FTZ`); `f32` keeps them. -/
inductive AtomType where
  | b32 | b64 | u32 | s32 | u64 | s64 | f32 | f32ftz | f64 | f16 | bf16 | f16x2 | bf16x2 | b128
  /-- `n` elements, each its own atomic (PTX's vector `atom`/`red`, §9.7.15.5). -/
  | vec (elem : AtomType) (n : Nat)
  deriving DecidableEq, Repr, Inhabited

/-- The 32-bit words of a value. -/
def AtomType.words : AtomType → Nat
  | .b64 | .u64 | .s64 | .f64 => 2
  | .b128 => 4
  | .vec t n => n * t.words
  | _ => 1

/-- Atomic and reduction operations (PTX ISA §9.7.13.5). -/
inductive AtomOp where
  | add | min | max | inc | dec | and | or | xor | exch | cas
  /-- Set the lowest run of `b` clear bits of a 32-bit word that starts at a multiple of `b` (a
  power of two), if there is one (SASS `UTCATOMSWS.FIND_AND_SET.ALIGN`, the tensor-memory
  column allocator: `NV.Async.Tcgen05.findSetAlign`). -/
  | findSetAlign
  /-- The same at any position: the lowest run of `b` clear bits (SASS `UTCATOMSWS.FIND_AND_SET`
  without `.ALIGN`: `NV.findSet`). -/
  | findSet
  deriving DecidableEq, Repr, Inhabited

/-- Read-modify-writes of an mbarrier object (PTX ISA 9.5 §9.7.15.16), an 8-byte shared word
(`NV.Async.MBarrierWord` gives its encoding). A wait (`test_wait`/`try_wait`) is an acquire load
of the word, decided in the issuer's semantics. -/
inductive MBarOp where
  /-- `mbarrier.init`: expected and pending arrivals `count`, tx-count 0, phase 0, or with `phase`
  1 a barrier whose previous phase is complete (a word stored directly). -/
  | init (count : Nat) (phase : Nat := 0)
  /-- expect-tx of `tx` bytes, then `count` arrive-on operations (`count = 0`: `expect_tx`
  alone); with `drop`, the expected count of later phases also falls by `count`
  (`arrive_drop`). -/
  | arrive (count : Nat) (tx : Nat := 0) (drop : Bool := false)
  /-- complete-tx of `bytes`, what an asynchronous copy performs once its data have landed. -/
  | completeTx (bytes : Nat)
  /-- `mbarrier.inval`. -/
  | inval
  deriving DecidableEq, Repr, Inhabited

inductive MemOp where
  | ld
  | st
  /-- Returns the old value. -/
  | atom (op : AtomOp) (ty : AtomType)
  /-- Returns nothing. -/
  | red (op : AtomOp) (ty : AtomType)
  /-- An mbarrier operation on the 8-byte object at the address; returns the old word (the state
  `mbarrier.arrive` returns). -/
  | mbar (op : MBarOp)
  deriving DecidableEq, Repr, Inhabited

def MemOp.returns : MemOp → Bool
  | .ld | .atom .. | .mbar _ => true
  | .st | .red .. => false

def MemOp.writes : MemOp → Bool
  | .ld => false
  | _ => true

/-- An atomic: it writes and returns the old value. -/
def MemOp.isAtom : MemOp → Bool
  | .atom .. => true
  | _ => false

/-- Cache operators and eviction hints (PTX ISA §9.7.9.1), as the instruction states them;
only the timing core reads them. -/
structure CacheHint where
  /-- `.ca`, `.cg`, `.cs`, `.lu`, `.cv`, `.wb`, `.wt`, or SASS's `CONSTANT`, `EF`, `EL`, … -/
  ops : Array String := #[]
  deriving DecidableEq, Repr, Inhabited

/-- Words per lane for an access of `bytes` bytes. -/
def words (bytes : Nat) : Nat := (bytes + 3) / 4

structure Req where
  space : Space
  op : MemOp
  /-- Bytes per lane: 1, 2, 4, 8 or 16. -/
  bytes : Nat
  sem : Sem := .weak
  scope : Scope := .thread
  proxy : Proxy := .generic
  hint : CacheHint := {}
  /-- Per lane: the address, or `none` for an inactive lane. -/
  addrs : Array (Option UInt64)
  /-- Store, atomic and `cas` operands, lane-major (`cas`: compare words then swap words). -/
  data : Array UInt32 := #[]
  /-- A generic access whose lanes name both the issuing CTA's shared memory and `space`: per
  lane, the shared address of a lane in shared memory (its `addrs` entry is then `none`). -/
  sharedPart : Array (Option UInt64) := #[]
  deriving Repr, Inhabited

/-- A request's answer: the loaded or old words, lane-major, `words bytes` per lane (inactive
lanes read as zero). -/
abbrev Resp := Array UInt32

end NV
