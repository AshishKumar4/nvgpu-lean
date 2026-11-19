import NV.Rel

/-!
# PTX memory model: events and candidate executions

The axiomatic model of the PTX ISA 9.4, chapter 8 ("Memory Consistency Model"), transcribed from
Lustig, Cooksey and Giroux's Alloy model of the mixed-proxy PTX memory model
(github.com/NVlabs/mixedproxy, `alloy/ptx.als`, itself extending Lustig, Sahasrabuddhe and Giroux,
ASPLOS 2019, github.com/NVlabs/ptxmemorymodel), with the Hopper additions the ISA now specifies:
the `.cluster` scope (§8.5), the async proxy (§8.6) and barrier synchronization (§8.9.4).

Where the Alloy artifact and the ISA text disagree, this file follows the ISA and says so at the
definition (see `Event.proxy` and `Docs/memory-model.md`).
-/

namespace NV.PTX

/-- Scopes, PTX ISA §8.5. `thread` is the scope of a weak operation: Alloy's `Thread` scope,
which makes a weak operation morally strong only with operations of its own thread. -/
inductive Scope where
  | thread | cta | cluster | gpu | sys
  deriving DecidableEq, Repr, Inhabited

/-- Proxies, PTX ISA §8.6. Texture, surface and constant are the abstract proxies of the
mixed-proxy paper; `async` is the proxy of `cp.async.bulk`, TMA and `wgmma` accesses, and
`tensormap` that of a TMA unit's reads of a tensor map (§8.6, `fence.proxy.tensormap`). -/
inductive Proxy where
  | generic | async | texture | surface | constant | tensormap
  deriving DecidableEq, Repr, Inhabited

/-- Where a thread runs. Identifiers are global: two threads share a CTA iff their `cta` fields
are equal, likewise for clusters and devices. -/
structure Place where
  cta : Nat
  cluster : Nat
  gpu : Nat
  deriving DecidableEq, Repr, Inhabited

/-- Whether threads `t` and `u` (at places `p`, `q`) lie in one instance of scope `s`. -/
def Scope.contains : Scope → (t u : Nat) → (p q : Place) → Bool
  | .thread, t, u, _, _ => t == u
  | .cta, _, _, p, q => p.cta == q.cta
  | .cluster, _, _, p, q => p.cluster == q.cluster
  | .gpu, _, _, p, q => p.gpu == q.gpu
  | .sys, _, _, _, _ => true

/-- Event kinds, following the signatures of `ptx.als`. A read-modify-write is a `read` and a
`write` joined by the execution's `rmw` relation. `barArrive`/`barWait` split a CTA barrier
(`bar.sync` is an arrive followed in program order by a wait) and `clusterArrive`/`clusterWait`
are `barrier.cluster.arrive`/`.wait`; splitting keeps base causality acyclic. -/
inductive Kind where
  | read | write
  | fenceAcqRel | fenceSC
  | proxyFence (p : Proxy)
  | aliasFence
  | barArrive (id : Nat) | barWait (id : Nat)
  | clusterArrive | clusterWait
  /-- `*.wait_group`: the issuer's events after it follow the waited operations' events in
  causality order through the issue relation, and it orders nothing else. -/
  | asyncWait
  deriving DecidableEq, Repr, Inhabited

structure Event where
  /-- Issuing thread. An asynchronous operation (§8.9.1.1) runs on its own agent thread,
  placed in the issuer's CTA. -/
  thread : Nat
  kind : Kind
  scope : Scope := .thread
  /-- `ld.acquire`, `atom.acquire`, `mbarrier.try_wait` returning true (default `.acquire`). -/
  acquire : Bool := false
  /-- `st.release`, `atom.release`, `mbarrier.arrive` (default `.release`). -/
  release : Bool := false
  /-- The read of a `red`: it starts no acquire pattern (§8.11.1). -/
  reduction : Bool := false
  /-- Virtual address; aliases (§8.2.2) are distinct addresses of one physical location. -/
  addr : Nat := 0
  /-- Fences carry the generic proxy. In `ptx.als` fences have no proxy at all, so `strong`
  never relates a fence to anything and fence release/acquire patterns never synchronize; the
  ISA's message-passing example (§8.10.6) requires them to. -/
  proxy : Proxy := .generic
  deriving DecidableEq, Repr, Inhabited

namespace Event

def isRead (e : Event) : Bool := e.kind == .read
def isWrite (e : Event) : Bool := e.kind == .write
def isMem (e : Event) : Bool := e.isRead || e.isWrite
def isReleaseFence (e : Event) : Bool := e.kind == .fenceAcqRel || e.kind == .fenceSC
def isAcquireFence (e : Event) : Bool := e.isReleaseFence
def isFenceSC (e : Event) : Bool := e.kind == .fenceSC
def isAliasFence (e : Event) : Bool := e.kind == .aliasFence
def proxyFenced (e : Event) : Option Proxy :=
  match e.kind with
  | .proxyFence p => some p
  | _ => none

end Event

/-- A candidate execution of `n` events whose memory events carry values of type `V`: the
program's events and placement, plus the existentially chosen relations (`rf`, `co`, `sc`) and the
program-given ones (`po`, `rmw`, `dep`, `issue`, `bar`). Litmus tests use integers; programs'
executions (`NV.Formal.Outcomes`) use the bytes an access reads or writes. -/
structure Execution (n : Nat) (V : Type := Int) where
  ev : Fin n → Event
  place : Nat → Place
  /-- Physical location of each virtual address. -/
  loc : Nat → Nat := id
  /-- Initial value of each physical location. -/
  init : Nat → V
  /-- Value read or written by each memory event. -/
  val : Fin n → V
  po : Rel n
  /-- Ordering of an asynchronous operation's events after the issuing thread's prior
  instructions (§8.9.1.1); it plays the role of program order for the async agent. -/
  issue : Rel n := Rel.empty
  rf : Rel n
  co : Rel n
  rmw : Rel n := Rel.empty
  dep : Rel n := Rel.empty
  sc : Rel n := Rel.empty
  /-- Arrivals related to the waits of the same barrier instance (§8.9.4). -/
  bar : Rel n := Rel.empty

end NV.PTX
