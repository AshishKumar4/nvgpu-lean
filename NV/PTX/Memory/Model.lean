import NV.PTX.Memory.Execution

/-!
# PTX memory model: derived relations and axioms

Each definition names the `ptx.als` function or predicate it transcribes (line numbers are those
of NVlabs/mixedproxy `alloy/ptx.als` at commit used for `tools/litmus`) and the PTX ISA 9.4
section that states it in prose.
-/

namespace NV.PTX.Execution

open Rel

variable {n : Nat} {V : Type} [DecidableEq V] (x : Execution n V)

def thr (e : Fin n) : Nat := (x.ev e).thread
def placeOf (e : Fin n) : Place := x.place (x.thr e)

def reads : Fin n → Bool := fun e => (x.ev e).isRead
def writes : Fin n → Bool := fun e => (x.ev e).isWrite
def mems : Fin n → Bool := fun e => (x.ev e).isMem
def generic : Fin n → Bool := fun e => (x.ev e).isMem && (x.ev e).proxy == .generic

/-- `same_alias_r` (ptx.als:230): memory events on one virtual address. -/
def sameAlias : Rel n := Rel.ofFn fun a b => x.mems a && x.mems b && (x.ev a).addr == (x.ev b).addr

/-- `same_location_r` (ptx.als:228): memory events on one physical location (§8.2.2). -/
def sameLocation : Rel n := Rel.ofFn fun a b =>
  x.mems a && x.mems b && x.loc (x.ev a).addr == x.loc (x.ev b).addr

/-- Program order extended with asynchronous issue order (§8.9.1.1). -/
def order : Rel n := x.po ∪ x.issue

/-- `po_loc` (ptx.als:95). -/
def poLoc : Rel n := tc x.order ∩ x.sameAlias

/-- `fr` (ptx.als:98): reads before the coherence-later writes; a read of the initial value
precedes every write of its location. -/
def fr : Rel n :=
  (inv x.rf ⨾ tc x.co) ∪
    Rel.ofFn fun r w => x.reads r && x.writes w && x.sameLocation r w &&
      !(List.finRange n).any fun w' => x.rf w' r

/-- `com` (ptx.als:106). -/
def com : Rel n := x.rf ∪ tc x.co ∪ x.fr

/-- `strong_r` (ptx.als:292) and §8.7: each operation's scope contains the other's thread, and
both use one proxy. A weak operation has scope `thread`, so it is strong only within its thread. -/
def strongR : Rel n := Rel.ofFn fun a b =>
  let ea := x.ev a; let eb := x.ev b
  ea.scope.contains (x.thr a) (x.thr b) (x.placeOf a) (x.placeOf b) &&
  eb.scope.contains (x.thr b) (x.thr a) (x.placeOf b) (x.placeOf a) &&
  ea.proxy == eb.proxy

def strong (r : Rel n) : Rel n := r ∩ x.strongR

/-- `release_sequence` (ptx.als:149) and §8.8's release patterns. -/
def releaseSequence : Rel n :=
  x.strong (dom (fun e => (x.ev e).release && x.writes e) ⨾ opt x.poLoc ⨾ dom x.writes)
    ∪ x.strong (dom (fun e => (x.ev e).isReleaseFence) ⨾ tc x.order ⨾ dom x.writes)

/-- `observation` (ptx.als:157) and §8.9.2. -/
def observation : Rel n := x.strong x.rf ∪ x.rmw

/-- `acquire_sequence` (ptx.als:164) and §8.8's acquire patterns. The read of a reduction starts
no acquire pattern (§8.11.1), a rule `ptx.als` lacks. -/
def acquireSequence : Rel n :=
  let start := dom fun e => x.reads e && !(x.ev e).reduction
  x.strong (start ⨾ opt x.poLoc ⨾ dom (fun e => (x.ev e).acquire && x.reads e))
    ∪ x.strong (start ⨾ tc x.order ⨾ dom (fun e => (x.ev e).isAcquireFence))

/-- `sync` (ptx.als:145) and §8.9.4, plus the barrier rules of §8.9.4: `bar.sync`/`bar.arrive`
synchronize with the waits of the same barrier, `barrier.cluster.arrive` with
`barrier.cluster.wait`. -/
def sync : Rel n :=
  x.strong (x.releaseSequence ⨾ tc x.observation ⨾ x.acquireSequence) ∪ x.bar

/-- `cause_base` (ptx.als:190) and §8.9.5 "base causality order". -/
def causeBase : Rel n :=
  let po := tc x.order
  po ∪ (rtc x.order ⨾ tc ((x.sc ∪ x.sync) ⨾ rtc x.order))

/-- `same_block_r` (ptx.als:182): events of threads in one CTA. -/
def sameBlock : Rel n := Rel.ofFn fun a b => (x.placeOf a).cta == (x.placeOf b).cta

/-- `proxy_fence_ops` (ptx.als:174): the memory events of a proxy fence's proxy in its CTA. -/
def proxyFenceOps : Rel n := Rel.ofFn fun f e =>
  match (x.ev f).proxyFenced with
  | some p => x.mems e && (x.ev e).proxy == p && x.sameBlock f e
  | none => false

def sameProxy : Rel n := Rel.ofFn fun a b => (x.ev a).proxy == (x.ev b).proxy

/-- `proxy_preserved_cause_base` (ptx.als:194) and §8.9.5. -/
def proxyPreservedCauseBase : Rel n :=
  let cb := x.causeBase
  let gen := dom x.generic
  let srcF := cb ∩ inv x.proxyFenceOps
  let dstF := cb ∩ x.proxyFenceOps
  let alias := dom (fun e => (x.ev e).isAliasFence)
  let sameAddr (r : Rel n) : Rel n := r ∩ x.sameAlias
  let sameLoc (r : Rel n) : Rel n := r ∩ x.sameLocation
  sameAddr (gen ⨾ cb ⨾ gen)
    ∪ sameAddr (cb ∩ x.sameBlock ∩ x.sameProxy)
    ∪ sameAddr (srcF ⨾ cb ⨾ gen)
    ∪ sameAddr (gen ⨾ cb ⨾ dstF)
    ∪ sameAddr (srcF ⨾ cb ⨾ dstF)
    ∪ sameLoc (gen ⨾ cb ⨾ alias ⨾ cb ⨾ gen)
    ∪ sameLoc (srcF ⨾ cb ⨾ alias ⨾ cb ⨾ gen)
    ∪ sameLoc (gen ⨾ cb ⨾ alias ⨾ cb ⨾ dstF)
    ∪ sameLoc (srcF ⨾ cb ⨾ alias ⨾ cb ⨾ dstF)

/-- `cause` (ptx.als:141) and §8.9.5 "causality order". -/
def cause : Rel n := opt x.observation ⨾ x.proxyPreservedCauseBase

/-! ## Axioms (ptx.als:116–134, PTX ISA §8.10) -/

/-- §8.10.6. -/
def causality : Bool := irreflexive (opt x.com ⨾ x.cause)

/-- §8.10.3, read-modify-write atomicity. -/
def atomicity : Bool := isEmpty ((x.strong x.fr ⨾ x.strong x.co) ∩ x.rmw)

/-- §8.10.1. -/
def coherence : Bool :=
  subset ((dom x.writes ⨾ x.cause ⨾ dom x.writes) ∩ x.sameLocation) (tc x.co)

/-- §8.10.2: morally strong `fence.sc` operations ordered by causality are ordered alike in
Fence-SC order. Causality order between fences is base causality order: the proxy-preserved
refinement of §8.9.5 is defined for memory operations only. (`ptx.als` uses its `cause`, which
relates memory operations only, so its `seq_cst` never constrains anything.) -/
def seqCst : Bool :=
  let f := dom fun e => (x.ev e).isFenceSC
  subset ((f ⨾ x.causeBase ⨾ f) ∩ x.strongR) x.sc

/-- §8.10.4. -/
def noThinAir : Bool := acyclic (x.rf ∪ x.dep)

/-! ## Well-formed candidate executions

What a candidate execution must satisfy before the axioms apply: the facts `ptx.als` lists as
"basic model constraints", stated here in ISA terms. -/

def strictOrder (r : Rel n) : Bool := irreflexive r && subset (r ⨾ r) r

def sameThread : Rel n := Rel.ofFn fun a b => x.thr a == x.thr b

/-- §8.9.1: program order totally and strictly orders each thread's operations and relates no
operations of different threads; asynchronous issue order does not create cycles with it. -/
def poWF : Bool :=
  strictOrder x.po && subset x.po x.sameThread &&
  (List.finRange n).all (fun a => (List.finRange n).all fun b =>
    a == b || !x.sameThread a b || x.po a b || x.po b a) &&
  acyclic x.order

/-- Each read reads at most one write, of its location, and returns that write's value; a read of
no write returns the initial value (§8.2.6). -/
def rfWF : Bool :=
  subset x.rf ((dom x.writes ⨾ x.sameLocation) ⨾ dom x.reads) &&
  (List.finRange n).all fun r =>
    let ws := (List.finRange n).filter fun w => x.rf w r
    !x.reads r || match ws with
      | [] => x.val r == x.init (x.loc (x.ev r).addr)
      | [w] => x.val r == x.val w
      | _ => false

/-- §8.9.6: coherence order is a strict partial order on writes to one location that relates
every two morally strong writes to one address (`ptx.als`'s `total[co, ...]`). -/
def coWF : Bool :=
  strictOrder x.co && subset x.co ((dom x.writes ⨾ x.sameLocation) ⨾ dom x.writes) &&
  (List.finRange n).all fun a => (List.finRange n).all fun b =>
    a == b || !(x.writes a && x.writes b && (x.ev a).addr == (x.ev b).addr && x.strongR a b) ||
      x.co a b || x.co b a

/-- §8.9.3: Fence-SC order is a strict partial order on `fence.sc` relating every morally strong
pair. -/
def scWF : Bool :=
  let f := fun e => (x.ev e).isFenceSC
  strictOrder x.sc && subset x.sc (prod f f) &&
  (List.finRange n).all fun a => (List.finRange n).all fun b =>
    a == b || !(f a && f b && x.strongR a b) || x.sc a b || x.sc b a

/-- A read-modify-write is a read and a write of one address by one thread at one scope, the write
depending on the read (`ptx.als`: `rmw in same_alias[po & dep & scope.~scope]`); dependencies
follow program order. -/
def rmwWF : Bool :=
  subset x.rmw ((dom x.reads ⨾ (x.po ∩ x.dep ∩ x.sameAlias)) ⨾ dom x.writes) &&
  (List.finRange n).all (fun a => (List.finRange n).all fun b =>
    !x.rmw a b || (x.ev a).scope == (x.ev b).scope) &&
  subset x.dep x.po

/-- Barrier synchronization relates an arrival to a wait of the same barrier in the same CTA
(`bar`) or cluster (`barrier.cluster`). -/
def barWF : Bool :=
  (List.finRange n).all fun a => (List.finRange n).all fun b =>
    !x.bar a b || match (x.ev a).kind, (x.ev b).kind with
      | .barArrive i, .barWait j => i == j && (x.placeOf a).cta == (x.placeOf b).cta
      | .clusterArrive, .clusterWait => (x.placeOf a).cluster == (x.placeOf b).cluster
      | _, _ => false

def wellFormed : Bool := x.poWF && x.rfWF && x.coWF && x.scWF && x.rmwWF && x.barWF

def consistent : Bool :=
  x.wellFormed && x.noThinAir && x.atomicity && x.coherence && x.causality && x.seqCst

end NV.PTX.Execution
