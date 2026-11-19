import NV.Machine.Access
import NV.Machine.Collective
import NV.Async.Op

/-!
# Effects

What one step of a thread (PTX) or a warp (SASS) asks of the machine. The semantics of an
instruction is a function from the executing state to an `Effect`; the machine answers it. Who
answers is what distinguishes the three uses of the model:

* proofs take every answer the memory model allows for a `mem` effect (the read values are the
  proof's to choose, constrained by consistency);
* the simulator answers from the flat memory at the cycle the timing core delivers the request,
  and writes the response back then;
* the interpreter answers from the flat memory in an order its scheduler picks.

Register-only work never leaves the semantics: it is a `next` or a `reg`.
-/

namespace NV

/-- Barrier and collective operations that synchronize with other threads or warps. -/
inductive SyncOp where
  /-- `bar.sync`/`barrier.sync` (`arrive = false`) or `bar.arrive` on named barrier `id`, with
  the expected thread count (`none`: every thread of the CTA). PTX ISA §9.7.13.1. -/
  | bar (id : Nat) (count : Option Nat) (arrive : Bool)
  /-- `barrier.cluster.arrive` / `.wait`. PTX ISA §9.7.13.2. -/
  | clusterArrive
  | clusterWait
  /-- `fence.acq_rel` (`sc = false`) or `fence.sc` at `scope`, and SASS's `MEMBAR`: the issuer
  waits until every memory request it issued earlier is performed. PTX ISA §9.7.13.4. -/
  | fence (sc : Bool) (scope : NV.PTX.Scope)
  /-- `fence.proxy.<proxy>` (§9.7.13.4): orders the issuer's accesses through the generic proxy
  with those through `p`. -/
  | proxyFence (p : NV.PTX.Proxy)
  /-- `fence.proxy.alias`: orders accesses through different virtual aliases of a location. -/
  | aliasFence
  /-- Close the issuer's open operations of group `g` into a group (`cp.async.commit_group`,
  `cp.async.bulk.commit_group`, `wgmma.commit_group`; SASS `LDGDEPBAR`, an `HGMMA` with `gsb0`). -/
  | asyncCommit (g : NV.Async.Group)
  /-- Wait until at most `n` of the issuer's committed groups of `g` are pending
  (`*.wait_group n`); with `read`, until their source reads are done
  (`cp.async.bulk.wait_group.read`). -/
  | asyncWait (g : NV.Async.Group) (n : Nat) (read : Bool := false)
  deriving DecidableEq, Repr, Inhabited

inductive Effect (σ : Type) where
  /-- No register result (branches, barriers, `NOP`): the new state. -/
  | next (s : σ)
  /-- A register result. `s` is the state right after issue (sources read, program counter
  advanced); `wb` writes the destinations, with values computed at issue, into whatever state the
  issuer has at writeback. The simulator applies it after the class's latency, so code whose
  control bits under-wait reads the old register, as on silicon (Huerta et al. 2025 §4); a
  blocking reading (PTX, proofs) is `wb s`. -/
  | reg (s : σ) (wb : σ → σ)
  /-- A memory request. `r` builds the request from the issuer's state when the instruction reads
  its sources, which for SASS is after issue (the read dependence counter guards the sources until then).
  `s` is the state right after issue; `wb` writes the response into whatever state the issuer has
  when it arrives, so a warp keeps issuing while a load is in flight. A blocking reading (PTX,
  proofs) is `wb resp s` with the response to `r s`. -/
  | mem (r : σ → Req) (s : σ) (wb : Resp → σ → σ)
  /-- A synchronizing operation; the continuation runs once the machine releases it. -/
  | sync (op : SyncOp) (s : σ)
  /-- An asynchronous operation (`NV.Async.Op`), handed to an async agent. Like `mem`'s request,
  `op` is built from the issuer's state when the instruction reads its sources; `s` is the state
  right after issue; `wb` writes the operation's register result into whatever state the issuer
  has when the operation completes (a warpgroup MMA's accumulators, read and written then, since
  consecutive MMAs chain through them), and is the identity for copies. A blocking reading (PTX
  interpreter, proofs) runs the agent's program (`NV.Async.Op.run`) on `op s`, then `wb`. -/
  | async (op : σ → NV.Async.Op) (s : σ) (wb : Resp → σ → σ)
  /-- A collective (`NV.Machine.Collective`): `contrib` is this thread's operand words and `mask`
  the instruction's member mask. Once every participant has arrived the machine computes the
  answers with `Collective.answer`, and `wb` writes this thread's answer into its state. -/
  | collect (op : Collective) (mask : UInt32) (contrib : Array UInt32) (s : σ)
      (wb : Resp → σ → σ)
  | exit
  /-- An instruction the semantics does not define, or an operand it cannot evaluate: the run
  stops with the reason. A defined instruction whose PTX behaviour is undefined (§ "undefined
  behaviour") also stops here, rather than picking a value. -/
  | fault (msg : String)
  deriving Inhabited

end NV
