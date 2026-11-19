import NV.Machine.Access

/-!
# Asynchronous operations

What a thread (PTX) or warp (SASS) hands to an asynchronous agent with `Effect.async`: bulk and
tensor copies (TMA), `cp.async` (`LDGSTS`), warpgroup MMAs (`wgmma`, `HGMMA`) and tensor-memory
MMAs (`tcgen05.mma`, `UTCHMMA`). The issuer
continues at once; the agent performs the operation's memory accesses later, as the machine
decides.

The memory model treats an agent like a thread of the issuer's CTA (PTX ISA 9.5 §8.9.1.1). A
copy's program is its moves in any order (each a load from the source and a store of the loaded
data at the destination, both through the async proxy), then the implicit generic-async proxy
fence that completion implies (§9.7.10.28.2), then its signals: the complete-tx or arrive-on
read-modify-writes on mbarriers, in the generic proxy with release semantics (§9.7.10.28.5.3).
`Copy.program` lists exactly that sequence. A warpgroup MMA's program is its shared-memory loads
(async proxy), after which it computes the issuer's register result from the loaded data.
-/

namespace NV.Async

/-- The asynchronous groups a thread commits and waits on. -/
inductive Group where
  /-- `cp.async.commit_group`/`wait_group` (§9.7.10.28.3.2–3); SASS `LDGDEPBAR`/`DEPBAR`. -/
  | cpAsync
  /-- The bulk async-group of `cp.async.bulk*` (§9.7.10.28.6). -/
  | bulk
  /-- `wgmma.commit_group`/`wait_group` (§9.7.17.5.4–5); SASS `gsb0`, `WARPGROUP.DEPBAR`. -/
  | wgmma
  /-- The thread's asynchronous `tcgen05` operations (`mma`, `cp`, `shift`), which a
  `tcgen05.commit` tracks (§9.7.18.6.2.1.1); SASS `UTCBAR`. -/
  | tcgen05
  deriving DecidableEq, Repr, Inhabited, Hashable

/-- One step of a copy: load `src`, then store the loaded words lane by lane with each request of
`dsts` (several for a multicast). A destination lane whose source lane is inactive is out of the
source tensor's bounds and stores `fill` in every word. `dsts` carry no data of their own. -/
structure Move where
  src : Req
  dsts : Array Req
  fill : UInt32 := 0
  /-- When nonempty, the move loads nothing: each destination stores these words (lane-major, as
  a response is), the issuer's register values (`st.async`, `red.async`). -/
  imm : Array UInt32 := #[]
  deriving Repr, Inhabited

/-- The words a destination request of `m` stores, given the load's response (lane-major,
`words bytes` per lane for every lane, as `Resp` is). -/
def Move.data (m : Move) (dst : Req) (resp : Resp) : Array UInt32 := Id.run do
  let n := words dst.bytes
  let mut out := Array.replicate (dst.addrs.size * n) 0
  for l in [0:dst.addrs.size] do
    if dst.addrs[l]!.isNone then continue
    let loaded := !m.imm.isEmpty || (m.src.addrs[l]?.join).isSome
    for j in [0:n] do
      out := out.set! (l * n + j) (if loaded then resp[l * n + j]?.getD 0 else m.fill)
  return out

/-- The store request a destination of `m` issues once the load answered `resp`. -/
def Move.store (m : Move) (dst : Req) (resp : Resp) : Req := { dst with data := m.data dst resp }

/-- A copy: global to shared (TMA and bulk loads, `cp.async`), shared to global (TMA and bulk
stores) or shared to shared in the cluster. -/
structure Copy where
  moves : Array Move
  /-- Read-modify-writes performed after every move and the implicit proxy fence: complete-tx
  of the bytes delivered, on the mbarrier of each destination CTA, or the arrive-on of
  `cp.async.mbarrier.arrive`. -/
  signals : Array Req := #[]
  /-- The issuer's group the copy belongs to, if any. -/
  group : Option Group := none
  /-- The copy starts only after the issuer's earlier operations of `group` completed
  (`cp.async.mbarrier.arrive`: its arrive-on tracks every earlier `cp.async`). -/
  afterGroup : Bool := false
  deriving Repr, Inhabited

/-- A step of an agent's program, in program order. -/
inductive Step where
  | move (m : Move)
  /-- The implicit generic-async proxy fence of completion. -/
  | proxyFence
  | signal (r : Req)
  deriving Repr, Inhabited

/-- A copy as the memory model's thread: its moves (unordered among themselves), the implicit
proxy fence, then its signals in order. -/
def Copy.program (c : Copy) : List Step :=
  c.moves.toList.map .move ++ (if c.signals.isEmpty then [] else .proxyFence :: c.signals.toList.map .signal)

/-- Bytes a move stores at one destination. -/
def Move.bytes (m : Move) : Nat :=
  match m.dsts[0]? with
  | some d => d.bytes * (d.addrs.filter (·.isSome)).size
  | none => 0

/-- Bytes a copy stores at one destination: the complete-tx count of a TMA or bulk load. -/
def Copy.bytes (c : Copy) : Nat := c.moves.foldl (· + ·.bytes) 0

/-- A warpgroup MMA (one warp's share in SASS, one thread's in PTX): it loads `loads` from shared
memory through the async proxy, and its result is `result data c`: the issuer's new accumulator
words from the loads' data (their responses concatenated in order) and the accumulator `c` it
starts from. That is `c0`, the issuer's accumulator when it issued, unless an earlier MMA of the
issuer had not completed then: consecutive MMAs chain through the accumulator (PTX ISA 9.5
§9.7.17.5.2), so the machine passes the earlier one's result instead. Both are lane-major. -/
structure Mma where
  loads : Array Req
  result : Resp → Array UInt32 → Array UInt32
  /-- The same result on the machine-word tile datapath where it covers the operation
  (`TensorModel.tileWords`), which the timing core computes instead of `result`: not part of the
  semantics, so the theorems speak of `result`; `async-tests` checks the two agree, and a recorded
  trace's MMA results are replayed against `result` (`NV.Formal.Trace.check`). -/
  fast : Resp → Array UInt32 → Option (Array UInt32) := fun _ _ => none
  c0 : Array UInt32
  /-- Multiply-adds, for the timing core: `M·N·K` of the issuer's share. -/
  macs : Nat
  /-- The instruction's `N`, which sets its latency. -/
  n : Nat
  /-- Bits of an input element (16 for f16/bf16, 32 for tf32, 8 for fp8 and integers). -/
  inBits : Nat
  /-- Whether A comes from registers. -/
  aRegs : Bool
  /-- Whether the instruction also closes the issuer's open wgmma operations into a group (SASS
  `HGMMA … gsb0`). -/
  commit : Bool := false
  /-- The first register it writes, and the first it accumulates from (`none`: it does not read
  an accumulator). An MMA chains onto the latest earlier one of the issuer that writes the
  registers it accumulates from; the defaults make every MMA of an issuer one chain (PTX). -/
  dst : Nat := 0
  src : Option Nat := some 0
  /-- SASS: the issuing warp's index in its warpgroup; its siblings issue the same MMA's other
  shares from the slots beside it (a CTA's warps take consecutive slots). -/
  warpInGroup : Option Nat := none
  /-- SASS: the loads from this index on read B alone, which the tensor core reads once for the
  warpgroup's four shares (`AsyncUnit`'s timing charges each share a quarter of their port time). -/
  bFrom : Option Nat := none
  deriving Inhabited

instance : Repr Mma where
  reprPrec m _ := s!"Mma(n := {m.n}, macs := {m.macs}, loads := {m.loads.size})"

/-- A 5th-generation tensor-core MMA (`tcgen05.mma`, PTX ISA 9.5 §9.7.18.10): it loads A and B
(shared memory through the async proxy, or A from tensor memory) and the accumulator from tensor
memory, and stores `D = result data` back into tensor memory with `stores`. Its issuer's
registers are untouched. The operations of one issuer run in issue order, each after the
previous one's stores (`tcgen05` pipelines, §9.7.18.6.2). -/
structure MmaT where
  loads : Array Req
  result : Resp → Array UInt32
  stores : Array UInt32 → Array Req
  /-- Multiply-adds, for the timing core: `M·N·K`. -/
  macs : Nat
  n : Nat
  /-- Bits of an input element. -/
  inBits : Nat
  /-- When it accumulates onto D: D's words in its load response (first word, count), in the order
  of its result (row-major). -/
  accIn : Option (Nat × Nat) := none
  /-- D's `tcgen05` address: MMAs on the same D accumulate in issue order. -/
  dKey : Nat := 0
  deriving Inhabited

instance : Repr MmaT where
  reprPrec m _ := s!"MmaT(n := {m.n}, macs := {m.macs}, loads := {m.loads.size})"

/-- A tensor copy whose tensor map lies in memory the issuer cannot read at issue (global): the
agent first loads the 128-byte map through `desc`, then performs the copy `build` makes of it. -/
structure TensorCopy where
  desc : Req
  build : Resp → Except String Copy
  group : Option Group := none
  deriving Inhabited

instance : Repr TensorCopy where
  reprPrec t _ := s!"TensorCopy(desc := {repr t.desc.addrs})"

inductive Op where
  | copy (c : Copy)
  | tensor (t : TensorCopy)
  | mma (m : Mma)
  | mmaT (m : MmaT)
  /-- `cp.async.bulk.prefetch(.tensor)`: bring global lines into L2; no architectural effect. -/
  | prefetch (lines : Array UInt64)
  /-- An operation whose behaviour the ISA leaves undefined or the model does not define, found
  when the operation is built from the issuer's sources: the run stops with the reason. -/
  | fault (msg : String)
  deriving Repr, Inhabited

/-- Whether the operation writes the issuer's registers when it completes (an MMA's accumulators);
copies and prefetches write only memory. -/
def Op.writesRegisters : Op → Bool
  | .mma _ => true
  | _ => false

/-- The group an operation joins. -/
def Op.group : Op → Option Group
  | .copy c => c.group
  | .tensor t => t.group
  | .mma _ => some .wgmma
  | .mmaT _ => some .tcgen05
  | .prefetch _ | .fault _ => none

/-- Run an operation to completion with `perform` answering each request at once, in the
agent's program order: the blocking reading of `Effect.async` (the PTX interpreter, proofs).
The result is what the issuer's `wb` takes: an MMA's words, nothing for a copy. -/
def Op.run {m : Type → Type} [Monad m] [MonadExcept String m] (perform : Req → m Resp) :
    Op → m Resp
  | .copy c => runCopy c
  | .tensor t => do
    match t.build (← perform t.desc) with
    | .ok c => runCopy c
    | .error e => throw e
  | .mma mm => do
    let data := (← mm.loads.mapM perform).foldl (· ++ ·) #[]
    pure (mm.result data mm.c0)
  | .mmaT mm => do
    let data := (← mm.loads.mapM perform).foldl (· ++ ·) #[]
    for r in mm.stores (mm.result data) do discard <| perform r
    pure #[]
  | .prefetch _ => pure #[]
  | .fault msg => throw msg
where
  runCopy (c : Copy) : m Resp := do
    for mv in c.moves do
      -- A move of the issuer's words (`imm`) loads nothing.
      let resp ← if mv.imm.isEmpty then perform mv.src else pure mv.imm
      for d in mv.dsts do discard <| perform (mv.store d resp)
    for r in c.signals do discard <| perform r
    pure #[]

end NV.Async
