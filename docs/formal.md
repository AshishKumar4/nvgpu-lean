# The formal side: programs, executions, the request machine

What is proved about the memory side of a program, where, and under which limits. The memory
model itself (PTX ISA chapter 8) is described in `memory-model.md`.

## From programs to outcomes

| result | statement | file |
|---|---|---|
| `Outcomes P o` | the outcomes of a program: each agent runs its step function with any bytes for its reads (`Run`); each asynchronous operation runs on its own agent (`AgentRun`: a copy's loads and stores, the implicit proxy fence and its signals; an MMA's shared-memory loads), on a thread of its own after the issuer's earlier events (issue order, §8.9.1.1), a `wait_group` ordering the waited operations before the issuer's later events; the events of all runs, in any arrangement, with chosen reads-from, coherence and Fence-SC orders and barrier instances that count right, form a candidate execution; a consistent candidate yields its final memory, a fault, or `unsupported` | `NV/Formal/Outcomes.lean` |
| `RunNB.blocking`, `footprinted` | a SASS warp's non-blocking run (writebacks pending until they land, in any order, as the engine applies them), in which every instruction issues clear of the pending write sets (the engine's hazard check), performs the acts of a blocking `Run` of the same warp, ending the same way up to a fault's message. Of every instruction, theorems of the SASS semantics: its outcome depends only on its footprint's reads (`Op.readsWithin`: a decoder sees the warp only through `View.of fp.reads` and the control state), its writebacks are local to its write set (`Op.wbLocal`), and it is confined to its footprint (`Op.confined`); that a footprint is complete (lists everything the semantics reads and writes) is checked on the test kernels (`Tests/SassFootprint`). Requests are read at issue (the engine reads them later, the same values unless an instruction in between writes their sources), and clock reads are outside it | `NV/Formal/NonBlocking.lean` |
| `Trace.check`, `Trace.check_run` | the engine's run of a warp refines the semantics: its trace (`NV.TraceEv`: each issue with what its effect took from outside the warp, each writeback landing, by the engine's pending ids) replayed against the warp's step, each issue clear of the pending write sets and each async operation's agent rebuilt from its reads (`agentOf`, a run of the operation) and ending with the engine's result, is a non-blocking run (`check_sound`), so a blocking `Run` of `Outcomes` with the same acts. Each landing must leave what the engine's warp holds (`agrees`), so a result the engine computed otherwise fails. The engine records the traces (`Launch.recordTrace`); `lake exe trace-check` replays every warp of the example kernels (CI row `trace-check`). An asynchronous operation's agent is recorded with what its reads returned; an MMA chained onto one whose result has not landed lands that result first in the trace, as the non-blocking reading needs (`Clear`) | `NV/Formal/TraceCheck.lean`, `NV/Machine/Trace.lean`, `Tools/TraceCheck.lean` |
| `Trace.kernel_run` | the bridge instantiated for a kernel: for `decode k` (the step `lake exe trace-check` replays with), every warp whose engine trace replays ran a `Run` of `Outcomes` with the trace's acts; no assumption is left (`footprinted`) | `NV/Formal/TraceCheck.lean` |
| `Req.laneEvents`, `SyncOp.events` | how a request lane and a synchronizing operation become memory-model events: weak `ld`/`st` in 4-byte granules, strong ones per element, atomics as a read and a write joined by `rmw` | `NV/Memory/Events.lean` |
| `drf_sc` | a consistent execution whose conflicting accesses are all ordered by proxy-preserved base causality order is sequentially consistent (`IsSC`: program and communication order acyclic, coherence total per location); `IsSC.reads_latest` gives the interleaving | `NV/PTX/Memory/DRF.lean` |
| `drf_sc_outcomes` | the program form: if every consistent candidate is race-free, every outcome is an outcome of an SC candidate | `NV/Formal/DRF.lean` |
| `ppcb_async_completion`, `ppcb_async_issue` | an async-proxy access and a generic one are ordered through a `fence.proxy.async` of the CTA (the implicit fence ending `cp.async.bulk` and `wgmma`, then a release/acquire chain such as an mbarrier's) | `NV/PTX/Memory/DRF.lean` |
| `rmw_atomic_of_adjacent` | in a race-free execution no write comes between an RMW's read and write | `NV/PTX/Memory/DRF.lean` |

## The request machine and the simulator

| result | statement | file |
|---|---|---|
| `Flat` | the operational memory: one flat copy of global and per-CTA shared memory, requests in flight per issuer, per-lane delivery that refuses what the ISA's ordering rules forbid, fences and barrier arrivals on drained issuers, counted barrier instances | `NV/Memory/Flat.lean` |
| `Timed.consistent` | an execution whose events have a total perform order obeying the machine's local rules satisfies every axiom of §8.10 | `NV/PTX/Memory/Timed.lean` |
| `Performed.consistent` | an event list in perform order, with reads-from the last earlier write, is consistent | `NV/Memory/Performed.lean` |
| `Flat.sound` (`Mem.sound`) | every sequence of steps the machine accepts performs events whose execution is consistent, for any thread placement | `NV/Memory/Sound.lean` |
| `Sim.done_outcome` | the reference machine (agents on `Flat`, blocking on their requests, any interleaving and delivery order): every run in which all agents exit leaves a memory that is an `Outcome`, with every read a dependency of its thread's later events | `NV/Formal/Sim.lean` |
| `Sim.fault_outcome` | a fault reached with no agent waiting is an outcome | `NV/Formal/Sim.lean` |

## Examples restated on the general semantics

| result | statement | file |
|---|---|---|
| `Ring.race_free`, `Ring.reads_stage` | the TMA ring in an execution: given the orders the full and empty barriers give (completion fence before each consumer read; each consumer read before the producer's reuse fence), every two conflicting slot accesses are ordered by proxy-preserved causality, and every consumer read of stage `q` reads stage `q`'s copy. The reuse needs a `fence.proxy.async` before the next copy: without it the generic read and the next async write are a model-level race (litmus `WAR_proxy`: allowed by mixedproxy's Alloy model and ours without the proxy fence, forbidden with it). The theorem concerns the memory model's orders; hardware reordering is outside it | `Examples/RingRace.lean` |
| `merge_correct` | the split-KV merge as a program (`red.relaxed.gpu.add.u32` into a workspace, `atom.acq_rel.gpu.add.u32` on a counter, the last split loads and stores): for any number of splits and any partials, every outcome is a completed run whose output word is the sum of the partials mod 2³² | `Examples/SplitMergeProgram.lean` |
| `copy_correct` | a TMA-style copy completed through an mbarrier (`mbarrier.init`, `arrive.expect_tx`, `cp.async.bulk` whose agent loads `X`, stores `Y` through the async proxy and performs complete-tx, a `try_wait.parity` loop, then a generic load of `Y` stored to `O`): for every layout and every `X`, every outcome is a completed run whose `O` holds `X`'s initial bytes; no outcome faults or leaves the model (the arrive and the complete-tx read barrier words they are defined on) | `Examples/AsyncCopy.lean` |
| `ring_correct` | the TMA ring as a program: a producer and a consumer thread pass `N` words through `S` shared slots, each slot guarded by a full mbarrier (arrive.expect_tx; the bulk copy's complete-tx) and an empty one (the consumer's arrive; the producer's wait and `fence.proxy.async` before reuse). For every layout and every `X`, every completed outcome holds word `q` of `X` at word `q` of `O`. The barriers' orders come from the reusable theory `NV.Formal.Pipeline` (`Full.raw`, `Empty.war`: a wait that returned true synchronizes with every write of its round through the release sequence, for any number of rounds and completes), the ring's from `Examples.Ring.Race`; fault outcomes are not covered | `Examples/RingProgram.lean`, `NV/Formal/Pipeline.lean`, `NV/PTX/Memory/Chain.lean` |
| `swap_correct` | two threads of a warp each load a word of `X`, exchange it with `shfl.sync.bfly` and store what they got: every completed outcome holds word `1 - t` of `X` at word `t` of `O`. Both threads join one instance because the member mask names them, and each answer is the other's word (`answer_swap`) | `Examples/ShflSwap.lean` |
| `WgmmaShare.plan_result` | a warp's share of a warpgroup MMA, A in shared memory, 16-bit inputs: when its loads returned the chunks of a shared memory `M`, its result is `Share.finish` of the matrices `M` holds at the descriptors' addresses (A's sixteen rows, B's columns), negated as the instruction asks; `operands_smem` is the step from addresses to operands. `plan_tile`: with both operands bf16, K-major, under the 128-byte swizzle and laid out as TMA writes a box of 128-byte rows, and shared memory holding each chunk as global memory does, the share multiplies the global tiles' elements. That shared memory holds them once the mbarrier's wait returns is the ring's and copy's business (`ring_correct`, `copy_correct`); the composition into one GEMM-tile theorem is open. `pack_get`: word `v` of lane `l` holds the product's element at `accIndex l v` (row `l / 4 + 8 ((v / 2) mod 2)`, column `2 (l mod 4) + v mod 2 + 8 (v / 4)`); `accIn_pack`: the next MMA reads that block back as 32-bit words; `finish_chain`: a chain of MMAs on one warp's 32-bit accumulator leaves the last product block, each step taking the previous one. `kloop`: a K loop of bf16 wgmmas on TMA boxes (each step `TileStep.Valid`: `plan_tile`'s conditions on its own stage) leaves the last product block of the global tiles, each step's product (`Share.product`, the tensor-core model's tile) taking the previous one as 32-bit words | `NV/Formal/WgmmaShare.lean` |

## Limits

- Accesses to one location must have one size and alignment (`Uniform`); the model is not defined
  for mixed-size races (§8.7.2).
- No-Thin-Air sees a candidate's dependencies (`Candidate.dep`), which contain the runs' semantic
  ones (`Program.SemDep`): an event depends on an earlier read of its thread when some run of the
  agent that makes every other earlier choice the same but changes that read does not perform the
  event. This covers address, data and control dependencies through the step function alike.
  Dependencies through asynchronous agents and across a warp agent's lanes are left out. Every
  semantic dependency lies in `readPo` (a read before a later event of its thread), and the
  reference machine performs each thread's events in program order (`Sim.reach_ordered`), so its
  outcomes keep `readPo` as their dependencies.
- Warp collectives and `bar.red` are in `Outcomes` (`Program.CollectsOk`, a candidate's choice of
  instances like its choice of reads): an instance's members share a warp (a CTA for `bar.red`),
  a warp collective is joined by every thread of its member mask, each answer is
  `Collective.answer` on the members' contributions, and an answer the ISA leaves undefined is a
  fault. Delimited: `activemask` may find any subset of the warp together; collectives order no
  memory accesses (`bar.warp.sync`'s and `bar.red`'s ordering are left out, so outcomes are a
  superset there); a thread's collectives may join instances in any order; a collective that a
  member never reaches (it exited) gives no outcome, where the ISA leaves it undefined; a warp
  agent's collective leaves the model.
- Local memory, `shared::cluster` through `Flat.issue`'s target CTA aside, and `.const`/`.param`
  are not in the request machine.
- The reference machine's agents block on their requests (the PTX reading of `Effect.mem`); the
  timed engine's non-blocking warps are outside `Sim.done_outcome`.
- The reference machine has no asynchronous operations yet (its theorems cover runs without them,
  `SimInv.plain`); `Outcomes` has them.
- An asynchronous copy's moves are performed in order on its agent's thread; a copy whose moves
  overlap is ordered here as the ISA may not order it.
- A warpgroup MMA's result is the blocking reading's: the product of what its loads returned and
  the accumulator at issue. The register-level rules (`wgmma.fence` before the accumulators are
  touched, `wait_group` before they are read) are not modelled, so a theorem about a program that
  breaks them says nothing about silicon; it is an assumption of every such theorem.

## Beside SF Tensor's PTX formalisation

SF Tensor's "Formalizing PTX in Lean" (Nov 2026; its §3) takes the same route: per-thread
semantics against an oracle for every externally determined value (our `Run`'s reads returning
any bytes), the Lustig et al. model with mixed proxies, and theorems over every consistent,
complete execution. Where we differ:

- **Axioms.** They drop No-Thin-Air and Fence-SC. We keep both, and all of §8.10's: Fence-SC as
  stated, No-Thin-Air over semantic dependencies through each thread's step function
  (not through asynchronous agents or across a warp agent's lanes).
- **Final memory.** Theirs is what observer reads, ordered after every kernel event, return; ours
  is each byte's coherence-last write (`Candidate.IsFinal`), or its initial byte. Coherence is a
  total order per location over writes and every write precedes such an observer, so the two
  agree.
- **Mixed sizes.** Both relate equal-sized accesses only. They withhold guarantees under
  mixed-size races; we make any partially overlapping pair `unsupported` (`Uniform`), raced or
  not, and size weak accesses in 4-byte granules so word, vector and byte-aligned word accesses
  stay comparable.
- **Values.** Their floats are reals; ours are the bits `NV.Num` computes, with real-arithmetic
  specifications an option for a theorem to state.
- **Non-vacuity.** They prove, per kernel, that some consistent complete execution exists. Ours
  comes from the reference machine: `Sim.done_outcome` turns any finished run into an outcome; a
  per-example existence theorem exists for the merge (`merge_nonvacuous`) and is open for the others.
- **Beyond them.** The operational request machine with `Flat.sound`, the reference machine's
  `done_outcome` (simulator outputs are outcomes), `drf_sc`, and SASS as well as PTX.
