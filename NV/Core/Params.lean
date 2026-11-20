import NV.Fact
import NV.SASS.Syntax

/-!
# The shared core's parameters

One `Params` value describes a GPU to the core: its structure (SMs, sub-partitions), its limits
(occupancy), its clock, and the timing of every opcode class. Every field is a `Fact`, so each
number carries its source; a part (`NV.Arch.*`) is a `Params` value and nothing else.
-/

namespace NV.Core

open NV.SASS

/-- Execution pipes of an SM sub-partition (Nsight Compute's pipe names). -/
inductive Pipe where
  /-- FP32 multiply-add, the half that also runs `IMAD` (`fmaheavy`) and the other (`fmalite`). -/
  | fmaHeavy | fmaLite
  | alu
  | fp16
  | fp64
  /-- Transcendentals and conversions (`MUFU`, `I2F`, …). -/
  | xu
  /-- Loads, stores, shuffles and other memory-unit work, through the MIO queue. -/
  | lsu
  /-- Warp-level tensor ops (`HMMA`, `IMMA`). -/
  | tensor
  /-- Warpgroup tensor ops (`HGMMA`). -/
  | gmma
  | uniform
  /-- Branches, barriers and other issue-only work. -/
  | cbu
  deriving DecidableEq, Repr, Inhabited, Hashable

def Pipe.index : Pipe → Nat
  | .fmaHeavy => 0 | .fmaLite => 1 | .alu => 2 | .fp16 => 3 | .fp64 => 4 | .xu => 5 | .lsu => 6
  | .tensor => 7 | .gmma => 8 | .uniform => 9 | .cbu => 10

def Pipe.count : Nat := 11

def Pipe.all : List Pipe :=
  [.fmaHeavy, .fmaLite, .alu, .fp16, .fp64, .xu, .lsu, .tensor, .gmma, .uniform, .cbu]

/-- What an opcode class costs a sub-partition. -/
structure OpTiming where
  pipe : Pipe
  /-- Cycles the pipe stays busy before it accepts the next instruction. -/
  interval : Nat
  /-- Cycles from issue to the result's writeback. For variable-latency classes this also
  releases the instruction's write dependence counter; for fixed-latency classes the stall counts
  cover it, and the engine uses it only to time the register write. -/
  latency : Nat := 0
  /-- For classes whose source registers are read after issue (memory instructions): cycles from
  issue to the release of the read dependence counter, after which the sources may be
  overwritten. 0: the part's `Params.readRelease`. -/
  readRelease : Nat := 0
  /-- For fixed-latency classes that read some sources after issue (a 64-bit addend's high word,
  in a wide multiply-add's second half): cycles from issue to those reads. The engine reads every
  source then (no earlier-issued result can land in between unseen). -/
  lateRead : Nat := 0
  /-- Cycles from issue until a predicate the instruction writes can guard another instruction,
  where the part reads guards later than data operands (0: its latency). -/
  guardLatency : Nat := 0
  deriving Repr, Inhabited

/-- One row of an opcode table: instructions with this opcode and all of `mods` have `timing`.
Rows are matched in order, so specific rows come first. -/
structure OpRule where
  opcode : String
  mods : Array String := #[]
  timing : Fact OpTiming
  deriving Repr, Inhabited

def OpRule.matches (r : OpRule) (i : Instr) : Bool :=
  r.opcode == i.opcode && r.mods.all i.hasMod

/-- Warp scheduler policies. `greedyThenYoungest` is Huerta et al.'s CGGTY. `leastRecent`: the
ready warp that issued longest ago, ties to the youngest; `greedyThenLeastRecent`: the warp that
issued last while it is ready (and did not yield), else `leastRecent`'s pick. -/
inductive IssuePolicy where
  | greedyThenOldest | greedyThenYoungest | leastRecent | greedyThenLeastRecent
  deriving DecidableEq, Repr, Inhabited

/-- The memory system of an idle machine. A latency is the cycles from a load's issue to the
earliest issue of an instruction that uses its result, for one thread. -/
structure MemoryParams where
  sectorBytes : Fact Nat
  lineBytes : Fact Nat
  /-- Combined L1 data cache and shared memory per SM. -/
  l1SmemBytes : Fact Nat
  smemBanks : Fact Nat
  smemBankBytes : Fact Nat
  smemLatency : Fact Nat
  l1HitLatency : Fact Nat
  /-- A load from a cluster peer's shared memory (`ld.shared::cluster`). -/
  dsmemLatency : Fact Nat
  l2Bytes : Fact Nat
  l2Slices : Fact Nat
  /-- The L2 is split in partitions; an SM reaches the far one through the crossbar. -/
  l2Partitions : Fact Nat
  l2NearHitLatency : Fact Nat
  l2FarHitLatency : Fact Nat
  /-- An L2 miss to DRAM, averaged over near and far partitions. -/
  dramLatency : Fact Nat
  dramBusBits : Fact Nat
  /-- DRAM interface clock; data moves on both edges. -/
  dramClockMHz : Fact Nat
  /-- Peak DRAM bandwidth, bytes per second. -/
  dramBytesPerSec : Fact Nat
  /-- The bandwidth all SMs streaming bulk copies from DRAM reach, bytes per second. -/
  dramStreamBytesPerSec : Fact Nat
  /-- Independent DRAM channels (64 bits each). -/
  dramChannels : Fact Nat
  /-- Memory instructions a sub-partition's MIO queue holds. -/
  mioQueueDepth : Fact Nat
  /-- Cycles between memory instructions the SM's shared memory pipeline accepts, from all
  sub-partitions together. -/
  lsuInterval : Fact Nat
  /-- Cycles between memory instructions one sub-partition's address calculation accepts. -/
  addressInterval : Fact Nat
  /-- Cycles more a global store holds its sub-partition's address calculation per 32-bit
  register of data each lane stores. -/
  storeRegCycles : Fact Nat
  /-- L1 or shared memory data bytes per cycle per SM. -/
  l1BytesPerCycle : Fact Nat
  /-- L1 tag lookups (sectors) per cycle per SM. -/
  l1TagSectorsPerCycle : Fact Nat
  l1Ways : Fact Nat
  /-- Outstanding L1 misses per SM. -/
  l1Mshrs : Fact Nat
  /-- Whether a global store invalidates the L1 line it writes (else it updates it). -/
  l1StoreInvalidates : Fact Bool
  l2Ways : Fact Nat
  /-- Read bytes per cycle (SM clock) one L2 slice returns. -/
  l2SliceBytesPerCycle : Fact Nat
  /-- Bytes per cycle (SM clock) one L2 slice takes in writes (stores, bulk stores). -/
  l2SliceWriteBytesPerCycle : Fact Nat
  /-- Bytes per cycle (SM clock) from the L2 to one SM through the crossbar. -/
  xbarBytesPerCycle : Fact Nat
  /-- Bytes per cycle (SM clock) one SM writes to the L2 through the crossbar (stores,
  reductions, atomics' operands). -/
  xbarWriteBytesPerCycle : Fact Nat
  /-- Bytes per cycle (SM clock) of reduction operands one SM sends to the L2. -/
  xbarRedBytesPerCycle : Fact Nat
  /-- Reduced bytes per second all SMs reach with `red.global.add.f32` into an L2-resident
  buffer. -/
  l2RedF32BytesPerSec : Fact Nat
  /-- Bytes per cycle per SM over the SM-to-SM (DSMEM) network. -/
  dsmemBytesPerCycle : Fact Nat
  /-- The shared memory sizes a kernel's carve-out can select, KiB per SM. -/
  smemCarveoutsKB : Fact (List Nat)
  /-- Bytes per cycle (SM clock) over the link between the two L2 partitions, whole chip. -/
  l2FarBytesPerCycle : Fact Nat
  /-- Cycles one shared-memory atomic wavefront occupies the shared memory pipeline. -/
  smemAtomicInterval : Fact Nat
  /-- A read of a line homed in the far L2 partition leaves a copy in the reader's partition. -/
  l2FarCopiesNear : Fact Bool
  /-- The SM clock the per-cycle L2-side rates (`l2SliceBytesPerCycle`, `l2SliceWriteBytesPerCycle`,
  `xbar*BytesPerCycle`, `l2FarBytesPerCycle`) were measured at: the L2 and crossbar keep their own
  clock, so at another SM clock each moves the same bytes per second, not per cycle. -/
  l2ClockRefMHz : Fact Nat
  /-- The share, in thousandths, of an L2 hit's latency that is fixed in time rather than in SM
  cycles. -/
  l2LatencyFixedPermille : Fact Nat
  /-- The SM clock the L2 hit latencies (`l2NearHitLatency`, `l2FarHitLatency`) were measured at. -/
  l2LatencyRefMHz : Fact Nat
  /-- Picoseconds a DRAM channel loses each time it turns between reading and writing. -/
  dramTurnaroundPs : Fact Nat
  deriving Repr, Inhabited

/-- A sub-partition's register file: banks, read ports and the operand reuse cache. -/
structure RegFileParams where
  /-- Banks per sub-partition; register `Rn` is in bank `n % banks`. -/
  banks : Fact Nat
  /-- Read ports per bank, each reading one 1024-bit warp register per cycle. -/
  readPortsPerBank : Fact Nat
  /-- Cycles a fixed-latency instruction spends reading its source operands. -/
  operandReadCycles : Fact Nat
  /-- Reuse-cache entries per bank, each holding one value per operand position. -/
  reuseEntriesPerBank : Fact Nat
  /-- Operand positions a reuse-cache entry holds a value for. -/
  reuseSlotsPerEntry : Fact Nat
  deriving Repr, Inhabited

/-- Instruction and constant delivery to a sub-partition. -/
structure FrontEndParams where
  instBytes : Fact Nat
  /-- Instructions fetched and decoded per cycle per sub-partition. -/
  fetchPerCycle : Fact Nat
  /-- Decoded instructions buffered per warp. -/
  instBufferEntries : Fact Nat
  /-- The SM's code caches, smallest first: capacity in bytes (LRU over 128-byte lines) and the
  bytes per cycle it is refilled at from the level below, the last level from L2. -/
  codeCaches : Fact (List (Nat × Nat))
  /-- Sequential code lines requested ahead of a warp entering a new line, from L2 (each a line
  request through the memory system, timed like a load). -/
  codePrefetchLines : Fact Nat
  /-- Cycles a warp loses to the fetch redirect after a taken branch, call or return. -/
  takenBranchCycles : Fact Nat
  /-- Whether an instruction encoded with a stall count of 0 interlocks: the next instruction
  waits for the fixed-latency results it reads. -/
  stallZeroInterlocks : Fact Bool
  /-- The constant cache's working set per SM. -/
  constCacheBytes : Fact Nat
  /-- The constant cache's line. -/
  constLineBytes : Fact Nat
  /-- Cycles a fixed-latency instruction's constant operand takes on a constant-cache miss. -/
  constMissCycles : Fact Nat
  /-- Cycles the scheduler waits on a constant miss before switching to another warp. -/
  constMissSwitchCycles : Fact Nat
  /-- Whether an SM rotates the sub-partitions of a CTA whose warps fill whole rows of them: the
  `k`-th CTA it places puts its warp `i` on sub-partition `(i + k) mod` the sub-partitions, so
  successive CTAs' first warps do not share one (a CTA of another warp count takes the next slots
  in order, which rotates it as well). -/
  rotateCtaWarps : Fact Bool
  deriving Repr, Inhabited

/-- Tensor-core throughput per SM per cycle, dense, and the warpgroup MMA's timing. -/
structure TensorParams where
  /-- FP16 or BF16 inputs (FP32 or FP16 accumulate), multiply-adds per cycle per SM. -/
  fmaPerCycle16 : Fact Nat
  /-- FP8 (e4m3, e5m2) inputs. -/
  fmaPerCycle8 : Fact Nat
  fmaPerCycleTF32 : Fact Nat
  fmaPerCycleF64 : Fact Nat
  /-- `wgmma.mma_async` m64nNk16 FP16 with both operands in shared memory: `(N, cycles)` from
  issue to completion. -/
  wgmmaLatency : Fact (List (Nat × Nat))
  /-- The same with A in registers (RS). -/
  wgmmaLatencyRS : Fact (List (Nat × Nat))
  /-- `mma.sync` m16n8k16 FP16/BF16 into FP32: dependent issue distance. -/
  mmaSyncLatency : Fact Nat
  /-- Per mille of a sub-partition's issue cycles its tensor core takes while it runs a warpgroup
  MMA (spread evenly); 0 on a part without warpgroup MMA. -/
  wgmmaIssueSharePermille : Fact Nat :=
    { value := 0, kind := .documented, source := { ref := "ptx-isa-9.5", loc := "§9.7.15: wgmma needs sm_90a" } }
  /-- Per mille of the sub-partition's pipes' time it takes meanwhile: an instruction's pipe
  occupancy stretches by `1000 / (1000 - it)`; 0 on a part without warpgroup MMA. -/
  wgmmaPipeSharePermille : Fact Nat :=
    { value := 0, kind := .documented, source := { ref := "ptx-isa-9.5", loc := "§9.7.15: wgmma needs sm_90a" } }
  /-- `tcgen05.mma` (`sm_100a`): cycles from an MMA's start on the tensor core to its result, at
  the least (its operand reads may take longer); 0 on a part without tensor memory. -/
  tcgen05Latency : Fact Nat :=
    { value := 0, kind := .documented, source := { ref := "ptx-isa-9.5", loc := "§9.7.18: tcgen05 needs sm_100a" } }
  /-- `tcgen05.ld`/`tcgen05.st`: cycles from the request leaving the LSU to its data. -/
  tmemLatency : Fact Nat :=
    { value := 0, kind := .documented, source := { ref := "ptx-isa-9.5", loc := "§9.7.18: tcgen05 needs sm_100a" } }
  deriving Repr, Inhabited

/-- Synchronization costs on an idle machine, in cycles unless the name says otherwise. -/
structure SyncParams where
  /-- `mbarrier.arrive` by one warp to the `try_wait` of another warp of the CTA observing it. -/
  mbarrierLatency : Fact Nat
  /-- The same between the two CTAs of a cluster (remote arrive). -/
  clusterMbarrierLatency : Fact Nat
  /-- Cycles from a cluster barrier's last arrival to its waiting warps going on. -/
  clusterBarrierCycles : Fact Nat
  /-- `bar.arrive` by one warp to the `bar.sync` of another returning. -/
  namedBarrierLatency : Fact Nat
  /-- `(threads, cycles)`: one `bar.sync 0` by every thread of a CTA of that size. -/
  barSync : Fact (List (Nat × Nat))
  /-- Fences: the cycles each adds to its warp. -/
  fenceProxyAsync : Fact Nat
  fenceAcqRelCta : Fact Nat
  fenceAcqRelGpu : Fact Nat
  fenceScGpu : Fact Nat
  fenceAcqRelSys : Fact Nat
  /-- `(SASS form, cycles)`: what a fence instruction holds its warp for after the warp's earlier
  requests have performed. -/
  fenceCycles : Fact (List (String × Nat))
  /-- `red.release.gpu`: the cycles it adds to its warp. -/
  redReleaseGpu : Fact Nat
  /-- The operation's round trip on an L2-resident word, its result consumed. -/
  atomAcqRelGpu : Fact Nat
  atomRelaxedGpu : Fact Nat
  ldAcquireGpu : Fact Nat
  /-- A release store on one SM seen by an acquire poll on another, picoseconds. -/
  flagHopPs : Fact Nat
  /-- A device `printf` (`vprintf` call) holds its warp this long when one lane prints, and this
  long more for each further lane printing in the same call. -/
  printfPs : Fact Nat
  printfLanePs : Fact Nat
  /-- Cycles an unfinished `mbarrier.try_wait` (SASS `SYNCS.PHASECHK.TRANS64.TRYWAIT`) holds its
  warp before returning false; the barrier's phase completing ends it sooner (0: it returns at
  once, as `test_wait`). -/
  tryWaitLimit : Fact Nat
  /-- Cycles from the write that completes a held `try_wait`'s phase to the wait looking again. -/
  tryWaitWake : Fact Nat
  deriving Repr, Inhabited

/-- Bulk copies (TMA, `cp.async.bulk`) from global to shared memory. -/
structure TmaParams where
  /-- `(bytes, cycles)` from issue to mbarrier completion of one copy alone, its source in DRAM. -/
  copyDram : Fact (List (Nat × Nat))
  /-- The same with the source resident in L2. -/
  copyL2 : Fact (List (Nat × Nat))
  /-- Bytes per second one SM's bulk requests are issued at. -/
  issueBytesPerSec : Fact Nat
  /-- Bytes of an SM's async-unit writes to global memory (bulk and tensor stores, reductions) that
  may be on their way to the L2 unanswered: a write's source counts as read only once the writes
  sent this many bytes before it are answered (0: no bound). -/
  storeCreditBytes : Fact Nat
  deriving Repr, Inhabited

/-- Host-side launch costs, picoseconds. -/
structure LaunchParams where
  emptyGraphPs : Fact Nat
  /-- From the last CTA of a kernel ending to the first CTA of the next starting. -/
  kernelGapPs : Fact Nat
  /-- The same with programmatic dependent launch. -/
  kernelGapPdlPs : Fact Nat
  /-- Cycles the GPU's CTA dispatcher takes per CTA: for the launch's first wave, and for a CTA
  replacing one that finished. -/
  ctaDispatchCycles : Fact (Nat × Nat)
  /-- Cycles from a kernel's start on the GPCs to its first CTA's dispatch, counted in the kernel's
  elapsed cycles (`gpc__cycles_elapsed`). -/
  gpcLaunchCycles : Fact Nat
  deriving Repr, Inhabited

/-- The links joining a board's GPUs, which a mesh's collectives cross (`NVGPU_LEAN_GPU=b200x8`). -/
structure LinkParams where
  /-- One GPU's bandwidth to the others in one direction, GB/s: NVLink (through NVSwitch, or a
  bridge), else PCIe. Half the bidirectional figure NVIDIA publishes. -/
  peerGBps : Fact Nat
  /-- The share of `peerGBps`, per mille, at which a large collective moves its data. -/
  peerPermille : Fact Nat
  deriving Repr, Inhabited

/-- What a part's SMs can do beyond the common core: the instructions and units its compiler target
enables. The engine refuses a kernel that uses a feature its part lacks. -/
structure Features where
  /-- `cp.async` (LDGSTS), compute capability 8.0 on. -/
  cpAsync : Bool
  /-- The tensor memory accelerator and bulk copies (`cp.async.bulk`, UTMALDG, UBLKCP), 9.0 on. -/
  tma : Bool
  /-- Warpgroup MMA (`wgmma`, HGMMA), `sm_90a` only. -/
  wgmma : Bool
  /-- Thread block clusters and distributed shared memory, 9.0 on. -/
  clusters : Bool
  /-- Tensor memory and the 5th-generation tensor cores (`tcgen05`, UTCHMMA, LDTM),
  `sm_100a`/`sm_103a` only. -/
  tcgen05 : Bool := false
  deriving Repr, Inhabited

/-- The unit a SASS opcode needs that the part may lack, if any. -/
def Features.lacks (f : Features) (opcode : String) : Option String :=
  if !f.wgmma && (opcode.endsWith "GMMA" || opcode == "WARPGROUP") then some "warpgroup MMA"
  else if !f.tma && (opcode.startsWith "UTMA" || opcode.startsWith "UBLK") then some "the TMA unit"
  else if !f.cpAsync && opcode == "LDGSTS" then some "cp.async"
  else if !f.clusters && opcode.startsWith "UCGABAR" then some "thread block clusters"
  else if !f.tcgen05 && (opcode.startsWith "UTC" || opcode == "LDTM" || opcode == "STTM") then
    some "tensor memory (tcgen05)"
  else none

structure Params where
  name : String
  /-- The compiler target its kernels are built for (`ptxas -arch`): `sm_80`, `sm_90a`, `sm_120a`. -/
  target : Fact String
  features : Fact Features
  sms : Fact Nat
  /-- SMs per GPC. A cluster's CTAs run on SMs of one GPC, so this bounds resident clusters. -/
  gpcSMs : Fact (List Nat)
  smsps : Fact Nat
  clockMHz : Fact Nat
  maxThreadsPerSM : Fact Nat
  maxCtasPerSM : Fact Nat
  maxWarpsPerSM : Fact Nat
  regsPerSM : Fact Nat
  /-- Registers are allocated per warp in units of this many (per thread × 32). -/
  regAllocUnit : Fact Nat
  smemPerSM : Fact Nat
  smemPerCtaMax : Fact Nat
  /-- Shared memory the system reserves per CTA. -/
  smemReservedPerCta : Fact Nat
  issuePolicy : Fact IssuePolicy
  /-- Cycles after issue at which a read dependence counter is released, for classes that do not
  give their own (`OpTiming.readRelease`). -/
  readRelease : Fact Nat
  /-- Cycles a `BAR.SYNC` adds after its last warp arrives. -/
  barrierCycles : Fact Nat
  /-- An SM takes one warp's barrier arrival per this many cycles. -/
  barrierArrivalCycles : Fact Nat
  maxClusterCtas : Fact Nat
  regFile : RegFileParams
  frontEnd : FrontEndParams
  mem : MemoryParams
  tensor : TensorParams
  sync : SyncParams
  tma : TmaParams
  launch : LaunchParams
  link : LinkParams
  ops : Array OpRule
  deriving Repr, Inhabited

def Params.timing (p : Params) (i : Instr) : Option OpTiming :=
  (p.ops.find? (·.matches i)).map (·.timing.value)

end NV.Core
