import NV.Core.Params
import NV.Core.Memory.Rate
import NV.Core.Memory.Map
import NV.Memory.Flat

/-!
# The memory system's configuration

What both halves of the memory system (`NV.Core.Memory.Sm`, the SMs' side; `NV.Core.Memory.L2`,
the crossbar, L2 and DRAM) share: who issued a request, what it delivers, the timing derived from
a part's `Params`, and the counters.
-/

namespace NV.Core

open NV

/-- Who issued a request: the engine's identifiers, echoed back in the delivery. -/
structure Origin where
  sm : Nat
  smsp : Nat
  /-- The CTA's index on its SM. -/
  cta : Nat
  /-- The warp's slot on its SM. -/
  slot : Nat
  deriving Repr, Inhabited, BEq, Hashable

/-- A request reaching its issuer: the response words (`Resp`; empty for stores, reductions and
fences) at cycle `at_`. Every request delivers: a store's delivery is its acknowledgement. -/
structure Delivery where
  id : Nat
  origin : Origin
  at_ : Nat
  resp : Resp
  fence : Bool
  deriving Inhabited

namespace Memory

/-- The memory system's timing, derived from a part's `Params` by `Config.ofParams`. -/
structure Config where
  /-- Whether the SMs check their CTAs' threads for data races (`RaceCheck`). -/
  races : Bool := false
  smsps : Nat
  /-- Per SM, the L2 partition it reaches without crossing the partition link. -/
  nearPartition : Array Nat
  sectorBytes : Nat
  lineBytes : Nat
  banks : Nat
  bankBytes : Nat
  mioDepth : Nat
  lsuInterval : Nat
  addressInterval : Nat
  storeRegCycles : Nat
  /-- Cycles per data-port wavefront. -/
  wavefrontCycles : Nat
  l1Sets : Nat
  l1Ways : Nat
  /-- Lines the L1 can have misses outstanding for at once (its MSHRs). -/
  l1Mshrs : Nat := 512
  /-- Sectors the L1 tag stage looks up per cycle. -/
  l1TagSectors : Nat := 4
  storeInvalidates : Bool
  /-- Issue to the tag stage (or to the data port, for shared memory). -/
  front : Nat
  /-- Tag stage or L1 fill to registers. -/
  backL1 : Nat
  /-- Data port to registers, shared memory. -/
  backShared : Nat
  /-- Crossbar request leg and response leg, to the near (0) and the far (1) partition. -/
  out : Array Nat
  back : Array Nat
  dramExtra : Nat
  slices : Nat
  partitions : Nat
  /-- Reads of a line homed in the far partition leave a copy in the reader's partition, which
  serves later reads there; writes go to the home slice. -/
  farCopies : Bool
  l2Sets : Nat
  l2Ways : Nat
  channels : Nat
  slice : Rate
  xbar : Rate
  /-- Write bytes a slice takes for every `sliceRead` it returns (`slice`'s rate). -/
  sliceRead : Nat
  sliceWrite : Nat
  xbarWrite : Rate
  /-- Reduction operands' bytes out of the SM for every `xbarWriteBytes` of stores' data
  (`xbarWrite`'s rate). -/
  xbarWriteBytes : Nat
  xbarRed : Nat
  far : Rate
  channel : Rate
  /-- Bytes of a channel's time a turn between reading and writing costs (`dramTurnaroundPs`). -/
  turnBytes : Nat := 0
  /-- Atomic operations per slice. -/
  atom : Rate
  /-- Cycles an atomic spends in the atomic unit beyond a hit (`MemParams.l2AtomicCycles`). -/
  atomCycles : Nat := 0
  dsmem : Rate
  dsmemOut : Nat
  dsmemBack : Nat
  smemAtomInterval : Nat
  /-- `tcgen05.ld`/`st` to their data (`TensorParams.tmemLatency`). -/
  tmemLatency : Nat := 0
  /-- A warpgroup MMA's operand reads take the data stage at this many per mille of a cycle a
  wavefront; 0, the tensor core's own port (`TensorParams.wgmmaOperandDataPermille`). -/
  wgmmaOperandData : Nat := 0
  /-- An unfinished `try_wait`'s hold (`SyncParams.tryWaitLimit`). -/
  tryWaitLimit : Nat := 0
  /-- A held `try_wait`'s wake (`SyncParams.tryWaitWake`). -/
  tryWaitWake : Nat := 0
  deriving Repr, Inhabited

/-- The shared-memory carve-out the driver picks for `demand` bytes of shared memory per SM: the
smallest supported size that holds it. -/
def carveout (sizesKB : List Nat) (demand : Nat) : Nat :=
  match (sizesKB.map (· * 1024)).filter (· ≥ demand) with
  | [] => (sizesKB.map (· * 1024)).foldl max 0
  | c :: cs => cs.foldl min c

/-- Per SM, its L2 partition. SMs are numbered GPC by GPC, and the first half of the GPCs reach
partition 0 directly (inferred; a100-whitepaper p. 35). -/
def nearPartitions (gpcSMs : List Nat) (partitions : Nat) : Array Nat := Id.run do
  let g := max gpcSMs.length 1
  let mut out := #[]
  let mut i := 0
  for n in gpcSMs do
    for _ in [0:n] do out := out.push (i * partitions / g)
    i := i + 1
  return out

/-- Timing for `p`, with `smemDemand` bytes of shared memory per SM fixing the L1's share of the
combined array. -/
def Config.ofParams (p : Params) (smemDemand : Nat) : Config :=
  let m := p.mem
  let s := m.smemLatency.value
  let h := m.l1HitLatency.value
  -- The L2 and crossbar keep their own clock (`l2ClockRefMHz`): at another SM clock their rates
  -- move the same bytes per second, and the time-fixed share of an L2 hit's latency
  -- (`l2LatencyFixedPermille`) takes the cycles that time is.
  let clock := p.clockMHz.value
  let ref := if m.l2ClockRefMHz.value == 0 then clock else m.l2ClockRefMHz.value
  let fixed := min m.l2LatencyFixedPermille.value 1000
  let lref := if m.l2LatencyRefMHz.value == 0 then clock else m.l2LatencyRefMHz.value
  let atClock (lat : Nat) : Nat :=
    (lat * (1000 - fixed) * lref + lat * fixed * clock + 500 * lref) / (1000 * max lref 1)
  let dramExtra := m.dramLatency.value - (m.l2NearHitLatency.value + m.l2FarHitLatency.value) / 2
  let near := atClock m.l2NearHitLatency.value
  let far := atClock m.l2FarHitLatency.value
  let front := s / 2
  let line := m.lineBytes.value
  let l1Bytes := m.l1SmemBytes.value - carveout m.smemCarveoutsKB.value smemDemand
  let slices := max m.l2Slices.value 1
  let channels := max m.dramChannels.value 1
  let dsExtra := m.dsmemLatency.value - s
  { smsps := p.smsps.value
    nearPartition := nearPartitions p.gpcSMs.value m.l2Partitions.value
    sectorBytes := m.sectorBytes.value
    lineBytes := line
    banks := m.smemBanks.value
    bankBytes := m.smemBankBytes.value
    mioDepth := m.mioQueueDepth.value
    lsuInterval := m.lsuInterval.value
    addressInterval := m.addressInterval.value
    storeRegCycles := m.storeRegCycles.value
    wavefrontCycles := max 1 ((m.smemBanks.value * m.smemBankBytes.value + m.l1BytesPerCycle.value - 1)
      / max m.l1BytesPerCycle.value 1)
    l1Sets := l1Bytes / line / max m.l1Ways.value 1
    l1Ways := m.l1Ways.value
    l1Mshrs := m.l1Mshrs.value
    l1TagSectors := m.l1TagSectorsPerCycle.value
    storeInvalidates := m.l1StoreInvalidates.value
    front
    backL1 := h - front
    backShared := s - front
    out := #[(near - h) / 2, (far - h) / 2]
    back := #[near - h - (near - h) / 2, far - h - (far - h) / 2]
    dramExtra
    slices
    partitions := max m.l2Partitions.value 1
    farCopies := m.l2FarCopiesNear.value
    l2Sets := m.l2Bytes.value / slices / line / max m.l2Ways.value 1
    l2Ways := m.l2Ways.value
    channels
    slice := Rate.atClock m.l2SliceBytesPerCycle.value ref clock
    sliceRead := m.l2SliceBytesPerCycle.value
    sliceWrite := m.l2SliceWriteBytesPerCycle.value
    xbar := Rate.atClock m.xbarBytesPerCycle.value ref clock
    xbarWrite := Rate.atClock m.xbarWriteBytesPerCycle.value ref clock
    xbarWriteBytes := m.xbarWriteBytesPerCycle.value
    xbarRed := m.xbarRedBytesPerCycle.value
    far := Rate.atClock m.l2FarBytesPerCycle.value ref clock
    channel := Rate.perSecond m.dramBytesPerSec.value (p.clockMHz.value * channels)
    turnBytes := m.dramTurnaroundPs.value * (m.dramBytesPerSec.value / channels) / 1000000000000
    atom := Rate.perSecond (m.l2RedF32BytesPerSec.value / 4) (p.clockMHz.value * slices)
    atomCycles := m.l2AtomicCycles.value
    dsmem := { units := m.dsmemBytesPerCycle.value }
    dsmemOut := dsExtra / 2
    dsmemBack := dsExtra - dsExtra / 2
    smemAtomInterval := m.smemAtomicInterval.value
    tmemLatency := p.tensor.tmemLatency.value
    wgmmaOperandData := p.tensor.wgmmaOperandDataPermille.value
    tryWaitLimit := p.sync.tryWaitLimit.value
    tryWaitWake := p.sync.tryWaitWake.value }

/-- The slice in partition `q` that holds copies of far lines: lines spread over the partition's
slices as over all slices. -/
def copySlice (c : Config) (line q : Nat) : Nat :=
  let per := max 1 (c.slices / c.partitions)
  q * per + spread line per

/-- The memory system's counters, named after the Nsight Compute metrics they correspond to. -/
structure Counters where
  /-- `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` -/
  globalLdRequests : Nat := 0
  /-- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` -/
  globalLdSectors : Nat := 0
  /-- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum` -/
  globalLdHitSectors : Nat := 0
  /-- `l1tex__t_requests_pipe_lsu_mem_global_op_st.sum` -/
  globalStRequests : Nat := 0
  /-- `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` -/
  globalStSectors : Nat := 0
  /-- `l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum`, `…_op_st.sum` -/
  localLdSectors : Nat := 0
  localStSectors : Nat := 0
  /-- `l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum` plus `…_op_red.sum` -/
  globalAtomSectors : Nat := 0
  /-- `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum` -/
  sharedWavefronts : Nat := 0
  /-- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` -/
  sharedBankConflicts : Nat := 0
  /-- `lts__t_sectors_op_read.sum` -/
  l2ReadSectors : Nat := 0
  /-- `lts__t_sectors_op_write.sum` -/
  l2WriteSectors : Nat := 0
  /-- `lts__t_sectors_op_atom.sum` plus `lts__t_sectors_op_red.sum` -/
  l2AtomSectors : Nat := 0
  /-- `lts__t_sectors_lookup_hit.sum` -/
  l2HitSectors : Nat := 0
  /-- `lts__t_sectors_lookup_miss.sum` -/
  l2MissSectors : Nat := 0
  /-- Sectors of requests that crossed the link between the L2 partitions (`lts__t_sectors_srcunit_ltcfabric`). -/
  l2FarSectors : Nat := 0
  /-- `dram__bytes_read.sum` -/
  dramReadBytes : Nat := 0
  /-- `dram__bytes_write.sum` -/
  dramWriteBytes : Nat := 0
  /-- Turns of a DRAM channel between reading and writing (no Nsight Compute metric). -/
  dramTurns : Nat := 0
  /-- Bytes requested over the SM-to-SM network (no single Nsight Compute metric). -/
  dsmemBytes : Nat := 0
  /-- The busiest SM's cycles of its tensor core's operand port into shared memory
  (`SmRes.tensor`: `tcgen05.mma`'s A and B reads; no Nsight Compute metric). -/
  tensorPortMax : Nat := 0
  deriving Repr, Inhabited

/-- Counters summed over the SMs and the L2 (`tensorPortMax`: the busiest's). -/
def Counters.add (a b : Counters) : Counters where
  globalLdRequests := a.globalLdRequests + b.globalLdRequests
  globalLdSectors := a.globalLdSectors + b.globalLdSectors
  globalLdHitSectors := a.globalLdHitSectors + b.globalLdHitSectors
  globalStRequests := a.globalStRequests + b.globalStRequests
  globalStSectors := a.globalStSectors + b.globalStSectors
  localLdSectors := a.localLdSectors + b.localLdSectors
  localStSectors := a.localStSectors + b.localStSectors
  globalAtomSectors := a.globalAtomSectors + b.globalAtomSectors
  sharedWavefronts := a.sharedWavefronts + b.sharedWavefronts
  sharedBankConflicts := a.sharedBankConflicts + b.sharedBankConflicts
  l2ReadSectors := a.l2ReadSectors + b.l2ReadSectors
  l2WriteSectors := a.l2WriteSectors + b.l2WriteSectors
  l2AtomSectors := a.l2AtomSectors + b.l2AtomSectors
  l2HitSectors := a.l2HitSectors + b.l2HitSectors
  l2MissSectors := a.l2MissSectors + b.l2MissSectors
  l2FarSectors := a.l2FarSectors + b.l2FarSectors
  dramReadBytes := a.dramReadBytes + b.dramReadBytes
  dramWriteBytes := a.dramWriteBytes + b.dramWriteBytes
  dramTurns := a.dramTurns + b.dramTurns
  dsmemBytes := a.dsmemBytes + b.dsmemBytes
  tensorPortMax := max a.tensorPortMax b.tensorPortMax

/-- The fewest cycles between an SM sending anything off the SM (to the L2, or to a peer SM) and
anything it causes reaching any SM: the longest quantum the two halves may run apart for. -/
def Config.lookahead (c : Config) : Nat :=
  [c.out[0]!, c.out[1]!, c.back[0]!, c.back[1]!, c.dsmemOut, c.dsmemBack].foldl min c.out[0]!

/-- Where local memory sits in the device address space, for the caches: above any real address. -/
def localBase : Nat := 1 <<< 56

/-- Loads that skip the L1: strong loads at GPU or system scope, and the `.cg`/`.cv` cache
operators (PTX ISA §9.7.9.1). -/
def bypassL1 (r : Req) : Bool :=
  (r.sem != .weak && (r.scope == .gpu || r.scope == .sys || r.scope == .cluster)) ||
    r.hint.ops.any (fun o => o == ".cg" || o == ".cv" || o == "STRONG" || o == "STRONG.GPU" ||
      o == "STRONG.SYS")

/-- The CTA rank and offset a `shared::cluster` address names: each CTA of the cluster has a
16 MiB window, rank in the bits above (the model's encoding; inferred). -/
def clusterRank (a : UInt64) : Nat := (a >>> 24).toNat
def clusterOffset (a : UInt64) : UInt64 := a &&& 0xFFFFFF

/-! ## Between the halves

In phase A of a quantum the SMs run apart; what leaves an SM is stamped with the cycle it leaves,
and the L2 half takes it in phase B, in stamp order. What reaches an SM is stamped with the cycle
it arrives, at least `lookahead` after it was caused. -/

/-- Who issues a request to the request machine: a warp, or a CTA's async unit. -/
inductive Agent where
  /-- Warp `slot` of CTA `cta` on `sm`. -/
  | warp (sm slot cta : Nat)
  /-- Asynchronous operation `op` (the async unit's agent) of CTA `cta` on `sm`: each copy,
  bulk operation or MMA is its own thread of the request machine, its requests in program order
  (a complete-tx after the copy's writes), independent of the CTA's other operations. -/
  | async (sm cta op : Nat)
  deriving BEq, Hashable, Repr, Inhabited

/-- A request (or, for a cluster request, the part going to one CTA) as the request machine knows
it: the SM's id for the request and the part. -/
structure Key where
  sm : Nat
  req : Nat
  part : Nat := 0
  deriving BEq, Hashable, Repr, Inhabited

/-- What an SM did to the request machine in phase A, replayed in phase B in stamp order. -/
inductive Log where
  /-- `agent` issued request part `key`, its shared addresses naming CTA `cta` (`(sm, index)`). -/
  | issue (key : Key) (agent : Agent) (cta : Nat × Nat) (r : Req)
  /-- Lane `lane` of `key` performed in phase A; its reads returned `read` there. -/
  | perform (key : Key) (lane : Nat) (read : List UInt8)
  /-- A warp's fence passed. -/
  | fence (agent : Agent)
  deriving Inhabited

/-- One line of a global request on its way from an SM to the L2. -/
structure LineReq where
  key : Key
  /-- The SM's transaction. -/
  txn : Nat
  req : Req
  line : Nat
  sectors : Nat
  /-- The lanes that perform at the slice. -/
  lanes : Array Nat
  /-- The line's home slice; for a line in the other partition, the slice in the SM's partition
  that holds copies, which every request of the SM to the line passes. -/
  slice : Nat
  copy : Option Nat
  /-- Bytes it carries (store and atomic data). -/
  bytes : Nat
  deriving Inhabited

/-- A cluster request's part for CTA `cta` of SM `to`, performed at that SM's data port. -/
structure PeerReq where
  key : Key
  req : Req
  to : Nat
  cta : Nat
  lanes : Array Nat
  deriving Inhabited

/-- What crosses between the halves. -/
inductive Msg where
  /-- SM → L2: a line request. -/
  | line (x : LineReq)
  /-- L2 → SM: transaction `txn` answered, with the response words of the lanes that performed. -/
  | answer (txn : Nat) (resp : Array (Nat × Array UInt32))
  /-- SM → peer SM, through the L2 half's network: a cluster request part. -/
  | peer (x : PeerReq)
  /-- Peer SM → SM: the answer to part `key`, with its lanes' response words. -/
  | peerAnswer (key : Key) (resp : Array (Nat × Array UInt32))
  deriving Inhabited

/-- A message, stamped with its cycle and its sender's sequence number (a total order). -/
structure Stamped (α : Type) where
  stamp : Nat
  sm : Nat
  seq : Nat
  body : α
  deriving Inhabited

def Stamped.before {α : Type} (a b : Stamped α) : Bool :=
  a.stamp < b.stamp || (a.stamp == b.stamp && (a.sm < b.sm || (a.sm == b.sm && a.seq < b.seq)))

/-- A lane's response words from the bytes its reads returned. -/
def laneWords (r : Req) (read : List UInt8) : Array UInt32 := (r.laneResp read).toArray

end Memory

end NV.Core
