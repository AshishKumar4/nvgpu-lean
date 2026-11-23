/-!
# What each parameter is for

Every field of a part's `Params` is a cited fact, but not every fact drives the simulation. This
inventory says, per field, who reads it, so a reader of the facts tables knows which numbers
change a simulated run. `lake exe facts` generates `docs/params.md` from it, and the `facts` CI
row checks that every field is listed and that every `engine` field is read by the core.
-/

namespace NV.Core

inductive Role where
  /-- The engine, memory system or async units read it: changing it changes simulated runs. -/
  | engine
  /-- A measured end-to-end cost the bounds layer uses as a floor; the engine produces it from mechanisms rather than reading it. -/
  | observed
  /-- An input to a derived engine parameter (e.g. bus width and clock to the DRAM bandwidth). -/
  | derivation
  /-- Documented hardware the engine does not model yet: recorded, not used. -/
  | notModelled
  /-- Read beyond one GPU's engine: a mesh's collective times. -/
  | mesh
  deriving DecidableEq, Repr

def Role.text : Role → String
  | .engine => "engine"
  | .observed => "observed: bound input, not read by the engine"
  | .derivation => "derivation input"
  | .notModelled => "not modelled yet"
  | .mesh => "mesh: the emulated device's collective times"

/-- Each field, by its path in `Params`, and its role. -/
def roles : List (String × Role) := [
  ("target", .engine), ("features", .engine), ("sms", .engine), ("gpcSMs", .engine),
  ("smsps", .engine), ("clockMHz", .engine), ("maxThreadsPerSM", .engine),
  ("maxCtasPerSM", .engine), ("maxWarpsPerSM", .engine), ("regsPerSM", .engine),
  ("regAllocUnit", .engine), ("smemPerSM", .engine), ("smemPerCtaMax", .engine),
  ("smemReservedPerCta", .engine), ("maxClusterCtas", .engine), ("issuePolicy", .engine),
  ("readRelease", .engine), ("barrierCycles", .engine), ("barrierArrivalCycles", .engine),
  ("ops", .engine),
  ("regFile.banks", .engine), ("regFile.readPortsPerBank", .engine),
  ("regFile.operandReadCycles", .notModelled), ("regFile.reuseEntriesPerBank", .engine),
  ("regFile.reuseSlotsPerEntry", .engine),
  ("frontEnd.instBytes", .notModelled), ("frontEnd.fetchPerCycle", .notModelled),
  ("frontEnd.instBufferEntries", .notModelled), ("frontEnd.codeCaches", .engine),
  ("frontEnd.codePrefetchLines", .engine), ("frontEnd.codeLaunchBytes", .engine),
  ("frontEnd.takenBranchCycles", .engine), ("frontEnd.forwardBranchCycles", .engine),
  ("frontEnd.indirectMissCycles", .engine), ("frontEnd.stallZeroInterlocks", .engine),
  ("frontEnd.rotateCtaWarps", .engine),
  ("frontEnd.constCacheBytes", .engine), ("frontEnd.constLineBytes", .engine),
  ("frontEnd.constMissCycles", .engine), ("frontEnd.constMissSwitchCycles", .engine),
  ("mem.dramBusBits", .derivation), ("mem.dramClockMHz", .derivation),
  ("mem.dramStreamBytesPerSec", .observed), ("mem.l1TagSectorsPerCycle", .engine),
  ("mem.l1Mshrs", .engine), ("mem.storeRegCycles", .engine),
  ("mem.xbarWriteBytesPerCycle", .engine), ("mem.xbarRedBytesPerCycle", .engine),
  ("mem.l2SliceWriteBytesPerCycle", .engine),
  ("mem.l2ClockRefMHz", .engine), ("mem.l2LatencyFixedPermille", .engine), ("mem.l2LatencyRefMHz", .engine),
  ("mem.dramTurnaroundPs", .engine),
  ("tma.storeCreditBytes", .engine),
  ("tensor.fmaPerCycle16", .engine), ("tensor.fmaPerCycle8", .engine),
  ("tensor.fmaPerCycleTF32", .engine), ("tensor.wgmmaLatency", .engine),
  ("tensor.wgmmaLatencyRS", .engine), ("tensor.fmaPerCycleF64", .derivation),
  ("tensor.mmaSyncLatency", .observed),
  ("tensor.wgmmaOperandDataPermille", .engine), ("tensor.wgmmaIssueSharePermille", .engine),
  ("tensor.wgmmaPipeSharePermille", .engine), ("tensor.tcgen05Latency", .engine), ("tensor.tmemLatency", .engine),
  ("sync.mbarrierLatency", .observed), ("sync.clusterMbarrierLatency", .observed),
  ("sync.namedBarrierLatency", .observed), ("sync.barArriveCycles", .engine), ("sync.tryWaitLimit", .engine),
  ("sync.tryWaitWake", .engine), ("sync.barSync", .observed),
  ("sync.fenceProxyAsync", .observed), ("sync.fenceAcqRelCta", .observed),
  ("sync.fenceAcqRelGpu", .observed), ("sync.fenceScGpu", .observed),
  ("sync.fenceAcqRelSys", .observed), ("sync.redReleaseGpu", .observed),
  ("sync.atomAcqRelGpu", .observed), ("sync.atomRelaxedGpu", .observed),
  ("sync.ldAcquireGpu", .observed), ("sync.flagHopPs", .observed),
  ("sync.printfPs", .engine), ("sync.printfLanePs", .engine),
  ("sync.fenceCycles", .engine), ("sync.clusterBarrierCycles", .engine),
  ("tma.copyDram", .observed), ("tma.copyL2", .observed), ("tma.issueBytesPerSec", .engine),
  ("launch.emptyGraphPs", .observed), ("launch.kernelGapPs", .engine),
  ("launch.kernelGapPdlPs", .engine), ("launch.ctaDispatchCycles", .engine),
  ("launch.gpcLaunchCycles", .engine),
  ("link.peerGBps", .mesh), ("link.peerPermille", .mesh)]

/-- The role of a field (`mem.*`, `tensor.*`… not listed are the engine's). -/
def roleOf (field : String) : Role :=
  (roles.lookup field).getD .engine

end NV.Core
