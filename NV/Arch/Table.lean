import NV.Core.Params

/-!
# Writing and printing a part's facts

Constructors for facts by kind, and a part's `Params` as named rows for `lake exe facts`, which
prints `docs/facts-<part>.md` from them: the table a reader sees is the one the model runs.
-/

namespace NV

def Kind.name : Kind → String
  | .documented => "documented"
  | .measured => "measured"
  | .published => "published"
  | .encoded => "encoded"
  | .simulator => "simulator"
  | .inferred => "inferred"

def Core.IssuePolicy.name : Core.IssuePolicy → String
  | .greedyThenOldest => "greedy then oldest"
  | .greedyThenYoungest => "greedy then youngest"
  | .leastRecent => "least recently issued"
  | .greedyThenLeastRecent => "greedy then least recently issued"

def Core.Pipe.name : Core.Pipe → String
  | .fmaHeavy => "fmaheavy" | .fmaLite => "fmalite" | .alu => "alu" | .fp16 => "fp16"
  | .fp64 => "fp64" | .xu => "xu" | .lsu => "lsu" | .tensor => "tensor" | .gmma => "gmma"
  | .uniform => "uniform" | .cbu => "cbu"

def Core.OpTiming.render (t : Core.OpTiming) : String :=
  s!"{t.pipe.name}, every {t.interval}" ++
    (if t.latency == 0 then "" else s!", latency {t.latency}") ++
    (if t.readRelease == 0 then "" else s!", reads released at {t.readRelease}") ++
    (if t.lateRead == 0 then "" else s!", reads its sources at {t.lateRead}") ++
    (if t.guardLatency == 0 then "" else s!", guards from {t.guardLatency}") ++
    (if t.holds then ", holds the warp until then" else "")

namespace Arch

open NV.Core

def doc {α : Type} (v : α) (ref loc : String) (note := "") : Fact α :=
  { value := v, kind := .documented, source := { ref, loc }, note }

/-- Our H200 measurement. -/
def meas {α : Type} (v : α) (loc : String) (note := "") : Fact α :=
  { value := v, kind := .measured, source := { ref := "meas-h200", loc }, note }

/-- Our measurement on another part than the H200, or a device attribute of a board. -/
def measOn {α : Type} (v : α) (loc : String) (note := "") : Fact α :=
  { value := v, kind := .measured, source := { ref := "meas", loc }, note }

def pub {α : Type} (v : α) (ref loc measuredOn : String) (note := "") : Fact α :=
  { value := v, kind := .published, source := { ref, loc }, measuredOn := some measuredOn, note }

def encoded {α : Type} (v : α) (loc : String) (note := "") : Fact α :=
  { value := v, kind := .encoded, source := { ref := "fixtures", loc }, note }

def inferred {α : Type} (v : α) (ref loc note : String) (measuredOn : Option String := none) :
    Fact α :=
  { value := v, kind := .inferred, source := { ref, loc }, measuredOn, note }

/-- A fact used for another Hopper part: if it rests on our H200 measurements, it is labelled with
the part it was measured on. -/
def onH200 {α : Type} (f : Fact α) : Fact α :=
  if f.source.ref == "meas-h200" then { f with measuredOn := some "H200" } else f

/-- A fact one part's evidence gives, used for another part: a measurement is labelled with the
part it was measured on. -/
def onPart {α : Type} (part : String) (f : Fact α) : Fact α :=
  if f.kind == .measured && f.measuredOn.isNone then { f with measuredOn := some part } else f

/-- One fact of a part: its name, its value printed, and its evidence. -/
structure Row where
  name : String
  value : String
  kind : Kind
  source : Source
  measuredOn : Option String
  note : String

def row {α : Type} (name : String) (f : Fact α) (render : α → String) : Row :=
  { name, value := render f.value, kind := f.kind, source := f.source,
    measuredOn := f.measuredOn, note := f.note }

def rowN (name : String) (f : Fact Nat) : Row := row name f toString

def Features.render (f : Features) : String :=
  let on := [("cp.async", f.cpAsync), ("TMA", f.tma), ("wgmma", f.wgmma), ("clusters", f.clusters)]
  ", ".intercalate ((on.filter (·.2)).map (·.1))

def showPairs (ps : List (Nat × Nat)) : String :=
  ", ".intercalate (ps.map fun (a, b) => s!"{a}: {b}")

def showList (ns : List Nat) : String := ", ".intercalate (ns.map toString)

/-- A part's facts in titled sections. Every field of `Params` appears: the patterns below name
all of them, so a new field does not compile until it has a row. -/
def sections (p : Params) : List (String × List Row) :=
  let ⟨_, target, features, sms, gpcSMs, smsps, clockMHz, maxThreadsPerSM, maxCtasPerSM, maxWarpsPerSM, regsPerSM,
    regAllocUnit, smemPerSM, smemPerCtaMax, smemReservedPerCta, issuePolicy, readRelease,
    barrierCycles, barrierArrivalCycles, maxClusterCtas, regFile, frontEnd, mem, tensor, sync, tma,
    launch, link, ops⟩ := p
  let ⟨banks, readPortsPerBank, operandReadCycles, reuseEntriesPerBank, reuseSlotsPerEntry⟩ :=
    regFile
  let ⟨instBytes, fetchPerCycle, instBufferEntries, codeCaches, codePrefetchLines, codeLaunchBytes,
    codeSharedLinesPerKCycle, takenBranchCycles, forwardBranchCycles, indirectMissCycles, stallZeroInterlocks,
    constCacheBytes, constLineBytes, constMissCycles, constMissSwitchCycles, constSharedLinesPerKCycle,
    rotateCtaWarps⟩ := frontEnd
  let ⟨sectorBytes, lineBytes, l1SmemBytes, smemBanks, smemBankBytes, smemLatency, l1HitLatency,
    dsmemLatency, l2Bytes, l2Slices, l2Partitions, l2NearHitLatency, l2FarHitLatency, dramLatency,
    dramBusBits, dramClockMHz, dramBytesPerSec, dramStreamBytesPerSec, dramChannels, mioQueueDepth,
    lsuInterval, addressInterval, storeRegCycles, l1BytesPerCycle, l1TagSectorsPerCycle, l1Ways, l1Mshrs,
    l1StoreInvalidates, l2Ways, l2SliceBytesPerCycle, l2SliceWriteBytesPerCycle, xbarBytesPerCycle, gpcLinkBytesPerCycle, xbarWriteBytesPerCycle, xbarRedBytesPerCycle, l2RedF32BytesPerSec,
    dsmemBytesPerCycle, smemCarveoutsKB, l2FarBytesPerCycle, smemAtomicInterval,
    l2FarCopiesNear, l2ClockRefMHz, l2LatencyFixedPermille, l2LatencyRefMHz, dramTurnaroundPs, l2AtomicCycles⟩ := mem
  let ⟨fmaPerCycle16, fmaPerCycle8, fmaPerCycleTF32, fmaPerCycleF64, wgmmaLatency, wgmmaLatencyRS,
    mmaSyncLatency, wgmmaOperandDataPermille, wgmmaIssueSharePermille, wgmmaPipeSharePermille, tcgen05Latency, tmemLatency⟩ := tensor
  let ⟨mbarrierLatency, clusterMbarrierLatency, clusterBarrierCycles, namedBarrierLatency, barArriveCycles, barSync,
    fenceProxyAsync,
    fenceAcqRelCta, fenceAcqRelGpu, fenceScGpu, fenceAcqRelSys, fenceCycles, redReleaseGpu,
    atomAcqRelGpu,
    atomRelaxedGpu, ldAcquireGpu, flagHopPs, printfPs, printfLanePs, tryWaitLimit, tryWaitWake⟩ := sync
  let ⟨copyDram, copyL2, issueBytesPerSec, storeCreditBytes⟩ := tma
  let ⟨emptyGraphPs, kernelGapPs, kernelGapPdlPs, ctaDispatchCycles, gpcLaunchCycles⟩ := launch
  let ⟨peerGBps, peerPermille⟩ := link
  [("Structure and clock",
    [row "target" target id, row "features" features Features.render, rowN "sms" sms, row "gpcSMs" gpcSMs showList, rowN "smsps" smsps, rowN "clockMHz" clockMHz,
     rowN "maxClusterCtas" maxClusterCtas]),
   ("Occupancy limits",
    [rowN "maxThreadsPerSM" maxThreadsPerSM, rowN "maxCtasPerSM" maxCtasPerSM,
     rowN "maxWarpsPerSM" maxWarpsPerSM, rowN "regsPerSM" regsPerSM,
     rowN "regAllocUnit" regAllocUnit, rowN "smemPerSM" smemPerSM,
     rowN "smemPerCtaMax" smemPerCtaMax, rowN "smemReservedPerCta" smemReservedPerCta]),
   ("Issue",
    [row "issuePolicy" issuePolicy IssuePolicy.name, rowN "readRelease" readRelease,
     rowN "barrierCycles" barrierCycles, rowN "barrierArrivalCycles" barrierArrivalCycles]),
   ("Register file (per sub-partition)",
    [rowN "regFile.banks" banks, rowN "regFile.readPortsPerBank" readPortsPerBank,
     rowN "regFile.operandReadCycles" operandReadCycles,
     rowN "regFile.reuseEntriesPerBank" reuseEntriesPerBank,
     rowN "regFile.reuseSlotsPerEntry" reuseSlotsPerEntry]),
   ("Instruction and constant delivery",
    [rowN "frontEnd.instBytes" instBytes, rowN "frontEnd.fetchPerCycle" fetchPerCycle,
     rowN "frontEnd.instBufferEntries" instBufferEntries,
     row "frontEnd.codeCaches" codeCaches showPairs,
     rowN "frontEnd.codePrefetchLines" codePrefetchLines,
     rowN "frontEnd.codeLaunchBytes" codeLaunchBytes,
     rowN "frontEnd.codeSharedLinesPerKCycle" codeSharedLinesPerKCycle,
     rowN "frontEnd.takenBranchCycles" takenBranchCycles,
     rowN "frontEnd.forwardBranchCycles" forwardBranchCycles,
     rowN "frontEnd.indirectMissCycles" indirectMissCycles,
     row "frontEnd.stallZeroInterlocks" stallZeroInterlocks toString,
     row "frontEnd.rotateCtaWarps" rotateCtaWarps toString,
     rowN "frontEnd.constCacheBytes" constCacheBytes,
     rowN "frontEnd.constLineBytes" constLineBytes,
     rowN "frontEnd.constMissCycles" constMissCycles,
     rowN "frontEnd.constMissSwitchCycles" constMissSwitchCycles,
     rowN "frontEnd.constSharedLinesPerKCycle" constSharedLinesPerKCycle]),
   ("Opcode classes (pipe, issue interval per sub-partition, latency, read release)",
    ops.toList.map fun r =>
      row (".".intercalate (r.opcode :: r.mods.toList) ++ (if r.indexed then " (indexed)" else ""))
        r.timing OpTiming.render),
   ("Memory (cycles)",
    [rowN "mem.sectorBytes" sectorBytes, rowN "mem.lineBytes" lineBytes,
     rowN "mem.l1SmemBytes" l1SmemBytes, rowN "mem.smemBanks" smemBanks,
     rowN "mem.smemBankBytes" smemBankBytes, rowN "mem.smemLatency" smemLatency,
     rowN "mem.l1HitLatency" l1HitLatency, rowN "mem.dsmemLatency" dsmemLatency,
     rowN "mem.l2Bytes" l2Bytes, rowN "mem.l2Slices" l2Slices,
     rowN "mem.l2Partitions" l2Partitions, rowN "mem.l2NearHitLatency" l2NearHitLatency,
     rowN "mem.l2FarHitLatency" l2FarHitLatency, rowN "mem.dramLatency" dramLatency,
     rowN "mem.dramBusBits" dramBusBits, rowN "mem.dramClockMHz" dramClockMHz,
     rowN "mem.dramBytesPerSec" dramBytesPerSec,
     rowN "mem.dramStreamBytesPerSec" dramStreamBytesPerSec, rowN "mem.dramChannels" dramChannels,
     rowN "mem.mioQueueDepth" mioQueueDepth, rowN "mem.lsuInterval" lsuInterval,
     rowN "mem.addressInterval" addressInterval, rowN "mem.storeRegCycles" storeRegCycles, rowN "mem.l1BytesPerCycle" l1BytesPerCycle,
     rowN "mem.l1TagSectorsPerCycle" l1TagSectorsPerCycle, rowN "mem.l1Ways" l1Ways,
     rowN "mem.l1Mshrs" l1Mshrs, row "mem.l1StoreInvalidates" l1StoreInvalidates toString,
     rowN "mem.l2Ways" l2Ways, rowN "mem.l2SliceBytesPerCycle" l2SliceBytesPerCycle, rowN "mem.l2SliceWriteBytesPerCycle" l2SliceWriteBytesPerCycle,
     rowN "mem.xbarBytesPerCycle" xbarBytesPerCycle, rowN "mem.gpcLinkBytesPerCycle" gpcLinkBytesPerCycle,
     rowN "mem.xbarWriteBytesPerCycle" xbarWriteBytesPerCycle,
     rowN "mem.xbarRedBytesPerCycle" xbarRedBytesPerCycle,
     rowN "mem.l2RedF32BytesPerSec" l2RedF32BytesPerSec,
     rowN "mem.l2AtomicCycles" l2AtomicCycles,
     rowN "mem.dsmemBytesPerCycle" dsmemBytesPerCycle,
     row "mem.smemCarveoutsKB" smemCarveoutsKB showList,
     rowN "mem.l2FarBytesPerCycle" l2FarBytesPerCycle,
     rowN "mem.smemAtomicInterval" smemAtomicInterval,
     row "mem.l2FarCopiesNear" l2FarCopiesNear toString,
     rowN "mem.l2ClockRefMHz" l2ClockRefMHz,
     rowN "mem.l2LatencyFixedPermille" l2LatencyFixedPermille,
     rowN "mem.l2LatencyRefMHz" l2LatencyRefMHz,
     rowN "mem.dramTurnaroundPs" dramTurnaroundPs]),
   ("Tensor cores (per SM per cycle, dense)",
    [rowN "tensor.fmaPerCycle16" fmaPerCycle16, rowN "tensor.fmaPerCycle8" fmaPerCycle8,
     rowN "tensor.fmaPerCycleTF32" fmaPerCycleTF32, rowN "tensor.fmaPerCycleF64" fmaPerCycleF64,
     row "tensor.wgmmaLatency" wgmmaLatency showPairs,
     row "tensor.wgmmaLatencyRS" wgmmaLatencyRS showPairs,
     rowN "tensor.mmaSyncLatency" mmaSyncLatency,
     rowN "tensor.wgmmaOperandDataPermille" wgmmaOperandDataPermille,
     rowN "tensor.wgmmaIssueSharePermille" wgmmaIssueSharePermille,
     rowN "tensor.wgmmaPipeSharePermille" wgmmaPipeSharePermille,
     rowN "tensor.tcgen05Latency" tcgen05Latency,
     rowN "tensor.tmemLatency" tmemLatency]),
   ("Synchronization (cycles unless the name says otherwise)",
    [rowN "sync.mbarrierLatency" mbarrierLatency,
     rowN "sync.clusterMbarrierLatency" clusterMbarrierLatency,
     rowN "sync.clusterBarrierCycles" clusterBarrierCycles,
     rowN "sync.namedBarrierLatency" namedBarrierLatency, rowN "sync.barArriveCycles" barArriveCycles, rowN "sync.tryWaitLimit" tryWaitLimit,
     rowN "sync.tryWaitWake" tryWaitWake,
     row "sync.barSync" barSync showPairs,
     rowN "sync.fenceProxyAsync" fenceProxyAsync, rowN "sync.fenceAcqRelCta" fenceAcqRelCta,
     rowN "sync.fenceAcqRelGpu" fenceAcqRelGpu, rowN "sync.fenceScGpu" fenceScGpu,
     rowN "sync.fenceAcqRelSys" fenceAcqRelSys,
     row "sync.fenceCycles" fenceCycles
       (fun fs => ", ".intercalate (fs.map fun (f, c) => s!"{f}: {c}")),
     rowN "sync.redReleaseGpu" redReleaseGpu,
     rowN "sync.atomAcqRelGpu" atomAcqRelGpu, rowN "sync.atomRelaxedGpu" atomRelaxedGpu,
     rowN "sync.ldAcquireGpu" ldAcquireGpu, rowN "sync.flagHopPs" flagHopPs,
     rowN "sync.printfPs" printfPs, rowN "sync.printfLanePs" printfLanePs]),
   ("Bulk copies (bytes: cycles)",
    [row "tma.copyDram" copyDram showPairs, row "tma.copyL2" copyL2 showPairs,
     rowN "tma.issueBytesPerSec" issueBytesPerSec, rowN "tma.storeCreditBytes" storeCreditBytes]),
   ("Launch (picoseconds)",
    [rowN "launch.emptyGraphPs" emptyGraphPs, rowN "launch.kernelGapPs" kernelGapPs,
     rowN "launch.kernelGapPdlPs" kernelGapPdlPs,
     row "launch.ctaDispatchCycles" ctaDispatchCycles (fun (a, b) => s!"{a} (first wave), {b} (after)"),
     rowN "launch.gpcLaunchCycles" gpcLaunchCycles]),
   ("Links between GPUs (a mesh's collectives)",
    [rowN "link.peerGBps" peerGBps, rowN "link.peerPermille" peerPermille])]

def rows (p : Params) : List Row := (sections p).flatMap (·.2)

end Arch
end NV
