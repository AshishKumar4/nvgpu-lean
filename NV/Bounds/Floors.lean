import NV.Bounds.Bound

/-!
# Decode-layer floors

A kernel of a decode layer is bounded by three floors, and the layer by `layerFloor`; each is an
instance of this layer's terms, with its assumption made a term's shape:

* **stream**: HBM bytes at the streaming plateau plus a fixed start-up and drain, or the dense
  tensor work if larger. Here: `seq (chain [fixed]) (rate plateau bytes)` beside
  `rate tensor fma`. `seq` is the assumption that the start-up and the stream do not overlap;
  the plateau (`mem.dramStreamBytesPerSec`, measured) is below the engine's documented channel
  rate, so this term is a floor of the ledger's model of HBM, not of the engine's.
* **issue**: the busiest sub-partition's warp instructions, one per cycle: `pipe 1 1 k`.
* **chain**: serial dependent latencies at their idle values: `chain lats`.

`kernelFloor_cycles` states the instance's value, `max (max stream issue) chain`, in cycles at the
part's clock, each quotient rounded up (time is whole cycles, so the rounded-up figure is still a
floor).

The layer floor `F + max(weights / BW + KV / BW_gather, pre-route ‖ steps + route + MoE + post)`
is `seq (chain [F]) (both (seq weights kv) chain-part)`, with the chain part's inner maximum a
`both`: `layerFloor_cycles`.
-/

namespace NV.Bounds.Floors

open NV NV.Core NV.Bounds

variable (p : Params)

/-- Picoseconds to cycles of the part's clock, rounded down (a floor stays a floor). -/
def psCycles (ps : Nat) : Nat := ps * p.clockMHz.value / 1000000

/-- HBM at the measured streaming plateau, one resource. -/
def plateau : Resource where
  name := "HBM (streaming plateau)"
  unit := "bytes"
  capacity := { instances := 1, rate := Rate.perSecond p.mem.dramStreamBytesPerSec.value p.clockMHz.value }
  facts := [.of "mem.dramStreamBytesPerSec" p.mem.dramStreamBytesPerSec, .of "clockMHz" p.clockMHz]

/-- The tensor cores alone, at the dense BF16 rate. -/
def tensor : Resource where
  name := "tensor cores (BF16 dense)"
  unit := "multiply-adds"
  capacity := { instances := p.sms.value, rate := { units := p.tensor.fmaPerCycle16.value } }
  facts := [.of "tensor.fmaPerCycle16" p.tensor.fmaPerCycle16, .of "sms" p.sms]

/-! ## Chains, from this part's facts -/

/-- A kernel boundary with programmatic dependent launch. -/
def pdl : Nat := psCycles p p.launch.kernelGapPdlPs.value

/-- One cold 36 KB bulk copy, issue to completion (`tma.copyDram`). -/
def tmaCold : Nat := ((p.tma.copyDram.value.find? (·.1 == 36864)).map (·.2)).getD 0

def l2Load : Nat := p.mem.l2NearHitLatency.value

def hbmLoad : Nat := p.mem.dramLatency.value

/-- `streamChain`: boundary, first cold copy, one dependent store of the result. -/
def streamChain : List Nat := [pdl p, tmaCold p, l2Load p]

/-- `attentionChain`: boundary, page-table read, K/V copy, the split's counter, the merge's read
of the partials and its store. -/
def attentionChain : List Nat :=
  [pdl p, hbmLoad p, tmaCold p, p.sync.atomAcqRelGpu.value, l2Load p, l2Load p]

/-- `moeChain`: boundary, first units, the phase credit (release, remote acquire), the grid
barrier (atomic, poll), the combine's store. -/
def moeChain : List Nat :=
  [pdl p, tmaCold p, p.sync.redReleaseGpu.value, psCycles p p.sync.flagHopPs.value,
   p.sync.atomAcqRelGpu.value, psCycles p p.sync.flagHopPs.value, l2Load p]

def chainFacts : List FactRef :=
  [.of "launch.kernelGapPdlPs" p.launch.kernelGapPdlPs, .of "mem.l2NearHitLatency" p.mem.l2NearHitLatency,
   .of "mem.dramLatency" p.mem.dramLatency, .of "sync.atomAcqRelGpu" p.sync.atomAcqRelGpu,
   .of "sync.redReleaseGpu" p.sync.redReleaseGpu, .of "sync.flagHopPs" p.sync.flagHopPs,
   .of "clockMHz" p.clockMHz, .of "tma.copyDram" p.tma.copyDram]

/-- A kernel: its bytes, multiply-adds, issue count, and its chain as its dependent steps' latencies in cycles. -/
structure Kernel where
  name : String
  bytes : Nat
  fma : Nat := 0
  issue : Option Nat := none
  chain : List Nat

/-- The three floors of a kernel as terms; `fixed` is the ledger's start-up and drain. -/
def kernelFloor (fixed : Nat) (k : Kernel) : Bound where
  spec := s!"floors of {k.name}"
  gpu := p.name
  terms := [
    { resource := (plateau p).name, unit := "bytes"
      demand := .seq (.chain [fixed]) (.rate (plateau p).capacity k.bytes)
      scope := "the ledger's model: weights streamed once at the plateau"
      reason := "stream floor: start-up and drain, then the bytes at the plateau, not overlapping"
      facts := (plateau p).facts },
    (tensor p).term k.fma "dense BF16 products on the tensor cores" "stream floor's tensor side",
    { resource := "busiest sub-partition's issue", unit := "warp instructions"
      demand := .pipe 1 1 (k.issue.getD 0)
      scope := "the kernel's SASS, any schedule"
      reason := "issue floor: one instruction per sub-partition per cycle (Slots.time_ge)" },
    { resource := "dependent chain", unit := "cycles"
      demand := .chain k.chain
      scope := "the kernel's serial dependences, idle latencies"
      reason := "chain floor: each step waits for the previous one (ChainOk.le)"
      facts := chainFacts p }]

/-- The instance's value: `max (max stream issue) chain`. -/
theorem kernelFloor_cycles (fixed : Nat) (k : Kernel) :
    (kernelFloor p fixed k).cycles =
      max (max (max (max 0 (fixed + (plateau p).capacity.minCycles k.bytes))
        ((tensor p).capacity.minCycles k.fma)) (pipeCycles 1 1 (k.issue.getD 0))) k.chain.sum := by
  simp [kernelFloor, Bound.cycles, Demand.cycles, Resource.term]

/-! ## The layer -/

structure Layer where
  /-- Weight bytes before routing, of the MoE experts routed to, and after the MoE. -/
  weightsPre : Nat
  moe : Nat
  post : Nat
  /-- K/V bytes, streamed at the gather ceiling. -/
  kv : Nat
  /-- Pre-route step latencies, the route hop, the MoE's synchronization, the post-MoE step. -/
  preSteps : List Nat
  routeHop : Nat
  moeSync : Nat
  postStep : Nat
  /-- One start-up for the layer. -/
  fixed : Nat

/-- HBM at the paged-gather ceiling. -/
def gather (bytesPerSec : Nat) : Capacity :=
  { instances := 1, rate := Rate.perSecond bytesPerSec p.clockMHz.value }

/-- `F + max(all bytes / BW + KV / BW_gather, max(pre-route bytes / BW + KV / BW_gather,
pre-route steps) + route hop + MoE bytes / BW + MoE sync + post step)`. -/
def layerDemand (gatherBytesPerSec : Nat) (l : Layer) : Demand :=
  let bw := (plateau p).capacity
  let kv := Demand.rate (gather p gatherBytesPerSec) l.kv
  .seq (.chain [l.fixed]) <|
    .both (.seq (.rate bw (l.weightsPre + l.moe + l.post)) kv) <|
      .seq (.both (.seq (.rate bw l.weightsPre) kv) (.chain l.preSteps)) <|
        .seq (.chain [l.routeHop]) (.seq (.rate bw l.moe) (.chain [l.moeSync, l.postStep]))

theorem layerFloor_cycles (g : Nat) (l : Layer) :
    (layerDemand p g l).cycles =
      l.fixed + max ((plateau p).capacity.minCycles (l.weightsPre + l.moe + l.post) +
          (gather p g).minCycles l.kv)
        (max ((plateau p).capacity.minCycles l.weightsPre + (gather p g).minCycles l.kv) l.preSteps.sum +
          (l.routeHop + ((plateau p).capacity.minCycles l.moe + (l.moeSync + l.postStep)))) := by
  simp [layerDemand, Demand.cycles]

end NV.Bounds.Floors
