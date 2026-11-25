import NV.Bounds.Rate
import NV.Core.Memory

/-!
# A part's resources

The capacities the bounds use, each built from the part's `Params` exactly as the engine builds
its ports (`Memory.Config.ofParams`, the opcode table), with the facts it rests on, so that a
report prints its sources. The theorems here are about the engine's definitions:

* `dram_rate_exact`: the engine's DRAM channels together serve exactly `mem.dramBytesPerSec`;
* `data_port_le`: the engine's L1/shared data stage moves at most `mem.l1BytesPerCycle` bytes per
  cycle per SM;
* the pipes, issue slots and memory ports are `Port`s whose histories are chained
  (`serveAll_chained`), so `Use` and `Slots` describe them.

That a run's use of each resource goes through these ports, and ends within the run's reported
cycles, is what a report assumes of the engine (`Engine.runOn` reserves a pipe's port for its
interval at every issue and issues at most one instruction per sub-partition per cycle;
`MemSys` grants bandwidth only through `Port.take` and `Port.reserve`). It is stated with every
bound, not proven of `runOn`.
-/

namespace NV.Bounds

open NV NV.Core

/-- A fact a bound used, as a report prints it. -/
structure FactRef where
  name : String
  value : String
  kind : Kind
  source : Source
  measuredOn : Option String := none
  deriving Repr, Inhabited

def FactRef.of {α : Type} [ToString α] (name : String) (f : Fact α) : FactRef :=
  { name, value := toString f.value, kind := f.kind, source := f.source, measuredOn := f.measuredOn }

def kindName : Kind → String
  | .documented => "documented" | .measured => "measured" | .published => "published"
  | .encoded => "encoded" | .simulator => "simulator" | .inferred => "inferred"

def FactRef.render (f : FactRef) : String :=
  let on := match f.measuredOn with | some p => s!", measured on {p}" | none => ""
  let loc := if f.source.loc.isEmpty then "" else s!" {f.source.loc}"
  s!"{f.name} = {f.value} ({kindName f.kind}{on}; {f.source.ref}{loc})"

/-- A resource of a part as a bound uses it: what it moves, its capacity, and the facts and
modelling the capacity rests on. -/
structure Resource where
  name : String
  /-- What it moves: bytes, multiply-adds, warp instructions. -/
  unit : String
  capacity : Capacity
  facts : List FactRef
  note : String := ""
  deriving Inhabited

/-- A pipe-like resource: `instances` copies, each taking one request per `interval` cycles. -/
structure PipeResource where
  name : String
  unit : String
  instances : Nat
  interval : Nat
  facts : List FactRef
  note : String := ""
  deriving Inhabited

namespace Part

variable (p : Params)

def channels : Nat := max p.mem.dramChannels.value 1

def slices : Nat := max p.mem.l2Slices.value 1

def clock : FactRef := .of "clockMHz" p.clockMHz

def smsFact : FactRef := .of "sms" p.sms

/-- DRAM: the engine's channels, each at `Rate.perSecond dramBytesPerSec (clock · channels)`. -/
def dram : Resource where
  name := "DRAM"
  unit := "bytes"
  capacity := { instances := channels p, rate := (Memory.Config.ofParams p 0).channel }
  facts := [.of "mem.dramBytesPerSec" p.mem.dramBytesPerSec, .of "mem.dramChannels" p.mem.dramChannels,
    clock p]
  note := ""

/-- L2 slices, `l2SliceBytesPerCycle` each. -/
def l2 : Resource where
  name := "L2 slices"
  unit := "bytes"
  capacity := { instances := slices p, rate := (Memory.Config.ofParams p 0).slice }
  facts := [.of "mem.l2SliceBytesPerCycle" p.mem.l2SliceBytesPerCycle, .of "mem.l2Slices" p.mem.l2Slices]
  note := ""

/-- The SMs' crossbar ports from the L2 (`Res.xin`), `xbarBytesPerCycle` each. -/
def smIn : Resource where
  name := "SM inbound (crossbar)"
  unit := "bytes"
  capacity := { instances := p.sms.value, rate := (Memory.Config.ofParams p 0).xbar }
  facts := [.of "mem.xbarBytesPerCycle" p.mem.xbarBytesPerCycle, smsFact p]
  note := ""

/-- The GPCs' links from the L2 (`L2Res.gpc`), `gpcLinkBytesPerCycle` each: every byte the L2
answers to an SM passes its GPC's. -/
def gpcIn : Resource where
  name := "GPC inbound (link from the L2)"
  unit := "bytes"
  capacity := { instances := max p.gpcSMs.value.length 1, rate := (Memory.Config.ofParams p 0).gpcLink }
  facts := [.of "mem.gpcLinkBytesPerCycle" p.mem.gpcLinkBytesPerCycle, .of "gpcSMs" p.gpcSMs, clock p]
  note := ""

/-- The SMs' L1/shared data stages: one wavefront of `banks · bankBytes` per `wavefrontCycles`. -/
def l1 : Resource :=
  let c := Memory.Config.ofParams p 0
  { name := "L1/shared data"
    unit := "bytes"
    capacity := { instances := p.sms.value, rate := { units := c.banks * c.bankBytes, cycles := c.wavefrontCycles } }
    facts := [.of "mem.l1BytesPerCycle" p.mem.l1BytesPerCycle, .of "mem.smemBanks" p.mem.smemBanks,
      .of "mem.smemBankBytes" p.mem.smemBankBytes, smsFact p]
    note := "" }

/-- Multiply-adds per cycle per SM over every unit that forms them: the tensor cores at the
input type's dense rate, plus the FP32 pipe (32 lanes per sub-partition per FFMA interval) and
the FP16 pipe (two per lane per HFMA2 interval). A program may use all at once, so the sum
bounds any program's multiply-add rate. -/
def macsPerCycle (tensorPerSM : Nat) : Rate :=
  let interval (op : String) : Nat :=
    ((p.ops.find? (·.opcode == op)).map (·.timing.value.interval)).getD 1
  let smsps := p.smsps.value
  -- `smsps · 32 / interval` FFMA lanes and `smsps · 64 / interval` HFMA2 lanes per cycle, over a
  -- common period.
  let f := max (interval "FFMA") 1
  let h := max (interval "HFMA2") 1
  { units := tensorPerSM * f * h + smsps * 32 * h + smsps * 64 * f, cycles := f * h }

def macs (name : String) (tensor : Fact Nat) : Resource where
  name := s!"multiply-adds ({name})"
  unit := "multiply-adds"
  capacity := { instances := p.sms.value, rate := macsPerCycle p tensor.value }
  facts := [.of s!"tensor.fmaPerCycle ({name})" tensor, smsFact p, .of "smsps" p.smsps]
  note := ""

/-- Bulk-copy (TMA) issue, `tma.issueBytesPerSec` per SM. Only kernels that use it are bound by
it: a spec can be computed without it. -/
def tma : Resource where
  name := "TMA issue"
  unit := "bytes"
  capacity := { instances := p.sms.value, rate := Rate.perSecond p.tma.issueBytesPerSec.value p.clockMHz.value }
  facts := [.of "tma.issueBytesPerSec" p.tma.issueBytesPerSec, clock p, smsFact p]
  note := ""

/-- One instruction per sub-partition per cycle. -/
def issue : PipeResource where
  name := "issue"
  unit := "warp instructions"
  instances := p.sms.value * p.smsps.value
  interval := 1
  facts := [smsFact p, .of "smsps" p.smsps]
  note := ""

/-- The pipe an opcode runs on, with its interval, from the part's opcode table. -/
def pipeOf (op : String) : PipeResource :=
  let rule := p.ops.find? (·.opcode == op)
  { name := s!"{op} pipe"
    unit := s!"{op} warp instructions"
    instances := p.sms.value * p.smsps.value
    interval := max ((rule.map (·.timing.value.interval)).getD 1) 1
    facts := [smsFact p, .of "smsps" p.smsps] ++
      ((rule.map fun r => FactRef.of s!"ops.{op}.interval"
        { value := r.timing.value.interval, kind := r.timing.kind, source := r.timing.source,
          measuredOn := r.timing.measuredOn, note := r.timing.note }).toList) }

end Part

/-! ## The engine's rates -/

/-- The engine's DRAM channels together serve exactly the part's documented bandwidth: per
channel `units / cycles = dramBytesPerSec / (clock · channels)`. -/
theorem dram_rate_exact (p : Params) (d : Nat) :
    (Memory.Config.ofParams p d).channel.units * (p.clockMHz.value * max p.mem.dramChannels.value 1 * 1000000) =
      p.mem.dramBytesPerSec.value * (Memory.Config.ofParams p d).channel.cycles :=
  perSecond_exact _ _

/-- The engine's L1/shared data stage moves a wavefront of `banks · bankBytes` bytes per
`wavefrontCycles`, at most `l1BytesPerCycle` per cycle. -/
theorem data_port_le (p : Params) (d : Nat) (h : 0 < p.mem.l1BytesPerCycle.value) :
    (Memory.Config.ofParams p d).banks * (Memory.Config.ofParams p d).bankBytes ≤
      (Memory.Config.ofParams p d).wavefrontCycles * p.mem.l1BytesPerCycle.value := by
  show p.mem.smemBanks.value * p.mem.smemBankBytes.value ≤
    max 1 ((p.mem.smemBanks.value * p.mem.smemBankBytes.value + p.mem.l1BytesPerCycle.value - 1) /
      max p.mem.l1BytesPerCycle.value 1) * p.mem.l1BytesPerCycle.value
  generalize p.mem.smemBanks.value * p.mem.smemBankBytes.value = w
  generalize p.mem.l1BytesPerCycle.value = r at h ⊢
  have hr : max r 1 = r := by omega
  rw [hr]
  have h₁ := le_mul_of_cdiv_le h (Nat.le_refl (cdiv w r))
  unfold cdiv at h₁
  have h₂ : r * ((w + r - 1) / r) ≤ r * max 1 ((w + r - 1) / r) :=
    Nat.mul_le_mul_left r (Nat.le_max_right _ _)
  rw [Nat.mul_comm (max 1 _) r]
  omega

end NV.Bounds
