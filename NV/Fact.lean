/-!
# Hardware facts

Every hardware number the model uses is a `Fact`: its value, what kind of evidence it rests on,
and where that evidence is. Parameter sets (`NV.Arch.*`) are built from facts, and the facts
tables in `docs/` are generated from these values, so the code and the documentation cannot
disagree.

Units are integers, named in the field that holds the fact: cycles of the SM clock, bytes,
picoseconds for wall time, MHz for clocks, bytes per second for bandwidths. A rate that is not an
integer in its natural unit is stated over a period (`bytesPerKiloCycle`, `opsPer…`) rather than
rounded.
-/

namespace NV

/-- The evidence a fact rests on, in order of authority. -/
inductive Kind where
  /-- NVIDIA documentation: PTX ISA, CUDA Programming Guide, tuning guides, whitepapers,
  datasheets. -/
  | documented
  /-- Our own measurement. -/
  | measured
  /-- A published measurement: a paper or article with its method. -/
  | published
  /-- Observed in our own compiled test kernels (`tests/sass/`). -/
  | encoded
  /-- Another simulator's tuned configuration, used only where nothing above states the fact. -/
  | simulator
  /-- Derived from other facts or by analogy. -/
  | inferred
  deriving DecidableEq, Repr, Inhabited

/-- Where a fact comes from: a key of `docs/sources.md` and the place inside it. -/
structure Source where
  /-- Key in `docs/sources.md`, e.g. `ptx-isa-9.4`, `luo2024`, `huerta2025`. -/
  ref : String
  /-- Section, table, figure or line in the source. -/
  loc : String := ""
  deriving DecidableEq, Repr, Inhabited

structure Fact (α : Type) where
  value : α
  kind : Kind
  source : Source
  /-- The part measured, when it is not the part the fact is used for (e.g. `"H800"` for a
  Hopper fact). -/
  measuredOn : Option String := none
  /-- How a derived or inferred value was obtained, and any disagreement between sources with
  the decision and its reason. -/
  note : String := ""
  deriving Repr, Inhabited

end NV
