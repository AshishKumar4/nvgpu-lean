import NV.Formal.Outcomes
import NV.PTX.Memory.DRF

/-!
# DRF-SC for programs

`drf_sc_outcomes`: when every consistent candidate execution of a program is race-free
(`Execution.RaceFree`: conflicting accesses ordered by proxy-preserved causality order), every
outcome of the program is the outcome of a sequentially consistent candidate, one whose memory
events have a total order extending program order in which each read reads the latest write
(`Execution.IsSC.reads_latest`). A user proves race freedom from the synchronization protocol and
the result over interleavings.
-/

namespace NV.Formal

variable {σ : Type} (P : Program σ)

/-- Outcomes of the sequentially consistent candidates. -/
def SCOutcomes (o : Outcome) : Prop :=
  ∃ c : Candidate P, c.Valid ∧ c.exec.consistent = true ∧ c.exec.IsSC ∧ c.Yields o

theorem drf_sc_outcomes
    (hrace : ∀ c : Candidate P, c.Valid → c.exec.consistent = true → c.exec.RaceFree)
    {o : Outcome} (h : Outcomes P o) : SCOutcomes P o := by
  obtain ⟨c, hv, hc, hy⟩ := h
  exact ⟨c, hv, hc, PTX.Execution.drf_sc hc (hrace c hv hc), hy⟩

theorem scOutcomes_sub {o : Outcome} (h : SCOutcomes P o) : Outcomes P o := by
  obtain ⟨c, hv, hc, _, hy⟩ := h
  exact ⟨c, hv, hc, hy⟩

end NV.Formal
