import Lake
open Lake DSL

package nvgpu where
  leanOptions := #[⟨`autoImplicit, false⟩]

@[default_target]
lean_lib NV

@[default_target]
lean_lib Examples

/-- The emulated device's C ABI (`tools/cudadev`): a shared library the driver shim loads. -/
lean_lib Device where
  roots := #[`Device.Api, `Device.Cache]
  defaultFacets := #[LeanLib.sharedFacet]

lean_exe audit where
  root := `Tools.Audit
  supportInterpreter := true

lean_exe litmus where
  root := `Tools.Litmus

lean_exe «litmus-tests» where
  root := `Tools.LitmusTests

lean_exe facts where
  root := `Tools.Facts

lean_exe «engine-smoke» where
  root := `Tests.EngineSmoke

lean_exe «memory-tests» where
  root := `Tests.Memory

lean_exe «sass-tests» where
  root := `Tests.SassReader

/-- Helpers shared by test executables. -/
lean_lib TestSupport where
  roots := #[`Tests.Kernels]

lean_exe «sass-sem» where
  root := `Tests.SassSem

lean_exe «sass-diff» where
  root := `Tests.SassDiff

lean_exe «sched-tests» where
  root := `Tests.Sched

lean_exe ptx where
  root := `Tools.Ptx

lean_exe axpy where
  root := `Tests.Axpy

lean_exe «ptx-tests» where
  root := `Tools.PtxTests

lean_exe «num-tests» where
  root := `Tools.NumTests

lean_exe «ptx-oracle» where
  root := `Tools.PtxOracle

lean_exe «sass-coverage» where
  root := `Tools.SassCoverage

lean_exe «ptx-coverage» where
  root := `Tools.PtxCoverage

lean_exe «spec-ptx» where
  root := `Tools.SpecPtx

lean_exe sim where
  root := `Tools.Sim

lean_exe «trace-check» where
  root := `Tools.TraceCheck

lean_exe «async-tests» where
  root := `Tests.Async

lean_exe «async-ptx» where
  root := `Tests.AsyncPtx

lean_exe «async-copies» where
  root := `Tests.AsyncCopies

lean_exe «async-gemm» where
  root := `Tests.AsyncGemm

lean_exe «sass-footprint» where
  root := `Tests.SassFootprint

lean_exe gap where
  root := `Tools.Gap

lean_exe «sass-printf» where
  root := `Tests.SassPrintf

lean_exe «sass-dump» where
  root := `Tools.SassDump

lean_exe «sass-conditional» where
  root := `Tests.SassConditional

lean_exe «sass-ctrl» where
  root := `Tools.SassCtrl
