import NV.PTX.Sem.Step

/-!
# PTX semantics

Per-thread semantics of PTX ISA 9.4: a prepared program (`Prog.ofModule`), thread
state (`Thread`), and `Thread.step`, which yields the `Effect` of the next instruction.
-/
