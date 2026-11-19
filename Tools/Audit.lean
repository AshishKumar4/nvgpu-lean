import NV
import Examples
import Lean

/-!
`lake exe audit`: sorts every theorem under `NV` by what its guarantee rests on.

* **kernel**: only Lean's standard axioms (`propext`, `Classical.choice`, `Quot.sound`).
* **compiler**: also `Lean.ofReduceBool` / `Lean.ofReduceNat`, the axioms behind `native_decide` and
  `bv_decide`'s certificate check: the guarantee trusts Lean's compiler. Allowed only for theorems
  in `NV.CompilerTrusted` and listed by name.

Anything else fails: `sorryAx`, a user `axiom` (a hardware fact must be a hypothesis, not an axiom),
or a compiler-trusted theorem outside its namespace.
-/

open Lean

def kernelAxioms : List Name := [``propext, ``Classical.choice, ``Quot.sound]
-- Names differ across Lean versions (`trustCompiler` replaced the `ofReduce*` axioms), so they are
-- unchecked names.
def compilerAxioms : List Name := [`Lean.ofReduceBool, `Lean.ofReduceNat, `Lean.trustCompiler]

def main : IO UInt32 := do
  initSearchPath (← findSysroot)
  let env ← importModules #[{ module := `NV }, { module := `Examples }] {}
  let mut kernel := 0
  let mut compiler : Array Name := #[]
  let mut bad := 0
  for (name, info) in env.constants.toList do
    unless ((`NV).isPrefixOf name || (`Examples).isPrefixOf name) && !name.isInternal do continue
    let .thmInfo _ := info | continue
    let (axioms, _) ← (collectAxioms name : CoreM (Array Name)).toIO
      { fileName := "audit", fileMap := default } { env }
    let extra := axioms.filter (!kernelAxioms.contains ·)
    if extra.isEmpty then
      kernel := kernel + 1
    else if extra.all compilerAxioms.contains && (`NV.CompilerTrusted).isPrefixOf name then
      compiler := compiler.push name
    else
      bad := bad + 1
      IO.println s!"FAIL {name}: {extra.toList}"
  for n in compiler do IO.println s!"compiler-trusted {n}"
  IO.println s!"{kernel + compiler.size + bad} theorems audited: {kernel} kernel-checked, {compiler.size} compiler-trusted, {bad} failing"
  return if bad == 0 && kernel > 0 then 0 else 1
