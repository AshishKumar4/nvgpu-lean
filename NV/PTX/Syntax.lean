import NV.PTX.Syntax.Parse
import NV.PTX.Syntax.Instr
import NV.PTX.Syntax.Check

/-!
# PTX syntax

The PTX front end (PTX ISA 9.4): `parse` turns PTX text into a `Module RawInstr`, and
`Module.decode` into a `Module Instr` of typed instructions, and `check` reports where a
module breaks the ISA's static rules.
-/

namespace NV.PTX

/-- Parse and decode a PTX module. -/
def read (src : String) : Except String (Module Instr) := do
  (← parse src).decode

end NV.PTX
