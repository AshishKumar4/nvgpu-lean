import NV.Arch.H100
import NV.Arch.A100
import NV.Arch.RTXPro6000
import NV.Arch.B200
import NV.Arch.GH200

/-!
# The parts

Every modelled GPU, by the key tools and documents use (`lake exe sim --gpu KEY`,
`docs/facts-KEY.md`). A part is one parameter set: the bounds and the simulator read it.
-/

namespace NV.Arch

open NV.Core

def parts : List (String × Params) :=
  [("h200", h200), ("gh200", gh200), ("h100", h100), ("h100-pcie", h100Pcie), ("h100-nvl", h100Nvl),
   ("a100-40", a100_40), ("a100-80", a100_80), ("rtxpro6000", rtxPro6000), ("b200", b200),
   ("b300", b300)]

/-- The default part for a compiler target when a kernel names only its target (`Kernel.sm`). -/
def forSm : Nat → Option Params
  | 80 => some a100_40
  | 90 => some h200
  | 100 => some b200
  | 103 => some b300
  | 120 => some rtxPro6000
  | _ => none

end NV.Arch
