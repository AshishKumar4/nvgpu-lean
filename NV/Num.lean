import NV.Num.Format
import NV.Num.Round
import NV.Num.Arith
import NV.Num.F32
import NV.Num.MufuWord
import NV.Num.Convert
import NV.Num.Approx
import NV.Num.Tensor
import NV.Num.Block
import NV.Num.Scaled
import NV.Num.Theorems
import NV.Num.Mufu

/-!
# Numerics

The floating-point and tensor-core arithmetic that the PTX and SASS semantics share: formats and
their encodings (`Format`), rounding (`Round`), IEEE operations with NVIDIA's flush and NaN rules
(`Arith`), every `cvt` conversion (`Convert`), the `.approx` functions (`Approx`) and tensor-core
accumulation (`Tensor`), the special-function unit (`Mufu`) and block-scaled formats (`Block`). Values are bit patterns (`Nat`); all arithmetic is exact on integers.
-/
