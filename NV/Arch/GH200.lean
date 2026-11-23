import NV.Arch.H200

/-!
# GH200 480 GB (`sm_90a`)

The Hopper GPU of a Grace Hopper superchip: GH100 with 132 SMs and HBM3. The SM and memory system
are the H200's (`NV.Arch.H200`), which the GH200's own measurements largely match; where the
GH200 measured differently its value is its own.
-/

namespace NV.Arch

open NV NV.Core

def gh200 : Params :=
  { h200 with
    name := "GH200 480GB"
    -- The second code level refills from the third more slowly: straight-line code from there
    -- runs at 2.21 cycles an instruction against the H200's 2.04, the same listing on both
    -- (icache.cu's 64 KB pass, 9040 cycles against 8352).
    frontEnd := { h200.frontEnd with
      codeCaches := measOn [(24576, 10), (49152, 18), (147456, 16)] "" "" } }

end NV.Arch
