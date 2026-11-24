import NV.Arch.H200

/-!
# GH200 480 GB (`sm_90a`)

The Hopper GPU of a Grace Hopper superchip: GH100 with 132 SMs and HBM3. The SM and memory system
are the H200's (`NV.Arch.H200`), which the GH200's own measurements largely match; where the
GH200 measured differently its value is its own: its HBM3 (a 6144-bit bus at 2619 MHz, 4023 GB/s
by the device's own report, 3605 GB/s read by measurements/gh200/out/dram.txt) is not the H200's
HBM3e.
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
      codeCaches := measOn [(24576, 10), (49152, 18), (147456, 16)] "" "" }
    mem := { h200.mem with
      dramChannels := measOn 96 "" "the 6144-bit bus in 64-bit channels, as the H200's 6016 in 94"
      dramBusBits := measOn 6144 "" ""
      dramClockMHz := measOn 2619 "" ""
      dramBytesPerSec := measOn 4023000000000 "" ""
      dramStreamBytesPerSec := measOn 3604900000000 "" "" } }

end NV.Arch
