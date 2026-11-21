# The simulator's speed

The `speed` row (`ci/rows/speed.sh`) runs a few whole kernels on every commit and prints each
one's wall time and peak memory, along with the simulated cycles and warp instructions. The
numbers below come from that row on one CI container (AMD EPYC 9684X; default workers).

| kernel | simulated | wall | peak memory | warp instructions per second |
|---|---|---|---|---|
| axpy, 65 536 elements (`examples/axpy`) | 26 831 cycles, 744 585 warp instructions | 2.6 s (0.58 s simulating) | 189 MB | 67 000 |
| TMA + wgmma GEMM 1024³, H100, timed (`gemm_tma_wgmma.json`) | 16 446 cycles, 786 744 warp instructions | 118 s | 697 MB | 6 700 |
| the same GEMM, functional | — | 20 s | 184 MB | — |
| tcgen05 GEMM, B200, timed (`tcgen05_gemm.json`) | — | 1.1 s | 102 MB | — |

These are far from the target of 10⁶ warp instructions per second. On the timed GEMM, most of
the time goes to the memory system rather than to executing instructions.

## Where the time goes

Measured with callgrind on the timed GEMM (H100, one worker), over its first 6 000 of 16 446
cycles (`--until 6000`) and over cycles 6 000 to 7 000 (the difference of runs to 6 000 and to
7 000). Percentages are inclusive instructions, so nested frames overlap. On one worker the
first 6 000 cycles take 7.6 s and the first 7 000 18 s of the run's 249 s, so the time is in the
later cycles, and from cycle 6 000 on it goes mostly to shared-memory accesses (38 G instructions
for the first 6 000 cycles, 79 G for the next 1 000).

Cycles 6 000 to 7 000:

- Performing shared-memory accesses on the store (`Flat.perform`): 44%. A lane's store is written
  byte by byte (`performWrites`, `Store.writeByte`), its bytes built as a list from its operand
  words (`Req.storeBytes`, `toBytes`).
- The L2 replaying the SMs' shared-memory accesses on the request machine (`L2Mem.replay`,
  `Flat.deliver`, one lane at a time): 32%. This is what keeps the whole run a sequence of
  request-machine steps (`Mem.sound`). Delivering a run of consecutive lanes of one request
  together (`deliverLanes`) was measured 4% slower over the whole GEMM, so the replay stays per lane.
- The SMs' memory pipelines (`SmMem.advance`, bank stage): 34%, mostly the same shared performs.

Cycles 0 to 6 000:

- The L2's request machine (`L2Mem.performLanes`, `Flat.deliver`): 36%. Before `Flat.deliverLanes`,
  each lane of a request part found its request again, scanned every earlier request of its
  issuer for one that holds it back (`blocker`, `overlap`), and rebuilt the issuer's queue
  (`retire`). Now a part's lanes go to the request machine together (25%, and 14% fewer
  instructions over the first 6 000 cycles: 44.4 G to 38.2 G; over the whole GEMM the wall time
  drops from 252 s to 249 s on one worker and from 122 s to 116 s on the default workers, measured
  in the same container). `NV.Memory.SoundLanes` proves `deliverLanes` equal to delivering the
  lanes one after another, so `Mem.sound` still covers the run. What remains is the scan for
  earlier requests that may hold a lane back (`mayBlock`, 12%), which runs over every earlier
  request of the issuer in flight.
- The async units: delivering copies' data (`AsyncUnit.deliver`, 18%) and starting operations
  (`startAll`, 12%). A TMA copy's box is computed element by element with `Int` addresses
  (`TensorMap.tile`), and addresses above 2³¹ are big integers.
- The SMs' memory pipelines (`MemSys.advanceSm`, 17%).
- Allocation and reference counting: about 20% of self time (`lean_copy_expand_array`,
  `lean_del_core_other`, `mi_free`). Part of it is arrays and hash maps copied because they are
  shared when updated, for example `SmMem.countSectors`'s pending-sector maps and `SmMem.modReq`.
