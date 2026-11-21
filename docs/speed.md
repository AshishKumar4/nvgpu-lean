# The simulator's speed

The `speed` row (`ci/rows/speed.sh`) runs a few whole kernels on every commit and prints each
one's wall time and peak memory, along with the simulated cycles and warp instructions. The
numbers below come from that row on one CI container (AMD EPYC 9684X; default workers).

| kernel | simulated | wall | peak memory | warp instructions per second |
|---|---|---|---|---|
| axpy, 65 536 elements (`examples/axpy`) | — | 2.6 s (0.58 s simulating) | 189 MB | 67 000 |
| TMA + wgmma GEMM 1024³, H100, timed (`gemm_tma_wgmma.json`) | 26 831 cycles, 744 585 warp instructions | 118 s | 697 MB | 6 300 |
| the same GEMM, functional | 786 744 warp instructions | 20 s | 184 MB | 40 000 |
| tcgen05 GEMM, B200, timed (`tcgen05_gemm.json`) | 16 387 cycles, 3 513 warp instructions | 1.1 s | 102 MB | 3 200 |

These are far from the target of 10⁶ warp instructions per second. On the timed GEMM, most of
the time goes to the memory system rather than to executing instructions.

## Where the time goes

The timed GEMM (H100, one worker, 250 s), sampled: its stacks every 0.6 s with gdb, 357
samples over the run (`ci/rows/speed.sh`'s kernel; inclusive shares, so nested frames overlap):

| frame | share |
|---|---|
| the SMs' memory pipelines (`MemSys.advanceSm`, `SmMem.advance`) | 32% |
| of which the shared-memory bank stage (`SmMem.bankStage`) | 22% |
| of which performing shared accesses on the store (`SmMem.performShared`, `Flat.perform`) | 19% |
| the L2's quantum (`L2Mem.run`) | 28% |
| of which replaying the SMs' accesses on the request machine (`L2Mem.replay`, `Flat.deliverAt`) | 16% |
| of which the L2's own requests (`L2Mem.performLanes`, `Flat.deliverLanes`) | 8% |
| warpgroup MMAs (`Wgmma.Share.plan`, `Datapath.dot`) | 11% |
| the async units (`AsyncUnit.progress`, `deliver`, `send`) | 9% |
| decoding and executing instructions (`decodeGeneral`, `Op.run`) | 5% |
| allocation and reference counting (self time) | 40% |

Every shared-memory access is performed twice: on its SM, and again when the L2 replays the
SMs' records on the request machine, which keeps the whole run a sequence of request-machine
steps (`Mem.sound`). Bytes go through `List UInt8` all along, a store's bytes built from its
operand words, written byte by byte, its reads turned back into words (`laneResp`, `toWords`).
So the time is spread over the memory system's per-byte work and its allocations; no single
frame holds a large share. Reaching 10⁶ warp instructions per second needs the shared-memory
path to work on words or byte arrays instead of lists, with the request machine's proofs carried
over, which this does not attempt yet.

Changes so far, each measured over the whole timed GEMM in one container (wall time; one worker,
then the default workers):

- `Flat.deliverLanes`: a part's lanes go to the request machine together (the request found
  once, the requests that may hold one back picked once, the queue rebuilt once), proved equal
  to delivering them one by one (`NV.Memory.SoundLanes.deliverLanes_eq`): 252 s to 249 s, 122 s
  to 116 s. Replaying the SMs' runs of lanes the same way was slower (235 s to 245 s) and was not
  kept.
- `Req.storeSlice`: a store piece's bytes read off its operand words, without building the lane's
  whole byte list once per piece (`Req.storeSlice_eq`): 255 s to 251 s, 139 s to 137 s.
