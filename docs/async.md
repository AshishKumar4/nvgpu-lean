# Asynchronous operations: TMA, bulk copies, `cp.async`, mbarriers, wgmma (`sm_90a`)

The async layer: what it covers, how each piece is checked, and what it does not cover yet.

## Vocabulary (`NV.Machine`, `NV.Async`)

- **mbarrier objects are words.** `MemOp.mbar` read-modify-writes an 8-byte shared word
  (`NV.Async.MBarrierWord`: pending count in bits 0–19, expected count 20–39, tx-count 40–60 as
  21-bit two's complement, phase parity in bit 63, where the hardware's arrive token keeps it too).
  `MBarOp` is `init`, `arrive count tx drop` (expect-tx then arrive-on; `count = 0` is
  `expect_tx` alone), `completeTx`, `inval`; each is `NV.PTX.MBarrier`'s step on the decoded word,
  and behaviour the ISA leaves undefined (over-arrival, tx-count out of range) stops the run.
  Arrivals are release RMWs, `test_wait`/`try_wait` acquire loads of the word whose parity test
  is the issuer's, complete-tx a release RMW at cluster scope in the generic proxy
  (§9.7.10.28.5.3), so mbarrier synchronization is the memory model's (the memory model's events).
- **`Effect.async (op : σ → Op)`** hands an operation to an agent; like `Effect.mem` it is built
  from the issuer's sources when the instruction reads them. `Op` is a `Copy` (moves: a load and
  the stores of its words, several for a multicast; then signals), a `TensorCopy` (the map is
  loaded first), an `Mma` (shared loads and a pure result), a prefetch, or a `fault`.
  `Copy.program` is the copy as the memory model's thread: moves, the implicit generic-async
  proxy fence of completion, signals. `Op.run` is the blocking reading.
- **Groups.** `SyncOp.asyncCommit g` and `asyncWait g n` for `cp.async`, bulk and wgmma groups.

## Copies (`NV.Async.Tma`)

`TensorMap.tile m coords smem` turns a tiled box into 16-byte pieces (single elements where a
chunk straddles the tensor's edge): global address or `none` outside the tensor (§5.5.3.3: loads
store zeros, stores skip), and the swizzled shared address. `Swizzle.apply` is §5.5.7 as a
function of the shared address (16-byte chunks XORed with the 128-byte line's index in the
pattern; the 32- and 64-byte atomicity modes; the 8-byte flip), which is also wgmma's
`Swizzle<B,4,3>`. Pieces are packed into requests of one global line each.

## Warpgroup MMA (`NV.Async.Wgmma`, `WgmmaNum`)

Matrix descriptors (§9.7.17.5.1.2.2) and the canonical layouts of §9.7.17.5.1.2.1.3 give each
operand element's shared address; the register fragments of A and D are `mma.sync`'s stacked over
the warpgroup (CUTLASS `ALayout_64x*`, `CLayout_64xN`), so a warp's 16 rows of D depend only on
its rows of A and on B: each warp (SASS) or thread (PTX) computes its share. Values come from
`NV.Num.TensorModel.tile` (Khattak and Mikaitis's H100/H200 datapath, one fused block per
instruction) or `dotInt`. Consecutive MMAs of a warp chain through the accumulator: an MMA
issued before its predecessor completed starts from the predecessor's result.

## Semantics

- **SASS** (`NV.SASS.Async.execAsync`), the forms:
  | PTX | SASS |
  |---|---|
  | `mbarrier.init` | `SYNCS.EXCH.64 URZ, [a], URv`, `URv = (2²⁰−n)<<1 \| (2²⁰−n)<<43` |
  | `mbarrier.arrive{.expect_tx}`, `expect_tx`, `complete_tx`, `arrive_drop` | `SYNCS.ARRIVE.TRANS64[.RED][.A1T0\|.A0T1\|.A0TR\|.A0TX][.OPTOUT]` |
  | `mbarrier.test_wait/try_wait{.parity}` | `SYNCS.PHASECHK.TRANS64[.TRYWAIT] Pd, [a], Rp` (`Rp` bit 31: the parity, or a state's phase) |
  | `mbarrier.inval` | `SYNCS.CCTL.IV` |
  | `cp.async` (+ `src-size`/`ignore-src`) | `LDGSTS[.ZFILL] …[, P]` (`ZFILL`: the source offset within the size is the bytes not read) |
  | `cp.async.commit_group`, `wait_group n` | `LDGDEPBAR` (write counter SB0), `DEPBAR.LE SB0, n` |
  | `cp.async.mbarrier.arrive[.noinc]` | `[SYNCS.ARRIVE.TRANS64.RED.A0T1;] ARRIVES.LDGSTSBAR.64.TRANSCNT\|ARVCNT` (per thread) |
  | `cp.async.bulk.tensor` load / store | `UTMALDG.nD [dst, mbar, coords…], [map]` / `UTMASTG.nD [src, coords…], [map]` |
  | `cp.async.bulk` load / store | `UBLKCP.S.G` / `UBLKCP.G.S` (size in 16-byte units) |
  | `cp.async.bulk.commit_group`, `wait_group` | `UTMACMDFLUSH` (read counter SB0), `DEPBAR.LE SB0, n` |
  | `wgmma.mma_async`, `commit_group`, `wait_group n`, `fence` | `HGMMA`/`QGMMA`/`IGMMA` (`gsb0` commits), `WARPGROUP.DEPBAR.LE gsb0, n`, `WARPGROUP.ARRIVE` |

  A `.param` tensor map's address is the parameter window's base (`c[0x0][0x198]`) plus its
  offset. Uniform registers and predicates are read at issue, vector registers when the read
  counter releases.
- **PTX** (`NV.PTX.Async.exec?`, called first by `Thread.step`): the same operations at thread
  level; the interpreter answers `Effect.async` with `Op.run`.

## Timing (`NV.Core.Async`, the engine)

Agents move data only through the memory system (`MemSys.issueAsync`; `LDGSTS`'s loads through
the LSU like the instruction). Tensor and bulk loads are generated at the SM's TMA port rate
(`TmaParams.issueBytesPerSec`, `Port.take`); an MMA holds its sub-partition's GMMA port for
`macs / (fmaPerCycle / smsps)` cycles and completes `wgmmaLatency(N)` (`wgmmaLatencyRS` with A in
registers) after it starts, not before its shared loads and its predecessor. A commit's group
releases the committing instruction's counter when its last operation completes;
`WARPGROUP.DEPBAR` is a readiness condition. MMA results are computed on `Task`s as soon as their
operands are in and forced only when written.

## Checks

- `lake exe async-tests` (row `async`): swizzles against §5.5.7's tables; descriptors against
  §9.7.17.5.1.2's worked examples (CuTe layouts evaluated independently); fragments against
  CUTLASS's layouts; mbarrier phases, tx-counts and the init value; a TMA box over the
  tensor's edges against §5.5.3's address arithmetic.
- `lake exe async-copies` (row `async-kernels`): `tests/kernels/async_copies.cu` through `nvcc`
  and the engine, every byte checked: `cp.async` with zero fill and both completions, bulk copies
  both ways, TMA loads over the edges and TMA stores.
- `lake exe async-gemm` (row `async-kernels`): `tests/kernels/gemm_tma_wgmma.cu`, a BF16 GEMM (TMA 128-byte swizzle, four-stage full/empty mbarrier ring, `wgmma m64n128k16`):
  integer-valued inputs against integer arithmetic, random BF16 against the tensor-core model,
  bit for bit.
- `lake exe async-ptx` (row `async-ptx`): the same kernels' PTX through the interpreter, same
  oracles.

## Not modelled yet

TMA multicast and `.cta_group::2`, im2col and gather/scatter modes, interleaved maps, the OOB-NaN
fill, packed sub-byte types, reductions (`cp.reduce.async.bulk*`, `UTMASTG.RED`),
`tensormap.replace`, prefetch traffic; a wgmma descriptor with a non-zero base offset, sparse
wgmma, A in registers at PTX thread level; `cp.async.mbarrier.arrive` without `.noinc` in PTX;
`try_wait`'s suspension (it answers at once, so waits spin); the tensor map's descriptor fetch
latency (no source); tensor-map proxy semantics on map loads.
