# SASS: reading, executing and checking compiled code (`sm_80`, `sm_90a`, `sm_100a`, `sm_120a`)

The `NV.SASS` layer. What it covers, how each piece is checked, and what it
does not cover yet.

## Toolchain

`tools/cuda/install.sh DEST` installs, from sha256-pinned NVIDIA PyPI wheels, `ptxas`/`nvcc`/
`cicc` and the CUDA headers 13.0.88, CCCL 13.0.85,
and `cuobjdump`/`nvdisasm` 13.2.51. No GPU or driver.

## Reading (`Read`, `Elf`, `Syntax`, `Opcodes`)

- `Read.kernels sass elf` reads a `cuobjdump -sass` or `nvdisasm -hex -c` listing plus the same
  cubin's `cuobjdump -elf` text into `Kernel`s: instructions with guard, opcode, modifiers,
  operands and control word; registers, parameter layout, static shared memory (without the
  1 KB system window, `Elf.reservedShared`), stack size, cluster shape, and constant banks other
  than 0 (jump tables).
- Operands are taken as printed, except float immediates (printed in decimal, read from encoding
  bits 32–63 and checked against the text) and branch targets (offsets or labels).
- The control word comes from bits 41–61 of the high word (`Ctrl.ofHi`).
- `Opcodes.opcodes`: NVIDIA's Hopper instruction table (CUDA Binary Utilities 13.5 §4.3).

Checks (`lake exe sass-tests`, CI row `sass`): every listing in `tests/sass/` (PTX probes and a
CUDA corpus; `tests/sass/gen.sh --check` proves they are the pinned tools' output) reads the
same from `cuobjdump` and `nvdisasm`; decoded reuse flags match the printed `.reuse` operands;
every waited counter is set somewhere; every opcode is in the catalogue; hand-read goldens; and
the H200 table's fixed latencies against the fixtures' spacing (`Tests.latencyProblems`).

## Module data and `printf`

`Elf.moduleData` places a cubin's device data (`.nv.global.init`, `.nv.global`) at addresses of
our choosing (`Elf.dataBase`), gives external functions (`vprintf`) addresses in a reserved range,
and fills the module's constant banks (3: `__constant__` data, 4: addresses), applying
`.rela.nv.constantN` (`R_CUDA_64`). `Kernel.data` is written into global memory at launch.

Device `printf` is `vprintf(fmt, buf)`, reached by `LEPC` then `CALL.ABS.NOINC` to `vprintf`'s
address: per executing lane, the argument buffer (`R6:R7`, the lane's stack through the local
window, `localWindow`) is loaded, and the writeback appends `(format address, argument bytes)`
to the warp's `printf` records and returns the argument count in `R4`. The records are host
output, not memory; `NV.Printf.render` formats them as C's `printf` would (`printfText`). Calls to
other externals fault with the symbol's name; argument buffers above 16 bytes fault (not yet
modelled). Checked by `lake exe sass-printf`.

## Executing (`Exec`, `Sem`, `Fast`, `Float`)

Per-warp semantics, yielding the shared `Effect`s:

- **Writeback timing.** Register results are `Effect.reg`: the state after issue and the writes,
  applied at the result's writeback. The hardware does not check dependences on fixed-latency
  results (Huerta et al. 2025 §4), so code with too small a stall or a missing wait reads the old
  register, as on silicon. Memory requests are built when the instruction reads its sources
  (`Effect.mem`), after issue.
- **Divergence.** Each lane has a program counter; the warp runs the lowest one among lanes not
  waiting. `BSSY`/`BSYNC`/`BREAK`/`BMOV` and `WARPSYNC` are convergence barriers on lane masks;
  `BRA.DIV` and `BRA.U.ANY` are taken by the warp's live lanes. Converged warps keep one program counter.
- **Uniform datapath.** `U`-prefixed opcodes share their vector form's semantics and execute once
  for the warp.
- **Numerics** are `NV.Num`'s (part A): float arithmetic and comparison in every rounding, `FTZ`,
  `SAT`; conversions; `MUFU` on the bit-accurate unit model; tensor-core accumulation for
  `HMMA`/`IMMA`/`DMMA` through the PTX part's fragment model.
- **Decoding once.** `decode k` resolves every instruction's form when the kernel is loaded.
  The hot forms (`Fast`) are compiled into loops over lane masks; the general definitions in
  `Sem` are the specification and a test compares both on every listed instruction.

Opcodes with semantics (others fault with "no semantics yet"; the asynchronous ones — TMA,
`SYNCS`, `LDGSTS`, `HGMMA`, cluster barriers — are `Async`'s, part H):

| group | opcodes |
|---|---|
| integer | `IMAD` (`.WIDE`, `.HI`, `.X`, carries), `IADD3` (`.X`, carries, `.64`), `VIADD`, `LEA` (`.HI` with a carry out, `.X`, `.SX32`), `SHF`, `LOP3`, `PLOP3`, `ISETP` (`.EX`), `SEL` (`.64`), `PRMT`, `POPC`, `FLO`, `BREV`, `IABS`, `SGXT`, `BMSK`, `IMNMX` (`.S64`/`.U64` pairs, sm_100 on)/`VIMNMX`/`VIMNMX3`/`VIADDMNMX`, `IDP`, `VABSDIFF`, `VABSDIFF4`, `I2I` |
| moves, special registers | `MOV` (`.64`), `S2R`, `S2UR`, `CS2R`, `R2UR` (with a predicate of the lanes whose value differs), `P2R`, `R2P`, `LDC` (any bank 0 form, bank 2 jump tables), `USETMAXREG` |
| warp | `SHFL`, `VOTE`, `VOTEU`, `MATCH`, `REDUX`, `ELECT` |
| float | `FADD`, `FMUL`, `FFMA`, `FMNMX`, `FSEL`, `FSETP`, `FCHK`, `FRND`, `DADD`, `DMUL`, `DFMA`, `DSETP`, `HADD2`, `HMUL2`, `HFMA2`, `HMNMX2`, `HSETP2`, `MUFU`, `I2F`, `I2FP`, `F2I`, `F2F`, `F2FP` |
| matrix | `HMMA`, `IMMA`, `DMMA`, `LDSM`, `STSM` |
| memory | `LDG`, `STG`, `LDS`, `STS`, `LDL`, `STL`, `ATOMG`, `ATOMS` (incl. `CAST.SPIN`), `ATOM`, `REDG`, `MEMBAR`, `FENCE`, `ERRBAR`, `CGAERRBAR`, `CCTL` |
| control | `BRA`, `BRX`, `EXIT`, `BSSY`, `BSYNC`, `BREAK`, `BMOV`, `WARPSYNC`, `CALL` (relative, through a register; absolute to `vprintf`), `LEPC`, `RET`, `BAR` (`.RED` with `B2R.RESULT`), `NOP`, `DEPBAR`, `NANOSLEEP`, `YIELD`, `BPT` |

Some forms rest on an inference, said in place: `FCHK`'s condition, `BRA.DIV`'s, `ATOMS.CAST.SPIN`
returning 1 when it stored, `VABSDIFF4` ignoring its third source, the mma register order, `R2UR`'s predicate, `IMNMX.S64`'s
min/max select. Each is
covered by a test that runs compiled code using it against the PTX semantics.

Checks (CI row `sass`):

- `lake exe sass-sem`: compiled kernels (`tests/sem/`: integer arithmetic over edge values,
  divergence and reconvergence, early exits, out-of-line calls, warp collectives on full and
  partial warps, atomics, jump tables) run on the H200 parameters against references written
  from the PTX ISA; hand-assembled kernels where correct control bits give the right result and a
  shortened stall or dropped wait reads the stale register; and the compiled forms against the
  general definitions.
- `lake exe sass-diff`: the PTX↔SASS differential test. Each `tests/diff/` kernel runs in the PTX
  interpreter and, compiled, in the SASS simulator, on edge and random inputs; every
  output word must agree. Covered: binary32 arithmetic in all roundings, FTZ and saturation,
  comparisons, division and square root, every `.approx` function, conversions; packed
  binary16/bfloat16; binary64; `mma.sync` (bf16, f16, tf32); `ldmatrix`/`stmatrix`; float
  atomics on global and shared memory; rounding to integral, packed compares, dot products,
  saturating narrowing; CTA barrier reductions and warp reductions; 64-bit min/max. One known difference: `div.full.f32`, which the PTX model rounds
  correctly and the SASS model computes as `a · MUFU.RCP(b)` with range scaling.

## Targets

`Kernel.sm` comes from the ELF header (`sm=80`). The control word has the same layout on every
target (`lake exe sass-ctrl DIR…` checks a listing's counters and lists each producer's least
encoded distance). Per-target fixtures are in
`tests/sass/sm_80/` and `tests/sass/sm_120a/` (`gen.sh`; `SKIPPED` lists sources a target cannot
compile). The semantics read per target: the tensor-core model (`NV.PTX.Sem.Matrix.model`), the
local window's place in constant bank 0 (`Exec.localWindowAt`), and the instruction forms of each
target's compiled code — sm_80's scaled shared addresses (`[R.X4]`), negated constants
(`-c[…]`), default `F2FP` formats; sm_120a's `LDCU`, `CS2UR`, `IADD.64`, uniform float forms
(`UFMUL`, `UFSEL`, `UFSETP`, `UI2F`), packed `VIMNMX`, `QMMA` (native FP8 MMA), `BRXU`,
`ACQBULK`/`PREEXIT` (grid dependency, nothing for one launch), `ENDCOLLECTIVE`, generic `LD`/`ST`
and `ATOMS.CAST.SPIN` with a predicate result, 64-bit `MOV`/`SEL`/`IMNMX`, and uniform registers
past `UR63` (sm_100 on encode `URZ` as `0xff`, not `0x3f`; cuda-samples use up to `UR79` on
sm_120a), so the model's uniform file has 255 registers and `URZ` is `UR255` on every target. The CI rows `sass-sm80` and `sass-sm120` run the
semantics, footprint and PTX↔SASS differential tests with `NV_ARCH` set: the PTX test kernels are
retargeted (`Tests.ptxFor`) and those a target cannot compile are reported as skipped.

## Not covered yet

- Texture and surface opcodes beyond the SASS forms of PTX `tex.{1d,2d,3d,a1d,a2d,cube,acube}`,
  `tld4`, `txq` (dimensions), `suld.b`/`sust.b`/`sured.b` with layered forms (`decodeTex`: `TEX`,
  `TLD`, `TLD4`, `TXQ TEX_HEADER_DIMENSION`, `SULD`/`SUST`/`SURED .D.BA`, `ARRAY_1D`/`ARRAY_2D`,
  `1D_ARRAY`/`2D_ARRAY`, `CUBE`/`ARRAY_CUBE`): samplers, fetches from a mipmapped texture other than `TEX.LL`, formatted
  surface accesses, and sampler queries;
  `HSET2` (a compare with a register result); `F2F` and `F2I` from packed sources.
- Local memory is executed (`LDL`/`STL`) but its per-lane stores are the engine's to allocate
  (`Kernel.localBytes`).
- Forms of covered opcodes not in the test listings fault rather than guess.
