# PTX: front end, semantics and interpreter

PTX ISA 9.4 text to typed instructions, per-thread semantics yielding the
machine's `Effect`s, and a functional interpreter.

## Front end (`NV/PTX/Syntax`)

| module | what |
|---|---|
| `Lex`, `Parse` | ASCII PTX to `Module RawInstr`: directives (ch. 11), declarations with vectors, arrays, `%r<n>`, initializers, `.attribute`, `.ptr` (§5.4), constant expressions with C precedence (§4.5.4), operands (§6), `.loc`/`.file`/`.section` |
| `Type`, `Qual`, `SReg` | every type keyword of §5.2, state spaces, every qualifier as a typed `Qual`, the special registers of ch. 10 |
| `Instr` | the opcodes of §9.7 (`Op`, one constructor per instruction form, 206) with the qualifier kinds each admits; `decode` reads a `RawInstr` |
| `Check` | the static rules: names declared and in scope, guards and predicate operands of type `.pred`, the instruction's types among those its syntax lists, operand shapes and register types compatible (§5.2.1, §6.1, the relaxed rules of §9.4.1), and the target: each instruction's base form against its Target ISA Notes (`Op.requires`; `sm_90a`-only and `sm_100a`-family features such as `wgmma` and `tcgen05` are refused elsewhere, `tcgen05` on `sm_120a` too) and the forms that need more (`Instr.formRequires`: `.bf16` arithmetic, half-precision `min`/`max`, `fma.relu`/`.oob`) |

`lake exe ptx check PATH…` reads and checks PTX files. The checker states type rules and operand
signatures for the instructions the semantics executes; texture, surface, tensor-core and fabric
instructions are checked for declared names and admitted qualifiers only.

**Corpus** (`tools/ptx/corpus.sh`, CI rows `ptx-corpus`, `ptx-corpus-cutlass`,
`ptx-corpus-fa3`): PTX that `nvcc` 13.0.88 (`tools/cuda`) emits for NVIDIA cuda-samples v13.0
(every sample that compiles without third-party libraries), CUTLASS v4.2.1 Hopper and Blackwell
GEMM examples, FlashAttention-3 (`hopper/` at 2ed4a1f) forward and backward instantiations, the
Triton 3.5.0 kernels of `tools/ptx/triton_kernels.py` (`sm_90a` and `sm_100a`). Every file is
read and checked with no diagnostic.

## Semantics (`NV/PTX/Sem`)

`Prog.ofModule` records the target's SM version (`Prog.sm`, from `.target`), which picks the
tensor-core datapath of `mma.sync` (`Matrix.model`: Ampere, Hopper or Blackwell numerics, see
docs/numerics.md; `sm_120a` takes the B200's, as the model's authors define it). It flattens each function (blocks only scope names), numbers registers, places
module, function-scope `.shared` and `.local` variables and the parameters, and resolves every
operand. `Thread.step : Env → Thread → Effect Thread` executes one instruction:

| instructions | effect |
|---|---|
| integer, bit, logic, comparison, selection, `mov`, `prmt`, the scalar and SIMD video instructions (selectors, masks, secondary operations, merges), integer `cvt`, extended precision (`CC.CF` per frame) | `next` |
| floating point on `NV.Num`: `add`/`sub`/`mul`/`fma`/`mad`/`div`/`rcp`/`sqrt` with rounding, `ftz`, `sat`, `relu`; `.approx` functions; `min`/`max` (`.NaN`, `.xorsign.abs`); `abs`/`neg`/`copysign`/`testp`; `setp`/`set`/`slct`; mixed precision; packed `f16x2`/`bf16x2`/`f32x2`; every float/int `cvt` including packed FP8/FP6/FP4, `.s2f6x2` and the `.scaled::n1/n2::ue8m0` forms | `next` |
| `ld`, `ldu`, `st` (vectors, `.b128`), `atom`, `red` (vectors of floats, `.b128` `exch`/`cas`, `.b16` `cas`: element by element, `AtomOp.apply`); generic addresses through `NV.Machine.Generic.Windows` | `mem` |
| `bra`, `brx.idx`, `call`/`ret` (frames; `.param` arguments by value), `exit` | `next`, `exit` |
| `call` of the external `vprintf` (device `printf`): the format and the argument buffer read byte by byte, a record appended to the thread's `printf` log (`Machine.printfText` renders them) | `mem` steps, then `next` |
| `bar`/`barrier` `.sync`/`.arrive`, `barrier.cluster`, fences, `membar` | `sync` |
| `shfl.sync`, `vote.sync`, `match.sync`, `redux.sync`, `elect.sync`, `activemask`, `bar.warp.sync`, `bar.red` | `collect` (`NV.Machine.Collective.answer`) |
| `cvta`, `isspacep`, `mapa`, `getctarank` | `next` |
| `mma.sync` (`m16n8k8`/`k16` f16/bf16 → f32/f16, `m16n8k4`/`k8` tf32, `m8n8k16`/`m16n8k16`/`k32` s8/u8, `m8n8k32`/`m16n8k32`/`k64` s4/u4, `m8n8k128`/`m16n8k128`/`k256` b1 `.and`/`.xor` `.popc`, `m8n8k4`/`m16n8k4`/`k8`/`k16` f64, `m16n8k32` e4m3/e5m2 → f32) on `NV.Num.Tensor`; on `sm_90a`, which has no FP8 datapath, FP8 as FP16 MMAs (`Matrix.hopperF8`: exact widening to FP16, two chained FP16 MMAs over alternate K pairs, an FADD of C); `movmatrix`; `ldmatrix`/`stmatrix` (`m8n8.b16`; `ldmatrix.m16n16.trans` `.b8` and from `.b4x16_p64`/`.b6x16_p32`, `.m8n16` from those, `stmatrix.m16n8.trans.b8`: `Matrix.MatMove`, the `.m16n16` layout CUTLASS's, the padded elements' place in a byte inferred) | `collect` (`gather`); `ldmatrix`/`stmatrix`: each lane giving a row's address moves its 16 bytes, and a gather exchanges the rows |
| `wmma.load`/`wmma.store` (every shape and type; the sub-byte and single-bit fragments a word a step), one element a step, in the target's fragment layout (`Matrix.wfrag`, the same on sm_80, sm_90a and sm_120a: the ISA leaves the layout to the target); `wmma.mma` as `mma.sync` | `mem` steps; `collect` |
| `cvt.pack.sat`; `createpolicy` (a constant: caches are not thread state) | `next` |
| `alloca`, `stacksave`, `stackrestore`: a stack pointer per frame over a 1 KB stack above the `.local` variables (the runtime's default `cudaLimitStackSize`), freed on return; overflow faults | `next` |
| `st.bulk` (sm_100+; `sm_120a`): shared memory zeroed 256 bytes a step | `mem` steps |
| `tex.{1d,2d,3d,a1d,a2d,cube,acube}`, `tld4`, `txq`, `istypep`, `suld.b`, `sust.b`, `sured.b` (with `.a1d`, `.a2d`), `suq` on the launch's texture and surface objects (`NV.Tex`, `Launch.textures`: handle `i + 1` names entry `i`): one load of every texel a fetch combines, then the CUDA Programming Guide's addressing and filtering (9-bit fixed-point weights, combined as a GH200 combines them; what it leaves open is marked inferred in `NV/Machine/Texture.lean`, among it the rounding of floats filtered across many binades and of binary16 sums), and `tex.level` from mipmapped textures | `mem` |

Undefined behaviour (integer division by zero, misaligned or unmapped accesses, `fns` base above
31, a shuffle reading a non-participating lane) is a `fault`, never a chosen value.
Asynchronous operations (`cp.async*`, TMA, `mbarrier`, `wgmma`) are part H's
(`NV.PTX.Async.exec?`; with `st.async`/`red.async` into a peer CTA, `cp.reduce.async.bulk.tensor`,
`tensormap.replace`, `tensormap.cp_fenceproxy`, `mbarrier.check_layout` and
`clusterlaunchcontrol`, whose `try_cancel` always fails: the model launches every cluster, which
the ISA allows). Not yet modelled, and faulting: stochastic rounding into the four-lane
FP8/FP6/FP4 forms (the ISA says which half of `rbits` feeds which pair of inputs, not how a half
divides between the pair), sparse `mma` of floats (no published numerics of the sparse datapath; the integer `mma.sp` forms are modelled, `Matrix.computeSparse`), `ldmatrix` `.s8.s4` (PTX ISA 9.4),
texture samplers, depth compare, fetches from a mipmapped texture by gradients (on a texture of
one level the forms with an LOD or gradients fetch from that level),
multisampling, formatted surface accesses (`.p`). `multimem.*` acts on a one-device multicast
object (the device's own memory). Outside the model, faulting so: fabric memory (`fabric.*`) and
`.layout::v1` mbarriers, which need PTX ISA 9.3.
`docs/ptx-coverage.md` (`.lake/build/bin/ptx-coverage --doc`) lists every instruction by target,
probed through the dispatch itself.

`elect.sync` elects the lowest participating lane: the ISA fixes only that the choice is
deterministic per member mask (inferred). Window bases of the generic address space are ours (`Generic.lean`).

## Interpreter (`NV/PTX/Interp`)

Flat stores per space (global, constant, kernel parameters, shared per CTA, local per thread),
requests performed at once (sequentially consistent). `runWarps` steps SIMT warps (the runnable
lanes at one instruction together, which `activemask` answers); `run` takes any scheduler;
`explore` enumerates every interleaving of memory and synchronization steps.

## Tests (CI row `ptx`)

* `lake exe ptx-tests`: single instructions, video instructions against §9.7.20's pseudocode
  worked by hand, float atomics, and nvcc-compiled kernels (`tests/ptx/kernels/*.cu`, their `sm_90a` PTX
  committed beside them) on fixed inputs against references written from the CUDA functions'
  meaning — integer ops on edge values, shuffles/votes/match/redux, shared-memory reductions with
  barriers and atomics, a transpose, float ops including directed rounding, f16/bf16 conversions
  and f64, and an `ldmatrix`/`mma.sync`/`stmatrix`/`movmatrix` tile on small integers — the float
references from an exact-rational IEEE oracle independent of `NV.Num`; `brx.idx`; and
  litmus programs whose outcome sets over all interleavings are the sequentially consistent ones.
* `lake exe ptx-oracle`: ROCetta below.

## ROCetta (evaluation)

The ROCetta project (University of Rochester, cs.rochester.edu/~sree/rocetta) specifies PTX in
XIR, with executable C and SMT-LIB semantics. Licences: `ptx-xir` (semantics and C runtime) is
LGPL-3.0-or-later; `ptxparser` and the PTX test suite carry no licence or are private; the
`ptx-math` library (bit-accurate MUFU models, MIT) is what `NV.Num.Approx` ports. So nothing is
vendored: `tools/ptx/rocetta/run.sh` downloads the 2026-02 beta C semantics (sha256-pinned) and
ptx-xir's C runtime (pinned commit), builds a harness that runs every scalar `execute_*` function
on edge and seeded random vectors, and `lake exe ptx-oracle` runs each as one PTX instruction
through `Thread.step`.

It covers integer, extended precision, float with every rounding and `ftz`/`sat`, comparisons
with Boolean combination, `set`, `selp`, `slct` and bit operations; `.approx` instructions
(ROCetta uses the host libm) and `fns` with an undefined base are expected to differ. NaN results
are compared as NaN (ROCetta keeps host NaN payloads; NVIDIA returns
the canonical NaN). ROCetta's C semantics predates PTX 7 for most forms (no `bf16`, FP8, packed
types, `redux`, `elect`), covers no memory, control or collective instructions, and models
`div.approx` as `a / b` rather than the ISA's `a · (1/b)`; it is a differential oracle for the
scalar ALU instructions, not a reference for the rest.
