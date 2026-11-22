# nvgpu-lean

A model of NVIDIA GPUs in Lean 4 that runs kernels and that kernels can be proven against.

- **Run** PTX, or SASS compiled from it, on real data, with cycle-level timing, on any x86-64 Linux
  machine. No GPU and no driver: `ptxas`, `nvcc`, `cuobjdump` and `nvdisasm` come from NVIDIA's
  PyPI wheels.
- **Prove** a kernel's output equal to a specification under every execution the PTX memory model
  allows.
- **Bound** a kernel's running time from below, from the GPU's resources, and report the gap.

## Parts

One parameter set each (`NV/Arch/`), every number a `Fact` with its kind and source
(`docs/facts-<key>.md`, generated from the code).

| key | part | target |
|---|---|---|
| `h200` | H200 SXM 141 GB | `sm_90a` |
| `h100` | H100 SXM5 80 GB | `sm_90a` |
| `h100-pcie` | H100 PCIe 80 GB | `sm_90a` |
| `h100-nvl` | H100 NVL 94 GB | `sm_90a` |
| `a100-40`, `a100-80` | A100 SXM4 40 / 80 GB | `sm_80` |
| `rtxpro6000` | RTX PRO 6000 Blackwell Server Edition | `sm_120a` |
| `b200` | B200 (tensor memory, `tcgen05.mma` of one CTA and of a CTA pair) | `sm_100a` |
| `b300` | B300 | `sm_103a` |

Kernels are compiled for the part's target; an instruction whose unit the part lacks (wgmma off
`sm_90a`, TMA and clusters before `sm_90`) is refused.

## What is modelled

PTX parsing, type checking and semantics; SASS reading and semantics for `sm_80`, `sm_90a`,
`sm_100a` and `sm_120a`, tested against the PTX semantics on each target; IEEE and tensor-core
numerics; the PTX memory model with DRF-SC, and a proof that the request machine the simulator
uses only produces consistent executions; a cycle-level core (warp schedulers, dependence
counters, register banks, code caches, L1/shared memory, L2, DRAM, TMA, wgmma, tcgen05, clusters),
its SMs simulated in parallel on worker threads; a hazard check that flags any instruction issuing
over a pending writeback, and static checks of warpgroup MMA code (`sass-ctrl`); a data-race check
over a CTA's threads and the async proxy's copies (`--races`) and an unwritten-output check
(`--unwritten`); a functional mode, also on worker threads; the accelerated mode
(`tools/lift`: a kernel's SASS lifted to CUDA C and run on a GPU, bit for bit equal to the
simulator); and the emulated CUDA device (`tools/cudadev`: a `libcuda.so.1`, NVML and NCCL that
run CUDA programs, and JAX on them, on the simulator).

## Running a kernel

```sh
tools/cuda/install.sh ~/cuda            # pinned CUDA compiler tools, no GPU needed
export PATH=~/cuda/bin:$PATH
lake build sim
lake exe sim examples/specs/axpy.json --gpu a100-40   # compile for sm_80, run, check, report
```

A launch spec (JSON) names the kernel source (`.cu`, `.ptx`, or a compiled `.cubin`), the grid,
buffers and their contents, the parameters, and the expected outputs. `sim` prints, per launch,
the cycles (launch, active, drain), instruction and memory counters, and where warps stalled.

On a machine with an NVIDIA GPU, the accelerated mode runs the same launches there:

```sh
lake exe sim examples/specs/tcgen05_gemm.json --gpu b200 --functional --export out/b200/tcgen05_gemm
sh tools/lift/check.sh out          # lift, compile and run each exported launch, compare its memory
```

`check.sh` lifts each launch's SASS to CUDA C (`tools/lift/lift.py`), compiles it for the local
GPU, runs it, and compares global memory with the simulator's result byte for byte. A
warpgroup or tensor-memory MMA, or a TMA or bulk copy, is split into independent tasks
(16-row output tiles, 4 KB of rows). The issuing warp runs them, along with helper CTAs on the
SMs the launch leaves idle. So the B200's tcgen05 GEMM runs on an A100 in 0.12 ms (32 ms in one
issuing thread), bit for bit equal to the simulator. All 22 example launches, exported for the
H100, A100 and B200 (clusters, distributed shared memory, TMA multicast, `mma.sync` with
`ldmatrix`/`stmatrix`, and divergent loops with `BREAK` among them), are bit-exact on an
A100-SXM4-40GB and on a GH200, with helper CTAs and without.

## Proving things

`Examples/` holds what the library can prove today:

- `SplitMergeProgram.lean`: a split-KV merge (atomic counter, last split reduces) writes the sum
  of all partials, for any number of splits, under every execution the PTX memory model allows.
- `RingSafety.lean`: a TMA producer/consumer ring over full/empty mbarriers never faults.
- `Warps.lean`, `JoinBugs.lean`: named-barrier protocols, with real bugs rejected.

`docs/formal.md` maps the theorems and their assumptions.

## Layout

| path | content |
|---|---|
| `NV/PTX` | PTX syntax, checker, semantics, interpreter; the memory model (`Memory/`) |
| `NV/SASS` | SASS reader, semantics, queries over compiled code (`Query.lean`) |
| `NV/Num` | numeric formats, rounding, NVIDIA's approximate functions, tensor-core arithmetic |
| `NV/Machine`, `NV/Memory` | requests and effects shared by both ISAs; stores; the request machine |
| `NV/Core` | the timing core: engine, ports, memory system, async units |
| `NV/Arch` | parts, every number a fact |
| `NV/Formal` | outcomes, DRF-SC, simulator soundness |
| `Examples/` | proofs about kernels |
| `Device/`, `tools/cudadev` | the emulated CUDA device: its Lean C ABI and the driver over it |
| `Tests/`, `tests/` | tests and their kernels |
| `docs/` | per-layer documentation and the generated fact tables |

## Rules

- Every hardware number is a `Fact` with its kind and source (`docs/sources.md`).
- No `sorry`, `axiom`, `partial`, `unsafe` or FFI in the library (`scripts/check.sh`).
- CI rows are files in `ci/rows/`; any machine with the toolchain runs them.
