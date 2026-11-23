# The emulated CUDA device

A CUDA driver (`libcuda.so.1`), management library (`libnvidia-ml.so.1`) and `nvidia-smi` that
run programs on the simulator instead of a GPU: CUDA programs, the runtime (cudart) and the
libraries and frameworks built on it load them where they would load NVIDIA's, and see the GPUs
`NVGPU_LEAN_GPU` names: one part, or a mesh of them (`b200x8`).

    sh tools/cudadev/build.sh        # builds .lake/build/cudadev/{libcuda.so.1,libnvidia-ml.so.1,nvidia-smi}
    sh tools/cudadev/check.sh        # the driver API, NVML and runtime-API programs on h200, a100-80, rtxpro6000
    LD_LIBRARY_PATH=.lake/build/cudadev NVGPU_LEAN_GPU=h100 ./my_cuda_program
    LD_LIBRARY_PATH=.lake/build/cudadev NVGPU_LEAN_GPU=h100 python tools/cudadev/jax_check.py run out.npz
    sh tools/cudadev/bundle.sh nvgpu-lean.tar.gz   # all of it relocatable, for machines without this repository:
    tar xzf nvgpu-lean.tar.gz && . nvgpu-lean/env.sh && python nvgpu-lean/python/jax_mesh.py PROGRAM.py

| variable | meaning |
|---|---|
| `NVGPU_LEAN_GPU` | `h200` (default), `h100`, `h100-pcie`, `h100-nvl`, `a100-40`, `a100-80`, `rtxpro6000`, `b200`; several, comma-separated, of one target (`h100,h100`), or a part times a count (`b200x8`, up to 16) |
| `NVGPU_LEAN_TIMED=1` | cycle-level launches (`cuEventElapsedTime` is then simulated time); default functional |
| `NVGPU_LEAN_CUDA` | the toolkit whose `cuobjdump` and `ptxas` load modules (default `/usr/local/cuda`) |
| `NVGPU_LEAN_CUOBJDUMP_OLD` | a CUDA 12 `cuobjdump`, which lists cubins in the ELF format before CUDA 13's (some CUDA 13 libraries still ship them); `cuobjdump.sh 12.8 DIR` fetches one |
| `NVGPU_LEAN_CUOBJDUMP_NEW` | a newer `cuobjdump`, for cubins the toolkit's cannot list (a newer nvJitLink's, Mosaic GPU's kernels); `cuobjdump.sh 13.5 DIR` fetches one |
| `NVGPU_LEAN_CACHE` | where loaded modules are kept, by image and target (default `$XDG_CACHE_HOME/nvgpu-lean`) |
| `NVGPU_LEAN_DRIVER_VERSION` | the driver version NVML reports |
| `NVGPU_LEAN_VERBOSE=1` | every external command and error to stderr |
| `NVGPU_LEAN_TRACE=1` | every driver call made through `cuGetProcAddress` or the runtime's tables that fails, with its error, to stderr (`2`: every call) |
| `NVGPU_LEAN_CLOCK` | the SM clock in MHz to run the part at (default the part's) |
| `NVGPU_LEAN_WORKERS` | the threads a launch's SMs are dealt to (default 8; functional results do not depend on it but for interleaving) |
| `NVGPU_LEAN_RACES=1` | each launch's race check (shared and global memory, async operations' accesses), its races and, timed, its hazards to stderr |
| `NVGPU_LEAN_ISSUE_TRACE=1` | every instruction a launch issues (cycle, SM, CTA, warp, pc, opcode) to stderr |
| `NVGPU_LEAN_LAUNCH_CACHE=DIR` | each launch's result recorded in DIR as it finishes, and a launch already recorded not simulated (`Device/Cache.lean`; not with the trace, spans or race check) |
| `NVGPU_LEAN_LAUNCH_CACHE_READ=DIR:…` | more directories entries are read from (`$ARMADA_RESUME/…` after a recall, an earlier job's artifacts) |
| `NVGPU_LEAN_LAUNCH_CACHE_PARTS=1` | each part of a launch's cache key to stderr (why a launch missed) |
| `NVGPU_LEAN_SPANS=1` | each CTA's span as it finishes (`cta ID sm SM start S end E`) to stderr |
| `NVGPU_LEAN_UNTIL=CYCLE` | each launch stopped at that cycle, a sample of a launch too long to run whole (`sim --until`); later launches read what it wrote; the launch cache off; its report to stderr |
| `NVGPU_LEAN_REPORT=1` | each launch's report to stderr, as `lake exe sim` prints it (cycles, instructions, memory traffic, stalls, the binding resource) |
| `NVGPU_LEAN_ACCEL` | the host GPU's target (`sm_80`, `sm_90a`, …): launches run on it as the lifter's CUDA (`accel.h`: each kernel compiled once to a cubin and cached, one harness process running every launch), with `NVGPU_LEAN_LIFT` (the `tools/lift` directory), `NVGPU_LEAN_ACCEL_NVCC` and `NVGPU_LEAN_ACCEL_NVCC_FLAGS` (the host's compiler), `NVGPU_LEAN_ACCEL_LIBRARY_PATH` (where the host's libcuda is) |
| `NVGPU_LEAN_ACCEL_CHECK=1` | every accelerated launch also simulated from the same memory and the results compared; one that differs is reported with its directory, kept with the simulator's result as `final.bin` (`tools/lift/main.cu` compares against it) |
| `NVGPU_LEAN_ACCEL_SPILL=1` | accelerated launches take the device's memory through files each time (`global.bin`, `module.bin`) instead of keeping it on the host's GPU between launches (`accel.h`'s resident memory: a launch sends only the 64 KB pages written since, a read fetches only the pages it covers, through a fixed 64 MB shared-memory window) |

## How it works

* `cuda.c` keeps the driver's bookkeeping: each device, its primary context and its device memory
  (device g's first fit from 16 TiB up, the devices' ranges end to end; module data past the part's
  memory), the calling thread's device (its current context's, or for an address the device whose
  window holds it), page-locked host memory, virtual
  memory reservations and pools, errors, and the entry points by name (`cuGetProcAddress`, with
  the older ABIs a runtime of an earlier CUDA asks for).
* `tex.h`: CUDA arrays (pitch-linear device memory, which a texture or surface instruction cannot
  tell from the hardware's block-linear layout) and texture and surface objects over arrays, linear
  and pitched memory (each the device's table entry its handle names, `Device.setTextures`, as the
  simulator's texture unit reads them, `NV/Machine/Texture.lean`); 2D and 3D copies reach arrays.
* `module.h`: modules and libraries. An image is a cubin, a fat binary (or cudart's wrapper of
  one, as `__cudaRegisterFatBinary` hands it over), or PTX. The driver's choice from a fat binary:
  the device's architecture's cubin (`sm_XYa` only on that architecture, `sm_XY` on its major with
  an equal or older minor), else the newest PTX not newer than the device, compiled by `ptxas` as
  the driver JITs it. The cubin is listed with `cuobjdump` (`-sass`, `-elf`) and kept in the cache,
  so a library's modules are prepared once per machine. A library (`cuLibraryLoadData`, how cudart
  registers every fat binary of a program and of the libraries it links) is read when one of its
  kernels or variables is first used. The Lean side reads a module once (`SASS.Read.Module`),
  places its `__device__` and `__constant__` data in device memory (so `cuModuleGetGlobal` and
  `cudaMemcpyToSymbol` reach what kernels read), and makes kernels by name.
* `stream.h`: work (launches, copies, sets, host functions, event records and waits, stream-ordered
  frees, memory-value operations, graph launches) is queued per stream; each created stream runs
  its queue in order on its own thread, the legacy default stream in the caller's thread after
  every blocking stream's work. An event wait waits for the recording it saw when enqueued. The
  device runs one piece of work at a time. Copies from host memory the driver does not know
  (pageable) return when done, as the driver's do.
* `graph.h`: graphs (kernel, copy, set, host, empty, child, event, allocation nodes; the generic
  `cuGraphAddNode`), instantiation in an order that respects every dependency, updates of an
  executable graph, and stream capture across streams joined by events.
* `tmap.h`: `cuTensorMapEncodeTiled`/`Im2col` write the model's own descriptor
  (`NV.Async.TensorMap.encode`), checked as the driver documents (`TensorMap.invalid?`).
* `runtime_tables.h`: the tables the runtime (cudart) asks the driver for before anything else,
  as ZLUDA defines them (MIT or Apache-2.0; its commit cited in the file).
* `Device/Api.lean` is the Lean side's C ABI: device memory, modules, kernels, launches, tensor
  maps. Kernels run on the one engine, `NV.Core`, functional (`Launch.functional`) or timed. A
  failed launch leaves the device unusable, as a GPU's context would be.
* Device attributes are the parts' Nsight Compute captures (`attributes/*.txt`) or, for parts not
  captured, inferred from them (sources in each file),
  compiled into `parts.h` by `gen.py`, which also writes the entry-point table (`procs.h`): every
  function the shim defines, and a stub naming itself for every other one `cuda.h` declares.
* `nvml.c` serves the public NVML API from what the driver of any process publishes in
  `/dev/shm/nvgpu-lean-$UID` (`state.h`): memory in use, processes, whether a launch is running.
  What an emulated device has no value for (temperature, power, fans, ECC) is not supported.
  `smi.c` is a `nvidia-smi` over the public API: the default panel, `-L`, `-q` and
  `--query-gpu=… --format=csv`.
* `nvml.c` and `smi.c` show every device.
* `jax_check.py` runs JAX programs (matmuls, a softmax, a transformer's loss and gradients) and
  bit-compares two runs' outputs: the emulated device's with a real GPU's. With JAX 0.12.0.1
  (`jax[cuda12]`), all 11 outputs of the emulated H100 equal a GH200's bit for bit, and all 11 of
  the emulated A100 an A100-SXM4-40GB's (2026-11-21; the device-jax row's outputs against
  `jax_check.py run` on each GPU).
* `torch_check.py` runs the same kind of programs through PyTorch. Its softmax, reductions and
  layer_norm cases (PyTorch 2.9.1, CUDA 12.9 wheels: sums, max, argmax, variance, logsumexp,
  cumsum, GELU, exp and tanh, a layer norm) on the emulated H100 and A100 equal a GH200's and an
  A100-SXM4-40GB's bit for bit, 12 of 12 outputs each (the device-torch row's outputs against
  `torch_check.py run` on each GPU).
* Accelerated (`NVGPU_LEAN_ACCEL`), on an A100-SXM4-40GB: `jax_check.py` on the emulated H100
  equals a GH200's bit for bit and on the emulated A100 the A100's (11 of 11 each), and
  `torch_check.py`'s softmax, reductions and layer norm on the emulated H100 a GH200's and on the
  emulated A100 the A100's (12 of 12 each). Once its 60 kernels are compiled (136 s, once, cached),
  the transformer case's 60 launches run in 1.3 s (0.2 s on the A100 itself). A two-layer
  transformer's three Adam steps (325 launches) on the emulated A100 equal the A100's bit for bit,
  every parameter: 3.4 s with the memory resident, 15.2 s through files.

## A mesh

`NVGPU_LEAN_GPU=b200x8` gives eight devices in one process, as an HGX board's GPUs are to a
program: each its own memory, contexts, streams and loaded modules, every pair peers
(`cuDeviceCanAccessPeer`), copies between them stream-ordered work through the host. A kernel
runs on its stream's device and reads and writes a peer's memory through the peer's pointers, as
over NVLink (the devices' memories lie end to end in its address space; a peer's bytes are timed
as the device's own, not at the link's latency); the devices run one piece of work at a time
between them.
Collectives run on the emulated NCCL, `libnccl.so.2` (`nccl.c`, NCCL 2.32.6's ABI): each call is
stream-ordered work that meets the other ranks' calls and is performed through the host, with
reductions in rank order (deterministic, not NVIDIA's ring order). XLA dlopens `libnccl.so.2`; a
program loads ours first (`ctypes.CDLL("libnccl.so.2")` after `import jax`, before the backend
starts), as `mesh_check.py` does. `mesh_check.py` runs JAX on two H100s and eight B200s (the
`mesh` row): per-device programs, pmap, shard_map, and psum, pmax, all_gather, psum_scatter,
ppermute and all_to_all, bit for bit against numpy.

A mesh keeps time: each device's clock is the cycles its kernels ran; a collective starts when its
last rank's device reaches it, takes its time on the part's links (`LinkParams`: NVLink or PCIe
bandwidth from NVIDIA's documents, the share a collective achieves from two H100s' measured
all-to-all; `Device.collectiveNs`), and every rank's device waits until it ends. With
`NVGPU_LEAN_VERBOSE` each collective's start and time are printed, and at exit each device's
simulated time (a step's time is the largest).

## Not provided

JAX's CUDA 13 plugin (its runtime asks the driver for a table not provided; the CUDA 12 plugin,
`jax[cuda12]`, runs). PyTorch's matmuls: NVIDIA's math libraries, as PyTorch 2.9's CUDA 12
wheels load them, ask the runtime for a table entry the device does not provide
(`runtime_tables.h` has only ZLUDA's), and stop; PyTorch's own kernels run (the `device-torch`
row). Managed memory, IPC, mipmapped arrays and texture resource views,
conditional graph nodes (they need the device runtime's `cudaGraphSetConditional`), batch
memory-operation nodes, external memory and semaphores, and kernels dereferencing host memory:
each fails with a CUDA error naming it.
