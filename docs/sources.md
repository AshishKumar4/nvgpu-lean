# Sources

Every `Fact` names one of these keys (`NV/Fact.lean`). One row per source; keys are short and
stable. A source with versions gets one key per version we transcribe.

| key | source | version / date | where |
|---|---|---|---|
| `ptx-isa-9.4` | NVIDIA, *Parallel Thread Execution ISA* | 9.4 | docs.nvidia.com/cuda/parallel-thread-execution |
| `ptx-isa-9.5` | NVIDIA, *Parallel Thread Execution ISA* (numerics: §5.2, §6.5, §9.7.3–9.7.6, §9.7.10.24, §9.7.16–17) | 9.5 | docs.nvidia.com/cuda/parallel-thread-execution |
| `cuda-headers-12.8` | NVIDIA CUDA runtime headers `cuda_fp8.hpp`, `cuda_fp16.hpp`, `cuda_bf16.hpp`, `math_constants.h` (host emulation of the conversions; canonical NaN constants) | `nvidia-cuda-runtime-cu12` 12.8.90 | pypi.org/project/nvidia-cuda-runtime-cu12 |
| `khattak2025` | F. A. Khattak, M. Mikaitis, *Accurate Models of NVIDIA Tensor Cores*, ACM TACO 23(3) (arXiv 2512.07004v4), and github.com/north-numerical-computing/MATLAB-tensor-core | arXiv v4, 2026-06-11; repository 79143bd | arxiv.org/abs/2512.07004 |
| `ptxmath` | B. Carleton, *PTX Math*: bit-accurate models of `rcp/sqrt/rsqrt/sin/cos/lg2/ex2.approx.f32` (MIT) | commit 2c7dc92 | github.com/bcarlet/ptx-math |
| `cuda-guide` | NVIDIA, *CUDA Programming Guide*, compute capabilities (§5.1.3 Tables 33–35) | CUDA 13.5 | docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html |
| `cuda-guide-async` | NVIDIA, *CUDA Programming Guide*, asynchronous data copies (§4.13.2.2.5, TMA swizzle: a box row narrower than the swizzle width takes the full width in shared memory) | CUDA 13.5 | docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html |
| `cuda-guide-12.6` | NVIDIA, *CUDA C++ Programming Guide*, arithmetic instruction throughput (§5.4.1 Table 4) | CUDA 12.6.3 | docs.nvidia.com/cuda/archive/12.6.3/cuda-c-programming-guide |
| `ncu-guide` | NVIDIA, *Nsight Compute Kernel Profiling Guide*, memory terminology (sectors, cache lines) | 2025 | docs.nvidia.com/nsight-compute/ProfilingGuide |
| `cuda-occupancy` | NVIDIA, CUDA Occupancy Calculator (allocation granularities per compute capability) | CUDA 13.0 | docs.nvidia.com/nsight-compute (occupancy calculator) |
| `a100-whitepaper` | NVIDIA, *NVIDIA A100 Tensor Core GPU Architecture* | 2020 | images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf |
| `hopper-whitepaper` | NVIDIA, *NVIDIA H100 Tensor Core GPU Architecture* | 2022 | resources.nvidia.com |
| `h100-product` | NVIDIA, H100 product page, specifications table | 2025 | nvidia.com/en-us/data-center/h100 |
| `h200-product` | NVIDIA, H200 product page, specifications table (marked preliminary) | 2025 | nvidia.com/en-us/data-center/h200 |
| `hopper-tuning` | NVIDIA, *Hopper Tuning Guide* | CUDA 13.0 | docs.nvidia.com/cuda/hopper-tuning-guide |
| `huerta2025` | R. Huerta, M. Abaie Shoushtary, J.-L. Cruz, A. González, *Dissecting and Modeling the Architecture of Modern GPU Cores*, MICRO 2025 (arXiv 2503.20481) | 2025 | arxiv.org/abs/2503.20481 |
| `luo2024` | W. Luo et al., *Benchmarking and Dissecting the Nvidia Hopper GPU Architecture*, IPDPS 2024 (arXiv 2402.13499) | v1 | arxiv.org/abs/2402.13499 |
| `luo2025` | W. Luo et al., *Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis* (arXiv 2501.12084) | 2025 | arxiv.org/abs/2501.12084 |
| `chipsandcheese-h100` | *Nvidia's H100: Funny L2, and Tons of Bandwidth*, Chips and Cheese (H100 PCIe, OpenCL and CUDA) | 2023 | chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth |
| `chipsandcheese-mi300x` | *Testing AMD's Giant MI300X*, Chips and Cheese (H100 PCIe as the comparison) | 2024 | chipsandcheese.com/p/testing-amds-giant-mi300x |
| `accelsim-h100`, `accelsim-h200` | Accel-Sim 2.0 tuned configs `SM90_H100`, `SM90_H200`: `gpgpusim.config` in accel-sim/gpgpu-sim_distribution and `trace.config` in accel-sim/accel-sim-framework | gpgpu-sim_distribution 4195dd4, accel-sim-framework 7ba54de | github.com/accel-sim |
| `meas-h200` | Measured on an H200 SXM | | |
| `meas-h100` | Measured on an H100 80GB HBM3 (SXM) | | |
| `meas-gh200` | Measured on a GH200 480 GB | | |
| `meas-a100` | Measured on an A100 SXM4 40 GB | | |
| `meas` | Measured on the part the fact names | | |
| `cuda-binutils` | NVIDIA, *CUDA Binary Utilities* (§4.3 Hopper Instruction Set, Table 8) | 13.5 (Nov 2026) | docs.nvidia.com/cuda/cuda-binary-utilities |
| `fixtures` | Observed in our compiled test kernels (`tests/sass/`) | | |
| `analogy` | Inferred: taken from a related fact or instruction class | | |
| `sf-tensor` | SF Tensor, *tcgen05-model*: a bit-level model of the B200's `tcgen05.mma` arithmetic (every dense and sparse floating, integer and block-scaled datapath) (MIT) | commit d96b821 | github.com/sf-tensor/tcgen05 |
| `h100-datasheet` | NVIDIA, *H100 Tensor Core GPU* datasheet (SXM, PCIe and NVL columns) | 2024 | resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306 |
| `h100-nvl-brief` | NVIDIA, *H100 NVL GPU* product brief, PB-11773-001 | v01 | nvidia.com/content/dam/en-zz/Solutions/Data-Center/h100/PB-11773-001_v01.pdf |
| `h100-pcie-brief` | NVIDIA, *H100 Tensor Core GPU PCIe* product brief, PB-11133-001 | v01 | nvidia.com/content/dam/en-zz/Solutions/gtcs22/data-center/h100/PB-11133-001_v01.pdf |
| `a100-datasheet-80` | NVIDIA, *A100 Tensor Core GPU* datasheet | May 2022 | nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf |
| `ampere-tuning` | NVIDIA, *NVIDIA Ampere GPU Architecture Tuning Guide* | CUDA 13.0 | docs.nvidia.com/cuda/ampere-tuning-guide |
| `abdelkhalik2022` | H. Abdelkhalik, Y. Arafa, N. Santhi, A. Badawy, *Demystifying the NVIDIA Ampere Architecture through Microbenchmarking and Instruction-level Analysis*, HPEC 2022 (arXiv 2208.11174) | 2022 | arxiv.org/abs/2208.11174 |
| `rtxpro-whitepaper` | NVIDIA, *NVIDIA RTX PRO Blackwell GPU Architecture* | v1.1 | nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf |
| `rtxpro6000-server` | NVIDIA, *RTX PRO 6000 Blackwell Server Edition* datasheet | 2025 | nvidia.com (product literature) |
| `blackwell-tuning` | NVIDIA, *Blackwell Tuning Guide* | CUDA 13.0 | docs.nvidia.com/cuda/blackwell-tuning-guide |
| `cuda-best-practices` | NVIDIA, *CUDA C++ Best Practices Guide*, §12.1.1 throughput of native arithmetic instructions | CUDA 13.0 | docs.nvidia.com/cuda/cuda-c-best-practices-guide |
| `chipsandcheese-blackwell` | C. Lam, *Blackwell: Nvidia's Massive GPU* (RTX PRO 6000 Blackwell workstation) | 2025 | chipsandcheese.com/p/blackwell-nvidias-massive-gpu |
| `jarmusch2025` | A. Jarmusch, S. Chandrasekaran, *Microbenchmarking NVIDIA's Blackwell Architecture: An in-depth Architectural Analysis*, IPDPS 2026 (arXiv 2512.02189v3), and github.com/UD-CRPL/IPDPS_26_B200_Microbenchmark | arXiv v3, 2026-03-02 | arxiv.org/abs/2512.02189 |
| `hgx-b200` | NVIDIA, *HGX Platform* specifications (HGX B300, HGX B200) | fetched 2026-11-09 | nvidia.com/en-us/data-center/hgx |
| `nvidia-blackwell-ultra` | NVIDIA Technical Blog, *Inside NVIDIA Blackwell Ultra: The Chip Powering the AI Factory Era* (Tables 1–2) | updated 2025-09-24, fetched 2026-11-09 | developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era |
| `yu2026` | S. Yu et al., *FlashGPU-Sim* (arXiv 2609.15311): RTX 5090 tensor-core and TMA microbenchmarks, §III-A, Table III | v1 | arxiv.org/abs/2609.15311 |
| `nvidia-forum-337324` | NVIDIA Developer Forums, *The latency of wave launch*, post 2 by NVIDIA staff (informal) | 2026-05-29 | forums.developer.nvidia.com/t/337324 |
| `jia2018` | Z. Jia, M. Maggioni, B. Staiger, D. P. Scarpazza, *Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking* (arXiv 1804.06826) | 2018 | arxiv.org/abs/1804.06826 |
| `blackwell-isa-stallfix` | K. (kacper-daftcode), *blackwell-isa* (MIT): `sm120.json` `stallfix`, stall floors on the RTX PRO 6000 Blackwell Server Edition | 2026-11-03/05 | github.com/kacper-daftcode/blackwell-isa |
