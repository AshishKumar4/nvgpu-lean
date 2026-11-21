"""JAX on a mesh of emulated devices (NVGPU_LEAN_GPU=h100x2, b200x8, ...): the setup a program needs
before JAX's CUDA backend starts, and a runner for any program.
    python tools/cudadev/jax_mesh.py PROGRAM.py [ARGS...]   # runs PROGRAM as __main__ on the mesh
"""
import ctypes
import os
import runpy
import sys


def devices_named() -> int:
    """The GPU count NVGPU_LEAN_GPU names (`b200x8` 8, `h100,h100` 2)."""
    spec = os.environ.get("NVGPU_LEAN_GPU", "h200")
    return sum(int(t.rsplit("x", 1)[1]) if "x" in t and t.rsplit("x", 1)[1].isdigit() else 1
               for t in spec.split(","))


def setup():
    """Import JAX for the emulated devices, before its backend starts; return the jax module."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    # The device has no profiler (CUPTI), whose version JAX's CUDA plugin checks at start.
    os.environ.setdefault("JAX_SKIP_CUDA_CONSTRAINTS_CHECK", "1")
    # No autotuning: it times kernels with XLA's delay kernel, which spins on host memory (jax_check.py).
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_gpu_autotune_level" not in flags:
        flags += " --xla_gpu_autotune_level=0"
    # The devices run one kernel at a time between them, so a device can reach a collective minutes
    # after its peers: XLA's rendezvous would end the program after 40 seconds (8 x B200 did).
    for timeout in ("xla_gpu_first_collective_call_terminate_timeout_seconds",
                    "xla_gpu_first_collective_call_warn_stuck_timeout_seconds",
                    "xla_gpu_executable_terminate_timeout", "xla_gpu_executable_warn_stuck_timeout"):
        if timeout not in flags:
            flags += f" --{timeout}=86400"
    os.environ["XLA_FLAGS"] = flags.strip()
    import jax
    from jax._src import hardware_utils

    # The driver's state file stands for the device nodes JAX looks for (jax_check.py says why).
    hardware_utils._NVIDIA_GPU_DEVICES.append(f"/dev/shm/nvgpu-lean-{os.getuid()}")
    # The emulated NCCL (nccl.c), loaded before the backend starts: XLA's dlopen("libnccl.so.2")
    # then finds it loaded. Local, as XLA loads its libraries: the Lean runtime under it stays out
    # of the global namespace.
    ctypes.CDLL("libnccl.so.2")
    return jax


if __name__ == "__main__":
    setup()
    program = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(program, run_name="__main__")
