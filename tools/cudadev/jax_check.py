"""JAX on a CUDA device, the emulated one or a real GPU: the same programs, their outputs saved.

    python jax_check.py run OUT.npz [CASE ...]   # run the cases (all by default), save every output
    python jax_check.py compare A.npz B.npz      # bit-compare two runs' outputs

Each case is a jitted program on fixed inputs (NumPy's generator, seeded, so every machine feeds
the same bits): a float32 and a bfloat16 matmul, a softmax, and a one-layer transformer's loss
and gradients. A run prints the device JAX sees and, per output, its shape, dtype and SHA-256;
`compare` prints per output whether the bits agree, and, where not, how many elements differ and
the largest difference in units in the last place. Exit status: 0 when every output agrees.

XLA's autotuning is off (`--xla_gpu_autotune_level=0`), on both sides, so the kernels XLA picks
depend only on the device's description, not on timings the emulated device would make up.
"""
import hashlib
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# The device has no profiler (CUPTI), whose version JAX's CUDA plugin checks at start.
os.environ.setdefault("JAX_SKIP_CUDA_CONSTRAINTS_CHECK", "1")
# The CUDA device or nothing: JAX would otherwise fall back to the CPU and print a warning.
os.environ.setdefault("JAX_PLATFORMS", "cuda")
flags = os.environ.get("XLA_FLAGS", "")
if "xla_gpu_autotune_level" not in flags:
    os.environ["XLA_FLAGS"] = (flags + " --xla_gpu_autotune_level=0").strip()

import numpy as np  # noqa: E402


def emulated_device_visible():
    """JAX initializes its CUDA backend only where an NVIDIA device node exists (`/dev/nvidia0`,
    `/dev/nvidiactl`: jax._src.hardware_utils). The emulated device has none; its driver's state
    file (`/dev/shm/nvgpu-lean-$UID`, tools/cudadev/state.h), which the driver writes when it is
    initialized (here, before JAX's backend starts), stands for one. Nothing changes on a machine
    with a GPU."""
    import ctypes

    from jax._src import hardware_utils

    state = f"/dev/shm/nvgpu-lean-{os.getuid()}"
    if not os.path.exists(state) and not hardware_utils.has_visible_nvidia_gpu():
        ctypes.CDLL("libcuda.so.1").cuInit(0)
    hardware_utils._NVIDIA_GPU_DEVICES.append(state)


def inputs(seed, *shapes, dtype=np.float32, scale=1.0):
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal(s) * scale).astype(dtype) for s in shapes]


def case_matmul_f32(jnp, jax):
    a, b = inputs(1, (128, 96), (96, 64))
    return {"matmul_f32": jax.jit(lambda x, y: x @ y)(a, b)}


def case_matmul_bf16(jnp, jax):
    a, b = inputs(2, (128, 128), (128, 128))
    f = jax.jit(lambda x, y: jnp.matmul(x.astype(jnp.bfloat16), y.astype(jnp.bfloat16),
                                        preferred_element_type=jnp.float32))
    return {"matmul_bf16": f(a, b)}


def case_softmax(jnp, jax):
    (x,) = inputs(3, (64, 257), scale=4.0)
    return {"softmax": jax.jit(lambda v: jax.nn.softmax(v, axis=-1))(x)}


# A one-layer pre-norm transformer: token embedding, causal two-head attention, a GELU MLP, a tied
# output projection, mean cross-entropy over the next token.
D, H, F, V, T, B = 64, 2, 128, 96, 16, 4


def params():
    rng = np.random.default_rng(4)
    w = lambda *s: (rng.standard_normal(s) / np.sqrt(s[0])).astype(np.float32)
    return {"emb": w(V, D), "qkv": w(D, 3 * D), "out": w(D, D), "up": w(D, F), "down": w(F, D),
            "g1": np.ones(D, np.float32), "g2": np.ones(D, np.float32)}


def transformer_loss(jnp, jax, p, tokens):
    def norm(x, g):
        return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6) * g

    # The embedding as a one-hot matmul: its gradient is then a matmul, not a scatter-add whose
    # atomic accumulation order differs from run to run on a GPU.
    x = jax.nn.one_hot(tokens[:, :-1], V, dtype=p["emb"].dtype) @ p["emb"]
    h = norm(x, p["g1"])
    q, k, v = jnp.split(h @ p["qkv"], 3, axis=-1)
    split = lambda t: t.reshape(B, T, H, D // H).transpose(0, 2, 1, 3)
    q, k, v = split(q), split(k), split(v)
    s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(D // H)
    s = jnp.where(np.tril(np.ones((T, T), bool)), s, -1e9)
    a = (jax.nn.softmax(s, axis=-1) @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    x = x + a @ p["out"]
    x = x + jax.nn.gelu(norm(x, p["g2"]) @ p["up"]) @ p["down"]
    logits = x @ p["emb"].T
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(logp, tokens[:, 1:, None], axis=-1))


def case_transformer(jnp, jax):
    p = params()
    tokens = np.random.default_rng(5).integers(0, V, (B, T + 1)).astype(np.int32)
    loss, grads = jax.jit(jax.value_and_grad(lambda p, t: transformer_loss(jnp, jax, p, t)))(p, tokens)
    out = {"transformer_loss": loss}
    out.update({f"transformer_grad_{k}": g for k, g in grads.items()})
    return out


CASES = {"matmul_f32": case_matmul_f32, "matmul_bf16": case_matmul_bf16, "softmax": case_softmax,
         "transformer": case_transformer}


def run(path, names):
    import jax
    import jax.numpy as jnp

    emulated_device_visible()
    devs = jax.devices()
    print("devices:", devs, flush=True)
    print("device_kind:", devs[0].device_kind, "platform:", devs[0].platform, flush=True)
    out = {}
    for name in names or CASES:
        t0 = time.time()
        res = CASES[name](jnp, jax)
        for k, v in res.items():
            a = np.asarray(v)
            out[k] = a
            print(f"{k}: {a.shape} {a.dtype} sha256={hashlib.sha256(a.tobytes()).hexdigest()[:16]}", flush=True)
        print(f"case {name}: {time.time() - t0:.1f} s", flush=True)
    np.savez(path, **out)
    return 0


def ulps(a, b):
    """The largest distance between `a` and `b` in units in the last place (float32 or float64)."""
    if a.dtype.kind != "f":
        return int(np.max(np.abs(a.astype(np.int64) - b.astype(np.int64)))) if a.size else 0
    it = {4: np.int32, 8: np.int64, 2: np.int16}[a.dtype.itemsize]
    ai, bi = a.view(it).astype(np.int64), b.view(it).astype(np.int64)
    # Sign-magnitude to a monotone integer line.
    top = np.int64(1) << (8 * a.dtype.itemsize - 1)
    ai = np.where(ai < 0, -top - ai, ai)
    bi = np.where(bi < 0, -top - bi, bi)
    return int(np.max(np.abs(ai - bi))) if a.size else 0


def compare(pa, pb):
    a, b = np.load(pa), np.load(pb)
    bad = 0
    for k in sorted(set(a.files) | set(b.files)):
        if k not in a.files or k not in b.files:
            print(f"{k}: only in {'first' if k in a.files else 'second'}")
            bad += 1
            continue
        x, y = np.atleast_1d(a[k]), np.atleast_1d(b[k])
        if x.shape != y.shape or x.dtype != y.dtype:
            print(f"{k}: {x.shape} {x.dtype} vs {y.shape} {y.dtype}")
            bad += 1
            continue
        same = x.tobytes() == y.tobytes()
        if same:
            print(f"{k}: bit-identical ({x.size} elements)")
        else:
            n = int(np.sum(x.view(np.uint8).reshape(x.size, -1) != y.view(np.uint8).reshape(y.size, -1),
                           axis=1).astype(bool).sum())
            print(f"{k}: {n} of {x.size} elements differ, at most {ulps(x, y)} ulp")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        sys.exit(run(sys.argv[2], sys.argv[3:]))
    if len(sys.argv) == 4 and sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    print(__doc__)
    sys.exit(2)
