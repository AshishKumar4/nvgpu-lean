"""PyTorch on a CUDA device, the emulated one or a real GPU: the same programs, their outputs saved.

    python torch_check.py run OUT.npz [CASE ...]   # run the cases (all by default), save every output
    python jax_check.py compare A.npz B.npz         # bit-compare two runs' outputs (shared with JAX)

The cases mirror jax_check.py's on the same inputs (NumPy's generator, seeded): a float32 and a
bfloat16 matmul, a softmax, reductions and elementwise maths, and a one-layer transformer's loss
and gradients, each through torch.cuda (NVIDIA's math libraries for the matmuls, PyTorch's own
kernels for the rest). TF32 is off, so
a float32 matmul is a float32 one on every part. A run prints the device and, per output, its
shape, dtype and SHA-256.
"""
import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jax_check import B, D, F, H, T, V, inputs  # noqa: E402  (the same shapes and inputs)


def case_matmul_f32(torch, dev):
    a, b = inputs(1, (128, 96), (96, 64))
    return {"matmul_f32": torch.from_numpy(a).to(dev) @ torch.from_numpy(b).to(dev)}


def case_matmul_bf16(torch, dev):
    a, b = inputs(2, (128, 128), (128, 128))
    x = torch.from_numpy(a).to(dev).bfloat16()
    y = torch.from_numpy(b).to(dev).bfloat16()
    return {"matmul_bf16": (x @ y).float()}


def case_softmax(torch, dev):
    (x,) = inputs(3, (64, 257), scale=4.0)
    return {"softmax": torch.softmax(torch.from_numpy(x).to(dev), dim=-1)}


def case_reductions(torch, dev):
    """PyTorch's own kernels, no BLAS: reductions and elementwise maths."""
    (x,) = inputs(6, (96, 333), scale=3.0)
    t = torch.from_numpy(x).to(dev)
    return {"sum": t.sum(), "sum_rows": t.sum(-1), "mean_cols": t.mean(0), "max_rows": t.max(-1).values,
            "argmax_rows": t.argmax(-1), "var_rows": t.var(-1), "logsumexp_rows": torch.logsumexp(t, -1),
            "cumsum_rows": t.cumsum(-1), "gelu": torch.nn.functional.gelu(t), "exp_tanh": torch.exp(t) * torch.tanh(t)}


def case_layer_norm(torch, dev):
    (x,) = inputs(6, (96, 333), scale=3.0)
    return {"layer_norm": torch.nn.functional.layer_norm(torch.from_numpy(x).to(dev), (333,))}


def case_transformer(torch, dev):
    from jax_check import params

    p = {k: torch.from_numpy(v).to(dev).requires_grad_() for k, v in params().items()}
    tokens = torch.from_numpy(np.random.default_rng(5).integers(0, V, (B, T + 1)).astype(np.int64)).to(dev)

    def norm(x, g):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * g

    x = p["emb"][tokens[:, :-1]]
    h = norm(x, p["g1"])
    q, k, v = (h @ p["qkv"]).split(D, dim=-1)
    split = lambda t: t.reshape(B, T, H, D // H).transpose(1, 2)
    q, k, v = split(q), split(k), split(v)
    s = q @ k.transpose(-1, -2) / np.sqrt(D // H)
    s = torch.where(torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev)), s, torch.tensor(-1e9, device=dev))
    a = (torch.softmax(s, -1) @ v).transpose(1, 2).reshape(B, T, D)
    x = x + a @ p["out"]
    x = x + torch.nn.functional.gelu(norm(x, p["g2"]) @ p["up"]) @ p["down"]
    logp = torch.log_softmax(x @ p["emb"].T, -1)
    loss = -logp.gather(-1, tokens[:, 1:, None]).mean()
    loss.backward()
    out = {"transformer_loss": loss.detach()}
    out.update({f"transformer_grad_{k}": t.grad for k, t in p.items()})
    return out


CASES = {"matmul_f32": case_matmul_f32, "matmul_bf16": case_matmul_bf16, "softmax": case_softmax,
         "reductions": case_reductions, "layer_norm": case_layer_norm, "transformer": case_transformer}


def run(path, names):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.init()  # raises, saying why, where there is no usable CUDA device
    dev = torch.device("cuda")
    print("device:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0), flush=True)
    out = {}
    for name in names or CASES:
        t0 = time.time()
        for k, v in CASES[name](torch, dev).items():
            a = v.detach().cpu().numpy()
            out[k] = a
            print(f"{k}: {a.shape} {a.dtype} sha256={hashlib.sha256(a.tobytes()).hexdigest()[:16]}", flush=True)
        print(f"case {name}: {time.time() - t0:.1f} s", flush=True)
    np.savez(path, **out)
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        sys.exit(run(sys.argv[2], sys.argv[3:]))
    print(__doc__)
    sys.exit(2)
