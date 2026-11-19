"""Compiles Triton kernels to PTX for sm_90a without a GPU: python triton_kernels.py OUTDIR.

The kernels are the standard Triton tutorial shapes (element-wise, softmax, matmul with a
pipelined K loop, layer norm, attention forward), written here so the corpus does not depend on
the tutorials' files.
"""
import os
import sys

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def add_kernel(x, y, out, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out + offs, tl.load(x + offs, mask=mask) + tl.load(y + offs, mask=mask), mask=mask)


@triton.jit
def softmax_kernel(out, inp, stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(inp + row * stride + cols, mask=cols < n_cols, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(out + row * stride + cols, e / tl.sum(e, axis=0), mask=cols < n_cols)


@triton.jit
def matmul_kernel(a, b, c, M, N, K, sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        ka = a + rm[:, None] * sam + (k * BK + rk)[None, :] * sak
        kb = b + (k * BK + rk)[:, None] * sbk + rn[None, :] * sbn
        acc += tl.dot(tl.load(ka), tl.load(kb))
    tl.store(c + rm[:, None] * scm + rn[None, :] * scn, acc.to(tl.bfloat16))


@triton.jit
def layernorm_kernel(x, y, w, b, stride, n, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n
    v = tl.load(x + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(v, axis=0) / n
    d = tl.where(mask, v - mean, 0.0)
    rstd = 1 / tl.sqrt(tl.sum(d * d, axis=0) / n + eps)
    out = d * rstd * tl.load(w + cols, mask=mask) + tl.load(b + cols, mask=mask)
    tl.store(y + row * stride + cols, out, mask=mask)


@triton.jit
def attn_fwd_kernel(q, k, v, o, seq, scale, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pm = tl.program_id(0)
    head = tl.program_id(1)
    base = head * seq * D
    rm = pm * BM + tl.arange(0, BM)
    rd = tl.arange(0, D)
    qb = tl.load(q + base + rm[:, None] * D + rd[None, :])
    m = tl.full((BM,), -float("inf"), tl.float32)
    l = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for start in range(0, seq, BN):
        rn = start + tl.arange(0, BN)
        kb = tl.load(k + base + rn[None, :] * D + rd[:, None])
        s = tl.dot(qb, kb) * scale
        mn = tl.maximum(m, tl.max(s, 1))
        p = tl.exp(s - mn[:, None])
        alpha = tl.exp(m - mn)
        l = l * alpha + tl.sum(p, 1)
        vb = tl.load(v + base + rn[:, None] * D + rd[None, :])
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vb)
        m = mn
    tl.store(o + base + rm[:, None] * D + rd[None, :], (acc / l[:, None]).to(tl.bfloat16))


KERNELS = [
    (add_kernel, {"x": "*fp32", "y": "*fp32", "out": "*fp32", "n": "i32", "BLOCK": "constexpr"},
     {"BLOCK": 1024}),
    (softmax_kernel, {"out": "*fp32", "inp": "*fp32", "stride": "i32", "n_cols": "i32",
                      "BLOCK": "constexpr"}, {"BLOCK": 1024}),
    (matmul_kernel, {"a": "*bf16", "b": "*bf16", "c": "*bf16", "M": "i32", "N": "i32", "K": "i32",
                     "sam": "i32", "sak": "i32", "sbk": "i32", "sbn": "i32", "scm": "i32",
                     "scn": "i32", "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
     {"BM": 128, "BN": 128, "BK": 64}),
    (layernorm_kernel, {"x": "*fp32", "y": "*fp32", "w": "*fp32", "b": "*fp32", "stride": "i32",
                        "n": "i32", "eps": "fp32", "BLOCK": "constexpr"}, {"BLOCK": 1024}),
    (attn_fwd_kernel, {"q": "*bf16", "k": "*bf16", "v": "*bf16", "o": "*bf16", "seq": "i32",
                       "scale": "fp32", "D": "constexpr", "BM": "constexpr", "BN": "constexpr"},
     {"D": 64, "BM": 128, "BN": 64}),
]


def main(out):
    os.makedirs(out, exist_ok=True)
    failed = 0
    for fn, sig, consts in KERNELS:
        for cap in (90, 100):
            try:
                src = ASTSource(fn=fn, signature=sig, constexprs=consts)
                k = triton.compile(src, target=GPUTarget("cuda", cap, 32))
                path = os.path.join(out, f"{fn.__name__}.sm_{cap}a.ptx")
                with open(path, "w") as f:
                    f.write(k.asm["ptx"])
            except Exception as e:  # noqa: BLE001 - report and continue with the next kernel
                failed += 1
                print(f"FAIL {fn.__name__} sm_{cap}: {type(e).__name__}: {e}")
    return failed


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1]) else 0)
