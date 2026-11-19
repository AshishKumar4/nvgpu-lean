"""Cases for `num-tests sftensor`: random tcgen05.mma dot products with SF Tensor's model's results.

    python3 tools/num/sftensor_vectors.py SFTENSOR_DIR N OUT

One line per case: `FAMILY K A… B… SA… SB… C D` in hexadecimal, FAMILY one of the datapaths below
(`f16`, `bf16`, `tf32`, `ATYPE.BTYPE` for kind::f8f6f4, `mx.ATYPE.BTYPE` for kind::mxf8f6f4,
`mxf4`, `nvf4.2x`, `nvf4.4x`, `nvfp4`), SA and SB empty for the unscaled ones. Elements are drawn
over every encoding (NaN and infinities included) or, for half the cases, from a moderate range
whose products stay finite; accumulators over every binary32. Seeded: the same N gives the same
file.
"""

import random
import sys

sys.path.insert(0, sys.argv[1])
from tcgen05_model.model import DotProduct, make_model  # noqa: E402

N = int(sys.argv[2])
WIDTH = {"e4m3": 8, "e5m2": 8, "e2m3": 6, "e3m2": 6, "e2m1": 4}
F8 = list(WIDTH)


def f32(rng, moderate):
    if moderate:
        return (rng.getrandbits(32) & 0x807FFFFF) | ((120 + rng.randrange(16)) << 23)
    return rng.getrandbits(32)


def elem(rng, fmt, moderate):
    if fmt == "f16":
        return (rng.getrandbits(16) & 0x83FF) | ((10 + rng.randrange(10)) << 10) if moderate else rng.getrandbits(16)
    if fmt in ("bf16", "tf32"):
        return f32(rng, moderate)
    v = rng.getrandbits(WIDTH[fmt])
    # Moderate FP8: no NaN or infinity (the exponent field not all ones).
    while moderate and fmt in ("e4m3", "e5m2") and (v >> (3 if fmt == "e4m3" else 2)) & 0xF == 0xF and (fmt == "e4m3" or (v >> 2) & 0x1F == 0x1F):
        v = rng.getrandbits(8)
    return v


def scale(rng, sf, moderate):
    if sf == "ue8m0":
        return 117 + rng.randrange(21) if moderate else rng.getrandbits(8)
    return 0x28 + rng.randrange(0x30) if moderate else rng.getrandbits(7)


def families():
    yield "f16", 16, make_model("f16"), "f16", "f16", None, 0
    yield "bf16", 16, make_model("bf16"), "bf16", "bf16", None, 0
    yield "tf32", 8, make_model("tf32"), "tf32", "tf32", None, 0
    for a in F8:
        for b in F8:
            yield f"{a}.{b}", 32, make_model(a, b), a, b, None, 0
            yield f"mx.{a}.{b}", 32, make_model(a, b, scaling="ue8m0", kind="mxf8f6f4"), a, b, "ue8m0", 32
    yield "mxf4", 64, make_model("e2m1", scaling="ue8m0", kind="mxf4"), "e2m1", "e2m1", "ue8m0", 32
    yield "nvf4.2x", 64, make_model("e2m1", scaling="ue8m0", kind="mxf4nvf4", scale_vec=2), "e2m1", "e2m1", "ue8m0", 32
    yield "nvf4.4x", 64, make_model("e2m1", scaling="ue8m0", kind="mxf4nvf4", scale_vec=4), "e2m1", "e2m1", "ue8m0", 16
    yield "nvfp4", 64, make_model("e2m1", scaling="ue4m3", scale_vec=4), "e2m1", "e2m1", "ue4m3", 16


with open(sys.argv[3], "w") as out:
    for name, k, model, af, bf, sf, block in families():
        rng = random.Random(name)
        for i in range(N):
            moderate = i % 2 == 0
            a = [elem(rng, af, moderate) for _ in range(k)]
            b = [elem(rng, bf, moderate) for _ in range(k)]
            sa = [scale(rng, sf, moderate) for _ in range(k // block)] if sf else []
            sb = [scale(rng, sf, moderate) for _ in range(k // block)] if sf else []
            c = f32(rng, moderate) if i % 3 else 0
            case = DotProduct(tuple(a), tuple(b), c, scale_a=tuple(sa) if sf else None,
                              scale_b=tuple(sb) if sf else None)
            d = model.eval(case)
            words = [f"{x:x}" for x in a + b + sa + sb + [c, d]]
            out.write(f"{name} {k} {' '.join(words)}\n")
