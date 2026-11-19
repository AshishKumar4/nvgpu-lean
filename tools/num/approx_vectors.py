"""Reference vectors for NV.Num.Approx: correctly rounded (nearest-even) results computed with
mpmath at 400 bits, written as `function format input expected` in hex, one per line. The
binary32 functions the special-function unit computes are checked against PTX's documented
accuracy around these values; the others must equal them.

Usage: python3 tools/num/approx_vectors.py > tools/num/approx-vectors.txt  (needs mpmath)
"""
import random
import mpmath

mpmath.mp.prec = 400
FORMATS = {"f32": (8, 23), "f16": (5, 10), "bf16": (8, 7)}


def decode(fmt, b):
    e, m = FORMATS[fmt]
    s = -1 if b >> (e + m) else 1
    ef, fr = (b >> m) & ((1 << e) - 1), b & ((1 << m) - 1)
    bias = (1 << (e - 1)) - 1
    if ef == (1 << e) - 1:
        return None
    if ef == 0:
        return s * mpmath.mpf(fr) * mpmath.mpf(2) ** (1 - bias - m)
    return s * mpmath.mpf((1 << m) + fr) * mpmath.mpf(2) ** (ef - bias - m)


def encode(fmt, v):
    """Nearest-even encoding of the mpmath value v (never a tie for these functions)."""
    e, m = FORMATS[fmt]
    bias = (1 << (e - 1)) - 1
    sign = (1 << (e + m)) if v < 0 else 0
    a = abs(v)
    if a == 0:
        return sign
    qmin = 1 - bias - m
    q = max(int(mpmath.floor(mpmath.log(a, 2))) - m, qmin)
    # guard against log rounding at binade edges
    while a >= mpmath.mpf(2) ** (q + m + 1):
        q += 1
    while q > qmin and a < mpmath.mpf(2) ** (q + m):
        q -= 1
    n = int(mpmath.nint(a / mpmath.mpf(2) ** q))
    if n == 1 << (m + 1):
        n, q = 1 << m, q + 1
    ef = 0 if n < (1 << m) else q - qmin + 1
    if ef >= (1 << e) - 1:
        return sign | (((1 << e) - 1) << m)
    return sign | (ef << m) | (n & ((1 << m) - 1))


FUNCS = {
    "ex2": lambda x: mpmath.mpf(2) ** x,
    "lg2": lambda x: mpmath.log(x, 2),
    "rsqrt": lambda x: 1 / mpmath.sqrt(x),
    "rcp": lambda x: 1 / x,
    "sqrt": mpmath.sqrt,
    "sin": mpmath.sin,
    "cos": mpmath.cos,
    "sin2pi": lambda x: mpmath.sinpi(2 * x),
    "cos2pi": lambda x: mpmath.cospi(2 * x),
    "tanh": mpmath.tanh,
}
CASES = [("ex2", "f32"), ("ex2", "f16"), ("ex2", "bf16"), ("lg2", "f32"), ("rsqrt", "f32"),
         ("rcp", "f32"), ("sqrt", "f32"),
         ("sin", "f32"), ("cos", "f32"), ("sin2pi", "f32"), ("cos2pi", "f32"),
         ("tanh", "f32"), ("tanh", "f16"), ("tanh", "bf16")]


# The exponent range each function's inputs are drawn from (binary exponents of |x|); a quarter
# of the inputs are random bit patterns instead.
RANGES = {"ex2": (-30, 7), "lg2": (-149, 127), "rsqrt": (-149, 127), "rcp": (-149, 127),
          "sqrt": (-149, 127), "sin": (-20, 12),
          "cos": (-20, 12), "sin2pi": (-20, 6), "cos2pi": (-20, 6), "tanh": (-20, 4)}


def inputs(fn, fmt, rng):
    e, m = FORMATS[fmt]
    w = 1 + e + m
    lo, hi = RANGES[fn]
    xs = set()
    while len(xs) < 300:
        if rng.random() < 0.25:
            b = rng.getrandbits(w)
        else:
            x = mpmath.mpf(1 + rng.random()) * mpmath.mpf(2) ** rng.randint(lo, hi)
            b = encode(fmt, -x if rng.random() < 0.5 else x)
        x = decode(fmt, b)
        if x is None or x == 0:
            continue
        if fn in ("lg2", "rsqrt", "sqrt") and x < 0:
            continue
        if fn == "ex2" and abs(x) > 200:
            continue
        if fn in ("sin2pi", "cos2pi") and x == mpmath.floor(4 * x) / 4:
            continue
        xs.add(b)
    return sorted(xs)


def main():
    rng = random.Random(2026)
    for fn, fmt in CASES:
        for b in inputs(fn, fmt, rng):
            y = encode(fmt, FUNCS[fn](decode(fmt, b)))
            print(f"{fn} {fmt} {b:x} {y:x}")


main()
