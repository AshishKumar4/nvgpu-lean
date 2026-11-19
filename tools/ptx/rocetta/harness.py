"""Writes a C program that runs every scalar function of ROCetta's PTX semantics in C
(`execute_<instruction>` in ptxc.c) on test vectors and prints one line per vector:

    # <function> <argument types...> -> <result types...>
    <function> <argument bits...> = <result bits...>

in hexadecimal, floats as their bit patterns; `pred` is a predicate (`unsigned int`). Usage: python3 harness.py ptxc.c > harness.c.
Functions with pointer or condition-code arguments are left out.
"""
import re
import sys

CTYPES = {
    "int16_t": ("int16_t", 16), "uint16_t": ("uint16_t", 16),
    "int32_t": ("int32_t", 32), "uint32_t": ("uint32_t", 32),
    "int64_t": ("int64_t", 64), "uint64_t": ("uint64_t", 64),
    "float": ("float", 32), "double": ("double", 64), "unsigned int": ("unsigned int", 1),
}

INTS = {
    16: [0, 1, 2, 7, 15, 16, 17, 0x7FFF, 0x8000, 0x8001, 0xFFFE, 0xFFFF, 0x1234, 0xFF, 0x100],
    32: [0, 1, 2, 7, 31, 32, 33, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFE, 0xFFFFFFFF,
         0x12345678, 0x9ABCDEF0, 0x00FF00FF],
    64: [0, 1, 2, 63, 64, 65, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000, 0xFFFFFFFFFFFFFFFE,
         0xFFFFFFFFFFFFFFFF, 0x123456789ABCDEF0, 0x00000000FFFFFFFF, 0xFFFFFFFF00000000, 0x80000000,
         0x100000000],
}
F32 = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x3F000000, 0x3FC00000, 0x40200000,
       0x3DCCCCCD, 0x40490FDB, 0x7F7FFFFF, 0x00800000, 0x00000001, 0x807FFFFF, 0x7F800000,
       0xFF800000, 0x7FC00000, 0x4B000001, 0x3F800001]
F64 = [0x0000000000000000, 0x8000000000000000, 0x3FF0000000000000, 0xBFF0000000000000,
       0x3FE0000000000000, 0x3FF8000000000000, 0x4004000000000000, 0x3FB999999999999A,
       0x400921FB54442D18, 0x7FEFFFFFFFFFFFFF, 0x0010000000000000, 0x0000000000000001,
       0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000000, 0x4330000000000001]
PRED = [0, 1]


def edges(ctype):
    if ctype == "float":
        return F32
    if ctype == "double":
        return F64
    if ctype == "unsigned int":
        return PRED
    return INTS[CTYPES[ctype][1]]


def vectors(params, name):
    """Products of the edge values (thinned for three or more arguments) and seeded random ones."""
    lists = [edges(p) for p in params]
    if len(lists) >= 3:
        lists = [l[: 8 if i < 2 else 4] for i, l in enumerate(lists)]
    out = [[]]
    for l in lists:
        out = [v + [x] for v in out for x in l]
    seed = sum(map(ord, name)) * 2654435761 % 2**64 or 1
    for _ in range(48):
        v = []
        for p in params:
            seed = (seed * 6364136223846793005 + 1442695040888963407) % 2**64
            bits = 1 if p == "unsigned int" else CTYPES[p][1]
            v.append((seed >> 11) % (2 if bits == 1 else 2**bits))
        out.append(v)
    return out


def main(path):
    src = open(path).read()
    sig = re.compile(r"^(struct retval_\w+|[a-z_0-9 ]+?)\s+execute_(\w+)\s*\(([^)]*)\)\s*\{", re.M)
    header = open(path[: -len("ptxc.c")] + "ptxc.h").read()
    structs = dict(re.findall(r"struct (retval_\w+)\s*\{([^}]*)\}", header))
    print('#include <stdio.h>\n#include <string.h>\n#include <stdint.h>\n#include "ptxc.h"\n')
    print("static uint64_t bitsf(float x) { uint32_t b; memcpy(&b, &x, 4); return b; }")
    print("static uint64_t bitsd(double x) { uint64_t b; memcpy(&b, &x, 8); return b; }")
    print("static float f32(uint32_t b) { float x; memcpy(&x, &b, 4); return x; }")
    print("static double f64(uint64_t b) { double x; memcpy(&x, &b, 8); return x; }")
    print("typedef unsigned long long u64;")
    calls = []
    for ret, name, params in sig.findall(src):
        ret = ret.strip()
        ps = [p.strip() for p in params.split(",") if p.strip()]
        types = []
        for p in ps:
            t = p.rsplit(" ", 1)[0].strip()
            if t not in CTYPES or "*" in p:
                types = None
                break
            types.append(t)
        if types is None:
            continue
        if ret.startswith("struct"):
            outs = [f.strip().rsplit(" ", 1)[0] for f in structs[ret.split()[1]].split(";") if f.strip()]
            if any(o not in CTYPES for o in outs):
                continue
        elif ret in CTYPES:
            outs = [ret]
        else:
            continue
        vs = vectors(types, name)
        # Integer division by zero or of the most negative value by -1 traps on the host (C
        # leaves both undefined; PTX leaves division by zero unspecified).
        if name.split("_")[0] in ("div", "rem") and types[0] not in ("float", "double"):
            w = CTYPES[types[0]][1]
            signed = not types[0].startswith("u")
            vs = [v for v in vs if v[1] % 2**w != 0 and
                  not (signed and v[0] % 2**w == 2**(w - 1) and v[1] % 2**w == 2**w - 1)]
        n = max(1, len(types))
        rows = ", ".join("{" + ", ".join(f"0x{x:X}ull" for x in v) + "}" if v else "{0}" for v in vs)

        def conv(t, e):
            if t == "float":
                return f"f32((uint32_t){e})"
            if t == "double":
                return f"f64({e})"
            return f"({t}){e}"

        def fmt(t, e):
            if t == "float":
                return f"(u64)bitsf({e})"
            if t == "double":
                return f"(u64)bitsd({e})"
            return f"(u64)(uint64_t)({e})"
        sig_types = " ".join(t.replace("unsigned int", "pred") for t in types)
        sig_outs = " ".join(t.replace("unsigned int", "pred") for t in outs)
        args = ", ".join(conv(t, f"v[i][{k}]") for k, t in enumerate(types))
        print(f"static void run_{name}(void) {{")
        print(f"  static const u64 v[][{n}] = {{{rows}}};")
        print(f'  printf("# {name} {sig_types} -> {sig_outs}\\n");')
        print(f"  for (unsigned i = 0; i < sizeof v / sizeof v[0]; i++) {{")
        print(f'    printf("{name}");')
        print(f"    for (unsigned k = 0; k < {len(types)}; k++) printf(\" %llx\", v[i][k]);")
        if ret.startswith("struct"):
            print(f"    struct {ret.split()[1]} r = execute_{name}({args});")
            vals = [fmt(t, f"r.out{i}") for i, t in enumerate(outs)]
        else:
            print(f"    {ret} r = execute_{name}({args});")
            vals = [fmt(outs[0], "r")]
        spec = " ".join(["%llx"] * len(vals))
        print(f'    printf(" = {spec}\\n", {", ".join(vals)});')
        print("  }\n}")
        calls.append(f"run_{name}")
    print("int main(void) {")
    for c in calls:
        print(f"  {c}();")
    print("  return 0;\n}")


if __name__ == "__main__":
    main(sys.argv[1])
