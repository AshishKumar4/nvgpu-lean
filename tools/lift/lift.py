#!/usr/bin/env python3
"""Lift one SASS kernel to CUDA C for the accelerated mode.

Input: the kernel as the simulator parses it (`lake exe sass-dump LISTING ELF KERNEL`, JSON
lines), so the lifter reads every instruction exactly as the simulator does. Output: a CUDA C
kernel `lifted` that runs it thread by thread on any CUDA GPU (sm_80 on), with the semantics of
the simulator's definitions (NV/SASS/Sem.lean, Exec.lean, Fast.lean, Async.lean) transcribed
per form; what an older GPU has no instruction for runs in `rt.cuh`.

Machine state per thread: registers `r[0..255]` (RZ reads as 0), predicates `p[0..6]` (PT reads
as true), and its own copy of the uniform registers and predicates (written by every executing
lane with the same value). Per warp, in the host GPU's shared memory: the convergence barrier
registers and the warp's live lanes. A real warp runs the lifted warp, so divergence is the host
GPU's; `BSSY` records the executing lanes and `BSYNC`/`WARPSYNC` are `__syncwarp` on them.
Instructions the simulator runs once per warp (`UTMALDG`, `UTMASTG`, `SYNCS.EXCH`) run on the
lowest executing lane. Warpgroup MMAs and TMA copies complete when issued (their waits are then
no-ops); the simulator's results do not depend on when they complete. Their work is spread over
the issuing warp and the helper CTAs on the GPU's idle SMs (rt.cuh's work pool), and the issuer
waits for it. The lifted CTAs are the grid's first; a CTA's coordinates are `ctaX`, `ctaY`, `ctaZ`.

Usage: lift.py [--threads N] KERNEL.jsonl > lifted.cu   (N: the launch's threads per CTA, 384 if not given;
       nvcc -cubin compiles lifted.cu, which main.cu's lift-run loads)
       lift.py --tables mufu_tables.cuh   (the special-function unit's tables, beside rt.cuh)

A form the lifter does not know is an error naming it: the lifted kernel never guesses.
"""
import json
import re
import sys
from pathlib import Path


class Lift(Exception):
    pass


# ---------------------------------------------------------------------------------------------
# Operands

def is_pred(o):
    return o["k"] == "reg" and o["r"]["f"] in ("p", "up")


def rname(r):
    f, i = r["f"], r["i"]
    if f == "r":
        return "0u" if i == 255 else f"r[{i}]"
    if f == "ur":
        return "0u" if i == 255 else f"UR({i})"
    raise Lift(f"register file {f}")


def rhi(r):
    """The pair's high word."""
    f, i = r["f"], r["i"]
    if f == "r":
        return "0u" if i == 255 else f"r[{i + 1}]"
    if f == "ur":
        return "0u" if i == 255 else f"UR({i + 1})"
    raise Lift(f"register file {f}")


SPECIAL = {
    "SR_TID.X": "tidx", "SR_TID.Y": "tidy", "SR_TID.Z": "tidz",
    "SR_CTAID.X": "ctaX", "SR_CTAID.Y": "ctaY", "SR_CTAID.Z": "ctaZ", "SR_LANEID": "lane",
    "SR_CgaCtaId": "ctx.clusterRank", "SR_SWINHI": "SWINHI", "SRZ": "0u",
    "SR_LTMASK": "((1u << lane) - 1u)", "SR_EQMASK": "(1u << lane)",
    "SR_LEMASK": "(uint32_t)((2ull << lane) - 1ull)", "SR_GEMASK": "(~((1u << lane) - 1u))",
    "SR_GTMASK": "(uint32_t)~((2ull << lane) - 1ull)",
}


def val(o):
    """`Exec.src`: a 32-bit source."""
    k = o["k"]
    if k == "reg":
        if o["r"]["f"] in ("p", "up", "b"):
            raise Lift("predicate as a value")
        x = rname(o["r"])
        if o["not"]:
            return f"(~{x})"
        if o["neg"]:
            return f"(0u - {x})"
        return x
    if k == "imm":
        return f"{o['v'] & 0xffffffff:#x}u"
    if k == "c":
        return cword(o, 0)
    if k == "name":
        s = o["s"]
        if s not in SPECIAL:
            raise Lift(f"special register {s}")
        return SPECIAL[s]
    if k == "neg":
        return f"(0u - {val(o['o'])})"
    if k == "not":
        return f"(~{val(o['o'])})"
    raise Lift(f"operand {k}")


# The kernel's constant banks other than 0 (`Kernel.banks`, the export's header), by bank.
BANKS = {}


def cword(o, add):
    """A constant bank's word at the operand's offset plus `add`: bank 0 is the launch's (`C32`),
    another the kernel's own (`Sem.resolveBanks`: a misaligned or past-the-end word reads 0)."""
    b = o["bank"]
    if b == 0:
        if o["base"] is None:
            return f"C32({o['off'] + add:#x})"
        return f"C32({rname(o['base'])} + {o['off'] + add:#x}u)"
    if b not in BANKS:
        raise Lift(f"constant bank {b}, not the kernel's")
    if o["base"] is None:
        a = o["off"] + add
        ws = BANKS[b]
        return f"{ws[a // 4] if a % 4 == 0 and 0 <= a < 4 * len(ws) else 0:#x}u"
    return f"CBANK(bank{b}, {len(BANKS[b])}u, {rname(o['base'])} + {o['off'] + add:#x}u)"


def valN(o):
    """`Exec.srcN`: a source as a natural number below 2^33 (`-R` is 2^32 - R)."""
    k = o["k"]
    if k == "reg":
        if o["neg"] and not o["not"]:
            return f"(0x100000000ull - (uint64_t){rname(o['r'])})"
        return f"(uint64_t){val(o)}"
    if k == "neg":
        return f"(0x100000000ull - (uint64_t){val(o['o'])})"
    if k == "not":
        return f"(uint64_t)(~{val(o['o'])})"
    return f"(uint64_t){val(o)}"


def hi(o):
    """`Exec.srcHi`."""
    k = o["k"]
    if k == "reg":
        return rhi(o["r"])
    if k == "c":
        return cword(o, 4)
    if k == "imm":
        return f"{(o['v'] >> 32) & 0xffffffff:#x}u"
    raise Lift(f"high word of {k}")


def val64(o):
    """`Exec.src64`."""
    k = o["k"]
    if k == "reg":
        x = f"U64({rname(o['r'])}, {rhi(o['r'])})"
        if o["not"]:
            return f"(~{x})"
        if o["neg"]:
            return f"(0ull - {x})"
        return x
    if k == "imm":
        return f"(uint64_t)(int64_t)(int32_t){o['v'] & 0xffffffff:#x}u"
    return f"U64({val(o)}, {hi(o)})"


def pred(o):
    """`Exec.pred`."""
    if o["k"] != "reg":
        raise Lift(f"predicate operand {o['k']}")
    f, i = o["r"]["f"], o["r"]["i"]
    if f == "p":
        x = "true" if i == 7 else f"p[{i}]"
    elif f == "up":
        x = "true" if i == 7 else f"UP({i})"
    elif "SIGN" in o["sfx"]:
        x = f"(({rname(o['r'])} >> 31) == 1u)"
    else:
        raise Lift("register as a predicate")
    return f"(!{x})" if o["not"] else x


def wr(o, e):
    """A 32-bit write."""
    if o["k"] != "reg":
        raise Lift("destination")
    f, i = o["r"]["f"], o["r"]["i"]
    if i == 255 and f in ("r", "ur"):
        return f"(void)({e});"
    if f == "r":
        return f"r[{i}] = {e};"
    if f == "ur":
        return f"URW({i}, {e});"
    raise Lift(f"destination file {f}")


def wr64(o, e):
    if o["k"] != "reg":
        raise Lift("destination")
    f, i = o["r"]["f"], o["r"]["i"]
    if i == 255:
        return f"(void)({e});"
    t = "_w"
    if f == "r":
        return f"{{ uint64_t {t} = {e}; r[{i}] = (uint32_t){t}; r[{i + 1}] = (uint32_t)({t} >> 32); }}"
    if f == "ur":
        return f"{{ uint64_t {t} = {e}; URW({i}, (uint32_t){t}); URW({i + 1}, (uint32_t)({t} >> 32)); }}"
    raise Lift(f"destination file {f}")


def wrP(o, e):
    if o["k"] != "reg":
        raise Lift("predicate destination")
    f, i = o["r"]["f"], o["r"]["i"]
    if i == 7:
        return f"(void)({e});"
    if f == "p":
        return f"p[{i}] = {e};"
    if f == "up":
        return f"UPW({i}, {e});"
    raise Lift("predicate destination file")


def nP(ops, start):
    n = 0
    for o in ops[start:]:
        if is_pred(o):
            n += 1
        else:
            break
    return n


def addr(o):
    """`Exec.addr`: a memory operand's 64-bit address."""
    if o["k"] == "scaled":
        m = o["m"]
        u = f" + (uint64_t){rname(m['ureg'])}" if m["ureg"] else ""
        return f"((uint64_t){rname(m['base'])} * {o['scale']}ull{u} + {m['off'] & 0xffffffffffffffff:#x}ull)"
    if o["k"] != "mem":
        raise Lift(f"address operand {o['k']}")
    b = "0ull"
    if o["base"]:
        r = o["base"]
        if r["i"] == 255:
            b = "0ull"
        elif o["wide"]:
            b = f"U64({rname(r)}, {rhi(r)})"
        else:
            b = f"(uint64_t){rname(r)}"
    u = f" + (uint64_t){rname(o['ureg'])}" if o["ureg"] else ""
    return f"({b}{u} + {o['off'] & 0xffffffffffffffff:#x}ull)"


# ---------------------------------------------------------------------------------------------
# Forms

H = {}


def form(*names):
    def deco(f):
        for n in names:
            H[n] = f
        return f
    return deco


def opname(op):
    return op[1:] if op.startswith("U") and op[1:] in VECTOR_OF_UNIFORM else op


VECTOR_OF_UNIFORM = {"IADD3", "IMAD", "ISETP", "LOP3", "MOV", "SEL", "SHF", "LEA", "PRMT", "LDC"}


@form("NOP", "MEMBAR_CTA_ONLY", "ERRBAR", "CGAERRBAR")
def f_nop(i):
    return ""


@form("EXIT")
def f_exit(i):
    # The warp's last lane out takes the warp out of its cluster barrier's and the work pool's
    # live warps. A predicate operand (`EXIT P1`) exits only the lanes it holds for.
    ops = i["ops"]
    return split(i, [pred(ops[0])] if ops else [], "LIFT_EXIT")


@form("UCGABAR_ARV")
def f_cgaarv(i):
    return "cgaPhase = clusterArrive(ctx, LIVE);"


@form("UCGABAR_WAIT")
def f_cgawait(i):
    # The warp leaves together, as it arrived (`UCGABAR_ARV`).
    return "{ clusterWait(ctx, cgaPhase); __syncwarp(LIVE); act = LIVE; }"


@form("UBLKCP")
def f_ublkcp(i):
    # `NV.SASS.Async.ublkcp`; the size register counts 16-byte units. The warp's lowest executing
    # lane copies, as the simulator runs the instruction once per warp.
    ops, mods = i["ops"], i["mods"]
    a, _ = ur_mem(ops[0])
    b, _ = ur_mem(ops[1])
    c = ops[2]["r"]["i"]
    size = f"16u * UR({c})"
    if mods == ["S", "S"]:
        call = f"bulkCopy(ctx, sharedAt(ctx, UR({a})), sharedAt(ctx, UR({b})), {size}, UR({a + 1}), lane, warp)"
    elif mods[:1] == ["S"]:
        call = f"bulkCopy(ctx, sharedAt(ctx, UR({a})), gptr(ctx, U64(UR({b}), UR({b + 1}))), {size}, UR({a + 1}), lane, warp)"
    elif mods[:2] == ["G", "S"]:
        call = f"bulkCopy(ctx, gptr(ctx, U64(UR({a}), UR({a + 1}))), sharedAt(ctx, UR({b})), {size}, NONE, lane, warp)"
    else:
        raise Lift(f"UBLKCP {mods}")
    return f"if (lane == __ffs(__EXEC__) - 1) {{ {call}; }}"


@form("UBLKRED")
def f_ublkred(i):
    ops, mods = i["ops"], i["mods"]
    a, _ = ur_mem(ops[0])
    b, _ = ur_mem(ops[1])
    c = ops[2]["r"]["i"]
    opn = next((k for k, m in enumerate(["ADD", "MIN", "MAX", "INC", "DEC", "AND", "OR", "XOR"]) if m in mods), None)
    if opn is None:
        raise Lift(f"UBLKRED {mods}")
    ty = next((t for m, t in [("S32", "int"), ("U32", "unsigned"), ("U64", "unsigned long long"),
                              ("S64", "long long"), ("F32", "float"), ("F64", "double")] if m in mods), "unsigned")
    if any(m in mods for m in ("F16", "BF16")) or (ty in ("float", "double") and opn != 0):
        raise Lift(f"UBLKRED {mods}")
    size = f"16u * UR({c})"
    if mods[:1] == ["S"]:
        dst = f"sharedAt(ctx, UR({a}))"
        tail = f" mbAtomic(MB(ctx, UR({a + 1})), 2, 0, (int64_t)({size}), false, &FAULT);"
    else:
        dst = f"gptr(ctx, U64(UR({a}), UR({a + 1})))"
        tail = ""
    return (f"if (lane == __ffs(__EXEC__) - 1) {{ bulkRed<{ty}>(ctx, {dst}, UR({b}), {size}, {opn});{tail} }}")


# The warp's SIMT state: each lane's `act` is the set of lanes executing with it, as the simulator's
# warp has them (`NV.SASS.Warp`): set from the live lanes at the start, split by a branch, an exit
# or a return its lanes take apart, and set again where they reconverge (`BSYNC`, `WARPSYNC`).
# Warp collectives (shuffles, votes, the lowest executing lane) name it as their lanes (`EXEC`):
# the host GPU's `__EXEC__` is the lanes it happens to run together, which independent
# thread scheduling may leave short of the lanes the simulator executes.

def split(i, conds, taken):
    """A control transfer its lanes may take apart: those for which the guard and `conds` hold run
    `taken` (which leaves) with the lanes taking it with them; the others go on without them."""
    cs = list(conds)
    g = i["guard"]
    if g:
        cs.insert(0, pred({"k": "reg", "r": g["r"], "not": g["not"], "neg": False, "abs": False, "sfx": []}))
    if not cs:
        return taken
    return (f"{{ const bool _c = {' && '.join(cs)}; const uint32_t _cm = __ballot_sync(act, _c); "
            f"if (_c) {{ act = _cm; {taken} }} act &= ~_cm; }}")


@form("BRA")
def f_bra(i):
    t = i["ops"][-1]
    if t["k"] != "target":
        raise Lift("branch target")
    lab = f"L_{t['pc']:x}"
    if "DIV" in i["mods"]:
        # Taken when lanes of the mask are live but not executing here (the warp is diverged).
        return split(i, [f"(({val(i['ops'][0])} & LIVE & ~act) != 0u)"], f"goto {lab};")
    if "ANY" in i["mods"]:
        # `@P BRA.U.ANY target`: every lane here goes when the guard holds in any of them.
        g = i["guard"]
        if not g:
            return f"goto {lab};"
        o = {"k": "reg", "r": g["r"], "not": g["not"], "neg": False, "abs": False, "sfx": []}
        return f"if (__any_sync(act, {pred(o)})) goto {lab};"
    return split(i, [pred(o) for o in i["ops"] if is_pred(o)], f"goto {lab};")


# Calls: the return address (the call's next instruction, relative to the function) is in a
# register beforehand; `RET.REL.NODEC Ra base` goes to base + Ra. The return sites are the
# instructions after each call (RET_SITES, set by `main`).
RET_SITES = set()


@form("CALL")
def f_call(i):
    t = i["ops"][0]
    if "ABS" in i["mods"] and t["k"] == "reg":
        # `CALL.ABS.NOINC Rn`: a call to one of the module's external functions (`printf`'s
        # `vprintf`, the device runtime's), which the lifted kernel does not make: the launch faults
        # when a thread reaches one (the assertion paths that hold them seldom run).
        return split(i, [], "{ atomicOr(&FAULT, 128u); LIFT_EXIT }")
    if t["k"] != "target" or "REL" not in i["mods"]:
        raise Lift("CALL form")
    return split(i, [], f"goto L_{t['pc']:x};")


@form("LEPC")
def f_lepc(i):
    # `LEPC Rd[, target]`: a code address, which only an external call (`CALL.ABS`, not lifted)
    # takes as its return address. The module's code offset is not known here: the offset within
    # the kernel.
    o = i["ops"]
    t = o[1] if len(o) > 1 else None
    pc = t["pc"] if t and t["k"] == "target" else (t["v"] if t and t["k"] == "imm" else i["pc"])
    return wr64(o[0], f"{pc:#x}ull")


@form("RET")
def f_ret(i):
    ops = i["ops"]
    if len(ops) != 2 or ops[1]["k"] != "target":
        raise Lift("RET form")
    cases = " ".join(f"case {pc:#x}u: goto L_{pc:x};" for pc in sorted(RET_SITES))
    # Lanes returning to different sites go apart.
    return split(i, [], f"{{ const uint32_t _t = {ops[1]['pc']:#x}u + {val(ops[0])}; act = __match_any_sync(act, _t); "
                        f"switch (_t) {{ {cases} default: __trap(); }} }}")


@form("BSSY")
def f_bssy(i):
    b = i["ops"][0]["r"]["i"]
    if not HAS_BREAK:
        return f"BREG({b}) = act;"
    # The barrier's shared state cleared by one lane before any of them arrives or breaks out.
    return (f"{{ const uint32_t _m = act; BREG({b}) = _m; if (lane == __ffs(_m) - 1u) "
            f"{{ s_bar[warp][{b}][0] = 0u; s_bar[warp][{b}][1] = 0u; }} __syncwarp(_m); }}")


@form("BREAK")
def f_break(i):
    # `BREAK Bb`: the executing lanes out of the barrier's mask (`bsync`).
    o = i["ops"][0]
    if o["k"] != "reg" or o["r"]["f"] != "b":
        raise Lift("BREAK operand")
    return f"atomicOr(&s_bar[warp][{o['r']['i']}][1], 1u << lane);"


@form("BSYNC")
def f_bsync(i):
    b = i["ops"][0]["r"]["i"]
    if HAS_BREAK:
        return f"act = bsync(s_bar[warp][{b}], BREG({b}), &LIVE, lane);"
    return f"{{ __syncwarp(BREG({b}) & LIVE); act = BREG({b}) & LIVE; }}"


@form("WARPSYNC")
def f_warpsync(i):
    if "ALL" in i["mods"]:
        return "{ __syncwarp(LIVE); act = LIVE; }"
    # `WARPSYNC.COLLECTIVE mask, end`: the region's lanes together.
    return f"{{ const uint32_t _w = {val(i['ops'][0])} & LIVE; __syncwarp(_w); act = _w; }}"


@form("ENDCOLLECTIVE")
def f_endc(i):
    return ""


@form("NANOSLEEP")
def f_nanosleep(i):
    return "__nanosleep(64);"


@form("MEMBAR", "FENCE")
def f_fence(i):
    return "__threadfence();"


@form("DEPBAR")
def f_depbar(i):
    # Bulk copies complete when issued.
    return ""


@form("UTMACMDFLUSH", "UTMACCTL")
def f_tmanop(i):
    return ""


@form("WARPGROUP")
def f_warpgroup(i):
    # Warpgroup MMAs complete when issued; the warpgroup's warps meet at their MMA (`hgmma`).
    return ""


@form("USETMAXREG")
def f_setmaxreg(i):
    if "TRY_ALLOC" in i["mods"]:
        return wrP(i["ops"][0], "true")
    return ""


@form("BAR")
def f_bar(i):
    ops = i["ops"]
    bid = val(ops[0])
    if "ARV" in i["mods"]:
        if len(ops) < 2:
            raise Lift("BAR.ARV without a thread count")
        return f'asm volatile("barrier.arrive %0, %1;" :: "r"({bid}), "r"({val(ops[1])}) : "memory");'
    if "SYNC" not in i["mods"]:
        raise Lift("BAR form")
    # `barrier.sync`, not `bar.sync` (which is `.aligned`): a lifted warp may arrive with its lanes
    # apart. Lanes leave a barrier as they arrived, so the warp's live lanes are then made one again,
    # as `BAR.SYNC`'s warp leaves it: the warp-wide instructions after it (`VOTEU`, `REDUX`, an
    # election) see every lane, as the kernel's do.
    sync = "__syncwarp(LIVE);"
    if len(ops) > 1:
        return f'asm volatile("barrier.sync %0, %1;" :: "r"({bid}), "r"({val(ops[1])}) : "memory"); {sync}'
    return f'asm volatile("barrier.sync %0;" :: "r"({bid}) : "memory"); {sync}'


@form("MOV", "UMOV")
def f_mov(i):
    ops = i["ops"]
    if "64" in i["mods"]:
        return wr64(ops[0], val64(ops[1]))
    return wr(ops[0], val(ops[1]))


@form("S2R", "S2UR", "R2UR")
def f_s2r(i):
    ops = i["ops"]
    if i["op"] == "R2UR":
        if is_pred(ops[0]):
            raise Lift("R2UR with a predicate")
        # The first executing lane's value.
        return wr(ops[0], f"__shfl_sync(__EXEC__, {val(ops[1])}, __ffs(__EXEC__) - 1)")
    return wr(ops[0], val(ops[1]))


@form("CS2R")
def f_cs2r(i):
    ops = i["ops"]
    if ops[1] in ({"k": "name", "s": "SR_CLOCKLO"}, {"k": "name", "s": "SR_GLOBALTIMERLO"}):
        # A clock: the host GPU's, which a result compared with the simulator's cannot match.
        global READS_CLOCK
        READS_CLOCK = True
        src = "clock64()" if ops[1]["s"] == "SR_CLOCKLO" else "globaltimer()"
        return wr(ops[0], f"(uint32_t){src}") if "32" in i["mods"] else wr64(ops[0], f"(uint64_t){src}")
    if ops[1] != {"k": "name", "s": "SRZ"}:
        raise Lift("CS2R source")
    if "32" in i["mods"]:
        return wr(ops[0], "0u")
    return wr64(ops[0], "0ull")


@form("LDC", "ULDC", "LDCU")
def f_ldc(i):
    ops = i["ops"]
    if "64" in i["mods"]:
        return wr64(ops[0], f"U64({val(ops[1])}, {hi(ops[1])})")
    narrow = {"U8": (8, False), "S8": (8, True), "U16": (16, False), "S16": (16, True)}
    if len(i["mods"]) == 1 and i["mods"][0] in narrow and ops[1]["k"] == "c":
        # A byte or half from the word holding it, zero- or sign-extended (`Sem`'s `narrow`).
        bits, signed = narrow[i["mods"][0]]
        o = ops[1]
        if o["base"] is None:
            word, sh = cword({**o, "off": o["off"] & ~3}, 0), f"{8 * (o['off'] % 4)}u"
        elif o["bank"] == 0:
            a = f"({rname(o['base'])} + {o['off']:#x}u)"
            word, sh = f"C32({a} & ~3u)", f"(8u * ({a} & 3u))"
        else:
            raise Lift("LDC of a narrow element at a register offset of the kernel's bank")
        v = f"(({word} >> {sh}) & {(1 << bits) - 1:#x}u)"
        return wr(ops[0], f"(uint32_t)(int32_t)(int{bits}_t){v}" if signed else v)
    if i["mods"]:
        raise Lift("LDC width")
    return wr(ops[0], val(ops[1]))


@form("IMAD", "UIMAD")
def f_imad(i):
    ops, mods = i["ops"], i["mods"]
    signed = "U32" not in mods
    if "WIDE" in mods:
        n = nP(ops, 1)
        a, b, c = ops[1 + n], ops[2 + n], ops[3 + n]
        prod = (f"(uint64_t)((int64_t)(int32_t){val(a)} * (int64_t)(int32_t){val(b)})" if signed
                else f"((uint64_t){val(a)} * (uint64_t){val(b)})")
        cin = f" + ({pred(ops[4 + n])} ? 1ull : 0ull)" if "X" in mods else ""
        s = [f"{{ unsigned __int128 _t = (unsigned __int128){prod} + (unsigned __int128)U64({val(c)}, {hi(c)}){cin};",
             wr64(ops[0], "(uint64_t)_t")]
        if n >= 1:
            s.append(wrP(ops[1], "(_t >> 64) != 0"))
        return " ".join(s) + " }"
    if "HI" in mods:
        # `IMAD.HI`: the high word of a·b plus the pair c:c+1 (a constant or immediate c is the low
        # word alone) plus the carry in, with the carry out of the high word.
        n = nP(ops, 1)
        a, b, c = ops[1 + n], ops[2 + n], ops[3 + n]
        prod = (f"(unsigned __int128)(uint64_t)((int64_t)(int32_t){val(a)} * (int64_t)(int32_t){val(b)})" if signed
                else f"(unsigned __int128)((uint64_t){val(a)} * (uint64_t){val(b)})")
        if c["k"] == "reg":
            cv = f"U64({rname(c['r'])}, {rhi(c['r'])})"
            if c["neg"]:
                cv = f"(0ull - {cv})"
        else:
            cv = f"(uint64_t){val(c)}"
        cin = f"(({pred(ops[4 + n])}) ? (1ull << 32) : 0ull)" if "X" in mods else "0ull"
        s = [f"{{ unsigned __int128 _t = (unsigned __int128)(uint64_t){prod} + (unsigned __int128){cv} + {cin};",
             wr(ops[0], "(uint32_t)((uint64_t)_t >> 32)")]
        if n >= 1:
            s.append(wrP(ops[1], "(_t >> 64) != 0"))
        return " ".join(s) + " }"
    if "X" in mods:
        return wr(ops[0], f"{val(ops[1])} * {val(ops[2])} + {val(ops[3])} + ({pred(ops[4])} ? 1u : 0u)")
    return wr(ops[0], f"{val(ops[1])} * {val(ops[2])} + {val(ops[3])}")


@form("VIADD")
def f_viadd(i):
    ops = i["ops"]
    return wr(ops[0], f"{val(ops[1])} + {val(ops[2])}")


@form("IADD3", "UIADD3")
def f_iadd3(i):
    ops, mods = i["ops"], i["mods"]
    if "64" in mods:
        return wr64(ops[0], f"{val64(ops[1])} + {val64(ops[2])} + {val64(ops[3])}")
    n = nP(ops, 1)
    cin = ""
    if "X" in mods:
        cin = f" + ({pred(ops[4 + n])} ? 1ull : 0ull) + ({pred(ops[5 + n])} ? 1ull : 0ull)"
    s = [f"{{ uint64_t _t = {valN(ops[1 + n])} + {valN(ops[2 + n])} + {valN(ops[3 + n])}{cin};",
         wr(ops[0], "(uint32_t)_t")]
    if n >= 1:
        s.append(wrP(ops[1], "(_t >> 32) >= 1"))
    if n >= 2:
        s.append(wrP(ops[2], "(_t >> 32) >= 2"))
    return " ".join(s) + " }"


@form("LEA", "ULEA")
def f_lea(i):
    ops, mods = i["ops"], i["mods"]
    n = nP(ops, 1)
    if "HI" in mods:
        sx = "SX32" in mods
        a, b = ops[1 + n], ops[2 + n]
        c = f"(((int32_t){val(a)} < 0) ? 0xffffffffu : 0u)" if sx else val(ops[3 + n])
        sh = val(ops[3 + n] if sx else ops[4 + n])
        cin = ""
        if "X" in mods:
            cin = f" + ({pred(ops[4 + n] if sx else ops[5 + n])} ? 1ull : 0ull)"
        s = [f"{{ uint64_t _t = (uint64_t){val(b)} + (uint64_t)shf(true, false, true, true, {val(a)}, {c}, {sh}){cin};",
             wr(ops[0], "(uint32_t)_t")]
        if n >= 1:
            s.append(wrP(ops[1], "(_t >> 32) != 0"))
        return " ".join(s) + " }"
    if "X" in mods:
        raise Lift("LEA.X")
    # `-a` is negated after the shift (`Exec.srcN`): a shifted 0 carries out.
    a = ops[1 + n]
    neg = (a["k"] == "reg" and a["neg"] and not a["not"]) or a["k"] == "neg"
    shifted = f"(uint64_t)(uint32_t)(_k >= 64u ? 0ull : ({valN(a)} << _k))"
    if neg:
        shifted = f"(({shifted}) == 0ull ? 0x100000000ull : ({shifted}))"
    s = [f"{{ uint32_t _k = {val(ops[3 + n])}; uint64_t _t = {shifted} + {valN(ops[2 + n])};",
         wr(ops[0], "(uint32_t)_t")]
    if n >= 1:
        s.append(wrP(ops[1], "(_t >> 32) != 0"))
    return " ".join(s) + " }"


@form("SHF", "USHF")
def f_shf(i):
    ops, mods = i["ops"], i["mods"]
    signed = "S32" in mods or "S64" in mods
    wide = "U64" in mods or "S64" in mods
    left = "L" in mods
    hiw = "HI" in mods
    b = lambda x: "true" if x else "false"
    return wr(ops[0], f"shf({b(left)}, {b(signed)}, {b(wide)}, {b(hiw)}, {val(ops[1])}, {val(ops[3])}, {val(ops[2])})")


@form("LOP3", "ULOP3")
def f_lop3(i):
    ops = i["ops"]
    if is_pred(ops[0]):
        return (f"{{ uint32_t _r = lop3({val(ops[2])}, {val(ops[3])}, {val(ops[4])}, {val(ops[5])}); "
                + wrP(ops[0], "_r != 0u") + " " + wr(ops[1], "_r") + " }")
    return wr(ops[0], f"lop3({val(ops[1])}, {val(ops[2])}, {val(ops[3])}, {val(ops[4])})")


@form("PLOP3", "UPLOP3")
def f_plop3(i):
    ops = i["ops"]
    bit = lambda o: f"({pred(o)} ? 1u : 0u)"
    a, b, c = bit(ops[2]), bit(ops[3]), bit(ops[4])
    return (f"{{ uint32_t _a = {a}, _b = {b}, _c = {c}; "
            + wrP(ops[0], f"(lop3(_a, _b, _c, {val(ops[5])}) & 1u) == 1u") + " "
            + wrP(ops[1], f"(lop3(_a, _b, _c, {val(ops[6])}) & 1u) == 1u") + " }")


@form("SEL", "USEL")
def f_sel(i):
    ops = i["ops"]
    if "64" in i["mods"]:
        raise Lift("SEL.64")
    return wr(ops[0], f"({pred(ops[3])} ? {val(ops[1])} : {val(ops[2])})")


@form("PRMT", "UPRMT")
def f_prmt(i):
    ops = i["ops"]
    if i["mods"]:
        raise Lift("PRMT mode")
    return wr(ops[0], f"prmt({val(ops[1])}, {val(ops[3])}, {val(ops[2])})")


def cmp_expr(c, signed, a, b):
    if signed:
        a, b = f"(int32_t){a}", f"(int32_t){b}"
    return {"EQ": f"({a} == {b})", "NE": f"({a} != {b})", "LT": f"({a} < {b})",
            "LE": f"({a} <= {b})", "GT": f"({a} > {b})", "GE": f"({a} >= {b})",
            "T": "true", "F": "false"}[c]


BOP = {"AND": "&&", "OR": "||", "XOR": "!="}


@form("ISETP", "UISETP")
def f_isetp(i):
    ops, mods = i["ops"], i["mods"]
    if "U64" in mods or "S64" in mods:
        raise Lift("ISETP.64")
    c = mods[0]
    bop = next(m for m in mods if m in BOP)
    signed = "U32" not in mods
    a, b = val(ops[2]), val(ops[3])
    if "EX" in mods:
        low = pred(ops[5])
        lt, eq, gt = cmp_expr("LT", signed, "_a", "_b"), "(_a == _b)", cmp_expr("GT", signed, "_a", "_b")
        r = {"LT": f"({lt} || ({eq} && {low}))", "LE": f"({lt} || ({eq} && {low}))",
             "GT": f"({gt} || ({eq} && {low}))", "GE": f"({gt} || ({eq} && {low}))",
             "EQ": f"({eq} && {low})", "NE": f"(!{eq} || {low})"}[c]
    else:
        r = cmp_expr(c, signed, "_a", "_b")
    pc = pred(ops[4])
    o = BOP[bop]
    return (f"{{ uint32_t _a = {a}, _b = {b}; bool _r = {r}, _c = {pc}; "
            + wrP(ops[0], f"(_r {o} _c)") + " " + wrP(ops[1], f"(!_r {o} _c)") + " }")


@form("ELECT")
def f_elect(i):
    ops = i["ops"]
    s = ["{ uint32_t _m = __EXEC__; uint32_t _l = __ffs(_m) - 1;", wrP(ops[0], "lane == _l")]
    s.append(wr(ops[1], "_l"))
    return " ".join(s) + " }"


@form("VOTEU", "VOTE")
def f_vote(i):
    ops, mods = i["ops"], i["mods"]
    n = len(ops)
    ps, pd = ops[n - 1], ops[n - 2]
    s = [f"{{ uint32_t _m = __EXEC__; uint32_t _b = __ballot_sync(_m, {pred(ps)});"]
    if "ANY" in mods:
        red = "_b != 0u"
    elif "ALL" in mods:
        red = "_b == _m"
    elif "EQ" in mods:
        red = "(_b == 0u || _b == _m)"
    else:
        raise Lift("VOTE mode")
    if n == 3:
        s.append(wr(ops[0], "_b"))
    s.append(wrP(pd, red))
    return " ".join(s) + " }"


@form("SHFL")
def f_shfl(i):
    ops, mods = i["ops"], i["mods"]
    mode = mods[0]
    b, c, a = val(ops[3]), val(ops[4]), val(ops[2])
    j = {"UP": "((int)lane - (int)_bv)", "DOWN": "((int)lane + (int)_bv)",
         "BFLY": "(int)(lane ^ _bv)", "IDX": "(int)((lane & _mask) | (_bv & (0x1fu ^ _mask)))"}[mode]
    ok = {"UP": "(_j >= (int)_max)", "DOWN": "(_j <= (int)_max)", "BFLY": "(_j <= (int)_max)",
          "IDX": "(_j <= (int)_max)"}[mode]
    return (f"{{ uint32_t _bv = {b} & 0x1fu, _cc = {c}; uint32_t _cv = _cc & 0x1fu, _mask = (_cc >> 8) & 0x1fu;"
            f" uint32_t _max = (lane & _mask) | (_cv & (0x1fu ^ _mask)); int _j = {j}; bool _ok = {ok};"
            f" uint32_t _v = __shfl_sync(__EXEC__, {a}, _ok ? _j : (int)lane); "
            + wrP(ops[0], "_ok") + " " + wr(ops[1], "_v") + " }")


# Floats: IEEE round to nearest even; `.FTZ` flushes subnormal inputs and results (`fadd32`…).

def fsrc(o):
    """`Float.src` with `0x80000000`: a binary32 source, its sign modifiers on the sign bit (a
    negated constant, `-c[…]`, flips its sign: not the integer negation `val` takes)."""
    if o["k"] == "reg":
        x = rname(o["r"])
        if o["abs"]:
            x = f"({x} & 0x7fffffffu)"
        if o["neg"]:
            x = f"({x} ^ 0x80000000u)"
        return x
    if o["k"] == "neg":
        return f"({fsrc(o['o'])} ^ 0x80000000u)"
    return val(o)


def plain_float(i, allowed=("FTZ",)):
    extra = [m for m in i["mods"] if m not in allowed]
    if extra:
        raise Lift(f"{i['op']} modifiers {extra}")
    return "true" if "FTZ" in i["mods"] else "false"


ROUND = {"RN": "rn", "RP": "ru", "RM": "rd", "RZ": "rz"}


def arith32(i, name, n):
    """`FADD`/`FMUL`/`FFMA` with `.FTZ`, `.SAT` and a rounding (`.RP`, `.RM`, `.RZ`; round to
    nearest even without one): CUDA's `__f{add,mul,maf}_r?`, inputs and result flushed under
    `.FTZ`, the result then clamped to [0, 1] under `.SAT` (`sat32`)."""
    mods = i["mods"]
    rnd = [m for m in mods if m in ROUND]
    extra = [m for m in mods if m not in ROUND and m not in ("FTZ", "SAT")]
    if extra or len(rnd) > 1:
        raise Lift(f"{i['op']} modifiers {extra or rnd}")
    r = ROUND[rnd[0]] if rnd else "rn"
    ftz = "FTZ" in mods
    o = i["ops"]
    srcs = [fsrc(x) for x in o[1:1 + n]]
    if ftz:
        srcs = [f"flush32({x})" for x in srcs]
    e = f"__float_as_uint(__{name}_{r}({', '.join(f'__uint_as_float({x})' for x in srcs)}))"
    if ftz:
        e = f"flush32({e})"
    return wr(o[0], f"sat32({e})" if "SAT" in mods else e)


@form("FFMA")
def f_ffma(i):
    return arith32(i, "fmaf", 3)


@form("FMUL")
def f_fmul(i):
    return arith32(i, "fmul", 2)


@form("FADD")
def f_fadd(i):
    return arith32(i, "fadd", 2)


@form("FADD2", "FMUL2", "FFMA2")
def f_float2(i):
    # Two binary32 lanes to the pair at the destination (`NV.SASS.Sem`): a `.F32x2` source is its
    # pair (`.LO_HI` swapped), a `.F32` one a register for both.
    mods = i["mods"]
    rnd = [m for m in mods if m in ROUND]
    extra = [m for m in mods if m not in ROUND and m != "FTZ"]
    if extra or len(rnd) > 1:
        raise Lift(f"{i['op']} modifiers {extra or rnd}")
    r = ROUND[rnd[0]] if rnd else "rn"
    ftz = "FTZ" in mods
    o = i["ops"]
    def lanes(x):
        if x["k"] != "reg" or x["abs"] or x["neg"]:
            raise Lift(f"{i['op']} source")
        sfx = x.get("sfx", [])
        lo, hi = rname(x["r"]), rhi(x["r"])
        if "F32x2" in sfx:
            return (hi, lo) if "LO_HI" in sfx else (lo, hi)
        if "F32" in sfx:
            return (lo, lo)
        raise Lift(f"{i['op']} source")
    n = 3 if i["op"] == "FFMA2" else 2
    name = {"FADD2": "fadd", "FMUL2": "fmul", "FFMA2": "fmaf"}[i["op"]]
    srcs = [lanes(x) for x in o[1:1 + n]]
    def lane(k):
        xs = [f"flush32({s[k]})" if ftz else s[k] for s in srcs]
        e = f"__float_as_uint(__{name}_{r}({', '.join(f'__uint_as_float({x})' for x in xs)}))"
        return f"(uint64_t)({f'flush32({e})' if ftz else e})"
    return wr64(o[0], f"{lane(0)} | ({lane(1)} << 32)")


@form("DADD", "DMUL", "DFMA")
def f_dfloat(i):
    # Binary64 register pairs, round to nearest even only.
    if i["mods"]:
        raise Lift(f"{i['op']} modifiers {i['mods']}")
    o = i["ops"]
    def d(x):
        if x["k"] == "reg":
            v = f"__hiloint2double((int){rhi(x['r'])}, (int){rname(x['r'])})"
            if x["abs"]:
                v = f"fabs({v})"
            if x["neg"]:
                v = f"(-{v})"
            return v
        if x["k"] == "imm":
            return f"__longlong_as_double({x['v']:#x}ll)"
        # A constant pair (`Float.src64`), `-c[…]` with its sign flipped.
        if x["k"] == "c":
            return f"__hiloint2double((int){cword(x, 4)}, (int){cword(x, 0)})"
        if x["k"] == "neg" and x["o"]["k"] == "c":
            return f"(-__hiloint2double((int){cword(x['o'], 4)}, (int){cword(x['o'], 0)}))"
        raise Lift("binary64 source")
    if i["op"] == "DFMA":
        e = f"__fma_rn({d(o[1])}, {d(o[2])}, {d(o[3])})"
    else:
        e = f"__d{i['op'][1:].lower()}_rn({d(o[1])}, {d(o[2])})"
    return wr64(o[0], f"(uint64_t)__double_as_longlong({e})")


@form("FCHK")
def f_fchk(i):
    # The simulator's condition (`NV.SASS.Sem`, inferred there): the fast path only for normal (or
    # zero) a and normal b whose exponents and their difference stay within 100 of 0.
    o = i["ops"]
    if i["mods"]:
        raise Lift(f"FCHK modifiers {i['mods']}")
    return (f"{{ uint32_t _a = {fsrc(o[1])}, _b = {fsrc(o[2])};"
            f" int _ea = (int)((_a >> 23) & 255u) - 127, _eb = (int)((_b >> 23) & 255u) - 127;"
            f" bool _na = ((_a >> 23) & 255u) != 0u && ((_a >> 23) & 255u) != 255u;"
            f" bool _nb = ((_b >> 23) & 255u) != 0u && ((_b >> 23) & 255u) != 255u;"
            f" bool _fast = _nb && ((_a & 0x7fffffffu) == 0u || (_na && _ea >= -100 && _ea <= 100 &&"
            f" _eb >= -100 && _eb <= 100 && _ea - _eb <= 100 && _eb - _ea <= 100));"
            f" {wrP(o[0], '!_fast')} }}")


@form("FMNMX")
def f_fmnmx(i):
    # `FMNMX Rd, a, b, P`: the minimum when P holds, else the maximum; NaN-ignoring, -0 below +0
    # (NV.Num.min/max, as fminf/fmaxf), inputs flushed under `.FTZ`.
    # `.NAN`: a NaN operand gives the canonical NaN.
    nan = "NAN" in i["mods"]
    ftz = plain_float(i, ("FTZ", "NAN"))
    o = i["ops"]
    fl = (lambda x: f"flush32({x})") if ftz == "true" else (lambda x: x)
    x, y = fl(fsrc(o[1])), fl(fsrc(o[2]))
    e = (f"({pred(o[3])} ? __float_as_uint(fminf(__uint_as_float(_x), __uint_as_float(_y)))"
         f" : __float_as_uint(fmaxf(__uint_as_float(_x), __uint_as_float(_y))))")
    if nan:
        e = f"(((_x & 0x7fffffffu) > 0x7f800000u || (_y & 0x7fffffffu) > 0x7f800000u) ? 0x7fffffffu : {e})"
    return f"{{ const uint32_t _x = {x}, _y = {y}; {wr(o[0], e)} }}"


@form("FSEL")
def f_fsel(i):
    o = i["ops"]
    return wr(o[0], f"({pred(o[3])} ? {fsrc(o[1])} : {fsrc(o[2])})")


@form("MUFU")
def f_mufu(i):
    fn = {"EX2": "mufuEx2", "RCP": "mufuRcp", "RSQ": "mufuRsq", "LG2": "mufuLg2", "SQRT": "mufuSqrt"}.get(i["mods"][0] if i["mods"] else "")
    if fn is None or len(i["mods"]) != 1:
        raise Lift(f"MUFU {i['mods']}")
    o = i["ops"]
    return wr(o[0], f"{fn}({fsrc(o[1])})")


@form("F2FP")
def f_f2fp(i):
    # `F2FP.{BF16|F16}.F32.PACK_AB Rd, Ra, Rb`: Ra to the high half, Rb to the low, to nearest even.
    mods = i["mods"]
    if "PACK_AB" not in mods or "RELU" in mods:
        raise Lift(f"F2FP {mods}")
    o = i["ops"]
    if "BF16" in mods:
        cv = "__bfloat16_as_ushort(__float2bfloat16_rn(__uint_as_float({})))"
    elif "F16" in mods or mods == ["PACK_AB"]:
        cv = "__half_as_ushort(__float2half_rn(__uint_as_float({})))"
    else:
        raise Lift(f"F2FP {mods}")
    return wr(o[0], f"(((uint32_t){cv.format(fsrc(o[1]))} << 16) | (uint32_t){cv.format(fsrc(o[2]))})")


@form("P2R")
def f_p2r(i):
    # `P2R Rd, PR, Ra, mask`: Rd = (Ra & ~mask) | (P0-P6 as bits & mask).
    o = i["ops"]
    bits = " | ".join(f"(p[{b}] ? {1 << b}u : 0u)" for b in range(7))
    return wr(o[0], f"(({val(o[2])} & ~{val(o[3])}) | (({bits}) & {val(o[3])}))")


@form("R2P")
def f_r2p(i):
    o = i["ops"]
    return (f"{{ uint32_t _a = {val(o[1])}, _m = {val(o[2])}; "
            + " ".join(f"if ((_m >> {b}) & 1u) p[{b}] = ((_a >> {b}) & 1u) != 0u;" for b in range(7)) + " }")


@form("BREV")
def f_brev(i):
    o = i["ops"]
    return wr(o[0], f"__brev({val(o[1])})")


@form("YIELD", "PREEXIT", "ACQBULK", "UTMAPF")
def f_yield(i):
    # Scheduling hints, programmatic dependent launch's trigger (`PREEXIT`, `griddepcontrol.
    # launch_dependents`) and tensor-map prefetches (`UTMAPF`): no effect on a launch's results.
    return ""


def hpair(o):
    """`Float.src` with `0x80008000` and `Float.halves`: a packed pair's 32 bits, its signs and
    `.H0_H0`/`.H1_H1`/`.H1` selection applied."""
    if o["k"] == "reg":
        x = rname(o["r"])
        if o["abs"]:
            x = f"({x} & 0x7fff7fffu)"
        if o["neg"]:
            x = f"({x} ^ 0x80008000u)"
        sfx = o.get("sfx", [])
        if "H0_H0" in sfx:
            return f"((({x}) & 0xffffu) * 0x10001u)"
        if "H1_H1" in sfx:
            return f"((({x}) >> 16) * 0x10001u)"
        if "H1" in sfx:
            return f"(({x}) >> 16)"
        if sfx:
            raise Lift(f"half selection {sfx}")
        return x
    if o["k"] == "neg":
        return f"({val(o['o'])} ^ 0x80008000u)"
    return val(o)


@form("HFMA2", "HADD2", "HMUL2")
def f_hfma2(i):
    # Per half, round to nearest even (`NV.Num.fma` on binary16 or bfloat16); `.MMA` is the pipe.
    mods = [m for m in i["mods"] if m != "MMA"]
    bf = "BF16_V2" in mods
    mods = [m for m in mods if m != "BF16_V2"]
    o = i["ops"]
    # `.F32` (`HADD2.F32 Rd, -RZ, Ra.H0_H0`): the low halves' sum as binary32.
    if i["op"] == "HADD2" and mods == ["F32"] and not bf:
        lo = lambda x: f"__half2float(__ushort_as_half((unsigned short)({hpair(x)} & 0xffffu)))"
        return wr(o[0], f"__float_as_uint(__fadd_rn({lo(o[1])}, {lo(o[2])}))")
    if mods:
        raise Lift(f"{i['op']} modifiers {mods}")
    ty, cv, back = (("__nv_bfloat162", "__halves2bfloat162", "bf2u") if bf else ("__half2", "__halves2half2", "h2u"))
    f = {"HFMA2": "__hfma2", "HADD2": "__hadd2", "HMUL2": "__hmul2"}[i["op"]]
    args = ", ".join(f"u2{'bf' if bf else 'h'}({hpair(x)})" for x in o[1:4 if i["op"] == "HFMA2" else 3])
    return wr(o[0], f"{back}({f}({args}))")


@form("ATOMG", "ATOMS", "RED", "REDG")
def f_atom(i):
    # `ATOMG.E.op[.type][.STRONG.scope] [Pd,] Rd, [addr], Rb[, Rc]`, `REDG.E.op[.type] [addr], Rb`:
    # each executing thread's atomic, as the memory system performs it per lane.
    ops = i["ops"]
    mods = i["mods"]
    red = i["op"].startswith("RED")
    a0 = next(k for k, x in enumerate(ops) if x["k"] in ("mem", "scaled"))
    op = next((m for m in mods if m in ("ADD", "MIN", "MAX", "INC", "DEC", "AND", "OR", "XOR", "EXCH", "CAS")), None)
    if op is None or "CAST" in mods:
        raise Lift(f"{i['op']} {mods}")
    wide = "64" in mods
    if "F32" in mods:
        if op != "ADD" or wide:
            raise Lift(f"{i['op']} {mods}")
        ty, cast = "float", lambda v: f"__uint_as_float({v})"
    elif "F64" in mods or "F16x2" in mods or "BF16x2" in mods:
        raise Lift(f"{i['op']} {mods}")
    elif wide:
        ty = "long long" if "S64" in mods else "unsigned long long"
        cast = None
    else:
        ty = "int" if "S32" in mods else "unsigned"
        cast = lambda v: f"({ty}){v}"
    ptr = f"({ty}*)(SH((uint32_t){addr(ops[a0])}))" if i["op"] == "ATOMS" else f"({ty}*)G({addr(ops[a0])})"
    def src(k):
        x = ops[k]
        if wide:
            return f"({ty})U64({rname(x['r'])}, {rhi(x['r'])})" if x["k"] == "reg" and x["r"]["i"] != 255 else f"({ty}){val(x)}"
        return cast(val(x))
    fn = {"ADD": "atomicAdd", "MIN": "atomicMin", "MAX": "atomicMax", "AND": "atomicAnd", "OR": "atomicOr",
          "XOR": "atomicXor", "EXCH": "atomicExch", "INC": "atomicInc", "DEC": "atomicDec", "CAS": "atomicCAS"}[op]
    args = f"{src(a0 + 1)}, {src(a0 + 2)}" if op == "CAS" else src(a0 + 1)
    call = f"{fn}({ptr}, {args})"
    if red:
        return f"{call};"
    d = ops[a0 - 1]
    if d["k"] != "reg" or d["r"]["f"] != "r":
        raise Lift("atomic destination")
    if d["r"]["i"] == 255:
        return f"{call};"
    if wide:
        return f"{{ unsigned long long _v = (unsigned long long){call}; r[{d['r']['i']}] = (uint32_t)_v; r[{d['r']['i'] + 1}] = (uint32_t)(_v >> 32); }}"
    if ty == "float":
        return f"{{ float _v = {call}; r[{d['r']['i']}] = __float_as_uint(_v); }}"
    return wr(d, f"(uint32_t){call}")


@form("CCTL")
def f_cctl(i):
    # `CCTL.IVALL`: the L1's lines invalidated, which the model's results do not depend on.
    if i["mods"] != ["IVALL"]:
        raise Lift(f"CCTL {i['mods']}")
    return ""


FCMP = {"EQ": "_a == _b", "NE": "(_a != _b && !_u)", "LT": "_a < _b", "LE": "_a <= _b",
        "GT": "_a > _b", "GE": "_a >= _b", "EQU": "(_a == _b || _u)", "NEU": "(_a != _b)",
        "LTU": "(_a < _b || _u)", "LEU": "(_a <= _b || _u)", "GTU": "(_a > _b || _u)",
        "GEU": "(_a >= _b || _u)", "NUM": "!_u", "NAN": "_u", "T": "true", "F": "false"}


@form("FSETP")
def f_fsetp(i):
    ops, mods = i["ops"], i["mods"]
    fl = (lambda x: f"flush32({x})") if "FTZ" in mods else (lambda x: x)
    c, bop = mods[0], next(m for m in mods if m in BOP)
    o = BOP[bop]
    return (f"{{ float _a = __uint_as_float({fl(fsrc(ops[2]))}), _b = __uint_as_float({fl(fsrc(ops[3]))});"
            f" bool _u = isnan(_a) || isnan(_b); bool _r = {FCMP[c]}, _c = {pred(ops[4])}; "
            + wrP(ops[0], f"(_r {o} _c)") + " " + wrP(ops[1], f"(!_r {o} _c)") + " }")


@form("I2FP")
def f_i2fp(i):
    # `I2FP.F32.{S32|U32}[.RZ|.RM|.RP]`: as `I2F`.
    mods = i["mods"]
    if mods[:1] != ["F32"] or mods[1:2] not in (["U32"], ["S32"]) or any(m not in RND_F for m in mods[2:]):
        raise Lift(f"I2FP {mods}")
    return f_i2f({**i, "mods": mods[1:]})


# Conversions as PTX `cvt` (whose rounding, clamping and NaN results `NV.Num.toInt`/`ofInt` model).
RND_I = {"TRUNC": "rzi", "FLOOR": "rmi", "CEIL": "rpi"}
RND_F = {"RZ": "rz", "RM": "rm", "RP": "rp"}
INTS = {"S32": ("s32", 32), "U32": ("u32", 32), "S64": ("s64", 64), "U64": ("u64", 64),
        "S16": ("s16", 16), "U16": ("u16", 16), "S8": ("s8", 8), "U8": ("u8", 8)}


@form("F2I")
def f_f2i(i):
    # `F2I[.FTZ][.type][.TRUNC|.FLOOR|.CEIL][.NTZ] Rd, Ra` from binary32 (`NV.SASS.Sem`).
    mods = i["mods"]
    if "F64" in mods or any(m not in RND_I and m not in INTS and m not in ("FTZ", "NTZ") for m in mods):
        raise Lift(f"F2I {mods}")
    ty, w = next((INTS[m] for m in mods if m in INTS), ("s32", 32))
    rnd = next((RND_I[m] for m in mods if m in RND_I), "rni")
    ftz = ".ftz" if "FTZ" in mods else ""
    o = i["ops"]
    if w == 16:
        # Clamped to the 16-bit type, then sign- or zero-extended to the register.
        ext = "(uint32_t)(int32_t)(int16_t)_v" if ty == "s16" else "(uint32_t)_v"
        return wr(o[0], f"([&]{{ unsigned short _v; asm(\"cvt.{rnd}{ftz}.{ty}.f32 %0, %1;\" : \"=h\"(_v) : \"f\"(__uint_as_float({fsrc(o[1])}))); return {ext}; }}())")
    if w == 64:
        return wr64(o[0], f"([&]{{ uint64_t _v; asm(\"cvt.{rnd}{ftz}.{ty}.f32 %0, %1;\" : \"=l\"(_v) : \"f\"(__uint_as_float({fsrc(o[1])}))); return _v; }}())")
    if w < 32:
        raise Lift(f"F2I {mods}")
    return wr(o[0], f"([&]{{ uint32_t _v; asm(\"cvt.{rnd}{ftz}.{ty}.f32 %0, %1;\" : \"=r\"(_v) : \"f\"(__uint_as_float({fsrc(o[1])}))); return _v; }}())")


@form("FRND")
def f_frnd(i):
    # `FRND[.TRUNC|.FLOOR|.CEIL][.FTZ][.F64] Rd, a`: a rounded to an integral value (PTX
    # `cvt.rXi.f32.f32`, `.f64.f64` on a register pair), to nearest even without a mode.
    mods = i["mods"]
    if any(m not in RND_I and m not in ("FTZ", "F64") for m in mods):
        raise Lift(f"FRND {mods}")
    rnd = next((RND_I[m] for m in mods if m in RND_I), "rni")
    o = i["ops"]
    if "F64" in mods:
        x = o[1]
        if x["k"] != "reg" or x["abs"] or x["neg"]:
            raise Lift("FRND.F64 source")
        src = f"__hiloint2double((int){rhi(x['r'])}, (int){rname(x['r'])})"
        return wr64(o[0], f"([&]{{ double _v; asm(\"cvt.{rnd}.f64.f64 %0, %1;\" : \"=d\"(_v) : \"d\"({src})); return (uint64_t)__double_as_longlong(_v); }}())")
    ftz = ".ftz" if "FTZ" in mods else ""
    return wr(o[0], f"([&]{{ float _v; asm(\"cvt.{rnd}{ftz}.f32.f32 %0, %1;\" : \"=f\"(_v) : \"f\"(__uint_as_float({fsrc(o[1])}))); return __float_as_uint(_v); }}())")


@form("F2F")
def f_f2f(i):
    # `F2F[.FTZ].dst.src[.RZ|.RM|.RP] Rd, Ra`: between binary64 (a register pair), binary32 and
    # binary16 (the low half), rounded (PTX `cvt`; a widening is exact and takes no mode); and
    # binary32 narrowed to bfloat16 (the low half).
    mods = i["mods"]
    fmts = [m for m in mods if m in ("F64", "F32", "F16", "BF16")]
    if len(fmts) != 2 or any(m not in RND_F and m not in ("FTZ", "F64", "F32", "F16", "BF16") for m in mods):
        raise Lift(f"F2F {mods}")
    dst, srcf = fmts
    width = {"F64": 64, "F32": 32, "F16": 16, "BF16": 16}
    if dst == srcf or ("BF16" in fmts and (fmts != ["BF16", "F32"] or "FTZ" in mods)):
        raise Lift(f"F2F {mods}")
    rnd = "" if width[dst] > width[srcf] else "." + next((RND_F[m] for m in mods if m in RND_F), "rn")
    ftz = ".ftz" if "FTZ" in mods and "F32" in fmts else ""
    o = i["ops"]
    x = o[1]
    if srcf == "F64":
        if x["k"] != "reg" or x["abs"] or x["neg"]:
            raise Lift("F2F.F64 source")
        src, sc = f"__hiloint2double((int){rhi(x['r'])}, (int){rname(x['r'])})", "d"
    elif srcf == "F32":
        src, sc = f"__uint_as_float({fsrc(x)})", "f"
    else:
        src, sc = f"(unsigned short)({val(x)} & 0xffffu)", "h"
    ty = {"F64": "f64", "F32": "f32", "F16": "f16", "BF16": "bf16"}
    asm = f"cvt{rnd}{ftz}.{ty[dst]}.{ty[srcf]} %0, %1;"
    if dst == "F64":
        return wr64(o[0], f"([&]{{ double _v; asm(\"{asm}\" : \"=d\"(_v) : \"{sc}\"({src})); return (uint64_t)__double_as_longlong(_v); }}())")
    if dst == "F32":
        return wr(o[0], f"([&]{{ float _v; asm(\"{asm}\" : \"=f\"(_v) : \"{sc}\"({src})); return __float_as_uint(_v); }}())")
    return wr(o[0], f"([&]{{ unsigned short _v; asm(\"{asm}\" : \"=h\"(_v) : \"{sc}\"({src})); return (uint32_t)_v; }}())")


@form("I2F")
def f_i2f(i):
    # `I2F[.F16|.F64][.type][.RZ|.RM|.RP] Rd, Ra`: to binary32 (binary16 in the low half, binary64
    # in the pair); a 64-bit integer from the pair Ra:Ra+1.
    mods = i["mods"]
    if any(m not in RND_F and m not in INTS and m not in ("F16", "F64", "BF16") for m in mods):
        raise Lift(f"I2F {mods}")
    ty, w = next((INTS[m] for m in mods if m in INTS), ("s32", 32))
    if "BF16" in mods:
        # In software (`i2bf16`): PTX's cvt to bf16 from an integer needs sm_90.
        if w == 8:
            raise Lift(f"I2F {mods}")
        o = i["ops"]
        mode = {"RZ": 1, "RM": 2, "RP": 3}.get(next((m for m in mods if m in RND_F), ""), 0)
        v = f"U64({val(o[1])}, {hi(o[1])})" if w == 64 else f"(uint64_t)(uint32_t){val(o[1])}"
        if w == 16:
            v = f"(uint64_t)(uint16_t){val(o[1])}"
        if ty.startswith("s"):
            sv = f"(int64_t){v}" if w == 64 else f"(int64_t)(int{w}_t){v}"
            return wr(o[0], f"([&]{{ const int64_t _s = {sv}; return i2bf16(_s < 0 ? 0ull - (uint64_t)_s : (uint64_t)_s, _s < 0, {mode}); }}())")
        return wr(o[0], f"i2bf16({v}, false, {mode})")
    if w == 8 or ("F16" in mods and w == 64):
        raise Lift(f"I2F {mods}")
    rnd = next((RND_F[m] for m in mods if m in RND_F), "rn")
    o = i["ops"]
    # A 64-bit integer from the register pair.
    src = f"(unsigned short){val(o[1])}" if w == 16 else f"U64({val(o[1])}, {hi(o[1])})" if w == 64 else val(o[1])
    rc = "h" if w == 16 else "l" if w == 64 else "r"
    if "F64" in mods:
        return wr64(o[0], f"([&]{{ double _v; asm(\"cvt.{rnd}.f64.{ty} %0, %1;\" : \"=d\"(_v) : \"{rc}\"({src})); return (uint64_t)__double_as_longlong(_v); }}())")
    if "F16" in mods:
        return wr(o[0], f"([&]{{ unsigned short _v; asm(\"cvt.{rnd}.f16.{ty} %0, %1;\" : \"=h\"(_v) : \"{rc}\"({src})); return (uint32_t)_v; }}())")
    return wr(o[0], f"([&]{{ uint32_t _v; asm(\"cvt.{rnd}.f32.{ty} %0, %1;\" : \"=r\"(_v) : \"{rc}\"({src})); return _v; }}())")


@form("VIADDMNMX")
def f_viaddmnmx(i):
    # `VIADDMNMX[.U32] Rd, a, b, c, P`: the minimum when P holds, else the maximum, of a + b
    # (wrapping) and c.
    ops, mods = i["ops"], i["mods"]
    if [m for m in mods if m not in ("U32", "S32")] or len(ops) != 5:
        raise Lift(f"VIADDMNMX {mods}")
    t = "uint32_t" if "U32" in mods else "int32_t"
    s, c = f"({t})({val(ops[1])} + {val(ops[2])})", f"({t}){val(ops[3])}"
    return wr(ops[0], f"(uint32_t)({pred(ops[4])} ? min({s}, {c}) : max({s}, {c}))")


@form("VIMNMX", "IMNMX")
def f_vimnmx(i):
    # `[V]IMNMX[.U32] Rd, a, b, P` (`IMNMX` before sm_90) and sm_100a's `VIMNMX[.U32] Rd, PT, PT,
    # a, b, P` (`NV.SASS.Sem`): the minimum when P holds, else the maximum.
    ops, mods = i["ops"], i["mods"]
    if [m for m in mods if m not in ("U32", "S32")]:
        raise Lift(f"{i['op']} {mods}")
    k = 2 if len(ops) == 6 else 0
    if k and not (all(ops[j]["k"] == "reg" and ops[j]["r"]["f"] == "p" and ops[j]["r"]["i"] == 7 and not ops[j]["not"] for j in (1, 2))):
        raise Lift("VIMNMX leading predicates other than PT")
    t = "uint32_t" if "U32" in mods else "int32_t"
    a, b = f"({t}){val(ops[1 + k])}", f"({t}){val(ops[2 + k])}"
    return wr(ops[0], f"(uint32_t)({pred(ops[3 + k])} ? min({a}, {b}) : max({a}, {b}))")


@form("IABS")
def f_iabs(i):
    # `IABS Rd, a`: |a| as a signed word, wrapping (INT_MIN stays INT_MIN).
    o = i["ops"]
    if i["mods"]:
        raise Lift(f"IABS {i['mods']}")
    return f"{{ const uint32_t _v = {val(o[1])}; {wr(o[0], '((int32_t)_v < 0 ? 0u - _v : _v)')} }}"


@form("POPC", "UPOPC")
def f_popc(i):
    # `POPC Rd, Rb`: the number of set bits.
    o = i["ops"]
    return wr(o[0], f"(uint32_t)__popc({val(o[1])})")


@form("FLO", "UFLO")
def f_flo(i):
    # `FLO.U32[.SH]`: the highest set bit's index (`.SH`: its distance from bit 31); all ones for 0.
    mods = i["mods"]
    if "S32" in mods:
        raise Lift("FLO.S32")
    o = i["ops"]
    v = val(o[1])
    e = f"({v} == 0u ? 0xffffffffu : (uint32_t)(31 - __clz({v})))" if "SH" not in mods else f"({v} == 0u ? 0xffffffffu : (uint32_t)__clz({v}))"
    return wr(o[0], e)


@form("REDUX")
def f_redux(i):
    # `REDUX[.op][.S32] URd, Ra`: over the executing lanes (no operation named: AND).
    mods = i["mods"]
    s32 = "S32" in mods
    op = next((m for m in mods if m in ("SUM", "AND", "OR", "XOR", "MIN", "MAX")), "AND")
    fn = {"SUM": "__reduce_add_sync", "AND": "__reduce_and_sync", "OR": "__reduce_or_sync",
          "XOR": "__reduce_xor_sync", "MIN": "__reduce_min_sync", "MAX": "__reduce_max_sync"}[op]
    o = i["ops"]
    arg = f"(int){val(o[1])}" if s32 and op in ("MIN", "MAX") else val(o[1])
    return wr(o[0], f"(uint32_t){fn}(__EXEC__, {arg})")


@form("BPT")
def f_bpt(i):
    # `BPT.TRAP`: the simulator faults the run.
    return "{ atomicOr(&FAULT, 64u); LIFT_EXIT }"


# Memory

def width(mods):
    for m, w in (("U8", (1, False)), ("S8", (1, True)), ("U16", (2, False)), ("S16", (2, True)),
                 ("64", (8, False)), ("128", (16, False))):
        if m in mods:
            return w
    return (4, False)


def load_regs(o, nbytes, signed, ptr, space="g"):
    """A load of `nbytes` into the destination registers. `space`: `g` global (volatile: other CTAs
    write it), `sh` the CTA's shared window (relaxed at CTA scope, served by the L1), `loc` the
    thread's local window (plain)."""
    d = o["r"]["i"]
    if nbytes < 4:
        t = {(1, False): "uint8_t", (1, True): "int8_t", (2, False): "uint16_t", (2, True): "int16_t"}[(nbytes, signed)]
        if space == "sh":
            pt = {(1, False): "u8", (1, True): "s8", (2, False): "u16", (2, True): "s16"}[(nbytes, signed)]
            return wr(o, f"LD_SH(uint32_t, \"{pt}\", {ptr})")
        q = "volatile " if space == "g" else ""
        return wr(o, f"(uint32_t)(int32_t)*(const {q}{t}*)({ptr})")
    words = nbytes // 4
    if d == 255:
        return ""
    if space == "sh":
        return " ".join(f"r[{d + k}] = LD_SH(uint32_t, \"b32\", (const uint32_t*)({ptr}) + {k});" for k in range(words))
    q = "volatile " if space == "g" else ""
    return " ".join(f"r[{d + k}] = ((const {q}uint32_t*)({ptr}))[{k}];" for k in range(words))


def store_regs(o, nbytes, ptr, space="g"):
    d = o["r"]["i"]
    if nbytes < 4:
        t = {1: "uint8_t", 2: "uint16_t"}[nbytes]
        if space == "sh":
            return f"ST_SH(\"{ {1: 'b8', 2: 'b16'}[nbytes] }\", {ptr}, (uint32_t){val(o)});"
        q = "volatile " if space == "g" else ""
        return f"*({q}{t}*)({ptr}) = ({t}){val(o)};"
    words = nbytes // 4
    src = lambda k: '0u' if d == 255 else f'r[{d + k}]'
    if space == "sh":
        return " ".join(f"ST_SH(\"b32\", (uint32_t*)({ptr}) + {k}, {src(k)});" for k in range(words))
    q = "volatile " if space == "g" else ""
    return " ".join(f"((({q}uint32_t*)({ptr}))[{k}] = {src(k)});" for k in range(words))


@form("LDS")
def f_lds(i):
    n, s = width(i["mods"])
    return f"{{ uint8_t* _p = SH((uint32_t){addr(i['ops'][1])}); {load_regs(i['ops'][0], n, s, '_p', 'sh')} }}"


@form("STS")
def f_sts(i):
    n, _ = width(i["mods"])
    return f"{{ uint8_t* _p = SH((uint32_t){addr(i['ops'][0])}); {store_regs(i['ops'][1], n, '_p', 'sh')} }}"


@form("LDL")
def f_ldl(i):
    n, s = width(i["mods"])
    return f"{{ uint8_t* _p = LOC((uint32_t){addr(i['ops'][1])}); {load_regs(i['ops'][0], n, s, '_p', 'loc')} }}"


@form("STL")
def f_stl(i):
    n, _ = width(i["mods"])
    return f"{{ uint8_t* _p = LOC((uint32_t){addr(i['ops'][0])}); {store_regs(i['ops'][1], n, '_p', 'loc')} }}"


@form("LDG")
def f_ldg(i):
    n, s = width(i["mods"])
    return f"{{ uint8_t* _p = G({addr(i['ops'][1])}); {load_regs(i['ops'][0], n, s, '_p')} }}"


@form("LD")
def f_ld(i):
    # Generic: each lane's address resolved through the windows (`GEN`).
    n, s = width(i["mods"])
    return f"{{ uint8_t* _p = GEN({addr(i['ops'][1])}); {load_regs(i['ops'][0], n, s, '_p')} }}"


@form("ST")
def f_st(i):
    n, _ = width(i["mods"])
    return f"{{ uint8_t* _p = GEN({addr(i['ops'][0])}); {store_regs(i['ops'][1], n, '_p')} }}"


@form("STG")
def f_stg(i):
    n, _ = width(i["mods"])
    return f"{{ uint8_t* _p = G({addr(i['ops'][0])}); {store_regs(i['ops'][1], n, '_p')} }}"


# Async

def ur_mem(o):
    if o["k"] != "mem" or not o["base"] or o["base"]["f"] != "ur" or o["wide"] or o["ureg"]:
        raise Lift("[URa+off] operand")
    return o["base"]["i"], o["off"]


@form("SYNCS")
def f_syncs(i):
    ops, mods = i["ops"], i["mods"]
    if "EXCH" in mods:
        r, off = ur_mem(ops[1])
        v = ops[2]["r"]["i"]
        return (f"if (lane == __ffs(__EXEC__) - 1) mbExch(ctx, UR({r}) + {off & 0xffffffff:#x}u, "
                f"U64(UR({v}), UR({v + 1})));")
    if "PHASECHK" in mods:
        # Polled plainly; once the phase has completed, read again with acquire (the copies that
        # completed it may have been written from other SMs: the work pool's helpers).
        # A warp whose wait is not over helps with its CTA's copies meanwhile (`helpCta`).
        return (f"{{ unsigned long long* _w = MB(ctx, (uint32_t){addr(ops[1])}); "
                f"bool _done = (uint32_t)(*(volatile unsigned long long*)_w >> 63) != ({val(ops[2])} >> 31); "
                f"if (_done) (void)ldAcq(_w); "
                f"if (__all_sync(__EXEC__, !_done)) helpCta(ctx, warp, lane, __EXEC__); "
                + wrP(ops[0], "_done") + " }")
    if "ARRIVE" in mods:
        tx = f"{val(ops[2])}" if ops[2]["k"] == "reg" and ops[2]["r"]["i"] != 255 else "0u"
        drop = "true" if "OPTOUT" in mods else "false"
        if "A1T0" in mods:
            kind, n, t = 1, "1u", "0"
        elif "ART0" in mods:
            kind, n, t = 1, tx, "0"
        elif "A0T1" in mods:
            kind, n, t = 1, "0u", "1"
        elif "A0TR" in mods:
            kind, n, t = 1, "0u", f"(int64_t){tx}"
        elif "A0TX" in mods:
            kind, n, t = 2, "0u", f"(int64_t){tx}"
        else:
            kind, n, t = 1, "1u", f"(int64_t){tx}"
        dst = ops[0]
        s = (f"{{ __threadfence(); uint64_t _old = mbAtomic(MB(ctx, (uint32_t){addr(ops[1])}), {kind}, {n}, {t}, {drop}, &FAULT);")
        if dst["k"] == "reg" and dst["r"]["i"] != 255:
            s += " " + wr64(dst, "_old")
        return s + " }"
    raise Lift(f"SYNCS {mods}")


# 5th-generation tensor cores (`NV.SASS.Async`): the MMA and the allocator run on the warp's
# lowest executing lane, as the simulator runs them once per warp; LDTM/STTM per thread.

def tmem_op(o):
    if o["k"] != "udesc" or o["kind"] != "tmem" or o["r"]["f"] != "ur":
        raise Lift("tmem operand")
    return f"(UR({o['r']['i']}) + {o['off'] & 0xffffffff:#x}u)"


@form("HMMA")
def f_hmma(i):
    # `mma.sync` into binary32 from f16, bf16 or tf32 (`hmma`): `Rd, Ra, Rb, Rc`.
    mods = i["mods"]
    k = {"16816": 16, "1688": 8, "1684": 4}.get(mods[0] if mods else "")
    ty = [m for m in mods[1:] if m != "F32"]
    t = ty[0] if ty else "F16"
    # The shapes: m16n8k16 and m16n8k8 of 16-bit inputs, m16n8k8 and m16n8k4 of TF32.
    if "F32" not in mods or len(ty) > 1 or k not in {"F16": (16, 8), "BF16": (16, 8), "TF32": (8, 4)}.get(t, ()):
        raise Lift(f"HMMA {'.'.join(mods)}")
    w, codec, man = {"F16": (16, "codecIeee(5, 10)", 10), "BF16": (16, "codecIeee(8, 7)", 7),
                     "TF32": (32, "codecPad(8, 10, 13)", 10)}[t]
    aw, bw = k * w // 64, k * w // 128
    o = i["ops"]
    if len(o) < 4 or any(x["k"] != "reg" or x["r"]["f"] != "r" for x in o[:4]):
        raise Lift("HMMA operands")
    reg = lambda x, j: "0u" if x["r"]["i"] == 255 else f"r[{x['r']['i'] + j}]"
    rd = o[0]["r"]["i"]
    out = "" if rd == 255 else " ".join(f"r[{rd + j}] = _d[{j}];" for j in range(4))
    return (f"{{ const uint32_t _a[{aw}] = {{{', '.join(reg(o[1], j) for j in range(aw))}}}; "
            f"const uint32_t _b[{bw}] = {{{', '.join(reg(o[2], j) for j in range(bw))}}}; "
            f"const uint32_t _c[4] = {{{', '.join(reg(o[3], j) for j in range(4))}}}; uint32_t _d[4]; "
            f"hmma<{k}, {aw}, {bw}>(ctx, mmaModel(SM, {str(t == 'TF32').lower()}), {w}, {codec}, {man}, "
            f"_a, _b, _c, _d, lane, __EXEC__); {out} }}")


@form("IMMA")
def f_imma(i):
    # `mma.sync` of 8- or 4-bit integers into s32 (`IMMA.{16816|16832|16864}.ta.tb[.SAT] Rd, Ra, Rb,
    # Rc`): exact (`.SAT` clamps to s32), so the hardware's own `mma.sync` gives the simulator's
    # result; the operands are the PTX fragments in order (`Sem.decodeMma`).
    mods = i["mods"]
    shape = {"16816": 16, "16832": 32, "16864": 64}.get(mods[0] if mods else "")
    tys = [m for m in mods[1:] if m != "SAT"]
    bits = {"S8": 8, "U8": 8, "S4": 4, "U4": 4}
    if shape is None or len(tys) != 2 or any(t not in bits for t in tys) or bits[tys[0]] != bits[tys[1]] \
            or shape * bits[tys[0]] not in (128, 256):
        raise Lift(f"IMMA {'.'.join(mods)}")
    aw, bw = shape * bits[tys[0]] // 64, shape * bits[tys[0]] // 128
    o = i["ops"]
    if len(o) < 4 or any(x["k"] != "reg" or x["r"]["f"] != "r" for x in o[:4]):
        raise Lift("IMMA operands")
    reg = lambda x, j: "0u" if x["r"]["i"] == 255 else f"r[{x['r']['i'] + j}]"
    sat = ".satfinite" if "SAT" in mods else ""
    ta, tb = (t.lower() for t in tys)
    n = 4 + aw + bw
    lst = lambda lo, cnt: "{" + ", ".join(f"%{lo + j}" for j in range(cnt)) + "}"
    asm = (f"mma.sync.aligned.m16n8k{shape}.row.col{sat}.s32.{ta}.{tb}.s32 {lst(0, 4)}, {lst(4, aw)}, "
           f"{lst(4 + aw, bw)}, {lst(4 + aw + bw, 4)};")
    ins = ", ".join([f'"r"({reg(o[1], j)})' for j in range(aw)] + [f'"r"({reg(o[2], j)})' for j in range(bw)] +
                    [f'"r"({reg(o[3], j)})' for j in range(4)])
    rd = o[0]["r"]["i"]
    out = "" if rd == 255 else " ".join(f"r[{rd + j}] = _d[{j}];" for j in range(4))
    return (f"{{ uint32_t _d[4]; asm volatile(\"{asm}\" : \"=r\"(_d[0]), \"=r\"(_d[1]), \"=r\"(_d[2]), "
            f"\"=r\"(_d[3]) : {ins}); {out} }}")


@form("LDSM", "STSM")
def f_ldsm(i):
    # `ldmatrix`/`stmatrix` of 16-bit elements, 8 x 8 matrices (`MatMove`), by the whole warp: lane
    # 8j + i gives row i of matrix j (16 bytes); lane t's register j is row 8j + t/4's elements
    # 2(t%4) and 2(t%4) + 1, or, transposed (`MT88`), element t/4 of rows 8j + 2(t%4) and
    # 8j + 2(t%4) + 1. Each row's address is read from its lane by shuffle.
    mods = i["mods"]
    trans = "MT88" in mods
    num = 4 if "4" in mods else 2 if "2" in mods else 1
    if "16" not in mods or not (trans or "M88" in mods) or \
            any(m not in ("16", "M88", "MT88", "2", "4") for m in mods):
        raise Lift(f"{i['op']} {'.'.join(mods)}")
    load = i["op"] == "LDSM"
    ao, ro = (i["ops"][1], i["ops"][0]) if load else (i["ops"][0], i["ops"][1])
    if ro["k"] != "reg" or ro["r"]["f"] != "r":
        raise Lift(f"{i['op']} register")
    d = ro["r"]["i"]
    src = lambda j: "0u" if d == 255 else f"r[{d + j}]"

    def row(lane_expr):
        # The row's shared address (the CTA's window: `ldmatrix` is not cluster-wide).
        return f"(__shfl_sync(0xffffffffu, _a, {lane_expr}) & 0xffffffu)"

    body = []
    for j in range(num):
        if not trans:
            p = f"SH({row(f'{8 * j}u + lane / 4')} + 4u * (lane % 4))"
            body.append(f'_v[{j}] = LD_SH(uint32_t, "b32", {p});' if load else f'ST_SH("b32", {p}, {src(j)});')
            continue
        for e in range(2):
            p = f"SH({row(f'{8 * j + e}u + 2u * (lane % 4)')} + 2u * (lane / 4))"
            body.append(f'_v[{j}] |= LD_SH(uint32_t, "u16", {p}) << {16 * e};' if load
                        else f'ST_SH("b16", {p}, {src(j)} >> {16 * e});')
    decl = ""
    if load:
        decl = f"uint32_t _v[{num}] = {{{', '.join('0u' for _ in range(num))}}}; "
        if d != 255:
            body.append(" ".join(f"r[{d + j}] = _v[{j}];" for j in range(num)))
    return (f"{{ if (__EXEC__ != 0xffffffffu) atomicOr(&FAULT, 32u); else {{ "
            f"const uint32_t _a = (uint32_t){addr(ao)}; {decl}{' '.join(body)} }} }}")


@form("UTCHMMA", "UTCQMMA", "UTCIMMA")
def f_utchmma(i):
    ops, mods = i["ops"], i["mods"]
    # The kind: `.kind::f16` (and tf32), `::f8f6f4`, `::i8` (`utcmma`).
    kind = {"UTCHMMA": 0, "UTCQMMA": 1, "UTCIMMA": 2}[i["op"]]
    if [m for m in mods if m not in ("2CTA", "WS")]:
        raise Lift(f"{i['op']} {mods}")
    ws = "WS" in mods
    # `.WS`'s collector-buffer hints on B (`.B_KEEP`, ...) change no result.
    # A from shared memory (a descriptor) or tensor memory (`aTmem`).
    a_tmem = ops[0]["k"] == "udesc" and ops[0].get("kind") == "tmem"
    if (ops[0]["k"] != "gdesc" and not a_tmem) or ops[1]["k"] != "gdesc" or (ops[0].get("mods") and not a_tmem) or \
            (ops[1].get("mods") and not ws):
        raise Lift(f"{i['op']} operands A and B")
    b = ops[1]["r"]["i"]
    da = "0ull" if a_tmem else f"U64(UR({ops[0]['r']['i']}), UR({ops[0]['r']['i'] + 1}))"
    a_at = tmem_op(ops[0]) if a_tmem else "NONE"
    d = tmem_op(ops[2])
    if ops[3]["k"] != "udesc" or ops[4]["k"] != "udesc" or ops[4]["kind"] != "idesc":
        raise Lift(f"{i['op']} operands")
    idx = ops[4]["r"]["i"]
    rest = ops[5:]
    if rest and rest[0]["k"] == "udesc":
        raise Lift(f"{i['op']} with scale factors")
    # `.WS`'s zero-column mask: a pair of uniform registers before enable-input-d.
    mask = "0ull"
    if ws and rest and rest[0]["k"] == "reg" and rest[0]["r"]["f"] == "ur":
        m = rest[0]["r"]["i"]
        mask = "0ull" if m == 255 else f"U64(UR({m}), UR({m + 1}))"
        rest = rest[1:]
    acc = pred(rest[0]) if rest else "true"
    pair = "true" if "2CTA" in mods else "false"
    # The fourth operand: sparse A's metadata (`tcgen05.mma.sp`), or 0.
    meta = "0u" if ops[3]["r"]["i"] == 255 else tmem_op(ops[3])
    return (f"if (lane == __ffs(__EXEC__) - 1) "
            f"utcmma(ctx, {da}, U64(UR({b}), UR({b + 1})), {d}, UR({idx}), {acc}, {pair}, "
            f"lane, warp, {str(ws).lower()}, {mask}, {meta}, {kind}u, {a_at});")


@form("UTCBAR")
def f_utcbar(i):
    # The warp's MMAs completed when they issued: the arrive-on at once (`utcbar`), on each CTA the
    # mask names with `.MULTICAST`.
    ops, mods = i["ops"], i["mods"]
    a, off = ur_mem(ops[0])
    at = f"(UR({a}) + {off & 0xffffffff:#x}u)"
    arrive = "mbAtomic(MB(ctx, {}), 1, 1u, 0, false, &FAULT)"
    if "MULTICAST" in mods:
        m = ops[2]["r"]["i"]
        body = (f"for (uint32_t _k = 0; _k < 16; ++_k) if ((UR({m}) >> _k) & 1u) "
                + arrive.format(f"(_k << 24) | ({at} & 0xffffffu)") + ";")
    else:
        body = arrive.format(at) + ";"
    return f"if (lane == __ffs(__EXEC__) - 1) {{ __threadfence(); {body} }}"


@form("UTCATOMSWS")
def f_utcatomsws(i):
    ops, mods = i["ops"], i["mods"]
    if "FIND_AND_SET" in mods:
        pd, ud, un = ops[0]["r"]["i"], ops[1]["r"]["i"], ops[2]["r"]["i"]
        return (f"{{ int _g = 0; if (lane == __ffs(__EXEC__) - 1) _g = tmemAlloc(ctx, UR({un})); "
                f"_g = __shfl_sync(__EXEC__, _g, __ffs(__EXEC__) - 1); "
                f"UPW({pd}, _g >= 0); if (_g >= 0) URW({ud}, (uint32_t)_g); }}")
    if "AND" in mods:
        ud, um = ops[0]["r"]["i"], ops[1]["r"]["i"]
        w = f"URW({ud}, _o);" if ud != 255 else ""
        return (f"{{ uint32_t _o = 0; if (lane == __ffs(__EXEC__) - 1) _o = tmemFree(ctx, UR({um})); "
                f"_o = __shfl_sync(__EXEC__, _o, __ffs(__EXEC__) - 1); {w} }}")
    raise Lift(f"UTCATOMSWS {mods}")


@form("UVIRTCOUNT")
def f_uvirtcount(i):
    # `tcgen05.relinquish_alloc_permit`: no permits in the model.
    return ""


def tmem_form(mods):
    shape, half = 0, None
    for m, sh in (("16dp64bit", 1), ("16dp128bit", 2), ("16dp256bit", 3), ("16dp32bit_t0_t15", 4), ("16dp32bit_t16_t31", 4)):
        if m in mods:
            shape = sh
            half = m if sh == 4 else None
    num = next((int(m[1:]) for m in mods if m.startswith("x") and m[1:].isdigit()), 1)
    pack = "PACK16BIT" in mods or "EXPAND16BIT" in mods
    regs = {0: 1, 1: 1, 2: 2, 3: 4, 4: 1}[shape] * num
    return shape, half, regs, pack


@form("LDTM", "STTM")
def f_ldtm(i):
    ops, mods = i["ops"], i["mods"]
    ld = i["op"] == "LDTM"
    shape, half, regs, pack = tmem_form(mods)
    t, rg = (ops[1], ops[0]) if ld else (ops[0], ops[1])
    base = rg["r"]["i"]
    # `.16x32bx2`'s two instructions move one half of the warp's threads each.
    who = {None: "true", "16dp32bit_t0_t15": "lane < 16u", "16dp32bit_t16_t31": "lane >= 16u"}[half]
    regs_c = ", ".join(f"r[{base + j}]" for j in range(regs))
    if ld:
        back = " ".join(f"r[{base + j}] = _r[{j}];" for j in range(regs))
        return (f"if ({who}) {{ uint32_t _r[{regs}] = {{{regs_c}}}; tmemMove(ctx, true, {shape}, {regs}u, {str(pack).lower()}, "
                f"{tmem_op(t)}, warp, lane, _r); {back} }}")
    return (f"if ({who}) {{ uint32_t _r[{regs}] = {{{regs_c}}}; tmemMove(ctx, false, {shape}, {regs}u, {str(pack).lower()}, "
            f"{tmem_op(t)}, warp, lane, _r); }}")


@form("STAS")
def f_stas(i):
    # `NV.SASS.Async.stas`: each lane's words into the cluster shared address in Ra, then a
    # complete-tx of their bytes on the mbarrier at Ra+1.
    ops, mods = i["ops"], i["mods"]
    o = ops[0]
    if o["k"] != "mem" or not o["base"] or o["base"]["f"] != "r" or not o["wide"] or o["ureg"]:
        raise Lift("STAS address")
    ra = o["base"]["i"]
    rb = ops[1]["r"]["i"]
    words = 4 if "128" in mods else 2 if "64" in mods else 1
    st = " ".join(f"_d[{j}] = r[{rb + j}];" for j in range(words))
    return (f"{{ uint32_t* _d = (uint32_t*)sharedAt(ctx, r[{ra}] + {o['off'] & 0xffffffff:#x}u); {st} __threadfence(); "
            f"mbAtomic(MB(ctx, r[{ra + 1}]), 2, 0, {4 * words}, false, &FAULT); }}")


@form("UTMALDG", "UTMASTG")
def f_utma(i):
    ops, mods = i["ops"], i["mods"]
    a, _ = ur_mem(ops[0])
    b, _ = ur_mem(ops[1])
    dims = next(int(m[:-1]) for m in mods if m.endswith("D") and m[:-1].isdigit())
    load = i["op"] == "UTMALDG"
    multicast = load and "MULTICAST" in mods
    extra = [m for m in mods if not (m.endswith("D") and m[:-1].isdigit()) and not (multicast and m == "MULTICAST")]
    if extra:
        raise Lift(f"{i['op']} modifiers {extra}")
    c0 = a + 2 if load else a + 1
    coords = ", ".join(f"(int32_t)UR({c0 + j})" for j in range(dims))
    if multicast:
        # `.MULTICAST`: the box into each cluster CTA the mask (the third operand's high half)
        # names, at the same shared offset, and its mbarrier at the same offset signalled.
        mreg = i["ops"][2]["r"]["i"]
        call = f"tmaLoad(ctx, _m, _c, UR({a}), UR({a + 1}), 0x10000u | (UR({mreg}) >> 16), lane, warp)"
    else:
        call = (f"tmaLoad(ctx, _m, _c, UR({a}), UR({a + 1}), 0u, lane, warp)" if load
                else f"tmaStore(ctx, _m, _c, UR({a}), lane, warp)")
    return (f"if (lane == __ffs(__EXEC__) - 1) {{ int32_t _c[5] = {{{coords}}}; "
            f"TensorMapView _m = mapAt(ctx, U64(UR({b}), UR({b + 1}))); {call}; }}")


@form("HGMMA", "QGMMA", "IGMMA")
def f_hgmma(i):
    ops, mods = i["ops"], i["mods"]
    shape = mods[0].split("x")
    # The input types and K: f16, bf16 (16), tf32 (8) into f32; E4M3, E5M2 (32) into f32; s8, u8
    # (32) into s32.
    tys = {("HGMMA", "F32"): ("F16", "F16", "16"), ("HGMMA", "F32", "BF16"): ("BF16", "BF16", "16"),
           ("HGMMA", "F32", "TF32"): ("TF32", "TF32", "8")}
    for a in ("E4M3", "E5M2"):
        for b in ("E4M3", "E5M2"):
            tys[("QGMMA", "F32", a, b)] = (a, b, "32")
    for a in ("S8", "U8"):
        for b in ("S8", "U8"):
            tys[("IGMMA", a, b)] = (a, b, "32")
    key = (i["op"], *mods[1:])
    if shape[0] != "64" or key not in tys or shape[2] != tys[key][2]:
        raise Lift(f"{i['op']} {mods}")
    n = int(shape[1])
    d = ops[0]["r"]["i"]
    # A from registers (the RS form): the descriptor operand after it.
    rs = ops[1]["k"] == "reg"
    if rs and (i["op"] != "HGMMA" or tys[key][0] not in ("F16", "BF16")):
        raise Lift(f"{i['op']} with A from registers")
    g = ops[2 if rs else 1]
    if g["k"] != "gdesc":
        raise Lift(f"{i['op']} operands")
    u = g["r"]["i"]
    dm = g["mods"]
    c = ops[3 if rs else 2]
    creg = c["k"] == "reg" and c["r"]["i"] != 255
    pi = 4 if rs else 3
    pr = pred(ops[pi]) if len(ops) > pi and is_pred(ops[pi]) else "true"
    scale = f"({'true' if creg else 'false'} && {pr})"
    if creg and c["r"]["i"] != d:
        raise Lift(f"{i['op']} accumulating from another register range")
    ty, tyb, _ = tys[key]
    if rs:
        ar = ops[1]["r"]["i"]
        areg = ", ".join(f"r[{ar + k}]" for k in range(4))
        load = " ".join(f"_acc[{k}] = r[{d + k}];" for k in range(n // 2))
        store = " ".join(f"r[{d + k}] = _acc[{k}];" for k in range(n // 2))
        return (f"{{ uint32_t _acc[{n // 2}]; const uint32_t _a[4] = {{{areg}}}; {load} hgmmaRS<{n}>(ctx, {ty}, _a, "
                f"U64(UR({u + 2}), UR({u + 3})), {'false' if 'tnspB' in dm else 'true'}, "
                f"{'true' if ops[1].get('neg') else 'false'}, {'true' if 'negB' in dm else 'false'}, {scale}, _acc, "
                f"lane, warp, __EXEC__); {store} }}")
    words = n // 2
    load = " ".join(f"_acc[{k}] = r[{d + k}];" for k in range(words))
    store = " ".join(f"r[{d + k}] = _acc[{k}];" for k in range(words))
    return (f"{{ uint32_t _acc[{words}]; {load} hgmma<{n}>(ctx, {ty}, U64(UR({u}), UR({u + 1})), "
            f"U64(UR({u + 2}), UR({u + 3})), {'false' if 'tnspA' in dm else 'true'}, "
            f"{'false' if 'tnspB' in dm else 'true'}, {'true' if 'negA' in dm else 'false'}, "
            f"{'true' if 'negB' in dm else 'false'}, {scale}, _acc, lane, warp, __EXEC__, {tyb}); {store} }}")


# ---------------------------------------------------------------------------------------------

def emit(i):
    fn = H.get(i["op"])
    if fn is None:
        raise Lift(f"opcode {i['op']}{''.join('.' + m for m in i['mods'])} at {i['pc']:#x}")
    try:
        body = fn(i)
    except Lift as e:
        raise Lift(f"{i['op']}{''.join('.' + m for m in i['mods'])} at {i['pc']:#x}: {e}")
    except (KeyError, IndexError, StopIteration) as e:
        raise Lift(f"{i['op']}{''.join('.' + m for m in i['mods'])} at {i['pc']:#x}: form ({e!r})")
    g = i["guard"]
    # Control transfers take their guard themselves (`split`).
    if g and body and i["op"] not in ("BRA", "EXIT", "CALL", "RET"):
        o = {"k": "reg", "r": g["r"], "not": g["not"], "neg": False, "abs": False, "sfx": []}
        if "__EXEC__" in body:
            # The executing lanes: those of the lanes together here whose guard holds.
            body = (f"{{ const bool _gd = {pred(o)}; const uint32_t _ex = __ballot_sync(act, _gd); "
                    f"if (_gd) {{ {body.replace('__EXEC__', '_ex')} }} }}")
        else:
            body = f"if ({pred(o)}) {{ {body} }}"
    return body.replace("__EXEC__", "act")


def main():
    lines = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    meta, prog = lines[0], lines[1:]
    global HAS_BREAK
    HAS_BREAK = any(i["op"] == "BREAK" for i in prog)
    BANKS.update({int(b): ws for b, ws in meta.get("banks", {}).items()})
    RET_SITES.update(i["pc"] + 16 for i in prog if i["op"] == "CALL")
    targets = {o["pc"] for i in prog for o in i["ops"] if o["k"] == "target"} | RET_SITES
    out = []
    errors = []
    for i in prog:
        try:
            body = emit(i)
        except Lift as e:
            errors.append(str(e))
            continue
        lab = f"L_{i['pc']:x}: " if i["pc"] in targets else ""
        out.append(f"  {lab}{body} // {i['pc']:#06x} {i['op']}{''.join('.' + m for m in i['mods'])}")
    if errors:
        seen = sorted(set(e.split(' at ')[0] for e in errors))
        raise Lift(f"{len(errors)} instructions not lifted; forms: " + "; ".join(seen[:40]) + "\n" + "\n".join(errors[:20]))
    print(f"// Lifted from {meta['kernel']} (sm_{meta['sm']}), {len(prog)} instructions, by tools/lift/lift.py.")
    print(f"#define LIFT_LOCAL {max(meta.get('localBytes', 0), 4)}")
    print(f"#define LIFT_THREADS {THREADS}")
    print(f"#define LIFT_BREAK {int(HAS_BREAK)}")
    # A kernel that reads the clock writes what no simulator's run can match (main.cu reports it).
    print(f"#define LIFT_CLOCK {int(READS_CLOCK)}")
    # `Exec.localWindowAt`: the local window's base in constant bank 0 (sm_100 on: 0x2f8).
    print(f"#define LIFT_LOCALWIN {0x2f8 if meta['sm'] >= 100 else 0x20:#x}")
    print('#include "rt.cuh"')
    print("using namespace lift;")
    for b, ws in sorted(BANKS.items()):
        print(f"__device__ const uint32_t bank{b}[{max(len(ws), 1)}] = {{{', '.join(str(w) for w in ws) or '0'}}};")
    print(f"constexpr uint32_t SM = {meta['sm']};")
    print(f"constexpr uint32_t PARAM_BASE = {meta['paramBase']}u, PARAM_BYTES = {meta['paramBytes']}u;")
    print("extern \"C\" __global__ void __launch_bounds__(LIFT_THREADS, 1) lifted(Ctx ctx0) {")
    print("  PROLOGUE")
    print("\n".join(out))
    print("  EPILOGUE")
    print("}")
    # What the harness (main.cu) reads from the kernel's cubin.
    print("extern \"C\" __device__ __attribute__((used)) const uint32_t lift_info[4] = {LIFT_THREADS, LIFT_CLOCK, PARAM_BASE, PARAM_BYTES};")


def tables(out):
    """mufu_tables.cuh: the special-function unit's tables from NV/Num/MufuTables.lean."""
    src = (Path(__file__).resolve().parents[2] / "NV/Num/MufuTables.lean").read_text()
    names = {"ex2Table": "kEx2", "lg2Table": "kLg2", "rcpTable": "kRcp", "rsqrtTable": "kRsqrt", "sqrtTable": "kSqrt"}
    lines = ["// Generated by tools/lift/lift.py --tables from NV/Num/MufuTables.lean; do not edit.", "#pragma once"]
    for m in re.finditer(r"def (\w+Table) : Array \(Nat × Nat × Nat\) := #\[(.*?)\]", src, re.S):
        if m.group(1) not in names:
            continue
        ents = re.findall(r"\((0x[0-9a-f]+|\d+), (0x[0-9a-f]+|\d+), (0x[0-9a-f]+|\d+)\)", m.group(2))
        body = ", ".join("{" + ", ".join(f"{int(x, 0):#x}u" for x in e) + "}" for e in ents)
        lines.append(f"__device__ const uint32_t {names[m.group(1)]}[{len(ents)}][3] = {{{body}}};")
    Path(out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--tables"]:
        tables(sys.argv[2])
        sys.exit(0)
    THREADS = 384
    HAS_BREAK = False
    READS_CLOCK = False
    if sys.argv[1:2] == ["--threads"]:
        THREADS = int(sys.argv[2])
        del sys.argv[1:3]
    try:
        main()
    except Lift as e:
        print(f"lift: {e}", file=sys.stderr)
        sys.exit(1)
