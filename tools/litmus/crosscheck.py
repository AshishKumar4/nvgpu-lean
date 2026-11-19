"""Cross-check the Lean PTX memory model against Lustig et al.'s Alloy model on nvlitmus tests.

Each test (NVlabs/mixedproxy `tests/*.test`, plus ours in `tools/litmus/tests`) is expanded and
parsed with the mixedproxy front end, so both models read it identically. Alloy decides every
command with the artifact's `alloy/ptx.als` and with the same model reading fences as generic-proxy
operations (`fence_generic_model`; see `docs/memory-model.md`). The Lean model decides the same commands
via `lake exe litmus`. Usage, from the repository root:

    python tools/litmus/crosscheck.py --mixedproxy PATH [--jobs N] [tests ...]
"""

import argparse
import concurrent.futures
import json
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
# The mixedproxy commit the cross-check was validated against (github.com/NVlabs/mixedproxy).
MIXEDPROXY_SHA = "0807da29ac98d86a44179a65d430b78532e662b5"

# `ptx.als` gives fences no proxy, so `strong_r` relates no fence to anything and fence-based
# release/acquire patterns never synchronize. The patched model reads a fence as generic.
STRONG_R = "symmetric[scope.~scopes] & proxy.~proxy"
STRONG_R_FENCE_GENERIC = (
    "symmetric[scope.~scopes] & (proxy + (Op - MemoryOp)->GenericProxy)"
    ".~(proxy + (Op - MemoryOp)->GenericProxy)")


def fence_generic_model(artifact, out):
    text = artifact.read_text()
    assert text.count(STRONG_R) == 1, "ptx.als changed: re-derive the fence patch"
    out.write_text(text.replace(STRONG_R, STRONG_R_FENCE_GENERIC))
    return out


def expand(text):
    """Instances of a template test: `$i` replaced by column i of each row below `$$`."""
    if "$$" not in text:
        return [text]
    body, rows = text.split("$$\n")
    out = []
    for row in rows.split("\n"):
        if not row.strip() or row[0] == "#":
            continue
        inst = body
        for i, cell in enumerate(row.split("|")):
            inst = inst.replace(f"${i}", cell.strip())
        out.append(inst)
    return out


class Encoder:
    """Translate a parsed mixedproxy test into the JSON `H200.PTX.Litmus.Test` derives."""

    def __init__(self, lit, test):
        self.lit = lit
        self.test = test
        self.addrs = {}
        for name, a in test.addresses.items():
            if a.alias_type == "virtually":
                continue
            self.addrs.setdefault(name, len(self.addrs))
        for name, a in test.addresses.items():
            if a.alias_type == "virtually":
                self.addrs[name] = self.addrs[a.alias]
        # Physical aliases are distinct addresses of one location.
        self.loc = [
            [self.addrs[name], self.addrs[a.alias]]
            for name, a in test.addresses.items()
            if a.alias_type == "physically"
        ]

    @staticmethod
    def reg(name):
        return int(str(name).lstrip("r"))

    def value(self, v):
        lit = self.lit
        if isinstance(v, lit.Integer):
            return {"const": {"v": int(v.n)}}
        if isinstance(v, lit.NamedValue):
            return {"reg": {"r": self.reg(v.name)}}
        raise ValueError(f"unsupported value {v}")

    def expect(self, rv):
        return int(rv.n) if isinstance(rv, self.lit.Integer) else None

    @staticmethod
    def sem(s):
        return {None: "weak", "weak": "weak", "relaxed": "relaxed", "acquire": "acquire",
                "release": "release", "acq_rel": "acqRel", "sc": "sc"}[s]

    @staticmethod
    def scope(s):
        return {None: "thread", "cta": "cta", "gpu": "gpu", "sys": "sys"}[s]

    def instr(self, i):
        lit = self.lit
        if isinstance(i, lit.Load):
            return {"ld": {"reg": self.reg(i.dst), "addr": self.addrs[i.src], "sem": self.sem(i.sem),
                           "scope": self.scope(i.scope), "proxy": i.proxy,
                           "expect": self.expect(i.return_value)}}
        if isinstance(i, lit.Store):
            return {"st": {"addr": self.addrs[i.dst], "v": self.value(i.value), "sem": self.sem(i.sem),
                           "scope": self.scope(i.scope), "proxy": i.proxy}}
        if isinstance(i, lit.Atom):
            return {"atomAdd": {"reg": self.reg(i.dst), "addr": self.addrs[i.src],
                                "v": self.value(i.value), "sem": self.sem(i.sem),
                                "scope": self.scope(i.scope), "proxy": i.proxy,
                                "expect": self.expect(i.return_value)}}
        if isinstance(i, lit.Fence):
            return {"fence": {"sem": self.sem(i.sem), "scope": self.scope(i.scope)}}
        if isinstance(i, lit.ProxyFence):
            return {"proxyFence": {"p": i.proxy}}
        if isinstance(i, lit.AliasFence):
            return "aliasFence"
        raise ValueError(f"unsupported instruction {i}")

    def cond(self, c):
        lit = self.lit
        if isinstance(c, lit.Not):
            return {"not": {"p": self.cond(c.a)}}
        if isinstance(c, lit.And):
            return {"and": {"p": self.cond(c.a), "q": self.cond(c.b)}}
        if isinstance(c, lit.Or):
            return {"or": {"p": self.cond(c.a), "q": self.cond(c.b)}}
        if isinstance(c, lit.Equal):
            return {"eq": {"a": self.value(c.a), "b": self.value(c.b)}}
        if isinstance(c, lit.Condition):
            kind = {"=": "eq", "==": "eq", "!=": "neq"}[c.op]
            return {kind: {"a": self.value(c.a), "b": self.value(c.b)}}
        raise ValueError(f"unsupported condition {c}")

    def encode(self, name):
        threads = []
        for k, (tid, th) in enumerate(self.test.threads.items()):
            cta = int(tid.d) * 1000 + int(tid.b)
            threads.append({"id": k, "place": {"cta": cta, "cluster": cta, "gpu": int(tid.d)},
                            "code": [self.instr(i) for i in th.insts]})
        commands = []
        for c in self.test.commands.values():
            expect = "permit" if c.expected else "assert"
            if c.name.startswith("check_"):
                expect = "check"
            commands.append({"name": c.name, "expect": expect, "cond": self.cond(c.expr)})
        return {"name": name, "threads": threads, "loc": self.loc, "commands": commands}


def alloy(mixedproxy, t2a, model, name, text):
    """Alloy's verdict per command: `allowed` when Alloy finds an instance (SAT). RunAlloy exits
    non-zero when a command breaks the test's own expectation, so the exit code is not a failure;
    a Java exception or a missing or extra verdict is."""
    als, commands = t2a.litmus_to_alloy(model.read_text(), text)
    proc = subprocess.run(
        ["java", "-cp", str(mixedproxy / "alloy" / "org.alloytools.alloy.dist.jar"),
         str(mixedproxy / "alloy" / "RunAlloy.java")],
        input=als.encode(), capture_output=True, check=False)
    if "Exception" in proc.stderr.decode():
        raise SystemExit(f"{name}: Alloy failed:\n{proc.stderr.decode()}")
    verdicts = {}
    for line in proc.stdout.decode().splitlines():
        m = re.match(r"^([A-Za-z_]\w*): (SAT|UNSAT)", line)
        if m and not m.group(1).startswith("sanity_"):
            if m.group(1) in verdicts:
                raise SystemExit(f"{name}: duplicate Alloy verdict for {m.group(1)}")
            verdicts[m.group(1)] = "allowed" if m.group(2) == "SAT" else "forbidden"
    if set(verdicts) != set(commands):
        raise SystemExit(f"{name}: Alloy decided {sorted(verdicts)}, the test has {sorted(commands)}")
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixedproxy", required=True, type=pathlib.Path)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("tests", nargs="*", type=pathlib.Path)
    args = ap.parse_args()
    sha = subprocess.run(["git", "-C", str(args.mixedproxy), "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=True).stdout.strip()
    if sha != MIXEDPROXY_SHA:
        raise SystemExit(f"mixedproxy checkout is {sha}, expected {MIXEDPROXY_SHA}")
    if subprocess.run(["git", "-C", str(args.mixedproxy), "diff", "--quiet"]).returncode:
        raise SystemExit("mixedproxy checkout has local changes")
    sys.path.insert(0, str(args.mixedproxy / "src"))
    import litmus as lit  # noqa: E402  (the mixedproxy front end)
    import litmus_parser  # noqa: E402
    import test_to_alloy as t2a  # noqa: E402

    files = args.tests or sorted((args.mixedproxy / "tests").glob("*.test")) + \
        sorted((HERE / "tests").glob("*.test"))
    instances = []
    for f in files:
        for k, text in enumerate(expand(f.read_text())):
            instances.append((f"{f.stem}#{k}", text))

    model = args.mixedproxy / "alloy" / "ptx.als"
    workdir = pathlib.Path(tempfile.mkdtemp())
    patched_model = fence_generic_model(model, workdir / "ptx-fence-generic.als")
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        artifact = list(pool.map(lambda it: alloy(args.mixedproxy, t2a, model, *it), instances))
        patched = list(pool.map(lambda it: alloy(args.mixedproxy, t2a, patched_model, *it),
                                instances))

    tests = [Encoder(lit, litmus_parser.parse(model.read_text(), text)).encode(name)
             for name, text in instances]
    tests_json = workdir / "tests.json"
    tests_json.write_text(json.dumps(tests))
    out = subprocess.run(["lake", "exe", "litmus", str(tests_json)], capture_output=True, text=True,
                         check=True).stdout
    lean = {}
    for line in out.splitlines():
        test, cmd, verdict = line.split()
        if verdict not in ("allowed", "forbidden") or (test, cmd) in lean:
            raise SystemExit(f"{test} {cmd}: Lean verdict {verdict!r}")
        lean[(test, cmd)] = verdict
    expected = {(name, cmd) for (name, _), a in zip(instances, artifact) for cmd in a}
    if set(lean) != expected or not expected:
        raise SystemExit(f"Lean decided {len(lean)} commands, Alloy {len(expected)}: "
                         f"{sorted(set(lean) ^ expected)[:10]}")

    rows, disagree = [], 0
    for (name, _), a, p in zip(instances, artifact, patched):
        for cmd in a:
            lv = lean[(name, cmd)]
            status = "agree" if lv == p[cmd] else "DIFFER"
            disagree += status == "DIFFER"
            rows.append((name, cmd, a[cmd], p[cmd], lv, status))
    print(f"{'test':28} {'command':24} {'ptx.als':10} {'fence-gen':10} {'lean':10} vs fence-gen")
    for r in rows:
        print(f"{r[0]:28} {r[1]:24} {r[2]:10} {r[3]:10} {r[4]:10} {r[5]}")
    print(f"\n{len(rows)} commands, {disagree} disagreements with the fence-generic Alloy model, "
          f"{sum(r[2] != r[3] for r in rows)} where the artifact and fence-generic models differ")
    return 1 if disagree else 0


if __name__ == "__main__":
    sys.exit(main())
