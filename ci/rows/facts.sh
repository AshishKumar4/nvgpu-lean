# The generated facts tables are current, every fact's citation resolves, and the parameter
# inventory (NV/Core/Roles.lean) lists every Params field, with each `engine` field read by the core.
lake build facts && lake exe facts --check
python3 - <<'PY'
import re, pathlib, sys
params = pathlib.Path("NV/Core/Params.lean").read_text()
roles = pathlib.Path("NV/Core/Roles.lean").read_text()
listed = dict(re.findall(r'\("([\w.]+)", \.(\w+)\)', roles))
prefix = {"MemoryParams": "mem.", "RegFileParams": "regFile.", "FrontEndParams": "frontEnd.",
          "TensorParams": "tensor.", "SyncParams": "sync.", "TmaParams": "tma.",
          "LaunchParams": "launch.", "LinkParams": "link.", "Params": ""}
core = "".join(p.read_text() for d in ["NV/Core", "NV/Core/Memory", "NV/Async", "NV/SASS", "NV/Launch"]
               for p in pathlib.Path(d).glob("*.lean") if p.name not in ("Params.lean", "Roles.lean"))
bad = []
for name, body in re.findall(r"structure (\w+) where\n(.*?)deriving", params, re.S):
    if name not in prefix: continue
    for f in re.findall(r"^  (\w+) : ", body, re.M):
        key = prefix[name] + f
        if name == "Params" and f in ("name", "regFile", "frontEnd", "mem", "tensor", "sync", "tma", "launch", "link"):
            continue
        role = listed.get(key, "engine" if name == "MemoryParams" else None)
        if role is None: bad.append(f"{key}: not in NV/Core/Roles.lean")
        elif role == "engine" and not re.search(r"\." + f + r"\b", core): bad.append(f"{key}: role engine, but the core never reads it")
print("\n".join(bad))
sys.exit(1 if bad else 0)
PY
