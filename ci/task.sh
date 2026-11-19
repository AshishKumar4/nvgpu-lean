#!/bin/sh
# One CI row (ci/rows/NAME.sh): runs it, keeps its full log as an artifact, and writes the row's
# verdict to $2. Add a row by adding its file; the first comment lines say what it checks.
set -u
row="$1"; out="$2"; art="${CI_ARTIFACTS:-/tmp}"
mkdir -p "$art"
start=$(date +%s)
if [ -f "ci/rows/$row.sh" ]; then sh -e "ci/rows/$row.sh" > "$art/$row.log" 2>&1; code=$?
else echo "unknown row $row" > "$art/$row.log"; code=2; fi
python3 - "$row" "$code" "$(( $(date +%s) - start ))" "$art/$row.log" "$out" <<'PY'
import json, sys
row, code, secs, log, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
text = open(log, errors="replace").read()
lines = [l for l in text.splitlines() if "warning:" not in l and "deprecated" not in l]
json.dump({"rows": [{"name": row, "exitCode": code, "seconds": secs,
                     "output": "\n".join(lines[-150:]), "artifacts": [row + ".log"]}]}, open(out, "w"))
PY
exit 0
