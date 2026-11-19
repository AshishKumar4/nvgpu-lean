#!/bin/sh
# The repository's gate: forbidden constructs, build, axiom audit, litmus regressions. The
# cross-check against Lustig et al.'s Alloy model is its own CI row (ci/rows/alloy.sh).
# Usage: scripts/check.sh. Runs in CI or on any build machine.
set -e
cd "$(dirname "$0")/.."
# No proof escape hatches in the library or the examples: no sorry/admit/axiom, no FFI
# (`@[extern]`, `implemented_by`), no `unsafe`, `partial` or `opaque` definitions (Tools are
# executables about which nothing is proven). `native_decide`/`bv_decide` only under
# NV/CompilerTrusted, which the audit lists separately. The audit is the semantic check; this
# catches the keywords before a build.
decl='^[[:space:]]*(@\[[^]]*\][[:space:]]*)*((private|protected|noncomputable)[[:space:]]+)*'
if grep -rnE "\b(sorry|admit|implemented_by)\b|@\[extern|${decl}(partial|unsafe|opaque|axiom)[[:space:]]" \
    --include=*.lean NV Examples || grep -rnE "\b(native_decide|bv_decide)\b" --include=*.lean \
    --exclude-dir=CompilerTrusted NV Examples; then
  echo "gate: forbidden construct in Lean sources" >&2; exit 1
fi
# The decided examples are the memory peaks: build each alone, after its dependencies, so no
# other Lean process shares the machine's memory with it.
lake build Examples.SplitMerge NV.PTX.Memory.Model
for heavy in NV.PTX.Memory.Examples Examples.SplitMergeExamples; do
  /usr/bin/time -f "$heavy: %M KB peak, %e s" lake build "$heavy"
done
lake build
lake exe audit
lake exe litmus-tests
lake exe engine-smoke
lake exe memory-tests
