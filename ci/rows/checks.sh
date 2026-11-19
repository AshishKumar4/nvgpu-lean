# The simulator's race and unwritten-output checks on kernels whose answer is known
# (examples/checks/checks.cu): each must report exactly what the kernel does.
lake build sim
d=examples/checks
expect() { # spec flags pattern [pass]: the pattern must appear, and the run fail (pass: succeed);
  # "clean": the run must succeed. GPU: the part (h200 unless set).
  st=0; out=$(lake exe sim "$d/$1.json" --gpu "${GPU:-h200}" $2 2>&1) || st=$?
  if [ "$3" = clean ]; then
    [ $st -eq 0 ] || { echo "FAIL $1: expected clean, got:"; echo "$out" | tail -5; fail=1; return; }
  else
    want=1; [ "${4:-}" = pass ] && want=0
    { [ $(( st != 0 )) -eq $want ] && echo "$out" | grep -q "$3"; } || { echo "FAIL $1: expected '$3', got:"; echo "$out" | tail -5; fail=1; return; }
  fi
  echo "ok   $1 $2"
}
fail=0
expect shared_race --races "race .*on shared"
expect shared_ordered --races clean
expect global_race --races "race .*on global"
expect named_bar_ordered --races clean
expect named_bar_race --races "read after write on shared"
expect mbar_ordered --races clean
expect mbar_race --races "race .*on shared"
expect wgmma_fenced --races clean
expect wgmma_unfenced --races "async-proxy read of a generic write with no proxy fence"
expect wgmma_reuse --races clean
expect bulk_ordered --races clean
expect bulk_race --races "race .*on shared"
expect cp_async_ordered --races clean
expect cp_async_race --races "race .*on shared"
expect cp_async_stages --races clean
expect cluster_ordered --races clean
expect cluster_race --races "race .*on shared"
expect shared_unwritten --races "read of shared memory its CTA never wrote"
expect cross_cta_read --races "a global byte one CTA read and another stored"
expect float_atomic_sum --races "floating-point atomic additions to one address from several warps"
expect cross_cta_store --races "stores of two CTAs to one global byte: 384 bytes, 96 with different values"
GPU=b200 expect tc_restage_ordered --races clean
GPU=b200 expect tc_restage_race --races "write after read on shared"
expect half_written --unwritten "unwritten: out, 1024 bytes, at \[(1024, 2048)\]"
expect shared_ordered --unwritten "unwritten: none" pass
exit $fail
