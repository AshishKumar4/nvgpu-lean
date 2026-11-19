# The PTX memory model in `NV/PTX/Memory`

Primary source: the PTX ISA 9.4, chapter 8 ("Memory Consistency Model"). Cross-check: Lustig,
Cooksey and Giroux's Alloy model of the mixed-proxy PTX memory model (github.com/NVlabs/mixedproxy,
`alloy/ptx.als`), which extends Lustig, Sahasrabuddhe and Giroux, "A Formal Analysis of the NVIDIA
PTX Memory Consistency Model", ASPLOS 2019 (github.com/NVlabs/ptxmemorymodel). Where the two
disagree the ISA decides and the disagreement is listed below.

## Supported fragment

The model decides candidate executions of programs made of:

- `ld`, `st`, `atom.add`, `red.add` at `.weak`, `.relaxed`, `.acquire`, `.release`, `.acq_rel`,
  with scopes `.cta`, `.cluster`, `.gpu`, `.sys` (§8.4–8.5);
- `fence.acq_rel` and `fence.sc` at any scope; `fence.proxy.<proxy>` and `fence.proxy.alias`;
- `bar.sync`/`bar.arrive` (as an arrival and a wait) and `barrier.cluster.arrive`/`.wait` (§8.9.4);
- the generic, async, texture, surface, constant and tensormap proxies, and virtual aliases
  (§8.2.2, §8.6).

Outside the fragment, and rejected by construction (no instruction form exists for them):
mixed-size accesses (every access is to one whole location; §8.7.2 says the axioms do not apply to
mixed-size races), the fabric proxy, `.mmio` and `.volatile` beyond nvlitmus's
`relaxed.sys` reading, and the API synchronization of §8.9.4 (stream order, events, graphs). The
asynchronous operations of §8.9.1.1 (`cp.async.bulk`, TMA, `wgmma`) enter through the `issue`
relation of an execution; the litmus front end does not yet generate them.

Executions are generic in the value an event carries (`Execution n V`): litmus tests use
integers, programs' executions (`docs/formal.md`) the bytes an access reads or writes.

## Definitions

`Execution.consistent` = `wellFormed` ∧ the five axioms of §8.10.

`wellFormed` (the facts `ptx.als` lists as "basic model constraints"):

| check | content | source |
|---|---|---|
| `poWF` | program order is a strict total order per thread, empty across threads; with issue order acyclic | §8.9.1, §8.9.1.1 |
| `rfWF` | a read reads at most one write, of its location, and returns its value; otherwise the initial value | §8.2.6, §8.9.7 |
| `coWF` | coherence is a strict partial order on same-location writes, total on morally strong writes to one address | §8.9.6; ptx.als `total[co, …]` |
| `scWF` | Fence-SC order is a strict partial order on `fence.sc`, total on morally strong pairs | §8.9.3 |
| `rmwWF` | an RMW is a read and a write of one address, one thread, one scope, the write depending on the read; dependencies follow program order | §8.10.3; ptx.als `rmw in same_alias[po & dep & scope.~scope]` |
| `barWF` | barrier synchronization relates an arrival to a wait of the same barrier and CTA (or cluster) | §8.9.4 |

Axioms: Coherence (§8.10.1), Fence-SC (§8.10.2), Atomicity (§8.10.3), No-Thin-Air (§8.10.4),
Causality (§8.10.6). Sequential consistency per location (§8.10.5) is not a separate axiom in
either source; it follows from Causality with program order in base causality order.

## Where this model and `ptx.als` differ

1. **Fences carry the generic proxy.** In `ptx.als` fences have no proxy, so `strong_r` (which
   requires `proxy.~proxy`) never relates a fence, and no fence-based release or acquire pattern
   ever synchronizes. The ISA's own message-passing example (§8.10.6, `fence.sys`) is then allowed.
   `tools/litmus/crosscheck.py` runs the artifact as is and with fences read as generic-proxy
   operations. They differ on three commands, all message passing through a fence pattern:
   `MP_fence#0`, `#1` (§8.10.6's example at `.sys` and `.gpu`) and mixedproxy's own
   `Release_acquire_pattern`, whose expectation (`permit r1 == 2 && r3 == 0`) contradicts §8.8: a
   release followed by a strong write is a release pattern and `ld.relaxed; fence.acq_rel` an
   acquire pattern, so `r3` must read 1.
2. **A reduction's read starts no acquire pattern** (§8.11.1). `ptx.als` has no such rule.
3. **Fence-SC uses base causality order.** `ptx.als` checks `FenceSC <: cause :> FenceSC in sc`,
   but its `cause` only relates memory operations (every summand of `proxy_preserved_cause_base`
   is filtered by address), so the check is vacuous. The ISA's §8.10.2 constrains fences ordered
   by causality; between fences that is base causality order.
4. **Barriers.** `ptx.als` has no barriers; §8.9.4's barrier rules are added with arrivals and
   waits as separate events, matched instance by instance (the `k`-th arrival of a barrier in a
   thread synchronizes with the `k`-th waits of the others).

## Evidence

- `Rel.tc_iff_transGen`: the closure used throughout is the transitive closure.
- `Examples.mp_gpu_forbidden` / `mp_cta_allowed`: §8.10.6 message passing, kernel-decided, with
  the allowed sibling that shows the model is not forbidding everything.
- `lake exe litmus-tests`: barrier, cluster-barrier, register, RMW-atomicity and reduction
  regressions, each forbidden outcome paired with an allowed one.
- `tools/litmus/crosscheck.py` (CI row `alloy`): every nvlitmus test of mixedproxy (pinned to commit
  `0807da29`, checked unmodified) plus
  `tools/litmus/tests`, decided by Alloy (artifact and fence-generic) and by the Lean model; the
  run fails on any Java error, on a missing, extra or duplicate verdict, and on any disagreement
  with the fence-generic model. This is finite regression evidence, not a correspondence proof.
