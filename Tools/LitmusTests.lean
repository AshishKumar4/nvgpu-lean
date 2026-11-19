import NV.PTX.Memory.Litmus

/-!
`lake exe litmus-tests`: litmus tests of the Hopper features the nvlitmus format cannot express
(CTA and cluster barriers) and regressions for the model's own past defects, each with its
expected verdict. Every forbidden outcome has an allowed sibling, so the model cannot pass by
forbidding everything. Exits non-zero on any mismatch.
-/

open NV.PTX NV.PTX.Litmus

def cta (c : Nat) (cluster : Nat := c) : Place := { cta := c, cluster, gpu := 0 }

def r (n : Nat) : Val := .reg n
def k (v : Int) : Val := .const v

/-- `test`, its commands, and whether each is expected to be witnessed (`allowed`). -/
def cases : List (Test × List Bool) := [
  -- §8.9.4: bar.sync orders a CTA's write before another thread's read after the barrier.
  ({ name := "bar_mp", commands := [⟨"sees", .assert, .eq (r 0) (k 1)⟩]
     threads := [⟨0, cta 0, [.st 0 (k 1) .weak .thread, .barSync 1]⟩,
                 ⟨1, cta 0, [.barSync 1, .ld 0 0 .weak .thread]⟩] }, [false]),
  -- Barriers synchronize only within their CTA.
  ({ name := "bar_mp_other_cta", commands := [⟨"stale", .permit, .eq (r 0) (k 0)⟩]
     threads := [⟨0, cta 0, [.st 0 (k 1) .weak .thread, .barSync 1]⟩,
                 ⟨1, cta 1, [.barSync 1, .ld 0 0 .weak .thread]⟩] }, [true]),
  -- A barrier used twice by one thread: its instances are matched in order, not all-to-all,
  -- so the program has executions at all.
  ({ name := "bar_reuse", commands := [⟨"exists", .permit, .eq (r 0) (k 1)⟩,
                                       ⟨"sees", .assert, .eq (r 0) (k 1)⟩]
     threads := [⟨0, cta 0, [.barSync 0, .st 0 (k 1) .weak .thread, .barSync 0, .ld 0 0 .weak .thread]⟩] },
   [true, false]),
  -- barrier.cluster.arrive (release) and .wait (acquire) order writes across a cluster's CTAs.
  ({ name := "cluster_mp", commands := [⟨"sees", .assert, .eq (r 0) (k 1)⟩]
     threads := [⟨0, cta 0 0, [.st 0 (k 1) .weak .thread, .clusterArrive, .clusterWait]⟩,
                 ⟨1, cta 1 0, [.clusterArrive, .clusterWait, .ld 0 0 .weak .thread]⟩] }, [false]),
  ({ name := "cluster_mp_no_barrier", commands := [⟨"stale", .permit, .eq (r 0) (k 0)⟩]
     threads := [⟨0, cta 0 0, [.st 0 (k 1) .weak .thread]⟩,
                 ⟨1, cta 1 0, [.ld 0 0 .weak .thread]⟩] }, [true]),
  -- A register's final value is its last definition.
  ({ name := "reg_latest", commands := [⟨"last", .assert, .eq (r 0) (k 1)⟩]
     threads := [⟨0, cta 0, [.st 0 (k 1) .weak .thread, .ld 0 1 .weak .thread, .ld 0 0 .weak .thread]⟩] },
   [false]),
  -- Two increments that each read the other's write would need r1 = r2 + 1 and r2 = r1 + 1.
  ({ name := "rmw_no_cycle", commands := [⟨"cycle", .permit, .and (.eq (r 1) (k 1)) (.eq (r 2) (k 1))⟩,
                                          ⟨"lost_update", .permit, .and (.eq (r 1) (k 0)) (.eq (r 2) (k 0))⟩]
     threads := [⟨0, cta 0, [.atomAdd 1 0 (k 1) .relaxed .cta]⟩,
                 ⟨1, cta 1, [.atomAdd 2 0 (k 1) .relaxed .cta]⟩] }, [false, true]),
  -- With scopes that include each other, the increments are atomic (§8.10.3).
  ({ name := "rmw_atomic", commands := [⟨"lost_update", .permit, .and (.eq (r 1) (k 0)) (.eq (r 2) (k 0))⟩,
                                        ⟨"ordered", .permit, .and (.eq (r 1) (k 0)) (.eq (r 2) (k 1))⟩]
     threads := [⟨0, cta 0, [.atomAdd 1 0 (k 1) .relaxed .gpu]⟩,
                 ⟨1, cta 1, [.atomAdd 2 0 (k 1) .relaxed .gpu]⟩] }, [false, true]),
  -- §8.11.1: a reduction's read does not start an acquire pattern with a later fence.
  ({ name := "red_not_acquire", commands := [⟨"stale", .permit, .eq (r 0) (k 0)⟩]
     threads := [⟨0, cta 0, [.st 0 (k 1) .weak .thread, .st 1 (k 1) .release .gpu]⟩,
                 ⟨1, cta 1, [.redAdd 1 (k 1) .relaxed .gpu, .fence .acqRel .gpu, .ld 0 0 .weak .thread]⟩] },
   [true]),
  ({ name := "atom_then_fence_acquires", commands := [⟨"stale", .permit,
       .and (.eq (r 2) (k 1)) (.eq (r 0) (k 0))⟩]
     threads := [⟨0, cta 0, [.st 0 (k 1) .weak .thread, .st 1 (k 1) .release .gpu]⟩,
                 ⟨1, cta 1, [.atomAdd 2 1 (k 1) .relaxed .gpu, .fence .acqRel .gpu, .ld 0 0 .weak .thread]⟩] },
   [false])
]

def main : IO UInt32 := do
  let mut bad := 0
  for (t, expected) in cases do
    let os := outcomes t
    for (c, e) in t.commands.zip expected do
      let got := c.witnessed os
      IO.println s!"{if got == e then "ok  " else "FAIL"} {t.name} {c.name} {if got then "allowed" else "forbidden"}"
      if got != e then bad := bad + 1
  return if bad == 0 then 0 else 1
