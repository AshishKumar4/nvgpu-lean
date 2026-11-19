import Examples.SplitMerge

/-!
# Split-merge witnesses, and the masked persistent launch's unwritten slot

Two splits in two CTAs. Split 0 writes its partial to `ws0`, then updates the counter; split 1
writes `ws1`, updates the counter after it, sees the total, and reads both slots.

* `merge_ok_consistent` and `mergeCounter`: an execution satisfying every hypothesis of
  `merge_sees` exists, so the theorem is not vacuous.
* `merge_stale_forbidden`: the same program where the last split reads `ws0`'s initial value is
  inconsistent, as `merge_sees` says; `merge_relaxed_stale_allowed`: with a relaxed counter update
  in the last split, the stale read is consistent, so the acquire is what forbids it.
* A masked persistent launch's tile whose key slice is empty returned before storing anything, and the merge read its slot.
  `empty_split_unwritten`: with split 1 storing nothing, the merge's read of `ws1` reads no write
  in a consistent execution: the slot holds whatever memory held. The fix stores the empty walk's
  neutral partial, which `merge_sees` then makes visible.
-/

namespace Examples.SplitMerge

open NV NV.Formal

open NV.PTX Rel

/-- Addresses. -/
def ws0 : Nat := 0
def ws1 : Nat := 1
def counter : Nat := 2

def place (t : Nat) : Place := { cta := t, cluster := t, gpu := 0 }

/-- Events 0–2: split 0's `st ws0; atom.add counter` (read, write); 3–7: split 1's
`st ws1; atom.add counter; ld ws0; ld ws1`. `lastAcquire` chooses the last split's update
semantics, `split1Writes` whether split 1 stores its partial. -/
def events (lastAcquire split1Writes : Bool) : Fin 8 → Event
  | 0 => { thread := 0, kind := .write, addr := ws0 }
  | 1 => { thread := 0, kind := .read, scope := .gpu, acquire := true, addr := counter }
  | 2 => { thread := 0, kind := .write, scope := .gpu, release := true, addr := counter }
  | 3 => if split1Writes then { thread := 1, kind := .write, addr := ws1 }
         else { thread := 1, kind := .fenceAcqRel, scope := .cta }
  | 4 => { thread := 1, kind := .read, scope := .gpu, acquire := lastAcquire, addr := counter }
  | 5 => { thread := 1, kind := .write, scope := .gpu, release := lastAcquire, addr := counter }
  | 6 => { thread := 1, kind := .read, addr := ws0 }
  | 7 => { thread := 1, kind := .read, addr := ws1 }

def thread (i : Fin 8) : Nat := if i.val < 3 then 0 else 1

/-- Split 1 reads split 0's counter write and both slots; `ws0Read` chooses whether the read of
`ws0` (event 6) reads split 0's partial or the initial value. -/
def exec (lastAcquire split1Writes ws0Read : Bool) : Execution 8 where
  ev := events lastAcquire split1Writes
  place := place
  init _ := 0
  val i := match i.val with
    | 1 => 0 | 2 => 1 | 4 => 1 | 5 => 2
    | 6 => if ws0Read then 1 else 0
    | 7 => if split1Writes then 1 else 0
    | _ => 1
  po := ofFn fun a b => thread a == thread b && a.val < b.val
  rf := ofFn fun w r =>
    (w.val = 2 && r.val = 4) || (ws0Read && w.val = 0 && r.val = 6) ||
      (split1Writes && w.val = 3 && r.val = 7)
  co := ofFn fun a b => a.val = 2 && b.val = 5
  rmw := ofFn fun a b => (a.val = 1 && b.val = 2) || (a.val = 4 && b.val = 5)
  dep := ofFn fun a b => (a.val = 1 && b.val = 2) || (a.val = 4 && b.val = 5)

theorem merge_ok_consistent : (exec true true true).consistent = true := by decide +kernel

theorem merge_stale_forbidden : (exec true true false).consistent = false := by decide +kernel

theorem merge_relaxed_stale_allowed : (exec false true false).consistent = true := by decide +kernel

theorem empty_split_unwritten :
    (exec true false true).consistent = true ∧ ∀ w, (exec true false true).rf w 7 = false := by
  decide +kernel

theorem two {j : Nat} (h : j < 2) : j = 0 ∨ j = 1 := by omega

/-- The counter of the consistent execution meets every hypothesis of `merge_sees`. -/
def mergeCounter : Counter (exec true true true) 2 Int.toNat (2 ^ 32) where
  rd j := if j = 0 then 1 else 4
  wr j := if j = 0 then 2 else 5
  mine _ := 1
  addr := counter
  rmw j hj := by rcases two hj with rfl | rfl <;> decide
  reads j hj := by rcases two hj with rfl | rfl <;> decide
  writes j hj := by rcases two hj with rfl | rfl <;> decide
  acquire j hj := by rcases two hj with rfl | rfl <;> decide
  release j hj := by rcases two hj with rfl | rfl <;> decide
  address j hj := by rcases two hj with rfl | rfl <;> decide
  generic j hj := by rcases two hj with rfl | rfl <;> decide
  strong j hj l hl := by rcases two hj with rfl | rfl <;> rcases two hl with rfl | rfl <;> decide
  distinct j hj l hl := by rcases two hj with rfl | rfl <;> rcases two hl with rfl | rfl <;> decide
  only := by decide
  zero := rfl
  add j hj := by rcases two hj with rfl | rfl <;> decide
  pos _ _ := Nat.one_pos

/-- `merge_sees` applies to the example: split 1, the last, sees split 0's partial. -/
theorem merge_example_sees : (∃ w, (exec true true true).rf w 6 = true) ∧
    ∀ w, (exec true true true).rf w 6 = true → (exec true true true).co w 0 = false :=
  merge_sees merge_ok_consistent mergeCounter (m := 1) (by decide +kernel) (by decide +kernel)
    (by decide +kernel) (j := 0) (by decide +kernel)
    (by decide +kernel) (by decide +kernel) (Execution.causeBase_of_order (Rel.tc_of (by decide +kernel)))
    (by decide +kernel) (by decide +kernel) (by decide +kernel) (Rel.rtc_of_tc (Rel.tc_of (by decide +kernel)))

end Examples.SplitMerge
