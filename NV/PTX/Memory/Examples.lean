import NV.PTX.Memory.Model

/-!
# Message passing, proven in the kernel

The ISA's message-passing example (§8.10.6) as one candidate execution: thread 0 writes `data`,
fences, sets `flag`; thread 1 reads the flag, fences, reads `data`. The execution in which thread 1
sees the flag but reads the initial `data` is inconsistent, decided by the kernel. With the
fences scoped to the CTA and the threads in different CTAs, the same execution is consistent.
-/

namespace NV.PTX.Examples

open Rel

/-- Events 0–2 are thread 0's `st data; fence; st flag`, events 3–5 thread 1's
`ld flag; fence; ld data`. -/
def mpEvents (s : Scope) : Fin 6 → Event
  | 0 => { thread := 0, kind := .write, addr := 0 }
  | 1 => { thread := 0, kind := .fenceAcqRel, scope := s }
  | 2 => { thread := 0, kind := .write, scope := s, addr := 1 }
  | 3 => { thread := 1, kind := .read, scope := s, addr := 1 }
  | 4 => { thread := 1, kind := .fenceAcqRel, scope := s }
  | 5 => { thread := 1, kind := .read, addr := 0 }

/-- Thread 1 reads the flag from thread 0's write and `data` from the initial state. -/
def mp (s : Scope) : Execution 6 where
  ev := mpEvents s
  place t := { cta := t, cluster := t, gpu := 0 }
  init _ := 0
  val i := if i.val = 5 then 0 else 1
  po := ofFn fun a b => (a.val < b.val) && (a.val < 3) == (b.val < 3)
  rf := ofFn fun w r => w.val = 2 && r.val = 3
  co := empty

theorem mp_gpu_forbidden : (mp .gpu).consistent = false := by decide +kernel

theorem mp_cta_allowed : (mp .cta).consistent = true := by decide +kernel

end NV.PTX.Examples
