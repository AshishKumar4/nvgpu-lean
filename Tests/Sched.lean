import NV.PTX.Interp

/-!
`lake exe sched-tests`: the PTX interpreter's random schedules (`Machine.runRandom`), and its
labels' block scope (`tests/sched/labels.ptx`). On an
mbarrier hand-off without back-pressure (`tests/sched/phase.ptx`, guard 0) some schedule lets the
producer complete both phases before the consumer looks, and the run stops with the consumer at
its wait; with back-pressure (guard 1) every schedule completes. A kernel of held-back `tcgen05`
operations (`tests/diff/tc5cp.ptx`: copies and shifts, then a commit the threads wait on) gives,
under every schedule, the result of the run that completes them at issue.
-/

open NV NV.PTX

def le (v : Nat) (bytes : Nat) : List UInt8 := (List.range bytes).map fun i => (v >>> (8 * i)).toUInt8

def base : UInt64 := 0x7f0000000000

def load (path : String) (sm : String := "") : IO Sem.Prog := do
  let text ← IO.FS.readFile path
  let text := if sm.isEmpty then text else text.replace ".target sm_100a" s!".target {sm}"
  IO.ofExcept (do Sem.Prog.ofModule (← read text))

/-- A parameter block of 8-byte words, then a 4-byte one if given. -/
def params (ws : List Nat) (w4 : Option Nat := none) : ByteArray :=
  ByteArray.mk ((ws.flatMap (le · 8) ++ (w4.map (le · 4)).getD []).toArray)

def main : IO UInt32 := do
  let mut ok := true
  let phase ← load "tests/sched/phase.ptx"
  let launch (guard : Nat) : Launch :=
    { entry := "phase", grid := (1, 1, 1), block := (64, 1, 1), params := params [base.toNat] guard,
      global := Store.empty base 4 }
  -- Without back-pressure: some schedule of the first 200 is stuck, the consumer at its wait.
  let mut found := none
  for seed in [0:200] do
    match (← IO.ofExcept (Machine.init phase (launch 0))).runRandom seed 8 200_000 with
    | .ok _ => pure ()
    | .error (e, where_) => found := some (seed, e, where_); break
  match found with
  | some (seed, e, where_) =>
    -- Warp 1 spins polling the full barrier for parity 0, which it holds again.
    let atWait := (e.splitOn "polls").length > 1 && where_.any fun w => (w.splitOn "warp 1: line").length > 1 &&
      (w.splitOn "phase parity 0").length > 1 && (w.splitOn "polled for parity 0").length > 1
    IO.println s!"phase, no back-pressure: seed {seed} stuck ({e}); {where_}"
    unless atWait do IO.println "  FAIL: the report does not place warp 1 polling parity 0 of a barrier at parity 0"; ok := false
  | none => IO.println "phase, no back-pressure: FAIL: 200 schedules all completed"; ok := false
  -- Priority schedules find it too.
  let mut foundPct := false
  for seed in [0:50] do
    match (← IO.ofExcept (Machine.init phase (launch 0))).runRandom seed 8 200_000 (pct := true) with
    | .ok _ => pure ()
    | .error _ => foundPct := true; break
  IO.println s!"phase, no back-pressure, priority schedules: {if foundPct then "stuck" else "FAIL: 50 completed"}"
  ok := ok && foundPct
  -- With back-pressure: every schedule completes and the consumer saw both phases.
  let mut bad := 0
  for seed in [0:200] do
    match (← IO.ofExcept (Machine.init phase (launch 1))).runRandom seed 8 200_000 with
    | .ok m => if m.global.read base 4 != #[1] then bad := bad + 1
    | .error _ => bad := bad + 1
  IO.println s!"phase, back-pressure: {200 - bad} of 200 schedules complete"
  ok := ok && bad == 0
  -- Labels scoped by block.
  let lab ← load "tests/sched/labels.ptx"
  let ll : Launch := { entry := "labels", grid := (1, 1, 1), block := (1, 1, 1),
                       params := params [base.toNat], global := Store.empty base 4 }
  let got ← IO.ofExcept (do return (← interpRun lab ll).global.read base 4)
  IO.println s!"labels: out[0] = {got[0]!} (11 when each block's branch reaches its own label)"
  ok := ok && got == #[11]
  -- Held-back tcgen05 operations: every schedule's result is the blocking run's.
  let tc ← load "tests/diff/tc5cp.ptx"
  let input := (Array.range 2048).map fun i => ((i * 2654435761) % 2 ^ 32).toUInt32
  let out := base
  let inb := base + 65536
  let g := (Store.empty base (65536 + 8192)).write inb (8192) input 0
  let l : Launch := { entry := "tc5cp", grid := (1, 1, 1), block := (128, 1, 1),
                      params := params [out.toNat, inb.toNat], global := g }
  let want ← IO.ofExcept (do return (← interpRun tc l).global.read out (4 * 64 * 128))
  let mut diff := 0
  for seed in [0:20] do
    match (← IO.ofExcept (Machine.init tc l)).runRandom seed 8 50_000_000 with
    | .ok m => if m.global.read out (4 * 64 * 128) != want then diff := diff + 1
    | .error (e, _) => IO.println s!"tc5cp seed {seed}: {e}"; diff := diff + 1
  IO.println s!"tc5cp, held-back tcgen05: {20 - diff} of 20 schedules match the blocking run"
  ok := ok && diff == 0
  return if ok then 0 else 1
