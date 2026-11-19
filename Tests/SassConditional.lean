import Tests.Kernels

/-!
`lake exe sass-conditional`: the device runtime's `cudaGraphSetConditional`, which kernels call to
steer a CUDA graph's conditional nodes (XLA's command buffers do, for loops and branches).
`tests/device/setcond.ptx`, compiled by the pinned `ptxas` for the tests' target, is read back with
the call as an external of its module and run on the engine: the thread named by the parameter
stores its word to the handle's word, and no other thread does.
-/

open NV NV.SASS NV.Core Tests

def main : IO UInt32 := do
  let ks ← compile "tests/device/setcond.ptx"
  let k ← kernel ks "setc"
  let mut bad := 0
  unless k.externs.any (·.2 == "cudaGraphSetConditional") do
    bad := bad + 1; IO.println "FAIL cudaGraphSetConditional is not an extern of the module"
  let values : Array UInt32 := (Array.range 64).map fun i => (0x100 + 7 * i).toUInt32
  for who in [0, 5, 33] do
    match ← run k 1 64 [#[0xdeadbeef], values] [le who 4] with
    | .error e => bad := bad + 1; IO.println s!"FAIL thread {who}: {e}"
    | .ok [cond, _] =>
      if cond == #[values[who]!] then IO.println s!"ok   thread {who} sets the condition to {cond[0]!}"
      else bad := bad + 1; IO.println s!"FAIL thread {who}: condition {cond}, expected {values[who]!}"
    | .ok _ => bad := bad + 1; IO.println s!"FAIL thread {who}: buffers lost"
  IO.println s!"{if bad == 0 then "ok" else "FAIL"}: cudaGraphSetConditional ({bad} failing)"
  return if bad == 0 then 0 else 1
