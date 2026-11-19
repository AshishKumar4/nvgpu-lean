import NV.PTX.Memory.Litmus
import Lean.Data.Json

/-!
`lake exe litmus tests.json`: decide litmus tests given as JSON (the encoding Lean derives for
`NV.PTX.Litmus.Test`, produced by `tools/litmus/crosscheck.py`) and print one line per command:
`test command allowed|forbidden`, where `allowed` means some consistent execution satisfies the
command's condition (for `assert` and `check`: violates it).
-/

open Lean (Json FromJson fromJson?)
open NV.PTX NV.PTX.Litmus

deriving instance FromJson for Scope, Proxy, Place, Sem, Val, Instr, Litmus.Thread, Cond, Expect,
  Litmus.Command, Test

def report (t : Litmus.Test) : List String :=
  if !regsDisjoint t then t.commands.map fun c => s!"{t.name} {c.name} invalid" else
  let os := outcomes t
  t.commands.map fun c =>
    s!"{t.name} {c.name} {if c.witnessed os then "allowed" else "forbidden"}"

def main (args : List String) : IO UInt32 := do
  let [path] := args | IO.eprintln "usage: litmus <tests.json>"; return 2
  let tests ← IO.ofExcept <| (Json.parse (← IO.FS.readFile path)).bind (fromJson? (α := List Litmus.Test))
  for t in tests do
    for line in report t do IO.println line
  return 0
