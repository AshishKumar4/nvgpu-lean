import NV.PTX.Syntax

/-!
`lake exe ptx check PATH…`: read (parse and decode) and type-check every PTX file named, or
found under a named directory, and report each failure and a summary. Exits nonzero if any file
fails.
-/

open NV.PTX

def countInstrs {ι : Type} : List (Stmt ι) → Nat
  | [] => 0
  | .instr .. :: ss => 1 + countInstrs ss
  | .block b :: ss => countInstrs b + countInstrs ss
  | _ :: ss => countInstrs ss

/-- The `.ptx` files under `p` (or `p` itself), sorted. -/
def ptxFiles (p : System.FilePath) : IO (Array System.FilePath) := do
  if ← p.isDir then
    let all ← p.walkDir
    return (all.filter (·.extension == some "ptx")).qsort (·.toString < ·.toString)
  return #[p]

def main (args : List String) : IO UInt32 := do
  match args with
  | "check" :: paths =>
    let mut files := 0
    let mut bad := 0
    let mut instrs := 0
    for p in paths do
      for f in ← ptxFiles p do
        files := files + 1
        match read (← IO.FS.readFile f) with
        | .ok m =>
          for it in m.items do
            if let .func fn := it then instrs := instrs + countInstrs (fn.body.getD [])
          let ds := check m
          if !ds.isEmpty then
            bad := bad + 1
            for d in ds.toList.take 5 do IO.println s!"FAIL {f}:{d.line}: {d.msg}"
        | .error e => bad := bad + 1; IO.println s!"FAIL {f}: {e}"
    IO.println s!"{files} files, {files - bad} read, {bad} failed; {instrs} instructions"
    return if bad == 0 then 0 else 1
  | _ => IO.eprintln "usage: ptx check PATH…"; return 2
