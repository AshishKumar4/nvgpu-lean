import NV.SASS.Read
import NV.SASS.Json

/-!
`lake exe sass-dump LISTING ELF KERNEL`: a kernel as JSON lines (`NV.SASS.Json`).
-/

open NV.SASS

def main (args : List String) : IO UInt32 := do
  let [sass, elf, name] := args | IO.eprintln "usage: sass-dump LISTING ELF KERNEL"; return 2
  match Read.kernels (← IO.FS.readFile sass) (← IO.FS.readFile elf) (some [name]) with
  | .error e => IO.eprintln e; return 1
  | .ok ks =>
    let some k := ks.find? (·.name == name) | IO.eprintln s!"no kernel {name}"; return 1
    for l in Json.kernelLines k do IO.println l
    return 0
