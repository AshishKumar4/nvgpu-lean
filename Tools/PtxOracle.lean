import NV.PTX.Interp

/-!
`lake exe ptx-oracle FILE`: the PTX semantics against ROCetta's (University of Rochester) PTX
semantics in C, as run by `tools/ptx/rocetta/run.sh`. FILE holds, for each ROCetta function
`execute_<name>`, a header `# <name> <argument types> -> <result types>` and lines
`<name> <argument bits> = <result bits>`. Each function becomes one PTX instruction (`add_rz_f32`
is `add.rz.f32`; `setp_q_…` writes `p|q`; `…_invert_…` negates the predicate source), run by
`Thread.step` on registers holding the arguments.

Results are compared bit for bit, except that any NaN matches any NaN (ROCetta computes with the
host's floating point, which keeps NaN payloads; NVIDIA returns the canonical NaN). The report
lists, per function that disagrees, how many vectors do and the first one; functions this
semantics cannot read are listed apart. Two kinds of disagreement are explained and allowed:

* `.approx` instructions: ROCetta computes them with the host's libm, not NVIDIA's MUFU, whose
  model is `NV.Num.Approx`'s;
* behaviour the ISA leaves undefined, where this semantics faults (`fns` with a base above 31,
  §9.7.1.18) and ROCetta picks a value.

Exit status 1 if any other disagreement, or any unread function, remains.
-/

open NV NV.PTX NV.PTX.Sem

def regTy : String → Option Ty
  | "int16_t" => some .s16 | "uint16_t" => some .u16 | "int32_t" => some .s32
  | "uint32_t" => some .u32 | "int64_t" => some .s64 | "uint64_t" => some .u64
  | "float" => some .f32 | "double" => some .f64 | "pred" => some .pred
  | _ => none

structure Fn where
  name : String
  one : OneInstr

/-- The PTX instruction for a ROCetta function. -/
def build (name : String) (ins outs : List Ty) : Except String Fn := do
  let toks := name.splitOn "_"
  let invert := toks.contains "invert"
  let toks := toks.filter (· != "invert")
  let toks := if toks.head? == some "setp" && toks[1]? == some "q" then toks.eraseIdx 1 else toks
  return { name, one := ← OneInstr.make (".".intercalate toks) ins outs invert }

/-- Why a disagreement on function `name` is expected, if it is: `faulted` says whether this
semantics faulted on the vector. -/
def explained (name : String) (faulted : Bool) : Option String :=
  if (name.splitOn "_").contains "approx" then some "approximation: host libm vs NV.Num.Approx"
  else if (name.splitOn "_").contains "full" then
    some "div.full: 2-ulp approximation vs host /"
  else if faulted && (name.splitOn "_").head? == some "fns" then some "undefined: ours faults"
  else none

def isNaNBits (t : Ty) (v : UInt64) : Bool :=
  match t with
  | .f32 => (v >>> 23) &&& 0xFF == 0xFF && v &&& 0x7FFFFF != 0
  | .f64 => (v >>> 52) &&& 0x7FF == 0x7FF && v &&& 0xFFFFFFFFFFFFF != 0
  | _ => false

def run (f : Fn) (args : List UInt64) : Except String (List UInt64) := f.one.run args

def hex (s : String) : Option UInt64 :=
  s.foldl (fun acc c => acc.bind fun v =>
    let d := if c.isDigit then some (c.toNat - '0'.toNat)
      else if 'a' ≤ c && c ≤ 'f' then some (c.toNat - 'a'.toNat + 10) else none
    d.map fun d => v * 16 + d.toUInt64) (some 0)

def main (args : List String) : IO UInt32 := do
  let [file] := args | IO.eprintln "usage: ptx-oracle FILE"; return 2
  let lines ← IO.FS.lines file
  let mut cur : Option Fn := none
  let mut unread : Array String := #[]
  let mut fns := 0
  let mut vecs := 0
  let mut bad : Array (String × Nat × Nat × String) := #[]
  let mut unexplained := 0
  let mut curBad := 0
  let mut curN := 0
  let mut first := ""
  let flush (cur : Option Fn) (curN curBad : Nat) (first : String)
      (bad : Array (String × Nat × Nat × String)) :=
    match cur with
    | some f => if curBad > 0 then bad.push (f.name, curBad, curN, first) else bad
    | none => bad
  for line in lines do
    if line.startsWith "# " then
      bad := flush cur curN curBad first bad
      curN := 0; curBad := 0; first := ""
      let ws := (String.ofList (line.toList.drop 2)).splitOn " "
      let name := ws.headD ""
      let (ins, outs) := (ws.drop 1).span (· != "->")
      match ins.mapM regTy, (outs.drop 1).mapM regTy with
      | some is, some os =>
        match build name is os with
        | .ok f => cur := some f; fns := fns + 1
        | .error e => cur := none; unread := unread.push s!"{name}: {e}"
      | _, _ => cur := none; unread := unread.push s!"{name}: argument types"
      continue
    let some f := cur | continue
    let ws := line.splitOn " "
    let (argsS, outS) := (ws.drop 1).span (· != "=")
    let some argv := argsS.mapM hex | continue
    let some want := (outS.drop 1).mapM hex | continue
    vecs := vecs + 1
    curN := curN + 1
    let ok : Bool := match run f argv with
      | .ok got => (got.zip (want.zip f.one.outs)).all fun (g, (w, t)) =>
          let m := if t == .pred then 1 else Sem.mask t.bits 0xFFFFFFFFFFFFFFFF
          (g &&& m) == (w &&& m) || (isNaNBits t g && isNaNBits t (w &&& m))
      | .error _ => false
    unless ok do
      let faulted := match run f argv with | .error _ => true | .ok _ => false
      if (explained f.name faulted).isNone then unexplained := unexplained + 1
      curBad := curBad + 1
      if first.isEmpty then
        let got := match run f argv with
          | .ok g => " ".intercalate (g.map fun v => s!"{v}") | .error e => s!"fault: {e}"
        first := s!"{line} ; ours: {got}"
  bad := flush cur curN curBad first bad
  IO.println s!"{fns} functions, {vecs} vectors; {bad.size} functions disagree \
    ({unexplained} vectors unexplained), {unread.size} not read"
  for (n, b, total, ex) in bad do
    let why := (explained n true).getD "UNEXPLAINED"
    IO.println s!"DIFF {n}: {b}/{total} [{why}]: {ex}"
  for u in unread do IO.println s!"UNREAD {u}"
  return if unexplained == 0 && unread.isEmpty then 0 else 1
