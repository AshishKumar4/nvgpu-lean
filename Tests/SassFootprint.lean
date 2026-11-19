import Tests.Kernels

/-!
`lake exe sass-footprint`: every instruction of the test kernels (the `tests/sass` listings and
the `tests/sem` and `tests/diff` PTX compiled by the pinned `ptxas` for `NV_ARCH`) against its
footprint (`NV.SASS.footprint`), on random warp states. That an outcome depends only on the reads,
that writebacks are local and that effects are confined are theorems (`Op.readsWithin`,
`Op.wbLocal`, `Op.confined`); what is checked here is that the footprint is complete, so that the
semantics the theorems are about is the one the instruction means:

* reads: given a view of every location it might read (`candidates`), the semantics' outcome
  from a second state that differs from the first only outside the reads (control state kept)
  is the same — continuation states carried over, the same writebacks on a common state, and the
  same request from two read-release states that differ only outside the reads;
* writes and sets: the outcome as the semantics builds it changes at issue nothing outside the
  sets and in its writeback nothing outside the writes, so the confinement `Op.run` applies
  changes nothing.

Each instruction runs on several random states, all lanes at it. Exits nonzero on any violation.
-/

open NV NV.SASS

/-- A deterministic generator (splitmix64). -/
structure Rng where
  s : UInt64

def Rng.next (g : Rng) : UInt64 × Rng :=
  let s := g.s + 0x9E3779B97F4A7C15
  let z := (s ^^^ (s >>> 30)) * 0xBF58476D1CE4E5B9
  let z := (z ^^^ (z >>> 27)) * 0x94D049BB133111EB
  (z ^^^ (z >>> 31), ⟨s⟩)

def Rng.words (g : Rng) (n : Nat) : Array UInt32 × Rng := Id.run do
  let mut g := g
  let mut out := Array.mkEmpty n
  for _ in [0:n] do
    let (v, g') := g.next
    g := g'
    -- A quarter small values (counts, shifts, indices), a quarter edge values (so comparisons
    -- and masks take both outcomes), half anything.
    let edges : Array UInt32 := #[0, 1, 2, 0x1f, 0x20, 0x7fffffff, 0x80000000, 0xffffffff]
    let x := (v >>> 32).toUInt32
    out := out.push (match v &&& 3 with
      | 0 => x &&& 0x3f
      | 1 => edges[(x % 8).toNat]!
      | _ => x)
  return (out, g)

/-- A random warp at instruction `pc`, all lanes live and there. -/
def randomWarp (regs pc : Nat) (g : Rng) : Warp × Rng :=
  let (r, g) := g.words (regs * 32)
  let (ur, g) := g.words 255
  let (p, g) := g.words 7
  let (up, g) := g.words 7
  let (b, g) := g.words 16
  let (bar, g) := g.words 1
  let w := Warp.init regs 32
  ({ w with conv := some pc, r := r, ur := ur, p := p, up := up.map (· &&& 1 == 1),
            b := b, barResult := bar[0]! }, g)

/-- `w` with every location outside `keep` taken from `x`. -/
def mixOutside (keep : Array SASS.Loc) (w x : Warp) : Warp := Id.run do
  let keepR := (Array.range 256).map fun n => keep.contains (.r n)
  let mut r := w.r
  for i in [0:r.size] do
    unless keepR[i / 32]! do r := r.set! i x.r[i]!
  let ur := (Array.range 255).map fun n => if keep.contains (.ur n) then w.ur[n]! else x.ur[n]!
  let p := (Array.range 7).map fun n => if keep.contains (.p n) then w.p[n]! else x.p[n]!
  let up := (Array.range 7).map fun n => if keep.contains (.up n) then w.up[n]! else x.up[n]!
  let b := (Array.range 16).map fun n => if keep.contains (.b n) then w.b[n]! else x.b[n]!
  let barResult := if keep.contains .bar then w.barResult else x.barResult
  return { w with r, ur, p, up, b, barResult }

/-- Locations where `a` and `b` differ. -/
def diff (a b : Warp) : Array SASS.Loc := Id.run do
  let mut out := #[]
  for i in [0:max a.r.size b.r.size] do
    if a.r[i]? != b.r[i]? then
      let l := SASS.Loc.r (i / 32)
      unless out.contains l do out := out.push l
  for n in [0:255] do
    if a.ur[n]? != b.ur[n]? then out := out.push (.ur n)
  for n in [0:7] do
    if a.p[n]? != b.p[n]? then out := out.push (.p n)
    if a.up[n]? != b.up[n]? then out := out.push (.up n)
  for n in [0:16] do
    if a.b[n]? != b.b[n]? then out := out.push (.b n)
  if a.barResult != b.barResult then out := out.push .bar
  return out

def reqText (r : Req) : String := toString (repr r)

def kind : Effect Warp → String
  | .next _ => "next" | .reg .. => "reg" | .mem .. => "mem" | .sync .. => "sync"
  | .collect .. => "collect" | .async .. => "async" | .exit => "exit" | .fault _ => "fault"

/-- Every location instruction `i`'s semantics might read: its operands' registers with the next
seven (pairs, vectors, fragments), its uniform registers with the next three, `R4`–`R7`
(`printf`'s arguments), every predicate and barrier, the `printf` log and the barrier answer. -/
def candidates (i : Instr) : Array SASS.Loc := Id.run do
  let mut out : Array SASS.Loc := #[]
  let regOf : Operand → List Reg
    | .reg g _ => [g]
    | .mem b _ u _ _ => b.toList ++ u.toList
    | .scaled (.mem b _ u _ _) _ => b.toList ++ u.toList
    | .cbank _ (some g) _ => [g]
    | .gdesc g _ | .udesc _ g _ => [g]
    | .neg (.reg g _) | .not (.reg g _) => [g]
    | _ => []
  for o in i.operands do
    for g in regOf o do
      match g.file with
      | .r => for j in [0:8] do out := out.push (.r (g.idx + j))
      | .ur => for j in [0:4] do out := out.push (.ur (g.idx + j))
      | _ => pure ()
  for j in [4:8] do out := out.push (.r j)
  for j in [0:7] do out := out ++ #[.p j, .up j]
  for j in [0:16] do out := out.push (.b j)
  return out ++ #[.log, .bar]

/-- Problems with instruction `pc` of `k` on one random state (and its perturbation). -/
def checkOne (k : Kernel) (op : Op) (env : Env) (pc : Nat) (g : Rng) : List String × Rng :=
  -- Every register: semantics may read the one after a named register (pairs, vectors).
  let regs := 256
  let _ := k
  let (w, g) := randomWarp regs pc g
  let (noise, g) := randomWarp regs pc g
  let (x, g) := randomWarp regs (pc + 1) g
  let (respW, g) := g.words (32 * 4)
  let (noise2, g) := randomWarp regs (pc + 1) g
  let fp := op.fp
  let w' := mixOutside fp.reads w noise
  -- The outcome as the semantics built it, before `Op.run` confines it to the footprint: that the
  -- footprint covers it is what makes the confinement change nothing.
  -- The semantics given a view of every location it might read (`candidates`), not only the
  -- footprint's reads: what it reads beyond them is then visible.
  -- The footprint's reads first: the compiled forms index the view by their position.
  -- Registers past the warp's file are not locations (a reading error, `decodeAt`).
  let cand := op.fp.reads ++ (candidates (k.code[pc]!)).filter fun
    | .r n => n < regs || n == 255 | .ur n => n < 255 | _ => true
  let e := (op.out env (View.of cand w) w.blank fullMask).plain cand w
  let e' := (op.out env (View.of cand w') w'.blank fullMask).plain cand w'
  let at_ := s!"{k.name} pc {pc} {(k.code[pc]!).full}"
  let outside (what : String) (ls : Array SASS.Loc) (allowed : Array SASS.Loc) : List String :=
    let bad := ls.filter (!allowed.contains ·)
    if bad.isEmpty then [] else [s!"{at_}: {what} {repr bad.toList}"]
  -- Carried: control and sets as in s, the rest as w'.
  let carried (s s' : Warp) : List String :=
    (if s'.ctrl != s.ctrl then [s!"{at_}: control state depends on unread state"] else []) ++
    outside "issue state depends on unread" ((diff s' s).filter fp.sets.contains) #[] ++
    outside "issue changes outside sets" (diff s w.blank) fp.sets ++
    outside "carried state differs" ((diff s' w'.blank).filter (!fp.sets.contains ·)) #[]
  let wbCheck (f f' : Warp → Warp) : List String :=
    outside "writeback depends on unread state" (diff (f x) (f' x)) #[] ++
    outside "writeback writes outside writes" (diff (f x) x) fp.writes
  let problems : List String := match e, e' with
    | .next s, .next s' => carried s s'
    | .reg s wb, .reg s' wb' => carried s s' ++ wbCheck wb wb'
    | .mem r s wb, .mem r' s' wb' =>
      carried s s' ++ wbCheck (wb respW) (wb' respW) ++
        (if reqText (r x) != reqText (r' (mixOutside fp.reads x noise2)) then
          [s!"{at_}: request depends on unread state"] else [])
    | .sync op s, .sync op' s' =>
      carried s s' ++ (if op != op' then [s!"{at_}: sync depends on unread state"] else [])
    | .collect _ _ _ s wb, .collect _ _ _ s' wb' => carried s s' ++ wbCheck (wb respW) (wb' respW)
    | .async op s wb, .async op' s' wb' =>
      carried s s' ++ wbCheck (wb respW) (wb' respW) ++
        (if toString (repr (op x)) != toString (repr (op' (mixOutside fp.reads x noise2))) then
          [s!"{at_}: async operation depends on unread state"] else [])
    | .exit, .exit | .fault _, .fault _ => []
    | a, b => [s!"{at_}: {kind a} becomes {kind b} when unread state changes"]
  (problems, g)

/-- Warp 1 of CTA 1 of four 128-thread CTAs, constant bank 0 filled with distinct words. -/
def env : Env :=
  let c := (List.range 1024).foldl (fun (s : Store) j =>
    s.writeWord (4 * j) (j.toUInt32 * 2654435761)) (Store.empty 0 4096)
  { ctx := { ntid := (128, 1, 1), nctaid := (4, 1, 1), ctaid := (1, 0, 0), warp := 1,
             threads := 128 },
    const0 := c, clock := 1000, timeNs := 5000 }

def checkKernel (k : Kernel) (samples : Nat) : List String × Nat := Id.run do
  let ops := decode k
  let mut out := []
  let mut g : Rng := ⟨k.name.hash⟩
  for pc in [0:ops.size] do
    for _ in [0:samples] do
      let (ps, g') := checkOne k ops[pc]! env pc g
      g := g'
      out := out ++ ps
  return (out.eraseDups, ops.size)

def fixtures : List String := ["basic", "misc", "wgmma", "tma", "elect", "ops", "jumps"]

def main : IO UInt32 := do
  let mut ks : Array Kernel := #[]
  let a ← Tests.arch
  let fdir := if a == "sm_90a" then "tests/sass" else s!"tests/sass/{a}"
  for f in fixtures do
    -- A fixture the target cannot compile is absent (`gen.sh`'s SKIPPED).
    unless ← System.FilePath.pathExists s!"{fdir}/{f}.sass" do continue
    let sass ← IO.FS.readFile s!"{fdir}/{f}.sass"
    let elf ← IO.FS.readFile s!"{fdir}/{f}.elf"
    ks := ks ++ (← IO.ofExcept (Read.kernels sass elf))
  for dir in ["tests/sem", "tests/diff"] do
    for e in (← System.FilePath.readDir dir) do
      if e.path.extension == some "ptx" then
        try ks := ks ++ (← Tests.compile e.path.toString)
        catch err => match Tests.unsupported err with
          | some why => IO.println s!"{e.path}: skipped on {a}: {why}"
          | none => throw err
  let mut bad := 0
  let mut instrs := 0
  for k in ks do
    let (ps, n) := checkKernel k 4
    instrs := instrs + n
    for p in ps.take 20 do IO.println p
    bad := bad + ps.length
  IO.println s!"{ks.size} kernels, {instrs} instructions, 4 states each: {bad} violations"
  -- The check can fail: an arithmetic instruction whose footprint forgets its sources, or its
  -- destination, must be caught.
  let mut mutants := 0
  let mut missed := 0
  for k in ks do
    let ops := decode k
    for pc in [0:ops.size] do
      let i := k.code[pc]!
      unless ["IMAD", "IADD3", "LOP3", "FFMA", "FADD", "SHF", "ISETP", "SEL", "LDG", "STG", "LDS",
          "STS", "LDC", "ATOMS", "ATOMG", "SHFL", "HMMA"].contains i.opcode &&
          i.guard.isNone do continue
      -- An instruction the semantics faults on (a target's unpublished tensor-core datapath)
      -- reads and writes nothing: no mutant of it can be caught.
      if (ops[pc]!).outOf env (Warp.init 256 32) fullMask matches .fault _ then continue
      for op in [{ ops[pc]! with fp.reads := #[] }, { ops[pc]! with fp.writes := #[] }] do
        if op.fp.reads == ops[pc]!.fp.reads && op.fp.writes == ops[pc]!.fp.writes then continue
        mutants := mutants + 1
        -- Caught on some state: one state may happen to compute the same value either way.
        let caught := (List.range 16).any fun j =>
          !(checkOne k op env pc ⟨(pc * 16 + j).toUInt64⟩).1.isEmpty
        unless caught do
          missed := missed + 1
          IO.println s!"mutant not caught: {k.name} pc {pc} {i.full}"
  -- An instruction can be insensitive to a source on every sampled state (a comparison with a
  -- constant it never meets), so a few mutants may survive; most must not.
  IO.println s!"{mutants} mutated footprints, {missed} not caught"
  return if bad == 0 && mutants > 0 && missed * 100 ≤ mutants then 0 else 1
