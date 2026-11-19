import NV.PTX.Memory.Model

/-!
# Litmus tests

A litmus test is a small PTX program plus a condition on its final registers. `outcomes`
enumerates every candidate execution (each read's write, each location's coherence order, the
`fence.sc` order) and keeps the ones `Execution.consistent` accepts, as herd and Alloy do.
The instruction forms and commands mirror the nvlitmus format of NVlabs/mixedproxy
(`src/grammar.lark`), so its tests run here unchanged (`tools/litmus/crosscheck.py`).
-/

namespace NV.PTX.Litmus

/-- Memory-operation semantics, PTX ISA §8.4. -/
inductive Sem where
  | weak | relaxed | acquire | release | acqRel | sc
  deriving DecidableEq, Repr, Inhabited

inductive Val where
  | const (v : Int)
  | reg (r : Nat)
  deriving DecidableEq, Repr, Inhabited

inductive Instr where
  /-- `ld{.sem}{.scope} r, [a]`, optionally asserting the value it returns (nvlitmus `== v`). -/
  | ld (reg addr : Nat) (sem : Sem) (scope : Scope) (proxy : Proxy := .generic)
      (expect : Option Int := none)
  | st (addr : Nat) (v : Val) (sem : Sem) (scope : Scope) (proxy : Proxy := .generic)
  /-- `atom.add`: a read and a write joined by `rmw`. -/
  | atomAdd (reg addr : Nat) (v : Val) (sem : Sem) (scope : Scope) (proxy : Proxy := .generic)
      (expect : Option Int := none)
  /-- `red.add`: like `atomAdd` without a result; its read never acquires (§8.11.1). -/
  | redAdd (addr : Nat) (v : Val) (sem : Sem) (scope : Scope) (proxy : Proxy := .generic)
  | fence (sem : Sem) (scope : Scope)
  | proxyFence (p : Proxy)
  | aliasFence
  /-- `bar.sync id`: an arrival, then a wait for every arrival on `id` in the CTA. -/
  | barSync (id : Nat)
  | barArrive (id : Nat)
  | clusterArrive
  | clusterWait
  deriving Repr, Inhabited

structure Thread where
  id : Nat
  place : Place
  code : List Instr
  deriving Repr, Inhabited

inductive Cond where
  | eq (a b : Val) | neq (a b : Val)
  | and (p q : Cond) | or (p q : Cond) | not (p : Cond)
  deriving Repr, Inhabited

/-- nvlitmus commands: `assert` expects the condition in every execution, `permit` in some,
`check` reports whether some execution violates it. -/
inductive Expect where
  | assert | permit | check
  deriving DecidableEq, Repr, Inhabited

structure Command where
  name : String
  expect : Expect
  cond : Cond
  deriving Repr, Inhabited

structure Test where
  name : String
  threads : List Thread
  /-- Physical location of each address: virtual aliases share one (§8.2.2). -/
  loc : List (Nat × Nat) := []
  commands : List Command
  deriving Repr, Inhabited

/-! ## Events of a program -/

/-- An event with the bookkeeping the enumeration needs. -/
structure Ev where
  e : Event
  /-- Destination register of a read. -/
  reg : Option Nat := none
  /-- Value a write stores; for an `rmw` write, the operand added to its read's value. -/
  wval : Val := .const 0
  expect : Option Int := none
  rmwRead : Bool := false
  deriving Repr, Inhabited

def Sem.acquires : Sem → Bool
  | .acquire | .acqRel | .sc => true
  | _ => false

def Sem.releases : Sem → Bool
  | .release | .acqRel | .sc => true
  | _ => false

/-- A weak access has the scope of its thread (Alloy's `scope in Thread`). -/
def effScope (sem : Sem) (s : Scope) : Scope := if sem == .weak then .thread else s

def Instr.events (t : Nat) : Instr → List Ev
  | .ld r a sem s p x =>
    [{ e := { thread := t, kind := .read, scope := effScope sem s, acquire := sem.acquires,
              addr := a, proxy := p }, reg := some r, expect := x }]
  | .st a v sem s p =>
    [{ e := { thread := t, kind := .write, scope := effScope sem s, release := sem.releases,
              addr := a, proxy := p }, wval := v }]
  | .atomAdd r a v sem s p x =>
    [{ e := { thread := t, kind := .read, scope := effScope sem s, acquire := sem.acquires,
              addr := a, proxy := p }, reg := some r, expect := x, rmwRead := true },
     { e := { thread := t, kind := .write, scope := effScope sem s, release := sem.releases,
              addr := a, proxy := p }, wval := v }]
  | .redAdd a v sem s p =>
    [{ e := { thread := t, kind := .read, scope := effScope sem s, addr := a, proxy := p,
              reduction := true }, rmwRead := true },
     { e := { thread := t, kind := .write, scope := effScope sem s, release := sem.releases,
              addr := a, proxy := p }, wval := v }]
  | .fence sem s =>
    [{ e := { thread := t, kind := if sem == .sc then .fenceSC else .fenceAcqRel, scope := s } }]
  | .proxyFence p => [{ e := { thread := t, kind := .proxyFence p } }]
  | .aliasFence => [{ e := { thread := t, kind := .aliasFence } }]
  | .barSync id =>
    [{ e := { thread := t, kind := .barArrive id, scope := .cta } },
     { e := { thread := t, kind := .barWait id, scope := .cta } }]
  | .barArrive id => [{ e := { thread := t, kind := .barArrive id, scope := .cta } }]
  | .clusterArrive => [{ e := { thread := t, kind := .clusterArrive, scope := .cluster } }]
  | .clusterWait => [{ e := { thread := t, kind := .clusterWait, scope := .cluster } }]

/-- The program's events, each thread's in program order. -/
def Test.events (test : Test) : Array Ev :=
  (test.threads.flatMap fun th => th.code.flatMap (·.events th.id)).toArray

/-! ## Enumerating candidate executions -/

/-- Every strict partial order on `xs`, as the list of its ordered pairs. -/
def strictOrders (xs : List Nat) : List (List (Nat × Nat)) :=
  let pairs := (xs.zip (List.range xs.length)).flatMap fun (a, i) =>
    (xs.drop (i + 1)).map fun b => (a, b)
  let choices := pairs.foldr (fun (a, b) acc =>
    acc.flatMap fun rest => [rest, (a, b) :: rest, (b, a) :: rest]) [[]]
  choices.filter fun rel =>
    rel.all fun (a, b) => rel.all fun (c, d) => b != c || rel.contains (a, d) && a != d

def cartesian {α : Type} : List (List α) → List (List α)
  | [] => [[]]
  | xs :: rest => xs.flatMap fun x => (cartesian rest).map (x :: ·)

structure Outcome where
  regs : List (Nat × Int)
  deriving Repr, BEq, Inhabited

def Val.eval (regs : List (Nat × Int)) : Val → Int
  | .const v => v
  | .reg r => (regs.lookup r).getD 0

def Cond.eval (regs : List (Nat × Int)) : Cond → Bool
  | .eq a b => a.eval regs == b.eval regs
  | .neq a b => a.eval regs != b.eval regs
  | .and p q => p.eval regs && q.eval regs
  | .or p q => p.eval regs || q.eval regs
  | .not p => !p.eval regs

section
variable (test : Test)

/-- Each register is written by one thread only, so final values are unambiguous. -/
def regsDisjoint : Bool :=
  let defs := test.threads.map fun th => (th.code.flatMap (·.events th.id)).filterMap (·.reg)
  (List.range defs.length).all fun i => (List.range defs.length).all fun j =>
    i == j || (defs[i]!).all fun r => !(defs[j]!).contains r

/-- Final registers of the consistent candidate executions. -/
def outcomes : List Outcome := Id.run do
  let evs := test.events
  let n := evs.size
  let ev (i : Fin n) : Ev := evs[i]
  let locOf (a : Nat) : Nat := (test.loc.lookup a).getD a
  let placeOf (t : Nat) : Place := ((test.threads.find? (·.id == t)).map (·.place)).getD default
  let idx := List.finRange n
  let isRead (i : Fin n) := (ev i).e.isRead
  let isWrite (i : Fin n) := (ev i).e.isWrite
  let sameThread (a b : Fin n) := (ev a).e.thread == (ev b).e.thread
  let po : Rel n := Rel.ofFn fun a b => sameThread a b && a.val < b.val
  -- The write of an rmw is the event right after its read.
  let rmw : Rel n := Rel.ofFn fun a b => (ev a).rmwRead && b.val == a.val + 1
  -- The latest definition of register `r` before event `i` in `i`'s thread.
  let lastDef (i : Fin n) (r : Nat) : Option (Fin n) :=
    (idx.filter fun j => sameThread j i && j.val < i.val && (ev j).reg == some r).getLast?
  -- Data dependencies: a write's register operand, and an rmw's read.
  let dep : Rel n := Rel.ofFn fun a b => rmw a b ||
    match (ev b).wval with
    | .reg r => isWrite b && lastDef b r == some a
    | .const _ => false
  let sameLoc (a b : Fin n) := locOf (ev a).e.addr == locOf (ev b).e.addr
  -- Instance `k` of a barrier in a thread: its `k`-th arrival (or wait) there.
  let ordinal (i : Fin n) (same : Kind → Bool) : Nat :=
    (idx.filter fun j => sameThread j i && j.val < i.val && same (ev j).e.kind).length
  let barRel : Rel n := Rel.ofFn fun a b =>
    match (ev a).e.kind, (ev b).e.kind with
    | .barArrive i, .barWait j =>
      i == j && (placeOf (ev a).e.thread).cta == (placeOf (ev b).e.thread).cta &&
        ordinal a (· == .barArrive i) == ordinal b (· == .barWait j)
    | .clusterArrive, .clusterWait =>
      (placeOf (ev a).e.thread).cluster == (placeOf (ev b).e.thread).cluster &&
        ordinal a (· == .clusterArrive) == ordinal b (· == .clusterWait)
    | _, _ => false
  let writesAt (l : Nat) := (idx.filter fun i => isWrite i && locOf (ev i).e.addr == l).map (·.val)
  let locs := (idx.filter isWrite).map (fun i => locOf (ev i).e.addr) |>.eraseDups
  let rfChoices := (idx.filter isRead).map fun r =>
    (none :: ((idx.filter fun w => isWrite w && sameLoc r w).map some)).map (r, ·)
  let coChoices := cartesian (locs.map fun l => strictOrders (writesAt l))
  let fences := (idx.filter fun i => (ev i).e.isFenceSC).map (·.val)
  let mut out : List Outcome := []
  for rfc in cartesian rfChoices do
    let rf : Rel n := Rel.ofFn fun w r => rfc.any fun (r', w') => r' == r && w' == some w
    -- Values flow along rf and dependencies; `n + 1` rounds reach the fixpoint when rf ∪ dep is
    -- acyclic. Whatever the rounds produce is checked below: reads by `rfWF`, writes here.
    let mut val : Array Int := Array.replicate n 0
    for _ in [0:n + 1] do
      val := Array.ofFn fun (i : Fin n) =>
        if isRead i then
          match rfc.find? (·.1 == i) with
          | some (_, some w) => val[w.val]!
          | _ => 0
        else if isWrite i then
          let operand := match (ev i).wval with
            | .const v => v
            | .reg r => ((lastDef i r).map fun j => val[j.val]!).getD 0
          if i.val > 0 && (ev ⟨i.val - 1, by omega⟩).rmwRead then val[i.val - 1]! + operand
          else operand
        else 0
    let writesOk := idx.all fun i => !isWrite i ||
      let operand := match (ev i).wval with
        | .const v => v
        | .reg r => ((lastDef i r).map fun j => val[j.val]!).getD 0
      val[i.val]! == if i.val > 0 && (ev ⟨i.val - 1, by omega⟩).rmwRead then val[i.val - 1]! + operand
        else operand
    let expectOk := idx.all fun i => match (ev i).expect with
      | some v => val[i.val]! == v
      | none => true
    if !(writesOk && expectOk) then continue
    for cos in coChoices do
      let co : Rel n := Rel.ofFn fun a b => cos.any (·.contains (a.val, b.val))
      for scs in strictOrders fences do
        let sc : Rel n := Rel.ofFn fun a b => scs.contains (a.val, b.val)
        let x : Execution n :=
          { ev := fun i => (ev i).e, place := placeOf, loc := locOf, init := fun _ => 0,
            val := fun i => val[i.val]!,
            po, rf, co, rmw, dep, sc, bar := barRel }
        if x.consistent then
          -- Final registers: each register's last definition in its thread.
          let regs := (idx.filterMap fun i => (ev i).reg).eraseDups.filterMap fun r =>
            ((idx.filter fun i => (ev i).reg == some r).getLast?).map fun i => (r, val[i.val]!)
          out := { regs } :: out
  return out

end

/-- Whether some consistent execution witnesses the command: satisfies a `permit` condition, or
violates an `assert`/`check` condition. -/
def Command.witnessed (c : Command) (os : List Outcome) : Bool :=
  match c.expect with
  | .permit => os.any (c.cond.eval ·.regs)
  | .assert | .check => os.any (!c.cond.eval ·.regs)

end NV.PTX.Litmus
