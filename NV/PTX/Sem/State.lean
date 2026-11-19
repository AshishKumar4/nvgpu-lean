import NV.PTX.Syntax
import NV.Machine.Effect
import NV.Machine.Generic
import NV.Machine.Texture

/-!
# PTX threads: programs, operands and state

A decoded module is prepared for execution once (`Prog.ofModule`): each function's body is
flattened into an array of instructions (blocks only scope names, §9.7.14.1), labels
become instruction indices, registers become slots of a register file, and every operand is
resolved to an `Arg`. Module variables are placed in their state spaces, kernel parameters in the
parameter space, and the parameters of device functions and of call sequences in a per-frame
parameter area (§5.1.6.4).

A thread (`Thread`) is a stack of frames: function, program counter, register file and
parameter area. Register values are stored at their declared width in 64-bit slots (`.b128`
takes two); reading at an instruction type keeps the type's low bits, writing extends to the
register's width by the instruction type's signedness (§9.4.1, Tables 27 and 28).
-/

namespace NV.PTX.Sem

open NV (Windows)

/-- A register: its first slot and declared type. -/
structure RegRef where
  slot : Nat
  ty : Ty
  deriving Repr, Inhabited, BEq

/-- Where a named variable is. -/
inductive Loc where
  /-- A module variable, kernel parameter or `.local` variable: space and address. -/
  | mem (s : NV.Space) (addr : UInt64)
  /-- A device-function or call-sequence parameter: offset in the frame's parameter area. -/
  | frame (off : Nat)
  deriving Repr, Inhabited, BEq

inductive AddrBase where
  | imm
  | reg (r : RegRef)
  | loc (l : Loc)
  deriving Repr, Inhabited, BEq

/-- A resolved operand. -/
inductive Arg where
  | sink
  | reg (r : RegRef)
  | imm (e : Expr)
  | sreg (s : SReg)
  /-- The address of a variable (in its space) plus an offset: `mov r, var+8`. -/
  | sym (l : Loc) (off : Int)
  | func (f : Nat)
  | label (pc : Nat)
  | addr (base : AddrBase) (off : Int)
  | vec (xs : Array Arg)
  | pair (a b : Arg)
  | not (a : Arg)
  /-- `-a` (`vmad`). -/
  | neg (a : Arg)
  /-- A register with a video byte or half-word selector or mask (`%r1.b0`, `%r2.h10`,
  §9.7.20), as written without its dot. -/
  | sel (a : Arg) (s : String)
  | list (xs : Array Arg)
  | args (xs : Array Arg)
  /-- A call prototype or other name the semantics does not read. -/
  | other (name : String)
  deriving Repr, Inhabited

structure PInstr where
  guard : Option (Bool × RegRef)
  i : Instr
  args : Array Arg
  line : Nat
  deriving Repr, Inhabited

/-- A formal parameter: its declaration and where the callee holds it. -/
structure Formal where
  decl : VarDecl
  /-- A register parameter's register, or `none` for a `.param` one at `frame` offset `off`. -/
  reg : Option RegRef
  off : Nat
  deriving Repr, Inhabited

structure Func where
  name : String
  entry : Bool
  code : Array PInstr
  slots : Nat
  rets : Array Formal
  params : Array Formal
  frameBytes : Nat
  deriving Repr, Inhabited

/-- A module variable with its placement. -/
structure Placed where
  decl : VarDecl
  space : NV.Space
  addr : UInt64
  deriving Repr, Inhabited

structure Prog where
  /-- The target's SM version (`.target sm_80` is 80): it picks the tensor-core datapath. -/
  sm : Nat := 90
  funcs : Array Func
  byName : Std.HashMap String Nat
  vars : Array Placed
  /-- Static shared memory per CTA; dynamic (`extern .shared`) starts here. -/
  sharedBytes : Nat
  /-- Bytes of module `.global` variables, placed from `globalBase`. -/
  globalBase : UInt64
  globalBytes : Nat
  constBytes : Nat
  /-- Local memory per thread: every function's `.local` variables, side by side. -/
  localBytes : Nat
  deriving Inhabited

/-- Bytes of local memory above the `.local` variables for the stack that `alloca` grows down
from: the CUDA runtime's default per-thread stack (`cudaLimitStackSize`, 1 KB). -/
def stackBytes : Nat := 1024

/-- A thread's local memory: its `.local` variables, then the stack. -/
def Prog.localTop (p : Prog) : Nat := p.localBytes + stackBytes

/-! ## Preparing a module -/

namespace Prep

def alignUp (n a : Nat) : Nat := if a ≤ 1 then n else (n + a - 1) / a * a

def declAlign (d : VarDecl) : Nat :=
  d.align.getD (d.ty.bytes * (d.vec.map Vec.length).getD 1)

/-- Register slots a declaration takes: two per `.b128`, `n` per `.vN`. -/
def slotsOf (d : VarDecl) : Nat :=
  (if d.ty.bits > 64 then 2 else 1) * (d.vec.map Vec.length).getD 1

structure Names where
  names : Std.HashMap String Arg := {}
  /-- `%r<100>` declarations: prefix, count, first slot, type. -/
  counted : List (String × Nat × Nat × VarDecl) := []

structure S where
  scopes : List Names
  slots : Nat := 0
  frameBytes : Nat := 0
  localBytes : Nat
  /-- Static shared memory so far: module variables, then function-scope ones. -/
  sharedBytes : Nat
  code : Array PInstr := #[]
  labels : Std.HashMap String Nat := {}

abbrev M := StateT S (Except String)

def lookupCounted (sc : Names) (n : String) : Option Arg :=
  sc.counted.findSome? fun (p, k, first, d) =>
    if n.startsWith p then
      let rest := String.ofList (n.toList.drop p.length)
      match (if !rest.isEmpty && rest.all Char.isDigit then rest.toNat? else none) with
      | some i => if i < k then some (.reg ⟨first + i * slotsOf d, d.ty⟩) else none
      | none => none
    else none

def lookup (n : String) : M (Option Arg) := do
  return (← get).scopes.findSome? fun sc => sc.names.get? n <|> lookupCounted sc n

def bind (n : String) (a : Arg) : M Unit := modify fun s =>
  match s.scopes with
  | sc :: rest => { s with scopes := { sc with names := sc.names.insert n a } :: rest }
  | [] => s

/-- Declare a body variable: a register, a call-sequence `.param`, or a `.local`/`.shared`
variable (placed at `localBytes`, or rejected for `.shared` inside functions). -/
def declare (d : VarDecl) : M Unit := do
  match d.space with
  | .reg =>
    let first := (← get).slots
    match d.count with
    | some k =>
      modify fun s => { s with
        slots := s.slots + k * slotsOf d
        scopes := match s.scopes with
          | sc :: rest => { sc with counted := (d.name, k, first, d) :: sc.counted } :: rest
          | [] => [] }
    | none =>
      modify fun s => { s with slots := s.slots + slotsOf d }
      bind d.name (.reg ⟨first, d.ty⟩)
  | .param | .paramFunc =>
    let off := alignUp (← get).frameBytes (declAlign d)
    modify fun s => { s with frameBytes := off + d.bytes }
    bind d.name (.sym (.frame off) 0)
  | .local =>
    let off := alignUp (← get).localBytes (declAlign d)
    modify fun s => { s with localBytes := off + max d.bytes 1 }
    bind d.name (.sym (.mem .local off.toUInt64) 0)
  | .shared | .sharedCta =>
    let off := alignUp (← get).sharedBytes (declAlign d)
    modify fun s => { s with sharedBytes := off + d.bytes }
    bind d.name (.sym (.mem .shared off.toUInt64) 0)
  | s => throw s!"a function-scope .{Keyword.name s} variable ({d.name}) is not supported"

/-- Resolve an operand. Names are registers, variables, functions or labels. -/
def resolve (funcs : Std.HashMap String Nat) (labels : Std.HashMap String Nat) :
    Nat → Operand → M Arg
  | 0, _ => throw "operand nested too deeply"
  | fuel + 1, o => do
    let name (n : String) : M Arg := do
      if let some a ← lookup n then return a
      if let some f := funcs.get? n then return .func f
      if let some pc := labels.get? n then return .label pc
      if let some s := SReg.lookup? n none then return .sreg s
      return .other n
    match o with
    | .sink => return .sink
    | .expr (.name n) => name n
    | .expr (.un .neg (.name n)) =>
      match ← name n with
      | .reg r => return .neg (.reg r)
      | _ => return .imm (.un .neg (.name n))
    | .expr (.bin .add (.name n) k) =>
      match ← name n, Parse.evalInt k with
      | .sym l off, some c => return .sym l (off + c)
      | _, _ => return .imm (.bin .add (.name n) k)
    | .expr e => return .imm e
    | .comp b c =>
      match SReg.lookup? b (some c) with
      | some s => return .sreg s
      | none =>
        match ← lookup b, SReg.component? c with
        | some (.reg r), some k =>
          return .reg ⟨r.slot + k.val * (if r.ty.bits > 64 then 2 else 1), r.ty⟩
        | some (.reg r), none => return .sel (.reg r) c
        | _, _ => throw s!"{b}.{c} is not a vector register or special register"
    | .not o => return .not (← resolve funcs labels fuel o)
    | .neg o => return .neg (← resolve funcs labels fuel o)
    | .pair a b => return .pair (← resolve funcs labels fuel a) (← resolve funcs labels fuel b)
    | .vec xs => return .vec (← xs.toArray.mapM (resolve funcs labels fuel))
    | .list xs => return .list (← xs.toArray.mapM (resolve funcs labels fuel))
    | .args xs => return .args (← xs.toArray.mapM (resolve funcs labels fuel))
    | .addr none off => return .addr .imm off
    | .addr (some n) off =>
      match ← name n with
      | .reg r => return .addr (.reg r) off
      | .sym l o => return .addr (.loc l) (o + off)
      | _ => throw s!"[{n}]: not a register or variable"

/-- The instructions of a block, nested blocks' included. -/
def instrCount : List (Stmt Instr) → Nat
  | [] => 0
  | .instr .. :: ss => instrCount ss + 1
  | .block b :: ss => instrCount b + instrCount ss
  | _ :: ss => instrCount ss

/-- The labels a block itself defines (not its nested blocks'), each with the index of the
instruction it names when the block's first instruction is at `pc`. -/
def ownLabels : List (Stmt Instr) → Nat → List (String × Nat)
  | [], _ => []
  | .label n :: ss, pc => (n, pc) :: ownLabels ss pc
  | .instr .. :: ss, pc => ownLabels ss (pc + 1)
  | .block b :: ss, pc => ownLabels ss (pc + instrCount b)
  | _ :: ss, pc => ownLabels ss pc

/-- A block's statements. A label is visible in the block that defines it and the blocks nested
in it, where it shadows a label of the same name outside: braces scope what is declared in them
(§9.7.14.1), so that every copy of an inline-assembly
`{ … LAB_WAIT: … DONE: }` sequence branches to its own labels (`tests/sched/labels.ptx`). -/
def body (funcs : Std.HashMap String Nat) (labels : Std.HashMap String Nat) :
    Nat → List (Stmt Instr) → M Unit
  | 0, _ => throw "blocks nested too deeply"
  | fuel + 1, ss => do
    let labels := (ownLabels ss (← get).code.size).foldl (fun m (n, pc) => m.insert n pc) labels
    modify fun s => { s with scopes := {} :: s.scopes }
    for st in ss do
      match st with
      | .decl d => declare d
      | .targets n false ls =>
        bind n (.args (ls.toArray.map fun l => match labels.get? l with
          | some pc => .label pc
          | none => .other l))
      | .block b => body funcs labels fuel b
      | .instr g i line =>
        let guard ← match g with
          | none => pure none
          | some g =>
            match ← lookup g.pred with
            | some (.reg r) => pure (some (g.neg, r))
            | _ => throw s!"line {line}: guard {g.pred} is not a register"
        let args ← i.args.mapM fun o => do
          try resolve funcs labels 64 o catch msg => throw s!"line {line}: {msg}"
        modify fun s => { s with code := s.code.push { guard, i, args, line } }
      | _ => pure ()
    modify fun s => { s with scopes := s.scopes.drop 1 }

/-- Prepare one function; `module` binds the module's variables. -/
def func (funcs : Std.HashMap String Nat) (module : Names) (localBytes sharedBytes : Nat)
    (f : Function Instr) : Except String (Func × Nat × Nat) := do
  let b := f.body.getD []
  let formals (ds : List VarDecl) : M (Array Formal) := ds.toArray.mapM fun d => do
    if d.space == .reg then
      let first := (← get).slots
      modify fun s => { s with slots := s.slots + slotsOf d }
      bind d.name (.reg ⟨first, d.ty⟩)
      return { decl := d, reg := some ⟨first, d.ty⟩, off := 0 }
    let off := alignUp (← get).frameBytes (declAlign d)
    modify fun s => { s with frameBytes := off + d.bytes }
    bind d.name (.sym (.frame off) 0)
    return { decl := d, reg := none, off }
  let go : M (Array Formal × Array Formal) := do
    let rets ← formals f.rets
    let params ← if f.entry then pure #[] else formals f.params
    body funcs {} 64 b
    return (rets, params)
  let ((rets, params), s) ← go.run { scopes := [{}, module], localBytes, sharedBytes }
  return ({ name := f.name, entry := f.entry, code := s.code, slots := s.slots, rets, params,
            frameBytes := s.frameBytes }, s.localBytes, s.sharedBytes)

end Prep

/-- Function-scope `.shared` declarations, in order. -/
def sharedDecls : List (Stmt Instr) → List VarDecl
  | [] => []
  | .decl d :: ss => (if d.space matches .shared | .sharedCta then [d] else []) ++ sharedDecls ss
  | .block b :: ss => sharedDecls b ++ sharedDecls ss
  | _ :: ss => sharedDecls ss

/-- Kernel parameters in the parameter space: each at its alignment (its `.align`, else its
element size), in order (§5.1.6.1). -/
def paramLayout (ps : List VarDecl) : List (VarDecl × Nat) × Nat :=
  ps.foldl (fun (acc, off) d =>
    let o := Prep.alignUp off (Prep.declAlign d)
    (acc ++ [(d, o)], o + d.bytes)) ([], 0)

/-- Prepare a module for execution, placing module `.global` variables from `globalBase`. -/
def Prog.ofModule (m : Module Instr) (globalBase : UInt64 := 0x1000) : Except String Prog := do
  let fs := m.items.filterMap fun | .func f => some f | _ => none
  -- One entry per function name: a definition replaces its declaration.
  let names := fs.foldl (fun (acc : Array String) f =>
    if acc.contains f.name then acc else acc.push f.name) #[]
  let defn (n : String) : Function Instr :=
    (fs.find? (fun f => f.name == n && f.body.isSome)).getD ((fs.find? (·.name == n)).get!)
  let byName := Std.HashMap.ofList (names.toList.zipIdx)
  -- Module variables.
  let mut module : Prep.Names := {}
  let mut vars := #[]
  let mut shared := 0
  let mut global := 0
  let mut const := 0
  let mut externShared : List VarDecl := []
  for it in m.items do
    let .var d := it | continue
    match d.space with
    | .shared | .sharedCta =>
      if d.linkage == some .extern && d.dims == [none] then externShared := d :: externShared
      else
        let o := Prep.alignUp shared (Prep.declAlign d)
        shared := o + d.bytes
        vars := vars.push { decl := d, space := .shared, addr := o.toUInt64 }
        module := { module with names := module.names.insert d.name (.sym (.mem .shared o.toUInt64) 0) }
    | .global =>
      let o := Prep.alignUp global (Prep.declAlign d)
      global := o + d.bytes
      let a := globalBase + o.toUInt64
      vars := vars.push { decl := d, space := .global, addr := a }
      module := { module with names := module.names.insert d.name (.sym (.mem .global a) 0) }
    | .const =>
      let o := Prep.alignUp const (Prep.declAlign d)
      const := o + d.bytes
      vars := vars.push { decl := d, space := .const 0, addr := o.toUInt64 }
      module := { module with names := module.names.insert d.name (.sym (.mem (.const 0) o.toUInt64) 0) }
    | _ => pure ()
  -- Dynamic shared memory starts after the static (module, then function-scope variables, as
  -- `Prep.declare` places them), at the largest alignment asked for.
  let static := names.foldl (fun n f => sharedDecls ((defn f).body.getD []) |>.foldl
    (fun n d => Prep.alignUp n (Prep.declAlign d) + d.bytes) n) shared
  let dynAlign := externShared.foldl (fun a d => max a (Prep.declAlign d)) 16
  let dynBase := Prep.alignUp static dynAlign
  for d in externShared do
    module := { module with names := module.names.insert d.name (.sym (.mem .shared dynBase.toUInt64) 0) }
  -- Functions; an entry's parameters are in the parameter space.
  let mut funcs := #[]
  let mut localBytes := 0
  for n in names do
    let f := defn n
    let mut scope := module
    if f.entry then
      for (d, o) in (paramLayout f.params).1 do
        scope := { scope with names := scope.names.insert d.name (.sym (.mem .param o.toUInt64) 0) }
    let (pf, lb, sb) ← (Prep.func byName scope localBytes shared f).mapError (s!"{f.name}: " ++ ·)
    funcs := funcs.push pf
    localBytes := lb
    shared := sb
  let sm := ((m.target.head?.map fun t => (t.drop 3).takeWhile Char.isDigit).bind
    (·.toString.toNat?)).getD 90
  return { sm, funcs, byName, vars, sharedBytes := dynBase, globalBase, globalBytes := global,
           constBytes := const, localBytes }

/-! ## Threads -/

/-- Where a thread runs. -/
structure Ctx where
  tid : Nat × Nat × Nat
  ntid : Nat × Nat × Nat
  ctaid : Nat × Nat × Nat
  nctaid : Nat × Nat × Nat
  clusterCtaid : Nat × Nat × Nat := (0, 0, 0)
  clusterNctaid : Nat × Nat × Nat := (1, 1, 1)
  clusterid : Nat × Nat × Nat := (0, 0, 0)
  nclusterid : Nat × Nat × Nat := (1, 1, 1)
  smid : Nat := 0
  dynShared : Nat := 0
  deriving Repr, Inhabited

namespace Ctx

def flat (v n : Nat × Nat × Nat) : Nat := v.1 + n.1 * (v.2.1 + n.2.1 * v.2.2)

/-- The thread's index in its CTA, x fastest (CUDA Programming Guide, thread hierarchy). -/
def linear (c : Ctx) : Nat := flat c.tid c.ntid
def laneid (c : Ctx) : Nat := c.linear % 32
def warpid (c : Ctx) : Nat := c.linear / 32
def clusterRank (c : Ctx) : Nat := flat c.clusterCtaid c.clusterNctaid

end Ctx

structure Env where
  prog : Prog
  windows : Windows := {}
  /-- `%clock64` and `%globaltimer` at this step (the interpreter's step count). -/
  clock : Nat := 0
  /-- The launch's texture and surface objects (`NV.Tex`). -/
  textures : Array NV.Tex.Texture := #[]
  /-- `tcgen05.mma` writes zeros in place of its products: for schedule searches over kernels whose
  control flow does not depend on them (`Machine.runRandom`). -/
  skipMma : Bool := false
  deriving Inhabited

structure Frame where
  fn : Nat
  pc : Nat
  regs : Array UInt64
  params : ByteArray
  /-- The caller's operands that receive this call's return values. -/
  retTo : Array Arg := #[]
  /-- The carry flag `CC.CF` of the extended-precision instructions (§9.7.2). -/
  carry : Bool := false
  /-- Progress through an instruction that takes several effects (`ldmatrix`, `stmatrix`): the
  next sub-step, 0 when none is under way, and what earlier sub-steps gathered. -/
  micro : Nat := 0
  scratch : Array UInt64 := #[]
  /-- The stack pointer (§9.7.19): a local address; `alloca` lowers it, returning frees what the
  function allocated (a callee starts from its caller's). -/
  sp : UInt64 := 0
  deriving Inhabited

structure Thread where
  ctx : Ctx
  frames : Array Frame
  /-- Device `printf` records, in order: the format string and the argument bytes
  (`NV.Printf`). Host output, not memory. -/
  printf : Array (String × ByteArray) := #[]
  deriving Inhabited

namespace Thread

def start (p : Prog) (ctx : Ctx) (entry : Nat) : Thread :=
  let f := p.funcs[entry]!
  { ctx, frames := #[{ fn := entry, pc := 0, regs := Array.replicate f.slots 0,
                       params := ByteArray.mk (Array.replicate f.frameBytes 0),
                       sp := p.localTop.toUInt64 }] }

def frame (t : Thread) : Frame := t.frames.back!

def setFrame (t : Thread) (f : Frame) : Thread :=
  { t with frames := t.frames.set! (t.frames.size - 1) f }

def done (t : Thread) : Bool := t.frames.isEmpty

/-- The instruction the thread executes next. -/
def current? (e : Env) (t : Thread) : Option PInstr := do
  let f ← t.frames.back?
  (← e.prog.funcs[f.fn]?).code[f.pc]?

def advance (t : Thread) : Thread := t.setFrame { t.frame with pc := t.frame.pc + 1 }

def jump (t : Thread) (pc : Nat) : Thread := t.setFrame { t.frame with pc }

end Thread

/-! ## Values -/

def mask (w : Nat) (x : UInt64) : UInt64 :=
  if w ≥ 64 then x else x &&& ((1 <<< w.toUInt64) - 1)

/-- Sign-extend the low `w` bits. -/
def sext (w : Nat) (x : UInt64) : UInt64 :=
  if w ≥ 64 || w == 0 then x else
  let s := (64 - w).toUInt64
  ((x <<< s).toInt64 >>> s.toInt64).toUInt64

/-- The low `w` bits as an integer, signed or not. -/
def toInt (signed : Bool) (w : Nat) (x : UInt64) : Int :=
  if signed then (sext w x).toInt64.toInt else (mask w x).toNat

/-- An integer modulo `2^w`. -/
def ofInt (w : Nat) (i : Int) : UInt64 := mask w (UInt64.ofInt i)

/-- Extend a `t`-typed result to a register of type `r` (Table 28). -/
def extendTo (t r : Ty) (v : UInt64) : UInt64 :=
  let v := mask t.bits v
  if r.bits ≤ t.bits then mask r.bits v
  else if t.isSigned then mask r.bits (sext t.bits v) else v

end NV.PTX.Sem
