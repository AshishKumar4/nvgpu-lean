import NV.PTX.Syntax.Type

/-!
# PTX modules

The tree of a PTX module (PTX ISA 9.4 chapters 4, 5 and 11): module directives, variable
declarations, kernels and functions, and their bodies. The tree is generic in the instruction
type `ι`: the parser produces `Module RawInstr` (opcode, qualifiers, operands as written), and
`decode` turns it into `Module Instr`, the typed instructions of `NV.PTX.Syntax.Instr`.
-/

namespace NV.PTX

inductive UnOp where
  | plus | neg | not | compl
  /-- `(.s64)` and `(.u64)` casts (§4.5.4). -/
  | castS64 | castU64
  deriving DecidableEq, Repr, Inhabited

inductive BinOp where
  | mul | div | rem | add | sub | shl | shr | lt | gt | le | ge | eq | ne
  | band | bxor | bor | land | lor
  deriving DecidableEq, Repr, Inhabited

/-- Constant expressions (§4.5.4), and the symbol references operands and initializers use:
a name is a register, variable, function or label, resolved by the checker. -/
inductive Expr where
  /-- An integer literal; `unsigned` per §4.5.1. -/
  | int (n : Nat) (unsigned : Bool)
  /-- A decimal floating literal `mant · 10^exp`, a `.f64` value (§4.5.2). -/
  | dec (mant : Nat) (exp : Int)
  | f32 (bits : UInt32)
  | f64 (bits : UInt64)
  | name (s : String)
  | un (op : UnOp) (a : Expr)
  | bin (op : BinOp) (a b : Expr)
  | cond (c a b : Expr)
  /-- `generic(var)` and `mask(e)` in initializers (§5.4.4); `fn` is `generic` or the mask
  written as a hex literal. -/
  | app (fn : String) (arg : Expr)
  deriving BEq, Repr, Inhabited

/-- Instruction operands (§6). -/
inductive Operand where
  /-- `_`, the sink. -/
  | sink
  /-- A register, symbol, label or constant expression. -/
  | expr (e : Expr)
  /-- A component of a vector variable or special register: `%tid.x`, `%v.y`. -/
  | comp (base : String) (c : String)
  /-- `!p`, a negated predicate source. -/
  | not (o : Operand)
  /-- `-a`, a negated register source with its selector (`vmad`, §9.7.20.1). -/
  | neg (o : Operand)
  /-- `p|q`, the two destinations of `setp` and friends. -/
  | pair (a b : Operand)
  /-- `{a, b, …}`, a vector of registers. -/
  | vec (xs : List Operand)
  /-- `[base+off]`, `[base]` or `[off]`: a memory address (§6.4.1). -/
  | addr (base : Option String) (off : Int)
  /-- `[a, b, …]`: texture and surface coordinates, two or more items. -/
  | list (xs : List Operand)
  /-- `(a, b, …)`: argument and return lists of `call`. -/
  | args (xs : List Operand)
  deriving BEq, Repr, Inhabited

namespace Operand

def name? : Operand → Option String
  | .expr (.name s) => some s
  | _ => none

end Operand

/-- An instruction as written: opcode, its dot-qualifiers in order (`ld.global.v4.u32` is
`⟨"ld", #["global", "v4", "u32"], …⟩`), and operands. -/
structure RawInstr where
  op : String
  mods : Array String
  args : Array Operand
  deriving BEq, Repr, Inhabited

inductive Init where
  | expr (e : Expr)
  | list (xs : List Init)
  deriving BEq, Repr, Inhabited

/-- `.visible`, `.extern`, `.weak`, `.common` (§11.6). -/
inductive Linkage where
  | visible | extern | weak | common
  deriving DecidableEq, Repr, Inhabited

instance : Keyword Linkage where
  all := [.visible, .extern, .weak, .common]
  name | .visible => "visible" | .extern => "extern" | .weak => "weak" | .common => "common"

/-- A variable declaration (§5.4), or one parameter. -/
structure VarDecl where
  linkage : Option Linkage := none
  space : Space
  /-- `.attribute(…)` contents as written, e.g. `managed`. -/
  attrs : List String := []
  align : Option Nat := none
  vec : Option Vec := none
  ty : Ty
  name : String
  /-- Array dimensions; `none` is an unsized dimension (`a[]`). -/
  dims : List (Option Nat) := []
  /-- `%r<100>`: one hundred variables `%r0` … `%r99` (§5.4.6). -/
  count : Option Nat := none
  init : Option Init := none
  /-- Kernel parameter attribute `.ptr {.space} {.align n}` (§5.1.6.3). -/
  ptr : Option (Option Space × Option Nat) := none
  deriving BEq, Repr, Inhabited

namespace VarDecl

/-- Elements of the array or vector, 1 for a scalar; unsized dimensions count 0. -/
def elems (d : VarDecl) : Nat :=
  d.dims.foldl (fun n k => n * k.getD 0) 1 * (d.vec.map Vec.length |>.getD 1)

def bytes (d : VarDecl) : Nat := d.elems * d.ty.bytes

end VarDecl

structure Guard where
  neg : Bool
  pred : String
  deriving DecidableEq, Repr, Inhabited

/-- `.callprototype` (§11.3.3): return and input parameters of an indirect call. -/
structure Proto where
  rets : List VarDecl
  params : List VarDecl
  noreturn : Bool := false
  deriving BEq, Repr, Inhabited

inductive Stmt (ι : Type) where
  | label (name : String)
  | decl (d : VarDecl)
  | instr (guard : Option Guard) (i : ι) (line : Nat)
  /-- `{ … }`: a nested scope (§9.7.14.1). -/
  | block (body : List (Stmt ι))
  | pragma (s : List String)
  /-- `.loc file line column` (§11.5.4). -/
  | loc (file line col : Nat)
  /-- `name: .callprototype …`. -/
  | proto (name : String) (p : Proto)
  /-- `name: .branchtargets …` or `name: .calltargets …`. -/
  | targets (name : String) (call : Bool) (labels : List String)
  deriving BEq, Repr, Inhabited

/-- Directives between a function's header and its body (§11.4, §11.7). -/
inductive FuncDirective where
  | maxntid (dims : List Nat)
  | reqntid (dims : List Nat)
  | minnctapersm (n : Nat)
  | maxnctapersm (n : Nat)
  | maxnreg (n : Nat)
  | reqnctapercluster (dims : List Nat)
  | explicitcluster
  | maxclusterrank (n : Nat)
  | blocksareclusters
  | noreturn
  | abiPreserve (n : Nat)
  | abiPreserveControl (n : Nat)
  | minperctamemory (n : Nat)
  | pragma (s : List String)
  deriving BEq, Repr, Inhabited

structure Function (ι : Type) where
  /-- `.entry` (a kernel) or `.func`. -/
  entry : Bool
  linkage : Option Linkage := none
  attrs : List String := []
  name : String
  rets : List VarDecl := []
  params : List VarDecl := []
  directives : List FuncDirective := []
  /-- `none` for a declaration without a body. -/
  body : Option (List (Stmt ι))
  deriving BEq, Repr, Inhabited

inductive Item (ι : Type) where
  | var (d : VarDecl)
  | func (f : Function ι)
  /-- `.file n "name"` (§11.5.3). -/
  | file (n : Nat) (name : String)
  /-- `.alias alias, aliasee` (§11.2.3). -/
  | alias (a b : String)
  | pragma (s : List String)
  deriving BEq, Repr, Inhabited

structure Module (ι : Type) where
  /-- `.version major.minor`. -/
  version : Nat × Nat
  /-- `.target`: the architecture first (`sm_90a`), then options (`texmode_unified`, `debug`). -/
  target : List String
  addressSize : Nat := 64
  items : List (Item ι)
  deriving BEq, Repr, Inhabited

end NV.PTX
