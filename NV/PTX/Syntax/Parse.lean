import NV.PTX.Syntax.Ast
import NV.PTX.Syntax.Lex

/-!
# PTX parser

Recursive descent over the tokens of `NV.PTX.Syntax.Lex`, producing `Module RawInstr`. The
grammar is PTX ISA 9.4 chapter 4 (statements, constant expressions with C precedence,
Table 4), §5.4 (variable declarations and initializers), §6 (operands) and chapter 11
(directives). Instruction qualifiers are kept as written; `decode` interprets them.

Every loop is bounded by the number of tokens and every recursion by a fuel argument, so the
parser is total; running out of fuel is reported as nesting too deep.
-/

namespace NV.PTX
namespace Parse

structure Ctx where
  toks : Array Token

abbrev P := ReaderT Ctx (StateT Nat (Except String))

def peekN (k : Nat) : P (Option Tok) := do
  return (← read).toks[(← get) + k]?.map (·.tok)

def peek : P (Option Tok) := peekN 0

def line : P Nat := do
  let c ← read
  let i ← get
  return (c.toks[i]? <|> c.toks.back?).map (·.line) |>.getD 0

def fail {α : Type} (msg : String) : P α := do throw s!"line {← line}: {msg}"

def bump : P Unit := modify (· + 1)

def next : P Tok := do
  match ← peek with
  | some t => bump; return t
  | none => fail "unexpected end of input"

def isSym (s : String) : P Bool := return (← peek) == some (.sym s)
def isDot (s : String) : P Bool := return (← peek) == some (.dot s)

def sym? (s : String) : P Bool := do
  if ← isSym s then bump; return true else return false

def dot? (s : String) : P Bool := do
  if ← isDot s then bump; return true else return false

def describe : Option Tok → String
  | some (.ident s) => s!"'{s}'"
  | some (.dot s) => s!"'.{s}'"
  | some (.sym s) => s!"'{s}'"
  | some (.str s) => s!"string \"{s}\""
  | some _ => "a number"
  | none => "end of input"

def expectSym (s : String) : P Unit := do
  unless ← sym? s do fail s!"expected '{s}', found {describe (← peek)}"

def ident : P String := do
  match ← peek with
  | some (.ident s) => bump; return s
  | t => fail s!"expected an identifier, found {describe t}"

def nat : P Nat := do
  match ← peek with
  | some (.int n _) => bump; return n
  | t => fail s!"expected an integer, found {describe t}"

def str : P String := do
  match ← peek with
  | some (.str s) => bump; return s
  | t => fail s!"expected a string, found {describe t}"

/-- Repeat `p` separated by commas, at least once. -/
def sepByComma {α : Type} (p : P α) : P (List α) := do
  let mut acc := #[← p]
  for _ in [0:(← read).toks.size] do
    if ← sym? "," then acc := acc.push (← p) else break
  return acc.toList

/-! ## Constant expressions (§4.5.4, Table 4) -/

def binLevels : List (List (String × BinOp)) :=
  [[("||", .lor)], [("&&", .land)], [("|", .bor)], [("^", .bxor)], [("&", .band)],
   [("==", .eq), ("!=", .ne)], [("<", .lt), (">", .gt), ("<=", .le), (">=", .ge)],
   [("<<", .shl), (">>", .shr)], [("+", .add), ("-", .sub)],
   [("*", .mul), ("/", .div), ("%", .rem)]]

def binOpAt (ops : List (String × BinOp)) : P (Option BinOp) := do
  match ← peek with
  | some (.sym s) => return ops.lookup s
  | _ => return none

mutual
/-- `cond ? a : b`, right associative, then the binary levels of Table 4, loosest first. Each
call spends one unit of fuel; `exprFuel` allows parentheses nested about twenty deep. -/
def expr : Nat → P Expr
  | 0 => fail "expression nested too deeply"
  | fuel + 1 => do
    let c ← binary binLevels fuel
    if ← sym? "?" then
      let a ← expr fuel
      expectSym ":"
      return .cond c a (← expr fuel)
    return c

def binary : List (List (String × BinOp)) → Nat → P Expr
  | _, 0 => fail "expression nested too deeply"
  | [], fuel + 1 => unary fuel
  | ops :: rest, fuel + 1 => do
    let mut e ← binary rest fuel
    for _ in [0:(← read).toks.size] do
      match ← binOpAt ops with
      | some op => bump; e := .bin op e (← binary rest fuel)
      | none => break
    return e

def unary : Nat → P Expr
  | 0 => fail "expression nested too deeply"
  | fuel + 1 => do
    match ← peek with
    | some (.sym "-") => bump; return .un .neg (← unary fuel)
    | some (.sym "+") => bump; return .un .plus (← unary fuel)
    | some (.sym "!") => bump; return .un .not (← unary fuel)
    | some (.sym "~") => bump; return .un .compl (← unary fuel)
    | some (.sym "(") =>
      match ← peekN 1, ← peekN 2 with
      | some (.dot "s64"), some (.sym ")") => bump; bump; bump; return .un .castS64 (← unary fuel)
      | some (.dot "u64"), some (.sym ")") => bump; bump; bump; return .un .castU64 (← unary fuel)
      | _, _ =>
        bump
        let e ← expr fuel
        expectSym ")"
        return e
    | _ => primary fuel

def primary : Nat → P Expr
  | 0 => fail "expression nested too deeply"
  | fuel + 1 => do
    match ← next with
    | .int n u =>
      -- `0xFF(e)`: the `mask()` operator of initializers (§5.4.4).
      if ← sym? "(" then
        let e ← expr fuel
        expectSym ")"
        return .app s!"mask{n}" e
      return .int n u
    | .dec m e => return .dec m e
    | .f32 b => return .f32 b
    | .f64 b => return .f64 b
    | .ident s =>
      if s == "generic" && (← isSym "(") then
        bump
        let e ← expr fuel
        expectSym ")"
        return .app "generic" e
      return .name s
    | t => do
      modify (· - 1)
      fail s!"expected an expression, found {describe (some t)}"
end

def exprFuel : Nat := 256

def constExpr : P Expr := expr exprFuel

/-- Integer value of a constant expression without symbols, for addresses, dimensions and
directive arguments. 64-bit wrapping is not modelled here; these are small numbers. -/
def evalInt : Expr → Option Int
  | .int n _ => some n
  | .un .neg a => (evalInt a).map (- ·)
  | .un .plus a => evalInt a
  | .bin .add a b => do return (← evalInt a) + (← evalInt b)
  | .bin .sub a b => do return (← evalInt a) - (← evalInt b)
  | .bin .mul a b => do return (← evalInt a) * (← evalInt b)
  | .bin .div a b => do return (← evalInt a) / (← evalInt b)
  | .bin .shl a b => do return (← evalInt a) * 2 ^ (← evalInt b).toNat
  | _ => none

/-- An address expression `name`, `name ± const` or `const` as base and byte offset. -/
def toAddr : Expr → Option (Option String × Int)
  | .name s => some (some s, 0)
  | .bin .add (.name s) k => (evalInt k).map (some s, ·)
  | .bin .sub (.name s) k => (evalInt k).map (some s, - ·)
  | e => (evalInt e).map (none, ·)

/-! ## Operands (§6) -/

/-- Vector components (`.x`, …, `.r`, …) and the byte and half-word selectors of the video
instructions (`.b0`, `.h1`, `.b3210`, §9.7.19). -/
def isComponent (c : String) : Bool :=
  ["x", "y", "z", "w", "r", "g", "b", "a"].contains c ||
  ((c.startsWith "b" || c.startsWith "h") && c.length > 1 && (c.drop 1).all Char.isDigit)

mutual
def operand (call : Bool) : Nat → P Operand
  | 0 => fail "operand nested too deeply"
  | fuel + 1 => do
    match ← peek, ← peekN 1 with
    | some (.ident "_"), _ => bump; return .sink
    | some (.sym "!"), _ => bump; return .not (← operand call fuel)
    | some (.sym "-"), some (.ident _) =>
      if (← peekN 2) matches some (.dot _) then bump; return .neg (← operand call fuel)
      return .expr (← constExpr)
    | some (.sym "{"), _ => bump; return .vec (← operandList "}" fuel)
    | some (.sym "["), _ =>
      bump
      let xs ← operandList "]" fuel
      match xs with
      | [.expr e] =>
        match toAddr e with
        | some (b, o) => return .addr b o
        | none => fail "malformed address"
      | [.comp b c] => fail s!"malformed address {b}.{c}"
      | xs => return .list xs
    | some (.sym "("), _ =>
      if call then bump; return .args (← operandList ")" fuel)
      return .expr (← constExpr)
    | some (.ident s), some (.dot c) =>
      if isComponent c then bump; bump; return .comp s c
      fail s!"unexpected '.{c}' after operand {s}"
    | some (.ident s), some (.sym "|") =>
      bump; bump
      return .pair (.expr (.name s)) (← operand call fuel)
    | _, _ => return .expr (← constExpr)

/-- Operands separated by commas up to the closing `close` (consumed). -/
def operandList (close : String) : Nat → P (List Operand)
  | 0 => fail "operand nested too deeply"
  | fuel + 1 => do
    if ← sym? close then return []
    let mut acc := #[← operand false fuel]
    for _ in [0:(← read).toks.size] do
      if ← sym? "," then acc := acc.push (← operand false fuel) else break
    expectSym close
    return acc.toList
end

/-! ## Declarations (§5.4) -/

def space? : P (Option Space) := do
  match ← peek with
  | some (.dot s) =>
    match Keyword.parse? (α := Space) s with
    | some sp => bump; return some sp
    | none => return none
  | _ => return none

def linkage? : P (Option Linkage) := do
  match ← peek with
  | some (.dot s) =>
    match Keyword.parse? (α := Linkage) s with
    | some l => bump; return some l
    | none => return none
  | _ => return none

/-- `.attribute(.managed)`, `.attribute(.unified(a, b))`: the contents as written. -/
def attributes : P (List String) := do
  expectSym "("
  let mut acc := #[]
  let mut depth := 1
  for _ in [0:(← read).toks.size] do
    match ← next with
    | .sym "(" => depth := depth + 1; acc := acc.push "("
    | .sym ")" =>
      depth := depth - 1
      if depth == 0 then break
      acc := acc.push ")"
    | .dot s => acc := acc.push s
    | .int n _ => acc := acc.push (toString n)
    | .sym s => acc := acc.push s
    | .ident s => acc := acc.push s
    | _ => fail "malformed .attribute"
  return acc.toList

def initializer : Nat → P Init
  | 0 => fail "initializer nested too deeply"
  | fuel + 1 => do
    if ← sym? "{" then
      if ← sym? "}" then return .list []
      let xs ← sepByComma (initializer fuel)
      expectSym "}"
      return .list xs
    return .expr (← constExpr)

/-- Type and attributes after the state space, in any order: `.attribute(…)`, `.align n`,
`.v2`/`.v4`/`.v8`, `.ptr {.space} {.align n}` (kernel parameters), then the type. -/
def declHead (d : VarDecl) : P VarDecl := do
  let mut d := d
  for _ in [0:(← read).toks.size] do
    match ← peek with
    | some (.dot "attribute") => bump; d := { d with attrs := d.attrs ++ (← attributes) }
    | some (.dot "align") => bump; d := { d with align := some (← nat) }
    | some (.dot "ptr") =>
      bump
      let sp ← space?
      let al ← if ← dot? "align" then some <$> nat else pure none
      d := { d with ptr := some (sp, al) }
    | some (.dot s) =>
      if let some v := Keyword.parse? (α := Vec) s then bump; d := { d with vec := some v }
      else if let some t := Keyword.parse? (α := Ty) s then
        bump
        d := { d with ty := t }
        -- `.ptr` may also follow the type (`.param .u64 .ptr .global .align 16 p`).
        if ← isDot "ptr" then continue
        return d
      else fail s!"unexpected '.{s}' in a declaration"
    | t =>
      if d.ptr.isSome then return d
      fail s!"expected a type, found {describe t}"
  return d

/-- A name with its `<n>` count, `[n]` dimensions and `= init`. -/
def declName (d : VarDecl) : P VarDecl := do
  let name ← match ← peek with
    | some (.ident s) => bump; pure s
    | t => fail s!"expected a variable name, found {describe t}"
  let mut d := { d with name }
  if ← sym? "<" then
    d := { d with count := some (← nat) }
    expectSym ">"
  let mut dims := #[]
  for _ in [0:(← read).toks.size] do
    unless ← sym? "[" do break
    if ← sym? "]" then dims := dims.push none; continue
    match evalInt (← constExpr) with
    | some n => dims := dims.push (some n.toNat)
    | none => fail "array dimension is not a constant"
    expectSym "]"
  d := { d with dims := dims.toList }
  if ← sym? "=" then d := { d with init := some (← initializer exprFuel) }
  return d

/-- A declaration statement after its linkage and space, up to and including `;`. -/
def declRest (base : VarDecl) : P (List VarDecl) := do
  let head ← declHead base
  let ds ← sepByComma (declName head)
  expectSym ";"
  return ds

/-- One parameter: `.param …` or `.reg …` with one name. -/
def param : P VarDecl := do
  match ← space? with
  | some sp => declName (← declHead { space := sp, ty := .b8, name := "" })
  | none => fail s!"expected a parameter, found {describe (← peek)}"

def paramList : P (List VarDecl) := do
  expectSym "("
  if ← sym? ")" then return []
  let ps ← sepByComma param
  expectSym ")"
  return ps

/-! ## Statements -/

/-- `.loc file line col`, optionally `, function_name f, inlined_at file line col`; no `;`. -/
def locRest : P (Stmt RawInstr) := do
  let f ← nat
  let l ← nat
  let c ← nat
  if ← sym? "," then
    let _ ← ident
    let _ ← ident
    expectSym ","
    let _ ← ident
    let _ ← nat
    let _ ← nat
    let _ ← nat
  return .loc f l c

def pragmaRest : P (List String) := do
  let ss ← sepByComma str
  expectSym ";"
  return ss

def instr (guard : Option Guard) : P (Stmt RawInstr) := do
  let ln ← line
  let op ← ident
  let mut mods := #[]
  for _ in [0:(← read).toks.size] do
    match ← peek with
    | some (.dot m) => bump; mods := mods.push m
    | _ => break
  let args ← if ← isSym ";" then pure [] else sepByComma (operand (op == "call") exprFuel)
  expectSym ";"
  return .instr guard ⟨op, mods, args.toArray⟩ ln

/-- `.callprototype (rets)? _ (params)? .noreturn? …;` after `name:`. -/
def protoRest : P Proto := do
  let rets ← if ← isSym "(" then paramList else pure []
  let u ← ident
  unless u == "_" do fail "expected '_' in .callprototype"
  let params ← if ← isSym "(" then paramList else pure []
  let mut noreturn := false
  for _ in [0:(← read).toks.size] do
    if ← dot? "noreturn" then noreturn := true
    else if (← dot? "abi_preserve") || (← dot? "abi_preserve_control") then let _ ← nat
    else break
  expectSym ";"
  return { rets, params, noreturn }

def stmts : Nat → P (List (Stmt RawInstr))
  | 0 => fail "blocks nested too deeply"
  | fuel + 1 => do
    let mut acc : Array (Stmt RawInstr) := #[]
    for _ in [0:(← read).toks.size] do
      match ← peek, ← peekN 1 with
      | some (.sym "}"), _ => bump; return acc.toList
      | some (.sym "{"), _ => bump; acc := acc.push (.block (← stmts fuel))
      | some (.sym "@"), _ =>
        bump
        let neg ← sym? "!"
        let p ← ident
        acc := acc.push (← instr (some ⟨neg, p⟩))
      | some (.ident name), some (.sym ":") =>
        bump; bump
        if ← dot? "callprototype" then acc := acc.push (.proto name (← protoRest))
        else if ← dot? "branchtargets" then
          let ls ← sepByComma ident
          expectSym ";"
          acc := acc.push (.targets name false ls)
        else if ← dot? "calltargets" then
          let ls ← sepByComma ident
          expectSym ";"
          acc := acc.push (.targets name true ls)
        else acc := acc.push (.label name)
      | some (.ident _), _ => acc := acc.push (← instr none)
      | some (.dot "loc"), _ => bump; acc := acc.push (← locRest)
      | some (.dot "pragma"), _ => bump; acc := acc.push (.pragma (← pragmaRest))
      | some (.dot _), _ =>
        let link ← linkage?
        match ← space? with
        | some sp =>
          for d in ← declRest { linkage := link, space := sp, ty := .b8, name := "" } do
            acc := acc.push (.decl d)
        | none => fail s!"unexpected {describe (← peek)} in a function body"
      | t, _ => fail s!"unexpected {describe t} in a function body"
    fail "function body too long"

/-! ## Functions and the module (chapter 11) -/

def dims : P (List Nat) := sepByComma nat

def funcDirective? : P (Option FuncDirective) := do
  match ← peek with
  | some (.dot d) =>
    let r ← match d with
      | "maxntid" => bump; pure (some (.maxntid (← dims)))
      | "reqntid" => bump; pure (some (.reqntid (← dims)))
      | "minnctapersm" => bump; pure (some (.minnctapersm (← nat)))
      | "maxnctapersm" => bump; pure (some (.maxnctapersm (← nat)))
      | "maxnreg" => bump; pure (some (.maxnreg (← nat)))
      | "reqnctapercluster" => bump; pure (some (.reqnctapercluster (← dims)))
      | "explicitcluster" => bump; pure (some .explicitcluster)
      | "maxclusterrank" => bump; pure (some (.maxclusterrank (← nat)))
      | "blocksareclusters" => bump; pure (some .blocksareclusters)
      | "noreturn" => bump; pure (some .noreturn)
      | "abi_preserve" => bump; pure (some (.abiPreserve (← nat)))
      | "abi_preserve_control" => bump; pure (some (.abiPreserveControl (← nat)))
      | "minperctamemory" => bump; pure (some (.minperctamemory (← nat)))
      | "pragma" => bump; pure (some (.pragma (← pragmaRest)))
      | _ => pure none
    return r
  | _ => return none

def function (linkage : Option Linkage) (entry : Bool) : P (Function RawInstr) := do
  let attrs ← if ← dot? "attribute" then attributes else pure []
  let rets ← if !entry && (← isSym "(") then paramList else pure []
  let name ← ident
  let params ← if ← isSym "(" then paramList else pure []
  let mut directives := #[]
  for _ in [0:(← read).toks.size] do
    match ← funcDirective? with
    | some d => directives := directives.push d
    | none => break
  let body ← if ← sym? ";" then pure none
    else do expectSym "{"; pure (some (← stmts 64))
  return { entry, linkage, attrs, name, rets, params, directives := directives.toList, body }

/-- `.section name { … }`: DWARF data, skipped. -/
def skipSection : P Unit := do
  let _ ← next
  expectSym "{"
  let mut depth := 1
  for _ in [0:(← read).toks.size] do
    match ← next with
    | .sym "{" => depth := depth + 1
    | .sym "}" =>
      depth := depth - 1
      if depth == 0 then return
    | _ => pure ()

/-- Tokens up to the end of the current line (`@@dwarf` lines). -/
def skipRestOfLine : P Unit := do
  let l ← line
  for _ in [0:(← read).toks.size] do
    if (← peek).isNone || (← line) != l then break
    bump

def item? : P (Option (List (Item RawInstr))) := do
  match ← peek with
  | none => return none
  | some (.dot "file") =>
    bump
    let n ← nat
    let s ← str
    if ← sym? "," then let _ ← nat; expectSym ","; let _ ← nat
    return some [.file n s]
  | some (.dot "section") => bump; skipSection; return some []
  | some (.dot "pragma") => bump; return some [.pragma (← pragmaRest)]
  | some (.dot "alias") =>
    bump
    let a ← ident
    expectSym ","
    let b ← ident
    expectSym ";"
    return some [.alias a b]
  | some (.sym "@") => skipRestOfLine; return some []
  | _ =>
    let link ← linkage?
    if ← dot? "entry" then return some [.func (← function link true)]
    if ← dot? "func" then return some [.func (← function link false)]
    match ← space? with
    | some sp => return some ((← declRest { linkage := link, space := sp, ty := .b8, name := "" }).map .var)
    | none => fail s!"unexpected {describe (← peek)} at module scope"

def version : P (Nat × Nat) := do
  match ← next with
  | .dec m e =>
    let k := 10 ^ e.natAbs
    return (m / k, m % k)
  | .int n _ => return (n, 0)
  | _ => fail "malformed .version"

def module : P (Module RawInstr) := do
  unless ← dot? "version" do fail "a module starts with .version"
  let v ← version
  unless ← dot? "target" do fail "expected .target"
  let target ← sepByComma ident
  let addressSize ← if ← dot? "address_size" then nat else pure 64
  let mut items := #[]
  for _ in [0:(← read).toks.size + 1] do
    match ← item? with
    | some is => items := items ++ is.toArray
    | none => return { version := v, target, addressSize, items := items.toList }
  fail "module too long"

end Parse

/-- Parse a PTX module. -/
def parse (src : String) : Except String (Module RawInstr) := do
  let toks ← lex src
  let (m, _) ← (Parse.module.run ⟨toks⟩).run 0
  return m

end NV.PTX
