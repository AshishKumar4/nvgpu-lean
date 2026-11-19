/-!
# PTX lexer

PTX ISA 9.4 §4.1–§4.5: ASCII source, C comments, identifiers `[a-zA-Z]{followsym}*` or
`[_$%]{followsym}+`, C integer literals with an optional `U`, decimal floating literals, and the
exact-bits forms `0f…` (8 hex digits, `.f32`) and `0d…` (16 hex digits, `.f64`).

A keyword or qualifier after a dot is one token, its `::` parts included: `.shared::cluster`,
`.L2::cache_hint` and `.mbarrier::complete_tx::bytes` each lex as one `dot`. Lines starting with
`#` (preprocessor output markers) are skipped. Every token records its line.
-/

namespace NV.PTX

inductive Tok where
  | ident (s : String)
  /-- `.name`, without the dot. -/
  | dot (s : String)
  /-- An integer literal; `unsigned` when written with `U` or too large for `.s64` (§4.5.1). -/
  | int (n : Nat) (unsigned : Bool)
  /-- A decimal floating literal, exactly `mant · 10^exp`. -/
  | dec (mant : Nat) (exp : Int)
  /-- `0f…`: the bits of an `.f32`. -/
  | f32 (bits : UInt32)
  /-- `0d…`: the bits of an `.f64`. -/
  | f64 (bits : UInt64)
  | str (s : String)
  /-- Punctuation and operators: `, ; : { } [ ] ( ) < > @ ! + - * / % & | ^ ~ ? =` and the
  two-character `== != <= >= << >> && ||`. -/
  | sym (s : String)
  deriving DecidableEq, Repr, Inhabited

structure Token where
  tok : Tok
  line : Nat
  deriving Repr, Inhabited

namespace Lex

def isFollow (c : UInt8) : Bool :=
  (c ≥ 'a'.toUInt8 && c ≤ 'z'.toUInt8) || (c ≥ 'A'.toUInt8 && c ≤ 'Z'.toUInt8) ||
  (c ≥ '0'.toUInt8 && c ≤ '9'.toUInt8) || c == '_'.toUInt8 || c == '$'.toUInt8

def isDigit (c : UInt8) : Bool := c ≥ '0'.toUInt8 && c ≤ '9'.toUInt8

def hexVal (c : UInt8) : Option Nat :=
  if isDigit c then some (c - '0'.toUInt8).toNat
  else if c ≥ 'a'.toUInt8 && c ≤ 'f'.toUInt8 then some (c - 'a'.toUInt8 + 10).toNat
  else if c ≥ 'A'.toUInt8 && c ≤ 'F'.toUInt8 then some (c - 'A'.toUInt8 + 10).toNat
  else none

structure St where
  src : ByteArray
  pos : Nat := 0
  line : Nat := 1
  out : Array Token := #[]

abbrev M := StateT St (Except String)

def peekAt (k : Nat) : M UInt8 := do
  let s ← get
  return if h : s.pos + k < s.src.size then s.src[s.pos + k] else 0

def peek : M UInt8 := peekAt 0

def advance (k : Nat := 1) : M Unit := modify fun s => { s with pos := s.pos + k }

def emit (t : Tok) : M Unit := modify fun s => { s with out := s.out.push ⟨t, s.line⟩ }

def fail {α : Type} (msg : String) : M α := do throw s!"line {(← get).line}: {msg}"

/-- The bytes `[a, b)` of the source as a string. -/
def slice (a b : Nat) : M String := do
  return String.fromUTF8! ((← get).src.extract a b)

/-- Advance while `p` holds; at most the rest of the input. -/
def skipWhile (p : UInt8 → Bool) : M Unit := do
  let n := (← get).src.size
  for _ in [0:n] do
    let c ← peek
    if c != 0 && p c then advance else break

/-- Skip to the end of the line (not consuming the newline). -/
def skipLine : M Unit := skipWhile (· != '\n'.toUInt8)

def skipBlockComment : M Unit := do
  advance 2
  let n := (← get).src.size
  for _ in [0:n] do
    let c ← peek
    if c == 0 then fail "unterminated comment"
    if c == '*'.toUInt8 && (← peekAt 1) == '/'.toUInt8 then advance 2; return
    if c == '\n'.toUInt8 then modify fun s => { s with line := s.line + 1 }
    advance

/-- Digits of base `b` as a natural number, with how many there were. `'` is not allowed. -/
def digits (b : Nat) : M (Nat × Nat) := do
  let mut v := 0
  let mut k := 0
  let n := (← get).src.size
  for _ in [0:n] do
    match hexVal (← peek) with
    | some d => if d < b then v := v * b + d; k := k + 1; advance else break
    | none => break
  return (v, k)

/-- The rest of a decimal floating literal after its integer part: `.fraction` and exponent. -/
def decimalRest (ipart : Nat) : M Unit := do
  let mut mant := ipart
  let mut exp : Int := 0
  if (← peek) == '.'.toUInt8 then
    advance
    let (f, k) ← digits 10
    mant := mant * 10 ^ k + f
    exp := -k
  let c ← peek
  if c == 'e'.toUInt8 || c == 'E'.toUInt8 then
    advance
    let sgn ← peek
    let neg := sgn == '-'.toUInt8
    if sgn == '-'.toUInt8 || sgn == '+'.toUInt8 then advance
    let (e, k) ← digits 10
    if k == 0 then fail "malformed exponent"
    exp := exp + (if neg then -(e : Int) else e)
  emit (.dec mant exp)

/-- Whether the `.` at the cursor starts a number (`.05`) rather than a keyword (`.1d`,
`.16x64b`): digits followed by neither a letter other than an exponent nor `_`. -/
def dotNumber : M Bool := do
  let s ← get
  let mut k := s.pos + 1
  let n := s.src.size
  for _ in [0:n] do
    if k < n && isDigit s.src[k]! then k := k + 1 else break
  if k == s.pos + 1 then return false
  let c := if k < n then s.src[k]! else 0
  return !isFollow c || c == 'e'.toUInt8 || c == 'E'.toUInt8

/-- A number: integer (decimal, hex, octal, binary, optional `U`), `0f`/`0d` bits, or decimal
floating (`1.5`, `2e-3`, `1.`). -/
def number : M Unit := do
  let c0 ← peek
  let c1 ← peekAt 1
  if c0 == '0'.toUInt8 && (c1 == 'f'.toUInt8 || c1 == 'F'.toUInt8 || c1 == 'd'.toUInt8 ||
      c1 == 'D'.toUInt8) && (hexVal (← peekAt 2)).isSome then
    advance 2
    let (v, k) ← digits 16
    if c1 == 'f'.toUInt8 || c1 == 'F'.toUInt8 then
      unless k == 8 do fail "0f literal needs 8 hex digits"
      emit (.f32 v.toUInt32)
    else
      unless k == 16 do fail "0d literal needs 16 hex digits"
      emit (.f64 v.toUInt64)
    return
  let intLit (v : Nat) : M Unit := do
    let c ← peek
    let u := c == 'U'.toUInt8
    if u then advance
    emit (.int v (u || v ≥ 2 ^ 63))
  if c0 == '0'.toUInt8 && (c1 == 'x'.toUInt8 || c1 == 'X'.toUInt8) then
    advance 2; intLit (← digits 16).1; return
  if c0 == '0'.toUInt8 && (c1 == 'b'.toUInt8 || c1 == 'B'.toUInt8) && isDigit (← peekAt 2) then
    advance 2; intLit (← digits 2).1; return
  -- Decimal (or octal) integer, or decimal floating.
  let start := (← get).pos
  let (ipart, _) ← digits 10
  let c ← peek
  if c == '.'.toUInt8 || c == 'e'.toUInt8 || c == 'E'.toUInt8 then decimalRest ipart
  else
    -- C rule: a leading 0 means octal.
    if c0 == '0'.toUInt8 && (← get).pos > start + 1 then
      modify fun s => { s with pos := start }
      intLit (← digits 8).1
    else intLit ipart

def twoChar : List String := ["==", "!=", "<=", ">=", "<<", ">>", "&&", "||"]
def oneChar : List Char := [',', ';', ':', '{', '}', '[', ']', '(', ')', '<', '>', '@', '!', '+',
  '-', '*', '/', '%', '&', '|', '^', '~', '?', '=']

def string : M Unit := do
  advance
  let start := (← get).pos
  skipWhile (fun c => c != '"'.toUInt8 && c != '\n'.toUInt8)
  if (← peek) != '"'.toUInt8 then fail "unterminated string"
  let s ← slice start (← get).pos
  advance
  emit (.str s)

/-- One token or one piece of whitespace. -/
def step : M Unit := do
  let c ← peek
  let pos := (← get).pos
  if c == '\n'.toUInt8 then
    modify fun s => { s with line := s.line + 1 }; advance
  else if c == ' '.toUInt8 || c == '\t'.toUInt8 || c == '\r'.toUInt8 || c == '\x0c'.toUInt8 then
    advance
  else if c == '/'.toUInt8 && (← peekAt 1) == '/'.toUInt8 then skipLine
  else if c == '/'.toUInt8 && (← peekAt 1) == '*'.toUInt8 then skipBlockComment
  else if c == '#'.toUInt8 then skipLine
  else if c == '"'.toUInt8 then string
  else if isDigit c then number
  else if c == '.'.toUInt8 && (← dotNumber) then decimalRest 0
  else if c == '.'.toUInt8 && isFollow (← peekAt 1) then
    advance
    let start := pos + 1
    -- `.name` with `::part` continuations
    let n := (← get).src.size
    for _ in [0:n] do
      skipWhile isFollow
      if (← peek) == ':'.toUInt8 && (← peekAt 1) == ':'.toUInt8 && isFollow (← peekAt 2) then
        advance 2
      else break
    emit (.dot (← slice start (← get).pos))
  else if isFollow c && !isDigit c ||
      ((c == '%'.toUInt8) && isFollow (← peekAt 1)) then
    advance
    skipWhile isFollow
    emit (.ident (← slice pos (← get).pos))
  else
    let c1 ← peekAt 1
    let two := String.ofList [Char.ofUInt8 c, Char.ofUInt8 c1]
    if twoChar.contains two then advance 2; emit (.sym two)
    else if oneChar.contains (Char.ofUInt8 c) then advance; emit (.sym (String.ofList [Char.ofUInt8 c]))
    else fail s!"unexpected character {repr (Char.ofUInt8 c)}"

end Lex

/-- The tokens of a PTX source text. -/
def lex (src : String) : Except String (Array Token) := do
  let bytes := src.toUTF8
  let go : Lex.M Unit := do
    for _ in [0:bytes.size] do
      if (← get).pos ≥ bytes.size then break
      Lex.step
    if (← get).pos < bytes.size then Lex.fail "lexer did not terminate"
  let ((), s) ← go.run { src := bytes }
  return s.out

end NV.PTX
