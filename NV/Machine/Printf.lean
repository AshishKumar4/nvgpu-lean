/-!
# Device-side `printf`

A kernel's `printf(fmt, args…)` is a call to `vprintf(fmt, buf)`: `buf` holds the arguments, each
at its natural alignment (integers of 4 bytes unless a length modifier makes them 8, `double` 8
bytes — `float` arguments arrive promoted — pointers and `%s` strings 8; CUDA Programming Guide, "Formatted Output"). The simulators record each call's
format address and argument bytes; `render` turns a record into the text the host would print,
as C's `printf` formats it: flags `-+ 0#`, width and precision (`*` too), lengths `hh h l ll z j
t`, conversions `d i u o x X c s p e E f F g G %`.
-/

namespace NV.Printf

/-- A conversion specification. -/
structure Spec where
  flags : List Char := []
  width : Option Nat := none
  widthArg : Bool := false
  prec : Option Nat := none
  precArg : Bool := false
  long : Bool := false
  conv : Char
  deriving Repr, Inhabited

inductive Piece where
  | lit (s : String)
  | spec (s : Spec)
  deriving Repr, Inhabited

/-- The pieces of a format string. -/
def pieces (fmt : String) : List Piece := Id.run do
  let cs := fmt.toList.toArray
  let mut out : Array Piece := #[]
  let mut lit : Array Char := #[]
  let mut i := 0
  for _ in [0:cs.size] do
    if i ≥ cs.size then break
    let c := cs[i]!
    if c != '%' then lit := lit.push c; i := i + 1; continue
    if cs[i + 1]? == some '%' then lit := lit.push '%'; i := i + 2; continue
    if !lit.isEmpty then out := out.push (.lit (String.ofList lit.toList)); lit := #[]
    i := i + 1
    let mut sp : Spec := { conv := '?' }
    for _ in [0:cs.size] do
      match cs[i]? with
      | some f => if "-+ 0#".contains f then sp := { sp with flags := sp.flags ++ [f] }; i := i + 1
                  else break
      | none => break
    let num (j : Nat) : Nat × Nat := Id.run do
      let mut v := 0
      let mut j := j
      for _ in [0:cs.size] do
        match cs[j]? with
        | some d => if d.isDigit then v := 10 * v + (d.toNat - '0'.toNat); j := j + 1 else break
        | none => break
      return (v, j)
    if cs[i]? == some '*' then sp := { sp with widthArg := true }; i := i + 1
    else if (cs[i]?.map Char.isDigit) == some true then
      let (v, j) := num i
      sp := { sp with width := some v }; i := j
    if cs[i]? == some '.' then
      i := i + 1
      if cs[i]? == some '*' then sp := { sp with precArg := true }; i := i + 1
      else
        let (v, j) := num i
        sp := { sp with prec := some v }; i := j
    for _ in [0:3] do
      match cs[i]? with
      | some 'l' | some 'z' | some 'j' | some 't' => sp := { sp with long := true }; i := i + 1
      | some 'h' | some 'L' => i := i + 1
      | _ => break
    match cs[i]? with
    | some c => sp := { sp with conv := c }; i := i + 1
    | none => pure ()
    out := out.push (.spec sp)
  if !lit.isEmpty then out := out.push (.lit (String.ofList lit.toList))
  return out.toList

/-- Bytes an argument of a conversion takes. -/
def argSize (s : Spec) : Nat :=
  if "eEfFgGaAsp".contains s.conv then 8 else if s.long then 8 else 4

/-- The argument slots of a format: per argument (`*` widths and precisions too), its offset in
the buffer and size; and the buffer's length. -/
def layout (fmt : String) : List (Nat × Nat) × Nat := Id.run do
  let mut off := 0
  let mut out := #[]
  for p in pieces fmt do
    let .spec s := p | continue
    for size in (if s.widthArg then [4] else []) ++ (if s.precArg then [4] else []) ++ [argSize s] do
      off := (off + size - 1) / size * size
      out := out.push (off, size)
      off := off + size
  return (out.toList, off)

def le (b : ByteArray) (off n : Nat) : Nat :=
  (List.range n).foldl (fun v j => v ||| ((b.get! (off + j)).toNat <<< (8 * j))) 0

def digits (base : Nat) (upper : Bool) (n : Nat) : String :=
  let ds := if upper then "0123456789ABCDEF" else "0123456789abcdef"
  let rec go (n : Nat) (fuel : Nat) (acc : List Char) : List Char :=
    match fuel with
    | 0 => acc
    | fuel + 1 => let acc := ds.toList[n % base]! :: acc; if n < base then acc else go (n / base) fuel acc
  String.ofList (go n 70 [])

def pad (s : Spec) (sign body : String) (numeric : Bool) : String :=
  let w := s.width.getD 0
  let len := sign.length + body.length
  if len ≥ w then sign ++ body
  else if s.flags.contains '-' then sign ++ body ++ String.ofList (List.replicate (w - len) ' ')
  else if numeric && s.flags.contains '0' then sign ++ String.ofList (List.replicate (w - len) '0') ++ body
  else String.ofList (List.replicate (w - len) ' ') ++ sign ++ body

def signOf (s : Spec) (neg : Bool) : String :=
  if neg then "-" else if s.flags.contains '+' then "+" else if s.flags.contains ' ' then " " else ""

/-- `n / d` rounded to the nearest integer, ties to even (glibc's printf rounds exactly). -/
def roundDiv (n d : Nat) : Nat :=
  let q := n / d
  let r := n % d
  if 2 * r > d || (2 * r == d && q % 2 == 1) then q + 1 else q

/-- A finite binary64 `m · 2^e` (`m` the integer significand) as `num / den`. -/
def ratio (m : Nat) (e : Int) : Nat × Nat :=
  if e ≥ 0 then (m * 2 ^ e.toNat, 1) else (m, 2 ^ (-e).toNat)

/-- Fixed notation with `prec` digits after the point. -/
def fixed (num den prec : Nat) (alt : Bool) : String :=
  let v := roundDiv (num * 10 ^ prec) den
  let ip := v / 10 ^ prec
  let fp := v % 10 ^ prec
  let fs := digits 10 false fp
  let fs := String.ofList (List.replicate (prec - fs.length) '0') ++ fs
  digits 10 false ip ++ (if prec > 0 || alt then "." else "") ++ (if prec > 0 then fs else "")

/-- Exponent notation: one digit, `prec` after the point, then the exponent. -/
def sci (num den prec : Nat) (upper alt : Bool) : String := Id.run do
  if num == 0 then
    return fixed 0 1 prec alt ++ (if upper then "E+00" else "e+00")
  -- The decimal exponent x with 10^x ≤ num/den < 10^(x+1), adjusted after rounding.
  let mut x : Int := (Nat.log2 num : Int) - Nat.log2 den
  x := x * 30103 / 100000
  let scale (x : Int) : Nat × Nat :=
    if x ≥ 0 then (num, den * 10 ^ x.toNat) else (num * 10 ^ (-x).toNat, den)
  for _ in [0:4] do
    let (n, d) := scale x
    if n < d then x := x - 1 else if n ≥ 10 * d then x := x + 1 else break
  let (n, d) := scale x
  let mut v := roundDiv (n * 10 ^ prec) d
  if v ≥ 10 ^ (prec + 1) then x := x + 1; v := v / 10
  let ip := v / 10 ^ prec
  let fp := v % 10 ^ prec
  let fs := digits 10 false fp
  let fs := String.ofList (List.replicate (prec - fs.length) '0') ++ fs
  let es := digits 10 false x.natAbs
  let es := if es.length < 2 then "0" ++ es else es
  return digits 10 false ip ++ (if prec > 0 || alt then "." else "") ++ (if prec > 0 then fs else "") ++
    (if upper then "E" else "e") ++ (if x < 0 then "-" else "+") ++ es

/-- Drop trailing zeros of a fraction (and a bare point), for `%g` without `#`. -/
def trimZeros (s : String) : String :=
  if !s.contains '.' then s else
  let (mant, ex) := match s.splitOn "e", s.splitOn "E" with
    | [m, e], _ => (m, "e" ++ e)
    | _, [m, e] => (m, "E" ++ e)
    | _, _ => (s, "")
  let m := (mant.toList.reverse.dropWhile (· == '0')).reverse
  let m := if m.getLast? == some '.' then m.dropLast else m
  String.ofList m ++ ex

/-- A binary64 argument by a floating conversion. -/
def float (s : Spec) (bits : Nat) : String :=
  let neg := bits >>> 63 == 1
  let e := (bits >>> 52) % 2048
  let f := bits % 2 ^ 52
  let upper := s.conv.isUpper
  if e == 2047 then
    let body := if f == 0 then (if upper then "INF" else "inf") else (if upper then "NAN" else "nan")
    pad { s with flags := s.flags.filter (· != '0') } (signOf s neg) body false
  else
    let (m, ex) : Nat × Int := if e == 0 then (f, -1074) else (f + 2 ^ 52, (e : Int) - 1075)
    let (num, den) := ratio m ex
    let prec := s.prec.getD 6
    let alt := s.flags.contains '#'
    let body := match s.conv with
      | 'f' | 'F' => fixed num den prec alt
      | 'e' | 'E' => sci num den prec upper alt
      | _ =>
        -- %g: exponent notation when the exponent is below -4 or at least the precision.
        let p := if prec == 0 then 1 else prec
        let e10 := sci num den (p - 1) upper true
        let xs := ((e10.splitOn (if upper then "E" else "e"))[1]?).getD "+0"
        let mag : Int := ((xs.drop 1).toString.toNat?.getD 0 : Nat)
        let x : Int := if xs.startsWith "-" then -mag else mag
        let useSci : Bool := num != 0 && (x < -4 || (p : Int) ≤ x)
        let r := if useSci then sci num den (p - 1) upper alt
          else fixed num den (Int.toNat ((p : Int) - 1 - x)) alt
        if alt then r else trimZeros r
    pad s (signOf s neg) body true

/-- The text of one record: format, argument bytes, and the strings `%s` arguments point to. -/
def render (fmt : String) (args : ByteArray) (str : UInt64 → Option String) : String := Id.run do
  let (slots, _) := layout fmt
  let mut slots := slots
  let mut out := ""
  let next (slots : List (Nat × Nat)) : Nat × List (Nat × Nat) :=
    match slots with
    | (off, n) :: rest => (le args off n, rest)
    | [] => (0, [])
  for p in pieces fmt do
    match p with
    | .lit t => out := out ++ t
    | .spec s0 =>
      let mut s := s0
      if s.widthArg then
        let (w, r) := next slots; slots := r
        let w := w % 2 ^ 32
        s := if w ≥ 2 ^ 31 then { s with width := some (2 ^ 32 - w), flags := s.flags ++ ['-'] }
          else { s with width := some w }
      if s.precArg then
        let (v, r) := next slots; slots := r
        s := { s with prec := if v % 2 ^ 32 ≥ 2 ^ 31 then none else some (v % 2 ^ 32) }
      let (v, r) := next slots; slots := r
      let bits := if s.long || "eEfFgGaAsp".contains s.conv then 64 else 32
      let intBody (n : Nat) : String :=
        let b := match s.conv with | 'o' => 8 | 'x' | 'X' => 16 | _ => 10
        let ds := digits b (s.conv == 'X') n
        let ds := match s.prec with
          | some 0 => if n == 0 then "" else ds
          | some p => String.ofList (List.replicate (p - ds.length) '0') ++ ds
          | none => ds
        let pre := if s.flags.contains '#' && n != 0 then
          (match s.conv with | 'x' => "0x" | 'X' => "0X" | 'o' => "0" | _ => "") else ""
        pre ++ ds
      let piece : String := match s.conv with
        | 'd' | 'i' =>
          let neg := v >>> (bits - 1) == 1
          let mag := if neg then 2 ^ bits - v else v
          pad (if s.prec.isSome then { s with flags := s.flags.filter (· != '0') } else s)
            (signOf s neg) (intBody mag) true
        | 'u' | 'o' | 'x' | 'X' =>
          pad (if s.prec.isSome then { s with flags := s.flags.filter (· != '0') } else s) ""
            (intBody v) true
        | 'c' => pad s "" (String.ofList [Char.ofNat (v % 256)]) false
        | 'p' => pad s "" ("0x" ++ digits 16 false v) false
        | 's' =>
          let t := (str v.toUInt64).getD "(null)"
          pad s "" (match s.prec with | some p => (t.take p).toString | none => t) false
        | 'e' | 'E' | 'f' | 'F' | 'g' | 'G' | 'a' | 'A' => float s v
        | c => s!"%{c}"
      out := out ++ piece
  return out

end NV.Printf
