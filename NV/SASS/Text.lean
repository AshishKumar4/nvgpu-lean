/-!
# Text helpers for the disassembly and ELF readers

Lines are handled as character lists, which keeps the readers to plain structural recursion.
-/

namespace NV.SASS.Text

def isWs (c : Char) : Bool := c == ' ' || c == '\t' || c == '\r'

def trimL (cs : List Char) : List Char := ((cs.dropWhile isWs).reverse.dropWhile isWs).reverse

def str (cs : List Char) : String := String.ofList cs

def trim (s : String) : String := str (trimL s.toList)

/-- Split at every character satisfying `p`. -/
def splitWhen (p : Char → Bool) (cs : List Char) : List (List Char) :=
  let (out, cur) := cs.foldl
    (fun (out, cur) c => if p c then (cur.reverse :: out, []) else (out, c :: cur)) ([], [])
  (cur.reverse :: out).reverse

def hexDigit? (c : Char) : Option Nat :=
  if '0' ≤ c && c ≤ '9' then some (c.toNat - '0'.toNat)
  else if 'a' ≤ c && c ≤ 'f' then some (c.toNat - 'a'.toNat + 10)
  else if 'A' ≤ c && c ≤ 'F' then some (c.toNat - 'A'.toNat + 10)
  else none

/-- The value of a non-empty digit string in `base` (at most 16). -/
def digitsVal (base : Nat) (cs : List Char) : Option Nat :=
  if cs.isEmpty then none
  else cs.foldlM (fun acc c => (hexDigit? c).bind fun d =>
    if d < base then some (acc * base + d) else none) 0

/-- `0x1f`. -/
def hex? : List Char → Option Nat
  | '0' :: 'x' :: ds => digitsVal 16 ds
  | _ => none

/-- Whether substring `t` starts with `p`, without copying. -/
@[inline] def starts (t : Substring.Raw) (p : String) : Bool :=
  t.startPos.byteIdx + p.utf8ByteSize ≤ t.stopPos.byteIdx &&
    String.Pos.Raw.substrEq t.str t.startPos p 0 p.utf8ByteSize

/-- Whether substring `t` ends with `p`, without copying. -/
@[inline] def ends (t : Substring.Raw) (p : String) : Bool :=
  t.startPos.byteIdx + p.utf8ByteSize ≤ t.stopPos.byteIdx &&
    String.Pos.Raw.substrEq t.str ⟨t.stopPos.byteIdx - p.utf8ByteSize⟩ p 0 p.utf8ByteSize

/-- The lines of `text` as slices of it, without copying or decoding: a scan of its bytes for
newlines (a newline byte never occurs inside a multi-byte UTF-8 character). -/
def lines (text : String) : Array Substring.Raw :=
  let bs := text.toUTF8
  -- A loop over a machine-word index.
  let rec go (i start : USize) (out : Array Substring.Raw) : Array Substring.Raw :=
    if h : i < bs.usize then
      have : i.toNat < bs.size := Nat.lt_of_lt_of_le (USize.lt_iff_toNat_lt.mp h)
        (by simp only [ByteArray.usize, Nat.toUSize_eq, USize.toNat_ofNat']; exact Nat.mod_le _ _)
      if bs.uget i this == 10 then go (i + 1) (i + 1) (out.push ⟨text, ⟨start.toNat⟩, ⟨i.toNat⟩⟩)
      else go (i + 1) start out
    else out.push ⟨text, ⟨start.toNat⟩, ⟨bs.size⟩⟩
  termination_by bs.size - i.toNat
  decreasing_by
    all_goals
      have h1 : i.toNat < bs.usize.toNat := USize.lt_iff_toNat_lt.mp h
      have h2 : bs.usize.toNat < 2 ^ System.Platform.numBits := USize.toNat_lt_size _
      have e : (i + 1).toNat = i.toNat + 1 := by
        rw [USize.toNat_add, USize.toNat_one]; exact Nat.mod_eq_of_lt (by omega)
      omega
  go 0 0 #[]

/-- `l` without its leading spaces and tabs, found by reading bytes rather than decoding. -/
def dropBlank (l : Substring.Raw) : Substring.Raw :=
  let rec go : Nat → Nat → Nat
    | 0, i => i
    | fuel + 1, i =>
      if h : (⟨i⟩ : String.Pos.Raw) < l.str.rawEndPos then
        if i ≥ l.stopPos.byteIdx then i else
        let b := l.str.getUTF8Byte ⟨i⟩ h
        if b == 32 || b == 9 then go fuel (i + 1) else i
      else i
  { l with startPos := ⟨go (l.stopPos.byteIdx - l.startPos.byteIdx) l.startPos.byteIdx⟩ }

/-- A hexadecimal (`0x1f`) or decimal number. -/
def num? (s : String) : Option Nat :=
  let cs := trimL s.toList
  (hex? cs).orElse fun _ => digitsVal 10 cs

end NV.SASS.Text
