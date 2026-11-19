import NV.SASS.Exec
import NV.Num

/-!
# Floating-point operands and opcode forms (`sm_90`)

What the float opcodes share: reading a float source with its sign modifiers (`-R0`, `|R0|` are
sign-bit operations here, not integer negation), half-word selection for packed formats
(`R0.H1`, `R0.H0_H0`), and the rounding, flush and saturation modifiers. The arithmetic is
`NV.Num`'s, so PTX and SASS compute with one definition.
-/

namespace NV.SASS.Float

open NV NV.Num Exec

/-- A float source's bits: the register (or immediate, or constant) with `|·|` then `-` applied
to the sign bits in `signs` (bit 31 for binary32, bits 15 and 31 for a packed pair). -/
def src (env : Env) (w : View) (lane : Nat) (signs : UInt32) : Operand → Option UInt32
  | .reg r m => (Exec.src env w lane (.reg r { m with neg := false, abs := false, not := false })).map
      fun v => let v := if m.abs then v &&& ~~~signs else v; if m.neg then v ^^^ signs else v
  -- `-c[…]` (sm_80): the constant's sign bits flipped.
  | .neg o => (src env w lane signs o).map (· ^^^ signs)
  | o => Exec.src env w lane o

/-- A binary64 source from a register pair (or an immediate's 64-bit pattern, or two constant
words), its sign modifiers on bit 63. -/
def src64 (env : Env) (w : View) (lane : Nat) (o : Operand) : Option UInt64 := do
  -- `-c[…]`: the constant pair, its sign flipped.
  if let .neg c := o then return (← src64 env w lane c) ^^^ 0x8000000000000000
  let lo ← Exec.src env w lane (match o with
    | .reg r m => .reg r { m with neg := false, abs := false, not := false } | o => o)
  let hi ← srcHi env w lane o
  let v := toU64 lo hi
  match o with
  | .reg _ m =>
    let sign : UInt64 := 0x8000000000000000
    let v := if m.abs then v &&& ~~~sign else v
    some (if m.neg then v ^^^ sign else v)
  | .imm b => some b
  | _ => some v

/-- The two 16-bit halves of a packed source after its selector: `.H0_H0` and `.H1_H1` repeat one
half, `.H1` (a lone half) is the high one in the low position. Returns (high, low). -/
def halves (o : Operand) (v : UInt32) : UInt32 × UInt32 :=
  let lo := v &&& 0xffff; let hi := v >>> 16
  match o with
  | .reg _ m =>
    if m.suffix.contains "H0_H0" then (lo, lo) else if m.suffix.contains "H1_H1" then (hi, hi)
    else if m.suffix.contains "H1" then (0, hi) else (hi, lo)
  | _ => (hi, lo)

/-- The rounding a modifier names; round to nearest even when none does. -/
def rounding (mods : Array String) : Rounding :=
  if mods.contains "RZ" || mods.contains "TRUNC" then .rz
  else if mods.contains "RM" || mods.contains "FLOOR" then .rm
  else if mods.contains "RP" || mods.contains "CEIL" then .rp
  else .rn

/-- The comparison a `FSETP`/`DSETP`/`HSETP2` modifier names. -/
def cmpOp : String → Option (Option CmpOp)
  | "EQ" => some (some .eq) | "NE" => some (some .ne) | "LT" => some (some .lt)
  | "LE" => some (some .le) | "GT" => some (some .gt) | "GE" => some (some .ge)
  | "EQU" => some (some .equ) | "NEU" => some (some .neu) | "LTU" => some (some .ltu)
  | "LEU" => some (some .leu) | "GTU" => some (some .gtu) | "GEU" => some (some .geu)
  | "NUM" => some (some .num) | "NAN" => some (some .nan)
  | "T" | "F" => some none
  | _ => none

/-- A float format named by a conversion modifier. -/
def format : String → Option Format
  | "F32" => some .f32 | "F64" => some .f64 | "F16" => some .f16 | "BF16" => some .bf16
  | "TF32" => some .tf32 | "E4M3" => some .e4m3 | "E5M2" => some .e5m2
  | _ => none

/-- An integer type named by a conversion modifier: signed and width. -/
def intType : String → Option (Bool × Nat)
  | "S32" => some (true, 32) | "U32" => some (false, 32) | "S64" => some (true, 64)
  | "U64" => some (false, 64) | "S16" => some (true, 16) | "U16" => some (false, 16)
  | "S8" => some (true, 8) | "U8" => some (false, 8)
  | _ => none

end NV.SASS.Float
