import NV.PTX.Syntax.Keyword

/-!
# PTX types and state spaces

PTX ISA 9.4 §5.1 (state spaces) and §5.2 (types). `Ty` holds every type keyword an instruction
may carry: the fundamental types (§5.2.1), the alternate floating-point formats (§5.2.3), the
fixed-point format (§5.2.4) and the packed types (§5.2.5), plus the sub-byte matrix element
types of `mma` (§9.7.14). Only fundamental types can declare variables; the others appear as
instruction types over bit-typed registers.
-/

namespace NV.PTX

inductive Ty where
  | b1 | b2 | b4 | b8 | b16 | b32 | b64 | b128 | b1024
  | u2 | u4 | u8 | u16 | u32 | u64
  | s2 | s4 | s8 | s16 | s32 | s64
  | f16 | f16x2 | f32 | f64 | pred
  | bf16 | bf16x2 | tf32 | f32x2
  | e4m3 | e5m2 | e2m3 | e3m2 | e2m1 | ue8m0 | ue4m3 | ue5m3 | s2f6
  | e4m3x2 | e5m2x2 | e2m3x2 | e3m2x2 | e2m1x2 | ue8m0x2 | ue5m3x2 | s2f6x2
  | e4m3x4 | e5m2x4 | e2m3x4 | e3m2x4 | e2m1x4 | ue8m0x4 | e2m1p4x4
  | u16x2 | s16x2 | u8x4 | s8x4 | b16x2
  /-- `ldmatrix` and `tcgen05.cp` element formats: 16 bytes, from 6- or 4-bit elements padded. -/
  | b8x16 | b6x16p32 | b4x16p64
  /-- Opaque texture, sampler and surface references (§5.3). -/
  | texref | samplerref | surfref
  deriving DecidableEq, Repr, Inhabited

instance : Keyword Ty where
  all := [.b1, .b2, .b4, .b8, .b16, .b32, .b64, .b128, .b1024, .u2, .u4, .u8, .u16, .u32, .u64,
    .s2, .s4, .s8, .s16, .s32, .s64, .f16, .f16x2, .f32, .f64, .pred, .bf16, .bf16x2, .tf32, .f32x2,
    .e4m3, .e5m2, .e2m3, .e3m2, .e2m1, .ue8m0, .ue4m3, .ue5m3, .s2f6,
    .e4m3x2, .e5m2x2, .e2m3x2, .e3m2x2, .e2m1x2, .ue8m0x2, .ue5m3x2, .s2f6x2,
    .e4m3x4, .e5m2x4, .e2m3x4, .e3m2x4, .e2m1x4, .ue8m0x4, .e2m1p4x4,
    .u16x2, .s16x2, .u8x4, .s8x4, .b16x2, .b8x16, .b6x16p32, .b4x16p64, .texref, .samplerref, .surfref]
  name
    | .b1 => "b1" | .b2 => "b2" | .b1024 => "b1024" | .u2 => "u2" | .s2 => "s2"
    | .b8x16 => "b8x16" | .b6x16p32 => "b6x16_p32" | .b4x16p64 => "b4x16_p64" | .b4 => "b4" | .b8 => "b8" | .b16 => "b16" | .b32 => "b32" | .b64 => "b64"
    | .b128 => "b128" | .u4 => "u4" | .u8 => "u8" | .u16 => "u16" | .u32 => "u32"
    | .u64 => "u64" | .s4 => "s4" | .s8 => "s8" | .s16 => "s16" | .s32 => "s32"
    | .s64 => "s64" | .f16 => "f16" | .f16x2 => "f16x2" | .f32 => "f32" | .f64 => "f64"
    | .pred => "pred" | .bf16 => "bf16" | .bf16x2 => "bf16x2" | .tf32 => "tf32"
    | .f32x2 => "f32x2" | .e4m3 => "e4m3" | .e5m2 => "e5m2" | .e2m3 => "e2m3"
    | .e3m2 => "e3m2" | .e2m1 => "e2m1" | .ue8m0 => "ue8m0" | .ue4m3 => "ue4m3"
    | .ue5m3 => "ue5m3" | .s2f6 => "s2f6" | .e4m3x2 => "e4m3x2" | .e5m2x2 => "e5m2x2"
    | .e2m3x2 => "e2m3x2" | .e3m2x2 => "e3m2x2" | .e2m1x2 => "e2m1x2" | .ue8m0x2 => "ue8m0x2"
    | .ue5m3x2 => "ue5m3x2" | .s2f6x2 => "s2f6x2" | .e4m3x4 => "e4m3x4" | .e5m2x4 => "e5m2x4"
    | .e2m3x4 => "e2m3x4" | .e3m2x4 => "e3m2x4" | .e2m1x4 => "e2m1x4" | .ue8m0x4 => "ue8m0x4"
    | .e2m1p4x4 => "e2m1p4x4" | .u16x2 => "u16x2" | .s16x2 => "s16x2" | .u8x4 => "u8x4"
    | .s8x4 => "s8x4" | .b16x2 => "b16x2" | .texref => "texref" | .samplerref => "samplerref"
    | .surfref => "surfref"

theorem Ty.complete (t : Ty) : t ∈ (Keyword.all : List Ty) := by cases t <;> decide

/-- The basic type of §5.2.1, which decides type compatibility (§5.2.1, "Two fundamental types
are compatible if they have the same basic type and are the same size"). -/
inductive Basic where
  | bits | unsigned | signed | float | pred
  /-- An alternate or packed format, held in bit-typed registers (§5.2.3, §5.2.5). -/
  | format
  /-- Texture, sampler and surface references (§5.3). -/
  | ref
  deriving DecidableEq, Repr, Inhabited

namespace Ty

def basic : Ty → Basic
  | .b1 | .b2 | .b4 | .b8 | .b16 | .b32 | .b64 | .b128 | .b1024 => .bits
  | .u2 | .u4 | .u8 | .u16 | .u32 | .u64 => .unsigned
  | .s2 | .s4 | .s8 | .s16 | .s32 | .s64 => .signed
  | .f16 | .f16x2 | .f32 | .f64 => .float
  | .pred => .pred
  | .texref | .samplerref | .surfref => .ref
  | _ => .format

/-- Width in bits of a value of this type as it sits in a register or in memory. Predicates are
one bit (§5.1.1); the sub-byte `mma` types are their element width; the six-bit formats sit in a
byte (§5.2.3). -/
def bits : Ty → Nat
  | .pred | .b1 => 1
  | .b2 | .u2 | .s2 => 2
  | .b4 | .u4 | .s4 | .e2m1 => 4
  | .b8 | .u8 | .s8 | .e4m3 | .e5m2 | .e2m3 | .e3m2 | .ue8m0 | .ue4m3 | .ue5m3 | .s2f6
  | .e2m1x2 | .e2m1p4x4 => 8
  | .b16 | .u16 | .s16 | .f16 | .bf16 | .e4m3x2 | .e5m2x2 | .e2m3x2 | .e3m2x2 | .ue8m0x2
  | .ue5m3x2 | .s2f6x2 | .e2m1x4 => 16
  | .b32 | .u32 | .s32 | .f32 | .f16x2 | .bf16x2 | .tf32 | .e4m3x4 | .e5m2x4 | .e2m3x4
  | .e3m2x4 | .ue8m0x4 | .u16x2 | .s16x2 | .u8x4 | .s8x4 | .b16x2 => 32
  | .b64 | .u64 | .s64 | .f64 | .f32x2 | .texref | .samplerref | .surfref => 64
  | .b128 | .b8x16 | .b6x16p32 | .b4x16p64 => 128
  | .b1024 => 1024

def bytes (t : Ty) : Nat := (t.bits + 7) / 8

/-- Types that can declare a variable (§5.2.1 Table 8, and `.b128` since PTX 8.3). -/
def fundamental (t : Ty) : Bool :=
  t.basic != .format && t.basic != .ref && !(t matches .b1 | .b2 | .b4 | .u2 | .u4 | .s2 | .s4 | .b1024)

def isInt (t : Ty) : Bool := t.basic matches .unsigned | .signed
def isSigned (t : Ty) : Bool := t.basic == .signed
def isFloat (t : Ty) : Bool := t.basic == .float

/-- §5.2.1: same basic type and size, signed with unsigned of one size, or bits with anything
of its size. -/
def compatible (a b : Ty) : Bool :=
  a.bits == b.bits &&
    (a.basic == b.basic || (a.isInt && b.isInt) || a.basic == .bits || b.basic == .bits)

def bitsOfWidth : Nat → Option Ty
  | 8 => some .b8 | 16 => some .b16 | 32 => some .b32 | 64 => some .b64 | 128 => some .b128
  | _ => none

end Ty

/-- State spaces, §5.1 Table 6, with the sub-spaces instructions name (`.shared::cta`,
`.shared::cluster`, `.param::entry`, `.param::func`). A bare `.shared` means `.shared::cta`
and a bare `.param` is resolved by context (§5.1.6), so both are kept as written. -/
inductive Space where
  | reg | sreg | const | global | local
  | param | paramEntry | paramFunc
  | shared | sharedCta | sharedCluster
  | tex
  deriving DecidableEq, Repr, Inhabited

instance : Keyword Space where
  all := [.reg, .sreg, .const, .global, .local, .param, .paramEntry, .paramFunc, .shared,
    .sharedCta, .sharedCluster, .tex]
  name
    | .reg => "reg" | .sreg => "sreg" | .const => "const" | .global => "global"
    | .local => "local" | .param => "param" | .paramEntry => "param::entry"
    | .paramFunc => "param::func" | .shared => "shared" | .sharedCta => "shared::cta"
    | .sharedCluster => "shared::cluster" | .tex => "tex"

theorem Space.complete (s : Space) : s ∈ (Keyword.all : List Space) := by cases s <;> decide

namespace Space

/-- `.shared` is `.shared::cta` (§5.1.7). -/
def normalize : Space → Space
  | .shared => .sharedCta
  | s => s

/-- Spaces whose variables live in memory and have addresses (§5.1 Table 7). -/
def addressable : Space → Bool
  | .reg | .sreg | .tex => false
  | _ => true

end Space

/-- Vector widths of variables and of `ld`/`st` data (§5.4.2, §9.7.9.8). -/
inductive Vec where
  | v2 | v4 | v8
  deriving DecidableEq, Repr, Inhabited

instance : Keyword Vec where
  all := [.v2, .v4, .v8]
  name | .v2 => "v2" | .v4 => "v4" | .v8 => "v8"

def Vec.length : Vec → Nat
  | .v2 => 2 | .v4 => 4 | .v8 => 8

end NV.PTX
