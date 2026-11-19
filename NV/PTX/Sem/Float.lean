import NV.PTX.Sem.Operand
import NV.Num

/-!
# Floating-point instructions

PTX ISA 9.4 §9.7.3–§9.7.8 and the float forms of `cvt` (§9.7.10.24), on the encodings of
`NV.Num`, which defines the arithmetic, NVIDIA's flush-to-zero and NaN rules, and the `.approx`
functions. This module chooses the format, rounding and flags each instruction states and splits
packed types (`.f16x2`, `.bf16x2`, `.f32x2`, the packed 8-, 6- and 4-bit formats) into lanes.
An instruction without a rounding modifier rounds to nearest even (§9.7.3: the default
`.rn`).
-/

namespace NV.PTX.Sem.Float

open NV.Num (Format Rounding)

/-- The element format of a float type, and how many elements a packed type holds. -/
def lanes? : Ty → Option (Format × Nat)
  | .f16 => some (.f16, 1) | .bf16 => some (.bf16, 1) | .f32 => some (.f32, 1)
  | .f64 => some (.f64, 1) | .tf32 => some (.tf32, 1)
  | .e4m3 => some (.e4m3, 1) | .e5m2 => some (.e5m2, 1) | .e2m3 => some (.e2m3, 1)
  | .e3m2 => some (.e3m2, 1) | .e2m1 => some (.e2m1, 1) | .ue8m0 => some (.ue8m0, 1)
  | .ue4m3 => some (.ue4m3, 1) | .ue5m3 => some (.ue5m3, 1)
  | .f16x2 => some (.f16, 2) | .bf16x2 => some (.bf16, 2) | .f32x2 => some (.f32, 2)
  | .e4m3x2 => some (.e4m3, 2) | .e5m2x2 => some (.e5m2, 2) | .e2m3x2 => some (.e2m3, 2)
  | .e3m2x2 => some (.e3m2, 2) | .e2m1x2 => some (.e2m1, 2) | .ue8m0x2 => some (.ue8m0, 2)
  | .ue5m3x2 => some (.ue5m3, 2)
  | .e4m3x4 => some (.e4m3, 4) | .e5m2x4 => some (.e5m2, 4) | .e2m3x4 => some (.e2m3, 4)
  | .e3m2x4 => some (.e3m2, 4) | .e2m1x4 => some (.e2m1, 4)
  | _ => none

def isFloatTy (t : Ty) : Bool := (lanes? t).isSome

/-- Bits one lane occupies in a register: the format's width, or a byte for the 6-bit formats
(§5.2.3: "2 MSB bits padded with zeros"). -/
def laneBits (f : Format) (t : Ty) : Nat :=
  match t with
  | .e2m3x2 | .e3m2x2 | .e2m3x4 | .e3m2x4 => 8
  | _ => f.width

def rounding : Rnd → Option Rounding
  | .rn => some .rn | .rz => some .rz | .rm => some .rm | .rp => some .rp | .rna => some .rna
  | .rs => none

def intRounding : IntRnd → Rounding
  | .rni => .rn | .rzi => .rz | .rmi => .rm | .rpi => .rp

def cmpOp : CmpOp → Option NV.Num.CmpOp
  | .eq => some .eq | .ne => some .ne | .lt => some .lt | .le => some .le | .gt => some .gt
  | .ge => some .ge | .equ => some .equ | .neu => some .neu | .ltu => some .ltu
  | .leu => some .leu | .gtu => some .gtu | .geu => some .geu | .num => some .num
  | .nan => some .nan
  | _ => none

/-- Lane `k` of a packed value, `w` bits per lane, lane 0 in the low bits. -/
def lane (w k : Nat) (v : UInt64) : Nat := (mask w (v >>> (w * k).toUInt64)).toNat

def pack (w : Nat) (xs : List Nat) : UInt64 :=
  xs.zipIdx.foldl (fun acc (x, k) => acc ||| (mask w x.toUInt64 <<< (w * k).toUInt64)) 0

/-- Apply `op` lane by lane to the operands of a float type. -/
def map (t : Ty) (op : Format → List Nat → Nat) (args : List UInt64) : Option UInt64 := do
  let (f, n) ← lanes? t
  let w := laneBits f t
  return pack w ((List.range n).map fun k => op f (args.map (lane w k)))

/-- The value of a float constant (§4.5.2) in format `f`: `0f`/`0d` bits converted exactly
or rounded to nearest even, decimals rounded once to binary64 and then to `f`. -/
def const (f : Format) : Expr → Option Nat
  | .f32 b => some (NV.Num.cvt .f32 f .rn {} b.toNat)
  | .f64 b => some (NV.Num.cvt .f64 f .rn {} b.toNat)
  | .dec m e => some (NV.Num.cvt .f64 f .rn {} (NV.Num.ofDecimal .f64 m e))
  | .un .neg x => (const f x).map (NV.Num.neg f false)
  | .un .plus x => const f x
  | _ => none

end Float

namespace Thread

/-- Read an operand of float type `ty`: registers and exact-bit constants as they are, other
float constants converted (`Float.const`). -/
def readF (e : Env) (t : Thread) (a : Arg) (ty : Ty) : Except String UInt64 :=
  match a, Float.lanes? ty with
  | .imm x, some (f, 1) =>
    match x with
    | .f32 b => if ty == .f32 then pure b.toUInt64 else conv f x
    | .f64 b => if ty == .f64 then pure b else conv f x
    | _ => conv f x
  | a, _ => t.read e a ty
where
  conv (f : NV.Num.Format) (x : Expr) : Except String UInt64 :=
    match Float.const f x with
    | some v => pure v.toUInt64
    | none => throw s!"{repr x} is not a floating-point constant"

end Thread

end NV.PTX.Sem
