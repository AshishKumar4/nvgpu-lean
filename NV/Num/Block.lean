import NV.Num.Round

/-!
# Block-scaled formats

A block of `size` elements shares one scale (OCP MX v1.0 §5): MXFP8, MXFP6 and MXFP4 scale 32
elements by a UE8M0 power of two; NVFP4 scales 16 E2M1 elements by a UE4M3 value (PTX §9.7.16.3,
Table 45: `.kind::mxf4nvf4` with `.ue4m3` scales at `.scale_vec::4X`, four per 64 elements).
-/

namespace NV.Num

structure BlockFormat where
  elem : Format
  scale : Format
  size : Nat
  deriving DecidableEq, Repr, Inhabited

namespace BlockFormat

def mxfp8e4m3 : BlockFormat := { elem := .e4m3, scale := .ue8m0, size := 32 }
def mxfp8e5m2 : BlockFormat := { elem := .e5m2, scale := .ue8m0, size := 32 }
def mxfp6e3m2 : BlockFormat := { elem := .e3m2, scale := .ue8m0, size := 32 }
def mxfp6e2m3 : BlockFormat := { elem := .e2m3, scale := .ue8m0, size := 32 }
def mxfp4 : BlockFormat := { elem := .e2m1, scale := .ue8m0, size := 32 }
def nvfp4 : BlockFormat := { elem := .e2m1, scale := .ue4m3, size := 16 }

/-- An element's value under its block's scale (OCP MX §5.1): NaN under a NaN scale, the element's
own infinity or NaN, otherwise the exact product. -/
def value (b : BlockFormat) (s x : Nat) : Val :=
  match b.scale.decode s, b.elem.decode x with
  | .fin _ ms es, .fin n mx ex => .fin n (ms * mx) (es + ex)
  | .fin _ _ _, .inf n => .inf n
  | _, _ => .nan

/-- OCP MX §6.3's conversion of a block of finite values with a UE8M0 scale: the scale is the
largest power of two at most the largest magnitude, divided by the element format's largest power
of two; each element is its value over the scale rounded to nearest even, clamped to the largest
finite value. An all-zero block gets the scale `2^0`; a non-finite value makes the scale NaN (the
specification leaves both to the implementation). -/
def quantize (b : BlockFormat) (vs : Array Val) : Nat × Array Nat :=
  if vs.any (fun v => match v with | .fin .. => false | _ => true) then
    (b.scale.nan, vs.map fun _ => b.elem.zero false)
  else
    let lead (v : Val) : Option Int := match v with
      | .fin _ m e => if m = 0 then none else some (e + Nat.log2 m)
      | _ => none
    let top : Int := (b.elem.topFinite.1 : Int) - b.elem.bias
    let amax := (vs.filterMap lead).foldl (fun acc l => some (Max.max (acc.getD l) l)) none
    let x : Int := match amax with
      | some l => Max.max (-127) (Min.min 127 (l - top))
      | none => 0
    (b.scale.round .rz .sat false 1 x,
     vs.map fun v => match v with
      | .fin n m e => b.elem.round .rn .sat n m (e - x)
      | _ => b.elem.zero false)

end BlockFormat

end NV.Num
