import NV.Num.Tensor
import NV.Num.Block

/-!
# Block-scaled FP4 dot products (`tcgen05.mma .kind::mxf4`, `mxf4nvf4`)

SF Tensor's model of the B200's E2M1 block-scaled datapaths (`sf-tensor`, `MODEL.md` §5). Each
product is exact, scale factors included: `a · b · sa · sb`
at the sum of the four exponents, its raw exponent the sum of the four encoded exponents. Let `R`
be the largest raw exponent of a nonzero product, `S` the largest raw exponent of a nonzero
product's scale pair, and `C` the accumulator's raw exponent:

* when the accumulator is nonzero and `C ≥ R`, each K = 16 group's exact sum and the accumulator
  are truncated into the window `2^(C - 35)` and added;
* otherwise the window is `max(max(R, C) - 39, S - 35)`: the UE8M0 kinds truncate each K = 16
  group's exact sum into it (into `C - 35` instead for a group `C` outranks), the UE4M3 one each
  K = 4 group's; a subnormal result of the UE8M0 kinds drops the groups of a lower scale pair
  below `2^-174`.

The sum is truncated into binary32, a zero result `+0`. A NaN element or scale gives the canonical NaN, an infinite
accumulator itself; without a nonzero product the result is the accumulator (`+0` for `±0`).
-/

namespace NV.Num.Scaled

/-- A finite value `m · 2^e` (signed `m`) with its raw exponent. -/
structure Term where
  m : Int
  e : Int
  raw : Int
  deriving Inhabited

/-- `v` of format `f` as a term: its raw exponent is the encoded exponent, subnormals at the
minimum normal one. `none` for NaN and infinities. -/
def term (f : Format) (v : Val) : Option Term :=
  match v with
  | .fin neg m e => some { m := if neg then -(m : Int) else m, e, raw := e + f.man }
  | _ => none

/-- `m · 2^e` in units of `2^q`, truncated toward zero. -/
def trunc (m e q : Int) : Int :=
  if m == 0 then 0
  else if e ≥ q then m * (2 ^ (e - q).toNat : Nat)
  else if m < 0 then -((-m) / (2 ^ (q - e).toNat : Nat)) else m / (2 ^ (q - e).toNat : Nat)

/-- The exact sum of terms, at the smallest exponent among the nonzero ones. -/
def sumExact (ts : List (Int × Int)) : Int × Int :=
  match (ts.filter (·.1 != 0)).map (·.2) |>.min? with
  | none => (0, 0)
  | some e0 => (ts.foldl (fun s (m, e) => if m == 0 then s else s + m * (2 ^ (e - e0).toNat : Nat)) 0, e0)

/-- Units of `2^q` truncated into binary32 (`+0` for zero). -/
def toF32 (u q : Int) : Nat :=
  let r := if u == 0 then Format.f32.zero false else Format.f32.round .rz .inf (u < 0) u.natAbs q
  if r == Format.f32.zero true then Format.f32.zero false else r

/-- One of the datapaths: K values a scale factor covers, and whether it is the UE4M3 one. -/
structure Kind where
  scaleBlock : Nat
  nvfp4 : Bool
  deriving Inhabited

/-- `Σ a[k]·b[k]·sa[k / block]·sb[k / block] + c`: `a`, `b` E2M1 values, `sa`, `sb` scale factors
of format `sf`, `c` binary32 bits. -/
def dot (kd : Kind) (sf : Format) (a b sa sb : Array Val) (c : Nat) : Nat := Id.run do
  let nan := Format.f32.nan
  let cv := Format.f32.decode c
  -- Products, NaN first.
  let mut prods : Array (Int × Int) := #[]
  let mut raws : Array (Option Int) := #[]
  let mut scaleRaws : Array (Option Int) := #[]
  for k in [0:Min.min a.size b.size] do
    let blk := k / kd.scaleBlock
    match term .e2m1 a[k]!, term .e2m1 b[k]!, term sf (sa[blk]?.getD .nan), term sf (sb[blk]?.getD .nan) with
    | some x, some y, some s, some t =>
      let m := x.m * y.m * s.m * t.m
      prods := prods.push (m, x.e + y.e + s.e + t.e)
      raws := raws.push (if m == 0 then none else some (x.raw + y.raw + s.raw + t.raw))
      scaleRaws := scaleRaws.push (if m == 0 then none else some (s.raw + t.raw))
    | _, _, _, _ => return nan
  match cv with
  | .nan => return nan
  | .inf neg => return Format.f32.inf neg
  | _ => pure ()
  let some ct := term .f32 cv | return nan
  let some r := (raws.filterMap id).toList.max? | return (if ct.m == 0 then Format.f32.zero false else c)
  let s := ((scaleRaws.filterMap id).toList.max?).getD 0
  let groups (n : Nat) : List (List (Int × Int)) :=
    (List.range ((prods.size + n - 1) / n)).map fun g => (prods.extract (g * n) (g * n + n)).toList
  -- The accumulator outranks every product.
  if ct.m != 0 && ct.raw ≥ r then
    let q := ct.raw - 35
    let u := (groups 16).foldl (fun u g => let (m, e) := sumExact g; u + trunc m e q) (trunc ct.m ct.e q)
    return toF32 u q
  let q := Max.max ((if ct.m != 0 then Max.max r ct.raw else r) - 39) (s - 35)
  if kd.nvfp4 then
    let u := (groups 4).foldl (fun u g => let (m, e) := sumExact g; u + trunc m e q) (trunc ct.m ct.e q)
    return toF32 u q
  -- Per K = 16 group: its units, and the largest scale pair of its nonzero products.
  let blocks := (List.range ((prods.size + 15) / 16)).map fun g =>
    let ix := (List.range 16).map (16 * g + ·) |>.filter (· < prods.size)
    let (m, e) := sumExact (ix.map (prods[·]!))
    let gr := (ix.filterMap (raws[·]!)).max?
    let q' := match gr with
      | some gr => if ct.m != 0 && ct.raw ≥ gr then Max.max q (ct.raw - 35) else q
      | none => q
    ((trunc m e q') * (2 ^ (q' - q).toNat : Nat), (ix.filterMap (scaleRaws[·]!)).max?)
  let c0 := trunc ct.m ct.e q
  let u := blocks.foldl (fun u (b, _) => u + b) c0
  let out := toF32 u q
  if out % 2 ^ 31 == 0 || out % 2 ^ 31 ≥ 2 ^ 23 then return out
  -- A subnormal result: the groups of a lower scale pair below 2^-174 drop out.
  let threshold : Nat := if q < -174 then 2 ^ (-174 - q).toNat else 1
  let u := blocks.foldl (fun u (b, gs) =>
    match gs with
    | some gs => if gs < s && b.natAbs < threshold then u else u + b
    | none => u + b) c0
  return toF32 u q

end NV.Num.Scaled
