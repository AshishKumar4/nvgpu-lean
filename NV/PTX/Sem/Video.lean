import NV.PTX.Sem.State

/-!
# Video instructions

PTX ISA 9.4 §9.7.20, by its pseudocode. Scalar video instructions (§9.7.20.1) select a byte or
half-word of each 32-bit source (`.b0`…`.b3`, `.h0`, `.h1`; none: the word), extend it by the
source type, compute a 34-bit result, optionally clamp it to the destination's range (that of the
byte or half-word with a destination selector), then optionally apply a secondary operation with
`c` or merge into `c`. SIMD video instructions (§9.7.20.2) select lanes from the 64-bit `{b, a}`
(`.hxy`, `.bxyzw`), operate lane by lane, and merge with `c` under a mask or accumulate into
`c`.
-/

namespace NV.PTX.Sem.Video

/-- Bit offset and width a scalar selector names. -/
def scalarSel : Option String → Option (Nat × Nat)
  | none => some (0, 32)
  | some "b0" => some (0, 8) | some "b1" => some (8, 8) | some "b2" => some (16, 8)
  | some "b3" => some (24, 8) | some "h0" => some (0, 16) | some "h1" => some (16, 16)
  | some _ => none

def extract (signed : Bool) (sel : Nat × Nat) (x : UInt64) : Int :=
  toInt signed sel.2 (x >>> sel.1.toUInt64)

def clamp (signed : Bool) (w : Nat) (v : Int) : Int :=
  if signed then max (-(2 ^ (w - 1))) (min (2 ^ (w - 1) - 1) v) else max 0 (min (2 ^ w - 1) v)

/-- The primary operation of `vadd`…`vmax`, `vshl`, `vshr`, `vset` on extended operands. -/
def primary (op : Op) (cmp : Option CmpOp) (clampShift : Bool) (a b : Int) : Option Int :=
  match op with
  | .vadd => some (a + b)
  | .vsub => some (a - b)
  | .vabsdiff => some (a - b).natAbs
  | .vmin => some (min a b)
  | .vmax => some (max a b)
  | .vshl | .vshr =>
    let n := if clampShift then min b.toNat 32 else b.toNat % 32
    some (if op == .vshl then a * 2 ^ n else a / 2 ^ n)
  | .vset =>
    let c : Bool := match cmp with
      | some .eq => a == b | some .ne => a != b | some .lt => a < b | some .le => a ≤ b
      | some .gt => a > b | some .ge => a ≥ b | _ => false
    some (if c then 1 else 0)
  | _ => none

/-- `optSaturate`, `optSecOp`, `optMerge` of §9.7.20.1, and the low 32 bits. -/
def finish (tmp : Int) (sat dsigned : Bool) (dsel : Option (Nat × Nat)) (secop : Option RedOp)
    (c : UInt64) : UInt64 :=
  let w := (dsel.map (·.2)).getD 32
  let tmp := if sat then clamp dsigned w tmp else tmp
  let cv := toInt dsigned 32 c
  match dsel with
  | some (off, w) =>
    let m : UInt64 := ((1 <<< w.toUInt64) - 1) <<< off.toUInt64
    mask 32 ((ofInt 32 tmp <<< off.toUInt64) &&& m ||| (c &&& ~~~m))
  | none =>
    let r := match secop with
      | some .add => tmp + cv | some .min => min tmp cv | some .max => max tmp cv | _ => tmp
    ofInt 32 r

/-- `vmad` (§9.7.20.1.3): `±(a·b) ± c (+1)`, optionally shifted right by 7 or 15 and
saturated. -/
def vmad (sa sb : Bool) (a b : Int) (negAB negC po : Bool) (shift : Nat) (sat : Bool)
    (c : UInt64) : UInt64 :=
  let signedFinal := sa || sb || negAB || negC
  let p := a * b
  let cv : Int := if signedFinal then toInt true 32 c else (mask 32 c).toNat
  let m := 2 ^ 128
  let (p, cv, lsb) : Int × Int × Int :=
    if po then (p, cv, 1) else if negAB then (-p - 1, cv, 1) else if negC then (p, -cv - 1, 1)
    else (p, cv, 0)
  let tmp := (p + cv + lsb) % m
  -- the 128-bit sum as a signed or unsigned number, then the arithmetic shift
  let tmp : Int := if signedFinal && tmp ≥ m / 2 then tmp - m else tmp
  let r := tmp / 2 ^ shift
  ofInt 32 (if sat then clamp signedFinal 32 r else r)

/-- Lane `i`'s source position in `{b, a}` for a SIMD selector: `.hxy` (2 lanes) lists lane 1's
half-word then lane 0's; `.bxyzw` lane 3's byte first. -/
def simdSel (lanes : Nat) (sel : Option String) (default : List Nat) : Option (List Nat) :=
  match sel with
  | none => some default
  | some s =>
    let ds := (s.toList.drop 1).map fun c => c.toNat - '0'.toNat
    let pre := if lanes == 2 then 'h' else 'b'
    if s.toList.head? == some pre && ds.length == lanes && ds.all (· < 2 * lanes) then
      some ds.reverse
    else none

/-- The mask of lanes a SIMD destination mask names (`.h10`, `.b310`, …); all lanes by default. -/
def simdMask (lanes : Nat) (m : Option String) : Option (List Bool) :=
  match m with
  | none => some (List.replicate lanes true)
  | some s =>
    let pre := if lanes == 2 then 'h' else 'b'
    let ds := (s.toList.drop 1).map fun c => c.toNat - '0'.toNat
    if s.toList.head? == some pre && !ds.isEmpty && ds.all (· < lanes) then
      some ((List.range lanes).map ds.contains)
    else none

/-- A SIMD video instruction (§9.7.20.2): `vop2`/`vop4` and `vset2`/`vset4`. -/
def simd (op : Op) (lanes : Nat) (dsigned sa sb sat add : Bool) (cmp : Option CmpOp)
    (asel bsel : List Nat) (msk : List Bool) (a b c : UInt64) : Option UInt64 := do
  let w := 32 / lanes
  let pair := (mask 32 a) ||| (mask 32 b <<< 32)
  let lane (s : Bool) (src : UInt64) (k : Nat) : Int := toInt s w (src >>> (w * k).toUInt64)
  let isSet := op matches .vset2 | .vset4
  let f : Int → Int → Int ← match op with
    | .vadd2 | .vadd4 => some (· + ·)
    | .vsub2 | .vsub4 => some (· - ·)
    | .vavrg2 | .vavrg4 => some fun x y => if x + y ≥ 0 then (x + y + 1) / 2 else (x + y) / 2
    | .vabsdiff2 | .vabsdiff4 => some fun x y => (x - y).natAbs
    | .vmin2 | .vmin4 => some min
    | .vmax2 | .vmax4 => some max
    | .vset2 | .vset4 => some fun x y =>
      let r : Bool := match cmp with
        | some .eq => x == y | some .ne => x != y | some .lt => x < y | some .le => x ≤ y
        | some .gt => x > y | some .ge => x ≥ y | _ => false
      if r then 1 else 0
    | _ => none
  let dsigned := dsigned && !isSet
  let t (i : Nat) : Int :=
    let r := f (lane sa pair asel[i]!) (lane sb pair bsel[i]!)
    if sat && !isSet then clamp dsigned w r else r
  if add then
    return ofInt 32 ((List.range lanes).foldl (fun acc i => if msk[i]! then acc + t i else acc)
      (toInt dsigned 32 c))
  return (List.range lanes).foldl (fun d i =>
    let v := if msk[i]! then ofInt w (t i) else mask w (c >>> (w * i).toUInt64)
    d ||| (v <<< (w * i).toUInt64)) 0

end NV.PTX.Sem.Video
