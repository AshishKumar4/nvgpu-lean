import NV.Memory.Store
import NV.Num.Arith
import NV.Async.MBarrierWord

/-!
# Performing one lane's access

The functional effect of an access on the store that holds its address: what a load returns,
what a store or an atomic leaves behind (PTX ISA §9.7.15.5). Floating-point `.add` rounds to
nearest even and `.min`/`.max` are the non-atomic instructions' (`NV.Num`); packed types act
lane by lane, and a vector atomic element by element. An mbarrier operation is
`NV.Async.MBarrierWord.apply` on the object's word.
-/

namespace NV

private def u64 (ws : Array UInt32) (i : Nat) : UInt64 :=
  ws[i]!.toUInt64 ||| (ws[i + 1]!.toUInt64 <<< 32)

private def ofU64 (v : UInt64) : Array UInt32 := #[v.toUInt32, (v >>> 32).toUInt32]

/-- The lowest index, a multiple of `n`, of `n` clear bits of the 32-bit `x` (`n` a power of two
from 1 to 32; `none` for any other `n` or when every such run has a bit set). -/
def findSetAlign (x : UInt32) (n : Nat) : Option Nat :=
  if n == 0 || n > 32 || n &&& (n - 1) != 0 then none
  else (List.range (32 / n)).findSome? fun k =>
    if (x.toNat >>> (k * n)) % 2 ^ n == 0 then some (k * n) else none

/-- The lowest index of `n` clear bits of the 32-bit `x`, at any position (`n` from 1 to 32; `none`
for any other `n` or when there is no such run). -/
def findSet (x : UInt32) (n : Nat) : Option Nat :=
  if n == 0 || n > 32 then none
  else (List.range (33 - n)).findSome? fun i =>
    if (x.toNat >>> i) % 2 ^ n == 0 then some i else none

/-- The new value of an atomic on `old` with operand `b` (and, for `cas`, the swap value `c`),
or `none` when the type is not yet supported. A vector acts on each element on its own. -/
def AtomOp.apply (op : AtomOp) (ty : AtomType) (old b c : Array UInt32) : Option (Array UInt32) :=
  match ty with
  | .b128 => match op with
    | .exch => some b
    | .cas => some (if old == b then c else old)
    | _ => none
  | .vec t n =>
    let w := t.words
    (List.range n).foldlM (fun acc k =>
      let part (a : Array UInt32) : Array UInt32 := a.extract (k * w) (k * w + w)
      (op.apply t (part old) (part b) (part c)).map (acc ++ ·)) #[]
  | .b32 | .u32 | .s32 =>
    let x := old[0]!; let y := b[0]!
    let r : Option UInt32 := match op with
      | .add => some (x + y)
      | .min => some (if ty == .s32 then (if x.toInt32 ≤ y.toInt32 then x else y) else Min.min x y)
      | .max => some (if ty == .s32 then (if x.toInt32 ≥ y.toInt32 then x else y) else Max.max x y)
      -- §9.7.13.5: inc wraps to 0 past `b`; dec wraps to `b` at 0 or above `b`.
      | .inc => some (if x ≥ y then 0 else x + 1)
      | .dec => some (if x == 0 || x > y then y else x - 1)
      | .and => some (x &&& y)
      | .or => some (x ||| y)
      | .xor => some (x ^^^ y)
      | .exch => some y
      | .cas => some (if x == y then c[0]! else x)
      | .findSetAlign => some ((_root_.NV.findSetAlign x y.toNat).elim x fun i => x ||| ((2 ^ y.toNat - 1) <<< i).toUInt32)
      | .findSet => some ((_root_.NV.findSet x y.toNat).elim x fun i => x ||| ((2 ^ y.toNat - 1) <<< i).toUInt32)
    r.map (#[·])
  | .b64 | .u64 | .s64 =>
    let x := u64 old 0; let y := u64 b 0
    let r : Option UInt64 := match op with
      | .add => some (x + y)
      | .min => some (if ty == .s64 then (if x.toInt64 ≤ y.toInt64 then x else y) else Min.min x y)
      | .max => some (if ty == .s64 then (if x.toInt64 ≥ y.toInt64 then x else y) else Max.max x y)
      | .and => some (x &&& y)
      | .or => some (x ||| y)
      | .xor => some (x ^^^ y)
      | .exch => some y
      | .cas => some (if x == y then u64 c 0 else x)
      | .inc | .dec | .findSetAlign | .findSet => none
    r.map ofU64
  | .f32 | .f32ftz | .f64 | .f16 | .bf16 | .f16x2 | .bf16x2 =>
    let (f, w, n) : NV.Num.Format × Nat × Nat := match ty with
      | .f64 => (.f64, 64, 1) | .f16 => (.f16, 16, 1) | .bf16 => (.bf16, 16, 1)
      | .f16x2 => (.f16, 16, 2) | .bf16x2 => (.bf16, 16, 2) | _ => (.f32, 32, 1)
    let ftz := ty == .f32ftz
    let word (ws : Array UInt32) : Nat :=
      if w == 64 then (u64 ws 0).toNat else ws[0]!.toNat
    let lane (v : Nat) (k : Nat) : Nat := (v >>> (w * k)) % 2 ^ w
    let x := word old
    let y := word b
    let f2 : Option (Nat → Nat → Nat) := match op with
      | .add => some (NV.Num.add f .rn ftz)
      | .min => some (NV.Num.minMax false f ftz false false)
      | .max => some (NV.Num.minMax true f ftz false false)
      | .exch => some fun _ y => y
      | _ => none
    let r : Option Nat := if op == .cas then some (if x == y then word c else x) else
      f2.map fun g => (List.range n).foldl (fun acc k => acc ||| (g (lane x k) (lane y k) <<< (w * k))) 0
    r.map fun v => if w == 64 then ofU64 v.toUInt64 else #[v.toUInt32 &&& (if w == 16 && n == 1 then 0xFFFF else 0xFFFFFFFF)]

/-- Perform lane `lane` of request `r` on store `s`: the new store and the lane's response words
(the loaded or old value; empty for stores and reductions). -/
def Store.perform (s : Store) (r : Req) (lane : Nat) : Except String (Store × Array UInt32) := do
  let some addr := r.addrs[lane]?.join | return (s, #[])
  unless s.contains addr r.bytes do
    throw s!"lane {lane}: {r.bytes}-byte access at {addr} outside {repr r.space}"
  let n := words r.bytes
  match r.op with
  | .ld => return (s, s.read addr r.bytes)
  | .st => return (s.write addr r.bytes r.data (lane * n), #[])
  | .atom op ty | .red op ty =>
    let old := s.read addr r.bytes
    let per := if op == .cas then 2 * n else n
    let b := r.data.extract (lane * per) (lane * per + n)
    let c := r.data.extract (lane * per + n) (lane * per + 2 * n)
    let some new := op.apply ty old b c | throw s!"atomic {repr op} on {repr ty} not supported yet"
    return (s.write addr r.bytes new 0, if r.op.returns then old else #[])
  | .mbar op =>
    unless r.bytes == 8 && addr % 8 == 0 do
      throw s!"lane {lane}: an mbarrier is an 8-byte aligned object, not {r.bytes} bytes at {addr}"
    let old := s.read addr 8
    let new ← (Async.MBarrierWord.apply op (u64 old 0)).mapError
      (s!"lane {lane}: {repr op} on the mbarrier at {addr} ({repr (Async.MBarrierWord.decode (u64 old 0))}): " ++ ·)
    return (s.write addr 8 (ofU64 new) 0, old)

end NV
