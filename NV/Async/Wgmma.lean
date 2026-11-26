import NV.Async.Tma
import Std.Data.HashMap
import Std.Data.HashSet

/-!
# Warpgroup MMA: operands and fragments

Where `wgmma.mma_async` (PTX ISA 9.5 §9.7.17.5) finds its operands and puts its result:

* shared-memory matrix descriptors (§9.7.17.5.1.2.2) and the canonical layouts they describe
  (§9.7.17.5.1.2.1.3, CuTe notation, normalized to 16-byte units), followed by the swizzle on the
  shared address (`Swizzle.apply`, the pattern TMA writes);
* the register fragments of A and of the accumulator D (§9.7.17.5.1.1, Figures 151–156), which
  are those of `mma.sync`'s m16n8k* fragments stacked over the warpgroup's four warps, warp `w`
  holding rows `16w … 16w + 15` (CUTLASS `cute/atom/mma_traits_sm90_gmma.hpp`, `ALayout_64x16`,
  `ALayout_64x8`, `ALayout_64x32`, `CLayout_64xN`).

So a warp's share of D depends only on its own rows of A and on all of B: each warp of the
warpgroup computes its 16 rows independently, which is how the SASS (`HGMMA`, one per warp) and
the PTX (one thread's fragment) semantics both use this module.
-/

namespace NV.Async.Wgmma

/-- Input element types (`.atype`/`.btype`). `.b1`'s operands are laid out as bytes of eight
elements (its `K` of 256 is 32 bytes of a row, as `.s8`'s 32), the unit of its descriptors and
fragments here; its product counts the bits set in both (`.and.popc`). -/
inductive InType where
  | f16 | bf16 | tf32 | e4m3 | e5m2 | s8 | u8 | b1
  /-- FP6 and FP4, `tcgen05.mma`'s alone (§9.7.18.10.4.2). -/
  | e2m3 | e3m2 | e2m1
  deriving DecidableEq, Repr, Inhabited

/-- The bits of the unit an operand is moved in: an element, a byte of `.b1` elements, or the
8-bit container of an FP6 or FP4 element. -/
def InType.bits : InType → Nat
  | .f16 | .bf16 => 16
  | .tf32 => 32
  | .e4m3 | .e5m2 | .s8 | .u8 | .b1 | .e2m3 | .e3m2 | .e2m1 => 8

/-- The bits of an element itself. -/
def InType.width : InType → Nat
  | .e2m3 | .e3m2 => 6
  | .e2m1 => 4
  | t => t.bits

/-- `K` of one instruction: 32 bytes of each row of A and column of B (§9.7.17.5.1). -/
def InType.k (t : InType) : Nat := 256 / t.bits

/-- Accumulator types (`.dtype`). -/
inductive Acc where
  | f16 | f32 | s32
  deriving DecidableEq, Repr, Inhabited

/-- A shared-memory matrix descriptor (§9.7.17.5.1.2.2): addresses and offsets in bytes. -/
structure Desc where
  start : Nat
  lbo : Nat
  sbo : Nat
  baseOffset : Nat
  /-- `none`, `b128`, `b64` or `b32`. -/
  swizzle : Swizzle
  deriving DecidableEq, Repr, Inhabited

def field (d : UInt64) (lo width : Nat) : Nat := (d >>> lo.toUInt64).toNat % 2 ^ width

def Desc.decode (d : UInt64) : Desc :=
  { start := field d 0 14 <<< 4, lbo := field d 16 14 <<< 4, sbo := field d 32 14 <<< 4,
    baseOffset := field d 49 3,
    swizzle := match field d 62 2 with | 0 => .none | 1 => .b128 | 2 => .b64 | _ => .b32 }

/-- The shared address of element `(mn, k)` of an operand: `mn` its row of A or column of B, `k`
its index along K within the instruction (`k < t.k`). `kMajor` is `imm-trans = 0`. A non-zero base
offset `b` names the 128-byte line of the swizzle's pattern the matrix starts at: the swizzle counts
lines from `b` lines before the address (a GH200 with the 128-byte swizzle: a matrix 128·j bytes
past a 1024-byte boundary with base offset j reads as one at the boundary with offset 0). -/
def Desc.addr (d : Desc) (t : InType) (kMajor : Bool) (mn k : Nat) : Except String Nat := do
  let esz := t.bits / 8
  let T := 16 / esz
  let W := d.swizzle.span
  let off :=
    if kMajor then
      if W == 0 then (mn % 8) * 16 + (mn / 8) * d.sbo + (k % T) * esz + (k / T) * d.lbo
      else (mn % 8) * W + (mn / 8) * d.sbo + k * esz
    else
      if W == 0 then (mn % T) * esz + (mn / T) * d.sbo + (k % 8) * 16 + (k / 8) * d.lbo
      else
        let w := W / 16
        (mn % T) * esz + ((mn / T) % w) * 16 + (mn / (T * w)) * d.lbo + (k % 8) * W + (k / 8) * d.sbo
  -- Shifting the address by whole lines moves only the swizzle's line index.
  let shift := 128 * ((8 - d.baseOffset % 8) % 8)
  return d.swizzle.apply (d.start + off + shift) - shift

/-- Row and column of value `v` of the accumulator fragment of lane `l` of warp `w` (0–3) in the
warpgroup: `d[v]` of an `.f32`/`.s32` accumulator; value `v` of an `.f16` one sits in the
`v % 2` half of register `v / 2`. -/
def accPos (w l v : Nat) : Nat × Nat :=
  (16 * w + l / 4 + 8 * ((v / 2) % 2), 2 * (l % 4) + v % 2 + 8 * (v / 4))

/-- Values per lane of the accumulator: `N / 2`. -/
def accCount (n : Nat) : Nat := n / 2

/-- Row and K index of value `v` of the A fragment (four registers) of lane `l` of warp `w`;
value `v` is element `v % (32 / bits)` of register `v / (32 / bits)`. -/
def aPos (t : InType) (w l v : Nat) : Nat × Nat :=
  let g := 16 * w + l / 4
  match t.bits with
  | 16 => (g + 8 * ((v / 2) % 2), 2 * (l % 4) + v % 2 + 8 * (v / 4))
  | 32 => (g + 8 * (v % 2), l % 4 + 4 * (v / 2))
  | _ => (g + 8 * ((v / 4) % 2), 4 * (l % 4) + v % 4 + 16 * (v / 8))

/-- Values per lane of the A fragment. -/
def aCount (t : InType) : Nat := 4 * (32 / t.bits)

/-- The 16-byte chunks (their addresses) holding the given elements' bytes. -/
def chunks (addrs : Array Nat) : Array Nat := Id.run do
  let mut seen : Std.HashSet Nat := {}
  let mut out := #[]
  for a in addrs do
    let c := a / 16 * 16
    if !seen.contains c then
      seen := seen.insert c
      out := out.push c
  return out

/-- Shared loads (async proxy, 16-byte lanes, 32 per request) of the given chunks; `addr` maps a
CTA-local shared address to the address the request names. -/
def loadReqs (cs : Array Nat) (addr : Nat → UInt64) : Array Req :=
  (Array.range ((cs.size + 31) / 32)).map fun i =>
    asyncReq .shared .ld 16 ((cs.extract (32 * i) (32 * i + 32)).map fun c => some (addr c))

/-- The shared bytes the loads of `cs` returned (their responses concatenated: four words per
chunk), by chunk address. -/
def image (cs : Array Nat) (data : Resp) : Std.HashMap Nat (Array UInt32) := Id.run do
  let mut m : Std.HashMap Nat (Array UInt32) := {}
  for i in [0:cs.size] do
    m := m.insert cs[i]! (data.extract (4 * i) (4 * i + 4))
  return m

/-- The `bits`-bit element at shared address `a` of an image. -/
def elem (img : Std.HashMap Nat (Array UInt32)) (bits a : Nat) : Nat :=
  let ws := img.getD (a / 16 * 16) #[0, 0, 0, 0]
  let off := a % 16
  let word := (ws[off / 4]?.getD 0).toNat
  if bits == 32 then word else (word >>> (8 * (off % 4))) % 2 ^ bits

end NV.Async.Wgmma
