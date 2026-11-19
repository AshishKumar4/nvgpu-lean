import NV.Async.Wgmma
import NV.Num

/-!
# Warpgroup MMA: one warp's share

`D = A·B + D` (or `A·B` when `scale-d` is false) for the sixteen rows of the warpgroup's
`m64nNkK` result that one warp holds (`NV.Async.Wgmma`), as one tile of the tensor-core
accumulation model (`NV.Num.TensorModel.tile`: each output element one fused block, Khattak and
Mikaitis's model
of the H100/H200 datapath): the instruction's K products and the accumulator, summed once and
rounded into the accumulator's format; instructions chain through the accumulator registers.
Integer MMAs are exact (`NV.Num.dotInt`).
-/

namespace NV.Async.Wgmma

open NV.Num

def InType.format : InType → Format
  | .f16 => Format.f16
  | .bf16 => Format.bf16
  | .tf32 => Format.tf32
  | .e4m3 => Format.e4m3
  | .e5m2 => Format.e5m2
  | .e2m3 => Format.e2m3
  | .e3m2 => Format.e3m2
  | .e2m1 => Format.e2m1
  | .s8 | .u8 | .b1 => Format.f32

def InType.isInt : InType → Bool
  | .s8 | .u8 | .b1 => true
  | _ => false

/-- The datapath for an input and accumulator type, `none` for pairs the ISA does not have. -/
def model : InType → Acc → Option TensorModel
  | .f16, .f32 | .bf16, .f32 => some Hopper.f16f32.value
  | .f16, .f16 => some Hopper.f16f16.value
  | .tf32, .f32 => some Hopper.tf32f32.value
  | .e4m3, .f32 | .e5m2, .f32 => some Hopper.f8f32.value
  | .e4m3, .f16 | .e5m2, .f16 => some Hopper.f8f16.value
  | _, _ => none

/-- Where A comes from: shared memory through a descriptor (`K`-major when `kMajor`), or the
warp's registers: per lane, the four words of its A fragment. -/
inductive ASource where
  | smem (d : Desc) (kMajor : Bool)
  | regs (frag : Array (Array UInt32))
  deriving Inhabited

/-- One warp's share of a `wgmma.mma_async`. -/
structure Share where
  n : Nat
  ty : InType
  /-- B's type, when it differs from A's (integer MMAs mix `.s8` and `.u8`). -/
  tyB : InType := ty
  acc : Acc
  /-- The warp's index in its warpgroup: it holds rows `16·warp …`. -/
  warp : Nat
  a : ASource
  b : Desc
  bKMajor : Bool
  /-- `imm-scale-a/b = -1`: the operand is negated. -/
  negA : Bool := false
  negB : Bool := false
  scaleD : Bool
  /-- The lanes whose results are computed: the warp's (SASS), or one thread's (PTX). -/
  lanes : Array Nat := Array.range 32

def accWords (acc : Acc) (n : Nat) : Nat := if acc == .f16 then n / 4 else n / 2

/-- Value `v` of an accumulator fragment's words, as a bit pattern. -/
def accGet (acc : Acc) (ws : Array UInt32) (v : Nat) : Nat :=
  if acc == .f16 then ((ws[v / 2]?.getD 0).toNat >>> (16 * (v % 2))) % 2 ^ 16
  else (ws[v]?.getD 0).toNat

/-- Element `v` of a lane's A registers. -/
def aRegGet (t : InType) (ws : Array UInt32) (v : Nat) : Nat :=
  let per := 32 / t.bits
  ((ws[v / per]?.getD 0).toNat >>> (t.bits * (v % per))) % 2 ^ t.bits

/-- `x`, negated when `on` (`imm-scale = -1`): its sign bit flipped. -/
def InType.negate (t : InType) (on : Bool) (x : Nat) : Nat := if on then x ^^^ 2 ^ (t.width - 1) else x

def signed8 (x : Nat) : Int := if x ≥ 128 then (x : Int) - 256 else x

/-- The bits set in a byte. -/
def popcount8 (x : Nat) : Nat := (List.range 8).foldl (fun n k => n + (x >>> k) % 2) 0

/-- What a share does: the shared loads its operands need (16-byte chunks, through `addr` from
CTA-local shared addresses to the addresses the requests name), and its result: given the loads'
responses concatenated in order, and the accumulator it starts from (`accWords` per lane of
`lanes`, in order), the new accumulator words of each of `lanes`. -/
structure Plan where
  loads : Array Req
  result : Resp → Array UInt32 → Array UInt32
  /-- `result` on the machine-word datapath, where it covers the share (`Mma.fast`). -/
  fast : Resp → Array UInt32 → Option (Array UInt32) := fun _ _ => none

/-- The rows of the warp's sixteen that the lanes' results need: all of them for a warp, two for a
thread. -/
def Share.rows (s : Share) : List Nat :=
  ((List.range 16).map (16 * s.warp + ·)).filter fun r => s.lanes.any (fun l => (r - 16 * s.warp) % 8 == l / 4)

/-- The share's operands from their elements: A's sixteen rows of the warp (`rows`' in shared
memory, at `aAddrs` row by row; or from the lanes' registers) and B's columns (at `bAddrs` column
by column), `K` elements each, negated as the instruction asks. `get a` is the element at shared
address `a`. -/
def Share.operands (s : Share) (rows : List Nat) (aAddrs bAddrs : Array Nat) (get : Nat → Nat) :
    Array (Array Nat) × Array (Array Nat) :=
  let t := s.ty
  let K := t.k
  let aRow : Array (Array Nat) := match s.a with
    | .smem _ _ => (Array.range 16).map fun r =>
      match rows.idxOf? (16 * s.warp + r) with
      | some i => (Array.range K).map fun k => t.negate s.negA (get aAddrs[i * K + k]!)
      | none => Array.replicate K 0
    | .regs frag => Id.run do
      let mut m := Array.replicate 16 (Array.replicate K 0)
      for l in [0:32] do
        for v in [0:aCount t] do
          let (r, k) := aPos t s.warp l v
          m := m.modify (r - 16 * s.warp) (·.set! k (t.negate s.negA (aRegGet t (frag[l]?.getD #[]) v)))
      return m
  let bCol (c : Nat) : Array Nat := (Array.range K).map fun k => t.negate s.negB (get bAddrs[c * K + k]!)
  (aRow, (Array.range s.n).map bCol)

/-- Where value `v` of lane `l`'s accumulator sits in the warp's 16 × N block, row-major: row
`l / 4 + 8 ((v / 2) mod 2)` of the warp's sixteen, column `2 (l mod 4) + v mod 2 + 8 (v / 4)`
(`accPos`). -/
def Share.accIndex (s : Share) (l v : Nat) : Nat :=
  let (r, c) := accPos s.warp l v
  (r - 16 * s.warp) * s.n + c

/-- The lanes' accumulator words `c` (`accWords` per lane of `lanes`, in order) as the warp's
16 × N block, row-major: position `(i, j)` holds value `j mod 2 + 2 (i / 8) + 4 (j / 8)` of lane
`4 (i mod 8) + (j mod 8) / 2` (`accIndex` inverted; `N` is a multiple of 8), and zero when
`scale-d` is false or that lane's result is not computed. -/
def Share.accIn (s : Share) (c : Array UInt32) : Array Nat :=
  let words := accWords s.acc s.n
  (Array.range (16 * s.n)).map fun x =>
    let i := x / s.n
    let j := x % s.n
    let l := 4 * (i % 8) + j % 8 / 2
    if s.scaleD && s.lanes.contains l then
      let k := s.lanes.idxOf l
      accGet s.acc (c.extract (k * words) (k * words + words)) (j % 2 + 2 * (i / 8) + 4 * (j / 8))
    else 0

/-- The product as the warp's 16 × N block, row-major, from A's sixteen rows, B's columns and the
accumulator block `cm`: the tensor-core model's tile, or for integer inputs the exact dot
products. -/
def Share.product (s : Share) (aRow bs : Array (Array Nat)) (cm : Array Nat) : Array Nat :=
  let t := s.ty
  match model t s.acc with
  | some m => m.tile t.format s.tyB.format (if s.acc == .f16 then Format.f16 else Format.f32) aRow bs cm
  | none => Id.run do
    let sg (ty : InType) (x : Nat) : Int := if ty == .s8 then signed8 x else x
    let mut out := cm
    for r in [0:16] do
      for c in [0:s.n] do
        let cv := cm[r * s.n + c]!
        let cInt : Int := if cv ≥ 2 ^ 31 then (cv : Int) - 2 ^ 32 else cv
        let d := if t == .b1 then
            -- `.and.popc`: the bits both bytes set, counted.
            cInt + ((aRow[r]!.zip bs[c]!).foldl (fun acc (u, v) => acc + popcount8 (u &&& v)) 0 : Nat)
          else dotInt (aRow[r]!.map (sg t)) (bs[c]!.map (sg s.tyB)) cInt false
        out := out.set! (r * s.n + c) (if d < 0 then d + 2 ^ 32 else d).toNat
    return out

/-- The lanes' accumulator words from the product block `dm`, lane by lane: value `v` of lane `l`
is `dm`'s element at `accIndex l v`; an `.f16` accumulator packs values `2 w` and `2 w + 1` into
word `w`, low half first. -/
def Share.pack (s : Share) (dm : Array Nat) : Array UInt32 :=
  s.lanes.flatMap fun l => (Array.range (accWords s.acc s.n)).map fun w =>
    if s.acc == .f16 then
      (dm[s.accIndex l (2 * w + 1)]! % 2 ^ 16 * 2 ^ 16 + dm[s.accIndex l (2 * w)]! % 2 ^ 16).toUInt32
    else dm[s.accIndex l w]!.toUInt32

/-- The share's result from its operands (A's sixteen rows, B's columns) and the lanes'
accumulator words `c`: the new accumulator words of each of `lanes`. -/
def Share.finish (s : Share) (aRow bs : Array (Array Nat)) (c : Array UInt32) : Array UInt32 :=
  s.pack (s.product aRow bs (s.accIn c))

/-- Where each element at the given shared addresses sits in the loads' responses (the chunks
`cs`, four words each, in order): its word's index and the shift to its bits. -/
def wordsAt (cs : Array Nat) (addrs : Array Nat) : Array UInt32 × Array UInt32 :=
  let ix : Std.HashMap Nat Nat := (cs.zipIdx).foldl (fun m (c, i) => m.insert c i) {}
  (addrs.map fun a => (4 * ix.getD (a / 16 * 16) 0 + a % 16 / 4).toUInt32,
   addrs.map fun a => (8 * (a % 4)).toUInt32)

/-- The fast result of a float share (`TensorModel.tileWords`): the same as `finish` on the
operands `operands` reads, with every position computed when the plan is made. `aw`/`ash` and
`bw`/`bsh` locate A's and B's elements in the responses (`wordsAt`); `accSrc[x]` is the word of
the lanes' accumulator holding block position `x` (`0` for none: then it starts at zero) plus
one, with `accHalf[x]` its half for an `.f16` accumulator; `packIdx` is `pack`'s block positions,
two a word for `.f16`. -/
structure Fast where
  aw : Array UInt32
  ash : Array UInt32
  bw : Array UInt32
  bsh : Array UInt32
  accSrc : Array UInt32
  accHalf : Array UInt32
  packIdx : Array UInt32

def Share.fast (s : Share) (cs aAddrs bAddrs : Array Nat) : Fast :=
  let (aw, ash) := wordsAt cs aAddrs
  let (bw, bsh) := wordsAt cs bAddrs
  let words := accWords s.acc s.n
  let half := s.acc == .f16
  let src := (Array.range (16 * s.n)).map fun x =>
    let i := x / s.n
    let j := x % s.n
    let l := 4 * (i % 8) + j % 8 / 2
    let v := j % 2 + 2 * (i / 8) + 4 * (j / 8)
    if s.scaleD then
      match s.lanes.idxOf? l with
      | some k => ((k * words + (if half then v / 2 else v) + 1).toUInt32, (v % 2).toUInt32)
      | none => (0, 0)
    else (0, 0)
  let packIdx := s.lanes.flatMap fun l => (Array.range words).flatMap fun w =>
    if half then #[(s.accIndex l (2 * w)).toUInt32, (s.accIndex l (2 * w + 1)).toUInt32]
    else #[(s.accIndex l w).toUInt32]
  { aw, ash, bw, bsh, accSrc := src.map (·.1), accHalf := src.map (·.2), packIdx }

/-- `finish` through `TensorModel.tileWords`, for a share whose datapath it covers: the lanes'
new accumulator words from the loads' responses `data` and their accumulator words `c`. -/
def Share.finishFast (s : Share) (m : TensorModel) (f : Fast) (rows : List Nat) (data : Resp)
    (c : Array UInt32) : Option (Array UInt32) :=
  let t := s.ty
  let K := t.k
  let bits := t.bits.toUInt32
  let mask : UInt32 := if bits == 32 then 0xFFFFFFFF else (1 <<< bits) - 1
  let signA : UInt32 := if s.negA then 1 <<< (bits - 1) else 0
  let signB : UInt32 := if s.negB then 1 <<< (bits - 1) else 0
  let get (w sh : UInt32) : UInt32 := ((data[w.toNat]?.getD 0) >>> sh) &&& mask
  let A : Array UInt32 := match s.a with
    | .smem _ _ => Id.run do
      let mut A := Array.replicate (16 * K) 0
      for (r, i) in rows.toArray.zipIdx do
        let base := (r - 16 * s.warp) * K
        for k in [0:K] do
          A := A.set! (base + k) (get f.aw[i * K + k]! f.ash[i * K + k]! ^^^ signA)
      return A
    | .regs frag => Id.run do
      let mut A := Array.replicate (16 * K) 0
      for l in [0:32] do
        for v in [0:aCount t] do
          let (r, k) := aPos t s.warp l v
          A := A.set! ((r - 16 * s.warp) * K + k) ((aRegGet t (frag[l]?.getD #[]) v).toUInt32 ^^^ signA)
      return A
  let B := (Array.range (s.n * K)).map fun x => get f.bw[x]! f.bsh[x]! ^^^ signB
  let half := s.acc == .f16
  let C : Array UInt64 := (Array.range (16 * s.n)).map fun x =>
    let w : UInt32 := f.accSrc[x]!
    if w == 0 then 0
    else
      let word : UInt32 := c[(w - 1).toNat]?.getD 0
      let h : UInt32 := f.accHalf[x]!
      let v : UInt32 := if half then (word >>> ((16 : UInt32) * h)) &&& (0xFFFF : UInt32) else word
      v.toUInt64
  let outF := if half then Format.f16 else Format.f32
  (m.tileWords t.format s.tyB.format outF 16 s.n K A B C).map fun (D : Array UInt64) =>
    if half then
      (Array.range (f.packIdx.size / 2)).map fun w =>
        let lo : UInt64 := D[f.packIdx[2 * w]!.toNat]!
        let hi : UInt64 := D[f.packIdx[2 * w + 1]!.toNat]!
        let v : UInt64 := (lo &&& (0xFFFF : UInt64)) ||| ((hi &&& (0xFFFF : UInt64)) <<< (16 : UInt64))
        v.toUInt32
    else f.packIdx.map fun (i : UInt32) => (D[i.toNat]! : UInt64).toUInt32

def Share.plan (s : Share) (addr : Nat → UInt64) : Except String Plan := do
  let t := s.ty
  let K := t.k
  if !t.isInt && (model t s.acc).isNone then throw s!"wgmma: {repr t} into {repr s.acc} is not an instruction"
  if t.isInt && s.acc != .s32 then throw "wgmma: integer inputs accumulate into .s32"
  let rows := s.rows
  let aAddrs ← match s.a with
    | .smem d km => (rows.toArray.flatMap fun r => (Array.range K).map fun k => d.addr t km r k).mapM id
    | .regs _ => pure #[]
  let bAddrs ← ((Array.range s.n).flatMap fun c => (Array.range K).map fun k => s.b.addr t s.bKMajor c k).mapM id
  let cs := chunks (aAddrs ++ bAddrs)
  let result (data : Resp) (c : Array UInt32) : Array UInt32 :=
    let img := image cs data
    let (aRow, bs) := s.operands rows aAddrs bAddrs (elem img t.bits)
    s.finish aRow bs c
  let fast := match model t s.acc with
    | some m => let f := s.fast cs aAddrs bAddrs; fun data c => s.finishFast m f rows data c
    | none => fun _ _ => none
  return { loads := loadReqs cs addr, result, fast }

/-- The MMA a share issues, starting from the accumulator words `c0` of its lanes. -/
def Plan.mma (pl : Plan) (s : Share) (c0 : Array UInt32) : Mma :=
  { loads := pl.loads, result := pl.result, fast := pl.fast, c0, macs := 16 * s.n * s.ty.k, n := s.n,
    inBits := s.ty.bits, aRegs := s.a matches .regs _ }

end NV.Async.Wgmma
