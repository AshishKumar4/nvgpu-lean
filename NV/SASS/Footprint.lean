import NV.SASS.Float
import NV.SASS.Async
import NV.PTX.Sem.Matrix

/-!
# Instruction footprints

The registers, uniform registers, predicates and convergence barriers an instruction reads, those
its writeback writes, and those it sets when it issues (convergence barriers), per register
(every lane). The control state (program counters, live and waiting lanes) is not a location:
every instruction reads and updates it at issue. The engine keeps each pending writeback's
write set and flags an instruction that issues while one of its reads or writes is pending (a
stale read or a lost write on silicon, Huerta et al. 2025 §4); the formal bridge from the engine
to the program semantics (`NV.Formal`) rests on instructions staying within their footprints.

A footprint is read off the instruction's operands, opcode and modifiers, the same facts the
semantics (`NV.SASS.Sem`) decodes. Where a form's operand roles are not known it is
conservative: every register operand read, four registers wide. `Tests/SassFootprint.lean`
checks every instruction of the test kernels against its semantics: perturbing state outside the
read set leaves the effect unchanged, and the writeback changes nothing outside the write set.
-/

namespace NV.SASS


structure Footprint where
  /-- Read at issue, or for a memory instruction when it reads its sources. -/
  reads : Array Loc := #[]
  /-- Written by the writeback (`Effect.reg`, or a memory instruction's response). -/
  writes : Array Loc := #[]
  /-- Changed at issue, in the state the effect continues with. -/
  sets : Array Loc := #[]
  deriving Repr, Inhabited, BEq

namespace Footprint

/-- What reading operand `o` as `k` consecutive registers reads. `wideAddr`: a memory operand's
register base is a 64-bit pair (`.64`, or `.E` on a global access). -/
def operand (o : Operand) (k : Nat := 1) (wideAddr : Bool := false) : Array Loc :=
  match o with
  | .reg g _ => Loc.span g k
  | .cbank _ (some g) _ => Loc.span g 1
  | .mem base wide ureg desc _ =>
    (match base with | some g => Loc.span g (if wide || wideAddr then 2 else 1) | none => #[]) ++
    (match ureg with | some g => Loc.span g 1 | none => #[]) ++
    (match desc with | some g => Loc.span g 2 | none => #[])
  | .gdesc g _ => Loc.span g 2
  | .udesc _ g _ => Loc.span g k
  | .neg o | .not o | .scaled o _ => operand o k wideAddr
  | _ => #[]

def dst (o : Operand) (k : Nat := 1) : Array Loc :=
  match o with
  | .reg g _ => Loc.span g k
  | _ => #[]

def dedup (ls : Array Loc) : Array Loc :=
  ls.foldl (fun acc l => if acc.contains l then acc else acc.push l) #[]

end Footprint

open Footprint

/-- The `mma.sync` (or `mma.sp`) a warp-level matrix instruction computes: its shape, for a
sparse one the shape whose A fragment holds the compressed A (`Matrix.sparseShape?`), and the A,
B, C and D element types. -/
structure MmaForm where
  x : NV.PTX.Sem.Matrix.Mma
  sparse : Option NV.PTX.Sem.Matrix.Mma := none
  a : NV.PTX.Ty
  b : NV.PTX.Ty
  c : NV.PTX.Ty
  d : NV.PTX.Ty
  /-- A block-scaled form's scale factors: their format and how many a row has. -/
  scaled : Option (NV.Num.Format × Nat) := none

/-- The form of `HMMA.16816.F32[.BF16]`, `HMMA.1688.F32.TF32`, `HMMA.16816.F16`,
`IMMA.{8816|16816|16832}.{S8|U8}.{S8|U8}`, `IMMA.{8832|16832|16864}.{S4|U4}.{S4|U4}`,
`BMMA.{88128|168128|168256}.{AND|XOR}.POPC`, `DMMA.{884|16x8x4|16x8x8|16x8x16}`,
`QMMA.16832.{F32|F16}.{E4M3|E5M2}.{E4M3|E5M2}` and, into F32, with FP6 and FP4 (`E2M1`, `E3M2`,
`E2M3`), the block-scaled `QMMA.SF.16832.F32.A.B.E8` (`.kind::mxf8f6f4`) and
`OMMA.SF.16864.F32.E2M1.E2M1.{E8|UE4M3}[.4X]` (`.kind::mxf4`, `mxf4nvf4`; `Rd, Ra, Rb, Rc, Rsa,
Rsb, URsel`), and the integer `IMMA.SP` (`Rd, Ra, Rb, Rc, Re, selector`: Re the metadata);
`none` for any other. -/
def mmaForm (i : Instr) : Option MmaForm := do
  let sp := i.hasMod "SP"
  let mods := i.mods.filter (fun m => m != "SP" && m != "SF")
  let (m, n, k) ← match mods[0]? with
    | some "16816" => some (16, 8, 16) | some "1688" => some (16, 8, 8)
    | some "16832" => some (16, 8, 32) | some "884" | some "8x8x4" => some (8, 8, 4)
    | some "1684" | some "16x8x4" => some (16, 8, 4) | some "16x8x8" => some (16, 8, 8)
    | some "16x8x16" => some (16, 8, 16) | some "8816" => some (8, 8, 16)
    | some "8832" => some (8, 8, 32) | some "16864" => some (16, 8, 64)
    | some "88128" => some (8, 8, 128) | some "168128" => some (16, 8, 128)
    | some "168256" => some (16, 8, 256) | _ => none
  let (a, b, c, d) : NV.PTX.Ty × NV.PTX.Ty × NV.PTX.Ty × NV.PTX.Ty ← match i.opcode with
    | "HMMA" =>
      let ab := if i.hasMod "BF16" then NV.PTX.Ty.bf16 else if i.hasMod "TF32" then .tf32 else .f16
      let acc := if i.hasMod "F32" then NV.PTX.Ty.f32 else .f16
      some (ab, ab, acc, acc)
    | "IMMA" =>
      let ty (s : String) : Option NV.PTX.Ty := match s with
        | "S8" => some .s8 | "U8" => some .u8 | "S4" => some .s4 | "U4" => some .u4 | _ => none
      some (← ty (mods[1]?.getD "S8"), ← ty (mods[2]?.getD "S8"), .s32, .s32)
    | "BMMA" => if i.hasMod "POPC" then some (.b1, .b1, .s32, .s32) else none
    | "DMMA" => some (.f64, .f64, .f64, .f64)
    | "QMMA" | "OMMA" =>
      -- FP8, FP6 and FP4 inputs, the A and B formats after the accumulator's (sm_120a).
      let ty (s : String) : Option NV.PTX.Ty := match s with
        | "E4M3" => some .e4m3 | "E5M2" => some .e5m2 | "E2M1" => some .e2m1
        | "E3M2" => some .e3m2 | "E2M3" => some .e2m3 | _ => none
      let acc := if i.hasMod "F32" then NV.PTX.Ty.f32 else .f16
      some (← ty (mods[2]?.getD ""), ← ty (mods[3]?.getD ""), acc, acc)
    | _ => none
  let scaled : Option (NV.Num.Format × Nat) ←
    if !i.hasMod "SF" then pure none
    else if i.hasMod "E8" then pure (some (.ue8m0, if i.hasMod "4X" then 4 else if i.opcode == "OMMA" then 2 else 1))
    else if i.hasMod "UE4M3" then pure (some (.ue4m3, if i.hasMod "4X" then 4 else 2))
    else none
  if sp then
    let (x, xa) ← NV.PTX.Sem.Matrix.sparseShape? m n k a d
    return { x, sparse := some xa, a, b, c, d }
  let x ← NV.PTX.Sem.Matrix.shape? m n k a d
  return { x := { x with xor := i.hasMod "XOR" }, a, b, c, d, scaled }

/-- A texture or surface instruction's form: its dimensions (a `1D`/`2D`/`3D` or
`ARRAY_1D`/`ARRAY_2D` operand, or such a modifier, `1D_ARRAY`/`2D_ARRAY` for surfaces; `CUBE`,
`ARRAY_CUBE`: a direction of three), whether it addresses a layered object and whether a
cubemap, the component mask (the immediate after the dimension operand; 15
without one) and the words a surface access moves (`.64`, `.128`). -/
structure TexForm where
  dims : Option Nat
  array : Bool
  cube : Bool
  mask : Nat
  words : Nat

def texForm (i : Instr) : TexForm :=
  let names := ["1D", "2D", "3D", "ARRAY_1D", "ARRAY_2D", "1D_ARRAY", "2D_ARRAY", "CUBE",
    "ARRAY_CUBE"]
  let dimOf (s : String) : Option (Nat × Bool) := match s with
    | "1D" => some (1, false) | "2D" => some (2, false) | "3D" | "CUBE" => some (3, false)
    | "ARRAY_1D" | "1D_ARRAY" => some (1, true) | "ARRAY_2D" | "2D_ARRAY" => some (2, true)
    | "ARRAY_CUBE" => some (3, true)
    | _ => none
  let g := (i.operands.findSome? fun | .name n => dimOf n | _ => none) <|>
    (i.mods.toList.filter names.contains).head?.bind dimOf
  { dims := g.map (·.1), array := (g.map (·.2)).getD false
    cube := i.operands.any (· matches .name "CUBE" | .name "ARRAY_CUBE")
    mask := match i.operands.back? with
      | some (.imm m) => if (i.operands.findIdx? (· matches .name _)).isSome then m.toNat else 15
      | _ => 15
    words := if i.hasMod "128" then 4 else if i.hasMod "64" then 2 else 1 }

/-- The element type and bytes of an atomic (`ATOMG`, `ATOMS`, `ATOM`, `REDG`, `RED`): `.F32.FTZ`
flushes subnormals, `.F32` keeps them; the vector
forms (PTX's vector `atom`/`red`, sm_90) `.F32x2`/`.F32x4`, `.F16x4`/`.F16x8`, `.BF16x4`/`.BF16x8`
are `AtomType.vec`s of 32-bit elements; `.128` is `.b128` `exch`/`cas`. -/
def atomForm (i : Instr) : AtomType × Nat :=
  let wide := i.hasMod "64" || i.hasMod "U64" || i.hasMod "S64" || i.hasMod "F64"
  let vec : Option (AtomType × Nat) := i.mods.findSome? fun
    | "F32x2" => some (.f32, 8) | "F32x4" => some (.f32, 16) | "F16x4" => some (.f16x2, 8)
    | "F16x8" => some (.f16x2, 16) | "BF16x4" => some (.bf16x2, 8) | "BF16x8" => some (.bf16x2, 16)
    | "128" => some (.b128, 16) | _ => none
  let ty := if let some (v, b) := vec then
      let e := if v == .f32 && i.hasMod "FTZ" then .f32ftz else v
      if e == .b128 then e else .vec e (b / 4 / e.words)
    else if i.hasMod "F32" then (if i.hasMod "FTZ" then AtomType.f32ftz else .f32)
    else if i.hasMod "F64" then .f64 else if i.hasMod "F16x2" then .f16x2
    else if i.hasMod "BF16x2" then .bf16x2 else if i.hasMod "S32" then .s32
    else if i.hasMod "U64" then .u64 else if i.hasMod "S64" then .s64
    else if wide then .b64 else .u32
  (ty, (vec.map (·.2)).getD (if wide then 8 else 4))

/-- The `ldmatrix`/`stmatrix` an `LDSM`/`STSM` performs: `LDSM.16.{M88|MT88}`,
`LDSM.8.MT1616`, `LDSM.{U4x16P64|U6x16P32}TO8.{M816|MT1616}`, `STSM.16.{M88|MT88}`,
`STSM.8.MT168`, each with `.2`/`.4` for two or four matrices. -/
def matMove (i : Instr) : Option NV.PTX.Sem.Matrix.MatMove := do
  let num := if i.hasMod "4" then 4 else if i.hasMod "2" then 2 else 1
  let src ← if i.hasMod "16" then some 16 else if i.hasMod "8" then some 8
    else if i.hasMod "U4x16P64TO8" then some 4 else if i.hasMod "U6x16P32TO8" then some 6 else none
  let (m, n, trans) ← if i.hasMod "M88" then some (8, 8, false) else if i.hasMod "MT88" then some (8, 8, true)
    else if i.hasMod "MT1616" then some (16, 16, true) else if i.hasMod "M816" then some (8, 16, false)
    else if i.hasMod "MT168" then some (16, 8, true) else none
  return { m, n, trans, src, num }

/-- The footprint of instruction `i` by its form, before the guard is added and the sets are
kept to convergence barriers (`footprint`). -/
def formFootprint (i : Instr) : Footprint :=
  let o (n : Nat) : Operand := i.operands[n]?.getD (.name "?")
  let nOps := i.operands.size
  let isPredOp := Exec.isPred
  -- The leading predicate operands from index `n` (carry outs, second destinations).
  let nP (n : Nat) := ((i.operands.toList.drop n).takeWhile isPredOp).length
  -- Reads of operands `ns`, one register each unless `w` says otherwise.
  let rd (ns : List Nat) (w : Nat → Nat := fun _ => 1) : Array Loc :=
    ns.foldl (fun acc n => acc ++ operand (o n) (w n)) #[]
  let rest (from_ : Nat) : List Nat := (List.range nOps).filter (· ≥ from_)
  -- Uniform-datapath forms (`UIADD3`) have their vector form's operand roles.
  let op := if i.opcode.startsWith "U" && i.opcode != "USETMAXREG" then (i.opcode.drop 1).toString
    else i.opcode
  let has := i.hasMod
  let (bytes, _) := Exec.width i
  let words := (bytes + 3) / 4
  -- One destination then sources.
  let simple (dw : Nat := 1) (sw : Nat → Nat := fun _ => 1) : Footprint :=
    { reads := rd (rest 1) sw, writes := dst (o 0) dw }
  -- Destinations at `0` and the `k` predicates after it, then sources.
  let withPreds (k : Nat) (dw : Nat := 1) (sw : Nat → Nat := fun _ => 1) : Footprint :=
    { reads := rd (rest (1 + k)) sw,
      writes := dst (o 0) dw ++ (List.range k).foldl (fun acc j => acc ++ dst (o (1 + j))) #[] }
  -- Two destinations then sources.
  let two (sw : Nat → Nat := fun _ => 1) : Footprint :=
    { reads := rd (rest 2) sw, writes := dst (o 0) ++ dst (o 1) }
  let conservative : Footprint :=
    { reads := rd (List.range nOps) (fun _ => 4), writes := dst (o 0) 4 }
  let allP : Array Loc := (Array.range 7).map Loc.p
  -- The predicates under a mask: the uniform ones for `UP2UR`.
  let uniformPreds := i.opcode == "UP2UR"
  let underMask : Operand → Array Loc
    | .imm m => (Array.range 7).filter (fun b => (m >>> b.toUInt64) &&& 1 == 1) |>.map
        (if uniformPreds then Loc.up else Loc.p)
    | _ => if uniformPreds then (Array.range 7).map Loc.up else allP
  match op with
    | "IMAD" =>
      if has "WIDE" then
        let n := nP 1
        withPreds n 2 fun k => if k == 3 + n then 2 else 1
      else if has "HI" then withPreds (nP 1) 1 fun k => if k == 3 + nP 1 then 2 else 1
      else withPreds (nP 1)
    | "IADD3" => if has "64" then simple 2 (fun _ => 2) else withPreds (nP 1)
    | "LEA" => withPreds (nP 1)
    | "LOP3" => if isPredOp (o 0) then two else simple
    | "ISETP" => two fun k => if (k == 2 || k == 3) && (has "U64" || has "S64") then 2 else 1
    | "PLOP3" | "FSETP" | "HSETP2" | "PSETP" => two
    | "DSETP" => two fun k => if k == 2 || k == 3 then 2 else 1
    | "SHFL" => two
    | "VOTE" | "VOTEU" =>
      if nOps == 3 then two else { reads := rd (rest 1), writes := dst (o 0) }
    | "MATCH" =>
      let w := if has "U64" then 2 else 1
      if isPredOp (o 0) then { reads := rd [nOps - 1] (fun _ => w), writes := dst (o 0) ++ dst (o 1) }
      else { reads := rd [nOps - 1] (fun _ => w), writes := dst (o 0) }
    | "ELECT" => two
    -- `P2R Rd, PR, Ra, mask` reads the predicates under the mask; `R2P PR, Ra, mask` writes them.
    | "P2R" | "P2UR" => { reads := rd (rest 2) ++ underMask (o 3), writes := dst (o 0) }
    | "R2P" => { reads := rd [1, 2], writes := underMask (o 2) }
    | "CS2R" | "CS2UR" => if has "32" then simple else simple 2
    | "LDCU" | "LDC" =>
      { reads := rd (rest 1), writes := dst (o 0) (if has "128" then 4 else if has "64" then 2 else 1) }
    | "IADD" => simple 2 (fun _ => 2)
    | "DADD" | "DMUL" | "DFMA" => simple 2 (fun _ => 2)
    -- A `.F32x2` source is a register pair, a `.F32` one a register.
    | "FADD2" | "FMUL2" | "FFMA2" => simple 2 fun n => match o n with
      | .reg _ m => if m.suffix.contains "F32x2" then 2 else 1
      | _ => 1
    | "FRND" => if has "F64" then simple 2 (fun _ => 2) else simple
    | "I2F" | "I2FP" =>
      let dw := if (i.mods.findSome? Float.format) == some .f64 then 2 else 1
      let sw := if ((i.mods.findSome? Float.intType).map (·.2)) == some 64 then 2 else 1
      simple dw (fun _ => sw)
    | "F2I" =>
      let dw := if ((i.mods.findSome? Float.intType).map (·.2)) == some 64 then 2 else 1
      simple dw (fun _ => if has "F64" then 2 else 1)
    | "F2F" =>
      let fs := (i.mods.filterMap Float.format).toList
      simple (if fs[0]? == some .f64 then 2 else 1) (fun _ => if fs[1]? == some .f64 then 2 else 1)
    | "HMMA" | "IMMA" | "BMMA" | "DMMA" | "QMMA" | "OMMA" =>
      match mmaForm i with
      | some f =>
        { reads := operand (o 1) (f.sparse.getD f.x).aWords ++ operand (o 2) f.x.bWords ++
            operand (o 3) f.x.cWords ++ (if f.sparse.isSome then operand (o 4) else #[]) ++
            (if f.scaled.isSome then operand (o 4) ++ operand (o 5) ++ operand (o 6) else #[]),
          writes := dst (o 0) f.x.cWords }
      | none => conservative
    | "QSPC" => { reads := operand (o 2) 1 (has "E"), writes := dst (o 0) ++ dst (o 1) }
    -- `.ENL2.256` (sm_100a): two register quads, the high one first for a load (`decodeMem`).
    | "LDG" =>
      if has "ENL2" then
        { reads := operand (o 2) 1 (has "E"), writes := dst (o 0) 4 ++ dst (o 1) 4 }
      else { reads := operand (o 1) 1 (has "E"), writes := dst (o 0) words }
    | "LDS" | "LDL" | "LD" | "LDGMC" =>
      { reads := operand (o 1) 1 ((op == "LD" || op == "LDGMC") && has "E"),
        writes := dst (o 0) words }
    | "STG" =>
      if has "ENL2" then
        { reads := operand (o 0) 1 (has "E") ++ operand (o 1) 4 ++ operand (o 2) 4 }
      else { reads := operand (o 0) 1 (has "E") ++ operand (o 1) words }
    | "STS" | "STL" | "ST" =>
      { reads := operand (o 0) 1 (op == "ST" && has "E") ++ operand (o 1) words }
    | "STAS" | "REDAS" => { reads := operand (o 0) ++ operand (o 1) words }
    | "MOVM" => if has "U4TO8" then simple 2 else simple
    -- Textures (`decodeTex`): coordinates from operand 2 (three registers) and 3 (two), the
    -- handle's uniform register; the mask's components into operand 1's pair, then operand 0's.
    | "TEX" | "TLD" | "TLD4" | "TXD" | "TXQ" =>
      let tf := texForm i
      let mask := tf.mask
      let c := (List.range 4).filter (fun b => (mask >>> b) % 2 == 1) |>.length
      let urs := i.operands.foldl (fun acc x => match x with
        | .reg ⟨.ur, _⟩ _ => acc ++ operand x | _ => acc) #[]
      -- The coordinates from Ra: an array's layer, then up to three (`ARRAY_CUBE`: four).
      let coords := max 3 (tf.dims.getD 3 + if tf.array then 1 else 0)
      -- Rb: a `TEX` without `.SCR` holds there just its explicit LOD (`.LL`) and offsets
      -- (`.AOFFI`), one register each (`NV.SASS.Sem`'s texture fetch); otherwise up to two are taken.
      let rb := if op == "TXD" then 4
        else if op == "TEX" && !i.hasMod "SCR" then
          (if i.hasMod "LL" then 1 else 0) + (if i.hasMod "AOFFI" then 1 else 0)
        else 2
      { reads := operand (o 2) (if op == "TXQ" then 1 else coords) ++
          (if op == "TXQ" || rb == 0 then #[] else operand (o 3) rb) ++ urs,
        writes := dst (o 1) (min c 2) ++ dst (o 0) (c - min c 2) }
    | "SULD" | "SUST" | "SURED" =>
      let urs := i.operands.foldl (fun acc x => match x with
        | .reg ⟨.ur, _⟩ _ => acc ++ operand x | _ => acc) #[]
      let tf := texForm i
      let coords (x : Operand) : Array Loc := match x with
        | .mem (some g) _ _ _ _ => Loc.span g (tf.dims.getD 1 + if tf.array then 1 else 0)
        | _ => #[]
      let w := tf.words
      if op == "SULD" then { reads := coords (o 1) ++ urs, writes := dst (o 0) w }
      else { reads := coords (o 0) ++ operand (o 1) w ++ urs }
    | "ATOMG" | "ATOMS" | "RED" | "REDG" | "ATOM" =>
      let red := op.startsWith "RED"
      let isAddrOp : Operand → Bool := fun | .mem .. | .scaled .. => true | _ => false
      let a0 := ((List.range nOps).find? (isAddrOp ∘ o)).getD (if red then 0 else 1)
      let k := (atomForm i).2 / 4
      let global := op != "ATOMS"
      { reads := operand (o a0) 1 (global && has "E") ++ rd ((rest (a0 + 1))) (fun _ => k),
        writes := if red then #[] else
          (if a0 == 2 then dst (o 0) else #[]) ++ dst (o (a0 - 1)) (if has "CAST" then 1 else k) }
    | "LDSM" | "STSM" =>
      match matMove i with
      | some x =>
        if op == "LDSM" then { reads := operand (o 1), writes := dst (o 0) (x.num * x.regs) }
        else { reads := operand (o 0) ++ operand (o 1) (x.num * x.regs) }
      | none => conservative
    | "BSSY" => { sets := dst (o 0) }
    | "BREAK" => { reads := rd [0], sets := dst (o 0) }
    | "BSYNC" => { reads := rd [0] }
    | "BMOV" =>
      -- `BMOV.32 Bx, Ra` sets the barrier at issue; `BMOV.32[.CLEAR] Rd, Bx` writes Rd back and
      -- clears Bx at issue.
      match o 0 with
      | .reg ⟨.b, _⟩ _ => { reads := rd [1], sets := dst (o 0) }
      | _ => { reads := rd [1], writes := dst (o 0), sets := if has "CLEAR" then dst (o 1) else #[] }
    | "BRX" => { reads := operand (o 0) 2 }
    | "BRXU" => { reads := operand (o 0) 2 }
    | "ACQBULK" | "PREEXIT" => {}
    | "LEPC" => { writes := dst (o 0) 2 }
    | "CALL" =>
      -- `CALL.ABS Rn` reads its target pair; a call to `vprintf` reads its arguments' addresses
      -- in R4–R7, returns in R4 and appends to the warp's `printf` records; a call to
      -- `cudaGraphSetConditional` reads its handle and value in R4–R6 (within the same span).
      match o 0 with
      | .reg ⟨.r, n⟩ _ =>
        if has "REL" then { reads := Loc.span ⟨.r, n⟩ 2 }
        else { reads := Loc.span ⟨.r, n⟩ 2 ++ Loc.span ⟨.r, 4⟩ 4, writes := #[.r 4, .log] }
      | _ => { reads := rd (List.range nOps) }
    | "BAR" => if has "RED" then { reads := rd (List.range nOps), writes := #[.bar] }
      else { reads := rd (List.range nOps) }
    | "B2R" => { reads := #[.bar], writes := dst (o 0) ++ (if nOps > 1 then dst (o 1) else #[]) }
    | "BRA" | "WARPSYNC" | "ENDCOLLECTIVE" | "RET" | "EXIT" | "NOP" | "DEPBAR" | "NANOSLEEP" | "YIELD" | "PMTRIG"
    | "BPT" | "MEMBAR" | "FENCE" | "ERRBAR" | "CGAERRBAR" | "CCTL" =>
      { reads := rd (List.range nOps) }
    | "IMNMX" =>
      if has "S64" || has "U64" then
        { reads := operand (o 3) 2 ++ operand (o 4) 2 ++ rd [5, 6], writes := dst (o 2) 2 }
      else simple
    | "MOV" | "SEL" => if has "64" then simple 2 (fun n => if n == 1 || n == 2 then 2 else 1) else simple
    | "VIMNMX" | "VIMNMX3" | "VIADDMNMX" | "SGXT" | "BMSK" | "IDP" | "VABSDIFF"
    | "VABSDIFF4" | "I2I" | "I2IP" | "F2IP" | "HSET2" | "PRMT" | "POPC" | "BREV" | "FLO" | "IABS"
    | "S2R" | "S2UR"
    | "VIADD" | "SHF" | "REDUX" | "CREDUX" | "FADD" | "FMUL" | "FFMA" | "FMNMX" | "FMNMX3" | "FCHK"
    | "FSEL" | "FSET"
    | "HADD2" | "HMUL2" | "HFMA2" | "HMNMX2" | "F2FP" | "MUFU" => simple
    | "R2UR" =>
      match o 0 with
      | .reg ⟨.p, _⟩ _ =>
        { reads := rd [2] ++ (if has "OR" then dst (o 0) else #[]), writes := dst (o 0) ++ dst (o 1) }
      | _ => simple
    | "USETMAXREG" => if has "TRY_ALLOC" then { writes := dst (o 0) } else {}
    -- The asynchronous opcodes (`NV.SASS.Async`).
    | "SYNCS" =>
      if has "ARRIVE" then { reads := rd (rest 1), writes := dst (o 0) 2 }
      else if has "PHASECHK" then { reads := rd (rest 1), writes := dst (o 0) }
      else if has "EXCH" then { reads := operand (o 1) ++ operand (o 2) 2 }
      else { reads := rd (List.range nOps) }
    | "ARRIVES" | "LDGDEPBAR" | "TMACMDFLUSH" | "TMAPF" | "TMACCTL" | "WARPGROUP" | "CGABAR_ARV"
    | "CGABAR_WAIT" =>
      { reads := rd (List.range nOps) }
    | "LDGSTS" => { reads := operand (o 0) ++ operand (o 1) 1 (has "E") ++ rd (rest 2) }
    | "TMALDG" | "TMASTG" | "TMAREDG" | "BLKCP" | "BLKRED" =>
      -- What `NV.SASS.Async` reads: `UTMALDG.nD [URa], [URb][, URc]` the shared address, the
      -- mbarrier and n coordinates from URa, `UTMASTG`/`UTMAREDG` the shared address and the
      -- coordinates, the tensor map's address pair from URb; `UBLKCP`/`UBLKRED [URa], [URb], URc`
      -- pairs and a size. A further operand (im2col offsets, a multicast mask) is one register.
      -- `UTMALDG.2D.GATHER4` and `UTMASTG.2D.SCATTER4` read five coordinates, the column and
      -- four rows.
      let first : Nat := match op, AsyncSem.dims i with
        | "TMALDG", some n => 2 + (if has "GATHER4" then 5 else n)
        | "TMASTG", some n | "TMAREDG", some n => 1 + (if has "SCATTER4" then 5 else n)
        | "TMALDG", none | "TMASTG", none | "TMAREDG", none => 7
        | _, _ => 2
      { reads := (List.range nOps).foldl (fun acc n => acc ++ match o n with
          | .mem (some g) _ _ _ _ => Loc.span g (if n == 0 then first else 2)
          | .reg g _ => Loc.span g 1
          | _ => #[]) #[] }
    | "BLKPF" =>
      -- A prefetch, a no-op here: every register from each base it might read.
      { reads := (List.range nOps).foldl (fun acc n => acc ++ match o n with
          | .mem (some g) _ _ _ _ | .reg g _ => Loc.span g 7
          | _ => #[]) #[] }
    -- `tcgen05` (`NV.SASS.Async`): the MMAs, barrier and copies read their uniform operands, a
    -- block-scaled MMA's scale factors' pair (its sixth) whole.
    | "TCHMMA" | "TCQMMA" | "TCIMMA" | "TCOMMA" | "TCBAR" | "TCCP" | "TCSHIFT" | "VIRTCOUNT" =>
      { reads := rd (List.range nOps) (fun n => if op.endsWith "MMA" && n == 5 then 2 else 1) }
    | "TCATOMSWS" =>
      if isPredOp (o 0) then { reads := rd (rest 2), writes := dst (o 0) ++ dst (o 1) } else simple
    | "LDTM" => { reads := rd (rest 1), writes := dst (o 0) (tmemRegs i) }
    | "STTM" => { reads := operand (o 0) ++ operand (o 1) (tmemRegs i) }
    | "HGMMA" | "QGMMA" | "IGMMA" | "BGMMA" =>
      -- `HGMMA.64xNxK… Rd, [Ra,] gdesc[URd], Rc[, UPp]`: the accumulator's words, A from four
      -- registers or a descriptor pair, B's descriptor pair after A's.
      let n := ((i.mods[0]?.map (·.splitOn "x")).bind (·[1]?)).bind String.toNat? |>.getD 256
      let words := match gmmaTypes i with
        | some (_, _, acc) => NV.Async.Wgmma.accWords acc n
        | none => 128
      let rs := match o 1 with | .reg ⟨.r, _⟩ _ => true | _ => false
      let gi := if rs then 2 else 1
      { reads := (if rs then operand (o 1) 4 else #[]) ++ operand (o gi) 2 ++
          (match o gi with | .gdesc g _ => Loc.span { g with idx := g.idx + 2 } 2 | _ => #[]) ++
          operand (o (gi + 1)) words ++ rd (rest (gi + 2)),
        writes := dst (o 0) words }
    | _ => conservative

/-- The footprint of instruction `i`: its form's, with its guard read. -/
def footprint (i : Instr) : Footprint :=
  let fp := formFootprint i
  let guard : Array Loc := match i.guard with
    | some (g, _) => Loc.span g 1
    | none => #[]
  -- Only convergence barriers change at issue (`Footprint.SetsB`).
  { reads := dedup (guard ++ fp.reads), writes := dedup fp.writes,
    sets := (dedup fp.sets).filter Loc.isB }


/-! ## What staying within a footprint means

`Op.ReadsWithin` and `Op.Confined` (`NV.SASS.Sem`) state it for a decoded instruction over these
definitions. Confinement holds by construction (`Op.confined`): `Footprint.effect` keeps, of an
instruction's outcome, only the issue-time changes to the control state and the sets and only the
writes to the write set. Read independence is a hypothesis of the engine-to-semantics bridge,
checked on every instruction of the test kernels (`Tests/SassFootprint.lean`). -/


/-- A writeback is local to the locations `W`: it keeps the control state and the register files'
sizes, and on `W` depends only on `W` (of states of one shape: a write to a register outside a
smaller file is lost, so states of different shapes can part). -/
structure WbLocal (W : Array Loc) (f : Warp → Warp) : Prop where
  ctrl : ∀ x, (f x).ctrl = x.ctrl
  shape : ∀ x, (f x).shape = x.shape
  agree : ∀ x y, x.shape = y.shape → Warp.AgreeOn W x y → Warp.AgreeOn W (f x) (f y)

/-- The state an effect continues with has the shape `sh`. -/
def effectShape (sh : Nat × Nat × Nat × Nat × Nat) : Effect Warp → Prop
  | .next s | .reg s _ | .mem _ s _ | .sync _ s | .collect _ _ _ s _ | .async _ s _ => s.shape = sh
  | _ => True

/-- Every writeback of an effect is local to `W`. -/
def effectWbLocal (W : Array Loc) : Effect Warp → Prop
  | .reg _ wb => WbLocal W wb
  | .mem _ _ wb | .collect _ _ _ _ wb | .async _ _ wb => ∀ resp, WbLocal W (wb resp)
  | _ => True


open Exec

/-- The location a write changes. -/
def Exec.Write.loc : Write → Loc
  | .r i _ _ | .lanes i _ _ => .r i
  | .ur i _ => .ur i
  | .p i _ _ | .pbits i _ _ => .p i
  | .up i _ => .up i
  | .b i _ => .b i
  | .printf .. | .log _ => .log
  | .bar _ => .bar

namespace Footprint

/-- The writes of `ws` to the write set. -/
def keep (fp : Footprint) (ws : List Write) : List Write :=
  ws.filter fun wr => decide (wr.loc ∈ fp.writes)

/-- The state an instruction issued from `w` continues with, given the state `s` its semantics
built: `s`'s control state and sets, `w`'s everything else. -/
def issue (fp : Footprint) (w s : Warp) : Warp :=
  { w with pcs := s.pcs, conv := s.conv, live := s.live, waiting := s.waiting,
           b := if fp.sets.isEmpty then w.b else
             w.b.mapIdx fun n v => if decide (Loc.b n ∈ fp.sets) then s.b[n]?.getD 0 else v }

/-- The effect of an outcome issued from `w`, confined to the footprint. -/
def effect (fp : Footprint) (w : Warp) : Out → Effect Warp
  | .next s => .next (fp.issue w s)
  | .reg s ws => .reg (fp.issue w s) (applyWrites (fp.keep ws))
  | .mem r s wb =>
    .mem (fun x => r (View.of fp.reads (w.readState x))) (fp.issue w s)
      fun resp => applyWrites (fp.keep (wb resp))
  | .sync op s => .sync op (fp.issue w s)
  | .async op s wb =>
    .async (fun x => op (View.of fp.reads (w.readState x))) (fp.issue w s)
      fun resp => applyWrites (fp.keep (wb resp))
  | .collect op m c s wb => .collect op m c (fp.issue w s) fun resp => applyWrites (fp.keep (wb resp))
  | .exit => .exit
  | .fault m => .fault m

/-- An outcome's effect from `w` as the semantics built it, unconfined: its continuation and
writebacks as they are (the requests read through `reads`, here all of them). Confining it
(`effect`) changes nothing exactly when the footprint is complete, which `Tests/SassFootprint`
checks. -/
def _root_.NV.SASS.Out.plain (reads : Array Loc) (w : Warp) : Out → Effect Warp
  | .next s => .next s
  | .reg s ws => .reg s (applyWrites ws)
  | .mem r s wb => .mem (fun x => r (View.of reads (w.readState x))) s fun resp => applyWrites (wb resp)
  | .sync op s => .sync op s
  | .async op s wb => .async (fun x => op (View.of reads (w.readState x))) s fun resp => applyWrites (wb resp)
  | .collect op m c s wb => .collect op m c s fun resp => applyWrites (wb resp)
  | .exit => .exit
  | .fault m => .fault m

/-- `s'` is what an instruction made of `w` (namely `s`), carried over to `w'`: control and the
issue-time sets as in `s`, every other location as in `w'`. -/
def Carried (fp : Footprint) (s w' s' : Warp) : Prop :=
  s'.ctrl = s.ctrl ∧ (∀ l ∈ fp.sets, s'.get l = s.get l) ∧ (∀ l, l ∉ fp.sets → s'.get l = w'.get l)

/-- Effects `e` (from `w`) and `e'` (from `w'`, which agrees with `w` on the reads and the
control state) are the same instruction outcome: same kind and operation, continuation states
carried over, the same writebacks, and the same request from any two states that agree on the
reads (a request reads its vector registers when it reads its sources, after issue). -/
def SameOutcome (fp : Footprint) (w' : Warp) : Effect Warp → Effect Warp → Prop
  | .next s, .next s' => fp.Carried s w' s'
  | .reg s wb, .reg s' wb' => fp.Carried s w' s' ∧ ∀ x, wb x = wb' x
  | .mem r s wb, .mem r' s' wb' =>
    fp.Carried s w' s' ∧ (∀ x x', Warp.AgreeOn fp.reads x x' → r x = r' x') ∧
      ∀ resp x, wb resp x = wb' resp x
  | .sync op s, .sync op' s' => op = op' ∧ fp.Carried s w' s'
  | .collect op mask c s wb, .collect op' mask' c' s' wb' =>
    op = op' ∧ mask = mask' ∧ c = c' ∧ fp.Carried s w' s' ∧ ∀ resp x, wb resp x = wb' resp x
  | .async op s wb, .async op' s' wb' =>
    fp.Carried s w' s' ∧ (∀ x x', Warp.AgreeOn fp.reads x x' → op x = op' x') ∧
      ∀ resp x, wb resp x = wb' resp x
  | .exit, .exit => True
  | .fault _, .fault _ => True
  | _, _ => False

/-- An effect from `w` changes, at issue, only the control state and the sets, and its writeback
only the writes. -/
def Within (fp : Footprint) (w : Warp) : Effect Warp → Prop
  | .next s | .sync _ s => ∀ l, l ∉ fp.sets → s.get l = w.get l
  | .reg s wb => (∀ l, l ∉ fp.sets → s.get l = w.get l) ∧ ∀ x l, l ∉ fp.writes → (wb x).get l = x.get l
  | .mem _ s wb | .collect _ _ _ s wb | .async _ s wb =>
    (∀ l, l ∉ fp.sets → s.get l = w.get l) ∧ ∀ resp x l, l ∉ fp.writes → (wb resp x).get l = x.get l
  | .exit | .fault _ => True

end Footprint

/-! ## Confinement, proved -/

section Frame

variable {α : Type}

theorem set!_getElem!_ne [Inhabited α] (a : Array α) {i j : Nat} (v : α) (h : i ≠ j) : (a.set! i v)[j]! = a[j]! := by
  simp [Array.set!, Array.getElem!_eq_getD, h]

theorem set!_getElem?_ne (a : Array α) {i j : Nat} (v : α) (h : i ≠ j) : (a.set! i v)[j]? = a[j]? := by
  simp [Array.set!, h]

theorem modify_getElem?_ne (a : Array α) {i j : Nat} (f : α → α) (h : i ≠ j) :
    (a.modify i f)[j]? = a[j]? := by
  simp [Array.getElem?_modify, h]

end Frame

namespace Exec

theorem mergeLanes_ne (base : Nat) (on : UInt32) (vals : Array UInt32) :
    ∀ (k : Nat) (r : Array UInt32) (j : Nat), (j < base ∨ base + k ≤ j) →
      (mergeLanes base on vals k r)[j]! = r[j]!
  | 0, _, _, _ => rfl
  | k + 1, r, j, h => by
    simp only [mergeLanes]
    rw [mergeLanes_ne base on vals k _ j (by omega)]
    split
    · exact set!_getElem!_ne _ _ (by omega)
    · rfl

end Exec

namespace Warp

@[simp] theorem setR_ur (w : Warp) (i l v) : (w.setR i l v).ur = w.ur := by unfold setR; split <;> rfl
@[simp] theorem setR_p (w : Warp) (i l v) : (w.setR i l v).p = w.p := by unfold setR; split <;> rfl
@[simp] theorem setR_up (w : Warp) (i l v) : (w.setR i l v).up = w.up := by unfold setR; split <;> rfl
@[simp] theorem setR_b (w : Warp) (i l v) : (w.setR i l v).b = w.b := by unfold setR; split <;> rfl
@[simp] theorem setUR_r (w : Warp) (i v) : (w.setUR i v).r = w.r := by unfold setUR; split <;> rfl
@[simp] theorem setUR_p (w : Warp) (i v) : (w.setUR i v).p = w.p := by unfold setUR; split <;> rfl
@[simp] theorem setUR_up (w : Warp) (i v) : (w.setUR i v).up = w.up := by unfold setUR; split <;> rfl
@[simp] theorem setUR_b (w : Warp) (i v) : (w.setUR i v).b = w.b := by unfold setUR; split <;> rfl
@[simp] theorem setP_r (w : Warp) (i l v) : (w.setP i l v).r = w.r := by unfold setP; split <;> rfl
@[simp] theorem setP_ur (w : Warp) (i l v) : (w.setP i l v).ur = w.ur := by unfold setP; split <;> rfl
@[simp] theorem setP_up (w : Warp) (i l v) : (w.setP i l v).up = w.up := by unfold setP; split <;> rfl
@[simp] theorem setP_b (w : Warp) (i l v) : (w.setP i l v).b = w.b := by unfold setP; split <;> rfl
@[simp] theorem setUP_r (w : Warp) (i v) : (w.setUP i v).r = w.r := by unfold setUP; split <;> rfl
@[simp] theorem setUP_ur (w : Warp) (i v) : (w.setUP i v).ur = w.ur := by unfold setUP; split <;> rfl
@[simp] theorem setUP_p (w : Warp) (i v) : (w.setUP i v).p = w.p := by unfold setUP; split <;> rfl
@[simp] theorem setUP_b (w : Warp) (i v) : (w.setUP i v).b = w.b := by unfold setUP; split <;> rfl
@[simp] theorem setR_printf (w : Warp) (i l v) : (w.setR i l v).printf = w.printf := by
  unfold setR; split <;> rfl
@[simp] theorem setUR_printf (w : Warp) (i v) : (w.setUR i v).printf = w.printf := by
  unfold setUR; split <;> rfl
@[simp] theorem setP_printf (w : Warp) (i l v) : (w.setP i l v).printf = w.printf := by
  unfold setP; split <;> rfl
@[simp] theorem setUP_printf (w : Warp) (i v) : (w.setUP i v).printf = w.printf := by
  unfold setUP; split <;> rfl

end Warp

@[simp] theorem setLanes_ur (w : Warp) (i on vs) : (setLanes w i on vs).ur = w.ur := by
  unfold setLanes; split <;> rfl
@[simp] theorem setLanes_p (w : Warp) (i on vs) : (setLanes w i on vs).p = w.p := by
  unfold setLanes; split <;> rfl
@[simp] theorem setLanes_up (w : Warp) (i on vs) : (setLanes w i on vs).up = w.up := by
  unfold setLanes; split <;> rfl
@[simp] theorem setLanes_b (w : Warp) (i on vs) : (setLanes w i on vs).b = w.b := by
  unfold setLanes; split <;> rfl
@[simp] theorem setLanes_printf (w : Warp) (i on vs) : (setLanes w i on vs).printf = w.printf := by
  unfold setLanes; split <;> rfl
@[simp] theorem setBits_printf (w : Warp) (i on bits) : (setBits w i on bits).printf = w.printf := by
  unfold setBits; split <;> rfl
@[simp] theorem setBits_r (w : Warp) (i on bits) : (setBits w i on bits).r = w.r := by
  unfold setBits; split <;> rfl
@[simp] theorem setBits_ur (w : Warp) (i on bits) : (setBits w i on bits).ur = w.ur := by
  unfold setBits; split <;> rfl
@[simp] theorem setBits_up (w : Warp) (i on bits) : (setBits w i on bits).up = w.up := by
  unfold setBits; split <;> rfl
@[simp] theorem setBits_b (w : Warp) (i on bits) : (setBits w i on bits).b = w.b := by
  unfold setBits; split <;> rfl

theorem getR_setR_ne (w : Warp) {i n : Nat} (l k : Fin 32) (v : UInt32) (h : i ≠ n) :
    (w.setR i l v).getR n k = w.getR n k := by
  have := l.isLt; have := k.isLt
  by_cases hi : i = 255
  · simp [Warp.setR, hi]
  · by_cases hn : n = 255
    · simp [Warp.getR, hn]
    · simp only [Warp.setR, Warp.getR, beq_iff_eq, hi, hn, ite_false]
      exact set!_getElem!_ne _ _ (by omega)

theorem getR_setLanes_ne (w : Warp) {i n : Nat} (on : UInt32) (vs : Array UInt32) (k : Fin 32)
    (h : i ≠ n) : (setLanes w i on vs).getR n k = w.getR n k := by
  have := k.isLt
  by_cases hi : i = 255
  · simp [setLanes, hi]
  · by_cases hn : n = 255
    · simp [Warp.getR, hn]
    · simp only [setLanes, Warp.getR, beq_iff_eq, hi, hn, ite_false]
      exact mergeLanes_ne _ _ _ _ _ _ (by omega)

theorem getUR_setUR_ne (w : Warp) {i n : Nat} (v : UInt32) (h : i ≠ n) :
    (w.setUR i v).getUR n = w.getUR n := by
  by_cases hi : i = 255
  · simp [Warp.setUR, hi]
  · by_cases hn : n = 255
    · simp [Warp.getUR, hn]
    · simp only [Warp.setUR, Warp.getUR, beq_iff_eq, hi, hn, ite_false]
      exact set!_getElem!_ne _ _ h

theorem p_setP_ne (w : Warp) {i n : Nat} (l : Nat) (v : Bool) (h : i ≠ n) :
    (w.setP i l v).p[n]? = w.p[n]? := by
  unfold Warp.setP; split
  · rfl
  · exact set!_getElem?_ne _ _ h

theorem p_setBits_ne (w : Warp) {i n : Nat} (on bits : UInt32) (h : i ≠ n) :
    (setBits w i on bits).p[n]? = w.p[n]? := by
  unfold setBits; split
  · rfl
  · exact modify_getElem?_ne _ _ h

theorem up_setUP_ne (w : Warp) {i n : Nat} (v : Bool) (h : i ≠ n) :
    (w.setUP i v).up[n]? = w.up[n]? := by
  unfold Warp.setUP; split
  · rfl
  · exact set!_getElem?_ne _ _ h

/-- A write changes its location only. -/
theorem Write.apply_get (wr : Write) (x : Warp) (l : Loc) (h : wr.loc ≠ l) :
    (Write.apply x wr).get l = x.get l := by
  funext k
  cases wr with
  | r i lane v => cases l with
    | r n => exact getR_setR_ne x lane k v (fun e => h (by simp [Write.loc, e]))
    | bar => simp only [Write.apply, Warp.get, Warp.setR]; split <;> rfl
    | _ => first | rfl | simp [Write.apply, Warp.get, Warp.getUR]
  | lanes i on vs => cases l with
    | r n => exact getR_setLanes_ne x on vs k (fun e => h (by simp [Write.loc, e]))
    | bar => simp only [Write.apply, Warp.get, setLanes]; split <;> rfl
    | _ => first | rfl | simp [Write.apply, Warp.get, Warp.getUR]
  | ur i v => cases l with
    | ur n => exact getUR_setUR_ne x v (fun e => h (by simp [Write.loc, e]))
    | bar => simp only [Write.apply, Warp.get, Warp.setUR]; split <;> rfl
    | _ => first | rfl | simp [Write.apply, Warp.get, Warp.getR]
  | p i lane v => cases l with
    | p n => simp only [Write.apply, Warp.get]; rw [p_setP_ne x lane v (fun e => h (by simp [Write.loc, e]))]
    | bar => simp only [Write.apply, Warp.get, Warp.setP]; split <;> rfl
    | _ => first | rfl | simp [Write.apply, Warp.get, Warp.getR, Warp.getUR]
  | pbits i on bits => cases l with
    | p n => simp only [Write.apply, Warp.get]; rw [p_setBits_ne x on bits (fun e => h (by simp [Write.loc, e]))]
    | bar => simp only [Write.apply, Warp.get, setBits]; split <;> rfl
    | _ => first | rfl | simp [Write.apply, Warp.get, Warp.getR, Warp.getUR]
  | up i v => cases l with
    | up n =>
      have hn : i ≠ n := fun e => h (by simp [Write.loc, e])
      simp only [Write.apply, Warp.get, up_setUP_ne x v hn]
    | bar => simp only [Write.apply, Warp.get, Warp.setUP]; split <;> rfl
    | _ => first | rfl | simp [Write.apply, Warp.get, Warp.getR, Warp.getUR]
  | b i v => cases l with
    | b n => simp only [Write.apply, Warp.get]; rw [set!_getElem?_ne _ _ (fun e => h (by simp [Write.loc, e]))]
    | _ => first | rfl | simp [Write.apply, Warp.get, Warp.getR, Warp.getUR]
  | printf f a => cases l with
    | log => exact absurd rfl h
    | _ => rfl
  | log es => cases l with
    | log => exact absurd rfl h
    | _ => rfl
  | bar v => cases l with
    | bar => exact absurd rfl h
    | _ => rfl

/-- Writes change their locations only. -/
theorem applyWrites_get : ∀ (ws : List Write) (x : Warp) (l : Loc), (∀ wr ∈ ws, wr.loc ≠ l) →
    (applyWrites ws x).get l = x.get l
  | [], _, _, _ => rfl
  | wr :: ws, x, l, h =>
    (applyWrites_get ws (Write.apply x wr) l (fun w hw => h w (List.mem_cons_of_mem _ hw))).trans
      (Write.apply_get wr x l (h wr (by simp)))

namespace Footprint

theorem keep_loc {fp : Footprint} {ws : List Write} {l : Loc} (hl : l ∉ fp.writes) :
    ∀ wr ∈ fp.keep ws, wr.loc ≠ l := by
  intro wr hw e
  simp only [keep, List.mem_filter, decide_eq_true_eq] at hw
  exact hl (e ▸ hw.2)

theorem issue_get (fp : Footprint) (w s : Warp) (l : Loc) (h : l ∉ fp.sets) :
    (fp.issue w s).get l = w.get l := by
  funext k
  cases l with
  | b n =>
    simp only [issue, Warp.get]
    split
    · rfl
    · simp [Array.getElem?_mapIdx, h]
  | _ => rfl

theorem effect_within (fp : Footprint) (w : Warp) (o : Out) : fp.Within w (fp.effect w o) := by
  cases o with
  | next s => exact fp.issue_get w s
  | reg s ws => exact ⟨fp.issue_get w s, fun x l hl => applyWrites_get _ x l (keep_loc hl)⟩
  | mem r s wb => exact ⟨fp.issue_get w s, fun resp x l hl => applyWrites_get _ x l (keep_loc hl)⟩
  | sync op s => exact fp.issue_get w s
  | async op s wb => exact ⟨fp.issue_get w s, fun resp x l hl => applyWrites_get _ x l (keep_loc hl)⟩
  | collect op m c s wb =>
    exact ⟨fp.issue_get w s, fun resp x l hl => applyWrites_get _ x l (keep_loc hl)⟩
  | exit => trivial
  | fault m => trivial

theorem issue_shape (fp : Footprint) (w s : Warp) : (fp.issue w s).shape = w.shape := by
  simp only [issue, Warp.shape]
  split <;> simp

/-- The state an instruction continues with keeps the issuer's shape. -/
theorem effect_shape (fp : Footprint) (w : Warp) (o : Out) :
    effectShape w.shape (fp.effect w o) := by
  cases o <;> first | exact fp.issue_shape w _ | trivial

end Footprint

/-! ## Local writebacks, proved for data writebacks -/

section Congr

variable {α : Type}

theorem set!_congr [Inhabited α] (a b : Array α) (hs : a.size = b.size) (i j : Nat) (v : α)
    (h : a[j]! = b[j]!) : (a.set! i v)[j]! = (b.set! i v)[j]! := by
  by_cases hij : i = j
  · subst hij
    simp [Array.set!, Array.getElem!_eq_getD, Array.getElem?_setIfInBounds, hs]
  · rw [set!_getElem!_ne _ _ hij, set!_getElem!_ne _ _ hij, h]

theorem set!_congr? (a b : Array α) (hs : a.size = b.size) (i j : Nat) (v : α)
    (h : a[j]? = b[j]?) : (a.set! i v)[j]? = (b.set! i v)[j]? := by
  by_cases hij : i = j
  · subst hij
    simp [Array.set!, Array.getElem?_setIfInBounds, hs]
  · rw [set!_getElem?_ne _ _ hij, set!_getElem?_ne _ _ hij, h]

theorem getElem?_of_getD (a b : Array α) (hs : a.size = b.size) (j : Nat) (d : α)
    (h : a[j]?.getD d = b[j]?.getD d) : a[j]? = b[j]? := by
  by_cases hj : j < a.size
  · have hj' : j < b.size := hs ▸ hj
    simp [Array.getElem?_eq_getElem hj, Array.getElem?_eq_getElem hj'] at h ⊢
    exact h
  · have hj' : ¬ j < b.size := hs ▸ hj
    simp [Array.getElem?_eq_none (Nat.le_of_not_lt hj), Array.getElem?_eq_none (Nat.le_of_not_lt hj')]

theorem getElem!_of_getElem? [Inhabited α] (a b : Array α) (j : Nat) (h : a[j]? = b[j]?) :
    a[j]! = b[j]! := by
  simp [Array.getElem!_eq_getD, h]

end Congr

namespace Exec

theorem mergeLanes_size (base : Nat) (on : UInt32) (vals : Array UInt32) :
    ∀ (k : Nat) (r : Array UInt32), (mergeLanes base on vals k r).size = r.size
  | 0, _ => rfl
  | k + 1, r => by
    simp only [mergeLanes]
    rw [mergeLanes_size base on vals k]
    split <;> simp [Array.set!]

theorem mergeLanes_congr (base : Nat) (on : UInt32) (vals : Array UInt32) :
    ∀ (k : Nat) (r r' : Array UInt32) (j : Nat), r.size = r'.size → r[j]! = r'[j]! →
      (mergeLanes base on vals k r)[j]! = (mergeLanes base on vals k r')[j]!
  | 0, _, _, _, _, h => h
  | k + 1, r, r', j, hs, h => by
    simp only [mergeLanes]
    split
    · exact mergeLanes_congr base on vals k _ _ j (by simp [Array.set!, hs])
        (set!_congr r r' hs _ j _ h)
    · exact mergeLanes_congr base on vals k _ _ j hs h

end Exec

/-- A write keeps the control state. -/
theorem Exec.Write.apply_ctrl (x : Warp) (wr : Write) : (Write.apply x wr).ctrl = x.ctrl := by
  cases wr <;> simp only [Write.apply, Warp.ctrl, Warp.setR, Warp.setUR, Warp.setP, Warp.setUP,
    setLanes, setBits] <;> (try split) <;> rfl

/-- A write keeps the register files' sizes. -/
theorem Exec.Write.apply_shape (x : Warp) (wr : Write) : (Write.apply x wr).shape = x.shape := by
  cases wr <;> simp only [Write.apply, Warp.shape, Warp.setR, Warp.setUR, Warp.setP, Warp.setUP,
    setLanes, setBits] <;> (try split) <;> simp [Array.set!, mergeLanes_size]

/-- What a write leaves at a location depends only on what was there (states of one shape). -/
theorem Exec.Write.apply_get_congr (wr : Write) (x y : Warp) (hs : x.shape = y.shape) (l : Loc)
    (h : x.get l = y.get l) : (Write.apply x wr).get l = (Write.apply y wr).get l := by
  by_cases hl : wr.loc = l
  · simp only [Warp.shape, Prod.mk.injEq] at hs
    obtain ⟨hr, hur, hp, hup, hb⟩ := hs
    subst hl
    funext k
    have hk := congrFun h k
    cases wr with
    | r i lane v =>
      simp only [Write.loc, Write.apply, Warp.get, Warp.getR, Warp.setR] at hk ⊢
      split
      · rfl
      · rename_i hi; simp only [hi] at hk; exact set!_congr _ _ hr _ _ _ hk
    | lanes i on vs =>
      simp only [Write.loc, Write.apply, Warp.get, Warp.getR, setLanes] at hk ⊢
      split
      · rfl
      · rename_i hi; simp only [hi] at hk; exact mergeLanes_congr _ _ _ _ _ _ _ hr hk
    | ur i v =>
      simp only [Write.loc, Write.apply, Warp.get, Warp.getUR, Warp.setUR] at hk ⊢
      split
      · rfl
      · rename_i hi; simp only [hi] at hk; exact set!_congr _ _ hur _ _ _ hk
    | p i lane v =>
      simp only [Write.loc, Write.apply, Warp.get, Warp.setP] at hk ⊢
      have e := getElem?_of_getD _ _ hp i 0 hk
      have e! := getElem!_of_getElem? _ _ i e
      split
      · exact hk
      · rw [e!]
        exact congrArg (·.getD 0) (set!_congr? _ _ hp _ _ _ e)
    | pbits i on bits =>
      simp only [Write.loc, Write.apply, Warp.get, setBits] at hk ⊢
      have e := getElem?_of_getD _ _ hp i 0 hk
      split
      · exact hk
      · simp [Array.getElem?_modify, e]
    | up i v =>
      simp only [Write.loc, Write.apply, Warp.get, Warp.setUP] at hk ⊢
      split
      · exact hk
      · simp [Array.set!, Array.getElem?_setIfInBounds, hup]
    | b i v =>
      simp only [Write.loc, Write.apply, Warp.get] at hk ⊢
      simp [Array.set!, Array.getElem?_setIfInBounds, hb]
    | printf f a =>
      simp only [Write.loc, Write.apply, Warp.get, Array.size_push] at hk ⊢
      simp only [Nat.toUInt32, UInt32.ofNat_add] at hk ⊢
      rw [hk]
    | log es => rfl
    | bar v => rfl
  · rw [Write.apply_get wr x l hl, Write.apply_get wr y l hl, h]

theorem applyWrites_ctrl : ∀ (ws : List Write) (x : Warp), (applyWrites ws x).ctrl = x.ctrl
  | [], _ => rfl
  | wr :: ws, x => (applyWrites_ctrl ws _).trans (Write.apply_ctrl x wr)

theorem applyWrites_shape : ∀ (ws : List Write) (x : Warp), (applyWrites ws x).shape = x.shape
  | [], _ => rfl
  | wr :: ws, x => (applyWrites_shape ws _).trans (Write.apply_shape x wr)

theorem applyWrites_get_congr : ∀ (ws : List Write) (x y : Warp), x.shape = y.shape →
    ∀ l, x.get l = y.get l → (applyWrites ws x).get l = (applyWrites ws y).get l
  | [], _, _, _, _, h => h
  | wr :: ws, x, y, hs, l, h =>
    applyWrites_get_congr ws _ _
      (by rw [Write.apply_shape, Write.apply_shape, hs]) l (Write.apply_get_congr wr x y hs l h)

namespace Footprint

/-- A writeback of fixed writes is local to any locations. -/
theorem wbLocal_writes (W : Array Loc) (ws : List Write) : WbLocal W (applyWrites ws) :=
  ⟨applyWrites_ctrl ws, applyWrites_shape ws,
   fun x y hs h l hl => applyWrites_get_congr ws x y hs l (h l hl)⟩

/-- Every outcome's writebacks are data, so local. -/
theorem effect_wbLocal (fp : Footprint) (w : Warp) (o : Out) :
    effectWbLocal fp.writes (fp.effect w o) := by
  cases o with
  | reg s ws => exact wbLocal_writes _ _
  | mem r s wb => exact fun resp => wbLocal_writes _ _
  | async op s wb => exact fun resp => wbLocal_writes _ _
  | collect op m c s wb => exact fun resp => wbLocal_writes _ _
  | _ => trivial

end Footprint

/-! ## Outcomes depend only on the reads

`Op.run` gives a decoder only what `View.of fp.reads w` and `Warp.blank w` hold, so two warps that
agree on the reads and the control state get the same outcome; confining it to the footprint from
each (`effect`) then gives the same instruction outcome (`SameOutcome`). -/

namespace Footprint

/-- Only convergence barriers are set at issue. -/
def SetsB (fp : Footprint) : Prop := ∀ l ∈ fp.sets, l.isB = true

theorem setsB_empty : SetsB {} := fun _ h => by simp at h

theorem issue_ctrl (fp : Footprint) (w s : Warp) : (fp.issue w s).ctrl = s.ctrl := rfl

/-- What an instruction sets at issue lands the same in any two issuers of one shape. -/
theorem issue_carried (fp : Footprint) (hb : fp.SetsB) (w w' s : Warp) (hs : w.shape = w'.shape) :
    fp.Carried (fp.issue w s) w' (fp.issue w' s) := by
  refine ⟨rfl, fun l hl => ?_, fun l hl => fp.issue_get w' s l hl⟩
  have hB := hb l hl
  simp only [Warp.shape, Prod.mk.injEq] at hs
  obtain ⟨_, _, _, _, hsz⟩ := hs
  cases l with
  | b n =>
    funext k
    have hne : fp.sets.isEmpty = false := by
      cases h : fp.sets.isEmpty
      · rfl
      · simp only [Array.isEmpty_iff] at h; simp [h] at hl
    simp only [issue, Warp.get, hne, Bool.false_eq_true, ↓reduceIte, Array.getElem?_mapIdx]
    by_cases hn : n < w.b.size
    · have hn' : n < w'.b.size := hsz ▸ hn
      simp [Array.getElem?_eq_getElem hn, Array.getElem?_eq_getElem hn', hl]
    · have hn' : ¬ n < w'.b.size := hsz ▸ hn
      simp [Array.getElem?_eq_none (Nat.le_of_not_lt hn), Array.getElem?_eq_none (Nat.le_of_not_lt hn')]
  | _ => simp [Loc.isB] at hB

/-- The state a request reads agrees on the reads when the issuers and the later states do. -/
theorem readState_agree {ls : Array Loc} {w w' x x' : Warp} (hw : Warp.AgreeOn ls w w')
    (hx : Warp.AgreeOn ls x x') : Warp.AgreeOn ls (w.readState x) (w'.readState x') := by
  intro l hl
  have h1 := hw l hl
  have h2 := hx l hl
  -- Vector registers, barriers and the logs from the later states; the rest from the issuers.
  cases l <;> first | exact h1 | exact h2

/-- One outcome, confined from two issuers that agree on the reads, the control state and the
shape, is one instruction outcome. -/
theorem effect_same (fp : Footprint) (hb : fp.SetsB) (w w' : Warp) (hs : w.shape = w'.shape)
    (ha : Warp.AgreeOn fp.reads w w') (o : Out) :
    fp.SameOutcome w' (fp.effect w o) (fp.effect w' o) := by
  have hc := fp.issue_carried hb w w'
  have hv : ∀ x x', Warp.AgreeOn fp.reads x x' →
      View.of fp.reads (w.readState x) = View.of fp.reads (w'.readState x') :=
    fun x x' h => View.of_congr (readState_agree ha h)
  cases o with
  | next s => exact hc s hs
  | reg s ws => exact ⟨hc s hs, fun _ => rfl⟩
  | mem r s wb => exact ⟨hc s hs, fun x x' h => by simp only [hv x x' h], fun _ _ => rfl⟩
  | sync op s => exact ⟨rfl, hc s hs⟩
  | async op s wb => exact ⟨hc s hs, fun x x' h => by simp only [hv x x' h], fun _ _ => rfl⟩
  | collect op m c s wb => exact ⟨rfl, rfl, rfl, hc s hs, fun _ _ => rfl⟩
  | exit => trivial
  | fault m => trivial

end Footprint

theorem footprint_setsB (i : Instr) : (footprint i).SetsB := by
  intro l hl
  simp only [footprint, Array.mem_filter] at hl
  exact hl.2

end NV.SASS
