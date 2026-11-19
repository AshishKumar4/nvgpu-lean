import NV.PTX.Syntax.Type

/-!
# Special registers

The predefined read-only registers of PTX ISA 9.4 chapter 10, with their types. The four
`.v4 .u32` registers of the thread hierarchy (`%tid`, `%ntid`, `%ctaid`, …) are read one component
at a time (`%tid.x`); the others are scalars.
-/

namespace NV.PTX

/-- A component `.x`, `.y`, `.z` of a vector special register; `.w` reads as 0. -/
abbrev Dim := Fin 4

inductive SReg where
  | tid (c : Dim) | ntid (c : Dim) | ctaid (c : Dim) | nctaid (c : Dim)
  | clusterid (c : Dim) | nclusterid (c : Dim) | clusterCtaid (c : Dim) | clusterNctaid (c : Dim)
  | laneid | warpid | nwarpid | smid | nsmid | gridid
  | isExplicitCluster | clusterCtarank | clusterNctarank
  | lanemaskEq | lanemaskLe | lanemaskLt | lanemaskGe | lanemaskGt
  | clock | clockHi | clock64
  | pm (i : Fin 8) | pm64 (i : Fin 8) | envreg (i : Fin 32)
  | globaltimer | globaltimerLo | globaltimerHi
  | reservedSmemOffsetBegin | reservedSmemOffsetEnd | reservedSmemOffsetCap
  | reservedSmemOffset (i : Fin 2)
  | totalSmemSize | aggrSmemSize | dynamicSmemSize | currentGraphExec
  | perctamemoryoffset | perctamemorysize
  deriving DecidableEq, Repr, Inhabited

namespace SReg

/-- Scalar special registers by name (§10.3–§10.35). -/
def scalars : List (String × SReg) :=
  [("%laneid", .laneid), ("%warpid", .warpid), ("%nwarpid", .nwarpid), ("%smid", .smid),
   ("%nsmid", .nsmid), ("%gridid", .gridid), ("%is_explicit_cluster", .isExplicitCluster),
   ("%cluster_ctarank", .clusterCtarank), ("%cluster_nctarank", .clusterNctarank),
   ("%lanemask_eq", .lanemaskEq), ("%lanemask_le", .lanemaskLe), ("%lanemask_lt", .lanemaskLt),
   ("%lanemask_ge", .lanemaskGe), ("%lanemask_gt", .lanemaskGt), ("%clock", .clock),
   ("%clock_hi", .clockHi), ("%clock64", .clock64), ("%globaltimer", .globaltimer),
   ("%globaltimer_lo", .globaltimerLo), ("%globaltimer_hi", .globaltimerHi),
   ("%reserved_smem_offset_begin", .reservedSmemOffsetBegin),
   ("%reserved_smem_offset_end", .reservedSmemOffsetEnd),
   ("%reserved_smem_offset_cap", .reservedSmemOffsetCap),
   ("%total_smem_size", .totalSmemSize), ("%aggr_smem_size", .aggrSmemSize),
   ("%dynamic_smem_size", .dynamicSmemSize), ("%current_graph_exec", .currentGraphExec),
   ("%perctamemoryoffset", .perctamemoryoffset), ("%perctamemorysize", .perctamemorysize)]

/-- Vector special registers by name (§10.1, §10.2, §10.6, §10.7, §10.12–§10.15). -/
def vectors : List (String × (Dim → SReg)) :=
  [("%tid", .tid), ("%ntid", .ntid), ("%ctaid", .ctaid), ("%nctaid", .nctaid),
   ("%clusterid", .clusterid), ("%nclusterid", .nclusterid), ("%cluster_ctaid", .clusterCtaid),
   ("%cluster_nctaid", .clusterNctaid)]

/-- `%name<i>` registers: `%pm0`…`%pm7`, `%pm0_64`…, `%envreg0`…`%envreg31`,
`%reserved_smem_offset_0`, `_1`. -/
def indexed? (s : String) : Option SReg :=
  let num (p : String) (suffix : String := "") : Option Nat :=
    if s.startsWith p && s.endsWith suffix then
      let mid := String.ofList ((s.toList.drop p.length).take (s.length - p.length - suffix.length))
      if mid.isEmpty || !mid.all Char.isDigit then none else mid.toNat?
    else none
  match num "%pm" "_64", num "%pm", num "%envreg", num "%reserved_smem_offset_" with
  | some i, _, _, _ => if h : i < 8 then some (.pm64 ⟨i, h⟩) else none
  | _, some i, _, _ => if h : i < 8 then some (.pm ⟨i, h⟩) else none
  | _, _, some i, _ => if h : i < 32 then some (.envreg ⟨i, h⟩) else none
  | _, _, _, some i => if h : i < 2 then some (.reservedSmemOffset ⟨i, h⟩) else none
  | _, _, _, _ => none

def component? : String → Option Dim
  | "x" => some 0 | "y" => some 1 | "z" => some 2 | "w" => some 3 | _ => none

/-- The special register `name` (with component `comp` for the vector ones). -/
def lookup? (name : String) (comp : Option String) : Option SReg :=
  match comp with
  | some c => do
    let f ← vectors.lookup name
    return f (← component? c)
  | none => scalars.lookup name <|> indexed? name

/-- The register's type (chapter 10). -/
def ty : SReg → Ty
  | .gridid | .clock64 | .pm64 _ | .globaltimer | .currentGraphExec => .u64
  | .isExplicitCluster => .pred
  | .lanemaskEq | .lanemaskLe | .lanemaskLt | .lanemaskGe | .lanemaskGt
  | .envreg _ | .reservedSmemOffsetBegin | .reservedSmemOffsetEnd | .reservedSmemOffsetCap
  | .reservedSmemOffset _ => .b32
  | _ => .u32

end SReg

end NV.PTX
