import NV.PTX.Syntax.Instr
import NV.PTX.Syntax.SReg

/-!
# The PTX type checker

The ISA's static rules over a decoded module (PTX ISA 9.4):

* every name an instruction uses is declared: a register, variable, parameter, function,
  label, call prototype or special register (§4.4, §5.4, chapter 10), in scope (§9.7.14.1);
* guards and predicate operands are `.pred` registers (§9.7.14.2);
* the instruction's types are among those its syntax lists (§9.7.1–§9.7.10, §9.7.15), for the
  instructions `Signature.of` describes;
* the target architecture provides the instruction (`Op.requires`, from §9.7's Target ISA
  Notes for each instruction's base form);
* operands have the shape the instruction requires (destination register, source register or
  immediate, address, vector of the right length, predicate) and register types compatible with
  the instruction type: same size and basic type, integers of one size, or bits of the size
  (§5.2.1, §6.1); `ld`, `st` and `cvt` (and the integer and bit forms of `mov`'s pack/unpack)
  take registers wider than the type (§9.4.1, Tables 27 and 28).

Instructions without a `Signature` (texture, surface, tensor-core and fabric instructions,
whose operands are lists the asynchronous-operation semantics interprets) are checked for
declared names only.
-/

namespace NV.PTX

/-- What a name denotes. -/
inductive Sym where
  | reg (d : VarDecl)
  | var (d : VarDecl)
  | func
  | label
  | proto
  deriving Inhabited

/-- One operand position of an instruction. -/
inductive Slot where
  /-- A destination register of the type, or `_`. -/
  | dst (t : Ty)
  /-- A source register or immediate of the type. -/
  | src (t : Ty)
  /-- A destination register, or `{…}` of `n` registers, of the type (`ld`, vector `mov`). -/
  | dstVec (t : Ty) (n : Nat)
  | srcVec (t : Ty) (n : Nat)
  /-- A predicate source, possibly negated (`!p`), or an integer immediate. -/
  | pred
  /-- A destination predicate or `p|q` pair (`setp`), or `_`. -/
  | dpred
  /-- A destination register and an optional predicate after `|` (`shfl.sync d|p`). -/
  | dstPred (t : Ty)
  /-- `[address]`. -/
  | addr
  /-- A label (branch target). -/
  | label
  /-- Not checked here (call argument lists, symbols of `mov`/`cvta`). -/
  | any
  deriving Inhabited

/-- An instruction's operand slots: required ones, then optional trailing ones. -/
structure Signature where
  slots : List Slot
  optional : List Slot := []
  /-- Registers may be wider than the type (`ld`, `st`, `cvt`; §9.4.1). -/
  relaxed : Bool := false

structure Diag where
  line : Nat
  msg : String
  deriving Repr

namespace Check

def ints : List Ty := [.u16, .u32, .u64, .s16, .s32, .s64]
def bits : List Ty := [.b16, .b32, .b64]
def floats : List Ty := [.f32, .f64]
def halves : List Ty := [.f16, .f16x2, .bf16, .bf16x2]
def ldst : List Ty :=
  [.b8, .b16, .b32, .b64, .b128, .u8, .u16, .u32, .u64, .s8, .s16, .s32, .s64, .f32, .f64,
   .f16, .f16x2, .bf16, .bf16x2]

def wide (t : Ty) : Ty :=
  match t with
  | .u16 => .u32 | .u32 => .u64 | .s16 => .s32 | .s32 => .s64 | t => t

/-- Types the instruction admits, as a check of its written type list; `none` for opcodes
whose type rules this checker does not state. -/
def typesOk (i : Instr) : Option Bool :=
  let one (ok : List Ty) := some (match i.types with | [t] => ok.contains t | _ => false)
  match i.op with
  | .add | .sub =>
    match i.types with
    | [t] => some ((ints ++ floats ++ halves ++ ([.u16x2, .s16x2, .u8x4, .s8x4, .f32x2] : List Ty)).contains t)
    | [_, _] | [_, _, _] => some true  -- mixed-precision and alternate formats (§9.7.5, §9.7.6)
    | _ => some false
  | .mul =>
    match i.types with
    | [t] => some ((ints ++ floats ++ halves ++ ([.f32x2] : List Ty)).contains t)
    | [_, _, _] => some true
    | _ => some false
  | .mad => one (ints ++ floats)
  | .fma =>
    match i.types with
    | [t] => some ((floats ++ halves ++ ([.f32x2] : List Ty)).contains t)
    | [_, _] | [_, _, _] | [_, _, _, _] => some true
    | _ => some false
  | .mul24 | .mad24 => one [.u32, .s32]
  | .sad | .div | .rem => one (ints ++ floats)
  | .abs | .neg => one (([.s16, .s32, .s64, .s8x4] : List Ty) ++ floats ++ halves)
  | .min | .max => one (ints ++ floats ++ halves ++ ([.u16x2, .s16x2, .u8x4, .s8x4] : List Ty))
  | .popc | .clz | .brev => one [.b32, .b64]
  | .bfind | .bfe => one [.u32, .u64, .s32, .s64]
  | .bfi => one [.b32, .b64]
  | .fns | .bmsk | .lop3 | .prmt => one [.b32]
  | .szext => one [.u32, .s32]
  | .dp4a | .dp2a => some (i.types.length == 2 && i.types.all ([.u32, .s32].contains ·))
  | .addc | .subc | .madc => one [.u32, .s32, .u64, .s64]
  | .testp | .copysign => one floats
  | .rcp | .sqrt | .rsqrt => one floats
  | .sin | .cos | .lg2 => one [.f32]
  | .ex2 | .tanh => one (([.f32] : List Ty) ++ halves)
  | .setp => one (bits ++ ints ++ floats ++ halves)
  | .set =>
    match i.types with
    | [d, s] => some ([.u32, .s32, .f32, .u16, .s16, .f16, .bf16, .f16x2, .bf16x2].contains d &&
        (bits ++ ints ++ floats ++ halves).contains s)
    | [t] => some ([.u8x4, .s8x4, .u16x2, .s16x2].contains t)
    | _ => some false
  | .selp => one (bits ++ ints ++ floats)
  | .slct =>
    match i.types with
    | [d, c] => some ((bits ++ ints ++ floats).contains d && [.s32, .f32].contains c)
    | _ => some false
  | .and | .or | .xor | .not => one (([.pred] : List Ty) ++ bits)
  | .cnot | .shl => one bits
  | .shr => one (bits ++ ints)
  | .shf => one [.b32]
  | .mov => one (([.pred, .b128] : List Ty) ++ bits ++ ints ++ floats)
  | .ld | .st | .ldu => one ldst
  | .cvt =>
    match i.types with
    | [_, _] => some true
    | _ => some false
  | .cvta | .mapa | .getctarank => one [.u32, .u64]
  | .shflSync | .shfl => one [.b32]
  | .voteSync | .vote => one [.pred, .b32]
  | .matchSync => one [.b32, .b64]
  | .reduxSync => one [.u32, .s32, .b32, .f32]
  | .activemask => one [.b32]
  | .nanosleep => one [.u32]
  | _ => none

/-- Operand slots of the instructions the checker types. -/
def signature (i : Instr) : Option Signature :=
  let t := i.types.headD .b32
  let bin := some { slots := [.dst t, .src t, .src t] : Signature }
  let un := some { slots := [.dst t, .src t] : Signature }
  let ter := some { slots := [.dst t, .src t, .src t, .src t] : Signature }
  match i.op with
  | .add | .sub | .addc | .subc | .div | .rem | .copysign | .and | .or | .xor =>
    if i.types.length == 1 then bin else none
  | .min | .max =>
    if i.types.length == 1 then
      if i.args.size == 4 then ter else bin
    else none
  | .mul | .mul24 =>
    if i.types.length != 1 then none
    else if i.has .wide then some { slots := [.dst (wide t), .src t, .src t] } else bin
  | .mad | .madc | .mad24 =>
    if i.types.length != 1 then none
    else if i.has .wide then some { slots := [.dst (wide t), .src t, .src t, .src (wide t)] }
    else ter
  | .fma => if i.types.length == 1 then ter else none
  | .sad => ter
  | .abs | .neg | .not | .cnot | .rcp | .sqrt | .rsqrt | .sin | .cos | .lg2 | .ex2 | .tanh
  | .brev => un
  | .popc | .clz | .bfind => some { slots := [.dst .u32, .src t] }
  | .bfe => some { slots := [.dst t, .src t, .src .u32, .src .u32] }
  | .bfi => some { slots := [.dst t, .src t, .src t, .src .u32, .src .u32] }
  | .fns => some { slots := [.dst .b32, .src .b32, .src .u32, .src .s32] }
  | .szext => some { slots := [.dst t, .src t, .src .u32] }
  | .bmsk => some { slots := [.dst .b32, .src .b32, .src .b32] }
  | .dp4a | .dp2a => some { slots := [.dst t, .src t, .src (i.types.getD 1 t), .src t] }
  | .prmt => ter
  | .lop3 =>
    if i.bool?.isSome then some { slots := [.dstPred .b32, .src .b32, .src .b32, .src .b32,
      .src .b32, .pred] }
    else some { slots := [.dst .b32, .src .b32, .src .b32, .src .b32, .src .b32] }
  | .shl | .shr => some { slots := [.dst t, .src t, .src .u32] }
  | .shf => some { slots := [.dst .b32, .src .b32, .src .b32, .src .u32] }
  | .testp => some { slots := [.dpred, .src t] }
  | .setp => some { slots := [.dpred, .src t, .src t], optional := [.pred] }
  | .set =>
    match i.types with
    | [d, s] => some { slots := [.dst d, .src s, .src s], optional := [.pred] }
    | _ => some { slots := [.dst t, .src t, .src t] }
  | .selp => some { slots := [.dst t, .src t, .src t, .pred] }
  | .slct => some { slots := [.dst t, .src t, .src t, .src (i.types.getD 1 .s32)] }
  | .cvt =>
    match i.types with
    | [d, s] =>
      if d.basic == .format || s.basic == .format || i.args.size != 2 then none
      else some { slots := [.dst d, .src s], relaxed := true }
    | _ => none
  | .ld =>
    let n := (i.vec?.map Vec.length).getD 1
    some { slots := [.dstVec t n, .addr], optional := [.src .b64], relaxed := true }
  | .ldu =>
    let n := (i.vec?.map Vec.length).getD 1
    some { slots := [.dstVec t n, .addr], relaxed := true }
  | .st =>
    let n := (i.vec?.map Vec.length).getD 1
    some { slots := [.addr, .srcVec t n], optional := [.src .b64], relaxed := true }
  | .mov =>
    if i.args.any (· matches .vec _) then some { slots := [.any, .any] }
    else some { slots := [.dst t, .src t] }
  | .cvta => some { slots := [.dst t, .any] }
  | .isspacep => some { slots := [.dpred, .src .u64] }
  | .mapa => some { slots := [.dst t, .any, .src .u32] }
  | .getctarank => some { slots := [.dst .u32, .any] }
  | .bra => some { slots := [.label] }
  | .exit | .ret | .trap | .brkpt | .wgmmaFence | .wgmmaCommitGroup | .cpAsyncCommitGroup
  | .cpAsyncWaitAll | .cpAsyncBulkCommitGroup | .griddepcontrol => some { slots := [] }
  | .shflSync => some { slots := [.dstPred .b32, .src .b32, .src .b32, .src .b32, .src .b32] }
  | .voteSync =>
    some { slots := [if t == .pred then .dpred else .dst .b32, .pred, .src .b32] }
  | .matchSync =>
    some { slots := [if i.has .all then .dstPred .b32 else .dst .b32, .src t, .src .b32] }
  | .reduxSync => some { slots := [.dst t, .src t, .src .b32] }
  | .electSync => some { slots := [.dstPred .b32, .src .b32] }
  | .activemask => some { slots := [.dst .b32] }
  | .barWarpSync => some { slots := [.src .b32] }
  | .barSync | .barrierSync => some { slots := [.src .b32], optional := [.src .b32] }
  | .barArrive | .barrierArrive => some { slots := [.src .b32, .src .b32] }
  | .nanosleep => some { slots := [.src .u32] }
  | .wgmmaWaitGroup | .cpAsyncWaitGroup | .cpAsyncBulkWaitGroup => some { slots := [.src .u32] }
  | _ => none

/-- A target architecture (`.target sm_90a`): the number and the suffix (`a`: architecture-
specific, `f`: family-specific, or none). -/
structure Arch where
  sm : Nat
  suffix : Option Char
  deriving Repr, Inhabited

def Arch.parse? (s : String) : Option Arch := do
  guard (s.startsWith "sm_")
  let rest := (s.toList.drop 3)
  let digits := rest.takeWhile Char.isDigit
  let sm ← (String.ofList digits).toNat?
  match rest.drop digits.length with
  | [] => some { sm, suffix := none }
  | [c] => some { sm, suffix := some c }
  | _ => none

/-- The architecture an instruction requires (the "Target ISA Notes" of §9.7, for the
instruction's base form): `(sm, specific)`, where `specific` asks for the architecture-specific
(`a`) or family (`f`) target of that generation or, for `sm_90a`, exactly `sm_90a`. -/
def Op.requires : Op → Option (Nat × Bool)
  | .szext | .bmsk | .nanosleep | .matchSync | .mma | .wmmaLoad | .wmmaStore | .wmmaMma => some (70, false)
  | .tanh => some (75, false)
  | .clmad | .reduxSync | .cpAsync | .cpAsyncCommitGroup | .cpAsyncWaitGroup | .cpAsyncWaitAll
  | .cpAsyncMbarrierArrive | .applypriority | .discard | .createpolicy | .mbarrierInit
  | .mbarrierInval | .mbarrierArrive | .mbarrierArriveDrop | .mbarrierTestWait
  | .mbarrierPendingCount | .ldmatrix => some (80, false)
  | .stAsync | .redAsync | .mapa | .getctarank | .cpAsyncBulk | .cpReduceAsyncBulk
  | .cpAsyncBulkPrefetch | .cpAsyncBulkTensor | .cpReduceAsyncBulkTensor
  | .cpAsyncBulkPrefetchTensor | .cpAsyncBulkCommitGroup | .cpAsyncBulkWaitGroup
  | .multimemCpAsyncBulk | .multimemCpReduceAsyncBulk | .barrierClusterArrive
  | .barrierClusterWait | .griddepcontrol | .electSync | .mbarrierExpectTx | .mbarrierCompleteTx
  | .mbarrierTryWait | .mbarrierCheckLayout | .tensormapCpFenceproxy | .stmatrix
  | .multimemLdReduce | .multimemSt | .multimemRed => some (90, false)
  | .wgmmaMmaAsync | .wgmmaFence | .wgmmaCommitGroup | .wgmmaWaitGroup
  | .tensormapReplace => some (90, true)
  -- §9.7.21.5: sm_90a, sm_100a, sm_101a/sm_110a, sm_120a and their families.
  | .setmaxnreg => some (91, true)
  | .stBulk | .multimemStAsync | .multimemRedAsync | .fabricTryGet | .fabricTryPut
  | .fabricTryRed | .fabricTryPullred | .fabricSubmit | .fabricWait | .fabricTryAtom
  | .clusterlaunchcontrolTryCancel | .clusterlaunchcontrolQueryCancel => some (100, false)
  | .tcgen05Alloc | .tcgen05Dealloc | .tcgen05RelinquishAllocPermit | .tcgen05Ld | .tcgen05St
  | .tcgen05WaitLd | .tcgen05WaitSt | .tcgen05Cp | .tcgen05Shift | .tcgen05Mma
  | .tcgen05FenceBeforeThreadSync | .tcgen05FenceAfterThreadSync | .tcgen05Commit => some (100, true)
  | .spcompress | .spdecompress | .applypriorityAsyncBulk | .applypriorityAsyncBulkTensor =>
    some (107, true)
  | _ => none

/-- The SM version a form of an instruction requires beyond its base form, by its types and
qualifiers (the half-precision Target ISA Notes of §9.7.4: `add`/`sub`/`mul` on `.bf16`,
`.bf16x2` need sm_90; `fma.relu` and `.bf16` forms sm_80, `fma.oob` sm_90; `neg`/`abs`/`min`/`max`
on `.bf16` forms sm_80, `min`/`max` `.xorsign.abs` sm_86; `mma` by shape and type, §9.7.16.5.14). -/
def Instr.formRequires (i : Instr) : Option Nat :=
  let bf := i.types.any (· matches .bf16 | .bf16x2)
  let half := i.types.any (· matches .f16 | .f16x2 | .bf16 | .bf16x2)
  match i.op with
  | .add | .sub | .mul => if bf then some 90 else none
  | .fma =>
    if half && i.has .oob then some 90
    else if half && (bf || i.has .relu) then some 80 else none
  | .neg | .abs => if bf then some 80 else none
  | .min | .max =>
    if half && i.has .xorsign then some 86 else if half then some 80 else none
  | .mma =>
    -- §9.7.16.5.14's Target ISA Notes: binary64 beyond `.m8n8k4` needs sm_90, FP8 sm_89, the
    -- `.m16n8` integer and single-bit shapes, `.bf16`/`.tf32`, binary64 and single-bit `.and`
    -- sm_80, the rest sm_75 (`.m8n8k4` with `.f16`: sm_70).
    let a := i.types[1]?
    if a == some .f64 && !(i.shape? == some (8, 8, 4)) then some 90
    else if a == some .e4m3 || a == some .e5m2 then some 89
    else if a == some .f64 || a == some .bf16 || a == some .tf32 || i.bool? == some .and then some 80
    else if (i.shape?.map (·.1)) == some 16 && [Ty.s8, .u8, .s4, .u4, .b1].any (some · == a) then
      some 80
    else if i.shape? == some (16, 8, 16) && a == some .f16 then some 80
    else if i.shape? == some (8, 8, 4) then none else some 75
  | .ldmatrix | .stmatrix =>
    -- §9.7.16.5.15–16: `.m16n16`, `.m8n16`, `.m16n8` and `.b8` on sm_100a, sm_110a, sm_120a.
    if i.types.any (· matches .b8 | .b8x16) then some 100 else none
  | .wmmaMma | .wmmaLoad | .wmmaStore =>
    -- §9.7.16.4's Target ISA Notes: integer sm_72, sub-byte and single-bit sm_75, binary64,
    -- `.bf16`/`.tf32` and single-bit `.and` sm_80.
    let ts := i.types
    if ts.any ([Ty.f64, .bf16, .tf32].contains ·) || i.bool? == some .and then some 80
    else if ts.any ([Ty.s4, .u4, .b1].contains ·) then some 75
    else if ts.any ([Ty.s8, .u8].contains ·) then some 72 else none
  | _ => none

/-- Whether target `a` provides what `Op.requires` asks. The architecture-specific features of
`sm_90a` exist only there (`(91, true)`: an `sm_90a` feature every later architecture-specific or
family target keeps, `setmaxnreg`); those of `sm_100a` (`tcgen05`, TMEM) on the data-center Blackwell
targets `sm_100a`, `sm_101a`/`sm_110a` and `sm_103a`, not on `sm_120a` (§9.7.18's Target ISA
Notes). -/
def Arch.provides (a : Arch) : Nat × Bool → Bool
  | (90, true) => a.sm == 90 && a.suffix == some 'a'
  | (91, true) => a.sm ≥ 90 && a.suffix.isSome
  | (107, true) => a.sm == 107 && a.suffix == some 'a'
  | (100, true) => a.sm ≥ 100 && a.sm < 120 && a.suffix.isSome
  | (sm, true) => a.sm ≥ sm && a.suffix.isSome
  | (sm, false) => a.sm ≥ sm

/-- The SM version from which a deprecated instruction is gone: `shfl` and `vote` without `.sync`
are not supported on `sm_70` or higher from PTX ISA 6.4 (§9.7.10.5, §9.7.15.9). -/
def Op.removedFrom : Op → Option Nat
  | .shfl | .vote => some 70
  | _ => none

/-- Whether target `a` has instruction `op`: it provides what `Op.requires` asks and has not
dropped it (`Op.removedFrom`). -/
def Arch.has (a : Arch) (op : Op) : Bool :=
  (Op.requires op |>.all a.provides) && (Op.removedFrom op |>.all (a.sm < ·))

/-- Scopes: module names, then nested function scopes, innermost first. -/
structure Env where
  module : Std.HashMap String Sym
  scopes : List (Std.HashMap String Sym)
  /-- `%r<100>` declarations by prefix: the count and declaration. -/
  counted : List (String × Nat × VarDecl)
  labels : Std.HashMap String Unit

namespace Env

def declare (e : Env) (d : VarDecl) : Env :=
  let sym := if d.space == .reg then Sym.reg d else Sym.var d
  match d.count with
  | some n => { e with counted := (d.name, n, d) :: e.counted }
  | none =>
    match e.scopes with
    | s :: ss => { e with scopes := s.insert d.name sym :: ss }
    | [] => { e with module := e.module.insert d.name sym }

/-- `%r57` against a `%r<100>` declaration. -/
def countedLookup (e : Env) (n : String) : Option Sym :=
  e.counted.findSome? fun (p, k, d) =>
    if n.startsWith p then
      let rest := String.ofList (n.toList.drop p.length)
      match (if rest.all Char.isDigit then rest.toNat? else none) with
      | some i => if i < k then some (if d.space == .reg then .reg d else .var d) else none
      | none => none
    else none

def lookup (e : Env) (n : String) : Option Sym :=
  (e.scopes.findSome? (·.get? n)) <|> e.countedLookup n <|> e.module.get? n <|>
    (if e.labels.contains n then some .label else none)

end Env

abbrev M := StateT (Array Diag) Id

def report (line : Nat) (msg : String) : M Unit := modify (·.push ⟨line, msg⟩)

/-- The type of a register operand, if it is one. -/
def regTy? (e : Env) : Operand → Option Ty
  | .expr (.name n) =>
    match e.lookup n with
    | some (.reg d) => if d.vec.isNone then some d.ty else none
    | _ => (SReg.lookup? n none).map SReg.ty
  | .comp b c =>
    match e.lookup b with
    | some (.reg d) => if d.vec.isSome then some d.ty else none
    | _ => (SReg.lookup? b (some c)).map SReg.ty
  | _ => none

/-- Every name in an expression is declared. -/
def namesOf : Expr → List String
  | .name s => [s]
  | .un _ a | .app _ a => namesOf a
  | .bin _ a b => namesOf a ++ namesOf b
  | .cond c a b => namesOf c ++ namesOf a ++ namesOf b
  | _ => []

def operandNames : Operand → List String
  | .expr e => namesOf e
  | .comp b _ => [b]
  | .not o | .neg o => operandNames o
  | .pair a b => operandNames a ++ operandNames b
  | .vec xs | .list xs | .args xs => xs.flatMap operandNames
  | .addr (some b) _ => [b]
  | _ => []

def known (e : Env) (n : String) : Bool :=
  (e.lookup n).isSome || (SReg.lookup? n none).isSome || n == "WARP_SZ" ||
    (SReg.vectors.lookup n).isSome

/-- Register type `r` used where the instruction expects type `t` (§6.1, §9.4.1). -/
def compatible (relaxed : Bool) (r t : Ty) : Bool :=
  Ty.compatible r t ||
    (relaxed && r.bits > t.bits &&
      (t.basic == .bits || t.isInt || r.basic == .bits) && (r.basic != .float || t.basic == .bits))

def isImm : Operand → Bool
  | .expr e => (namesOf e).isEmpty
  | _ => false

def checkReg (e : Env) (line : Nat) (relaxed : Bool) (t : Ty) (o : Operand) (what : String) :
    M Unit := do
  match regTy? e o with
  | some r =>
    unless compatible relaxed r t do
      report line s!"{what}: register of type .{Keyword.name r} where .{Keyword.name t} is expected"
  | none =>
    if what.startsWith "destination" then report line s!"{what}: not a register"

def checkPred (e : Env) (line : Nat) (o : Operand) (what : String) : M Unit := do
  match o with
  | .not o => checkPred e line o what
  | .sink => pure ()
  | _ =>
    if isImm o then return
    match regTy? e o with
    | some .pred => pure ()
    | _ => report line s!"{what}: not a predicate"

def checkSlot (e : Env) (line : Nat) (relaxed : Bool) (k : Nat) (s : Slot) (o : Operand) :
    M Unit := do
  let what := s!"operand {k + 1}"
  match s, o with
  | _, .sink => pure ()
  | .any, _ => pure ()
  | .label, .expr (.name n) =>
    unless e.labels.contains n do report line s!"{what}: {n} is not a label"
  | .label, _ => report line s!"{what}: not a label"
  | .addr, .addr .. => pure ()
  | .addr, _ => report line s!"{what}: not an address"
  | .pred, o => checkPred e line o what
  | .dpred, .pair a b => do checkPred e line a what; checkPred e line b what
  | .dpred, o => checkPred e line o what
  | .dstPred t, .pair a b => do
    checkReg e line relaxed t a s!"destination {what}"; checkPred e line b what
  | .dstPred t, o => checkReg e line relaxed t o s!"destination {what}"
  | .dst t, o => checkReg e line relaxed t o s!"destination {what}"
  | .src t, o => if isImm o then pure () else checkReg e line relaxed t o what
  | .dstVec t n, .vec xs | .srcVec t n, .vec xs =>
    if xs.length != n then report line s!"{what}: {xs.length} registers where {n} are expected"
    for x in xs do
      if !(x matches .sink) && !isImm x then checkReg e line relaxed t x what
  | .dstVec t 1, o => checkReg e line relaxed t o s!"destination {what}"
  | .srcVec t 1, o => if isImm o then pure () else checkReg e line relaxed t o what
  | .dstVec _ n, .expr (.name r) | .srcVec _ n, .expr (.name r) =>
    match e.lookup r with
    | some (.reg d) =>
      unless (d.vec.map Vec.length) == some n do report line s!"{what}: {r} is not a .v{n} register"
    | _ => report line s!"{what}: {r} is not a register"
  | .dstVec .., _ | .srcVec .., _ => report line s!"{what}: not a vector of registers"

def checkInstr (arch : Option Arch) (e : Env) (line : Nat) (g : Option Guard) (i : Instr) :
    M Unit := do
  if let (some a, some req) := (arch, Op.requires i.op) then
    unless a.provides req do
      report line s!"{i.op.spelling} requires sm_{if req.1 == 91 then 90 else req.1}{if req.2 then "a" else ""} or higher"
  if let (some a, some sm) := (arch, Op.removedFrom i.op) then
    unless a.sm < sm do
      report line s!"{i.op.spelling} without .sync is not supported on sm_{sm} or higher"
  if let (some a, some sm) := (arch, Instr.formRequires i) then
    unless a.sm ≥ sm do
      let ts := ".".intercalate (i.types.map Keyword.name)
      report line s!"{i.op.spelling}.{ts} requires sm_{sm} or higher"
  if let some g := g then
    match e.lookup g.pred with
    | some (.reg d) =>
      unless d.ty == .pred do report line s!"guard {g.pred} is not a predicate"
    | _ => report line s!"guard {g.pred} is not declared"
  for o in i.args do
    for n in operandNames o do
      unless known e n do report line s!"{i.op.spelling}: {n} is not declared"
  if typesOk i == some false then
    let ts := ".".intercalate (i.types.map Keyword.name)
    report line s!"{i.op.spelling} does not take types .{ts}"
  let some sig := signature i | return
  let n := i.args.size
  if n < sig.slots.length || n > sig.slots.length + sig.optional.length then
    report line s!"{i.op.spelling}: {n} operands where {sig.slots.length} are expected"
    return
  for (s, k) in (sig.slots ++ sig.optional).zipIdx do
    if h : k < n then checkSlot e line sig.relaxed k s i.args[k]

def labelsOf : List (Stmt Instr) → List String
  | [] => []
  | .label n :: ss => n :: labelsOf ss
  | .proto n _ :: ss | .targets n _ _ :: ss => n :: labelsOf ss
  | .block b :: ss => labelsOf b ++ labelsOf ss
  | _ :: ss => labelsOf ss

def body (arch : Option Arch) (fuel : Nat) (e : Env) (ss : List (Stmt Instr)) : M Unit :=
  match fuel with
  | 0 => report 0 "blocks nested too deeply"
  | fuel + 1 => do
    let mut e := { e with scopes := {} :: e.scopes }
    for s in ss do
      match s with
      | .decl d => e := e.declare d
      | .instr g i l => checkInstr arch e l g i
      | .block b => body arch fuel e b
      | _ => pure ()

def function (arch : Option Arch) (m : Std.HashMap String Sym) (f : Function Instr) :
    M Unit := do
  let some b := f.body | return
  let e : Env := { module := m, scopes := [{}], counted := [],
                   labels := Std.HashMap.ofList ((labelsOf b).map (·, ())) }
  let e := (f.rets ++ f.params).foldl Env.declare e
  body arch 64 e b

end Check

/-- The diagnostics of a module: empty when it satisfies the rules this checker states. -/
def check (m : Module Instr) : Array Diag :=
  let syms : Std.HashMap String Sym := m.items.foldl (init := {}) fun acc it =>
    match it with
    | .var d => acc.insert d.name (if d.space == .reg then .reg d else .var d)
    | .func f => acc.insert f.name .func
    | _ => acc
  let run : Check.M Unit := do
    for it in m.items do
      if let .func f := it then Check.function (m.target.head?.bind Check.Arch.parse?) syms f
  (run.run #[]).2

end NV.PTX
