import NV.PTX.Sem.Operand
import NV.Async.Tma
import NV.Async.WgmmaNum
import NV.Async.MBarrierWord
import NV.Async.Tcgen05

/-!
# PTX semantics of the asynchronous instructions

One thread's step for `cp.async*`, `cp.async.bulk*`, `mbarrier.*` and `wgmma.*` (PTX ISA 9.5
§9.7.10.28, §9.7.15.16, §9.7.17.5), as `Effect`s over the shared async vocabulary
(`NV.Async.Op`): the thread semantics (`NV.PTX.Sem.Step`) defers to `exec?` first.

* Copies are `Effect.async` with the copy's moves and its completion: a complete-tx on the
  instruction's mbarrier (release at cluster scope, generic proxy, §9.7.10.28.5.3), or the
  thread's bulk or `cp.async` group.
* Tensor copies read their tensor map when the agent runs (`Op.tensor`): the map is accessed
  through the tensormap proxy at the generic address the instruction names (§9.7.10.28.5.3).
* mbarrier operations are read-modify-writes of the object (`MemOp.mbar`), waits acquire loads
  of it; `try_wait` returns at once, as `test_wait` (the ISA lets it return false whenever the
  time limit passes).
* `wgmma.mma_async` computes the thread's own accumulator fragment
  (`NV.Async.Wgmma.Share` with this lane), reading A and B through their descriptors.

* `tcgen05` (§9.7.18, `NV.Async.Tcgen05`): tensor memory is the `.tmem` space. `alloc` and
  `dealloc` are warp-wide and performed by the warp's lane 0 (every thread executes the
  `.aligned` instruction): an `AtomOp.findSetAlign` on the column allocator, retried until it
  succeeds, then the address stored to shared memory; `mma` is the executing thread's own
  operation (`Op.mmaT`), as are `cp` and `shift`, and `commit` an arrive-on after the thread's
  earlier `tcgen05` operations; `ld` and `st` move the executing thread's fragment.

Not modelled yet, and a fault: the `.im2col::w` modes, `.tile::scatter4`, `tcgen05` with
`.cta_group::2`.
-/

namespace NV.PTX.Async

open NV NV.Async NV.PTX.Sem

abbrev X := Except String

variable (e : Env) (t : Thread) (pi : PInstr)

def arg (k : Nat) : Arg := pi.args[k]?.getD .sink
def rd (k : Nat) (ty : Ty) : X UInt64 := t.read e (arg pi k) ty
def wr (s : Thread) (k : Nat) (ty : Ty) (v : UInt64) : X Thread := s.write (arg pi k) ty v

/-- The space and address of memory operand `k`, generic operands resolved through the windows. -/
def target (k : Nat) (sp : Option Space := pi.i.space?) : X (NV.Space × UInt64) := do
  match ← t.target e sp (arg pi k) with
  | .mem s a => return (s, a)
  | .frame _ => throw "a frame parameter is not an asynchronous operand"

/-- A shared-memory operand as the CTA-local offset it names, and the space and address of the
offset `off` of the same CTA (its own memory, or a peer's in the cluster). -/
def sharedBase (s : NV.Space) (a : UInt64) : X (UInt64 × (Nat → NV.Space × UInt64)) := do
  match s with
  | .shared => return (a, fun off => (.shared, off.toUInt64))
  | .sharedCluster =>
    match e.windows.sharedTarget a with
    | (none, off) => return (off, fun o => (.shared, o.toUInt64))
    | (some r, off) => return (off, fun o => (.sharedCluster, e.windows.clusterAddr t.ctx.clusterRank r o.toUInt64))
  | _ => throw s!"{repr s}: not shared memory"

/-- The CTA ranks a `.multicast::cluster` copy's 16-bit mask at operand `k` names, each with the
place of shared offset `o` in its CTA, seen from the executing one (§9.7.10.28.4.1: each
destination CTA gets the data at the same offset, and its mbarrier at the same offset the
complete-tx). -/
def multicastTargets (k : Nat) : X (Array (Nat → NV.Space × UInt64)) := do
  let mask := (← rd e t pi k .b16).toNat
  let own := t.ctx.clusterRank
  return ((Array.range 16).filter fun r => (mask >>> r) % 2 == 1).map fun r o =>
    if r == own then (.shared, o.toUInt64) else (.sharedCluster, e.windows.clusterAddr own r o.toUInt64)

/-- An mbarrier read-modify-write on the 8-byte object at operand `k`. -/
def mbarReq (k : Nat) (op : MBarOp) (sem : NV.Sem) (scope : Scope) : X Req := do
  let (s, a) ← target e t pi k
  unless s == .shared || s == .sharedCluster do throw "an mbarrier lives in shared memory"
  unless a % 8 == 0 do throw s!"mbarrier at {a} is not 8-byte aligned"
  return { space := s, op := .mbar op, bytes := 8, sem, scope, addrs := #[some a] }

def scopeOr (d : Scope) : Scope := pi.i.scope?.getD d

/-- A thread's request, built when it issues, with no register result. -/
def rmw (r : Req) : Effect Thread := .mem (fun _ => r) t.advance fun _ s => s

/-- The 64-bit word of a response's first two words. -/
def word64 (resp : Resp) : UInt64 :=
  (resp[0]?.getD 0).toUInt64 ||| ((resp[1]?.getD 0).toUInt64 <<< (32 : UInt64))

/-- The 64-bit state an arrive returns, written to operand 0. -/
def stateWb (resp : Resp) (s : Thread) : Thread :=
  s.writeOr (arg pi 0) .b64 (word64 resp)

def mbarrier : X (Effect Thread) := do
  let i := pi.i
  let rel := if i.sem? == some .relaxed then NV.Sem.relaxed else .release
  match i.op with
  | .mbarrierInit =>
    if i.has .layoutV1 then throw ".layout::v1 mbarriers (fabric operations) are outside the model"
    let n := (← rd e t pi 1 .b32).toNat
    return rmw t (← mbarReq e t pi 0 (.init n) .weak .cta)
  | .mbarrierInval =>
    return rmw t (← mbarReq e t pi 0 .inval .weak .cta)
  | .mbarrierExpectTx =>
    let tx := (← rd e t pi 1 .b32).toNat
    return rmw t (← mbarReq e t pi 0 (.arrive 0 tx) .relaxed (scopeOr pi .cta))
  | .mbarrierCompleteTx =>
    let tx := (← rd e t pi 1 .b32).toNat
    return rmw t (← mbarReq e t pi 0 (.completeTx tx) .relaxed (scopeOr pi .cta))
  | .mbarrierArrive | .mbarrierArriveDrop =>
    -- `mbarrier.arrive{.expect_tx} state, [addr]{, count|txCount}`; `_` discards the state.
    let drop := i.op == .mbarrierArriveDrop
    let extra ← if pi.args.size > 2 then do pure (some (← rd e t pi 2 .b32).toNat) else pure none
    let op : MBarOp :=
      if i.has .expectTx then .arrive 1 (extra.getD 0) drop else .arrive (extra.getD 1) 0 drop
    let req ← mbarReq e t pi 1 op rel (scopeOr pi .cta)
    return .mem (fun _ => req) t.advance (stateWb pi)
  | .mbarrierTestWait | .mbarrierTryWait =>
    let (s, a) ← target e t pi 1
    let req : Req := { space := s, op := .ld, bytes := 8, sem := .acquire, scope := scopeOr pi .cta,
                       addrs := #[some a] }
    let v ← rd e t pi 2 (if i.has .parity then .b32 else .b64)
    return .mem (fun _ => req) t.advance fun resp s =>
      let w := word64 resp
      let done := if i.has .parity then MBarrierWord.parityDone w v.toNat else MBarrierWord.stateDone w v
      s.writeOr (arg pi 0) .pred (if done then 1 else 0)
  | .mbarrierPendingCount =>
    let st ← rd e t pi 1 .b64
    return .next (← wr pi t.advance 0 .b32 (MBarrierWord.pendingCount st).toUInt64)
  | op => throw s!"{op.spelling}: not an mbarrier instruction"

def issue (op : NV.Async.Op) : Effect Thread := .async (fun _ => op) t.advance fun _ s => s

/-- `cp.async` (§9.7.10.28.3.1): `cp.async.ca|cg.shared{::cta}.global [dst], [src], cp-size{,
src-size}{, ignore-src}`. Bytes past `src-size` are zero; `ignore-src` true reads nothing. -/
def cpAsync : X (Effect Thread) := do
  let (_, d) ← target e t pi 0 (some .shared)
  let (_, s) ← target e t pi 1 (some .global)
  let size := (← rd e t pi 2 .u32).toNat
  -- Optional operands: `src-size` (an integer) and `ignore-src` (a predicate).
  let isPred : Arg → Bool
    | .reg r => r.ty == .pred
    | .not _ => true
    | _ => false
  let mut srcSize := size
  let mut ignore := false
  for j in [3:pi.args.size] do
    if isPred (arg pi j) then ignore := (← rd e t pi j .pred) != 0
    else srcSize := (← rd e t pi j .u32).toNat
  let valid := if ignore then 0 else min srcSize size
  let move : Move :=
    if valid == size then
      { src := asyncReq .global .ld size #[some s], dsts := #[asyncReq .shared .st size #[some d]] }
    else
      { src := asyncReq .global .ld 1 ((Array.range size).map fun j => if j < valid then some (s + j.toUInt64) else none)
        dsts := #[asyncReq .shared .st 1 ((Array.range size).map fun j => some (d + j.toUInt64))] }
  return issue t (NV.Async.Op.copy { moves := #[move], group := some .cpAsync })

/-- `cp.async.bulk` (§9.7.10.28.4.1): global to shared with complete-tx, shared to global in
the bulk group, or shared to a cluster CTA's shared memory with complete-tx. -/
def cpAsyncBulk : X (Effect Thread) := do
  let i := pi.i
  match i.spaces with
  | dst :: .global :: _ =>
    let (ds, da) ← target e t pi 0 (some dst)
    let (_, src) ← target e t pi 1 (some .global)
    let size := (← rd e t pi 2 .u32).toNat
    let (off, at_) ← sharedBase e t ds da
    let ps ← bulkPieces src off.toNat size
    let (ms, ma) ← target e t pi 3 (some .sharedCluster)
    let sig (s : NV.Space) (a : UInt64) : Req :=
      { space := s, op := .mbar (.completeTx size), bytes := 8, sem := .release,
        scope := .cluster, addrs := #[some a] }
    if i.has .multicastCluster then
      let dsts ← multicastTargets e t pi 4
      let (_, moff) := e.windows.sharedTarget ma
      return issue t (NV.Async.Op.copy (loadCopy ps dsts (dsts.map fun d =>
        let (s, a) := d moff.toNat; sig s a)))
    return issue t (NV.Async.Op.copy (loadCopy ps #[at_] #[sig ms ma]))
  | .global :: src :: _ =>
    let (_, dst) ← target e t pi 0 (some .global)
    let (ss, sa) ← target e t pi 1 (some src)
    let size := (← rd e t pi 2 .u32).toNat
    let (off, at_) ← sharedBase e t ss sa
    let ps ← bulkPieces dst off.toNat size
    return issue t (NV.Async.Op.copy (storeCopy ps (at_ 0).1 fun o => (at_ o).2))
  | .sharedCluster :: src :: _ =>
    -- `….shared::cluster.shared::cta.mbarrier::complete_tx::bytes [dstMem], [srcMem], size,
    -- [mbar]`: into a cluster CTA's shared memory, its mbarrier signalled with the bytes.
    let dst ← target e t pi 0 (some .sharedCluster)
    let srcAt ← target e t pi 1 (some src)
    let size := (← rd e t pi 2 .u32).toNat
    let (ms, ma) ← target e t pi 3 (some .sharedCluster)
    let sig : Req := { space := ms, op := .mbar (.completeTx size), bytes := 8, sem := .release,
                       scope := .cluster, addrs := #[some ma] }
    return issue t (NV.Async.Op.copy (← elementCopy srcAt dst size 16 .st #[sig]))
  | _ => throw "cp.async.bulk between these spaces is not modelled"

/-- `cp.reduce.async.bulk.global.shared::cta.bulk_group.redOp.type [dstMem], [srcMem], size`
(§9.7.10.28.4.2): each element of the shared source reduced into global memory, in the bulk
group; `….shared::cluster.shared::cta.mbarrier::complete_tx::bytes… [dstMem], [srcMem], size,
[mbar]`: into a cluster CTA's shared memory, its mbarrier signalled with the bytes. -/
def cpReduceAsyncBulk : X (Effect Thread) := do
  let i := pi.i
  match i.spaces with
  | .global :: src :: _ =>
    let some o := i.red? | throw "no reduction operator"
    let ty := i.types.headD .b32
    let some aty := atomType ty | throw s!"reductions on .{Keyword.name ty} are not modelled"
    let (_, dst) ← target e t pi 0 (some .global)
    let (ss, sa) ← target e t pi 1 (some src)
    let size := (← rd e t pi 2 .u32).toNat
    let (off, at_) ← sharedBase e t ss sa
    let ps ← bulkPieces dst off.toNat size
    return issue t (NV.Async.Op.copy
      (storeCopy (elementPieces ty.bytes ps) (at_ 0).1 (fun o => (at_ o).2) (.red (atomOp o) aty)))
  | .sharedCluster :: src :: _ =>
    let some o := i.red? | throw "no reduction operator"
    let ty := i.types.headD .b32
    let some aty := atomType ty | throw s!"reductions on .{Keyword.name ty} are not modelled"
    let dst ← target e t pi 0 (some .sharedCluster)
    let srcAt ← target e t pi 1 (some src)
    let size := (← rd e t pi 2 .u32).toNat
    let (ms, ma) ← target e t pi 3 (some .sharedCluster)
    let sig : Req := { space := ms, op := .mbar (.completeTx size), bytes := 8, sem := .release,
                       scope := .cluster, addrs := #[some ma] }
    return issue t (NV.Async.Op.copy (← elementCopy srcAt dst size ty.bytes (.red (atomOp o) aty) #[sig]))
  | _ => throw "cp.reduce.async.bulk between these spaces is not modelled"

/-- The tensor map and coordinates of a `[tensorMap, {c0, …}]` operand: a load of the map's 128
bytes (tensormap proxy, here the generic one) and the signed coordinates. -/
def tensorOperand (k : Nat) : X (Req × Array Int) := do
  let .list xs := arg pi k | throw "expected [tensorMap, {coordinates}]"
  let some base := xs[0]? | throw "no tensor map"
  let some (Arg.vec cs) := (xs[1]? : Option Arg) | throw "no coordinates"
  let (s, a) ← match base with
    | .sym (.mem s x) off => pure (s, x + UInt64.ofInt off)
    | .reg _ => pure (e.windows.resolve (← t.read e base .b64))
    | _ => throw "tensor map operand"
  let coords ← cs.mapM fun c => do pure (toInt true 32 (← t.read e c .s32))
  let desc : Req := { space := s, op := .ld, bytes := 16,
                      addrs := (Array.range 8).map fun j => some (a + (16 * j).toUInt64) }
  return (desc, coords)

def mapOfResp (r : Resp) : X TensorMap :=
  let bytes := ByteArray.mk ((Array.range TensorMap.size).map fun j =>
    ((r[j / 4]?.getD 0) >>> (8 * (j % 4)).toUInt32).toUInt8)
  match TensorMap.decode bytes with
  | some m => .ok m
  | none => .error "no tensor map at the operand's address"

/-- `cp.async.bulk.tensor` (§9.7.10.28.5.3), tiled mode, and `.tile::gather4` (§5.5.3.4: the
coordinates `{col, row0, row1, row2, row3}`, four one-row boxes as one tile, `TensorMap.gather4`). -/
def cpAsyncBulkTensor : X (Effect Thread) := do
  let i := pi.i
  if i.has .tileScatter4 then throw "this load mode is not modelled yet"
  let gather := i.has .tileGather4
  let im2col := i.has .im2col
  match i.spaces with
  | dst :: .global :: _ =>
    let (ds, da) ← target e t pi 0 (some dst)
    let (off, at_) ← sharedBase e t ds da
    let (desc, coords) ← tensorOperand e t pi 1
    let (ms, ma) ← target e t pi 2 (some .sharedCluster)
    -- `.im2col`: the im2col offsets after the mbarrier (`.u16` each, `W` first), then the mask.
    let offsets ← if !im2col then pure #[] else
      match arg pi 3 with
      | .vec xs => xs.mapM fun x => do pure (← t.read e x .u16).toNat
      | x => do pure #[(← t.read e x .u16).toNat]
    let dsts ← if i.has .multicastCluster then multicastTargets e t pi (if im2col then 4 else 3)
      else pure #[at_]
    let (_, moff) := e.windows.sharedTarget ma
    let build (r : Resp) : X Copy := do
      let m ← mapOfResp r
      if im2col != m.im2col.isSome then throw "the copy's mode is not the tensor map's"
      let ps ← if im2col then m.im2colTile coords offsets off.toNat
        else if gather then m.gather4 coords[0]! (coords.extract 1 5) off.toNat
        else m.tile coords off.toNat
      let bytes := if gather then 4 * m.boxBytes else m.boxBytes
      let sig (s : NV.Space) (a : UInt64) : Req :=
        { space := s, op := .mbar (.completeTx bytes), bytes := 8, sem := .release,
          scope := .cluster, addrs := #[some a] }
      return loadCopy ps dsts (if i.has .multicastCluster then
        dsts.map fun d => let (s, a) := d moff.toNat; sig s a else #[sig ms ma])
    return issue t (NV.Async.Op.tensor { desc, build })
  | .global :: src :: _ =>
    if gather then throw ".tile::gather4 is a load mode"
    let (desc, coords) ← tensorOperand e t pi 0
    let (ss, sa) ← target e t pi 1 (some src)
    let (off, at_) ← sharedBase e t ss sa
    let build (r : Resp) : X Copy := do
      let ps ← (← mapOfResp r).tile coords off.toNat
      return storeCopy ps (at_ 0).1 fun o => (at_ o).2
    return issue t (NV.Async.Op.tensor { desc, build, group := some .bulk })
  | _ => throw "cp.async.bulk.tensor between these spaces is not modelled yet"

/-- `cp.reduce.async.bulk.tensor.dim.global.shared::cta.redOp.tile.bulk_group [tensorMap,
coords], [srcMem]` (§9.7.10.28.5.4): each element of the shared box reduced into the tensor in
global memory, the element type the tensor map's, in the bulk group. -/
def cpReduceAsyncBulkTensor : X (Effect Thread) := do
  let i := pi.i
  if i.has .im2colNoOffs || i.has .im2colNoOffsW then throw "this load mode is not modelled yet"
  let some o := i.red? | throw "no reduction operator"
  match i.spaces with
  | .global :: src :: _ =>
    let (desc, coords) ← tensorOperand e t pi 0
    let (ss, sa) ← target e t pi 1 (some src)
    let (off, at_) ← sharedBase e t ss sa
    let build (r : Resp) : X Copy := do
      let m ← mapOfResp r
      let some aty := tensorRedType m.dtype (atomOp o)
        | throw s!".{repr o} on a {repr m.dtype} tensor is not one the ISA allows"
      let some eb := m.dtype.bytes? | throw "packed tensor types do not reduce"
      let ps ← m.tile coords off.toNat
      return storeCopy (elementPieces eb ps) (at_ 0).1 (fun o => (at_ o).2) (.red (atomOp o) aty)
    return issue t (NV.Async.Op.tensor { desc, build, group := some .bulk })
  | _ => throw "cp.reduce.async.bulk.tensor goes from shared::cta to global"

/-- The field a `tensormap.replace` names. -/
def replaceFieldOf (i : Instr) : X TensorMap.Field :=
  if i.has .globalAddress then pure .globalAddress else if i.has .rank then pure .rank
  else if i.has .boxDim then pure .boxDim else if i.has .globalDim then pure .globalDim
  else if i.has .elementStride then pure .elementStride else if i.has .globalStride then pure .globalStride
  else if i.has .elemtype then pure .elemtype else if i.has .interleaveLayout then pure .interleaveLayout
  else if i.has .fillMode then pure .fillMode else if i.has .swizzleMode then pure .swizzleMode
  else if i.has .swizzleAtomicity then pure .swizzleAtomicity
  else throw "tensormap.replace: no field"

/-- 128 bytes as 32 little-endian words, from the number they spell, and back. -/
def wordsOfNat (n : Nat) : Array UInt32 := (Array.range 32).map fun j => (n / 2 ^ (32 * j) % 2 ^ 32).toUInt32
def natOfWords (ws : Array UInt32) : Nat := (List.range 32).foldr (fun j acc => (ws[j]?.getD 0).toNat + 2 ^ 32 * acc) 0

/-- A 128-byte object at `a` in space `s`, as one request of eight 16-byte lanes. -/
def block128 (s : NV.Space) (a : UInt64) (op : MemOp) (data : Array UInt32 := #[]) : Req :=
  let addrs := (Array.range 8).map fun j => some (a + (16 * j).toUInt64)
  { space := s, op, bytes := 16, addrs, data }

/-- `tensormap.replace.tile.field.ss.b1024.type [addr], {ord,} new_val` (§9.7.10.29): a weak
read of the 128-byte map, then a weak write of it with the field replaced (two steps of the
thread). -/
def tensormapReplace : X (Effect Thread) := do
  let (s, a) ← target e t pi 0
  unless a % 128 == 0 do throw s!"tensor map at {a} is not 128-byte aligned"
  let f := t.frame
  if f.micro == 0 then
    return .mem (fun _ => block128 s a .ld) t fun resp s =>
      s.setFrame { s.frame with micro := 1, scratch := resp.map (·.toUInt64) }
  let ty := pi.i.types.getLastD .b32
  let indexed := pi.i.has .boxDim || pi.i.has .globalDim || pi.i.has .globalStride || pi.i.has .elementStride
  let ord ← if indexed then do pure (← rd e t pi 1 .b32).toNat else pure 0
  let v ← rd e t pi (if indexed then 2 else 1) ty
  let n' ← TensorMap.replace (← replaceFieldOf pi.i) ord v.toNat (natOfWords (f.scratch.map (·.toUInt32)))
  return .mem (fun _ => block128 s a .st (wordsOfNat n')) t fun _ s =>
    (s.setFrame { s.frame with micro := 0, scratch := #[] }).advance

/-- `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.scope.sync.aligned
[dst], [src], 128` (§9.7.15.17): a generic copy of the 128 bytes, then the release side of the
proxy pattern. Tensor maps are read in the generic proxy here (see `tensorOperand`), so that is a
release fence at the scope. -/
def tensormapCpFenceproxy : X (Effect Thread) := do
  let (_, d) ← target e t pi 0 (some .global)
  let (_, src) ← target e t pi 1 (some .shared)
  unless (← rd e t pi 2 .u32) == 128 do throw "the size must be 128"
  let f := t.frame
  match f.micro with
  | 0 => return .mem (fun _ => block128 .shared src .ld) t fun resp s =>
      s.setFrame { s.frame with micro := 1, scratch := resp.map (·.toUInt64) }
  | 1 => return .mem (fun _ => block128 .global d .st (f.scratch.map (·.toUInt32))) t fun _ s =>
      s.setFrame { s.frame with micro := 2, scratch := #[] }
  | _ => return .sync (.fence false (pi.i.scope?.getD .sys)) (t.setFrame { t.frame with micro := 0 }).advance

/-- The words of operand `k`: `n` elements of type `ty` (`.b128` as four words). -/
def valueWords (k : Nat) (ty : Ty) (n : Nat) : X (Array UInt32) := do
  let xs ← Thread.elems (arg pi k) n
  let mut out := #[]
  for x in xs do
    if ty.bits > 64 then
      let (lo, hi) ← t.read128 e x
      out := out ++ #[lo.toUInt32, (lo >>> 32).toUInt32, hi.toUInt32, (hi >>> 32).toUInt32]
    else
      let v ← t.read e x ty
      out := if ty.bits == 64 then out ++ #[v.toUInt32, (v >>> 32).toUInt32] else out.push v.toUInt32
  return out

/-- `st.async` and `red.async` with mbarrier completion (§9.7.10.12, §9.7.15.7):
`[a], b, [mbar]` in `.shared::cluster`, the value stored (or reduced) by the async unit in the
generic proxy, then a complete-tx of its bytes on `mbar` (release, cluster scope). Both operands
belong to one CTA of the cluster, other than the executing one (the ISA leaves anything else
undefined). The `.release.gpu|sys.global` forms need sm_100 and no target of ours has them. -/
def asyncStore : X (Effect Thread) := do
  let i := pi.i
  unless i.has .mbarrierCompleteTxBytes do
    throw "the .release.global forms need sm_100 or higher: not on the modelled targets"
  let ty := i.types.headD .b32
  let n := (i.vec?.map Vec.length).getD 1
  let (s, a) ← target e t pi 0 (some .sharedCluster)
  let (ms, ma) ← target e t pi 2 (some .sharedCluster)
  let words ← valueWords e t pi 1 ty n
  let bytes := 4 * words.size
  unless a % bytes.toUInt64 == 0 do throw s!"address {a} is not aligned to {bytes} bytes"
  let op : MemOp ← if i.op == .redAsync then do
      let some o := i.red? | throw "no reduction operator"
      let some aty := atomType ty | throw s!"reductions on .{Keyword.name ty} are not modelled"
      pure (.red (atomOp o) aty)
    else pure .st
  let dst : Req := { space := s, op, bytes, addrs := #[some a] }
  let sig : Req := { space := ms, op := .mbar (.completeTx bytes), bytes := 8, sem := .release,
                     scope := .cluster, addrs := #[some ma] }
  return issue t (NV.Async.Op.copy { moves := #[{ src := default, dsts := #[dst], imm := words }],
                                      signals := #[sig] })

/-- `clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes
{.multicast::cluster::all}.b128 [addr], [mbar]` (§9.7.15.18): the model launches every cluster,
so every request fails, which the ISA allows (it cancels only a cluster not yet running). The
16-byte response, all zeros (`query_cancel`'s failure), is written by weak async-proxy writes,
to every CTA of the cluster with `.multicast::cluster::all`, each followed by a complete-tx of 16
on that CTA's mbarrier. -/
def tryCancel : X (Effect Thread) := do
  let (s, a) ← target e t pi 0 (some .shared)
  let (ms, ma) ← target e t pi 1 (some .shared)
  unless a % 16 == 0 do throw s!"response address {a} is not 16-byte aligned"
  let (off, at_) ← sharedBase e t s a
  let (moff, mat) ← sharedBase e t ms ma
  let ranks := if pi.i.has .multicastClusterAll then
      let (x, y, z) := t.ctx.clusterNctaid
      (List.range (x * y * z)).map some
    else [none]
  let place (r : Option Nat) (at' : Nat → NV.Space × UInt64) (o : UInt64) : NV.Space × UInt64 :=
    match r with
    | none => at' o.toNat
    | some r => (.sharedCluster, e.windows.clusterAddr t.ctx.clusterRank r o)
  let dsts := ranks.toArray.map fun r =>
    let (sp, x) := place r at_ off
    asyncReq sp .st 16 #[some x]
  let sigs := ranks.toArray.map fun r =>
    let (sp, x) := place r mat moff
    ({ space := sp, op := .mbar (.completeTx 16), bytes := 8, sem := .release, scope := .cluster,
       addrs := #[some x] } : Req)
  return issue t (NV.Async.Op.copy { moves := #[{ src := default, dsts, imm := #[0, 0, 0, 0] }],
                                      signals := sigs })

/-- `clusterlaunchcontrol.query_cancel` (§9.7.15.19) on a response: `.is_canceled` is false for
the model's responses (`tryCancel`); the first CTA of a cancelled cluster exists only on success. -/
def queryCancel : X (Effect Thread) := do
  let (lo, hi) ← t.read128 e (arg pi 1)
  let canceled := lo != 0 || hi != 0
  if pi.i.has .isCanceled then return .next (← wr pi t.advance 0 .pred (if canceled then 1 else 0))
  throw "get_first_ctaid of a response that did not cancel a cluster is undefined"

def inType : Ty → Option Wgmma.InType
  | .f16 => some .f16 | .bf16 => some .bf16 | .tf32 => some .tf32 | .e4m3 => some .e4m3
  | .e5m2 => some .e5m2 | .s8 => some .s8 | .u8 => some .u8 | .b1 => some .b1
  | _ => none

/-- `wgmma.mma_async` (§9.7.17.5.2): this thread's fragment. With A in registers the warp's A
registers are gathered first (the warp's sixteen rows of A are its lanes' fragments, §9.7.17.5.1),
leaving the program counter in place; the operands after B's descriptor then lack `imm-trans-a`. -/
def wgmma : X (Effect Thread) := do
  let i := pi.i
  let some (64, n, k) := i.shape? | throw "wgmma shape"
  let some acc := (match i.types.head? with
    | some .f32 => some Wgmma.Acc.f32 | some .f16 => some .f16 | some .s32 => some .s32
    | _ => none) | throw "accumulator type"
  -- `.s32.b1.b1.and.popc` names three types; the others four (`.dtype.atype.btype`, with the
  -- accumulator's type first).
  let tys := i.types
  let some ta := (tys[1]?).bind inType | throw "A type"
  let some tb := (tys[2]?).bind inType | throw "B type"
  if ta == .b1 && !(pi.i.bool? == some .and) then throw "single-bit wgmma is .and.popc"
  if (if ta == .b1 then 256 else ta.k) != k then throw s!"K = {k} for .{repr ta}"
  let .vec ds := arg pi 0 | throw "accumulator vector"
  let words := Wgmma.accWords acc n
  if ds.size != words then throw s!"{ds.size} accumulator registers for {words}"
  let regsA := !((arg pi 1) matches .reg _)
  let f := t.frame
  if regsA && f.micro == 0 then
    let .vec xs := arg pi 1 | throw "A's registers"
    unless xs.size == 4 do throw s!"{xs.size} A registers for 4"
    let contrib ← xs.mapM fun x => do pure (← t.read e x .b32).toUInt32
    return .collect .gather 0xFFFFFFFF contrib t fun resp s =>
      s.setFrame { s.frame with micro := 1, scratch := resp.map (·.toUInt64) }
  let scaleD := (← rd e t pi 3 .pred) != 0
  let float := !ta.isInt
  let imm (j : Nat) : X Int := do pure (toInt true 32 (← rd e t pi j .s32))
  -- The immediates after scale-d, which only the float forms have (`←` runs regardless of `&&`).
  let immIf (b : Bool) (j : Nat) : X Int := if b then imm j else pure 0
  let negA := float && (← immIf float 4) == -1
  let negB := float && (← immIf float 5) == -1
  let transA := float && !regsA && (← immIf (float && !regsA) 6) == 1
  let transB := float && (← immIf float (if regsA then 6 else 7)) == 1
  let lane := t.ctx.laneid
  let a : Wgmma.ASource ← if regsA then
      pure (.regs ((Array.range 32).map fun l =>
        (f.scratch.extract (4 * l) (4 * l + 4)).map (·.toUInt32)))
    else pure (.smem (Wgmma.Desc.decode (← rd e t pi 1 .b64)) (!transA))
  let share : Wgmma.Share :=
    { n, ty := ta, tyB := tb, acc, warp := t.ctx.warpid % 4, a,
      b := Wgmma.Desc.decode (← rd e t pi 2 .b64), bKMajor := !transB, negA, negB, scaleD,
      lanes := #[lane] }
  let plan ← share.plan (·.toUInt64)
  let c0 ← ds.mapM fun d => do pure (← t.read e d .b32).toUInt32
  let t := t.setFrame { f with micro := 0, scratch := #[] }
  return .async (fun _ => NV.Async.Op.mma (plan.mma share c0)) t.advance fun out s => Id.run do
    let mut s := s
    for j in [0:words] do
      s := s.writeOr ds[j]! .b32 (out[j]?.getD 0).toUInt64
    return s

/-- A `tcgen05` address operand `[r]` or `[r+off]`: its 32-bit value. -/
def taddr (k : Nat) : X Nat := do
  match arg pi k with
  | .addr (.reg r) off => return ((t.slot r.slot + UInt64.ofInt off) % 2 ^ 32).toNat
  | .addr .imm off => return off.toNat
  | _ => throw "a tensor-memory address operand"

/-- A `.cta_group::2` operation's CTA pair: the issuing CTA's cluster rank and its peer's
(§9.7.18.5.1). The model has the even CTA of the pair issue it, as `Tcgen05.Inst.pair` lays the
pair out (rows 0–127 the issuer's). -/
def pairOf : X (Option (Nat × Nat)) := do
  if !pi.i.has .ctaGroup2 then return none
  let rank := t.ctx.clusterRank
  if rank % 2 != 0 then throw ".cta_group::2 from the odd CTA of its pair is not modelled"
  return some (rank, rank + 1)

/-- A `.cta_group::2` operation's addresses of its CTAs' shared memory: the cluster window. -/
def clusterOf (rank a : Nat) : UInt64 := e.windows.clusterAddr t.ctx.clusterRank rank a.toUInt64

/-- The column allocator's word in the `.tmem` space. -/
def allocReq (op : AtomOp) (v : Nat) : Req :=
  { space := .tmem, op := .atom op .b32, bytes := 4, addrs := #[some Tcgen05.allocAddr.toUInt64],
    data := #[v.toUInt32] }

/-- `tcgen05.alloc.cta_group::N.sync.aligned.shared::cta.b32 [dst], ncols` (§9.7.18.7.1). With
`::2` a warp of each CTA of the pair executes it, each claiming in its own CTA's allocator: the
two claim the same columns when the pair allocates alike (inferred; the ISA gives the pair one
allocation). -/
def tcAlloc : X (Effect Thread) := do
  if t.ctx.laneid != 0 then return .next t.advance
  let n := (← rd e t pi 1 .u32).toNat
  unless n ≥ Tcgen05.granule && n ≤ Tcgen05.columns && n &&& (n - 1) == 0 do
    throw s!"{n} columns: a power of two from 32 to 512"
  let g := n / Tcgen05.granule
  let f := t.frame
  if f.micro == 0 then
    -- Claim the columns; while none are free the thread tries again (the allocation blocks).
    return .mem (fun _ => allocReq .findSetAlign g) t fun resp s =>
      match Tcgen05.findSetAlign? (resp[0]?.getD 0) g with
      | some first => s.setFrame { s.frame with micro := 1, scratch := #[(first * Tcgen05.granule).toUInt64] }
      | none => s
  let (_, d) ← target e t pi 0 (some .shared)
  let col := (f.scratch[0]?.getD 0).toUInt32
  return .mem (fun _ => { space := .shared, op := .st, bytes := 4, addrs := #[some d], data := #[col] })
    (t.setFrame { f with micro := 0, scratch := #[] }).advance fun _ s => s

/-- `tcgen05.dealloc.cta_group::N.sync.aligned.b32 taddr, ncols`: each CTA frees its own. -/
def tcDealloc : X (Effect Thread) := do
  if t.ctx.laneid != 0 then return .next t.advance
  let a := (← rd e t pi 0 .u32).toNat
  let n := (← rd e t pi 1 .u32).toNat
  let g := n / Tcgen05.granule
  let first := Tcgen05.colOf a / Tcgen05.granule
  let mask := ((2 ^ g - 1) <<< first) % 2 ^ 32
  return .mem (fun _ => allocReq .and (2 ^ 32 - 1 - mask)) t.advance fun _ s => s

def tcKind : X Tcgen05.Kind := do
  if pi.i.has .kindF16 then return .f16
  if pi.i.has .kindTf32 then return .tf32
  if pi.i.has .kindF8f6f4 then return .f8f6f4
  if pi.i.has .kindI8 then return .i8
  if pi.i.has .kindMxf8f6f4 then return .mxf8f6f4
  if pi.i.has .kindMxf4 then return .mxf4
  if pi.i.has .kindMxf4nvf4 then return .mxf4nvf4
  throw "this .kind is not modelled"

/-- Scale factors a row of A has in one block-scaled MMA (Table 68): `.scale_vec::NX`, `.block16`
(four) or `.block32` (one under `mxf8f6f4`, else two); without either, the kind's fewest. -/
def tcScaleVec (k : Tcgen05.Kind) : Nat :=
  if pi.i.has .scaleVec4X || pi.i.has .block16 then 4
  else if pi.i.has .scaleVec2X then 2
  else if pi.i.has .scaleVec1X then 1
  else if k == .mxf8f6f4 then 1 else 2

/-- `tcgen05.mma.cta_group::1.kind [d], a-desc|[a], b-desc, idesc, enable-input-d`,
`.kind.block_scale [d], a-desc|[a], b-desc, idesc, [scale-A], [scale-B], enable-input-d`
(§9.7.18.10.9) and `tcgen05.mma.ws.cta_group::1.kind{.collector_usage} [d], a-desc, b-desc, idesc,
enable-input-d{, zero-column-mask-desc}` (§9.7.18.10.10.3): the executing thread's MMA. -/
def tcMma : X (Effect Thread) := do
  let pair ← pairOf t pi
  if pi.i.has .sp then throw "sparse MMAs are not modelled"
  let ws := pi.i.has .ws
  let kind ← tcKind pi
  if kind.scaled != pi.i.has .blockScale then throw ".block_scale goes with .kind::mxf8f6f4, mxf4 and mxf4nvf4"
  let d ← taddr t pi 0
  let a ← match arg pi 1 with
    | .addr .. => pure (Tcgen05.ASrc.tmem (← taddr t pi 1))
    | _ => pure (.smem (← Tcgen05.sdesc (← rd e t pi 1 .b64)))
  let b ← Tcgen05.sdesc (← rd e t pi 2 .b64)
  let word := (← rd e t pi 3 .b32).toNat
  let (idesc, scales, en) ← if kind.scaled then do
      let (idesc, form) ← Tcgen05.IDesc.decodeScaled kind (tcScaleVec pi kind) word
      let scales : Tcgen05.Scales := { form, a := ← taddr t pi 4, b := ← taddr t pi 5 }
      pure (idesc, some scales, 6)
    else pure (Tcgen05.IDesc.decode word, none, 4)
  let masked := ws && pi.args.size == en + 2
  if pi.args.size != en + 1 && !masked then
    throw "operands after enable-input-d (disable-output-lane, scale) are not modelled"
  let accumulate := (← rd e t pi en .pred) != 0
  let zeroMask ← if masked then (·.toNat) <$> rd e t pi (en + 1) .b64 else pure 0
  let inst : Tcgen05.Inst := { kind, idesc, a, b, d, accumulate, scales, pair, ws, zeroMask }
  let op ← inst.op (·.toUInt64) (clusterOf e t)
  let op := if e.skipMma then { op with result := fun _ => Array.replicate (idesc.m * idesc.n) 0 } else op
  return issue t (.mmaT op)

/-- `tcgen05.cp.cta_group::1.shape{.multicast}{.b8x16.src_fmt} [taddr], s-desc` (§9.7.18.9.2). -/
def tcCp : X (Effect Thread) := do
  let pair ← pairOf t pi
  let has := pi.i.has
  let shape ← if has .shape128x256b then pure Tcgen05.CpShape.s128x256b
    else if has .shape4x256b then pure .s4x256b
    else if has .shape128x128b then pure .s128x128b
    else if has .shape64x128b && has .warpx2Rows02x13 then pure .s64x128b02_13
    else if has .shape64x128b && has .warpx2Rows01x23 then pure .s64x128b01_23
    else if has .shape32x128b && has .warpx4 then pure .s32x128bx4
    else throw "tcgen05.cp: a shape with its multicast"
  let unpack := if has .b6x16P32 then Tcgen05.Unpack.b6 else if has .b4x16P64 then .b4 else .none
  let cp : Tcgen05.Cp := { shape, unpack, src := ← Tcgen05.sdesc (← rd e t pi 1 .b64), dst := ← taddr t pi 0,
                           pair }
  return issue t (.mmaT (← cp.op (·.toUInt64) (clusterOf e t)))

/-- `tcgen05.shift.cta_group::1.down [taddr]` (§9.7.18.9.3). -/
def tcShift : X (Effect Thread) := do
  return issue t (.mmaT (← Tcgen05.shiftOp (← taddr t pi 0) (← pairOf t pi)))

/-- `tcgen05.commit.cta_group::N.mbarrier::arrive::one{.shared::cluster}{.multicast::cluster}.b64
[mbar]{, ctaMask}`: once the thread's earlier `tcgen05` operations have completed, an arrive-on on
the mbarrier, or with `.multicast::cluster` on the mbarrier at its offset in each CTA of the mask
(§9.7.18.6.3). -/
def tcCommit : X (Effect Thread) := do
  discard <| pairOf t pi
  let sig ← mbarReq e t pi 0 (.arrive 1) .release .cluster
  let signals ← if pi.i.has .multicastCluster then do
      let (s, a) ← target e t pi 0
      let (off, _) ← sharedBase e t s a
      pure ((← multicastTargets e t pi 1).map fun place =>
        let (sp, addr) := place off.toNat
        { sig with space := sp, addrs := #[some addr] })
    else pure #[sig]
  return issue t (.copy { moves := #[], signals, group := some .tcgen05, afterGroup := true })

def tcShape : X Tcgen05.Shape := do
  if pi.i.has .shape32x32b then return .s32x32b
  if pi.i.has .shape16x64b then return .s16x64b
  if pi.i.has .shape16x128b then return .s16x128b
  if pi.i.has .shape16x256b then return .s16x256b
  if pi.i.has .shape16x32bx2 then return .s16x32bx2
  throw "shape"

/-- `tcgen05.ld.sync.aligned.shape.num{.pack::16b}.b32 r, [taddr]{, half}` and `tcgen05.st`
(§9.7.18.8.3–4): the executing thread's fragment. -/
def tcMove (ld : Bool) : X (Effect Thread) := do
  if ld && (pi.i.red?.isSome || pi.i.has .spcompress) then throw "tcgen05.ld.red and .spcompress are not modelled"
  let shape ← tcShape pi
  let num := pi.i.num?.getD 1
  let pack := pi.i.has .pack16b || pi.i.has .unpack16b
  let (ri, ai) := if ld then (0, 1) else (1, 0)
  let addr ← taddr t pi ai
  let half ← if shape == .s16x32bx2 then pure (← rd e t pi 2 .u32).toNat else pure 0
  let lane := t.ctx.laneid
  let warp := t.ctx.warpid
  let regs := shape.regs * num
  let rs ← Thread.elems (arg pi ri) regs
  if ld then
    let (r, _) ← Tcgen05.moveReq .ld shape num half addr warp pack (threads := (· == lane))
    return .mem (fun _ => r) t.advance fun resp s => Id.run do
      let ws := Tcgen05.unpackLoad resp lane regs pack
      let mut s := s
      for j in [0:regs] do
        s := s.writeOr rs[j]! .b32 (ws[j]!).toUInt64
      return s
  let mine ← rs.mapM fun x => do pure (← t.read e x .b32).toUInt32
  let data := (Array.range 32).flatMap fun l => if l == lane then mine else Array.replicate regs 0
  let (r, _) ← Tcgen05.moveReq .st shape num half addr warp pack data (threads := (· == lane))
  return .mem (fun _ => r) t.advance fun _ s => s

/-- The step of an asynchronous instruction, `none` for any other instruction. -/
def exec? : Option (Effect Thread) :=
  let r : Option (X (Effect Thread)) := match pi.i.op with
    | .mbarrierInit | .mbarrierInval | .mbarrierExpectTx | .mbarrierCompleteTx | .mbarrierArrive
    | .mbarrierArriveDrop | .mbarrierTestWait | .mbarrierTryWait | .mbarrierPendingCount =>
      some (mbarrier e t pi)
    | .cpAsync => some (cpAsync e t pi)
    | .cpAsyncCommitGroup => some (pure (.sync (.asyncCommit .cpAsync) t.advance))
    | .cpAsyncWaitGroup => some do pure (.sync (.asyncWait .cpAsync (← rd e t pi 0 .u32).toNat) t.advance)
    | .cpAsyncWaitAll => some (pure (.sync (.asyncWait .cpAsync 0) t.advance))
    | .cpAsyncMbarrierArrive => some do
      -- Without `.noinc` the arrive-on is counted in advance: one expected transaction now,
      -- which the asynchronous arrive-on completes.
      let noinc := pi.i.has .noinc
      if !noinc && t.frame.micro == 0 then
        let pre ← mbarReq e t pi 0 (.arrive 0 1) .relaxed .cta
        return .mem (fun _ => pre) t fun _ s => s.setFrame { s.frame with micro := 1 }
      let sig ← mbarReq e t pi 0 (if noinc then .arrive 1 else .completeTx 1) .release .cta
      pure (issue (t.setFrame { t.frame with micro := 0 })
        (NV.Async.Op.copy { moves := #[], signals := #[sig], group := some .cpAsync, afterGroup := true }))
    | .cpAsyncBulk => some (cpAsyncBulk e t pi)
    | .cpReduceAsyncBulk => some (cpReduceAsyncBulk e t pi)
    -- A one-device multicast object (`multimem`, see `NV.PTX.Sem.Step`): the bulk copy or
    -- reduction into the device's own global memory.
    | .multimemCpAsyncBulk => some do
      if pi.i.has .cpMask then throw ".cp_mask is not modelled"
      cpAsyncBulk e t { pi with i := { pi.i with op := .cpAsyncBulk } }
    | .multimemCpReduceAsyncBulk =>
      some (cpReduceAsyncBulk e t { pi with i := { pi.i with op := .cpReduceAsyncBulk } })
    | .cpAsyncBulkTensor => some (cpAsyncBulkTensor e t pi)
    | .cpReduceAsyncBulkTensor => some (cpReduceAsyncBulkTensor e t pi)
    | .tensormapReplace => some (tensormapReplace e t pi)
    | .stAsync | .redAsync => some (asyncStore e t pi)
    | .clusterlaunchcontrolTryCancel => some (tryCancel e t pi)
    | .clusterlaunchcontrolQueryCancel => some (queryCancel e t pi)
    | .tensormapCpFenceproxy => some (tensormapCpFenceproxy e t pi)
    -- `mbarrier.check_layout.layout::vN [addr]`: every mbarrier of the model is `.layout::v0`
    -- (`mbarrier.init` without a layout; `.layout::v1` tracks fabric operations, outside it).
    | .mbarrierCheckLayout => some do
      let _ ← target e t pi 1 (some .shared)
      pure (.next (← wr pi t.advance 0 .pred (if pi.i.has .layoutV1 then 0 else 1)))
    | .cpAsyncBulkPrefetch | .cpAsyncBulkPrefetchTensor => some (pure (.next t.advance))
    | .cpAsyncBulkCommitGroup => some (pure (.sync (.asyncCommit .bulk) t.advance))
    | .cpAsyncBulkWaitGroup => some do
      pure (.sync (.asyncWait .bulk (← rd e t pi 0 .u32).toNat (pi.i.has .read)) t.advance)
    | .wgmmaFence => some (pure (.next t.advance))
    | .wgmmaCommitGroup => some (pure (.sync (.asyncCommit .wgmma) t.advance))
    | .wgmmaWaitGroup => some do pure (.sync (.asyncWait .wgmma (← rd e t pi 0 .u32).toNat) t.advance)
    | .wgmmaMmaAsync => some (wgmma e t pi)
    | .tcgen05Alloc => some (tcAlloc e t pi)
    | .tcgen05Dealloc => some (tcDealloc e t pi)
    | .tcgen05Mma => some (tcMma e t pi)
    | .tcgen05Commit => some (tcCommit e t pi)
    | .tcgen05Ld => some (tcMove e t pi true)
    | .tcgen05St => some (tcMove e t pi false)
    -- Ordering only: the model performs tcgen05 operations in the thread's order, and a load's
    -- registers are written when it performs.
    | .tcgen05RelinquishAllocPermit | .tcgen05WaitLd | .tcgen05WaitSt
    | .tcgen05FenceBeforeThreadSync | .tcgen05FenceAfterThreadSync => some (pure (.next t.advance))
    | .tcgen05Cp => some (tcCp e t pi)
    | .tcgen05Shift => some (tcShift t pi)
    | _ => none
  r.map fun
    | .ok eff => eff
    | .error m => .fault s!"line {pi.line}: {pi.i.op.spelling}: {m}"

end NV.PTX.Async
