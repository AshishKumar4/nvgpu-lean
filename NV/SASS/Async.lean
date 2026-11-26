import NV.SASS.Exec
import NV.Async.Tma
import NV.Async.WgmmaNum
import NV.Async.Tcgen05

/-!
# SASS semantics of the asynchronous opcodes (`sm_90`)

Bulk and tensor copies, mbarrier operations, `cp.async`, and the wgmma pipeline. `exec` defers
to `execAsync` for every opcode it does not define itself; `none` means not an async opcode.

Each definition names the PTX instruction whose SASS form it covers (`tests/sass/async.ptx` and its
listing):

* mbarrier: `mbarrier.init` is `SYNCS.EXCH.64 URZ, [a], URv` with the value
  `(2²⁰ - n) << 1 | (2²⁰ - n) << 43`; arrive-on and tx-count updates are `SYNCS.ARRIVE.TRANS64`
  with `A1T0` (arrive), `ART0` (arrive a register's count), `A0T1` (expect one transaction),
  `A0TR` (expect-tx by a register),
  `A0TX` (complete-tx by a register), no such modifier (arrive and expect-tx by a register,
  `arrive.expect_tx`), `OPTOUT` (`arrive_drop`) and `RED` (no state returned); waits are
  `SYNCS.PHASECHK.TRANS64[.TRYWAIT] Pd, [a], Rp`, true when the barrier's phase bit differs from
  bit 31 of `Rp` (a parity as `p << 31`, a state's high word as is); `inval` is
  `SYNCS.CCTL.IV`; `SYNCS.CCTL.IVALL` changes nothing the model holds.
* `cp.async` is `LDGSTS` (`ZFILL`: the source address's offset within the access size is the
  number of bytes not read, which are zero; a false predicate operand reads nothing), its
  `commit_group` `LDGDEPBAR` (whose write dependence counter the group releases) and
  `cp.async.mbarrier.arrive[.noinc]` `ARRIVES.LDGSTSBAR.64.TRANSCNT|ARVCNT` (after
  `SYNCS.ARRIVE.TRANS64.RED.A0T1` without `.noinc`), one arrival or transaction per thread.
* Bulk and tensor copies are uniform-datapath instructions: `UTMALDG.nD [URa], [URb]` with the
  shared destination in `URa`, the mbarrier in `URa+1`, coordinates from `URa+2`, the tensor
  map's address in `URb:URb+1`; `UTMASTG.nD [URa], [URb]` with the source in `URa` and coordinates
  from `URa+1`; `UBLKCP.S.G [URa], [URb], URc` (destination, mbarrier, global source, size) and
  `UBLKCP.G.S`, the size register in 16-byte units; `UTMACMDFLUSH` is
  `cp.async.bulk.commit_group`.
* `wgmma.mma_async` is one `HGMMA` (f16, bf16, tf32 inputs), `QGMMA` (fp8) or `IGMMA` (8-bit
  integers) per warp of the warpgroup: `HGMMA.64xNxK.<acc>[.<type>] Rd, [Ra,] gdesc[URd], Rc
  [, [!]UPp] [, gsb0]`. `URd:URd+1` is A's descriptor, `URd+2:URd+3` B's; with `Ra` (A from
  registers) only B's is used. `Rc = RZ` or a false predicate is `scale-d = 0`; the descriptor
  suffixes `negA`, `negB`, `tnspA`, `tnspB` are `imm-scale-a/b = -1` and `imm-trans-a/b = 1`;
  `gsb0` commits the warp's wgmma operations into a group (`wgmma.commit_group`), and
  `WARPGROUP.DEPBAR.LE gsb0, n` is `wgmma.wait_group n`. `WARPGROUP.ARRIVE` (`wgmma.fence`)
  orders nothing here: an MMA reads its register operands at issue.

* `tcgen05` (`sm_100a`; `NV.Async.Tcgen05`): `tcgen05.mma.cta_group::1` is
  `UTCHMMA` (`.kind::f16`, `.kind::tf32`), `UTCQMMA` (`.kind::f8f6f4`, `::mxf8f6f4`), `UTCOMMA`
  (`::mxf4`, `::mxf4nvf4`) or `UTCIMMA` (`.kind::i8`), `A, B, tmem[URd], tmem[URx], idesc[URi],
  [tmem[URs],] [!]UPp` with A `gdesc[URa]` or `tmem[URa]`, B `gdesc[URb]`, `URx:URi` the 64-bit
  instruction descriptor (the PTX one in `URi`, `URx` zero), `URs` and `URs+1` A's and B's scale
  factors, and `UPp` `enable-input-d`; one
  thread's `tcgen05.mma` is one execution.
  `tcgen05.commit…mbarrier::arrive::one` is `UTCBAR [URa], URZ`. `tcgen05.ld`/`st` are
  `LDTM`/`STTM` with the shape as a modifier (`16dp64bit`, `16dp128bit`, `16dp256bit`, none for
  `.32x32b`, and `.16x32bx2` as two instructions `16dp32bit_t0_t15` and `16dp32bit_t16_t31`, the
  second at the address plus the half-split offset), `.xN`, and `PACK16BIT`/`EXPAND16BIT` for
  `.pack::16b`/`.unpack::16b`. `tcgen05.alloc`/`dealloc` are software over the column allocator:
  `UTCATOMSWS.FIND_AND_SET[.ALIGN] UPd, URd, URn` claims `URn` granules (`AtomOp.findSetAlign`,
  without `.ALIGN` `AtomOp.findSet`),
  `URd` the first and `UPd` whether it succeeded, `UTCATOMSWS.AND URZ, URm` frees; the
  bookkeeping is ordinary shared memory in the kernel's reserved window.
  `tcgen05.relinquish_alloc_permit` (`UVIRTCOUNT.DEALLOC.SMPOOL`) changes nothing the model
  holds, nor do `tcgen05.wait::ld/st` (the dependence counters order the registers) or the fences
  (`FENCE.VIEW.ASYNC.T`). `.2CTA` (`.cta_group::2`): the MMA from the even CTA of a pair (M = 256 or 128), the commit to the
  pair's mbarriers (`UTCBAR.2CTA.MULTICAST`), the allocator in the even CTA's word. `UTCCP` and
  `UTCSHIFT` are `tcgen05.cp` and `tcgen05.shift` (`utccp`). `.WS`: `tcgen05.mma.ws`, its zero-column
  mask a pair of uniform registers before enable-input-d.

Shared addresses in registers are in the shared-window form: the CTA's cluster rank in bits
24 and up, the offset below.
-/

namespace NV.SASS

open NV NV.Async Exec

namespace AsyncSem

/-- The space and address a shared-window address names. -/
def sharedAt (env : Env) (a : UInt64) : Space × UInt64 := Exec.sharedAddr env a

/-- The space and lane addresses of a request whose lanes each name an mbarrier in the CTA's own
shared memory or a peer's (lanes on several CTAs' barriers at once):
`.shared` at CTA offsets when every lane names its own CTA's, else `.sharedCluster` at each lane's
cluster address (its own CTA's at its rank). -/
def sharedLanes (env : Env) (as : List (Option UInt64)) : Space × Array (Option UInt64) :=
  let at_ := as.map (·.map (sharedAt env))
  if at_.all (·.all (·.1 == .shared)) then (.shared, (at_.map (·.map (·.2))).toArray)
  else (.sharedCluster, as.toArray)

/-- The 64-bit uniform register pair at `URi`; `URZ`'s is zero. -/
def ur64 (w : View) (i : Nat) : UInt64 := if i == 255 then 0 else toU64 (w.getUR i) (w.getUR (i + 1))

/-- The uniform register and offset of a `[URa+off]` operand. -/
def urMem : Operand → Option (Nat × Int)
  | .mem (some ⟨.ur, i⟩) false none _ off => some (i, off)
  | _ => none

def regIdx : Operand → Option Nat
  | .reg ⟨.r, i⟩ _ => some i
  | _ => none

def urIdx : Operand → Option Nat
  | .reg ⟨.ur, i⟩ _ => some i
  | _ => none

/-- The value that initializes an mbarrier for `n` arrivals, and back. -/
def initWord (n : Nat) : UInt64 :=
  let c : UInt64 := (2 ^ 20 - n : Nat).toUInt64
  (c <<< (1 : UInt64)) ||| (c <<< (43 : UInt64))

def initCount? (v : UInt64) : Option Nat :=
  let c := ((v >>> 1) % (1 <<< 20)).toNat
  if c ≥ 1 && initWord (2 ^ 20 - c) == v then some (2 ^ 20 - c) else none

/-- An init word with its phase parity in bit 63 (where the arrive token keeps it): its count and
phase. `mbarrier.init` stores phase 0; a word stored directly may hold 1. -/
def initOf? (v : UInt64) : Option (Nat × Nat) :=
  (initCount? (v &&& ~~~((1 : UInt64) <<< 63))).map fun n => (n, (v >>> 63).toNat)

/-- The number of dimensions of a `UTMALDG.2D`-style opcode. -/
def dims (i : Instr) : Option Nat :=
  i.mods.findSome? fun m => if m.endsWith "D" then (String.ofList m.toList.dropLast).toNat? else none

/-- The tensor map at `addr`: in the parameter block when the address is an offset within it
(the address of a `.param` map), else in global memory (`none`). -/
def paramMap? (k : Kernel) (env : Env) (addr : UInt64) : Option (Except String TensorMap) :=
  if addr.toNat + TensorMap.size ≤ k.paramBytes then
    let bytes := ByteArray.mk ((Array.range TensorMap.size).map fun j =>
      env.const0.readByte (k.paramBase + addr.toNat + j))
    some (match TensorMap.decode bytes with
      | some m => .ok m
      | none => .error s!"no tensor map at parameter offset {addr}")
  else none

/-- A load of the 128-byte tensor map at global `addr`. -/
def mapReq (addr : UInt64) : Req :=
  { space := .global, op := .ld, bytes := 16,
    addrs := (Array.range 8).map fun j => some (addr + (16 * j).toUInt64) }

def mapOfResp (r : Resp) : Except String TensorMap :=
  let bytes := ByteArray.mk ((Array.range TensorMap.size).map fun j =>
    ((r[j / 4]?.getD 0) >>> (8 * (j % 4)).toUInt32).toUInt8)
  match TensorMap.decode bytes with
  | some m => .ok m
  | none => .error "no tensor map at the descriptor address"

/-- The complete-tx an mbarrier-tracked copy performs: a release RMW at cluster scope, in the
generic proxy (§9.7.10.28.5.3). -/
def completeTx (env : Env) (mbar : UInt64) (bytes : Nat) : Req :=
  let (space, a) := sharedAt env mbar
  { space, op := .mbar (.completeTx bytes), bytes := 8, sem := .release, scope := .cluster,
    addrs := #[some a] }

/-- An operation built from the issuer's sources when it reads them (the effect's request
`View`, `Warp.readState`), with no register result. -/
def issue (op : View → Except String Op) (s : Warp) : Out :=
  .async (fun v => match op v with | .ok o => o | .error e => .fault e) s fun _ => []

/-- A TMA or bulk load into shared memory at the shared-window address `smem`, signalling the
mbarrier at `mbar` with the bytes it delivered. -/
def loadOp (env : Env) (ps : Array Piece) (smem mbar : UInt64) (bytes : Nat) : Copy :=
  let hi := smem >>> 24 <<< 24
  loadCopy ps #[fun off => sharedAt env (hi ||| off.toUInt64)] #[completeTx env mbar bytes]

/-- A load multicast to the CTAs of rank `ranks`: each gets the pieces at the same shared offset,
and its mbarrier at `mbar`'s offset the complete-tx of the box's bytes. `.2CTA` sends each
completion to the member of the destination's pair with `mbar`'s rank parity
(PTX ISA §9.7.10.28.5.3: `.cta_group::2` with `.multicast::cluster`). -/
def loadMulticast (env : Env) (ps : Array Piece) (smem mbar : UInt64) (bytes : Nat)
    (ranks : Array Nat) (paired : Bool := false) : Copy :=
  let at_ (r : Nat) (off : UInt64) : UInt64 := (r.toUInt64 <<< 24) ||| (off % (1 <<< 24))
  let signalRank (r : Nat) := if paired then r / 2 * 2 + ((mbar >>> 24) % 2).toNat else r
  loadCopy ps (ranks.map fun r => fun off => sharedAt env (at_ r off.toUInt64))
    (ranks.map fun r => completeTx env (at_ (signalRank r) mbar) bytes)

end AsyncSem

open AsyncSem

/-- `cp.async` (`LDGSTS`) for the lanes `on`: one move of whole accesses, plus byte-wide moves for
lanes whose `ZFILL` source reads only part of the access. -/
def ldgsts (env : Env) (w : View) (on : UInt32) (i : Instr) : Option Op := do
  let size := if i.hasMod "128" then 16 else if i.hasMod "64" then 8 else 4
  let dstOp ← i.operands[0]?
  let srcOp ← i.operands[1]?
  let pred : Nat → Bool := fun l => match i.operands[2]? with
    | some p => (Exec.pred w l p).getD true
    | none => true
  let zfill := i.hasMod "ZFILL"
  let mut src : Array (Option UInt64) := Array.replicate 32 none
  let mut dst : Array (Option UInt64) := Array.replicate 32 none
  let mut short : Array Move := #[]
  for l in Exec.lanes on do
    let d := (sharedAt env (← Exec.addr w l dstOp)).2
    let a ← Exec.addr w l srcOp
    let skip := (a % size.toUInt64).toNat
    if !pred l then dst := dst.set! l (some d)
    else if zfill && skip != 0 then
      let base := a - skip.toUInt64
      let valid := size - skip
      short := short.push
        { src := asyncReq .global .ld 1 ((Array.range size).map fun j =>
            if j < valid then some (base + j.toUInt64) else none)
          dsts := #[asyncReq .shared .st 1 ((Array.range size).map fun j => some (d + j.toUInt64))] }
    else
      src := src.set! l (some a)
      dst := dst.set! l (some d)
  let whole : Move := { src := asyncReq .global .ld size src, dsts := #[asyncReq .shared .st size dst] }
  let moves := (if dst.any (·.isSome) then #[whole] else #[]) ++ short
  return .copy { moves, group := some .cpAsync }

/-- `SYNCS.ARRIVE.TRANS64` for the lanes `on`. Lanes whose counts differ (a warp-aggregated
`mbarrier.arrive` of per-thread counts) arrive on one barrier as one arrival of their summed counts
(the barrier word sees the same total); on different barriers they are not modelled. -/
def syncsArrive (env : Env) (w : View) (on : UInt32) (i : Instr) (s : Warp) :
    Option (Out) := do
  let mem ← i.operands[1]?
  let txReg := i.operands[2]?.bind regIdx
  let lanes := Exec.lanes on
  let txOf (w : View) (l : Nat) : Nat := match txReg with
    | some r => (w.getR r l).toNat
    | none => 0
  let first := lanes.head?.getD 0
  let uniform (w : View) : Bool := lanes.all (txOf w · == txOf w first)
  if !uniform w && lanes.any (Exec.addr w · mem != Exec.addr w first mem) then
    return .fault s!"{i.opcode} at pc {i.pc}: lanes with different tx-counts on different barriers are not modelled"
  -- `n` lanes arriving, their counts summing to `tx`.
  let mkOp (n tx : Nat) : MBarOp :=
    if i.hasMod "A1T0" then .arrive n 0 (i.hasMod "OPTOUT")
    -- `ART0`: the register is an arrival count, no transactions (`mbarrier.arrive … , count`).
    else if i.hasMod "ART0" then .arrive tx 0 (i.hasMod "OPTOUT")
    else if i.hasMod "A0T1" then .arrive 0 n
    else if i.hasMod "A0TR" then .arrive 0 tx
    else if i.hasMod "A0TX" then .completeTx tx
    else .arrive n tx (i.hasMod "OPTOUT")
  -- The address and tx-count registers are read when the instruction reads its sources.
  let req (w : View) : Req :=
    if uniform w then
      let (space, addrs) := sharedLanes env ((List.range 32).map fun l =>
        if lanes.contains l then Exec.addr w l mem else none)
      { space, op := .mbar (mkOp 1 (txOf w first)), bytes := 8, sem := .release,
        scope := if space == .shared then .cta else .cluster, addrs }
    else
      let (space, addrs) := sharedLanes env ((List.range 32).map fun l =>
        if l == first then Exec.addr w l mem else none)
      { space, op := .mbar (mkOp lanes.length (lanes.foldl (· + txOf w ·) 0)), bytes := 8,
        sem := .release, scope := if space == .shared then .cta else .cluster, addrs }
  let dst := (i.operands[0]?.bind regIdx).filter (· != 255)
  let wb (resp : Resp) : List Write := match dst with
    | some r => [.lanes r on ((Array.range 32).map fun l => resp[2 * l]!),
                 .lanes (r + 1) on ((Array.range 32).map fun l => resp[2 * l + 1]!)]
    | none => []
  return .mem req s wb

/-- `SYNCS.PHASECHK.TRANS64[.TRYWAIT] Pd, [a], Rp`: an acquire load of the barrier word. -/
def phaseChk (env : Env) (w : View) (on : UInt32) (i : Instr) (s : Warp) :
    Option (Out) := do
  let pd ← i.operands[0]?
  let mem ← i.operands[1]?
  let pr ← i.operands[2]?
  let lanes := Exec.lanes on
  let req (w : View) : Req :=
    let (space, addrs) := sharedLanes env ((List.range 32).map fun l =>
      if lanes.contains l then Exec.addr w l mem else none)
    { space, op := .ld, bytes := 8, sem := .acquire, scope := if space == .shared then .cta else .cluster,
      addrs }
  let want (l : Nat) : UInt32 := ((Exec.src env w l pr).getD 0) >>> 31
  -- `.TRYWAIT`: the timing core holds the request while the barrier is still in the phase all its
  -- lanes wait past (`SmMem`'s `tryWait`), marked by the hint "TRYWAIT" and that phase.
  let req (w : View) : Req :=
    let r := req w
    let wants := lanes.map want
    match wants with
    | p :: ps => if i.hasMod "TRYWAIT" && ps.all (· == p) then { r with hint := { ops := #["TRYWAIT", s!"want{p}"] } } else r
    | [] => r
  let wb (resp : Resp) : List Write :=
    lanes.flatMap fun l =>
      let phase := resp[2 * l + 1]! >>> 31
      (Exec.wrP pd l (phase != want l)).getD []
  return .mem req s wb

/-- `UTMALDG`, `UTMASTG`: tensor copies; `UTMAREDG.nD.op [URa], [URb]` (PTX
`cp.reduce.async.bulk.tensor`): `UTMASTG`'s operands, each element of
the box reduced into the tensor, the element type the map's (`tensorRedType`). -/
def utma (k : Kernel) (env : Env) (w : View) (i : Instr) : Except String Op := do
  let some (a, _) := (i.operands[0]?.bind urMem) | throw "operand 0"
  let some (b, _) := (i.operands[1]?.bind urMem) | throw "operand 1"
  let some n := dims i | throw "no dimension modifier"
  let load := i.opcode == "UTMALDG"
  let red : Option AtomOp := if i.opcode != "UTMAREDG" then none else i.mods.findSome? fun
    | "ADD" => some .add | "MIN" => some .min | "MAX" => some .max | "INC" => some .inc
    | "DEC" => some .dec | "AND" => some .and | "OR" => some .or | "XOR" => some .xor | _ => none
  if i.opcode == "UTMAREDG" && red.isNone then throw "no reduction operator"
  -- `UTMALDG.nD.IM2COL [URa], [URb], URc`: the im2col offsets in `URc`'s low half, `W` first, in
  -- fields of 16, 8 or 5 bits for rank 3, 4 or 5 (the widths of their ranges, PTX ISA §5.5.4); a
  -- multicast copy's CTA mask in its high half, as without im2col (tests/diff/im2col.ptx).
  let im2col := i.hasMod "IM2COL"
  let offsets : Array Nat := if !im2col then #[] else
    match i.operands[2]?.bind urIdx with
    | some r =>
      let bits := if n == 3 then 16 else if n == 4 then 8 else 5
      (Array.range (n - 2)).map fun k => ((w.getUR r).toNat >>> (bits * k)) % 2 ^ bits
    | none => #[]
  -- `.MULTICAST`: the CTAs of the cluster named by the 16-bit mask in the third operand's high
  -- half (PTX's mask `m`, tests/diff/tmamc.ptx) each get the box at the same shared offset, and their mbarriers at the
  -- same offset its complete-tx (PTX ISA §9.7.9.27, `.multicast::cluster`).
  let ranks : Option (Array Nat) :=
    if load && i.hasMod "MULTICAST" then
      (i.operands[2]?.bind urIdx).map fun r =>
        let m := ((w.getUR r).toNat >>> 16) % 65536
        (Array.range 16).filter fun k => (m >>> k) % 2 == 1
    else none
  if load && i.hasMod "MULTICAST" && ranks.isNone then throw "MULTICAST without a CTA mask operand"
  let smem := (w.getUR a).toUInt64
  let c0 := if load then a + 2 else a + 1
  -- `.GATHER4` (PTX `.tile::gather4`) and the store's `.SCATTER4` (`.tile::scatter4`): five
  -- coordinates, the column then the four rows (§9.7.10.28.5.3).
  let fourRows := if load then i.hasMod "GATHER4" else i.hasMod "SCATTER4"
  let coords := (Array.range (if fourRows then 5 else n)).map fun j => (w.getUR (c0 + j)).toInt32.toInt
  let mapAddr := ur64 w b
  let mk (m : TensorMap) : Except String Copy := do
    if im2col != m.im2col.isSome then throw "the copy's mode is not the tensor map's"
    let ps ← if im2col then m.im2colTile coords offsets (smem % (1 <<< 24)).toNat
      else if fourRows then m.gather4 coords[0]! (coords.extract 1 5) (smem % (1 <<< 24)).toNat
      else m.tile coords (smem % (1 <<< 24)).toNat
    let bytes := if fourRows then 4 * m.boxBytes else m.boxBytes
    if load then
      match ranks with
      | some rs =>
        return loadMulticast env ps smem (w.getUR (a + 1)).toUInt64 bytes rs (i.hasMod "2CTA")
      | none => return loadOp env ps smem (w.getUR (a + 1)).toUInt64 bytes
    else
      let hi := smem >>> 24 <<< 24
      let at_ (off : Nat) : UInt64 := (sharedAt env (hi ||| off.toUInt64)).2
      match red with
      | none => return storeCopy ps (sharedAt env smem).1 at_
      | some o =>
        let some aty := tensorRedType m.dtype o
          | throw s!"{repr o} on a {repr m.dtype} tensor is not one the ISA allows"
        let some eb := m.dtype.bytes? | throw "packed tensor types do not reduce"
        return storeCopy (elementPieces eb ps) (sharedAt env smem).1 at_ (.red o aty)
  match paramMap? k env mapAddr with
  -- A parameter's map: fetched from the parameter block in memory like any descriptor, its
  -- contents the bank's.
  | some m =>
    let m ← m.mapError (s!"{·} (the map at parameter offset {mapAddr})")
    return .tensor { desc := mapReq (env.paramAddr + mapAddr),
                     build := fun _ => (mk m).mapError (s!"{·} (the map at parameter offset {mapAddr})"),
                     group := if load then none else some .bulk }
  | none =>
    let at_ := s!"(the map at global address 0x{String.mk (Nat.toDigits 16 mapAddr.toNat)})"
    return .tensor { desc := mapReq mapAddr,
                     build := fun r => (do mk (← mapOfResp r)).mapError (s!"{·} {at_}"),
                     group := if load then none else some .bulk }

/-- `UBLKCP.S.G` (global to shared, mbarrier; `.MULTICAST` to the cluster CTAs a mask names) and
`UBLKCP.G.S` (shared to global, bulk group);
`UBLKCP.G.S.DST_G_BYTE_MASK` (`.cp_mask`) packs the size in `URc`'s low half and the byte mask
in its high half. -/
def ublkcp (env : Env) (w : View) (i : Instr) : Except String Op := do
  let some (a, _) := (i.operands[0]?.bind urMem) | throw "operand 0"
  let some (b, _) := (i.operands[1]?.bind urMem) | throw "operand 1"
  let some c := (i.operands[2]?.bind urIdx) | throw "operand 2"
  -- The size register counts 16-byte units (a 4096-byte copy is 0x100).
  let masked := i.hasMod "DST_G_BYTE_MASK"
  -- `UBLKCP.S.G.MULTICAST` (`.multicast::cluster`): the CTA mask in `URc`'s high half, as with
  -- `.DST_G_BYTE_MASK`'s byte mask.
  let multicast := i.hasMod "MULTICAST"
  let units := (w.getUR c).toNat
  let size := 16 * (if masked || multicast then units % 65536 else units)
  -- `UBLKCP.S.S [URa], [URb], URc` (`cp.async.bulk.shared::cluster.shared::cta`): into the cluster CTA's shared memory at `URa`, its mbarrier at `URa+1`.
  if i.mods.toList == ["S", "S"] then
    let dst := (w.getUR a).toUInt64
    return .copy (← elementCopy (sharedAt env (w.getUR b).toUInt64) (sharedAt env dst) size 16 .st
      #[completeTx env (w.getUR (a + 1)).toUInt64 size])
  if i.mods[0]? == some "S" then
    let smem := (w.getUR a).toUInt64
    let ps ← bulkPieces (ur64 w b) (smem % (1 <<< 24)).toNat size
    if multicast then
      let m := units / 65536
      return .copy (loadMulticast env ps smem (w.getUR (a + 1)).toUInt64 size
        ((Array.range 16).filter fun k => (m >>> k) % 2 == 1))
    return .copy (loadOp env ps smem (w.getUR (a + 1)).toUInt64 size)
  else
    let smem := (w.getUR b).toUInt64
    let ps ← bulkPieces (ur64 w a) (smem % (1 <<< 24)).toNat size
    let ps := if masked then (elementPieces 1 ps).filter fun p => (units / 65536).testBit (p.smem % 16)
      else ps
    let hi := smem >>> 24 <<< 24
    return .copy (storeCopy ps (sharedAt env smem).1 fun off => (sharedAt env (hi ||| off.toUInt64)).2)

/-- `UBLKRED.G.S.op[.type] [URa], [URb], URc`: `cp.reduce.async.bulk.global.shared::cta.bulk_group`
with the global destination in `URa:URa+1`, the shared source in `URb` and the size in 16-byte
units in `URc` (as `UBLKCP`); each element of the source reduced into the destination. -/
def ublkred (env : Env) (w : View) (i : Instr) : Except String Op := do
  let some (a, _) := (i.operands[0]?.bind urMem) | throw "operand 0"
  let some (b, _) := (i.operands[1]?.bind urMem) | throw "operand 1"
  let some c := (i.operands[2]?.bind urIdx) | throw "operand 2"
  let size := 16 * (w.getUR c).toNat
  let some aop := i.mods.findSome? fun
      | "ADD" => some AtomOp.add | "MIN" => some .min | "MAX" => some .max | "INC" => some .inc
      | "DEC" => some .dec | "AND" => some .and | "OR" => some .or | "XOR" => some .xor
      | _ => none
    | throw "no reduction operator"
  let (ty, elem) : AtomType × Nat := (i.mods.findSome? fun
      | "S32" => some (.s32, 4) | "U32" => some (.u32, 4) | "U64" => some (.u64, 8)
      | "S64" => some (.s64, 8) | "F32" => some (.f32, 4) | "F64" => some (.f64, 8)
      | "F16" => some (.f16, 2) | "BF16" => some (.bf16, 2)
      | _ => none).getD (.u32, 4)
  let smem := (w.getUR b).toUInt64
  -- `UBLKRED.S.S.op [URa], [URb], URc` (PTX `cp.reduce.async.bulk.shared::cluster.shared::cta`): into the cluster CTA's shared memory at `URa`, the mbarrier at
  -- `URa+1` signalled with the bytes.
  if i.mods[0]? == some "S" then
    let dst := (w.getUR a).toUInt64
    return .copy (← elementCopy (sharedAt env smem) (sharedAt env dst) size elem (.red aop ty)
      #[completeTx env (w.getUR (a + 1)).toUInt64 size])
  let ps ← bulkPieces (ur64 w a) (smem % (1 <<< 24)).toNat size
  let hi := smem >>> 24 <<< 24
  return .copy (storeCopy (elementPieces elem ps) (sharedAt env smem).1
    (fun off => (sharedAt env (hi ||| off.toUInt64)).2) (.red aop ty))

/-- The input types an `HGMMA`/`QGMMA`/`IGMMA`'s modifiers name, and its accumulator. -/
def gmmaTypes (i : Instr) : Option (Wgmma.InType × Wgmma.InType × Wgmma.Acc) :=
  let ty : String → Option Wgmma.InType
    | "F16" => some .f16 | "BF16" => some .bf16 | "TF32" => some .tf32
    | "E4M3" => some .e4m3 | "E5M2" => some .e5m2 | "S8" => some .s8 | "U8" => some .u8
    | _ => none
  let ms := i.mods.toList.drop 1
  match i.opcode, ms with
  | "IGMMA", a :: b :: _ => do return (← ty a, ← ty b, .s32)
  -- `BGMMA.64xNx256.AND.POPC` (PTX `wgmma….s32.b1.b1.and.popc`).
  | "BGMMA", _ => if i.hasMod "AND" && i.hasMod "POPC" then some (.b1, .b1, .s32) else none
  | "QGMMA", acc :: a :: b :: _ => do
    return (← ty a, ← ty b, ← if acc == "F32" then some .f32 else if acc == "F16" then some .f16 else none)
  | "HGMMA", ["F16"] => some (.f16, .f16, .f16)
  | "HGMMA", ["F32"] => some (.f16, .f16, .f32)
  | "HGMMA", ["F32", t] => do let t ← ty t; return (t, t, .f32)
  | _, _ => none

/-- `HGMMA`, `QGMMA`, `IGMMA`: this warp's share of a warpgroup MMA, starting from the accumulator
`Rc` holds at issue (or the warp's previous MMA's result, which the machine chains), and the write
of its result into the accumulator registers when it completes. -/
def gmma (env : Env) (w : View) (i : Instr) : Except String (Op × (Resp → List Write)) := do
  let some shape := i.mods[0]? | throw "no shape"
  let some (n, k) := (match (shape.splitOn "x").map String.toNat? with
    | [some 64, some n, some k] => some (n, k)
    | _ => none) | throw s!"shape {shape}"
  let some (ta, tb, acc) := gmmaTypes i | throw "types"
  if (if ta == .b1 then 256 else ta.k) != k then throw s!"K = {k} for {repr ta}"
  let some d := i.operands[0]?.bind regIdx | throw "destination"
  let rs := (i.operands[1]?.bind regIdx).isSome
  let gi := if rs then 2 else 1
  let some (Operand.gdesc ⟨.ur, u⟩ dmods) := (i.operands[gi]? : Option Operand) | throw "descriptor"
  let cReg := (i.operands[gi + 1]?.bind regIdx).filter (· != 255)
  let pred : Bool := match (i.operands[gi + 2]? : Option Operand) with
    | some (.reg ⟨.up, p⟩ m) => w.getUP p != m.not
    | _ => true
  let words := Wgmma.accWords acc n
  let a : Wgmma.ASource :=
    if rs then
      let r := (i.operands[1]?.bind regIdx).getD 0
      .regs ((Array.range 32).map fun l => (Array.range 4).map fun j => w.getR (r + j) l)
    else .smem (Wgmma.Desc.decode (ur64 w u)) (!dmods.contains "tnspA")
  let share : Wgmma.Share :=
    -- `gdesc[URZ]` (the empty `HGMMA.64x8x16.F16 RZ, gdesc[URZ], RZ, !UPT` that fills an empty
    -- commit group) names zero descriptors, A's and B's alike.
    { n, ty := ta, tyB := tb, acc, warp := env.ctx.warp % 4, a,
      b := Wgmma.Desc.decode (if u == 255 then 0 else ur64 w (u + 2)),
      -- A from registers is negated by the operand's sign (`HGMMA … -R32, gdesc[UR8]`, PTX's
      -- `imm-scale-a = -1`).
      bKMajor := !dmods.contains "tnspB",
      negA := dmods.contains "negA" || (rs && match i.operands[1]? with
        | some (Operand.reg _ m) => m.neg | _ => false),
      negB := dmods.contains "negB", scaleD := cReg.isSome && pred }
  let plan ← share.plan (·.toUInt64)
  let commit := i.operands.any (· == .name "gsb0")
  -- The accumulator at issue, lane-major (`RZ`: zero).
  let c0 := (Array.range 32).flatMap fun l => match cReg with
    | some r => (Array.range words).map fun j => w.getR (r + j) l
    | none => Array.replicate words 0
  let wb (out : Resp) : List Write :=
    (List.range words).map fun j =>
      .lanes (d + j) fullMask ((Array.range 32).map fun l => out[l * words + j]?.getD 0)
  let mma := { plan.mma share c0 with commit, dst := d, src := if share.scaleD then cReg else none }
  -- The loads after A's chunks (32 a request) read B alone, which the four shares read once.
  let aChunks ← match share.a with
    | .smem dsc km => do
      let addrs ← (share.rows.toArray.flatMap fun r => (Array.range ta.k).map fun k => dsc.addr ta km r k).mapM id
      pure (Wgmma.chunks addrs).size
    | .regs _ => pure 0
  return (.mma { mma with warpInGroup := some share.warp, bFrom := some ((aChunks + 31) / 32) }, wb)

/-- `STAS[.64|.128] [Ra.64+off], Rb` and `REDAS.op[.S32][.64] [Ra.64+off], Rb` (PTX `st.async`,
`red.async`, `sm_90a`): each lane stores (or reduces) Rb… at the
`.shared::cluster` address Ra (+ off) and then signals a complete-tx of the bytes on the mbarrier
at Ra+1, by the async unit in the generic proxy. -/
def stas (env : Env) (w : View) (on : UInt32) (i : Instr) : Except String Op := do
  let some (.mem (some ⟨.r, ra⟩) true none none off) := (i.operands[0]? : Option Operand)
    | throw "address operand"
  let some rb := i.operands[1]?.bind regIdx | throw "data operand"
  let words := if i.hasMod "128" then 4 else if i.hasMod "64" then 2 else 1
  let op : MemOp ← if i.opcode == "STAS" then pure .st else do
    let o : AtomOp ← if i.hasMod "ADD" then pure .add else if i.hasMod "MIN" then pure .min
      else if i.hasMod "MAX" then pure .max else if i.hasMod "INC" then pure .inc
      else if i.hasMod "DEC" then pure .dec else if i.hasMod "AND" then pure .and
      else if i.hasMod "OR" then pure .or else if i.hasMod "XOR" then pure .xor
      else throw "reduction operator"
    let bitwise := o == .and || o == .or || o == .xor
    let ty : AtomType := if words == 2 then (if bitwise then .b64 else .u64)
      else if bitwise then .b32 else if i.hasMod "S32" then .s32 else .u32
    pure (.red o ty)
  let lanes := Exec.lanes on
  let addr (l : Nat) : UInt64 := ((w.getR ra l).toUInt64 + UInt64.ofInt off) &&& 0xffffffff
  let addrs := (Array.range 32).map fun l =>
    if lanes.contains l then some (sharedAt env (addr l)).2 else none
  let dst : Req := { space := (sharedAt env (addr (lanes.headD 0))).1, op, bytes := 4 * words, addrs }
  if (lanes.any fun l => (sharedAt env (addr l)).1 != dst.space) then
    throw "lanes in different spaces"
  let imm := (Array.range 32).flatMap fun l => (Array.range words).map fun j => w.getR (rb + j) l
  let signals := lanes.toArray.map fun l => completeTx env (w.getR (ra + 1) l).toUInt64 (4 * words)
  return .copy { moves := #[{ src := default, dsts := #[dst], imm }], signals }

/-- `UGETNEXTWORKID.{SELFCAST|BROADCAST} [URr], [URm]` (PTX `clusterlaunchcontrol.try_cancel`,
`.multicast::cluster::all` with `BROADCAST`; §9.7.15.18): as `PTX.Async.tryCancel`, every request
fails (the model launches every cluster, and the ISA lets a request fail): 16 zero bytes, the
response `query_cancel` reads as not cancelled, to the CTA's shared memory at `URr` (every cluster
CTA's with `BROADCAST`), each followed by a complete-tx of 16 on the mbarrier at `URm`'s offset in
that CTA. -/
def getNextWorkId (env : Env) (w : View) (i : Instr) : Except String Op := do
  let some (r, ro) := (i.operands[0]?.bind urMem) | throw "response operand"
  let some (m, mo) := (i.operands[1]?.bind urMem) | throw "mbarrier operand"
  let resp := ((w.getUR r).toUInt64 + UInt64.ofInt ro) &&& 0xffffffff
  let mbar := ((w.getUR m).toUInt64 + UInt64.ofInt mo) &&& 0xffffffff
  if resp % 16 != 0 then throw s!"response address {resp} is not 16-byte aligned"
  let ranks : Array (Option Nat) :=
    if i.hasMod "BROADCAST" then (Array.range env.ctx.clusterSize).map some else #[none]
  let at_ (r : Option Nat) (a : UInt64) : UInt64 := match r with
    | some r => (r.toUInt64 <<< 24) ||| (a % (1 <<< 24))
    | none => a
  let dsts := ranks.map fun k =>
    let (space, a) := sharedAt env (at_ k resp)
    ({ space, op := .st, bytes := 16, proxy := .async, addrs := #[some a] } : Req)
  return .copy { moves := #[{ src := default, dsts, imm := #[0, 0, 0, 0] }],
                 signals := ranks.map fun k => completeTx env (at_ k mbar) 16 }

/-- A `tmem[URa+off]` operand's address. -/
def tmemAddr (w : View) : Operand → Option Nat
  | .udesc "tmem" r off => some ((w.getUR r.idx).toNat + off.toNat)
  | _ => none

/-- The CTA pair of a `.2CTA` instruction, issued by the even CTA of the pair, its peer the next
rank (§9.7.18.2.1). -/
def pairOf (env : Env) (i : Instr) : Except String (Option (Nat × Nat)) := do
  if !i.hasMod "2CTA" then return none
  let rank := env.ctx.clusterRank
  if rank % 2 != 0 then throw s!"a .2CTA {i.opcode} from the odd CTA of its pair"
  return some (rank, rank + 1)

/-- `UTCHMMA`, `UTCQMMA`, `UTCIMMA`, `UTCOMMA`: one `tcgen05.mma`; `.2CTA`, `.cta_group::2`. A
block-scaled one has its scale factors' addresses after the instruction descriptor, A's and B's
in a pair of uniform registers: `UTCQMMA` is then `.kind::mxf8f6f4`, `UTCOMMA` `::mxf4` or
`::mxf4nvf4` (the same encoding; the descriptor's scale type tells UE4M3 from
UE8M0), two factors a row, four with `.4X`. -/
def utcmma (env : Env) (w : View) (i : Instr) : Except String Op := do
  if i.mods.any (fun m => m != "2CTA" && m != "4X" && m != "WS") then throw s!"{i.mods.toList} is not modelled"
  let ws := i.hasMod "WS"
  let pair ← pairOf env i
  let ops := i.operands
  let some (Operand.udesc "idesc" ri _) := (ops[4]? : Option Operand)
    | throw "operands other than A, B, D, a zero word or scale factors, the instruction descriptor and enable-input-d"
  -- A block-scaled MMA's scale factors: A's at `URs`, B's at `URs+1`.
  let sf : Option (Nat × Nat) := match (ops[5]? : Option Operand) with
    | some (.udesc "tmem" r off) =>
      some ((w.getUR r.idx).toNat + off.toNat, (w.getUR (r.idx + 1)).toNat + off.toNat)
    | _ => none
  let kind ← match i.opcode, sf.isSome with
    | "UTCQMMA", false => pure Tcgen05.Kind.f8f6f4
    | "UTCQMMA", true => pure .mxf8f6f4
    | "UTCOMMA", true => pure .mxf4nvf4
    | "UTCIMMA", false => pure .i8
    | "UTCHMMA", false => pure .f16
    | o, s => throw s!"{o}{if s then " with scale factors" else ""} is not modelled"
  let word := (w.getUR ri.idx).toNat
  let (idesc, form) ← if kind.scaled then do
      let (d, f) ← Tcgen05.IDesc.decodeScaled kind (if kind == .mxf8f6f4 then 1 else if i.hasMod "4X" then 4 else 2) word
      pure (d, some f)
    else pure (Tcgen05.IDesc.decode word, none)
  -- `UTCHMMA` is `.kind::tf32` when A's type code is TF32's.
  let kind := if kind == .f16 && idesc.atype == 2 then .tf32 else kind
  let a ← match (ops[0]? : Option Operand) with
    | some (.gdesc ⟨.ur, u⟩ ms) =>
      if !ms.isEmpty then throw s!"descriptor suffixes {ms.toList} are not modelled"
      pure (Tcgen05.ASrc.smem (← Tcgen05.sdesc (ur64 w u)))
    | some o => match tmemAddr w o with
      | some x => pure (.tmem x)
      | none => throw "A operand"
    | none => throw "A operand"
  -- `.B_KEEP` and its kin are `.ws`'s collector-buffer hints, which change no result.
  let b ← match (ops[1]? : Option Operand) with
    | some (.gdesc ⟨.ur, u⟩ ms) =>
      if !ms.isEmpty && !ws then throw s!"descriptor suffixes {ms.toList} are not modelled"
      Tcgen05.sdesc (ur64 w u)
    | _ => throw "B operand"
  let some d := ops[2]?.bind (tmemAddr w) | throw "D operand"
  let some x := ops[3]?.bind (tmemAddr w) | throw "fourth operand"
  -- The sparse metadata's address, or 0.
  if x != 0 && !idesc.sparse then throw "sparse metadata without the descriptor's sparse flag"
  let spMeta := if idesc.sparse then some x else none
  let en := if sf.isSome then 6 else 5
  -- `.ws`'s zero-column mask: a pair of uniform registers before enable-input-d.
  let (zeroMask, en) := match ws, (ops[en]? : Option Operand) with
    | true, some (.reg ⟨.ur, u⟩ _) => ((ur64 w u).toNat, en + 1)
    | _, _ => (0, en)
  let accumulate ← match (ops[en]? : Option Operand) with
    | some (.reg ⟨.up, p⟩ m) => pure (w.getUP p != m.not)
    | none => pure true
    | _ => throw "operands after enable-input-d are not modelled"
  let scales := match form, sf with
    | some form, some (sa, sb) => some { form, a := sa, b := sb : Tcgen05.Scales }
    | _, _ => none
  let inst : Tcgen05.Inst := { kind, idesc, a, b, d, accumulate, pair, scales, ws, zeroMask, spMeta }
  return .mmaT (← inst.op (·.toUInt64))

/-- `UTCCP.T.S[.shape][.U6x16P32|.U4x16P64][.2CTA] tmem[URd], gdesc[URs]`: one `tcgen05.cp`: no shape modifier is `.128x256b`, `4dp256bit` `.4x256b`, `128dp128bit`
`.128x128b`, `2x64dp128bit_lw02_lw13` and `_lw01_lw23` `.64x128b.warpx2::02_13` and `::01_23`,
`4x32dp128bit` `.32x128b.warpx4`; `U6x16P32` and `U4x16P64` `.b8x16.b6x16_p32` and
`.b8x16.b4x16_p64`. -/
def utccp (env : Env) (w : View) (i : Instr) : Except String Op := do
  let known := ["T", "S", "2CTA", "4dp256bit", "128dp128bit", "2x64dp128bit_lw02_lw13",
    "2x64dp128bit_lw01_lw23", "4x32dp128bit", "U6x16P32", "U4x16P64"]
  if i.mods.any (!known.contains ·) then throw s!"{i.mods.toList} is not modelled"
  unless i.hasMod "T" && i.hasMod "S" do throw "a copy other than shared to tensor memory"
  let shape := if i.hasMod "4dp256bit" then Tcgen05.CpShape.s4x256b
    else if i.hasMod "128dp128bit" then .s128x128b
    else if i.hasMod "2x64dp128bit_lw02_lw13" then .s64x128b02_13
    else if i.hasMod "2x64dp128bit_lw01_lw23" then .s64x128b01_23
    else if i.hasMod "4x32dp128bit" then .s32x128bx4
    else .s128x256b
  let unpack := if i.hasMod "U6x16P32" then Tcgen05.Unpack.b6 else if i.hasMod "U4x16P64" then .b4 else .none
  let some dst := i.operands[0]?.bind (tmemAddr w) | throw "destination operand"
  let src ← match (i.operands[1]? : Option Operand) with
    | some (.gdesc ⟨.ur, u⟩ ms) =>
      if !ms.isEmpty then throw s!"descriptor suffixes {ms.toList} are not modelled"
      Tcgen05.sdesc (ur64 w u)
    | _ => throw "source descriptor"
  let cp : Tcgen05.Cp := { shape, unpack, src, dst, pair := ← pairOf env i }
  return .mmaT (← cp.op (·.toUInt64))

/-- `UTCSHIFT.DOWN[.2CTA] tmem[URd]`: one `tcgen05.shift.down`. -/
def utcshift (env : Env) (w : View) (i : Instr) : Except String Op := do
  if i.mods.any (fun m => m != "DOWN" && m != "2CTA") || !i.hasMod "DOWN" then throw s!"{i.mods.toList} is not modelled"
  let some dst := i.operands[0]?.bind (tmemAddr w) | throw "address operand"
  return .mmaT (← Tcgen05.shiftOp dst (← pairOf env i))

/-- `UTCBAR[.2CTA] [URa], URZ`: `tcgen05.commit.mbarrier::arrive::one`: once the warp's earlier
`tcgen05` operations have completed, one arrive-on on the mbarrier at `URa` (release, as
`cp.async.mbarrier.arrive.noinc`'s). `.MULTICAST [URa], URZ, URm`: one on the mbarrier at `URa`'s
offset in each CTA of the cluster whose rank's bit `URm` sets (`.multicast::cluster`). -/
def utcbar (env : Env) (w : View) (i : Instr) : Except String Op := do
  if i.mods.any (fun m => m != "2CTA" && m != "MULTICAST") then throw s!"{i.mods.toList} is not modelled"
  let some (r, off) := i.operands[0]?.bind urMem | throw "mbarrier operand"
  if i.operands[1]? != some (.reg Reg.URZ) then throw "a second operand other than URZ"
  let at_ := (w.getUR r).toUInt64 + UInt64.ofInt off
  let sig (a : UInt64) : Req :=
    let (space, a) := sharedAt env a
    { space, op := .mbar (.arrive 1), bytes := 8, sem := .release,
      scope := if space == .shared then .cta else .cluster, addrs := #[some a] }
  let signals ← if i.hasMod "MULTICAST" then do
      let some m := i.operands[2]?.bind urIdx | throw "multicast mask"
      let mask := (w.getUR m).toNat
      pure ((List.range 16).filter (fun k => (mask >>> k) % 2 == 1) |>.toArray |>.map fun (k : Nat) =>
        sig ((k.toUInt64 <<< (24 : UInt64)) ||| (at_ &&& 0xFFFFFF)))
    else pure #[sig at_]
  return .copy { moves := #[], signals, group := some .tcgen05, afterGroup := true }

/-- An `LDTM`/`STTM`'s shape (its modifier), the threads it moves (one half for `.16x32bx2`'s two
instructions), its repeat count and whether it packs 16-bit halves. -/
def tmemForm (i : Instr) : Tcgen05.Shape × (Nat → Bool) × Nat × Bool :=
  let (shape, threads) : Tcgen05.Shape × (Nat → Bool) :=
    if i.hasMod "16dp64bit" then (.s16x64b, fun _ => true)
    else if i.hasMod "16dp128bit" then (.s16x128b, fun _ => true)
    else if i.hasMod "16dp256bit" then (.s16x256b, fun _ => true)
    else if i.hasMod "16dp32bit_t0_t15" then (.s16x32bx2, (· < 16))
    else if i.hasMod "16dp32bit_t16_t31" then (.s16x32bx2, (· ≥ 16))
    else (.s32x32b, fun _ => true)
  let num := (i.mods.findSome? fun m => if m.startsWith "x" then (m.drop 1).toString.toNat? else none).getD 1
  (shape, threads, num, i.hasMod "PACK16BIT" || i.hasMod "EXPAND16BIT")

/-- The registers a thread's `LDTM`/`STTM` moves. -/
def tmemRegs (i : Instr) : Nat := let (s, _, num, _) := tmemForm i; s.regs * num

/-- `LDTM` and `STTM`: `tcgen05.ld`/`tcgen05.st` for the lanes of `on`. -/
def tmemMove (env : Env) (w : View) (on : UInt32) (i : Instr) (s : Warp) : Except String Out := do
  let ld := i.opcode == "LDTM"
  let (shape, threads, num, pack) := tmemForm i
  let known := ["16dp64bit", "16dp128bit", "16dp256bit", "16dp32bit_t0_t15", "16dp32bit_t16_t31",
    "PACK16BIT", "EXPAND16BIT"]
  for m in i.mods do
    unless known.contains m || (m.startsWith "x" && (m.drop 1).toString.toNat?.isSome) do
      throw s!"modifier {m}"
  let some addr := (i.operands[if ld then 1 else 0]?).bind (tmemAddr w) | throw "tensor-memory operand"
  let some reg := (i.operands[if ld then 0 else 1]?).bind regIdx | throw "register operand"
  let mask : UInt32 := (List.range 32).foldl (fun m t =>
    if threads t && (on >>> t.toUInt32) &&& 1 == 1 then m ||| (1 <<< t.toUInt32) else m) 0
  let active (t : Nat) : Bool := (mask >>> t.toUInt32) &&& 1 == 1
  let warp := env.ctx.warp
  -- The half-split offset is already in the second instruction's address.
  let (r, regs) ← Tcgen05.moveReq (if ld then .ld else .st) shape num 0 addr warp pack (threads := active)
  if ld then
    return .mem (fun _ => r) s fun resp =>
      (List.range regs).map fun j =>
        .lanes (reg + j) mask ((Array.range 32).map fun t => (Tcgen05.unpackLoad resp t regs pack)[j]!)
  -- A store reads its registers when it reads its sources.
  let req (v : View) : Req :=
    let data := (Array.range 32).flatMap fun t => (Array.range regs).map fun j => v.getR (reg + j) t
    match Tcgen05.moveReq .st shape num 0 addr warp pack data (threads := active) with
    | .ok (r, _) => r
    | .error _ => default
  return .mem req s fun _ => []

/-- `UTCATOMSWS`: the tensor-memory column allocator's word. -/
def utcatomsws (w : View) (i : Instr) (s : Warp) : Except String Out := do
  -- `.2CTA` (`tcgen05.alloc.cta_group::2`), run by the even CTA of the pair: the model keeps the
  -- pair's columns in that CTA's allocator, and each CTA frees with its own (inferred).
  let at_ : Array (Option UInt64) := #[some Tcgen05.allocAddr.toUInt64]
  if i.hasMod "FIND_AND_SET" then
    -- Without `.ALIGN`: the lowest run at any position (inferred).
    let align := i.hasMod "ALIGN"
    let some (.reg ⟨.up, pd⟩ _) := (i.operands[0]? : Option Operand) | throw "predicate destination"
    let some ud := i.operands[1]?.bind urIdx | throw "destination"
    let some un := i.operands[2]?.bind urIdx | throw "granule count"
    let n := (w.getUR un).toNat
    let req : Req := { space := .tmem, op := .atom (if align then .findSetAlign else .findSet) .b32,
                       bytes := 4, addrs := at_, data := #[n.toUInt32] }
    return .mem (fun _ => req) s fun resp =>
      match (if align then Tcgen05.findSetAlign? else fun x n => NV.findSet x n) ((resp[0]?.getD 0)) n with
      | some g => [.up pd true, .ur ud g.toUInt32]
      | none => [.up pd false]
  if i.hasMod "AND" then
    let some ud := i.operands[0]?.bind urIdx | throw "destination"
    let some um := i.operands[1]?.bind urIdx | throw "mask"
    let req : Req := { space := .tmem, op := .atom .and .b32, bytes := 4, addrs := at_,
                       data := #[w.getUR um] }
    return .mem (fun _ => req) s fun resp => [.ur ud (resp[0]?.getD 0)]
  throw s!"{i.mods.toList} is not modelled"

/-- Execute instruction `i` (index `pc` of `k`) for the lanes `m`, if it is an async opcode. -/
def execAsync (k : Kernel) (env : Env) (v : View) (w0 : Warp) (pc : Nat) (m : UInt32)
    (i : Instr) : Option (Out) :=
  let on := guarded v m i
  let s := w0.advance m pc
  let w := v
  let bad := s!"{i.opcode}{i.mods.toList} at pc {i.pc}: unsupported form"
  let orBad (e : Option (Out)) : Out := e.getD (.fault bad)
  -- An operation is checked against the issuing state and built from the state that reads its
  -- sources.
  let lift (build : View → Except String Op) : Out := match build w with
    | .ok _ => issue build s
    | .error msg => .fault s!"{i.opcode}{i.mods.toList} at pc {i.pc}: {msg}"
  -- Uniform-datapath instructions run once for the warp, when any lane's guard holds.
  let uniform := on != 0
  match i.opcode with
  | "LDGSTS" =>
    if on == 0 then some (.next s) else
    some (lift fun w => match ldgsts env w on i with | some op => .ok op | none => .error "unsupported form")
  | "LDGDEPBAR" => some (.sync (.asyncCommit .cpAsync) s)
  | "ARRIVES" =>
    some <| orBad do
      let (r, _) ← i.operands[0]?.bind urMem
      let (space, a) := sharedAt env (w.getUR r).toUInt64
      -- One arrive-on (or one transaction) per thread of the warp executing it.
      let n := (Exec.lanes on).length
      let op : MBarOp := if i.hasMod "ARVCNT" then .arrive n else .completeTx n
      let sig : Req := { space, op := .mbar op, bytes := 8, sem := .release,
                         scope := if space == .shared then .cta else .cluster, addrs := #[some a] }
      if on == 0 then return .next s
      return issue (fun _ => .ok (.copy { moves := #[], signals := #[sig], group := some .cpAsync,
                                          afterGroup := true })) s
  | "SYNCS" =>
    if on == 0 then some (.next s)
    else if i.hasMod "ARRIVE" then some (orBad (syncsArrive env w on i s))
    else if i.hasMod "PHASECHK" then some (orBad (phaseChk env w on i s))
    else if i.hasMod "EXCH" then
      some <| orBad do
        let (r, off) ← i.operands[1]?.bind urMem
        let v ← i.operands[2]?.bind urIdx
        let some (n, phase) := initOf? (ur64 w v)
          | return .fault s!"{i.opcode}{i.mods.toList} at pc {i.pc}: an exchange of a value that is not an mbarrier init (not modelled)"
        let (space, a) := sharedAt env ((w.getUR r).toUInt64 + UInt64.ofInt off)
        let req : Req := { space, op := .mbar (.init n phase), bytes := 8, sem := .relaxed, scope := .cta,
                           addrs := #[some a] }
        return .mem (fun _ => req) s fun _ => []
    else if i.hasMod "CCTL" && i.hasMod "IV" then
      -- `SYNCS.CCTL.IV [URa+off]` or `[Ra+URb+off]`: each executing lane's mbarrier.
      some <| orBad do
        let mem ← i.operands[0]?
        let lanes := Exec.lanes on
        let req (w : View) : Req :=
          let (space, addrs) := sharedLanes env ((List.range 32).map fun l =>
            if lanes.contains l then Exec.addr w l mem else none)
          { space, op := .mbar .inval, bytes := 8, addrs }
        return .mem req s fun _ => []
    else if i.hasMod "CCTL" && i.hasMod "IVALL" then
      -- `SYNCS.CCTL.IVALL` (sm_120a, after `FENCE.VIEW.ASYNC.S`): the sync unit's cached barrier lines dropped,
      -- which the model has none of; the barriers stay valid (inferred).
      some (.next s)
    else some (.fault bad)
  | "UTMALDG" | "UTMASTG" | "UTMAREDG" => some (if uniform then lift (utma k env · i) else .next s)
  | "UBLKCP" => some (if uniform then lift (ublkcp env · i) else .next s)
  | "UBLKRED" => some (if uniform then lift (ublkred env · i) else .next s)
  | "STAS" | "REDAS" => some (if on == 0 then .next s else lift (stas env · on i))
  | "UTMACMDFLUSH" => some (.sync (.asyncCommit .bulk) s)
  -- Prefetches (`UTMAPF`, `UBLKPF.L2 [URa], URb`: PTX `cp.async.bulk.prefetch`) and cache
  -- control change nothing the model holds.
  | "UTMAPF" | "UTMACCTL" | "UBLKPF" => some (.next s)
  | "HGMMA" | "QGMMA" | "IGMMA" | "BGMMA" =>
    if on == 0 then some (.next s) else
    match gmma env w i with
    | .error msg => some (.fault s!"{i.opcode}{i.mods.toList} at pc {i.pc}: {msg}")
    -- `HGMMA` has no read counter: its operands are read at issue.
    | .ok (op, wb) => some (.async (fun _ => op) s wb)
  | "UTCHMMA" | "UTCQMMA" | "UTCIMMA" | "UTCOMMA" =>
    some (if uniform then lift (utcmma env · i) else .next s)
  | "UTCBAR" => some (if uniform then lift (utcbar env · i) else .next s)
  | "LDTM" | "STTM" =>
    if on == 0 then some (.next s) else
    some (match tmemMove env w on i s with
      | .ok o => o
      | .error msg => .fault s!"{i.opcode}{i.mods.toList} at pc {i.pc}: {msg}")
  | "UTCATOMSWS" =>
    if !uniform then some (.next s) else
    some (match utcatomsws w i s with
      | .ok o => o
      | .error msg => .fault s!"{i.opcode}{i.mods.toList} at pc {i.pc}: {msg}")
  -- `tcgen05.relinquish_alloc_permit`: no allocation permits in the model.
  | "UVIRTCOUNT" => some (.next s)
  | "UTCCP" => some (if uniform then lift (utccp env · i) else .next s)
  | "UTCSHIFT" => some (if uniform then lift (utcshift env · i) else .next s)
  | "UGETNEXTWORKID" => some (if uniform then lift (getNextWorkId env · i) else .next s)
  | "WARPGROUP" =>
    -- `WARPGROUP.WAIT` after `WARPGROUP.ARRIVE`: the warpgroup's own barrier, its four warps'
    -- arrivals (inferred). The barrier follows the CTA's 16 named ones.
    if i.hasMod "WAIT" then some (.sync (.bar (16 + env.ctx.warp / 4) (some 128) false) s)
    else if i.hasMod "ARRIVE" then some (.next s)
    else if i.hasMod "DEPBAR" then
      match (i.operands[1]? : Option Operand) with
      | some (.imm n) => some (.sync (.asyncWait .wgmma n.toNat) s)
      | _ => some (.fault bad)
    else some (.fault bad)
  | _ => none

end NV.SASS
