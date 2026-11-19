import NV.Machine.Access
import NV.Num.Arith

/-!
# Collectives

Operations whose result for one thread depends on other threads' operands: the warp collectives
of PTX ISA 9.4 (`shfl.sync` §9.7.10.6, `vote.sync` §9.7.15.10, `match.sync` §9.7.15.11,
`activemask` §9.7.15.12, `redux.sync` §9.7.15.13, `elect.sync` §9.7.15.15, `bar.warp.sync`
§9.7.15.2) and the CTA-wide barrier reduction `bar.red` (§9.7.15.1).

A thread executing one issues `Effect.collect` with its contribution (its operand words). The
machine gathers the contributions of the participating threads and computes every thread's
answer with `Collective.answer`, the one definition shared by the interpreter, the proofs and
the checks of SASS's warp-wide instructions against PTX.
-/

namespace NV

inductive ShflMode where
  | up | down | bfly | idx
  deriving DecidableEq, Repr, Inhabited

inductive VoteMode where
  | all | any | uni | ballot
  deriving DecidableEq, Repr, Inhabited

inductive ReduxOp where
  | add | min | max | and | or | xor
  deriving DecidableEq, Repr, Inhabited

inductive BarRedOp where
  | popc | and | or
  deriving DecidableEq, Repr, Inhabited

inductive Collective where
  /-- Contribution `[a, b, c]`; answer `[d, p]` with `p` 0 or 1. -/
  | shfl (mode : ShflMode)
  /-- Contribution `[a]`, the predicate as 0 or 1; answer `[d]`: the ballot mask, or 0 or 1. -/
  | vote (mode : VoteMode)
  /-- Contribution: the value's words (one for `.b32`, two for `.b64`); answer `[d]`. -/
  | matchAny
  /-- As `matchAny`; answer `[d, p]`. -/
  | matchAll
  /-- Contribution `[src]`; answer `[dst]`. `signed` selects `.s32` for `min`/`max`. -/
  | redux (op : ReduxOp) (signed : Bool)
  /-- `redux.sync.{min,max}{.abs}{.NaN}.f32`: contribution `[src]`, the binary32 bits; answer
  `[dst]`. `+0` is above `-0`; without `.NaN` NaN inputs are ignored (all NaN: the canonical
  NaN), with it any NaN gives the canonical NaN (§9.7.15.13). -/
  | reduxF32 (isMax abs nan : Bool)
  /-- No contribution; answer `[leader lane, p]`. -/
  | elect
  /-- No contribution; answer `[mask of the threads executing it together]`. -/
  | activemask
  /-- `bar.warp.sync`: no contribution, empty answer. -/
  | warpSync
  /-- The warp-wide matrix instructions (`mma.sync`, `ldmatrix`, `stmatrix`, `movmatrix`,
  §9.7.16), which need every lane: each lane contributes its operand words, all of one length,
  and receives every lane's, lane by lane. A missing lane is an error (§9.7.16: "undefined if any
  thread in the warp has exited"). -/
  | gather
  /-- `bar.red` on barrier `id` with `count` threads (`none`: the CTA). Contribution `[c]` (0 or
  1); answer `[d]`: the count of true contributions (`popc`), or 0 or 1. -/
  | barRed (id : Nat) (count : Option Nat) (op : BarRedOp)
  deriving DecidableEq, Repr, Inhabited

namespace Collective

def bit (m : UInt32) (l : Nat) : Bool := (m >>> l.toUInt32) &&& 1 == 1

/-- The mask of participants. -/
def present (c : Array (Option (Array UInt32))) : UInt32 := Id.run do
  let mut m : UInt32 := 0
  for l in [0:min c.size 32] do
    if c[l]!.isSome then m := m ||| ((1 : UInt32) <<< l.toUInt32)
  return m

def word (c : Array (Option (Array UInt32))) (l k : Nat) : UInt32 :=
  ((c[l]?.join.map (·[k]?)).join).getD 0

def ofBool (b : Bool) : UInt32 := if b then 1 else 0

def maskOf (lanes : List Nat) : UInt32 :=
  lanes.foldl (fun m l => m ||| ((1 : UInt32) <<< l.toUInt32)) 0

/-- The source lane of a shuffle and whether it is in range, by the pseudocode of §9.7.10.6. -/
def shflSource (mode : ShflMode) (lane : Nat) (b c : UInt32) : Nat × Bool :=
  let bval := (b &&& 31).toNat
  let cval := (c &&& 31).toNat
  let seg := ((c >>> 8) &&& 31).toNat
  let maxLane := (lane &&& seg) ||| (cval &&& (31 - seg))
  let minLane := lane &&& seg
  let (j, ok) : Int × Bool := match mode with
    | .up => ((lane : Int) - bval, decide ((lane : Int) - bval ≥ maxLane))
    | .down => (lane + bval, lane + bval ≤ maxLane)
    | .bfly => ((lane ^^^ bval : Nat), (lane ^^^ bval) ≤ maxLane)
    | .idx =>
      let j := minLane ||| (bval &&& (31 - seg))
      (j, j ≤ maxLane)
  if ok then (j.toNat, true) else (lane, false)

def reduce (op : ReduxOp) (signed : Bool) (x y : UInt32) : UInt32 :=
  match op with
  | .add => x + y
  | .min => if signed then (if x.toInt32 ≤ y.toInt32 then x else y) else Min.min x y
  | .max => if signed then (if x.toInt32 ≥ y.toInt32 then x else y) else Max.max x y
  | .and => x &&& y
  | .or => x ||| y
  | .xor => x ^^^ y

/-- The answer to participant `l`, given every participant's contribution and the list and mask
of participants. -/
def respond (op : Collective) (contrib : Array (Option (Array UInt32))) (lanes : List Nat)
    (here : UInt32) (l : Nat) : Except String Resp :=
  let w (k i : Nat) := word contrib k i
  match op with
  | .shfl mode =>
    let (j, ok) := shflSource mode l (w l 1) (w l 2)
    if contrib[j]?.join.isNone then
      .error s!"shfl.sync: lane {l} reads lane {j}, which does not participate"
    else pure #[w j 0, ofBool ok]
  | .vote .all => pure #[ofBool (lanes.all (w · 0 != 0))]
  | .vote .any => pure #[ofBool (lanes.any (w · 0 != 0))]
  | .vote .uni => pure #[ofBool (lanes.all (w · 0 != 0) || lanes.all (w · 0 == 0))]
  | .vote .ballot => pure #[maskOf (lanes.filter (w · 0 != 0))]
  | .matchAny => pure #[maskOf (lanes.filter (contrib[·]! == contrib[l]!))]
  | .matchAll =>
    let same := lanes.all (contrib[·]! == contrib[l]!)
    pure #[if same then here else 0, ofBool same]
  | .redux op signed =>
    match lanes with
    | [] => pure #[0]
    | k :: ks => pure #[ks.foldl (fun acc j => reduce op signed acc (w j 0)) (w k 0)]
  | .reduxF32 isMax abs nan =>
    let v (j : Nat) : Nat := let x := (w j 0).toNat; if abs then NV.Num.abs .f32 false x else x
    match lanes with
    | [] => pure #[0]
    | k :: ks =>
      pure #[(ks.foldl (fun acc j => NV.Num.minMax isMax .f32 false nan false acc (v j)) (v k)).toUInt32]
  | .elect =>
    -- The model elects the lowest participating lane (inferred).
    let leader := lanes.head?.getD l
    pure #[leader.toUInt32, ofBool (l == leader)]
  | .activemask => pure #[here]
  | .warpSync => pure #[]
  | .gather =>
    if lanes.length != 32 then .error "a warp-wide matrix instruction needs all 32 lanes"
    else pure ((List.range 32).foldl (fun acc j => acc ++ (contrib[j]!).getD #[]) #[])
  | .barRed _ _ .popc => pure #[(lanes.filter (w · 0 != 0)).length.toUInt32]
  | .barRed _ _ .and => pure #[ofBool (lanes.all (w · 0 != 0))]
  | .barRed _ _ .or => pure #[ofBool (lanes.any (w · 0 != 0))]

/-- Every participant's answer, indexed like `contrib`: entry `l` is `some words` for a
participating lane (warp collectives, `l` the lane id) or thread (`barRed`, `l` the thread's
index among the barrier's participants), `none` otherwise; non-participants get `#[]`. `mask` is
the instruction's member mask (unused by `activemask` and `barRed`). Behaviour the ISA leaves
undefined is an error: a participant outside `mask`, or a shuffle reading a lane that does not
participate. -/
def answer (op : Collective) (mask : UInt32) (contrib : Array (Option (Array UInt32))) :
    Except String (Array Resp) := do
  let here := present contrib
  let warp := !(op matches .barRed .. | .activemask)
  if warp && here &&& ~~~mask != 0 then
    throw s!"{repr op}: a participating thread is not in membermask {mask}"
  let lanes := List.range contrib.size |>.filter (contrib[·]!.isSome)
  (List.range contrib.size).toArray.mapM fun l =>
    if contrib[l]!.isSome then respond op contrib lanes here l else pure #[]

end Collective

end NV
