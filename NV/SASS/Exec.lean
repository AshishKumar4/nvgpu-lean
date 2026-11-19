import NV.SASS.Warp
import NV.Memory.Store
import NV.Machine.Generic
import NV.Machine.Texture

/-!
# Executing SASS operands

What every opcode's semantics shares: the environment a warp executes in, reading and writing
operands lane by lane, address and guard evaluation. `NV.SASS.exec` (`Sem`) and the async
opcodes (`Async`) are built on these.
-/

namespace NV.SASS

open NV

/-- What `exec` reads besides the warp: the warp's place and the constant banks. -/
structure Env where
  ctx : Ctx
  /-- Constant bank 0: launch constants and parameters (`Kernel.paramBase`). -/
  const0 : Store
  /-- The SM clock at issue (`SR_CLOCKLO`/`HI`, PTX `%clock64`). -/
  clock : Nat := 0
  /-- Nanoseconds since launch at issue (`SR_GLOBALTIMERLO`/`HI`, PTX `%globaltimer`). -/
  timeNs : Nat := 0
  /-- The launch's texture and surface objects (`NV.Tex`). -/
  textures : Array NV.Tex.Texture := #[]
  /-- Where the parameter block sits in device memory, as the driver backs constant bank 0 with
  it: the async unit fetches a parameter's tensor map from there. -/
  paramAddr : UInt64 := 0
  deriving Inhabited

namespace Exec

def toU64 (lo hi : UInt32) : UInt64 := lo.toUInt64 ||| (hi.toUInt64 <<< 32)

/-- An address offset as a 64-bit two's-complement word, without `Int` modular arithmetic. -/
def offset64 (off : Int) : UInt64 :=
  if off ≥ 0 then off.toNat.toUInt64 else 0 - (-off).toNat.toUInt64

/-- The high word of the generic addresses of shared memory (`SR_SWINHI`: a generic shared
address is this word over the 32-bit shared address). Ours: `NV.Windows`'s shared window. -/
def sharedWindowHi : UInt32 := (({} : NV.Windows).shared >>> 32).toUInt32

def specialReg (env : Env) (lane : Nat) (name : String) : Option UInt32 :=
  let c := env.ctx
  let tid := c.warp * 32 + lane
  let (nx, ny, _) := c.ntid
  match name with
  | "SR_TID.X" => some (tid % nx).toUInt32
  | "SR_TID.Y" => some ((tid / nx) % ny).toUInt32
  | "SR_TID.Z" => some (tid / (nx * ny)).toUInt32
  | "SR_CTAID.X" => some c.ctaid.1.toUInt32
  | "SR_CTAID.Y" => some c.ctaid.2.1.toUInt32
  | "SR_CTAID.Z" => some c.ctaid.2.2.toUInt32
  | "SR_LANEID" => some lane.toUInt32
  | "SR_VIRTUALSMID" | "SR_SMID" => some c.smid.toUInt32
  | "SR_VIRTID" => some (c.warpSlot.toUInt32 <<< 8)
  | "SR_CgaCtaId" => some c.clusterRank.toUInt32
  | "SR_SWINHI" => some sharedWindowHi
  | "SR_SMEMSZ" => some c.smemSize.toUInt32
  -- Indexes sm_120a's per-cluster-size tables in constant bank 0 (`c[0x0][R + 0x2ac]`); the
  -- model writes the launch's values at the tables' first entries, so 0 (inferred).
  | "SR_CgaSize" => some 0
  | "SR_SWINLO" => some 0
  | "SR_LTMASK" => some (((1 : UInt64) <<< lane.toUInt64) - 1).toUInt32
  | "SR_LEMASK" => some (((1 : UInt64) <<< (lane + 1).toUInt64) - 1).toUInt32
  | "SR_EQMASK" => some ((1 : UInt32) <<< lane.toUInt32)
  | "SR_GTMASK" => some (~~~((((1 : UInt64) <<< (lane + 1).toUInt64) - 1).toUInt32))
  | "SR_GEMASK" => some (~~~((((1 : UInt64) <<< lane.toUInt64) - 1).toUInt32))
  | "SRZ" => some 0
  | "SR_CLOCKLO" => some env.clock.toUInt32
  | "SR_CLOCKHI" => some (env.clock >>> 32).toUInt32
  | "SR_GLOBALTIMERLO" => some env.timeNs.toUInt32
  | "SR_GLOBALTIMERHI" => some (env.timeNs >>> 32).toUInt32
  | _ => none

/-- A 32-bit source operand for one lane. -/
def src (env : Env) (w : View) (lane : Nat) : Operand → Option UInt32
  | .reg ⟨.r, i⟩ m =>
    let v := w.getR i lane
    some (if m.not then ~~~v else if m.neg then 0 - v else v)
  | .reg ⟨.ur, i⟩ m =>
    let v := w.getUR i
    some (if m.not then ~~~v else if m.neg then 0 - v else v)
  | .imm b => some b.toUInt32
  | .cbank 0 none off => some (env.const0.readWord off.toNat)
  | .cbank 0 (some ⟨.r, i⟩) off => some (env.const0.readWord ((w.getR i lane).toNat + off.toNat))
  | .cbank 0 (some ⟨.ur, i⟩) off => some (env.const0.readWord ((w.getUR i).toNat + off.toNat))
  | .name s => specialReg env lane s
  | .neg o => (src env w lane o).map (0 - ·)
  | .not o => (src env w lane o).map (~~~ ·)
  | _ => none

/-- The high word of a 64-bit source pair: the next register, or the next constant word. -/
def srcHi (env : Env) (w : View) (lane : Nat) : Operand → Option UInt32
  | .reg ⟨.r, 255⟩ _ => some 0
  | .reg ⟨.r, i⟩ _ => some (w.getR (i + 1) lane)
  | .reg ⟨.ur, 255⟩ _ => some 0
  | .reg ⟨.ur, i⟩ _ => some (w.getUR (i + 1))
  | .cbank 0 none off => some (env.const0.readWord (off.toNat + 4))
  | .cbank 0 (some ⟨.r, i⟩) off => some (env.const0.readWord ((w.getR i lane).toNat + off.toNat + 4))
  | .cbank 0 (some ⟨.ur, i⟩) off => some (env.const0.readWord ((w.getUR i).toNat + off.toNat + 4))
  | .imm b => some (b >>> 32).toUInt32
  | _ => none

/-- A 64-bit source: a register pair (with `-` and `~` applied to the pair), a constant word
pair, or an immediate sign-extended from its 32-bit encoding. -/
def src64 (env : Env) (w : View) (lane : Nat) : Operand → Option UInt64
  | .reg r m => do
    let v := toU64 (← src env w lane (.reg r {})) (← srcHi env w lane (.reg r {}))
    some (if m.not then ~~~v else if m.neg then 0 - v else v)
  | .imm b => some (b.toUInt32.toInt32.toInt64.toUInt64)
  | o => do toU64 (← src env w lane o) (← srcHi env w lane o)

/-- A predicate source: a predicate, or with `.SIGN` a register's sign bit (`PLOP3.LUT P6, PT,
R3.SIGN, R10.SIGN, R13.SIGN, …`). -/
def pred (w : View) (lane : Nat) : Operand → Option Bool
  | .reg ⟨.p, i⟩ m => some (w.getP i lane != m.not)
  | .reg ⟨.up, i⟩ m => some (w.getUP i != m.not)
  | .reg ⟨.r, i⟩ m => if m.suffix.contains "SIGN" then some ((w.getR i lane >>> 31 == 1) != m.not) else none
  | .reg ⟨.ur, i⟩ m => if m.suffix.contains "SIGN" then some ((w.getUR i >>> 31 == 1) != m.not) else none
  | _ => none

def isPred : Operand → Bool
  | .reg ⟨.p, _⟩ _ | .reg ⟨.up, _⟩ _ => true
  | _ => false

def lanes (m : UInt32) : List Nat := Id.run do
  let mut out := []
  for l in [0:32] do
    if Warp.laneOn m (31 - l) then out := (31 - l) :: out
  return out

/-- The three-input bitwise function with truth table `lut` (`LOP3.LUT`). -/
def lop3 (a b c : UInt32) (lut : UInt32) : UInt32 := Id.run do
  let mut r : UInt32 := 0
  for k in [0:8] do
    if (lut >>> k.toUInt32) &&& 1 == 1 then
      let ma := if k &&& 4 != 0 then a else ~~~a
      let mb := if k &&& 2 != 0 then b else ~~~b
      let mc := if k &&& 1 != 0 then c else ~~~c
      r := r ||| (ma &&& mb &&& mc)
  return r

def cmp (signed : Bool) (op : String) (a b : UInt32) : Option Bool :=
  let (x, y) : Int × Int :=
    if signed then (a.toInt32.toInt, b.toInt32.toInt) else (a.toNat, b.toNat)
  match op with
  | "EQ" => some (x == y) | "NE" => some (x != y)
  | "LT" => some (x < y) | "LE" => some (x ≤ y)
  | "GT" => some (x > y) | "GE" => some (x ≥ y)
  | "T" => some true | "F" => some false
  | _ => none

def boolOp (op : String) (x y : Bool) : Option Bool :=
  match op with
  | "AND" => some (x && y) | "OR" => some (x || y) | "XOR" => some (x != y)
  | _ => none

/-- Funnel shift (`SHF`): shift the 64-bit `hi:lo` and return its low or high word. -/
def shf (left : Bool) (signed : Bool) (wide : Bool) (hiWord : Bool) (lo hi s : UInt32) : UInt32 :=
  let x := toU64 lo hi
  let bits : UInt64 := if wide then 64 else 32
  let n := min s.toUInt64 bits
  let r : UInt64 :=
    if left then (if n ≥ 64 then 0 else x <<< n)
    else if signed then (x.toInt64 >>> (min n 63).toInt64).toUInt64
    else (if n ≥ 64 then 0 else x >>> n)
  if hiWord then (r >>> 32).toUInt32 else r.toUInt32

/-- Bytes and sign extension of a load or store from its modifiers. -/
def width (i : Instr) : Nat × Bool :=
  if i.hasMod "U8" then (1, false) else if i.hasMod "S8" then (1, true)
  else if i.hasMod "U16" then (2, false) else if i.hasMod "S16" then (2, true)
  else if i.hasMod "64" then (8, false) else if i.hasMod "128" then (16, false)
  else if i.hasMod "256" then (32, false)
  else (4, false)

/-- The byte address of a `[R.64+UR+off]` / `desc[UR][R.64+off]` operand for one lane. -/
def addr (w : View) (lane : Nat) : Operand → Option UInt64
  | .mem base wide ureg _ off =>
    let b : UInt64 := match base with
      | some ⟨.r, i⟩ => if wide then toU64 (w.getR i lane) (w.getR (i + 1) lane) else (w.getR i lane).toUInt64
      | some ⟨.ur, i⟩ => if wide then toU64 (w.getUR i) (w.getUR (i + 1)) else (w.getUR i).toUInt64
      | _ => 0
    let u : UInt64 := match ureg with
      | some ⟨.ur, i⟩ => (w.getUR i).toUInt64
      | _ => 0
    -- A 32-bit base's address is 32 bits, its sum wrapping (shared, cluster and local windows).
    let a := b + u + offset64 off
    some (if wide then a else a % (1 <<< 32))
  | .scaled (.mem (some ⟨.r, i⟩) _ ureg _ off) sc =>
    let u : UInt64 := match ureg with
      | some ⟨.ur, j⟩ => (w.getUR j).toUInt64
      | _ => 0
    some ((w.getR i lane).toUInt64 * sc.toUInt64 + u + offset64 off)
  | _ => none

/-- An integer source as a natural number for carry arithmetic: `-x` is `2^32 - x` (so `-0`
carries out, as subtraction by addition of the complement does), `~x` the complement. -/
def srcN (env : Env) (w : View) (lane : Nat) : Operand → Option Nat
  | .reg r m =>
    (src env w lane (.reg r { m with neg := false })).map fun v =>
      if m.neg && !m.not then 2 ^ 32 - v.toNat else v.toNat
  | .neg o => (src env w lane o).map fun v => 2 ^ 32 - v.toNat
  | .not o => (src env w lane o).map fun v => (~~~v).toNat
  | o => (src env w lane o).map (·.toNat)

/-- A shared-memory address, `(rank << 24) | offset` (`ULEA` of `SR_CgaCtaId`):
the executing CTA's own shared memory, or another cluster CTA's. -/
def sharedAddr (env : Env) (a : UInt64) : Space × UInt64 :=
  if a >>> 24 == env.ctx.clusterRank.toUInt64 then (.shared, a % (1 <<< 24)) else (.sharedCluster, a)

/-- Where constant bank 0 holds the local window's base (`cvta.local`): (up to sm_90, from sm_100). -/
def localWindowBank : Fact (Nat × Nat) :=
  { value := (0x20, 0x2f8), kind := .encoded, source := { ref := "fixtures", loc := "tests/sass/sm_120a fixtures, cvta.local" }
    note := "" }

/-- The offset of the local window's base in constant bank 0 for SM version `sm`. -/
def localWindowAt (sm : Nat) : Nat :=
  if sm ≥ 100 then localWindowBank.value.2 else localWindowBank.value.1

/-- The local window's base for a kernel of SM version `sm`. -/
def localWindow (env : Env) (sm : Nat) : UInt64 :=
  let o := localWindowAt sm
  toU64 (env.const0.readWord o) (env.const0.readWord (o + 4))

/-- The space and address a generic address names (PTX ISA §6.4.1.1): the shared window
(`SR_SWINHI` over a shared address, `sharedAddr`), the lane's local window (`localWindow`; 16
MB), or global memory. -/
def genericTarget (env : Env) (sm : Nat) (a : UInt64) : Space × UInt64 :=
  let lw := localWindow env sm
  if (a >>> 32).toUInt32 == sharedWindowHi then sharedAddr env (a &&& 0xFFFFFFFF)
  else if lw != 0 && lw ≤ a && a < lw + ((1 : UInt64) <<< 24) then (.local, a - lw)
  else (.global, a)

/-- A lane index. -/
def lane32 (l : Nat) : Fin 32 := ⟨l % 32, Nat.mod_lt _ (by decide)⟩

/-- `r` with indices `base + k` (`k < n`) set from `vals[k]` for the lanes `k` of `on`, in place. -/
def mergeLanes (base : Nat) (on : UInt32) (vals : Array UInt32) : Nat → Array UInt32 → Array UInt32
  | 0, r => r
  | k + 1, r => mergeLanes base on vals k
      (if (on >>> k.toUInt32) &&& 1 == 1 then r.set! (base + k) vals[k]! else r)

/-- A destination write, its value computed when the instruction issued: a writeback is a list of
them, data rather than a function, so what it can change is evident (`Write.loc`). -/
inductive Write where
  | r (reg : Nat) (lane : Fin 32) (v : UInt32)
  /-- Register `reg` for the lanes of `on`, lane `l` taking `vals[l]`. -/
  | lanes (reg : Nat) (on : UInt32) (vals : Array UInt32)
  | ur (reg : Nat) (v : UInt32)
  | p (idx : Nat) (lane : Fin 32) (v : Bool)
  /-- Predicate `idx` takes `bits` on the lanes of `on`. -/
  | pbits (idx : Nat) (on bits : UInt32)
  | up (idx : Nat) (v : Bool)
  /-- Convergence barrier register `idx`. -/
  | b (idx : Nat) (v : UInt32)
  /-- Append a `printf` record. -/
  | printf (fmt : UInt64) (args : ByteArray)
  /-- Replace the `printf` records (a copy of another state's). -/
  | log (entries : Array (UInt64 × ByteArray))
  /-- The barrier reduction's answer. -/
  | bar (v : UInt32)
  deriving Inhabited

/-- Register `reg` for the lanes of `on` from `vals` (lane `l` from `vals[l]`), in place. -/
def setLanes (w : Warp) (reg : Nat) (on : UInt32) (vals : Array UInt32) : Warp :=
  if reg == 255 then w else
  let r := w.r
  let w := { w with r := #[] }
  { w with r := mergeLanes (reg * 32) on vals 32 r }

/-- Set predicate `i` to `bits` on the lanes of `on`. -/
@[inline] def setBits (w : Warp) (i : Nat) (on bits : UInt32) : Warp :=
  if i == 7 then w else { w with p := w.p.modify i fun p => (p &&& ~~~on) ||| (bits &&& on) }

def Write.apply (w : Warp) : Write → Warp
  | .r i l v => w.setR i l v
  | .lanes i on vs => setLanes w i on vs
  | .ur i v => w.setUR i v
  | .p i l v => w.setP i l v
  | .pbits i on bits => setBits w i on bits
  | .up i v => w.setUP i v
  | .b i v => { w with b := w.b.set! i v }
  | .printf f a => { w with printf := w.printf.push (f, a) }
  | .log es => { w with printf := es }
  | .bar v => { w with barResult := v }

def applyWrites (ws : List Write) (w : Warp) : Warp := ws.foldl Write.apply w

/-- Writes of a 32-bit result to register operand `d` for one lane. -/
def wr (d : Operand) (lane : Nat) (v : UInt32) : Option (List Write) :=
  match d with
  | .reg ⟨.r, i⟩ _ => some [.r i (lane32 lane) v]
  | .reg ⟨.ur, i⟩ _ => some [.ur i v]
  | _ => none

/-- Writes of a 64-bit result to the register pair starting at `d`. -/
def wr64 (d : Operand) (lane : Nat) (v : UInt64) : Option (List Write) :=
  match d with
  | .reg ⟨.r, 255⟩ _ | .reg ⟨.ur, 255⟩ _ => some []
  | .reg ⟨.r, i⟩ _ =>
    some [.r i (lane32 lane) v.toUInt32, .r (i + 1) (lane32 lane) (v >>> 32).toUInt32]
  | .reg ⟨.ur, i⟩ _ => some [.ur i v.toUInt32, .ur (i + 1) (v >>> 32).toUInt32]
  | _ => none

/-- Writes of a predicate result. -/
def wrP (d : Operand) (lane : Nat) (v : Bool) : Option (List Write) :=
  match d with
  | .reg ⟨.p, i⟩ _ => some [.p i (lane32 lane) v]
  | .reg ⟨.up, i⟩ _ => some [.up i v]
  | _ => none

end Exec

open Exec

/-- Advance the lanes in `m` past instruction index `pc`. -/
def Warp.advance (w : Warp) (m : UInt32) (pc : Nat) : Warp := w.jump m (pc + 1)

/-- An instruction's outcome: an `Effect` whose writebacks are data (`Write` lists), so what they
can change is evident. `NV.SASS.Op.run` turns it into the effect. -/
inductive Out where
  | next (s : Warp)
  | reg (s : Warp) (ws : List Write)
  | mem (r : View → Req) (s : Warp) (wb : Resp → List Write)
  | sync (op : SyncOp) (s : Warp)
  | async (op : View → NV.Async.Op) (s : Warp) (wb : Resp → List Write)
  | collect (op : Collective) (mask : UInt32) (contrib : Array UInt32) (s : Warp)
      (wb : Resp → List Write)
  | exit
  | fault (msg : String)
  deriving Inhabited

/-- An instruction's semantics with its form resolved: given the warp's environment, what it reads
of the warp (`View`), the warp it continues from (`Warp.blank`) and the lanes at the instruction,
its outcome. -/
abbrev Run := Env → View → Warp → UInt32 → Out

/-- The lanes of `m` whose guard predicate holds, word-wide. -/
def guarded (w : View) (m : UInt32) (i : Instr) : UInt32 :=
  match i.guard with
  | none => m
  | some (g, neg) =>
    let held : UInt32 := match g.file with
      | .p => w.pmask g.idx
      | .up => if w.getUP g.idx then fullMask else 0
      | _ => fullMask
    m &&& (if neg then ~~~held else held)

/-! ## Operands resolved at decode

The hot path: an operand's form is resolved once, when the kernel is decoded, into a function
that reads it in a lane, so executing an instruction is a loop over lanes with no matching and
no allocation per lane. Each function means exactly what `src`, `srcHi`, `pred` and `addr` mean
for that operand (`Tests/SassSem.lean` and the differential test run through it). -/

/-- A 32-bit source: its value in a lane. -/
abbrev Val := Env → View → Nat → UInt32

namespace Exec

/-- The lane-independent special registers, and the lane-dependent ones, resolved by name. -/
def special? (name : String) : Option Val :=
  match name with
  | "SR_TID.X" => some fun env _ l => let c := env.ctx; ((c.warp * 32 + l) % c.ntid.1).toUInt32
  | "SR_LANEID" => some fun _ _ l => l.toUInt32
  | "SR_CTAID.X" => some fun env _ _ => env.ctx.ctaid.1.toUInt32
  | "SRZ" => some fun _ _ _ => 0
  | _ =>
    -- The rest are rarer: the generic definition, its name matched per read.
    if (specialReg default 0 name).isSome then some fun env _ l => (specialReg env l name).getD 0
    else none

/-- A register's value with its integer modifiers: `~` complements, `-` negates. -/
@[inline] def modInt (m : RegMods) (v : UInt32) : UInt32 :=
  if m.not then ~~~v else if m.neg then 0 - v else v

/-- Where location `l` sits among the instruction's reads, as a reader of it: `none` if it is not
read (the form is then left to the general definition). -/
@[inline] def slotOf (reads : Array Loc) (l : Loc) : Option (View → Nat → UInt32) :=
  (View.slot reads l).map fun k v lane => v.read k lane

/-- `src` for operand `o`, resolved against the instruction's reads. -/
def val? (reads : Array Loc) (o : Operand) : Option Val :=
  match o with
  | .reg ⟨.r, 255⟩ m => let v := modInt m 0; some fun _ _ _ => v
  | .reg ⟨.r, i⟩ m => do
    let f ← slotOf reads (.r i)
    if m.not || m.neg then some fun _ w l => modInt m (f w l) else some fun _ w l => f w l
  | .reg ⟨.ur, 255⟩ m => let v := modInt m 0; some fun _ _ _ => v
  | .reg ⟨.ur, i⟩ m => do let f ← slotOf reads (.ur i); some fun _ w _ => modInt m (f w 0)
  | .imm b => let v := b.toUInt32; some fun _ _ _ => v
  | .cbank 0 none off | .cbank 0 (some ⟨.r, 255⟩) off | .cbank 0 (some ⟨.ur, 255⟩) off =>
    let a := off.toNat; some fun env _ _ => env.const0.readWord a
  | .cbank 0 (some _) _ => some fun env w l => (src env w l o).getD 0
  | .name s => special? s
  | _ => none

/-- `srcHi` for operand `o`, resolved. -/
def hi? (reads : Array Loc) (o : Operand) : Option Val :=
  match o with
  | .reg ⟨.r, 255⟩ _ | .reg ⟨.ur, 255⟩ _ => some fun _ _ _ => 0
  | .reg ⟨.r, i⟩ _ => do let f ← slotOf reads (.r (i + 1)); some fun _ w l => f w l
  | .reg ⟨.ur, i⟩ _ => do let f ← slotOf reads (.ur (i + 1)); some fun _ w _ => f w 0
  | .imm b => let v := (b >>> 32).toUInt32; some fun _ _ _ => v
  | .cbank 0 none off | .cbank 0 (some ⟨.r, 255⟩) off | .cbank 0 (some ⟨.ur, 255⟩) off =>
    let a := off.toNat + 4; some fun env _ _ => env.const0.readWord a
  | _ => (val? reads o).map fun _ env w l => (srcHi env w l o).getD 0

/-- The lanes where predicate operand `o` holds, word-wide. -/
def pmask? (reads : Array Loc) (o : Operand) : Option (View → UInt32) :=
  match o with
  | .reg ⟨.p, 7⟩ m | .reg ⟨.up, 7⟩ m => let v := if m.not then 0 else fullMask; some fun _ => v
  | .reg ⟨.p, i⟩ m => do
    let f ← slotOf reads (.p i)
    some fun w => if m.not then ~~~f w 0 else f w 0
  | .reg ⟨.up, i⟩ m => do
    let f ← slotOf reads (.up i)
    some fun w => if (f w 0 != 0) != m.not then fullMask else 0
  | _ => none

/-- `addr` for address operand `o`, resolved. -/
def addr? (reads : Array Loc) (o : Operand) : Option (View → Nat → UInt64) :=
  let ur (i : Nat) : Option (View → UInt64) :=
    if i == 255 then some fun _ => 0 else (slotOf reads (.ur i)).map fun f w => (f w 0).toUInt64
  let r (i : Nat) : Option (View → Nat → UInt32) :=
    if i == 255 then some fun _ _ => 0 else slotOf reads (.r i)
  match o with
  | .mem base wide ureg _ off => do
    let k := offset64 off
    let u : View → UInt64 ← match ureg with
      | some ⟨.ur, i⟩ => ur i
      | _ => some fun _ => 0
    -- As `addr`: a 32-bit base's address wraps at 32 bits.
    let w32 (a : UInt64) : UInt64 := if wide then a else a % (1 <<< 32)
    match base, wide with
      | some ⟨.r, 255⟩, _ | some ⟨.ur, 255⟩, _ | none, _ => some fun w _ => w32 (u w + k)
      | some ⟨.r, i⟩, true => do
        let lo ← r i; let hi ← r (i + 1)
        some fun w l => toU64 (lo w l) (hi w l) + u w + k
      | some ⟨.r, i⟩, false => do let lo ← r i; some fun w l => w32 ((lo w l).toUInt64 + u w + k)
      | some ⟨.ur, i⟩, true => do
        let lo ← ur i; let hi ← ur (i + 1)
        some fun w _ => (lo w ||| (hi w <<< 32)) + u w + k
      | some ⟨.ur, i⟩, false => do let lo ← ur i; some fun w _ => w32 (lo w + u w + k)
      | some _, _ => some fun _ _ => w32 k
  | .scaled (.mem (some ⟨.r, i⟩) _ ureg _ off) sc => do
    let k := offset64 off
    let u : View → UInt64 ← match ureg with
      | some ⟨.ur, j⟩ => ur j
      | _ => some fun _ => 0
    let a ← r i
    some fun w l => (a w l).toUInt64 * sc.toUInt64 + u w + k
  | _ => none

/-- The executing lane of a uniform-datapath instruction: the first of `on`, as a mask. -/
@[inline] def firstLane (on : UInt32) : UInt32 := on &&& (0 - on)

/-- Per-lane values of `f` for the lanes of `on` (0 elsewhere). -/
@[inline] def valsOf (f : Nat → UInt32) (on : UInt32) : Array UInt32 := Id.run do
  let mut a := Array.mkEmpty 32
  for l in [0:32] do
    a := a.push (if (on >>> l.toUInt32) &&& 1 == 1 then f l else 0)
  return a

/-- Bits of `f` for the lanes of `on`. -/
@[inline] def bitsOf (f : Nat → Bool) (on : UInt32) : UInt32 := Id.run do
  let mut b : UInt32 := 0
  for l in [0:32] do
    if (on >>> l.toUInt32) &&& 1 == 1 && f l then b := b ||| ((1 : UInt32) <<< l.toUInt32)
  return b

/-- The lowest lane of a mask. -/
def lowLane (m : UInt32) : Nat := Id.run do
  for l in [0:32] do
    if (m >>> l.toUInt32) &&& 1 == 1 then return l
  return 0

/-- The writes of a 32-bit result to destination operand `d` for the executing lanes `on`, from
per-lane values computed at issue (`valsOf`); a uniform register takes the lowest lane's. -/
def write32? (d : Operand) : Option (UInt32 → Array UInt32 → List Write) :=
  match d with
  | .reg ⟨.r, 255⟩ _ | .reg ⟨.ur, 255⟩ _ => some fun _ _ => []
  | .reg ⟨.r, i⟩ _ => some fun on vs => [.lanes i on vs]
  | .reg ⟨.ur, i⟩ _ => some fun on vs => if on == 0 then [] else [.ur i vs[lowLane on]!]
  | _ => none

/-- The writes of a 64-bit result to the register pair at `d`, from low and high words. -/
def write64? (d : Operand) : Option (UInt32 → Array UInt32 → Array UInt32 → List Write) :=
  match d with
  | .reg ⟨.r, 255⟩ _ | .reg ⟨.ur, 255⟩ _ => some fun _ _ _ => []
  | .reg ⟨.r, i⟩ _ => some fun on lo hi => [.lanes i on lo, .lanes (i + 1) on hi]
  | .reg ⟨.ur, i⟩ _ => some fun on lo hi =>
    if on == 0 then [] else let l := lowLane on; [.ur i lo[l]!, .ur (i + 1) hi[l]!]
  | _ => none

/-- The writes of a predicate result to `d`, from the bits of the executing lanes `on`. -/
def writeP? (d : Operand) : Option (UInt32 → UInt32 → List Write) :=
  match d with
  | .reg ⟨.p, i⟩ _ => some fun on bits => [.pbits i on bits]
  | .reg ⟨.up, i⟩ _ => some fun on bits =>
    if on == 0 || i == 7 then [] else [.up i ((bits &&& firstLane on) != 0)]
  | _ => none

end Exec

end NV.SASS
