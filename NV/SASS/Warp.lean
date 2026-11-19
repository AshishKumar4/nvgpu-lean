import NV.Fact
import NV.SASS.Syntax
import NV.Machine.Effect

/-!
# A warp's architectural state

Registers are lane-major per register (`r[reg * 32 + lane]`); predicates are lane masks; uniform
registers and predicates have one value per warp. Each lane has its own program counter
(independent thread scheduling, CUDA Programming Guide §"Independent Thread Scheduling"); the
lanes that share the selected program counter execute the next instruction together.
-/

namespace NV.SASS

/-- Where a warp runs: launch geometry and identifiers the special registers read. -/
structure Ctx where
  ntid : Nat × Nat × Nat
  nctaid : Nat × Nat × Nat
  ctaid : Nat × Nat × Nat
  /-- Warp index within its CTA. -/
  warp : Nat
  /-- Threads in the CTA (the last warp may be partial). -/
  threads : Nat
  smid : Nat := 0
  /-- The warp's slot on its SM (`SR_VIRTID` bits 8–14, PTX `%warpid`); its sub-partition is it
  modulo the sub-partitions. -/
  warpSlot : Nat := 0
  /-- The CTA's rank in its cluster (`SR_CgaCtaId`, PTX `%cluster_ctarank`). -/
  clusterRank : Nat := 0
  /-- The CTAs in its cluster (PTX `%cluster_nctarank`). -/
  clusterSize : Nat := 1
  /-- The CTA's shared memory, reserved, static and dynamic bytes (`SR_SMEMSZ`;
  `%total_smem_size` is it minus the reserved bytes). -/
  smemSize : Nat := 0
  deriving Repr, Inhabited

def fullMask : UInt32 := 0xFFFFFFFF

structure Warp where
  /-- Per lane: index into the kernel's code of its next instruction, when the warp is diverged
  (`conv` is `none`); read it with `Warp.pc`. -/
  pcs : Array Nat
  /-- The one instruction index every live lane is at, while the warp is converged. -/
  conv : Option Nat := some 0
  /-- Lanes that exist and have not exited. -/
  live : UInt32
  r : Array UInt32
  ur : Array UInt32
  /-- `p[i]`: lanes where predicate `Pi` holds (`P0`–`P6`). -/
  p : Array UInt32
  /-- Uniform predicates `UP0`–`UP6`. -/
  up : Array Bool
  /-- Convergence barrier registers: lane masks. -/
  b : Array UInt32
  /-- Lanes stopped at a `BSYNC` or `WARPSYNC` until the lanes they wait for arrive. -/
  waiting : UInt32 := 0
  /-- Device `printf` records the warp produced, in order: the format string's address and the
  argument bytes (`NV.Printf`). Host output, not memory: no instruction reads it back. -/
  printf : Array (UInt64 × ByteArray) := #[]
  /-- The answer of the warp's last barrier reduction (`BAR.RED`), which `B2R.RESULT` reads. -/
  barResult : UInt32 := 0
  /-- Lanes of a `printf` call the warp just made, which the engine holds it for
  (`SyncParams.printfPs`) and then clears. -/
  printing : Nat := 0
  deriving Inhabited

namespace Warp

/-- Uniform registers the model gives a warp: the most any target encodes (`URZ` is `UR255`). -/
def uniformRegs : Fact Nat :=
  { value := 255, kind := .inferred, source := { ref := "fixtures", loc := "instruction encodings" }
    note := "" }

def init (regs lanes : Nat) : Warp where
  pcs := Array.replicate 32 0
  live := if lanes ≥ 32 then fullMask else ((1 : UInt32) <<< lanes.toUInt32) - 1
  r := Array.replicate (regs * 32) 0
  ur := Array.replicate uniformRegs.value 0
  p := Array.replicate 7 0
  up := Array.replicate 7 false
  b := Array.replicate 16 0

def done (w : Warp) : Bool := w.live == 0

def laneOn (m : UInt32) (l : Nat) : Bool := (m >>> l.toUInt32) &&& 1 == 1

/-- Lane `l`'s next instruction index. -/
def pc (w : Warp) (l : Nat) : Nat := w.conv.getD w.pcs[l]!

/-- The instruction index to execute next and the lanes executing it: the live lanes whose
program counter is the smallest, among the lanes not waiting at a convergence barrier if there are
any. (Which divergent path the hardware runs first is not documented; this choice is ours.) -/
def current (w : Warp) : Option (Nat × UInt32) := Id.run do
  if w.live == 0 then return none
  if let some pc := w.conv then
    if w.waiting &&& w.live == 0 then return some (pc, w.live)
  let runnable := w.live &&& ~~~w.waiting
  let pool := if runnable != 0 then runnable else w.live
  let mut best : Option Nat := none
  for l in [0:32] do
    if laneOn pool l then
      let pc := w.pc l
      best := some (match best with | some b => min b pc | none => pc)
  match best with
  | none => return none
  | some pc =>
    let mut m : UInt32 := 0
    for l in [0:32] do
      if laneOn pool l && w.pc l == pc then m := m ||| ((1 : UInt32) <<< l.toUInt32)
    return some (pc, m)

def getR (w : Warp) (reg lane : Nat) : UInt32 :=
  if reg == 255 then 0 else w.r[reg * 32 + lane]!

def setR (w : Warp) (reg lane : Nat) (v : UInt32) : Warp :=
  if reg == 255 then w else { w with r := w.r.set! (reg * 32 + lane) v }

def getUR (w : Warp) (reg : Nat) : UInt32 := if reg == 255 then 0 else w.ur[reg]!

def setUR (w : Warp) (reg : Nat) (v : UInt32) : Warp :=
  if reg == 255 then w else { w with ur := w.ur.set! reg v }

def getP (w : Warp) (i lane : Nat) : Bool := if i == 7 then true else laneOn w.p[i]! lane

def setP (w : Warp) (i lane : Nat) (v : Bool) : Warp :=
  if i == 7 then w else
  let bit := (1 : UInt32) <<< lane.toUInt32
  { w with p := w.p.set! i (if v then w.p[i]! ||| bit else w.p[i]! &&& ~~~bit) }

def getUP (w : Warp) (i : Nat) : Bool := if i == 7 then true else w.up[i]!

def setUP (w : Warp) (i : Nat) (v : Bool) : Warp :=
  if i == 7 then w else { w with up := w.up.set! i v }

/-- Move the lanes in `m` to instruction index `pc`. -/
def jump (w : Warp) (m : UInt32) (pc : Nat) : Warp := Id.run do
  if m &&& w.live == 0 then return w
  -- All live lanes move together: the warp stays (or becomes) converged.
  if w.live &&& ~~~m == 0 then return { w with conv := some pc }
  let mut pcs := match w.conv with
    | some p => Array.replicate 32 p
    | none => w.pcs
  for l in [0:32] do
    if laneOn m l then pcs := pcs.set! l pc
  -- Lanes that met again (at a reconvergence point) make the warp converged.
  let same := (List.range 32).all fun l => !laneOn w.live l || pcs[l]! == pc
  return { w with pcs, conv := if same then some pc else none }

end Warp

/-- A unit of warp state. The sinks (`RZ`, `URZ`, `PT`, `UPT`) are never locations. -/
inductive Loc where
  | r (n : Nat)
  | ur (n : Nat)
  | p (n : Nat)
  | up (n : Nat)
  | b (n : Nat)
  /-- The warp's `printf` records. -/
  | log
  /-- The answer of the warp's last barrier reduction. -/
  | bar
  deriving DecidableEq, Repr, Inhabited, Hashable

namespace Loc

def ofReg (g : Reg) : Option Loc :=
  if g.isSink then none else
  match g.file with
  | .r => some (.r g.idx) | .ur => some (.ur g.idx) | .p => some (.p g.idx)
  | .up => some (.up g.idx) | .b => some (.b g.idx)

/-- A convergence barrier register. -/
def isB : Loc → Bool
  | .b _ => true
  | _ => false

/-- Registers `g`, `g+1`, …, `g+k-1`; nothing for a sink (`RZ` as a pair reads zeros). -/
def span (g : Reg) (k : Nat) : Array Loc :=
  if g.isSink then #[] else
  (Array.range k).filterMap fun j => ofReg { g with idx := g.idx + j }

end Loc

namespace Warp

/-- A location's contents, lane by lane (one word for the warp-wide ones). -/
def get (w : Warp) : Loc → Fin 32 → UInt32
  | .r n => fun l => w.getR n l
  | .ur n => fun _ => w.getUR n
  | .p n => fun _ => w.p[n]?.getD 0
  | .up n => fun _ => if w.up[n]?.getD false then 1 else 0
  | .b n => fun _ => w.b[n]?.getD 0
  | .log => fun _ => w.printf.size.toUInt32
  | .bar => fun _ => w.barResult

/-- The control state: program counters, live and waiting lanes. -/
def ctrl (w : Warp) : Array Nat × Option Nat × UInt32 × UInt32 := (w.pcs, w.conv, w.live, w.waiting)

def AgreeOn (ls : Array Loc) (w w' : Warp) : Prop := ∀ l ∈ ls, w.get l = w'.get l

/-- The sizes of the register files: every warp of a launch has the same, and no instruction
changes them (`applyWrites_shape`). -/
def shape (w : Warp) : Nat × Nat × Nat × Nat × Nat :=
  (w.r.size, w.ur.size, w.p.size, w.up.size, w.b.size)

end Warp

/-- What an instruction's semantics reads of a warp: the contents of its footprint's read
locations, in the footprint's order (`View.of`; a view of more locations must list the reads first,
in that order, for the compiled forms' indices to hold). A decoder reads a location through its index in
that order (`View.read`), found when the instruction is decoded (`View.slot`), or on a cold path by
search (`View.get`); it reads nothing else, so an outcome depends only on the reads
(`Op.readsWithin`). -/
structure View where
  reads : Array Loc
  vals : Array (Fin 32 → UInt32)
  /-- One more than the index of `.r n` among the reads at `n`, or 0: `get`'s lookup of a vector
  register, a function of the reads alone. -/
  regSlot : Array Nat := #[]
  deriving Inhabited

namespace View

/-- The index of `l` among the reads. -/
def slot (reads : Array Loc) (l : Loc) : Option Nat := reads.idxOf? l

/-- `regSlot` of a view of `reads`: for each vector register, its first index among them. -/
def regSlots (reads : Array Loc) : Array Nat :=
  (reads.size.fold (init := Array.replicate 256 0) fun k _ acc =>
    match reads[reads.size - 1 - k]! with
    | .r n => if n < 256 then acc.set! n (reads.size - k) else acc
    | _ => acc)

def of (reads : Array Loc) (w : Warp) : View := ⟨reads, reads.map w.get, regSlots reads⟩

theorem of_congr {reads : Array Loc} {w w' : Warp} (h : Warp.AgreeOn reads w w') :
    of reads w = of reads w' := by
  simp only [of, mk.injEq, true_and, and_true]
  apply Array.ext (by simp)
  intro i h1 h2
  simp only [Array.getElem_map]
  exact h _ (Array.getElem_mem _)

/-- Read location number `k` in lane `lane`. -/
@[inline] def read (v : View) (k lane : Nat) : UInt32 :=
  match v.vals[k]? with
  | some f => f ⟨lane % 32, Nat.mod_lt _ (by decide)⟩
  | none => 0

/-- A location's contents in a lane, found by search (0 if it is not read); a vector register
through `regSlot`. -/
def get (v : View) (l : Loc) (lane : Nat) : UInt32 :=
  match l with
  | .r n => match v.regSlot[n]? with
    | some (k + 1) => v.read k lane
    | _ => 0
  | _ => match slot v.reads l with
    | some k => v.read k lane
    | none => 0

def getR (v : View) (reg lane : Nat) : UInt32 := if reg == 255 then 0 else v.get (.r reg) lane
def getUR (v : View) (reg : Nat) : UInt32 := if reg == 255 then 0 else v.get (.ur reg) 0
/-- The lanes where predicate `Pi` holds. -/
def pmask (v : View) (i : Nat) : UInt32 := if i == 7 then fullMask else v.get (.p i) 0
def getP (v : View) (i lane : Nat) : Bool := Warp.laneOn (v.pmask i) lane
def getUP (v : View) (i : Nat) : Bool := if i == 7 then true else v.get (.up i) 0 != 0
def getB (v : View) (n : Nat) : UInt32 := v.get (.b n) 0
def barResult (v : View) : UInt32 := v.get .bar 0

end View

namespace Warp

/-- The warp an instruction's semantics continues from: its control state, no register contents
(a decoder reads them through its `View`), and the convergence barriers zero (an instruction that
sets one at issue sets it here). -/
def blank (w : Warp) : Warp :=
  { w with r := #[], ur := #[], p := #[], up := #[], b := Array.replicate 16 0, printf := #[],
           barResult := 0, printing := 0 }

theorem blank_congr {w w' : Warp} (h : w.ctrl = w'.ctrl) : w.blank = w'.blank := by
  cases w; cases w'
  simp only [ctrl, Prod.mk.injEq] at h
  obtain ⟨h1, h2, h3, h4⟩ := h
  subst h1 h2 h3 h4
  rfl

/-- The state a memory instruction's request is read from: its vector registers when it reads its
sources, after issue (the read dependence counter guards them), but its uniform registers and all
predicates as at issue. -/
@[inline] def readState (issue w : Warp) : Warp := { w with ur := issue.ur, up := issue.up, p := issue.p }

end Warp

end NV.SASS
