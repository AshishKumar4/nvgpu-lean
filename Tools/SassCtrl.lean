import NV.SASS.Sem
import NV.SASS.Read

/-!
`lake exe sass-ctrl DIR…`: checks and summarizes the control words of the
`NAME.sass`/`NAME.elf` listings in each directory (`tests/sass/gen.sh` writes them per target).

* Scoreboards: every instruction that sets a write counter is followed, before the first
  instruction that reads one of its destinations in straight-line code, by an instruction whose
  wait mask holds that counter; likewise a read counter before the first overwrite of a source.
  A later memory instruction's overwrite of an earlier one's source needs no wait (the MIO
  queue is in order), nor does an instruction on the other lanes of the producer's guard. The
  count of pairings that hold and fail is reported.
* Warpgroup MMAs: no instruction but a later MMA reads or writes an `HGMMA`'s accumulators
  before the `WARPGROUP.DEPBAR.LE` that retires its group, in straight-line code. A static check
  that needs no inputs, so a compile gate can run it on any sm_90 cubin; `tests/sass/red` holds a
  listing that breaks it (`sass-ctrl tests/sass/red` must fail). An `HGMMA` into RZ is reported
  (a wgmma whose results the program never reads, `tests/sass/red/wgmma-dead`).
* Coverage: every instruction run once by the semantics (`NV.SASS.decode`) from a plain warp
  state; the opcodes whose form it does not define (`no semantics yet`, `unsupported form`) are
  listed, and the opcodes it accepts.
* Distances: per producer opcode with its modifiers, without a write counter (fixed latency), the
  least number of stall cycles between the producer and its first reader in the listing, with no
  counter wait between, with the kernel and address where that least distance
  occurs (a memory instruction's vector reads and a pair's high register read later, and are
  passed over).

Reads and writes are the footprints of `NV.SASS.footprint`. Control flow (branches, calls,
barriers, exits) ends a straight-line run.
-/

open NV NV.SASS

def controlOps : List String :=
  ["BRA", "BRX", "JMP", "JMX", "CALL", "RET", "EXIT", "BSSY", "BSYNC", "BREAK", "WARPSYNC", "BAR",
   "BPT", "KILL", "NOP", "YIELD"]

def ends (i : Instr) : Bool := controlOps.contains i.opcode

def regsOf (ls : Array Loc) : Array Loc := ls.filter fun | .r _ | .ur _ | .p _ | .up _ => true | _ => false

/-- The opcode with its modifiers except operand-size and cache hints, as a class name. -/
def cls (i : Instr) : String := i.full

structure Report where
  waitOk : Nat := 0
  waitBad : Array String := #[]
  readOk : Nat := 0
  readBad : Array String := #[]
  gmmaOk : Nat := 0
  gmmaBad : Array String := #[]
  gmmaDead : Array String := #[]
  /-- Producer class → (least distance, where). -/
  least : Std.HashMap String (Nat × String) := {}
  accepted : Std.HashMap String Unit := {}
  /-- Opcode with modifiers → an example. -/
  rejected : Std.HashMap String String := {}

def plainEnv : Env :=
  { ctx := { ntid := (128, 1, 1), nctaid := (1, 1, 1), ctaid := (0, 0, 0), warp := 0,
             threads := 128 },
    const0 := Store.empty 0 4096 }

def coverage (r : Report) (file : String) (k : Kernel) : Report := Id.run do
  let ops := decode k
  let mut r := r
  for pc in [0:ops.size] do
    let i := k.code[pc]!
    let w := { Warp.init 256 32 with conv := some pc }
    let undefined := match (ops[pc]!).outOf plainEnv w fullMask with
      | .fault m => (m.splitOn "no semantics yet").length > 1 || (m.splitOn "unsupported form").length > 1
      | _ => false
    if undefined then
      unless r.rejected.contains i.full do
        let at_ := s!"{file}:{k.name}:{i.pc}: {(repr i.operands).pretty 1000}"
        r := { r with rejected := r.rejected.insert i.full at_ }
    else r := { r with accepted := r.accepted.insert i.opcode () }
  return r

/-- Whether the instruction at `pc` is a memory instruction (its outcome from a plain warp state is
a memory request): the warp's memory instructions pass the MIO queue in order. -/
def isMem (ops : Array Op) (code : Array Instr) (pc : Nat) : Bool :=
  -- Constant loads go through the MIO queue too (the semantics reads the bank directly), and so
  -- do texture and surface instructions (whose plain state has no object to fetch from).
  ["LDC", "TEX", "TLD", "TLD4", "TXQ", "SULD", "SUST", "SURED"].contains code[pc]!.opcode ||
    (match (ops[pc]!).outOf plainEnv { Warp.init 256 32 with conv := some pc } fullMask with
     | .mem .. | .async .. => true
     | _ => false)

/-- Whether `j` runs on none of the lanes `i` runs on: both guarded by one predicate with
opposite senses, which `between` (the instructions from `i` to `j`) leave unchanged. -/
def disjoint (i j : Instr) (between : Array Instr) : Bool :=
  match i.guard, j.guard with
  | some (p, n), some (q, m) =>
    p == q && n != m && !between.any fun x => (footprint x).writes.contains (.p p.idx)
  | _, _ => false

/-- A warpgroup MMA. -/
def gmma (i : Instr) : Bool := ["HGMMA", "QGMMA", "IGMMA"].contains i.opcode

def survey (r : Report) (file : String) (k : Kernel) : Report := Id.run do
  let code := k.code
  let ops := decode k
  let mut r := coverage r file k
  for n in [0:code.size] do
    let i := code[n]!
    if ends i then continue
    let fp := footprint i
    let dsts := regsOf fp.writes
    -- A read counter covers the vector registers a memory instruction reads after issue; its
    -- uniform registers and predicates are read at issue.
    let srcs := fp.reads.filter fun | .r _ => true | _ => false
    -- Scoreboard pairing for a write counter. `DEPBAR.LE SBk, c` waits until at most `c` of the
    -- counter's events are pending; they release in order, so it waits for this one when at
    -- least `c` producers on the counter follow it.
    if let some b := i.ctrl.wbar then
      let mut waited := false
      let mut later := 0
      for m in [n + 1:code.size] do
        let j := code[m]!
        if (j.ctrl.wait >>> b.val) &&& 1 == 1 then waited := true
        if j.opcode == "DEPBAR" && j.operands[0]? == some (Operand.name s!"SB{b.val}") then
          if let some (Operand.imm c) := j.operands[1]? then
            if later ≥ c.toNat then waited := true
        if j.ctrl.wbar == some b then later := later + 1
        if (regsOf (footprint j).reads).any dsts.contains && !disjoint i j (code.extract n m) then
          if waited then r := { r with waitOk := r.waitOk + 1 }
          else r := { r with waitBad := r.waitBad.push s!"{file} {k.name} {i.pc} {i.full}: reader {j.pc} {j.full} without waiting on SB{b.val}" }
          break
        if waited || ends j then break
    if let some b := i.ctrl.rbar then
      let mut waited := false
      let mem := isMem ops code n
      for m in [n + 1:code.size] do
        let j := code[m]!
        if (j.ctrl.wait >>> b.val) &&& 1 == 1 then waited := true
        -- A later memory instruction may overwrite an earlier one's source unwaited: the earlier
        -- reads its sources first (the engine's hazard check follows the same rule).
        if mem && isMem ops code m then
          if waited then break
          continue
        if (regsOf (footprint j).writes).any srcs.contains && !disjoint i j (code.extract n m) then
          if waited then r := { r with readOk := r.readOk + 1 }
          else r := { r with readBad := r.readBad.push s!"{file} {k.name} {i.pc} {i.full}: overwrite {j.pc} {j.full} without waiting on SB{b.val}" }
          break
        if waited || ends j then break
    -- A warpgroup MMA's accumulators are the tensor unit's until a `WARPGROUP.DEPBAR.LE gsb0, k`
    -- retires its group (the first group committed, by a `gsb0` operand, at or after it; retired
    -- once `k` later groups are committed): no other instruction may read or write them before
    -- (PTX ISA §9.7.17.5.5: accessing them before `wgmma.wait_group` is undefined). Later MMAs
    -- chain on them legally.
    -- An MMA into RZ: a wgmma whose results the program never reads.
    if gmma i && (match i.operands[0]? with | some (Operand.reg ⟨.r, 255⟩ _) => true | _ => false) then
      r := { r with gmmaDead := r.gmmaDead.push s!"{file} {k.name}: HGMMA with an RZ destination at {i.pc}: the wgmma's results are unused" }
    if gmma i then
      let accs := regsOf fp.writes
      let mut committed := false
      let mut later := 0
      let mut retired := false
      for m in [n:code.size] do
        let j := code[m]!
        if gmma j && j.operands.contains (.name "gsb0") then
          if committed then later := later + 1 else committed := true
        if m == n then continue
        if j.opcode == "WARPGROUP" && j.hasMod "DEPBAR" then
          if let some (Operand.imm k) := j.operands[1]? then
            if committed && later ≥ k.toNat then retired := true; break
        let fj := footprint j
        if !gmma j && (regsOf (fj.reads ++ fj.writes)).any accs.contains then
          r := { r with gmmaBad := r.gmmaBad.push s!"{file} {k.name} {i.pc} {i.full}: {j.pc} {j.full} touches its accumulators before the group retires" }
          break
        if ends j then break
      if retired then r := { r with gmmaOk := r.gmmaOk + 1 }
    -- Least encoded distance of a fixed-latency producer (a memory instruction is never one). Only a reader
    -- the stall counts alone order bounds the latency, so the scan stops at any wait on a counter
    -- (a later instruction of the producer's pipe, on a counter, may cover it), and passes over a
    -- memory instruction's vector reads (the MIO queue reads them after issue, `isMem`) and a 64-bit
    -- operand's odd half (a pair's high register: an `IMAD.WIDE`'s addend, a `TLD`'s coordinates),
    -- which the reader takes after its low half.
    if i.ctrl.wbar.isNone && !dsts.isEmpty && !isMem ops code n then
      let mut dist := i.ctrl.stall
      for m in [n + 1:code.size] do
        let j := code[m]!
        if ends j || j.ctrl.wait != 0 then break
        let fj := footprint j
        let reads := regsOf fj.reads
        let late (d : Loc) : Bool := match d with
          | .r x => isMem ops code m || (x % 2 == 1 && reads.contains (.r (x - 1)))
          | _ => false
        if reads.any fun d => dsts.contains d && !late d then
          let key := cls i
          let at_ := s!"{file}:{k.name}:{i.pc}"
          r := match r.least.get? key with
            | some (d, _) => if dist < d then { r with least := r.least.insert key (dist, at_) } else r
            | none => { r with least := r.least.insert key (dist, at_) }
          break
        if (regsOf fj.writes).any dsts.contains then break
        dist := dist + j.ctrl.stall
  return r

def main (dirs : List String) : IO UInt32 := do
  let mut r : Report := {}
  let mut kernels := 0
  let mut instrs := 0
  for d in dirs do
    let entries ← System.FilePath.readDir d
    for e in entries.qsort (·.path.toString < ·.path.toString) do
      if e.path.extension != some "sass" then continue
      let elf := e.path.withExtension "elf"
      unless ← elf.pathExists do continue
      match Read.kernels (← IO.FS.readFile e.path) (← IO.FS.readFile elf) with
      | .error msg => IO.println s!"UNREAD {e.path}: {msg}"
      | .ok ks =>
        for k in ks do
          kernels := kernels + 1
          instrs := instrs + k.code.size
          r := survey r (e.path.fileName.getD "") k
  IO.println s!"{kernels} kernels, {instrs} instructions"
  IO.println s!"write counters: {r.waitOk} readers wait on the counter, {r.waitBad.size} do not"
  for b in r.waitBad.toList.take 20 do IO.println s!"  {b}"
  IO.println s!"read counters: {r.readOk} overwrites wait on the counter, {r.readBad.size} do not"
  for b in r.readBad.toList.take 20 do IO.println s!"  {b}"
  IO.println s!"warpgroup MMAs: {r.gmmaOk} retire before their accumulators are touched, {r.gmmaBad.size} do not"
  for b in r.gmmaBad.toList.take 20 do IO.println s!"  {b}"
  IO.println s!"warpgroup MMAs into RZ: {r.gmmaDead.size}"
  for b in r.gmmaDead.toList.take 20 do IO.println s!"  {b}"
  IO.println s!"\naccepted opcodes ({r.accepted.size}): {" ".intercalate (r.accepted.toList.map (·.1) |>.toArray.qsort (· < ·) |>.toList)}"
  IO.println s!"forms without semantics ({r.rejected.size}):"
  for (f, at_) in r.rejected.toList.toArray.qsort (·.1 < ·.1) do IO.println s!"  {f}  ({at_})"
  IO.println "\n| producer | least encoded distance | where |\n|---|---|---|"
  let rows := r.least.toList.toArray.qsort (·.1 < ·.1)
  for (c, (d, at_)) in rows do IO.println s!"| `{c}` | {d} | {at_} |"
  return if r.waitBad.isEmpty && r.readBad.isEmpty && r.gmmaBad.isEmpty && r.gmmaDead.isEmpty then 0
    else 1
