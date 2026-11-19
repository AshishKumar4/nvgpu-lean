import NV.SASS.Read
import NV.Core.Engine
import NV.Arch.Parts

/-!
Helpers for tests that compile PTX with the pinned `ptxas` and run it on the parameters of the
kernel's target (`NV.Arch.forSm`: A100, H200, RTX PRO 6000).
-/

open NV NV.SASS NV.Core

namespace Tests

def tool (cmd : String) (args : Array String) : IO String := do
  let out ← IO.Process.output { cmd, args }
  if out.exitCode != 0 then throw (IO.userError s!"{cmd} {args}: {out.stdout}{out.stderr}")
  return out.stdout

/-- The target the tests compile for: `NV_ARCH` (`sm_80`, `sm_120a`), `sm_90a` by default. -/
def arch : IO String := return (← IO.getEnv "NV_ARCH").getD "sm_90a"

/-- The message of a PTX file a target cannot compile: it uses an instruction the target lacks
(`Feature 'elect' requires .target sm_90 or higher`), or the file is written for a later target
than the tests' (`tests/diff/tcgen05.ptx`, `.target sm_100a`). Tests report it as skipped. -/
def unsupported (e : IO.Error) : Option String :=
  let m := toString e
  let has (s : String) : Bool := (m.splitOn s).length > 1
  if has "requires .target" || has "not supported on .target" then
    some ((m.splitOn "error   : ")[1]?.getD m |>.takeWhile (· != '\n') |>.toString)
  else if has "is higher than default SM version" || has "cannot be compiled to future architecture" then
    some "the file's .target is later than the tests'"
  else if has "is another architecture's" then some (m.takeWhile (· != '\n') |>.toString)
  else none

/-- A PTX file's text for the tests' target: its `.target sm_90a` (or `sm_90`) replaced by
`arch`. Both simulators read this text. -/
def ptxFor (ptx : String) : IO String := do
  let a ← arch
  let text := (← IO.FS.readFile ptx).replace ".target sm_90a" s!".target {a}"
  return text.replace ".target sm_90\n" s!".target {a}\n"

/-- The kernels of a PTX file, compiled for the tests' target (`ptxFor`) and read back from
`cuobjdump`. -/
def compile (ptx : String) : IO (Array Kernel) := do
  let dir ← IO.FS.createTempDir
  let cubin := (dir / "k.cubin").toString
  let a ← arch
  let src := (dir / "k.ptx").toString
  let text ← ptxFor ptx
  -- An architecture-specific target (`sm_120a`) compiles for that architecture alone; ptxas
  -- fails some such files for another with errors that do not name the target.
  let target := ((text.splitOn ".target ")[1]?).map fun r =>
    (r.takeWhile fun c => c.isAlphanum || c == '_').toString
  if let some t := target then
    if t.endsWith "a" && t != a then
      throw (IO.userError s!"the file's .target {t} is another architecture's")
  IO.FS.writeFile src text
  discard <| tool "ptxas" #[s!"-arch={a}", "-m64", src, "-o", cubin]
  IO.ofExcept (Read.kernels (← tool "cuobjdump" #["-sass", cubin]) (← tool "cuobjdump" #["-elf", cubin]))

def kernel (ks : Array Kernel) (name : String) : IO Kernel :=
  match ks.find? (·.name == name) with
  | some k => pure k
  | none => throw (IO.userError s!"no kernel {name}")

/-- Little-endian bytes of `v`. -/
def le (v : Nat) (bytes : Nat) : List UInt8 := (List.range bytes).map fun i => (v >>> (8 * i)).toUInt8

/-- Global memory starts here; buffers are laid out from it in order, 256-byte aligned. -/
def base : UInt64 := 0x7f0000000000

/-- Run `k` over `grid` CTAs of `block` threads with buffers initialized to `bufs` (32-bit words);
each buffer's address is passed as a 64-bit parameter, then `extra` parameters. `textures`: the
launch's texture and surface objects, each over a buffer (by index; its `base` an offset into it).
Returns each buffer's final words and the run's counters. -/
def runWith (k : Kernel) (grid block : Nat) (bufs : List (Array UInt32)) (extra : List (List UInt8) := [])
    (cluster : Nat × Nat × Nat := (1, 1, 1))
    (textures : List (Nat × NV.Tex.Texture) := []) :
    IO (Except String (List (Array UInt32) × NV.Core.Counters)) := do
  let sizes := bufs.map fun b => (4 * b.size + 255) / 256 * 256
  let offs := (sizes.foldl (fun (acc : List Nat × Nat) n => (acc.1 ++ [acc.2], acc.2 + n)) ([], 0)).1
  let mut g := Store.empty base (max (sizes.foldl (· + ·) 0) 256)
  for (b, off) in bufs.zip offs do
    for j in [0:b.size] do g := g.writeWord (off + 4 * j) b[j]!
  let args := offs.map (fun off => le (base.toNat + off) 8) ++ extra
  let mut params := ByteArray.mk (Array.replicate k.paramBytes 0)
  for ((off, sz), bytes) in k.params.toList.zip args do
    for j in [0:sz] do params := params.set! (off + j) bytes[j]!
  -- Stopped at 50M cycles: a kernel that spins (a semantics fault) fails rather than hangs.
  let launch : Launch := { kernel := k, grid := (grid, 1, 1), block := (block, 1, 1), params,
                           global := g, cluster, untilCycle := some 50000000,
                           textures := (textures.map fun (b, t) =>
                             { t with base := base + (offs.getD b 0).toUInt64 + t.base }).toArray }
  let some p := NV.Arch.forSm k.sm | return .error s!"no part modelled for sm_{k.sm}"
  return (NV.Core.run p launch).map fun res =>
    ((bufs.zip offs).map fun (b, off) => res.global.read (base + off.toUInt64) (4 * b.size), res.counters)

/-- `runWith`'s buffers alone. -/
def run (k : Kernel) (grid block : Nat) (bufs : List (Array UInt32)) (extra : List (List UInt8) := [])
    (cluster : Nat × Nat × Nat := (1, 1, 1))
    (textures : List (Nat × NV.Tex.Texture) := []) :
    IO (Except String (List (Array UInt32))) :=
  return (← runWith k grid block bufs extra cluster textures).map (·.1)

/-! ## Fixed latencies against the fixtures

Each part's fixed latencies against the stall cycles between a producer and its first consumer in
the fixtures' listings. Memory instructions are left out as consumers: they read their sources
after issue.
-/

def regsIn : Operand → List Reg
  | .reg r _ => [r]
  | .mem b _ u d _ => b.toList ++ u.toList ++ d.toList
  | .cbank _ b _ => b.toList
  | .gdesc r _ | .udesc _ r _ => [r]
  | _ => []

/-- Registers an instruction surely reads: its operands after the destinations (leading
predicates, then one register), and its guard. -/
def surelyReads (i : Instr) : List Reg :=
  let ops := i.operands.toList
  let isPredOp : Operand → Bool := fun | .reg ⟨.p, _⟩ _ | .reg ⟨.up, _⟩ _ => true | _ => false
  let compare := i.opcode.endsWith "SETP" || i.opcode.endsWith "SETP2" || i.opcode.endsWith "PLOP3"
  let dsts := match ops.head? with
    | some (.mem ..) | some (.gdesc ..) | none => 0
    | _ => (ops.takeWhile isPredOp).length + (if compare then 0 else 1)
  (ops.drop dsts).flatMap regsIn ++ (i.guard.map (·.1)).toList

def controlOps : List String :=
  ["BRA", "EXIT", "BSSY", "BSYNC", "BREAK", "CALL", "RET", "WARPSYNC", "BAR", "BPT"]

/-- The registers an instruction writes: its leading predicates and the register after them
(`ELECT P0, UR4`, `LOP3.LUT P0, R2`), or the predicates alone for comparisons (`ISETP P0, PT`). -/
def written (i : Instr) : List Reg :=
  let ops := i.operands.toList
  let isPredOp : Operand → Bool := fun | .reg ⟨.p, _⟩ _ | .reg ⟨.up, _⟩ _ => true | _ => false
  let compare := i.opcode.endsWith "SETP" || i.opcode.endsWith "SETP2" || i.opcode.endsWith "PLOP3"
  let n := (ops.takeWhile isPredOp).length + (if compare then 0 else 1)
  (ops.take n).filterMap fun | .reg r _ => if r.file != .b && !r.isSink then some r else none | _ => none

/-- Fixed-latency producers (no write counter) whose table latency exceeds the stall cycles before
the first consumer of one of their results in straight-line code. -/
def latencyProblems (p : NV.Core.Params) (k : Kernel) : List String :=
  let code := k.code.toList
  code.zipIdx.flatMap fun (i, n) =>
    if i.ctrl.wbar.isSome || controlOps.contains i.opcode ||
      i.operands.any (fun | .mem .. => true | _ => false) then [] else
    match p.timing i with
    | none => []
    | some t => (written i).filterMap fun d => do
      let (_, dist, _) := ((code.drop (n + 1)).take 16).foldl (fun (acc : Nat × Option Nat × Bool) j =>
        let (sum, _, stop) := acc
        if stop then acc
        else if j.operands.any (fun | .mem .. => true | _ => false) then (sum + j.ctrl.stall, none, false)
        else if (surelyReads j).contains d then (sum, some sum, true)
        else if controlOps.contains j.opcode || (written j).contains d then (sum, none, true)
        else (sum + j.ctrl.stall, none, false)) (i.ctrl.stall, none, false)
      let dist ← dist
      if dist < t.latency then
        some s!"{k.name} {i.pc} {i.full} → {repr d.file} {d.idx}: latency {t.latency} in the part's table, the code waits {dist}"
      else none

end Tests
