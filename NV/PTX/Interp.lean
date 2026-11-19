import NV.PTX.Sem
import NV.Memory.Atomic

/-!
# The PTX interpreter

The functional reference for PTX programs: every thread of a grid runs
`Thread.step`, and the machine answers the effects at thread granularity:

* memory requests are performed at once on flat stores (global, constant, kernel parameters,
  each CTA's shared memory, each thread's local memory), so a run is a sequentially consistent
  interleaving of the threads' accesses;
* `bar.sync`/`barrier.sync` and `barrier.cluster` release their waiting threads once the
  expected number have arrived (the CTA's or cluster's non-exited threads by default);
* warp collectives release once every non-exited lane of the member mask has arrived at the same
  collective, and `Collective.answer` computes each lane's answer; `activemask` answers the lanes
  of the warp at the same instruction;
* fences are performed (the memory is already sequentially consistent).

Which runnable thread steps next is the scheduler's choice (`Sched`); `run` takes one scheduler,
`explore` takes every choice at each step that touches memory or synchronizes (register-only
steps commute with other threads' steps, so they run eagerly) and collects all final states, for
small instances.
-/

namespace NV.PTX

open NV (Store Effect Req Resp Windows Collective SyncOp)
open NV.PTX.Sem

/-- A kernel launch. -/
structure Launch where
  entry : String
  grid : Nat × Nat × Nat
  block : Nat × Nat × Nat
  cluster : Nat × Nat × Nat := (1, 1, 1)
  /-- The kernel parameters, laid out as `Sem.paramLayout` says. -/
  params : ByteArray
  /-- Global memory: the caller's buffers; module `.global` variables are placed from
  `Prog.globalBase` and must lie in it. -/
  global : Store
  dynShared : Nat := 0
  windows : Windows := {}
  /-- Texture and surface objects; handle `i + 1` names entry `i` (`NV.Tex`). -/
  textures : Array NV.Tex.Texture := #[]

/-- Where a thread is held. -/
inductive Wait where
  | run
  | bar (cta id : Nat) (count : Option Nat) (arrived : Nat)
  | cluster (id : Nat) (arrived : Nat)
  | collect (op : Collective) (mask : UInt32) (contrib : Array UInt32) (wb : Resp → Thread → Thread)
  | exited

instance : Inhabited Wait := ⟨.run⟩

structure Machine where
  prog : Prog
  windows : Windows
  textures : Array NV.Tex.Texture := #[]
  threads : Array Thread
  waits : Array Wait
  global : Store
  const : Store
  param : Store
  shared : Array Store
  «local» : Array Store
  /-- Per CTA, its tensor memory (`sm_100a`): each CTA's own here, where an SM's CTAs share one
  (`NV.Core.Memory.SmMem.tmem`) whose allocator keeps them apart. -/
  tmem : Array Store := #[]
  /-- Threads per CTA and CTAs per cluster, to find a thread's CTA, warp and cluster. -/
  ctaThreads : Nat
  clusterCtas : Nat
  /-- `bar.arrive` arrivals at (CTA, barrier) not yet completed: they count without waiting. -/
  arrivals : List ((Nat × Nat) × Nat) := []
  steps : Nat := 0
  /-- Asynchronous operations performed: tensor-core (`wgmma`, `tcgen05.mma`) and copies (bulk,
  tensor and `cp.async`), the evidence that a run did its work and not an early exit. -/
  mmaOps : Nat := 0
  copyOps : Nat := 0
  /-- Hold back the asynchronous operations whose completion a thread observes only through
  memory (`deferrable`) in `pending`, for the scheduler to complete when it chooses. -/
  defer : Bool := false
  /-- `Env.skipMma`. -/
  skipMma : Bool := false
  /-- Held-back operations, oldest first, each with its issuing thread and PTX line. -/
  pending : Array (Nat × NV.Async.Op × Nat) := #[]
  deriving Inhabited

namespace Machine

def cta (m : Machine) (k : Nat) : Nat := k / m.ctaThreads
def warp (m : Machine) (k : Nat) : Nat :=
  m.cta k * ((m.ctaThreads + 31) / 32) + (k % m.ctaThreads) / 32
def lane (m : Machine) (k : Nat) : Nat := (k % m.ctaThreads) % 32
def clusterOf (m : Machine) (k : Nat) : Nat := m.cta k / m.clusterCtas

def live (m : Machine) (k : Nat) : Bool := !(m.waits[k]! matches .exited)

/-- Bytes of a little-endian value. -/
def bytesLE (n : Nat) (v : Nat) : Array UInt32 :=
  let ws := (n + 3) / 4
  (Array.range ws).map fun w => (v >>> (32 * w)).toUInt32

/-- The values of an initializer of element type `t` (§5.4.4), flattened: integer and float
constants and variable addresses. -/
def initValues (prog : Prog) (w : Windows) (t : Ty) : Init → Except String (Array UInt64)
  | .list xs => do
    let mut out := #[]
    for x in xs.attach do
      out := out ++ (← initValues prog w t x.1)
    return out
  | .expr e => do
    let addrOf (n : String) : Option UInt64 :=
      (prog.vars.find? (·.decl.name == n)).map (·.addr)
    match e, Sem.Float.lanes? t with
    | e, some (f, 1) =>
      match Sem.Float.const f e with
      | some v => return #[v.toUInt64]
      | none => throw s!"initializer {repr e} is not a .{Keyword.name t} constant"
    | .name n, _ =>
      match addrOf n with
      | some a => return #[a]
      | none => throw s!"initializer names {n}, not a module variable"
    | .app "generic" (.name n), _ =>
      match prog.vars.find? (·.decl.name == n) with
      | some p => return #[w.toGeneric p.space p.addr]
      | none => throw s!"generic({n}): not a module variable"
    | e, _ =>
      match Sem.evalConst e with
      | some c => return #[c.v]
      | none => throw s!"initializer {repr e} needs the numerics library or is not supported"
termination_by i => sizeOf i
decreasing_by
  all_goals simp_wf
  have := List.sizeOf_lt_of_mem x.2
  omega

/-- The machine for a launch: every thread at the kernel's first instruction. -/
def init (prog : Prog) (l : Launch) : Except String Machine := do
  let some entry := prog.byName.get? l.entry | throw s!"no kernel {l.entry}"
  let (gx, gy, gz) := l.grid
  let (bx, by_, bz) := l.block
  let (cx, cy, cz) := l.cluster
  let ctaThreads := bx * by_ * bz
  let ctas := gx * gy * gz
  let mut threads := #[]
  for c in [0:ctas] do
    let ctaid := (c % gx, (c / gx) % gy, c / (gx * gy))
    for k in [0:ctaThreads] do
      let tid := (k % bx, (k / bx) % by_, k / (bx * by_))
      let ctx : Ctx :=
        { tid, ntid := l.block, ctaid, nctaid := l.grid,
          clusterCtaid := (ctaid.1 % cx, ctaid.2.1 % cy, ctaid.2.2 % cz),
          clusterNctaid := l.cluster,
          clusterid := (ctaid.1 / cx, ctaid.2.1 / cy, ctaid.2.2 / cz),
          nclusterid := (gx / cx, gy / cy, gz / cz), dynShared := l.dynShared }
      threads := threads.push (Thread.start prog ctx entry)
  let mut global := l.global
  let mut const := Store.empty 0 (max prog.constBytes 1)
  for p in prog.vars do
    let some ini := p.decl.init | continue
    let vals ← initValues prog l.windows p.decl.ty ini
    let size := p.decl.ty.bytes
    for (v, k) in vals.zipIdx do
      let addr := p.addr + (k * size).toUInt64
      match p.space with
      | .global =>
        unless global.contains addr size do throw s!"{p.decl.name} lies outside global memory"
        global := global.write addr size (bytesLE size v.toNat) 0
      | .const _ => const := const.write addr size (bytesLE size v.toNat) 0
      | _ => throw s!"{p.decl.name}: only .global and .const variables take initializers"
  let param := (Store.empty 0 (max l.params.size 1)).write 0 l.params.size
    ((Array.range ((l.params.size + 3) / 4)).map fun w =>
      (List.range 4).foldl (fun v b =>
        v ||| ((l.params.get! (4 * w + b) |>.toUInt32) <<< (8 * b).toUInt32)) 0) 0
  return {
      prog, windows := l.windows, textures := l.textures, threads, waits := Array.replicate threads.size .run,
      global, const, param, ctaThreads, clusterCtas := cx * cy * cz,
      shared := Array.replicate ctas (Store.empty 0 (prog.sharedBytes + l.dynShared)),
      tmem := Array.replicate ctas (Store.empty 0 Space.tmemBytes),
      «local» := Array.replicate threads.size (Store.empty 0 prog.localTop) }

/-- Perform thread `k`'s one-lane request on the store its space names. -/
def perform (m : Machine) (k : Nat) (r : Req) : Except String (Machine × Resp) := do
  -- Each store is taken out of the machine before it is written, so that, used linearly, it is
  -- updated in place rather than copied a page at a time.
  match r.space with
  | .global =>
    let s := m.global
    let m := { m with global := default }
    let (s, out) ← s.perform r 0
    return ({ m with global := s }, out)
  | .const _ =>
    let (s, out) ← m.const.perform r 0
    return ({ m with const := s }, out)
  | .param =>
    let (s, out) ← m.param.perform r 0
    return ({ m with param := s }, out)
  | .local =>
    let s := m.local[k]!
    let m := { m with «local» := m.local.set! k default }
    let (s, out) ← s.perform r 0
    return ({ m with «local» := m.local.set! k s }, out)
  | .shared =>
    let c := m.cta k
    let s := m.shared[c]!
    let m := { m with shared := m.shared.set! c default }
    let (s, out) ← s.perform r 0
    return ({ m with shared := m.shared.set! c s }, out)
  | .tmem =>
    -- Past `2 ^ 24`, the tensor memory of the CTA of cluster rank `a / 2 ^ 24 - 1`
    -- (`Tcgen05.peerByte`: a CTA pair's peer).
    let addr := ((r.addrs[0]!).getD 0).toNat
    let c := if addr < 2 ^ 24 then m.cta k else m.clusterOf k * m.clusterCtas + (addr / 2 ^ 24 - 1)
    unless c < m.tmem.size do throw s!"tensor memory address {addr} names no CTA of the cluster"
    let s := m.tmem[c]!
    let m := { m with tmem := m.tmem.set! c default }
    let (s', out) ← s.perform { r with addrs := #[some (addr % 2 ^ 24).toUInt64] } 0
    return ({ m with tmem := m.tmem.set! c s' }, out)
  | .sharedCluster =>
    let addr := (r.addrs[0]!).getD 0
    let (rank, off) := m.windows.sharedTarget addr
    let c := match rank with
      | none => m.cta k
      | some r => m.clusterOf k * m.clusterCtas + r
    unless c < m.shared.size && (rank.isNone || rank.get! < m.clusterCtas) do
      throw s!"shared::cluster address {addr} names no CTA of the cluster"
    let s := m.shared[c]!
    let m := { m with shared := m.shared.set! c default }
    let (s', out) ← s.perform { r with addrs := #[some off] } 0
    return ({ m with shared := m.shared.set! c s' }, out)

/-- Perform every lane of a request (an asynchronous agent's requests have up to 32), one lane at
a time in lane order; the response is lane-major, `words bytes` per lane. -/
def performLanes (m : Machine) (k : Nat) (r : Req) : Except String (Machine × Resp) := do
  let n := words r.bytes
  let per := match r.op with
    | .atom .cas _ | .red .cas _ => 2 * n
    | .st | .atom .. | .red .. => n
    | _ => 0
  let mut m := m
  let mut out : Resp := Array.replicate (r.addrs.size * n) 0
  for l in [0:r.addrs.size] do
    let some a := r.addrs[l]! | continue
    let (m', resp) ← m.perform k { r with addrs := #[some a], data := r.data.extract (l * per) (l * per + per) }
    m := m'
    for j in [0:resp.size] do out := out.set! (l * n + j) resp[j]!
  return (m, out)

/-- Run an asynchronous operation to completion at once (the blocking reading of
`Effect.async`), its requests performed for thread `k`. -/
def runAsync (m : Machine) (k : Nat) (op : NV.Async.Op) : Except String (Machine × Resp) :=
  -- The machine passed on, not read with `get`, so that it stays unshared.
  let perform (r : Req) : StateT Machine (Except String) Resp := fun m => do
    let (m', resp) ← m.performLanes k r
    return (resp, m')
  let m := match op with
    | .mma _ | .mmaT _ => { m with mmaOps := m.mmaOps + 1 }
    | .copy _ | .tensor _ => { m with copyOps := m.copyOps + 1 }
    | _ => m
  (op.run perform).run m |>.map fun (resp, m) => (m, resp)

/-- Release every barrier and collective whose participants have all arrived. One pass groups
the waiting threads; each group is then checked against its expected count. -/
def release (m : Machine) : Except String Machine := do
  -- Nothing waits: nothing to release (most steps).
  unless m.waits.any (fun w => w matches .bar .. | .cluster .. | .collect ..) do return m
  let n := m.threads.size
  let mut liveCta : Array Nat := Array.replicate (m.shared.size) 0
  let mut liveCluster : Std.HashMap Nat Nat := {}
  let mut bars : Std.HashMap (Nat × Nat) (Array Nat × Option Nat) := {}
  let mut clusters : Std.HashMap Nat (Array Nat) := {}
  let mut colls : Std.HashMap (Nat × Bool) (Array Nat) := {}
  for k in [0:n] do
    if m.live k then
      liveCta := liveCta.modify (m.cta k) (· + 1)
      liveCluster := liveCluster.insert (m.clusterOf k) ((liveCluster.getD (m.clusterOf k) 0) + 1)
    match m.waits[k]! with
    | .bar c id cnt _ =>
      let (ws, _) := bars.getD (c, id) (#[], cnt)
      bars := bars.insert (c, id) (ws.push k, cnt)
    | .cluster cl _ => clusters := clusters.insert cl ((clusters.getD cl #[]).push k)
    | .collect op _ _ _ =>
      -- Keyed by CTA for `bar.red`, by warp for the warp collectives.
      let key := match op with | .barRed .. => (m.cta k, true) | _ => (m.warp k, false)
      colls := colls.insert key ((colls.getD key #[]).push k)
    | _ => pure ()
  let mut m := m
  for ((c, id), (ws, cnt)) in bars.toList do
    let arrived := ws.size + (m.arrivals.lookup (c, id)).getD 0
    if arrived ≥ cnt.getD liveCta[c]! then
      for j in ws do m := { m with waits := m.waits.set! j .run }
      m := { m with arrivals := m.arrivals.filter (·.1 != (c, id)) }
  for (cl, ws) in clusters.toList do
    if ws.size ≥ liveCluster.getD cl 0 then
      for j in ws do m := { m with waits := m.waits.set! j .run }
  for ((cta, isBar), ws) in colls.toList do
    let some first := ws[0]? | continue
    let .collect op mask _ _ := m.waits[first]! | continue
    let here := ws.toList.filter fun j => match m.waits[j]! with
      | .collect op' mask' _ _ => op' == op && mask' == mask
      | _ => false
    let expected : Nat := match op with
      | .barRed _ (some k) _ => k
      | .barRed _ none _ => liveCta[cta]!
      | _ =>
        -- The warp's first thread; the group key is the warp here, not a CTA.
        let w0 := first - m.lane first
        ((List.range 32).filter fun l => w0 + l < n && m.warp (w0 + l) == m.warp first &&
          m.live (w0 + l) && Collective.bit mask l).length
    if here.length < expected then continue
    let idx (j : Nat) := if isBar then here.idxOf j else m.lane j
    let width := if isBar then here.length else 32
    let contrib : Array (Option (Array UInt32)) := (Array.range width).map fun l =>
      (here.find? (idx · == l)).bind fun j => match m.waits[j]! with
        | .collect _ _ c _ => some c | _ => none
    let answers ← Collective.answer op mask contrib
    for j in here do
      match m.waits[j]! with
      | .collect _ _ _ wb =>
        m := { m with threads := m.threads.set! j (wb answers[idx j]! m.threads[j]!),
                      waits := m.waits.set! j .run }
      | _ => pure ()
  return m

/-- Where a thread is: its call depth, function and instruction. -/
def position (m : Machine) (k : Nat) : Nat × Nat × Nat :=
  let t := m.threads[k]!
  (t.frames.size, t.frame.fn, t.frame.pc)

/-- The live threads of `k`'s warp at `k`'s instruction, `k` included, in lane order. -/
def together (m : Machine) (k : Nat) : List Nat :=
  let first := k - m.lane k
  ((List.range 32).map (first + ·)).filter fun j =>
    j < m.threads.size && m.warp j == m.warp k && m.waits[j]! matches .run &&
      m.position j == m.position k

/-- Threads that can step. -/
def runnable (m : Machine) : List Nat :=
  (List.range m.threads.size).filter (m.waits[·]! matches .run)

def env (m : Machine) : Env :=
  { prog := m.prog, windows := m.windows, clock := m.steps,
    textures := m.textures, skipMma := m.skipMma }

/-- An asynchronous operation's kind, for an error it raises. -/
def asyncName : NV.Async.Op → String
  | .copy _ => "bulk copy" | .tensor _ => "tensor copy" | .mma _ => "wgmma"
  | .mmaT _ => "tcgen05 operation" | .prefetch _ => "prefetch" | .fault _ => "fault"

/-- Whether a thread observes `o`'s completion only through memory, so that completing it later
than its issue is a schedule the hardware may take: a `tcgen05` operation or commit (their
mbarrier arrivals), and a copy that signals mbarriers and belongs to no group a thread waits on
(TMA and bulk loads). Register results (`wgmma`) and group-waited copies (`cp.async`, bulk
stores) complete at issue. -/
def deferrable : NV.Async.Op → Bool
  | .mmaT _ => true
  | .copy c => c.group == some NV.Async.Group.tcgen05 || (c.group.isNone && !c.signals.isEmpty)
  | .tensor t => t.group.isNone
  | _ => false

/-- Step thread `k` once; `true` if the step touched memory or synchronized. `group` is the mask
of the lanes executing this instruction together, which `activemask` answers (0: the lanes of
the warp at the same instruction now). -/
def stepThread (m : Machine) (k : Nat) (group : UInt32 := 0) :
    Except String (Machine × Bool) := do
  -- `activemask` without a group: the lanes at `k`'s instruction, before `k` steps.
  let together := if group != 0 then group else
    (m.together k).foldl (fun a j => a ||| ((1 : UInt32) <<< (m.lane j).toUInt32)) 0
  -- The thread taken out of the machine, so that its registers are written in place.
  let t := m.threads[k]!
  -- What an exited thread keeps (its `printf` records, `printfText`).
  let (ctx, printf) := (t.ctx, t.printf)
  let m := { m with steps := m.steps + 1, threads := m.threads.set! k default }
  match t.step m.env with
  | .next s => return ({ m with threads := m.threads.set! k s }, false)
  | .reg s wb => return ({ m with threads := m.threads.set! k (wb s) }, false)
  | .mem r s wb =>
    -- A thread's request names one address, or several (a texture fetch's footprint). A fault
    -- names the PTX line of the access.
    let req := r s
    let line := ((m.prog.funcs[t.frame.fn]?).bind (·.code[t.frame.pc]?)).map (·.line) |>.getD 0
    let (m, resp) ← (if req.addrs.size > 1 then m.performLanes k req else m.perform k req).mapError
      fun e => s!"line {line}: {e}"
    return ({ m with threads := m.threads.set! k (wb resp s) }, true)
  | .sync (.bar id cnt arrive) s =>
    let m := { m with threads := m.threads.set! k s }
    if arrive then
      let key := (m.cta k, id)
      let n := (m.arrivals.lookup key).getD 0
      return ({ m with arrivals := (key, n + 1) :: m.arrivals.filter (·.1 != key) }, true)
    return ({ m with waits := m.waits.set! k (.bar (m.cta k) id cnt 1) }, true)
  | .sync .clusterArrive s | .sync .clusterWait s =>
    let m := { m with threads := m.threads.set! k s }
    return ({ m with waits := m.waits.set! k (.cluster (m.clusterOf k) 1) }, true)
  | .async op s wb =>
    let o := op s
    let line := ((m.prog.funcs[t.frame.fn]?).bind (·.code[t.frame.pc]?)).map (·.line) |>.getD 0
    if m.defer && deferrable o then
      return ({ m with threads := m.threads.set! k (wb #[] s), pending := m.pending.push (k, o, line) }, true)
    -- Otherwise it completes at issue in a blocking run, so group commits and waits pass.
    let (m, resp) ← (m.runAsync k o).mapError fun e => s!"line {line}: {asyncName o}: {e}"
    return ({ m with threads := m.threads.set! k (wb resp s) }, true)
  | .sync _ s => return ({ m with threads := m.threads.set! k s }, true)
  | .collect .activemask _ _ s wb => return ({ m with threads := m.threads.set! k (wb #[together] s) }, true)
  | .collect op mask contrib s wb =>
    let m := { m with threads := m.threads.set! k s }
    return ({ m with waits := m.waits.set! k (.collect op mask contrib wb) }, true)
  | .exit =>
    return ({ m with threads := m.threads.set! k { ctx, frames := #[], printf }, waits := m.waits.set! k .exited }, true)
  | .fault msg => throw s!"thread {k}: {msg}"

/-- A scheduler picks the next thread among the runnable ones (never empty). -/
abbrev Sched := Machine → List Nat → Nat

/-- Lowest-numbered runnable thread. -/
def lowest : Sched := fun _ rs => rs.head!

/-- Run to completion under `sched`, at most `fuel` steps. -/
def run (sched : Sched) (fuel : Nat) (m : Machine) : Except String Machine := do
  let mut m := m
  for _ in [0:fuel] do
    m ← m.release
    match m.runnable with
    | [] =>
      if (List.range m.threads.size).all (m.waits[·]! matches .exited) then return m
      throw "deadlock: every remaining thread waits at a barrier or collective"
    | rs => m := (← m.stepThread (sched m rs)).1
  throw s!"no completion within {fuel} steps"

/-- Run to completion as SIMT warps, round-robin: the first runnable thread at or after the
last group stepped (wrapping around) has its warp execute its instruction for every runnable lane
at that instruction (`together`), in lane order, and `activemask` answers that group. Every warp
keeps stepping, so one spinning on another (an mbarrier wait on a peer CTA's stores) cannot
starve it. At most `fuel` steps. -/
def runWarps (fuel : Nat) (m : Machine) : Except String Machine := do
  let mut m := m
  let mut cur := 0
  for _ in [0:fuel] do
    m ← m.release
    let rs := m.runnable
    match rs.find? (· ≥ cur) <|> rs.head? with
    | none =>
      if (List.range m.threads.size).all (m.waits[·]! matches .exited) then return m
      throw "deadlock: every remaining thread waits at a barrier or collective"
    | some k =>
      let g := m.together k
      let msk := g.foldl (fun a j => a ||| ((1 : UInt32) <<< (m.lane j).toUInt32)) 0
      for j in g do m := (← m.stepThread j msk).1
      cur := (g.foldl max k) + 1
  throw s!"no completion within {fuel} steps"

/-- Every final machine over all interleavings of memory and synchronization steps (register
steps run eagerly), with at most `fuel` steps per path; an error on any path is returned. -/
def explore (fuel : Nat) (m : Machine) : Except String (List Machine) :=
  match fuel with
  | 0 => throw "exploration exceeded its step bound"
  | fuel + 1 => do
    let m ← m.release
    match m.runnable with
    | [] =>
      if (List.range m.threads.size).all (m.waits[·]! matches .exited) then return [m]
      throw "deadlock: every remaining thread waits at a barrier or collective"
    | rs =>
      let mut out := []
      for k in rs do
        -- Run thread k's register steps, then its one visible step.
        let mut m' := m
        let mut visible := false
        for _ in [0:fuel] do
          let (m'', vis) ← m'.stepThread k
          m' := m''
          if vis || !(m'.waits[k]! matches .run) then visible := true; break
        unless visible do throw "thread ran out of steps"
        out := out ++ (← explore fuel m')
      return out

/-- Whether held-back operation `i` may complete now: a CTA's `tcgen05` operations complete in
the order they were issued, across its threads. That is the order a commit needs (after the
operations its thread issued before it, §9.7.18.12.1) and the one a `tcgen05.fence` carries across
a thread synchronization (§9.7.18.6.3); §9.7.18.6.2 lets some complete out of issue order, which
these schedules do not explore. -/
def completable (m : Machine) (i : Nat) : Bool :=
  let (k, o, _) := m.pending[i]!
  let tc (o : NV.Async.Op) : Bool := o.group == some NV.Async.Group.tcgen05
  !tc o || !((m.pending.extract 0 i).any fun (k', o', _) => m.cta k' == m.cta k && tc o')

/-- Complete held-back operation `i`. -/
def complete (m : Machine) (i : Nat) : Except String Machine := do
  let (k, o, line) := m.pending[i]!
  let m := { m with pending := m.pending.eraseIdxIfInBounds i }
  return (← (m.runAsync k o).mapError fun e => s!"line {line}: {asyncName o}: {e}").1

/-- Where each live warp is, for a report: its first live thread's instruction (source line and
text) and wait, and the warps' held-back operations. -/
def stuck (m : Machine) : List String := Id.run do
  let mut out := []
  let mut seen : Std.HashSet Nat := {}
  for k in [0:m.threads.size] do
    if !m.live k || seen.contains (m.warp k) then continue
    seen := seen.insert (m.warp k)
    let (_, fn, pc) := m.position k
    let some f := m.prog.funcs[fn]? | continue
    let at_ := match f.code[pc]? with
      | some pi => s!"line {pi.line} {pi.i.op.spelling}"
      | none => s!"pc {pc}"
    -- A warp spinning on an mbarrier (at its wait or the branch after it): the barrier's state.
    let poll := [pc, pc - 1].findSome? fun i => (f.code[i]?).filter fun pi =>
      pi.i.op == .mbarrierTryWait || pi.i.op == .mbarrierTestWait
    let bar := match poll with
      | some pi =>
        match m.threads[k]!.target m.env pi.i.space? (pi.args[1]?.getD .sink) with
        | .ok (.mem .shared a) =>
          let ws := m.shared[m.cta k]!.read a 8
          let b := NV.Async.MBarrierWord.decode ((ws[0]?.getD 0).toUInt64 ||| ((ws[1]?.getD 0).toUInt64 <<< 32))
          let parity := pi.i.has .parity
          let want := match pi.args[2]? with
            | some x => match m.threads[k]!.read m.env x (if parity then .b32 else .b64) with
              | .ok v => if parity then s!", polled for parity {v}" else s!", polled with state {v}"
              | .error _ => ""
            | none => ""
          s!"; line {pi.line}'s mbarrier at shared 0x{String.ofList (Nat.toDigits 16 a.toNat)}: phase parity {b.phase % 2}, {b.pending} of {b.expected} arrivals pending, tx {b.tx}{want}"
        | _ => ""
      | none => ""
    let w := match m.waits[k]! with
      | .run => "running"
      | .bar c id _ n => s!"at bar {id} of CTA {c} ({n} arrived)"
      | .cluster id n => s!"at the cluster barrier of cluster {id} ({n} arrived)"
      | .collect .. => "in a warp collective"
      | .exited => "exited"
    out := out ++ [s!"CTA {m.cta k} warp {(k % m.ctaThreads) / 32}: {at_}, {w}{bar}"]
  return out ++ (m.pending.toList.map fun (k, o, _) =>
    s!"held back: thread {k}'s {match o with | .mmaT _ => "tcgen05 operation" | .copy c => if c.group == some NV.Async.Group.tcgen05 then "tcgen05.commit" else "copy" | .tensor _ => "tensor copy" | _ => "operation"}")

/-- A 64-bit xorshift step. -/
def xorshift (x : UInt64) : UInt64 :=
  let x := x ^^^ (x <<< 13)
  let x := x ^^^ (x >>> 7)
  x ^^^ (x <<< 17)

/-- Whether a step at `k`'s instruction can change nothing another thread sees: a poll of an
mbarrier, or the compare-and-branch of its loop. -/
def polls (m : Machine) (k : Nat) : Bool :=
  let (_, fn, pc) := m.position k
  match m.prog.funcs[fn]? >>= (·.code[pc]?) with
  | some pi => [Op.mbarrierTryWait, .mbarrierTestWait, .bra, .setp, .selp].contains pi.i.op
  | none => false

/-- The steps of nothing but polls, with nothing held back, after which a run is stuck: no
memory can change, so no poll can succeed. -/
def stallSteps : Nat := 20000

/-- Under `pct`, the polls in a row after which a warp drops below every other. -/
def pctSpin : Nat := 64

/-- Run as SIMT warps under a seeded random schedule: at each step, with probability `1 / delay`
when one may, a random held-back operation completes (`defer`); otherwise a random runnable warp
executes, at a random one of its runnable lanes, the instruction there for the lanes with it
(`together`), so that a warp's diverged paths interleave too. With `pct` the warp is not drawn
but the highest of random priorities, a polling warp dropping below the rest (`pctSpin`): the
other warps run far ahead of a waiting one, as a warp suspended in `mbarrier.try_wait` would
let them. The result, or on a deadlock (every
thread waiting, or `stallSteps` steps of nothing but polls with nothing held back), or with no
completion within `fuel` steps, the error and where every warp is (`stuck`). -/
def runRandom (seed delay fuel : Nat) (m : Machine) (pct : Bool := false) :
    Except (String × List String) Machine := do
  let mut m := { m with defer := true }
  let mut x : UInt64 := seed.toUInt64 * 0x9E3779B97F4A7C15 + 1
  let mut quiet := 0
  -- `pct`: each warp a random priority; the highest runnable runs, and one that has polled
  -- `pctSpin` times running drops below every other, so that the others run on while it waits.
  let nw := m.warp (m.threads.size - 1) + 1
  let mut prio : Array Nat := Array.replicate nw 0
  for w in [0:nw] do
    x := xorshift x
    prio := prio.set! w (((x >>> 16) % 1000000).toNat + 1000000)
  let mut spin : Array Nat := Array.replicate nw 0
  let mut floor := 1000000
  for _ in [0:fuel] do
    if quiet ≥ stallSteps && m.pending.isEmpty then
      throw (s!"deadlock: {stallSteps} steps of nothing but polls, nothing in flight", m.stuck)
    -- Errors other than a stall carry no report: capturing `m` for one would share it, and every
    -- write after copy it.
    m ← (m.release).mapError fun e => (e, [])
    x := xorshift x
    let ready := (List.range m.pending.size).filter m.completable
    let rs := m.runnable
    if !ready.isEmpty && (rs.isEmpty || (x >>> 32) % delay.toUInt64 == 0) then
      x := xorshift x
      m ← (m.complete ready[(x % ready.length.toUInt64).toNat]!).mapError fun e => (e, [])
      quiet := 0
      continue
    -- Warps, each by its first runnable thread.
    let leads := rs.filter fun k => !(rs.any fun j => j < k && m.warp j == m.warp k)
    match leads with
    | [] =>
      if (List.range m.threads.size).all (m.waits[·]! matches .exited) then return m
      throw ("deadlock: every remaining thread waits at a barrier or collective", m.stuck)
    | _ =>
      x := xorshift x
      let w := if pct then
          m.warp (leads.foldl (fun b k => if prio[m.warp k]! > prio[m.warp b]! then k else b) leads[0]!)
        else m.warp leads[(x % leads.length.toUInt64).toNat]!
      let lanes := rs.filter (m.warp · == w)
      x := xorshift x
      let k := lanes[(x % lanes.length.toUInt64).toNat]!
      let g := m.together k
      quiet := if m.polls k then quiet + 1 else 0
      if pct then
        if m.polls k then
          spin := spin.set! w (spin[w]! + 1)
          if spin[w]! ≥ pctSpin then
            floor := floor - 1
            prio := prio.set! w floor
            spin := spin.set! w 0
        else spin := spin.set! w 0
      let msk := g.foldl (fun a j => a ||| ((1 : UInt32) <<< (m.lane j).toUInt32)) 0
      for j in g do m := (← (m.stepThread j msk).mapError fun e => (e, [])).1
  throw (s!"no completion within {fuel} steps", m.stuck)

end Machine

/-- Run a kernel to completion as SIMT warps (`Machine.runWarps`); the final machine. -/
def interpRun (prog : Prog) (l : Launch) (fuel : Nat := 100_000_000) : Except String Machine := do
  (← Machine.init prog l).runWarps fuel

/-- Run a kernel to completion as SIMT warps (`Machine.runWarps`); the final global memory. -/
def interp (prog : Prog) (l : Launch) (fuel : Nat := 100_000_000) : Except String Store := do
  return (← interpRun prog l fuel).global

/-- The `printf` output of a finished run, as the host prints it: thread by thread, each thread's
records in order (the order between threads is the hardware's, not modelled). `%s` strings are
read from global memory. -/
def Machine.printfText (m : Machine) : List String :=
  let str (a : UInt64) : Option String :=
    if m.global.contains a 1 then
      some (String.ofList ((List.range 4096).map (fun j => (m.global.read (a + j.toUInt64) 1)[0]!)
        |>.takeWhile (· != 0) |>.map (Char.ofNat ·.toNat)))
    else none
  m.threads.toList.flatMap fun t => t.printf.toList.map fun (f, a) => NV.Printf.render f a str

/-- A kernel's parameter block from each parameter's bytes, at the offsets of `Sem.paramLayout`
(a parameter's bytes past its size dropped). -/
def paramBytes (m : Module Instr) (entry : String) (vals : List (List UInt8)) :
    Except String ByteArray := do
  let kernel? : Item Instr → Option (Function Instr)
    | .func f => if f.name == entry && f.entry then some f else none
    | _ => none
  let some f := m.items.findSome? kernel? | throw s!"no kernel {entry}"
  let (layout, size) := paramLayout f.params
  unless layout.length == vals.length do
    throw s!"{entry} takes {layout.length} parameters, given {vals.length}"
  let mut b := ByteArray.mk (Array.replicate size 0)
  for ((d, off), v) in layout.zip vals do
    for (x, j) in (v.take d.bytes).zipIdx do b := b.set! (off + j) x
  return b

/-- `paramBytes` of parameters given as integers, little-endian, up to 8 bytes each. -/
def paramBlock (m : Module Instr) (entry : String) (vals : List UInt64) :
    Except String ByteArray :=
  paramBytes m entry (vals.map fun v => (List.range 8).map fun j => (v >>> (8 * j).toUInt64).toUInt8)

/-- One register instruction, prepared: the program and the slots of its sources (`%a0`, …) and
destinations (`%d0`, …). -/
structure OneInstr where
  prog : Prog
  ins : List Ty
  outs : List Ty

/-- Prepare `mnemonic %d0[|%d1], %a0, %a1, …` with sources of types `ins` and destinations of
types `outs` (two destinations are written `p|q`); `negLast` negates the last source (`!p`). -/
def OneInstr.make (mnemonic : String) (ins outs : List Ty) (negLast : Bool := false) :
    Except String OneInstr := do
  let decls := (ins.zipIdx.map fun (t, k) => s!"  .reg .{Keyword.name t} %a{k};") ++
    (outs.zipIdx.map fun (t, k) => s!"  .reg .{Keyword.name t} %d{k};")
  let dst := "|".intercalate ((List.range outs.length).map (s!"%d{·}"))
  let srcs := (List.range ins.length).map fun k =>
    if negLast && k + 1 == ins.length then s!"!%a{k}" else s!"%a{k}"
  let src := ".version 9.0\n.target sm_90a\n.address_size 64\n.visible .entry k()\n{\n" ++
    "\n".intercalate decls ++ s!"\n  {mnemonic} {", ".intercalate (dst :: srcs)};\n  ret;\n}\n"
  return { prog := ← Prog.ofModule (← read src), ins, outs }

/-- Run it on source bits; the destination bits. Sources take slots `0 … n-1` and destinations
the next ones, in declaration order. -/
def OneInstr.run (o : OneInstr) (args : List UInt64) : Except String (List UInt64) := do
  let e : Env := { prog := o.prog }
  let ctx : Ctx := { tid := (0, 0, 0), ntid := (1, 1, 1), ctaid := (0, 0, 0), nctaid := (1, 1, 1) }
  let mut t := Thread.start o.prog ctx 0
  for ((v, ty), k) in (args.zip o.ins).zipIdx do
    t := t.setSlot k (if ty == .pred then (if v != 0 then 1 else 0) else v)
  match t.step e with
  | .next t' => return (List.range o.outs.length).map fun k => t'.slot (o.ins.length + k)
  | .fault m => throw m
  | _ => throw "not a register instruction"

end NV.PTX
