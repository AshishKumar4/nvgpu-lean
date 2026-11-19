import NV.Async.Tma
import NV.Async.Wgmma
import NV.Async.WgmmaNum
import NV.Async.MBarrierWord
import NV.SASS.Async
import NV.Core.Async
import NV.Arch.B200

/-!
`lake exe async-tests`: the async vocabulary against the ISA's own statements of it.

* Swizzle patterns against the tables of PTX ISA 9.5 §5.5.7, transcribed cell by cell.
* wgmma matrix descriptors against the "exact layouts" of the worked examples of
  §9.7.17.5.1.2.1.3 (CuTe layouts, evaluated here independently of `Desc.addr`).
* Fragment positions against CUTLASS's `CLayout_64xN` and `ALayout_64x16` (CuTe layouts).
* The mbarrier word through phases, tx-counts and the init value.
* The tensor-memory lanes a warp's `tcgen05.ld` reaches.
* A tiled TMA box that crosses the tensor's edge, against the address arithmetic of §5.5.3.
* The fast warpgroup MMA (`Share.finishFast`, through `TensorModel.tileWords`) against the
  operand-by-operand one (`Share.operands`, `Share.finish`) on random shares: every float input
  and accumulator, A in shared memory or registers, negations, `scale-d`, whole warps and single
  threads.
-/

open NV NV.Async NV.Async.Wgmma

/-- A CuTe layout `(shape):(stride)`, flattened: the offset of coordinate `i` (colexicographic). -/
def cute (shape stride : List Nat) (i : Nat) : Nat :=
  (shape.zip stride).foldl (fun (acc, rest) (s, d) => (acc + rest % s * d, rest / s)) (0, i) |>.1

/-- CuTe's `Swizzle<B, 4, 3>` on a byte address. -/
def cuteSwizzle (b a : Nat) : Nat := a ^^^ (((a >>> 7) % 2 ^ b) <<< 4)

structure Failures where
  n : Nat := 0
  msgs : Array String := #[]

def check (f : Failures) (ok : Bool) (msg : String) : Failures :=
  if ok then f else { n := f.n + 1, msgs := if f.msgs.size < 20 then f.msgs.push msg else f.msgs }

/-- §5.5.7: per mode, the rows of its table (16-byte cells of successive 128-byte lines). -/
def swizzleTables : List (Swizzle × List (List Nat)) :=
  [(.b32, [[0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6]]),
   (.b64, [[0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6], [2, 3, 0, 1, 6, 7, 4, 5],
           [3, 2, 1, 0, 7, 6, 5, 4]]),
   (.b128, [[0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6], [2, 3, 0, 1, 6, 7, 4, 5],
            [3, 2, 1, 0, 7, 6, 5, 4], [4, 5, 6, 7, 0, 1, 2, 3], [5, 4, 7, 6, 1, 0, 3, 2],
            [6, 7, 4, 5, 2, 3, 0, 1], [7, 6, 5, 4, 3, 2, 1, 0]]),
   (.b128Atom32, [[0, 1, 2, 3, 4, 5, 6, 7], [2, 3, 0, 1, 6, 7, 4, 5], [4, 5, 6, 7, 0, 1, 2, 3],
                  [6, 7, 4, 5, 2, 3, 0, 1]]),
   (.b128Atom64, [[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 0, 1, 2, 3]])]

def swizzles (f : Failures) : Failures := Id.run do
  let mut f := f
  for (mode, rows) in swizzleTables do
    -- Two repetitions of the pattern, from an aligned base.
    for rep in [0:2] do
      for r in [0:rows.length] do
        for p in [0:8] do
          let line := rep * rows.length + r
          let cell := rows[r]![p]!
          let got := mode.apply (8192 + line * 128 + cell * 16 + 5)
          f := check f (got == 8192 + line * 128 + p * 16 + 5)
            s!"{repr mode}: line {line} cell {cell} went to {got}"
  return f

/-- The worked examples of §9.7.17.5.1.2.1.3: type, K-major, swizzle, LBO and SBO in bytes, the
exact layout's shape and stride over `(mn, k)` in elements, and the CuTe swizzle's `B`. -/
def descExamples : List (InType × Bool × Swizzle × Nat × Nat × List Nat × List Nat × Nat × Nat × Nat) :=
  [(.tf32, true, .none, 256, 128, [8, 2, 4, 4], [4, 32, 1, 64], 0, 16, 16),
   (.tf32, true, .b32, 16, 256, [8, 2, 4, 2], [8, 64, 1, 4], 1, 16, 8),
   (.bf16, false, .none, 256, 128, [8, 1, 2, 8, 2], [1, 8, 64, 8, 128], 0, 16, 16),
   (.bf16, false, .b32, 256, 512, [8, 2, 2, 8, 2], [1, 8, 128, 16, 256], 1, 32, 16),
   (.bf16, false, .b64, 512, 1024, [8, 4, 2, 8, 2], [1, 8, 256, 32, 512], 2, 64, 16)]

def descs (f : Failures) : Failures := Id.run do
  let mut f := f
  for (t, km, sw, lbo, sbo, shape, stride, b, mnN, kN) in descExamples do
    let d : Desc := { start := 4096, lbo, sbo, baseOffset := 0, swizzle := sw }
    let esz := t.bits / 8
    -- The MN mode is the shape's first part (two entries K-major, three MN-major).
    let mnModes := if km then 2 else 3
    let mnShape := shape.take mnModes
    let mnSize := mnShape.foldl (· * ·) 1
    for mn in [0:mnN] do
      for k in [0:kN] do
        let i := mn % mnSize + mnSize * k
        let want := cuteSwizzle b (4096 + esz * cute shape stride i)
        match d.addr t km mn k with
        | .ok got => f := check f (got == want) s!"{repr t} {km} {repr sw}: ({mn}, {k}) at {got}, want {want}"
        | .error e => f := check f false e
  return f

def fragments (f : Failures) : Failures := Id.run do
  let mut f := f
  for n in [8, 64, 256] do
    for w in [0:4] do
      for l in [0:32] do
        for v in [0:n / 2] do
          let tid := 32 * w + l
          let i := cute [4, 8, 4] [128, 1, 16] tid + cute [2, 2, n / 8] [64, 8, 512] v
          f := check f (accPos w l v == (i % 64, i / 64)) s!"D m64n{n}: warp {w} lane {l} value {v}"
  for w in [0:4] do
    for l in [0:32] do
      for v in [0:8] do
        let i := cute [4, 8, 4] [128, 1, 16] (32 * w + l) + cute [2, 2, 2] [64, 8, 512] v
        f := check f (aPos .bf16 w l v == (i % 64, i / 64)) s!"A m64k16: warp {w} lane {l} value {v}"
  return f

def isOk (e : Except String Bool) (b : Bool) : Bool :=
  match e with
  | .ok x => x == b
  | .error _ => false

/-- Pair TMA keeps its data destinations while selecting completion destinations by mbar parity. -/
def tmaPairSignals (f : Failures) : Failures := Id.run do
  let mut f := f
  let ctx : SASS.Ctx :=
    { ntid := (32, 1, 1), nctaid := (8, 1, 1), ctaid := (1, 0, 0), warp := 0, threads := 32,
      clusterRank := 1, clusterSize := 8 }
  let env : SASS.Env := { ctx, const0 := Store.empty 0 0 }
  let ps : Array NV.Async.Piece := #[{ global := some 0x10000, smem := 1024, bytes := 16 }]
  let ranks := #[1, 3, 5, 7]
  for paired in [false, true] do
    for parity in [0, 1] do
      let mbar : UInt64 := (parity.toUInt64 <<< 24) ||| 0x32430
      let copy := SASS.AsyncSem.loadMulticast env ps 0x1000400 mbar 16 ranks paired
      let signalRanks := if paired && parity == 0 then #[0, 2, 4, 6] else #[1, 3, 5, 7]
      f := check f (copy.moves.size == 1 && copy.signals.size == 4) "TMA pair: four destinations"
      for j in [0:4] do
        let rank := ranks[j]!
        let signalRank := signalRanks[j]!
        let dataAt := SASS.AsyncSem.sharedAt env ((rank.toUInt64 <<< 24) ||| 1024)
        let signalAt := SASS.AsyncSem.sharedAt env ((signalRank.toUInt64 <<< 24) ||| 0x32430)
        let dst := copy.moves[0]!.dsts[j]!
        let sig := copy.signals[j]!
        f := check f (dst.space == dataAt.1 && dst.addrs == #[some dataAt.2])
          s!"TMA pair: data stays at rank {rank}"
        f := check f (sig.space == signalAt.1 && sig.addrs == #[some signalAt.2] &&
            sig.op == .mbar (.completeTx 16))
          s!"TMA pair {paired}, parity {parity}: completion rank {signalRank}"
  return f

/-- The SASS operand pair, high-half multicast mask, and box byte count reach pair completions. -/
def tmaPairInstruction (f : Failures) : Failures := Id.run do
  let mut f := f
  let map : TensorMap :=
    { dtype := .bf16, rank := 3, globalAddress := 0x10000, globalDim := #v[64, 64, 1, 1, 1],
      globalStrides := #v[128, 8192, 0, 0], boxDim := #v[64, 64, 1, 1, 1],
      elementStrides := #v[1, 1, 1, 1, 1] }
  let ctx : SASS.Ctx :=
    { ntid := (32, 1, 1), nctaid := (8, 1, 1), ctaid := (1, 0, 0), warp := 0, threads := 32,
      clusterRank := 1, clusterSize := 8 }
  let env : SASS.Env := { ctx, const0 := (Store.empty 0 TensorMap.size).writeBytes 0 map.encode }
  let kernel : SASS.Kernel :=
    { name := "paired-tma", regs := 1, code := #[], paramBase := 0, paramBytes := TensorMap.size }
  let warp := (SASS.Warp.init 1 32).setUR 8 0x1000400 |>.setUR 9 0x32430
    |>.setUR 33 0xaa0000
  let view := SASS.View.of (#[8, 9, 10, 11, 12, 33, 48, 49].map SASS.Loc.ur) warp
  for multicast in [false, true] do
    let instr : SASS.Instr :=
      { pc := 0, opcode := "UTMALDG",
        mods := if multicast then #["3D", "MULTICAST", "2CTA"] else #["3D", "2CTA"],
        operands := #[.mem (some ⟨.ur, 8⟩) false none none 0,
                      .mem (some ⟨.ur, 48⟩) false none none 0] ++
          (if multicast then #[.reg ⟨.ur, 33⟩] else #[]),
        ctrl := { stall := 1, yield := false, wbar := none, rbar := none, wait := 0, reuse := 0 } }
    match SASS.utma kernel env view instr with
    | .ok (.tensor t) =>
      match t.build #[] with
      | .ok copy =>
        let ranks := if multicast then #[0, 2, 4, 6] else #[0]
        f := check f (copy.signals.size == ranks.size) "UTMALDG.2CTA: signal count"
        for j in [0:ranks.size] do
          let at_ := (ranks[j]!.toUInt64 <<< 24) ||| 0x32430
          f := check f (copy.signals[j]!.space == .sharedCluster &&
              copy.signals[j]!.addrs == #[some at_] &&
              copy.signals[j]!.op == .mbar (.completeTx 8192))
            s!"UTMALDG.2CTA multicast{multicast}: completion {j}"
      | .error e => f := check f false s!"UTMALDG build: {e}"
    | .error e => f := check f false s!"UTMALDG: {e}"
    | _ => f := check f false "UTMALDG: expected tensor copy"
  return f

def mbarriers (f : Failures) : Failures := Id.run do
  let mut f := f
  let step (w : Except String UInt64) (op : MBarOp) := w.bind (MBarrierWord.apply op)
  let w0 := MBarrierWord.apply (.init 2) 0
  -- A fresh barrier reports the phase before the first complete (parity 1).
  f := check f (isOk (w0.map (MBarrierWord.parityDone · 1)) true) "fresh barrier, parity 1"
  f := check f (isOk (w0.map (MBarrierWord.parityDone · 0)) false) "fresh barrier, parity 0"
  let w1 := step w0 (.arrive 1 4096)
  f := check f (isOk (w1.map (MBarrierWord.parityDone · 0)) false) "one of two arrivals"
  let w2 := step w1 (.arrive 1)
  f := check f (isOk (w2.map (MBarrierWord.parityDone · 0)) false) "arrivals done, bytes pending"
  let w3 := step w2 (.completeTx 4000)
  f := check f (isOk (w3.map (MBarrierWord.parityDone · 0)) false) "96 bytes pending"
  let w4 := step w3 (.completeTx 96)
  f := check f (isOk (w4.map (MBarrierWord.parityDone · 0)) true) "phase 0 complete"
  f := check f (isOk (w4.map (MBarrierWord.parityDone · 1)) false) "phase 1 open"
  -- The state an arrive returns names its phase.
  match w0, w4 with
  | .ok a, .ok b =>
    f := check f (MBarrierWord.stateDone b a) "state of phase 0 is done in phase 1"
    f := check f (!MBarrierWord.stateDone b b) "state of phase 1 is not done"
  | _, _ => f := check f false "mbarrier sequence faulted"
  -- complete_tx before expect_tx: a negative tx-count holds the phase.
  let v := step (step (MBarrierWord.apply (.init 1) 0) (.completeTx 64)) (.arrive 1)
  f := check f (isOk (v.map (MBarrierWord.parityDone · 0)) false) "negative tx holds the phase"
  -- Only arrive-on and complete-tx may complete a phase (§9.7.15.16.6.1, §9.7.15.16.9).
  f := check f (isOk ((step v (.arrive 0 64)).map (MBarrierWord.parityDone · 0)) false)
    "expect_tx completes nothing"
  f := check f ((step (MBarrierWord.apply (.init 1) 0) (.arrive 2)) matches .error _) "over-arrival"
  for n in [1, 2, 128, 2 ^ 20 - 1] do
    f := check f (SASS.AsyncSem.initCount? (SASS.AsyncSem.initWord n) == some n) s!"init word {n}"
    f := check f (SASS.AsyncSem.initOf? (SASS.AsyncSem.initWord n ||| ((1 : UInt64) <<< 63)) == some (n, 1))
      s!"init word {n} in phase 1"
  -- A barrier initialized in phase 1: the phase of
  -- parity 0 counts as complete, 1 not, until its arrivals.
  let p1 := MBarrierWord.apply (.init 6 1) 0
  f := check f (isOk (p1.map (MBarrierWord.parityDone · 0)) true) "phase-1 init: parity 0 done"
  f := check f (isOk (p1.map (MBarrierWord.parityDone · 1)) false) "phase-1 init: parity 1 pending"
  f := check f (isOk ((step p1 (.arrive 6)).map (MBarrierWord.parityDone · 1)) true)
    "phase-1 init: its arrivals complete parity 1"
  -- The init value for 128 arrivals.
  f := check f (SASS.AsyncSem.initCount? ((((2 ^ 20 - 128 : Nat) <<< 1) ||| ((2 ^ 20 - 128) <<< 43)).toUInt64) == some 128)
    "init value"
  return f

/-- A 2-D f32 tensor of 10 × 5 elements, rows 48 bytes apart, and an 8 × 4 box at (6, 3): the
box covers columns 6–13 and rows 3–6, of which columns 6–9 and rows 3–4 are inside. -/
def tma (f : Failures) : Failures := Id.run do
  let mut f := f
  let m : TensorMap :=
    { dtype := .f32, rank := 2, globalAddress := 0x10000, globalDim := #v[10, 5, 1, 1, 1],
      globalStrides := #v[48, 0, 0, 0], boxDim := #v[8, 4, 1, 1, 1],
      elementStrides := #v[1, 1, 1, 1, 1] }
  match m.tile #[6, 3] 1024 with
  | .error e => f := check f false s!"tile: {e}"
  | .ok ps =>
    -- Expected bytes per shared offset: the global address, or none out of bounds.
    let mut want : Array (Option UInt64) := #[]
    for y in [0:4] do
      for x in [0:8] do
        let (gx, gy) := (6 + x, 3 + y)
        want := want.push (if gx < 10 && gy < 5 then some (0x10000 + (gy * 48 + gx * 4).toUInt64) else none)
    let mut got : Array (Option (Option UInt64)) := Array.replicate 32 none
    for p in ps do
      for e in [0:p.bytes / 4] do
        let i := (p.smem - 1024) / 4 + e
        got := got.set! i (some (p.global.map (· + (4 * e).toUInt64)))
    for i in [0:32] do
      f := check f (got[i]! == some want[i]!) s!"tile element {i}: {repr got[i]!}, want {repr want[i]!}"
    f := check f (m.boxBytes == 128) "box bytes"
  return f

/-- A 128-byte-swizzled box of 32-byte rows (8 f32 × 4): each row takes the swizzle's 128
bytes in shared memory, row `r`'s 16-byte chunk `q` at `128 r + 16 (q ^ r)`. -/
def tmaNarrow (f : Failures) : Failures := Id.run do
  let mut f := f
  let m : TensorMap :=
    { dtype := .f32, rank := 2, globalAddress := 0x10000, globalDim := #v[64, 64, 1, 1, 1],
      globalStrides := #v[256, 0, 0, 0], boxDim := #v[8, 4, 1, 1, 1],
      elementStrides := #v[1, 1, 1, 1, 1], swizzle := .b128 }
  match m.tile #[0, 0] 2048 with
  | .error e => f := check f false s!"narrow tile: {e}"
  | .ok ps =>
    let want : Array NV.Async.Piece := Id.run do
      let mut w := #[]
      for r in [0:4] do
        for q in [0:2] do
          w := w.push { global := some (0x10000 + r * 256 + q * 16).toUInt64,
                        smem := 2048 + 128 * r + 16 * (q ^^^ r), bytes := 16 }
      return w
    f := check f (ps == want) s!"narrow tile: {repr ps}"
    f := check f (m.boxBytes == 128) "narrow box bytes"
  return f

/-- The tensor-memory column allocator's two claims: `.ALIGN` takes the lowest aligned run,
without it the lowest run anywhere (granule 0 taken, 2 granules: 2 against 1). -/
def allocClaims (f : Failures) : Failures := Id.run do
  let mut f := f
  f := check f (NV.findSetAlign 0b0001 2 == some 2) "findSetAlign 0b0001 2"
  f := check f (NV.findSet 0b0001 2 == some 1) "findSet 0b0001 2"
  f := check f (NV.findSet 0 16 == some 0 && NV.findSet 0xffff 16 == some 16) "findSet 16"
  f := check f (NV.findSet 0xfffffffe 2 == none) "findSet full"
  f := check f (NV.AtomOp.apply .findSet .b32 #[1] #[2] #[] == some #[7]) "findSet sets its run"
  return f

/-- A tcgen05 MMA after a commit while the MMA before the commit is still running: it starts at
once (the commit tracks the earlier MMA, it orders nothing after it). -/
def mmaAfterCommit (f : Failures) : Failures := Id.run do
  let mut f := f
  let p := NV.Arch.b200
  let cfg := NV.Core.Memory.Config.ofParams p 128
  let mem := (NV.Core.MemSys.init cfg 1 (NV.Store.empty 0 128)).addCta 0 0 0 0 128
  let mma : NV.Async.MmaT := {
    loads := #[{ space := .shared, op := .ld, bytes := 16, addrs := #[some 0] }]
    result := fun _ => #[0]
    stores := fun d => #[{ space := .tmem, op := .st, bytes := 4, addrs := #[some 0], data := d }]
    macs := 4096, n := 64, inBits := 16 }
  let commit : NV.Async.Copy := { moves := #[], signals := #[], group := some .tcgen05, afterGroup := true }
  let o : NV.Core.Origin := { sm := 0, smsp := 0, cta := 0, slot := 0 }
  let run : Except String Bool := do
    let u := NV.Core.AsyncUnit.init 1 p.smsps.value
    let (u, mem, _) ← NV.Core.AsyncUnit.start p u mem 0 o (.mmaT mma) (fun _ w => w)
    let (u, mem, _) ← NV.Core.AsyncUnit.start p u mem 1 o (.copy commit) (fun _ w => w)
    let (u, _, _) ← NV.Core.AsyncUnit.start p u mem 2 o (.mmaT mma) (fun _ w => w)
    return u.agents[2]!.started
  match run with
  | .ok started => f := check f started "an MMA after a commit waited for the commit"
  | .error e => f := check f false s!"mma after commit: {e}"
  return f

/-- A wgmma group whose completion is known but later than a next commit stays pending to
`WARPGROUP.DEPBAR` until then: `wait_group 1` after the second commit waits for the first group's
writeback (cycle 1000), and at 1000 only the second is pending. -/
def wgmmaGroupKnownLater (f : Failures) : Failures := Id.run do
  let mut f := f
  let key := (0, 0, NV.Async.Group.wgmma.index)
  let agent : NV.Core.AsyncUnit.Agent :=
    { origin := { sm := 0, smsp := 0, slot := 0, cta := 0 }, body := .mma default, wb := fun _ w => w,
      group := some .wgmma }
  let first : NV.Core.AsyncUnit.GroupRec :=
    { sm := 0, slot := 0, kind := .wgmma, bar := none, remaining := 0, doneAt := some 1000, released := true }
  let u := NV.Core.AsyncUnit.init 1 4
  let u := { u with agents := #[agent], groups := #[first], open_ := ({} : Std.HashMap _ _).insert key #[0],
                    wgmma := ({} : Std.HashMap _ _).insert (0, 0) #[0],
                    lastGroup := ({} : Std.HashMap _ _).insert key 0,
                    groupCount := ({} : Std.HashMap _ _).insert key 1 }
  let (u, _, _) := NV.Core.AsyncUnit.commit u 200 0 0 .wgmma none
  f := check f (NV.Core.AsyncUnit.wgmmaPending u 0 0 200 == #[some 1000, none])
    s!"wgmma groups pending at 200: {repr (NV.Core.AsyncUnit.wgmmaPending u 0 0 200)}"
  f := check f (NV.Core.AsyncUnit.wgmmaPending u 0 0 1000 == #[none])
    s!"wgmma groups pending at 1000: {repr (NV.Core.AsyncUnit.wgmmaPending u 0 0 1000)}"
  return f

/-- A pseudo-random word. -/
def rword (seed x : Nat) : UInt32 :=
  let h := (x * 2654435761 + seed * 40503 + 12345) % 2 ^ 32
  let h := (h ^^^ (h >>> 13)) * 1274126177 % 2 ^ 32
  (h ^^^ (h >>> 16)).toUInt32

def fastWgmma (f : Failures) : Failures := Id.run do
  let mut f := f
  let mut seed := 0
  for (ty, acc) in [(InType.f16, Acc.f32), (.f16, .f16), (.bf16, .f32), (.tf32, .f32),
                    (.e4m3, .f32), (.e5m2, .f16)] do
    for n in [8, 24] do
      for regsA in [false, true] do
        for (negA, negB, scaleD) in [(false, false, true), (true, true, true), (false, true, false)] do
          for lanes in [Array.range 32, #[5], #[30]] do
            seed := seed + 1
            let K := ty.k
            let b : Desc := { start := 4096, lbo := 128, sbo := 256, baseOffset := 0, swizzle := .b64 }
            let aDesc : Desc := { start := 0, lbo := 128, sbo := 256, baseOffset := 0, swizzle := .none }
            let frag := (Array.range 32).map fun l => (Array.range 4).map fun r => rword seed (1000 + 4 * l + r)
            let s : Share :=
              { n, ty, acc, warp := seed % 4, a := if regsA then .regs frag else .smem aDesc true,
                b, bKMajor := seed % 2 == 0, negA := !ty.isInt && negA, negB := !ty.isInt && negB,
                scaleD, lanes }
            let some m := model ty acc | continue
            let rows := s.rows
            let aAddrs := match s.a with
              | .smem d km => rows.toArray.flatMap fun r => (Array.range K).map fun k =>
                  (d.addr ty km r k).toOption.getD 0
              | .regs _ => #[]
            let bAddrs := (Array.range n).flatMap fun c => (Array.range K).map fun k =>
              (b.addr ty s.bKMajor c k).toOption.getD 0
            let cs := chunks (aAddrs ++ bAddrs)
            let data := (Array.range (4 * cs.size)).map (rword (seed + 7))
            let c := (Array.range (lanes.size * accWords acc n)).map fun x =>
              -- moderate values: small exponents, so sums neither overflow nor vanish
              let w := rword (seed + 9) x
              if acc == .f16 then (w &&& 0x83ff83ff) ||| 0x30003000 else (w &&& 0x807fffff) ||| 0x3c000000
            let (aRow, bs) := s.operands rows aAddrs bAddrs (elem (image cs data) ty.bits)
            let want := s.finish aRow bs c
            let got := s.finishFast m (s.fast cs aAddrs bAddrs) rows data c
            f := check f (got == some want)
              s!"wgmma fast {repr ty}→{repr acc} n {n} regsA {regsA} neg {negA}/{negB} scaleD {scaleD} lanes {lanes.size}: {got.map (·.size)} vs {want.size} words, first difference at {(List.range want.size).find? fun i => (got.getD #[])[i]? != want[i]?}"
  return f

/-- `tcgen05.ld` lanes by warp: an address in lanes 0–31 reaches the warp's own quarter, as one
naming the quarter does; one naming another warp's quarter is refused. -/
def tmemQuarters (f : Failures) : Failures := Id.run do
  let mut f := f
  let at_ (lane col : Nat) : Nat := lane * 2 ^ 16 + col
  let lanes (r : Except String (Req × Nat)) : Option (List Nat) := match r with
    | .ok (q, _) => some (q.addrs.toList.filterMap fun a => a.map fun b => b.toNat / 4 / Tcgen05.columns)
    | .error _ => none
  for (shape, warp) in ([(.s32x32b, 1), (.s16x256b, 3), (.s16x64b, 2)] : List (Tcgen05.Shape × Nat)) do
    let implicit : Option (List Nat) := lanes (Tcgen05.moveReq MemOp.ld shape 1 0 (at_ 0 8) warp false)
    let named : Option (List Nat) := lanes (Tcgen05.moveReq MemOp.ld shape 1 0 (at_ (32 * warp) 8) warp false)
    f := check f (implicit.isSome && implicit == named)
      s!"{repr shape} warp {warp}: lane 0 reaches {implicit}, lane {32 * warp} {named}"
    f := check f ((implicit.getD []).all fun l => 32 * warp ≤ l && l < 32 * warp + 32)
      s!"{repr shape} warp {warp}: lanes {implicit} outside the warp's quarter"
    -- Another warp's quarter, not quarter 0 (which names the warp's own).
    let other := if warp % 4 == 1 then 64 else 32
    f := check f (lanes (Tcgen05.moveReq MemOp.ld shape 1 0 (at_ other 8) warp false)).isNone
      s!"{repr shape} warp {warp}: lane {other} accepted"
  return f

def main : IO UInt32 := do
  let f := ({} : Failures) |> swizzles |> descs |> fragments |> mbarriers |> tma |> tmaNarrow
    |> fastWgmma |> tmemQuarters |> allocClaims |> mmaAfterCommit |> wgmmaGroupKnownLater |> tmaPairSignals
    |> tmaPairInstruction
  for m in f.msgs do IO.eprintln m
  IO.println s!"async-tests: {f.n} failures"
  return if f.n == 0 then 0 else 1
