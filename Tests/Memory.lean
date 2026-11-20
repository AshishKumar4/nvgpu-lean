import NV.Core.Memory
import NV.Arch.H200

/-!
`lake exe memory-tests`: the memory system on the H200 parameters, request by request. An idle
machine reproduces each latency the parameters state; bank conflicts, coalescing, caches,
bandwidths, atomics, fences and the MIO queue behave as `NV.Core.Memory` describes; data come out
as a sequential execution in issue order gives them.
-/

open NV NV.Core NV.Core.Memory

def p : Params := NV.Arch.h200
def cfg : Config := Config.ofParams p 0
def base : UInt64 := 0x10000000
def size : Nat := 1 <<< 24

/-- Every SM with one CTA of 48 KiB, clusters of four (SMs 0–3 the first). -/
def fresh : MemSys := Id.run do
  let mut m := MemSys.init cfg p.sms.value (Store.empty base size)
  for sm in [0:p.sms.value] do m := m.addCta sm 0 (sm / 4) (sm % 4) 49152
  return m

def origin (sm : Nat := 0) (slot : Nat := 0) : Origin := { sm, smsp := slot % 4, cta := 0, slot }

def req (space : Space) (op : MemOp) (addrs : List (Option UInt64)) (bytes : Nat := 4)
    (data : Array UInt32 := #[]) (hint : Array String := #[]) : Req :=
  { space, op, bytes, addrs := addrs.toArray, data, hint := { ops := hint } }

def lanes (f : Nat → UInt64) (n : Nat := 32) : List (Option UInt64) :=
  (List.range n).map (some ∘ f)

/-- Run everything in flight, event by event, so time stays where the last event was. -/
def drain (m : MemSys) : Except String (MemSys × Array Delivery) := do
  let mut m := m
  let mut out := #[]
  for _ in [0:10000000] do
    let some t := m.next | break
    let (m', ds) ← m.advance t
    m := m'
    out := out ++ ds
  return (m, out)

/-- Issue each request at its cycle, run to the end, and return the deliveries in issue order
(ids are reused once delivered, so a delivery belongs to the earliest undelivered request with its
id). -/
def runAll (m : MemSys) (rs : List (Nat × Origin × Req)) : Except String (MemSys × Array Delivery) := do
  let mut m := m
  let mut ids : Array Nat := #[]
  let mut got : Array (Option Delivery) := #[]
  let place (ids : Array Nat) (got : Array (Option Delivery)) (d : Delivery) :=
    match (List.range ids.size).find? (fun i => ids[i]! == d.id && got[i]!.isNone) with
    | some i => got.set! i (some d)
    | none => got
  for (t, o, r) in rs do
    let (m', ds) ← m.advance t
    for d in ds do got := place ids got d
    let (m'', id) := m'.issue t o r
    m := m''
    ids := ids.push id
    got := got.push none
  let (m', ds) ← drain m
  for d in ds do got := place ids got d
  return (m', got.map (·.getD default))

/-- The first line at or after `from` whose slice is (`near = true`) or is not in SM 0's partition. -/
def lineIn (near : Bool) (from_ : Nat) : Nat := Id.run do
  for i in [0:4096] do
    let line := base.toNat / cfg.lineBytes + from_ + i
    let s := spread line cfg.slices
    if (s * cfg.partitions / cfg.slices == cfg.nearPartition[0]!) == near then return line
  return 0

structure T where
  failures : Array String := #[]

def T.check (t : T) (name : String) (ok : Bool) (detail : String := "") : T :=
  if ok then t else { failures := t.failures.push s!"{name}: {detail}" }

def tests : Except String T := do
  let mut t : T := {}
  let mem := p.mem
  let g (line : Nat) : UInt64 := (line * cfg.lineBytes).toUInt64

  -- Idle latencies: shared, L1 hit, L2 near and far hits, DRAM, cluster shared memory.
  let lds := req .shared .ld [some 0]
  let (_, ds) ← runAll fresh [(100, origin, lds)]
  t := t.check "smem latency" (ds[0]!.at_ - 100 == mem.smemLatency.value) s!"{ds[0]!.at_ - 100}"
  let near := g (lineIn true 0)
  let far := g (lineIn false 0)
  let dramExtra := mem.dramLatency.value - (mem.l2NearHitLatency.value + mem.l2FarHitLatency.value) / 2
  let (m, ds) ← runAll fresh [(0, origin, req .global .ld [some near]),
                              (100, origin, req .global .ld [some far])]
  let dNear := ds[0]!.at_
  let dFar := ds[1]!.at_ - 100
  t := t.check "dram, near slice" (dNear ≥ mem.l2NearHitLatency.value + dramExtra &&
    dNear ≤ mem.l2NearHitLatency.value + dramExtra + 4) s!"{dNear}"
  t := t.check "dram, far slice" (dFar ≥ mem.l2FarHitLatency.value + dramExtra &&
    dFar ≤ mem.l2FarHitLatency.value + dramExtra + 4) s!"{dFar}"
  t := t.check "dram latency is the near/far mean" ((dNear + dFar) / 2 ≥ mem.dramLatency.value &&
    (dNear + dFar) / 2 ≤ mem.dramLatency.value + 4) s!"{dNear} {dFar}"
  -- A far line first stored (so it is in its home slice only), then read twice: the first read
  -- crosses to the home slice, the second hits the copy it left in the near partition.
  let farB := g (lineIn false 200)
  let (m, ds) ← runAll m [(5000, origin, req .global .ld [some near]),
                          (6000, origin, req .global .ld [some near] (hint := #[".cg"])),
                          (6500, origin, req .global .st [some farB] (data := #[3])),
                          (7000, origin, req .global .ld [some farB] (hint := #[".cg"])),
                          (8000, origin, req .global .ld [some farB] (hint := #[".cg"]))]
  t := t.check "L1 hit latency" (ds[0]!.at_ - 5000 == mem.l1HitLatency.value) s!"{ds[0]!.at_ - 5000}"
  t := t.check "L2 near hit latency" (ds[1]!.at_ - 6000 == mem.l2NearHitLatency.value)
    s!"{ds[1]!.at_ - 6000}"
  t := t.check "L2 far hit latency" (ds[3]!.at_ - 7000 == mem.l2FarHitLatency.value)
    s!"{ds[3]!.at_ - 7000}"
  let again := if cfg.farCopies then mem.l2NearHitLatency.value else mem.l2FarHitLatency.value
  t := t.check "far line read again (near copy if copies are on)" (ds[4]!.at_ - 8000 == again && ds[4]!.resp == #[3])
    s!"{ds[4]!.at_ - 8000}"
  let k := m.counters
  t := t.check "L1 and L2 counters" (k.globalLdRequests == 6 && k.globalLdHitSectors == 1 &&
    k.l2MissSectors == 2 && k.l2HitSectors == 3 && k.dramReadBytes == 64) s!"{repr k}"
  -- Two SMs of one partition read a far line, in its home slice only, a cycle apart: the second
  -- waits at the copy slice for the sector the first is bringing over the link, rather than
  -- crossing too.
  if cfg.farCopies then
    let farC := g (lineIn false 400)
    let sm2 := ((List.range p.sms.value).filter fun s =>
      s != 0 && cfg.nearPartition[s]! == cfg.nearPartition[0]!).head!
    let (m, _) ← runAll fresh [(0, origin, req .global .st [some farC] (data := #[5]))]
    let crossed := m.counters.l2FarSectors
    let (m, ds) ← runAll m [(7000, origin, req .global .ld [some farC] (hint := #[".cg"])),
                            (7001, origin sm2, req .global .ld [some farC] (hint := #[".cg"]))]
    t := t.check "far reads of a line in flight cross once" (m.counters.l2FarSectors - crossed == 1 &&
      ds[1]!.resp == #[5] && ds[1]!.at_ - 7001 > mem.l2NearHitLatency.value)
      s!"{m.counters.l2FarSectors - crossed} sectors crossed; the second answered after {ds[1]!.at_ - 7001}"
  let remote := req .sharedCluster .ld [some (((1 : UInt64) <<< 24) + 64)]
  let (_, ds) ← runAll fresh [(100, origin, remote)]
  t := t.check "dsmem latency" (ds[0]!.at_ - 100 == mem.dsmemLatency.value) s!"{ds[0]!.at_ - 100}"

  -- MSHRs: a load waits for the transaction bringing each of its missing sectors. Sector 1 of a
  -- line is in the L2 only, sector 0 nowhere; the third load (sector 0 again) must wait for the
  -- first one's DRAM fill, not for the second one's L2 hit.
  let ln := g (lineIn true 300)
  let (m, _) ← runAll fresh [(0, origin, req .global .ld [some (ln + 32)] (hint := #[".cg"]))]
  let (_, ds) ← runAll m [(2000, origin, req .global .ld [some ln]),
                          (2001, origin, req .global .ld [some (ln + 32)]),
                          (2002, origin, req .global .ld [some ln])]
  t := t.check "a merged miss waits for its own sector's fill" (ds[2]!.at_ ≥ ds[0]!.at_ && ds[1]!.at_ < ds[0]!.at_)
    s!"{ds[0]!.at_} {ds[1]!.at_} {ds[2]!.at_}"
  -- Local memory: lanes' words interleave, so one lane's 16 bytes span four lines.
  let ml := fresh.addWarp 0 0 0 64
  let (m, ds) ← runAll ml [(0, origin, req .«local» .st [some 16] (bytes := 16) (data := #[1, 2, 3, 4])),
                           (1, origin, req .«local» .ld [some 16] (bytes := 16))]
  t := t.check "local 16-byte access: four lines, data back" (ds[1]!.resp.extract 0 4 == #[1, 2, 3, 4] &&
    m.counters.localStSectors == 4 && m.counters.localLdSectors == 4) s!"{ds[1]!.resp} {repr m.counters}"

  -- Bank conflicts: wavefronts per request.
  let w (bytes : Nat) (f : Nat → Nat) := (wavefronts 32 4 bytes ((lanes (fun i => (f i).toUInt64)).toArray)).1
  t := t.check "stride 1 is one wavefront" (w 4 (4 * ·) == 1)
  t := t.check "one word is a broadcast" (w 4 (fun _ => 64) == 1)
  t := t.check "stride 32 words is 32-way" (w 4 (128 * ·) == 32)
  t := t.check "stride 2 words is 2-way" (w 4 (8 * ·) == 2)
  t := t.check "8-byte stride 1 is two wavefronts" (w 8 (8 * ·) == 2)
  t := t.check "16-byte stride 1 is four wavefronts" (w 16 (16 * ·) == 4)
  t := t.check "16-byte stride 2 is eight wavefronts" (w 16 (32 * ·) == 8)
  let (m, ds) ← runAll fresh [(100, origin, req .shared .ld (lanes (128 * ·.toUInt64)))]
  t := t.check "32-way conflict takes 31 more cycles" (ds[0]!.at_ - 100 == mem.smemLatency.value + 31)
    s!"{ds[0]!.at_ - 100}"
  t := t.check "conflict counter" (m.counters.sharedBankConflicts == 31 && m.counters.sharedWavefronts == 32)
    s!"{repr m.counters}"

  -- Coalescing.
  let c1 := coalesce 32 128 4 (lanes (fun i => base + 4 * i.toUInt64)).toArray
  t := t.check "unit stride is one line of four sectors" (c1.size == 1 && c1[0]!.sectors == 15)
  let c2 := coalesce 32 128 4 (lanes (fun i => base + 128 * i.toUInt64)).toArray
  t := t.check "line stride is 32 lines of one sector" (c2.size == 32 && c2.all (·.sectors == 1))
  let c3 := coalesce 32 128 16 (lanes (fun i => base + 16 * i.toUInt64)).toArray
  t := t.check "16-byte unit stride is four full lines" (c3.size == 4 && c3.all (·.sectors == 15))

  -- Data: a warp's stores, loads and atomics come out in issue order.
  let a := base + 4096
  let st (v : Nat) := req .global .st [some a] (data := #[v.toUInt32])
  let ld := req .global .ld [some a]
  let (_, ds) ← runAll fresh [(0, origin, st 7), (1, origin, ld), (2, origin, st 9), (3, origin, ld),
    (4, origin, req .global (.atom .add .u32) [some a] (data := #[1])), (5, origin, ld)]
  t := t.check "issue order per address" (ds[1]!.resp == #[7] && ds[3]!.resp == #[9] &&
    ds[4]!.resp == #[9] && ds[5]!.resp == #[10])
    s!"{ds[1]!.resp} {ds[3]!.resp} {ds[4]!.resp} {ds[5]!.resp}"
  let atom := req .global (.atom .add .u32) (lanes (fun _ => a)) (data := Array.replicate 32 1)
  let (m, ds) ← runAll fresh [(0, origin, atom)]
  let olds := (ds[0]!.resp).toList.map (·.toNat)
  t := t.check "32 atomic adds to one word" ((m.global.read a 4) == #[32] &&
    olds.mergeSort == List.range 32) s!"{m.global.read a 4} {olds}"
  -- The same through shared memory, and a peer's shared memory.
  let (m, ds) ← runAll fresh [(0, origin, req .sharedCluster .st [some (((2 : UInt64) <<< 24) + 8)] (data := #[5])),
                                   (1, origin, req .sharedCluster .ld [some (((2 : UInt64) <<< 24) + 8)])]
  t := t.check "store then load to a peer CTA" (ds[1]!.resp == #[5] && ((m.sharedOf 2 0).read 8 4) == #[5])
    s!"{ds[1]!.resp}"

  -- L1 replacement: `ways + 1` lines of one set evict the least recently used.
  let setStride := cfg.l1Sets * cfg.lineBytes
  let fill := (List.range (cfg.l1Ways + 1)).map fun i =>
    (i * 2000, origin, req .global .ld [some (base + (i * setStride).toUInt64)])
  let (m, _) ← runAll fresh fill
  let hits0 := m.counters.globalLdHitSectors
  let (m, _) ← runAll m [(200000, origin, req .global .ld [some base]),
                            (201000, origin, req .global .ld [some (base + (cfg.l1Ways * setStride).toUInt64)])]
  t := t.check "LRU: oldest line evicted, newest kept" (m.counters.globalLdHitSectors == hits0 + 1)
    s!"{hits0} {m.counters.globalLdHitSectors}"

  -- Bandwidth: a slice serves `l2SliceBytesPerCycle`, so the SMs of its partition reading one
  -- warm line at once deliver spread over at least their bytes at that rate.
  let line := lineIn true 64
  let read (sm : Nat) := req .global .ld (lanes (fun i => g line + 4 * i.toUInt64)) (hint := #[".cg"])
  let nearSms := (List.range 132).filter (cfg.nearPartition[·]! == cfg.nearPartition[0]!)
  let (m, _) ← runAll fresh [(0, origin, read 0)]
  let (_, ds) ← runAll m (nearSms.map fun sm => (10000, origin sm, read sm))
  let first := ds.foldl (fun x d => min x d.at_) ds[0]!.at_
  let last := ds.foldl (fun x d => max x d.at_) 0
  let spreadFloor := (nearSms.length - 1) * cfg.lineBytes / mem.l2SliceBytesPerCycle.value
  t := t.check "one slice's bandwidth spreads simultaneous reads" (last - first ≥ spreadFloor - 1)
    s!"{nearSms.length} SMs: {first}..{last}, floor {spreadFloor}"
  -- DRAM: 132 SMs each reading 32 distinct cold lines take at least bytes / peak beyond the latency.
  let stream := (List.range 132).flatMap fun sm => (List.range 32).map fun j =>
    (j, origin sm, req .global .ld (lanes (fun i => base + ((sm * 32 + j) * 128 + 4 * i).toUInt64)))
  let (m, ds) ← runAll fresh stream
  let last := ds.foldl (fun x d => max x d.at_) 0
  let bytes := 132 * 32 * 128
  let floor := dNear + bytes * p.clockMHz.value * 1000000 / mem.dramBytesPerSec.value - 32
  t := t.check "DRAM peak bounds a cold stream" (last ≥ floor && m.counters.dramReadBytes == bytes)
    s!"{last} < {floor}, {m.counters.dramReadBytes}"

  -- Fences wait for the warp's requests; the MIO queue holds `mioQueueDepth`.
  let m0 := fresh
  let (m1, sid) := m0.issue 0 origin (req .global .st [some far] (data := #[1]))
  let (m2, fid) := m1.fence 1 origin 9
  let (_, ds) ← drain m2
  let sAt := ((ds.find? (·.id == sid)).map (·.at_)).getD 0
  let fAt := ((ds.find? (·.id == fid)).map (·.at_)).getD 0
  t := t.check "fence its own cost after the store's acknowledgement" (fAt == sAt + 9 && sAt > 0)
    s!"{sAt} {fAt}"
  let mut m := fresh
  for _ in [0:mem.mioQueueDepth.value] do m := ((m.reserve 0 0 0).issue 0 origin lds).1
  t := t.check "MIO queue full" (!m.accepts 0 0 && m.accepts 0 1)
  let (m', _) ← m.advance 0
  t := t.check "MIO queue drains" (m'.accepts 0 0)
  -- A barrier may count a warp once its requests have performed, not before.
  let (m1, _) := (fresh.reserve 0 0 0).issue 10 origin (req .shared .st [some 64] (data := #[7]))
  let (m2, _) ← m1.advance 10
  let (m3, _) ← m2.advance (10 + cfg.front)
  t := t.check "performed after the data port, not before" (!m1.performed 0 0 && !m2.performed 0 0 &&
    m3.performed 0 0 && (m3.sharedOf 0 0).read 64 4 == #[7])
  -- A CTA retired while a reserved request has not arrived keeps its memory until that request
  -- has performed (`drain` fails if the request finds it gone), and frees it after.
  let m1 := (fresh.reserve 0 0 0).removeCta 0 0
  let (m2, _) := m1.issue 20 origin (req .shared .st [some 8] (data := #[9]))
  let (m3, _) ← drain m2
  t := t.check "retired CTA's memory outlives its reserved request" ((m3.sharedOf 0 0).size == 0)
    s!"{(m3.sharedOf 0 0).size}"

  return t

/-- Requests per second through the memory system alone: every SM streaming coalesced loads. -/
def bench : IO Unit := do
  let n := 20000
  let t0 ← IO.monoMsNow
  let mut m := fresh
  let mut got := 0
  for i in [0:n] do
    let t := i / 132
    match m.advance t with
    | .error e => IO.eprintln s!"memory-tests: bench: {e}"; return
    | .ok (m', ds) =>
      got := got + ds.size
      m := (m'.issue t (origin (i % 132))
        (req .global .ld (lanes (fun l => base + ((i % 4096) * 128 + 4 * l).toUInt64)))).1
  match drain m with
  | .error e => IO.eprintln s!"memory-tests: bench: {e}"
  | .ok (_, ds) =>
    let ms := (← IO.monoMsNow) - t0
    IO.println s!"memory-tests: {got + ds.size} warp loads in {ms} ms ({(got + ds.size) * 1000 / max ms 1} per second)"

def main : IO UInt32 := do
  bench
  match tests with
  | .error e => IO.eprintln s!"memory-tests: {e}"; return 1
  | .ok t =>
    for f in t.failures do IO.eprintln s!"memory-tests: FAIL {f}"
    if t.failures.isEmpty then IO.println "memory-tests: all passed"; return 0
    return 1
