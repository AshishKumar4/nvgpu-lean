import NV.Memory.Flat
import NV.Memory.Performed
import NV.Memory.StoreLemmas

/-!
# The request machine is sound: `Mem.sound`

A run of the request machine is a sequence of `Op`s it accepts: issues, per-lane deliveries,
fences, barrier arrivals and waits, in the order the timing core made them. `Ghost` replays a run
and lists the memory-model events it performed, in the order they were performed: a delivered
lane's events (`Req.laneEvents`, with the bytes its reads returned), and one event per lane of
each fence, arrival and wait. Each thread's events are numbered in issue order.

`sound`: the events of every run form a `Performed` list, so their execution is consistent
(`Performed.consistent`): every run of the simulator is an execution the PTX memory model
accepts. The run's issuers must be fixed per queue slot with disjoint threads (`Op.Ok`), and the
accesses must not overlap partially (`Uniform`), the model's own limit.
-/

namespace NV.Flat

open NV.Store NV.PTX


theorem writeBytes_props (bs : List UInt8) : ∀ (s : Store) (off : Nat), s.WF → s.Fits →
    off + bs.length ≤ s.size →
    (writeBytes s off bs).WF ∧ (writeBytes s off bs).Fits ∧ (writeBytes s off bs).base = s.base ∧
      (writeBytes s off bs).size = s.size ∧
      ∀ o, (writeBytes s off bs).readByte o =
        if off ≤ o ∧ o < off + bs.length then bs.getD (o - off) 0 else s.readByte o := by
  induction bs with
  | nil =>
    intro s off hw hf _
    exact ⟨hw, hf, rfl, rfl, fun o => by
      simp only [writeBytes, List.length_nil, Nat.add_zero]; rw [if_neg (by omega)]⟩
  | cons b bs ih =>
    intro s off hw hf hlen
    simp only [List.length_cons] at hlen
    have hpg := page_lt hf (show off < s.size by omega)
    obtain ⟨h₁, h₂, h₃, h₄, h₅⟩ := ih (s.writeByte off b) (off + 1) (hw.writeByte off b)
      (hf.writeByte off b) (by have := sizeField_writeByte s off b; omega)
    refine ⟨h₁, h₂, by rw [writeBytes, h₃, base_writeByte], by rw [writeBytes, h₄,
      sizeField_writeByte], fun o => ?_⟩
    rw [writeBytes, h₅ o, readByte_writeByte hw off o b hpg]
    simp only [List.length_cons]
    by_cases hlo : off + 1 ≤ o ∧ o < off + 1 + bs.length
    · rw [if_pos hlo, if_pos ⟨by omega, by omega⟩]
      obtain ⟨k, hk⟩ : ∃ k, o - off = k + 1 := ⟨o - (off + 1), by omega⟩
      have : o - (off + 1) = k := by omega
      rw [hk, this]; rfl
    · rw [if_neg hlo]
      by_cases he : o = off
      · subst he; rw [if_pos rfl, if_pos ⟨Nat.le_refl _, by omega⟩]; simp
      · rw [if_neg he, if_neg (by omega)]

/-! ## The pieces of a lane -/

/-- What every request's pieces satisfy: they lie inside the access, reads come first, a
read-modify-write's read is followed by its write of the same bytes, and writes do not overlap. -/
structure PiecesOk (r : Req) : Prop where
  size_pos : ∀ p ∈ r.pieces, 0 < p.size
  inside : ∀ p ∈ r.pieces, p.off + p.size ≤ r.bytes
  reads_first : ∀ (a b : Nat) (ha : a < r.pieces.length) (hb : b < r.pieces.length), a < b →
    r.pieces[b].write = false → r.pieces[a].write = false
  rmw : ∀ (a : Nat) (ha : a < r.pieces.length), r.pieces[a].rmwRead = true →
    ∃ h : a + 1 < r.pieces.length, r.pieces[a].write = false ∧ r.pieces[a + 1].write = true ∧
      r.pieces[a + 1].off = r.pieces[a].off ∧ r.pieces[a + 1].size = r.pieces[a].size
  disjoint : ∀ (a b : Nat) (ha : a < r.pieces.length) (hb : b < r.pieces.length), a < b →
    r.pieces[a].write = true → r.pieces[b].write = true →
      r.pieces[a].off + r.pieces[a].size ≤ r.pieces[b].off
  writes : (r.pieces.filter (·.write)).Pairwise fun a b => a.off + a.size ≤ b.off

theorem uniform_ok {r : Req} (w : Bool) (k : Nat) (hk : 0 < k)
    (hp : r.pieces = (List.range (r.bytes / k)).map fun j => { write := w, off := j * k, size := k }) :
    PiecesOk r := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;> simp only [hp]
  · intro p hp; simp only [List.mem_map, List.mem_range] at hp; obtain ⟨j, _, rfl⟩ := hp; exact hk
  · intro p hp
    simp only [List.mem_map, List.mem_range] at hp
    obtain ⟨j, hj, rfl⟩ := hp
    show j * k + k ≤ r.bytes
    calc j * k + k = (j + 1) * k := by rw [Nat.succ_mul]
      _ ≤ r.bytes / k * k := Nat.mul_le_mul_right _ hj
      _ ≤ r.bytes := Nat.div_mul_le_self _ _
  · intro a b ha hb _ h; simpa using h
  · intro a ha h; simp at h
  · intro a b ha hb hab _ _
    simp only [List.getElem_map, List.getElem_range]
    calc a * k + k = (a + 1) * k := by rw [Nat.succ_mul]
      _ ≤ b * k := Nat.mul_le_mul_right _ hab
  · apply List.Pairwise.filter
    rw [List.pairwise_map]
    exact List.pairwise_lt_range.imp fun {a b} hab =>
      show a * k + k ≤ b * k by
        calc a * k + k = (a + 1) * k := by rw [Nat.succ_mul]
          _ ≤ b * k := Nat.mul_le_mul_right _ hab

theorem rmw_ok {r : Req} (hpos : 0 < r.bytes)
    (hp : r.pieces = [{ write := false, off := 0, size := r.bytes, rmwRead := true },
                      { write := true, off := 0, size := r.bytes }]) : PiecesOk r := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;> simp only [hp]
  · intro p hp; simp at hp; rcases hp with rfl | rfl <;> exact hpos
  · intro p hp; simp at hp; rcases hp with rfl | rfl <;> simp
  · intro a b ha hb hab h
    simp only [List.length_cons, List.length_nil] at ha hb
    have : a = 0 := by omega
    have : b = 1 := by omega
    subst_vars; simp at h
  · intro a ha h
    simp only [List.length_cons, List.length_nil] at ha
    have : a = 0 := by
      rcases Nat.lt_or_ge a 1 with c | c
      · omega
      · have : a = 1 := by omega
        subst this; simp at h
    subst this
    exact ⟨by simp, by simp⟩
  · intro a b ha hb hab h₁ h₂
    simp only [List.length_cons, List.length_nil] at ha hb
    have : a = 0 := by omega
    subst this; simp at h₁
  · simp

theorem pieces_ok {r : Req} (hb : [1, 2, 4, 8, 16].contains r.bytes = true) : PiecesOk r := by
  have hpos : 0 < r.bytes := by
    cases h : r.bytes with
    | zero => rw [h] at hb; simp at hb
    | succ n => omega
  have hk : 0 < r.granule := by
    unfold Req.granule Req.elemBytes; split <;> (try split) <;> omega
  cases hop : r.op with
  | ld => exact uniform_ok false _ hk (by simp [Req.pieces, hop])
  | st => exact uniform_ok true _ hk (by simp [Req.pieces, hop])
  | atom op ty => exact rmw_ok hpos (by simp [Req.pieces, hop])
  | red op ty => exact rmw_ok hpos (by simp [Req.pieces, hop])
  | mbar op => exact rmw_ok hpos (by simp [Req.pieces, hop])

theorem laneEvents_getElem (r : Req) (lane t base : Nat) (l : Loc) (bytes : List UInt8) (a : Nat)
    (ha : a < (r.laneEvents lane t base l bytes).length) :
    (r.laneEvents lane t base l bytes)[a] =
      { e := r.event t (r.pieces[a]'(by simpa [Req.laneEvents] using ha)), seq := base + a,
        loc := { l with addr := l.addr + (r.pieces[a]'(by simpa [Req.laneEvents] using ha)).off },
        size := (r.pieces[a]'(by simpa [Req.laneEvents] using ha)).size,
        rmwRead := (r.pieces[a]'(by simpa [Req.laneEvents] using ha)).rmwRead,
        val := if (r.pieces[a]'(by simpa [Req.laneEvents] using ha)).write then
            (r.written lane (r.pieces[a]'(by simpa [Req.laneEvents] using ha)) bytes).getD []
          else bytes.extract (r.pieces[a]'(by simpa [Req.laneEvents] using ha)).off
            ((r.pieces[a]'(by simpa [Req.laneEvents] using ha)).off +
              (r.pieces[a]'(by simpa [Req.laneEvents] using ha)).size) } := by
  simp [Req.laneEvents, List.getElem_zipIdx]

theorem laneEvents_length (r : Req) (lane t base : Nat) (l : Loc) (bytes : List UInt8) :
    (r.laneEvents lane t base l bytes).length = r.pieces.length := by
  simp [Req.laneEvents]

/-! ## Performing a lane -/

theorem performWrites_ok {r : Req} {lane off : Nat} {read : List UInt8} :
    ∀ (ps : List Piece) (s s' : Store), performWrites r lane off read ps s = .ok s' →
      s.WF → s.Fits → (∀ p ∈ ps, off + p.off + p.size ≤ s.size) →
      ps.Pairwise (fun a b => a.off + a.size ≤ b.off) →
      s'.WF ∧ s'.Fits ∧ s'.base = s.base ∧ s'.size = s.size ∧
      (∀ p ∈ ps, ∃ v, r.written lane p read = some v ∧ v.length = p.size) ∧
      (∀ o, (∀ p ∈ ps, ¬(off + p.off ≤ o ∧ o < off + p.off + p.size)) → s'.readByte o = s.readByte o) ∧
      (∀ p ∈ ps, ∀ o, off + p.off ≤ o → o < off + p.off + p.size →
        s'.readByte o = ((r.written lane p read).getD []).getD (o - (off + p.off)) 0)
  | [], s, s', h, hw, hf, _, _ => by
    simp only [performWrites, Except.ok.injEq] at h
    subst h
    exact ⟨hw, hf, rfl, rfl, by simp, fun _ _ => rfl, by simp⟩
  | p :: ps, s, s', h, hw, hf, hin, hpw => by
    simp only [performWrites] at h
    split at h
    · next v hv =>
      split at h
      · next hlen =>
        have hin₀ := hin p (by simp)
        obtain ⟨w₁, f₁, b₁, z₁, r₁⟩ := writeBytes_props v s (off + p.off) hw hf (by omega)
        obtain ⟨w₂, f₂, b₂, z₂, e₂, u₂, c₂⟩ := performWrites_ok ps _ s' h w₁ f₁
          (fun p' hp' => by rw [z₁]; exact hin p' (by simp [hp'])) (List.pairwise_cons.1 hpw).2
        have hdis := (List.pairwise_cons.1 hpw).1
        refine ⟨w₂, f₂, b₂.trans b₁, z₂.trans z₁, ?_, ?_, ?_⟩
        · intro p' hp'
          rcases List.mem_cons.1 hp' with rfl | hp'
          · exact ⟨v, hv, hlen⟩
          · exact e₂ p' hp'
        · intro o ho
          rw [u₂ o (fun p' hp' => ho p' (by simp [hp'])), r₁ o, hlen, if_neg (ho p (by simp))]
        · intro p' hp' o h₁ h₂
          rcases List.mem_cons.1 hp' with rfl | hp'
          · -- Later pieces lie after this one.
            rw [u₂ o (fun q hq hc => by have := hdis q hq; omega), r₁ o, hlen, if_pos ⟨h₁, by omega⟩, hv]
            rfl
          · exact c₂ p' hp' o h₁ h₂
      · cases h
    · cases h

/-! ## Runs and their events -/

inductive Op where
  | issue (who : Issuer) (r : Req) (cta : Nat)
  | deliver (slot id lane : Nat)
  /-- `fence.*` (`SyncOp.fence`, `proxyFence`, `aliasFence`) by lanes `lanes`. -/
  | fence (who : Issuer) (lanes : List Nat) (op : SyncOp)
  /-- Lanes `lanes` arrive on barrier `b` (`bar` id `id`, or the cluster barrier when `none`),
  which completes at `expected` threads. -/
  | arrive (who : Issuer) (lanes : List Nat) (id : Option Nat) (b expected : Nat)
  /-- Lanes `lanes` wait on instance `inst` of barrier `b`, which expects `expected` threads. -/
  | wait (who : Issuer) (lanes : List Nat) (id : Option Nat) (b inst expected : Nat)
  /-- A new issuer slot (`Flat.addSlot`). -/
  | addSlot
  /-- A new CTA's shared memory of `size` bytes (`Flat.addCta`). -/
  | addCta (size : Nat)

/-- The machine, the events performed so far in perform order, each thread's next position in
program order, and the first position of each in-flight request's lanes. -/
structure Ghost where
  m : Flat
  evs : List Ev
  ctr : Nat → Nat
  base : Nat → Nat → Nat

/-- Thread `t` is one of lanes `lanes` of `who`. -/
def isLane (who : Issuer) (lanes : List Nat) (t : Nat) : Bool := lanes.any (who.tid + · == t)

def bump (ctr : Nat → Nat) (who : Issuer) (lanes : List Nat) : Nat → Nat :=
  fun t => if isLane who lanes t then ctr t + 1 else ctr t

def Ghost.syncEvents (g : Ghost) (who : Issuer) (lanes : List Nat) (mk : Nat → Nat → Ev) : List Ev :=
  lanes.map fun l => mk (who.tid + l) (g.ctr (who.tid + l))

namespace Ghost

def init (m : Flat) : Ghost := { m, evs := [], ctr := fun _ => 0, base := fun _ _ => 0 }

def step (g : Ghost) : Op → Option Ghost
  | .issue who r cta => match g.m.issue who r cta with
    | .ok (m, id) => some
      { m, evs := g.evs,
        ctr := fun t => if who.tid ≤ t ∧ r.active (t - who.tid) = true then g.ctr t + r.pieces.length
          else g.ctr t,
        base := fun i l => if i = id then g.ctr (who.tid + l) else g.base i l }
    | .error _ => none
  | .deliver slot id lane => match g.m.find slot id, g.m.deliver slot id lane with
    | some (_, q), .ok (m, bytes) => (loc q.cta q.req lane).map fun l =>
      { g with m, evs := g.evs ++ q.req.laneEvents lane (q.who.tid + lane) (g.base id lane) l bytes }
    | _, _ => none
  | .fence who lanes op => if !isFenceOp op then none else match g.m.fence who.slot with
    | .ok m => some { g with m, evs := g.evs ++ g.syncEvents who lanes (fenceEvent op),
                             ctr := bump g.ctr who lanes }
    | .error _ => none
  | .arrive who lanes id b expected => match g.m.arrive who.slot b lanes.length expected with
    | .ok (m, k) =>
      let es := g.syncEvents who lanes fun t s => barEvent t s id false b k expected
      some { g with m, evs := g.evs ++ es, ctr := bump g.ctr who lanes }
    | .error _ => none
  | .wait who lanes id b k expected => match g.m.wait who.slot b k with
    | .ok m =>
      let es := g.syncEvents who lanes fun t s => barEvent t s id true b k expected
      some { g with m, evs := g.evs ++ es, ctr := bump g.ctr who lanes }
    | .error _ => none
  | .addSlot => some { g with m := g.m.addSlot.1 }
  | .addCta size => some { g with m := (g.m.addCta size).1 }

def run (g : Ghost) : List Op → Option Ghost
  | [] => some g
  | op :: ops => (g.step op).bind (run · ops)

end Ghost

/-- The run's issuers: slot `s` always issues as `W s`, and thread `t`'s requests and
synchronization go through slot `slotOf t`. -/
def Op.Ok (W : Nat → Issuer) (slotOf : Nat → Nat) : Op → Prop
  | .issue who r _ => who = W who.slot ∧ ∀ l, r.active l = true → slotOf (who.tid + l) = who.slot
  | .deliver .. | .addSlot | .addCta _ => True
  | .fence who lanes _ | .arrive who lanes .. | .wait who lanes .. =>
    who = W who.slot ∧ lanes.Nodup ∧ ∀ l ∈ lanes, slotOf (who.tid + l) = who.slot

/-- The byte `L` names, in the machine's memory. -/
def byteAt (m : Flat) (L : Loc) : UInt8 :=
  match m.store? L with
  | some s => s.readByte (L.addr - s.base.toNat)
  | none => 0

/-- The initial memory of a run. -/
def initOf (m : Flat) : Loc → UInt8 := byteAt m


/-! ## The invariant -/

/-- Lane `lane` of in-flight `p` touches the bytes of event `e`. -/
def touches (e : Ev) (p : InFlight) (lane : Nat) : Bool :=
  match loc p.cta p.req lane with
  | some x => x.space == e.loc.space && x.owner == e.loc.owner && x.addr < e.loc.addr + e.size &&
      e.loc.addr < x.addr + p.req.bytes
  | none => false

/-- Byte `L` lies in a store of the machine. -/
def InBounds (m : Flat) (L : Loc) : Prop :=
  ∃ s, m.store? L = some s ∧ s.base.toNat ≤ L.addr ∧ L.addr - s.base.toNat < s.size

section
variable (init : Loc → UInt8) (W : Nat → Issuer) (slotOf : Nat → Nat)

structure Inv (g : Ghost) : Prop where
  perf : Performed g.evs init
  seq_lt : ∀ e ∈ g.evs, e.seq < g.ctr e.e.thread
  mem_size : ∀ e ∈ g.evs, e.e.isMem = true → 0 < e.size ∧ (e.e.isWrite = true → e.val.length = e.size)
  queued : ∀ s, ∀ q ∈ g.m.queue s, q.who = W s ∧ q.id < g.m.next ∧ q.lanes.Nodup ∧
    [1, 2, 4, 8, 16].contains q.req.bytes = true ∧
    ∀ l ∈ q.lanes, q.req.active l = true ∧ slotOf (q.who.tid + l) = s ∧
      g.base q.id l + q.req.pieces.length ≤ g.ctr (q.who.tid + l) ∧
      ∀ e ∈ g.evs, e.e.thread = q.who.tid + l →
        e.seq < g.base q.id l ∨ g.base q.id l + q.req.pieces.length ≤ e.seq
  /-- A queue is in issue order: distinct requests, and a thread's positions increase along it. -/
  ordered : ∀ s, (g.m.queue s).Pairwise fun p p' => p.id ≠ p'.id ∧
    ∀ l ∈ p.lanes, l ∈ p'.lanes → g.base p.id l + p.req.pieces.length ≤ g.base p'.id l
  pending : ∀ e ∈ g.evs, ∀ s, ∀ q ∈ g.m.queue s, ∀ l ∈ q.lanes, e.e.thread = q.who.tid + l →
    g.base q.id l < e.seq →
      e.e.drains = false ∧ q.req.acquires = false ∧ (e.e.isMem = true → touches e q l = false)
  store : ∀ L, InBounds g.m L →
    (∃ j : Fin g.evs.length, g.evs[j].covers L = true ∧
      (∀ j' : Fin g.evs.length, j < j' → g.evs[j'].covers L = false) ∧
      byteAt g.m L = g.evs[j].val.getD (L.addr - g.evs[j].loc.addr) 0) ∨
    ((∀ j : Fin g.evs.length, g.evs[j].covers L = false) ∧ byteAt g.m L = init L)
  bars : ∀ e ∈ g.evs, (e.arrives = true → e.inst ≤ (g.m.bar e.obj).done) ∧
    (e.waits = true → e.inst < (g.m.bar e.obj).done)
  stores : g.m.global.WF ∧ g.m.global.Fits ∧ ∀ s ∈ g.m.shared, s.WF ∧ s.Fits
  /-- Shared-memory events lie in CTAs the machine has. -/
  owners : ∀ e ∈ g.evs, e.e.isMem = true → e.loc.space = .shared → e.loc.owner < g.m.shared.size
  /-- A CTA the machine does not have yet starts zero-filled. -/
  fresh : ∀ L : Loc, L.space = .shared → g.m.shared.size ≤ L.owner → init L = 0

variable {init W slotOf}

/-- The stores of the machine: global memory, and each CTA's shared memory. -/
def StoresOk (m : Flat) : Prop := m.global.WF ∧ m.global.Fits ∧ ∀ s ∈ m.shared, s.WF ∧ s.Fits

theorem inv_init {m : Flat} (hq : ∀ s, m.queue s = []) (hs : StoresOk m) :
    Inv (initOf m) W slotOf (Ghost.init m) where
  perf := {
    sync_only := fun i => i.elim0, dist := fun i => i.elim0, po := fun i => i.elim0,
    rmw := fun i => i.elim0,
    bar := fun i => i.elim0, val := fun i => i.elim0 }
  seq_lt e h := by simp [Ghost.init] at h
  mem_size e h := by simp [Ghost.init] at h
  queued s q h := by simp [Ghost.init, hq] at h
  ordered s := by simp [Ghost.init, hq]
  pending e h := by simp [Ghost.init] at h
  store L _ := .inr ⟨fun j => j.elim0, rfl⟩
  bars e h := by simp [Ghost.init] at h
  stores := hs
  owners e h := by simp [Ghost.init] at h
  fresh L hs hle := by
    have hle' : m.shared.size ≤ L.owner := hle
    simp only [initOf, byteAt, store?, hs]
    rw [Array.getElem?_eq_none hle']

/-- The invariant holds of a fresh machine for any initial memory that agrees with it. -/
theorem inv_init_of {m : Flat} {init : Loc → UInt8} (hq : ∀ s, m.queue s = []) (hs : StoresOk m)
    (hm : ∀ L, InBounds m L → byteAt m L = init L)
    (hf : ∀ L : Loc, L.space = .shared → m.shared.size ≤ L.owner → init L = 0) :
    Inv init W slotOf (Ghost.init m) where
  perf := {
    sync_only := fun i => i.elim0, dist := fun i => i.elim0, po := fun i => i.elim0,
    rmw := fun i => i.elim0,
    bar := fun i => i.elim0, val := fun i => i.elim0 }
  seq_lt e h := by simp [Ghost.init] at h
  mem_size e h := by simp [Ghost.init] at h
  queued s q h := by simp [Ghost.init, hq] at h
  ordered s := by simp [Ghost.init, hq]
  pending e h := by simp [Ghost.init] at h
  store L hL := .inr ⟨fun j => j.elim0, hm L hL⟩
  bars e h := by simp [Ghost.init] at h
  stores := hs
  owners e h := by simp [Ghost.init] at h
  fresh := hf

theorem queue_eq {m m' : Flat} (h : m'.queues = m.queues) (s : Nat) : m'.queue s = m.queue s := by
  simp [queue, h]

theorem store?_eq {m m' : Flat} (hg : m'.global = m.global) (hs : m'.shared = m.shared) (L : Loc) :
    m'.store? L = m.store? L := by
  unfold store?; split <;> simp [hg, hs]

theorem bump_ge (ctr : Nat → Nat) (who : Issuer) (lanes : List Nat) (t : Nat) :
    ctr t ≤ bump ctr who lanes t := by
  unfold bump; split <;> omega

theorem covers_of_not_write {v : Ev} (h : v.e.isWrite = false) (L : Loc) : v.covers L = false := by
  simp [Ev.covers, h]

/-- The store invariant survives appending events that write nothing, on an unchanged memory. -/
theorem store_append {evs es : List Ev} {m m' : Flat}
    (hst : ∀ L, m'.store? L = m.store? L) (hes : ∀ f ∈ es, f.e.isWrite = false)
    (h : ∀ L, InBounds m L →
      (∃ j : Fin evs.length, evs[j].covers L = true ∧
        (∀ j' : Fin evs.length, j < j' → evs[j'].covers L = false) ∧
        byteAt m L = evs[j].val.getD (L.addr - evs[j].loc.addr) 0) ∨
      ((∀ j : Fin evs.length, evs[j].covers L = false) ∧ byteAt m L = init L)) :
    ∀ L, InBounds m' L →
      (∃ j : Fin (evs ++ es).length, (evs ++ es)[j].covers L = true ∧
        (∀ j' : Fin (evs ++ es).length, j < j' → (evs ++ es)[j'].covers L = false) ∧
        byteAt m' L = (evs ++ es)[j].val.getD (L.addr - (evs ++ es)[j].loc.addr) 0) ∨
      ((∀ j : Fin (evs ++ es).length, (evs ++ es)[j].covers L = false) ∧ byteAt m' L = init L) := by
  have hb : ∀ L, byteAt m' L = byteAt m L := fun L => by simp [byteAt, hst]
  have hib : ∀ L, InBounds m' L → InBounds m L := fun L ⟨s, h₁, h₂⟩ => ⟨s, (hst L) ▸ h₁, h₂⟩
  have newc : ∀ (j : Fin (evs ++ es).length), evs.length ≤ j.val → ∀ L, (evs ++ es)[j].covers L = false :=
    fun j hj L => by
      rcases Performed.fin_append j with ⟨h, _⟩ | ⟨h, _, e⟩
      · omega
      · rw [e]; exact covers_of_not_write (hes _ (List.getElem_mem _)) L
  intro L hL
  rw [hb]
  rcases h L (hib L hL) with ⟨j, h₁, h₂, h₃⟩ | ⟨h₁, h₂⟩
  · have ej : (evs ++ es)[j.val]'(by simp; omega) = evs[j] := List.getElem_append_left j.isLt
    refine .inl ⟨⟨j, by simp; omega⟩, by show (evs ++ es)[j.val].covers L = true; rw [ej]; exact h₁,
      fun j' hj' => ?_, by
        show _ = (evs ++ es)[j.val].val.getD (L.addr - (evs ++ es)[j.val].loc.addr) 0
        rw [ej]; exact h₃⟩
    by_cases hl : j'.val < evs.length
    · show (evs ++ es)[j'.val].covers L = false
      rw [List.getElem_append_left hl]; exact h₂ ⟨j', hl⟩ hj'
    · exact newc j' (by omega) L
  · refine .inr ⟨fun j' => ?_, h₂⟩
    by_cases hl : j'.val < evs.length
    · show (evs ++ es)[j'.val].covers L = false
      rw [List.getElem_append_left hl]; exact h₁ ⟨j', hl⟩
    · exact newc j' (by omega) L

/-- The store invariant for byte `L` survives appending events none of which covers `L`. -/
theorem store_lift {evs es : List Ev} {L : Loc} {byte : UInt8}
    (h : (∃ j : Fin evs.length, evs[j].covers L = true ∧
        (∀ j' : Fin evs.length, j < j' → evs[j'].covers L = false) ∧
        byte = evs[j].val.getD (L.addr - evs[j].loc.addr) 0) ∨
      ((∀ j : Fin evs.length, evs[j].covers L = false) ∧ byte = init L))
    (hes : ∀ (a : Nat) (ha : a < es.length), es[a].covers L = false) :
    (∃ j : Fin (evs ++ es).length, (evs ++ es)[j].covers L = true ∧
        (∀ j' : Fin (evs ++ es).length, j < j' → (evs ++ es)[j'].covers L = false) ∧
        byte = (evs ++ es)[j].val.getD (L.addr - (evs ++ es)[j].loc.addr) 0) ∨
      ((∀ j : Fin (evs ++ es).length, (evs ++ es)[j].covers L = false) ∧ byte = init L) := by
  have newc : ∀ (j : Fin (evs ++ es).length), evs.length ≤ j.val → (evs ++ es)[j].covers L = false :=
    fun j hj => by
      rcases Performed.fin_append j with ⟨h, _⟩ | ⟨h, _, e⟩
      · omega
      · rw [e]; exact hes _ h
  rcases h with ⟨j, h₁, h₂, h₃⟩ | ⟨h₁, h₂⟩
  · have ej : (evs ++ es)[j.val]'(by simp; omega) = evs[j] := List.getElem_append_left j.isLt
    refine .inl ⟨⟨j, by simp; omega⟩, by show (evs ++ es)[j.val].covers L = true; rw [ej]; exact h₁,
      fun j' hj' => ?_, by
        show _ = (evs ++ es)[j.val].val.getD (L.addr - (evs ++ es)[j.val].loc.addr) 0
        rw [ej]; exact h₃⟩
    by_cases hl : j'.val < evs.length
    · show (evs ++ es)[j'.val].covers L = false
      rw [List.getElem_append_left hl]; exact h₂ ⟨j', hl⟩ hj'
    · exact newc j' (by omega)
  · refine .inr ⟨fun j' => ?_, h₂⟩
    by_cases hl : j'.val < evs.length
    · show (evs ++ es)[j'.val].covers L = false
      rw [List.getElem_append_left hl]; exact h₁ ⟨j', hl⟩
    · exact newc j' (by omega)

/-- Synchronization events of a drained issuer keep the invariant. -/
theorem Inv.sync {g : Ghost} (hi : Inv init W slotOf g) {who : Issuer} {lanes : List Nat}
    {mk : Nat → Nat → Ev} {m' : Flat}
    (hok : who = W who.slot ∧ lanes.Nodup ∧ ∀ l ∈ lanes, slotOf (who.tid + l) = who.slot)
    (hdr : g.m.queue who.slot = [])
    (hq : m'.queues = g.m.queues) (hn : m'.next = g.m.next) (hg : m'.global = g.m.global)
    (hsh : m'.shared = g.m.shared)
    (hmk : ∀ t s, (mk t s).e.thread = t ∧ (mk t s).seq = s ∧ (mk t s).e.isMem = false ∧
      (mk t s).rmwRead = false ∧ (mk t s).after = [])
    (hkind : (∀ t s, (mk t s).arrives = false) ∨ (∀ t s, (mk t s).waits = false))
    (hbar_cross : ∀ a ∈ g.syncEvents who lanes mk, ∀ e ∈ g.evs, a.arrives = true → e.waits = true →
      a.obj = e.obj → a.inst ≠ e.inst)
    (hbars : ∀ e ∈ g.evs ++ g.syncEvents who lanes mk,
      (e.arrives = true → e.inst ≤ (m'.bar e.obj).done) ∧ (e.waits = true → e.inst < (m'.bar e.obj).done)) :
    Inv init W slotOf { g with m := m', evs := g.evs ++ g.syncEvents who lanes mk,
                               ctr := bump g.ctr who lanes } := by
  obtain ⟨hw, hnd, hsl⟩ := hok
  have hq' := queue_eq hq
  have hst := store?_eq hg hsh
  have hlen : (g.syncEvents who lanes mk).length = lanes.length := by simp [Ghost.syncEvents]
  have hel : ∀ a : Fin (g.syncEvents who lanes mk).length, (g.syncEvents who lanes mk)[a] =
      mk (who.tid + lanes[a.val]'(by omega)) (g.ctr (who.tid + lanes[a.val]'(by omega))) :=
    fun a => List.getElem_map _
  have hmem : ∀ f ∈ g.syncEvents who lanes mk, ∃ l ∈ lanes,
      f = mk (who.tid + l) (g.ctr (who.tid + l)) := fun f h => by
    simp only [Ghost.syncEvents, List.mem_map] at h
    obtain ⟨l, hl, rfl⟩ := h
    exact ⟨l, hl, rfl⟩
  have notw : ∀ f ∈ g.syncEvents who lanes mk, f.e.isWrite = false := fun f hf => by
    obtain ⟨l, _, rfl⟩ := hmem f hf
    have := (hmk (who.tid + l) (g.ctr (who.tid + l))).2.2.1
    simp only [Event.isMem, Bool.or_eq_false_iff] at this
    exact this.2
  -- No in-flight request belongs to a thread of `lanes`.
  have none_inflight : ∀ l ∈ lanes, ∀ s, ∀ q ∈ g.m.queue s, ∀ l', l' ∈ q.lanes →
      q.who.tid + l' ≠ who.tid + l := fun l hl s q hq' l' hl' e => by
    obtain ⟨_, _, _, _, h⟩ := hi.queued s q hq'
    have := (h l' hl').2.1
    rw [e, hsl l hl] at this
    subst this
    rw [hdr] at hq'; cases hq'
  have new_thread : ∀ f ∈ g.syncEvents who lanes mk, ∃ l ∈ lanes, f.e.thread = who.tid + l ∧
      f.seq = g.ctr (who.tid + l) := fun f hf => by
    obtain ⟨l, hl, rfl⟩ := hmem f hf
    exact ⟨l, hl, (hmk _ _).1, (hmk _ _).2.1⟩
  refine
    { perf := hi.perf.append ⟨?_, ?_, ?_, ?_, ?_, ?_, hbar_cross, ?_, ?_⟩
      seq_lt := ?_, mem_size := ?_, queued := ?_, ordered := ?_, pending := ?_
      store := store_append hst notw hi.store, bars := hbars, stores := ?_
      owners := fun e he hm => ?_, fresh := hsh ▸ hi.fresh }
  · intro f hf
    obtain ⟨l, _, rfl⟩ := hmem f hf
    exact (hmk _ _).2.2.2.2
  · intro a b hab ht
    rw [hel a, hel b, (hmk _ _).1, (hmk _ _).1] at ht
    exact absurd (Fin.ext ((hnd.getElem_inj).1 (by omega))) hab
  · intro e he f hf ht
    obtain ⟨l, _, hft, hfs⟩ := new_thread f hf
    rw [hfs, ← hft, ← ht]
    exact Nat.ne_of_lt (hi.seq_lt e he)
  · intro a b hab ht
    rw [hel a, hel b, (hmk _ _).1, (hmk _ _).1] at ht
    exact absurd (Fin.ext ((hnd.getElem_inj).1 (by omega))) (Fin.ne_of_lt hab)
  · intro e he f hf ht hs
    obtain ⟨l, _, hft, hfs⟩ := new_thread f hf
    rw [hfs, ← hft, ← ht] at hs
    exact absurd (hi.seq_lt e he) (Nat.not_lt.2 (Nat.le_of_lt hs))
  · intro a ha
    rw [hel a, (hmk _ _).2.2.2.1] at ha; cases ha
  · intro a b ha hb _ _
    rw [hel a] at ha; rw [hel b] at hb
    rcases hkind with h | h
    · rw [h] at ha; cases ha
    · rw [h] at hb; cases hb
  · intro a ha
    rw [hel a] at ha
    have := (hmk (who.tid + lanes[a.val]'(by omega)) (g.ctr (who.tid + lanes[a.val]'(by omega)))).2.2.1
    simp [Event.isMem, ha] at this
  · intro e he
    rcases List.mem_append.1 he with he | he
    · exact Nat.lt_of_lt_of_le (hi.seq_lt e he) (bump_ge _ _ _ _)
    · obtain ⟨l, hl, hft, hfs⟩ := new_thread e he
      show e.seq < bump g.ctr who lanes e.e.thread
      rw [hft, hfs]
      have : isLane who lanes (who.tid + l) = true := List.any_eq_true.2 ⟨l, hl, by simp⟩
      simp [bump, this]
  · intro e he hm
    rcases List.mem_append.1 he with he | he
    · exact hi.mem_size e he hm
    · obtain ⟨l, _, rfl⟩ := hmem e he
      rw [(hmk _ _).2.2.1] at hm; cases hm
  · intro s q hq₁
    rw [hq'] at hq₁
    obtain ⟨h₁, h₂, h₃, h₀, h₄⟩ := hi.queued s q hq₁
    refine ⟨h₁, hn ▸ h₂, h₃, h₀, fun l hl => ?_⟩
    obtain ⟨a₁, a₂, a₃, a₄⟩ := h₄ l hl
    refine ⟨a₁, a₂, Nat.le_trans a₃ (bump_ge _ _ _ _), fun e he ht => ?_⟩
    rcases List.mem_append.1 he with he | he
    · exact a₄ e he ht
    · obtain ⟨l', hl', hft, _⟩ := new_thread e he
      exact absurd (ht.symm.trans hft) (none_inflight l' hl' s q hq₁ l hl)
  · intro s; show (m'.queue s).Pairwise _; rw [hq']; exact hi.ordered s
  · intro e he s q hq₁ l hl ht hlt
    rw [hq'] at hq₁
    rcases List.mem_append.1 he with he | he
    · exact hi.pending e he s q hq₁ l hl ht hlt
    · obtain ⟨l', hl', hft, _⟩ := new_thread e he
      exact absurd (ht.symm.trans hft) (none_inflight l' hl' s q hq₁ l hl)
  · obtain ⟨h₁, h₂, h₃⟩ := hi.stores
    exact ⟨hg ▸ h₁, hg ▸ h₂, fun s hs => h₃ s (hsh ▸ hs)⟩
  · rcases List.mem_append.1 he with he | he
    · show _ → e.loc.owner < m'.shared.size; rw [hsh]; exact hi.owners e he hm
    · obtain ⟨l, _, rfl⟩ := hmem e he
      rw [(hmk _ _).2.2.1] at hm; cases hm

/-- The in-flight request an issue appends. -/
def issued (m : Flat) (who : Issuer) (r : Req) (cta : Nat) : InFlight :=
  { id := m.next, who, cta, req := r,
    lanes := (List.range r.addrs.size).filter fun l => (r.addrs[l]?).join.isSome }

/-- The machine after issuing `r`. -/
def afterIssue (m : Flat) (who : Issuer) (r : Req) (cta : Nat) : Flat :=
  { m with queues := m.queues.modify who.slot (· ++ [issued m who r cta]), next := m.next + 1 }

theorem queue_issue {m : Flat} {who : Issuer} (h : who.slot < m.queues.size) (r : Req) (cta s : Nat) :
    (afterIssue m who r cta).queue s =
      if s = who.slot then m.queue s ++ [issued m who r cta] else m.queue s := by
  simp only [afterIssue, queue, Array.getElem?_modify]
  by_cases hs : s = who.slot
  · subst hs; simp [h, Array.getElem?_eq_getElem h]
  · simp [Ne.symm hs, hs]

theorem issue_ok {m m' : Flat} {who : Issuer} {r : Req} {cta id : Nat}
    (h : m.issue who r cta = .ok (m', id)) :
    who.slot < m.queues.size ∧ [1, 2, 4, 8, 16].contains r.bytes = true ∧ id = m.next ∧
      m' = afterIssue m who r cta := by
  unfold Flat.issue at h
  split at h
  · cases h
  · next h₁ =>
    split at h
    · cases h
    · next h₂ =>
      simp only [Except.ok.injEq, Prod.mk.injEq] at h
      simp only [Bool.not_eq_true', decide_eq_false_iff_not, Classical.not_not] at h₁
      simp only [Bool.not_eq_true', Bool.not_eq_false] at h₂
      exact ⟨h₁, h₂, h.2.symm, h.1.symm⟩

theorem Inv.issue {g g' : Ghost} (hi : Inv init W slotOf g) {who : Issuer} {r : Req} {cta : Nat}
    (hok : who = W who.slot ∧ ∀ l, r.active l = true → slotOf (who.tid + l) = who.slot)
    (hs : g.step (.issue who r cta) = some g') : Inv init W slotOf g' := by
  simp only [Ghost.step] at hs
  split at hs
  · next m id heq =>
    obtain ⟨hsl, hbytes, rfl, rfl⟩ := issue_ok heq
    simp only [Option.some.injEq] at hs
    subst hs
    obtain ⟨q, hq⟩ : ∃ q, q = issued g.m who r cta := ⟨_, rfl⟩
    have hqid : q.id = g.m.next := by rw [hq]; rfl
    have hqwho : q.who = who := by rw [hq]; rfl
    have hqreq : q.req = r := by rw [hq]; rfl
    have hqs : ∀ s, (afterIssue g.m who r cta).queue s =
        if s = who.slot then g.m.queue s ++ [q] else g.m.queue s := fun s => by
      rw [hq]; exact queue_issue hsl r cta s
    -- Old requests keep their positions; the new one starts at its threads' next positions.
    have old_id : ∀ s, ∀ p ∈ g.m.queue s, p.id ≠ g.m.next := fun s p hp e =>
      Nat.lt_irrefl _ (e ▸ (hi.queued s p hp).2.1)
    have hctr : ∀ t, g.ctr t ≤ (if who.tid ≤ t ∧ r.active (t - who.tid) = true then
        g.ctr t + r.pieces.length else g.ctr t) := fun t => by split <;> omega
    have hlanes : ∀ l ∈ q.lanes, r.active l = true := fun l hl => by
      simp only [hq, issued, List.mem_filter, List.mem_range] at hl
      exact hl.2
    have hctr_new : ∀ l ∈ q.lanes, (if who.tid ≤ who.tid + l ∧ r.active (who.tid + l - who.tid) = true
        then g.ctr (who.tid + l) + r.pieces.length else g.ctr (who.tid + l)) =
          g.ctr (who.tid + l) + r.pieces.length := fun l hl => by
      rw [if_pos ⟨by omega, by rw [Nat.add_sub_cancel_left]; exact hlanes l hl⟩]
    have hmem : ∀ s p, p ∈ (afterIssue g.m who r cta).queue s →
        p ∈ g.m.queue s ∨ (s = who.slot ∧ p = q) := by
      intro s p hp
      rw [hqs] at hp
      split at hp
      · rcases List.mem_append.1 hp with hp | hp
        · exact .inl hp
        · exact .inr ⟨‹_›, List.mem_singleton.1 hp⟩
      · exact .inl hp
    refine
      { perf := hi.perf, seq_lt := fun e he => Nat.lt_of_lt_of_le (hi.seq_lt e he) (hctr _)
        mem_size := hi.mem_size, queued := ?_, ordered := ?_, pending := ?_
        store := hi.store, bars := hi.bars, stores := hi.stores, owners := hi.owners,
        fresh := hi.fresh }
    · intro s p hp
      rcases hmem s p hp with hp | ⟨rfl, hpq⟩
      · obtain ⟨h₁, h₂, h₃, h₀, h₄⟩ := hi.queued s p hp
        refine ⟨h₁, show p.id < g.m.next + 1 from Nat.lt_succ_of_lt h₂, h₃, h₀, fun l hl => ?_⟩
        obtain ⟨a₁, a₂, a₃, a₄⟩ := h₄ l hl
        simp only [if_neg (old_id s p hp)]
        exact ⟨a₁, a₂, Nat.le_trans a₃ (hctr _), a₄⟩
      · rw [hpq]
        refine ⟨hqwho.trans hok.1, show q.id < g.m.next + 1 by rw [hqid]; exact Nat.lt_succ_self _,
          ?_, by rw [hqreq]; exact hbytes, fun l hl => ?_⟩
        · rw [hq]; exact List.nodup_range.sublist (List.filter_sublist)
        · simp only [if_pos hqid, hqwho, hqreq, hctr_new l hl]
          refine ⟨hlanes l hl, hok.2 l (hlanes l hl), Nat.le_refl _, fun e he ht => .inl ?_⟩
          rw [← ht]; exact hi.seq_lt e he
    · intro s
      have old : ∀ s, (g.m.queue s).Pairwise fun p p' => p.id ≠ p'.id ∧
          ∀ l ∈ p.lanes, l ∈ p'.lanes →
            (if p.id = g.m.next then g.ctr (who.tid + l) else g.base p.id l) + p.req.pieces.length ≤
              (if p'.id = g.m.next then g.ctr (who.tid + l) else g.base p'.id l) := fun s =>
        (hi.ordered s).imp_of_mem fun ha hb h => by
          simp only [if_neg (old_id _ _ ha), if_neg (old_id _ _ hb)]; exact h
      show (Flat.queue _ s).Pairwise _
      rw [hqs]
      split
      · next hs =>
        subst hs
        rw [List.pairwise_append]
        refine ⟨old _, List.pairwise_singleton _ _, fun a ha b hb => ?_⟩
        rw [List.mem_singleton] at hb
        subst hb
        refine ⟨fun e => old_id _ _ ha (e.trans hqid), fun l h₁ _ => ?_⟩
        simp only [if_neg (old_id _ _ ha), if_pos hqid]
        obtain ⟨hw, -, -, -, h₄⟩ := hi.queued _ _ ha
        have := (h₄ l h₁).2.2.1
        rw [hw, ← hok.1] at this
        exact this
      · exact old s
    · intro e he s p hp l hl ht hlt
      rcases hmem s p hp with hp | ⟨rfl, hpq⟩
      · simp only [if_neg (old_id s p hp)] at hlt
        exact hi.pending e he s p hp l hl ht hlt
      · rw [hpq] at hlt ht
        exfalso
        simp only [if_pos hqid] at hlt
        have := hi.seq_lt e he
        rw [ht, hqwho] at this
        omega
  · cases hs

/-! ## Delivery -/

theorem perform_ok {r : Req} {lane : Nat} {l : Loc} {st st' : Store} {bytes : List UInt8}
    (h : perform r lane l st = .ok (st', bytes)) :
    st.contains (UInt64.ofNat l.addr) r.bytes = true ∧
      bytes = readPart r (l.addr - st.base.toNat) st ∧
      performWrites r lane (l.addr - st.base.toNat) bytes (r.pieces.filter (·.write)) st = .ok st' := by
  unfold perform at h
  split at h
  · cases h
  · next hc =>
    simp only [Bool.not_eq_true', Bool.not_eq_false] at hc
    dsimp only at h
    cases hw : performWrites r lane (l.addr - st.base.toNat) (readPart r (l.addr - st.base.toNat) st)
        (r.pieces.filter (·.write)) st with
    | error e => rw [hw] at h; cases h
    | ok s =>
      rw [hw] at h
      simp only [Except.map, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨hc, rfl, hw⟩

theorem store?_some {m : Flat} {l : Loc} {st : Store} (h : m.store? l = some st) :
    (l.space = .global ∧ l.owner = 0 ∧ st = m.global) ∨
      (l.space = .shared ∧ ∃ h : l.owner < m.shared.size, st = m.shared[l.owner]) := by
  unfold store? at h
  split at h
  · next hs =>
    split at h
    · next ho => simp only [Option.some.injEq] at h; exact .inl ⟨hs, ho, h.symm⟩
    · cases h
  · next hs =>
    rw [Array.getElem?_eq_some_iff] at h
    obtain ⟨hlt, he⟩ := h
    exact .inr ⟨hs, hlt, he.symm⟩
  · cases h

theorem commit_eq {m : Flat} {l : Loc} {st : Store} (h : m.store? l = some st) (slot : Nat)
    (s' : Store) (x : List InFlight) :
    { ({ m.setStore l default with queues := (m.setStore l default).queues.set! slot [] } : Flat).setStore
        l s' with
      queues := (m.setStore l default).queues.set! slot [] |>.set! slot x } =
      { m.setStore l s' with queues := m.queues.set! slot x } := by
  rcases store?_some h with ⟨hs, _, _⟩ | ⟨hs, _, _⟩ <;>
    simp [setStore, hs, Array.set!_eq_setIfInBounds, Array.setIfInBounds_setIfInBounds]

theorem deliver_facts {g g' : Ghost} {slot id lane : Nat}
    (hs : g.step (.deliver slot id lane) = some g') :
    ∃ (i : Nat) (hi : i < (g.m.queue slot).length) (l : Loc) (st st' : Store) (bytes : List UInt8),
      (g.m.queue slot)[i].id = id ∧ lane ∈ (g.m.queue slot)[i].lanes ∧
      (∀ p ∈ (g.m.queue slot).take i, blocker p (g.m.queue slot)[i] lane = none) ∧
      loc (g.m.queue slot)[i].cta (g.m.queue slot)[i].req lane = some l ∧ g.m.store? l = some st ∧
      st.contains (UInt64.ofNat l.addr) (g.m.queue slot)[i].req.bytes = true ∧
      bytes = readPart (g.m.queue slot)[i].req (l.addr - st.base.toNat) st ∧
      performWrites (g.m.queue slot)[i].req lane (l.addr - st.base.toNat) bytes
        ((g.m.queue slot)[i].req.pieces.filter (·.write)) st = .ok st' ∧
      g.m.deliver slot id lane = .ok ({ g.m.setStore l st' with
        queues := g.m.queues.set! slot (retire (g.m.queue slot) id lane) }, bytes) ∧
      g' = { g with
        m := { g.m.setStore l st' with queues := g.m.queues.set! slot (retire (g.m.queue slot) id lane) }
        evs := g.evs ++ (g.m.queue slot)[i].req.laneEvents lane ((g.m.queue slot)[i].who.tid + lane)
          (g.base id lane) l bytes } := by
  simp only [Ghost.step] at hs
  split at hs
  · next i q m bytes hf hd =>
    have hf₀ := hf
    have hd₀ := hd
    simp only [Flat.find, Option.map_eq_some_iff, Prod.mk.injEq] at hf
    obtain ⟨i', hi', rfl, hq⟩ := hf
    obtain ⟨hlt, hid, -⟩ := List.findIdx?_eq_some_iff_getElem.1 hi'
    rw [getElem!_pos (g.m.queue slot) i' hlt] at hq
    subst hq
    simp only [beq_iff_eq] at hid
    simp only [Flat.deliver, hf₀, Flat.deliverAt] at hd
    split at hd
    · cases hd
    · next hc =>
      simp only [Bool.not_eq_true', Bool.not_eq_false, List.contains_iff_mem] at hc
      split at hd
      · cases hd
      · next hb =>
        rw [List.findSome?_eq_none_iff] at hb
        split at hd
        · cases hd
        · next l hl =>
          split at hd
          · cases hd
          · next st hst =>
            split at hd
            · cases hd
            · next st' bytes' hp =>
              simp only [Except.ok.injEq, Prod.mk.injEq] at hd
              obtain ⟨hm, rfl⟩ := hd
              rw [commit_eq hst] at hm
              subst hm
              obtain ⟨hc', hb', hw⟩ := perform_ok hp
              rw [hl] at hs
              simp only [Option.map_some, Option.some.injEq] at hs
              refine ⟨i', hlt, l, st, st', bytes', hid, hc, hb, hl, hst, hc', hb', hw, ?_, ?_⟩
              · rw [hid] at hd₀; exact hd₀
              · rw [← hs, hid]
  · cases hs

theorem readBytes_extract (st : Store) (off n a k : Nat) (h : a + k ≤ n) :
    (readBytes st off n).extract a (a + k) = readBytes st (off + a) k := by
  apply List.ext_getElem
  · simp [readBytes]; omega
  · intro j h₁ h₂
    simp [readBytes, Nat.add_assoc]

theorem readPart_extract {r : Req} (hp : PiecesOk r) {p : Piece} (hpm : p ∈ r.pieces)
    (hr : p.write = false) (off : Nat) (st : Store) :
    (readPart r off st).extract p.off (p.off + p.size) = readBytes st (off + p.off) p.size := by
  have : r.pieces.any (!·.write) = true := List.any_eq_true.2 ⟨p, hpm, by simp [hr]⟩
  rw [readPart, if_pos this]
  exact readBytes_extract st off r.bytes p.off p.size (hp.inside p hpm)

theorem retireOne_some {id lane : Nat} {a b : InFlight} (h : retireOne id lane a = some b) :
    b.id = a.id ∧ b.who = a.who ∧ b.req = a.req ∧ (∀ l ∈ b.lanes, l ∈ a.lanes) ∧
      (a.id = id → a.lanes.Nodup → lane ∉ b.lanes) ∧ (a.lanes.Nodup → b.lanes.Nodup) ∧
      b.cta = a.cta := by
  unfold retireOne at h
  split at h
  · split at h
    · cases h
    · simp only [Option.some.injEq] at h
      subst h
      exact ⟨rfl, rfl, rfl, fun l hl => List.mem_of_mem_erase hl,
        fun _ hnd hl => ((hnd.mem_erase_iff).1 hl).1 rfl, fun hnd => hnd.erase _, rfl⟩
  · next he =>
    simp only [Option.some.injEq] at h
    subst h
    simp only [beq_iff_eq] at he
    exact ⟨rfl, rfl, rfl, fun l hl => hl, fun e => absurd e he, fun h => h, rfl⟩

theorem mem_retire {qs : List InFlight} {id lane : Nat} {p : InFlight} (h : p ∈ retire qs id lane) :
    ∃ p₀ ∈ qs, p.id = p₀.id ∧ p.who = p₀.who ∧ p.req = p₀.req ∧ (∀ l ∈ p.lanes, l ∈ p₀.lanes) ∧
      (p₀.id = id → p₀.lanes.Nodup → lane ∉ p.lanes) ∧ (p₀.lanes.Nodup → p.lanes.Nodup) ∧
      p.cta = p₀.cta := by
  simp only [retire, List.mem_filterMap] at h
  obtain ⟨p₀, hp₀, hf⟩ := h
  exact ⟨p₀, hp₀, retireOne_some hf⟩

theorem pairwise_retire {qs : List InFlight} {id lane : Nat} {R : InFlight → InFlight → Prop}
    (hR : ∀ a a' b b', R a a' → b.id = a.id → b.req = a.req → (∀ l ∈ b.lanes, l ∈ a.lanes) →
      b'.id = a'.id → b'.req = a'.req → (∀ l ∈ b'.lanes, l ∈ a'.lanes) → R b b')
    (h : qs.Pairwise R) : (retire qs id lane).Pairwise R := by
  apply List.Pairwise.filterMap _ _ h
  intro a a' haa b hb b' hb'
  obtain ⟨e₁, -, e₂, e₃, -⟩ := retireOne_some hb
  obtain ⟨e₄, -, e₅, e₆, -⟩ := retireOne_some hb'
  exact hR a a' b b' haa e₁ e₂ e₃ e₄ e₅ e₆

theorem pairwise_either {α : Type} {R : α → α → Prop} {l : List α} (h : l.Pairwise R) {a b : α}
    (ha : a ∈ l) (hb : b ∈ l) (hne : a ≠ b) : R a b ∨ R b a := by
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 ha
  obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 hb
  have hij : i ≠ j := fun e => hne (by subst e; rfl)
  rcases Nat.lt_or_gt_of_ne hij with c | c
  · exact .inl (List.pairwise_iff_getElem.1 h i j hi hj c)
  · exact .inr (List.pairwise_iff_getElem.1 h j i hj hi c)

/-- An element not related from position `i` lies before it. -/
theorem mem_take_of_pairwise {α : Type} {R : α → α → Prop} {l : List α} (h : l.Pairwise R) {i : Nat}
    (hi : i < l.length) {a : α} (ha : a ∈ l) (hne : a ≠ l[i]) (hn : ¬ R l[i] a) : a ∈ l.take i := by
  obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 ha
  have hij : j ≠ i := fun e => hne (by subst e; rfl)
  rcases Nat.lt_or_gt_of_ne hij with c | c
  · exact List.mem_iff_getElem.2 ⟨j, by simp; omega, by simp⟩
  · exact absurd (List.pairwise_iff_getElem.1 h i j hi hj c) hn

theorem loc_addr_lt {cta : Nat} {r : Req} {lane : Nat} {l : Loc} (h : loc cta r lane = some l) :
    l.addr < 2 ^ 64 := by
  simp only [loc, Option.map_eq_some_iff] at h
  obtain ⟨a, _, rfl⟩ := h
  exact a.toNat_lt

theorem store?_wf {m : Flat} (hm : StoresOk m) {l : Loc} {st : Store} (h : m.store? l = some st) :
    st.WF ∧ st.Fits := by
  rcases store?_some h with ⟨_, _, rfl⟩ | ⟨_, hlt, rfl⟩
  · exact ⟨hm.1, hm.2.1⟩
  · exact hm.2.2 _ (Array.getElem_mem hlt)

theorem store?_setStore {m : Flat} {l : Loc} {st st' : Store} (h : m.store? l = some st) (L : Loc) :
    (m.setStore l st').store? L =
      if L.space = l.space ∧ L.owner = l.owner then some st' else m.store? L := by
  rcases store?_some h with ⟨hs, ho, _⟩ | ⟨hs, hlt, _⟩
  · simp only [setStore, hs]
    unfold store?
    split
    · next hL =>
      by_cases hLo : L.owner = 0
      · simp [hL, hLo, ho]
      · simp [hL, hLo, ho]
    · next hL => simp [hL]
    · next hL₁ hL₂ =>
      have : L.space ≠ .global := hL₁
      simp [this]
  · simp only [setStore, hs]
    unfold store?
    split
    · next hL => simp [hL]
    · next hL =>
      by_cases hLo : L.owner = l.owner
      · simp [hL, hLo, Array.getElem?_setIfInBounds, hlt]
      · simp [hL, hLo, Array.getElem?_setIfInBounds, Ne.symm hLo]
    · next hL₁ hL₂ =>
      have : L.space ≠ .shared := hL₂
      simp [this]

theorem queue_set {m : Flat} {slot : Nat} (x : List InFlight) (hs : slot < m.queues.size) (s : Nat) :
    ({ m with queues := m.queues.set! slot x } : Flat).queue s = if s = slot then x else m.queue s := by
  simp only [queue, Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds]
  by_cases h : s = slot
  · subst h; simp [hs]
  · have : ¬ slot = s := fun e => h e.symm
    simp [this, h]

theorem slot_lt {m : Flat} {slot : Nat} {q : InFlight} (h : q ∈ m.queue slot) : slot < m.queues.size := by
  unfold queue at h
  cases hq : m.queues[slot]? with
  | none => rw [hq] at h; simp at h
  | some _ => exact (Array.getElem?_eq_some_iff.1 hq).1

/-- What each event of a delivered lane is. -/
theorem laneEvent_props {r : Req} (hP : PiecesOk r) (lane t base : Nat) (l : Loc) (bytes : List UInt8)
    (a : Nat) (ha : a < (r.laneEvents lane t base l bytes).length) :
    let f := (r.laneEvents lane t base l bytes)[a]
    let p := r.pieces[a]'(by rwa [laneEvents_length] at ha)
    f.e.thread = t ∧ f.seq = base + a ∧ f.e.isMem = true ∧ 0 < f.size ∧
      f.loc = { l with addr := l.addr + p.off } ∧ f.size = p.size ∧
      f.e.isWrite = p.write ∧ f.e.isRead = !p.write ∧ f.rmwRead = p.rmwRead ∧
      f.e.drains = (p.write && r.releases) ∧ f.e.blocks = (!p.write && r.acquires) ∧
      f.arrives = false ∧ f.waits = false ∧ f.e.scope = r.eventScope ∧
      f.val = (if p.write then (r.written lane p bytes).getD [] else bytes.extract p.off (p.off + p.size)) := by
  intro f p
  have ha' : a < r.pieces.length := by rwa [laneEvents_length] at ha
  have hpm : p ∈ r.pieces := List.getElem_mem _
  have hpos := hP.size_pos p hpm
  simp only [f, p, laneEvents_getElem]
  cases hw : (r.pieces[a]'ha').write <;>
    simp [Req.event, hw, Event.isMem, Event.isRead, Event.isWrite, Event.drains, Event.blocks,
      Event.isReleaseFence, Event.isAcquireFence, Ev.arrives, Ev.waits] <;>
    exact hpos

theorem store?_congr (m : Flat) {L l : Loc} (h₁ : L.space = l.space) (h₂ : L.owner = l.owner) :
    m.store? L = m.store? l := by
  simp only [store?, h₁, h₂]

theorem map_getD_range (v : List UInt8) : (List.range v.length).map (fun i => v.getD i 0) = v := by
  apply List.ext_getElem
  · simp
  · intro j h₁ h₂; simp [List.getElem?_eq_getElem h₂]

theorem setStore_fields (m : Flat) (l : Loc) (s : Store) :
    (m.setStore l s).next = m.next ∧ (m.setStore l s).bars = m.bars ∧ (m.setStore l s).queues = m.queues := by
  unfold setStore; split <;> exact ⟨rfl, rfl, rfl⟩

theorem setStore_shared_size (m : Flat) (l : Loc) (s : Store) :
    (m.setStore l s).shared.size = m.shared.size := by
  unfold setStore; split <;> simp

theorem storesOk_setStore {m : Flat} (hm : StoresOk m) {l : Loc} {st st' : Store}
    (h : m.store? l = some st) (hw : st'.WF) (hf : st'.Fits) : StoresOk (m.setStore l st') := by
  rcases store?_some h with ⟨hs, _, _⟩ | ⟨hs, hlt, _⟩
  · simp only [setStore, hs]
    exact ⟨hw, hf, hm.2.2⟩
  · simp only [setStore, hs]
    refine ⟨hm.1, hm.2.1, fun x hx => ?_⟩
    simp only [Array.set!_eq_setIfInBounds] at hx
    rcases Array.mem_or_eq_of_mem_setIfInBounds hx with hx | rfl
    · exact hm.2.2 x hx
    · exact ⟨hw, hf⟩

theorem getD_of_lt {α : Type} {l : List α} {i : Nat} (h : i < l.length) (a : α) : l.getD i a = l[i] := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]; rfl

theorem isWriteTo_iff {v : Ev} {k : Loc × Nat} : v.isWriteTo k = true ↔ v.e.isWrite = true ∧ v.key = k := by
  simp [Ev.isWriteTo]

theorem touches_iff {e : Ev} {p : InFlight} {lane : Nat} {x : Loc} (h : loc p.cta p.req lane = some x) :
    touches e p lane = true ↔ x.space = e.loc.space ∧ x.owner = e.loc.owner ∧
      x.addr < e.loc.addr + e.size ∧ e.loc.addr < x.addr + p.req.bytes := by
  simp [touches, h, and_assoc]

theorem overlap_iff {a b : InFlight} {lane : Nat} {x y : Loc} (ha : loc a.cta a.req lane = some x)
    (hb : loc b.cta b.req lane = some y) :
    overlap a b lane = true ↔ x.space = y.space ∧ x.owner = y.owner ∧
      x.addr < y.addr + b.req.bytes ∧ y.addr < x.addr + a.req.bytes := by
  simp [overlap, ha, hb, and_assoc]

/-- **A delivery keeps the invariant.** -/
theorem Inv.deliver {g g' : Ghost} (hi : Inv init W slotOf g) {slot id lane : Nat}
    (hs : g.step (.deliver slot id lane) = some g') (hu : Uniform g'.evs) :
    Inv init W slotOf g' := by
  obtain ⟨i, hlt, l, st, st', bytes, hid, hlane, hblk, hl, hst, hc, hb, hw, -, rfl⟩ := deliver_facts hs
  obtain ⟨q, hq⟩ : ∃ q, (g.m.queue slot)[i] = q := ⟨_, rfl⟩
  have hqin : q ∈ g.m.queue slot := hq ▸ List.getElem_mem hlt
  simp only [hq] at hid hlane hblk hl hc hb hw hu ⊢
  subst hid
  obtain ⟨hW, hnext, hnd, hbytes, hL⟩ := hi.queued slot q hqin
  obtain ⟨hact, hslot, hctr, hrange⟩ := hL lane hlane
  have hP := pieces_ok hbytes
  have hslt := slot_lt hqin
  obtain ⟨hbase, hfit⟩ := Store.contains_nat (loc_addr_lt hl) hc
  obtain ⟨swf, sfit⟩ := store?_wf hi.stores hst
  obtain ⟨w₁, f₁, b₁, z₁, e₁, u₁, c₁⟩ := performWrites_ok _ st st' hw swf sfit
    (fun p hp => by
      have := hP.inside p (List.mem_filter.1 hp).1
      omega) hP.writes
  -- Names for the delivered lane.
  generalize ht : q.who.tid + lane = t at *
  generalize hbs : g.base q.id lane = b at *
  generalize hE : q.req.laneEvents lane t b l bytes = E at *
  have hEl : E.length = q.req.pieces.length := by rw [← hE, laneEvents_length]
  have props := fun (a : Nat) (ha : a < E.length) => by
    have := laneEvent_props hP lane t b l bytes a (by rw [hE]; exact ha)
    simp only [hE] at this
    exact this
  have hmemE : ∀ f ∈ E, ∃ a, ∃ h : a < E.length, E[a] = f := fun f hf => by
    obtain ⟨a, h, e⟩ := List.mem_iff_getElem.1 hf; exact ⟨a, h, e⟩
  have hpiece : ∀ a (h : a < E.length), (q.req.pieces[a]'(hEl ▸ h)) ∈ q.req.pieces :=
    fun a h => List.getElem_mem _
  -- The machine after the delivery.
  obtain ⟨hnx, hbr, -⟩ := setStore_fields g.m l st'
  have hq' : ∀ s, Flat.queue { g.m.setStore l st' with
      queues := g.m.queues.set! slot (retire (g.m.queue slot) q.id lane) } s =
      if s = slot then retire (g.m.queue slot) q.id lane else g.m.queue s := fun s => by
    have := queue_set (m := g.m) (retire (g.m.queue slot) q.id lane) hslt s
    simp only [queue] at this ⊢
    exact this
  have hst' : ∀ L, Flat.store? { g.m.setStore l st' with
      queues := g.m.queues.set! slot (retire (g.m.queue slot) q.id lane) } L =
      (g.m.setStore l st').store? L := fun L => by simp only [store?]
  have hbars' : ∀ b', Flat.bar { g.m.setStore l st' with
      queues := g.m.queues.set! slot (retire (g.m.queue slot) q.id lane) } b' = g.m.bar b' :=
    fun b' => by simp only [bar]; rw [hbr]
  -- Requests in flight after the delivery.
  have hin' : ∀ s p', p' ∈ Flat.queue { g.m.setStore l st' with
      queues := g.m.queues.set! slot (retire (g.m.queue slot) q.id lane) } s →
      ∃ p₀ ∈ g.m.queue s, p'.id = p₀.id ∧ p'.who = p₀.who ∧ p'.req = p₀.req ∧
        (∀ l ∈ p'.lanes, l ∈ p₀.lanes) ∧ (p₀ = q → lane ∉ p'.lanes) ∧
        (p₀.lanes.Nodup → p'.lanes.Nodup) ∧ p'.cta = p₀.cta := by
    intro s p' hp'
    rw [hq'] at hp'
    split at hp'
    · next hseq =>
      subst hseq
      obtain ⟨p₀, h₀, e₁, e₂, e₃, e₄, e₅, e₆, e₇⟩ := mem_retire hp'
      exact ⟨p₀, h₀, e₁, e₂, e₃, e₄, fun hpq => e₅ (hpq ▸ rfl) (hpq ▸ hnd), e₆, e₇⟩
    · next hsne =>
      refine ⟨p', hp', rfl, rfl, rfl, fun _ h => h, fun hpq => ?_, id, rfl⟩
      exfalso
      rw [hpq] at hp'
      obtain ⟨-, -, -, -, h⟩ := hi.queued s q hp'
      have h1 := (h lane hlane).2.1
      rw [ht, hslot] at h1
      exact hsne h1.symm
  -- An in-flight lane of thread `t` other than the delivered one is in the same queue.
  have same_slot : ∀ s, ∀ p ∈ g.m.queue s, ∀ l' ∈ p.lanes, p.who.tid + l' = t →
      s = slot ∧ l' = lane := fun s p hp l' hl' hpt => by
    obtain ⟨hpw, -, -, -, h⟩ := hi.queued s p hp
    have h1 := (h l' hl').2.1
    rw [hpt, hslot] at h1
    subst h1
    rw [hpw, ← hW] at hpt
    exact ⟨rfl, by omega⟩
  -- The new events touch only the delivered lane's bytes.
  have new_inside : ∀ f ∈ E, f.loc.space = l.space ∧ f.loc.owner = l.owner ∧ l.addr ≤ f.loc.addr ∧
      f.loc.addr + f.size ≤ l.addr + q.req.bytes := fun f hf => by
    obtain ⟨a, ha, rfl⟩ := hmemE f hf
    obtain ⟨-, -, -, -, hloc, hsz, -⟩ := props a ha
    have := hP.inside _ (hpiece a ha)
    rw [hloc, hsz]
    exact ⟨rfl, rfl, by simp, by simp; omega⟩
  refine
    { perf := hi.perf.append ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
      seq_lt := ?_, mem_size := ?_, queued := ?_, ordered := ?_, pending := ?_
      store := ?_, bars := ?_, stores := ?_, owners := ?_, fresh := ?_ }
  · intro f hf
    rw [← hE] at hf
    simp only [Req.laneEvents, List.mem_map] at hf
    obtain ⟨_, _, rfl⟩ := hf
    rfl
  · -- dist_new
    rintro ⟨a, ha⟩ ⟨a', ha'⟩ hne _
    simp only [Fin.getElem_fin]
    rw [(props a ha).2.1, (props a' ha').2.1]
    exact fun e => hne (Fin.ext (by simp only; omega))
  · -- dist_cross
    intro e he f hf hth
    obtain ⟨a, ha, rfl⟩ := hmemE f hf
    obtain ⟨hft, hfs, -⟩ := props a ha
    rw [hft] at hth
    rcases hrange e he hth with c | c <;> rw [hfs] <;> omega
  · -- po_new
    rintro ⟨a, ha⟩ ⟨a', ha'⟩ hlt' _
    simp only [Fin.getElem_fin]
    rw [(props a ha).2.1, (props a' ha').2.1]
    have : a < a' := hlt'
    omega
  · -- po_cross
    intro e he f hf hth hlt'
    obtain ⟨a, ha, rfl⟩ := hmemE f hf
    obtain ⟨hft, hfs, hfm, hfpos, hloc, hsz, -, -, -, -, hfb, -⟩ := props a ha
    rw [hft] at hth
    have hbe : g.base q.id lane < e.seq := by rw [hbs]; omega
    obtain ⟨hd, hacq, htouch⟩ := hi.pending e he slot q hqin lane hlane (by rw [ht]; exact hth) hbe
    refine ⟨fun ⟨hem, _, hk⟩ => ?_, hd, by rw [hfb, hacq]; simp⟩
    have h₁ := htouch hem
    simp only [Ev.key, Prod.mk.injEq] at hk
    have hin := hP.inside _ (hpiece a ha)
    have : touches e q lane = true := by
      rw [touches_iff hl, hk.1, hk.2, hloc, hsz]
      exact ⟨rfl, rfl, by simp only; omega, by simp only; omega⟩
    rw [h₁] at this; cases this
  · -- rmw_new
    rintro ⟨a, ha⟩ hr
    simp only [Fin.getElem_fin] at hr ⊢
    obtain ⟨-, hfs, -, -, hloc, hsz, -, hfr, hrr, -, -, -, -, hsc, -⟩ := props a ha
    rw [hrr] at hr
    obtain ⟨h1, hw0, hw1, hoff, hsize⟩ := hP.rmw a (by rw [← hEl]; exact ha) hr
    have h1' : a + 1 < E.length := by rw [hEl]; exact h1
    obtain ⟨hft', hfs', -, -, hloc', hsz', hfw', -, -, -, -, -, -, hsc', -⟩ := props (a + 1) h1'
    refine ⟨h1', by simp [hfr, hw0], by simp [hfw', hw1], ?_, by rw [hfs', hfs]; omega, ?_,
      by rw [hsc', hsc]⟩
    · rw [hft', (props a ha).1]
    · simp only [Ev.key, hloc, hloc', hsz, hsz', hoff, hsize]
  · -- bar_cross
    intro f hf e _ hfa
    obtain ⟨a, ha, rfl⟩ := hmemE f hf
    rw [(props a ha).2.2.2.2.2.2.2.2.2.2.2.1] at hfa; cases hfa
  · -- bar_new
    rintro ⟨a, ha⟩ _ hfa
    simp only [Fin.getElem_fin] at hfa
    rw [(props a ha).2.2.2.2.2.2.2.2.2.2.2.1] at hfa; cases hfa
  · -- val_new: the read returned the last write to its bytes, or the initial bytes.
    rintro ⟨a, ha⟩ hr
    simp only [Fin.getElem_fin] at hr ⊢
    obtain ⟨-, -, -, hpos, hloc, hsz, -, hfr, -, -, -, -, -, -, hv⟩ := props a ha
    have hk : a < q.req.pieces.length := hEl ▸ ha
    have hpw : (q.req.pieces[a]'hk).write = false := by
      cases h : (q.req.pieces[a]'hk).write
      · rfl
      · rw [hfr, h] at hr; cases hr
    refine ⟨fun ⟨a', ha'⟩ hlt' => ?_, ?_⟩
    · simp only [Fin.getElem_fin]
      rw [(props a' ha').2.2.2.2.2.2.1]
      exact hP.reads_first a' a _ hk hlt' hpw
    have hpm : q.req.pieces[a]'hk ∈ q.req.pieces := List.getElem_mem _
    generalize hp : q.req.pieces[a]'hk = p at hpw hpm hloc hsz hv
    have hin := hP.inside p hpm
    have hsp := hP.size_pos p hpm
    let Lj : Nat → Loc := fun j => { l with addr := l.addr + p.off + j }
    have hEa : E[a] ∈ g.evs ++ E := List.mem_append_right _ (List.getElem_mem _)
    have hkey : E[a].key = ({ l with addr := l.addr + p.off }, p.size) := by
      simp only [Ev.key, hloc, hsz]
    -- An earlier write covers byte `j` exactly when it writes the read's location.
    have hcov : ∀ w ∈ g.evs, ∀ j < p.size, w.covers (Lj j) = w.isWriteTo E[a].key := by
      intro w hw' j hj
      apply Bool.eq_iff_iff.2
      rw [covers_iff, isWriteTo_iff, hkey]
      constructor
      · rintro ⟨hww, h₁, h₂, h₃, h₄⟩
        refine ⟨hww, ?_⟩
        have hov : w.overlaps E[a] = true := by
          rw [overlaps_iff, hloc, hsz]
          exact ⟨Performed.isMem_of_isWrite hww, (props a ha).2.2.1, h₁, h₂,
            by simp only [Lj] at h₃; simp only; omega, by simp only [Lj] at h₄; simp only; omega⟩
        rw [hu w (List.mem_append_left _ hw') E[a] hEa hov, hkey]
      · rintro ⟨hww, hwk⟩
        simp only [Ev.key, Prod.mk.injEq] at hwk
        rw [hwk.1, hwk.2]
        exact ⟨hww, rfl, rfl, by simp only [Lj]; omega, by simp only [Lj]; omega⟩
    have hinb : ∀ j < p.size, InBounds g.m (Lj j) := fun j hj =>
      ⟨st, by rw [store?_congr g.m (L := Lj j) (l := l) rfl rfl]; exact hst,
        by simp only [Lj]; omega, by simp only [Lj]; omega⟩
    have hbyte : ∀ j < p.size, byteAt g.m (Lj j) =
        (readBytes st (l.addr - st.base.toNat + p.off) p.size).getD j 0 := by
      intro j hj
      simp only [byteAt, store?_congr g.m (L := Lj j) (l := l) rfl rfl, hst, readBytes]
      rw [getD_of_lt (by simpa [readBytes] using hj)]
      simp only [List.getElem_map, List.getElem_range, Lj]
      congr 1; omega
    have hval : E[a].val = readBytes st (l.addr - st.base.toNat + p.off) p.size := by
      rw [hv, if_neg (by simp [hpw]), hb, readPart_extract hP hpm hpw]
    have hlen : (readBytes st (l.addr - st.base.toNat + p.off) p.size).length = p.size := by
      simp [readBytes]
    rcases hi.store (Lj 0) (hinb 0 hsp) with ⟨w, hw₁, hw₂, -⟩ | ⟨hn, -⟩
    · -- The last write to the location.
      have hwk : g.evs[w].isWriteTo E[a].key = true := by
        rw [← hcov g.evs[w] (List.getElem_mem _) 0 hsp]; exact hw₁
      have hwlast : ∀ w' : Fin g.evs.length, w < w' → g.evs[w'].isWriteTo E[a].key = false :=
        fun w' hw' => by rw [← hcov g.evs[w'] (List.getElem_mem _) 0 hsp]; exact hw₂ w' hw'
      refine .inl ⟨w, hwk, hwlast, ?_⟩
      obtain ⟨hww, hwkey⟩ := isWriteTo_iff.1 hwk
      rw [hkey] at hwkey
      simp only [Ev.key, Prod.mk.injEq] at hwkey
      have hwv : g.evs[w].val.length = p.size :=
        ((hi.mem_size _ (List.getElem_mem _) (Performed.isMem_of_isWrite hww)).2 hww).trans hwkey.2
      rw [hval]
      apply List.ext_getElem (by rw [hlen]; exact hwv.symm)
      intro j h₁ h₂
      have h₁' : j < p.size := by rw [hlen] at h₁; exact h₁
      rw [← getD_of_lt h₁ 0, ← hbyte j h₁', ← getD_of_lt h₂ 0]
      rcases hi.store (Lj j) (hinb j h₁') with ⟨w', h₃, h₄, h₅⟩ | ⟨h₃, -⟩
      · have hw'k : g.evs[w'].isWriteTo E[a].key = true := by
          rw [← hcov g.evs[w'] (List.getElem_mem _) j h₁']; exact h₃
        have : w' = w := by
          rcases Nat.lt_trichotomy w'.val w.val with c | c | c
          · have := h₄ w c
            rw [hcov g.evs[w] (List.getElem_mem _) j h₁', hwk] at this; cases this
          · exact Fin.ext c
          · rw [hwlast w' c] at hw'k; cases hw'k
        subst this
        rw [h₅, hwkey.1]
        congr 1; simp only [Lj]; omega
      · have := h₃ w
        rw [hcov g.evs[w] (List.getElem_mem _) j h₁', hwk] at this; cases this
    · -- No write to the location yet: the initial bytes.
      refine .inr ⟨fun w => by
        have := hn w; rw [hcov g.evs[w] (List.getElem_mem _) 0 hsp] at this; exact this, ?_⟩
      rw [hval, hkey, initBytes]
      apply List.ext_getElem (by simp [hlen])
      intro j h₁ h₂
      rw [hlen] at h₁
      rw [← getD_of_lt (by rw [hlen]; exact h₁) 0, ← hbyte j h₁]
      simp only [List.getElem_map, List.getElem_range]
      rcases hi.store (Lj j) (hinb j h₁) with ⟨w', h₃, -, -⟩ | ⟨-, h₄⟩
      · have := hn w'
        rw [hcov g.evs[w'] (List.getElem_mem _) 0 hsp, ← hcov g.evs[w'] (List.getElem_mem _) j h₁,
          h₃] at this
        cases this
      · rw [h₄]
  · -- seq_lt
    intro e he
    rcases List.mem_append.1 he with he | he
    · exact hi.seq_lt e he
    · obtain ⟨a, ha, rfl⟩ := hmemE e he
      obtain ⟨hft, hfs, -⟩ := props a ha
      show E[a].seq < g.ctr E[a].e.thread
      rw [hft, hfs]
      have : a < q.req.pieces.length := hEl ▸ ha
      omega
  · -- mem_size
    intro e he hm
    rcases List.mem_append.1 he with he | he
    · exact hi.mem_size e he hm
    · obtain ⟨a, ha, rfl⟩ := hmemE e he
      obtain ⟨-, -, -, hpos, -, hsz, hfw, -, -, -, -, -, -, -, hv⟩ := props a ha
      refine ⟨hpos, fun hw' => ?_⟩
      rw [hfw] at hw'
      rw [hv, if_pos hw', hsz]
      obtain ⟨v, hv', hl'⟩ := e₁ _ (List.mem_filter.2 ⟨hpiece a ha, hw'⟩)
      rw [hv']; exact hl'
  · -- queued
    intro s p' hp'
    obtain ⟨p₀, h₀, eid, ewho, ereq, esub, enq, enod, -⟩ := hin' s p' hp'
    obtain ⟨h₁, h₂, h₃, h₄, h₅⟩ := hi.queued s p₀ h₀
    refine ⟨ewho.trans h₁, by rw [eid, hnx]; exact h₂, enod h₃, by rw [ereq]; exact h₄,
      fun l' hl' => ?_⟩
    obtain ⟨a₁, a₂, a₃, a₄⟩ := h₅ l' (esub l' hl')
    rw [eid, ewho, ereq]
    refine ⟨a₁, a₂, a₃, fun e he hth => ?_⟩
    show e.seq < g.base p₀.id l' ∨ g.base p₀.id l' + p₀.req.pieces.length ≤ e.seq
    rcases List.mem_append.1 he with he | he
    · exact a₄ e he hth
    · obtain ⟨a, ha, rfl⟩ := hmemE e he
      obtain ⟨hft, hfs, -⟩ := props a ha
      obtain ⟨hs_eq, hl_eq⟩ := same_slot s p₀ h₀ l' (esub l' hl') (hth.symm.trans hft)
      have hs_eq' := hs_eq.symm
      have hl_eq' := hl_eq.symm
      subst hs_eq'; subst hl_eq'
      have hne : p₀ ≠ q := fun e => enq e hl'
      have hk : a < q.req.pieces.length := hEl ▸ ha
      rcases pairwise_either (hi.ordered slot) h₀ hqin hne with h | h
      · have := h.2 lane (esub lane hl') hlane; rw [hbs] at this; rw [hfs]; omega
      · have := h.2 lane hlane (esub lane hl'); rw [hbs] at this; rw [hfs]; omega
  · -- ordered
    intro s
    show (Flat.queue _ s).Pairwise _
    rw [hq']
    split
    · next hseq =>
      subst hseq
      apply pairwise_retire _ (hi.ordered _)
      intro a a' b' b'' hR e₁ e₂ e₃ e₄ e₅ e₆
      refine ⟨by rw [e₁, e₄]; exact hR.1, fun l' h₁ h₂ => ?_⟩
      rw [e₁, e₂, e₄]
      exact hR.2 l' (e₃ l' h₁) (e₆ l' h₂)
    · exact hi.ordered s
  · -- pending
    intro e he s p' hp' l' hl' hth hlt'
    obtain ⟨p₀, h₀, eid, ewho, ereq, esub, enq, -, ecta⟩ := hin' s p' hp'
    have ht' : touches e p' l' = touches e p₀ l' := by simp [touches, ecta, ereq]
    rw [ht', ereq]
    rw [ewho] at hth
    rw [eid] at hlt'
    replace hlt' : g.base p₀.id l' < e.seq := hlt'
    rcases List.mem_append.1 he with he | he
    · exact hi.pending e he s p₀ h₀ l' (esub l' hl') hth hlt'
    · obtain ⟨a, ha, rfl⟩ := hmemE e he
      obtain ⟨hft, hfs, -, -, -, -, -, -, -, hfd, -⟩ := props a ha
      obtain ⟨hs_eq, hl_eq⟩ := same_slot s p₀ h₀ l' (esub l' hl') (hth.symm.trans hft)
      have hs_eq' := hs_eq.symm
      have hl_eq' := hl_eq.symm
      subst hs_eq'; subst hl_eq'
      have hne : p₀ ≠ q := fun e => enq e hl'
      have hk : a < q.req.pieces.length := hEl ▸ ha
      -- `p₀` was issued before `q`, so the delivery checked it.
      have hnot : ¬ ((q.id ≠ p₀.id) ∧ ∀ l ∈ q.lanes, l ∈ p₀.lanes →
          g.base q.id l + q.req.pieces.length ≤ g.base p₀.id l) := fun h => by
        have := h.2 lane hlane (esub lane hl'); rw [hbs] at this; rw [hfs] at hlt'; omega
      have hbefore := mem_take_of_pairwise (hi.ordered slot) hlt h₀ (by rw [hq]; exact hne)
        (by rw [hq]; exact hnot)
      have hbl := hblk p₀ hbefore
      have hcontains : p₀.lanes.contains lane = true := List.contains_iff_mem.2 (esub lane hl')
      simp only [blocker, hcontains, Bool.not_true, Bool.false_eq_true, if_false] at hbl
      split at hbl
      · cases hbl
      · next hov =>
        split at hbl
        · cases hbl
        · next hrel =>
          split at hbl
          · cases hbl
          · next hacq =>
            simp only [Bool.not_eq_true] at hov hrel hacq
            refine ⟨by rw [hfd, hrel]; simp, hacq, fun _ => ?_⟩
            obtain ⟨hs₁, hs₂, hs₃, hs₄⟩ := new_inside E[a] (List.getElem_mem _)
            unfold touches
            unfold overlap at hov
            rw [hl] at hov
            split
            · next x hx =>
              rw [hx] at hov
              simp only [Bool.and_eq_false_iff, beq_iff_eq, decide_eq_false_iff_not] at hov ⊢
              rw [hs₁, hs₂]
              rcases hov with (((h | h) | h) | h)
              · exact .inl (.inl (.inl h))
              · exact .inl (.inl (.inr h))
              · exact .inl (.inr (by omega))
              · exact .inr (by omega)
            · rfl
  · -- store
    intro L hL
    have hbyte' : byteAt { g.m.setStore l st' with
        queues := g.m.queues.set! slot (retire (g.m.queue slot) q.id lane) } L =
        byteAt (g.m.setStore l st') L := by simp only [byteAt, hst']
    show _ ∨ _
    dsimp only
    rw [hbyte']
    obtain ⟨sL, hsL, hL₁, hL₂⟩ := hL
    rw [hst', store?_setStore hst] at hsL
    -- Events of the delivery that cover `L` are its writes over `L`'s byte.
    have newcov : ∀ (a : Nat) (ha : a < E.length), E[a].covers L = true →
        L.space = l.space ∧ L.owner = l.owner ∧ (q.req.pieces[a]'(hEl ▸ ha)).write = true ∧
          l.addr + (q.req.pieces[a]'(hEl ▸ ha)).off ≤ L.addr ∧
          L.addr < l.addr + (q.req.pieces[a]'(hEl ▸ ha)).off + (q.req.pieces[a]'(hEl ▸ ha)).size := by
      intro a ha hcv
      obtain ⟨-, -, -, -, hloc, hsz, hfw, -⟩ := props a ha
      rw [covers_iff, hloc, hsz, hfw] at hcv
      obtain ⟨hw', h₁, h₂, h₃, h₄⟩ := hcv
      exact ⟨h₁.symm, h₂.symm, hw', h₃, h₄⟩
    split at hsL
    · next hsame =>
      obtain ⟨hs₁, hs₂⟩ := hsame
      simp only [Option.some.injEq] at hsL
      subst hsL
      have hb' : byteAt (g.m.setStore l st') L = st'.readByte (L.addr - st.base.toNat) := by
        simp only [byteAt, store?_setStore hst, hs₁, hs₂, and_self, ite_true, b₁]
      rw [hb']
      rw [b₁] at hL₁ hL₂
      rw [z₁] at hL₂
      by_cases hcovw : ∃ (a : Nat) (ha : a < q.req.pieces.length), (q.req.pieces[a]'ha).write = true ∧
          l.addr + (q.req.pieces[a]'ha).off ≤ L.addr ∧
          L.addr < l.addr + (q.req.pieces[a]'ha).off + (q.req.pieces[a]'ha).size
      · obtain ⟨a, ha, hpw, hp₁, hp₂⟩ := hcovw
        have ha' : a < E.length := hEl ▸ ha
        obtain ⟨-, -, -, -, hloc, hsz, hfw, -, -, -, -, -, -, -, hv⟩ := props a ha'
        have hidx : g.evs.length + a < (g.evs ++ E).length := by simp; omega
        have hEget : (g.evs ++ E)[g.evs.length + a]'hidx = E[a] := by
          rw [List.getElem_append_right (by omega)]; simp
        have hpf : q.req.pieces[a]'ha ∈ q.req.pieces.filter (·.write) :=
          List.mem_filter.2 ⟨List.getElem_mem _, hpw⟩
        refine .inl ⟨⟨g.evs.length + a, hidx⟩, ?_, ?_, ?_⟩
        · simp only [Fin.getElem_fin, hEget]
          rw [covers_iff, hloc, hsz, hfw]
          exact ⟨hpw, hs₁.symm, hs₂.symm, hp₁, hp₂⟩
        · rintro ⟨j', hj'⟩ hlt'
          have hlt' : g.evs.length + a < j' := hlt'
          simp only [Fin.getElem_fin]
          rw [List.getElem_append_right (by omega)]
          have hb₁ : j' - g.evs.length < E.length := by simp at hj'; omega
          cases hcv : E[j' - g.evs.length].covers L
          · rfl
          · exfalso
            obtain ⟨-, -, hw'', h₁, h₂⟩ := newcov _ hb₁ hcv
            have := hP.disjoint a (j' - g.evs.length) ha (hEl ▸ hb₁) (by omega) hpw hw''
            omega
        · simp only [Fin.getElem_fin, hEget]
          rw [c₁ _ hpf _ (by omega) (by omega), hv, if_pos hpw, hloc]
          congr 1
          simp only
          omega
      · have hnc : ∀ p ∈ q.req.pieces.filter (·.write),
            ¬(l.addr - st.base.toNat + p.off ≤ L.addr - st.base.toNat ∧
              L.addr - st.base.toNat < l.addr - st.base.toNat + p.off + p.size) := by
          intro p hp h
          obtain ⟨hpm, hpw⟩ := List.mem_filter.1 hp
          obtain ⟨a, ha, rfl⟩ := List.mem_iff_getElem.1 hpm
          exact hcovw ⟨a, ha, hpw, by omega, by omega⟩
        rw [u₁ _ hnc]
        have hold : st.readByte (L.addr - st.base.toNat) = byteAt g.m L := by
          simp only [byteAt, store?_congr g.m hs₁ hs₂, hst]
        rw [hold]
        refine store_lift (hi.store L ⟨st, by rw [store?_congr g.m hs₁ hs₂]; exact hst, hL₁, hL₂⟩)
          fun a ha => ?_
        cases hcv : E[a].covers L
        · rfl
        · exfalso
          obtain ⟨-, -, hw'', h₁, h₂⟩ := newcov a ha hcv
          exact hcovw ⟨a, hEl ▸ ha, hw'', h₁, h₂⟩
    · next hdiff =>
      -- Another store: unchanged, and nothing of the delivery covers it.
      have hb' : byteAt (g.m.setStore l st') L = byteAt g.m L := by
        simp only [byteAt, store?_setStore hst, hdiff, ite_false]
      rw [hb']
      refine store_lift (hi.store L ⟨sL, hsL, hL₁, hL₂⟩) fun a ha => ?_
      cases hcv : E[a].covers L
      · rfl
      · exact absurd ⟨(newcov a ha hcv).1, (newcov a ha hcv).2.1⟩ hdiff
  · -- bars
    intro e he
    rw [hbars']
    rcases List.mem_append.1 he with he | he
    · exact hi.bars e he
    · obtain ⟨a, ha, rfl⟩ := hmemE e he
      obtain ⟨-, -, -, -, -, -, -, -, -, -, -, hfa, hfw, -, -⟩ := props a ha
      exact ⟨fun h => (by rw [hfa] at h; cases h), fun h => (by rw [hfw] at h; cases h)⟩
  · -- stores
    have := storesOk_setStore hi.stores hst w₁ f₁
    exact this
  · -- owners
    intro e he hm hsh
    show e.loc.owner < (g.m.setStore l st').shared.size
    rw [setStore_shared_size]
    rcases List.mem_append.1 he with he | he
    · exact hi.owners e he hm hsh
    · obtain ⟨a, ha, rfl⟩ := hmemE e he
      obtain ⟨-, -, -, -, hloc, -⟩ := props a ha
      rw [hloc] at hsh ⊢
      rcases store?_some hst with ⟨hs, -, -⟩ | ⟨-, hlt', -⟩
      · rw [hs] at hsh; cases hsh
      · exact hlt'
  · -- fresh
    intro L hs hle
    exact hi.fresh L hs (by rw [setStore_shared_size] at hle; exact hle)

/-! ## Fences and barriers -/

theorem drained_iff {m : Flat} {slot : Nat} : m.drained slot = true ↔ m.queue slot = [] := by
  simp [drained, List.isEmpty_iff]

theorem fence_ok {m m' : Flat} {slot : Nat} (h : m.fence slot = .ok m') :
    m.queue slot = [] ∧ m' = m := by
  unfold Flat.fence at h
  split at h
  · next hd => simp only [Except.ok.injEq] at h; exact ⟨drained_iff.1 hd, h.symm⟩
  · cases h

theorem wait_ok {m m' : Flat} {slot b k : Nat} (h : m.wait slot b k = .ok m') :
    m.queue slot = [] ∧ k < (m.bar b).done ∧ m' = m := by
  unfold Flat.wait at h
  split at h
  · cases h
  · next hd =>
    simp only [Bool.not_eq_true', Bool.not_eq_false] at hd
    split at h
    · next hp =>
      simp only [Except.ok.injEq] at h
      simp only [passed, decide_eq_true_eq] at hp
      exact ⟨drained_iff.1 hd, hp, h.symm⟩
    · cases h

theorem bar_set (m : Flat) (b b' : Nat) (x : Barrier) :
    ({ m with bars := (if b < m.bars.size then m.bars
        else m.bars ++ Array.replicate (b + 1 - m.bars.size) default).set! b x } : Flat).bar b' =
      if b' = b then x else m.bar b' := by
  generalize hB : (if b < m.bars.size then m.bars
      else m.bars ++ Array.replicate (b + 1 - m.bars.size) (default : Barrier)) = B
  have hsz : b < B.size := by
    rw [← hB]; split
    · assumption
    · simp; omega
  have hget : ∀ c : Nat, B[c]?.getD default = m.bars[c]?.getD default := by
    intro c
    rw [← hB]
    split
    · rfl
    · by_cases hc : c < m.bars.size
      · rw [Array.getElem?_append_left hc]
      · rw [Array.getElem?_append_right (by omega), Array.getElem?_replicate,
          Array.getElem?_eq_none (show m.bars.size ≤ c by omega)]
        split <;> rfl
  simp only [bar, Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds]
  by_cases hb : b' = b
  · subst hb; simp [hsz]
  · have hb' : ¬ b = b' := fun e => hb e.symm
    rw [if_neg hb', if_neg hb, hget]

theorem arrive_ok {m m' : Flat} {slot b n e k : Nat} (h : m.arrive slot b n e = .ok (m', k)) :
    m.queue slot = [] ∧ k = (m.bar b).done ∧ m'.queues = m.queues ∧ m'.next = m.next ∧
      m'.global = m.global ∧ m'.shared = m.shared ∧ ∀ b', (m.bar b').done ≤ (m'.bar b').done := by
  unfold Flat.arrive at h
  split at h
  · cases h
  · next hd =>
    simp only [Bool.not_eq_true', Bool.not_eq_false] at hd
    dsimp only at h
    split at h
    · cases h
    · split at h
      · cases h
      · simp only [Except.ok.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        refine ⟨drained_iff.1 hd, rfl, rfl, rfl, rfl, rfl, fun b' => ?_⟩
        rw [bar_set]
        split
        · next hb => subst hb; split <;> simp
        · exact Nat.le_refl _

theorem fenceEvent_props (op : SyncOp) (t s : Nat) :
    (fenceEvent op t s).e.thread = t ∧ (fenceEvent op t s).seq = s ∧
      (fenceEvent op t s).e.isMem = false ∧ (fenceEvent op t s).rmwRead = false ∧
      (fenceEvent op t s).arrives = false ∧ (fenceEvent op t s).waits = false := by
  cases op with
  | fence sc scope =>
    cases sc <;> simp [fenceEvent, Event.isMem, Event.isRead, Event.isWrite, Ev.arrives, Ev.waits]
  | _ => simp [fenceEvent, Event.isMem, Event.isRead, Event.isWrite, Ev.arrives, Ev.waits]

theorem Inv.fence {g g' : Ghost} (hi : Inv init W slotOf g) {who : Issuer} {lanes : List Nat}
    {op : SyncOp} (hok : (Op.fence who lanes op).Ok W slotOf) (hs : g.step (.fence who lanes op) = some g') :
    Inv init W slotOf g' := by
  simp only [Ghost.step] at hs
  split at hs
  · cases hs
  · split at hs
    · next m hf =>
      obtain ⟨hdr, rfl⟩ := fence_ok hf
      simp only [Option.some.injEq] at hs
      subst hs
      refine hi.sync hok hdr rfl rfl rfl rfl (fun t s => ?_) (.inl fun t s => ?_)
        (fun a ha e _ hfa => ?_) (fun e he => ?_)
      · obtain ⟨h₁, h₂, h₃, h₄, -⟩ := fenceEvent_props op t s
        exact ⟨h₁, h₂, h₃, h₄, by cases op <;> rfl⟩
      · exact (fenceEvent_props op t s).2.2.2.2.1
      · simp only [Ghost.syncEvents, List.mem_map] at ha
        obtain ⟨l, _, rfl⟩ := ha
        rw [(fenceEvent_props op _ _).2.2.2.2.1] at hfa; cases hfa
      · rcases List.mem_append.1 he with he | he
        · exact hi.bars e he
        · simp only [Ghost.syncEvents, List.mem_map] at he
          obtain ⟨l, _, rfl⟩ := he
          obtain ⟨-, -, -, -, h₁, h₂⟩ := fenceEvent_props op (who.tid + l) (g.ctr (who.tid + l))
          exact ⟨fun h => (by rw [h₁] at h; cases h), fun h => (by rw [h₂] at h; cases h)⟩
    · cases hs

theorem barEvent_props (t s : Nat) (id : Option Nat) (w : Bool) (b k e : Nat) :
    (barEvent t s id w b k e).e.thread = t ∧ (barEvent t s id w b k e).seq = s ∧
      (barEvent t s id w b k e).e.isMem = false ∧ (barEvent t s id w b k e).rmwRead = false ∧
      (barEvent t s id w b k e).obj = b ∧ (barEvent t s id w b k e).inst = k ∧
      (barEvent t s id w b k e).arrives = !w ∧ (barEvent t s id w b k e).waits = w := by
  unfold barEvent
  cases id <;> cases w <;> simp [Event.isMem, Event.isRead, Event.isWrite, Ev.arrives, Ev.waits]

theorem Inv.arrive {g g' : Ghost} (hi : Inv init W slotOf g) {who : Issuer} {lanes : List Nat}
    {id : Option Nat} {b e : Nat} (hok : (Op.arrive who lanes id b e).Ok W slotOf)
    (hs : g.step (.arrive who lanes id b e) = some g') : Inv init W slotOf g' := by
  simp only [Ghost.step] at hs
  split at hs
  · next m k ha =>
    obtain ⟨hdr, rfl, hq, hn, hg, hsh, hmono⟩ := arrive_ok ha
    simp only [Option.some.injEq] at hs
    subst hs
    have hnew : ∀ f ∈ g.syncEvents who lanes (fun t s => barEvent t s id false b (g.m.bar b).done e),
        f.obj = b ∧ f.inst = (g.m.bar b).done ∧ f.arrives = true ∧ f.waits = false := by
      intro f hf
      simp only [Ghost.syncEvents, List.mem_map] at hf
      obtain ⟨l, _, rfl⟩ := hf
      obtain ⟨-, -, -, -, h₁, h₂, h₃, h₄⟩ := barEvent_props (who.tid + l) (g.ctr (who.tid + l)) id
        false b (g.m.bar b).done e
      exact ⟨h₁, h₂, by rw [h₃]; rfl, h₄⟩
    refine hi.sync hok hdr hq hn hg hsh (fun t s => ?_) (.inr fun t s => ?_)
      (fun a ha e he _ hwe ho => ?_) (fun e he => ?_)
    · obtain ⟨h₁, h₂, h₃, h₄, -⟩ := barEvent_props t s id false b (g.m.bar b).done e
      exact ⟨h₁, h₂, h₃, h₄, rfl⟩
    · exact (barEvent_props t s id false b (g.m.bar b).done e).2.2.2.2.2.2.2
    · obtain ⟨h₁, h₂, -⟩ := hnew a ha
      have := (hi.bars e he).2 hwe
      rw [h₂]; rw [h₁] at ho; rw [← ho] at this; omega
    · rcases List.mem_append.1 he with he | he
      · obtain ⟨h₁, h₂⟩ := hi.bars e he
        exact ⟨fun h => Nat.le_trans (h₁ h) (hmono _), fun h => Nat.lt_of_lt_of_le (h₂ h) (hmono _)⟩
      · obtain ⟨h₁, h₂, -, h₄⟩ := hnew e he
        refine ⟨fun _ => ?_, fun h => (by rw [h₄] at h; cases h)⟩
        rw [h₂, h₁]; exact hmono b
  · cases hs

theorem Inv.wait {g g' : Ghost} (hi : Inv init W slotOf g) {who : Issuer} {lanes : List Nat}
    {id : Option Nat} {b k ex : Nat} (hok : (Op.wait who lanes id b k ex).Ok W slotOf)
    (hs : g.step (.wait who lanes id b k ex) = some g') : Inv init W slotOf g' := by
  simp only [Ghost.step] at hs
  split at hs
  · next m hw =>
    obtain ⟨hdr, hlt, rfl⟩ := wait_ok hw
    simp only [Option.some.injEq] at hs
    subst hs
    refine hi.sync hok hdr rfl rfl rfl rfl (fun t s => ?_) (.inl fun t s => ?_)
      (fun a ha e _ hfa => ?_) (fun e he => ?_)
    · obtain ⟨h₁, h₂, h₃, h₄, -⟩ := barEvent_props t s id true b k ex
      exact ⟨h₁, h₂, h₃, h₄, rfl⟩
    · exact (barEvent_props t s id true b k ex).2.2.2.2.2.2.1
    · simp only [Ghost.syncEvents, List.mem_map] at ha
      obtain ⟨l, _, rfl⟩ := ha
      rw [(barEvent_props _ _ id true b k ex).2.2.2.2.2.2.1] at hfa; cases hfa
    · rcases List.mem_append.1 he with he | he
      · exact hi.bars e he
      · simp only [Ghost.syncEvents, List.mem_map] at he
        obtain ⟨l, _, rfl⟩ := he
        obtain ⟨-, -, -, -, h₁, h₂, h₃, -⟩ := barEvent_props (who.tid + l) (g.ctr (who.tid + l)) id
          true b k ex
        refine ⟨fun h => (by rw [h₃] at h; cases h), fun _ => ?_⟩
        rw [h₂, h₁]; exact hlt
  · cases hs

/-! ## Growing the machine -/

theorem Inv.addSlot {g g' : Ghost} (hi : Inv init W slotOf g) (hs : g.step .addSlot = some g') :
    Inv init W slotOf g' := by
  simp only [Ghost.step, Option.some.injEq] at hs
  subst hs
  have hq : ∀ s, (g.m.addSlot.1).queue s = g.m.queue s := fun s => by
    simp only [Flat.addSlot, queue, Array.getElem?_push]
    split
    · next h => subst h; simp [Array.getElem?_eq_none]
    · rfl
  have hst : ∀ L, (g.m.addSlot.1).store? L = g.m.store? L := fun L => by
    simp only [Flat.addSlot, store?]
  have hb : ∀ L, byteAt (g.m.addSlot.1) L = byteAt g.m L := fun L => by simp only [byteAt, hst]
  refine
    { perf := hi.perf, seq_lt := hi.seq_lt, mem_size := hi.mem_size, queued := ?_, ordered := ?_
      pending := ?_, store := fun L hL => ?_, bars := hi.bars, stores := hi.stores
      owners := hi.owners, fresh := hi.fresh }
  · intro s q h; rw [hq] at h; exact hi.queued s q h
  · intro s; show (Flat.queue _ s).Pairwise _; rw [hq]; exact hi.ordered s
  · intro e he s q h; rw [hq] at h; exact hi.pending e he s q h
  · obtain ⟨st, h₁, h₂⟩ := hL
    rw [hst] at h₁
    rw [hb]; exact hi.store L ⟨st, h₁, h₂⟩

theorem empty_readByte (base : UInt64) (size off : Nat) : (Store.empty base size).readByte off = 0 := by
  simp only [Store.readByte, Store.page_empty]

theorem empty_ok (base : UInt64) (size : Nat) : (Store.empty base size).WF ∧ (Store.empty base size).Fits := by
  refine ⟨⟨fun i p h => ?_, Store.shaped_empty base size⟩, ?_⟩
  · rw [Store.page_empty] at h; cases h
  · simp only [Store.Fits, Store.empty]
    have := Nat.div_add_mod' (size + pageSize - 1) pageSize
    have := Nat.mod_lt (size + pageSize - 1) Store.pageSize_pos
    have := Store.pageSize_pos
    omega

theorem Inv.addCta {g g' : Ghost} (hi : Inv init W slotOf g) {size : Nat}
    (hs : g.step (.addCta size) = some g') : Inv init W slotOf g' := by
  simp only [Ghost.step, Option.some.injEq] at hs
  subst hs
  have hsz : (g.m.addCta size).1.shared.size = g.m.shared.size + 1 := by simp [Flat.addCta]
  have hst : ∀ L, L.space ≠ .shared ∨ L.owner < g.m.shared.size →
      (g.m.addCta size).1.store? L = g.m.store? L := fun L h => by
    simp only [Flat.addCta, store?]
    split
    · rfl
    · next hs =>
      rcases h with h | h
      · exact absurd hs h
      · rw [Array.getElem?_push_lt h, Array.getElem?_eq_getElem h]
    · rfl
  refine
    { perf := hi.perf, seq_lt := hi.seq_lt, mem_size := hi.mem_size, queued := hi.queued
      ordered := hi.ordered, pending := hi.pending, store := fun L hL => ?_, bars := hi.bars
      stores := ?_, owners := ?_, fresh := ?_ }
  · by_cases hnew : L.space = .shared ∧ L.owner = g.m.shared.size
    · -- The new CTA: nothing has touched it, and it starts zero-filled.
      right
      refine ⟨fun j => ?_, ?_⟩
      · cases hc : g.evs[j].covers L
        · rfl
        · exfalso
          rw [covers_iff] at hc
          have := hi.owners g.evs[j] (List.getElem_mem _) (Performed.isMem_of_isWrite hc.1)
            (hc.2.1.trans hnew.1)
          rw [hc.2.2.1, hnew.2] at this
          exact Nat.lt_irrefl _ this
      · rw [hi.fresh L hnew.1 (by rw [hnew.2]; exact Nat.le_refl _)]
        simp only [byteAt, Flat.addCta, store?, hnew.1, hnew.2, Array.getElem?_push_size]
        exact empty_readByte _ _ _
    · have hold : L.space ≠ .shared ∨ L.owner < g.m.shared.size := by
        by_cases hs : L.space = .shared
        · right
          obtain ⟨st, h₁, -⟩ := hL
          simp only [Flat.addCta, store?, hs] at h₁
          rw [Array.getElem?_push] at h₁
          split at h₁
          · next h => exact absurd ⟨hs, h⟩ hnew
          · exact (Array.getElem?_eq_some_iff.1 h₁).1
        · exact .inl hs
      have hb : byteAt (g.m.addCta size).1 L = byteAt g.m L := by simp only [byteAt, hst L hold]
      obtain ⟨st, h₁, h₂⟩ := hL
      rw [hst L hold] at h₁
      rw [hb]; exact hi.store L ⟨st, h₁, h₂⟩
  · obtain ⟨h₁, h₂, h₃⟩ := hi.stores
    refine ⟨h₁, h₂, fun s hs => ?_⟩
    simp only [Flat.addCta, Array.mem_push] at hs
    rcases hs with hs | rfl
    · exact h₃ s hs
    · exact empty_ok _ _
  · intro e he hm hs
    rw [hsz]; exact Nat.lt_succ_of_lt (hi.owners e he hm hs)
  · intro L hs hle
    exact hi.fresh L hs (by rw [hsz] at hle; omega)

/-! ## Runs -/

theorem step_evs {g g' : Ghost} {op : Op} (h : g.step op = some g') : ∃ es, g'.evs = g.evs ++ es := by
  cases op with
  | issue who r cta =>
    simp only [Ghost.step] at h
    split at h
    · simp only [Option.some.injEq] at h; subst h; exact ⟨[], by simp⟩
    · cases h
  | deliver slot id lane =>
    simp only [Ghost.step] at h
    split at h
    · simp only [Option.map_eq_some_iff] at h; obtain ⟨_, _, rfl⟩ := h; exact ⟨_, rfl⟩
    · cases h
  | fence who lanes op =>
    simp only [Ghost.step] at h
    split at h
    · cases h
    · split at h
      · simp only [Option.some.injEq] at h; subst h; exact ⟨_, rfl⟩
      · cases h
  | arrive who lanes id b e =>
    simp only [Ghost.step] at h
    split at h
    · simp only [Option.some.injEq] at h; subst h; exact ⟨_, rfl⟩
    · cases h
  | wait who lanes id b k e =>
    simp only [Ghost.step] at h
    split at h
    · simp only [Option.some.injEq] at h; subst h; exact ⟨_, rfl⟩
    · cases h
  | addSlot => simp only [Ghost.step, Option.some.injEq] at h; subst h; exact ⟨[], by simp⟩
  | addCta size => simp only [Ghost.step, Option.some.injEq] at h; subst h; exact ⟨[], by simp⟩

theorem run_evs {g g' : Ghost} {ops : List Op} (h : g.run ops = some g') :
    ∃ es, g'.evs = g.evs ++ es := by
  induction ops generalizing g with
  | nil => simp only [Ghost.run, Option.some.injEq] at h; subst h; exact ⟨[], by simp⟩
  | cons op ops ih =>
    simp only [Ghost.run, Option.bind_eq_some_iff] at h
    obtain ⟨g₂, h₃, h₄⟩ := h
    obtain ⟨es₁, e₁⟩ := step_evs h₃
    obtain ⟨es₂, e₂⟩ := ih h₄
    exact ⟨es₁ ++ es₂, by rw [e₂, e₁, List.append_assoc]⟩

theorem Inv.step {g g' : Ghost} (hi : Inv init W slotOf g) {op : Op} (hok : op.Ok W slotOf)
    (hs : g.step op = some g') (hu : Uniform g'.evs) : Inv init W slotOf g' := by
  cases op with
  | issue who r cta => exact hi.issue hok hs
  | deliver slot id lane => exact hi.deliver hs hu
  | fence who lanes op => exact hi.fence hok hs
  | arrive who lanes id b e => exact hi.arrive hok hs
  | wait who lanes id b k e => exact hi.wait hok hs
  | addSlot => exact hi.addSlot hs
  | addCta size => exact hi.addCta hs

theorem Inv.run {g g' : Ghost} (hi : Inv init W slotOf g) {ops : List Op}
    (hok : ∀ op ∈ ops, op.Ok W slotOf) (hr : g.run ops = some g') (hu : Uniform g'.evs) :
    Inv init W slotOf g' := by
  induction ops generalizing g with
  | nil => simp only [Ghost.run, Option.some.injEq] at hr; subst hr; exact hi
  | cons op ops ih =>
    simp only [Ghost.run, Option.bind_eq_some_iff] at hr
    obtain ⟨g₁, h₁, h₂⟩ := hr
    -- The events so far are a prefix of the run's.
    obtain ⟨es, hes⟩ := run_evs h₂
    have hu₁ : Uniform g₁.evs := fun a ha b hb => hu a (by rw [hes]; exact List.mem_append_left _ ha)
      b (by rw [hes]; exact List.mem_append_left _ hb)
    exact ih (hi.step (hok op (by simp)) h₁ hu₁) (fun o ho => hok o (by simp [ho])) h₂

end

/-- **`Mem.sound`: every run of the request machine performs a consistent execution.** From a
machine with nothing in flight and well-formed stores, any sequence of steps it accepts, whose
issuers are fixed per slot with disjoint threads and whose accesses do not overlap partially,
lists events (`Ghost.evs`, in the order they performed, with the bytes the machine returned)
whose execution is consistent under the PTX memory model, wherever the threads are placed. -/
theorem sound {m₀ : Flat} (hq : ∀ s, m₀.queue s = []) (hs : StoresOk m₀) {ops : List Op}
    {W : Nat → Issuer} {slotOf : Nat → Nat} (hok : ∀ op ∈ ops, op.Ok W slotOf) {g : Ghost}
    (hr : (Ghost.init m₀).run ops = some g) (hu : Uniform g.evs) (place : Nat → PTX.Place) :
    (Performed.exec g.evs (initOf m₀) place).consistent = true :=
  ((inv_init hq hs).run hok hr hu).perf.consistent place

end NV.Flat
