import NV.Memory.Sound

/-!
# A request's lanes delivered together

`Flat.deliverLanes` (the timing core's delivery of a request part's lanes) equals delivering them
one after another (`Flat.deliverSeq`): a run that uses it is a sequence of `deliver` steps, which
`Mem.sound` covers.
-/

namespace NV.Flat

theorem filterMap_congr' {α β : Type _} {f g : α → Option β} :
    ∀ {l : List α}, (∀ x ∈ l, f x = g x) → l.filterMap f = l.filterMap g
  | [], _ => rfl
  | x :: xs, h => by
    rw [List.filterMap_cons, List.filterMap_cons, h x (by simp),
      filterMap_congr' fun y hy => h y (by simp [hy])]

theorem retireOne_other {id lane : Nat} {p : InFlight} (h : p.id ≠ id) : retireOne id lane p = some p := by
  have : (p.id == id) = false := by simpa using h
  simp [retireOne, this]

theorem retireLanes_nil (id : Nat) (p : InFlight) : retireLanes id [] p = some p := by
  unfold retireLanes; split <;> rfl

theorem retireLanes_append (id : Nat) (done : List Nat) (l : Nat) (p : InFlight) :
    (retireLanes id done p).bind (retireOne id l) = retireLanes id (done ++ [l]) p := by
  unfold retireLanes
  split
  · simp [List.foldlM_append]
  · next h =>
    have : p.id ≠ id := by simpa using h
    simp [retireOne_other this]

theorem foldl_retire_keeps {id : Nat} : ∀ (done : List Nat) {p q : InFlight},
    done.foldlM (fun p l => retireOne id l p) p = some q → q.id = p.id ∧ q.req = p.req ∧ q.cta = p.cta
  | [], p, q, h => by
    simp at h
    subst h
    exact ⟨rfl, rfl, rfl⟩
  | l :: ls, p, q, h => by
    simp only [List.foldlM_cons] at h
    cases hr : retireOne id l p with
    | none => simp [hr] at h
    | some r =>
      simp [hr] at h
      obtain ⟨h1, -, h3, -, -, -, h7⟩ := retireOne_some hr
      obtain ⟨i1, i3, i7⟩ := foldl_retire_keeps ls h
      exact ⟨i1.trans h1, i3.trans h3, i7.trans h7⟩

/-- `retireLanes` keeps a request's id, request and CTA. -/
theorem retireLanes_keeps {id : Nat} {done : List Nat} {p q : InFlight} (h : retireLanes id done p = some q) :
    q.id = p.id ∧ q.req = p.req ∧ q.cta = p.cta := by
  unfold retireLanes at h
  split at h
  · exact foldl_retire_keeps done h
  · cases h
    exact ⟨rfl, rfl, rfl⟩

theorem retire_filterMap (qs : List InFlight) (id : Nat) (done : List Nat) (l : Nat) :
    retire (qs.filterMap (retireLanes id done)) id l = qs.filterMap (retireLanes id (done ++ [l])) := by
  unfold retire
  rw [List.filterMap_filterMap]
  exact filterMap_congr' fun p _ => retireLanes_append id done l p

/-- A queue in which request `id` is found at index `i`, as `q`: the requests before it are others'. -/
structure Split (qs : List InFlight) (id i : Nat) (q : InFlight) : Prop where
  get : qs[i]? = some q
  self : q.id = id
  before : ∀ p ∈ qs.take i, p.id ≠ id

theorem split_of_find {m : Flat} {slot id i : Nat} {q : InFlight} (h : m.find slot id = some (i, q)) :
    Split (m.queue slot) id i q := by
  simp only [find, Option.map_eq_some_iff, Prod.mk.injEq] at h
  obtain ⟨i', hi', rfl, hq⟩ := h
  obtain ⟨hlt, hid, hbefore⟩ := List.findIdx?_eq_some_iff_getElem.1 hi'
  rw [getElem!_pos (m.queue slot) i' hlt] at hq
  subst hq
  refine ⟨List.getElem?_eq_getElem hlt, by simpa using hid, fun p hp => ?_⟩
  obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hp
  have hji : j < i' := by simp only [List.length_take] at hj; omega
  rw [List.getElem_take]
  simpa using hbefore j hji

theorem find_of_split {m : Flat} {slot id i : Nat} {q : InFlight} (hs : Split (m.queue slot) id i q) :
    m.find slot id = some (i, q) := by
  obtain ⟨hlt, hq⟩ := List.getElem?_eq_some_iff.1 hs.get
  simp only [find, Option.map_eq_some_iff, Prod.mk.injEq]
  refine ⟨i, ?_, rfl, ?_⟩
  · rw [List.findIdx?_eq_some_iff_getElem]
    refine ⟨hlt, by simp [hq, hs.self], fun j hj => ?_⟩
    have hmem : (m.queue slot)[j] ∈ (m.queue slot).take i := by
      have := List.getElem_mem (by simp only [List.length_take]; omega : j < ((m.queue slot).take i).length)
      rwa [List.getElem_take] at this
    simpa using hs.before _ hmem
  · rw [getElem!_pos (m.queue slot) i hlt, hq]

/-- Once lanes `done` of the request have performed, the request still in flight as `q'`: the same
requests before it, then `q'`. -/
theorem split_retired {qs : List InFlight} {id i : Nat} {q q' : InFlight} (hs : Split qs id i q)
    {done : List Nat} (hr : retireLanes id done q = some q') :
    Split (qs.filterMap (retireLanes id done)) id i q' ∧
      (qs.filterMap (retireLanes id done)).take i = qs.take i := by
  obtain ⟨hlt, hq⟩ := List.getElem?_eq_some_iff.1 hs.get
  have hsplit : qs = qs.take i ++ q :: qs.drop (i + 1) := by
    rw [← hq, ← List.drop_eq_getElem_cons hlt, List.take_append_drop]
  have hkeep : ∀ p ∈ qs.take i, retireLanes id done p = some p := by
    intro p hp
    have : (p.id == id) = false := by simpa using hs.before p hp
    simp [retireLanes, this]
  have hpre : (qs.take i).filterMap (retireLanes id done) = qs.take i := by
    rw [filterMap_congr' hkeep, List.filterMap_some]
  have hlen : (qs.take i).length = i := by
    simp only [List.length_take]; exact Nat.min_eq_left (Nat.le_of_lt hlt)
  have hf : qs.filterMap (retireLanes id done) =
      qs.take i ++ q' :: (qs.drop (i + 1)).filterMap (retireLanes id done) := by
    conv => lhs; rw [hsplit]
    simp only [List.filterMap_append, hpre, List.filterMap_cons, hr]
  have hid' : q'.id = id := (retireLanes_keeps hr).1.trans hs.self
  have htake : (qs.filterMap (retireLanes id done)).take i = qs.take i := by
    rw [hf, List.take_append_of_le_length (Nat.le_of_eq hlen.symm), List.take_of_length_le (Nat.le_of_eq hlen)]
  refine ⟨⟨?_, hid', ?_⟩, htake⟩
  · rw [hf, List.getElem?_append_right (Nat.le_of_eq hlen), hlen, Nat.sub_self]
    simp
  · rw [htake]; exact hs.before

theorem setStore_setStore (m : Flat) (l : Loc) (a b : Store) : (m.setStore l a).setStore l b = m.setStore l b := by
  cases hsp : l.space <;> simp [setStore, hsp, Array.set!_eq_setIfInBounds, Array.setIfInBounds_setIfInBounds]

theorem withRetired_queue {m : Flat} {slot id : Nat} {qs : List InFlight} {done : List Nat}
    (hq : m.queue slot = qs) (hsz : slot < m.queues.size) :
    (m.withRetired slot id qs done).queue slot = qs.filterMap (retireLanes id done) := by
  unfold withRetired
  split
  · next h =>
    rw [List.isEmpty_iff] at h
    subst h
    rw [hq, filterMap_congr' (g := some) (fun p _ => retireLanes_nil id p), List.filterMap_some]
  · rw [queue_set _ hsz]
    simp

theorem withRetired_store? (m : Flat) (slot id : Nat) (qs : List InFlight) (done : List Nat) (l : Loc) :
    (m.withRetired slot id qs done).store? l = m.store? l := by
  unfold withRetired; split
  · rfl
  · exact store?_eq rfl rfl l

/-- The request machine's commit of one more lane on the machine after lanes `done` is the machine
after `done` and that lane. -/
theorem commit_retired (m : Flat) (slot id : Nat) (qs : List InFlight) (done : List Nat) (l : Loc)
    (s' : Store) (lane : Nat) :
    { (m.withRetired slot id qs done).setStore l s' with
        queues := (m.withRetired slot id qs done).queues.set! slot
          (qs.filterMap (retireLanes id (done ++ [lane]))) } =
      (m.setStore l s').withRetired slot id qs (done ++ [lane]) := by
  have hne : (done ++ [lane]).isEmpty = false := by simp
  unfold withRetired
  rw [hne]
  cases hd : done.isEmpty <;> cases hsp : l.space <;>
    simp [setStore, hsp, Array.set!_eq_setIfInBounds, Array.setIfInBounds_setIfInBounds]

theorem blocker_none {ls : List Nat} {p q : InFlight} {lane : Nat} (h : mayBlock ls p q = false)
    (hl : lane ∈ ls) : blocker p q lane = none := by
  simp only [mayBlock, Bool.or_eq_false_iff, List.any_eq_false] at h
  obtain ⟨⟨hrel, hacq⟩, hov⟩ := h
  have h1 := hov lane hl
  unfold blocker
  by_cases hc : lane ∈ p.lanes
  · have hov' : overlap p q lane = false := by simpa [hc] using h1
    simp [hc, hov', hrel, hacq]
  · simp [hc]

theorem blocker_congr {p q q₀ : InFlight} {lane : Nat} (h1 : q.req = q₀.req) (h2 : q.cta = q₀.cta) :
    blocker p q lane = blocker p q₀ lane := by
  unfold blocker overlap
  rw [h1, h2]

/-- What a lane `performLane` performs satisfies: it is the request's, nothing in `cand` holds it
back, and its store is updated by `perform`. -/
theorem performLane_some {m m' : Flat} {cand : List InFlight} {q : InFlight} {lane : Nat} {b : List UInt8}
    (hp : m.performLane cand q lane = some (m', b)) :
    q.lanes.contains lane = true ∧ cand.findSome? (blocker · q lane) = none ∧
      ∃ l st s', loc q.cta q.req lane = some l ∧ m.store? l = some st ∧
        perform q.req lane l st = .ok (s', b) ∧ m' = (m.setStore l default).setStore l s' := by
  cases hcon : q.lanes.contains lane
  · have : lane ∉ q.lanes := by simpa using hcon
    simp [performLane, this] at hp
  cases hb : cand.findSome? (blocker · q lane) with
  | some w => simp [performLane, hcon, hb] at hp
  | none =>
  cases hl : loc q.cta q.req lane with
  | none => simp [performLane, hcon, hb, hl] at hp
  | some l =>
  cases hst : m.store? l with
  | none => simp [performLane, hcon, hb, hl, hst] at hp
  | some st =>
  cases hperf : perform q.req lane l st with
  | error e => simp [performLane, hcon, hb, hl, hst, hperf] at hp
  | ok r =>
  obtain ⟨s', bytes⟩ := r
  simp only [performLane, hcon, Bool.not_true, Bool.false_eq_true, ↓reduceIte, hb, hl, hst, hperf,
    Option.some.injEq, Prod.mk.injEq] at hp
  obtain ⟨rfl, rfl⟩ := hp
  refine ⟨?_, ?_, l, st, s', ?_, ?_, ?_, rfl⟩ <;> first | rfl | assumption

theorem performLane_queues {m m' : Flat} {cand : List InFlight} {q : InFlight} {lane : Nat} {b : List UInt8}
    (hp : m.performLane cand q lane = some (m', b)) : m'.queues = m.queues := by
  obtain ⟨-, -, l, -, s', -, -, -, rfl⟩ := performLane_some hp
  rw [(setStore_fields _ _ _).2.2, (setStore_fields _ _ _).2.2]

/-- A lane `performLane` performs is the request machine's `deliver` of it, from the machine after
the lanes before it. -/
theorem deliver_of_performLane {m : Flat} {slot id i : Nat} {qs : List InFlight} {q q₀ : InFlight}
    {done lanes : List Nat} {lane : Nat} {m' : Flat} {b : List UInt8}
    (hq : m.queue slot = qs) (hsz : slot < m.queues.size) (hs : Split qs id i q₀)
    (hr : retireLanes id done q₀ = some q) (hlane : lane ∈ lanes)
    (hp : m.performLane ((qs.take i).filter (mayBlock lanes · q₀)) q lane = some (m', b)) :
    (m.withRetired slot id qs done).deliver slot id lane =
      .ok (m'.withRetired slot id qs (done ++ [lane]), b) := by
  obtain ⟨hs', htake⟩ := split_retired hs hr
  have hqr := withRetired_queue (id := id) (done := done) hq hsz
  have hfind : (m.withRetired slot id qs done).find slot id = some (i, q) := by
    apply find_of_split; rw [hqr]; exact hs'
  obtain ⟨-, hreq, hcta⟩ := retireLanes_keeps hr
  obtain ⟨hcon, hb, l, st, s', hl, hst, hperf, rfl⟩ := performLane_some hp
  -- No request before `q` holds this lane back: those `mayBlock` keeps do not, by `hb`; the
  -- others cannot.
  have hnone : (qs.take i).findSome? (blocker · q lane) = none := by
    rw [List.findSome?_eq_none_iff] at hb ⊢
    intro p hp
    by_cases hm : mayBlock lanes p q₀ = true
    · exact hb p (List.mem_filter.2 ⟨hp, hm⟩)
    · rw [blocker_congr hreq hcta]
      exact blocker_none (by simpa using hm) hlane
  have hst' : (m.withRetired slot id qs done).store? l = some st := by
    rw [withRetired_store?]; exact hst
  simp only [deliver, hfind, deliverAt, hqr, htake, hcon, Bool.not_true, Bool.false_eq_true, ↓reduceIte,
    hnone, hl, hst', hperf]
  rw [commit_eq hst', hs'.self, retire_filterMap, commit_retired, setStore_setStore]

/-- `deliverLanes`' loop is `deliverSeq` from the machine after the lanes it has performed. -/
theorem deliverLanesGo_eq {slot id i : Nat} {qs : List InFlight} {q₀ : InFlight} {lanes : List Nat}
    (hs : Split qs id i q₀) :
    ∀ (ls : List Nat) (m : Flat) (q : InFlight) (done : List Nat) (acc : List (List UInt8)),
      (∀ x ∈ ls, x ∈ lanes) → m.queue slot = qs → slot < m.queues.size → retireLanes id done q₀ = some q →
      deliverLanesGo slot id qs ((qs.take i).filter (mayBlock lanes · q₀)) m q done acc ls =
        (deliverSeq (m.withRetired slot id qs done) slot id ls).map fun (m, bs) => (m, acc.reverse ++ bs)
  | [], m, q, done, acc, _, _, _, _ => by simp [deliverLanesGo, deliverSeq, Except.map]
  | l :: ls, m, q, done, acc, hin, hq, hsz, hr => by
    simp only [deliverLanesGo]
    cases hp : m.performLane ((qs.take i).filter (mayBlock lanes · q₀)) q l with
    | none => rfl
    | some r =>
      obtain ⟨m', b⟩ := r
      have hd := deliver_of_performLane hq hsz hs hr (hin l (by simp)) hp
      have hqs : m'.queues = m.queues := performLane_queues hp
      have hq' : m'.queue slot = qs := by rw [queue_eq hqs, hq]
      have hsz' : slot < m'.queues.size := by rw [hqs]; exact hsz
      simp only [deliverSeq, hd]
      cases hr' : retireOne id l q with
      | some q' =>
        simp only
        rw [deliverLanesGo_eq hs ls m' q' (done ++ [l]) (b :: acc) (fun x hx => hin x (by simp [hx])) hq' hsz'
          (by rw [← retireLanes_append, hr]; exact hr')]
        cases deliverSeq (m'.withRetired slot id qs (done ++ [l])) slot id ls <;>
          simp [Except.map, List.reverse_cons]
      | none =>
        simp only
        cases deliverSeq (m'.withRetired slot id qs (done ++ [l])) slot id ls <;>
          simp [Except.map]

theorem withRetired_nil (m : Flat) (slot id : Nat) (qs : List InFlight) : m.withRetired slot id qs [] = m := by
  simp [withRetired]

/-- `deliverLanes` is `deliverSeq`. -/
theorem deliverLanes_eq (m : Flat) (slot id : Nat) (lanes : List Nat) :
    m.deliverLanes slot id lanes = m.deliverSeq slot id lanes := by
  unfold deliverLanes
  split
  · rfl
  · next i q hf =>
    have hs := split_of_find hf
    have hsz : slot < m.queues.size := slot_lt (List.mem_of_getElem? hs.get)
    dsimp only
    rw [deliverLanesGo_eq hs lanes m q [] [] (fun _ h => h) rfl hsz (retireLanes_nil id q), withRetired_nil]
    cases deliverSeq m slot id lanes <;> simp [Except.map]

end NV.Flat
