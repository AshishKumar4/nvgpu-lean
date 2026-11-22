import NV.Memory.Sound

/-!
# Shared memory performed by its SM

`Flat.deliverShared` (the request machine's step for a shared-memory lane its SM performed) is
`Flat.deliver` once the machine's shared memory is the SMs' (`deliverShared_spec`): the run of
the request machine whose shared memory the SMs hold, each CTA's performed in the order its SM
performed it, is a run of `Flat` (`Mem.sound`), and its reads are the SMs'.
-/

namespace NV.Flat

/-- `m` with `sh` as its shared memory. -/
def withShared (m : Flat) (sh : Array Store) : Flat := { m with shared := sh }

theorem withShared_queue (m : Flat) (sh : Array Store) (s : Nat) : (m.withShared sh).queue s = m.queue s := rfl

theorem withShared_find (m : Flat) (sh : Array Store) (slot id : Nat) :
    (m.withShared sh).find slot id = m.find slot id := rfl

/-- **A shared lane performed by its SM is a delivery.** Where `deliverShared` accepts lane `lane`
of request `id`, and the SM's copy `s` of the lane's CTA's shared memory performs the lane as
`perform` does, `deliver` on the machine with the SMs' shared memory accepts it, returns the SM's
bytes, and leaves the machine `deliverShared` left with the SM's new store. -/
theorem deliverShared_spec {m m' : Flat} {sh : Array Store} {slot id lane i : Nat} {q : InFlight}
    (h : m.deliverShared slot id lane = .ok m') (hf : m.find slot id = some (i, q)) {l : Loc}
    (hl : loc q.cta q.req lane = some l) {s s' : Store} {bytes : List UInt8}
    (hs : sh[l.owner]? = some s) (hp : perform q.req lane l s = .ok (s', bytes)) :
    (m.withShared sh).deliver slot id lane = .ok (m'.withShared (sh.set! l.owner s'), bytes) := by
  simp only [deliverShared, hf, deliverSharedAt] at h
  split at h
  · cases h
  · next hc =>
    split at h
    · cases h
    · next hb =>
      rw [hl] at h
      simp only at h
      split at h
      · cases h
      · next hsp =>
        simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hsp
        simp only [Except.ok.injEq] at h
        subst h
        have hst : (m.withShared sh).store? l = some s := by
          simp [store?, withShared, hsp, hs]
        simp only [deliver, withShared_find, hf, deliverAt, withShared_queue, hc, hb, hl, hst, hp]
        simp [setStore, hsp, withShared, Array.set!_eq_setIfInBounds, Array.setIfInBounds_setIfInBounds]

end NV.Flat
