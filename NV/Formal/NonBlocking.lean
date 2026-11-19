import NV.Formal.Outcomes
import NV.SASS.Sem

/-!
# Non-blocking runs

The timed engine applies a result's writeback (`Effect.reg`'s `wb`, a memory response, an async
operation's result) some time after the instruction issued, while the warp keeps issuing.
`Outcomes` reads every writeback as applied at once. `RunNB` is a warp's run in the engine's
reading: writebacks stay pending until they land, in any order, and an instruction issues only
when its footprint (`NV.SASS.Footprint`: what it reads, writes back and sets at issue) is clear of
every pending writeback's write set. That condition is the engine's hazard check: an instruction
issuing while one of its locations is pending is a stale read or a lost write on silicon (Huerta et
al. 2025 §4), and such a run is not covered.

`RunNB.blocking`: a non-blocking run is a blocking `Run` with the same memory, synchronization
and async acts, ending the same way up to a fault's message. So the engine's warps, paired with
the events the request machine performed (`Flat.sound`), form a candidate of `Outcomes`.

What the theorem assumes of the warp's step (`Footprinted`):

* the footprint is a function of the control state (the instruction at the program counter);
* the step's outcome depends only on the control state and the footprint's reads
  (`Footprint.SameOutcome`; `NV.SASS.Op.readsWithin`);
* the step changes, at issue, only the control state and the footprint's sets, and its writeback
  only the write set (`Footprint.Within`; `NV.SASS.Op.confined`, a theorem);
* a writeback keeps the control state, and what it leaves in the write set depends only on what
  the write set held (`WbLocal`).

A request is built from the state right after issue here; the engine builds it when the request
reads its sources, which is the same state if no instruction issued in between writes them. The
step is one function, so an instruction reading the clock (`Env.clock`, `Env.timeNs`), whose value
the engine sets per issue, is outside the theorem.
-/

namespace NV.Formal

open NV.SASS

/-- What the bridge assumes of a warp's step `step` and its footprints `fpOf`. -/
structure Footprinted (step : Warp → Effect Warp) (fpOf : Warp → Footprint) : Prop where
  fp_ctrl : ∀ w w', w.ctrl = w'.ctrl → fpOf w = fpOf w'
  reads : ∀ w w', w.ctrl = w'.ctrl → w.shape = w'.shape → Warp.AgreeOn (fpOf w).reads w w' →
    (fpOf w).SameOutcome w' (step w) (step w')
  within : ∀ w, (fpOf w).Within w (step w)
  shape : ∀ w, effectShape w.shape (step w)
  wb : ∀ w, effectWbLocal (fpOf w).writes (step w)

/-- A pending writeback: its write set and what it does. -/
abbrev Pending := Array SASS.Loc × (Warp → Warp)

/-- An instruction with footprint `fp` may issue: none of its locations is pending. -/
def Clear (fp : Footprint) (ps : List Pending) : Prop :=
  ∀ q ∈ ps, ∀ l ∈ q.1, l ∉ fp.reads ∧ l ∉ fp.writes ∧ l ∉ fp.sets

/-- How the blocking reading ends a non-blocking run ending `e`: the same way, a fault's message
aside (`SameOutcome` relates faults whatever their messages). -/
def End.Like : End → End → Prop
  | .fault _, .fault _ => True
  | e, e' => e = e'

theorem End.Like.refl : ∀ e, End.Like e e
  | .exit | .fault _ | .unsupported _ | .stop => by simp [End.Like]

variable (step : Warp → Effect Warp) (fpOf : Warp → Footprint)

/-- A run with pending writebacks: `RunNB s ps acts e` from state `s` with writebacks `ps` not yet
landed. Each issue is `Clear` of the pending write sets. -/
inductive RunNB : Warp → List Pending → List Act → End → Prop
  | exit {s ps} : step s = .exit → Clear (fpOf s) ps → RunNB s ps [] .exit
  | fault {s ps msg} : step s = .fault msg → Clear (fpOf s) ps → RunNB s ps [] (.fault msg)
  | stop {s ps} : RunNB s ps [] .stop
  | next {s s' ps acts e} : step s = .next s' → Clear (fpOf s) ps → RunNB s' ps acts e →
      RunNB s ps acts e
  | reg {s s' wb ps acts e} : step s = .reg s' wb → Clear (fpOf s) ps →
      RunNB s' (ps ++ [((fpOf s).writes, wb)]) acts e → RunNB s ps acts e
  | mem {s s' rf wb ps read acts e} : step s = .mem rf s' wb → Clear (fpOf s) ps →
      (∀ l, (rf s').active l → (read l).length = (rf s').readLen) → (rf s').unsupported read = false →
      RunNB s' (ps ++ [((fpOf s).writes, wb ((rf s').resp read))]) acts e →
      RunNB s ps (.mem (rf s') read :: acts) e
  | memUnsupported {s s' rf wb ps read} : step s = .mem rf s' wb → Clear (fpOf s) ps →
      (∀ l, (rf s').active l → (read l).length = (rf s').readLen) → (rf s').unsupported read = true →
      RunNB s ps [.fail (rf s') read] (failureOf (rf s'))
  | sync {s s' op inst ps acts e} : step s = .sync op s' → Clear (fpOf s) ps → RunNB s' ps acts e →
      RunNB s ps (.sync op inst :: acts) e
  | collect {s op mask contrib s' wb ps inst resp acts e} : step s = .collect op mask contrib s' wb →
      Clear (fpOf s) ps → RunNB s' (ps ++ [((fpOf s).writes, wb resp)]) acts e →
      RunNB s ps (.collect op mask contrib inst (.ok resp) :: acts) e
  | collectFail {s op mask contrib s' wb ps inst e} : step s = .collect op mask contrib s' wb →
      Clear (fpOf s) ps → RunNB s ps [.collect op mask contrib inst (.error e)] e
  | async {s op s' wb agent res ps acts e} : step s = .async op s' wb → Clear (fpOf s) ps →
      AgentRun (op s') agent (.ok res) → RunNB s' (ps ++ [((fpOf s).writes, wb res)]) acts e →
      RunNB s ps (.async (op s') agent :: acts) e
  | asyncFail {s op s' wb agent ps e} : step s = .async op s' wb → Clear (fpOf s) ps →
      AgentRun (op s') agent (.error e) → RunNB s ps [.async (op s') agent] e
  /-- A pending writeback lands, in any order. -/
  | land {s pre q post acts e} : RunNB (q.2 s) (pre ++ post) acts e → RunNB s (pre ++ q :: post) acts e

/-- A blocking state `t` is a non-blocking state `s` with its writebacks `ps` landed: the same
control state, the same contents outside the pending write sets and each pending write set's
contents as its writeback leaves them. The writebacks are local and their write sets disjoint. -/
structure Landed (t s : Warp) (ps : List Pending) : Prop where
  ctrl : t.ctrl = s.ctrl
  shape : t.shape = s.shape
  out : ∀ l, (∀ q ∈ ps, l ∉ q.1) → t.get l = s.get l
  pend : ∀ q ∈ ps, ∀ l ∈ q.1, t.get l = (q.2 s).get l
  wbLocal : ∀ q ∈ ps, WbLocal q.1 q.2
  frame : ∀ q ∈ ps, ∀ x l, l ∉ q.1 → (q.2 x).get l = x.get l
  apart : ps.Pairwise fun q q' => ∀ l ∈ q.1, l ∉ q'.1

namespace Landed

variable {t s : Warp} {ps : List Pending}

theorem init (s : Warp) : Landed s s [] :=
  ⟨rfl, rfl, fun _ _ => rfl, by simp, by simp, by simp, List.Pairwise.nil⟩

/-- A pending writeback leaves what is outside its write set as it was. -/
theorem agree_frame {q : Pending} (hf : ∀ x l, l ∉ q.1 → (q.2 x).get l = x.get l) {x y : Warp}
    {W : Array SASS.Loc} (hW : ∀ l ∈ W, l ∉ q.1) (h : Warp.AgreeOn W x y) : Warp.AgreeOn W (q.2 x) y :=
  fun l hl => (hf x l (hW l hl)).trans (h l hl)

/-- Landing one of the pending writebacks. -/
theorem land {pre post : List Pending} {q : Pending} (h : Landed t s (pre ++ q :: post)) :
    Landed t (q.2 s) (pre ++ post) := by
  have hq : q ∈ pre ++ q :: post := by simp
  have sub : ∀ q' ∈ pre ++ post, q' ∈ pre ++ q :: post := fun q' h' => by
    simp only [List.mem_append, List.mem_cons] at h' ⊢
    rcases h' with h' | h'
    · exact .inl h'
    · exact .inr (.inr h')
  have apart' : ∀ q' ∈ pre ++ post, ∀ l ∈ q.1, l ∉ q'.1 := by
    intro q' hq' l hl hl'
    have hp := h.apart
    rw [List.pairwise_append, List.pairwise_cons] at hp
    simp only [List.mem_append] at hq'
    rcases hq' with hq' | hq'
    · exact hp.2.2 q' hq' q (by simp) l hl' hl
    · exact hp.2.1.1 q' hq' l hl hl'
  refine ⟨(h.ctrl).trans ((h.wbLocal q hq).ctrl s).symm, h.shape.trans ((h.wbLocal q hq).shape s).symm,
    fun l hl => ?_, fun q' hq' l hl => ?_,
    fun q' hq' => h.wbLocal q' (sub q' hq'), fun q' hq' => h.frame q' (sub q' hq'), ?_⟩
  · by_cases hlq : l ∈ q.1
    · exact h.pend q hq l hlq
    · rw [h.frame q hq s l hlq]
      exact h.out l fun q' hq'' => by
        simp only [List.mem_append, List.mem_cons] at hq''
        rcases hq'' with h₁ | rfl | h₂
        · exact hl q' (by simp [h₁])
        · exact hlq
        · exact hl q' (by simp [h₂])
  · rw [h.pend q' (sub q' hq') l hl]
    exact (h.wbLocal q' (sub q' hq')).agree s (q.2 s) ((h.wbLocal q hq).shape s).symm
      (fun l' hl' => (h.frame q hq s l' fun hq₁ => apart' q' hq' l' hq₁ hl').symm) l hl
  · have hp := h.apart
    rw [List.pairwise_append, List.pairwise_cons] at hp
    exact List.pairwise_append.2 ⟨hp.1, hp.2.1.2, fun a ha b hb => hp.2.2 a ha b (by simp [hb])⟩

/-- The footprint's reads are not pending, so the blocking state agrees with the issuing one on
them. -/
theorem reads (h : Landed t s ps) {fp : Footprint} (hc : Clear fp ps) : Warp.AgreeOn fp.reads s t :=
  fun l hl => (h.out l fun q hq hlq => (hc q hq l hlq).1 hl).symm

/-- After an issue from `s` to `s'` and from `t` to `t'` (`Carried`), the blocking state still
has the pending writebacks landed. -/
theorem issue (h : Landed t s ps) {fp : Footprint} (hc : Clear fp ps) {s' t' : Warp}
    (hs : ∀ l, l ∉ fp.sets → s'.get l = s.get l) (ht : fp.Carried s' t t') (hss : s'.shape = s.shape)
    (hts : t'.shape = t.shape) : Landed t' s' ps := by
  obtain ⟨hctrl, hsets, hother⟩ := ht
  have hout : ∀ l, (∀ q ∈ ps, l ∉ q.1) → t'.get l = s'.get l := fun l hl => by
    by_cases hl' : l ∈ fp.sets
    · exact hsets l hl'
    · rw [hother l hl', hs l hl']; exact h.out l hl
  refine ⟨hctrl, hts.trans (h.shape.trans hss.symm), hout, fun q hq l hl => ?_, h.wbLocal, h.frame,
    h.apart⟩
  have hns : l ∉ fp.sets := (hc q hq l hl).2.2
  rw [hother l hns, h.pend q hq l hl]
  exact (h.wbLocal q hq).agree s s' hss.symm (fun l' hl' => (hs l' (hc q hq l' hl').2.2).symm) l hl

/-- After an issue that leaves writeback `f` on the footprint's write set pending. -/
theorem push (h : Landed t s ps) {fp : Footprint} (hc : Clear fp ps) {f : Warp → Warp}
    (hl : WbLocal fp.writes f) (hf : ∀ x l, l ∉ fp.writes → (f x).get l = x.get l) :
    Landed (f t) s (ps ++ [(fp.writes, f)]) := by
  have hW : Warp.AgreeOn fp.writes t s := fun l hl' =>
    h.out l fun q hq hlq => (hc q hq l hlq).2.1 hl'
  refine ⟨(hl.ctrl t).trans h.ctrl, (hl.shape t).trans h.shape, fun l hl' => ?_, fun q hq l hlq => ?_, ?_, ?_, ?_⟩
  · have : l ∉ fp.writes := hl' (fp.writes, f) (by simp)
    rw [hf t l this]
    exact h.out l fun q hq => hl' q (by simp [hq])
  · simp only [List.mem_append, List.mem_singleton] at hq
    rcases hq with hq | rfl
    · rw [hf t l fun hw => (hc q hq l hlq).2.1 hw]; exact h.pend q hq l hlq
    · exact hl.agree t s h.shape hW l hlq
  · intro q hq; simp only [List.mem_append, List.mem_singleton] at hq
    rcases hq with hq | rfl
    · exact h.wbLocal q hq
    · exact hl
  · intro q hq; simp only [List.mem_append, List.mem_singleton] at hq
    rcases hq with hq | rfl
    · exact h.frame q hq
    · exact hf
  · exact List.pairwise_append.2 ⟨h.apart, by simp, fun q hq q' hq' l hl hl' => by
      simp only [List.mem_singleton] at hq'; subst hq'
      exact (hc q hq l hl).2.1 hl'⟩

end Landed

variable {step fpOf}

/-- **A non-blocking run is a blocking run** with the same acts, from any state that is its
start with the pending writebacks landed, ending the same way up to a fault's message. -/
theorem RunNB.blocking (hF : Footprinted step fpOf) {s t : Warp} {ps : List Pending}
    {acts : List Act} {e : End} (h : RunNB step fpOf s ps acts e) (hl : Landed t s ps) :
    ∃ e', Run step t acts e' ∧ End.Like e e' := by
  induction h generalizing t with
  | @exit s ps hs hc =>
    have hso := hF.reads s t hl.ctrl.symm hl.shape.symm (hl.reads hc)
    rw [hs] at hso
    refine ⟨.exit, Run.exit ?_, .refl _⟩
    revert hso; cases step t <;> simp [Footprint.SameOutcome]
  | @fault s ps msg hs hc =>
    have hso := hF.reads s t hl.ctrl.symm hl.shape.symm (hl.reads hc)
    rw [hs] at hso
    revert hso; cases ht : step t <;> simp [Footprint.SameOutcome]
    exact ⟨_, Run.fault ht, by simp [End.Like]⟩
  | stop => exact ⟨.stop, Run.stop, .refl _⟩
  | @next s s' ps acts e hs hc _ ih =>
    have hso := hF.reads s t hl.ctrl.symm hl.shape.symm (hl.reads hc)
    have hw := hF.within s
    have hss := hF.shape s
    have hts := hF.shape t
    rw [hs] at hso hw hss
    revert hso; cases ht : step t <;> simp [Footprint.SameOutcome]
    rw [ht] at hts
    intro hcar
    obtain ⟨e', hr, he⟩ := ih (hl.issue hc hw hcar hss hts)
    exact ⟨e', Run.next ht hr, he⟩
  | @reg s s' wb ps acts e hs hc _ ih =>
    have hso := hF.reads s t hl.ctrl.symm hl.shape.symm (hl.reads hc)
    have hw := hF.within s
    have hss := hF.shape s
    have hts := hF.shape t
    have hloc := hF.wb s
    rw [hs] at hso hw hloc hss
    revert hso; cases ht : step t <;> simp [Footprint.SameOutcome]
    rw [ht] at hts
    rename_i t' wb'
    intro hcar hwb
    obtain ⟨e', hr, he⟩ := ih ((hl.issue hc hw.1 hcar hss hts).push hc hloc hw.2)
    refine ⟨e', Run.reg ht ?_, he⟩
    rwa [show wb' = wb from funext fun x => (hwb x).symm]
  | @mem s s' rf wb ps read acts e hs hc hlen hu _ ih =>
    have hso := hF.reads s t hl.ctrl.symm hl.shape.symm (hl.reads hc)
    have hw := hF.within s
    have hss := hF.shape s
    have hts := hF.shape t
    have hloc := hF.wb s
    rw [hs] at hso hw hloc hss
    revert hso; cases ht : step t <;> simp [Footprint.SameOutcome]
    rw [ht] at hts
    rename_i rf' t' wb'
    intro hcar hrf hwb
    have hl' := hl.issue hc hw.1 hcar hss hts
    have hreq : rf s' = rf' t' := hrf s' t' (hl'.reads hc)
    obtain ⟨e', hr, he⟩ := ih (hl'.push hc (hloc _) (hw.2 _))
    refine ⟨e', ?_, he⟩
    rw [hreq] at hlen hu hr ⊢
    refine Run.mem ht hlen hu ?_
    rwa [show wb' ((rf' t').resp read) = wb ((rf' t').resp read) from funext fun x => (hwb _ x).symm]
  | @memUnsupported s s' rf wb ps read hs hc hlen hu =>
    have hso := hF.reads s t hl.ctrl.symm hl.shape.symm (hl.reads hc)
    have hw := hF.within s
    have hss := hF.shape s
    have hts := hF.shape t
    rw [hs] at hso hw hss
    revert hso; cases ht : step t <;> simp [Footprint.SameOutcome]
    rw [ht] at hts
    rename_i rf' t' wb'
    intro hcar hrf _
    have hreq : rf s' = rf' t' := hrf s' t' ((hl.issue hc hw.1 hcar hss hts).reads hc)
    rw [hreq] at hlen hu ⊢
    exact ⟨_, Run.memUnsupported ht hlen hu, .refl _⟩
  | @sync s s' op inst ps acts e hs hc _ ih =>
    have hso := hF.reads s t hl.ctrl.symm hl.shape.symm (hl.reads hc)
    have hw := hF.within s
    have hss := hF.shape s
    have hts := hF.shape t
    rw [hs] at hso hw hss
    revert hso; cases ht : step t <;> simp [Footprint.SameOutcome]
    rw [ht] at hts
    rintro rfl hcar
    obtain ⟨e', hr, he⟩ := ih (hl.issue hc hw hcar hss hts)
    exact ⟨e', Run.sync ht hr, he⟩
  | @collect s op mask contrib s' wb ps inst resp acts e hs hc _ ih =>
    have hso := hF.reads s t hl.ctrl.symm hl.shape.symm (hl.reads hc)
    have hw := hF.within s
    have hss := hF.shape s
    have hts := hF.shape t
    have hloc := hF.wb s
    rw [hs] at hso hw hloc hss
    revert hso; cases ht : step t <;> simp [Footprint.SameOutcome]
    rw [ht] at hts
    rename_i t' wb'
    rintro rfl rfl rfl hcar hwb
    obtain ⟨e', hr, he⟩ := ih ((hl.issue hc hw.1 hcar hss hts).push hc (hloc _) (hw.2 _))
    refine ⟨e', Run.collect ht ?_, he⟩
    rwa [show wb' resp = wb resp from funext fun x => (hwb _ x).symm]
  | @collectFail s op mask contrib s' wb ps inst e hs hc =>
    have hso := hF.reads s t hl.ctrl.symm hl.shape.symm (hl.reads hc)
    rw [hs] at hso
    revert hso; cases ht : step t <;> simp [Footprint.SameOutcome]
    rintro rfl rfl rfl _ _
    exact ⟨_, Run.collectFail ht, .refl _⟩
  | @async s op s' wb agent res ps acts e hs hc ha _ ih =>
    have hso := hF.reads s t hl.ctrl.symm hl.shape.symm (hl.reads hc)
    have hw := hF.within s
    have hss := hF.shape s
    have hts := hF.shape t
    have hloc := hF.wb s
    rw [hs] at hso hw hloc hss
    revert hso; cases ht : step t <;> simp [Footprint.SameOutcome]
    rw [ht] at hts
    rename_i op' t' wb'
    intro hcar hop hwb
    have hl' := hl.issue hc hw.1 hcar hss hts
    have hreq : op s' = op' t' := hop s' t' (hl'.reads hc)
    obtain ⟨e', hr, he⟩ := ih (hl'.push hc (hloc _) (hw.2 _))
    rw [hreq] at ha ⊢
    refine ⟨e', Run.async ht ha ?_, he⟩
    rwa [show wb' res = wb res from funext fun x => (hwb _ x).symm]
  | @asyncFail s op s' wb agent ps e hs hc ha =>
    have hso := hF.reads s t hl.ctrl.symm hl.shape.symm (hl.reads hc)
    have hw := hF.within s
    have hss := hF.shape s
    have hts := hF.shape t
    rw [hs] at hso hw hss
    revert hso; cases ht : step t <;> simp [Footprint.SameOutcome]
    rw [ht] at hts
    rename_i op' t' wb'
    intro hcar hop _
    have hreq : op s' = op' t' := hop s' t' ((hl.issue hc hw.1 hcar hss hts).reads hc)
    rw [hreq] at ha ⊢
    exact ⟨e, Run.asyncFail ht ha, .refl _⟩
  | land _ ih => exact ih hl.land

/-- The instruction a warp runs next is a function of its control state. -/
theorem _root_.NV.SASS.Warp.current_eq {w w' : Warp} (h : w.ctrl = w'.ctrl) :
    w.current = w'.current := by
  simp only [Warp.ctrl, Prod.mk.injEq] at h
  obtain ⟨h₁, h₂, h₃, h₄⟩ := h
  have : w = { w' with r := w.r, ur := w.ur, p := w.p, up := w.up, b := w.b, printf := w.printf,
                         barResult := w.barResult, printing := w.printing } := by
    cases w; cases w'; simp_all
  rw [this]; rfl

/-- A warp of a kernel decoded as `ops`, in environment `env`: it runs the instruction at its
program counter (`Warp.current`), as the engine issues it. -/
def warpStep (ops : Nat → Op) (env : Env) (w : Warp) : Effect Warp :=
  match w.current with
  | some (pc, m) => (ops pc).run env w m
  | none => .exit

def warpFp (ops : Nat → Op) (w : Warp) : Footprint :=
  match w.current with
  | some (pc, _) => (ops pc).fp
  | none => {}

/-- A kernel's warps meet the bridge's assumptions: every instruction's outcome depends only on its
reads (`Op.readsWithin`), its writebacks are local (`Op.wbLocal`), and confinement and shape hold
(`Op.confined`, `Op.shape`), all theorems of the SASS semantics. -/
theorem footprinted (ops : Nat → Op) (env : Env) :
    Footprinted (warpStep ops env) (warpFp ops) where
  fp_ctrl w w' h := by simp only [warpFp, Warp.current_eq h]
  reads w w' h hs ha := by
    simp only [warpStep, warpFp, Warp.current_eq h] at ha ⊢
    split
    · next pc m hc => simp only [hc] at ha; exact (ops pc).readsWithin env w w' m h hs ha
    · simp [Footprint.SameOutcome]
  within w := by
    simp only [warpStep, warpFp]
    split
    · exact (ops _).confined env w _
    · trivial
  shape w := by
    simp only [warpStep]
    split
    · exact (ops _).shape env w _
    · trivial
  wb w := by
    simp only [warpStep, warpFp]
    split
    · exact (ops _).wbLocal env w _
    · trivial

/-- From an initial state with nothing pending, a non-blocking run is a run of `Outcomes`, ending
the same way up to a fault's message. -/
theorem RunNB.run (hF : Footprinted step fpOf) {s : Warp} {acts : List Act} {e : End}
    (h : RunNB step fpOf s [] acts e) : ∃ e', Run step s acts e' ∧ End.Like e e' :=
  h.blocking hF (Landed.init s)

end NV.Formal
