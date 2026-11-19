import NV.Formal.NonBlocking
import NV.Formal.Sim
import NV.Machine.Trace

/-!
# Checking the engine's trace of a warp

`Trace.check` replays a warp's trace (`NV.TraceEv`: the engine's issues, with what each effect took
from outside the warp, and its writebacks landing) against the warp's step. Each issue must be clear
of the pending write sets (the engine's hazard check), and each landing must name a pending
writeback and leave in its write set what the engine's warp holds there (`agrees`), so a result the
engine computed differently from the step (a late read, a wrong accumulator chain) fails. When the replay agrees, it returns the acts and how the run ended, and `check_sound`
makes that a non-blocking run; `RunNB.run` then makes it a blocking `Run` of `Outcomes`.

An asynchronous operation's agent is rebuilt from what its reads returned (`agentOf`, a run of the
operation: `agentOf_ok`), and its result must be the one the engine applied.
-/

namespace NV.Formal

open NV.SASS

/-- `clear fp ps`: `Clear`, decided. -/
def clear (fp : Footprint) (ps : List Pending) : Bool :=
  ps.all fun q => q.1.all fun l => !(fp.reads.contains l || fp.writes.contains l || fp.sets.contains l)

theorem clear_sound {fp : Footprint} {ps : List Pending} (h : clear fp ps = true) : Clear fp ps := by
  intro q hq l hl
  simp only [clear, List.all_eq_true, Array.all_eq_true', Bool.not_eq_true', Bool.or_eq_false_iff] at h
  obtain ⟨⟨h₁, h₂⟩, h₃⟩ := h q hq l hl
  refine ⟨fun m => ?_, fun m => ?_, fun m => ?_⟩
  · rw [Array.contains_iff_mem.2 m] at h₁; cases h₁
  · rw [Array.contains_iff_mem.2 m] at h₂; cases h₂
  · rw [Array.contains_iff_mem.2 m] at h₃; cases h₃

/-- Every active lane's reads returned the request's length. -/
def readsOk (r : Req) (read : Nat → List UInt8) : Bool :=
  (List.range r.addrs.size).all fun l => !r.active l || (read l).length == r.readLen

theorem readsOk_sound {r : Req} {read : Nat → List UInt8} (h : readsOk r read = true) :
    ∀ l, r.active l = true → (read l).length = r.readLen := by
  intro l hl
  simp only [readsOk, List.all_eq_true, List.mem_range, Bool.or_eq_true, Bool.not_eq_true',
    beq_iff_eq] at h
  rcases h l (active_lt hl) with h | h
  · rw [hl] at h; cases h
  · exact h

/-- The requests' reads returned what they must (`ReqOk`), decided. -/
def reqOk (r : Req) (read : Nat → List UInt8) : Bool := readsOk r read && !r.unsupported read

theorem reqOk_sound {r : Req} {read : Nat → List UInt8} (h : reqOk r read = true) : ReqOk r read := by
  simp only [reqOk, Bool.and_eq_true, Bool.not_eq_true'] at h
  exact ⟨readsOk_sound h.1, h.2⟩

def actsOk (acts : List AAct) : Bool :=
  acts.all fun | .mem r read => reqOk r read | _ => true

theorem actsOk_sound {acts : List AAct} (h : actsOk acts = true) :
    ∀ r read, AAct.mem r read ∈ acts → ReqOk r read := by
  intro r read hm
  simp only [actsOk, List.all_eq_true] at h
  exact reqOk_sound (h _ hm)

/-- What an agent's `i`-th request returned to lane `l`. -/
def readOf (rd : Array (Array ByteArray)) (i l : Nat) : List UInt8 := ((rd.getD i #[]).getD l .empty).toList

/-- A copy's agent from its reads: its moves' loads, then its signals. -/
def copyAgent (c : Async.Copy) (rd : Array (Array ByteArray)) : Option (List AAct × Except End Resp) :=
  let acts := copyActs c (readOf rd) fun i => readOf rd (c.moves.size + i)
  if actsOk acts then some (acts, .ok #[]) else none

/-- An asynchronous operation's agent, rebuilt from what its reads returned, in the agent's order
(a tensor copy's descriptor first). -/
def agentOf : Async.Op → Array (Array ByteArray) → Option (List AAct × Except End Resp)
  | .copy c, rd => copyAgent c rd
  | .tensor t, rd =>
    if !reqOk t.desc (readOf rd 0) then none else
    match t.build (t.desc.resp (readOf rd 0)) with
    | .ok c => (copyAgent c (rd.extract 1 rd.size)).map fun r => (.mem t.desc (readOf rd 0) :: r.1, r.2)
    | .error e => some ([.mem t.desc (readOf rd 0)], .error (.fault e))
  | .mma mm, rd =>
    if (List.finRange mm.loads.size).all fun i => reqOk mm.loads[i] (readOf rd i) then
      some (mm.loads.toList.zipIdx.map (fun (r, i) => .mem r (readOf rd i)),
        .ok (mm.result ((mm.loads.toList.zipIdx.map fun (r, i) => r.resp (readOf rd i)).foldl (· ++ ·) #[])
          mm.c0))
    else none
  -- A tensor-memory MMA's run is not rebuilt yet (`AgentRun` has no case for it): a trace that
  -- issues one is not checked.
  | .mmaT _, _ => none
  | .prefetch _, _ => some ([], .ok #[])
  | .fault msg, _ => some ([], .error (.fault msg))

theorem copyAgent_ok {c : Async.Copy} {rd : Array (Array ByteArray)} {ag : List AAct}
    {r : Except End Resp} (h : copyAgent c rd = some (ag, r)) : AgentRun (.copy c) ag r := by
  simp only [copyAgent] at h
  split at h
  · next ho =>
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact .copy (actsOk_sound ho)
  · cases h

/-- **The rebuilt agents are runs** of their operations. -/
theorem agentOf_ok {o : Async.Op} {rd : Array (Array ByteArray)} {ag : List AAct}
    {r : Except End Resp} (h : agentOf o rd = some (ag, r)) : AgentRun o ag r := by
  cases o with
  | copy c => exact copyAgent_ok h
  | tensor t =>
    simp only [agentOf] at h
    split at h
    · cases h
    · next hd =>
      simp only [Bool.not_eq_true', Bool.not_eq_false] at hd
      split at h
      · next c hb =>
        cases hc : copyAgent c (rd.extract 1 rd.size) with
        | none => rw [hc] at h; cases h
        | some p =>
          rw [hc] at h
          simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨rfl, rfl⟩ := h
          exact .tensor (reqOk_sound hd) hb (copyAgent_ok hc)
      · next e hb =>
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        exact .tensorFail (reqOk_sound hd) hb
  | mma mm =>
    simp only [agentOf] at h
    split at h
    · next ho =>
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      refine .mma fun i hi => reqOk_sound ?_
      simp only [List.all_eq_true, List.mem_finRange, true_implies] at ho
      exact ho ⟨i, hi⟩
    · cases h
  | mmaT mm => simp [agentOf] at h
  | prefetch ls =>
    simp only [agentOf, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact .prefetch
  | fault msg =>
    simp only [agentOf, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact .fault

namespace Trace

variable (step : Warp → Effect Warp) (fpOf : Warp → Footprint)

/-- The bytes lane `l` read. -/
def bytes (rd : Array ByteArray) (l : Nat) : List UInt8 := (rd.getD l .empty).toList

/-- Acts `a` before a replayed rest. -/
def cons (a : Act) : Option (List Act × End) → Option (List Act × End) :=
  Option.map fun r => (a :: r.1, r.2)

theorem cons_some {a : Act} {r : Option (List Act × End)} {acts : List Act} {e : End}
    (h : cons a r = some (acts, e)) : ∃ rest, acts = a :: rest ∧ r = some (rest, e) := by
  cases r with
  | none => cases h
  | some p =>
    obtain ⟨rest, e'⟩ := p
    simp only [cons, Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨rest, rfl, rfl⟩

/-- An asynchronous operation's replay: its agent, rebuilt from what its reads returned, must end
as the engine's did (the same result, or both failing), then the rest `k` from its result, pending
under the engine's id. -/
def asyncStep (o : Async.Op) (rd : Array (Array ByteArray)) (res : Except String Resp)
    (wb : Option Nat) (k : Nat → Resp → Option (List Act × End)) : Option (List Act × End) :=
  match agentOf o rd, res, wb with
  | some (ag, .ok r), .ok r', some id => if r' == r then cons (.async o ag) (k id r) else none
  | some (ag, .error e), .error _, _ => some ([.async o ag], e)
  | _, _, _ => none

theorem asyncStep_some {o : Async.Op} {rd : Array (Array ByteArray)} {res : Except String Resp}
    {wb : Option Nat} {k : Nat → Resp → Option (List Act × End)} {acts : List Act} {e : End}
    (h : asyncStep o rd res wb k = some (acts, e)) :
    (∃ ag r id rest, agentOf o rd = some (ag, .ok r) ∧ acts = .async o ag :: rest ∧ k id r = some (rest, e)) ∨
      ∃ ag, agentOf o rd = some (ag, .error e) ∧ acts = [.async o ag] := by
  unfold asyncStep at h
  rcases ha : agentOf o rd with _ | ⟨ag, e' | r⟩ <;> cases res <;> cases wb <;>
    simp only [ha, reduceCtorEq, Option.some.injEq, Prod.mk.injEq] at h
  all_goals first
    | (obtain ⟨rfl, rfl⟩ := h; exact .inr ⟨ag, rfl, rfl⟩)
    | (split at h
       · obtain ⟨rest, rfl, h'⟩ := cons_some h
         exact .inl ⟨ag, r, _, rest, rfl, rfl, h'⟩
       · cases h)

/-- The engine's warp agrees with the replay on a landed write set: each compared location holds
the words the engine recorded (their `digest`). -/
def agrees (ls : Array SASS.Loc) (w : Warp) (vals : Array (Option UInt64)) : Bool :=
  vals.size == ls.size && (ls.zip vals).all fun (l, v) => v.all (· == digest (w.get l))

/-- Replay `tr` from state `s`, with writebacks `ps` pending under their ids. Whatever follows an
exit or a fault is ignored. -/
def check : Warp → List (Nat × Pending) → List TraceEv → Option (List Act × End)
  | _, _, [] => some ([], .stop)
  | s, ps, .land id vals :: tr =>
    let k := ps.findIdx (·.1 == id)
    if h : k < ps.length then
      if agrees ps[k].2.1 (ps[k].2.2 s) vals then check (ps[k].2.2 s) (ps.eraseIdx k) tr else none
    else none
  | s, ps, .issue pc c wb :: tr =>
    if !((s.current.map (·.1)).all (· == pc) && clear (fpOf s) (ps.map (·.2))) then none else
    match step s, c, wb with
    | .exit, .none, _ => some ([], .exit)
    | .fault msg, .none, _ => some ([], .fault msg)
    | .next s', .none, _ => check s' ps tr
    | .reg s' f, .none, some id => check s' (ps ++ [(id, ((fpOf s).writes, f))]) tr
    | .mem rf s' f, .mem rd, wb =>
      if !readsOk (rf s') (bytes rd) then none
      else if (rf s').unsupported (bytes rd) then some ([.fail (rf s') (bytes rd)], failureOf (rf s'))
      else match wb with
        | some id => cons (.mem (rf s') (bytes rd))
            (check s' (ps ++ [(id, ((fpOf s).writes, f ((rf s').resp (bytes rd))))]) tr)
        | none => none
    | .sync op s', .sync inst, _ => cons (.sync op inst) (check s' ps tr)
    | .collect op mask contrib s' f, .collect inst (.ok resp), some id =>
      cons (.collect op mask contrib inst (.ok resp))
        (check s' (ps ++ [(id, ((fpOf s).writes, f resp))]) tr)
    | .collect op mask contrib _ _, .collect inst (.error msg), _ =>
      some ([.collect op mask contrib inst (.error (.fault msg))], .fault msg)
    | .async op s' f, .async rd res, wb =>
      asyncStep (op s') rd res wb fun id r => check s' (ps ++ [(id, ((fpOf s).writes, f r))]) tr
    | _, _, _ => none

theorem map_snd_eraseIdx {ps : List (Nat × Pending)} {k : Nat} (h : k < ps.length) :
    ps.map (·.2) = (ps.take k).map (·.2) ++ ps[k].2 :: (ps.drop (k + 1)).map (·.2) ∧
      (ps.eraseIdx k).map (·.2) = (ps.take k).map (·.2) ++ (ps.drop (k + 1)).map (·.2) := by
  refine ⟨?_, by rw [List.eraseIdx_eq_take_drop_succ, List.map_append]⟩
  rw [← List.map_cons, ← List.map_append, ← List.drop_eq_getElem_cons h, List.take_append_drop]

/-- **A trace that replays is a non-blocking run** of the warp, with the acts it returned. -/
theorem check_sound :
    ∀ (tr : List TraceEv) (s : Warp) (ps : List (Nat × Pending)) (acts : List Act) (e : End),
      check step fpOf s ps tr = some (acts, e) → RunNB step fpOf s (ps.map (·.2)) acts e
  | [], s, ps, acts, e, h => by
    simp only [check, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact .stop
  | .land id vals :: tr, s, ps, acts, e, h => by
    simp only [check] at h
    split at h
    · next hk =>
      split at h
      · have ih := check_sound tr _ _ acts e h
        obtain ⟨h₁, h₂⟩ := map_snd_eraseIdx hk
        rw [h₂] at ih
        rw [h₁]
        exact .land ih
      · cases h
    · cases h
  | .issue pc c wb :: tr, s, ps, acts, e, h => by
    simp only [check] at h
    split at h
    · cases h
    · next hc =>
      simp only [Bool.not_eq_true', Bool.not_eq_false, Bool.and_eq_true] at hc
      have hcl := clear_sound hc.2
      split at h
      · next hs =>
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        exact .exit hs hcl
      · next hs =>
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        exact .fault hs hcl
      · next hs => exact .next hs hcl (check_sound tr _ _ acts e h)
      · next hs =>
        have ih := check_sound tr _ _ acts e h
        rw [List.map_append] at ih
        exact .reg hs hcl ih
      · next hs =>
        split at h
        · cases h
        · next hr =>
          simp only [Bool.not_eq_true', Bool.not_eq_false] at hr
          split at h
          · next hu =>
            simp only [Option.some.injEq, Prod.mk.injEq] at h
            obtain ⟨rfl, rfl⟩ := h
            exact .memUnsupported hs hcl (readsOk_sound hr) hu
          · next hu =>
            split at h
            · obtain ⟨rest, rfl, h'⟩ := cons_some h
              have ih := check_sound tr _ _ rest e h'
              rw [List.map_append] at ih
              exact .mem hs hcl (readsOk_sound hr) (by simpa using hu) ih
            · cases h
      · next hs =>
        obtain ⟨rest, rfl, h'⟩ := cons_some h
        exact .sync hs hcl (check_sound tr _ _ rest e h')
      · next hs =>
        obtain ⟨rest, rfl, h'⟩ := cons_some h
        have ih := check_sound tr _ _ rest e h'
        rw [List.map_append] at ih
        exact .collect hs hcl ih
      · next hs =>
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        exact .collectFail hs hcl
      · next hs =>
        rcases asyncStep_some h with ⟨ag, r, id, rest, ha, rfl, h'⟩ | ⟨ag, ha, rfl⟩
        · have ih := check_sound tr _ _ rest e h'
          rw [List.map_append] at ih
          exact .async hs hcl (agentOf_ok ha) ih
        · exact .asyncFail hs hcl (agentOf_ok ha)
      · cases h

/-- **A replayed trace from a fresh warp is a run of `Outcomes`**, ending the same way up to a
fault's message, for a step that meets the bridge's assumptions. -/
theorem check_run (hF : Footprinted step fpOf) {tr : List TraceEv} {s : Warp}
    {acts : List Act} {e : End} (h : check step fpOf s [] tr = some (acts, e)) :
    ∃ e', Run step s acts e' ∧ End.Like e e' :=
  RunNB.run (step := step) (fpOf := fpOf) hF
    (check_sound (step := step) (fpOf := fpOf) tr s [] acts e h)

/-- **A replayed warp of a kernel ran a run of `Outcomes`.** For kernel `k`'s instructions as the
SASS semantics decodes them (`decode k`, the step `lake exe trace-check` replays with), a warp
whose engine trace replays from its initial state ran, under the semantics, a `Run` with the
trace's acts, ending the same way up to a fault's message. No assumption remains: `footprinted`
discharges the bridge's from the SASS semantics' own theorems. -/
theorem kernel_run (k : Kernel) (env : Env) {tr : List TraceEv} {s : Warp} {acts : List Act}
    {e : End}
    (h : check (warpStep (fun pc => (decode k)[pc]!) env) (warpFp fun pc => (decode k)[pc]!) s []
      tr = some (acts, e)) :
    ∃ e', Run (warpStep (fun pc => (decode k)[pc]!) env) s acts e' ∧ End.Like e e' :=
  check_run _ _ (footprinted _ env) h

end Trace

end NV.Formal
