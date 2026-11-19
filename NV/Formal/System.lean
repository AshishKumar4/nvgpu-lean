/-!
# Transition systems

A synchronization protocol is a labelled transition system: a start state and a partial step
function, `none` where an action is not enabled (a thread spinning in `try_wait`, a copy that has
not been issued). Safety is an invariant of every reachable state; a violation is witnessed by an
action trace, which `run` replays so a counterexample is checked by evaluation.
-/

namespace NV.Formal

structure System (σ α : Type) where
  init : σ
  step : σ → α → Option σ

namespace System

variable {σ α : Type} (S : System σ α)

inductive Reachable : σ → Prop
  | init : Reachable S.init
  | step {s t : σ} (a : α) : Reachable s → S.step s a = some t → Reachable t

/-- Replay an action trace from `s`; `none` if some action is not enabled. -/
def run (s : σ) : List α → Option σ
  | [] => some s
  | a :: as => (S.step s a).bind (run · as)

theorem reachable_of_run {s t : σ} {as : List α} (hs : S.Reachable s) (h : S.run s as = some t) :
    S.Reachable t := by
  induction as generalizing s with
  | nil => cases h; exact hs
  | cons a as ih =>
    simp only [run, Option.bind_eq_some_iff] at h
    obtain ⟨s', h₁, h₂⟩ := h
    exact ih (.step a hs h₁) h₂

/-- An inductive invariant holds in every reachable state. -/
theorem Reachable.induct {P : σ → Prop} (h₀ : P S.init)
    (h : ∀ s a t, S.Reachable s → P s → S.step s a = some t → P t) {s : σ}
    (hs : S.Reachable s) : P s := by
  induction hs with
  | init => exact h₀
  | step a hr hstep ih => exact h _ a _ hr ih hstep

/-! ## Exhaustive exploration of small instances -/

/-- Every state reachable from `s` (stepping only through `moves`) satisfies `ok`, and every path
of `fuel` steps ends in a state with no enabled move: exploring to depth `fuel` covered it all. -/
def exhaust (moves : List α) (ok : σ → Bool) : Nat → σ → Bool
  | 0, s => ok s && moves.all fun a => (S.step s a).isNone
  | f + 1, s => ok s && moves.all fun a => match S.step s a with
    | some t => exhaust moves ok f t
    | none => true

/-- `t` is reachable from `s`. -/
inductive ReachFrom (s : σ) : σ → Prop
  | refl : ReachFrom s s
  | step {t u : σ} (a : α) : ReachFrom s t → S.step t a = some u → ReachFrom s u

theorem exhaust_step {moves : List α} {ok : σ → Bool} {f : Nat} {s t : σ} {a : α}
    (h : S.exhaust moves ok f s = true) (ha : a ∈ moves) (hs : S.step s a = some t) :
    ∃ f', S.exhaust moves ok f' t = true := by
  cases f with
  | zero =>
    simp only [exhaust, Bool.and_eq_true, List.all_eq_true] at h
    have := h.2 a ha; simp [hs] at this
  | succ f =>
    simp only [exhaust, Bool.and_eq_true, List.all_eq_true] at h
    have := h.2 a ha; rw [hs] at this; exact ⟨f, this⟩

theorem exhaust_ok {moves : List α} {ok : σ → Bool} {f : Nat} {s : σ}
    (h : S.exhaust moves ok f s = true) : ok s = true := by
  cases f <;> (simp only [exhaust, Bool.and_eq_true] at h; exact h.1)

/-- If exploration covered the start, every reachable state is ok, provided `moves` contains
every enabled action. -/
theorem exhaust_sound {moves : List α} {ok : σ → Bool} {f : Nat}
    (hcover : ∀ s a t, S.step s a = some t → a ∈ moves) (h : S.exhaust moves ok f S.init = true)
    {t : σ} (ht : S.Reachable t) : ok t = true := by
  have key : ∀ u, S.ReachFrom S.init u → ∃ f', S.exhaust moves ok f' u = true := by
    intro u hu
    induction hu with
    | refl => exact ⟨f, h⟩
    | step a _ hs ih =>
      obtain ⟨f', hf'⟩ := ih
      exact S.exhaust_step hf' (hcover _ _ _ hs) hs
  have hr : S.ReachFrom S.init t := by
    induction ht with
    | init => exact .refl
    | step a _ hs ih => exact .step a ih hs
  obtain ⟨f', hf'⟩ := key t hr
  exact S.exhaust_ok hf'

end System
end NV.Formal
