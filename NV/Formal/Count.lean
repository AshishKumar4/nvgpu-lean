/-!
# Counting over thread and stage indices

Protocol invariants count threads (how many consumers have arrived) and stages (how many of the
first `p` stages use slot `w`). These lemmas say how a count changes when one index changes, and
give the closed form of the stage count that ties `q % S` and `q / S` in the kernels to rounds of
a slot.
-/

namespace NV.Formal

/-- How many of `0, …, n - 1` satisfy `p`. -/
def countBelow (n : Nat) (p : Nat → Bool) : Nat := (List.range n).countP p

/-- `f 0 + … + f (n - 1)`. -/
def sumBelow (n : Nat) (f : Nat → Nat) : Nat := ((List.range n).map f).sum

theorem countBelow_succ (n : Nat) (p : Nat → Bool) :
    countBelow (n + 1) p = countBelow n p + if p n then 1 else 0 := by
  simp [countBelow, List.range_succ, List.countP_append]

theorem sumBelow_succ (n : Nat) (f : Nat → Nat) : sumBelow (n + 1) f = sumBelow n f + f n := by
  simp [sumBelow, List.range_succ]

theorem countBelow_congr {n : Nat} {p q : Nat → Bool} (h : ∀ j < n, p j = q j) :
    countBelow n p = countBelow n q := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [countBelow_succ, countBelow_succ, ih (fun j hj => h j (by omega)), h n (by omega)]

theorem sumBelow_congr {n : Nat} {f g : Nat → Nat} (h : ∀ j < n, f j = g j) :
    sumBelow n f = sumBelow n g := by
  induction n with
  | zero => rfl
  | succ n ih => rw [sumBelow_succ, sumBelow_succ, ih (fun j hj => h j (by omega)), h n (by omega)]

/-- Changing the predicate at one index `i < n` changes the count by that index alone. -/
theorem countBelow_update {n i : Nat} {p q : Nat → Bool} (hi : i < n)
    (h : ∀ j < n, j ≠ i → p j = q j) :
    countBelow n q + (if p i then 1 else 0) = countBelow n p + if q i then 1 else 0 := by
  induction n with
  | zero => omega
  | succ n ih =>
    rw [countBelow_succ, countBelow_succ]
    by_cases hn : n = i
    · subst hn
      rw [countBelow_congr (fun j hj => (h j (by omega) (by omega)).symm)]
      split <;> split <;> omega
    · have := ih (by omega) (fun j hj hji => h j (by omega) hji)
      rw [h n (by omega) hn]
      omega

theorem sumBelow_update {n i : Nat} {f g : Nat → Nat} (hi : i < n)
    (h : ∀ j < n, j ≠ i → f j = g j) : sumBelow n g + f i = sumBelow n f + g i := by
  induction n with
  | zero => omega
  | succ n ih =>
    rw [sumBelow_succ, sumBelow_succ]
    by_cases hn : n = i
    · subst hn
      rw [sumBelow_congr (fun j hj => (h j (by omega) (by omega)).symm)]
      omega
    · have := ih (by omega) (fun j hj hji => h j (by omega) hji)
      rw [h n (by omega) hn]
      omega

/-- One index switching from unsatisfied to satisfied adds one to the count. -/
theorem countBelow_bump {n i : Nat} {p q : Nat → Bool} (hi : i < n) (h : ∀ j < n, j ≠ i → p j = q j)
    (hp : p i = false) (hq : q i = true) : countBelow n q = countBelow n p + 1 := by
  have := countBelow_update hi h
  rw [hp, hq] at this
  simp at this
  omega

theorem countBelow_le (n : Nat) (p : Nat → Bool) : countBelow n p ≤ n := by
  induction n with
  | zero => exact Nat.le_refl 0
  | succ n ih => rw [countBelow_succ]; split <;> omega

/-- If every index satisfies `p`, the count is `n`; if the count is `n`, every index does. -/
theorem countBelow_eq_iff (n : Nat) (p : Nat → Bool) : countBelow n p = n ↔ ∀ j < n, p j = true := by
  induction n with
  | zero => simp [countBelow]
  | succ n ih =>
    rw [countBelow_succ]
    have := countBelow_le n p
    constructor
    · intro h j hj
      by_cases hpn : p n = true
      · rw [ite_eq_left hpn] at h
        by_cases hjn : j = n
        · subst hjn; exact hpn
        · exact (ih.1 (by omega)) j (by omega)
      · rw [ite_eq_right hpn] at h; omega
    · intro h
      rw [ih.2 (fun j hj => h j (by omega)), ite_eq_left (h n (by omega))]

theorem countBelow_zero_iff (n : Nat) (p : Nat → Bool) :
    countBelow n p = 0 ↔ ∀ j < n, p j = false := by
  induction n with
  | zero => simp [countBelow]
  | succ n ih =>
    rw [countBelow_succ]
    constructor
    · intro h j hj
      by_cases hjn : j = n
      · subst hjn; revert h; split <;> simp_all
      · exact ih.1 (by split at h <;> omega) j (by omega)
    · intro h
      rw [ih.2 (fun j hj => h j (by omega)), ite_eq_right (by simp [h n (by omega)])]

theorem sumBelow_le {n : Nat} {f g : Nat → Nat} (h : ∀ j < n, f j ≤ g j) :
    sumBelow n f ≤ sumBelow n g := by
  induction n with
  | zero => exact Nat.le_refl 0
  | succ n ih =>
    rw [sumBelow_succ, sumBelow_succ]
    have := ih (fun j hj => h j (by omega)); have := h n (by omega); omega

theorem le_sumBelow {n i : Nat} (f : Nat → Nat) (hi : i < n) : f i ≤ sumBelow n f := by
  induction n with
  | zero => omega
  | succ n ih =>
    rw [sumBelow_succ]
    by_cases h : i = n
    · subst h; omega
    · have := ih (by omega); omega

theorem sumBelow_zero_iff (n : Nat) (f : Nat → Nat) : sumBelow n f = 0 ↔ ∀ j < n, f j = 0 := by
  induction n with
  | zero => simp [sumBelow]
  | succ n ih =>
    rw [sumBelow_succ]
    constructor
    · intro h j hj
      by_cases hjn : j = n
      · subst hjn; omega
      · exact ih.1 (by omega) j (by omega)
    · intro h
      rw [ih.2 (fun j hj => h j (by omega)), h n (by omega)]

/-! ## Stages of a slot -/

/-- How many of the stages `0, …, p - 1` use slot `w` of an `S`-slot ring (slot `q % S`). -/
def rounds (S p w : Nat) : Nat := countBelow p fun q => q % S == w

theorem rounds_succ (S p w : Nat) : rounds S (p + 1) w = rounds S p w + if p % S = w then 1 else 0 := by
  simp [rounds, countBelow_succ]

theorem succ_mod_div {S : Nat} (hS : 0 < S) (p : Nat) :
    (p % S + 1 < S → (p + 1) % S = p % S + 1 ∧ (p + 1) / S = p / S) ∧
    (p % S + 1 = S → (p + 1) % S = 0 ∧ (p + 1) / S = p / S + 1) := by
  have hdm := Nat.mod_add_div p S
  have hlt := Nat.mod_lt p hS
  constructor
  · intro h
    have := (Nat.div_mod_unique hS (a := p + 1) (d := p / S) (c := p % S + 1)).2 ⟨by omega, h⟩
    exact ⟨this.2, this.1⟩
  · intro h
    have := (Nat.div_mod_unique hS (a := p + 1) (d := p / S + 1) (c := 0)).2
      ⟨by rw [Nat.mul_succ]; omega, hS⟩
    exact ⟨this.2, this.1⟩

/-- Of the first `p` stages, slot `w < S` holds `p / S`, plus one if `w < p % S`. -/
theorem rounds_eq {S : Nat} (hS : 0 < S) (p : Nat) {w : Nat} (hw : w < S) :
    rounds S p w = p / S + if w < p % S then 1 else 0 := by
  induction p with
  | zero => simp [rounds, countBelow]
  | succ p ih =>
    rw [rounds_succ, ih]
    have ⟨h₁, h₂⟩ := succ_mod_div hS p
    by_cases hlast : p % S + 1 = S
    · rw [(h₂ hlast).1, (h₂ hlast).2]
      split <;> split <;> split <;> omega
    · have hlt : p % S + 1 < S := by have := Nat.mod_lt p hS; omega
      rw [(h₁ hlt).1, (h₁ hlt).2]
      split <;> split <;> split <;> omega

/-- Stage `q` is round `q / S` of its slot. -/
theorem rounds_self {S : Nat} (hS : 0 < S) (q : Nat) : rounds S q (q % S) = q / S := by
  rw [rounds_eq hS q (Nat.mod_lt q hS)]; simp

theorem rounds_mono (S : Nat) {p p' : Nat} (h : p ≤ p') (w : Nat) : rounds S p w ≤ rounds S p' w := by
  induction p' with
  | zero => have : p = 0 := by omega
            subst this; exact Nat.le_refl _
  | succ p' ih =>
    by_cases hp : p = p' + 1
    · subst hp; exact Nat.le_refl _
    · have := ih (by omega); rw [rounds_succ]; omega

end NV.Formal
