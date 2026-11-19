/-!
# Finite sums, AM–GM, and the Loomis–Whitney inequality

The counting the matmul I/O bound (`NV.Bounds.Pebble`) rests on, over finite boxes of natural
numbers given by indicator functions.

* `sumR n f`: `Σ_{x < n} f x`, with monotonicity, additivity and exchange of summation.
* `amgm2`: `4ab ≤ (a + b)²`; `amgm3`: `27x²y ≤ 4(x + y)³`, the two-variable form of the
  three-term AM–GM that the bound needs (`4(x + y)³ - 27x²y = (4x + y)(x - 2y)²`).
* `loomis_whitney`: a set `V` of triples `(i, j, k)` in an `M × N × K` box has
  `|V|² ≤ |π_ik V| · |π_kj V| · |π_ij V|` (Loomis and Whitney 1949, the discrete form Irony,
  Toledo and Tiskin 2004 use for matrix multiplication): per slice `k`, `|V_k|` is at most both
  `|π_ij V|` and the product of its row and column counts, and Cauchy–Schwarz sums the slices.
-/

namespace NV.Bounds

/-! ## Sums over ranges -/

/-- `Σ_{x < n} f x`. -/
def sumR : Nat → (Nat → Nat) → Nat
  | 0, _ => 0
  | n + 1, f => sumR n f + f n

/-- `1` if `b`, else `0`. -/
def ind (b : Bool) : Nat := if b then 1 else 0

/-- Whether some `x < n` has `p x`. -/
def anyR : Nat → (Nat → Bool) → Bool
  | 0, _ => false
  | n + 1, p => anyR n p || p n

theorem anyR_iff {n : Nat} {p : Nat → Bool} : anyR n p = true ↔ ∃ x < n, p x = true := by
  induction n with
  | zero => simp [anyR]
  | succ n ih =>
    simp only [anyR, Bool.or_eq_true, ih]
    constructor
    · rintro (⟨x, hx, h⟩ | h)
      · exact ⟨x, by omega, h⟩
      · exact ⟨n, by omega, h⟩
    · rintro ⟨x, hx, h⟩
      by_cases hxn : x < n
      · exact .inl ⟨x, hxn, h⟩
      · have : x = n := by omega
        subst this
        exact .inr h

theorem sumR_le {n : Nat} {f g : Nat → Nat} (h : ∀ x < n, f x ≤ g x) : sumR n f ≤ sumR n g := by
  induction n with
  | zero => exact Nat.le_refl _
  | succ n ih =>
    simp only [sumR]
    exact Nat.add_le_add (ih fun x hx => h x (by omega)) (h n (by omega))

theorem sumR_congr {n : Nat} {f g : Nat → Nat} (h : ∀ x < n, f x = g x) : sumR n f = sumR n g :=
  Nat.le_antisymm (sumR_le fun x hx => Nat.le_of_eq (h x hx)) (sumR_le fun x hx => Nat.le_of_eq (h x hx).symm)

theorem sumR_zero (n : Nat) : sumR n (fun _ => 0) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [sumR, ih]

theorem sumR_one (n : Nat) : sumR n (fun _ => 1) = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [sumR, ih]

theorem sumR_const (n c : Nat) : sumR n (fun _ => c) = n * c := by
  induction n with
  | zero => simp [sumR]
  | succ n ih => simp only [sumR, ih, Nat.succ_mul]

theorem sumR_add (n : Nat) (f g : Nat → Nat) :
    sumR n (fun x => f x + g x) = sumR n f + sumR n g := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [sumR, ih]; omega

theorem sumR_mul_left (n c : Nat) (f : Nat → Nat) : sumR n (fun x => c * f x) = c * sumR n f := by
  induction n with
  | zero => simp [sumR]
  | succ n ih => simp only [sumR, ih, Nat.mul_add]

theorem sumR_comm (n m : Nat) (f : Nat → Nat → Nat) :
    sumR n (fun x => sumR m (f x)) = sumR m (fun y => sumR n (fun x => f x y)) := by
  induction n with
  | zero => simp only [sumR]; exact (sumR_zero m).symm
  | succ n ih =>
    simp only [sumR]
    rw [ih, sumR_add]

theorem ind_le_one (b : Bool) : ind b ≤ 1 := by cases b <;> simp [ind]

theorem ind_le_of {a b : Bool} (h : a = true → b = true) : ind a ≤ ind b := by
  cases a <;> cases b <;> simp_all [ind]

theorem ind_or (a b : Bool) : ind (a || b) ≤ ind a + ind b := by
  cases a <;> cases b <;> simp [ind]

theorem ind_true : ind true = 1 := rfl

theorem ind_false : ind false = 0 := rfl

/-- Pointwise `f ≤ g + h` on a two-dimensional box sums to the same. -/
theorem sum2_le_add {M K : Nat} {f g h : Nat → Nat → Nat}
    (hf : ∀ i < M, ∀ k < K, f i k ≤ g i k + h i k) :
    sumR M (fun i => sumR K (f i)) ≤ sumR M (fun i => sumR K (g i)) + sumR M (fun i => sumR K (h i)) := by
  rw [← sumR_add]
  apply sumR_le
  intro i hi
  rw [← sumR_add]
  exact sumR_le fun k hk => hf i hi k hk

theorem sum2_le {M K : Nat} {f g : Nat → Nat → Nat} (hf : ∀ i < M, ∀ k < K, f i k ≤ g i k) :
    sumR M (fun i => sumR K (f i)) ≤ sumR M (fun i => sumR K (g i)) :=
  sumR_le fun i hi => sumR_le fun k hk => hf i hi k hk

theorem sumR_ind_eq (n x₀ : Nat) : sumR n (fun x => ind (decide (x = x₀))) = if x₀ < n then 1 else 0 := by
  induction n with
  | zero => simp [sumR]
  | succ n ih =>
    simp only [sumR, ih]
    by_cases h : x₀ < n
    · have h₁ : n ≠ x₀ := by omega
      have h₂ : x₀ < n + 1 := by omega
      simp [h, h₁, h₂, ind]
    · by_cases h' : n = x₀
      · subst h'; simp [ind]
      · have h₂ : ¬ x₀ < n + 1 := by omega
        simp [h, h', h₂, ind]

theorem sumR_ind_eq_le (n x₀ : Nat) : sumR n (fun x => ind (decide (x = x₀))) ≤ 1 := by
  rw [sumR_ind_eq]; split <;> omega

theorem sum2_ind_eq_le (M K i₀ k₀ : Nat) :
    sumR M (fun i => sumR K (fun k => ind (decide (i = i₀ ∧ k = k₀)))) ≤ 1 := by
  have h : ∀ i < M, sumR K (fun k => ind (decide (i = i₀ ∧ k = k₀))) ≤ ind (decide (i = i₀)) := by
    intro i _
    by_cases hi : i = i₀
    · subst hi
      simpa [ind] using sumR_ind_eq_le K k₀
    · have h₀ : ∀ k < K, ind (decide (i = i₀ ∧ k = k₀)) = 0 := fun k _ => by simp [hi, ind]
      rw [sumR_congr h₀, sumR_zero]
      exact Nat.zero_le _
  exact Nat.le_trans (sumR_le h) (sumR_ind_eq_le M i₀)

/-! ## AM–GM -/

theorem sq_nonneg_int (a : Int) : 0 ≤ a * a := by
  rcases Int.le_total 0 a with h | h
  · exact Int.mul_nonneg h h
  · have := Int.mul_nonneg (Int.neg_nonneg_of_nonpos h) (Int.neg_nonneg_of_nonpos h)
    have e : -a * -a = a * a := by grind
    omega

/-- `4ab ≤ (a + b)²`. -/
theorem amgm2 (a b : Nat) : 4 * (a * b) ≤ (a + b) * (a + b) := by
  have key : ∀ x y : Int, 4 * (x * y) ≤ (x + y) * (x + y) := by
    intro x y
    have hid : (x + y) * (x + y) = 4 * (x * y) + (x - y) * (x - y) := by grind
    have := sq_nonneg_int (x - y)
    omega
  have := key a b
  exact_mod_cast this

/-- `27x²y ≤ 4(x + y)³`: with `x = |A| + |B|` and `y = |C|` this bounds `|A||B||C|` by
`((|A| + |B| + |C|) / 3)³`. -/
theorem amgm3 (x y : Nat) : 27 * (x * x * y) ≤ 4 * ((x + y) * (x + y) * (x + y)) := by
  have key : ∀ X Y : Int, 0 ≤ X → 0 ≤ Y → 27 * (X * X * Y) ≤ 4 * ((X + Y) * (X + Y) * (X + Y)) := by
    intro X Y hX hY
    have hid : 4 * ((X + Y) * (X + Y) * (X + Y)) =
        27 * (X * X * Y) + (4 * X + Y) * ((X - 2 * Y) * (X - 2 * Y)) := by grind
    have := Int.mul_nonneg (by omega : (0 : Int) ≤ 4 * X + Y) (sq_nonneg_int (X - 2 * Y))
    omega
  have := key x y (by omega) (by omega)
  exact_mod_cast this

theorem cube_mono {x y : Nat} (h : x ≤ y) : x * x * x ≤ y * y * y :=
  Nat.mul_le_mul (Nat.mul_le_mul h h) h

/-- `27v² ≤ X³` bounds `v` by `⌊√(X³ / 27)⌋`. -/
theorem le_sqrt_of {v n : Nat} (h : 27 * (v * v) ≤ n) : v ≤ Nat.sqrt (n / 27) := by
  have h₁ : v * v ≤ n / 27 := (Nat.le_div_iff_mul_le (by decide)).2 (by rw [Nat.mul_comm]; exact h)
  apply Classical.byContradiction
  intro hc
  have h₂ : Nat.sqrt (n / 27) + 1 ≤ v := by omega
  have h₃ := Nat.mul_le_mul h₂ h₂
  have h₄ := Nat.lt_succ_sqrt (n / 27)
  simp only [Nat.succ_eq_add_one] at h₄
  omega

/-! ## Cauchy–Schwarz and Loomis–Whitney -/

theorem cs_step (S P Q c x a b : Nat) (h₁ : S * S ≤ c * P * Q) (h₂ : x * x ≤ c * a * b) :
    (S + x) * (S + x) ≤ c * (P + a) * (Q + b) := by
  have e : (S * x) * (S * x) ≤ (c * P * b) * (c * a * Q) := by
    have := Nat.mul_le_mul h₁ h₂
    have l : (S * x) * (S * x) = (S * S) * (x * x) := by grind
    have r : (c * P * b) * (c * a * Q) = (c * P * Q) * (c * a * b) := by grind
    rw [l, r]; exact this
  have h₃ : 2 * (S * x) ≤ c * P * b + c * a * Q := by
    apply Nat.mul_self_le_mul_self_iff.1
    have := amgm2 (c * P * b) (c * a * Q)
    have l : (2 * (S * x)) * (2 * (S * x)) = 4 * ((S * x) * (S * x)) := by grind
    rw [l]
    exact Nat.le_trans (Nat.mul_le_mul_left 4 e) this
  grind

/-- Cauchy–Schwarz, in the form the slices need: `x_k² ≤ c a_k b_k` for every `k` gives
`(Σ x)² ≤ c (Σ a)(Σ b)`. -/
theorem cauchy_schwarz (K c : Nat) (x a b : Nat → Nat) (h : ∀ k < K, x k * x k ≤ c * a k * b k) :
    sumR K x * sumR K x ≤ c * sumR K a * sumR K b := by
  induction K with
  | zero => simp [sumR]
  | succ K ih =>
    simp only [sumR]
    exact cs_step _ _ _ _ _ _ _ (ih fun k hk => h k (by omega)) (h K (by omega))

section
variable (M N K : Nat) (V : Nat → Nat → Nat → Bool)

/-- The triples of `V` in the box `M × N × K`. -/
def card : Nat := sumR M fun i => sumR N fun j => sumR K fun k => ind (V i j k)

/-- Its projections: the `(i, k)` (an `A` entry), `(k, j)` (a `B` entry) and `(i, j)` (a `C`
entry) of its triples. -/
def projIK : Nat := sumR M fun i => sumR K fun k => ind (anyR N fun j => V i j k)
def projKJ : Nat := sumR K fun k => sumR N fun j => ind (anyR M fun i => V i j k)
def projIJ : Nat := sumR M fun i => sumR N fun j => ind (anyR K fun k => V i j k)

/-- **Loomis–Whitney**: `|V|² ≤ |π_ik V| · |π_kj V| · |π_ij V|`. -/
theorem loomis_whitney : card M N K V * card M N K V ≤
    projIK M N K V * projKJ M N K V * projIJ M N K V := by
  -- Per slice `k`: its triples, its rows, its columns.
  let v k := sumR M fun i => sumR N fun j => ind (V i j k)
  let a k := sumR M fun i => ind (anyR N fun j => V i j k)
  let b k := sumR N fun j => ind (anyR M fun i => V i j k)
  have hcard : card M N K V = sumR K v := by
    have h₁ : ∀ i < M, (sumR N fun j => sumR K fun k => ind (V i j k)) =
        sumR K fun k => sumR N fun j => ind (V i j k) :=
      fun i _ => sumR_comm N K (fun j k => ind (V i j k))
    show sumR M (fun i => sumR N fun j => sumR K fun k => ind (V i j k)) = sumR K v
    rw [sumR_congr h₁]
    exact sumR_comm M K (fun i k => sumR N fun j => ind (V i j k))
  have hA : projIK M N K V = sumR K a := sumR_comm M K (fun i k => ind (anyR N fun j => V i j k))
  have hB : projKJ M N K V = sumR K b := rfl
  have hvab : ∀ k < K, v k ≤ a k * b k := by
    intro k hk
    show sumR M (fun i => sumR N fun j => ind (V i j k)) ≤
      (sumR M fun i => ind (anyR N fun j => V i j k)) * b k
    rw [Nat.mul_comm, ← sumR_mul_left]
    apply sumR_le
    intro i hi
    cases hrow : anyR N (fun j => V i j k)
    · have h₀ : ∀ j < N, ind (V i j k) = 0 := by
        intro j hj
        cases hv : V i j k
        · rfl
        · have := (anyR_iff (n := N) (p := fun j => V i j k)).2 ⟨j, hj, hv⟩
          rw [hrow] at this
          cases this
      rw [sumR_congr h₀, sumR_zero]
      exact Nat.zero_le _
    · rw [ind_true, Nat.mul_one]
      exact sumR_le fun j _ => ind_le_of fun hv => anyR_iff.2 ⟨i, hi, hv⟩
  have hvc : ∀ k < K, v k ≤ projIJ M N K V := fun k hk =>
    sum2_le fun i _ j _ => ind_le_of fun hv => anyR_iff.2 ⟨k, hk, hv⟩
  have hcs := cauchy_schwarz K (projIJ M N K V) v a b (fun k hk => by
    have := Nat.mul_le_mul (hvc k hk) (hvab k hk)
    rw [← Nat.mul_assoc] at this
    exact this)
  rw [hcard, hA, hB]
  have : projIJ M N K V * sumR K a * sumR K b = sumR K a * sumR K b * projIJ M N K V := by grind
  rw [← this]
  exact hcs

end

end NV.Bounds
