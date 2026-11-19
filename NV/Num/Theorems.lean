import NV.Num.Convert

/-!
# Rounding theorems

Every result in `NV.Num` passes through one rounding step, and it is correct against the exact
value:

* the significand step (`Rounding.shift`): each mode returns the multiple of `2^k` its definition
  names; nearest-even a nearest one, the even one on a tie (`shift_rn_nearest`, `shift_rn_tie`);
  a sticky remainder rounds like one more bit set below (`shift_sticky`);
* the format: rounding sees only the value (`round_scale`), returns any finite encoding's value
  as that encoding (`round_pack`), and to nearest even no finite encoding of an IEEE or NaN-only
  format is closer than its result (`round_rn_nearest`);
* conversions: a finite encoding converted into a format that holds its value and back, in any
  modes, is itself (`cvt_round_trip`).

Distances are on naturals, `dist a b = (a - b) + (b - a) = |a - b|`; values are compared exactly,
scaled by a common power of two.
-/

namespace NV.Num

/-- `|a - b|` on naturals. -/
def dist (a b : Nat) : Nat := (a - b) + (b - a)

namespace Rounding

private theorem pow_half {k : Nat} (h : 0 < k) : 2 ^ k = 2 * 2 ^ (k - 1) := by
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
  rw [Nat.add_sub_cancel, Nat.pow_succ, Nat.mul_comm]

/-- The pieces `shift` works with: `m = t · 2^k + ρ` with `ρ < 2^k`. -/
private theorem split (m k : Nat) :
    m = m / 2 ^ k * 2 ^ k + m % 2 ^ k ∧ m % 2 ^ k < 2 ^ k :=
  ⟨by rw [Nat.mul_comm]; exact (Nat.div_add_mod m (2 ^ k)).symm, Nat.mod_lt _ (Nat.two_pow_pos k)⟩

private theorem succ_mul' (t P : Nat) : (t + 1) * P = t * P + P := by
  rw [Nat.add_mul, Nat.one_mul]

/-- A multiple of `2^k` is at least as far from `m` as the nearer of the two that bracket it. -/
private theorem bracket (m k z : Nat) :
    Min.min (m % 2 ^ k) (2 ^ k - m % 2 ^ k) ≤ dist m (z * 2 ^ k) := by
  obtain ⟨hm, hρ⟩ := split m k
  generalize m / 2 ^ k = t at hm
  generalize m % 2 ^ k = ρ at hm hρ
  generalize 2 ^ k = P at hm hρ
  unfold dist; try simp only [pow2_eq]
  rcases Nat.lt_or_ge t z with h | h
  · have := Nat.mul_le_mul_right P (show t + 1 ≤ z from h)
    rw [succ_mul'] at this
    omega
  · have := Nat.mul_le_mul_right P h
    omega

/-- Rounding down to `t`, or up to `t + 1`, is nearest when the remainder is on that side of
half. -/
private theorem near_down (m k z : Nat) (h : m % 2 ^ k ≤ 2 ^ k - m % 2 ^ k) :
    dist m (m / 2 ^ k * 2 ^ k) ≤ dist m (z * 2 ^ k) := by
  have hb := bracket m k z
  obtain ⟨hm, hρ⟩ := split m k
  generalize m / 2 ^ k = t at hm ⊢
  generalize m % 2 ^ k = ρ at hm hρ hb h
  generalize 2 ^ k = P at hm hρ hb h
  unfold dist at hb ⊢; try simp only [pow2_eq] at hb ⊢
  omega

private theorem near_up (m k z : Nat) (h : 2 ^ k - m % 2 ^ k ≤ m % 2 ^ k) :
    dist m ((m / 2 ^ k + 1) * 2 ^ k) ≤ dist m (z * 2 ^ k) := by
  have hb := bracket m k z
  obtain ⟨hm, hρ⟩ := split m k
  rw [succ_mul']
  generalize m / 2 ^ k = t at hm ⊢
  generalize m % 2 ^ k = ρ at hm hρ hb h
  generalize 2 ^ k = P at hm hρ hb h
  unfold dist at hb ⊢; try simp only [pow2_eq] at hb ⊢
  omega

/-- Truncation: `.rz` gives `⌊m / 2^k⌋`. -/
theorem shift_rz (neg : Bool) (m k : Nat) (s : Bool) : shift .rz neg m k s = m / 2 ^ k := by
  simp [shift, up]

/-- `shift` given its decision. -/
private theorem shift_of_up {r : Rounding} {neg : Bool} {m k : Nat} {s b : Bool}
    (h : up r neg (m / 2 ^ k % 2 == 1) (m % 2 ^ k) k s = b) :
    shift r neg m k s = m / 2 ^ k + (if b then 1 else 0) := by
  cases b <;> simp [shift, h]

/-- Nearest-even returns a nearest multiple of `2^k`. -/
theorem shift_rn_nearest (neg : Bool) (m k z : Nat) :
    dist m (shift .rn neg m k * 2 ^ k) ≤ dist m (z * 2 ^ k) := by
  by_cases hk : 0 < k
  · have hh := pow_half hk
    have hρ := Nat.mod_lt m (Nat.two_pow_pos k)
    rcases Nat.lt_trichotomy (m % 2 ^ k) (2 ^ (k - 1)) with hlt | heq | hgt
    · rw [shift_of_up (b := false) (by simp [up] <;> omega)]
      exact near_down m k z (by omega)
    · cases ho : (m / 2 ^ k % 2 == 1)
      · rw [shift_of_up (b := false) (by simp [up, heq, ho])]
        exact near_down m k z (by omega)
      · rw [shift_of_up (b := true) (by simp [up, hk, heq, ho])]
        exact near_up m k z (by omega)
    · rw [shift_of_up (b := true) (by simp [up] <;> omega)]
      exact near_up m k z (by omega)
  · have hk0 : k = 0 := by omega
    subst hk0
    simp [shift, up, dist]

/-- On a tie nearest-even returns the even multiple. -/
theorem shift_rn_tie (neg : Bool) (m k : Nat) (hk : 0 < k) (h : m % 2 ^ k = 2 ^ (k - 1)) :
    shift .rn neg m k % 2 = 0 := by
  cases ho : (m / 2 ^ k % 2 == 1)
  · rw [shift_of_up (b := false) (by simp [up, h, ho])]
    simp at ho ⊢; omega
  · rw [shift_of_up (b := true) (by simp [up, hk, h, ho])]
    simp at ho ⊢; omega

/-- Nearest-away returns a nearest multiple of `2^k`, the larger one on a tie. -/
theorem shift_rna_nearest (neg : Bool) (m k z : Nat) :
    dist m (shift .rna neg m k * 2 ^ k) ≤ dist m (z * 2 ^ k) ∧
      (0 < k → m % 2 ^ k = 2 ^ (k - 1) → shift .rna neg m k = m / 2 ^ k + 1) := by
  by_cases hk : 0 < k
  · have hh := pow_half hk
    have hρ := Nat.mod_lt m (Nat.two_pow_pos k)
    rcases Nat.lt_or_ge (m % 2 ^ k) (2 ^ (k - 1)) with hlt | hge
    · rw [shift_of_up (b := false) (by simp [up] <;> omega)]
      exact ⟨near_down m k z (by omega), fun _ h => by omega⟩
    · rw [shift_of_up (b := true) (by simp [up] <;> omega)]
      exact ⟨near_up m k z (by omega), fun _ _ => rfl⟩
  · have hk0 : k = 0 := by omega
    subst hk0
    simp [shift, up, dist]

/-- Toward the magnitude's floor: `.rz`, `.rm` on a positive value and `.rp` on a negative one
give the largest multiple of `2^k` at most `m`. -/
theorem shift_floor (r : Rounding) (neg : Bool) (m k : Nat)
    (hr : r = .rz ∨ (r = .rm ∧ neg = false) ∨ (r = .rp ∧ neg = true)) :
    shift r neg m k * 2 ^ k ≤ m ∧ ∀ z, z * 2 ^ k ≤ m → z ≤ shift r neg m k := by
  have hdown : shift r neg m k = m / 2 ^ k := by
    rcases hr with h | ⟨h, hn⟩ | ⟨h, hn⟩ <;> subst h <;> (try subst hn) <;> simp [shift, up]
  rw [hdown]
  exact ⟨Nat.div_mul_le_self m _, fun z hz => (Nat.le_div_iff_mul_le (Nat.two_pow_pos k)).2 hz⟩

/-- Toward the magnitude's ceiling: `.rp` on a positive value and `.rm` on a negative one give
the smallest multiple of `2^k` at least `m`. -/
theorem shift_ceil (r : Rounding) (neg : Bool) (m k : Nat)
    (hr : (r = .rp ∧ neg = false) ∨ (r = .rm ∧ neg = true)) :
    m ≤ shift r neg m k * 2 ^ k ∧ ∀ z, m ≤ z * 2 ^ k → shift r neg m k ≤ z := by
  obtain ⟨hm, hρ⟩ := split m k
  by_cases h0 : m % 2 ^ k = 0
  · have hs : shift r neg m k = m / 2 ^ k := by
      rcases hr with ⟨h, hn⟩ | ⟨h, hn⟩ <;> subst h <;> subst hn <;> simp [shift, up, h0]
    rw [hs]
    refine ⟨by omega, fun z hz => ?_⟩
    rcases Nat.lt_or_ge z (m / 2 ^ k) with h | h
    · have := Nat.mul_le_mul_right (2 ^ k) (show z + 1 ≤ m / 2 ^ k from h)
      rw [succ_mul'] at this
      omega
    · exact h
  · have hs : shift r neg m k = m / 2 ^ k + 1 := by
      rcases hr with ⟨h, hn⟩ | ⟨h, hn⟩ <;> subst h <;> subst hn <;> simp [shift, up, h0]
    rw [hs, succ_mul']
    refine ⟨by omega, fun z hz => ?_⟩
    rcases Nat.lt_or_ge z (m / 2 ^ k + 1) with h | h
    · have := Nat.mul_le_mul_right (2 ^ k) (show z ≤ m / 2 ^ k by omega)
      omega
    · exact h

/-- With one bit or more discarded, a sticky remainder decides like one more bit set below. -/
theorem up_sticky (r : Rounding) (neg odd : Bool) (ρ k : Nat) (hk : 0 < k) :
    up r neg odd ρ k true = up r neg odd (2 * ρ + 1) (k + 1) false := by
  have hh := pow_half hk
  have h1 : 2 ^ (k + 1 - 1) = 2 ^ k := by rw [Nat.add_sub_cancel]
  generalize hH : 2 ^ (k - 1) = H at hh
  have hk1 : decide (k + 1 > 0) = true := decide_eq_true (by omega)
  have hk0 : decide (k > 0) = true := decide_eq_true hk
  have hn0 : (2 * ρ + 1 != 0) = true := by simp
  rcases Nat.lt_trichotomy ρ H with h | h | h
  · have e1 : decide (ρ > H) = false := decide_eq_false (by omega)
    have e2 : (ρ == H) = false := by simp; omega
    have e3 : decide (ρ ≥ H) = false := decide_eq_false (by omega)
    have e4 : decide (2 * ρ + 1 > 2 * H) = false := decide_eq_false (by omega)
    have e5 : (2 * ρ + 1 == 2 * H) = false := by simp; omega
    have e6 : decide (2 * ρ + 1 ≥ 2 * H) = false := decide_eq_false (by omega)
    cases r <;> cases neg <;> cases odd <;>
      simp only [pow2_eq, up, h1, hh, hH, hk0, hk1, hn0, e1, e2, e3, e4, e5, e6,
        Bool.or_false, Bool.or_true, Bool.and_false, Bool.not_true,
        Bool.not_false, Bool.and_true]
  · subst h
    have e1 : decide (ρ > ρ) = false := decide_eq_false (by omega)
    have e2 : (ρ == ρ) = true := by simp
    have e3 : decide (ρ ≥ ρ) = true := decide_eq_true (by omega)
    have e4 : decide (2 * ρ + 1 > 2 * ρ) = true := decide_eq_true (by omega)
    have e5 : (2 * ρ + 1 == 2 * ρ) = false := by simp
    have e6 : decide (2 * ρ + 1 ≥ 2 * ρ) = true := decide_eq_true (by omega)
    cases r <;> cases neg <;> cases odd <;>
      simp only [pow2_eq, up, h1, hh, hH, hk0, hk1, hn0, e1, e2, e3, e4, e5, e6,
        Bool.or_false, Bool.or_true, Bool.and_false, Bool.not_true,
        Bool.not_false, Bool.and_true]
  · have e1 : decide (ρ > H) = true := decide_eq_true (by omega)
    have e2 : (ρ == H) = false := by simp; omega
    have e3 : decide (ρ ≥ H) = true := decide_eq_true (by omega)
    have e4 : decide (2 * ρ + 1 > 2 * H) = true := decide_eq_true (by omega)
    have e5 : (2 * ρ + 1 == 2 * H) = false := by simp; omega
    have e6 : decide (2 * ρ + 1 ≥ 2 * H) = true := decide_eq_true (by omega)
    cases r <;> cases neg <;> cases odd <;>
      simp only [pow2_eq, up, h1, hh, hH, hk0, hk1, hn0, e1, e2, e3, e4, e5, e6,
        Bool.or_false, Bool.or_true, Bool.and_false, Bool.not_true,
        Bool.not_false, Bool.and_true]

/-- A sticky remainder below `m`'s last bit rounds like one more bit set below it: the exact value
`(m + δ) · 2^-k`, `0 < δ < 1`, rounds like `(2m + 1) · 2^-(k+1)`, as long as one bit is
discarded. -/
theorem shift_sticky (r : Rounding) (neg : Bool) (m k : Nat) (hk : 0 < k) :
    shift r neg m k true = shift r neg (2 * m + 1) (k + 1) false := by
  have hρ := Nat.mod_lt m (Nat.two_pow_pos k)
  have hdm : 2 * m + 1 = (2 * (m % 2 ^ k) + 1) + m / 2 ^ k * (2 ^ k * 2) := by
    have := Nat.div_add_mod m (2 ^ k)
    rw [← Nat.mul_assoc, Nat.mul_comm (m / 2 ^ k) (2 ^ k)]
    omega
  have hq : (2 * m + 1) / 2 ^ (k + 1) = m / 2 ^ k := by
    rw [Nat.pow_succ, hdm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt (by omega)]
    omega
  have hr : (2 * m + 1) % 2 ^ (k + 1) = 2 * (m % 2 ^ k) + 1 := by
    rw [Nat.pow_succ, hdm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (by omega)]
  simp only [pow2_eq, shift, hq, hr, up_sticky r neg _ _ _ hk]

end Rounding

/-! ## Encodings -/

namespace Format

private theorem div_digit {A F n : Nat} (hF : F < 2 ^ n) : (A * 2 ^ n + F) / 2 ^ n = A := by
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ (Nat.two_pow_pos n), Nat.div_eq_of_lt hF,
    Nat.zero_add]

private theorem mod_digit {A F n : Nat} (hF : F < 2 ^ n) : (A * 2 ^ n + F) % 2 ^ n = F := by
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hF]

private theorem digits_lt {E F a b : Nat} (hE : E < 2 ^ a) (hF : F < 2 ^ b) :
    E * 2 ^ b + F < 2 ^ (a + b) := by
  have : (E + 1) * 2 ^ b ≤ 2 ^ a * 2 ^ b := Nat.mul_le_mul_right _ hE
  rw [Nat.add_mul, Nat.one_mul] at this
  rw [Nat.pow_add]
  omega

/-- An encoding is its sign digit, exponent field, fraction field and padding. -/
theorem pack_eq (f : Format) (s : Bool) (E F : Nat) :
    f.pack s E F =
      (((if s && f.signed then 1 else 0) * 2 ^ f.exp + E) * 2 ^ f.man + F) * 2 ^ f.pad := by
  unfold pack signBit width; try simp only [pow2_eq]
  cases s <;> cases hs : f.signed <;> simp only [Bool.and_true, Bool.and_false,
    ite_true, ite_false, Bool.false_eq_true, Nat.zero_mul, Nat.zero_add, Nat.one_mul]
  rw [show 1 + f.exp + f.man + f.pad - 1 = f.exp + f.man + f.pad by omega, Nat.pow_add,
    Nat.pow_add]
  simp only [Nat.add_mul, Nat.mul_assoc]
  omega

theorem fields_pack (f : Format) (s : Bool) (E F : Nat) (hE : E < 2 ^ f.exp)
    (hF : F < 2 ^ f.man) : f.fields (f.pack s E F) = (s && f.signed, E, F) := by
  have hb := pack_eq f s E F
  generalize hS : (if s && f.signed then 1 else 0) = S at hb
  have hX := digits_lt hE hF
  have h1 : f.pack s E F / 2 ^ f.pad = (S * 2 ^ f.exp + E) * 2 ^ f.man + F := by
    rw [hb, Nat.mul_div_cancel _ (Nat.two_pow_pos _)]
  have h2 : f.pack s E F / 2 ^ f.pad / 2 ^ f.man % 2 ^ f.exp = E := by
    rw [h1, div_digit hF, Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hE]
  have h3 : f.pack s E F / 2 ^ f.pad % 2 ^ f.man = F := by rw [h1, mod_digit hF]
  have h4 : f.signed = true → f.pack s E F / 2 ^ (f.width - 1) = S := by
    intro hsg
    have hw : f.width - 1 = f.exp + f.man + f.pad := by simp [width, hsg]; omega
    have h5 : (S * 2 ^ f.exp + E) * 2 ^ f.man + F = (E * 2 ^ f.man + F) + S * 2 ^ (f.exp + f.man) := by
      rw [Nat.pow_add, Nat.add_mul, Nat.mul_assoc]; omega
    rw [hw, Nat.pow_add, Nat.mul_comm, ← Nat.div_div_eq_div_mul, h1, h5,
      Nat.add_mul_div_right _ _ (Nat.two_pow_pos _), Nat.div_eq_of_lt hX, Nat.zero_add]
  unfold fields; try simp only [pow2_eq]
  simp only [h2, h3]
  cases hsg : f.signed
  · simp
  · rw [h4 hsg]; cases s <;> simp [← hS, hsg]

theorem pack_signed (f : Format) (s : Bool) (E F : Nat) :
    f.pack (s && f.signed) E F = f.pack s E F := by
  simp [pack]

theorem decode_pack (f : Format) (s : Bool) (E F : Nat) (hE : E < 2 ^ f.exp)
    (hF : F < 2 ^ f.man) : f.decode (f.pack s E F) =
      if E = f.expMax ∧ f.special = .ieee then (if F = 0 then .inf (s && f.signed) else .nan)
      else if E = f.expMax ∧ f.special = .nanOnly ∧ F = 2 ^ f.man - 1 then .nan
      else if E = 0 ∧ f.subnormals then .fin (s && f.signed) F f.qMin
      else .fin (s && f.signed) (2 ^ f.man + F) ((E : Int) - f.bias - f.man) := by
  simp only [pow2_eq, decode, fields_pack f s E F hE hF]

/-- A finite encoding is at most the largest finite one. -/
theorem le_top (f : Format) (s : Bool) (E F : Nat) (hE : E < 2 ^ f.exp) (hF : F < 2 ^ f.man)
    (n : Bool) (m : Nat) (e : Int) (hd : f.decode (f.pack s E F) = .fin n m e) :
    E < f.topFinite.1 ∨ (E = f.topFinite.1 ∧ F ≤ f.topFinite.2) := by
  rw [decode_pack f s E F hE hF] at hd
  have hx : f.expMax = 2 ^ f.exp - 1 := by simp [expMax]
  unfold topFinite; try simp only [pow2_eq]
  cases hsp : f.special <;> simp only [hsp] at hd ⊢
  · -- ieee
    by_cases h : E = f.expMax
    · simp [h] at hd; split at hd <;> cases hd
    · omega
  · -- nanOnly
    by_cases hm : f.man = 0
    · simp only [hm, ite_true]
      have : F = 0 := by rw [hm] at hF; omega
      by_cases h : E = f.expMax
      · simp [h, this, hm] at hd
      · omega
    · simp only [hm, ite_false]
      by_cases h : E = f.expMax ∧ F = 2 ^ f.man - 1
      · simp [h] at hd
      · have := Nat.two_pow_pos f.man
        omega
  · omega

end Format

theorem log2_eq {m a : Nat} (h1 : 2 ^ a ≤ m) (h2 : m < 2 ^ (a + 1)) : Nat.log2 m = a := by
  have hm : m ≠ 0 := by have := Nat.two_pow_pos a; omega
  have hlt := (Nat.log2_lt hm).2 h2
  have hge : ¬ Nat.log2 m < a := fun h => absurd ((Nat.log2_lt hm).1 h) (by omega)
  omega

/-- Rounding the value of a finite encoding, in any mode, gives back the encoding. -/
theorem round_pack (f : Format) (r : Rounding) (ovf : Overflow) (s : Bool) (E F : Nat)
    (hE : E < 2 ^ f.exp) (hF : F < 2 ^ f.man) (n : Bool) (m : Nat) (e : Int)
    (hd : f.decode (f.pack s E F) = .fin n m e) : f.round r ovf n m e = f.pack s E F := by
  have htop := Format.le_top f s E F hE hF n m e hd
  rw [Format.decode_pack f s E F hE hF] at hd
  split at hd
  · split at hd <;> cases hd
  split at hd
  · cases hd
  split at hd
  · rename_i _ _ hsub
    cases hd
    obtain ⟨rfl, hs⟩ := hsub
    by_cases hF0 : F = 0
    · subst hF0
      simp [Format.round, Format.roundWith, Format.zero, Format.pack_signed]
    · have hl := (Nat.log2_lt hF0).2 hF
      have hq : Max.max (f.qMin + ↑(Nat.log2 F) - ↑f.man) f.qMin = f.qMin := by omega
      have hft : ¬ (0 > f.topFinite.1 ∨ (0 = f.topFinite.1 ∧ F % 2 ^ f.man > f.topFinite.2)) := by
        rw [Nat.mod_eq_of_lt hF]; omega
      simp [Format.round, Format.roundWith, Format.quantize, hF0, hq, hs, hF, Format.pack_signed,
        Nat.mod_eq_of_lt hF]
      intros; omega
  · rename_i hsub
    cases hd
    have hm1 : 2 ^ f.man ≤ 2 ^ f.man + F := by omega
    have hm2 : 2 ^ f.man + F < 2 ^ (f.man + 1) := by rw [Nat.pow_succ]; omega
    have hl : Nat.log2 (2 ^ f.man + F) = f.man := log2_eq hm1 hm2
    have hEs : (if f.subnormals then 1 else 0) ≤ E := by
      cases hs : f.subnormals <;> simp [hs] at hsub ⊢ <;> omega
    have hq : Max.max ((E : Int) - f.bias - f.man + ↑(Nat.log2 (2 ^ f.man + F)) - ↑f.man) f.qMin =
        (E : Int) - f.bias - f.man := by
      rw [hl]; unfold Format.qMin; try simp only [pow2_eq]
      cases hs : f.subnormals <;> simp only [hs, ite_true, ite_false, Bool.false_eq_true] at hEs ⊢ <;>
        omega
    have hef : ((E : Int) - f.bias - f.man - f.qMin).toNat + (if f.subnormals then 1 else 0) = E := by
      unfold Format.qMin; try simp only [pow2_eq]
      cases hs : f.subnormals <;> simp only [hs, ite_true, ite_false, Bool.false_eq_true] at hEs ⊢ <;>
        omega
    have hnlt : ¬ 2 ^ f.man + F < 2 ^ f.man := by omega
    have hfr : (2 ^ f.man + F) % 2 ^ f.man = F := by rw [Nat.add_mod_left, Nat.mod_eq_of_lt hF]
    simp [Format.round, Format.roundWith, Format.quantize, hq, hef, hnlt, hfr, Format.pack_signed]
    intros; omega

/-! ## Rounding depends only on the value -/

namespace Rounding

/-- Nothing discarded: no mode rounds up. -/
theorem up_zero (r : Rounding) (neg odd : Bool) (k : Nat) :
    up r neg odd 0 k false = false := by
  have hp := Nat.two_pow_pos (k - 1)
  have h1 : decide (0 > 2 ^ (k - 1)) = false := decide_eq_false (by omega)
  have h2 : (0 == 2 ^ (k - 1)) = false := beq_eq_false_iff_ne.mpr (by omega)
  have h3 : decide (0 ≥ 2 ^ (k - 1)) = false := decide_eq_false (by omega)
  have h4 : (0 != 0) = false := rfl
  cases r <;> simp only [pow2_eq, up, h1, h2, h3, h4, Bool.and_false, Bool.false_and, Bool.or_false,
    Bool.false_or]

/-- The decision is the same for the discarded bits scaled by `2^j`. -/
theorem up_scale (r : Rounding) (neg odd : Bool) (ρ k j : Nat) (hk : 0 < k) :
    up r neg odd (ρ * 2 ^ j) (k + j) false = up r neg odd ρ k false := by
  have hh : 2 ^ (k + j - 1) = 2 ^ (k - 1) * 2 ^ j := by
    rw [← Nat.pow_add]; congr 1; omega
  have hk' : decide (k + j > 0) = true := decide_eq_true (by omega)
  have hk0 : decide (k > 0) = true := decide_eq_true hk
  have hj := Nat.two_pow_pos j
  generalize hH : 2 ^ (k - 1) = H at hh
  generalize hP : 2 ^ j = P at hh hj ⊢
  have hz : (ρ * P != 0) = (ρ != 0) := by
    by_cases h0 : ρ = 0
    · subst h0; simp
    · have h1 : ρ * P ≠ 0 := Nat.mul_ne_zero h0 (by omega)
      rw [bne_iff_ne.mpr h1, bne_iff_ne.mpr h0]
  rcases Nat.lt_trichotomy ρ H with h | h | h
  · have a := Nat.mul_lt_mul_of_pos_right h hj
    have e1 : decide (ρ > H) = false := decide_eq_false (by omega)
    have e2 : (ρ == H) = false := by simp; omega
    have e3 : decide (ρ ≥ H) = false := decide_eq_false (by omega)
    have e4 : decide (ρ * P > H * P) = false := decide_eq_false (by omega)
    have e5 : (ρ * P == H * P) = false := by simp; omega
    have e6 : decide (ρ * P ≥ H * P) = false := decide_eq_false (by omega)
    cases r <;> cases neg <;> cases odd <;>
      simp only [pow2_eq, up, hh, hH, hk0, hk', hz, e1, e2, e3, e4, e5, e6, Bool.true_and,
        Bool.false_and, Bool.or_false, Bool.false_or, Bool.and_false, Bool.not_true,
        Bool.not_false, Bool.and_true]
  · subst h
    have e1 : decide (ρ > ρ) = false := decide_eq_false (by omega)
    have e2 : (ρ == ρ) = true := by simp
    have e3 : decide (ρ ≥ ρ) = true := decide_eq_true (by omega)
    have e4 : decide (ρ * P > ρ * P) = false := decide_eq_false (by omega)
    have e5 : (ρ * P == ρ * P) = true := by simp
    have e6 : decide (ρ * P ≥ ρ * P) = true := decide_eq_true (by omega)
    cases r <;> cases neg <;> cases odd <;>
      simp only [pow2_eq, up, hh, hH, hk0, hk', hz, e1, e2, e3, e4, e5, e6, Bool.true_and,
        Bool.false_and, Bool.or_false, Bool.false_or, Bool.and_false, Bool.not_true,
        Bool.not_false, Bool.and_true]
  · have a := Nat.mul_lt_mul_of_pos_right h hj
    have e1 : decide (ρ > H) = true := decide_eq_true (by omega)
    have e2 : (ρ == H) = false := by simp; omega
    have e3 : decide (ρ ≥ H) = true := decide_eq_true (by omega)
    have e4 : decide (ρ * P > H * P) = true := decide_eq_true (by omega)
    have e5 : (ρ * P == H * P) = false := by simp; omega
    have e6 : decide (ρ * P ≥ H * P) = true := decide_eq_true (by omega)
    cases r <;> cases neg <;> cases odd <;>
      simp only [pow2_eq, up, hh, hH, hk0, hk', hz, e1, e2, e3, e4, e5, e6, Bool.true_and,
        Bool.false_and, Bool.or_false, Bool.false_or, Bool.and_false, Bool.not_true,
        Bool.not_false, Bool.and_true]

end Rounding

namespace Format

theorem log2_mul_pow {m : Nat} (hm : m ≠ 0) (j : Nat) :
    Nat.log2 (m * 2 ^ j) = Nat.log2 m + j := by
  have h1 := Nat.log2_self_le hm
  have h2 := Nat.lt_log2_self (n := m)
  apply log2_eq
  · rw [Nat.pow_add]; exact Nat.mul_le_mul_right _ h1
  · rw [show Nat.log2 m + j + 1 = (Nat.log2 m + 1) + j by omega, Nat.pow_add]
    exact Nat.mul_lt_mul_of_pos_right h2 (Nat.two_pow_pos j)

/-- The significand step sees only the value: `m · 2^j` at `e - j` rounds like `m` at `e`. -/
theorem quantize_scale (f : Format) (r : Rounding) (neg : Bool) (m : Nat) (e : Int) (j : Nat)
    (hm : m ≠ 0) :
    f.quantize r.up neg (m * 2 ^ j) (e - j) false = f.quantize r.up neg m e false := by
  have hl := log2_mul_pow hm j
  have hq : e - ↑j + ↑(Nat.log2 (m * 2 ^ j)) - ↑f.man = e + ↑(Nat.log2 m) - ↑f.man := by
    rw [hl]; omega
  have hj := Nat.two_pow_pos j
  have hlog := Nat.lt_log2_self (n := m)
  unfold quantize; try simp only [pow2_eq]
  simp only [hq]
  generalize hQ : Max.max (e + ↑(Nat.log2 m) - ↑f.man) f.qMin = q
  by_cases h1 : q ≤ e - j
  · -- exact on both sides
    have h2 : q ≤ e := by omega
    simp only [h1, h2, ite_true]
    congr 1
    rw [Nat.mul_assoc, ← Nat.pow_add]
    congr 2
    omega
  · by_cases h2 : q ≤ e
    · -- exact for `m`; the scaled form discards only zeros
      simp only [h1, h2, ite_true, ite_false]
      have hmj : m * 2 ^ j = m * 2 ^ (e - q).toNat * 2 ^ (q - (e - ↑j)).toNat := by
        rw [Nat.mul_assoc, ← Nat.pow_add]; congr 2; omega
      have ht : m * 2 ^ j / 2 ^ (q - (e - ↑j)).toNat = m * 2 ^ (e - q).toNat := by
        rw [hmj, Nat.mul_div_cancel _ (Nat.two_pow_pos _)]
      have hr : m * 2 ^ j % 2 ^ (q - (e - ↑j)).toNat = 0 := by rw [hmj, Nat.mul_mod_left]
      have hlt : m * 2 ^ (e - q).toNat < 2 ^ (f.man + 1) := by
        have hle : Nat.log2 m + 1 + (e - q).toNat ≤ f.man + 1 := by omega
        calc m * 2 ^ (e - q).toNat < 2 ^ (Nat.log2 m + 1) * 2 ^ (e - q).toNat :=
              Nat.mul_lt_mul_of_pos_right hlog (Nat.two_pow_pos _)
          _ = 2 ^ (Nat.log2 m + 1 + (e - q).toNat) := (Nat.pow_add 2 _ _).symm
          _ ≤ 2 ^ (f.man + 1) := Nat.pow_le_pow_right (by decide) hle
      simp only [ht, hr, Rounding.up_zero r neg _ _, Bool.false_eq_true, ite_false,
        Nat.ne_of_lt hlt]
    · -- both inexact: the same bits are discarded, scaled
      simp only [h1, h2, ite_false]
      have hk : 0 < (q - e).toNat := by omega
      have hk' : (q - (e - ↑j)).toNat = (q - e).toNat + j := by omega
      have ht : m * 2 ^ j / 2 ^ ((q - e).toNat + j) = m / 2 ^ (q - e).toNat := by
        rw [Nat.pow_add, Nat.mul_div_mul_right _ _ hj]
      have hr : m * 2 ^ j % 2 ^ ((q - e).toNat + j) = m % 2 ^ (q - e).toNat * 2 ^ j := by
        rw [Nat.pow_add, Nat.mul_mod_mul_right]
      simp only [hk', ht, hr, Rounding.up_scale r neg _ _ _ j hk]

/-- Rounding sees only the value. -/
theorem round_scale (f : Format) (r : Rounding) (ovf : Overflow) (neg : Bool) (m : Nat) (e : Int)
    (j : Nat) : f.round r ovf neg (m * 2 ^ j) (e - j) = f.round r ovf neg m e := by
  by_cases hm : m = 0
  · subst hm; simp [round, roundWith]
  · have h0 : m * 2 ^ j ≠ 0 := Nat.mul_ne_zero hm (Nat.pos_iff_ne_zero.1 (Nat.two_pow_pos j))
    unfold round roundWith; try simp only [pow2_eq]
    simp only [hm, h0, ite_false, quantize_scale f r _ m e j hm]

end Format

/-! ## Conversion round trips -/

theorem CvtOpts.finish_default (dst : Format) (d : Nat) : ({} : CvtOpts).finish dst d = d := by
  simp [CvtOpts.finish, Format.flushResult]

/-- `cvt` without modifiers rounds the source's exact value once. -/
theorem cvt_fin (src dst : Format) (r : Rounding) (b : Nat) (n : Bool) (m : Nat) (e : Int)
    (hd : src.decode b = .fin n m e) : cvt src dst r {} b = dst.round r .ieee n m e := by
  simp [cvt, cvtVal, Format.decodeIn, hd, CvtOpts.finish_default]

/-- Converting a finite encoding into a format that holds its value, then back, in any rounding
modes, gives the encoding back. -/
theorem cvt_round_trip (f g : Format) (r r' : Rounding) (s : Bool) (E F : Nat)
    (hE : E < 2 ^ f.exp) (hF : F < 2 ^ f.man) (n : Bool) (m : Nat) (e : Int)
    (hd : f.decode (f.pack s E F) = .fin n m e)
    (s' : Bool) (E' F' : Nat) (hE' : E' < 2 ^ g.exp) (hF' : F' < 2 ^ g.man) (m' : Nat) (e' : Int)
    (hd' : g.decode (g.pack s' E' F') = .fin n m' e')
    (j : Nat) (hv : (m' = m * 2 ^ j ∧ e' = e - j) ∨ (m = m' * 2 ^ j ∧ e = e' - j)) :
    cvt g f r {} (cvt f g r' {} (f.pack s E F)) = f.pack s E F := by
  have same (h : Format) (q : Rounding) : h.round q .ieee n m e = h.round q .ieee n m' e' := by
    rcases hv with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
    · exact (Format.round_scale h q .ieee n m e j).symm
    · exact Format.round_scale h q .ieee n m' e' j
  rw [cvt_fin f g r' _ n m e hd, same g r', round_pack g r' .ieee s' E' F' hE' hF' n m' e' hd',
    cvt_fin g f r _ n m' e' hd', ← same f r, round_pack f r .ieee s E F hE hF n m e hd]

/-! ## Nearest-even at the format level

Every finite value of a format is `M · 2^Q` with `Q ≥ qMin` and `M < 2^(man+1)` (`decode_pack`).
Rounding to nearest even gives a value at least as close to the exact one as every such value:
the significand step is nearest among the multiples of its quantum (`shift_rn_nearest`), and a
value with a finer quantum lies below the binade, under the rounding's lower neighbour. Values are
compared exactly, scaled by a common `2^-L`. -/

theorem dist_mul (a b u : Nat) : dist (a * u) (b * u) = dist a b * u := by
  unfold dist; (try simp only [pow2_eq]); rw [Nat.add_mul, Nat.sub_mul, Nat.sub_mul]

namespace Format

/-- `quantize` when it discards bits: the significand step is `Rounding.shift`. -/
theorem quantize_shift (f : Format) (r : Rounding) (neg : Bool) (m : Nat) (e q : Int)
    (hq : Max.max (e + ↑(Nat.log2 m) - ↑f.man) f.qMin = q) (h : ¬ q ≤ e) :
    f.quantize r.up neg m e false =
      if r.shift neg m (q - e).toNat = 2 ^ (f.man + 1) then (2 ^ f.man, q + 1)
      else (r.shift neg m (q - e).toNat, q) := by
  unfold quantize Rounding.shift
  simp only [pow2_eq, hq, h, ite_false]

theorem quantize_rn_nearest (f : Format) (neg : Bool) (m : Nat) (e : Int) (hm : m ≠ 0)
    (L : Int) (hL1 : L ≤ e) (hL2 : L ≤ f.qMin) (M : Nat) (Q : Int) (hQ : f.qMin ≤ Q)
    (hM : M < 2 ^ (f.man + 1)) :
    dist (m * 2 ^ (e - L).toNat)
        ((f.quantize Rounding.rn.up neg m e false).1 *
          2 ^ ((f.quantize Rounding.rn.up neg m e false).2 - L).toNat) ≤
      dist (m * 2 ^ (e - L).toNat) (M * 2 ^ (Q - L).toNat) := by
  have hlog := Nat.lt_log2_self (n := m)
  have hlog' := Nat.log2_self_le hm
  generalize hq : Max.max (e + ↑(Nat.log2 m) - ↑f.man) f.qMin = q
  have hq1 : e + ↑(Nat.log2 m) - ↑f.man ≤ q := by omega
  have hq2 : f.qMin ≤ q := by omega
  by_cases he : q ≤ e
  · -- exact
    have hx : f.quantize Rounding.rn.up neg m e false = (m * 2 ^ (e - q).toNat, q) := by
      unfold quantize; simp only [pow2_eq, hq, he, ite_true]
    rw [hx]
    have : m * 2 ^ (e - q).toNat * 2 ^ (q - L).toNat = m * 2 ^ (e - L).toNat := by
      rw [Nat.mul_assoc, ← Nat.pow_add]; congr 2; omega
    simp only [this, dist, Nat.sub_self, Nat.zero_add, Nat.zero_le]
  · rw [quantize_shift f .rn neg m e q hq he]
    generalize hk : (q - e).toNat = k
    have hk1 : 0 < k := by omega
    -- the result's value, carry or not, is `shift · 2^q`
    have hval : ∀ s : Nat,
        (if s = 2 ^ (f.man + 1) then (2 ^ f.man, q + 1) else (s, q)).1 *
          2 ^ ((if s = 2 ^ (f.man + 1) then (2 ^ f.man, q + 1) else (s, q)).2 - L).toNat =
        s * 2 ^ k * 2 ^ (e - L).toNat := by
      intro s
      have hqL : 2 ^ (q - L).toNat = 2 ^ k * 2 ^ (e - L).toNat := by
        rw [← Nat.pow_add]; congr 1; omega
      split
      · rename_i hs
        rw [hs, Nat.mul_assoc, hqL.symm, ← Nat.pow_add, ← Nat.pow_add]
        congr 1; omega
      · rw [hqL, Nat.mul_assoc]
    rw [hval]
    generalize hu : 2 ^ (e - L).toNat = u
    have hu0 : 0 < u := by rw [← hu]; exact Nat.two_pow_pos _
    by_cases hQq : q ≤ Q
    · -- the other value is a multiple of the quantum
      have hz : M * 2 ^ (Q - L).toNat = M * 2 ^ (Q - q).toNat * 2 ^ k * u := by
        rw [← hu, Nat.mul_assoc, Nat.mul_assoc, ← Nat.pow_add, ← Nat.pow_add]
        congr 2; omega
      rw [hz, dist_mul, dist_mul]
      exact Nat.mul_le_mul_right u (Rounding.shift_rn_nearest neg m k _)
    · -- the other value lies below the binade, under the lower neighbour
      have hqe : q = e + ↑(Nat.log2 m) - ↑f.man := by omega
      have hkl : k ≤ Nat.log2 m := by omega
      -- t · 2^k ≥ 2^log2 m
      have hpow : 2 ^ (Nat.log2 m - k) * 2 ^ k = 2 ^ Nat.log2 m := by
        rw [← Nat.pow_add, Nat.sub_add_cancel hkl]
      have h1 : 2 ^ (Nat.log2 m - k) ≤ m / 2 ^ k := by
        apply (Nat.le_div_iff_mul_le (Nat.two_pow_pos k)).2
        rw [hpow]; exact hlog'
      have ht : 2 ^ Nat.log2 m ≤ m / 2 ^ k * 2 ^ k := by
        rw [← hpow]; exact Nat.mul_le_mul_right _ h1
      -- the other value is below 2^log2 m (in units of 2^e)
      have hW : M * 2 ^ (Q - L).toNat < 2 ^ Nat.log2 m * u := by
        have h1 : M * 2 ^ (Q - L).toNat < 2 ^ (f.man + 1) * 2 ^ (Q - L).toNat :=
          Nat.mul_lt_mul_of_pos_right hM (Nat.two_pow_pos _)
        have h2 : 2 ^ (f.man + 1) * 2 ^ (Q - L).toNat ≤ 2 ^ Nat.log2 m * u := by
          rw [← hu, ← Nat.pow_add, ← Nat.pow_add]
          exact Nat.pow_le_pow_right (by decide) (by omega)
        omega
      have hn := Rounding.shift_rn_nearest neg m k (m / 2 ^ k)
      have hdiv := Nat.div_mul_le_self m (2 ^ k)
      have hn' := Nat.mul_le_mul_right u hn
      rw [← dist_mul, ← dist_mul] at hn'
      have hmu : m / 2 ^ k * 2 ^ k * u ≤ m * u := Nat.mul_le_mul_right u hdiv
      have hlo : 2 ^ Nat.log2 m * u ≤ m / 2 ^ k * 2 ^ k * u := Nat.mul_le_mul_right u ht
      unfold dist at hn' ⊢; try simp only [pow2_eq] at hn' ⊢
      omega


/-- Every finite value has an exponent at least `qMin` and a significand below `2^(man+1)`. -/
theorem decode_bounds (f : Format) (s : Bool) (E F : Nat) (hE : E < 2 ^ f.exp)
    (hF : F < 2 ^ f.man) (n : Bool) (M : Nat) (Q : Int)
    (hd : f.decode (f.pack s E F) = .fin n M Q) : f.qMin ≤ Q ∧ M < 2 ^ (f.man + 1) := by
  have hp := Nat.two_pow_pos f.man
  have h2 : 2 ^ (f.man + 1) = 2 * 2 ^ f.man := by rw [Nat.pow_succ, Nat.mul_comm]
  rw [decode_pack f s E F hE hF] at hd
  split at hd
  · split at hd <;> cases hd
  split at hd
  · cases hd
  split at hd
  · cases hd; omega
  · rename_i _ _ hsub
    cases hd
    refine ⟨?_, by omega⟩
    unfold qMin; try simp only [pow2_eq]
    cases hs : f.subnormals <;> simp [hs] at hsub ⊢ <;> omega

/-- The overflow results of an IEEE or NaN-only format, not saturating, are not finite. -/
theorem overflow_not_fin (f : Format) (hspec : f.special ≠ .finite) (ovf : Overflow)
    (hovf : ovf ≠ .sat) (neg : Bool) (n : Bool) (M : Nat) (Q : Int)
    (he : 0 < f.exp) : f.decode (f.overflow ovf true neg) ≠ .fin n M Q := by
  have hx : f.expMax < 2 ^ f.exp := by
    unfold expMax; simp only [pow2_eq]; have := Nat.two_pow_pos f.exp; omega
  have hm : 2 ^ f.man - 1 < 2 ^ f.man := by have := Nat.two_pow_pos f.man; omega
  have hz : (0 : Nat) < 2 ^ f.man := Nat.two_pow_pos f.man
  have hne : f.expMax ≠ 0 := by
    unfold expMax; try simp only [pow2_eq]
    have : 2 ≤ 2 ^ f.exp := by
      calc 2 = 2 ^ 1 := rfl
        _ ≤ 2 ^ f.exp := Nat.pow_le_pow_right (by decide) he
    omega
  cases hsp : f.special
  · cases ovf
    · simp only [overflow, hsp, ite_true, inf]
      rw [decode_pack f _ _ _ hx hz]; simp [hsp]
    · exact absurd rfl hovf
    · simp only [overflow, hsp, inf]
      rw [decode_pack f _ _ _ hx hz]; simp [hsp]
  · cases ovf
    · simp only [pow2_eq, overflow, hsp, ite_true]
      rw [decode_pack f _ _ _ hx hm]; simp [hsp]
    · exact absurd rfl hovf
    · simp only [pow2_eq, overflow, hsp]
      rw [decode_pack f _ _ _ hx hm]; simp [hsp]
  · exact absurd hsp hspec


/-- The significand step's result: below `2^(man+1)`, at an exponent at least `qMin`, and with the
leading bit unless at the subnormals' exponent. -/
theorem quantize_bounds (f : Format) (r : Rounding) (neg : Bool) (m : Nat) (e : Int)
    (hm : m ≠ 0) :
    (f.quantize r.up neg m e false).1 < 2 ^ (f.man + 1) ∧
      f.qMin ≤ (f.quantize r.up neg m e false).2 ∧
      ((f.quantize r.up neg m e false).2 = f.qMin ∨
        2 ^ f.man ≤ (f.quantize r.up neg m e false).1) := by
  have hlog := Nat.lt_log2_self (n := m)
  have hlog' := Nat.log2_self_le hm
  generalize hq : Max.max (e + ↑(Nat.log2 m) - ↑f.man) f.qMin = q
  have hq1 : e + ↑(Nat.log2 m) - ↑f.man ≤ q := by omega
  have hq2 : f.qMin ≤ q := by omega
  have hq3 : q = f.qMin ∨ q = e + ↑(Nat.log2 m) - ↑f.man := by omega
  by_cases he : q ≤ e
  · have hx : f.quantize r.up neg m e false = (m * 2 ^ (e - q).toNat, q) := by
      unfold quantize; simp only [pow2_eq, hq, he, ite_true]
    rw [hx]
    refine ⟨?_, hq2, ?_⟩
    · calc m * 2 ^ (e - q).toNat < 2 ^ (Nat.log2 m + 1) * 2 ^ (e - q).toNat :=
            Nat.mul_lt_mul_of_pos_right hlog (Nat.two_pow_pos _)
        _ = 2 ^ (Nat.log2 m + 1 + (e - q).toNat) := (Nat.pow_add 2 _ _).symm
        _ ≤ 2 ^ (f.man + 1) := Nat.pow_le_pow_right (by decide) (by omega)
    · rcases hq3 with h | h
      · exact Or.inl h
      · right
        calc 2 ^ f.man = 2 ^ Nat.log2 m * 2 ^ (e - q).toNat := by
              rw [← Nat.pow_add]; congr 1; omega
          _ ≤ m * 2 ^ (e - q).toNat := Nat.mul_le_mul_right _ hlog'
  · rw [quantize_shift f r neg m e q hq he]
    generalize hk : (q - e).toNat = k
    have hs : r.shift neg m k = m / 2 ^ k ∨ r.shift neg m k = m / 2 ^ k + 1 := by
      cases h : r.up neg (m / 2 ^ k % 2 == 1) (m % 2 ^ k) k false <;> simp [Rounding.shift, h]
    have ht : m / 2 ^ k < 2 ^ (f.man + 1) := by
      apply (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos k)).2
      calc m < 2 ^ (Nat.log2 m + 1) := hlog
        _ ≤ 2 ^ (f.man + 1 + k) := Nat.pow_le_pow_right (by decide) (by omega)
        _ = 2 ^ (f.man + 1) * 2 ^ k := Nat.pow_add 2 _ _
    have hpm : 2 ^ f.man < 2 ^ (f.man + 1) := Nat.pow_lt_pow_right (by decide) (by omega)
    split
    · exact ⟨hpm, by omega, Or.inr (Nat.le_refl _)⟩
    · rename_i hne
      refine ⟨by omega, hq2, ?_⟩
      rcases hq3 with h | h
      · exact Or.inl h
      · right
        have hpow : 2 ^ f.man * 2 ^ k = 2 ^ Nat.log2 m := by
          rw [← Nat.pow_add]; congr 1; omega
        have : 2 ^ f.man ≤ m / 2 ^ k :=
          (Nat.le_div_iff_mul_le (Nat.two_pow_pos k)).2 (by rw [hpow]; exact hlog')
        omega

theorem expMax_pos (f : Format) (he : 0 < f.exp) : f.expMax ≠ 0 := by
  unfold expMax; try simp only [pow2_eq]
  have : 2 ≤ 2 ^ f.exp := by
    calc 2 = 2 ^ 1 := rfl
      _ ≤ 2 ^ f.exp := Nat.pow_le_pow_right (by decide) he
  omega

/-- Exponent field 0 of a format with subnormals: the subnormals and zero. -/
theorem decode_pack_zero (f : Format) (s : Bool) (F : Nat) (hF : F < 2 ^ f.man)
    (hs : f.subnormals = true) (he : 0 < f.exp) :
    f.decode (f.pack s 0 F) = .fin (s && f.signed) F f.qMin := by
  have hne := expMax_pos f he
  rw [decode_pack f s 0 F (Nat.two_pow_pos _) hF]
  simp [hs, Ne.symm hne]

/-- A normal encoding at or below the largest finite one. -/
theorem decode_pack_normal (f : Format) (he : 0 < f.exp) (s : Bool) (E F : Nat)
    (hE : E < 2 ^ f.exp) (hF : F < 2 ^ f.man) (hsub : ¬(E = 0 ∧ f.subnormals = true))
    (htop : E < f.topFinite.1 ∨ (E = f.topFinite.1 ∧ F ≤ f.topFinite.2)) :
    f.decode (f.pack s E F) = .fin (s && f.signed) (2 ^ f.man + F) ((E : Int) - f.bias - f.man) := by
  rw [decode_pack f s E F hE hF]
  have hp := Nat.two_pow_pos f.man
  have hne := expMax_pos f he
  have h1 : ¬(E = f.expMax ∧ f.special = .ieee) := by
    rintro ⟨h1, h2⟩; unfold topFinite at htop; simp only [pow2_eq, h2] at htop; omega
  have hp2 : f.man = 0 ∨ 2 ≤ 2 ^ f.man := by
    rcases Nat.eq_zero_or_pos f.man with h | h
    · exact Or.inl h
    · exact Or.inr (Nat.pow_le_pow_right (n := 2) (by decide) h)
  have h2 : ¬(E = f.expMax ∧ f.special = .nanOnly ∧ F = 2 ^ f.man - 1) := by
    rintro ⟨h1, h2, h3⟩
    unfold topFinite at htop; try simp only [pow2_eq] at htop
    by_cases hm0 : f.man = 0
    · simp only [h2, hm0, ite_true] at htop; omega
    · simp only [h2, hm0, ite_false] at htop; omega
  simp only [h1, h2, hsub, ite_false]

/-- A finite rounded result denotes the significand step's value (no overflow happened: an IEEE
or NaN-only format overflows to a non-finite encoding unless saturating). -/
theorem decode_round (f : Format) (hs : f.subnormals = true) (hspec : f.special ≠ .finite)
    (he : 0 < f.exp) (r : Rounding) (ovf : Overflow) (hovf : ovf ≠ .sat) (neg : Bool) (m : Nat)
    (e : Int) (hm : m ≠ 0) (hr : ∀ b, r.away b = true) (n : Bool) (M0 : Nat) (Q0 : Int)
    (hb : f.decode (f.round r ovf neg m e) = .fin n M0 Q0) (L : Int) :
    M0 * 2 ^ (Q0 - L).toNat = (f.quantize r.up (neg && f.signed) m e false).1 *
      2 ^ ((f.quantize r.up (neg && f.signed) m e false).2 - L).toNat := by
  obtain ⟨hlt, hqm, hnorm⟩ := quantize_bounds f r (neg && f.signed) m e hm
  have hx : f.expMax < 2 ^ f.exp := by
    unfold expMax; simp only [pow2_eq]; have := Nat.two_pow_pos f.exp; omega
  have htop : f.topFinite.1 ≤ f.expMax := by
    unfold topFinite; (try simp only [pow2_eq]); split <;> (try split) <;> simp <;> omega
  unfold round roundWith at hb
  simp only [pow2_eq] at hb
  obtain ⟨⟨hi, q⟩, hpq⟩ : ∃ p, f.quantize r.up (neg && f.signed) m e false = p := ⟨_, rfl⟩
  rw [hpq] at hb hlt hqm hnorm ⊢
  simp only at hlt hqm hnorm ⊢
  simp only [hm, hs, hr, Bool.not_true, Bool.false_and, Bool.false_eq_true, ite_false, ite_true]
    at hb
  have hfr : hi % 2 ^ f.man < 2 ^ f.man := Nat.mod_lt _ (Nat.two_pow_pos _)
  by_cases h0 : hi = 0
  · subst h0
    simp only [ite_true, zero, decode_pack_zero f _ 0 (Nat.two_pow_pos _) hs he,
      Val.fin.injEq] at hb
    obtain ⟨_, rfl, _⟩ := hb
    simp
  · simp only [h0, ite_false] at hb
    by_cases hsub : hi < 2 ^ f.man
    · have hq : q = f.qMin := by rcases hnorm with h | h <;> omega
      simp only [hsub, ite_true] at hb
      split at hb
      · exact absurd hb (overflow_not_fin f hspec ovf hovf _ n M0 Q0 he)
      · rw [decode_pack_zero f _ _ hfr hs he, Val.fin.injEq] at hb
        obtain ⟨_, rfl, rfl⟩ := hb
        rw [hq, Nat.mod_eq_of_lt hsub]
    · simp only [hsub, ite_false] at hb
      split at hb
      · exact absurd hb (overflow_not_fin f hspec ovf hovf _ n M0 Q0 he)
      · rename_i hno
        rw [decode_pack_normal f he _ _ _ (by omega) hfr (by omega) (by omega), Val.fin.injEq] at hb
        obtain ⟨_, rfl, rfl⟩ := hb
        have hhi : 2 ^ f.man + hi % 2 ^ f.man = hi := by
          have h1 : hi / 2 ^ f.man = 1 := by
            apply Nat.div_eq_of_lt_le
            · omega
            · rw [Nat.pow_succ] at hlt; omega
          have := Nat.div_add_mod hi (2 ^ f.man)
          rw [h1] at this; omega
        have hex : ((((q - f.qMin).toNat + 1 : Nat) : Int) - f.bias - f.man) = q := by
          unfold qMin at hqm ⊢; simp only [hs, ite_true] at hqm ⊢; omega
        rw [hhi, hex]

/-- Nearest even at the format level: when rounding an exact value `m · 2^e` gives a finite
result in an IEEE or NaN-only format with subnormals (not saturating), no finite encoding of the
format is closer to it. Magnitudes are compared, exactly, scaled by a common `2^-L`. -/
theorem round_rn_nearest (f : Format) (hs : f.subnormals = true) (hspec : f.special ≠ .finite)
    (he : 0 < f.exp) (ovf : Overflow) (hovf : ovf ≠ .sat) (neg : Bool) (m : Nat) (e : Int)
    (hm : m ≠ 0) (n : Bool) (M0 : Nat) (Q0 : Int)
    (hb : f.decode (f.round .rn ovf neg m e) = .fin n M0 Q0)
    (s : Bool) (E F : Nat) (hE : E < 2 ^ f.exp) (hF : F < 2 ^ f.man) (n' : Bool) (M : Nat)
    (Q : Int) (hw : f.decode (f.pack s E F) = .fin n' M Q) (L : Int) (hL1 : L ≤ e)
    (hL2 : L ≤ f.qMin) :
    dist (m * 2 ^ (e - L).toNat) (M0 * 2 ^ (Q0 - L).toNat) ≤
      dist (m * 2 ^ (e - L).toNat) (M * 2 ^ (Q - L).toNat) := by
  obtain ⟨hQ, hM⟩ := decode_bounds f s E F hE hF n' M Q hw
  rw [decode_round f hs hspec he .rn ovf hovf neg m e hm (fun _ => rfl) n M0 Q0 hb L]
  exact quantize_rn_nearest f _ m e hm L hL1 hL2 M Q hQ hM

end Format

end NV.Num
