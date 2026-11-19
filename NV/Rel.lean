/-!
# Finite relations

The memory model works over executions with finitely many events, so relations are Boolean
tables over `Fin n`. A relation is stored as its table, the bits of one natural number (row-major),
so every relation built from others is computed once: in compiled code, and in the kernel, which
evaluates a table to a numeral and shares it (an array of unevaluated entries would be recomputed
at every lookup). `tc` is proven to be the transitive closure (`tc_iff_transGen`), so definitions
read as the relational algebra of the Alloy models they transcribe.
-/

namespace NV

/-- A binary relation over the events `Fin n`: bit `a * n + b` holds when `a` relates to `b`. -/
structure Rel (n : Nat) where
  bits : Nat

namespace Rel

variable {n : Nat}

def get (r : Rel n) (a b : Fin n) : Bool := r.bits.testBit (a.val * n + b.val)

instance : CoeFun (Rel n) (fun _ => Fin n → Fin n → Bool) := ⟨get⟩

theorem div_lt {i : Nat} (h : i < n * n) : i / n < n := Nat.div_lt_of_lt_mul h
theorem mod_lt {i : Nat} (h : i < n * n) : i % n < n :=
  Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 => by simp [h0] at h)

/-- Entry `i` of the row-major table of `f`. -/
def entry (f : Fin n → Fin n → Bool) (i : Nat) : Bool :=
  if h : i < n * n then f ⟨i / n, div_lt h⟩ ⟨i % n, mod_lt h⟩ else false

/-- The first `k` entries of the table of `f`, as bits. -/
def bitsBelow (f : Fin n → Fin n → Bool) : Nat → Nat
  | 0 => 0
  | k + 1 => bitsBelow f k ||| (if entry f k then 2 ^ k else 0)

theorem testBit_bitsBelow (f : Fin n → Fin n → Bool) (k j : Nat) :
    (bitsBelow f k).testBit j = (decide (j < k) && entry f j) := by
  induction k with
  | zero => simp [bitsBelow, Nat.zero_testBit]
  | succ k ih =>
    simp only [bitsBelow, Nat.testBit_or, ih]
    by_cases hj : j = k
    · subst hj; cases entry f j <;> simp [Nat.testBit_two_pow, Nat.zero_testBit]
    · have : (if entry f k = true then 2 ^ k else 0).testBit j = false := by
        split <;> simp [Nat.testBit_two_pow, Nat.zero_testBit] <;> omega
      rw [this]
      by_cases hl : j < k
      · simp [hl, show j < k + 1 by omega]
      · simp [hl, show ¬ j < k + 1 by omega]

/-- The relation with the given membership function, tabulated. -/
def ofFn (f : Fin n → Fin n → Bool) : Rel n := ⟨bitsBelow f (n * n)⟩

theorem index_lt {a b : Fin n} : a.val * n + b.val < n * n := by
  have := a.isLt; have := b.isLt
  calc a.val * n + b.val < a.val * n + n := by omega
    _ = (a.val + 1) * n := by rw [Nat.succ_mul]
    _ ≤ n * n := Nat.mul_le_mul_right _ (by omega)

@[simp] theorem get_ofFn (f : Fin n → Fin n → Bool) (a b : Fin n) : ofFn f a b = f a b := by
  have hb := b.isLt
  have hd : (a.val * n + b.val) / n = a.val := by
    rw [Nat.mul_comm, Nat.mul_add_div (by omega), Nat.div_eq_of_lt hb, Nat.add_zero]
  have hm : (a.val * n + b.val) % n = b.val := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hb]
  show (bitsBelow f (n * n)).testBit (a.val * n + b.val) = f a b
  rw [testBit_bitsBelow, entry, dif_pos index_lt]
  simp only [index_lt, decide_true, Bool.true_and]
  congr 1 <;> exact Fin.ext (by simp [hd, hm])

def empty : Rel n := ofFn fun _ _ => false
def iden : Rel n := ofFn fun a b => a == b
def union (r s : Rel n) : Rel n := ofFn fun a b => r a b || s a b
def inter (r s : Rel n) : Rel n := ofFn fun a b => r a b && s a b
def diff (r s : Rel n) : Rel n := ofFn fun a b => r a b && !s a b
def inv (r : Rel n) : Rel n := ofFn fun a b => r b a
def comp (r s : Rel n) : Rel n := ofFn fun a c => (List.finRange n).any fun b => r a b && s b c
/-- `r?`: the reflexive closure. -/
def opt (r : Rel n) : Rel n := union iden r
/-- `[p]`: the identity restricted to the events satisfying `p`. -/
def dom (p : Fin n → Bool) : Rel n := ofFn fun a b => a == b && p a
/-- `p × q`. -/
def prod (p q : Fin n → Bool) : Rel n := ofFn fun a b => p a && q b

instance : Union (Rel n) := ⟨union⟩
instance : Inter (Rel n) := ⟨inter⟩
instance : SDiff (Rel n) := ⟨diff⟩

infixl:75 " ⨾ " => comp

/-- One Warshall step: allow `k` as an intermediate vertex. -/
def through (r : Rel n) (k : Fin n) : Rel n := ofFn fun a b => r a b || (r a k && r k b)

/-- `warshall r k` relates `a` to `b` when a path joins them through vertices below `k`. -/
def warshall (r : Rel n) : Nat → Rel n
  | 0 => r
  | k + 1 => if h : k < n then through (warshall r k) ⟨k, h⟩ else warshall r k

/-- `r⁺`: the transitive closure. -/
def tc (r : Rel n) : Rel n := warshall r n

/-- `r*`: the reflexive-transitive closure. -/
def rtc (r : Rel n) : Rel n := opt (tc r)

def irreflexive (r : Rel n) : Bool := (List.finRange n).all fun a => !r a a
def acyclic (r : Rel n) : Bool := irreflexive (tc r)
def isEmpty (r : Rel n) : Bool := (List.finRange n).all fun a => (List.finRange n).all fun b => !r a b
def subset (r s : Rel n) : Bool := isEmpty (diff r s)

/-- Paths of `r` whose intermediate vertices are all below `k`. -/
inductive PathBelow (r : Rel n) (k : Nat) : Fin n → Fin n → Prop
  | edge {a b} : r a b = true → PathBelow r k a b
  | trans {a m b} : PathBelow r k a m → m.val < k → PathBelow r k m b → PathBelow r k a b

theorem PathBelow.mono {r : Rel n} {k k' : Nat} (hk : k ≤ k') {a b : Fin n}
    (h : PathBelow r k a b) : PathBelow r k' a b := by
  induction h with
  | edge e => exact .edge e
  | trans _ hm _ ih₁ ih₂ => exact .trans ih₁ (by omega) ih₂

/-- The Warshall invariant: a path through vertices below `k + 1` either avoids `k`, or splits at
`k` into two paths through vertices below `k`. -/
theorem PathBelow.split {r : Rel n} {k : Nat} {a b : Fin n} (hk : k < n)
    (h : PathBelow r (k + 1) a b) :
    PathBelow r k a b ∨ (PathBelow r k a ⟨k, hk⟩ ∧ PathBelow r k ⟨k, hk⟩ b) := by
  induction h with
  | edge e => exact .inl (.edge e)
  | @trans a m b _ hm _ ih₁ ih₂ =>
    by_cases hmk : m.val < k
    · rcases ih₁ with p₁ | ⟨p₁, q₁⟩ <;> rcases ih₂ with p₂ | ⟨p₂, q₂⟩
      · exact .inl (.trans p₁ hmk p₂)
      · exact .inr ⟨.trans p₁ hmk p₂, q₂⟩
      · exact .inr ⟨p₁, .trans q₁ hmk p₂⟩
      · exact .inr ⟨p₁, q₂⟩
    · have hm' : m = ⟨k, hk⟩ := Fin.ext (by simp only; omega)
      subst hm'
      have l : PathBelow r k a ⟨k, hk⟩ := by rcases ih₁ with p | ⟨p, _⟩ <;> exact p
      have rr : PathBelow r k ⟨k, hk⟩ b := by rcases ih₂ with p | ⟨_, p⟩ <;> exact p
      exact .inr ⟨l, rr⟩

theorem warshall_iff (r : Rel n) (k : Nat) (hk : k ≤ n) (a b : Fin n) :
    warshall r k a b = true ↔ PathBelow r k a b := by
  induction k generalizing a b with
  | zero =>
    refine ⟨.edge, fun h => ?_⟩
    induction h with
    | edge e => exact e
    | trans _ hm _ => omega
  | succ k ih =>
    have hk' : k < n := by omega
    have ih := ih (by omega)
    simp only [warshall, hk', dite_true, through, get_ofFn, Bool.or_eq_true, Bool.and_eq_true]
    constructor
    · rintro (h | ⟨h₁, h₂⟩)
      · exact ((ih a b).1 h).mono (by omega)
      · exact .trans (((ih _ _).1 h₁).mono (by omega)) (by simp) (((ih _ _).1 h₂).mono (by omega))
    · intro h
      rcases h.split hk' with p | ⟨p, q⟩
      · exact .inl ((ih a b).2 p)
      · exact .inr ⟨(ih _ _).2 p, (ih _ _).2 q⟩

theorem tc_iff_transGen (r : Rel n) (a b : Fin n) :
    tc r a b = true ↔ Relation.TransGen (fun x y => r x y = true) a b := by
  rw [tc, warshall_iff r n (Nat.le_refl n)]
  constructor
  · intro h
    induction h with
    | edge e => exact .single e
    | trans _ _ _ ih₁ ih₂ => exact ih₁.trans ih₂
  · intro h
    induction h with
    | single e => exact .edge e
    | tail _ e ih => exact .trans ih (Fin.isLt _) (.edge e)

/-! ## Membership lemmas -/

@[simp] theorem get_union (r s : Rel n) (a b : Fin n) : (r ∪ s) a b = (r a b || s a b) := by
  show (union r s) a b = _; simp [union]
@[simp] theorem get_inter (r s : Rel n) (a b : Fin n) : (r ∩ s) a b = (r a b && s a b) := by
  show (inter r s) a b = _; simp [inter]
@[simp] theorem get_inv (r : Rel n) (a b : Fin n) : inv r a b = r b a := by simp [inv]
@[simp] theorem get_dom (p : Fin n → Bool) (a b : Fin n) : dom p a b = (a == b && p a) := by simp [dom]
@[simp] theorem get_iden (a b : Fin n) : iden a b = (a == b) := by simp [iden]
@[simp] theorem get_opt (r : Rel n) (a b : Fin n) : opt r a b = (a == b || r a b) := by simp [opt, union]
@[simp] theorem get_prod (p q : Fin n → Bool) (a b : Fin n) : prod p q a b = (p a && q b) := by simp [prod]

theorem get_comp {r s : Rel n} {a c : Fin n} : (r ⨾ s) a c = true ↔ ∃ b, r a b = true ∧ s b c = true := by
  simp [comp, List.any_eq_true]

theorem dom_comp_iff {p : Fin n → Bool} {r : Rel n} {a b : Fin n} :
    (dom p ⨾ r) a b = true ↔ p a = true ∧ r a b = true := by
  rw [get_comp]
  constructor
  · rintro ⟨c, h₁, h₂⟩
    simp only [get_dom, Bool.and_eq_true, beq_iff_eq] at h₁
    obtain ⟨rfl, h⟩ := h₁
    exact ⟨h, h₂⟩
  · rintro ⟨h₁, h₂⟩
    exact ⟨a, by simp [h₁], h₂⟩

theorem comp_dom_iff {p : Fin n → Bool} {r : Rel n} {a b : Fin n} :
    (r ⨾ dom p) a b = true ↔ r a b = true ∧ p b = true := by
  rw [get_comp]
  constructor
  · rintro ⟨c, h₁, h₂⟩
    simp only [get_dom, Bool.and_eq_true, beq_iff_eq] at h₂
    obtain ⟨rfl, h⟩ := h₂
    exact ⟨h₁, h⟩
  · rintro ⟨h₁, h₂⟩
    exact ⟨b, h₁, by simp [h₂]⟩

theorem comp_of {r s : Rel n} {a b c : Fin n} (h₁ : r a b = true) (h₂ : s b c = true) :
    (r ⨾ s) a c = true := get_comp.2 ⟨b, h₁, h₂⟩

theorem tc_of {r : Rel n} {a b : Fin n} (h : r a b = true) : tc r a b = true :=
  (tc_iff_transGen r a b).2 (.single h)

theorem tc_trans {r : Rel n} {a b c : Fin n} (h₁ : tc r a b = true) (h₂ : tc r b c = true) :
    tc r a c = true :=
  (tc_iff_transGen r a c).2 (((tc_iff_transGen r a b).1 h₁).trans ((tc_iff_transGen r b c).1 h₂))

theorem tc_mono {r s : Rel n} (h : ∀ a b, r a b = true → s a b = true) {a b : Fin n}
    (hab : tc r a b = true) : tc s a b = true := by
  rw [tc_iff_transGen] at hab ⊢
  induction hab with
  | single e => exact .single (h _ _ e)
  | tail _ e ih => exact ih.tail (h _ _ e)

/-- A path starts with an edge. -/
theorem tc_first {r : Rel n} {a b : Fin n} (h : tc r a b = true) : ∃ v, r a v = true := by
  rw [tc_iff_transGen] at h
  induction h with
  | single e => exact ⟨_, e⟩
  | tail _ _ ih => exact ih

theorem rtc_refl (r : Rel n) (a : Fin n) : rtc r a a = true := by simp [rtc]

theorem rtc_of_tc {r : Rel n} {a b : Fin n} (h : tc r a b = true) : rtc r a b = true := by
  simp [rtc, h]

theorem irreflexive_iff (r : Rel n) : irreflexive r = true ↔ ∀ a, r a a = false := by
  simp [irreflexive, List.all_eq_true]

theorem isEmpty_iff (r : Rel n) : isEmpty r = true ↔ ∀ a b, r a b = false := by
  simp [isEmpty, List.all_eq_true]

theorem subset_iff (r s : Rel n) : subset r s = true ↔ ∀ a b, r a b = true → s a b = true := by
  simp only [subset, isEmpty_iff, diff, get_ofFn]
  constructor
  · intro h a b hab; have := h a b; simp_all
  · intro h a b; cases hr : r a b <;> simp [h a b, hr]

/-! ## A linear extension of an acyclic relation -/

theorem countP_le_of_imp {α : Type} {p q : α → Bool} :
    ∀ {l : List α}, (∀ a ∈ l, p a = true → q a = true) → l.countP p ≤ l.countP q
  | [], _ => by simp
  | a :: l, h => by
    simp only [List.countP_cons]
    have := countP_le_of_imp (l := l) fun b hb => h b (by simp [hb])
    have ha := h a (by simp)
    split <;> split <;> simp_all <;> omega

theorem countP_lt_of_imp {α : Type} {p q : α → Bool} :
    ∀ {l : List α} {y : α}, (∀ a ∈ l, p a = true → q a = true) → y ∈ l → q y = true → p y = false →
      l.countP p < l.countP q
  | [], _, _, hy, _, _ => by simp at hy
  | a :: l, y, h, hy, hq, hp => by
    simp only [List.countP_cons]
    have hle := countP_le_of_imp (l := l) fun b hb => h b (by simp [hb])
    have ha := h a (by simp)
    rcases List.mem_cons.1 hy with rfl | hy
    · simp [hq, hp]; omega
    · have := countP_lt_of_imp (fun b hb => h b (by simp [hb])) hy hq hp
      split <;> split <;> simp_all <;> omega

/-- The number of predecessors of `a` under `r⁺`. -/
def preds (r : Rel n) (a : Fin n) : Nat := (List.finRange n).countP fun b => tc r b a

theorem preds_lt {r : Rel n} (hr : acyclic r = true) {a b : Fin n} (h : tc r a b = true) :
    preds r a < preds r b := by
  simp only [acyclic, irreflexive_iff] at hr
  exact countP_lt_of_imp (fun c _ hc => tc_trans hc h) (List.mem_finRange a) h (hr a)

/-- A strict total order on `Fin n` extending `r⁺` when `r` is acyclic: by predecessor count, ties
broken by index. -/
def rank (r : Rel n) (a : Fin n) : Nat := preds r a * n + a.val

theorem rank_lt {r : Rel n} (hr : acyclic r = true) {a b : Fin n} (h : tc r a b = true) :
    rank r a < rank r b := by
  have hp := preds_lt hr h
  have := a.isLt
  unfold rank
  calc preds r a * n + a.val < preds r a * n + n := by omega
    _ = (preds r a + 1) * n := by rw [Nat.succ_mul]
    _ ≤ preds r b * n := Nat.mul_le_mul_right _ hp
    _ ≤ preds r b * n + b.val := Nat.le_add_right _ _

theorem rank_inj (r : Rel n) {a b : Fin n} (h : rank r a = rank r b) : a = b := by
  unfold rank at h
  have := congrArg (· % n) h
  simp only [Nat.mul_comm _ n, Nat.mul_add_mod, Nat.mod_eq_of_lt a.isLt, Nat.mod_eq_of_lt b.isLt] at this
  exact Fin.ext this

end Rel
end NV
