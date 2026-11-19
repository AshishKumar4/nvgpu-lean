import NV.Bounds.LoomisWhitney

/-!
# The I/O lower bound for classical matrix multiplication

Hong and Kung's red-blue pebble game (STOC 1981) asks how much data a computation must move
between a slow memory and a fast memory of `S` words. For `C = A·B` with `A` of `M × K` and `B`
of `K × N`, in the class of *classical* algorithms (each `c_ij` a sum of products
`a_ik · b_kj`, every product formed in fast memory), the moves are at least about
`M N K / √S`; Irony, Toledo and Tiskin (2004) prove it with Loomis–Whitney, Kwasniewski et al.
(SC'19) sharpen the constant to `2 M N K / √S`.

## The class, stated

A processor runs a trace of operations over its fast memory, which holds at most `S` words:

* `load w` and `store w` move one word between slow and fast memory: the I/O counted;
* `mac i j k t` forms `a_ik · b_kj` into accumulator `t` of `c_ij`: `a_ik`, `b_kj` and that
  accumulator must be in fast memory;
* `fresh i j t` starts an accumulator at zero, `merge i j t u` adds accumulator `u` into `t` and
  frees `u` (split-K inside one memory), `drop w` frees a word;
* an accumulator that has taken a product since it was last stored is *dirty*; a dirty word is
  never dropped, and the trace ends with none dirty: no partial sum is discarded, every one
  reaches an output through slow memory.

Recomputation is allowed (a product may be formed many times); a partial sum may be stored and
reloaded (split-K across processors).

## The theorems

* `segment_bound`: a stretch of trace that does `q` I/Os, between states with at most `S`
  resident words, forms at most `v` distinct products with `27 v² ≤ (2S + q)³`. Its `A` and
  `B` operands were resident at its start or loaded during it; each `c_ij` it touches has a
  dirty accumulator at its end or was stored during it; Loomis–Whitney and AM–GM bound `v`.
* `mmm_io`: for every `T ≥ 1`, a trace that forms all `M N K` products has
  `T · M N K ≤ U(S, T) · (I/O + T)`, with `U(S, T) = ⌊√((2S + T)³ / 27)⌋`: cut the trace into
  stretches of `T` I/Os. At `T = 4S` this is `I/O ≥ √2 · M N K / √S - 4S`, the Irony–Toledo–Tiskin
  form; Kwasniewski et al.'s constant 2 is not proven here.
* `mmm_compulsory`: such a trace from an empty fast memory does at least `MK + KN + MN` I/Os.
* `mmm_io_parallel`: processors with their own fast memories, together forming every product,
  move at least `T · M N K / U(S, T) - P · T` words in total.
-/

namespace NV.Bounds.Pebble

open NV.Bounds

/-- A word of fast memory: an entry of `A` or `B`, or accumulator `t` of `c_ij`. -/
inductive Word where
  | a (i k : Nat)
  | b (k j : Nat)
  | c (i j t : Nat)
  deriving DecidableEq, Repr

inductive Op where
  | load (w : Word)
  | store (w : Word)
  | drop (w : Word)
  | mac (i j k t : Nat)
  | fresh (i j t : Nat)
  | merge (i j t u : Nat)
  deriving DecidableEq, Repr

structure St where
  fast : List Word
  dirty : List Word

/-- An accumulator of `c_ij`. -/
def isC (i j : Nat) : Word → Bool
  | .c i' j' _ => i' == i && j' == j
  | _ => false

def Word.isAB : Word → Bool
  | .c .. => false
  | _ => true

def step (s : St) : Op → Option St
  | .load w => some { s with fast := if w ∈ s.fast then s.fast else w :: s.fast }
  | .store w => if w ∈ s.fast then some { s with dirty := s.dirty.filter (· ≠ w) } else none
  | .drop w =>
    if w ∈ s.fast ∧ w ∉ s.dirty then some { s with fast := s.fast.filter (· ≠ w) } else none
  | .mac i j k t =>
    if Word.a i k ∈ s.fast ∧ Word.b k j ∈ s.fast ∧ Word.c i j t ∈ s.fast then
      some { s with dirty := Word.c i j t :: s.dirty }
    else none
  | .fresh i j t => if Word.c i j t ∈ s.fast then none else some { s with fast := Word.c i j t :: s.fast }
  | .merge i j t u =>
    if t ≠ u ∧ Word.c i j t ∈ s.fast ∧ Word.c i j u ∈ s.fast then
      some { fast := s.fast.filter (· ≠ Word.c i j u),
             dirty := Word.c i j t :: s.dirty.filter (· ≠ Word.c i j u) }
    else none

/-- Run a trace with at most `S` resident words after every operation. -/
def exec (S : Nat) : St → List Op → Option St
  | s, [] => some s
  | s, op :: ops =>
    match step s op with
    | none => none
    | some s₁ => if s₁.fast.length ≤ S then exec S s₁ ops else none

def loadOf : Op → List Word
  | .load w => [w]
  | _ => []

def storeOf : Op → List Word
  | .store w => [w]
  | _ => []

def loads (σ : List Op) : List Word := σ.flatMap loadOf
def stores (σ : List Op) : List Word := σ.flatMap storeOf

/-- Words moved between slow and fast memory. -/
def io (σ : List Op) : Nat := (loads σ).length + (stores σ).length

def isMac (i j k : Nat) : Op → Bool
  | .mac i' j' k' _ => i' == i && j' == j && k' == k
  | _ => false

/-- The products a trace forms. -/
def prods (σ : List Op) (i j k : Nat) : Bool := σ.any (isMac i j k)

/-! ## One step -/

theorem exec_cons {S : Nat} {s s' : St} {op : Op} {ops : List Op}
    (h : exec S s (op :: ops) = some s') :
    ∃ s₁, step s op = some s₁ ∧ s₁.fast.length ≤ S ∧ exec S s₁ ops = some s' := by
  simp only [exec] at h
  cases hs : step s op with
  | none => simp [hs] at h
  | some s₁ =>
    simp only [hs] at h
    split at h
    · exact ⟨s₁, rfl, ‹_›, h⟩
    · cases h

theorem exec_append {S : Nat} : ∀ {s s' : St} {σ ρ : List Op}, exec S s (σ ++ ρ) = some s' →
    ∃ s₁, exec S s σ = some s₁ ∧ exec S s₁ ρ = some s'
  | s, _, [], _, h => ⟨s, rfl, h⟩
  | s, _, op :: σ, ρ, h => by
    obtain ⟨s₁, h₁, hc, h₂⟩ := exec_cons h
    obtain ⟨s₂, h₃, h₄⟩ := exec_append h₂
    refine ⟨s₂, ?_, h₄⟩
    simp only [exec, h₁, hc, ↓reduceIte]
    exact h₃

theorem exec_cap {S : Nat} : ∀ {s s' : St} {σ : List Op}, exec S s σ = some s' →
    s.fast.length ≤ S → s'.fast.length ≤ S
  | _, _, [], h, hc => by cases h; exact hc
  | _, _, _ :: _, h, _ => by
    obtain ⟨_, _, hc₁, h₂⟩ := exec_cons h
    exact exec_cap h₂ hc₁

theorem step_inv {s s₁ : St} {op : Op} (h : step s op = some s₁) (hi : s.dirty ⊆ s.fast) :
    s₁.dirty ⊆ s₁.fast := by
  cases op with
  | load w =>
    simp only [step] at h; cases h
    intro x hx
    have := hi hx
    show x ∈ (if w ∈ s.fast then s.fast else w :: s.fast)
    split <;> simp_all
  | store w =>
    simp only [step] at h; split at h <;> cases h
    intro x hx
    exact hi ((List.mem_filter.1 hx).1)
  | drop w =>
    simp only [step] at h; split at h
    · next hw =>
      cases h
      intro x hx
      refine List.mem_filter.2 ⟨hi hx, ?_⟩
      have : x ≠ w := fun e => hw.2 (e ▸ hx)
      simpa using this
    · cases h
  | mac i j k t =>
    simp only [step] at h; split at h
    · next hw =>
      cases h
      intro x hx
      rcases List.mem_cons.1 hx with rfl | hx
      · exact hw.2.2
      · exact hi hx
    · cases h
  | fresh i j t =>
    simp only [step] at h; split at h
    · cases h
    · cases h
      intro x hx
      exact List.mem_cons_of_mem _ (hi hx)
  | merge i j t u =>
    simp only [step] at h; split at h
    · next hw =>
      cases h
      intro x hx
      rcases List.mem_cons.1 hx with rfl | hx
      · refine List.mem_filter.2 ⟨hw.2.1, ?_⟩
        have : Word.c i j t ≠ Word.c i j u := fun e => hw.1 (by cases e; rfl)
        simpa using this
      · obtain ⟨hx₁, hx₂⟩ := List.mem_filter.1 hx
        exact List.mem_filter.2 ⟨hi hx₁, hx₂⟩
    · cases h

theorem exec_inv {S : Nat} : ∀ {s s' : St} {σ : List Op}, exec S s σ = some s' →
    s.dirty ⊆ s.fast → s'.dirty ⊆ s'.fast
  | _, _, [], h, hi => by cases h; exact hi
  | _, _, _ :: _, h, hi => by
    obtain ⟨_, h₁, _, h₂⟩ := exec_cons h
    exact exec_inv h₂ (step_inv h₁ hi)

/-- `A` and `B` entries enter fast memory only by a load. -/
theorem step_ab {s s₁ : St} {op : Op} {w : Word} (h : step s op = some s₁) (hw : w ∈ s₁.fast)
    (hab : w.isAB = true) : w ∈ s.fast ∨ op = .load w := by
  cases op with
  | load w' =>
    simp only [step] at h; cases h
    have hw' : w ∈ (if w' ∈ s.fast then s.fast else w' :: s.fast) := hw
    split at hw'
    · exact .inl hw'
    · rcases List.mem_cons.1 hw' with rfl | hw'
      · exact .inr rfl
      · exact .inl hw'
  | store w' => simp only [step] at h; split at h <;> cases h; exact .inl hw
  | drop w' =>
    simp only [step] at h; split at h
    · cases h; exact .inl ((List.mem_filter.1 hw).1)
    · cases h
  | mac i j k t => simp only [step] at h; split at h <;> cases h; exact .inl hw
  | fresh i j t =>
    simp only [step] at h; split at h
    · cases h
    · cases h
      rcases List.mem_cons.1 hw with rfl | hw
      · cases hab
      · exact .inl hw
  | merge i j t u =>
    simp only [step] at h; split at h
    · cases h; exact .inl ((List.mem_filter.1 hw).1)
    · cases h

theorem step_mac {s s₁ : St} {i j k t : Nat} (h : step s (.mac i j k t) = some s₁) :
    Word.a i k ∈ s.fast ∧ Word.b k j ∈ s.fast ∧ Word.c i j t ∈ s₁.dirty := by
  simp only [step] at h; split at h
  · next hw => cases h; exact ⟨hw.1, hw.2.1, List.mem_cons_self⟩
  · cases h

/-! ## What a stretch of trace touches -/

theorem loads_cons (op : Op) (σ : List Op) : loads (op :: σ) = loadOf op ++ loads σ := by
  simp [loads]

theorem stores_cons (op : Op) (σ : List Op) : stores (op :: σ) = storeOf op ++ stores σ := by
  simp [stores]

/-- Every `A` and `B` operand of a product was resident at the start or loaded. -/
theorem exec_operands {S : Nat} : ∀ {s s' : St} {σ : List Op}, exec S s σ = some s' →
    ∀ {i j k t : Nat}, Op.mac i j k t ∈ σ →
      Word.a i k ∈ s.fast ++ loads σ ∧ Word.b k j ∈ s.fast ++ loads σ
  | _, _, [], _, _, _, _, _, hm => by cases hm
  | s, _, op :: σ, h, i, j, k, t, hm => by
    obtain ⟨s₁, h₁, _, h₂⟩ := exec_cons h
    rw [loads_cons]
    rcases List.mem_cons.1 hm with rfl | hm
    · have := step_mac h₁
      exact ⟨List.mem_append_left _ this.1, List.mem_append_left _ this.2.1⟩
    · have ih := exec_operands h₂ hm
      have lift : ∀ w, w.isAB = true → w ∈ s₁.fast ++ loads σ → w ∈ s.fast ++ (loadOf op ++ loads σ) := by
        intro w hab hw
        rcases List.mem_append.1 hw with hw | hw
        · rcases step_ab h₁ hw hab with hw | rfl
          · exact List.mem_append_left _ hw
          · exact List.mem_append_right _ (List.mem_append_left _ (by simp [loadOf]))
        · exact List.mem_append_right _ (List.mem_append_right _ hw)
      exact ⟨lift _ rfl ih.1, lift _ rfl ih.2⟩

/-- Some accumulator of `c_ij` is dirty, or one was stored. -/
def Good (s : St) (st : List Word) (i j : Nat) : Prop := (s.dirty ++ st).any (isC i j) = true

theorem good_mono {s : St} {st st' : List Word} {i j : Nat} (hs : ∀ w ∈ st, w ∈ st')
    (h : Good s st i j) : Good s st' i j := by
  unfold Good at *
  obtain ⟨w, hw, hc⟩ := List.any_eq_true.1 h
  refine List.any_eq_true.2 ⟨w, ?_, hc⟩
  rcases List.mem_append.1 hw with hw | hw
  · exact List.mem_append_left _ hw
  · exact List.mem_append_right _ (hs w hw)

theorem isC_eq {i j : Nat} {w : Word} (h : isC i j w = true) : ∃ t, w = .c i j t := by
  cases w with
  | c i' j' t =>
    simp only [isC, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨t, rfl⟩
  | _ => simp [isC] at h

/-- No operation loses the record that `c_ij` has a partial sum in flight. -/
theorem step_good {s s₁ : St} {op : Op} {st : List Word} {i j : Nat} (h : step s op = some s₁)
    (hg : Good s st i j) : Good s₁ (st ++ storeOf op) i j := by
  unfold Good at *
  obtain ⟨w, hw, hc⟩ := List.any_eq_true.1 hg
  apply List.any_eq_true.2
  have keep : w ∈ s₁.dirty → ∃ x, x ∈ s₁.dirty ++ (st ++ storeOf op) ∧ isC i j x = true :=
    fun h' => ⟨w, List.mem_append_left _ h', hc⟩
  rcases List.mem_append.1 hw with hw | hw
  · cases op with
    | load w' => simp only [step] at h; cases h; exact keep hw
    | store w' =>
      simp only [step] at h; split at h
      · cases h
        by_cases e : w = w'
        · subst e
          exact ⟨w, List.mem_append_right _ (List.mem_append_right _ (by simp [storeOf])), hc⟩
        · exact keep (List.mem_filter.2 ⟨hw, by simpa using e⟩)
      · cases h
    | drop w' =>
      simp only [step] at h; split at h
      · cases h; exact keep hw
      · cases h
    | mac i' j' k' t' =>
      simp only [step] at h; split at h
      · cases h; exact keep (List.mem_cons_of_mem _ hw)
      · cases h
    | fresh i' j' t' =>
      simp only [step] at h; split at h
      · cases h
      · cases h; exact keep hw
    | merge i' j' t' u' =>
      simp only [step] at h; split at h
      · cases h
        by_cases e : w = Word.c i' j' u'
        · subst e
          obtain ⟨_, he⟩ := isC_eq hc
          cases he
          exact ⟨_, List.mem_append_left _ List.mem_cons_self, by simp [isC]⟩
        · exact keep (List.mem_cons_of_mem _ (List.mem_filter.2 ⟨hw, by simpa using e⟩))
      · cases h
  · exact ⟨w, List.mem_append_right _ (List.mem_append_left _ hw), hc⟩

theorem exec_good {S : Nat} : ∀ {s s' : St} {σ : List Op} {st : List Word} {i j : Nat},
    exec S s σ = some s' → Good s st i j → Good s' (st ++ stores σ) i j
  | _, _, [], _, _, _, h, hg => by cases h; simpa [stores] using hg
  | _, _, op :: σ, st, _, _, h, hg => by
    obtain ⟨_, h₁, _, h₂⟩ := exec_cons h
    have := exec_good h₂ (step_good h₁ hg)
    rw [stores_cons]
    simpa [List.append_assoc] using this

/-- Every `c_ij` a stretch forms a product of has a dirty accumulator at its end, or had one
stored during it. -/
theorem exec_products {S : Nat} : ∀ {s s' : St} {σ : List Op}, exec S s σ = some s' →
    ∀ {i j k t : Nat}, Op.mac i j k t ∈ σ → Good s' (stores σ) i j
  | _, _, [], _, _, _, _, _, hm => by cases hm
  | s, _, op :: σ, h, i, j, k, t, hm => by
    obtain ⟨s₁, h₁, _, h₂⟩ := exec_cons h
    rw [stores_cons]
    rcases List.mem_cons.1 hm with rfl | hm
    · have hg : Good s₁ [] i j :=
        List.any_eq_true.2 ⟨Word.c i j t, by simpa using (step_mac h₁).2.2, by simp [isC]⟩
      have := exec_good h₂ hg
      simpa [storeOf] using this
    · exact good_mono (fun w hw => List.mem_append_right _ hw) (exec_products h₂ hm)

/-! ## Counting words -/

section
variable (M N K : Nat)

/-- The box points a list of words covers: its `A` entries, `B` entries and `C` positions. -/
def mass (L : List Word) : Nat :=
  (sumR M fun i => sumR K fun k => ind (decide (Word.a i k ∈ L))) +
  (sumR K fun k => sumR N fun j => ind (decide (Word.b k j ∈ L))) +
  (sumR M fun i => sumR N fun j => ind (L.any (isC i j)))

theorem mass_nil : mass M N K [] = 0 := by
  simp [mass, ind, sumR_zero]

/-- One word covers at most one box point. -/
theorem mass_cons (w : Word) (L : List Word) : mass M N K (w :: L) ≤ mass M N K L + 1 := by
  have hA := sum2_le_add (M := M) (K := K)
    (f := fun i k => ind (decide (Word.a i k ∈ w :: L)))
    (g := fun i k => ind (decide (Word.a i k ∈ L))) (h := fun i k => ind (decide (Word.a i k = w)))
    (fun i _ k _ => by
      by_cases e : Word.a i k = w <;> by_cases m : Word.a i k ∈ L <;> simp [e, m, ind])
  have hB := sum2_le_add (M := K) (K := N)
    (f := fun k j => ind (decide (Word.b k j ∈ w :: L)))
    (g := fun k j => ind (decide (Word.b k j ∈ L))) (h := fun k j => ind (decide (Word.b k j = w)))
    (fun k _ j _ => by
      by_cases e : Word.b k j = w <;> by_cases m : Word.b k j ∈ L <;> simp [e, m, ind])
  have hC := sum2_le_add (M := M) (K := N)
    (f := fun i j => ind ((w :: L).any (isC i j)))
    (g := fun i j => ind (L.any (isC i j))) (h := fun i j => ind (isC i j w))
    (fun i _ j _ => by
      rw [List.any_cons, Bool.or_comm]
      exact ind_or _ _)
  -- The word's own points.
  have hw : (sumR M fun i => sumR K fun k => ind (decide (Word.a i k = w))) +
      (sumR K fun k => sumR N fun j => ind (decide (Word.b k j = w))) +
      (sumR M fun i => sumR N fun j => ind (isC i j w)) ≤ 1 := by
    cases w with
    | a i₀ k₀ =>
      have h₁ : ∀ i < M, ∀ k < K, ind (decide (Word.a i k = Word.a i₀ k₀)) =
          ind (decide (i = i₀ ∧ k = k₀)) := fun i _ k _ => by simp
      have h₂ : ∀ k < K, ∀ j < N, ind (decide (Word.b k j = Word.a i₀ k₀)) = 0 :=
        fun _ _ _ _ => by simp [ind]
      have h₃ : ∀ i < M, ∀ j < N, ind (isC i j (Word.a i₀ k₀)) = 0 :=
        fun _ _ _ _ => by simp [isC, ind]
      rw [sumR_congr fun i hi => sumR_congr (h₁ i hi), sumR_congr fun k hk => sumR_congr (h₂ k hk),
        sumR_congr fun i hi => sumR_congr (h₃ i hi)]
      simp only [sumR_zero]
      have := sum2_ind_eq_le M K i₀ k₀
      omega
    | b k₀ j₀ =>
      have h₁ : ∀ i < M, ∀ k < K, ind (decide (Word.a i k = Word.b k₀ j₀)) = 0 :=
        fun _ _ _ _ => by simp [ind]
      have h₂ : ∀ k < K, ∀ j < N, ind (decide (Word.b k j = Word.b k₀ j₀)) =
          ind (decide (k = k₀ ∧ j = j₀)) := fun _ _ _ _ => by simp
      have h₃ : ∀ i < M, ∀ j < N, ind (isC i j (Word.b k₀ j₀)) = 0 :=
        fun _ _ _ _ => by simp [isC, ind]
      rw [sumR_congr fun i hi => sumR_congr (h₁ i hi), sumR_congr fun k hk => sumR_congr (h₂ k hk),
        sumR_congr fun i hi => sumR_congr (h₃ i hi)]
      simp only [sumR_zero]
      have := sum2_ind_eq_le K N k₀ j₀
      omega
    | c i₀ j₀ t₀ =>
      have h₁ : ∀ i < M, ∀ k < K, ind (decide (Word.a i k = Word.c i₀ j₀ t₀)) = 0 :=
        fun _ _ _ _ => by simp [ind]
      have h₂ : ∀ k < K, ∀ j < N, ind (decide (Word.b k j = Word.c i₀ j₀ t₀)) = 0 :=
        fun _ _ _ _ => by simp [ind]
      have h₃ : ∀ i < M, ∀ j < N, ind (isC i j (Word.c i₀ j₀ t₀)) =
          ind (decide (i = i₀ ∧ j = j₀)) := fun i _ j _ => by
        simp only [isC]
        by_cases e₁ : i = i₀ <;> by_cases e₂ : j = j₀ <;>
          simp [e₁, e₂, ind, @eq_comm _ i₀ i, @eq_comm _ j₀ j]
      rw [sumR_congr fun i hi => sumR_congr (h₁ i hi), sumR_congr fun k hk => sumR_congr (h₂ k hk),
        sumR_congr fun i hi => sumR_congr (h₃ i hi)]
      simp only [sumR_zero]
      have := sum2_ind_eq_le M N i₀ j₀
      omega
  unfold mass
  omega

/-- A list of words covers at most as many box points as it has words. -/
theorem mass_le : ∀ L : List Word, mass M N K L ≤ L.length
  | [] => by rw [mass_nil]; exact Nat.zero_le _
  | w :: L => by
    have := mass_cons M N K w L
    have := mass_le L
    simp only [List.length_cons]
    omega

end

/-! ## One stretch -/

theorem prods_mac {σ : List Op} {i j k : Nat} (h : prods σ i j k = true) : ∃ t, Op.mac i j k t ∈ σ := by
  obtain ⟨op, hop, hm⟩ := List.any_eq_true.1 h
  cases op with
  | mac i' j' k' t =>
    simp only [isMac, Bool.and_eq_true, beq_iff_eq] at hm
    obtain ⟨⟨rfl, rfl⟩, rfl⟩ := hm
    exact ⟨t, hop⟩
  | _ => simp [isMac] at hm

/-- **One stretch.** A stretch of trace that does `io σ` I/Os, from a state with at most `S`
resident words, forms at most `v` distinct products in the box, `27 v² ≤ (2S + io σ)³`. -/
theorem segment_bound (M N K S : Nat) {s s' : St} {σ : List Op} (h : exec S s σ = some s')
    (hi : s.dirty ⊆ s.fast) (hc : s.fast.length ≤ S) :
    27 * (card M N K (prods σ) * card M N K (prods σ)) ≤
      (2 * S + io σ) * (2 * S + io σ) * (2 * S + io σ) := by
  let X := s.fast ++ loads σ
  let Y := s'.fast ++ stores σ
  have hc' := exec_cap h hc
  have hi' := exec_inv h hi
  have hAB : projIK M N K (prods σ) + projKJ M N K (prods σ) ≤ X.length := by
    have hA : projIK M N K (prods σ) ≤ sumR M fun i => sumR K fun k => ind (decide (Word.a i k ∈ X)) :=
      sum2_le fun i _ k _ => ind_le_of fun hp => by
        obtain ⟨j, _, hj⟩ := anyR_iff.1 hp
        obtain ⟨t, ht⟩ := prods_mac hj
        exact decide_eq_true (exec_operands h ht).1
    have hB : projKJ M N K (prods σ) ≤ sumR K fun k => sumR N fun j => ind (decide (Word.b k j ∈ X)) :=
      sum2_le fun k _ j _ => ind_le_of fun hp => by
        obtain ⟨i, _, hj⟩ := anyR_iff.1 hp
        obtain ⟨t, ht⟩ := prods_mac hj
        exact decide_eq_true (exec_operands h ht).2
    have := mass_le M N K X
    unfold mass at this
    omega
  have hC : projIJ M N K (prods σ) ≤ Y.length := by
    have hC : projIJ M N K (prods σ) ≤ sumR M fun i => sumR N fun j => ind (Y.any (isC i j)) :=
      sum2_le fun i _ j _ => ind_le_of fun hp => by
        obtain ⟨k, _, hk⟩ := anyR_iff.1 hp
        obtain ⟨t, ht⟩ := prods_mac hk
        obtain ⟨w, hw, hcw⟩ := List.any_eq_true.1 (exec_products h ht)
        refine List.any_eq_true.2 ⟨w, ?_, hcw⟩
        rcases List.mem_append.1 hw with hw | hw
        · exact List.mem_append_left _ (hi' hw)
        · exact List.mem_append_right _ hw
    have := mass_le M N K Y
    unfold mass at this
    omega
  have hX : X.length + Y.length ≤ 2 * S + io σ := by
    simp only [X, Y, List.length_append, io]
    omega
  have hLW := loomis_whitney M N K (prods σ)
  generalize projIK M N K (prods σ) = A at hAB hLW
  generalize projKJ M N K (prods σ) = B at hAB hLW
  generalize projIJ M N K (prods σ) = C at hC hLW
  generalize card M N K (prods σ) = v at hLW ⊢
  have h₂ := amgm2 A B
  have h₃ := amgm3 (A + B) C
  have h₄ := cube_mono (show A + B + C ≤ 2 * S + io σ by omega)
  have e₁ : 4 * (27 * (v * v)) ≤ 27 * ((A + B) * (A + B) * C) := by
    have := Nat.mul_le_mul_left 27 (Nat.mul_le_mul_right C h₂)
    have := Nat.mul_le_mul_left 108 hLW
    have e : 27 * (4 * (A * B) * C) = 108 * (A * B * C) := by grind
    have e' : 4 * (27 * (v * v)) = 108 * (v * v) := by grind
    omega
  have : 4 * (27 * (v * v)) ≤ 4 * ((2 * S + io σ) * (2 * S + io σ) * (2 * S + io σ)) := by
    have := Nat.mul_le_mul_left 4 h₄
    omega
  exact Nat.le_of_mul_le_mul_left this (by decide)

/-! ## The whole trace -/

/-- The most distinct products a stretch of at most `T` I/Os can form, at capacity `S`. -/
def U (S T : Nat) : Nat := Nat.sqrt ((2 * S + T) * (2 * S + T) * (2 * S + T) / 27)

theorem segment_le_U (M N K S T : Nat) {s s' : St} {σ : List Op} (h : exec S s σ = some s')
    (hi : s.dirty ⊆ s.fast) (hc : s.fast.length ≤ S) (hT : io σ ≤ T) :
    card M N K (prods σ) ≤ U S T :=
  le_sqrt_of (Nat.le_trans (segment_bound M N K S h hi hc) (cube_mono (by omega)))

theorem io_cons (op : Op) (σ : List Op) : io (op :: σ) = io [op] + io σ := by
  simp only [io, loads, stores, List.flatMap_cons, List.flatMap_nil, List.append_nil,
    List.length_append]
  omega

theorem io_single_le (op : Op) : io [op] ≤ 1 := by
  cases op <;> simp [io, loads, stores, loadOf, storeOf]

theorem io_append (σ ρ : List Op) : io (σ ++ ρ) = io σ + io ρ := by
  simp only [io, loads, stores, List.flatMap_append, List.length_append]
  omega

/-- A trace with at least `T` I/Os has a prefix with exactly `T`. -/
theorem prefix_io : ∀ (ops : List Op) (T : Nat), T ≤ io ops → ∃ k, io (ops.take k) = T
  | [], T, h => ⟨0, by
      have e₁ : io ([] : List Op) = 0 := rfl
      have e₂ : io (([] : List Op).take 0) = 0 := rfl
      omega⟩
  | op :: ops, T, h => by
    rcases Nat.eq_zero_or_pos T with rfl | hT
    · exact ⟨0, rfl⟩
    · rw [io_cons] at h
      have h₁ := io_single_le op
      by_cases e : io [op] = 1
      · obtain ⟨k, hk⟩ := prefix_io ops (T - 1) (by omega)
        refine ⟨k + 1, ?_⟩
        rw [List.take_succ_cons, io_cons, hk]
        omega
      · obtain ⟨k, hk⟩ := prefix_io ops T (by omega)
        refine ⟨k + 1, ?_⟩
        rw [List.take_succ_cons, io_cons, hk]
        omega

theorem card_append (M N K : Nat) (σ ρ : List Op) :
    card M N K (prods (σ ++ ρ)) ≤ card M N K (prods σ) + card M N K (prods ρ) := by
  unfold card
  rw [← sumR_add]
  apply sumR_le
  intro i _
  rw [← sumR_add]
  apply sumR_le
  intro j _
  rw [← sumR_add]
  apply sumR_le
  intro k _
  simp only [prods, List.any_append]
  exact ind_or _ _

theorem one_segment {M N K S T : Nat} {s s' : St} {ops : List Op} (h : exec S s ops = some s')
    (hi : s.dirty ⊆ s.fast) (hc : s.fast.length ≤ S) (hio : io ops ≤ T) :
    T * card M N K (prods ops) ≤ U S T * (io ops + T) := by
  have h₁ := Nat.mul_le_mul_left T (segment_le_U M N K S T h hi hc hio)
  have h₂ : T * U S T ≤ U S T * (io ops + T) := by
    rw [Nat.mul_comm]; exact Nat.mul_le_mul_left _ (by omega)
  omega

/-- **The I/O bound.** For every `T ≥ 1`: a run at capacity `S` from a state with at most `S`
resident words has `T · (distinct products) ≤ U(S, T) · (I/O + T)`. -/
theorem io_bound (M N K S T : Nat) (hT : 0 < T) (n : Nat) :
    ∀ (ops : List Op), ops.length ≤ n → ∀ {s s' : St}, exec S s ops = some s' →
      s.dirty ⊆ s.fast → s.fast.length ≤ S →
      T * card M N K (prods ops) ≤ U S T * (io ops + T) := by
  induction n with
  | zero =>
    intro ops hn s s' h hi hc
    have : ops = [] := List.eq_nil_of_length_eq_zero (by omega)
    subst this
    exact one_segment h hi hc (by show 0 ≤ T; omega)
  | succ n ih =>
    intro ops hn s s' h hi hc
    by_cases hio : io ops ≤ T
    · exact one_segment h hi hc hio
    · obtain ⟨k, hk⟩ := prefix_io ops T (by omega)
      have hsplit := List.take_append_drop k ops
      have h' := h
      rw [← hsplit] at h'
      obtain ⟨s₁, h₁, h₂⟩ := exec_append h'
      have hσ := segment_le_U M N K S T h₁ hi hc (by omega)
      have hk0 : 0 < k := by
        rcases Nat.eq_zero_or_pos k with rfl | hk0
        · simp [io, loads, stores] at hk; omega
        · exact hk0
      have hne : 0 < ops.length := by
        apply Nat.pos_of_ne_zero
        intro e
        rw [List.eq_nil_of_length_eq_zero e] at hk
        simp [io, loads, stores] at hk
        omega
      have hlen : (ops.drop k).length ≤ n := by
        rw [List.length_drop]; omega
      have hr := ih (ops.drop k) hlen h₂ (exec_inv h₁ hi) (exec_cap h₁ hc)
      have hcard := card_append M N K (ops.take k) (ops.drop k)
      rw [hsplit] at hcard
      have hio' : io ops = io (ops.take k) + io (ops.drop k) := by
        rw [← io_append, hsplit]
      rw [hk] at hio'
      have e₁ := Nat.mul_le_mul_left T hcard
      have e₂ := Nat.mul_le_mul_left T hσ
      rw [hio']
      generalize card M N K (prods ops) = c₀ at e₁
      generalize card M N K (prods (ops.take k)) = c₁ at e₁ e₂
      generalize card M N K (prods (ops.drop k)) = c₂ at e₁ hr
      generalize U S T = u at e₂ hr ⊢
      generalize io (ops.drop k) = d at hr ⊢
      grind

/-- The products of the box, all formed. -/
def Covers (M N K : Nat) (ops : List Op) : Prop :=
  ∀ i < M, ∀ j < N, ∀ k < K, ∃ t, Op.mac i j k t ∈ ops

theorem card_full {M N K : Nat} {ops : List Op} (h : Covers M N K ops) :
    card M N K (prods ops) = M * (N * K) := by
  have h₁ : ∀ i < M, ∀ j < N, ∀ k < K, ind (prods ops i j k) = 1 := by
    intro i hi j hj k hk
    obtain ⟨t, ht⟩ := h i hi j hj k hk
    have : prods ops i j k = true := List.any_eq_true.2 ⟨_, ht, by simp [isMac]⟩
    rw [this]; rfl
  unfold card
  rw [sumR_congr fun i hi => sumR_congr fun j hj => sumR_congr fun k hk => h₁ i hi j hj k hk]
  simp only [sumR_const, Nat.mul_one]

/-- **Classical matrix multiplication, one fast memory.** For every `T ≥ 1`, a trace of the
class that forms every product of `C = A·B` (`A` of `M × K`, `B` of `K × N`) at capacity `S`, from
a state with at most `S` resident words, moves `I/O ≥ T · M N K / U(S, T) - T` words. -/
theorem mmm_io {M N K S T : Nat} (hT : 0 < T) {s s' : St} {ops : List Op}
    (h : exec S s ops = some s') (hi : s.dirty ⊆ s.fast) (hc : s.fast.length ≤ S)
    (hcov : Covers M N K ops) : T * (M * (N * K)) ≤ U S T * (io ops + T) := by
  have := io_bound M N K S T hT ops.length ops (Nat.le_refl _) h hi hc
  rwa [card_full hcov] at this

/-- Projections of what a stretch forms, bounded by what it had and moved: its `A` and `B`
operands by the words resident at its start and those it loaded; its `C` positions by any list
`D` holding its final dirty words, and the words it stored. -/
theorem proj_le (M N K S : Nat) {s s' : St} {σ : List Op} (h : exec S s σ = some s')
    (D : List Word) (hD : s'.dirty ⊆ D) :
    projIK M N K (prods σ) + projKJ M N K (prods σ) ≤ s.fast.length + (loads σ).length ∧
      projIJ M N K (prods σ) ≤ D.length + (stores σ).length := by
  constructor
  · have hA : projIK M N K (prods σ) ≤
        sumR M fun i => sumR K fun k => ind (decide (Word.a i k ∈ s.fast ++ loads σ)) :=
      sum2_le fun i _ k _ => ind_le_of fun hp => by
        obtain ⟨j, _, hj⟩ := anyR_iff.1 hp
        obtain ⟨t, ht⟩ := prods_mac hj
        exact decide_eq_true (exec_operands h ht).1
    have hB : projKJ M N K (prods σ) ≤
        sumR K fun k => sumR N fun j => ind (decide (Word.b k j ∈ s.fast ++ loads σ)) :=
      sum2_le fun k _ j _ => ind_le_of fun hp => by
        obtain ⟨i, _, hj⟩ := anyR_iff.1 hp
        obtain ⟨t, ht⟩ := prods_mac hj
        exact decide_eq_true (exec_operands h ht).2
    have := mass_le M N K (s.fast ++ loads σ)
    unfold mass at this
    simp only [List.length_append] at this
    omega
  · have hC : projIJ M N K (prods σ) ≤
        sumR M fun i => sumR N fun j => ind ((D ++ stores σ).any (isC i j)) :=
      sum2_le fun i _ j _ => ind_le_of fun hp => by
        obtain ⟨k, _, hk⟩ := anyR_iff.1 hp
        obtain ⟨t, ht⟩ := prods_mac hk
        obtain ⟨w, hw, hcw⟩ := List.any_eq_true.1 (exec_products h ht)
        refine List.any_eq_true.2 ⟨w, ?_, hcw⟩
        rcases List.mem_append.1 hw with hw | hw
        · exact List.mem_append_left _ (hD hw)
        · exact List.mem_append_right _ hw
    have := mass_le M N K (D ++ stores σ)
    unfold mass at this
    simp only [List.length_append] at this
    omega

/-- **Compulsory traffic in the class.** A trace that forms every product from an empty fast
memory and ends with no dirty accumulator loads every entry of `A` and `B` and stores a value of
every `c_ij`: `I/O ≥ MK + KN + MN`. -/
theorem mmm_compulsory {M N K S : Nat} {s s' : St} {ops : List Op}
    (h : exec S s ops = some s') (h₀ : s.fast = []) (hend : s'.dirty = [])
    (hcov : Covers M N K ops) (hM : 0 < M) (hN : 0 < N) (hK : 0 < K) :
    M * K + K * N + M * N ≤ io ops := by
  obtain ⟨hAB, hC⟩ := proj_le M N K S h [] (by rw [hend]; exact List.Subset.refl _)
  have one : ∀ {n : Nat} {p : Nat → Bool}, (∃ x < n, p x = true) → ind (anyR n p) = 1 :=
    fun hx => by rw [anyR_iff.2 hx]; rfl
  have mac_at : ∀ i < M, ∀ j < N, ∀ k < K, prods ops i j k = true := by
    intro i hi j hj k hk
    obtain ⟨t, ht⟩ := hcov i hi j hj k hk
    exact List.any_eq_true.2 ⟨_, ht, by simp [isMac]⟩
  have pIK : projIK M N K (prods ops) = M * K := by
    unfold projIK
    rw [sumR_congr fun i hi => sumR_congr fun k hk => one ⟨0, hN, mac_at i hi 0 hN k hk⟩]
    simp only [sumR_const, Nat.mul_one]
  have pKJ : projKJ M N K (prods ops) = K * N := by
    unfold projKJ
    rw [sumR_congr fun k hk => sumR_congr fun j hj => one ⟨0, hM, mac_at 0 hM j hj k hk⟩]
    simp only [sumR_const, Nat.mul_one]
  have pIJ : projIJ M N K (prods ops) = M * N := by
    unfold projIJ
    rw [sumR_congr fun i hi => sumR_congr fun j hj => one ⟨0, hK, mac_at i hi j hj 0 hK⟩]
    simp only [sumR_const, Nat.mul_one]
  rw [pIK, pKJ, h₀] at hAB
  rw [pIJ] at hC
  simp only [List.length_nil] at hAB hC
  unfold io
  omega

/-- A trace run by one processor: from `start` to `stop`. -/
structure Proc where
  start : St
  ops : List Op
  stop : St

/-- **Classical matrix multiplication on processors with their own fast memories** (an SM's
shared memory and registers each): if processors, each at capacity `S` and starting with at most
`S` resident words, together form every product, then for every `T ≥ 1` their I/O totals at least
`T · M N K / U(S, T) - P · T`. -/
theorem mmm_io_parallel {M N K S T : Nat} (hT : 0 < T) (ps : List Proc)
    (hrun : ∀ p ∈ ps, exec S p.start p.ops = some p.stop)
    (hi : ∀ p ∈ ps, p.start.dirty ⊆ p.start.fast) (hc : ∀ p ∈ ps, p.start.fast.length ≤ S)
    (hcov : ∀ i < M, ∀ j < N, ∀ k < K, ∃ p ∈ ps, ∃ t, Op.mac i j k t ∈ p.ops) :
    T * (M * (N * K)) ≤ U S T * ((ps.map fun p => io p.ops).sum + ps.length * T) := by
  -- Every product is formed by some processor, so the union's count bounds `M N K`.
  have key : ∀ qs : List Proc, (∀ p ∈ qs, p ∈ ps) →
      T * card M N K (fun i j k => qs.any fun p => prods p.ops i j k) ≤
        U S T * ((qs.map fun p => io p.ops).sum + qs.length * T) := by
    intro qs
    induction qs with
    | nil =>
      intro _
      have : card M N K (fun i j k => ([] : List Proc).any fun p => prods p.ops i j k) = 0 := by
        unfold card; simp [ind, sumR_zero]
      rw [this]; exact Nat.zero_le _
    | cons q qs ih =>
      intro hq
      have hq₀ := hq q List.mem_cons_self
      have hb := io_bound M N K S T hT q.ops.length q.ops (Nat.le_refl _) (hrun q hq₀) (hi q hq₀)
        (hc q hq₀)
      have hr := ih fun p hp => hq p (List.mem_cons_of_mem _ hp)
      have hu : card M N K (fun i j k => (q :: qs).any fun p => prods p.ops i j k) ≤
          card M N K (prods q.ops) + card M N K (fun i j k => qs.any fun p => prods p.ops i j k) := by
        unfold card
        rw [← sumR_add]
        apply sumR_le; intro i _
        rw [← sumR_add]
        apply sumR_le; intro j _
        rw [← sumR_add]
        apply sumR_le; intro k _
        simp only [List.any_cons]
        exact ind_or _ _
      have e := Nat.mul_le_mul_left T hu
      simp only [List.map_cons, List.sum_cons, List.length_cons]
      generalize card M N K (fun i j k => (q :: qs).any fun p => prods p.ops i j k) = c₀ at e
      generalize card M N K (prods q.ops) = c₁ at e hb
      generalize card M N K (fun i j k => qs.any fun p => prods p.ops i j k) = c₂ at e hr
      generalize U S T = u at hb hr
      grind
  have hfull : card M N K (fun i j k => ps.any fun p => prods p.ops i j k) = M * (N * K) := by
    have h₁ : ∀ i < M, ∀ j < N, ∀ k < K, ind (ps.any fun p => prods p.ops i j k) = 1 := by
      intro i hi j hj k hk
      obtain ⟨p, hp, t, ht⟩ := hcov i hi j hj k hk
      have : (ps.any fun p => prods p.ops i j k) = true :=
        List.any_eq_true.2 ⟨p, hp, List.any_eq_true.2 ⟨_, ht, by simp [isMac]⟩⟩
      rw [this]; rfl
    unfold card
    rw [sumR_congr fun i hi => sumR_congr fun j hj => sumR_congr fun k hk => h₁ i hi j hj k hk]
    simp only [sumR_const, Nat.mul_one]
  have := key ps fun _ h => h
  rwa [hfull] at this

end NV.Bounds.Pebble
