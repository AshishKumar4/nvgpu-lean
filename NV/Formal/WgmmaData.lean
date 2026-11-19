import NV.Async.Wgmma

/-!
# What a warpgroup MMA reads

A share's loads return its operands' 16-byte chunks; `image` files each chunk's four words under
its address and `elem` reads an element back. `image_get`: the image of distinct chunks holds,
under the `i`-th chunk, the `i`-th four words of the data; `elem_image`: so the element at an
address is cut from the words of the chunk that contains it.
-/

namespace NV.Formal.WgmmaData

open NV.Async.Wgmma

/-- The image after the first `n` chunks. -/
def imageUpTo (cs : Array Nat) (data : Resp) (n : Nat) : Std.HashMap Nat (Array UInt32) :=
  (List.range n).foldl (fun m i => m.insert cs[i]! (data.extract (4 * i) (4 * i + 4))) {}

theorem image_eq (cs : Array Nat) (data : Resp) : image cs data = imageUpTo cs data cs.size := by
  simp [image, imageUpTo, Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl,
    List.range_eq_range']
  rfl

theorem imageUpTo_get (cs : Array Nat) (data : Resp) :
    ∀ n, ∀ i < n, (∀ j < n, j ≠ i → cs[j]! ≠ cs[i]!) →
      (imageUpTo cs data n)[cs[i]!]? = some (data.extract (4 * i) (4 * i + 4)) := by
  intro n
  induction n with
  | zero => intro i hi; omega
  | succ n ih =>
    intro i hi hd
    simp only [imageUpTo, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil] at ih ⊢
    rw [Std.HashMap.getElem?_insert]
    by_cases hin : i = n
    · subst hin; simp
    · have hne : cs[n]! ≠ cs[i]! := hd n (by omega) (Ne.symm hin)
      simp only [beq_iff_eq, hne, ite_false]
      exact ih i (by omega) fun j hj hji => hd j (by omega) hji

/-- **The image of distinct chunks** holds the `i`-th chunk's four words under its address. -/
theorem image_get {cs : Array Nat} (hnd : cs.toList.Nodup) (data : Resp) {i : Nat} (hi : i < cs.size) :
    (image cs data)[cs[i]]? = some (data.extract (4 * i) (4 * i + 4)) := by
  rw [image_eq, ← getElem!_pos cs i hi]
  refine imageUpTo_get cs data _ i hi fun j hj hji e => hji ?_
  rw [getElem!_pos cs j hj, getElem!_pos cs i hi] at e
  have e' : cs.toList[j]'(by simpa using hj) = cs.toList[i]'(by simpa using hi) := by simpa using e
  exact hnd.eq_of_getElem_eq _ _ e'

/-- **An element read from the image** is cut from its chunk's words. -/
theorem elem_image {img : Std.HashMap Nat (Array UInt32)} {a : Nat} {ws : Array UInt32}
    (h : img[a / 16 * 16]? = some ws) (bits : Nat) :
    elem img bits a = (if bits == 32 then (ws[a % 16 / 4]?.getD 0).toNat
      else ((ws[a % 16 / 4]?.getD 0).toNat >>> (8 * (a % 16 % 4))) % 2 ^ bits) := by
  simp [elem, Std.HashMap.getD_eq_getD_getElem?, h]

/-- One step of `chunks`. -/
def chunkStep (st : Std.HashSet Nat × Array Nat) (a : Nat) : Std.HashSet Nat × Array Nat :=
  if st.1.contains (a / 16 * 16) then st else (st.1.insert (a / 16 * 16), st.2.push (a / 16 * 16))

theorem chunks_eq (addrs : Array Nat) : chunks addrs = (addrs.toList.foldl chunkStep ({}, #[])).2 := by
  simp only [chunks, Id.run]
  rw [← Array.forIn_toList]
  generalize ({} : Std.HashSet Nat) = s₀
  generalize (#[] : Array Nat) = o₀
  induction addrs.toList generalizing s₀ o₀ with
  | nil => rfl
  | cons a l ih =>
    simp only [List.forIn_cons, List.foldl_cons, chunkStep]
    split <;> simp_all

/-- What `chunks` keeps: the seen set is the output's members, and the output has no repeats. -/
def ChunkInv (st : Std.HashSet Nat × Array Nat) : Prop :=
  (∀ c, st.1.contains c = true ↔ c ∈ st.2) ∧ st.2.toList.Nodup

theorem chunkStep_inv {st : Std.HashSet Nat × Array Nat} (h : ChunkInv st) (a : Nat) :
    ChunkInv (chunkStep st a) := by
  unfold chunkStep
  split
  · exact h
  · rename_i hn
    refine ⟨fun c => ?_, ?_⟩
    · simp only [Std.HashSet.contains_insert, Bool.or_eq_true, beq_iff_eq, Array.mem_push, h.1 c]
      constructor <;> rintro (h | h) <;> first | exact .inl h | exact .inr h.symm | exact .inl h.symm | exact .inr h
    · simp only [Array.toList_push]
      refine List.nodup_append.2 ⟨h.2, by simp, fun x hx y hy e => ?_⟩
      simp only [List.mem_singleton] at hy
      subst hy e
      exact hn ((h.1 _).2 (by simpa using hx))

theorem foldl_inv : ∀ (l : List Nat) (st : Std.HashSet Nat × Array Nat), ChunkInv st →
    ChunkInv (l.foldl chunkStep st) ∧ ∀ a ∈ l, a / 16 * 16 ∈ (l.foldl chunkStep st).2
  | [], st, h => ⟨h, by simp⟩
  | a :: l, st, h => by
    obtain ⟨h₁, h₂⟩ := foldl_inv l (chunkStep st a) (chunkStep_inv h a)
    refine ⟨h₁, fun b hb => ?_⟩
    rcases List.mem_cons.1 hb with rfl | hb
    · -- `b`'s chunk is in after its step and stays.
      have hin : b / 16 * 16 ∈ (chunkStep st b).2 := by
        unfold chunkStep; split
        · rename_i hc; exact (h.1 _).1 hc
        · simp
      exact mono l _ hin
    · exact h₂ b hb
where
  mono : ∀ (l : List Nat) (st : Std.HashSet Nat × Array Nat) {c : Nat}, c ∈ st.2 →
      c ∈ (l.foldl chunkStep st).2
    | [], _, _, h => h
    | a :: l, st, c, h => mono l _ (by unfold chunkStep; split <;> simp [h])

/-- **`chunks` lists distinct chunks**, among them every address's. -/
theorem chunks_spec (addrs : Array Nat) :
    (chunks addrs).toList.Nodup ∧ ∀ a ∈ addrs, a / 16 * 16 ∈ chunks addrs := by
  rw [chunks_eq]
  obtain ⟨h₁, h₂⟩ := foldl_inv addrs.toList ({}, #[]) ⟨by simp, by simp⟩
  exact ⟨h₁.2, fun a ha => h₂ a (by simpa using ha)⟩

/-- The little-endian word of memory `M` at `x`. -/
def word (M : Nat → UInt8) (x : Nat) : UInt32 :=
  ((M x).toNat + 2 ^ 8 * (M (x + 1)).toNat + 2 ^ 16 * (M (x + 2)).toNat +
    2 ^ 24 * (M (x + 3)).toNat).toUInt32

/-- The four words of the 16-byte chunk at `c`. -/
def chunkWords (M : Nat → UInt8) (c : Nat) : Array UInt32 :=
  #[word M c, word M (c + 4), word M (c + 8), word M (c + 12)]

/-- **A 16-bit element read from loads of memory `M`** is `M`'s little-endian half-word, when
the loads returned each chunk's words. -/
theorem elem16 {M : Nat → UInt8} {cs : Array Nat} (hnd : cs.toList.Nodup) {data : Resp}
    (hdata : ∀ i (hi : i < cs.size), data.extract (4 * i) (4 * i + 4) = chunkWords M cs[i])
    {a : Nat} (ha : a / 16 * 16 ∈ cs) (h2 : a % 2 = 0) :
    elem (image cs data) 16 a = (M a).toNat + 2 ^ 8 * (M (a + 1)).toNat := by
  obtain ⟨i, hi, hci⟩ := Array.mem_iff_getElem.1 ha
  have himg := image_get hnd data hi
  rw [hdata i hi, hci] at himg
  rw [elem_image himg]
  have hb : ∀ x, (M x).toNat < 256 := fun x => (M x).toNat_lt
  have hw : ∀ x, (word M x).toNat = (M x).toNat + 2 ^ 8 * (M (x + 1)).toNat +
      2 ^ 16 * (M (x + 2)).toNat + 2 ^ 24 * (M (x + 3)).toNat := fun x => by
    simp only [word, Nat.toUInt32, UInt32.toNat_ofNat']
    have := hb x; have := hb (x + 1); have := hb (x + 2); have := hb (x + 3)
    omega
  have hc : a / 16 * 16 + a % 16 = a := by omega
  have hq : a % 16 / 4 < 4 := by omega
  have hsel : (chunkWords M (a / 16 * 16))[a % 16 / 4]?.getD 0 = word M (a / 16 * 16 + 4 * (a % 16 / 4)) := by
    rcases (by omega : a % 16 / 4 = 0 ∨ a % 16 / 4 = 1 ∨ a % 16 / 4 = 2 ∨ a % 16 / 4 = 3) with
      h | h | h | h <;> simp [chunkWords, h]
  simp only [show (16 == 32) = false from rfl, Bool.false_eq_true, ite_false, hsel, hw,
    Nat.shiftRight_eq_div_pow]
  have := hb a; have := hb (a + 1)
  rcases (by omega : a % 16 % 4 = 0 ∨ a % 16 % 4 = 2) with h | h
  · have e : a / 16 * 16 + 4 * (a % 16 / 4) = a := by omega
    rw [e, h]
    have := hb (a + 2); have := hb (a + 3)
    simp only [show (2 : Nat) ^ (8 * 0) = 1 from rfl, show (2 : Nat) ^ 16 = 65536 from rfl,
      show (2 : Nat) ^ 8 = 256 from rfl, show (2 : Nat) ^ 24 = 16777216 from rfl]
    omega
  · have e : a / 16 * 16 + 4 * (a % 16 / 4) = a - 2 := by omega
    rw [e, h, show a - 2 + 3 = a + 1 by omega]
    have := hb (a - 2); have := hb (a - 2 + 1)
    have e2 : M (a - 2 + 2) = M a := congrArg M (by omega)
    simp only [e2, show (2 : Nat) ^ (8 * 2) = 65536 from rfl,
      show (2 : Nat) ^ 8 = 256 from rfl, show (2 : Nat) ^ 24 = 16777216 from rfl]
    omega

end NV.Formal.WgmmaData
