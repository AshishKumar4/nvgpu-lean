import NV.Memory.Store

/-!
# Reading what was written

A byte store returns the last byte written at an offset, and is unchanged elsewhere, as long as
its pages are whole and its page table holds a place for every page (`Store.WF`, kept by every
write).
-/

namespace NV.Store

/-- Every page has its place in the page table: a chunk long enough to hold it. -/
def Shaped (s : Store) : Prop :=
  ∀ i, i < s.npages → ∃ ch, s.chunks[i >>> chunkBits]? = some ch ∧ i % chunkPages < ch.size

/-- Every allocated page is a whole page, and every page has its place. -/
abbrev WF (s : Store) : Prop :=
  (∀ (i : Nat) (p : ByteArray), s.page i = some p → p.size = pageSize) ∧ s.Shaped

/-- A fresh page: the shared zero page. -/
abbrev zeros : ByteArray := zeroPage

theorem shift_eq (off : Nat) : off >>> pageBits = off / pageSize := by
  simp [pageBits, pageSize, Nat.shiftRight_eq_div_pow]

theorem chunk_shift_eq (i : Nat) : i >>> chunkBits = i / chunkPages := by
  simp [chunkBits, chunkPages, Nat.shiftRight_eq_div_pow]

theorem pageSize_pos : 0 < pageSize := by unfold pageSize pageBits; decide

theorem chunkPages_pos : 0 < chunkPages := by unfold chunkPages chunkBits; decide

/-- A page index is its chunk and its place in the chunk. -/
theorem eq_iff_chunk (i j : Nat) : j = i ↔ (j >>> chunkBits = i >>> chunkBits ∧ j % chunkPages = i % chunkPages) := by
  rw [chunk_shift_eq, chunk_shift_eq]
  constructor
  · rintro rfl; exact ⟨rfl, rfl⟩
  · rintro ⟨h₁, h₂⟩
    rw [← Nat.div_add_mod j chunkPages, ← Nat.div_add_mod i chunkPages, h₁, h₂]

theorem zeros_size : zeros.size = pageSize := by
  show (Array.replicate pageSize (0 : UInt8)).size = pageSize
  exact Array.size_replicate

theorem zeros_get! (i : Nat) : zeros.get! i = 0 := by
  show (Array.replicate pageSize (0 : UInt8))[i]! = 0
  rw [getElem!_def, Array.getElem?_replicate]
  by_cases h : i < pageSize <;> simp [h] <;> rfl

theorem get!_set! (q : ByteArray) (i j : Nat) (b : UInt8) (hi : i < q.size) :
    (q.set! i b).get! j = if j = i then b else q.get! j := by
  obtain ⟨a⟩ := q
  have hi' : i < a.size := hi
  show (a.set! i b)[j]! = if j = i then b else a[j]!
  by_cases h : j = i
  · subst h; rw [if_pos rfl]; exact Array.getElem!_set!_self a j b hi'
  · rw [if_neg h]; exact Array.getElem!_set!_ne a i j b (Ne.symm h)

theorem size_set! (q : ByteArray) (i : Nat) (b : UInt8) : (q.set! i b).size = q.size := by
  obtain ⟨a⟩ := q
  show (a.set! i b).size = a.size
  simp

/-! ## The page table -/

theorem npages_setPage (s : Store) (i : Nat) (p : Option ByteArray) : (s.setPage i p).npages = s.npages := by
  unfold setPage; split
  · split <;> rfl
  · rfl

theorem base_setPage (s : Store) (i : Nat) (p : Option ByteArray) : (s.setPage i p).base = s.base := by
  unfold setPage; split
  · split <;> rfl
  · rfl

theorem size_setPage (s : Store) (i : Nat) (p : Option ByteArray) : (s.setPage i p).size = s.size := by
  unfold setPage; split
  · split <;> rfl
  · rfl

/-- The chunks after page `i` of chunk `ch` is set. -/
theorem chunks_setPage {s : Store} {i : Nat} {ch : Array (Option ByteArray)} (p : Option ByteArray)
    (hi : i < s.npages) (hch : s.chunks[i >>> chunkBits]? = some ch) :
    (s.setPage i p).chunks = (s.chunks.set! (i >>> chunkBits) #[]).set! (i >>> chunkBits) (ch.set! (i % chunkPages) p) := by
  simp [setPage, hi, hch]

theorem chunk_setPage {s : Store} {i : Nat} {ch : Array (Option ByteArray)} (p : Option ByteArray)
    (hi : i < s.npages) (hch : s.chunks[i >>> chunkBits]? = some ch) (c : Nat) :
    (s.setPage i p).chunks[c]? =
      if c = i >>> chunkBits then some (ch.set! (i % chunkPages) p) else s.chunks[c]? := by
  have hc : i >>> chunkBits < s.chunks.size := by
    rcases Nat.lt_or_ge (i >>> chunkBits) s.chunks.size with h | h
    · exact h
    · rw [Array.getElem?_eq_none h] at hch; cases hch
  rw [chunks_setPage p hi hch]
  simp only [Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds, Array.size_setIfInBounds]
  by_cases h : i >>> chunkBits = c
  · subst h; simp [hc]
  · have h' : ¬ c = i >>> chunkBits := fun e => h e.symm
    simp [h, h']

theorem shaped_setPage {s : Store} (hs : s.Shaped) (i : Nat) (p : Option ByteArray) : (s.setPage i p).Shaped := by
  intro j hj
  rw [npages_setPage] at hj
  obtain ⟨cj, hcj, hlt⟩ := hs j hj
  by_cases hi : i < s.npages
  · obtain ⟨ch, hch, _⟩ := hs i hi
    rw [chunk_setPage p hi hch]
    split
    · rename_i he
      refine ⟨_, rfl, ?_⟩
      rw [he] at hcj; rw [hch] at hcj; cases hcj
      simp [Array.set!_eq_setIfInBounds, hlt]
    · exact ⟨cj, hcj, hlt⟩
  · have : s.setPage i p = s := by simp [setPage, hi]
    rw [this]; exact ⟨cj, hcj, hlt⟩

/-- Page `j` after page `i` is set. -/
theorem page_setPage {s : Store} (hs : s.Shaped) {i : Nat} (hi : i < s.npages) (p : Option ByteArray) (j : Nat) :
    (s.setPage i p).page j = if j = i then p else s.page j := by
  obtain ⟨ch, hch, hlt⟩ := hs i hi
  unfold page
  rw [npages_setPage]
  by_cases hj : j < s.npages
  · rw [if_pos hj, if_pos hj, chunk_setPage p hi hch]
    by_cases he : j = i
    · subst he; simp [Array.set!_eq_setIfInBounds, hlt]
    · rw [if_neg he]
      split
      · rename_i hc
        have ho : ¬ j % chunkPages = i % chunkPages := fun e => he ((eq_iff_chunk i j).2 ⟨hc, e⟩)
        rw [hc, hch]
        simp only [Option.bind_some, Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds]
        simp [Ne.symm ho, ho]
      · rfl
  · rw [if_neg hj, if_neg hj]
    have : ¬ j = i := fun e => hj (e ▸ hi)
    simp [this]

theorem emptyChunks_get? (n c : Nat) : (emptyChunks n)[c]? =
    if c < (n + chunkPages - 1) / chunkPages then some (Array.replicate (min chunkPages (n - c * chunkPages)) none)
    else none := by
  unfold emptyChunks
  rw [Array.getElem?_map, Array.getElem?_range]
  split <;> rfl

theorem page_empty (base : UInt64) (size i : Nat) : (empty base size).page i = none := by
  unfold page empty
  dsimp only
  split
  · rw [emptyChunks_get?]
    split
    · simp only [Option.bind_some, Array.getElem?_replicate]
      split <;> rfl
    · rfl
  · rfl

theorem shaped_empty (base : UInt64) (size : Nat) : (empty base size).Shaped := by
  intro i hi
  simp only [empty] at hi ⊢
  have hc : i >>> chunkBits < ((size + pageSize - 1) / pageSize + chunkPages - 1) / chunkPages := by
    rw [chunk_shift_eq, Nat.div_lt_iff_lt_mul chunkPages_pos]
    have h₁ := Nat.div_add_mod' ((size + pageSize - 1) / pageSize + chunkPages - 1) chunkPages
    have h₂ := Nat.mod_lt ((size + pageSize - 1) / pageSize + chunkPages - 1) chunkPages_pos
    have := chunkPages_pos
    omega
  refine ⟨_, by rw [emptyChunks_get?, if_pos hc], ?_⟩
  rw [Array.size_replicate, chunk_shift_eq]
  have h₁ := Nat.div_add_mod' i chunkPages
  have h₂ := Nat.mod_lt i chunkPages_pos
  have : i / chunkPages * chunkPages ≤ i := Nat.div_mul_le_self i chunkPages
  omega

/-! ## Writing a byte -/

/-- The page a write leaves at `j`. -/
theorem page_writeByte {s : Store} (hs : s.Shaped) (off j : Nat) (b : UInt8) (h : off >>> pageBits < s.npages) :
    (s.writeByte off b).page j =
      if j = off >>> pageBits then
        some (((s.page (off >>> pageBits)).getD zeros).set! (off % pageSize) b)
      else s.page j := by
  unfold writeByte
  rw [if_pos h]
  rw [page_setPage (shaped_setPage hs _ _) (by rw [npages_setPage]; exact h)]
  split
  · rfl
  · rw [page_setPage hs h]; rename_i hne; rw [if_neg hne]

theorem npages_writeByte (s : Store) (off : Nat) (b : UInt8) :
    (s.writeByte off b).npages = s.npages := by
  by_cases h : off >>> pageBits < s.npages <;> simp [writeByte, h, npages_setPage]

theorem shaped_writeByte {s : Store} (hs : s.Shaped) (off : Nat) (b : UInt8) : (s.writeByte off b).Shaped := by
  by_cases h : off >>> pageBits < s.npages
  · simp only [writeByte, h, ite_true]; exact shaped_setPage (shaped_setPage hs _ _) _ _
  · simp only [writeByte, h, ite_false]; exact hs

theorem WF.writeByte {s : Store} (hs : s.WF) (off : Nat) (b : UInt8) : (s.writeByte off b).WF := by
  refine ⟨fun i p hp => ?_, shaped_writeByte hs.2 off b⟩
  by_cases h : off >>> pageBits < s.npages
  · rw [page_writeByte hs.2 off i b h] at hp
    split at hp
    · simp only [Option.some.injEq] at hp
      subst hp
      rw [size_set!]
      cases hq : s.page (off >>> pageBits) with
      | none => exact zeros_size
      | some q => exact hs.1 _ q hq
    · exact hs.1 i p hp
  · have : s.writeByte off b = s := by unfold Store.writeByte; simp [h]
    rw [this] at hp; exact hs.1 i p hp

theorem readByte_writeByte {s : Store} (hs : s.WF) (off off' : Nat) (b : UInt8)
    (h : off >>> pageBits < s.npages) :
    (s.writeByte off b).readByte off' = if off' = off then b else s.readByte off' := by
  have hdiv : ∀ x, x / pageSize * pageSize + x % pageSize = x := fun x => Nat.div_add_mod' x _
  have hmod : off % pageSize < pageSize := Nat.mod_lt _ pageSize_pos
  unfold readByte
  rw [page_writeByte hs.2 off _ b h]
  by_cases hp : off' >>> pageBits = off >>> pageBits
  · rw [if_pos hp]
    have hp' : off' / pageSize = off / pageSize := by rw [← shift_eq, ← shift_eq]; exact hp
    have hsz : ((s.page (off >>> pageBits)).getD zeros).size = pageSize := by
      cases hq : s.page (off >>> pageBits) with
      | none => exact zeros_size
      | some q => exact hs.1 _ q hq
    show (((s.page (off >>> pageBits)).getD zeros).set! (off % pageSize) b).get! (off' % pageSize) = _
    rw [get!_set! _ _ _ _ (by rw [hsz]; exact hmod)]
    by_cases he : off' = off
    · subst he; simp
    · have hm : ¬ off' % pageSize = off % pageSize := fun e => he (by
        rw [← hdiv off, ← hdiv off', hp', e])
      rw [if_neg hm, if_neg he, hp]
      cases hq : s.page (off >>> pageBits) with
      | none => exact zeros_get! _
      | some q => rfl
  · have he : ¬ off' = off := fun e => hp (by rw [e])
    rw [if_neg hp, if_neg he]

/-- The pages cover the store. -/
def Fits (s : Store) : Prop := s.size ≤ s.npages * pageSize

theorem base_writeByte (s : Store) (off : Nat) (b : UInt8) : (s.writeByte off b).base = s.base := by
  by_cases h : off >>> pageBits < s.npages <;> simp [writeByte, h, base_setPage]

theorem sizeField_writeByte (s : Store) (off : Nat) (b : UInt8) : (s.writeByte off b).size = s.size := by
  by_cases h : off >>> pageBits < s.npages <;> simp [writeByte, h, size_setPage]

theorem Fits.writeByte {s : Store} (h : s.Fits) (off : Nat) (b : UInt8) : (s.writeByte off b).Fits := by
  unfold Fits at *
  rw [npages_writeByte, sizeField_writeByte]; exact h

theorem page_lt {s : Store} (h : s.Fits) {off : Nat} (ho : off < s.size) :
    off >>> pageBits < s.npages := by
  rw [shift_eq]
  unfold Fits at h
  exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; omega)

/-- In-bounds byte range of an access: `contains` in naturals. -/
theorem contains_nat {s : Store} {a k : Nat} (ha : a < 2 ^ 64)
    (h : s.contains (UInt64.ofNat a) k = true) :
    s.base.toNat ≤ a ∧ a - s.base.toNat + k ≤ s.size := by
  simp only [contains, Bool.and_eq_true, decide_eq_true_eq] at h
  have hof : (UInt64.ofNat a).toNat = a := UInt64.toNat_ofNat_of_lt' ha
  have h₁ : s.base.toNat ≤ a := by
    have := UInt64.le_iff_toNat_le.1 h.1; rwa [hof] at this
  refine ⟨h₁, ?_⟩
  have := h.2
  rw [UInt64.toNat_sub_of_le _ _ h.1, hof] at this
  exact this

end NV.Store
