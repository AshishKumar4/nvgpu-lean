import NV.Async.Wgmma

/-!
# Where a 128-byte-swizzled tile's elements sit

TMA's tiled mode stores a box row by row, 16-byte chunks, and then swizzles each chunk's shared
address (`Swizzle.apply`); a warpgroup MMA's K-major operand descriptor names the same swizzle on
its element addresses (`Wgmma.Desc.addr`). With 128-byte rows and the 128-byte swizzle, both put
element `(r, c)` of a tile of 2-byte elements at `apply (base + 128 r + 2 c)`:

* `apply_chunk`: the swizzle moves whole 16-byte chunks, so a byte keeps its place in its chunk;
* `addr_kMajor_b128`: the descriptor's address of element `(mn, k)` is
  `apply (start + 128 mn + 2 k)` when its stride byte offset is 8 rows (1024 bytes);
* `tile_box2`: a box of 128-byte rows inside its tensor is its 16-byte chunks, chunk `q` of row
  `r` going to `apply (smem + 128 r + 16 q)`.
-/

namespace NV.Formal.TileLayout

open NV.Async NV.Async.Wgmma

theorem xor_low {a b : Nat} (hb : b < 16) (s : Nat) :
    (16 * a + b) ^^^ (s <<< 4) = 16 * (a ^^^ s) + b := by
  apply Nat.eq_of_testBit_eq
  intro i
  have h16 : (16 : Nat) = 2 ^ 4 := rfl
  rw [Nat.testBit_xor, h16, Nat.testBit_two_pow_mul_add _ hb, Nat.testBit_two_pow_mul_add _ hb,
    Nat.testBit_shiftLeft]
  by_cases h : i < 4
  · simp [h, show ¬ 4 ≤ i by omega]
  · simp [h, show 4 ≤ i by omega, Nat.testBit_xor]

/-- **The 128-byte swizzle moves whole 16-byte chunks.** -/
theorem apply_chunk (a : Nat) {b : Nat} (hb : b < 16) :
    Swizzle.b128.apply (16 * a + b) = Swizzle.b128.apply (16 * a) + b := by
  have hl : (16 * a + b) >>> 7 = (16 * a) >>> 7 := by
    simp only [Nat.shiftRight_eq_div_pow]; omega
  simp only [Swizzle.apply, hl]
  rw [xor_low hb, show 16 * a = 16 * a + 0 by omega, xor_low (by omega : 0 < 16)]
  omega

/-- **A K-major operand's element address** under the 128-byte swizzle, 8-row groups 1024 bytes
apart: row `mn` is 128 bytes, element `k` two. -/
theorem addr_kMajor_b128 {d : Desc} (hs : d.swizzle = .b128) (hb : d.baseOffset = 0)
    (hsbo : d.sbo = 1024) (mn k : Nat) :
    d.addr .bf16 true mn k = .ok (Swizzle.b128.apply (d.start + 128 * mn + 2 * k)) := by
  simp only [Desc.addr, hb, hs, hsbo, bne_self_eq_false, Bool.false_eq_true, ite_false]
  simp [Swizzle.span, InType.bits]
  congr 2
  omega


theorem foldlM_ok {α β ε : Type} (f : β → α → Except ε β) (g : β → α → β) :
    ∀ (l : List α) (b : β), (∀ a ∈ l, ∀ b, f b a = .ok (g b a)) → l.foldlM f b = .ok (l.foldl g b)
  | [], _, _ => rfl
  | a :: l, b, h => by
    rw [List.foldlM_cons, h a (List.mem_cons_self ..) b]
    exact foldlM_ok f g l (g b a) fun a' ha' b' => h a' (List.mem_cons_of_mem _ ha') b'

theorem forIn_ok {α β ε : Type} (body : α → β → Except ε (ForInStep β)) (g : β → α → β) :
    ∀ (l : List α) (b : β), (∀ a ∈ l, ∀ s, body a s = .ok (.yield (g s a))) →
      forIn l b body = .ok (l.foldl g b)
  | [], _, _ => rfl
  | a :: l, b, h => by
    rw [List.forIn_cons, h a (List.mem_cons_self ..) b]
    exact forIn_ok body g l (g b a) fun a' ha' s' => h a' (List.mem_cons_of_mem _ ha') s'

theorem foldl_push {α β : Type} (f : α → β) : ∀ (l : List α) (b : Array β),
    l.foldl (fun s a => s.push (f a)) b = b ++ (l.map f).toArray
  | [], b => by simp
  | a :: l, b => by rw [List.foldl_cons, foldl_push f l]; simp

theorem foldl_append {α β : Type} (f : α → List β) : ∀ (l : List α) (b : Array β),
    l.foldl (fun s a => s ++ (f a).toArray) b = b ++ (l.flatMap f).toArray
  | [], b => by simp
  | a :: l, b => by rw [List.foldl_cons, foldl_append f l]; simp

/-- Chunk `q` of row `row` of a box of 64 two-byte elements at `(c0, c1)`, its image at `smem`. -/
def chunk (m : TensorMap) (c0 c1 : Int) (smem row q : Nat) : Piece :=
  { global := some (UInt64.ofNat ((m.globalAddress.toNat : Int) + (c1 + row) * m.globalStrides[0]! +
      (c0 + q * 8) * 2).toNat)
    smem := m.swizzle.apply (smem + row * 128 + q * 16)
    bytes := 16 }

/-- The pieces of a box of `R` rows of 64 two-byte elements inside its tensor: row by row, eight
16-byte chunks each, the chunk's shared address swizzled. -/
def box2 (m : TensorMap) (R : Nat) (c0 c1 : Int) (smem : Nat) : List Piece :=
  (List.range' 0 R).flatMap fun row => (List.range' 0 8).map (chunk m c0 c1 smem row)

/-- **A TMA box of 128-byte rows inside its tensor** is the box's 16-byte chunks, row by row. -/
theorem tile_box2 {m : TensorMap} {d0 d1 st R : Nat} (hv : m.invalid? = none)
    (hil : m.interleave = .none) (hdt : m.dtype.bytes? = some 2) (hr : m.rank = 2)
    (hgd : m.globalDim = #v[d0, d1, 1, 1, 1]) (hbox : m.boxDim = #v[64, R, 1, 1, 1])
    (hes : m.elementStrides = #v[1, 1, 1, 1, 1]) (hgs : m.globalStrides = #v[st, 0, 0, 0])
    (hoob : m.oobFill = .none) {c0 c1 : Int} {smem : Nat} (hs : smem % 16 = 0)
    (h0 : 0 ≤ c0) (h0' : c0 + 64 ≤ d0) (h1 : 0 ≤ c1) (h1' : c1 + R ≤ d1) :
    m.tile #[c0, c1] smem = .ok (box2 m R c0 c1 smem).toArray := by
  simp [TensorMap.tile, hv, hil, hdt, hr, hgd, hbox, hes, hgs, hoob, hs, TensorMap.boxCount,
    TensorMap.rowPitch, Nat.max_eq_left m.swizzle.span_le]
  have hrows : (List.foldl (fun acc i => if i = 0 then acc else acc *
      if i < 2 then (#v[64, R, 1, 1, 1][i]! + #v[1, 1, 1, 1, 1][i]! - 1) / max #v[1, 1, 1, 1, 1][i]! 1
      else 1) 1 (List.range 2)) = R := by
    simp [List.range_succ]
  rw [hrows]
  have hst : m.globalStrides[0] = st := by simp [hgs]
  refine (foldlM_ok _ (fun b (a : Nat) => b ++ ((List.range' 0 8).map (chunk m c0 c1 smem a)).toArray)
    _ _ ?_).trans ?_
  · intro a ha b
    simp only [List.mem_range'] at ha
    have hmod : (a : Int) % R = a := Int.emod_eq_of_lt (by omega) (by omega)
    rw [forIn_ok _ (fun s (q : Nat) => s.push (chunk m c0 c1 smem a q))]
    · rw [foldl_push]
    · intro q hq s
      simp only [List.mem_range'] at hq
      rw [hmod]
      split
      · simp [chunk, hst]; rfl
      · exfalso; omega
  · rw [foldl_append]
    simp [box2]

/-- **A K-major wgmma operand reads the tile TMA wrote.** With the 128-byte swizzle on both, a
box of 128-byte rows placed at `smem` and a descriptor starting there: the descriptor's element
`(mn, k)` lies in chunk `(mn, k / 8)` of the box, `2 (k mod 8)` bytes in, and that chunk holds
the tensor's elements `(c0 + 8 (k / 8) …, c1 + mn)`; so element `(mn, k)` is the tensor's
`(c0 + k, c1 + mn)`. -/
theorem operand_chunk {m : TensorMap} (hsw : m.swizzle = .b128) {dsc : Desc}
    (hs : dsc.swizzle = .b128) (hb : dsc.baseOffset = 0) (hsbo : dsc.sbo = 1024) {smem : Nat}
    (hst : dsc.start = smem) (h16 : smem % 16 = 0) {mn k : Nat} {c0 c1 : Int} :
    dsc.addr .bf16 true mn k = .ok ((chunk m c0 c1 smem mn (k / 8)).smem + 2 * (k % 8)) ∧
      2 * (k % 8) < (chunk m c0 c1 smem mn (k / 8)).bytes := by
  refine ⟨?_, by simp [chunk]; omega⟩
  rw [addr_kMajor_b128 hs hb hsbo, hst]
  have e : smem + 128 * mn + 2 * k = 16 * (smem / 16 + 8 * mn + k / 8) + 2 * (k % 8) := by omega
  have e' : smem + mn * 128 + k / 8 * 16 = 16 * (smem / 16 + 8 * mn + k / 8) := by omega
  rw [e, apply_chunk _ (by omega)]
  simp only [chunk, hsw, e']

end NV.Formal.TileLayout
