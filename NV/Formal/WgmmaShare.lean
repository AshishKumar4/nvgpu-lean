import NV.Async.WgmmaNum
import NV.Formal.WgmmaData
import NV.Formal.TileLayout

/-!
# What a warp's share of a warpgroup MMA computes on

A share's result is `Share.finish` of its operands (`Share.operands`) and its accumulator.
`operands_smem`: for a warp's share with A in shared memory, the operands are A's sixteen rows and
B's columns read element by element at their descriptor addresses. `plan_result`: when the share's
loads returned the chunks of a shared memory `M`, its result is `finish` of the 16-bit elements of
`M` at those addresses: the MMA multiplies the matrices the descriptors name. `plan_tile`: with
both operands K-major bf16 tiles under the 128-byte swizzle, laid out as TMA writes a box of
128-byte rows (`TileLayout`), it multiplies the global tensors' tiles the boxes were copied from.
-/

namespace NV.Formal.WgmmaShare

open NV.Async NV.Async.Wgmma NV.Formal.WgmmaData NV.Formal.TileLayout

/-- A warp's share computes all sixteen of the warp's rows. -/
theorem rows_full {s : Share} (hl : s.lanes = Array.range 32) :
    s.rows = (List.range 16).map (16 * s.warp + ·) := by
  simp only [Share.rows, hl]
  refine List.filter_eq_self.2 fun r hr => ?_
  obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hr
  simp only [List.mem_range] at hi
  have key : ∀ i < 16, (Array.range 32).any (fun l => i % 8 == l / 4) = true := by
    intro i _
    rw [Array.any_eq_true]
    exact ⟨4 * (i % 8), by simp <;> omega, by simp <;> omega⟩
  simpa [Nat.add_sub_cancel_left] using key i hi

theorem idxOf_full (w : Nat) {r : Nat} (hr : r < 16) :
    ((List.range 16).map (16 * w + ·)).idxOf? (16 * w + r) = some r := by
  rw [List.idxOf?_eq_some_iff]
  refine ⟨by simpa using hr, by simp, fun j hj => ?_⟩
  simp only [List.getElem_map, List.getElem_range]
  omega

/-- **The operands of a warp's share** with A in shared memory: A's rows and B's columns, each
element read at its address (`A i k` for row `i` of the warp's sixteen, `B c k` for column `c`). -/
theorem operands_smem {s : Share} {d : Desc} {km : Bool} (ha : s.a = .smem d km)
    {aAddrs bAddrs : Array Nat} {A B : Nat → Nat → Nat}
    (hA : ∀ i < 16, ∀ k < s.ty.k, aAddrs[i * s.ty.k + k]! = A i k)
    (hB : ∀ c < s.n, ∀ k < s.ty.k, bAddrs[c * s.ty.k + k]! = B c k) (get : Nat → Nat) :
    s.operands ((List.range 16).map (16 * s.warp + ·)) aAddrs bAddrs get =
      ((Array.range 16).map fun i => (Array.range s.ty.k).map fun k => s.ty.negate s.negA (get (A i k)),
        (Array.range s.n).map fun c => (Array.range s.ty.k).map fun k =>
          s.ty.negate s.negB (get (B c k))) := by
  simp only [Share.operands, ha]
  refine Prod.ext ?_ ?_
  · refine Array.ext (by simp) fun i h₁ h₂ => ?_
    simp only [Array.size_map, Array.size_range] at h₁
    simp only [Array.getElem_map, Array.getElem_range, idxOf_full s.warp h₁]
    refine Array.ext (by simp) fun k h₃ h₄ => ?_
    simp only [Array.size_map, Array.size_range] at h₃
    simp only [Array.getElem_map, Array.getElem_range, hA i h₁ k h₃]
  · refine Array.ext (by simp) fun c h₁ h₂ => ?_
    simp only [Array.size_map, Array.size_range] at h₁
    simp only [Array.getElem_map, Array.getElem_range]
    refine Array.ext (by simp) fun k h₃ h₄ => ?_
    simp only [Array.size_map, Array.size_range] at h₃
    simp only [Array.getElem_map, Array.getElem_range, hB c h₁ k h₃]

/-- The values `f i k` for `i < n`, `k < K`, row by row. -/
def blocks {α : Type} (f : Nat → Nat → α) (n K : Nat) : Array α :=
  (Array.range n).flatMap fun i => (Array.range K).map (f i)

theorem flatMap_congr' {α β : Type} {l : List α} {f g : α → List β} (h : ∀ a ∈ l, f a = g a) :
    l.flatMap f = l.flatMap g := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    simp only [List.flatMap_cons, h a (List.mem_cons_self ..), ih fun b hb => h b (List.mem_cons_of_mem _ hb)]

theorem blocks_toList {α : Type} (f : Nat → Nat → α) (n K : Nat) :
    (blocks f n K).toList = (List.range n).flatMap fun i => (List.range K).map (f i) := by
  simp [blocks, Array.toList_flatMap]

theorem flat_length {α : Type} (f : Nat → Nat → α) (K : Nat) :
    ∀ n, ((List.range n).flatMap fun i => (List.range K).map (f i)).length = n * K
  | 0 => by simp
  | n + 1 => by
    rw [List.range_succ, List.flatMap_append, List.length_append, flat_length f K n]
    simp [Nat.succ_mul]

theorem flat_get {α : Type} (f : Nat → Nat → α) (K : Nat) : ∀ n i k, i < n → k < K →
    ∀ h : i * K + k < ((List.range n).flatMap fun i => (List.range K).map (f i)).length,
    ((List.range n).flatMap fun i => (List.range K).map (f i))[i * K + k] = f i k
  | n + 1, i, k, hi, hk, h => by
    simp only [List.range_succ, List.flatMap_append, List.getElem_append, flat_length]
    split
    · exact flat_get f K n i k (by
        rcases Nat.lt_or_ge i n with c | c
        · exact c
        · exfalso; have := Nat.mul_le_mul_right K c; omega) hk _
    · next hlt =>
      have hin : i = n := by
        rcases Nat.lt_or_ge i n with c | c
        · exfalso; have := Nat.mul_le_mul_right K (Nat.succ_le_of_lt c)
          simp only [Nat.succ_mul] at this; omega
        · omega
      subst hin
      simp

theorem blocks_size {α : Type} (f : Nat → Nat → α) (n K : Nat) : (blocks f n K).size = n * K := by
  rw [← Array.length_toList, blocks_toList, flat_length]

/-- Element `(i, k)` of the blocks is `f i k`. -/
theorem blocks_get {α : Type} [Inhabited α] (f : Nat → Nat → α) {n K i k : Nat} (hi : i < n)
    (hk : k < K) : (blocks f n K)[i * K + k]! = f i k := by
  have hlen : i * K + k < (blocks f n K).size := by
    rw [blocks_size]
    have := Nat.mul_le_mul_right K (Nat.succ_le_of_lt hi)
    simp only [Nat.succ_mul] at this; omega
  rw [getElem!_pos (blocks f n K) _ hlen, ← Array.getElem_toList]
  simp only [blocks_toList]
  exact flat_get f K n i k hi hk _

theorem blocks_mem {α : Type} (f : Nat → Nat → α) {n K i k : Nat} (hi : i < n) (hk : k < K) :
    f i k ∈ blocks f n K := by
  rw [← Array.mem_toList_iff, blocks_toList, List.mem_flatMap]
  exact ⟨i, by simpa using hi, List.mem_map.2 ⟨k, by simpa using hk, rfl⟩⟩

theorem mapM_ok {xs : Array Nat} {src : Array (Except String Nat)} (h : src = xs.map Except.ok) :
    src.mapM id = pure xs := by
  subst h
  rw [Array.mapM_map]
  show xs.mapM (fun x => (pure x : Except String Nat)) = _
  rw [Array.mapM_pure, Array.map_id']

/-- The operand addresses a share computes, row by row (`rows`, the warp's sixteen from `w`). -/
theorem rows_blocks {w n K : Nat} {g : Nat → Nat → Except String Nat} {f : Nat → Nat → Nat}
    (hg : ∀ i < n, ∀ k < K, g (w + i) k = .ok (f i k)) :
    ((List.range n).map (w + ·)).toArray.flatMap (fun r => (Array.range K).map (g r)) =
      (blocks f n K).map Except.ok := by
  apply Array.toList_inj.1
  simp only [Array.toList_flatMap, Array.toList_map, Array.toList_range, blocks_toList,
    List.map_flatMap, List.flatMap_map, List.map_map]
  apply flatMap_congr'
  intro i hi
  simp only [List.mem_range] at hi
  apply List.map_congr_left
  intro k hk
  simp only [List.mem_range] at hk
  simp [hg i hi k hk]

theorem cols_blocks {n K : Nat} {g : Nat → Nat → Except String Nat} {f : Nat → Nat → Nat}
    (hg : ∀ c < n, ∀ k < K, g c k = .ok (f c k)) :
    (Array.range n).flatMap (fun c => (Array.range K).map (g c)) = (blocks f n K).map Except.ok := by
  have := rows_blocks (w := 0) (n := n) (K := K) (g := g) (f := f) (by simpa using hg)
  simpa [Array.toList_inj.symm, List.map_id'] using this

/-- The 16-bit little-endian element of `M` at `a`. -/
def half (M : Nat → UInt8) (a : Nat) : Nat := (M a).toNat + 2 ^ 8 * (M (a + 1)).toNat

/-- **What a warp's share of a warpgroup MMA computes**, A in shared memory, 16-bit inputs: when
its loads returned the chunks of shared memory `M` (`data`: four words per chunk, in the plan's
order), its result is `finish` of the matrices `M` holds at the descriptors' addresses (`A i k`
for row `i` of the warp's sixteen, `B c k` for column `c`), negated as the instruction asks. -/
theorem plan_result {s : Share} {d : Desc} {km : Bool} (ha : s.a = .smem d km)
    (hl : s.lanes = Array.range 32) (h16 : s.ty.bits = 16) {addr : Nat → UInt64} {pl : Plan}
    (hp : s.plan addr = .ok pl) {A B : Nat → Nat → Nat}
    (hA : ∀ i < 16, ∀ k < s.ty.k, d.addr s.ty km (16 * s.warp + i) k = .ok (A i k))
    (hB : ∀ c < s.n, ∀ k < s.ty.k, s.b.addr s.ty s.bKMajor c k = .ok (B c k))
    (hevA : ∀ i < 16, ∀ k < s.ty.k, A i k % 2 = 0) (hevB : ∀ c < s.n, ∀ k < s.ty.k, B c k % 2 = 0)
    {M : Nat → UInt8} {data : Resp}
    (hdata : ∀ i (hi : i < (chunks (blocks A 16 s.ty.k ++ blocks B s.n s.ty.k)).size),
      data.extract (4 * i) (4 * i + 4) =
        chunkWords M (chunks (blocks A 16 s.ty.k ++ blocks B s.n s.ty.k))[i]) (c : Array UInt32) :
    pl.result data c = s.finish
      ((Array.range 16).map fun i => (Array.range s.ty.k).map fun k => s.ty.negate s.negA (half M (A i k)))
      ((Array.range s.n).map fun j => (Array.range s.ty.k).map fun k => s.ty.negate s.negB (half M (B j k)))
      c := by
  have hsA := rows_blocks (w := 16 * s.warp) (g := fun r k => d.addr s.ty km r k) hA
  have hsB := cols_blocks (g := fun j k => s.b.addr s.ty s.bKMajor j k) hB
  simp only [Share.plan, ha, rows_full hl, mapM_ok hsA, mapM_ok hsB] at hp
  split at hp
  · cases hp
  split at hp
  · cases hp
  simp only [pure_bind] at hp
  cases hp
  have hnd := (chunks_spec (blocks A 16 s.ty.k ++ blocks B s.n s.ty.k)).1
  have hin := (chunks_spec (blocks A 16 s.ty.k ++ blocks B s.n s.ty.k)).2
  dsimp only
  rw [operands_smem ha (fun i hi k hk => blocks_get A hi hk) (fun j hj k hk => blocks_get B hj hk)]
  dsimp only
  congr 1
  · refine Array.map_congr_left fun i hi => Array.map_congr_left fun k hk => ?_
    simp only [Array.mem_range] at hi hk
    rw [h16, elem16 hnd hdata (hin _ (Array.mem_append.2 (.inl (blocks_mem A hi hk)))) (hevA i hi k hk)]
    rfl
  · refine Array.map_congr_left fun j hj => Array.map_congr_left fun k hk => ?_
    simp only [Array.mem_range] at hj hk
    rw [h16, elem16 hnd hdata (hin _ (Array.mem_append.2 (.inr (blocks_mem B hj hk)))) (hevB j hj k hk)]
    rfl

/-- The 128-byte swizzle keeps a 16-byte-aligned address even. -/
theorem apply_even (a : Nat) : Swizzle.b128.apply (16 * a) % 2 = 0 := by
  simp only [Swizzle.apply]
  rw [show 16 * a = 16 * a + 0 by omega, xor_low (by omega : 0 < 16)]
  omega

/-- Where element `(row, k)` of a box of 128-byte rows of 2-byte elements sits in shared memory:
chunk `k / 8` of the row, `2 (k mod 8)` bytes in. -/
def tileAddr (m : TensorMap) (c0 c1 : Int) (smem row k : Nat) : Nat :=
  (chunk m c0 c1 smem row (k / 8)).smem + 2 * (k % 8)

/-- The global address of chunk `q` of row `row` of the box at `(c0, c1)`. -/
def gaddr (m : TensorMap) (c0 c1 : Int) (row q : Nat) : Nat :=
  ((m.globalAddress.toNat : Int) + (c1 + row) * m.globalStrides[0]! + (c0 + q * 8) * 2).toNat

theorem tileAddr_even {m : TensorMap} (hsw : m.swizzle = .b128) {smem : Nat} (h16 : smem % 16 = 0)
    (c0 c1 : Int) (row k : Nat) : tileAddr m c0 c1 smem row k % 2 = 0 := by
  simp only [tileAddr, chunk, hsw]
  rw [show smem + row * 128 + k / 8 * 16 = 16 * (smem / 16 + 8 * row + k / 8) by omega]
  have := apply_even (smem / 16 + 8 * row + k / 8)
  omega

/-- **A warp's share of a bf16 warpgroup MMA on TMA tiles** multiplies the global tiles: with A and
B K-major in shared memory under the 128-byte swizzle (descriptors at `smA`, `smB`, 8-row groups
1024 bytes apart), each a box of 128-byte rows TMA placed there (`tileAddr`), and shared memory `M`
holding each chunk of the boxes as global memory `G` holds it, the share's result is `finish` of
the global elements: row `i` of the warp's sixteen, element `k`, of A's box at `(a0, a1)`, and
column `c` of B's box at `(b0, b1)`. -/
theorem plan_tile {s : Share} {d : Desc} (ha : s.a = .smem d true) (hkb : s.bKMajor = true)
    (hl : s.lanes = Array.range 32) (hty : s.ty = .bf16)
    {smA smB : Nat} (hdA : d.swizzle = .b128) (hbA : d.baseOffset = 0) (hsA : d.sbo = 1024)
    (hstA : d.start = smA) (h16A : smA % 16 = 0)
    (hdB : s.b.swizzle = .b128) (hbB : s.b.baseOffset = 0) (hsB : s.b.sbo = 1024)
    (hstB : s.b.start = smB) (h16B : smB % 16 = 0)
    {mA mB : TensorMap} (hswA : mA.swizzle = .b128) (hswB : mB.swizzle = .b128) {a0 a1 b0 b1 : Int}
    {addr : Nat → UInt64} {pl : Plan} (hp : s.plan addr = .ok pl) {M G : Nat → UInt8} {data : Resp}
    (hdata : ∀ i (hi : i < (chunks (blocks (fun i k => tileAddr mA a0 a1 smA (16 * s.warp + i) k) 16 s.ty.k ++
        blocks (tileAddr mB b0 b1 smB) s.n s.ty.k)).size),
      data.extract (4 * i) (4 * i + 4) = chunkWords M (chunks (blocks (fun i k =>
        tileAddr mA a0 a1 smA (16 * s.warp + i) k) 16 s.ty.k ++ blocks (tileAddr mB b0 b1 smB) s.n s.ty.k))[i])
    (hTA : ∀ row q j, j < 16 → M ((chunk mA a0 a1 smA row q).smem + j) = G (gaddr mA a0 a1 row q + j))
    (hTB : ∀ row q j, j < 16 → M ((chunk mB b0 b1 smB row q).smem + j) = G (gaddr mB b0 b1 row q + j))
    (c : Array UInt32) :
    pl.result data c = s.finish
      ((Array.range 16).map fun i => (Array.range s.ty.k).map fun k =>
        s.ty.negate s.negA (half G (gaddr mA a0 a1 (16 * s.warp + i) (k / 8) + 2 * (k % 8))))
      ((Array.range s.n).map fun j => (Array.range s.ty.k).map fun k =>
        s.ty.negate s.negB (half G (gaddr mB b0 b1 j (k / 8) + 2 * (k % 8))))
      c := by
  have h16 : s.ty.bits = 16 := by rw [hty]; rfl
  have hA : ∀ i < 16, ∀ k < s.ty.k,
      d.addr s.ty true (16 * s.warp + i) k = .ok (tileAddr mA a0 a1 smA (16 * s.warp + i) k) :=
    fun i _ k _ => by rw [hty]; exact (operand_chunk hswA hdA hbA hsA hstA h16A).1
  have hB : ∀ j < s.n, ∀ k < s.ty.k, s.b.addr s.ty s.bKMajor j k = .ok (tileAddr mB b0 b1 smB j k) :=
    fun j _ k _ => by rw [hty, hkb]; exact (operand_chunk hswB hdB hbB hsB hstB h16B).1
  rw [plan_result ha hl h16 hp hA hB (fun i _ k _ => tileAddr_even hswA h16A _ _ _ _)
    (fun j _ k _ => tileAddr_even hswB h16B _ _ _ _) hdata c]
  have hM : ∀ (m : TensorMap) (c0 c1 : Int) (sm row k : Nat),
      (∀ row q j, j < 16 → M ((chunk m c0 c1 sm row q).smem + j) = G (gaddr m c0 c1 row q + j)) →
      half M (tileAddr m c0 c1 sm row k) = half G (gaddr m c0 c1 row (k / 8) + 2 * (k % 8)) := by
    intro m c0 c1 sm row k hT
    simp only [half, tileAddr]
    rw [hT row (k / 8) (2 * (k % 8)) (by omega), Nat.add_assoc, hT row (k / 8) (2 * (k % 8) + 1) (by omega),
      ← Nat.add_assoc]
  congr 1
  · refine Array.map_congr_left fun i _ => Array.map_congr_left fun k _ => ?_
    rw [hM mA a0 a1 smA _ k hTA]
  · refine Array.map_congr_left fun j _ => Array.map_congr_left fun k _ => ?_
    rw [hM mB b0 b1 smB j k hTB]

/-! ## Where the accumulator's values sit, and a chain of MMAs -/

theorem idxOf_range : ∀ n l, l < n → (Array.range n).idxOf l = l
  | 0, _, h => absurd h (Nat.not_lt_zero _)
  | n + 1, l, h => by
    rw [Array.range_succ, Array.idxOf_append]
    split
    · next hm => exact idxOf_range n l (Array.mem_range.1 hm)
    · next hm =>
      have : l = n := by
        have : ¬ l < n := fun c => hm (Array.mem_range.2 c)
        omega
      subst this
      simp

/-- **A warp's accumulator words** (`f32` or `s32`): word `v` of lane `l` holds the product's
element at `accIndex l v`. -/
theorem pack_get {s : Share} (hl : s.lanes = Array.range 32) (hacc : s.acc ≠ .f16) (dm : Array Nat)
    {l v : Nat} (hlt : l < 32) (hv : v < accWords s.acc s.n) :
    (s.pack dm)[l * accWords s.acc s.n + v]! = dm[s.accIndex l v]!.toUInt32 := by
  have hf : (s.acc == .f16) = false := by simpa using hacc
  have : s.pack dm = blocks (fun l w => dm[s.accIndex l w]!.toUInt32) 32 (accWords s.acc s.n) := by
    simp only [Share.pack, hl, hf, Bool.false_eq_true, ite_false, blocks]
  rw [this, blocks_get _ hlt hv]

/-- `accIndex` and `accIn`'s positions are inverse: position `(i, j)` of the block is value
`j mod 2 + 2 (i / 8) + 4 (j / 8)` of lane `4 (i mod 8) + (j mod 8) / 2`. -/
theorem accIndex_inv (s : Share) {i j : Nat} (hi : i < 16) (hj : j < s.n) :
    s.accIndex (4 * (i % 8) + j % 8 / 2) (j % 2 + 2 * (i / 8) + 4 * (j / 8)) = i * s.n + j := by
  have e1 : 16 * s.warp + (4 * (i % 8) + j % 8 / 2) / 4 + 8 * ((j % 2 + 2 * (i / 8) + 4 * (j / 8)) / 2 % 2) -
      16 * s.warp = i := by omega
  have e2 : 2 * ((4 * (i % 8) + j % 8 / 2) % 4) + (j % 2 + 2 * (i / 8) + 4 * (j / 8)) % 2 +
      8 * ((j % 2 + 2 * (i / 8) + 4 * (j / 8)) / 4) = j := by omega
  show (16 * s.warp + (4 * (i % 8) + j % 8 / 2) / 4 + 8 * ((j % 2 + 2 * (i / 8) + 4 * (j / 8)) / 2 % 2) -
      16 * s.warp) * s.n + (2 * ((4 * (i % 8) + j % 8 / 2) % 4) + (j % 2 + 2 * (i / 8) + 4 * (j / 8)) % 2 +
      8 * ((j % 2 + 2 * (i / 8) + 4 * (j / 8)) / 4)) = i * s.n + j
  rw [e1, e2]

theorem extract_getD {xs : Array UInt32} {a w v : Nat} (hv : v < w) (h : a + w ≤ xs.size) :
    (xs.extract a (a + w))[v]?.getD 0 = xs[a + v]! := by
  have hs : v < (xs.extract a (a + w)).size := by simp only [Array.size_extract]; omega
  rw [getElem?_pos (xs.extract a (a + w)) v hs, Option.getD_some, Array.getElem_extract,
    getElem!_pos xs (a + v) (by omega)]

/-- **One MMA's result read back by the next**: with every lane computed, a 32-bit accumulator
and `scale-d`, the next share sees the product block, as 32-bit words. -/
theorem accIn_pack {s : Share} (hl : s.lanes = Array.range 32) (hacc : s.acc = .f32)
    (hn : s.n % 8 = 0) (hs : s.scaleD = true) (dm : Array Nat) :
    s.accIn (s.pack dm) = (Array.range (16 * s.n)).map fun x => dm[x]! % 2 ^ 32 := by
  have hne : s.acc ≠ .f16 := by rw [hacc]; decide
  have hW : accWords s.acc s.n = s.n / 2 := by simp [accWords, hacc]
  have hsize : (s.pack dm).size = 32 * (s.n / 2) := by
    have hp : s.pack dm = blocks (fun l w => dm[s.accIndex l w]!.toUInt32) 32 (accWords s.acc s.n) := by
      simp only [Share.pack, hl, hacc, blocks]
      rfl
    rw [hp, blocks_size, hW]
  refine Array.ext (by simp [Share.accIn]) fun x h₁ _ => ?_
  simp only [Share.accIn, Array.size_map, Array.size_range] at h₁
  have hnpos : 0 < s.n := by
    rcases Nat.eq_zero_or_pos s.n with h | h
    · rw [h] at h₁; omega
    · exact h
  have hi : x / s.n < 16 := (Nat.div_lt_iff_lt_mul hnpos).2 h₁
  have hj : x % s.n < s.n := Nat.mod_lt _ hnpos
  have hx : x / s.n * s.n + x % s.n = x := by rw [Nat.mul_comm]; exact Nat.div_add_mod x s.n
  have hlt : 4 * (x / s.n % 8) + x % s.n % 8 / 2 < 32 := by omega
  have hv : x % s.n % 2 + 2 * (x / s.n / 8) + 4 * (x % s.n / 8) < s.n / 2 := by omega
  have hc : s.lanes.contains (4 * (x / s.n % 8) + x % s.n % 8 / 2) = true := by
    rw [hl, Array.contains_iff_mem, Array.mem_range]; exact hlt
  have hk : s.lanes.idxOf (4 * (x / s.n % 8) + x % s.n % 8 / 2) = 4 * (x / s.n % 8) + x % s.n % 8 / 2 := by
    rw [hl]; exact idxOf_range 32 _ hlt
  have hget := pack_get hl hne dm hlt (hW ▸ hv)
  rw [hW, accIndex_inv s hi hj, hx] at hget
  have hW' : accWords .f32 s.n = s.n / 2 := by simp [accWords]
  simp only [Share.accIn, Array.getElem_map, Array.getElem_range, hs, hc, Bool.and_self, ite_true, hk,
    hacc, hW', accGet]
  simp only [show (Acc.f32 == Acc.f16) = false from rfl, Bool.false_eq_true, ite_false]
  generalize 4 * (x / s.n % 8) + x % s.n % 8 / 2 = L at hget hlt ⊢
  generalize x % s.n % 2 + 2 * (x / s.n / 8) + 4 * (x % s.n / 8) = V at hget hv ⊢
  have hroom : L * (s.n / 2) + s.n / 2 ≤ (s.pack dm).size := by
    have := Nat.mul_le_mul_right (s.n / 2) (Nat.succ_le_of_lt hlt)
    rw [Nat.succ_mul] at this
    omega
  rw [extract_getD hv hroom, hget]
  simp [Nat.toUInt32, UInt32.toNat_ofNat']

/-- **A chain of MMAs on one warp's accumulator** (every lane computed, 32-bit accumulator,
`scale-d`, `N` a multiple of 8): its registers after the steps are the last product block, packed,
where each step's product takes the previous one as 32-bit words. -/
theorem finish_chain {s : Share} (hl : s.lanes = Array.range 32) (hacc : s.acc = .f32)
    (hn : s.n % 8 = 0) (hs : s.scaleD = true) :
    ∀ (ops : List (Array (Array Nat) × Array (Array Nat))) (dm : Array Nat),
      ops.foldl (fun c ab => s.finish ab.1 ab.2 c) (s.pack dm) =
        s.pack (ops.foldl (fun d ab => s.product ab.1 ab.2
          ((Array.range (16 * s.n)).map fun x => d[x]! % 2 ^ 32)) dm)
  | [], _ => rfl
  | ab :: ops, dm => by
    rw [List.foldl_cons, List.foldl_cons]
    show ops.foldl _ (s.pack (s.product ab.1 ab.2 (s.accIn (s.pack dm)))) = _
    rw [accIn_pack hl hacc hn hs dm]
    exact finish_chain hl hacc hn hs ops _

/-- Two shares compute their results alike: the same `N`, warp, lanes, accumulator, `scale-d` and
input types (their operands' descriptors may differ). -/
structure SameAcc (s s' : Share) : Prop where
  n : s'.n = s.n
  warp : s'.warp = s.warp
  lanes : s'.lanes = s.lanes
  acc : s'.acc = s.acc
  scaleD : s'.scaleD = s.scaleD
  ty : s'.ty = s.ty
  tyB : s'.tyB = s.tyB

/-- Shares that compute alike finish alike: each step of a K loop (its own stage's descriptors)
is `finish_chain`'s step. -/
theorem finish_congr {s s' : Share} (h : SameAcc s s') : s'.finish = s.finish := by
  funext A B c
  simp only [Share.finish, Share.pack, Share.product, Share.accIn, Share.accIndex, h.n, h.warp, h.lanes,
    h.acc, h.scaleD, h.ty, h.tyB]

/-! ## A K loop of MMAs on TMA tiles -/

/-- One step of a K loop: a warp's share of a bf16 wgmma, its plan and its loads' data, on A and B
boxes placed by TMA at `smA`, `smB` from the global tensors `mA` at `(a0, a1)` and `mB` at
`(b0, b1)`, shared memory holding `M` when it reads. -/
structure TileStep where
  s : Share
  pl : Plan
  data : Resp
  d : Desc
  smA : Nat
  smB : Nat
  mA : TensorMap
  mB : TensorMap
  a0 : Int
  a1 : Int
  b0 : Int
  b1 : Int
  M : Nat → UInt8

/-- What `plan_tile` asks of a step, with global memory `G`. -/
structure TileStep.Valid (st : TileStep) (addr : Nat → UInt64) (G : Nat → UInt8) : Prop where
  a : st.s.a = .smem st.d true
  bKMajor : st.s.bKMajor = true
  lanes : st.s.lanes = Array.range 32
  ty : st.s.ty = .bf16
  dA : st.d.swizzle = .b128
  bA : st.d.baseOffset = 0
  sA : st.d.sbo = 1024
  stA : st.d.start = st.smA
  alA : st.smA % 16 = 0
  dB : st.s.b.swizzle = .b128
  bB : st.s.b.baseOffset = 0
  sB : st.s.b.sbo = 1024
  stB : st.s.b.start = st.smB
  alB : st.smB % 16 = 0
  swA : st.mA.swizzle = .b128
  swB : st.mB.swizzle = .b128
  plan : st.s.plan addr = .ok st.pl
  data : ∀ i (hi : i < (chunks (blocks (fun i k => tileAddr st.mA st.a0 st.a1 st.smA (16 * st.s.warp + i) k)
      16 st.s.ty.k ++ blocks (tileAddr st.mB st.b0 st.b1 st.smB) st.s.n st.s.ty.k)).size),
    st.data.extract (4 * i) (4 * i + 4) = chunkWords st.M (chunks (blocks (fun i k =>
      tileAddr st.mA st.a0 st.a1 st.smA (16 * st.s.warp + i) k) 16 st.s.ty.k ++
      blocks (tileAddr st.mB st.b0 st.b1 st.smB) st.s.n st.s.ty.k))[i]
  tileA : ∀ row q j, j < 16 → st.M ((chunk st.mA st.a0 st.a1 st.smA row q).smem + j) =
    G (gaddr st.mA st.a0 st.a1 row q + j)
  tileB : ∀ row q j, j < 16 → st.M ((chunk st.mB st.b0 st.b1 st.smB row q).smem + j) =
    G (gaddr st.mB st.b0 st.b1 row q + j)

/-- The step's A rows and B columns, read from global memory `G`. -/
def TileStep.A (st : TileStep) (G : Nat → UInt8) : Array (Array Nat) :=
  (Array.range 16).map fun i => (Array.range st.s.ty.k).map fun k =>
    st.s.ty.negate st.s.negA (half G (gaddr st.mA st.a0 st.a1 (16 * st.s.warp + i) (k / 8) + 2 * (k % 8)))

def TileStep.B (st : TileStep) (G : Nat → UInt8) : Array (Array Nat) :=
  (Array.range st.s.n).map fun j => (Array.range st.s.ty.k).map fun k =>
    st.s.ty.negate st.s.negB (half G (gaddr st.mB st.b0 st.b1 j (k / 8) + 2 * (k % 8)))

/-- **A K loop of wgmmas on TMA tiles**: a warp's registers after the steps, each a share of a
bf16 wgmma that computes like `s` (every lane, 32-bit accumulator, `scale-d`, `N` a multiple of
8) on boxes TMA placed from global memory `G`, are the last product block of the global tiles,
packed, each step's product taking the previous one as 32-bit words. -/
theorem kloop {s : Share} (hl : s.lanes = Array.range 32) (hacc : s.acc = .f32) (hn : s.n % 8 = 0)
    (hs : s.scaleD = true) {addr : Nat → UInt64} {G : Nat → UInt8} :
    ∀ (steps : List TileStep), (∀ st ∈ steps, st.Valid addr G ∧ SameAcc s st.s) → ∀ (dm : Array Nat),
      steps.foldl (fun c st => st.pl.result st.data c) (s.pack dm) =
        s.pack (steps.foldl (fun d st => s.product (st.A G) (st.B G)
          ((Array.range (16 * s.n)).map fun x => d[x]! % 2 ^ 32)) dm)
  | [], _, _ => rfl
  | st :: steps, h, dm => by
    obtain ⟨v, hsame⟩ := h st (List.mem_cons_self ..)
    rw [List.foldl_cons, List.foldl_cons]
    have hstep : st.pl.result st.data (s.pack dm) =
        s.pack (s.product (st.A G) (st.B G) ((Array.range (16 * s.n)).map fun x => dm[x]! % 2 ^ 32)) := by
      rw [plan_tile v.a v.bKMajor v.lanes v.ty v.dA v.bA v.sA v.stA v.alA v.dB v.bB v.sB v.stB v.alB
        v.swA v.swB v.plan v.data v.tileA v.tileB, finish_congr hsame]
      show s.pack (s.product _ _ (s.accIn (s.pack dm))) = _
      rw [accIn_pack hl hacc hn hs]
      rfl
    rw [hstep]
    exact kloop hl hacc hn hs steps (fun st' h' => h st' (List.mem_cons_of_mem _ h')) _

end NV.Formal.WgmmaShare
