/-!
# FlashAttention's Proposition 3

Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
(NeurIPS 2022), Proposition 3: *let `N` be the sequence length, `d` the head dimension and `M`
the size of SRAM with `d ≤ M ≤ Nd`; there does not exist an algorithm to compute exact attention
with `o(N²d²M⁻¹)` HBM accesses for all `M` in `[d, Nd]`.*

Their proof: at `M = Nd` the claimed count `N²d²/M` is `Nd`, and any exact algorithm reads its
`Nd` inputs. So the proposition is a statement about an algorithm *family* over all `M`, and
follows from compulsory traffic alone (`NV.Bounds.compulsory_traffic`, which gives an exact
attention program at least the bytes its outputs depend on). At any one `M` it bounds nothing
beyond compulsory traffic; it is restated here as the theorem it is, and the attention bounds
(`NV.Bounds.Specs.attention`) use compulsory traffic, the multiply-adds and the exponentials.

`LittleO Q`: the accesses `Q N d M` of an algorithm family are `o(N²d²/M)` uniformly over
`d ≤ M ≤ Nd`: for every `e > 0`, from some `N₀` on, `e · Q · M < N²d²`.
-/

namespace NV.Bounds.Attention

/-- `Q N d M = o(N² d² / M)` as `N` grows, uniformly in `d ≥ 1` and `d ≤ M ≤ N d`. -/
def LittleO (Q : Nat → Nat → Nat → Nat) : Prop :=
  ∀ e > 0, ∃ N₀, ∀ N ≥ N₀, ∀ d ≥ 1, ∀ M, d ≤ M → M ≤ N * d → e * Q N d M * M < N * N * (d * d)

/-- **Proposition 3.** An algorithm family that reads at least its `N d` query elements (as every
exact attention program does: `compulsory_traffic`) does not have `o(N²d²/M)` HBM accesses over
`[d, Nd]`. -/
theorem prop3 (Q : Nat → Nat → Nat → Nat) (hQ : ∀ N d M, N * d ≤ Q N d M) : ¬ LittleO Q := by
  intro h
  obtain ⟨N₀, hN₀⟩ := h 1 (by decide)
  have := hN₀ (N₀ + 1) (by omega) 1 (Nat.le_refl 1) (N₀ + 1) (by omega) (by omega)
  have hq := hQ (N₀ + 1) 1 (N₀ + 1)
  simp only [Nat.mul_one, Nat.one_mul] at this hq
  have := Nat.mul_le_mul_right (N₀ + 1) hq
  omega

end NV.Bounds.Attention
