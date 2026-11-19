import NV.PTX.Memory.Chain
import NV.Async.MBarrierWord
import NV.Formal.Count
import NV.Memory.Events

/-!
# Pipelines synchronized by mbarriers

A producer fills the slots of a ring and consumers drain them, each slot guarded by a *full*
mbarrier (the producer arrives expecting the round's bytes, the copies complete them) and an
*empty* one (the consumer arrives when it has read the slot). This file proves, in any consistent
execution, what the two barriers give the ring (`Examples.Ring.Race.Ring`'s `raw` and `war`):

* each full barrier's writes follow coherence order round by round, its arrive first, then its
  completes in any order (`Full.shape`): the producer arrives on a round only after the previous
  round's arrive, and an arrive on a phase still waiting for its bytes is undefined (§9.7.15.16.7,
  over-arrival), so a run in which every operation is defined has no other order;
* a consumer's wait that returned true for round `r` synchronizes with every complete of round
  `r` (`Full.raw`), by the release sequence, since its earlier wait bounds it below and the
  round's words before its last complete report the phase not done;
* likewise a producer's wait on the empty barrier for round `r` synchronizes with the consumer's
  arrive of round `r - 1` (`Empty.war`).
-/

namespace NV.Formal.Pipeline

open NV NV.PTX NV.PTX.Execution NV.Async Rel

/-! ## The barrier words -/

/-- A phase waiting for its one arrival. -/
def fresh (p : Nat) : UInt64 := MBarrierWord.encode { phase := p, pending := 1, expected := 1, tx := 0 }

/-- A phase whose arrival came, waiting for `rem` bytes. -/
def part (p rem : Nat) : UInt64 :=
  MBarrierWord.encode { phase := p, pending := 0, expected := 1, tx := rem }

/-- A word depends on the phase count's parity only, and not on the ghost. -/
theorem encode_congr {b b' : NV.PTX.MBarrier} (h₁ : b.pending = b'.pending)
    (h₂ : b.expected = b'.expected) (h₃ : b.tx = b'.tx) (h₄ : b.phase % 2 = b'.phase % 2) :
    MBarrierWord.encode b = MBarrierWord.encode b' := by
  simp [MBarrierWord.encode, h₁, h₂, h₃, h₄]

theorem decode_fresh (p : Nat) :
    MBarrierWord.decode (fresh p) = { phase := p % 2, pending := 1, expected := 1, tx := 0 } :=
  MBarrierWord.decode_encode _ (by simp) (by simp) (by simp) (by simp)

theorem decode_part (p : Nat) {rem : Nat} (h : rem < 2 ^ 20) :
    MBarrierWord.decode (part p rem) = { phase := p % 2, pending := 0, expected := 1, tx := rem } :=
  MBarrierWord.decode_encode _ (by simp) (by simp) (by simp) (by simp; omega)

theorem txOk_of {t : Int} (h₁ : -(2 ^ 20 : Int) < t) (h₂ : t < 2 ^ 20) :
    NV.PTX.MBarrier.txOk t = true := by
  unfold NV.PTX.MBarrier.txOk NV.PTX.MBarrier.countMax
  simp only [decide_eq_true_eq, show ((2 ^ 20 - 1 : Nat) : Int) = 1048575 from rfl]
  omega

theorem arrive_fresh (p : Nat) {T : Nat} (h0 : 0 < T) (h : T < 2 ^ 20) :
    MBarrierWord.apply (.arrive 1 T false) (fresh p) = .ok (part p T) := by
  have htx := txOk_of (t := (T : Int)) (by omega) (by omega)
  have hT : T ≠ 0 := by omega
  simp [MBarrierWord.apply, decode_fresh, NV.PTX.MBarrier.expectTx, htx, NV.PTX.MBarrier.arrive,
    NV.PTX.MBarrier.settle, hT, bind, Except.bind]
  exact encode_congr (by simp) (by simp) (by simp) (by simp)

theorem arrive_part (p : Nat) {T rem : Nat} (h : rem < 2 ^ 20) (w : UInt64) :
    MBarrierWord.apply (.arrive 1 T false) (part p rem) ≠ .ok w := by
  simp only [MBarrierWord.apply, decode_part p h, NV.PTX.MBarrier.expectTx]
  by_cases ht : NV.PTX.MBarrier.txOk ((rem : Int) + T) = true
  · simp [ht, NV.PTX.MBarrier.arrive, bind, Except.bind]
  · simp [ht, bind, Except.bind]

theorem complete_part (p : Nat) {b rem : Nat} (hb : b < rem) (h : rem < 2 ^ 20) :
    MBarrierWord.apply (.completeTx b) (part p rem) = .ok (part p (rem - b)) := by
  have htx := txOk_of (t := (rem : Int) - b) (by omega) (by omega)
  have hr : ((rem : Int) - b) ≠ 0 := by omega
  simp [MBarrierWord.apply, decode_part p h, NV.PTX.MBarrier.completeTx, htx,
    NV.PTX.MBarrier.settle, hr]
  exact encode_congr (by simp) (by simp) (by simp; omega) (by simp)

theorem complete_last (p : Nat) {b : Nat} (h : b < 2 ^ 20) :
    MBarrierWord.apply (.completeTx b) (part p b) = .ok (fresh (p + 1)) := by
  have htx := txOk_of (t := 0) (by omega) (by omega)
  simp [MBarrierWord.apply, decode_part p h, NV.PTX.MBarrier.completeTx, htx,
    NV.PTX.MBarrier.settle]
  exact encode_congr (by simp) (by simp) (by simp) (by simp)

theorem arrive_empty (p : Nat) :
    MBarrierWord.apply (.arrive 1 0 false) (fresh p) = .ok (fresh (p + 1)) := by
  have htx := txOk_of (t := 0) (by omega) (by omega)
  simp [MBarrierWord.apply, decode_fresh, NV.PTX.MBarrier.expectTx, htx, NV.PTX.MBarrier.arrive,
    NV.PTX.MBarrier.settle, bind, Except.bind]
  exact encode_congr (by simp) (by simp) (by simp) (by simp)

theorem done_fresh (p q : Nat) : MBarrierWord.parityDone (fresh p) q = (p % 2 != q % 2) := by
  simp [MBarrierWord.parityDone, decode_fresh, NV.PTX.MBarrier.parityDone]

theorem done_part (p q : Nat) {rem : Nat} (h : rem < 2 ^ 20) :
    MBarrierWord.parityDone (part p rem) q = (p % 2 != q % 2) := by
  simp [MBarrierWord.parityDone, decode_part p h, NV.PTX.MBarrier.parityDone]

/-! ## Words as bytes

An mbarrier word is 8 bytes of memory, little-endian; a read-modify-write reads it as `mword` and
writes `bytes64` of the new word (`Req.atomNew`); a wait's response is two 32-bit words, read
back as `word64`. -/

/-- The word 8 little-endian bytes spell, as an mbarrier operation reads it. -/
def mword (bs : List UInt8) : UInt64 := UInt64.ofNat (leNat bs)

/-- The bytes of a 64-bit word, low first. -/
def bytes64 (w : UInt64) : List UInt8 := wordBytes w.toUInt32 ++ wordBytes (w >>> 32).toUInt32

/-- The 64-bit word a response's two 32-bit words spell. -/
def word64 (resp : Resp) : UInt64 :=
  (resp[0]?.getD 0).toUInt64 ||| ((resp[1]?.getD 0).toUInt64 <<< (32 : UInt64))

theorem leNat_append : ∀ (xs ys : List UInt8), leNat (xs ++ ys) = leNat xs + 256 ^ xs.length * leNat ys
  | [], ys => by simp [leNat]
  | x :: xs, ys => by
    simp only [List.cons_append, leNat, List.foldr_cons, List.length_cons] at *
    rw [show (List.foldr (fun b acc => b.toNat + 256 * acc) 0 (xs ++ ys)) = leNat (xs ++ ys) from rfl,
      leNat_append xs ys, Nat.pow_succ]
    simp only [leNat]
    rw [Nat.mul_add, ← Nat.mul_assoc, Nat.mul_comm 256 (256 ^ xs.length)]
    omega

theorem bytes64_length (w : UInt64) : (bytes64 w).length = 8 := by simp [bytes64, wordBytes]

theorem mword_bytes64 (w : UInt64) : mword (bytes64 w) = w := by
  simp only [mword, bytes64, leNat_append, leNat_wordBytes, show (wordBytes w.toUInt32).length = 4 from rfl,
    UInt64.toNat_toUInt32, UInt64.toNat_shiftRight]
  apply UInt64.toNat_inj.1
  simp only [UInt64.toNat_ofNat', Nat.shiftRight_eq_div_pow]
  have := w.toNat_lt
  simp only [show (32 : UInt64).toNat % 64 = 32 from rfl]
  omega

/-- A defined mbarrier operation writes its new word. -/
theorem written_mbar {r : Req} {op : MBarOp} (hop : r.op = .mbar op) {bs : List UInt8} {w : UInt64}
    (h : MBarrierWord.apply op (mword bs) = .ok w) (lane : Nat) (p : Piece) :
    r.written lane p bs = some (bytes64 w) := by
  simp only [Req.written, Req.atomNew, hop]
  simp only [mword] at h
  simp [h, bytes64]

/-- The response of an 8-byte load spells the word its bytes do. -/
theorem word64_resp {bs : List UInt8} (h : bs.length = 8) :
    word64 ((toWords bs).take 2 ++ List.replicate (2 - (toWords bs).length) 0).toArray = mword bs := by
  match bs, h with
  | [b₀, b₁, b₂, b₃, b₄, b₅, b₆, b₇], _ =>
    have h₁ := b₀.toNat_lt; have h₂ := b₁.toNat_lt; have h₃ := b₂.toNat_lt; have h₄ := b₃.toNat_lt
    have h₅ := b₄.toNat_lt; have h₆ := b₅.toNat_lt; have h₇ := b₆.toNat_lt; have h₈ := b₇.toNat_lt
    generalize hn₀ : b₀.toNat + 256 * (b₁.toNat + 256 * (b₂.toNat + 256 * b₃.toNat)) = n₀
    generalize hn₁ : b₄.toNat + 256 * (b₅.toNat + 256 * (b₆.toNat + 256 * b₇.toNat)) = n₁
    have ht : toWords [b₀, b₁, b₂, b₃, b₄, b₅, b₆, b₇] = [UInt32.ofNat n₀, UInt32.ofNat n₁] := by
      simp [toWords, leNat, ← hn₀, ← hn₁]
    have hl : leNat [b₀, b₁, b₂, b₃, b₄, b₅, b₆, b₇] = n₀ + 2 ^ 32 * n₁ := by
      simp only [leNat, List.foldr]; omega
    rw [ht, mword, hl]
    show ((UInt32.ofNat n₀).toUInt64 ||| (UInt32.ofNat n₁).toUInt64 <<< (32 : UInt64)) = _
    apply UInt64.toNat_inj.1
    simp only [UInt64.toNat_or, UInt64.toNat_shiftLeft, UInt32.toNat_toUInt64, UInt32.toNat_ofNat',
      UInt64.toNat_ofNat', Nat.shiftLeft_eq, show (32 : UInt64).toNat % 64 = 32 from rfl]
    rw [Nat.mod_eq_of_lt (show n₀ < 2 ^ 32 by omega), Nat.mod_eq_of_lt (show n₁ < 2 ^ 32 by omega),
      Nat.mod_eq_of_lt (show n₁ * 2 ^ 32 < 2 ^ 64 by omega), Nat.or_comm, Nat.mul_comm n₁,
      ← Nat.two_pow_add_eq_or_of_lt (show n₀ < 2 ^ 32 by omega), Nat.mod_eq_of_lt (by omega)]
    omega

/-- A defined mbarrier operation: its read's word has a new word. -/
theorem mbar_ok {r : Req} {op : MBarOp} (hop : r.op = .mbar op) {u : UInt64} (ha : r.addrs = #[some u])
    {read : Nat → List UInt8} (hu : r.unsupported read = false) :
    ∃ w, MBarrierWord.apply op (mword (read 0)) = .ok w := by
  cases hw : MBarrierWord.apply op (mword (read 0)) with
  | ok w => exact ⟨w, rfl⟩
  | error e =>
    exfalso
    have hw' : MBarrierWord.apply op (UInt64.ofNat (leNat (read 0))) = .error e := hw
    simp [Req.unsupported, hop, MemOp.writes, ha, Req.active, Req.atomNew, hw'] at hu

/-! ## The full barrier -/

variable {n : Nat} {V : Type} [DecidableEq V] {x : Execution n V}

/-- `u` is `v` or coherence-before it. -/
def _root_.NV.PTX.Execution.coLe (x : Execution n V) (u v : Fin n) : Bool := u == v || x.co u v

/-- A full barrier's writes: its initialization `i` and, per round `r < R`, the producer's arrive
expecting `T` bytes (`aR r`, `aW r`) and `K` completes of `amt j` bytes each (`cR r j`, `cW r j`),
in a run where each operation is defined (`a_v`, `c_v`), on a word read as `word`. The producer
arrives round by round (`a_next`) and each complete comes after its round's arrive (`a_c`), in
coherence order: coherence gives both from causality. -/
structure Full (x : Execution n V) (word : V → UInt64) where
  R : Nat
  K : Nat
  T : Nat
  amt : Nat → Nat
  K_pos : 0 < K
  amt_pos : ∀ j < K, 0 < amt j
  amt_sum : sumBelow K amt = T
  T_lt : T < 2 ^ 20
  i : Fin n
  aR : Nat → Fin n
  aW : Nat → Fin n
  cR : Nat → Nat → Fin n
  cW : Nat → Nat → Fin n
  loc : OneLoc x i
  i_v : word (x.val i) = fresh 0
  /-- An arrive is undefined on the location's bytes before initialization. -/
  mem0 : ∀ w, MBarrierWord.apply (.arrive 1 T false) (word (x.init (x.loc (x.ev i).addr))) ≠ .ok w
  aW_w : ∀ r < R, x.writes (aW r) = true ∧ x.sameLocation (aW r) i = true
  cW_w : ∀ r < R, ∀ j < K, x.writes (cW r j) = true ∧ x.sameLocation (cW r j) i = true
  i_ne : ∀ r < R, aW r ≠ i ∧ ∀ j < K, cW r j ≠ i
  ac_ne : ∀ r < R, ∀ r' < R, ∀ j < K, aW r ≠ cW r' j
  all : ∀ v, x.writes v = true → x.sameLocation v i = true →
    v = i ∨ (∃ r < R, v = aW r) ∨ ∃ r < R, ∃ j < K, v = cW r j
  a_rmw : ∀ r < R, x.rmw (aR r) (aW r) = true ∧ x.proxyPreservedCauseBase (aR r) (aW r) = true ∧
    x.sameLocation (aR r) i = true
  c_rmw : ∀ r < R, ∀ j < K, x.rmw (cR r j) (cW r j) = true ∧
    x.proxyPreservedCauseBase (cR r j) (cW r j) = true ∧ x.sameLocation (cR r j) i = true
  a_next : ∀ r, r + 1 < R → x.co (aW r) (aW (r + 1)) = true
  a_c : ∀ r < R, ∀ j < K, x.co (aW r) (cW r j) = true
  c_inj : ∀ r < R, ∀ r' < R, ∀ j < K, ∀ j' < K, cW r j = cW r' j' → r = r' ∧ j = j'
  a_v : ∀ r < R, MBarrierWord.apply (.arrive 1 T false) (word (x.val (aR r))) = .ok (word (x.val (aW r)))
  c_v : ∀ r < R, ∀ j < K,
    MBarrierWord.apply (.completeTx (amt j)) (word (x.val (cR r j))) = .ok (word (x.val (cW r j)))

namespace Full

variable {word : V → UInt64} (F : Full x word)

/-- The bytes of round `r`'s completes at or before `v`. -/
def landed (r : Nat) (v : Fin n) : Nat := sumBelow F.K fun j => if x.coLe (F.cW r j) v then F.amt j else 0

/-- Where a write of the barrier is in the protocol: rounds before `ρ` are done, rounds after it
untouched, and round `ρ` either not begun (the word is `fresh ρ`) or armed and waiting for some of
its bytes (the word is `part ρ (T - landed)`). -/
def At (v : Fin n) (ρ : Nat) : Prop :=
  ρ ≤ F.R ∧
  (∀ r < ρ, x.coLe (F.aW r) v = true ∧ ∀ j < F.K, x.coLe (F.cW r j) v = true) ∧
  (∀ r, ρ < r → r < F.R → x.coLe (F.aW r) v = false ∧ ∀ j < F.K, x.coLe (F.cW r j) v = false) ∧
  ((∀ _ : ρ < F.R, x.coLe (F.aW ρ) v = false ∧ ∀ j < F.K, x.coLe (F.cW ρ j) v = false) ∧
      word (x.val v) = fresh ρ ∨
    ρ < F.R ∧ x.coLe (F.aW ρ) v = true ∧ (∃ j < F.K, x.coLe (F.cW ρ j) v = false) ∧
      word (x.val v) = part ρ (F.T - F.landed ρ v))

end Full

section
variable (hc : x.consistent = true)
include hc

theorem coLe_trans {a b c : Fin n} (h₁ : x.coLe a b = true) (h₂ : x.co b c = true) :
    x.co a c = true := by
  simp only [coLe, Bool.or_eq_true, beq_iff_eq] at h₁
  rcases h₁ with rfl | h
  · exact h₂
  · exact co_trans hc h h₂

theorem not_coLe {a b : Fin n} (h : x.co b a = true) : x.coLe a b = false := by
  cases h' : x.coLe a b
  · rfl
  · have := coLe_trans hc h' h; rw [co_irrefl hc] at this; cases this

/-- Below a write whose coherence predecessor is `p`, the writes at or before it are those at or
before `p`, and itself. -/
theorem coLe_mono {p v u : Fin n} (hpv : x.co p v = true) (h : x.coLe u p = true) :
    x.coLe u v = true := by
  simp only [coLe, Bool.or_eq_true, beq_iff_eq] at h ⊢
  rcases h with rfl | h
  · exact .inr hpv
  · exact .inr (co_trans hc h hpv)

theorem coLe_pred {p v : Fin n} (hpv : x.co p v = true)
    (hlast : ∀ u, x.co u v = true → u = p ∨ x.co u p = true) (u : Fin n) :
    x.coLe u v = (x.coLe u p || u == v) := by
  apply Bool.eq_iff_iff.2
  simp only [coLe, Bool.or_eq_true, beq_iff_eq]
  constructor
  · rintro (rfl | h)
    · exact .inr rfl
    · rcases hlast u h with rfl | h'
      · exact .inl (.inl rfl)
      · exact .inl (.inr h')
  · rintro ((rfl | h) | rfl)
    · exact .inr hpv
    · exact .inr (co_trans hc h hpv)
    · exact .inl rfl

end

theorem sumBelow_lt {K : Nat} {f g : Nat → Nat} (hle : ∀ j < K, f j ≤ g j) {j : Nat} (hj : j < K)
    (hlt : f j < g j) : sumBelow K f < sumBelow K g := by
  induction K with
  | zero => omega
  | succ K ih =>
    rw [sumBelow_succ, sumBelow_succ]
    by_cases hjK : j = K
    · subst hjK
      have := sumBelow_le (n := j) (f := f) (g := g) fun i hi => hle i (by omega)
      omega
    · have := ih (fun i hi => hle i (by omega)) (by omega)
      have := hle K (by omega)
      omega

namespace Full

variable {word : V → UInt64} (F : Full x word)

omit [DecidableEq V] in
theorem landed_le (r : Nat) (v : Fin n) : F.landed r v ≤ F.T := by
  rw [← F.amt_sum]
  exact sumBelow_le fun j _ => by split <;> omega

omit [DecidableEq V] in
theorem landed_eq_iff (r : Nat) (v : Fin n) :
    F.landed r v = F.T ↔ ∀ j < F.K, x.coLe (F.cW r j) v = true := by
  constructor
  · intro h j hj
    cases hd : x.coLe (F.cW r j) v
    · exfalso
      have : F.landed r v < sumBelow F.K F.amt :=
        sumBelow_lt (fun j _ => by split <;> omega) hj (by simp [hd, F.amt_pos j hj])
      rw [F.amt_sum] at this; omega
    · rfl
  · intro h
    unfold landed
    rw [← F.amt_sum]
    exact sumBelow_congr fun j hj => by simp [h j hj]

end Full

namespace Full

variable {word : V → UInt64} (F : Full x word) (hc : x.consistent = true)
include hc

omit [DecidableEq V] hc in
theorem T_pos : 0 < F.T := by
  have := le_sumBelow F.amt F.K_pos
  have := F.amt_pos 0 F.K_pos
  rw [F.amt_sum] at *; omega

theorem aW_lt {r r' : Nat} (h : r < r') (hr' : r' < F.R) : x.co (F.aW r) (F.aW r') = true := by
  induction r' with
  | zero => omega
  | succ r' ih =>
    rcases Nat.lt_succ_iff_lt_or_eq.1 h with h | rfl
    · exact co_trans hc (ih h (by omega)) (F.a_next r' hr')
    · exact F.a_next r hr'

/-- A write of the barrier other than its initialization has a coherence predecessor. -/
theorem pred_of {v : Fin n} (hv : x.writes v = true) (hvl : x.sameLocation v F.i = true)
    (hne : v ≠ F.i) :
    ∃ p, x.co p v = true ∧ (∀ u, x.co u v = true → u = p ∨ x.co u p = true) ∧
      x.writes p = true ∧ x.sameLocation p F.i = true := by
  obtain ⟨p, hpv, hlast⟩ := exists_pred hc F.loc hvl (F.loc.first v hv hvl hne)
  obtain ⟨hp, -, hpl⟩ := co_parts hc hpv
  exact ⟨p, hpv, hlast, hp, sameLocation_trans hpl hvl⟩

omit [DecidableEq V] hc in
theorem coLe_of_last {p v u : Fin n} (hlast : ∀ u, x.co u v = true → u = p ∨ x.co u p = true)
    (hu : x.co u v = true) : x.coLe u p = true := by
  rcases hlast u hu with rfl | h
  · simp [coLe]
  · simp [coLe, h]

/-- A write that is not `v` is at or before `v` when it is at or before `v`'s predecessor. -/
theorem coLe_ne {p v u : Fin n} (hpv : x.co p v = true)
    (hlast : ∀ u, x.co u v = true → u = p ∨ x.co u p = true) (hne : u ≠ v) :
    x.coLe u v = x.coLe u p := by
  rw [coLe_pred hc hpv hlast]; simp [hne]

/-- **The full barrier's shape**: every write of the barrier is somewhere in the protocol. -/
theorem shape : ∀ v, x.writes v = true → x.sameLocation v F.i = true → ∃ ρ, F.At v ρ := by
  intro v
  induction v using co_induction hc with
  | h v ih =>
  intro hv hvl
  have notAfter : ∀ {u w : Fin n}, x.co u w = true → x.coLe w u = false := fun h => not_coLe hc h
  rcases F.all v hv hvl with rfl | ⟨r, hr, rfl⟩ | ⟨r, hr, j, hj, rfl⟩
  · -- The initialization: nothing has happened.
    have untouched : ∀ r < F.R, x.coLe (F.aW r) F.i = false ∧ ∀ j < F.K, x.coLe (F.cW r j) F.i = false :=
      fun r hr => ⟨notAfter (F.loc.first _ (F.aW_w r hr).1 (F.aW_w r hr).2 (F.i_ne r hr).1),
        fun j hj => notAfter (F.loc.first _ (F.cW_w r hr j hj).1 (F.cW_w r hr j hj).2
          ((F.i_ne r hr).2 j hj))⟩
    exact ⟨0, Nat.zero_le _, fun r h => absurd h (Nat.not_lt_zero _),
      fun r _ hr => untouched r hr, .inl ⟨fun h0 => untouched 0 h0, F.i_v⟩⟩
  · -- An arrive: on a round not begun, the next one.
    obtain ⟨p, hpv, hlast, hp, hpl⟩ := F.pred_of hc hv hvl (F.i_ne r hr).1
    have hA := F.a_v r hr
    have hval : x.val (F.aR r) = x.val p := by
      rcases rmw_src hc F.loc (F.a_rmw r hr).1 (F.a_rmw r hr).2.1 hvl hpv hlast with h | ⟨-, h⟩
      · exact h
      · rw [h] at hA; exact absurd hA (F.mem0 _)
    rw [hval] at hA
    obtain ⟨ρ, hρ, hbefore, hafter, hphase⟩ := ih p hpv hp hpl
    rcases Nat.lt_trichotomy r ρ with hlt | rfl | hgt
    · exfalso
      have := coLe_trans hc (hbefore r hlt).1 hpv
      rw [co_irrefl hc] at this; cases this
    · rcases hphase with ⟨hfresh, hw⟩ | ⟨-, harm, -, -⟩
      · obtain ⟨-, hc0⟩ := hfresh hr
        rw [hw, arrive_fresh r F.T_pos F.T_lt] at hA
        have hval : word (x.val (F.aW r)) = part r F.T := (Except.ok.inj hA).symm
        have hne : ∀ r', r < r' → r' < F.R → F.aW r' ≠ F.aW r := fun r' h hr' e => by
          have := F.aW_lt hc h hr'; rw [e, co_irrefl hc] at this; cases this
        have hland : F.landed r (F.aW r) = 0 := by
          unfold landed
          rw [sumBelow_congr (g := fun _ => 0) fun j hj => by
            rw [coLe_ne hc hpv hlast (F.ac_ne r hr r hr j hj).symm, hc0 j hj]; rfl]
          exact (sumBelow_zero_iff _ _).2 fun _ _ => rfl
        refine ⟨r, hρ, fun r' h' => ⟨coLe_mono hc hpv (hbefore r' h').1,
            fun j hj => coLe_mono hc hpv ((hbefore r' h').2 j hj)⟩,
          fun r' h' hr' => ⟨?_, fun j hj => ?_⟩, .inr ⟨hr, by simp [coLe], ⟨0, F.K_pos, ?_⟩, ?_⟩⟩
        · rw [coLe_ne hc hpv hlast (hne r' h' hr')]; exact (hafter r' h' hr').1
        · rw [coLe_ne hc hpv hlast (F.ac_ne r hr r' hr' j hj).symm]; exact (hafter r' h' hr').2 j hj
        · rw [coLe_ne hc hpv hlast (F.ac_ne r hr r hr 0 F.K_pos).symm]; exact hc0 0 F.K_pos
        · rw [hland, Nat.sub_zero]; exact hval
      · exfalso
        have := coLe_trans hc harm hpv
        rw [co_irrefl hc] at this; cases this
    · exfalso
      have hρR : ρ < F.R := by omega
      rcases hphase with ⟨hfresh, -⟩ | ⟨-, -, ⟨j, hj, -⟩, hw⟩
      · have := coLe_of_last hlast (F.aW_lt hc hgt hr)
        rw [(hfresh hρR).1] at this; cases this
      · have hrem : F.T - F.landed ρ p < 2 ^ 20 := by have := F.T_lt; omega
        exact arrive_part ρ hrem _ (hw ▸ hA)
  · -- A complete: of the armed round, landing its bytes.
    obtain ⟨p, hpv, hlast, hp, hpl⟩ := F.pred_of hc hv hvl ((F.i_ne r hr).2 j hj)
    have harm : x.coLe (F.aW r) p = true := coLe_of_last hlast (F.a_c r hr j hj)
    have hC := F.c_v r hr j hj
    have hval : x.val (F.cR r j) = x.val p := by
      rcases rmw_src hc F.loc (F.c_rmw r hr j hj).1 (F.c_rmw r hr j hj).2.1 hvl hpv hlast with
        h | ⟨hpi, -⟩
      · exact h
      · exfalso
        subst hpi
        simp only [coLe, Bool.or_eq_true, beq_iff_eq] at harm
        rcases harm with h | h
        · exact (F.i_ne r hr).1 h
        · exact ne_first hc F.loc h rfl
    rw [hval] at hC
    obtain ⟨ρ, hρ, hbefore, hafter, hphase⟩ := ih p hpv hp hpl
    -- Writes other than this complete are at or before it as they are at or before `p`.
    have otherA : ∀ r' < F.R, x.coLe (F.aW r') (F.cW r j) = x.coLe (F.aW r') p :=
      fun r' hr' => coLe_ne hc hpv hlast (F.ac_ne r' hr' r hr j hj)
    have otherC : ∀ r' < F.R, ∀ j' < F.K, r' ≠ r ∨ j' ≠ j →
        x.coLe (F.cW r' j') (F.cW r j) = x.coLe (F.cW r' j') p := fun r' hr' j' hj' h =>
      coLe_ne hc hpv hlast fun e => by
        obtain ⟨h₁, h₂⟩ := F.c_inj r' hr' r hr j' hj' j hj e
        rcases h with h | h <;> contradiction
    rcases Nat.lt_trichotomy r ρ with hlt | rfl | hgt
    · exfalso
      have := coLe_trans hc ((hbefore r hlt).2 j hj) hpv
      rw [co_irrefl hc] at this; cases this
    · rcases hphase with ⟨hfresh, -⟩ | ⟨-, -, -, hw⟩
      · rw [(hfresh hr).1] at harm; cases harm
      have hj0 : x.coLe (F.cW r j) p = false := notAfter hpv
      have hjj : x.coLe (F.cW r j) (F.cW r j) = true := by simp [coLe]
      have hl : F.landed r (F.cW r j) = F.landed r p + F.amt j := by
        have := sumBelow_update (f := fun j' => if x.coLe (F.cW r j') p then F.amt j' else 0)
          (g := fun j' => if x.coLe (F.cW r j') (F.cW r j) then F.amt j' else 0) hj
          (fun j' hj' hne => by rw [otherC r hr j' hj' (.inr hne)])
        rw [hj0, hjj] at this
        simp only [Bool.false_eq_true, ite_false, ite_true, Nat.add_zero] at this
        unfold landed; omega
      have hle := F.landed_le r (F.cW r j)
      have hrem : F.T - F.landed r p < 2 ^ 20 := by have := F.T_lt; omega
      have after' : ∀ r', r < r' → r' < F.R →
          x.coLe (F.aW r') (F.cW r j) = false ∧ ∀ j' < F.K, x.coLe (F.cW r' j') (F.cW r j) = false :=
        fun r' h' hr' => ⟨by rw [otherA r' hr']; exact (hafter r' h' hr').1, fun j' hj' => by
          rw [otherC r' hr' j' hj' (.inl (by omega))]; exact (hafter r' h' hr').2 j' hj'⟩
      have before' : ∀ r' < r,
          x.coLe (F.aW r') (F.cW r j) = true ∧ ∀ j' < F.K, x.coLe (F.cW r' j') (F.cW r j) = true :=
        fun r' h' => ⟨coLe_mono hc hpv (hbefore r' h').1,
          fun j' hj' => coLe_mono hc hpv ((hbefore r' h').2 j' hj')⟩
      by_cases hdone : F.landed r (F.cW r j) = F.T
      · -- The round's last bytes: the phase completes.
        rw [hw, show F.T - F.landed r p = F.amt j by omega,
          complete_last r (show F.amt j < 2 ^ 20 by have := F.T_lt; omega)] at hC
        have hval : word (x.val (F.cW r j)) = fresh (r + 1) := (Except.ok.inj hC).symm
        have hall := (F.landed_eq_iff r _).1 hdone
        refine ⟨r + 1, hr, fun r' h' => ?_, fun r' h' hr' => after' r' (by omega) hr',
          .inl ⟨fun h1 => after' (r + 1) (by omega) h1, hval⟩⟩
        rcases Nat.lt_succ_iff_lt_or_eq.1 h' with h' | rfl
        · exact before' r' h'
        · exact ⟨coLe_mono hc hpv harm, hall⟩
      · -- Bytes still to come.
        rw [hw, complete_part r (show F.amt j < F.T - F.landed r p by omega) hrem] at hC
        have hval : word (x.val (F.cW r j)) = part r (F.T - F.landed r (F.cW r j)) := by
          rw [← Except.ok.inj hC, hl, Nat.sub_sub]
        obtain ⟨j', hj', hund⟩ : ∃ j' < F.K, x.coLe (F.cW r j') (F.cW r j) = false := by
          by_cases h : ∀ j' < F.K, x.coLe (F.cW r j') (F.cW r j) = true
          · exact absurd ((F.landed_eq_iff r _).2 h) hdone
          · simp only [Classical.not_forall, Bool.not_eq_true] at h
            obtain ⟨j', hj', h⟩ := h
            exact ⟨j', hj', h⟩
        exact ⟨r, hρ, before', after', .inr ⟨hr, coLe_mono hc hpv harm, ⟨j', hj', hund⟩, hval⟩⟩
    · exfalso
      rw [(hafter r hgt hr).1] at harm; cases harm

/-- A consumer's waits on the full barrier: wait `Rf r` returned true for the parity of round `r`
(`mbarrier.try_wait.parity`, an acquire read of the word), each after the consumer's wait of the
round before. A wait may read the location before its initialization, whose word reports phase 0
not done (`mem0`). -/
structure Waits (F : Full x word) where
  Rf : Nat → Fin n
  read : ∀ r < F.R, x.reads (Rf r) = true ∧ x.sameLocation (Rf r) F.i = true ∧
    (x.ev (Rf r)).acquire = true ∧ (x.ev (Rf r)).reduction = false ∧ (x.ev (Rf r)).scope ≠ .thread
  done : ∀ r < F.R, MBarrierWord.parityDone (word (x.val (Rf r))) r = true
  mem0 : MBarrierWord.parityDone (word (x.init (x.loc (x.ev F.i).addr))) 0 = false
  next : ∀ r, r + 1 < F.R → x.causeBase (Rf r) (Rf (r + 1)) = true
  release : ∀ r < F.R, ∀ j < F.K, (x.ev (F.cW r j)).release = true
  /-- The barrier's accesses are generic, so base causality between them is proxy-preserved. -/
  generic : ∀ a, x.sameLocation a F.i = true → x.generic a = true

/-- **The full barrier orders every copy of a round before the consumer's reads of it**: the
consumer's wait that returned true for round `r` synchronizes with every complete of round `r`. -/
theorem raw (W : Waits F) : ∀ r < F.R, ∀ j < F.K, x.sync (F.cW r j) (W.Rf r) = true := by
  intro r
  induction r using Nat.strongRecOn with
  | ind r ih =>
  intro hr j hj
  obtain ⟨hR, hRl, hacq, hnr, hRs⟩ := W.read r hr
  -- The wait reads a write: the initial word reports phase 0 not done, and a later round's wait
  -- follows a write of the barrier.
  have hcb : ∀ r', r' + 1 = r → ∀ j' < F.K,
      x.proxyPreservedCauseBase (F.cW r' j') (W.Rf r) = true := fun r' hr' j' hj' => by
    subst hr'
    have hwl := (F.cW_w r' (by omega) j' hj').2
    exact ppcb_of_generic (W.generic _ hwl) (W.generic _ hRl)
      ((F.loc.addr _ hwl).trans (F.loc.addr _ hRl).symm)
      (causeBase_trans (causeBase_of_sync (ih r' (by omega) (by omega) j' hj')) (W.next r' hr))
  obtain ⟨v, hrf⟩ : ∃ v, x.rf v (W.Rf r) = true := by
    rcases Nat.eq_zero_or_pos r with rfl | hpos
    · rcases rf_cases hc hR with ⟨-, hv0⟩ | ⟨v, hrf, -⟩
      · exfalso
        have := W.done 0 hr
        rw [hv0, show x.loc (x.ev (W.Rf 0)).addr = x.loc (x.ev F.i).addr from
          (sameLocation_iff.1 hRl).2.2, W.mem0] at this
        cases this
      · exact ⟨v, hrf⟩
    · have hwl := (F.cW_w (r - 1) (by omega) 0 F.K_pos).2
      exact (read_sees_ppcb hc (F.cW_w (r - 1) (by omega) 0 F.K_pos).1 hR
        (mems_of_writes (F.cW_w (r - 1) (by omega) 0 F.K_pos).1) (sameLocation_iff.1 hRl).1
        ((F.loc.addr _ hwl).trans (F.loc.addr _ hRl).symm) (hcb (r - 1) (by omega) 0 F.K_pos)).1
  obtain ⟨hv, -, hvR⟩ := rf_parts hc hrf
  have hvl := sameLocation_trans hvR hRl
  obtain ⟨ρ, hρ, hbefore, hafter, hphase⟩ := F.shape hc v hv hvl
  -- The consumer's earlier wait puts the round before at or below `v`.
  have hprev : ∀ r', r' + 1 = r → ∀ j' < F.K, x.coLe (F.cW r' j') v = true := fun r' hr' j' hj' => by
    have hr'' : r' < F.R := by omega
    have hwl := (F.cW_w r' hr'' j' hj').2
    obtain ⟨-, hnot⟩ := read_sees_ppcb hc (F.cW_w r' hr'' j' hj').1 hR
      (mems_of_writes (F.cW_w r' hr'' j' hj').1) (sameLocation_iff.1 hRl).1
      ((F.loc.addr _ hwl).trans (F.loc.addr _ hRl).symm) (hcb r' hr' j' hj')
    have := hnot v hrf
    by_cases he : F.cW r' j' = v
    · simp [coLe, he]
    · rcases co_total_loc hc F.loc he (F.cW_w r' (by omega) j' hj').1 hv hwl hvl with h | h
      · simp [coLe, h]
      · rw [h] at this; cases this
  have hval : MBarrierWord.parityDone (word (x.val v)) r = true := by
    rw [← rf_val hc hrf]; exact W.done r hr
  -- Round `r` is done at `v`: before it the word reports the phase not done.
  have hρr : r < ρ := by
    rcases Nat.lt_or_ge r ρ with h | h
    · exact h
    exfalso
    have hge : r ≤ ρ := by
      rcases Nat.eq_zero_or_pos r with rfl | hpos
      · exact Nat.zero_le _
      · apply Classical.byContradiction; intro hlt
        have hlt : ρ < r - 1 ∨ ρ = r - 1 := by omega
        have hd := hprev (r - 1) (by omega) 0 F.K_pos
        rcases hlt with hlt | heq
        · rw [(hafter (r - 1) hlt (by omega)).2 0 F.K_pos] at hd; cases hd
        · subst heq
          rcases hphase with ⟨hfresh, -⟩ | ⟨-, -, ⟨j', hj', hund⟩, -⟩
          · rw [(hfresh (by omega)).2 0 F.K_pos] at hd; cases hd
          · rw [hprev (r - 1) (by omega) j' hj'] at hund; cases hund
    have : ρ = r := by omega
    subst this
    rcases hphase with ⟨-, hw⟩ | ⟨-, -, -, hw⟩
    · rw [hw, done_fresh] at hval; simp at hval
    · rw [hw, done_part _ _ (by have := F.T_lt; omega)] at hval; simp at hval
  have hle := (hbefore r hρr).2 j hj
  have hcwl := (F.cW_w r hr j hj).2
  refine sync_of_rmws hc F.loc (F.cW_w r hr j hj).1 hcwl ((F.i_ne r hr).2 j hj) (W.release r hr j hj)
    hrf hRl hRs hacq hnr
    (by simp only [coLe, Bool.or_eq_true, beq_iff_eq] at hle; exact hle) fun w hw _ => ?_
  obtain ⟨-, hww, hwl⟩ := co_parts hc hw
  have hwl := sameLocation_trans (sameLocation_symm hwl) hcwl
  rcases F.all w hww hwl with rfl | ⟨r', hr', rfl⟩ | ⟨r', hr', j', hj', rfl⟩
  · exfalso
    have := co_trans hc (F.loc.first _ (F.cW_w r hr j hj).1 hcwl ((F.i_ne r hr).2 j hj)) hw
    rw [co_irrefl hc] at this; cases this
  · exact ⟨_, (F.a_rmw r' hr').1, (F.a_rmw r' hr').2.1, (F.a_rmw r' hr').2.2⟩
  · exact ⟨_, (F.c_rmw r' hr' j' hj').1, (F.c_rmw r' hr' j' hj').2.1, (F.c_rmw r' hr' j' hj').2.2⟩

end Full

/-! ## The empty barrier -/

/-- An empty barrier's writes: its initialization `i` and, per round `r < R`, the consumer's
arrive (`eR r`, `eW r`), one after another in coherence order, in a run where each is defined. -/
structure Empty (x : Execution n V) (word : V → UInt64) where
  R : Nat
  i : Fin n
  eR : Nat → Fin n
  eW : Nat → Fin n
  loc : OneLoc x i
  i_v : word (x.val i) = fresh 0
  /-- An arrive is undefined on the location's bytes before initialization. -/
  mem0 : ∀ w, MBarrierWord.apply (.arrive 1 0 false) (word (x.init (x.loc (x.ev i).addr))) ≠ .ok w
  eW_w : ∀ r < R, x.writes (eW r) = true ∧ x.sameLocation (eW r) i = true
  i_ne : ∀ r < R, eW r ≠ i
  all : ∀ v, x.writes v = true → x.sameLocation v i = true → v = i ∨ ∃ r < R, v = eW r
  e_rmw : ∀ r < R, x.rmw (eR r) (eW r) = true ∧ x.proxyPreservedCauseBase (eR r) (eW r) = true ∧
    x.sameLocation (eR r) i = true
  e_next : ∀ r, r + 1 < R → x.co (eW r) (eW (r + 1)) = true
  e_v : ∀ r < R, MBarrierWord.apply (.arrive 1 0 false) (word (x.val (eR r))) = .ok (word (x.val (eW r)))

namespace Empty

variable {word : V → UInt64} (E : Empty x word) (hc : x.consistent = true)
include hc

theorem eW_lt {r r' : Nat} (h : r < r') (hr' : r' < E.R) : x.co (E.eW r) (E.eW r') = true := by
  induction r' with
  | zero => omega
  | succ r' ih =>
    rcases Nat.lt_succ_iff_lt_or_eq.1 h with h | rfl
    · exact co_trans hc (ih h (by omega)) (E.e_next r' hr')
    · exact E.e_next r hr'

/-- Where a write of the barrier is: the first `ρ` arrives are at or before it, and its word is
`fresh ρ`. -/
def At (v : Fin n) (ρ : Nat) : Prop :=
  ρ ≤ E.R ∧ (∀ r < E.R, x.coLe (E.eW r) v = decide (r < ρ)) ∧ word (x.val v) = fresh ρ

/-- **The empty barrier's shape**: its writes are its arrives in order. -/
theorem shape : ∀ v, x.writes v = true → x.sameLocation v E.i = true → ∃ ρ, E.At v ρ := by
  intro v
  induction v using co_induction hc with
  | h v ih =>
  intro hv hvl
  rcases E.all v hv hvl with rfl | ⟨r, hr, rfl⟩
  · refine ⟨0, Nat.zero_le _, fun r hr => ?_, E.i_v⟩
    simpa using not_coLe hc (E.loc.first _ (E.eW_w r hr).1 (E.eW_w r hr).2 (E.i_ne r hr))
  · obtain ⟨p, hpv, hlast⟩ := exists_pred hc E.loc hvl (E.loc.first _ hv hvl (E.i_ne r hr))
    obtain ⟨hp, -, hpl⟩ := co_parts hc hpv
    have hpl := sameLocation_trans hpl hvl
    have hA := E.e_v r hr
    have hval : x.val (E.eR r) = x.val p := by
      rcases rmw_src hc E.loc (E.e_rmw r hr).1 (E.e_rmw r hr).2.1 hvl hpv hlast with h | ⟨-, h⟩
      · exact h
      · rw [h] at hA; exact absurd hA (E.mem0 _)
    rw [hval] at hA
    obtain ⟨ρ, hρ, hdone, hw⟩ := ih p hpv hp hpl
    rw [hw, arrive_empty] at hA
    have hne : ∀ r', r' < E.R → r' ≠ r → E.eW r' ≠ E.eW r := fun r' hr' hne e => by
      rcases Nat.lt_or_gt_of_ne hne with h | h
      · have := E.eW_lt hc h hr; rw [e, co_irrefl hc] at this; cases this
      · have := E.eW_lt hc h hr'; rw [e, co_irrefl hc] at this; cases this
    have hrρ : r = ρ := by
      rcases Nat.lt_trichotomy r ρ with h | h | h
      · exfalso
        have := hdone r hr
        simp only [h, decide_true] at this
        have := coLe_trans hc this hpv
        rw [co_irrefl hc] at this; cases this
      · exact h
      · exfalso
        have := hdone ρ (by omega)
        rw [Full.coLe_of_last hlast (E.eW_lt hc h hr)] at this
        simp at this
    subst hrρ
    refine ⟨r + 1, hr, fun r' hr' => ?_, (Except.ok.inj hA).symm⟩
    by_cases he : r' = r
    · subst he; simp [coLe]
    · rw [Full.coLe_ne hc hpv hlast (hne r' hr' he), hdone r' hr']
      simp only [decide_eq_decide]; omega

/-- A producer's waits on the empty barrier: wait `Pe r` (`r < M`) returned true for the parity of
round `r`, waiting for the consumer's arrive of round `r`. -/
structure Waits (E : Empty x word) where
  M : Nat
  M_le : M ≤ E.R
  Pe : Nat → Fin n
  read : ∀ r < M, x.reads (Pe r) = true ∧ x.sameLocation (Pe r) E.i = true ∧
    (x.ev (Pe r)).acquire = true ∧ (x.ev (Pe r)).reduction = false ∧ (x.ev (Pe r)).scope ≠ .thread
  done : ∀ r < M, MBarrierWord.parityDone (word (x.val (Pe r))) r = true
  init : ∀ r < M, x.proxyPreservedCauseBase E.i (Pe r) = true
  next : ∀ r, r + 1 < M → x.causeBase (Pe r) (Pe (r + 1)) = true
  release : ∀ r < E.R, (x.ev (E.eW r)).release = true
  generic : ∀ a, x.sameLocation a E.i = true → x.generic a = true

/-- **The empty barrier orders a consumer's reads of a round before the producer's reuse**: the
producer's wait that returned true for round `r` synchronizes with the consumer's arrive of round
`r`. -/
theorem war (W : Waits E) : ∀ r < W.M, x.sync (E.eW r) (W.Pe r) = true := by
  intro r
  induction r using Nat.strongRecOn with
  | ind r ih =>
  intro hr'
  have hr : r < E.R := Nat.lt_of_lt_of_le hr' W.M_le
  obtain ⟨hR, hRl, hacq, hnr, hRs⟩ := W.read r hr'
  obtain ⟨⟨v, hrf⟩, -⟩ := read_sees_ppcb hc E.loc.write hR (mems_of_writes E.loc.write)
    (sameLocation_iff.1 hRl).1 (E.loc.addr _ hRl).symm (W.init r hr')
  obtain ⟨hv, -, hvR⟩ := rf_parts hc hrf
  have hvl := sameLocation_trans hvR hRl
  obtain ⟨ρ, hρ, hdone, hw⟩ := E.shape hc v hv hvl
  have hge : r ≤ ρ := by
    rcases Nat.eq_zero_or_pos r with rfl | hpos
    · exact Nat.zero_le _
    have hcb : x.causeBase (E.eW (r - 1)) (W.Pe r) = true :=
      causeBase_trans (causeBase_of_sync (ih (r - 1) (by omega) (by omega)))
        (by have := W.next (r - 1) (by omega); rwa [show r - 1 + 1 = r by omega] at this)
    have hwl := (E.eW_w (r - 1) (by omega)).2
    have hp := ppcb_of_generic (W.generic _ hwl) (W.generic _ hRl)
      ((E.loc.addr _ hwl).trans (E.loc.addr _ hRl).symm) hcb
    obtain ⟨-, hnot⟩ := read_sees_ppcb hc (E.eW_w (r - 1) (by omega)).1 hR
      (mems_of_writes (E.eW_w (r - 1) (by omega)).1) (sameLocation_iff.1 hRl).1
      ((E.loc.addr _ hwl).trans (E.loc.addr _ hRl).symm) hp
    have hco := hnot v hrf
    have hle : x.coLe (E.eW (r - 1)) v = true := by
      by_cases he : E.eW (r - 1) = v
      · simp [coLe, he]
      · rcases co_total_loc hc E.loc he (E.eW_w (r - 1) (by omega)).1 hv hwl hvl with h | h
        · simp [coLe, h]
        · rw [h] at hco; cases hco
    rw [hdone (r - 1) (by omega)] at hle
    simp at hle; omega
  have hval : MBarrierWord.parityDone (word (x.val v)) r = true := by
    rw [← rf_val hc hrf]; exact W.done r hr'
  have hlt : r < ρ := by
    rcases Nat.lt_or_ge r ρ with h | h
    · exact h
    · have : ρ = r := by omega
      subst this
      rw [hw, done_fresh] at hval; simp at hval
  have hle := hdone r hr
  simp only [hlt, decide_true] at hle
  have hwl := (E.eW_w r hr).2
  refine sync_of_rmws hc E.loc (E.eW_w r hr).1 hwl (E.i_ne r hr) (W.release r hr) hrf hRl hRs hacq hnr
    (by simp only [coLe, Bool.or_eq_true, beq_iff_eq] at hle; exact hle) fun w hw' _ => ?_
  obtain ⟨-, hww, hwl'⟩ := co_parts hc hw'
  have hwl' := sameLocation_trans (sameLocation_symm hwl') hwl
  rcases E.all w hww hwl' with rfl | ⟨r', hr', rfl⟩
  · exfalso
    have := co_trans hc (E.loc.first _ (E.eW_w r hr).1 hwl (E.i_ne r hr)) hw'
    rw [co_irrefl hc] at this; cases this
  · exact ⟨_, (E.e_rmw r' hr').1, (E.e_rmw r' hr').2.1, (E.e_rmw r' hr').2.2⟩

end Empty

end NV.Formal.Pipeline
