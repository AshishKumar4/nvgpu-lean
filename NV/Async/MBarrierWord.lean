import NV.PTX.Async.MBarrier
import NV.Machine.Access

/-!
# The mbarrier object as a word

An mbarrier is an opaque 8-byte shared object (PTX ISA 9.5 §9.7.15.16.1). The model gives it its
own encoding, so that its operations are read-modify-writes of one location (`MemOp.mbar`) and
mbarrier synchronization follows from the memory model: arrivals are release RMWs, a wait that
returns true is an acquire load that reads the phase-completing write.

Layout (ours, `.layout::v0`): bits 0–19 the pending arrival count, 20–39 the expected count,
40–60 the tx-count as a 21-bit two's complement number, 63 the parity of the current phase. The
counts' ranges (Table 43: up to `2²⁰ - 1`, tx-count down to `-(2²⁰ - 1)`) fit their fields. The
word keeps the phase's parity only, as the hardware exposes; `NV.PTX.MBarrier`'s full phase
count and its `observed` ghost are the protocol model's, and a decoded word starts a fresh count.
-/

namespace NV.Async

open NV.PTX (MBarrier)

namespace MBarrierWord

def field (w : UInt64) (lo width : Nat) : Nat := (w >>> lo.toUInt64).toNat % 2 ^ width

/-- The word of a barrier state. -/
def encode (b : MBarrier) : UInt64 :=
  let tx : Nat := if b.tx < 0 then (2 ^ 21 + b.tx).toNat else b.tx.toNat
  let f (v lo : Nat) : UInt64 := v.toUInt64 <<< lo.toUInt64
  f (b.pending % 2 ^ 20) 0 ||| f (b.expected % 2 ^ 20) 20 ||| f (tx % 2 ^ 21) 40 ||| f (b.phase % 2) 63

/-- The barrier state a word holds, its phase count being the parity. -/
def decode (w : UInt64) : MBarrier :=
  let t := field w 40 21
  { phase := field w 63 1, pending := field w 0 20, expected := field w 20 20,
    tx := if t ≥ 2 ^ 20 then (t : Int) - 2 ^ 21 else t, observed := true }

def phaseParity (w : UInt64) : Nat := field w 63 1

/-- `mbarrier.test_wait/try_wait.parity`: whether the phase with parity `parity` has completed,
read from the word (`MBarrier.parityDone`). -/
def parityDone (w : UInt64) (parity : Nat) : Bool := (decode w).parityDone parity

/-- `mbarrier.test_wait/try_wait` with the `state` an earlier arrive returned: whether the phase
that state was taken in has completed. -/
def stateDone (w state : UInt64) : Bool := phaseParity w != phaseParity state

/-- `mbarrier.pending_count` of a state. -/
def pendingCount (state : UInt64) : Nat := field state 0 20

def faultMsg : MBarrier.Fault → String
  | .initCount => "mbarrier.init: count outside [1, 2^20 - 1]"
  | .overArrival => "mbarrier: more arrivals than pending (undefined)"
  | .txRange => "mbarrier: tx-count outside [-(2^20 - 1), 2^20 - 1] (undefined)"
  | .noComplete => "mbarrier.arrive.noComplete completed the phase (undefined)"
  | .staleParity => "mbarrier: parity wait on a phase that is neither current nor previous"
  | .unobserved => "mbarrier: arrive-on before the previous phase was observed complete"

/-- The new word after `op` on `old`. Behaviour the ISA leaves undefined is an error. -/
def apply (op : MBarOp) (old : UInt64) : Except String UInt64 :=
  let b := decode old
  let r : Except MBarrier.Fault MBarrier := match op with
    | .init n p => if p % 2 = 0 then MBarrier.init n else (MBarrier.init n).map ({ · with phase := 1 })
    | .arrive n tx drop => do
      let b ← b.expectTx tx
      let b := if drop then { b with expected := b.expected - n } else b
      if n == 0 then pure b else b.arrive n
    | .completeTx bytes => b.completeTx bytes
    | .inval => pure { phase := 0, pending := 0, expected := 0, tx := 0 }
  match r with
  | .ok b => .ok (encode b)
  | .error f => .error (faultMsg f)

/-! ## Round trip -/

theorem toNat_shift {v lo : Nat} (h : v * 2 ^ lo < 2 ^ 64) (hlo : lo < 64) (hv : v < 2 ^ 64) :
    (v.toUInt64 <<< lo.toUInt64).toNat = v * 2 ^ lo := by
  have h₁ : v.toUInt64.toNat = v := by
    show (UInt64.ofNat v).toNat = v; rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt hv]
  have h₂ : lo.toUInt64.toNat = lo := by
    show (UInt64.ofNat lo).toNat = lo; rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]
  rw [UInt64.toNat_shiftLeft, h₁, h₂, Nat.mod_eq_of_lt hlo, Nat.shiftLeft_eq, Nat.mod_eq_of_lt h]

theorem or_fields {a b c d : Nat} (ha : a < 2 ^ 20) (hb : b < 2 ^ 20) (hc : c < 2 ^ 21) (hd : d < 2) :
    a * 2 ^ 0 ||| b * 2 ^ 20 ||| c * 2 ^ 40 ||| d * 2 ^ 63 =
      a + 2 ^ 20 * b + 2 ^ 40 * c + 2 ^ 63 * d := by
  rw [Nat.pow_zero, Nat.mul_one, Nat.mul_comm b, Nat.mul_comm c, Nat.mul_comm d,
    Nat.or_comm a, ← Nat.two_pow_add_eq_or_of_lt ha,
    Nat.or_comm _ (2 ^ 40 * c), ← Nat.two_pow_add_eq_or_of_lt (by omega),
    Nat.or_comm _ (2 ^ 63 * d), ← Nat.two_pow_add_eq_or_of_lt (by omega)]
  omega

/-- The word of a state, as a number: its fields side by side. -/
theorem toNat_encode (b : MBarrier) :
    (encode b).toNat = b.pending % 2 ^ 20 + 2 ^ 20 * (b.expected % 2 ^ 20) +
      2 ^ 40 * ((if b.tx < 0 then (2 ^ 21 + b.tx).toNat else b.tx.toNat) % 2 ^ 21) +
      2 ^ 63 * (b.phase % 2) := by
  simp only [encode, UInt64.toNat_or]
  generalize (if b.tx < 0 then (2 ^ 21 + b.tx).toNat else b.tx.toNat) = t
  have hp : b.pending % 2 ^ 20 < 2 ^ 20 := Nat.mod_lt _ (by decide)
  have he : b.expected % 2 ^ 20 < 2 ^ 20 := Nat.mod_lt _ (by decide)
  have htt : t % 2 ^ 21 < 2 ^ 21 := Nat.mod_lt _ (by decide)
  have hph : b.phase % 2 < 2 := Nat.mod_lt _ (by decide)
  rw [toNat_shift (v := b.pending % 2 ^ 20) (lo := 0) (by omega) (by decide) (by omega),
    toNat_shift (v := b.expected % 2 ^ 20) (lo := 20) (by omega) (by decide) (by omega),
    toNat_shift (v := t % 2 ^ 21) (lo := 40) (by omega) (by decide) (by omega),
    toNat_shift (v := b.phase % 2) (lo := 63) (by omega) (by decide) (by omega)]
  exact or_fields hp he htt hph

/-- **A state decodes from its word**, its phase count taken mod 2, when its counts fit their
fields. -/
theorem decode_encode (b : MBarrier) (hp : b.pending < 2 ^ 20) (he : b.expected < 2 ^ 20)
    (ht₁ : -(2 ^ 20 : Int) ≤ b.tx) (ht₂ : b.tx < 2 ^ 20) :
    decode (encode b) = { b with phase := b.phase % 2, observed := true } := by
  have hn := toNat_encode b
  obtain ⟨ph, pe, ex, tx, ob⟩ := b
  simp only at hp he ht₁ ht₂ hn ⊢
  have h40 : (40 : Nat).toUInt64.toNat % 64 = 40 := rfl
  have h20 : (20 : Nat).toUInt64.toNat % 64 = 20 := rfl
  have h63 : (63 : Nat).toUInt64.toNat % 64 = 63 := rfl
  have h0 : (0 : Nat).toUInt64.toNat % 64 = 0 := rfl
  simp only [decode, field, UInt64.toNat_shiftRight, Nat.shiftRight_eq_div_pow, hn, h40, h20, h63,
    h0, MBarrier.mk.injEq, and_true]
  refine ⟨by omega, by omega, by omega, ?_⟩
  split <;> split at * <;> omega

end MBarrierWord

end NV.Async
