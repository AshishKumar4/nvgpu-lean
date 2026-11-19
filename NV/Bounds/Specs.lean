import NV.Bounds.Bound
import NV.Bounds.Pebble
import NV.Bounds.Attention

/-!
# Bounds of specs

`bound spec gpu`: per resource, the work every program in a stated class must put on it, at the
part's capacities. Each term names its class and the theorem that forces the work:

* compulsory traffic (`compulsory_traffic`, `compulsory_write_traffic`): every program. What it
  assumes of the run: every input byte is depended on by an output at the run's data (true of
  generic data; a zero that cancels a product is not generic), and a fresh machine, whose empty L2
  sends a byte's first read to DRAM. Every global store, and every read that misses L1, passes an
  L2 slice.
* multiply-adds: the classical class (every product `a_ik · b_kj` formed; Strassen-like
  algorithms are outside it), at the rate of every unit that forms them (`Part.macs`).
* I/O of classical matrix multiplication (`Pebble.mmm_io_parallel`): the SMs as processors with
  their own fast memories (shared memory, L1 and registers), the L2 and the other SMs as slow
  memory; and the whole chip's storage as one fast memory against DRAM (`Pebble.mmm_io`). Fast
  memory is counted in words of the smallest element, so the capacity is over-counted and the
  bound weakened, never strengthened.
* exponentials: the classical attention class (every score exponentiated on the MUFU pipe).
-/

namespace NV.Bounds.Specs

open NV NV.Core NV.Bounds

/-! ## The classical-matmul I/O bound as a number -/

/-- Segment lengths the bound tries; each gives a valid bound, the best is kept. -/
def segments (S : Nat) : List Nat := [S, 2 * S, 4 * S, 8 * S]

/-- Words `P` processors with `S` words of fast memory each must move to form every product of
an `M × N × K` matmul: the best over segment lengths `T` of `⌈T · MNK / U(S, T)⌉ - P · T`
(`Pebble.mmm_io_parallel`). -/
def pebbleWords (M N K S P : Nat) : Nat :=
  (segments S).foldl (fun best T =>
    if T = 0 ∨ Pebble.U S T = 0 then best
    else max best (cdiv (T * (M * (N * K))) (Pebble.U S T) - P * T)) 0

theorem pebbleWords_le {M N K S P Q : Nat}
    (h : ∀ T, 0 < T → T * (M * (N * K)) ≤ Pebble.U S T * (Q + P * T)) :
    pebbleWords M N K S P ≤ Q := by
  unfold pebbleWords
  suffices ∀ (l : List Nat) (m : Nat), m ≤ Q → l.foldl (fun best T =>
      if T = 0 ∨ Pebble.U S T = 0 then best
      else max best (cdiv (T * (M * (N * K))) (Pebble.U S T) - P * T)) m ≤ Q from
    this _ 0 (Nat.zero_le _)
  intro l
  induction l with
  | nil => intro m hm; exact hm
  | cons T rest ih =>
    intro m hm
    simp only [List.foldl_cons]
    apply ih
    split
    · exact hm
    · next hT =>
      have hT0 : 0 < T := by omega
      have hu : 0 < Pebble.U S T := by omega
      have := cdiv_le hu (h T hT0)
      exact Nat.max_le.2 ⟨hm, by omega⟩

/-! ## Resources the specs use beyond `Part` -/

/-- Data reaching an SM from outside it: its crossbar port from the L2 and its port from the
SM-to-SM network. -/
def smInbound (p : Params) : Resource :=
  let c := Memory.Config.ofParams p 0
  { name := "SM inbound (crossbar + DSMEM)"
    unit := "bytes"
    capacity := { instances := p.sms.value,
                  rate := { units := c.xbar.units * c.dsmem.cycles + c.dsmem.units * c.xbar.cycles,
                            cycles := c.xbar.cycles * c.dsmem.cycles } }
    facts := [.of "mem.xbarBytesPerCycle" p.mem.xbarBytesPerCycle,
      .of "mem.dsmemBytesPerCycle" p.mem.dsmemBytesPerCycle, .of "sms" p.sms]
    note := "" }

/-- Bytes of fast memory on one SM: the combined L1/shared array and the register file. -/
def smFastBytes (p : Params) : Nat := p.mem.l1SmemBytes.value + 4 * p.regsPerSM.value

/-- Bytes of storage on the chip: the L2 and every SM's fast memory. -/
def chipFastBytes (p : Params) : Nat := p.mem.l2Bytes.value + p.sms.value * smFastBytes p

def capacityFacts (p : Params) : List FactRef :=
  [.of "mem.l1SmemBytes" p.mem.l1SmemBytes, .of "regsPerSM" p.regsPerSM, .of "mem.l2Bytes" p.mem.l2Bytes]

/-! ## Specs -/

/-- `y ← a·x + y` over `n` elements of `bytes` bytes. -/
def axpy (p : Params) (n bytes : Nat) : Bound where
  spec := s!"axpy, {n} elements of {bytes} bytes"
  gpu := p.name
  terms := [
    (Part.dram p).term (2 * n * bytes) "every program, fresh machine"
      "compulsory_traffic: each byte of x and y is depended on by an output (assumed of the run's \
      data: a odd, and a·x[i] with no zero byte); an empty L2 sends each first read to DRAM",
    (Part.l2 p).term (3 * n * bytes) "every program, fresh machine"
      "compulsory_traffic for x and y, compulsory_write_traffic for the y bytes that change; every \
      L1 miss and every store passes an L2 slice"]

structure Gemm where
  name : String
  m : Nat
  n : Nat
  k : Nat
  /-- Bytes per element of `A` and `B`, and of `C`. -/
  inBytes : Nat
  outBytes : Nat
  /-- The tensor cores' dense rate for the input type. -/
  tensor : Fact Nat

/-- `C = A · B`, `A` of `m × k`, `B` of `k × n`. -/
def gemm (p : Params) (g : Gemm) : Bound :=
  let words := g.m * (g.n * g.k)
  let sm := smFastBytes p / max g.inBytes 1
  let chip := chipFastBytes p / max g.inBytes 1
  { spec := s!"GEMM {g.name} {g.m}×{g.n}×{g.k}, {g.inBytes}-byte inputs, {g.outBytes}-byte output"
    gpu := p.name
    terms := [
      (Part.dram p).term ((g.m * g.k + g.k * g.n) * g.inBytes) "every program, fresh machine"
        "compulsory_traffic: every byte of A and B is depended on by an output (assumed of the run's \
        data); an empty L2 sends each first read to DRAM",
      (Part.l2 p).term ((g.m * g.k + g.k * g.n) * g.inBytes + g.m * g.n * g.outBytes)
        "every program, fresh machine"
        "compulsory_traffic for A and B, compulsory_write_traffic for C; every L1 miss and every \
        store passes an L2 slice",
      (Part.macs p g.name g.tensor).term words "classical algorithms (every product formed)"
        "each of the m·n·k products a_ik·b_kj is formed by some unit",
      { (smInbound p).term (pebbleWords g.m g.n g.k sm p.sms.value * g.inBytes)
          "classical algorithms, no partial sum discarded (NV.Bounds.Pebble)"
          s!"mmm_io_parallel with {p.sms.value} SMs of {sm} words each (L1/shared and registers \
          over {g.inBytes}-byte words); every word moved is at least {g.inBytes} bytes" with
        facts := (smInbound p).facts ++ capacityFacts p },
      { (Part.dram p).term (pebbleWords g.m g.n g.k chip 1 * g.inBytes)
          "classical algorithms, no partial sum discarded (NV.Bounds.Pebble)"
          s!"mmm_io with the chip's {chip} words (L2 and every SM) as one fast memory against DRAM"
        with resource := "DRAM, classical-matmul I/O", facts := (Part.dram p).facts ++ capacityFacts p }] }

structure Attn where
  name : String
  /-- Batch × heads. -/
  heads : Nat
  seq : Nat
  dim : Nat
  bytes : Nat
  causal : Bool
  tensor : Fact Nat

/-- Exact attention `O = softmax(Q Kᵀ) V` per head, `Q, K, V, O` of `seq × dim`. -/
def attention (p : Params) (a : Attn) : Bound :=
  let io := a.heads * a.seq * a.dim * a.bytes
  -- Scores formed: all of them, or the lower triangle with the diagonal.
  let scores := a.heads * (if a.causal then a.seq * (a.seq + 1) / 2 else a.seq * a.seq)
  { spec := s!"attention {a.name}: {a.heads} heads, {a.seq} × {a.dim}, {a.bytes}-byte elements{if a.causal then ", causal" else ""}"
    gpu := p.name
    terms := [
      (Part.dram p).term (3 * io) "every program, fresh machine"
        "compulsory_traffic: every byte of Q, K and V is depended on by an output (assumed of the \
        run's data); an empty L2 sends each first read to DRAM. FlashAttention's Proposition 3 \
        (Attention.prop3) adds nothing at a fixed SRAM size",
      (Part.l2 p).term (4 * io) "every program, fresh machine"
        "compulsory_traffic for Q, K, V; compulsory_write_traffic for O",
      (Part.macs p a.name a.tensor).term (2 * scores * a.dim) "classical attention (scores and P·V as dot products)"
        "each score is a dot product of dim multiply-adds, and each output a sum of dim-wide products",
      (Part.pipeOf p "MUFU").term (cdiv scores 32) "classical attention, exponentials on MUFU"
        "each score is exponentiated once, 32 lanes per MUFU warp instruction"] }

end NV.Bounds.Specs
