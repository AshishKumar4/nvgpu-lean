# Numerics (`NV.Num`)

The floating-point and tensor-core arithmetic shared by the PTX and SASS semantics. Values are bit
patterns (`Nat`); every operation is exact integer arithmetic followed by one rounding step. No
host floating point is used.

| module | contents |
|---|---|
| `Format` | the formats: f64, f32, f16, bf16, tf32, e4m3, e5m2, e2m3, e3m2, e2m1, ue8m0, ue4m3, ue5m3; decoding to exact values |
| `Round` | the rounding step (`rn rna rz rm rp`, stochastic), overflow policies (IEEE, `.satfinite`, tensor-core infinity), flush of subnormal results |
| `Arith` | add, sub, mul, fma, div, rcp, sqrt, min/max (`.NaN`, `.xorsign.abs`), abs, neg, copysign, testp, setp comparisons, `.sat`, `.relu` |
| `Convert` | `cvt`'s conversions, one element at a time: float↔float (`.ftz .sat .relu .satfinite .pzo`), with UE8M0 scale factors (`.scaled::n1/n2::ue8m0`), `.rs`, float→int, int→float, integral rounding, to and from the fixed-point `s2f6`, decimal literals. Packing `x2`/`x4` registers is the PTX semantics' |
| `Mufu`, `MufuTables` | the special-function unit: `rcp sqrt rsqrt lg2 ex2 sin cos .approx.f32` |
| `Approx` | the `.approx` instructions on every format, with PTX's documented accuracy as facts |
| `Tensor` | tensor-core accumulation (`TensorModel.block`, `blockVals`, `dot`), Hopper datapaths as facts, `dotF64`, `dotInt`; the simulator's fast path (`prepare`, `operand`, `Datapath.block`) |
| `Block` | block-scaled formats (MXFP8/6/4, NVFP4): element values, OCP MX conversion |
| `Theorems` | correctness of rounding and round trips (below) |

## What rests on what

Kinds as in `NV/Fact.lean`; keys in `docs/sources.md`.

- **Documented** (`ptx-isa-9.5`): formats and their special values, rounding modes, `.ftz`, `.sat`,
  `.relu`, `.satfinite` (including NaN → +MAX_NORM for the 4- and 6-bit formats), min/max and
  comparison rules, float→int NaN results, binary64 NaN payloads, the `.approx` accuracy bounds,
  `div.approx`'s behaviour for `2^126 < |b| < 2^128`, the canonical NaN of the binary64 `.approx`
  forms. NaN constants and the host conversions' NaN and saturation rules: `cuda-headers-12.8`
  (`CUDART_NAN_F` = 0x7fffffff, `CUDART_NAN_FP16`/`_BF16` = 0x7fff, E4M3/E5M2 NaN 0x7f, UE8M0
  rounding).
- **Published, bit-accurate**: tensor cores (`khattak2025`), per family, selected by the target's
  SM version (`NV.PTX.Sem.Matrix.model`):

  | family | FP16/BF16 → FP32 | FP16 → FP16 | TF32 → FP32 | FP8 → FP32 | FP8 → FP16 | FP6/FP4 |
  |---|---|---|---|---|---|---|
  | `sm_80` (A100) | `Ampere.f16f32`, checked | `Ampere.f16f16`, checked | `Ampere.tf32f32`, checked | no datapath | no datapath | no datapath |
  | `sm_90a` (H100, H200) | `Hopper.f16f32`, checked | checked | checked | `Hopper.f8f32` (wgmma), checked; `mma.sync` as two FP16 MMAs | `Hopper.f8f16`, inferred | no datapath |
  | `sm_100a` (B200) | `Blackwell.f16f32`, checked | checked | checked | `Blackwell.f8f32` (`sf-tensor`, `tcgen05.mma`); `mma.sync` as two FP16 MMAs, checked | not modelled: faults | `Blackwell.f8f32`, block-scaled `NV.Num.Scaled` (`sf-tensor`) |
  | `sm_120a` (RTX PRO 6000) | the B200's (`Blackwell.rtxPro6000`, inferred) | same | same | same | not modelled: faults | the B200's, inferred |

  "Checked" means equal to the published outputs (`num-data`). The RTX PRO 6000's datapaths are
  taken as the B200's (inferred). The B200's FP8/FP6/FP4 and block-scaled datapaths
  (`tcgen05.mma`) are SF Tensor's model (`sf-tensor`): FP8, FP6, FP4 and `mxf8f6f4`'s scaled
  elements in one 25-bit window, truncated (`Blackwell.f8f32`); `mxf4`/`mxf4nvf4` in their wider
  windows (`NV.Num.Scaled`).
- **Published, bit-accurate**: the special-function unit (`ptxmath`; applied to `sm_90`, inferred).
- **Defined by the model** for forms the ISA leaves open: `mma.sync` with FP8 on `sm_90a` and
  `sm_100a` is two chained FP16 MMAs (`Matrix.loweredF8`: each element widened exactly to FP16,
  the first from zero, over alternate K pairs, then C added); `div.full.f32` is a scaled
  `a · MUFU.RCP(b)`; `div.approx.f32` scales both operands by `2^24` when `|b| < 2^-126`;
  `sin.approx.f32` is a multiplication by `1/2π` then `MUFU.SIN` on turns.
- **Inferred** (each marked in the code): binary32/16 NaN results are the canonical NaN (PTX:
  "unspecified"); which operand's payload a binary64 NaN result carries (the first); `.ftz` flushes
  results after rounding; `.sat` and `.relu` turn −0 into +0; widening conversions return the
  canonical NaN; scale factors apply to the exponent exactly (PTX writes `convert(a / scale)`);
  `s2f6` saturation compares the input's magnitude, so `-2` is never produced; overflow under
  `.rs` goes to infinity; `.rs` uses only the top discarded bits; the
  `.2pi` forms of `sin`/`cos` feed the unit the exact quarter-turn fraction; Hopper's FP8 wgmma
  aligns the accumulator like the products;
  FP8 → FP16 accumulation; the clamp point of integer `.satfinite` MMA.
- **Correctly rounded where no model is public**: f16/bf16 `ex2`, every `tanh`, binary64
  `rcp`/`rsqrt`. These lie within PTX's bounds but are not the hardware's bits.

## Theorems (`NV/Num/Theorems.lean`, kernel-checked)

- `Rounding.shift_rn_nearest`, `shift_rn_tie`, `shift_rna_nearest`, `shift_floor`, `shift_ceil`:
  each mode against the exact value; `shift_sticky`: a sticky remainder is one more set bit.
- `round_pack`: rounding a finite encoding's value gives the encoding back, in every mode.
- `Format.round_scale`: rounding depends only on the value.
- `Format.round_rn_nearest`: to nearest even, no finite encoding of an IEEE or NaN-only format
  with subnormals is closer to the exact value than the result.
- `cvt_round_trip`: a finite encoding converted into a format holding its value and back is
  itself.

## Speed

All arithmetic is on integers; powers of two come from a table (`pow2`, proved equal to `2 ^ n`:
the runtime's `Nat.pow` goes through GMP). The tensor cores have a word-level fast path:
`prepare` turns a datapath and its formats into machine words once, `operand` decodes each operand
once, and `Datapath.block` runs allocation-free loops; `TensorModel.tile` computes a warp's tile of
dot products that way. `lake exe num-tests bench` times the hot paths. It is not proved equal to
`blockVals`; it is tested equal on every operand encoding, every single FP8 product with
accumulators of each class, random blocks of every Hopper datapath, and `tile` equal to `dot`
elementwise on random tiles.

## Tests

- `lake exe num-tests` (CI row `num`): every encoding of every format of 16 bits or fewer decoded
  and re-encoded; every f16/bf16 input converted into every format in every mode with and without
  `.satfinite`; f32 inputs at every rounding boundary of each narrow format; 16-bit arithmetic,
  binary32/64 arithmetic, integer conversions and decimal literals on random and special operands —
  all against an independent rational reference that rounds by searching the encodings. NVIDIA's
  rules on named cases; MUFU outputs within PTX's documented accuracy of mpmath's values
  (`tools/num/approx_vectors.py`); every single FP8 product through the tensor model. Four injected
  bugs (ties to odd, no result flush, one alignment bit short, E4M3 overflow saturating) are each
  caught.
- Exhaustive 16-bit arithmetic (`tools/num/exhaustive.sh`, run on demand, not in CI): every pair
  of binary16 and of bfloat16 operands through add, sub, mul and div in each of the four IEEE
  modes, against a host reference in binary64 (`tools/num/exhaustive_ref.c`).
- CI row `num-data`: the tensor models against Khattak and Mikaitis's A100, H100/H200 and B200
  outputs
  (`tools/num/tensor-data.sh`), and the MUFU model against PTX Math's C model
  (`tools/num/mufu-check.sh`), both fetched at pinned commits.

## Open

- `tanh` and the 16-bit `.approx` forms are correctly rounded, not the hardware's bits.
- The RTX PRO 6000's tensor-core datapaths, its FP8 → FP16 and FP6/FP4 forms are the B200's
  (inferred).
- FP8 wgmma with a nonzero accumulator, FP8 → FP16 accumulation and `mma.sync` with FP8 are
  defined by the model, not checked against published outputs.
