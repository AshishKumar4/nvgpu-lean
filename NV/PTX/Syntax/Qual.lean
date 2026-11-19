import NV.PTX.Syntax.Type
import NV.PTX.Memory.Execution

/-!
# Instruction qualifiers

Everything after an opcode's dots that is not a type (PTX ISA 9.4 §9.7): rounding modes,
comparisons, state spaces, memory semantics and scopes, cache operators and eviction priorities,
reduction operators, vector widths, matrix shapes, dimensions, counts, and the nullary
qualifiers (`Flag`). Each instruction admits some qualifier kinds (`QKind`), listed by its
opcode (`NV.PTX.Syntax.Instr`); a qualifier string is read as the kind its instruction admits,
so `.lo` is a comparison in `setp.lo.u32` and a multiplication mode in `mul.lo.u32`.
-/

namespace NV.PTX

/-- Floating rounding (§9.7.3, §9.7.10.24): to nearest even, toward zero, down, up, to nearest
away from zero (`.rna`), stochastic (`.rs`). -/
inductive Rnd where
  | rn | rz | rm | rp | rna | rs
  deriving DecidableEq, Repr, Inhabited

instance : Keyword Rnd where
  all := [.rn, .rz, .rm, .rp, .rna, .rs]
  name | .rn => "rn" | .rz => "rz" | .rm => "rm" | .rp => "rp" | .rna => "rna" | .rs => "rs"

/-- Rounding to an integral value (`cvt`, §9.7.10.24). -/
inductive IntRnd where
  | rni | rzi | rmi | rpi
  deriving DecidableEq, Repr, Inhabited

instance : Keyword IntRnd where
  all := [.rni, .rzi, .rmi, .rpi]
  name | .rni => "rni" | .rzi => "rzi" | .rmi => "rmi" | .rpi => "rpi"

/-- Comparison operators of `set`/`setp` (§9.7.7.1, Tables 31–33). `lo ls hi hs` are the
unsigned comparisons; the `u` forms and `num`/`nan` are the unordered float comparisons. -/
inductive CmpOp where
  | eq | ne | lt | le | gt | ge | lo | ls | hi | hs
  | equ | neu | ltu | leu | gtu | geu | num | nan
  deriving DecidableEq, Repr, Inhabited

instance : Keyword CmpOp where
  all := [.eq, .ne, .lt, .le, .gt, .ge, .lo, .ls, .hi, .hs, .equ, .neu, .ltu, .leu, .gtu, .geu,
    .num, .nan]
  name
    | .eq => "eq" | .ne => "ne" | .lt => "lt" | .le => "le" | .gt => "gt" | .ge => "ge"
    | .lo => "lo" | .ls => "ls" | .hi => "hi" | .hs => "hs" | .equ => "equ" | .neu => "neu"
    | .ltu => "ltu" | .leu => "leu" | .gtu => "gtu" | .geu => "geu" | .num => "num"
    | .nan => "nan"

/-- Boolean combining operators (`setp`, `bar.red`, `lop3`, single-bit `mma`). -/
inductive BoolOp where
  | and | or | xor
  deriving DecidableEq, Repr, Inhabited

instance : Keyword BoolOp where
  all := [.and, .or, .xor]
  name | .and => "and" | .or => "or" | .xor => "xor"

/-- Memory-operation semantics (§8.4, §9.7.10.8): `.weak`, `.volatile` (relaxed `.sys` for
non-shared spaces), `.relaxed`, `.acquire`, `.release`, `.acq_rel`, `.sc` (fences), `.mmio`. -/
inductive MemSem where
  | weak | volatile | relaxed | acquire | release | acqRel | sc | mmio
  deriving DecidableEq, Repr, Inhabited

instance : Keyword MemSem where
  all := [.weak, .volatile, .relaxed, .acquire, .release, .acqRel, .sc, .mmio]
  name
    | .weak => "weak" | .volatile => "volatile" | .relaxed => "relaxed" | .acquire => "acquire"
    | .release => "release" | .acqRel => "acq_rel" | .sc => "sc" | .mmio => "mmio"

/-- Scope qualifiers (§8.5); `Scope.thread` has no keyword. -/
instance : Keyword Scope where
  all := [.cta, .cluster, .gpu, .sys]
  name
    | .thread => "thread" | .cta => "cta" | .cluster => "cluster" | .gpu => "gpu"
    | .sys => "sys"

/-- Cache operators (§9.7.10.1). -/
inductive CacheOp where
  | ca | cg | cs | lu | cv | wb | wt
  deriving DecidableEq, Repr, Inhabited

instance : Keyword CacheOp where
  all := [.ca, .cg, .cs, .lu, .cv, .wb, .wt]
  name
    | .ca => "ca" | .cg => "cg" | .cs => "cs" | .lu => "lu" | .cv => "cv" | .wb => "wb"
    | .wt => "wt"

/-- Cache eviction priorities (§9.7.10.2). -/
inductive Evict where
  | l1Normal | l1Unchanged | l1First | l1Last | l1NoAllocate
  | l2Normal | l2First | l2Last | l2Unchanged
  deriving DecidableEq, Repr, Inhabited

instance : Keyword Evict where
  all := [.l1Normal, .l1Unchanged, .l1First, .l1Last, .l1NoAllocate, .l2Normal, .l2First,
    .l2Last, .l2Unchanged]
  name
    | .l1Normal => "L1::evict_normal" | .l1Unchanged => "L1::evict_unchanged"
    | .l1First => "L1::evict_first" | .l1Last => "L1::evict_last"
    | .l1NoAllocate => "L1::no_allocate" | .l2Normal => "L2::evict_normal"
    | .l2First => "L2::evict_first" | .l2Last => "L2::evict_last"
    | .l2Unchanged => "L2::evict_unchanged"

/-- Atomic and reduction operators (`atom`, `red`, `redux`, `multimem`, `cp.reduce`). -/
inductive RedOp where
  | add | inc | dec | min | max | and | or | xor | exch | cas
  deriving DecidableEq, Repr, Inhabited

instance : Keyword RedOp where
  all := [.add, .inc, .dec, .min, .max, .and, .or, .xor, .exch, .cas]
  name
    | .add => "add" | .inc => "inc" | .dec => "dec" | .min => "min" | .max => "max"
    | .and => "and" | .or => "or" | .xor => "xor" | .exch => "exch" | .cas => "cas"

/-- Nullary qualifiers, spelled as in the ISA. -/
inductive Flag where
  -- Arithmetic
  | ftz | sat | satfinite | relu | approx | full | nan | xorsign | abs | oob | noftz | cc | hi
  | lo | wide | shiftamt | clamp | wrap | l | r | f4e | b4e | rc8 | ecl | ecr | rc16 | po | shr7
  | shr15 | pzo | scaledN1Ue8m0 | scaledN2Ue8m0 | inc | dec
  -- `testp` classes
  | finite | infinite | number | notanumber | normal | subnormal
  -- Control and synchronization
  | uni | aligned | sync | to | popc | red | up | down | bfly | idx | all | any | ballot | mask
  | launchDependents | wait
  -- Memory access
  | nc | unified | l1 | l2 | l2CacheHint | l2Size64 | l2Size128 | l2Size256 | l1Size32
  | proxyReadonly | tensormap | validAddr
  -- Fences
  | proxy | aliasProxy | async | asyncGeneric | generic | tensormapGeneric | genericFabric
  | fabricGeneric | fabricFabric | gl | syncRestrictSharedCluster | syncRestrictSharedCta
  | syncRestrictSharedClusterRead | syncRestrictReads | syncRestrictSharedReadMmaA
  | mbarrierInit | opRestrictFetching
  -- mbarrier and async copies
  | expectTx | noComplete | parity | phaseTypePrimary | phaseTypeConditional | layoutV0
  | layoutV1 | noinc | mbarrierArriveOne | mbarrierCompleteTxBytes | mbarrierCompleteTx16B
  | mbarrierReportDisabled | mbarrierReportValidityPer16bytes80000000
  | mbarrierReportValidityPer16bytes8000 | mbarrierReportValidityPer16bytes80
  | mbarrierReportValidityPer16bytes8 | mbarrierReportValidityPerElementFf
  | mbarrierReportFabric | countedBytes | bulkGroup | multicastCluster | multicastCluster16b
  | multicastCluster32b | multicastClusterAll | ignoreOob | cpMask | tile | tileGather4
  | tileScatter4 | tiled | im2col | im2colW | im2colW128 | im2colNoOffs | im2colNoOffsW
  | ctaGroup1 | ctaGroup2 | overrideGlobalAddress | overrideGlobalDim | overrideGlobalDimStride
  | read | multimem | tensor | spcompress | ws
  -- Matrix instructions
  | row | col | trans | a | b | c | d | kindF16 | kindTf32 | kindF8f6f4 | kindI8 | kindTi16
  | kindMxf4 | kindMxf4nvf4 | kindMxf8f6f4 | blockScale | scaleVec1X | scaleVec2X | scaleVec4X
  | block16 | block32 | sp | spOrderedMetadata | accF32 | accF16
  -- tcgen05
  | shape16x64b | shape16x128b | shape16x256b | shape32x32b | shape16x32bx2 | shape128x256b
  | shape4x256b | shape128x128b | shape64x128b | shape32x128b | pack16b | unpack16b
  | warpx2Rows02x13 | warpx2Rows01x23 | warpx4 | b8x16 | b6x16P32 | b4x16P64 | exclusive | ashift
  | decompressLutB
  | collectorAFill | collectorAUse | collectorALastuse | collectorADiscard | collectorBFill
  | collectorBUse | collectorBLastuse | collectorBDiscard | collectorB0Fill | collectorB0Use
  | collectorB0Lastuse | collectorB0Discard | collectorB1Fill | collectorB1Use
  | collectorB1Lastuse | collectorB1Discard | collectorB2Fill | collectorB2Use
  | collectorB2Lastuse | collectorB2Discard | collectorB3Fill | collectorB3Use
  | collectorB3Lastuse | collectorB3Discard | sp1of2 | sp1of4 | sp1of8 | sp1of16 | sp2of4
  | sp2of8 | sp2of16 | sp4of8 | sp4of16
  -- Textures and surfaces
  | a1d | a2d | cube | acube | geom2dms | a2dms | base | level | grad | g | p | trap | zero
  | width | height | depth | channelDataType | channelOrder | normalizedCoords | arraySize
  | numMipmapLevels | numSamples | forceUnnormalizedCoords | filterMode | addrMode0 | addrMode1
  | addrMode2 | memoryLayout
  -- tensormap.replace
  | globalAddress | rank | boxDim | globalDim | globalStride | elementStride | elemtype
  | interleaveLayout | swizzleMode | swizzleAtomicity | fillMode
  -- Policies and cluster launch control
  | range | fractional | cvt | fabric | isCanceled | getFirstCtaid | getFirstCtaidX
  | getFirstCtaidY | getFirstCtaidZ
  deriving DecidableEq, Repr, Inhabited

instance : Keyword Flag where
  all := [.ftz, .sat, .satfinite, .relu, .approx, .full, .nan, .xorsign, .abs, .oob, .noftz, .cc,
    .hi, .lo, .wide, .shiftamt, .clamp, .wrap, .l, .r, .f4e, .b4e, .rc8, .ecl, .ecr, .rc16, .po,
    .shr7, .shr15, .pzo, .scaledN1Ue8m0, .scaledN2Ue8m0, .inc, .dec, .finite, .infinite, .number,
    .notanumber, .normal, .subnormal, .uni, .aligned, .sync, .to, .popc, .red, .up, .down, .bfly,
    .idx, .all, .any, .ballot, .mask, .launchDependents, .wait, .nc, .unified, .l1, .l2,
    .l2CacheHint, .l2Size64, .l2Size128, .l2Size256, .l1Size32, .proxyReadonly, .tensormap,
    .validAddr, .proxy, .aliasProxy, .async, .asyncGeneric, .generic, .tensormapGeneric,
    .genericFabric, .fabricGeneric, .fabricFabric, .gl, .syncRestrictSharedCluster,
    .syncRestrictSharedCta, .syncRestrictSharedClusterRead, .syncRestrictReads,
    .syncRestrictSharedReadMmaA, .mbarrierInit, .opRestrictFetching, .expectTx, .noComplete,
    .parity, .phaseTypePrimary, .phaseTypeConditional, .layoutV0, .layoutV1, .noinc,
    .mbarrierArriveOne, .mbarrierCompleteTxBytes, .mbarrierCompleteTx16B, .mbarrierReportDisabled,
    .mbarrierReportValidityPer16bytes80000000, .mbarrierReportValidityPer16bytes8000,
    .mbarrierReportValidityPer16bytes80, .mbarrierReportValidityPer16bytes8,
    .mbarrierReportValidityPerElementFf, .mbarrierReportFabric, .countedBytes, .bulkGroup,
    .multicastCluster, .multicastCluster16b, .multicastCluster32b, .multicastClusterAll,
    .ignoreOob, .cpMask, .tile, .tileGather4, .tileScatter4, .tiled, .im2col, .im2colW,
    .im2colW128, .im2colNoOffs, .im2colNoOffsW, .ctaGroup1, .ctaGroup2, .overrideGlobalAddress,
    .overrideGlobalDim, .overrideGlobalDimStride, .read, .multimem, .tensor, .spcompress, .ws,
    .row, .col, .trans, .a, .b, .c, .d, .kindF16, .kindTf32, .kindF8f6f4, .kindI8, .kindTi16,
    .kindMxf4, .kindMxf4nvf4, .kindMxf8f6f4, .blockScale, .scaleVec1X, .scaleVec2X, .scaleVec4X,
    .block16, .block32, .sp, .spOrderedMetadata, .accF32, .accF16, .shape16x64b, .shape16x128b,
    .shape16x256b, .shape32x32b, .shape16x32bx2, .shape128x256b, .shape4x256b, .shape128x128b,
    .shape64x128b, .shape32x128b, .pack16b, .unpack16b, .warpx2Rows02x13, .warpx2Rows01x23,
    .warpx4, .b8x16, .b6x16P32, .b4x16P64, .exclusive, .ashift, .decompressLutB, .collectorAFill,
    .collectorAUse, .collectorALastuse, .collectorADiscard, .collectorBFill, .collectorBUse,
    .collectorBLastuse, .collectorBDiscard, .collectorB0Fill, .collectorB0Use, .collectorB0Lastuse,
    .collectorB0Discard, .collectorB1Fill, .collectorB1Use, .collectorB1Lastuse,
    .collectorB1Discard, .collectorB2Fill, .collectorB2Use, .collectorB2Lastuse,
    .collectorB2Discard, .collectorB3Fill, .collectorB3Use, .collectorB3Lastuse,
    .collectorB3Discard, .sp1of2, .sp1of4, .sp1of8, .sp1of16, .sp2of4, .sp2of8, .sp2of16, .sp4of8,
    .sp4of16, .a1d, .a2d, .cube, .acube, .geom2dms, .a2dms, .base, .level, .grad, .g, .p, .trap,
    .zero, .width, .height, .depth, .channelDataType, .channelOrder, .normalizedCoords,
    .arraySize, .numMipmapLevels, .numSamples, .forceUnnormalizedCoords, .filterMode, .addrMode0,
    .addrMode1, .addrMode2, .memoryLayout, .globalAddress, .rank, .boxDim, .globalDim,
    .globalStride, .elementStride, .elemtype, .interleaveLayout, .swizzleMode, .swizzleAtomicity,
    .fillMode, .range, .fractional, .cvt, .fabric, .isCanceled, .getFirstCtaid, .getFirstCtaidX,
    .getFirstCtaidY, .getFirstCtaidZ]
  name
    | .ftz => "ftz"
    | .sat => "sat"
    | .satfinite => "satfinite"
    | .relu => "relu"
    | .approx => "approx"
    | .full => "full"
    | .nan => "NaN"
    | .xorsign => "xorsign"
    | .abs => "abs"
    | .oob => "oob"
    | .noftz => "noftz"
    | .cc => "cc"
    | .hi => "hi"
    | .lo => "lo"
    | .wide => "wide"
    | .shiftamt => "shiftamt"
    | .clamp => "clamp"
    | .wrap => "wrap"
    | .l => "l"
    | .r => "r"
    | .f4e => "f4e"
    | .b4e => "b4e"
    | .rc8 => "rc8"
    | .ecl => "ecl"
    | .ecr => "ecr"
    | .rc16 => "rc16"
    | .po => "po"
    | .shr7 => "shr7"
    | .shr15 => "shr15"
    | .pzo => "pzo"
    | .scaledN1Ue8m0 => "scaled::n1::ue8m0"
    | .scaledN2Ue8m0 => "scaled::n2::ue8m0"
    | .inc => "inc"
    | .dec => "dec"
    | .finite => "finite"
    | .infinite => "infinite"
    | .number => "number"
    | .notanumber => "notanumber"
    | .normal => "normal"
    | .subnormal => "subnormal"
    | .uni => "uni"
    | .aligned => "aligned"
    | .sync => "sync"
    | .to => "to"
    | .popc => "popc"
    | .red => "red"
    | .up => "up"
    | .down => "down"
    | .bfly => "bfly"
    | .idx => "idx"
    | .all => "all"
    | .any => "any"
    | .ballot => "ballot"
    | .mask => "mask"
    | .launchDependents => "launch_dependents"
    | .wait => "wait"
    | .nc => "nc"
    | .unified => "unified"
    | .l1 => "L1"
    | .l2 => "L2"
    | .l2CacheHint => "L2::cache_hint"
    | .l2Size64 => "L2::64B"
    | .l2Size128 => "L2::128B"
    | .l2Size256 => "L2::256B"
    | .l1Size32 => "L1::32B"
    | .proxyReadonly => "proxy::readonly"
    | .tensormap => "tensormap"
    | .validAddr => "valid_addr"
    | .proxy => "proxy"
    | .aliasProxy => "alias"
    | .async => "async"
    | .asyncGeneric => "async::generic"
    | .generic => "generic"
    | .tensormapGeneric => "tensormap::generic"
    | .genericFabric => "generic::fabric"
    | .fabricGeneric => "fabric::generic"
    | .fabricFabric => "fabric::fabric"
    | .gl => "gl"
    | .syncRestrictSharedCluster => "sync_restrict::shared::cluster"
    | .syncRestrictSharedCta => "sync_restrict::shared::cta"
    | .syncRestrictSharedClusterRead => "sync_restrict::shared::cluster::read"
    | .syncRestrictReads => "sync_restrict::reads"
    | .syncRestrictSharedReadMmaA => "sync_restrict::shared::read::mma::a"
    | .mbarrierInit => "mbarrier_init"
    | .opRestrictFetching => "op_restrict::fetching"
    | .expectTx => "expect_tx"
    | .noComplete => "noComplete"
    | .parity => "parity"
    | .phaseTypePrimary => "phase_type::primary"
    | .phaseTypeConditional => "phase_type::conditional"
    | .layoutV0 => "layout::v0"
    | .layoutV1 => "layout::v1"
    | .noinc => "noinc"
    | .mbarrierArriveOne => "mbarrier::arrive::one"
    | .mbarrierCompleteTxBytes => "mbarrier::complete_tx::bytes"
    | .mbarrierCompleteTx16B => "mbarrier::complete_tx::16B"
    | .mbarrierReportDisabled => "mbarrier::report::disabled"
    | .mbarrierReportValidityPer16bytes80000000 => "mbarrier::report::validity::per_16bytes::80000000"
    | .mbarrierReportValidityPer16bytes8000 => "mbarrier::report::validity::per_16bytes::8000"
    | .mbarrierReportValidityPer16bytes80 => "mbarrier::report::validity::per_16bytes::80"
    | .mbarrierReportValidityPer16bytes8 => "mbarrier::report::validity::per_16bytes::8"
    | .mbarrierReportValidityPerElementFf => "mbarrier::report::validity::per_element::ff"
    | .mbarrierReportFabric => "mbarrier::report::fabric"
    | .countedBytes => "counted::bytes"
    | .bulkGroup => "bulk_group"
    | .multicastCluster => "multicast::cluster"
    | .multicastCluster16b => "multicast::cluster::16b"
    | .multicastCluster32b => "multicast::cluster::32b"
    | .multicastClusterAll => "multicast::cluster::all"
    | .ignoreOob => "ignore_oob"
    | .cpMask => "cp_mask"
    | .tile => "tile"
    | .tileGather4 => "tile::gather4"
    | .tileScatter4 => "tile::scatter4"
    | .tiled => "tiled"
    | .im2col => "im2col"
    | .im2colW => "im2col::w"
    | .im2colW128 => "im2col::w::128"
    | .im2colNoOffs => "im2col_no_offs"
    | .im2colNoOffsW => "im2col_no_offs::w"
    | .ctaGroup1 => "cta_group::1"
    | .ctaGroup2 => "cta_group::2"
    | .overrideGlobalAddress => "override::global_address"
    | .overrideGlobalDim => "override::global_dim"
    | .overrideGlobalDimStride => "override::global_dim_stride"
    | .read => "read"
    | .multimem => "multimem"
    | .tensor => "tensor"
    | .spcompress => "spcompress"
    | .ws => "ws"
    | .row => "row"
    | .col => "col"
    | .trans => "trans"
    | .a => "a"
    | .b => "b"
    | .c => "c"
    | .d => "d"
    | .kindF16 => "kind::f16"
    | .kindTf32 => "kind::tf32"
    | .kindF8f6f4 => "kind::f8f6f4"
    | .kindI8 => "kind::i8"
    | .kindTi16 => "kind::ti16"
    | .kindMxf4 => "kind::mxf4"
    | .kindMxf4nvf4 => "kind::mxf4nvf4"
    | .kindMxf8f6f4 => "kind::mxf8f6f4"
    | .blockScale => "block_scale"
    | .scaleVec1X => "scale_vec::1X"
    | .scaleVec2X => "scale_vec::2X"
    | .scaleVec4X => "scale_vec::4X"
    | .block16 => "block16"
    | .block32 => "block32"
    | .sp => "sp"
    | .spOrderedMetadata => "sp::ordered_metadata"
    | .accF32 => "acc::f32"
    | .accF16 => "acc::f16"
    | .shape16x64b => "16x64b"
    | .shape16x128b => "16x128b"
    | .shape16x256b => "16x256b"
    | .shape32x32b => "32x32b"
    | .shape16x32bx2 => "16x32bx2"
    | .shape128x256b => "128x256b"
    | .shape4x256b => "4x256b"
    | .shape128x128b => "128x128b"
    | .shape64x128b => "64x128b"
    | .shape32x128b => "32x128b"
    | .pack16b => "pack::16b"
    | .unpack16b => "unpack::16b"
    | .warpx2Rows02x13 => "warpx2::02_13"
    | .warpx2Rows01x23 => "warpx2::01_23"
    | .warpx4 => "warpx4"
    | .b8x16 => "b8x16"
    | .b6x16P32 => "b6x16_p32"
    | .b4x16P64 => "b4x16_p64"
    | .exclusive => "exclusive"
    | .ashift => "ashift"
    | .decompressLutB => "decompress::lut::b"
    | .collectorAFill => "collector::a::fill"
    | .collectorAUse => "collector::a::use"
    | .collectorALastuse => "collector::a::lastuse"
    | .collectorADiscard => "collector::a::discard"
    | .collectorBFill => "collector::b::fill"
    | .collectorBUse => "collector::b::use"
    | .collectorBLastuse => "collector::b::lastuse"
    | .collectorBDiscard => "collector::b::discard"
    | .collectorB0Fill => "collector::b0::fill"
    | .collectorB0Use => "collector::b0::use"
    | .collectorB0Lastuse => "collector::b0::lastuse"
    | .collectorB0Discard => "collector::b0::discard"
    | .collectorB1Fill => "collector::b1::fill"
    | .collectorB1Use => "collector::b1::use"
    | .collectorB1Lastuse => "collector::b1::lastuse"
    | .collectorB1Discard => "collector::b1::discard"
    | .collectorB2Fill => "collector::b2::fill"
    | .collectorB2Use => "collector::b2::use"
    | .collectorB2Lastuse => "collector::b2::lastuse"
    | .collectorB2Discard => "collector::b2::discard"
    | .collectorB3Fill => "collector::b3::fill"
    | .collectorB3Use => "collector::b3::use"
    | .collectorB3Lastuse => "collector::b3::lastuse"
    | .collectorB3Discard => "collector::b3::discard"
    | .sp1of2 => "sp::1:2"
    | .sp1of4 => "sp::1:4"
    | .sp1of8 => "sp::1:8"
    | .sp1of16 => "sp::1:16"
    | .sp2of4 => "sp::2:4"
    | .sp2of8 => "sp::2:8"
    | .sp2of16 => "sp::2:16"
    | .sp4of8 => "sp::4:8"
    | .sp4of16 => "sp::4:16"
    | .a1d => "a1d"
    | .a2d => "a2d"
    | .cube => "cube"
    | .acube => "acube"
    | .geom2dms => "2dms"
    | .a2dms => "a2dms"
    | .base => "base"
    | .level => "level"
    | .grad => "grad"
    | .g => "g"
    | .p => "p"
    | .trap => "trap"
    | .zero => "zero"
    | .width => "width"
    | .height => "height"
    | .depth => "depth"
    | .channelDataType => "channel_data_type"
    | .channelOrder => "channel_order"
    | .normalizedCoords => "normalized_coords"
    | .arraySize => "array_size"
    | .numMipmapLevels => "num_mipmap_levels"
    | .numSamples => "num_samples"
    | .forceUnnormalizedCoords => "force_unnormalized_coords"
    | .filterMode => "filter_mode"
    | .addrMode0 => "addr_mode_0"
    | .addrMode1 => "addr_mode_1"
    | .addrMode2 => "addr_mode_2"
    | .memoryLayout => "memory_layout"
    | .globalAddress => "global_address"
    | .rank => "rank"
    | .boxDim => "box_dim"
    | .globalDim => "global_dim"
    | .globalStride => "global_stride"
    | .elementStride => "element_stride"
    | .elemtype => "elemtype"
    | .interleaveLayout => "interleave_layout"
    | .swizzleMode => "swizzle_mode"
    | .swizzleAtomicity => "swizzle_atomicity"
    | .fillMode => "fill_mode"
    | .range => "range"
    | .fractional => "fractional"
    | .cvt => "cvt"
    | .fabric => "fabric"
    | .isCanceled => "is_canceled"
    | .getFirstCtaid => "get_first_ctaid"
    | .getFirstCtaidX => "get_first_ctaid::x"
    | .getFirstCtaidY => "get_first_ctaid::y"
    | .getFirstCtaidZ => "get_first_ctaid::z"

theorem Flag.complete (f : Flag) : f ∈ (Keyword.all : List Flag) := by cases f <;> decide

inductive Qual where
  | rnd (r : Rnd)
  | irnd (r : IntRnd)
  | cmp (c : CmpOp)
  | bool (b : BoolOp)
  | space (s : Space)
  | sem (s : MemSem)
  | scope (s : Scope)
  | cop (c : CacheOp)
  | evict (e : Evict)
  | red (o : RedOp)
  | vec (v : Vec)
  /-- A matrix shape `.mMnNkK`, or `.mMnN` with `k = 0`. -/
  | shape (m n k : Nat)
  /-- `.1d` … `.5d`: tensor dimensions, or texture and surface geometry. -/
  | dim (n : Nat)
  /-- `.x1`, `.x2`, …: matrix and tensor-memory counts. -/
  | num (n : Nat)
  | flag (f : Flag)
  deriving DecidableEq, Repr, Inhabited

/-- The kinds of qualifier an instruction admits; a flag kind admits that one flag. -/
inductive QKind where
  | rnd | irnd | cmp | bool | space | sem | scope | cop | evict | red | vec | shape | dim | num
  | flag (f : Flag)
  deriving DecidableEq, Repr, Inhabited

namespace Qual

/-- Digits of a string as a number, if it is all digits. -/
def digitsOf? (s : String) : Option Nat :=
  if s.isEmpty || !s.all Char.isDigit then none else s.toNat?

/-- `m16n8k16` as `(16, 8, 16)`, `m8n8` as `(8, 8, 0)`. -/
def shapeOf? (s : String) : Option (Nat × Nat × Nat) := do
  guard (s.startsWith "m")
  let rest := String.ofList (s.toList.drop 1)
  match rest.splitOn "n" with
  | [m, nk] =>
    let m ← digitsOf? m
    match nk.splitOn "k" with
    | [n] => return (m, ← digitsOf? n, 0)
    | [n, k] => return (m, ← digitsOf? n, ← digitsOf? k)
    | _ => none
  | _ => none

/-- Read `s` as a qualifier of kind `k`. -/
def parse? (k : QKind) (s : String) : Option Qual :=
  match k with
  | .rnd => (Keyword.parse? s).map .rnd
  | .irnd => (Keyword.parse? s).map .irnd
  | .cmp => (Keyword.parse? s).map .cmp
  | .bool => (Keyword.parse? s).map .bool
  | .space => (Keyword.parse? s).map .space
  | .sem => (Keyword.parse? s).map .sem
  | .scope => (Keyword.parse? s).map .scope
  | .cop => (Keyword.parse? s).map .cop
  | .evict => (Keyword.parse? s).map .evict
  | .red => (Keyword.parse? s).map .red
  | .vec => (Keyword.parse? s).map .vec
  | .shape => (shapeOf? s).map fun (m, n, k) => .shape m n k
  | .dim =>
    if s.endsWith "d" then (digitsOf? (String.ofList s.toList.dropLast)).map .dim else none
  | .num =>
    if s.startsWith "x" then (digitsOf? (String.ofList (s.toList.drop 1))).map .num else none
  | .flag f => if Keyword.name f == s then some (.flag f) else none

def spelling : Qual → String
  | .rnd r => Keyword.name r
  | .irnd r => Keyword.name r
  | .cmp c => Keyword.name c
  | .bool b => Keyword.name b
  | .space s => Keyword.name s
  | .sem s => Keyword.name s
  | .scope s => Keyword.name s
  | .cop c => Keyword.name c
  | .evict e => Keyword.name e
  | .red o => Keyword.name o
  | .vec v => Keyword.name v
  | .shape m n k => if k == 0 then s!"m{m}n{n}" else s!"m{m}n{n}k{k}"
  | .dim n => s!"{n}d"
  | .num n => s!"x{n}"
  | .flag f => Keyword.name f

end Qual

end NV.PTX
