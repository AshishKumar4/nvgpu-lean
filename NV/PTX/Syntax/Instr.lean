import NV.PTX.Syntax.Ast
import NV.PTX.Syntax.Qual
import Std.Data.HashMap

/-!
# Typed instructions

Every instruction of PTX ISA 9.4 §9.7 as an opcode (`Op`), its qualifiers (`Qual`), its types
in written order, and its operands. An opcode is spelled by one or more dot-separated parts
(`cp.async.bulk.tensor`); `Op.spec` gives the parts and the qualifier kinds the opcode admits.

`decode` reads a `RawInstr`: it picks the opcode with the longest spelling whose parts appear in
order among the written parts (so `barrier.cta.sync` is `barrierSync` with scope `cta`), then
reads every other part as a qualifier of a kind the opcode admits, or as a type. A part that is
neither is an error. Which combinations of qualifiers and types an opcode allows is the checker's
business (`NV.PTX.Syntax.Check`), as are the operands.
-/

namespace NV.PTX

/-- Instruction opcodes, grouped as in §9.7. -/
inductive Op where
  -- Integer arithmetic (§9.7.1, §9.7.2)
  | add | sub | mul | mad | clmad | mul24 | mad24 | sad | div | rem | abs | neg | min | max
  | popc | clz | bfind | fns | brev | bfe | bfi | szext | bmsk | dp4a | dp2a | addc | subc
  | madc
  -- Floating point (§9.7.3–§9.7.6)
  | testp | copysign | fma | rcp | sqrt | rsqrt | sin | cos | lg2 | ex2 | tanh
  -- Comparison and selection (§9.7.7, §9.7.8)
  | set | setp | selp | slct
  -- Logic and shift (§9.7.9)
  | and | or | xor | not | cnot | lop3 | shf | shl | shr
  -- Data movement and conversion (§9.7.10)
  | mov | shfl | shflSync | prmt | ld | ldu | st | stAsync | stBulk | multimemLdReduce
  | multimemSt | multimemRed | multimemStAsync | multimemRedAsync | multimemCpAsyncBulk
  | multimemCpReduceAsyncBulk | prefetch | prefetchu | applypriority | applypriorityAsyncBulk
  | applypriorityAsyncBulkTensor | discard | createpolicy | isspacep | cvta | cvt | cvtPack
  | mapa | getctarank | cpAsync | cpAsyncCommitGroup | cpAsyncWaitGroup | cpAsyncWaitAll
  | cpAsyncMbarrierArrive | cpAsyncBulk | cpReduceAsyncBulk | cpAsyncBulkPrefetch
  | cpAsyncBulkTensor | cpReduceAsyncBulkTensor | cpAsyncBulkPrefetchTensor
  | cpAsyncBulkCommitGroup | cpAsyncBulkWaitGroup | tensormapReplace | spcompress | spdecompress
  -- Fabric (§9.7.11)
  | fabricTryGet | fabricTryPut | fabricTryRed | fabricTryPullred | fabricTryAtom | fabricSubmit
  | fabricWait
  -- Texture and surface (§9.7.12, §9.7.13)
  | tex | tld4 | txq | istypep | suld | sust | sured | suq
  -- Control flow (§9.7.14)
  | bra | brxIdx | call | ret | exit
  -- Synchronization and communication (§9.7.15)
  | barSync | barArrive | barRed | barrierSync | barrierArrive | barrierRed | barWarpSync
  | barrierClusterArrive | barrierClusterWait | membar | fence | atom | red | redAsync | vote
  | voteSync | matchSync | activemask | reduxSync | griddepcontrol | electSync | mbarrierInit
  | mbarrierInval | mbarrierExpectTx | mbarrierCompleteTx | mbarrierArrive | mbarrierArriveDrop
  | mbarrierTestWait | mbarrierTryWait | mbarrierPendingCount | mbarrierCheckLayout
  | tensormapCpFenceproxy | clusterlaunchcontrolTryCancel | clusterlaunchcontrolQueryCancel
  -- Warp-level matrix instructions (§9.7.16)
  | wmmaLoad | wmmaStore | wmmaMma | mma | ldmatrix | stmatrix | movmatrix
  -- Warpgroup matrix instructions (§9.7.17)
  | wgmmaMmaAsync | wgmmaFence | wgmmaCommitGroup | wgmmaWaitGroup
  -- Tensor core 5th generation (§9.7.18)
  | tcgen05Alloc | tcgen05Dealloc | tcgen05RelinquishAllocPermit | tcgen05Ld | tcgen05St
  | tcgen05WaitLd | tcgen05WaitSt | tcgen05Cp | tcgen05Shift | tcgen05Mma
  | tcgen05FenceBeforeThreadSync | tcgen05FenceAfterThreadSync | tcgen05Commit
  -- Stack manipulation (§9.7.19)
  | stacksave | stackrestore | alloca
  -- Video (§9.7.20)
  | vadd | vsub | vabsdiff | vmin | vmax | vshl | vshr | vmad | vset | vadd2 | vsub2 | vavrg2
  | vabsdiff2 | vmin2 | vmax2 | vset2 | vadd4 | vsub4 | vavrg4 | vabsdiff4 | vmin4 | vmax4
  | vset4
  -- Miscellaneous (§9.7.21)
  | brkpt | nanosleep | pmevent | trap | setmaxnreg
  deriving DecidableEq, Repr, Inhabited


/-- The opcode's spelling as dot-separated parts, and the qualifier kinds it admits. -/
def Op.spec : Op → List String × List QKind
  | .add => (["add"], [.rnd, .flag .ftz, .flag .sat, .flag .cc, .flag .satfinite])
  | .sub => (["sub"], [.rnd, .flag .ftz, .flag .sat, .flag .cc, .flag .satfinite])
  | .mul => (["mul"], [
      .rnd, .flag .ftz, .flag .sat, .flag .hi, .flag .lo, .flag .wide, .flag .satfinite])
  | .mad => (["mad"], [.rnd, .flag .ftz, .flag .sat, .flag .hi, .flag .lo, .flag .wide, .flag .cc])
  | .clmad => (["clmad"], [.flag .hi, .flag .lo])
  | .mul24 => (["mul24"], [.flag .hi, .flag .lo])
  | .mad24 => (["mad24"], [.flag .hi, .flag .lo, .flag .sat])
  | .sad => (["sad"], [])
  | .div => (["div"], [.rnd, .flag .approx, .flag .full, .flag .ftz])
  | .rem => (["rem"], [])
  | .abs => (["abs"], [.flag .ftz])
  | .neg => (["neg"], [.flag .ftz])
  | .min => (["min"], [.flag .ftz, .flag .nan, .flag .xorsign, .flag .abs, .flag .relu])
  | .max => (["max"], [.flag .ftz, .flag .nan, .flag .xorsign, .flag .abs, .flag .relu])
  | .popc => (["popc"], [])
  | .clz => (["clz"], [])
  | .bfind => (["bfind"], [.flag .shiftamt])
  | .fns => (["fns"], [])
  | .brev => (["brev"], [])
  | .bfe => (["bfe"], [])
  | .bfi => (["bfi"], [])
  | .szext => (["szext"], [.flag .clamp, .flag .wrap])
  | .bmsk => (["bmsk"], [.flag .clamp, .flag .wrap])
  | .dp4a => (["dp4a"], [])
  | .dp2a => (["dp2a"], [.flag .hi, .flag .lo])
  | .addc => (["addc"], [.flag .cc])
  | .subc => (["subc"], [.flag .cc])
  | .madc => (["madc"], [.flag .hi, .flag .lo, .flag .cc])
  | .testp => (["testp"], [
      .flag .finite, .flag .infinite, .flag .number, .flag .notanumber, .flag .normal,
      .flag .subnormal])
  | .copysign => (["copysign"], [])
  | .fma => (["fma"], [.rnd, .flag .ftz, .flag .sat, .flag .relu, .flag .oob, .flag .satfinite])
  | .rcp => (["rcp"], [.rnd, .flag .approx, .flag .ftz])
  | .sqrt => (["sqrt"], [.rnd, .flag .approx, .flag .ftz])
  | .rsqrt => (["rsqrt"], [.flag .approx, .flag .ftz])
  | .sin => (["sin"], [.flag .approx, .flag .ftz])
  | .cos => (["cos"], [.flag .approx, .flag .ftz])
  | .lg2 => (["lg2"], [.flag .approx, .flag .ftz])
  | .ex2 => (["ex2"], [.flag .approx, .flag .ftz])
  | .tanh => (["tanh"], [.flag .approx])
  | .set => (["set"], [.cmp, .bool, .flag .ftz])
  | .setp => (["setp"], [.cmp, .bool, .flag .ftz])
  | .selp => (["selp"], [])
  | .slct => (["slct"], [.flag .ftz])
  | .and => (["and"], [])
  | .or => (["or"], [])
  | .xor => (["xor"], [])
  | .not => (["not"], [])
  | .cnot => (["cnot"], [])
  | .lop3 => (["lop3"], [.bool])
  | .shf => (["shf"], [.flag .l, .flag .r, .flag .clamp, .flag .wrap])
  | .shl => (["shl"], [])
  | .shr => (["shr"], [])
  | .mov => (["mov"], [])
  | .shfl => (["shfl"], [.flag .up, .flag .down, .flag .bfly, .flag .idx])
  | .shflSync => (["shfl", "sync"], [.flag .up, .flag .down, .flag .bfly, .flag .idx])
  | .prmt => (["prmt"], [.flag .f4e, .flag .b4e, .flag .rc8, .flag .ecl, .flag .ecr, .flag .rc16])
  | .ld => (["ld"], [
      .sem, .scope, .space, .cop, .evict, .vec, .flag .l2CacheHint, .flag .l2Size64,
      .flag .l2Size128, .flag .l2Size256, .flag .nc, .flag .unified, .flag .proxyReadonly])
  | .ldu => (["ldu"], [.space, .vec])
  | .st => (["st"], [.sem, .scope, .space, .cop, .evict, .vec, .flag .l2CacheHint])
  | .stAsync => (["st", "async"], [.sem, .scope, .space, .vec, .flag .mbarrierCompleteTxBytes])
  | .stBulk => (["st", "bulk"], [.sem, .space])
  | .multimemLdReduce => (["multimem", "ld_reduce"], [
      .sem, .scope, .space, .red, .vec, .flag .accF32, .flag .accF16])
  | .multimemSt => (["multimem", "st"], [.sem, .scope, .space, .vec])
  | .multimemRed => (["multimem", "red"], [.sem, .scope, .space, .red, .vec])
  | .multimemStAsync => (["multimem", "st", "async"], [.sem, .scope, .space])
  | .multimemRedAsync => (["multimem", "red", "async"], [.sem, .scope, .space, .red])
  | .multimemCpAsyncBulk => (["multimem", "cp", "async", "bulk"], [
      .sem, .scope, .space, .flag .bulkGroup, .flag .cpMask])
  | .multimemCpReduceAsyncBulk => (["multimem", "cp", "reduce", "async", "bulk"], [
      .sem, .scope, .space, .flag .bulkGroup, .red, .flag .noftz])
  | .prefetch => (["prefetch"], [
      .space, .evict, .flag .l1, .flag .l2, .flag .l1Size32, .flag .validAddr, .flag .tensormap])
  | .prefetchu => (["prefetchu"], [.flag .l1])
  | .applypriority => (["applypriority"], [.space, .evict])
  | .applypriorityAsyncBulk => (["applypriority", "async", "bulk"], [
      .space, .flag .bulkGroup, .evict])
  | .applypriorityAsyncBulkTensor => (["applypriority", "async", "bulk", "tensor"], [
      .dim, .space, .flag .bulkGroup, .evict, .flag .tile, .flag .tileGather4, .flag .im2col,
      .flag .im2colW, .flag .im2colW128, .flag .overrideGlobalAddress, .flag .overrideGlobalDim,
      .flag .overrideGlobalDimStride])
  | .discard => (["discard"], [.space, .flag .l2])
  | .createpolicy => (["createpolicy"], [
      .space, .evict, .flag .range, .flag .fractional, .flag .cvt, .flag .fabric, .flag .l2])
  | .isspacep => (["isspacep"], [.space])
  | .cvta => (["cvta"], [.space, .flag .to])
  | .cvt => (["cvt"], [
      .rnd, .irnd, .flag .ftz, .flag .sat, .flag .relu, .flag .satfinite, .flag .pzo,
      .flag .scaledN1Ue8m0, .flag .scaledN2Ue8m0])
  | .cvtPack => (["cvt", "pack"], [.flag .sat])
  | .mapa => (["mapa"], [.space])
  | .getctarank => (["getctarank"], [.space])
  | .cpAsync => (["cp", "async"], [
      .cop, .space, .flag .l2CacheHint, .flag .l2Size64, .flag .l2Size128, .flag .l2Size256])
  | .cpAsyncCommitGroup => (["cp", "async", "commit_group"], [])
  | .cpAsyncWaitGroup => (["cp", "async", "wait_group"], [])
  | .cpAsyncWaitAll => (["cp", "async", "wait_all"], [])
  | .cpAsyncMbarrierArrive => (["cp", "async", "mbarrier", "arrive"], [.flag .noinc, .space])
  | .cpAsyncBulk => (["cp", "async", "bulk"], [
      .sem, .scope, .space, .flag .mbarrierCompleteTxBytes, .flag .mbarrierReportDisabled,
      .flag .mbarrierReportValidityPer16bytes80000000, .flag .mbarrierReportValidityPer16bytes8000,
      .flag .mbarrierReportValidityPer16bytes80, .flag .mbarrierReportValidityPer16bytes8,
      .flag .mbarrierReportValidityPerElementFf, .flag .bulkGroup, .flag .multicastCluster,
      .flag .multicastCluster16b, .flag .multicastCluster32b, .flag .l2CacheHint, .flag .ignoreOob,
      .flag .cpMask])
  | .cpReduceAsyncBulk => (["cp", "reduce", "async", "bulk"], [
      .sem, .scope, .space, .flag .mbarrierCompleteTxBytes, .flag .bulkGroup, .flag .l2CacheHint,
      .red, .flag .noftz])
  | .cpAsyncBulkPrefetch => (["cp", "async", "bulk", "prefetch"], [
      .flag .l2, .space, .flag .l2CacheHint, .evict])
  | .cpAsyncBulkTensor => (["cp", "async", "bulk", "tensor"], [
      .dim, .space, .flag .tile, .flag .tileGather4, .flag .tileScatter4, .flag .im2col,
      .flag .im2colW, .flag .im2colW128, .flag .im2colNoOffs, .flag .im2colNoOffsW,
      .flag .mbarrierCompleteTxBytes, .flag .mbarrierReportDisabled,
      .flag .mbarrierReportValidityPer16bytes80000000, .flag .mbarrierReportValidityPer16bytes8000,
      .flag .mbarrierReportValidityPer16bytes80, .flag .mbarrierReportValidityPer16bytes8,
      .flag .mbarrierReportValidityPerElementFf, .flag .bulkGroup, .flag .multicastCluster,
      .flag .multicastCluster16b, .flag .multicastCluster32b, .flag .ctaGroup1, .flag .ctaGroup2,
      .flag .l2CacheHint, .flag .overrideGlobalAddress, .flag .overrideGlobalDim,
      .flag .overrideGlobalDimStride])
  | .cpReduceAsyncBulkTensor => (["cp", "reduce", "async", "bulk", "tensor"], [
      .dim, .space, .red, .flag .tile, .flag .im2colNoOffs, .flag .im2colNoOffsW, .flag .bulkGroup,
      .flag .l2CacheHint, .flag .overrideGlobalAddress, .flag .overrideGlobalDim,
      .flag .overrideGlobalDimStride])
  | .cpAsyncBulkPrefetchTensor => (["cp", "async", "bulk", "prefetch", "tensor"], [
      .dim, .flag .l2, .space, .flag .tile, .flag .tileGather4, .flag .im2col, .flag .im2colW,
      .flag .im2colW128, .flag .l2CacheHint, .evict, .flag .overrideGlobalAddress,
      .flag .overrideGlobalDim, .flag .overrideGlobalDimStride])
  | .cpAsyncBulkCommitGroup => (["cp", "async", "bulk", "commit_group"], [])
  | .cpAsyncBulkWaitGroup => (["cp", "async", "bulk", "wait_group"], [.flag .read])
  | .tensormapReplace => (["tensormap", "replace"], [
      .space, .flag .tile, .flag .globalAddress, .flag .rank, .flag .boxDim, .flag .globalDim,
      .flag .globalStride, .flag .elementStride, .flag .elemtype, .flag .interleaveLayout,
      .flag .swizzleMode, .flag .swizzleAtomicity, .flag .fillMode])
  | .spcompress => (["spcompress"], [
      .num, .flag .sp1of2, .flag .sp1of4, .flag .sp1of8, .flag .sp1of16, .flag .sp2of4,
      .flag .sp2of8, .flag .sp2of16, .flag .sp4of8, .flag .sp4of16])
  | .spdecompress => (["spdecompress"], [
      .num, .flag .sp1of2, .flag .sp1of4, .flag .sp1of8, .flag .sp1of16, .flag .sp2of4,
      .flag .sp2of8, .flag .sp2of16, .flag .sp4of8, .flag .sp4of16])
  | .fabricTryGet => (["fabric", "try_get"], [
      .flag .async, .flag .tensor, .dim, .space, .flag .tiled, .flag .mbarrierCompleteTxBytes,
      .flag .mbarrierReportFabric, .flag .l2CacheHint, .flag .overrideGlobalAddress,
      .flag .overrideGlobalDim, .flag .overrideGlobalDimStride, .sem, .scope])
  | .fabricTryPut => (["fabric", "try_put"], [
      .flag .async, .flag .tensor, .flag .multimem, .dim, .space, .flag .tiled,
      .flag .mbarrierCompleteTx16B, .flag .mbarrierReportFabric, .flag .countedBytes,
      .flag .l2CacheHint, .flag .cpMask, .flag .overrideGlobalAddress, .flag .overrideGlobalDim,
      .flag .overrideGlobalDimStride, .sem, .scope])
  | .fabricTryRed => (["fabric", "try_red"], [
      .flag .async, .flag .tensor, .flag .multimem, .dim, .space, .flag .tiled,
      .flag .mbarrierCompleteTx16B, .flag .mbarrierReportFabric, .flag .countedBytes,
      .flag .l2CacheHint, .flag .overrideGlobalAddress, .flag .overrideGlobalDim,
      .flag .overrideGlobalDimStride, .sem, .scope, .red])
  | .fabricTryPullred => (["fabric", "try_pullred"], [
      .flag .async, .flag .multimem, .space, .flag .mbarrierCompleteTxBytes,
      .flag .mbarrierReportFabric, .sem, .scope, .red, .flag .accF16, .flag .accF32, .flag .sync])
  | .fabricTryAtom => (["fabric", "try_atom"], [
      .flag .async, .space, .flag .mbarrierCompleteTx16B, .flag .mbarrierReportFabric,
      .flag .l2CacheHint, .sem, .scope, .red])
  | .fabricSubmit => (["fabric", "submit"], [.flag .opRestrictFetching])
  | .fabricWait => (["fabric", "wait"], [.flag .syncRestrictReads])
  | .tex => (["tex"], [
      .dim, .flag .a1d, .flag .a2d, .flag .cube, .flag .acube, .flag .geom2dms, .flag .a2dms,
      .flag .base, .flag .level, .flag .grad, .vec])
  | .tld4 => (["tld4"], [
      .flag .r, .flag .g, .flag .b, .flag .a, .dim, .flag .a2d, .flag .cube, .flag .acube, .vec])
  | .txq => (["txq"], [
      .flag .level, .flag .width, .flag .height, .flag .depth, .flag .channelDataType,
      .flag .channelOrder, .flag .normalizedCoords, .flag .arraySize, .flag .numMipmapLevels,
      .flag .numSamples, .flag .forceUnnormalizedCoords, .flag .filterMode, .flag .addrMode0,
      .flag .addrMode1, .flag .addrMode2, .flag .memoryLayout])
  | .istypep => (["istypep"], [])
  | .suld => (["suld"], [
      .flag .b, .dim, .flag .a1d, .flag .a2d, .cop, .vec, .flag .trap, .flag .clamp, .flag .zero])
  | .sust => (["sust"], [
      .flag .b, .flag .p, .dim, .flag .a1d, .flag .a2d, .cop, .vec, .flag .trap, .flag .clamp,
      .flag .zero])
  | .sured => (["sured"], [.flag .b, .flag .p, .red, .dim, .flag .trap, .flag .clamp, .flag .zero])
  | .suq => (["suq"], [
      .flag .width, .flag .height, .flag .depth, .flag .channelDataType, .flag .channelOrder,
      .flag .normalizedCoords, .flag .arraySize, .flag .numMipmapLevels, .flag .numSamples,
      .flag .forceUnnormalizedCoords, .flag .filterMode, .flag .addrMode0, .flag .addrMode1,
      .flag .addrMode2, .flag .memoryLayout])
  | .bra => (["bra"], [.flag .uni])
  | .brxIdx => (["brx", "idx"], [.flag .uni])
  | .call => (["call"], [.flag .uni])
  | .ret => (["ret"], [.flag .uni])
  | .exit => (["exit"], [])
  | .barSync => (["bar", "sync"], [.scope, .flag .aligned])
  | .barArrive => (["bar", "arrive"], [.scope, .flag .aligned])
  | .barRed => (["bar", "red"], [.scope, .flag .popc, .bool, .flag .aligned])
  | .barrierSync => (["barrier", "sync"], [.scope, .flag .aligned])
  | .barrierArrive => (["barrier", "arrive"], [.scope, .flag .aligned])
  | .barrierRed => (["barrier", "red"], [.scope, .flag .popc, .bool, .flag .aligned])
  | .barWarpSync => (["bar", "warp", "sync"], [])
  | .barrierClusterArrive => (["barrier", "cluster", "arrive"], [.sem, .flag .aligned])
  | .barrierClusterWait => (["barrier", "cluster", "wait"], [.sem, .flag .aligned])
  | .membar => (["membar"], [
      .scope, .flag .gl, .flag .proxy, .flag .aliasProxy, .flag .async, .space])
  | .fence => (["fence"], [
      .sem, .scope, .space, .flag .proxy, .flag .aliasProxy, .flag .async, .flag .asyncGeneric,
      .flag .tensormapGeneric, .flag .genericFabric, .flag .fabricGeneric, .flag .fabricFabric,
      .flag .syncRestrictSharedCluster, .flag .syncRestrictSharedCta,
      .flag .syncRestrictSharedClusterRead, .flag .mbarrierInit])
  | .atom => (["atom"], [.sem, .scope, .space, .red, .flag .noftz, .flag .l2CacheHint, .vec])
  | .red => (["red"], [.sem, .scope, .space, .red, .flag .noftz, .flag .l2CacheHint, .vec])
  | .redAsync => (["red", "async"], [.sem, .scope, .space, .flag .mbarrierCompleteTxBytes, .red])
  | .vote => (["vote"], [.flag .all, .flag .any, .flag .uni, .flag .ballot])
  | .voteSync => (["vote", "sync"], [.flag .all, .flag .any, .flag .uni, .flag .ballot])
  | .matchSync => (["match"], [.flag .any, .flag .all, .flag .sync])
  | .activemask => (["activemask"], [])
  | .reduxSync => (["redux", "sync"], [.red, .flag .abs, .flag .nan])
  | .griddepcontrol => (["griddepcontrol"], [.flag .launchDependents, .flag .wait])
  | .electSync => (["elect", "sync"], [])
  | .mbarrierInit => (["mbarrier", "init"], [.flag .layoutV0, .flag .layoutV1, .space])
  | .mbarrierInval => (["mbarrier", "inval"], [.space])
  | .mbarrierExpectTx => (["mbarrier", "expect_tx"], [
      .sem, .scope, .space, .flag .multicastCluster32b])
  | .mbarrierCompleteTx => (["mbarrier", "complete_tx"], [
      .sem, .scope, .space, .flag .multicastCluster32b])
  | .mbarrierArrive => (["mbarrier", "arrive"], [
      .flag .expectTx, .flag .noComplete, .sem, .scope, .space, .flag .multicastCluster32b])
  | .mbarrierArriveDrop => (["mbarrier", "arrive_drop"], [
      .flag .expectTx, .flag .noComplete, .sem, .scope, .space, .flag .multicastCluster32b])
  | .mbarrierTestWait => (["mbarrier", "test_wait"], [
      .flag .parity, .flag .phaseTypePrimary, .flag .phaseTypeConditional, .sem, .scope, .space])
  | .mbarrierTryWait => (["mbarrier", "try_wait"], [
      .flag .parity, .flag .phaseTypePrimary, .flag .phaseTypeConditional, .sem, .scope, .space])
  | .mbarrierPendingCount => (["mbarrier", "pending_count"], [.flag .layoutV0])
  | .mbarrierCheckLayout => (["mbarrier", "check_layout"], [
      .flag .layoutV0, .flag .layoutV1, .space])
  | .tensormapCpFenceproxy => (["tensormap", "cp_fenceproxy"], [
      .space, .flag .tensormapGeneric, .sem, .scope, .flag .sync, .flag .aligned])
  | .clusterlaunchcontrolTryCancel => (["clusterlaunchcontrol", "try_cancel"], [
      .flag .async, .space, .flag .mbarrierCompleteTxBytes, .flag .multicastClusterAll])
  | .clusterlaunchcontrolQueryCancel => (["clusterlaunchcontrol", "query_cancel"], [
      .flag .isCanceled, .flag .getFirstCtaid, .flag .getFirstCtaidX, .flag .getFirstCtaidY,
      .flag .getFirstCtaidZ, .vec])
  | .wmmaLoad => (["wmma", "load"], [
      .flag .a, .flag .b, .flag .c, .flag .sync, .flag .aligned, .flag .row, .flag .col, .shape,
      .space])
  | .wmmaStore => (["wmma", "store"], [
      .flag .d, .flag .sync, .flag .aligned, .flag .row, .flag .col, .shape, .space])
  | .wmmaMma => (["wmma", "mma"], [
      .flag .sync, .flag .aligned, .flag .row, .flag .col, .shape, .rnd, .flag .satfinite, .bool,
      .flag .popc])
  | .mma => (["mma"], [
      .flag .sync, .flag .aligned, .shape, .flag .row, .flag .col, .flag .satfinite, .bool,
      .flag .popc, .flag .kindF16, .flag .kindTf32, .flag .kindF8f6f4, .flag .kindI8,
      .flag .kindTi16, .flag .kindMxf4, .flag .kindMxf4nvf4, .flag .kindMxf8f6f4,
      .flag .blockScale, .flag .scaleVec1X, .flag .scaleVec2X, .flag .scaleVec4X, .flag .sp,
      .flag .spOrderedMetadata])
  | .ldmatrix => (["ldmatrix"], [.flag .sync, .flag .aligned, .shape, .num, .flag .trans, .space])
  | .stmatrix => (["stmatrix"], [.flag .sync, .flag .aligned, .shape, .num, .flag .trans, .space])
  | .movmatrix => (["movmatrix"], [.flag .sync, .flag .aligned, .shape, .flag .trans])
  | .wgmmaMmaAsync => (["wgmma", "mma_async"], [
      .flag .sp, .flag .sync, .flag .aligned, .shape, .flag .satfinite, .bool, .flag .popc])
  | .wgmmaFence => (["wgmma", "fence"], [.flag .sync, .flag .aligned])
  | .wgmmaCommitGroup => (["wgmma", "commit_group"], [.flag .sync, .flag .aligned])
  | .wgmmaWaitGroup => (["wgmma", "wait_group"], [.flag .sync, .flag .aligned])
  | .tcgen05Alloc => (["tcgen05", "alloc"], [
      .flag .exclusive, .flag .ctaGroup1, .flag .ctaGroup2, .flag .sync, .flag .aligned, .space])
  | .tcgen05Dealloc => (["tcgen05", "dealloc"], [
      .flag .exclusive, .flag .ctaGroup1, .flag .ctaGroup2, .flag .sync, .flag .aligned])
  | .tcgen05RelinquishAllocPermit => (["tcgen05", "relinquish_alloc_permit"], [
      .flag .ctaGroup1, .flag .ctaGroup2, .flag .sync, .flag .aligned])
  | .tcgen05Ld => (["tcgen05", "ld"], [
      .red, .flag .spcompress, .flag .sync, .flag .aligned, .flag .shape16x64b,
      .flag .shape16x128b, .flag .shape16x256b, .flag .shape32x32b, .flag .shape16x32bx2, .num,
      .flag .pack16b, .flag .abs, .flag .nan, .flag .sp2of4])
  | .tcgen05St => (["tcgen05", "st"], [
      .flag .sync, .flag .aligned, .flag .shape16x64b, .flag .shape16x128b, .flag .shape16x256b,
      .flag .shape32x32b, .flag .shape16x32bx2, .num, .flag .unpack16b])
  | .tcgen05WaitLd => (["tcgen05", "wait::ld"], [.flag .sync, .flag .aligned])
  | .tcgen05WaitSt => (["tcgen05", "wait::st"], [.flag .sync, .flag .aligned])
  | .tcgen05Cp => (["tcgen05", "cp"], [
      .flag .ctaGroup1, .flag .ctaGroup2, .flag .shape128x256b, .flag .shape4x256b,
      .flag .shape128x128b, .flag .shape64x128b, .flag .shape32x128b, .flag .warpx2Rows02x13,
      .flag .warpx2Rows01x23, .flag .warpx4, .flag .b8x16, .flag .b6x16P32, .flag .b4x16P64])
  | .tcgen05Shift => (["tcgen05", "shift"], [.flag .ctaGroup1, .flag .ctaGroup2, .flag .down])
  | .tcgen05Mma => (["tcgen05", "mma"], [
      .flag .sp, .flag .ws, .flag .ctaGroup1, .flag .ctaGroup2, .flag .kindF16, .flag .kindTf32,
      .flag .kindF8f6f4, .flag .kindI8, .flag .kindTi16, .flag .kindMxf4, .flag .kindMxf4nvf4,
      .flag .kindMxf8f6f4, .flag .blockScale, .flag .scaleVec1X, .flag .scaleVec2X,
      .flag .scaleVec4X, .flag .block16, .flag .block32, .flag .ashift, .flag .decompressLutB,
      .flag .collectorAFill, .flag .collectorAUse, .flag .collectorALastuse,
      .flag .collectorADiscard, .flag .collectorBFill, .flag .collectorBUse,
      .flag .collectorBLastuse, .flag .collectorBDiscard, .flag .collectorB0Fill,
      .flag .collectorB0Use, .flag .collectorB0Lastuse, .flag .collectorB0Discard,
      .flag .collectorB1Fill, .flag .collectorB1Use, .flag .collectorB1Lastuse,
      .flag .collectorB1Discard, .flag .collectorB2Fill, .flag .collectorB2Use,
      .flag .collectorB2Lastuse, .flag .collectorB2Discard, .flag .collectorB3Fill,
      .flag .collectorB3Use, .flag .collectorB3Lastuse, .flag .collectorB3Discard])
  | .tcgen05FenceBeforeThreadSync => (["tcgen05", "fence::before_thread_sync"], [])
  | .tcgen05FenceAfterThreadSync => (["tcgen05", "fence::after_thread_sync"], [])
  | .tcgen05Commit => (["tcgen05", "commit"], [
      .flag .ctaGroup1, .flag .ctaGroup2, .flag .mbarrierArriveOne,
      .flag .syncRestrictSharedReadMmaA, .space, .flag .multicastCluster,
      .flag .multicastCluster16b, .flag .multicastCluster32b])
  | .stacksave => (["stacksave"], [])
  | .stackrestore => (["stackrestore"], [])
  | .alloca => (["alloca"], [])
  | .vadd => (["vadd"], [.flag .sat, .red, .flag .clamp, .flag .wrap])
  | .vsub => (["vsub"], [.flag .sat, .red, .flag .clamp, .flag .wrap])
  | .vabsdiff => (["vabsdiff"], [.flag .sat, .red, .flag .clamp, .flag .wrap])
  | .vmin => (["vmin"], [.flag .sat, .red, .flag .clamp, .flag .wrap])
  | .vmax => (["vmax"], [.flag .sat, .red, .flag .clamp, .flag .wrap])
  | .vshl => (["vshl"], [.flag .sat, .red, .flag .clamp, .flag .wrap])
  | .vshr => (["vshr"], [.flag .sat, .red, .flag .clamp, .flag .wrap])
  | .vmad => (["vmad"], [
      .flag .sat, .red, .flag .clamp, .flag .wrap, .flag .po, .flag .shr7, .flag .shr15])
  | .vset => (["vset"], [.flag .sat, .red, .flag .clamp, .flag .wrap, .cmp])
  | .vadd2 => (["vadd2"], [.flag .sat, .red, .flag .clamp, .flag .wrap])
  | .vsub2 => (["vsub2"], [.flag .sat, .red, .flag .clamp, .flag .wrap])
  | .vavrg2 => (["vavrg2"], [.flag .sat, .red, .flag .clamp, .flag .wrap])
  | .vabsdiff2 => (["vabsdiff2"], [.flag .sat, .red, .flag .clamp, .flag .wrap])
  | .vmin2 => (["vmin2"], [.flag .sat, .red, .flag .clamp, .flag .wrap])
  | .vmax2 => (["vmax2"], [.flag .sat, .red, .flag .clamp, .flag .wrap])
  | .vset2 => (["vset2"], [.flag .sat, .red, .flag .clamp, .flag .wrap, .cmp])
  | .vadd4 => (["vadd4"], [.flag .sat, .red, .flag .clamp, .flag .wrap])
  | .vsub4 => (["vsub4"], [.flag .sat, .red, .flag .clamp, .flag .wrap])
  | .vavrg4 => (["vavrg4"], [.flag .sat, .red, .flag .clamp, .flag .wrap])
  | .vabsdiff4 => (["vabsdiff4"], [.flag .sat, .red, .flag .clamp, .flag .wrap])
  | .vmin4 => (["vmin4"], [.flag .sat, .red, .flag .clamp, .flag .wrap])
  | .vmax4 => (["vmax4"], [.flag .sat, .red, .flag .clamp, .flag .wrap])
  | .vset4 => (["vset4"], [.flag .sat, .red, .flag .clamp, .flag .wrap, .cmp])
  | .brkpt => (["brkpt"], [])
  | .nanosleep => (["nanosleep"], [])
  | .pmevent => (["pmevent"], [.flag .mask])
  | .trap => (["trap"], [])
  | .setmaxnreg => (["setmaxnreg"], [.flag .inc, .flag .dec, .flag .sync, .flag .aligned])

namespace Op

def all : List Op :=
  [.add, .sub, .mul, .mad, .clmad, .mul24, .mad24, .sad, .div, .rem, .abs, .neg, .min, .max,
    .popc, .clz, .bfind, .fns, .brev, .bfe, .bfi, .szext, .bmsk, .dp4a, .dp2a, .addc, .subc, .madc,
    .testp, .copysign, .fma, .rcp, .sqrt, .rsqrt, .sin, .cos, .lg2, .ex2, .tanh, .set, .setp,
    .selp, .slct, .and, .or, .xor, .not, .cnot, .lop3, .shf, .shl, .shr, .mov, .shfl, .shflSync,
    .prmt, .ld, .ldu, .st, .stAsync, .stBulk, .multimemLdReduce, .multimemSt, .multimemRed,
    .multimemStAsync, .multimemRedAsync, .multimemCpAsyncBulk, .multimemCpReduceAsyncBulk,
    .prefetch, .prefetchu, .applypriority, .applypriorityAsyncBulk, .applypriorityAsyncBulkTensor,
    .discard, .createpolicy, .isspacep, .cvta, .cvt, .cvtPack, .mapa, .getctarank, .cpAsync,
    .cpAsyncCommitGroup, .cpAsyncWaitGroup, .cpAsyncWaitAll, .cpAsyncMbarrierArrive, .cpAsyncBulk,
    .cpReduceAsyncBulk, .cpAsyncBulkPrefetch, .cpAsyncBulkTensor, .cpReduceAsyncBulkTensor,
    .cpAsyncBulkPrefetchTensor, .cpAsyncBulkCommitGroup, .cpAsyncBulkWaitGroup, .tensormapReplace,
    .spcompress, .spdecompress, .fabricTryGet, .fabricTryPut, .fabricTryRed, .fabricTryPullred,
    .fabricTryAtom, .fabricSubmit, .fabricWait, .tex, .tld4, .txq, .istypep, .suld, .sust, .sured,
    .suq, .bra, .brxIdx, .call, .ret, .exit, .barSync, .barArrive, .barRed, .barrierSync,
    .barrierArrive, .barrierRed, .barWarpSync, .barrierClusterArrive, .barrierClusterWait, .membar,
    .fence, .atom, .red, .redAsync, .vote, .voteSync, .matchSync, .activemask, .reduxSync,
    .griddepcontrol, .electSync, .mbarrierInit, .mbarrierInval, .mbarrierExpectTx,
    .mbarrierCompleteTx, .mbarrierArrive, .mbarrierArriveDrop, .mbarrierTestWait, .mbarrierTryWait,
    .mbarrierPendingCount, .mbarrierCheckLayout, .tensormapCpFenceproxy,
    .clusterlaunchcontrolTryCancel, .clusterlaunchcontrolQueryCancel, .wmmaLoad, .wmmaStore,
    .wmmaMma, .mma, .ldmatrix, .stmatrix, .movmatrix, .wgmmaMmaAsync, .wgmmaFence,
    .wgmmaCommitGroup, .wgmmaWaitGroup, .tcgen05Alloc, .tcgen05Dealloc,
    .tcgen05RelinquishAllocPermit, .tcgen05Ld, .tcgen05St, .tcgen05WaitLd, .tcgen05WaitSt,
    .tcgen05Cp, .tcgen05Shift, .tcgen05Mma, .tcgen05FenceBeforeThreadSync,
    .tcgen05FenceAfterThreadSync, .tcgen05Commit, .stacksave, .stackrestore, .alloca, .vadd, .vsub,
    .vabsdiff, .vmin, .vmax, .vshl, .vshr, .vmad, .vset, .vadd2, .vsub2, .vavrg2, .vabsdiff2,
    .vmin2, .vmax2, .vset2, .vadd4, .vsub4, .vavrg4, .vabsdiff4, .vmin4, .vmax4, .vset4, .brkpt,
    .nanosleep, .pmevent, .trap, .setmaxnreg]

theorem complete (o : Op) : o ∈ all := by cases o <;> decide

def parts (o : Op) : List String := o.spec.1
def kinds (o : Op) : List QKind := o.spec.2
def spelling (o : Op) : String := ".".intercalate o.parts

/-- Opcodes by their first part. -/
def byHead : Std.HashMap String (List Op) :=
  all.foldl (fun m o => m.insert o.parts.head! (o :: (m.getD o.parts.head! []))) {}

end Op

structure Instr where
  op : Op
  quals : List Qual
  /-- Type qualifiers in written order: `cvt.f32.s32` has `[.f32, .s32]`. -/
  types : List Ty
  args : Array Operand
  deriving Repr, Inhabited

namespace Instr

def has (i : Instr) (f : Flag) : Bool := i.quals.contains (.flag f)
def rnd? (i : Instr) : Option Rnd := i.quals.findSome? fun | .rnd r => some r | _ => none
def irnd? (i : Instr) : Option IntRnd := i.quals.findSome? fun | .irnd r => some r | _ => none
def cmp? (i : Instr) : Option CmpOp := i.quals.findSome? fun | .cmp c => some c | _ => none
def bool? (i : Instr) : Option BoolOp := i.quals.findSome? fun | .bool b => some b | _ => none
def sem? (i : Instr) : Option MemSem := i.quals.findSome? fun | .sem s => some s | _ => none
def scope? (i : Instr) : Option Scope := i.quals.findSome? fun | .scope s => some s | _ => none
def cop? (i : Instr) : Option CacheOp := i.quals.findSome? fun | .cop c => some c | _ => none
def red? (i : Instr) : Option RedOp := i.quals.findSome? fun | .red o => some o | _ => none
def vec? (i : Instr) : Option Vec := i.quals.findSome? fun | .vec v => some v | _ => none
def dim? (i : Instr) : Option Nat := i.quals.findSome? fun | .dim n => some n | _ => none
def num? (i : Instr) : Option Nat := i.quals.findSome? fun | .num n => some n | _ => none
def shape? (i : Instr) : Option (Nat × Nat × Nat) :=
  i.quals.findSome? fun | .shape m n k => some (m, n, k) | _ => none
/-- State spaces in written order (`cp.async.bulk.shared::cluster.global`: destination first). -/
def spaces (i : Instr) : List Space := i.quals.filterMap fun | .space s => some s | _ => none
def space? (i : Instr) : Option Space := i.spaces.head?
def evicts (i : Instr) : List Evict := i.quals.filterMap fun | .evict e => some e | _ => none

end Instr

/-- Remove the parts `ps` from `xs`, matching each in order at its first occurrence; `none`
if some part is missing. -/
def removeInOrder : List String → List String → Option (List String)
  | [], xs => some xs
  | p :: ps, xs =>
    match xs.idxOf? p with
    | some k => (removeInOrder ps (xs.drop (k + 1))).map ((xs.take k) ++ ·)
    | none => none

/-- The typed form of an instruction as written. -/
def decode (r : RawInstr) : Except String Instr := do
  let mods := r.mods.toList
  let candidates := (Op.byHead.getD r.op []).filterMap fun o =>
    (removeInOrder o.parts.tail mods).map (o, ·)
  let some (op, rest) := candidates.foldl (fun best c =>
      match best with
      | some b => if c.1.parts.length > b.1.parts.length then some c else some b
      | none => some c) none
    | throw s!"unknown instruction {r.op}"
  let mut quals := #[]
  let mut types := #[]
  for m in rest do
    match op.kinds.findSome? (Qual.parse? · m) with
    | some q => quals := quals.push q
    | none =>
      match Keyword.parse? (α := Ty) m with
      | some t => types := types.push t
      | none => throw s!"{op.spelling} does not take .{m}"
  return { op, quals := quals.toList, types := types.toList, args := r.args }

mutual
def Stmt.decode : Stmt RawInstr → Except String (Stmt Instr)
  | .instr g i l =>
    match PTX.decode i with
    | .ok d => return .instr g d l
    | .error e => throw s!"line {l}: {e}"
  | .block b => return .block (← Stmt.decodeList b)
  | .label n => return .label n
  | .decl d => return .decl d
  | .pragma s => return .pragma s
  | .loc f l c => return .loc f l c
  | .proto n p => return .proto n p
  | .targets n c ls => return .targets n c ls

def Stmt.decodeList : List (Stmt RawInstr) → Except String (List (Stmt Instr))
  | [] => return []
  | s :: ss => return (← Stmt.decode s) :: (← Stmt.decodeList ss)
end

def Function.decode (f : Function RawInstr) : Except String (Function Instr) := do
  let body ← match f.body with
    | some b => some <$> Stmt.decodeList b
    | none => pure none
  return { f with body }

def Module.decode (m : Module RawInstr) : Except String (Module Instr) := do
  let items ← m.items.mapM fun
    | .func f => Item.func <$> f.decode
    | .var d => pure (.var d)
    | .file n s => pure (.file n s)
    | .alias a b => pure (.alias a b)
    | .pragma s => pure (.pragma s)
  return { m with items }

end NV.PTX
