// Tensor maps (`cuTensorMapEncode*`) of the emulated driver. Included by cuda.c.
//
// The 128-byte descriptor is the model's own encoding (`NV.Async.TensorMap.encode`), which the
// TMA model decodes wherever a kernel finds it; the driver's checks are the model's
// (`TensorMap.invalid?`). The arguments go to the Lean side packed as `Device.tensorMapOf` reads
// them.

#define TMAP_ARGS 184

static void put32(unsigned char *b, size_t off, uint32_t v) { memcpy(b + off, &v, 4); }
static void put64(unsigned char *b, size_t off, uint64_t v) { memcpy(b + off, &v, 8); }

// The descriptor for the packed arguments, written to `map`; the model's refusal noted.
static CUresult encodeMap(CUtensorMap *map, unsigned char *args) {
  if (!map) return CUDA_ERROR_INVALID_VALUE;
  pthread_mutex_lock(&lock);
  enterLean();
  lean_object *a = lean_alloc_sarray(1, TMAP_ARGS, TMAP_ARGS);
  memcpy(lean_sarray_cptr(a), args, TMAP_ARGS);
  lean_object *pair = nvgpu_tensor_map(a);
  lean_object *bytes = lean_ctor_get(pair, 0), *msg = lean_ctor_get(pair, 1);
  CUresult r = CUDA_SUCCESS;
  if (lean_string_size(msg) > 1) {
    note("%s", lean_string_cstr(msg));
    r = CUDA_ERROR_INVALID_VALUE;
  } else
    memcpy(map, lean_sarray_cptr(bytes), sizeof *map < lean_sarray_size(bytes) ? sizeof *map : lean_sarray_size(bytes));
  lean_dec(pair);
  pthread_mutex_unlock(&lock);
  return r;
}

// The arguments both encoders share.
static void packCommon(unsigned char *b, int kind, CUtensorMapDataType dt, cuuint32_t rank, void *addr,
                       const cuuint64_t *dims, const cuuint64_t *strides, const cuuint32_t *estrides,
                       CUtensorMapInterleave il, CUtensorMapSwizzle sw, CUtensorMapL2promotion l2,
                       CUtensorMapFloatOOBfill oob) {
  memset(b, 0, TMAP_ARGS);
  put32(b, 0, (uint32_t)kind);
  put32(b, 4, (uint32_t)dt);
  put32(b, 8, rank);
  put32(b, 12, (uint32_t)il);
  put32(b, 16, (uint32_t)sw);
  put32(b, 20, (uint32_t)l2);
  put32(b, 24, (uint32_t)oob);
  put64(b, 32, (uint64_t)(uintptr_t)addr);
  for (cuuint32_t i = 0; i < rank && i < 5; i++) {
    put64(b, 40 + 8 * i, dims[i]);
    put32(b, 132 + 4 * i, estrides[i]);
    if (i + 1 < rank) put64(b, 80 + 8 * i, strides[i]);
  }
}

CUresult CUDAAPI cuTensorMapEncodeTiled(CUtensorMap *map, CUtensorMapDataType dt, cuuint32_t rank,
                                        void *addr, const cuuint64_t *dims, const cuuint64_t *strides,
                                        const cuuint32_t *box, const cuuint32_t *estrides,
                                        CUtensorMapInterleave il, CUtensorMapSwizzle sw,
                                        CUtensorMapL2promotion l2, CUtensorMapFloatOOBfill oob) {
  NEED_INIT();
  if (rank == 0 || rank > 5) {
    note("tensor map: rank %u must be 1 to 5", rank);
    return CUDA_ERROR_INVALID_VALUE;
  }
  unsigned char b[TMAP_ARGS];
  packCommon(b, 0, dt, rank, addr, dims, strides, estrides, il, sw, l2, oob);
  for (cuuint32_t i = 0; i < rank; i++) put32(b, 112 + 4 * i, box[i]);
  if (getenv("NVGPU_LEAN_VERBOSE")) {
    fprintf(stderr, "nvgpu-lean: tensor map: type %d rank %u swizzle %d interleave %d at %p, dims", (int)dt, rank,
            (int)sw, (int)il, addr);
    for (cuuint32_t i = 0; i < rank; i++) fprintf(stderr, " %llu", (unsigned long long)dims[i]);
    fprintf(stderr, ", strides");
    for (cuuint32_t i = 0; i + 1 < rank; i++) fprintf(stderr, " %llu", (unsigned long long)strides[i]);
    fprintf(stderr, ", box");
    for (cuuint32_t i = 0; i < rank; i++) fprintf(stderr, " %u", box[i]);
    fprintf(stderr, ", element strides");
    for (cuuint32_t i = 0; i < rank; i++) fprintf(stderr, " %u", estrides[i]);
    fprintf(stderr, "\n");
  }
  return encodeMap(map, b);
}

CUresult CUDAAPI cuTensorMapEncodeIm2col(CUtensorMap *map, CUtensorMapDataType dt, cuuint32_t rank,
                                         void *addr, const cuuint64_t *dims, const cuuint64_t *strides,
                                         const int *lower, const int *upper, cuuint32_t channels,
                                         cuuint32_t pixels, const cuuint32_t *estrides,
                                         CUtensorMapInterleave il, CUtensorMapSwizzle sw,
                                         CUtensorMapL2promotion l2, CUtensorMapFloatOOBfill oob) {
  NEED_INIT();
  if (rank < 3 || rank > 5) {
    note("tensor map: an im2col map has rank 3 to 5, not %u", rank);
    return CUDA_ERROR_INVALID_VALUE;
  }
  unsigned char b[TMAP_ARGS];
  packCommon(b, 1, dt, rank, addr, dims, strides, estrides, il, sw, l2, oob);
  for (cuuint32_t i = 0; i + 2 < rank; i++) {
    put32(b, 152 + 4 * i, (uint32_t)lower[i]);
    put32(b, 164 + 4 * i, (uint32_t)upper[i]);
  }
  put32(b, 176, channels);
  put32(b, 180, pixels);
  return encodeMap(map, b);
}

CUresult CUDAAPI cuTensorMapReplaceAddress(CUtensorMap *map, void *addr) {
  NEED_INIT();
  pthread_mutex_lock(&lock);
  enterLean();
  lean_object *m = lean_alloc_sarray(1, sizeof *map, sizeof *map);
  memcpy(lean_sarray_cptr(m), map, sizeof *map);
  lean_object *out = nvgpu_tensor_map_address(m, (uint64_t)(uintptr_t)addr);
  CUresult r = CUDA_SUCCESS;
  if (lean_sarray_size(out) == sizeof *map)
    memcpy(map, lean_sarray_cptr(out), sizeof *map);
  else {
    note("cuTensorMapReplaceAddress: not a tensor map this driver encoded");
    r = CUDA_ERROR_INVALID_VALUE;
  }
  lean_dec(out);
  pthread_mutex_unlock(&lock);
  return r;
}
