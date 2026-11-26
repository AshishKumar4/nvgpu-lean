// CUDA arrays and texture and surface objects (cuArrayCreate, cuTexObjectCreate, cuSurfObjectCreate):
// an array is device memory laid out in pitch-linear rows (a texture or surface instruction
// cannot tell it from the hardware's block-linear layout, `NV.Tex`), an object an entry of the
// device's table (`nvgpu_set_textures`), named by its index plus one as a kernel's handle.
// Included by cuda.c.

struct CUarray_st {
  CUdeviceptr base;
  size_t width, height, depth;  // texels; height and depth 0 where the array has fewer dimensions
  unsigned format, channels, flags;
  size_t pitch, slice;          // bytes from a row to the next, from a slice (or layer) to the next
  int device;
  int level;                    // a mipmapped array's level, whose memory the mipmapped array owns
};

// A mipmapped array: its levels, each an array of half the last one's extent (at least 1), laid
// out one after another in one allocation (`NV.Tex.Texture.level` recomputes the layout).
struct CUmipmappedArray_st {
  struct CUarray_st *levels;
  unsigned count;
};

// Bits of one channel of `format`, and its kind (0 unsigned, 1 signed, 2 float); 0 bits if unknown.
static unsigned formatBits(unsigned format, unsigned *kind) {
  switch (format) {
    case CU_AD_FORMAT_UNSIGNED_INT8: *kind = 0; return 8;
    case CU_AD_FORMAT_UNSIGNED_INT16: *kind = 0; return 16;
    case CU_AD_FORMAT_UNSIGNED_INT32: *kind = 0; return 32;
    case CU_AD_FORMAT_SIGNED_INT8: *kind = 1; return 8;
    case CU_AD_FORMAT_SIGNED_INT16: *kind = 1; return 16;
    case CU_AD_FORMAT_SIGNED_INT32: *kind = 1; return 32;
    case CU_AD_FORMAT_HALF: *kind = 2; return 16;
    case CU_AD_FORMAT_FLOAT: *kind = 2; return 32;
    default: *kind = 0; return 0;
  }
}

// A block-compressed format (CU_AD_FORMAT_BC1_UNORM to BC7_UNORM_SRGB): its BC number, 1 to 7,
// and its 4x4 blocks' bytes; 0 for any other format.
static unsigned bcOf(unsigned format, unsigned *blockBytes) {
  if (format < CU_AD_FORMAT_BC1_UNORM || format > CU_AD_FORMAT_BC7_UNORM_SRGB) return 0;
  static const unsigned bc[] = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7};
  unsigned n = bc[format - CU_AD_FORMAT_BC1_UNORM];
  *blockBytes = n == 1 || n == 4 ? 8 : 16;
  return n;
}

// A block-compressed format's texels (`NV.Tex.Texture.bc`): 8-bit channels of the format's kind
// (signed for BC4 and BC5 SNORM), BC6H's binary16 halves (three, read as four); the format goes
// in word 27.
static void bcTexels(unsigned format, unsigned bc, unsigned *bits, unsigned *kind, unsigned *chans) {
  static const unsigned bcChannels[] = {0, 4, 4, 4, 1, 2, 4, 4};
  *bits = bc == 6 ? 16 : 8;
  *kind = bc == 6 ? 2 : format == CU_AD_FORMAT_BC4_SNORM || format == CU_AD_FORMAT_BC5_SNORM;
  *chans = bcChannels[bc];
}

// An array's row pitch: its texels' bytes rounded up to 128; a block-compressed array's rows are
// of blocks.
static size_t arrayPitch(size_t w, unsigned format, unsigned channels) {
  unsigned kind, bb;
  if (bcOf(format, &bb)) return ((w + 3) / 4 * bb + 127) / 128 * 128;
  return (w * channels * formatBits(format, &kind) / 8 + 127) / 128 * 128;
}

// An array's rows a slice: a block-compressed array's are rows of 4x4 blocks.
static size_t arrayRows(const struct CUarray_st *a) {
  unsigned bb;
  size_t h = a->height ? a->height : 1;
  return bcOf(a->format, &bb) ? (h + 3) / 4 : h;
}

static CUresult arrayCreate(CUarray *out, size_t w, size_t h, size_t d, unsigned format, unsigned channels,
                            unsigned flags) {
  unsigned kind, bb;
  unsigned bits = formatBits(format, &kind);
  // Block-compressed arrays are 2D, of the format's channels (BC6H's three).
  static const unsigned bcChannels[] = {0, 4, 4, 4, 1, 2, 3, 4};
  unsigned bc = bcOf(format, &bb);
  if (bc && (channels != bcChannels[bc] || !h || d || flags)) return CUDA_ERROR_INVALID_VALUE;
  if (!out || !w || (!bc && (bits == 0 || (channels != 1 && channels != 2 && channels != 4))))
    return CUDA_ERROR_INVALID_VALUE;
  if (flags & ~(unsigned)(CUDA_ARRAY3D_LAYERED | CUDA_ARRAY3D_SURFACE_LDST | CUDA_ARRAY3D_CUBEMAP |
                          CUDA_ARRAY3D_TEXTURE_GATHER)) {
    note("CUDA array flags %#x are not modelled", flags);
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  struct CUarray_st *a = calloc(1, sizeof *a);
  if (!a) return CUDA_ERROR_OUT_OF_MEMORY;
  a->width = w;
  a->height = h;
  a->depth = d;
  a->format = format;
  a->channels = channels;
  a->flags = flags;
  a->pitch = arrayPitch(w, format, channels);
  a->slice = a->pitch * arrayRows(a);
  a->device = gcur;
  CUresult r = cuMemAlloc(&a->base, a->slice * (d ? d : 1));
  if (r != CUDA_SUCCESS) {
    free(a);
    return r;
  }
  *out = a;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuArrayCreate(CUarray *out, const CUDA_ARRAY_DESCRIPTOR *d) {
  NEED_INIT();
  if (!d) return CUDA_ERROR_INVALID_VALUE;
  return arrayCreate(out, d->Width, d->Height, 0, d->Format, d->NumChannels, 0);
}
CUresult CUDAAPI cuArray3DCreate(CUarray *out, const CUDA_ARRAY3D_DESCRIPTOR *d) {
  NEED_INIT();
  if (!d) return CUDA_ERROR_INVALID_VALUE;
  return arrayCreate(out, d->Width, d->Height, d->Depth, d->Format, d->NumChannels, d->Flags);
}
CUresult CUDAAPI cuArrayDestroy(CUarray a) {
  NEED_INIT();
  if (!a || a->level) return CUDA_ERROR_INVALID_HANDLE;
  CUresult r = cuMemFree(a->base);
  free(a);
  return r;
}
CUresult CUDAAPI cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR *d, CUarray a) {
  if (!d || !a) return CUDA_ERROR_INVALID_VALUE;
  d->Width = a->width;
  d->Height = a->height;
  d->Format = (CUarray_format)a->format;
  d->NumChannels = a->channels;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR *d, CUarray a) {
  if (!d || !a) return CUDA_ERROR_INVALID_VALUE;
  d->Width = a->width;
  d->Height = a->height;
  d->Depth = a->depth;
  d->Format = (CUarray_format)a->format;
  d->NumChannels = a->channels;
  d->Flags = a->flags;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMipmappedArrayCreate(CUmipmappedArray *out, const CUDA_ARRAY3D_DESCRIPTOR *d, unsigned int levels) {
  NEED_INIT();
  if (!out || !d || !d->Width || levels == 0) return CUDA_ERROR_INVALID_VALUE;
  const unsigned layers = CUDA_ARRAY3D_LAYERED | CUDA_ARRAY3D_CUBEMAP;
  if (d->Flags & ~(unsigned)(CUDA_ARRAY3D_SURFACE_LDST | CUDA_ARRAY3D_TEXTURE_GATHER | layers)) {
    note("mipmapped arrays with flags %#x are not modelled", d->Flags);
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  // A layered array's depth counts its layers, a cubemap's its faces: each level has them all.
  if ((d->Flags & layers) && !d->Depth) return CUDA_ERROR_INVALID_VALUE;
  if ((d->Flags & CUDA_ARRAY3D_CUBEMAP) &&
      (d->Width != d->Height || d->Depth % 6 || (!(d->Flags & CUDA_ARRAY3D_LAYERED) && d->Depth != 6)))
    return CUDA_ERROR_INVALID_VALUE;
  size_t big = d->Width > d->Height ? d->Width : d->Height;
  if (d->Depth > big && !(d->Flags & layers)) big = d->Depth;
  unsigned most = 1;
  while ((big >> most) > 0) most++;
  if (levels > most) return CUDA_ERROR_INVALID_VALUE;
  struct CUmipmappedArray_st *m = calloc(1, sizeof *m);
  struct CUarray_st *ls = calloc(levels, sizeof *ls);
  if (!m || !ls) { free(m); free(ls); return CUDA_ERROR_OUT_OF_MEMORY; }
  size_t total = 0;
  for (unsigned k = 0; k < levels; k++) {
    struct CUarray_st *a = &ls[k];
    a->width = d->Width >> k ? d->Width >> k : 1;
    a->height = d->Height ? (d->Height >> k ? d->Height >> k : 1) : 0;
    a->depth = (d->Flags & layers) ? d->Depth : d->Depth ? (d->Depth >> k ? d->Depth >> k : 1) : 0;
    a->format = d->Format;
    a->channels = d->NumChannels;
    a->flags = d->Flags;
    a->pitch = arrayPitch(a->width, d->Format, d->NumChannels);
    a->slice = a->pitch * (a->height ? a->height : 1);
    a->device = gcur;
    a->level = k + 1;
    a->base = total;
    total += a->slice * (a->depth ? a->depth : 1);
  }
  unsigned kind;
  if (formatBits(d->Format, &kind) == 0 || (d->NumChannels != 1 && d->NumChannels != 2 && d->NumChannels != 4)) {
    free(m); free(ls);
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUdeviceptr base;
  CUresult r = cuMemAlloc(&base, total);
  if (r != CUDA_SUCCESS) { free(m); free(ls); return r; }
  for (unsigned k = 0; k < levels; k++) ls[k].base += base;
  m->levels = ls;
  m->count = levels;
  *out = m;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMipmappedArrayGetLevel(CUarray *out, CUmipmappedArray m, unsigned int level) {
  NEED_INIT();
  if (!out || !m) return CUDA_ERROR_INVALID_VALUE;
  if (level >= m->count) return CUDA_ERROR_INVALID_VALUE;
  *out = &m->levels[level];
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMipmappedArrayDestroy(CUmipmappedArray m) {
  NEED_INIT();
  if (!m) return CUDA_ERROR_INVALID_HANDLE;
  CUresult r = cuMemFree(m->levels[0].base);
  free(m->levels);
  free(m);
  return r;
}

// The objects: 36 words each (`Device.setTextures`), per device; a destroyed one's words zero.
#define TEX_WORDS 36
static uint64_t *texTable[MAX_GPUS];
static size_t texCount[MAX_GPUS];

// The table handed to device `g` (the lock held).
static int texPublish(int g) {
  int was = gpuEnter(g);
  enterLean();
  size_t n = texCount[g] * TEX_WORDS * 8;
  lean_object *b = lean_alloc_sarray(1, n, n);
  memcpy(lean_sarray_cptr(b), texTable[g], n);
  G.lean = nvgpu_set_textures(G.lean, b);
  gpuLeave(was);
  return 1;
}

// A new object of words `w` on the calling thread's device: its handle.
static CUresult texAdd(const uint64_t *w, unsigned long long *handle) {
  pthread_mutex_lock(&lock);
  int g = gcur;
  uint64_t *t = realloc(texTable[g], (texCount[g] + 1) * TEX_WORDS * 8);
  if (!t) {
    pthread_mutex_unlock(&lock);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  texTable[g] = t;
  memcpy(t + texCount[g] * TEX_WORDS, w, TEX_WORDS * 8);
  *handle = ++texCount[g];
  texPublish(g);
  pthread_mutex_unlock(&lock);
  return CUDA_SUCCESS;
}

static CUresult texRemove(unsigned long long h) {
  pthread_mutex_lock(&lock);
  int g = gcur;
  CUresult r = CUDA_ERROR_INVALID_VALUE;
  if (h >= 1 && h <= texCount[g]) {
    memset(texTable[g] + (h - 1) * TEX_WORDS, 0, TEX_WORDS * 8);
    texPublish(g);
    r = CUDA_SUCCESS;
  }
  pthread_mutex_unlock(&lock);
  return r;
}

// The words of an object over resource `rd`: its memory and shape (an array's, a mipmapped array's
// first level's and its levels, linear memory's or pitched 2D memory's); 0 if not modelled.
static int texResource(const CUDA_RESOURCE_DESC *rd, uint64_t *w) {
  unsigned kind, bits, chans;
  memset(w, 0, TEX_WORDS * 8);
  CUmipmappedArray mip = rd->resType == CU_RESOURCE_TYPE_MIPMAPPED_ARRAY ? rd->res.mipmap.hMipmappedArray : NULL;
  switch (rd->resType) {
    case CU_RESOURCE_TYPE_MIPMAPPED_ARRAY:
    case CU_RESOURCE_TYPE_ARRAY: {
      CUarray a = mip ? &mip->levels[0] : rd->res.array.hArray;
      if (!a) return 0;
      bits = formatBits(a->format, &kind);
      chans = a->channels;
      unsigned bb, bc = bcOf(a->format, &bb);
      if (bc) {
        bcTexels(a->format, bc, &bits, &kind, &chans);
        w[27] = a->format;
      }
      int layered = (a->flags & CUDA_ARRAY3D_LAYERED) != 0, cube = (a->flags & CUDA_ARRAY3D_CUBEMAP) != 0;
      w[0] = a->base;
      w[1] = cube ? 2 : (a->depth && !layered) ? 3 : a->height ? 2 : 1;
      w[2] = a->width;
      w[3] = a->height ? a->height : 1;
      w[4] = a->depth && !layered && !cube ? a->depth : 1;
      w[5] = a->pitch;
      w[6] = a->slice;
      w[20] = layered || cube ? a->depth : 0;
      w[21] = cube;
      // A 1D layered array's layers are `pitch` apart (`NV.Tex.Texture.layers`).
      if (layered && !a->height) { w[1] = 1; w[3] = 1; }
      w[22] = mip ? mip->count : 1;
      break;
    }
    case CU_RESOURCE_TYPE_LINEAR:
      bits = formatBits(rd->res.linear.format, &kind);
      chans = rd->res.linear.numChannels;
      if (!bits || !chans) return 0;
      w[0] = rd->res.linear.devPtr;
      w[1] = 1;
      w[2] = rd->res.linear.sizeInBytes / (chans * bits / 8);
      w[3] = w[4] = 1;
      w[5] = rd->res.linear.sizeInBytes;
      break;
    case CU_RESOURCE_TYPE_PITCH2D:
      bits = formatBits(rd->res.pitch2D.format, &kind);
      chans = rd->res.pitch2D.numChannels;
      w[0] = rd->res.pitch2D.devPtr;
      w[1] = 2;
      w[2] = rd->res.pitch2D.width;
      w[3] = rd->res.pitch2D.height;
      w[4] = 1;
      w[5] = rd->res.pitch2D.pitchInBytes;
      w[6] = w[5] * w[3];
      break;
    default:
      return 0;
  }
  if (!bits || (chans != 1 && chans != 2 && chans != 4)) return 0;
  w[7] = chans;
  for (unsigned c = 0; c < chans; c++) w[8 + c] = bits;
  w[12] = kind;
  return 1;
}

// A block-compressed view's format as an array format (CU_AD_FORMAT_BC*); 0 for the others.
static unsigned viewBc(CUresourceViewFormat f) {
  switch (f) {
    case CU_RES_VIEW_FORMAT_UNSIGNED_BC1: return CU_AD_FORMAT_BC1_UNORM;
    case CU_RES_VIEW_FORMAT_UNSIGNED_BC2: return CU_AD_FORMAT_BC2_UNORM;
    case CU_RES_VIEW_FORMAT_UNSIGNED_BC3: return CU_AD_FORMAT_BC3_UNORM;
    case CU_RES_VIEW_FORMAT_UNSIGNED_BC4: return CU_AD_FORMAT_BC4_UNORM;
    case CU_RES_VIEW_FORMAT_SIGNED_BC4: return CU_AD_FORMAT_BC4_SNORM;
    case CU_RES_VIEW_FORMAT_UNSIGNED_BC5: return CU_AD_FORMAT_BC5_UNORM;
    case CU_RES_VIEW_FORMAT_SIGNED_BC5: return CU_AD_FORMAT_BC5_SNORM;
    case CU_RES_VIEW_FORMAT_UNSIGNED_BC6H: return CU_AD_FORMAT_BC6H_UF16;
    case CU_RES_VIEW_FORMAT_SIGNED_BC6H: return CU_AD_FORMAT_BC6H_SF16;
    case CU_RES_VIEW_FORMAT_UNSIGNED_BC7: return CU_AD_FORMAT_BC7_UNORM;
    default: return 0;
  }
}

// A resource view's texel format as an array format and its channels; 0 for the
// block-compressed formats.
static unsigned viewFormat(CUresourceViewFormat f, unsigned *chans) {
  static const struct { CUresourceViewFormat view; unsigned format, chans; } formats[] = {
    {CU_RES_VIEW_FORMAT_UINT_1X8, CU_AD_FORMAT_UNSIGNED_INT8, 1}, {CU_RES_VIEW_FORMAT_UINT_2X8, CU_AD_FORMAT_UNSIGNED_INT8, 2},
    {CU_RES_VIEW_FORMAT_UINT_4X8, CU_AD_FORMAT_UNSIGNED_INT8, 4}, {CU_RES_VIEW_FORMAT_SINT_1X8, CU_AD_FORMAT_SIGNED_INT8, 1},
    {CU_RES_VIEW_FORMAT_SINT_2X8, CU_AD_FORMAT_SIGNED_INT8, 2}, {CU_RES_VIEW_FORMAT_SINT_4X8, CU_AD_FORMAT_SIGNED_INT8, 4},
    {CU_RES_VIEW_FORMAT_UINT_1X16, CU_AD_FORMAT_UNSIGNED_INT16, 1}, {CU_RES_VIEW_FORMAT_UINT_2X16, CU_AD_FORMAT_UNSIGNED_INT16, 2},
    {CU_RES_VIEW_FORMAT_UINT_4X16, CU_AD_FORMAT_UNSIGNED_INT16, 4}, {CU_RES_VIEW_FORMAT_SINT_1X16, CU_AD_FORMAT_SIGNED_INT16, 1},
    {CU_RES_VIEW_FORMAT_SINT_2X16, CU_AD_FORMAT_SIGNED_INT16, 2}, {CU_RES_VIEW_FORMAT_SINT_4X16, CU_AD_FORMAT_SIGNED_INT16, 4},
    {CU_RES_VIEW_FORMAT_UINT_1X32, CU_AD_FORMAT_UNSIGNED_INT32, 1}, {CU_RES_VIEW_FORMAT_UINT_2X32, CU_AD_FORMAT_UNSIGNED_INT32, 2},
    {CU_RES_VIEW_FORMAT_UINT_4X32, CU_AD_FORMAT_UNSIGNED_INT32, 4}, {CU_RES_VIEW_FORMAT_SINT_1X32, CU_AD_FORMAT_SIGNED_INT32, 1},
    {CU_RES_VIEW_FORMAT_SINT_2X32, CU_AD_FORMAT_SIGNED_INT32, 2}, {CU_RES_VIEW_FORMAT_SINT_4X32, CU_AD_FORMAT_SIGNED_INT32, 4},
    {CU_RES_VIEW_FORMAT_FLOAT_1X16, CU_AD_FORMAT_HALF, 1}, {CU_RES_VIEW_FORMAT_FLOAT_2X16, CU_AD_FORMAT_HALF, 2},
    {CU_RES_VIEW_FORMAT_FLOAT_4X16, CU_AD_FORMAT_HALF, 4}, {CU_RES_VIEW_FORMAT_FLOAT_1X32, CU_AD_FORMAT_FLOAT, 1},
    {CU_RES_VIEW_FORMAT_FLOAT_2X32, CU_AD_FORMAT_FLOAT, 2}, {CU_RES_VIEW_FORMAT_FLOAT_4X32, CU_AD_FORMAT_FLOAT, 4}};
  for (size_t i = 0; i < sizeof formats / sizeof formats[0]; i++)
    if (formats[i].view == f) { *chans = formats[i].chans; return formats[i].format; }
  return 0;
}

// An array's or a mipmapped array's resource view (`view`) on the object's words: its texels read
// in another format of their size; of a layered array the layers from `firstLayer` to `lastLayer`;
// of a mipmapped array the levels from `firstMipmapLevel` to `lastMipmapLevel`, the first of them
// the view's level 0 (a GH200 and an A100: LOD 0 reads it, and LODs past the last read the last).
// The view's extent is the array's, its level 0's for a mipmapped array (an array's view may give
// 0 to take it; a GH200 and an A100 refuse a mipmapped array's view of another extent, 0 included);
// a block-compressed view's is four times a uint32 array's. A view of a cubemap is refused.
static CUresult texView(const CUDA_RESOURCE_DESC *rd, const CUDA_RESOURCE_VIEW_DESC *v, uint64_t *w) {
  CUmipmappedArray mip = rd->resType == CU_RESOURCE_TYPE_MIPMAPPED_ARRAY ? rd->res.mipmap.hMipmappedArray : NULL;
  if (rd->resType != CU_RESOURCE_TYPE_ARRAY && !mip) return CUDA_ERROR_INVALID_VALUE;
  CUarray a = mip ? &mip->levels[0] : rd->res.array.hArray;
  const int layered = w[20] != 0;
  // A GH200 and an A100 refuse every view of a cubemap array, of all its faces and with no format
  // too; of a layered or mipmapped cubemap, unmeasured.
  if (w[21]) {
    if (!mip && !(a->flags & CUDA_ARRAY3D_LAYERED)) return CUDA_ERROR_INVALID_VALUE;
    note("texture views of layered and mipmapped cubemaps are not modelled");
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  unsigned bcFormat = viewBc(v->format);
  if (bcFormat) {
    // A block-compressed view: of a 2D array of two uint32 channels (BC1, BC4) or four (the
    // others), each element one block, four times its width and height.
    unsigned kind, bb, bits, chans, bc = bcOf(bcFormat, &bb);
    if (mip || layered || a->depth) {
      note("block-compressed views of mipmapped, layered and 3D arrays are not modelled");
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (w[27] || formatBits(a->format, &kind) != 32 || kind != 0 || a->channels * 4 != bb ||
        v->width != 4 * a->width || v->height != 4 * (a->height ? a->height : 1) || v->depth ||
        v->firstLayer || v->lastLayer || v->firstMipmapLevel || v->lastMipmapLevel)
      return CUDA_ERROR_INVALID_VALUE;
    bcTexels(bcFormat, bc, &bits, &kind, &chans);
    w[1] = 2;
    w[2] = v->width;
    w[3] = v->height;
    w[7] = chans;
    for (unsigned c = 0; c < 4; c++) w[8 + c] = c < chans ? bits : 0;
    w[12] = kind;
    w[27] = bcFormat;
    w[35] = 1;
    return CUDA_SUCCESS;
  }
  if (mip ? v->width != a->width || v->height != (a->height ? a->height : 0) ||
              v->depth != (a->depth ? a->depth : 0)
          : (v->width && v->width != a->width) || (v->height && v->height != (a->height ? a->height : 1)) ||
              (!layered && v->depth && v->depth != (a->depth ? a->depth : 1)))
    return CUDA_ERROR_INVALID_VALUE;
  if (!mip && (v->firstMipmapLevel || v->lastMipmapLevel)) return CUDA_ERROR_INVALID_VALUE;
  if (mip) {
    if (v->firstMipmapLevel > v->lastMipmapLevel || v->lastMipmapLevel >= mip->count) return CUDA_ERROR_INVALID_VALUE;
    CUarray f = &mip->levels[v->firstMipmapLevel];
    w[0] = f->base;
    w[2] = f->width;
    w[3] = f->height ? f->height : 1;
    w[4] = f->depth ? f->depth : 1;
    w[5] = f->pitch;
    w[6] = f->slice;
    w[22] = v->lastMipmapLevel - v->firstMipmapLevel + 1;
  }
  if (v->format != CU_RES_VIEW_FORMAT_NONE) {
    unsigned chans, kind, format = viewFormat(v->format, &chans);
    if (!format) return CUDA_ERROR_INVALID_VALUE;
    unsigned bits = formatBits(format, &kind);
    if (bits * chans != w[8] * w[7]) return CUDA_ERROR_INVALID_VALUE;
    w[7] = chans;
    for (unsigned c = 0; c < 4; c++) w[8 + c] = c < chans ? bits : 0;
    w[12] = kind;
  }
  if (!layered) return v->firstLayer || v->lastLayer ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
  if (v->firstLayer > v->lastLayer || v->lastLayer >= w[20]) return CUDA_ERROR_INVALID_VALUE;
  // The layers each level holds and the view's first (`NV.Tex.Texture.levelLayers`, `firstLayer`):
  // a mipmapped array's levels lie apart by all its layers.
  w[33] = w[20];
  w[34] = v->firstLayer;
  w[20] = v->lastLayer - v->firstLayer + 1;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexObjectCreate(CUtexObject *out, const CUDA_RESOURCE_DESC *rd, const CUDA_TEXTURE_DESC *td,
                                   const CUDA_RESOURCE_VIEW_DESC *view) {
  NEED_INIT();
  if (!out || !rd || !td) return CUDA_ERROR_INVALID_VALUE;
  uint64_t w[TEX_WORDS];
  if (!texResource(rd, w)) {
    note("a texture over resource type %d is not modelled", (int)rd->resType);
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  if (view) {
    CUresult r = texView(rd, view, w);
    if (r != CUDA_SUCCESS) return r;
  }
  if (td->maxAnisotropy > 1) {
    note("anisotropic texture sampling is not modelled");
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  // A mipmapped array's sampling: the filter between levels, the level bias and clamps (binary32
  // bits); over anything else, level 0 alone.
  if (rd->resType == CU_RESOURCE_TYPE_MIPMAPPED_ARRAY) {
    float f[3] = {td->mipmapLevelBias, td->minMipmapLevelClamp, td->maxMipmapLevelClamp};
    uint32_t b[3];
    memcpy(b, f, sizeof b);
    w[23] = td->mipmapFilterMode == CU_TR_FILTER_MODE_LINEAR;
    w[24] = b[0];
    w[25] = b[1];
    w[26] = b[2];
  }
  // Integer texels of 8 and 16 bits read as normalized floats unless read as integers.
  w[13] = !(td->flags & CU_TRSF_READ_AS_INTEGER) && w[12] != 2 && w[8] < 32;
  // sRGB texels (word 28; `NV.Tex.Texture.srgb`). Disabling the trilinear optimization changes
  // nothing a GH200 shows; seamless cubemap filtering is not modelled.
  w[28] = (td->flags & CU_TRSF_SRGB) != 0;
  // The border colour (words 29 to 32, binary32 bits), which texels past the edge read under
  // border addressing.
  memcpy(&w[29], &td->borderColor[0], 4);
  memcpy(&w[30], &td->borderColor[1], 4);
  memcpy(&w[31], &td->borderColor[2], 4);
  memcpy(&w[32], &td->borderColor[3], 4);
  if (td->flags & CU_TRSF_SEAMLESS_CUBEMAP) {
    note("seamless cubemap filtering is not modelled");
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  w[14] = (td->flags & CU_TRSF_NORMALIZED_COORDINATES) != 0;
  w[15] = td->filterMode == CU_TR_FILTER_MODE_LINEAR;
  // A coordinate the texture does not have is clamped (its mode, often left zero, is wrap).
  for (int k = 0; k < 3; k++) w[16 + k] = k < (int)w[1] ? (uint64_t)td->addressMode[k] : CU_TR_ADDRESS_MODE_CLAMP;
  return texAdd(w, out);
}
CUresult CUDAAPI cuTexObjectDestroy(CUtexObject t) {
  NEED_INIT();
  return texRemove(t);
}
CUresult CUDAAPI cuSurfObjectCreate(CUsurfObject *out, const CUDA_RESOURCE_DESC *rd) {
  NEED_INIT();
  if (!out || !rd) return CUDA_ERROR_INVALID_VALUE;
  uint64_t w[TEX_WORDS];
  if (rd->resType != CU_RESOURCE_TYPE_ARRAY || !texResource(rd, w)) return CUDA_ERROR_INVALID_VALUE;
  w[16] = w[17] = w[18] = CU_TR_ADDRESS_MODE_CLAMP;
  w[19] = 1;
  return texAdd(w, out);
}
CUresult CUDAAPI cuSurfObjectDestroy(CUsurfObject s) {
  NEED_INIT();
  return texRemove(s);
}
