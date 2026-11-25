import NV.Core.Engine
import NV.Arch.Parts
import NV.SASS.Read
import NV.Async.TensorMap
import NV.Launch.Load
import NV.Launch.Report
import Device.Cache

/-!
# The emulated device's C ABI

What the CUDA driver shim (`tools/cudadev`, unproven C glue) calls (every argument owned, as
exported functions take them): a device is one part's parameters, its machine (caches carried
across launches), its global memory, the modules loaded into it and the kernels made from them.
Each call takes the device and returns the new one, so the C side holds one reference and the
engine updates it in place.

A module is a cubin read once (`SASS.Read.Module`): its data (`__device__` and `__constant__`
variables) are placed in device memory where the shim says when it is opened, and its kernels are
made by name when the program first asks for one (a library's cubin holds hundreds). Bank 3
(`__constant__` data) lives in device memory, so a host copy to a `__constant__` variable reaches
the next launch.

A call that changes the device returns it with an error message, empty on success; a failed one
returns the device as it was (but for a failed launch, which, as on a GPU, leaves the device
unusable: the shim then fails every call). Other errors are `IO` errors. The shim turns both into
CUDA error codes.
-/

namespace NV.Device

open NV NV.Core

structure Device where
  p : Params
  machine : Machine
  global : Store
  modules : Array SASS.Read.Module := #[]
  kernels : Array SASS.Kernel := #[]
  /-- Each kernel's module (an index into `modules`). -/
  kernelModule : Array Nat := #[]
  /-- Each module's listings and placement, digested (`Cache`'s key). -/
  moduleDigests : Array Cache.Digest := #[]
  /-- Cycles the device has run, over every launch. -/
  cycles : Nat := 0
  /-- The texture and surface objects the driver has created (`cuTexObjectCreate`,
  `cuSurfObjectCreate`): object `i + 1` is entry `i` (`NV.Tex`), as every launch sees them. -/
  textures : Array NV.Tex.Texture := #[]

/-- The part `name` (`NV.Arch.parts`), with `bytes` of global memory from address `base`. -/
@[export nvgpu_device_create]
def create (name : String) (base bytes : UInt64) : IO Device := do
  let some p := NV.Arch.parts.lookup name | throw (IO.userError s!"no part {name}")
  -- `NVGPU_LEAN_CLOCK`: the SM clock in MHz to run the part at (a measurement's clock).
  let p := match (← IO.getEnv "NVGPU_LEAN_CLOCK").bind String.toNat? with
    | some m => { p with clockMHz := { p.clockMHz with value := m } }
    | none => p
  return { p, machine := Machine.fresh p, global := Store.empty base bytes.toNat }

/-- Numbers about the part the driver reports (`cuDeviceGetAttribute`): 0 SMs, 1 the clock in
MHz. -/
@[export nvgpu_device_info]
def info (d : Device) (what : UInt32) : UInt64 :=
  match what with
  | 0 => d.p.sms.value.toUInt64
  | 1 => d.p.clockMHz.value.toUInt64
  | _ => 0

/-- The part's target (`sm_90a`, …), whose cubins the device runs. -/
@[export nvgpu_device_target]
def target (d : Device) : String := d.p.target.value

/-- Why `n` bytes at `addr` cannot be accessed, or `none`. -/
def outside? (d : Device) (what : String) (addr : UInt64) (n : Nat) : Option String :=
  if d.global.contains addr n then none
  else some s!"{what} of {n} bytes at {addr} outside device memory"

@[export nvgpu_write]
def write (d : Device) (addr : UInt64) (data : ByteArray) : Device × String :=
  if let some e := outside? d "write" addr data.size then (d, e) else
  let g := d.global
  let d := { d with global := default }  -- `g` unshared, so updated in place
  ({ d with global := g.writeBytes (addr - g.base).toNat data }, "")

@[export nvgpu_read]
def read (d : Device) (addr : UInt64) (n : UInt64) : IO ByteArray := do
  if let some e := outside? d "read" addr n.toNat then throw (IO.userError e)
  return d.global.readBytes (addr - d.global.base).toNat n.toNat

@[export nvgpu_memset]
def memset (d : Device) (addr : UInt64) (v : UInt8) (n : UInt64) : Device × String :=
  if let some e := outside? d "memset" addr n.toNat then (d, e) else
  let g := d.global
  let d := { d with global := default }
  ({ d with global := g.fill (addr - g.base).toNat n.toNat v }, "")

/-- Freed device memory: its whole pages are given back (`Store.release`). -/
@[export nvgpu_release]
def release (d : Device) (addr n : UInt64) : Device :=
  if !d.global.contains addr n.toNat then d else
  let g := d.global
  let d := { d with global := default }
  { d with global := g.release (addr - g.base).toNat n.toNat }

/-- A copy within device memory (`cuMemcpyDtoD`); the ranges may overlap. -/
@[export nvgpu_copy]
def copy (d : Device) (dst src n : UInt64) : Device × String :=
  if let some e := outside? d "copy source" src n.toNat then (d, e) else
  if let some e := outside? d "copy destination" dst n.toNat then (d, e) else
  let g := d.global
  let d := { d with global := default }
  let bytes := g.readBytes (src - g.base).toNat n.toNat
  ({ d with global := g.writeBytes (dst - g.base).toNat bytes }, "")

/-- Open a module from its cubin's `cuobjdump -sass` and `cuobjdump -elf` listings, its data placed
from `base` and written to device memory. Its index is the old module count. -/
@[export nvgpu_module_open]
def moduleOpen (d : Device) (sass elf : String) (base : UInt64) : Device × String := Id.run do
  let m := SASS.Read.Module.of sass elf base.toNat
  for (a, bytes) in m.data.data do
    if let some e := outside? d "module data" a.toUInt64 bytes.size then return (d, e)
  let mut g := d.global
  let d := { d with global := default }
  for (a, bytes) in m.data.data do
    g := g.writeBytes (a - g.base.toNat) ⟨bytes⟩
  let md := (({} : Cache.Digest).str sass |>.str elf).nat base.toNat
  return ({ d with global := g, modules := d.modules.push m, moduleDigests := d.moduleDigests.push md }, "")

/-- The end of module `m`'s placed data (its base if it has none): where the next module's may
start. -/
@[export nvgpu_module_data_end]
def moduleDataEnd (d : Device) (m : UInt32) (base : UInt64) : UInt64 :=
  match d.modules[m.toNat]? with
  | none => base
  | some md => md.data.data.foldl (fun e (a, b) => max e (a + b.size).toUInt64) base

/-- Make module `m`'s kernel `name`; its index is the old kernel count. -/
@[export nvgpu_module_kernel]
def moduleKernel (d : Device) (m : UInt32) (name : String) : Device × String :=
  match d.modules[m.toNat]? with
  | none => (d, s!"no module {m}")
  | some md =>
    match md.kernels (· == name) with
    | .error e => (d, e)
    | .ok #[k] =>
      -- The module's data are in device memory already (`moduleOpen`): a launch does not write
      -- them again over what the program has done to them since; `printf` still finds its
      -- format strings in them.
      ({ d with kernels := d.kernels.push { k with dataLoaded := true },
                kernelModule := d.kernelModule.push m.toNat }, "")
    | .ok _ => (d, s!"no kernel {name} in the module")

/-- Module `m`'s data symbol `name`: `what` 0 its address, 1 its bytes; 0 if it has none. -/
@[export nvgpu_module_global]
def moduleGlobal (d : Device) (m : UInt32) (name : String) (what : UInt32) : UInt64 :=
  match (d.modules[m.toNat]?).bind fun md => md.data.symbols.find? (·.1 == name) with
  | some (_, a, n) => (if what == 0 then a else n).toUInt64
  | none => 0

@[export nvgpu_kernel_count]
def kernelCount (d : Device) : UInt32 := d.kernels.size.toUInt32

@[export nvgpu_kernel_name]
def kernelName (d : Device) (k : UInt32) : String := (d.kernels[k.toNat]?.map (·.name)).getD ""

/-- A kernel's numbers: 0 parameter-block bytes, 1 parameters, 2 static shared bytes, 3 registers
per thread, 4 local bytes per thread; `5 + 2i`, `6 + 2i`: parameter `i`'s offset and size. -/
@[export nvgpu_kernel_info]
def kernelInfo (d : Device) (k what : UInt32) : UInt64 :=
  match d.kernels[k.toNat]? with
  | none => 0
  | some kn =>
    match what.toNat with
    | 0 => kn.paramBytes.toUInt64
    | 1 => kn.params.size.toUInt64
    | 2 => kn.sharedBytes.toUInt64
    | 3 => kn.regs.toUInt64
    | 4 => kn.localBytes.toUInt64
    | w =>
      let i := (w - 5) / 2
      match kn.params[i]? with
      | some (off, size) => (if (w - 5) % 2 == 0 then off else size).toUInt64
      | none => 0

/-- A kernel's declared shapes from its ELF attributes: 0 the most threads per block it allows
(`.maxntid` or `.reqntid`, as a product; 0 if it declares neither), 1–3 its required cluster shape
(0 if none), 4 its SM version. -/
@[export nvgpu_kernel_attr]
def kernelAttr (d : Device) (k what : UInt32) : UInt64 :=
  match d.kernels[k.toNat]?, d.kernelModule[k.toNat]? with
  | some kn, some m =>
    let a := ((d.modules[m]?).bind fun md => md.attrs.lookup kn.name).getD {}
    let prod : Option (Nat × Nat × Nat) → Nat := fun | some (x, y, z) => x * y * z | none => 0
    let (cx, cy, cz) := kn.cluster.getD (0, 0, 0)
    match what with
    | 0 => (prod (a.reqThreads <|> a.maxThreads)).toUInt64
    | 1 => cx.toUInt64
    | 2 => cy.toUInt64
    | 3 => cz.toUInt64
    | 4 => kn.sm.toUInt64
    | _ => 0
  | _, _ => 0

/-- Kernel `k` as it runs now: bank 3 read from where its module placed it in device memory. -/
def current (d : Device) (k : Nat) : Option SASS.Kernel := do
  let kn ← d.kernels[k]?
  let md := (d.kernelModule[k]?).bind (d.modules[·]?)
  match md.bind (·.data.bank3), kn.banks.findIdx? (·.1 == 3) with
  | some a, some i =>
    let n := 4 * kn.banks[i]!.2.size
    let bytes := d.global.readBytes (a - d.global.base.toNat) n
    some { kn with banks := kn.banks.set! i (3, SASS.Elf.wordsOf bytes) }
  | _, _ => some kn

/-- Run kernel `k` on the device: its grid, block and cluster shapes, dynamic shared bytes and
parameter block; functional (untimed) or timed. The device's memory and machine after it. -/
def launchOf (d : Device) (k gx gy gz bx by_ bz cx cy cz smem : UInt32) (params : ByteArray)
    (functional : Bool) : Option Launch :=
  (current d k.toNat).map fun kernel =>
    { kernel, grid := (gx.toNat, gy.toNat, gz.toNat), block := (bx.toNat, by_.toNat, bz.toNat),
      params := params, global := d.global, dynShared := smem.toNat, textures := d.textures,
      cluster := (cx.toNat, cy.toNat, cz.toNat), functional }

/-- `launch`'s run; `cacheable`: its result may come from, and go to, the launch cache. -/
def launchWith (cacheable : Bool) (d : Device) (k gx gy gz bx by_ bz cx cy cz smem : UInt32) (params : ByteArray)
    (functional : UInt8) : IO (Device × String) := do
  let some l := launchOf d k gx gy gz bx by_ bz cx cy cz smem params (functional != 0)
    | return (d, s!"no kernel {k}")
  -- `NVGPU_LEAN_ISSUE_TRACE`: every issued instruction to standard error (`Launch.trace`).
  let l := { l with trace := (← IO.getEnv "NVGPU_LEAN_ISSUE_TRACE").isSome }
  -- `NVGPU_LEAN_SPANS`: each CTA's span to standard error as it finishes (`Launch.spans`).
  let l := { l with spans := (← IO.getEnv "NVGPU_LEAN_SPANS").isSome }
  -- `NVGPU_LEAN_RACES`: the race check (`Launch.races`); its races and the timed mode's hazards
  -- (a read before its producer lands) to standard error, the kernel's name first.
  let checks := (← IO.getEnv "NVGPU_LEAN_RACES").isSome
  -- `NVGPU_LEAN_WORKERS`: the workers the SMs are dealt to (`Launch.workers`; 0, the default: 8).
  let workers := ((← IO.getEnv "NVGPU_LEAN_WORKERS").bind String.toNat?).getD 0
  -- `NVGPU_LEAN_UNTIL`: each launch stopped at that cycle (`Launch.untilCycle`), a sample of a launch
  -- too long to run whole; its memory then holds what the sample wrote, and later launches read that.
  let untilCycle := (← IO.getEnv "NVGPU_LEAN_UNTIL").bind String.toNat?
  -- `NVGPU_LEAN_REPORT` (or a stop): each launch's report to standard error, as `lake exe sim` prints it.
  let reported := (← IO.getEnv "NVGPU_LEAN_REPORT").isSome || untilCycle.isSome
  let l := { l with races := checks, workers, untilCycle }
  let kernel := l.kernel
  -- `NVGPU_LEAN_LAUNCH_CACHE`: a launch whose result is recorded is not simulated (`Cache`); not
  -- with the trace, the spans or the race check, whose output a recorded result does not hold, nor
  -- with a stop, whose result is a sample's.
  let stores ← if !cacheable || l.trace || l.spans || checks || untilCycle.isSome then pure none
    else Cache.Stores.fromEnv
  let mut cacheKey : Option (String × Array UInt64) := none
  if let some st := stores then
    let mut key := ((← Cache.version).str (reprStr d.p)).nat k.toNat
    key := match d.kernelModule[k.toNat]?.bind (d.moduleDigests[·]?) with
      | some md => (key.nat md.a.toNat).nat md.b.toNat
      | none => key
    key := key.str kernel.name
    -- The texture and surface objects the kernel may name.
    if !d.textures.isEmpty then key := key.str (reprStr d.textures)
    for (b, ws) in kernel.banks do
      key := ws.foldl (fun key w => key.nat w.toNat) ((key.nat b).nat ws.size)
    let (gx, gy, gz) := l.grid
    let (bx, by_, bz) := l.block
    let (cx, cy, cz) := l.cluster
    key := [gx, gy, gz, bx, by_, bz, cx, cy, cz, l.dynShared, if l.functional then 1 else 0, workers].foldl
      Cache.Digest.nat key
    key := (key.add l.params).add (Cache.encodeMachine d.machine)
    -- `NVGPU_LEAN_LAUNCH_CACHE_PARTS`: each part of the key to standard error (why a launch missed).
    if (← IO.getEnv "NVGPU_LEAN_LAUNCH_CACHE_PARTS").isSome then
      let part (what : String) (x : Cache.Digest) := IO.eprintln s!"nvgpu-lean: {kernel.name}: key part {what} {x.hex}"
      part "version" (← Cache.version)
      part "params" (({} : Cache.Digest).str (reprStr d.p))
      part "module" ((d.kernelModule[k.toNat]?.bind (d.moduleDigests[·]?)).getD {})
      part "banks" (kernel.banks.foldl (fun x (b, ws) => ws.foldl (fun x w => x.nat w.toNat) ((x.nat b).nat ws.size)) {})
      part "launch" ((({} : Cache.Digest).add l.params).nat l.dynShared)
      part "machine" (({} : Cache.Digest).add (Cache.encodeMachine d.machine))
      part "memory" (Cache.storeDigest {} d.global).1
    let (full, pages) := Cache.storeDigest key d.global
    let hex := full.hex
    cacheKey := some (hex, pages)
    if let some bytes ← st.get hex then
      match Cache.Entry.decode hex bytes with
      | .ok e =>
        if (← IO.getEnv "NVGPU_LEAN_VERBOSE").isSome then
          IO.eprintln s!"nvgpu-lean: {kernel.name}: launch cache hit {hex}"
        let g := d.global
        let d := { d with global := default }
        let global := Cache.applyPages g e.base e.size e.pages
        let cycles := d.cycles + e.cycles
        return ({ d with machine := e.machine, global, cycles }, "")
      | .error e => IO.eprintln s!"nvgpu-lean: {kernel.name}: launch cache entry {hex} unread: {e}"
  let d := { d with global := default }
  let t0 ← IO.monoMsNow
  let res ← runOnIO d.p d.machine l
  if (← IO.getEnv "NVGPU_LEAN_VERBOSE").isSome then
    IO.eprintln s!"nvgpu-lean: {kernel.name}: the engine's run {(← IO.monoMsNow) - t0} ms"
  match res with
  | .ok (machine, r) =>
    if reported then
      let lines := (NV.Launch.Report.report 0 kernel.name r.counters).splitOn "\n" |>.drop 1
      IO.eprint (s!"nvgpu-lean: {kernel.name}: report\n" ++ "\n".intercalate lines ++
        NV.Launch.Report.binding d.p r.counters)
    if checks then
      for h in r.hazards do IO.eprintln s!"nvgpu-lean: {kernel.name}: hazard {h}"
      for x in r.races do IO.eprintln s!"nvgpu-lean: {kernel.name}: race {x}"
      if r.hazards.isEmpty && r.races.isEmpty then IO.eprintln s!"nvgpu-lean: {kernel.name}: no race, no hazard"
    if let (some st, some (hex, before)) := (stores, cacheKey) then
      let pages := Cache.changedPages before r.global
      let e : Cache.Entry := { cycles := r.counters.cycles, machine, base := r.global.base.toNat, size := r.global.size, pages }
      try st.put hex (e.encode hex)
      catch err => IO.eprintln s!"nvgpu-lean: {kernel.name}: launch cache entry {hex} unwritten: {err}"
    return ({ d with machine, global := r.global, cycles := d.cycles + r.counters.cycles }, "")
  | .error e => return (d, s!"{kernel.name}: {e}")

@[export nvgpu_launch]
def launch (d : Device) (k gx gy gz bx by_ bz cx cy cz smem : UInt32) (params : ByteArray)
    (functional : UInt8) : IO (Device × String) :=
  launchWith true d k gx gy gz bx by_ bz cx cy cz smem params functional

/-- A launch on device `g` of a mesh whose kernel may reach the other devices' memory (a peer's
pointer): the devices' memories, laid end to end in their address order (`Store.concat`), are its
global memory, and are given back to each device after it. The other devices' memory is timed
as the launching device's own (its L2 and DRAM); the launch cache is not used. -/
@[export nvgpu_launch_mesh]
def launchMesh (ds : Array Device) (g : UInt32) (k gx gy gz bx by_ bz cx cy cz smem : UInt32)
    (params : ByteArray) (functional : UInt8) : IO (Array Device × String) := do
  let some d := ds[g.toNat]? | return (ds, s!"no device {g}")
  let sizes := ds.map (·.global.size)
  let globals := ds.map (·.global)
  let ds := ds.map fun x => { x with global := default }
  let some merged := Store.concat globals
    | return (ds.zipWith (fun x s => { x with global := s }) globals, "the devices' memories are not laid end to end")
  let (d', msg) ← launchWith false { d with global := merged } k gx gy gz bx by_ bz cx cy cz smem params functional
  let back := d'.global.split sizes
  let ds := ds.set! g.toNat { d' with global := default }
  return (ds.zipWith (fun x s => { x with global := s }) back, msg)

/-- The launch `launch` would run, exported for the accelerated backend into `dir`
(`Launch.exportLaunchFiles`: its constant bank 0, kernel and shapes, the device memory's segments
`extra`'s lines); the device is unchanged. An error message, empty on success. -/
@[export nvgpu_launch_export]
def launchExport (d : Device) (k gx gy gz bx by_ bz cx cy cz smem : UInt32) (params : ByteArray)
    (dir extra : String) : IO String := do
  let some l := launchOf d k gx gy gz bx by_ bz cx cy cz smem params true | return s!"no kernel {k}"
  NV.Launch.exportLaunchFiles d.p dir l (extra.splitOn "\n" |>.filter (· != ""))
  return ""

/-- The texture and surface objects, as the driver records them (`tools/cudadev/tex.h`): per
object 24 little-endian 64-bit words: texel (0, 0, 0)'s address, the dimensions addressed, width,
height, depth, the row and slice pitches, the channels and each one's bits (four words), the
channels' kind (0 unsigned, 1 signed, 2 float), integers read normalized, coordinates normalized,
the filter (0 point, 1 linear), the three coordinates' addressing modes (0 wrap, 1 clamp,
2 mirror, 3 border), a surface, the layers, a cubemap, a mipmapped array's levels (1 for any
other object), linear filtering between levels, and the level bias and clamps (binary32); 28 words
an object, a freed object's all zero. -/
@[export nvgpu_set_textures]
def setTextures (d : Device) (table : ByteArray) : Device := Id.run do
  let word (i : Nat) : Nat := (List.range 8).foldl (fun a b => a + (table.get! (8 * i + b)).toNat * 256 ^ b) 0
  let mut ts := #[]
  for o in [0:table.size / 224] do
    let w (k : Nat) := word (28 * o + k)
    let n := w 7
    let addr (m : Nat) : NV.Tex.Address := match m with | 0 => .wrap | 2 => .mirror | 3 => .border | _ => .clamp
    ts := ts.push {
      base := (w 0).toUInt64, dims := w 1, width := w 2, height := w 3, depth := w 4, pitch := w 5,
      slice := w 6, chans := (Array.range (min n 4)).map (fun c => w (8 + c)),
      kind := match w 12 with | 1 => .signed | 2 => .float | _ => .unsigned,
      readNormalized := w 13 != 0, normalizedCoords := w 14 != 0,
      filter := if w 15 == 1 then .linear else .point, address := #[addr (w 16), addr (w 17), addr (w 18)],
      surface := w 19 != 0, layers := w 20, cube := w 21 != 0, levels := max 1 (w 22),
      mipLinear := w 23 != 0, lodBias := (w 24).toUInt32, lodMin := (w 25).toUInt32,
      lodMax := (w 26).toUInt32 }
  return { d with textures := ts }

@[export nvgpu_error_string]
def errorString (e : IO.Error) : String := toString e

@[export nvgpu_device_cycles]
def cycles (d : Device) : UInt64 := d.cycles.toUInt64

/-- The device idle until its `upTo`-th cycle (a collective's end, waited for on the device). -/
@[export nvgpu_device_wait]
def wait (d : Device) (upTo : UInt64) : Device := { d with cycles := max d.cycles upTo.toNat }

/-- A collective's time in nanoseconds on this part's links (`LinkParams`), its `ranks` GPUs each
reached at `peerGBps` one way (NVSwitch's full bandwidth to every GPU) at `peerPermille` of it:
the bytes that cross a GPU's link one way, by NCCL's algorithms for the kind, over that rate.
`kind`: 0 all-reduce, 1 broadcast, 2 reduce, 3 all-gather, 4 reduce-scatter, 5 all-to-all, 6 a
send to one peer; `bytes`: the call's buffer on one rank, its input (all-reduce, broadcast,
reduce, reduce-scatter, all-to-all, send) or its output (all-gather). -/
@[export nvgpu_collective_ns]
def collectiveNs (d : Device) (kind ranks bytes : UInt64) : UInt64 :=
  let n := ranks.toNat
  let b := bytes.toNat
  -- Bytes a GPU sends one way: a ring all-reduce sends each of its n chunks twice less its own,
  -- a gather or scatter every chunk but its own once, a pipelined broadcast or reduce the buffer.
  let wire := match kind with
    | 0 => 2 * (n - 1) * b / max n 1
    | 1 | 2 | 6 => b
    | _ => (n - 1) * b / max n 1
  -- GB/s is bytes per ns; at `peerPermille` of it, wire × 1000 / (GB/s × ‰) ns.
  let rate := d.p.link.peerGBps.value * d.p.link.peerPermille.value
  if n ≤ 1 || rate == 0 then 0 else (wire * 1000 / rate).toUInt64

/-! ## Tensor maps

`cuTensorMapEncodeTiled` and `cuTensorMapEncodeIm2col` write the model's own 128-byte encoding
(`NV.Async.TensorMap.encode`), which the TMA model decodes wherever a kernel finds it. The shim
passes the arguments packed little-endian (`tools/cudadev/tmap.h`): kind (0 tiled, 1 im2col),
data type, rank, interleave, swizzle, L2 promotion and fill as 32-bit words from byte 0; the
address at 32; five 64-bit dimensions at 40; four 64-bit strides at 80; five 32-bit box
dimensions at 112 and element strides at 132; for im2col, three signed 32-bit lower and upper
corners at 152 and 164, channels per pixel at 176 and pixels per column at 180. -/

def leNat (b : ByteArray) (off n : Nat) : Nat :=
  (List.range n).foldl (fun v j => v ||| ((b.get! (off + j)).toNat <<< (8 * j))) 0

def leInt32 (b : ByteArray) (off : Nat) : Int :=
  let v := leNat b off 4
  if v ≥ 2 ^ 31 then (v : Int) - 2 ^ 32 else v

/-- The enum value a driver number names, or why it names none. -/
def enumOf {α : Type} (all : List α) (what : String) (v : Nat) : Except String α :=
  match NV.Async.TensorMap.ofIndex? all v with
  | some x => pure x
  | none => throw s!"tensor map: {what} {v} is not one the driver defines"

open NV.Async in
/-- The map the packed arguments describe, or why the driver refuses them. -/
def tensorMapOf (b : ByteArray) : Except String TensorMap := do
  if b.size < 184 then throw "tensor map arguments: too short"
  let w (i : Nat) := leNat b (4 * i) 4
  let rank := w 2
  let im2col : Option Im2col :=
    if w 0 == 1 then
      some { lower := #v[leInt32 b 152, leInt32 b 156, leInt32 b 160],
             upper := #v[leInt32 b 164, leInt32 b 168, leInt32 b 172],
             channelsPerPixel := leNat b 176 4, pixelsPerColumn := leNat b 180 4 }
    else none
  let m : TensorMap :=
    { dtype := ← enumOf TmDType.all "data type" (w 1), rank, globalAddress := (leNat b 32 8).toUInt64
      globalDim := #v[leNat b 40 8, leNat b 48 8, leNat b 56 8, leNat b 64 8, leNat b 72 8]
      globalStrides := #v[leNat b 80 8, leNat b 88 8, leNat b 96 8, leNat b 104 8]
      boxDim := #v[leNat b 112 4, leNat b 116 4, leNat b 120 4, leNat b 124 4, leNat b 128 4]
      elementStrides := #v[leNat b 132 4, leNat b 136 4, leNat b 140 4, leNat b 144 4, leNat b 148 4]
      interleave := ← enumOf Interleave.all "interleave" (w 3), swizzle := ← enumOf Swizzle.all "swizzle" (w 4)
      l2Promotion := ← enumOf L2Promotion.all "L2 promotion" (w 5)
      oobFill := ← enumOf OobFill.all "out-of-bounds fill" (w 6), im2col }
  if let some e := m.invalid? then throw e
  return m

/-- The descriptor for the packed arguments, and an error (then the descriptor is empty). -/
@[export nvgpu_tensor_map]
def tensorMap (args : ByteArray) : ByteArray × String :=
  match tensorMapOf args with
  | .ok m => (m.encode, "")
  | .error e => (.empty, e)

/-- `cuTensorMapReplaceAddress`: the descriptor with its global address replaced (empty if the
128 bytes are not a descriptor). -/
@[export nvgpu_tensor_map_address]
def tensorMapAddress (map : ByteArray) (addr : UInt64) : ByteArray :=
  match NV.Async.TensorMap.decode map with
  | some m => ({ m with globalAddress := addr }).encode
  | none => .empty

end NV.Device
