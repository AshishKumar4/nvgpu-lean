import NV.Machine.Access

/-!
# Generic addresses and shared-memory windows

PTX ISA 9.4 §6.4.1.1: the `.const`, kernel `.param`, `.local` and `.shared` state spaces are
windows in the generic address space; a generic address outside every window is global. The
`.shared::cta` window falls within the `.shared::cluster` window (§5.1.7), and `mapa` maps a
shared address to the corresponding one in another CTA of the cluster (§9.7.10.26).

Window bases are not documented and no program may depend on them beyond what the ISA states, so
the values here are ours: one source for PTX semantics (`cvta`, `isspacep`, `mapa`, generic
`ld`/`st`) and for SASS (the engine publishes them where compiled code reads them).

Shared addresses (`.shared` state-space addresses, 32-bit) are encoded so that one CTA's
addresses also name it in the cluster: an address below `ctaStride` is in the executing CTA;
`(r + 1) · ctaStride + off` is offset `off` in the CTA of cluster rank `r`. `mapa` to the
executing CTA's own rank returns the plain offset, so each location has one address.
-/

namespace NV

structure Windows where
  /-- Bytes reserved per CTA in the shared window (more than any CTA's shared memory). -/
  ctaStride : UInt64 := 0x100_0000
  /-- The most CTAs in a cluster (16 with the non-portable cluster size attribute). -/
  maxCluster : Nat := 16
  «local» : UInt64 := 0x7FF0_0000_0000
  localSize : UInt64 := 0x100_0000
  shared : UInt64 := 0x7FF1_0000_0000
  const : UInt64 := 0x7FF2_0000_0000
  constSize : UInt64 := 0x100_0000
  /-- The kernel parameter window, inside the global window (§6.4.1.1). -/
  param : UInt64 := 0x7FF3_0000_0000
  paramSize : UInt64 := 0x1_0000
  deriving Repr, Inhabited

namespace Windows

def sharedSize (w : Windows) : UInt64 := (w.maxCluster.toUInt64 + 1) * w.ctaStride

/-- The shared address of offset `off` in the CTA of cluster rank `rank`, seen from the CTA of
rank `self`. -/
def clusterAddr (w : Windows) (self rank : Nat) (off : UInt64) : UInt64 :=
  if rank == self then off else (rank.toUInt64 + 1) * w.ctaStride + off

/-- The CTA rank (`none`: the executing CTA) and offset a shared address names. -/
def sharedTarget (w : Windows) (a : UInt64) : Option Nat × UInt64 :=
  if a < w.ctaStride then (none, a) else (some (a / w.ctaStride - 1).toNat, a % w.ctaStride)

/-- The space and space address of a generic address. -/
def resolve (w : Windows) (g : UInt64) : Space × UInt64 :=
  if w.«local» ≤ g && g < w.«local» + w.localSize then (.local, g - w.«local»)
  else if w.shared ≤ g && g < w.shared + w.sharedSize then
    let a := g - w.shared
    ((if a < w.ctaStride then .shared else .sharedCluster), a)
  else if w.const ≤ g && g < w.const + w.constSize then (.const 0, g - w.const)
  else if w.param ≤ g && g < w.param + w.paramSize then (.param, g - w.param)
  else (.global, g)

/-- The generic address of a space address (`cvta`). -/
def toGeneric (w : Windows) : Space → UInt64 → UInt64
  | .local, a => w.«local» + a
  | .shared, a | .sharedCluster, a => w.shared + a
  | .const _, a => w.const + a
  | .param, a => w.param + a
  | .global, a => a
  -- Tensor memory has no generic window: only `tcgen05` instructions address it.
  | .tmem, a => a

/-- Whether a generic address lies in a space's window (`isspacep`); the parameter window lies
in the global one. -/
def inWindow (w : Windows) (s : Space) (g : UInt64) : Bool :=
  let (s', a) := w.resolve g
  match s, s' with
  | .global, .global | .global, .param => true
  | .shared, .shared => true
  | .sharedCluster, .shared | .sharedCluster, .sharedCluster => a < w.sharedSize
  | .local, .local | .param, .param => true
  | .const _, .const _ => true
  | _, _ => false

end Windows

end NV
