"""JAX on a mesh of emulated devices (NVGPU_LEAN_GPU=h100x2, b200x8, ...): every device is seen,
holds data, runs jitted programs, and passes arrays to the others; programs mapped over the devices
without collectives (pmap, shard_map) give what one device gives. Exits nonzero on a mismatch.
    python tools/cudadev/mesh_check.py
"""
import sys

import numpy as np

from jax_mesh import devices_named, setup

jax = setup()
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

want = devices_named()
devs = jax.devices()
n = len(devs)
print(f"{n} devices:", ", ".join(f"{d.id} {d.platform} {d.device_kind}" for d in devs))
if n != want or devs[0].platform != "gpu":
    print(f"expected {want} GPUs")
    sys.exit(1)
ok = True


def check(name, got, want):
    global ok
    got, want = np.asarray(got), np.asarray(want)
    same = got.shape == want.shape and np.array_equal(got.view(np.uint8), want.view(np.uint8))
    ok &= same
    print(f"{name}: {'bit-exact' if same else 'MISMATCH'}")


rng = np.random.default_rng(0)
x = rng.standard_normal((n, 128, 64), dtype=np.float32)
w = rng.standard_normal((64, 32), dtype=np.float32)
f = jax.jit(lambda a, b: jnp.tanh(a @ b) * 2.0)
ref = np.stack([np.asarray(f(x[i], w)) for i in range(n)])

# Each device on its own, then the result carried to the next device and back.
for i, d in enumerate(devs):
    xi, wi = jax.device_put(x[i], d), jax.device_put(w, d)
    yi = f(xi, wi)
    check(f"device {i}: jit", yi, ref[i])
    moved = jax.device_put(yi, devs[(i + 1) % n])
    check(f"device {i} -> {(i + 1) % n}: copy", moved, ref[i])

check("pmap", jax.pmap(f, in_axes=(0, None))(x, w), ref)

mesh = Mesh(np.array(devs), ("d",))
xs = jax.device_put(x, NamedSharding(mesh, P("d")))
g = jax.jit(jax.shard_map(lambda a, b: f(a[0], b)[None], mesh=mesh, in_specs=(P("d"), P()),
                          out_specs=P("d")))
check("shard_map", g(xs, w), ref)

# Collectives through NCCL, on integer-valued floats (exact in any order) and integers.
nccl = [l.split()[-1] for l in open("/proc/self/maps") if "libnccl.so" in l]
print("NCCL:", sorted(set(nccl)))
xi = rng.integers(-1000, 1000, size=(n, 16, 8)).astype(np.float32)
ii = rng.integers(-2**20, 2**20, size=(n, 16, 8)).astype(np.int32)
xs = jax.device_put(xi, NamedSharding(mesh, P("d")))
smap = lambda f, out=P("d"): jax.jit(jax.shard_map(f, mesh=mesh, in_specs=P("d"), out_specs=out))
check("psum", smap(lambda a: jax.lax.psum(a, "d"))(xs), np.broadcast_to(xi.sum(0, keepdims=True), xi.shape))
check("psum int32", smap(lambda a: jax.lax.psum(a, "d"))(jax.device_put(ii, NamedSharding(mesh, P("d")))),
      np.broadcast_to(ii.sum(0, keepdims=True, dtype=np.int32), ii.shape))
check("pmax", smap(lambda a: jax.lax.pmax(a, "d"))(xs), np.broadcast_to(xi.max(0, keepdims=True), xi.shape))
check("all_gather", smap(lambda a: jax.lax.all_gather(a[0], "d")[None])(xs), np.broadcast_to(xi[None], (n,) + xi.shape))
check("psum_scatter", smap(lambda a: jax.lax.psum_scatter(a[0], "d", scatter_dimension=0, tiled=True)[None])(xs),
      xi.sum(0).reshape(n, 16 // n, 8))
perm = [(i, (i + 1) % n) for i in range(n)]
check("ppermute", smap(lambda a: jax.lax.ppermute(a, "d", perm))(xs), np.roll(xi, 1, axis=0))
x2 = rng.integers(-1000, 1000, size=(n, n, 4)).astype(np.float32)
check("all_to_all", smap(lambda a: jax.lax.all_to_all(a, "d", 1, 1, tiled=True))(jax.device_put(x2, NamedSharding(mesh, P("d")))),
      x2.transpose(1, 0, 2))
sys.exit(0 if ok else 1)
