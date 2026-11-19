// Distributed shared memory in a cluster of two CTAs: each CTA writes into its shared memory,
// the cluster synchronizes, and each thread reads its neighbour CTA's word through the cluster
// window (mapa + ld.shared::cluster) and writes it out. A second cluster barrier keeps the
// neighbour's shared memory alive until every read is done.
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__ void __cluster_dims__(2, 1, 1) cluster_dsmem(int* out) {
  __shared__ int s[128];
  cg::cluster_group cluster = cg::this_cluster();
  unsigned rank = cluster.block_rank();
  s[threadIdx.x] = (int)(blockIdx.x * 1000000 + rank * 1000 + threadIdx.x);
  cluster.sync();
  int* peer = cluster.map_shared_rank(s, rank ^ 1);
  int v = peer[threadIdx.x];
  cluster.sync();
  out[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
