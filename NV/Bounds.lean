import NV.Bounds.Rate
import NV.Bounds.Chain
import NV.Bounds.Compulsory
import NV.Bounds.LoomisWhitney
import NV.Bounds.Pebble
import NV.Bounds.Attention
import NV.Bounds.Machine
import NV.Bounds.Bound
import NV.Bounds.Sass
import NV.Bounds.Specs
import NV.Bounds.Floors

/-!
# L5: bounds

Lower bounds on a kernel's running time, as theorems.

* Resource invariants (`Rate`, `Machine`): no run moves more through a port than its rate allows
  (`Use.units_le`, `Slots.time_ge`); the engine's ports are chained (`serveAll_chained`) and its
  rates are the facts' (`dram_rate_exact`, `data_port_le`). The dependent-latency chain
  (`Stream.depths_le`, `ChainOk.le`).
* Spec bounds, each with its class: compulsory traffic for every program (`compulsory_traffic`,
  `compulsory_write_traffic`, by non-interference on `NV.Formal.Outcomes`); the classical-matmul
  I/O bound at a fast-memory capacity (`Pebble.mmm_io`, `mmm_io_parallel`, from
  `loomis_whitney`); FlashAttention's Proposition 3 (`Attention.prop3`).
* `Bound`: terms over resources, `Bound.cycles = max`, `Bound.cycles_le`; `Gap`: simulated time
  over each term, naming the binding one, printing the facts each used (`Specs`).
* Class optimality of a fixed instruction stream: no schedule beats its critical path and
  per-resource work (`Stream.bound_le`, `Sass.pathStream`).
* decode-layer stream, issue and chain floors and the layer floor as instances (`Floors`).
-/
