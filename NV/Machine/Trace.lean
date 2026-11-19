import NV.Machine.Access

/-!
# A warp's trace

What the timed engine did with one warp, in order: each issue, with what its effect took from
outside the warp, and each writeback landing. `NV.Formal.Trace.check` replays a trace against the
warp's step and, when it agrees, gives a non-blocking run (`NV.Formal.RunNB`) of the same acts: the
engine's run of that warp refines the semantics `Outcomes` reads.
-/

namespace NV

/-- What an issue's effect took from outside the warp. -/
inductive Choice where
  /-- Nothing: `next`, `reg`, `exit`, `fault`. -/
  | none
  /-- A memory request: what each lane's reads returned, lane `l` at index `l`. -/
  | mem (read : Array ByteArray)
  /-- A synchronizing operation: the barrier instance it joined. -/
  | sync (inst : Nat)
  /-- A collective: the instance it joined, and its answer or, where the ISA leaves the answer
  undefined, the fault. -/
  | collect (inst : Nat) (answer : Except String Resp)
  /-- An asynchronous operation: what its agent's reads returned, request by request in the
  agent's order and lane by lane, and its result or how it failed. -/
  | async (reads : Array (Array ByteArray)) (result : Except String Resp)
  deriving Inhabited

/-- One step of a warp's trace. -/
inductive TraceEv where
  /-- The instruction at `pc` issued with `choice`; its writeback, if it has one, is pending
  under `wb`. -/
  | issue (pc : Nat) (choice : Choice) (wb : Option Nat)
  /-- The writeback pending under `id` landed, leaving `vals` in its write set: per location, in
  the footprint's order, the `digest` of its 32 lanes' words (`none`: not compared, as for the
  `printf` records, which the engine hands to the run's output). -/
  | land (id : Nat) (vals : Array (Option UInt64))
  deriving Inhabited

/-- A digest of a location's 32 lanes' words, which a landing records instead of the words. -/
def digest (f : Fin 32 → UInt32) : UInt64 :=
  (List.finRange 32).foldl (fun h i => mixHash h (f i).toUInt64) 7

end NV
