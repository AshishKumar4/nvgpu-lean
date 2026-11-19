import NV.Launch.Load
import NV.Arch.Parts
import NV.Formal.TraceCheck

/-!
`lake exe trace-check SPEC.json [--gpu h200|…]`: run a launch spec with every warp's trace recorded
(`Launch.recordTrace`) and replay each trace against the program semantics
(`NV.Formal.Trace.check`). A warp whose trace replays ran, on the engine, a run of `Outcomes`
(`Trace.check_run`). Prints per launch how many warps replay, and for the first that does not,
the event its replay stops at; and checks that the first warp's trace with one landed digest changed
does not replay. Exits 1 unless every warp replays and the changed one does not.
-/

open NV NV.Core NV.Launch NV.Formal

def describe : TraceEv → String
  | .issue pc _ wb => s!"issue of pc {pc}" ++ (match wb with | some id => s!", writeback {id}" | none => "")
  | .land id _ => s!"landing of writeback {id}"

def main (args : List String) : IO UInt32 := do
  let some path := args.find? (!·.startsWith "--") | do
    IO.eprintln "usage: lake exe trace-check SPEC.json [--gpu h200|h100|…]"; return 2
  let gpu := match args.dropWhile (· != "--gpu") with
    | _ :: g :: _ => g
    | _ => "h200"
  let some p := NV.Arch.parts.lookup gpu | do IO.eprintln s!"no parameter set {gpu}"; return 2
  let path : System.FilePath := path
  let spec ← IO.ofExcept (Spec.parse (← IO.FS.readFile path))
  match ← simulate p spec (path.parent.getD ".") (recordTrace := true) with
  | .error e => IO.eprintln e; return 1
  | .ok r =>
    let mut ok := true
    for (k, wts) in r.traces do
      let ops := SASS.decode k
      let opAt (pc : Nat) : SASS.Op := ops[pc]!
      let replays (wt : WarpTrace) (n : Nat) : Bool :=
        (Trace.check (warpStep opAt wt.env) (warpFp opAt) wt.init [] (wt.evs.toList.take n)).isSome
      let good := wts.filter fun wt => replays wt wt.evs.size
      IO.println s!"{gpu} {k.name}: {good.size} of {wts.size} warps replay"
      if let some wt := wts.find? fun wt => !replays wt wt.evs.size then
        ok := false
        -- Replay is sequential, so the prefixes that replay are those shorter than the first event
        -- that fails.
        let mut lo := 0
        let mut hi := wt.evs.size
        while lo + 1 < hi do
          let mid := (lo + hi) / 2
          if replays wt mid then lo := mid else hi := mid
        IO.println s!"  CTA {wt.cta} warp {wt.warp}: stops at event {lo} of {wt.evs.size}, the {describe wt.evs[lo]!}"
      -- The replay compares what it computes with what the engine left: a trace whose first
      -- compared word is changed must not replay.
      if let some wt := good[0]? then
        if let some i := wt.evs.findIdx? fun | .land _ vals => vals.any (·.isSome) | _ => false then
          let changed := match wt.evs[i]! with
            | .land id vals =>
              let j := (vals.findIdx? (·.isSome)).getD 0
              TraceEv.land id (vals.modify j (·.map (· ^^^ 1)))
            | e => e
          if replays { wt with evs := wt.evs.set! i changed } wt.evs.size then
            IO.println s!"  a changed landing at event {i} still replays"
            ok := false
          else IO.println s!"  CTA {wt.cta} warp {wt.warp} with landing {i} changed: refused"
    return if ok then 0 else 1
