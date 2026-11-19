import NV.Core.Engine
import NV.Arch.H200

/-!
`lake exe engine-smoke`: a hand-assembled kernel run on the H200 parameters:
every thread stores `3 * tid + ctaid` at `out[ctaid * ntid + tid]`. Two CTAs of 100 threads, so
the last warp of each is partial. Checks every output word and that the run takes time.
-/

open NV NV.SASS NV.Core

def r (i : Nat) : Operand := .reg ⟨.r, i⟩
def ctrl (stall : Nat) (wbar : Option (Fin 6) := none) (wait : Nat := 0) : Ctrl :=
  { stall, yield := false, wbar, rbar := none, wait, reuse := 0 }

def code : Array Instr := #[
  { pc := 0x00, opcode := "S2R", operands := #[r 0, .name "SR_TID.X"], ctrl := ctrl 1 (some 0) },
  { pc := 0x10, opcode := "S2R", operands := #[r 3, .name "SR_CTAID.X"], ctrl := ctrl 1 (some 1) },
  { pc := 0x20, opcode := "LDC", mods := #["64"], operands := #[r 4, .cbank 0 none 0x210],
    ctrl := ctrl 1 (some 2) },
  { pc := 0x30, opcode := "IMAD", operands := #[r 2, r 3, .cbank 0 none 0x0, r 0], ctrl := ctrl 1 none 0b11 },
  { pc := 0x40, opcode := "IMAD", operands := #[r 6, r 0, .imm 3, r 3], ctrl := ctrl 4 },
  { pc := 0x50, opcode := "IMAD", mods := #["WIDE"], operands := #[r 4, r 2, .imm 4, r 4],
    ctrl := ctrl 4 none 0b100 },
  { pc := 0x60, opcode := "STG", mods := #["E"],
    operands := #[.mem (some ⟨.r, 4⟩) true none (some ⟨.ur, 4⟩) 0, r 6], ctrl := ctrl 1 },
  { pc := 0x70, opcode := "EXIT", ctrl := ctrl 5 }
]

def base : UInt64 := 0x7f0000000000

def main : IO UInt32 := do
  let kernel : Kernel := { name := "smoke", code, regs := 8, paramBytes := 8 }
  let params := ByteArray.mk ((List.range 8).map (fun i => (base >>> (8 * i).toUInt64).toUInt8)).toArray
  let launch : Launch := { kernel, grid := (2, 1, 1), block := (100, 1, 1), params,
                           global := Store.empty base 1024 }
  match run NV.Arch.h200 launch with
  | .error e => IO.eprintln s!"engine-smoke: {e}"; return 1
  | .ok res =>
    let out := res.global.read base 800
    let mut bad := 0
    for i in [0:200] do
      let want := (3 * (i % 100) + i / 100).toUInt32
      if out[i]! != want then
        bad := bad + 1
        if bad ≤ 5 then IO.eprintln s!"out[{i}] = {out[i]!}, expected {want}"
    IO.println s!"engine-smoke: {repr res.counters}"
    if bad > 0 || res.counters.cycles == 0 || res.counters.instExecuted != 8 * 8 then
      IO.eprintln s!"engine-smoke: {bad} wrong outputs"; return 1
    return 0
