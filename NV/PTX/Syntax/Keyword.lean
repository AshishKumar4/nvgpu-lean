/-!
# PTX keywords

PTX spells types, state spaces and instruction qualifiers as keywords (`.u32`, `.shared::cta`,
`.rn`). Each keyword family is a Lean enumeration with an instance of `Keyword`: its spelling
without the leading dot, and the list of all its values. Parsing a keyword is a lookup in that
list; `Keyword.complete` proofs keep the lists exhaustive.
-/

namespace NV.PTX

class Keyword (α : Type) where
  all : List α
  name : α → String

namespace Keyword

/-- The value spelled `s`, if any. -/
def parse? {α : Type} [Keyword α] (s : String) : Option α :=
  (all : List α).find? (name · == s)

end Keyword

end NV.PTX
