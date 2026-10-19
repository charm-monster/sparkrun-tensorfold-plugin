# No fallbacks

Logic code MUST NOT substitute a plausible value for a missing or invalid one. When a value is
absent or invalid, raise, return `None`, or propagate the absence.

Prohibited:

* Default argument values that paper over required data.
* `try/except` that swallows an error and returns a substitute.
* `or` / `if x is None` chains that invent a value.
* An `else` branch that quietly handles a state no one designed for.

Required recipe keys are validated in `recipe_block.py`. A missing key is a `RecipeError` that
names the key. Optional keys stay absent in the rendered command; absence is not a default.

A fallback is permitted only when it is a named, designed use case, a comment says why it is
correct, and a test asserts the fallback value.
