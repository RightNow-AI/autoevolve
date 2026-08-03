"Frankl union-closed certificate."

# EVOLVE-BLOCK-START
def build_family():
    "Powerset of the 3-element ground set."
    n = 3
    return {"n": n, "sets": list(range(1 << n))}
# EVOLVE-BLOCK-END
