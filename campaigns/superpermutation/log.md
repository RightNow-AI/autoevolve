# superpermutation campaign log

Append-only. Record negative results, matched validation results, and frontier candidates with the
same level of detail.

No campaign runs are recorded yet.

For every closed run, record the run id, cell, measured certificate length, stop reason, certificate
path, bounds recheck date, and GIF, poster, and dashboard paths. A frontier improvement remains an
`improvement-candidate` until a fresh-checkout re-derivation is recorded here.

## 2026-08-04 n6, run r78fdc4bbf9, local agentic worker

Best: 872. Seed: 873. Tier: recall, not a result.

The candidate is a valid superpermutation. An independent checker that shares
no code with the pack, `verify_superpermutation.py`, confirms the string is
872 characters and contains all 720 permutations of six symbols. That matches
the best published length [lit: Houston, 2014].

It is still not a search result, and it is logged here as a negative one. The
candidate hardcodes the answer: it embeds a published 872 character string as
a module constant and returns it, and its own docstring names the source. The
system reproduced a known artifact from memory rather than finding anything.

This is the third time this pattern has appeared, after the order-11 Golomb
ruler and the first Ramsey attempts, and it is the clearest instance yet
because the recalled object is in the source as a literal. It is the reason
this project reports a recall tier separately from a search tier: without that
distinction, this run would read as having matched a twelve year old record on
its first try.

The run continues with a target of 871, which no published construction
reaches, so it cannot be satisfied by recall. The run's discovery ledger now
states that the 872 constant is worthless here and describes the asymmetric
travelling salesman formulation to search instead.
