# The Method — how Eden_V2 avoids fooling itself

Most of what makes this project credible is not a single result: it's a **discipline**, designed so the project cannot lie to itself. This page summarizes it in plain English. (In the private repository the full ruleset is enforced operationally as rules **B1–B16**.)

## The core rules

**1. Traced provenance of every source (B1).**
Every citation — paper, formula, theory, external number — carries a status tag:
- *fetched in-session* (with an openable link/DOI),
- *recalled from the model's memory* (a sanity-check only — **it cannot ground a signed decision** until verified),
- *own derivation* (with explicit, verifiable steps).

The promise is deliberately honest: **100% traced, not 100% consulted** — paywalled full text isn't always openable, and we say so rather than pretend.

**2. Pre-registration.**
Hypotheses and numeric thresholds are fixed *before* looking at the data. This blocks HARKing (hypothesizing after the results are known) and the temptation to move the goalposts.

**3. Falsifiability before construction.**
If a property cannot be tested and falsified, it does not get built. Every milestone defines its failure test before its code.

**4. Signed specifications.**
Each component has a signed mathematical specification *before* implementation. A signed spec is not edited in place — it is amended through a signed ADDENDUM, with every dependent component reviewed. This keeps a multi-year project from drifting.

**5. Adversarial external review.**
Before any scientifically irreversible step, the artifact is handed to a reviewer from a **different AI family**, explicitly instructed to attack it — to treat the project's own self-defenses as theses to be demolished, and to hunt for the objections we did *not* raise. Convergence with our choice counts as weak evidence; a new objection counts as a strong alarm.

**6. "FAILED = FINDING."**
Failures are not hidden or quietly retried. They are archived as findings, with thresholds left untouched. A large part of this project is the honest history of what did *not* work and why — and that history is what makes a success believable. See [evidence/](../evidence/).

## Why this is the point

A demo can be faked; a discipline cannot. The artifacts of this method — pre-registrations, signed specs, external review minutes, the catalogue of honest failures — are the actual deliverable a serious reviewer should weigh. They demonstrate something rarer than a working prototype: a research process that refuses to deceive itself, and a director who governs an AI toolchain with that rigor.

## Trait-by-trait mapping

Every component must serve at least one of the [7 target traits](SCIENTIFIC_TARGET.md). If a piece doesn't map to a trait, it isn't built. This keeps the project anchored to its falsifiable goal rather than wandering into capability for its own sake.
