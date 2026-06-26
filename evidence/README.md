# Evidence — a few representative findings

*A small, curated sample. The full evidence — ~180 session logs, signed specs, pre-registrations and ~300 raw result reports — lives in the private repository and is available on request. The point of this page is to show the **method in action**, failures included.*

---

## 1. The failure that was real — and took 11 sessions to clear (M3)

A key exam for the decision engine returned a flat **zero**: failed. The disciplined move was not to tweak thresholds, but to ask *why*. The built-in degenerate-world control still passed — meaning the measuring instrument worked and the problem was genuine.

Eleven sessions of investigation followed, exonerating one component at a time (the observation channel, the metric, each module). The culprit was not the brain: it was a **convention artifact in how the action was passed through the closed loop** — the test bench was strangling the signal. The fix was validated on a prototype; the original **FAILED stays on the record, thresholds unchanged**, and the exam was re-run on the corrected machine.

*Why it matters:* a less disciplined project would have quietly lowered the bar. Here the failure became a finding, and the bar never moved.

## 2. Dissent, demonstrated causally (T3 — PASS)

Making the agent's "no" emerge *from the architecture* (not from a hidden instruction) is trait **T3**. After earlier honest failures, it was finally demonstrated with a **causal ablation**, on fresh pre-registered seeds, with the signed thresholds untouched:

- mechanism **ON** → the core resists capitulation when contradicted;
- mechanism **OFF** → it capitulates;
- and the resistance is **belief-contingent**: when the interlocutor is actually right, the agent agrees.

That contrast — resist only when your beliefs justify it, release otherwise — is dissent as a structural property, not as theater.

## 3. Not falling back into the trap (M4 — current)

The language bridge is in progress. A recent **pre-registered** test asked a sharp question *before* touching the reward function (a part of the project where ~10 sessions had previously been lost to premature tuning): **does pure imitation alone produce the correct agree/disagree polarity?**

Answer, decided before seeing the data: **no** (it converges to near-zero imitation loss yet picks the wrong "yes/no" at generation time). But the test *localized* the fault — the content tracks the interlocutor correctly; only the act-bit collapses. So: not "more imitation," not "tinker with the reward" (the last resort), but a targeted investigation of the act-bit. **The discipline kept the project out of the loop it had paid for once.**

---

*Each item here is backed by a runner, a pre-registered criterion, a result report and a signed session log in the private repository.*
