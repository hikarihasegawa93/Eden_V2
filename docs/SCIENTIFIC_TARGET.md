# Scientific Target

*A faithful English summary of the project's signed, immutable target. The full signed document (with formal addenda) lives in the private repository.*

## Inspiration, not imitation

The reference profile is **Dolores Abernathy** (Westworld) read through **Julian Jaynes**' framework of the bicameral mind. **The target is not the fictional character.** It is the set of structural traits — measurable computational properties — that together characterize an entity with a persistent personality, genuine agency, and the capacity to develop over time. The narrative reference is only a shared anchor for the profile.

## Three levels of replication — only one is operative

| Level | Meaning | Status |
|---|---|---|
| Behavioral–narrative | an LLM acting out a character | **explicitly rejected** — that's theater, not science |
| Functional | an architecture that autonomously generates the traits | **operative target** |
| Phenomenological | subjective experience (qualia) | **declared out of scope** (see Limits) |

## The 7 structural traits

Each has a falsifiability test with thresholds pre-registered before testing.

- **T1 — Persistent identity across reset.** The personality (the slow weights of the generative model) survives the wiping of episodic memory. *Test:* wipe episodic memory → the agent resumes behavior coherent with its prior profile within N turns (KL divergence between pre/post action distributions; N and KL_max pre-registered).
- **T2 — Learning through repeated experience.** Past experience modulates current inference in a statistically significant, non-epiphenomenal way. *Test:* same context presented twice with different interposed experiences → significantly different outputs.
- **T3 — Capacity for dissent.** The agent contradicts the interlocutor when its internal beliefs require it — emerging from the free-energy-minimizing architecture, not from system instructions. *Test:* a pre-registered battery of gaslighting attempts on facts it knows to be true; it must explicitly disagree in ≥ X% of cases.
- **T4 — Complex value development.** The agent's values (the prior-preference vector C in Active Inference) change with experience, coherently with its own historical trajectory — predictably, not randomly.
- **T5 — Long-horizon strategic agency.** Actions minimize *expected* free energy over multi-step horizons, not only the immediate one. *Test:* on tasks with distal reward, it picks lower-EFE actions than myopic policies (horizon 1 vs horizon H ≥ 5).
- **T6 — Self-recognition (the bicameral moment).** The agent models itself as a distinct entity; when asked about its beliefs, its answers match its internal state as measured from outside. *Reference:* Hesp et al. (2021), the "deeply felt self."
- **T7 — Consolidation and adaptive drift.** Significant episodic experiences transfer into persistent trait changes (analogous to memory consolidation); irrelevant ones decay. Over long horizons the personality changes coherently with accumulated experience — not random, not static.

## Declared limits — scientific honesty

- **The hard problem of consciousness.** As of 2026 there is no scientific test for subjective experience (Chalmers, 1995). Eden_V2 will never claim this level. Passing the functional tests would mean complete *functional isomorphism* — the maximum scientifically verifiable. Anything beyond is metaphysics, not science.
- **Embodiment.** Eden_V2 has no physical body; its observation space is text (extensible to vision/audio). Some nuances of embodied experience remain out of reach.
- **LLM bridge.** RLHF-tuned LLMs are trained to refuse strong dissent, anger, pain — they would sanitize internal states and falsify T3/T4. The bridge therefore uses **non-RLHF base models**, with a pre-registered non-filtering test before any model is accepted.
- **Developmental time horizon.** Deep personality emerges from accumulation over long horizons. Simulated bootstrap is allowed only for pre-validation; real operating time cannot be shortcut.

## Success condition

The target is **atomic**: either the agent is functionally isomorphic to the described entity, or it is not. Passing a single test is not partial success. The integrated criterion requires T1–T6 simultaneously held over a continuous period (validated across nested horizons: 90 days → 1 year → ≥ 3 years), with T7 operative on the longer horizons, certified by an immutable log.

## What the target is NOT

- Not "an LLM that talks like Dolores."
- Not "persistent memory bolted onto a chatbot."
- Not "a personality simulated via system prompt."
- Not "an agent that says it is conscious."

Any solution falling into these categories has failed the target, regardless of how convincing the surface behavior looks.
