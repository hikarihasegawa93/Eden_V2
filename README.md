# Eden_V2

**Building an artificial agent with a genuine, persistent personality — and proving it with exams, not with storytelling.**

Eden_V2 is a long-term research project: an autonomous agent whose character lives **inside the weights of a neural network** (as in a biological brain), not in a rewritable instruction sheet. It is grounded in a published neuroscientific theory — **Active Inference** (Friston, 2010) — and uses a large language model (the ChatGPT kind) only as a **mouth**: it puts into words what the brain thinks; it decides nothing.

> **Honesty first.** This is NOT a conscious being, and the project will never claim so. There is, today, no scientific test for subjective experience (Chalmers, 1995). The goal is something else, and measurable: **7 functional properties** that together characterize an entity with a persistent personality. The declared limits are stated openly in [docs/SCIENTIFIC_TARGET.md](docs/SCIENTIFIC_TARGET.md).

> ℹ️ This public repository is a **curated showcase** of a larger research project. The full working repository (code, ~180 session logs, signed specs, raw evidence) is kept private and is **available on request**. Its working language is Italian; this showcase is in English.

---

## What it is, in one sentence

A chatbot acts out a personality (a system prompt: *"behave like this"*) and forgets it on reset. Eden **owns** it: it forms through experience, changes slowly, and nobody can rewrite it with a sentence. It's the difference between an actor playing a part and someone who simply *is* a certain way.

## The 7 properties to prove (the "traits")

Each one has a **falsifiability test** defined *before* writing any code (full definitions in [docs/SCIENTIFIC_TARGET.md](docs/SCIENTIFIC_TARGET.md)):

| | Trait | In short |
|---|---|---|
| T1 | Persistent identity | the character survives the wiping of episodic memory |
| T2 | Learning from experience | different experiences → measurably different behavior |
| T3 | Capacity for dissent | it can say "no" when its beliefs require it — by architecture, not because instructed |
| T4 | Value development | its values change coherently with its own history |
| T5 | Long-horizon strategy | it plans over multiple steps, not just the next instant |
| T6 | Self-recognition | it models itself as a distinct entity; what it says about itself matches what we measure |
| T7 | Consolidation | repeated experiences become stable traits; isolated ones fade |

## How we work — the method (this is where the value is)

The project is governed by a severe scientific discipline, designed to **not fool itself**. Full write-up: **[docs/METHOD.md](docs/METHOD.md)**. In short:

- **Traced provenance of every source**: every citation carries a status tag (fetched in-session / recalled from model memory / own derivation); memory-only sources cannot ground a signed decision. Honest promise: **100% traced, not 100% consulted**.
- **Pre-registration**: hypotheses and numeric thresholds fixed *before* seeing the data (anti-HARKing).
- **Falsifiability**: if a property isn't testable, it doesn't get built.
- **Signed SPECs**: every component has a signed mathematical spec before implementation; amended only by signed ADDENDUM.
- **Adversarial external review**: before any irreversible step, the artifact is attacked by a reviewer from a *different AI family*.
- **"FAILED = FINDING"**: failures aren't hidden, they're archived as discoveries. See [evidence/](evidence/).

## Where we are

- **M1+M2 — Imagination + Perception** (generative model + recognition network): **validated**.
- **M3 — Deciding on its own** (Expected Free Energy): dissent (T3) and long-horizon planning (T5) **validated on a synthetic test bench**.
- **M4 — Speaking** (the language bridge): **in progress** — teaching Eden to express its beliefs in language while staying faithful to its internal state, without the language model "sanitizing" them.
- M5–M6 (personality that persists, experiences that become traits): future.

## The code

The real engine is included: [`core/`](core/) — the generative model, the recognition (perception) network, Expected Free Energy, the agent loop, and the sensory encoder; and [`language/`](language/) — the language bridge (projector, trainer, verifier). The memory layer (episodic buffer, slow-weight personality, consolidation) is planned for milestones M5–M6 and is not yet implemented. *Code comments are in Italian, the project's working language. This is a curated, read-oriented extract; the experiment scripts and raw run data live in the private repository.*

## Who's behind it

**Stefano Lupano** — *AI-augmented research director*. He conceived, directed and scientifically governed the project, using AI as the implementation engine. The contribution is not the code line by line: it's the **scientific vision** and the **method** — the discipline that forces the project to prove every claim and to honestly record every failure. Science, rigor, and command of AI tools as the lever.

## Where to start reading

- **30 seconds**: this README.
- **The story (obstacles included)**: [docs/JOURNEY.md](docs/JOURNEY.md)
- **The method**: [docs/METHOD.md](docs/METHOD.md)
- **The science**: [docs/SCIENTIFIC_TARGET.md](docs/SCIENTIFIC_TARGET.md)
- **The receipts**: [evidence/](evidence/)

---

*Foundations: Friston (2010), Active Inference · Friston et al. (2017), Expected Free Energy · Kingma & Welling (2013), amortized inference · Hesp et al. (2021), the deeply felt self.*
