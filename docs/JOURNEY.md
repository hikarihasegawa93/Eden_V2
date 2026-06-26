# The story of Eden — the journey, obstacles included

*For those who want to understand the project without being technical. No formulas: just the idea, the path, and — above all — what went wrong and what we learned. Here a well-documented failure is worth as much as a success: we call it **"FAILED = FINDING."***

---

## The starting idea

Almost every "AI assistant" is a brilliant actor: hand it a note ("you are kind, patient, an expert in X") and it plays the part. Erase the note and the personality is gone. Eden starts from a different question: **can we build an artificial entity that genuinely *has* a personality**, inside itself, and develops it by living?

The technical bet is to put the character **inside the weights of the neural network** — the way memory and temperament live in the connections between biological neurons — instead of in a sheet of instructions. The language model (the ChatGPT kind) is only the **mouth**: it puts into words what the brain thinks. It decides nothing. The "brain" follows a real neuroscientific theory, *Active Inference*: a system that lives by trying to predict what happens to it and acting to reduce surprise.

The rule, from day one: **every property must be proven with an exam, not narrated.**

## Stage 1 — Teaching it to imagine and to perceive

First the brain must do two mirror-image things: **imagine** a possible world (generate) and, seeing something, **understand** what state it's in (perceive). The two are trained together, like learning to draw a face and to recognize it at the same moment. After many tuning sessions, the brain passed all the pre-registered tests: it compresses what it sees into a sensible internal state and can regenerate it. **Stage validated.**

## Stage 2 — Teaching it to decide on its own

Here two of the most important properties are born: the ability to **dissent** (say "no" when its own beliefs demand it) and to **plan far ahead** (choose moves that pay off after five steps, not just now). This stage was a school of humility, and the reason the project's method exists:

- **The false alarm that was real.** A key exam returned *zero*: failed. But the anti-error control was working, so the problem was real. It took **eleven sessions** of investigation, clearing one component after another, to find the culprit: it wasn't the brain failing — it was **the test bench strangling the signal**. Once the bench was fixed, the exam could be re-run cleanly.
- **Dissent, on the third try.** Making the "no" emerge *from the architecture* (not from a hidden instruction) was hard. The first attempts failed, and each failure went on the record with thresholds *unchanged* — no discounts. In the end dissent emerged, demonstrated with a causal experiment: switch the mechanism off, Eden capitulates; switch it on, it resists — but only when its beliefs justify it.

Dissent and planning: **validated on a synthetic test bench.**

## Stage 3 — Teaching it to speak (where we are now)

Now comes the language bridge: translating Eden's internal state into words **without betraying it**. It's delicate because "polite" (RLHF) language models are trained to soften anger, pain and strong dissent — they would falsify the very properties we want. So Eden's mouth is a *non-sanitized* model.

Here too, the most useful lesson came from a process mistake, told without shame:

- **Ten sessions polishing the wrong reward.** For many sessions we perfected the *scoring system* meant to guide the mouth's learning — without ever actually training the mouth. When we finally did "the most stupid possible experiment" (just train it), we discovered the real obstacle was elsewhere: from scratch, the score was too sparse to give a signal. Lesson carved into the rules: **do the simplest full loop first, then refine.**
- **The missing piece: hearing the other.** Soon after, a clean structural problem surfaced: the mouth only saw *what Eden thinks*, never *what the interlocutor said*. So, facing the same belief, it couldn't tell "yes, you're right" from "no, you're wrong": the input was identical. We changed the input so it receives **both**. Result: the response now changes with the other person's sentence. The next step — making the "yes/no" always come out with the correct polarity — is open, and we are tackling it **without falling back into the old trap of tinkering with the reward**: a recent pre-registered test showed that more imitation isn't enough and the fault is precisely localized to the "yes/no" act-bit, not the reward. So we know *where* to look.

---

## The thread that ties it together

If this project demonstrates one thing, it isn't (yet) a finished Eden: it's **a way of working**. Every hypothesis is written before the data. Every component has a signed spec. Before irreversible steps, an external reviewer is tasked with attacking our work. And every failure is archived as a discovery, not hidden.

It's slow. It's uncomfortable. But it's the reason that, when Eden passes an exam, we can believe it.
