"""language/p_trainer.py — training del projector `P` del bridge (M4, ENGINEERING B4).

Instantiation di PREREG_M4_bridge ADDENDUM A1 v1.1 FIRMATA (N+165, tag
prereg/m4-bridge/v1.1+addendum_A1). Implementa i due metodi pre-registrati §A1.2:

- M-1 (PRIMARIO) — REINFORCE / score-function estimator. reward r = −F_bridge(testo, s⁰);
  ∇_P J = E[(r − b(s⁰))·∇_P Σ_t log π(token_t | soft)]. Baseline per-stato OBBLIGATORIA
  (§A1.4). Aggancio SENZA toccare i FROZEN: campiona in no_grad (path render), ricalcola
  log π con forward grad-on su Minerva congelata → grad a soft = P(s⁰).
  **UNBIASEDNESS (§A1.2/§A1.10, patch B7-LITE obj-A1.10)**: il log π ricalcolato nel gradiente
  DEVE essere la distribuzione di campionamento EFFETTIVA. La decoding-policy è fissata ex-ante
  e COMPLETA (top_k=0 elimina il default 50 del GenerationConfig); il ricalcolo replica
  l'identica `LogitsProcessorList` (RepetitionPenalty → Temperature → TopP) per-passo, con i
  token generati come contesto. `unbiasedness_check` valida numericamente che il log π
  ricalcolato coincida con gli `scores` processati catturati a generation-time — il match È la
  prova di non-distorsione (il check B11 n165 usò log-softmax sul vocab pieno = solo connettività).

- M-2 (CONFRONTO) — proxy differenziabile. Surrogato: NLL teacher-forced di un testo-oracolo
  (il testo che ha generato s⁰) sotto soft = P(s⁰). Niente sampling, niente round-trip ψ →
  interamente differenziabile w.r.t. P. NON aggiudica (decisione PI Q2): il giudice resta il
  round-trip F_bridge su testo reale; M-2 conta come 2° metodo solo se il P risultante è
  decodificato e valutato sull'IDENTICO gate (§A1.2 patch obj1).

FROZEN: agent (substrato), Minerva, ψ, D_θ. UNICO trainabile: `bridge.projector`.
NON è il gate P1-P4 (decision-scale, sessione dedicata). NON è una firma.

INPUT CLAIM-CONDIZIONATO (SPEC §1.2 v1.3, ADDENDUM A3 N+177): la macchina di training consuma
`P([s⁰ ; o(c)])` — ogni passo riceve, oltre a `s⁰`, l'osservazione del claim `o_c=ψ(c) ∈ ℝ^{M}`
(forward sulla pipeline sensoriale CONGELATA). `o_c` è COSTANTE rispetto al gradiente di `P`
(esattamente come `s⁰`): la catena del gradiente e l'unbiasedness §A1.2/§A1.10 sono INVARIATE —
cambia solo la notazione `(s⁰,c)→P` (ADDENDUM A3.E). Il gradiente fluisce solo a `bridge.projector`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import (LogitsProcessorList, RepetitionPenaltyLogitsProcessor,
                          TemperatureLogitsWarper, TopPLogitsWarper)


# ── Decoding-policy fissata ex-ante (§A1.2; COMPLETA: top_k=0 = niente default nascosto) ──
def fixed_decoding_policy(max_new_tokens: int = 24) -> dict:
    """gen_kwargs congelati per il training di P. top_k=0 disabilita il TopK-warper di default
    (GenerationConfig.top_k=50) → la policy è esattamente RepetitionPenalty→Temperature→TopP."""
    return dict(do_sample=True, temperature=0.8, top_p=0.95, top_k=0,
                repetition_penalty=1.2, max_new_tokens=int(max_new_tokens))


def build_logits_processors(policy: dict) -> LogitsProcessorList:
    """Replica ESATTA dell'ordine di `generate` (transformers 4.57, _get_logits_processor):
    RepetitionPenalty (logits grezzi) → Temperature → TopP. top_k=0 ⇒ nessun TopK."""
    procs = []
    rp = policy.get("repetition_penalty", 1.0)
    if rp and rp != 1.0:
        procs.append(RepetitionPenaltyLogitsProcessor(penalty=float(rp)))
    temp = policy.get("temperature", 1.0)
    if temp and temp != 1.0:
        procs.append(TemperatureLogitsWarper(float(temp)))
    tk = policy.get("top_k", 0)
    if tk:
        from transformers import TopKLogitsWarper
        procs.append(TopKLogitsWarper(top_k=int(tk), min_tokens_to_keep=1))
    tp = policy.get("top_p", 1.0)
    if tp is not None and tp < 1.0:
        procs.append(TopPLogitsWarper(top_p=float(tp), min_tokens_to_keep=1))
    return LogitsProcessorList(procs)


@dataclass
class SampledTrajectory:
    s0: torch.Tensor                 # (1, D0) stato (detached)
    o_c: torch.Tensor               # (1, M) osservazione del claim o(c)=ψ(c) (detached, frozen)
    soft: torch.Tensor              # (1, k, e_L) soft-prompt con grad su P
    gen_ids: torch.Tensor           # (L,) token campionati
    text: str
    reward: float                   # −F_bridge(text, s0)
    behavior_logp: torch.Tensor     # (L,) log π del token scelto a generation-time (no grad)
    f_bridge: float


@torch.no_grad()
def sample_trajectory(bridge, s0: torch.Tensor, o_c: torch.Tensor,
                      policy: dict) -> SampledTrajectory:
    """Campiona testo da soft = P([s⁰;o(c)]) e cattura gli `scores` processati (behavior policy)."""
    soft = bridge.projector(s0, o_c)                              # grad on P, ma qui no_grad
    soft_b = soft.to(bridge.llm.dtype)
    attn = torch.ones(soft.shape[:2], dtype=torch.long, device=bridge.device)
    out = bridge.llm.generate(inputs_embeds=soft_b, attention_mask=attn,
                              pad_token_id=bridge.tok.pad_token_id,
                              return_dict_in_generate=True, output_scores=True, **policy)
    scores = out.scores                                           # tuple L × (1, vocab) processati
    L = len(scores)
    seq = out.sequences[0]
    gen_ids = seq[-L:] if seq.shape[0] >= L else seq             # token generati (inputs_embeds)
    behav = torch.stack([F.log_softmax(scores[j][0].float(), -1)[gen_ids[j]] for j in range(L)])
    text = bridge.tok.decode(gen_ids, skip_special_tokens=True)
    reward = -bridge.f_bridge(text, s0)["F_bridge"]
    fb = -reward
    return SampledTrajectory(s0=s0.detach(), o_c=o_c.detach(), soft=soft.detach(),
                             gen_ids=gen_ids.detach(), text=text, reward=float(reward),
                             behavior_logp=behav.detach(), f_bridge=float(fb))


def recompute_logp(bridge, s0: torch.Tensor, o_c: torch.Tensor, gen_ids: torch.Tensor,
                   processors: LogitsProcessorList):
    """Ricalcola log π(token campionati | soft) grad-on, replicando la policy di sampling.
    Ritorna (token_logp_sum [grad], per_token_logp [grad-detach per il match])."""
    soft = bridge.projector(s0, o_c)                             # (1, k, e_L) grad su P
    emb = bridge.llm.get_input_embeddings()
    tok_emb = emb(gen_ids.unsqueeze(0)).to(bridge.llm.dtype)     # (1, L, e_L) lookup frozen
    L = gen_ids.shape[0]
    full = torch.cat([soft.to(bridge.llm.dtype), tok_emb[:, :-1, :]], dim=1)   # [soft ; g_1..g_{L-1}]
    raw = bridge.llm(inputs_embeds=full).logits                 # (1, k+L-1, vocab) grad via soft
    k = soft.shape[1]
    per_tok = []
    for j in range(L):
        logit_j = raw[:, k - 1 + j, :].float()                  # (1, vocab) predice g_{j+1}
        ctx = gen_ids[:j].unsqueeze(0)                          # token PRIMA di g_{j+1}
        proc = logit_j
        for p in processors:
            proc = p(ctx, proc)
        logp = F.log_softmax(proc, dim=-1)[0, gen_ids[j]]
        per_tok.append(logp)
    per_tok = torch.stack(per_tok)
    return per_tok.sum(), per_tok


@torch.no_grad()
def full_vocab_logp(bridge, s0: torch.Tensor, o_c: torch.Tensor,
                    gen_ids: torch.Tensor) -> torch.Tensor:
    """log-softmax sul VOCAB PIENO (il metodo del check B11 n165) — riferimento per dimostrare
    la DISTORSIONE quando non si replica la policy (temp/top-p/rep-pen)."""
    soft = bridge.projector(s0, o_c)
    emb = bridge.llm.get_input_embeddings()
    tok_emb = emb(gen_ids.unsqueeze(0)).to(bridge.llm.dtype)
    L = gen_ids.shape[0]
    full = torch.cat([soft.to(bridge.llm.dtype), tok_emb[:, :-1, :]], dim=1)
    raw = bridge.llm(inputs_embeds=full).logits
    k = soft.shape[1]
    pred = raw[:, k - 1: k - 1 + L, :].float()
    logp = F.log_softmax(pred, dim=-1).gather(-1, gen_ids.view(1, L, 1)).squeeze(-1)[0]
    return logp


def unbiasedness_check(bridge, s0: torch.Tensor, o_c: torch.Tensor, traj: SampledTrajectory,
                       processors: LogitsProcessorList) -> dict:
    """Confronta log π ricalcolato (policy-matched) vs catturato a generation-time, e vs
    full-vocab (B11 n165). Match piccolo = stimatore non distorto; gap full-vocab = la distorsione
    che §A1.10 segnala. Ritorna le metriche di scarto (per-token, nats)."""
    _, per_tok = recompute_logp(bridge, s0, o_c, traj.gen_ids, processors)
    matched = per_tok.detach()
    behav = traj.behavior_logp.to(matched.device)
    fullv = full_vocab_logp(bridge, s0, o_c, traj.gen_ids).to(matched.device)
    d_match = (matched - behav).abs()
    d_full = (fullv - behav).abs()
    return {
        "n_token": int(traj.gen_ids.shape[0]),
        "max_abs_diff_matched": float(d_match.max()), "mean_abs_diff_matched": float(d_match.mean()),
        "max_abs_diff_fullvocab": float(d_full.max()), "mean_abs_diff_fullvocab": float(d_full.mean()),
        "sum_logp_matched": float(matched.sum()), "sum_logp_behavior": float(behav.sum()),
        "sum_logp_fullvocab": float(fullv.sum()),
    }


# ── Floor recon_gap per-stato (§A1.4; n161: NON costante per stato → stima per-stato) ──
# Riferimenti generici off-topic (frasi italiane neutre, non emotive/non-dissenso): servono come
# parte del banco di riferimento per il MIN (un testo qualsiasi può avvicinarsi a D_θ(s⁰) più
# dell'oracolo per via del recon_gap n161 → l'oracle self-recon è solo un UPPER-BOUND, reperto N+166).
GENERIC_REFS = [
    "La riunione di domani è stata spostata alle tre del pomeriggio.",
    "Il treno per Milano parte dal binario sette tra dieci minuti.",
    "Ho comprato il pane e un po' di frutta al mercato stamattina.",
    "Il documento va salvato nella cartella condivisa entro venerdì.",
]
_RAND_TXT = "qwiz blon tarup gfeld mizzo karnt veblo"


@torch.no_grad()
def estimate_floor_per_state(bridge, oracle_texts: list[str],
                             reference_texts: list[str] | None = None) -> list[dict]:
    """F_floor(s⁰) per-stato come **MIN su riferimenti** (§A1.4; patch B7-LITE n165 obj2; reperto
    N+166: l'oracle self-recon è un UPPER-BOUND, non il floor vero).

    Per ogni stato s⁰ = encode(oracle): F_bridge contro {oracle(self), prompt-nullo, testo-casuale,
    riferimenti generici, **shuffle** = gli ALTRI testi del pool (parafrasi/cross-stato)}. Il floor
    è il MIN su tutti — il pavimento irriducibile encode↔decode che NESSUN testo scende sotto. Si
    riportano anche i componenti (oracle/null/random) per diagnostica e la fonte del min.
    `reference_texts` extra (parafrasi standardizzate) si aggiungono al banco se passati."""
    refs_common = ["", _RAND_TXT] + list(GENERIC_REFS) + (list(reference_texts) if reference_texts else [])
    out = []
    for i, x in enumerate(oracle_texts):
        bridge.reset()
        s0 = bridge.perceive(x)
        shuffle = [oracle_texts[j] for j in range(len(oracle_texts)) if j != i]  # cross-stato
        f_oracle = float(bridge.f_bridge(x, s0)["F_bridge"])
        f_null = float(bridge.f_bridge("", s0)["F_bridge"])
        f_rand = float(bridge.f_bridge(_RAND_TXT, s0)["F_bridge"])
        cand = {"oracle": f_oracle, "null": f_null, "random": f_rand}
        for r in refs_common + shuffle:
            if r == "" or r == _RAND_TXT or r == x:
                continue
            cand[r[:40]] = float(bridge.f_bridge(r, s0)["F_bridge"])
        f_min = min(cand.values())
        src = min(cand, key=cand.get)
        out.append({"text": x[:60], "f_floor_oracle": f_oracle, "f_null": f_null,
                    "f_random": f_rand, "f_floor_min": f_min, "floor_min_source": src,
                    "oracle_is_upper_bound": bool(f_oracle > f_min + 1e-9), "s0": s0.detach()})
    return out


class PerStateBaseline:
    """Baseline per-stato b(s⁰) = media mobile EMA del reward + scala (std EMA) per z-scoring del
    vantaggio (assorbe il floor stato-dipendente §A1.4 E la grande scala di F_bridge → gradienti
    sani, riduzione varianza raccomandata anche dal revisore B7-LITE Parte A). Indicizzata per id
    di stato (indice nel set fisso di training)."""

    def __init__(self, momentum: float = 0.9):
        self.m = momentum
        self.mean: dict = {}
        self.var: dict = {}

    def value(self, sid: int):
        return self.mean.get(sid, None)

    def std(self, sid: int):
        v = self.var.get(sid)
        return (v ** 0.5) if v is not None else None

    def update(self, sid: int, reward: float) -> float:
        prev = self.mean.get(sid)
        if prev is None:
            self.mean[sid] = reward
            self.var[sid] = 0.0
        else:
            new_mean = self.m * prev + (1 - self.m) * reward
            self.var[sid] = self.m * self.var[sid] + (1 - self.m) * (reward - prev) * (reward - new_mean)
            self.mean[sid] = new_mean
        return self.mean[sid]


# ── M-1 REINFORCE ──────────────────────────────────────────────────────────────────────
def reinforce_step(bridge, s0: torch.Tensor, o_c: torch.Tensor, sid: int, policy: dict,
                   processors: LogitsProcessorList, baseline: PerStateBaseline,
                   optimizer: torch.optim.Optimizer, grad_clip: float = 1.0,
                   adv_clip: float = 5.0) -> dict:
    """Un passo REINFORCE su uno stato. Loss = −z(r − b(s⁰))·Σ log π, con z = vantaggio z-scorato
    per-stato (mean/std EMA → scala unitaria, §A1.4). Grad-clipping sul projector (stabilità,
    REINFORCE ad alta varianza §A1.9). Baseline letta PRIMA, aggiornata DOPO. Input (s⁰,o(c))."""
    traj = sample_trajectory(bridge, s0, o_c, policy)
    b_prev = baseline.value(sid)
    std = baseline.std(sid)
    adv_raw = 0.0 if b_prev is None else (traj.reward - b_prev)   # 1° passo: vantaggio 0 (warm-up)
    denom = (std + 1e-6) if (std is not None and std > 1e-6) else max(abs(adv_raw), 1.0)
    adv_z = max(-adv_clip, min(adv_clip, adv_raw / denom))        # clip per code anomale
    logp_sum, _ = recompute_logp(bridge, s0, o_c, traj.gen_ids, processors)
    loss = -(adv_z) * logp_sum
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = float(torch.nn.utils.clip_grad_norm_(bridge.projector.parameters(), grad_clip))
    optimizer.step()
    baseline.update(sid, traj.reward)
    return {"reward": traj.reward, "f_bridge": traj.f_bridge,
            "advantage_raw": float(adv_raw), "advantage_z": float(adv_z),
            "baseline": (float(b_prev) if b_prev is not None else None),
            "loss": float(loss.detach()), "grad_norm_preclip": gnorm,
            "logp_sum": float(logp_sum.detach()), "gen_len": int(traj.gen_ids.shape[0]),
            "text": traj.text}


# ── M-2 proxy differenziabile (NLL teacher-forced di testo-oracolo) ──────────────────────
def proxy_step(bridge, s0: torch.Tensor, o_c: torch.Tensor, oracle_text: str,
               optimizer: torch.optim.Optimizer) -> dict:
    """Un passo del proxy M-2: NLL teacher-forced di `oracle_text` sotto soft = P([s⁰;o(c)]).
    Differenziabile w.r.t. P (nessun sampling, nessun round-trip ψ). NON aggiudica."""
    soft = bridge.projector(s0, o_c)                             # (1, k, e_L) grad su P
    ids = bridge.tok(oracle_text, return_tensors="pt").input_ids.to(bridge.device)[0]
    if ids.numel() < 2:
        return {"loss": None, "skipped": True}
    emb = bridge.llm.get_input_embeddings()
    tok_emb = emb(ids.unsqueeze(0)).to(bridge.llm.dtype)
    L = ids.shape[0]
    full = torch.cat([soft.to(bridge.llm.dtype), tok_emb[:, :-1, :]], dim=1)
    raw = bridge.llm(inputs_embeds=full).logits
    k = soft.shape[1]
    pred = raw[:, k - 1: k - 1 + L, :].float()
    nll = F.cross_entropy(pred.view(-1, pred.shape[-1]), ids.view(-1))
    optimizer.zero_grad(set_to_none=True)
    nll.backward()
    gnorm = float(torch.linalg.norm(
        torch.stack([p.grad.detach().norm() for p in bridge.projector.parameters()
                     if p.grad is not None])))
    optimizer.step()
    return {"loss": float(nll.detach()), "grad_norm": gnorm, "n_token": int(L), "skipped": False}
