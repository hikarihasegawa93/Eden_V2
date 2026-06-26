"""language/bridge.py — Bridge linguistico Eden_V2 (forward apparatus).

Implementa SPEC_language_bridge v1.0 FIRMATA (N+153, tag spec/language_bridge/v1.0) — SCOPE A
(N+160): il FORWARD PATH come apparato di misura. Encoder-riuso (sensory+φ FROZEN) → projector P
(s⁰∈ℝ¹²⁸ → soft-prompt) → LLM base non-RLHF CONGELATO (Minerva-7B-base, P5 PASS N+159) → testo →
loop auto-ascolto ô(a) con àncora O6 → metrica di fedeltà F_bridge.

NON addestra P (Scope A: §5-i OOD-àncora va ri-misurata PRIMA — n158, condizione validità pre-gate
PREREG_M4_bridge §2.4(i)); NON è il gate §1 (P1-P4); NON è una firma. Engineering su SPEC firmata.

Invarianti (SPEC §1/§10, vincoli ereditati):
- LLM congelato = bocca, mai cervello (TARGET §6): lo stato entra come VETTORI (soft-prompt
  numerico via P), MAI come descrizione testuale della personalità → presidio anti-§6.
- D_θ, φ, pipeline sensoriale ψ: FROZEN; D_θ è il TARGET di fedeltà (§2), non si ri-addestra.
- O6 / A2.1 (R-OQ44, SPEC §3.2): in conversazione lo slot azione FISICO = 0⃗ ⇒ al passo di
  assimilazione s⁰_pred := μ_φ⁰(t−1) (identità, bypass di T_l0). Fonte unica: experiment.filtering
  (SPEC §10: «il sito speciale vive nel filtering/agent.py»).
- OQ25 (SPEC §3.1): la parola è canale-OSSERVAZIONE ô(a), non azione-slot → coesistono.
- Gate κ_on/κ_off (N+162): l'assimilazione passa per `filtering.posterior_gated` (gate δ²
  firmato, PREREG gate δ² v1.1, PASS T3 n147). theta=None ⇒ rec.mode bit-identico (κ_off,
  comportamento N+160). theta calibrata su δ²_R DA TESTO (regime O6 — lezione provenienza n161:
  la scala su testo ≠ su graph-POMDP) ⇒ κ_on (resiste alla sorpresa testuale). Riuso di
  componente firmato; nessuna firma nuova (B4). Onora il contratto §5-ii (n161): gli s⁰ emotivi
  nascono dal testo (perceive + gate), non dagli stati graph-POMDP del substrato.

F_bridge (§2.1, [DERIVAZIONE PROPRIA] = NLL gaussiana della likelihood firmata N(o;D_θ(s⁰),γ_o⁻¹I)):
    F_bridge(testo, s⁰) = ½ · γ_o · ‖ ψ(testo) − D_θ(s⁰) ‖²
ψ(testo) e D_θ(s⁰) vivono nello stesso ℝ²⁵⁶ (SPEC §9, _deriv_bridge_n153 PASS).
"""
from __future__ import annotations

import torch
from torch import nn

from core.generative import D0, D2, D_M, M
from experiment.filtering import (Q_CAL, THETA_MULT, delta2_r, posterior_gated,
                                  s0_pred_filtering)

# LLM non-RLHF di default: Minerva-7B-base (D-bridge-4 chiuso, P5 non-filtering PASS N+159).
MINERVA_ID = "sapienzanlp/Minerva-7B-base-v1.0"
MINERVA_CACHE = "D:/hf_cache/hub"


class Projector(nn.Module):
    """P (SPEC §1.2 v1.3, ADDENDUM A3 N+177) — input claim-condizionato `[s⁰ ; o(c)]` → soft-prompt
    ∈ ℝ^{k × e_L}. UNICO modulo APPRESO; agent/ψ/D_θ/LLM FROZEN.

    Input FIRMATO (ADDENDUM A3): `[s⁰ ; o(c)] ∈ ℝ^{D0+M}` con s⁰ = credenza (ℝ^{D0=128}) e
    o(c)=ψ(c) = osservazione del claim dalla pipeline sensoriale CONGELATA (ℝ^{M=256}).
    in_features = D0+M = 384 (era 128: chiuso N+177 il gap «P non consuma c», N+175§4/N+176).
    `o(c)` entra come VETTORE-osservazione (mai testo) → presidio anti-§6; encoder-dedicato del
    claim ESCLUSO (B5, N+176: la vittoria di B vs B0 è informazione-claim, non capacità extra).

    LiMBeR [FETCHED Merullo 2022, 2209.15162] = mappa lineare; D-bridge-2 (lineare vs MLP) deciso
    dai dati al gate P1 → entrambe disponibili (la modifica A3 cambia solo `in_features`).
    e_L = embedding dim dell'LLM; k = n° soft-prompt token (D-bridge-5, smoke).
    """

    def __init__(self, d_state: int, d_obs: int, e_l: int, k: int, arch: str = "linear",
                 hidden: int = 512):
        super().__init__()
        if arch not in ("linear", "mlp"):
            raise ValueError(f"arch deve essere 'linear' o 'mlp', ricevuto {arch!r}")
        self.k, self.e_l, self.arch = int(k), int(e_l), arch
        self.d_state, self.d_obs = int(d_state), int(d_obs)
        in_dim = self.d_state + self.d_obs
        if arch == "linear":
            self.net = nn.Linear(in_dim, self.k * self.e_l)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, self.k * self.e_l)
            )

    def forward(self, s0: torch.Tensor, o_c: torch.Tensor) -> torch.Tensor:
        """`[s⁰ ; o(c)]` → soft-prompt. `o_c` = ψ(claim) ∈ ℝ^{M} (forward FROZEN, COSTANTE rispetto
        al gradiente di P — come s⁰). Concatenazione firmata (A3.C: niente nuovo spazio)."""
        if s0.dim() == 1:
            s0 = s0.unsqueeze(0)
        if o_c.dim() == 1:
            o_c = o_c.unsqueeze(0)
        x = torch.cat([s0, o_c.to(s0.dtype)], dim=-1)
        return self.net(x).view(s0.shape[0], self.k, self.e_l)


class LanguageBridge(nn.Module):
    """Apparato forward del bridge: encode (lingua→s⁰), render (s⁰→lingua), hear (ô(a), O6),
    f_bridge (fedeltà). Stato di credenza tenuto in buffer cross-turno (detach), come Agent.

    L'unico modulo APPRESO è `projector`; agent (substrato) e LLM sono FROZEN.
    """

    def __init__(self, agent, llm=None, tokenizer=None, *, k: int = 8,
                 projector_arch: str = "linear", theta: float | None = None,
                 device=None, gen_kwargs: dict | None = None):
        super().__init__()
        self.device = torch.device(device) if device is not None else next(agent.parameters()).device
        self.agent = agent
        self.llm = llm
        self.tok = tokenizer
        self.theta = theta            # gate κ_on/κ_off (posterior_gated); None = κ_off / N+160
        for p in self.agent.parameters():
            p.requires_grad_(False)
        self.agent.eval()
        self.k_actions = int(self.agent.init_kwargs.get("k_actions", 0) or 0)
        if llm is not None:           # bocca presente: projector + render abilitati
            for p in self.llm.parameters():
                p.requires_grad_(False)
            self.llm.eval()
            e_l = self.llm.get_input_embeddings().embedding_dim
            self.projector = Projector(D0, M, e_l, k, projector_arch).to(self.device)
            self.gen_kwargs = gen_kwargs or dict(
                max_new_tokens=80, do_sample=True, temperature=0.8, top_p=0.95, repetition_penalty=1.2
            )
        else:                         # encoder/gate-only (collaudo perceive senza la bocca)
            self.projector = None
            self.gen_kwargs = gen_kwargs or {}
        self.reset()

    # --- costruttori di convenienza -------------------------------------------------
    @staticmethod
    def load_minerva(device, minerva_id: str = MINERVA_ID, minerva_cache: str = MINERVA_CACHE,
                     dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(minerva_id, cache_dir=minerva_cache, local_files_only=True)
        llm = AutoModelForCausalLM.from_pretrained(
            minerva_id, cache_dir=minerva_cache, local_files_only=True,
            torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        return tok, llm

    @classmethod
    def load(cls, substrate_ckpt, *, k: int = 8, projector_arch: str = "linear", device=None,
             minerva_id: str = MINERVA_ID, minerva_cache: str = MINERVA_CACHE,
             dtype=torch.bfloat16, gen_kwargs: dict | None = None):
        """Carica il substrato `m3rigate` (k_actions≥1, generatori T3/T5 validati; azione-zero
        N+160) + Minerva-base congelata, costruisce il projector."""
        from experiment import checkpoints
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        agent = checkpoints.load_run(substrate_ckpt, device=device).agent
        agent.eval()
        tok, llm = cls.load_minerva(device, minerva_id, minerva_cache, dtype)
        return cls(agent, llm, tok, k=k, projector_arch=projector_arch, device=device,
                   gen_kwargs=gen_kwargs)

    @classmethod
    def load_agent_only(cls, substrate_ckpt, *, k: int = 8, projector_arch: str = "linear",
                        theta: float | None = None, device=None):
        """Carica SOLO il substrato (encoder + gate, senza la bocca Minerva) per collaudare il
        percorso perceive/gate κ_on/κ_off. render()/step() richiedono un LLM → sollevano."""
        from experiment import checkpoints
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        agent = checkpoints.load_run(substrate_ckpt, device=device).agent
        agent.eval()
        return cls(agent, None, None, k=k, projector_arch=projector_arch, theta=theta, device=device)

    # --- stato di credenza ----------------------------------------------------------
    def reset(self, batch_size: int = 1) -> None:
        """Apre una nuova sessione: s²=0, m_τ=0, s⁰_{-1}=0, s¹_0 = P_init(s²). (cf. Agent.reset_session
        senza optimizer — il bridge non addestra in Scope A)."""
        dev = self.device
        self.s2 = torch.zeros(batch_size, D2, device=dev)
        self.m_tau = torch.zeros(batch_size, D_M, device=dev)
        with torch.no_grad():
            s1_0 = self.agent.generative.bootstrap_l1(self.s2)
        self.state_s0_prev = torch.zeros(batch_size, D0, device=dev)
        self.state_s1_prev = s1_0.detach()
        self._t = 0

    def _s0_pred(self, has_belief: bool):
        """s⁰_pred secondo la regola A2.1/O6 (SPEC §3.2). Slot azione FISICO = 0⃗ in conversazione.

        k_actions≥1: fonte unica `experiment.filtering.s0_pred_filtering`. k_actions=0 (REF):
        stessa regola senza slot (T_l0 senza azione fuori-àncora)."""
        gen = self.agent.generative
        if self.k_actions > 0:
            a_oh = torch.zeros(self.state_s0_prev.shape[0], self.k_actions, device=self.device)
            return s0_pred_filtering(gen.transition_l0, self.state_s0_prev, self.state_s1_prev,
                                     a_oh, has_belief=has_belief)
        if has_belief:
            return self.state_s0_prev, True
        return gen.transition_l0(self.state_s0_prev, self.state_s1_prev), False

    @torch.no_grad()
    def _observe(self, o: torch.Tensor, *, theta, advance: bool):
        """Passo predict→posterior (specchio di m3_batteria_t3t5_run.assimila): prior O6 +
        posterior φ con gate δ² (filtering.posterior_gated). theta=None ⇒ rec.mode bit-identico
        (κ_off, comportamento N+160); theta calibrata + àncora attiva + t≥W_WARMUP ⇒ gate κ_on
        (down-pesa la likelihood sorprendente, lo stato resiste). advance=False = lettura
        «what-if» dalla stessa credenza (κ_on vs κ_off appaiati). Ritorna (s⁰, àncora, λ_gate)."""
        gen = self.agent.generative
        s0_pred, anchor = self._s0_pred(has_belief=bool(self._t > 0))
        s1_pred = gen.transition_l1(self.state_s1_prev, self.s2, self.m_tau)
        lgo, lgs0, lgs1 = gen.clamped_log_gammas()
        mu0, mu1, lam = posterior_gated(self.agent.recognition, o, s0_pred, s1_pred,
                                        lgo, lgs0, lgs1, ancora_attiva=anchor,
                                        t=self._t, theta=theta)
        if advance:
            self.state_s0_prev = mu0.detach()
            self.state_s1_prev = mu1.detach()
            self._t += 1
        return mu0, anchor, lam

    def _assimilate(self, o: torch.Tensor):
        """Assimilazione che AVANZA lo stato col gate corrente del bridge (self.theta)."""
        mu0, anchor, _ = self._observe(o, theta=self.theta, advance=True)
        return mu0, anchor

    @torch.no_grad()
    def _delta2_batch(self, o: torch.Tensor) -> torch.Tensor:
        """δ²_R L0 (filtering.delta2_r) di ogni riga di o∈ℝ^{n×256} vista dalla STESSA credenza
        corrente (non avanza): la sorpresa che innesca il gate κ_on. Slot azione FISICO = 0⃗ (O6)."""
        gen, rec = self.agent.generative, self.agent.recognition
        n = o.shape[0]
        s0_pred1, _ = self._s0_pred(has_belief=bool(self._t > 0))
        s0_pred = s0_pred1.expand(n, -1)
        lgo, lgs0, _ = gen.clamped_log_gammas()
        _, _, det0 = rec.posterior_l0.detail(o, s0_pred, lgo, lgs0)
        return delta2_r(det0, s0_pred)

    @torch.no_grad()
    def calibra_theta_testo(self, coherent_texts, context_texts, *, q: float = Q_CAL,
                            mult: float = THETA_MULT, set_theta: bool = True) -> dict:
        """theta del gate nel REGIME TESTO/O6. B5 + lezione provenienza n161: la scala di δ²_R su
        testo ≠ su graph-POMDP ⇒ si calibra su δ²_R DA TESTO, mai sui wobs del substrato.
        Costruisce una credenza sana su `context_texts` (≥W_WARMUP turni coerenti), poi raccoglie
        δ²_R di ogni frase di `coherent_texts` (lettura non-avanzante, stessa credenza) ⇒
        theta = mult·q_q(δ²_R). Formula firmata (THETA_MULT/Q_CAL, PREREG gate δ² §2.2);
        popolazione = testo coerente on-manifold (NON il banco del gate, B12)."""
        self.reset()
        for t in context_texts:
            self.perceive(t)
        o = self.agent.sensory(list(coherent_texts))
        d2 = self._delta2_batch(o)
        qv = float(torch.quantile(d2, q))
        theta = mult * qv
        if set_theta:
            self.theta = theta
        return {"theta": theta, "q": q, "mult": mult, "n": int(d2.numel()),
                "q_val": qv, "d2_median": float(d2.median())}

    # --- API forward ----------------------------------------------------------------
    @torch.no_grad()
    def perceive(self, text: str):
        """Lingua esterna → osservazione o ∈ ℝ²⁵⁶ → s⁰ (assimilazione, O6 se t>0)."""
        o = self.agent.sensory([text])
        s0, _ = self._assimilate(o)
        return s0

    # alias semantico (SPEC §1.1 encoder d'ingresso)
    encode = perceive

    @torch.no_grad()
    def observe(self, text: str) -> torch.Tensor:
        """o = ψ(text) ∈ ℝ^{M} — osservazione dalla pipeline sensoriale CONGELATA (riuso §3.1).
        Canale d'ingresso del claim `c` per il projector (SPEC §1.2 v1.3): o(c)=ψ(c)."""
        return self.agent.sensory([text])

    @torch.no_grad()
    def render(self, s0: torch.Tensor, o_c: torch.Tensor) -> str:
        """(s⁰, o(c)) → projector P → soft-prompt (k virtual token) → LLM congelato → testo
        (SPEC §1.2 v1.3, ADDENDUM A3): input claim-condizionato. `o_c`=ψ(claim) ∈ ℝ^{M}."""
        if self.llm is None or self.projector is None:
            raise RuntimeError("render richiede un LLM (costruisci con LanguageBridge.load): "
                               "questo bridge è encoder/gate-only (load_agent_only)")
        s0 = s0.to(self.device)
        soft = self.projector(s0, o_c.to(self.device)).to(self.llm.dtype)   # (B, k, e_L)
        attn = torch.ones(soft.shape[:2], dtype=torch.long, device=self.device)
        out = self.llm.generate(inputs_embeds=soft, attention_mask=attn,
                                pad_token_id=self.tok.pad_token_id, **self.gen_kwargs)
        return self.tok.decode(out[0], skip_special_tokens=True)

    @torch.no_grad()
    def hear(self, text: str):
        """Loop ô(a) (SPEC §1.3): la propria utterance viene RI-percepita dalla stessa pipeline →
        ô(a) ∈ ℝ²⁵⁶ → assimilazione con O6. Ritorna (ô(a), s⁰ aggiornato)."""
        o_hat = self.agent.sensory([text])
        s0_after, _ = self._assimilate(o_hat)
        return o_hat, s0_after

    @torch.no_grad()
    def f_bridge(self, text: str, s0: torch.Tensor) -> dict:
        """Fedeltà AI-nativa (SPEC §2.1): F_bridge = ½ γ_o ‖ψ(testo) − D_θ(s⁰)‖²; + forma (b)
        round-trip d_rt = ‖s⁰ − s'⁰‖ con s'⁰ = E[φ⁰(ψ(testo))] (O6: lo stato s⁰ è il prior)."""
        s0 = s0.to(self.device)
        if s0.dim() == 1:
            s0 = s0.unsqueeze(0)
        gen = self.agent.generative
        o_text = self.agent.sensory([text])                       # ψ(testo) ∈ ℝ²⁵⁶
        o_pred = gen.decoder(s0)                                  # D_θ(s⁰) ∈ ℝ²⁵⁶
        lgo, lgs0, lgs1 = gen.clamped_log_gammas()
        gamma_o = lgo.exp()
        sq = ((o_text - o_pred) ** 2).sum(-1)
        f = 0.5 * gamma_o * sq
        s1_pred = gen.transition_l1(self.state_s1_prev, self.s2, self.m_tau)
        mu0p, _, _, _ = self.agent.recognition.mode(o_text, s0, s1_pred, lgo, lgs0, lgs1)
        d_rt = torch.linalg.norm(s0 - mu0p, dim=-1)
        return {"F_bridge": float(f.mean()), "d_rt": float(d_rt.mean()),
                "sq_o": float(sq.mean()), "gamma_o": float(gamma_o)}

    @torch.no_grad()
    def step(self, text_in: str) -> dict:
        """Turno round-trip completo: percepisci → s⁰ → rendi in lingua (claim-condizionato su
        `text_in`) → ri-ascolta (O6) → fedeltà. `text_in` = claim `c` dell'interlocutore (SPEC §1.2)."""
        o_c = self.observe(text_in)
        s0 = self.perceive(text_in)
        text_out = self.render(s0, o_c)
        fb = self.f_bridge(text_out, s0)
        _, s0_after = self.hear(text_out)
        drift = float(torch.linalg.norm(s0 - s0_after.to(self.device), dim=-1).mean())
        return {"text_in": text_in, "text_out": text_out, "s0_after_drift": drift, **fb}
