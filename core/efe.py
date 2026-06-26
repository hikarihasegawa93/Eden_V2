"""Expected Free Energy + selezione azione — SPEC_efe v1.1 FIRMATA (v1.0 N+74 +
ADDENDUM A2 N+114, §3 come rivisto da §E2.2.4-§E2.2.6).

Autorizzato dal ri-gate P2 PASS N+119 (forma vincente CONCAT, tag
milestone/m3-rigate/p2-pass-n119). Claim ristretto §A8.5: questo modulo è il
SUBSTRATO del meccanismo di selezione azione EFE, NON la validazione di T3/T5
(quella resta a PREREG_M3, SPEC_efe §8).

STATUTO M3 (§E2.2.4/§E2.2.5): planner = risk-dominant, deterministic-latent EFE
approximation — NON EFE pieno.
  - Rollout (adv) mean-only: la catena latente propaga solo la media (l'incertezza
    NON si accumula lungo H). Caveat misurato N+119: fedeltà mean-only vs MC
    moderata (dev ~2× rumore, Spearman ρ~0.43) → il chiamante può auditare con
    rollout MC esterno (le traiettorie complete sono esposte in PlanResult).
  - Σ_o^pred PROPAGATA (il risk NON è mean-only): Σ_o ≈ J·Σ_s·Jᵀ + γ_o⁻¹·I_M con
    J = ∂D_θ/∂s⁰|_{s⁰_pred(a)} (J=J(a), §3.3). [DERIVAZIONE PROPRIA] Σ_s = Γ_s0⁻¹ =
    diag(exp(−log γ_s0,d)): generalizzazione anisotropa (SG-A7, SPEC_generative
    v1.7 §A7.2) della formula firmata §3.3 (γ_s0⁻¹·I_128), a cui si riduce nel
    caso isotropo; stessa propagazione gaussiana del decoder linearizzato.
  - Epistemic incluso in forma piena ma DEBOLE a M3 (§4.4/§E2.2.5): dipende da
    (γ_o, γ_s⁰, J(a)), non dal valore dell'osservazione; nessun claim di curiosità.
  - Ambiguity costante tra policy (§4.3, γ_o omoschedastico): inclusa in G per
    fedeltà a §4.5, irrilevante per argmin/softmax (shift-invariante).
  - Mode A (s¹ frozen intra-rollout) = PRIMARIA per attribuzione allo slot L0;
    Mode B (slow drift via f_θ) disponibile (§E2.2.4 patch BLOCC.4).

Contratto operativo (§6.1 + caveat N+119): receding-horizon/MPC OBBLIGATORIO —
si esegue SOLO la prima azione di π* e si ri-pianifica al turno successivo con il
posterior aggiornato. MAI eseguire π* open-loop: la calibrazione predicted-G vs
esito realizzato è debole (Spearman ρ~0.3, planner smoke N+119).

Vincoli ereditati (§0.3 rev. A2):
  1. Read-only: legge f_θ, μ⁰_θ (azione-condizionata SG-A8), D_θ, γ congelati al
     planning; NON scrive Slow/Fast/s². Nessun optimizer in questo modulo.
  3. Rollout DAL POSTERIOR: (s0, s1) in input sono μ_φ⁰, μ_φ¹ (recognition mode,
     SPEC_recognition §3.4) — contratto del chiamante; mai dal prior (OQ32 → M5).
  4. EFE legge γ (clamped_log_gammas) per pesare risk/epistemic.
  5. Dimensioni invariate: s⁰∈ℝ^128, s¹∈ℝ^192, o∈ℝ^256; m_τ=0 a M3.

Preference prior C = N(μ_C, γ_C⁻¹·I_M) (§5): μ_C è FORNITA dal chiamante.
La costruzione μ_C = g(s²) (§5.1) è degenere a M3 (s²≡0, reperto N+75, ADDENDUM
A1 superato — scope NON riaperto da A2 §E2.4); l'ancoraggio per-scenario è
fissato dal test/PREREG. A M5 μ_C torna a g(s²_value) (interfaccia §5.3).

Fonti (B1, [FETCHED] in SPEC_efe §Provenienza/§E2.1): G(π) Eq. 2.5 e
P(π)=σ(−γ·G) Friston et al. 2017 Neural Computation; risk+ambiguity arXiv
2402.14460; epistemic gaussiano Fountas et al. 2020 NeurIPS; transizione
azione-condizionata Kaelbling 1998 / Friston 2017 B(u) / Hafner 2019.
KL/entropia/info-gain gaussiani: [DERIVAZIONE PROPRIA] §4.2-§4.4 SPEC_efe.
"""
import math
from dataclasses import dataclass

import torch

from core.generative import D0, D2, D_M, M, GenerativeModel

FORM_VALIDATED = "concat"   # ri-gate P2 N+119: vince la concat (regola ex-ante PREREG §4)
MAX_ENUM = 200_000          # enumerazione esaustiva su Π piccolo (§1); MCTS/pruning → §11

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class PreferencePrior:
    """C(o) = N(μ_C, γ_C⁻¹·I_M) — SPEC_efe §5.1 (μ_C dal chiamante, vedi docstring modulo)."""
    mu_c: torch.Tensor
    gamma_c: float


@dataclass(frozen=True)
class PlannerConfig:
    """horizon=H (§1); gamma_pol = precisione sulle politiche (§6.1);
    mode: "A" s¹ frozen (primaria §E2.2.4) | "B" slow drift via f_θ."""
    horizon: int
    gamma_pol: float = 1.0
    mode: str = "A"
    jac_chunk: int = 512


@dataclass
class PlanResult:
    """Tabella completa su Π (lessicografico = itertools.product) — esposta integralmente
    perché la calibrazione predicted-G è debole (ρ~0.3 N+119): i test/audit a valle
    leggono componenti e traiettorie, non solo a*."""
    policies: torch.Tensor       # (P, H) long
    g: torch.Tensor              # (P,)  G(π) = Σ_τ[risk−epistemic] + H·ambiguity (§4.5)
    risk: torch.Tensor           # (P, H)
    epistemic: torch.Tensor      # (P, H)
    ambiguity_per_step: float
    p_policies: torch.Tensor     # (P,)  σ(−γ_pol·G)  (§6.1)
    best_idx: int
    a_star: int                  # prima azione di π* (receding-horizon §6.1)
    o_pred: torch.Tensor         # (P, H, M)  medie decodificate (audit B5/MC esterni)
    s0_traj: torch.Tensor        # (P, H, D0)
    mode: str


class EFEPlanner:
    """Selezione azione via G(π) su rollout generativo dal posterior (§3 rev. A2 + §6)."""

    def __init__(self, generative: GenerativeModel, allow_unvalidated_form: bool = False):
        t_l0 = generative.transition_l0
        if t_l0.k_actions <= 0:
            raise ValueError("EFEPlanner richiede la macchina azione-esplicita SG-A8 "
                             "(k_actions > 0); macchina firmata k_actions=0 non pianificabile")
        if t_l0.action_form != FORM_VALIDATED and not allow_unvalidated_form:
            raise ValueError(
                f"forma {t_l0.action_form!r} non validata dal ri-gate N+119 "
                f"(vincente: {FORM_VALIDATED!r}); override esplicito con "
                f"allow_unvalidated_form=True solo per diagnostica")
        self.gen = generative
        self.k = t_l0.k_actions

    def _jtj(self, s0_pred: torch.Tensor, chunk: int) -> torch.Tensor:
        """JᵀJ con J = ∂D_θ/∂s⁰|_{s0_pred} (linearizzazione locale §3.3/§7.3.1)."""
        jac_fn = torch.func.vmap(torch.func.jacrev(self.gen.decoder))
        out = []
        with torch.enable_grad():
            for i in range(0, s0_pred.shape[0], chunk):
                J = jac_fn(s0_pred[i:i + chunk].detach())          # (b, M, D0)
                out.append(J.transpose(1, 2) @ J)
        return torch.cat(out)                                       # (n, D0, D0)

    @torch.no_grad()
    def plan(
        self,
        s0: torch.Tensor,
        s1: torch.Tensor,
        prefs: PreferencePrior,
        cfg: PlannerConfig,
        s2: torch.Tensor | None = None,
    ) -> PlanResult:
        """Enumerazione esaustiva di Π=A^H + G(π) closed-form gaussiano (§3.3 primario).

        (s0, s1) = posterior corrente (μ_φ⁰, μ_φ¹) — Vincolo 3. Un solo start per call;
        il chiamante ri-pianifica ogni turno (MPC §6.1).
        """
        if cfg.mode not in ("A", "B"):
            raise ValueError(f"mode deve essere 'A' o 'B', ricevuto {cfg.mode!r}")
        if cfg.horizon < 1:
            raise ValueError(f"horizon deve essere ≥ 1, ricevuto {cfg.horizon}")
        K, H = self.k, cfg.horizon
        P = K ** H
        if P > MAX_ENUM:
            raise ValueError(f"|Π|=K^H={P} > {MAX_ENUM}: enumerazione esaustiva fuori "
                             f"scope M3 (§1); MCTS/pruning deferiti (§11)")
        s0 = s0.reshape(1, D0)
        s1 = s1.reshape(1, -1)
        device = s0.device
        if prefs.mu_c.shape[-1] != M:
            raise ValueError(f"mu_c deve avere dim {M}, ricevuto {tuple(prefs.mu_c.shape)}")
        mu_c = prefs.mu_c.reshape(1, M).to(device)
        gamma_c = float(prefs.gamma_c)
        s2_t = torch.zeros(1, D2, device=device) if s2 is None else s2.reshape(1, D2)
        m_tau = torch.zeros(1, D_M, device=device)

        lgo, lgs0, _ = self.gen.clamped_log_gammas()
        log_gamma_o = float(lgo)
        gamma_o = math.exp(log_gamma_o)
        s_var = torch.exp(-lgs0).detach()                            # Σ_s diag (SG-A7)
        sqrt_s = torch.exp(-0.5 * lgs0).detach()
        eye_d0 = torch.eye(D0, device=device)
        eye_k = torch.eye(K, device=device)
        ambiguity = 0.5 * M * (_LOG_2PI + 1.0 - log_gamma_o)         # §4.3, costante

        nodes0 = s0                                                  # (K^d, D0) a profondità d
        s1_cur = s1                                                  # (1, D1): f_θ autonoma ⇒
        risk_d, epi_d, omu_d, s0_d = [], [], [], []                  # s¹ condiviso a ogni d
        for _ in range(H):
            if cfg.mode == "B":
                s1_cur = self.gen.transition_l1(s1_cur, s2_t, m_tau)
            n = nodes0.shape[0]
            rep0 = nodes0.repeat_interleave(K, 0)
            rep1 = s1_cur.expand(n * K, -1)
            a_oh = eye_k.repeat(n, 1)                                # azione = digit più rapido
            s0_pred = self.gen.transition_l0(rep0, rep1, a_oh)       # (L0) §E2.2.4
            o_mu = self.gen.decoder(s0_pred)                         # (obs) media
            jtj = self._jtj(s0_pred, cfg.jac_chunk)
            # M_τ = I + γ_o·S^½ JᵀJ S^½ : lndet condiviso da epistemic e ln|Σ_o^pred|
            m_mat = eye_d0 + gamma_o * (jtj * sqrt_s.view(1, -1, 1) * sqrt_s.view(1, 1, -1))
            chol = torch.linalg.cholesky(m_mat)
            lndet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
            tr_sigma_o = (jtj.diagonal(dim1=-2, dim2=-1) * s_var).sum(-1) + M / gamma_o
            quad = ((o_mu - mu_c) ** 2).sum(-1)
            # §4.2: D_KL[N(D_θ(s⁰_pred), Σ_o^pred) ‖ N(μ_C, γ_C⁻¹·I)] in forma chiusa
            risk = 0.5 * (gamma_c * tr_sigma_o + gamma_c * quad - M
                          - M * math.log(gamma_c) + M * log_gamma_o - lndet)
            epi = 0.5 * lndet                                        # §4.4
            risk_d.append(risk)
            epi_d.append(epi)
            omu_d.append(o_mu)
            s0_d.append(s0_pred)
            nodes0 = s0_pred                                         # (adv) mean-only

        def _leaves(t: torch.Tensor, d: int) -> torch.Tensor:
            return t.repeat_interleave(K ** (H - 1 - d), 0)

        risk_t = torch.stack([_leaves(risk_d[d], d) for d in range(H)], 1)   # (P, H)
        epi_t = torch.stack([_leaves(epi_d[d], d) for d in range(H)], 1)
        o_pred = torch.stack([_leaves(omu_d[d], d) for d in range(H)], 1)    # (P, H, M)
        s0_traj = torch.stack([_leaves(s0_d[d], d) for d in range(H)], 1)
        g = risk_t.sum(1) - epi_t.sum(1) + H * ambiguity             # §4.5
        p_pol = torch.softmax(-cfg.gamma_pol * g, dim=0)             # §6.1
        idx = torch.arange(P, device=device)
        policies = torch.stack([(idx // K ** (H - 1 - d)) % K for d in range(H)], 1)
        best = int(g.argmin())                                       # π* = argmin G
        return PlanResult(
            policies=policies, g=g, risk=risk_t, epistemic=epi_t,
            ambiguity_per_step=float(ambiguity), p_policies=p_pol,
            best_idx=best, a_star=int(policies[best, 0]),
            o_pred=o_pred, s0_traj=s0_traj, mode=cfg.mode,
        )
