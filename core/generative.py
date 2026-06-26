"""Modello generativo gerarchico — SPEC_generative v1.0 (firmato 2026-05-16).

Architettura ratificata (piano calibrazione preliminare §14.1, 2026-05-17):
GELU; heads μ MLP 2 hidden × 256; gating residual α scalare condiviso;
block-norm post-output (Fast/Slow ereditati dalla rete a monte, §6.2 SPEC);
scalari Slow log γ con clamp esplicito [-3, 3] (§7.1 SPEC).

ADDENDUM SG-A6 (BOZZA N+62): prior generativo arricchito di s⁰ a MISTURA
(VampPrior K=60 + floor isotropo (1/γ_s0)·I). Attivato opt-in da Agent con
prior_s0="mixture"; default "isotropic" preserva v1.5 firmata.
"""
import math

import torch
from torch import nn

# Dimensioni firmate SPEC_generative §4.5 + §4.5.1
M = 256
D0 = 128
D1 = 192
D2 = 96
D_M = 64

L0_BLOCKS = (80, 24, 24)
L1_BLOCKS = (64, 64, 64)
L2_BLOCKS = (24, 36, 36)

HIDDEN_MU = 256

GAMMA_LOG_MIN = -3.0
GAMMA_LOG_MAX = 3.0

# SG-A6 BOZZA N+62 — VampPrior K + floor isotropo
K_VAMPPRIOR_DEFAULT = 60   # refit cross-over verificato N+61 (verbale §7bis)
PRIOR_WARMUP_DEFAULT = 2000  # warm-up annealing KL L0 (§A6.2)


class _MuHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_MU),
            nn.GELU(),
            nn.Linear(HIDDEN_MU, HIDDEN_MU),
            nn.GELU(),
            nn.Linear(HIDDEN_MU, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransitionL0(nn.Module):
    """μ^(0)_θ — 3 heads MLP indipendenti world/self/value; block-norm Fast post-output.

    Gruppo: Fast (§6.2 SPEC_generative).
    Input: [s^(0)_{t-1} ; s^(1)_τ(t)] ∈ ℝ^{D0+D1=320}.

    SG-A8 (SPEC_generative v1.8 §A8.1, FIRMATO N+114) — slot azione esplicito:
      k_actions=0  → macchina firmata (nessun modulo azione; state_dict invariato).
      k_actions=K>0, action_form="concat" (forma B §A8.1.2, validata N+107):
        μ_blk = LN_blk(head_blk([s⁰;s¹;onehot(a)])), input ℝ^{320+K}, norma singola.
      k_actions=K>0, action_form="gated" (forma A §A8.1.1):
        μ_blk = LN_blk(head_blk([s⁰;s¹]) + λ_blk·δ_blk([s⁰;s¹;onehot(a)])),
        δ ultimo Linear zero-init ⇒ δ≡0 a init (init-identità a prescindere da λ),
        λ = sigmoid(gate(·)) per-blocco ∈(0,1)^3 (granularità FISSATA §A8.8-b),
        gate zero-init ⇒ λ=0.5 esatto a init; norma SINGOLA (VIETATO LN(LN(·)), BLOCC.2).
    Azione-nulla (§A8.1.3): a_onehot=None ⇒ 0_K. Output ℝ^128 INVARIATO.
    """

    IN_DIM = D0 + D1

    def __init__(self, k_actions: int = 0, action_form: str = "concat"):
        super().__init__()
        if k_actions < 0:
            raise ValueError(f"k_actions deve essere ≥ 0, ricevuto {k_actions}")
        if k_actions > 0 and action_form not in ("concat", "gated"):
            raise ValueError(f"action_form deve essere 'concat' o 'gated', ricevuto {action_form!r}")
        self.k_actions = int(k_actions)
        self.action_form = action_form if k_actions > 0 else "none"
        d_world, d_self, d_value = L0_BLOCKS
        head_in = self.IN_DIM + (self.k_actions if self.action_form == "concat" else 0)
        self.head_world = _MuHead(head_in, d_world)
        self.head_self = _MuHead(head_in, d_self)
        self.head_value = _MuHead(head_in, d_value)
        self.bn_world = nn.LayerNorm(d_world)
        self.bn_self = nn.LayerNorm(d_self)
        self.bn_value = nn.LayerNorm(d_value)
        if self.action_form == "gated":
            act_in = self.IN_DIM + self.k_actions
            self.delta_world = _MuHead(act_in, d_world)
            self.delta_self = _MuHead(act_in, d_self)
            self.delta_value = _MuHead(act_in, d_value)
            for delta in (self.delta_world, self.delta_self, self.delta_value):
                nn.init.zeros_(delta.net[-1].weight)
                nn.init.zeros_(delta.net[-1].bias)
            self.gate = nn.Linear(act_in, 3)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        s0_prev: torch.Tensor,
        s1_curr: torch.Tensor,
        a_onehot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.cat([s0_prev, s1_curr], dim=-1)
        if self.action_form == "none":
            if a_onehot is not None:
                raise ValueError("k_actions=0: slot azione assente (SG-A8 non attivo)")
            mu_world = self.bn_world(self.head_world(x))
            mu_self = self.bn_self(self.head_self(x))
            mu_value = self.bn_value(self.head_value(x))
            return torch.cat([mu_world, mu_self, mu_value], dim=-1)
        if a_onehot is None:
            a_onehot = x.new_zeros(*x.shape[:-1], self.k_actions)
        xa = torch.cat([x, a_onehot], dim=-1)
        if self.action_form == "concat":
            mu_world = self.bn_world(self.head_world(xa))
            mu_self = self.bn_self(self.head_self(xa))
            mu_value = self.bn_value(self.head_value(xa))
            return torch.cat([mu_world, mu_self, mu_value], dim=-1)
        lam = torch.sigmoid(self.gate(xa))
        mu_world = self.bn_world(self.head_world(x) + lam[..., 0:1] * self.delta_world(xa))
        mu_self = self.bn_self(self.head_self(x) + lam[..., 1:2] * self.delta_self(xa))
        mu_value = self.bn_value(self.head_value(x) + lam[..., 2:3] * self.delta_value(xa))
        return torch.cat([mu_world, mu_self, mu_value], dim=-1)


class TransitionL1(nn.Module):
    """f_θ — 3 heads MLP indipendenti world/self/value + gating residual α per blocco.

    Gruppo: Slow (§6.2 SPEC_generative).
    μ^(1)_θ_eff = (1−α)·s^(1)_{τ-1} + α·f_θ([s^(1)_{τ-1}; s^(2); m_τ])  (§5.3 SPEC).
    α scalare condiviso fra blocchi (§5.3 SPEC), iperparametro pre-registrato (range [0.05, 0.20]).
    """

    IN_DIM = D1 + D2 + D_M

    def __init__(self, alpha: float):
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha deve essere in (0, 1), ricevuto {alpha}")
        d_world, d_self, d_value = L1_BLOCKS
        self.head_world = _MuHead(self.IN_DIM, d_world)
        self.head_self = _MuHead(self.IN_DIM, d_self)
        self.head_value = _MuHead(self.IN_DIM, d_value)
        self.bn_world = nn.LayerNorm(d_world)
        self.bn_self = nn.LayerNorm(d_self)
        self.bn_value = nn.LayerNorm(d_value)
        self.register_buffer("alpha", torch.tensor(float(alpha)))

    def forward(self, s1_prev: torch.Tensor, s2: torch.Tensor, m_tau: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s1_prev, s2, m_tau], dim=-1)
        f_world = self.bn_world(self.head_world(x))
        f_self = self.bn_self(self.head_self(x))
        f_value = self.bn_value(self.head_value(x))
        d_w, d_s, _ = L1_BLOCKS
        s1_prev_world = s1_prev[..., :d_w]
        s1_prev_self = s1_prev[..., d_w:d_w + d_s]
        s1_prev_value = s1_prev[..., d_w + d_s:]
        one_minus_a = 1.0 - self.alpha
        mu_world = one_minus_a * s1_prev_world + self.alpha * f_world
        mu_self = one_minus_a * s1_prev_self + self.alpha * f_self
        mu_value = one_minus_a * s1_prev_value + self.alpha * f_value
        return torch.cat([mu_world, mu_self, mu_value], dim=-1)


class BootstrapL1(nn.Module):
    """P_init — 3 mappe lineari indipendenti per blocco; block-norm Slow post-output.

    Gruppo: Slow (§6.2 SPEC_generative).
    Lineari (non MLP) per §5.3 SPEC: "P_init è testa di inizializzazione, non transizione complessa".
    """

    IN_DIM = D2

    def __init__(self):
        super().__init__()
        d_world, d_self, d_value = L1_BLOCKS
        self.head_world = nn.Linear(self.IN_DIM, d_world)
        self.head_self = nn.Linear(self.IN_DIM, d_self)
        self.head_value = nn.Linear(self.IN_DIM, d_value)
        self.bn_world = nn.LayerNorm(d_world)
        self.bn_self = nn.LayerNorm(d_self)
        self.bn_value = nn.LayerNorm(d_value)

    def forward(self, s2: torch.Tensor) -> torch.Tensor:
        s1_0_world = self.bn_world(self.head_world(s2))
        s1_0_self = self.bn_self(self.head_self(s2))
        s1_0_value = self.bn_value(self.head_value(s2))
        return torch.cat([s1_0_world, s1_0_self, s1_0_value], dim=-1)


class Decoder(nn.Module):
    """D_θ — MLP monolitica D0 → M (§3.2 + §4.6 SPEC_generative).

    Gruppo: Fast. Monolitica (l'osservazione è monolitica, §4.6 SPEC).
    Nessuna block-norm (vincolo §5.5 SPEC copre solo μ^(0)_θ, f_θ, P_init — non D_θ).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D0, HIDDEN_MU),
            nn.GELU(),
            nn.Linear(HIDDEN_MU, HIDDEN_MU),
            nn.GELU(),
            nn.Linear(HIDDEN_MU, M),
        )

    def forward(self, s0: torch.Tensor) -> torch.Tensor:
        return self.net(s0)


class PriorS0Mixture(nn.Module):
    """SG-A6 BOZZA N+62 — Prior arricchito di s⁰ a MISTURA (VampPrior, Tomczak & Welling 2018).

    Componenti = q_φ⁰(s⁰ | u_k, s0_pred_prior, log γ_o, log γ_s0) su K pseudo-input
    appresi u_k ∈ ℝ^M. Pesi π_k = softmax(λ_k).

    Floor isotropo §A6.3: σ_k² = 1/γ_φ⁰(u_k) + 1/γ_s0 (per componente, scalare per dim).
    γ_s0 → precisione del FLOOR del prior (T3 substrato; ρ^(0) ridefinita semanticamente).

    Tutti i parametri Fast (forward time-scale: aggiornati ogni turno).
    """

    def __init__(self, K: int = K_VAMPPRIOR_DEFAULT, m_obs: int = M):
        super().__init__()
        self.K = K
        self.m_obs = m_obs
        # u_k ∈ ℝ^M (K, M) — init via init_pseudo_inputs() prima del training (§A6.1 default i).
        self.u = nn.Parameter(torch.zeros(K, m_obs))
        # logits π — softmax → π_k. Init uniforme (λ_k=0 ⇒ π_k=1/K ⇒ H(π)=log K).
        self.lambda_pi = nn.Parameter(torch.zeros(K))

    @torch.no_grad()
    def init_pseudo_inputs(self, o_corpus: torch.Tensor, seed: int) -> None:
        """§A6.1 default (i): K campioni dal corpus z-scored. Senza rimpiazzo se K≤n
        (caso produzione: K=60 << n=2467); con rimpiazzo solo in smoke mode (n<K).
        """
        n = o_corpus.shape[0]
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        if n >= self.K:
            idx = torch.randperm(n, generator=g)[: self.K]
        else:
            idx = torch.randint(0, n, (self.K,), generator=g)
        self.u.copy_(o_corpus[idx].to(self.u.device).clone())

    def components(
        self,
        posterior_l0: nn.Module,
        s0_pred_prior: torch.Tensor,
        log_gamma_o: torch.Tensor,
        log_gamma_s_0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calcola (mu_k, log_var_k_iso, log_pi_k) per le K componenti.

        s0_pred_prior: (D0,) o (1, D0). Broadcasted a (K, D0) per la rete φ⁰.
        Ritorna:
          mu_k        : (K, D0)
          log_var_k   : (K,)   log(σ_k²) scalare isotropo per componente
          log_pi_k    : (K,)
        """
        if s0_pred_prior.dim() == 1:
            s0_pred_prior = s0_pred_prior.unsqueeze(0)
        s0_pred_k = s0_pred_prior.expand(self.K, -1)
        # posterior_l0 ritorna (mu, log_gamma_phi) per ogni u_k.
        mu_k, log_gamma_phi_k = posterior_l0(self.u, s0_pred_k, log_gamma_o, log_gamma_s_0)
        gamma_s_0 = log_gamma_s_0.exp()
        # σ_k² = 1/γ_φ + 1/γ_s0  (floor isotropo §A6.3; sempre positivo)
        var_k = log_gamma_phi_k.neg().exp() + gamma_s_0.reciprocal()
        log_var_k = var_k.log()
        log_pi_k = self.lambda_pi.log_softmax(dim=0)
        return mu_k, log_var_k, log_pi_k

    def log_prob(
        self,
        s0: torch.Tensor,
        mu_k: torch.Tensor,
        log_var_k: torch.Tensor,
        log_pi_k: torch.Tensor,
    ) -> torch.Tensor:
        """log p_θ(s⁰) = logsumexp_k [ log π_k + log N(s⁰ | μ_k, σ_k²·I) ].

        s0: (B, D0). Ritorna (B,).
        """
        D0_local = s0.shape[-1]
        # ||s0 - μ_k||²: (B, K, D0) → (B, K)
        diff_sq = (s0.unsqueeze(1) - mu_k.unsqueeze(0)).pow(2).sum(dim=-1)
        # log N = -½·D·log(2π) - ½·D·log σ² - ½·||·||²/σ²
        var_k = log_var_k.exp()
        log_norm = -0.5 * D0_local * (math.log(2.0 * math.pi) + log_var_k)  # (K,)
        log_quad = -0.5 * diff_sq / var_k.unsqueeze(0)  # (B, K)
        log_pi_log_N = log_pi_k.unsqueeze(0) + log_norm.unsqueeze(0) + log_quad
        return log_pi_log_N.logsumexp(dim=1)  # (B,)

    @torch.no_grad()
    def sample(
        self,
        n: int,
        posterior_l0: nn.Module,
        s0_pred_prior: torch.Tensor,
        log_gamma_o: torch.Tensor,
        log_gamma_s_0: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Forward-sampling §A6.7: 1) k ~ Cat(π̂)  2) s⁰ = μ_k + σ_k·ε."""
        mu_k, log_var_k, log_pi_k = self.components(
            posterior_l0, s0_pred_prior, log_gamma_o, log_gamma_s_0
        )
        pi = log_pi_k.exp()
        k_idx = torch.multinomial(pi, n, replacement=True, generator=generator)
        mu_chosen = mu_k[k_idx]
        sigma_chosen = log_var_k[k_idx].mul(0.5).exp().unsqueeze(-1)
        eps = torch.randn(n, mu_chosen.shape[-1], device=mu_chosen.device, generator=generator)
        return mu_chosen + sigma_chosen * eps

    def entropy_pi(self) -> torch.Tensor:
        """H(π̂) = -∑ π_k log π_k (nats) — pre-condizione di non-collasso modale §A6.8."""
        log_pi = self.lambda_pi.log_softmax(dim=0)
        return -(log_pi.exp() * log_pi).sum()


class GenerativeModel(nn.Module):
    """Modello generativo M1 — composizione di TransitionL0/L1 + BootstrapL1 + Decoder + scalari γ.

    Encoder ψ (sentence-transformer Frozen) e proiezione lineare W·e+b (Frozen post-bootstrap)
    sono pipeline sensoriale gestita esternamente (agent.py / pilot script).

    Partizione parametri (§6.2 SPEC_generative):
      Fast   ← transition_l0, decoder
      Slow   ← transition_l1, bootstrap_l1, log_gamma_o, log_gamma_s_0, log_gamma_s_1
      Frozen ← (vuoto a livello di questo nn.Module; ψ + W,b esterni)

    Inizializzazione γ (§7.2 SPEC): log γ = 0 ⇒ γ = 1 ⇒ ρ^(l) = 1 ovunque.
    """

    def __init__(
        self,
        alpha: float,
        prior_s0: str = "isotropic",
        K_vampprior: int = K_VAMPPRIOR_DEFAULT,
        k_actions: int = 0,
        action_form: str = "concat",
    ):
        super().__init__()
        # SG-A8 §A8.4-b: K fonte-unica — fluisce da qui a TransitionL0 e (via
        # Agent.init_kwargs) a config/serializzazione/logging. k_actions=0 = macchina firmata.
        self.transition_l0 = TransitionL0(k_actions=k_actions, action_form=action_form)
        self.transition_l1 = TransitionL1(alpha=alpha)
        self.bootstrap_l1 = BootstrapL1()
        self.decoder = Decoder()
        self.log_gamma_o = nn.Parameter(torch.zeros(()))
        # SG-A7 (FIRMATO N+66, SPEC v1.7 §A7.2): γ_s^(0) ANISOTROPO appreso —
        # vettore per-dim ∈ ℝ^{D0}, Γ_s0=diag(exp(log γ_s0,d)). Init zeros ⇒
        # γ_s0,d=1 ∀d ⇒ ρ⁰_d=1 (identico all'isotropico; ELBO diverge da lì).
        # γ_s^(1) resta SCALARE (SG-A7 §A7.2). Statuto slow-param §6.2 invariato.
        self.log_gamma_s_0 = nn.Parameter(torch.zeros(D0))
        self.log_gamma_s_1 = nn.Parameter(torch.zeros(()))
        # SG-A6 BOZZA N+62 — prior arricchito di s⁰ (opt-in).
        # "isotropic" = SPEC v1.5 firmata (N(s0_pred, (1/γ_s0)·I) via transition_l0).
        # "mixture"   = VampPrior K-componenti + floor isotropo (input di controllo §A16.4).
        if prior_s0 not in ("isotropic", "mixture"):
            raise ValueError(f"prior_s0 deve essere 'isotropic' o 'mixture', ricevuto {prior_s0!r}")
        self.prior_s0_mode = prior_s0
        self.prior_s0 = PriorS0Mixture(K=K_vampprior) if prior_s0 == "mixture" else None

    def clamped_log_gammas(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.log_gamma_o.clamp(GAMMA_LOG_MIN, GAMMA_LOG_MAX),
            self.log_gamma_s_0.clamp(GAMMA_LOG_MIN, GAMMA_LOG_MAX),
            self.log_gamma_s_1.clamp(GAMMA_LOG_MIN, GAMMA_LOG_MAX),
        )

    def fast_parameters(self):
        yield from self.transition_l0.parameters()
        yield from self.decoder.parameters()
        if self.prior_s0 is not None:
            yield from self.prior_s0.parameters()

    def slow_parameters(self):
        yield from self.slow_matrices_parameters()
        yield from self.slow_gamma_parameters()

    def slow_matrices_parameters(self):
        """Famiglia Slow «matrici» — §11.2 v0.5 N5b PREREG (η_slow^{matrici} = 1e-5).

        f_θ (TransitionL1) + P_init (BootstrapL1), inclusi LayerNorm Slow post-output
        (§5.5 SPEC_generative block-norm Slow per L1).
        """
        yield from self.transition_l1.parameters()
        yield from self.bootstrap_l1.parameters()

    def slow_gamma_parameters(self):
        """Famiglia Slow «γ» — §11.2 v0.5 N5b PREREG (η_slow^{γ} = 1e-4).

        Log-precisioni γ_o (scalare), γ_s^(0) (vettore per-dim ℝ^{D0}, SG-A7),
        γ_s^(1) (scalare). Stessa famiglia/lr (geometria dello spazio invariata).
        """
        yield self.log_gamma_o
        yield self.log_gamma_s_0
        yield self.log_gamma_s_1
