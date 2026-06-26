"""Recognition net amortizzata q_φ — SPEC_recognition v1.1 (v1.0 firmato 2026-05-17 + ADDENDUM A1 firmato N+78).

v1.0 (firmata): GELU; heads μ MLP 2 hidden × 256; head γ MLP 1 hidden × 64; block-norm
Fast sulle heads di media. Tutti i parametri Fast (§6.2 SPEC_generative ereditato).

ADDENDUM A1 (§A1.3, FIRMATO N+78) — precision-weighting strutturale (override circoscritto):
le heads per-blocco producono il termine di likelihood `μ_lik^(l)` (ex `μ_φ^(l)` v1.0,
invariato in arch/dim/input/block-norm); la testa scalare `φ_h^(l)` SOSTITUISCE `φ_γ^(l)` e
produce il guadagno di likelihood appreso `h_φ^(l)=exp(clamp(·,-3,3))` (analogo di ‖J‖²/‖K‖²).
Il posterior è IMPOSTO (non più output libero):
    γ_lik^(l) = γ_dat^(l)·h_φ^(l)
    γ_φ^(l)   = γ_lik^(l) + γ_pr^(l)            (precisione del posterior = SOMMA, §7.3.1 isotropo)
    w_pr^(l)  = γ_pr^(l)/γ_φ^(l) = ρ_loc^(l)/(h_φ^(l)+ρ_loc^(l))
    μ_φ^(l)   = (1−w_pr^(l))·μ_lik^(l) + w_pr^(l)·s^(l)_pred
con ρ_loc^(l)=γ_pr^(l)/γ_dat^(l); L0: (γ_dat,γ_pr)=(γ_o,γ_s^(0)); L1: (γ_dat,γ_pr)=(γ_s^(0),γ_s^(1)).
γ_φ NON è clampato direttamente (D2 §A1.11): il clamp vive sul guadagno log h_φ∈[-3,3].

[DERIVAZIONE PROPRIA — riduzione scalare di γ_s^(0)] SG-A7 (§A7.5.4) rende `γ_s^(0)` ANISOTROPO
(per-dim ℝ^{D0}); la SPEC dichiara esplicitamente che la precisione per-dim vive SOLO nel prior
generativo + KL, l'architettura amortizzata §8.4 NON cambia (IN_DIM invariato), e l'API §A1.8
ritorna `log γ_φ^(0)` scalare. Il sweep FAM.2 §4.2 tratta `ρ_loc` come scalare ({0.5,1,2,4,8}).
⟹ la combinazione precision-weighted usa lo SCALAR-HINT `mean_d(log γ_s0,d)` (geom-mean, stessa
convenzione di `_broadcast_scalar`) per γ_pr^(0) (L0) e γ_dat^(1) (L1). Posterior isotropo
scalare ⇒ ELBO §4.4 e reparam §3.4 invariati in forma (consumano (μ_φ, log γ_φ) come prima).
"""
import torch
from torch import nn

from core.generative import (
    D0,
    D1,
    GAMMA_LOG_MAX,
    GAMMA_LOG_MIN,
    HIDDEN_MU,
    L0_BLOCKS,
    L1_BLOCKS,
    M,
)

HIDDEN_GAMMA = 64


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


class _GainHead(nn.Module):
    """φ_h^(l) — guadagno di likelihood appreso (ADDENDUM A1 §A1.3, sostituisce φ_γ^(l)).

    Output: log h_φ^(l) ∈ [-3, 3] ⇒ h_φ^(l)=exp(·) ∈ [e⁻³, e³] (clamp del guadagno, NON di γ_φ).
    Stessa arch/input di φ_γ^(l) v1.0 (1 hidden × 64); cambia solo la semantica dell'output.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_GAMMA),
            nn.GELU(),
            nn.Linear(HIDDEN_GAMMA, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _broadcast_scalar(scalar: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Espande uno scalare a (B, 1) per la concat con tensori batched.

    SG-A7 §A7.5.4: se riceve il vettore per-dim γ_s^(0) (anisotropo, ℝ^{D0}),
    usa lo SCALARE-HINT mean_d(log γ_s0,d). La precisione per-dim vive SOLO nel
    prior generativo + KL (dove (P-loc) e T3 agiscono); IN_DIM resta invariato,
    l'architettura amortizzata §8.4 non cambia.
    """
    s = scalar.mean() if scalar.numel() > 1 else scalar
    return s.view(1, 1).expand(ref.shape[0], 1)


def _scalar_log(log_gamma: torch.Tensor) -> torch.Tensor:
    """Scalar-hint mean_d(log γ_d) (geom-mean in spazio lineare) — convenzione _broadcast_scalar.

    Per γ_s^(0) anisotropo (ℝ^{D0}) ritorna lo scalare usato nella combinazione precision-weighted
    (§A1.3 [DERIVAZIONE PROPRIA]). Scalare già scalare → invariato.
    """
    return log_gamma.mean() if log_gamma.numel() > 1 else log_gamma.reshape(())


def _precision_weighted(
    mu_lik: torch.Tensor,
    log_h: torch.Tensor,
    s_pred: torch.Tensor,
    log_gamma_dat: torch.Tensor,
    log_gamma_pr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Posterior imposto §A1.3 (livello-agnostico; tutte le precisioni scalari).

    Args:
      mu_lik       : (B, d)  termine di likelihood (output heads per-blocco).
      log_h        : (B,)    log h_φ^(l) (clampato a monte).
      s_pred       : (B, d)  media del prior top-down.
      log_gamma_dat: () o (B,)  log γ_dat^(l) (scalare).
      log_gamma_pr : () o (B,)  log γ_pr^(l)  (scalare).
    Ritorna (μ_φ, log γ_φ, detail) con detail = {h_phi, rho_loc, w_pr, gamma_phi} (B,).
    """
    gamma_dat = log_gamma_dat.exp()
    gamma_pr = log_gamma_pr.exp()
    h = log_h.exp()                                  # (B,)
    gamma_lik = gamma_dat * h                         # (B,)
    gamma_phi = gamma_lik + gamma_pr                  # (B,)  — somma, non clampata (D2 §A1.11)
    w_pr = gamma_pr / gamma_phi                       # (B,)  = ρ_loc/(h_φ+ρ_loc)
    w = w_pr.unsqueeze(-1)
    mu_phi = (1.0 - w) * mu_lik + w * s_pred
    detail = {
        "h_phi": h,
        "rho_loc": gamma_pr / gamma_dat * torch.ones_like(h),
        "w_pr": w_pr,
        "gamma_phi": gamma_phi,
        "mu_lik": mu_lik,
    }
    return mu_phi, gamma_phi.log(), detail


class PosteriorL0(nn.Module):
    """φ^(0) — 3 heads μ_lik (world/self/value) + φ_h; posterior precision-weighted §A1.3.

    Input x^(0) := [o_t ; s^(0)_pred ; log γ_o ; log γ_s^(0)] ∈ ℝ^{M+D0+2 = 386}  (§2.2 SPEC).
    Output (μ_φ^(0), log γ_φ^(0)) con γ_φ^(0)=γ_o·h_φ^(0)+γ̄_s^(0) (γ̄ = scalar-hint).
    """

    IN_DIM = M + D0 + 2

    def __init__(self):
        super().__init__()
        d_world, d_self, d_value = L0_BLOCKS
        self.head_world = _MuHead(self.IN_DIM, d_world)
        self.head_self = _MuHead(self.IN_DIM, d_self)
        self.head_value = _MuHead(self.IN_DIM, d_value)
        self.bn_world = nn.LayerNorm(d_world)
        self.bn_self = nn.LayerNorm(d_self)
        self.bn_value = nn.LayerNorm(d_value)
        self.head_h = _GainHead(self.IN_DIM)

    def _compute(
        self,
        o: torch.Tensor,
        s0_pred: torch.Tensor,
        log_gamma_o: torch.Tensor,
        log_gamma_s_0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        log_g_o = _broadcast_scalar(log_gamma_o, o)
        log_g_s = _broadcast_scalar(log_gamma_s_0, o)
        x = torch.cat([o, s0_pred, log_g_o, log_g_s], dim=-1)
        mu_lik = torch.cat(
            [
                self.bn_world(self.head_world(x)),
                self.bn_self(self.head_self(x)),
                self.bn_value(self.head_value(x)),
            ],
            dim=-1,
        )
        log_h = self.head_h(x).clamp(GAMMA_LOG_MIN, GAMMA_LOG_MAX)
        return _precision_weighted(
            mu_lik, log_h, s0_pred,
            log_gamma_dat=_scalar_log(log_gamma_o),
            log_gamma_pr=_scalar_log(log_gamma_s_0),
        )

    def forward(
        self,
        o: torch.Tensor,
        s0_pred: torch.Tensor,
        log_gamma_o: torch.Tensor,
        log_gamma_s_0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu_phi, log_gphi, _ = self._compute(o, s0_pred, log_gamma_o, log_gamma_s_0)
        return mu_phi, log_gphi

    def detail(
        self,
        o: torch.Tensor,
        s0_pred: torch.Tensor,
        log_gamma_o: torch.Tensor,
        log_gamma_s_0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Come forward ma espone (h_φ, ρ_loc, w_pr, γ_φ) per il gate FAM.2 (G-pw1/2/3)."""
        return self._compute(o, s0_pred, log_gamma_o, log_gamma_s_0)


class PosteriorL1(nn.Module):
    """φ^(1) — analogo a φ^(0) ma per L1 (§A1.3).

    Input x^(1) := [s^(0)_t ; s^(1)_pred ; log γ_s^(0) ; log γ_s^(1)] ∈ ℝ^{D0+D1+2 = 322}  (§2.3 SPEC).
    s^(0)_t è il sample reparametrizzato da q_φ^(0) (conditioning bottom-up, §1.2 SPEC).
    γ_o NON entra a L1 (§3.5 SPEC). Precisioni: γ_dat^(1)=γ̄_s^(0) (scalar-hint), γ_pr^(1)=γ_s^(1).
    """

    IN_DIM = D0 + D1 + 2

    def __init__(self):
        super().__init__()
        d_world, d_self, d_value = L1_BLOCKS
        self.head_world = _MuHead(self.IN_DIM, d_world)
        self.head_self = _MuHead(self.IN_DIM, d_self)
        self.head_value = _MuHead(self.IN_DIM, d_value)
        self.bn_world = nn.LayerNorm(d_world)
        self.bn_self = nn.LayerNorm(d_self)
        self.bn_value = nn.LayerNorm(d_value)
        self.head_h = _GainHead(self.IN_DIM)

    def _compute(
        self,
        s0: torch.Tensor,
        s1_pred: torch.Tensor,
        log_gamma_s_0: torch.Tensor,
        log_gamma_s_1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        log_g_s0 = _broadcast_scalar(log_gamma_s_0, s0)
        log_g_s1 = _broadcast_scalar(log_gamma_s_1, s0)
        x = torch.cat([s0, s1_pred, log_g_s0, log_g_s1], dim=-1)
        mu_lik = torch.cat(
            [
                self.bn_world(self.head_world(x)),
                self.bn_self(self.head_self(x)),
                self.bn_value(self.head_value(x)),
            ],
            dim=-1,
        )
        log_h = self.head_h(x).clamp(GAMMA_LOG_MIN, GAMMA_LOG_MAX)
        return _precision_weighted(
            mu_lik, log_h, s1_pred,
            log_gamma_dat=_scalar_log(log_gamma_s_0),
            log_gamma_pr=_scalar_log(log_gamma_s_1),
        )

    def forward(
        self,
        s0: torch.Tensor,
        s1_pred: torch.Tensor,
        log_gamma_s_0: torch.Tensor,
        log_gamma_s_1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu_phi, log_gphi, _ = self._compute(s0, s1_pred, log_gamma_s_0, log_gamma_s_1)
        return mu_phi, log_gphi

    def detail(
        self,
        s0: torch.Tensor,
        s1_pred: torch.Tensor,
        log_gamma_s_0: torch.Tensor,
        log_gamma_s_1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        return self._compute(s0, s1_pred, log_gamma_s_0, log_gamma_s_1)


def reparametrize(mu: torch.Tensor, log_gamma: torch.Tensor) -> torch.Tensor:
    """s = μ + γ^{-½} · ε,  ε ∼ N(0, I_d)  (§3.4 SPEC_recognition, Kingma & Welling 2013).

    log_gamma shape (B,) ; mu shape (B, d). Sigma broadcastato su d.
    """
    sigma = log_gamma.mul(-0.5).exp().unsqueeze(-1)
    eps = torch.randn_like(mu)
    return mu + sigma * eps


class RecognitionNet(nn.Module):
    """Recognition net q_φ — ordine fisso L0 → reparametrize → L1 (§1.2 + §6.2 SPEC).

    forward: training mode (reparametrization trick). Ritorna posterior params + samples per ELBO.
    mode: inference mode deterministico. Ritorna solo posterior params (sample = mode della gaussiana).
    """

    def __init__(self):
        super().__init__()
        self.posterior_l0 = PosteriorL0()
        self.posterior_l1 = PosteriorL1()

    def forward(
        self,
        o: torch.Tensor,
        s0_pred: torch.Tensor,
        s1_pred: torch.Tensor,
        log_gamma_o: torch.Tensor,
        log_gamma_s_0: torch.Tensor,
        log_gamma_s_1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu0, log_g0 = self.posterior_l0(o, s0_pred, log_gamma_o, log_gamma_s_0)
        s0 = reparametrize(mu0, log_g0)
        mu1, log_g1 = self.posterior_l1(s0, s1_pred, log_gamma_s_0, log_gamma_s_1)
        s1 = reparametrize(mu1, log_g1)
        return mu0, log_g0, s0, mu1, log_g1, s1

    def mode(
        self,
        o: torch.Tensor,
        s0_pred: torch.Tensor,
        s1_pred: torch.Tensor,
        log_gamma_o: torch.Tensor,
        log_gamma_s_0: torch.Tensor,
        log_gamma_s_1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu0, log_g0 = self.posterior_l0(o, s0_pred, log_gamma_o, log_gamma_s_0)
        mu1, log_g1 = self.posterior_l1(mu0, s1_pred, log_gamma_s_0, log_gamma_s_1)
        return mu0, log_g0, mu1, log_g1
