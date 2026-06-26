"""Agent loop minimo training joint M1+M2 — SPEC_recognition §4.4 ELBO operativo.

Loop online turno-per-turno (no EFE, no azione: M3 separato).
Composizione: SensoryPipeline (Frozen) + GenerativeModel + RecognitionNet.

ELBO §4.4 SPEC_recognition (convenzione: massimizzare L_t ⇒ loss = -L_t).
Costante -(m/2)·log(2π) omessa (irrilevante per gradient).

Stato interno (buffers, detach cross-turn):
  state_s0_prev : (B, D0)  s^(0)_{t-1} per s^(0)_pred
  state_s1_prev : (B, D1)  s^(1)_{τ-1} per s^(0)_pred (lag R10a §1.2) + s^(1)_pred
  s2            : (B, D2)  Slow-state invariante intra-sessione (§5.4 SPEC_generative)
  m_tau         : (B, D_M) modulazione episodica = 0 placeholder §14.1 Q3 (rimpiazzato M6)

Optimizer (§4.5 SPEC_recognition + §6.5 SPEC_generative + §11.2 v0.5 N5b PREREG):
  Fast       → {D_θ, μ^(0)_θ, μ_φ^(0), μ_φ^(1), φ_γ^(0), φ_γ^(1), block-norm Fast}, step ogni turno
  Slow-matrici → {f_θ, P_init, block-norm Slow}, step ogni B_slow turni, lr = 1e-5
  Slow-γ       → {γ_o, γ_s^(0), γ_s^(1)} scalari, step ogni B_slow turni, lr = 1e-4
         (lr differenziato per famiglia §11.2 v0.5 N5b: scalari evolvono ~10× più velocemente
          delle matrici per natura geometrica dello spazio dei parametri — Friston 2017)
  Frozen → {ψ, W, b}, mai (requires_grad=False, esclusi automaticamente da Adam)

Gradient clipping (§4.6 SPEC_recognition + §5.5 SPEC_generative): C_grad = 1.0
sull'union θ_fast ∪ φ (candidato PREREG_M1+M2).
"""
import math

import torch
from torch import nn

from core.generative import D0, D1, D2, D_M, K_VAMPPRIOR_DEFAULT, PRIOR_WARMUP_DEFAULT, GenerativeModel, M
from core.recognition import RecognitionNet
from core.sensory import SensoryPipeline

B_SLOW_DEFAULT = 1000
C_GRAD_DEFAULT = 1.0
LR_SLOW_MATRICES_DEFAULT = 1e-5  # §11.2 v0.5 N5b PREREG
LR_SLOW_GAMMA_DEFAULT = 1e-4     # §11.2 v0.5 N5b PREREG


class Agent(nn.Module):
    def __init__(
        self,
        alpha: float,
        lr_fast: float = 1e-3,
        lr_slow_matrices: float = LR_SLOW_MATRICES_DEFAULT,
        lr_slow_gamma: float = LR_SLOW_GAMMA_DEFAULT,
        lr_slow_gamma_s0: float | None = None,
        b_slow: int = B_SLOW_DEFAULT,
        c_grad: float = C_GRAD_DEFAULT,
        free_bits_lambda: float | None = None,
        beta_kl: float = 1.0,
        prior_s0: str = "isotropic",
        K_vampprior: int = K_VAMPPRIOR_DEFAULT,
        prior_warmup_steps: int = PRIOR_WARMUP_DEFAULT,
        k_actions: int = 0,
        action_form: str = "concat",
    ):
        super().__init__()
        # kwargs di costruzione, per ricostruzione self-describing al load checkpoint
        # (experiment/checkpoints.py). Solo serializzazione: nessun effetto sul modello.
        self.init_kwargs = {
            "alpha": alpha, "lr_fast": lr_fast,
            "lr_slow_matrices": lr_slow_matrices, "lr_slow_gamma": lr_slow_gamma,
            "lr_slow_gamma_s0": lr_slow_gamma_s0,
            "b_slow": b_slow, "c_grad": c_grad,
            "free_bits_lambda": free_bits_lambda, "beta_kl": beta_kl,
            "prior_s0": prior_s0, "K_vampprior": K_vampprior,
            "prior_warmup_steps": prior_warmup_steps,
            "k_actions": k_actions, "action_form": action_form,
        }
        self.sensory = SensoryPipeline()
        self.generative = GenerativeModel(
            alpha=alpha, prior_s0=prior_s0, K_vampprior=K_vampprior,
            k_actions=k_actions, action_form=action_form,
        )
        self.recognition = RecognitionNet()
        self.prior_s0_mode = prior_s0
        self.prior_warmup_steps = int(prior_warmup_steps)

        self.b_slow = b_slow
        self.c_grad = c_grad
        # Free-bits floor su L0 (ADDENDUM SG-A3 §A3.4 + PREREG A12.4): input di controllo,
        # NON peso. None ⇒ KL classico (no-op, preserva diagnostici pre-esistenti).
        self.free_bits_lambda = free_bits_lambda
        # β-VAE rate lever (DIR.1 N+48 «curare collasso via β<1»): INVESTIGAZIONE, NON
        # ancora in SPEC firmata. β=1.0 ⇒ ELBO esatto (no-op). β<1 down-pesa il KL
        # (rate-distortion tempering, Higgins 2017 / Alemi 2018). Formalizzazione in
        # SPEC subordinata a firma PI (interazione ρ/T3 + termine ½logγ).
        self.beta_kl = beta_kl

        fast_params = [
            *self.generative.fast_parameters(),
            *self.recognition.parameters(),
        ]
        slow_matrices_params = list(self.generative.slow_matrices_parameters())
        slow_gamma_params = list(self.generative.slow_gamma_parameters())
        self.opt_fast = torch.optim.Adam(fast_params, lr=lr_fast)
        self.opt_slow_matrices = torch.optim.Adam(slow_matrices_params, lr=lr_slow_matrices)
        # OQ30 (N+72): lr DEDICATO per γ_s^(0) (param-group separato). γ_o e γ_s^(1)
        # restano a lr_slow_gamma (pre-reg §11.2, γ_o≈ottimo ELBO audit N+47). γ_s0
        # resta slow-param appreso da ELBO-gradient (SG-A7 §A7.2 INVARIATO: NON M-step,
        # NON floored ad A). lr_slow_gamma_s0=None ⇒ lr identica ⇒ no-op (back-compat).
        lr_gs0 = lr_slow_gamma if lr_slow_gamma_s0 is None else lr_slow_gamma_s0
        gs0 = self.generative.log_gamma_s_0
        other_gamma = [p for p in slow_gamma_params if p is not gs0]
        self.opt_slow_gamma = torch.optim.Adam([
            {"params": other_gamma, "lr": lr_slow_gamma},
            {"params": [gs0], "lr": lr_gs0},
        ])
        self.lr_slow_gamma_s0_effective = lr_gs0
        self._fast_params = fast_params
        self._slow_matrices_params = slow_matrices_params
        self._slow_gamma_params = slow_gamma_params
        self._slow_params = slow_matrices_params + slow_gamma_params

        self.register_buffer("s2", torch.zeros(1, D2))
        self.register_buffer("m_tau", torch.zeros(1, D_M))
        self.register_buffer("state_s0_prev", torch.zeros(1, D0))
        self.register_buffer("state_s1_prev", torch.zeros(1, D1))
        self.register_buffer("_step_counter", torch.zeros((), dtype=torch.long))

    def reset_session(self, batch_size: int = 1) -> None:
        """Apre nuova sessione: bootstrap s^(1)_τ=0 = P_init(s^(2)); s^(0)_{-1} = 0."""
        device = self.s2.device
        self.s2 = torch.zeros(batch_size, D2, device=device)
        self.m_tau = torch.zeros(batch_size, D_M, device=device)
        with torch.no_grad():
            s1_0 = self.generative.bootstrap_l1(self.s2)
        self.state_s0_prev = torch.zeros(batch_size, D0, device=device)
        self.state_s1_prev = s1_0.detach()
        self.opt_fast.zero_grad()
        self.opt_slow_matrices.zero_grad()
        self.opt_slow_gamma.zero_grad()
        self._step_counter.zero_()

    def _s0_pred_prior(self) -> torch.Tensor:
        """s0_pred al setting INCONDIZIONATO del prior (s²=0, m_τ=0, state_s0_prev=0,
        state_s1_prev=P_init(0)). Costante a θ fisso; ricalcolato ogni step.

        SG-A6 §A6.1: VampPrior valuta componenti φ⁰(u_k, s0_pred_prior, …). Usato come
        riferimento «contesto del prior», non gradient-flow (no_grad) — i parametri
        Fast del prior (u_k, λ_k) e di φ⁰ ricevono gradiente attraverso log_prob.
        """
        with torch.no_grad():
            device = self.s2.device
            z_d0 = torch.zeros(1, D0, device=device)
            z_d2 = torch.zeros(1, D2, device=device)
            s1_init = self.generative.bootstrap_l1(z_d2)
            s0_pred = self.generative.transition_l0(z_d0, s1_init)
        return s0_pred  # (1, D0)

    def _prior_components_cached(
        self,
        log_gamma_o: torch.Tensor,
        log_gamma_s_0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forwarder a PriorS0Mixture.components con s0_pred_prior dal setting incondizionato.

        Ricalcolato ogni step (poco costoso, K×forward φ⁰); cache nominale (no memoization).
        """
        s0_pred_prior = self._s0_pred_prior()
        return self.generative.prior_s0.components(
            self.recognition.posterior_l0, s0_pred_prior,
            log_gamma_o, log_gamma_s_0,
        )

    def _elbo_loss(
        self,
        o: torch.Tensor,
        s0_pred: torch.Tensor,
        s1_pred: torch.Tensor,
        mu0: torch.Tensor,
        log_g0: torch.Tensor,
        s0_sample: torch.Tensor,
        mu1: torch.Tensor,
        log_g1: torch.Tensor,
        log_gamma_o: torch.Tensor,
        log_gamma_s_0: torch.Tensor,
        log_gamma_s_1: torch.Tensor,
    ) -> torch.Tensor:
        """ELBO §4.4 SPEC_recognition con segno invertito (minimizzazione).

        loss = -L_t (costante -(m/2)·log 2π omessa). Media su batch dim.
        Reparametrization §3.4 SPEC: aspettativa E_q ricostruzione approssimata da 1 sample.
        """
        gamma_o = log_gamma_o.exp()
        gamma_s_0 = log_gamma_s_0.exp()
        gamma_s_1 = log_gamma_s_1.exp()
        gamma_phi_0 = log_g0.exp()
        gamma_phi_1 = log_g1.exp()

        o_recon = self.generative.decoder(s0_sample)
        recon_sq = ((o - o_recon) ** 2).sum(dim=-1)
        recon = 0.5 * gamma_o * recon_sq - 0.5 * M * log_gamma_o

        if self.prior_s0_mode == "mixture":
            # SG-A6 BOZZA N+62 — KL MC un-campione su VampPrior + floor isotropo.
            # log q(s⁰|o) gaussiana isotropa: -½D log(2π) + ½D log γ_φ⁰ - ½γ_φ⁰·‖s⁰-μ_φ⁰‖²
            sq_q = ((s0_sample - mu0) ** 2).sum(dim=-1)
            log_q = -0.5 * D0 * math.log(2.0 * math.pi) + 0.5 * D0 * log_g0 - 0.5 * gamma_phi_0 * sq_q
            # log p_θ(s⁰) mistura — componenti calcolate al setting del prior INCONDIZIONATO
            # da o (A16 §A16.2): s²=0, m_τ=0, state_s0_prev=0, state_s1_prev=P_init(0).
            mu_k, log_var_k, log_pi_k = self._prior_components_cached(
                log_gamma_o, log_gamma_s_0
            )
            log_p = self.generative.prior_s0.log_prob(s0_sample, mu_k, log_var_k, log_pi_k)
            kl0 = log_q - log_p
            # free-bits SG-A3 §A3.4 NON applicabile per-dim sulla logsumexp di mistura (§A6.5).
        elif self.free_bits_lambda is None:
            # SG-A7 §A7.3 — KL L0 per-dim (γ_s0 anisotropo ℝ^{D0}; γ_φ0 scalare per-sample).
            # KL = ½ Σ_d [γ_s0,d/γ_φ0 + γ_s0,d(μ0,d−s0_pred,d)² − 1 + log γ_φ0 − log γ_s0,d].
            kl0_gamma_dim = 0.5 * (log_g0.unsqueeze(-1) - log_gamma_s_0
                                   + gamma_s_0 / gamma_phi_0.unsqueeze(-1) - 1.0)
            kl0_mean_dim = 0.5 * gamma_s_0 * (mu0 - s0_pred) ** 2
            kl0 = (kl0_gamma_dim + kl0_mean_dim).sum(dim=-1)
        else:
            # Free-bits per-dim (SG-A3 §A3.4 PRESERVATO, SG-A7 §A7.3): kl0 = Σ_d max(λ, kl0_dim_d).
            # γ_s0,d per-dim (anisotropo); γ_φ^(0) scalare per-sample (recognition §μ-head).
            kl0_gamma_dim = 0.5 * (log_g0.unsqueeze(-1) - log_gamma_s_0
                                   + gamma_s_0 / gamma_phi_0.unsqueeze(-1) - 1.0)
            kl0_mean_dim = 0.5 * gamma_s_0 * (mu0 - s0_pred) ** 2
            kl0_dim = kl0_gamma_dim + kl0_mean_dim
            kl0 = kl0_dim.clamp_min(self.free_bits_lambda).sum(dim=-1)

        kl1_gamma = 0.5 * D1 * (log_g1 - log_gamma_s_1 + gamma_s_1 / gamma_phi_1 - 1.0)
        kl1_mean = 0.5 * gamma_s_1 * ((mu1 - s1_pred) ** 2).sum(dim=-1)
        kl1 = kl1_gamma + kl1_mean

        # SG-A6 §A6.2 — warm-up annealing del KL L0 (solo regime mixture).
        # β_eff_kl0(t) = β_kl · min(1, t/T_warmup); KL L1 INVARIATA.
        if self.prior_s0_mode == "mixture" and self.prior_warmup_steps > 0:
            t = float(self._step_counter.item())
            warmup_factor = min(1.0, t / float(self.prior_warmup_steps))
            beta_kl0 = self.beta_kl * warmup_factor
        else:
            beta_kl0 = self.beta_kl

        return (recon + beta_kl0 * kl0 + self.beta_kl * kl1).mean()

    def step(self, texts: list[str]) -> dict:
        """Un turno di training joint M1+M2.

        Side effect: avanza state_s0_prev/state_s1_prev (detach), step optimizer Fast,
        accumula gradiente Slow; step optimizer Slow ogni `b_slow` turni.
        """
        if len(texts) != self.s2.shape[0]:
            self.reset_session(batch_size=len(texts))

        o = self.sensory(texts)

        s0_pred = self.generative.transition_l0(self.state_s0_prev, self.state_s1_prev)
        s1_pred = self.generative.transition_l1(self.state_s1_prev, self.s2, self.m_tau)

        log_gamma_o, log_gamma_s_0, log_gamma_s_1 = self.generative.clamped_log_gammas()

        mu0, log_g0, s0, mu1, log_g1, s1 = self.recognition(
            o, s0_pred, s1_pred, log_gamma_o, log_gamma_s_0, log_gamma_s_1
        )

        loss = self._elbo_loss(
            o, s0_pred, s1_pred,
            mu0, log_g0, s0, mu1, log_g1,
            log_gamma_o, log_gamma_s_0, log_gamma_s_1,
        )

        loss.backward()
        # §4.6 SPEC_recognition: clip ‖∇_{θ_fast ∪ φ}‖₂ ≤ C_grad
        torch.nn.utils.clip_grad_norm_(self._fast_params, self.c_grad)
        self.opt_fast.step()
        self.opt_fast.zero_grad()

        self._step_counter += 1
        slow_step_now = bool(self._step_counter.item() % self.b_slow == 0)
        if slow_step_now:
            torch.nn.utils.clip_grad_norm_(self._slow_params, self.c_grad)
            self.opt_slow_matrices.step()
            self.opt_slow_gamma.step()
            self.opt_slow_matrices.zero_grad()
            self.opt_slow_gamma.zero_grad()

        with torch.no_grad():
            self.state_s0_prev = s0.detach()
            self.state_s1_prev = s1.detach()

        return {
            "loss": loss.item(),
            "log_gamma_o": log_gamma_o.item(),
            # SG-A7: γ_s^(0) per-dim → riassunto scalare = mean_d (scalare-hint §A7.5.4).
            "log_gamma_s_0": log_gamma_s_0.mean().item(),
            "log_gamma_s_1": log_gamma_s_1.item(),
            "log_gamma_phi_0": log_g0.detach().squeeze(0).cpu(),
            "log_gamma_phi_1": log_g1.detach().squeeze(0).cpu(),
            "slow_step": slow_step_now,
        }
