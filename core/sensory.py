"""Pipeline sensoriale Eden_V2 — SPEC_generative §3.1 + §6.4 (ψ + W·e+b Frozen).

ψ encoder pre-trained (sentence-transformer multilingue) + proiezione lineare appresa
una volta al bootstrap, poi entrambi Frozen post-bootstrap (§6.4, §8.5.2).

Ratifiche §14.1 piano calibrazione preliminare 2026-05-17:
  Q2 ψ = sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
        (Reimers & Gurevych 2019; 384-dim output; multilingue → copertura italiana).

Architettura: text → ψ(text) ∈ ℝ^{384} → W·e + b ∈ ℝ^{M=256} → z-score per-dim
(ADDENDUM A1 §A1.2: o = (o_raw−μ_o)⊘σ_o, μ_o,σ_o Frozen stimati una volta sul corpus).

NOTA bootstrap: i pesi (W, b) sono inizializzati con default di nn.Linear (Kaiming
uniform per W, uniform [-bound, bound] per b). L'apprendimento di (W, b) sul corpus di
bootstrap §8.5.2 SPEC_generative vive in script dedicato (N+3 piano: pre_check_corpus
+ bootstrap_projection). Durante il loop training joint M1+M2 (core/agent.py), (W, b)
restano Frozen — gradiente bloccato strutturalmente via requires_grad=False.
"""
import torch
from torch import nn

from core.generative import M

EMBEDDING_DIM = 384
ENCODER_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


STANDARDIZATION_EPS = 1e-6


class LinearProjection(nn.Module):
    """W·e + b — proiezione ℝ^{384} → ℝ^{M=256} Frozen (§3.1 + §6.4 SPEC_generative)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(EMBEDDING_DIM, M, bias=True)
        for p in self.linear.parameters():
            p.requires_grad_(False)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.linear(e)


class StandardizationLayer(nn.Module):
    """o = (o_raw − μ_o) ⊘ σ_o — z-score per-dim Frozen (ADDENDUM A1 §A1.2).

    Terzo stadio della pipeline osservazionale: whitening diagonale che porta ogni
    dimensione di `o_raw = W·e+b` a varianza unitaria → likelihood isotropa
    N(o; D_θ(s⁰), γ_o⁻¹·I) ben specificata (fix mismatch scala obs↔γ_o, OQ22).

    `μ_o, σ_o ∈ ℝ^{M=256}` sono buffer (requires_grad=False, mai nell'optimizer),
    stimati UNA volta sul corpus di bootstrap via `fit(...)` e poi congelati — stesso
    status Frozen di W, b (§A1.2). Default identità (μ=0, σ=1) finché non fittato.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("mu_o", torch.zeros(M))
        self.register_buffer("sigma_o", torch.ones(M))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    def forward(self, o_raw: torch.Tensor) -> torch.Tensor:
        return (o_raw - self.mu_o) / self.sigma_o

    @torch.no_grad()
    def fit(self, o_raw: torch.Tensor) -> None:
        """Stima e CONGELA μ_o, σ_o per-dim su `o_raw` (N, M). σ con floor ε."""
        self.mu_o.copy_(o_raw.mean(dim=0))
        var = o_raw.var(dim=0, unbiased=False)
        self.sigma_o.copy_(torch.sqrt(var + STANDARDIZATION_EPS))
        self.fitted.fill_(True)


class SensoryPipeline(nn.Module):
    """ψ + W·e + b — pipeline sensoriale completa Frozen (§3.1 SPEC_generative).

    Sentence-transformer importato lazy in __init__ (CLAUDE.md «Import opzionali»);
    requires_grad=False su tutti i parametri (§6.4 SPEC_generative «mai aggiornato»).
    """

    def __init__(self):
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence_transformers richiesto per ψ (§14.1 Q2 piano calibrazione). "
                "Aggiungilo a requirements.txt: pip install sentence-transformers"
            ) from exc
        self.encoder = SentenceTransformer(ENCODER_MODEL_ID)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.projection = LinearProjection()
        self.standardization = StandardizationLayer()

    @torch.no_grad()
    def _project(self, texts: list[str]) -> torch.Tensor:
        """ψ + W·e+b → o_raw (pre-standardizzazione). Usato per fit z-score."""
        e = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        e = e.to(next(self.projection.parameters()).device)
        return self.projection(e)

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        return self.standardization(self._project(texts))

    @torch.no_grad()
    def fit_standardization(self, texts: list[str], batch_size: int = 512) -> dict:
        """Stima e CONGELA μ_o, σ_o sul corpus di bootstrap (§A1.2, §8.5 passo 1).

        Calcola `o_raw` per l'intero corpus (pre-standardizzazione) sulla STESSA
        istanza che il training userà → μ_o, σ_o coerenti con i W, b effettivi.
        Da chiamare una volta prima del training; le statistiche restano Frozen.
        """
        chunks = []
        for i in range(0, len(texts), batch_size):
            chunks.append(self._project(texts[i:i + batch_size]).cpu())
        o_raw = torch.cat(chunks, dim=0)
        self.standardization.fit(o_raw.to(self.standardization.mu_o.device))
        std_raw = o_raw.std(dim=0, unbiased=False)
        return {
            "n_corpus": int(o_raw.shape[0]),
            "raw_std_per_dim_mean": float(std_raw.mean().item()),
            "raw_mu_norm": float(o_raw.mean(dim=0).norm().item()),
        }
