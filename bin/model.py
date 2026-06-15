import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROFILE_FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent"]
UNKNOWN_VALUES = {"", "nan", "none", "unknown", "not_computed"}




class GaussianMDN(nn.Module):
    """
    Mixture Density Network for p(y | x).

    Example:
        x = SBERT embedding, shape [B, input_dim]
        y = x-vector target, shape [B, output_dim]

    Outputs K diagonal-covariance Gaussians.
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 512,
        num_components: int = 5,
        hidden_dims=(1024, 2048, 2048, 1024),
        dropout: float = 0.1,
        min_sigma: float = 1e-4,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.K = num_components
        self.min_sigma = min_sigma

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h

        self.backbone = nn.Sequential(*layers)

        self.pi_head = nn.Linear(prev_dim, self.K)
        self.mu_head = nn.Linear(prev_dim, self.K * output_dim)
        self.sigma_head = nn.Linear(prev_dim, self.K * output_dim)

    def forward(self, x):
        """
        Args:
            x: [B, input_dim]

        Returns:
            pi_logits: [B, K]
            mu:        [B, K, output_dim]
            sigma:     [B, K, output_dim]
        """
        h = self.backbone(x)

        pi_logits = self.pi_head(h)

        mu = self.mu_head(h)
        mu = mu.view(-1, self.K, self.output_dim)

        raw_sigma = self.sigma_head(h)
        raw_sigma = raw_sigma.view(-1, self.K, self.output_dim)

        sigma = F.softplus(raw_sigma) + self.min_sigma

        return pi_logits, mu, sigma

    def negative_log_likelihood(self, x, y):
        """
        Args:
            x: [B, input_dim]
            y: [B, output_dim]

        Returns:
            scalar NLL loss
        """
        pi_logits, mu, sigma = self.forward(x)

        y = y.unsqueeze(1)  # [B, 1, D]

        log_pi = F.log_softmax(pi_logits, dim=-1)  # [B, K]

        # Diagonal Gaussian log-probability
        log_prob = -0.5 * (
            ((y - mu) / sigma) ** 2
            + 2.0 * torch.log(sigma)
            + torch.log(torch.tensor(2.0 * torch.pi, device=y.device))
        )

        log_prob = log_prob.sum(dim=-1)  # [B, K]

        log_mix_prob = torch.logsumexp(log_pi + log_prob, dim=-1)  # [B]

        return -log_mix_prob.mean()

    def mixture_entropy_loss(self, x):
        """
        Optional regularization term to discourage component collapse.

        Add as:
            loss = nll + lambda_entropy * entropy_loss

        Since this returns negative entropy, minimizing it encourages
        higher entropy over mixture weights.
        """
        pi_logits, _, _ = self.forward(x)
        pi = F.softmax(pi_logits, dim=-1)
        log_pi = F.log_softmax(pi_logits, dim=-1)

        entropy = -(pi * log_pi).sum(dim=-1).mean()
        return -entropy

    def variance_reduction_loss(self, x):
        """Encourage shallower component variances by minimizing average log sigma."""
        _, _, sigma = self.forward(x)
        return torch.log(sigma.clamp_min(self.min_sigma)).mean()

    @torch.no_grad()
    def sample(self, x, deterministic: bool = False):
        """
        Sample one y from p(y | x).

        Args:
            x: [B, input_dim]
            deterministic:
                If True, returns the mean of the most likely component.

        Returns:
            y_sample: [B, output_dim]
        """
        pi_logits, mu, sigma = self.forward(x)
        pi = F.softmax(pi_logits, dim=-1)

        if deterministic:
            k = torch.argmax(pi, dim=-1)
        else:
            k = torch.distributions.Categorical(pi).sample()

        batch_idx = torch.arange(x.size(0), device=x.device)

        chosen_mu = mu[batch_idx, k]
        chosen_sigma = sigma[batch_idx, k]

        if deterministic:
            return chosen_mu

        eps = torch.randn_like(chosen_mu)
        return chosen_mu + chosen_sigma * eps

    @torch.no_grad()
    def expected_value(self, x):
        """
        Returns E[y | x].

        Shape:
            [B, output_dim]
        """
        pi_logits, mu, _ = self.forward(x)
        pi = F.softmax(pi_logits, dim=-1)

        return (pi.unsqueeze(-1) * mu).sum(dim=1)
    

class LearnablePi_MDN(GaussianMDN):
    """ Takes from the classic MDN, except pi are the only parameters influcenced by the input. The mu and sigma are learnable parameters that are shared across all inputs. """
    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 512,
        num_components: int = 5,
        hidden_dims=(1024, 2048, 1024),
        dropout: float = 0.1,
        min_sigma: float = 1e-4,
        mu_init: torch.Tensor = None,
        sigma_init: torch.Tensor = None,
    ):
        super().__init__()
        self.pi_threshold = 0.1 / num_components
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.K = num_components
        self.min_sigma = min_sigma

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h

        self.backbone = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)
        self.pi_head = nn.Linear(prev_dim, self.K)
        self.mu = nn.Parameter(torch.randn(self.K, output_dim) if mu_init is None else mu_init)
        self.sigma = nn.Parameter(0.01*torch.ones(self.K, output_dim) if sigma_init is None else sigma_init)

    def forward(self, x):
        """
        Args:
            x: [B, input_dim]

        Returns:
            pi_logits: [B, K]
            mu:        [B, K, output_dim]
            sigma:     [B, K, output_dim]
        """
        h = self.backbone(x)

        pi_logits = self.softmax(self.pi_head(h))
        # pi_logits[pi_logits < self.pi_threshold] = 0
        mu = self.mu.unsqueeze(0).repeat(x.size(0), 1, 1)
        raw_sigma = self.sigma.unsqueeze(0).repeat(x.size(0), 1, 1)

        sigma = F.softplus(raw_sigma) + self.min_sigma

        return pi_logits, mu, sigma
    
class ComposedGMM_MDN(GaussianMDN):
    """ Takes from the classic MDN, except it loads precomputed GMMS. """
    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 512,
        num_components: int = 5,
        hidden_dims=(1024, 2048, 1024),
        dropout: float = 0.1,
        min_sigma: float = 1e-4,
        gmm_init_path: str = None,
        trainable_mu_sigma=False
    ):
        super().__init__()
        mu_init, sigma_init, pi_init, component_profile_keys = self.load_gmm_initializers(gmm_init_path, output_dim, min_sigma)
        effective_num_components = int(mu_init.shape[0])

        self.pi_threshold = 0.1 / effective_num_components
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.K = effective_num_components
        self.min_sigma = min_sigma

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h

        self.backbone = nn.Sequential(*layers)
        self.pi_head = nn.Linear(prev_dim, self.K)
        nn.init.zeros_(self.pi_head.weight)
        with torch.no_grad():
            self.pi_head.bias.copy_(torch.log(torch.clamp(pi_init, min=1e-12)))
        self.mu = nn.Parameter(mu_init, requires_grad=trainable_mu_sigma)
        self.raw_sigma = nn.Parameter(self.inverse_softplus(sigma_init - min_sigma), requires_grad=trainable_mu_sigma)
        self.component_profile_keys = component_profile_keys
        

        if num_components != self.K:
            print(
                f"Initialized ComposedGMM_MDN with {self.K} components loaded from {gmm_init_path} "
                f"(requested num_components={num_components})",
                flush=True,
            )

    @staticmethod
    def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=1e-12)
        return x + torch.log(-torch.expm1(-x))

    @staticmethod
    def normalize_profile_value(value) -> str:
        text = str(value).strip() if value is not None else "unknown"
        return "unknown" if text.lower() in UNKNOWN_VALUES else text

    @classmethod
    def metadata_profile_key(cls, row: dict[str, str]) -> tuple[str, ...]:
        return tuple(cls.normalize_profile_value(row.get(field, "unknown")) for field in PROFILE_FIELDS)

    @staticmethod
    def gmm_files_from_path(gmm_init_path: str | Path | None) -> list[tuple[Path, float, tuple[str, ...] | None]]:
        if gmm_init_path is None:
            raise ValueError("ComposedGMM_MDN requires gmm_init_path")

        path = Path(gmm_init_path)
        if path.is_file():
            return [(path, 1.0, None)]
        if not path.is_dir():
            raise FileNotFoundError(f"Missing GMM init path: {path}")

        metadata_path = path / "metadata.csv"
        if metadata_path.exists():
            files = []
            with metadata_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                if "gmm_path" not in fieldnames:
                    raise ValueError(f"{metadata_path} must contain a gmm_path column")
                for row in reader:
                    gmm_path = Path(row["gmm_path"])
                    if not gmm_path.exists():
                        gmm_path = path / row["gmm_path"]
                    try:
                        profile_weight = float(row.get("num_xvectors", "") or 1.0)
                    except ValueError:
                        profile_weight = 1.0
                    if profile_weight <= 0.0:
                        profile_weight = 1.0
                    files.append((gmm_path, profile_weight, ComposedGMM_MDN.metadata_profile_key(row)))
            if files:
                return files

        profiles_dir = path / "profiles"
        search_dir = profiles_dir if profiles_dir.is_dir() else path
        return [(gmm_path, 1.0, None) for gmm_path in sorted(search_dir.rglob("*.npz"))]

    @classmethod
    def load_gmm_initializers(
        cls,
        gmm_init_path: str | Path | None,
        output_dim: int,
        min_sigma: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[str, ...] | None]]:
        mu_parts = []
        sigma_parts = []
        pi_parts = []
        component_profile_keys: list[tuple[str, ...] | None] = []

        gmm_entries = cls.gmm_files_from_path(gmm_init_path)
        for gmm_path, profile_weight, profile_key in gmm_entries:
            if not gmm_path.exists():
                raise FileNotFoundError(f"Missing precomputed GMM file: {gmm_path}")
            with np.load(gmm_path) as data:
                if "mu" not in data or "sigma" not in data:
                    raise ValueError(f"{gmm_path} must contain mu and sigma arrays")
                mu = np.asarray(data["mu"], dtype=np.float32)
                sigma = np.asarray(data["sigma"], dtype=np.float32)
                pi = np.asarray(data["pi"], dtype=np.float32) if "pi" in data else None

            if mu.ndim == 1:
                mu = mu.reshape(1, -1)
            if sigma.ndim == 1:
                sigma = sigma.reshape(1, -1)
            if pi is None:
                pi = np.full(mu.shape[0], 1.0 / mu.shape[0], dtype=np.float32)
            else:
                pi = pi.reshape(-1).astype(np.float32)
            if mu.shape != sigma.shape:
                raise ValueError(f"{gmm_path} has mismatched mu/sigma shapes: {mu.shape} vs {sigma.shape}")
            if mu.shape[1] != output_dim:
                raise ValueError(f"{gmm_path} has output_dim={mu.shape[1]}, expected {output_dim}")
            if pi.shape[0] != mu.shape[0]:
                raise ValueError(f"{gmm_path} has pi shape {pi.shape}, expected {mu.shape[0]} weights")
            pi_sum = float(pi.sum())
            if pi_sum <= 0.0:
                raise ValueError(f"{gmm_path} has non-positive pi sum: {pi_sum}")

            mu_parts.append(mu)
            sigma_parts.append(np.maximum(sigma, min_sigma))
            pi_parts.append((pi / pi_sum) * float(profile_weight))
            component_profile_keys.extend([profile_key] * mu.shape[0])

        if not mu_parts:
            raise ValueError(f"No .npz GMM files found in {gmm_init_path}")

        mu_init = torch.tensor(np.concatenate(mu_parts, axis=0), dtype=torch.float32)
        sigma_init = torch.tensor(np.concatenate(sigma_parts, axis=0), dtype=torch.float32)
        pi_init = np.concatenate(pi_parts, axis=0)
        pi_init = torch.tensor(pi_init, dtype=torch.float32)
        pi_init = pi_init / pi_init.sum()
        return mu_init, sigma_init, pi_init, component_profile_keys

    def forward(self, x):
        """
        Args:
            x: [B, input_dim]

        Returns:
            pi_logits: [B, K]
            mu:        [B, K, output_dim]
            sigma:     [B, K, output_dim]
        """
        h = self.backbone(x)

        pi_logits = self.pi_head(h)
        mu = self.mu.unsqueeze(0).repeat(x.size(0), 1, 1)
        raw_sigma = self.raw_sigma.unsqueeze(0).repeat(x.size(0), 1, 1)

        sigma = F.softplus(raw_sigma) + self.min_sigma

        return pi_logits, mu, sigma
