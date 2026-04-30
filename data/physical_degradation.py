import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from core.base import Layer1DataEngine
except ImportError:
    pass

class MicrostructureInverseSolver(nn.Module):
    """
    Solves the inverse problem of market microstructure degradation.

    Mathematical Formulation:
    Let $x_t$ be the true, continuous price/liquidity intent.
    Let $y_t$ be the observed discrete Level-2 order book.
    The forward physical degradation is modeled as:
    $$ y_t = \mathcal{H}(x_t) + \eta(x_t) $$
    where $\mathcal{H}$ represents the tick-discretization and matching engine constraints,
    and $\eta$ represents state-dependent adversarial noise (e.g., spoofing).

    This module trains an encoder to approximate the inverse $\mathcal{H}^{-1}$.
    """
    def __init__(self, l2_feature_dim: int, latent_intent_dim: int):
        super().__init__()

        # Inverse mapping: Obversed L2 Book -> True Latent Intent
        self.inverse_mapper = nn.Sequential(
            nn.Linear(l2_feature_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, latent_intent_dim)
        )

        # Forward operator (learned degradation): Latent Intent -> Reconstructed Book
        self.forward_operator = nn.Sequential(
            nn.Linear(latent_intent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, l2_feature_dim)
        )

    def forward(self, y_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            y_obs: Observed L2 order book features (batch_size, seq_len, l2_feature_dim)
        Returns:
            x_true: Reconstructed continuous intent signal
            y_recon: Reconstructed L2 book (used for training the autoencoder loop)
        """
        # 1. Solve the inverse problem to find the clean signal
        x_true = self.inverse_mapper(y_obs)

        # 2. Apply forward physical operator to reconstruct observation
        y_recon = self.forward_operator(x_true)

        return x_true, y_recon

    def compute_physics_loss(self, y_obs: torch.Tensor, y_recon: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss using standard reconstruction + a physical continuity penalty.
        True intent ($x_t$) should not exhibit the high-frequency micro-reversions
        seen in tick-bouncing ($y_t$).
        """
        recon_loss = F.mse_loss(y_recon, y_obs)

        # Penalty: First derivative of true intent (velocity of intent) should be smooth
        # Minimizing the variance of the first difference penalizes high-frequency jitter
        dx = torch.diff(x_true, dim=1)
        smoothness_penalty = torch.var(dx)

        # Penalty: Bid-Ask spread constraints (example custom constraint)
        # If the first two dims of y_obs are Best Bid and Best Ask, x_true price must lay between them.

        return recon_loss + 0.1 * smoothness_penalty

class MicrostructureOnlineLearner:
    """
    Wraps the MicrostructureInverseSolver to provide real-time online learning
    from the streaming L2 order book data.
    """
    def __init__(self, solver: MicrostructureInverseSolver, lr: float = 1e-4):
        self.solver = solver
        self.optimizer = torch.optim.AdamW(self.solver.parameters(), lr=lr)
        self.loss_history = []

    def train_step(self, y_obs: torch.Tensor) -> float:
        """
        Performs a single online gradient descent step on incoming live data.
        """
        self.solver.train()
        self.optimizer.zero_grad()

        # Forward pass to get clean intent and reconstructed observation
        x_true, y_recon = self.solver(y_obs)

        # Compute continuity preserving physics loss
        loss = self.solver.compute_physics_loss(y_obs, y_recon, x_true)

        # Backpropagate and update weights
        loss.backward()

        # Gradient clipping to prevent exploding gradients from erratic ticks
        torch.nn.utils.clip_grad_norm_(self.solver.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        # Keep history short
        if len(self.loss_history) > 1000:
            self.loss_history.pop(0)

        return loss_val