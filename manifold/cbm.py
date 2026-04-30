import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from core.base import Layer2ManifoldMapper
except ImportError:
    Layer2ManifoldMapper = object

from manifold.disentangle import concept_independence_loss

class OrthogonalCBM(nn.Module, Layer2ManifoldMapper if Layer2ManifoldMapper != object else object):
    def __init__(self, input_dim: int, num_concepts: int, output_dim: int = 1):
        """
        Concept Bottleneck Model for Quantitative Finance.
        Forces intermediate representation into human-understandable concepts.
        """
        super(OrthogonalCBM, self).__init__()
        self.num_concepts = num_concepts

        # Feature extractor mapping raw reconstructed signals to latent space
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Mish(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.Mish()
        )

        # Maps latent space to explicit economic concepts
        self.concept_predictor = nn.Linear(64, num_concepts)

        # Maps concepts to final return prediction (must be simple/linear for interpretability)
        self.return_predictor = nn.Linear(num_concepts, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.feature_extractor(x)
        # Concept activations (e.g., mapped between 0 and 1 using Sigmoid, or unbounded)
        concepts = self.concept_predictor(latents)
        predictions = self.return_predictor(concepts)
        return concepts, predictions

    def compute_orthogonal_loss(self, concepts: torch.Tensor) -> torch.Tensor:
        """
        Calculates the orthogonal penalty to disentangle concepts.
        Penalizes off-diagonal elements of the concept correlation matrix.
        """
        # Normalize concepts to mean 0, std 1 along the batch dimension
        concepts_centered = concepts - concepts.mean(dim=0)
        concepts_norm = F.normalize(concepts_centered, p=2, dim=0)

        # Compute the Gram matrix (correlation matrix)
        gram_matrix = torch.matmul(concepts_norm.T, concepts_norm)

        # Identity matrix for target (perfect orthogonality)
        identity = torch.eye(self.num_concepts, device=concepts.device)

        # Frobenius norm of the difference
        ortho_loss = torch.norm(gram_matrix - identity, p='fro') ** 2
        return ortho_loss

    def compute_hsic_loss(self, concepts: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        Calculates the Hilbert-Schmidt Independence Criterion to strictly minimize
        mutual information between concepts (stronger than linear orthogonality).
        """
        return concept_independence_loss(concepts, sigma)

# Example Usage & Loss Calculation
# model = OrthogonalCBM(input_dim=50, num_concepts=5)
# concepts, preds = model(market_data)
# mse_loss = F.mse_loss(preds, targets)
# ortho_loss = model.compute_orthogonal_loss(concepts)
# total_loss = mse_loss + lambda_ortho * ortho_loss