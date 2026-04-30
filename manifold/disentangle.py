import torch

def rbf_kernel(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Computes the Radial Basis Function (RBF) / Gaussian kernel matrix.
    Formula: $K(x, x') = \exp(-\frac{||x - x'||^2}{2\sigma^2})$
    """
    # x shape: (batch_size, 1) for a single concept dimension
    dist_sq = torch.cdist(x, x, p=2) ** 2
    return torch.exp(-dist_sq / (2 * sigma ** 2))

def compute_hsic(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Computes the Hilbert-Schmidt Independence Criterion between two variables.
    A value of 0 implies strict statistical independence.
    """
    n = x.size(0)

    # 1. Compute kernel matrices
    K = rbf_kernel(x, sigma)
    L = rbf_kernel(y, sigma)

    # 2. Compute the centering matrix H = I - (1/n) * 1 1^T
    H = torch.eye(n, device=x.device) - (1.0 / n) * torch.ones((n, n), device=x.device)

    # 3. Centered kernel matrices: Kc = H K H, Lc = H L H
    Kc = torch.mm(torch.mm(H, K), H)
    Lc = torch.mm(torch.mm(H, L), H)

    # 4. HSIC is the trace of the product of centered kernel matrices scaled by 1/(n-1)^2
    hsic_value = torch.trace(torch.mm(Kc, Lc)) / ((n - 1) ** 2)

    return hsic_value

def concept_independence_loss(concepts: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Iterates through all pairs of concepts in the Concept Bottleneck Model
    and penalizes any statistical dependence between them.

    Args:
        concepts: Tensor of shape (batch_size, num_concepts)
    """
    num_concepts = concepts.size(1)
    total_hsic_loss = 0.0

    # Sum HSIC across all unique pairs of concept dimensions
    for i in range(num_concepts):
        for j in range(i + 1, num_concepts):
            x_i = concepts[:, i].unsqueeze(1)
            x_j = concepts[:, j].unsqueeze(1)
            total_hsic_loss += compute_hsic(x_i, x_j, sigma)

    # Average the loss over the number of pairs
    num_pairs = (num_concepts * (num_concepts - 1)) / 2
    return total_hsic_loss / num_pairs

# Usage in your CBM training loop:
# hsic_penalty = concept_independence_loss(concept_activations)
# total_loss = prediction_mse + (config.manifold.disentanglement_weight * hsic_penalty)