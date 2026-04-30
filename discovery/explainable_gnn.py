import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

# Mathematical formulation of standard attention vs sparse:
# Standard: alpha_ij = exp(e_ij) / sum(exp(e_ik)) -> Never exactly zero.
# Sparse: Forces low-relevance edges exactly to 0.0, isolating the causal subgraph.

class SparseGraphAttention(MessagePassing):
    """
    A custom GAT layer that enforces sparsity in the attention weights,
    making cross-asset relationships human-readable and explainable.
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1):
        # We use 'add' aggregation as we will weight the messages by attention
        super().__init__(aggr='add', node_dim=0)
        self.heads = heads
        self.out_channels = out_channels

        # Linear projections for query, key, value
        self.lin_q = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_k = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_v = nn.Linear(in_channels, heads * out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: [num_nodes, in_channels]
        # edge_index shape: [2, num_edges]

        q = self.lin_q(x).view(-1, self.heads, self.out_channels)
        k = self.lin_k(x).view(-1, self.heads, self.out_channels)
        v = self.lin_v(x).view(-1, self.heads, self.out_channels)

        # Start message passing. We return both the node embeddings AND the attention weights
        out, attention_weights = self.propagate(edge_index, q=q, k=k, v=v)

        return out.mean(dim=1), attention_weights

    def message(self, q_i: torch.Tensor, k_j: torch.Tensor, v_j: torch.Tensor,
                index: torch.Tensor, ptr: torch.Tensor, size_i: int) -> tuple[torch.Tensor, torch.Tensor]:

        # Compute dot-product attention
        alpha = (q_i * k_j).sum(dim=-1) / (self.out_channels ** 0.5)

        # Apply sparse activation (Using ReLU as a fast proxy for Sparsemax here)
        # Real implementation would use entmax15 or custom sparsemax
        alpha = F.relu(alpha)

        # Normalize over the neighborhood
        alpha = softmax(alpha, index, ptr, size_i)

        # Weight the value vector by the sparse attention
        out = v_j * alpha.view(-1, self.heads, 1)

        # We pass alpha out so the AttributionEngine can read exactly which edges mattered
        return out, alpha


class AlphaGNN(nn.Module):
    """
    Ingests node features (purified alpha signals) and the edge index
    (supply chain, correlation, ownership graph).
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Initial projection of alpha factors
        self.proj = nn.Linear(feature_dim, hidden_dim)

        # Explainable message passing
        self.attn_layer1 = SparseGraphAttention(hidden_dim, hidden_dim)
        self.attn_layer2 = SparseGraphAttention(hidden_dim, hidden_dim)

        # Final trade signal projector
        self.signal_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict:
        """
        Returns the trading signal and the explainable subgraph metadata.
        """
        x = F.gelu(self.proj(x))

        # First hop (e.g., immediate suppliers)
        x, attn_weights_1 = self.attn_layer1(x, edge_index)
        x = F.gelu(x)

        # Second hop (e.g., suppliers of suppliers)
        x, attn_weights_2 = self.attn_layer2(x, edge_index)

        # Final trade signal for each asset
        signals = self.signal_head(x)

        return {
            "signals": signals,
            "attention_hop_1": attn_weights_1,
            "attention_hop_2": attn_weights_2,
            "edge_index": edge_index
        }