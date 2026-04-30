from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    """Configuration for Layer 1: Market Data and Denoising."""
    symbols: List[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD"])
    l2_depth: int = 10                  # Number of Bid/Ask levels to ingest
    l2_feature_dim: int = 40            # (10 bids + 10 asks) * (price + size)
    latent_intent_dim: int = 20         # Dimension of the denoised signal space
    ingestion_interval_ms: int = 100    # Milliseconds between synchronized snapshots

@dataclass
class ManifoldConfig:
    """Configuration for Layer 2: Concept Mapping and Continuous Flows."""
    num_concepts: int = 5               # E.g., Momentum, Volatility, Liquidity, etc.
    flow_steps: int = 10                # Number of ODE solver steps for Schrodinger Bridge
    disentanglement_weight: float = 0.1 # Lambda weight for HSIC/Orthogonal penalties

@dataclass
class DiscoveryConfig:
    """Configuration for Layer 3: Causal Discovery and Graph Networks."""
    causal_p_threshold: float = 0.01    # Strictness of time-series causal filter
    max_causal_lag: int = 5             # How far back (in ticks/bars) to check for causality
    gnn_hidden_dim: int = 64            # Capacity of the explainable graph

@dataclass
class ExecutionConfig:
    """Configuration for Layer 4: Sim-to-Real RL and Attribution."""
    latency_ticks: int = 3              # Simulated network/queue delay
    initial_inventory: int = 1000       # Shares/Contracts to execute
    rl_action_space: int = 3            # Hold, Limit (Passive), Market (Aggressive)

@dataclass
class CausalFlowConfig:
    """Master configuration object."""
    data: DataConfig = field(default_factory=DataConfig)
    manifold: ManifoldConfig = field(default_factory=ManifoldConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

# Example global instantiation
GLOBAL_CONFIG = CausalFlowConfig()