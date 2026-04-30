"""
Project CausalFlow: End-to-end causal alpha discovery and execution architecture.
"""
__version__ = "1.0.0"

# Expose the central orchestrator and config at the root level
from .core.orchestrator import CausalFlowOrchestrator
from .core.config import GLOBAL_CONFIG, CausalFlowConfig

__all__ = ["CausalFlowOrchestrator", "GLOBAL_CONFIG", "CausalFlowConfig"]