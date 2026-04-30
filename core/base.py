from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import torch
import pandas as pd

class Layer1DataEngine(ABC):
    @abstractmethod
    async def ingest_multimodal_stream(self) -> Dict[str, Any]:
        """Asynchronously ingest L2 Order Book, News, and Alt Data."""
        pass

    @abstractmethod
    def reconstruct_signal(self, observed_data: torch.Tensor) -> torch.Tensor:
        """Inverse physical degradation model to find 'true' price intent."""
        pass

class Layer2ManifoldMapper(ABC):
    @abstractmethod
    def forward(self, clean_signals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Maps signals to concepts and predictions.
        Returns: (concept_activations, return_predictions)
        """
        pass

class Layer3AlphaDiscoverer(ABC):
    @abstractmethod
    def discover_alphas(self, concept_space: pd.DataFrame) -> List[str]:
        """Uses LLM + Symbolic Regression to output readable Alpha formulas."""
        pass

    @abstractmethod
    def verify_causality(self, alpha_formulas: List[str], target_returns: pd.Series) -> List[str]:
        """Filters alphas using causal inference (Time-Series Causal Graphs)."""
        pass

class Layer4Execution(ABC):
    @abstractmethod
    def execute_and_simulate(self, alphas: List[str], market_state: Dict) -> Dict[str, Any]:
        """Staleness-Aware Sim-to-Real execution pipeline."""
        pass

    @abstractmethod
    def generate_attribution_report(self, trade_data: Dict[str, Any]) -> str:
        """LLM-generated Markdown report explaining the trade rationale."""
        pass