# CausalFlow

**End-to-end causal alpha discovery and execution for quantitative finance.**

CausalFlow is a modular, research-grade Python framework that chains together four architectural layers — from raw market microstructure ingestion to LLM-generated post-trade attribution — using causal inference, deep learning, and reinforcement learning as its core primitives.

The central thesis of the project is that raw market data is a *degraded, noisy observation* of a true latent intent signal. CausalFlow solves the inverse problem, lifts that signal into a human-interpretable concept space, discovers causal alpha formulas inside it, and executes them through a staleness-aware RL environment — closing the loop with an LLM that explains every trade in institutional-grade prose.

---

## Architecture Overview

```
Raw Market Stream (L2, News, Alt Data)
         │
         ▼
┌─────────────────────────────┐
│  LAYER 1 · Data Engine      │  AsyncMultimodalEngine
│  Physical Denoising         │  MicrostructureInverseSolver
└────────────┬────────────────┘
             │  clean latent intent x̂
             ▼
┌─────────────────────────────┐
│  LAYER 2 · Concept Manifold │  OrthogonalCBM  (+ HSIC disentanglement)
│  Capital Flow Modeling      │  SchrodingerFlowMatching
└────────────┬────────────────┘
             │  concept activations
             ▼
┌─────────────────────────────┐
│  LAYER 3 · Alpha Discovery  │  LLMGuidedSymbolicRegression
│  Causal Validation          │  CausalDiscoveryEngine
│  Knowledge Graph            │  AlphaGNN + SparseGraphAttention
└────────────┬────────────────┘
             │  causal alpha signals
             ▼
┌─────────────────────────────┐
│  LAYER 4 · Execution        │  StalenessExecutionEnv (Gym)
│  Post-Trade Attribution     │  AttributionEngine (LLM)
└─────────────────────────────┘
```

The `CausalFlowOrchestrator` wires all four layers into a single `async` event loop suitable for live or paper trading.

---

## Installation

**Prerequisites:** Python ≥ 3.10, CUDA-capable GPU recommended.

```bash
git clone https://github.com/pastretender/CausalFlow.git
cd CausalFlow
pip install -r requirements.txt
```

Key dependencies include PyTorch 2.1, PyTorch Geometric, `tigramite` for time-series causal discovery, `gplearn` for symbolic regression, `dowhy` for formal causal graphs, and `openai` / `langchain` for LLM integration.

---

## Project Structure

```
CausalFlow/
├── core/
│   ├── orchestrator.py        # CausalFlowOrchestrator — the main async event loop
│   ├── config.py              # Dataclass configs for all four layers
│   └── base.py                # Abstract base classes (Layer1–4 interfaces)
│
├── data/
│   ├── async_ingestion.py     # AsyncMultimodalEngine, MemoryInStreamBuffer
│   └── physical_degradation.py# MicrostructureInverseSolver
│
├── manifold/
│   ├── cbm.py                 # OrthogonalCBM
│   ├── schrodinger_bridge.py  # SchrodingerFlowMatching, CapitalFlowVectorField
│   └── disentangle.py         # HSIC independence loss
│
├── discovery/
│   ├── llm_symreg.py          # LLMGuidedSymbolicRegression
│   ├── causal_discovery.py    # CausalDiscoveryEngine
│   └── explainable_gnn.py     # AlphaGNN, SparseGraphAttention
│
├── execution/
│   ├── sim_to_real.py         # StalenessExecutionEnv (Gymnasium)
│   └── attribution_engine.py  # AttributionEngine
│
└── requirements.txt
```

---

## Layer-by-Layer Reference

### Layer 1 · Data Ingestion & Physical Denoising

**`AsyncMultimodalEngine`** (`data/async_ingestion.py`)

Manages concurrent WebSocket streams for L2 order book data, unstructured news/social feeds, and alternative data sources. A `MemoryInStreamBuffer` aligns these heterogeneous, asynchronous streams into coherent 100ms snapshots using an `asyncio` lock. If the downstream PyTorch pipeline is slower than the ingestion rate, frames are dropped rather than queuing unboundedly.

**`MicrostructureInverseSolver`** (`data/physical_degradation.py`)

Models L2 order book observation as a *physical degradation* of the true latent price intent:

$$y_t = \mathcal{H}(x_t) + \eta(x_t)$$

where $\mathcal{H}$ represents tick-discretization and matching-engine constraints, and $\eta$ is state-dependent adversarial noise (e.g., spoofing). The module trains a neural inverse mapper $\mathcal{H}^{-1}$ in an autoencoder loop, with a smoothness penalty on $\frac{dx}{dt}$ to suppress high-frequency micro-reversions that are artifacts of the order book, not genuine intent.

---

### Layer 2 · Concept Manifold

**`OrthogonalCBM`** (`manifold/cbm.py`)

A Concept Bottleneck Model that forces the network's intermediate representation into a small number of named, human-understandable economic concepts (e.g., Momentum, Volatility, Liquidity). The bottleneck means every return prediction is a *linear* function of these concepts, making the model fully auditable. An orthogonal loss penalises the Frobenius distance of the concept Gram matrix from the identity, driving the concepts toward decorrelated, independent axes.

**`SchrodingerFlowMatching`** + **`CapitalFlowVectorField`** (`manifold/schrodinger_bridge.py`)

Models macro capital rotation as a Schrödinger Bridge — an optimal-transport path between the current cross-sectional concept state $x_0$ and a future observed state $x_1$. A neural vector field $v_\theta(x, t)$ is trained with the Flow Matching objective (MSE between predicted and target velocity along the straight ODE path). At inference, an Euler integrator traces the predicted trajectory of capital flows over a configurable number of steps.

**`concept_independence_loss` / `compute_hsic`** (`manifold/disentangle.py`)

Supplements the CBM's orthogonal loss with a kernel-based independence test. The Hilbert-Schmidt Independence Criterion (HSIC) is computed over all unique concept pairs using RBF kernels. Because HSIC is zero *if and only if* the variables are statistically independent (not just uncorrelated), this provides a strictly stronger disentanglement signal than the Gram-matrix penalty alone.

---

### Layer 3 · Alpha Discovery

**`LLMGuidedSymbolicRegression`** (`discovery/llm_symreg.py`)

Implements a three-phase discovery loop that runs for a configurable number of iterations:

1. **Propose** — The LLM is prompted with current features, target variable, and performance feedback from the previous iteration. It proposes a constrained set of financially meaningful operators (e.g., `Rank`, `Ts_Mean`, `CrossSectional_ZScore`).
2. **Search** — Genetic programming (via `gplearn`) searches the operator space *restricted* to the LLM's proposals, dramatically reducing the combinatorial explosion of unconstrained symbolic regression.
3. **Prune** — The LLM reviews discovered candidates and discards formulas that violate microstructure logic or appear to be pure mathematical overfit.

The result is a `factor_library` of symbolic alpha expressions with attached IC and Sharpe metrics.

**`CausalDiscoveryEngine`** (`discovery/causal_discovery.py`)

Acts as a causal filter on the factor library. For each candidate alpha, it tests for directed predictive causality — whether the lagged alpha time-series explains future returns better than chance — using a Granger-style independence test (production deployments should replace the Pearson proxy with `tigramite`'s `ParCorr` or Transfer Entropy). Only alphas that pass the p-value threshold survive to the execution layer.

**`AlphaGNN`** + **`SparseGraphAttention`** (`discovery/explainable_gnn.py`)

Ingests node-level alpha signals and a structured knowledge graph (supply chains, ownership networks, correlated pairs) and propagates information via two-hop sparse attention. Unlike standard softmax attention, sparse activation (via ReLU + softmax, extensible to `entmax`) drives low-relevance edges exactly to zero, producing a human-readable causal subgraph. The `AttributionEngine` reads the returned attention weight tensors to explain *which* cross-asset relationships drove a given trade.

---

### Layer 4 · Execution & Attribution

**`StalenessExecutionEnv`** (`execution/sim_to_real.py`)

A `gymnasium`-compatible RL environment that simulates the *sim-to-real gap* in high-frequency execution. An `action_queue` of configurable depth delays submitted orders by `latency_ticks` steps before they reach the simulated matching engine, faithfully modelling network and queue latency. The three-action space (Hold, Passive Limit, Aggressive Market) carries differentiated cost structures: passive fills have fill-probability dependent on alpha direction, while aggressive fills pay a guaranteed spread plus a nonlinear market-impact term. Unsold inventory at episode end incurs a large terminal penalty, incentivising the agent to clear position before alpha decay.

**`AttributionEngine`** (`execution/attribution_engine.py`)

Aggregates telemetry from all four layers — CBM concept activations, symbolic alpha formula, causal p-value, GNN attention subgraph, RL execution route, and estimated slippage — into a single structured prompt and calls the configured LLM to generate a Markdown post-trade report. The prompt template enforces a rigid `## Trade Summary / ## Factor & Causal Attribution / ## Structural Network Drivers / ## Execution Quality` structure, preventing the LLM from hallucinating data not present in the telemetry.

---

## Configuration

All hyperparameters are declared as Python dataclasses in `core/config.py` with safe defaults. The master config object is `CausalFlowConfig`:

```python
from causalflow import GLOBAL_CONFIG, CausalFlowConfig
from causalflow.core.config import DataConfig, ManifoldConfig, DiscoveryConfig, ExecutionConfig

config = CausalFlowConfig(
    data=DataConfig(
        symbols=["BTC-USD", "ETH-USD"],
        l2_depth=10,              # Bid/Ask levels to ingest
        l2_feature_dim=40,        # (10 bids + 10 asks) × (price + size)
        latent_intent_dim=20,     # Denoised signal dimension
        ingestion_interval_ms=100
    ),
    manifold=ManifoldConfig(
        num_concepts=5,           # e.g., Momentum, Volatility, Liquidity …
        flow_steps=10,            # Euler steps for Schrödinger Bridge ODE
        disentanglement_weight=0.1
    ),
    discovery=DiscoveryConfig(
        causal_p_threshold=0.01,  # Granger causality strictness
        max_causal_lag=5,
        gnn_hidden_dim=64
    ),
    execution=ExecutionConfig(
        latency_ticks=3,          # Sim-to-Real delay
        initial_inventory=1000,
        rl_action_space=3
    )
)
```

---

## Quick Start

```python
import asyncio
import numpy as np
from causalflow import CausalFlowOrchestrator

# 1. Define config (dict form for the orchestrator)
sys_config = {
    "l2_dim": 40,
    "latent_dim": 20,
    "num_concepts": 5,
    "latency_ticks": 3,
    "target_asset": "BTC-USD"
}

# 2. Plug in your LLM client (OpenAI, Anthropic, local HuggingFace, etc.)
class MyLLMClient:
    def chat(self, prompt: str) -> str:
        # return your_llm_api.complete(prompt)
        ...

# 3. Instantiate and run
orchestrator = CausalFlowOrchestrator(config=sys_config, llm_client=MyLLMClient())

# 4. Feed a market data stream — shape (batch, seq_len, features)
mock_stream = np.random.randn(1, 100, 40)
asyncio.run(orchestrator.run_live_pipeline(mock_stream))
```

The pipeline will print CBM concept activations, the RL execution reward, and a full Markdown attribution report to stdout.

---

## Training Individual Modules

Each layer can be trained independently before being wired into the orchestrator.

**Denoiser (Layer 1)**
```python
from causalflow.data.physical_degradation import MicrostructureInverseSolver

denoiser = MicrostructureInverseSolver(l2_feature_dim=40, latent_intent_dim=20)
x_true, y_recon = denoiser(y_observed)
loss = denoiser.compute_physics_loss(y_observed, y_recon, x_true)
loss.backward()
```

**CBM (Layer 2)**
```python
from causalflow.manifold.cbm import OrthogonalCBM
from causalflow.manifold.disentangle import concept_independence_loss

model = OrthogonalCBM(input_dim=20, num_concepts=5)
concepts, preds = model(clean_signals)

mse   = F.mse_loss(preds, targets)
ortho = model.compute_orthogonal_loss(concepts)
hsic  = concept_independence_loss(concepts)
loss  = mse + 0.1 * ortho + 0.1 * hsic
```

**Schrödinger Bridge (Layer 2)**
```python
from causalflow.manifold.schrodinger_bridge import SchrodingerFlowMatching

flow = SchrodingerFlowMatching(state_dim=5)
loss = flow.compute_loss(current_concepts, future_concepts)
loss.backward()

# Inference: predict where capital rotates in 10 steps
trajectory = flow.predict_trajectory(current_concepts, steps=10)
```

**Alpha Discovery Loop (Layer 3)**
```python
from causalflow.discovery.llm_symreg import LLMGuidedSymbolicRegression

symreg = LLMGuidedSymbolicRegression(
    llm_client=my_llm,
    initial_features=["Volume", "Close", "Spread", "OBI"],
    target_variable="FwdReturn_5min"
)
factor_library = symreg.discovery_loop(X_train, y_train, iterations=3)
```

**Causal Filter (Layer 3)**
```python
from causalflow.discovery.causal_discovery import CausalDiscoveryEngine

engine = CausalDiscoveryEngine(max_lag=5, p_value_threshold=0.01)
causal_alphas = engine.filter_spurious_alphas(factor_library, market_df, returns)
```

---

## Design Decisions

**Why treat market data as physical degradation?**  
Limit order books are discrete, adversarially noisy observations of a continuous underlying supply/demand intent. Framing this as an inverse problem (analogous to deconvolution in imaging) lets us apply principled regularization — specifically, a smoothness prior on the first derivative of latent intent — that no standard financial preprocessor encodes.

**Why a Concept Bottleneck Model?**  
Regulatory and risk-management requirements increasingly demand that trade decisions be explainable in human terms. The CBM guarantees by construction that every prediction is a linear combination of interpretable concepts, making the explanation faithful rather than post-hoc.

**Why HSIC over a simple correlation penalty?**  
Pearson correlation only detects linear dependence. HSIC, as a kernel-based statistic, detects arbitrary statistical dependence. Two concept dimensions can be uncorrelated but still share nonlinear information — HSIC catches this, orthogonal loss does not.

**Why LLM-guided symbolic regression over end-to-end neural?**  
Neural return predictors are black boxes. Symbolic formulas are auditable, transferable across assets, and can be tested causally. The LLM acts as a domain-expert prior to keep the search space financially meaningful, while genetic programming handles the combinatorial search.

**Why model latency in the RL environment?**  
In live markets, the action that "hits" the matching engine is not the one you submitted at the current tick — it is the one you submitted `latency_ticks` ago. Models trained without this delay systematically overestimate fill quality and underestimate adverse selection. The `action_queue` makes the sim-to-real gap an explicit, learnable part of the environment.

---

## Extending CausalFlow

Each layer is defined by an abstract base class in `core/base.py`. To swap in a custom implementation, subclass the relevant interface:

```python
from causalflow.core.base import Layer3AlphaDiscoverer

class MyBayesianAlphaDiscoverer(Layer3AlphaDiscoverer):
    def discover_alphas(self, concept_space): ...
    def verify_causality(self, alpha_formulas, target_returns): ...
```

The orchestrator accepts any object that satisfies the interface contract, so layers are independently replaceable without touching the rest of the pipeline.

---

## Roadmap

- Live WebSocket integration with major crypto and equity venues
- `tigramite` PCMCI+ replacing the Pearson-proxy causal filter
- `entmax` / `sparsemax` for exact-zero sparse attention in the GNN
- PPO/SAC agent training script for `StalenessExecutionEnv`
- Portfolio-level multi-asset orchestration with cross-asset GNN signals
- Docker + Ray cluster deployment guide for distributed GP search

---
