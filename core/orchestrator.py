import torch
import numpy as np
import asyncio
from typing import Dict, Any

# Importing our architectural layers (assuming they are built and instantiable)
from data.async_ingestion import AsyncMultimodalEngine
from data.physical_degradation import MicrostructureInverseSolver, MicrostructureOnlineLearner
from manifold.cbm import OrthogonalCBM
from manifold.schrodinger_bridge import SchrodingerFlowMatching
from discovery.llm_symreg import LLMGuidedSymbolicRegression
from execution.sim_to_real import StalenessExecutionEnv
from execution.attribution_engine import AttributionEngine

class CausalFlowOrchestrator:
    """
    The central event loop for Project CausalFlow.
    Wires together Data Denoising -> Concept Mapping -> Alpha Discovery -> Execution -> Attribution.
    """
    def __init__(self, config: Dict[str, Any], llm_client):
        self.config = config
        self.llm_client = llm_client

        # Initialize sub-modules based on config dimensions
        print("Initializing CausalFlow sub-systems...")
        self.denoiser = MicrostructureInverseSolver(
            l2_feature_dim=config['l2_dim'],
            latent_intent_dim=config['latent_dim']
        )
        self.denoiser_learner = MicrostructureOnlineLearner(self.denoiser, lr=1e-4)
        self.cbm = OrthogonalCBM(
            input_dim=config['latent_dim'],
            num_concepts=config['num_concepts']
        )
        self.cbm_optimizer = torch.optim.Adam(self.cbm.parameters(), lr=1e-3)

        # Add the Schrödinger Bridge model to map capital rotation
        self.flow_matching = SchrodingerFlowMatching(state_dim=config['num_concepts'])
        self.flow_optimizer = torch.optim.Adam(self.flow_matching.parameters(), lr=1e-3)
        self.historical_concepts = None
        self.alpha_discoverer = LLMGuidedSymbolicRegression(
            llm_client=self.llm_client,
            initial_features=[f"concept_{i}" for i in range(config['num_concepts'])],
            target_variable="target_return"
        )
        self.attribution = AttributionEngine(llm_client=self.llm_client)

    async def run_live_pipeline(self):
        """
        The main asynchronous pipeline for live or paper trading.
        """
        print("Starting CausalFlow Live Pipeline...")

        # 1. Layer 1: Ingest and Denoise (Physical Degradation Inverse)
        ingestion_engine = AsyncMultimodalEngine(config={"symbols": [self.config.get("target_asset", "BTC-USD")]})
        ingestion_task = asyncio.create_task(ingestion_engine.start_engine())

        # Load real RL Agent if present in config, else fallback to random
        rl_agent = self.config.get("rl_model", None)

        try:
            total_reward = 0

            # Iterate over the live stream of synchronized frames
            async for aligned_frame in ingestion_engine.ingest_multimodal_stream():
                # Extract L2 book - flatten top 10 bids/asks to a flat array for the tensor
                # Assuming top 10 levels, each with price & size -> 40 features
                # Plus padding if config['l2_dim'] is larger
                bids = aligned_frame['l2_book'].get('bids', [])
                asks = aligned_frame['l2_book'].get('asks', [])

                flat_l2 = []
                for b in (bids + [[0,0]]*10)[:10]: flat_l2.extend(b)
                for a in (asks + [[0,0]]*10)[:10]: flat_l2.extend(a)

                # Zero pad if required by the model arch
                if len(flat_l2) < self.config['l2_dim']:
                    flat_l2.extend([0] * (self.config['l2_dim'] - len(flat_l2)))
                flat_l2 = flat_l2[:self.config['l2_dim']]

                raw_tensor = torch.tensor([flat_l2], dtype=torch.float32)

                # 1.a Interleave Online Training (e.g., train the denoiser autoencoder on every new tick)
                # Adds shape parameter (batch_size=1, seq_len=1, l2_dim)
                train_tensor = raw_tensor.unsqueeze(0)
                physics_loss = self.denoiser_learner.train_step(train_tensor)

                # Pass to Level 1 Denoiser for inference
                self.denoiser.eval()
                with torch.no_grad():
                    clean_intent, _ = self.denoiser(train_tensor)
                    # Squeeze back the seq_len for downstream operations
                    clean_intent = clean_intent.squeeze(0)

                # 2. Layer 2: Map to Human-Understandable Concepts
                # 2.a Interleave Online Training of CBM (Simulated continuous learning)
                self.cbm.train()
                self.cbm_optimizer.zero_grad()

                # Mock target return for online training (In reality, calculated using historical prices delayed by execution horizon)
                # target_ret = fetch_historical_future_return(T - horizon)
                mock_target = torch.randn(1, 1) * 0.001

                concepts, return_preds = self.cbm(clean_intent)

                mse_loss = torch.nn.functional.mse_loss(return_preds, mock_target)
                ortho_loss = self.cbm.compute_orthogonal_loss(concepts)
                hsic_loss = self.cbm.compute_hsic_loss(concepts)

                total_cbm_loss = mse_loss + 0.1 * ortho_loss + 0.1 * hsic_loss
                total_cbm_loss.backward()
                self.cbm_optimizer.step()

                # 2.b Train the Schrodinger Bridge Flow Matching (Predict Capital Flow)
                self.flow_matching.train()
                self.flow_optimizer.zero_grad()

                # We need the previous state to train the continuous time flow
                if self.historical_concepts is not None:
                    # x0 = historical concept state, x1 = current concept state
                    flow_loss = self.flow_matching.compute_loss(self.historical_concepts.detach(), concepts.detach())
                    flow_loss.backward()
                    self.flow_optimizer.step()

                self.historical_concepts = concepts.clone()

                # Predict capital trajectory
                self.flow_matching.eval()
                with torch.no_grad():
                    predicted_flow_traj = self.flow_matching.predict_trajectory(concepts, steps=5)
                    future_capital_state = predicted_flow_traj[:, -1, :] # The estimated future concept activations

                # 3. Layer 3: Alpha Discovery (Periodic Re-calibration)
                # In a live setup this runs on a slow-loop. We simulate a periodic trigger here.
                import pandas as pd
                if total_reward == 0:
                    # Convert to dataframe to match the discoverer interface
                    # In real operation, we would use the predicted future capital state
                    # from the Schrodinger flow rather than the raw instantaneous concepts
                    df_concepts = pd.DataFrame(future_capital_state.cpu().detach().numpy())
                    df_concepts["target_return"] = return_preds.cpu().detach().numpy()

                    # Run LLM + GP discovery
                    candidate_alphas = self.alpha_discoverer.discover_alphas(df_concepts)

                    # Run causal validation filter
                    target_returns_series = df_concepts["target_return"]

                    # Accumulate return horizons into an orchestrator-level rolling history so
                    # we actually have enough datapoints to run statsmodels Granger tests
                    if not hasattr(self, '_causal_history'):
                        self._causal_history = {"concepts": [], "returns": []}

                    # Store up to 100 rolling periods
                    self._causal_history["concepts"].append(df_concepts)
                    self._causal_history["returns"].append(target_returns_series.values[0])

                    if len(self._causal_history["returns"]) > 20: # Enough for max_lag=5
                        # We merge history
                        hist_df = pd.concat(self._causal_history["concepts"], axis=0, ignore_index=True)
                        hist_returns = pd.Series(self._causal_history["returns"])

                        active_alphas = self.alpha_discoverer.verify_causality(
                            candidate_alphas, hist_df, hist_returns
                        )
                    else:
                        print("Building up capital transition history before attempting causal filtering...")
                        active_alphas = []

                    if not active_alphas:
                         active_alphas = ["Rank(Ts_Mean(Volume, 5) / Close)"] # Mocking fallback

                current_alpha = active_alphas[0] if active_alphas else "Default_Alpha"

                # 4. Layer 4: Execution
                execution_env = StalenessExecutionEnv(market_data=np.array([flat_l2]), alpha_signals=np.array([1.0])) # mock alpha signal internally
                trade_result = execution_env.execute_and_simulate(active_alphas, {'inventory': 0})

                # 5. Layer 5: Attribution
                trade_telemetry = {
                    "asset": "BTC-USD", "direction": "BUY" if trade_result["action_taken"][0] > 0 else "SELL",
                    "size": trade_result["action_taken"][0],
                    "cbm_concepts": {"Concept_Active": True},
                    "alpha_formula": current_alpha,
                    "causal_p_value": 0.01,
                    "gnn_attention": {},
                    "execution_route": "Simulated RL",
                    "slippage": 1.2, "latency_ms": trade_result["latency_experienced"]
                }

                report = self.attribution.generate_report(trade_telemetry)
                print("Trade Executed:")
                print(report)

                # 4. Layer 4: RL Execution Routing
                market_state_np = raw_tensor.detach().numpy()
                alpha_signals_np = return_preds.detach().numpy()

                # NOTE: For a true live event loop, maintaining a stateful Env is complex
                # because `step()` advances state. We simulate the RL inference here:
                env = StalenessExecutionEnv(
                    market_data=market_state_np,
                    alpha_signals=alpha_signals_np,
                    latency_ticks=self.config['latency_ticks']
                )

                obs, _ = env.reset()

                if rl_agent is not None:
                    action, _states = rl_agent.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()

                _, step_reward, _, _, _ = env.step(action)
                total_reward += step_reward

                print(f"Executed tick. Inst. Reward: {step_reward:.4f} | Concept 0: {concepts[0,0]:.4f}")

                # For the sake of the orchestrator demo, break after 10 updates
                if total_reward != 0 and np.random.rand() > 0.9:
                    break

        except KeyboardInterrupt:
            print("Pipeline interrupted by user.")
        finally:
            ingestion_engine.stop_engine()
            ingestion_task.cancel()

        print(f"Execution loop complete. Net Simulated Reward: {total_reward:.2f}")

        # 5. Layer 4: Close the loop with Generative Attribution
        trade_telemetry = {
            "asset": self.config.get("target_asset", "UNKNOWN"),
            "direction": "NET_LONG" if return_preds.mean().item() > 0 else "NET_SHORT",
            "cbm_concepts": {"Momentum": concepts[:, 0].mean().item(), "Liquidity": concepts[:, 1].mean().item()},
            "alpha_formula": current_alpha,
            "causal_p_value": 0.005, # Mocking the p-value result
            "gnn_attention": {"TSMC_Supply_Chain": 0.8}, # Mock GNN input since GNN is not on critical path
            "execution_route": "RL_Optimized_VWAP",
            "slippage": abs(total_reward), # Simplification for report
            "latency_ms": self.config['latency_ticks'] * 10
        }

        report = self.attribution.generate_report(trade_telemetry)
        print("\n--- GENERATED ATTRIBUTION REPORT ---\n")
        print(report)

# ==========================================
# Example Usage (To be placed at the bottom of the script or in a separate main.py)
# ==========================================
if __name__ == "__main__":
    # Provides mocked execution dependencies to run the full simulated pipeline locally
    main_config = {
        "l2_dim": 50,
        "latent_dim": 10,
        "num_concepts": 2,
        "latency_ticks": 3,
        "target_asset": "BTC-USD"
    }

    class MockLLM:
        def chat(self, prompt):
            return "['Rank', 'Ts_Mean', 'Log_Diff']"

    mock_market_data = np.random.randn(100, 50)  # 100 ticks, 50 L2 features

    orchestrator = CausalFlowOrchestrator(main_config, MockLLM())
    asyncio.run(orchestrator.run_live_pipeline())

    import os
    try:
        from stable_baselines3 import PPO
        # Initialize a dummy PPO model for demonstration of completed integration
        from execution.sim_to_real import StalenessExecutionEnv
        dummy_env = StalenessExecutionEnv(np.zeros((1, 40)), np.zeros((1, 1)), 3)
        dummy_rl_model = PPO("MlpPolicy", dummy_env, verbose=0)
    except ImportError:
        dummy_rl_model = None
        print("stable-baselines3 not installed, running with random execution agent.")

    # Mock configuration
    sys_config = {
        "l2_dim": 40,          # e.g., 10 levels of bid/ask price and sizes
        "latent_dim": 20,      # Cleaned signal space
        "num_concepts": 5,     # Number of economic concepts
        "latency_ticks": 3,    # Sim-to-Real delay
        "target_asset": "BTC-USD",
        "rl_model": dummy_rl_model
    }

    # Real LLM Client setup (e.g., OpenAI)
    class OpenAILLMClient:
        def __init__(self, api_key: str):
            self.api_key = api_key
            try:
                import openai
                openai.api_key = self.api_key
                self.client = openai
            except ImportError:
                self.client = None

        def chat(self, prompt: str) -> str:
            if not self.client or not self.api_key:
                return "Mock LLM Response (OpenAI not installed or absent API key)"

            response = self.client.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message["content"]

    api_key = os.getenv("OPENAI_API_KEY", "")
    llm_client = OpenAILLMClient(api_key=api_key)

    orchestrator = CausalFlowOrchestrator(config=sys_config, llm_client=llm_client)

    # Mock 100 ticks of data (Batch Size 1, Seq Len 100, Features 40)
    mock_data_stream = np.random.randn(1, 100, 40)

    # Run the pipeline
    asyncio.run(orchestrator.run_live_pipeline(mock_data_stream))