import asyncio
import os
from dotenv import load_dotenv

from core.orchestrator import CausalFlowOrchestrator
from core.config import GLOBAL_CONFIG

# Mock LLM Client for setup purposes
# (Replace with real OpenAI/Anthropic client initialization)
class MockLLMClient:
    def __init__(self):
        self.model = "gpt-4"

    def generate_prompt(self, prompt: str):
        return "Rank(Ts_Mean(Volume, 5) / Close)"

async def main():
    print("=========================================")
    print(" Booting CausalFlow Quantitative Engine  ")
    print("=========================================")

    # 1. Load Environment Config (API keys for LLM, broker, etc.)
    load_dotenv()

# 2. Config assembly
    config = {
        **GLOBAL_CONFIG.__dict__,
        "target_asset": "BTC-USD",
        "l2_dim": GLOBAL_CONFIG.data.l2_feature_dim,
        "latent_dim": GLOBAL_CONFIG.data.latent_intent_dim,
        "num_concepts": GLOBAL_CONFIG.manifold.num_concepts,
        "mode": "live",
    }

    llm_client = MockLLMClient()

    # 3. Initialize Orchestrator
    try:
        orchestrator = CausalFlowOrchestrator(config=config, llm_client=llm_client)

        # 4. Start Event Loop Pipeline
        await orchestrator.run_live_pipeline()

    except KeyboardInterrupt:
        print("\nPipeline stopped by user. Shutting down gracefully...")
    except Exception as e:
        print(f"\nPipeline crashed with error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
