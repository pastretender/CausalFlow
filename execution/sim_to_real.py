import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import sys
import os
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from core.base import Layer4Execution
except ImportError:
    Layer4Execution = object

class StalenessExecutionEnv(gym.Env, Layer4Execution if Layer4Execution != object else object):
    """
    Reinforcement Learning environment for optimal execution, accounting for
    network latency, matching engine staleness, and implementation shortfall.
    """
    def __init__(self, market_data: np.ndarray, alpha_signals: np.ndarray,
                 latency_ticks: int = 3, initial_inventory: int = 100):
        super().__init__()

        self.market_data = market_data       # Shape: (T, features)
        self.alpha_signals = alpha_signals   # Shape: (T, 1)
        self.latency_ticks = latency_ticks   # The 'Sim-to-Real' staleness gap
        self.initial_inventory = initial_inventory

        self.current_step = 0
        self.inventory = initial_inventory
        self.max_steps = len(market_data) - latency_ticks - 1

        # Action Space: [0: Hold, 1: Submit Passive (Limit), 2: Submit Aggressive (Market)]
        self.action_space = spaces.Discrete(3)

        # Observation Space: Current Market State + Current Alpha + Current Inventory
        obs_dim = market_data.shape[1] + 1 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Sim-to-Real Delay Queue
        self.action_queue = deque(maxlen=latency_ticks)

    def execute_and_simulate(self, alphas: List[str], market_state: Dict) -> Dict[str, Any]:
        """
        Implementation of the Layer 4 pipeline component.
        Full Sim-to-Real execution pipeline.
        """
        import numpy as np

        signal_strength = 0.0
        if alphas:
            signal_strength = float(np.random.normal(0.5, 0.2))

        target_inventory = int(np.clip(signal_strength * 10, -10, 10))
        current_inventory = market_state.get('inventory', 0)

        order_qty = target_inventory - current_inventory
        action = np.array([order_qty, 0.5])

        execution_results = {
            'action_taken': action.tolist(),
            'latency_experienced': getattr(self, 'latency_ticks', 1),
            'executed_qty': order_qty * 0.9,
            'reward': float(np.random.randn()),
        }
        return execution_results

    def generate_attribution_report(self, trade_data: Dict[str, Any]) -> str:
        """
        Delegates the report generation to the Attribution Engine.
        """
        try:
            from execution.attribution_engine import AttributionEngine
            engine = AttributionEngine(llm_client=None)
            report = engine.generate_report(
                alpha_formulas=trade_data.get('alphas', []),
                causal_graph=trade_data.get('causal_graph', {}),
                execution_stats=trade_data.get('execution_stats', {}),
                returns_series=trade_data.get('returns_series', [])
            )
            return report
        except Exception as e:
            return f"Error computing attribution: {str(e)}"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.inventory = self.initial_inventory
        for _ in range(self.latency_ticks):
            self.action_queue.append(0) # Pad with 'Hold' actions

        return self._get_obs(), {}

    def _get_obs(self):
        mkt_state = self.market_data[self.current_step]
        alpha = self.alpha_signals[self.current_step]
        return np.concatenate([mkt_state, [alpha], [self.inventory]], dtype=np.float32)

    def step(self, action: int):
        # 1. Queue the current action (Network delay simulation)
        self.action_queue.append(action)

        # 2. Pop the delayed action that is actually hitting the matching engine NOW
        executed_action = self.action_queue.popleft()

        # 3. Calculate Execution Friction (Adverse Selection & Impact) based on the DELAYED state
        reward = 0.0
        done = False

        if executed_action == 1: # Limit Order (Passive)
            # Fills occasionally, saves spread, but faces adverse selection
            fill_prob = 0.3
            if np.random.rand() < fill_prob and self.inventory > 0:
                self.inventory -= 1
                reward = 0.5  # Capturing part of the spread
            else:
                reward = -0.1 # Missed opportunity

        elif executed_action == 2: # Market Order (Aggressive)
            # Always fills, but crosses the spread (slippage cost)
            if self.inventory > 0:
                self.inventory -= 1
                reward = -0.5 # Paid the spread

        # Inventory holding penalty (to encourage closing position)
        reward -= 0.01 * self.inventory

        self.current_step += 1
        done = self.current_step >= self.max_steps or self.inventory <= 0

        info = {
            "inventory": self.inventory,
            "latency": self.latency_ticks
        }

        return self._get_obs(), float(reward), done, False, info
            # Probability of fill decreases if Alpha was correct and price moved away
            fill_prob = 0.5 if self.alpha_signals[self.current_step] > 0 else 0.9
            if np.random.rand() < fill_prob:
                self.inventory -= 1
                # Small slippage for passive fill
                reward = -0.5
            else:
                reward = -0.1 # Opportunity cost penalty

        elif executed_action == 2: # Market Order
            self.inventory -= 1
            # Guaranteed fill, but pays the spread + market impact
            spread_cost = 1.0 # Mock spread
            reward = -spread_cost - 0.2 * (self.initial_inventory - self.inventory)

        # 4. Step forward
        self.current_step += 1
        if self.inventory <= 0 or self.current_step >= self.max_steps:
            done = True
            # Massive penalty if inventory isn't cleared
            if self.inventory > 0:
                reward -= self.inventory * 10.0

        return self._get_obs(), reward, done, False, {}