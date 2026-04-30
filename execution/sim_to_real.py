import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class StalenessExecutionEnv(gym.Env):
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

        if executed_action == 1: # Limit Order
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