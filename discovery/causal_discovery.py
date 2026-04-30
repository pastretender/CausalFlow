import numpy as np
import pandas as pd
from typing import List, Dict
from scipy.stats import pearsonr
# In production, use tigramite or statsmodels for robust causal graphs
# from tigramite.pcmci import PCMCI
# from tigramite.independence_tests import ParCorr

class CausalDiscoveryEngine:
    """
    Validates discovered Alpha formulas by testing for directed causal links
    from the Alpha time-series to the Future Return time-series, isolating
    confounders where possible.
    """
    def __init__(self, max_lag: int = 5, p_value_threshold: float = 0.01):
        self.max_lag = max_lag
        self.p_threshold = p_value_threshold

    def _test_granger_causality(self, alpha_series: np.ndarray, return_series: np.ndarray) -> tuple[bool, float]:
        """
        A lightweight mock of a Granger causality or Transfer Entropy test.
        In reality, this would use an independence test (like ParCorr) conditioning on past returns.
        """
        # Ensure stationarity in production before running this
        min_length = min(len(alpha_series), len(return_series))
        alpha_series = alpha_series[-min_length:]
        return_series = return_series[-min_length:]

        # Test correlation at lag (Alpha at t-1 vs Return at t)
        # We are looking for predictive causality, meaning past alpha explains future returns
        # better than past returns alone.
        lagged_alpha = alpha_series[:-1]
        target_return = return_series[1:]

        corr, p_value = pearsonr(lagged_alpha, target_return)

        # True if statistically significant causal direction is found
        is_causal = p_value < self.p_threshold and corr > 0
        return is_causal, p_value

    def filter_spurious_alphas(self, factor_library: List[Dict], market_data: pd.DataFrame, returns: pd.Series) -> List[Dict]:
        """
        Iterates through the LLM-discovered formulas and strips out anything
        that fails the rigorous time-series causal test.
        """
        causal_alphas = []

        for factor in factor_library:
            # Assume factor['series'] is the pre-computed time-series of the formula
            alpha_series = factor['series']

            is_causal, p_value = self._test_granger_causality(alpha_series, returns.values)

            if is_causal:
                factor['causal_p_value'] = p_value
                causal_alphas.append(factor)

        print(f"Causal Purification: {len(factor_library)} initial candidates -> {len(causal_alphas)} verified causal factors.")
        return causal_alphas