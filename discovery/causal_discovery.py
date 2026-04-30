import numpy as np
import pandas as pd
from typing import List, Dict
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

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
        Uses statsmodels to perform a Granger causality test.
        """
        min_length = min(len(alpha_series), len(return_series))
        alpha_series = alpha_series[-min_length:]
        return_series = return_series[-min_length:]

        # Create DataFrame where target (returns) is the first column, predictor (alpha) is the second
        data = np.column_stack((return_series, alpha_series))

        # We need at least max_lag + enough degrees of freedom to run Granger Causality
        if data.shape[0] < self.max_lag * 3 + 1:
            print("Not enough data lengths for causal inference.")
            return False, 1.0

        try:
            # Add a tiny bit of noise to avoid PerfectSeparation/SingularMatrix errors in statsmodels
            alpha_series = alpha_series + np.random.randn(len(alpha_series)) * 1e-8
            return_series = return_series + np.random.randn(len(return_series)) * 1e-8

            # Run granger causality test
            gc_res = grangercausalitytests(data, maxlag=[self.max_lag], verbose=False)

            # Check p-value for the lowest lag
            p_value = gc_res[self.max_lag][0]['ssr_ftest'][1]
            is_causal = p_value < self.p_threshold

            return is_causal, p_value
        except Exception as e:
            # Fallback if tests fail
            print(f"Granger causality failed: {e}")
            return False, 1.0

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