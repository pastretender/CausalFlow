import ast
from typing import List, Dict
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from core.base import Layer3AlphaDiscoverer
except ImportError:
    Layer3AlphaDiscoverer = object

from discovery.causal_discovery import CausalDiscoveryEngine

# Assuming gplearn or a custom genetic programming library is used
# from gplearn.genetic import SymbolicRegressor

class LLMGuidedSymbolicRegression(Layer3AlphaDiscoverer if Layer3AlphaDiscoverer != object else object):
    def __init__(self, llm_client, initial_features: List[str], target_variable: str):
        self.llm = llm_client
        self.features = initial_features
        self.target = target_variable
        self.factor_library = []

    def discover_alphas(self, concept_space: pd.DataFrame) -> List[str]:
        """
        Implementation of the Layer 3 interface.
        Uses LLM + Symbolic Regression to output readable Alpha formulas based on the concept space.
        """
        # We need a target variable internally. Assuming target is last column if not provided
        target_col = self.target if self.target in concept_space.columns else concept_space.columns[-1]

        # Features are everything else
        X = concept_space.drop(columns=[target_col]).values
        y = concept_space[target_col].values

        # Run iterative discovery loop
        factors = self.discovery_loop(X, y, iterations=2)

        return [f['formula'] for f in factors]

    def verify_causality(self, alpha_formulas: List[str], concept_space: pd.DataFrame, target_returns: pd.Series) -> List[str]:
        """
        Implementation of the Layer 3 interface.
        Filters alphas using causal inference (Time-Series Causal Graphs).
        """
        causal_engine = CausalDiscoveryEngine()

        # Build dummy factor library matching the expected input for `filter_spurious_alphas`
        # In a real environment, the formula string would be parsed to generate a time series.
        # Here we mock the series evaluation since the mathematical evaluator is not strictly built.
        factor_library = []
        for formula in alpha_formulas:
            factor_library.append({
                "formula": formula,
                # Simulated evaluation of the formula yielding a time series
                "series": np.random.randn(len(target_returns))
            })

        verified_factors = causal_engine.filter_spurious_alphas(
            factor_library, concept_space, target_returns
        )
        return [f['formula'] for f in verified_factors]

    def _llm_propose_operators(self, performance_feedback: str = "") -> List[str]:
        """Queries the LLM to propose functional forms based on financial theory."""
        prompt = f"""
        Act as a Quant Alpha Researcher. We are trying to predict {self.target}.
        Current features: {self.features}.
        Feedback from previous iteration: {performance_feedback}

        Propose 3 highly logical symbolic operators or combinations (e.g., Rank, Ts_Mean, CrossSectional_ZScore)
        that make economic sense for discovering new alphas. Format as a python list of strings.
        """
        # Mock LLM call
        try:
            response = self.llm.generate_prompt(prompt)
            # Try to safely parse response if it's a list, otherwise fallback
            if isinstance(response, str) and '[' in response:
                return ast.literal_eval(response[response.find('['):response.rfind(']')+1])
            return ["add", "sub", "mul", "div"]
        except Exception:
            return ["add", "sub", "mul", "div"] # fallback

    def _run_genetic_search(self, priors: List[str], X, y) -> List[Dict]:
        """
        Executes constrained symbolic regression using LLM priors.
        """
        print(f"Running GP Search restricted to operators: {priors}")

        try:
            from gplearn.genetic import SymbolicRegressor

            valid_funcs = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min']
            safe_priors = [p.lower() for p in priors if p.lower() in valid_funcs]
            if not safe_priors:
                safe_priors = ['add', 'sub', 'mul', 'div']

            sr = SymbolicRegressor(
                population_size=100,
                generations=5,
                function_set=safe_priors,
                metric='pearson',
                stopping_criteria=0.9,
                p_crossover=0.7,
                p_subtree_mutation=0.1,
                p_hoist_mutation=0.05,
                p_point_mutation=0.1,
                max_samples=0.9,
                verbose=0,
                random_state=42
            )
            sr.fit(X, y)
            best_formula = str(sr._program)

            return [{"formula": best_formula, "fitness": sr._program.raw_fitness_}]
        except ImportError:
            print("gplearn not installed. Using mock GP search result.")
            return [{"formula": "add(X0, div(X1, X2))", "fitness": 0.8}]

    def discovery_loop(self, X, y, iterations=2) -> List[Dict]:
        """
        Main loop interleaving LLM concept proposal and GP optimization.
        """
        best_factors = []
        feedback = ""
        for i in range(iterations):
            priors = self._llm_propose_operators(performance_feedback=feedback)
            factors = self._run_genetic_search(priors, X, y)
            best_factors.extend(factors)

            # Simple feedback generation
            best_fitness = factors[0]['fitness']
            feedback = f"Best formula achieved fitness {best_fitness}. Try non-linear terms."

        return best_factors