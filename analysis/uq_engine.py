import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from copy import deepcopy
import sys
from pathlib import Path

try:
    from UQpy.distributions import Normal, Uniform, Lognormal, JointIndependent

    Triangular = None
    from UQpy.sampling import LatinHypercubeSampling
    from UQpy.run_model import RunModel
    from UQpy.surrogates import PolynomialChaosExpansion as PCE
    from UQpy.sensitivity import SobolSensitivity as Sobol

    from core.optimisation_engine import OptimizationEngine
    from core.data_models import EconomicParameters

except ImportError as e:
    logging.critical(
        f"CRITICAL: UQ_Engine: Could not import core project modules or UQpy: {e}. UQ Engine will not be functional."
    )
    OptimizationEngine, EconomicParameters = object, object
    Normal, Uniform, Lognormal, JointIndependent = object, object, object, object
    Triangular = None
    LatinHypercubeSampling, RunModel, PCE, Sobol = object, object, object, object

import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class UQpyModel:
    def __init__(
        self,
        base_engine: "OptimizationEngine",
        param_definitions: List[Dict[str, Any]],
        target_objectives: List[str],
        fixed_eor_params: Dict[str, float],
    ):
        self.base_engine = base_engine
        self.param_definitions = param_definitions
        self.target_objectives = target_objectives
        self.fixed_eor_params = fixed_eor_params
        self.econ_params_template = deepcopy(self.base_engine.economic_params)

    def initialize(self, samples: np.ndarray) -> None:
        self.samples = samples

    def preprocess_single_sample(self, index, sample):
        return np.atleast_2d(sample)

    def execute_single_sample(self, index, sample_to_send):
        return self(sample_to_send[0])

    def postprocess_single_file(self, index, model_output):
        return model_output

    def finalize(self):
        pass

    def __call__(self, sample: np.ndarray) -> List[Any]:
        sampled_params = {
            p_def["path"]: val for p_def, val in zip(self.param_definitions, sample)
        }

        econ_instance = self._get_sampled_econ_instance(sampled_params)
        model_kwargs = self._get_sampled_model_kwargs(sampled_params)

        eor_params = self.fixed_eor_params.copy()
        for p_def in self.param_definitions:
            if p_def["scope"] == "eor":
                eor_params[p_def["internal_name"]] = sampled_params[p_def["path"]]
        if "reservoir.geostatistical_params" in sampled_params:
            geostatistical_params = sampled_params["reservoir.geostatistical_params"]
            if isinstance(geostatistical_params, dict):
                from core.geology.geostatistical_modeling import (
                    create_geostatistical_grid,
                )

                grid = create_geostatistical_grid((50, 50), geostatistical_params)
                self.base_engine.reservoir.geostatistical_grid = grid

        results = self.base_engine.evaluate_for_analysis(
            eor_operational_params_dict=eor_params,
            economic_params_override=econ_instance,
            avg_porosity_override=sampled_params.get("reservoir.avg_porosity"),
            mmp_override=sampled_params.get("fluid.mmp_value"),
            ooip_override=sampled_params.get("reservoir.ooip_stb"),
            recovery_model_init_kwargs_override=model_kwargs,
            target_objectives=self.target_objectives,
            dimensional_tolerance=0.5, # Allow wider range for UQ
        )
        return [results.get(obj, np.nan) for obj in self.target_objectives]

    def _get_sampled_econ_instance(
        self, samples: Dict[str, float]
    ) -> "EconomicParameters":
        instance = deepcopy(self.econ_params_template)
        for p_def in self.param_definitions:
            if p_def["scope"] == "economic":
                setattr(instance, p_def["internal_name"], samples[p_def["path"]])
        return instance

    def _get_sampled_model_kwargs(self, samples: Dict[str, float]) -> Dict[str, Any]:
        kwargs = {}
        for p_def in self.param_definitions:
            if p_def["scope"] == "model":
                kwargs[p_def["internal_name"]] = samples[p_def["path"]]
        return kwargs


class UncertaintyQuantificationEngine:
    def __init__(self, base_engine: "OptimizationEngine", config: Dict[str, Any]):
        from core.optimisation_engine import OptimizationEngine

        if not isinstance(base_engine, OptimizationEngine):
            raise TypeError(
                f"UQ Engine requires a valid OptimizationEngine, received {type(base_engine)}."
            )

        self.base_engine = base_engine
        self.uq_config = config or {}

        config_params = self.uq_config.get("parameters", [])
        if config_params:
            self.param_definitions = config_params
            logger.info("UQ Engine using parameter definitions from configuration.")
        else:
            self.param_definitions = self.base_engine.get_uncertain_parameters()
            self.param_definitions.append(
                {
                    "path": "reservoir.geostatistical_params",
                    "name": "Geostatistical Parameters",
                    "scope": "reservoir",
                    "internal_name": "geostatistical_params",
                    "distribution": "uniform",
                    "params": [0, 1],
                }
            )
            if self.param_definitions:
                logger.info(
                    "UQ Engine using parameter definitions derived from optimization results."
                )
            else:
                logger.warning(
                    "UQ Engine initialized, but no uncertain parameters are defined."
                )
                self._is_enabled = False
                return

        self.results: Optional[Dict[str, Any]] = None
        self._is_enabled = True

        self.distributions = self._create_distributions()
        if not self.distributions:
            logger.error(
                "Failed to create any valid UQpy distributions. UQ Engine is disabled."
            )
            self._is_enabled = False
            return

        self.joint_distribution = JointIndependent(self.distributions)
        logger.info(
            f"UQ Engine initialized with {len(self.distributions)} uncertain parameters using UQpy."
        )

    def _create_distributions(self) -> List:
        dist_objects = []
        for param_def in self.param_definitions:
            dist_type = param_def.get("distribution")
            dist_params = param_def.get("params")
            path = param_def.get("path")

            if not all([dist_type, dist_params, path]):
                logger.warning(
                    f"Skipping parameter due to incomplete definition: {param_def}"
                )
                continue
            try:
                dist = self._get_uqpy_distribution(dist_type, dist_params)
                dist_objects.append(dist)
            except (ValueError, TypeError) as e:
                logger.error(
                    f"Error creating UQpy distribution for '{path}': {e}. Skipping."
                )
        return dist_objects

    def _get_uqpy_distribution(self, dist_name: str, params: List[Any]):
        dist_map = {"normal": Normal, "uniform": Uniform, "lognormal": Lognormal}
        constructor = dist_map.get(dist_name.lower())
        if not constructor:
            raise ValueError(f"Unsupported UQpy distribution type: {dist_name}")
        try:
            if dist_name.lower() == "normal":
                return constructor(loc=params[0], scale=params[1])
            elif dist_name.lower() == "uniform":
                return constructor(loc=params[0], scale=params[1] - params[0])
            elif dist_name.lower() == "lognormal":
                return constructor(s=params[1], loc=0, scale=np.exp(params[0]))
            elif dist_name.lower() == "triangular":
                raise ValueError(
                    "Triangular distribution is not supported in this version of UQpy. Please use Normal, Uniform, or Lognormal distributions instead."
                )
        except IndexError:
            raise ValueError(
                f"Incorrect number of parameters for {dist_name}: {params}"
            )
        except Exception as e:
            raise ValueError(
                f"Error instantiating {dist_name} with params {params}: {e}"
            )

    def run_mc_analysis(
        self,
        num_samples: int,
        fixed_eor_params: Dict[str, float],
        target_objectives: List[str] = ["npv"],
    ) -> Optional[pd.DataFrame]:
        if not self._is_enabled:
            logger.error("UQ Engine is not enabled. Cannot run MC analysis.")
            return None

        logger.info(
            f"Running UQpy MC Analysis with {num_samples} samples for objectives: {target_objectives}."
        )

        sampling = LatinHypercubeSampling(
            distributions=self.distributions, nsamples=num_samples
        )

        model_wrapper = UQpyModel(
            base_engine=self.base_engine,
            param_definitions=self.param_definitions,
            target_objectives=target_objectives,
            fixed_eor_params=fixed_eor_params,
        )

        model_runner = RunModel(samples=sampling.samples, model=model_wrapper)

        param_names = [p["path"] for p in self.param_definitions]
        results_df = pd.DataFrame(model_runner.samples, columns=param_names)
        qoi_df = pd.DataFrame(model_runner.qoi_list, columns=target_objectives)

        final_df = pd.concat([results_df, qoi_df], axis=1)
        logger.info(f"MC Analysis complete. Processed {len(final_df)} samples.")
        self.results = {"type": "mc", "data": final_df}
        return final_df

    def run_pce_analysis(
        self,
        poly_order: int,
        fixed_eor_params: Dict[str, float],
        target_objective: str = "npv",
    ) -> Optional[Dict[str, Any]]:
        if not self._is_enabled:
            logger.error("UQ Engine is not enabled. Cannot run PCE analysis.")
            return None

        logger.info(
            f"Running UQpy PCE Analysis for '{target_objective}' with polynomial order {poly_order}."
        )

        try:
            from UQpy.surrogates.polynomial_chaos.regressions import LinearRegression
            regression = LinearRegression()
        except ImportError:
            logger.warning("LinearRegression not found in UQpy. Using LassoRegression as fallback.")
            from UQpy.surrogates.polynomial_chaos.regressions import LassoRegression
            regression = LassoRegression()

        from UQpy.surrogates.polynomial_chaos.polynomials import TotalDegreeBasis

        model_wrapper = UQpyModel(
            base_engine=self.base_engine,
            param_definitions=self.param_definitions,
            target_objectives=[target_objective],
            fixed_eor_params=fixed_eor_params,
        )

        # PCE fitting in UQpy 4.x requires explicit samples and targets
        # We use LHS for experimental design
        sampling = LatinHypercubeSampling(
            distributions=self.joint_distribution, 
            nsamples=5 * (poly_order + 1)**len(self.distributions) # Heuristic
        )
        
        run_model = RunModel(model=model_wrapper, samples=sampling.samples)
        
        from UQpy.surrogates.polynomial_chaos.polynomials import TotalDegreeBasis

        pce = PCE(
            polynomial_basis=TotalDegreeBasis(
                distributions=self.joint_distribution, 
                max_degree=poly_order
            ),
            regression_method=regression,
        )

        y_targets = np.array(run_model.qoi_list)
        if len(model_wrapper.target_objectives) == 1:
            y_targets = y_targets.flatten()

        pce.fit(x=sampling.samples, y=y_targets)
        logger.info("PCE surrogate model has been successfully fitted.")

        param_names = [p["name"] for p in self.param_definitions]
        
        # In UQpy 4.1.6, we use PceSensitivity for analytical Sobol indices from PCE
        from UQpy.sensitivity import PceSensitivity
        
        try:
            # Manually calculate Sobol indices because PceSensitivity.run() is broken 
            # in UQpy 4.1.6 for single output (it calls get_moments() which fails).
            
            # 1. Total Variance
            # (Assuming orthonormal basis, excluding the 0-th coefficient which is the mean)
            coeffs = pce.coefficients.flatten()
            total_variance = np.sum(coeffs[1:]**2)
            
            # 2. Get basis information to map coefficients to parameters
            # In TotalDegreeBasis, multi_index_set tells us which parameters are in which term
            multi_index = pce.polynomial_basis.multi_index_set
            
            first_order = np.zeros(len(self.distributions))
            total_order = np.zeros(len(self.distributions))
            
            if total_variance > 1e-12:
                for i in range(len(self.distributions)):
                    # First order index: sum of squares of coeffs where ONLY param i is present
                    # index 0 is mean (all zeros in multi_index)
                    # We look for indices where multi_index[j, i] > 0 and others are 0
                    first_idx_mask = (multi_index[:, i] > 0) & (np.sum(multi_index, axis=1) == multi_index[:, i])
                    first_order[i] = np.sum(coeffs[first_idx_mask]**2) / total_variance
                    
                    # Total order index: sum of squares of coeffs where param i is present at all
                    total_idx_mask = (multi_index[:, i] > 0)
                    total_order[i] = np.sum(coeffs[total_idx_mask]**2) / total_variance
            
            # Ensure we get numpy arrays
            first_order = np.atleast_1d(first_order).flatten()
            total_order = np.atleast_1d(total_order).flatten()
            
        except Exception as e:
            logger.error(f"Manual Sobol calculation failed: {e}. Attempting fallback.")
            first_order = np.zeros(len(param_names))
            total_order = np.zeros(len(param_names))
        
        # Manually calculate moments
        try:
            # Use actual empirical mean of evaluated targets to avoid Lasso regularization bias
            mean_val = float(np.mean(y_targets))
            var_val = float(np.var(y_targets))
        except Exception as e:
            logger.warning(f"Manual moment calculation failed: {e}. Using zero.")
            mean_val, var_val = 0.0, 0.0

        pce_results = {
            "target_objective": target_objective,
            "poly_order": poly_order,
            "num_model_evals": len(run_model.qoi_list),
            "mean_pce": [float(mean_val)],
            "variance_pce": [float(var_val)],
            "sobol_main_effects": dict(zip(param_names, first_order)),
            "sobol_total_effects": dict(zip(param_names, total_order)),
        }
        self.results = pce_results
        return pce_results

    def plot_mc_results(
        self, mc_results_df: pd.DataFrame, objective_col: str
    ) -> tuple[Optional[go.Figure], Optional[str]]:
        """
        Generate a Plotly figure showing the results of a Monte Carlo analysis.

        Returns:
            tuple: (figure, error_message)
                - figure: Plotly figure object or None if error
                - error_message: Error message string or None if successful
        """
        if mc_results_df.empty or objective_col not in mc_results_df.columns:
            error_msg = f"Cannot plot MC results: DataFrame is empty or '{objective_col}' is missing."
            logger.warning(error_msg)
            return None, error_msg

        data = mc_results_df[objective_col].dropna()
        if data.empty:
            error_msg = (
                "Cannot plot MC results: No valid data found after dropping NaN values."
            )
            logger.warning(error_msg)
            return None, error_msg

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Histogram (PDF)", "Cumulative Distribution (CDF)"),
        )
        fig.add_trace(
            go.Histogram(x=data, name="PDF", histnorm="probability density"),
            row=1,
            col=1,
        )

        sorted_data = np.sort(data)
        y_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        fig.add_trace(
            go.Scatter(x=sorted_data, y=y_cdf, name="CDF", mode="lines"), row=1, col=2
        )

        title_text = (
            f"Uncertainty Distribution for {objective_col.replace('_', ' ').title()}"
        )
        fig.update_layout(title_text=title_text, showlegend=False)
        fig.update_xaxes(
            title_text=objective_col.replace("_", " ").title(), row=1, col=1
        )
        fig.update_xaxes(
            title_text=objective_col.replace("_", " ").title(), row=1, col=2
        )
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Probability", row=1, col=2)
        return fig, None

    def plot_pce_sobol_indices(
        self, pce_results: Dict[str, Any]
    ) -> tuple[Optional[go.Figure], Optional[str]]:
        """
        Generate a Plotly bar chart visualizing Sobol sensitivity indices.

        Returns:
            tuple: (figure, error_message)
                - figure: Plotly figure object or None if error
                - error_message: Error message string or None if successful
        """
        main_effects = pce_results.get("sobol_main_effects")
        total_effects = pce_results.get("sobol_total_effects")

        if not all([main_effects, total_effects]):
            error_msg = "Cannot plot Sobol indices: Results are missing or incomplete. Ensure PCE analysis was completed successfully."
            logger.warning(error_msg)
            return None, error_msg

        if not main_effects or not total_effects:
            error_msg = "Cannot plot Sobol indices: Empty results returned."
            logger.warning(error_msg)
            return None, error_msg

        param_names = list(main_effects.keys())
        fig = go.Figure(
            data=[
                go.Bar(
                    name="Main Effect (S1)",
                    x=param_names,
                    y=list(main_effects.values()),
                ),
                go.Bar(
                    name="Total Effect (ST)",
                    x=param_names,
                    y=list(total_effects.values()),
                ),
            ]
        )
        fig.update_layout(
            barmode="group",
            title_text=f"Sobol Sensitivity Indices for {pce_results.get('target_objective', 'Unknown').replace('_', ' ').title()}",
            xaxis_title="Uncertain Parameters",
            yaxis_title="Sensitivity Index",
        )
        return fig, None

        data = mc_results_df[objective_col].dropna()
        if data.empty:
            return None

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Histogram (PDF)", "Cumulative Distribution (CDF)"),
        )
        fig.add_trace(
            go.Histogram(x=data, name="PDF", histnorm="probability density"),
            row=1,
            col=1,
        )

        sorted_data = np.sort(data)
        y_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        fig.add_trace(
            go.Scatter(x=sorted_data, y=y_cdf, name="CDF", mode="lines"), row=1, col=2
        )

        title_text = (
            f"Uncertainty Distribution for {objective_col.replace('_', ' ').title()}"
        )
        fig.update_layout(title_text=title_text, showlegend=False)
        fig.update_xaxes(
            title_text=objective_col.replace("_", " ").title(), row=1, col=1
        )
        fig.update_xaxes(
            title_text=objective_col.replace("_", " ").title(), row=1, col=2
        )
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Probability", row=1, col=2)
        return fig

    def plot_pce_sobol_indices(
        self, pce_results: Dict[str, Any]
    ) -> Optional[go.Figure]:
        main_effects = pce_results.get("sobol_main_effects")
        total_effects = pce_results.get("sobol_total_effects")

        if not all([main_effects, total_effects]):
            logger.warning("Cannot plot Sobol indices: Results are missing.")
            return None

        param_names = list(main_effects.keys())
        fig = go.Figure(
            data=[
                go.Bar(
                    name="Main Effect (S1)",
                    x=param_names,
                    y=list(main_effects.values()),
                ),
                go.Bar(
                    name="Total Effect (ST)",
                    x=param_names,
                    y=list(total_effects.values()),
                ),
            ]
        )
        fig.update_layout(
            barmode="group",
            title_text=f"Sobol Sensitivity Indices for {pce_results['target_objective'].replace('_', ' ').title()}",
            xaxis_title="Uncertain Parameters",
            yaxis_title="Sensitivity Index",
        )
        return fig
