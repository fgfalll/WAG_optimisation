"""
Decline Curve Analysis (DCA) module for CO₂ EOR optimization.
Implements various decline curve models for production forecasting.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
from dataclasses import dataclass
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

@dataclass
class DCAResult:
    """Container for Decline Curve Analysis results."""
    time: np.ndarray
    production_rate: np.ndarray
    cumulative_production: np.ndarray
    model_type: str
    parameters: Dict[str, float]
    r_squared: float
    forecast_time: np.ndarray
    forecast_rate: np.ndarray
    forecast_cumulative: np.ndarray
    economic_limit: float
    economic_life: float

class DeclineCurveAnalyzer:
    """
    Performs decline curve analysis using various models:
    - Exponential decline
    - Hyperbolic decline
    - Harmonic decline
    - Modified hyperbolic decline for CO₂ EOR
    """
    
    def __init__(self, economic_limit_factor: float = 0.1):
        """
        Initialize the decline curve analyzer.
        
        Args:
            economic_limit_factor: Fraction of peak rate considered economic limit
        """
        self.economic_limit_factor = economic_limit_factor
    
    def exponential_decline(self, t: np.ndarray, qi: float, di: float) -> np.ndarray:
        """Exponential decline model: q = qi * exp(-di * t)"""
        return qi * np.exp(-di * t)
    
    def hyperbolic_decline(self, t: np.ndarray, qi: float, di: float, b: float) -> np.ndarray:
        """Hyperbolic decline model: q = qi / (1 + b * di * t)^(1/b)"""
        return qi / (1 + b * di * t) ** (1/b)
    
    def harmonic_decline(self, t: np.ndarray, qi: float, di: float) -> np.ndarray:
        """Harmonic decline model: q = qi / (1 + di * t)"""
        return qi / (1 + di * t)
    
    def modified_hyperbolic_decline(self, t: np.ndarray, qi: float, di: float, b: float, d_min: float) -> np.ndarray:
        """
        Modified hyperbolic decline model for CO₂ EOR.
        Transitions to exponential decline at minimum decline rate.
        """
        # Find transition time to exponential decline
        t_transition = (1 / (b * di)) * ((di / d_min) ** b - 1) if b > 0 else float('inf')
        
        # Calculate rates
        rate = np.zeros_like(t)
        hyperbolic_mask = t <= t_transition
        exponential_mask = t > t_transition
        
        if np.any(hyperbolic_mask):
            rate[hyperbolic_mask] = self.hyperbolic_decline(t[hyperbolic_mask], qi, di, b)
        
        if np.any(exponential_mask):
            q_transition = self.hyperbolic_decline(t_transition, qi, di, b)
            rate[exponential_mask] = q_transition * np.exp(-d_min * (t[exponential_mask] - t_transition))
        
        return rate
    
    def fit_exponential(self, time: np.ndarray, rate: np.ndarray) -> Tuple[float, float]:
        """Fit exponential decline model to data."""
        # Linearize: ln(q) = ln(qi) - di * t
        log_rate = np.log(rate[rate > 0])
        valid_time = time[rate > 0]
        
        if len(valid_time) < 2:
            raise ValueError("Insufficient data for exponential decline fit")
        
        # Linear regression
        coeffs = np.polyfit(valid_time, log_rate, 1)
        qi = np.exp(coeffs[1])
        di = -coeffs[0]
        
        return qi, di
    
    def fit_hyperbolic(self, time: np.ndarray, rate: np.ndarray, initial_b: float = 0.5) -> Tuple[float, float, float]:
        """Fit hyperbolic decline model to data."""
        def hyperbolic_func(t, qi, di, b):
            return self.hyperbolic_decline(t, qi, di, b)
        
        # Initial guesses
        qi_guess = rate[0] if rate[0] > 0 else np.max(rate)
        di_guess = 0.1
        b_guess = initial_b
        
        try:
            params, _ = curve_fit(
                hyperbolic_func, time, rate,
                p0=[qi_guess, di_guess, b_guess],
                bounds=([0, 0, 0], [np.inf, np.inf, 2.0]),
                maxfev=10000
            )
            return tuple(params)
        except (RuntimeError, ValueError, TypeError) as e:
            logger.warning(
                "Hyperbolic curve_fit failed on %d data points (max rate=%.1f): %s. Falling back to exponential.",
                len(rate),
                float(np.max(rate)) if len(rate) > 0 else 0.0,
                e,
            )
            qi, di = self.fit_exponential(time, rate)
            return qi, di, 0.0  # b=0 for exponential

    def fit_hyperbolic_qi_di(self, time: np.ndarray, rate: np.ndarray, b: float) -> Tuple[float, float]:
        """Fit hyperbolic decline model for qi and di with a fixed b-factor."""
        def hyperbolic_func_fixed_b(t, qi, di):
            return self.hyperbolic_decline(t, qi, di, b)

        qi_guess = rate[0] if rate[0] > 0 else np.max(rate)
        di_guess = 0.1

        try:
            params, _ = curve_fit(
                hyperbolic_func_fixed_b, time, rate,
                p0=[qi_guess, di_guess],
                bounds=([0, 0], [np.inf, np.inf]),
                maxfev=10000
            )
            return params[0], params[1]
        except (RuntimeError, ValueError, TypeError) as e:
            logger.warning(
                "Hyperbolic curve_fit with fixed b=%.3f failed on %d data points: %s. Falling back to exponential.",
                b,
                len(rate),
                e,
            )
            return self.fit_exponential(time, rate)
    
    def calculate_cumulative(self, time: np.ndarray, rate: np.ndarray) -> np.ndarray:
        """Calculate cumulative production using trapezoidal integration."""
        if len(time) <= 1 or len(rate) <= 1:
            return np.zeros_like(rate, dtype=float)
        return cumulative_trapezoid(rate, time, initial=0.0)
    
    def calculate_r_squared(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate R-squared value for model fit."""
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    def analyze_production(
        self,
        time: np.ndarray,
        production_rate: np.ndarray,
        model_type: str = "auto",
        forecast_years: int = 20,
        time_unit: str = "years",
        b_factor: Optional[float] = None
    ) -> DCAResult:
        """
        Perform decline curve analysis on production data.
        
        Args:
            time: Time array (years, months, or days)
            production_rate: Production rate array
            model_type: Decline model type ("exponential", "hyperbolic", "harmonic", "auto")
            forecast_years: Number of years to forecast
            time_unit: Unit of time ("years", "months", or "days")
            b_factor: Optional pre-determined hyperbolic b-factor to use.
        
        Returns:
            DCAResult object containing analysis results
        """
        # Validate inputs
        if len(time) != len(production_rate):
            raise ValueError("Time and production rate arrays must have the same length")
        
        if len(time) < 3:
            raise ValueError("At least 3 data points required for decline curve analysis")
        
        # Convert time to years if needed
        time_conversion = {"years": 1.0, "months": 1/12, "days": 1/365}
        if time_unit not in time_conversion:
            raise ValueError("time_unit must be 'years', 'months', or 'days'")
        
        time_years = time * time_conversion[time_unit]
        
        if b_factor is not None:
            model_type = "hyperbolic"
        
        # Identify plateau and decline onset
        peak_idx = int(np.argmax(production_rate))
        peak_rate = float(production_rate[peak_idx])

        # Look for onset of sustained boundary-dominated decline after peak (rate drops below 95% peak)
        decline_start_idx = peak_idx
        for idx in range(peak_idx, len(production_rate)):
            if production_rate[idx] < 0.95 * peak_rate:
                decline_start_idx = idx
                break

        # Check if a distinct declining tail exists (at least 3 data points in decline)
        has_decline_segment = (len(production_rate) - decline_start_idx) >= 3 and decline_start_idx > 0

        if has_decline_segment:
            t_onset = float(time_years[decline_start_idx])
            fit_time = time_years[decline_start_idx:] - t_onset
            fit_rate = production_rate[decline_start_idx:]
        else:
            t_onset = float(time_years[0])
            fit_time = time_years - t_onset
            fit_rate = production_rate

        # Determine best model if auto-selection
        if model_type == "auto":
            model_type = self._select_best_model(fit_time, fit_rate)

        # Fit selected model on declining data
        if model_type == "exponential":
            qi, di = self.fit_exponential(fit_time, fit_rate)
            b = 0.0
            decline_func = lambda t: self.exponential_decline(t, qi, di)
        elif model_type == "hyperbolic":
            if b_factor is not None:
                b = b_factor
                qi, di = self.fit_hyperbolic_qi_di(fit_time, fit_rate, b)
            else:
                qi, di, b = self.fit_hyperbolic(fit_time, fit_rate)
            decline_func = lambda t: self.hyperbolic_decline(t, qi, di, b)
        elif model_type == "harmonic":
            qi, di = self.fit_exponential(fit_time, fit_rate)
            b = 1.0
            decline_func = lambda t: self.harmonic_decline(t, qi, di)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Compute predicted rates over historical time
        predicted_rate = np.zeros_like(production_rate)
        for idx, t in enumerate(time_years):
            if has_decline_segment and idx < decline_start_idx:
                predicted_rate[idx] = production_rate[idx]
            else:
                predicted_rate[idx] = float(decline_func(np.array([max(0.0, t - t_onset)]))[0])

        # Calculate R-squared over historical time
        r_squared = self.calculate_r_squared(production_rate, predicted_rate)

        # Calculate cumulative production
        cumulative = self.calculate_cumulative(time_years, production_rate)

        # Generate forecast spanning the full life
        t_max = max(float(forecast_years), float(time_years[-1]))
        forecast_time = np.linspace(0.0, t_max, 100)
        forecast_rate = np.zeros_like(forecast_time)

        for idx, t in enumerate(forecast_time):
            if has_decline_segment and t < t_onset:
                forecast_rate[idx] = float(np.interp(t, time_years[:decline_start_idx+1], production_rate[:decline_start_idx+1]))
            else:
                forecast_rate[idx] = float(decline_func(np.array([max(0.0, t - t_onset)]))[0])

        forecast_cumulative = self.calculate_cumulative(forecast_time, forecast_rate)

        # Calculate economic limit and life
        economic_limit = self.economic_limit_factor * np.max(production_rate)
        economic_life = self._calculate_economic_life(forecast_time, forecast_rate, economic_limit)

        return DCAResult(
            time=time_years,
            production_rate=production_rate,
            cumulative_production=cumulative,
            model_type=model_type,
            parameters={"qi": qi, "di": di, "b": b, "decline_onset_year": t_onset},
            r_squared=r_squared,
            forecast_time=forecast_time,
            forecast_rate=forecast_rate,
            forecast_cumulative=forecast_cumulative,
            economic_limit=economic_limit,
            economic_life=economic_life,
        )
    
    def _select_best_model(self, time: np.ndarray, rate: np.ndarray) -> str:
        """Automatically select the best decline model based on data."""
        try:
            # Try hyperbolic first
            qi_hyper, di_hyper, b_hyper = self.fit_hyperbolic(time, rate)
            pred_hyper = self.hyperbolic_decline(time, qi_hyper, di_hyper, b_hyper)
            r2_hyper = self.calculate_r_squared(rate, pred_hyper)
            
            # Try exponential
            qi_exp, di_exp = self.fit_exponential(time, rate)
            pred_exp = self.exponential_decline(time, qi_exp, di_exp)
            r2_exp = self.calculate_r_squared(rate, pred_exp)
            
            # Select model with better R-squared
            if r2_hyper > r2_exp and b_hyper > 0.1:  # Prefer hyperbolic if meaningful
                return "hyperbolic"
            else:
                return "exponential"
                
        except (RuntimeError, ValueError, TypeError, FloatingPointError) as e:
            logger.warning("Auto model selection failed: %s. Using exponential.", e)
            return "exponential"
    
    def _calculate_economic_life(self, time: np.ndarray, rate: np.ndarray, economic_limit: float) -> float:
        """Calculate economic life based on forecast."""
        # Find when rate drops below economic limit
        below_limit = rate < economic_limit
        if np.any(below_limit):
            first_below = np.where(below_limit)[0][0]
            return time[first_below]
        return time[-1]  # Return last time point if never below limit
    
    def generate_dca_report_data(self, result: DCAResult) -> Dict[str, Union[np.ndarray, float, str]]:
        """Generate structured data for reporting."""
        return {
            "time": result.time,
            "production_rate": result.production_rate,
            "cumulative_production": result.cumulative_production,
            "model_type": result.model_type,
            "parameters": result.parameters,
            "r_squared": result.r_squared,
            "forecast_time": result.forecast_time,
            "forecast_rate": result.forecast_rate,
            "forecast_cumulative": result.forecast_cumulative,
            "economic_limit": result.economic_limit,
            "economic_life": result.economic_life,
            "peak_rate": np.max(result.production_rate),
            "ultimate_recovery": result.forecast_cumulative[-1] if len(result.forecast_cumulative) > 0 else 0
        }

    def plot_decline_curve(self, result: Union[DCAResult, Dict[str, Any]]) -> go.Figure:
        """
        Generate a plot of the decline curve analysis.
        
        Args:
            result: DCAResult object or dictionary of DCA results
            
        Returns:
            Plotly figure
        """
        if result is None:
            return go.Figure()

        if isinstance(result, dict):
            time_arr = result.get("time", np.array([]))
            prod_rate = result.get("production_rate", np.array([]))
            forecast_time = result.get("forecast_time", np.array([]))
            forecast_rate = result.get("forecast_rate", np.array([]))
            economic_limit = result.get("economic_limit", 0.0)
            model_type = result.get("model_type", "DCA")
        else:
            time_arr = getattr(result, "time", np.array([]))
            prod_rate = getattr(result, "production_rate", np.array([]))
            forecast_time = getattr(result, "forecast_time", np.array([]))
            forecast_rate = getattr(result, "forecast_rate", np.array([]))
            economic_limit = getattr(result, "economic_limit", 0.0)
            model_type = getattr(result, "model_type", "DCA")

        fig = go.Figure()
        if len(time_arr) > 0 and len(prod_rate) > 0:
            fig.add_trace(go.Scatter(x=time_arr, y=prod_rate, mode='markers', name='Actual Production'))
        if len(forecast_time) > 0 and len(forecast_rate) > 0:
            fig.add_trace(go.Scatter(x=forecast_time, y=forecast_rate, mode='lines', name='Forecasted Production'))
            fig.add_shape(
                type="line",
                x0=0, y0=economic_limit, x1=forecast_time[-1], y1=economic_limit,
                line=dict(color="Red", width=2, dash="dash"),
                name="Economic Limit"
            )
        fig.update_layout(
            title_text=f"Decline Curve Analysis ({str(model_type).title()})",
            xaxis_title="Time (Years)",
            yaxis_title="Production Rate (STB/year)",
            yaxis_type="log"
        )
        return fig