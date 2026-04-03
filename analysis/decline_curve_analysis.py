"""
Decline Curve Analysis (DCA) module for CO₂ EOR optimization.
Implements various decline curve models for production forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy.optimize import curve_fit
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
        except Exception as e:
            logger.warning(f"Hyperbolic fit failed: {e}. Falling back to exponential.")
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
        except Exception as e:
            logger.warning(f"Hyperbolic fit with fixed b failed: {e}. Falling back to exponential for qi, di.")
            return self.fit_exponential(time, rate)
    
    def calculate_cumulative(self, time: np.ndarray, rate: np.ndarray) -> np.ndarray:
        """Calculate cumulative production using trapezoidal integration."""
        return np.cumsum(np.trapz(rate, time, axis=0))
    
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
        
        # Determine best model if auto-selection
        if model_type == "auto":
            model_type = self._select_best_model(time_years, production_rate)
        
        # Fit selected model
        if model_type == "exponential":
            qi, di = self.fit_exponential(time_years, production_rate)
            b = 0.0
            predicted_rate = self.exponential_decline(time_years, qi, di)
        
        elif model_type == "hyperbolic":
            if b_factor is not None:
                b = b_factor
                qi, di = self.fit_hyperbolic_qi_di(time_years, production_rate, b)
            else:
                qi, di, b = self.fit_hyperbolic(time_years, production_rate)
            predicted_rate = self.hyperbolic_decline(time_years, qi, di, b)
        
        elif model_type == "harmonic":
            qi, di = self.fit_exponential(time_years, production_rate)  # Initial fit
            b = 1.0
            predicted_rate = self.harmonic_decline(time_years, qi, di)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Calculate R-squared
        r_squared = self.calculate_r_squared(production_rate, predicted_rate)
        
        # Calculate cumulative production
        cumulative = self.calculate_cumulative(time_years, production_rate)
        
        # Generate forecast
        forecast_time = np.linspace(0, forecast_years, 100)
        if model_type == "exponential":
            forecast_rate = self.exponential_decline(forecast_time, qi, di)
        elif model_type == "hyperbolic":
            forecast_rate = self.hyperbolic_decline(forecast_time, qi, di, b)
        elif model_type == "harmonic":
            forecast_rate = self.harmonic_decline(forecast_time, qi, di)
        
        forecast_cumulative = self.calculate_cumulative(forecast_time, forecast_rate)
        
        # Calculate economic limit and life
        economic_limit = self.economic_limit_factor * np.max(production_rate)
        economic_life = self._calculate_economic_life(forecast_time, forecast_rate, economic_limit)
        
        return DCAResult(
            time=time_years,
            production_rate=production_rate,
            cumulative_production=cumulative,
            model_type=model_type,
            parameters={"qi": qi, "di": di, "b": b},
            r_squared=r_squared,
            forecast_time=forecast_time,
            forecast_rate=forecast_rate,
            forecast_cumulative=forecast_cumulative,
            economic_limit=economic_limit,
            economic_life=economic_life
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
                
        except Exception as e:
            logger.warning(f"Auto model selection failed: {e}. Using exponential.")
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

    def plot_decline_curve(self, result: DCAResult) -> go.Figure:
        """
        Generate a plot of the decline curve analysis.
        
        Args:
            result: DCAResult object
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result.time, y=result.production_rate, mode='markers', name='Actual Production'))
        fig.add_trace(go.Scatter(x=result.forecast_time, y=result.forecast_rate, mode='lines', name='Forecasted Production'))
        fig.add_shape(
            type="line",
            x0=0, y0=result.economic_limit, x1=result.forecast_time[-1], y1=result.economic_limit,
            line=dict(color="Red", width=2, dash="dash"),
            name="Economic Limit"
        )
        fig.update_layout(
            title_text=f"Decline Curve Analysis ({result.model_type.title()})",
            xaxis_title="Time (Years)",
            yaxis_title="Production Rate (STB/year)",
            yaxis_type="log"
        )
        return fig