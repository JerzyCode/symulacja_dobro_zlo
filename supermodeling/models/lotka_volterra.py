"""
Lotka-Volterra models for supermodeling experiment.

Baseline: Full predator-prey model with 4 parameters (α, β, γ, δ)
Surrogate: Simplified model with 2 parameters (a, b)
"""

import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Tuple


@dataclass
class LotkaVolterraBaseline:
    """
    Full Lotka-Volterra predator-prey model.
    
    Equations:
        dx/dt = α*x - β*x*y   (prey)
        dy/dt = δ*x*y - γ*y   (predator)
    
    Parameters:
        alpha: prey birth rate
        beta: prey death rate due to predation
        gamma: predator death rate
        delta: predator birth rate from consuming prey
    """
    alpha: float = 1.0
    beta: float = 0.1
    gamma: float = 1.5
    delta: float = 0.075
    
    def derivatives(self, state: np.ndarray, t: float) -> list:
        """Compute derivatives for ODE solver."""
        x, y = state
        dxdt = self.alpha * x - self.beta * x * y
        dydt = self.delta * x * y - self.gamma * y
        return [dxdt, dydt]
    
    def simulate(
        self, 
        t: np.ndarray, 
        initial_state: Tuple[float, float] = (10.0, 5.0)
    ) -> np.ndarray:
        """
        Simulate the model over time.
        
        Args:
            t: Time points
            initial_state: Initial (prey, predator) populations
            
        Returns:
            Array of shape (len(t), 2) with prey and predator populations
        """
        solution = odeint(self.derivatives, initial_state, t)
        return solution
    
    def get_parameters(self) -> dict:
        """Return model parameters as dictionary."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
        }


@dataclass
class LotkaVolterraSurrogate:
    """
    Simplified Lotka-Volterra surrogate model.
    
    Equations (simplified with 2 parameters):
        dx/dt = a*x - b*x*y   (prey)
        dy/dt = b*x*y - a*y   (predator)
    
    This is a simplified version where:
    - Prey birth rate = Predator death rate (parameter a)
    - Prey death rate from predation = Predator birth rate from prey (parameter b)
    
    Parameters:
        a: growth/death rate
        b: interaction rate
    """
    a: float = 1.0
    b: float = 0.1
    
    def derivatives(self, state: np.ndarray, t: float) -> list:
        """Compute derivatives for ODE solver."""
        x, y = state
        dxdt = self.a * x - self.b * x * y
        dydt = self.b * x * y - self.a * y
        return [dxdt, dydt]
    
    def simulate(
        self, 
        t: np.ndarray, 
        initial_state: Tuple[float, float] = (10.0, 5.0)
    ) -> np.ndarray:
        """
        Simulate the model over time.
        
        Args:
            t: Time points
            initial_state: Initial (prey, predator) populations
            
        Returns:
            Array of shape (len(t), 2) with prey and predator populations
        """
        solution = odeint(self.derivatives, initial_state, t)
        return solution
    
    def get_parameters(self) -> dict:
        """Return model parameters as dictionary."""
        return {
            "a": self.a,
            "b": self.b,
        }
    
    @classmethod
    def from_params(cls, params: dict) -> "LotkaVolterraSurrogate":
        """Create model from parameter dictionary."""
        return cls(a=params["a"], b=params["b"])


def compute_mse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Compute Mean Squared Error between observed and simulated data."""
    return np.mean((observed - simulated) ** 2)


def sample_observations(
    model: LotkaVolterraBaseline,
    t_full: np.ndarray,
    n_samples: int,
    initial_state: Tuple[float, float] = (10.0, 5.0),
    noise_std: float = 0.5,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate noisy observations from the baseline model.
    
    Args:
        model: Baseline model to sample from
        t_full: Full time array
        n_samples: Number of observation points to sample
        initial_state: Initial state
        noise_std: Standard deviation of observation noise
        seed: Random seed
        
    Returns:
        Tuple of (sample_times, observations)
    """
    np.random.seed(seed)
    
    # Generate full trajectory
    full_solution = model.simulate(t_full, initial_state)
    
    # Sample time indices (excluding first and last)
    sample_indices = np.sort(
        np.random.choice(
            range(1, len(t_full) - 1), 
            size=min(n_samples, len(t_full) - 2), 
            replace=False
        )
    )
    
    # Get sample times and add noise to observations
    sample_times = t_full[sample_indices]
    observations = full_solution[sample_indices] + np.random.normal(
        0, noise_std, (len(sample_indices), 2)
    )
    
    # Ensure non-negative populations
    observations = np.maximum(observations, 0.1)
    
    return sample_times, observations
