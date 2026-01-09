"""
Supermodel implementation with pairwise coupling.

Combines multiple imperfect submodels into a coupled dynamical system
where each model is nudged toward other models via coupling coefficients.
"""

import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass, field
from typing import Tuple, List, Dict

from .lotka_volterra import LotkaVolterraSurrogate


@dataclass
class Supermodel:
    """
    Supermodel combining 3 submodels with pairwise coupling.
    
    For each model k, the coupled dynamics are:
        dx_k/dt = f_x(x_k, y_k) + SUM over j≠k of: C^x_kj * (x_j - x_k)
        dy_k/dt = f_y(x_k, y_k) + SUM over j≠k of: C^y_kj * (y_j - y_k)
    
    Where f_x, f_y are the standalone Lotka-Volterra surrogate dynamics.
    
    Output is the average of all 3 models:
        x_super = (x_1 + x_2 + x_3) / 3
        y_super = (y_1 + y_2 + y_3) / 3
    
    Attributes:
        submodels: List of 3 pretrained LotkaVolterraSurrogate models
        Cx_12, Cx_13, ..., Cx_32: Coupling coefficients for prey (x)
        Cy_12, Cy_13, ..., Cy_32: Coupling coefficients for predator (y)
    """
    submodels: List[LotkaVolterraSurrogate]
    
    # Coupling coefficients for x (prey)
    Cx_12: float = 0.0  # M2 -> M1
    Cx_13: float = 0.0  # M3 -> M1
    Cx_21: float = 0.0  # M1 -> M2
    Cx_23: float = 0.0  # M3 -> M2
    Cx_31: float = 0.0  # M1 -> M3
    Cx_32: float = 0.0  # M2 -> M3
    
    # Coupling coefficients for y (predator)
    Cy_12: float = 0.0  # M2 -> M1
    Cy_13: float = 0.0  # M3 -> M1
    Cy_21: float = 0.0  # M1 -> M2
    Cy_23: float = 0.0  # M3 -> M2
    Cy_31: float = 0.0  # M1 -> M3
    Cy_32: float = 0.0  # M2 -> M3
    
    def __post_init__(self):
        if len(self.submodels) != 3:
            raise ValueError("Supermodel requires exactly 3 submodels")
    
    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Compute coupled derivatives for all 3 models.
        
        State layout: [x1, y1, x2, y2, x3, y3]
        """
        x1, y1, x2, y2, x3, y3 = state
        
        # Get submodel parameters
        a1, b1 = self.submodels[0].a, self.submodels[0].b
        a2, b2 = self.submodels[1].a, self.submodels[1].b
        a3, b3 = self.submodels[2].a, self.submodels[2].b
        
        # Standalone dynamics for each model
        # f_x(x, y) = a*x - b*x*y
        # f_y(x, y) = b*x*y - a*y
        fx1 = a1 * x1 - b1 * x1 * y1
        fy1 = b1 * x1 * y1 - a1 * y1
        
        fx2 = a2 * x2 - b2 * x2 * y2
        fy2 = b2 * x2 * y2 - a2 * y2
        
        fx3 = a3 * x3 - b3 * x3 * y3
        fy3 = b3 * x3 * y3 - a3 * y3
        
        # Coupled dynamics: dx_k/dt = f_x + sum of C^x_kj * (x_j - x_k)
        
        # Model 1
        dx1 = fx1 + self.Cx_12 * (x2 - x1) + self.Cx_13 * (x3 - x1)
        dy1 = fy1 + self.Cy_12 * (y2 - y1) + self.Cy_13 * (y3 - y1)
        
        # Model 2
        dx2 = fx2 + self.Cx_21 * (x1 - x2) + self.Cx_23 * (x3 - x2)
        dy2 = fy2 + self.Cy_21 * (y1 - y2) + self.Cy_23 * (y3 - y2)
        
        # Model 3
        dx3 = fx3 + self.Cx_31 * (x1 - x3) + self.Cx_32 * (x2 - x3)
        dy3 = fy3 + self.Cy_31 * (y1 - y3) + self.Cy_32 * (y2 - y3)
        
        return np.array([dx1, dy1, dx2, dy2, dx3, dy3])
    
    def simulate(
        self, 
        t: np.ndarray, 
        initial_state: Tuple[float, float] = (10.0, 5.0)
    ) -> np.ndarray:
        """
        Simulate the coupled supermodel.
        
        All 3 models start from the same initial state.
        
        Args:
            t: Time points
            initial_state: Initial (prey, predator) populations
            
        Returns:
            Array of shape (len(t), 2) with average prey and predator
        """
        x0, y0 = initial_state
        
        # All models share the same initial state
        state0 = np.array([x0, y0, x0, y0, x0, y0])
        
        # Solve coupled ODE system
        solution = odeint(self.derivatives, state0, t)
        
        # Extract individual model states
        x1, y1 = solution[:, 0], solution[:, 1]
        x2, y2 = solution[:, 2], solution[:, 3]
        x3, y3 = solution[:, 4], solution[:, 5]
        
        # Output is average
        x_super = (x1 + x2 + x3) / 3
        y_super = (y1 + y2 + y3) / 3
        
        return np.column_stack([x_super, y_super])
    
    def get_coupling_coefficients(self) -> Dict[str, float]:
        """Return all coupling coefficients as a dictionary."""
        return {
            "Cx_12": self.Cx_12, "Cx_13": self.Cx_13,
            "Cx_21": self.Cx_21, "Cx_23": self.Cx_23,
            "Cx_31": self.Cx_31, "Cx_32": self.Cx_32,
            "Cy_12": self.Cy_12, "Cy_13": self.Cy_13,
            "Cy_21": self.Cy_21, "Cy_23": self.Cy_23,
            "Cy_31": self.Cy_31, "Cy_32": self.Cy_32,
        }
    
    @classmethod
    def from_params(
        cls, 
        submodels: List[LotkaVolterraSurrogate], 
        params: Dict[str, float]
    ) -> "Supermodel":
        """
        Create Supermodel from a dictionary of coupling coefficients.
        
        Args:
            submodels: List of 3 pretrained submodels
            params: Dictionary with keys Cx_12, Cx_13, ..., Cy_32
            
        Returns:
            Supermodel instance
        """
        return cls(
            submodels=submodels,
            Cx_12=params.get("Cx_12", 0.0),
            Cx_13=params.get("Cx_13", 0.0),
            Cx_21=params.get("Cx_21", 0.0),
            Cx_23=params.get("Cx_23", 0.0),
            Cx_31=params.get("Cx_31", 0.0),
            Cx_32=params.get("Cx_32", 0.0),
            Cy_12=params.get("Cy_12", 0.0),
            Cy_13=params.get("Cy_13", 0.0),
            Cy_21=params.get("Cy_21", 0.0),
            Cy_23=params.get("Cy_23", 0.0),
            Cy_31=params.get("Cy_31", 0.0),
            Cy_32=params.get("Cy_32", 0.0),
        )


# List of all coupling coefficient names for ABC
COUPLING_PARAM_NAMES = [
    "Cx_12", "Cx_13", "Cx_21", "Cx_23", "Cx_31", "Cx_32",
    "Cy_12", "Cy_13", "Cy_21", "Cy_23", "Cy_31", "Cy_32",
]
