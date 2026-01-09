"""
Data assimilation using custom ABC-SMC implementation.

Provides ABC-SMC (Approximate Bayesian Computation - Sequential Monte Carlo)
inference for surrogate model parameters without external ABC libraries.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Callable, Optional
from scipy.integrate import odeint


@dataclass
class ABCHistory:
    """
    Stores results from ABC-SMC inference.

    Attributes:
        populations: List of parameter populations per generation
        weights: List of particle weights per generation
        epsilons: Acceptance thresholds per generation
        distances: List of distances per generation
    """

    populations: List[np.ndarray] = field(default_factory=list)
    weights: List[np.ndarray] = field(default_factory=list)
    epsilons: List[float] = field(default_factory=list)
    distances: List[np.ndarray] = field(default_factory=list)
    param_names: List[str] = field(default_factory=list)

    def get_distribution(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Get final population distribution.

        Returns:
            Tuple of (parameter dict, weights array)
        """
        if not self.populations:
            raise ValueError("No populations available")

        final_pop = self.populations[-1]
        final_weights = self.weights[-1]

        param_dict = {name: final_pop[:, i] for i, name in enumerate(self.param_names)}

        return param_dict, final_weights

    @property
    def n_generations(self) -> int:
        """Number of completed generations."""
        return len(self.populations)


def create_model_simulator(
    t_obs: np.ndarray, initial_state: Tuple[float, float], t_full: np.ndarray
) -> Callable:
    """
    Create a simulator function for ABC.

    Args:
        t_obs: Observation time points
        initial_state: Initial state (prey, predator)
        t_full: Full time array for integration

    Returns:
        Simulator function that takes parameters and returns simulated observations
    """

    def simulator(a: float, b: float) -> np.ndarray:
        """Simulate surrogate model with given parameters."""

        def derivatives(state, t):
            x, y = state
            dxdt = a * x - b * x * y
            dydt = b * x * y - a * y
            return [dxdt, dydt]

        try:
            # Simulate over full time range
            solution = odeint(derivatives, initial_state, t_full)

            # Find indices closest to observation times
            obs_indices = [np.argmin(np.abs(t_full - t)) for t in t_obs]
            simulated = solution[obs_indices]

            # Flatten for comparison
            return simulated.flatten()
        except Exception:
            # Return large values if simulation fails
            return np.full(len(t_obs) * 2, 1e6)

    return simulator


def mse_distance(simulated: np.ndarray, observed: np.ndarray) -> float:
    """Compute MSE distance between simulated and observed data."""
    return np.mean((simulated - observed) ** 2)


def sample_from_prior(
    prior_bounds: Dict[str, Tuple[float, float]], n_samples: int
) -> np.ndarray:
    """
    Sample parameters from uniform prior.

    Args:
        prior_bounds: Dictionary of parameter bounds {"name": (min, max)}
        n_samples: Number of samples to draw

    Returns:
        Array of shape (n_samples, n_params)
    """
    samples = []
    for name in sorted(prior_bounds.keys()):
        low, high = prior_bounds[name]
        samples.append(np.random.uniform(low, high, n_samples))
    return np.column_stack(samples)


def compute_kernel_bandwidth(population: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute adaptive kernel bandwidth for each parameter (twice weighted std).

    Args:
        population: Array of shape (n_particles, n_params)
        weights: Normalized weights

    Returns:
        Bandwidth for each parameter
    """
    weighted_mean = np.average(population, axis=0, weights=weights)
    weighted_var = np.average(
        (population - weighted_mean) ** 2, axis=0, weights=weights
    )
    return 2 * np.sqrt(weighted_var)


def perturb_particle(
    particle: np.ndarray,
    bandwidth: np.ndarray,
    prior_bounds: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    """
    Perturb a particle using Gaussian kernel with reflection at boundaries.

    Args:
        particle: Parameter values
        bandwidth: Kernel bandwidth per parameter
        prior_bounds: Parameter bounds

    Returns:
        Perturbed particle
    """
    param_names = sorted(prior_bounds.keys())
    perturbed = particle + np.random.normal(0, bandwidth)

    # Reflect at boundaries
    for i, name in enumerate(param_names):
        low, high = prior_bounds[name]
        while perturbed[i] < low or perturbed[i] > high:
            if perturbed[i] < low:
                perturbed[i] = 2 * low - perturbed[i]
            if perturbed[i] > high:
                perturbed[i] = 2 * high - perturbed[i]

    return perturbed


def run_abc(
    observations: np.ndarray,
    t_obs: np.ndarray,
    initial_state: Tuple[float, float],
    t_full: np.ndarray,
    n_populations: int = 10,
    min_epsilon: float = 0.1,
    population_size: int = 100,
    prior_bounds: Dict[str, Tuple[float, float]] = None,
    db_path: str = None,  # Kept for API compatibility, not used
    quantile: float = 0.5,
    max_attempts_factor: int = 100,
    verbose: bool = True,
) -> ABCHistory:
    """
    Run ABC-SMC inference for surrogate model parameters.

    This is a custom implementation of the ABC-SMC algorithm that doesn't
    require any external ABC libraries.

    Args:
        observations: Observed data points (n_obs, 2)
        t_obs: Observation time points
        initial_state: Initial state for simulation
        t_full: Full time array
        n_populations: Maximum number of ABC-SMC generations
        min_epsilon: Minimum acceptance threshold (stopping criterion)
        population_size: Number of particles per population
        prior_bounds: Prior bounds for parameters {"a": (min, max), "b": (min, max)}
        db_path: Ignored (kept for API compatibility)
        quantile: Quantile for epsilon selection (default 0.5 = median)
        max_attempts_factor: Maximum attempts per particle as factor of population_size
        verbose: Whether to print progress

    Returns:
        ABCHistory object with inference results
    """
    if prior_bounds is None:
        prior_bounds = {
            "a": (0.1, 3.0),
            "b": (0.01, 0.5),
        }

    param_names = sorted(prior_bounds.keys())
    n_params = len(param_names)

    # Create simulator
    simulator = create_model_simulator(t_obs, initial_state, t_full)

    # Prepare observed data
    observed_flat = observations.flatten()

    # Initialize history
    history = ABCHistory(param_names=param_names)

    # ========================================================================
    # Generation 0: Sample from prior
    # ========================================================================
    if verbose:
        print("    ABC-SMC Generation 0 (sampling from prior)...")

    population = []
    distances = []
    max_attempts = population_size * max_attempts_factor
    attempts = 0

    while len(population) < population_size and attempts < max_attempts:
        attempts += 1

        # Sample from prior
        params = sample_from_prior(prior_bounds, 1)[0]

        # Simulate
        simulated = simulator(params[0], params[1])

        # Compute distance
        dist = mse_distance(simulated, observed_flat)

        # Accept all in first generation (we'll set epsilon based on these)
        if np.isfinite(dist):
            population.append(params)
            distances.append(dist)

    if len(population) < population_size:
        raise RuntimeError(
            f"Could not generate enough valid particles. "
            f"Got {len(population)}/{population_size} after {max_attempts} attempts."
        )

    population = np.array(population)
    distances = np.array(distances)
    weights = np.ones(population_size) / population_size

    # Set initial epsilon as the quantile of distances
    epsilon = np.quantile(distances, quantile)

    # Filter to keep only particles below epsilon
    mask = distances <= epsilon
    population = population[mask]
    distances = distances[mask]
    weights = np.ones(len(population)) / len(population)

    history.populations.append(population.copy())
    history.weights.append(weights.copy())
    history.distances.append(distances.copy())
    history.epsilons.append(epsilon)

    if verbose:
        print(f"      Epsilon: {epsilon:.4f}, Accepted: {len(population)}")

    # ========================================================================
    # Subsequent generations: Sample from previous population with perturbation
    # ========================================================================
    for gen in range(1, n_populations):
        if epsilon <= min_epsilon:
            if verbose:
                print(f"    Reached min_epsilon ({min_epsilon}), stopping.")
            break

        if verbose:
            print(f"    ABC-SMC Generation {gen}...")

        prev_population = history.populations[-1]
        prev_weights = history.weights[-1]

        # Compute kernel bandwidth
        bandwidth = compute_kernel_bandwidth(prev_population, prev_weights)

        new_population = []
        new_distances = []
        new_weights = []
        attempts = 0

        while len(new_population) < population_size and attempts < max_attempts:
            attempts += 1

            # Sample particle from previous population
            idx = np.random.choice(len(prev_population), p=prev_weights)
            particle = prev_population[idx]

            # Perturb
            perturbed = perturb_particle(particle, bandwidth, prior_bounds)

            # Simulate
            simulated = simulator(perturbed[0], perturbed[1])

            # Compute distance
            dist = mse_distance(simulated, observed_flat)

            # Accept if distance < epsilon
            if dist < epsilon and np.isfinite(dist):
                new_population.append(perturbed)
                new_distances.append(dist)

                # Compute weight: prior / (sum of kernel densities from prev pop)
                # For uniform prior, this simplifies to 1 / kernel_density
                kernel_sum = 0
                for j in range(len(prev_population)):
                    kernel_prob = np.prod(
                        np.exp(
                            -0.5 * ((perturbed - prev_population[j]) / bandwidth) ** 2
                        )
                        / (bandwidth * np.sqrt(2 * np.pi))
                    )
                    kernel_sum += prev_weights[j] * kernel_prob

                new_weights.append(1.0 / max(kernel_sum, 1e-10))

        if len(new_population) < population_size // 2:
            if verbose:
                print(
                    f"      Warning: Only got {len(new_population)} particles, stopping early."
                )
            break

        new_population = np.array(new_population)
        new_distances = np.array(new_distances)
        new_weights = np.array(new_weights)
        new_weights = new_weights / new_weights.sum()  # Normalize

        # Update epsilon for next generation
        epsilon = np.quantile(new_distances, quantile)
        epsilon = max(epsilon, min_epsilon)

        history.populations.append(new_population)
        history.weights.append(new_weights)
        history.distances.append(new_distances)
        history.epsilons.append(epsilon)

        if verbose:
            print(f"      Epsilon: {epsilon:.4f}, Particles: {len(new_population)}")

    return history


def get_best_parameters(history: ABCHistory, n_best: int = 1) -> List[Dict[str, float]]:
    """
    Extract best parameters from ABC history.

    Args:
        history: ABCHistory object
        n_best: Number of best parameter sets to return

    Returns:
        List of parameter dictionaries, sorted by weight
    """
    param_dict, weights = history.get_distribution()

    # Sort by weight (descending)
    sorted_indices = np.argsort(weights)[::-1]

    best_params = []
    for i in range(min(n_best, len(sorted_indices))):
        idx = sorted_indices[i]
        params = {name: param_dict[name][idx] for name in history.param_names}
        best_params.append(params)

    return best_params


def get_posterior_mean(history: ABCHistory) -> Dict[str, float]:
    """
    Get weighted posterior mean of parameters.

    Args:
        history: ABCHistory object

    Returns:
        Dictionary with posterior mean parameters
    """
    param_dict, weights = history.get_distribution()

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    return {
        name: np.average(param_dict[name], weights=weights)
        for name in history.param_names
    }


def run_abc_supermodel(
    observations: np.ndarray,
    t_obs: np.ndarray,
    initial_state: Tuple[float, float],
    t_full: np.ndarray,
    submodels: List,  # List of LotkaVolterraSurrogate
    n_populations: int = 3,
    min_epsilon: float = 0.1,
    population_size: int = 100,
    prior_bounds: Dict[str, Tuple[float, float]] = None,
    quantile: float = 0.5,
    max_attempts_factor: int = 100,
    verbose: bool = True,
) -> ABCHistory:
    """
    Run ABC-SMC to find optimal supermodel coupling coefficients.

    Learns 12 coupling coefficients (C^x_ij and C^y_ij for all pairs).

    Args:
        observations: Observed data points (n_obs, 2)
        t_obs: Observation time points
        initial_state: Initial state for simulation
        t_full: Full time array
        submodels: List of 3 pretrained LotkaVolterraSurrogate models
        n_populations: Number of ABC-SMC generations
        min_epsilon: Minimum acceptance threshold
        population_size: Number of particles per population
        prior_bounds: Prior bounds for coupling coefficients (default [0, 1])
        quantile: Quantile for epsilon selection
        max_attempts_factor: Maximum attempts factor
        verbose: Print progress

    Returns:
        ABCHistory with learned coupling coefficients
    """
    from .supermodel import Supermodel, COUPLING_PARAM_NAMES

    # Default prior bounds for coupling coefficients
    if prior_bounds is None:
        prior_bounds = {name: (0.0, 1.0) for name in COUPLING_PARAM_NAMES}

    param_names = sorted(prior_bounds.keys())
    n_params = len(param_names)

    # Prepare observed data
    observed_flat = observations.flatten()

    def simulate_supermodel(params: np.ndarray) -> np.ndarray:
        """Simulate supermodel with given coupling coefficients."""
        param_dict = {name: params[i] for i, name in enumerate(param_names)}

        try:
            supermodel = Supermodel.from_params(submodels, param_dict)
            solution = supermodel.simulate(t_full, initial_state)

            # Get values at observation times
            obs_indices = [np.argmin(np.abs(t_full - t)) for t in t_obs]
            simulated = solution[obs_indices]

            return simulated.flatten()
        except Exception:
            return np.full(len(t_obs) * 2, 1e6)

    # Initialize history
    history = ABCHistory(param_names=param_names)

    # ========================================================================
    # Generation 0: Sample from prior
    # ========================================================================
    if verbose:
        print("    ABC-SMC Supermodel Gen 0 (sampling from prior)...")

    population = []
    distances = []
    max_attempts = population_size * max_attempts_factor
    attempts = 0

    while len(population) < population_size and attempts < max_attempts:
        attempts += 1

        # Sample from prior
        params = sample_from_prior(prior_bounds, 1)[0]

        # Simulate
        simulated = simulate_supermodel(params)

        # Compute distance
        dist = mse_distance(simulated, observed_flat)

        if np.isfinite(dist):
            population.append(params)
            distances.append(dist)

    if len(population) < population_size:
        raise RuntimeError(
            f"Could not generate enough valid particles. "
            f"Got {len(population)}/{population_size} after {max_attempts} attempts."
        )

    population = np.array(population)
    distances = np.array(distances)
    weights = np.ones(population_size) / population_size

    # Set initial epsilon
    epsilon = np.quantile(distances, quantile)

    # Filter to keep only particles below epsilon
    mask = distances <= epsilon
    population = population[mask]
    distances = distances[mask]
    weights = np.ones(len(population)) / len(population)

    history.populations.append(population.copy())
    history.weights.append(weights.copy())
    history.distances.append(distances.copy())
    history.epsilons.append(epsilon)

    if verbose:
        print(f"      Epsilon: {epsilon:.4f}, Accepted: {len(population)}")

    # ========================================================================
    # Subsequent generations
    # ========================================================================
    for gen in range(1, n_populations):
        if epsilon <= min_epsilon:
            if verbose:
                print(f"    Reached min_epsilon ({min_epsilon}), stopping.")
            break

        if verbose:
            print(f"    ABC-SMC Supermodel Gen {gen}...")

        prev_population = history.populations[-1]
        prev_weights = history.weights[-1]

        # Compute kernel bandwidth
        bandwidth = compute_kernel_bandwidth(prev_population, prev_weights)

        new_population = []
        new_distances = []
        new_weights = []
        attempts = 0

        while len(new_population) < population_size and attempts < max_attempts:
            attempts += 1

            # Sample particle from previous population
            idx = np.random.choice(len(prev_population), p=prev_weights)
            particle = prev_population[idx]

            # Perturb
            perturbed = perturb_particle(particle, bandwidth, prior_bounds)

            # Simulate
            simulated = simulate_supermodel(perturbed)

            # Compute distance
            dist = mse_distance(simulated, observed_flat)

            # Accept if distance < epsilon
            if dist < epsilon and np.isfinite(dist):
                new_population.append(perturbed)
                new_distances.append(dist)

                # Compute weight
                kernel_sum = 0
                for j in range(len(prev_population)):
                    kernel_prob = np.prod(
                        np.exp(
                            -0.5 * ((perturbed - prev_population[j]) / bandwidth) ** 2
                        )
                        / (bandwidth * np.sqrt(2 * np.pi))
                    )
                    kernel_sum += prev_weights[j] * kernel_prob

                new_weights.append(1.0 / max(kernel_sum, 1e-10))

        if len(new_population) < population_size // 2:
            if verbose:
                print(
                    f"      Warning: Only got {len(new_population)} particles, stopping early."
                )
            break

        new_population = np.array(new_population)
        new_distances = np.array(new_distances)
        new_weights = np.array(new_weights)
        new_weights = new_weights / new_weights.sum()

        # Update epsilon
        epsilon = np.quantile(new_distances, quantile)
        epsilon = max(epsilon, min_epsilon)

        history.populations.append(new_population)
        history.weights.append(new_weights)
        history.distances.append(new_distances)
        history.epsilons.append(epsilon)

        if verbose:
            print(f"      Epsilon: {epsilon:.4f}, Particles: {len(new_population)}")

    return history
