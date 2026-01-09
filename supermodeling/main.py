"""
Supermodeling Experiment with Lotka-Volterra Model

This script implements the supermodeling assignment:
1. Create baseline predator-prey model (ground truth)
2. Generate observations from baseline
3. Train Surrogate 1 with full ABC inference
4. Train Surrogates 2, 3, 4 with 1/8 the training time
5. Compare all models' predictions

Usage:
    uv run python supermodeling/main.py
"""

import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from models.lotka_volterra import (
    LotkaVolterraBaseline, 
    LotkaVolterraSurrogate,
    sample_observations,
    compute_mse
)
from models.data_assimilation import run_abc, run_abc_supermodel, get_best_parameters, get_posterior_mean
from models.supermodel import Supermodel
from visualization import (
    plot_trajectories,
    plot_prediction_errors,
    plot_parameter_comparison,
    print_results_table
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Baseline (ground truth) parameters
BASELINE_PARAMS = {
    "alpha": 1.0,    # prey birth rate
    "beta": 0.1,     # prey death rate from predation
    "gamma": 1.5,    # predator death rate
    "delta": 0.075,  # predator birth rate from consumption
}

# Initial state
INITIAL_STATE = (10.0, 5.0)  # (prey, predator)

# Time configuration
T_START = 0.0
T_TRAIN_END = 20.0   # End of training period
T_END = 30.0         # End of test period
DT = 0.1             # Time step

# Number of observation samples (2-3x number of parameters = 2-3 * 4 = 8-12)
N_SAMPLES = 10

# ABC configuration for full training (Surrogate 1)
ABC_FULL_POPULATIONS = 8
ABC_FULL_POP_SIZE = 100

# ABC configuration for quick training (Surrogates 2, 3, 4) - 1/8 of full
ABC_QUICK_POPULATIONS = 1  # 8 / 8 = 1
ABC_QUICK_POP_SIZE = 100

# ABC configuration for supermodel coupling (uses remaining budget)
ABC_SUPER_POPULATIONS = 3
ABC_SUPER_POP_SIZE = 100

# Output directory
OUTPUT_DIR = Path(__file__).parent / "results"


def main():
    """Run the supermodeling experiment."""
    
    print("\n" + "=" * 80)
    print("SUPERMODELING EXPERIMENT - LOTKA-VOLTERRA MODEL")
    print("=" * 80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # ========================================================================
    # STEP 1: Create baseline model and generate observations
    # ========================================================================
    print("\n[STEP 1] Creating baseline model and generating observations...")
    
    baseline = LotkaVolterraBaseline(**BASELINE_PARAMS)
    
    # Time arrays
    t_full = np.arange(T_START, T_END + DT, DT)
    t_train = t_full[t_full <= T_TRAIN_END]
    t_test = t_full[t_full > T_TRAIN_END]
    
    # Generate baseline trajectory
    baseline_solution = baseline.simulate(t_full, INITIAL_STATE)
    
    # Sample observations from training period
    t_obs, observations = sample_observations(
        baseline, t_train, N_SAMPLES, INITIAL_STATE, noise_std=0.5, seed=42
    )
    
    print(f"  - Baseline parameters: {BASELINE_PARAMS}")
    print(f"  - Time range: {T_START} to {T_END}")
    print(f"  - Training period: {T_START} to {T_TRAIN_END}")
    print(f"  - Test period: {T_TRAIN_END} to {T_END}")
    print(f"  - Number of observations: {len(t_obs)}")
    
    # ========================================================================
    # STEP 2: Train Surrogate Model 1 with full ABC
    # ========================================================================
    print("\n[STEP 2] Training Surrogate Model 1 (full ABC inference)...")
    print(f"  - Populations: {ABC_FULL_POPULATIONS}, Population size: {ABC_FULL_POP_SIZE}")
    
    history_full = run_abc(
        observations=observations,
        t_obs=t_obs,
        initial_state=INITIAL_STATE,
        t_full=t_train,
        n_populations=ABC_FULL_POPULATIONS,
        population_size=ABC_FULL_POP_SIZE,
        min_epsilon=0.01,
    )
    
    # Get best parameters for Surrogate 1
    surrogate1_params = get_posterior_mean(history_full)
    surrogate1 = LotkaVolterraSurrogate.from_params(surrogate1_params)
    
    print(f"  - Surrogate 1 parameters: a={surrogate1_params['a']:.4f}, "
          f"b={surrogate1_params['b']:.4f}")
    
    # ========================================================================
    # STEP 3: Train Surrogate Models 2, 3, 4 with quick training (1/8 time)
    # ========================================================================
    print("\n[STEP 3] Training Surrogate Models 2, 3, 4 (1/8 training time)...")
    print(f"  - Populations: {ABC_QUICK_POPULATIONS}, Population size: {ABC_QUICK_POP_SIZE}")
    
    history_quick = run_abc(
        observations=observations,
        t_obs=t_obs,
        initial_state=INITIAL_STATE,
        t_full=t_train,
        n_populations=ABC_QUICK_POPULATIONS,
        population_size=ABC_QUICK_POP_SIZE,
        min_epsilon=0.5,  # Higher epsilon for quicker convergence
    )
    
    # Get 3 best parameter sets for Surrogates 2, 3, 4
    best_params_list = get_best_parameters(history_quick, n_best=3)
    
    surrogate_models = [surrogate1]
    surrogate_params_list = [surrogate1_params]
    
    for i, params in enumerate(best_params_list):
        surrogate = LotkaVolterraSurrogate.from_params(params)
        surrogate_models.append(surrogate)
        surrogate_params_list.append(params)
        print(f"  - Surrogate {i+2} parameters: a={params['a']:.4f}, b={params['b']:.4f}")
    
    # ========================================================================
    # STEP 4: Train Supermodel (coupling submodels 2, 3, 4)
    # ========================================================================
    print("\n[STEP 4] Training Supermodel (coupling submodels 2, 3, 4)...")
    print(f"  - Populations: {ABC_SUPER_POPULATIONS}, Population size: {ABC_SUPER_POP_SIZE}")
    print(f"  - Learning 12 coupling coefficients...")
    
    # Submodels are surrogates 2, 3, 4 (indices 1, 2, 3 in surrogate_models)
    submodels = [surrogate_models[1], surrogate_models[2], surrogate_models[3]]
    
    history_super = run_abc_supermodel(
        observations=observations,
        t_obs=t_obs,
        initial_state=INITIAL_STATE,
        t_full=t_train,
        submodels=submodels,
        n_populations=ABC_SUPER_POPULATIONS,
        population_size=ABC_SUPER_POP_SIZE,
        min_epsilon=0.1,
    )
    
    # Get best coupling coefficients
    super_params = get_posterior_mean(history_super)
    supermodel = Supermodel.from_params(submodels, super_params)
    
    print(f"  - Coupling coefficients learned:")
    for name, val in super_params.items():
        print(f"      {name}: {val:.4f}")
    
    # ========================================================================
    # STEP 5: Compare all models (Baseline, Surrogate 1, Supermodel)
    # ========================================================================
    print("\n[STEP 5] Comparing Baseline, Surrogate 1, and Supermodel...")
    
    # Models to compare
    model_names = ["Surrogate 1 (full)", "Supermodel (coupled)"]
    
    model_solutions = []
    train_mse = []
    test_mse = []
    comparison_params = []
    
    # Get training and test indices
    train_mask = t_full <= T_TRAIN_END
    test_mask = t_full > T_TRAIN_END
    
    # Surrogate 1
    sol1 = surrogate1.simulate(t_full, INITIAL_STATE)
    model_solutions.append(("Surrogate 1 (full)", sol1))
    train_mse.append(compute_mse(baseline_solution[train_mask], sol1[train_mask]))
    test_mse.append(compute_mse(baseline_solution[test_mask], sol1[test_mask]))
    comparison_params.append(surrogate1_params)
    
    # Supermodel
    sol_super = supermodel.simulate(t_full, INITIAL_STATE)
    model_solutions.append(("Supermodel (coupled)", sol_super))
    train_mse.append(compute_mse(baseline_solution[train_mask], sol_super[train_mask]))
    test_mse.append(compute_mse(baseline_solution[test_mask], sol_super[test_mask]))
    comparison_params.append({"coupling": "12 params"})
    
    # Print comparison
    print("\n" + "-" * 60)
    print(f"{'Model':<25} {'Train MSE':>12} {'Test MSE':>12}")
    print("-" * 60)
    for i, name in enumerate(model_names):
        print(f"{name:<25} {train_mse[i]:>12.4f} {test_mse[i]:>12.4f}")
    print("-" * 60)
    
    # ========================================================================
    # STEP 6: Visualizations
    # ========================================================================
    print("\n[STEP 6] Generating visualizations...")
    
    # Plot trajectories
    plot_trajectories(
        t_full,
        baseline_solution,
        model_solutions,
        observations=observations,
        t_obs=t_obs,
        t_train_end=T_TRAIN_END,
        save_path=OUTPUT_DIR / "trajectories.png",
        title="Supermodeling: Baseline vs Surrogate 1 vs Supermodel"
    )
    
    # Plot prediction errors
    plot_prediction_errors(
        model_names,
        train_mse,
        test_mse,
        save_path=OUTPUT_DIR / "prediction_errors.png"
    )
    
    # Note: Parameter comparison not applicable for supermodel (coupling coefficients)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
