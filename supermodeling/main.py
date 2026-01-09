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

warnings.filterwarnings("ignore")

from models.lotka_volterra import (
    LotkaVolterraBaseline,
    LotkaVolterraSurrogate,
    sample_observations,
    compute_mse,
)
from models.data_assimilation import (
    run_abc,
    run_abc_supermodel,
    get_best_parameters,
    get_posterior_mean,
)
from models.supermodel import Supermodel
from visualization import (
    plot_trajectories,
    plot_prediction_errors,
    plot_parameter_comparison,
    print_results_table,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Baseline (ground truth) parameters
BASELINE_PARAMS = {
    "alpha": 1.0,  # prey birth rate
    "beta": 0.1,  # prey death rate from predation
    "gamma": 1.5,  # predator death rate
    "delta": 0.075,  # predator birth rate from consumption
}

# Initial state
INITIAL_STATE = (10.0, 5.0)  # (prey, predator)

# Time configuration
T_START = 0.0
T_TRAIN_END = 20.0  # End of training period
T_END = 30.0  # End of test period
DT = 0.1  # Time step

# Number of observation samples (2-3x number of parameters = 2-3 * 4 = 8-12)
N_SAMPLES = 10

# ABC configuration for full training (Surrogate 1)
# Cost: 20 * 200 = 4000 evaluations
ABC_FULL_POPULATIONS = 20
ABC_FULL_POP_SIZE = 200

# ABC configuration for quick training (Surrogates 2, 3, 4) - 1/8 of full
# Requirement: Time < 1/4 of Surrogate 1 (4000/4 = 1000)
# Cost: 1 * 100 = 100 evaluations
ABC_QUICK_POPULATIONS = 1
ABC_QUICK_POP_SIZE = 100

# ABC configuration for supermodel coupling (uses remaining budget)
# Requirement: Pretrain + Super <= Surrogate 1
# 100 + Super <= 4000 => Super <= 3900
# Cost: 15 * 250 = 3750 evaluations
ABC_SUPER_POPULATIONS = 15
ABC_SUPER_POP_SIZE = 250

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
    print(
        f"  - Populations: {ABC_FULL_POPULATIONS}, Population size: {ABC_FULL_POP_SIZE}"
    )

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

    print(
        f"  - Surrogate 1 parameters: a={surrogate1_params['a']:.4f}, "
        f"b={surrogate1_params['b']:.4f}"
    )

    # ========================================================================
    # STEP 3: Train Surrogate Models 2, 3, 4 with quick training (1/8 time)
    # ========================================================================
    # ========================================================================
    # STEP 3: Train Surrogate Models 2, 3, 4 with quick training (1/8 time)
    # ========================================================================
    print("\n[STEP 3] Training Surrogate Models 2, 3, 4 (1/8 training time)...")
    print(
        f"  - Populations: {ABC_QUICK_POPULATIONS}, Population size: {ABC_QUICK_POP_SIZE}"
    )
    print("  - Performing 3 INDEPENDENT training runs for diversity...")

    surrogate_models = [surrogate1]
    surrogate_params_list = [surrogate1_params]

    # Run ABC 3 times independently to get 3 distinct "quick" models
    for i in range(3):
        print(f"    Run {i + 1}/3...")
        history_quick = run_abc(
            observations=observations,
            t_obs=t_obs,
            initial_state=INITIAL_STATE,
            t_full=t_train,
            n_populations=ABC_QUICK_POPULATIONS,
            population_size=ABC_QUICK_POP_SIZE,
            min_epsilon=0.5,
            verbose=False,  # less spam
        )

        # Take the SINGLE best parameter set from this run
        best_params = get_best_parameters(history_quick, n_best=1)[0]

        surrogate = LotkaVolterraSurrogate.from_params(best_params)
        surrogate_models.append(surrogate)
        surrogate_params_list.append(best_params)

        print(
            f"  - Surrogate {i + 2} parameters: a={best_params['a']:.4f}, b={best_params['b']:.4f}"
        )

    # ========================================================================
    # STEP 4: Train Supermodel (coupling submodels 2, 3, 4)
    # ========================================================================
    print("\n[STEP 4] Training Supermodel (coupling submodels 2, 3, 4)...")
    print(
        f"  - Populations: {ABC_SUPER_POPULATIONS}, Population size: {ABC_SUPER_POP_SIZE}"
    )
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
    # STEP 5: Compare all models (Baseline, Surrogate 1, Surrogates 2-4, Supermodel)
    # ========================================================================
    print("\n[STEP 5] Comparing Baseline, Surrogates, and Supermodel...")

    # Models to compare
    model_labels = [
        "Surrogate 1 (full)",
        "Surrogate 2 (quick)",
        "Surrogate 3 (quick)",
        "Surrogate 4 (quick)",
        "Supermodel (coupled)",
    ]

    model_solutions = []
    train_mse = []
    test_mse = []

    # Get training and test indices
    train_mask = t_full <= T_TRAIN_END
    test_mask = t_full > T_TRAIN_END

    # 1. Surrogate 1 (already simulated above, but let's be uniform)
    sol1 = surrogate_models[0].simulate(t_full, INITIAL_STATE)
    model_solutions.append(("Surrogate 1 (full)", sol1))

    # 2, 3, 4. Surrogates 2, 3, 4
    for i in range(1, 4):
        sol_i = surrogate_models[i].simulate(t_full, INITIAL_STATE)
        model_solutions.append((f"Surrogate {i + 1} (quick)", sol_i))

    # 5. Supermodel
    sol_super = supermodel.simulate(t_full, INITIAL_STATE)
    model_solutions.append(("Supermodel (coupled)", sol_super))

    # Compute MSEs
    for label, sol in model_solutions:
        t_mse = compute_mse(baseline_solution[train_mask], sol[train_mask])
        v_mse = compute_mse(baseline_solution[test_mask], sol[test_mask])
        train_mse.append(t_mse)
        test_mse.append(v_mse)

    # Print comparison
    print("\n" + "-" * 70)
    print(f"{'Model':<25} {'Train MSE':>12} {'Test MSE':>12}")
    print("-" * 70)
    for i, name in enumerate(model_labels):
        print(f"{name:<25} {train_mse[i]:>12.4f} {test_mse[i]:>12.4f}")
    print("-" * 70)

    # ========================================================================
    # STEP 6: Visualizations
    # ========================================================================
    print("\n[STEP 6] Generating visualizations...")

    # Plot trajectories
    plot_trajectories(
        t_full,
        baseline_solution,
        model_solutions,  # Now contains all 5 models
        observations=observations,
        t_obs=t_obs,
        t_train_end=T_TRAIN_END,
        save_path=OUTPUT_DIR / "trajectories.png",
        title="Supermodeling: All Models Comparison",
    )

    # Plot prediction errors
    plot_prediction_errors(
        model_labels,
        train_mse,
        test_mse,
        save_path=OUTPUT_DIR / "prediction_errors.png",
    )

    # Note: Parameter comparison not applicable for supermodel (coupling coefficients)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
