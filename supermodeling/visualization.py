"""
Visualization functions for supermodeling experiment.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from pathlib import Path


def plot_trajectories(
    t: np.ndarray,
    baseline_solution: np.ndarray,
    surrogate_solutions: List[Tuple[str, np.ndarray]],
    observations: np.ndarray = None,
    t_obs: np.ndarray = None,
    t_train_end: float = None,
    save_path: str = None,
    title: str = "Model Trajectories Comparison"
):
    """
    Plot trajectories of baseline and surrogate models.
    
    Args:
        t: Time array
        baseline_solution: Baseline model solution (n_t, 2)
        surrogate_solutions: List of (name, solution) tuples
        observations: Observed data points (optional)
        t_obs: Observation times (optional)
        t_train_end: End of training period (for vertical line)
        save_path: Path to save figure (optional)
        title: Plot title
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Colors for surrogates
    colors = plt.cm.tab10(np.linspace(0, 1, len(surrogate_solutions)))
    
    # Plot prey (x)
    ax = axes[0]
    ax.plot(t, baseline_solution[:, 0], 'k-', linewidth=2, label='Baseline (Ground Truth)')
    
    for (name, sol), color in zip(surrogate_solutions, colors):
        ax.plot(t, sol[:, 0], '--', color=color, linewidth=1.5, label=name)
    
    if observations is not None and t_obs is not None:
        ax.scatter(t_obs, observations[:, 0], c='red', s=50, zorder=5, 
                   label='Observations', edgecolors='black')
    
    if t_train_end is not None:
        ax.axvline(x=t_train_end, color='gray', linestyle=':', 
                   linewidth=2, label='Train/Test Split')
    
    ax.set_ylabel('Prey Population', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot predator (y)
    ax = axes[1]
    ax.plot(t, baseline_solution[:, 1], 'k-', linewidth=2, label='Baseline (Ground Truth)')
    
    for (name, sol), color in zip(surrogate_solutions, colors):
        ax.plot(t, sol[:, 1], '--', color=color, linewidth=1.5, label=name)
    
    if observations is not None and t_obs is not None:
        ax.scatter(t_obs, observations[:, 1], c='red', s=50, zorder=5, 
                   label='Observations', edgecolors='black')
    
    if t_train_end is not None:
        ax.axvline(x=t_train_end, color='gray', linestyle=':', linewidth=2)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Predator Population', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_prediction_errors(
    model_names: List[str],
    train_mse: List[float],
    test_mse: List[float],
    save_path: str = None
):
    """
    Plot bar chart comparing prediction errors.
    
    Args:
        model_names: Names of models
        train_mse: Training MSE for each model
        test_mse: Test MSE for each model
        save_path: Path to save figure (optional)
    """
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, train_mse, width, label='Training MSE', color='steelblue')
    bars2 = ax.bar(x + width/2, test_mse, width, label='Test MSE', color='coral')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('Prediction Error Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_parameter_comparison(
    baseline_params: Dict[str, float],
    surrogate_params: List[Tuple[str, Dict[str, float]]],
    save_path: str = None
):
    """
    Plot comparison of surrogate parameters.
    
    Args:
        baseline_params: Baseline model parameters
        surrogate_params: List of (name, params) for surrogates
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    names = [name for name, _ in surrogate_params]
    a_values = [params["a"] for _, params in surrogate_params]
    b_values = [params["b"] for _, params in surrogate_params]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    
    # Parameter 'a'
    ax = axes[0]
    bars = ax.bar(names, a_values, color=colors)
    ax.axhline(y=baseline_params["alpha"], color='red', linestyle='--', 
               linewidth=2, label=f'Baseline α = {baseline_params["alpha"]:.3f}')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Parameter a', fontsize=12)
    ax.set_title('Parameter a Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Parameter 'b'
    ax = axes[1]
    bars = ax.bar(names, b_values, color=colors)
    ax.axhline(y=baseline_params["beta"], color='red', linestyle='--', 
               linewidth=2, label=f'Baseline β = {baseline_params["beta"]:.3f}')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Parameter b', fontsize=12)
    ax.set_title('Parameter b Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def print_results_table(
    model_names: List[str],
    parameters: List[Dict[str, float]],
    train_mse: List[float],
    test_mse: List[float],
    baseline_params: Dict[str, float]
):
    """
    Print formatted results table.
    
    Args:
        model_names: Names of models
        parameters: Parameter dictionaries
        train_mse: Training MSE values
        test_mse: Test MSE values
        baseline_params: Baseline parameters for reference
    """
    print("\n" + "=" * 80)
    print("SUPERMODELING RESULTS")
    print("=" * 80)
    
    print(f"\nBaseline Parameters: α={baseline_params['alpha']:.4f}, "
          f"β={baseline_params['beta']:.4f}, γ={baseline_params['gamma']:.4f}, "
          f"δ={baseline_params['delta']:.4f}")
    
    print("\n" + "-" * 80)
    print(f"{'Model':<20} {'a':<10} {'b':<10} {'Train MSE':<12} {'Test MSE':<12}")
    print("-" * 80)
    
    for name, params, tr_mse, te_mse in zip(model_names, parameters, train_mse, test_mse):
        print(f"{name:<20} {params['a']:<10.4f} {params['b']:<10.4f} "
              f"{tr_mse:<12.4f} {te_mse:<12.4f}")
    
    print("-" * 80)
    
    # Find best model
    best_idx = np.argmin(test_mse)
    print(f"\nBest Test MSE: {model_names[best_idx]} ({test_mse[best_idx]:.4f})")
    print("=" * 80 + "\n")
