"""
Continuous Cellular Automaton based on Coupled Map Lattice (CML) with Logistic Map.
Simulates spatiotemporal chaos in a 2D population grid.

Reference: Dzwinel, W. (2010). Spatially extended populations reproducing logistic map.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize


def logistic_map(x: np.ndarray, r: float) -> np.ndarray:
    """Logistic map: f(x) = r * x * (1 - x)"""
    return r * x * (1 - x)


def get_neighbors_sum(grid: np.ndarray, mode: str = "von_neumann") -> np.ndarray:
    """
    Calculate sum of neighbor states.

    Modes:
    - 'von_neumann': 4 neighbors (up, down, left, right)
    - 'moore': 8 neighbors (including diagonals)
    """
    if mode == "von_neumann":
        neighbors = (
            np.roll(grid, 1, axis=0)  # up
            + np.roll(grid, -1, axis=0)  # down
            + np.roll(grid, 1, axis=1)  # left
            + np.roll(grid, -1, axis=1)  # right
        )
        return neighbors, 4
    elif mode == "moore":
        neighbors = (
            np.roll(grid, 1, axis=0)
            + np.roll(grid, -1, axis=0)
            + np.roll(grid, 1, axis=1)
            + np.roll(grid, -1, axis=1)
            + np.roll(np.roll(grid, 1, axis=0), 1, axis=1)
            + np.roll(np.roll(grid, 1, axis=0), -1, axis=1)
            + np.roll(np.roll(grid, -1, axis=0), 1, axis=1)
            + np.roll(np.roll(grid, -1, axis=0), -1, axis=1)
        )
        return neighbors, 8
    else:
        raise ValueError(f"Unknown mode: {mode}")


def cml_step(
    grid: np.ndarray,
    r: float,
    epsilon: float,
    neighbor_mode: str = "von_neumann",
    random_neighbors: bool = False,
) -> np.ndarray:
    """
    Single step of Coupled Map Lattice evolution.

    x_{n+1}(i,j) = (1 - ε) * f(x_n(i,j)) + (ε/k) * Σ f(x_n(neighbors))

    Parameters:
    - grid: Current state (2D array with values in [0,1])
    - r: Logistic map parameter (bifurcation parameter)
    - epsilon: Coupling strength (0 = no coupling, 1 = full coupling)
    - neighbor_mode: 'von_neumann' (4) or 'moore' (8) neighbors
    - random_neighbors: If True, randomly select neighbors each step
    """
    f_grid = logistic_map(grid, r)

    if random_neighbors:
        # Random neighbor selection - each cell gets random subset of neighbors
        n, m = grid.shape
        k = 4  # number of random neighbors
        neighbor_sum = np.zeros_like(grid)

        for _ in range(k):
            di = np.random.randint(-2, 3, size=(n, m))
            dj = np.random.randint(-2, 3, size=(n, m))
            i_idx = (np.arange(n)[:, None] + di) % n
            j_idx = (np.arange(m)[None, :] + dj) % m
            neighbor_sum += logistic_map(grid[i_idx, j_idx], r)

        new_grid = (1 - epsilon) * f_grid + (epsilon / k) * neighbor_sum
    else:
        neighbor_sum, k = get_neighbors_sum(grid, neighbor_mode)
        f_neighbors = logistic_map(neighbor_sum / k, r)  # Apply f to mean of neighbors
        # Alternative: sum of f(neighbors)
        neighbor_f_sum, _ = get_neighbors_sum(f_grid, neighbor_mode)
        new_grid = (1 - epsilon) * f_grid + (epsilon / k) * neighbor_f_sum

    return np.clip(new_grid, 0, 1)


def simulate(
    size: int = 100,
    r: float = 3.7,
    epsilon: float = 0.3,
    steps: int = 500,
    neighbor_mode: str = "von_neumann",
    random_neighbors: bool = False,
    seed: int = None,
) -> tuple:
    """Run simulation and return history."""
    if seed is not None:
        np.random.seed(seed)

    # Initialize with random densities
    grid = np.random.rand(size, size)

    history = [grid.copy()]
    mean_history = [grid.mean()]

    # Track a few individual cells
    cell_positions = [
        (size // 4, size // 4),
        (size // 2, size // 2),
        (3 * size // 4, 3 * size // 4),
    ]
    cell_histories = {pos: [grid[pos]] for pos in cell_positions}

    for _ in range(steps):
        grid = cml_step(grid, r, epsilon, neighbor_mode, random_neighbors)
        history.append(grid.copy())
        mean_history.append(grid.mean())
        for pos in cell_positions:
            cell_histories[pos].append(grid[pos])

    return np.array(history), np.array(mean_history), cell_histories


def plot_spatial_evolution(history: np.ndarray, times: list = None, title: str = ""):
    """Plot grid state at different time steps."""
    if times is None:
        times = [
            0,
            len(history) // 4,
            len(history) // 2,
            3 * len(history) // 4,
            len(history) - 1,
        ]

    fig, axes = plt.subplots(1, len(times), figsize=(4 * len(times), 4))

    for ax, t in zip(axes, times):
        im = ax.imshow(history[t], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"t = {t}")
        ax.axis("off")

    fig.colorbar(im, ax=axes, label="Density", shrink=0.6)
    fig.suptitle(title or "Spatial Evolution", fontsize=14)
    plt.tight_layout()
    return fig


def plot_time_series(mean_history: np.ndarray, cell_histories: dict, title: str = ""):
    """Plot time series of mean population and individual cells."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Mean population
    axes[0].plot(mean_history, "b-", linewidth=0.5)
    axes[0].set_ylabel("Mean Population Density")
    axes[0].set_title("Global Mean Density Over Time")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    # Individual cells
    colors = plt.cm.Set1(np.linspace(0, 1, len(cell_histories)))
    for (pos, values), color in zip(cell_histories.items(), colors):
        axes[1].plot(values, color=color, linewidth=0.5, label=f"Cell {pos}")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Cell Density")
    axes[1].set_title("Individual Cell Dynamics")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    fig.suptitle(title or "Time Series Analysis", fontsize=14)
    plt.tight_layout()
    return fig


def compute_bifurcation_diagram(
    size: int = 50,
    r_range: tuple = (2.5, 4.0),
    r_steps: int = 200,
    epsilon: float = 0.3,
    warmup: int = 200,
    samples: int = 100,
) -> tuple:
    """
    Compute bifurcation diagram for the CML.
    Returns r values and corresponding mean population densities.
    """
    r_values = np.linspace(r_range[0], r_range[1], r_steps)
    bifurcation_data = []

    print("Computing bifurcation diagram...")
    for i, r in enumerate(r_values):
        if i % 20 == 0:
            print(f"  r = {r:.3f} ({100 * i / len(r_values):.0f}%)")

        grid = np.random.rand(size, size)

        # Warmup phase
        for _ in range(warmup):
            grid = cml_step(grid, r, epsilon)

        # Sample phase
        means = []
        for _ in range(samples):
            grid = cml_step(grid, r, epsilon)
            means.append(grid.mean())

        bifurcation_data.append((r, means))

    return bifurcation_data


def plot_bifurcation_diagram(bifurcation_data: list, title: str = ""):
    """Plot bifurcation diagram."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for r, means in bifurcation_data:
        ax.plot([r] * len(means), means, "k.", markersize=0.3, alpha=0.5)

    ax.set_xlabel("r (Bifurcation Parameter)")
    ax.set_ylabel("Mean Population Density")
    ax.set_title(title or "Bifurcation Diagram of CML")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def analyze_synchronization(
    size: int = 50,
    r: float = 3.7,
    epsilon_range: tuple = (0.0, 1.0),
    epsilon_steps: int = 20,
    steps: int = 300,
) -> tuple:
    """
    Analyze synchronization as a function of coupling strength.
    Returns epsilon values and corresponding variance (measure of synchronization).
    """
    epsilon_values = np.linspace(epsilon_range[0], epsilon_range[1], epsilon_steps)
    sync_measure = []

    print("Analyzing synchronization...")
    for eps in epsilon_values:
        grid = np.random.rand(size, size)

        # Evolve
        for _ in range(steps):
            grid = cml_step(grid, r, eps)

        # Measure: spatial variance (low = synchronized)
        spatial_var = grid.var()
        sync_measure.append(spatial_var)
        print(f"  ε = {eps:.2f}: variance = {spatial_var:.6f}")

    return epsilon_values, np.array(sync_measure)


def plot_synchronization(
    epsilon_values: np.ndarray, sync_measure: np.ndarray, title: str = ""
):
    """Plot synchronization analysis."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(epsilon_values, sync_measure, "bo-", linewidth=2, markersize=6)
    ax.set_xlabel("Coupling Strength (ε)")
    ax.set_ylabel("Spatial Variance")
    ax.set_title(
        title
        or "Synchronization vs Coupling Strength\n(Low variance = High synchronization)"
    )
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    return fig


def create_animation(history: np.ndarray, interval: int = 50, skip: int = 1):
    """Create animation of the CML evolution."""
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(history[0], cmap="viridis", vmin=0, vmax=1)
    ax.axis("off")
    title = ax.set_title("t = 0")
    fig.colorbar(im, ax=ax, label="Density", shrink=0.8)

    def update(frame):
        t = frame * skip
        im.set_array(history[t])
        title.set_text(f"t = {t}")
        return [im, title]

    frames = len(history) // skip
    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    return anim


def main():
    """Main function demonstrating various analyses."""
    print("=" * 60)
    print("Continuous Cellular Automaton - Spatiotemporal Chaos")
    print("=" * 60)

    # Parameters
    SIZE = 100
    R = 3.8  # Chaotic regime (r > 3.57 for logistic map)
    EPSILON = 0.3  # Coupling strength
    STEPS = 500

    # 1. Basic simulation
    print("\n[1] Running basic simulation...")
    history, mean_history, cell_histories = simulate(
        size=SIZE, r=R, epsilon=EPSILON, steps=STEPS, seed=42
    )

    # Plot spatial evolution
    fig1 = plot_spatial_evolution(
        history, title=f"Spatial Evolution (r={R}, ε={EPSILON})"
    )
    fig1.savefig("spatial_evolution.png", dpi=150)
    print("    Saved: spatial_evolution.png")

    # Plot time series
    fig2 = plot_time_series(
        mean_history, cell_histories, title=f"Time Series (r={R}, ε={EPSILON})"
    )
    fig2.savefig("time_series.png", dpi=150)
    print("    Saved: time_series.png")

    # 2. Bifurcation diagram
    print("\n[2] Computing bifurcation diagram...")
    bif_data = compute_bifurcation_diagram(
        size=50, r_range=(2.5, 4.0), r_steps=150, epsilon=0.3
    )
    fig3 = plot_bifurcation_diagram(
        bif_data, title="Bifurcation Diagram of CML (ε=0.3)"
    )
    fig3.savefig("bifurcation_diagram.png", dpi=150)
    print("    Saved: bifurcation_diagram.png")

    # 3. Synchronization analysis
    print("\n[3] Analyzing synchronization...")
    eps_vals, sync_vals = analyze_synchronization(
        size=50, r=R, epsilon_range=(0.0, 0.8), epsilon_steps=15
    )
    fig4 = plot_synchronization(
        eps_vals, sync_vals, title=f"Synchronization Analysis (r={R})"
    )
    fig4.savefig("synchronization.png", dpi=150)
    print("    Saved: synchronization.png")

    # 4. Compare different r values
    print("\n[4] Comparing different r values...")
    fig5, axes = plt.subplots(2, 3, figsize=(15, 8))

    r_values = [2.5, 3.2, 3.5, 3.7, 3.9, 4.0]
    for ax, r in zip(axes.flat, r_values):
        hist, _, _ = simulate(size=SIZE, r=r, epsilon=EPSILON, steps=200, seed=42)
        ax.imshow(hist[-1], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"r = {r}")
        ax.axis("off")

    fig5.suptitle(f"Final State for Different r Values (ε={EPSILON})", fontsize=14)
    plt.tight_layout()
    fig5.savefig("r_comparison.png", dpi=150)
    print("    Saved: r_comparison.png")

    # 5. Compare regular vs random neighbors
    print("\n[5] Comparing regular vs random neighbors...")
    fig6, axes = plt.subplots(1, 2, figsize=(12, 5))

    hist_regular, _, _ = simulate(
        size=SIZE, r=R, epsilon=EPSILON, steps=300, random_neighbors=False, seed=42
    )
    hist_random, _, _ = simulate(
        size=SIZE, r=R, epsilon=EPSILON, steps=300, random_neighbors=True, seed=42
    )

    axes[0].imshow(hist_regular[-1], cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Regular Neighbors (von Neumann)")
    axes[0].axis("off")

    axes[1].imshow(hist_random[-1], cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Random Neighbors")
    axes[1].axis("off")

    fig6.suptitle(f"Effect of Neighbor Selection (r={R}, ε={EPSILON})", fontsize=14)
    plt.tight_layout()
    fig6.savefig("neighbor_comparison.png", dpi=150)
    print("    Saved: neighbor_comparison.png")

    print("\n" + "=" * 60)
    print("Analysis complete! Check the generated PNG files.")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
