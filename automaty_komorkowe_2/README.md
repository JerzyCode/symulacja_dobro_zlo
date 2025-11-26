# Continuous Cellular Automaton - Spatiotemporal Chaos

Simulation of spatiotemporal chaos using Coupled Map Lattice (CML) based on the logistic map.

## Model Description

The system evolves according to the coupled map lattice equation:

```
x_{n+1}(i,j) = (1 - ε) * f(x_n(i,j)) + (ε/k) * Σ f(x_n(neighbors))
```

Where:

- `f(x) = r * x * (1 - x)` - logistic map
- `r` - bifurcation parameter (controls chaos)
- `ε` - coupling strength (controls synchronization)
- `k` - number of neighbors (4 for von Neumann, 8 for Moore)

## Key Parameters

| Parameter    | Range           | Behavior                 |
| ------------ | --------------- | ------------------------ |
| r < 3.0      | Stable          | Converges to fixed point |
| r ≈ 3.0-3.57 | Periodic        | Period doubling cascade  |
| r > 3.57     | Chaotic         | Spatiotemporal chaos     |
| r = 4.0      | Fully chaotic   | Maximum entropy          |
| ε = 0        | Uncoupled       | Independent cells        |
| ε → 1        | Strong coupling | Synchronization          |

## Installation

```bash
uv sync
```

## Usage

```bash
uv run python main.py
```

This will generate:

- `spatial_evolution.png` - Grid state at different time steps
- `time_series.png` - Mean population and individual cell dynamics
- `bifurcation_diagram.png` - Bifurcation diagram showing chaos transition
- `synchronization.png` - Synchronization vs coupling strength
- `r_comparison.png` - Final states for different r values
- `neighbor_comparison.png` - Regular vs random neighbor selection

## Chaos Indicators

1. **Bifurcation diagram**: Shows transition from stable → periodic → chaotic regimes
2. **Spatial variance**: Low variance indicates synchronization
3. **Time series**: Irregular patterns indicate chaos
4. **Sensitivity to initial conditions**: Chaotic systems diverge exponentially

## References

Dzwinel, W. (2010). Spatially extended populations reproducing logistic map.
_Central European Journal of Physics_, 8(1), 33-41.
