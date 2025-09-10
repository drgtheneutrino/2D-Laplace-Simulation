# Heat Equilibrium Simulation

A Python implementation of a 2D heat equilibrium solver using the finite difference method. This simulation visualizes how temperature distributes across a rectangular grid with customizable boundary conditions until thermal equilibrium is reached.

## Installation

### Prerequisites
- Python 3.7 or higher
- NumPy
- Matplotlib

### Install Dependencies
```bash
pip install numpy matplotlib
```

## Usage

### Basic Example
```bash
python heat_simulation.py
```

This runs the simulation with default parameters:
- 50×75 grid
- Top wall: 100°
- Bottom wall: 0°
- Left wall: 50°
- Right wall: 75°

### Customization

Edit the parameters in `heat_simulation.py`:

```python
sim = HeatEquilibriumSimulation(
    rows=100,                      # Number of grid rows
    cols=150,                      # Number of grid columns
    temp_top=100,                  # Top boundary temperature
    temp_bottom=0,                 # Bottom boundary temperature
    temp_left=50,                  # Left boundary temperature
    temp_right=75,                 # Right boundary temperature
    convergence_threshold=0.001    # Stop when avg change < this value
)
```

### High Resolution Example

For a detailed simulation with thousands of cells:

```python
sim = HeatEquilibriumSimulation(
    rows=1000,
    cols=1500,
    temp_top=1000,
    temp_bottom=200,
    temp_left=500,
    temp_right=750,
    convergence_threshold=0.0001
)
```

## How It Works

The simulation uses the **Jacobi iterative method** to solve the 2D Laplace equation:

1. Initialize a grid with boundary conditions (wall temperatures)
2. Each interior cell updates to the average of its 4 neighbors
3. Repeat until changes are below the convergence threshold
4. Visualize the final temperature distribution


Where each interior point converges to:
```
T[i,j] = (T[i-1,j] + T[i+1,j] + T[i,j-1] + T[i,j+1]) / 4
```

## Output

The simulation provides:

### Console Output
- Progress updates every 100 iterations
- Step counter and elapsed time
- Convergence status and final statistics

### Visualization
- **Left plot**: Heat distribution map with labeled boundary temperatures
- **Right plot**: Convergence history (log scale) showing solution stability

## Performance

- **Small grids** (50×75): ~1 second
- **Medium grids** (500×500): ~10-30 seconds
- **Large grids** (1000×1500): ~1-5 minutes

*Times vary based on boundary conditions and convergence threshold*

## Algorithm Details

- **Method**: Jacobi iteration (parallelizable)
- **Boundary Conditions**: Dirichlet (fixed temperature)
- **Convergence Criterion**: Average absolute change per cell
