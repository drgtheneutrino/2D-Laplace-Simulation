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
