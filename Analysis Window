# heat_simulation.py
# Save this entire file and run with: python heat_simulation.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

class HeatEquilibriumSimulation:
    def __init__(self, rows=100, cols=150, 
                 temp_top=100, temp_bottom=0, 
                 temp_left=50, temp_right=75,
                 convergence_threshold=0.0001):
        """Initialize the heat equilibrium simulation."""
        self.rows = rows
        self.cols = cols
        self.temp_top = temp_top
        self.temp_bottom = temp_bottom
        self.temp_left = temp_left
        self.temp_right = temp_right
        self.convergence_threshold = convergence_threshold
        
        # Initialize grid with zeros
        self.grid = np.zeros((rows, cols))
        
        # Set boundary conditions
        self.set_boundaries()
        
        # Store previous grid for convergence check
        self.prev_grid = self.grid.copy()
        
        # Iteration counter
        self.iteration = 0
        
        # Convergence tracking
        self.avg_changes = []
        
        # Runtime tracking
        self.start_time = None
        self.end_time = None
        
    def set_boundaries(self):
        """Set the boundary conditions."""
        # Top boundary
        self.grid[0, :] = self.temp_top
        # Bottom boundary
        self.grid[-1, :] = self.temp_bottom
        # Left boundary
        self.grid[:, 0] = self.temp_left
        # Right boundary
        self.grid[:, -1] = self.temp_right
        
        # Corner corrections (average of adjacent walls)
        self.grid[0, 0] = (self.temp_top + self.temp_left) / 2
        self.grid[0, -1] = (self.temp_top + self.temp_right) / 2
        self.grid[-1, 0] = (self.temp_bottom + self.temp_left) / 2
        self.grid[-1, -1] = (self.temp_bottom + self.temp_right) / 2
    
    def update_vectorized(self):
        """Vectorized version of update for better performance."""
        self.prev_grid = self.grid.copy()
        
        # Update interior cells using numpy slicing (much faster)
        self.grid[1:-1, 1:-1] = (self.prev_grid[:-2, 1:-1] +  # top neighbors
                                  self.prev_grid[2:, 1:-1] +   # bottom neighbors
                                  self.prev_grid[1:-1, :-2] +  # left neighbors
                                  self.prev_grid[1:-1, 2:]) / 4  # right neighbors
        
        # Calculate average change
        change = np.abs(self.grid - self.prev_grid)
        avg_change = np.mean(change[1:-1, 1:-1])
        self.avg_changes.append(avg_change)
        
        self.iteration += 1
        
        return avg_change
    
    def run_until_convergence(self, max_iterations=10000):
        """Run the simulation until convergence or max iterations."""
        print(f"Starting simulation with {self.rows}x{self.cols} grid...")
        print(f"Boundary temperatures: Top={self.temp_top}°, Bottom={self.temp_bottom}°, "
              f"Left={self.temp_left}°, Right={self.temp_right}°")
        print(f"Convergence threshold: {self.convergence_threshold}")
        print("-" * 60)
        
        # Start timer
        self.start_time = time.time()
        
        while self.iteration < max_iterations:
            avg_change = self.update_vectorized()
            
            # Display progress with step counter and runtime
            if self.iteration % 100 == 0:
                elapsed = time.time() - self.start_time
                print(f"Step {self.iteration:5d} | Time: {elapsed:6.2f}s | Avg change: {avg_change:.6f}")
            
            if avg_change < self.convergence_threshold:
                self.end_time = time.time()
                total_time = self.end_time - self.start_time
                print("-" * 60)
                print(f"✓ CONVERGED after {self.iteration} steps!")
                print(f"✓ Total runtime: {total_time:.2f} seconds")
                print(f"✓ Final average change: {avg_change:.8f}")
                break
        else:
            self.end_time = time.time()
            total_time = self.end_time - self.start_time
            print("-" * 60)
            print(f"⚠ Reached maximum iterations ({max_iterations})")
            print(f"⚠ Total runtime: {total_time:.2f} seconds")
            print(f"⚠ Final average change: {avg_change:.8f}")
    
    def visualize_static(self):
        """Create a static visualization of the current state."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calculate min and max temperatures for proper scaling
        all_temps = [self.temp_top, self.temp_bottom, self.temp_left, self.temp_right]
        vmin = min(all_temps)
        vmax = max(all_temps)
        
        # Heat map with proper scale
        im = ax1.imshow(self.grid, cmap='hot', interpolation='bilinear',
                       vmin=vmin, vmax=vmax)
        
        # Add title with step count and runtime
        total_time = self.end_time - self.start_time if self.end_time else 0
        ax1.set_title(f'Heat Distribution\nSteps: {self.iteration} | Runtime: {total_time:.2f}s', 
                     fontsize=12, fontweight='bold')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        # Add colorbar with proper scale
        cbar = plt.colorbar(im, ax=ax1, label='Temperature (°)')
        
        # Add boundary temperature labels
        ax1.text(self.cols/2, -5, f'Top: {self.temp_top}°', 
                ha='center', va='bottom', fontweight='bold', color='red')
        ax1.text(self.cols/2, self.rows + 2, f'Bottom: {self.temp_bottom}°', 
                ha='center', va='top', fontweight='bold', color='blue')
        ax1.text(-5, self.rows/2, f'Left: {self.temp_left}°', 
                ha='right', va='center', fontweight='bold', rotation=90, color='green')
        ax1.text(self.cols + 2, self.rows/2, f'Right: {self.temp_right}°', 
                ha='left', va='center', fontweight='bold', rotation=270, color='orange')
        
        # Convergence plot
        if len(self.avg_changes) > 0:
            ax2.semilogy(self.avg_changes, linewidth=2, color='darkblue')
            ax2.set_xlabel('Iteration Step')
            ax2.set_ylabel('Average Change per Cell (log scale)')
            ax2.set_title(f'Convergence History\nTotal Steps: {self.iteration}', 
                         fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=self.convergence_threshold, color='r', 
                       linestyle='--', linewidth=2,
                       label=f'Threshold = {self.convergence_threshold}')
            
            # Add step counter annotations at key points
            n_annotations = min(5, len(self.avg_changes))
            if n_annotations > 0:
                indices = np.linspace(0, len(self.avg_changes)-1, n_annotations, dtype=int)
                for idx in indices:
                    ax2.annotate(f'Step {idx}', 
                               xy=(idx, self.avg_changes[idx]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7)
            
            ax2.legend(loc='upper right')
        
        plt.suptitle(f'Heat Equilibrium Simulation ({self.rows}×{self.cols} grid)', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

# MAIN EXECUTION
print("=" * 60)
print("HEAT EQUILIBRIUM SIMULATION")
print("=" * 60)

# Create and run a simple example
sim = HeatEquilibriumSimulation(
    rows=50,           # Grid rows
    cols=75,           # Grid columns
    temp_top=100,      # Top wall temperature
    temp_bottom=0,     # Bottom wall temperature
    temp_left=50,      # Left wall temperature
    temp_right=75,     # Right wall temperature
    convergence_threshold=0.001  # Convergence threshold
)

# Run simulation
sim.run_until_convergence()

# Show results
sim.visualize_static()

print("\n" + "=" * 60)
print("Simulation complete! Close the plot window to exit.")
print("=" * 60)

# Example with higher temperatures (uncomment to test):
"""
sim_hot = HeatEquilibriumSimulation(
    rows=100,
    cols=150,
    temp_top=1000,     # 1000 degrees!
    temp_bottom=200,
    temp_left=500,
    temp_right=750,
    convergence_threshold=0.001
)
sim_hot.run_until_convergence()
sim_hot.visualize_static()
"""
