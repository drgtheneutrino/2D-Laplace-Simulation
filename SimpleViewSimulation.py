import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

class HeatEquilibriumSimulation:
    def __init__(self, rows=100, cols=150, 
                 temp_top=100, temp_bottom=0, 
                 temp_left=50, temp_right=75,
                 convergence_threshold=0.0001):
        """
        Initialize the heat equilibrium simulation.
        
        Parameters:
        -----------
        rows : int
            Number of rows in the grid
        cols : int
            Number of columns in the grid
        temp_top : float
            Temperature at the top boundary
        temp_bottom : float
            Temperature at the bottom boundary
        temp_left : float
            Temperature at the left boundary
        temp_right : float
            Temperature at the right boundary
        convergence_threshold : float
            Stop when average change per cell is below this value
        """
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
        
    def update(self):
        """
        Perform one iteration of the Jacobi method.
        Each interior cell takes the average of its 4 neighbors.
        """
        self.prev_grid = self.grid.copy()
        
        # Update only interior cells (not boundaries)
        for i in range(1, self.rows - 1):
            for j in range(1, self.cols - 1):
                self.grid[i, j] = (self.prev_grid[i-1, j] + 
                                   self.prev_grid[i+1, j] + 
                                   self.prev_grid[i, j-1] + 
                                   self.prev_grid[i, j+1]) / 4
        
        # Calculate average change
        change = np.abs(self.grid - self.prev_grid)
        avg_change = np.mean(change[1:-1, 1:-1])  # Only consider interior cells
        self.avg_changes.append(avg_change)
        
        self.iteration += 1
        
        return avg_change
    
    def update_vectorized(self):
        """
        Vectorized version of update for better performance.
        """
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
    
    def run_until_convergence(self, max_iterations=10000, use_vectorized=True):
        """
        Run the simulation until convergence or max iterations.
        """
        print(f"Starting simulation with {self.rows}x{self.cols} grid...")
        print(f"Boundary temperatures: Top={self.temp_top}, Bottom={self.temp_bottom}, "
              f"Left={self.temp_left}, Right={self.temp_right}")
        print(f"Convergence threshold: {self.convergence_threshold}")
        
        while self.iteration < max_iterations:
            if use_vectorized:
                avg_change = self.update_vectorized()
            else:
                avg_change = self.update()
            
            if self.iteration % 100 == 0:
                print(f"Iteration {self.iteration}: Average change = {avg_change:.6f}")
            
            if avg_change < self.convergence_threshold:
                print(f"\nConverged after {self.iteration} iterations!")
                print(f"Final average change: {avg_change:.8f}")
                break
        else:
            print(f"\nReached maximum iterations ({max_iterations})")
            print(f"Final average change: {avg_change:.8f}")
    
    def visualize_static(self):
        """
        Create a static visualization of the current state.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Heat map
        im = ax1.imshow(self.grid, cmap='hot', interpolation='bilinear')
        ax1.set_title(f'Heat Distribution (Iteration {self.iteration})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(im, ax=ax1, label='Temperature')
        
        # Convergence plot
        if len(self.avg_changes) > 0:
            ax2.semilogy(self.avg_changes)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Average Change (log scale)')
            ax2.set_title('Convergence History')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=self.convergence_threshold, color='r', 
                       linestyle='--', label=f'Threshold = {self.convergence_threshold}')
            ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def animate_simulation(self, interval=50, save_as=None):
        """
        Create an animated visualization of the simulation.
        
        Parameters:
        -----------
        interval : int
            Milliseconds between frames
        save_as : str
            Filename to save animation (e.g., 'heat_sim.gif')
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Initial plot
        im = ax.imshow(self.grid, cmap='hot', interpolation='bilinear', 
                      vmin=min(self.temp_top, self.temp_bottom, 
                              self.temp_left, self.temp_right),
                      vmax=max(self.temp_top, self.temp_bottom, 
                              self.temp_left, self.temp_right))
        
        ax.set_title(f'Heat Distribution - Iteration 0')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        cbar = plt.colorbar(im, ax=ax, label='Temperature')
        
        def animate(frame):
            avg_change = self.update_vectorized()
            im.set_array(self.grid)
            ax.set_title(f'Heat Distribution - Iteration {self.iteration} '
                        f'(Avg Change: {avg_change:.6f})')
            
            # Stop animation if converged
            if avg_change < self.convergence_threshold:
                ani.event_source.stop()
                print(f"Animation stopped - Converged at iteration {self.iteration}")
            
            return [im]
        
        ani = animation.FuncAnimation(fig, animate, interval=interval, 
                                     blit=True, repeat=False)
        
        if save_as:
            print(f"Saving animation as {save_as}...")
            ani.save(save_as, writer='pillow')
            print("Animation saved!")
        
        plt.show()
        
        return ani


# Example usage
if __name__ == "__main__":
    # Create simulation with custom parameters
    sim = HeatEquilibriumSimulation(
        rows=50,           # Reduce for faster visualization (use 1000+ for production)
        cols=75,           # Reduce for faster visualization (use 1000+ for production)
        temp_top=100,      # Top wall temperature
        temp_bottom=0,     # Bottom wall temperature
        temp_left=50,      # Left wall temperature
        temp_right=75,     # Right wall temperature
        convergence_threshold=0.001  # Stop when avg change < this value
    )
    
    # Option 1: Run to convergence then visualize
    sim.run_until_convergence()
    sim.visualize_static()
    
    # Option 2: Animate the simulation (comment out Option 1 to use this)
    # sim.animate_simulation(interval=10)
    
    # Option 3: Large grid simulation (no animation due to performance)
    # large_sim = HeatEquilibriumSimulation(
    #     rows=1000, 
    #     cols=1500,
    #     temp_top=100,
    #     temp_bottom=0,
    #     temp_left=25,
    #     temp_right=50,
    #     convergence_threshold=0.0001
    # )
    # large_sim.run_until_convergence()
    # large_sim.visualize_static()
