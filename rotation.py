import matplotlib.pyplot as plt
import numpy as np
# Re-implementing the proper N-state rotation calculation dynamically

def calculate_n_state_rotation(N):
    """
    Calculate rotation angles for N-state universes based on recursive principles.
    The calculation follows the principles outlined for bifurcating symmetry states.
    """
    # Base angle for N-state universes (360 degrees divided by number of bifurcations)
    base_angle = 360 / (2 ** (N - 1))  # Recursive division for symmetry bifurcations

    # Calculate angles for each universe
    rotations = [(i * base_angle / 2) % 360 for i in range(2 ** N)]
    return rotations

# Calculate the N=4 rotation angles dynamically
N = 2
angles_primary = calculate_n_state_rotation(N)  # Primary continuum angles
angles_secondary = [(angle + 90) % 360 for angle in angles_primary]  # Secondary continuum angles

# Convert angles to radians for plotting
radians_primary = np.radians(angles_primary)
radians_secondary = np.radians(angles_secondary)

# Define radii for universes (equal for primary and secondary continua for central axis alignment)
radii = [1] * len(angles_primary)

# Define labels for universes
universes = [f"Universe {i + 1}" for i in range(len(angles_primary))]

# Define a unique color for each universe
colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow'] * (len(angles_primary) // 8 + 1)

# Create a polar plot to visualize the matter and antimatter continua for each universe with distinct colors
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
ax.set_theta_zero_location('N')  # Set 0 degrees to the top
ax.set_theta_direction(-1)  # Set clockwise direction

# Plot matter and antimatter continua extending symmetrically across the axis for each universe
for i, (rad1, rad2) in enumerate(zip(radians_primary, radians_secondary)):
    # Plot matter continuum as a 5px wide dashed line in a unique color extending symmetrically
    ax.plot([rad1, rad1 + np.pi], [0, radii[i]], color=colors[i], linestyle='--', lw=5, label=f"{universes[i]} - Matter" if i < 8 else None)
    ax.plot([rad1 + np.pi, rad1], [0, radii[i]], color=colors[i], linestyle='--', lw=5)  # Symmetry
    
    # Plot antimatter continuum as a shaded band extending both directions in the same unique color
    ax.fill_betweenx([0, radii[i]], rad2 - 0.05, rad2 + 0.05, color=colors[i], alpha=0.5, label=f"{universes[i]} - Antimatter" if i < 8 else None)
    ax.fill_betweenx([0, radii[i]], rad2 - 0.05 + np.pi, rad2 + 0.05 + np.pi, color=colors[i], alpha=0.5)

    # Mark the central axis for each universe
    ax.scatter(0, 0, s=50, color='black', label="Central Axis" if i == 0 else None)

# Customize the plot
ax.set_rmax(1.5)
ax.set_rticks([])  # Remove radial ticks for clarity
ax.set_title(f"Universes and Their Continua Orientations (N={N})", va='bottom', fontsize=14)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.show()

# Display calculated angles for verification
angles_primary
