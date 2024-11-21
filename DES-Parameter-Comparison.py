import numpy as np
import matplotlib.pyplot as plt

# Data with uncertainties
DATA = {
    "Observed": {  # DES measurements
        "S8": (0.773, 0.026, 0.020),  # (value, error_plus, error_minus)
        "omega_m": (0.267, 0.030, 0.017),
        "DM_rd": (18.92, 0.51, 0.51)
    },
    "Planck": {  # Planck ΛCDM
        "S8": (0.834, 0.016, 0.016),
        "omega_m": (0.315, 0.007, 0.007),
        "DM_rd": (20.1, 0.25, 0.25)
    },
    "HU": {  # HU predictions
        "S8": (0.781, 0.023, 0.023),
        "omega_m": (0.272, 0.020, 0.020),
        "DM_rd": (18.96, 0.38, 0.38)
    }
}

def create_parameter_comparison():
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle('Cosmological Parameter Comparison', fontsize=14, y=0.95)
    
    parameters = ['S8', 'omega_m', 'DM_rd']
    labels = ['S₈', 'Ωₘ', 'DM/rd']
    colors = {'Observed': '#2ecc71', 'Planck': '#3498db', 'HU': '#e74c3c'}
    
    for i, (param, label) in enumerate(zip(parameters, labels)):
        ax = axes[i]
        y_positions = np.arange(len(DATA))
        
        for j, (source, values) in enumerate(DATA.items()):
            value, err_plus, err_minus = values[param]
            ax.errorbar(value, j, xerr=[[err_minus], [err_plus]], 
                       fmt='o', capsize=5, capthick=2, 
                       color=colors[source], label=source,
                       markersize=8)
        
        # Add vertical line for Planck value to show reference
        ax.axvline(x=DATA['Planck'][param][0], color='gray', 
                  linestyle='--', alpha=0.3)
        
        # Customize each subplot
        ax.set_yticks(y_positions)
        ax.set_yticklabels(['DES', 'Planck', 'HU'])
        ax.set_xlabel(label)
        ax.grid(True, alpha=0.3)
        
        # Add legend only to first plot
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
        # Calculate and display tension with Planck
        obs_val, obs_err_p, obs_err_m = DATA['Observed'][param]
        planck_val, planck_err_p, planck_err_m = DATA['Planck'][param]
        tension = (planck_val - obs_val) / np.sqrt(planck_err_p**2 + obs_err_p**2)
        ax.text(1.05, 0.5, f'Tension: {tension:.1f}σ', 
                transform=ax.transAxes)
        
        # Adjust x-axis limits to show all data points clearly
        all_values = [v[param][0] for v in DATA.values()]
        all_errors = [v[param][1:] for v in DATA.values()]
        x_min = min(all_values) - max([e[0] for e in all_errors]) - 0.1
        x_max = max(all_values) + max([e[1] for e in all_errors]) + 0.1
        ax.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    return fig

# Create and show the plot
fig = create_parameter_comparison()
plt.show()