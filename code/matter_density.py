import numpy as np
import matplotlib.pyplot as plt

# Data
surveys = ['DES Y1', 'DES Y3']
omega_m_values = [0.267, 0.298]  # Measured values
omega_m_hu = [0.268, 0.298]     # HU predictions

# Errors
measured_errors = [[0.017, 0.030], [0.007, 0.007]]  # Measured uncertainties
hu_errors = [[0.018, 0.018], [0.007, 0.007]]       # HU prediction uncertainties

# Create figure and axis
plt.figure(figsize=(8, 10))

# Plot measured values
plt.errorbar(np.arange(len(surveys)) - 0.1, omega_m_values, 
            yerr=measured_errors,
            fmt='o', 
            capsize=5,
            capthick=2,
            markersize=8,
            color='darkblue',
            ecolor='darkblue',
            label='DES Measurements')

# Plot HU predictions
plt.errorbar(np.arange(len(surveys)) + 0.1, omega_m_hu, 
            yerr=hu_errors,
            fmt='s', 
            capsize=5,
            capthick=2,
            markersize=8,
            color='green',
            ecolor='green',
            label='HU Predictions')

# Customize the plot
plt.xticks(np.arange(len(surveys)), surveys, fontsize=12, rotation=0)
plt.ylabel('Ωm', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('DES Matter Density (Ωm) Measurements\nvs HU Predictions', fontsize=12, pad=20)

# Add a horizontal line for Planck ΛCDM value
planck_omega_m = 0.315
plt.axhline(y=planck_omega_m, color='red', linestyle='--', alpha=0.7, label='Planck ΛCDM')

# Calculate current y-axis range and extend it by 50%
y_min, y_max = plt.ylim()
y_range = y_max - y_min
plt.ylim(y_min - 0.25*y_range, y_max + 0.25*y_range)

plt.legend(fontsize=10)

# Adjust layout
plt.tight_layout()

plt.show()