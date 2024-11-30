import matplotlib.pyplot as plt
import numpy as np

# Constants and parameters
H0_lambda_CDM = 67.4  # Lambda-CDM Hubble constant (km/s/Mpc)
H0_holographic = 71.0  # Holographic Universe Hubble constant (km/s/Mpc)
Omega_m = 0.315  # Matter density parameter (Planck 2018)
Omega_Lambda = 1 - Omega_m  # Dark energy density (flat universe assumption)
Omega_holo = 1 - Omega_m  # Energy density for the Holographic Universe
n = 2  # Exponent for holographic corrections

# Define redshift range for the late Universe
z_late = np.linspace(0.24, 2, 200)

# Lambda-CDM H(z) calculation
H_lambda_CDM_late = H0_lambda_CDM * np.sqrt(Omega_m * (1 + z_late)**3 + Omega_Lambda)

# Holographic Universe H(z) calculation
H_holographic_late = H0_holographic * np.sqrt(Omega_m * (1 + z_late)**3 + Omega_holo * (1 + z_late)**n)

# Observational data for the late Universe (example data)
obs_redshift = np.array([0.3, 0.5, 0.7, 1.0, 1.5])  # Observed redshifts
obs_Hubble = np.array([78.0, 88.0, 100.0, 120.0, 140.0])  # Observed H(z) in km/s/Mpc
obs_Hubble_err = np.array([5.0, 4.5, 6.0, 7.0, 8.0])  # Observational uncertainties

# Define y-axis range large enough to show both models and data clearly
y_min_late = min(H_lambda_CDM_late.min(), H_holographic_late.min()) - 10
y_max_late = max(H_lambda_CDM_late.max(), H_holographic_late.max()) + 10

# Plotting
plt.figure(figsize=(10, 6))

# Lambda-CDM model curve
plt.plot(z_late, H_lambda_CDM_late, label="Lambda-CDM Model", color="blue")

# Holographic Universe model curve
plt.plot(z_late, H_holographic_late, label="Holographic Universe Model", color="orange")

# Observational data with error bars
plt.errorbar(obs_redshift, obs_Hubble, yerr=obs_Hubble_err, fmt='o', color='red', label="Observational Data")

# Title and axis labels
plt.title("Hubble Parameter (H(z)) in the Late Universe: Lambda-CDM vs Holographic Universe")
plt.xlabel("Redshift (z)")
plt.ylabel("Hubble Parameter H(z) [km/s/Mpc]")

# Adjust y-axis limits
plt.ylim(y_min_late, y_max_late)

# Add legend and grid
plt.legend(loc="upper left")
plt.grid(True)

# Show plot
plt.show()
