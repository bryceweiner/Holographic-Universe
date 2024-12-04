import matplotlib.pyplot as plt
import numpy as np

# Constants and parameters
H0_lambda_CDM = 67.4  # Lambda-CDM Hubble constant (km/s/Mpc)
H0_holographic = 71.0  # Holographic Universe Hubble constant (km/s/Mpc)
H0_observed = 73.0  # Observed local Hubble constant (km/s/Mpc)
H0_observed_err = 1.5  # Observational uncertainty (km/s/Mpc)
Omega_m = 0.315  # Matter density parameter (Planck 2018)
Omega_Lambda = 1 - Omega_m  # Dark energy density (flat universe assumption)
Omega_holo = 1 - Omega_m  # Energy density for the Holographic Universe
n = 2  # Exponent for holographic corrections

# Define restricted redshift range for plotting
z_restricted = np.linspace(0, 0.24, 100)

# Lambda-CDM H(z) calculation
H_lambda_CDM_restricted = H0_lambda_CDM * np.sqrt(Omega_m * (1 + z_restricted)**3 + Omega_Lambda)

# Holographic Universe H(z) calculation
H_holographic_restricted = H0_holographic * np.sqrt(Omega_m * (1 + z_restricted)**3 + Omega_holo * (1 + z_restricted)**n)

# Plotting
plt.figure(figsize=(10, 6))

# Lambda-CDM model curve
plt.plot(z_restricted, H_lambda_CDM_restricted, label="Lambda-CDM Model", color="blue")

# Holographic Universe model curve
plt.plot(z_restricted, H_holographic_restricted, label="Holographic Universe Model", color="orange")

# Observational H0 as a horizontal line
plt.axhline(y=H0_observed, color='green', linestyle='--', label=f"Observed H0 = {H0_observed} ± {H0_observed_err}")

# Fill uncertainty band for the observed H0
plt.fill_between(z_restricted, H0_observed - H0_observed_err, H0_observed + H0_observed_err, color='green', alpha=0.3, label="Observation Uncertainty")

# Title and axis labels
plt.title("Hubble Parameter (H(z)) Comparison: Lambda-CDM vs Holographic Universe")
plt.xlabel("Redshift (z)")
plt.ylabel("Hubble Parameter H(z) [km/s/Mpc]")

# Legend and grid
plt.legend(loc="upper left")
plt.grid(True)

# Show plot
plt.show()
