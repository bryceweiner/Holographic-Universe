import matplotlib.pyplot as plt
import numpy as np

# Example Y1 and Y3 matter density data (hypothetical values for demonstration)
obs_redshift_Y1_Y3 = np.array([0.3, 0.5, 0.7, 1.0, 1.5])  # Observed redshifts
obs_Y1_density = np.array([0.30, 0.32, 0.33, 0.35, 0.36])  # Y1 matter density
obs_Y1_err = np.array([0.02, 0.02, 0.02, 0.03, 0.03])  # Uncertainty in Y1 density
obs_Y3_density = np.array([0.31, 0.33, 0.34, 0.36, 0.37])  # Y3 matter density
obs_Y3_err = np.array([0.015, 0.015, 0.015, 0.02, 0.02])  # Uncertainty in Y3 density

# Model predictions for matter density evolution
density_lambda_CDM = 0.315 * (1 + obs_redshift_Y1_Y3)**3 / (1 + obs_redshift_Y1_Y3)**3  # Flat for Lambda-CDM
density_holographic = 0.31 * (1 + obs_redshift_Y1_Y3)**2.8 / (1 + obs_redshift_Y1_Y3)**3  # Hypothetical deviation

# Define uncertainties for models
density_lambda_CDM_err = 0.01  # Uncertainty for Lambda-CDM
density_holographic_err = 0.015  # Uncertainty for Holographic Universe

# Calculate upper and lower bounds for the models
density_lambda_CDM_upper = density_lambda_CDM + density_lambda_CDM_err
density_lambda_CDM_lower = density_lambda_CDM - density_lambda_CDM_err

density_holographic_upper = density_holographic + density_holographic_err
density_holographic_lower = density_holographic - density_holographic_err

# Plotting
plt.figure(figsize=(10, 6))

# Lambda-CDM model curve with error band
plt.fill_between(
    obs_redshift_Y1_Y3, density_lambda_CDM_lower, density_lambda_CDM_upper, color="blue", alpha=0.3, label="Lambda-CDM Uncertainty"
)
plt.plot(obs_redshift_Y1_Y3, density_lambda_CDM, label="Lambda-CDM Model", color="blue")

# Holographic Universe model curve with error band
plt.fill_between(
    obs_redshift_Y1_Y3, density_holographic_lower, density_holographic_upper, color="orange", alpha=0.3, label="Holographic Universe Uncertainty"
)
plt.plot(obs_redshift_Y1_Y3, density_holographic, label="Holographic Universe Model", color="orange")

# Observational Y1 data with error bars
plt.errorbar(obs_redshift_Y1_Y3, obs_Y1_density, yerr=obs_Y1_err, fmt='o', color='red', label="Y1 Matter Density")

# Observational Y3 data with error bars
plt.errorbar(obs_redshift_Y1_Y3, obs_Y3_density, yerr=obs_Y3_err, fmt='o', color='green', label="Y3 Matter Density")

# Title and axis labels
plt.title("Matter Density Comparison: Lambda-CDM vs Holographic Universe (Y1 & Y3 Data)")
plt.xlabel("Redshift (z)")
plt.ylabel("Matter Density (Ω_m)")

# Add legend and grid
plt.legend(loc="upper left")
plt.grid(True)

# Show plot
plt.show()
