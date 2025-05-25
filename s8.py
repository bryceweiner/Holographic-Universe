import matplotlib.pyplot as plt
import numpy as np

# Example observational S8 parameter data (hypothetical values)
obs_S8_redshift = np.array([0.3, 0.5, 0.8, 1.2, 1.5])  # Observed redshifts
obs_S8_values = np.array([0.76, 0.75, 0.74, 0.72, 0.71])  # Observed S8 values
obs_S8_err = np.array([0.02, 0.02, 0.03, 0.03, 0.04])  # Observational uncertainties

# Lambda-CDM S8 evolution (simple extrapolation for demonstration)
S8_lambda_CDM = 0.83 - 0.1 * (1 - (1 / (1 + obs_S8_redshift)))  # Hypothetical Lambda-CDM evolution
S8_lambda_CDM_err = 0.03  # Uncertainty in Lambda-CDM predictions

# Holographic Universe S8 evolution (example adjustment)
S8_holographic = 0.8 - 0.1 * (1 - (1 / (1 + obs_S8_redshift))) * 0.95  # Modified scaling for Holographic Universe
S8_holographic_err = 0.04  # Uncertainty in Holographic Universe predictions

# Calculate upper and lower bounds for the Lambda-CDM model
S8_lambda_CDM_upper = S8_lambda_CDM + S8_lambda_CDM_err
S8_lambda_CDM_lower = S8_lambda_CDM - S8_lambda_CDM_err

# Calculate upper and lower bounds for the Holographic Universe model
S8_holographic_upper = S8_holographic + S8_holographic_err
S8_holographic_lower = S8_holographic - S8_holographic_err

# Plotting
plt.figure(figsize=(10, 6))

# Lambda-CDM model curve with error band
plt.fill_between(
    obs_S8_redshift, S8_lambda_CDM_lower, S8_lambda_CDM_upper, color="blue", alpha=0.3, label="Lambda-CDM Uncertainty"
)
plt.plot(obs_S8_redshift, S8_lambda_CDM, label="Lambda-CDM Model", color="blue")

# Holographic Universe model curve with error band
plt.fill_between(
    obs_S8_redshift, S8_holographic_lower, S8_holographic_upper, color="orange", alpha=0.3, label="Holographic Universe Uncertainty"
)
plt.plot(obs_S8_redshift, S8_holographic, label="Holographic Universe Model", color="orange")

# Observational data with error bars
plt.errorbar(obs_S8_redshift, obs_S8_values, yerr=obs_S8_err, fmt='o', color='red', label="Observational Data")

# Title and axis labels
plt.title("S8 Parameter Comparison: Lambda-CDM vs Holographic Universe")
plt.xlabel("Redshift (z)")
plt.ylabel("S8 Parameter")

# Add legend and grid
plt.legend(loc="upper right")
plt.grid(True)

# Show plot
plt.show()