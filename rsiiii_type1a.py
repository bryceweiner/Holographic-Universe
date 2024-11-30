import matplotlib.pyplot as plt
import numpy as np

# Observational data for R(Si II) ratios at various epochs
epochs = np.array([-15, -10, -5, 0, 5, 10, 15])  # Days relative to peak brightness
R_SiII_observed = np.array([0.4, 0.5, 0.7, 1.0, 1.2, 1.1, 0.9])  # Observed R(Si II) values
R_SiII_err = np.array([0.05, 0.05, 0.07, 0.1, 0.1, 0.08, 0.07])  # Uncertainty in observed values

# Lambda-CDM model predictions for R(Si II) evolution
R_SiII_model = 0.4 + 0.6 * np.exp(-((epochs) / 10)**2)  # Example Gaussian-shaped evolution
R_SiII_model_err = 0.05  # Uncertainty in Lambda-CDM model predictions

# Holographic Universe (HU) model predictions for R(Si II) evolution
R_SiII_HU_model = 0.42 + 0.58 * np.exp(-((epochs - 2) / 12)**2)  # Slightly shifted Gaussian evolution
R_SiII_HU_model_err = 0.06  # Uncertainty in HU model predictions

# Calculate upper and lower bounds for the Lambda-CDM model predictions
R_SiII_model_upper = R_SiII_model + R_SiII_model_err
R_SiII_model_lower = R_SiII_model - R_SiII_model_err

# Calculate upper and lower bounds for the HU model predictions
R_SiII_HU_model_upper = R_SiII_HU_model + R_SiII_HU_model_err
R_SiII_HU_model_lower = R_SiII_HU_model - R_SiII_HU_model_err

# Plotting
plt.figure(figsize=(10, 6))

# Lambda-CDM model curve with error band
plt.fill_between(
    epochs, R_SiII_model_lower, R_SiII_model_upper, color="blue", alpha=0.3, label="Lambda-CDM Model Uncertainty"
)
plt.plot(epochs, R_SiII_model, label="Lambda-CDM Model", color="blue")

# Holographic Universe model curve with error band
plt.fill_between(
    epochs, R_SiII_HU_model_lower, R_SiII_HU_model_upper, color="orange", alpha=0.3, label="HU Model Uncertainty"
)
plt.plot(epochs, R_SiII_HU_model, label="Holographic Universe Model", color="orange")

# Observational data with error bars
plt.errorbar(epochs, R_SiII_observed, yerr=R_SiII_err, fmt='o', color='red', label="Observational Data")

# Title and axis labels
plt.title("R(Si II) Evolution in Type Ia Supernovae")
plt.xlabel("Days Relative to Peak Brightness")
plt.ylabel("R(Si II) Ratio")

# Add legend and grid
plt.legend(loc="upper left")
plt.grid(True)

# Show plot
plt.show()
