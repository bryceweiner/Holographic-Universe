import matplotlib.pyplot as plt
import numpy as np

# Define constants and data for the plot
gamma = 0.01  # Information processing parameter (example value)
H = 67.4  # Hubble parameter (km/s/Mpc)
C0 = 1e-2  # Normalization constant for power spectrum

# Define angular multipole range
ell = np.linspace(100, 10000, 1000)  # Range of multipoles

# Calculate E-mode power spectrum (with exponential suppression)
C_ell_EE = C0 * ell**-2 * np.exp(-gamma * ell / H)

# Observed phase transitions and uncertainties
observed_transitions = [1750, 3250, 4500]  # Observed multipole centers
observed_errors = [35, 65, 90]  # Observational uncertainties for transitions

# Add Gaussian peaks for phase transitions to the E-mode power spectrum
for center, error in zip(observed_transitions, observed_errors):
    C_ell_EE += 0.005 * np.exp(-((ell - center) / error)**2)

# Define phase shift spikes and gamma(t) curve
phase_shift = np.zeros_like(ell)
gamma_t = np.zeros_like(ell)

for center, error in zip(observed_transitions, observed_errors):
    phase_shift += np.exp(-((ell - center) / error)**2)  # Phase shift spikes
    gamma_t += np.maximum(0, 1 - np.abs((ell - center) / (2 * error)))  # Linear peaks for gamma(t)

# Normalize for visualization
phase_shift = phase_shift / phase_shift.max()
gamma_t = gamma_t / gamma_t.max()

# Define uncertainties for gamma(t) and phase shifts
gamma_t_err = 0.05  # Uncertainty for gamma(t)
phase_shift_err = 0.1  # Uncertainty for phase shifts

# Calculate error bands for gamma(t)
gamma_t_upper = gamma_t + gamma_t_err
gamma_t_lower = gamma_t - gamma_t_err

# Calculate uncertainty bands for phase shifts
phase_shift_upper = phase_shift + phase_shift_err
phase_shift_lower = phase_shift - phase_shift_err

# Plotting combined panels
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Top panel: E-mode power spectrum
axs[0].plot(ell, C_ell_EE, color='black', label="E-mode Power Spectrum")
axs[0].set_yscale('log')
axs[0].set_ylabel("E-mode Power Spectrum ($C_{\ell}^{EE}$)")
axs[0].legend()

# Middle panel: Phase shifts with uncertainty
axs[1].fill_between(ell, phase_shift_lower, phase_shift_upper, color='blue', alpha=0.3, label="Phase Shift Uncertainty")
axs[1].plot(ell, phase_shift, color='blue', label="Phase Shift")
axs[1].set_ylabel("Phase Shift (radians)")
axs[1].legend()

# Bottom panel: Gamma(t) with error bands and normalized phase shifts
axs[2].fill_between(ell, gamma_t_lower, gamma_t_upper, color='red', alpha=0.3, label="Gamma(t) Error Band")
axs[2].plot(ell, gamma_t, color='red', label="Normalized $\gamma(t)$")
axs[2].plot(ell, phase_shift, color='blue', linestyle='--', label="Normalized Phase Shift")
axs[2].set_xlabel("Angular Multipole (ℓ)")
axs[2].set_ylabel("Normalized Amplitude")
axs[2].legend()

# Finalize and display
plt.tight_layout()
plt.show()
