import numpy as np
import matplotlib.pyplot as plt

# Multipole range (hypothetical BAO observational scales)
ell_bao = np.linspace(10, 300, 30)

# Lambda-CDM BAO data (model predictions, hypothetical)
lcdm_bao = 150 * (1 + 0.01 * np.sin(ell_bao / 50))

# Holographic Universe BAO data (model predictions, hypothetical)
holo_bao = 152 * (1 + 0.012 * np.sin(ell_bao / 55))

# Observational BAO scale data with uncertainties (hypothetical example)
obs_bao = 150 + 0.5 * np.sin(ell_bao / 60)  # Hypothetical observed BAO scale
obs_bao_uncertainty = 1.5  # +/- 1.5 Mpc uncertainty for observational data

# Observational error bands (for demonstration)
error_margin_lcdm = 2  # +/- 2 units for Lambda-CDM
error_margin_holo = 2.5  # +/- 2.5 units for Holographic Universe

# Plotting combined BAO scale comparison with observational data
fig, ax = plt.subplots(figsize=(12, 8))

# Lambda-CDM model with error bands
ax.plot(ell_bao, lcdm_bao, label='Lambda-CDM (BAO Scale)', color='blue', linestyle='-', marker='o')
ax.fill_between(
    ell_bao,
    lcdm_bao - error_margin_lcdm,
    lcdm_bao + error_margin_lcdm,
    color='blue',
    alpha=0.2,
    label='Lambda-CDM Error Band',
)

# Holographic Universe model with error bands
ax.plot(ell_bao, holo_bao, label='Holographic Universe (BAO Scale)', color='orange', linestyle='--', marker='s')
ax.fill_between(
    ell_bao,
    holo_bao - error_margin_holo,
    holo_bao + error_margin_holo,
    color='orange',
    alpha=0.2,
    label='Holographic Universe Error Band',
)

# Observational data with uncertainties
ax.errorbar(
    ell_bao,
    obs_bao,
    yerr=obs_bao_uncertainty,
    fmt='o',
    color='red',
    ecolor='red',
    elinewidth=1.5,
    capsize=3,
    label='Observed BAO Scale',
)

# Adding labels and legend
ax.set_xlabel('Multipole (ℓ)', fontsize=14)
ax.set_ylabel('BAO Scale (Mpc)', fontsize=12)
ax.set_title('BAO Scale Comparison: Models vs Observations', fontsize=16)
ax.legend(loc='upper left', fontsize=12)
ax.grid(alpha=0.3)

# Show plot
plt.tight_layout()
plt.show()
