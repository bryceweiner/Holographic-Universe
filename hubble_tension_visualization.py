import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import matplotlib.gridspec as gridspec
from scipy.integrate import quad
import matplotlib

# Use a more professional style
plt.style.use('seaborn-v0_8')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

# Constants
c = 3.0e5  # Speed of light in km/s
H0_early = 67.4  # Early universe H0 in km/s/Mpc (Planck)
H0_late = 73.2   # Late universe H0 in km/s/Mpc (SH0ES)
gamma = 1.89e-29  # Holographic information processing rate in s^-1
Mpc_to_cm = 3.086e24  # Conversion from Mpc to cm

# Derived parameter
gamma_over_c = gamma * Mpc_to_cm / c  # Unitless parameter for scale dependence

# Cosmological parameters
Omega_m = 0.315  # Matter density
Omega_Lambda = 0.685  # Dark energy density

# Setup the figure with a more sophisticated layout using GridSpec
fig = plt.figure(figsize=(14, 8), dpi=300)  # Reduced height since we're removing the explanation
gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 0.8])  # Changed to just 1 row instead of 2

# Main plots
ax1 = plt.subplot(gs[0, 0])  # Left: H(z) evolution 
ax2 = plt.subplot(gs[0, 1])  # Right: Scale dependence of H0

# Function for the standard ΛCDM Hubble parameter
def H_LCDM(z, H0=H0_early):
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)

# Scale-dependent effective Hubble parameter in holographic theory
def H_eff(r, H0=H0_early):
    # r is the characteristic scale in Mpc
    # Correction factor from holographic theory
    correction = 1.0 / (1.0 - gamma_over_c * r)
    return H0 * correction

# Generate redshift values
z_values = np.linspace(0, 2, 200)
H_lcdm_values = [H_LCDM(z, H0_early) for z in z_values]

# Define the observation scales and corresponding H0 values
scales = {
    'CMB': 147,           # Sound horizon at recombination
    'BAO': 100,           # Typical BAO scale
    'Strong Lensing': 50, # Typical strong lensing scale
    'Type Ia SN': 10      # Typical SN Ia distance
}

probe_colors = {
    'CMB': 'darkgreen',
    'BAO': 'darkblue',
    'Strong Lensing': 'purple',
    'Type Ia SN': 'crimson'
}

# Calculate H0 values for each probe scale
H0_values = {probe: H_eff(scale, H0_early) for probe, scale in scales.items()}

###########################################
# PLOT 1: Hubble parameter vs redshift
###########################################

# Plot ΛCDM curve
ax1.plot(z_values, H_lcdm_values, 'b-', linewidth=2.5, label='Standard $\Lambda$CDM', alpha=0.8)

# Create a more visually clear illustration of the Hubble tension
# Let's define a region of interest for the Hubble tension
z_roi = np.linspace(0, 0.15, 100)
ax1.set_xlim(-0.01, 0.18)
ax1.set_ylim(65, 76)

# Add a rectangle showing the tension region
tension_rect = Rectangle((0, H0_early), 0.15, H0_late-H0_early, 
                         facecolor='mistyrose', edgecolor='none', alpha=0.5,
                         label='Hubble Tension Region')
ax1.add_patch(tension_rect)

# Plot bands representing each measurement with errors
ax1.axhspan(H0_early - 0.5, H0_early + 0.5, color='darkgreen', alpha=0.2)
ax1.axhspan(H0_late - 1.3, H0_late + 1.3, color='crimson', alpha=0.2)

# Draw horizontal lines for the two H0 measurements
ax1.axhline(y=H0_early, color='darkgreen', linestyle='-', linewidth=2, alpha=0.7)
ax1.axhline(y=H0_late, color='crimson', linestyle='-', linewidth=2, alpha=0.7)

# Add data points for each probe
z_probes = {
    'CMB': 0.08,           # For visualization purposes only
    'BAO': 0.06,
    'Strong Lensing': 0.04,
    'Type Ia SN': 0.02
}

for probe, z in z_probes.items():
    y_val = H0_values[probe]
    ax1.scatter(z, y_val, s=100, marker='*', color=probe_colors[probe], 
                edgecolor='k', linewidth=1, zorder=10,
                label=f'{probe} ({scales[probe]} Mpc): {y_val:.1f} km/s/Mpc')

# Add a bold arrow indicating the Hubble tension
arrow_props = dict(arrowstyle='->', linewidth=2.5, color='black', shrinkA=0, shrinkB=0)
ax1.annotate('', xy=(0.125, H0_late), xytext=(0.125, H0_early), 
             arrowprops=arrow_props)
ax1.text(0.13, 70.2, 'Hubble\nTension\n~5.8 km/s/Mpc', 
         fontsize=12, fontweight='bold', ha='left', va='center')

# Add annotations for the measured values
ax1.text(0.005, H0_early-0.8, f'CMB: $H_0 = {H0_early} \pm 0.5$ km/s/Mpc', 
         color='darkgreen', fontsize=10, ha='left', fontweight='bold')
ax1.text(0.005, H0_late+0.8, f'SH0ES: $H_0 = {H0_late} \pm 1.3$ km/s/Mpc', 
         color='crimson', fontsize=10, ha='left', fontweight='bold')

# Label the axes
ax1.set_xlabel('Redshift (z)', fontsize=12)
ax1.set_ylabel('Hubble Parameter $H_0$ [km/s/Mpc]', fontsize=12)
ax1.set_title('Hubble Tension and Holographic Resolution', fontsize=14, fontweight='bold')

# Add a custom legend
handles, labels = ax1.get_legend_handles_labels()
legend = ax1.legend(handles, labels, loc='upper right', fontsize=9, framealpha=0.9)
legend.get_frame().set_linewidth(1)

###########################################
# PLOT 2: Scale dependence of H0
###########################################

# Generate a range of scales for the theory curve
r_values = np.logspace(0, 3, 1000)  # From 1 Mpc to 1000 Mpc
H0_theory = [H_eff(r, H0_early) for r in r_values]

# Plot the theoretical curve
ax2.semilogx(r_values, H0_theory, 'r-', linewidth=3, label='Holographic Theory')

# Add horizontal bands for the observed H0 values
ax2.axhspan(H0_early - 0.5, H0_early + 0.5, color='darkgreen', alpha=0.2)
ax2.axhspan(H0_late - 1.3, H0_late + 1.3, color='crimson', alpha=0.2)

# Add points for each probe
for probe, scale in scales.items():
    h0_val = H0_values[probe]
    ax2.scatter(scale, h0_val, s=150, marker='*', color=probe_colors[probe], 
                edgecolor='k', linewidth=1, zorder=10)
    
    # Add vertical lines to show scales
    ax2.axvline(x=scale, color=probe_colors[probe], linestyle='--', alpha=0.4)
    
    # Add annotations
    ax2.annotate(f"{probe}", xy=(scale, h0_val), xytext=(scale, h0_val+1.5),
                 fontsize=10, ha='center', va='center', fontweight='bold',
                 color=probe_colors[probe])

# Set axes limits
ax2.set_xlim(1, 1000)
ax2.set_ylim(H0_early-1, H0_late+3)

# Add horizontal lines for reference
ax2.axhline(y=H0_early, color='darkgreen', linestyle='-', linewidth=2, alpha=0.7)
ax2.axhline(y=H0_late, color='crimson', linestyle='-', linewidth=2, alpha=0.7)

# Label the axes
ax2.set_xlabel('Characteristic Scale [Mpc]', fontsize=12)
ax2.set_ylabel('Effective $H_0$ [km/s/Mpc]', fontsize=12)
ax2.set_title('Scale-Dependent $H_0$ in Holographic Theory', fontsize=14, fontweight='bold')

# Add gridlines
ax2.grid(True, which='both', linestyle='--', alpha=0.6)

# Add equation for holographic correction
equation_text = r"$H_0^{\mathrm{eff}}(r) = H_0^{\Lambda\mathrm{CDM}} \cdot \left[1 - \frac{\gamma r}{c}\right]^{-1}$"
ax2.text(0.5, 0.05, equation_text, transform=ax2.transAxes, 
         fontsize=12, ha='center', va='bottom', 
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# Add a main title for the entire figure
fig.suptitle('Holographic Resolution to the Hubble Tension', fontsize=16, fontweight='bold', y=0.98)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.97])
# plt.subplots_adjust(hspace=0.3)  # No need for this anymore since we don't have multiple rows

# Save the figure
plt.savefig('hubble_tension_resolution.png', dpi=300, bbox_inches='tight')
plt.show() 