import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import binned_statistic

def calculate_gamma(T, S):
    """Calculate gamma value based on temperature and entropy"""
    k = 1.380649e-23  # Boltzmann constant
    hbar = 1.054571817e-34  # Reduced Planck constant
    return 2 * np.pi * k * T / hbar * np.log(S)

def generate_emode_data():
    """Generate simulated E-mode power spectrum data based on ACTPol"""
    # Multipole range
    ell = np.linspace(100, 9000, 1000)
    
    # Base E-mode spectrum (simplified model)
    baseline = 2e-5 * (ell/1500)**(-2.5)
    
    # Add transitions at key points
    transitions = [1750, 3250, 4500]
    phases = np.zeros_like(ell)
    
    for l_trans in transitions:
        # Add phase shift
        mask = ell > l_trans
        phases[mask] += np.pi/4 * np.exp(-(ell[mask]-l_trans)/500)
    
    # Add some noise
    noise = np.random.normal(0, 0.1 * baseline, len(ell))
    
    return ell, baseline, phases, noise

def plot_phase_shifts():
    """Create visualization of E-mode phase shifts and gamma correlation"""
    ell, power, phases, noise = generate_emode_data()
    
    # Calculate example gamma values
    T = np.linspace(1e6, 1e8, len(ell))  # Temperature range
    S = np.exp(ell/1000)  # Example entropy scaling
    gamma = calculate_gamma(T, S)
    
    # Create figure with multiple panels
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot 1: E-mode power spectrum
    ax1.plot(ell, power + noise, 'k-', alpha=0.5, label='E-mode power')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Multipole l')
    ax1.set_ylabel('E-mode power')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Phase shifts
    ax2.plot(ell, phases, 'b-', label='Phase angle')
    ax2.set_xlabel('Multipole l')
    ax2.set_ylabel('Phase shift (radians)')
    ax2.grid(True)
    ax2.legend()
    
    # Plot 3: Gamma correlation
    # Normalize gamma for comparison
    gamma_norm = (gamma - gamma.min()) / (gamma.max() - gamma.min())
    phase_norm = (phases - phases.min()) / (phases.max() - phases.min())
    
    ax3.plot(ell, gamma_norm, 'r-', label='Normalized γ(τ)')
    ax3.plot(ell, phase_norm, 'b--', alpha=0.5, label='Normalized phase')
    ax3.set_xlabel('Multipole l')
    ax3.set_ylabel('Normalized amplitude')
    ax3.grid(True)
    ax3.legend()
    
    # Annotate transitions
    for l_trans in [1750, 3250, 4500]:
        for ax in [ax1, ax2, ax3]:
            ax.axvline(l_trans, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_phase_shifts()