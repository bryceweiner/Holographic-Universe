#!/usr/bin/env python3
"""
Fine Structure Constant: Universal Function Consistency Checks

This script evaluates the universal function alpha(H) that arises from
the causal diamond information-processing framework and checks that it
is numerically consistent with key values quoted in the manuscript.

alpha^-1 = (1/2)ln(pi*c^5/(hbar*G*H^2)) - ln(4*pi^2) - 1/(2*pi)

The focus here is on cosmological scales where H represents the Hubble
parameter and on illustrating how the same expression can be evaluated
at different causal-diamond geometries. Extension to particle physics
scales requires a separate mapping between interaction energy E and an
effective expansion rate, and is used only to illustrate possible
boundary conditions rather than to rederive QED running.

Reference: Weiner (2025), "The Fine Structure Constant as a Universal 
           Function of Causal Diamond Geometry"
"""

import numpy as np

# =============================================================================
# FUNDAMENTAL CONSTANTS (SI Units, CODATA 2018)
# =============================================================================
c = 2.99792458e8        # Speed of light (m/s)
hbar = 1.054571817e-34  # Reduced Planck constant (J s)
G = 6.67430e-11         # Gravitational constant (m^3 kg^-1 s^-2)
pi = np.pi
Mpc_to_m = 3.08567758e22  # Mpc to meters conversion

# Experimental reference value
ALPHA_INV_EXP = 137.035999084  # CODATA 2018


def alpha_inv_from_H(H: float) -> float:
    """
    Evaluate the universal function alpha^-1(H) used in the paper.

    H is the local Hubble parameter (or an effective expansion rate)
    characterizing the causal-diamond geometry. The same information-
    theoretic expression is evaluated at different H to explore its
    behavior across scales; this does not modify the underlying QED
    description of running, but provides geometric boundary values
    it must remain compatible with.
    """
    ln_SH = np.log((pi * c**5) / (hbar * G * H**2))
    return 0.5 * ln_SH - np.log(4 * pi**2) - 1/(2*pi)


def H_from_z(z: float, H0: float, omega_m: float = 0.315, omega_lambda: float = 0.685) -> float:
    """Hubble parameter at redshift z in LCDM cosmology."""
    return H0 * np.sqrt(omega_m * (1 + z)**3 + omega_lambda)


def run_cosmological_validation():
    """Validate at present cosmological H = H0."""
    print("=" * 80)
    print("VALIDATION AT PRESENT COSMOLOGICAL EXPANSION")
    print("=" * 80)
    
    H0_km = 67.4  # Planck 2018 central value
    H0 = H0_km * 1000 / Mpc_to_m
    
    print(f"\nHubble constant: H0 = {H0_km} km/s/Mpc = {H0:.3e} s^-1")
    
    # Evaluate universal formula
    alpha_inv = alpha_inv_from_H(H0)
    
    # Individual terms
    ln_SH = np.log((pi * c**5) / (hbar * G * H0**2))
    term1 = 0.5 * ln_SH
    term2 = -np.log(4 * pi**2)
    term3 = -1/(2*pi)
    
    print(f"\nUniversal formula: alpha^-1 = (1/2)ln(S_H) - ln(4pi^2) - 1/(2pi)")
    print(f"\nEvaluation:")
    print(f"  ln(S_H) = ln(pi*c^5/(hbar*G*H^2)) = {ln_SH:.3f}")
    print(f"  Holographic term:  (1/2)ln(S_H)   = {term1:.3f}")
    print(f"  Geometric term:    -ln(4*pi^2)    = {term2:.3f}")
    print(f"  Vacuum term:       -1/(2*pi)      = {term3:.3f}")
    print(f"  ----------------------------------------")
    print(f"  alpha^-1(H0)                      = {alpha_inv:.3f}")
    
    print(f"\nComparison with experiment:")
    print(f"  CODATA 2018: alpha^-1 = {ALPHA_INV_EXP}")
    print(f"  Difference:  {abs(alpha_inv - ALPHA_INV_EXP):.6f}")
    print(f"  Agreement:   {abs(alpha_inv - ALPHA_INV_EXP)/ALPHA_INV_EXP*100:.4f}%")
    print()


def run_cosmic_evolution():
    """Show alpha^-1 at different cosmic epochs."""
    print("=" * 80)
    print("COSMIC EVOLUTION: Same Formula at Different H(z)")
    print("=" * 80)
    
    H0_km = 67.4
    H0 = H0_km * 1000 / Mpc_to_m
    
    redshifts = [0, 0.5, 1, 2, 3, 10, 100, 1100]
    
    print(f"\n{'Redshift z':>12} | {'H(z) (s^-1)':>14} | {'ln(S_H)':>10} | {'alpha^-1':>10}")
    print("-" * 60)
    
    for z in redshifts:
        H_z = H_from_z(z, H0)
        ln_SH = np.log((pi * c**5) / (hbar * G * H_z**2))
        alpha_inv = alpha_inv_from_H(H_z)
        print(f"{z:>12} | {H_z:>14.3e} | {ln_SH:>10.2f} | {alpha_inv:>10.2f}")
    
    print("-" * 60)
    print("\nAs universe expands (z decreases), H decreases, S_H increases,")
    print("and alpha^-1 increases. The same formula evaluated at different H(z).")
    print()


def run_qed_connection():
    """Illustrate a possible connection to QED running without replacing it."""
    print("=" * 80)
    print("CONNECTION TO QED RUNNING")
    print("=" * 80)
    
    print("""
QED describes alpha as 'running' with energy E through renormalization
group equations. At low energies alpha^-1 ≈ 137, while at the Z boson
mass (91 GeV) one finds alpha^-1 ≈ 128 in the Standard Model.

Within the entropy mechanics framework, these values are interpreted
as lying between two holographically defined boundary conditions for
the universal function alpha^-1(H):

  High-energy interactions → small effective causal diamonds
  Low-energy interactions  → large effective causal diamonds (H ~ H0)

The mapping between interaction energy E and effective H is not derived
here; instead we solve for an H_eff(E) that makes alpha^-1(H_eff)
numerically coincide with the QED value at the Z mass. This should be
read as an illustrative consistency check on boundary values, not as
an attempt to rederive QED running from first principles.
""")
    
    # Calculate what H_eff would be needed for alpha^-1 = 128 (Z mass)
    alpha_inv_Z = 128.0
    # alpha^-1 = 0.5*ln(S_H) - 3.676 - 0.159
    # 128 = 0.5*ln(S_H) - 3.835
    # ln(S_H) = 2*(128 + 3.835) = 263.67
    # S_H = exp(263.67)
    # pi*c^5/(hbar*G*H^2) = exp(263.67)
    # H^2 = pi*c^5/(hbar*G*exp(263.67))
    
    numerator = pi * c**5 / (hbar * G)
    ln_SH_Z = 2 * (alpha_inv_Z + np.log(4*pi**2) + 1/(2*pi))
    H_eff_Z = np.sqrt(numerator / np.exp(ln_SH_Z))
    
    print(f"For alpha^-1 = 128 (Z boson mass):")
    print(f"  Required ln(S_H) = {ln_SH_Z:.2f}")
    print(f"  Required H_eff   = {H_eff_Z:.2e} s^-1")
    
    # Compare to present H0
    H0 = 67.4 * 1000 / Mpc_to_m
    print(f"\nCompare to present H0 = {H0:.2e} s^-1")
    print(f"  Ratio H_eff/H0 = {H_eff_Z/H0:.2e}")

    # Compare to Recombination H (z=1100)
    H_rec = H_from_z(1100, H0)
    print(f"\nCompare to Recombination H (z=1100) = {H_rec:.2e} s^-1")
    print(f"  Ratio H_eff/H_rec = {H_eff_Z/H_rec:.4f}")
    
    # Forward calculation from Z-boson timescale (heuristic entropy-mechanics check)
    E_Z = 91.2e9 * 1.602e-19  # Joules
    tau_Z = hbar / E_Z        # approx 1e-26 s
    # The following estimate uses natural-units intuition (H ~ 1/tau) as it appears
    # in entropy mechanics and related holographic arguments, and is not strictly
    # dimensionally consistent in SI units. It is included only to illustrate that
    # a naive identification of a local process timescale with an effective Hubble-
    # like rate does not reproduce alpha^-1 ≈ 128 within this simple scaling.
    H_local_Z = c / tau_Z     # heuristic "local" rate, not a physical Hubble parameter
    
    print(f"\n--- HEURISTIC LOCAL TIMESCALE CHECK ---")
    print(f"Local timescale for Z-boson (tau_Z ~ hbar/E_Z): {tau_Z:.2e} s")
    print(f"Naive effective rate estimate H_eff ~ c/tau_Z: {H_local_Z:.2e} s^-1")
    alpha_local_Z = alpha_inv_from_H(H_local_Z)
    print(f"Alpha^-1 evaluated at this heuristic H_eff: {alpha_local_Z:.2f}")
    print(f"  (This difference from 128 highlights that a complete mapping")
    print(f"   between interaction energy E and effective H requires more")
    print(f"   than a simple inverse-timescale identification.)")
    
    print(f"\nComparing inferred H_eff for alpha^-1=128 to cosmic values:")
    print(f"  Present H0                   = {H0:.2e} s^-1")
    print(f"  Recombination H(z=1100)      = {H_rec:.2e} s^-1")
    print(f"  Inferred H_eff(alpha^-1=128) = {H_eff_Z:.2e} s^-1")
    print(f"  H_eff/H0                     = {H_eff_Z/H0:.2e}")
    print(f"  H_eff/H_rec                  = {H_eff_Z/H_rec:.4f}")
    print("\nWithin order-unity factors, the effective H_eff associated with")
    print("alpha^-1 ≈ 128 lies between present and recombination values.")
    print("This supports viewing the universal function as supplying boundary")
    print("conditions compatible with QED running rather than as an alternative.")


def run_sensitivity_analysis():
    """Test alpha^-1 predictions across H0 range."""
    print("=" * 80)
    print("SENSITIVITY TO HUBBLE PARAMETER")
    print("=" * 80)
    
    H0_values = {
        'Planck 2018 (low)': 66.9,
        'Planck 2018 (central)': 67.4,
        'Planck 2018 (high)': 67.9,
        'ACT DR6': 68.3,
        'Intermediate': 70.0,
        'SH0ES (central)': 73.0,
    }
    
    print(f"\n{'Source':<25} | {'H0 (km/s/Mpc)':>14} | {'alpha^-1':>12} | {'Error (%)':>10}")
    print("-" * 70)
    
    for source, H0_km in H0_values.items():
        H0 = H0_km * 1000 / Mpc_to_m
        alpha_inv = alpha_inv_from_H(H0)
        error_pct = abs(alpha_inv - ALPHA_INV_EXP) / ALPHA_INV_EXP * 100
        print(f"{source:<25} | {H0_km:>14.1f} | {alpha_inv:>12.3f} | {error_pct:>10.4f}")
    
    print("\nWithin this framework, the best numerical agreement with the")
    print("experimental alpha occurs for Planck 2018 H0 values, while a")
    print("higher SH0ES value corresponds to a slightly stronger predicted")
    print("electromagnetic coupling. This is a mapping between datasets,")
    print("not an independent determination of H0.")
    print()


def run_atomic_constancy():
    """Explain why atomic measurements show constant alpha in this framework."""
    print("=" * 80)
    print("WHY ATOMIC MEASUREMENTS SHOW CONSTANT ALPHA")
    print("=" * 80)
    
    print("""
At any redshift z, atomic spectroscopy measures transitions determined by
electron binding energies and the local electromagnetic coupling at atomic
scales. The cosmic expansion H(z) affects the large-scale causal-diamond
structure, but atomic transitions probe local physics at fixed characteristic
length and time scales.

Key point: In this picture atomic processes do not respond directly to H(z);
they are governed by the same local binding-scale geometry at all redshifts,
so they always probe the same branch of the universal function alpha^-1(H).
""")
    
    H0 = 67.4 * 1000 / Mpc_to_m
    
    print("Cosmic H(z) evolution:")
    print(f"{'z':>8} | {'H(z)/H0':>10} | {'Cosmic alpha^-1':>15}")
    print("-" * 40)
    
    for z in [0, 1, 2, 3]:
        H_z = H_from_z(z, H0)
        alpha_inv = alpha_inv_from_H(H_z)
        print(f"{z:>8} | {H_z/H0:>10.2f} | {alpha_inv:>15.2f}")
    
    print("""
While cosmic H(z) varies significantly with redshift, atomic measurements
always probe the same local physics. In the entropy mechanics framework the
universal formula is evaluated at an effective H appropriate to atomic scales
that does not track H(z), so alpha^-1 ≈ 137 is expected to remain constant.

This provides a causal-diamond interpretation of null results from quasar
absorption spectroscopy without altering the local QED description of the
transitions themselves.
""")


def run_numerical_stability():
    """Test numerical stability under parameter variations."""
    print("=" * 80)
    print("NUMERICAL STABILITY ANALYSIS")
    print("=" * 80)
    
    H0 = 67.4 * 1000 / Mpc_to_m
    base_val = alpha_inv_from_H(H0)
    
    print(f"\nBase value: alpha^-1 = {base_val:.6f}")
    print()
    
    def calc_with_params(c_val, hbar_val, G_val, H_val):
        ln_SH = np.log((pi * c_val**5) / (hbar_val * G_val * H_val**2))
        return 0.5 * ln_SH - np.log(4 * pi**2) - 1/(2*pi)
    
    variations = [
        ("c + 0.1%", calc_with_params(c*1.001, hbar, G, H0)),
        ("c - 0.1%", calc_with_params(c*0.999, hbar, G, H0)),
        ("hbar + 0.1%", calc_with_params(c, hbar*1.001, G, H0)),
        ("hbar - 0.1%", calc_with_params(c, hbar*0.999, G, H0)),
        ("G + 0.1%", calc_with_params(c, hbar, G*1.001, H0)),
        ("G - 0.1%", calc_with_params(c, hbar, G*0.999, H0)),
        ("H0 + 0.1%", calc_with_params(c, hbar, G, H0*1.001)),
        ("H0 - 0.1%", calc_with_params(c, hbar, G, H0*0.999)),
    ]
    
    print(f"{'Parameter':<15} | {'alpha^-1':>12} | {'Delta':>10} | {'Change (%)':>12}")
    print("-" * 55)
    
    for name, val in variations:
        delta = val - base_val
        pct = delta / base_val * 100
        print(f"{name:<15} | {val:>12.6f} | {delta:>+10.6f} | {pct:>+12.6f}")
    
    print("\nFormula shows smooth, continuous dependence on all parameters.")
    print()


def main():
    """Run the suite of numerical consistency checks."""
    print("\n" + "=" * 80)
    print("FINE STRUCTURE CONSTANT AS UNIVERSAL FUNCTION")
    print("Information-Theoretic Consistency Checks")
    print("=" * 80 + "\n")
    
    run_cosmological_validation()
    run_cosmic_evolution()
    run_qed_connection()
    run_atomic_constancy()
    run_sensitivity_analysis()
    run_numerical_stability()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The universal formula:

  alpha^-1 = (1/2)ln(pi*c^5/(hbar*G*H^2)) - ln(4*pi^2) - 1/(2*pi)

At H = H0 (present cosmological expansion):
  alpha^-1 = 137.032  [0.003% level agreement with experiment]

At H(z=1100) (recombination):
  alpha^-1 = 127.1    [close to the QED value at the Z mass]

The same expression is evaluated at different effective H to explore
how causal-diamond information geometry can supply boundary values
for electromagnetic coupling across scales. These consistency checks
are intended to complement, not replace, the Standard Model treatment
of renormalization-group running.
""")


if __name__ == "__main__":
    main()
