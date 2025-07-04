"""
QTEP-Mediated Projection Theory
===============================

Mathematical framework for understanding how the Quantum-Thermodynamic Entropy 
Partition (QTEP) mediates dimensional reduction from E8×E8 to observable 3D space.
"""

import numpy as np
from scipy.linalg import expm, logm
from scipy.special import gamma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Dict
import sympy as sp


class QTEPProjectionFramework:
    """Mathematical framework for QTEP-mediated projections."""
    
    def __init__(self):
        self.qtep_ratio = 2.257
        self.qtep_angle = 35.3  # degrees
        self.e8_dimension = 248
        self.heterotic_dimension = 496  # E8×E8
        
        # Coherent and decoherent entropy components
        self.s_coh = np.log(2)  # ≈ 0.693
        self.s_decoh = np.log(2) - 1  # ≈ -0.307
        
        # Information processing rate
        self.gamma_rate = 1.89e-29  # s^-1
        
    def qtep_projection_operator(self, n_dim: int) -> np.ndarray:
        """
        Construct the QTEP projection operator for n-dimensional space.
        
        The operator encodes the entropy partition constraints during
        dimensional reduction.
        """
        # Create base projection matrix
        P = np.zeros((n_dim, n_dim))
        
        # QTEP constraint matrix
        theta_qtep = np.radians(self.qtep_angle)
        
        # Coherent subspace projector
        P_coh = np.zeros((n_dim, n_dim))
        coh_dim = int(n_dim * self.qtep_ratio / (1 + self.qtep_ratio))
        P_coh[:coh_dim, :coh_dim] = np.eye(coh_dim)
        
        # Decoherent subspace projector
        P_decoh = np.eye(n_dim) - P_coh
        
        # QTEP-mediated coupling
        coupling = np.zeros((n_dim, n_dim))
        for i in range(min(coh_dim, n_dim - coh_dim)):
            coupling[i, coh_dim + i] = np.sin(theta_qtep)
            coupling[coh_dim + i, i] = np.sin(theta_qtep)
        
        # Full QTEP operator
        P = (self.s_coh * P_coh + 
             self.s_decoh * P_decoh + 
             coupling / self.qtep_ratio)
        
        # Normalize
        P = P / np.linalg.norm(P)
        
        return P
    
    def heterotic_projection_cascade(self) -> Dict[str, np.ndarray]:
        """
        Calculate the cascade of projections from E8×E8 to 3D.
        
        Returns projection operators at each level.
        """
        projections = {}
        
        # Level 1: E8×E8 → 26D (bosonic string critical dimension)
        P_496_26 = self.construct_heterotic_projector(496, 26)
        projections['496_to_26'] = P_496_26
        
        # Level 2: 26D → 10D (superstring critical dimension)
        P_26_10 = self.construct_heterotic_projector(26, 10)
        projections['26_to_10'] = P_26_10
        
        # Level 3: 10D → 4D (spacetime)
        P_10_4 = self.construct_heterotic_projector(10, 4)
        projections['10_to_4'] = P_10_4
        
        # Level 4: 4D → 3D (spatial dimensions)
        P_4_3 = self.construct_heterotic_projector(4, 3)
        projections['4_to_3'] = P_4_3
        
        # Composite projection
        projections['composite'] = P_4_3 @ P_10_4 @ P_26_10 @ P_496_26
        
        return projections
    
    def construct_heterotic_projector(self, dim_in: int, dim_out: int) -> np.ndarray:
        """
        Construct projection operator with QTEP constraints.
        """
        # Random projection matrix (placeholder for E8 structure)
        np.random.seed(42)  # For reproducibility
        P_random = np.random.randn(dim_out, dim_in)
        
        # Apply QTEP constraint
        qtep_factor = self.qtep_ratio / (1 + self.qtep_ratio)
        
        # Decompose into coherent and decoherent parts
        U, S, Vt = np.linalg.svd(P_random, full_matrices=False)
        
        # Modify singular values according to QTEP
        n_coherent = int(len(S) * qtep_factor)
        S[:n_coherent] *= self.s_coh
        S[n_coherent:] *= abs(self.s_decoh)
        
        # Reconstruct
        P = U @ np.diag(S) @ Vt[:dim_out, :]
        
        # Normalize
        P = P / np.linalg.norm(P)
        
        return P
    
    def angular_projection_formula(self, theta_e8: float) -> float:
        """
        Calculate how E8 angles project to 3D under QTEP constraints.
        
        Args:
            theta_e8: Angle in E8 space (radians)
            
        Returns:
            Projected angle in 3D (radians)
        """
        # QTEP modulation function
        qtep_mod = lambda x: (1 + self.qtep_ratio * np.sin(x)) / (1 + self.qtep_ratio)
        
        # Information loss factor
        info_loss = np.exp(-self.gamma_rate * 1e29)  # Normalized
        
        # Projection formula
        theta_3d = theta_e8 * qtep_mod(theta_e8) * info_loss
        
        # Add interference terms for second-order effects
        theta_qtep_rad = np.radians(self.qtep_angle)
        interference = np.sin(theta_e8 - theta_qtep_rad) * np.sin(theta_e8 + theta_qtep_rad)
        
        theta_3d += 0.1 * interference  # Small correction
        
        return theta_3d
    
    def information_channel_capacity(self, theta: float) -> float:
        """
        Calculate information channel capacity for a given angle.
        
        Based on holographic constraints and QTEP partition.
        """
        theta_rad = np.radians(theta)
        
        # Base capacity from geometric factors
        C_geom = np.sin(theta_rad) ** 2
        
        # QTEP enhancement/suppression
        theta_qtep_rad = np.radians(self.qtep_angle)
        qtep_factor = 1 + self.qtep_ratio * np.cos(theta_rad - theta_qtep_rad)
        
        # Information processing constraint
        bandwidth_factor = 1 - np.exp(-abs(theta - self.qtep_angle) / self.qtep_angle)
        
        # Total capacity
        C_total = C_geom * qtep_factor * bandwidth_factor
        
        return C_total
    
    def derive_selection_rules(self) -> Dict[str, callable]:
        """
        Derive selection rules for allowed angular combinations.
        """
        rules = {}
        
        # Rule 1: QTEP-mediated transitions
        def qtep_transition_allowed(angle1: float, angle2: float) -> bool:
            """Transitions involving QTEP angle are enhanced."""
            return (abs(angle1 - self.qtep_angle) < 1.0 or 
                   abs(angle2 - self.qtep_angle) < 1.0)
        
        rules['qtep_mediated'] = qtep_transition_allowed
        
        # Rule 2: Information conservation
        def info_conserved(angle1: float, angle2: float) -> bool:
            """Combined information must not exceed holographic bound."""
            C1 = self.information_channel_capacity(angle1)
            C2 = self.information_channel_capacity(angle2)
            C_combined = self.information_channel_capacity((angle1 + angle2) % 180)
            return C_combined <= C1 + C2
        
        rules['info_conservation'] = info_conserved
        
        # Rule 3: Crystallographic compatibility
        def crystallographic_compatible(angle1: float, angle2: float) -> bool:
            """Angles must be compatible with E8 crystallographic structure."""
            # Check if combination preserves some E8 symmetry
            combination = (angle1 + angle2) % 180
            e8_angles = [30, 45, 60, 90, 120, 135, 150]
            return any(abs(combination - e8_angle) < 5.0 for e8_angle in e8_angles)
        
        rules['crystallographic'] = crystallographic_compatible
        
        # Rule 4: Entropy balance
        def entropy_balanced(angle1: float, angle2: float) -> bool:
            """Entropy production must balance between coherent/decoherent."""
            # Approximate entropy production
            S_prod = np.sin(np.radians(angle1)) * np.sin(np.radians(angle2))
            S_threshold = abs(self.s_coh * self.s_decoh)
            return S_prod > S_threshold
        
        rules['entropy_balance'] = entropy_balanced
        
        return rules
    
    def plot_projection_flow(self):
        """Visualize the projection flow from E8×E8 to 3D."""
        fig = plt.figure(figsize=(12, 10))
        
        # Create 3D subplot for projection visualization
        ax = fig.add_subplot(111, projection='3d')
        
        # Generate points in high-dimensional space (simplified)
        n_points = 1000
        
        # E8×E8 points (projected to 3D for visualization)
        theta = np.random.uniform(0, 2*np.pi, n_points)
        phi = np.random.uniform(0, np.pi, n_points)
        
        # Apply cascade of projections
        cascade = self.heterotic_projection_cascade()
        
        # Visualize different stages
        colors = ['red', 'orange', 'yellow', 'green', 'blue']
        labels = ['E8×E8 (496D)', '26D', '10D', '4D', '3D']
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            # Simulate projection effects
            r = 10 - 2*i  # Decreasing radius
            noise = 0.1 * i  # Increasing decoherence
            
            x = r * np.sin(phi) * np.cos(theta) + np.random.normal(0, noise, n_points)
            y = r * np.sin(phi) * np.sin(theta) + np.random.normal(0, noise, n_points)
            z = r * np.cos(phi) + np.random.normal(0, noise, n_points)
            
            # Apply QTEP modulation
            qtep_mod = 1 + 0.2 * np.sin(self.qtep_ratio * theta)
            x *= qtep_mod
            y *= qtep_mod
            z *= qtep_mod
            
            ax.scatter(x[::20], y[::20], z[::20], c=color, label=label, 
                      alpha=0.6, s=30-5*i)
        
        # Add QTEP angle indicator
        qtep_rad = np.radians(self.qtep_angle)
        ax.plot([0, 5*np.cos(qtep_rad)], [0, 5*np.sin(qtep_rad)], [0, 0], 
                'k-', linewidth=3, label=f'QTEP angle ({self.qtep_angle}°)')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('QTEP-Mediated Projection Cascade\nE8×E8 → 3D', fontsize=14)
        ax.legend()
        
        return fig
    
    def analyze_qtep_angle_origin(self) -> Dict[str, float]:
        """Detailed analysis of possible QTEP angle origins."""
        origins = {}
        
        # Symbolic computation for exact values
        sp.init_printing()
        qtep = sp.Rational(2257, 1000)  # QTEP ratio as exact fraction
        
        # Various mathematical origins
        origins['atan(1/QTEP)'] = float(sp.atan(1/qtep) * 180/sp.pi)
        origins['asin(1/sqrt(1+QTEP²))'] = float(sp.asin(1/sp.sqrt(1 + qtep**2)) * 180/sp.pi)
        
        # E8-related angles
        origins['π/5 radians'] = 36.0  # Pentagon angle
        origins['π/8 + π/24'] = 180/8 + 180/24  # E8 + E8 correction
        
        # Information theory related
        h_binary = -0.5 * np.log2(0.5) - 0.5 * np.log2(0.5)  # Binary entropy
        origins['entropy_angle'] = np.degrees(np.arcsin(h_binary))
        
        # Dimensional reduction factor
        dim_factor = np.sqrt(3/496)  # 3D/E8×E8
        origins['dim_reduction'] = np.degrees(np.arcsin(dim_factor))
        
        # Golden ratio family
        phi = (1 + np.sqrt(5))/2
        origins['golden_arctan'] = np.degrees(np.arctan(1/phi))
        origins['golden_arcsin'] = np.degrees(np.arcsin(1/phi))
        
        # Holographic bound related
        origins['holographic'] = np.degrees(np.arctan(np.sqrt(np.pi/np.e)))
        
        return origins


def generate_theoretical_framework_plots():
    """Generate all plots for the theoretical framework."""
    framework = QTEPProjectionFramework()
    
    # Plot 1: QTEP angle origin analysis
    fig1, ax = plt.subplots(figsize=(10, 8))
    origins = framework.analyze_qtep_angle_origin()
    
    angles = list(origins.values())
    labels = list(origins.keys())
    errors = [abs(a - 35.3) for a in angles]
    
    # Sort by error
    sorted_indices = np.argsort(errors)
    angles = [angles[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    errors = [errors[i] for i in sorted_indices]
    
    colors = ['green' if e < 2.0 else 'blue' if e < 5.0 else 'gray' for e in errors]
    bars = ax.barh(range(len(angles)), angles, color=colors)
    
    # Add error annotations
    for i, (angle, error) in enumerate(zip(angles, errors)):
        ax.text(angle + 0.5, i, f'{angle:.2f}° (Δ={error:.2f}°)', 
                va='center', fontsize=9)
    
    ax.axvline(35.3, color='red', linestyle='--', linewidth=2, label='QTEP angle (35.3°)')
    ax.set_yticks(range(len(angles)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Angle (degrees)')
    ax.set_title('Potential Mathematical Origins of the QTEP Angle')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qtep_angle_origins.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('qtep_angle_origins.png', dpi=300, bbox_inches='tight')
    
    # Plot 2: Information channel capacity
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    angles = np.linspace(0, 180, 361)
    capacities = [framework.information_channel_capacity(a) for a in angles]
    
    ax1.plot(angles, capacities, 'b-', linewidth=2)
    ax1.axvline(35.3, color='red', linestyle='--', label='QTEP angle')
    
    # Mark special angles
    special_angles = [30, 45, 60, 90, 120, 135, 150, 35.3, 48.2, 70.5, 85, 105, 165]
    for angle in special_angles:
        cap = framework.information_channel_capacity(angle)
        ax1.plot(angle, cap, 'ro' if angle in [85, 105, 165] else 'go', markersize=8)
    
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Channel Capacity')
    ax1.set_title('Information Channel Capacity vs Angle')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2b: Selection rules visualization
    angles1 = np.arange(0, 180, 5)
    angles2 = np.arange(0, 180, 5)
    
    rules = framework.derive_selection_rules()
    
    # Check which combinations satisfy all rules
    allowed = np.zeros((len(angles1), len(angles2)))
    
    for i, a1 in enumerate(angles1):
        for j, a2 in enumerate(angles2):
            if all(rule(a1, a2) for rule in rules.values()):
                allowed[i, j] = 1
    
    im = ax2.imshow(allowed, cmap='RdYlGn', aspect='equal', origin='lower', 
                   extent=[0, 180, 0, 180])
    ax2.set_xlabel('Angle 2 (degrees)')
    ax2.set_ylabel('Angle 1 (degrees)')
    ax2.set_title('Allowed Angular Combinations\n(Green = Allowed, Red = Forbidden)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Selection Rule Status')
    
    plt.tight_layout()
    plt.savefig('qtep_selection_rules.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('qtep_selection_rules.png', dpi=300, bbox_inches='tight')
    
    # Plot 3: Projection flow visualization
    fig3 = framework.plot_projection_flow()
    plt.savefig('qtep_projection_flow.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('qtep_projection_flow.png', dpi=300, bbox_inches='tight')
    
    # Plot 4: Angular projection mapping
    fig4, ax = plt.subplots(figsize=(10, 6))
    
    e8_angles = np.linspace(0, np.pi, 100)
    projected_angles = [framework.angular_projection_formula(a) for a in e8_angles]
    
    ax.plot(np.degrees(e8_angles), np.degrees(projected_angles), 'b-', linewidth=2)
    ax.plot([0, 180], [0, 180], 'k--', alpha=0.5, label='Identity projection')
    
    # Mark special E8 angles
    special_e8 = [30, 45, 60, 90, 120, 135, 150]
    for angle in special_e8:
        e8_rad = np.radians(angle)
        proj_rad = framework.angular_projection_formula(e8_rad)
        ax.plot(angle, np.degrees(proj_rad), 'go', markersize=8)
        ax.text(angle, np.degrees(proj_rad) + 2, f'{angle}°', ha='center', fontsize=8)
    
    ax.set_xlabel('E8 Angle (degrees)')
    ax.set_ylabel('Projected 3D Angle (degrees)')
    ax.set_title('QTEP-Mediated Angular Projection\nE8 Space → 3D Space')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('angular_projection_mapping.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('angular_projection_mapping.png', dpi=300, bbox_inches='tight')
    
    print("QTEP projection theory plots saved to current directory")


def main():
    """Run the QTEP projection framework analysis."""
    framework = QTEPProjectionFramework()
    
    print("QTEP Projection Theory Analysis")
    print("="*50)
    
    # Analyze QTEP angle origins
    print("\nPotential QTEP Angle Origins:")
    print("-"*30)
    origins = framework.analyze_qtep_angle_origin()
    for name, angle in sorted(origins.items(), key=lambda x: abs(x[1] - 35.3)):
        error = abs(angle - 35.3)
        print(f"{name:25s}: {angle:6.2f}° (Δ = {error:5.2f}°)")
    
    # Test selection rules
    print("\nSelection Rules Test:")
    print("-"*30)
    rules = framework.derive_selection_rules()
    
    test_pairs = [
        (120, 35.3),  # Should produce 85°
        (70.5, 35.3), # Should produce 105°
        (135, 30),    # Should produce 165°
        (45, 60),     # Crystallographic pair
        (35.3, 48.2), # Heterotic pair
    ]
    
    for a1, a2 in test_pairs:
        print(f"\nTesting {a1}° + {a2}° = {(a1+a2)%180}°:")
        for rule_name, rule_func in rules.items():
            result = rule_func(a1, a2)
            print(f"  {rule_name:20s}: {'✓' if result else '✗'}")
    
    # Generate all plots
    generate_theoretical_framework_plots()
    
    # Calculate projection cascade properties
    print("\nProjection Cascade Properties:")
    print("-"*30)
    cascade = framework.heterotic_projection_cascade()
    for name, projector in cascade.items():
        if name != 'composite':
            print(f"{name}: shape = {projector.shape}")


if __name__ == "__main__":
    main() 