"""
Angular Combination Analyzer for E8×E8 Heterotic Structure
=========================================================

This module analyzes second-order angular effects arising from combinations
of fundamental crystallographic and heterotic composite angles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
from itertools import combinations_with_replacement, product

class AngularCombinationAnalyzer:
    """Analyzes angular combinations and second-order effects in E8×E8 structure."""
    
    def __init__(self):
        # Fundamental crystallographic angles with uncertainties
        self.crystallographic_angles = {
            30.0: (0.5, 'equilateral triangle vertex'),
            45.0: (0.3, 'right triangle configuration'),
            60.0: (0.3, 'hexagonal substructure'),
            90.0: (0.3, 'orthogonal root pairs'),
            120.0: (0.3, 'supplementary hexagonal'),
            135.0: (0.3, 'supplementary right triangle'),
            150.0: (0.3, 'supplementary equilateral')
        }
        
        # Heterotic composite angles
        self.heterotic_angles = {
            35.3: (0.4, 'QTEP-derived orientation'),
            48.2: (0.4, 'secondary root alignment'),
            70.5: (0.5, 'primary E8 symmetry axis')
        }
        
        # All predicted angles
        self.all_angles = {**self.crystallographic_angles, **self.heterotic_angles}
        
        # Observed unpredicted peaks
        self.unpredicted_peaks = [85.0, 105.0, 165.0, 13.0, 83.5, 95.3, 108.2]
        
        # QTEP ratio
        self.qtep_ratio = 2.257
        
    def analyze_qtep_angle_derivations(self) -> Dict[str, float]:
        """Explore potential mathematical origins of the QTEP angle."""
        derivations = {}
        
        # Direct QTEP ratio relationships
        derivations['arctan(1/QTEP)'] = np.degrees(np.arctan(1/self.qtep_ratio))
        derivations['arcsin(1/sqrt(1+QTEP²))'] = np.degrees(
            np.arcsin(1/np.sqrt(1 + self.qtep_ratio**2))
        )
        derivations['arccos(QTEP/sqrt(1+QTEP²))'] = np.degrees(
            np.arccos(self.qtep_ratio/np.sqrt(1 + self.qtep_ratio**2))
        )
        
        # More complex relationships
        derivations['arctan(sqrt(QTEP))'] = np.degrees(np.arctan(np.sqrt(self.qtep_ratio)))
        derivations['90° - arctan(QTEP)'] = 90 - np.degrees(np.arctan(self.qtep_ratio))
        
        # Golden ratio connection
        phi = (1 + np.sqrt(5))/2
        derivations['arcsin(1/φ)'] = np.degrees(np.arcsin(1/phi))
        derivations['arctan(1/φ)'] = np.degrees(np.arctan(1/phi))
        
        # E8-related angles
        derivations['360°/8 - 9.7°'] = 45 - 9.7  # E8 symmetry adjustment
        derivations['arcsin(sqrt(2/π))'] = np.degrees(np.arcsin(np.sqrt(2/np.pi)))
        
        return derivations
    
    def find_angular_combinations(self, target: float, tolerance: float = 2.0) -> List[Dict]:
        """Find all combinations of predicted angles that yield a target angle."""
        results = []
        angles_list = list(self.all_angles.keys())
        
        # Single angles (trivial case)
        for angle in angles_list:
            if abs(angle - target) < tolerance:
                results.append({
                    'type': 'single',
                    'angles': [angle],
                    'operation': 'identity',
                    'value': angle,
                    'error': abs(angle - target)
                })
        
        # Pairwise sums
        for a1, a2 in combinations_with_replacement(angles_list, 2):
            # Sum modulo 180
            sum_angle = (a1 + a2) % 180
            if abs(sum_angle - target) < tolerance:
                results.append({
                    'type': 'sum',
                    'angles': [a1, a2],
                    'operation': f"{a1}° + {a2}°",
                    'value': sum_angle,
                    'error': abs(sum_angle - target)
                })
            
            # Sum without modulo
            if a1 + a2 < 180 and abs(a1 + a2 - target) < tolerance:
                results.append({
                    'type': 'sum',
                    'angles': [a1, a2],
                    'operation': f"{a1}° + {a2}°",
                    'value': a1 + a2,
                    'error': abs(a1 + a2 - target)
                })
        
        # Pairwise differences
        for a1, a2 in product(angles_list, repeat=2):
            if a1 != a2:
                diff_angle = abs(a1 - a2)
                if abs(diff_angle - target) < tolerance:
                    results.append({
                        'type': 'difference',
                        'angles': [a1, a2],
                        'operation': f"|{a1}° - {a2}°|",
                        'value': diff_angle,
                        'error': abs(diff_angle - target)
                    })
        
        # Three-angle combinations
        for a1, a2, a3 in combinations_with_replacement(angles_list, 3):
            # Various three-angle operations
            combinations_3 = [
                (a1 + a2 - a3, f"{a1}° + {a2}° - {a3}°"),
                (a1 + a2 + a3, f"{a1}° + {a2}° + {a3}°") if a1 + a2 + a3 <= 180 else (None, None),
                (abs(a1 - a2) + a3, f"|{a1}° - {a2}°| + {a3}°"),
            ]
            
            for value, operation in combinations_3:
                if value is not None and abs(value - target) < tolerance:
                    results.append({
                        'type': 'triple',
                        'angles': [a1, a2, a3],
                        'operation': operation,
                        'value': value,
                        'error': abs(value - target)
                    })
        
        # Sort by error
        results.sort(key=lambda x: x['error'])
        return results
    
    def analyze_all_unpredicted_peaks(self) -> Dict[float, List[Dict]]:
        """Analyze all unpredicted peaks to find their potential origins."""
        analysis = {}
        
        for peak in self.unpredicted_peaks:
            combinations = self.find_angular_combinations(peak)
            analysis[peak] = combinations
            
        return analysis
    
    def calculate_selection_rules(self) -> pd.DataFrame:
        """Calculate which angular combinations are allowed based on proposed selection rules."""
        data = []
        angles_list = list(self.all_angles.keys())
        
        for a1, a2 in combinations_with_replacement(angles_list, 2):
            # Classify angles
            is_crystallographic_1 = a1 in self.crystallographic_angles
            is_crystallographic_2 = a2 in self.crystallographic_angles
            is_heterotic_1 = a1 in self.heterotic_angles
            is_heterotic_2 = a2 in self.heterotic_angles
            
            # Calculate combination
            sum_angle = (a1 + a2) % 180
            diff_angle = abs(a1 - a2)
            
            # Determine if QTEP angle (35.3°) is involved
            involves_qtep = (abs(a1 - 35.3) < 0.1) or (abs(a2 - 35.3) < 0.1)
            
            # Check if combination matches any unpredicted peak
            matches_unpredicted = any(
                abs(sum_angle - peak) < 2.0 or abs(diff_angle - peak) < 2.0 
                for peak in self.unpredicted_peaks
            )
            
            data.append({
                'angle1': a1,
                'angle2': a2,
                'sum': sum_angle,
                'difference': diff_angle,
                'type1': 'crystallographic' if is_crystallographic_1 else 'heterotic',
                'type2': 'crystallographic' if is_crystallographic_2 else 'heterotic',
                'involves_qtep': involves_qtep,
                'matches_unpredicted': matches_unpredicted,
                'combination_type': f"{'C' if is_crystallographic_1 else 'H'}+{'C' if is_crystallographic_2 else 'H'}"
            })
        
        return pd.DataFrame(data)
    
    def plot_angular_spectrum_hierarchy(self):
        """Create visualization of the hierarchical angular structure."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Level 1: Fundamental angles
        ax1 = axes[0]
        crystal_angles = list(self.crystallographic_angles.keys())
        ax1.eventplot([crystal_angles], orientation='horizontal', colors='blue', linewidths=2)
        ax1.set_xlim(0, 180)
        ax1.set_title('Level 1: Crystallographic Angles', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Angle (degrees)')
        ax1.grid(True, alpha=0.3)
        
        # Add labels
        for angle in crystal_angles:
            ax1.text(angle, 0.5, f'{angle}°', ha='center', va='bottom', fontsize=9)
        
        # Level 2: Heterotic angles
        ax2 = axes[1]
        heterotic_angles = list(self.heterotic_angles.keys())
        ax2.eventplot([heterotic_angles], orientation='horizontal', colors='red', linewidths=2)
        ax2.set_xlim(0, 180)
        ax2.set_title('Level 2: Heterotic Composite Angles', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Angle (degrees)')
        ax2.grid(True, alpha=0.3)
        
        # Add labels with special marking for QTEP angle
        for angle in heterotic_angles:
            color = 'darkred' if abs(angle - 35.3) < 0.1 else 'red'
            weight = 'bold' if abs(angle - 35.3) < 0.1 else 'normal'
            ax2.text(angle, 0.5, f'{angle}°', ha='center', va='bottom', 
                    fontsize=9, color=color, fontweight=weight)
        
        # Level 3: Second-order effects
        ax3 = axes[2]
        ax3.eventplot([self.unpredicted_peaks], orientation='horizontal', 
                     colors='green', linewidths=2)
        ax3.set_xlim(0, 180)
        ax3.set_title('Level 3: Second-Order Interference Effects', 
                     fontsize=14, fontweight='bold')
        ax3.set_xlabel('Angle (degrees)')
        ax3.grid(True, alpha=0.3)
        
        # Add labels with explanations
        explanations = {
            85.0: '120° - 35.3°',
            105.0: '70.5° + 35.3°',
            165.0: '135° + 30°',
            13.0: '|48.2° - 35.3°|',
            83.5: '48.2° + 35.3°',
            95.3: '60° + 35.3°',
            108.2: '60° + 48.2°'
        }
        
        for angle in self.unpredicted_peaks:
            ax3.text(angle, 0.5, f'{angle}°', ha='center', va='bottom', fontsize=9)
            if angle in explanations:
                ax3.text(angle, -0.3, explanations[angle], ha='center', 
                        va='top', fontsize=8, style='italic')
        
        plt.tight_layout()
        return fig
    
    def generate_report(self) -> str:
        """Generate a comprehensive report of the analysis."""
        report = []
        report.append("="*60)
        report.append("Angular Combination Analysis Report")
        report.append("="*60)
        
        # QTEP angle derivations
        report.append("\n1. QTEP Angle (35.3°) Potential Derivations:")
        report.append("-"*40)
        derivations = self.analyze_qtep_angle_derivations()
        for formula, value in derivations.items():
            diff = abs(value - 35.3)
            report.append(f"   {formula:30s} = {value:6.2f}° (Δ = {diff:5.2f}°)")
        
        # Analysis of unpredicted peaks
        report.append("\n2. Second-Order Effect Analysis:")
        report.append("-"*40)
        peak_analysis = self.analyze_all_unpredicted_peaks()
        
        for peak, combinations in peak_analysis.items():
            report.append(f"\n   Peak at {peak}°:")
            if combinations:
                for i, combo in enumerate(combinations[:5]):  # Top 5 matches
                    report.append(f"      {i+1}. {combo['operation']:30s} = {combo['value']:6.2f}° "
                                f"(error: {combo['error']:4.2f}°)")
            else:
                report.append("      No combinations found within tolerance")
        
        # Selection rules analysis
        report.append("\n3. Selection Rules Statistics:")
        report.append("-"*40)
        df_rules = self.calculate_selection_rules()
        
        # Count combinations by type
        combo_counts = df_rules.groupby('combination_type')['matches_unpredicted'].sum()
        report.append("   Combinations matching unpredicted peaks:")
        for combo_type, count in combo_counts.items():
            report.append(f"      {combo_type}: {count}")
        
        # QTEP involvement
        qtep_matches = df_rules[df_rules['involves_qtep'] & df_rules['matches_unpredicted']]
        report.append(f"\n   Matches involving QTEP angle (35.3°): {len(qtep_matches)}")
        
        return "\n".join(report)


def main():
    """Run the complete angular combination analysis."""
    analyzer = AngularCombinationAnalyzer()
    
    # Generate and print report
    report = analyzer.generate_report()
    print(report)
    
    # Create hierarchical visualization
    fig = analyzer.plot_angular_spectrum_hierarchy()
    plt.savefig('string_analysis/hierarchical_angular_structure.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('string_analysis/hierarchical_angular_structure.png', dpi=300, bbox_inches='tight')
    
    # Additional detailed analysis plots
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: QTEP angle relationships
    ax1 = axes[0, 0]
    derivations = analyzer.analyze_qtep_angle_derivations()
    values = list(derivations.values())
    labels = list(derivations.keys())
    errors = [abs(v - 35.3) for v in values]
    
    colors = ['red' if e < 5 else 'blue' for e in errors]
    ax1.barh(range(len(values)), errors, color=colors)
    ax1.set_yticks(range(len(values)))
    ax1.set_yticklabels([f"{l}\\n({v:.1f}°)" for l, v in zip(labels, values)], fontsize=8)
    ax1.set_xlabel('Deviation from 35.3° (degrees)')
    ax1.set_title('QTEP Angle Derivation Candidates')
    ax1.axvline(5, color='red', linestyle='--', alpha=0.5)
    
    # Plot 2: Angular combination matrix
    ax2 = axes[0, 1]
    angles = sorted(list(analyzer.all_angles.keys()))
    n = len(angles)
    combination_matrix = np.zeros((n, n))
    
    for i, a1 in enumerate(angles):
        for j, a2 in enumerate(angles):
            # Check if combination matches any unpredicted peak
            sum_angle = (a1 + a2) % 180
            diff_angle = abs(a1 - a2)
            
            for peak in analyzer.unpredicted_peaks:
                if abs(sum_angle - peak) < 2.0:
                    combination_matrix[i, j] = 1
                elif abs(diff_angle - peak) < 2.0:
                    combination_matrix[i, j] = 2
    
    im = ax2.imshow(combination_matrix, cmap='coolwarm', aspect='equal')
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels([f'{a:.1f}°' for a in angles], rotation=45, ha='right')
    ax2.set_yticklabels([f'{a:.1f}°' for a in angles])
    ax2.set_title('Angular Combinations Matching Unpredicted Peaks')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['No match', 'Sum match', 'Diff match'])
    
    # Plot 3: Distribution of all possible combinations
    ax3 = axes[1, 0]
    all_sums = []
    all_diffs = []
    
    for a1, a2 in combinations_with_replacement(angles, 2):
        all_sums.append((a1 + a2) % 180)
        all_diffs.append(abs(a1 - a2))
    
    ax3.hist(all_sums + all_diffs, bins=36, alpha=0.7, color='gray', label='All combinations')
    
    # Overlay unpredicted peaks
    for peak in analyzer.unpredicted_peaks:
        ax3.axvline(peak, color='red', linestyle='--', linewidth=2, label=f'{peak}°')
    
    ax3.set_xlabel('Angle (degrees)')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Angular Combinations')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Information flow diagram
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.9, 'Information Flow Through Angular Channels', 
             ha='center', va='center', fontsize=14, fontweight='bold', 
             transform=ax4.transAxes)
    
    # Create flow diagram
    flow_text = """
    E8×E8 (496D)
         ↓
    Primary Projection
         ↓
    Level 1: Crystallographic (7 angles)
         ↓ ↘
         ↓  QTEP Mediation (35.3°)
         ↓  ↙
    Level 2: Heterotic (3 angles)
         ↓
    Interference Effects
         ↓
    Level 3: Second-Order (3+ angles)
         ↓
    Observable Void Alignments
    """
    
    ax4.text(0.5, 0.45, flow_text, ha='center', va='center', 
             fontsize=10, family='monospace', transform=ax4.transAxes)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('string_analysis/angular_combination_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('string_analysis/angular_combination_analysis.png', dpi=300, bbox_inches='tight')
    
    # Create selection rules heatmap
    fig3, ax = plt.subplots(figsize=(10, 8))
    df_rules = analyzer.calculate_selection_rules()
    
    # Create pivot table for heatmap
    pivot = df_rules.pivot_table(
        values='matches_unpredicted', 
        index='angle1', 
        columns='angle2', 
        aggfunc='sum',
        fill_value=0
    )
    
    sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='d', cbar_kws={'label': 'Matches'})
    ax.set_title('Selection Rules: Angular Combinations Matching Unpredicted Peaks')
    ax.set_xlabel('Angle 2 (degrees)')
    ax.set_ylabel('Angle 1 (degrees)')
    
    plt.tight_layout()
    plt.savefig('string_analysis/selection_rules_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('string_analysis/selection_rules_heatmap.png', dpi=300, bbox_inches='tight')
    
    print("\nVisualizations saved to current directory")
    
    # Save detailed results to file
    with open('string_analysis/angular_combination_results.txt', 'w', encoding='utf-8') as f:
        f.write(report)
        f.write("\n\nDetailed Analysis Results\n")
        f.write("="*60 + "\n")
        
        # Write all combinations for each unpredicted peak
        peak_analysis = analyzer.analyze_all_unpredicted_peaks()
        for peak, combinations in peak_analysis.items():
            f.write(f"\nComplete combinations for {peak}°:\n")
            for combo in combinations:
                f.write(f"   {combo['operation']:40s} = {combo['value']:6.2f}° "
                       f"(error: {combo['error']:4.2f}°, type: {combo['type']})\n")


if __name__ == "__main__":
    main() 