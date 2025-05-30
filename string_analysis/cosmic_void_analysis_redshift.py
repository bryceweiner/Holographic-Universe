"""
Enhanced Cosmic Void Analysis with Redshift Binning
General void structure analysis incorporating:
1. Redshift evolution of void properties
2. Survey selection effects and completeness
3. Scale-dependent cosmological corrections
4. Modern statistical techniques
5. Comparison with OUT predictions across cosmic time
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats, spatial
from scipy.optimize import curve_fit
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from astropy.cosmology import Planck18
from astropy import units as u
import networkx as nx

# Theoretical parameters
C_G_OUT = 25/32  # OUT prediction
ASPECT_RATIO_OUT = 2.257  # OUT QTEP ratio
GAMMA_R_0 = 1.89e-29  # s^-1

def gamma_evolution(z, alpha=0.05):
    """Evolution of information processing rate with refined cosmological coupling"""
    # IMPROVED: More accurate gamma evolution with expansion coupling
    # Base power-law evolution
    gamma_base = GAMMA_R_0 * (1 + z)**alpha
    
    # IMPROVED: Couple to cosmic expansion with damping factor
    H_z = hubble_parameter(z)
    H_0 = 70  # km/s/Mpc
    expansion_coupling = (H_0/H_z)**0.2
    
    # IMPROVED: Information processing maturation factor
    maturation = 1 - 0.15 * z / (1 + 0.7*z)
    
    return gamma_base * expansion_coupling * maturation

def hubble_parameter(z):
    """Hubble parameter at redshift z"""
    return Planck18.H(z).to(u.km/u.s/u.Mpc).value

def void_size_function_evolution(z):
    """Evolution of void size function with redshift"""
    # Larger voids at higher z due to less evolved structure
    return 1 + 0.3 * z

def density_profile_out(r, r_void, C_G_eff):
    """OUT-predicted void density profile with evolved C(G)"""
    r_norm = r / r_void
    return 1 - np.exp(-(r_norm)**C_G_eff)

def survey_completeness(z, void_size_mpc, survey='Combined'):
    """Model survey completeness as function of redshift and size"""
    
    survey_params = {
        'SDSS': {'z_max': 0.3, 'size_min': 10, 'completeness': 0.85},
        'DESI': {'z_max': 1.4, 'size_min': 8, 'completeness': 0.92},
        'Combined': {'z_max': 1.2, 'size_min': 5, 'completeness': 0.90}
    }
    
    params = survey_params.get(survey, survey_params['Combined'])
    
    # Redshift cutoff
    if z > params['z_max']:
        return 0.0
    
    # Size cutoff  
    if void_size_mpc < params['size_min']:
        return 0.0
    
    # Angular size effect
    d_A = Planck18.angular_diameter_distance(z).to(u.Mpc).value
    angular_size = np.degrees(void_size_mpc / d_A)
    
    # Completeness function
    z_factor = np.exp(-(z / params['z_max'])**2)
    size_factor = 1.0 / (1 + np.exp(-(angular_size - 0.1) * 20))
    
    return params['completeness'] * z_factor * size_factor

def load_real_void_catalogs():
    """Load the actual void catalogs from data files"""
    print("Loading real void catalogs...")
    
    catalogs = {}
    
    # Load individual survey catalogs
    survey_files = {
        'SDSS': 'data/sdss_voids.txt',
        'ZOBOV': 'data/zobov_voids.txt', 
        'VIDE': 'data/vide_voids.txt',
        '2MRS': 'data/2mrs_voids.txt'
    }
    
    for survey, filepath in survey_files.items():
        if os.path.exists(filepath):
            try:
                # Load data, skipping comment lines
                data = pd.read_csv(filepath, sep='\s+', comment='#')
                
                # Standardize column names to match analysis expectations
                column_mapping = {
                    'VoidID': 'void_id',
                    'RA': 'ra_deg', 
                    'Dec': 'dec_deg',
                    'Redshift': 'redshift',
                    'Radius_Mpc': 'radius_mpc',
                    'AspectRatio': 'aspect_ratio',
                    'Ellipticity': 'ellipticity',
                    'Prolateness': 'prolateness', 
                    'DensityContrast': 'central_density',
                    'x_Mpc': 'x_mpc',
                    'y_Mpc': 'y_mpc',
                    'z_Mpc': 'z_mpc'
                }
                
                # Rename columns that exist
                for old_name, new_name in column_mapping.items():
                    if old_name in data.columns:
                        data = data.rename(columns={old_name: new_name})
                
                # Add survey identifier
                data['survey'] = survey
                
                # Add derived properties expected by analysis functions
                if 'redshift' in data.columns and 'x_mpc' in data.columns:
                    # Calculate comoving distance 
                    data['comoving_distance'] = np.sqrt(data['x_mpc']**2 + data['y_mpc']**2 + data['z_mpc']**2)
                
                # Calculate gamma_z for each void
                if 'redshift' in data.columns:
                    data['gamma_z'] = gamma_evolution(data['redshift'])
                    
                # Calculate effective C(G) for each void
                if 'gamma_z' in data.columns and 'redshift' in data.columns:
                    data['c_g_effective'] = C_G_OUT * (data['gamma_z'] / GAMMA_R_0) * (1 - 0.25 * data['redshift'] / (1 + data['redshift']))
                
                # Calculate completeness for each void
                if 'redshift' in data.columns and 'radius_mpc' in data.columns:
                    data['completeness'] = [survey_completeness(z, r, survey) 
                                          for z, r in zip(data['redshift'], data['radius_mpc'])]
                
                catalogs[survey] = data
                print(f"✓ Loaded {survey}: {len(data)} voids")
                print(f"  Columns: {list(data.columns)}")
                
            except Exception as e:
                print(f"✗ Failed to load {survey}: {e}")
        else:
            print(f"✗ File not found: {filepath}")
    
    # Create combined catalog
    if catalogs:
        combined = pd.concat(catalogs.values(), ignore_index=True)
        print(f"✓ Combined catalog: {len(combined)} voids")
        print(f"✓ Final columns: {list(combined.columns)}")
        return combined
    else:
        print("No void catalogs found - generating synthetic data")
        return generate_realistic_void_catalog()

def generate_realistic_void_catalog(n_total=2500, z_max=1.2):
    """Generate realistic void catalog with proper cosmological evolution"""
    
    np.random.seed(456)
    
    # Realistic redshift distribution
    z_samples = np.linspace(0.01, z_max, 800)
    
    # Volume element × number density evolution
    comoving_vol = 4 * np.pi * (Planck18.comoving_distance(z_samples).to(u.Mpc).value)**2
    comoving_vol *= Planck18.differential_comoving_volume(z_samples).to(u.Mpc**3/u.sr).value
    
    # Number density evolution (fewer voids at high z)
    density_evolution = (1 + z_samples)**(-1.5)
    
    weights = comoving_vol * density_evolution
    weights = np.nan_to_num(weights)
    weights /= np.sum(weights)
    
    redshifts = np.random.choice(z_samples, size=n_total, p=weights)
    
    void_data = []
    
    for i, z in enumerate(redshifts):
        # Size distribution evolution
        size_evolution = void_size_function_evolution(z)
        base_scale = 25 * size_evolution  # Mpc
        
        # Log-normal size distribution
        void_radius = np.random.lognormal(np.log(base_scale), 0.8)
        void_radius = np.clip(void_radius, 3, 400)
        
        # Survey selection
        completeness = survey_completeness(z, void_radius, 'Combined')
        
        if np.random.random() < completeness:
            # Position on sky
            d_c = Planck18.comoving_distance(z).to(u.Mpc).value
            
            ra = np.random.uniform(0, 360)
            dec = np.arcsin(2 * np.random.random() - 1) * 180 / np.pi
            
            # Convert to Cartesian
            dec_rad, ra_rad = np.radians(dec), np.radians(ra)
            x = d_c * np.cos(dec_rad) * np.cos(ra_rad)
            y = d_c * np.cos(dec_rad) * np.sin(ra_rad)
            z_cart = d_c * np.sin(dec_rad)
            
            # Void properties with evolution
            gamma_z = gamma_evolution(z)
            
            # Aspect ratio with OUT evolution
            aspect_ratio_base = ASPECT_RATIO_OUT * (gamma_z / GAMMA_R_0)**0.1
            aspect_ratio = np.random.normal(aspect_ratio_base, 0.5)
            aspect_ratio = np.clip(aspect_ratio, 1.1, 5.0)
            
            # Density contrast with redshift evolution
            c_g_eff = C_G_OUT * (gamma_z / GAMMA_R_0) * (1 - 0.25 * z / (1 + z))
            central_density = -0.9 * (1 - np.exp(-(void_radius/40)**c_g_eff))
            
            # Velocity dispersion (for peculiar velocity field)
            velocity_dispersion = 150 * (void_radius / 50)**0.6 * (1 + z)**0.3  # km/s
            
            void_data.append({
                'void_id': i,
                'redshift': z,
                'radius_mpc': void_radius,
                'aspect_ratio': aspect_ratio,
                'central_density': central_density,
                'velocity_dispersion': velocity_dispersion,
                'ra_deg': ra,
                'dec_deg': dec,
                'x_mpc': x,
                'y_mpc': y,
                'z_mpc': z_cart,
                'comoving_distance': d_c,
                'completeness': completeness,
                'c_g_effective': c_g_eff,
                'gamma_z': gamma_z
            })
    
    return pd.DataFrame(void_data)

def analyze_void_size_function_by_redshift(void_catalog, z_bins):
    """Analyze void size function in redshift bins"""
    
    results = {}
    
    for i in range(len(z_bins) - 1):
        z_min, z_max = z_bins[i], z_bins[i+1]
        z_center = (z_min + z_max) / 2
        
        mask = (void_catalog['redshift'] >= z_min) & (void_catalog['redshift'] < z_max)
        bin_voids = void_catalog[mask]
        
        if len(bin_voids) < 50:
            continue
            
        radii = bin_voids['radius_mpc'].values
        
        # Size distribution analysis
        # IMPROVED: OUT prediction with redshift evolution
        # Base exponent from theory
        out_base_exponent = -0.66
        
        # IMPROVED: Account for structure growth with redshift
        # Evolve with redshift - structure grows more organized with time
        z_factor = 1.0 + 0.1 * np.log10(1 + z_center)
        out_exponent = out_base_exponent * z_factor
        
        # Calculate cumulative distribution
        radii_sorted = np.sort(radii)
        n_cumulative = np.arange(len(radii_sorted), 0, -1)
        
        # Fit power law in log space
        try:
            # Use middle range to avoid edge effects
            mask_fit = (radii_sorted > np.percentile(radii_sorted, 20)) & \
                      (radii_sorted < np.percentile(radii_sorted, 80))
            
            if np.sum(mask_fit) > 10:
                log_r = np.log10(radii_sorted[mask_fit])
                log_n = np.log10(n_cumulative[mask_fit])
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_r, log_n)
                
                results[f'z_{z_min:.1f}_{z_max:.1f}'] = {
                    'z_center': z_center,
                    'z_min': z_min,
                    'z_max': z_max,
                    'n_voids': len(bin_voids),
                    'radii': radii,
                    'radii_sorted': radii_sorted,
                    'n_cumulative': n_cumulative,
                    'size_slope': slope,
                    'size_slope_err': std_err,
                    'size_r_squared': r_value**2,
                    'out_prediction': out_exponent,  # Now uses evolving prediction
                    'out_base_prediction': out_base_exponent,  # Reference to base value
                    'slope_deviation': abs(slope - out_exponent),
                    'mean_radius': np.mean(radii),
                    'median_radius': np.median(radii)
                }
                
                print(f"z={z_center:.2f}: Size slope = {slope:.3f} ± {std_err:.3f} "
                      f"(OUT: {out_exponent:.3f})")
                
        except Exception as e:
            print(f"z={z_center:.2f}: Size function fitting failed: {e}")
    
    return results

def analyze_aspect_ratios_by_redshift(void_catalog, z_bins):
    """Analyze void aspect ratios in redshift bins"""
    
    results = {}
    
    for i in range(len(z_bins) - 1):
        z_min, z_max = z_bins[i], z_bins[i+1]
        z_center = (z_min + z_max) / 2
        
        mask = (void_catalog['redshift'] >= z_min) & (void_catalog['redshift'] < z_max)
        bin_voids = void_catalog[mask]
        
        if len(bin_voids) < 30:
            continue
            
        aspect_ratios = bin_voids['aspect_ratio'].values
        
        # OUT prediction with evolution
        gamma_z = gamma_evolution(z_center)
        predicted_aspect = ASPECT_RATIO_OUT * (gamma_z / GAMMA_R_0)**0.1
        
        # Statistical analysis
        mean_aspect = np.mean(aspect_ratios)
        std_aspect = np.std(aspect_ratios)
        sem_aspect = std_aspect / np.sqrt(len(aspect_ratios))
        
        # KS test against normal distribution around prediction
        theoretical_sample = np.random.normal(predicted_aspect, 0.5, 10000)
        ks_stat, ks_p = stats.ks_2samp(aspect_ratios, theoretical_sample)
        
        results[f'z_{z_min:.1f}_{z_max:.1f}'] = {
            'z_center': z_center,
            'n_voids': len(bin_voids),
            'aspect_ratios': aspect_ratios,
            'mean_aspect': mean_aspect,
            'std_aspect': std_aspect,
            'sem_aspect': sem_aspect,
            'predicted_aspect': predicted_aspect,
            'deviation': abs(mean_aspect - predicted_aspect),
            'deviation_sigma': abs(mean_aspect - predicted_aspect) / sem_aspect,
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p
        }
        
        print(f"z={z_center:.2f}: Aspect = {mean_aspect:.3f} ± {sem_aspect:.3f}, "
              f"predicted = {predicted_aspect:.3f} "
              f"({abs(mean_aspect - predicted_aspect)/sem_aspect:.1f}σ)")
    
    return results

# Legacy interface for compatibility with main pipeline
class VoidDataProcessor:
    """Compatibility wrapper for the original void data processing interface"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.void_catalog = None
        
    def download_void_catalogs(self):
        """Load real void catalogs or generate synthetic ones"""
        print("="*60)
        print("PROCESSING COSMIC VOID CATALOGS")
        print("="*60)
        
        self.void_catalog = load_real_void_catalogs()
        return self.void_catalog
        
    def analyze_survey_properties(self):
        """Analyze properties by survey"""
        if self.void_catalog is None:
            return
            
        print("="*60)
        print("SURVEY PROPERTIES ANALYSIS")
        print("="*60)
        
        if 'survey' in self.void_catalog.columns:
            for survey in self.void_catalog['survey'].unique():
                survey_data = self.void_catalog[self.void_catalog['survey'] == survey]
                print(f"\n{survey} Survey:")
                print(f"  Number of voids: {len(survey_data)}")
                print(f"  Redshift range: {survey_data['redshift'].min():.3f} - {survey_data['redshift'].max():.3f}")
                print(f"  Mean redshift: {survey_data['redshift'].mean():.3f}")
                print(f"  Size range: {survey_data['radius_mpc'].min():.1f} - {survey_data['radius_mpc'].max():.1f} Mpc")
                print(f"  Mean size: {survey_data['radius_mpc'].mean():.1f} Mpc")
                if 'aspect_ratio' in survey_data.columns:
                    print(f"  Mean aspect ratio: {survey_data['aspect_ratio'].mean():.3f} ± {survey_data['aspect_ratio'].std():.3f}")
                    print(f"  QTEP prediction: {ASPECT_RATIO_OUT:.3f}")
                    
                    # Calculate deviation from QTEP
                    deviation = abs(survey_data['aspect_ratio'].mean() - ASPECT_RATIO_OUT)
                    significance = deviation / (survey_data['aspect_ratio'].std() / np.sqrt(len(survey_data)))
                    print(f"  QTEP deviation: {deviation:.3f} ({significance:.1f}σ)")
        
    def save_catalogs(self):
        """Save catalogs"""
        if self.void_catalog is not None:
            filename = os.path.join(self.data_dir, 'combined_voids.csv')
            self.void_catalog.to_csv(filename, index=False)
            print(f"✓ Saved combined catalog: {filename}")
    
    def get_combined_catalog(self):
        """Get the combined void catalog"""
        if self.void_catalog is None:
            self.download_void_catalogs()
        return self.void_catalog
    
    def get_survey_catalog(self, survey_name):
        """Get catalog for specific survey"""
        if self.void_catalog is None:
            self.download_void_catalogs()
        if 'survey' in self.void_catalog.columns:
            return self.void_catalog[self.void_catalog['survey'] == survey_name]
        return self.void_catalog 