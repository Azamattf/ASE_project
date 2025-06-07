import pandas as pd
import numpy as np

# Define geometric parameters
panel_volume = 600000  # mm^3
shell_element_volume = panel_volume / 3  # each panel is made up of 3 shell elements

# Geometric parameters for buckling analysis (you may need to adjust these based on your model)
a = 750  # panel length (mm) (of WHOLE PANEL, not INDIVIDUAL ELEMENTS)
b = 200  # panel width (mm)
t = 4.0  # skin thickness (mm)
E = 64885.28  # E-modulus B-basis (MPa)
nu = 0.34  # Poisson's ratio - found from aluminium model card in HyperMesh

def calculate_critical_buckling_stress_compression(k_c, t, b):
    """Calculate critical buckling stress for compression"""
    return k_c * (np.pi**2 * E) / (12 * (1 - nu**2)) * (t/b)**2

def calculate_critical_buckling_stress_shear(k_s, t, b):
    """Calculate critical buckling stress for shear"""
    return k_s * (np.pi**2 * E) / (12 * (1 - nu**2)) * (t/b)**2

def find_optimal_m_for_compression(aspect_ratio):
    """Find optimal m that minimizes k_c for compression"""
    # Try different values of m and find the one that minimizes k_c
    m_values = np.arange(1, 10, 1)
    k_c_values = []
    
    for m in m_values:
        k_c = (m/aspect_ratio + aspect_ratio/m)**2
        k_c_values.append(k_c)
    
    min_idx = np.argmin(k_c_values)
    return m_values[min_idx], k_c_values[min_idx]

def calculate_volume_averaged_stresses(panel_elements):
    """Calculate volume-averaged stresses for a panel (3 elements)"""
    total_volume = len(panel_elements) * shell_element_volume
    
    sigma_avg_xx = sum(row['XX'] * shell_element_volume for _, row in panel_elements.iterrows()) / total_volume
    sigma_avg_yy = sum(row['YY'] * shell_element_volume for _, row in panel_elements.iterrows()) / total_volume
    sigma_avg_xy = sum(row['XY'] * shell_element_volume for _, row in panel_elements.iterrows()) / total_volume
    
    return sigma_avg_xx, sigma_avg_yy, sigma_avg_xy

# Read the CSV files
print("Reading CSV files...")

# Read 1D stress data
df_1d = pd.read_csv('ProjectElementStresses1D.csv', skiprows=9)
df_1d.columns = ['Elements', 'FileID', 'Loadcase', 'Step', 'Axial_Stress', 'Empty']

# Read 2D/3D stress data  
df_2d3d = pd.read_csv('ProjectElementStresses2D3D.csv', skiprows=9)
df_2d3d.columns = ['Elements', 'FileID', 'Loadcase', 'Step', 'Layer', 'XX', 'XY', 'YY', 'Empty']

# Drop the empty columns
df_1d = df_1d.drop('Empty', axis=1)
df_2d3d = df_2d3d.drop('Empty', axis=1)

print("Panel Buckling Analysis")
print("="*50)

# Calculate buckling coefficients
aspect_ratio = a/b
optimal_m, k_c = find_optimal_m_for_compression(aspect_ratio)
k_s = 5.34  # For shear (long plate approximation)

print(f"Panel dimensions: {a} x {b} mm")
print(f"Skin thickness: {t} mm")
print(f"Aspect ratio (a/b): {aspect_ratio:.2f}")
print(f"Optimal m for compression: {optimal_m:.2f}")
print(f"Compression buckling coefficient k_c: {k_c:.2f}")
print(f"Shear buckling coefficient k_s: {k_s:.2f}")

# Calculate critical stresses
sigma_xx_cr = calculate_critical_buckling_stress_compression(k_c, t, b)
tau_xy_cr = calculate_critical_buckling_stress_shear(k_s, t, b)

print(f"\nCritical Stresses:")
print(f"σ_xx,cr = {sigma_xx_cr:.2f} MPa")
print(f"τ_xy,cr = {tau_xy_cr:.2f} MPa")

# Process each load case
results = []

for loadcase in sorted(df_2d3d['Loadcase'].unique()):
    print(f"\n" + "="*60)
    print(f"LOAD CASE {loadcase}")
    print("="*60)
    
    lc_data = df_2d3d[df_2d3d['Loadcase'] == loadcase].sort_values('Elements')
    
    # Group elements into panels (every 3 elements form a panel)
    num_panels = len(lc_data) // 3
    
    print(f"{'Panel':<6} {'σ_avg,XX':<10} {'σ_avg,YY':<10} {'τ_avg,XY':<10} {'RF_buckling':<12}")
    print("-" * 60)
    
    panel_results = []
    
    for panel_id in range(num_panels):
        start_idx = panel_id * 3
        end_idx = start_idx + 3
        
        panel_elements = lc_data.iloc[start_idx:end_idx]
        
        # Step 1: Calculate volume-averaged stresses
        sigma_avg_xx, sigma_avg_yy, sigma_avg_xy = calculate_volume_averaged_stresses(panel_elements)
        
        # Step 3: Calculate reserve factor using interaction equation
        # RF = 1 / sqrt((σ_avg,XX/σ_cr)² + (τ_avg,XY/τ_cr)²)
        
        # Use absolute values for the interaction equation
        stress_ratio_xx = abs(sigma_avg_xx) / sigma_xx_cr
        stress_ratio_xy = abs(sigma_avg_xy) / tau_xy_cr
        
        interaction_factor = np.sqrt(stress_ratio_xx**2 + stress_ratio_xy**2)
        
        if interaction_factor > 0:
            rf_buckling = 1.0 / interaction_factor
        else:
            rf_buckling = float('inf')
        
        print(f"{panel_id+1:<6} {sigma_avg_xx:<10.2f} {sigma_avg_yy:<10.2f} {sigma_avg_xy:<10.2f} {rf_buckling:<12.2f}")
        
        panel_results.append({
            'LoadCase': loadcase,
            'Panel': panel_id + 1,
            'sigma_avg_xx': sigma_avg_xx,
            'sigma_avg_yy': sigma_avg_yy,
            'sigma_avg_xy': sigma_avg_xy,
            'RF_buckling': rf_buckling,
            'Elements': list(panel_elements['Elements'])
        })
    
    # Find critical panel for this load case
    min_rf = min(result['RF_buckling'] for result in panel_results)
    critical_panel = next(result for result in panel_results if result['RF_buckling'] == min_rf)
    
    print(f"\nCritical Panel: Panel {critical_panel['Panel']} (Elements {critical_panel['Elements']})")
    print(f"Minimum RF_buckling: {min_rf:.3f}")
    
    results.extend(panel_results)

# Overall summary
print("\n" + "="*70)
print("OVERALL BUCKLING ANALYSIS SUMMARY")
print("="*70)

all_rf_values = [result['RF_buckling'] for result in results if result['RF_buckling'] != float('inf')]
overall_min_rf = min(all_rf_values)
critical_result = next(result for result in results if result['RF_buckling'] == overall_min_rf)

print(f"Overall Minimum RF_buckling: {overall_min_rf:.3f}")
print(f"Critical Panel: Load Case {critical_result['LoadCase']}, Panel {critical_result['Panel']}")
print(f"Critical Panel Elements: {critical_result['Elements']}")

if overall_min_rf < 1.0:
    print("⚠️  WARNING: Buckling reserve factor < 1.0 indicates potential buckling failure!")
else:
    print("✅ All panels have acceptable buckling reserve factors (RF > 1.0)")

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('Panel_Buckling_Results.csv', index=False)
print(f"\nResults saved to: Panel_Buckling_Results.csv")

