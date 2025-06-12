import pandas as pd
import numpy as np
import sys

# Check command line arguments
if len(sys.argv) != 4:
    print("Usage: python script_e_p11.py <1D_csv_file> <2D3D_csv_file> <E_modulus>")
    sys.exit(1)

csv_1d_file = sys.argv[1]
csv_2d3d_file = sys.argv[2]
E = float(sys.argv[3])  # E-modulus from command line argument

# Define geometric parameters
panel_volume = 600000  # mm^3
shell_element_volume = panel_volume / 3  # each panel is made up of 3 shell elements

# Geometric parameters for buckling analysis
a = 750  # panel length (mm) (of WHOLE PANEL, not INDIVIDUAL ELEMENTS)
b = 200  # panel width (mm)
t = 4.0  # skin thickness (mm)
nu = 0.34  # Poisson's ratio - found from aluminium model card in HyperMesh

def find_optimal_mn_for_biaxial_buckling(alpha, beta):
    """
    Find optimal m and n that minimize k_xxx for biaxial buckling, 
    ensuring k_xxx is always positive.
    """
    min_k_xxx = float('inf')
    optimal_m, optimal_n = 1, 1
    
    # Try different combinations of m and n (integers from 1 to 10)
    for m in range(1, 11):
        for n in range(1, 11):
            
            # This is the term that can become negative if beta is negative
            buckling_term = (m**2 + beta * n**2 * alpha**2)
            
            # --- CORRECTION ---
            # Only proceed if the buckling term is positive.
            # A non-positive term means this mode is stabilized by tension and won't buckle.
            if buckling_term > 0:
                denominator = alpha**2 * buckling_term
                
                # This check is still good practice, though less likely to be zero now.
                if denominator != 0:
                    numerator = (m**2 + n**2 * alpha**2)**2
                    k_xxx = numerator / denominator
                    
                    if k_xxx < min_k_xxx:
                        min_k_xxx = k_xxx
                        optimal_m, optimal_n = m, n
            # If buckling_term is <= 0, we simply ignore this (m, n) combination.

    return optimal_m, optimal_n, min_k_xxx

def calculate_shear_buckling_coefficient(alpha):
    """Calculate shear buckling coefficient based on aspect ratio"""
    if alpha >= 1:
        return 5.34 + 4/(alpha**2)
    else:
        return 4 + 5.34/(alpha**2)

def calculate_critical_stress(k, t, b):
    """Calculate critical stress using the general formula"""
    return k * (np.pi**2 * E) / (12 * (1 - nu**2)) * (t/b)**2

def solve_combined_reserve_factor(sigma_avg_xx, tau_avg_xy, sigma_x_cr, tau_xy_cr):
    """Solve the quadratic equation for combined reserve factor"""
    # Calculate coefficients for the quadratic equation A*RF² + B*RF - 1 = 0
    B = abs(sigma_avg_xx) / sigma_x_cr
    A = (abs(tau_avg_xy) / tau_xy_cr)**2
    
    # Solve quadratic equation: A*RF² + B*RF - 1 = 0
    # Using quadratic formula: RF = (-B + sqrt(B² + 4A)) / (2A)
    if A == 0:  # Pure biaxial case
        if B != 0:
            return 1.0 / B
        else:
            return float('inf')
    
    discriminant = B**2 + 4*A
    if discriminant < 0:
        return float('inf')
    
    rf = (-B + np.sqrt(discriminant)) / (2*A)
    return rf

def calculate_volume_averaged_stresses(panel_elements):
    """Calculate volume-averaged stresses for a panel (3 elements)"""
    total_volume = len(panel_elements) * shell_element_volume
    
    sigma_avg_xx = sum(row['XX'] * shell_element_volume for _, row in panel_elements.iterrows()) / total_volume
    sigma_avg_yy = sum(row['YY'] * shell_element_volume for _, row in panel_elements.iterrows()) / total_volume
    sigma_avg_xy = sum(row['XY'] * shell_element_volume for _, row in panel_elements.iterrows()) / total_volume
    
    return sigma_avg_xx, sigma_avg_yy, sigma_avg_xy

# Read the CSV files
print("Reading CSV files...")

# Read 1D stress data with flexible column handling
df_1d = pd.read_csv(csv_1d_file, skiprows=9)
print(f"1D CSV has {len(df_1d.columns)} columns")

# Handle different column structures for 1D data
if len(df_1d.columns) == 5:
    df_1d.columns = ['Elements', 'FileID', 'Loadcase', 'Step', 'Axial_Stress']
elif len(df_1d.columns) == 6:
    df_1d.columns = ['Elements', 'FileID', 'Loadcase', 'Step', 'Axial_Stress', 'Empty']
    df_1d = df_1d.drop('Empty', axis=1)
else:
    print(f"Unexpected number of columns in 1D CSV: {len(df_1d.columns)}")
    print("Columns:", df_1d.columns.tolist())

# Convert numeric columns safely
df_1d['Elements'] = pd.to_numeric(df_1d['Elements'], errors='coerce')
df_1d['Axial_Stress'] = pd.to_numeric(df_1d['Axial_Stress'], errors='coerce')

# Read 2D/3D stress data with flexible column handling
df_2d3d = pd.read_csv(csv_2d3d_file, skiprows=9)
print(f"2D/3D CSV has {len(df_2d3d.columns)} columns")

# Handle different column structures for 2D/3D data
if len(df_2d3d.columns) == 8:
    df_2d3d.columns = ['Elements', 'FileID', 'Loadcase', 'Step', 'Layer', 'XX', 'XY', 'YY']
elif len(df_2d3d.columns) == 9:
    df_2d3d.columns = ['Elements', 'FileID', 'Loadcase', 'Step', 'Layer', 'XX', 'XY', 'YY', 'Empty']
    df_2d3d = df_2d3d.drop('Empty', axis=1)
else:
    print(f"Unexpected number of columns in 2D/3D CSV: {len(df_2d3d.columns)}")
    print("Columns:", df_2d3d.columns.tolist())

# Convert numeric columns safely
df_2d3d['Elements'] = pd.to_numeric(df_2d3d['Elements'], errors='coerce')
df_2d3d['XX'] = pd.to_numeric(df_2d3d['XX'], errors='coerce')
df_2d3d['XY'] = pd.to_numeric(df_2d3d['XY'], errors='coerce')
df_2d3d['YY'] = pd.to_numeric(df_2d3d['YY'], errors='coerce')

# Drop any rows with NaN values that might have been created from conversion errors
df_1d = df_1d.dropna()
df_2d3d = df_2d3d.dropna()

print(f"Successfully loaded {len(df_1d)} rows of 1D data and {len(df_2d3d)} rows of 2D/3D data")

print("Panel Buckling Analysis with Biaxial Loading")
print("="*60)

# Calculate basic parameters
aspect_ratio = a/b  # α = a/b
print(f"Panel dimensions: {a} x {b} mm")
print(f"Skin thickness: {t} mm")
print(f"Aspect ratio (α = a/b): {aspect_ratio:.3f}")

# Process each load case
results = []

for loadcase in sorted(df_2d3d['Loadcase'].unique()):
    print(f"\n" + "="*80)
    print(f"LOAD CASE {loadcase}")
    print("="*80)
    
    lc_data = df_2d3d[df_2d3d['Loadcase'] == loadcase].sort_values('Elements')
    
    # Group elements into panels (every 3 elements form a panel)
    num_panels = len(lc_data) // 3
    
    print(f"{'Panel':<6} {'σ_avg,XX':<10} {'σ_avg,YY':<10} {'τ_avg,XY':<10} {'β':<8} {'k_xxx':<8} {'k_s':<8} {'σ_x,cr':<10} {'τ_xy,cr':<10} {'RF':<10} {'m':<4} {'n':<4}")
    print("-" * 120)
    
    panel_results = []
    
    for panel_id in range(num_panels):
        start_idx = panel_id * 3
        end_idx = start_idx + 3
        
        panel_elements = lc_data.iloc[start_idx:end_idx]
        
        # Step 1: Calculate volume-averaged stresses
        sigma_avg_xx, sigma_avg_yy, sigma_avg_xy = calculate_volume_averaged_stresses(panel_elements)
        
        # Step 2: Calculate biaxial stress ratio and critical biaxial buckling stress
        if sigma_avg_xx != 0:
            beta = sigma_avg_yy / sigma_avg_xx  # β = σ_avg,yy / σ_avg,xx
        else:
            beta = 0
        
        # Find optimal m, n for biaxial buckling
        optimal_m, optimal_n, k_xxx = find_optimal_mn_for_biaxial_buckling(aspect_ratio, beta)
        
        # Step 3: Calculate shear buckling coefficient
        k_s = calculate_shear_buckling_coefficient(aspect_ratio)
        
        # Calculate critical stresses
        sigma_x_cr = calculate_critical_stress(k_xxx, t, b)
        tau_xy_cr = calculate_critical_stress(k_s, t, b)
        
        # Step 4: Calculate combined reserve factor using interaction equation
        rf_combined = solve_combined_reserve_factor(sigma_avg_xx, sigma_avg_xy, sigma_x_cr, tau_xy_cr)
        
        print(f"{panel_id+1:<6} {sigma_avg_xx:<10.2f} {sigma_avg_yy:<10.2f} {sigma_avg_xy:<10.2f} "
              f"{beta:<8.3f} {k_xxx:<8.2f} {k_s:<8.2f} {sigma_x_cr:<10.1f} {tau_xy_cr:<10.1f} {rf_combined:<10.3f} "
              f"{optimal_m:<4} {optimal_n:<4}")
        
        panel_results.append({
            'LoadCase': loadcase,
            'Panel': panel_id + 1,
            'sigma_avg_xx': sigma_avg_xx,
            'sigma_avg_yy': sigma_avg_yy,
            'sigma_avg_xy': sigma_avg_xy,
            'beta': beta,
            'k_xxx': k_xxx,
            'k_s': k_s,
            'sigma_x_cr': sigma_x_cr,
            'tau_xy_cr': tau_xy_cr,
            'optimal_m': optimal_m,
            'optimal_n': optimal_n,
            'RF_combined': rf_combined,
            'Elements': list(panel_elements['Elements'])
        })
    
    # Find critical panel for this load case
    finite_rf_results = [r for r in panel_results if r['RF_combined'] != float('inf')]
    if finite_rf_results:
        min_rf = min(result['RF_combined'] for result in finite_rf_results)
        critical_panel = next(result for result in finite_rf_results if result['RF_combined'] == min_rf)
        
        print(f"\nCritical Panel: Panel {critical_panel['Panel']} (Elements {critical_panel['Elements']})")
        print(f"Optimal buckling mode: m={critical_panel['optimal_m']}, n={critical_panel['optimal_n']}")
        print(f"Minimum RF_combined: {min_rf:.4f}")
    
    results.extend(panel_results)

# Overall summary
print("\n" + "="*80)
print("OVERALL BIAXIAL BUCKLING ANALYSIS SUMMARY")
print("="*80)

finite_rf_values = [result['RF_combined'] for result in results if result['RF_combined'] != float('inf')]
if finite_rf_values:
    overall_min_rf = min(finite_rf_values)
    critical_result = next(result for result in results if result['RF_combined'] == overall_min_rf)

    print(f"Overall Minimum RF_combined: {overall_min_rf:.4f}")
    print(f"Critical Panel: Load Case {critical_result['LoadCase']}, Panel {critical_result['Panel']}")
    print(f"Critical Panel Elements: {critical_result['Elements']}")
    print(f"Critical Panel Stresses: σ_xx={critical_result['sigma_avg_xx']:.2f}, σ_yy={critical_result['sigma_avg_yy']:.2f}, τ_xy={critical_result['sigma_avg_xy']:.2f}")
    print(f"Critical Panel β = {critical_result['beta']:.3f}")
    print(f"Critical Buckling Coefficients: k_xxx={critical_result['k_xxx']:.2f}, k_s={critical_result['k_s']:.2f}")

    if overall_min_rf < 1.0:
        print("⚠️  WARNING: Combined buckling reserve factor < 1.0 indicates potential buckling failure!")
    else:
        print("✅ All panels have acceptable combined buckling reserve factors (RF > 1.0)")

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('Biaxial_Panel_Buckling_Results.csv', index=False)
print(f"\nDetailed results saved to: Biaxial_Panel_Buckling_Results.csv")
