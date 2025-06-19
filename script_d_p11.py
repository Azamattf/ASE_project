import pandas as pd
import numpy as np
import sys

# Check command line arguments
if len(sys.argv) != 4:
    print("Usage: python script_d_p11.py <1D_csv_file> <2D3D_csv_file> <sigma_max>")
    sys.exit(1)

csv_1d_file = sys.argv[1]
csv_2d3d_file = sys.argv[2]
sigma_max = float(sys.argv[3])

def calculate_reserve_factor(stress_value, sigma_max):
    """Calculate reserve factor RF = sigma_max / |stress|"""
    return sigma_max / (1.5 * abs(stress_value)) if stress_value != 0 else float('inf')

def calculate_von_mises_stress(sigma_xx, sigma_yy, tau_xy):
    """Calculate Von Mises stress using the formula: sqrt(σ_XX² - σ_XX*σ_YY + σ_YY² + 3*τ_XY²)"""
    return np.sqrt(sigma_xx**2 - sigma_xx*sigma_yy + sigma_yy**2 + 3*tau_xy**2)

# Read the CSV files
print("Reading CSV files...")

# Read 1D stress data
df_1d = pd.read_csv(csv_1d_file, skiprows=9)  # Skip header rows
print(f"1D CSV has {len(df_1d.columns)} columns")
print("First few rows of 1D data:")
print(df_1d.head())

# Assign column names based on actual number of columns
if len(df_1d.columns) == 6:
    df_1d.columns = ['Elements', 'FileID', 'Loadcase', 'Step', 'Axial_Stress', 'Empty']
    df_1d = df_1d.drop('Empty', axis=1)
elif len(df_1d.columns) == 5:
    df_1d.columns = ['Elements', 'FileID', 'Loadcase', 'Step', 'Axial_Stress']
else:
    print(f"Unexpected number of columns in 1D CSV: {len(df_1d.columns)}")
    print("Column names will be auto-assigned")

# Convert numeric columns to float
df_1d['Axial_Stress'] = pd.to_numeric(df_1d['Axial_Stress'], errors='coerce')

# Remove rows with NaN values
df_1d = df_1d.dropna(subset=['Axial_Stress'])
print(f"1D data after cleaning: {len(df_1d)} rows")

# Read 2D/3D stress data  
df_2d3d = pd.read_csv(csv_2d3d_file, skiprows=9)  # Skip header rows
print(f"2D3D CSV has {len(df_2d3d.columns)} columns")
print("First few rows of 2D3D data:")
print(df_2d3d.head())

# Assign column names based on actual number of columns
if len(df_2d3d.columns) == 9:
    df_2d3d.columns = ['Elements', 'FileID', 'Loadcase', 'Step', 'Layer', 'XX', 'XY', 'YY', 'Empty']
    df_2d3d = df_2d3d.drop('Empty', axis=1)
elif len(df_2d3d.columns) == 8:
    df_2d3d.columns = ['Elements', 'FileID', 'Loadcase', 'Step', 'Layer', 'XX', 'XY', 'YY']
else:
    print(f"Unexpected number of columns in 2D3D CSV: {len(df_2d3d.columns)}")
    print("Column names will be auto-assigned")

# Convert numeric columns to float
df_2d3d['XX'] = pd.to_numeric(df_2d3d['XX'], errors='coerce')
df_2d3d['XY'] = pd.to_numeric(df_2d3d['XY'], errors='coerce')
df_2d3d['YY'] = pd.to_numeric(df_2d3d['YY'], errors='coerce')

# Remove rows with NaN values in stress columns
df_2d3d = df_2d3d.dropna(subset=['XX', 'XY', 'YY'])
print(f"2D3D data after cleaning: {len(df_2d3d)} rows")

print("Processing 1D Axial Stress Data...")
print("="*50)

# Process 1D stress data
df_1d['Reserve_Factor'] = df_1d['Axial_Stress'].apply(lambda x: calculate_reserve_factor(x, sigma_max))

# Group by load case and display results
for loadcase in sorted(df_1d['Loadcase'].unique()):
    lc_data = df_1d[df_1d['Loadcase'] == loadcase]
    print(f"\nLoad Case {loadcase}:")
    print(f"{'Element':<8} {'Axial Stress (MPa)':<18} {'Reserve Factor':<15}")
    print("-" * 45)
    
    for _, row in lc_data.iterrows():
        print(f"{row['Elements']:<8} {row['Axial_Stress']:<18.5f} {row['Reserve_Factor']:<15.5f}")
    
    # Summary statistics
    if len(df_1d) == 0:
        print("No valid 1D stress data found!")
    else:
        min_rf = lc_data['Reserve_Factor'].min()
        if pd.isna(min_rf):
            print("No valid reserve factors calculated for this load case")
        else:
            critical_element = lc_data.loc[lc_data['Reserve_Factor'].idxmin(), 'Elements']
            print(f"\nMinimum Reserve Factor: {min_rf:.5f} (Element {critical_element})")

print("\n" + "="*70)
print("Processing 2D/3D Stress Data...")
print("="*70)

# Process 2D/3D stress data
df_2d3d['Von_Mises_Stress'] = calculate_von_mises_stress(
    df_2d3d['XX'], df_2d3d['YY'], df_2d3d['XY']
)
df_2d3d['Reserve_Factor'] = df_2d3d['Von_Mises_Stress'].apply(
    lambda x: calculate_reserve_factor(x, sigma_max)
)

# Group by load case and display results
for loadcase in sorted(df_2d3d['Loadcase'].unique()):
    lc_data = df_2d3d[df_2d3d['Loadcase'] == loadcase]
    print(f"\nLoad Case {loadcase}:")
    print(f"{'Element':<8} {'σ_XX':<8} {'τ_XY':<8} {'σ_YY':<8} {'Von Mises':<12} {'Reserve Factor':<15}")
    print("-" * 70)
    
    for _, row in lc_data.iterrows():
        print(f"{row['Elements']:<8} {row['XX']:<8.1f} {row['XY']:<8.1f} {row['YY']:<8.1f} "
              f"{row['Von_Mises_Stress']:<12.5f} {row['Reserve_Factor']:<15.5f}")
    
    # Summary statistics
    if len(df_2d3d) == 0:
        print("No valid 2D/3D stress data found!")
    else:
        min_rf = lc_data['Reserve_Factor'].min()
        if pd.isna(min_rf):
            print("No valid reserve factors calculated for this load case")
        else:
            critical_element = lc_data.loc[lc_data['Reserve_Factor'].idxmin(), 'Elements']
            print(f"\nMinimum Reserve Factor: {min_rf:.5f} (Element {critical_element})")

print("\n" + "="*70)
print("OVERALL SUMMARY")
print("="*70)

# Overall critical analysis
all_1d_rf = df_1d['Reserve_Factor'].min()
all_2d3d_rf = df_2d3d['Reserve_Factor'].min()

print(f"Minimum Reserve Factor in 1D Elements: {all_1d_rf:.5f}")
print(f"Minimum Reserve Factor in 2D/3D Elements: {all_2d3d_rf:.5f}")

overall_min = min(all_1d_rf, all_2d3d_rf)
print(f"\nOverall Minimum Reserve Factor: {overall_min:.5f}")

if overall_min < 1.0:
    print("⚠️  WARNING: Reserve factor < 1.0 indicates potential failure!")
else:
    print("✅ All elements have acceptable reserve factors (RF > 1.0)")

# Save results to CSV files
df_1d.to_csv('1D_Stress_Results_with_RF.csv', index=False, float_format='%.5f')
df_2d3d.to_csv('2D3D_Stress_Results_with_RF.csv', index=False, float_format='%.5f')

print(f"\nResults saved to:")
print(f"- 1D_Stress_Results_with_RF.csv")
print(f"- 2D3D_Stress_Results_with_RF.csv")
