import pandas as pd
import numpy as np
import sys

# Check command line arguments
if len(sys.argv) != 5:
    print("Usage: python script_f_p11.py <1D_csv_file> <2D3D_csv_file> <E_modulus> <yield_strength>")
    sys.exit(1)

csv_1d_file = sys.argv[1]
csv_2d3d_file = sys.argv[2]
E = float(sys.argv[3])  # E-modulus from command line argument
sigma_y = float(sys.argv[4])  # Yield strength from command line argument

# --- 1. Parameters and Assumptions ---

# Material Properties
nu = 0.34      # Poisson's ratio
FoS = 1.5      # Factor of Safety

# Geometric Parameters
L = 750        # Total length of one stringer (column length) in mm
stringer_pitch = 200 # Stringer spacing (mm)
num_stringers = 9
stringers_per_row = 3
t_skin = 4.0   # Skin thickness is constant for this task

# Lipped C-Channel Stringer Dimensions
height = 25.0      # Overall Height of the stringer's vertical flanges
thickness = 2.0    # Uniform thickness of the stringer
web_width = 20.0   # Width of the stringer's web
lip_width = 15.0   # Width of the lips

# --- 2. Calculation Functions ---

def calculate_combined_section_properties(t_skin):
    """
    Calculates geometric properties for the combined lipped C-channel stringer + effective skin.
    """
    w_eff_skin = stringer_pitch
    h_flange = height - thickness

    parts = {
        'skin':    {'w': w_eff_skin, 'h': t_skin, 'z_local': t_skin / 2.0},
        'web':     {'w': web_width, 'h': thickness, 'z_local': t_skin + height - (thickness / 2.0)},
        'flange':  {'w': thickness, 'h': h_flange, 'z_local': t_skin + (h_flange / 2.0)},
        'lip':     {'w': lip_width, 'h': thickness, 'z_local': t_skin + (thickness / 2.0)}
    }

    total_area = 0
    first_moment_of_area = 0
    for name, p in parts.items():
        p['area'] = p['w'] * p['h']
        p['I_local'] = (p['w'] * p['h']**3) / 12.0
        num = 2 if name in ['flange', 'lip'] else 1
        total_area += num * p['area']
        first_moment_of_area += num * p['area'] * p['z_local']

    z_c = first_moment_of_area / total_area

    I_comb = 0
    for name, p in parts.items():
        d = p['z_local'] - z_c
        num = 2 if name in ['flange', 'lip'] else 1
        I_comb += num * (p['I_local'] + p['area'] * d**2)

    stringer_area = parts['web']['area'] + 2*parts['flange']['area'] + 2*parts['lip']['area']
    r_gyr = np.sqrt(I_comb / total_area)
    lambda_slenderness = (1.0 * L) / r_gyr

    print("Combined Section Properties (Lipped C-Channel - Corrected):")
    print("="*60)
    print(f"Total Area (A_total):         {total_area:.5f} mm²")
    print(f"Centroid Location (z_c):      {z_c:.5f} mm (from top of skin)")
    print(f"Moment of Inertia (I_comb):   {I_comb:.5f} mm⁴")
    print(f"Radius of Gyration (r_gyr):   {r_gyr:.5f} mm")
    print(f"Slenderness (λ):              {lambda_slenderness:.5f}")
    print()

    stringer_element_vol = stringer_area * (L / stringers_per_row)

    return I_comb, r_gyr, lambda_slenderness, stringer_element_vol


def calculate_ejc_stress_and_lambda_crit(lambda_val, compressive_limit):
    lambda_crit = np.sqrt(2 * np.pi**2 * E / compressive_limit)
    if lambda_val > lambda_crit:
        sigma_cr = (np.pi**2 * E) / (lambda_val**2)
    else:
        sigma_cr = compressive_limit - (1/E) * (compressive_limit/(2*np.pi))**2 * lambda_val**2
    return sigma_cr, lambda_crit

def calculate_volume_averaged_stress(stresses_1d, stresses_2d_left, stresses_2d_right, str_elem_vol, skin_elem_vol):
    stress_sum_1d = sum(stresses_1d['Axial_Stress']) * str_elem_vol
    stress_sum_2d_left = sum(stresses_2d_left['XX']) * skin_elem_vol
    stress_sum_2d_right = sum(stresses_2d_right['XX']) * skin_elem_vol
    total_stress_moment = stress_sum_1d + 0.5 * stress_sum_2d_left + 0.5 * stress_sum_2d_right
    total_volume = (len(stresses_1d) * str_elem_vol) + 0.5 * (len(stresses_2d_left) * skin_elem_vol) + 0.5 * (len(stresses_2d_right) * skin_elem_vol)
    return total_stress_moment / total_volume if total_volume > 0 else 0

# --- 3. Main Execution ---
try:
    df_1d = pd.read_csv(csv_1d_file, skiprows=9, header=None, usecols=[0,2,4], names=['Elements', 'Loadcase', 'Axial_Stress'])
    df_2d = pd.read_csv(csv_2d3d_file, skiprows=9, header=None, usecols=[0,2,5], names=['Elements', 'Loadcase', 'XX'])
    df_1d = df_1d.apply(pd.to_numeric, errors='coerce').dropna()
    df_2d = df_2d.apply(pd.to_numeric, errors='coerce').dropna()
    print("Successfully loaded 1D and 2D stress data.\n")
except FileNotFoundError as e:
    print(f"⚠️ Error: Could not find a required data file: {e.filename}")
    sys.exit(1)

shell_element_volume = 200000.0

# Perform section calculations once
I_comb, r_gyr, lambda_slenderness, stringer_element_volume = calculate_combined_section_properties(t_skin)

results = []
for loadcase in sorted(df_1d['Loadcase'].unique()):
    lc_1d_data = df_1d[df_1d['Loadcase'] == loadcase]
    lc_2d_data = df_2d[df_2d['Loadcase'] == loadcase]

    for i in range(num_stringers):
        stringer_num = i + 1
        stringer_element_ids = [40 + i*3, 41 + i*3, 42 + i*3]
        left_panel_element_ids = [1 + i*3, 2 + i*3, 3 + i*3]
        right_panel_element_ids = [4 + i*3, 5 + i*3, 6 + i*3]

        stresses_1d = lc_1d_data[lc_1d_data['Elements'].isin(stringer_element_ids)]
        stresses_2d_left = lc_2d_data[lc_2d_data['Elements'].isin(left_panel_element_ids)]
        stresses_2d_right = lc_2d_data[lc_2d_data['Elements'].isin(right_panel_element_ids)]

        if len(stresses_1d) < 3 or len(stresses_2d_left) < 3 or len(stresses_2d_right) < 3: continue

        sig_avg = calculate_volume_averaged_stress(stresses_1d, stresses_2d_left, stresses_2d_right, stringer_element_volume, shell_element_volume)

        # Set the cutoff stress directly to the yield strength
        sigma_cutoff = sigma_y

        sigma_cr, lambda_crit = calculate_ejc_stress_and_lambda_crit(lambda_slenderness, sigma_cutoff)

        # --- MODIFICATION: Apply 1.5 Factor of Safety to the applied stress ---
        # The reserve factor is calculated against the ULTIMATE load (FoS * limit load)
        if sig_avg < 0: # Buckling is only relevant for compressive loads
            rf = sigma_cr / (FoS * abs(sig_avg))
        else:
            rf = float('inf')

        results.append({
            'LoadCase': int(loadcase),
            'Stringer': stringer_num,
            'sig_axial,comb,avg': sig_avg,
            'lambda': lambda_slenderness,
            'lambda_crit': lambda_crit,
            'RF_ult_columnbuckl': rf
        })

if results:
    results_df = pd.DataFrame(results)
    output_filename = 'Submission_Column_Buckling_Final_with_FoS.csv'
    results_df.to_csv(output_filename, index=False, float_format='%.3f')
    print(f"\n\n✅ Final column buckling results (with FoS) saved to: {output_filename}")
