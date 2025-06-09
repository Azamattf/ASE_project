import pandas as pd
import numpy as np

# --- 1. Parameters and Assumptions ---

# Material Properties
E = 65022.77  # E-modulus B-basis (MPa)
sigma_y = 490   # Yield strength (MPa)
nu = 0.34     # Poisson's ratio

# Geometric Parameters
L = 300         # Total length of one stringer (column length) in mm
stringer_pitch = 150 # Stringer spacing (mm)
num_stringers = 9
stringers_per_row = 3

# --- MODIFICATION: U-Section Stringer Dimensions based on user's text description ---
height = 25.0      # DIM1: Overall Height of the vertical flanges
thickness = 2.0    # DIM2: Uniform thickness
web_width = 20.0   # DIM3: Width of the bottom web
# DIM4 is ignored as per user instruction.

# --- 2. Calculation Functions ---

def calculate_crippling_coefficient(x_i):
    """Calculates crippling coefficient based on the formulas from the provided image."""
    if x_i < 0.4: return 1.0
    if 0.4 <= x_i <= 1.095: return 1.4 - 0.628 * x_i
    if 1.095 < x_i <= 1.633: return 0.78
    return 0.69 / (x_i**0.75)

# --- MODIFICATION: Crippling calculation rewritten for a U-Section ---
def calculate_stringer_crippling_stress():
    """
    Calculates crippling stress for the U-section by analyzing its 3 plate elements
    (bottom web + 2 vertical flanges) and finding the area-weighted average.
    """
    elements = []
    
    # Element 1: Bottom Web (supported on both sides by vertical flanges)
    b_web = web_width - thickness # Average width of the flat web portion
    K_web = 3.60  # K-factor for element supported on both sides
    x_web = (b_web / thickness) * np.sqrt(sigma_y / (K_web * E))
    alpha_web = calculate_crippling_coefficient(x_web)
    sigma_crip_web = alpha_web * sigma_y
    area_web = b_web * thickness
    elements.append({'sigma_crip': sigma_crip_web, 'area': area_web})

    # Elements 2 & 3: Two identical Vertical Flanges (supported on one side by bottom web)
    b_flange = height - (thickness / 2) # Average height of the flange from the web's centerline
    K_flange = 0.41 # K-factor for element supported on one side
    x_flange = (b_flange / thickness) * np.sqrt(sigma_y / (K_flange * E))
    alpha_flange = calculate_crippling_coefficient(x_flange)
    sigma_crip_flange = alpha_flange * sigma_y
    area_flange = b_flange * thickness
    elements.append({'sigma_crip': sigma_crip_flange, 'area': area_flange})
    elements.append({'sigma_crip': sigma_crip_flange, 'area': area_flange})

    # Calculate area-weighted average crippling stress
    total_weighted_stress = sum(el['sigma_crip'] * el['area'] for el in elements)
    total_area = sum(el['area'] for el in elements)
    sigma_crip_avg = total_weighted_stress / total_area

    final_sigma_crip = min(sigma_crip_avg, sigma_y)

    print("Stringer Crippling Analysis (U-Section):")
    print("="*60)
    print(f"Bottom Web (K=3.60):   χ={x_web:.3f}, α={alpha_web:.3f}, σ_crip={sigma_crip_web:.1f} MPa")
    print(f"Vertical Flange (K=0.41): χ={x_flange:.3f}, α={alpha_flange:.3f}, σ_crip={sigma_crip_flange:.1f} MPa")
    print(f"Area-Weighted Average Crippling Stress (σ_crip_avg): {sigma_crip_avg:.2f} MPa")
    print(f"Final Section Crippling Stress (limited by yield): {final_sigma_crip:.2f} MPa")
    print()
    return final_sigma_crip

# --- MODIFICATION: Section properties rewritten for a U-Section ---
def calculate_combined_section_properties(t_skin):
    """Calculates geometric properties for the combined U-section stringer + skin."""
    skin_eff_area = stringer_pitch * t_skin
    
    # U-Section composed of a bottom web and two vertical flanges
    base_web_area = web_width * thickness
    # Area of vertical flanges excluding the part overlapping the base web
    vertical_flange_area = (height - thickness) * thickness
    stringer_area = base_web_area + 2 * vertical_flange_area
    total_area = skin_eff_area + stringer_area
    
    # Centroid calculation (z=0 at the skin mid-plane)
    z_skin = 0
    # Centroid of the U-section stringer relative to its own bottom edge
    z_c_stringer_local = ( (base_web_area * (thickness/2)) + 2 * (vertical_flange_area * (thickness + (height-thickness)/2)) ) / stringer_area
    # Centroid of stringer in the combined section's global coordinate system
    z_c_stringer_global = (t_skin/2) + z_c_stringer_local
    z_centroid_global = (skin_eff_area * z_skin + stringer_area * z_c_stringer_global) / total_area

    # Parallel Axis Theorem for Moment of Inertia (I_yy)
    I_skin = (stringer_pitch * t_skin**3)/12 + skin_eff_area * (z_skin - z_centroid_global)**2
    I_c_base_web = (web_width * thickness**3) / 12
    I_c_vert_flange = (thickness * (height-thickness)**3) / 12
    I_c_stringer = I_c_base_web + 2*I_c_vert_flange # This is simplified, needs parallel axis theorem for stringer parts
    # Full I_c for stringer about its own centroid
    I_c_stringer_full = (I_c_base_web + base_web_area * ((thickness/2) - z_c_stringer_local)**2) + \
                        2 * (I_c_vert_flange + vertical_flange_area * ((thickness + (height-thickness)/2) - z_c_stringer_local)**2)
    I_stringer_global = I_c_stringer_full + stringer_area * (z_c_stringer_global - z_centroid_global)**2
    I_comb = I_skin + I_stringer_global
    
    r_gyr = np.sqrt(I_comb / total_area)
    lambda_slenderness = (1.0 * L) / r_gyr
    
    print("Combined Section Properties (U-Section):")
    print("="*60)
    print(f"Moment of Inertia (I_comb): {I_comb:.1f} mm⁴")
    print(f"Total Area (total_area): {stringer_area:.1f} mm²")
    print(f"Radius of Gyration (r_gyr): {r_gyr:.2f} mm")
    print(f"Slenderness (lambda): {lambda_slenderness:.1f}")
    print()
    
    stringer_element_vol = stringer_area * (L / stringers_per_row)
    
    return I_comb, r_gyr, lambda_slenderness, stringer_element_vol

def calculate_ejc_stress_and_lambda_crit(lambda_val, compressive_limit):
    lambda_crit = np.sqrt(2 * np.pi**2 * E / compressive_limit)
    if lambda_val > lambda_crit: sigma_cr = (np.pi**2 * E) / (lambda_val**2)
    else: sigma_cr = compressive_limit - (1/E) * (compressive_limit/(2*np.pi))**2 * lambda_val**2
    return sigma_cr, lambda_crit

def calculate_volume_averaged_stress(stresses_1d, stresses_2d_left, stresses_2d_right, str_elem_vol, skin_elem_vol):
    stress_sum_1d = sum(stresses_1d['Axial_Stress']) * str_elem_vol
    stress_sum_2d_left = sum(stresses_2d_left['XX']) * skin_elem_vol
    stress_sum_2d_right = sum(stresses_2d_right['XX']) * skin_elem_vol
    total_stress_moment = stress_sum_1d + 0.5 * stress_sum_2d_left + 0.5 * stress_sum_2d_right
    total_volume = (3 * str_elem_vol) + 0.5 * (3 * skin_elem_vol) + 0.5 * (3 * skin_elem_vol)
    return total_stress_moment / total_volume if total_volume > 0 else 0

# --- 3. Main Execution ---
try:
    df_1d = pd.read_csv('ProjectElementStresses1D_Amjad.csv', skiprows=9, header=None, usecols=[0,2,4], names=['Elements', 'Loadcase', 'Axial_Stress'])
    df_2d = pd.read_csv('ProjectElementStresses2D3D_Amjad.csv', skiprows=9, header=None, usecols=[0,2,5], names=['Elements', 'Loadcase', 'XX'])
    df_1d = df_1d.apply(pd.to_numeric, errors='coerce').dropna()
    df_2d = df_2d.apply(pd.to_numeric, errors='coerce').dropna()
    print("Successfully loaded 1D and 2D stress data.")
except FileNotFoundError as e:
    print(f"⚠️ Error: Could not find a required data file: {e.filename}")
    exit()

# Assuming a skin thickness for the combined property calculation
t_skin = 4.0
shell_element_volume = (600000 / 3) # From 1.1e script

sigma_crip = calculate_stringer_crippling_stress()
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
        sigma_cutoff = min(sigma_y, sigma_crip)
        sigma_cr, lambda_crit = calculate_ejc_stress_and_lambda_crit(lambda_slenderness, sigma_cutoff)
        rf = sigma_cr / abs(sig_avg) if sig_avg < 0 else float('inf')
            
        results.append({
            'LoadCase': int(loadcase), 'Stringer': stringer_num, 'sig_axial,comb,avg': sig_avg,
            'sig_crip': sigma_crip, 'I_comb': I_comb, 'r_gyr': r_gyr, 'lambda': lambda_slenderness,
            'lambda_crit': lambda_crit, 'RF_columnbuckl_comb': rf
        })

if results:
    results_df = pd.DataFrame(results)
    output_filename = 'Submission_Column_Buckling_U_Section.csv'
    results_df.to_csv(output_filename, index=False, float_format='%.3f')
    print(f"\n\n✅ Combined column buckling results for U-Section saved to: {output_filename}")