import numpy as np
import pandas as pd

# It is assumed that the main script will have imported and made available
# the necessary dataframes and constants in the global namespace.
# It is also assumed that a function 'calculate_abd_matrix' is available.

def combined_buckling_analysis(
    axial_stress_df, element_stresses_df,
    A_skin, D_skin,
    A_flange, D_flange,
    A_web_inv, D_web_inv,
    constants
):
    """
    Performs a combined skin-stringer column buckling analysis FOR EACH STRINGER.

    This function calculates the reserve factor against column buckling for each of the 4
    composite T-stiffeners bonded to a skin panel.

    Args:
        axial_stress_df (pd.DataFrame): DataFrame with axial stresses for ALL stringer elements.
        element_stresses_df (pd.DataFrame): DataFrame with in-plane stresses for ALL skin elements.
        A_skin, D_skin (np.array): ABD matrices for the skin.
        A_flange, D_flange (np.array): ABD matrices for the stringer flange.
        A_web_inv, D_web_inv (np.array): Inverse A and D matrices for the stringer web.
        constants (dict): A dictionary containing all required geometric and material constants.

    Returns:
        pd.DataFrame: A DataFrame containing the buckling analysis results for each of the 4 stringers.
    """
    print("\nStarting Combined Skin-Stringer Buckling Analysis (Task e)...")

    # --- 1. UNPACK GEOMETRY AND CONSTANTS ---
    PANEL_LENGTH_A = constants['PANEL_LENGTH_A']
    PANEL_WIDTH_B = constants['PANEL_WIDTH_B']
    SKIN_THICKNESS = constants['SKIN_THICKNESS']
    STRINGER_PLY_THICKNESS = constants['STRINGER_PLY_THICKNESS']
    SAFETY_FACTOR = constants['SAFETY_FACTOR']
    KNOCKDOWN_FACTOR = constants['KNOCKDOWN_FACTOR']
    SIGMA_U_C = 650.0

    # --- 2. GEOMETRY SETUP (CORRECTED) ---
    # Coordinate system: Origin at BOTTOM surface of skin panel
    # y-axis: horizontal along panel width
    # z-axis: vertical, POSITIVE UPWARDS from skin's bottom surface
    
    skin_eff_width = PANEL_WIDTH_B
    skin_thickness = SKIN_THICKNESS
    flange_width = 70.0
    web_height = 40.0
    flange_thickness = 16 * STRINGER_PLY_THICKNESS  # Should be 4mm total
    web_thickness = 16 * STRINGER_PLY_THICKNESS     # Should be 4mm total

    # --- 3. HOMOGENIZED ENGINEERING CONSTANTS (WITH KNOCKDOWN FACTOR) ---
    # Apply knockdown factor to all elastic moduli at the beginning
    # Skin Properties (Restricted Lateral Deformation)
    E_x_skin = (A_skin[0, 0] / skin_thickness) * KNOCKDOWN_FACTOR
    E_b_z_skin = (D_skin[0, 0] * 12 / skin_thickness**3) * KNOCKDOWN_FACTOR
    
    # Flange Properties (Restricted Lateral Deformation)  
    E_x_flange = (A_flange[0, 0] / flange_thickness) * KNOCKDOWN_FACTOR
    E_b_z_flange = (D_flange[0, 0] * 12 / flange_thickness**3) * KNOCKDOWN_FACTOR
    
    # Web Properties (Free in-plane, Restricted out-of-plane)
    E_x_web = (1 / (A_web_inv[0, 0] * web_thickness)) * KNOCKDOWN_FACTOR
    E_b_y_web = (12 / (D_web_inv[0, 0] * web_thickness**3)) * KNOCKDOWN_FACTOR

    # --- 4. SEGMENT PROPERTIES CALCULATION ---
    # Segment 1: Skin (effective width under stringer)
    area_skin = skin_eff_width * skin_thickness
    I_y_skin = (skin_eff_width * skin_thickness**3) / 12.0
    I_z_skin = (skin_thickness * skin_eff_width**3) / 12.0
    y_c_skin = 0.0
    z_c_skin = skin_thickness / 2.0  # Centroid at half thickness from bottom

    # Segment 2: Flange (sits on top of skin)
    area_flange = flange_width * flange_thickness
    I_y_flange = (flange_width * flange_thickness**3) / 12.0
    I_z_flange = (flange_thickness * flange_width**3) / 12.0
    y_c_flange = 0.0
    z_c_flange = skin_thickness + (flange_thickness / 2.0)  # Sits above skin

    # Segment 3: Web (extends upward from flange)
    area_web = web_height * web_thickness
    I_y_web = (web_thickness * web_height**3) / 12.0
    I_z_web = (web_height * web_thickness**3) / 12.0
    y_c_web = 0.0
    z_c_web = skin_thickness + flange_thickness + (web_height / 2.0)  # Sits above flange

    area_total = area_skin + area_flange + area_web

    # --- 5. ELASTIC CENTER (EC) CALCULATION ---
    y_ec_num = (E_x_skin * area_skin * y_c_skin) + \
               (E_x_flange * area_flange * y_c_flange) + \
               (E_x_web * area_web * y_c_web)
    z_ec_num = (E_x_skin * area_skin * z_c_skin) + \
               (E_x_flange * area_flange * z_c_flange) + \
               (E_x_web * area_web * z_c_web)
    ec_den = (E_x_skin * area_skin) + (E_x_flange * area_flange) + (E_x_web * area_web)

    y_EC = y_ec_num / ec_den
    z_EC = z_ec_num / ec_den

    # --- 6. COMBINED BENDING STIFFNESS (EI) CALCULATION ---
    # Distances from combined EC to each segment's centroid
    y_skin_EC = y_c_skin - y_EC
    z_skin_EC = z_c_skin - z_EC
    y_flange_EC = y_c_flange - y_EC
    z_flange_EC = z_c_flange - z_EC
    y_web_EC = y_c_web - y_EC
    z_web_EC = z_c_web - z_EC

    # (EI)_y: Bending about y-axis (weak axis bending of stringer)
    # Skin/Flange bend out-of-plane. Web bends in-plane (stiff).
    # Parallel axis terms always use axial modulus E_x.
    EI_y = (E_b_z_skin * I_y_skin + E_x_skin * area_skin * z_skin_EC**2) + \
           (E_b_z_flange * I_y_flange + E_x_flange * area_flange * z_flange_EC**2) + \
           (E_x_web * I_y_web + E_x_web * area_web * z_web_EC**2)

    # (EI)_z: Bending about z-axis (strong axis bending of stringer)  
    # Skin/Flange bend in-plane. Web bends out-of-plane (weak).
    EI_z = (E_x_skin * I_z_skin + E_x_skin * area_skin * y_skin_EC**2) + \
           (E_x_flange * I_z_flange + E_x_flange * area_flange * y_flange_EC**2) + \
           (E_b_y_web * I_z_web + E_x_web * area_web * y_web_EC**2)

    # Calculate total moment of inertia for radius of gyration
    I_y_total = (I_y_skin + area_skin * z_skin_EC**2) + (I_y_flange + area_flange * z_flange_EC**2) + (I_y_web + area_web * z_web_EC**2)
    I_z_total = (I_z_skin + area_skin * y_skin_EC**2) + (I_z_flange + area_flange * y_flange_EC**2) + (I_z_web + area_web * y_web_EC**2)

    # --- 7. BUCKLING ANALYSIS CALCULATIONS ---
    EI_min, I_min = (EI_y, I_y_total) if EI_y < EI_z else (EI_z, I_z_total)

    sigma_crippling = (1.63 / ((web_height / web_thickness)**0.717)) * SIGMA_U_C
    E_avg = EI_min / I_min
    r = np.sqrt(I_min / area_total)
    lambda_slenderness = (1.0 * PANEL_LENGTH_A) / r
    
    # Corrected lambda_cr calculation - transition slenderness for Johnson-Euler formula
    # lambda_cr = sqrt(2*pi^2*E / sigma_crippling) is the typical form
    lambda_cr = np.sqrt((2 * np.pi**2 * E_avg) / sigma_crippling)

    # Johnson-Euler formula for critical stress
    if lambda_slenderness < lambda_cr:
        # Johnson formula (short column)
        sigma_crit = sigma_crippling - (1 / E_avg) * (sigma_crippling / (2 * np.pi))**2 * lambda_slenderness**2
    else:
        # Euler formula (long column)
        sigma_crit = (np.pi**2 * E_avg) / lambda_slenderness**2

    # Debug output to verify calculations
    print(f"\n--- GEOMETRY VERIFICATION ---")
    print(f"Skin thickness: {skin_thickness:.3f} mm")
    print(f"Flange thickness: {flange_thickness:.3f} mm") 
    print(f"Web thickness: {web_thickness:.3f} mm")
    print(f"z_c_skin: {z_c_skin:.3f} mm")
    print(f"z_c_flange: {z_c_flange:.3f} mm")
    print(f"z_c_web: {z_c_web:.3f} mm")
    print(f"z_EC: {z_EC:.3f} mm")
    print(f"E_x_skin (with knockdown): {E_x_skin:.1f} MPa")
    print(f"E_x_flange (with knockdown): {E_x_flange:.1f} MPa")
    print(f"EI_y: {EI_y:.2f} N*mm^2")
    print(f"EI_z: {EI_z:.2f} N*mm^2")
    print(f"EI_min: {EI_min:.2f} N*mm^2")
    print(f"r_gyr: {r:.2f} mm")
    print(f"lambda: {lambda_slenderness:.1f}")
    print(f"lambda_cr: {lambda_cr:.1f}")
    print(f"sigma_crippling: {sigma_crippling:.1f} MPa")

    # --- 3. LOOP THROUGH EACH STRINGER TO CALCULATE APPLIED STRESS AND RF ---
    all_results = []
    num_stringers = 4
    skin_elems_per_bay = 6
    
    # Define the specific, non-sequential element IDs for each stringer as provided.
    stringer_id_map = {
        1: [40, 41, 42],
        2: [46, 47, 48],
        3: [52, 53, 54],
        4: [58, 59, 60]
    }

    print("\n--- Combined Column Buckling Results (Task e) ---")
    print("\n--- Common Properties for All Stringer-Skin Columns ---")
    print(f"  - Homogenized Bending Modulus (E_b_z_skin):    {E_b_z_skin:.2f} MPa")
    print(f"  - Homogenized Bending Modulus (E_b_z_flange):  {E_b_z_flange:.2f} MPa")
    print(f"  - Homogenized Bending Modulus (E_b_y_web):     {E_b_y_web:.2f} MPa")
    print(f"  - Elastic Center (z_EC,comb):                  {z_EC:.2f} mm")
    print(f"  - Minimum Bending Stiffness (EI_comb):         {EI_min:.2f} N*mm^2")
    print(f"  - Radius of Gyration (r_gyr):                  {r:.2f} mm")
    print(f"  - Column Slenderness (lambda):                 {lambda_slenderness:.2f}")
    print(f"  - Transition Slenderness (lambda_crit):        {lambda_cr:.2f}")
    print("-" * 50)

    # Get unique load cases from the input dataframes
    unique_load_cases = sorted(axial_stress_df['Load_Case'].unique())

    for load_case in unique_load_cases:
        print(f"\n--- Analyzing Load Case: {load_case} ---")
        for i in range(num_stringers):
            stringer_id_num = i + 1
            
            # Get the specific element IDs for the current stringer
            stringer_element_ids = stringer_id_map[stringer_id_num]
            
            # The skin panels are numbered 1-30. We associate one panel bay with each stringer.
            # Stringer 1 -> Panel Bay 1 (Elements 1-6)
            # Stringer 2 -> Panel Bay 2 (Elements 7-12)
            # ... and so on.
            start_skin_elem = 1 + (i * skin_elems_per_bay)
            end_skin_elem = start_skin_elem + skin_elems_per_bay - 1
            skin_element_ids = list(range(start_skin_elem, end_skin_elem + 1))

            # Filter the dataframes for the relevant elements AND current load case
            stringer_stresses = axial_stress_df[
                (axial_stress_df['Element_ID'].isin(stringer_element_ids)) &
                (axial_stress_df['Load_Case'] == load_case)
            ]
            skin_stresses = element_stresses_df[
                (element_stresses_df['Element_ID'].isin(skin_element_ids)) &
                (element_stresses_df['Load_Case'] == load_case)
            ]
            
            if stringer_stresses.empty or skin_stresses.empty:
                print(f"Warning: Could not find stress data for all elements of Stringer {stringer_id_num} in Load Case {load_case}. Skipping.")
                continue

            # Calculate volume-weighted average stress for this specific column
            stringer_stress_avg = stringer_stresses['Longitudinal_Stress'].mean()
            skin_stress_avg = skin_stresses['Stress_XX'].mean()

            stringer_volume = (area_flange + area_web) * PANEL_LENGTH_A
            skin_volume = area_skin * PANEL_LENGTH_A
            total_volume = stringer_volume + skin_volume

            sigma_axial_combined_avg = ((stringer_stress_avg * stringer_volume) + (skin_stress_avg * skin_volume)) / total_volume
            sigma_axial_ult = sigma_axial_combined_avg * SAFETY_FACTOR

            # Calculate the final reserve factor for this stringer
            # RF is infinity for tensile loads as buckling is not a concern
            if sigma_axial_ult >= 0:
                RF_combined = float('inf')
            else:
                RF_combined = sigma_crit / abs(sigma_axial_ult)

            all_results.append({
                'Load_Case': load_case,
                'Stringer_ID': stringer_id_num,
                'sigma_axial_combined_avg': sigma_axial_combined_avg,
                'sigma_crippling': sigma_crippling,
                'sigma_crit': sigma_crit,
                'RF_combined': RF_combined
            })

    results_df = pd.DataFrame(all_results)
    print("\n--- Summary of Combined Column Buckling Results (per Stringer per Load Case) ---")
    print(results_df.to_string(index=False))
    
    return results_df
