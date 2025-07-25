import pandas as pd
import numpy as np

def perform_full_puck_analysis(
    composite_stress_x_df, 
    composite_stress_y_df, 
    composite_stress_xy_df,
    axial_strain_df,
    stringer_flange_layup_symmetric,
    stringer_ply_thickness,
    ud_ply_properties,
    SAFETY_FACTOR
):
    """
    Performs a corrected Puck failure analysis on a stiffened composite panel.

    This function operates on LIMIT load data from FEA and applies a safety
    factor to calculate the final reserve factors against ULTIMATE loads.

    Args:
        composite_stress_x_df (pd.DataFrame): Skin stress (sigma_1) at LIMIT load.
        composite_stress_y_df (pd.DataFrame): Skin stress (sigma_2) at LIMIT load.
        composite_stress_xy_df (pd.DataFrame): Skin stress (tau_12) at LIMIT load.
        axial_strain_df (pd.DataFrame): Stringer axial strain (epsilon_x) at LIMIT load.
        stringer_flange_layup_symmetric (list): Symmetric half-stack for the stringer flange.
        stringer_ply_thickness (float): Thickness of a single stringer ply.
        ud_ply_properties (dict): E1, E2, G12, v12 for the composite material.
        SAFETY_FACTOR (float): The required global safety factor (e.g., 1.5).

    Returns:
        pd.DataFrame: A comprehensive DataFrame with full Puck analysis results.
    """
    # --- 1. DEFINE CONSTANTS AND HELPER FUNCTIONS ---
    strengths = {
        "R_para_t": 3050, "R_para_c": 1500, "R_perp_t": 300,
        "R_perp_c": 50, "R_perp_para": 100
    }
    puck_params = {
        "p_perp_para_plus": 0.25, "p_perp_para_minus": 0.25,
        "p_perp_perp_minus": 0.25
    }

    def get_Q_matrix(properties):
        E1, E2, G12, v12 = properties["E1"], properties["E2"], properties["G12"], properties["v12"]
        v21 = v12 * E2 / E1
        denominator = 1 - v12 * v21
        Q11 = E1 / denominator
        Q22 = E2 / denominator
        Q12 = v12 * E2 / denominator
        Q66 = G12
        return np.array([[Q11, Q12, 0], [Q12, Q22, 0], [0, 0, Q66]])

    def get_T_epsilon_matrix(theta_deg):
        theta_rad = np.deg2rad(theta_deg)
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        return np.array([
            [c**2, s**2, c*s],
            [s**2, c**2, -c*s],
            [-2*c*s, 2*c*s, c**2 - s**2]
        ])

    # --- 2. CALCULATE STRINGER PLY STRESSES AT ULTIMATE LOAD ---
    print("--- Calculating Stringer ULTIMATE Stresses ---")
    Q_matrix = get_Q_matrix(ud_ply_properties)
    full_stringer_layup = stringer_flange_layup_symmetric + stringer_flange_layup_symmetric[::-1]
    
    stringer_ultimate_stresses = []
    for _, row in axial_strain_df.iterrows():
        epsilon_x_limit = row['Longitudinal_Strain']
        global_strain_vector = np.array([epsilon_x_limit, 0, 0])
        
        for i, angle in enumerate(full_stringer_layup):
            T_epsilon = get_T_epsilon_matrix(angle)
            local_strain_vector = T_epsilon @ global_strain_vector
            local_stress_vector_limit = Q_matrix @ local_strain_vector
            
            # Apply safety factor to get ULTIMATE stresses
            local_stress_vector_ultimate = local_stress_vector_limit * SAFETY_FACTOR
            
            stringer_ultimate_stresses.append({
                'element_id': row['Element_ID'],
                'load_case': row['Load_Case'],
                'ply_id': f'Ply  {i + 1}',
                'sigma_1': local_stress_vector_ultimate[0],
                'sigma_2': local_stress_vector_ultimate[1],
                'tau_12': local_stress_vector_ultimate[2]
            })
    stringer_stress_df = pd.DataFrame(stringer_ultimate_stresses)

    # --- 3. PREPARE SKIN STRESSES AT ULTIMATE LOAD ---
    print("--- Preparing Skin ULTIMATE Stresses ---")
    skin_limit_stress_df = composite_stress_x_df.merge(
        composite_stress_y_df, on=['Element_ID', 'Load_Case', 'Ply']
    ).merge(
        composite_stress_xy_df, on=['Element_ID', 'Load_Case', 'Ply']
    )
    skin_limit_stress_df = skin_limit_stress_df.rename(columns={
        'Element_ID': 'element_id', 'Ply': 'ply_id', 'Load_Case': 'load_case',
        'Stress_Normal_X': 'sigma_1', 'Stress_Normal_Y': 'sigma_2', 'Stress_Shear_XY': 'tau_12'
    })

    # Apply safety factor to get ULTIMATE stresses
    skin_ultimate_stress_df = skin_limit_stress_df.copy()
    skin_ultimate_stress_df[['sigma_1', 'sigma_2', 'tau_12']] *= SAFETY_FACTOR

    # --- 4. COMBINE DATA AND RUN PUCK ANALYSIS ---
    print("--- Unifying Data and Running Puck Analysis ---")
    master_ultimate_stress_df = pd.concat([skin_ultimate_stress_df, stringer_stress_df], ignore_index=True)
    
    def puck_analysis_core(df, strengths, puck_params):
        results = []
        R_para_t, R_para_c = strengths["R_para_t"], strengths["R_para_c"]
        R_perp_t, R_perp_c, R_perp_para = strengths["R_perp_t"], strengths["R_perp_c"], strengths["R_perp_para"]
        p_plus, p_minus, p_perp_perp_minus = puck_params["p_perp_para_plus"], puck_params["p_perp_para_minus"], puck_params["p_perp_perp_minus"]
        
        for _, row in df.iterrows():
            s1, s2, t12 = row['sigma_1'], row['sigma_2'], row['tau_12']
            elem_id = row['element_id']
            ply_id = row['ply_id'] if 'ply_id' in row else 'N/A'
            load_case = row['load_case'] if 'load_case' in row else 'N/A'

            # Fiber Fracture (FF) Reserve Factor
            ff_factor = (R_para_t / s1) if s1 >= 0 else (R_para_c / abs(s1))
            if s1 == 0: ff_factor = np.inf

            # Inter-Fiber Fracture (IFF) Reserve Factor
            f_E_IFF = 0
            iff_mode = 'N/A'
            if s2 >= 0: # Mode A
                iff_mode = 'A'
                term1 = (t12 / R_perp_para)**2
                term2 = (1 - p_plus * R_perp_t / R_perp_para)**2 * (s2 / R_perp_t)**2
                term3 = p_plus * s2 / R_perp_para
                f_E_IFF = np.sqrt(term1 + term2) + term3
            else: # Modes B or C
                R_perp_perp_A = R_perp_c / (2 * (1 + p_perp_perp_minus))
                tau_21_c = R_perp_para * np.sqrt(1 + 2 * p_perp_perp_minus)
                if abs(s2) == 0 or abs(t12 / s2) > abs(tau_21_c / R_perp_perp_A): # Mode B
                    iff_mode = 'B'
                    f_E_IFF = (1 / R_perp_para) * (np.sqrt(t12**2 + (p_minus * s2)**2) + p_minus * s2)
                else: # Mode C
                    iff_mode = 'C'
                    term1 = (t12 / (2 * (1 + p_perp_perp_minus) * R_perp_para))**2
                    term2 = (s2 / R_perp_c)**2
                    f_E_IFF = (term1 + term2) * (R_perp_c / -s2)
            iff_factor = 1 / f_E_IFF if f_E_IFF > 0 else np.inf

            # Final Reserve Factor is the minimum of the two, since we used ULTIMATE stresses
            reserve_factor = min(ff_factor, iff_factor)
            
            results.append({
                'RF_FF': ff_factor,
                'RF_IFF': iff_factor,
                'IFF_Mode': iff_mode,
                'RF_Strength': reserve_factor,
            })
            
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)

    final_results_df = puck_analysis_core(master_ultimate_stress_df, strengths, puck_params)
    
    # --- 5. FINALIZE AND DISPLAY RESULTS ---
    print("\n--- Analysis Complete ---")
    final_results_df['ply_id_num'] = final_results_df['ply_id'].str.extract('(\d+)').astype(int)
    # Always output the IFF mode (A, B, or C) in the 'Mode' column
    final_results_df['Mode'] = final_results_df['IFF_Mode']
    return final_results_df
