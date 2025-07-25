import pandas as pd
import numpy as np

def panel_buckling_analysis(element_stresses_df, D_matrix, knockdown_factor, panel_length_a, panel_width_b, skin_layup, skin_ply_thickness, safety_factor):
    """
    Performs a full panel buckling analysis on a set of finite elements.

    This version is updated to align with standard reserve factor calculation methods,
    where the final RF represents the safety margin against the ultimate load (RF > 1.0 is safe).

    Parameters:
    - element_stresses_df (pd.DataFrame): DataFrame with LIMIT stresses.
    - D_matrix (np.ndarray): The 3x3 bending stiffness matrix from CLT.
    - knockdown_factor (float): Factor to apply to the stiffness matrix for B-Values.
    - panel_length_a (float): The length of the panel ('a').
    - panel_width_b (float): The width of the panel ('b').
    - skin_layup (list): The half-stack layup to determine total plies.
    - skin_ply_thickness (float): The thickness of a single ply.
    - safety_factor (float): Factor of Safety (e.g., 1.5) used to define ultimate load.
    
    Returns:
    - pd.DataFrame: A DataFrame with the calculated reserve factors for each panel.
    """
    
    # --- 1. Pre-computation of Constants ---
    D_b = D_matrix * knockdown_factor
    total_thickness = len(skin_layup) * 2 * skin_ply_thickness
    D11, D12, D22, D66 = D_b[0, 0], D_b[0, 1], D_b[1, 1], D_b[2, 2]

    # --- 2. Group Elements into Panels and Average Stresses ---
    panel_results = []
    
    for load_case, group in element_stresses_df.groupby('Load_Case'):
        group = group.reset_index(drop=True)
        group['Panel_ID'] = group.index // 6
        avg_stresses = group.groupby('Panel_ID')[['Stress_XX', 'Stress_YY', 'Stress_XY']].mean()

        # --- 3. Iterate Through Each Panel for Analysis ---
        for panel_id, row in avg_stresses.iterrows():
            # USE LIMIT STRESSES FOR CALCULATIONS
            s_xx_limit = row['Stress_XX']
            s_yy_limit = row['Stress_YY']
            s_xy_limit = row['Stress_XY']

            # Initialize critical stresses and combined RF
            sigma_cr = np.inf # Critical biaxial buckling stress
            tau_cr = np.inf  # Critical shear buckling stress
            rf_combined = np.inf

            # --- 4. Align Coordinate System & Calculate Critical Stresses ---
            D11_eff, D22_eff = D11, D22
            if abs(s_yy_limit) > abs(s_xx_limit):
                a, b = panel_width_b, panel_length_a
                sigma_x_limit, sigma_y_limit = s_yy_limit, s_xx_limit
                D11_eff, D22_eff = D22, D11
            else:
                a, b = panel_length_a, panel_width_b
                sigma_x_limit, sigma_y_limit = s_xx_limit, s_yy_limit

            if sigma_x_limit >= 0:
                # If sigma_x_limit is positive (tension or zero), no compression buckling occurs.
                # Critical stresses remain np.inf, and RF_Combined remains np.inf.
                pass
            else:
                alpha = a / b
                beta = sigma_y_limit / sigma_x_limit if sigma_x_limit != 0 else 0

                # --- 5. Biaxial Buckling Critical Stress ---
                def get_sigma_cr_biaxial(m, n):
                    num = (D11_eff * (m/alpha)**4 + 2*(D12+D66)*((m*n)/alpha)**2 + D22_eff*n**4)
                    den = (m/alpha)**2 + beta*n**2
                    if den <= 0: return np.inf
                    return (np.pi**2 / (b**2 * total_thickness)) * (num / den)

                m_modes = range(1, int(2 * alpha) + 3)
                sigma_cr = min(get_sigma_cr_biaxial(m, 1) for m in m_modes)

                # --- 6. Shear Buckling Critical Stress ---
                delta = (np.sqrt(D11_eff * D22_eff)) / (D12 + 2 * D66)
                if delta < 1:
                    term1 = np.sqrt(D22_eff * (D12 + 2 * D66))
                    term2 = 11.7 + 0.532*delta + 0.938*delta**2
                else:
                    term1 = (D11_eff * D22_eff**3)**0.25
                    term2 = 8.12 + 5.05/delta
                tau_cr = (4 / (total_thickness * b**2)) * (term1 * term2)

                # --- 7. Calculate Combined Reserve Factor ---
                # This approach calculates the safety margin on the ULTIMATE load.
                # It uses the interaction formula: R_biaxial + (R_shear)^2 = 1/RF
                if sigma_cr <= 0 or tau_cr <= 0:
                    rf_combined = np.inf
                else:
                    # Calculate load utilization ratios using ULTIMATE loads
                    # R_biaxial = Ultimate Applied Stress / Critical Stress
                    R_biaxial = abs(sigma_x_limit * safety_factor) / sigma_cr
                    R_shear = abs(s_xy_limit * safety_factor) / tau_cr

                    # The inverse of the sum is the Reserve Factor
                    # This RF is the factor by which the ultimate load can be multiplied.
                    # An RF of 1.0 means the panel fails exactly at ultimate load.
                    inverse_rf = R_biaxial + R_shear**2
                    rf_combined = 1.0 / inverse_rf if inverse_rf > 0 else np.inf
            
            panel_results.append({
                'Load_Case': load_case, 'Panel_ID': panel_id,
                'Avg_Stress_XX': s_xx_limit, 'Avg_Stress_YY': s_yy_limit, 'Avg_Stress_XY': s_xy_limit,
                'Critical_Biaxial_Stress': sigma_cr,
                'Critical_Shear_Stress': tau_cr,
                'RF_Combined': rf_combined
            })
            
    results_df = pd.DataFrame(panel_results)

    if not results_df.empty:
        # Output as CSV with specified columns and header
        output_df = results_df[['Load_Case', 'Panel_ID', 'Avg_Stress_XX', 'Avg_Stress_YY', 
                                'Avg_Stress_XY', 'Critical_Shear_Stress', 'Critical_Biaxial_Stress', 
                                'RF_Combined']].copy()
        output_df = output_df.rename(columns={
            'Avg_Stress_XX': 'sig_xx_avg',
            'Avg_Stress_YY': 'sig_yy_avg',
            'Avg_Stress_XY': 'sig_xy_avg',
            'Critical_Shear_Stress': 'sig_crit_shear',
            'Critical_Biaxial_Stress': 'sig_crit_biax',
            'RF_Combined': 'rf_panelbuckl'
        })
        import sys
        output_df[['Load_Case', 'Panel_ID', 'sig_xx_avg', 'sig_yy_avg', 'sig_xy_avg', 'sig_crit_shear', 'sig_crit_biax', 'rf_panelbuckl']].to_csv(sys.stdout, index=False)
    else:
        print("⚠️ Analysis did not produce any results.")


    return results_df