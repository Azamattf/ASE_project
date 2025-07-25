import pandas as pd
import argparse
import sys
import numpy as np

# Import the refactored functions
from calculate_abd_matrix import calculate_abd_matrix
from puck_failure_analysis import perform_full_puck_analysis
from panel_buckling_analysis import panel_buckling_analysis
from combined_buckling_analysis import combined_buckling_analysis
from combined_section_properties import calculate_stringer_section_properties # Import the new function

# --- Panel Geometry ---
# Source: ASE_Project2025_Part2.pdf, Task 2.1d. Assuming standard bay dimensions.
PANEL_LENGTH_A = 750    # mm, assumed rib spacing
PANEL_WIDTH_B = 400     # mm, assumed stringer pitch

# --- Laminate Properties for Skin Panel ---
# Layup: (+45/+45/-45/-45/0/0/90/90)s, t_ply = 0.552 mm 
# Total plies = 16, Total thickness = 16 * 0.552 = 8.832 mm
SKIN_PLY_THICKNESS = 0.552  # mm
SKIN_THICKNESS = 16 * 0.552 # mm
SKIN_LAYUP = [+45, +45, -45, -45, 0, 0, 90, 90] # This is the half-stack for a symmetric layup
UD_PLY_PROPERTIES = {
    "E1": 131203.67,  # MPa, Young's Modulus in fiber direction
    "E2": 10092.59,    # MPa, Young's Modulus transverse to fiber
    "G12": 5046.3,   # MPa, In-plane Shear Modulus
    "v12": 0.33      # Major Poisson's Ratio (CHANGE IN MODEL)
}

# --- Stringer Properties ---
STRINGER_LAYUP_FLANGE = [+45, +45, -45, -45, 0, 0, 90, 90] # This is the half-stack for a symmetric layup
STRINGER_LAYUP_WEB = [-45, -45, +45, +45, 0, 0, 90, 90] # This is the half-stack for a symmetric layup
STRINGER_PLY_THICKNESS = 0.25  # mm
STRINGER_E_MODULUS = 51222.83  # MPa
STRINGER_G_MODULUS = 19461.69  # MPa
STRINGER_EI_Y = 3636135829.33 # N*mm^2
STRINGER_EI_Z = 5867400997.33 # N*mm^2
STRINGER_ELASTIC_CENTER_Y = 0 # mm
STRINGER_ELASTIC_CENTER_Z = -14.0 # mm

# --- Analysis Factors ---
# Source: Standard factors used in course examples.
SAFETY_FACTOR = 1.5
KNOCKDOWN_FACTOR = 0.9


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CFRP Structural Analysis')
    parser.add_argument('--axial-stress', default='AxialStress.csv', 
                       help='Path to axial stress CSV file (default: AxialStress.csv)')
    parser.add_argument('--axial-strain', default='AxialStrain.csv',
                       help='Path to axial strain CSV file (default: AxialStrain.csv)')
    parser.add_argument('--composite-stress-x', default='CompositeStressNormalX.csv',
                       help='Path to composite stress normal X CSV file (default: CompositeStressNormalX.csv)')
    parser.add_argument('--composite-stress-y', default='CompositeStressNormalY.csv',
                       help='Path to composite stress normal Y CSV file (default: CompositeStressNormalY.csv)')
    parser.add_argument('--composite-stress-xy', default='CompositeStressShearXY.csv',
                       help='Path to composite stress shear XY CSV file (default: CompositeStressShearXY.csv)')
    parser.add_argument('--element-stresses', default='ElementStresses2D3D.csv',
                       help='Path to element stresses CSV file (default: ElementStresses2D3D.csv)')
    
    args = parser.parse_args()
    
    # Material properties

    # Call calculate_abd_matrix from the imported module
    print("Calculating Skin ABD Matrix...")
    A_matrix_skin, D_matrix_skin, A_inv_matrix_skin, D_inv_matrix_skin = calculate_abd_matrix(UD_PLY_PROPERTIES, SKIN_LAYUP, SKIN_PLY_THICKNESS)
    A_matrix_flange, D_matrix_flange, A_inv_matrix_flange, D_inv_matrix_flange = calculate_abd_matrix(UD_PLY_PROPERTIES, STRINGER_LAYUP_FLANGE, STRINGER_PLY_THICKNESS)
    A_matrix_web, D_matrix_web, A_inv_matrix_web, D_inv_matrix_web = calculate_abd_matrix(UD_PLY_PROPERTIES, STRINGER_LAYUP_WEB, STRINGER_PLY_THICKNESS)
    
    # Define stringer dimensions for combined_section_properties
    flange_width_val = 70.0
    web_height_val = 40.0

    # Run Stringer Section Properties Calculation
    print("\n" + "="*60)
    print("RUNNING STRINGER SECTION PROPERTIES CALCULATION")
    print("="*60)

    calculate_stringer_section_properties(A_matrix_skin, D_matrix_skin, A_matrix_flange, D_matrix_flange, A_inv_matrix_web, D_inv_matrix_web, D_matrix_web, flange_width_val, web_height_val, STRINGER_PLY_THICKNESS, SKIN_THICKNESS, PANEL_WIDTH_B, KNOCKDOWN_FACTOR)

    print("="*60)

    
    try:
        # Load axial stress data (bar elements)
        # A12-A47: element IDs, C12-C47: load case, E12-E47: longitudinal stress
        axial_stress_df = pd.read_csv(args.axial_stress, skiprows=11, usecols=[0, 2, 4], 
                                     names=['Element_ID', 'Load_Case', 'Longitudinal_Stress'])
        axial_stress_df = axial_stress_df.head(36)  # A12-A47 = 36 rows
        # Convert numeric columns to proper types
        axial_stress_df['Element_ID'] = pd.to_numeric(axial_stress_df['Element_ID'], errors='coerce')
        axial_stress_df['Load_Case'] = pd.to_numeric(axial_stress_df['Load_Case'], errors='coerce')
        axial_stress_df['Longitudinal_Stress'] = pd.to_numeric(axial_stress_df['Longitudinal_Stress'], errors='coerce')
        
        # Load axial strain data (bar elements) from file
        axial_strain_df = pd.read_csv(args.axial_strain, skiprows=11, usecols=[0, 2, 4], 
                                     names=['Element_ID', 'Load_Case', 'Longitudinal_Strain'])
        axial_strain_df = axial_strain_df.head(36)  # A12-A47 = 36 rows
        # Convert numeric columns to proper types
        axial_strain_df['Element_ID'] = pd.to_numeric(axial_strain_df['Element_ID'], errors='coerce')
        axial_strain_df['Load_Case'] = pd.to_numeric(axial_strain_df['Load_Case'], errors='coerce')
        axial_strain_df['Longitudinal_Strain'] = pd.to_numeric(axial_strain_df['Longitudinal_Strain'], errors='coerce')
        
        # Load composite stress data (panel elements - ply level)
        # A12-A1451: element IDs, C12-C1451: load case, E12-E1451: ply, F12-F1451: stress
        composite_stress_x_df = pd.read_csv(args.composite_stress_x, skiprows=11, usecols=[0, 2, 4, 5],
                                           names=['Element_ID', 'Load_Case', 'Ply', 'Stress_Normal_X'])
        composite_stress_x_df = composite_stress_x_df.head(1440)  # A12-A1451 = 1440 rows
        # Convert numeric columns to proper types
        composite_stress_x_df['Element_ID'] = pd.to_numeric(composite_stress_x_df['Element_ID'], errors='coerce')
        composite_stress_x_df['Load_Case'] = pd.to_numeric(composite_stress_x_df['Load_Case'], errors='coerce')
        composite_stress_x_df['Stress_Normal_X'] = pd.to_numeric(composite_stress_x_df['Stress_Normal_X'], errors='coerce')
        
        composite_stress_y_df = pd.read_csv(args.composite_stress_y, skiprows=11, usecols=[0, 2, 4, 5],
                                           names=['Element_ID', 'Load_Case', 'Ply', 'Stress_Normal_Y'])
        composite_stress_y_df = composite_stress_y_df.head(1440)
        # Convert numeric columns to proper types
        composite_stress_y_df['Element_ID'] = pd.to_numeric(composite_stress_y_df['Element_ID'], errors='coerce')
        composite_stress_y_df['Load_Case'] = pd.to_numeric(composite_stress_y_df['Load_Case'], errors='coerce')
        composite_stress_y_df['Stress_Normal_Y'] = pd.to_numeric(composite_stress_y_df['Stress_Normal_Y'], errors='coerce')
        
        composite_stress_xy_df = pd.read_csv(args.composite_stress_xy, skiprows=11, usecols=[0, 2, 4, 5],
                                            names=['Element_ID', 'Load_Case', 'Ply', 'Stress_Shear_XY'])
        composite_stress_xy_df = composite_stress_xy_df.head(1440)
        # Convert numeric columns to proper types
        composite_stress_xy_df['Element_ID'] = pd.to_numeric(composite_stress_xy_df['Element_ID'], errors='coerce')
        composite_stress_xy_df['Load_Case'] = pd.to_numeric(composite_stress_xy_df['Load_Case'], errors='coerce')
        composite_stress_xy_df['Stress_Shear_XY'] = pd.to_numeric(composite_stress_xy_df['Stress_Shear_XY'], errors='coerce')
        
        # Load element-level stresses (panel elements - element level)
        # A12-A101: element IDs, C12-C101: load case, F12-F101: XX, G12-G101: XY, H12-H101: YY
        element_stresses_df = pd.read_csv(args.element_stresses, skiprows=11, usecols=[0, 2, 5, 6, 7],
                                         names=['Element_ID', 'Load_Case', 'Stress_XX', 'Stress_XY', 'Stress_YY'])
        element_stresses_df = element_stresses_df.head(90)  # A12-A101 = 90 rows
        # Convert numeric columns to proper types
        element_stresses_df['Element_ID'] = pd.to_numeric(element_stresses_df['Element_ID'], errors='coerce')
        element_stresses_df['Load_Case'] = pd.to_numeric(element_stresses_df['Load_Case'], errors='coerce')
        element_stresses_df['Stress_XX'] = pd.to_numeric(element_stresses_df['Stress_XX'], errors='coerce')
        element_stresses_df['Stress_XY'] = pd.to_numeric(element_stresses_df['Stress_XY'], errors='coerce')
        element_stresses_df['Stress_YY'] = pd.to_numeric(element_stresses_df['Stress_YY'], errors='coerce')
        
        # Remove any rows with NaN values that might have been created during conversion
        axial_stress_df = axial_stress_df.dropna()
        axial_strain_df = axial_strain_df.dropna()
        composite_stress_x_df = composite_stress_x_df.dropna()
        composite_stress_y_df = composite_stress_y_df.dropna()
        composite_stress_xy_df = composite_stress_xy_df.dropna()
        element_stresses_df = element_stresses_df.dropna()
        
        # Make data frames available globally for further analysis
        globals().update({
            'axial_stress_df': axial_stress_df,
            'axial_strain_df': axial_strain_df,
            'composite_stress_x_df': composite_stress_x_df,
            'composite_stress_y_df': composite_stress_y_df,
            'composite_stress_xy_df': composite_stress_xy_df,
            'element_stresses_df': element_stresses_df
        })
        
        # Run Puck failure analysis
        print("\n" + "="*60)
        print("RUNNING PUCK FAILURE ANALYSIS")
        print("="*60)
        
        # --- Run Puck Failure Analysis ---
        # Call puck_failure_analysis from the imported module, passing necessary constants
        puck_results_df = perform_full_puck_analysis(composite_stress_x_df, composite_stress_y_df, composite_stress_xy_df, axial_strain_df, STRINGER_LAYUP_FLANGE, STRINGER_PLY_THICKNESS, UD_PLY_PROPERTIES, SAFETY_FACTOR)
        globals()['puck_results_df'] = puck_results_df
        # --- Extract and Print Specific Homework Results ---
        print("\n" + "="*80)
        print("SPECIFIC RESULTS FOR HOMEWORK")
        print("="*80)

        # Filter for the specific elements
        panel_8_results = puck_results_df[puck_results_df['element_id'] == 8].copy()
        stringer_60_results = puck_results_df[puck_results_df['element_id'] == 60].copy()

        # Rename columns for clarity in the final output
        panel_8_results.rename(columns={
            'FF_Fail_Factor': 'RF_FF', 'IFF_Fail_Factor': 'RF_IFF', 
            'Reserve_Factor': 'RF_Strength'
        }, inplace=True)
        
        stringer_60_results.rename(columns={
            'FF_Fail_Factor': 'RF_FF', 'IFF_Fail_Factor': 'RF_IFF', 
            'Reserve_Factor': 'RF_Strength'
        }, inplace=True)

        # --- Print Panel 8 Results ---
        print("\n## Results for Panel Element ID: 8")
        if not panel_8_results.empty:
            # Sort for consistent output by load case and ply number
            panel_8_results_sorted = panel_8_results.sort_values(by=['load_case', 'ply_id_num'])
            print("Full results for each ply and load case:")
            print(panel_8_results_sorted[['load_case', 'ply_id', 'RF_FF', 'RF_IFF', 'Mode', 'RF_Strength']].round(3).to_string(index=False))
        else:
            print("No data found for panel element 8.")

        # --- Print Stringer 60 Results ---
        print("\n## Results for Stringer Flange Element ID: 60")
        if not stringer_60_results.empty:
            # Sort for consistent output by load case and ply number
            stringer_60_results_sorted = stringer_60_results.sort_values(by=['load_case', 'ply_id_num'])
            print("Full results for each ply and load case:")
            print(stringer_60_results_sorted[['load_case', 'ply_id', 'RF_FF', 'RF_IFF', 'Mode', 'RF_Strength']].round(3).to_string(index=False))
        else:
            print("No data found for stringer element 60.")

        # --- Run Panel Buckling Analysis ---
        print("\n" + "="*60)
        print("RUNNING PANEL BUCKLING ANALYSIS")
        print("="*60)
        
        # Call panel_buckling_analysis from the imported module, passing necessary constants
        buckling_results_df = panel_buckling_analysis(element_stresses_df, D_matrix_skin, KNOCKDOWN_FACTOR, PANEL_LENGTH_A, PANEL_WIDTH_B, SKIN_LAYUP, SKIN_PLY_THICKNESS, SAFETY_FACTOR)
        globals()['buckling_results_df'] = buckling_results_df
        
        # --- Run Column Buckling Analysis ---
        print("\n" + "="*60)
        print("RUNNING COLUMN BUCKLING ANALYSIS")
        print("="*60)
        
        # Define constants dictionary for combined buckling analysis
        constants_combined_buckling = {
            'PANEL_LENGTH_A': PANEL_LENGTH_A,
            'PANEL_WIDTH_B': PANEL_WIDTH_B,
            'SKIN_THICKNESS': SKIN_THICKNESS,
            'STRINGER_PLY_THICKNESS': STRINGER_PLY_THICKNESS,
            'SAFETY_FACTOR': SAFETY_FACTOR,
            'KNOCKDOWN_FACTOR': KNOCKDOWN_FACTOR,
        }

        # Call combined_buckling_analysis from the imported module
        combined_buckling_results_df = combined_buckling_analysis(
            axial_stress_df,
            element_stresses_df,
            A_matrix_skin, D_matrix_skin,
            A_matrix_flange, D_matrix_flange,
            A_inv_matrix_web, D_inv_matrix_web,
            constants_combined_buckling
        )
        globals()['combined_buckling_results_df'] = combined_buckling_results_df

        print("="*60)
        
        # Output panel 8 and stringer 60 results as separate CSV tables
        #import sys
        #for df in [panel_8_results, stringer_60_results]:
        #    if 'ply_id_num' not in df.columns:
        #        df['ply_id_num'] = df['ply_id'].str.extract(r'(\d+)').astype(int)
        #    sort_cols = []
        #    if 'load_case' in df.columns:
        #        sort_cols.append('load_case')
        #    sort_cols.append('ply_id_num')
        #    df_sorted = df.sort_values(by=sort_cols)
        #    cols = ['element_id', 'load_case', 'ply_id', 'RF_FF', 'RF_IFF', 'Mode', 'RF_Strength'] if 'load_case' in df.columns else ['element_id', 'ply_id', 'RF_FF', 'RF_IFF', 'Mode', 'RF_Strength']
        #    df_sorted[cols].to_csv(sys.stdout, index=False)
        #    print()  # Blank line between tables
        #
        return (axial_stress_df, axial_strain_df, composite_stress_x_df, 
                composite_stress_y_df, composite_stress_xy_df, element_stresses_df, 
                puck_results_df, buckling_results_df, combined_buckling_results_df) # Include new DF in return
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

# The original function definitions for calculate_abd_matrix, puck_failure_analysis,
# and panel_buckling_analysis have been moved to their respective files and removed from here.

if __name__ == "__main__":
    main()
