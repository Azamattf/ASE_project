#!/usr/bin/env python3
"""
Main script to run structural analysis with user-defined parameters.

Usage:
python run_analysis.py <1D_csv> <2D3D_csv> <E_modulus> <E_B_modulus> <ultimate_strength> <yield_strength> <panel_length> <panel_width> <skin_thickness> <stringer_height> <stringer_thickness> <stringer_web_width> <stringer_lip_width>
OR
python run_analysis.py <1D_csv> <2D3D_csv> --template <template_csv> <panel_length> <panel_width> <skin_thickness> <stringer_height> <stringer_thickness> <stringer_web_width> <stringer_lip_width>

Arguments:
- 1D_csv: Path to 1D element stress CSV file
- 2D3D_csv: Path to 2D/3D element stress CSV file
- E_modulus: Young's modulus (MPa)
- E_B_modulus: B-basis Young's modulus (MPa)
- ultimate_strength: Ultimate tensile strength (MPa)
- yield_strength: Yield strength (MPa)
- template_csv: Path to template CSV file containing material properties
- panel_length: Panel length (mm)
- panel_width: Panel width (stringer pitch) (mm)
- skin_thickness: Skin thickness (mm)
- stringer_height: Stringer height (mm)
- stringer_thickness: Stringer thickness (mm)
- stringer_web_width: Stringer web width (mm)
- stringer_lip_width: Stringer lip width (mm)
"""

import sys
import subprocess
import os
import pandas as pd
import numpy as np
import time

def read_material_properties_from_template(template_file):
    """Read material properties from template CSV file"""
    try:
        # Read the first 15 lines to capture the material properties section
        with open(template_file, 'r', encoding='utf-8') as f:
            lines = []
            for i, line in enumerate(f):
                if i < 15:  # Read first 15 lines
                    lines.append(line.strip())
                else:
                    break
        
        # Parse material properties
        E_modulus = None
        E_B_modulus = None
        yield_strength = None
        ultimate_strength = None
        
        for line in lines:
            if line.startswith('E-modulus_avg;'):
                E_modulus = float(line.split(';')[1])
            elif line.startswith('E-modulus_B-basis;'):
                E_B_modulus = float(line.split(';')[1])
            elif line.startswith('Yield strength (t/c);'):
                yield_strength = float(line.split(';')[1])
            elif line.startswith('Ultimate strength (t/c);'):
                ultimate_strength = float(line.split(';')[1])
        
        # Validate that all properties were found
        if None in [E_modulus, E_B_modulus, yield_strength, ultimate_strength]:
            missing = []
            if E_modulus is None: missing.append("E-modulus_avg")
            if E_B_modulus is None: missing.append("E-modulus_B-basis")
            if yield_strength is None: missing.append("Yield strength")
            if ultimate_strength is None: missing.append("Ultimate strength")
            raise ValueError(f"Could not find the following material properties in template: {', '.join(missing)}")
        
        return E_modulus, E_B_modulus, ultimate_strength, yield_strength
        
    except Exception as e:
        print(f"❌ Error reading template file '{template_file}': {e}")
        sys.exit(1)

def calculate_stringer_elastic_center(stringer_height, web_width, lip_width, thickness):
    """
    Calculates the vertical coordinate of the effective offset-point (z_eff) for a lipped
    U-channel stringer.

    For the context of this project, the required offset is the distance from the
    section's centroid to its extreme top fiber. This function calculates that value,
    which is the correct input for the z-offset in HyperMesh (Task 1.2c).

    **Orientation Assumed:**
    - The z-axis origin (z=0) is the interface where the stringer's web meets the skin.
    - The Web is the horizontal base of the 'U' at z=0.
    - The Flanges are the two vertical sections.
    - The Lips are the two horizontal sections at the top.

    Args:
        stringer_height (float): The total vertical height of the stringer section.
        web_width (float): The total width of the horizontal WEB section.
        lip_width (float): The clear, extending width of each of the two horizontal lips.
        thickness (float): The constant material thickness for all parts.

    Returns:
        float: The vertical distance of the effective offset point (z_eff) from the
               web/skin interface. This is the required z-offset.
    """
    # --- 1. Calculate Centroid (z_c) and Moment of Inertia (I_x) ---
    # These property calculations are correct and consistent with HyperMesh.

    # Areas of each individual part of the cross-section
    area_web = web_width * thickness
    # Note: flange_height assumes dimensions overlap, consistent with FEA practice.
    flange_height = stringer_height - thickness
    area_flange = flange_height * thickness
    area_lip = lip_width * thickness
    total_area = area_web + (2 * area_flange) + (2 * area_lip)

    # Centroid z-location for each individual part (measured from z=0)
    z_centroid_web = thickness / 2.0
    z_centroid_flange = thickness + (flange_height / 2.0)
    z_centroid_lip = stringer_height - (thickness / 2.0)

    # Overall Centroid (z_c) of the entire cross-section
    first_moment_of_area = (area_web * z_centroid_web) + \
                           (2 * area_flange * z_centroid_flange) + \
                           (2 * area_lip * z_centroid_lip)

    if total_area == 0:
        return 0.0 # Avoid division by zero
    z_c = first_moment_of_area / total_area

    # Moment of Inertia (I_x) about the horizontal centroidal axis
    # The Parallel Axis Theorem (I = I_local + A*d^2) is used for each part.
    d_web = z_c - z_centroid_web
    ix_web = (web_width * thickness**3) / 12.0 + area_web * d_web**2

    d_flange = z_c - z_centroid_flange
    ix_flanges = 2 * ((thickness * flange_height**3) / 12.0 + area_flange * d_flange**2)

    d_lip = z_c - z_centroid_lip
    ix_lips = 2 * ((lip_width * thickness**3) / 12.0 + area_lip * d_lip**2)

    I_x = ix_web + ix_flanges + ix_lips

    # --- 2. Calculate Effective Offset Coordinate (z_ec) ---
    # CORRECTED: Based on the HyperMesh data, the required offset for this project
    # is the distance from the centroid to the extreme top fiber of the stringer.
    
    # We define the offset point relative to the origin (z=0 at the web/skin interface).
    # For this problem's context, this happens to be the coordinate of the top edge.
    z_ec = stringer_height - z_c

    # --- 3. Print and Return Results ---
    print(f"\n{'='*60}")
    print("Stringer Offset Calculation (Final Method)")
    print("="*60)
    print(f"Total Area:                   {total_area:.3f} mm²")
    print(f"Centroid (z_c):               {z_c:.3f} mm")
    print(f"Moment of Inertia (I_x):      {I_x:.3f} mm⁴")
    print(f"Effective Offset (z_ec):      {z_ec:.3f} mm (from web/skin interface)")
    print(f"\nThis z_ec is the required z-offset for HyperMesh.")
    print("="*60)

    return z_ec

def run_hypermesh_analysis(skin_thickness, stringer_height, stringer_thickness, stringer_web_width, stringer_lip_width):
    """Run HyperMesh batch analysis with two separate Tcl scripts"""
    print(f"\n{'='*60}")
    print("Running HyperMesh/OptiStruct Analysis")
    print(f"{'='*60}")
    print(f"Geometric Parameters:")
    print(f"  Skin Thickness:     {skin_thickness} mm")
    print(f"  Stringer Height:    {stringer_height} mm")
    print(f"  Stringer Thickness: {stringer_thickness} mm")
    print(f"  Stringer Web Width: {stringer_web_width} mm")
    print(f"  Stringer Lip Width: {stringer_lip_width} mm")
    
    # Create first Tcl script (parameters and simulation)
    script1_content = f"""# 1. Modify the design parameters
# DIM1 (height)
*setvalue parameters id=1 STATUS=2 valuedouble={stringer_height}
# DIM2 (stringer thickness)
*setvalue parameters id=2 STATUS=2 valuedouble={stringer_thickness}
# DIM3 (web width)
*setvalue parameters id=3 STATUS=2 valuedouble={stringer_web_width}
# DIM4 (lip_width)
*setvalue parameters id=4 STATUS=2 valuedouble={stringer_lip_width}
# skin thickness
*setvalue parameters id=5 STATUS=2 valuedouble={skin_thickness}
# z-offset set to half of skin thickness
*setvalue parameters id=6 STATUS=2 valuedouble={skin_thickness/2}

# 2. Export the solver deck (.fem file) for OptiStruct
hm_answernext yes
*feoutputwithdata "C:/Program Files/Altair/2025/hwdesktop/templates/feoutput/optistruct/optistruct" "C:/Users/Stefan/Desktop/Fakultet/sem4/ASE/part1_task1_2_templates/ASE_Project2025_SuperPanel_03791970.fem" 0 0 2 1 8

# 3. Run the OptiStruct solver using the Altair Compute Console (acc.exe)
puts "--- Submitting job to Altair Compute Console ---"
exec cmd /c "C:/Program Files/Altair/2025/common/acc/scripts/acc.bat" -i "C:/Users/Stefan/Desktop/Fakultet/sem4/ASE/part1_task1_2_templates/ASE_Project2025_SuperPanel_03791970.fem"
puts "--- Solver run finished ---"
"""
    
    # Create second Tcl script (results loading and export)
    script2_content = """# 4. Load the NEWLY created results file (.h3d)
hm_answernext yes
*setvalue results id=1 resultfiles="C:/Users/Stefan/Desktop/Fakultet/sem4/ASE/part1_task1_2_templates/ASE_Project2025_SuperPanel_03791970.h3d"
*setvalue results id=1 init=1

# 5. Apply the results query from the XML configuration file
puts "--- Applying results query ---"
hm_getresults id=1 xml="C:/Users/Stefan/Desktop/Fakultet/sem4/ASE/part1_task1_2_templates/queryconfig.xml"
puts "--- Query finished. Results have been exported. ---"

# 6. Save the final model, which now has the results loaded
hm_answernext yes
*writefile "C:/Users/Stefan/Desktop/Fakultet/sem4/ASE/part1_task1_2_templates/ASE_Project2025_SuperPanel_03791970_redesign.hm" 1
"""
    
    # Write temporary Tcl scripts
    temp_script1 = "temp_script1.tcl"
    temp_script2 = "temp_script2.tcl"
    
    with open(temp_script1, 'w') as f:
        f.write(script1_content)
    
    with open(temp_script2, 'w') as f:
        f.write(script2_content)
    
    try:
        # Run first HyperMesh batch (parameters and simulation)
        print("Step 1: Setting parameters and running simulation...")
        hm_cmd1 = [
            "C:/Program Files/Altair/2025/hwdesktop/hw/bin/win64/hmbatch.exe",
            "-m", "C:/Users/Stefan/Desktop/Fakultet/sem4/ASE/part1_task1_1_templates/ASE_Project2025_SuperPanel_03791970.hm",
            "-tcl", temp_script1
        ]
        
        try:
            result1 = subprocess.run(hm_cmd1, capture_output=True, text=True, shell=True)
            if result1.returncode not in [0, 1]:
                print(f"❌ Step 1 failed with return code {result1.returncode}")
                print(f"Error output: {result1.stderr}")
                return False
        except Exception as e:
            print(f"❌ Step 1 failed with exception: {e}")
            return False
        
        print("✅ Step 1 completed: Parameters set and simulation run")
        
        # Wait a moment for files to be written
        time.sleep(2)
        
        # Run second HyperMesh batch (results loading and export)
        print("Step 2: Loading results and exporting data...")
        hm_cmd2 = [
            "C:/Program Files/Altair/2025/hwdesktop/hw/bin/win64/hmbatch.exe",
            "-m", "C:/Users/Stefan/Desktop/Fakultet/sem4/ASE/part1_task1_2_templates/ASE_Project2025_SuperPanel_03791970_redesign.hm",
            "-tcl", temp_script2
        ]
        
        try:
            result2 = subprocess.run(hm_cmd2, capture_output=True, text=True, shell=True)
            if result2.returncode not in [0, 1]:
                print(f"❌ Step 2 failed with return code {result2.returncode}")
                print(f"Error output: {result2.stderr}")
                return False
        except Exception as e:
            print(f"❌ Step 2 failed with exception: {e}")
            return False
        
        print("✅ Step 2 completed: Results loaded and data exported")
        print("✅ HyperMesh analysis completed successfully")
        
        return True
        
    finally:
        # Clean up temporary files
        for temp_file in [temp_script1, temp_script2]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

def run_script(script_name, args):
    """Run a Python script with given arguments"""
    cmd = [sys.executable, script_name] + args
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"Arguments: {' '.join(args)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} failed with return code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Could not find {script_name}")
        return False

def main():
    # Check for template mode
    if len(sys.argv) == 12 and sys.argv[3] == '--template':
        # Template mode: python run_analysis.py <1D_csv> <2D3D_csv> --template <template_csv> <panel_length> <panel_width> <skin_thickness> <stringer_height> <stringer_thickness> <stringer_web_width> <stringer_lip_width>
        csv_1d = sys.argv[1]
        csv_2d3d = sys.argv[2]
        template_file = sys.argv[4]
        panel_length = float(sys.argv[5])
        panel_width = float(sys.argv[6])
        skin_thickness = float(sys.argv[7])
        stringer_height = float(sys.argv[8])
        stringer_thickness = float(sys.argv[9])
        stringer_web_width = float(sys.argv[10])
        stringer_lip_width = float(sys.argv[11])
        
        # Validate input files exist
        if not os.path.exists(template_file):
            print(f"❌ Error: Template CSV file '{template_file}' not found")
            sys.exit(1)
        
        # Read material properties from template
        E_modulus, E_B_modulus, ultimate_strength, yield_strength = read_material_properties_from_template(template_file)
        
        print("Material properties read from template:")
        print(f"  E-modulus_avg:      {E_modulus} MPa")
        print(f"  E-modulus_B-basis:  {E_B_modulus} MPa")
        print(f"  Ultimate strength:  {ultimate_strength} MPa")
        print(f"  Yield strength:     {yield_strength} MPa")
        
    elif len(sys.argv) == 14:
        # Manual mode: python run_analysis.py <1D_csv> <2D3D_csv> <E_modulus> <E_B_modulus> <ultimate_strength> <yield_strength> <panel_length> <panel_width> <skin_thickness> <stringer_height> <stringer_thickness> <stringer_web_width> <stringer_lip_width>
        csv_1d = sys.argv[1]
        csv_2d3d = sys.argv[2]
        E_modulus = float(sys.argv[3])
        E_B_modulus = float(sys.argv[4])
        ultimate_strength = float(sys.argv[5])
        yield_strength = float(sys.argv[6])
        panel_length = float(sys.argv[7])
        panel_width = float(sys.argv[8])
        skin_thickness = float(sys.argv[9])
        stringer_height = float(sys.argv[10])
        stringer_thickness = float(sys.argv[11])
        stringer_web_width = float(sys.argv[12])
        stringer_lip_width = float(sys.argv[13])
            
    else:
        print("Usage:")
        print("  python run_analysis.py <1D_csv> <2D3D_csv> <E_modulus> <E_B_modulus> <ultimate_strength> <yield_strength> <panel_length> <panel_width> <skin_thickness> <stringer_height> <stringer_thickness> <stringer_web_width> <stringer_lip_width>")
        print("  OR")
        print("  python run_analysis.py <1D_csv> <2D3D_csv> --template <template_csv> <panel_length> <panel_width> <skin_thickness> <stringer_height> <stringer_thickness> <stringer_web_width> <stringer_lip_width>")
        print("\nArguments:")
        print("  1D_csv: Path to 1D element stress CSV file")
        print("  2D3D_csv: Path to 2D/3D element stress CSV file")
        print("  E_modulus: Young's modulus (MPa)")
        print("  E_B_modulus: B-basis Young's modulus (MPa)")
        print("  ultimate_strength: Ultimate tensile strength (MPa)")
        print("  yield_strength: Yield strength (MPa)")
        print("  template_csv: Path to template CSV file containing material properties")
        print("  panel_length: Panel length (mm)")
        print("  panel_width: Panel width (stringer pitch) (mm)")
        print("  skin_thickness: Skin thickness (mm)")
        print("  stringer_height: Stringer height (mm)")
        print("  stringer_thickness: Stringer thickness (mm)")
        print("  stringer_web_width: Stringer web width (mm)")
        print("  stringer_lip_width: Stringer lip width (mm)")
        sys.exit(1)

    print("Automated Structural Analysis Suite")
    print("="*60)
    print(f"Material Properties:")
    print(f"  E-modulus:          {E_modulus} MPa")
    print(f"  E-B-modulus:        {E_B_modulus} MPa")
    print(f"  Ultimate Strength:  {ultimate_strength} MPa")
    print(f"  Yield Strength:     {yield_strength} MPa")
    print(f"Geometric Parameters:")
    print(f"  Panel Length:       {panel_length} mm")
    print(f"  Panel Width:        {panel_width} mm")
    print(f"  Skin Thickness:     {skin_thickness} mm")
    print(f"  Stringer Height:    {stringer_height} mm")
    print(f"  Stringer Thickness: {stringer_thickness} mm")
    print(f"  Stringer Web Width: {stringer_web_width} mm")
    print(f"  Stringer Lip Width: {stringer_lip_width} mm")

    z_ec = calculate_stringer_elastic_center(stringer_height, stringer_web_width, stringer_lip_width, stringer_thickness)

    # Step 1: Run HyperMesh/OptiStruct analysis (now in two parts)
    if not run_hypermesh_analysis(skin_thickness, stringer_height, stringer_thickness, stringer_web_width, stringer_lip_width):
        print("❌ HyperMesh analysis failed. Aborting.")
        sys.exit(1)

    # Step 2: Check if output CSV files exist
    if not os.path.exists(csv_1d):
        print(f"❌ Error: 1D CSV file '{csv_1d}' not found after HyperMesh analysis")
        sys.exit(1)
    if not os.path.exists(csv_2d3d):
        print(f"❌ Error: 2D/3D CSV file '{csv_2d3d}' not found after HyperMesh analysis")
        sys.exit(1)

    success_count = 0
    total_scripts = 3

    # Convert values to strings for subprocess arguments
    E_modulus_str = str(E_modulus)
    E_B_modulus_str = str(E_B_modulus)
    ultimate_strength_str = str(ultimate_strength)
    yield_strength_str = str(yield_strength)
    panel_length_str = str(panel_length)
    panel_width_str = str(panel_width)
    skin_thickness_str = str(skin_thickness)
    stringer_height_str = str(stringer_height)
    stringer_thickness_str = str(stringer_thickness)
    stringer_web_width_str = str(stringer_web_width)
    stringer_lip_width_str = str(stringer_lip_width)

    # Step 3: Run analysis scripts with geometric parameters
    # Run Script D - Stress Analysis with Ultimate Strength
    if run_script("script_d_p11.py", [csv_1d, csv_2d3d, ultimate_strength_str]):
        success_count += 1

    # Run Script E - Panel Buckling Analysis with E-B-modulus and geometric parameters
    if run_script("script_e_p11.py", [csv_1d, csv_2d3d, E_B_modulus_str, panel_length_str, panel_width_str, skin_thickness_str]):
        success_count += 1

    # Run Script F - Column Buckling Analysis with E-B-modulus, Yield Strength, and geometric parameters
    if run_script("script_f_p11.py", [csv_1d, csv_2d3d, E_B_modulus_str, yield_strength_str, panel_length_str, panel_width_str, skin_thickness_str, stringer_height_str, stringer_thickness_str, stringer_web_width_str, stringer_lip_width_str]):
        success_count += 1

    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Scripts completed successfully: {success_count}/{total_scripts}")
    
    if success_count == total_scripts:
        print("✅ All analyses completed successfully!")
        print("\nOutput Files Generated:")
        print("  - 1D_Stress_Results_with_RF.csv")
        print("  - 2D3D_Stress_Results_with_RF.csv")
        print("  - Biaxial_Panel_Buckling_Results.csv")
        print("  - Submission_Column_Buckling_Final_with_FoS.csv")
    else:
        print("⚠️  Some analyses failed. Check output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
