#!/usr/bin/env python3
"""
Main script to run structural analysis with user-defined parameters.

Usage:
python run_analysis.py <1D_csv> <2D3D_csv> <E_modulus> <E_B_modulus> <ultimate_strength> <yield_strength>
OR
python run_analysis.py <1D_csv> <2D3D_csv> --template <template_csv>

Arguments:
- 1D_csv: Path to 1D element stress CSV file
- 2D3D_csv: Path to 2D/3D element stress CSV file
- E_modulus: Young's modulus (MPa)
- E_B_modulus: B-basis Young's modulus (MPa)
- ultimate_strength: Ultimate tensile strength (MPa)
- yield_strength: Yield strength (MPa)
- template_csv: Path to template CSV file containing material properties
"""

import sys
import subprocess
import os
import pandas as pd

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
    if len(sys.argv) == 5 and sys.argv[3] == '--template':
        # Template mode: python run_analysis.py <1D_csv> <2D3D_csv> --template <template_csv>
        csv_1d = sys.argv[1]
        csv_2d3d = sys.argv[2]
        template_file = sys.argv[4]
        
        # Validate input files exist
        if not os.path.exists(csv_1d):
            print(f"❌ Error: 1D CSV file '{csv_1d}' not found")
            sys.exit(1)
        if not os.path.exists(csv_2d3d):
            print(f"❌ Error: 2D/3D CSV file '{csv_2d3d}' not found")
            sys.exit(1)
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
        
    elif len(sys.argv) == 7:
        # Manual mode: python run_analysis.py <1D_csv> <2D3D_csv> <E_modulus> <E_B_modulus> <ultimate_strength> <yield_strength>
        csv_1d = sys.argv[1]
        csv_2d3d = sys.argv[2]
        E_modulus = float(sys.argv[3])
        E_B_modulus = float(sys.argv[4])
        ultimate_strength = float(sys.argv[5])
        yield_strength = float(sys.argv[6])
        
        # Validate input files exist
        if not os.path.exists(csv_1d):
            print(f"❌ Error: 1D CSV file '{csv_1d}' not found")
            sys.exit(1)
        if not os.path.exists(csv_2d3d):
            print(f"❌ Error: 2D/3D CSV file '{csv_2d3d}' not found")
            sys.exit(1)
            
    else:
        print("Usage:")
        print("  python run_analysis.py <1D_csv> <2D3D_csv> <E_modulus> <E_B_modulus> <ultimate_strength> <yield_strength>")
        print("  OR")
        print("  python run_analysis.py <1D_csv> <2D3D_csv> --template <template_csv>")
        print("\nArguments:")
        print("  1D_csv: Path to 1D element stress CSV file")
        print("  2D3D_csv: Path to 2D/3D element stress CSV file")
        print("  E_modulus: Young's modulus (MPa)")
        print("  E_B_modulus: B-basis Young's modulus (MPa)")
        print("  ultimate_strength: Ultimate tensile strength (MPa)")
        print("  yield_strength: Yield strength (MPa)")
        print("  template_csv: Path to template CSV file containing material properties")
        sys.exit(1)

    print("Structural Analysis Suite")
    print("="*60)
    print(f"Input Files:")
    print(f"  1D Stress Data:     {csv_1d}")
    print(f"  2D/3D Stress Data:  {csv_2d3d}")
    print(f"Material Properties:")
    print(f"  E-modulus:          {E_modulus} MPa")
    print(f"  E-B-modulus:        {E_B_modulus} MPa")
    print(f"  Ultimate Strength:  {ultimate_strength} MPa")
    print(f"  Yield Strength:     {yield_strength} MPa")

    success_count = 0
    total_scripts = 3

    # Convert values to strings for subprocess arguments
    E_modulus_str = str(E_modulus)
    E_B_modulus_str = str(E_B_modulus)
    ultimate_strength_str = str(ultimate_strength)
    yield_strength_str = str(yield_strength)

    # Run Script D - Stress Analysis with Ultimate Strength
    if run_script("script_d_p11.py", [csv_1d, csv_2d3d, ultimate_strength_str]):
        success_count += 1

    # Run Script E - Panel Buckling Analysis with E-B-modulus
    if run_script("script_e_p11.py", [csv_1d, csv_2d3d, E_B_modulus_str]):
        success_count += 1

    # Run Script F - Column Buckling Analysis with E-B-modulus and Yield Strength
    if run_script("script_f_p11.py", [csv_1d, csv_2d3d, E_B_modulus_str, yield_strength_str]):
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
        print("  - Submission_Column_Buckling_U_Section.csv")
    else:
        print("⚠️  Some analyses failed. Check output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
