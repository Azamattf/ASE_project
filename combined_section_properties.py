import numpy as np

def calculate_stringer_section_properties(
    A_skin_mat, D_skin_mat, A_flange_mat, D_flange_mat, A_web_inv_mat, D_web_inv_mat, D_web_mat,
    flange_width_val, web_height_val, stringer_ply_thickness_val, skin_thickness_val, skin_eff_width_val, knockdown_factor
):
    """
    Calculates and prints the combined cross-sectional properties for a skin-stringer combination.
    """
    # --- Geometric Properties ---
    flange_thickness = 16 * stringer_ply_thickness_val
    web_thickness = 16 * stringer_ply_thickness_val

    # --- Homogenized Engineering Constants (WITH KNOCKDOWN FACTOR) ---
    # Skin Properties (Restricted Lateral Deformation)
    E_x_skin_calc = (A_skin_mat[0, 0] / skin_thickness_val) * knockdown_factor
    E_b_z_skin_calc = (D_skin_mat[0, 0] * 12 / skin_thickness_val**3) * knockdown_factor
    
    # Flange Properties (Restricted Lateral Deformation)
    E_x_flange_calc = (A_flange_mat[0, 0] / flange_thickness) * knockdown_factor
    E_y_b_flange_calc = (D_flange_mat[0, 0] * 12 / flange_thickness**3) * knockdown_factor
    E_z_b_flange_calc = E_x_flange_calc

    # Web Properties (Free in-plane, Restricted out-of-plane)
    E_x_web_calc = (1 / (A_web_inv_mat[0, 0] * web_thickness)) * knockdown_factor
    E_y_b_web_calc = (D_web_mat[0, 0] * 12 / web_thickness**3) * knockdown_factor  # With knockdown
    E_z_b_web_calc = E_x_web_calc

    # --- 2. SEGMENT PROPERTIES CALCULATION ---
    # Per user feedback, coordinate system is updated:
    # Origin is at the BOTTOM surface of the skin panel.
    # y-axis: horizontal along the panel width.
    # z-axis: vertical, POSITIVE UPWARDS from the skin's bottom surface.

    # Segment 1: Skin (effective width under stringer)
    area_skin = skin_eff_width_val * skin_thickness_val
    I_y_skin = (skin_eff_width_val * skin_thickness_val**3) / 12
    I_z_skin = (skin_thickness_val * skin_eff_width_val**3) / 12
    y_c_skin = 0.0
    z_c_skin = skin_thickness_val / 2.0  # Centroid is at half the skin thickness from the bottom

    # Segment 2: Flange (sits on top of skin)
    area_flange = flange_width_val * flange_thickness
    I_y_flange = (flange_width_val * flange_thickness**3) / 12
    I_z_flange = (flange_thickness * flange_width_val**3) / 12
    y_c_flange = 0.0
    z_c_flange = skin_thickness_val + (flange_thickness / 2.0) # Sits above skin

    # Segment 3: Web (extends upward from flange)
    area_web = web_height_val * web_thickness
    I_y_web = (web_thickness * web_height_val**3) / 12
    I_z_web = (web_height_val * web_thickness**3) / 12
    y_c_web = 0.0
    z_c_web = skin_thickness_val + flange_thickness + (web_height_val / 2.0) # Sits above flange

    print(f"z_c_web: {z_c_web}")
    print(f"z_c_flange: {z_c_flange}")
    print(f"z_c_skin: {z_c_skin}")



    # --- 3. ELASTIC CENTER (EC) CALCULATION ---
    y_ec_num = (E_x_skin_calc * area_skin * y_c_skin) + \
               (E_x_flange_calc * area_flange * y_c_flange) + \
               (E_x_web_calc * area_web * y_c_web)
    z_ec_num = (E_x_skin_calc * area_skin * z_c_skin) + \
               (E_x_flange_calc * area_flange * z_c_flange) + \
               (E_x_web_calc * area_web * z_c_web)
    ec_den = (E_x_skin_calc * area_skin) + (E_x_flange_calc * area_flange) + (E_x_web_calc * area_web)

    y_EC = y_ec_num / ec_den
    z_EC = z_ec_num / ec_den


    # --- 4. COMBINED BENDING STIFFNESS (EI) CALCULATION ---
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
    EI_y = (E_b_z_skin_calc * I_y_skin + E_x_skin_calc * area_skin * z_skin_EC**2) + \
           (E_y_b_flange_calc * I_y_flange + E_x_flange_calc * area_flange * z_flange_EC**2) + \
           (E_x_web_calc * I_y_web + E_x_web_calc * area_web * z_web_EC**2)

    # (EI)_z: Bending about z-axis (strong axis bending of stringer)
    # Skin/Flange bend in-plane. Web bends out-of-plane (weak).
    EI_z = (E_x_skin_calc * I_z_skin + E_x_skin_calc * area_skin * y_skin_EC**2) + \
           (E_x_flange_calc * I_z_flange + E_x_flange_calc * area_flange * y_flange_EC**2) + \
           (E_y_b_web_calc * I_z_web + E_x_web_calc * area_web * y_web_EC**2)


    # --- 5. RESULTS ---
    print("="*60)
    print("      COMBINED SKIN-STRINGER CROSS-SECTIONAL PROPERTIES")
    print("="*60)
    print("\n--- Geometric & Material Inputs ---")
    print(f"Skin Dimensions (w x t):    {skin_eff_width_val:.2f} x {skin_thickness_val:.2f} mm")
    print(f"Flange Dimensions (w x t):  {flange_width_val:.2f} x {flange_thickness:.2f} mm")
    print(f"Web Dimensions (h x t):     {web_height_val:.2f} x {web_thickness:.2f} mm")
    print(f"Knockdown Factor:           {knockdown_factor:.2f}")

    print("\n--- Homogenized Elastic Moduli (with knockdown) ---")
    print(f"E_x_skin:        {E_x_skin_calc:.2f} MPa")
    print(f"E_b_z_skin:      {E_b_z_skin_calc:.2f} MPa")
    print(f"E_x_flange:      {E_x_flange_calc:.2f} MPa")
    print(f"E_y_b_flange:    {E_y_b_flange_calc:.2f} MPa")
    print(f"E_z_b_flange:    {E_z_b_flange_calc:.2f} MPa")
    print(f"E_x_web:         {E_x_web_calc:.2f} MPa")
    print(f"E_y_b_web:       {E_y_b_web_calc:.2f} MPa")
    print(f"E_z_b_web:       {E_z_b_web_calc:.2f} MPa")

    print("\n--- Elastic Center (EC) ---")
    print(f"The Elastic Center is located at (y, z): ({y_EC:.3f}, {z_EC:.3f}) mm")
    print("(Origin is at the BOTTOM of the skin, z is positive UPWARDS)")

    print("\n--- Combined Bending Stiffness (EI) ---")
    print(f"Bending Stiffness about y-axis (EI_y): {EI_y:.3f} N*mm^2")
    print(f"Bending Stiffness about z-axis (EI_z): {EI_z:.3e} N*mm^2")
    print("="*60)

    if EI_y < EI_z:
        print("\nBuckling will occur about the y-axis (EI_y is weaker).")
    else:
        print("\nBuckling will occur about the z-axis (EI_z is weaker).")
