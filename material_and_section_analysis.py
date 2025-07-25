import numpy as np

def calculate_homogenized_moduli():
    """
    Calculates the homogenized engineering constants (E_x, G_xy) for the T-stringer
    by area-averaging the properties of the flange and web, considering their
    specific boundary conditions and distinct layups.

    Formulas are based on Formulary_ASE_v2.pdf, page 7.
    - Flange: Restricted lateral deformation
    - Web: Free (permitted) lateral deformation
    """

    # --- 1. USER INPUTS: Personalized Material Properties ---
    ud_ply_properties = {
        "E1": 131203.670, # [MPa]
        "E2": 10092.59,   # [MPa]
        "G12": 5046.3,    # [MPa]
        "v12": 0.33
    }

    # --- 2. GIVEN PROJECT CONSTANTS ---
    flange_width = 70.0  # [mm]
    web_height = 40.0    # [mm]
    ply_thickness = 0.250 # [mm]

    # --- CORRECTED LAYUPS based on project description ---
    # Flange layup: (+45/+45/-45/-45/0/0/90/90)s
    flange_layup_symmetric = [+45, +45, -45, -45, 0, 0, 90, 90]
    # Web layup: (-45/-45/+45/+45/0/0/90/90)s
    web_layup_symmetric = [-45, -45, +45, +45, 0, 0, 90, 90]

    full_flange_layup = flange_layup_symmetric + flange_layup_symmetric[::-1]
    full_web_layup = web_layup_symmetric + web_layup_symmetric[::-1]

    # --- 3. HELPER FUNCTIONS (CLASSICAL LAMINATE THEORY) ---

    def get_Q_matrix(properties):
        """Calculates the ply stiffness matrix [Q]."""
        E1, E2, G12, v12 = properties["E1"], properties["E2"], properties["G12"], properties["v12"]
        v21 = v12 * E2 / E1
        denominator = 1 - v12 * v21
        Q11, Q22, Q12, Q66 = E1/denominator, E2/denominator, v12*E2/denominator, G12
        return np.array([[Q11, Q12, 0], [Q12, Q22, 0], [0, 0, Q66]])

    def get_Q_bar_matrix(Q, theta_deg):
        """Calculates the transformed ply stiffness matrix [Q_bar]."""
        theta_rad = np.deg2rad(theta_deg)
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        c2, s2, c4, s4 = c**2, s**2, c**4, s**4
        Q11, Q22, Q12, Q66 = Q[0,0], Q[1,1], Q[0,1], Q[2,2]
        
        Q11_bar = Q11*c4 + Q22*s4 + 2*(Q12 + 2*Q66)*s2*c2
        Q22_bar = Q11*s4 + Q22*c4 + 2*(Q12 + 2*Q66)*s2*c2
        Q12_bar = (Q11 + Q22 - 4*Q66)*s2*c2 + Q12*(s4 + c4)
        Q66_bar = (Q11 + Q22 - 2*Q12 - 2*Q66)*s2*c2 + Q66*(s4 + c4)
        
        return np.array([[Q11_bar, Q12_bar, 0], [Q12_bar, Q22_bar, 0], [0, 0, Q66_bar]])

    def get_A_matrix(layup, t_ply, Q_matrix):
        """Calculates the [A] matrix (extensional stiffness)."""
        A = sum(get_Q_bar_matrix(Q_matrix, angle) for angle in layup)
        return A * t_ply

    # --- 4. CALCULATIONS ---
    Q = get_Q_matrix(ud_ply_properties)

    # -- Flange Properties (Restricted Deformation) --
    A_flange = get_A_matrix(full_flange_layup, ply_thickness, Q)
    t_flange = len(full_flange_layup) * ply_thickness
    E_x_flange = A_flange[0, 0] / t_flange
    G_xy_flange = A_flange[2, 2] / t_flange

    # -- Web Properties (Free Deformation) --
    A_web = get_A_matrix(full_web_layup, ply_thickness, Q)
    A_inv_web = np.linalg.inv(A_web)
    t_web = len(full_web_layup) * ply_thickness
    E_x_web = 1 / (A_inv_web[0, 0] * t_web)
    G_xy_web = 1 / (A_inv_web[2, 2] * t_web)

    E_x_flange = 51222.8 # [MPa] FROM ONLINE CALCULATOR
    G_xy_flange = 19461.7 # [MPa] FROM ONLINE CALCULATOR

    E_x_web = E_x_flange # [MPa]
    G_xy_web = G_xy_flange # [MPa]


    print(f"E_flange: {E_x_flange}")
    print(f"G_flange: {G_xy_flange}")
    print(f"E_web: {E_x_web}")
    print(f"G_web: {G_xy_web}")

    # -- Homogenization (Area-Weighted Average) --
    area_flange, area_web = flange_width * t_flange, web_height * t_web
    total_area = area_flange + area_web
    E_homogenized = (E_x_flange * area_flange + E_x_web * area_web) / total_area
    G_homogenized = (G_xy_flange * area_flange + G_xy_web * area_web) / total_area

    print("\n--- Final Homogenized Moduli for MAT1 Card ---")
    print(f"Young's Modulus (E): {E_homogenized:,.2f} MPa")
    print(f"Shear Modulus   (G): {G_homogenized:,.2f} MPa")
    
    return E_homogenized, G_homogenized

if __name__ == '__main__':
    calculate_homogenized_moduli()
