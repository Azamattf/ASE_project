import numpy as np

def calculate_abd_matrix(ud_properties, layup_half, ply_thickness):
    """
    Calculates the [A], [B], [D] and their inverse matrices for a symmetric laminate.

    This function implements Classical Lamination Theory (CLT) based on the course materials:
    - Formulas from: Formulary_ASE_v1.pdf, Pages 5-6

    Args:
        ud_properties (dict): Dictionary with the UD ply's E1, E2, G12, v12.
        layup_half (list): List of angles for the top half of the symmetric laminate.
        ply_thickness (float): The thickness of a single ply (mm).

    Returns:
        tuple: A tuple containing the following numpy arrays:
            - A (3x3): Extensional stiffness matrix.
            - B (3x3): Coupling stiffness matrix.
            - D (3x3): Bending stiffness matrix.
            - A_inv (3x3): Inverse of the extensional stiffness matrix.
            - D_inv (3x3): Inverse of the bending stiffness matrix.
    """
    print("\n" + "="*60)
    print("RUNNING CLASSICAL LAMINATION THEORY (CLT) CALCULATION")
    print("="*60)

    # --- 1. Complete Stacking Sequence and z-coordinates ---
    stacking_sequence = layup_half + layup_half[::-1]
    num_plies = len(stacking_sequence)
    total_thickness = num_plies * ply_thickness
    
    # z-coordinates are measured from the laminate mid-plane
    z = np.linspace(-total_thickness / 2, total_thickness / 2, num_plies + 1)

    # --- 2. Calculate Reduced Stiffness Matrix [Q] ---
    # Source: Formulary_ASE_v1.pdf, Page 5
    E1, E2, G12, v12 = ud_properties['E1'], ud_properties['E2'], ud_properties['G12'], ud_properties['v12']
    v21 = v12 * E2 / E1
    
    Q11 = E1 / (1 - v12 * v21)
    Q22 = E2 / (1 - v12 * v21)
    Q12 = v12 * E2 / (1 - v12 * v21)
    Q66 = G12
    Q = np.array([[Q11, Q12, 0], [Q12, Q22, 0], [0, 0, Q66]])
    print("Calculated [Q] Matrix:\n", Q.round(2))

    # --- 3. Calculate [A], [B], [D] matrices by iterating through plies ---
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))

    for k, angle_deg in enumerate(stacking_sequence):
        # Convert angle to radians for trigonometric functions
        angle_rad = np.deg2rad(angle_deg)
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)

        # Transformation Matrix [T] components for Q_bar calculation
        c2 = c**2
        s2 = s**2
        c4 = c**4
        s4 = s**4
        
        # Calculate Transformed Stiffness Matrix [Q_bar] for the k-th ply
        # Source: Formulary_ASE_v1.pdf, Page 5
        Q_bar11 = Q11*c4 + 2*(Q12 + 2*Q66)*s2*c2 + Q22*s4
        Q_bar22 = Q11*s4 + 2*(Q12 + 2*Q66)*s2*c2 + Q22*c4
        Q_bar12 = (Q11 + Q22 - 4*Q66)*s2*c2 + Q12*(c4 + s4)
        Q_bar66 = (Q11 + Q22 - 2*Q12 - 2*Q66)*s2*c2 + Q66*(c4 + s4)
        Q_bar16 = (Q11 - Q12 - 2*Q66)*s*c**3 - (Q22 - Q12 + 2*Q66)*s**3*c
        Q_bar26 = (Q11 - Q12 - 2*Q66)*s**3*c - (Q22 - Q12 + 2*Q66)*s*c**3
        
        Q_bar = np.array([[Q_bar11, Q_bar12, Q_bar16],
                          [Q_bar12, Q_bar22, Q_bar26],
                          [Q_bar16, Q_bar26, Q_bar66]])

        # Integrate through thickness to get A, B, D
        # Source: Formulary_ASE_v1.pdf, Pages 5-6
        A += Q_bar * (z[k+1] - z[k])
        B += (1/2) * Q_bar * (z[k+1]**2 - z[k]**2)
        D += (1/3) * Q_bar * (z[k+1]**3 - z[k]**3)

    # --- 4. Calculate Inverse Matrices ---
    # The inverse of A is needed for calculating homogenized properties
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        print("Error: A matrix is singular and cannot be inverted.")
        A_inv = np.full((3, 3), np.nan)

    # The inverse of D is also needed for calculating homogenized properties
    try:
        D_inv = np.linalg.inv(D)
    except np.linalg.LinAlgError:
        print("Error: D matrix is singular and cannot be inverted.")
        D_inv = np.full((3, 3), np.nan)

    print("\nCalculated [A] Matrix (Extensional Stiffness):\n", A.round(2))
    print("\nCalculated [B] Matrix (Coupling Stiffness):\n", B.round(2))
    print("\nCalculated [D] Matrix (Bending Stiffness):\n", D.round(2))
    print("\nCalculated Inverse [A] Matrix:\n", A_inv)
    print("\nCalculated Inverse [D] Matrix:\n", D_inv)
    
    return A, D, A_inv, D_inv
