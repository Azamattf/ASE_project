import numpy as np

# --- 1. INPUTS ---
# These are the properties for a single unidirectional (UD) ply.
E1 = 131203.67   # Modulus in fiber direction (MPa)
E2 = 10092.59    # Modulus transverse to fiber (MPa)
G12 = 5046.3     # Shear modulus (MPa)
v12 = 0.35       # Major Poisson's ratio

# --- 2. LAMINATE DEFINITION ---
# Define the layup for one half of the symmetric laminate.
# The full layup is (+45/+45/-45/-45/0/0/90/90)s
ply_angles_deg_half = [45, 45, -45, -45, 0, 0, 90, 90]
ply_thickness = 0.250  # (mm)

# Construct the full, 16-ply symmetric laminate stack
ply_angles_deg = ply_angles_deg_half + ply_angles_deg_half[::-1]
num_plies = len(ply_angles_deg)
t_total = num_plies * ply_thickness

print(f"--- Laminate Definition ---")
print(f"Full layup: {ply_angles_deg}")
print(f"Total number of plies: {num_plies}")
print(f"Total laminate thickness: {t_total:.3f} mm\n")


# --- 3. LAMINA PROPERTIES (in local 1-2 coordinate system) ---
# Calculate minor Poisson's ratio using the reciprocity relation
v21 = v12 * E2 / E1

# Calculate the components of the lamina stiffness matrix [Q]
# These formulas are from Formulary_ASE_v1.pdf, page 5
Q11 = E1 / (1 - v12 * v21)
Q22 = E2 / (1 - v12 * v21)
Q12 = v12 * E2 / (1 - v12 * v21)
Q66 = G12
Q = np.array([[Q11, Q12, 0],
              [Q12, Q22, 0],
              [0,   0,   Q66]])

print("--- Lamina Stiffness Matrix [Q] (MPa) ---")
print(np.round(Q, 2))


# --- 4. LAMINATE ANALYSIS (in global x-y coordinate system) ---
# Initialize the extensional stiffness matrix [A]
A = np.zeros((3, 3))

# Define the z-coordinates for each ply interface, measured from the laminate mid-plane
z_coords = np.linspace(-t_total / 2, t_total / 2, num_plies + 1)

print("\n--- Calculating [A] Matrix ---")
# Loop through each ply of the full laminate
for i, angle_deg in enumerate(ply_angles_deg):
    # Convert angle to radians for trigonometric functions
    angle_rad = np.deg2rad(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    # Calculate the transformed stiffness matrix [Q_bar] for the current ply
    # using the explicit transformation equations from Formulary_ASE_v1.pdf, page 5.
    Q_bar_11 = Q11*c**4 + Q22*s**4 + 2*(Q12 + 2*Q66)*s**2*c**2
    Q_bar_22 = Q11*s**4 + Q22*c**4 + 2*(Q12 + 2*Q66)*s**2*c**2
    Q_bar_12 = (Q11 + Q22 - 4*Q66)*s**2*c**2 + Q12*(s**4 + c**4)
    Q_bar_66 = (Q11 + Q22 - 2*Q12 - 2*Q66)*s**2*c**2 + Q66*(s**4 + c**4)
    Q_bar_16 = (Q11 - Q12 - 2*Q66)*s*c**3 + (Q12 - Q22 + 2*Q66)*s**3*c
    Q_bar_26 = (Q11 - Q12 - 2*Q66)*s**3*c + (Q12 - Q22 + 2*Q66)*s*c**3
    
    Q_bar = np.array([[Q_bar_11, Q_bar_12, Q_bar_16],
                      [Q_bar_12, Q_bar_22, Q_bar_26],
                      [Q_bar_16, Q_bar_26, Q_bar_66]])

    # Add this ply's contribution to the [A] matrix
    # A_ij = sum over k (Q_bar_ij_k * (z_k+1 - z_k))
    # where (z_k+1 - z_k) is the ply thickness.
    ply_t = z_coords[i+1] - z_coords[i]
    A += Q_bar * ply_t

print("\n--- Extensional Stiffness Matrix [A] (N/mm) ---")
print(np.round(A, 2))

# --- 5. HOMOGENIZED PROPERTIES ---
# Invert the [A] matrix to get the compliance matrix [a] = [A]^-1
# Note: some texts use [a*] or [A*] for the inverted, thickness-normalized matrix.
# Here we use the direct definition from exercises.txt
A_inv = np.linalg.inv(A)

print("\n--- Inverted [A] Matrix (mm/N) ---")
print(A_inv)

# Calculate the homogenized engineering constants for the laminate
# These formulas are from exercises.txt, Part 3.1
E_x_homogenized = 1 / (A_inv[0, 0] * t_total)
G_xy_homogenized = 1 / (A_inv[2, 2] * t_total)

# --- 6. RESULTS ---
print("\n" + "="*50)
print("            HOMOGENIZED LAMINATE PROPERTIES")
print("="*50)
print(f"Total thickness of the laminate (t_total): {t_total:.3f} mm")
print(f"Calculated Homogenized Axial Modulus (E_x): {E_x_homogenized:.2f} MPa")
print(f"Calculated Homogenized Shear Modulus (G_xy): {G_xy_homogenized:.2f} MPa")
print("="*50 + "\n")
