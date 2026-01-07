import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# --- Physical Constants (SI Units) ---
hbar = 1.0545718e-34      # Reduced Planck's constant (J·s)
me = 9.10938356e-31       # Electron mass (kg)
e_charge = 1.60217663e-19 # Electron charge (Coulomb)
angstrom = 1e-10          # 1 Angstrom to meter
nm = 1e-9
meV = 1.60217663e-22      # 1 meV to Joules

# --- Initialize ---
m_star = 0.067 * me       
L = 15.1 * nm     
N1 = 150
dz = 1 * angstrom         
omega_z = 5e13            
B = 20
F = 5e4      

# z ranges from 0 to L. We solve for the interior points.
z = np.linspace(-L, L, N1 + 2)
z_int = z[1:-1]           # Interior points (Boundary conditions: psi=0 at ends)
N = len(z_int)

dz = z[1] - z[0]

# V_conf centered in the well: 1/2 * m* * omega^2 * (z - L/2)^2
V_conf = 0.5 * m_star * (omega_z**2) * (z_int)**2

# Magnetic term: (e^2 * B^2 / 2m*) * z^2
V_B = ((e_charge**2 * B**2) / (2 * m_star)) * (z_int)**2

# Electric term: -e * F * (z + L/2)
V_F = -e_charge * F * (z_int + L/2)

# Total Potential (V_H set to 0 as it usually requires iterative Poisson solving)
V_total = V_conf + V_B + V_F 
# V_total = V_total - np.min(V_total)
# V_total = V_conf
# --- Compute ---

# Kinetic energy factor
t = hbar**2 / (2 * m_star * dz**2)
# t = 1/(dz**2)

# Main diagonal and off-diagonals
diag = 2 * t * np.ones(N) + V_total
off_diag = -t * np.ones(N - 1)

# Solve the eigenvalue problem H*psi = E*psi
energies, wavefunctions = la.eigh_tridiagonal(diag, off_diag)

# --- 6. Results and Visualization ---
print(f"Ground state energy: {energies[0]/meV:.2f} meV")
print(f"First excited state: {energies[1]/meV:.2f} meV")
print(f"Vtotal: {V_total[0]/meV:.2f} meV")
plt.figure(figsize=(10, 6))
plt.plot(z_int/angstrom, V_total/meV, 'k--', label='Potential V(z) [meV]')
print(dz)
for i in range(7): # Plot first 3 states
    psi = wavefunctions[:, i]
    # Normalize and scale for visibility, then shift by energy level
    plt.plot((z_int)/angstrom, (psi * 1000) + (energies[i]/meV), 
             label=f'Level {i}: {energies[i]/meV:.1f} meV')

plt.title("1D Schrödinger Equation Solution (FDM)")
plt.xlabel("Position z ($\AA$)")
plt.ylabel("Energy (meV)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()