import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

# ==========================================
# 1. Physical Parameters & Constants
# ==========================================
TR = 0.5                  # Color factor for g -> q qbar
CA = 3.0                  # Color factor for g -> gg
CF = 4.0 / 3.0            # Color factor for q -> qg
nf = 3                    # Number of active quark flavors (u, d, s)
beta0 = 11.0 - (2.0/3.0) * nf 

Q0_2 = 1.0                # Initial non-perturbative scale in GeV^2
Q_final_2 = 100.0         # Final scale in GeV^2
alpha_s_Q02 = 0.35        # Strong coupling at Q0^2

# ==========================================
# 2. Running Coupling & Splitting Functions
# ==========================================
def get_alpha_s(Q2):
    """1-loop running alpha_s"""
    if Q2 <= Q0_2:
        return alpha_s_Q02
    log_ratio = np.log(Q2 / Q0_2)
    return alpha_s_Q02 / (1.0 + beta0 * (alpha_s_Q02 / (4.0 * np.pi)) * log_ratio)

def P_qq(z):
    """Splitting function for q -> q + g (momentum loss of valence quarks)"""
    z = np.clip(z, 1e-5, 0.999)
    return CF * (1.0 + z**2) / (1.0 - z)

def P_qg(z):
    """Splitting function for g -> q + qbar (sea quark generation)"""
    return TR * (z**2 + (1.0 - z)**2)

def P_gg(z):
    """Splitting function for g -> g + g (gluon self-replication)"""
    z = np.clip(z, 1e-5, 0.999)
    return 2.0 * CA * (z / (1.0 - z) + (1.0 - z) / z + z * (1.0 - z))

# ==========================================
# 3. Initial Proton PDFs at Q0^2
# ==========================================
def valence_initial(y):
    """Valence quarks: peak around x ~ 0.3, drop to 0 at x=1"""
    # Normalized toy representation of 2 u_v + 1 d_v
    return 3.0 * (y**0.5) * ((1.0 - y)**3) / 0.2

def g_initial(y):
    """Initial gluons: high density at low-x"""
    return 12.0 * (1.0 - y)**5

# ==========================================
# 4. Multi-Step Coupled Proton Evolution
# ==========================================
x_grid = np.geomspace(0.01, 0.95, 100)

# Initialize grid arrays
val_grid = valence_initial(x_grid)
g_grid = g_initial(x_grid)
sea_grid = np.full_like(x_grid, 1e-5) # Initialize sea quarks close to 0

num_steps = 15
t_start = np.log(Q0_2)
t_end = np.log(Q_final_2)
dt = (t_end - t_start) / num_steps

# Save initial values for final comparison plots
xval_initial = x_grid * valence_initial(x_grid)
xg_initial = x_grid * g_initial(x_grid)
xsea_initial = x_grid * 1e-5

for step in range(num_steps):
    t_current = t_start + step * dt
    Q2_current = np.exp(t_current)
    as_current = get_alpha_s(Q2_current)
    prefactor = as_current / (2.0 * np.pi)
    
    # Interpolate current grids
    val_interp = interp1d(x_grid, val_grid, kind='cubic', fill_value="extrapolate")
    g_interp = interp1d(x_grid, g_grid, kind='cubic', fill_value="extrapolate")
    
    dval_dt = np.zeros_like(x_grid)
    dsea_dt = np.zeros_like(x_grid)
    dg_dt = np.zeros_like(x_grid)
    
    for i, x in enumerate(x_grid):
        upper_limit = 0.999 if x < 0.999 else x
        if x >= upper_limit:
            continue
            
        # 1. Valence Quark Evolution (Shifts left due to radiating gluons)
        integrand_val = lambda y: (1.0 / y) * P_qq(x / y) * val_interp(y)
        int_val, _ = quad(integrand_val, x, upper_limit, limit=50)
        # We subtract the self-energy loss of the valence quark at x
        dval_dt[i] = prefactor * (int_val - val_interp(x) * quad(P_qq, 0, upper_limit)[0])
        
        # 2. Sea Quark Evolution (Generated from the gluon reservoir)
        integrand_sea = lambda y: (1.0 / y) * P_qg(x / y) * g_interp(y)
        int_sea, _ = quad(integrand_sea, x, upper_limit, limit=50)
        dsea_dt[i] = prefactor * int_sea
        
        # 3. Gluon Evolution (Replicated by g -> gg, fed by radiated gluons from valence)
        integrand_g_self = lambda y: (1.0 / y) * P_gg(x / y) * g_interp(y)
        int_g_self, _ = quad(integrand_g_self, x, upper_limit, limit=50)
        dg_dt[i] = prefactor * int_g_self

    # Update all distributions simultaneously
    val_grid += dval_dt * dt
    # Keep valence physically positive
    val_grid = np.clip(val_grid, 1e-5, None)
    
    sea_grid += dsea_dt * dt
    g_grid += dg_dt * dt

# ==========================================
# 5. Visualization (Log-Log Scale)
# ==========================================
plt.figure(figsize=(12, 7))

# Initial State Plotting
plt.plot(x_grid, xg_initial, label=r'Initial Gluon: $x \cdot g(x, Q_0^2)$', color='orange', linestyle='--')
plt.plot(x_grid, xval_initial, label=r'Initial Valence: $x \cdot q_v(x, Q_0^2)$', color='green', linestyle='--')
plt.plot(x_grid, xsea_initial, label=r'Initial Sea: $x \cdot q_s(x, Q_0^2) \approx 0$', color='blue', linestyle='--')

# Evolved State Plotting
plt.plot(x_grid, x_grid * g_grid, label=r'Evolved Gluon: $x \cdot g(x, Q^2)$', color='red', linewidth=2.5)
plt.plot(x_grid, x_grid * val_grid, label=r'Evolved Valence: $x \cdot q_v(x, Q^2)$', color='darkgreen', linewidth=2.5)
plt.plot(x_grid, x_grid * sea_grid, label=r'Evolved Sea: $x \cdot q_s(x, Q^2)$', color='blue', linewidth=2.5)

plt.xscale('log')
plt.yscale('log')

plt.title(f'Proton Structure Evolution ($Q_0^2 = {Q0_2}\\ \\mathrm{{GeV}}^2 \\to Q^2 = {Q_final_2}\\ \\mathrm{{GeV}}^2$)\nSimulating Valence Quarks, Gluon Self-Splitting, and Sea Generation')
plt.xlabel('Momentum Fraction ($x$, log scale)')
plt.ylabel('Momentum Density ($x \\cdot f(x)$, log scale)')
plt.xlim(0.01, 0.95)
plt.ylim(1e-4, 1000.0)
plt.grid(True, which="both", linestyle=':', alpha=0.5)
plt.legend(loc='lower left', ncol=2)
plt.show()
