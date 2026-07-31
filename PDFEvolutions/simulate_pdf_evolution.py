import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

# ==========================================
# 1. Physical Parameters & Constants
# ==========================================
TR = 0.5                  # Color factor for g -> q qbar
CA = 3.0                  # Color factor for g -> gg (SU(3) adjoint representation)
nf = 3                    # Number of active quark flavors
beta0 = 11.0 - (2.0/3.0) * nf 

Q0_2 = 1.0                # Initial scale in GeV^2
Q_final_2 = 100.0         # Final scale in GeV^2
alpha_s_Q02 = 0.35        # Strong coupling at Q0^2

# ==========================================
# 2. Running Coupling & Splitting Functions
# ==========================================
def get_alpha_s(Q2):
    """Calculates 1-loop running coupling alpha_s(Q^2)"""
    if Q2 <= Q0_2:
        return alpha_s_Q02
    log_ratio = np.log(Q2 / Q0_2)
    denom = 1.0 + beta0 * (alpha_s_Q02 / (4.0 * np.pi)) * log_ratio
    return alpha_s_Q02 / denom

def P_qg(z):
    """Splitting function for g -> q + qbar"""
    return TR * (z**2 + (1.0 - z)**2)

def P_gg(z):
    """Splitting function for g -> g + g (simplified away from z=1)"""
    z = np.clip(z, 1e-5, 0.999)
    return 2.0 * CA * (z / (1.0 - z) + (1.0 - z) / z + z * (1.0 - z))

# ==========================================
# 3. Initial Conditions
# ==========================================
def g_initial(y):
    """Initial gluon PDF at Q0^2"""
    return 10.0 * (1.0 - y)**5

# ==========================================
# 4. Multi-Step Coupled Evolution Engine
# ==========================================
x_grid = np.geomspace(0.01, 0.95, 100)

# Initialize grid arrays
g_grid = g_initial(x_grid)
# We use 1e-5 instead of exactly 0 so it can be cleanly plotted on a log-y scale
q_grid = np.full_like(x_grid, 1e-5)  

num_steps = 15
t_start = np.log(Q0_2)
t_end = np.log(Q_final_2)
dt = (t_end - t_start) / num_steps

# Copy initial states for plotting reference
xg_initial = x_grid * g_initial(x_grid)
xq_initial = x_grid * 1e-5

for step in range(num_steps):
    t_current = t_start + step * dt
    Q2_current = np.exp(t_current)
    as_current = get_alpha_s(Q2_current)
    prefactor = as_current / (2.0 * np.pi)
    
    # Interpolate current grids
    g_interp = interp1d(x_grid, g_grid, kind='cubic', fill_value="extrapolate")
    
    dq_dt = np.zeros_like(x_grid)
    dg_dt = np.zeros_like(x_grid)
    
    for i, x in enumerate(x_grid):
        upper_limit = 0.999 if x < 0.999 else x
        
        if x >= upper_limit:
            continue
            
        # 1. Quark Evolution (g -> q qbar)
        integrand_q = lambda y: (1.0 / y) * P_qg(x / y) * g_interp(y)
        int_q, _ = quad(integrand_q, x, upper_limit, limit=50)
        dq_dt[i] = prefactor * int_q
        
        # 2. Gluon Evolution (g -> gg)
        integrand_g = lambda y: (1.0 / y) * P_gg(x / y) * g_interp(y)
        int_g, _ = quad(integrand_g, x, upper_limit, limit=50)
        dg_dt[i] = prefactor * int_g

    # Simultaneously update both grids
    q_grid += dq_dt * dt
    g_grid += dg_dt * dt

# ==========================================
# 5. Plotting the Log-Log Coupled Evolution
# ==========================================
plt.figure(figsize=(11, 6))

# Gluon curves
plt.plot(x_grid, xg_initial, label=r'Initial Gluon: $x \cdot g(x, Q_0^2)$', color='orange', linestyle='--')
plt.plot(x_grid, x_grid * g_grid, label=r'Evolved Gluon: $x \cdot g(x, Q^2)$', color='red', linewidth=2)

# Quark curves
plt.plot(x_grid, xq_initial, label=r'Initial Sea Quark: $x \cdot q(x, Q_0^2) \approx 0$', color='blue', linestyle=':', alpha=0.5)
plt.plot(x_grid, x_grid * q_grid, label=r'Evolved Sea Quark: $x \cdot q(x, Q^2)$', color='blue', linewidth=2)

# Logarithmic scaling on both axes
plt.xscale('log')
plt.yscale('log')

plt.title(f'Coupled DGLAP Evolution ($Q_0^2 = {Q0_2}\\ \\mathrm{{GeV}}^2 \\to Q^2 = {Q_final_2}\\ \\mathrm{{GeV}}^2$)\n $g \\to q \\bar{{q}}$ and $g \\to gg$ branching processes')
plt.xlabel('Momentum Fraction ($x$, log scale)')
plt.ylabel('Momentum Density ($x \\cdot f(x)$, log scale)')
plt.xlim(0.01, 0.95)
plt.ylim(1e-4, 1000.0) # Confines the view to the physical region of interest
plt.grid(True, which="both", linestyle=':', alpha=0.5)
plt.legend(loc='lower left')
plt.show()
