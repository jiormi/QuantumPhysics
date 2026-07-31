import numpy as np
import matplotlib.pyplot as plt

# 1. Define the physically scaled Higgs potential function
def higgs_potential(phi, top_quark_pull):
    # phi is now in units of GeV
    # We normalize the math so the local minimum (our vacuum) sits at exactly 246 GeV
    phi_normalized = phi / 246.0
    
    # Base "Mexican Hat" potential: peak at 0, local valley at 1.0 (which is 246 GeV)
    base_potential = -2.0 * phi_normalized**2 + 1.0 * phi_normalized**4
    
    # The quantum correction from the top quark mass.
    # It acts as a negative term that dominates at ultra-high field values.
    quantum_correction = -top_quark_pull * (phi_normalized**6)
    
    return base_potential + quantum_correction

# 2. Set up the Higgs field values around the 246 GeV mark
# We look from 0 GeV up to about 500 GeV to see the shape clearly
phi_values = np.linspace(0, 550, 1000)

# 3. Define the scenarios based on the top quark's influence
scenarios = {
    "Lighter Top Quark (Stable: Current VEV is the absolute minimum)": 0.00,
    "Actual Top Quark (Metastable: Deeper canyon exists at higher energies)": 0.08,
    "Heavier Top Quark (Unstable: Barrier vanishes, field collapses immediately)": 0.25
}

# 4. Plotting
plt.figure(figsize=(12, 8))

for title, pull in scenarios.items():
    potential = higgs_potential(phi_values, pull)
    plt.plot(phi_values, potential, linewidth=2.5, label=title)

# Highlight the 246 GeV Vacuum Expectation Value (vev)
plt.axvline(x=246, color='purple', linestyle=':', alpha=0.7, label='Actual VEV (246 GeV)')
plt.plot(246, higgs_potential(246, 0.08), 'ko', markersize=8, label='Our Universe State')

# Annotations for physical accuracy
plt.annotate('Unstable Peak\n(Zero Field / $\phi=0$)', xy=(0, 0), xytext=(40, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6))

plt.annotate('Our Universe\n(False Vacuum at 246 GeV)', xy=(246, higgs_potential(246, 0.08)), xytext=(150, -3.5),
             arrowprops=dict(facecolor='purple', shrink=0.08, width=1, headwidth=6))

plt.annotate('The Drop toward\nTrue Vacuum', xy=(480, -6.0), xytext=(350, -8.0),
             arrowprops=dict(facecolor='red', shrink=0.08, width=1, headwidth=6))

# Formatting the graph
plt.title("Physically Scaled Toy Model: The Higgs Potential & Vacuum Stability", fontsize=14, fontweight='bold')
plt.xlabel("Higgs Field Vacuum Expectation Value $\phi$ (in GeV)", fontsize=12)
plt.ylabel("Relative Potential Energy $V(\phi)$", fontsize=12)
plt.ylim(-10, 4)
plt.xlim(-10, 550)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=10, loc='lower left')
plt.savefig("HiggsPotential_stability.png")
# Show the plot
plt.show()
