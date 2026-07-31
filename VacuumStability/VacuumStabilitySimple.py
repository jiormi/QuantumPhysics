import numpy as np
import matplotlib.pyplot as plt

# 1. Define the Higgs potential function (the "Hill")
# We use a polynomial function to create a local minimum (False Vacuum)
# and a global minimum (True Vacuum).
def higgs_potential(phi, top_quark_pull):
    # phi represents the value of the Higgs field
    # Standard components of the potential
    base_potential = 0.05 * phi**4 - 0.4 * phi**3 + phi**2
    # The top quark mass acts as a downward pull at higher field values (larger phi)
    quantum_correction = -top_quark_pull * (phi**4)
    return base_potential + quantum_correction

# 2. Set up the Higgs field values for plotting
phi_values = np.linspace(-1, 6, 500)

# 3. Define three scenarios for the Top Quark mass
# scenario = top_quark_pull value
scenarios = {
    "Lighter Top Quark (Stable Universe)": 0.00,
    "Actual Top Quark (Metastable Universe)": 0.02,
    "Heavier Top Quark (Unstable Universe)": 0.045
}

# 4. Plot the results
plt.figure(figsize=(12, 8))

for title, pull in scenarios.items():
    potential = higgs_potential(phi_values, pull)
    plt.plot(phi_values, potential, label=title, linewidth=2.5)

# Add annotations to explain the "Physics" of the plot
# Our current universe sits around phi = 1.5
plt.annotate('Our Universe\n(False Vacuum)', xy=(1.4, 0.7), xytext=(0.2, 2.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

plt.annotate('The Doomsday\n(True Vacuum)', xy=(4.8, -2.0), xytext=(5.0, 1.0),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6))

# Formatting the graph
plt.title("Toy Model: How the Top Quark Mass Reshapes the Universe", fontsize=14, fontweight='bold')
plt.xlabel("Higgs Field Value ($\phi$)", fontsize=12)
plt.ylabel("Potential Energy $V(\phi)$", fontsize=12)
plt.ylim(-5, 6)
plt.xlim(-0.5, 6)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11, loc='upper right')

# Show the plot
plt.show()
