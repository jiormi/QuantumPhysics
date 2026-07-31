import numpy as np
import matplotlib.pyplot as plt

def apply_pulse(state, theta):
    # Rotation matrix on the Bloch sphere
    # theta = pi is a full flip; theta = pi/2 is a 50/50 splitter
    matrix = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                       [-1j*np.sin(theta/2), np.cos(theta/2)]])
    return np.dot(matrix, state)

def apply_phase(state, phi):
    # Phase accumulation matrix (the 'ticking' of nature)
    matrix = np.array([[1, 0],
                       [0, np.exp(1j * phi)]])
    return np.dot(matrix, state)

# Phase range (representing time or gravity shift)
phases = np.linspace(0, 4 * np.pi, 200)

# 1. ATOMIC CLOCK (Ramsey Sequence: pi/2 -> Phase -> pi/2)
clock_probs = []
for phi in phases:
    atom = np.array([1, 0], dtype=complex) 
    atom = apply_pulse(atom, np.pi/2)       
    atom = apply_phase(atom, phi)           
    atom = apply_pulse(atom, np.pi/2)       
    clock_probs.append(np.abs(atom[1])**2)

# 2. ATOM INTERFEROMETER (MZ Sequence: pi/2 -> Mirror -> Phase -> pi/2)
interf_probs = []
for phi in phases:
    atom = np.array([1, 0], dtype=complex) 
    atom = apply_pulse(atom, np.pi/2)       
    atom = apply_pulse(atom, np.pi)         
    atom = apply_phase(atom, phi)           
    atom = apply_pulse(atom, np.pi/2)       
    interf_probs.append(np.abs(atom[1])**2)

# Plotting the Fringes
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(phases/np.pi, clock_probs, 'b')
plt.title('Atomic Clock Fringes')
plt.xlabel('Phase Shift (units of $\pi$)')
plt.ylabel('Excitation Prob $P_e$')

plt.subplot(1, 2, 2)
plt.plot(phases/np.pi, interf_probs, 'r')
plt.title('Interferometer Fringes')
plt.xlabel('Phase Shift (units of $\pi$)')
plt.ylabel('Excitation Prob $P_e$')
plt.tight_layout()
plt.savefig('quantum_fringes.png')
