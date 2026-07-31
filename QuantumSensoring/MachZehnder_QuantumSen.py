import numpy as np
import matplotlib.pyplot as plt

def simular_sensor_cuantico(b_field, t_interaction, n_atoms=1):
    """
    Simula un sensor cuántico expuesto a un campo externo B.
    
    - b_field: Intensidad del campo a medir (Tesla / unidades arb.)
    - t_interaction: Tiempo que el sensor interactúa con el campo
    - n_atoms: Número de átomos entrelazados (1 = sin entrelazar, N = estado GHZ)
    """
    # 1. Constante de acoplamiento (gamma * B)
    # La fase acumulada phi es proporcional a B, al tiempo t y al entrelazamiento N
    gamma = 1.0  # Sensibilidad del sistema
    phi = gamma * b_field * t_interaction
    
    # 2. Evolución del Estado Cuántico
    # En un estado GHZ de N átomos, la fase colectiva se escala por N: (N * phi)
    fase_colectiva = n_atoms * phi
    
    # Vector de estado antes de la medida final:
    # |psi> = cos(N*phi/2)|0> + i*sin(N*phi/2)|1>
    prob_estado_1 = np.sin(fase_colectiva / 2)**2
    
    return prob_estado_1

# ==========================================
# SIMULACIÓN Y GRAFICACIÓN
# ==========================================

# Rango de campos magnéticos B que queremos escanear/detectar
b_range = np.linspace(0, 2 * np.pi, 500)
t_sensado = 1.0  # Tiempo fijo de interacción

# Caso 1: Sensor clásico / átomo individual (N = 1)
prob_n1 = simular_sensor_cuantico(b_range, t_sensado, n_atoms=1)

# Caso 2: Sensor cuántico entrelazado (N = 5 átomos en estado GHZ)
prob_n5 = simular_sensor_cuantico(b_range, t_sensado, n_atoms=5)

# Crear la figura
plt.figure(figsize=(10, 5))

# Dibujar las franjas de interferencia
plt.plot(b_range, prob_n1, label='1 atom (No entanglement)', color='#1f77b4', linewidth=2)
plt.plot(b_range, prob_n5, label='5 entangled atoms', color='#e377c2', linewidth=2, linestyle='--')

# Formato del gráfico
plt.title("Toy simulation of interference pattern for a quantum sensor", fontsize=14)
plt.xlabel("Intensity of external magnetic field $B$ (u.a.)", fontsize=12)
plt.ylabel("Detection probability $P(|1\\rangle)$", fontsize=12)
plt.axhline(0.5, color='gray', linestyle=':', alpha=0.6, label='Operating point')

plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=11)
plt.tight_layout()

plt.show()
