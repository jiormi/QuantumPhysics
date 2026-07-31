import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURACIÓN DEL TERRENO Y ANOMALÍAS
# ==========================================
grid_size = 50  # Área de 50x50 metros
x = np.linspace(-25, 25, grid_size)
y = np.linspace(-25, 25, grid_size)
X, Y = np.meshgrid(x, y)

# Campo magnético terrestre de fondo (en nT - nanoteslas)
B_fondo = 50000.0 

# Anomalía 1: Un objeto metálico sepultado en (x=10, y=10)
anomalia_objeto = 150 * np.exp(-((X - 10)**2 + (Y - 10)**2) / 15.0)

# Anomalía 2: Una tubería enterrada en diagonal
anomalia_tuberia = 80 * np.exp(-((X + Y + 10)**2) / 20.0)

# Campo magnético real total en el suelo
B_real = B_fondo + anomalia_objeto + anomalia_tuberia

# ==========================================
# 2. MODELADO DEL SENSOR CUÁNTICO
# ==========================================
def medicion_sensor_cuantico(campo_B, ruido_snr=5.0):
    """
    Simula la respuesta del sensor cuántico (Ramsey / Fluorescencia):
    Convierte el campo B real en un conteo de fotones.
    """
    gamma_t = 0.005  # Sensibilidad del sensor
    
    # Trabajamos en el punto de máxima pendiente (bias point)
    # Variación de fase respecto al campo de fondo
    delta_phi = gamma_t * (campo_B - B_fondo)
    
    # Cuentas de fotones (P = 0.5 equivale a 250 fotones de media)
    fotones_base = 500
    probabilidad = 0.5 * (1 + np.sin(delta_phi))
    
    # Ruido del conteo de fotones (Shot Noise / Poisson)
    ruido = np.random.normal(0, ruido_snr, size=campo_B.shape)
    
    cuentas_fotones = fotones_base * probabilidad + ruido
    return cuentas_fotones

# Simulamos el vuelo del dron tomando muestras
fotones_detectados = medicion_sensor_cuantico(B_real)

# ==========================================
# 3. RECONSTRUCCIÓN Y MAPPING 2D
# ==========================================
# El software del dron convierte el conteo de fotones a la anomalía re-calculada (en nT)
gamma_t = 0.005
probabilidad_medida = np.clip(fotones_detectados / 500.0, 0, 1)
anomalia_reconstruida = np.arcsin(2 * probabilidad_medida - 1) / gamma_t

# ==========================================
# 4. VISUALIZACIÓN
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfica 1: Lo que mide directamente el detector (Fotones)
im1 = ax1.imshow(fotones_detectados, extent=[-25, 25, -25, 25], origin='lower', cmap='plasma')
ax1.set_title("1. Photon counts", fontsize=12)
ax1.set_xlabel(" X (m)")
ax1.set_ylabel(" Y (me)")
fig.colorbar(im1, ax=ax1, label="N photons")

# Gráfica 2: El mapa de calor final procesado (Anomalía detectada)
im2 = ax2.imshow(anomalia_reconstruida, extent=[-25, 25, -25, 25], origin='lower', cmap='viridis')
ax2.set_title("2. 2D mapping of detected anomalies (nT)", fontsize=12)
ax2.set_xlabel(" X (m)")
ax2.set_ylabel(" Y (m)")

# Marcamos las anomalías encontradas
ax2.annotate('Metalic object', xy=(10, 10), xytext=(12, 18),
             arrowprops=dict(facecolor='white', shrink=0.05, width=1.5, headwidth=8),
             color='white', fontweight='bold')

ax2.annotate('Underground pipe', xy=(-10, 0), xytext=(-22, -10),
             arrowprops=dict(facecolor='white', shrink=0.05, width=1.5, headwidth=8),
             color='white', fontweight='bold')

fig.colorbar(im2, ax=ax2, label=" ΔB (nT)")

plt.tight_layout()
plt.show()
