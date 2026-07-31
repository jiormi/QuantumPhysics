import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. CONFIGURACIÓN DE LA SIMULACIÓN ---
N_X = 30              # Dimensión espacial X
N_Y = 30              # Dimensión espacial Y
MUTATION_SIZE = 0.4   # Tamaño de la fluctuación propuesta
N_STEPS_PER_FRAME = 500  # Pasos Monte Carlo por fotograma

# Inicializar el campo de vacío
phi = np.zeros((N_X, N_Y))

# --- 2. CONFIGURACIÓN DEL ELECTRÓN ---
# Posición inicial (X, Y, Z)
electron_x = 0.0
electron_y = N_Y / 2.0  # Empieza en el medio del eje Y
electron_z = 0.0

# Velocidad inicial (se mueve principalmente a lo largo del eje X)
vx = 0.5
vy = 0.0

# Parámetros físicos del electrón
acoplamiento = 0.8  # Qué tanto le afectan las fluctuaciones al electrón
inercia = 0.95      # Amortiguación para estabilizar la trayectoria

def calcular_accion_local(phi, x, y, nuevo_valor):
    """Calcula el coste de energía (acción) del vacío"""
    v_arriba  = phi[(x - 1) % N_X, y]
    v_abajo   = phi[(x + 1) % N_X, y]
    v_izquierda = phi[x, (y - 1) % N_Y]
    v_derecha   = phi[x, (y + 1) % N_Y]
    
    val_antiguo = phi[x, y]
    
    accion_antes = (val_antiguo - v_arriba)**2 + (val_antiguo - v_abajo)**2 + \
                   (val_antiguo - v_izquierda)**2 + (val_antiguo - v_derecha)**2 + 0.5 * (val_antiguo**2)
                   
    accion_despues = (nuevo_valor - v_arriba)**2 + (nuevo_valor - v_abajo)**2 + \
                     (nuevo_valor - v_izquierda)**2 + (nuevo_valor - v_derecha)**2 + 0.5 * (nuevo_valor**2)
                     
    return accion_despues - accion_antes

def actualizar_fisica():
    """Actualiza el vacío por Monte Carlo y desplaza al electrón"""
    global phi, electron_x, electron_y, electron_z, vy
    
    # 1. Actualizar el vacío cuántico (Monte Carlo)
    for _ in range(N_STEPS_PER_FRAME):
        x = np.random.randint(0, N_X)
        y = np.random.randint(0, N_Y)
        cambio = np.random.uniform(-MUTATION_SIZE, MUTATION_SIZE)
        nuevo_valor = phi[x, y] + cambio
        delta_S = calcular_accion_local(phi, x, y, nuevo_valor)
        if delta_S <= 0 or np.random.rand() < np.exp(-delta_S):
            phi[x, y] = nuevo_valor
            
    # 2. Mover el electrón
    electron_x += vx
    
    # Si el electrón llega al final del mapa, reaparece al principio (bucle)
    if electron_x >= N_X:
        electron_x = 0.0
        electron_y = N_Y / 2.0
        vy = 0.0
        
    # Obtener las coordenadas enteras de la cuadrícula más cercanas al electrón
    ix = int(electron_x) % N_X
    iy = int(electron_y) % N_Y
    
    # Calcular la fuerza (gradiente local del campo en el eje Y)
    # Si el campo a la derecha es mayor que a la izquierda, empuja al electrón
    iy_sig = (iy + 1) % N_Y
    iy_ant = (iy - 1) % N_Y
    fuerza_y = -(phi[ix, iy_sig] - phi[ix, iy_ant]) * acoplamiento
    
    # Aplicar las leyes del movimiento (Newton modificado por el vacío)
    vy = (vy + fuerza_y) * inercia
    electron_y = (electron_y + vy) % N_Y
    
    # El electrón "flota" sobre la altura actual del campo cuántico
    electron_z = phi[ix, iy] + 0.1  # Un pequeño offset para que no se entierre

# --- 3. CONFIGURACIÓN DEL GRÁFICO 3D ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(np.arange(N_X), np.arange(N_Y))

ax.set_zlim(-2.5, 2.5)
ax.set_title("Trayectoria de un Electrón en el Vacío Cuántico", fontsize=14, fontweight='bold')
ax.set_xlabel("Espacio (Eje X - Vuelo)")
ax.set_ylabel("Espacio (Eje Y - Desviación)")
ax.set_zlabel("Amplitud del Campo ($\phi$)")

# Render inicial del vacío y del electrón
wframe = ax.plot_surface(X, Y, phi, cmap='coolwarm', edgecolor='none', alpha=0.8, antialiased=True)
electron_plot, = ax.plot([electron_x], [electron_y], [electron_z], marker='o', color='gold', markersize=10, label='Electrón', zorder=100)
ax.legend()

# --- 4. FUNCIÓN DE ANIMACIÓN ---
def update(frame):
    global wframe
    actualizar_fisica()
    
    # Actualizar la superficie del vacío
    wframe.remove()
    wframe = ax.plot_surface(X, Y, phi, cmap='coolwarm', edgecolor='none', alpha=0.8, antialiased=True)
    
    # Actualizar la posición de la partícula dorada
    electron_plot.set_data([electron_x], [electron_y])
    electron_plot.set_3d_properties([electron_z])
    
    # Rotación suave de la cámara
    ax.view_init(elev=25, azim=-45 + frame * 0.5)
    
    if frame % 10 == 0:
        print(f"Renderizando fotograma: {frame}/120...")
        
    return wframe, electron_plot

# Configurar y guardar la animación como GIF
ani = FuncAnimation(fig, update, frames=120, interval=50, blit=False)
nombre_archivo = "vuelo_electron_vacio.gif"

print("Iniciando renderizado del GIF con el electrón en movimiento...")
ani.save(nombre_archivo, writer='pillow', fps=20)
print(f"¡Completado! Archivo guardado como '{nombre_archivo}'")

