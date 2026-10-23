import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Setup figure and canvas
fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
fig.patch.set_facecolor('#050510')
ax.set_facecolor('#050510')

# Simulation parameters
n_particles = 1500
n_frames = 40
np.random.seed(42)

# Asymmetric radial velocity field
theta = np.random.uniform(0, 2 * np.pi, n_particles)
asymmetry = 1 + 0.2 * np.cos(3 * theta)  # Tri-lobed ejection profile
v_radial = np.random.normal(1.2, 0.25, n_particles) * asymmetry
v_radial = np.clip(v_radial, 0.3, 2.5)

# Initial offsets around core
x_0 = np.random.normal(0, 0.2, n_particles)
y_0 = np.random.normal(0, 0.2, n_particles)

# Background grid for continuous gas shockwave
grid_size = 200
gx = np.linspace(-40, 40, grid_size)
gy = np.linspace(-40, 40, grid_size)
GX, GY = np.meshgrid(gx, gy)
R_grid = np.sqrt(GX**2 + GY**2)

# Render objects
im = ax.imshow(np.zeros((grid_size, grid_size)), extent=[-40, 40, -40, 40], 
               cmap='magma', origin='lower', vmin=0, vmax=1, alpha=0.85)

scatter = ax.scatter([], [], c=[], cmap='plasma', s=[], alpha=0.9, edgecolors='none')
core = ax.plot([0], [0], marker='o', markersize=4, color='cyan', alpha=0.9)[0]

ax.set_xlim(-40, 40)
ax.set_ylim(-40, 40)
ax.axis('off')

def update(frame):
    t = frame + 1
    
    # 1. Shockwave density shell (Expanding Gaussian ring)
    r_shock = 0.9 * t
    sigma = 1.5 + 0.08 * t
    shock_density = np.exp(-((R_grid - r_shock)**2) / (2 * sigma**2))
    shock_density *= np.exp(-t / 25)  # Radiative cooling
    im.set_data(shock_density)
    
    # 2. Particle kinematics
    x = x_0 + v_radial * np.cos(theta) * t * 0.95
    y = y_0 + v_radial * np.sin(theta) * t * 0.95
    
    # 3. Thermal spectrum shifting (hot yellow/white -> cooling red)
    temp_factor = np.clip(1.0 - t / (n_frames * 1.1), 0.1, 1.0)
    colors = plt.cm.inferno(v_radial / np.max(v_radial) * temp_factor)
    sizes = np.maximum(0.5, 25 * np.exp(-t / 30))
    
    scatter.set_offsets(np.column_stack([x, y]))
    scatter.set_color(colors)
    scatter.set_sizes(np.full(n_particles, sizes))
    
    # 4. Central remnant pulse decay
    core_size = max(2, 8 * (1 - t / n_frames))
    core.set_markersize(core_size)
    
    return im, scatter, core

# Generate and save animation
anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=80, blit=True)
anim.save('supernova_simulation.gif', writer='pillow', fps=15)
