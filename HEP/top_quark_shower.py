"""
Top Quark Fragmentation Shower -> Hadronization Cascade -> Detector Hits
==========================================================================

A "vibe video" (stylized animation, not a physics plot) of:

  1. Top quark decay:           t -> b + W
  2. W decay:                   W -> q + qbar
  3. Parton shower:              each colored parton radiates more partons
                                  (q -> q g, g -> g g) with probability that
                                  falls with generation/energy, until a
                                  hadronization cutoff energy is reached
  4. Hadronization:               each final-state parton "showers" into a
                                  little cone (jet) of hadron dots
  5. Secondary decays:            some hadrons decay again mid-flight into a
                                  couple of softer daughter particles
  6. Detector:                    everything eventually flies out to a
                                  schematic semicircular detector made of
                                  concentric layers (Tracker / ECAL / HCAL /
                                  Muon system). Crossing a layer registers a
                                  flash + a permanent hit-dot, like a crude
                                  event display.

None of the splitting functions, angles, energy fractions, decay widths or
detector physics are accurate. It's meant to *look* like a cascading collider
event, not reproduce one.

Requirements:
    pip install matplotlib numpy

Usage:
    python top_quark_shower.py
    python top_quark_shower.py --out my_shower.mp4 --duration 14 --fps 30
"""

import argparse
import math
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Arc
import matplotlib.patheffects as pe

# ----------------------------------------------------------------------------
# Visual / physics constants
# ----------------------------------------------------------------------------
COLORS = {
    "t": "#ffd24a",     # top quark - gold
    "b": "#4ad1ff",     # b quark - cyan
    "W": "#c084fc",     # W boson - purple
    "q": "#ff5b6e",     # light quark - red
    "g": "#5dff8f",     # gluon - green
    "hadron": "#ffffff",
}

E_TOP = 500.0
E_CUTOFF = 1.0
MAX_GEN = 9
BASE_SPEED = 3.5
HADRON_SPEED_BOOST = 5.2   # so hadrons can cross the whole detector in frame

# Schematic semicircular detector: concentric arcs spanning these angles (deg)
DETECTOR_THETA = (-82, 82)
DETECTOR_LAYERS = [
    (1.0, "#3b6ea5", "INNER"),
    (3.3, "#e8b53b", "ECAL"),
    (4.6, "#7a4fd1", "HCAL"),
    (5.7, "#9a9a9a", "MUON"),
]
FLASH_LIFETIME = 9   # frames a hit-flash stays bright before fading out


class Parton:
    _next_id = 0

    def __init__(self, ptype, energy, x, y, angle, t_birth, generation, parent=None):
        self.id = Parton._next_id
        Parton._next_id += 1
        self.ptype = ptype
        self.energy = energy
        self.x0, self.y0 = x, y
        self.angle = angle
        self.t_birth = t_birth
        self.t_death = None
        self.generation = generation
        self.parent = parent
        self.children = []
        self.is_final = False
        self.speed = BASE_SPEED * (0.4 + 0.6 * math.sqrt(energy / E_TOP))

    def pos_at(self, t):
        dt = max(0.0, t - self.t_birth)
        r = self.speed * dt
        return self.x0 + r * math.cos(self.angle), self.y0 + r * math.sin(self.angle)


def split_probability(generation, energy):
    if energy < E_CUTOFF or generation >= MAX_GEN:
        return 0.0
    base = 0.85 * (energy / E_TOP) ** 0.25
    decay = 0.85 ** generation
    return min(0.95, base * decay)


def sample_z():
    z = np.random.beta(0.35, 0.35)
    return min(max(z, 0.05), 0.95)


def opening_angle(generation, energy):
    base = 0.9 / (1 + generation) ** 1.3
    jitter = random.uniform(0.6, 1.4)
    energy_factor = 0.5 + 0.5 * (E_TOP / max(energy, 1.0)) ** 0.15
    return base * jitter * energy_factor


def build_shower(parton, t_split_gap=0.75, jitter_gap=0.3):
    p_split = split_probability(parton.generation, parton.energy)
    will_split = random.random() < p_split
    gap = max(0.15, random.gauss(t_split_gap, jitter_gap))
    parton.t_death = parton.t_birth + gap

    if not will_split:
        parton.is_final = True
        return

    x, y = parton.pos_at(parton.t_death)
    z = sample_z()
    e1, e2 = parton.energy * z, parton.energy * (1 - z)
    dtheta = opening_angle(parton.generation, parton.energy)
    base_angle = parton.angle
    a1 = base_angle + dtheta * (1 - z)
    a2 = base_angle - dtheta * z

    if parton.ptype == "g":
        t1, t2 = ("g", "g") if random.random() < 0.6 else ("q", "g")
    else:
        t1, t2 = (parton.ptype, "g")

    c1 = Parton(t1, e1, x, y, a1, parton.t_death, parton.generation + 1, parent=parton)
    c2 = Parton(t2, e2, x, y, a2, parton.t_death, parton.generation + 1, parent=parton)
    parton.children = [c1, c2]
    build_shower(c1, t_split_gap, jitter_gap)
    build_shower(c2, t_split_gap, jitter_gap)


def build_full_event():
    random.seed()
    top = Parton("t", E_TOP, 0.0, 0.0, 0.0, 0.0, 0)
    top.t_death = 0.6

    z = np.random.uniform(0.35, 0.55)
    b_e, w_e = E_TOP * z, E_TOP * (1 - z)
    dtheta = 0.55
    b = Parton("b", b_e, 0.0, 0.0, dtheta / 2, top.t_death, 1, parent=top)
    w = Parton("W", w_e, 0.0, 0.0, -dtheta / 2, top.t_death, 1, parent=top)
    top.children = [b, w]
    build_shower(b)

    w.t_death = w.t_birth + 0.5
    x, y = w.pos_at(w.t_death)
    zz = np.random.uniform(0.3, 0.7)
    q1 = Parton("q", w.energy * zz, x, y, w.angle + 0.4, w.t_death, 2, parent=w)
    q2 = Parton("q", w.energy * (1 - zz), x, y, w.angle - 0.4, w.t_death, 2, parent=w)
    w.children = [q1, q2]
    build_shower(q1)
    build_shower(q2)

    all_partons = []

    def collect(p):
        all_partons.append(p)
        for c in p.children:
            collect(c)

    collect(top)
    return top, all_partons


def make_hadron_cloud(parton, n=10):
    """Initial jet of hadrons spawned where a final-state parton hadronizes."""
    x, y = parton.pos_at(parton.t_death)
    cloud = []
    for _ in range(n):
        ang = parton.angle + random.uniform(-0.5, 0.5)
        speed = parton.speed * HADRON_SPEED_BOOST * random.uniform(0.6, 1.3)
        cloud.append({
            "x0": x, "y0": y, "angle": ang, "speed": speed,
            "t0": parton.t_death, "size": random.uniform(8, 26),
        })
    return cloud


def add_secondary_decays(hadron_clouds, flight_time_window):
    """Some hadrons decay again mid-flight into 2 softer daughters."""
    secondary = []
    for h in hadron_clouds:
        if random.random() < 0.35:
            frac = random.uniform(0.25, 0.7)
            t_decay = h["t0"] + frac * flight_time_window
            dt = t_decay - h["t0"]
            r = h["speed"] * dt
            dx, dy = h["x0"] + r * math.cos(h["angle"]), h["y0"] + r * math.sin(h["angle"])
            for sign in (+1, -1):
                secondary.append({
                    "x0": dx, "y0": dy,
                    "angle": h["angle"] + sign * random.uniform(0.15, 0.45),
                    "speed": h["speed"] * random.uniform(0.6, 1.0),
                    "t0": t_decay,
                    "size": random.uniform(5, 14),
                })
    return secondary


# ----------------------------------------------------------------------------
# Animation
# ----------------------------------------------------------------------------
def run(out_path="top_quark_shower.mp4", duration=14.0, fps=30, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    top, all_partons = build_full_event()
    t_end_shower = max(p.t_death for p in all_partons)

    hadron_clouds = []
    for p in all_partons:
        if p.is_final:
            hadron_clouds.extend(make_hadron_cloud(p))

    outer_radius = DETECTOR_LAYERS[-1][0]
    avg_speed = np.mean([h["speed"] for h in hadron_clouds]) if hadron_clouds else 1.0
    flight_time_window = (outer_radius + 2.0) / max(avg_speed, 0.1)

    secondary_decays = add_secondary_decays(hadron_clouds, flight_time_window)
    all_hadron_particles = hadron_clouds + secondary_decays

    total_time = t_end_shower + flight_time_window * 0.9 + 1.0
    n_frames = int(duration * fps)

    # The shower phase (fast, ~5-7 time units) and the flight-to-detector phase
    # (slow, can be 10+ time units) get squashed unevenly if mapped linearly
    # over total_time -> the shower would whip by in a fraction of a second.
    # Instead, give the shower a fixed share of the *video* frames so it's
    # actually visible, then spend the rest of the frames on the flight/hits.
    SHOWER_FRAME_SHARE = 0.45
    n_shower_frames = max(1, int(n_frames * SHOWER_FRAME_SHARE))

    fig = plt.figure(figsize=(9, 9), facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("black")
    ax.set_xlim(-9.5, 9.5)
    ax.set_ylim(-9.5, 9.5)
    ax.axis("off")

    rng = np.random.default_rng(0)
    star_x = rng.uniform(-9.5, 9.5, 220)
    star_y = rng.uniform(-9.5, 9.5, 220)
    star_s = rng.uniform(0.5, 2.2, 220)
    ax.scatter(star_x, star_y, s=star_s, color="white", alpha=0.2, zorder=0)

    th1, th2 = DETECTOR_THETA
    for radius, color, name in DETECTOR_LAYERS:
        arc = Arc((0, 0), 2 * radius, 2 * radius, angle=0, theta1=th1, theta2=th2,
                  color=color, lw=2.0, alpha=0.55, zorder=1)
        ax.add_patch(arc)
        lx = (radius + 0.25) * math.cos(math.radians((th1 + th2) / 2 + 35))
        ly = (radius + 0.25) * math.sin(math.radians((th1 + th2) / 2 + 35))
        ax.text(lx, ly, name, color=color, fontsize=8, alpha=0.7,
                fontfamily="monospace", ha="center")

    title = ax.text(0, 9.0, "TOP QUARK SHOWER -> HADRONIZATION -> DETECTOR HITS",
                     color="white", fontsize=12.5, ha="center", weight="bold", alpha=0.9,
                     fontfamily="monospace",
                     path_effects=[pe.withStroke(linewidth=3, foreground="black")])
    subtitle = ax.text(0, -9.2, "", color="#9fd8ff", fontsize=10, ha="center",
                        fontfamily="monospace", alpha=0.8)

    glow_layers = [4, 8, 16]
    glow_alphas = [0.10, 0.18, 0.9]
    glow_scatters = [ax.scatter([], [], s=[], c=[], alpha=a, linewidths=0, zorder=4)
                     for a in glow_alphas]
    hadron_sc = ax.scatter([], [], s=[], c=[], alpha=0.8, linewidths=0, zorder=3)
    hit_marks_sc = ax.scatter([], [], s=[], c=[], alpha=0.9, linewidths=0, zorder=2)
    flash_sc = ax.scatter([], [], s=[], c=[], alpha=1.0, linewidths=0, zorder=5)

    trail_lines = {}
    for p in all_partons:
        line, = ax.plot([], [], lw=1.2, color=COLORS.get(p.ptype, "white"), alpha=0.55, zorder=2)
        trail_lines[p.id] = line

    hit_marks = []
    flashes = []
    crossed = {}

    def time_for_frame(frame):
        if frame <= n_shower_frames:
            # shower phase: stretched to fill its frame share, in slow-mo
            return (frame / n_shower_frames) * t_end_shower
        # flight/detector phase: fills the remaining frame share
        remaining_frames = max(1, n_frames - n_shower_frames)
        progress = (frame - n_shower_frames) / remaining_frames
        return t_end_shower + progress * (total_time - t_end_shower)

    def update(frame):
        t = time_for_frame(frame)
        subtitle.set_text(f"t = {t:5.2f}  |  shower -> hadronization -> detector cascade")

        xs, ys, sizes, colors = [], [], [], []
        for p in all_partons:
            if t < p.t_birth:
                continue
            t_show = min(t, p.t_death) if p.t_death else t
            x, y = p.pos_at(t_show)
            trail_lines[p.id].set_data([p.x0, x], [p.y0, y])
            if t < p.t_death:
                xs.append(x)
                ys.append(y)
                sizes.append(18 + 40 * (p.energy / E_TOP))
                colors.append(COLORS.get(p.ptype, "white"))

        for sc, mult in zip(glow_scatters, glow_layers):
            sc.set_offsets(np.column_stack([xs, ys]) if xs else np.empty((0, 2)))
            sc.set_sizes([s * mult for s in sizes])
            sc.set_color(colors)

        hx, hy, hs, hc = [], [], [], []
        for idx, cl in enumerate(all_hadron_particles):
            if t < cl["t0"]:
                continue
            dt = t - cl["t0"]
            r = cl["speed"] * dt
            x = cl["x0"] + r * math.cos(cl["angle"])
            y = cl["y0"] + r * math.sin(cl["angle"])
            dist = math.hypot(x, y)
            ang_deg = math.degrees(math.atan2(y, x))
            in_range = DETECTOR_THETA[0] <= ang_deg <= DETECTOR_THETA[1]

            stopped = False
            for li, (radius, color, name) in enumerate(DETECTOR_LAYERS):
                key = (idx, li)
                if in_range and dist >= radius and key not in crossed:
                    crossed[key] = True
                    hit_marks.append((x, y, color))
                    flashes.append({"x": x, "y": y, "color": color, "age": 0})
                if in_range and dist >= DETECTOR_LAYERS[-1][0]:
                    stopped = True

            if stopped:
                continue

            fade = max(0.0, 1.0 - dt / max(flight_time_window, 1.0))
            hx.append(x)
            hy.append(y)
            hs.append(cl["size"] * (0.4 + 0.6 * fade))
            hc.append((1, 1, 1, 0.85 * fade))

        hadron_sc.set_offsets(np.column_stack([hx, hy]) if hx else np.empty((0, 2)))
        hadron_sc.set_sizes(hs)
        hadron_sc.set_color(hc if hc else None)

        if hit_marks:
            hmx = [h[0] for h in hit_marks]
            hmy = [h[1] for h in hit_marks]
            hmc = [h[2] for h in hit_marks]
            hit_marks_sc.set_offsets(np.column_stack([hmx, hmy]))
            hit_marks_sc.set_sizes([30] * len(hit_marks))
            hit_marks_sc.set_color(hmc)

        for f in flashes:
            f["age"] += 1
        flashes[:] = [f for f in flashes if f["age"] <= FLASH_LIFETIME]
        if flashes:
            fx = [f["x"] for f in flashes]
            fy = [f["y"] for f in flashes]
            fc = [f["color"] for f in flashes]
            fsz = [120 * max(0.0, 1.0 - f["age"] / FLASH_LIFETIME) + 10 for f in flashes]
            flash_sc.set_offsets(np.column_stack([fx, fy]))
            flash_sc.set_sizes(fsz)
            flash_sc.set_color(fc)
        else:
            flash_sc.set_offsets(np.empty((0, 2)))

        return (list(trail_lines.values()) + glow_scatters +
                [hadron_sc, hit_marks_sc, flash_sc, subtitle])

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)

    try:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps, bitrate=3500)
        if not out_path.endswith(".mp4"):
            out_path += ".mp4"
        anim.save(out_path, writer=writer, dpi=150)
        print(f"Saved MP4 to {out_path}")
    except Exception as e:
        gif_path = out_path.rsplit(".", 1)[0] + ".gif"
        print(f"FFmpeg unavailable ({e}); falling back to GIF: {gif_path}")
        anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=110)
        print(f"Saved GIF to {gif_path}")

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Top quark shower + hadronization + detector hits vibe video")
    parser.add_argument("--out", default="top_quark_shower.mp4", help="output file path")
    parser.add_argument("--duration", type=float, default=14.0, help="video length in seconds")
    parser.add_argument("--fps", type=int, default=30, help="frames per second")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    args = parser.parse_args()

    run(out_path=args.out, duration=args.duration, fps=args.fps, seed=args.seed)
