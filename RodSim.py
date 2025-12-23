import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
from matplotlib.animation import FuncAnimation
import sounddevice as sd
import threading
import time

#parameters
L = 1.0
c = 1.0
epsilon = 0.1
FPS = 60
SPEED = 0.7
BASE_FREQ = 440
MAX_FREQ = 880
AUDIO_SR = 44100
AUDIO_BLOCK = 1024
SMOOTHING = 0.05
t = 0.0
running = False
audio_freq = BASE_FREQ
prev_freq = BASE_FREQ
audio_phase = 0.0
audio_volume = 0.3
audio_running = True

NX = 400
x = np.linspace(0, L, NX)

N_POINTS = 9
x0 = np.linspace(0, L, N_POINTS)
y0 = np.zeros_like(x0)

modes = []
cos_x = None
cos_x0 = None
omega = None

def rebuild_modes(n_modes):
    global modes, cos_x, cos_x0, omega
    modes = np.arange(1, 2 * n_modes, 2)
    omega = modes * np.pi * c / L
    cos_x = np.cos(np.outer(modes * np.pi / L, x))
    cos_x0 = np.cos(np.outer(modes * np.pi / L, x0))

rebuild_modes(1)

def bar_displacement(t):
    phase = np.cos(omega * t)
    u = np.sum((phase[:, None] / modes[:, None]**2) * cos_x, axis=0)
    return (4 * epsilon * L / np.pi**2) * u

def bar_displacement_points(t):
    phase = np.cos(omega * t)
    u = np.sum((phase[:, None] / modes[:, None]**2) * cos_x0, axis=0)
    return (4 * epsilon * L / np.pi**2) * u

#layout
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(3, 1, height_ratios=[3.5, 2.5, 1.2], hspace=0.4, bottom=0.28)
fig.subplots_adjust(top=0.93)


ax_bar  = fig.add_subplot(gs[0])
ax_time = fig.add_subplot(gs[1])
ax_long = fig.add_subplot(gs[2])

#graph1
line_bar, = ax_bar.plot(x, np.zeros_like(x), lw=2)
ax_bar.set_xlim(0, L)
ax_bar.set_ylim(-0.08, 0.08)
ax_bar.set_ylabel("y")
ax_bar.set_xlabel("x")
ax_bar.set_title("5.8")
ax_bar.grid(True)

#graph2
line_time, = ax_time.plot([], [], lw=2)
ax_time.set_ylim(-0.08, 0.08)
ax_time.set_ylabel("y")
ax_time.set_xlabel("t")
ax_time.grid(True)
ax_time.tick_params(axis='x', bottom=False, labelbottom=False)

#graph3
points_long, = ax_long.plot(x0, y0, "o-", lw=2)
ax_long.set_xlim(-0.1 * L, 1.1 * L)
ax_long.set_ylim(-0.05, 0.05)
ax_long.set_xlabel("L")
ax_long.set_yticks([])
ax_long.grid(True)

time_history = []
y_history = []
MAX_POINTS = 600

#audio
def audio_callback(outdata, frames, time_info, status):
    global audio_phase, prev_freq

    smooth = prev_freq + SMOOTHING * (audio_freq - prev_freq)
    prev_freq = smooth

    vol = audio_volume if running else 0.0
    phase_inc = 2 * np.pi * smooth / AUDIO_SR
    phase = audio_phase + phase_inc * np.arange(frames)
    audio_phase = (audio_phase + phase_inc * frames) % (2 * np.pi)

    outdata[:] = (vol * np.sin(phase)).reshape(-1, 1)

def audio_thread():
    with sd.OutputStream(
        samplerate=AUDIO_SR,
        channels=1,
        blocksize=AUDIO_BLOCK,
        callback=audio_callback
    ):
        while audio_running:
            time.sleep(0.05)

threading.Thread(target=audio_thread, daemon=True).start()

last_time = time.perf_counter()

def update(_):
    global t, audio_freq, last_time

    now = time.perf_counter()
    dt = now - last_time
    last_time = now

    if not running:
        return line_bar, line_time, points_long

    t += SPEED * dt

    u = bar_displacement(t)
    line_bar.set_ydata(u)

    y_left = u[0]
    time_history.append(t)
    y_history.append(y_left)

    if len(time_history) > MAX_POINTS:
        time_history.pop(0)
        y_history.pop(0)

    line_time.set_data(time_history, y_history)

    if len(time_history) > 1:
        ax_time.set_xlim(time_history[0], time_history[-1])

    x_move = x0 + bar_displacement_points(t)
    points_long.set_data(x_move, y0)

    energy = np.mean(np.abs(u))
    audio_freq = BASE_FREQ + np.clip(energy / 0.05, 0, 1) * (MAX_FREQ - BASE_FREQ)

    return line_bar, line_time, points_long

#animation
ani = FuncAnimation(
    fig,
    update,
    interval=1000 / FPS,
    blit=True,
    cache_frame_data=False
)

#ui
ax_start = plt.axes([0.25, 0.10, 0.15, 0.05])
ax_reset = plt.axes([0.45, 0.10, 0.15, 0.05])
ax_modes = plt.axes([0.65, 0.10, 0.15, 0.05])
ax_vol   = plt.axes([0.15, 0.05, 0.7, 0.03])

btn_start = Button(ax_start, "Start")
btn_reset = Button(ax_reset, "Reset")
slider_vol = Slider(ax_vol, "Volume", 0, 1, valinit=audio_volume)
box_modes = TextBox(ax_modes, "N", initial="1")

def start_clicked(event):
    global running, t
    t = 0.0
    rebuild_modes(int(box_modes.text))
    time_history.clear()
    y_history.clear()
    running = True

def reset_clicked(event):
    global running, t, audio_freq
    running = False
    t = 0.0
    audio_freq = BASE_FREQ
    line_bar.set_ydata(np.zeros_like(x))
    line_time.set_data([], [])
    points_long.set_data(x0, y0)

def volume_changed(val):
    global audio_volume
    audio_volume = val

btn_start.on_clicked(start_clicked)
btn_reset.on_clicked(reset_clicked)
slider_vol.on_changed(volume_changed)

plt.show()
audio_running = False