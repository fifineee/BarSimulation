import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import sounddevice as sd
import threading
import time

#parameters
L = 1.0
c = 1.0
epsilon = 0.05
N_MODES = 20
FPS = 60
SPEED = 2.0
BASE_FREQ = 440
MIN_FREQ = 220
MAX_FREQ = 880
AUDIO_SR = 44100
AUDIO_BLOCK = 1024
SMOOTHING = 0.05
audio_freq = BASE_FREQ
prev_freq = BASE_FREQ
audio_volume = 0.3
audio_phase = 0.0
audio_running = True


x = np.linspace(0, L, 500)
modes = np.arange(1, 2 * N_MODES, 2)

def bar_displacement(x, t):
    u = np.zeros_like(x)
    for m in modes:
        omega = m * np.pi * c / L
        u += (1 / m**2) * np.cos(m * np.pi * x / L) * (np.cos(omega * t) - 1)
    return (4 * epsilon * L / np.pi**2) * u

fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(bottom=0.28)

line, = ax.plot(x, np.zeros_like(x), lw=2)
ax.set_xlim(0, L)
ax.set_ylim(-0.12, 0.12)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("5.8")
ax.grid(True)

t = 0.0
running = False

#audio
def audio_callback(outdata, frames, time_info, status):
    global audio_phase, prev_freq

    smooth_freq = prev_freq + SMOOTHING * (audio_freq - prev_freq)
    prev_freq = smooth_freq

    volume = audio_volume if running else 0.0

    phase_inc = 2 * np.pi * smooth_freq / AUDIO_SR
    phase_array = audio_phase + phase_inc * np.arange(frames)
    audio_phase = (audio_phase + phase_inc * frames) % (2 * np.pi)

    tone = volume * np.sin(phase_array)
    outdata[:] = tone.reshape(-1, 1)

def audio_thread():
    with sd.OutputStream(
        samplerate=AUDIO_SR,
        channels=1,
        callback=audio_callback,
        blocksize=AUDIO_BLOCK
    ):
        while audio_running:
            time.sleep(0.05)

threading.Thread(target=audio_thread, daemon=True).start()

#animation
def update_frame(dt):
    global t, audio_freq

    if running:
        t += SPEED * dt
        u = bar_displacement(x, t)
        line.set_ydata(u)
        energy = np.mean(np.abs(u))
        norm = np.clip(energy / 0.05, 0, 1)

        audio_freq = BASE_FREQ + norm * (MAX_FREQ - BASE_FREQ)

        fig.canvas.draw_idle()

def loop():
    last = time.time()
    while plt.fignum_exists(fig.number):
        now = time.time()
        dt = now - last
        last = now
        update_frame(dt)
        time.sleep(max(0, 1 / FPS - dt))

threading.Thread(target=loop, daemon=True).start()

#ui
ax_start = plt.axes([0.25, 0.10, 0.15, 0.08])
ax_reset = plt.axes([0.45, 0.10, 0.15, 0.08])
ax_vol   = plt.axes([0.15, 0.03, 0.7, 0.03])

btn_start = Button(ax_start, "Start")
btn_reset = Button(ax_reset, "Reset")
slider_vol = Slider(ax_vol, "Volume", 0.0, 1.0, valinit=audio_volume)

def start_clicked(event):
    global running, t
    t = 0.0
    running = True

def reset_clicked(event):
    global running, t, audio_freq
    running = False
    t = 0.0
    audio_freq = BASE_FREQ
    line.set_ydata(np.zeros_like(x))
    fig.canvas.draw_idle()

def volume_changed(val):
    global audio_volume
    audio_volume = val

btn_start.on_clicked(start_clicked)
btn_reset.on_clicked(reset_clicked)
slider_vol.on_changed(volume_changed)

#run
plt.show()
audio_running = False
