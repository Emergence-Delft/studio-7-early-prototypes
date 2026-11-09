#!/usr/bin/env python3
"""
Pixel Life - dye in milk

Inspired by "dye tablets in milk": saturated colours blooming, bleeding and
swirling through white milk, mixing like real dye where they meet.

1. The window shows your webcam as dye in milk: the picture is split into its
   5 main colours (picked from what the camera sees, made vivid).
2. Press SPACE or ENTER: the frame freezes and each colour becomes its own
   Game of Life. Wherever a colour's cells are alive they release dye into the
   milk; newborn cells push the liquid outward (like a drop of soap) and slow
   currents swirl everything together.
3. Press SPACE / ENTER again to go back to the camera.

Install:  python -m pip install opencv-python numpy
Run:      python pixel_life_milk.py
Q or ESC quits. Click the window once first so it receives your key presses.
"""

# ---------------- settings: change these numbers ----------------
PIXEL_SIZE = 15          # size of one game cell in camera pixels (bigger = fewer, bigger blooms)
SPEED = 8                # game generations per second
WINDOW_WIDTH = 1000      # width of the window on screen
MIRROR = True            # flip the camera like a mirror

NUM_COLORS = 5           # how many colours (= how many Games of Life)
COLOR_BOOST = 2.5        # saturation multiplier for the camera colours
MIN_SATURATION = 200     # 0-255: every dye is at least this saturated
MIN_BRIGHTNESS = 210     # 0-255: every dye is at least this bright

# the milk
MILK_COLOR = (250, 248, 243)  # (R, G, B) slightly warm white
DYE_STRENGTH = 1.6       # how intense the dye looks (higher = deeper, less milky)
BIRTH_DROP = 0.35        # dye dropped where a new cell is born (the blooms)
LIVE_RELEASE = 0.006     # dye a cell keeps releasing while it stays alive
FADE = 0.982             # 0-1: how long dye lingers (closer to 1 = stays longer)
BLEED = 0.35             # how fast dye spreads and softens (lower = finer marbled threads)
SWIRL = 1.0              # strength of the slow swirling currents
SOAP_PUSH = 14.0         # how hard newborn cells push the milk outward
DEEPEN = 0.15            # 0-1: how much thick dye darkens (0 = always bright like ink)
DETAIL = 5               # milk resolution per game cell (higher = smoother, slower)
# -----------------------------------------------------------------

import os
import sys
import time

# Makes the Windows camera driver start more reliably (must be set before importing cv2)
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np

WIN = "Pixel Life - dye in milk"
MILK = np.array(MILK_COLOR[::-1], np.float32) / 255.0      # BGR, 0-1


# ---------------- camera ----------------
def open_camera():
    """Try every camera number and driver until one actually delivers frames."""
    if sys.platform.startswith("win"):
        backends = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "Media Foundation"), (cv2.CAP_ANY, "default")]
    else:
        backends = [(cv2.CAP_ANY, "default")]
    for index in range(4):
        for backend, name in backends:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            for _ in range(30):                     # cameras often need a moment to warm up
                ok, frame = cap.read()
                if ok and frame is not None:
                    print(f"Using camera {index} via {name}")
                    return cap
                time.sleep(0.05)
            cap.release()
    raise SystemExit(
        "Could not get any picture from a webcam.\n"
        "- Close other apps that use the camera (Teams, Zoom, Discord, the browser, the Camera app).\n"
        "- Windows Settings > Privacy & security > Camera: turn on camera access for desktop apps.\n"
        "- Check that the webcam works in the Windows Camera app."
    )


# ---------------- colours from the camera ----------------
def find_colors(pixels, centers):
    """Group the pixels into NUM_COLORS colour groups (k-means), starting from last frame's colours."""
    if centers is None:
        order = np.argsort(pixels.sum(1))
        picks = order[np.linspace(0, len(order) - 1, NUM_COLORS).astype(int)]
        centers = pixels[picks].copy()
    for _ in range(4):
        dist = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        labels = dist.argmin(1)
        for i in range(NUM_COLORS):
            members = pixels[labels == i]
            centers[i] = members.mean(0) if len(members) else pixels[np.random.randint(len(pixels))]
    return labels, centers


def vivid(centers):
    """Bright, saturated versions of the camera colours; greys get a hue far from the others."""
    hsv = cv2.cvtColor(np.clip(centers, 0, 255).astype(np.uint8)[None], cv2.COLOR_BGR2HSV).astype(np.float32)[0]
    raw_sat = hsv[:, 1].copy()
    grey = raw_sat < 45
    taken = list(hsv[~grey, 0])
    for i in sorted(np.nonzero(grey)[0], key=lambda i: hsv[i, 2]):
        candidates = np.arange(0, 180, 6)
        if taken:
            gap = np.min(np.abs(((candidates[:, None] - np.array(taken)[None]) + 90) % 180 - 90), 1)
            hue = candidates[gap.argmax()]
        else:
            hue = 120 + 25 * i
        hsv[i, 0] = hue % 180
        taken.append(hsv[i, 0])
    hsv[:, 1] = np.clip(np.maximum(raw_sat * COLOR_BOOST, MIN_SATURATION), 0, 255)
    hsv[:, 2] = np.clip(np.maximum(hsv[:, 2] * 1.3, MIN_BRIGHTNESS), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8)[None], cv2.COLOR_HSV2BGR)[0].astype(np.float32)


def pixelate(frame, centers):
    """Shrink the frame to the game grid and give every cell a colour number."""
    h, w = frame.shape[:2]
    cols, rows = max(8, w // PIXEL_SIZE), max(8, h // PIXEL_SIZE)
    small = cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)
    labels, centers = find_colors(small.reshape(-1, 3).astype(np.float32), centers)
    return labels.reshape(rows, cols), centers


# ---------------- game of life ----------------
def game_of_life_step(alive):
    """One generation of Conway's Game of Life on a wrapping grid."""
    n = sum(np.roll(np.roll(alive, dy, 0), dx, 1)
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
    return ((n == 3) | (alive & (n == 2))).astype(np.uint8)


# ---------------- the milk ----------------
class Milk:
    """Dye concentrations floating in milk: they spread, fade, swirl and get pushed around."""

    def __init__(self, rows, cols):
        self.h, self.w = rows * DETAIL, cols * DETAIL
        self.dye = np.zeros((NUM_COLORS, self.h, self.w), np.float32)
        self.push = np.zeros((self.h, self.w), np.float32)
        ys, xs = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        self.xs, self.ys = xs, ys
        # a few slow waves make up the swirling current (a "stream function")
        rng = np.random.default_rng()
        size = max(self.w, self.h)
        self.waves = []
        for n in range(8):
            freq = rng.uniform(1.0, 3.0) if n < 4 else rng.uniform(4.0, 9.0)   # big currents + fine marbling
            angle = rng.uniform(0, 2 * np.pi)
            k = np.array([np.cos(angle), np.sin(angle)]) * freq * 2 * np.pi / size
            amp = rng.uniform(0.6, 1.0) / np.linalg.norm(k)                    # similar speed for every wave
            self.waves.append((k, amp, rng.uniform(0.1, 0.35), rng.uniform(0, 2 * np.pi)))
        self.t = 0.0

    def to_field(self, cells):
        """Game grid -> soft blob shapes at milk resolution."""
        f = cv2.resize(cells.astype(np.float32), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(f, (0, 0), DETAIL * 0.55)

    def show_still(self, groups):
        """Camera preview: the picture as dye resting in the milk."""
        for i in range(NUM_COLORS):
            target = self.to_field(groups == i) * 0.9
            self.dye[i] += (target - self.dye[i]) * 0.5     # ease towards the live picture
        self.push[:] = 0

    def release(self, layers, births):
        """New cells drop dye (blooms) and push the milk outward; living cells leak a little."""
        for i, alive in enumerate(layers):
            self.dye[i] += self.to_field(alive) * LIVE_RELEASE
            if births is not None:
                self.dye[i] += self.to_field(births[i]) * BIRTH_DROP
        if births is not None:
            b = cv2.GaussianBlur(self.to_field(np.any(births, axis=0)), (0, 0), DETAIL * 1.5)
            self.push = np.maximum(self.push * 0.6, b)

    def flow(self, dt):
        """Move, spread and fade the dye for one frame."""
        self.t += dt
        # swirling current: velocity = rotated gradient of a sum of slow sine waves
        vx = np.zeros((self.h, self.w), np.float32)
        vy = np.zeros((self.h, self.w), np.float32)
        for (kx, ky), amp, speed, phase in self.waves:
            c = amp * np.cos(kx * self.xs + ky * self.ys + speed * self.t + phase)
            vx += ky * c
            vy -= kx * c
        scale = SWIRL * max(self.w, self.h) * 0.05
        vx *= scale
        vy *= scale
        # soap push: flow away from where cells are being born
        gy, gx = np.gradient(self.push)
        vx -= gx * SOAP_PUSH * DETAIL * 10
        vy -= gy * SOAP_PUSH * DETAIL * 10
        map_x = self.xs - vx * dt
        map_y = self.ys - vy * dt
        for i in range(NUM_COLORS):
            moved = cv2.remap(self.dye[i], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            self.dye[i] = cv2.GaussianBlur(moved, (0, 0), BLEED) * FADE
        np.clip(self.dye, 0, 1.5, out=self.dye)

    def render(self, palette):
        """Mix the dyes like ink: overlapping colours blend, thin dye looks pastel, thick dye vivid."""
        total = self.dye.sum(0)
        colors = np.clip(palette / 255.0, 0, 1)                       # (NUM_COLORS, 3) BGR
        weight = self.dye ** 3                                          # the strongest dye in a spot wins (no mud)
        mix = np.tensordot(weight.transpose(1, 2, 0), colors, axes=1) / np.maximum(weight.sum(0), 1e-9)[..., None]
        cover = 1 - np.exp(-(total * DYE_STRENGTH) ** 1.6)              # 0 = pure milk, 1 = pure dye (crisp edges)
        # dye piles up at its edges like a soap ring: slightly deeper colour along the fronts
        gy, gx = np.gradient(total)
        rim = np.clip(np.sqrt(gx * gx + gy * gy) * DETAIL * 1.2, 0, 1)
        cover = np.clip(cover + rim * 0.25, 0, 1)[..., None]
        mix *= 1 - DEEPEN * np.clip(total - 0.6, 0, 1)[..., None] - 0.18 * rim[..., None]
        out = MILK * (1 - cover) + mix * cover
        # soft vignette, like the curved edge of a dish
        yy = (self.ys / self.h - 0.5) * 2
        xx = (self.xs / self.w - 0.5) * 2
        out *= (1 - 0.07 * (xx ** 2 + yy ** 2))[..., None]
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def show(img):
    h, w = img.shape[:2]
    out_h = int(WINDOW_WIDTH * h / w)
    cv2.imshow(WIN, cv2.resize(img, (WINDOW_WIDTH, out_h), interpolation=cv2.INTER_CUBIC))


def main():
    cap = open_camera()
    fails = 0
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    mode = "camera"
    groups = centers = palette = None
    milk = None
    layers = None
    last_step = last_frame = time.perf_counter()

    while True:
        now = time.perf_counter()
        dt = min(0.1, now - last_frame)
        last_frame = now

        if mode == "camera":
            ok, frame = cap.read()
            if ok and frame is not None:
                fails = 0
                if MIRROR:
                    frame = cv2.flip(frame, 1)
                groups, centers = pixelate(frame, centers)
                palette = vivid(centers)
            else:
                fails += 1
                if fails > 60:
                    cap.release()
                    cap = open_camera()
                    fails = 0
            if groups is None:
                cv2.waitKey(30)
                continue
            if milk is None or milk.dye.shape[1:] != (groups.shape[0] * DETAIL, groups.shape[1] * DETAIL):
                milk = Milk(*groups.shape)
            milk.show_still(groups)
            milk.flow(dt * 0.3)                     # gentle drift even in the preview
            show(milk.render(palette))
        else:
            births = None
            if now - last_step >= 1.0 / SPEED:
                new = [game_of_life_step(a) for a in layers]
                births = [(n == 1) & (a == 0) for n, a in zip(new, layers)]
                layers = new
                last_step = now
            milk.release(layers, births)
            milk.flow(dt)
            show(milk.render(life_palette))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord(" "), 13, 10):
            if mode == "camera":
                layers = [(groups == i).astype(np.uint8) for i in range(NUM_COLORS)]
                life_palette = palette.copy()
                mode = "life"
                last_step = time.perf_counter()
            else:
                mode = "camera"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
