#!/usr/bin/env python3
"""
Pixel Life - glow

Soft grainy colour glows on warm paper, black dots and thin wandering lines.

1. The window shows your webcam in this style: the picture is split into its
   5 main colours (picked from what the camera sees). Every patch of one
   colour is a soft grainy glow with a black dot in the middle.
2. Press SPACE or ENTER: the frame freezes and each colour becomes its own
   Game of Life. Every cluster of living cells is a dot (bigger cluster =
   bigger dot) with a glow around it. When a cluster moves, its dot travels
   and draws a thin line behind it; clusters that wobble in place scribble
   little loops.
3. Press SPACE / ENTER again to go back to the camera.

Install:  python -m pip install opencv-python numpy
Run:      python pixel_life_glow.py
Q or ESC quits. Click the window once first so it receives your key presses.
"""

# ---------------- settings: change these numbers ----------------
PIXEL_SIZE = 5          # size of one game cell in camera pixels (bigger = fewer cells)
SPEED = 8                # game generations per second
WINDOW_WIDTH = 1000      # width of the window on screen
MIRROR = True            # flip the camera like a mirror

NUM_COLORS = 5           # how many colours (= how many Games of Life)
COLOR_BOOST = 2.0        # saturation multiplier for the camera colours
MIN_SATURATION = 170     # 0-255: every colour is at least this saturated
MIN_BRIGHTNESS = 215     # 0-255: every colour is at least this bright

PAPER_COLOR = (230, 227, 218)  # (R, G, B) warm off-white background
GLOW_SIZE = 1.6          # how soft the glows are (in cells)
HALO_SIZE = 2.2          # size of the round glow around each dot, relative to the dot
MIN_HALO = 1.8           # smallest glow around a dot (in cells), so tiny dots still glow
SHAPE_GLOW = 0.22        # 0-1: how much the glow also follows the exact shape of the cells
GLOW_OPACITY = 0.8       # 0-1: how strong the glows are
GLOW_LINGER = 0.93       # 0-1: how long a glow stays after cells move on (makes smears)
GRAIN = 0.55             # 0-1: how grainy / speckled the glows are

DOT_COLOR = (20, 20, 22) # (R, G, B) colour of the dots
DOT_SIZE = 1.1           # dot radius per cluster size (bigger = bigger dots)
MAX_DOT = 46             # largest dot radius in screen pixels
TRAIL_FADE = 0.996       # 0-1: how long lines stay visible (closer to 1 = longer)
TRAIL_OPACITY = 0.7      # 0-1: how dark the lines are
TRAIL_DARKNESS = 0.55    # 0 = black lines, 1 = lines in the full colour of their cluster
SMOOTHING = 0.07         # 0-1: how smoothly dots glide (lower = smoother, curvier lines)
# -----------------------------------------------------------------

import os
import sys
import time

# Makes the Windows camera driver start more reliably (must be set before importing cv2)
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np

WIN = "Pixel Life - glow"
PAPER = np.array(PAPER_COLOR[::-1], np.float32) / 255.0     # BGR 0-1
DOT_BGR = tuple(int(v) for v in DOT_COLOR[::-1])
MATCH_DIST = 3.0         # how far (in cells) a cluster may move and still be "the same" dot
GLOW_RES = 4             # glow resolution per cell


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
    """One generation of Conway's Game of Life. Cells outside the grid count as dead."""
    p = np.pad(alive, 1)
    h, w = alive.shape
    n = sum(p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
    return ((n == 3) | (alive.astype(bool) & (n == 2))).astype(np.uint8)


def clusters(alive):
    """Groups of touching living cells: list of (centre_row, centre_col, number_of_cells)."""
    n, _, stats, cents = cv2.connectedComponentsWithStats(alive.astype(np.uint8), connectivity=8)
    return [(cents[i][1], cents[i][0], stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]


# ---------------- drawing ----------------
class Canvas:
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.W = WINDOW_WIDTH
        self.H = int(WINDOW_WIDTH * rows / cols)
        self.sx, self.sy = self.W / cols, self.H / rows
        self.glow = np.zeros((NUM_COLORS, rows * GLOW_RES, cols * GLOW_RES), np.float32)
        self.trail_a = np.zeros((self.H, self.W), np.float32)
        self.trail_c = np.zeros((self.H, self.W, 3), np.float32)
        rng = np.random.default_rng()
        self.grains = [np.clip(1 + GRAIN * 2 * (rng.random((self.H, self.W), np.float32) - 0.5), 0, 2)
                       for _ in range(4)]
        self.frame_no = 0
        self.dots = [[] for _ in range(NUM_COLORS)]   # per colour: list of [x, y, tx, ty, r, tr]

    def to_screen(self, row, col):
        return (col + 0.5) * self.sx, (row + 0.5) * self.sy

    def dot_radius(self, cells):
        return min(MAX_DOT, 1.5 + DOT_SIZE * np.sqrt(cells) * self.sx / 10)

    # --- glows ---
    def update_glow(self, layers, linger):
        """Glow = a round halo around every dot + a faint glow in the shape of the cells."""
        gw, gh = self.cols * GLOW_RES, self.rows * GLOW_RES
        k = GLOW_RES / self.sx                      # screen pixels -> glow pixels
        for i, alive in enumerate(layers):
            halo = np.zeros((gh, gw), np.float32)
            for d in self.dots[i]:
                radius = max(d[4] * HALO_SIZE, MIN_HALO * self.sx) * k
                cv2.circle(halo, (int(d[0] * k), int(d[1] * k)), max(1, int(radius)), 1.0, -1)
            shape = cv2.resize(alive.astype(np.float32), (gw, gh), interpolation=cv2.INTER_LINEAR)
            f = cv2.GaussianBlur(halo + shape * SHAPE_GLOW, (0, 0), GLOW_SIZE * GLOW_RES)
            self.glow[i] = np.maximum(self.glow[i] * linger, np.clip(f, 0, 1))

    # --- dots and trails ---
    def set_dots(self, layers):
        """Camera preview: dots jump straight to the clusters, no lines."""
        for i, alive in enumerate(layers):
            self.dots[i] = []
            for row, col, cells in clusters(alive):
                x, y = self.to_screen(row, col)
                r = self.dot_radius(cells)
                self.dots[i].append([x, y, x, y, r, r])

    def track_dots(self, layers):
        """After each generation: give every dot its cluster's new position (or create / remove dots)."""
        max_d = MATCH_DIST * self.sx
        for i, alive in enumerate(layers):
            old = self.dots[i]
            used = [False] * len(old)
            new = []
            for row, col, cells in clusters(alive):
                tx, ty = self.to_screen(row, col)
                tr = self.dot_radius(cells)
                best, best_d = -1, max_d
                for j, d in enumerate(old):
                    if not used[j]:
                        dist = np.hypot(d[2] - tx, d[3] - ty)
                        if dist < best_d:
                            best, best_d = j, dist
                if best >= 0:
                    used[best] = True
                    d = old[best]
                    new.append([d[0], d[1], tx, ty, d[4], tr])
                else:
                    new.append([tx, ty, tx, ty, tr * 0.3, tr])   # new dot grows in
            self.dots[i] = new

    def move_dots(self, palette, draw_trails):
        """Glide every dot towards its target, drawing a thin line where it went."""
        scratch_a = np.zeros((self.H, self.W), np.uint8)
        scratch_c = np.zeros((self.H, self.W, 3), np.uint8)
        for i in range(NUM_COLORS):
            line_col = tuple(int(v) for v in palette[i] * TRAIL_DARKNESS)
            for d in self.dots[i]:
                nx = d[0] + (d[2] - d[0]) * SMOOTHING
                ny = d[1] + (d[3] - d[1]) * SMOOTHING
                if draw_trails and np.hypot(nx - d[0], ny - d[1]) > 0.2:
                    p0 = (int(d[0] * 16), int(d[1] * 16))
                    p1 = (int(nx * 16), int(ny * 16))
                    cv2.line(scratch_a, p0, p1, 255, 1, cv2.LINE_AA, shift=4)
                    cv2.line(scratch_c, p0, p1, line_col, 1, cv2.LINE_AA, shift=4)
                d[0], d[1] = nx, ny
                d[4] += (d[5] - d[4]) * 0.2
        self.trail_a *= TRAIL_FADE
        if draw_trails:
            a = scratch_a.astype(np.float32) / 255.0
            hit = a > 0
            self.trail_c[hit] = scratch_c[hit].astype(np.float32) / 255.0
            np.maximum(self.trail_a, a, out=self.trail_a)

    def clear_trails(self):
        self.trail_a[:] = 0

    # --- final picture ---
    def render(self, palette):
        # glows, composited at low resolution
        A = np.zeros(self.glow.shape[1:], np.float32)
        C = np.zeros(self.glow.shape[1:] + (3,), np.float32)
        for i in range(NUM_COLORS):
            a = np.clip(self.glow[i] * GLOW_OPACITY, 0, 1)
            C = C * (1 - a[..., None]) + (palette[i] / 255.0) * a[..., None]
            A = A * (1 - a) + a
        A = cv2.resize(A, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        C = cv2.resize(C, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        # grain: speckle the glows like airbrush / risograph texture
        grain = self.grains[self.frame_no % len(self.grains)]
        self.frame_no += 1
        a = np.clip(A * grain, 0, 1)[..., None]
        out = PAPER * (1 - a) + C * grain[..., None]          # C is already weighted by A
        # thin lines
        ta = (self.trail_a * TRAIL_OPACITY)[..., None]
        out = out * (1 - ta) + self.trail_c * ta
        img = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        # black dots on top
        for i in range(NUM_COLORS):
            for d in self.dots[i]:
                cv2.circle(img, (int(d[0] * 16), int(d[1] * 16)), max(1, int(d[4] * 16)),
                           DOT_BGR, -1, cv2.LINE_AA, shift=4)
        return img


def main():
    cap = open_camera()
    fails = 0
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    mode = "camera"
    groups = centers = palette = life_palette = None
    canvas = None
    layers = None
    last_step = time.perf_counter()

    while True:
        now = time.perf_counter()
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
            if canvas is None or (canvas.rows, canvas.cols) != groups.shape:
                canvas = Canvas(*groups.shape)
            live = [(groups == i).astype(np.uint8) for i in range(NUM_COLORS)]
            canvas.set_dots(live)
            canvas.update_glow(live, 0.0)
            canvas.clear_trails()
            cv2.imshow(WIN, canvas.render(palette))
        else:
            if now - last_step >= 1.0 / SPEED:
                layers = [game_of_life_step(a) for a in layers]
                canvas.track_dots(layers)
                last_step = now
            canvas.move_dots(life_palette, draw_trails=True)
            canvas.update_glow(layers, GLOW_LINGER)
            cv2.imshow(WIN, canvas.render(life_palette))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord(" "), 13, 10):
            if mode == "camera":
                layers = [(groups == i).astype(np.uint8) for i in range(NUM_COLORS)]
                life_palette = palette.copy()
                canvas.clear_trails()
                mode = "life"
                last_step = time.perf_counter()
            else:
                mode = "camera"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
