#!/usr/bin/env python3
"""
Pixel Life 3D - 5 colours

Based on the idea of https://github.com/pbelmans/conway : every generation of
the Game of Life is drawn as a layer of cubes, and the layers stack up on top
of each other, so time becomes the height of a 3D tower.

1. A window shows your webcam live, pixelated into 5 colours picked from what
   the camera sees (made extra vivid).
2. Press SPACE or ENTER: that frame becomes the bottom layer of the tower.
   Each colour is its own Game of Life, and all 5 grow upwards at the same
   time as translucent cubes. Where two colours are alive in the same spot,
   their cubes sit inside each other.
   Drag with the mouse to rotate, scroll to zoom. SPACE pauses / resumes.
3. Close the 3D window (or press Q in it) to go back to the camera and take
   another frame. Press Q or ESC in the camera window to quit.

Install:  python -m pip install opencv-python numpy pyvista
Run:      python pixel_life_3d_colors.py
"""

# ---------------- settings: change these numbers ----------------
PIXEL_SIZE = 2          # size of one cell in camera pixels (bigger = fewer, chunkier cells)
MIRROR = True            # flip the camera like a mirror

STEP_MS = 120            # milliseconds between generations (smaller = faster)
MAX_LAYERS = 150         # how many generations stay visible (old ones disappear at the bottom)
FOLLOW_CAMERA = True     # camera rises with the newest layer

NUM_COLORS = 5           # how many colours (= how many Games of Life)
COLOR_BOOST = 2.5        # saturation multiplier: 1.0 = camera colours as they are, higher = more colourful
MIN_SATURATION = 190     # 0-255: every colour is at least this saturated (255 = pure neon)
MIN_BRIGHTNESS = 220     # 0-255: every colour is at least this bright

CUBE_OPACITY = 1       # 0-1: 1 = solid, lower = more see-through
NESTING = 0.08           # each colour's cubes are this much smaller, so overlapping colours nest
EDGE_COLOR = "#202020"   # colour of the cube outlines (set SHOW_EDGES = False to hide them)
SHOW_EDGES = True
BACKGROUND = "#101014"

PREVIEW_WIDTH = 960      # width of the camera window on screen
# -----------------------------------------------------------------

import os
import sys
import time

# Makes the Windows camera driver start more reliably (must be set before importing cv2)
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np
import pyvista as pv

PREVIEW_WIN = "Pixel Life 3D colours - camera (SPACE / ENTER = start)"


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


def find_colors(pixels, centers):
    """Group the pixels into NUM_COLORS colour groups (k-means).
    Starts from last frame's colours so they stay steady instead of flickering."""
    if centers is None:
        # first frame: spread the starting colours from dark to bright
        order = np.argsort(pixels.sum(1))
        picks = order[np.linspace(0, len(order) - 1, NUM_COLORS).astype(int)]
        centers = pixels[picks].copy()
    for _ in range(4):
        dist = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        labels = dist.argmin(1)
        for i in range(NUM_COLORS):
            members = pixels[labels == i]
            if len(members):
                centers[i] = members.mean(0)
            else:                                   # unused colour: jump to a random pixel
                centers[i] = pixels[np.random.randint(len(pixels))]
    return labels, centers


def vivid(centers):
    """Turn the camera's colours into bright, saturated versions of themselves.
    Greyish colours (white wall, black hair...) have no real hue, so they get
    a hue far away from the other colours, to keep everything colourful."""
    bgr = np.clip(centers, 0, 255).astype(np.uint8)[None]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)[0]   # hue 0-180
    raw_sat = hsv[:, 1].copy()
    grey = raw_sat < 45
    taken = list(hsv[~grey, 0])
    # order greys from dark to bright so the result is stable between frames
    for i in sorted(np.nonzero(grey)[0], key=lambda i: hsv[i, 2]):
        candidates = np.arange(0, 180, 6)
        if taken:
            gap = np.min(np.abs(((candidates[:, None] - np.array(taken)[None]) + 90) % 180 - 90), 1)
            hue = candidates[gap.argmax()]
        else:
            hue = 120 + 25 * i          # nothing to avoid: start from blue/purple
        hsv[i, 0] = hue % 180
        taken.append(hsv[i, 0])
    hsv[:, 1] = np.clip(np.maximum(raw_sat * COLOR_BOOST, MIN_SATURATION), 0, 255)
    hsv[:, 2] = np.clip(np.maximum(hsv[:, 2] * 1.3, MIN_BRIGHTNESS), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8)[None], cv2.COLOR_HSV2BGR)[0].astype(np.float32)


def pixelate(frame, centers):
    """Shrink the frame to a grid and give every cell a colour number 0..NUM_COLORS-1."""
    h, w = frame.shape[:2]
    cols = max(8, w // PIXEL_SIZE)
    rows = max(8, h // PIXEL_SIZE)
    small = cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)
    labels, centers = find_colors(small.reshape(-1, 3).astype(np.float32), centers)
    return labels.reshape(rows, cols), centers


def camera_preview(cap):
    """Live 5-colour pixel preview. Returns (colour grid, palette), or None to quit."""
    cv2.namedWindow(PREVIEW_WIN, cv2.WINDOW_AUTOSIZE)
    groups = centers = palette = None
    fails = 0
    while True:
        ok, frame = cap.read()
        if ok and frame is not None:
            fails = 0
            if MIRROR:
                frame = cv2.flip(frame, 1)
            groups, centers = pixelate(frame, centers)
            palette = vivid(centers)
        else:
            fails += 1
            if fails > 60:                          # camera stopped sending pictures: reconnect
                cap.release()
                cap = open_camera()
                fails = 0
        if groups is not None:
            img = palette[groups].astype(np.uint8)
            scale = max(1, PREVIEW_WIDTH // groups.shape[1])
            big = cv2.resize(img, (groups.shape[1] * scale, groups.shape[0] * scale),
                             interpolation=cv2.INTER_NEAREST)
            cv2.imshow(PREVIEW_WIN, big)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(PREVIEW_WIN, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            return None, cap
        if key in (ord(" "), 13, 10) and groups is not None:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            return (groups.copy(), palette.copy()), cap


# ---------------- game of life ----------------
def game_of_life_step(alive):
    """One generation of Conway's Game of Life. Cells outside the grid count as dead."""
    p = np.pad(alive, 1)
    h, w = alive.shape
    n = sum(p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
    return ((n == 3) | (alive.astype(bool) & (n == 2))).astype(np.uint8)


# ---------------- 3D ----------------
# one cube size per colour, slightly smaller each time, so colours in the same spot nest
CUBES = [pv.Cube(x_length=1 - NESTING * i, y_length=1 - NESTING * i, z_length=1 - NESTING * i)
         for i in range(NUM_COLORS)]


def layer_mesh(layers, palette, z):
    """All living cells of all colours of one generation, merged into one coloured mesh."""
    parts = []
    for i, cells in enumerate(layers):
        rows, cols = np.nonzero(cells)
        if len(rows) == 0:
            continue
        h, w = cells.shape
        points = np.column_stack([cols - w / 2, h / 2 - rows, np.full(len(rows), z)]).astype(np.float32)
        cubes = pv.PolyData(points).glyph(geom=CUBES[i], scale=False, orient=False)
        b, g, r = palette[i]
        cubes.cell_data["rgb"] = np.tile(np.array([r, g, b], np.uint8), (cubes.n_cells, 1))
        parts.append(cubes)
    if not parts:
        return None
    return pv.merge(parts) if len(parts) > 1 else parts[0]


def run_3d(start):
    groups, palette = start
    h, w = groups.shape
    plotter = pv.Plotter(window_size=(1200, 850), title="Pixel Life 3D - 5 colours")
    plotter.set_background(BACKGROUND)
    plotter.enable_anti_aliasing()
    if CUBE_OPACITY < 1:
        plotter.enable_depth_peeling(number_of_peels=4)   # draws overlapping see-through cubes correctly

    layers = [(groups == i).astype(np.uint8) for i in range(NUM_COLORS)]   # one game per colour
    state = {"layers_alive": layers, "gen": 0, "paused": False, "layers": []}

    def add_layer():
        mesh = layer_mesh(state["layers_alive"], palette, state["gen"])
        actor = None
        if mesh is not None:
            actor = plotter.add_mesh(mesh, scalars="rgb", rgb=True, show_edges=SHOW_EDGES,
                                     edge_color=EDGE_COLOR, line_width=1, smooth_shading=False, opacity=CUBE_OPACITY,
                                     reset_camera=False)
        state["layers"].append(actor)
        if len(state["layers"]) > MAX_LAYERS:       # drop the oldest layer
            old = state["layers"].pop(0)
            if old is not None:
                plotter.remove_actor(old, render=False)

    add_layer()   # generation 0 = the photo

    # starting view: from the side and a bit above, like looking at a growing tower
    size = max(w, h)
    plotter.camera_position = [(0.9 * size, -1.4 * size, 0.9 * size), (0, 0, 0), (0, 0, 1)]

    def tick():
        state["layers_alive"] = [game_of_life_step(a) for a in state["layers_alive"]]
        state["gen"] += 1
        add_layer()
        if FOLLOW_CAMERA:
            # move the camera up by one layer, keeping whatever angle/zoom the user chose
            cam = plotter.camera
            cam.position = np.array(cam.position) + (0, 0, 1)
            cam.focal_point = np.array(cam.focal_point) + (0, 0, 1)
        plotter.renderer.reset_camera_clipping_range()   # keep the whole tower in view

    def toggle_pause():
        state["paused"] = not state["paused"]

    def stop():
        state["quit"] = True

    plotter.add_key_event("space", toggle_pause)
    plotter.add_key_event("q", stop)
    plotter.add_text("drag = rotate   scroll = zoom   SPACE = pause   Q / close = back to camera",
                     position="lower_left", font_size=9, color="#aaaaaa")

    # Open the window without blocking, then run our own loop:
    # step the game every STEP_MS and keep the window responsive in between.
    plotter.show(interactive_update=True, auto_close=False)
    last_step = time.perf_counter()
    try:
        while not state.get("quit"):
            if plotter.render_window is None or getattr(plotter, "_closed", False):
                break                                   # window was closed with the X
            now = time.perf_counter()
            if not state["paused"] and (now - last_step) * 1000 >= STEP_MS:
                tick()
                last_step = now
                plotter.render()
            plotter.update(stime=1)                     # handle mouse / keyboard, redraw
            time.sleep(0.005)
    except Exception:
        pass                                            # window closed while drawing
    try:
        plotter.close()
    except Exception:
        pass


def main():
    cap = open_camera()
    while True:
        cells, cap = camera_preview(cap)
        if cells is None:
            break
        run_3d(cells)
    cap.release()


if __name__ == "__main__":
    main()
