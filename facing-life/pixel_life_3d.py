#!/usr/bin/env python3
"""
Pixel Life 3D

Based on the idea of https://github.com/pbelmans/conway : every generation of
the Game of Life is drawn as a layer of cubes, and the layers stack up on top
of each other, so time becomes the height of a 3D tower.

1. A window shows your webcam live, pixelated in black and white.
2. Press SPACE or ENTER: that frame becomes the bottom layer of the tower and
   the Game of Life grows upwards from it, one layer per generation.
   Drag with the mouse to rotate, scroll to zoom. SPACE pauses / resumes.
3. Close the 3D window (or press Q in it) to go back to the camera and take
   another frame. Press Q or ESC in the camera window to quit.

Install:  python -m pip install opencv-python numpy pyvista
Run:      python pixel_life_3d.py
"""

# ---------------- settings: change these numbers ----------------
PIXEL_SIZE = 2          # size of one cell in camera pixels (bigger = fewer, chunkier cells)
THRESHOLD = 110          # 0-255: how bright a spot must be to become a living cell
INVERT = False           # True = dark parts of the picture become the living cells
MIRROR = True            # flip the camera like a mirror

STEP_MS = 120            # milliseconds between generations (smaller = faster)
MAX_LAYERS = 150         # how many generations stay visible (old ones disappear at the bottom)
FOLLOW_CAMERA = True     # camera rises with the newest layer

CUBE_COLOR = "#ff2020"   # colour of the cubes
CUBE_OPACITY = 0.3       # 0-1: 1 = solid, lower = more see-through
EDGE_COLOR = "#eeeeee"   # colour of the cube outlines (set SHOW_EDGES = False to hide them)
SHOW_EDGES = True
BACKGROUND = "#000000"

PREVIEW_WIDTH = 1500      # width of the camera window on screen
# -----------------------------------------------------------------

import os
import sys
import time

# Makes the Windows camera driver start more reliably (must be set before importing cv2)
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np
import pyvista as pv

PREVIEW_WIN = "Pixel Life 3D - camera (SPACE / ENTER = start)"


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


def pixelate(frame):
    """Shrink the frame to a small grid and make every cell dead (0) or alive (1)."""
    h, w = frame.shape[:2]
    cols = max(8, w // PIXEL_SIZE)
    rows = max(8, h // PIXEL_SIZE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
    cells = (small > THRESHOLD).astype(np.uint8)
    return 1 - cells if INVERT else cells


def camera_preview(cap):
    """Live black & white pixel preview. Returns the frozen grid, or None to quit."""
    cv2.namedWindow(PREVIEW_WIN, cv2.WINDOW_AUTOSIZE)
    cells = None
    fails = 0
    while True:
        ok, frame = cap.read()
        if ok and frame is not None:
            fails = 0
            if MIRROR:
                frame = cv2.flip(frame, 1)
            cells = pixelate(frame)
        else:
            fails += 1
            if fails > 60:                          # camera stopped sending pictures: reconnect
                cap.release()
                cap = open_camera()
                fails = 0
        if cells is not None:
            scale = max(1, PREVIEW_WIDTH // cells.shape[1])
            big = cv2.resize((cells * 255).astype(np.uint8),
                             (cells.shape[1] * scale, cells.shape[0] * scale),
                             interpolation=cv2.INTER_NEAREST)
            cv2.imshow(PREVIEW_WIN, big)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(PREVIEW_WIN, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            return None, cap
        if key in (ord(" "), 13, 10) and cells is not None:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            return cells.copy(), cap


# ---------------- game of life ----------------
def game_of_life_step(alive):
    """One generation of Conway's Game of Life. Cells outside the grid count as dead."""
    p = np.pad(alive, 1)
    h, w = alive.shape
    n = sum(p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
    return ((n == 3) | (alive.astype(bool) & (n == 2))).astype(np.uint8)


# ---------------- 3D ----------------
CUBE = pv.Cube()   # 1 x 1 x 1


def layer_mesh(cells, z):
    """All living cells of one generation as cubes at height z, merged into one mesh."""
    rows, cols = np.nonzero(cells)
    if len(rows) == 0:
        return None
    h, w = cells.shape
    points = np.column_stack([cols - w / 2, h / 2 - rows, np.full(len(rows), z)]).astype(np.float32)
    return pv.PolyData(points).glyph(geom=CUBE, scale=False, orient=False)


def run_3d(start):
    h, w = start.shape
    plotter = pv.Plotter(window_size=(1200, 850), title="Pixel Life 3D")
    plotter.set_background(BACKGROUND)
    plotter.enable_anti_aliasing()
    if CUBE_OPACITY < 1:
        plotter.enable_depth_peeling(number_of_peels=4)   # draws overlapping see-through cubes correctly

    state = {"cells": start, "gen": 0, "paused": False, "layers": []}

    def add_layer():
        mesh = layer_mesh(state["cells"], state["gen"])
        actor = None
        if mesh is not None:
            actor = plotter.add_mesh(mesh, color=CUBE_COLOR, show_edges=SHOW_EDGES,
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
        state["cells"] = game_of_life_step(state["cells"])
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
                     position="lower_left", font_size=9, color="#444444")

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