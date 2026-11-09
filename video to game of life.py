#!/usr/bin/env python3
"""

1. The window shows your webcam live, converted to big black-and-white pixels.
2. Click the "TAKE PHOTO & START" button: that pixel picture freezes and every
   white pixel becomes a living cell in Conway's Game of Life.
3. Watch the game unfold. Click "BACK TO CAMERA" to take another photo.

Install:  python -m pip install opencv-python numpy
Run:      python pixel_life.py

Keyboard: SPACE or ENTER = same as the button, Q or ESC = quit.
"""

# ---------------- settings ----------------
PIXEL_SIZE = 1        # size of one cell
THRESHOLD = 110        # 0-255: how bright a spot must be to become a white cell
INVERT = True         
SPEED = 10             # game generations per second
WINDOW_WIDTH = 1200     # width of the window on screen
MIRROR = True          # flip the camera like a mirror
# -----------------------------------------------------------------

import os
import sys
import time

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np

WIN = "Pixel Life"
BAR_HEIGHT = 70
BUTTON_W, BUTTON_H = 300, 46

clicked = False          
button_rect = (0, 0, 0, 0)


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
            for _ in range(30):                     
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


def game_of_life_step(alive):
    """One generation of Conway's Game of Life on a wrapping grid."""
    n = sum(np.roll(np.roll(alive, dy, 0), dx, 1)
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
    return ((n == 3) | (alive & (n == 2))).astype(np.uint8)


def pixelate(frame):
    """Shrink the frame to a small grid and make every cell black (0) or white (1)."""
    h, w = frame.shape[:2]
    cols = max(8, w // PIXEL_SIZE)
    rows = max(8, h // PIXEL_SIZE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
    cells = (small > THRESHOLD).astype(np.uint8)
    return 1 - cells if INVERT else cells


def on_mouse(event, x, y, flags, param):
    global clicked
    bx, by, bw, bh = button_rect
    if event == cv2.EVENT_LBUTTONDOWN and bx <= x <= bx + bw and by <= y <= by + bh:
        clicked = True


def show(cells, label):
    global button_rect
    scale = max(1, WINDOW_WIDTH // cells.shape[1])
    grid = cv2.resize((cells * 255).astype(np.uint8),
                      (cells.shape[1] * scale, cells.shape[0] * scale),
                      interpolation=cv2.INTER_NEAREST)
    grid = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)

    bar = np.full((BAR_HEIGHT, grid.shape[1], 3), 40, np.uint8)
    bx = (grid.shape[1] - BUTTON_W) // 2
    by = (BAR_HEIGHT - BUTTON_H) // 2
    cv2.rectangle(bar, (bx, by), (bx + BUTTON_W, by + BUTTON_H), (255, 255, 255), -1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(bar, label, (bx + (BUTTON_W - tw) // 2, by + (BUTTON_H + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    button_rect = (bx, grid.shape[0] + by, BUTTON_W, BUTTON_H)  

    cv2.imshow(WIN, np.vstack([grid, bar]))


def main():
    global clicked
    cap = open_camera()
    fails = 0

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)

    mode = "camera"            # "camera" = live preview, "life" = game running
    cells = None
    last_step = time.perf_counter()

    while True:
        if mode == "camera":
            ok, frame = cap.read()
            if ok and frame is not None:
                fails = 0
                if MIRROR:
                    frame = cv2.flip(frame, 1)
                cells = pixelate(frame)
            else:
                fails += 1
                if fails > 60:                      # camera stopped sending pictures: reconnect
                    cap.release()
                    cap = open_camera()
                    fails = 0
            if cells is None:
                cv2.waitKey(30)
                continue
        else:
            now = time.perf_counter()
            if now - last_step >= 1.0 / SPEED:
                cells = game_of_life_step(cells)
                last_step = now

        show(cells, "TAKE PHOTO & START" if mode == "camera" else "BACK TO CAMERA")

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if clicked or key in (ord(" "), 13, 10):        # space or Enter
            clicked = False
            if mode == "camera":
                mode = "life"
                time.sleep(2)
                last_step = time.perf_counter()
            else:
                mode = "camera"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()