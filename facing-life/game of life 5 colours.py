#!/usr/bin/env python3
"""
Pixel Life - 5 colours

1. The window shows your webcam live, pixelated into 5 colours. The colours are
   picked automatically from what the camera sees: the 5 main colours of the
   picture (e.g. your shirt, skin, wall, hair...).
2. Press SPACE or ENTER (or click the button): the frame freezes and each colour
   becomes its own Game of Life. The 5 games run at the same time, drawn
   translucent on top of each other, so where they overlap the colours mix.
3. Press SPACE / ENTER again to go back to the camera.

Install:  python -m pip install opencv-python numpy
Run:      python pixel_life_colors.py
Q or ESC quits. Click the window once first so it receives your key presses.
"""

# ---------------- settings: change these numbers ----------------
PIXEL_SIZE = 1        # size of one cell in camera pixels (bigger = fewer, chunkier cells)
SPEED = 10             # game generations per second
OPACITY = 0.7          # 0-1: how see-through each colour layer is in the game
WINDOW_WIDTH = 2000     # width of the window on screen
MIRROR = True          # flip the camera like a mirror

NUM_COLORS = 5        # how many colours (= how many Games of Life)
COLOR_BOOST = 2.5     # saturation multiplier: 1.0 = camera colours as they are, higher = more colourful
MIN_SATURATION = 190  # 0-255: every colour is at least this saturated (255 = pure neon colours)
MIN_BRIGHTNESS = 220  # 0-255: every colour is at least this bright (so nothing is dull or dark)
# -----------------------------------------------------------------

import os
import sys
import time

# Makes the Windows camera driver start more reliably (must be set before importing cv2)
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np

WIN = "Pixel Life - 5 colours"
BAR_HEIGHT = 70
BUTTON_W, BUTTON_H = 300, 46
BACKGROUND = np.array([0, 0, 0], np.float32)

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


def game_of_life_step(alive):
    """One generation of Conway's Game of Life on a wrapping grid."""
    n = sum(np.roll(np.roll(alive, dy, 0), dx, 1)
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
    return ((n == 3) | (alive & (n == 2))).astype(np.uint8)


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


def render_camera(groups, palette):
    """Every cell painted fully in its colour."""
    return palette[groups]


def render_life(layers, palette):
    """Black background, then each colour layer painted translucent on top."""
    out = np.broadcast_to(BACKGROUND, layers[0].shape + (3,)).copy()
    for color, alive in zip(palette, layers):
        mask = alive.astype(bool)
        out[mask] = out[mask] * (1 - OPACITY) + color * OPACITY
    return out


def on_mouse(event, x, y, flags, param):
    global clicked
    bx, by, bw, bh = button_rect
    if event == cv2.EVENT_LBUTTONDOWN and bx <= x <= bx + bw and by <= y <= by + bh:
        clicked = True


def show(image, label):
    """Draw the grid big with crisp square pixels, plus a button bar underneath."""
    global button_rect
    h, w = image.shape[:2]
    scale = max(1, WINDOW_WIDTH // w)
    grid = cv2.resize(np.clip(image, 0, 255).astype(np.uint8), (w * scale, h * scale),
                      interpolation=cv2.INTER_NEAREST)

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

    mode = "camera"            # "camera" = live preview, "life" = 5 games running
    groups = None
    centers = None
    palette = None
    layers = None
    last_step = time.perf_counter()

    while True:
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
                if fails > 60:                      # camera stopped sending pictures: reconnect
                    cap.release()
                    cap = open_camera()
                    fails = 0
            if groups is None:
                cv2.waitKey(30)
                continue
            show(render_camera(groups, palette), "TAKE PHOTO & START")
        else:
            now = time.perf_counter()
            if now - last_step >= 1.0 / SPEED:
                layers = [game_of_life_step(layer) for layer in layers]
                last_step = now
            show(render_life(layers, life_palette), "BACK TO CAMERA")

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if clicked or key in (ord(" "), 13, 10):        # button, space or Enter
            clicked = False
            if mode == "camera":
                # one separate Game of Life per colour
                layers = [(groups == i).astype(np.uint8) for i in range(NUM_COLORS)]
                life_palette = palette.copy()          # keep the colours of the photo
                mode = "life"
                last_step = time.perf_counter()
            else:
                mode = "camera"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()