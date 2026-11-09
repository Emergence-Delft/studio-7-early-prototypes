#!/usr/bin/env python3
"""
Pixel Life - network

Soft grainy glows on warm paper, with black nodes joined by thin lines into a
growing network. Every line always runs between two nodes: nothing trails off.

1. The window shows your webcam in this style: the picture is split into its
   5 main colours (picked from what the camera sees). Every patch of one
   colour is a glowing node.
2. Press SPACE or ENTER: the frame freezes and each colour becomes its own
   Game of Life, and the network starts growing:
   - every new cluster of living cells sprouts a new node, connected by a line
     that grows out of the nearest existing node
   - a cluster that travels (like a glider) lays down a chain of nodes
   - old nodes eventually fade away together with all their lines
3. Press SPACE / ENTER again to go back to the camera.

Install:  python -m pip install opencv-python numpy
Run:      python pixel_life_network.py
Q or ESC quits. Click the window once first so it receives your key presses.
"""

# ---------------- settings: change these numbers ----------------
PIXEL_SIZE = 5          # size of one game cell in camera pixels (bigger = fewer cells)
SPEED = 6                # game generations per second
WINDOW_WIDTH = 1000      # width of the window on screen
MIRROR = False            # flip the camera like a mirror

NUM_COLORS = 5           # how many colours (= how many Games of Life)
COLOR_BOOST = 2.0        # saturation multiplier for the camera colours
MIN_SATURATION = 170     # 0-255: every colour is at least this saturated
MIN_BRIGHTNESS = 215     # 0-255: every colour is at least this bright

PAPER_COLOR = (230, 227, 218)  # (R, G, B) warm off-white background
GLOW_SIZE = 1.6          # how soft the glows are (in cells)
HALO_SIZE = 2.6          # size of the glow around each node, relative to the node
MIN_HALO = 1.6           # smallest glow around a node (in cells)
GLOW_OPACITY = 0.65      # 0-1: how strong the glows are
GRAIN = 0.55             # 0-1: how grainy / speckled the glows are

DOT_COLOR = (20, 20, 22) # (R, G, B) colour of the nodes
DOT_SIZE = 0.55          # node radius per cluster size
MAX_DOT = 26             # largest node radius in screen pixels
MIN_DOT = 2.2            # smallest node radius in screen pixels

# the network
LINK_DIST = 8            # a new node connects to existing nodes up to this far away (in cells)
EXTRA_LINK = 0.35        # 0-1: chance a new node also links to a second neighbour (more web-like)
CHAIN_SPACING = 4.0      # a travelling cluster drops a new node every this many cells
GROW_TIME = 0.6          # seconds a new line takes to grow out to its node
CURVE = 0.18             # how curved the lines are (0 = straight)
NODE_LIFE = 10           # seconds a node stays after its cluster is gone
FADE_TIME = 3            # seconds a node (and its lines) takes to fade out
MAX_NODES = 220          # when there are more nodes than this, the oldest fade out early
SETTLE = 2               # a new cluster must survive this many generations before it gets a node
LINE_OPACITY = 0.75      # 0-1: how dark the lines are
LINE_DARKNESS = 0.5      # 0 = black lines, 1 = lines in the full colour of their node
# -----------------------------------------------------------------


import os
import sys
import time

# Makes the Windows camera driver start more reliably (must be set before importing cv2)
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np

WIN = "Pixel Life - network"
PAPER = np.array(PAPER_COLOR[::-1], np.float32) / 255.0     # BGR 0-1
DOT_BGR = tuple(int(v) for v in DOT_COLOR[::-1])
MATCH_DIST = 3.0         # how far (in cells) a cluster may move and still be "the same" cluster
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



# ---------------- the network ----------------
class Network:
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.W = WINDOW_WIDTH
        self.H = int(WINDOW_WIDTH * rows / cols)
        self.sx, self.sy = self.W / cols, self.H / rows
        rng = np.random.default_rng()
        self.rng = rng
        self.grains = [np.clip(1 + GRAIN * 2 * (rng.random((self.H, self.W), np.float32) - 0.5), 0, 2)
                       for _ in range(4)]
        self.frame_no = 0
        self.reset()

    def reset(self):
        self.nodes = {}       # id -> dict(x, y, r, tr, color, seen, born, alive_head, fade_start)
        self.edges = []       # list of dict(a, b, born, bend)
        self.heads = []       # clusters being followed: dict(color, row, col, node)
        self.next_id = 0

    def to_screen(self, row, col):
        return (col + 0.5) * self.sx, (row + 0.5) * self.sy

    def dot_radius(self, cells):
        return float(np.clip(1.5 + DOT_SIZE * np.sqrt(cells) * self.sx / 10, MIN_DOT, MAX_DOT))

    # --- building the network ---
    def add_node(self, x, y, r, color, now):
        nid = self.next_id
        self.next_id += 1
        self.nodes[nid] = dict(x=x, y=y, r=r * 0.3, tr=r, color=color, seen=now, born=now, fade_start=None)
        return nid

    def add_edge(self, a, b, now):
        self.edges.append(dict(a=a, b=b, born=now, bend=self.rng.uniform(-CURVE, CURVE)))

    def nearest(self, x, y, exclude, count):
        """Up to `count` nearest visible nodes within LINK_DIST."""
        best = []
        max_d = LINK_DIST * self.sx
        for nid, n in self.nodes.items():
            if nid in exclude or n["fade_start"] is not None:
                continue
            d = np.hypot(n["x"] - x, n["y"] - y)
            if d < max_d:
                best.append((d, nid))
        best.sort()
        return [nid for _, nid in best[:count]]

    def sprout(self, x, y, r, color, now):
        """New node, connected by growing lines to its nearest neighbour(s)."""
        links = self.nearest(x, y, set(), 2)
        nid = self.add_node(x, y, r, color, now)
        if links:
            self.add_edge(links[0], nid, now)
            if len(links) > 1 and self.rng.random() < EXTRA_LINK:
                self.add_edge(links[1], nid, now)
        return nid

    def start(self, layers, now):
        """Frozen frame: one node per cluster, each joined to its nearest neighbour."""
        self.reset()
        for c, alive in enumerate(layers):
            for row, col, cells in clusters(alive):
                x, y = self.to_screen(row, col)
                nid = self.add_node(x, y, self.dot_radius(cells), c, now)
                self.nodes[nid]["r"] = self.nodes[nid]["tr"]
                self.heads.append(dict(color=c, row=row, col=col, node=nid, age=SETTLE))
        ids = list(self.nodes)
        for nid in ids:
            n = self.nodes[nid]
            near = self.nearest(n["x"], n["y"], {nid}, 1)
            if near:
                self.edges.append(dict(a=near[0], b=nid, born=now - GROW_TIME, bend=self.rng.uniform(-CURVE, CURVE)))

    def generation(self, layers, now):
        """After each Game of Life step: follow clusters, sprout nodes, extend chains."""
        new_heads = []
        for c, alive in enumerate(layers):
            old = [h for h in self.heads if h["color"] == c]
            used = [False] * len(old)
            for row, col, cells in clusters(alive):
                r = self.dot_radius(cells)
                best, best_d = -1, MATCH_DIST
                for j, h in enumerate(old):
                    if not used[j]:
                        d = np.hypot(h["row"] - row, h["col"] - col)
                        if d < best_d:
                            best, best_d = j, d
                x, y = self.to_screen(row, col)
                if best >= 0:
                    used[best] = True
                    h = old[best]
                    age = h["age"] + 1
                    node = self.nodes.get(h["node"]) if h["node"] is not None else None
                    if node is None:
                        nid = self.sprout(x, y, r, c, now) if age >= SETTLE else None
                    elif np.hypot(node["x"] - x, node["y"] - y) > CHAIN_SPACING * self.sx:
                        nid = self.add_node(x, y, r, c, now)      # travelling: extend the chain
                        self.add_edge(h["node"], nid, now)
                    else:
                        nid = h["node"]
                        node["tr"] = r
                        node["seen"] = now
                    new_heads.append(dict(color=c, row=row, col=col, node=nid, age=age))
                else:
                    new_heads.append(dict(color=c, row=row, col=col, node=None, age=1))
        self.heads = new_heads
        self.retire(now)

    def retire(self, now):
        """Start fading old nodes; remove fully faded nodes together with all their lines."""
        active = {h["node"] for h in self.heads if h["node"] is not None}
        for nid, n in self.nodes.items():
            if nid in active:
                n["seen"] = now
            elif n["fade_start"] is None and now - n["seen"] > NODE_LIFE:
                n["fade_start"] = now
        visible = [nid for nid, n in self.nodes.items() if n["fade_start"] is None]
        if len(visible) > MAX_NODES:
            for nid in sorted(visible, key=lambda i: self.nodes[i]["seen"])[:len(visible) - MAX_NODES]:
                self.nodes[nid]["fade_start"] = now
        gone = {nid for nid, n in self.nodes.items()
                if n["fade_start"] is not None and now - n["fade_start"] > FADE_TIME}
        for nid in gone:
            del self.nodes[nid]
        self.edges = [e for e in self.edges if e["a"] in self.nodes and e["b"] in self.nodes]
        self.heads = [h for h in self.heads if h["node"] is None or h["node"] in self.nodes]

    # --- drawing ---
    def alpha(self, n, now):
        if n["fade_start"] is None:
            return 1.0
        return max(0.0, 1 - (now - n["fade_start"]) / FADE_TIME)

    def curve_points(self, a, b, bend, upto):
        """Points along a gently bent line from a to b, from 0 up to `upto` (0-1)."""
        ax, ay, bx, by = a["x"], a["y"], b["x"], b["y"]
        mx, my = (ax + bx) / 2 - (by - ay) * bend, (ay + by) / 2 + (bx - ax) * bend
        t = np.linspace(0, upto, 14)[:, None]
        p = (1 - t) ** 2 * [ax, ay] + 2 * (1 - t) * t * [mx, my] + t ** 2 * [bx, by]
        return p

    def render(self, palette, now):
        # where every node is drawn: a node still being reached by its line sits at the line's tip
        draw_pos = {nid: (n["x"], n["y"]) for nid, n in self.nodes.items()}
        hidden = set()
        line_a = np.zeros((self.H, self.W), np.uint8)
        line_c = np.zeros((self.H, self.W, 3), np.uint8)
        for e in self.edges:
            a, b = self.nodes[e["a"]], self.nodes[e["b"]]
            grow = min(1.0, (now - e["born"]) / GROW_TIME)
            if grow <= 0:
                continue
            pts = self.curve_points(a, b, e["bend"], grow)
            if grow < 1 and b["born"] >= e["born"] - 1e-6:
                draw_pos[e["b"]] = tuple(pts[-1])          # the new node rides on the growing tip
            alpha = min(self.alpha(a, now), self.alpha(b, now))
            col = tuple(int(v) for v in palette[b["color"]] * LINE_DARKNESS)
            poly = (pts * 16).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(line_a, [poly], False, int(255 * alpha), 1, cv2.LINE_AA, shift=4)
            cv2.polylines(line_c, [poly], False, col, 1, cv2.LINE_AA, shift=4)

        # glows around nodes
        gw, gh = self.cols * GLOW_RES, self.rows * GLOW_RES
        k = GLOW_RES / self.sx
        A = np.zeros((gh, gw), np.float32)
        C = np.zeros((gh, gw, 3), np.float32)
        halos = [np.zeros((gh, gw), np.float32) for _ in range(NUM_COLORS)]
        for nid, n in self.nodes.items():
            n["r"] += (n["tr"] - n["r"]) * 0.15
            x, y = draw_pos[nid]
            radius = max(n["r"] * HALO_SIZE, MIN_HALO * self.sx) * k
            cv2.circle(halos[n["color"]], (int(x * k), int(y * k)), max(1, int(radius)), self.alpha(n, now), -1)
        for i in range(NUM_COLORS):
            a = np.clip(cv2.GaussianBlur(halos[i], (0, 0), GLOW_SIZE * GLOW_RES) * GLOW_OPACITY, 0, 1)
            C = C * (1 - a[..., None]) + (palette[i] / 255.0) * a[..., None]
            A = A * (1 - a) + a
        A = cv2.resize(A, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        C = cv2.resize(C, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        grain = self.grains[self.frame_no % len(self.grains)]
        self.frame_no += 1
        a = np.clip(A * grain, 0, 1)[..., None]
        out = PAPER * (1 - a) + C * grain[..., None]

        # lines
        la = (line_a.astype(np.float32) / 255.0 * LINE_OPACITY)[..., None]
        out = out * (1 - la) + (line_c.astype(np.float32) / 255.0) * la
        img = (np.clip(out, 0, 1) * 255).astype(np.uint8)

        # nodes on top (fading nodes shrink away)
        for nid, n in self.nodes.items():
            x, y = draw_pos[nid]
            r = n["r"] * self.alpha(n, now)
            if r > 0.4:
                cv2.circle(img, (int(x * 16), int(y * 16)), max(1, int(r * 16)), DOT_BGR, -1, cv2.LINE_AA, shift=4)
        return img

    def preview(self, layers, palette, now):
        """Camera preview: a node for every patch of colour, no lines."""
        self.reset()
        for c, alive in enumerate(layers):
            for row, col, cells in clusters(alive):
                x, y = self.to_screen(row, col)
                nid = self.add_node(x, y, self.dot_radius(cells), c, now)
                self.nodes[nid]["r"] = self.nodes[nid]["tr"]
        return self.render(palette, now)


def main():
    cap = open_camera()
    fails = 0
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    mode = "camera"
    groups = centers = palette = life_palette = None
    net = None
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
            if net is None or (net.rows, net.cols) != groups.shape:
                net = Network(*groups.shape)
            live = [(groups == i).astype(np.uint8) for i in range(NUM_COLORS)]
            cv2.imshow(WIN, net.preview(live, palette, now))
        else:
            if now - last_step >= 1.0 / SPEED:
                layers = [game_of_life_step(a) for a in layers]
                net.generation(layers, now)
                last_step = now
            cv2.imshow(WIN, net.render(life_palette, now))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord(" "), 13, 10):
            if mode == "camera":
                layers = [(groups == i).astype(np.uint8) for i in range(NUM_COLORS)]
                life_palette = palette.copy()
                net.start(layers, time.perf_counter())
                mode = "life"
                last_step = time.perf_counter()
            else:
                mode = "camera"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
