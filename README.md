# Studio 7 early prototypes

Early digital prototypes Studio 7 worked on during the Concept Phase.

Most prototypes are single HTML files that run in a browser. The live links below open them directly. You can also download the repo and open any `.html` file in your browser.

| Prototype | What it is | Try it |
|---|---|---|
| [Spindelpaal 3D](#spindelpaal-3d) | A pole of ball screws that makes spheres breathe | [Open](https://emergence-delft.github.io/studio-7-early-prototypes/spindelpaal-3d.html) |
| [Light-tree lamp](#light-tree-lamp) | Sliding tube units that fold open into flowers of light | [Open](https://emergence-delft.github.io/studio-7-early-prototypes/light-tree-lamp/Light-tree%20lamp%20prototype.html) |
| [Arrow Swarm](#arrow-swarm) | A swarm that comes when called and flees when scared | [Open](https://emergence-delft.github.io/studio-7-early-prototypes/arrow-swarm/Arrow%20Swarm.html) |
| [Facing Life](#facing-life) | Your face as a pixel grid, brought to life by the Game of Life | [Open](https://emergence-delft.github.io/studio-7-early-prototypes/facing-life/hex_mirror.html) |

---

## Spindelpaal 3D

An interactive 3D model of a spindle pole. Ball screws inside one pole each move a collar, and the strips between the collars form spheres that open and close ("breathe"). You choose how many spheres and poles there are, then work through the tabs from left to right: movement, sphere shape, pole shape, light, build and setup. The interface is available in Dutch and English.

- **[▶ Open Spindelpaal 3D](https://emergence-delft.github.io/studio-7-early-prototypes/spindelpaal-3d.html)**
- File: [`spindelpaal-3d.html`](spindelpaal-3d.html)

## Light-tree lamp

Variations on a lamp made of stacked tube units. Drag a tape ring (or a knob on the left rail) to slide the units in and out, and fold each pipe open into a flower. You can switch between a "photonic" and an "electronic" state, set the number of strips per unit and play LED patterns such as Wave and Breathe.

| Version | File |
|---|---|
| Light-tree lamp | [Open](https://emergence-delft.github.io/studio-7-early-prototypes/light-tree-lamp/Light-tree%20lamp%20prototype.html) · [`Light-tree lamp prototype.html`](light-tree-lamp/Light-tree%20lamp%20prototype.html) |
| Tree prototype (earlier version) | [Open](https://emergence-delft.github.io/studio-7-early-prototypes/light-tree-lamp/tree-prototype.html) · [`tree-prototype.html`](light-tree-lamp/tree-prototype.html) |
| Light-tube lamp (6 strips per unit) | [Open](https://emergence-delft.github.io/studio-7-early-prototypes/light-tree-lamp/Light-tube%20lamp%20prototype.html) · [`Light-tube lamp prototype.html`](light-tree-lamp/Light-tube%20lamp%20prototype.html) |
| Paper tube | [Open](https://emergence-delft.github.io/studio-7-early-prototypes/light-tree-lamp/Paper%20tube%20prototype.html) · [`Paper tube prototype.html`](light-tree-lamp/Paper%20tube%20prototype.html) |

## Arrow Swarm

A swarm of arrows that reacts to you. Move the cursor to call them in, click to scare them, and scare them often enough and they leave. A "Fury" meter shows how agitated they are, and you can calm them down again. The cubed version puts the swarm inside a cube you can drag to rotate.

| Version | File |
|---|---|
| Arrow Swarm | [Open](https://emergence-delft.github.io/studio-7-early-prototypes/arrow-swarm/Arrow%20Swarm.html) · [`Arrow Swarm.html`](arrow-swarm/Arrow%20Swarm.html) |
| Arrow Swarm cubed | [Open](https://emergence-delft.github.io/studio-7-early-prototypes/arrow-swarm/Arrow%20Swarm%20cubed.html) · [`Arrow Swarm cubed.html`](arrow-swarm/Arrow%20Swarm%20cubed.html) |

## Facing Life

Experiments that turn a webcam image into a coarse grid of pixels and then let Conway's Game of Life take over, so a portrait slowly evolves into something living.

### Browser prototypes

These need webcam access, so allow the camera when your browser asks.

| Prototype | What it does | File |
|---|---|---|
| Hex Mirror | Face-tracked 20 × 20 hexagonal pixel mirror in 4 tones. Take a picture and watch it evolve as a Game of Life. Can send the pixel values to an ESP32 over USB to drive servos. | [Open](https://emergence-delft.github.io/studio-7-early-prototypes/facing-life/hex_mirror.html) · [`hex_mirror.html`](facing-life/hex_mirror.html) |
| Hex Mirror (variant) | Another iteration of Hex Mirror. | [`hex_mirror (5).html`](facing-life/hex_mirror%20(5).html) |
| Webcam Stickman | Tracks your body through the webcam and draws you as a stickman in a separate window. | [Open](https://emergence-delft.github.io/studio-7-early-prototypes/facing-life/stickman.html) · [`stickman.html`](facing-life/stickman.html) |

### Python prototypes

These run on your computer and use your webcam.

**Install** (Python 3):

```bash
python -m pip install opencv-python numpy pyvista
```

`pyvista` is only needed for the two 3D scripts.

**Run** a script from the repo folder, for example:

```bash
python facing-life/pixel_life_glow.py
```

Click the window once so it receives key presses. **Q** or **Esc** quits.

| Script | What it does |
|---|---|
| [`video to game of life.py`](facing-life/video%20to%20game%20of%20life.py) | The base version: turns the webcam image into a Game of Life. **Space** or **Enter** takes the picture. |
| [`game of life 5 colours.py`](facing-life/game%20of%20life%205%20colours.py) | The same idea with 5 colours. |
| [`pixel_life_glow.py`](facing-life/pixel_life_glow.py) | Soft, grainy colour glows on warm paper with black dots and thin wandering lines. |
| [`pixel_life_milk.py`](facing-life/pixel_life_milk.py) | Inspired by dye tablets in milk: saturated colours bloom, bleed and swirl together. |
| [`pixel_life_network.py`](facing-life/pixel_life_network.py) | Glows on warm paper with black nodes joined into a growing network of lines. |
| [`pixel_life_3d.py`](facing-life/pixel_life_3d.py) | Each generation becomes a layer of cubes, stacking into a 3D tower where height is time. Based on the idea of [pbelmans/conway](https://github.com/pbelmans/conway). |
| [`pixel_life_3d_colors.py`](facing-life/pixel_life_3d_colors.py) | The 3D tower in 5 colours. |

---

## Repository layout

```
studio-7-early-prototypes/
├── spindelpaal-3d.html
├── arrow-swarm/
├── facing-life/
└── light-tree-lamp/
```

## Enabling the live links

The "Open" links use GitHub Pages. To turn it on, a repo admin goes to **Settings → Pages**, sets the source to **Deploy from a branch**, and chooses **main** and **/ (root)**. The links work a minute or two later.

## Credits

Made by Studio 7 during the Concept Phase of Emergence Delft.
