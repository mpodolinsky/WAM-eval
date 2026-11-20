"""Draws the architecture diagram of Cosmos3-Edge (`architecture.png` next to this file).

    uv run --no-sync python scripts/cosmos3/architecture_diagram.py [output.png]
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1500, 900
BG, WHITE, GREY, DIM = (24, 24, 24), (255, 255, 255), (165, 165, 165), (95, 95, 95)
YELLOW, BLUE, GREEN, RED = (255, 210, 0), (90, 170, 255), (110, 220, 130), (255, 120, 110)


def font(size):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


big, mid, small, title = font(13), font(12), font(11), font(21)
img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def dashed(xy, color, width=2, dash=6):
    x0, y0, x1, y1 = xy
    for x in range(x0, x1, 2 * dash):
        d.line([x, y0, min(x + dash, x1), y0], fill=color, width=width)
        d.line([x, y1, min(x + dash, x1), y1], fill=color, width=width)
    for y in range(y0, y1, 2 * dash):
        d.line([x0, y, x0, min(y + dash, y1)], fill=color, width=width)
        d.line([x1, y, x1, min(y + dash, y1)], fill=color, width=width)


def box(x, y, w, h, head, lines=(), color=WHITE, dash=False, tag=None, tag_color=GREY):
    if dash:
        dashed((x, y, x + w, y + h), color)
    else:
        d.rounded_rectangle([x, y, x + w, y + h], radius=6, outline=color, width=2)
    d.text((x + 9, y + 6), head, font=big, fill=WHITE)
    for i, line in enumerate(lines):
        d.text((x + 9, y + 25 + i * 15), line, font=small, fill=GREY)
    if tag:
        tw = d.textlength(tag, font=small)
        d.text((x + w - tw - 9, y + 7), tag, font=small, fill=tag_color)


def arrow(p0, p1, color=GREY, width=2):
    d.line([p0, p1], fill=color, width=width)
    (x0, y0), (x1, y1) = p0, p1
    v = np.array([x1 - x0, y1 - y0], float)
    v /= np.linalg.norm(v)
    n = np.array([-v[1], v[0]])
    tip = np.array([x1, y1], float)
    d.polygon([tuple(tip), tuple(tip - 10 * v + 5 * n), tuple(tip - 10 * v - 5 * n)], fill=color)


d.text((24, 14), "Cosmos3-Edge: a Mixture-of-Transformers world action model (4.6B parameters)", font=title, fill=WHITE)
d.text((24, 44), "one network denoises the video latents and the actions together; here as fine-tuned on LIBERO-10", font=mid, fill=GREY)

# ---------------- inputs (left) ----------------
X, WL = 24, 290
d.text((X, 78), "INPUTS", font=mid, fill=GREY)
box(X, 100, WL, 56, "task text", ["JSON prompt: task, camera layout, fps"], YELLOW)
box(X, 180, WL, 56, "tokenizer", ["text tokens (vocabulary 131072)"], YELLOW)
box(X, 260, WL, 56, "SigLIP2 vision tower", ["27 layers, 2x2 patch merge to 2048", "optional image tokens"], YELLOW, dash=True, tag="frozen")

box(X, 372, WL, 56, "2 cameras, 256 x 256", ["tiled into one canvas, 160 x 320"], BLUE)
box(X, 452, WL, 56, "Wan2.2 VAE encoder", ["latents [48, 5, 10, 20]; frame 0 = current"], BLUE, tag="frozen")
box(X, 532, WL, 56, "2x2 patches", ["250 tokens x 192 values"], BLUE)
box(X, 612, WL, 56, "linear (vae2llm)", ["250 video tokens x 2048"], BLUE, tag="trained", tag_color=BLUE)

box(X, 724, WL, 56, "noisy actions + noise level", ["16 steps x 64 (10 used)"], GREEN)
box(X, 804, WL, 56, "action2llm (per embodiment)", ["16 action tokens x 2048"], GREEN, tag="trained", tag_color=GREEN)
for y0, y1 in ((156, 180), (428, 452), (508, 532), (588, 612), (780, 804)):
    arrow((X + WL // 2, y0), (X + WL // 2, y1), DIM)

# ---------------- backbone (centre) ----------------
CX, CW = 400, 640
d.rounded_rectangle([CX, 78, CX + CW, 880], radius=10, outline=GREY, width=2)
d.text((CX + 14, 88), "x 28 MoT decoder layers", font=font(16), fill=WHITE)
d.text((CX + 14, 112), "width 2048, 16 heads / 8 key-value heads, MLP 9216 (squared ReLU), 3D rotary positions", font=small, fill=GREY)

# understanding lane
LX, LW = CX + 20, CW - 40
d.rounded_rectangle([LX, 146, LX + LW, 270], radius=8, outline=YELLOW, width=2)
d.text((LX + 12, 154), "understanding pathway", font=big, fill=YELLOW)
d.text((LX + LW - 62, 155), "frozen", font=small, fill=GREY)
d.text((LX + 12, 175), "reads the text tokens (and image tokens); attends to the text only, causally", font=small, fill=GREY)
for i, name in enumerate(("RMSNorm", "Q K V (und)", "MLP (und)")):
    bx = LX + 14 + i * 190
    d.rounded_rectangle([bx, 208, bx + 170, 250], radius=5, outline=DIM, width=1)
    d.text((bx + 12, 220), name, font=mid, fill=WHITE)

# joint attention
d.rounded_rectangle([LX, 300, LX + LW, 400], radius=8, outline=WHITE, width=2)
d.text((LX + 12, 308), "joint attention: one-way, from the text to the generation pathway", font=big, fill=WHITE)
d.text((LX + 12, 330), "the only place where the two pathways meet (two attention calls, one per pathway)", font=small, fill=GREY)
d.text((LX + 12, 350), "video and action tokens attend to the text and to each other (bidirectional)", font=small, fill=GREY)
d.text((LX + 12, 368), "text tokens never see them: nothing flows back to the understanding pathway", font=small, fill=YELLOW)
arrow((LX + LW // 2, 270), (LX + LW // 2, 300), YELLOW)
d.text((LX + LW // 2 + 10, 278), "text keys / values", font=small, fill=YELLOW)

# generation lane
d.rounded_rectangle([LX, 430, LX + LW, 600], radius=8, outline=BLUE, width=2)
d.text((LX + 12, 438), "generation pathway", font=big, fill=BLUE)
d.text((LX + LW - 62, 439), "trained", font=small, fill=BLUE)
d.text((LX + 12, 459), "250 video tokens + 16 action tokens, bidirectional attention", font=small, fill=GREY)
for i, name in enumerate(("RMSNorm", "Q K V (gen)", "MLP (gen)")):
    bx = LX + 14 + i * 190
    d.rounded_rectangle([bx, 492, bx + 170, 534], radius=5, outline=DIM, width=1)
    d.text((bx + 12, 504), name, font=mid, fill=WHITE)
d.text((LX + 12, 548), "tokens:", font=small, fill=GREY)
d.rectangle([LX + 62, 546, LX + 62 + 250, 562], outline=BLUE, width=2, fill=None)
d.text((LX + 70, 547), "video x 250", font=small, fill=BLUE)
d.rectangle([LX + 322, 546, LX + 322 + 90, 562], outline=GREEN, width=2)
d.text((LX + 330, 547), "actions x 16", font=small, fill=GREEN)
d.text((LX + 12, 574), "own norms, attention projections and MLP: the 'gen' copy of every layer", font=small, fill=GREY)
arrow((LX + LW // 2 - 90, 400), (LX + LW // 2 - 90, 430), BLUE)
arrow((LX + LW // 2 + 90, 430), (LX + LW // 2 + 90, 400), BLUE)

# denoising note
d.rounded_rectangle([CX + 20, 640, CX + CW - 20, 860], radius=8, outline=DIM, width=1)
d.text((CX + 32, 650), "flow matching, as it runs at inference", font=big, fill=WHITE)
notes = [
    "start: video and actions are noise; the current frame is kept clean",
    "each step predicts a velocity for the video latents and for the actions",
    "30 UniPC steps, noise shift 5, no classifier-free guidance",
    "fine-tuned: the generation pathway and the projections in and out",
    "(understanding pathway, vision tower and VAE frozen; 2000 updates, EMA)",
]
for i, t in enumerate(notes):
    d.text((CX + 32, 678 + i * 20), t, font=small, fill=GREY)
d.text((CX + 32, 790), "x  <-  x + dt * v      repeated 30 times", font=mid, fill=YELLOW)

# arrows from the inputs into the lanes
arrow((X + WL, 208), (LX, 208), YELLOW)
arrow((X + WL, 288), (LX, 252), YELLOW)
arrow((X + WL, 640), (LX, 540), BLUE)
arrow((X + WL, 832), (LX, 590), GREEN)

# ---------------- outputs (right) ----------------
RX, RW = 1090, 386
d.text((RX, 78), "OUTPUTS", font=mid, fill=GREY)
box(RX, 450, RW, 56, "video tokens -> llm2vae", ["velocity of the latents, unpatchified to [48, 5, 10, 20]"], BLUE, tag="trained", tag_color=BLUE)
box(RX, 530, RW, 74, "Wan2.2 VAE decoder", ["17 frames, 160 x 320: current + 16 predicted", "only needed to look at the video"], BLUE, dash=True, tag="frozen")
box(RX, 724, RW, 56, "action tokens -> llm2action", ["velocity of the 16 x 10 actions"], GREEN, tag="trained", tag_color=GREEN)
box(RX, 804, RW, 56, "LIBERO command", ["rot6d -> axis-angle: 7-D, 16 steps, executed in full"], GREEN)
arrow((LX + LW, 520), (RX, 478), BLUE)
arrow((LX + LW, 575), (RX, 752), GREEN)
arrow((RX + RW // 2, 506), (RX + RW // 2, 530), BLUE)
arrow((RX + RW // 2, 780), (RX + RW // 2, 804), GREEN)

# legend
d.text((RX, 120), "colours", font=mid, fill=GREY)
for i, (c, t) in enumerate(((YELLOW, "text path"), (BLUE, "video path"), (GREEN, "action path"))):
    d.rectangle([RX, 146 + i * 24, RX + 16, 160 + i * 24], outline=c, width=2)
    d.text((RX + 26, 145 + i * 24), t, font=mid, fill=WHITE)
d.text((RX, 226), "dashed: not needed to act", font=mid, fill=GREY)
d.text((RX, 250), "'frozen' / 'trained': in the LIBERO-10", font=small, fill=GREY)
d.text((RX, 266), "fine-tune of this checkpoint", font=small, fill=GREY)

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("architecture.png")
img.save(out)
print("saved", out, img.size)
