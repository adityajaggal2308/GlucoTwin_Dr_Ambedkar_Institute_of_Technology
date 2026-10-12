"""GlucoTwin logo: two overlapping glucose drops (the patient and their digital twin)
joined by a glucose trace. Available as inline SVG (dashboard) and drawn with matplotlib
(PNG favicon, PDF report, slides) from the same geometry."""
from __future__ import annotations

from pathlib import Path

import numpy as np

TEAL, BLUE, VIOLET, SKY = "#14b8a6", "#3b82f6", "#a78bfa", "#38bdf8"


def _drop(cx: float) -> str:
    """Drop centred at x=cx in a 64x64 box: tip at y=6, round bottom of radius 16 centred at y=38."""
    l, r, k = cx - 16, cx + 16, 8.84
    return (f"M{cx} 6 C{cx} 6 {l} 26 {l} 38 C{l} {38 + k} {cx - k} 54 {cx} 54 "
            f"C{cx + k} 54 {r} {38 + k} {r} 38 C{r} 26 {cx} 6 {cx} 6 Z")


TRACE = [(4, 40), (16, 40), (21, 31), (27, 49), (33, 25), (38, 44), (42, 37), (60, 37)]


def logo_svg(size: int = 44) -> str:
    pts = " ".join(f"{x},{y}" for x, y in TRACE)
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="gtA" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{TEAL}"/><stop offset="1" stop-color="{BLUE}"/></linearGradient>
    <linearGradient id="gtB" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{VIOLET}"/><stop offset="1" stop-color="{SKY}"/></linearGradient>
  </defs>
  <path d="{_drop(40)}" fill="url(#gtB)" opacity="0.75"/>
  <path d="{_drop(24)}" fill="url(#gtA)"/>
  <polyline points="{pts}" fill="none" stroke="#ffffff" stroke-width="3.6" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="33" cy="25" r="2.6" fill="#ffffff"/>
</svg>"""


def wordmark_html(size: int = 44, tagline: str = "") -> str:
    return (f'<div style="display:flex;align-items:center;gap:12px">{logo_svg(size)}'
            f'<div><div style="font-size:{int(size * 0.6)}px;font-weight:800;line-height:1.05;letter-spacing:-.5px">'
            f'<span style="color:#f8fafc">Gluco</span><span style="color:{TEAL}">Twin</span></div>'
            + (f'<div style="font-size:13px;color:#94a3b8">{tagline}</div>' if tagline else "") + "</div></div>")


def draw_logo(ax, x0: float = 0, y0: float = 0, size: float = 1.0):
    """Draw the logo on a matplotlib axes, bottom-left at (x0, y0), width/height = size (data units)."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    def tf(x, y):  # svg (y down, 64 box) -> axes data
        return x0 + x / 64 * size, y0 + (64 - y) / 64 * size

    def drop_path(cx):
        l, r, k = cx - 16, cx + 16, 8.84
        pts = [(cx, 6), (cx, 6), (l, 26), (l, 38), (l, 38 + k), (cx - k, 54), (cx, 54),
               (cx + k, 54), (r, 38 + k), (r, 38), (r, 26), (cx, 6), (cx, 6), (cx, 6)]
        codes = [MPath.MOVETO] + [MPath.CURVE4] * 12 + [MPath.CLOSEPOLY]
        return MPath([tf(*p) for p in pts], codes)

    ax.add_patch(PathPatch(drop_path(40), fc=VIOLET, ec="none", alpha=0.75))
    ax.add_patch(PathPatch(drop_path(24), fc=TEAL, ec="none"))
    xs, ys = zip(*[tf(x, y) for x, y in TRACE])
    ax.plot(xs, ys, color="white", lw=3.2 * size, solid_capstyle="round", solid_joinstyle="round")
    cx, cy = tf(33, 25)
    ax.plot([cx], [cy], "o", color="white", ms=4 * size)


def save_logo_png(path: str | Path, px: int = 256):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    draw_logo(ax, 0, 0, 1.0)
    fig.savefig(path, transparent=True); plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).parent.parent / "docs" / "figures"
    save_logo_png(root / "logo.png")
    (root / "logo.svg").write_text(logo_svg(256))
    print("saved", root / "logo.png")
