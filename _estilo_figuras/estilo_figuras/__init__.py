"""Shared figure style for the articles of Jean Firmino Cardoso (UFPE).

    import estilo_figuras as ef
    ef.use()                                         # font, rcParams, colormaps
    fig, axs = ef.figure("springer-2", 70, ncols=2)  # 174 mm wide, 70 mm tall
    axs[0].plot(x, y, **ef.series(0, line=True), label="Sample A")
    ef.panel_labels(axs)                             # (a), (b)
    ef.save(fig, out / "fig03_profiles", journal="springer-2")

Rules (from the data-visualisation method the projects already follow)
------------------------------------------------------------------------
* Lettering 8-12 pt sans-serif (Springer and Elsevier artwork rules), in
  Source Sans 3 (SIL Open Font License, bundled in ``fonts/``).
* Categorical colour encodes identity, in a fixed order that is never cycled
  and always follows the entity (a filter never repaints the survivors).
  The order passes the colour-vision-deficiency checks on adjacent pairs; the
  first three slots also pass on all pairs (scatter, maps, small multiples).
  Colour is always paired with a second channel (marker, line style or a
  direct label) so a figure survives greyscale printing.
* Sequential maps: one hue, light -> dark. Diverging maps: two opposite hues
  around a light neutral midpoint. No rainbow maps; grey for CT images.
* Hairline, solid, recessive grid and axes; no dual y-axes.
* Text wears ink colours, never a series colour.
* Export: vector PDF with embedded TrueType fonts plus 600-dpi PNG and/or
  LZW TIFF (RGB, no alpha), sized to the journal column.
"""
from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D

__version__ = "1.0.0"
__all__ = [
    "use", "figure", "width_in", "series", "tint", "legend_handles", "panel_labels", "scalebar",
    "label_points", "cmap", "image", "check", "fit_width", "save", "pdf_fonts",
    "COLORS", "COLOR_NAMES", "MARKERS", "LINESTYLES", "BLUES", "WIDTHS_MM", "MAX_HEIGHT_MM",
    "INK", "INK2", "MUTED", "GRID", "SURFACE", "NEUTRAL", "GAP", "RING", "MM",
]

log = logging.getLogger("estilo_figuras")

HERE = Path(__file__).resolve().parent
STYLE_FILE = HERE / "artigo.mplstyle"
FONT_DIR = HERE / "fonts"
FONT = "Source Sans 3"

MM = 1 / 25.4

#: Column widths (mm). Springer: 84 / 129 / 174; Elsevier: 90 / 140 / 190.
WIDTHS_MM = {
    "springer-1": 84.0, "springer-1.5": 129.0, "springer-2": 174.0,
    "elsevier-1": 90.0, "elsevier-1.5": 140.0, "elsevier-2": 190.0,
}
#: Largest printable artwork height (mm) per publisher.
MAX_HEIGHT_MM = {"springer": 234.0, "elsevier": 240.0}

# ---------------------------------------------------------------- colours
#: Categorical palette in its validated order (blue, orange, aqua, yellow,
#: magenta, green, violet, red). Validated on a white page: lightness band,
#: chroma floor, adjacent CVD separation (worst dE 9.1) and normal-vision floor
#: (worst dE 19.6). Aqua, yellow and magenta sit below 3:1 contrast on white,
#: so they always come with a marker or a direct label.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLOR_NAMES = ["blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red"]
#: Marker paired with each slot (secondary encoding; filled shapes first).
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
#: Line style paired with each slot when lines carry no markers.
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1.5, 1, 1.5, 1, 1.5)), (0, (8, 2)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1))]

#: Steps 100-700 of slot 1 (blue), light -> dark. For an ordinal encoding
#: (ordered levels drawn as separate lines or marks) start at index 3, the
#: lightest step that still clears 2:1 contrast on white.
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

INK = "#0b0b0b"       # primary text
INK2 = "#52514e"      # secondary text, axes, ticks
MUTED = "#898781"     # annotations, leader lines
GRID = "#e1e0d9"      # hairline grid
SURFACE = "#ffffff"   # page
NEUTRAL = "#52514e"   # "other", reference or control group (not a series hue)
REF_BAND = "#e8e7e2"  # shaded reference interval

#: Surface gap between touching fills (bars, stacked segments, histogram bins).
GAP = {"edgecolor": SURFACE, "linewidth": 0.6}
#: Surface ring around scatter markers so overlapping points stay legible.
RING = {"edgecolors": SURFACE, "linewidths": 0.5}

#: Colormaps by job. cmcrameri maps are perceptually uniform and readable
#: under colour-vision deficiency (Crameri et al. 2020, Nat. Commun. 11, 5444).
CMAPS = {
    "sequential": "cmc.devon_r",   # one hue, light -> dark (magnitude)
    "diverging": "cmc.vik",        # blue <-> red around a light midpoint (sign)
    "cyclic": "cmc.romaO",         # angles, orientations, phase
    "multi": "cmc.batlow",         # long continuous fields needing fine discrimination
    "ct": "gray",                  # tomography and other grey-level images
}

_FONT_READY = False


# ---------------------------------------------------------------- setup
def _register_fonts() -> bool:
    """Make the bundled Source Sans 3 visible to matplotlib (idempotent)."""
    global _FONT_READY
    from matplotlib import font_manager as fm

    if not _FONT_READY:
        for ttf in sorted(FONT_DIR.glob("*.ttf")):
            try:
                fm.fontManager.addfont(str(ttf))
            except Exception as exc:  # noqa: BLE001 - a broken font file must not stop a pipeline
                log.warning("could not register %s: %s", ttf.name, exc)
        _FONT_READY = True
    return any(f.name == FONT for f in fm.fontManager.ttflist)


def _register_cmaps() -> None:
    try:
        import cmcrameri.cm  # noqa: F401 - registers the "cmc.*" colormaps
    except ImportError:
        log.info("cmcrameri not installed; falling back to matplotlib colormaps")


def use(font_size: float | None = None, grid: bool | str = True) -> str:
    """Apply the style. Returns the font family actually used.

    font_size: base size in pt (default 8, the journal minimum); every other
               size keeps its offset from it.
    grid:      True (both axes), "x", "y" or False.
    """
    has_font = _register_fonts()
    _register_cmaps()
    plt.style.use(["default", STYLE_FILE])
    if not has_font:
        log.warning("font %r not found; using %s", FONT, mpl.rcParams["font.sans-serif"][1:])
        mpl.rcParams["mathtext.fontset"] = "dejavusans"
    if font_size is not None:
        d = float(font_size) - 8.0
        for k in ("font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize", "ytick.labelsize",
                  "legend.fontsize", "legend.title_fontsize", "figure.titlesize"):
            mpl.rcParams[k] = float(mpl.rcParams[k]) + d
    if grid in (False, None):
        mpl.rcParams["axes.grid"] = False
    elif grid in ("x", "y"):
        mpl.rcParams["axes.grid.axis"] = grid
    mpl.rcParams["image.cmap"] = cmap("sequential").name
    return FONT if has_font else mpl.rcParams["font.sans-serif"][1]


def width_in(width: str | float) -> float:
    """Column width in inches, from a name in WIDTHS_MM or a number in mm."""
    mm = WIDTHS_MM[width] if isinstance(width, str) else float(width)
    return mm * MM


def figure(width: str | float = "springer-2", height_mm: float = 70.0, nrows: int = 1, ncols: int = 1,
           **subplots_kw):
    """Figure sized to a journal column, with constrained layout.

    Returns (fig, axes) exactly like ``plt.subplots``.
    """
    subplots_kw.setdefault("layout", "constrained")
    return plt.subplots(nrows, ncols, figsize=(width_in(width), height_mm * MM), **subplots_kw)


# ---------------------------------------------------------------- encodings
def series(i: int, line: bool = False, open_marker: bool = False) -> dict:
    """Style for categorical slot ``i`` (0-7): colour plus its paired marker.

    line=True adds the paired line style (for lines without markers, use
    ``marker=None`` afterwards or pop it). open_marker=True draws a hollow
    marker (white face) for a binary attribute such as "does not percolate".
    """
    if not 0 <= i < len(COLORS):
        raise IndexError(f"slot {i}: the palette has {len(COLORS)} slots. Fold the tail into 'Other', "
                         "facet into small multiples, or encode with markers; never generate a 9th hue.")
    st = {"color": COLORS[i], "marker": MARKERS[i]}
    if line:
        st["linestyle"] = LINESTYLES[i]
    if open_marker:
        st.update(markerfacecolor=SURFACE, markeredgecolor=COLORS[i])
    return st


def tint(color: int | str, strength: float = 0.45) -> str:
    """Light wash of a colour for areas (violins, bands, filled regions):
    ``strength`` 1 = the colour itself, 0 = white. ``color`` may be a slot."""
    rgb = np.array(mpl.colors.to_rgb(COLORS[color] if isinstance(color, int) else color))
    return mpl.colors.to_hex(1 - strength * (1 - rgb))


def legend_handles(labels, slots=None, line: bool = False, **kw) -> list[Line2D]:
    """Legend proxies for categories (marker + colour, optional line)."""
    slots = range(len(labels)) if slots is None else slots
    out = []
    for lab, i in zip(labels, slots):
        st = series(i, line=line)
        if not line:
            st["linestyle"] = "none"
        st.update(kw)
        out.append(Line2D([], [], label=lab, **st))
    return out


def panel_labels(axes, start: str = "a", fmt: str = "({})", inside: bool = False, size: float | None = None,
                 dx_pt: float = -2.0, dy_pt: float = 3.0, **text_kw) -> list:
    """Bold panel letters (a), (b), ... at the top-left corner of each axes.

    inside=False puts the letter above the top-left corner (plots);
    inside=True puts it inside with a white halo (images, maps).
    """
    import matplotlib.patheffects as pe

    axes = list(np.atleast_1d(axes).ravel())
    size = size if size is not None else mpl.rcParams["font.size"] + 1.5
    out = []
    for k, ax in enumerate(axes):
        s = fmt.format(chr(ord(start) + k))
        kw = dict(fontsize=size, fontweight="bold", color=INK)
        if inside:
            kw.update(xy=(0, 1), xytext=(4, -4), ha="left", va="top",
                      path_effects=[pe.withStroke(linewidth=2.2, foreground=SURFACE)])
        else:
            kw.update(xy=(0, 1), xytext=(dx_pt, dy_pt), ha="right", va="bottom")
        kw.update(text_kw)
        out.append(ax.annotate(s, xycoords="axes fraction", textcoords="offset points", **kw))
    return out


def scalebar(ax, length: float, label: str, loc: str = "lower right", color: str = SURFACE,
             thickness_frac: float = 0.012, pad: float = 0.5):
    """Physical scale bar on an image axes. ``length`` is in data units
    (pixels, or mm when the image was drawn with a physical ``extent``)."""
    import matplotlib.patheffects as pe
    from matplotlib.font_manager import FontProperties
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    y0, y1 = sorted(ax.get_ylim())
    halo = INK if color == SURFACE else SURFACE
    bar = AnchoredSizeBar(ax.transData, length, label, loc, pad=pad, sep=2, color=color, frameon=False,
                          size_vertical=thickness_frac * (y1 - y0), label_top=True,
                          fontproperties=FontProperties(size=mpl.rcParams["font.size"], weight="semibold"))
    bar.txt_label._text.set_path_effects([pe.withStroke(linewidth=1.6, foreground=halo)])
    bar.size_bar.get_children()[0].set_path_effects([pe.withStroke(linewidth=1.2, foreground=halo)])
    ax.add_artist(bar)
    return bar


def label_points(ax, x, y, labels, color: str = INK2, size: float | None = None, **adjust_kw) -> list:
    """Direct labels next to points, moved apart with adjustText and tied to
    their point by a thin leader line. Use it sparingly (extremes, the points
    the text discusses), never on every point."""
    size = size if size is not None else mpl.rcParams["font.size"]
    texts = [ax.text(xi, yi, str(s), fontsize=size, color=color) for xi, yi, s in zip(x, y, labels)]
    try:
        from adjustText import adjust_text
    except ImportError:
        log.warning("adjustText not installed; labels left at their points")
        return texts
    adjust_kw.setdefault("arrowprops", dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0, shrinkB=2))
    adjust_kw.setdefault("expand", (1.2, 1.4))
    adjust_text(texts, x=np.asarray(x, float), y=np.asarray(y, float), ax=ax, **adjust_kw)
    return texts


def cmap(kind: str = "sequential", n: int | None = None):
    """Colormap for a job: 'sequential', 'diverging', 'cyclic', 'multi' or 'ct'.
    ``n`` returns a discrete (classed) version with n colours."""
    name = CMAPS[kind]
    try:
        cm = mpl.colormaps[name]
    except KeyError:
        _register_cmaps()
        fallback = {"sequential": "Blues", "diverging": "RdBu_r", "cyclic": "twilight", "multi": "cividis"}
        cm = mpl.colormaps.get(name) or mpl.colormaps[fallback.get(kind, "gray")]
    return cm.resampled(n) if n else cm


def image(ax, img, kind: str = "ct", **imshow_kw):
    """Show an image (CT slice, map tile) without grid, ticks or frame."""
    imshow_kw.setdefault("cmap", cmap(kind) if np.ndim(img) == 2 else None)
    imshow_kw.setdefault("interpolation", "nearest")
    im = ax.imshow(img, **imshow_kw)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return im


# ---------------------------------------------------------------- checks and export
def _final_size_in(fig) -> tuple[float, float]:
    """Size (in) of the saved file with bbox_inches='tight'."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    pad = float(mpl.rcParams["savefig.pad_inches"])
    return bb.width + 2 * pad, bb.height + 2 * pad


def check(fig, journal: str | None = None, min_pt: float = 8.0, max_pt: float = 12.0) -> list[str]:
    """List artwork problems: lettering outside min_pt-max_pt (after the
    scaling to the column width, if a journal width is given) and artwork
    larger than the column or the printable height."""
    issues = []
    w_in, h_in = _final_size_in(fig)
    scale = 1.0
    if journal:
        col = width_in(journal)
        if w_in > col + 0.5 * MM:
            scale = col / w_in
            issues.append(f"final width {w_in / MM:.1f} mm exceeds the {journal} column ({col / MM:.0f} mm); "
                          f"it will be shrunk by {scale:.2f}")
        hmax = MAX_HEIGHT_MM[journal.split("-")[0]]
        if h_in * scale > hmax * MM:
            issues.append(f"final height {h_in * scale / MM:.1f} mm exceeds {hmax:.0f} mm")
    sizes = {}
    for t in fig.findobj(mpl.text.Text):
        s = t.get_text().strip()
        if not s or not t.get_visible() or t.figure is None:
            continue
        sizes.setdefault(round(t.get_fontsize() * scale, 2), []).append(s)
    for pt, texts in sorted(sizes.items()):
        if pt < min_pt - 1e-6 or pt > max_pt + 1e-6:
            issues.append(f"{len(texts)} text(s) at {pt:g} pt (allowed {min_pt:g}-{max_pt:g}), e.g. {texts[0]!r}")
    return issues


def fit_width(fig, width: str | float, tol_mm: float = 0.2, max_iter: int = 8) -> float:
    """Rescale the canvas so the *saved* (tight) artwork is exactly the column
    width. Text sizes in pt are unchanged; the layout reflows. Returns mm."""
    target = width_in(width)
    w_in = _final_size_in(fig)[0]
    for _ in range(max_iter):
        if abs(w_in - target) < tol_mm * MM:
            break
        size0 = fig.get_size_inches().copy()
        fig.set_size_inches(size0[0] * target / w_in, size0[1])
        w_new = _final_size_in(fig)[0]
        if abs(w_new - w_in) < 0.1 * MM:      # width set by fixed aspects (images): leave it
            fig.set_size_inches(*size0)
            break
        w_in = w_new
    return float(w_in / MM)


def save(fig, stem, formats=("pdf", "png"), dpi: int = 600, journal: str | None = None, close: bool = True,
         strict: bool = False, exact_width: bool = True) -> dict:
    """Write the figure as vector PDF (TrueType fonts embedded, reproducible
    bytes) and/or 600-dpi PNG / LZW TIFF (RGB, no alpha).

    formats:     any of "pdf", "png", "tiff"/"tif", "svg", "eps".
    journal:     a WIDTHS_MM key; the artwork is checked against that column
                 and, with exact_width=True, rescaled to fill it exactly.
    strict:      raise instead of logging when check() finds a problem.
    Returns {"paths": [...], "size_mm": (w, h), "issues": [...]}.
    """
    from PIL import Image

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    if journal and exact_width:
        fit_width(fig, journal)
    issues = check(fig, journal)
    for msg in issues:
        log.warning("%s: %s", stem.name, msg)
    if strict and issues:
        raise ValueError(f"{stem.name}: " + "; ".join(issues))
    w_in, h_in = _final_size_in(fig)
    paths = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        p = stem.with_suffix("." + fmt)
        if fmt == "pdf":
            fig.savefig(p, format="pdf", metadata={"CreationDate": None, "Creator": f"estilo_figuras {__version__}"})
        elif fmt in ("tif", "tiff"):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi)
            buf.seek(0)
            with Image.open(buf) as im:
                im.convert("RGB").save(p, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
        elif fmt == "png":
            fig.savefig(p, format="png", dpi=dpi, metadata={"Software": f"estilo_figuras {__version__}"})
        else:
            fig.savefig(p, format=fmt, dpi=dpi)
        paths.append(p)
    if close:
        plt.close(fig)
    return {"paths": paths, "size_mm": (round(float(w_in / MM), 1), round(float(h_in / MM), 1)), "issues": issues}


def pdf_fonts(path) -> dict[str, bool]:
    """Fonts named in a PDF and whether each one is embedded."""
    raw = Path(path).read_bytes()
    names = set(re.findall(rb"/BaseFont\s*/([A-Za-z0-9+_\-,]+)", raw))
    embedded = bool(re.search(rb"/FontFile[23]?\b", raw))
    return {n.decode(): embedded for n in names}
