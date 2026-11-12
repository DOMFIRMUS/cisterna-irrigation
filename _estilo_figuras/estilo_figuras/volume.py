"""3-D renders of micro-CT volumes with PyVista (off-screen, for figures).

    from estilo_figuras import volume
    img = volume.render_mask(pore, voxel_mm=0.03, out="fig_pores_3d.png")
    ef.image(ax, img, kind=None)      # place the render in a matplotlib panel

The render is a bitmap; put it in a journal-sized matplotlib figure so that
lettering, panel letters and the scale bar follow the shared style.
Memory: a binary volume of N voxels needs about 12 N bytes while contouring;
``step`` downsamples by striding before anything is allocated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import COLORS, INK2, SURFACE


def _pv():
    import pyvista as pv

    pv.OFF_SCREEN = True
    return pv


def render_mask(mask, voxel_mm: float | tuple = 1.0, out=None, color: str = COLORS[0], step: int = 1,
                size_px: tuple[int, int] = (1600, 1600), view: str = "iso", box: bool = True,
                smooth: int = 20, opacity: float = 1.0, background: str = SURFACE) -> np.ndarray:
    """Isosurface of a binary volume (z, y, x), e.g. the pore space.

    Returns the RGB render as an array; writes it to ``out`` (PNG) if given.
    """
    pv = _pv()
    m = np.asarray(mask, dtype=np.uint8)[::step, ::step, ::step]
    sp = np.broadcast_to(np.asarray(voxel_mm, float), (3,)) * step
    grid = pv.ImageData(dimensions=m.shape[::-1], spacing=tuple(sp[::-1]))
    grid.point_data["v"] = m.ravel()  # C order of (z, y, x) = VTK order (x fastest)
    surf = grid.contour([0.5], scalars="v")
    if smooth and surf.n_points:
        surf = surf.smooth_taubin(n_iter=smooth, pass_band=0.1)
    pl = pv.Plotter(off_screen=True, window_size=list(size_px))
    pl.set_background(background)
    if surf.n_points:
        pl.add_mesh(surf, color=color, opacity=opacity, smooth_shading=True, specular=0.2, specular_power=15)
    if box:
        pl.add_mesh(grid.outline(), color=INK2, line_width=1.5)
    if view in ("xy", "xz", "yz"):
        getattr(pl, f"view_{view}")()
    else:
        pl.view_isometric()
    img = pl.screenshot(None if out is None else str(Path(out)), return_img=True, transparent_background=False)
    pl.close()
    return img
