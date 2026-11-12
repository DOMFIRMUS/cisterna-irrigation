"""Low-cost capacitive soil-moisture probe: field model, equivalent circuit, relaxation-oscillator
readout and dual-frequency inversion.

Probe: double-sided FR-4 board (electrodes A and B on both faces, joined by vias), covered by a
dielectric coating, buried in soil. Quasi-static 2-D finite-volume model of the cross-section with
complex relative permittivity eps* = eps' - j sigma / (omega eps0) in every region; the symmetry of the
layout reduces the problem to a quarter domain (x >= 0, y >= 0) with V = 1/2 on the plane x = 0.
The admittance per unit length is Y' = j omega eps0 F(eps*_soil), F dimensionless.

Equivalent circuit (validated against the field model): F = F0 + Fc Fs eps / (Fc + Fs eps),
i.e. a direct capacitance C0 in parallel with the coating capacitance Cc in series with the soil
(Cs || Gs), Cs = eps0 Fs eps'_soil L, Gs = Fs sigma_b L.
"""
import math
from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq, least_squares
from scipy.special import ellipk

EPS0 = 8.8541878128e-12


# ------------------------------------------------------------------------------------ dielectrics
def eps_water(t_c):
    """Static relative permittivity of water (Malmberg & Maryott, 1956)."""
    return 87.740 - 0.40008 * t_c + 9.398e-4 * t_c ** 2 - 1.410e-6 * t_c ** 3


def eps_crim(theta, t_c=25.0, porosity=0.42, eps_solid=4.7):
    """Complex refractive index model (Roth et al., 1990), alpha = 0.5."""
    return (theta * np.sqrt(eps_water(t_c)) + (1 - porosity) * np.sqrt(eps_solid) + (porosity - theta)) ** 2


def theta_crim(eps_b, t_c=25.0, porosity=0.42, eps_solid=4.7):
    return (np.sqrt(eps_b) - (1 - porosity) * np.sqrt(eps_solid) - porosity) / (np.sqrt(eps_water(t_c)) - 1)


def theta_topp(eps_b):
    """Topp et al. (1980)."""
    e = np.asarray(eps_b)
    return -5.3e-2 + 2.92e-2 * e - 5.5e-4 * e ** 2 + 4.3e-6 * e ** 3


def sigma_bulk(eps_b, sigma_pore, t_c=25.0, eps_offset=4.1, sigma_min=1e-4):
    """Bulk conductivity from pore-water conductivity (Hilhorst, 2000): sigma_b = sigma_p (eps_b - 4.1)/eps_p,
    with the pore-water conductivity referred to 25 degC by +2 %/K."""
    sp_t = sigma_pore * (1 + 0.02 * (t_c - 25.0))
    return np.maximum(sp_t * (eps_b - eps_offset) / eps_water(t_c), sigma_min)


# ----------------------------------------------------------------------------- finite-volume model
@dataclass(frozen=True)
class Probe:
    w: float = 4.0e-3          # electrode width (m)
    s: float = 2.0e-3          # gap between electrodes (m)
    margin: float = 1.0e-3     # FR-4 beyond the outer electrode edge (m)
    tb: float = 1.6e-3         # board thickness (m)
    tc: float = 25e-6          # coating thickness (m)
    eps_board: float = 4.4     # FR-4
    eps_coat: float = 3.8      # solder mask / epoxy
    gap: float = 0.0           # air gap between coating and soil (m)
    length: float = 0.08       # sensing length (m)

    @property
    def half_width(self):
        return self.s / 2 + self.w + self.margin


def _segment(a, b, ha, hb, hmax, r):
    """Points in (a, b] graded from spacing ha at a and hb at b up to hmax."""
    left, h, x = [], ha, a
    right, g, y = [], hb, b
    while y - x > 1.5 * max(h, g):
        if h <= g:                      # advance the finer side, so both ends grade smoothly
            x += h; left.append(x); h = min(h * r, hmax)
        else:
            y -= g; right.append(y); g = min(g * r, hmax)
    pts = left + right[::-1] + [b]
    return np.array(pts)


def graded_axis(anchors, fine, hmax, r=1.25):
    """1-D grid through the sorted anchors; spacing fine[k] at anchor k."""
    out = [np.array([anchors[0]])]
    for k in range(len(anchors) - 1):
        out.append(_segment(anchors[k], anchors[k + 1], fine[k], fine[k + 1], hmax, r))
    return np.concatenate(out)


def build_grid(pr, extent=0.06, hmin=None, hmax=4e-3, r=1.25, validation=False):
    hmin = hmin or max(min(pr.tc / 4, 5e-6), 1e-6) if not validation else (hmin or 5e-6)
    xa = [0.0, pr.s / 2, pr.s / 2 + pr.w, pr.half_width, extent] if not validation else [0.0, pr.s / 2, pr.s / 2 + pr.w, extent]
    xf = [hmin * 20, hmin, hmin, hmin * 4, hmax] if not validation else [hmin * 20, hmin, hmin, hmax]
    x = graded_axis(xa, xf, hmax, r)
    if validation:
        y = graded_axis([0.0, extent], [hmin, hmax], hmax, r)
    else:
        top = pr.tb / 2
        ya = [0.0, top, top + pr.tc] + ([top + pr.tc + pr.gap] if pr.gap > 0 else []) + [extent]
        yf = [min(pr.tb / 8, 1e-4), hmin, hmin] + ([hmin] if pr.gap > 0 else []) + [hmax]
        y = graded_axis(ya, yf, hmax, r)
    return x, y


def solve_field(pr, eps_soil, extent=0.06, hmin=None, hmax=4e-3, r=1.25, validation=False, return_field=False):
    """Return F = C'/eps0 (complex, both faces, both halves) for a homogeneous soil eps_soil (complex)."""
    x, y = build_grid(pr, extent, hmin, hmax, r, validation)
    nx, ny = len(x), len(y)
    xc, yc = (x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    eps = np.full(X.shape, complex(eps_soil))
    if not validation:
        top = pr.tb / 2
        inb = X < pr.half_width
        eps[inb & (Y < top)] = pr.eps_board
        eps[inb & (Y >= top) & (Y < top + pr.tc)] = pr.eps_coat
        if pr.gap > 0:
            eps[inb & (Y >= top + pr.tc) & (Y < top + pr.tc + pr.gap)] = 1.0
    dx, dy = np.diff(x), np.diff(y)
    # face conductances (relative permittivity x geometric factor)
    ep = np.pad(eps, 1, constant_values=0)                 # cells padded so edge faces see zero outside
    dyp = np.pad(dy, 1, constant_values=0)
    dxp = np.pad(dx, 1, constant_values=0)
    gx = (ep[1:-1, :-1] * dyp[None, :-1] / 2 + ep[1:-1, 1:] * dyp[None, 1:] / 2) / dx[:, None]   # (nx-1, ny)
    gy = (ep[:-1, 1:-1] * dxp[:-1, None] / 2 + ep[1:, 1:-1] * dxp[1:, None] / 2) / dy[None, :]   # (nx, ny-1)
    idx = np.arange(nx * ny).reshape(nx, ny)
    rows, cols, vals = [], [], []
    for (a, b, g) in ((idx[:-1, :], idx[1:, :], gx), (idx[:, :-1], idx[:, 1:], gy)):
        a, b, g = a.ravel(), b.ravel(), g.ravel()
        rows += [a, b, a, b]; cols += [a, b, b, a]; vals += [g, g, -g, -g]
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny, nx * ny))
    # Dirichlet nodes
    vfix = np.full(nx * ny, np.nan, complex)
    vfix[idx[0, :]] = 0.5; vfix[idx[-1, :]] = 0.5; vfix[idx[:, -1]] = 0.5
    je = 0 if validation else int(np.argmin(np.abs(y - pr.tb / 2)))
    ie = np.where((x >= pr.s / 2 - 1e-12) & (x <= pr.s / 2 + pr.w + 1e-12))[0]
    elec = idx[ie, je]
    vfix[elec] = 0.0
    fixed = ~np.isnan(vfix)
    free = ~fixed
    Kff = K[free][:, free].tocsc()
    rhs = -K[free][:, fixed] @ vfix[fixed]
    V = vfix.copy()
    V[free] = spla.spsolve(Kff, rhs)
    q = (K[elec] @ V)                         # flux out of electrode B nodes (quarter domain)
    F = -2 * q.sum()                           # both faces / both halves, unit potential difference
    if return_field:
        return F, (x, y, V.reshape(nx, ny), eps)
    return F


def cps_analytic(w, s):
    """Coplanar strips of zero thickness in a homogeneous medium: C'/(eps0 eps) = K(k')/K(k), k = s/(s+2w)."""
    k = s / (s + 2 * w)
    return ellipk(1 - k ** 2) / ellipk(k ** 2)


@dataclass(frozen=True)
class Lumped:
    F0: float
    Fc: float
    Fs: float

    def F(self, eps_soil):
        e = np.asarray(eps_soil, complex)
        return self.F0 + self.Fc * self.Fs * e / (self.Fc + self.Fs * e)


def fit_lumped(pr, eps_points, **kw):
    """Fit (F0, Fc, Fs) to field solutions at complex soil permittivities; returns the fit and the
    worst relative error |F_fit - F_fv| / |F_fv| at the fitting points."""
    F = np.array([solve_field(pr, e, **kw) for e in eps_points])

    def res(p):
        m = Lumped(*np.exp(p)).F(eps_points)
        d = (m - F) / np.abs(F)
        return np.concatenate([d.real, d.imag])

    p0 = np.log([abs(F[0]) * 0.5, 50.0, abs(F[0]) * 0.5])
    sol = least_squares(res, p0)
    lm = Lumped(*np.exp(sol.x))
    err = np.max(np.abs(lm.F(eps_points) - F) / np.abs(F))
    return lm, err, F


# --------------------------------------------------------------------------- relaxation oscillator
@dataclass(frozen=True)
class Oscillator:
    vdd: float = 3.3
    vtp: float = 1.80          # positive-going threshold (V)
    vtn: float = 1.10          # negative-going threshold (V)
    tpd: float = 4e-9          # propagation delay (s)
    rout: float = 25.0         # output resistance (ohm)
    c_par: float = 4e-12       # input capacitance + trace (F)


def period(R, C0, Cc, Cs, Gs, osc=Oscillator()):
    """Period of a Schmitt-trigger RC oscillator whose timing element is the probe network
    (C_par + C0 in parallel with Cc in series with Cs || Gs). Exact piecewise-linear-ODE solution:
    the periodic orbit is the fixed point of the one-cycle map, found by Newton iteration."""
    Re = R + osc.rout
    Ca = C0 + osc.c_par
    M = np.array([[Ca + Cc, -Cc], [-Cc, Cc + Cs]])
    A = np.array([[-1 / Re, 0.0], [0.0, -max(Gs, 1e-13)]])
    L = np.linalg.solve(M, A)
    lam, P = np.linalg.eig(L)
    lam, P = lam.real, P.real
    Pinv = np.linalg.inv(P)
    l1, l2 = lam

    def phase(s0, vo, vt):
        c = Pinv @ (s0 - np.array([vo, 0.0]))
        a1, a2 = P[0, 0] * c[0], P[0, 1] * c[1]
        f = lambda t: vo + a1 * math.exp(l1 * t) + a2 * math.exp(l2 * t) - vt
        tau = 1 / max(abs(l1), abs(l2))
        lo, hi = 0.0, tau * 0.01
        f0 = f(0.0)
        while np.sign(f(hi)) == np.sign(f0):
            lo, hi = hi, hi * 2
            if hi > 1e3 / min(abs(l1), abs(l2)):
                raise RuntimeError("no threshold crossing")
        tc = brentq(f, lo, hi, xtol=1e-16, rtol=1e-13)
        te = tc + osc.tpd
        s1 = np.array([vo, 0.0]) + P @ (c * np.exp(lam * te))
        return tc, s1

    def cycle(s0):
        t1, s1 = phase(s0, osc.vdd, osc.vtp)
        t2, s2 = phase(s1, 0.0, osc.vtn)
        return t1 + t2 + 2 * osc.tpd, s2

    vbar = (osc.vtp + osc.vtn) / 2
    s = np.array([osc.vtn, (osc.vtn - vbar) * Cc / (Cc + Cs)])
    for _ in range(30):
        T, s1 = cycle(s)
        g = s1 - s
        if np.max(np.abs(g)) < 1e-12:
            break
        J = np.zeros((2, 2))
        for k in range(2):
            d = np.zeros(2); d[k] = 1e-6
            J[:, k] = (cycle(s + d)[1] - (s + d) - g) / 1e-6
        s = s - np.linalg.solve(J, g)
    return T


def period_ideal(R, C, osc=Oscillator()):
    """Closed form for a lossless capacitor and tpd = 0."""
    return (R + osc.rout) * C * (math.log((osc.vdd - osc.vtn) / (osc.vdd - osc.vtp)) + math.log(osc.vtp / osc.vtn))


def probe_elements(lm, pr, eps_b, sigma_b):
    L = pr.length
    return EPS0 * lm.F0 * L, EPS0 * lm.Fc * L, EPS0 * lm.Fs * eps_b * L, lm.Fs * sigma_b * L


def frequency(R, lm, pr, theta, sigma_pore, t_c=25.0, osc=Oscillator(), porosity=0.42):
    eb = eps_crim(theta, t_c, porosity)
    sb = sigma_bulk(eb, sigma_pore, t_c)
    C0, Cc, Cs, Gs = probe_elements(lm, pr, eb, sb)
    return 1.0 / period(R, C0, Cc, Cs, Gs, osc)


def apparent_c(f, R, osc=Oscillator()):
    """Capacitance that a lossless probe would need to oscillate at f (what a naive reading assumes)."""
    k = math.log((osc.vdd - osc.vtn) / (osc.vdd - osc.vtp)) + math.log(osc.vtp / osc.vtn)
    return (1 / f - 2 * osc.tpd) / ((R + osc.rout) * k) - osc.c_par
