#!/usr/bin/env python3
"""Generate runtime_benchmark.ipynb from cell strings."""
import json

def code_cell(source):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": source}

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


# ─────────────────────────────────────────────────────────────────────────────
CELLS = []

# ── 0 · title ─────────────────────────────────────────────────────────────────
CELLS.append(md_cell("""\
# Empirical Runtime Comparison — KF / EKF / UKF / PF

Measures mean per-step wall time (µs) using `timeit.default_timer()`.

* **50 warm-up steps** (filter reaches steady state, not timed).
* **500 timed steps** (steady-state only) → mean and std-dev reported.
* Pre-generated trajectories and measurements; simulation cost excluded.

## Three filter scenarios
| Scenario | Dynamics | Measurements | Purpose |
|---|---|---|---|
| S1 Linear CV | constant-velocity | Cartesian (x, y) | KF optimal baseline |
| S2 Nonlinear CTRV | constant-turn-rate | range + bearing | EKF / UKF / PF doing real nonlinear work |
| S3 Non-differentiable | CTRV (same as S2) | range + bearing | EKF-FD (numerical Jacobian) vs UKF sigma points |

## Particle-count scaling
Runtime vs *N* for two state dimensions (1-D and 2-D), with horizontal
reference lines at KF / EKF / UKF cost so the crossover is visible.
"""))

# ── 1 · imports ───────────────────────────────────────────────────────────────
CELLS.append(code_cell("""\
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import timeit
import os

os.makedirs('figures', exist_ok=True)

FS = 20; FS_LEG = 16; FS_TICK = 18
plt.rcParams.update({
    'font.size': FS, 'axes.labelsize': FS, 'axes.titlesize': FS,
    'figure.titlesize': FS + 2, 'legend.fontsize': FS_LEG,
    'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK,
    'lines.linewidth': 2.0, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'figure.facecolor': 'white',
})

# Consistent with comparative_analysis.ipynb
COLORS = {
    'KF': '#1f77b4', 'EKF': '#ff7f0e', 'EKF-FD': '#8c564b',
    'UKF': '#2ca02c', 'PF': '#d62728',
}
MARKERS = {'KF': 'o', 'EKF': 's', 'EKF-FD': 'D', 'UKF': '^', 'PF': 'v'}

np.random.seed(0)   # global reproducibility seed (reset per scenario below)
"""))

# ── 2 · shared filter implementations ─────────────────────────────────────────
CELLS.append(code_cell("""\
# ── helper matrices ──────────────────────────────────────────────────────────
def make_F(dt):
    \"\"\"CV state-transition matrix. State order: [px, py, vx, vy].\"\"\"
    return np.array([[1, 0, dt,  0],
                     [0, 1,  0, dt],
                     [0, 0,  1,  0],
                     [0, 0,  0,  1]], dtype=float)

def make_Q(sigma_a2, dt):
    \"\"\"Discrete white-noise acceleration process-noise covariance.\"\"\"
    return sigma_a2 * np.array([
        [dt**4/4,       0, dt**3/2,       0],
        [      0, dt**4/4,       0, dt**3/2],
        [dt**3/2,       0,   dt**2,       0],
        [      0, dt**3/2,       0,   dt**2]], dtype=float)

H_CART = np.array([[1., 0., 0., 0.],
                   [0., 1., 0., 0.]])   # observe (px, py)

# ── Kalman filter ─────────────────────────────────────────────────────────────
def kf_predict(x, P, F, Q):
    return F @ x, F @ P @ F.T + Q

def kf_update(x_pred, P_pred, z, H, R):
    y   = z - H @ x_pred
    S   = H @ P_pred @ H.T + R
    K   = P_pred @ H.T @ np.linalg.inv(S)
    x_u = x_pred + K @ y
    P_u = (np.eye(len(x_pred)) - K @ H) @ P_pred
    P_u = 0.5 * (P_u + P_u.T)
    return x_u, P_u, float(y @ np.linalg.solve(S, y))

# ── EKF helpers ───────────────────────────────────────────────────────────────
def ekf_predict(x, P, F, Q):
    \"\"\"CV predict — Jacobian of F equals F for linear dynamics.\"\"\"
    return F @ x, F @ P @ F.T + Q

def h_rb(x):
    \"\"\"Range-bearing measurement function: [px,py,vx,vy] → [r, θ].\"\"\"
    return np.array([np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])])

def H_rb_jac(x):
    \"\"\"Analytical 2×4 Jacobian of h_rb.\"\"\"
    px, py = x[0], x[1]; r = np.hypot(px, py); r2 = r**2
    return np.array([[ px/r,  py/r, 0., 0.],
                     [-py/r2, px/r2, 0., 0.]])

def ekf_update_rb(x_pred, P_pred, z, R):
    H     = H_rb_jac(x_pred)
    inn   = z - h_rb(x_pred)
    inn[1] = (inn[1] + np.pi) % (2*np.pi) - np.pi
    S     = H @ P_pred @ H.T + R
    K     = P_pred @ H.T @ np.linalg.inv(S)
    x_u   = x_pred + K @ inn
    P_u   = 0.5 * ((np.eye(4) - K @ H) @ P_pred); P_u = P_u + P_u.T
    return x_u, P_u, float(inn @ np.linalg.inv(S) @ inn)

def ekf_update_cartesian(x_pred, P_pred, z, H, R):
    y   = z - H @ x_pred
    S   = H @ P_pred @ H.T + R
    K   = P_pred @ H.T @ np.linalg.inv(S)
    x_u = x_pred + K @ y
    P_u = 0.5 * ((np.eye(4) - K @ H) @ P_pred); P_u = P_u + P_u.T
    return x_u, P_u, float(y @ np.linalg.solve(S, y))

# ── CTRV dynamics ─────────────────────────────────────────────────────────────
def ctrv_f(x, dt, omega):
    px, py, vx, vy = x
    ct = np.cos(omega*dt); st = np.sin(omega*dt)
    return np.array([px + vx*dt, py + vy*dt, vx*ct - vy*st, vx*st + vy*ct])

def ctrv_F_jac(x, dt, omega):
    ct = np.cos(omega*dt); st = np.sin(omega*dt)
    return np.array([[1., 0., dt, 0.], [0., 1., 0., dt],
                     [0., 0., ct, -st], [0., 0., st,  ct]])

def ekf_predict_ctrv(x, P, Q, dt, omega):
    F = ctrv_F_jac(x, dt, omega)
    return ctrv_f(x, dt, omega), F @ P @ F.T + Q

# ── UKF helpers ───────────────────────────────────────────────────────────────
def ukf_weights(n, alpha=1e-3, beta=2.0, kappa=0.0):
    lam   = alpha**2 * (n + kappa) - n
    Wm    = np.full(2*n+1, 0.5/(n+lam)); Wc = Wm.copy()
    Wm[0] = lam/(n+lam); Wc[0] = lam/(n+lam) + (1 - alpha**2 + beta)
    return Wm, Wc, lam

def ukf_sigma_points(x, P, lam):
    n = len(x); L = np.linalg.cholesky((n + lam) * P)
    sig = np.empty((2*n+1, n)); sig[0] = x
    for i in range(n):
        sig[i+1] = x + L[:, i]; sig[n+i+1] = x - L[:, i]
    return sig

def ukf_predict(x, P, Q, F, Wm, Wc, lam):
    \"\"\"Linear-F predict via sigma points.\"\"\"
    sig = ukf_sigma_points(x, P, lam); sig_f = (F @ sig.T).T
    xp  = Wm @ sig_f; d = sig_f - xp
    Pp  = Q + (d * Wc[:, None]).T @ d
    return xp, Pp, sig_f

def ukf_predict_nl(x, P, Q, f_func, Wm, Wc, lam):
    \"\"\"Nonlinear-f predict via sigma points.\"\"\"
    sig   = ukf_sigma_points(x, P, lam)
    sig_f = np.array([f_func(s) for s in sig])
    xp    = Wm @ sig_f; d = sig_f - xp
    Pp    = Q + (d * Wc[:, None]).T @ d
    return xp, Pp, sig_f

def ukf_update_rb(xp, Pp, sig_f, z, R, Wm, Wc):
    \"\"\"Nonlinear range-bearing update.\"\"\"
    sig_h = np.array([h_rb(s) for s in sig_f]); zp = Wm @ sig_h
    dz = sig_h - zp; dx = sig_f - xp
    S   = R + (dz * Wc[:, None]).T @ dz
    Pxz = (dx * Wc[:, None]).T @ dz
    K   = Pxz @ np.linalg.inv(S)
    inn = z - zp; inn[1] = (inn[1] + np.pi) % (2*np.pi) - np.pi
    xu  = xp + K @ inn; Pu = 0.5 * (Pp - K @ S @ K.T); Pu = Pu + Pu.T
    return xu, Pu, float(inn @ np.linalg.inv(S) @ inn)

def ukf_update_cartesian(xp, Pp, sig_f, z, H, R, Wm, Wc):
    \"\"\"Linear Cartesian update.\"\"\"
    sig_h = (H @ sig_f.T).T; zp = Wm @ sig_h
    dz = sig_h - zp; dx = sig_f - xp
    S   = R + (dz * Wc[:, None]).T @ dz
    Pxz = (dx * Wc[:, None]).T @ dz
    K   = Pxz @ np.linalg.inv(S); inn = z - zp
    xu  = xp + K @ inn; Pu = 0.5 * (Pp - K @ S @ K.T); Pu = Pu + Pu.T
    return xu, Pu, float(inn @ np.linalg.inv(S) @ inn)

# ── Particle filter helpers ───────────────────────────────────────────────────
def _nw(w):
    s = w.sum(); return w / s if s > 1e-300 else np.ones(len(w)) / len(w)

def _ess(w):
    return 1.0 / np.sum(w**2)

def _sysresample(w):
    N = len(w)
    pos = (np.arange(N) + np.random.uniform(0, 1)) / N
    return np.searchsorted(np.cumsum(w), pos)

def pf_resample(particles, weights, thr=0.5):
    N = len(weights)
    if _ess(weights) < thr * N:
        particles = particles[_sysresample(weights)]
        weights   = np.ones(N) / N
    return particles, weights

def pf_init(N_p, x0_mean, x0_std):
    return (x0_mean + np.random.randn(N_p, 4) * x0_std,
            np.ones(N_p) / N_p)

def pf_predict_cv(particles, dt, sigma_a):
    N = len(particles)
    ax = np.random.randn(N) * sigma_a; ay = np.random.randn(N) * sigma_a
    p = particles.copy()
    p[:, 0] = particles[:, 0] + particles[:, 2]*dt + 0.5*ax*dt**2
    p[:, 1] = particles[:, 1] + particles[:, 3]*dt + 0.5*ay*dt**2
    p[:, 2] = particles[:, 2] + ax*dt
    p[:, 3] = particles[:, 3] + ay*dt
    return p

def pf_predict_ctrv(particles, dt, omega, sigma_a):
    N = len(particles)
    ax = np.random.randn(N) * sigma_a; ay = np.random.randn(N) * sigma_a
    p = particles.copy(); ct = np.cos(omega*dt); st = np.sin(omega*dt)
    p[:, 0] = particles[:, 0] + particles[:, 2]*dt
    p[:, 1] = particles[:, 1] + particles[:, 3]*dt
    p[:, 2] = particles[:, 2]*ct - particles[:, 3]*st + ax*dt
    p[:, 3] = particles[:, 2]*st + particles[:, 3]*ct + ay*dt
    return p

def pf_update_cartesian(particles, weights, z, R):
    inn = z - particles[:, :2]; Ri = np.linalg.inv(R)
    lw  = -0.5 * np.einsum('ni,ij,nj->n', inn, Ri, inn)
    return _nw(weights * np.exp(lw - lw.max()))

def pf_update_rb(particles, weights, z, R):
    rp = np.hypot(particles[:, 0], particles[:, 1])
    tp = np.arctan2(particles[:, 1], particles[:, 0])
    ir = z[0] - rp; it = (z[1] - tp + np.pi) % (2*np.pi) - np.pi
    lw = -0.5 * (ir**2/R[0,0] + it**2/R[1,1])
    return _nw(weights * np.exp(lw - lw.max()))

def pf_estimate(p, w): return w @ p
def pf_cov(p, w):
    m = w @ p; d = p - m; return (d * w[:, None]).T @ d

# ── KF linearised range-bearing helper ───────────────────────────────────────
def rb_to_cart_R(x_pred, R_rb):
    \"\"\"Propagate range-bearing R to Cartesian via Jacobian at x_pred.\"\"\"
    r  = max(np.hypot(x_pred[0], x_pred[1]), 1e-6)
    ct = x_pred[0]/r; st = x_pred[1]/r
    J  = np.array([[ct, -r*st], [st, r*ct]])
    return J @ R_rb @ J.T
"""))

# ── 3 · FD-EKF + 1-D PF variants ─────────────────────────────────────────────
CELLS.append(code_cell("""\
# ── Finite-difference Jacobian EKF ───────────────────────────────────────────
# Models the "non-differentiable" regime: Jacobians computed numerically.
# Central differences: 2*n evaluations per Jacobian (n=4 → 8 per matrix).

def fd_jacobian(f, x, eps=1e-5):
    \"\"\"Central-difference Jacobian of f at x.  Cost: 2*n evaluations.\"\"\"
    n = len(x)
    fx = f(x); m = len(fx)
    J  = np.zeros((m, n))
    for i in range(n):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        J[:, i] = (f(xp) - f(xm)) / (2.0 * eps)
    return J

def ekf_fd_predict_ctrv(x, P, Q, dt, omega):
    \"\"\"EKF predict using numerical Jacobian of CTRV f (no closed form used).\"\"\"
    f_func = lambda s: ctrv_f(s, dt, omega)
    F      = fd_jacobian(f_func, x)          # 2*4 = 8 ctrv_f calls
    return ctrv_f(x, dt, omega), F @ P @ F.T + Q

def ekf_fd_update_rb(x_pred, P_pred, z, R):
    \"\"\"EKF update using numerical Jacobian of range-bearing h.\"\"\"
    H   = fd_jacobian(h_rb, x_pred)          # 2*4 = 8 h_rb calls
    inn = z - h_rb(x_pred)
    inn[1] = (inn[1] + np.pi) % (2*np.pi) - np.pi
    S   = H @ P_pred @ H.T + R
    K   = P_pred @ H.T @ np.linalg.inv(S)
    x_u = x_pred + K @ inn
    P_u = 0.5 * ((np.eye(4) - K @ H) @ P_pred); P_u = P_u + P_u.T
    return x_u, P_u

# ── 1-D (px, vx) system for particle-count scaling ───────────────────────────
# 2-element state, 1-element measurement — minimises state-space overhead so
# PF particle count is the dominant variable.

F_1d = np.array([[1., 1.], [0., 1.]])       # CV, dt=1
H_1d = np.array([[1., 0.]])                 # observe px

def make_Q_1d(sigma_a2, dt=1.0):
    return sigma_a2 * np.array([[dt**4/4, dt**3/2],
                                [dt**3/2,   dt**2]])

def pf_init_1d(N_p, x0_mean, x0_std):
    \"\"\"Initialise particles for 1-D (px, vx) state.\"\"\"
    return (x0_mean + np.random.randn(N_p, 2) * x0_std,
            np.ones(N_p) / N_p)

def pf_predict_1d(particles, dt, sigma_a):
    N  = len(particles)
    ax = np.random.randn(N) * sigma_a
    p  = particles.copy()
    p[:, 0] += particles[:, 1]*dt + 0.5*ax*dt**2
    p[:, 1] += ax * dt
    return p

def pf_update_1d(particles, weights, z_px, r_var):
    \"\"\"Gaussian likelihood on scalar px measurement.\"\"\"
    inn = z_px - particles[:, 0]
    lw  = -0.5 * inn**2 / r_var
    return _nw(weights * np.exp(lw - lw.max()))

def pf_resample_1d(particles, weights, thr=0.5):
    N = len(weights)
    if _ess(weights) < thr * N:
        particles = particles[_sysresample(weights)]
        weights   = np.ones(N) / N
    return particles, weights
"""))

# ── 4 · timing utility ────────────────────────────────────────────────────────
CELLS.append(code_cell("""\
# Benchmark parameters
N_WARMUP = 50    # steady-state warm-up steps (not timed)
N_BENCH  = 500   # timed steps (averaged for mean cost, std for error bars)

def run_and_time(init_fn, step_fn, measurements,
                 n_warmup=N_WARMUP, n_bench=N_BENCH):
    \"\"\"
    Run a filter for n_warmup + n_bench steps and return per-step
    wall times (µs) for the n_bench steady-state steps.

    Parameters
    ----------
    init_fn  : callable() → state
        Returns the initial filter state (called once).
    step_fn  : callable(state, z) → state
        Performs one predict-update cycle and returns the new state.
    measurements : array-like, shape (n_warmup+n_bench, ...)
        Pre-generated observations; no simulation cost in timing.
    \"\"\"
    state = init_fn()

    # ── warm-up: advance filter into steady state, don't time ─────────────
    for k in range(n_warmup):
        state = step_fn(state, measurements[k])

    # ── timed section: one wall-clock measurement per step ─────────────────
    times = np.empty(n_bench)
    for k in range(n_bench):
        t0 = timeit.default_timer()
        state = step_fn(state, measurements[n_warmup + k])
        times[k] = (timeit.default_timer() - t0) * 1e6   # → µs

    return times   # shape (n_bench,)

def summarise(name, times):
    print(f'  {name:8s}  mean={times.mean():7.2f} µs  '
          f'std={times.std():6.2f} µs  '
          f'[{times.min():.1f}, {times.max():.1f}]')
"""))

# ── 5 · Scenario 1: Linear CV ─────────────────────────────────────────────────
CELLS.append(code_cell("""\
# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — Linear Constant-Velocity  (KF optimal baseline)
# All four filters applied to a linear system; EKF ≡ KF here.
# UKF adds sigma-point overhead on top of equivalent math.
# ─────────────────────────────────────────────────────────────────────────────
DT       = 1.0
SIGMA_A2 = 0.1
SIGMA_P  = 2.0
N_PF     = 2000

F1, Q1   = make_F(DT), make_Q(SIGMA_A2, DT)
R1       = np.diag([SIGMA_P**2] * 2)
Wm1, Wc1, lam1 = ukf_weights(4)

N_TOT = N_WARMUP + N_BENCH
np.random.seed(42)
x0_true1 = np.array([0., 0., 2., 1.])
true1 = np.zeros((N_TOT, 4)); true1[0] = x0_true1
for k in range(1, N_TOT):
    true1[k] = F1 @ true1[k-1]
z1 = true1[:, :2] + np.random.randn(N_TOT, 2) * SIGMA_P

x0_1   = np.array([0.5, 0.5, 1.5, 0.8])
P0_1   = np.diag([10.**2, 10.**2, 5.**2, 5.**2])
std_pf1 = np.array([10., 10., 5., 5.])

times1 = {}

# KF ─────────────────────────────────────────────────────────────────────────
def kf1_init(): return (x0_1.copy(), P0_1.copy())
def kf1_step(s, z):
    x, P = s; x, P = kf_predict(x, P, F1, Q1)
    x, P, _ = kf_update(x, P, z, H_CART, R1); return (x, P)
times1['KF'] = run_and_time(kf1_init, kf1_step, z1)

# EKF (same as KF on linear system) ──────────────────────────────────────────
def ekf1_init(): return (x0_1.copy(), P0_1.copy())
def ekf1_step(s, z):
    x, P = s; x, P = ekf_predict(x, P, F1, Q1)
    x, P, _ = ekf_update_cartesian(x, P, z, H_CART, R1); return (x, P)
times1['EKF'] = run_and_time(ekf1_init, ekf1_step, z1)

# UKF ─────────────────────────────────────────────────────────────────────────
def ukf1_init(): return (x0_1.copy(), P0_1.copy())
def ukf1_step(s, z):
    x, P = s
    x, P, sf = ukf_predict(x, P, Q1, F1, Wm1, Wc1, lam1)
    x, P, _  = ukf_update_cartesian(x, P, sf, z, H_CART, R1, Wm1, Wc1)
    return (x, P)
times1['UKF'] = run_and_time(ukf1_init, ukf1_step, z1)

# PF ─────────────────────────────────────────────────────────────────────────
def pf1_init():
    np.random.seed(142)
    return pf_init(N_PF, x0_1, std_pf1)
def pf1_step(s, z):
    p, w = s
    p = pf_predict_cv(p, DT, np.sqrt(SIGMA_A2))
    w = pf_update_cartesian(p, w, z, R1)
    p, w = pf_resample(p, w)
    return (p, w)
times1['PF'] = run_and_time(pf1_init, pf1_step, z1)

print('Scenario 1 — Linear CV:')
for f, t in times1.items():
    summarise(f, t)
"""))

# ── 6 · Scenario 2: Nonlinear CTRV ───────────────────────────────────────────
CELLS.append(code_cell("""\
# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — Nonlinear CTRV + range-bearing measurements
# EKF / UKF / PF all performing genuine nonlinear approximation.
# ─────────────────────────────────────────────────────────────────────────────
OMEGA2   = 0.05
SIGMA_R  = 15.0
SIGMA_TH = np.deg2rad(2.0)
F2, Q2   = make_F(DT), make_Q(SIGMA_A2, DT)
R2       = np.diag([SIGMA_R**2, SIGMA_TH**2])
Wm2, Wc2, lam2 = ukf_weights(4)
ctrv_s2 = lambda s: ctrv_f(s, DT, OMEGA2)

np.random.seed(99)
x0_true2 = np.array([500., 0., 0., 10.])
true2 = np.zeros((N_TOT, 4)); true2[0] = x0_true2
for k in range(1, N_TOT):
    px, py, vx, vy = true2[k-1]
    ct = np.cos(OMEGA2*DT); st = np.sin(OMEGA2*DT)
    true2[k] = [px+vx*DT, py+vy*DT, vx*ct-vy*st, vx*st+vy*ct]
z2 = np.zeros((N_TOT, 2))
z2[:, 0] = np.hypot(true2[:, 0], true2[:, 1]) + np.random.randn(N_TOT)*SIGMA_R
z2[:, 1] = np.arctan2(true2[:, 1], true2[:, 0]) + np.random.randn(N_TOT)*SIGMA_TH

x0_2    = np.array([500., 0.5, 0.5, 9.5])
P0_2    = np.diag([SIGMA_R**2, SIGMA_R**2, 5.**2, 5.**2])
std_pf2 = np.array([SIGMA_R, SIGMA_R, 5., 5.])

times2 = {}

# KF (linearised measurement covariance via Jacobian) ─────────────────────────
def kf2_init(): return (x0_2.copy(), P0_2.copy())
def kf2_step(s, z):
    x, P = s; x, P = kf_predict(x, P, F2, Q2)
    Rc = rb_to_cart_R(x, R2)
    zc = np.array([z[0]*np.cos(z[1]), z[0]*np.sin(z[1])])
    x, P, _ = kf_update(x, P, zc, H_CART, Rc); return (x, P)
times2['KF'] = run_and_time(kf2_init, kf2_step, z2)

# EKF (analytical Jacobians for CTRV + range-bearing) ─────────────────────────
def ekf2_init(): return (x0_2.copy(), P0_2.copy())
def ekf2_step(s, z):
    x, P = s; x, P = ekf_predict_ctrv(x, P, Q2, DT, OMEGA2)
    x, P, _ = ekf_update_rb(x, P, z, R2); return (x, P)
times2['EKF'] = run_and_time(ekf2_init, ekf2_step, z2)

# UKF (sigma points for CTRV predict + range-bearing update) ──────────────────
def ukf2_init(): return (x0_2.copy(), P0_2.copy())
def ukf2_step(s, z):
    x, P = s
    x, P, sf = ukf_predict_nl(x, P, Q2, ctrv_s2, Wm2, Wc2, lam2)
    x, P, _  = ukf_update_rb(x, P, sf, z, R2, Wm2, Wc2); return (x, P)
times2['UKF'] = run_and_time(ukf2_init, ukf2_step, z2)

# PF ─────────────────────────────────────────────────────────────────────────
def pf2_init():
    np.random.seed(299)
    return pf_init(N_PF, x0_2, std_pf2)
def pf2_step(s, z):
    p, w = s
    p = pf_predict_ctrv(p, DT, OMEGA2, np.sqrt(SIGMA_A2))
    w = pf_update_rb(p, w, z, R2)
    p, w = pf_resample(p, w); return (p, w)
times2['PF'] = run_and_time(pf2_init, pf2_step, z2)

print('Scenario 2 — Nonlinear CTRV:')
for f, t in times2.items():
    summarise(f, t)
"""))

# ── 7 · Scenario 3: FD-EKF vs UKF sigma points ───────────────────────────────
CELLS.append(code_cell("""\
# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3 — Non-differentiable (finite-difference Jacobian regime)
# Same CTRV + range-bearing system as S2, but EKF-FD replaces the analytical
# Jacobian with central differences (2*4 = 8 function evals per Jacobian).
# Comparison: EKF (analytical) vs EKF-FD (numerical) vs UKF (sigma points).
# ─────────────────────────────────────────────────────────────────────────────

# Reuse S2 trajectories and parameters (same system, different filter impls)
times3 = {}

# EKF (analytical Jacobian) — reference
times3['EKF']    = times2['EKF'].copy()

# EKF-FD: numerical Jacobian via central differences ──────────────────────────
def ekffd_init(): return (x0_2.copy(), P0_2.copy())
def ekffd_step(s, z):
    x, P = s
    x, P = ekf_fd_predict_ctrv(x, P, Q2, DT, OMEGA2)   # 8 ctrv_f calls
    x, P = ekf_fd_update_rb(x, P, z, R2)               # 8 h_rb calls
    return (x, P)
times3['EKF-FD'] = run_and_time(ekffd_init, ekffd_step, z2)

# UKF (sigma points) ──────────────────────────────────────────────────────────
times3['UKF']    = times2['UKF'].copy()

# PF (for completeness) ───────────────────────────────────────────────────────
times3['PF']     = times2['PF'].copy()

print('Scenario 3 — Non-differentiable (FD-EKF comparison):')
for f, t in times3.items():
    summarise(f, t)

print()
print(f'EKF-FD overhead vs EKF (analytical):  '
      f'{times3[\"EKF-FD\"].mean() / times3[\"EKF\"].mean():.2f}×')
print(f'EKF-FD overhead vs UKF (sigma points): '
      f'{times3[\"EKF-FD\"].mean() / times3[\"UKF\"].mean():.2f}×')
"""))

# ── 8 · Figure 1: scenario runtime comparison ────────────────────────────────
CELLS.append(code_cell("""\
# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Mean step time per scenario, with std-dev error bars
# ─────────────────────────────────────────────────────────────────────────────
scenarios = [
    ('S1: Linear CV\\n(KF optimal)',             times1, ['KF', 'EKF', 'UKF', 'PF']),
    ('S2: Nonlinear CTRV\\n(real nonlinear work)', times2, ['KF', 'EKF', 'UKF', 'PF']),
    ('S3: Non-differentiable\\n(FD-EKF vs sigma pts)', times3, ['EKF', 'EKF-FD', 'UKF', 'PF']),
]

fig, axes = plt.subplots(1, 3, figsize=(22, 8), sharey=False)
fig.suptitle('Empirical Per-Step Runtime: KF / EKF / EKF-FD / UKF / PF',
             fontweight='bold')

for ax, (title, times_d, filters) in zip(axes, scenarios):
    x_pos  = np.arange(len(filters))
    means  = np.array([times_d[f].mean() for f in filters])
    stds   = np.array([times_d[f].std()  for f in filters])
    colors = [COLORS[f] for f in filters]

    bars = ax.bar(x_pos, means, yerr=stds, capsize=6,
                  color=colors, edgecolor='k', linewidth=0.8,
                  error_kw=dict(elinewidth=1.8, capthick=1.8))

    # Label each bar with mean ± std
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + s + 0.5,
                f'{m:.0f}\\n±{s:.0f}',
                ha='center', va='bottom', fontsize=11, linespacing=1.3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(filters, fontsize=FS_TICK)
    ax.set_ylabel('Mean step time (µs)', fontsize=FS)
    ax.set_title(title, fontweight='bold', fontsize=FS)
    ax.set_ylim(0, (means + stds).max() * 1.45)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/runtime_comparison.png')
plt.show()
print('Saved → figures/runtime_comparison.png')
"""))

# ── 9 · Particle-count scaling sweep ─────────────────────────────────────────
CELLS.append(code_cell("""\
# ─────────────────────────────────────────────────────────────────────────────
# Particle-count scaling: runtime vs N_particles
# Two state-space dimensions: 1-D (px, vx) and 2-D (px, py, vx, vy).
# Reference lines at KF / EKF / UKF cost show the crossover point.
# ─────────────────────────────────────────────────────────────────────────────

N_LIST = [50, 100, 200, 500, 1000, 2000, 5000]
N_BENCH_PF = 300   # fewer steps to keep total time reasonable

SIGMA_A2_SC = 0.1
SIGMA_POS_SC = 2.0

# ── 1-D reference times (KF, EKF, UKF for 2-state system) ─────────────────
Q1d = make_Q_1d(SIGMA_A2_SC)
R1d = np.array([[SIGMA_POS_SC**2]])
Wm1d, Wc1d, lam1d = ukf_weights(2)

N_TOT_SC = N_WARMUP + N_BENCH_PF
np.random.seed(7)
true1d = np.zeros((N_TOT_SC, 2)); true1d[0] = [0., 2.]
for k in range(1, N_TOT_SC):
    true1d[k] = F_1d @ true1d[k-1]
z1d = true1d[:, :1] + np.random.randn(N_TOT_SC, 1) * SIGMA_POS_SC

x0_1d = np.array([0.5, 1.5]); P0_1d = np.diag([10.**2, 5.**2])

def kf1d_init(): return (x0_1d.copy(), P0_1d.copy())
def kf1d_step(s, z):
    x, P = s; x, P = kf_predict(x, P, F_1d, Q1d)
    x, P, _ = kf_update(x, P, z, H_1d, R1d); return (x, P)
ref1d_kf = run_and_time(kf1d_init, kf1d_step, z1d, n_bench=N_BENCH_PF)

def ekf1d_init(): return (x0_1d.copy(), P0_1d.copy())
def ekf1d_step(s, z):
    x, P = s; x, P = kf_predict(x, P, F_1d, Q1d)   # linear ≡ EKF
    x, P, _ = kf_update(x, P, z, H_1d, R1d); return (x, P)
ref1d_ekf = run_and_time(ekf1d_init, ekf1d_step, z1d, n_bench=N_BENCH_PF)

def ukf1d_init(): return (x0_1d.copy(), P0_1d.copy())
def ukf1d_step(s, z):
    x, P = s
    x, P, sf = ukf_predict(x, P, Q1d, F_1d, Wm1d, Wc1d, lam1d)
    x, P, _  = ukf_update_cartesian(x, P, sf, z, H_1d, R1d, Wm1d, Wc1d)
    return (x, P)
ref1d_ukf = run_and_time(ukf1d_init, ukf1d_step, z1d, n_bench=N_BENCH_PF)

ref1d = {'KF': ref1d_kf, 'EKF': ref1d_ekf, 'UKF': ref1d_ukf}

# ── 2-D reference times (KF, EKF, UKF for 4-state system, same as S1) ──────
ref2d = {'KF': times1['KF'], 'EKF': times1['EKF'], 'UKF': times1['UKF']}

# ── generate trajectories for particle sweep ──────────────────────────────
# 1-D: reuse true1d / z1d above
# 2-D: reuse true1 / z1 from Scenario 1 (same CV system)

print('Sweeping N_particles ...')

pf_scaling = {}
for dim_label, true_arr, z_arr, x0_pf, P0_pf, std_pf, \
        predict_fn, update_fn, init_fn_kw in [
    ('1D',
     true1d, z1d,
     x0_1d, P0_1d, np.array([10., 5.]),
     lambda p, N: pf_predict_1d(p, 1.0, np.sqrt(SIGMA_A2_SC)),
     lambda p, w, z: pf_update_1d(p, w, z[0], SIGMA_POS_SC**2),
     '1d'),
    ('2D',
     true1, z1,
     x0_1, P0_1, std_pf1,
     lambda p, N: pf_predict_cv(p, DT, np.sqrt(SIGMA_A2_SC)),
     lambda p, w, z: pf_update_cartesian(p, w, z, R1),
     '2d'),
]:
    pf_scaling[dim_label] = {}
    for N in N_LIST:
        if dim_label == '1D':
            def _init(N=N):
                np.random.seed(500 + N)
                return pf_init_1d(N, x0_pf, std_pf)
            def _step(s, z, N=N):
                p, w = s
                p = pf_predict_1d(p, 1.0, np.sqrt(SIGMA_A2_SC))
                w = pf_update_1d(p, w, z[0], SIGMA_POS_SC**2)
                p, w = pf_resample_1d(p, w)
                return (p, w)
        else:
            def _init(N=N):
                np.random.seed(600 + N)
                return pf_init(N, x0_pf, std_pf)
            def _step(s, z, N=N):
                p, w = s
                p = pf_predict_cv(p, DT, np.sqrt(SIGMA_A2_SC))
                w = pf_update_cartesian(p, w, z, R1)
                p, w = pf_resample(p, w)
                return (p, w)
        t = run_and_time(_init, _step, z_arr, n_bench=N_BENCH_PF)
        pf_scaling[dim_label][N] = t
        print(f'  PF-{dim_label}  N={N:5d}  mean={t.mean():.1f} µs  '
              f'std={t.std():.1f} µs')

print('Sweep complete.')
"""))

# ── 10 · Figure 2: particle scaling ──────────────────────────────────────────
CELLS.append(code_cell("""\
# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — PF runtime vs N_particles with reference lines (KF / EKF / UKF)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=False)
fig.suptitle(
    'Particle Filter Runtime vs Number of Particles  —  '
    'Horizontal lines: KF / EKF / UKF reference cost  (\u00b11\u03c3 shaded)',
    fontweight='bold')

configs = [
    ('1D state  $(p_x, v_x)$',  '1D', ref1d),
    ('2D state  $(p_x, p_y, v_x, v_y)$', '2D', ref2d),
]

for ax, (title, dim_label, refs) in zip(axes, configs):
    N_arr = np.array(N_LIST)
    means = np.array([pf_scaling[dim_label][N].mean() for N in N_LIST])
    stds  = np.array([pf_scaling[dim_label][N].std()  for N in N_LIST])

    # PF line + shaded ±1σ band (resampling variance is interesting!)
    ax.plot(N_arr, means, 'o-', color=COLORS['PF'], lw=2.5,
            ms=8, label=f'PF  (±1σ shaded)', zorder=4)
    ax.fill_between(N_arr, means - stds, means + stds,
                    color=COLORS['PF'], alpha=0.20, zorder=3)
    ax.errorbar(N_arr, means, yerr=stds, fmt='none',
                ecolor=COLORS['PF'], elinewidth=1.5, capsize=5, zorder=5)

    # Reference horizontal lines ─────────────────────────────────────────────
    ref_styles = [
        ('KF',  '--',  1.8),
        ('EKF', ':',   2.0),
        ('UKF', '-.',  2.0),
    ]
    for fname, ls, lw in ref_styles:
        m = refs[fname].mean(); s = refs[fname].std()
        ax.axhline(m, color=COLORS[fname], ls=ls, lw=lw,
                   label=f'{fname}  {m:.0f} µs', zorder=2)
        ax.axhspan(m - s, m + s, color=COLORS[fname], alpha=0.07, zorder=1)

    # Annotate crossover with UKF ─────────────────────────────────────────────
    ukf_mean = refs['UKF'].mean()
    crossover_candidates = np.where(means >= ukf_mean)[0]
    if len(crossover_candidates):
        ci = crossover_candidates[0]
        ax.axvline(N_arr[ci], color='grey', ls='--', lw=1.4, alpha=0.7)
        ax.text(N_arr[ci] * 1.05, ukf_mean * 1.05,
                f'PF ≈ UKF\\nat N={N_arr[ci]}',
                fontsize=13, color='grey', va='bottom')

    ax.set_xlabel('Number of particles', fontsize=FS)
    ax.set_ylabel('Mean step time (µs)', fontsize=FS)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks(N_LIST); ax.set_xticklabels(N_LIST, fontsize=FS_TICK - 2)
    ax.set_title(title, fontweight='bold', fontsize=FS)
    ax.legend(fontsize=FS_LEG - 2, loc='upper left')
    ax.grid(True, which='both', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/runtime_vs_particles.png')
plt.show()
print('Saved → figures/runtime_vs_particles.png')
"""))


# ─────────────────────────────────────────────────────────────────────────────
# Assemble notebook
# ─────────────────────────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": CELLS
}

with open('runtime_benchmark.ipynb', 'w') as fh:
    json.dump(notebook, fh, indent=1, ensure_ascii=False)

print(f'Written runtime_benchmark.ipynb  ({len(CELLS)} cells)')
