#!/usr/bin/env python3
"""
comparative_analysis.py
Comparative Analysis: Kalman Filter (KF), Extended KF (EKF),
Unscented KF (UKF), Particle Filter (PF).

Sources (extracted verbatim, adapted to shared [px, py, vx, vy] state convention):
  KF  — kalman_filter.ipynb
  EKF — ekf_experimentation.ipynb
  UKF — ukf_experimentation.ipynb
  PF  — pf_experimentation.ipynb

Scenarios
---------
1. Linear CV, Gaussian noise       → state estimates, RMSE bar+ts, NIS
2. Nonlinear range-bearing, CTRV   → state estimates, NEES divergence, MC RMSE
3. Non-Gaussian (Student-t ν=3)    → RMSE bar+ts, NEES consistency breakdown
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import chi2, t as t_dist
import os

os.makedirs('figures', exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.8,
    'figure.facecolor': 'white',
})

COLORS = {'KF': '#1f77b4', 'EKF': '#ff7f0e', 'UKF': '#2ca02c', 'PF': '#d62728'}
LSTYLE = {'KF': '-',       'EKF': '--',       'UKF': '-.',       'PF': ':'}
FILTERS = ['KF', 'EKF', 'UKF', 'PF']

def legend_handles(extra=None):
    h = [Line2D([0],[0], color=COLORS[f], ls=LSTYLE[f], lw=2, label=f) for f in FILTERS]
    return (extra or []) + h

# ==============================================================================
#  SECTION 0 — SHARED DYNAMICS
# ==============================================================================

def make_F(dt):
    """CV state-transition matrix. State: [px, py, vx, vy]."""
    return np.array([[1, 0, dt,  0],
                     [0, 1,  0, dt],
                     [0, 0,  1,  0],
                     [0, 0,  0,  1]])

def make_Q(sigma_a2, dt):
    """Discrete white-noise acceleration process-noise covariance."""
    return sigma_a2 * np.array([
        [dt**4/4,       0, dt**3/2,       0],
        [      0, dt**4/4,       0, dt**3/2],
        [dt**3/2,       0,   dt**2,       0],
        [      0, dt**3/2,       0,   dt**2]])

H_CART = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0]])   # observe (px, py)

# ==============================================================================
#  SECTION 1 — KF   (kalman_filter.ipynb)
# ==============================================================================

def kf_predict(x, P, F, Q):
    return F @ x, F @ P @ F.T + Q

def kf_update(x_pred, P_pred, z, H, R):
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_upd = x_pred + K @ y
    P_upd = (np.eye(len(x_pred)) - K @ H) @ P_pred
    P_upd = 0.5 * (P_upd + P_upd.T)
    nis = float(y @ np.linalg.solve(S, y))
    return x_upd, P_upd, nis

# ==============================================================================
#  SECTION 2 — EKF   (ekf_experimentation.ipynb)
# ==============================================================================

def ekf_predict(x, P, F, Q):
    """CV predict — Jacobian of F equals F exactly for linear dynamics."""
    return F @ x, F @ P @ F.T + Q

def h_rb(x):
    """Range-bearing from state [px, py, vx, vy]."""
    return np.array([np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])])

def H_rb_jac(x):
    """2×4 Jacobian of h_rb w.r.t. [px, py, vx, vy]."""
    px, py = x[0], x[1]
    r  = np.hypot(px, py)
    r2 = r**2
    return np.array([[ px/r,  py/r, 0, 0],
                     [-py/r2, px/r2, 0, 0]])

def ekf_update_rb(x_pred, P_pred, z, R):
    H     = H_rb_jac(x_pred)
    innov = z - h_rb(x_pred)
    innov[1] = (innov[1] + np.pi) % (2*np.pi) - np.pi
    S     = H @ P_pred @ H.T + R
    K     = P_pred @ H.T @ np.linalg.inv(S)
    x_upd = x_pred + K @ innov
    P_upd = 0.5 * ((np.eye(4) - K @ H) @ P_pred)
    P_upd = P_upd + P_upd.T
    nis   = float(innov @ np.linalg.inv(S) @ innov)
    return x_upd, P_upd, nis

def ekf_update_cartesian(x_pred, P_pred, z, H, R):
    """EKF with linear Cartesian h — identical to KF update."""
    y     = z - H @ x_pred
    S     = H @ P_pred @ H.T + R
    K     = P_pred @ H.T @ np.linalg.inv(S)
    x_upd = x_pred + K @ y
    P_upd = 0.5 * ((np.eye(4) - K @ H) @ P_pred)
    P_upd = P_upd + P_upd.T
    nis   = float(y @ np.linalg.solve(S, y))
    return x_upd, P_upd, nis

# ==============================================================================
#  SECTION 3 — UKF   (ukf_experimentation.ipynb)
# ==============================================================================

def ukf_weights(n, alpha=1e-3, beta=2.0, kappa=0.0):
    lam   = alpha**2 * (n + kappa) - n
    Wm    = np.full(2*n + 1, 0.5 / (n + lam))
    Wc    = Wm.copy()
    Wm[0] = lam / (n + lam)
    Wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)
    return Wm, Wc, lam

def ukf_sigma_points(x, P, lam):
    n   = len(x)
    L   = np.linalg.cholesky((n + lam) * P)
    sig = np.empty((2*n + 1, n))
    sig[0] = x
    for i in range(n):
        sig[i + 1]     = x + L[:, i]
        sig[n + i + 1] = x - L[:, i]
    return sig

def ukf_predict(x, P, Q, F, Wm, Wc, lam):
    """UKF predict with linear CV dynamics (sigma points through F)."""
    sig   = ukf_sigma_points(x, P, lam)
    sig_f = (F @ sig.T).T
    x_pred = Wm @ sig_f
    diffs  = sig_f - x_pred
    P_pred = Q + (diffs * Wc[:, None]).T @ diffs
    return x_pred, P_pred, sig_f

def ukf_update_rb(x_pred, P_pred, sig_f, z, R, Wm, Wc):
    """UKF update with nonlinear range-bearing h (no Jacobian needed)."""
    sig_h  = np.array([h_rb(s) for s in sig_f])
    z_pred = Wm @ sig_h
    dz     = sig_h - z_pred
    dx     = sig_f  - x_pred
    S      = R   + (dz * Wc[:, None]).T @ dz
    Pxz    =       (dx * Wc[:, None]).T @ dz
    K      = Pxz @ np.linalg.inv(S)
    innov  = z - z_pred
    innov[1] = (innov[1] + np.pi) % (2*np.pi) - np.pi
    x_upd  = x_pred + K @ innov
    P_upd  = 0.5 * (P_pred - K @ S @ K.T)
    P_upd  = P_upd + P_upd.T
    nis    = float(innov @ np.linalg.inv(S) @ innov)
    return x_upd, P_upd, nis

def ukf_update_cartesian(x_pred, P_pred, sig_f, z, H, R, Wm, Wc):
    """UKF update with linear Cartesian h."""
    sig_h  = (H @ sig_f.T).T
    z_pred = Wm @ sig_h
    dz     = sig_h - z_pred
    dx     = sig_f  - x_pred
    S      = R   + (dz * Wc[:, None]).T @ dz
    Pxz    =       (dx * Wc[:, None]).T @ dz
    K      = Pxz @ np.linalg.inv(S)
    innov  = z - z_pred
    x_upd  = x_pred + K @ innov
    P_upd  = 0.5 * (P_pred - K @ S @ K.T)
    P_upd  = P_upd + P_upd.T
    nis    = float(innov @ np.linalg.inv(S) @ innov)
    return x_upd, P_upd, nis

# ==============================================================================
#  SECTION 4 — PF   (pf_experimentation.ipynb)
# ==============================================================================

def pf_init(N_p, x0_mean, x0_std):
    particles = x0_mean + np.random.randn(N_p, 4) * x0_std
    weights   = np.ones(N_p) / N_p
    return particles, weights

def pf_predict_cv(particles, dt, sigma_a):
    N  = len(particles)
    ax = np.random.randn(N) * sigma_a
    ay = np.random.randn(N) * sigma_a
    p  = particles.copy()
    p[:, 0] = particles[:, 0] + particles[:, 2]*dt + 0.5*ax*dt**2
    p[:, 1] = particles[:, 1] + particles[:, 3]*dt + 0.5*ay*dt**2
    p[:, 2] = particles[:, 2] + ax*dt
    p[:, 3] = particles[:, 3] + ay*dt
    return p

def _norm_weights(weights):
    s = weights.sum()
    if s < 1e-300:
        return np.ones(len(weights)) / len(weights)
    return weights / s

def pf_update_cartesian(particles, weights, z, R):
    innov = z - particles[:, :2]
    Rinv  = np.linalg.inv(R)
    log_w = -0.5 * np.einsum('ni,ij,nj->n', innov, Rinv, innov)
    log_w -= log_w.max()
    return _norm_weights(weights * np.exp(log_w))

def pf_update_rb(particles, weights, z, R):
    r_p  = np.hypot(particles[:, 0], particles[:, 1])
    th_p = np.arctan2(particles[:, 1], particles[:, 0])
    ir   = z[0] - r_p
    ith  = (z[1] - th_p + np.pi) % (2*np.pi) - np.pi
    log_w = -0.5 * (ir**2/R[0,0] + ith**2/R[1,1])
    log_w -= log_w.max()
    return _norm_weights(weights * np.exp(log_w))

def pf_update_student_t(particles, weights, z, R, nu=3):
    """Student-t likelihood — robust to measurement outliers."""
    innov  = z - particles[:, :2]
    Rinv   = np.linalg.inv(R)
    maha2  = np.einsum('ni,ij,nj->n', innov, Rinv, innov)
    log_w  = -0.5 * (nu + 2) * np.log(1.0 + maha2 / nu)
    log_w -= log_w.max()
    return _norm_weights(weights * np.exp(log_w))

def pf_nis_cartesian(particles, weights, z, R):
    z_mean = weights @ particles[:, :2]
    dz     = particles[:, :2] - z_mean
    S      = R + (dz * weights[:, None]).T @ dz
    innov  = z - z_mean
    return float(innov @ np.linalg.inv(S) @ innov)

def pf_nis_rb(particles, weights, z, R):
    """NIS for PF with range-bearing measurement (pf_experimentation.ipynb)."""
    r_p  = np.hypot(particles[:, 0], particles[:, 1])
    th_p = np.arctan2(particles[:, 1], particles[:, 0])
    zp   = np.column_stack([r_p, th_p])
    zm   = weights @ zp
    dz   = zp - zm
    dz[:, 1] = (dz[:, 1] + np.pi) % (2*np.pi) - np.pi
    S    = R + (dz * weights[:, None]).T @ dz
    inn  = z - zm
    inn[1] = (inn[1] + np.pi) % (2*np.pi) - np.pi
    return float(inn @ np.linalg.inv(S) @ inn)

def ess(weights):
    return 1.0 / np.sum(weights**2)

def systematic_resample(weights):
    N   = len(weights)
    pos = (np.arange(N) + np.random.uniform(0, 1)) / N
    return np.searchsorted(np.cumsum(weights), pos)

def pf_resample(particles, weights, thr=0.5):
    N = len(weights)
    if ess(weights) < thr * N:
        idx       = systematic_resample(weights)
        particles = particles[idx]
        weights   = np.ones(N) / N
    return particles, weights

def pf_estimate(particles, weights):
    return weights @ particles

def pf_cov(particles, weights):
    mean  = weights @ particles
    diffs = particles - mean
    return (diffs * weights[:, None]).T @ diffs

# ==============================================================================
#  SECTION 5 — METRIC HELPERS
# ==============================================================================

def compute_nees(x_true, x_est, P):
    e = x_true - x_est
    try:
        return float(e @ np.linalg.solve(P, e))
    except np.linalg.LinAlgError:
        return np.nan

def nees_ts(ests, covs, truth):
    return np.array([compute_nees(truth[k], ests[k], covs[k]) for k in range(len(ests))])

def pos_err_ts(ests, truth):
    return np.sqrt((ests[:, 0]-truth[:, 0])**2 + (ests[:, 1]-truth[:, 1])**2)

chi2_lo2, chi2_hi2 = chi2.ppf(0.025, df=2), chi2.ppf(0.975, df=2)
chi2_lo4, chi2_hi4 = chi2.ppf(0.025, df=4), chi2.ppf(0.975, df=4)

def nis_band_plot(ax, t, nis_dict, df=2):
    lo = chi2.ppf(0.025, df); hi = chi2.ppf(0.975, df)
    for f in FILTERS:
        pct = np.mean((nis_dict[f] >= lo) & (nis_dict[f] <= hi)) * 100
        ax.plot(t, nis_dict[f], color=COLORS[f], ls=LSTYLE[f], lw=1.4,
                alpha=0.85, label=f'{f} ({pct:.0f}% in bounds)')
    ax.axhline(hi, color='red',    ls='--', lw=1.4, label=f'$\\chi^2_{df}$ 97.5% = {hi:.2f}')
    ax.axhline(lo, color='orange', ls='--', lw=1.4, label=f'$\\chi^2_{df}$ 2.5%  = {lo:.3f}')
    ax.axhline(float(df), color='grey', ls=':', lw=1.1, label=f'Expected mean = {df}')
    ax.fill_between(t, lo, hi, color='green', alpha=0.06, label='95% band')
    ax.set_ylim(0, max(15, np.nanpercentile(
        np.concatenate([nis_dict[f] for f in FILTERS]), 97)))
    ax.set_xlabel('Time step'); ax.set_ylabel('NIS')
    ax.legend(ncol=2, fontsize=9); ax.grid(True, alpha=0.3)

def nees_band_plot(ax, t, nees_dict, df=4, clip=60):
    lo = chi2.ppf(0.025, df); hi = chi2.ppf(0.975, df)
    for f in FILTERS:
        vals = np.clip(nees_dict[f], 0, clip)
        pct  = np.mean((nees_dict[f] >= lo) & (nees_dict[f] <= hi)) * 100
        ax.plot(t, vals, color=COLORS[f], ls=LSTYLE[f], lw=1.4,
                label=f'{f} ({pct:.0f}% in $\\chi^2_{df}$ bounds)')
    ax.axhline(hi, color='red',    ls='--', lw=1.4, label=f'$\\chi^2_{df}$ 97.5% = {hi:.2f}')
    ax.axhline(lo, color='orange', ls='--', lw=1.4)
    ax.axhline(float(df), color='grey', ls=':', lw=1.1, label=f'Expected mean = {df}')
    ax.fill_between(t, lo, hi, color='green', alpha=0.06, label='95% band')
    ax.set_ylim(0, clip * 1.05)
    ax.set_xlabel('Time step'); ax.set_ylabel('NEES')
    ax.legend(ncol=2, fontsize=9); ax.grid(True, alpha=0.3)

# ==============================================================================
#  SCENARIO 1 — Linear CV, Gaussian noise
# ==============================================================================
print("Running Scenario 1 — Linear CV, Gaussian noise...")

DT1        = 1.0
N1         = 150
SIGMA_A2_1 = 0.1
SIGMA_POS1 = 2.0
R1         = np.diag([SIGMA_POS1**2, SIGMA_POS1**2])
F1         = make_F(DT1)
Q1         = make_Q(SIGMA_A2_1, DT1)
N_PF       = 2000

x0_true_1 = np.array([0.0, 0.0,  2.0, 1.0])
x0_1      = np.array([0.5, 0.5,  1.5, 0.8])
P0_1      = np.diag([10.**2, 10.**2, 5.**2, 5.**2])

np.random.seed(42)
true_states_1      = np.zeros((N1, 4)); true_states_1[0] = x0_true_1
for k in range(1, N1):
    true_states_1[k] = F1 @ true_states_1[k-1]
z1 = true_states_1[:, :2] + np.random.randn(N1, 2) * SIGMA_POS1

Wm1, Wc1, lam1 = ukf_weights(4)
t1 = np.arange(N1)

# — KF
def run_s1_kf():
    x, P = x0_1.copy(), P0_1.copy()
    ests, covs, nis = [], [], []
    for k in range(N1):
        x, P = kf_predict(x, P, F1, Q1)
        x, P, ni = kf_update(x, P, z1[k], H_CART, R1)
        ests.append(x.copy()); covs.append(P.copy()); nis.append(ni)
    return np.array(ests), covs, np.array(nis)

# — EKF
def run_s1_ekf():
    x, P = x0_1.copy(), P0_1.copy()
    ests, covs, nis = [], [], []
    for k in range(N1):
        x, P = ekf_predict(x, P, F1, Q1)
        x, P, ni = ekf_update_cartesian(x, P, z1[k], H_CART, R1)
        ests.append(x.copy()); covs.append(P.copy()); nis.append(ni)
    return np.array(ests), covs, np.array(nis)

# — UKF
def run_s1_ukf():
    x, P = x0_1.copy(), P0_1.copy()
    ests, covs, nis = [], [], []
    for k in range(N1):
        x, P, sig_f = ukf_predict(x, P, Q1, F1, Wm1, Wc1, lam1)
        x, P, ni    = ukf_update_cartesian(x, P, sig_f, z1[k], H_CART, R1, Wm1, Wc1)
        ests.append(x.copy()); covs.append(P.copy()); nis.append(ni)
    return np.array(ests), covs, np.array(nis)

# — PF
def run_s1_pf(seed=42):
    np.random.seed(seed + 100)
    particles, weights = pf_init(N_PF, x0_1, np.array([10., 10., 5., 5.]))
    ests, covs, nis    = [], [], []
    sa = np.sqrt(SIGMA_A2_1)
    for k in range(N1):
        particles = pf_predict_cv(particles, DT1, sa)
        weights   = pf_update_cartesian(particles, weights, z1[k], R1)
        ni        = pf_nis_cartesian(particles, weights, z1[k], R1)
        particles, weights = pf_resample(particles, weights)
        ests.append(pf_estimate(particles, weights))
        covs.append(pf_cov(particles, weights))
        nis.append(ni)
    return np.array(ests), covs, np.array(nis)

est1 = {}; cov1 = {}; nis1 = {}
est1['KF'], cov1['KF'], nis1['KF'] = run_s1_kf()
est1['EKF'], cov1['EKF'], nis1['EKF'] = run_s1_ekf()
est1['UKF'], cov1['UKF'], nis1['UKF'] = run_s1_ukf()
est1['PF'],  cov1['PF'],  nis1['PF']  = run_s1_pf()
nees1 = {f: nees_ts(est1[f], cov1[f], true_states_1) for f in FILTERS}

# ── Fig S1-1: State estimates (px, py) ──────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for ax, dim, label in zip(axes, [0, 1], ['$p_x$ (m)', '$p_y$ (m)']):
    ax.plot(t1, true_states_1[:, dim], 'k-', lw=2.2, label='Ground truth', zorder=5)
    ax.scatter(t1, z1[:, dim], s=7, alpha=0.35, color='grey',
               label='Measurements', zorder=2)
    for f in FILTERS:
        ax.plot(t1, est1[f][:, dim], color=COLORS[f], ls=LSTYLE[f],
                lw=1.6, label=f, zorder=4)
    ax.set_ylabel(label); ax.grid(True, alpha=0.3)
    if dim == 0:
        handles = ([Line2D([0],[0],color='k',lw=2.2,label='Ground truth'),
                    Line2D([0],[0],marker='o',color='grey',ms=5,ls='',
                           alpha=0.6,label='Measurements')]
                   + legend_handles())
        ax.legend(handles=handles, ncol=3)
axes[-1].set_xlabel('Time step')
plt.suptitle('Scenario 1 — Linear CV: State Estimates vs Ground Truth',
             fontweight='bold')
plt.tight_layout()
plt.savefig('figures/s1_state_estimates.png')
plt.close()
print("  saved s1_state_estimates.png")

# ── Fig S1-2: RMSE (bar + time-series) ──────────────────────────────────────
overall1 = {f: np.sqrt(np.mean((est1[f][:,:2]-true_states_1[:,:2])**2))
            for f in FILTERS}
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
bars = ax.bar(FILTERS, [overall1[f] for f in FILTERS],
              color=[COLORS[f] for f in FILTERS], edgecolor='k', lw=0.8)
for bar, f in zip(bars, FILTERS):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f'{overall1[f]:.3f}', ha='center', va='bottom', fontsize=11)
ax.set_ylabel('RMSE (m)'); ax.set_title('Overall Position RMSE')
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0, max(overall1.values())*1.4)

ax = axes[1]
for f in FILTERS:
    ax.plot(t1, pos_err_ts(est1[f], true_states_1),
            color=COLORS[f], ls=LSTYLE[f], lw=1.6, label=f)
ax.set_xlabel('Time step'); ax.set_ylabel('Position error (m)')
ax.set_title('Position Error over Time')
ax.legend(); ax.grid(True, alpha=0.3)
plt.suptitle('Scenario 1 — Linear CV: RMSE Comparison', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/s1_rmse.png')
plt.close()
print("  saved s1_rmse.png")

# ── Fig S1-3: NIS (2×2 grid, df=2) ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, f in zip(axes.flatten(), FILTERS):
    ni  = nis1[f]
    pct = np.mean((ni >= chi2_lo2) & (ni <= chi2_hi2)) * 100
    ax.plot(t1, ni, color=COLORS[f], lw=1.3, alpha=0.85)
    ax.axhline(chi2_hi2, color='red',    ls='--', lw=1.3,
               label=f'$\\chi^2_2$ 97.5% = {chi2_hi2:.2f}')
    ax.axhline(chi2_lo2, color='orange', ls='--', lw=1.3,
               label=f'$\\chi^2_2$ 2.5%  = {chi2_lo2:.3f}')
    ax.axhline(2.0, color='grey', ls=':', lw=1.0, label='Expected = 2')
    ax.fill_between(t1, chi2_lo2, chi2_hi2, color='green', alpha=0.07)
    ax.set_title(f'{f}  —  {pct:.0f}% within 95% $\\chi^2_2$ bounds')
    ax.set_xlabel('Time step'); ax.set_ylabel('NIS')
    ax.set_ylim(0, max(15, float(np.percentile(ni, 97))))
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.suptitle('Scenario 1 — Linear CV: NIS with $\\chi^2_2$ Bounds',
             fontweight='bold')
plt.tight_layout()
plt.savefig('figures/s1_nis.png')
plt.close()
print("  saved s1_nis.png")

print("Scenario 1 complete.\n")

# ==============================================================================
#  SCENARIO 2 — Nonlinear: range-bearing measurements, CTRV trajectory
#
#  Measurement model: z = [r, θ] = [‖p‖, arctan2(py,px)] — nonlinear.
#  KF uses pseudo-Cartesian conversion (biased linearisation).
#  EKF uses analytical H Jacobian; UKF uses sigma points; PF uses exact
#  range-bearing likelihood.  Trajectory is a coordinated turn (omega=0.05
#  rad/s) to stress the measurement nonlinearity.
# ==============================================================================
print("Running Scenario 2 — Nonlinear range-bearing, CTRV trajectory...")

DT2        = 1.0
N2         = 120
OMEGA2     = 0.05           # rad/s turn rate
SIGMA_A2_2 = 0.5
SIGMA_R2   = 15.0
SIGMA_TH2  = np.deg2rad(2.0)
R2         = np.diag([SIGMA_R2**2, SIGMA_TH2**2])
F2         = make_F(DT2)
Q2         = make_Q(SIGMA_A2_2, DT2)

x0_true_2  = np.array([500.0, 0.0, 0.0, 10.0])
x0_2       = np.array([500.0, 0.5, 0.5,  9.5])
P0_2       = np.diag([SIGMA_R2**2, SIGMA_R2**2, 5.**2, 5.**2])
Wm2, Wc2, lam2 = ukf_weights(4)

def simulate_ctrv_4d(x0, N, dt, omega):
    """Constant-turn-rate trajectory in [px, py, vx, vy]."""
    states = np.zeros((N, 4)); states[0] = x0
    for k in range(1, N):
        px, py, vx, vy = states[k-1]
        ct = np.cos(omega*dt); st = np.sin(omega*dt)
        states[k] = [px + vx*dt, py + vy*dt,
                     vx*ct - vy*st, vx*st + vy*ct]
    return states

def simulate_rb(states, sr, sth, rng=None):
    rng = rng or np.random
    N   = len(states)
    z   = np.zeros((N, 2))
    z[:, 0] = np.hypot(states[:, 0], states[:, 1]) + rng.randn(N)*sr
    z[:, 1] = np.arctan2(states[:, 1], states[:, 0]) + rng.randn(N)*sth
    return z

np.random.seed(99)
true_states_2 = simulate_ctrv_4d(x0_true_2, N2, DT2, OMEGA2)
z2            = simulate_rb(true_states_2, SIGMA_R2, SIGMA_TH2)

# KF: convert range-bearing → pseudo-Cartesian.  Use inflated R to partially
# account for the linearisation bias (second-order approximation).
z2_cart = np.column_stack([z2[:,0]*np.cos(z2[:,1]),
                            z2[:,0]*np.sin(z2[:,1])])
r_nom   = np.hypot(x0_true_2[0], x0_true_2[1])
R2_cart = np.diag([(SIGMA_R2**2 + (r_nom*SIGMA_TH2)**2)] * 2)

t2 = np.arange(N2)

def run_s2_kf():
    x, P = x0_2.copy(), P0_2.copy()
    ests, covs, nis = [], [], []
    for k in range(N2):
        x, P = kf_predict(x, P, F2, Q2)
        x, P, ni = kf_update(x, P, z2_cart[k], H_CART, R2_cart)
        ests.append(x.copy()); covs.append(P.copy()); nis.append(ni)
    return np.array(ests), covs, np.array(nis)

def run_s2_ekf():
    x, P = x0_2.copy(), P0_2.copy()
    ests, covs, nis = [], [], []
    for k in range(N2):
        x, P = ekf_predict(x, P, F2, Q2)
        x, P, ni = ekf_update_rb(x, P, z2[k], R2)
        ests.append(x.copy()); covs.append(P.copy()); nis.append(ni)
    return np.array(ests), covs, np.array(nis)

def run_s2_ukf():
    x, P = x0_2.copy(), P0_2.copy()
    ests, covs, nis = [], [], []
    for k in range(N2):
        x, P, sig_f = ukf_predict(x, P, Q2, F2, Wm2, Wc2, lam2)
        x, P, ni    = ukf_update_rb(x, P, sig_f, z2[k], R2, Wm2, Wc2)
        ests.append(x.copy()); covs.append(P.copy()); nis.append(ni)
    return np.array(ests), covs, np.array(nis)

def run_s2_pf(seed=99):
    np.random.seed(seed + 200)
    particles, weights = pf_init(N_PF, x0_2, np.array([SIGMA_R2, SIGMA_R2, 5., 5.]))
    ests, covs, nis    = [], [], []
    sa = np.sqrt(SIGMA_A2_2)
    for k in range(N2):
        particles = pf_predict_cv(particles, DT2, sa)
        weights   = pf_update_rb(particles, weights, z2[k], R2)
        ni        = pf_nis_rb(particles, weights, z2[k], R2)
        particles, weights = pf_resample(particles, weights)
        ests.append(pf_estimate(particles, weights))
        covs.append(pf_cov(particles, weights))
        nis.append(ni)
    return np.array(ests), covs, np.array(nis)

est2 = {}; cov2 = {}; nis2 = {}
est2['KF'],  cov2['KF'],  nis2['KF']  = run_s2_kf()
est2['EKF'], cov2['EKF'], nis2['EKF'] = run_s2_ekf()
est2['UKF'], cov2['UKF'], nis2['UKF'] = run_s2_ukf()
est2['PF'],  cov2['PF'],  nis2['PF']  = run_s2_pf()
nees2 = {f: nees_ts(est2[f], cov2[f], true_states_2) for f in FILTERS}

# ── Fig S2-1: Trajectory + Position error ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes[0]
ax.plot(true_states_2[:,0], true_states_2[:,1], 'k-', lw=2.5,
        label='Ground truth', zorder=6)
ax.scatter(z2_cart[:,0], z2_cart[:,1], s=10, color='grey', alpha=0.45,
           label='Meas. (projected)', zorder=2)
for f in FILTERS:
    ax.plot(est2[f][:,0], est2[f][:,1], color=COLORS[f], ls=LSTYLE[f],
            lw=1.7, label=f, zorder=4)
ax.scatter([0],[0], s=220, c='black', marker='^', zorder=7, label='Sensor')
ax.set_xlabel('$p_x$ (m)'); ax.set_ylabel('$p_y$ (m)')
ax.set_title('Trajectory'); ax.set_aspect('equal')
ax.legend(ncol=2, fontsize=10); ax.grid(True, alpha=0.3)

ax = axes[1]
for f in FILTERS:
    err = pos_err_ts(est2[f], true_states_2)
    ax.plot(t2, err, color=COLORS[f], ls=LSTYLE[f], lw=1.7,
            label=f'{f} (RMSE={np.sqrt(np.mean(err**2)):.1f} m)')
ax.set_xlabel('Time step'); ax.set_ylabel('Position error (m)')
ax.set_title('Position Error'); ax.legend(); ax.grid(True, alpha=0.3)
plt.suptitle('Scenario 2 — Nonlinear (Range-Bearing): State Estimates',
             fontweight='bold')
plt.tight_layout()
plt.savefig('figures/s2_state_estimates.png')
plt.close()
print("  saved s2_state_estimates.png")

# ── Fig S2-2: NEES divergence (key plot) ────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
nees_band_plot(ax, t2, nees2, df=4, clip=60)
ax.set_title('Scenario 2 — NEES over Time: KF/EKF vs UKF/PF Divergence',
             fontweight='bold')
plt.tight_layout()
plt.savefig('figures/s2_nees.png')
plt.close()
print("  saved s2_nees.png")

# ── Monte Carlo RMSE (100 runs) ──────────────────────────────────────────────
print("  Running Monte Carlo (100 runs) for Scenario 2...")
N_MC     = 100
mc_err   = {f: np.zeros((N_MC, N2)) for f in FILTERS}
sa2      = np.sqrt(SIGMA_A2_2)

for mc in range(N_MC):
    rng_mc = np.random.RandomState(2000 + mc)
    z_mc   = simulate_rb(true_states_2, SIGMA_R2, SIGMA_TH2, rng=rng_mc)
    zc_mc  = np.column_stack([z_mc[:,0]*np.cos(z_mc[:,1]),
                               z_mc[:,0]*np.sin(z_mc[:,1])])
    # KF
    x, P = x0_2.copy(), P0_2.copy()
    for k in range(N2):
        x, P = kf_predict(x, P, F2, Q2)
        x, P, _ = kf_update(x, P, zc_mc[k], H_CART, R2_cart)
        mc_err['KF'][mc, k] = np.hypot(x[0]-true_states_2[k,0],
                                        x[1]-true_states_2[k,1])
    # EKF
    x, P = x0_2.copy(), P0_2.copy()
    for k in range(N2):
        x, P = ekf_predict(x, P, F2, Q2)
        x, P, _ = ekf_update_rb(x, P, z_mc[k], R2)
        mc_err['EKF'][mc, k] = np.hypot(x[0]-true_states_2[k,0],
                                          x[1]-true_states_2[k,1])
    # UKF
    x, P = x0_2.copy(), P0_2.copy()
    for k in range(N2):
        x, P, sf = ukf_predict(x, P, Q2, F2, Wm2, Wc2, lam2)
        x, P, _  = ukf_update_rb(x, P, sf, z_mc[k], R2, Wm2, Wc2)
        mc_err['UKF'][mc, k] = np.hypot(x[0]-true_states_2[k,0],
                                          x[1]-true_states_2[k,1])
    # PF
    np.random.seed(3000 + mc)
    p, w = pf_init(N_PF, x0_2, np.array([SIGMA_R2, SIGMA_R2, 5., 5.]))
    for k in range(N2):
        p = pf_predict_cv(p, DT2, sa2)
        w = pf_update_rb(p, w, z_mc[k], R2)
        p, w = pf_resample(p, w)
        est_k = pf_estimate(p, w)
        mc_err['PF'][mc, k] = np.hypot(est_k[0]-true_states_2[k,0],
                                         est_k[1]-true_states_2[k,1])

fig, ax = plt.subplots(figsize=(13, 5))
for f in FILTERS:
    mean_e = mc_err[f].mean(axis=0)
    lo_e   = np.percentile(mc_err[f], 10, axis=0)
    hi_e   = np.percentile(mc_err[f], 90, axis=0)
    rmse_f = np.sqrt(np.mean(mc_err[f]**2))
    ax.plot(t2, mean_e, color=COLORS[f], ls=LSTYLE[f], lw=2,
            label=f'{f}  (mean RMSE = {rmse_f:.1f} m)')
    ax.fill_between(t2, lo_e, hi_e, color=COLORS[f], alpha=0.12)
ax.set_xlabel('Time step'); ax.set_ylabel('Position error (m)')
ax.set_title(f'Scenario 2 — Monte Carlo RMSE ({N_MC} runs, shaded 10–90th pctile)',
             fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/s2_mc_rmse.png')
plt.close()
print("  saved s2_mc_rmse.png")

print("Scenario 2 complete.\n")

# ==============================================================================
#  SCENARIO 3 — Non-Gaussian noise: Student-t (ν=3)
#
#  Same linear CV as Scenario 1 but measurement noise ~ t(ν=3, scale=σ).
#  KF/EKF/UKF assume Gaussian R → consistency breaks on outliers.
#  PF uses the exact Student-t likelihood → remains consistent.
# ==============================================================================
print("Running Scenario 3 — Non-Gaussian noise (Student-t ν=3)...")

DT3        = 1.0
N3         = 150
NU         = 3
SIGMA_POS3 = 2.0
R3_gauss   = np.diag([SIGMA_POS3**2, SIGMA_POS3**2])
F3         = make_F(DT3)
Q3         = make_Q(SIGMA_A2_1, DT3)

x0_true_3  = np.array([0.0, 0.0, 2.0, 1.0])
x0_3       = np.array([0.5, 0.5, 1.5, 0.8])
P0_3       = np.diag([10.**2, 10.**2, 5.**2, 5.**2])
Wm3, Wc3, lam3 = ukf_weights(4)

np.random.seed(77)
true_states_3      = np.zeros((N3, 4)); true_states_3[0] = x0_true_3
for k in range(1, N3):
    true_states_3[k] = F3 @ true_states_3[k-1]
t_noise = t_dist.rvs(df=NU, scale=SIGMA_POS3, size=(N3, 2))
z3      = true_states_3[:, :2] + t_noise

t3 = np.arange(N3)
# Timesteps where at least one axis has a large outlier
outlier_thresh = t_dist.ppf(0.975, df=NU) * SIGMA_POS3
outlier_steps  = np.where(np.any(np.abs(t_noise) > outlier_thresh, axis=1))[0]

def run_s3_kf():
    x, P = x0_3.copy(), P0_3.copy()
    ests, covs = [], []
    for k in range(N3):
        x, P = kf_predict(x, P, F3, Q3)
        x, P, _ = kf_update(x, P, z3[k], H_CART, R3_gauss)
        ests.append(x.copy()); covs.append(P.copy())
    return np.array(ests), covs

def run_s3_ekf():
    x, P = x0_3.copy(), P0_3.copy()
    ests, covs = [], []
    for k in range(N3):
        x, P = ekf_predict(x, P, F3, Q3)
        x, P, _ = ekf_update_cartesian(x, P, z3[k], H_CART, R3_gauss)
        ests.append(x.copy()); covs.append(P.copy())
    return np.array(ests), covs

def run_s3_ukf():
    x, P = x0_3.copy(), P0_3.copy()
    ests, covs = [], []
    for k in range(N3):
        x, P, sig_f = ukf_predict(x, P, Q3, F3, Wm3, Wc3, lam3)
        x, P, _     = ukf_update_cartesian(x, P, sig_f, z3[k], H_CART, R3_gauss, Wm3, Wc3)
        ests.append(x.copy()); covs.append(P.copy())
    return np.array(ests), covs

def run_s3_pf(seed=77):
    np.random.seed(seed + 300)
    particles, weights = pf_init(N_PF, x0_3, np.array([10., 10., 5., 5.]))
    ests, covs = [], []
    sa = np.sqrt(SIGMA_A2_1)
    for k in range(N3):
        particles = pf_predict_cv(particles, DT3, sa)
        weights   = pf_update_student_t(particles, weights, z3[k], R3_gauss, nu=NU)
        particles, weights = pf_resample(particles, weights)
        ests.append(pf_estimate(particles, weights))
        covs.append(pf_cov(particles, weights))
    return np.array(ests), covs

est3 = {}; cov3 = {}
est3['KF'],  cov3['KF']  = run_s3_kf()
est3['EKF'], cov3['EKF'] = run_s3_ekf()
est3['UKF'], cov3['UKF'] = run_s3_ukf()
est3['PF'],  cov3['PF']  = run_s3_pf()
nees3   = {f: nees_ts(est3[f], cov3[f], true_states_3) for f in FILTERS}
overall3 = {f: np.sqrt(np.mean((est3[f][:,:2]-true_states_3[:,:2])**2))
            for f in FILTERS}

# ── Fig S3-1: RMSE comparison ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
bars = ax.bar(FILTERS, [overall3[f] for f in FILTERS],
              color=[COLORS[f] for f in FILTERS], edgecolor='k', lw=0.8)
for bar, f in zip(bars, FILTERS):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{overall3[f]:.3f}', ha='center', va='bottom', fontsize=11)
ax.set_ylabel('RMSE (m)'); ax.set_title('Overall Position RMSE')
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0, max(overall3.values())*1.4)

ax = axes[1]
for f in FILTERS:
    ax.plot(t3, pos_err_ts(est3[f], true_states_3),
            color=COLORS[f], ls=LSTYLE[f], lw=1.6, label=f)
# Shade outlier timesteps
for k in outlier_steps:
    ax.axvspan(k-0.5, k+0.5, color='grey', alpha=0.18, lw=0)
ax.plot([], [], color='grey', alpha=0.5, lw=8, label='Outlier meas.')
ax.set_xlabel('Time step'); ax.set_ylabel('Position error (m)')
ax.set_title('Position Error (grey = outlier meas.)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.suptitle(f'Scenario 3 — Student-t Noise ($\\nu={NU}$): RMSE Comparison',
             fontweight='bold')
plt.tight_layout()
plt.savefig('figures/s3_rmse.png')
plt.close()
print("  saved s3_rmse.png")

# ── Fig S3-2: NEES consistency breakdown ────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
nees_band_plot(ax, t3, nees3, df=4, clip=80)
# Overlay outlier shading
for k in outlier_steps:
    ax.axvspan(k-0.5, k+0.5, color='grey', alpha=0.18, lw=0)
ax.set_title(
    f'Scenario 3 — NEES: Consistency Breakdown under Student-t Noise ($\\nu={NU}$)\n'
    '(grey bars = outlier measurement steps)',
    fontweight='bold')
plt.tight_layout()
plt.savefig('figures/s3_nees.png')
plt.close()
print("  saved s3_nees.png")

print("Scenario 3 complete.\n")

# ==============================================================================
#  SUMMARY
# ==============================================================================
print("All figures saved to figures/:")
for fn in sorted(os.listdir('figures')):
    print(f"  {fn}")
