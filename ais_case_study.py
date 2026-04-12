"""
AIS Case Study — MAERSK SAIGON arrival at Port Newark, NJ
Generates all figures for the dissertation's real-data chapter.

Vessel:   MAERSK SAIGON (MMSI 636091221)
Day:      22 November 2016
Event:    Transatlantic arrival from the Atlantic into New York Harbour → Port Newark
Filters:  KF (CV model), EKF / UKF / PF (CTRV model)
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.colorbar import ColorbarBase
from scipy.stats import chi2
from scipy.linalg import block_diag
import contextily as ctx
from pyproj import Transformer

warnings.filterwarnings("ignore")

os.makedirs("figures", exist_ok=True)

# ── shared style (matches comparative_analysis.ipynb) ─────────────────────────
FS, FS_LEG, FS_TICK = 20, 16, 18
plt.rcParams.update({
    "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
    "figure.titlesize": FS + 2, "legend.fontsize": FS_LEG,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "lines.linewidth": 2.0, "savefig.dpi": 200,
    "savefig.bbox": "tight", "figure.facecolor": "white",
})
COLORS = {"KF": "#1f77b4", "EKF": "#ff7f0e", "UKF": "#2ca02c", "PF": "#d62728"}
LSTYLE = {"KF": "-",       "EKF": "--",       "UKF": "-.",      "PF": ":"}
FILTERS = ["KF", "EKF", "UKF", "PF"]

# ── WGS84 → Web Mercator transformer ─────────────────────────────────────────
WGS2MERC = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def to_merc(lon, lat):
    return WGS2MERC.transform(lon, lat)

# ═════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════
print("Loading AIS_2016_11_22.csv …")
df_all = pd.read_csv(
    "/Users/moyin/Dissertation/Data/AIS_2016_11_22.csv",
    usecols=["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG",
             "Heading", "VesselName", "VesselType"],
    dtype={"MMSI": str},
)
df_all["BaseDateTime"] = pd.to_datetime(df_all["BaseDateTime"])
print(f"  {len(df_all):,} rows, {df_all['MMSI'].nunique():,} vessels")

# ── select MAERSK SAIGON (MMSI 636091221) ────────────────────────────────────
MMSI_TARGET = "636091221"
df = (df_all[df_all["MMSI"] == MMSI_TARGET]
      .sort_values("BaseDateTime")
      .reset_index(drop=True))

# keep only the moving phase (first 615 pts, 01:53–13:44 UTC)
df_mov = df[df["SOG"] > 0.5].reset_index(drop=True)
print(f"  Moving phase: {len(df_mov)} pts  "
      f"{df_mov['BaseDateTime'].iloc[0].strftime('%H:%M')} → "
      f"{df_mov['BaseDateTime'].iloc[-1].strftime('%H:%M')} UTC")

# ═════════════════════════════════════════════════════════════════════════════
# 2.  COORDINATE TRANSFORMATION  (equirectangular, origin = first moving point)
# ═════════════════════════════════════════════════════════════════════════════
R_EARTH = 6_371_000.0   # m

lat0_rad = np.deg2rad(df_mov["LAT"].iloc[0])
lon0_rad = np.deg2rad(df_mov["LON"].iloc[0])

def latlon_to_xy(lat_deg, lon_deg):
    lat_r = np.deg2rad(lat_deg)
    lon_r = np.deg2rad(lon_deg)
    x = R_EARTH * (lon_r - lon0_rad) * np.cos(lat0_rad)
    y = R_EARTH * (lat_r - lat0_rad)
    return x, y

x_mov, y_mov = latlon_to_xy(df_mov["LAT"].values, df_mov["LON"].values)
positions_all = np.column_stack([x_mov, y_mov])

times_s_all = df_mov["BaseDateTime"].values.astype("datetime64[s]").astype(float)

SOG_MS_all  = df_mov["SOG"].values * 0.514_44
COG_DEG_all = df_mov["COG"].values

# ── Filter sub-track: skip first 3 anomalous observations (01:53-02:20) ──────
# Rows 0-2 are at slow speed (≤12 kn) with a 24-minute gap in between.
# Filters are initialised from row 3 (02:28 UTC, 16.4 kn, well into cruise).
FILT_OFFSET = 3
positions = positions_all[FILT_OFFSET:]
times_s   = times_s_all[FILT_OFFSET:]
dt_arr    = np.diff(times_s, prepend=times_s[0])
SOG_MS    = SOG_MS_all[FILT_OFFSET:]
COG_DEG   = COG_DEG_all[FILT_OFFSET:]

# Full-track dt array (for preprocessing figure — shows all 615 observations)
dt_arr_all = np.diff(times_s_all, prepend=times_s_all[0])
t_hours_all = (times_s_all - times_s_all[0]) / 3600.0

# Convert COG (degrees from north, clockwise) → math-angle θ (from +x-axis, CCW)
THETA_0 = np.pi / 2.0 - np.deg2rad(COG_DEG[0])

N = len(positions)

# ═════════════════════════════════════════════════════════════════════════════
# 3.  FILTER IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════════════════
SEED = 42
rng  = np.random.default_rng(SEED)

# --- Noise parameters --------------------------------------------------------
SIGMA_GPS   = 20.0    # m   (AIS class-A positional accuracy)
N_PARTICLES = 2000

R_meas = np.diag([SIGMA_GPS**2, SIGMA_GPS**2])
H_meas = np.array([[1,0,0,0], [0,0,1,0]])   # KF measurement matrix (4-state)
H5     = np.array([[1,0,0,0,0], [0,1,0,0,0]])  # CTRV measurement (5-state)

Q_DT_CAP = 120.0   # cap process noise growth at 2 minutes (handles long AIS gaps)

def kf_Q_mat(dt, sigma_a=0.08):
    """CWNA (continuous white noise acceleration) process noise, dt-capped."""
    t = min(dt, Q_DT_CAP)
    q = sigma_a**2 * np.array([[t**3/3, t**2/2],
                                 [t**2/2, t]])
    return block_diag(q, q)

def ctrv_Q(dt, sq_x=3.0, sq_v=0.3, sq_th=0.002, sq_om=0.0002):
    """Diagonal random-walk process noise for CTRV, dt-capped."""
    t = min(dt, Q_DT_CAP)
    return np.diag([sq_x**2*t, sq_x**2*t, sq_v**2*t, sq_th**2*t, sq_om**2*t])


# ── 3a: KF (Constant Velocity, 4-state) ──────────────────────────────────────
def kf_run(positions, dt_arr):
    N = len(positions)
    x = np.array([positions[0,0], SOG_MS[0]*np.cos(THETA_0),
                  positions[0,1], SOG_MS[0]*np.sin(THETA_0)])
    P = np.diag([SIGMA_GPS**2, 4.0, SIGMA_GPS**2, 4.0])

    est, innov, nis = [], [], []
    for k in range(N):
        dt = dt_arr[k]
        F  = np.array([[1,dt,0,0],[0,1,0,0],[0,0,1,dt],[0,0,0,1]])
        # predict
        x  = F @ x
        P  = F @ P @ F.T + kf_Q_mat(dt)
        # update
        z  = positions[k]
        y  = z - H_meas @ x
        S  = H_meas @ P @ H_meas.T + R_meas
        K  = P @ H_meas.T @ np.linalg.inv(S)
        x  = x + K @ y
        P  = (np.eye(4) - K @ H_meas) @ P
        P  = (P + P.T) / 2
        est.append([x[0], x[2]])
        innov.append(y)
        nis.append(float(y @ np.linalg.inv(S) @ y))
    return np.array(est), innov, nis


# ── CTRV helpers ──────────────────────────────────────────────────────────────
def ctrv_f(s, dt):
    """Propagate CTRV state s = [x, y, v, theta, omega]."""
    x, y, v, th, om = s
    eps = 1e-6
    if abs(om) > eps:
        x2 = x + (v/om)*(np.sin(th + om*dt) - np.sin(th))
        y2 = y + (v/om)*(-np.cos(th + om*dt) + np.cos(th))
    else:
        x2 = x + v*np.cos(th)*dt
        y2 = y + v*np.sin(th)*dt
    return np.array([x2, y2, v, th + om*dt, om])


def ctrv_F_jac(s, dt):
    """Jacobian of CTRV transition (for EKF)."""
    x, y, v, th, om = s
    eps = 1e-6
    F = np.eye(5)
    F[3, 4] = dt
    if abs(om) > eps:
        sth  = np.sin(th);          cth  = np.cos(th)
        sth2 = np.sin(th + om*dt);  cth2 = np.cos(th + om*dt)
        F[0, 2] = (sth2 - sth) / om
        F[0, 3] = (v/om)*(cth2 - cth)
        F[0, 4] = v*(dt*cth2/om - (sth2 - sth)/om**2)
        F[1, 2] = (-cth2 + cth) / om
        F[1, 3] = (v/om)*(sth2 - sth)
        F[1, 4] = v*(dt*sth2/om - (-cth2 + cth)/om**2)
    else:
        F[0, 2] = np.cos(th)*dt
        F[0, 3] = -v*np.sin(th)*dt
        F[1, 2] = np.sin(th)*dt
        F[1, 3] =  v*np.cos(th)*dt
    return F


def _ctrv_Q_old_unused(s, dt):
    """Kept for reference — replaced by module-level ctrv_Q()."""
    pass


# ── 3b: EKF (CTRV, 5-state) ──────────────────────────────────────────────────
def ekf_run(positions, dt_arr):
    N = len(positions)
    x = np.array([positions[0,0], positions[0,1],
                  SOG_MS[0], THETA_0, 0.0])
    P = np.diag([SIGMA_GPS**2, SIGMA_GPS**2, 2.0, 0.01, 0.0005])

    est, innov, nis = [], [], []
    for k in range(N):
        dt = max(dt_arr[k], 0.5)
        # predict
        F  = ctrv_F_jac(x, dt)
        x  = ctrv_f(x, dt)
        x[3] = np.arctan2(np.sin(x[3]), np.cos(x[3]))  # normalise heading
        P  = F @ P @ F.T + ctrv_Q(dt)
        P  = (P + P.T) / 2 + np.eye(5) * 1e-8
        # update
        z  = positions[k]
        y  = z - H5 @ x
        S  = H5 @ P @ H5.T + R_meas
        K  = P @ H5.T @ np.linalg.inv(S)
        x  = x + K @ y
        x[2] = max(x[2], 0.0)            # speed ≥ 0
        x[3] = np.arctan2(np.sin(x[3]), np.cos(x[3]))  # normalise heading
        P  = (np.eye(5) - K @ H5) @ P
        P  = (P + P.T) / 2 + np.eye(5) * 1e-8
        est.append([x[0], x[1]])
        innov.append(y)
        nis.append(float(y @ np.linalg.inv(S) @ y))
    return np.array(est), innov, nis


# ── 3c: UKF (CTRV, 5-state) ──────────────────────────────────────────────────
def sigma_points(x, P, alpha=1.0, beta=2, kappa=0):
    n = len(x)
    lam = alpha**2 * (n + kappa) - n
    Wm  = np.full(2*n+1, 0.5/(n+lam))
    Wc  = Wm.copy()
    Wm[0] = lam / (n + lam)
    Wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)
    # ensure positive-definite with incremental regularisation
    M = (n + lam) * ((P + P.T) / 2)
    jitter = 1e-6
    for _ in range(20):
        try:
            S = np.linalg.cholesky(M)
            break
        except np.linalg.LinAlgError:
            M += np.eye(n) * jitter
            jitter *= 10
    else:
        raise np.linalg.LinAlgError("P not positive-definite after regularisation")
    sps = np.zeros((2*n+1, n))
    sps[0] = x
    for i in range(n):
        sps[i+1]   = x + S[:,i]
        sps[n+i+1] = x - S[:,i]
    return sps, Wm, Wc


def ctrv_circular_mean(sp_pred, Wm):
    """Weighted mean of CTRV sigma points with circular mean for heading (idx 3)."""
    x_mean = Wm @ sp_pred
    # Circular mean prevents wrap-around error near ±π
    sin_h = np.sum(Wm * np.sin(sp_pred[:, 3]))
    cos_h = np.sum(Wm * np.cos(sp_pred[:, 3]))
    x_mean[3] = np.arctan2(sin_h, cos_h)
    return x_mean


def ukf_run(positions, dt_arr):
    N = len(positions)
    x = np.array([positions[0,0], positions[0,1],
                  SOG_MS[0], THETA_0, 0.0])
    P = np.diag([SIGMA_GPS**2, SIGMA_GPS**2, 2.0, 0.01, 0.0005])

    est, innov, nis = [], [], []
    for k in range(N):
        dt = max(dt_arr[k], 0.5)
        # ── predict
        sps, Wm, Wc = sigma_points(x, P)
        sp_pred = np.array([ctrv_f(sp, dt) for sp in sps])
        x_pred  = ctrv_circular_mean(sp_pred, Wm)
        P_pred  = ctrv_Q(dt)
        for i, sp in enumerate(sp_pred):
            d    = sp - x_pred
            d[3] = np.arctan2(np.sin(d[3]), np.cos(d[3]))  # wrap angle diff
            P_pred += Wc[i] * np.outer(d, d)
        P_pred = (P_pred + P_pred.T) / 2 + np.eye(5) * 1e-8
        # ── update
        z_sigma = sp_pred[:, :2]
        z_pred  = Wm @ z_sigma
        Pzz = R_meas.copy()
        Pxz = np.zeros((5, 2))
        for i, sp in enumerate(sp_pred):
            dz   = z_sigma[i] - z_pred
            dx   = sp - x_pred
            dx[3] = np.arctan2(np.sin(dx[3]), np.cos(dx[3]))
            Pzz += Wc[i] * np.outer(dz, dz)
            Pxz += Wc[i] * np.outer(dx, dz)
        S  = Pzz
        K  = Pxz @ np.linalg.inv(S)
        y  = positions[k] - z_pred
        x  = x_pred + K @ y
        x[2] = max(x[2], 0.0)
        x[3] = np.arctan2(np.sin(x[3]), np.cos(x[3]))
        P  = P_pred - K @ S @ K.T
        P  = (P + P.T) / 2 + np.eye(5) * 1e-8
        est.append([x[0], x[1]])
        innov.append(y)
        nis.append(float(y @ np.linalg.inv(S) @ y))
    return np.array(est), innov, nis


# ── 3d: PF (CTRV, SIR with roughening) ──────────────────────────────────────
# Roughening (post-resample jitter) is added to prevent sample impoverishment.
# Without it the bootstrap PF collapses to a single particle on this 12-hour
# track, causing catastrophic divergence at the harbour approach.
def pf_run(positions, dt_arr, n_par=N_PARTICLES):
    N   = len(positions)
    par = np.column_stack([
        rng.normal(positions[0,0], SIGMA_GPS, n_par),
        rng.normal(positions[0,1], SIGMA_GPS, n_par),
        np.clip(rng.normal(SOG_MS[0], 1.0,   n_par), 0, None),
        rng.normal(THETA_0,         0.1,      n_par),
        rng.normal(0.0,             0.0002,   n_par),
    ])
    w   = np.ones(n_par) / n_par
    est, innov, nis_list = [], [], []

    # roughening scale per state (tuned to ship dynamics)
    ROUGH_SIGMA = np.array([15.0, 15.0, 0.3, 0.01, 0.0002])

    for k in range(N):
        dt = max(dt_arr[k], 0.5)
        # ── propagate through CTRV with process noise
        for i in range(n_par):
            par[i] = ctrv_f(par[i], dt)
        t_q = min(dt, Q_DT_CAP)
        par[:, 0] += rng.normal(0, 3.0 * np.sqrt(t_q), n_par)
        par[:, 1] += rng.normal(0, 3.0 * np.sqrt(t_q), n_par)
        par[:, 2] += rng.normal(0, 0.3 * np.sqrt(t_q), n_par)
        par[:, 4] += rng.normal(0, 0.0002 * np.sqrt(t_q), n_par)
        par[:, 2]  = np.clip(par[:, 2], 0, None)

        # ── weight by likelihood of measurement
        z  = positions[k]
        dx = par[:, 0] - z[0]
        dy = par[:, 1] - z[1]
        log_w  = -0.5 * (dx**2 + dy**2) / SIGMA_GPS**2
        log_w -= log_w.max()
        w      = np.exp(log_w)
        w     /= w.sum()

        # ── weighted estimate
        x_est = w @ par
        est.append([x_est[0], x_est[1]])

        # ── NIS (innovation covariance = particle spread + measurement noise)
        z_pred = x_est[:2]
        y_inn  = z - z_pred
        cov_par = np.cov(par[:, :2].T, aweights=w) + np.eye(2) * 1e-6
        S_approx = cov_par + R_meas
        innov.append(y_inn)
        try:
            nis_val = float(y_inn @ np.linalg.inv(S_approx) @ y_inn)
        except Exception:
            nis_val = float("nan")
        nis_list.append(nis_val)

        # ── systematic resample + roughen to prevent collapse
        N_eff = 1.0 / np.sum(w**2)
        if N_eff < n_par / 2:
            cumsum = np.cumsum(w)
            u   = (rng.random() + np.arange(n_par)) / n_par
            idx = np.searchsorted(cumsum, u)
            par = par[idx].copy()
            w   = np.ones(n_par) / n_par
            # roughening: small jitter to restore diversity
            par += rng.normal(0, ROUGH_SIGMA, (n_par, 5))
            par[:, 2] = np.clip(par[:, 2], 0, None)

    return np.array(est), innov, nis_list


# ═════════════════════════════════════════════════════════════════════════════
# 4.  RUN ALL FILTERS
# ═════════════════════════════════════════════════════════════════════════════
print("Running KF …");  kf_est,  kf_inn,  kf_nis  = kf_run(positions, dt_arr)
print("Running EKF …"); ekf_est, ekf_inn, ekf_nis = ekf_run(positions, dt_arr)
print("Running UKF …"); ukf_est, ukf_inn, ukf_nis = ukf_run(positions, dt_arr)
print("Running PF …");  pf_est,  pf_inn,  pf_nis  = pf_run(positions, dt_arr)

estimates = {"KF": kf_est, "EKF": ekf_est, "UKF": ukf_est, "PF": pf_est}
nis_vals  = {"KF": kf_nis, "EKF": ekf_nis, "UKF": ukf_nis, "PF": pf_nis}

# ═════════════════════════════════════════════════════════════════════════════
# 5.  FIGURES
# ═════════════════════════════════════════════════════════════════════════════

# --- helper: convert local Cartesian back to (lon,lat) then to Mercator ------
def xy_to_latlon(x, y):
    lat_rad = y / R_EARTH + lat0_rad
    lon_rad = x / (R_EARTH * np.cos(lat0_rad)) + lon0_rad
    return np.rad2deg(lon_rad), np.rad2deg(lat_rad)

def xy_to_merc(x, y):
    lon, lat = xy_to_latlon(x, y)
    return to_merc(lon, lat)

# Mercator coords of raw measurements (all 615 moving points)
lon_mov = df_mov["LON"].values
lat_mov = df_mov["LAT"].values
mx_raw, my_raw = to_merc(lon_mov, lat_mov)

# Mercator coords for filter estimates (starts at FILT_OFFSET into the raw track)
mx_raw_f = mx_raw[FILT_OFFSET:]
my_raw_f = my_raw[FILT_OFFSET:]

mx_est = {}; my_est = {}
for f in FILTERS:
    ex, ey = xy_to_merc(estimates[f][:,0], estimates[f][:,1])
    mx_est[f], my_est[f] = ex, ey

# ── Figure 1: Dataset overview — all vessels on US East/Gulf coast map ────────
print("Figure 1: dataset overview …")
# sample 8000 points from the full dataset for a fast overview scatter
sample = df_all.sample(8000, random_state=42)
sx, sy = to_merc(sample["LON"].values, sample["LAT"].values)

fig, ax = plt.subplots(figsize=(12, 9))
ax.scatter(sx, sy, s=1, alpha=0.35, color="#555555", rasterized=True)
# highlight MAERSK SAIGON track
ax.plot(mx_raw, my_raw, color="#d62728", lw=1.8, zorder=5, label="MAERSK SAIGON")
ax.scatter(mx_raw[-1], my_raw[-1], marker="*", s=220, color="#d62728", zorder=6)

# zoom to interesting region (US east coast + gulf): lon -100→-60, lat 22→52
x_lo, y_lo = to_merc(-100, 22);  x_hi, y_hi = to_merc(-60, 52)
ax.set_xlim(x_lo, x_hi);  ax.set_ylim(y_lo, y_hi)
try:
    ctx.add_basemap(ax, crs="EPSG:3857",
                    source=ctx.providers.CartoDB.Positron, zoom=5)
except Exception:
    pass
ax.set_xlabel("Longitude");  ax.set_ylabel("Latitude")
ax.set_title("AIS Dataset Overview — 22 November 2016\n"
             f"({df_all['MMSI'].nunique():,} vessels sampled)")
# replace axis ticks with degree labels
xt = np.arange(-100, -55, 10)
yt = np.arange(25, 55, 5)
xtm, _ = to_merc(xt, np.zeros_like(xt))
_, ytm = to_merc(np.zeros_like(yt), yt)
ax.set_xticks(xtm); ax.set_xticklabels([f"{v}°W" for v in abs(xt)], fontsize=FS_TICK-2)
ax.set_yticks(ytm); ax.set_yticklabels([f"{v}°N" for v in yt],       fontsize=FS_TICK-2)
leg = ax.legend(loc="lower right", fontsize=FS_LEG, framealpha=0.85)
plt.savefig("figures/ais_overview_map.png"); plt.close()
print("  → figures/ais_overview_map.png")

# ── Figure 2: MAERSK SAIGON full arrival — zoomed to NY/NJ area ──────────────
print("Figure 2: vessel arrival map …")
fig, ax = plt.subplots(figsize=(12, 8))

# colour by SOG
sog_vals = df_mov["SOG"].values
norm  = mcolors.Normalize(vmin=0, vmax=sog_vals.max())
cmap  = plt.cm.plasma
sc = ax.scatter(mx_raw, my_raw, c=sog_vals, cmap=cmap, norm=norm,
                s=6, zorder=4, rasterized=True)
cbar = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Speed over ground (knots)", fontsize=FS_LEG)

# start / end markers
ax.scatter(mx_raw[0],  my_raw[0],  marker="^", s=200, color="green",
           zorder=6, label="Start (01:54 UTC)")
ax.scatter(mx_raw[-1], my_raw[-1], marker="s", s=200, color="#d62728",
           zorder=6, label="Berth — Port Newark")

# zoom to NY/NJ corridor + some open ocean
x_lo, y_lo = to_merc(-74.5, 40.3);  x_hi, y_hi = to_merc(-70.8, 40.8)
ax.set_xlim(x_lo, x_hi);  ax.set_ylim(y_lo, y_hi)
try:
    ctx.add_basemap(ax, crs="EPSG:3857",
                    source=ctx.providers.CartoDB.Positron, zoom=10)
except Exception:
    pass

xt = np.arange(-74.5, -70.5, 1.0)
yt = np.array([40.3, 40.5, 40.7])
xtm, _ = to_merc(xt, np.zeros_like(xt))
_, ytm = to_merc(np.zeros_like(yt), yt)
ax.set_xticks(xtm); ax.set_xticklabels([f"{abs(v):.1f}°W" for v in xt], fontsize=FS_TICK-2)
ax.set_yticks(ytm); ax.set_yticklabels([f"{v:.1f}°N"       for v in yt], fontsize=FS_TICK-2)
ax.set_title("MAERSK SAIGON — Arrival at Port Newark, NJ\n22 November 2016  (coloured by speed)")
ax.legend(loc="upper right", fontsize=FS_LEG, framealpha=0.9)
plt.savefig("figures/ais_vessel_arrival.png"); plt.close()
print("  → figures/ais_vessel_arrival.png")

# ── Figure 3: Coordinate transformation illustration ─────────────────────────
print("Figure 3: coordinate transformation …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(lon_mov, lat_mov, ".", ms=2, color="#555555", rasterized=True)
ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
ax.set_title("(a) Geographic Coordinates")
ax.set_aspect("equal")

ax = axes[1]
ax.plot(x_mov / 1000, y_mov / 1000, ".", ms=2, color="#1f77b4", rasterized=True)
ax.set_xlabel("$x$ (km)"); ax.set_ylabel("$y$ (km)")
ax.set_title("(b) Local Cartesian (equirectangular)")
ax.set_aspect("equal")

plt.suptitle("Coordinate Transformation — MAERSK SAIGON track", fontsize=FS+2)
plt.tight_layout()
plt.savefig("figures/ais_coordinate_transform.png"); plt.close()
print("  → figures/ais_coordinate_transform.png")

# ── Figure 4: Preprocessing — sampling interval & SOG time-series ────────────
print("Figure 4: preprocessing …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# dt histogram (exclude first zero-entry; use full moving-phase intervals)
dts_pos = dt_arr_all[dt_arr_all > 0]
ax = axes[0]
ax.hist(np.clip(dts_pos, 0, 500), bins=60, color="#1f77b4",
        edgecolor="white", linewidth=0.4)
ax.axvline(np.median(dts_pos), color="#d62728", ls="--", lw=1.8,
           label=f"Median = {np.median(dts_pos):.0f} s")
ax.set_xlabel("Sampling interval (s, clipped at 500 s)")
ax.set_ylabel("Count")
ax.set_title("(a) Inter-message Intervals")
ax.legend(fontsize=FS_LEG)
ax.text(0.98, 0.85, f"1 gap = {int(dts_pos.max())} s\n(24-min AIS outage)",
        transform=ax.transAxes, ha="right", fontsize=FS_LEG-2, color="#555")

# SOG time-series (full moving phase)
ax = axes[1]
ax.plot(t_hours_all, df_mov["SOG"].values, color="#2ca02c", lw=1.4)
ax.set_xlabel("Time since first observation (h)")
ax.set_ylabel("Speed over ground (knots)")
ax.set_title("(b) Speed Profile — Transatlantic Approach")
ax.annotate("Open-ocean\ncruise ≈18 kn", xy=(3.0, 17.8),
            xytext=(4.5, 16.0), fontsize=FS_LEG,
            arrowprops=dict(arrowstyle="->", color="black"))
ax.annotate("Harbour\napproach", xy=(10.0, 8.0),
            xytext=(7.5, 12.0), fontsize=FS_LEG,
            arrowprops=dict(arrowstyle="->", color="black"))

plt.suptitle("Data Preprocessing — MAERSK SAIGON", fontsize=FS+2)
plt.tight_layout()
plt.savefig("figures/ais_preprocessing.png"); plt.close()
print("  → figures/ais_preprocessing.png")

# ── Figure 5: Filter comparison — geographic map (full track) ─────────────────
print("Figure 5: filter comparison map …")
fig, ax = plt.subplots(figsize=(14, 7))

# raw measurements (thin grey) — all 615 points for geographic context
ax.plot(mx_raw, my_raw, color="#aaaaaa", lw=0.8, zorder=2, label="Raw AIS")

for f in FILTERS:
    ax.plot(mx_est[f], my_est[f],
            color=COLORS[f], ls=LSTYLE[f], lw=1.8, zorder=3+FILTERS.index(f),
            label=f)

ax.scatter(mx_raw[0],  my_raw[0],  marker="^", s=180, color="green",
           zorder=10, label="Start (01:54 UTC)")
ax.scatter(mx_raw[-1], my_raw[-1], marker="s", s=180, color="black",
           zorder=10, label="Port Newark berth")

x_lo, y_lo = to_merc(-74.5, 40.3);  x_hi, y_hi = to_merc(-70.8, 40.8)
ax.set_xlim(x_lo, x_hi);  ax.set_ylim(y_lo, y_hi)
try:
    ctx.add_basemap(ax, crs="EPSG:3857",
                    source=ctx.providers.CartoDB.Positron, zoom=10)
except Exception:
    pass

xt = np.arange(-74.5, -70.5, 1.0)
yt = np.array([40.3, 40.5, 40.7])
xtm, _ = to_merc(xt, np.zeros_like(xt))
_, ytm = to_merc(np.zeros_like(yt), yt)
ax.set_xticks(xtm); ax.set_xticklabels([f"{abs(v):.1f}°W" for v in xt], fontsize=FS_TICK-2)
ax.set_yticks(ytm); ax.set_yticklabels([f"{v:.1f}°N"       for v in yt], fontsize=FS_TICK-2)
ax.set_title("Filter Comparison — MAERSK SAIGON Arrival\n22 November 2016")
ax.legend(loc="upper right", fontsize=FS_LEG, framealpha=0.9)
plt.savefig("figures/ais_filter_comparison_map.png"); plt.close()
print("  → figures/ais_filter_comparison_map.png")

# ── Figure 6: zoomed panel — harbour approach (last ~120 pts) ────────────────
print("Figure 6: harbour approach zoom …")
Z = 180   # last Z moving-phase points for zoom

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for col, (subset_idx, title) in enumerate([
    (slice(100, 220),  "(a) Open-Ocean Cruise — Mid-Atlantic"),
    (slice(-Z, None),  "(b) Harbour Approach & Deceleration"),
]):
    ax = axes[col]
    ax.plot(mx_raw_f[subset_idx], my_raw_f[subset_idx],
            ".", ms=4, color="#aaaaaa", zorder=2, label="Raw AIS")
    for f in FILTERS:
        ax.plot(mx_est[f][subset_idx], my_est[f][subset_idx],
                color=COLORS[f], ls=LSTYLE[f], lw=2.2,
                zorder=3+FILTERS.index(f), label=f)

    xi = mx_raw_f[subset_idx]; yi = my_raw_f[subset_idx]
    xi_lo, xi_hi = xi.min()-500, xi.max()+500
    yi_lo, yi_hi = yi.min()-500, yi.max()+500
    ax.set_xlim(xi_lo, xi_hi);  ax.set_ylim(yi_lo, yi_hi)
    try:
        ctx.add_basemap(ax, crs="EPSG:3857",
                        source=ctx.providers.CartoDB.Positron)
    except Exception:
        pass
    ax.set_title(title, fontsize=FS)
    ax.set_xticks([]); ax.set_yticks([])

axes[0].legend(loc="upper left", fontsize=FS_LEG-2, framealpha=0.9)
plt.suptitle("Filter Comparison — Zoomed Segments", fontsize=FS+2)
plt.tight_layout()
plt.savefig("figures/ais_filter_zoomed.png"); plt.close()
print("  → figures/ais_filter_zoomed.png")

# ── Figure 7: NIS consistency (chi² bounds, DOF=2) ───────────────────────────
print("Figure 7: NIS consistency …")
chi2_lo = chi2.ppf(0.025, df=2)   # ≈ 0.051
chi2_hi = chi2.ppf(0.975, df=2)   # ≈ 7.378

fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
axes = axes.flatten()

for ax, f in zip(axes, FILTERS):
    nv = np.array(nis_vals[f])
    steps = np.arange(len(nv))
    # rolling median for readability
    win = 20
    rm  = pd.Series(nv).rolling(win, center=True, min_periods=1).median().values
    ax.plot(steps, nv, alpha=0.25, lw=0.8, color=COLORS[f])
    ax.plot(steps, rm, lw=2.0,   color=COLORS[f], label="Rolling median")
    ax.axhline(chi2_hi, ls="--", lw=1.5, color="black",  label="95% upper (7.38)")
    ax.axhline(chi2_lo, ls=":",  lw=1.5, color="#888888", label="95% lower (0.05)")
    frac_ok = np.mean((nv >= chi2_lo) & (nv <= chi2_hi))
    ax.set_title(f"{f}  ({frac_ok*100:.0f}% within bounds)", fontsize=FS, color=COLORS[f])
    finite_max = np.nanmax(nv[np.isfinite(nv)]) if np.any(np.isfinite(nv)) else 40
    ax.set_ylim(0, min(finite_max * 1.05, 40))
    ax.set_ylabel("NIS")
    if ax is axes[0]:
        ax.legend(fontsize=FS_LEG-2)

for ax in axes[-2:]:
    ax.set_xlabel("Time step")

plt.suptitle("NIS Consistency Test — χ² bounds (DOF = 2)", fontsize=FS+2)
plt.tight_layout()
plt.savefig("figures/ais_nis_consistency.png"); plt.close()
print("  → figures/ais_nis_consistency.png")

# ── Figure 8: innovation magnitude over time ──────────────────────────────────
print("Figure 8: innovation magnitudes …")
fig, ax = plt.subplots(figsize=(13, 5))

inn_dict = {"KF": kf_inn, "EKF": ekf_inn, "UKF": ukf_inn, "PF": pf_inn}
for f in FILTERS:
    mags = np.linalg.norm(np.array(inn_dict[f]), axis=1)
    ax.plot(mags, color=COLORS[f], ls=LSTYLE[f], lw=1.6, alpha=0.85, label=f)

ax.set_xlabel("Time step")
ax.set_ylabel("Innovation magnitude (m)")
ax.set_title("Innovation Magnitude — All Filters")
ax.legend(fontsize=FS_LEG)
plt.tight_layout()
plt.savefig("figures/ais_innovations.png"); plt.close()
print("  → figures/ais_innovations.png")

# ── Figure 9: 4-panel per-filter map (for a single clean comparison figure) ──
print("Figure 9: 4-panel map comparison …")
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

x_lo, y_lo = to_merc(-74.5, 40.3);  x_hi, y_hi = to_merc(-70.8, 40.8)

for ax, f in zip(axes, FILTERS):
    ax.plot(mx_raw, my_raw, ".", ms=2, color="#cccccc", zorder=2)
    ax.plot(mx_est[f], my_est[f], color=COLORS[f], lw=2.0, zorder=4)
    ax.scatter(mx_raw[0],  my_raw[0],  marker="^", s=120, color="green",  zorder=6)
    ax.scatter(mx_raw[-1], my_raw[-1], marker="s", s=120, color="black",  zorder=6)
    ax.set_xlim(x_lo, x_hi);  ax.set_ylim(y_lo, y_hi)
    try:
        ctx.add_basemap(ax, crs="EPSG:3857",
                        source=ctx.providers.CartoDB.Positron, zoom=10)
    except Exception:
        pass
    ax.set_title(f, fontsize=FS+2, color=COLORS[f], fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

legend_elements = [
    Line2D([0],[0], color="#cccccc", lw=2, label="Raw AIS"),
    Line2D([0],[0], marker="^", color="w", markerfacecolor="green",
           markersize=10, label="Start"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor="black",
           markersize=10, label="Port Newark berth"),
]
fig.legend(handles=legend_elements, loc="lower center",
           ncol=3, fontsize=FS_LEG, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
plt.suptitle("Filter Estimates — MAERSK SAIGON Arrival at Port Newark, NJ",
             fontsize=FS+4, y=1.01)
plt.tight_layout()
plt.savefig("figures/ais_filter_4panel_map.png", bbox_inches="tight")
plt.close()
print("  → figures/ais_filter_4panel_map.png")

# ═════════════════════════════════════════════════════════════════════════════
# 10. PSEUDO-RMSE VIA DOWNSAMPLING
#
#     Approach: each filter runs at FULL temporal resolution (all 612 steps,
#     full dt_arr) but receives measurement updates only at every-Nth step.
#     At the N-1 intermediate steps the filter only predicts.
#     RMSE is then computed over ALL steps against the full-resolution
#     positions — capturing interpolation quality, not just fit-at-observations.
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 10 & 11: Pseudo-RMSE via downsampling …")


def kf_run_gapped(positions, dt_arr, obs_mask):
    """KF at full resolution; update only where obs_mask[k] is True."""
    n = len(positions)
    x = np.array([positions[0,0], SOG_MS[0]*np.cos(THETA_0),
                  positions[0,1], SOG_MS[0]*np.sin(THETA_0)])
    P = np.diag([SIGMA_GPS**2, 4.0, SIGMA_GPS**2, 4.0])
    est = []
    for k in range(n):
        dt = dt_arr[k]
        F  = np.array([[1,dt,0,0],[0,1,0,0],[0,0,1,dt],[0,0,0,1]])
        x  = F @ x
        P  = F @ P @ F.T + kf_Q_mat(dt)
        if obs_mask[k]:
            z = positions[k];  y = z - H_meas @ x
            S = H_meas @ P @ H_meas.T + R_meas
            K = P @ H_meas.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(4) - K @ H_meas) @ P
            P = (P + P.T) / 2
        est.append([x[0], x[2]])
    return np.array(est)


def ekf_run_gapped(positions, dt_arr, obs_mask):
    """EKF at full resolution; update only where obs_mask[k] is True."""
    n = len(positions)
    x = np.array([positions[0,0], positions[0,1], SOG_MS[0], THETA_0, 0.0])
    P = np.diag([SIGMA_GPS**2, SIGMA_GPS**2, 2.0, 0.01, 0.0005])
    est = []
    for k in range(n):
        dt = max(dt_arr[k], 0.5)
        F  = ctrv_F_jac(x, dt);  x = ctrv_f(x, dt)
        x[3] = np.arctan2(np.sin(x[3]), np.cos(x[3]))
        P  = F @ P @ F.T + ctrv_Q(dt)
        P  = (P + P.T) / 2 + np.eye(5) * 1e-8
        if obs_mask[k]:
            z = positions[k];  y = z - H5 @ x
            S = H5 @ P @ H5.T + R_meas
            K = P @ H5.T @ np.linalg.inv(S)
            x = x + K @ y
            x[2] = max(x[2], 0.0)
            x[3] = np.arctan2(np.sin(x[3]), np.cos(x[3]))
            P = (np.eye(5) - K @ H5) @ P
            P = (P + P.T) / 2 + np.eye(5) * 1e-8
        est.append([x[0], x[1]])
    return np.array(est)


def ukf_run_gapped(positions, dt_arr, obs_mask):
    """UKF at full resolution; update only where obs_mask[k] is True."""
    n = len(positions)
    x = np.array([positions[0,0], positions[0,1], SOG_MS[0], THETA_0, 0.0])
    P = np.diag([SIGMA_GPS**2, SIGMA_GPS**2, 2.0, 0.01, 0.0005])
    est = []
    for k in range(n):
        dt = max(dt_arr[k], 0.5)
        sps, Wm, Wc = sigma_points(x, P)
        sp_pred = np.array([ctrv_f(sp, dt) for sp in sps])
        x_pred  = ctrv_circular_mean(sp_pred, Wm)
        P_pred  = ctrv_Q(dt)
        for i, sp in enumerate(sp_pred):
            d    = sp - x_pred;  d[3] = np.arctan2(np.sin(d[3]), np.cos(d[3]))
            P_pred += Wc[i] * np.outer(d, d)
        P_pred = (P_pred + P_pred.T) / 2 + np.eye(5) * 1e-8
        x, P = x_pred, P_pred
        if obs_mask[k]:
            z_sigma = sp_pred[:, :2];  z_pred = Wm @ z_sigma
            Pzz = R_meas.copy();  Pxz = np.zeros((5, 2))
            for i, sp in enumerate(sp_pred):
                dz = z_sigma[i] - z_pred
                dx = sp - x_pred;  dx[3] = np.arctan2(np.sin(dx[3]), np.cos(dx[3]))
                Pzz += Wc[i] * np.outer(dz, dz)
                Pxz += Wc[i] * np.outer(dx, dz)
            K  = Pxz @ np.linalg.inv(Pzz)
            y  = positions[k] - z_pred
            x  = x_pred + K @ y
            x[2] = max(x[2], 0.0)
            x[3] = np.arctan2(np.sin(x[3]), np.cos(x[3]))
            P  = P_pred - K @ Pzz @ K.T
            P  = (P + P.T) / 2 + np.eye(5) * 1e-8
        est.append([x[0], x[1]])
    return np.array(est)


def pf_run_gapped(positions, dt_arr, obs_mask, n_par=N_PARTICLES, seed=SEED):
    """PF at full resolution; update only where obs_mask[k] is True."""
    rng_loc = np.random.default_rng(seed)
    n = len(positions)
    par = np.column_stack([
        rng_loc.normal(positions[0,0], SIGMA_GPS, n_par),
        rng_loc.normal(positions[0,1], SIGMA_GPS, n_par),
        np.clip(rng_loc.normal(SOG_MS[0], 1.0,   n_par), 0, None),
        rng_loc.normal(THETA_0,         0.1,      n_par),
        rng_loc.normal(0.0,             0.0002,   n_par),
    ])
    w   = np.ones(n_par) / n_par
    ROUGH_SIGMA = np.array([15.0, 15.0, 0.3, 0.01, 0.0002])
    est = []
    for k in range(n):
        dt = max(dt_arr[k], 0.5)
        for i in range(n_par):
            par[i] = ctrv_f(par[i], dt)
        t_q = min(dt, Q_DT_CAP)
        par[:, 0] += rng_loc.normal(0, 3.0  * np.sqrt(t_q), n_par)
        par[:, 1] += rng_loc.normal(0, 3.0  * np.sqrt(t_q), n_par)
        par[:, 2] += rng_loc.normal(0, 0.3  * np.sqrt(t_q), n_par)
        par[:, 4] += rng_loc.normal(0, 0.0002 * np.sqrt(t_q), n_par)
        par[:, 2]  = np.clip(par[:, 2], 0, None)
        if obs_mask[k]:
            z = positions[k]
            dx = par[:, 0] - z[0];  dy = par[:, 1] - z[1]
            log_w  = -0.5 * (dx**2 + dy**2) / SIGMA_GPS**2
            log_w -= log_w.max()
            w      = np.exp(log_w);  w /= w.sum()
            N_eff  = 1.0 / np.sum(w**2)
            if N_eff < n_par / 2:
                cumsum = np.cumsum(w)
                u   = (rng_loc.random() + np.arange(n_par)) / n_par
                idx = np.searchsorted(cumsum, u)
                par = par[idx].copy();  w = np.ones(n_par) / n_par
                par += rng_loc.normal(0, ROUGH_SIGMA, (n_par, 5))
                par[:, 2] = np.clip(par[:, 2], 0, None)
        x_est = w @ par
        est.append([x_est[0], x_est[1]])
    return np.array(est)


DOWNSAMPLE_FACTORS = [2, 3, 5, 10, 15]

rmse_table   = {f: [] for f in FILTERS}
mean_dt_vals = []

for step in DOWNSAMPLE_FACTORS:
    # obs_mask: True at every step-th index (filter sees those measurements)
    obs_mask = np.zeros(N, dtype=bool)
    obs_mask[::step] = True
    mean_dt_vals.append(float(np.mean(np.diff(times_s[obs_mask]))))

    kf_e  = kf_run_gapped(positions, dt_arr, obs_mask)
    ekf_e = ekf_run_gapped(positions, dt_arr, obs_mask)
    ukf_e = ukf_run_gapped(positions, dt_arr, obs_mask)
    pf_e  = pf_run_gapped(positions, dt_arr, obs_mask)

    # RMSE over ALL N steps vs full-resolution pseudo-truth
    for f, est in zip(FILTERS, [kf_e, ekf_e, ukf_e, pf_e]):
        err = np.linalg.norm(est - positions, axis=1)
        rmse_table[f].append(float(np.sqrt(np.mean(err**2))))

    print(f"  step={step:2d}  mean_dt={mean_dt_vals[-1]:.0f} s  "
          + "  ".join(f"{f}={rmse_table[f][-1]:.1f} m" for f in FILTERS))

# ── Figure 10: RMSE vs mean sampling interval ─────────────────────────────────
print("Figure 10: RMSE vs sampling interval …")
fig, ax = plt.subplots(figsize=(10, 6))
for f in FILTERS:
    ax.plot(mean_dt_vals, rmse_table[f],
            color=COLORS[f], ls=LSTYLE[f], marker="o", lw=2.0, ms=8, label=f)

ax.set_xlabel("Mean sampling interval (s)")
ax.set_ylabel("Pseudo-RMSE (m)")
ax.set_title("Pseudo-RMSE vs Sampling Interval\n"
             "(filters run at full rate, updates withheld at intermediate steps)")
ax.legend(fontsize=FS_LEG)
# light vertical guides at each downsampling rate
for step, mdt in zip(DOWNSAMPLE_FACTORS, mean_dt_vals):
    ax.axvline(mdt, color="#dddddd", lw=0.8, zorder=0)
    ax.text(mdt, ax.get_ylim()[1] * 0.97, f"×{step}",
            ha="center", va="top", fontsize=FS_LEG - 2, color="#555555")
plt.tight_layout()
plt.savefig("figures/ais_rmse_vs_downsampling.png")
plt.close()
print("  → figures/ais_rmse_vs_downsampling.png")

# ── Figure 11: Geographic view — full-res vs ×5 gapped + filter traces ────────
print("Figure 11: Downsampled track map …")
SHOW_STEP  = 5
obs_mask5  = np.zeros(N, dtype=bool);  obs_mask5[::SHOW_STEP] = True

# Convert to Mercator helper (defined once here, reused below)
def est_to_merc(est_xy):
    lons, lats = xy_to_latlon(est_xy[:, 0], est_xy[:, 1])
    return np.array([to_merc(lo, la) for lo, la in zip(lons, lats)])

kf_e5  = kf_run_gapped(positions, dt_arr, obs_mask5)
ekf_e5 = ekf_run_gapped(positions, dt_arr, obs_mask5)
ukf_e5 = ukf_run_gapped(positions, dt_arr, obs_mask5)
pf_e5  = pf_run_gapped(positions, dt_arr, obs_mask5)

est5 = {"KF": kf_e5, "EKF": ekf_e5, "UKF": ukf_e5, "PF": pf_e5}

# Mercator coords for the ×5 observation markers (used as filter inputs)
obs_pos5 = positions[obs_mask5]
mx_ds5, my_ds5 = zip(*[to_merc(*xy_to_latlon(p[0], p[1])) for p in obs_pos5])
mean_dt5 = int(np.mean(np.diff(times_s[obs_mask5])))

fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(mx_raw, my_raw, ".", ms=2.5, color="#cccccc", zorder=2,
        label=f"Full-res AIS (~66 s, pseudo-truth)")
ax.scatter(mx_ds5, my_ds5, s=40, color="black", zorder=5,
           label=f"Downsampled obs (×{SHOW_STEP}, ~{mean_dt5//60} min)")

x_span = x_hi - x_lo;  y_span = y_hi - y_lo
for f in FILTERS:
    m = est_to_merc(est5[f])
    in_bounds = (np.all(np.isfinite(m)) and
                 np.percentile(np.abs(m[:, 0] - (x_lo+x_hi)/2), 95) < x_span * 1.5 and
                 np.percentile(np.abs(m[:, 1] - (y_lo+y_hi)/2), 95) < y_span * 1.5)
    if in_bounds:
        ax.plot(m[:, 0], m[:, 1], color=COLORS[f], ls=LSTYLE[f],
                lw=1.8, alpha=0.85, label=f)
    else:
        ax.plot([], [], color=COLORS[f], ls=LSTYLE[f],
                lw=1.8, label=f + " (diverged)")

ax.set_xlim(x_lo, x_hi);  ax.set_ylim(y_lo, y_hi)
try:
    ctx.add_basemap(ax, crs="EPSG:3857",
                    source=ctx.providers.CartoDB.Positron, zoom=10)
except Exception:
    pass
ax.set_xticks([]); ax.set_yticks([])
ax.legend(fontsize=FS_LEG - 2, loc="upper left", framealpha=0.9, ncol=2)
ax.set_title(f"Filter Interpolation — ×{SHOW_STEP} Downsampled AIS — MAERSK SAIGON",
             fontsize=FS + 2)
plt.tight_layout()
plt.savefig("figures/ais_downsampled_track.png", bbox_inches="tight")
plt.close()
print("  → figures/ais_downsampled_track.png")

print("\nAll figures saved to figures/")
