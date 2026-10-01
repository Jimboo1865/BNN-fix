from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.special import kv
from scipy.stats import qmc
import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================================
# 1. CONFIGURATION & REPRODUCIBILITY
# =====================================================================
MODE = "publication"  # Use "student" for rapid testing, "publication" for production
SEED = 260210692

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT_DIR = Path("bnn_flux_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class RunConfig:
  mode: str
  y_train_min: float
  y_train_max: float
  y_report_min: float
  y_report_max: float
  b_min_fm: float
  b_hard_max_fm: float
  xi_max: float
  prior_widths: float
  n_r: int
  n_phi: int
  n_z: int
  n_norm: int
  tail_widths: float
  n_train: int
  n_val: int
  n_test: int
  hidden_units: tuple[int, ...]
  dropout_rate: float
  learning_rate: float
  batch_size: int
  epochs: int
  patience: int
  mc_passes: int
  run_sample_size_study: bool
  sample_size_study_sizes: tuple[int, ...]
  sample_size_study_epochs: int
  sample_size_mc_passes: int


if MODE == "student":
  CFG = RunConfig(
      mode=MODE,
      y_train_min=-4.5,
      y_train_max=4.5,
      y_report_min=-4.0,
      y_report_max=4.0,
      b_min_fm=14.0,
      b_hard_max_fm=300_000.0,
      xi_max=8.0,
      prior_widths=3.0,
      n_r=32,
      n_phi=48,
      n_z=48,
      n_norm=80,
      tail_widths=12.0,
      n_train=512,
      n_val=256,
      n_test=512,
      hidden_units=(128, 128, 64),
      dropout_rate=0.05,
      learning_rate=1.0e-3,
      batch_size=64,
      epochs=150,
      patience=25,
      mc_passes=40,
      run_sample_size_study=True,
      sample_size_study_sizes=(128, 256, 512),
      sample_size_study_epochs=150,
      sample_size_mc_passes=30,
  )
else:
  CFG = RunConfig(
      mode=MODE,
      y_train_min=-4.5,
      y_train_max=4.5,
      y_report_min=-4.0,
      y_report_max=4.0,
      b_min_fm=14.0,
      b_hard_max_fm=300_000.0,
      xi_max=8.0,
      prior_widths=3.0,
      n_r=48,
      n_phi=72,
      n_z=64,
      n_norm=120,
      tail_widths=14.0,
      n_train=8192,
      n_val=2048,
      n_test=4096,
      hidden_units=(256, 256, 128, 64),
      dropout_rate=0.05,
      learning_rate=5.0e-4,
      batch_size=256,
      epochs=800,
      patience=80,
      mc_passes=200,
      run_sample_size_study=True,
      sample_size_study_sizes=(1024, 2048, 4096, 8192, 16384, 32768, 65536),
      sample_size_study_epochs=800,
      sample_size_mc_passes=80,
  )

# =====================================================================
# 2. PHYSICS CONSTANTS & KINEMATICS
# =====================================================================
ALPHA_EM = 1.0 / 137.035999084
Z_ION = 82
GAMMA_BEAM = 2950.0
HBARC_GEV_FM = 0.1973269804
M_JPSI_GEV = 3.0969

R0_BASE_FM = 6.62
DELTA_R_MEAN_FM = 0.283
DELTA_R_SIGMA_FM = 0.071
A_MEAN_FM = 0.546
A_SIGMA_FM = 0.015

DELTA_R_MIN_FM = DELTA_R_MEAN_FM - CFG.prior_widths * DELTA_R_SIGMA_FM
DELTA_R_MAX_FM = DELTA_R_MEAN_FM + CFG.prior_widths * DELTA_R_SIGMA_FM
A_MIN_FM = A_MEAN_FM - CFG.prior_widths * A_SIGMA_FM
A_MAX_FM = A_MEAN_FM + CFG.prior_widths * A_SIGMA_FM


def rapidity_to_omega(y):
  return 0.5 * M_JPSI_GEV * np.exp(np.asarray(y, dtype=float))


def omega_to_rapidity(omega_gev):
  return np.log(2.0 * np.asarray(omega_gev, dtype=float) / M_JPSI_GEV)


LOG_OMEGA_TRAIN_MIN = math.log(float(rapidity_to_omega(CFG.y_train_min)))
LOG_OMEGA_TRAIN_MAX = math.log(float(rapidity_to_omega(CFG.y_train_max)))
LOG_B_MIN = math.log(CFG.b_min_fm)
LOG_B_HARD_MAX = math.log(CFG.b_hard_max_fm)


def b_max_for_omega(omega_gev, *, xi_max=None, hard_max_fm=None):
  xi_max = CFG.xi_max if xi_max is None else xi_max
  hard_max_fm = CFG.b_hard_max_fm if hard_max_fm is None else hard_max_fm
  omega = np.asarray(omega_gev, dtype=float)
  kinematic_max = xi_max * GAMMA_BEAM * HBARC_GEV_FM / omega
  return np.maximum(CFG.b_min_fm, np.minimum(hard_max_fm, kinematic_max))


FEATURE_COLUMNS = ["log_omega", "log_b", "delta_r_fm", "a_fm"]
FEATURE_DOMAIN = {
    "log_omega": (LOG_OMEGA_TRAIN_MIN, LOG_OMEGA_TRAIN_MAX),
    "log_b": (LOG_B_MIN, LOG_B_HARD_MAX),
    "delta_r_fm": (DELTA_R_MIN_FM, DELTA_R_MAX_FM),
    "a_fm": (A_MIN_FM, A_MAX_FM),
}


# =====================================================================
# 3. NUMERICAL INTEGRATORS & ANALYTICAL EDFF
# =====================================================================
def gauss_legendre_interval(lower: float, upper: float, n_points: int):
  nodes, weights = leggauss(n_points)
  mapped_nodes = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
  mapped_weights = 0.5 * (upper - lower) * weights
  return mapped_nodes, mapped_weights


def woods_saxon_shape(r_fm, r0_fm: float, a_fm: float):
  exponent = np.clip((np.asarray(r_fm) - r0_fm) / a_fm, -700.0, 700.0)
  return 1.0 / (1.0 + np.exp(exponent))


def woods_saxon_3d_normalisation(r0_fm: float, a_fm: float):
  r_max = r0_fm + CFG.tail_widths * a_fm
  r_nodes, r_weights = gauss_legendre_interval(0.0, r_max, CFG.n_norm)
  return float(
      4.0
      * np.pi
      * np.sum(
          r_weights * r_nodes**2 * woods_saxon_shape(r_nodes, r0_fm, a_fm)
      )
  )


def projected_density_at_nodes(transverse_r_fm, r0_fm: float, a_fm: float):
  z_max = r0_fm + CFG.tail_widths * a_fm
  z_nodes, z_weights = gauss_legendre_interval(0.0, z_max, CFG.n_z)
  norm_3d = woods_saxon_3d_normalisation(r0_fm, a_fm)
  radius_3d = np.sqrt(
      np.asarray(transverse_r_fm)[:, None] ** 2 + z_nodes[None, :] ** 2
  )
  rho_3d = woods_saxon_shape(radius_3d, r0_fm, a_fm) / norm_3d
  return 2.0 * np.sum(z_weights[None, :] * rho_3d, axis=1)


def analytical_edff_flux(omega_gev, b_fm):
  b = np.maximum(np.asarray(b_fm, dtype=float), 1.0e-10)
  omega = np.asarray(omega_gev, dtype=float)
  xi = omega * b / (GAMMA_BEAM * HBARC_GEV_FM)
  prefactor = (Z_ION**2 * ALPHA_EM) / (np.pi**2 * b**2) * xi**2
  return prefactor * (kv(1, xi) ** 2 + kv(0, xi) ** 2 / GAMMA_BEAM**2)


def finite_size_epa_flux(
    omega_gev: float, b_fm: float, delta_r_fm: float, a_fm: float
):
  r0_fm = float(R0_BASE_FM + delta_r_fm)
  r_max = r0_fm + CFG.tail_widths * a_fm
  r_nodes, r_weights = gauss_legendre_interval(0.0, r_max, CFG.n_r)
  phi_nodes, phi_weights = gauss_legendre_interval(0.0, 2.0 * np.pi, CFG.n_phi)

  rho_2d = projected_density_at_nodes(r_nodes, r0_fm, a_fm)
  distance = np.sqrt(
      b_fm**2
      + r_nodes[:, None] ** 2
      + 2.0 * b_fm * r_nodes[:, None] * np.cos(phi_nodes)[None, :]
  )
  angular_integral = np.sum(
      phi_weights[None, :] * analytical_edff_flux(omega_gev, distance), axis=1
  )
  return float(np.sum(r_weights * r_nodes * rho_2d * angular_integral))


# =====================================================================
# 4. SAMPLING, SCALING & DATASET GENERATION
# =====================================================================
def require_power_of_two(n: int) -> int:
  exponent = int(round(math.log2(n)))
  if 2**exponent != n:
    raise ValueError(f"Sobol sample size must be a power of two, received {n}.")
  return exponent


def sample_physics_domain(
    n: int, seed: int, *, y_min: float, y_max: float
) -> pd.DataFrame:
  exponent = require_power_of_two(n)
  unit = qmc.Sobol(d=4, scramble=True, seed=seed).random_base2(exponent)
  log_omega_min = math.log(float(rapidity_to_omega(y_min)))
  log_omega_max = math.log(float(rapidity_to_omega(y_max)))
  log_omega = log_omega_min + unit[:, 0] * (log_omega_max - log_omega_min)
  omega = np.exp(log_omega)
  log_b_upper = np.log(b_max_for_omega(omega))
  log_b = LOG_B_MIN + unit[:, 1] * (log_b_upper - LOG_B_MIN)
  delta_r = DELTA_R_MIN_FM + unit[:, 2] * (DELTA_R_MAX_FM - DELTA_R_MIN_FM)
  a_values = A_MIN_FM + unit[:, 3] * (A_MAX_FM - A_MIN_FM)
  return pd.DataFrame({
      "log_omega": log_omega,
      "log_b": log_b,
      "delta_r_fm": delta_r,
      "a_fm": a_values,
  })


def transform_features(frame: pd.DataFrame) -> np.ndarray:
  values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
  lower = np.array(
      [FEATURE_DOMAIN[col][0] for col in FEATURE_COLUMNS], dtype=np.float32
  )
  upper = np.array(
      [FEATURE_DOMAIN[col][1] for col in FEATURE_COLUMNS], dtype=np.float32
  )
  return (2.0 * (values - lower) / (upper - lower) - 1.0).astype(np.float32)


def evaluate_direct_flux(frame: pd.DataFrame) -> np.ndarray:
  flux_values = [
      finite_size_epa_flux(
          omega_gev=float(np.exp(row.log_omega)),
          b_fm=float(np.exp(row.log_b)),
          delta_r_fm=float(row.delta_r_fm),
          a_fm=float(row.a_fm),
      )
      for row in frame.itertuples(index=False)
  ]
  return np.asarray(flux_values, dtype=np.float64)


def edff_flux_for_frame(frame: pd.DataFrame) -> np.ndarray:
  omega = np.exp(frame["log_omega"].to_numpy(dtype=float))
  b_fm = np.exp(frame["log_b"].to_numpy(dtype=float))
  return np.asarray(analytical_edff_flux(omega, b_fm), dtype=np.float64)


@dataclass
class StandardScaler1D:
  mean: float
  std: float

  @classmethod
  def fit(cls, values: np.ndarray) -> "StandardScaler1D":
    values = np.asarray(values, dtype=float)
    return cls(mean=float(values.mean()), std=float(values.std()))

  def transform(self, values: np.ndarray) -> np.ndarray:
    return ((np.asarray(values) - self.mean) / self.std).astype(np.float32)

  def inverse(self, values: np.ndarray) -> np.ndarray:
    return self.mean + self.std * np.asarray(values)


print("--- Sampling and calculating numerical flux labels ---")
train_frame = sample_physics_domain(
    CFG.n_train, SEED + 1, y_min=CFG.y_train_min, y_max=CFG.y_train_max
)
val_frame = sample_physics_domain(
    CFG.n_val, SEED + 2, y_min=CFG.y_report_min, y_max=CFG.y_report_max
)
test_frame = sample_physics_domain(
    CFG.n_test, SEED + 3, y_min=CFG.y_report_min, y_max=CFG.y_report_max
)

train_flux = evaluate_direct_flux(train_frame)
val_flux = evaluate_direct_flux(val_frame)
test_flux = evaluate_direct_flux(test_frame)

X_train = transform_features(train_frame)
X_val = transform_features(val_frame)
X_test = transform_features(test_frame)

log_correction_train = np.log(train_flux) - np.log(
    edff_flux_for_frame(train_frame)
)
log_correction_val = np.log(val_flux) - np.log(edff_flux_for_frame(val_frame))
log_correction_test = np.log(test_flux) - np.log(
    edff_flux_for_frame(test_frame)
)

target_scaler = StandardScaler1D.fit(log_correction_train)
y_train = target_scaler.transform(log_correction_train)
y_val = target_scaler.transform(log_correction_val)


# =====================================================================
# 5. PYTORCH MODEL ARCHITECTURE & UTILITIES
# =====================================================================
class PyTorchSurrogate(nn.Module):

  def __init__(
      self, hidden_units=(128, 128, 64), dropout_rate=0.05, seed_offset=0
  ):
    super().__init__()
    torch.manual_seed(SEED + seed_offset)
    layers_list = []
    in_dim = 4
    for units in hidden_units:
      layers_list.append(nn.Linear(in_dim, units))
      layers_list.append(nn.SiLU())  # PyTorch equivalent of swish
      layers_list.append(nn.Dropout(p=dropout_rate))
      in_dim = units
    layers_list.append(nn.Linear(in_dim, 1))
    self.network = nn.Sequential(*layers_list)

  def forward(self, x):
    return self.network(x)


def train_pytorch_model(
    model,
    X_tr,
    y_tr,
    X_v,
    y_v,
    epochs,
    batch_size,
    learning_rate,
    patience,
):
  criterion = nn.HuberLoss(delta=1.0)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
  )

  X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
  y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
  X_v_t = torch.tensor(X_v, dtype=torch.float32)
  y_v_t = torch.tensor(y_v, dtype=torch.float32).unsqueeze(1)

  dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
  loader = torch.utils.data.DataLoader(
      dataset, batch_size=batch_size, shuffle=True
  )

  best_val_loss = float("inf")
  best_weights = None
  no_improve = 0

  for epoch in range(epochs):
    model.train()
    for batch_x, batch_y in loader:
      optimizer.zero_grad()
      out = model(batch_x)
      loss = criterion(out, batch_y)
      loss.backward()
      optimizer.step()

    model.eval()
    with torch.no_grad():
      val_out = model(X_v_t)
      val_loss = criterion(val_out, y_v_t).item()

    scheduler.step(val_loss)

    if val_loss < best_val_loss - 1e-6:
      best_val_loss = val_loss
      best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
      no_improve = 0
    else:
      no_improve += 1
      if no_improve >= patience:
        break

  if best_weights is not None:
    model.load_state_dict(best_weights)


def predict_log_correction_deterministic(
    fitted_model, features, scaler, batch_size=4096
):
  fitted_model.eval()
  features_t = torch.tensor(features, dtype=torch.float32)
  with torch.no_grad():
    scaled = fitted_model(features_t).numpy().reshape(-1)
  return scaler.inverse(scaled)


def predict_flux_deterministic(
    fitted_model, frame, scaler, batch_size=4096
):
  features = transform_features(frame)
  log_correction = predict_log_correction_deterministic(
      fitted_model, features, scaler, batch_size=batch_size
  )
  return edff_flux_for_frame(frame) * np.exp(log_correction)


def residual_summary(direct_flux, predicted_flux):
  relative = predicted_flux / direct_flux - 1.0
  absolute_relative = np.abs(relative)
  return {
      "median_absolute_relative": np.median(absolute_relative),
      "p95_absolute_relative": np.quantile(absolute_relative, 0.95),
      "p99_absolute_relative": np.quantile(absolute_relative, 0.99),
  }


def mc_dropout_log_correction_samples(
    fitted_model, features, scaler, *, n_passes
):
  # Keep dropout active during inference for MC-Dropout uncertainty
  fitted_model.train()
  features_t = torch.tensor(features, dtype=torch.float32)
  samples = []
  with torch.no_grad():
    for _ in range(n_passes):
      scaled_pred = fitted_model(features_t).numpy().reshape(-1)
      samples.append(scaler.inverse(scaled_pred))
  return np.asarray(samples)


def fit_dropout_spread_scale(true_log_correction, log_correction_samples):
  mean = log_correction_samples.mean(axis=0)
  variance = log_correction_samples.var(axis=0, ddof=1)
  valid = variance > 1.0e-12
  if not np.any(valid):
    return 1.0
  scale = np.sqrt(
      np.mean((true_log_correction[valid] - mean[valid]) ** 2 / variance[valid])
  )
  return float(scale)


def scale_mc_samples(samples, factor):
  centre = samples.mean(axis=0, keepdims=True)
  return centre + factor * (samples - centre)


def correction_samples_to_flux(frame, log_correction_samples):
  log_edff = np.log(edff_flux_for_frame(frame))
  return np.exp(log_correction_samples + log_edff[None, :])


# =====================================================================
# 6. TRAIN PRIMARY MODEL
# =====================================================================
print("--- Training primary surrogate (PyTorch) ---")
model = PyTorchSurrogate(
    hidden_units=CFG.hidden_units,
    dropout_rate=CFG.dropout_rate,
    seed_offset=0,
)
start = time.perf_counter()
train_pytorch_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=CFG.epochs,
    batch_size=CFG.batch_size,
    learning_rate=CFG.learning_rate,
    patience=CFG.patience,
)
TRAINING_SECONDS = time.perf_counter() - start
print(f"Primary model trained in {TRAINING_SECONDS:.2f} s")


# =====================================================================
# 7. PART 13: SAMPLE SIZE STUDY & PLOTS
# =====================================================================
def train_subset_model(n_train_subset: int, *, seed_offset: int):
  subset_scaler = StandardScaler1D.fit(log_correction_train[:n_train_subset])
  subset_model = PyTorchSurrogate(
      hidden_units=CFG.hidden_units,
      dropout_rate=CFG.dropout_rate,
      seed_offset=seed_offset,
  )
  t0 = time.perf_counter()
  train_pytorch_model(
      subset_model,
      X_train[:n_train_subset],
      subset_scaler.transform(log_correction_train[:n_train_subset]),
      X_val,
      subset_scaler.transform(log_correction_val),
      epochs=CFG.sample_size_study_epochs,
      batch_size=CFG.batch_size,
      learning_rate=CFG.learning_rate,
      patience=CFG.patience,
  )
  return subset_model, subset_scaler, (time.perf_counter() - t0)


sample_size_rows = []
print("--- Starting Part 13: Sobol sample-size study ---")

for sample_size in CFG.sample_size_study_sizes:
  print(f"Evaluating subset size N = {sample_size}...")
  if sample_size == CFG.n_train:
    subset_model = model
    subset_scaler = target_scaler
    t_sec = TRAINING_SECONDS
  else:
    subset_model, subset_scaler, t_sec = train_subset_model(
        sample_size, seed_offset=0
    )

  subset_prediction = predict_flux_deterministic(
      subset_model, test_frame, subset_scaler
  )
  summary = residual_summary(test_flux, subset_prediction)

  val_samples_raw = mc_dropout_log_correction_samples(
      subset_model, X_val, subset_scaler, n_passes=CFG.sample_size_mc_passes
  )
  subset_scale = fit_dropout_spread_scale(log_correction_val, val_samples_raw)

  test_samples_raw = mc_dropout_log_correction_samples(
      subset_model, X_test, subset_scaler, n_passes=CFG.sample_size_mc_passes
  )
  test_samples_calibrated = scale_mc_samples(test_samples_raw, subset_scale)
  flux_samples = correction_samples_to_flux(
      test_frame, test_samples_calibrated
  )
  relative_epistemic = flux_samples.std(axis=0, ddof=1) / flux_samples.mean(
      axis=0
  )

  sample_size_rows.append({
      "n_train": sample_size,
      "median_absolute_relative": summary["median_absolute_relative"],
      "p95_absolute_relative": summary["p95_absolute_relative"],
      "p99_absolute_relative": summary["p99_absolute_relative"],
      "median_epistemic_relative": np.median(relative_epistemic),
      "p90_epistemic_relative": np.quantile(relative_epistemic, 0.90),
      "dropout_spread_scale": subset_scale,
      "training_seconds": t_sec,
  })

sample_size_table = pd.DataFrame(sample_size_rows)
print("\n--- Summary Table ---")
print(sample_size_table.to_string(index=False))

# --- GENERATE PLOTS ---
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Row 0: Log-Scale
axes[0, 0].plot(
    sample_size_table["n_train"],
    100.0 * sample_size_table["median_absolute_relative"],
    "o-",
    label="Median absolute residual",
)
axes[0, 0].plot(
    sample_size_table["n_train"],
    100.0 * sample_size_table["p95_absolute_relative"],
    "s--",
    label="95th-percentile residual",
)
axes[0, 0].set_xscale("log", base=2)
axes[0, 0].set_yscale("log")
axes[0, 0].set_xlabel("Sobol training points")
axes[0, 0].set_ylabel("Held-out residual [%] (log)")
axes[0, 0].set_title("Accuracy vs. Sample Size (Log Scale)")
axes[0, 0].legend()
axes[0, 0].grid(True, which="both", ls=":")

axes[0, 1].plot(
    sample_size_table["n_train"],
    100.0 * sample_size_table["median_epistemic_relative"],
    "o-",
    label="Median epistemic uncertainty",
)
axes[0, 1].plot(
    sample_size_table["n_train"],
    100.0 * sample_size_table["p90_epistemic_relative"],
    "s--",
    label="90th-percentile epistemic uncertainty",
)
axes[0, 1].set_xscale("log", base=2)
axes[0, 1].set_yscale("log")
axes[0, 1].set_xlabel("Sobol training points")
axes[0, 1].set_ylabel("Calibrated epistemic uncertainty [%] (log)")
axes[0, 1].set_title("Epistemic Uncertainty vs. Sample Size (Log Scale)")
axes[0, 1].legend()
axes[0, 1].grid(True, which="both", ls=":")

# Row 1: Linear-Scale
axes[1, 0].plot(
    sample_size_table["n_train"],
    100.0 * sample_size_table["median_absolute_relative"],
    "o-",
    label="Median absolute residual",
)
axes[1, 0].plot(
    sample_size_table["n_train"],
    100.0 * sample_size_table["p95_absolute_relative"],
    "s--",
    label="95th-percentile residual",
)
axes[1, 0].set_xscale("log", base=2)
axes[1, 0].set_yscale("linear")
axes[1, 0].set_ylim(bottom=0)
axes[1, 0].set_xlabel("Sobol training points")
axes[1, 0].set_ylabel("Held-out residual [%] (linear)")
axes[1, 0].set_title("Accuracy vs. Sample Size (Linear Scale)")
axes[1, 0].legend()
axes[1, 0].grid(True, ls=":")

axes[1, 1].plot(
    sample_size_table["n_train"],
    100.0 * sample_size_table["median_epistemic_relative"],
    "o-",
    label="Median epistemic uncertainty",
)
axes[1, 1].plot(
    sample_size_table["n_train"],
    100.0 * sample_size_table["p90_epistemic_relative"],
    "s--",
    label="90th-percentile epistemic uncertainty",
)
axes[1, 1].set_xscale("log", base=2)
axes[1, 1].set_yscale("linear")
axes[1, 1].set_ylim(bottom=0)
axes[1, 1].set_xlabel("Sobol training points")
axes[1, 1].set_ylabel("Calibrated epistemic uncertainty [%] (linear)")
axes[1, 1].set_title("Epistemic Uncertainty vs. Sample Size (Linear Scale)")
axes[1, 1].legend()
axes[1, 1].grid(True, ls=":")

plt.tight_layout()
plt.show()