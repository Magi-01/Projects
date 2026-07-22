"""
SDSS DR17 Image Classification Pipeline  —  FITS Tensors
=========================================================

Image-only classification of SDSS DR17 objects into {GALAXY, QSO, STAR},
using preprocessed FITS cutouts saved as .npy tensors (single-channel
intensity maps, shape (H, W) or (1, H, W)).

The CSV is consumed ONLY by the Exploratory Data Analysis stage — no
tabular features are passed to any model. All classifiers see images only.

Pipeline
--------
Stage 0a.  EDA                        — class balance, photometry, sky distribution
Stage 0b.  Image-tensor diagnostics    — pixel statistics straight from the .npy files
Stage 1.   CNN baseline               — supervised image classifier
Stage 2.   CAE                        — Convolutional Autoencoder (deterministic latent)
Stage 3.   VAE                        — Variational Autoencoder (probabilistic latent)
Stage 4.   Fused-CAE                  — CNN feats ⊕ frozen CAE latent → classifier
Stage 5.   Fused-VAE                  — CNN feats ⊕ frozen VAE μ-latent → classifier
Stage 6.   Comparison + significance  — CNN vs CAE, CNN vs VAE, CAE vs VAE

Every section is annotated with the mathematical foundation so the code
doubles as a study reference.

──────────────────────────────────────────────────────────────────────────────
WHY THIS DESIGN — high-level proof sketch
──────────────────────────────────────────────────────────────────────────────
Let X be the image, Y∈{0,1,2} the class. We want q_θ(Y|X) ≈ p(Y|X).
A pure CNN learns q directly. An autoencoder learns a representation
φ(X) such that X ≈ ψ(φ(X)); the bottleneck forces φ to capture the
information-dense subspace of X. Concatenating φ(X) with the CNN's
discriminative features gives the classifier two complementary views:

  (i)  discriminative   — features tuned to separate classes (CNN)
  (ii) generative       — features tuned to reconstruct X (AE / VAE)

The classifier head can then exploit signals that the supervised loss
alone would have failed to amplify (e.g. fine morphology relevant to
reconstruction but only weakly correlated with the loss gradient).
For the VAE specifically, the KL term regularizes the latent toward
N(0,I), which by aggregation gives a smooth, scale-calibrated latent
geometry — empirically easier to classify on than a raw CAE latent.
──────────────────────────────────────────────────────────────────────────────

Debug
-----
DEBUG             = True   → ipdb breakpoints at key checkpoints
BREAK_AFTER_BATCH = True   → stop after first batch (shape/device sanity)
"""

# ── Stdlib ────────────────────────────────────────────────────────────────────
import os, sys, time, json, random, warnings, math, re
from pathlib import Path
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    cohen_kappa_score, matthews_corrcoef, roc_auc_score,
    balanced_accuracy_score, log_loss, top_k_accuracy_score,
)
from sklearn.decomposition import PCA

try:
    import ipdb as pdb
except ImportError:
    import pdb


# ═════════════════════════════════════════════════════════════════════════════
# 0.  CONFIG
# ═════════════════════════════════════════════════════════════════════════════
DEBUG             = False          # ipdb breakpoints at key checkpoints
BREAK_AFTER_BATCH = False          # one-batch dry run

ROOT       = os.getcwd()
BASE_DIR   = os.path.join(ROOT, "sdss_dr17_dataset_fits/tensors")
CSV_PATH   = os.path.join(ROOT, "SkyObjects_FITS.csv")

OUT_DIR    = "/media/Ubunt_2/Project/pipeline_output_FITS"
CKPT_DIR   = os.path.join(OUT_DIR, "checkpoints_FITS")
EDA_DIR    = os.path.join(OUT_DIR, "eda")
METRIC_DIR = os.path.join(OUT_DIR, "metrics")
for d in (OUT_DIR, CKPT_DIR, EDA_DIR, METRIC_DIR):
    os.makedirs(d, exist_ok=True)

CLASSES    = ["GALAXY", "QSO", "STAR"]
N_CLASSES  = len(CLASSES)
CLASS2IDX  = {c: i for i, c in enumerate(CLASSES)}
PALETTE    = {"GALAXY": "#4C82C4", "QSO": "#E07B53", "STAR": "#5FAD56"}

IMG_SIZE   = 128
LATENT_DIM = 128

BATCH      = 64                    # image-only → larger batch is fine
EPOCHS_CNN = 10
EPOCHS_CAE = 10
EPOCHS_VAE = 10
EPOCHS_FUS = 20
LR         = 1e-4
DROPOUT    = 0.3
WEIGHT_DECAY = 1e-5
GRAD_CLIP  = 1.0
SEED       = 42

# β-VAE coefficient + KL annealing.
# β trades reconstruction (small β) vs. latent regularisation (large β).
# Annealing from 0 → β_max stops the KL term from collapsing the encoder
# in the very early epochs (a well-known VAE failure mode called
# "posterior collapse", where q(z|x) becomes the prior N(0, I) for all x
# and the decoder learns to ignore z entirely).
VAE_BETA_MAX          = 1.0
VAE_KL_ANNEAL_EPOCHS  = 10

# Class weighting — used in CE loss to compensate imbalance. Final values
# are filled in after EDA from the training-split frequencies.
CLASS_WEIGHTS_OVERRIDE = None      # set to a list to override; None → auto

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

print(f"[config] device={DEVICE}  AMP={USE_AMP}  latent={LATENT_DIM}  batch={BATCH}")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True   # speed when input shape is fixed


# ═════════════════════════════════════════════════════════════════════════════
# 1.  UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def debug_break(label: str):
    if DEBUG:
        print(f"\n[DEBUG] breakpoint → {label}")
        pdb.set_trace()


def _safe(label: str) -> str:
    """Filename-safe label."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", label)


def ckpt_path(mode: str, label: str, epoch: int) -> str:
    return os.path.join(CKPT_DIR, f"ckpt_{mode}_{_safe(label)}_ep{epoch:03d}.pth")


def save_json(obj, path):
    """Write JSON, coercing numpy scalars/arrays to native types."""
    def _coerce(o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        if isinstance(o, dict):           return {k: _coerce(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):  return [_coerce(x) for x in o]
        return o
    with open(path, "w") as f:
        json.dump(_coerce(obj), f, indent=2)
    print(f"[save] {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 2.  DATASET
# ═════════════════════════════════════════════════════════════════════════════
#
# The .npy files are single-channel intensity maps derived from FITS
# cutouts. We apply *per-image* z-score normalisation:
#
#     x̂ = (x - μ_img) / (σ_img + ε)
#
# Why per-image (not global)?
#   FITS frames vary in zero-point and exposure. A global mean/std would
#   bake exposure variability into the inputs as informative noise.
#   Per-image normalisation removes the additive/multiplicative gain
#   ambiguity and lets the CNN focus on relative spatial structure —
#   which is what carries the morphological information.
#
# ─────────────────────────────────────────────────────────────────────────────

class SDSSImageDataset(Dataset):
    """
    Returns (img, label_int, filepath).
    img : float32 tensor of shape (1, H, W).
    """
    def __init__(self, records: pd.DataFrame, augment: bool = False):
        self.records = records.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        # Replace any NaN/Inf left over from FITS reduction.
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        mu  = arr.mean()
        sd  = arr.std()
        return (arr - mu) / (sd + 1e-6)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        img_path = row["filepath"]

        arr = np.load(img_path)
        arr = self._normalize(arr)

        img = torch.from_numpy(arr).float()
        if img.ndim == 2:
            img = img.unsqueeze(0)           # (1, H, W)

        # Resize if necessary so the CNN's 4× downsampling lands exactly.
        if img.shape[-1] != IMG_SIZE or img.shape[-2] != IMG_SIZE:
            img = F.interpolate(img.unsqueeze(0),
                                size=(IMG_SIZE, IMG_SIZE),
                                mode="bilinear", align_corners=False).squeeze(0)

        if self.augment:
            # Astronomical images are rotation+flip invariant — exploit it.
            if random.random() < 0.5:
                img = torch.flip(img, dims=[-1])
            if random.random() < 0.5:
                img = torch.flip(img, dims=[-2])
            k = random.randint(0, 3)
            if k:
                img = torch.rot90(img, k=k, dims=[-2, -1])

        label = CLASS2IDX[row["class"]]
        return img, label, img_path


def build_dataframe(csv_path: str) -> pd.DataFrame:
    """Read CSV, attach filepaths, keep only rows whose tensor exists on disk."""
    df = pd.read_csv(csv_path)

    def _path(row):
        return os.path.join(BASE_DIR, f"{row['objid']}_{row['class']}.npy")
    df["filepath"] = df.apply(_path, axis=1)

    before = len(df)
    df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)
    print(f"[data] {len(df)}/{before} tensors found on disk")

    df = df[df["class"].isin(CLASSES)].reset_index(drop=True)
    return df


def stratified_split(df: pd.DataFrame, val=0.10, test=0.10):
    """70/20/10 stratified split — proportions are reproducible from SEED."""
    labels = df["class"].values
    idx    = np.arange(len(df))

    idx_tv, idx_test = train_test_split(
        idx, test_size=test, stratify=labels, random_state=SEED)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=val / (1.0 - test), stratify=labels[idx_tv],
        random_state=SEED)

    print(f"[data] train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")
    return idx_train, idx_val, idx_test


def build_dataloaders(df: pd.DataFrame):
    idx_train, idx_val, idx_test = stratified_split(df)

    train_ds = SDSSImageDataset(df.iloc[idx_train], augment=True)
    val_ds   = SDSSImageDataset(df.iloc[idx_val],   augment=False)
    test_ds  = SDSSImageDataset(df.iloc[idx_test],  augment=False)

    kw = dict(num_workers=4, pin_memory=torch.cuda.is_available(),
              persistent_workers=True)

    return (
        DataLoader(train_ds, batch_size=BATCH, shuffle=True,  drop_last=True,  **kw),
        DataLoader(val_ds,   batch_size=BATCH, shuffle=False, drop_last=False, **kw),
        DataLoader(test_ds,  batch_size=BATCH, shuffle=False, drop_last=False, **kw),
        df.iloc[idx_train], df.iloc[idx_val], df.iloc[idx_test],
    )


# ═════════════════════════════════════════════════════════════════════════════
# 3.  EXPLORATORY DATA ANALYSIS  (before any model training)
# ═════════════════════════════════════════════════════════════════════════════
#
# Why EDA matters mathematically
# ──────────────────────────────
# For any classifier f: X → Y, the Bayes-optimal error is
#
#     R*  =  E_X [ 1 - max_y p(y | X) ]
#
# bounded below by the marginal max:
#
#     R*  ≥  1 - max_y p(y)        (achieved by the constant predictor).
#
# So if classes are 95-3-2, a constant predictor gets 0.95 accuracy.
# Accuracy alone is therefore meaningless without knowing p(y); EDA
# tells us what "doing nothing" already buys.
#
# EDA also surfaces:
#   • outliers that will dominate the loss gradient
#   • class overlap in feature space (Bayes error proxy)
#   • coverage gaps in (RA, Dec) that hurt generalisation
# ─────────────────────────────────────────────────────────────────────────────

def eda_class_distribution(df: pd.DataFrame, save_path: str) -> dict:
    counts = df["class"].value_counts().reindex(CLASSES, fill_value=0)
    total  = counts.sum()
    fracs  = counts / total

    # Shannon entropy H = -Σ p log p  (max log(K) when uniform)
    p   = fracs.values
    H   = -(p * np.log(p + 1e-12)).sum()
    Hmx = math.log(N_CLASSES)
    norm_entropy = H / Hmx          # 1 = perfectly balanced, 0 = all-one-class

    imbalance_ratio = counts.max() / max(counts.min(), 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[c] for c in counts.index], edgecolor="black")
    for b, v, fr in zip(bars, counts.values, fracs.values):
        ax.text(b.get_x() + b.get_width()/2, v + total*0.005,
                f"{v}\n({fr:.1%})", ha="center", fontsize=9)
    ax.set_title(f"Class distribution  —  H/H_max = {norm_entropy:.3f},  "
                 f"imbalance ratio = {imbalance_ratio:.1f}×")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[plot] {save_path}")

    return {
        "counts":            counts.to_dict(),
        "fractions":         fracs.to_dict(),
        "normalized_entropy": float(norm_entropy),
        "imbalance_ratio":   float(imbalance_ratio),
        "baseline_acc":      float(fracs.max()),
    }


def eda_photometry(df: pd.DataFrame, save_path: str) -> dict:
    """
    Per-band magnitude + colour distributions. Even though the models do
    NOT use photometry, this is a useful sanity check: if classes are
    *also* separable in photometry, the image features should at least
    not be worse than photometric classification.
    """
    bands  = [b for b in ["u", "g", "r", "i", "z"] if b in df.columns]
    colors = []
    if {"u", "g"}.issubset(df.columns): df["u_g"] = df["u"] - df["g"]; colors.append("u_g")
    if {"g", "r"}.issubset(df.columns): df["g_r"] = df["g"] - df["r"]; colors.append("g_r")
    if {"r", "i"}.issubset(df.columns): df["r_i"] = df["r"] - df["i"]; colors.append("r_i")
    if {"i", "z"}.issubset(df.columns): df["i_z"] = df["i"] - df["z"]; colors.append("i_z")

    plot_cols = bands + colors
    if not plot_cols:
        return {"skipped": True, "reason": "no photometric bands in CSV"}

    n_cols = 5
    n_rows = math.ceil(len(plot_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.4, n_rows*3))
    axes = np.atleast_2d(axes)

    summary = {}
    for ax, col in zip(axes.flat, plot_cols):
        data = [df[df["class"] == c][col].dropna().values for c in CLASSES]
        bp   = ax.boxplot(data, patch_artist=True, widths=0.55,
                          medianprops=dict(color="black", linewidth=1.6),
                          flierprops=dict(marker=".", markersize=1, alpha=0.3))
        for patch, c in zip(bp["boxes"], CLASSES):
            patch.set_facecolor(PALETTE[c]); patch.set_alpha(0.7)
        ax.set_xticklabels(CLASSES, fontsize=8, rotation=15)
        ax.set_title(col, fontsize=10); ax.grid(axis="y", alpha=0.3)

        summary[col] = {
            c: {
                "mean":   float(np.nanmean(d)) if len(d) else None,
                "median": float(np.nanmedian(d)) if len(d) else None,
                "std":    float(np.nanstd(d))  if len(d) else None,
            } for c, d in zip(CLASSES, data)
        }

    for ax in axes.flat[len(plot_cols):]:
        ax.axis("off")
    plt.suptitle("Photometric distributions per class (EDA only — not model input)",
                 y=1.01, fontsize=12)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[plot] {save_path}")
    return summary


def eda_color_color(df: pd.DataFrame, save_path: str):
    """Classic g-r vs r-i diagram — STAR/GALAXY/QSO famously separate here."""
    if not {"g_r", "r_i"}.issubset(df.columns):
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in CLASSES:
        sub = df[df["class"] == c]
        ax.scatter(sub["g_r"], sub["r_i"], s=4, alpha=0.35, label=c,
                   color=PALETTE[c])
    ax.set_xlabel("g − r"); ax.set_ylabel("r − i")
    ax.set_title("Colour–colour diagram")
    ax.legend(markerscale=3); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[plot] {save_path}")


def eda_sky_distribution(df: pd.DataFrame, save_path: str):
    if not {"ra", "dec"}.issubset(df.columns): return
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in CLASSES:
        sub = df[df["class"] == c]
        ax.scatter(sub["ra"], sub["dec"], s=2, alpha=0.4, label=c, color=PALETTE[c])
    ax.set_xlabel("RA (deg)"); ax.set_ylabel("Dec (deg)")
    ax.set_title("Sky coverage")
    ax.legend(markerscale=4); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[plot] {save_path}")


def eda_image_statistics(df: pd.DataFrame, save_path: str, n_per_class=200) -> dict:
    """
    Pixel-level statistics computed by loading a random subsample of the
    actual .npy tensors. Useful to detect:
      • dead pixels / saturation / NaNs
      • per-class brightness offsets (could bias the model)
      • severely different dynamic ranges
    """
    stats_per_class = {}
    rng = np.random.RandomState(SEED)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    for col, c in enumerate(CLASSES):
        sub  = df[df["class"] == c]
        take = min(n_per_class, len(sub))
        idxs = rng.choice(len(sub), size=take, replace=False)
        means, stds, mins, maxs, p99 = [], [], [], [], []
        for i in idxs:
            a = np.load(sub.iloc[i]["filepath"]).astype(np.float32)
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            means.append(a.mean()); stds.append(a.std())
            mins.append(a.min());   maxs.append(a.max())
            p99.append(np.percentile(a, 99))

        stats_per_class[c] = {
            "n_sampled":   take,
            "mean_of_means": float(np.mean(means)),
            "mean_of_stds":  float(np.mean(stds)),
            "global_min":    float(np.min(mins)),
            "global_max":    float(np.max(maxs)),
            "mean_p99":      float(np.mean(p99)),
        }

        # Robust binning: check variance and use 'auto' if near-constant
        means_range = np.max(means) - np.min(means)
        stds_range  = np.max(stds)  - np.min(stds)

        bins_mean = 'auto' if means_range > 1e-10 else 1
        bins_std  = 'auto' if stds_range  > 1e-10 else 1

        try:
            axes[0, col].hist(means, bins=bins_mean, color=PALETTE[c], alpha=0.8)
        except ValueError:
            # Fallback for degenerate case
            axes[0, col].text(0.5, 0.5, f"All means ≈ {means[0]:.4f}\n(constant)", 
                             ha='center', va='center', transform=axes[0, col].transAxes)
        axes[0, col].set_title(f"{c} — pixel mean"); axes[0, col].grid(alpha=0.3)

        try:
            axes[1, col].hist(stds, bins=bins_std, color=PALETTE[c], alpha=0.8)
        except ValueError:
            # Fallback for degenerate case
            axes[1, col].text(0.5, 0.5, f"All stds ≈ {stds[0]:.4f}\n(constant)", 
                             ha='center', va='center', transform=axes[1, col].transAxes)
        axes[1, col].set_title(f"{c} — pixel std");  axes[1, col].grid(alpha=0.3)

    plt.suptitle("Pixel statistics from .npy tensors (pre-normalisation)",
                 y=1.01, fontsize=12)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[plot] {save_path}")
    return stats_per_class


def eda_sample_images(df: pd.DataFrame, save_path: str, n_per_class=6):
    rng = np.random.RandomState(SEED)
    fig, axes = plt.subplots(N_CLASSES, n_per_class,
                             figsize=(n_per_class*2, N_CLASSES*2))
    for r, c in enumerate(CLASSES):
        sub  = df[df["class"] == c]
        idxs = rng.choice(len(sub), size=min(n_per_class, len(sub)), replace=False)
        for k, i in enumerate(idxs):
            a = np.load(sub.iloc[i]["filepath"]).astype(np.float32)
            a = np.nan_to_num(a)
            # arcsinh stretch for display (handles huge dynamic range)
            disp = np.arcsinh(a - np.median(a))
            axes[r, k].imshow(disp, cmap="gray")
            axes[r, k].axis("off")
            if k == 0: axes[r, k].set_ylabel(c, fontsize=11)
    plt.suptitle("Sample tensors (arcsinh-stretched for display only)",
                 y=1.02, fontsize=12)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[plot] {save_path}")


def run_eda(df: pd.DataFrame) -> dict:
    print("\n" + "═"*60 + "\n  Stage 0a · Exploratory Data Analysis\n" + "═"*60)

    out = {}
    out["n_total"]      = int(len(df))
    out["class_dist"]   = eda_class_distribution(df, os.path.join(EDA_DIR, "class_distribution.png"))
    out["photometry"]   = eda_photometry       (df, os.path.join(EDA_DIR, "photometry.png"))
    eda_color_color    (df, os.path.join(EDA_DIR, "color_color.png"))
    eda_sky_distribution(df, os.path.join(EDA_DIR, "sky_distribution.png"))

    print("\n" + "═"*60 + "\n  Stage 0b · Image-tensor diagnostics\n" + "═"*60)
    out["image_stats"]  = eda_image_statistics(df, os.path.join(EDA_DIR, "pixel_statistics.png"))
    eda_sample_images(df, os.path.join(EDA_DIR, "sample_images.png"))

    save_json(out, os.path.join(EDA_DIR, "eda_summary.json"))
    debug_break("after EDA")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 4.  MODELS
# ═════════════════════════════════════════════════════════════════════════════
#
# Mathematical foundation — 2D discrete convolution
# ─────────────────────────────────────────────────
#     (I * K)[i, j]  =  Σ_m Σ_n  I[i+m, j+n] · K[m, n]
#
# Three properties that make convolution the right inductive bias for
# images (and SDSS cutouts specifically):
#
#   (a) Translation equivariance: f(T_x I) = T_x f(I).
#       A galaxy shifted by Δ pixels produces feature maps shifted by Δ.
#   (b) Local connectivity: each output depends on a small receptive
#       field — matching the locality of physical structures (PSF FWHM
#       ≈ 1.4″ ≈ 3-4 px at SDSS scale).
#   (c) Parameter sharing: the same kernel K scans the image, so we
#       have O(k²·C_in·C_out) parameters per layer instead of the
#       O(H²·W²·C²) of a fully-connected layer.
#
# Universal approximation (Cybenko 1989; Hornik 1991): a feed-forward
# net with any non-polynomial activation can approximate any continuous
# function on a compact set arbitrarily well. Depth gives exponentially
# more representational efficiency for compositional functions
# (Telgarsky 2016), which justifies stacking conv blocks.
#
# Batch normalisation (Ioffe & Szegedy 2015):
#     x̂ = (x - μ_B) / √(σ²_B + ε);   y = γ x̂ + β
# Empirically smooths the loss landscape (Santurkar et al. 2018),
# allowing larger LRs and faster convergence.
# ─────────────────────────────────────────────────────────────────────────────


# ── 4.0  Reusable conv block ─────────────────────────────────────────────────
def conv_block(c_in, c_out, drop=0.1):
    """
    Conv → BN → ReLU → Conv → BN → ReLU → MaxPool(2) → SpatialDropout
    Halves H,W; doubles channels.
    """
    return nn.Sequential(
        nn.Conv2d(c_in,  c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(drop),
    )


# ── 4.1  Image encoder (shared backbone) ─────────────────────────────────────
class ImageEncoder(nn.Module):
    """
    1 → 32 → 64 → 128 → 256 channels, four ×2 downsamplings.
    For IMG_SIZE = 128 → feature map = 8×8 → flat dim = 256·8·8 = 16,384.
    """
    def __init__(self, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(1,   32,  drop),
            conv_block(32,  64,  drop),
            conv_block(64,  128, drop),
            conv_block(128, 256, drop),
        )
        self.feature_dim = 256 * (IMG_SIZE // 16) * (IMG_SIZE // 16)

    def forward(self, x):
        return self.net(x).flatten(1)


# ── 4.2  Plain CNN classifier (Stage 1) ──────────────────────────────────────
class CNNClassifier(nn.Module):
    """
    Image-only classifier.
    Cross-entropy loss is the negative log-likelihood of a categorical
    observation model:
         L(θ) = - Σ_i  log p_θ(y_i | x_i)
    Its gradient w.r.t. logits z is the bounded vector (softmax(z) − y),
    giving stable optimisation. Class weighting scales each term by w_{y_i}.
    """
    def __init__(self, n_classes=N_CLASSES, drop=DROPOUT):
        super().__init__()
        self.encoder = ImageEncoder(drop=0.1)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, 512),
            nn.ReLU(inplace=True), nn.Dropout(drop),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True), nn.Dropout(drop),
        )
        self.classifier = nn.Linear(128, n_classes)

    def features(self, x):
        return self.head(self.encoder(x))

    def forward(self, x):
        return self.classifier(self.features(x))


# ── 4.3  Convolutional Autoencoder (Stage 2) ─────────────────────────────────
#
# The autoencoder learns φ: X → ℝ^d and ψ: ℝ^d → X minimising
#     L_CAE(φ, ψ)  =  E_X ‖X − ψ(φ(X))‖²
#
# Manifold hypothesis: natural images live near a low-dimensional manifold
# M ⊂ ℝ^{H·W}. With dim(φ) ≪ H·W, the bottleneck forces φ to parametrise
# M, so nearby z's correspond to perceptually similar images.
# The CAE latent therefore captures morphology that may not be tied to
# the class-discriminative signal — this is the complementary view that
# helps the fused classifier.
# ─────────────────────────────────────────────────────────────────────────────

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        # ⌊IMG_SIZE / 16⌋ after four stride-2 convs.
        self.feat_hw  = IMG_SIZE // 16
        self.flat_dim = 256 * self.feat_hw * self.feat_hw
        self.latent_dim = latent_dim

        self.enc_conv = nn.Sequential(
            nn.Conv2d(1,   32,  4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.Conv2d(32,  64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64,  128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
        )
        self.enc_fc = nn.Linear(self.flat_dim, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, self.flat_dim)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.ConvTranspose2d(32,  1,   4, stride=2, padding=1),
        )

    def encode(self, x):
        return self.enc_fc(self.enc_conv(x).flatten(1))

    def decode(self, z):
        h = self.dec_fc(z).view(-1, 256, self.feat_hw, self.feat_hw)
        return self.dec_conv(h)

    def forward(self, x):
        return self.decode(self.encode(x))


# ── 4.4  Variational Autoencoder (Stage 3) ───────────────────────────────────
#
# Generative model: p_θ(x, z) = p_θ(x|z) p(z),  p(z) = N(0, I).
# We want p_θ(x), but ∫ p_θ(x|z) p(z) dz is intractable. Introduce a
# learned variational posterior q_φ(z|x) = N(μ_φ(x), diag σ²_φ(x)).
#
# ELBO derivation (Kingma & Welling 2014):
#
#   log p_θ(x)
#     = log ∫ p_θ(x|z) p(z) dz
#     = log E_{z∼q_φ(z|x)} [ p_θ(x|z) p(z) / q_φ(z|x) ]
#     ≥ E_{q_φ}[ log p_θ(x|z) ]  −  D_KL( q_φ(z|x) ‖ p(z) )      [Jensen]
#       ──────────────────────       ──────────────────────────
#       reconstruction term          KL regulariser
#
# We minimise   L_VAE  =  L_recon  +  β · L_KL .
# For diagonal Gaussians vs. N(0,I):
#
#   D_KL  =  ½ Σ_j ( μ_j²  +  σ_j²  −  log σ_j²  −  1 )
#
# Reparameterisation trick. The naive sample z ∼ q_φ(z|x) breaks
# back-prop. Write z = μ + σ ⊙ ε, ε ∼ N(0, I); now the only randomness
# is ε (independent of θ, φ) and the chain rule flows through (μ, σ).
#
# β-VAE / KL annealing.
#   • β = 1 is the standard ELBO.
#   • β > 1 forces a more factorised latent (disentanglement).
#   • β < 1 prioritises reconstruction.
#   • Annealing β: 0 → β_max during the first few epochs prevents
#     posterior collapse, where q_φ(z|x) ≡ p(z) for all x and the
#     decoder ignores z (Bowman 2016).
#
# Why the VAE latent is better than the CAE latent for downstream
# classification (proof sketch):
#   The KL term pulls q_φ(z|x) toward N(0, I). Integrating over data,
#   the aggregate posterior q_φ(z) = ∫ q_φ(z|x) p_data(x) dx is close
#   to N(0, I). This means the latent is bounded, isotropic, and
#   smooth — semantically similar x map to nearby z. A linear
#   classifier on that geometry is well-posed; on a CAE latent
#   (unconstrained scale, no smoothness guarantee) it is not.
# ─────────────────────────────────────────────────────────────────────────────

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.feat_hw  = IMG_SIZE // 16
        self.flat_dim = 256 * self.feat_hw * self.feat_hw
        self.latent_dim = latent_dim

        self.enc_conv = nn.Sequential(
            nn.Conv2d(1,   32,  4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.Conv2d(32,  64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64,  128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
        )
        self.fc_mu     = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, self.flat_dim)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.ConvTranspose2d(32,  1,   4, stride=2, padding=1),
        )

    def encode(self, x):
        h = self.enc_conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        # σ = exp(½ · log σ²) — always positive, numerically stable.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = self.dec_fc(z).view(-1, 256, self.feat_hw, self.feat_hw)
        return self.dec_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def latent(self, x):
        """
        Deterministic latent for downstream use: take the mean μ of
        q_φ(z|x). At inference this is what you want — no stochasticity.
        """
        mu, _ = self.encode(x)
        return mu


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """
    Returns (total, recon, KL) — all *per-pixel* / *per-sample* means
    so the magnitudes don't change when batch size or image size changes.
    """
    # Reconstruction: MSE per element, then summed over pixels, averaged over batch
    recon_per_sample = F.mse_loss(recon, x, reduction="none").flatten(1).sum(dim=1).mean()
    # KL closed form for q = N(μ, diag σ²) vs p = N(0, I)
    kl_per_sample = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    total = recon_per_sample + beta * kl_per_sample
    return total, recon_per_sample.detach(), kl_per_sample.detach()


# ── 4.5  Fused classifier — generic over CAE or VAE ──────────────────────────
class FusedClassifier(nn.Module):
    """
    Concatenates the CNN's 128-D feature with the AE's d-dim latent and
    feeds the result through a trainable MLP head. The CNN and the AE
    are FROZEN — only the head is updated.

    Why fuse, mathematically?
      Suppose the CNN feature φ_C(x) and the AE feature φ_A(x) are
      conditionally independent given the relevant signal in x. Then
      mutual information satisfies
          I([φ_C, φ_A]; Y)  ≥  max( I(φ_C; Y),  I(φ_A; Y) ),
      with equality only when one is redundant. So concatenation can
      *never hurt* (in the information-theoretic sense) and often
      strictly helps when φ_A captures structure φ_C missed.
    """
    def __init__(self, cnn: CNNClassifier, autoenc,
                 ae_kind: str, latent_dim=LATENT_DIM,
                 n_classes=N_CLASSES, drop=DROPOUT):
        super().__init__()
        assert ae_kind in ("cae", "vae")
        self.ae_kind = ae_kind

        # Freeze backbones — keep their BatchNorm running stats fixed too.
        self.cnn = cnn.eval()
        self.ae  = autoenc.eval()
        for p in self.cnn.parameters(): p.requires_grad = False
        for p in self.ae.parameters():  p.requires_grad = False

        # Layer-normalise each stream — they live on very different scales.
        cnn_feat_dim = 128
        self.norm_cnn = nn.LayerNorm(cnn_feat_dim)
        self.norm_ae  = nn.LayerNorm(latent_dim)

        fused_dim = cnn_feat_dim + latent_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(True), nn.Dropout(drop),
            nn.Linear(256, 128),       nn.ReLU(True), nn.Dropout(drop),
            nn.Linear(128, n_classes),
        )

    def _ae_latent(self, x):
        if self.ae_kind == "cae":
            return self.ae.encode(x)
        else:                                  # vae → μ only
            mu, _ = self.ae.encode(x)
            return mu

    def forward(self, x):
        with torch.no_grad():
            f_cnn = self.cnn.features(x)
            f_ae  = self._ae_latent(x)
        f = torch.cat([self.norm_cnn(f_cnn), self.norm_ae(f_ae)], dim=1)
        return self.head(f)


# ═════════════════════════════════════════════════════════════════════════════
# 5.  TRAINING UTILITIES
# ═════════════════════════════════════════════════════════════════════════════
#
# Optimiser — Adam (Kingma & Ba 2015):
#   m_t = β₁ m_{t-1} + (1-β₁) g_t
#   v_t = β₂ v_{t-1} + (1-β₂) g_t²
#   θ_{t+1} = θ_t − lr · m̂_t / (√v̂_t + ε)
# Adaptive per-parameter learning rate; robust to sparse gradients.
#
# Gradient clipping by global norm (Pascanu 2013):
#   if ‖g‖₂ > c:   g ← g · c / ‖g‖₂
# Prevents one bad batch from blowing up the weights.
#
# Mixed precision (AMP):
#   Forward/backward in fp16, master weights in fp32 with loss scaling.
#   ~2× faster on tensor-core GPUs without accuracy loss.
# ─────────────────────────────────────────────────────────────────────────────

def _list_completed(mode, label, max_epoch):
    return sorted([
        ep for ep in range(1, max_epoch + 1)
        if os.path.exists(ckpt_path(mode, label, ep))
    ])


def _train_step_cls(model, imgs, labels, criterion, optimizer, scaler):
    optimizer.zero_grad(set_to_none=True)
    with autocast(enabled=USE_AMP):
        out  = model(imgs)
        loss = criterion(out, labels)
    if USE_AMP:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
        scaler.step(optimizer); scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
        optimizer.step()
    return loss.item(), out


def train_epoch_cls(model, loader, criterion, optimizer, scaler):
    model.train()
    tot_loss, correct, n = 0., 0, 0
    for imgs, labels, _ in loader:
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        loss, out = _train_step_cls(model, imgs, labels, criterion, optimizer, scaler)
        tot_loss += loss * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        n        += imgs.size(0)
        if BREAK_AFTER_BATCH:
            debug_break(f"first cls batch — out.shape={tuple(out.shape)}"); break
    return tot_loss / n, correct / n


@torch.no_grad()
def eval_epoch_cls(model, loader, criterion):
    model.eval()
    tot_loss, correct, n = 0., 0, 0
    for imgs, labels, _ in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast(enabled=USE_AMP):
            out  = model(imgs)
            loss = criterion(out, labels)
        tot_loss += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        n        += imgs.size(0)
    return tot_loss / n, correct / n


def train_epoch_cae(model, loader, criterion, optimizer, scaler):
    model.train()
    tot, n = 0., 0
    for imgs, _, _ in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=USE_AMP):
            loss = criterion(model(imgs), imgs)
        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        tot += loss.item() * imgs.size(0); n += imgs.size(0)
        if BREAK_AFTER_BATCH: debug_break("first CAE batch"); break
    return tot / n


@torch.no_grad()
def eval_epoch_cae(model, loader, criterion):
    model.eval()
    tot, n = 0., 0
    for imgs, _, _ in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with autocast(enabled=USE_AMP):
            tot += criterion(model(imgs), imgs).item() * imgs.size(0)
        n += imgs.size(0)
    return tot / n


def train_epoch_vae(model, loader, optimizer, scaler, beta):
    model.train()
    tot, tot_r, tot_k, n = 0., 0., 0., 0
    for imgs, _, _ in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=USE_AMP):
            recon, mu, logvar = model(imgs)
            loss, lr_, lk_ = vae_loss(recon, imgs, mu, logvar, beta=beta)
        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        bs = imgs.size(0)
        tot   += loss.item() * bs
        tot_r += lr_.item()   * bs
        tot_k += lk_.item()   * bs
        n     += bs
        if BREAK_AFTER_BATCH: debug_break("first VAE batch"); break
    return tot / n, tot_r / n, tot_k / n


@torch.no_grad()
def eval_epoch_vae(model, loader, beta):
    model.eval()
    tot, tot_r, tot_k, n = 0., 0., 0., 0
    for imgs, _, _ in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with autocast(enabled=USE_AMP):
            recon, mu, logvar = model(imgs)
            loss, lr_, lk_    = vae_loss(recon, imgs, mu, logvar, beta=beta)
        bs = imgs.size(0)
        tot   += loss.item() * bs
        tot_r += lr_.item()  * bs
        tot_k += lk_.item()  * bs
        n     += bs
    return tot / n, tot_r / n, tot_k / n


# ── Generic trainer for cls + cae (single-loss). VAE uses its own loop. ──────
def run_training(model, train_dl, val_dl, epochs, criterion, optimizer,
                 scheduler=None, mode="cls", label="model"):
    history = {"train_loss": [], "val_loss": []}
    if mode == "cls":
        history["train_acc"] = []; history["val_acc"] = []

    best_val, best_state, best_epoch = float("inf"), None, 1
    start_epoch = 1

    scaler = GradScaler(enabled=USE_AMP)

    # Resume from latest checkpoint.
    completed = _list_completed(mode, label, epochs)
    if completed:
        last_ep = completed[-1]
        ckpt    = torch.load(ckpt_path(mode, label, last_ep), map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        history     = ckpt["history"]
        best_val    = ckpt["best_val"]
        best_epoch  = ckpt["best_epoch"]
        start_epoch = last_ep + 1
        print(f"[{label}] resuming from epoch {last_ep}/{epochs}")

        if start_epoch > epochs:
            best_ckpt = torch.load(ckpt_path(mode, label, best_epoch),
                                   map_location=DEVICE)
            model.load_state_dict(best_ckpt["model_state_dict"])
            print(f"[{label}] already complete — best epoch {best_epoch}")
            return history

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        if mode == "cls":
            tr_loss, tr_acc = train_epoch_cls(model, train_dl, criterion, optimizer, scaler)
            vl_loss, vl_acc = eval_epoch_cls (model, val_dl,   criterion)
            history["train_acc"].append(tr_acc); history["val_acc"].append(vl_acc)
            print(f"[{label}] ep {epoch:>3}/{epochs}  loss {tr_loss:.4f}/{vl_loss:.4f}  "
                  f"acc {tr_acc:.4f}/{vl_acc:.4f}  ({time.time()-t0:.1f}s)")
        else:  # cae
            tr_loss = train_epoch_cae(model, train_dl, criterion, optimizer, scaler)
            vl_loss = eval_epoch_cae (model, val_dl,   criterion)
            print(f"[{label}] ep {epoch:>3}/{epochs}  recon {tr_loss:.6f}/{vl_loss:.6f}  "
                  f"({time.time()-t0:.1f}s)")

        history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)

        improved = vl_loss < best_val
        if improved:
            prev_best = ckpt_path(mode, label, best_epoch)
            if os.path.exists(prev_best) and best_epoch != epoch - 1:
                os.remove(prev_best)
            best_val   = vl_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        torch.save({
            "epoch": epoch, "best_epoch": best_epoch, "best_val": best_val,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if USE_AMP else None,
            "history": history,
        }, ckpt_path(mode, label, epoch))

        # Rolling cleanup — keep current + best.
        old = ckpt_path(mode, label, epoch - 2)
        if os.path.exists(old) and (epoch - 2) != best_epoch:
            os.remove(old)

        if scheduler is not None:
            scheduler.step(vl_loss) if isinstance(
                scheduler, optim.lr_scheduler.ReduceLROnPlateau) else scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    debug_break(f"after training [{label}]")
    return history


def run_training_vae(model, train_dl, val_dl, epochs, optimizer,
                     scheduler=None, label="vae",
                     beta_max=VAE_BETA_MAX, anneal_epochs=VAE_KL_ANNEAL_EPOCHS):
    """Custom loop for the VAE — multi-component loss + KL annealing."""
    history = {"train_loss": [], "val_loss": [],
               "train_recon": [], "val_recon": [],
               "train_kl": [],    "val_kl": [],
               "beta": []}
    best_val, best_state, best_epoch = float("inf"), None, 1
    start_epoch = 1
    scaler = GradScaler(enabled=USE_AMP)

    completed = _list_completed("vae", label, epochs)
    if completed:
        last_ep = completed[-1]
        ckpt    = torch.load(ckpt_path("vae", label, last_ep), map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt and USE_AMP and ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        history     = ckpt["history"]
        best_val    = ckpt["best_val"]
        best_epoch  = ckpt["best_epoch"]
        start_epoch = last_ep + 1
        print(f"[{label}] resuming from epoch {last_ep}/{epochs}")
        if start_epoch > epochs:
            best_ckpt = torch.load(ckpt_path("vae", label, best_epoch),
                                   map_location=DEVICE)
            model.load_state_dict(best_ckpt["model_state_dict"])
            return history

    for epoch in range(start_epoch, epochs + 1):
        # Linear KL annealing.
        beta = beta_max * min(1.0, epoch / max(anneal_epochs, 1))
        t0 = time.time()
        tr_loss, tr_r, tr_k = train_epoch_vae(model, train_dl, optimizer, scaler, beta)
        vl_loss, vl_r, vl_k = eval_epoch_vae (model, val_dl,                 beta)
        history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)
        history["train_recon"].append(tr_r);   history["val_recon"].append(vl_r)
        history["train_kl"].append(tr_k);      history["val_kl"].append(vl_k)
        history["beta"].append(beta)

        print(f"[{label}] ep {epoch:>3}/{epochs}  β={beta:.2f}  "
              f"loss {tr_loss:.2f}/{vl_loss:.2f}  "
              f"recon {tr_r:.2f}/{vl_r:.2f}  KL {tr_k:.2f}/{vl_k:.2f}  "
              f"({time.time()-t0:.1f}s)")

        if vl_loss < best_val:
            prev = ckpt_path("vae", label, best_epoch)
            if os.path.exists(prev) and best_epoch != epoch - 1:
                os.remove(prev)
            best_val   = vl_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        torch.save({
            "epoch": epoch, "best_epoch": best_epoch, "best_val": best_val,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if USE_AMP else None,
            "history": history,
        }, ckpt_path("vae", label, epoch))

        old = ckpt_path("vae", label, epoch - 2)
        if os.path.exists(old) and (epoch - 2) != best_epoch:
            os.remove(old)

        if scheduler is not None:
            scheduler.step(vl_loss) if isinstance(
                scheduler, optim.lr_scheduler.ReduceLROnPlateau) else scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    debug_break(f"after training [{label}]")
    return history


# ═════════════════════════════════════════════════════════════════════════════
# 6.  EVALUATION  — metrics + their mathematical definitions
# ═════════════════════════════════════════════════════════════════════════════
#
# Notation:
#   N        — number of samples
#   C        — number of classes
#   TP_c, FP_c, FN_c, TN_c   — one-vs-rest contingency for class c
#
# Accuracy:               acc = Σ_c TP_c / N
# Balanced accuracy:      ½·(TPR_avg + TNR_avg) — robust to imbalance
# Precision_c = TP_c / (TP_c + FP_c)
# Recall_c    = TP_c / (TP_c + FN_c)
# F1_c        = 2·P·R / (P + R)
#
# Macro vs micro vs weighted:
#   macro:    unweighted mean over classes — emphasises small classes
#   weighted: weighted by class support     — matches accuracy under heavy imbalance
#   micro:    global TP / (TP + FN) — equals accuracy for multiclass
#
# Cohen's κ:
#   κ = (p_o − p_e) / (1 − p_e),
#   p_o = observed agreement,  p_e = chance agreement under independence
#
# Matthews Correlation Coefficient (multiclass form, Gorodkin 2004):
#     MCC = (N·Σ_k C_{kk} − Σ_k (Σ_l C_{kl})(Σ_l C_{lk})) /
#           √[ ( N² − Σ_k(Σ_l C_{kl})² ) · ( N² − Σ_k(Σ_l C_{lk})² ) ]
#   Range [-1, +1]; +1 iff perfect prediction; 0 iff independent.
#
# ROC-AUC (OvR, macro): for each class c, treat as binary, compute AUC,
#   then average. Captures rank-based separability.
#
# Log-loss (cross-entropy):  − (1/N) Σ_i log p̂_{i, y_i}
#   Calibration-aware: a confident-wrong predictor is punished hard.
#
# Brier score (multiclass):  (1/N) Σ_i Σ_c (p̂_{i,c} − 1_{y_i=c})²
#   Proper scoring rule; lower is better.
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_with_probs(model, loader):
    """Return predictions (argmax), softmax probabilities, and labels."""
    model.eval()
    preds, probs, labels = [], [], []
    for imgs, lbls, _ in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with autocast(enabled=USE_AMP):
            logits = model(imgs)
        p = F.softmax(logits.float(), dim=1).cpu().numpy()
        preds.append(p.argmax(1)); probs.append(p); labels.append(lbls.numpy())
    return (np.concatenate(preds),
            np.concatenate(probs, axis=0),
            np.concatenate(labels))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def compute_all_metrics(label: str, preds, probs, labels) -> dict:
    """Compute the full metric battery and per-class breakdown."""
    out = {"label": label}

    out["accuracy"]            = float(accuracy_score(labels, preds))
    out["balanced_accuracy"]   = float(balanced_accuracy_score(labels, preds))
    out["top2_accuracy"]       = float(top_k_accuracy_score(labels, probs, k=2,
                                                           labels=list(range(N_CLASSES))))
    out["precision_macro"]     = float(precision_score(labels, preds, average="macro",   zero_division=0))
    out["precision_weighted"]  = float(precision_score(labels, preds, average="weighted", zero_division=0))
    out["recall_macro"]        = float(recall_score   (labels, preds, average="macro",   zero_division=0))
    out["recall_weighted"]     = float(recall_score   (labels, preds, average="weighted", zero_division=0))
    out["f1_macro"]            = float(f1_score       (labels, preds, average="macro",   zero_division=0))
    out["f1_weighted"]         = float(f1_score       (labels, preds, average="weighted", zero_division=0))
    out["cohen_kappa"]         = float(cohen_kappa_score(labels, preds))
    out["matthews_corrcoef"]   = float(matthews_corrcoef(labels, preds))
    try:
        out["roc_auc_ovr_macro"] = float(roc_auc_score(
            labels, probs, multi_class="ovr", average="macro",
            labels=list(range(N_CLASSES))))
    except Exception as e:
        out["roc_auc_ovr_macro"] = None
    out["log_loss"]            = float(log_loss(labels, np.clip(probs, 1e-12, 1.0),
                                                labels=list(range(N_CLASSES))))
    out["brier"]               = brier_score(probs, labels)

    # Per-class breakdown
    per_class = {}
    for c_idx, c_name in enumerate(CLASSES):
        y_bin = (labels == c_idx).astype(int)
        p_bin = (preds  == c_idx).astype(int)
        tp = int(((p_bin == 1) & (y_bin == 1)).sum())
        fp = int(((p_bin == 1) & (y_bin == 0)).sum())
        fn = int(((p_bin == 0) & (y_bin == 1)).sum())
        tn = int(((p_bin == 0) & (y_bin == 0)).sum())
        per_class[c_name] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / max(tp + fp, 1),
            "recall":    tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "f1":        2*tp / max(2*tp + fp + fn, 1),
            "support":   int((labels == c_idx).sum()),
        }
    out["per_class"] = per_class

    cm = confusion_matrix(labels, preds, labels=list(range(N_CLASSES)))
    out["confusion_matrix"] = cm.tolist()

    # Pretty print
    print(f"\n{'─'*64}\n {label}\n{'─'*64}")
    print(f" Accuracy           : {out['accuracy']:.4f}")
    print(f" Balanced accuracy  : {out['balanced_accuracy']:.4f}")
    print(f" Top-2 accuracy     : {out['top2_accuracy']:.4f}")
    print(f" Macro F1           : {out['f1_macro']:.4f}")
    print(f" Weighted F1        : {out['f1_weighted']:.4f}")
    print(f" Cohen's κ          : {out['cohen_kappa']:.4f}")
    print(f" Matthews CC        : {out['matthews_corrcoef']:.4f}")
    print(f" ROC-AUC (OvR mac)  : {out['roc_auc_ovr_macro']}")
    print(f" Log-loss           : {out['log_loss']:.4f}")
    print(f" Brier              : {out['brier']:.4f}")
    print(classification_report(labels, preds, target_names=CLASSES,
                                digits=4, zero_division=0))
    return out


# ── Statistical comparison of two classifiers on the same test set ──────────
#
# McNemar's test (paired, two related classifiers).
#   Build a 2×2 contingency:
#       n11 = both correct,   n10 = only A correct,
#       n01 = only B correct, n00 = both wrong.
#   Under H0 (the two classifiers have the same error rate),
#       n10  ~  Binomial(n10 + n01, 0.5).
#   Test statistic (with continuity correction, Edwards 1948):
#       χ² = (|n10 − n01| − 1)² / (n10 + n01),     df = 1
#   For small (n10 + n01), use the exact binomial form.
#
# Why McNemar (not a paired t-test)?
#   Errors are Bernoulli, not Gaussian; both classifiers see the same x's,
#   so observations are paired and not independent across models.
# ─────────────────────────────────────────────────────────────────────────────

def mcnemar(preds_a, preds_b, labels):
    a_right = (preds_a == labels)
    b_right = (preds_b == labels)
    n10 = int(( a_right & ~b_right).sum())   # A right, B wrong
    n01 = int((~a_right &  b_right).sum())   # A wrong, B right
    discordant = n10 + n01

    if discordant == 0:
        return {"n10": 0, "n01": 0, "chi2": 0.0, "p_value": 1.0, "method": "tie"}

    if discordant < 25:
        # Exact two-sided binomial: P(K ≤ min ∨ K ≥ max | p = .5)
        from math import comb
        k = min(n10, n01)
        p = 2 * sum(comb(discordant, i) * 0.5**discordant for i in range(k + 1))
        p = min(p, 1.0)
        return {"n10": n10, "n01": n01, "chi2": None, "p_value": float(p),
                "method": "exact_binomial"}

    chi2 = (abs(n10 - n01) - 1) ** 2 / discordant
    # 1-CDF of χ²(1) at chi2 — closed form: erfc(sqrt(chi2 / 2))
    p = float(math.erfc(math.sqrt(chi2 / 2.0)))
    return {"n10": n10, "n01": n01, "chi2": float(chi2),
            "p_value": p, "method": "chi2_continuity"}


# ═════════════════════════════════════════════════════════════════════════════
# 7.  PLOTTING
# ═════════════════════════════════════════════════════════════════════════════

def plot_training_curves(history, label, save_path):
    has_acc = "train_acc" in history
    has_vae = "train_recon" in history

    if has_vae:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(history["train_loss"], label="train")
        axes[0].plot(history["val_loss"],   label="val")
        axes[0].set_title(f"{label} — total ELBO");  axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(history["train_recon"], label="train")
        axes[1].plot(history["val_recon"],   label="val")
        axes[1].set_title(f"{label} — reconstruction"); axes[1].legend(); axes[1].grid(alpha=0.3)
        axes[2].plot(history["train_kl"], label="train")
        axes[2].plot(history["val_kl"],   label="val")
        ax2 = axes[2].twinx(); ax2.plot(history["beta"], "--", color="grey", label="β")
        axes[2].set_title(f"{label} — KL term");  axes[2].legend(loc="upper left")
        axes[2].grid(alpha=0.3); ax2.legend(loc="upper right")
    else:
        fig, axes = plt.subplots(1, 2 if has_acc else 1, figsize=(12 if has_acc else 6, 4))
        if not has_acc: axes = [axes]
        axes[0].plot(history["train_loss"], label="train")
        axes[0].plot(history["val_loss"],   label="val")
        axes[0].set_title(f"{label} — Loss"); axes[0].set_xlabel("Epoch")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        if has_acc:
            axes[1].plot(history["train_acc"], label="train")
            axes[1].plot(history["val_acc"],   label="val")
            axes[1].set_title(f"{label} — Accuracy"); axes[1].set_xlabel("Epoch")
            axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[plot] {save_path}")


def plot_confusion_matrix(cm, label, save_path):
    cm = np.asarray(cm, dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_title(f"{label} — confusion (row-normalised)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[plot] {save_path}")


def _denorm_for_show(t):
    t = t - t.min()
    t = t / (t.max() + 1e-8)
    return t.clamp(0, 1)


def plot_ae_reconstructions(model, loader, save_path, label="AE", n=8, is_vae=False):
    model.eval()
    imgs, _, _ = next(iter(loader))
    imgs = imgs[:n].to(DEVICE)
    with torch.no_grad():
        if is_vae:
            recon, _, _ = model(imgs)
        else:
            recon = model(imgs)
    imgs, recon = imgs.cpu(), recon.cpu()
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i in range(n):
        axes[0, i].imshow(_denorm_for_show(imgs[i]).squeeze(),  cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(_denorm_for_show(recon[i]).squeeze(), cmap="gray"); axes[1, i].axis("off")
    axes[0, 0].set_title("input",     fontsize=10, loc="left")
    axes[1, 0].set_title("reconstr.", fontsize=10, loc="left")
    plt.suptitle(f"{label} — reconstructions", y=1.02)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[plot] {save_path}")


def plot_vae_samples(vae, save_path, n=8):
    """Samples from the prior — only meaningful for VAE."""
    vae.eval()
    with torch.no_grad():
        z = torch.randn(n, vae.latent_dim, device=DEVICE)
        out = vae.decode(z).cpu()
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2.4))
    for i in range(n):
        axes[i].imshow(_denorm_for_show(out[i]).squeeze(), cmap="gray")
        axes[i].axis("off")
    plt.suptitle("VAE — samples from prior z ~ N(0, I)", y=1.05)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[plot] {save_path}")


def plot_latent_pca(autoenc, loader, save_path, kind="cae"):
    """2D PCA of the latent — visual check on class clustering."""
    autoenc.eval()
    zs, ys = [], []
    with torch.no_grad():
        for imgs, lbls, _ in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            if kind == "cae":
                z = autoenc.encode(imgs)
            else:
                z, _ = autoenc.encode(imgs)
            zs.append(z.cpu().numpy()); ys.append(lbls.numpy())
    Z = np.concatenate(zs); Y = np.concatenate(ys)
    Z2 = PCA(n_components=2, random_state=SEED).fit_transform(Z)
    fig, ax = plt.subplots(figsize=(7, 6))
    for c_idx, c_name in enumerate(CLASSES):
        m = Y == c_idx
        ax.scatter(Z2[m, 0], Z2[m, 1], s=6, alpha=0.5, label=c_name,
                   color=PALETTE[c_name])
    ax.set_title(f"{kind.upper()} latent — first two PCs")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(markerscale=3); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[plot] {save_path}")


def plot_pair_comparison(m_a, m_b, save_path, mcn=None):
    """Side-by-side bar chart of every shared scalar metric for two models."""
    skip = {"label", "per_class", "confusion_matrix"}
    keys = [k for k in m_a if k not in skip
            and isinstance(m_a[k], (int, float))
            and m_a[k] is not None and m_b.get(k) is not None]
    a_vals = [m_a[k] for k in keys]
    b_vals = [m_b[k] for k in keys]

    x = np.arange(len(keys)); w = 0.4
    fig, ax = plt.subplots(figsize=(max(8, len(keys)*0.8), 5))
    ax.bar(x - w/2, a_vals, w, label=m_a["label"], color="#4C82C4")
    ax.bar(x + w/2, b_vals, w, label=m_b["label"], color="#E07B53")
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=9)
    title = f"{m_a['label']}  vs  {m_b['label']}"
    if mcn is not None and mcn.get("p_value") is not None:
        title += f"   |   McNemar p = {mcn['p_value']:.4g}  ({mcn['method']})"
    ax.set_title(title)
    ax.set_ylim(0, max(max(a_vals + b_vals) * 1.1, 1.05))
    ax.grid(axis="y", alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[plot] {save_path}")


def plot_overall_comparison(metric_list, save_path):
    """One figure summarising all three models on the key headline metrics."""
    keys  = ["accuracy", "balanced_accuracy", "f1_macro",
             "matthews_corrcoef", "cohen_kappa", "roc_auc_ovr_macro"]
    names = [m["label"] for m in metric_list]
    x = np.arange(len(keys)); w = 0.27
    colors = ["#4C82C4", "#E07B53", "#5FAD56"]
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, m in enumerate(metric_list):
        vals = [m.get(k) if m.get(k) is not None else 0.0 for k in keys]
        ax.bar(x + (i-1)*w, vals, w, label=m["label"], color=colors[i % 3])
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=15, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Model comparison — headline metrics")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[plot] {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 8.  MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("═"*64 + "\n  SDSS FITS-tensor classification pipeline\n" + "═"*64)

    # ── 0a / 0b · DATA + EDA ─────────────────────────────────────────────
    df = build_dataframe(CSV_PATH)
    eda_summary = run_eda(df)

    # Compute class weights from EDA result (effective sample-size form).
    # w_c = N / (C · n_c)   — normalises so Σ w_c · n_c / N  = 1.
    counts = eda_summary["class_dist"]["counts"]
    N = sum(counts.values())
    if CLASS_WEIGHTS_OVERRIDE is None:
        class_weights = [N / (N_CLASSES * counts[c]) for c in CLASSES]
    else:
        class_weights = list(CLASS_WEIGHTS_OVERRIDE)
    weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    print(f"[loss] class weights = {dict(zip(CLASSES, [round(w,3) for w in class_weights]))}")

    train_dl, val_dl, test_dl, *_ = build_dataloaders(df)

    all_metrics = {}

    # ── Stage 1 · CNN baseline ───────────────────────────────────────────
    print("\n" + "═"*64 + "\n  Stage 1 · CNN baseline (image-only)\n" + "═"*64)
    cnn = CNNClassifier(n_classes=N_CLASSES, drop=DROPOUT).to(DEVICE)
    cnn_opt   = optim.Adam(cnn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    cnn_sched = optim.lr_scheduler.CosineAnnealingLR(cnn_opt, T_max=EPOCHS_CNN)
    debug_break("before CNN training")

    cnn_hist = run_training(cnn, train_dl, val_dl, EPOCHS_CNN,
                            nn.CrossEntropyLoss(weight=weights),
                            cnn_opt, cnn_sched, mode="cls", label="CNN")
    plot_training_curves(cnn_hist, "CNN", os.path.join(OUT_DIR, "cnn_curves.png"))

    p, pr, y = predict_with_probs(cnn, test_dl)
    m_cnn = compute_all_metrics("CNN", p, pr, y)
    plot_confusion_matrix(m_cnn["confusion_matrix"], "CNN",
                          os.path.join(OUT_DIR, "cnn_confusion.png"))
    torch.save(cnn.state_dict(), os.path.join(OUT_DIR, "cnn.pt"))
    all_metrics["CNN"] = m_cnn
    cnn_preds_test = p
    debug_break("after CNN")

    # ── Stage 2 · CAE ────────────────────────────────────────────────────
    print("\n" + "═"*64 + "\n  Stage 2 · Convolutional Autoencoder\n" + "═"*64)
    cae = ConvAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    cae_opt   = optim.Adam(cae.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    cae_sched = optim.lr_scheduler.ReduceLROnPlateau(cae_opt, patience=3, factor=0.5)
    debug_break("before CAE training")

    cae_hist = run_training(cae, train_dl, val_dl, EPOCHS_CAE,
                            nn.MSELoss(), cae_opt, cae_sched,
                            mode="cae", label="CAE")
    plot_training_curves(cae_hist, "CAE", os.path.join(OUT_DIR, "cae_curves.png"))
    plot_ae_reconstructions(cae, test_dl,
                            os.path.join(OUT_DIR, "cae_reconstructions.png"),
                            label="CAE", is_vae=False)
    plot_latent_pca(cae, test_dl, os.path.join(OUT_DIR, "cae_latent_pca.png"),
                    kind="cae")
    torch.save(cae.state_dict(), os.path.join(OUT_DIR, "cae.pt"))
    debug_break("after CAE")

    # ── Stage 3 · VAE ────────────────────────────────────────────────────
    print("\n" + "═"*64 + "\n  Stage 3 · Variational Autoencoder\n" + "═"*64)
    vae = VariationalAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    vae_opt   = optim.Adam(vae.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    vae_sched = optim.lr_scheduler.ReduceLROnPlateau(vae_opt, patience=3, factor=0.5)
    debug_break("before VAE training")

    vae_hist = run_training_vae(vae, train_dl, val_dl, EPOCHS_VAE,
                                vae_opt, vae_sched, label="VAE")
    plot_training_curves(vae_hist, "VAE", os.path.join(OUT_DIR, "vae_curves.png"))
    plot_ae_reconstructions(vae, test_dl,
                            os.path.join(OUT_DIR, "vae_reconstructions.png"),
                            label="VAE", is_vae=True)
    plot_vae_samples(vae, os.path.join(OUT_DIR, "vae_prior_samples.png"))
    plot_latent_pca(vae, test_dl, os.path.join(OUT_DIR, "vae_latent_pca.png"),
                    kind="vae")
    torch.save(vae.state_dict(), os.path.join(OUT_DIR, "vae.pt"))
    debug_break("after VAE")

    # ── Stage 4 · Fused CNN + CAE ────────────────────────────────────────
    print("\n" + "═"*64 + "\n  Stage 4 · Fused  (CNN ⊕ CAE latent)\n" + "═"*64)
    # Reload best-saved CNN + CAE into fresh objects to be safe.
    cnn.load_state_dict(torch.load(os.path.join(OUT_DIR, "cnn.pt"), map_location=DEVICE))
    cae.load_state_dict(torch.load(os.path.join(OUT_DIR, "cae.pt"), map_location=DEVICE))

    fused_cae = FusedClassifier(cnn, cae, ae_kind="cae",
                                latent_dim=LATENT_DIM,
                                n_classes=N_CLASSES, drop=DROPOUT).to(DEVICE)
    fc_opt   = optim.Adam(filter(lambda p: p.requires_grad, fused_cae.parameters()),
                          lr=LR, weight_decay=WEIGHT_DECAY)
    fc_sched = optim.lr_scheduler.CosineAnnealingLR(fc_opt, T_max=EPOCHS_FUS)
    debug_break("before Fused-CAE training")

    fc_hist = run_training(fused_cae, train_dl, val_dl, EPOCHS_FUS,
                           nn.CrossEntropyLoss(weight=weights),
                           fc_opt, fc_sched, mode="cls", label="Fused-CAE")
    plot_training_curves(fc_hist, "Fused-CAE",
                         os.path.join(OUT_DIR, "fused_cae_curves.png"))
    p, pr, y = predict_with_probs(fused_cae, test_dl)
    m_fc = compute_all_metrics("Fused-CAE", p, pr, y)
    plot_confusion_matrix(m_fc["confusion_matrix"], "Fused-CAE",
                          os.path.join(OUT_DIR, "fused_cae_confusion.png"))
    torch.save(fused_cae.state_dict(), os.path.join(OUT_DIR, "fused_cae.pt"))
    all_metrics["Fused-CAE"] = m_fc
    fcae_preds_test = p
    debug_break("after Fused-CAE")

    # ── Stage 5 · Fused CNN + VAE ────────────────────────────────────────
    print("\n" + "═"*64 + "\n  Stage 5 · Fused  (CNN ⊕ VAE μ-latent)\n" + "═"*64)
    vae.load_state_dict(torch.load(os.path.join(OUT_DIR, "vae.pt"), map_location=DEVICE))

    fused_vae = FusedClassifier(cnn, vae, ae_kind="vae",
                                latent_dim=LATENT_DIM,
                                n_classes=N_CLASSES, drop=DROPOUT).to(DEVICE)
    fv_opt   = optim.Adam(filter(lambda p: p.requires_grad, fused_vae.parameters()),
                          lr=LR, weight_decay=WEIGHT_DECAY)
    fv_sched = optim.lr_scheduler.CosineAnnealingLR(fv_opt, T_max=EPOCHS_FUS)
    debug_break("before Fused-VAE training")

    fv_hist = run_training(fused_vae, train_dl, val_dl, EPOCHS_FUS,
                           nn.CrossEntropyLoss(weight=weights),
                           fv_opt, fv_sched, mode="cls", label="Fused-VAE")
    plot_training_curves(fv_hist, "Fused-VAE",
                         os.path.join(OUT_DIR, "fused_vae_curves.png"))
    p, pr, y = predict_with_probs(fused_vae, test_dl)
    m_fv = compute_all_metrics("Fused-VAE", p, pr, y)
    plot_confusion_matrix(m_fv["confusion_matrix"], "Fused-VAE",
                          os.path.join(OUT_DIR, "fused_vae_confusion.png"))
    torch.save(fused_vae.state_dict(), os.path.join(OUT_DIR, "fused_vae.pt"))
    all_metrics["Fused-VAE"] = m_fv
    fvae_preds_test = p
    test_labels     = y       # same loader → same order
    debug_break("after Fused-VAE")

    # ── Stage 6 · Comparison ─────────────────────────────────────────────
    print("\n" + "═"*64 + "\n  Stage 6 · Pairwise comparison + significance tests\n" + "═"*64)

    mcn_cnn_vs_cae = mcnemar(cnn_preds_test,  fcae_preds_test, test_labels)
    mcn_cnn_vs_vae = mcnemar(cnn_preds_test,  fvae_preds_test, test_labels)
    mcn_cae_vs_vae = mcnemar(fcae_preds_test, fvae_preds_test, test_labels)

    print(f" CNN  vs Fused-CAE   p = {mcn_cnn_vs_cae['p_value']:.4g}  "
          f"({mcn_cnn_vs_cae['method']})")
    print(f" CNN  vs Fused-VAE   p = {mcn_cnn_vs_vae['p_value']:.4g}  "
          f"({mcn_cnn_vs_vae['method']})")
    print(f" CAE  vs Fused-VAE   p = {mcn_cae_vs_vae['p_value']:.4g}  "
          f"({mcn_cae_vs_vae['method']})")

    plot_pair_comparison(m_cnn, m_fc, os.path.join(OUT_DIR, "cmp_cnn_vs_cae.png"),
                         mcn=mcn_cnn_vs_cae)
    plot_pair_comparison(m_cnn, m_fv, os.path.join(OUT_DIR, "cmp_cnn_vs_vae.png"),
                         mcn=mcn_cnn_vs_vae)
    plot_pair_comparison(m_fc,  m_fv, os.path.join(OUT_DIR, "cmp_cae_vs_vae.png"),
                         mcn=mcn_cae_vs_vae)

    plot_overall_comparison([m_cnn, m_fc, m_fv],
                            os.path.join(OUT_DIR, "comparison_overall.png"))

    # ── Save EVERYTHING ──────────────────────────────────────────────────
    save_json(all_metrics,
              os.path.join(METRIC_DIR, "all_metrics.json"))
    save_json({
        "cnn_vs_fused_cae": mcn_cnn_vs_cae,
        "cnn_vs_fused_vae": mcn_cnn_vs_vae,
        "fused_cae_vs_fused_vae": mcn_cae_vs_vae,
    }, os.path.join(METRIC_DIR, "significance_tests.json"))
    save_json({"eda": eda_summary, "class_weights": dict(zip(CLASSES, class_weights))},
              os.path.join(METRIC_DIR, "config_and_eda.json"))

    print("\n[summary]")
    for k, m in all_metrics.items():
        print(f"  {k:<12}  acc={m['accuracy']:.4f}  "
              f"bal_acc={m['balanced_accuracy']:.4f}  "
              f"f1={m['f1_macro']:.4f}  "
              f"mcc={m['matthews_corrcoef']:.4f}")

    print(f"\n[done] outputs → {OUT_DIR}/")


if __name__ == "__main__":
    main()