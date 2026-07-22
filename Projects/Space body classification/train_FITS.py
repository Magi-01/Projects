"""
SDSS Multimodal Classification Pipeline
=========================================
Every classifier receives THREE inputs:
  • image   — 128×128 RGB JPEG cutout
  • tabular — photometric + spatial features derived from the CSV:
                u, g, r, i, z         (5 raw magnitudes)
                u-g, g-r, r-i, i-z    (4 colour indices)
                sin(ra), cos(ra)       (2 — circular encoding, avoids 359°≈1° gap)
                dec                    (1 — linear, -90 to +90)
              ──────────────────────────────────────────
              TAB_DIM = 12 total

Why NOT use 'class' as a feature?
----------------------------------
'class' is the label we are predicting. Including it as an input
is direct data leakage — the model would learn to copy it rather than
classify by image/photometry.

Why sin/cos for RA?
-------------------
RA is a circular coordinate (0° = 360°). A raw value makes the model
think RA=1° and RA=359° are far apart when they are actually 2° apart.
Encoding as (sin(ra_rad), cos(ra_rad)) preserves angular proximity.

Stage 1 : Multimodal CNN          (image + tabular → class)
Stage 2 : Convolutional AE        (image → latent)          [unsupervised]
Stage 3 : Fused CNN + AE + tab    (image + latent + tab → class)
Stage 4 : Comparison

Checkpointing
-------------
A checkpoint is saved every epoch (regardless of whether val_loss improved).
On resume, the loop fast-forwards through already-completed epochs by loading
the most recent checkpoint, then continues training from there.
Best weights are tracked separately in memory and restored at the end.

Debug
-----
DEBUG             = True   → ipdb breakpoints at key checkpoints
BREAK_AFTER_BATCH = True   → stop after first batch (shape/device sanity check)
"""

# ── Stdlib ────────────────────────────────────────────────────────────────────
import os, sys, time, random, warnings
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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

try:
    import ipdb as pdb
except ImportError:
    import pdb

import optuna
from optuna.exceptions import TrialPruned
import pickle

# ═════════════════════════════════════════════════════════════════════════════
# 0.  CONFIG
# ═════════════════════════════════════════════════════════════════════════════
DEBUG             = False          # set True to hit breakpoints
BREAK_AFTER_BATCH = False          # set True for a single-batch sanity run

ROOT       = os.getcwd()

BASE_DIR   = os.path.join(ROOT, "sdss_dr17_dataset_fits/tensors")
CSV_PATH   = os.path.join(ROOT, "SkyObjects_FITS.csv")

OUT_DIR    = "/media/Ubunt_2/Project/pipeline_output_FITS"
CKPT_DIR   = os.path.join(OUT_DIR, "checkpoints_FITS")
NPY_DIR    = BASE_DIR
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

CLASSES    = ["GALAXY", "QSO", "STAR"]

IMG_SIZE   = 128
LATENT_DIM = 128
BATCH      = 16
EPOCHS_CNN = 100
EPOCHS_AE  = 100
EPOCHS_FUS = 100
LR         = 1e-4
SEED       = 42
TUNE_EPOCHS  = 15     # short trials — just enough to compare
TUNE_TRIALS  = 20    # number of hyperparameter combinations to try

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[config] device={DEVICE} latent_dim={LATENT_DIM}")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)



def debug_break(label: str):
    if DEBUG:
        print(f"\n[DEBUG] breakpoint → {label}")
        pdb.set_trace()


def ckpt_path(mode: str, label: str, epoch: int) -> str:
    """Canonical checkpoint filename."""
    return os.path.join(CKPT_DIR, f"ckpt_{mode}_{label}_ep{epoch:03d}.pth")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# 2.  DATASET
# ═════════════════════════════════════════════════════════════════════════════

class SDSSDataset(Dataset):
    """
    Returns (image_tensor, tab_tensor, label_int, filepath).
    tab_tensor : float32 (TAB_DIM,) — z-score normalised using a scaler
                 that was fit on the training split only.
    """
    def __init__(self, records: pd.DataFrame,
                 transform=None):
        self.records   = records.reset_index(drop=True)
        self.transform = transform
        self.class2idx = {c: i for i, c in enumerate(CLASSES)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]

        img_path = row["filepath"]
        arr = np.load(img_path).astype(np.float32)

        img = torch.from_numpy(arr).float()

        if img.ndim == 2:
            img = img.unsqueeze(0)      # (1,H,W)

        if self.transform:
            img = self.transform(img)

        label = self.class2idx[row["class"]]

        return img, label, img_path


def build_dataloaders(csv_path: str):
    """
    Reads CSV → engineers features → stratified split 70/20/10
    → fits StandardScaler on train only → returns DataLoaders.
    """
    df = pd.read_csv(csv_path)

    def _path(row):
        return os.path.join(BASE_DIR, f"{row['objid']}_{row['class']}.npy")
    df["filepath"] = df.apply(_path, axis=1)
    before = len(df)
    df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)
    print(f"[data] {len(df)}/{before} images found on disk")

    debug_break("after feature engineering")

    idx    = np.arange(len(df))
    labels = df["class"].values

    idx_tv,    idx_test  = train_test_split(
        idx,    test_size=0.10,       stratify=labels,         random_state=SEED)
    idx_train, idx_val   = train_test_split(
        idx_tv, test_size=0.10/0.90,  stratify=labels[idx_tv], random_state=SEED)

    print(f"[data] train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
    eval_tf = None

    train_ds = SDSSDataset(df.iloc[idx_train], train_tf)
    val_ds   = SDSSDataset(df.iloc[idx_val], eval_tf)
    test_ds  = SDSSDataset(df.iloc[idx_test], eval_tf)

    kw = dict(num_workers=2, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=BATCH, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=BATCH, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=BATCH, shuffle=False, **kw),
        df
    )


# ═════════════════════════════════════════════════════════════════════════════
# 3.  MODELS
# ═════════════════════════════════════════════════════════════════════════════

class ImageEncoder(nn.Module):
    """4-block conv backbone → flat (feature_dim,) vector."""
    def __init__(self):
        super().__init__()
        def _block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True),
                nn.MaxPool2d(2), nn.Dropout2d(0.1),
            )
        self.net = nn.Sequential(
            _block(1,   32),
            _block(32,  64),
            _block(64,  128),
            _block(128, 256),
        )
        self.feature_dim = 256 * 8 * 8   # 16 384

    def forward(self, x):
        return self.net(x).flatten(1)


# ── 3a. Multimodal CNN ────────────────────────────────────────────────────────
class MultimodalCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()

        self.encoder = ImageEncoder()

        self.head = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
        )

        self.classifier = nn.Linear(128, n_classes)

    def encode(self, x):
        x = self.encoder(x)
        return self.head(x)

    def forward(self, x):
        return self.classifier(self.head(self.encoder(x)))


# ── 3b. Convolutional Autoencoder (image only — unsupervised) ─────────────────
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1,   32,  3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(32,  64,  3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(64,  128, 3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(True),
        )
        self.enc_fc   = nn.Linear(256 * 8 * 8, latent_dim)
        self.dec_fc   = nn.Linear(latent_dim, 256 * 8 * 8)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,  3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(64,  32,  3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(32,  1,   3, stride=2, padding=1, output_padding=1), nn.Tanh(),
        )

    def encode(self, x):
        return self.enc_fc(self.enc_conv(x).flatten(1))

    def decode(self, z):
        return self.dec_conv(self.dec_fc(z).view(-1, 256, 8, 8))

    def forward(self, x):
        return self.decode(self.encode(x))

# ── 3c. Fused: CNN + AE latent + tabular ──────────────────────────────────────
class FusedClassifier(nn.Module):
    def __init__(self, mm_cnn: MultimodalCNN, ae: ConvAutoencoder,
                 latent_dim=LATENT_DIM, n_classes=3, input_shape=(3,224,224)):
        """
        Three frozen representation streams → trainable FC head:
        image  → ImageEncoder       (16 384-d) — discriminative visual features
        image  → AE encoder         (128-d)    — reconstruction-optimal morphology
        tabular → TabularEncoder    (64-d)     — colour + spatial context
        ─────────────────────────────────────────
        total input to head : 16 576-d
        """
        super().__init__()
        self.img_enc  = mm_cnn.encode
        self.ae_enc   = ae.encode
        self.ae_norm  = nn.LayerNorm(latent_dim)

        mm_cnn.eval()
        ae.eval()
        
        for p in mm_cnn.parameters():
            p.requires_grad = False
        for p in ae.parameters():
            p.requires_grad = False

        # compute fused feature size automatically
        print('input shape', input_shape, '-' ,*input_shape)
        device = next(mm_cnn.parameters()).device
        with torch.no_grad():
            
            dummy = torch.zeros(1, *input_shape).to(device)
            cnn_feat = self.img_enc(dummy)
            ae_feat  = self.ae_enc(dummy)
            fused = cnn_feat.size(1) + ae_feat.size(1)
            print(f"[FusedClassifier] Fused feature dim = {fused}")

        self.head = nn.Sequential(
            nn.Linear(fused, 256),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, n_classes),
        )

    def forward(self, img):
        with torch.no_grad():
            cnn_feat = self.img_enc(img)
            ae_feat = self.ae_enc(img)

        fused = torch.cat([cnn_feat, ae_feat], dim=1)

        return self.head(fused)


# ═════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def train_epoch_cls(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, n = 0., 0, 0
    for bi, (imgs, labels, _) in enumerate(loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(imgs)
        correct    += (out.argmax(1) == labels).sum().item()
        n          += len(imgs)
        if BREAK_AFTER_BATCH:
            debug_break(f"first train batch — out.shape={out.shape}")
            break
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch_cls(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0., 0, 0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * len(imgs)
        correct    += (out.argmax(1) == labels).sum().item()
        n          += len(imgs)
    return total_loss / n, correct / n


def train_epoch_ae(model, loader, criterion, optimizer):
    """AE is image-only; tab and label are ignored."""
    model.train()
    total_loss, n = 0., 0
    for bi, (imgs, _, _) in enumerate(loader):
        imgs = imgs.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), imgs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(imgs); n += len(imgs)
        if BREAK_AFTER_BATCH:
            debug_break("first AE batch")
            break
    return total_loss / n


@torch.no_grad()
def eval_epoch_ae(model, loader, criterion):
    model.eval()
    total_loss, n = 0., 0
    for imgs, _, _ in loader:
        imgs = imgs.to(DEVICE)
        total_loss += criterion(model(imgs), imgs).item() * len(imgs)
        n += len(imgs)
    return total_loss / n


def run_training(model, train_dl, val_dl, epochs, criterion, optimizer,
                 scheduler=None, mode="cls", label="Model"):
    """
    Generic training loop.  mode = 'cls' | 'ae'.
    Returns history dict {train_loss, val_loss, [train_acc, val_acc]}.

    Checkpointing strategy
    ----------------------
    • A checkpoint is written every epoch (not just on best-val).
      This makes resume unambiguous: find the highest epoch checkpoint,
      restore it, then continue from epoch+1.
    • Best weights are tracked separately in memory and restored at the end.
      The on-disk checkpoint is for crash recovery; the in-memory best_state
      is for final model quality.
    • Checkpoint saved as:  CKPT_DIR/ckpt_{mode}_{label}_ep{epoch:03d}.pth
    """
    history = {"train_loss": [], "val_loss": []}
    if mode == "cls":
        history["train_acc"] = []
        history["val_acc"]   = []

    best_val, best_state = float("inf"), None
    start_epoch = 1

    # ── Resume: find the most recent completed epoch checkpoint ──────────
    completed = [
        int(f.split("_ep")[1].replace(".pth", ""))
        for f in os.listdir(CKPT_DIR)
        if f == os.path.basename(ckpt_path(mode, label, int(
            f.split("_ep")[1].replace(".pth", "")
        ))) if "_ep" in f
    ]
    # simpler glob-free version:
    """
    completed = []
    for ep in range(1, epochs + 1):
        if os.path.exists(ckpt_path(mode, label, ep)):
            completed.append(ep)
    """

    if completed:
        last_ep = max(completed)
        print(f"[{label}] resuming from epoch {last_ep}/{epochs}")
        ckpt = torch.load(ckpt_path(mode, label, last_ep), map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        history   = ckpt["history"]
        best_val  = ckpt["best_val"]
        start_epoch = last_ep + 1

        if start_epoch > epochs:
            print(f"[{label}] already complete, restoring best weights")
            best_ckpt = torch.load(ckpt_path(mode, label, ckpt["best_epoch"]),
                                   map_location=DEVICE)
            model.load_state_dict(best_ckpt["model_state_dict"])
            return history

    best_epoch = completed[0] if completed else 1   # fallback; updated below

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        if mode == "cls":
            tr_loss, tr_acc = train_epoch_cls(model, train_dl, criterion, optimizer)
            vl_loss, vl_acc = eval_epoch_cls (model, val_dl,   criterion)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(vl_acc)
            print(f"[{label}] epoch {epoch:>3}/{epochs}  "
                  f"loss {tr_loss:.4f}/{vl_loss:.4f}  "
                  f"acc {tr_acc:.4f}/{vl_acc:.4f}  "
                  f"({time.time()-t0:.1f}s)")
        else:
            tr_loss = train_epoch_ae(model, train_dl, criterion, optimizer)
            vl_loss = eval_epoch_ae (model, val_dl,   criterion)
            print(f"[{label}] epoch {epoch:>3}/{epochs}  "
                  f"recon {tr_loss:.6f}/{vl_loss:.6f}  "
                  f"({time.time()-t0:.1f}s)")

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)

        if vl_loss < best_val:
            # delete previous best if it won't be caught by the rolling delete
            prev_best_path = ckpt_path(mode, label, best_epoch)
            if os.path.exists(prev_best_path) and best_epoch != epoch - 2:
                os.remove(prev_best_path)
            best_val   = vl_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Save every epoch — safe to resume from any point
        torch.save({
            "epoch":                epoch,
            "best_epoch":           best_epoch,
            "best_val":             best_val,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history":              history,
        }, ckpt_path(mode, label, epoch))

        old = ckpt_path(mode, label, epoch - 2)
        if os.path.exists(old) and (epoch - 2) != best_epoch:
            os.remove(old)

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                scheduler.step()   # called per batch, not per epoch
            else:
                scheduler.step(vl_loss)

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    debug_break(f"after training [{label}]")
    return history


# ═════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATION & VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, labels = [], []
    for imgs, lbls, _ in loader:
        imgs = imgs.to(DEVICE)
        preds.append(model(imgs).argmax(1).cpu().numpy())
        labels.append(lbls.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def plot_training_curves(history, label, save_path):
    has_acc = "train_acc" in history
    fig, axes = plt.subplots(1, 2 if has_acc else 1,
                             figsize=(12 if has_acc else 6, 4))
    if not has_acc: axes = [axes]
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title(f"{label} — Loss")
    axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    if has_acc:
        axes[1].plot(history["train_acc"], label="Train")
        axes[1].plot(history["val_acc"],   label="Val")
        axes[1].set_title(f"{label} — Accuracy")
        axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[plot] {save_path}")


def plot_confusion_matrix(preds, labels, label, save_path):
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_title(f"{label} — Confusion Matrix (normalised)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[plot] {save_path}")


def report_metrics(preds, labels, label):
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro")
    print(f"\n{'─'*55}\n {label}\n{'─'*55}")
    print(f" Accuracy : {acc:.4f}   Macro-F1 : {f1:.4f}")
    print(classification_report(labels, preds, target_names=CLASSES))
    return {"label": label, "accuracy": acc, "macro_f1": f1}


def plot_comparison(metrics_list, save_path):
    names = [m["label"] for m in metrics_list]
    accs  = [m["accuracy"] for m in metrics_list]
    f1s   = [m["macro_f1"] for m in metrics_list]
    x, w  = np.arange(len(names)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, accs, w, label="Accuracy", color="#4C82C4")
    ax.bar(x + w/2, f1s,  w, label="Macro-F1",  color="#E07B53")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title("Model Comparison — With and Without Latent Space")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + w/2, f + 0.01, f"{f:.3f}", ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[plot] {save_path}")


def plot_ae_reconstructions(ae, loader, save_path, n=8):
    ae.eval()
    imgs, _, _ = next(iter(loader))
    imgs = imgs[:n].to(DEVICE)
    with torch.no_grad():
        recon = ae(imgs).cpu()
    imgs = imgs.cpu()
    denorm = lambda t: ((t - t.min()) / (t.max() - t.min() + 1e-8)).clamp(0, 1)
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i in range(n):
        axes[0, i].imshow(denorm(imgs[i]).squeeze(), cmap='gray'); axes[0, i].axis("off")
        axes[1, i].imshow(denorm(recon[i]).squeeze(), cmap='gray'); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=10)
    plt.suptitle("Autoencoder Reconstructions", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[plot] {save_path}")


def plot_feature_distributions(df, save_path):
    """
    Box plots of every tabular feature per class.
    Useful sanity check: if the features are NOT discriminative,
    the distributions will heavily overlap → model will rely on images only.
    """
    plot_cols = ["u", "g", "r", "i", "z", "u_g", "g_r", "r_i", "i_z"]
    palette   = {"GALAXY": "#4C82C4", "QSO": "#E07B53", "STAR": "#5FAD56"}
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))

    for ax, col in zip(axes.flat, plot_cols):
        data  = [df[df["class"] == cls][col].dropna().values for cls in CLASSES]
        bp    = ax.boxplot(data, patch_artist=True, widths=0.5,
                           medianprops=dict(color="black", linewidth=2),
                           flierprops=dict(marker=".", markersize=1, alpha=0.3))
        for patch, cls in zip(bp["boxes"], CLASSES):
            patch.set_facecolor(palette[cls]); patch.set_alpha(0.65)
        ax.set_title(col, fontsize=10)
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(CLASSES, fontsize=8, rotation=15)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Tabular Feature Distributions per Class", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[plot] {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 6.  MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def tune_hyperparameters(train_dl, val_dl):
    def objective(trial):
        lr      = trial.suggest_float("lr",      1e-5, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1,  0.6)
        
        model = MultimodalCNN(n_classes=3).to(DEVICE)
        # patch dropout in the head
        for m in model.head:
            if isinstance(m, nn.Dropout):
                m.p = dropout
        
        opt   = optim.Adam(model.parameters(), lr=lr)
        crit  = nn.CrossEntropyLoss()

        for epoch in range(TUNE_EPOCHS):
            train_epoch_cls(model, train_dl, crit, opt)
            vl, _ = eval_epoch_cls(model, val_dl, crit)
            trial.report(vl, epoch)
            if trial.should_prune():
                raise TrialPruned()

        return vl

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=2)
    study   = optuna.create_study(direction="minimize",
                                  sampler=sampler, pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=TUNE_TRIALS)

    print(f"[tune] best trial:  lr={study.best_params['lr']:.2e}  "
          f"dropout={study.best_params['dropout']:.3f}  "
          f"val_loss={study.best_value:.4f}")
    return study.best_params

def main():

    print("\n" + "═"*58)
    print("  Stage 0 : Data & Feature Engineering")
    print("═"*58)

    train_dl, val_dl, test_dl, df= build_dataloaders(CSV_PATH)
    #plot_feature_distributions(df, os.path.join(OUT_DIR, "feature_distributions.png"))

    print("\n" + "═"*58)
    print("  Stage 0b : Hyperparameter Tuning")
    print("═"*58)
    #best_params = tune_hyperparameters(train_dl, val_dl)
    best_lr      = LR
    best_dropout = 0.3

    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═"*58)
    print("  Stage 1 : Multimodal CNN  (image + tabular)")
    print("═"*58)

    mm_cnn = MultimodalCNN(n_classes=3).to(DEVICE)
    for m in mm_cnn.head:
        if isinstance(m, nn.Dropout): m.p = best_dropout
    cnn_opt   = optim.Adam(mm_cnn.parameters(), lr=best_lr)

    debug_break("before MM-CNN training")

    weights = torch.tensor([1.0, 1.0, 2.0], device=DEVICE)

    cnn_hist = run_training(mm_cnn, train_dl, val_dl, EPOCHS_CNN,
                            nn.CrossEntropyLoss(weight=weights), cnn_opt,
                            mode="cls", label="MM-CNN")

    plot_training_curves(cnn_hist, "Multimodal CNN",
                         os.path.join(OUT_DIR, "cnn_curves.png"))
    cnn_preds, cnn_labels = predict(mm_cnn, test_dl)
    cnn_metrics = report_metrics(cnn_preds, cnn_labels, "Multimodal CNN")
    plot_confusion_matrix(cnn_preds, cnn_labels, "Multimodal CNN",
                          os.path.join(OUT_DIR, "cnn_confusion.png"))
    torch.save(mm_cnn.state_dict(), os.path.join(OUT_DIR, "mm_cnn.pt"))
    debug_break("after MM-CNN evaluation")

    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═"*58)
    print("  Stage 2 : Convolutional Autoencoder  (image only)")
    print("═"*58)

    ae       = ConvAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae_opt   = optim.Adam(ae.parameters(), lr=best_lr)
    ae_sched = optim.lr_scheduler.ReduceLROnPlateau(ae_opt, patience=3, factor=0.5)

    debug_break("before AE training")

    ae_hist = run_training(ae, train_dl, val_dl, EPOCHS_AE,
                           nn.MSELoss(), ae_opt, ae_sched,
                           mode="ae", label="AE")

    plot_training_curves(ae_hist, "Autoencoder",
                         os.path.join(OUT_DIR, "ae_curves.png"))
    plot_ae_reconstructions(ae, test_dl,
                            os.path.join(OUT_DIR, "ae_reconstructions.png"))
    torch.save(ae.state_dict(), os.path.join(OUT_DIR, "ae.pt"))
    debug_break("after AE training")

    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═"*58)
    print("  Stage 3 : Fused  (image + AE latent + tabular)")
    print("═"*58)

    mm_cnn.load_state_dict(torch.load(os.path.join(OUT_DIR, "mm_cnn.pt"), map_location=DEVICE))
    ae.load_state_dict    (torch.load(os.path.join(OUT_DIR, "ae.pt"),     map_location=DEVICE))

    imgs, _, _ = next(iter(train_dl))
    INPUT_SHAPE = imgs.shape[1:]  # (C, H, W)

    fused = FusedClassifier(mm_cnn, ae, latent_dim=LATENT_DIM, input_shape=INPUT_SHAPE).to(DEVICE)
    fus_opt   = optim.Adam(filter(lambda p: p.requires_grad, fused.parameters()), lr=LR)
    fus_sched = optim.lr_scheduler.ReduceLROnPlateau(fus_opt, patience=3, factor=0.5)

    debug_break("before Fused training")

    fus_hist = run_training(fused, train_dl, val_dl, EPOCHS_FUS,
                            nn.CrossEntropyLoss(weight=weights), fus_opt, fus_sched,
                            mode="cls", label="Fused")

    plot_training_curves(fus_hist, "Fused CNN+AE+Tab",
                         os.path.join(OUT_DIR, "fused_curves.png"))
    fus_preds, fus_labels = predict(fused, test_dl)
    fus_metrics = report_metrics(fus_preds, fus_labels, "Fused CNN+AE+Tab")
    plot_confusion_matrix(fus_preds, fus_labels, "Fused CNN+AE+Tab",
                          os.path.join(OUT_DIR, "fused_confusion.png"))
    torch.save(fused.state_dict(), os.path.join(OUT_DIR, "fused.pt"))
    debug_break("after Fused evaluation")

    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═"*58)
    print("  Stage 4 : Comparison")
    print("═"*58)

    plot_comparison([cnn_metrics, fus_metrics],
                    os.path.join(OUT_DIR, "comparison.png"))

    print("\n[summary]")
    for m in [cnn_metrics, fus_metrics]:
        print(f"  {m['label']:<30}  acc={m['accuracy']:.4f}  f1={m['macro_f1']:.4f}")

    print(f"\n[done] outputs → ./{OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")

if __name__ == "__refit_scaler__":
    import pickle
    df = pd.read_csv(CSV_PATH)
    def _path(row):
        return os.path.join(BASE_DIR, f"{row['objid']}_{row['label']}.npy")
    df["filepath"] = df.apply(_path, axis=1)
    df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)
    idx = np.arange(len(df))
    idx_tv, _ = train_test_split(idx, test_size=0.10, stratify=df["class"].values, random_state=SEED)
    idx_train, _ = train_test_split(idx_tv, test_size=0.10/0.90, stratify=df["class"].values[idx_tv], random_state=SEED)
    print(f"[done] scaler saved to {OUT_DIR}/scaler.pkl")

if __name__ == "__main__":
    main()