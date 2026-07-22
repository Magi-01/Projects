"""
SDSS Multimodal Classification Pipeline
=========================================
Every classifier receives up to TWO inputs:
  .) image   — 128×128 RGB JPEG cutout
  .) tabular — photometric feature vector derived from the CSV, one of:

     tabular1 : u, g, r, i           (4 raw magnitudes)
                u-g, g-r, r-i        (3 colour indices)
                ──────────────────────────────────────
                TAB_DIM = 7 total

     tabular2 : u, g, r, i, z            (5 raw magnitudes)
                u-g, g-r, r-i, z-i       (4 colour indices)
                ──────────────────────────────────────────
                TAB_DIM = 9 total

Image encoder: ResNet18 (ImageNet-pretrained) backbone for Stages 1-3.

Image Augmentation:
The training images are:
    .) Horizontally and Vertically flipped
    .) Rotated (full range) and rescaled

Stage 1 : ResNet CNN with Adam                (image → class)
Stage 2 : ResNet CNN with Adam and table1     (image + tabular1 → class)
Stage 3 : ResNet CNN with Adam and table2     (image + tabular2 → class)
Stage 4 : VAE pretraining (unsupervised) + frozen-latent classifier
              4a. train a conv-VAE to reconstruct images (recon + KL loss)
              4b. freeze its encoder, classify from the 64-d mu vector
Stage 5 : Metrics

Latent Visualisation
---------------------
After VAE pretraining, the test-set latent means (mu) are projected to 2-D
with both t-SNE and UMAP and coloured by true class, plus a grid of
original-vs-reconstructed images, to sanity-check what the unsupervised
encoder actually learned before it's handed to the classifier.

Stages 1-3 share one stratified 70/20/10 train/val/test split (fit once in
build_splits), and Stage 4 reuses the same split, so all four models are
directly comparable.

ResNet fine-tuning schedule (Stages 1-3)
-----------------------------------------
The backbone is frozen for the first `freeze_epochs` epochs (head/tabular
branch only), then unfrozen with a lower LR for the remaining epochs.

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
import os, time, random, warnings, pickle
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
import torchvision.models as models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

try:
    import ipdb as pdb
except ImportError:
    import pdb

# ═════════════════════════════════════════════════════════════════════════════
# 0.  CONFIG
# ═════════════════════════════════════════════════════════════════════════════
DEBUG             = False          # set True to hit breakpoints
BREAK_AFTER_BATCH = False          # set True for a single-batch sanity run

ROOT       = os.getcwd()
BASE_DIR   = os.path.join(ROOT, "sdss_dr17_dataset")  # root where class subfolders live
CSV_PATH   = os.path.join(ROOT, "SkyObjects.csv")
OUT_DIR    = "/media/Ubunt_2/Project/pipeline_output_resnet_vae"
CKPT_DIR   = os.path.join(OUT_DIR, "checkpoints")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

CLASSES    = ["GALAXY", "QSO", "STAR"]

# Raw photometric columns present in the CSV (before feature engineering)
RAW_TAB_COLS = ["u", "g", "r", "i", "z"]

# After build_features() the two candidate feature vectors are:
TAB_DIM1   = 7
FEAT_COLS1 = ["u", "g", "r", "i",
              "u_g", "g_r", "r_i"]
TAB_DIM2   = 9
FEAT_COLS2 = ["u", "g", "r", "i", "z",
              "u_g", "g_r", "r_i", "z_i"]
# Union of both, used only to decide which rows to drop for missing data —
# keeping the same rows for all stages keeps their metrics comparable.
FEAT_COLS_ALL = list(dict.fromkeys(FEAT_COLS1 + FEAT_COLS2))

IMG_SIZE       = 128
BATCH          = 128
EPOCHS         = 15     # Stages 1-3 (classification) and Stage 4b/4c (VAE classifier)
EPOCHS_VAE     = 10     # Stage 4a (VAE pretraining) — reconstruction needs longer
VAE_LATENT_DIM = 64
LR             = 1e-4
SEED           = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[config] device={DEVICE}  tabular1_dim={TAB_DIM1}  tabular2_dim={TAB_DIM2}")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def debug_break(label: str):
    if DEBUG:
        print(f"\n[DEBUG] breakpoint → {label}")
        pdb.set_trace()


def ckpt_path(label: str, epoch: int) -> str:
    """Canonical checkpoint filename."""
    return os.path.join(CKPT_DIR, f"ckpt_{label}_ep{epoch:03d}.pth")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives all tabular features from raw CSV columns.

    Colour indices (u-g, g-r, r-i, z-i)
    -------------------------------------
    Magnitude differences encode the spectral energy distribution slope.
    They are among the strongest class separators in SDSS photometry:
      • Stars    → tight, well-defined colour locus
      • QSOs     → blue excess, u-g < 0.6 typically
      • Galaxies → redder, broader spread
    """
    df = df.copy()

    # ── clip raw magnitudes (SDSS sentinel values: 999, -9999) ──────────
    for col in RAW_TAB_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(-5, 35)

    # ── colour indices ────────────────────────────────────────────────────
    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["z_i"] = df["z"] - df["i"]

    # ── drop rows with any NaN in a feature needed by EITHER tabular set ─
    before = len(df)
    df = df.dropna(subset=FEAT_COLS_ALL).reset_index(drop=True)
    print(f"[features] dropped {before - len(df)} rows with NaN features, "
          f"{len(df)} remain")

    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2.  DATASET
# ═════════════════════════════════════════════════════════════════════════════

class SDSSDataset(Dataset):
    """
    Returns (image_tensor, tab_tensor, label_int, filepath).

    feat_cols : list of tabular columns to use, or None for image-only
                (Stage 1, Stage 4) — in which case tab_tensor is an empty
                tensor that the model simply never looks at.
    tab_tensor, when present, is z-score normalised using a scaler that
    was fit on the training split only.
    """
    def __init__(self, records: pd.DataFrame, feat_cols, scaler, transform=None):
        self.records   = records.reset_index(drop=True)
        self.transform = transform
        self.class2idx = {c: i for i, c in enumerate(CLASSES)}

        if feat_cols:
            raw = self.records[feat_cols].values.astype(np.float32)
            self.tab_array = scaler.transform(raw).astype(np.float32)
        else:
            self.tab_array = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row   = self.records.iloc[idx]
        label = self.class2idx[row["class"]]
        path  = row["filepath"]
        if self.tab_array is not None:
            tab = torch.tensor(self.tab_array[idx], dtype=torch.float32)
        else:
            tab = torch.empty(0, dtype=torch.float32)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, tab, label, path


def build_splits(csv_path: str):
    """
    Reads CSV → engineers features → ONE stratified 70/20/10 split shared
    by all stages, so their held-out metrics are directly comparable.
    Returns (df, idx_train, idx_val, idx_test).
    """
    df = pd.read_csv(csv_path)

    def _path(row):
        return os.path.join(BASE_DIR, str(row["class"]), f"{row['objid']}.jpg")
    df["filepath"] = df.apply(_path, axis=1)
    before = len(df)
    df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)
    print(f"[data] {len(df)}/{before} images found on disk")

    df = build_features(df)
    debug_break("after feature engineering")

    idx    = np.arange(len(df))
    labels = df["class"].values

    idx_tv,    idx_test  = train_test_split(
        idx,    test_size=0.10,       stratify=labels,         random_state=SEED)
    idx_train, idx_val   = train_test_split(
        idx_tv, test_size=0.10/0.90,  stratify=labels[idx_tv], random_state=SEED)

    print(f"[data] train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")
    return df, idx_train, idx_val, idx_test


def build_dataloaders(df, idx_train, idx_val, idx_test, feat_cols, scaler_name,
                       normalize="imagenet"):
    """
    Builds train/val/test DataLoaders for a single stage.

    feat_cols   : FEAT_COLS1 / FEAT_COLS2 / None (image-only).
    scaler_name : filename under OUT_DIR to persist the fitted scaler to,
                  or None when feat_cols is None (nothing to fit).
    normalize   : "imagenet" — ResNet stages (mean/std shift, values can go
                               negative).
                  "unit"     — VAE stage — ToTensor already scales to [0,1]
                               and the decoder ends in Sigmoid, so no further
                               shift is applied; reconstruction targets must
                               stay in [0,1].
    """
    scaler = None
    if feat_cols:
        scaler = StandardScaler()
        scaler.fit(df.iloc[idx_train][feat_cols].values.astype(np.float32))
        with open(os.path.join(OUT_DIR, scaler_name), "wb") as f:
            pickle.dump(scaler, f)

    if normalize == "imagenet":
        norm_tf = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    elif normalize == "unit":
        norm_tf = None
    else:
        raise ValueError(f"unknown normalize={normalize!r}")

    train_tf_list = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=180, scale=(0.85, 1.15)),  # rotate + rescale
        transforms.ToTensor(),
    ]
    eval_tf_list = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
    if norm_tf is not None:
        train_tf_list.append(norm_tf)
        eval_tf_list.append(norm_tf)

    train_tf = transforms.Compose(train_tf_list)
    eval_tf  = transforms.Compose(eval_tf_list)

    train_ds = SDSSDataset(df.iloc[idx_train], feat_cols, scaler, train_tf)
    val_ds   = SDSSDataset(df.iloc[idx_val],   feat_cols, scaler, eval_tf)
    test_ds  = SDSSDataset(df.iloc[idx_test],  feat_cols, scaler, eval_tf)

    kw = dict(num_workers=2, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=BATCH, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=BATCH, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=BATCH, shuffle=False, **kw),
    )


# ═════════════════════════════════════════════════════════════════════════════
# 3.  MODELS
# ═════════════════════════════════════════════════════════════════════════════

class ImageEncoder(nn.Module):
    """ResNet18 (ImageNet-pretrained) backbone, final FC layer removed."""
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.net = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 512

    def forward(self, x):
        return self.net(x).flatten(1)


# ── 3a. Stage 1 : image-only CNN ──────────────────────────────────────────────
class ImageCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.img_enc = ImageEncoder()
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, img):
        return self.head(self.img_enc(img))


# ── 3b. Stage 2 / 3 : image + tabular CNN ─────────────────────────────────────
class MultimodalCNN(nn.Module):
    def __init__(self, tab_dim, n_classes=3):
        super().__init__()
        self.img_enc = ImageEncoder()
        self.tab_enc = nn.Sequential(
            nn.Linear(tab_dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64,      64), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(512 + 64, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256,      128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, img, tab):
        img_feat = self.img_enc(img)
        tab_feat = self.tab_enc(tab)
        return self.head(torch.cat([img_feat, tab_feat], dim=1))


# ── 3c. Stage 4 : convolutional VAE + frozen-latent classifier ───────────────
class VAEEncoder(nn.Module):
    """128×128×3 → (mu, logvar), each (latent_dim,)."""
    def __init__(self, latent_dim=VAE_LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,   32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32,  64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64,  128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
        )
        self.fc     = nn.Linear(256 * 8 * 8, 512)
        self.mu     = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.conv(x).flatten(1)
        h = torch.relu(self.fc(x))
        return self.mu(h), self.logvar(h)


class VAEDecoder(nn.Module):
    """(latent_dim,) → 128×128×3, values in [0, 1] (Sigmoid output)."""
    def __init__(self, latent_dim=VAE_LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32,  3,   4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 8, 8)
        return self.deconv(x)


class VAE(nn.Module):
    """Encoder + reparameterization + decoder, for unsupervised pretraining."""
    def __init__(self, latent_dim=VAE_LATENT_DIM):
        super().__init__()
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(latent_dim)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def vae_loss(recon, img, mu, logvar):
    """Per-sample-averaged reconstruction + KL, so the scale is batch-size
    independent (unlike a raw reduction='sum')."""
    n = img.size(0)
    recon_loss = nn.functional.mse_loss(recon, img, reduction="sum") / n
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / n
    return recon_loss + kl, recon_loss, kl


class VAEClassifier(nn.Module):
    """Stage 4b: frozen (or fine-tunable) VAE encoder + small classifier head
    on top of the deterministic mu vector (not a sampled z)."""
    def __init__(self, vae: VAE, latent_dim=VAE_LATENT_DIM, n_classes=3):
        super().__init__()
        self.encoder = vae.encoder
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, img):
        mu, _ = self.encoder(img)
        return self.head(mu)
    
# ── 4c. VAE + tabular CNN ─────────────────────────────────────
class MultimodalVAE(nn.Module):

    def __init__(
        self,
        vae: VAE,
        tab_dim,
        latent_dim=VAE_LATENT_DIM,
        n_classes=3
    ):
        super().__init__()

        self.encoder = vae.encoder

        self.tab_enc = nn.Sequential(
            nn.Linear(tab_dim,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(latent_dim+64,64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(32,n_classes),
        )


    def forward(self,img,tab):

        # deterministic VAE feature
        mu,_ = self.encoder(img)

        tab_feat = self.tab_enc(tab)

        fused = torch.cat(
            [mu,tab_feat],
            dim=1
        )

        return self.head(fused)


# ═════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def train_epoch_cls(model, loader, criterion, optimizer, use_tab, scheduler=None):
    model.train()
    total_loss, correct, n = 0., 0, 0
    for bi, (imgs, tab, labels, _) in enumerate(loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        if use_tab:
            tab = tab.to(DEVICE)
            out = model(imgs, tab)
        else:
            out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # OneCycleLR is stepped once per BATCH, not per epoch
        if isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        total_loss += loss.item() * len(imgs)
        correct    += (out.argmax(1) == labels).sum().item()
        n          += len(imgs)
        if BREAK_AFTER_BATCH:
            debug_break(f"first train batch — out.shape={out.shape}")
            break
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch_cls(model, loader, criterion, use_tab):
    model.eval()
    total_loss, correct, n = 0., 0, 0
    for imgs, tab, labels, _ in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        if use_tab:
            tab = tab.to(DEVICE)
            out = model(imgs, tab)
        else:
            out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * len(imgs)
        correct    += (out.argmax(1) == labels).sum().item()
        n          += len(imgs)
    return total_loss / n, correct / n


def _build_finetune_optimizer(model, remaining_epochs, steps_per_epoch):
    """
    Unfreezes the ResNet backbone and builds a fresh optimizer/scheduler pair
    for the fine-tuning phase (lower LR for the backbone than the head /
    tabular branch). Used both at the freeze→finetune transition inside the
    training loop, and when RESUMING a checkpoint that already falls inside
    that phase (in which case the previously-saved optimizer_state_dict no
    longer matches — its param groups only covered the frozen-phase
    parameters — so it can't just be reloaded; we rebuild it here instead,
    at the cost of losing Adam's momentum state for the just-unfrozen
    backbone params).
    """
    for p in model.img_enc.net.parameters():
        p.requires_grad = True

    if hasattr(model, "tab_enc"):
        param_groups = [
            {"params": model.img_enc.parameters(), "lr": 1e-5},
            {"params": model.tab_enc.parameters(), "lr": 1e-4},
            {"params": model.head.parameters(),    "lr": 1e-4},
        ]
    else:
        param_groups = [
            {"params": model.img_enc.parameters(), "lr": 1e-5},
            {"params": model.head.parameters(),    "lr": 1e-4},
        ]

    optimizer = torch.optim.Adam(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in optimizer.param_groups],
        epochs=max(remaining_epochs, 1),
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
    )
    return optimizer, scheduler


def run_training(model, train_dl, val_dl, epochs, criterion, optimizer,
                  scheduler=None, label="Model", use_tab=True,
                  freeze_epochs=5):
    """
    Classification training loop shared by Stages 1-3 and Stage 4b.
    Returns history dict {train_loss, val_loss, train_acc, val_acc}.

    ResNet freeze/unfreeze (Stages 1-3 only)
    -----------------------------------------
    model.img_enc.net is frozen for the first `freeze_epochs` epochs, then
    unfrozen with a lower LR for the rest of training. Pass
    freeze_epochs >= epochs (e.g. for Stage 4b's VAE encoder, which has no
    .img_enc.net attribute) to disable this entirely and train with a fixed
    optimizer/scheduler throughout.

    Checkpointing strategy
    ----------------------
    • A checkpoint is written every epoch (not just on best-val).
      This makes resume unambiguous: find the highest epoch checkpoint,
      restore it, then continue from epoch+1.
    • Best weights are tracked separately in memory and restored at the end.
      The on-disk checkpoint is for crash recovery; the in-memory best_state
      is for final model quality.
    • Checkpoint saved as:  CKPT_DIR/ckpt_{label}_ep{epoch:03d}.pth
    """
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_epoch, best_state = float("inf"), 1, None
    start_epoch = 1

    # ── Resume: find the most recent completed epoch checkpoint ──────────
    completed = [ep for ep in range(1, epochs + 1)
                 if os.path.exists(ckpt_path(label, ep))]

    if completed:
        last_ep = max(completed)
        print(f"[{label}] resuming from epoch {last_ep}/{epochs}")
        ckpt = torch.load(ckpt_path(label, last_ep), map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])

        if last_ep <= freeze_epochs:
            # still inside the frozen-backbone phase — optimizer shape matches
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            print(f"[{label}] resuming inside the fine-tuning phase — "
                  f"rebuilding optimizer/scheduler")
            optimizer, scheduler = _build_finetune_optimizer(
                model, epochs - last_ep, len(train_dl))

        history     = ckpt["history"]
        best_val    = ckpt["best_val"]
        best_epoch  = ckpt["best_epoch"]
        start_epoch = last_ep + 1

        if start_epoch > epochs:
            print(f"[{label}] already complete, restoring best weights")
            best_ckpt = torch.load(ckpt_path(label, best_epoch), map_location=DEVICE)
            model.load_state_dict(best_ckpt["model_state_dict"])
            return history

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # Unfreeze backbone once, right after the frozen phase ends
        if epoch == freeze_epochs + 1:
            print(f"[{label}] Unfreezing ResNet backbone...")
            optimizer, scheduler = _build_finetune_optimizer(
                model, epochs - freeze_epochs, len(train_dl))

        tr_loss, tr_acc = train_epoch_cls(model, train_dl, criterion, optimizer,
                                           use_tab, scheduler)
        vl_loss, vl_acc = eval_epoch_cls(model, val_dl, criterion, use_tab)

        history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(vl_acc)

        print(f"[{label}] epoch {epoch:>3}/{epochs}  "
              f"loss {tr_loss:.4f}/{vl_loss:.4f}  "
              f"acc {tr_acc:.4f}/{vl_acc:.4f}  "
              f"({time.time()-t0:.1f}s)")

        if vl_loss < best_val:
            best_val, best_epoch = vl_loss, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Save every epoch — safe to resume from any point
        torch.save({
            "epoch":                epoch,
            "best_epoch":           best_epoch,
            "best_val":             best_val,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history":              history,
        }, ckpt_path(label, epoch))

        # ReduceLROnPlateau (or similar) is stepped once per epoch;
        # OneCycleLR was already stepped per batch inside train_epoch_cls.
        if scheduler and not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step(vl_loss)

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    debug_break(f"after training [{label}]")
    return history


def train_epoch_vae(model, loader, optimizer):
    """VAE is image-only and unsupervised; tab and label are ignored."""
    model.train()
    total, total_recon, total_kl, n = 0., 0., 0., 0
    for imgs, _, _, _ in loader:
        imgs = imgs.to(DEVICE)
        optimizer.zero_grad()
        recon, mu, logvar = model(imgs)
        loss, recon_loss, kl = vae_loss(recon, imgs, mu, logvar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total       += loss.item()       * len(imgs)
        total_recon += recon_loss.item() * len(imgs)
        total_kl    += kl.item()         * len(imgs)
        n           += len(imgs)
        if BREAK_AFTER_BATCH:
            debug_break(f"first VAE batch — recon.shape={recon.shape}")
            break
    return total / n, total_recon / n, total_kl / n


@torch.no_grad()
def eval_epoch_vae(model, loader):
    model.eval()
    total, total_recon, total_kl, n = 0., 0., 0., 0
    for imgs, _, _, _ in loader:
        imgs = imgs.to(DEVICE)
        recon, mu, logvar = model(imgs)
        loss, recon_loss, kl = vae_loss(recon, imgs, mu, logvar)
        total       += loss.item()       * len(imgs)
        total_recon += recon_loss.item() * len(imgs)
        total_kl    += kl.item()         * len(imgs)
        n           += len(imgs)
    return total / n, total_recon / n, total_kl / n


def run_training_vae(model, train_dl, val_dl, epochs, optimizer,
                      scheduler=None, label="VAE"):
    """
    Unsupervised training loop for Stage 4a (VAE pretraining). Mirrors
    run_training's checkpointing strategy (see its docstring) but tracks
    reconstruction/KL instead of accuracy, and has no freeze/unfreeze phase.
    """
    history = {"train_loss": [], "val_loss": [],
               "train_recon": [], "val_recon": [],
               "train_kl": [], "val_kl": []}
    best_val, best_epoch, best_state = float("inf"), 1, None
    start_epoch = 1

    completed = [ep for ep in range(1, epochs + 1)
                 if os.path.exists(ckpt_path(label, ep))]

    if completed:
        last_ep = max(completed)
        print(f"[{label}] resuming from epoch {last_ep}/{epochs}")
        ckpt = torch.load(ckpt_path(label, last_ep), map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        history     = ckpt["history"]
        best_val    = ckpt["best_val"]
        best_epoch  = ckpt["best_epoch"]
        start_epoch = last_ep + 1

        if start_epoch > epochs:
            print(f"[{label}] already complete, restoring best weights")
            best_ckpt = torch.load(ckpt_path(label, best_epoch), map_location=DEVICE)
            model.load_state_dict(best_ckpt["model_state_dict"])
            return history

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        tr_loss, tr_recon, tr_kl = train_epoch_vae(model, train_dl, optimizer)
        vl_loss, vl_recon, vl_kl = eval_epoch_vae(model, val_dl)

        history["train_loss"].append(tr_loss);   history["val_loss"].append(vl_loss)
        history["train_recon"].append(tr_recon); history["val_recon"].append(vl_recon)
        history["train_kl"].append(tr_kl);       history["val_kl"].append(vl_kl)

        print(f"[{label}] epoch {epoch:>3}/{epochs}  "
              f"loss {tr_loss:.2f}/{vl_loss:.2f}  "
              f"recon {tr_recon:.2f}/{vl_recon:.2f}  "
              f"kl {tr_kl:.2f}/{vl_kl:.2f}  ({time.time()-t0:.1f}s)")

        if vl_loss < best_val:
            best_val, best_epoch = vl_loss, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        torch.save({
            "epoch":                epoch,
            "best_epoch":           best_epoch,
            "best_val":             best_val,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history":              history,
        }, ckpt_path(label, epoch))

        if scheduler:
            scheduler.step(vl_loss)

    if best_state:
        model.load_state_dict(best_state)

    debug_break(f"after training [{label}]")
    return history


# ═════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATION & VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(model, loader, use_tab):
    model.eval()
    preds, labels = [], []
    for imgs, tab, lbls, _ in loader:
        imgs = imgs.to(DEVICE)
        if use_tab:
            tab = tab.to(DEVICE)
            out = model(imgs, tab)
        else:
            out = model(imgs)
        preds.append(out.argmax(1).cpu().numpy())
        labels.append(lbls.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def plot_training_curves(history, label, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title(f"{label} — Loss")
    axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Val")
    axes[1].set_title(f"{label} — Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[plot] {save_path}")


def plot_vae_training_curves(history, label, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(
        axes,
        ["train_loss", "train_recon", "train_kl"],
        ["Total loss (recon + KL)", "Reconstruction (MSE)", "KL divergence"],
    ):
        ax.plot(history[key], label="Train")
        ax.plot(history[key.replace("train_", "val_")], label="Val")
        ax.set_title(f"{label} — {title}")
        ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True, alpha=0.3)
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
    ax.set_title("Model Comparison")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + w/2, f + 0.01, f"{f:.3f}", ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[plot] {save_path}")


def plot_feature_distributions(df, save_path):
    """
    Box plots of every tabular feature per class.
    Useful sanity check: if the features are NOT discriminative,
    the distributions will heavily overlap → model will rely on images only.
    """
    plot_cols = ["u", "g", "r", "i", "z", "u_g", "g_r", "r_i", "z_i"]
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

    axes.flat[-1].axis("off")   # 10th subplot slot is unused (9 feature columns)
    plt.suptitle("Tabular Feature Distributions per Class", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[plot] {save_path}")


def plot_vae_reconstructions(vae, loader, save_path, n=8):
    """Original vs reconstructed image grid — quick sanity check that the
    VAE is keeping meaningful structure, not just averaging away detail."""
    vae.eval()
    imgs, _, _, _ = next(iter(loader))
    imgs = imgs[:n].to(DEVICE)
    with torch.no_grad():
        recon, _, _ = vae(imgs)
    imgs, recon = imgs.cpu(), recon.cpu()

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i in range(n):
        axes[0, i].imshow(imgs[i].permute(1, 2, 0).clamp(0, 1)); axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].permute(1, 2, 0).clamp(0, 1)); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=10)
    plt.suptitle("VAE Reconstructions", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[plot] {save_path}")


@torch.no_grad()
def _collect_latents(vae, loader, max_points=2000):
    """Runs the VAE encoder over a loader and returns (mu, true_label) pairs,
    stopping once max_points samples have been collected (t-SNE/UMAP get
    slow well before a full test set is needed to see class structure)."""
    vae.eval()
    mus, labels, seen = [], [], 0
    for imgs, _, lbls, _ in loader:
        imgs = imgs.to(DEVICE)
        mu, _ = vae.encoder(imgs)
        mus.append(mu.cpu().numpy())
        labels.append(lbls.numpy())
        seen += len(lbls)
        if seen >= max_points:
            break
    mus    = np.concatenate(mus)[:max_points]
    labels = np.concatenate(labels)[:max_points]
    return mus, labels


def _scatter_latent(emb, labels, title, save_path):
    palette = {"GALAXY": "#4C82C4", "QSO": "#E07B53", "STAR": "#5FAD56"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cls in enumerate(CLASSES):
        mask = labels == i
        ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.6,
                   color=palette[cls], label=cls)
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[plot] {save_path}")


def plot_latent_tsne(vae, loader, label, save_path, max_points=2000):
    """2-D t-SNE of the VAE's mu vectors, coloured by true class — shows
    whether the *unsupervised* latent space already separates the classes
    before any classifier head has been trained on it."""
    mus, labels = _collect_latents(vae, loader, max_points)
    perplexity = min(30, max(5, (len(mus) - 1) // 3))
    emb = TSNE(n_components=2, init="pca", random_state=SEED,
               perplexity=perplexity).fit_transform(mus)
    _scatter_latent(emb, labels, f"{label} — t-SNE", save_path)


def plot_latent_umap(vae, loader, label, save_path, max_points=2000):
    """Same idea as plot_latent_tsne but with UMAP, which tends to preserve
    more global structure. Requires `pip install umap-learn`; skipped with
    a warning if it isn't installed."""
    try:
        import umap
    except ImportError:
        print("[warn] umap-learn not installed — skipping UMAP plot "
              "(pip install umap-learn)")
        return
    mus, labels = _collect_latents(vae, loader, max_points)
    emb = umap.UMAP(n_components=2, random_state=SEED).fit_transform(mus)
    _scatter_latent(emb, labels, f"{label} — UMAP", save_path)


# ═════════════════════════════════════════════════════════════════════════════
# 6.  MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def main():

    print("\n" + "═"*58)
    print("  Stage 0 : Data & Feature Engineering")
    print("═"*58)

    df, idx_train, idx_val, idx_test = build_splits(CSV_PATH)
    plot_feature_distributions(df, os.path.join(OUT_DIR, "feature_distributions.png"))

    weights   = torch.tensor([1.0, 1.0, 2.0], device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═"*58)
    print("  Stage 1 : CNN  (image → class)")
    print("═"*58)

    train_dl1, val_dl1, test_dl1 = build_dataloaders(
        df, idx_train, idx_val, idx_test, feat_cols=None, scaler_name=None)

    model1 = ImageCNN(n_classes=3).to(DEVICE)
    for p in model1.img_enc.net.parameters():
        p.requires_grad = False   # frozen for the first `freeze_epochs` epochs

    opt1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model1.parameters()),
        lr=1e-3, weight_decay=1e-4)
    sched1 = optim.lr_scheduler.OneCycleLR(
        opt1, max_lr=1e-3, epochs=EPOCHS, steps_per_epoch=len(train_dl1))

    debug_break("before Stage 1 (image-only) training")
    hist1 = run_training(model1, train_dl1, val_dl1, EPOCHS, criterion, opt1, sched1,
                          label="Stage1-image", use_tab=False, freeze_epochs=5)

    plot_training_curves(hist1, "Stage 1 — CNN (image)",
                         os.path.join(OUT_DIR, "stage1_curves.png"))
    preds1, labels1 = predict(model1, test_dl1, use_tab=False)
    metrics1 = report_metrics(preds1, labels1, "Stage 1 — CNN (image)")
    plot_confusion_matrix(preds1, labels1, "Stage 1 — CNN (image)",
                          os.path.join(OUT_DIR, "stage1_confusion.png"))
    torch.save(model1.state_dict(), os.path.join(OUT_DIR, "stage1_image_cnn.pt"))
    debug_break("after Stage 1 evaluation")

    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═"*58)
    print("  Stage 2 : CNN + tabular1  (image + 7-d photometry → class)")
    print("═"*58)

    train_dl2, val_dl2, test_dl2 = build_dataloaders(
        df, idx_train, idx_val, idx_test, FEAT_COLS1, "scaler_tab1.pkl")

    model2 = MultimodalCNN(tab_dim=TAB_DIM1, n_classes=3).to(DEVICE)
    for p in model2.img_enc.net.parameters():
        p.requires_grad = False

    opt2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model2.parameters()),
        lr=1e-3, weight_decay=1e-4)
    sched2 = optim.lr_scheduler.OneCycleLR(
        opt2, max_lr=1e-3, epochs=EPOCHS, steps_per_epoch=len(train_dl2))

    debug_break("before Stage 2 (image+tab1) training")
    hist2 = run_training(model2, train_dl2, val_dl2, EPOCHS, criterion, opt2, sched2,
                          label="Stage2-image", use_tab=True, freeze_epochs=5)

    plot_training_curves(hist2, "Stage 2 — CNN + Tab1",
                         os.path.join(OUT_DIR, "stage2_curves.png"))
    preds2, labels2 = predict(model2, test_dl2, use_tab=True)
    metrics2 = report_metrics(preds2, labels2, "Stage 2 — CNN + Tab1")
    plot_confusion_matrix(preds2, labels2, "Stage 2 — CNN + Tab1",
                          os.path.join(OUT_DIR, "stage2_confusion.png"))
    torch.save(model2.state_dict(), os.path.join(OUT_DIR, "stage2_cnn_tab1.pt"))
    debug_break("after Stage 2 evaluation")

    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═"*58)
    print("  Stage 3 : CNN + tabular2  (image + 9-d photometry → class)")
    print("═"*58)

    train_dl3, val_dl3, test_dl3 = build_dataloaders(
        df, idx_train, idx_val, idx_test, FEAT_COLS2, "scaler_tab2.pkl")

    model3 = MultimodalCNN(tab_dim=TAB_DIM2, n_classes=3).to(DEVICE)
    for p in model3.img_enc.net.parameters():
        p.requires_grad = False

    opt3 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model3.parameters()),
        lr=1e-3, weight_decay=1e-4)
    sched3 = optim.lr_scheduler.OneCycleLR(
        opt3, max_lr=1e-3, epochs=EPOCHS, steps_per_epoch=len(train_dl3))

    debug_break("before Stage 3 (image+tab2) training")
    hist3 = run_training(model3, train_dl3, val_dl3, EPOCHS, criterion, opt3, sched3,
                          label="Stage3-image", use_tab=True, freeze_epochs=5)

    plot_training_curves(hist3, "Stage 3 — CNN + Tab2",
                         os.path.join(OUT_DIR, "stage3_curves.png"))
    preds3, labels3 = predict(model3, test_dl3, use_tab=True)
    metrics3 = report_metrics(preds3, labels3, "Stage 3 — CNN + Tab2")
    plot_confusion_matrix(preds3, labels3, "Stage 3 — CNN + Tab2",
                          os.path.join(OUT_DIR, "stage3_confusion.png"))
    torch.save(model3.state_dict(), os.path.join(OUT_DIR, "stage3_cnn_tab2.pt"))
    debug_break("after Stage 3 evaluation")

    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═"*58)
    print("  Stage 4a : VAE Pretraining  (image → image, unsupervised)")
    print("═"*58)

    train_dl4, val_dl4, test_dl4 = build_dataloaders(
        df, idx_train, idx_val, idx_test, feat_cols=None, scaler_name=None,
        normalize="unit")   # decoder ends in Sigmoid → targets must be in [0,1]

    vae = VAE(latent_dim=VAE_LATENT_DIM).to(DEVICE)
    vae_opt   = torch.optim.Adam(vae.parameters(), lr=1e-3, weight_decay=1e-4)
    vae_sched = optim.lr_scheduler.ReduceLROnPlateau(vae_opt, patience=3, factor=0.5)

    debug_break("before VAE pretraining")
    vae_hist = run_training_vae(vae, train_dl4, val_dl4, EPOCHS_VAE,
                                 vae_opt, vae_sched, label="Stage4-vae-pretrain")

    plot_vae_training_curves(vae_hist, "VAE Pretraining",
                             os.path.join(OUT_DIR, "vae_pretrain_curves.png"))
    plot_vae_reconstructions(vae, test_dl4,
                             os.path.join(OUT_DIR, "vae_reconstructions.png"))
    plot_latent_tsne(vae, test_dl4, "VAE Latent Space",
                     os.path.join(OUT_DIR, "vae_latent_tsne.png"))
    plot_latent_umap(vae, test_dl4, "VAE Latent Space",
                     os.path.join(OUT_DIR, "vae_latent_umap.png"))
    torch.save(vae.state_dict(), os.path.join(OUT_DIR, "vae_pretrained.pt"))
    debug_break("after VAE pretraining")

    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═"*58)
    print("  Stage 4b : VAE Classifier  (frozen latent → class)")
    print("═"*58)

    # Image-only dataloader
    train_dl4_img, val_dl4_img, test_dl4_img = build_dataloaders(
        df,
        idx_train,
        idx_val,
        idx_test,
        feat_cols=None,
        scaler_name=None,
        normalize="unit"
    )

    vae_clf_img = VAEClassifier(
        vae,
        latent_dim=VAE_LATENT_DIM,
        n_classes=3
    ).to(DEVICE)


    # Freeze VAE encoder
    for p in vae_clf_img.encoder.parameters():
        p.requires_grad = False


    clf_opt_img = torch.optim.Adam(
        filter(lambda p: p.requires_grad, vae_clf_img.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )

    clf_sched_img = optim.lr_scheduler.OneCycleLR(
        clf_opt_img,
        max_lr=1e-3,
        epochs=EPOCHS,
        steps_per_epoch=len(train_dl4_img)
    )


    debug_break("before VAE image classifier training")

    hist4 = run_training(
        vae_clf_img,
        train_dl4_img,
        val_dl4_img,
        EPOCHS,
        criterion,
        clf_opt_img,
        clf_sched_img,
        label="Stage4-vae-classify",
        use_tab=False,
        freeze_epochs=EPOCHS
    )


    plot_training_curves(
        hist4,
        "Stage 4b — VAE Classifier",
        os.path.join(OUT_DIR, "stage4b_curves.png")
    )


    preds4, labels4 = predict(
        vae_clf_img,
        test_dl4_img,
        use_tab=False
    )

    metrics4 = report_metrics(
        preds4,
        labels4,
        "Stage 4b — VAE Classifier"
    )

    plot_confusion_matrix(
        preds4,
        labels4,
        "Stage 4b — VAE Classifier",
        os.path.join(OUT_DIR, "stage4_confusion.png")
    )

    torch.save(
        vae_clf_img.state_dict(),
        os.path.join(OUT_DIR, "stage4_vae_classifier.pt")
    )

    debug_break("after Stage 4b evaluation")



    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═"*58)
    print("  Stage 4c : VAE + Tab2  (latent + photometry → class)")
    print("═"*58)


    # Image + tabular dataloader
    train_dl4_tab, val_dl4_tab, test_dl4_tab = build_dataloaders(
        df,
        idx_train,
        idx_val,
        idx_test,
        FEAT_COLS2,
        "scaler_tab2.pkl",
        normalize="unit"
    )


    vae_clf_tab = MultimodalVAE(
        vae,
        tab_dim=TAB_DIM2,
        latent_dim=VAE_LATENT_DIM,
        n_classes=3
    ).to(DEVICE)


    # Freeze VAE encoder
    for p in vae_clf_tab.encoder.parameters():
        p.requires_grad = False


    clf_opt_tab = torch.optim.Adam(
        filter(lambda p: p.requires_grad, vae_clf_tab.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )


    clf_sched_tab = optim.lr_scheduler.OneCycleLR(
        clf_opt_tab,
        max_lr=1e-3,
        epochs=EPOCHS,
        steps_per_epoch=len(train_dl4_tab)
    )


    debug_break("before VAE + tab classifier training")


    hist5 = run_training(
        vae_clf_tab,
        train_dl4_tab,
        val_dl4_tab,
        EPOCHS,
        criterion,
        clf_opt_tab,
        clf_sched_tab,
        label="Stage4-vae-tab2-classify",
        use_tab=True,
        freeze_epochs=EPOCHS
    )


    plot_training_curves(
        hist5,
        "Stage 4c — VAE + Tab2",
        os.path.join(OUT_DIR, "stage4c_curves.png")
    )


    preds5, labels5 = predict(
        vae_clf_tab,
        test_dl4_tab,
        use_tab=True
    )


    metrics5 = report_metrics(
        preds5,
        labels5,
        "Stage 4c — VAE + Tab2"
    )


    plot_confusion_matrix(
        preds5,
        labels5,
        "Stage 4c — VAE + Tab2",
        os.path.join(OUT_DIR, "stage4c_confusion.png")
    )


    torch.save(
        vae_clf_tab.state_dict(),
        os.path.join(OUT_DIR, "stage4c_vae_tab2.pt")
    )

    debug_break("after Stage 4c evaluation")

    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═"*58)
    print("  Stage 5 : Metrics")
    print("═"*58)

    plot_comparison([metrics1, metrics2, metrics3, metrics4],
                    os.path.join(OUT_DIR, "comparison.png"))

    print("\n[summary]")
    for m in [metrics1, metrics2, metrics3, metrics4]:
        print(f"  {m['label']:<30}  acc={m['accuracy']:.4f}  f1={m['macro_f1']:.4f}")

    print(f"\n[done] outputs → {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()