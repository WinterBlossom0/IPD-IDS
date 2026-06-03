#!/usr/bin/env python
# coding: utf-8

# # CIC neural open-world classifier
# 
# This notebook trains one neural model for both in-distribution multiclass classification and final attack-vs-benign OOD evaluation.
# 
# The final OOD CSV is intentionally loaded only inside the final evaluation function after model training, scorer fitting, and threshold selection are finished.
# 

# In[1]:


import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

RANDOM_STATE = 42
DETERMINISTIC_RUN = True

VARIANT_CONFIGS = {
    "primary": {
        "suffix": "_primary",
        "train": "df_train_primary.csv",
        "final_ood": "df_test_primary.csv",
    },
    "no_time_adv": {
        "suffix": "_no_time_adv",
        "train": "df_train_no_time_adv.csv",
        "final_ood": "df_test_no_time_adv.csv",
    },
    "no_student": {
        "suffix": "_no_student",
        "train": "df_train_no_student.csv",
        "final_ood": "df_test_no_student.csv",
    },
}

ACTIVE_VARIANT_NAME = "primary"
DATA_SUFFIX = VARIANT_CONFIGS[ACTIVE_VARIANT_NAME]["suffix"]
TRAIN_FILENAME = VARIANT_CONFIGS[ACTIVE_VARIANT_NAME]["train"]
FINAL_OOD_FILENAME = VARIANT_CONFIGS[ACTIVE_VARIANT_NAME]["final_ood"]

TRAIN_RATIO = 4
VAL_RATIO = 1
INDIST_TEST_RATIO = 1
N_ROW_ORDER_DOMAINS = 6

EPOCHS = 50
BATCH_SIZE = 65536
PRED_BATCH_SIZE = 524288        # doubled from 262144 — fewer GPU round-trips at inference
LR = 3.0e-4
WEIGHT_DECAY = 2.0e-4
GRAD_CLIP = 4.0

MODEL_WIDTH = 224
MODEL_DEPTH = 5
EMBED_DIM = 112
MODEL_DROPOUT = 0.14

# Loss weights
BINARY_LOSS_WEIGHT = 1.60
ATTACK_SUBTYPE_LOSS_WEIGHT = 1.10
PROTOTYPE_LOSS_WEIGHT = 0.22
BENIGN_COMPACT_WEIGHT = 0.020
ATTACK_REPEL_WEIGHT = 0.025
DOMAIN_ADV_WEIGHT = 0.075
DOMAIN_GRL_LAMBDA = 0.55
DOMAIN_ADV_WARMUP_EPOCHS = 8

BENIGN_COS_MARGIN = 0.28
LABEL_SMOOTHING = 0.012
PROTOTYPE_TEMPERATURE = 0.14

QUANTILE_N_QUANTILES = 2048
QUANTILE_SUBSAMPLE = 200000

BINARY_SCORE_NAME = "binary_head_domain_guarded"

# VAE recon-loss fusion for OOD scoring (train-only, zero leakage)
RECON_OOD_WEIGHT = 0.35

# Threshold selection floors (relaxed for better OOD recall)
VAL_ATTACK_RECALL_FLOOR = 0.970
VAL_BENIGN_SPEC_FLOOR = 0.985
VAL_WORST_DOMAIN_BENIGN_FLOOR = 0.960
EPS = 1e-8

# ── Speed flags ──────────────────────────────────────────────────────────
# Mixed-precision: ~2x faster on Ampere/Turing GPUs, negligible accuracy loss
USE_AMP = True
# torch.compile: ~30% faster after warm-up (PyTorch >= 2.0)
USE_COMPILE = False      # torch.compile needs Triton — unavailable on Windows


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if DETERMINISTIC_RUN:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = (RANDOM_STATE + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


seed_everything()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
AMP_ENABLED = USE_AMP and DEVICE.type == "cuda"
print(f"device={DEVICE} | train_file={TRAIN_FILENAME} | final_ood_file={FINAL_OOD_FILENAME}")
print(f"AMP={AMP_ENABLED} | compile={USE_COMPILE} | pred_batch={PRED_BATCH_SIZE}")


# ## Data loading
# 
# Only the training CSV is loaded here. The final OOD CSV is not read in this section.
# 

# In[2]:


def find_data_file(filename: str) -> Path:
    direct = Path(filename)
    if direct.exists():
        return direct
    matches = sorted(Path(".").rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename!r} under {Path.cwd()}")
    return matches[0]


def load_train_variant(variant_name: str) -> pd.DataFrame:
    if variant_name not in VARIANT_CONFIGS:
        raise KeyError(f"Unknown variant {variant_name!r}; expected one of {sorted(VARIANT_CONFIGS)}")
    train_path = find_data_file(VARIANT_CONFIGS[variant_name]["train"])
    frame = pd.read_csv(train_path)
    if "label" not in frame.columns:
        raise ValueError(f"Expected a 'label' column in {train_path}.")
    label_counts = frame["label"].value_counts().sort_index()
    print(f"loaded train-only variant={variant_name} rows={len(frame):,} from {train_path}")
    print(label_counts.to_string())
    print("Final OOD data is still unopened; it is only loaded inside run_final_ood_evaluation().")
    return frame


# In[3]:


def stratified_three_way_split(
    labels: np.ndarray,
    train_ratio: int,
    val_ratio: int,
    test_ratio: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[np.ndarray] = []
    val_idx: list[np.ndarray] = []
    test_idx: list[np.ndarray] = []
    total_ratio = train_ratio + val_ratio + test_ratio

    for label in np.sort(np.unique(labels)):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(n * test_ratio / total_ratio)))
        n_val = max(1, int(round(n * val_ratio / total_ratio)))
        if n_test + n_val >= n:
            n_test = max(1, n // 6)
            n_val = max(1, n // 6)
        test_idx.append(idx[:n_test])
        val_idx.append(idx[n_test : n_test + n_val])
        train_idx.append(idx[n_test + n_val :])

    return (
        np.sort(np.concatenate(train_idx)),
        np.sort(np.concatenate(val_idx)),
        np.sort(np.concatenate(test_idx)),
    )


def build_label_mapping(labels: pd.Series) -> tuple[dict[int, int], dict[int, int], list[int]]:
    original_labels = sorted(int(v) for v in labels.unique())
    if 0 not in original_labels:
        raise ValueError("Benign label 0 is required for the binary OOD task.")
    attack_labels = [v for v in original_labels if v != 0]
    original_to_internal = {0: 0}
    for internal_id, original_id in enumerate(attack_labels, start=1):
        original_to_internal[original_id] = internal_id
    internal_to_original = {v: k for k, v in original_to_internal.items()}
    return original_to_internal, internal_to_original, attack_labels


def map_labels(series: pd.Series, original_to_internal: dict[int, int]) -> np.ndarray:
    mapped = series.map(original_to_internal)
    if mapped.isna().any():
        missing = sorted(int(v) for v in series[mapped.isna()].unique())
        raise ValueError(f"Training split contains labels missing from mapping: {missing}")
    return np.ascontiguousarray(mapped.to_numpy(dtype=np.int64, copy=True))


def take_split(frame: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    out = frame.iloc[indices].copy()
    out["_source_row"] = indices.astype(np.int64)
    return out.reset_index(drop=True)


def row_order_domains(frame: pd.DataFrame) -> np.ndarray:
    if "_source_row" not in frame.columns:
        raise ValueError("Expected '_source_row' in train-derived frames.")
    raw = frame["_source_row"].to_numpy(dtype=np.float64)
    domain = np.floor(raw * N_ROW_ORDER_DOMAINS / max(1, TOTAL_SOURCE_ROWS)).astype(np.int64)
    return np.ascontiguousarray(np.clip(domain, 0, N_ROW_ORDER_DOMAINS - 1))


def activate_variant(variant_name: str, force_reload: bool = False) -> None:
    global ACTIVE_VARIANT_NAME, DATA_SUFFIX, TRAIN_FILENAME, FINAL_OOD_FILENAME
    global df_train, df_val, df_indist_test, original_to_internal, internal_to_original
    global known_attack_labels, num_classes, num_attack_classes, TOTAL_SOURCE_ROWS

    if (
        not force_reload
        and globals().get("ACTIVE_VARIANT_NAME") == variant_name
        and "df_train" in globals()
        and "df_val" in globals()
        and "df_indist_test" in globals()
    ):
        print(f"variant already active: {variant_name}")
        return

    config = VARIANT_CONFIGS[variant_name]
    ACTIVE_VARIANT_NAME = variant_name
    DATA_SUFFIX = config["suffix"]
    TRAIN_FILENAME = config["train"]
    FINAL_OOD_FILENAME = config["final_ood"]

    df_all_variant = load_train_variant(variant_name)
    original_to_internal, internal_to_original, known_attack_labels = build_label_mapping(df_all_variant["label"])
    num_classes = len(original_to_internal)
    num_attack_classes = max(1, num_classes - 1)
    TOTAL_SOURCE_ROWS = int(len(df_all_variant))
    train_idx, val_idx, indist_test_idx = stratified_three_way_split(
        df_all_variant["label"].to_numpy(),
        TRAIN_RATIO,
        VAL_RATIO,
        INDIST_TEST_RATIO,
        RANDOM_STATE,
    )

    df_train = take_split(df_all_variant, train_idx)
    df_val = take_split(df_all_variant, val_idx)
    df_indist_test = take_split(df_all_variant, indist_test_idx)
    del df_all_variant

    print(f"known attack labels={known_attack_labels}")
    print(f"internal class map={original_to_internal}")
    print(f"split rows | train={len(df_train):,} val={len(df_val):,} in_dist_test={len(df_indist_test):,}")
    print("train row-order domains:")
    print(pd.Series(row_order_domains(df_train)).value_counts().sort_index().to_string())


activate_variant("primary")


# ## Feature sets

# In[4]:


def refresh_feature_sets() -> dict[str, list[str]]:
    global LATENT_COLUMNS, LOSS_COLUMNS, ALL_NUMERIC_COLUMNS, FEATURE_SETS, PRIMARY_FEATURE_SET
    LATENT_COLUMNS = [c for c in df_train.columns if c.startswith("latent_")]
    LOSS_COLUMNS = [c for c in ["student_loss", "kld_loss", "mse_loss"] if c in df_train.columns]
    ALL_NUMERIC_COLUMNS = [
        c
        for c in df_train.columns
        if c not in {"label", "_source_row"} and pd.api.types.is_numeric_dtype(df_train[c])
    ]
    FEATURE_SETS = {
        "compressed + losses": LATENT_COLUMNS + LOSS_COLUMNS,
        "compressed only": LATENT_COLUMNS,
        "losses only": LOSS_COLUMNS,
        "all compressed features": ALL_NUMERIC_COLUMNS,
    }
    FEATURE_SETS = {name: cols for name, cols in FEATURE_SETS.items() if cols}
    PRIMARY_FEATURE_SET = "compressed + losses"
    return FEATURE_SETS


refresh_feature_sets()
print(json.dumps(FEATURE_SETS, indent=2))


# ## Metrics

# In[5]:


def safe_divide(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def hierarchical_f1(y_true_internal: np.ndarray, y_pred_internal: np.ndarray) -> float:
    y_true_binary = (y_true_internal != 0).astype(np.int64)
    y_pred_binary = (y_pred_internal != 0).astype(np.int64)
    binary_f1 = f1_score(y_true_binary, y_pred_binary, average="macro", zero_division=0)
    subtype_mask = y_true_internal != 0
    if subtype_mask.any():
        subtype_f1 = f1_score(
            y_true_internal[subtype_mask],
            y_pred_internal[subtype_mask],
            average="macro",
            zero_division=0,
        )
    else:
        subtype_f1 = binary_f1
    return float(0.5 * binary_f1 + 0.5 * subtype_f1)


def binary_metrics(y_true_binary: np.ndarray, y_pred_binary: np.ndarray, score: np.ndarray | None = None) -> dict[str, float]:
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision, recall, f1_values, _ = precision_recall_fscore_support(
        y_true_binary,
        y_pred_binary,
        labels=[0, 1],
        zero_division=0,
    )
    metrics = {
        "bal_acc": float(balanced_accuracy_score(y_true_binary, y_pred_binary)),
        "macro_f1": float(f1_score(y_true_binary, y_pred_binary, average="macro", zero_division=0)),
        "benign_precision": float(precision[0]),
        "attack_precision": float(precision[1]),
        "benign_recall": float(recall[0]),
        "attack_recall": float(recall[1]),
        "benign_f1": float(f1_values[0]),
        "attack_f1": float(f1_values[1]),
        "benign_specificity": safe_divide(tn, tn + fp),
        "attack_rate": float(np.mean(y_pred_binary)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if score is not None:
        metrics["score_mean"] = float(np.mean(score))
        metrics["score_p95"] = float(np.quantile(score, 0.95))
    return metrics


def multiclass_metrics(y_true_internal: np.ndarray, y_pred_internal: np.ndarray) -> dict[str, float]:
    labels = list(range(num_classes))
    subtype_mask = y_true_internal != 0
    if subtype_mask.any():
        subtype_f1 = f1_score(
            y_true_internal[subtype_mask],
            y_pred_internal[subtype_mask],
            labels=labels[1:],
            average="macro",
            zero_division=0,
        )
    else:
        subtype_f1 = 0.0
    return {
        "bal_acc": float(balanced_accuracy_score(y_true_internal, y_pred_internal)),
        "macro_f1": float(f1_score(y_true_internal, y_pred_internal, labels=labels, average="macro", zero_division=0)),
        "hier_f1": hierarchical_f1(y_true_internal, y_pred_internal),
        "subtype_f1": float(subtype_f1),
    }


def ood_unseen_recall(
    original_labels: np.ndarray,
    y_pred_binary: np.ndarray,
    known_labels: list[int] | None = None,
) -> float:
    reference_labels = known_attack_labels if known_labels is None else known_labels
    unseen_mask = (original_labels != 0) & ~np.isin(original_labels, np.array(reference_labels, dtype=np.int64))
    if not unseen_mask.any():
        return float("nan")
    return float(np.mean(y_pred_binary[unseen_mask] == 1))


# ## Open-world model and loss

# In[6]:


def make_class_weights(y: np.ndarray, n_classes: int, power: float = 0.5) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = (counts[nonzero].sum() / (counts[nonzero] + EPS)) ** power
    weights[nonzero] = weights[nonzero] / weights[nonzero].mean()
    weights[~nonzero] = 0.0
    return torch.tensor(weights, dtype=torch.float32)


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


def grad_reverse(x: torch.Tensor, scale: float) -> torch.Tensor:
    return GradientReverse.apply(x, scale)


class GatedResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.gate = nn.Linear(width, width * 2)
        self.out = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, gates = self.gate(self.norm(x)).chunk(2, dim=-1)
        update = values * F.silu(gates)
        return x + self.dropout(self.out(update))


class BinaryFirstTabularNet(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, n_attack_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, MODEL_WIDTH),
            nn.LayerNorm(MODEL_WIDTH),
            nn.SiLU(),
            nn.Dropout(MODEL_DROPOUT),
        )
        self.blocks = nn.Sequential(*[GatedResidualBlock(MODEL_WIDTH, MODEL_DROPOUT) for _ in range(MODEL_DEPTH)])
        self.embed = nn.Sequential(
            nn.LayerNorm(MODEL_WIDTH),
            nn.Linear(MODEL_WIDTH, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM),
        )
        self.binary_head = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM // 2),
            nn.SiLU(),
            nn.Dropout(MODEL_DROPOUT),
            nn.Linear(EMBED_DIM // 2, 2),
        )
        self.attack_subtype_head = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.SiLU(),
            nn.Dropout(MODEL_DROPOUT),
            nn.Linear(EMBED_DIM, n_attack_classes),
        )
        self.domain_head = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM // 2),
            nn.SiLU(),
            nn.Dropout(MODEL_DROPOUT),
            nn.Linear(EMBED_DIM // 2, N_ROW_ORDER_DOMAINS),
        )
        self.prototypes = nn.Parameter(torch.randn(n_classes, EMBED_DIM) * 0.05)

    def forward(self, x: torch.Tensor, grl_lambda: float = 0.0) -> dict[str, torch.Tensor]:
        hidden = self.blocks(self.stem(x))
        embedding = self.embed(hidden)
        reversed_embedding = grad_reverse(embedding, grl_lambda) if grl_lambda else embedding.detach()
        return {
            "embedding": embedding,
            "binary_logits": self.binary_head(embedding),
            "attack_subtype_logits": self.attack_subtype_head(embedding),
            "prototype_logits": self.prototype_logits(embedding),
            "domain_logits": self.domain_head(reversed_embedding),
        }

    def prototype_logits(self, embedding: torch.Tensor) -> torch.Tensor:
        z = F.normalize(embedding, dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        return z @ proto.T / PROTOTYPE_TEMPERATURE


OpenWorldTabularNet = BinaryFirstTabularNet


class BinaryFirstLoss(nn.Module):
    def __init__(self, y_train_internal: np.ndarray) -> None:
        super().__init__()
        y_train_binary = (y_train_internal != 0).astype(np.int64)
        attack_y = y_train_internal[y_train_internal != 0] - 1
        self.register_buffer("binary_weights", make_class_weights(y_train_binary, 2))
        self.register_buffer("attack_class_weights", make_class_weights(attack_y, num_attack_classes))
        self.register_buffer("prototype_class_weights", make_class_weights(y_train_internal, num_classes))

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        y_internal: torch.Tensor,
        prototypes: torch.Tensor,
        domain_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        y_binary = (y_internal != 0).long()
        attack_mask = y_internal != 0
        benign_mask = y_internal == 0

        binary_loss = F.cross_entropy(
            outputs["binary_logits"],
            y_binary,
            weight=self.binary_weights,
            label_smoothing=LABEL_SMOOTHING,
        )
        if attack_mask.any():
            attack_target = y_internal[attack_mask] - 1
            attack_subtype_loss = F.cross_entropy(
                outputs["attack_subtype_logits"][attack_mask],
                attack_target,
                weight=self.attack_class_weights,
                label_smoothing=LABEL_SMOOTHING,
            )
        else:
            attack_subtype_loss = outputs["binary_logits"].new_tensor(0.0)

        prototype_loss = F.cross_entropy(
            outputs["prototype_logits"],
            y_internal,
            weight=self.prototype_class_weights,
            label_smoothing=LABEL_SMOOTHING,
        )

        z = F.normalize(outputs["embedding"], dim=-1)
        proto = F.normalize(prototypes, dim=-1)
        benign_sim = (z * proto[0]).sum(dim=-1)
        benign_compact = (1.0 - benign_sim[benign_mask]).mean() if benign_mask.any() else z.new_tensor(0.0)
        attack_repel = F.relu(benign_sim[attack_mask] - BENIGN_COS_MARGIN).mean() if attack_mask.any() else z.new_tensor(0.0)

        if domain_id is not None and benign_mask.any():
            domain_loss = F.cross_entropy(outputs["domain_logits"][benign_mask], domain_id[benign_mask])
        else:
            domain_loss = z.new_tensor(0.0)

        total = (
            BINARY_LOSS_WEIGHT * binary_loss
            + ATTACK_SUBTYPE_LOSS_WEIGHT * attack_subtype_loss
            + PROTOTYPE_LOSS_WEIGHT * prototype_loss
            + BENIGN_COMPACT_WEIGHT * benign_compact
            + ATTACK_REPEL_WEIGHT * attack_repel
            + DOMAIN_ADV_WEIGHT * domain_loss
        )
        parts = {
            "binary": float(binary_loss.detach().cpu()),
            "attack_subtype": float(attack_subtype_loss.detach().cpu()),
            "prototype": float(prototype_loss.detach().cpu()),
            "benign_compact": float(benign_compact.detach().cpu()),
            "attack_repel": float(attack_repel.detach().cpu()),
            "domain_adv": float(domain_loss.detach().cpu()),
        }
        return total, parts


OpenWorldLoss = BinaryFirstLoss


# ## Preprocessing and benign evidence scorer

# In[7]:


def clean_float32(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    med = np.nanmedian(values, axis=0).astype(np.float32)
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = np.take(med, np.where(bad)[1])
    return values.astype(np.float32, copy=False)


def fit_feature_scaler(x_train_raw: np.ndarray) -> QuantileTransformer:
    n_quantiles = min(QUANTILE_N_QUANTILES, max(10, len(x_train_raw)))
    subsample = min(QUANTILE_SUBSAMPLE, len(x_train_raw))
    scaler = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="normal",
        subsample=subsample,
        random_state=RANDOM_STATE,
        copy=True,
    )
    scaler.fit(x_train_raw)
    return scaler


def transform_features(scaler: QuantileTransformer, x_raw: np.ndarray) -> np.ndarray:
    x = np.asarray(scaler.transform(x_raw), dtype=np.float32).copy()
    np.clip(x, -5.0, 5.0, out=x)
    return np.ascontiguousarray(x)


@torch.no_grad()
def predict_arrays(
    model: OpenWorldTabularNet,
    x_scaled: np.ndarray,
    batch_size: int = PRED_BATCH_SIZE,
    return_embedding: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    model.eval()
    binary_chunks: list[np.ndarray] = []
    attack_subtype_chunks: list[np.ndarray] = []
    embedding_chunks: list[np.ndarray] = []
    for start in range(0, len(x_scaled), batch_size):
        xb = torch.from_numpy(x_scaled[start : start + batch_size]).to(DEVICE)
        outputs = model(xb, grl_lambda=0.0)
        binary_chunks.append(torch.softmax(outputs["binary_logits"], dim=1)[:, 1].detach().cpu().numpy().astype(np.float32))
        attack_subtype_chunks.append(torch.softmax(outputs["attack_subtype_logits"], dim=1).detach().cpu().numpy().astype(np.float32))
        if return_embedding:
            embedding_chunks.append(outputs["embedding"].detach().cpu().numpy().astype(np.float32))
    binary_prob = np.concatenate(binary_chunks)
    attack_subtype_prob = np.concatenate(attack_subtype_chunks)
    embeddings = np.concatenate(embedding_chunks) if return_embedding else None
    return binary_prob, attack_subtype_prob, embeddings


# ---------------------------------------------------------------------------
# Recon-loss OOD fusion helpers
# ---------------------------------------------------------------------------
# The VAE reconstruction losses (mse_loss, kld_loss) are already present as
# pre-computed feature columns.  OOD flows typically have MUCH higher recon
# loss than in-distribution traffic.  We exploit this by fitting a simple
# percentile scaler from TRAINING data only and fusing the resulting anomaly
# score into the final binary attack probability at inference time.
# The OOD test CSV is NEVER read inside these helpers.

def _extract_recon_cols(
    x_raw: np.ndarray, feature_columns: list[str]
) -> np.ndarray | None:
    wanted = ["mse_loss", "kld_loss"]
    idxs = [feature_columns.index(c) for c in wanted if c in feature_columns]
    if not idxs:
        return None
    return x_raw[:, idxs]  # (N, n_recon_cols)


def fit_recon_scorer(recon_train: np.ndarray) -> dict:
    p5 = np.percentile(recon_train, 5, axis=0).astype(np.float32)
    p95 = np.percentile(recon_train, 95, axis=0).astype(np.float32)
    return {"p5": p5, "p95": p95}


def score_recon(rs: dict, recon_vals: np.ndarray) -> np.ndarray:
    span = np.maximum(rs["p95"] - rs["p5"], EPS)
    scaled = (recon_vals - rs["p5"]) / span   # 0 at p5, 1 at p95
    raw = np.clip(scaled.mean(axis=1), 0.0, None)  # (N,)
    return np.clip(1.0 - np.exp(-raw), 0.0, 1.0).astype(np.float32)


def binary_attack_score(
    binary_prob: np.ndarray,
    recon_scorer: dict | None = None,
    recon_vals: np.ndarray | None = None,
) -> np.ndarray:
    base = np.clip(binary_prob.astype(np.float32), 0.0, 1.0)
    if recon_scorer is None or recon_vals is None or RECON_OOD_WEIGHT <= 0.0:
        return base
    recon_score = score_recon(recon_scorer, recon_vals)
    w = float(RECON_OOD_WEIGHT)
    return np.clip((1.0 - w) * base + w * recon_score, 0.0, 1.0).astype(np.float32)


def class_predictions(attack_subtype_prob: np.ndarray, attack_score: np.ndarray, threshold: float) -> np.ndarray:
    pred = np.zeros(len(attack_score), dtype=np.int64)
    attack_mask = attack_score >= threshold
    if attack_mask.any():
        pred[attack_mask] = 1 + np.argmax(attack_subtype_prob[attack_mask], axis=1)
    return pred


# ---------------------------------------------------------------------------
# GPU-resident batch sampler
# ---------------------------------------------------------------------------
# For small tabular datasets (e.g. 8 features × 2M rows ≈ 64 MB) the entire
# training set fits in VRAM. Uploading once and shuffling with torch.randperm
# on the GPU eliminates ALL CPU→GPU transfers during training, letting the
# GPU run at 100% utilisation instead of stalling for data.

class GPUBatchSampler:
    def __init__(
        self,
        *arrays: np.ndarray,
        batch_size: int,
        device: torch.device,
        seed: int = RANDOM_STATE,
    ) -> None:
        self.tensors = tuple(
            torch.as_tensor(arr, device=device) for arr in arrays
        )
        self.n = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.device = device
        self._gen = torch.Generator(device=device)
        self._gen.manual_seed(seed)
        mem_mb = sum(t.nbytes for t in self.tensors) / 1024**2
        print(f"  GPUBatchSampler: {self.n:,} rows | {mem_mb:.1f} MB on {device} | batch={batch_size}")

    def __iter__(self):
        perm = torch.randperm(self.n, device=self.device, generator=self._gen)
        for start in range(0, self.n, self.batch_size):
            idx = perm[start : start + self.batch_size]
            yield tuple(t[idx] for t in self.tensors)

    def __len__(self) -> int:
        import math
        return math.ceil(self.n / self.batch_size)


@torch.no_grad()
def predict_arrays_gpu(
    model: OpenWorldTabularNet,
    x_gpu: torch.Tensor,
    batch_size: int = PRED_BATCH_SIZE,
    return_embedding: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Fast path: data already on GPU — no H2D transfers during inference."""
    model.eval()
    binary_chunks: list[np.ndarray] = []
    attack_chunks: list[np.ndarray] = []
    emb_chunks: list[np.ndarray] = []
    for start in range(0, x_gpu.shape[0], batch_size):
        xb = x_gpu[start : start + batch_size]
        outputs = model(xb, grl_lambda=0.0)
        binary_chunks.append(torch.softmax(outputs["binary_logits"], dim=1)[:, 1].cpu().numpy().astype(np.float32))
        attack_chunks.append(torch.softmax(outputs["attack_subtype_logits"], dim=1).cpu().numpy().astype(np.float32))
        if return_embedding:
            emb_chunks.append(outputs["embedding"].cpu().numpy().astype(np.float32))
    binary_prob = np.concatenate(binary_chunks)
    attack_prob = np.concatenate(attack_chunks)
    embeddings = np.concatenate(emb_chunks) if return_embedding else None
    return binary_prob, attack_prob, embeddings


# ## Threshold policy

# In[8]:


def worst_domain_benign_recall(y_binary: np.ndarray, y_pred_binary: np.ndarray, domain_id: np.ndarray | None) -> float:
    if domain_id is None:
        return float(binary_metrics(y_binary, y_pred_binary)["benign_recall"])
    recalls: list[float] = []
    for domain in np.sort(np.unique(domain_id)):
        mask = (domain_id == domain) & (y_binary == 0)
        if mask.any():
            recalls.append(float(np.mean(y_pred_binary[mask] == 0)))
    return float(min(recalls)) if recalls else 0.0


def choose_open_world_threshold(
    score: np.ndarray,
    y_internal: np.ndarray,
    domain_id: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    y_binary = (y_internal != 0).astype(np.int64)
    if len(np.unique(y_binary)) != 2:
        raise ValueError("Validation split needs both benign and attack rows for threshold selection.")

    quantiles = np.linspace(0.001, 0.999, 600)
    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.quantile(score, quantiles),
                np.linspace(0.02, 0.98, 250),
            ]
        )
    )

    is_attack = (y_binary == 1)
    is_benign = (y_binary == 0)
    n_attack = np.sum(is_attack)
    n_benign = np.sum(is_benign)

    # Vectorized computation of tp, fp, fn, tn for all candidate thresholds
    score_col = score[:, np.newaxis]
    thresh_row = candidate_thresholds[np.newaxis, :]
    pred_matrix = (score_col >= thresh_row)

    tp = np.sum(pred_matrix[is_attack, :], axis=0)
    fp = np.sum(pred_matrix[is_benign, :], axis=0)

    fn = n_attack - tp
    tn = n_benign - fp

    attack_recall = tp / n_attack
    benign_specificity = tn / n_benign

    denom_bp = tn + fn
    benign_precision = np.zeros_like(tn, dtype=np.float32)
    nz_bp = denom_bp > 0
    benign_precision[nz_bp] = tn[nz_bp] / denom_bp[nz_bp]

    denom_ap = tp + fp
    attack_precision = np.zeros_like(tp, dtype=np.float32)
    nz_ap = denom_ap > 0
    attack_precision[nz_ap] = tp[nz_ap] / denom_ap[nz_ap]

    benign_f1 = np.zeros_like(tn, dtype=np.float32)
    denom_bf1 = benign_precision + benign_specificity
    nz_bf1 = denom_bf1 > 0
    benign_f1[nz_bf1] = 2.0 * benign_precision[nz_bf1] * benign_specificity[nz_bf1] / denom_bf1[nz_bf1]

    attack_f1 = np.zeros_like(tp, dtype=np.float32)
    denom_af1 = attack_precision + attack_recall
    nz_af1 = denom_af1 > 0
    attack_f1[nz_af1] = 2.0 * attack_precision[nz_af1] * attack_recall[nz_af1] / denom_af1[nz_af1]

    bal_acc = 0.5 * (attack_recall + benign_specificity)
    macro_f1 = 0.5 * (benign_f1 + attack_f1)

    if domain_id is None:
        worst_benign = benign_specificity
    else:
        unique_domains = np.sort(np.unique(domain_id))
        recalls_by_domain = []
        for domain in unique_domains:
            domain_mask = (domain_id == domain) & is_benign
            if domain_mask.any():
                n_benign_d = np.sum(domain_mask)
                tn_d = np.sum(~pred_matrix[domain_mask, :], axis=0)
                recalls_by_domain.append(tn_d / n_benign_d)
        if recalls_by_domain:
            worst_benign = np.minimum.reduce(recalls_by_domain)
        else:
            worst_benign = np.zeros(len(candidate_thresholds), dtype=np.float32)

    guarded_core = np.minimum(np.minimum(attack_recall, benign_specificity), worst_benign)
    min_spec_worst = np.minimum(benign_specificity, worst_benign)

    utility = (
        0.42 * bal_acc
        + 0.22 * macro_f1
        + 0.18 * attack_recall
        + 0.18 * min_spec_worst
        + 1e-4 * candidate_thresholds
    )

    any_key = utility + 0.10 * guarded_core + 0.05 * candidate_thresholds

    guarded_mask = (
        (attack_recall >= VAL_ATTACK_RECALL_FLOOR)
        & (benign_specificity >= VAL_BENIGN_SPEC_FLOOR)
        & (worst_benign >= VAL_WORST_DOMAIN_BENIGN_FLOOR)
    )

    guarded_key = utility + 0.25 * guarded_core + 0.12 * candidate_thresholds

    best_any_idx = np.argmax(any_key)

    if np.any(guarded_mask):
        masked_guarded_key = np.where(guarded_mask, guarded_key, -np.inf)
        best_idx = np.argmax(masked_guarded_key)
        source = "in_dist_val_domain_guarded"
    else:
        best_idx = best_any_idx
        source = "in_dist_val_domain_guarded_relaxed"

    threshold = float(candidate_thresholds[best_idx])

    metrics = {
        "bal_acc": float(bal_acc[best_idx]),
        "macro_f1": float(macro_f1[best_idx]),
        "benign_precision": float(benign_precision[best_idx]),
        "attack_precision": float(attack_precision[best_idx]),
        "benign_recall": float(benign_specificity[best_idx]),
        "attack_recall": float(attack_recall[best_idx]),
        "benign_f1": float(benign_f1[best_idx]),
        "attack_f1": float(attack_f1[best_idx]),
        "benign_specificity": float(benign_specificity[best_idx]),
        "worst_domain_benign_recall": float(worst_benign[best_idx]),
        "threshold": float(threshold),
        "attack_rate": float(np.mean(score >= threshold)),
        "tn": int(tn[best_idx]),
        "fp": int(fp[best_idx]),
        "fn": int(fn[best_idx]),
        "tp": int(tp[best_idx]),
        "score_mean": float(np.mean(score)),
        "score_p95": float(np.quantile(score, 0.95)),
        "threshold_source": source,
        "val_attack_recall_floor": VAL_ATTACK_RECALL_FLOOR,
        "val_benign_spec_floor": VAL_BENIGN_SPEC_FLOOR,
        "val_worst_domain_benign_floor": VAL_WORST_DOMAIN_BENIGN_FLOOR,
    }

    return float(threshold), metrics


# ## Training

# In[9]:


def train_one_model(feature_columns: list[str]) -> dict[str, object]:
    x_train_raw = clean_float32(df_train, feature_columns)
    x_val_raw   = clean_float32(df_val,   feature_columns)
    y_train = map_labels(df_train["label"], original_to_internal)
    y_val   = map_labels(df_val["label"],   original_to_internal)
    domain_train = row_order_domains(df_train)
    domain_val   = row_order_domains(df_val)

    scaler  = fit_feature_scaler(x_train_raw)
    x_train = transform_features(scaler, x_train_raw)
    x_val   = transform_features(scaler, x_val_raw)

    # Fit recon scorer from TRAINING data only — zero OOD leakage
    recon_train_raw = _extract_recon_cols(x_train_raw, feature_columns)
    if recon_train_raw is not None:
        recon_scorer  = fit_recon_scorer(recon_train_raw)
        recon_val_raw = _extract_recon_cols(x_val_raw, feature_columns)
        print(f"  recon_scorer | p5={recon_scorer['p5'].tolist()} | p95={recon_scorer['p95'].tolist()}")
    else:
        recon_scorer  = None
        recon_val_raw = None

    # ── GPU-resident datasets ────────────────────────────────────────────
    # Data is tiny (~64 MB for 8 features). Upload once, shuffle on-GPU.
    # Eliminates 1050+ CPU→GPU copies over 50 epochs → 100% GPU saturation.
    gpu_train = GPUBatchSampler(
        x_train, y_train, domain_train,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        seed=RANDOM_STATE,
    )
    # Pre-upload val tensor for zero-copy inference
    x_val_gpu = torch.as_tensor(x_val, device=DEVICE)
    print(f"  val GPU tensor: {x_val_gpu.shape} | {x_val_gpu.nbytes/1024**2:.1f} MB")

    model     = OpenWorldTabularNet(
        input_dim=x_train.shape[1],
        n_classes=num_classes,
        n_attack_classes=num_attack_classes,
    ).to(DEVICE)
    criterion = OpenWorldLoss(y_train).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.15)
    scaler_amp = torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)

    best_state: dict[str, torch.Tensor] | None = None
    best_score = -np.inf
    history: list[dict[str, float]] = []

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        model.train()
        loss_sum = 0.0
        n_seen   = 0
        grl_lambda = DOMAIN_GRL_LAMBDA * min(1.0, epoch / max(1, DOMAIN_ADV_WARMUP_EPOCHS))

        # ── GPU-resident training loop (zero H2D per batch) ──────────────
        for xb, yb, db in gpu_train:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=AMP_ENABLED):
                outputs = model(xb, grl_lambda=grl_lambda)
                loss, _ = criterion(outputs, yb, model.prototypes, db)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            loss_sum += float(loss.detach()) * len(xb)
            n_seen   += len(xb)
        scheduler.step()

        # ── Validation — GPU-tensor fast-path ────────────────────────────
        binary_prob, attack_subtype_prob, embeddings = predict_arrays_gpu(
            model, x_val_gpu, return_embedding=True
        )
        if embeddings is None:
            raise RuntimeError("Validation embeddings were not returned.")
        provisional_score = binary_attack_score(binary_prob, recon_scorer, recon_val_raw)
        threshold, bin_val = choose_open_world_threshold(provisional_score, y_val, domain_val)
        y_pred  = class_predictions(attack_subtype_prob, provisional_score, threshold)
        mc_val  = multiclass_metrics(y_val, y_pred)
        robust_binary = min(bin_val["attack_recall"], bin_val["benign_specificity"], bin_val["worst_domain_benign_recall"])
        val_score = 0.52 * mc_val["hier_f1"] + 0.48 * robust_binary

        row = {
            "epoch":              float(epoch),
            "train_loss":         loss_sum / max(1, n_seen),
            "val_score":          val_score,
            "hier_f1":            mc_val["hier_f1"],
            "binary_bal":         bin_val["bal_acc"],
            "attack_recall":      bin_val["attack_recall"],
            "benign_specificity": bin_val["benign_specificity"],
            "worst_domain_benign":bin_val["worst_domain_benign_recall"],
            "lr":                 optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        improved = val_score > best_score
        if improved:
            best_score = val_score
            best_state = copy.deepcopy(model.state_dict())
        marker = " *best*" if improved else ""
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | train={row['train_loss']:.4f} | "
            f"val_score={val_score:.4f} | hier_f1={mc_val['hier_f1']:.4f} | "
            f"attack_recall={bin_val['attack_recall']:.4f} | "
            f"benign_spec={bin_val['benign_specificity']:.4f} | "
            f"worst_dom={bin_val['worst_domain_benign_recall']:.4f} | "
            f"lr={row['lr']:.2e} | {time.time()-start_time:.1f}s{marker}"
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint.")
    model.load_state_dict(best_state)

    # Final val pass on best checkpoint
    binary_prob, attack_subtype_prob, _ = predict_arrays_gpu(model, x_val_gpu, return_embedding=False)
    val_score_arr      = binary_attack_score(binary_prob, recon_scorer, recon_val_raw)
    threshold, thr_m   = choose_open_world_threshold(val_score_arr, y_val, domain_val)
    y_pred             = class_predictions(attack_subtype_prob, val_score_arr, threshold)
    val_mc             = multiclass_metrics(y_val, y_pred)
    scorer             = {"score_name": BINARY_SCORE_NAME}

    # Free GPU-resident val tensor to reclaim VRAM before OOD eval
    del x_val_gpu
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    print(
        f"Best val_score={best_score:.4f} | threshold={threshold:.3f} | "
        f"source={thr_m['threshold_source']} | "
        f"val_attack_recall={thr_m['attack_recall']:.4f} | "
        f"val_benign_spec={thr_m['benign_specificity']:.4f} | "
        f"val_hier_f1={val_mc['hier_f1']:.4f}"
    )

    return {
        "model":               model,
        "scaler":              scaler,
        "scorer":              scorer,
        "threshold":           threshold,
        "threshold_metrics":   thr_m,
        "history":             pd.DataFrame(history),
        "feature_columns":     feature_columns,
        "val_multiclass":      val_mc,
        "variant_name":        ACTIVE_VARIANT_NAME,
        "train_filename":      TRAIN_FILENAME,
        "final_ood_filename":  FINAL_OOD_FILENAME,
        "data_suffix":         DATA_SUFFIX,
        "original_to_internal":dict(original_to_internal),
        "known_attack_labels": list(known_attack_labels),
        "recon_scorer":        recon_scorer,
    }


# ## Evaluation

# In[10]:


def evaluate_frame(
    bundle: dict[str, object],
    frame: pd.DataFrame,
    mode: str,
    original_label_for_unseen: bool = False,
) -> tuple[dict[str, float], dict[str, float] | None]:
    feature_columns = bundle["feature_columns"]  # type: ignore
    model           = bundle["model"]             # type: ignore
    scaler          = bundle["scaler"]            # type: ignore
    threshold       = float(bundle["threshold"])
    recon_scorer    = bundle.get("recon_scorer")

    x_raw    = clean_float32(frame, feature_columns)  # type: ignore
    x_scaled = transform_features(scaler, x_raw)      # type: ignore

    # Upload once to GPU → zero-copy inference
    x_gpu = torch.as_tensor(x_scaled, device=DEVICE)
    binary_prob, attack_subtype_prob, embeddings = predict_arrays_gpu(
        model, x_gpu, return_embedding=True  # type: ignore
    )
    del x_gpu

    if embeddings is None:
        raise RuntimeError("Embeddings were not returned during evaluation.")

    recon_vals = _extract_recon_cols(x_raw, feature_columns)  # type: ignore
    score      = binary_attack_score(binary_prob, recon_scorer, recon_vals)

    y_pred_internal = class_predictions(attack_subtype_prob, score, threshold)
    y_pred_binary   = (y_pred_internal != 0).astype(np.int64)

    y_true_binary = (frame["label"].to_numpy(dtype=np.int64) != 0).astype(np.int64)
    bin_result = binary_metrics(y_true_binary, y_pred_binary, score)
    bin_result["threshold"] = threshold
    bin_result["mode"]      = mode
    bin_result["rows"]      = int(len(frame))
    if original_label_for_unseen:
        known_labels = bundle.get("known_attack_labels", known_attack_labels)  # type: ignore
        bin_result["unseen_recall"] = ood_unseen_recall(
            frame["label"].to_numpy(dtype=np.int64), y_pred_binary, known_labels  # type: ignore
        )

    mc_result = None
    if mode == "in_dist":
        y_true_internal = map_labels(frame["label"], original_to_internal)
        mc_result = multiclass_metrics(y_true_internal, y_pred_internal)
        mc_result["mode"] = mode
        mc_result["rows"] = int(len(frame))
    return bin_result, mc_result


def run_final_ood_evaluation(bundle: dict[str, object]) -> dict[str, float]:
    # OOD file is loaded HERE and ONLY here — never used for training,
    # val, threshold tuning, or recon fitting.
    final_ood_filename = str(bundle.get("final_ood_filename", FINAL_OOD_FILENAME))
    final_ood_path     = find_data_file(final_ood_filename)
    df_final_ood       = pd.read_csv(final_ood_path)
    result, _ = evaluate_frame(bundle, df_final_ood, mode="final_ood", original_label_for_unseen=True)
    del df_final_ood
    return result


# ## Main Run: Full Compressed + Losses
# 
# This is the main experiment. Read this result first; the rest are ablations.
# 

# In[11]:


experiment_results: dict[str, dict[str, object]] = {}
ood_rows: list[dict[str, float]] = []
mc_rows: list[dict[str, float]] = []


def run_feature_set(variant_name: str, feature_set_name: str, experiment_name: str) -> dict[str, object]:
    activate_variant(variant_name)
    feature_sets = refresh_feature_sets()
    if feature_set_name not in feature_sets:
        raise KeyError(f"Feature set {feature_set_name!r} is unavailable for variant {variant_name!r}.")

    feature_columns = feature_sets[feature_set_name]
    print("=" * 80)
    print(f"MAIN/ABLATION RUN: {experiment_name}")
    print(f"variant={variant_name} | feature_set={feature_set_name} | columns={feature_columns}")
    print("Binary-first hierarchy: binary attack detector decides OOD; attack subtype head runs only after attack.")
    bundle = train_one_model(feature_columns)
    bundle["experiment_name"] = experiment_name
    bundle["feature_set_name"] = feature_set_name
    experiment_results[experiment_name] = bundle

    # Save logic for our main variant:
    if experiment_name == "primary__compressed_plus_losses":
        import torch, joblib
        torch.save(bundle["model"].state_dict(), "nn_classifier_model.pth")
        joblib.dump(bundle["scaler"], "nn_scaler.joblib")
        thresholds_data = {
            "threshold": float(bundle["threshold"]),
            "recon_scorer": bundle.get("recon_scorer"),
            "feature_cols": list(feature_columns)
        }
        joblib.dump(thresholds_data, "anomaly_thresholds.joblib")
        print("\n[INFO] Saved nn_classifier_model.pth, nn_scaler.joblib, and anomaly_thresholds.joblib.")

    ood_metrics = run_final_ood_evaluation(bundle)
    indist_binary, indist_mc = evaluate_frame(bundle, df_indist_test, mode="in_dist")
    if indist_mc is None:
        raise RuntimeError("Expected multiclass metrics for in-distribution evaluation.")

    common = {
        "experiment": experiment_name,
        "variant": variant_name,
        "feature_set": feature_set_name,
        "train_file": TRAIN_FILENAME,
        "final_ood_file": FINAL_OOD_FILENAME,
        "n_features": len(feature_columns),
    }
    ood_metrics.update(common)
    indist_mc.update(common)
    indist_binary.update({f"in_dist_{k}": v for k, v in common.items()})
    ood_rows.append(ood_metrics)
    mc_rows.append(indist_mc)

    print(
        f"  [OOD binary]  bal_acc={ood_metrics['bal_acc']:.4f}  macro_f1={ood_metrics['macro_f1']:.4f}  "
        f"attack_recall={ood_metrics['attack_recall']:.4f}  benign_spec={ood_metrics['benign_specificity']:.4f}  "
        f"unseen_recall={ood_metrics.get('unseen_recall', float('nan')):.4f}  threshold={ood_metrics['threshold']:.3f}  "
        f"attack_rate={ood_metrics['attack_rate']:.3f}"
    )
    print(
        f"  [MC in-dist]  bal_acc={indist_mc['bal_acc']:.4f}  hier_f1={indist_mc['hier_f1']:.4f}  "
        f"subtype_f1={indist_mc['subtype_f1']:.4f}  macro_f1={indist_mc['macro_f1']:.4f}"
    )
    return bundle


primary_compressed_losses_bundle = run_feature_set(
    variant_name="primary",
    feature_set_name="compressed + losses",
    experiment_name="primary__compressed_plus_losses",
)


# ## Full compressed-only ablation
# 

# In[12]:

primary_compressed_only_bundle = run_feature_set(
    variant_name="primary",
    feature_set_name="compressed only",
    experiment_name="primary__compressed_only",
)


# ## Full All-Features Ablation
# 

# In[13]:

primary_all_compressed_features_bundle = run_feature_set(
    variant_name="primary",
    feature_set_name="all compressed features",
    experiment_name="primary__all_compressed_features",
)


# ## No-time-adversary compressed + losses ablation
# 

# In[14]:

no_time_adv_compressed_losses_bundle = run_feature_set(
    variant_name="no_time_adv",
    feature_set_name="compressed + losses",
    experiment_name="no_time_adv__compressed_plus_losses",
)


# ## No-student compressed + losses ablation
# 

# In[15]:

no_student_compressed_losses_bundle = run_feature_set(
    variant_name="no_student",
    feature_set_name="compressed + losses",
    experiment_name="no_student__compressed_plus_losses",
)


# ## Save metrics

# In[16]:


ood_df = pd.DataFrame(ood_rows)
mc_df = pd.DataFrame(mc_rows)

ood_df.to_csv("neural_ood_binary_metrics.csv", index=False)
mc_df.to_csv("neural_multiclass_metrics.csv", index=False)
ood_df.to_csv("neural_vae_ablation_ood_binary_metrics.csv", index=False)
mc_df.to_csv("neural_vae_ablation_multiclass_metrics.csv", index=False)

metadata = {
    "variant_configs": VARIANT_CONFIGS,
    "experiments": list(experiment_results.keys()),
    "feature_sets_run": [
        {
            "experiment": row.get("experiment"),
            "variant": row.get("variant"),
            "feature_set": row.get("feature_set"),
            "train_file": row.get("train_file"),
            "final_ood_file": row.get("final_ood_file"),
            "n_features": row.get("n_features"),
        }
        for row in ood_rows
    ],
    "known_ood_constraint": "each final OOD CSV is loaded only inside run_final_ood_evaluation after that experiment trains and selects its threshold",
    "score_name_by_experiment": {
        name: bundle["scorer"]["score_name"] for name, bundle in experiment_results.items()
    },
    "threshold_policy_by_experiment": {
        name: bundle["threshold_metrics"] for name, bundle in experiment_results.items()
    },
}

with open("neural_experiment_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("saved neural_ood_binary_metrics.csv")
print("saved neural_multiclass_metrics.csv")
print("saved neural_vae_ablation_ood_binary_metrics.csv")
print("saved neural_vae_ablation_multiclass_metrics.csv")
print("saved neural_experiment_metadata.json")

