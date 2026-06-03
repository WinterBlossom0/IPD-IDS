import os, random, time, copy, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support

# Constants
RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 65536
LR = 3e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define MLP
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=(256, 128, 64), dropout=0.2):
        super().__init__()
        layers = [] 
        prev = input_dim
        for dim in hidden_dims:
            layers += [nn.Linear(prev, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = dim
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def hierarchical_f1(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1_bin = f1_score((y_true > 0), (y_pred > 0), average='macro', zero_division=0)
    mask   = y_true > 0
    f1_att = f1_score(y_true[mask], y_pred[mask], average='macro', zero_division=0) if mask.any() else 0.0
    return min(f1_bin, f1_att)

# Load data
df_all_X = pd.read_csv('df_train.csv')
df_test_odd = pd.read_csv('df_test.csv')

selected_attacks = [3, 4, 1, 2] # top 4 attacks
keep_labels = [0] + selected_attacks
df_base = df_all_X[df_all_X['label'].isin(keep_labels)].copy()

label_to_model_id = {orig_label: model_id for model_id, orig_label in enumerate(keep_labels)}
df_base['label'] = df_base['label'].map(label_to_model_id).astype(int)

# Split 4:1:1
rng = np.random.default_rng(RANDOM_STATE)
train_indices, val_indices, test_indices = [], [], []
for cls in sorted(df_base['label'].unique()):
    cls_idx = df_base.index[df_base['label'] == cls].to_numpy()
    n_total = len(cls_idx)
    perm = rng.permutation(cls_idx)
    n_train = n_total * 4 // 6
    n_val   = n_total * 1 // 6
    train_indices.extend(perm[:n_train])
    val_indices.extend(perm[n_train:n_train + n_val])
    test_indices.extend(perm[n_train + n_val:])

train_indices = rng.permutation(np.array(train_indices)).tolist()
val_indices   = rng.permutation(np.array(val_indices)).tolist()
test_indices  = rng.permutation(np.array(test_indices)).tolist()

df_train_X = df_base.loc[train_indices].reset_index(drop=True)
df_val_X   = df_base.loc[val_indices].reset_index(drop=True)
df_test_X  = df_base.loc[test_indices].reset_index(drop=True)

# Feature set: compressed + losses
latent_cols = [c for c in df_train_X.columns if c.startswith('latent_')]
loss_cols   = ['student_loss', 'mse_loss', 'kld_loss']
cols = latent_cols + loss_cols

X_train = df_train_X[cols].values
y_train = df_train_X['label'].values
X_val = df_val_X[cols].values
y_val = df_val_X['label'].values
X_test_mc = df_test_X[cols].values
y_test_mc = df_test_X['label'].values
X_test_ood = df_test_odd[cols].values
y_test_ood = df_test_odd['label'].values

# Scale
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_v = scaler.transform(X_val)
X_te_mc = scaler.transform(X_test_mc)
X_te_ood = scaler.transform(X_test_ood)

def evaluate(model, X, y_true):
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        proba = F.softmax(model(tensor), dim=1).cpu().numpy()
    preds_mc = np.where(1.0 - proba[:, 0] >= 0.5, 1 + np.argmax(proba[:, 1:], axis=1), 0)
    preds_bin = (preds_mc != 0).astype(int)
    y_true_bin = (y_true != 0).astype(int)
    bal_acc = balanced_accuracy_score(y_true_bin, preds_bin)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_bin, preds_bin, average='binary', zero_division=0)
    return bal_acc, recall, 1.0 - preds_bin.mean()

# Test different loss strategies
strategies = ['unweighted', 'binary_balanced', 'subclass_inverse']

for strat in strategies:
    print(f"\n--- Strategy: {strat} ---")
    
    # Calculate weights
    num_classes = len(keep_labels)
    if strat == 'unweighted':
        weights = None
    elif strat == 'binary_balanced':
        counts = np.bincount(y_train, minlength=num_classes).astype(float)
        benign_count = counts[0]
        attack_count = counts[1:].sum()
        w_attack = benign_count / attack_count
        w = np.ones(num_classes)
        w[1:] = w_attack
        weights = torch.tensor(w, dtype=torch.float32).to(DEVICE)
    elif strat == 'subclass_inverse':
        counts = np.bincount(y_train, minlength=num_classes).astype(float)
        N = counts.sum()
        w = N / (num_classes * counts)
        w = w / w.sum() * num_classes
        weights = torch.tensor(w, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights)

    # Loader
    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Train
    torch.manual_seed(RANDOM_STATE)
    model = MLP(X_tr.shape[1], num_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    best_val_hier_f1 = -np.inf
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        
        # Eval val
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(X_v, dtype=torch.float32).to(DEVICE))
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
        val_hier_f1 = hierarchical_f1(y_val, val_preds)
        if val_hier_f1 > best_val_hier_f1:
            best_val_hier_f1 = val_hier_f1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    
    # Evaluate on OOD and In-Dist
    ood_bal, ood_rec, ood_spec = evaluate(model, X_te_ood, y_test_ood)
    ind_bal, ind_rec, ind_spec = evaluate(model, X_te_mc, y_test_mc)
    print(f"OOD: bal_acc={ood_bal:.4f} | recall={ood_rec:.4f} | specificity={ood_spec:.4f}")
    print(f"In-Dist: bal_acc={ind_bal:.4f} | recall={ind_rec:.4f} | specificity={ind_spec:.4f}")
