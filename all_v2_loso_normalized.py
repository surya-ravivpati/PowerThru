"""
WESAD Multi-Modal Stress Detection with Leave-One-Subject-Out Cross-Validation
WITH PER-SUBJECT NORMALIZATION

Key improvements:
1. Per-subject z-score normalization before windowing
2. Subject-specific baseline subtraction
3. Robust scaling to handle outliers
"""
import os
import pickle
import numpy as np
from collections import Counter

from scipy.signal import resample
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

DATA_ROOT = "/Users/yashravipati/Desktop/PowerThruAI/WESAD/"
USE_SUBJECTS = None

# WESAD sampling rates
FS_LABEL = 700.0
FS_EDA   = 4.0
FS_TEMP  = 4.0
FS_ACC   = 32.0
FS_TARGET = 16.0

# Windowing
WINDOW_SEC = 60.0
STEP_SEC = 30.0
MIN_LABEL_PURITY = 0.8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# ------------------------------------------------------------
# DATA LOADING WITH PER-SUBJECT NORMALIZATION
# ------------------------------------------------------------

def resample_to_target(x, fs_in, fs_out, axis=0):
    if fs_in == fs_out:
        return x
    T = x.shape[axis]
    duration = T / fs_in
    T_new = int(round(duration * fs_out))
    return resample(x, T_new, axis=axis)


def downsample_labels(labels, fs_in=FS_LABEL, fs_out=FS_TARGET):
    ratio = int(round(fs_in / fs_out))
    return labels[::ratio]


def normalize_subject_data(eda, temp, acc):
    """
    Apply per-subject z-score normalization to reduce subject-specific variability.
    Uses robust statistics (median, IQR) to handle outliers.
    """
    # EDA normalization
    eda_median = np.median(eda)
    eda_iqr = np.percentile(eda, 75) - np.percentile(eda, 25)
    eda_normalized = (eda - eda_median) / (eda_iqr + 1e-8)
    
    # TEMP normalization
    temp_median = np.median(temp)
    temp_iqr = np.percentile(temp, 75) - np.percentile(temp, 25)
    temp_normalized = (temp - temp_median) / (temp_iqr + 1e-8)
    
    # ACC normalization (per axis)
    acc_normalized = np.zeros_like(acc)
    for i in range(acc.shape[1]):
        acc_median = np.median(acc[:, i])
        acc_iqr = np.percentile(acc[:, i], 75) - np.percentile(acc[:, i], 25)
        acc_normalized[:, i] = (acc[:, i] - acc_median) / (acc_iqr + 1e-8)
    
    return eda_normalized, temp_normalized, acc_normalized


def load_wesad_subject(path):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    wrist = data["signal"]["wrist"]
    labels = data["label"].reshape(-1)
    eda  = wrist["EDA"].reshape(-1)
    temp = wrist["TEMP"].reshape(-1)
    acc  = wrist["ACC"]
    return eda, temp, acc, labels


def align_modalities(eda, temp, acc, labels):
    eda_r  = resample_to_target(eda,  FS_EDA,  FS_TARGET)
    temp_r = resample_to_target(temp, FS_TEMP, FS_TARGET)
    acc_r  = resample_to_target(acc,  FS_ACC,  FS_TARGET, axis=0)
    labels_r = downsample_labels(labels, FS_LABEL, FS_TARGET)
    
    T = min(len(eda_r), len(temp_r), len(labels_r), acc_r.shape[0])
    return eda_r[:T], temp_r[:T], acc_r[:T], labels_r[:T]


def make_windows(eda, temp, acc, labels, fs_target=FS_TARGET,
                 window_sec=WINDOW_SEC, step_sec=STEP_SEC, 
                 min_label_purity=MIN_LABEL_PURITY):
    win_len = int(window_sec * fs_target)
    step = int(step_sec * fs_target)
    
    X_list, y_list = [], []
    T = len(labels)
    
    for start in range(0, T - win_len + 1, step):
        end = start + win_len
        win_labels = labels[start:end]
        
        label_counts = Counter(win_labels)
        lbl, count = label_counts.most_common(1)[0]
        purity = count / len(win_labels)
        
        if lbl == 0 or lbl >= 5 or purity < min_label_purity:
            continue
        
        acc_win  = acc[start:end, :]
        eda_win  = eda[start:end].reshape(-1, 1)
        temp_win = temp[start:end].reshape(-1, 1)
        seq = np.concatenate([acc_win, eda_win, temp_win], axis=1)
        
        X_list.append(seq)
        y_list.append(lbl)
    
    if not X_list:
        return np.empty((0, win_len, 5)), np.empty((0,), dtype=int)
    
    return np.stack(X_list, axis=0), np.array(y_list, dtype=int)


def load_all_subjects(data_root=DATA_ROOT, use_subjects=USE_SUBJECTS):
    X_all, y_all, subj_ids = [], [], []
    
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    
    subject_dirs = sorted([d for d in os.listdir(data_root) 
                          if os.path.isdir(os.path.join(data_root, d)) and d.startswith("S")])
    
    if not subject_dirs:
        raise FileNotFoundError(f"No S* subject directories found in {data_root}")
    
    print(f"Found {len(subject_dirs)} subject directories")
    
    for subj_dir in subject_dirs:
        try:
            sid = int(subj_dir[1:])
        except ValueError:
            continue
        
        if use_subjects is not None and sid not in use_subjects:
            continue
        
        pkl_path = os.path.join(data_root, subj_dir, f"{subj_dir}.pkl")
        if not os.path.exists(pkl_path):
            continue
        
        try:
            eda, temp, acc, labels = load_wesad_subject(pkl_path)
            
            # Apply per-subject normalization BEFORE alignment and windowing
            eda_norm, temp_norm, acc_norm = normalize_subject_data(eda, temp, acc)
            
            eda_r, temp_r, acc_r, labels_r = align_modalities(eda_norm, temp_norm, acc_norm, labels)
            X, y = make_windows(eda_r, temp_r, acc_r, labels_r)
            
            if len(X) > 0:
                print(f"Subject {sid}: {len(X)} windows, labels: {dict(Counter(y))}")
                X_all.append(X)
                y_all.append(y)
                subj_ids.append(np.full(len(y), sid))
        except Exception as e:
            print(f"ERROR loading subject {sid}: {e}")
            continue
    
    if not X_all:
        raise RuntimeError("No valid subjects found")
    
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    subj_ids = np.concatenate(subj_ids, axis=0)
    
    return X_all, y_all, subj_ids


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

class WESADDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BiLSTMAttention(nn.Module):
    def __init__(self, input_size=5, hidden_size=48, num_layers=1,
                 num_classes=4, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.attn_fc = nn.Linear(hidden_size * 2, 1)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attn_fc(torch.tanh(lstm_out)).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = (lstm_out * attn_weights).sum(dim=1)
        x = torch.relu(self.fc1(context))
        x = self.dropout(x)
        return self.fc2(x)


# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()
    
    return total_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    return total_loss / len(dataloader), 100.0 * correct / total, all_preds, all_labels


def train_one_fold(X_train, y_train, X_test, y_test, num_classes, 
                   input_channels, seq_length, device, epochs=30):
    """Train model for one LOSO fold"""
    
    # Scale features using StandardScaler (data already normalized per-subject)
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, input_channels)
    X_test_flat = X_test.reshape(-1, input_channels)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)
    
    # Create datasets
    train_dataset = WESADDataset(X_train_scaled, y_train)
    test_dataset = WESADDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model
    model = BiLSTMAttention(
        input_size=input_channels,
        hidden_size=48,
        num_layers=1,
        num_classes=num_classes,
        dropout=0.4
    ).to(device)
    
    # Class weights
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    class_weights = len(y_train) / (len(unique_classes) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    
    # Training loop (simplified - no validation, just train for fixed epochs)
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    
    # Final evaluation on test subject
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    return test_acc, test_preds, test_labels


# ------------------------------------------------------------
# LOSO CROSS-VALIDATION
# ------------------------------------------------------------

def loso_cross_validation(X, y, subj_ids, num_classes, input_channels, seq_length, device):
    """
    Leave-One-Subject-Out Cross-Validation
    """
    unique_subjects = np.unique(subj_ids)
    print(f"\n{'='*60}")
    print(f"Leave-One-Subject-Out Cross-Validation")
    print(f"{'='*60}")
    print(f"Total subjects: {len(unique_subjects)}")
    print(f"Subjects: {sorted(unique_subjects)}\n")
    
    all_test_accs = []
    all_test_f1s = []
    all_preds = []
    all_labels = []
    
    for i, test_subj in enumerate(unique_subjects, 1):
        print(f"[Fold {i}/{len(unique_subjects)}] Test subject: {test_subj}")
        
        # Split data
        test_mask = (subj_ids == test_subj)
        train_mask = ~test_mask
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        print(f"  Train: {len(X_train)} windows, Test: {len(X_test)} windows")
        print(f"  Train labels: {dict(Counter(y_train))}")
        print(f"  Test labels: {dict(Counter(y_test))}")
        
        # Train and evaluate
        test_acc, test_preds, test_labels = train_one_fold(
            X_train, y_train, X_test, y_test,
            num_classes, input_channels, seq_length, device, epochs=30
        )
        
        # Calculate F1 score
        test_f1 = f1_score(test_labels, test_preds, average='weighted')
        
        all_test_accs.append(test_acc)
        all_test_f1s.append(test_f1)
        all_preds.extend(test_preds)
        all_labels.extend(test_labels)
        
        print(f"  → Test Accuracy: {test_acc:.2f}%, F1: {test_f1:.4f}\n")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"LOSO Cross-Validation Results")
    print(f"{'='*60}")
    print(f"Mean Accuracy: {np.mean(all_test_accs):.2f}% ± {np.std(all_test_accs):.2f}%")
    print(f"Mean F1 Score: {np.mean(all_test_f1s):.4f} ± {np.std(all_test_f1s):.4f}")
    print(f"\nPer-subject accuracies:")
    for subj, acc in zip(unique_subjects, all_test_accs):
        print(f"  Subject {subj}: {acc:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"Overall Classification Report")
    print(f"{'='*60}")
    print(classification_report(all_labels, all_preds))
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    return {
        'mean_acc': np.mean(all_test_accs),
        'std_acc': np.std(all_test_accs),
        'per_subject_accs': dict(zip(unique_subjects, all_test_accs)),
        'all_preds': all_preds,
        'all_labels': all_labels
    }


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("WESAD LOSO Cross-Validation WITH SUBJECT NORMALIZATION")
    print("="*60)
    
    try:
        # Load data
        X, y, subj_ids = load_all_subjects()
        
        print(f"\nTotal windows: {len(X)}")
        print(f"Window shape: {X.shape[1:]} (time_steps, channels)")
        print(f"Label distribution: {Counter(y)}")
        
        # Remap labels to 0-indexed
        unique_labels = np.unique(y)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y_remapped = np.array([label_map[lbl] for lbl in y])
        
        print(f"\nLabel mapping: {label_map}")
        
        num_classes = len(unique_labels)
        seq_length = X.shape[1]
        input_channels = X.shape[2]
        
        # Run LOSO CV
        results = loso_cross_validation(
            X, y_remapped, subj_ids,
            num_classes, input_channels, seq_length, DEVICE
        )
        
        print(f"\n{'='*60}")
        print(f"Final Result: {results['mean_acc']:.2f}% ± {results['std_acc']:.2f}%")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
