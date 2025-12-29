"""
EMG Muscle-Fatigue Detection using Deep Learning

This script implements a deep learning approach for detecting muscle fatigue
from EMG signals using a hybrid CNN-LSTM architecture with attention.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import joblib
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Import preprocessing from existing code
from svm_freq_baseline import EMGDataLoader, EMGPreprocessor

class EMGDataset(Dataset):
    """PyTorch Dataset for EMG data with on-the-fly preprocessing and augmentation"""
    
    def __init__(self, emg_data: np.ndarray, labels: np.ndarray, 
                 preprocessor: EMGPreprocessor, window_size: int = 1000, 
                 hop_size: int = 500, augment: bool = False):
        """
        Args:
            emg_data: Raw EMG data (samples x channels)
            labels: Corresponding labels (0=no fatigue, 1=fatigue)
            preprocessor: Configured EMGPreprocessor instance
            window_size: Size of sliding window in samples
            hop_size: Hop size between windows in samples
            augment: Whether to apply data augmentation
        """
        self.emg_data = emg_data
        self.labels = labels
        self.preprocessor = preprocessor
        self.window_size = window_size
        self.hop_size = hop_size
        self.augment = augment
        
        # Preprocess entire signal
        self.processed_data = self.preprocessor.preprocess(emg_data)
        
        # Generate window indices
        self.windows = []
        for start in range(0, len(self.processed_data) - window_size + 1, hop_size):
            end = start + window_size
            self.windows.append((start, end))
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        start, end = self.windows[idx]
        window = self.processed_data[start:end].copy()
        label = self.labels[start:end].mean() > 0.5  # Majority vote for window
        
        if self.augment:
            window = self._augment_window(window)
            
        # Add channel dimension if needed (for 1D conv)
        window = torch.FloatTensor(window).permute(1, 0)  # (channels, time)
        return window, torch.FloatTensor([label])
    
    def _augment_window(self, window: np.ndarray) -> np.ndarray:
        """Apply data augmentation to a window"""
        # Time warping
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            new_length = int(len(window) * scale)
            window = np.interp(
                np.linspace(0, len(window)-1, new_length),
                np.arange(len(window)),
                window
            )
            if len(window) > self.window_size:
                window = window[:self.window_size]
            elif len(window) < self.window_size:
                window = np.pad(window, (0, self.window_size - len(window)))
        
        # Additive white noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.05 * window.std(), window.shape)
            window = window + noise
            
        return window


class FatigueDetector(nn.Module):
    """Hybrid CNN-LSTM model for EMG fatigue detection"""
    
    def __init__(self, input_channels: int, num_classes: int = 1, 
                 dropout: float = 0.3, lstm_units: int = 64):
        super().__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            # Conv1d(in_channels, out_channels, kernel_size)
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_units,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if 2 > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_units * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_units * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # x shape: (batch, channels, time)
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = self.cnn(x)  # (batch, 128, time')
        
        # Prepare for LSTM (batch, time, features)
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, time, lstm_units*2)
        
        # Attention
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), 
            dim=1
        ).unsqueeze(-1)
        context = (lstm_out * attention_weights).sum(dim=1)
        
        # Classification
        output = self.classifier(context)
        return output.squeeze(-1)


def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-4):
    """Train the model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("  Model saved!")
    
    return model


def evaluate(model, data_loader, criterion, device):
    """Evaluate the model"""
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = val_loss / len(data_loader.dataset)
    val_acc = 100. * val_correct / val_total
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, (np.array(all_preds) > 0.5).astype(int)))
    
    return val_loss, val_acc


def train_fatigue_detector(data_root: str, output_dir: str = "dl_model_out", 
                          n_folds: int = 5, batch_size: int = 32, 
                          num_epochs: int = 50):
    """
    Train the deep learning model for EMG fatigue detection.
    
    Args:
        data_root: Path to folder containing subject_* folders
        output_dir: Directory to save model artifacts
        n_folds: Number of cross-validation folds
        batch_size: Batch size for training
        num_epochs: Number of training epochs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data loader and preprocessor
    data_loader = EMGDataLoader(data_root)
    preprocessor = EMGPreprocessor(sampling_rate=2000)
    
    # Load and preprocess all data
    print("Loading and preprocessing data...")
    all_data = []
    all_labels = []
    subject_ids = []
    
    # This is a simplified example - you'll need to adapt this to your data structure
    for subject_dir in os.listdir(data_root):
        if not subject_dir.startswith('subject_'):
            continue
            
        subject_path = os.path.join(data_root, subject_dir)
        subject_id = int(subject_dir.split('_')[-1])
        
        for trial_file in os.listdir(subject_path):
            if not trial_file.endswith('.csv'):
                continue
                
            # Load and preprocess trial
            filepath = os.path.join(subject_path, trial_file)
            emg_data, _ = data_loader.load_trial(filepath)
            
            # For this example, we'll use the pseudo-label generator
            # In practice, you should use your actual labels
            features = FrequencyFeatureExtractor().extract_features(emg_data)
            feature_names = [f'ch{i}_{feat}' for i in range(emg_data.shape[1]) 
                           for feat in ['mnf', 'mdf', 'power']]
            
            pl_generator = PseudoLabelGenerator()
            labels = pl_generator.generate_labels(features, feature_names)
            
            # Add to dataset
            all_data.append(emg_data)
            all_labels.extend(labels)
            subject_ids.extend([subject_id] * len(labels))
    
    # Convert to numpy arrays
    all_data = np.vstack(all_data)
    all_labels = np.array(all_labels)
    subject_ids = np.array(subject_ids)
    
    # Setup cross-validation
    group_kfold = GroupKFold(n_splits=n_folds)
    
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(all_data, all_labels, subject_ids)):
        print(f"\nFold {fold+1}/{n_folds}")
        print("-" * 30)
        
        # Create datasets
        train_dataset = EMGDataset(
            all_data[train_idx], all_labels[train_idx],
            preprocessor=preprocessor,
            window_size=2000,  # 1 second window at 2000Hz
            hop_size=1000,     # 0.5 second hop
            augment=True
        )
        
        val_dataset = EMGDataset(
            all_data[val_idx], all_labels[val_idx],
            preprocessor=preprocessor,
            window_size=2000,
            hop_size=1000,
            augment=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize model
        model = FatigueDetector(
            input_channels=all_data.shape[1],  # Number of EMG channels
            num_classes=1,
            dropout=0.3,
            lstm_units=64
        )
        
        # Train model
        train_model(
            model, 
            train_loader, 
            val_loader,
            num_epochs=num_epochs,
            lr=1e-4
        )
        
        # Save the best model for this fold
        model_path = os.path.join(output_dir, f'model_fold_{fold+1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Saved model for fold {fold+1} to {model_path}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train EMG fatigue detection model')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to folder containing subject_* folders')
    parser.add_argument('--output_dir', type=str, default='dl_model_out',
                        help='Directory to save model artifacts')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    
    args = parser.parse_args()
    
    train_fatigue_detector(
        data_root=args.data_root,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        batch_size=args.batch_size,
        num_epochs=args.epochs
    )
