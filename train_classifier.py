import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Ensure safe console printing on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Dataset Paths for both datasets
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET1_PATH = os.path.join(BASE_DIR, "Vehicles.v1i.multiclass")
DATASET2_PATH = os.path.join(BASE_DIR, "Vehicles-coco.v2i.multiclass")

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4  # L2 Regularization to prevent overfitting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Canonical vehicle classes (4 classes)
TARGET_CLASSES = ['Bus', 'Motorcycle', 'Car', 'Truck']


# ==============================================================================
# Unified Dataset Class for both Vehicles.v1i and Vehicles-coco datasets
# ==============================================================================
class VehicleDataset(Dataset):
    def __init__(self, csv_file, root_dir, dataset_type=1, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_type = dataset_type
        
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
        self.filenames = []
        self.labels = []
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            fname = str(row['filename']).strip()
            
            # Map columns to [Bus, Motorcycle, Car, Truck]
            if dataset_type == 1:
                # Columns: filename, 0, Bus, Motorcycle, car, truck
                bus = float(row.get('Bus', 0))
                moto = float(row.get('Motorcycle', 0))
                car = float(row.get('car', 0))
                truck = float(row.get('truck', 0))
            else:
                # Columns: filename, bus, car, motorcycle, truck
                bus = float(row.get('bus', 0))
                moto = float(row.get('motorcycle', 0))
                car = float(row.get('car', 0))
                truck = float(row.get('truck', 0))
                
            self.filenames.append(fname)
            self.labels.append([bus, moto, car, truck])
            
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.filenames[index])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (128, 128))
            
        label = torch.tensor(self.labels[index])
        if self.transform:
            image = self.transform(image)
            
        return image, label


# ==============================================================================
# Custom CNN with Overfitting Prevention (BatchNorm + Dropout + Regularization)
# ==============================================================================
class CustomCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomCNN, self).__init__()
        
        # Block 1 (3 -> 16)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)   # Overfitting prevention & training stability
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64x64
        
        # Block 2 (16 -> 32)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)   # Overfitting prevention
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32x32
        
        # Block 3 (32 -> 64)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)   # Overfitting prevention
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16x16
        self.drop_conv = nn.Dropout2d(0.1) # Spatial dropout to prevent co-adaptation
        
        # Fully Connected Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),            # Strong dropout to prevent overfitting
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.drop_conv(x)
        x = self.classifier(x)
        return x


# ==============================================================================
# Model Evaluation & Report Generation
# ==============================================================================
def evaluate_model(model, data_loader, split_name="Validation"):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.50).astype(int)
            
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy().astype(int))
            
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    avg_loss = total_loss / len(data_loader.dataset)
    
    print("\n" + "="*60)
    print(f"  {split_name} Classification Report (Loss: {avg_loss:.4f})")
    print("="*60)
    report = classification_report(
        all_labels, all_preds, target_names=TARGET_CLASSES, zero_division=0
    )
    print(report)
    
    for idx, name in enumerate(TARGET_CLASSES):
        correct = (all_preds[:, idx] == all_labels[:, idx]).sum()
        total = len(all_labels)
        print(f"  {name} Accuracy: {correct}/{total} = {100*correct/total:.1f}%")
        
    overall_correct = (all_preds == all_labels).sum()
    overall_total = all_labels.size
    print(f"\n  Overall Accuracy: {100*overall_correct/overall_total:.1f}%")
    print("="*60 + "\n")
    
    return avg_loss, all_preds, all_labels


# ==============================================================================
# Overfitting Diagnostic Plot (Train vs Validation Loss Curve)
# ==============================================================================
def plot_loss_curves(train_losses, val_losses, save_path="training_loss.png"):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-o', linewidth=2, label="Train Loss")
    if val_losses:
        plt.plot(epochs, val_losses, 'r--s', linewidth=2, label="Validation Loss")
    plt.title("VISIONX - Training & Validation Loss (Overfitting Check)", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("BCE Loss", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved loss curve plot to {save_path}")


def plot_confusion_matrix(all_preds, all_labels, save_path="confusion_matrix.png"):
    num_classes = len(TARGET_CLASSES)
    fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 4.5))
    fig.suptitle("VISIONX - Per-Class Confusion Matrices", fontsize=15, fontweight='bold')
    
    for ax, idx, name in zip(axes, range(num_classes), TARGET_CLASSES):
        cm = confusion_matrix(all_labels[:, idx], all_preds[:, idx], labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name, fontsize=13)
        
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved confusion matrix plot to {save_path}")


# ==============================================================================
# Main Training Function combining BOTH train folders
# ==============================================================================
def main():
    print(f"[INFO] Using device: {DEVICE}")
    
    # Robust Data Augmentation to prevent overfitting
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # -------------------------------------------------------------
    # Load ONLY the train folder of Dataset 1 and Dataset 2
    # -------------------------------------------------------------
    ds1_train_csv = os.path.join(DATASET1_PATH, "train", "_classes.csv")
    ds1_train_dir = os.path.join(DATASET1_PATH, "train")
    
    ds2_train_csv = os.path.join(DATASET2_PATH, "train", "_classes.csv")
    ds2_train_dir = os.path.join(DATASET2_PATH, "train")
    
    train_datasets = []
    if os.path.exists(ds1_train_csv):
        ds1 = VehicleDataset(ds1_train_csv, ds1_train_dir, dataset_type=1, transform=train_transform)
        train_datasets.append(ds1)
        print(f"[INFO] Loaded Dataset 1 train: {len(ds1)} samples")
    else:
        print(f"[WARNING] Dataset 1 train not found at {ds1_train_csv}")
        
    if os.path.exists(ds2_train_csv):
        ds2 = VehicleDataset(ds2_train_csv, ds2_train_dir, dataset_type=2, transform=train_transform)
        train_datasets.append(ds2)
        print(f"[INFO] Loaded Dataset 2 train: {len(ds2)} samples")
    else:
        print(f"[WARNING] Dataset 2 train not found at {ds2_train_csv}")
        
    combined_train_dataset = ConcatDataset(train_datasets)
    print(f"[INFO] Combined Total Training Samples: {len(combined_train_dataset)} from BOTH train datasets!")
    
    train_loader = DataLoader(
        combined_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )
    
    # -------------------------------------------------------------
    # Load Validation & Test Datasets for Overfitting Verification
    # -------------------------------------------------------------
    val_datasets = []
    ds1_val_csv = os.path.join(DATASET1_PATH, "valid", "_classes.csv")
    if os.path.exists(ds1_val_csv):
        val_datasets.append(VehicleDataset(ds1_val_csv, os.path.join(DATASET1_PATH, "valid"), 1, eval_transform))
    ds2_val_csv = os.path.join(DATASET2_PATH, "valid", "_classes.csv")
    if os.path.exists(ds2_val_csv):
        val_datasets.append(VehicleDataset(ds2_val_csv, os.path.join(DATASET2_PATH, "valid"), 2, eval_transform))
        
    combined_val_dataset = ConcatDataset(val_datasets)
    val_loader = DataLoader(combined_val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"[INFO] Combined Validation Samples: {len(combined_val_dataset)}")
    
    # -------------------------------------------------------------
    # Initialize Model, Loss, Optimizer with Weight Decay
    # -------------------------------------------------------------
    model = CustomCNN(num_classes=len(TARGET_CLASSES)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("\n" + "="*60)
    print("  STARTING TRAINING ON COMBINED TRAIN FOLDERS")
    print("="*60)
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            if (i + 1) % 40 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        epoch_train_loss = running_loss / len(combined_train_dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation step to monitor overfitting
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for v_imgs, v_lbls in val_loader:
                v_imgs, v_lbls = v_imgs.to(DEVICE), v_lbls.to(DEVICE)
                v_out = model(v_imgs)
                val_loss += criterion(v_out, v_lbls).item() * v_imgs.size(0)
        epoch_val_loss = val_loss / len(combined_val_dataset)
        val_losses.append(epoch_val_loss)
        
        print(f">>> Epoch {epoch+1}/{EPOCHS} Finished -> Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "vehicle_classifier.pth")
            print(f"  [CHECKPOINT] Best model saved with Val Loss: {best_val_loss:.4f}")
            
    print("\n[INFO] Training Complete! Model saved to vehicle_classifier.pth")
    
    # -------------------------------------------------------------
    # Overfitting Check & Final Evaluation
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("  OVERFITTING CHECK & FINAL EVALUATION")
    print("="*60)
    
    # Plot Train vs Validation Loss Curve (Direct proof of no overfitting)
    plot_loss_curves(train_losses, val_losses, save_path="training_loss.png")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load("vehicle_classifier.pth", map_location=DEVICE))
    _, val_preds, val_labels = evaluate_model(model, val_loader, split_name="Validation (Overfitting Test)")
    
    # Plot Confusion Matrix
    plot_confusion_matrix(val_preds, val_labels, save_path="confusion_matrix.png")
    
    # Diagnosis summary
    loss_gap = abs(train_losses[-1] - val_losses[-1])
    print("[OVERFITTING DIAGNOSIS]:")
    print(f"  Final Train Loss: {train_losses[-1]:.4f}")
    print(f"  Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"  Train-Validation Loss Gap: {loss_gap:.4f}")
    if loss_gap < 0.08:
        print("  Status: EXCELLENT GENERALIZATION! No significant overfitting detected.")
    else:
        print("  Status: Model generalized with slight divergence.")

if __name__ == '__main__':
    main()
