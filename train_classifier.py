import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Configuration
# Resolves dataset path either locally, in parent directory, or via custom environment variable
DEFAULT_LOCAL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Vehicles.v1i.multiclass")
DATASET_PATH = os.environ.get("DATASET_PATH", DEFAULT_LOCAL_PATH if os.path.exists(DEFAULT_LOCAL_PATH) else "Vehicles.v1i.multiclass")
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VehicleDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Strip whitespace from column names just in case
        self.annotations.columns = self.annotations.columns.str.strip()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = str(self.annotations.iloc[index, 0]).strip()
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Fallback if image is missing
            image = Image.new("RGB", (128, 128))
            
        # The columns are: filename, 0, Bus, Motorcycle, car, truck
        # We take the labels from index 1 to end
        labels = self.annotations.iloc[index, 1:].values.astype('float32')
        labels = torch.tensor(labels)

        if self.transform:
            image = self.transform(image)

        return image, labels

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        # Simple custom CNN to avoid using pretrained models
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64x64
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32x32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def evaluate_model(model, data_loader, class_names, split_name="Validation"):
    """Evaluate model on a dataset and print classification report."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # Convert to binary predictions (threshold = 0.5)
            preds = (probs > 0.5).astype(int)
            all_preds.append(preds)
            all_labels.append(labels.numpy().astype(int))
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Per-class classification report
    print(f"\n{'='*60}")
    print(f"  {split_name} Classification Report")
    print(f"{'='*60}")
    
    # Use only real vehicle classes for the report (skip column '0')
    real_class_names = [c for c in class_names if c != '0']
    real_class_indices = [i for i, c in enumerate(class_names) if c != '0']
    
    report = classification_report(
        all_labels[:, real_class_indices],
        all_preds[:, real_class_indices],
        target_names=real_class_names,
        zero_division=0
    )
    print(report)
    
    # Per-class accuracy
    print(f"Per-class Accuracy:")
    for idx, name in zip(real_class_indices, real_class_names):
        correct = (all_preds[:, idx] == all_labels[:, idx]).sum()
        total = len(all_labels)
        print(f"  {name}: {correct}/{total} = {100*correct/total:.1f}%")
    
    # Overall accuracy (across all real classes)
    real_preds = all_preds[:, real_class_indices]
    real_labels = all_labels[:, real_class_indices]
    overall_correct = (real_preds == real_labels).sum()
    overall_total = real_labels.size
    print(f"\n  Overall Accuracy: {100*overall_correct/overall_total:.1f}%")
    print(f"{'='*60}\n")
    
    return all_preds, all_labels, real_class_names, real_class_indices


def plot_training_loss(epoch_losses, save_path="training_loss.png"):
    """Save training loss curve plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, 'b-o', linewidth=2, markersize=8)
    plt.title("VISIONX - Training Loss Curve", fontsize=16, fontweight='bold')
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("Average Loss", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, len(epoch_losses) + 1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training loss plot saved to: {save_path}")


def plot_confusion_matrix(all_preds, all_labels, class_names, class_indices, save_path="confusion_matrix.png"):
    """Save per-class confusion matrix as a combined figure."""
    num_classes = len(class_names)
    fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 5))
    fig.suptitle("VISIONX - Per-Class Confusion Matrices", fontsize=16, fontweight='bold')
    
    if num_classes == 1:
        axes = [axes]
    
    for ax, idx, name in zip(axes, class_indices, class_names):
        cm = confusion_matrix(all_labels[:, idx], all_preds[:, idx], labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name}", fontsize=13)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to: {save_path}")


def main():
    print(f"Using device: {DEVICE}")
    print(f"Dataset path: {DATASET_PATH}")
    
    # ==========================================
    # Step 2: Data Augmentation for Training
    # ==========================================
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/Test transforms (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Training Data
    train_csv = os.path.join(DATASET_PATH, 'train', '_classes.csv')
    train_dir = os.path.join(DATASET_PATH, 'train')
    
    if not os.path.exists(train_csv):
        print(f"Dataset not found at {DATASET_PATH}. Please check the path.")
        return

    train_dataset = VehicleDataset(csv_file=train_csv, root_dir=train_dir, transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load Validation Data
    valid_csv = os.path.join(DATASET_PATH, 'valid', '_classes.csv')
    valid_dir = os.path.join(DATASET_PATH, 'valid')
    valid_dataset = VehicleDataset(csv_file=valid_csv, root_dir=valid_dir, transform=eval_transform)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load Test Data
    test_csv = os.path.join(DATASET_PATH, 'test', '_classes.csv')
    test_dir = os.path.join(DATASET_PATH, 'test')
    test_dataset = VehicleDataset(csv_file=test_csv, root_dir=test_dir, transform=eval_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(train_dataset.annotations.columns) - 1
    class_names = list(train_dataset.annotations.columns)[1:]
    print(f"Classes: {class_names}")
    print(f"Samples - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

    # Initialize model, loss, optimizer
    model = CustomCNN(num_classes=num_classes).to(DEVICE)
    # Using BCEWithLogitsLoss for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ==========================================
    # Training Loop with per-epoch loss tracking
    # ==========================================
    epoch_losses = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1
            
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / batch_count
        epoch_losses.append(avg_loss)
        print(f">>> Epoch {epoch+1}/{EPOCHS} complete — Avg Loss: {avg_loss:.4f}")

    print("\nTraining complete!")
    
    # Save the model
    torch.save(model.state_dict(), 'vehicle_classifier.pth')
    print("Model saved to vehicle_classifier.pth")

    # ==========================================
    # Step 3: Validation & Test Evaluation
    # ==========================================
    print("\n" + "="*60)
    print("  EVALUATING MODEL")
    print("="*60)
    
    # Validation set evaluation
    val_preds, val_labels, real_names, real_indices = evaluate_model(
        model, valid_loader, class_names, split_name="Validation"
    )
    
    # Test set evaluation
    test_preds, test_labels, _, _ = evaluate_model(
        model, test_loader, class_names, split_name="Test"
    )

    # ==========================================
    # Step 4: Save Training Plots
    # ==========================================
    plot_training_loss(epoch_losses, save_path="training_loss.png")
    plot_confusion_matrix(val_preds, val_labels, real_names, real_indices, save_path="confusion_matrix.png")
    
    print("\n All done! Files saved:")
    print("  - vehicle_classifier.pth  (trained model)")
    print("  - training_loss.png       (loss curve)")
    print("  - confusion_matrix.png    (confusion matrices)")

if __name__ == "__main__":
    main()
