import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

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
            
        # The columns are: filename, Bus, Motorcycle, car, truck (or similar)
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

def main():
    print(f"Using device: {DEVICE}")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Data
    train_csv = os.path.join(DATASET_PATH, 'train', '_classes.csv')
    train_dir = os.path.join(DATASET_PATH, 'train')
    
    if not os.path.exists(train_csv):
        print(f"Dataset not found at {DATASET_PATH}. Please check the path.")
        return

    train_dataset = VehicleDataset(csv_file=train_csv, root_dir=train_dir, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    num_classes = len(train_dataset.annotations.columns) - 1
    class_names = list(train_dataset.annotations.columns)[1:]
    print(f"Classes: {class_names}")

    # Initialize model, loss, optimizer
    model = CustomCNN(num_classes=num_classes).to(DEVICE)
    # Using BCEWithLogitsLoss for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
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

            running_loss += loss.item()
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print("Training complete!")
    
    # Save the model
    torch.save(model.state_dict(), 'vehicle_classifier.pth')
    print("Model saved to vehicle_classifier.pth")

if __name__ == "__main__":
    main()
