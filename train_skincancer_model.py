import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# --- CONFIGURATION ---
# Ensure your dataset is at this path relative to the script
DATA_DIR = 'data/skincancer_dataset' 
MODEL_SAVE_PATH = 'skin_cancer_cnn_v1.pth'
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = 64

class SimpleCNN(nn.Module):
    """
    Standard SimpleCNN architecture for Pathoscope AI.
    Processes 64x64 RGB images.
    """
    def __init__(self, input_size=64):
        super(SimpleCNN, self).__init__()
        # Layer 1: 3 channels (RGB) -> 16 filters
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2) # Size: 32x32

        # Layer 2: 16 -> 32 filters
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) # Size: 16x16

        # Linear layers: 32 channels * 16 * 16 = 8192 features
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2) # 2 Classes: Benign, Malignant

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)

def train():
    # Preprocessing for Clinical Photos (Skin)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(), # Augmentation for better skin lesion detection
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if not os.path.exists(DATA_DIR):
        print(f"Error: Folder '{DATA_DIR}' not found.")
        print("Please ensure your dataset is organized in subfolders by class name.")
        return

    # Load dataset from folders
    try:
        dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"Loaded {len(dataset)} images across {len(dataset.classes)} classes: {dataset.classes}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training Skin Cancer Model on {device}...")
    
    for epoch in range(EPOCHS):
        model.train()
        loss_val = 0
        correct = 0
        total = 0
        
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            loss_val += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += lbls.size(0)
            correct += (predicted == lbls).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss_val/len(loader):.4f} | Accuracy: {accuracy:.2f}%")

    # Save weights
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ SUCCESS: Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()