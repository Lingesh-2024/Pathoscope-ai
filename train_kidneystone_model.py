import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# --- CONFIGURATION ---
# 1. Create folder: data/kidney_dataset/train
# 2. Put 'Normal' and 'Stone' subfolders inside.
DATA_DIR = 'data/kidneystone_dataset'
MODEL_SAVE_PATH = 'kidney_stone_cnn_v1.pth'
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = 64

class SimpleCNN(nn.Module):
    """
    Matches the SimpleCNN architecture in ml_logic.py.
    """
    def __init__(self, input_size=64):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 64 -> 32 -> 16
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)

def train():
    # Normalization matches your updated ml_logic.py (0.5/0.5)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3), # CT scans are grayscale
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if not os.path.exists(DATA_DIR):
        print(f"Error: Folder '{DATA_DIR}' not found.")
        print("Please ensure your dataset is in data/kidney_dataset/train/")
        return

    try:
        dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"✅ Dataset Loaded! Found categories: {dataset.class_to_idx}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Starting Training on {device}...")
    for epoch in range(EPOCHS):
        model.train()
        loss_val = 0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss_val/len(loader):.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Training Complete! Model saved as: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()