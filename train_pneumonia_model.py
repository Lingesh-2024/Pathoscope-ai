import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# --- CONFIGURATION ---
# Based on your move, ensures the path points to the 'train' folder
# containing 'NORMAL' and 'PNEUMONIA' subdirectories.
DATA_DIR = 'data/pneumonia_dataset' 
MODEL_SAVE_PATH = 'pneumonia_cnn_v1.pth'
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = 64

class SimpleCNN(nn.Module):
    """
    CNN Architecture optimized for X-ray image analysis.
    Reduces 64x64 input down to feature maps for binary classification.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2) # Output: 16 x 32 x 32
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) # Output: 32 x 16 x 16
        
        # Calculation: 32 channels * 16 height * 16 width = 8192
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2) # 2 Classes: Normal, Pneumonia

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)

def train():
    # Image transformations for Chest X-Rays
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # Convert to 3-channel grayscale for compatibility with standard CNN input layers
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory '{DATA_DIR}' not found.")
        print("Please ensure you have moved the dataset to the project folder.")
        return

    # Load dataset using ImageFolder (detects NORMAL and PNEUMONIA folders automatically)
    try:
        dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"✅ Dataset Loaded! Found {len(dataset)} images in categories: {dataset.class_to_idx}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Starting Training on {device}...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(loader):.4f}")

    # Save the trained model weights
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ SUCCESS: Model saved as '{MODEL_SAVE_PATH}'")

if __name__ == "__main__":
    train()