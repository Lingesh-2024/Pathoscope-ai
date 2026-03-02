import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# --- Configuration ---
# Since you are moving it yourself, make sure your folder looks like this:
# data/tb_dataset/normal/ (images here)
# data/tb_dataset/tuberculosis/ (images here)
DATA_DIR = 'data/tb_dataset' 
MODEL_SAVE_PATH = 'tb_cnn_v1.pth'
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
IMG_SIZE = 64

# --- Architecture (Must match your ml_logic.py) ---
class SimpleCNN(nn.Module):
    def __init__(self, input_size=64):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Calculation: 64 -> 32 -> 16
        flattened_size = 32 * 16 * 16 
        self.fc1 = nn.Linear(flattened_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)

def train():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Folder {DATA_DIR} not found. Ensure you moved 'normal' and 'tuberculosis' folders there.")
        return

    # Data Augmentation (Good for small datasets)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Loading the dataset
    # IMPORTANT: 
    # Index 0 will be 'normal'
    # Index 1 will be 'tuberculosis'
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Detected Classes: {dataset.class_to_idx}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(input_size=IMG_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()