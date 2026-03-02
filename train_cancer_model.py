import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# --- Configuration ---
# CRITICAL: This MUST point to the parent folder containing your 'Train' directory.
# Based on your file structure, this is correct: data/cancer_dataset
DATA_DIR = os.path.join(os.getcwd(), 'data', 'cancer_dataset') 
MODEL_PATH = 'biopsy_vgg_v3.pth'
IMAGE_SIZE = 64 # Images will be resized to 64x64 pixels
BATCH_SIZE = 32
NUM_EPOCHS = 3 # Keep low for quick testing
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# --- Model Definition (MUST match ml_logic.py) ---

class SimpleCNN(nn.Module):
    """
    Defines a simple Convolutional Neural Network (CNN) architecture 
    for 2-class image classification (Benign vs. Malignant).
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Layer 1: Conv -> ReLU -> Pool (Output size: 16 channels, 32x32)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 2: Conv -> ReLU -> Pool (Output size: 32 channels, 16x16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Linear layers: Input size is 32 channels * 16 * 16 = 8192
        self.fc1 = nn.Linear(32 * 16 * 16, 128) 
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2) # Output 2 classes (Benign, Malignant)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten the feature maps
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model():
    print("--- PyTorch Cancer Training Script Started ---")
    print(f"Using device: {DEVICE}")

    # --- Data Transformations ---
    transform = transforms.Compose([
        # Resize all images to the target size (64x64)
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # Convert images to PyTorch Tensors
        transforms.ToTensor(),
        # Normalize with ImageNet standards (common practice for pre-trained CNNs)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Load Training Dataset ---
    # Looks for subfolders (classes) inside 'data/cancer_dataset/Train'
    train_dir = os.path.join(DATA_DIR, 'Train')
    try:
        # ImageFolder expects subdirectories named after classes (e.g., Train/Benign)
        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        if len(train_dataset) == 0:
             raise Exception("Dataset is empty. Check your 'Train' subfolder structure and file paths.")
             
        print(f"Loaded {len(train_dataset)} training images from: {train_dir}")
        print(f"Classes found: {train_dataset.classes}")
        
    except Exception as e:
        print(f"ERROR: Failed to load dataset. Ensure 'Benign' and 'Malignant' folders are inside 'Train'. Error: {e}")
        return

    # --- Initialize Model, Loss, and Optimizer ---
    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 10 == 9:  # Print every 10 batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1:5d}] Loss: {running_loss / 10:.4f}')
                running_loss = 0.0
        
        accuracy = 100 * correct / total
        print(f'\n--- Epoch {epoch + 1} completed. Training Accuracy: {accuracy:.2f}% ---')

    print('Finished Training')

    # --- Save Model ---
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n✅ Model saved successfully as: {MODEL_PATH}")

if __name__ == '__main__':
    train_model()
