import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time

# --- CONFIGURATION ---
# IMPORTANT: Adjusted the path to match your folder structure: 
# 'data/cell_images/malaria_dataset' is now the base directory containing 'Train'
DATA_DIR = os.path.join(os.getcwd(), 'data', 'cell_images', 'malaria_dataset')
MODEL_SAVE_PATH = 'malaria_cnn_v1.pth'
BATCH_SIZE = 32
NUM_EPOCHS = 3 # Keep low for quick testing
IMAGE_SIZE = 64 # Use a small size for faster training
# ---------------------

def create_simple_cnn():
    """Defines a very simple CNN architecture for quick training."""
    model = nn.Sequential(
        # 3 input channels (RGB), 16 output channels, 3x3 kernel
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        nn.Flatten(),
        # The image size (64x64) is reduced by two pooling layers (4x reduction)
        # 32 channels * (64/4) * (64/4) = 32 * 16 * 16 = 8192 features
        nn.Linear(32 * 16 * 16, 128), 
        nn.ReLU(),
        nn.Linear(128, 2) # 2 classes: Uninfected, Parasitized
    )
    return model

def train_and_save_model():
    """Handles data loading, training loop, and model saving."""
    print("--- PyTorch Training Script Started ---")
    
    # 1. Check Data Availability
    # This now checks for: C:\...\pathoscope\data\cell_images\malaria_dataset\Train
    train_dir = os.path.join(DATA_DIR, 'Train')
    if not os.path.exists(train_dir):
        print(f"ERROR: Dataset not found at {train_dir}.")
        print(f"Expected path: {DATA_DIR} must contain a 'Train' folder.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Preparation and Augmentation
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        # Load the dataset using the corrected base DATA_DIR
        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"Loaded {len(train_dataset)} training images.")
    except Exception as e:
        print(f"ERROR: Failed to load dataset. Check the structure inside {DATA_DIR}. Error: {e}")
        return

    # 3. Model, Loss, and Optimizer
    model = create_simple_cnn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Loss: {epoch_loss:.4f}")

    end_time = time.time()
    print(f"Training complete in {(end_time - start_time):.2f} seconds.")

    # 5. Save the Trained Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ SUCCESS: Model saved to {MODEL_SAVE_PATH}")
    print("You can now run 'streamlit run pathoscope_app.py' to use the real model!")


if __name__ == '__main__':
    train_and_save_model()
