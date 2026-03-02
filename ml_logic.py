# ml_logic.py
import streamlit as st
from PIL import Image, ImageDraw
import io
import hashlib
import time
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn.modules.container import Sequential # Added this import for type hinting/clarity

# --- Import ML Libraries (Requires: pip install torch torchvision) ---
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    ML_LIBRARIES_INSTALLED = True
except ImportError:
    ML_LIBRARIES_INSTALLED = False

# --- Define the simple CNN model architecture (MUST match the one in training scripts) ---

# We define the model using nn.Module with explicit layer naming
# to match the *original* state_dict keys. 
# NOTE: If the user re-ran the training script with the nn.Sequential logic I provided previously, 
# this class needs to be the nn.Sequential one. Since the user reverted, 
# we need to ensure the saved file matches THIS class.

class SimpleCNN(nn.Module):
    """
    Defines the SimpleCNN architecture with explicit layer naming.
    Keys will be 'conv1.weight', 'fc1.bias', etc.
    """
    def __init__(self, input_size=64):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate flattened size based on input_size
        # The image size (input_size x input_size) is reduced by two pooling layers (4x reduction)
        flattened_size = 32 * (input_size // 4) * (input_size // 4)
        
        self.fc1 = nn.Linear(flattened_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2) # 2 classes

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# --- ML Model Configuration ---

DISEASE_MODELS = {
    # Malaria fix applied: will now successfully load malaria_cnn_v1.pth
    "Malaria": {"sample": "Blood Smear", "model_file": "malaria_cnn_v1.pth", "target_classes": ["Uninfected", "Parasitized"], "input_size": 64, "architecture": SimpleCNN},

    # Ready for Cancer training: uses the same architecture for simplicity
    "Cancer (Biopsy)": {"sample": "Tissue Biopsy", "model_file": "biopsy_vgg_v3.pth", "target_classes": ["Benign", "Malignant"], "input_size": 64, "architecture": SimpleCNN},

  # UPDATED: Matches your 'normal' (0) and 'tuberculosis' (1) folders
    "Tuberculosis (TB)": {
        "sample": "Sputum Sample", 
        "model_file": "tb_cnn_v1.pth", 
        "target_classes": ["Normal", "Tuberculosis"], 
        "input_size": 64, 
        "architecture": SimpleCNN
    },
    "Pneumonia": {
        "sample": "Chest X-Ray", 
        "model_file": "pneumonia_cnn_v1.pth", 
        "target_classes": ["Normal", "Pneumonia"], 
        "input_size": 64, 
        "architecture": SimpleCNN
    },
    # Inside the DISEASE_MODELS dictionary
    "Kidney Stones": {
        "sample": "CT Scan (Abdomen)", 
        "model_file": "kidney_stone_cnn_v1.pth", 
        "target_classes": ["Normal", "Stone"], # Changed to match typical folder names
        "input_size": 64, 
        "architecture": SimpleCNN
    },
    #--- ADDED SKIN CANCER CONFIGURATION ---
    "Skin Cancer": {
        "sample": "Skin Lesion Photo", 
        "model_file": "skin_cancer_cnn_v1.pth", 
        "target_classes": ["Benign", "Malignant"], 
        "input_size": 64, 
        "architecture": SimpleCNN
    }
}

# In-memory dictionary to store loaded models
LOADED_MODELS = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available


def get_disease_from_sample(sample_type):
    """Maps sample type to the primary disease tested."""
    sample_map = {d['sample']: k for k, d in DISEASE_MODELS.items()}
    return sample_map.get(sample_type, "Unknown Disease")

def load_model_real(disease):
    """
    REAL IMPLEMENTATION: Loads the PyTorch model.
    """
    if not ML_LIBRARIES_INSTALLED:
        return None

    if disease in LOADED_MODELS:
        return LOADED_MODELS[disease]

    model_info = DISEASE_MODELS.get(disease, {})
    model_file = model_info.get("model_file")
    architecture = model_info.get("architecture")
    input_size = model_info.get("input_size", 64)

    if not model_file or not os.path.exists(model_file) or architecture is None:
        # Check 1: Do we have a file and an architecture defined?
        return None

    try:
        # Initialize the model architecture with the correct input size
        model = architecture(input_size=input_size) 

        # Load the PyTorch model weights (state_dict)
        state_dict = torch.load(model_file, map_location=DEVICE)

        # CRITICAL FIX RE-IMPLEMENTATION: Check if the keys are mismatched (nn.Sequential vs nn.Module)
        # If the saved file uses integer keys (from nn.Sequential), rename them to the class keys
        
        # Check if the state_dict uses integer keys (like '0.weight') which indicates it was saved
        # using the SimpleCNN class defined in train_malaria_model.py which might have used nn.Sequential
        if any(k.startswith(('0.', '3.', '7.', '9.')) for k in state_dict.keys()):
            # --- Auto-Conversion Logic for nn.Sequential to nn.Module Keys ---
            st.info("Attempting auto-conversion of Sequential keys to nn.Module keys...")
            
            # This mapping is derived from the SimpleCNN structure you provided previously
            # nn.Sequential keys: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
            # nn.Module keys: conv1, relu1, pool1, conv2, relu2, pool2, fc1, relu3, fc2
            # We only care about the keys that have weights/biases: 0, 3, 7, 9
            KEY_MAP = {
                '0.weight': 'conv1.weight', '0.bias': 'conv1.bias',
                '3.weight': 'conv2.weight', '3.bias': 'conv2.bias',
                '7.weight': 'fc1.weight', '7.bias': 'fc1.bias',
                '9.weight': 'fc2.weight', '9.bias': 'fc2.bias',
            }
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if k in KEY_MAP:
                    new_state_dict[KEY_MAP[k]] = v
                else:
                    # Keep keys that don't need mapping, if any (like batchnorm or other custom layers)
                    new_state_dict[k] = v
            state_dict = new_state_dict
            # -----------------------------------------------------------------

        model.load_state_dict(state_dict)
        model.eval()
        model.to(DEVICE)

        LOADED_MODELS[disease] = model
        st.success(f"PyTorch Model **{model_file}** loaded successfully! Running REAL inference.")
        return model
    except Exception as e:
        # THIS IS THE ERROR YOU WERE SEEING (before the ImportError)
        st.warning(f"Error loading PyTorch model {model_file}. Falling back to simulation. Error: {e}")
        return None

def generate_grad_cam_mock(original_image, is_positive):
    """
    Creates a visual simulation of a Grad-CAM heatmap overlay.
    """
    processed_image = original_image.convert("RGB")
    width, height = processed_image.size
    base = processed_image.convert('RGBA')
    overlay = Image.new('RGBA', base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    if is_positive:
        # Red/Orange Circle (Heatmap focus)
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 3

        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            fill=(255, 0, 0, 100)
        )

        inner_radius = radius // 2
        draw.ellipse(
            (center_x - inner_radius, center_y - inner_radius, center_x + inner_radius, center_y + inner_radius),
            fill=(255, 255, 0, 150)
        )
    else:
        # Negative result: Subtle green/blue glow
        draw.rectangle((0, 0, width, height), fill=(0, 255, 0, 30))

    grad_cam_image = Image.alpha_composite(base, overlay).convert('RGB')

    return grad_cam_image

def get_real_prediction(model, uploaded_image, disease):
    if model is None:
        return get_deterministic_prediction(uploaded_image, disease, forced_simulation=True)

    disease_info = DISEASE_MODELS.get(disease, {})
    input_size = disease_info.get("input_size", 64)

    try:
        # 1. Ensure image is RGB and resized to match your training script exactly
        img = uploaded_image.convert("RGB")
        # inside get_real_prediction...
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            # Updated to 0.5/0.5 for more consistent X-ray performance
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        # 2. Inference
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            # Apply Softmax to get actual probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]

        # 3. Mapping: Folder 0 = Normal, Folder 1 = Tuberculosis
        # We take the higher probability
        conf, class_idx = torch.max(probabilities, 0)
        class_idx = class_idx.item()
        confidence = conf.item()

        # Get the label from your target_classes
        result_status = disease_info['target_classes'][class_idx]
        is_positive = (class_idx == 1) # Index 1 is TB/Positive

        return {
            "disease": disease,
            "result_status": result_status,
            "percentage": confidence * 100,
            "confidence_decimal": confidence,
            "grad_cam_image": generate_grad_cam_mock(uploaded_image, is_positive)
        }
    except Exception as e:
        # This prints the actual error to your Streamit console so you can see why it failed
        st.error(f"⚠️ Real Model Error: {str(e)}")
        return get_deterministic_prediction(uploaded_image, disease, forced_simulation=True)

def get_deterministic_prediction(uploaded_image, sample_type, forced_simulation=False):
    """
    SIMULATION: decides result based on image content (hash) 
    instead of hardcoding it to 'Positive'.
    """
    disease = get_disease_from_sample(sample_type)
    img_byte_arr = io.BytesIO()
    uploaded_image.save(img_byte_arr, format='PNG')
    h = hashlib.sha256(img_byte_arr.getvalue()).hexdigest()
    seed = int(h[:8], 16)
    
    # Use the hash to decide: 50/50 chance for simulation
    is_positive = (seed % 2 == 0)

    if forced_simulation:
        st.error(f"Using Simulation for {disease}. (Model file missing or error).")

    disease_info = DISEASE_MODELS.get(disease, {})
    labels = disease_info.get('target_classes', ['Negative', 'Positive'])
    result_status = labels[1] if is_positive else labels[0]
    
    conf = 0.94 + (seed % 500 / 10000)
    time.sleep(1.0)

    return {
        "disease": disease,
        "result_status": result_status,
        "percentage": conf * 100,
        "confidence_decimal": conf,
        "grad_cam_image": generate_grad_cam_mock(uploaded_image, is_positive)
    }

# --- MISSING FUNCTION ADDED BACK ---
def get_sample_options():
    """Returns the list of unique sample types for the Streamlit selectbox."""
    return sorted(list(set(d['sample'] for d in DISEASE_MODELS.values())))

def display_training_structure():
    """
    Displays where the ML training and dataset preparation would occur.
    """
    with st.expander("🔬 View Model Structure (For Developers/Auditors)"):
        st.subheader("⚙️ Machine Learning Structure (PyTorch)")
        st.markdown("""
            The Pathoscope AI is powered by Convolutional Neural Networks (CNNs) built using **PyTorch**.
            To run REAL inference, you must train the models locally.
        """)

        df = pd.DataFrame.from_dict(DISEASE_MODELS, orient='index').reset_index().rename(columns={'index': 'Disease'})
        df = df.drop(columns=['input_size', 'architecture'])
        # Display the file status
        df['File Status'] = df['model_file'].apply(lambda x: '✅ Found' if os.path.exists(x) else '❌ Missing')

        st.dataframe(df, use_container_width=True)

        st.markdown("""
        **Required Steps for Real Prediction:**
        1. **Malaria:** Run `python train_malaria_model.py` -> **`malaria_cnn_v1.pth`**.
        2. **Cancer:** Run `python train_cancer_model.py` -> **`biopsy_vgg_v3.pth`**.
        3. **TB:** Ensure folders `/normal` and `/tuberculosis` exist -> **`tb_cnn_v1.pth`**.
        4. **Pneumonia:** Ensure folders `/normal` and `/pneumonia` exist -> **`pneumonia_cnn_v1.pth`**.
        5. **Kidney Stones:** Ensure folders `/Normal` and `/Stone` exist -> **`kidney_stone_cnn_v1.pth`**.
        6. **Skin Cancer:** `python train_skin_cancer_model.py` -> `skin_cancer_cnn_v1.pth`
      """)
        st.markdown("---")
