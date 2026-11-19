"""
Project: Facial Recognition with Occlusion Robustness (FROB)
Author: Ryan Kamp
Institution: University of Cincinnati
Course: CS 6073 - Deep Learning (Fall 2025)
Last Updated: December 4, 2025

Quick Start Examples for the FROB Project (frfpv1).

This file contains example usage patterns for the refactored project.
"""

# ============================================================================
# Example 1: Running the Complete Project
# ============================================================================

# Simply execute the main script:
# python frfpv1_main.py

# This will:
# 1. Run occlusion and background removal demonstrations
# 2. Run experiments for sample sizes [100, 500, 1000]
# 3. Generate all results and plots


# ============================================================================
# Example 2: Running Only Demonstrations
# ============================================================================

from frfpv1.demos import run_all_demos, run_occlusion_demo, run_background_removal_demo

# Run all demonstrations
run_all_demos()

# Or run individual demos on one image
run_occlusion_demo("path/to/image.jpg")
run_background_removal_demo("path/to/image.jpg")


# ============================================================================
# Example 3: Running a Single Experiment
# ============================================================================

import torch
from frfpv1.config import *
from frfpv1.data_utils import setup_experiment_dirs, sample_and_backup_photos
from frfpv1.evaluation import run_evaluation
from frfpv1.training import train_novel_model


# Setup
sample_size = 100
exp_dir, subdirs = setup_experiment_dirs(PROJECT_ROOT, sample_size)

# Sample images
image_paths = sample_and_backup_photos(DATASET_DIR, subdirs['sample_photos'], sample_size)

# Train/load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(NOVEL_MODEL_DIR, 'model.pth')
model = train_novel_model(device, model_path, image_paths)

# Evaluate
results_csv = run_evaluation(sample_size, model, device, (exp_dir, subdirs))
print(f"Results saved to: {results_csv}")


# ============================================================================
# Example 4: Using Individual Components
# ============================================================================

from frfpv1.models import OcclusionAwareFaceModel, SpatialAttention
from frfpv1.data_utils import apply_occlusion, OcclusionFaceDataset
from PIL import Image
import torch

# Create model
model = OcclusionAwareFaceModel(embedding_size=128)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Apply occlusion to an image
img = Image.open("path/to/image.jpg")
occluded_img = apply_occlusion(img, coverage_ratio=0.5)  # 50% occlusion
occluded_img.show()

# Create dataset for training
image_paths = ["path/to/img1.jpg", "path/to/img2.jpg", "..."]
dataset = OcclusionFaceDataset(image_paths, transform=None)


# ============================================================================
# Example 5: Customizing Configuration
# ============================================================================

# Edit frfpv1/config.py to customize:

# Training parameters
BATCH_SIZE = 32              # Reduce if GPU memory is limited
LEARNING_RATE = 0.001        # Adjust learning rate
NUM_EPOCHS = 10              # Train for more epochs

# Evaluation parameters
import numpy as np
OCCLUSION_STEPS = np.linspace(0, 1, 11)  # Even faster: 10% increments

# Sample sizes
SAMPLE_SIZES = [50, 100, 200]  # Quick test


# ============================================================================
# Example 6: Running Experiments Programmatically
# ============================================================================

from frfpv1_main import run_experiment

# Run specific experiments
for size in [100, 500]:
    print(f"\nRunning experiment with N={size}")
    exp_dir = run_experiment(size)
    if exp_dir:
        print(f"Success! Results in: {exp_dir}")
    else:
        print(f"Failed for N={size}")


# ============================================================================
# Example 7: Analyzing Results
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt

# Load results
results_csv = "path/to/experiments/exper__size_100__timestamp/results/accuracy_vs_occlusion.csv"
df = pd.read_csv(results_csv)

# Display summary statistics
print(df.describe())

# Plot specific model
plt.figure(figsize=(10, 6))
plt.plot(df['Occlusion'], df['Custom'], label='Custom Model', color='red')
plt.plot(df['Occlusion'], df['VGG-Face'], label='VGG-Face', color='blue')
plt.xlabel('Occlusion Ratio')
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Find accuracy at 50% occlusion
row_50 = df.loc[(df['Occlusion'] - 0.5).abs().idxmin()]
print(f"\nAccuracy at 50% occlusion:")
for model in ['Custom', 'VGG-Face', 'Facenet']:
    print(f"  {model}: {row_50[model]:.2%}")


# ============================================================================
# Example 8: Background Removal Integration
# ============================================================================

from frfpv1.data_utils import process_dataset_remove_bg

# Remove backgrounds from a list of images
source_images = ["img1.jpg", "img2.jpg", "img3.jpg"]
output_dir = "path/to/output"

processed_paths = process_dataset_remove_bg(source_images, output_dir)
print(f"Processed {len(processed_paths)} images")


# ============================================================================
# Example 9: Generating Embeddings
# ============================================================================

import torch
import torchvision.transforms as transforms
from frfpv1.models import OcclusionAwareFaceModel
from frfpv1.data_utils import get_custom_model_embeddings_batched

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OcclusionAwareFaceModel().to(device)
model.load_state_dict(torch.load("path/to/model.pth", map_location=device))
model.eval()

# Setup transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Generate embeddings for multiple images
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
embeddings, indices = get_custom_model_embeddings_batched(
    model, image_paths, transform, device, batch_size=32
)

print(f"Generated {len(embeddings)} embeddings of shape {embeddings[0].shape}")


# ============================================================================
# Example 10: Comparing Two Images
# ============================================================================

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from frfpv1.models import OcclusionAwareFaceModel

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OcclusionAwareFaceModel().to(device)
model.load_state_dict(torch.load("path/to/model.pth", map_location=device))
model.eval()

# Load and preprocess images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

img1 = transform(Image.open("face1.jpg").convert('RGB')).unsqueeze(0).to(device)
img2 = transform(Image.open("face2.jpg").convert('RGB')).unsqueeze(0).to(device)

# Generate embeddings
with torch.no_grad():
    emb1 = model(img1).cpu().numpy()[0]
    emb2 = model(img2).cpu().numpy()[0]

# Calculate similarity (smaller distance = more similar)
distance = np.linalg.norm(emb1 - emb2)
print(f"Distance between faces: {distance:.4f}")

# Threshold example (adjust based on your data)
threshold = 1.0
if distance < threshold:
    print("Same person!")
else:
    print("Different people!")
