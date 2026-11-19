"""
Configuration module containing paths, hyperparameters, and constants.
"""
import os
import numpy as np

# ===== Path Configuration =====
# Auto-detect if running locally or on Google Colab
import sys
if 'google.colab' in sys.modules:
    PROJECT_ROOT = '/content/drive/MyDrive/frfp1'
else:
    # Use the directory where config.py is located as base
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(PROJECT_ROOT, 'celeba_dataset', 'celeba_hq_256')
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')
NOVEL_MODEL_DIR = os.path.join(PROJECT_ROOT, 'novel_model')
DEMO_DIR = os.path.join(PROJECT_ROOT, 'demo')

# ===== Training Hyperparameters =====
BATCH_SIZE = 64          # inference batch size
TRAIN_BATCH_SIZE = 16    # the saved model was trained with batches of 16
TRAIN_SUBSET_SIZE = 1000 # images drawn from the training pool
LEARNING_RATE = 0.0005
EMBEDDING_SIZE = 128
NUM_EPOCHS = 5
TRIPLET_MARGIN = 1.0
SEED = 0

# ===== Evaluation Configuration =====
OCCLUSION_STEPS = np.linspace(0, 1, 21)  # 0% to 100% in 5% increments
SAMPLE_SIZES = [100, 500, 1000, 5000, 10000, 15000]

# ===== Model Names =====
SOTA_MODELS = ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'ArcFace', 'SFace']
ALL_MODELS = ['Custom'] + SOTA_MODELS

# ===== Visualization Configuration =====
COLORS = {
    'Custom': 'red',
    'VGG-Face': 'blue',
    'Facenet': 'green',
    'Facenet512': 'orange',
    'OpenFace': 'purple',
    'ArcFace': 'brown',
    'SFace': 'cyan'
}

# ===== Image Configuration =====
IMAGE_SIZE = (256, 256)
OCCLUSION_TRAINING_RANGE = (0.1, 0.5)  # Random occlusion range for training (see data_utils.OcclusionFaceDataset)
