"""
Project: Facial Recognition with Occlusion Robustness (FROB)
Author: Ryan Kamp
Institution: University of Cincinnati
Course: CS 6073 - Deep Learning (Fall 2025)
Last Updated: November 11, 2026

Training functions for the novel occlusion-aware face recognition model.
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .config import *
from .data_utils import OcclusionFaceDataset
from .models import OcclusionAwareFaceModel


def seed_everything(seed):
    """Seed Python, NumPy, and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_files_path(model_save_path):
    """The list of training images saved next to a model's weights."""
    return os.path.splitext(model_save_path)[0] + '_train_files.txt'


def load_train_files(model_save_path):
    """Return the training file names saved with a model, or None if no list was saved."""
    path = train_files_path(model_save_path)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def train_novel_model(device, model_save_path, train_image_paths, seed=0):
    """
    Train or load the novel occlusion-aware face recognition model.

    The caller passes the training images; they must not overlap the images the model is
    evaluated on (frfpv1_main.run_experiment draws them from disjoint pools). The names of the
    images used are saved next to the weights, so later experiments can exclude them.

    Args:
        device: torch device (CPU or CUDA)
        model_save_path: Path to save/load model weights
        train_image_paths: List of training image paths
        seed: Seed for the training subset, triplet sampling, and occlusion sizes

    Returns:
        Trained model
    """
    model = OcclusionAwareFaceModel(embedding_size=EMBEDDING_SIZE).to(device)

    # Load existing model if available
    if os.path.exists(model_save_path):
        print(f"Loading existing model from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        return model

    seed_everything(seed)
    print("Training new model...")
    print(f"Device: {device}")

    # Use a subset for faster training if the pool is large
    train_files = (random.sample(train_image_paths, min(len(train_image_paths), TRAIN_SUBSET_SIZE))
                   if len(train_image_paths) > TRAIN_SUBSET_SIZE
                   else list(train_image_paths))
    print(f"Training on {len(train_files)} images (seed {seed})")

    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    # Setup dataset and dataloader
    dataset = OcclusionFaceDataset(train_files, transform=transform)
    generator = torch.Generator().manual_seed(seed)
    dataloader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, generator=generator)

    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)

    # Training loop
    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        batch_count = 0

        for anchor, positive, negative in dataloader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            # Forward pass
            optimizer.zero_grad()
            a_emb = model(anchor)
            p_emb = model(positive)
            n_emb = model(negative)

            # Compute loss and backward pass
            loss = criterion(a_emb, p_emb, n_emb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        avg_loss = running_loss / batch_count
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

    # Save model and the names of the images it was trained on
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    with open(train_files_path(model_save_path), 'w') as fh:
        fh.write('\n'.join(sorted(os.path.basename(p) for p in train_files)) + '\n')
    print(f"Model saved to {model_save_path}")

    return model
