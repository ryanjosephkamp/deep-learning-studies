"""
Project: Facial Recognition with Occlusion Robustness (FROB)
Author: Ryan Kamp
Institution: University of Cincinnati
Course: CS 6073 - Deep Learning (Fall 2025)
Last Updated: December 4, 2025

Data processing utilities including dataset classes, occlusion, and background removal.
"""
import datetime
import numpy as np
import os
from PIL import Image
import random
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from .config import OCCLUSION_TRAINING_RANGE

# Optional: rembg for background removal
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("Warning: rembg not available. Background removal functionality will be disabled.")


def apply_occlusion(image, coverage_ratio):
    """
    Apply a centered black square occlusion to an image.
    
    Args:
        image: PIL Image
        coverage_ratio: Float between 0 and 1 representing the proportion of the image to occlude
        
    Returns:
        PIL Image with occlusion applied
    """
    img_arr = np.array(image)
    h, w, _ = img_arr.shape
    side = int(np.sqrt(coverage_ratio * h * w))
    top = max(0, (h - side) // 2)
    left = max(0, (w - side) // 2)
    bottom = min(h, top + side)
    right = min(w, left + side)
    img_arr[top:bottom, left:right, :] = 0
    return Image.fromarray(img_arr)


def process_dataset_remove_bg(source_paths, dest_dir):
    """
    Remove backgrounds from images using rembg.
    
    Args:
        source_paths: List of image file paths
        dest_dir: Directory to save processed images
        
    Returns:
        List of paths to processed images
    """
    if not REMBG_AVAILABLE:
        print("Error: rembg is not installed. Cannot remove backgrounds.")
        return []
    
    os.makedirs(dest_dir, exist_ok=True)
    processed_paths = []
    
    print(f"Removing backgrounds for {len(source_paths)} images...")
    for p in tqdm(source_paths, desc="BG Removal"):
        try:
            filename = os.path.basename(p)
            out_path = os.path.join(dest_dir, filename)
            
            # Skip if already processed
            if os.path.exists(out_path):
                processed_paths.append(out_path)
                continue
            
            # Read and process image
            with open(p, 'rb') as i:
                input_data = i.read()
                output_data = remove(input_data)
            
            # Save temporary PNG
            temp_png_path = out_path + ".png"
            with open(temp_png_path, 'wb') as o:
                o.write(output_data)
            
            # Convert RGBA to RGB with white background
            img = Image.open(temp_png_path).convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, (0, 0), img)
            bg.save(out_path)
            
            os.remove(temp_png_path)
            processed_paths.append(out_path)
            
        except Exception as e:
            print(f"Failed to process {p}: {e}")
    
    return processed_paths


def generate_synthetic_images(directory, count=100):
    """
    Generate synthetic face images as fallback when dataset is unavailable.
    
    Args:
        directory: Directory to save synthetic images
        count: Number of images to generate
    """
    os.makedirs(directory, exist_ok=True)
    print(f"Generating {count} synthetic images...")
    
    for i in tqdm(range(count), desc="Synthetic Generation"):
        # Create random colored image
        img_arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_arr)
        img.save(os.path.join(directory, f"synthetic_{i:05d}.png"))
    
    print(f"Synthetic images saved to {directory}")


def setup_experiment_dirs(base_dir, sample_size):
    """
    Create directory structure for an experiment.
    
    Args:
        base_dir: Base directory (typically PROJECT_ROOT)
        sample_size: Number of images in the experiment
        
    Returns:
        Tuple of (experiment_dir, subdirs_dict)
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_folder_name = f"exper__size_{sample_size}__{timestamp}"
    exp_dir = os.path.join(base_dir, 'experiments', exp_folder_name)
    
    subdirs = {
        'sample_photos': os.path.join(exp_dir, 'sample_photos'),
        'results': os.path.join(exp_dir, 'results'),
        'reference_dbs': os.path.join(exp_dir, 'reference_dbs'),
        'baseline_models': os.path.join(exp_dir, 'baseline_models'),
        'novel_model': os.path.join(exp_dir, 'novel_model')
    }
    
    for path in subdirs.values():
        os.makedirs(path, exist_ok=True)
    
    print(f"Experiment setup complete. Directory: {exp_dir}")
    return exp_dir, subdirs


def list_dataset_images(source_dir):
    """Sorted names of the image files in source_dir (sorted, so a seed picks the same files)."""
    return sorted(f for f in os.listdir(source_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png')))


def sample_and_backup_photos(source_dir, dest_dir, n, exclude=(), seed=None):
    """
    Sample n random images from source_dir and copy to dest_dir.
    
    Args:
        source_dir: Source directory containing images
        dest_dir: Destination directory
        n: Number of images to sample
        exclude: File names that must not be sampled (for example, the model's training images)
        seed: Seed for the draw (None draws from Python's global generator)
        
    Returns:
        List of paths to sampled images in dest_dir
    """
    if not os.path.exists(source_dir):
        print(f"Source directory not found: {source_dir}")
        return []
    
    excluded = set(exclude)
    all_files = [f for f in list_dataset_images(source_dir) if f not in excluded]
    
    if not all_files:
        print("No images found in source directory.")
        return []
    
    rng = random.Random(seed) if seed is not None else random
    sampled_files = rng.sample(all_files, min(n, len(all_files)))
    new_paths = []
    
    print(f"Sampling {len(sampled_files)} images ({len(excluded)} excluded)...")
    for filename in sampled_files:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(dest_dir, filename)
        shutil.copy2(src, dst)
        new_paths.append(dst)
    
    return new_paths


class OcclusionTransform:
    """
    Callable transform to apply occlusion on the fly.
    """
    def __init__(self, ratio):
        self.ratio = ratio
    
    def __call__(self, img):
        return apply_occlusion(img, self.ratio)


class OcclusionFaceDataset(Dataset):
    """
    Dataset for triplet loss training with occluded positive samples.
    
    Args:
        image_paths: List of image file paths
        transform: torchvision transform to apply
    """
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load anchor image
        anchor_path = self.image_paths[idx]
        anchor_img = Image.open(anchor_path).convert('RGB')
        
        # Create positive sample (same image with random occlusion)
        ratio = random.uniform(*OCCLUSION_TRAINING_RANGE)
        positive_img = apply_occlusion(anchor_img, ratio)
        
        # Load negative sample (random different image)
        neg_idx = idx
        while neg_idx == idx and len(self.image_paths) > 1:
            neg_idx = random.randint(0, len(self.image_paths) - 1)
        
        negative_path = self.image_paths[neg_idx]
        negative_img = Image.open(negative_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            anchor_t = self.transform(anchor_img)
            positive_t = self.transform(positive_img)
            negative_t = self.transform(negative_img)
            return anchor_t, positive_t, negative_t
        
        return anchor_img, positive_img, negative_img


class SimpleImageDataset(Dataset):
    """
    Simple dataset for batched inference without occlusion.
    
    Args:
        image_paths: List of image file paths
        transform: torchvision transform to apply
    """
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, idx
        except:
            return torch.zeros((3, 256, 256)), -1


def get_custom_model_embeddings_batched(model, image_paths, transform, device, batch_size=64):
    """
    Generate embeddings for images using the custom model in batches.
    
    Args:
        model: PyTorch model
        image_paths: List of image file paths
        transform: torchvision transform
        device: torch device
        batch_size: Batch size for inference
        
    Returns:
        Tuple of (embeddings_list, indices_list)
    """
    dataset = SimpleImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    embeddings, indices = [], []
    model.eval()
    
    with torch.no_grad():
        for images, idxs in dataloader:
            valid_mask = idxs != -1
            if valid_mask.any():
                valid_imgs = images[valid_mask].to(device)
                out = model(valid_imgs).cpu().numpy()
                embeddings.extend(out)
                indices.extend(idxs[valid_mask].numpy())
    
    return embeddings, indices
