"""
Project: Facial Recognition with Occlusion Robustness (FROB)
Author: Ryan Kamp
Institution: University of Cincinnati
Course: CS 6073 - Deep Learning (Fall 2025)
Last Updated: December 4, 2025

Evaluation functions for SOTA baseline models and novel model.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
import sys
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

# Optional: DeepFace for SOTA models
try:
    import tensorflow as tf
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("Warning: DeepFace not available. SOTA model evaluation will be disabled.")
    
from .config import *
from .data_utils import (get_custom_model_embeddings_batched, 
                          apply_occlusion, 
                          OcclusionTransform)


def run_evaluation(sample_size, novel_model, device, exp_dirs):
    """
    Run comprehensive evaluation across occlusion levels.
    
    Args:
        sample_size: Number of images in the experiment
        novel_model: Trained novel model
        device: torch device
        exp_dirs: Tuple of (exp_dir, subdirs_dict)
        
    Returns:
        Path to results CSV file
    """
    exp_dir, subdirs = exp_dirs
    results_dir = subdirs['results']
    
    # Get image paths
    image_paths = [os.path.join(subdirs['sample_photos'], f) 
                   for f in sorted(os.listdir(subdirs['sample_photos'])) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_paths:
        print("No images found for evaluation.")
        return None
    
    # Pre-load images
    print("Pre-loading images into memory...")
    loaded_images = [Image.open(p).convert('RGB') for p in image_paths]
    
    # Generate reference embeddings (0% occlusion)
    print("Generating reference embeddings...")
    ref_dbs = generate_reference_embeddings(novel_model, device, image_paths, loaded_images)
    
    # Save reference databases
    with open(os.path.join(subdirs['reference_dbs'], 'reference_dbs.pkl'), 'wb') as f:
        pickle.dump(ref_dbs, f)
    
    # Run evaluation loop across occlusion levels
    print(f"Evaluating over {len(OCCLUSION_STEPS)} occlusion steps...")
    results = []
    
    for occlusion in tqdm(OCCLUSION_STEPS, file=sys.stdout, desc="Occlusion"):
        row = {'Occlusion': occlusion}
        
        # Evaluate custom model
        row['Custom'] = evaluate_custom_model(
            novel_model, device, image_paths, occlusion, ref_dbs['Custom']
        )
        
        # Evaluate SOTA models
        if DEEPFACE_AVAILABLE:
            for model_name in SOTA_MODELS:
                row[model_name], row[f'{model_name} face found'] = evaluate_sota_model(
                    model_name, loaded_images, occlusion, ref_dbs[model_name]
                )
        else:
            for model_name in SOTA_MODELS:
                row[model_name] = 0.0
        
        results.append(row)
    
    # Save and visualize results
    save_results(results, results_dir, sample_size)
    
    return os.path.join(results_dir, 'accuracy_vs_occlusion.csv')


def generate_reference_embeddings(model, device, image_paths, loaded_images):
    """
    Generate reference embeddings for all models at 0% occlusion.
    
    Args:
        model: Novel model instance
        device: torch device
        image_paths: List of image paths
        loaded_images: List of pre-loaded PIL Images
        
    Returns:
        Dictionary of reference databases for each model
    """
    ref_dbs = {m: {'matrix': None, 'indices': []} for m in ALL_MODELS}
    
    # Custom model embeddings (batched)
    base_tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])
    
    c_emb, c_idx = get_custom_model_embeddings_batched(
        model, image_paths, base_tf, device, BATCH_SIZE
    )
    
    if c_emb:
        ref_dbs['Custom']['matrix'] = np.vstack(c_emb)
        ref_dbs['Custom']['indices'] = c_idx
    
    # SOTA model embeddings
    if DEEPFACE_AVAILABLE:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        for model_name in SOTA_MODELS:
            print(f"Generating {model_name} reference embeddings...")
            embs, idxs = [], []
            
            for i, img in enumerate(loaded_images):
                try:
                    img_np = np.array(img)[:, :, ::-1]  # RGB to BGR
                    with tf.device('/CPU:0'):
                        obj = DeepFace.represent(
                            img_path=img_np, 
                            model_name=model_name, 
                            enforce_detection=False
                        )
                    if obj:
                        embs.append(obj[0]['embedding'])
                        idxs.append(i)
                except Exception as e:
                    print(f"{model_name}: gallery image {i} failed to embed: {e}")
            
            if len(embs) < len(loaded_images):
                print(f"{model_name}: {len(loaded_images) - len(embs)} of {len(loaded_images)} gallery "
                      "images failed; their probes can't be matched and count as misses.")
            if embs:
                ref_dbs[model_name]['matrix'] = np.vstack(embs)
                ref_dbs[model_name]['indices'] = idxs
    
    return ref_dbs


def evaluate_custom_model(model, device, image_paths, occlusion, ref_db):
    """
    Evaluate custom model at given occlusion level.
    
    Args:
        model: Novel model instance
        device: torch device
        image_paths: List of image paths
        occlusion: Occlusion ratio
        ref_db: Reference database dict
        
    Returns:
        Accuracy score
    """
    if ref_db['matrix'] is None:
        return 0.0
    
    # Generate probe embeddings with occlusion
    occ_tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        OcclusionTransform(occlusion),
        transforms.ToTensor()
    ])
    
    p_emb, p_idx = get_custom_model_embeddings_batched(
        model, image_paths, occ_tf, device, BATCH_SIZE
    )
    
    if not p_emb:
        return 0.0
    
    # Calculate accuracy
    p_mat = np.vstack(p_emb)
    r_mat = ref_db['matrix']
    r_idxs = np.array(ref_db['indices'])
    
    correct = 0
    for i in range(len(p_mat)):
        dists = np.linalg.norm(r_mat - p_mat[i], axis=1)
        if r_idxs[np.argmin(dists)] == p_idx[i]:
            correct += 1
    
    return correct / len(p_idx) if p_idx else 0.0


def evaluate_sota_model(model_name, loaded_images, occlusion, ref_db):
    """
    Evaluate SOTA model at given occlusion level.

    DeepFace runs its default face detector on every probe. With enforce_detection=False, a
    probe whose face isn't found is embedded from the whole image, while most gallery embeddings
    come from detected face crops, so the share of probes with a detected face is returned too.
    A probe that fails to embed counts as a miss.
    
    Args:
        model_name: Name of SOTA model
        loaded_images: List of pre-loaded PIL Images
        occlusion: Occlusion ratio
        ref_db: Reference database dict
        
    Returns:
        Tuple of (accuracy over all probes, share of probes with a detected face)
    """
    if ref_db['matrix'] is None:
        return 0.0, 0.0
    
    correct, found = 0, 0
    
    for i, img in enumerate(loaded_images):
        try:
            occ = apply_occlusion(img, occlusion)
            img_np = np.array(occ)[:, :, ::-1]  # RGB to BGR
            
            with tf.device('/CPU:0'):
                obj = DeepFace.represent(
                    img_path=img_np, 
                    model_name=model_name, 
                    enforce_detection=False
                )
            
            if obj:
                probe = np.array(obj[0]['embedding'])
                found += obj[0].get('face_confidence', 0) > 0
                dists = np.linalg.norm(ref_db['matrix'] - probe, axis=1)
                if ref_db['indices'][np.argmin(dists)] == i:
                    correct += 1
        except Exception as e:
            print(f"{model_name}: probe {i} failed at occlusion {occlusion:.2f}: {e}")
    
    n = len(loaded_images)
    return (correct / n if n else 0.0), (found / n if n else 0.0)


def save_results(results, results_dir, sample_size):
    """
    Save results to CSV and generate plots.
    
    Args:
        results: List of result dictionaries
        results_dir: Directory to save results
        sample_size: Number of images in experiment
    """
    # Save CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, 'accuracy_vs_occlusion.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Generate combined plot
    plt.figure(figsize=(12, 8))
    styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (1, 1))]
    
    for idx, model in enumerate(ALL_MODELS):
        if model in df.columns:
            plt.plot(
                df['Occlusion'], 
                df[model], 
                label=model, 
                color=COLORS.get(model, 'black'),
                linestyle=styles[idx % len(styles)]
            )
    
    plt.title(f"Accuracy vs Occlusion (N={sample_size})")
    plt.xlabel("Occlusion Ratio")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'accuracy_plot.png'))
    plt.close()
    
    # Generate individual plots
    for model in ALL_MODELS:
        if model in df.columns:
            plt.figure()
            plt.plot(df['Occlusion'], df[model], color=COLORS.get(model, 'black'))
            plt.title(f"{model} Accuracy (N={sample_size})")
            plt.xlabel("Occlusion Ratio")
            plt.ylabel("Accuracy")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(results_dir, f'plot_{model}.png'))
            plt.close()
