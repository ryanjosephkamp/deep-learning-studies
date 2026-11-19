"""
Project: Facial Recognition with Occlusion Robustness (FROB)
Author: Ryan Kamp
Institution: University of Cincinnati
Course: CS 6073 - Deep Learning (Fall 2025)
Last Updated: November 11, 2026

Main orchestration script for the facial recognition with occlusion robustness project.

This script coordinates the entire workflow:
1. Run demonstrations (occlusion and background removal)
2. Run experiments across different sample sizes
"""
import os
import random
import torch

# Import project modules
from frfpv1.config import *
from frfpv1.data_utils import setup_experiment_dirs, sample_and_backup_photos, list_dataset_images
from frfpv1.demos import run_all_demos
from frfpv1.evaluation import run_evaluation
from frfpv1.training import train_novel_model, load_train_files, seed_everything


def run_experiment(sample_size, seed=SEED):
    """
    Run a complete experiment for a given sample size.
    
    Workflow:
    1. Setup experiment directories
    2. Train the novel model on a seeded draw of training images (or load the saved model)
    3. Sample evaluation images that exclude the model's training images
    4. Run evaluation across occlusion levels
    
    Args:
        sample_size: Number of images to use in the experiment
        seed: Seed for the training draw, the evaluation sample, and training
        
    Returns:
        Path to experiment directory, or None if failed
    """
    print(f"\n{'='*70}")
    print(f"STARTING EXPERIMENT: SAMPLE SIZE = {sample_size}")
    print(f"{'='*70}\n")
    
    # Check if dataset exists
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset not found at {DATASET_DIR}")
        print("Please ensure the CelebA-HQ dataset is available.")
        return None
    
    seed_everything(seed)
    
    # Setup directories
    exp_dir, subdirs = setup_experiment_dirs(PROJECT_ROOT, sample_size)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train or load model. A new model trains on a seeded draw of TRAIN_SUBSET_SIZE images,
    # and their names are saved with the weights so evaluation can exclude them.
    master_model_path = os.path.join(NOVEL_MODEL_DIR, 'model.pth')
    os.makedirs(os.path.dirname(master_model_path), exist_ok=True)
    
    train_names = load_train_files(master_model_path)
    if not os.path.exists(master_model_path):
        pool = list_dataset_images(DATASET_DIR)
        random.Random(seed).shuffle(pool)
        train_names = sorted(pool[:TRAIN_SUBSET_SIZE])
    elif train_names is None:
        print("Warning: the saved model has no list of its training images, so the evaluation "
              "sample may overlap them.")
    
    novel_model = train_novel_model(
        device, master_model_path,
        [os.path.join(DATASET_DIR, f) for f in (train_names or [])], seed=seed
    )
    
    if novel_model is None:
        print("Error: Model training/loading failed. Aborting experiment.")
        return None
    
    # Sample evaluation images, excluding the training images
    image_paths = sample_and_backup_photos(
        DATASET_DIR, 
        subdirs['sample_photos'], 
        sample_size,
        exclude=train_names or (),
        seed=seed + sample_size
    )
    
    if not image_paths:
        print("Error: No images sampled. Aborting experiment.")
        return None
    
    # Save model copy for this experiment
    torch.save(
        novel_model.state_dict(), 
        os.path.join(subdirs['novel_model'], 'model.pth')
    )
    
    # Run evaluation
    result_csv = run_evaluation(sample_size, novel_model, device, (exp_dir, subdirs))
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETED: N={sample_size}")
    print(f"Results: {result_csv}")
    print(f"{'='*70}\n")
    
    return exp_dir


def main():
    """
    Main function to run the complete project workflow.
    """
    print("\n" + "="*70)
    print("FACIAL RECOGNITION WITH OCCLUSION ROBUSTNESS")
    print("="*70 + "\n")
    
    # Step 1: Run demonstrations
    print("STEP 1: Running Demonstrations")
    print("-" * 70)
    run_all_demos()
    
    # Step 2: Run experiments
    print("\n\nSTEP 2: Running Experiments")
    print("-" * 70)
    
    # You can modify which sample sizes to run
    sample_sizes_to_run = [100, 500, 1000]  # Subset for faster execution
    # For full experiments, use: sample_sizes_to_run = SAMPLE_SIZES
    
    results_summary = []
    
    for size in sample_sizes_to_run:
        exp_dir = run_experiment(size)
        if exp_dir:
            results_summary.append((size, exp_dir))
        else:
            print(f"Warning: Experiment for N={size} failed.\n")
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    if results_summary:
        print("\nSuccessfully completed experiments:")
        for size, path in results_summary:
            print(f"  - N={size}: {path}")
    else:
        print("\nNo experiments completed successfully.")
    
    print("\n" + "="*70)
    print("PROJECT EXECUTION COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
