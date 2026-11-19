"""
Project: Facial Recognition with Occlusion Robustness (FROB)
Author: Ryan Kamp
Institution: University of Cincinnati
Course: CS 6073 - Deep Learning (Fall 2025)
Last Updated: December 4, 2025

Demonstration scripts for occlusion and background removal capabilities.
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless execution
import matplotlib.pyplot as plt
import os
from PIL import Image
import random

from .config import DEMO_DIR, DATASET_DIR
from .data_utils import apply_occlusion, process_dataset_remove_bg


def run_occlusion_demo(image_path, prefix=""):
    """
    Demonstrate the occlusion method on a given image.
    
    Args:
        image_path: Path to the image to use for the demo
        prefix: Optional prefix for saved filenames (e.g., "nobg_")
    
    Returns:
        PIL Image object of the original image used
    """
    demo_name = "OCCLUSION DEMONSTRATION" if not prefix else f"OCCLUSION DEMONSTRATION (Background Removed)"
    print("\n" + "="*60)
    print(demo_name)
    print("="*60 + "\n")
    
    os.makedirs(DEMO_DIR, exist_ok=True)
    
    original_img = Image.open(image_path).convert('RGB')
    print(f"Using image: {os.path.basename(image_path)}")
    
    # Apply different occlusion levels
    occlusion_levels = [0.0, 0.25, 0.50, 0.75]
    
    fig, axes = plt.subplots(1, len(occlusion_levels), figsize=(16, 4))
    
    for idx, level in enumerate(occlusion_levels):
        if level == 0.0:
            img_to_show = original_img
        else:
            img_to_show = apply_occlusion(original_img.copy(), level)
        
        axes[idx].imshow(img_to_show)
        axes[idx].set_title(f"Occlusion: {int(level*100)}%")
        axes[idx].axis('off')
        
        # Save individual images
        save_path = os.path.join(
            DEMO_DIR, 
            f"{prefix}occlusion_{int(level*100)}percent.png"
        )
        img_to_show.save(save_path)
    
    plt.tight_layout()
    plot_path = os.path.join(DEMO_DIR, f'{prefix}occlusion_demo.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Occlusion demo completed. Results saved to: {DEMO_DIR}")
    return original_img


def run_background_removal_demo(image_path):
    """
    Demonstrate the background removal method on a given image.
    
    Args:
        image_path: Path to the image to use for the demo
    
    Returns:
        Path to the background-removed image
    """
    print("\n" + "="*60)
    print("BACKGROUND REMOVAL DEMONSTRATION")
    print("="*60 + "\n")
    
    os.makedirs(DEMO_DIR, exist_ok=True)
    
    original_img = Image.open(image_path).convert('RGB')
    print(f"Using image: {os.path.basename(image_path)}")
    
    # Process image to remove background
    temp_dir = os.path.join(DEMO_DIR, 'temp_processing')
    os.makedirs(temp_dir, exist_ok=True)
    
    processed_paths = process_dataset_remove_bg([image_path], temp_dir)
    
    if not processed_paths:
        print("Background removal failed.")
        return None
    
    # Load processed image
    processed_img = Image.open(processed_paths[0]).convert('RGB')
    
    # Display comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(processed_img)
    axes[1].set_title("Background Removed")
    axes[1].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(DEMO_DIR, 'background_removal_demo.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save final processed image
    final_path = os.path.join(DEMO_DIR, 'background_removed.png')
    processed_img.save(final_path)
    
    # Cleanup temp directory
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"Background removal demo completed. Results saved to: {DEMO_DIR}")
    return final_path


def run_all_demos():
    """
    Run all demonstration scripts using the same random image.
    
    Workflow:
    1. Select a random image from the dataset
    2. Run occlusion demo on the original image
    3. Remove background from the same image
    4. Run occlusion demo on the background-removed image
    """
    print("\n" + "="*70)
    print("RUNNING ALL DEMONSTRATIONS")
    print("="*70)
    
    # Check if dataset exists
    if not os.path.exists(DATASET_DIR):
        print(f"Dataset directory not found: {DATASET_DIR}")
        print("Skipping all demos.")
        return
    
    # Get list of images
    try:
        all_image_files = [os.path.join(DATASET_DIR, f) 
                          for f in os.listdir(DATASET_DIR) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    except OSError as e:
        print(f"Error accessing dataset directory: {e}")
        print("Skipping all demos.")
        return
    
    if not all_image_files:
        print("No image files found in dataset directory.")
        return
    
    # Select random image (will be used for all demos)
    random_image_path = random.choice(all_image_files)
    print(f"\nSelected image for all demos: {os.path.basename(random_image_path)}")
    
    # Demo 1: Occlusion on original image
    run_occlusion_demo(random_image_path)
    
    # Demo 2: Background removal
    bg_removed_path = run_background_removal_demo(random_image_path)
    
    # Demo 3: Occlusion on background-removed image
    if bg_removed_path:
        run_occlusion_demo(bg_removed_path, prefix="nobg_")
    else:
        print("\nSkipping occlusion demo on background-removed image due to previous error.")
    
    print("\n" + "="*70)
    print("ALL DEMONSTRATIONS COMPLETED")
    print("="*70)
