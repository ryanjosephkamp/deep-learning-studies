"""
FRFPV1: Facial Recognition with Occlusion Robustness

A comprehensive facial recognition system with novel spatial attention mechanisms
designed to maintain high accuracy under partial facial occlusions.
"""

__version__ = "1.0.0"
__author__ = "Ryan Kamp"

from .config import *
from .models import SpatialAttention, OcclusionAwareFaceModel
from .data_utils import (
    apply_occlusion,
    process_dataset_remove_bg,
    OcclusionFaceDataset,
    SimpleImageDataset,
    get_custom_model_embeddings_batched
)
from .training import train_novel_model
from .evaluation import run_evaluation
from .demos import run_all_demos

__all__ = [
    # Models
    'SpatialAttention',
    'OcclusionAwareFaceModel',
    
    # Data utilities
    'apply_occlusion',
    'process_dataset_remove_bg',
    'OcclusionFaceDataset',
    'SimpleImageDataset',
    'get_custom_model_embeddings_batched',
    
    # Training and evaluation
    'train_novel_model',
    'run_evaluation',
    
    # Demonstrations
    'run_all_demos',
]
