"""
Project: Facial Recognition with Occlusion Robustness (FROB)
Author: Ryan Kamp
Institution: University of Cincinnati
Course: CS 6073 - Deep Learning (Fall 2025)
Last Updated: December 4, 2025

Neural network model definitions including custom spatial attention mechanism.
"""
import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """
    Custom Spatial Attention Module for focusing on informative facial regions.
    
    Args:
        in_channels: Number of input channels
        kernel_size: Convolutional kernel size (default: 7)
    """
    def __init__(self, in_channels, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn_map = self.conv(x)
        attn_map = self.sigmoid(attn_map)
        return x * attn_map


class OcclusionAwareFaceModel(nn.Module):
    """
    Novel CNN architecture with spatial attention for occlusion-robust facial recognition.
    
    Architecture:
        - Block 1: Conv + BN + ReLU + MaxPool
        - Spatial Attention Layer (key innovation)
        - Block 2: Conv + BN + ReLU + MaxPool
        - Block 3: Conv + BN + ReLU + MaxPool
        - Block 4: Conv + BN + ReLU + MaxPool
        - Global Average Pooling
        - Fully Connected Layer
    
    Args:
        embedding_size: Dimension of output embedding (default: 128)
        attention: Include the spatial attention layer (False replaces it with an identity,
            for ablations; the weights of the other layers keep their names)
    """
    def __init__(self, embedding_size=128, attention=True):
        super(OcclusionAwareFaceModel, self).__init__()
        
        # Backbone Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Novel Spatial Attention Mechanism
        self.attention = SpatialAttention(in_channels=64) if attention else nn.Identity()
        
        # Backbone Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Backbone Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Backbone Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embedding_size)

    def forward(self, x):
        x = self.block1(x)
        x = self.attention(x)  # Apply spatial attention after first block
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
