"""
nets.py - the models the study trains and the pretrained ones it compares against.

Trained: the project's OcclusionAwareFaceModel (four convolution blocks, a 7 x 7 spatial
attention layer after the first, a 128-dimensional embedding), with the attention layer on or
off, trained on identities with a CosFace loss or a batch-hard triplet loss.

Pretrained: ONNX face recognizers run through OpenCV's DNN module on the aligned BGR crops
(SFace, which DeepFace also wraps, and InsightFace's ArcFace ResNet-50 trained on WebFace600K).
The smoke test replaces them with fixed random projections.
"""

import os
import importlib.util

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load frfpv1/models.py on its own (it needs only torch), not the frfpv1 package, whose other
# modules import TensorFlow, DeepFace, rembg, and matplotlib.
_spec = importlib.util.spec_from_file_location(
    'frfpv1_models', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frfpv1', 'models.py'))
_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_models)
OcclusionAwareFaceModel = _models.OcclusionAwareFaceModel


def build_model(attention, embedding_size):
    return OcclusionAwareFaceModel(embedding_size=embedding_size, attention=attention)


def to_input(batch_bgr_uint8, device):
    """uint8 BGR (B, H, W, 3) -> float RGB (B, 3, H, W) in [-1, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(batch_bgr_uint8[..., ::-1])).to(device)
    return x.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)


class CosFaceHead(nn.Module):
    """Large-margin cosine loss (Wang et al., 2018): s * (cos(theta_y) - m) for the true class."""

    def __init__(self, n_classes, dim, scale, margin):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.scale, self.margin = scale, margin

    def forward(self, emb, labels):
        cos = F.linear(F.normalize(emb), F.normalize(self.weight))
        logits = self.scale * (cos - self.margin * F.one_hot(labels, cos.shape[1]))
        return F.cross_entropy(logits, labels)


def batch_hard_triplet(emb, labels, margin):
    """Batch-hard triplet loss (Hermans et al., 2017) on L2-normalized embeddings."""
    e = F.normalize(emb)
    d = torch.cdist(e, e)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=e.device)
    hardest_pos = d.masked_fill(~same | eye, 0).max(1).values
    hardest_neg = d.masked_fill(same, float('inf')).min(1).values
    return F.relu(hardest_pos - hardest_neg + margin).mean()


@torch.no_grad()
def embed_torch(model, crops_bgr, device, batch=512, normalize=True):
    model.eval()
    out = []
    for i in range(0, len(crops_bgr), batch):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            e = model(to_input(crops_bgr[i:i + batch], device)).float()
        out.append((F.normalize(e) if normalize else e).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def embed_frames_original(model, frames_rgb_uint8, device, batch=128):
    """The original model's own input: RGB frames scaled to [0, 1], unnormalized embeddings."""
    model.eval()
    out = []
    for i in range(0, len(frames_rgb_uint8), batch):
        x = torch.from_numpy(frames_rgb_uint8[i:i + batch]).to(device).permute(0, 3, 1, 2).float().div_(255)
        out.append(model(x).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def attention_maps(model, crops_bgr, device):
    """The attention layer's sigmoid map (B, h, w) for a batch, or None without attention."""
    if not hasattr(model.attention, 'conv'):
        return None
    model.eval()
    x = to_input(crops_bgr, device)
    a = torch.sigmoid(model.attention.conv(model.block1(x)))
    return a[:, 0].float().cpu().numpy()


class OnnxEmbedder:
    """A face recognizer in ONNX, through OpenCV DNN (CPU), on aligned 112 x 112 BGR crops."""

    def __init__(self, path, spec, threads=0):
        import cv2
        self.cv2 = cv2
        if threads:
            cv2.setNumThreads(threads)
        self.net = cv2.dnn.readNetFromONNX(path)
        self.spec = spec

    def __call__(self, crops_bgr, batch=64):
        s = self.spec
        out = []
        for i in range(0, len(crops_bgr), batch):
            blob = self.cv2.dnn.blobFromImages(list(crops_bgr[i:i + batch]), s['scale'], (112, 112),
                                               (s['mean'],) * 3, swapRB=s['swap_rb'], crop=False)
            self.net.setInput(blob)
            out.append(self.net.forward().reshape(len(blob), -1))
        e = np.concatenate(out).astype(np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True).clip(1e-12)


class MockEmbedder:
    """Smoke-test stand-in for a pretrained model: a fixed random projection of the pixels."""

    def __init__(self, dim, size, seed):
        g = np.random.default_rng(seed)
        self.proj = g.standard_normal((size * size * 3, dim)).astype(np.float32) / np.sqrt(size * size * 3)

    def __call__(self, crops_bgr, batch=None):
        e = (crops_bgr.reshape(len(crops_bgr), -1).astype(np.float32) / 255.0) @ self.proj
        return e / np.linalg.norm(e, axis=1, keepdims=True).clip(1e-12)


class DeepFaceEmbedder:
    """One of DeepFace's models on the aligned crops, with its detector skipped (CPU TensorFlow)."""

    def __init__(self, name):
        from deepface import DeepFace
        self.DeepFace, self.name = DeepFace, name

    def __call__(self, crops_bgr, batch=None):
        out = []
        for crop in crops_bgr:
            r = self.DeepFace.represent(img_path=np.ascontiguousarray(crop), model_name=self.name,
                                        detector_backend='skip', enforce_detection=False, align=False)
            out.append(np.asarray(r[0]['embedding'], dtype=np.float32))
        e = np.stack(out)
        return e / np.linalg.norm(e, axis=1, keepdims=True).clip(1e-12)
