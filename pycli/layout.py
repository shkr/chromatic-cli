from __future__ import annotations

import asyncio
import base64
import io
import logging
import sys
import torch
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple
import numpy as np
import PIL.Image as Image
from .chromatic_api import _fetch_image

# Add pycli directory to sys.path so surya can be imported as a top-level package
_pycli_dir = str(Path(__file__).resolve().parent)
if _pycli_dir not in sys.path:
    sys.path.insert(0, _pycli_dir)

from surya.layout import LayoutPredictor
from .clip import encode_image, get_clip_pipeline

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if _PROJECT_ROOT.exists():
    project_root_str = str(_PROJECT_ROOT)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

layout_predictor = None


async def download_images(diff_data: Mapping[str, Any]) -> Tuple[Image.Image, Image.Image, Image.Image]:
    base_task = _fetch_image(diff_data["base_capture_image_url"])
    head_task = _fetch_image(diff_data["head_capture_image_url"])
    comparison_task = _fetch_image(diff_data["comparison_capture_image_url"])
    return await asyncio.gather(base_task, head_task, comparison_task)


def get_layout_predictor():
    """Load layout detection model"""
    global layout_predictor
    if layout_predictor is None:
        layout_predictor = LayoutPredictor()
    return layout_predictor


async def run_layout_prediction(
    base_image: Image.Image, head_image: Image.Image
) -> Tuple[Any, Any]:
    predictor = get_layout_predictor()

    def _predict():
        return predictor([base_image, head_image])

    predictions = await asyncio.to_thread(_predict)
    return predictions[0], predictions[1]


def image_to_base64(image: Image.Image) -> str:
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def apply_mask_to_image(image: Image.Image, mask: Image.Image) -> Image.Image:
    if image.size != mask.size:
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)

    if image.mode != "RGBA":
        image = image.convert("RGBA")
    if mask.mode != "RGBA":
        mask = mask.convert("RGBA")

    image_array = np.array(image)
    mask_array = np.array(mask)
    result_array = np.zeros_like(image_array)
    mask_indices = mask_array[:, :, 3] > 0
    result_array[mask_indices] = image_array[mask_indices]
    return Image.fromarray(result_array, "RGBA")


def find_top_k_text(image_features, k, label_embeddings, label_texts):
    """Find top k text labels by L2 distance to image features (matches PostgreSQL pgvector)"""
    # Convert image features to numpy array and normalize
    image_vec = np.array(image_features)
    norm = np.linalg.norm(image_vec)
    if norm > 0:
        image_vec = image_vec / norm
    
    # Compute L2 distances for all labels (matching pgvector's <-> operator)
    items = []
    for text, embedding in zip(label_texts, label_embeddings):
        label_vec = np.array(embedding)
        label_norm = np.linalg.norm(label_vec)
        if label_norm > 0:
            label_vec = label_vec / label_norm
        # L2 distance = sqrt(2 * (1 - cosine_similarity)) for normalized vectors
        cosine_distance = np.linalg.norm(image_vec - label_vec)
        items.append({"text": text, "cosine_distance": float(cosine_distance)})
    
    # Sort by distance and return top k
    items.sort(key=lambda x: x["cosine_distance"])
    return items[:k]


def process_images_to_vectors(
    base_image: Image.Image,
    head_image: Image.Image,
    comparison_image: Image.Image,
    base_pred,
    head_pred,
    clip_model,
    clip_processor,
    label_embeddings: np.array,
    label_texts: List[str]
) -> List[Dict[str, Any]]:
    items = []
    
    # Apply mask to base and head images
    masked_base_image = apply_mask_to_image(base_image, comparison_image)
    masked_head_image = apply_mask_to_image(head_image, comparison_image)
    
    # Preprocess images
    base_image_tensor = clip_processor(base_image).unsqueeze(0)
    device = next(clip_model.parameters()).device
    base_image_tensor = base_image_tensor.to(device)
    head_image_tensor = clip_processor(head_image).unsqueeze(0)
    head_image_tensor = head_image_tensor.to(device)
    
    # Get features
    with torch.no_grad():
        # Process base image
        base_image_features = clip_model.encode_image(base_image_tensor)
        base_image_features = base_image_features / base_image_features.norm(dim=-1, keepdim=True)
        
        # Convert PIL image to PNG bytes for proper base64 encoding
        img_buffer = io.BytesIO()
        base_image.save(img_buffer, format='PNG')
        image_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        base_bbox_info = {
            "label": "BaseComponent",
            "position": 0,
            "confidence": 1.0,
            "image": image_base64,
            "image_src": "BaseImage",
            "polygon": [[0.0, 0.0], [float(base_image.width), float(base_image.height)]],
            "bbox": [0, 0, base_image.width, base_image.height]
        }
        
        embedding = base_image_features.cpu().numpy().flatten().tolist()
        base_bbox_info["embedding"] = embedding
        texts = find_top_k_text(embedding, 3, label_embeddings, label_texts)
        base_bbox_info["texts"] = texts
        items.append(base_bbox_info)
        
        # Process head image
        head_image_features = clip_model.encode_image(head_image_tensor)
        head_image_features = head_image_features / head_image_features.norm(dim=-1, keepdim=True)
        
        # Convert PIL image to PNG bytes for proper base64 encoding
        img_buffer = io.BytesIO()
        head_image.save(img_buffer, format='PNG')
        image_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        head_bbox_info = {
            "label": "HeadComponent",
            "position": 1,
            "confidence": 1.0,
            "image": image_base64,
            "image_src": "HeadImage",
            "polygon": [[0.0, 0.0], [float(head_image.width), float(head_image.height)]],
            "bbox": [0, 0, head_image.width, head_image.height]
        }
        
        embedding = head_image_features.cpu().numpy().flatten().tolist()
        head_bbox_info["embedding"] = embedding
        texts = find_top_k_text(embedding, 3, label_embeddings, label_texts)
        head_bbox_info["texts"] = texts
        items.append(head_bbox_info)
    
    # Add masked base image as an item
    masked_base_image_tensor = clip_processor(masked_base_image).unsqueeze(0)
    masked_base_image_tensor = masked_base_image_tensor.to(device)
    with torch.no_grad():
        masked_base_image_features = clip_model.encode_image(masked_base_image_tensor)
        masked_base_image_features = masked_base_image_features / masked_base_image_features.norm(dim=-1, keepdim=True)
        
        # Convert PIL image to PNG bytes for proper base64 encoding
        img_buffer = io.BytesIO()
        masked_base_image.save(img_buffer, format='PNG')
        masked_base_image_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        masked_base_bbox_info = {
            "label": "MaskedBaseComponent",
            "position": 2,
            "confidence": 1.0,
            "image": masked_base_image_base64,
            "image_src": "MaskedBaseImage",
            "polygon": [[0.0, 0.0], [float(masked_base_image.width), float(masked_base_image.height)]],
            "bbox": [0, 0, masked_base_image.width, masked_base_image.height]
        }
        
        embedding = masked_base_image_features.cpu().numpy().flatten().tolist()
        masked_base_bbox_info["embedding"] = embedding
        texts = find_top_k_text(embedding, 3, label_embeddings, label_texts)
        masked_base_bbox_info["texts"] = texts
        items.append(masked_base_bbox_info)
    
    # Add masked head image as an item
    masked_head_image_tensor = clip_processor(masked_head_image).unsqueeze(0)
    masked_head_image_tensor = masked_head_image_tensor.to(device)
    with torch.no_grad():
        masked_head_image_features = clip_model.encode_image(masked_head_image_tensor)
        masked_head_image_features = masked_head_image_features / masked_head_image_features.norm(dim=-1, keepdim=True)
        
        # Convert PIL image to PNG bytes for proper base64 encoding
        img_buffer = io.BytesIO()
        masked_head_image.save(img_buffer, format='PNG')
        masked_head_image_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        masked_head_bbox_info = {
            "label": "MaskedHeadComponent",
            "position": 3,
            "confidence": 1.0,
            "image": masked_head_image_base64,
            "image_src": "MaskedHeadImage",
            "polygon": [[0.0, 0.0], [float(masked_head_image.width), float(masked_head_image.height)]],
            "bbox": [0, 0, masked_head_image.width, masked_head_image.height]
        }
        
        embedding = masked_head_image_features.cpu().numpy().flatten().tolist()
        masked_head_bbox_info["embedding"] = embedding
        texts = find_top_k_text(embedding, 3, label_embeddings, label_texts)
        masked_head_bbox_info["texts"] = texts
        items.append(masked_head_bbox_info)
    
    # Process base prediction bboxes
    for bbox in base_pred.bboxes:
        bbox_image = base_image.crop(bbox.bbox)
        # Convert PIL image to PNG bytes for proper base64 encoding
        img_buffer = io.BytesIO()
        bbox_image.save(img_buffer, format='PNG')
        bbox_image_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        bbox_info = {
            "label": bbox.label,
            "position": bbox.position + 3,
            "confidence": getattr(bbox, 'confidence', 1.0),
            "polygon": bbox.polygon,
            "image_src": "BaseImage",
            "image": bbox_image_base64,
            "bbox": bbox.bbox
        }
        
        bbox_image_tensor = clip_processor(bbox_image).unsqueeze(0)
        bbox_image_tensor = bbox_image_tensor.to(device)
        with torch.no_grad():
            bbox_image_features = clip_model.encode_image(bbox_image_tensor)
            bbox_image_features = bbox_image_features / bbox_image_features.norm(dim=-1, keepdim=True)
            embedding = bbox_image_features.cpu().numpy().flatten().tolist()
            bbox_info["embedding"] = embedding
            texts = find_top_k_text(embedding, 3, label_embeddings, label_texts)
            bbox_info["texts"] = texts
        items.append(bbox_info)
    
    # Process head prediction bboxes
    for bbox in head_pred.bboxes:
        bbox_image = head_image.crop(bbox.bbox)
        # Convert PIL image to PNG bytes for proper base64 encoding
        img_buffer = io.BytesIO()
        bbox_image.save(img_buffer, format='PNG')
        bbox_image_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        bbox_info = {
            "label": bbox.label,
            "position": bbox.position,
            "polygon": bbox.polygon,
            "image_src": "HeadImage",
            "image": bbox_image_base64,
            "confidence": getattr(bbox, 'confidence', 1.0),
            "bbox": getattr(bbox, 'bbox', None)
        }
        
        bbox_image_tensor = clip_processor(bbox_image).unsqueeze(0)
        bbox_image_tensor = bbox_image_tensor.to(device)
        with torch.no_grad():
            bbox_image_features = clip_model.encode_image(bbox_image_tensor)
            bbox_image_features = bbox_image_features / bbox_image_features.norm(dim=-1, keepdim=True)
            embedding = bbox_image_features.cpu().numpy().flatten().tolist()
            bbox_info["embedding"] = embedding
            texts = find_top_k_text(embedding, 3, label_embeddings, label_texts)
            bbox_info["texts"] = texts
        items.append(bbox_info)
    
    return items


async def process_images_to_vectors_from_diff_data(base_image: Image.Image, 
                                                   head_image: Image.Image, 
                                                   comparison_image: Image.Image, 
                                                   label_embeddings: np.array, 
                                                   label_texts: List[str]) -> List[Dict[str, Any]]:
    base_pred, head_pred = await run_layout_prediction(base_image, head_image)
    clip_model, clip_processor, _ = get_clip_pipeline()
    return process_images_to_vectors(
        base_image, 
        head_image, 
        comparison_image, 
        base_pred, 
        head_pred, 
        clip_model, 
        clip_processor, 
        label_embeddings, 
        label_texts)