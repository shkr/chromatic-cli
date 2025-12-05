from __future__ import annotations

import logging
import os
from typing import List, Tuple

import open_clip
import torch
from PIL import Image
from typing import Optional

logger = logging.getLogger(__name__)

clip_model = None
clip_processor = None
clip_tokenizer = None

def get_latest_clip_checkpoint(checkpoint_dir: str = None) -> Optional[str]:
    if checkpoint_dir is None:
        checkpoint_dir = str(os.environ['CLIP_CHECKPOINT_DIR'])
    """Get the latest checkpoint file from the checkpoints directory"""
    try:
        if not os.path.exists(checkpoint_dir):
            logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist")
            return None
            
        checkpoint_files = []
        for f in os.listdir(checkpoint_dir):
            if f.startswith('epoch_') and f.endswith('.pt'):
                parts = f.split('_')
                epoch = int(parts[1].split('.')[0])
                batch_idx = 0
                if len(parts) > 2:
                    batch_idx = int(parts[2].split('.')[0])
                
                checkpoint_files.append({
                    "file": f, 
                    "epoch": epoch, 
                    "batch_idx": batch_idx,
                    "path": os.path.join(checkpoint_dir, f)
                })
        
        if not checkpoint_files:
            logger.warning("No checkpoint files found")
            return None
            
        # Sort by epoch and batch_idx
        checkpoint_files.sort(key=lambda x: (x["epoch"], x.get("batch_idx", 0)))
        latest_checkpoint = checkpoint_files[-1]["path"]
        
        logger.info(f"Latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
        
    except Exception as e:
        logger.error(f"Error getting latest checkpoint: {e}")
        return None
    

def load_clip_model():
    """Load CLIP model and processor"""
    global clip_model, clip_processor, clip_tokenizer
    
    if clip_model is not None:
        return
    
    try:
        logger.info("Loading CLIP model...")
        checkpoint_path = get_latest_clip_checkpoint()
        if checkpoint_path and os.path.exists(checkpoint_path):
            # Load custom trained model
            logger.info(f"Loading custom CLIP model from {checkpoint_path}")
            # This would need to be adapted based on your custom training setup
            # For now, we'll use the pretrained model
            pass
        
        # Load pretrained CLIP model
        clip_model, clip_processor, clip_tokenizer = open_clip.create_model_and_transforms(
            'ViT-B-32', 
            pretrained="metaclip_fullcc"
        )
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_model = clip_model.to(device)
        clip_model.eval()
        
        logger.info(f"CLIP model loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading CLIP model: {e}")
        return False
    
    
def get_clip_pipeline():
    global clip_model, clip_processor, clip_tokenizer
    load_clip_model()
    return clip_model, clip_processor, clip_tokenizer

def encode_image(clip_model, clip_processor, image: Image.Image) -> List[float]:
    """Encode an image using CLIP and return the embedding as a list of floats."""
    device = next(clip_model.parameters()).device
    tensor = clip_processor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten().tolist()


def encode_text(clip_model, clip_tokenizer, text: str) -> List[float]:
    """Encode text using CLIP and return the embedding as a list of floats."""
    device = next(clip_model.parameters()).device
    tokens = open_clip.tokenize([text]).to(device)
    with torch.no_grad():
        features = clip_model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten().tolist()

