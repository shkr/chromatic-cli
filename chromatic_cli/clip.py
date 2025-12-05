from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import open_clip
import torch
from PIL import Image

logger = logging.getLogger(__name__)

clip_model = None
clip_processor = None
clip_tokenizer = None

# Hardcoded S3 path for CLIP model
CLIP_S3_PATH = "transformer/clip"


def get_clip_cache_dir() -> Path:
    """Get the cache directory for CLIP models."""
    return Path(os.environ["CACHE_DIR"]) / "models" / "private" / "clip"


def download_clip_from_s3() -> Optional[str]:
    """Download CLIP checkpoint from private S3 bucket."""
    from surya.common.s3 import get_private_s3_client, download_file_from_private_s3

    cache_dir = get_clip_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        s3_client, bucket = get_private_s3_client()

        # List objects in the CLIP directory to find .pt files
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=CLIP_S3_PATH + "/")

        if "Contents" not in response:
            logger.warning(f"No files found in s3://{bucket}/{CLIP_S3_PATH}/")
            return None

        # Find all .pt files and download them
        pt_files = []
        for obj in response["Contents"]:
            key = obj["Key"]
            if key.endswith(".pt"):
                filename = os.path.basename(key)
                local_path = cache_dir / filename

                if not local_path.exists():
                    logger.info(f"Downloading {key} from S3...")
                    download_file_from_private_s3(key, str(local_path), s3_client, bucket)

                pt_files.append(str(local_path))

        if not pt_files:
            logger.warning("No .pt checkpoint files found in S3")
            return None

        logger.info(f"Downloaded {len(pt_files)} checkpoint files to {cache_dir}")
        return str(cache_dir)

    except Exception as e:
        logger.error(f"Error downloading CLIP model from S3: {e}")
        return None


def get_latest_clip_checkpoint(checkpoint_dir: str = None) -> Optional[str]:
    """Get the latest checkpoint file from the checkpoints directory.
    
    If checkpoint_dir is not provided, attempts to download from S3.
    """
    # If no checkpoint_dir provided, try to download from S3
    if checkpoint_dir is None:
        # First check if we have cached checkpoints
        cache_dir = get_clip_cache_dir()
        if cache_dir.exists() and any(cache_dir.glob("*.pt")):
            checkpoint_dir = str(cache_dir)
            logger.info(f"Using cached CLIP checkpoints from {checkpoint_dir}")
        else:
            # Download from S3
            logger.info("Downloading CLIP model from private S3...")
            checkpoint_dir = download_clip_from_s3()
            if checkpoint_dir is None:
                return None

    try:
        if not os.path.exists(checkpoint_dir):
            logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist")
            return None

        checkpoint_files = []
        for f in os.listdir(checkpoint_dir):
            if f.startswith("epoch_") and f.endswith(".pt"):
                parts = f.split("_")
                epoch = int(parts[1].split(".")[0])
                batch_idx = 0
                if len(parts) > 2:
                    batch_idx = int(parts[2].split(".")[0])

                checkpoint_files.append(
                    {
                        "file": f,
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "path": os.path.join(checkpoint_dir, f),
                    }
                )

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
    """Load CLIP model and processor.
    
    Attempts to load a custom checkpoint from S3 first, falling back to pretrained model.
    """
    global clip_model, clip_processor, clip_tokenizer

    if clip_model is not None:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading CLIP model...")
    checkpoint_path = get_latest_clip_checkpoint()

    if checkpoint_path and os.path.exists(checkpoint_path):
        # Load custom trained model from checkpoint
        logger.info(f"Loading custom CLIP model from {checkpoint_path}")

        # Create base model architecture
        model, processor, tokenizer = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="metaclip_fullcc",
        )

        # Load custom checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        clip_model = model.to(device)
        clip_model.eval()
        clip_processor = processor
        clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        logger.info(f"Custom CLIP model loaded successfully on {device}")
    
    
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

