import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import boto3
import requests
from botocore.config import Config
from tqdm import tqdm

from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()

# Lock file expiration time in seconds (10 minutes)
LOCK_EXPIRATION = 600

# Private S3 prefix for authenticated downloads
S3_PRIVATE_PREFIX = "s3-private://"


def get_private_s3_client():
    """Create a boto3 S3 client using environment variables."""
    bucket = os.environ.get("S3_BUCKET")
    region = os.environ.get("S3_REGION")
    access_key = os.environ.get("S3_ACCESS_KEY_ID")
    secret_key = os.environ.get("S3_SECRET_ACCESS_KEY")

    if not all([bucket, access_key, secret_key]):
        raise ValueError(
            "S3_BUCKET, S3_ACCESS_KEY_ID, and S3_SECRET_ACCESS_KEY environment variables are required for private S3 downloads"
        )

    config = Config(
        region_name=region,
        signature_version="s3v4",
    )

    client = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=config,
    )

    return client, bucket


def download_file_from_private_s3(
    s3_key: str, local_path: str, s3_client=None, bucket: str = None
) -> Path:
    """Download a single file from private S3 bucket."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if s3_client is None or bucket is None:
        s3_client, bucket = get_private_s3_client()

    try:
        logger.debug(f"Downloading s3://{bucket}/{s3_key} to {local_path}")
        s3_client.download_file(bucket, s3_key, str(local_path))
        return local_path
    except Exception as e:
        if local_path.exists():
            local_path.unlink()
        logger.error(f"Download error for s3://{bucket}/{s3_key}: {str(e)}")
        raise


def download_directory_from_private_s3(remote_path: str, local_dir: str):
    """Download a directory from private S3 bucket using manifest.json."""
    model_name = get_model_name(remote_path)
    
    # Check if already downloaded
    model_exists = check_manifest(local_dir)
    if model_exists:
        logger.debug(f"Model {model_name} already exists at {local_dir}")
        return

    s3_client, bucket = get_private_s3_client()

    # Use tempfile.TemporaryDirectory to automatically clean up
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the manifest file
        manifest_key = f"{remote_path}/manifest.json"
        manifest_path = os.path.join(temp_dir, "manifest.json")
        download_file_from_private_s3(manifest_key, manifest_path, s3_client, bucket)

        # List and download all files
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        pbar = tqdm(
            desc=f"Downloading {model_name} model from private S3 to {local_dir}",
            total=len(manifest["files"]),
        )

        with ThreadPoolExecutor(
            max_workers=settings.PARALLEL_DOWNLOAD_WORKERS
        ) as executor:
            futures = []
            for file in manifest["files"]:
                remote_key = f"{remote_path}/{file}"
                local_file = os.path.join(temp_dir, file)
                futures.append(
                    executor.submit(
                        download_file_from_private_s3,
                        remote_key,
                        local_file,
                        s3_client,
                        bucket,
                    )
                )

            for future in futures:
                future.result()
                pbar.update(1)

        pbar.close()

        # Ensure local_dir exists
        os.makedirs(local_dir, exist_ok=True)

        # Move all files to new directory
        for file in os.listdir(temp_dir):
            shutil.move(os.path.join(temp_dir, file), local_dir)


def join_urls(url1: str, url2: str):
    url1 = url1.rstrip("/")
    url2 = url2.lstrip("/")
    return f"{url1}/{url2}"


def get_model_name(pretrained_model_name_or_path: str):
    return pretrained_model_name_or_path.split("/")[0]


def download_file(remote_path: str, local_path: str, chunk_size: int = 1024 * 1024):
    local_path = Path(local_path)
    try:
        response = requests.get(remote_path, stream=True, allow_redirects=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

        return local_path
    except Exception as e:
        if local_path.exists():
            local_path.unlink()
        logger.error(f"Download error for file {remote_path}: {str(e)}")
        raise


def check_manifest(local_dir: str):
    local_dir = Path(local_dir)
    manifest_path = local_dir / "manifest.json"
    if not os.path.exists(manifest_path):
        return False

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        for file in manifest["files"]:
            if not os.path.exists(local_dir / file):
                return False
    except Exception:
        return False

    return True


def download_directory(remote_path: str, local_dir: str):
    model_name = get_model_name(remote_path)
    s3_url = join_urls(settings.S3_BASE_URL, remote_path)
    # Check to see if it's already downloaded
    model_exists = check_manifest(local_dir)
    if model_exists:
        return

    # Use tempfile.TemporaryDirectory to automatically clean up
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the manifest file
        manifest_file = join_urls(s3_url, "manifest.json")
        manifest_path = os.path.join(temp_dir, "manifest.json")
        download_file(manifest_file, manifest_path)

        # List and download all files
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        pbar = tqdm(
            desc=f"Downloading {model_name} model to {local_dir}",
            total=len(manifest["files"]),
        )

        with ThreadPoolExecutor(
            max_workers=settings.PARALLEL_DOWNLOAD_WORKERS
        ) as executor:
            futures = []
            for file in manifest["files"]:
                remote_file = join_urls(s3_url, file)
                local_file = os.path.join(temp_dir, file)
                futures.append(executor.submit(download_file, remote_file, local_file))

            for future in futures:
                future.result()
                pbar.update(1)

        pbar.close()

        # Move all files to new directory
        for file in os.listdir(temp_dir):
            shutil.move(os.path.join(temp_dir, file), local_dir)


class S3DownloaderMixin:
    s3_prefix = "s3://"
    s3_private_prefix = S3_PRIVATE_PREFIX

    @classmethod
    def get_local_path(cls, pretrained_model_name_or_path) -> str:
        if pretrained_model_name_or_path.startswith(cls.s3_private_prefix):
            pretrained_model_name_or_path = pretrained_model_name_or_path.replace(
                cls.s3_private_prefix, ""
            )
            cache_dir = settings.MODEL_CACHE_DIR
            local_path = os.path.join(cache_dir, "private", pretrained_model_name_or_path)
            os.makedirs(local_path, exist_ok=True)
        elif pretrained_model_name_or_path.startswith(cls.s3_prefix):
            pretrained_model_name_or_path = pretrained_model_name_or_path.replace(
                cls.s3_prefix, ""
            )
            cache_dir = settings.MODEL_CACHE_DIR
            local_path = os.path.join(cache_dir, pretrained_model_name_or_path)
            os.makedirs(local_path, exist_ok=True)
        else:
            local_path = ""
        return local_path

    @classmethod
    def _list_s3_keys_for_debug(cls, s3_prefix: str) -> list:
        """List S3 keys under a prefix for debugging purposes."""
        try:
            s3_client, bucket = get_private_s3_client()
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=s3_prefix, MaxKeys=50)
            if "Contents" not in response:
                return []
            return [obj["Key"] for obj in response["Contents"]]
        except Exception as list_err:
            logger.debug(f"Could not list S3 keys for debugging: {list_err}")
            return []

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # Handle private S3 downloads (authenticated via boto3)
        if pretrained_model_name_or_path.startswith(cls.s3_private_prefix):
            local_path = cls.get_local_path(pretrained_model_name_or_path)
            remote_path = pretrained_model_name_or_path.replace(cls.s3_private_prefix, "")

            # Retry logic for downloading the model folder
            retries = 3
            delay = 5
            attempt = 0
            success = False
            while not success and attempt < retries:
                try:
                    download_directory_from_private_s3(remote_path, local_path)
                    success = True
                except Exception as e:
                    logger.error(
                        f"Error downloading model from private S3 {remote_path}. Attempt {attempt + 1} of {retries}. Error: {e}"
                    )
                    # List available keys for debugging on first attempt
                    if attempt == 0:
                        available_keys = cls._list_s3_keys_for_debug(remote_path)
                        if available_keys:
                            logger.error(f"Available keys in {remote_path}/: {available_keys}")
                        else:
                            logger.error(f"No keys found in {remote_path}/ (path may not exist or be empty)")
                    attempt += 1
                    if attempt < retries:
                        logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Failed to download {remote_path} from private S3 after {retries} attempts."
                        )
                        raise e

            return super().from_pretrained(local_path, *args, **kwargs)

        # Handle public S3 downloads (via HTTP)
        if pretrained_model_name_or_path.startswith(cls.s3_prefix):
            local_path = cls.get_local_path(pretrained_model_name_or_path)
            remote_path = pretrained_model_name_or_path.replace(cls.s3_prefix, "")

            # Retry logic for downloading the model folder
            retries = 3
            delay = 5
            attempt = 0
            success = False
            while not success and attempt < retries:
                try:
                    download_directory(remote_path, local_path)
                    success = True
                except Exception as e:
                    logger.error(
                        f"Error downloading model from {remote_path}. Attempt {attempt + 1} of {retries}. Error: {e}"
                    )
                    attempt += 1
                    if attempt < retries:
                        logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Failed to download {remote_path} after {retries} attempts."
                        )
                        raise e

            return super().from_pretrained(local_path, *args, **kwargs)

        # Allow loading models directly from local path or hub
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
