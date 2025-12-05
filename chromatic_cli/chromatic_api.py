from __future__ import annotations

from typing import Dict

import httpx
from PIL import Image
import io
from .config import get_asset_token

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
}


def _build_cookies() -> Dict[str, str]:
    token = get_asset_token()
    return {"asset-token": token}




DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
}


def _build_cookies() -> Dict[str, str]:
    token = get_asset_token()
    if not token:
        return {}
    return {"asset-token": token}




async def _fetch_image(url: str) -> Image.Image:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            url, headers=DEFAULT_HEADERS, cookies=_build_cookies(), follow_redirects=True
        )
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))


async def get_url_response(url: str) -> httpx.Response:
    """
    Download an image using a browser-like HTTP client.
    """

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            url, headers=DEFAULT_HEADERS, cookies=_build_cookies(), follow_redirects=True
        )
        response.raise_for_status()
        return response

