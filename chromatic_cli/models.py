from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(slots=True)
class DatasetRecord:
    appId: str
    buildNumber: int
    baseCaptureImageUrl: str
    headCaptureImageUrl: str
    comparisonCaptureImageUrl: str
    componentName: Optional[str] = ""
    storyName: Optional[str] = ""
    headCaptureImageHtml: Optional[str] = ""
    baseCaptureImageHtml: Optional[str] = ""
    testStatus: Optional[str] = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DiffRecord:
    project_id: str
    build_id: str
    diff_id: str
    component_name: str
    story_name: str
    head_capture_image_url: str
    head_capture_image_html: str
    base_capture_image_url: str
    base_capture_image_html: str
    comparison_capture_image_url: str
    metadata: Dict[str, Any]
    status: Optional[str] = None


@dataclass(slots=True)
class VectorDiffItem:
    label: str
    position: float
    confidence: float
    image_src: str
    image: str
    bbox: Optional[Sequence[int]]
    polygon: Optional[Any]
    embedding: Sequence[float]
    texts: Optional[List[Dict[str, Any]]] = None




