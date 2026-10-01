"""Object proposals for close-up capsule-belt video inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CapsuleBox:
    """A detected capsule crop in source-frame coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    area: int

    @property
    def center(self) -> tuple[int, int]:
        """Return the center pixel of the proposal."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


def detect_capsules(
    frame: Any,
    *,
    min_area: int = 15,
    max_area: int = 800,
    min_saturation: int = 30,
) -> list[CapsuleBox]:
    """Detect green capsule caps and expand them to full capsule crops.

    The selected source clip shows green/white gelatin capsules on a light belt.
    The green cap is a stable, interpretable proposal signal. This is intentionally
    a lightweight belt-specific proposal stage; the trained classifier still makes
    the Normal/Anomaly decision on every crop.
    """
    if min_area < 1 or max_area < min_area:
        raise ValueError("min_area and max_area must define a positive range")
    if min_saturation < 0 or min_saturation > 255:
        raise ValueError("min_saturation must be between 0 and 255")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([35, min_saturation, 20], dtype=np.uint8),
        np.array([105, 255, 255], dtype=np.uint8),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
    component_count, _labels, stats, _centers = cv2.connectedComponentsWithStats(mask, 8)

    height, width = frame.shape[:2]
    proposals: list[CapsuleBox] = []
    for index in range(1, component_count):
        x, y, box_width, box_height, area = (int(value) for value in stats[index])
        if area < min_area or area > max_area or box_width < 4 or box_height < 4:
            continue

        center_x = x + box_width // 2
        center_y = y + box_height // 2
        expanded_width = max(24, int(max(box_width * 2.5, box_height * 1.6)))
        expanded_height = max(24, int(max(box_height * 2.5, box_width * 1.6)))
        x1 = max(0, center_x - expanded_width // 2)
        y1 = max(0, center_y - expanded_height // 2)
        x2 = min(width, center_x + expanded_width // 2)
        y2 = min(height, center_y + expanded_height // 2)
        proposals.append(CapsuleBox(x1=x1, y1=y1, x2=x2, y2=y2, area=area))

    return _remove_near_duplicate_centers(proposals)


def _remove_near_duplicate_centers(proposals: list[CapsuleBox]) -> list[CapsuleBox]:
    """Keep one proposal when a blurred cap creates nearby components."""
    ordered = sorted(proposals, key=lambda proposal: proposal.area, reverse=True)
    kept: list[CapsuleBox] = []
    for proposal in ordered:
        center_x, center_y = proposal.center
        too_close = any(
            abs(center_x - other.center[0]) < 10 and abs(center_y - other.center[1]) < 10
            for other in kept
        )
        if not too_close:
            kept.append(proposal)
    return sorted(kept, key=lambda proposal: (proposal.y1, proposal.x1))
