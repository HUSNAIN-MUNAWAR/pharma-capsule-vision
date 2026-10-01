import cv2
import numpy as np
import pytest

from defect_detector.video import detect_capsules


def test_detect_capsules_returns_expanded_green_cap_proposal() -> None:
    frame = np.full((120, 180, 3), 255, dtype=np.uint8)
    cv2.rectangle(frame, (70, 50), (86, 65), (40, 170, 40), -1)

    proposals = detect_capsules(frame, min_area=20, max_area=500)

    assert len(proposals) == 1
    assert proposals[0].x1 < 70 < proposals[0].x2
    assert proposals[0].y1 < 50 < proposals[0].y2


def test_detect_capsules_rejects_invalid_area_range() -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="positive range"):
        detect_capsules(frame, min_area=100, max_area=10)
