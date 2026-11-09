from __future__ import annotations


def _finger_extended(lm, tip: int, pip: int) -> bool:
    # Image y-axis increases downward. This is a simple upright-hand heuristic.
    return lm[tip].y < lm[pip].y


def classify_gesture(landmarks) -> tuple[str, float]:
    """Classify common static gestures from MediaPipe's normalized hand landmarks.

    This heuristic assumes the palm is roughly facing the camera and fingers point upward.
    Rotated hands may be classified as Unknown.
    """
    lm = landmarks.landmark
    index = _finger_extended(lm, 8, 6)
    middle = _finger_extended(lm, 12, 10)
    ring = _finger_extended(lm, 16, 14)
    pinky = _finger_extended(lm, 20, 18)

    # Thumb extension is measured horizontally relative to the index MCP.
    thumb = abs(lm[4].x - lm[2].x) > 0.055

    pattern = (thumb, index, middle, ring, pinky)
    labels = {
        (False, False, False, False, False): "Fist",
        (True, True, True, True, True): "Open_Palm",
        (True, False, False, False, False): "Thumbs_Up",
        (False, True, True, False, False): "Peace",
        (False, True, False, False, False): "One",
        (False, True, True, True, False): "Three",
        (False, True, True, True, True): "Four",
    }
    label = labels.get(pattern, "Unknown")
    return label, 0.80 if label != "Unknown" else 0.35
