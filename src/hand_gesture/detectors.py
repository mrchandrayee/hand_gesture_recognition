from __future__ import annotations

import cv2


class HumanDetector:
    """OpenCV HOG people detector; useful as a lightweight baseline."""

    def __init__(self, confidence: float = 0.25):
        self.confidence = confidence
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        rects, weights = self.hog.detectMultiScale(
            frame, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        detections = []
        for (x, y, w, h), weight in zip(rects, weights):
            score = float(weight)
            if score >= self.confidence:
                detections.append((int(x), int(y), int(x + w), int(y + h), score))
        return detections
