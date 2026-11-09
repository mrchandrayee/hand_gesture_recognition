from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import mediapipe as mp

from src.common.report import FrameInferenceReport, GesturePrediction
from src.hand_gesture.detectors import HumanDetector
from src.hand_gesture.classifier import classify_gesture


@dataclass(frozen=True)
class GesturePipelineConfig:
    mapping_path: str | None = None
    human_conf: float = 0.25
    hand_conf: float = 0.45
    imgsz: int = 640
    iou: float = 0.45
    draw_human: bool = True
    verbose: bool = False


class GesturePipeline:
    def __init__(self, config: GesturePipelineConfig | None = None):
        self.config = config or GesturePipelineConfig()
        self.human_detector = HumanDetector(self.config.human_conf) if self.config.draw_human else None
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=self.config.hand_conf,
            min_tracking_confidence=0.45,
        )
        self.drawer = mp.solutions.drawing_utils

    def process_frame(self, frame, frame_index=None, source_name=None):
        started = time.perf_counter()
        height, width = frame.shape[:2]
        annotated = frame.copy()
        report = FrameInferenceReport(
            frame_index=frame_index,
            source_name=source_name,
            width=width,
            height=height,
        )

        if self.human_detector is not None:
            humans = self.human_detector.detect(frame)
            report.raw_human_count = len(humans)
            report.person_count = len(humans)
            for x1, y1, x2, y2, score in humans:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 170, 0), 2)
                cv2.putText(annotated, f"Human {score:.2f}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 170, 0), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if result.multi_hand_landmarks:
            report.raw_hand_count = len(result.multi_hand_landmarks)
            report.hand_count = len(result.multi_hand_landmarks)
            for hand_landmarks in result.multi_hand_landmarks:
                self.drawer.draw_landmarks(
                    annotated, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                label, confidence = classify_gesture(hand_landmarks)
                xs = [p.x for p in hand_landmarks.landmark]
                ys = [p.y for p in hand_landmarks.landmark]
                x1 = max(0, int(min(xs) * width))
                y1 = max(0, int(min(ys) * height))
                x2 = min(width - 1, int(max(xs) * width))
                y2 = min(height - 1, int(max(ys) * height))
                report.gestures.append(
                    GesturePrediction(label, confidence, (x1, y1, x2, y2))
                )
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(annotated, f"{label} {confidence:.2f}",
                            (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 220, 0), 2)

        report.elapsed_ms = (time.perf_counter() - started) * 1000
        labels = ", ".join(g.label for g in report.gestures) or "none"
        status = (
            f"persons={report.person_count} hands={report.hand_count} "
            f"gestures={labels} time={report.elapsed_ms:.1f}ms"
        )
        cv2.rectangle(annotated, (0, 0), (width, 32), (25, 25, 25), -1)
        cv2.putText(annotated, status[:120], (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
        return report, annotated

    def close(self):
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
