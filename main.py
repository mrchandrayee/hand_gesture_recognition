from __future__ import annotations

import argparse
import cv2

from src.hand_gesture.pipeline import GesturePipeline, GesturePipelineConfig


def parse_source(value: str):
    # Numeric source values select a camera; all other values are treated as paths/URLs.
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Hand Gesture Recognition")
    parser.add_argument("--source", default="0", help="Camera index or video path/URL (default: 0)")
    parser.add_argument("--confidence", type=float, default=0.45, help="Minimum hand detection confidence")
    parser.add_argument("--no-human", action="store_true", help="Disable HOG human detection")
    parser.add_argument("--imgsz", type=int, default=640, help="Reserved input-size setting")
    args = parser.parse_args()

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")

    config = GesturePipelineConfig(
        hand_conf=args.confidence,
        draw_human=not args.no_human,
        imgsz=args.imgsz,
    )

    with GesturePipeline(config) as pipeline:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            report, annotated = pipeline.process_frame(frame, frame_index=frame_index, source_name=str(source))
            cv2.imshow("Hand Gesture Recognition", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                out = f"snapshot_{frame_index:06d}.jpg"
                cv2.imwrite(out, annotated)
                print(f"Saved {out}")

            frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
