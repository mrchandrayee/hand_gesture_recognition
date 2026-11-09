from dataclasses import dataclass, field


@dataclass(frozen=True)
class GesturePrediction:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass
class FrameInferenceReport:
    frame_index: int | None = None
    source_name: str | None = None
    width: int = 0
    height: int = 0
    raw_human_count: int = 0
    person_count: int = 0
    raw_hand_count: int = 0
    hand_count: int = 0
    gestures: list[GesturePrediction] = field(default_factory=list)
    elapsed_ms: float = 0.0
