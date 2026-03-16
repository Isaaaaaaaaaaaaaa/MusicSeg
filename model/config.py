from dataclasses import dataclass
from typing import List


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_mels: int = 128
    hop_length: int = 512
    n_fft: int = 2048


@dataclass
class BoundaryConfig:
    window_frames: int = 512
    window_hop: int = 256
    window_offsets: tuple = (0, 64, 128)
    threshold: float = 0.3
    min_distance: int = 48
    smooth_window: int = 3
    prominence: float = 0.0
    max_peaks: int = 0
    edge_margin_sec: float = 2.0
    label_radius: int = 36
    label_sigma: float = 14.0
    beat_snap: bool = False
    beat_snap_max: float = 0.35
    beat_refine_radius_frames: int = 3
    use_hierarchical: bool = False
    coarse_window_frames: int = 256
    coarse_hop: int = 128
    refine_radius: int = 24
    use_songformer_postprocess: bool = False
    local_maxima_filter_size: int = 3
    postprocess_window_past_sec: float = 12.0
    postprocess_window_future_sec: float = 12.0
    postprocess_downsample_factor: int = 3


@dataclass
class TransformerConfig:
    num_layers: int = 4
    nhead: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1


@dataclass
class ClassifierConfig:
    labels: List[str]


def default_labels() -> List[str]:
    return ["intro", "verse", "pre_chorus", "chorus", "bridge", "outro"]


def functional_labels() -> List[str]:
    return [
        "intro",
        "pre_verse",
        "verse",
        "pre_chorus",
        "chorus",
        "post_chorus",
        "bridge",
        "interlude",
        "instrumental",
        "solo",
        "transition",
        "theme",
        "main_theme",
        "secondary_theme",
        "coda",
        "outro",
        "fade_out",
        "head",
    ]
