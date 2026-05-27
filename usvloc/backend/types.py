from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class FeatureBank:
    sequence_name: str
    sequence_dir: str
    descriptors: np.ndarray
    position: np.ndarray
    xy: np.ndarray
    yaw: np.ndarray
    indices: np.ndarray
    dataset: Any
    tta_descriptors: np.ndarray | None = None


@dataclass
class PairResult:
    translation_xy: np.ndarray
    yaw_rad: float
    score: float
    pose_valid: bool
    num_inliers: int = 0
    num_matches: int = 0
    inlier_mean_residual_m: float = float("inf")
    inlier_median_residual_m: float = float("inf")
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def empty(num_matches: int = 0, diagnostics: Dict[str, Any] | None = None) -> "PairResult":
        return PairResult(
            translation_xy=np.zeros(2, dtype=np.float32),
            yaw_rad=0.0,
            score=0.0,
            pose_valid=False,
            num_inliers=0,
            num_matches=int(num_matches),
            inlier_mean_residual_m=float("inf"),
            inlier_median_residual_m=float("inf"),
            diagnostics={} if diagnostics is None else dict(diagnostics),
        )
