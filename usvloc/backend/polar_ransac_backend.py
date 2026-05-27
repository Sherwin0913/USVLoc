from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .ransac import bevplace2_rigid_ransac, legacy_translation_to_standard, pixel_to_bevplace2_legacy_coords
from .types import PairResult


TOP_K_RETRIEVAL = 10
TOP_V_VERIFY = 5
RERANK_ALPHA = 1.0
RERANK_BETA = 0.05
FAST_THRESHOLD = 10
MAX_KEYPOINTS = 0
RANSAC_ITERATIONS = 1000
RANSAC_THRESHOLD_M = 0.5
LOOP_INLIER_THRESHOLD = 8
LOC_INLIER_THRESHOLD = 12
TTA_ROTATIONS = (0, 1, 2, 3)


@dataclass(frozen=True)
class PolarCandidate:
    local_index: int
    retrieval_rank: int
    retrieval_dist: float
    theta_rad: float
    peak_ratio: float
    rerank_score: float


@dataclass
class PolarVerifyOutput:
    result: PairResult
    candidate: PolarCandidate | None


def polar_profile_from_features(polar_features: torch.Tensor) -> torch.Tensor:
    """Build the ACC profile from polar features after PolarMixStyle.

    Input shape is [B,C,R,A] or [C,R,A]. The returned shape is [B,C,A] or
    [C,A]. It follows the fixed backend definition: radial mean, then
    channel-wise angular L2 normalization before phase correlation.
    """
    squeeze = False
    if polar_features.ndim == 3:
        polar_features = polar_features.unsqueeze(0)
        squeeze = True
    if polar_features.ndim != 4:
        raise ValueError(f"Expected polar features [B,C,R,A] or [C,R,A], got {tuple(polar_features.shape)}")
    profile = polar_features.float().mean(dim=2)
    profile = F.normalize(profile, dim=-1)
    return profile.squeeze(0) if squeeze else profile


def angular_xcorr_profiles(query_profile: np.ndarray, candidate_profile: np.ndarray) -> tuple[float, float, int]:
    q = np.asarray(query_profile, dtype=np.float32)
    c = np.asarray(candidate_profile, dtype=np.float32)
    if q.ndim != 2 or c.ndim != 2 or q.shape != c.shape:
        raise ValueError(f"ACC expects matching [C,A] profiles, got {q.shape} and {c.shape}")
    angular_bins = int(q.shape[-1])
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1.0e-6)
    c = c / (np.linalg.norm(c, axis=-1, keepdims=True) + 1.0e-6)
    q_fft = np.fft.rfft(q, axis=-1)
    c_fft = np.fft.rfft(c, axis=-1)
    xcorr = np.fft.irfft(q_fft * np.conj(c_fft), n=angular_bins, axis=-1).sum(axis=0)
    bin_idx = int(np.argmax(xcorr))
    bin_shift = bin_idx if bin_idx <= angular_bins // 2 else bin_idx - angular_bins
    theta = float(bin_shift * (2.0 * math.pi / float(angular_bins)))
    peak_ratio = float(np.max(xcorr) / (np.median(xcorr) + 1.0e-6))
    return theta, peak_ratio, bin_shift


def rerank_score(retrieval_dist: float, peak_ratio: float) -> float:
    return RERANK_ALPHA * (1.0 / (1.0 + float(retrieval_dist))) + RERANK_BETA * math.log1p(max(float(peak_ratio), 0.0))


def _image_tensor_to_gray_u8(image: torch.Tensor) -> np.ndarray:
    img = image.detach().float().cpu()
    if img.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got {tuple(img.shape)}")
    gray = img[0] if img.shape[0] == 1 else img.mean(dim=0)
    gray_np = gray.numpy()
    if float(np.nanmax(gray_np)) <= 1.5:
        gray_np = gray_np * 255.0
    return np.clip(gray_np, 0.0, 255.0).astype(np.uint8)


def _dense_grid_keypoints(height: int, width: int, target_count: int) -> np.ndarray:
    side = int(np.ceil(np.sqrt(max(int(target_count), 4))))
    xs = np.linspace(0.0, float(max(width - 1, 0)), side, dtype=np.float32)
    ys = np.linspace(0.0, float(max(height - 1, 0)), side, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)[:target_count].astype(np.float32, copy=False)


def detect_and_describe(image: torch.Tensor, local_features: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    gray = _image_tensor_to_gray_u8(image)
    detector = cv2.FastFeatureDetector_create(threshold=FAST_THRESHOLD, nonmaxSuppression=True)
    keypoints = detector.detect(gray, None)
    if len(keypoints) == 0:
        detector = cv2.FastFeatureDetector_create(threshold=max(5, FAST_THRESHOLD // 4), nonmaxSuppression=True)
        keypoints = detector.detect(gray, None)
    keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
    if MAX_KEYPOINTS > 0:
        keypoints = keypoints[:MAX_KEYPOINTS]
    points = np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
    if points.ndim != 2:
        points = np.zeros((0, 2), dtype=np.float32)

    local = local_features.detach().float().cpu()
    if local.ndim != 3:
        raise ValueError(f"Expected local features [C,H,W], got {tuple(local.shape)}")
    local = F.normalize(local, dim=0)
    _, height, width = local.shape
    xs = np.clip(np.floor(points[:, 0]).astype(np.int64), 0, width - 1)
    ys = np.clip(np.floor(points[:, 1]).astype(np.int64), 0, height - 1)
    desc = local[:, ys, xs].transpose(0, 1).contiguous().numpy().astype(np.float32, copy=False)
    desc = desc / (np.linalg.norm(desc, axis=1, keepdims=True) + 1.0e-6)
    return points.astype(np.float32, copy=False), desc


def match_descriptors(query_desc: np.ndarray, candidate_desc: np.ndarray) -> np.ndarray:
    """BEVPlace2-style single nearest-neighbor BF matching.

    No ratio test and no mutual nearest-neighbor filtering. The downstream
    2-point rigid RANSAC is responsible for rejecting outliers.
    """
    if query_desc.shape[0] == 0 or candidate_desc.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64)
    matcher = cv2.BFMatcher()
    matches = matcher.match(
        np.asarray(query_desc, dtype=np.float32),
        np.asarray(candidate_desc, dtype=np.float32),
    )
    if not matches:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray([[int(m.queryIdx), int(m.trainIdx)] for m in matches], dtype=np.int64)


def full_rigid_ransac(
    query_points_m: np.ndarray,
    candidate_points_m: np.ndarray,
    matches: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if matches.shape[0] < 2:
        return None
    query = np.asarray(query_points_m[matches[:, 0]], dtype=np.float32)
    candidate = np.asarray(candidate_points_m[matches[:, 1]], dtype=np.float32)

    # Match SparseRansacBackend exactly: pre-swap before the BEVPlace2-compatible
    # solver, whose implementation keeps the original BEVPlace2 internal swap.
    mat, mask, _ = bevplace2_rigid_ransac(
        query[:, [1, 0]],
        candidate[:, [1, 0]],
        iterations=RANSAC_ITERATIONS,
        threshold_m=RANSAC_THRESHOLD_M,
        rng=rng,
    )
    mask = np.asarray(mask).reshape(-1).astype(bool)
    if int(mask.sum()) < 2:
        return None

    pred = query[:, [1, 0]].dot(mat[:2, :2].T) + mat[:2, 2]
    residuals = np.linalg.norm(pred - candidate[:, [1, 0]], axis=1).astype(np.float32, copy=False)
    return mat[:2, :2], mat[:2, 2], mask, residuals


class PolarRansacBackend:
    """USVLoc-specific backend: FAISS top-10, polar ACC rerank, top-5 local verification."""

    def __init__(
        self,
        loop_inlier_threshold: int = LOOP_INLIER_THRESHOLD,
        loc_inlier_threshold: int = LOC_INLIER_THRESHOLD,
        random_seed: int = 1024,
    ) -> None:
        self.loop_inlier_threshold = int(loop_inlier_threshold)
        self.loc_inlier_threshold = int(loc_inlier_threshold)
        self.random_seed = int(random_seed)
        self._rng = np.random.default_rng(self.random_seed)
        self.debug_limit = int(os.environ.get("USVLOC_POLAR_DEBUG_LIMIT", "0") or 0)
        self._debug_count = 0

    def rerank(
        self,
        query_profile: np.ndarray,
        candidate_profiles: np.ndarray,
        candidate_indices: Sequence[int],
        retrieval_dists: Sequence[float],
    ) -> list[PolarCandidate]:
        candidates: list[PolarCandidate] = []
        for rank, (local_idx, dist) in enumerate(zip(candidate_indices, retrieval_dists)):
            theta, peak_ratio, _ = angular_xcorr_profiles(query_profile, candidate_profiles[int(local_idx)])
            candidates.append(
                PolarCandidate(
                    local_index=int(local_idx),
                    retrieval_rank=int(rank),
                    retrieval_dist=float(dist),
                    theta_rad=float(theta),
                    peak_ratio=float(peak_ratio),
                    rerank_score=float(rerank_score(float(dist), float(peak_ratio))),
                )
            )
        candidates.sort(key=lambda item: item.rerank_score, reverse=True)
        return candidates[:TOP_V_VERIFY]

    def verify_top_candidates(
        self,
        query_image: torch.Tensor,
        candidate_images: torch.Tensor,
        query_local: torch.Tensor,
        candidate_local: torch.Tensor,
        candidates: Sequence[PolarCandidate],
        meters_per_pixel: float,
    ) -> PolarVerifyOutput:
        if len(candidates) == 0:
            return PolarVerifyOutput(PairResult.empty(), None)
        query_points_px, query_desc = detect_and_describe(query_image, query_local)
        if query_points_px.shape[0] < 2:
            return PolarVerifyOutput(PairResult.empty(diagnostics={"query_keypoints": int(query_points_px.shape[0])}), None)
        height, width = int(query_image.shape[-2]), int(query_image.shape[-1])
        query_points_m = pixel_to_bevplace2_legacy_coords(
            torch.from_numpy(query_points_px),
            height=height,
            width=width,
            meters_per_pixel=float(meters_per_pixel),
        ).numpy()

        best_result = PairResult.empty(diagnostics={"num_verified_candidates": int(len(candidates))})
        best_candidate: PolarCandidate | None = None
        best_debug: dict[str, float | int] = {
            "q_kp_n": int(query_points_px.shape[0]),
            "c_kp_n": 0,
            "matches_n": 0,
            "n_inliers": 0,
            "dyaw_deg": 0.0,
            "retrieval_rank": -1,
            "theta_acc_deg": 0.0,
        }
        for verify_pos, candidate in enumerate(candidates):
            candidate_points_px, candidate_desc = detect_and_describe(candidate_images[verify_pos], candidate_local[verify_pos])
            candidate_points_m = pixel_to_bevplace2_legacy_coords(
                torch.from_numpy(candidate_points_px),
                height=height,
                width=width,
                meters_per_pixel=float(meters_per_pixel),
            ).numpy()
            matches = match_descriptors(query_desc, candidate_desc)
            if matches.shape[0] < 2:
                result = PairResult.empty(
                    num_matches=int(matches.shape[0]),
                    diagnostics={
                        "query_keypoints": int(query_points_px.shape[0]),
                        "candidate_keypoints": int(candidate_points_px.shape[0]),
                        "retrieval_rank": int(candidate.retrieval_rank),
                        "peak_ratio": float(candidate.peak_ratio),
                    },
                )
            else:
                solved = full_rigid_ransac(
                    query_points_m,
                    candidate_points_m,
                    matches,
                    rng=self._rng,
                )
                if solved is None:
                    result = PairResult.empty(
                        num_matches=int(matches.shape[0]),
                        diagnostics={
                            "query_keypoints": int(query_points_px.shape[0]),
                            "candidate_keypoints": int(candidate_points_px.shape[0]),
                            "retrieval_rank": int(candidate.retrieval_rank),
                            "peak_ratio": float(candidate.peak_ratio),
                        },
                    )
                else:
                    rotation, translation_legacy, inlier_mask, residuals = solved
                    num_inliers = int(np.sum(inlier_mask))
                    inlier_residuals = residuals[inlier_mask] if np.any(inlier_mask) else residuals
                    result = PairResult(
                        translation_xy=legacy_translation_to_standard(translation_legacy),
                        yaw_rad=float(math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))),
                        score=float(num_inliers),
                        pose_valid=True,
                        num_inliers=num_inliers,
                        num_matches=int(matches.shape[0]),
                        inlier_mean_residual_m=float(np.mean(inlier_residuals)) if inlier_residuals.size else float("inf"),
                        inlier_median_residual_m=float(np.median(inlier_residuals)) if inlier_residuals.size else float("inf"),
                        diagnostics={
                            "query_keypoints": int(query_points_px.shape[0]),
                            "candidate_keypoints": int(candidate_points_px.shape[0]),
                            "retrieval_rank": int(candidate.retrieval_rank),
                            "retrieval_dist": float(candidate.retrieval_dist),
                            "theta_acc_deg": float(math.degrees(candidate.theta_rad)),
                            "peak_ratio": float(candidate.peak_ratio),
                            "rerank_score": float(candidate.rerank_score),
                        },
                    )
            if result.num_inliers > best_result.num_inliers:
                best_result = result
                best_candidate = candidate
                best_debug = {
                    "q_kp_n": int(query_points_px.shape[0]),
                    "c_kp_n": int(candidate_points_px.shape[0]),
                    "matches_n": int(result.num_matches),
                    "n_inliers": int(result.num_inliers),
                    "dyaw_deg": float(math.degrees(result.yaw_rad)) if result.pose_valid else 0.0,
                    "retrieval_rank": int(candidate.retrieval_rank),
                    "theta_acc_deg": float(math.degrees(candidate.theta_rad)),
                }
        if self._debug_count < self.debug_limit:
            self._debug_count += 1
            q_norm = np.linalg.norm(query_desc, axis=1) if query_desc.size else np.asarray([0.0], dtype=np.float32)
            if query_points_m.size:
                qx_min, qx_max = float(query_points_m[:, 0].min()), float(query_points_m[:, 0].max())
                qy_min, qy_max = float(query_points_m[:, 1].min()), float(query_points_m[:, 1].max())
            else:
                qx_min = qx_max = qy_min = qy_max = 0.0
            print(
                "[debug] "
                f"geom_query_rot_k=0 "
                f"q_kp_n={int(best_debug['q_kp_n'])} "
                f"c_kp_n={int(best_debug['c_kp_n'])} "
                f"matches_n={int(best_debug['matches_n'])} "
                f"q_pts_range_x=[{qx_min:.1f},{qx_max:.1f}] "
                f"q_pts_range_y=[{qy_min:.1f},{qy_max:.1f}] "
                f"feat_norm_mean={float(q_norm.mean()):.3f} "
                f"feat_dim={int(query_desc.shape[1]) if query_desc.ndim == 2 else 0} "
                f"n_inliers={int(best_debug['n_inliers'])} "
                f"dyaw_deg={float(best_debug['dyaw_deg']):.1f} "
                f"best_retrieval_rank={int(best_debug['retrieval_rank'])} "
                f"theta_acc_deg={float(best_debug['theta_acc_deg']):.1f}",
                flush=True,
            )
        return PolarVerifyOutput(best_result, best_candidate)
