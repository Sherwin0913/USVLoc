from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .types import PairResult


def svd_icp(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    """Solve a 2D rigid transform with SVD.

    Inputs are two matched sets of 2D points, and the output is a 2x3 [R|t]
    matrix. The math follows BEVPlace2-main/RANSAC.py for fair baseline
    reproduction.
    """
    src = np.asarray(src_xy).T
    dst = np.asarray(dst_xy).T
    mean_src = np.mean(src, axis=1, keepdims=True)
    mean_dst = np.mean(dst, axis=1, keepdims=True)
    src_norm = src - mean_src
    dst_norm = dst - mean_dst
    mat_s = src_norm.dot(dst_norm.T)
    u, sigma, v_t = np.linalg.svd(mat_s)
    temp = u.dot(v_t)
    det = np.linalg.det(temp)
    correction = np.asarray([[1, 0], [0, det]])
    rotation = v_t.T.dot(correction).dot(u.T)
    translation = mean_dst.T - mean_src.T.dot(rotation.T)
    return np.hstack((rotation, translation.reshape(-1, 1)))


def bevplace2_rigid_ransac(
    points1: np.ndarray,
    points2: np.ndarray,
    iterations: int = 1000,
    threshold_m: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """2D rigid RANSAC in the BEVPlace++ style.

    Each iteration samples two point pairs to estimate an SE(2) rigid transform,
    counts inliers with the configured threshold, and finally refines with
    SVD-ICP over all inliers. This implementation keeps the BEVPlace++ coordinate
    swapping convention, so callers also keep the corresponding pre-swap.
    """
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    if points1.ndim != 2 or points2.ndim != 2 or points1.shape != points2.shape or points1.shape[1] != 2:
        raise ValueError("RANSAC expects matching arrays with shape (N, 2).")
    if points1.shape[0] < 2:
        raise ValueError("RANSAC needs at least two correspondences.")

    points1 = points1[:, [1, 0]]
    points2 = points2[:, [1, 0]]
    max_consensus = 0
    best_mask = np.zeros((points1.shape[0],), dtype=bool)
    best_mat = np.zeros((2, 3))

    for _ in range(int(iterations)):
        idx1 = np.random.randint(points1.shape[0])
        idx2 = np.random.randint(points1.shape[0])
        _idx3 = np.random.randint(points1.shape[0])
        mat = svd_icp(points1[[idx1, idx2], :], points2[[idx1, idx2], :])
        pred = points1.dot(mat[:2, :2].T) + mat[:2, 2]
        err = np.linalg.norm(pred - points2, axis=1)
        mask = err < float(threshold_m)
        consensus = int(np.sum(mask))
        if consensus > max_consensus:
            max_consensus = consensus
            best_mask = mask
            best_mat = mat

    if int(np.sum(best_mask)) >= 2:
        best_mat = svd_icp(points1[best_mask], points2[best_mask])
    return best_mat.astype(np.float32), best_mask, int(max_consensus)


def _image_tensor_to_gray_u8(image: torch.Tensor) -> np.ndarray:
    img = image.detach().float().cpu()
    if img.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got {tuple(img.shape)}")
    if img.shape[0] == 1:
        gray = img[0]
    else:
        gray = img.mean(dim=0)
    gray = gray.numpy()
    max_value = float(np.nanmax(gray)) if gray.size else 1.0
    if max_value <= 1.5:
        gray = gray * 255.0
    return np.clip(gray, 0.0, 255.0).astype(np.uint8)


def _dense_grid_keypoints(height: int, width: int, target_count: int) -> np.ndarray:
    target_count = max(int(target_count), 4)
    side = int(np.ceil(np.sqrt(target_count)))
    xs = np.linspace(0.0, max(width - 1, 0), side, dtype=np.float32)
    ys = np.linspace(0.0, max(height - 1, 0), side, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    return pts[:target_count].astype(np.float32, copy=False)


def detect_sparse_keypoints(
    image: torch.Tensor,
    max_keypoints: int,
    fast_threshold: int,
    min_keypoints: int,
    dense_fallback: bool = True,
) -> torch.Tensor:
    """Detect FAST keypoints on a BEV image.

    The geometric backend samples local descriptors only at these sparse points
    to avoid matching all pixels.
    """
    gray = _image_tensor_to_gray_u8(image)
    detector = cv2.FastFeatureDetector_create(threshold=int(fast_threshold), nonmaxSuppression=True)
    keypoints = detector.detect(gray, None)
    if len(keypoints) < int(min_keypoints):
        detector = cv2.FastFeatureDetector_create(threshold=max(1, int(fast_threshold) // 2), nonmaxSuppression=True)
        keypoints = detector.detect(gray, None)
    keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
    if int(max_keypoints) > 0:
        keypoints = keypoints[: int(max_keypoints)]
    points = np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
    if (points.ndim != 2 or points.shape[0] < int(min_keypoints)) and bool(dense_fallback):
        target_count = int(max_keypoints) if int(max_keypoints) > 0 else max(int(min_keypoints), 256)
        points = _dense_grid_keypoints(gray.shape[0], gray.shape[1], target_count=max(target_count, int(min_keypoints)))
    if points.ndim != 2:
        points = np.zeros((0, 2), dtype=np.float32)
    return torch.from_numpy(points).to(device=image.device, dtype=torch.float32)


def sample_local_descriptors(
    local_features: torch.Tensor,
    pixel_coords: torch.Tensor,
    mode: str = "pixel_lookup",
) -> torch.Tensor:
    """Sample descriptors from a local feature map at pixel coordinates."""
    if local_features.ndim != 3:
        raise ValueError(f"Expected local features [C,H,W], got {tuple(local_features.shape)}")
    if pixel_coords.ndim != 2 or pixel_coords.shape[1] != 2:
        raise ValueError(f"Expected pixel coords [N,2], got {tuple(pixel_coords.shape)}")
    features = local_features.unsqueeze(0)
    _, _, height, width = features.shape
    if str(mode).lower() in {"pixel_lookup", "integer", "nearest"}:
        xs = torch.clamp(torch.floor(pixel_coords[:, 0]).long(), 0, width - 1)
        ys = torch.clamp(torch.floor(pixel_coords[:, 1]).long(), 0, height - 1)
        return features[0, :, ys, xs].transpose(0, 1).contiguous()

    norm_x = 2.0 * (pixel_coords[:, 0] / max(width - 1, 1)) - 1.0
    norm_y = 2.0 * (pixel_coords[:, 1] / max(height - 1, 1)) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=1).view(1, -1, 1, 2)
    sampled = F.grid_sample(features, grid, mode="bilinear", align_corners=True)
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()


def pixel_to_bevplace2_legacy_coords(
    pixel_coords: torch.Tensor,
    height: int,
    width: int,
    meters_per_pixel: float,
) -> torch.Tensor:
    """Convert image pixel coordinates to legacy BEVPlace++ metric coordinates.

    The image center is treated as the vehicle position, and
    ``meters_per_pixel`` converts pixel offsets to meters.
    """
    center_x = float(width // 2)
    center_y = float(height // 2)
    legacy_x = (center_y - pixel_coords[:, 1]) * float(meters_per_pixel)
    legacy_y = (center_x - pixel_coords[:, 0]) * float(meters_per_pixel)
    return torch.stack([legacy_x, legacy_y], dim=1)


def legacy_translation_to_standard(translation_xy: np.ndarray) -> np.ndarray:
    translation = np.asarray(translation_xy, dtype=np.float32).reshape(2)
    return np.asarray([-translation[1], translation[0]], dtype=np.float32)


@dataclass
class SparseRansacBackend:
    """Sparse local matching plus RANSAC pose-estimation backend.

    Pipeline: FAST keypoints -> local feature sampling -> BF descriptor
    matching -> BEVPlace++-style rigid RANSAC -> SVD-ICP refinement. The output
    ``PairResult`` contains the estimated query-to-candidate relative
    translation and yaw.
    """

    max_keypoints: int = 0
    min_keypoints: int = 0
    fast_threshold: int = 10
    max_correspondences: int = 0
    min_correspondences: int = 2
    min_valid_inliers: int = 0
    ransac_iterations: int = 1000
    ransac_threshold_m: float = 0.5
    descriptor_sampling: str = "pixel_lookup"
    score_mode: str = "inlier_ratio"
    random_seed: int = 1024
    num_threads: int = 1
    dense_fallback: bool = False

    def _match_descriptors(
        self,
        query_desc: torch.Tensor,
        candidate_desc: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Match local descriptors with OpenCV BFMatcher nearest neighbors."""
        q_desc = query_desc.detach().cpu().numpy().astype(np.float32, copy=False)
        c_desc = candidate_desc.detach().cpu().numpy().astype(np.float32, copy=False)
        if q_desc.shape[0] == 0 or c_desc.shape[0] == 0:
            empty = np.asarray([], dtype=np.int64)
            return empty, empty, np.asarray([], dtype=np.float32)
        matcher = cv2.BFMatcher()
        knn_k = 2 if c_desc.shape[0] >= 2 else 1
        matches = matcher.knnMatch(q_desc, c_desc, k=knn_k)
        best = [item[0] for item in matches if len(item) > 0]
        if int(self.max_correspondences) > 0 and len(best) > int(self.max_correspondences):
            best = sorted(best, key=lambda m: m.distance)[: int(self.max_correspondences)]
        query_idx = np.asarray([int(m.queryIdx) for m in best], dtype=np.int64)
        cand_idx = np.asarray([int(m.trainIdx) for m in best], dtype=np.int64)
        distances = np.asarray([float(m.distance) for m in best], dtype=np.float32)
        return query_idx, cand_idx, distances

    def solve_one(
        self,
        query_image: torch.Tensor,
        candidate_image: torch.Tensor,
        query_local: torch.Tensor,
        candidate_local: torch.Tensor,
        meters_per_pixel: float,
    ) -> PairResult:
        """Estimate the relative pose for one query-candidate BEV image pair."""
        if query_local.shape[-2:] != query_image.shape[-2:]:
            query_local = F.interpolate(
                query_local.unsqueeze(0),
                size=query_image.shape[-2:],
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)
        if candidate_local.shape[-2:] != candidate_image.shape[-2:]:
            candidate_local = F.interpolate(
                candidate_local.unsqueeze(0),
                size=candidate_image.shape[-2:],
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)
        query_local = F.normalize(query_local, dim=0)
        candidate_local = F.normalize(candidate_local, dim=0)

        query_points = detect_sparse_keypoints(
            query_image,
            max_keypoints=int(self.max_keypoints),
            fast_threshold=int(self.fast_threshold),
            min_keypoints=int(self.min_keypoints),
            dense_fallback=bool(self.dense_fallback),
        )
        candidate_points = detect_sparse_keypoints(
            candidate_image,
            max_keypoints=int(self.max_keypoints),
            fast_threshold=int(self.fast_threshold),
            min_keypoints=int(self.min_keypoints),
            dense_fallback=bool(self.dense_fallback),
        )
        if query_points.shape[0] < int(self.min_correspondences) or candidate_points.shape[0] < 2:
            return PairResult.empty(
                diagnostics={
                    "query_keypoints": int(query_points.shape[0]),
                    "candidate_keypoints": int(candidate_points.shape[0]),
                }
            )

        query_desc = sample_local_descriptors(query_local, query_points, mode=self.descriptor_sampling)
        candidate_desc = sample_local_descriptors(candidate_local, candidate_points, mode=self.descriptor_sampling)
        # match_q/match_c are indices into the query/candidate keypoint lists.
        match_q, match_c, match_dist = self._match_descriptors(query_desc, candidate_desc)
        num_matches = int(match_q.shape[0])
        if num_matches < max(2, int(self.min_correspondences)):
            return PairResult.empty(
                num_matches=num_matches,
                diagnostics={
                    "query_keypoints": int(query_points.shape[0]),
                    "candidate_keypoints": int(candidate_points.shape[0]),
                },
            )

        height, width = int(query_local.shape[-2]), int(query_local.shape[-1])
        query_coords = pixel_to_bevplace2_legacy_coords(
            query_points[torch.as_tensor(match_q, device=query_points.device)],
            height=height,
            width=width,
            meters_per_pixel=float(meters_per_pixel),
        )
        candidate_coords = pixel_to_bevplace2_legacy_coords(
            candidate_points[torch.as_tensor(match_c, device=candidate_points.device)],
            height=height,
            width=width,
            meters_per_pixel=float(meters_per_pixel),
        )

        try:
            # Pre-swap before calling the BEVPlace++ solver; the solver itself
            # swaps internally, matching SADA's wrapper behavior.
            mat, mask, max_consensus = bevplace2_rigid_ransac(
                query_coords.detach().cpu().numpy()[:, [1, 0]],
                candidate_coords.detach().cpu().numpy()[:, [1, 0]],
                iterations=int(self.ransac_iterations),
                threshold_m=float(self.ransac_threshold_m),
            )
        except Exception as exc:
            return PairResult.empty(
                num_matches=num_matches,
                diagnostics={
                    "query_keypoints": int(query_points.shape[0]),
                    "candidate_keypoints": int(candidate_points.shape[0]),
                    "error": str(exc),
                },
            )

        mask = np.asarray(mask).reshape(-1).astype(bool)
        num_inliers = int(max(int(np.sum(mask)), int(max_consensus)))
        if num_inliers < int(self.min_valid_inliers):
            return PairResult.empty(
                num_matches=num_matches,
                diagnostics={
                    "query_keypoints": int(query_points.shape[0]),
                    "candidate_keypoints": int(candidate_points.shape[0]),
                    "max_consensus": int(max_consensus),
                },
            )

        src = query_coords.detach().cpu().numpy().astype(np.float32, copy=False)
        dst = candidate_coords.detach().cpu().numpy().astype(np.float32, copy=False)
        pred = src.dot(mat[:2, :2].T) + mat[:2, 2]
        residuals = np.linalg.norm(pred - dst, axis=1).astype(np.float32, copy=False)
        inlier_residuals = residuals[mask] if mask.shape[0] == residuals.shape[0] and np.any(mask) else residuals
        score_mode = str(self.score_mode).lower()
        if score_mode in {"num_inliers", "inliers", "count"}:
            score = float(num_inliers)
        else:
            score = float(num_inliers) / float(max(num_matches, 1))

        return PairResult(
            # RANSAC uses BEVPlace++ legacy coordinates internally; convert back to standard x/y here.
            translation_xy=legacy_translation_to_standard(mat[:2, 2]),
            yaw_rad=float(np.arctan2(mat[1, 0], mat[0, 0])),
            score=score,
            pose_valid=True,
            num_inliers=num_inliers,
            num_matches=num_matches,
            inlier_mean_residual_m=float(np.mean(inlier_residuals)) if inlier_residuals.size else float("inf"),
            inlier_median_residual_m=float(np.median(inlier_residuals)) if inlier_residuals.size else float("inf"),
            diagnostics={
                "query_keypoints": int(query_points.shape[0]),
                "candidate_keypoints": int(candidate_points.shape[0]),
                "mean_match_distance": float(np.mean(match_dist)) if match_dist.size else 0.0,
            },
        )

    def solve_batch(
        self,
        query_images: torch.Tensor,
        candidate_images: torch.Tensor,
        query_local: torch.Tensor,
        candidate_local: torch.Tensor,
        meters_per_pixel: float,
    ) -> List[PairResult]:
        batch_size = int(query_images.shape[0])
        query_images = query_images.detach().cpu()
        candidate_images = candidate_images.detach().cpu()
        query_local = query_local.detach().cpu()
        candidate_local = candidate_local.detach().cpu()

        def _solve(index: int) -> PairResult:
            return self.solve_one(
                query_image=query_images[index],
                candidate_image=candidate_images[index],
                query_local=query_local[index],
                candidate_local=candidate_local[index],
                meters_per_pixel=float(meters_per_pixel),
            )

        if int(self.num_threads) <= 1 or batch_size <= 1:
            return [_solve(index) for index in range(batch_size)]
        workers = min(int(self.num_threads), batch_size)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_solve, range(batch_size)))
