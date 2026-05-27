from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

_FAISS_GPU_LOGGED = False
_FAISS_CPU_LOGGED = False
_TORCH_FALLBACK_LOGGED = False


def faiss_gpu_available() -> bool:
    return (
        faiss is not None
        and hasattr(faiss, "StandardGpuResources")
        and hasattr(faiss, "get_num_gpus")
        and faiss.get_num_gpus() > 0
        and torch.cuda.is_available()
    )


def search_topk(
    db_descs: np.ndarray,
    query_descs: np.ndarray,
    topk: int,
    metric: str = "l2",
    use_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Unified Top-K retrieval entry point.

    FAISS is preferred. If FAISS-GPU is unavailable or runs out of memory, the
    code falls back to FAISS-CPU. If FAISS is not installed, chunked torch/numpy
    matrix computation is used. Returns ``scores, indices``; in L2 mode, lower
    scores are more similar.
    """
    global _FAISS_GPU_LOGGED, _FAISS_CPU_LOGGED, _TORCH_FALLBACK_LOGGED
    db_descs = np.asarray(db_descs, dtype=np.float32)
    query_descs = np.asarray(query_descs, dtype=np.float32)
    topk = int(max(1, min(int(topk), int(db_descs.shape[0]))))
    metric = str(metric).lower()
    if faiss is not None:
        cpu_index = faiss.IndexFlatL2(db_descs.shape[1]) if metric == "l2" else faiss.IndexFlatIP(db_descs.shape[1])
        if bool(use_gpu) and faiss_gpu_available():
            try:
                if not _FAISS_GPU_LOGGED:
                    print(
                        f"[Retrieval] using faiss-gpu metric={metric} "
                        f"db={db_descs.shape} query={query_descs.shape} "
                        f"device={torch.cuda.current_device()}",
                        flush=True,
                    )
                    _FAISS_GPU_LOGGED = True
                resources = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(resources, torch.cuda.current_device(), cpu_index)
                index.add(db_descs)
                return index.search(query_descs, topk)
            except RuntimeError as exc:
                message = str(exc).lower()
                if "out of memory" not in message and "alloc fail" not in message:
                    raise
                if not _FAISS_CPU_LOGGED:
                    print(f"[Retrieval] faiss-gpu fallback to faiss-cpu due to: {exc}", flush=True)
                    _FAISS_CPU_LOGGED = True
        if not _FAISS_CPU_LOGGED:
            print(
                f"[Retrieval] using faiss-cpu metric={metric} db={db_descs.shape} query={query_descs.shape} "
                f"use_gpu_requested={bool(use_gpu)} faiss_gpu_available={faiss_gpu_available()}",
                flush=True,
            )
            _FAISS_CPU_LOGGED = True
        cpu_index.add(db_descs)
        return cpu_index.search(query_descs, topk)

    def _chunked_search(prefer_gpu: bool) -> tuple[np.ndarray, np.ndarray]:
        num_queries = int(query_descs.shape[0])
        num_db = int(db_descs.shape[0])
        result_scores = np.empty((num_queries, topk), dtype=np.float32)
        result_indices = np.empty((num_queries, topk), dtype=np.int64)

        use_torch_gpu = bool(prefer_gpu) and torch.cuda.is_available()
        target_bytes = 256 * 1024 * 1024 if use_torch_gpu else 32 * 1024 * 1024
        target_elements = max(1, int(target_bytes // 4))
        chunk_size = max(1, min(num_queries, target_elements // max(1, num_db)))

        if use_torch_gpu:
            device = torch.device("cuda")
            db_tensor = torch.from_numpy(np.ascontiguousarray(db_descs, dtype=np.float32)).to(device=device, non_blocking=True)
            db_transposed = db_tensor.transpose(0, 1).contiguous()
            db_norms = torch.sum(db_tensor * db_tensor, dim=1).unsqueeze(0) if metric == "l2" else None
            with torch.no_grad():
                for start in range(0, num_queries, chunk_size):
                    end = min(start + chunk_size, num_queries)
                    query_tensor = torch.from_numpy(np.ascontiguousarray(query_descs[start:end], dtype=np.float32)).to(
                        device=device,
                        non_blocking=True,
                    )
                    if metric == "l2":
                        query_norms = torch.sum(query_tensor * query_tensor, dim=1, keepdim=True)
                        scores_chunk = query_norms + db_norms - (2.0 * (query_tensor @ db_transposed))
                        scores_chunk.clamp_min_(0.0)
                        top_scores, top_indices = torch.topk(scores_chunk, k=topk, dim=1, largest=False, sorted=True)
                    else:
                        scores_chunk = query_tensor @ db_transposed
                        top_scores, top_indices = torch.topk(scores_chunk, k=topk, dim=1, largest=True, sorted=True)
                    result_scores[start:end] = top_scores.detach().cpu().numpy().astype(np.float32, copy=False)
                    result_indices[start:end] = top_indices.detach().cpu().numpy().astype(np.int64, copy=False)
            return result_scores, result_indices

        db_descs_local = np.ascontiguousarray(db_descs, dtype=np.float32)
        db_norms = np.sum(db_descs_local * db_descs_local, axis=1, dtype=np.float32)[None, :] if metric == "l2" else None
        for start in range(0, num_queries, chunk_size):
            end = min(start + chunk_size, num_queries)
            query_chunk = np.ascontiguousarray(query_descs[start:end], dtype=np.float32)
            if metric == "l2":
                query_norms = np.sum(query_chunk * query_chunk, axis=1, dtype=np.float32)[:, None]
                scores_chunk = query_norms + db_norms - (2.0 * (query_chunk @ db_descs_local.T))
                np.maximum(scores_chunk, 0.0, out=scores_chunk)
                partition = np.argpartition(scores_chunk, kth=topk - 1, axis=1)[:, :topk]
                partition_scores = np.take_along_axis(scores_chunk, partition, axis=1)
                order = np.argsort(partition_scores, axis=1)
            else:
                scores_chunk = query_chunk @ db_descs_local.T
                partition = np.argpartition(-scores_chunk, kth=topk - 1, axis=1)[:, :topk]
                partition_scores = np.take_along_axis(scores_chunk, partition, axis=1)
                order = np.argsort(-partition_scores, axis=1)
            top_indices = np.take_along_axis(partition, order, axis=1)
            top_scores = np.take_along_axis(partition_scores, order, axis=1)
            result_indices[start:end] = top_indices
            result_scores[start:end] = top_scores.astype(np.float32, copy=False)
        return result_scores, result_indices

    if not _TORCH_FALLBACK_LOGGED:
        print(
            f"[Retrieval] faiss unavailable, using torch/numpy fallback metric={metric} "
            f"db={db_descs.shape} query={query_descs.shape} prefer_gpu={bool(use_gpu)}",
            flush=True,
        )
        _TORCH_FALLBACK_LOGGED = True
    return _chunked_search(prefer_gpu=use_gpu)


def compute_recall_at_k(
    db_xy: np.ndarray,
    query_xy: np.ndarray,
    predictions: np.ndarray,
    ks: Sequence[int],
    positive_radius_m: float,
) -> Dict[str, float]:
    """Compute Recall@K.

    Only queries with at least one positive database sample within the radius
    are included in the denominator, matching the place recognition evaluation
    protocol used in the paper.
    """
    ks = sorted({int(k) for k in ks if int(k) > 0})
    hits = {k: 0 for k in ks}
    total_positive = 0
    for query_index, pred in enumerate(predictions):
        distances = np.linalg.norm(db_xy - query_xy[query_index : query_index + 1], axis=1)
        positives = np.where(distances < float(positive_radius_m))[0]
        if positives.size == 0:
            continue
        total_positive += 1
        for k in ks:
            pred_k = pred[: min(int(k), len(pred))]
            if np.intersect1d(pred_k, positives).size > 0:
                hits[k] += 1
    denom = max(total_positive, 1)
    return {f"Recall@{k}": float(hits[k]) / float(denom) for k in ks}


def compute_recall_at_one_percent(
    db_xy: np.ndarray,
    query_xy: np.ndarray,
    predictions: np.ndarray,
    positive_radius_m: float,
) -> float:
    one_percent = max(1, int(round(len(db_xy) / 100.0)))
    hits = 0
    total_positive = 0
    for query_index, pred in enumerate(predictions):
        distances = np.linalg.norm(db_xy - query_xy[query_index : query_index + 1], axis=1)
        positives = np.where(distances < float(positive_radius_m))[0]
        if positives.size == 0:
            continue
        total_positive += 1
        if np.intersect1d(pred[:one_percent], positives).size > 0:
            hits += 1
    return float(hits) / float(max(total_positive, 1))
