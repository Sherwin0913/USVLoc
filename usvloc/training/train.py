from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from ..data import SBEVLocFrameTripletDataset, resolve_sequence_dir, sbevloc_collate_fn
from ..data.common import ProcessedSequenceDataset
from ..evaluation import build_shared_frontend_cache, evaluate_place_all, evaluate_usvinland_place
from ..io import ensure_dir, save_json, save_tsv, timestamp_string
from ..losses import lazy_triplet_loss
from ..models import USVLoc


def _configure_runtime_threads(torch_num_threads: int, torch_num_interop_threads: int, cv2_num_threads: int) -> None:
    if int(cv2_num_threads) > 0:
        try:
            cv2.setNumThreads(int(cv2_num_threads))
        except Exception:
            pass
    if int(torch_num_threads) > 0:
        try:
            torch.set_num_threads(int(torch_num_threads))
        except Exception:
            pass
    if int(torch_num_interop_threads) > 0:
        try:
            torch.set_num_interop_threads(int(torch_num_interop_threads))
        except RuntimeError:
            pass


def _build_worker_init_fn(train_cfg: Dict):
    worker_torch_num_threads = int(train_cfg.get("worker_torch_num_threads", 1))
    worker_torch_num_interop_threads = int(train_cfg.get("worker_torch_num_interop_threads", 1))
    worker_cv2_num_threads = int(train_cfg.get("worker_cv2_num_threads", 1))

    def _init_worker(_: int) -> None:
        _configure_runtime_threads(
            torch_num_threads=worker_torch_num_threads,
            torch_num_interop_threads=worker_torch_num_interop_threads,
            cv2_num_threads=worker_cv2_num_threads,
        )
        worker_seed = int(torch.initial_seed() % (2**32))
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init_worker


def _seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _build_run_dir(cfg: Dict, explicit_output_dir: str | Path | None = None) -> Path:
    if explicit_output_dir is not None:
        return ensure_dir(explicit_output_dir)
    outputs_root = ensure_dir(cfg["paths"]["outputs_root"])
    return ensure_dir(outputs_root / f"{cfg['project']['name']}_{timestamp_string()}")


def _save_checkpoint(path: str | Path, model, optimizer, epoch: int, metrics: Dict, stage: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": int(epoch),
            "stage": str(stage),
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "metrics": metrics,
        },
        path,
    )


def _summarize_epoch_rows(epoch_rows: list[Dict[str, float]]) -> Dict[str, float]:
    if not epoch_rows:
        return {"loss": 0.0}
    keys = sorted({key for row in epoch_rows for key in row.keys()})
    summary: Dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in epoch_rows if key in row]
        summary[key] = float(sum(values) / max(len(values), 1))
    return summary


def _sequence_dir_from_cfg(cfg: Dict) -> Path:
    dataset_cfg = cfg["dataset"]
    return resolve_sequence_dir(
        dataset_cfg["processed_root"],
        str(dataset_cfg.get("name", "kitti")),
        dataset_cfg.get("train_sequence", "00"),
    )


def _build_processed_dataset_kwargs(cfg: Dict, *, for_eval: bool = False) -> Dict[str, object]:
    dataset_cfg = cfg.get("dataset", {})
    dataset_name = str(dataset_cfg.get("name", "kitti")).lower()
    kwargs: Dict[str, object] = {}
    if dataset_name == "kitti":
        mode_key = "kitti_loader_eval_mode" if for_eval else "kitti_loader_train_mode"
        kitti_loader_mode = dataset_cfg.get(mode_key, dataset_cfg.get("kitti_loader_mode", None))
        if kitti_loader_mode is not None:
            kwargs["kitti_loader_mode"] = str(kitti_loader_mode)
    if dataset_cfg.get("expected_meters_per_pixel", None) is not None:
        kwargs["expected_meters_per_pixel"] = float(dataset_cfg["expected_meters_per_pixel"])
    if dataset_cfg.get("expected_model_input_size", None) is not None:
        kwargs["expected_model_input_size"] = int(dataset_cfg["expected_model_input_size"])
    return kwargs


def _validate_dataset_geometry(cfg: Dict) -> Dict[str, float]:
    dataset = ProcessedSequenceDataset(
        _sequence_dir_from_cfg(cfg),
        image_size=int(cfg["model"].get("input_size", 201)),
        split_tags=None,
        **_build_processed_dataset_kwargs(cfg, for_eval=False),
    )
    meta_resolution = float(getattr(dataset.meta, "meters_per_pixel"))
    config_resolution = float(cfg["dataset"].get("bev_resolution_m_per_pixel", meta_resolution))
    if abs(meta_resolution - config_resolution) > 1.0e-6:
        raise AssertionError(
            f"BEV resolution mismatch: meta.yaml={meta_resolution}, config={config_resolution}"
        )
    cfg["dataset"]["bev_resolution_m_per_pixel"] = meta_resolution
    return {
        "bev_resolution_m_per_pixel": meta_resolution,
        "source_image_height": float(getattr(dataset.meta, "source_image_height")),
        "source_image_width": float(getattr(dataset.meta, "source_image_width")),
        "model_input_size": float(getattr(dataset.meta, "model_input_size", cfg["model"].get("input_size", 201))),
    }


def _build_model(cfg: Dict, device: torch.device) -> USVLoc:
    return USVLoc(cfg["model"]).to(device)


def _build_optimizer(model: USVLoc, stage2_cfg: Dict):
    optimizer_name = str(stage2_cfg.get("optimizer", "adamw")).lower()
    base_lr = float(stage2_cfg.get("lr", 5.0e-5))
    base_weight_decay = float(stage2_cfg.get("weight_decay", 1.0e-4))
    param_group_cfg = stage2_cfg.get("param_groups", {})
    if not isinstance(param_group_cfg, dict):
        param_group_cfg = {}

    if not param_group_cfg:
        optimizer = (
            Adam(model.parameters(), lr=base_lr, weight_decay=base_weight_decay)
            if optimizer_name == "adam"
            else AdamW(model.parameters(), lr=base_lr, weight_decay=base_weight_decay)
        )
        return optimizer, [{"group_name": "all", "lr": base_lr, "weight_decay": base_weight_decay}]

    assigned: set[int] = set()
    optimizer_groups: list[Dict[str, object]] = []
    optimizer_meta: list[Dict[str, float | str]] = []

    def _add_group(name: str, params, lr_mult_key: str, wd_mult_key: str) -> None:
        group_params = []
        for parameter in params:
            if id(parameter) in assigned:
                continue
            assigned.add(id(parameter))
            group_params.append(parameter)
        if not group_params:
            return
        lr_mult = float(param_group_cfg.get(lr_mult_key, 1.0))
        wd_mult = float(param_group_cfg.get(wd_mult_key, 1.0))
        group_lr = base_lr * lr_mult
        group_weight_decay = base_weight_decay * wd_mult
        optimizer_groups.append(
            {
                "params": group_params,
                "lr": group_lr,
                "weight_decay": group_weight_decay,
                "group_name": name,
            }
        )
        optimizer_meta.append({"group_name": name, "lr": group_lr, "weight_decay": group_weight_decay})

    _add_group("backbone", list(model.backbone.parameters()), "backbone_lr_mult", "backbone_weight_decay_mult")
    _add_group("polar_attention", list(model.polar_attention.parameters()), "polar_attention_lr_mult", "polar_attention_weight_decay_mult")
    _add_group("other", [parameter for parameter in model.parameters() if id(parameter) not in assigned], "other_lr_mult", "other_weight_decay_mult")

    optimizer = Adam(optimizer_groups) if optimizer_name == "adam" else AdamW(optimizer_groups)
    return optimizer, optimizer_meta


def _build_scheduler(optimizer, stage2_cfg: Dict):
    scheduler_cfg = stage2_cfg.get("scheduler", {})
    if not isinstance(scheduler_cfg, dict) or not bool(scheduler_cfg.get("enabled", False)):
        return None

    scheduler_type = str(scheduler_cfg.get("type", "cosine")).lower()
    if scheduler_type not in {"cosine", "cosineannealing", "cosine_annealing"}:
        raise RuntimeError(f"Unsupported stage2.scheduler.type={scheduler_type}")

    total_epochs = max(int(scheduler_cfg.get("t_max", scheduler_cfg.get("T_max", stage2_cfg.get("epochs", 50)))), 1)
    warmup_epochs = max(int(scheduler_cfg.get("warmup_epochs", 0)), 0)
    eta_min = float(scheduler_cfg.get("eta_min", 1.0e-6))
    warmup_start_factor = float(scheduler_cfg.get("warmup_start_factor", 1.0 / float(max(warmup_epochs, 1))))

    class WarmupCosineScheduler:
        def __init__(self, optimizer):
            self.optimizer = optimizer
            self.base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
            self.current_train_epoch = 1
            self.last_lrs = self._compute_lrs(self.current_train_epoch)
            for group, lr in zip(self.optimizer.param_groups, self.last_lrs):
                group["lr"] = lr

        def _compute_lrs(self, train_epoch: int) -> list[float]:
            lrs: list[float] = []
            for base_lr in self.base_lrs:
                if warmup_epochs > 0 and train_epoch <= warmup_epochs:
                    if warmup_epochs == 1:
                        factor = 1.0
                    else:
                        progress = float(train_epoch - 1) / float(warmup_epochs - 1)
                        factor = warmup_start_factor + (1.0 - warmup_start_factor) * progress
                    lrs.append(base_lr * factor)
                    continue
                cosine_epochs = max(total_epochs - warmup_epochs, 1)
                if cosine_epochs == 1:
                    progress = 1.0
                else:
                    progress = float(train_epoch - warmup_epochs - 1) / float(cosine_epochs - 1)
                    progress = min(max(progress, 0.0), 1.0)
                cosine_lr = eta_min + 0.5 * (base_lr - eta_min) * (1.0 + np.cos(np.pi * progress))
                lrs.append(float(cosine_lr))
            return lrs

        def step(self) -> None:
            self.current_train_epoch += 1
            self.last_lrs = self._compute_lrs(self.current_train_epoch)
            for group, lr in zip(self.optimizer.param_groups, self.last_lrs):
                group["lr"] = lr

    return WarmupCosineScheduler(optimizer)


def _build_cross_eval_cfgs(cfg: Dict) -> list[tuple[str, Dict]]:
    evaluation_cfg = cfg.get("evaluation", {})
    cross_eval_cfg = evaluation_cfg.get("cross_dataset_eval", None)
    if not isinstance(cross_eval_cfg, dict) or not bool(cross_eval_cfg.get("enabled", False)):
        return []

    metric_prefix = str(cross_eval_cfg.get("metric_prefix", cross_eval_cfg.get("dataset_name", "cross"))).strip() or "cross"
    cross_cfg = copy.deepcopy(cfg)
    cross_cfg["dataset"]["name"] = str(cross_eval_cfg.get("dataset_name", "nclt")).lower()
    cross_cfg["dataset"]["processed_root"] = cross_eval_cfg.get("processed_root", cfg["dataset"]["processed_root"])
    if "sequences" in cross_eval_cfg:
        cross_cfg["evaluation"]["sequences"] = [str(sequence) for sequence in cross_eval_cfg.get("sequences", [])]
    for key, value in cross_eval_cfg.items():
        if key in {"enabled", "metric_prefix", "dataset_name", "processed_root"}:
            continue
        if key == "database_sequence":
            cross_cfg["evaluation"]["database_sequence"] = str(value)
            if cross_cfg["dataset"]["name"] == "nclt":
                cross_cfg["dataset"]["eval_database_sequence"] = str(value)
            continue
        cross_cfg["evaluation"][key] = value
    cross_cfg["evaluation"].pop("cross_dataset_eval", None)
    return [(metric_prefix, cross_cfg)]


def _evaluate_place_for_cfg(
    model,
    eval_cfg: Dict,
    device: torch.device,
    output_dir: str | Path | None = None,
    shared_frontend: Dict[str, object] | None = None,
) -> Dict[str, object]:
    dataset_name = str(eval_cfg.get("dataset", {}).get("name", "kitti")).lower()
    if dataset_name == "usvinland":
        summary = evaluate_usvinland_place(
            model=model,
            cfg=eval_cfg,
            device=device,
            output_dir=output_dir,
            raw_root=eval_cfg.get("evaluation", {}).get("raw_root", "data/USVInlandRaw"),
            sequences=eval_cfg.get("evaluation", {}).get("sequences", None),
            positive_radius_m=float(
                eval_cfg.get("evaluation", {}).get(
                    "place_positive_radius_m",
                    eval_cfg.get("evaluation", {}).get("positive_radius_m", 5.0),
                )
            ),
            split_ratio=float(eval_cfg.get("evaluation", {}).get("split_ratio", 0.5)),
            eval_batch_size=int(eval_cfg.get("evaluation", {}).get("eval_batch_size", 64)),
            num_workers=int(eval_cfg.get("evaluation", {}).get("num_workers", 4)),
            normalization_divisor=float(eval_cfg.get("evaluation", {}).get("normalization_divisor", 255.0)),
            faiss_gpu=bool(eval_cfg.get("evaluation", {}).get("faiss_gpu", False)),
        )
        return summary["place"]
    return evaluate_place_all(
        model,
        eval_cfg,
        device,
        output_dir=output_dir,
        shared_frontend=shared_frontend,
    )


def _append_place_metrics(epoch_payload: Dict[str, object], place: Dict, prefix: str = "place") -> None:
    epoch_payload[f"{prefix}_mean_recall_at_1"] = float(place["mean_recall_at_1"])
    epoch_payload[f"{prefix}_mean_recall_at_5"] = float(place["mean_recall_at_5"])
    epoch_payload[f"{prefix}_mean_recall_at_1_percent"] = float(place["mean_recall_at_1_percent"])
    epoch_payload[f"{prefix}_sequences"] = list(place["sequences"])
    for sequence_payload in place["per_sequence"]:
        epoch_payload[f"{prefix}_recall_at_1_seq_{sequence_payload['sequence']}"] = float(sequence_payload["Recall@1"])
        epoch_payload[f"{prefix}_recall_at_5_seq_{sequence_payload['sequence']}"] = float(sequence_payload.get("Recall@5", 0.0))
        epoch_payload[f"{prefix}_recall_at_1_percent_seq_{sequence_payload['sequence']}"] = float(sequence_payload.get("Recall@1%", 0.0))


def _load_training_checkpoint(
    checkpoint_path: str | Path,
    model,
    optimizer=None,
    device: torch.device | str = "cpu",
    load_optimizer: bool = True,
) -> tuple[int, Dict[str, object]]:
    """Load a training checkpoint.

    If the checkpoint contains optimizer state and ``load_optimizer=True``, the
    optimizer state is restored. Release checkpoints usually contain only model
    weights, so only model weights are loaded in that case. The returned
    ``start_epoch`` is checkpoint epoch + 1.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    if (
        bool(load_optimizer)
        and optimizer is not None
        and isinstance(checkpoint, dict)
        and checkpoint.get("optimizer", None) is not None
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1 if isinstance(checkpoint, dict) else 1
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    return start_epoch, metadata


def train_usvloc(
    cfg: Dict,
    output_dir: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    load_optimizer: bool = True,
) -> Path:
    """Main USVLoc training function.

    Training flow:
    1. Validate BEV data geometry.
    2. Build the model, optimizer, and scheduler.
    3. Optionally resume from a checkpoint.
    4. Build the hard-mining descriptor cache before each epoch.
    5. Train with lazy triplet loss.
    6. Run place recognition / cross-domain evaluation according to the config.
    7. Save latest, epoch, and best checkpoints.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = _build_run_dir(cfg, explicit_output_dir=output_dir)
    geometry = _validate_dataset_geometry(cfg)
    save_json(run_dir / "config.json", cfg)

    train_cfg = cfg["training"]
    stage2_cfg = train_cfg["stage2"]
    data_cfg = stage2_cfg.get("dataset", {})
    _configure_runtime_threads(
        torch_num_threads=int(train_cfg.get("torch_num_threads", 4)),
        torch_num_interop_threads=int(train_cfg.get("torch_num_interop_threads", 1)),
        cv2_num_threads=int(train_cfg.get("cv2_num_threads", 1)),
    )
    _seed_everything(int(train_cfg.get("seed", 1024)))

    model = _build_model(cfg, device)
    optimizer, optimizer_meta = _build_optimizer(model, stage2_cfg)
    scheduler = _build_scheduler(optimizer, stage2_cfg)
    configured_init_checkpoint = train_cfg.get("init_checkpoint", None)
    resume_path = resume_checkpoint if resume_checkpoint is not None else configured_init_checkpoint
    start_epoch = 1
    resume_metadata: Dict[str, object] = {}
    if resume_path:
        # After checkpoint resume, advance the scheduler to the matching epoch to keep the learning rate continuous.
        start_epoch, resume_metadata = _load_training_checkpoint(
            resume_path,
            model,
            optimizer=optimizer,
            device=device,
            load_optimizer=bool(load_optimizer),
        )
        if scheduler is not None:
            while int(getattr(scheduler, "current_train_epoch", 1)) < int(start_epoch):
                scheduler.step()
        print(
            f"[USVLoc] resumed from {Path(resume_path).resolve()} "
            f"start_epoch={start_epoch} load_optimizer={bool(load_optimizer)}",
            flush=True,
        )
    cross_eval_cfgs = _build_cross_eval_cfgs(cfg)
    best_metric_name = str(cfg.get("evaluation", {}).get("best_metric", "place_mean_recall_at_1"))

    run_meta = {
        "device": str(device),
        "seed": int(train_cfg.get("seed", 1024)),
        "descriptor_dim": int(model.global_descriptor_dim),
        "faiss_gpu_requested": bool(cfg.get("evaluation", {}).get("faiss_gpu", False)),
        "optimizer_groups": optimizer_meta,
        "bev_resolution_m_per_pixel": float(geometry["bev_resolution_m_per_pixel"]),
        "source_image_height": int(geometry["source_image_height"]),
        "source_image_width": int(geometry["source_image_width"]),
        "validated_model_input_size": int(geometry["model_input_size"]),
        "best_metric_name": best_metric_name,
        "cross_dataset_eval": cross_eval_cfgs[0][1]["evaluation"] if cross_eval_cfgs else None,
        "resume_checkpoint": str(Path(resume_path).resolve()) if resume_path else None,
        "resume_start_epoch": int(start_epoch),
        "resume_loaded_optimizer": bool(load_optimizer) if resume_path else False,
        "resume_checkpoint_epoch": int(resume_metadata.get("epoch", 0)) if resume_metadata else None,
    }
    save_json(run_dir / "run_meta.json", run_meta)

    dataset = SBEVLocFrameTripletDataset(
        sequence_dir=_sequence_dir_from_cfg(cfg),
        image_size=int(cfg["model"].get("input_size", 201)),
        split_tags=tuple(cfg["dataset"].get("train_split_tags")) if cfg["dataset"].get("train_split_tags", None) else None,
        max_frame_id=int(cfg["dataset"].get("train_max_frame_id")) if cfg["dataset"].get("train_max_frame_id", None) is not None else None,
        positive_distance_threshold_m=float(data_cfg.get("positive_distance_threshold_m", 5.0)),
        negative_distance_threshold_m=float(data_cfg.get("negative_distance_threshold_m", 7.0)),
        num_negatives=int(data_cfg.get("num_negatives", 10)),
        seed=int(cfg["training"].get("seed", 1024)),
        augment_random_rotation=bool(data_cfg.get("augment_random_rotation", True)),
        hard_mining_enabled=bool(data_cfg.get("hard_mining_enabled", True)),
        hard_negative_candidate_pool_size=int(data_cfg.get("hard_negative_candidate_pool_size", data_cfg.get("num_negatives", 10))),
        hard_positive_mining_enabled=bool(data_cfg.get("hard_positive_mining_enabled", False)),
        processed_dataset_kwargs=_build_processed_dataset_kwargs(cfg, for_eval=False),
    )
    num_workers = int(train_cfg.get("num_workers", 0))
    worker_init_fn = _build_worker_init_fn(train_cfg) if num_workers > 0 else None
    loader = DataLoader(
        dataset,
        batch_size=int(stage2_cfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=sbevloc_collate_fn,
        worker_init_fn=worker_init_fn,
        persistent_workers=False,
    )

    history: list[Dict[str, object]] = []
    best_metric_value = float("-inf")
    log_interval = int(stage2_cfg.get("log_interval", train_cfg.get("log_interval", 25)))
    epochs = int(stage2_cfg.get("epochs", 50))
    eval_every = int(stage2_cfg.get("eval_every_epochs", 1))
    grad_clip_norm = float(stage2_cfg.get("grad_clip_norm", 0.0))

    if start_epoch > epochs:
        print(
            f"[USVLoc] start_epoch={start_epoch} is greater than configured epochs={epochs}; nothing to train.",
            flush=True,
        )
        return run_dir

    for epoch in range(start_epoch, epochs + 1):
        if bool(data_cfg.get("hard_mining_enabled", True)) and str(data_cfg.get("hard_mining_mode", "epoch_cache")) == "epoch_cache":
            # This implementation uses epoch-cache hard mining: each epoch first extracts descriptors for the full
            # training set, then training batches prefer negatives with closer descriptor distances.
            print(f"[USVLoc] building hard mining cache for epoch {epoch}/{epochs}", flush=True)
            dataset.prepare_epoch_hard_mining(
                model=model,
                device=device,
                chunk_size=int(stage2_cfg.get("hard_mining_forward_chunk_size", 16)),
                batch_size=int(stage2_cfg.get("batch_size", 4)),
            )
        else:
            dataset.clear_epoch_hard_mining()

        model.train()
        epoch_rows: list[Dict[str, float]] = []
        for batch_index, batch in enumerate(loader, start=1):
            query = batch["query"].to(device)
            positive = batch["positive"].to(device)
            negative_candidates = batch["negative_candidates"].to(device)
            batch_size_local = int(query.shape[0])
            candidate_count = int(batch["negative_candidates_per_query"][0]) if batch["negative_candidates_per_query"] else 0

            optimizer.zero_grad(set_to_none=True)
            stacked = torch.cat([query, positive, negative_candidates], dim=0)
            # Compute query/positive/negative descriptors in one forward pass to reduce repeated forward overhead.
            desc = model.forward_retrieval(stacked)["global_descriptor"]
            query_desc = desc[:batch_size_local]
            positive_desc = desc[batch_size_local : batch_size_local * 2]
            negative_desc = desc[batch_size_local * 2 :].view(batch_size_local, candidate_count, -1)
            loss, stats = lazy_triplet_loss(
                anchor_descriptor=query_desc,
                positive_descriptor=positive_desc,
                negative_descriptors=negative_desc,
                margin=float(stage2_cfg.get("loss", {}).get("margin", 0.3)),
            )
            loss.backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            row = {key: float(value) for key, value in stats.items()}
            row["triplet_loss"] = float(loss.detach().cpu().item())
            row["candidate_pool_size"] = float(candidate_count)
            epoch_rows.append(row)
            if batch_index == 1 or batch_index % log_interval == 0:
                print(
                    f"[USVLoc][Train] epoch {epoch}/{epochs} step {batch_index}/{len(loader)} "
                    f"loss={row['loss']:.4f} posd={row['positive_distance']:.4f} "
                    f"hardnegd={row['hardest_negative_distance']:.4f}",
                    flush=True,
                )

        epoch_payload: Dict[str, object] = {
            "epoch": epoch,
            **_summarize_epoch_rows(epoch_rows),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        if eval_every > 0 and epoch % eval_every == 0:
            shared_frontend = None
            if str(cfg.get("dataset", {}).get("name", "kitti")).lower() != "usvinland":
                shared_frontend = build_shared_frontend_cache(model, cfg, device)
            place = _evaluate_place_for_cfg(
                model,
                cfg,
                device,
                output_dir=run_dir / f"eval_epoch_{epoch:03d}" / str(cfg["dataset"].get("name", "kitti")),
                shared_frontend=shared_frontend,
            )
            _append_place_metrics(epoch_payload, place, prefix="place")
            print(
                f"[USVLoc][Eval] epoch {epoch}/{epochs} "
                f"meanR1={float(place['mean_recall_at_1']):.4f} "
                f"meanR5={float(place['mean_recall_at_5']):.4f}",
                flush=True,
            )
            for metric_prefix, cross_cfg in cross_eval_cfgs:
                cross_place = _evaluate_place_for_cfg(
                    model,
                    cross_cfg,
                    device,
                    output_dir=run_dir / f"eval_epoch_{epoch:03d}" / metric_prefix,
                )
                _append_place_metrics(epoch_payload, cross_place, prefix=metric_prefix)
                print(
                    f"[USVLoc][CrossEval:{metric_prefix}] epoch {epoch}/{epochs} "
                    f"meanR1={float(cross_place['mean_recall_at_1']):.4f} "
                    f"meanR5={float(cross_place['mean_recall_at_5']):.4f}",
                    flush=True,
                )

        history.append(epoch_payload)
        save_json(run_dir / "training_history.json", history)
        save_tsv(run_dir / "training_history.tsv", history)
        _save_checkpoint(run_dir / "checkpoint_latest.pt", model, optimizer, epoch, epoch_payload, stage="stage2")
        _save_checkpoint(run_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, optimizer, epoch, epoch_payload, stage="stage2")

        if best_metric_name not in epoch_payload:
            available_metrics = sorted(key for key in epoch_payload.keys() if "recall" in key)
            raise KeyError(
                f"best metric {best_metric_name!r} not found in epoch payload. "
                f"Available recall metrics: {available_metrics}"
            )
        current_best_metric = float(epoch_payload[best_metric_name])
        if current_best_metric > best_metric_value:
            best_metric_value = current_best_metric
            _save_checkpoint(run_dir / "model_best.pt", model, optimizer, epoch, epoch_payload, stage="stage2")
            save_json(
                run_dir / "best_metrics.json",
                {
                    "best_metric_name": best_metric_name,
                    "best_metric_value": best_metric_value,
                    "best_epoch": epoch,
                    "best_checkpoint_path": str(run_dir / "model_best.pt"),
                    "best_epoch_checkpoint_path": str(run_dir / f"checkpoint_epoch_{epoch:03d}.pt"),
                    "metrics": epoch_payload,
                },
            )

        if scheduler is not None:
            scheduler.step()

    return run_dir
