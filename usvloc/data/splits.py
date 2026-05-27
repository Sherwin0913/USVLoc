from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .common import load_sequence, resolve_dataset_root, resolve_sequence_dir

KITTI_SEQ_SPLIT_POINTS = {"00": 3000, "02": 3400, "05": 1000, "06": 600, "08": 1000}
NCLT_DEFAULT_DATABASE_SEQUENCE = "2012-01-15"
NCLT_DEFAULT_QUERY_SEQUENCES = (
    "2012-02-04",
    "2012-03-17",
    "2012-06-15",
    "2012-09-28",
    "2012-11-16",
    "2013-02-23",
)
USVINLAND_DEFAULT_SEQUENCES: tuple[str, ...] = (
    "H05_7_Sequence_160_270",
    "H05_9_Sequence_115_700",
    "N02_4_Sequence_155_370",
    "N03_2_Sequence_80_536",
    "N03_3_Sequence_605_760",
    "N03_4_Sequence_440_523",
    "N03_5_Sequence_12_340",
    "W06_2_Sequence_57_115",
)
NCLT_DB_SPLIT_TAGS = {"train_db", "db", "database"}
NCLT_QUERY_SPLIT_TAGS = {"query", "val_query"}
POHANG_DEFAULT_SEQUENCE_PAIRS: tuple[tuple[str, str], ...] = (
    ("pohang00", "pohang01"),
    ("pohang01", "pohang00"),
)
POHANG_DB_SPLIT_TAGS = {"db", "database", "eval", "train_db"}
POHANG_QUERY_SPLIT_TAGS = {"query", "val_query", "eval"}


@dataclass
class KittiPlaceSpec:
    sequence_name: str
    sequence_dir: Path
    db_indices: np.ndarray
    query_indices: np.ndarray
    positive_radius_m: float


@dataclass
class NcltPlaceSpec:
    query_sequence_name: str
    database_sequence_name: str
    query_sequence_dir: Path
    database_sequence_dir: Path
    query_indices: np.ndarray
    database_indices: np.ndarray
    positive_radius_m: float


@dataclass
class PohangPlaceSpec:
    query_sequence_name: str
    database_sequence_name: str
    query_sequence_dir: Path
    database_sequence_dir: Path
    query_indices: np.ndarray
    database_indices: np.ndarray
    positive_radius_m: float


@dataclass
class USVInlandPlaceSpec:
    query_sequence_name: str
    database_sequence_name: str
    query_sequence_dir: Path
    database_sequence_dir: Path
    query_indices: np.ndarray
    database_indices: np.ndarray
    positive_radius_m: float


def build_kitti_place_spec(
    processed_root: str | Path,
    sequence: str,
    positive_radius_m: float = 5.0,
) -> KittiPlaceSpec:
    sequence_dir = resolve_sequence_dir(processed_root, "kitti", sequence)
    frames, _ = load_sequence(sequence_dir)
    split_point = int(KITTI_SEQ_SPLIT_POINTS[str(sequence)])
    db_indices = frames.index[frames["frame_id"] < split_point].to_numpy(dtype=np.int64)
    query_indices = frames.index[frames["frame_id"] >= split_point + 200].to_numpy(dtype=np.int64)
    if db_indices.size == 0 or query_indices.size == 0:
        raise RuntimeError(f"Invalid KITTI eval split for sequence={sequence} in {sequence_dir}")
    return KittiPlaceSpec(
        sequence_name=str(sequence),
        sequence_dir=sequence_dir,
        db_indices=db_indices,
        query_indices=query_indices,
        positive_radius_m=float(positive_radius_m),
    )


def _list_nclt_sequences(processed_root: str | Path) -> list[str]:
    nclt_root = resolve_dataset_root(processed_root, "nclt")
    sequences = sorted(path.name for path in nclt_root.iterdir() if path.is_dir())
    if not sequences:
        raise RuntimeError(f"No NCLT sequences found in {nclt_root}")
    return sequences


def _list_pohang_sequences(processed_root: str | Path) -> list[str]:
    pohang_root = resolve_dataset_root(processed_root, "pohang")
    sequences = sorted(path.name for path in pohang_root.iterdir() if path.is_dir())
    if not sequences:
        raise RuntimeError(f"No Pohang sequences found in {pohang_root}")
    return sequences


def _list_usvinland_sequences(processed_root: str | Path) -> list[str]:
    usvinland_root = resolve_dataset_root(processed_root, "usvinland")
    sequences = sorted(path.name for path in usvinland_root.iterdir() if path.is_dir())
    if not sequences:
        raise RuntimeError(f"No USVInland sequences found in {usvinland_root}")
    return sequences


def _select_indices_by_tags(frames, preferred_tags: set[str]) -> np.ndarray:
    if "split_tag" not in frames.columns:
        return frames.index.to_numpy(dtype=np.int64)
    mask = frames["split_tag"].astype(str).isin(preferred_tags)
    indices = frames.index[mask].to_numpy(dtype=np.int64)
    if indices.size == 0:
        return frames.index.to_numpy(dtype=np.int64)
    return indices


def build_nclt_place_specs(
    processed_root: str | Path,
    database_sequence: str = NCLT_DEFAULT_DATABASE_SEQUENCE,
    query_sequences: Sequence[str] | None = None,
    positive_radius_m: float = 5.0,
) -> list[NcltPlaceSpec]:
    processed_root = Path(processed_root)
    nclt_root = resolve_dataset_root(processed_root, "nclt")
    available_sequences = _list_nclt_sequences(processed_root)
    database_sequence = str(database_sequence)
    if database_sequence not in available_sequences:
        raise RuntimeError(f"NCLT database sequence {database_sequence} not found in {nclt_root}")

    if query_sequences is None:
        query_sequences = [seq for seq in available_sequences if seq != database_sequence]
    else:
        query_sequences = [str(seq) for seq in query_sequences]

    database_sequence_dir = nclt_root / database_sequence
    database_frames, _ = load_sequence(database_sequence_dir)
    database_indices = _select_indices_by_tags(database_frames, NCLT_DB_SPLIT_TAGS)
    if database_indices.size == 0:
        raise RuntimeError(f"No NCLT database frames found in {database_sequence_dir}")

    specs: list[NcltPlaceSpec] = []
    for query_sequence in query_sequences:
        if query_sequence == database_sequence:
            continue
        query_sequence_dir = nclt_root / query_sequence
        if not query_sequence_dir.is_dir():
            raise RuntimeError(f"NCLT query sequence {query_sequence} not found in {nclt_root}")
        query_frames, _ = load_sequence(query_sequence_dir)
        query_indices = _select_indices_by_tags(query_frames, NCLT_QUERY_SPLIT_TAGS)
        if query_indices.size == 0:
            raise RuntimeError(f"No NCLT query frames found in {query_sequence_dir}")
        specs.append(
            NcltPlaceSpec(
                query_sequence_name=str(query_sequence),
                database_sequence_name=database_sequence,
                query_sequence_dir=query_sequence_dir,
                database_sequence_dir=database_sequence_dir,
                query_indices=query_indices,
                database_indices=database_indices,
                positive_radius_m=float(positive_radius_m),
            )
        )
    if not specs:
        raise RuntimeError(f"No NCLT query sequences resolved under {nclt_root}")
    return specs


def build_pohang_place_specs(
    processed_root: str | Path,
    sequence_pairs: Sequence[tuple[str, str]] | None = None,
    positive_radius_m: float = 5.0,
) -> list[PohangPlaceSpec]:
    processed_root = Path(processed_root)
    pohang_root = resolve_dataset_root(processed_root, "pohang")
    available_sequences = _list_pohang_sequences(processed_root)

    if sequence_pairs is None:
        sequence_pairs = list(POHANG_DEFAULT_SEQUENCE_PAIRS)
    else:
        sequence_pairs = [(str(database_sequence), str(query_sequence)) for database_sequence, query_sequence in sequence_pairs]

    specs: list[PohangPlaceSpec] = []
    for database_sequence, query_sequence in sequence_pairs:
        if database_sequence not in available_sequences:
            raise RuntimeError(f"Pohang database sequence {database_sequence} not found in {pohang_root}")
        if query_sequence not in available_sequences:
            raise RuntimeError(f"Pohang query sequence {query_sequence} not found in {pohang_root}")
        if database_sequence == query_sequence:
            raise RuntimeError("Pohang cross-sequence evaluation requires database and query sequences to differ.")

        database_sequence_dir = pohang_root / database_sequence
        query_sequence_dir = pohang_root / query_sequence
        database_frames, _ = load_sequence(database_sequence_dir)
        query_frames, _ = load_sequence(query_sequence_dir)
        database_indices = _select_indices_by_tags(database_frames, POHANG_DB_SPLIT_TAGS)
        query_indices = _select_indices_by_tags(query_frames, POHANG_QUERY_SPLIT_TAGS)
        if database_indices.size == 0:
            raise RuntimeError(f"No Pohang database frames found in {database_sequence_dir}")
        if query_indices.size == 0:
            raise RuntimeError(f"No Pohang query frames found in {query_sequence_dir}")
        specs.append(
            PohangPlaceSpec(
                query_sequence_name=str(query_sequence),
                database_sequence_name=str(database_sequence),
                query_sequence_dir=query_sequence_dir,
                database_sequence_dir=database_sequence_dir,
                query_indices=query_indices,
                database_indices=database_indices,
                positive_radius_m=float(positive_radius_m),
            )
        )

    if not specs:
        raise RuntimeError(f"No Pohang sequence pairs resolved under {pohang_root}")
    return specs


def build_usvinland_place_specs(
    processed_root: str | Path,
    sequences: Sequence[str] | None = None,
    positive_radius_m: float = 5.0,
) -> list[USVInlandPlaceSpec]:
    processed_root = Path(processed_root)
    usvinland_root = resolve_dataset_root(processed_root, "usvinland")
    available_sequences = _list_usvinland_sequences(processed_root)
    if sequences is None:
        sequences = [seq for seq in USVINLAND_DEFAULT_SEQUENCES if seq in available_sequences]
        if not sequences:
            sequences = available_sequences
    else:
        sequences = [str(sequence) for sequence in sequences]

    specs: list[USVInlandPlaceSpec] = []
    for sequence in sequences:
        if sequence not in available_sequences:
            raise RuntimeError(f"USVInland sequence {sequence} not found in {usvinland_root}")
        sequence_dir = usvinland_root / sequence
        frames, _ = load_sequence(sequence_dir)
        database_indices = _select_indices_by_tags(frames, POHANG_DB_SPLIT_TAGS)
        query_indices = _select_indices_by_tags(frames, POHANG_QUERY_SPLIT_TAGS)
        if database_indices.size == 0 or query_indices.size == 0:
            split_index = max(1, min(len(frames) - 1, int(round(len(frames) * 0.5))))
            database_indices = frames.index[:split_index].to_numpy(dtype=np.int64)
            query_indices = frames.index[split_index:].to_numpy(dtype=np.int64)
        if database_indices.size == 0:
            raise RuntimeError(f"No USVInland database frames found in {sequence_dir}")
        if query_indices.size == 0:
            raise RuntimeError(f"No USVInland query frames found in {sequence_dir}")
        specs.append(
            USVInlandPlaceSpec(
                query_sequence_name=str(sequence),
                database_sequence_name=str(sequence),
                query_sequence_dir=sequence_dir,
                database_sequence_dir=sequence_dir,
                query_indices=query_indices,
                database_indices=database_indices,
                positive_radius_m=float(positive_radius_m),
            )
        )

    if not specs:
        raise RuntimeError(f"No USVInland sequences resolved under {usvinland_root}")
    return specs
