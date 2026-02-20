from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def get_reference_dataset(datamodule: Any) -> Any:
    if datamodule is None:
        return None
    for attr in ("val_dataset", "train_dataset", "test_dataset", "dataset"):
        dataset = getattr(datamodule, attr, None)
        if dataset is not None:
            return dataset
    return None


def build_class_registry(
    *,
    cfg: Any,
    datamodule: Any,
    val_class_ids: list[int],
    experiment_name: str,
    experiment_dir: str,
) -> dict[str, Any]:
    dataset = get_reference_dataset(datamodule)

    if val_class_ids:
        class_ids = [int(cid) for cid in val_class_ids]
    elif dataset is not None and hasattr(dataset, "class_ids"):
        class_ids = [int(cid) for cid in dataset.class_ids]
    else:
        class_ids = list(range(int(cfg.model.num_classes)))

    class_splits: dict[str, Any] = {}
    if dataset is not None and hasattr(dataset, "class_splits"):
        class_splits = to_jsonable(getattr(dataset, "class_splits", {})) or {}

    split_lookup: dict[int, list[str]] = {}
    for split_name, split_ids in class_splits.items():
        if not isinstance(split_ids, list):
            continue
        for split_id in split_ids:
            split_lookup.setdefault(int(split_id), []).append(str(split_name))

    classes_cfg = None
    if datamodule is not None and getattr(datamodule, "cfg", None) is not None:
        classes_cfg = getattr(datamodule.cfg, "classes", None)
    if classes_cfg is None and dataset is not None and getattr(dataset, "cfg", None) is not None:
        classes_cfg = getattr(dataset.cfg, "classes", None)
    classes_cfg = classes_cfg or {}

    classes_out: dict[str, dict[str, Any]] = {}
    for class_id in class_ids:
        params = classes_cfg.get(class_id, None)
        if params is None and str(class_id) in classes_cfg:
            params = classes_cfg[str(class_id)]
        classes_out[str(class_id)] = {
            "label": f"class_{class_id}",
            "sweeps": split_lookup.get(class_id, []),
            "params": to_jsonable(params) if params is not None else {},
        }

    return {
        "schema_version": "1.0",
        "experiment_name": experiment_name,
        "experiment_dir": experiment_dir,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "class_splits": class_splits,
        "classes": classes_out,
    }
