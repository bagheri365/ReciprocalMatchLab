from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


@dataclass(frozen=True)
class TaxonomyValidation:
    missing_columns: tuple[str, ...]
    duplicate_columns: tuple[str, ...]
    unexpected_columns: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not (self.missing_columns or self.duplicate_columns or self.unexpected_columns)


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def category_map(config: dict) -> dict[str, list[str]]:
    categories = config.get("categories")
    if not isinstance(categories, dict):
        raise ValueError("feature taxonomy requires a 'categories' mapping")
    return {name: list(cols or []) for name, cols in categories.items()}


def validate_taxonomy_columns(
    dataset_columns: Iterable[str],
    config: dict,
) -> TaxonomyValidation:
    dataset = list(dataset_columns)
    categories = category_map(config)

    seen: dict[str, int] = {}
    for cols in categories.values():
        for col in cols:
            seen[col] = seen.get(col, 0) + 1

    dataset_set = set(dataset)
    taxonomy_set = set(seen)

    missing = sorted(dataset_set - taxonomy_set)
    unexpected = sorted(taxonomy_set - dataset_set)
    duplicates = sorted(col for col, count in seen.items() if count > 1)

    return TaxonomyValidation(
        missing_columns=tuple(missing),
        duplicate_columns=tuple(duplicates),
        unexpected_columns=tuple(unexpected),
    )


def primary_model_features(config: dict) -> list[str]:
    categories = category_map(config)
    modeling = config.get("modeling", {})
    groups = modeling.get("allowed_primary_groups", [])
    out: list[str] = []
    for group in groups:
        if group not in categories:
            raise ValueError(f"Unknown allowed primary feature group: {group}")
        out.extend(categories[group])
    return out


def blocked_features(config: dict) -> set[str]:
    categories = category_map(config)
    modeling = config.get("modeling", {})
    groups = modeling.get("explicitly_blocked_groups", [])
    out: set[str] = set()
    for group in groups:
        if group not in categories:
            raise ValueError(f"Unknown blocked feature group: {group}")
        out.update(categories[group])
    return out


def validate_primary_vs_blocked(config: dict) -> None:
    primary = set(primary_model_features(config))
    blocked = blocked_features(config)
    overlap = sorted(primary & blocked)
    if overlap:
        raise ValueError(f"Primary features overlap blocked features: {overlap}")
