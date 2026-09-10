from __future__ import annotations

from pathlib import Path

import yaml


def load_protocol_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_protocol_groups(config: dict) -> None:
    groups = config.get("groups", {})
    expected = set(config.get("all_expected_waves", []))

    seen: dict[int, str] = {}
    duplicates: list[int] = []

    for name, spec in groups.items():
        for wave in spec.get("waves", []):
            if wave in seen:
                duplicates.append(wave)
            else:
                seen[wave] = name

    missing = sorted(expected - set(seen))
    extra = sorted(set(seen) - expected)

    if duplicates or missing or extra:
        raise ValueError(
            f"Invalid protocol partition: duplicates={sorted(set(duplicates))}, "
            f"missing={missing}, extra={extra}"
        )

    primary = config.get("primary_protocol_group", {})
    primary_name = primary.get("name")
    primary_waves = list(primary.get("waves", []))

    if primary_name not in groups:
        raise ValueError(f"Unknown primary protocol group: {primary_name}")

    if primary_waves != list(groups[primary_name].get("waves", [])):
        raise ValueError("primary_protocol_group waves must exactly match the named group")
