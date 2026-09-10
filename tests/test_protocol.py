from pathlib import Path

import yaml

from reciprocal_match.data.protocol import validate_protocol_groups


def _load_protocols():
    with Path("configs/protocol_groups.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_protocol_groups_partition_all_21_waves():
    config = _load_protocols()
    validate_protocol_groups(config)


def test_primary_waves_are_frozen_standard_protocol_waves():
    config = _load_protocols()
    assert config["primary_protocol_group"]["waves"] == [
        1, 2, 3, 4, 10, 11, 15, 16, 17
    ]


def test_wave_12_is_not_primary():
    config = _load_protocols()
    primary = set(config["primary_protocol_group"]["waves"])
    assert 12 not in primary
