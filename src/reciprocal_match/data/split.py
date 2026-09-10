from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from reciprocal_match.data.preprocessing import leave_one_wave_out


def load_contract(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_loow_contract(df: pd.DataFrame, contract: dict) -> None:
    waves = list(contract["protocol"]["primary_waves"])
    observed = set(df["wave"].dropna().astype(int).unique())
    missing = sorted(set(waves) - observed)
    if missing:
        raise ValueError(f"Primary waves absent from dataset: {missing}")

    folds = list(leave_one_wave_out(df, waves))
    if len(folds) != len(waves):
        raise ValueError("LOOW fold count does not equal number of primary waves")

    for fold in folds:
        train_waves = set(fold.train["wave"].astype(int).unique())
        test_waves = set(fold.test["wave"].astype(int).unique())
        if test_waves != {fold.test_wave}:
            raise ValueError("Test fold contains unexpected waves")
        if fold.test_wave in train_waves:
            raise ValueError("Held-out wave leaked into training fold")
