from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from reciprocal_match.evaluation.marketplace import (
    allocation_metrics, aggregate_random_repeats,
    paired_tradeoff_differences, tradeoff_summary,
)
from reciprocal_match.marketplace.allocation import (
    prepare_marketplace_pairs, random_greedy_allocation,
    solve_exposure_controlled_allocation, target_pair_count,
)


@dataclass(frozen=True)
class MarketplaceExperimentResult:
    per_wave: pd.DataFrame
    tradeoff: pd.DataFrame
    paired: pd.DataFrame
    user_exposure: pd.DataFrame


def run_marketplace_allocation(
    predictions: pd.DataFrame,
    capacity: int,
    target_capacity_fraction: float,
    lambda_grid: list[float],
    random_repeats: int,
    seed: int,
) -> MarketplaceExperimentResult:
    pairs = prepare_marketplace_pairs(predictions)
    rows=[]; exposure_frames=[]

    for wave, wave_pairs in pairs.groupby("wave", sort=True):
        n_users = len(set(wave_pairs["u"]) | set(wave_pairs["v"]))
        target = target_pair_count(n_users, capacity, target_capacity_fraction, len(wave_pairs))

        random_rows=[]
        for rep in range(random_repeats):
            result = random_greedy_allocation(
                wave_pairs, capacity, target,
                seed=seed + int(wave) * 10000 + rep,
            )
            random_rows.append(allocation_metrics(result.selected_pairs, result.user_exposure))
        random_metrics=aggregate_random_repeats(random_rows)
        rows.append({"wave":int(wave),"policy":"random","lambda":np.nan,
                     "n_eligible_pairs":len(wave_pairs),"n_allocated_pairs":target,**random_metrics})

        for lam in lambda_grid:
            result=solve_exposure_controlled_allocation(wave_pairs,capacity,target,float(lam))
            metrics=allocation_metrics(result.selected_pairs,result.user_exposure)
            rows.append({"wave":int(wave),"policy":"optimized","lambda":float(lam),
                         "n_eligible_pairs":len(wave_pairs),"n_allocated_pairs":target,**metrics})
            exp=result.user_exposure.copy(); exp["wave"]=int(wave); exposure_frames.append(exp)

    per_wave=pd.DataFrame(rows).sort_values(["wave","policy","lambda"],na_position="first").reset_index(drop=True)
    tradeoff=tradeoff_summary(per_wave)
    paired=paired_tradeoff_differences(per_wave)
    exposure=pd.concat(exposure_frames,ignore_index=True).sort_values(["wave","lambda","user"]).reset_index(drop=True)
    return MarketplaceExperimentResult(per_wave,tradeoff,paired,exposure)
