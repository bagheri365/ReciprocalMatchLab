from __future__ import annotations
import math
import numpy as np
import pandas as pd

POLICY_SCORE_COLUMNS={
    "requester":"score_requester",
    "candidate":"score_candidate",
    "product":"score_product",
    "minimum":"score_minimum",
}


def _binary_ndcg_at_k(sorted_relevance: np.ndarray, k: int) -> float:
    rel=np.asarray(sorted_relevance,dtype=float)
    if rel.sum()==0:
        return float("nan")
    upto=min(k,len(rel))
    weights=1.0/np.log2(np.arange(2,upto+2,dtype=float))
    dcg=float(np.dot(rel[:upto],weights))
    ideal_hits=min(int(rel.sum()),upto)
    idcg=float(weights[:ideal_hits].sum())
    return dcg/idcg if idcg>0 else float("nan")


def evaluate_reciprocal_policies(scored: pd.DataFrame,k_values:list[int]) -> pd.DataFrame:
    rows=[]
    k_values=sorted(set(int(k) for k in k_values))
    for wave,wdf in scored.groupby("wave",sort=True):
        for policy,col in POLICY_SCORE_COLUMNS.items():
            like_sums={k:0.0 for k in k_values}
            match_sums={k:0.0 for k in k_values}
            selected_counts={k:0 for k in k_values}
            ndcg_values={k:[] for k in k_values}

            for _,g in wdf.groupby("iid",sort=False):
                ranked=g.sort_values([col,"pid"],ascending=[False,True],kind="mergesort")
                likes=ranked["y_like"].to_numpy(float)
                matches=ranked["y_match"].to_numpy(float)
                c_like=np.cumsum(likes)
                c_match=np.cumsum(matches)

                for k in k_values:
                    n=min(k,len(ranked))
                    if n==0:
                        continue
                    like_sums[k]+=float(c_like[n-1])
                    match_sums[k]+=float(c_match[n-1])
                    selected_counts[k]+=n
                    ndcg=_binary_ndcg_at_k(matches,k)
                    if np.isfinite(ndcg):
                        ndcg_values[k].append(ndcg)

            for k in k_values:
                denom=selected_counts[k]
                rows.append({
                    "wave":int(wave),"policy":policy,"k":k,
                    "like_rate_at_k":like_sums[k]/denom if denom else float("nan"),
                    "mutual_outcome_yield_at_k":match_sums[k]/denom if denom else float("nan"),
                    "ndcg_match_at_k":float(np.mean(ndcg_values[k])) if ndcg_values[k] else float("nan"),
                })
    return pd.DataFrame(rows).sort_values(["wave","policy","k"]).reset_index(drop=True)


def render_reciprocal_summary(metrics):
    agg=metrics.groupby(["policy","k"])[["like_rate_at_k","mutual_outcome_yield_at_k","ndcg_match_at_k"]].mean().reset_index()
    lines=["# Calibrated Reciprocal Scoring","",
           "Platt calibration is fit only on each outer model's training scores and labels.",
           "This avoids held-out-wave leakage but may be optimistic; it is used for reciprocal score construction rather than definitive calibration claims.","",
           "| policy | K | LikeRate@K | MutualOutcomeYield@K | NDCG(match)@K |",
           "|---|---:|---:|---:|---:|"]
    for r in agg.itertuples(index=False):
        lines.append(f"| {r.policy} | {int(r.k)} | {r.like_rate_at_k:.4f} | {r.mutual_outcome_yield_at_k:.4f} | {r.ndcg_match_at_k:.4f} |")
    lines += ["","`MutualOutcomeYield@K` is a directed static-replay metric, not a realized match count.",""]
    return "\n".join(lines)
