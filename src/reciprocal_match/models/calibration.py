from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression


def clipped_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


@dataclass
class PlattCalibrator:
    model: LogisticRegression
    eps: float = 1e-6

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        x = clipped_logit(raw_probability, self.eps).reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]


def fit_platt_calibrator(raw_probability, y_true, eps=1e-6, random_state=365):
    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("Calibration target must contain both classes")
    x = clipped_logit(raw_probability, eps).reshape(-1, 1)
    model = LogisticRegression(random_state=random_state, max_iter=2000)
    model.fit(x, y)
    return PlattCalibrator(model=model, eps=eps)
