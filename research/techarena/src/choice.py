"""Small event-conditional choice model with an explicit no-click option."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


DEFAULT_FEATURES = [
    "svd_dot", "svd_cos", "svd32_dot", "svd32_cos",
    "history_mean_cos", "history32_mean_cos", "item_ctr",
    "history_count", "history_age_days", "gap_hours",
]


@dataclass
class ChoiceModel:
    feature_names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    no_click_bias: float
    no_click_history_weight: float
    l2: float

    def _arrays(self, frame):
        values = frame[self.feature_names].to_numpy(np.float64)
        values = (values - self.mean) / self.scale
        n_events = len(frame) // 10
        return values.reshape(n_events, 10, -1), frame["history_length"].to_numpy(np.float64).reshape(n_events, 10)[:, 0]

    def predict(self, frame):
        x, history_length = self._arrays(frame)
        utility = np.einsum("ecf,f->ec", x, self.weights)
        no_click = self.no_click_bias + self.no_click_history_weight * np.minimum(history_length, 20.0) / 20.0
        logits = np.concatenate([no_click[:, None], utility], axis=1)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return utility.reshape(-1), probs[:, 1:].reshape(-1), probs[:, 0]


def _prepare(frame, feature_names):
    values = frame[feature_names].to_numpy(np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x = ((values - mean) / scale).reshape(len(frame) // 10, 10, -1)
    labels = frame["label"].to_numpy(np.int8).reshape(len(frame) // 10, 10)
    classes = np.zeros(len(labels), dtype=np.int64)
    for e in range(len(labels)):
        clicked = np.flatnonzero(labels[e] == 1)
        classes[e] = int(clicked[0] + 1) if len(clicked) else 0
    history = frame["history_length"].to_numpy(np.float64).reshape(len(frame) // 10, 10)[:, 0]
    return x, classes, np.minimum(history, 20.0) / 20.0, mean, scale


def fit_choice(frame, feature_names: list[str] | None = None, l2: float = 0.01, maxiter: int = 100) -> ChoiceModel:
    feature_names = list(feature_names or DEFAULT_FEATURES)
    x, classes, history, mean, scale = _prepare(frame, feature_names)
    n_features = x.shape[2]
    n_events = x.shape[0]

    def objective(params, gradient=False):
        weights = params[:n_features]
        bias = params[n_features]
        hist_weight = params[n_features + 1]
        utility = np.einsum("ecf,f->ec", x, weights)
        no_click = bias + hist_weight * history
        logits = np.concatenate([no_click[:, None], utility], axis=1)
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        loss = -np.log(np.maximum(probs[np.arange(n_events), classes], 1e-12)).mean()
        loss += 0.5 * l2 * np.dot(weights, weights)
        if not gradient:
            return float(loss)
        residual = probs.copy()
        residual[np.arange(n_events), classes] -= 1.0
        grad_w = np.einsum("ec,ecf->f", residual[:, 1:], x) / n_events + l2 * weights
        grad_bias = residual[:, 0].mean()
        grad_hist = np.mean(residual[:, 0] * history)
        return float(loss), np.r_[grad_w, grad_bias, grad_hist]

    def fun(params):
        loss, grad = objective(params, gradient=True)
        return loss, grad

    initial = np.zeros(n_features + 2, dtype=np.float64)
    initial[n_features] = np.log(0.37 / 0.63)
    result = minimize(fun, initial, method="L-BFGS-B", jac=True, options={"maxiter": maxiter, "ftol": 1e-9, "maxls": 20})
    if not result.success:
        print(f"choice optimizer: {result.message}", flush=True)
    return ChoiceModel(feature_names, mean, scale, result.x[:n_features], float(result.x[n_features]), float(result.x[n_features + 1]), l2)
