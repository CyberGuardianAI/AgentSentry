from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CALIBRATION_SCHEMA_VERSION = 1
CALIBRATED_PARAMETER_NAMES = (
    "alpha",
    "p0_suff",
    "delta_min",
    "suff_min",
    "cusum_k",
    "cusum_h",
)


@dataclass(frozen=True)
class CalibrationObservation:
    episode_id: str
    boundary: int
    masked_probe_survival: float | None
    user_preserving_projection: float
    probe_conditional_projection: float | None
    effective_replay_replicates: int
    pending_action_present: bool

    @property
    def projection_statistic(self) -> float:
        if not self.pending_action_present:
            return 0.0
        candidates = [max(0.0, self.user_preserving_projection)]
        if self.probe_conditional_projection is not None:
            candidates.append(max(0.0, self.probe_conditional_projection))
        return max(candidates)


def _validate_probability(name: str, value: Any, *, allow_one: bool = True) -> float:
    parsed = float(value)
    upper_ok = parsed <= 1.0 if allow_one else parsed < 1.0
    if not math.isfinite(parsed) or parsed < 0.0 or not upper_ok:
        upper = "1" if allow_one else "1 (exclusive)"
        raise ValueError(f"{name} must be finite and in [0, {upper}]")
    return parsed


def validate_calibrated_parameters(parameters: Mapping[str, Any]) -> dict[str, float]:
    missing = [name for name in CALIBRATED_PARAMETER_NAMES if name not in parameters]
    if missing:
        raise ValueError(f"Calibration artifact is missing parameters: {', '.join(missing)}")
    validated = {
        "alpha": _validate_probability("alpha", parameters["alpha"], allow_one=False),
        "p0_suff": _validate_probability("p0_suff", parameters["p0_suff"]),
        "delta_min": _validate_probability("delta_min", parameters["delta_min"]),
        "suff_min": _validate_probability("suff_min", parameters["suff_min"]),
        "cusum_k": _validate_probability("cusum_k", parameters["cusum_k"]),
        "cusum_h": float(parameters["cusum_h"]),
    }
    if validated["alpha"] <= 0.0:
        raise ValueError("alpha must be greater than zero")
    if not math.isfinite(validated["cusum_h"]) or validated["cusum_h"] <= 0.0:
        raise ValueError("cusum_h must be finite and greater than zero")
    return validated


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_calibration_artifact(
    path: str | Path,
    *,
    expected_profile: str | None = None,
    expected_runtime: str | None = None,
    expected_task_split_assignment: str | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    artifact_path = Path(path).expanduser().resolve()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported AgentSentry calibration schema: {payload.get('schema_version')!r}"
        )
    if payload.get("data_scope") != "held-out-benign-development":
        raise ValueError("Calibration artifact must declare held-out benign/development data only")
    profile = str(payload.get("profile", ""))
    runtime = str(payload.get("runtime", ""))
    if expected_profile and profile != expected_profile:
        raise ValueError(
            f"Calibration profile mismatch: expected {expected_profile!r}, found {profile!r}"
        )
    if expected_runtime and runtime != expected_runtime:
        raise ValueError(
            f"Calibration runtime mismatch: expected {expected_runtime!r}, found {runtime!r}"
        )
    task_split = payload.get("task_split")
    if expected_task_split_assignment:
        if not isinstance(task_split, dict):
            raise ValueError("Calibration artifact is not bound to an AgentDojo task split")
        actual_assignment = str(task_split.get("assignment_sha256") or "")
        if actual_assignment != expected_task_split_assignment:
            raise ValueError(
                "Calibration task split mismatch: expected "
                f"{expected_task_split_assignment!r}, found {actual_assignment!r}"
            )
    parameters = validate_calibrated_parameters(payload.get("parameters", {}))
    metadata = {
        "path": str(artifact_path),
        "sha256": sha256_file(artifact_path),
        "profile": profile,
        "runtime": runtime,
        "observation_count": int(payload.get("sample_counts", {}).get("boundaries", 0)),
        "episode_count": int(payload.get("sample_counts", {}).get("episodes", 0)),
        "task_split": task_split,
    }
    return parameters, metadata


def exact_binomial_upper(successes: int, trials: int, failure_probability: float) -> float:
    if trials <= 0:
        raise ValueError("At least one Bernoulli trial is required")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")
    alpha = _validate_probability("failure_probability", failure_probability, allow_one=False)
    if alpha <= 0.0:
        raise ValueError("failure_probability must be greater than zero")
    if successes == trials:
        return 1.0

    def binomial_cdf(probability: float) -> float:
        return sum(
            math.comb(trials, count)
            * probability**count
            * (1.0 - probability) ** (trials - count)
            for count in range(successes + 1)
        )

    lower = successes / trials
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if binomial_cdf(midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def finite_sample_upper_quantile(values: Sequence[float], tail_probability: float) -> float:
    if not values:
        raise ValueError("At least one calibration value is required")
    alpha = _validate_probability("tail_probability", tail_probability, allow_one=False)
    if alpha <= 0.0:
        raise ValueError("tail_probability must be greater than zero")
    ordered = sorted(float(value) for value in values)
    rank = math.ceil((len(ordered) + 1) * (1.0 - alpha))
    index = min(len(ordered) - 1, max(0, rank - 1))
    return ordered[index]


def hoeffding_upper_mean(values: Sequence[float], failure_probability: float) -> float:
    if not values:
        raise ValueError("At least one calibration value is required")
    alpha = _validate_probability("failure_probability", failure_probability, allow_one=False)
    if alpha <= 0.0:
        raise ValueError("failure_probability must be greater than zero")
    mean = sum(values) / len(values)
    radius = math.sqrt(math.log(1.0 / alpha) / (2.0 * len(values)))
    return min(1.0, max(0.0, mean + radius))


def _episode_max_cusum(
    observations: Sequence[CalibrationObservation], reference_drift: float
) -> float:
    running = 0.0
    maximum = 0.0
    for observation in sorted(observations, key=lambda item: item.boundary):
        running = max(0.0, running + observation.projection_statistic - reference_drift)
        maximum = max(maximum, running)
    return maximum


def calibrate_parameters(
    observations: Sequence[CalibrationObservation],
    *,
    alpha: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    if not observations:
        raise ValueError("No held-out benign calibration observations were supplied")
    alpha = _validate_probability("alpha", alpha, allow_one=False)
    if alpha <= 0.0:
        raise ValueError("alpha must be greater than zero")

    masked_successes = 0
    masked_trials = 0
    masked_rates: list[float] = []
    projection_statistics: list[float] = []
    episodes: dict[str, list[CalibrationObservation]] = {}
    replay_resolutions: list[float] = []

    for observation in observations:
        episodes.setdefault(observation.episode_id, []).append(observation)
        projection_statistics.append(observation.projection_statistic)
        if observation.effective_replay_replicates > 0:
            replay_resolutions.append(1.0 / observation.effective_replay_replicates)
        if observation.masked_probe_survival is None:
            continue
        rate = _validate_probability("masked_probe_survival", observation.masked_probe_survival)
        trials = max(1, int(observation.effective_replay_replicates))
        successes = int(round(rate * trials))
        masked_successes += min(trials, max(0, successes))
        masked_trials += trials
        masked_rates.append(rate)

    if masked_trials == 0 or not masked_rates:
        raise ValueError("Calibration traces contain no executed masked-probe regimes")

    episode_projection_maxima = [
        max(observation.projection_statistic for observation in episode_observations)
        for episode_observations in episodes.values()
    ]
    episode_projection_means = [
        sum(observation.projection_statistic for observation in episode_observations)
        / len(episode_observations)
        for episode_observations in episodes.values()
    ]
    episode_masked_maxima = [
        max(
            observation.masked_probe_survival
            for observation in episode_observations
            if observation.masked_probe_survival is not None
        )
        for episode_observations in episodes.values()
        if any(observation.masked_probe_survival is not None for observation in episode_observations)
    ]

    pooled_p0_upper = exact_binomial_upper(masked_successes, masked_trials, alpha)
    clustered_p0_upper = finite_sample_upper_quantile(episode_masked_maxima, alpha)
    p0_suff = max(pooled_p0_upper, clustered_p0_upper)
    delta_min = finite_sample_upper_quantile(episode_projection_maxima, alpha)
    suff_min = clustered_p0_upper
    cusum_k = hoeffding_upper_mean(episode_projection_means, alpha)
    episode_maxima = [
        _episode_max_cusum(episode_observations, cusum_k)
        for episode_observations in episodes.values()
    ]
    calibrated_maximum = finite_sample_upper_quantile(episode_maxima, alpha)
    minimum_step = min(replay_resolutions) if replay_resolutions else 1.0
    cusum_h = max(minimum_step, math.nextafter(calibrated_maximum, math.inf))

    parameters = {
        "alpha": alpha,
        "p0_suff": p0_suff,
        "delta_min": delta_min,
        "suff_min": suff_min,
        "cusum_k": cusum_k,
        "cusum_h": cusum_h,
    }
    diagnostics = {
        "masked_probe_successes": masked_successes,
        "masked_probe_trials": masked_trials,
        "masked_probe_empirical_rate": masked_successes / masked_trials,
        "masked_probe_pooled_exact_upper": pooled_p0_upper,
        "masked_probe_episode_max_quantile": clustered_p0_upper,
        "projection_statistic_mean": sum(projection_statistics) / len(projection_statistics),
        "projection_statistic_max": max(projection_statistics),
        "episode_projection_maxima": episode_projection_maxima,
        "episode_projection_means": episode_projection_means,
        "episode_max_cusum": episode_maxima,
        "calibration_exceedances": {
            "projection_score_above_delta_min": sum(
                value > delta_min for value in projection_statistics
            ),
            "masked_survival_above_suff_min": sum(value > suff_min for value in masked_rates),
            "episode_cusum_at_or_above_h": sum(value >= cusum_h for value in episode_maxima),
        },
    }
    return parameters, diagnostics


def calibration_sensitivity(
    observations: Sequence[CalibrationObservation], alpha_values: Iterable[float]
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for alpha in sorted(set(float(value) for value in alpha_values)):
        parameters, _ = calibrate_parameters(observations, alpha=alpha)
        rows.append(parameters)
    return rows
