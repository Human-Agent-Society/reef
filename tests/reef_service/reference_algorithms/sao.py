"""Dependency-free SAO numerics used as a parity-test oracle."""

from __future__ import annotations

from math import exp

DEFAULT_EPS_LOW = 0.3
DEFAULT_EPS_HIGH = 5.0
DEFAULT_GAMMA = 1.0
DEFAULT_LAMBDA = 1.0
LENGTH_ADAPTIVE_ALPHA = 1.5
# lambda_critic = 1: value targets regress toward Monte-Carlo returns while
# the policy's advantages may use the length-adaptive lambda (Section 4).
CRITIC_LAMBDA = 1.0


def importance_ratios(policy_log_probs: list[float], rollout_log_probs: list[float]) -> list[float]:
    """Return token ratios ``exp(logπ_policy - logπ_rollout)``."""
    if len(policy_log_probs) != len(rollout_log_probs):
        raise ValueError(
            f"policy/rollout log-prob length mismatch: {len(policy_log_probs)} vs {len(rollout_log_probs)}"
        )
    return [exp(policy - rollout) for policy, rollout in zip(policy_log_probs, rollout_log_probs, strict=True)]


def _validate_eps(eps_low: float, eps_high: float) -> None:
    if eps_low < 0 or eps_high < 0:
        raise ValueError(f"eps bounds must be non-negative: eps_low={eps_low}, eps_high={eps_high}")
    if eps_low >= 1.0:
        raise ValueError(f"eps_low must be < 1 so the lower trust bound stays positive: {eps_low}")


def calibration_weights(
    ratios: list[float], eps_low: float = DEFAULT_EPS_LOW, eps_high: float = DEFAULT_EPS_HIGH
) -> list[float]:
    """Keep ratios inside the open trust region and zero the others."""
    _validate_eps(eps_low, eps_high)
    low, high = 1.0 - eps_low, 1.0 + eps_high
    return [ratio if low < ratio < high else 0.0 for ratio in ratios]


def clip_mask(ratios: list[float], eps_low: float = DEFAULT_EPS_LOW, eps_high: float = DEFAULT_EPS_HIGH) -> list[int]:
    """Return the boolean shadow of :func:`calibration_weights`."""
    _validate_eps(eps_low, eps_high)
    low, high = 1.0 - eps_low, 1.0 + eps_high
    return [1 if low < ratio < high else 0 for ratio in ratios]


def clip_ratio(
    ratios: list[float],
    eps_low: float = DEFAULT_EPS_LOW,
    eps_high: float = DEFAULT_EPS_HIGH,
    loss_mask: list[int] | None = None,
) -> float:
    """Return the fraction of trained tokens rejected by calibration."""
    keep = clip_mask(ratios, eps_low, eps_high)
    if loss_mask is None:
        trained = list(range(len(ratios)))
    else:
        if len(loss_mask) != len(ratios):
            raise ValueError(f"loss_mask/ratios length mismatch: {len(loss_mask)} vs {len(ratios)}")
        trained = [index for index, included in enumerate(loss_mask) if included]
    return 0.0 if not trained else sum(1 for index in trained if keep[index] == 0) / len(trained)


def policy_gradient_coeffs(
    ratios: list[float], advantages: list[float], eps_low: float = DEFAULT_EPS_LOW, eps_high: float = DEFAULT_EPS_HIGH
) -> list[float]:
    """Return the detached calibration coefficient for each policy gradient."""
    if len(ratios) != len(advantages):
        raise ValueError(f"ratios/advantages length mismatch: {len(ratios)} vs {len(advantages)}")
    return [
        weight * advantage
        for weight, advantage in zip(calibration_weights(ratios, eps_low, eps_high), advantages, strict=True)
    ]


def length_adaptive_lambda(response_length: int, alpha: float = LENGTH_ADAPTIVE_ALPHA) -> float:
    """Compute SAO's response-length-adaptive GAE lambda."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if response_length <= 0:
        return 0.0
    return min(max(1.0 - 1.0 / (alpha * response_length), 0.0), 1.0)


def role_lambda(
    role: str,
    response_length: int,
    *,
    length_adaptive: bool = False,
    alpha: float = LENGTH_ADAPTIVE_ALPHA,
    lambd: float = DEFAULT_LAMBDA,
    critic_lambda: float = CRITIC_LAMBDA,
) -> float:
    """Select the GAE lambda for one role on one sample.

    The critic role always uses ``critic_lambda`` (paper: 1, Monte-Carlo value
    targets); the actor role uses the length-adaptive lambda when enabled and
    the fixed ``lambd`` otherwise.
    """
    if role not in {"actor", "critic"}:
        raise ValueError(f"unknown role: {role!r}")
    if role == "critic":
        return critic_lambda
    return length_adaptive_lambda(response_length, alpha) if length_adaptive else lambd


def skip_observation_gae(
    values: list[float],
    rewards: list[float],
    action_mask: list[int],
    gamma: float = DEFAULT_GAMMA,
    lambd: float = DEFAULT_LAMBDA,
) -> tuple[list[float], list[float]]:
    """Propagate GAE over action tokens while skipping observation spans."""
    size = len(values)
    if not (len(rewards) == len(action_mask) == size):
        raise ValueError(f"values/rewards/action_mask length mismatch: {size}/{len(rewards)}/{len(action_mask)}")
    advantages, returns = [0.0] * size, [0.0] * size
    last_gae = next_value = 0.0
    for index in reversed(range(size)):
        if not action_mask[index]:
            continue
        delta = rewards[index] + gamma * next_value - values[index]
        last_gae = delta + gamma * lambd * last_gae
        advantages[index], returns[index], next_value = last_gae, last_gae + values[index], values[index]
    return advantages, returns


def explained_variance(returns: list[float], values: list[float], action_mask: list[int] | None = None) -> float:
    """Measure critic fit, optionally over action tokens only."""
    if len(returns) != len(values):
        raise ValueError(f"returns/values length mismatch: {len(returns)} vs {len(values)}")
    if action_mask is None:
        indices = list(range(len(returns)))
    else:
        if len(action_mask) != len(returns):
            raise ValueError(f"action_mask/returns length mismatch: {len(action_mask)} vs {len(returns)}")
        indices = [index for index, included in enumerate(action_mask) if included]
    if not indices:
        return 0.0
    count = len(indices)
    mean_return = sum(returns[index] for index in indices) / count
    var_return = sum((returns[index] - mean_return) ** 2 for index in indices) / count
    if var_return == 0.0:
        return 0.0
    differences = [returns[index] - values[index] for index in indices]
    mean_difference = sum(differences) / count
    return 1.0 - sum((value - mean_difference) ** 2 for value in differences) / count / var_return
