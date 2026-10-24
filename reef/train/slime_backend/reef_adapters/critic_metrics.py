"""Compute critic explained variance after microbatch and DP/CP reduction."""

from collections.abc import Mapping


def finalize_critic_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    """Replace additive moments with token-weighted explained variance.

    All five moments must have the same reduction factor. Their ratios then
    recover global moments, independent of microbatch sizes and rank splits.
    Zero return variance has no defined EV; report validity instead of zero.
    """
    result = dict(metrics)
    for key in metrics:
        if not key.endswith("ev_count"):
            continue
        prefix = key.removesuffix("ev_count")
        names = ("count", "return_sum", "return_square_sum", "error_sum", "error_square_sum")
        moments = [result.pop(f"{prefix}ev_{name}") for name in names]
        if not all(isinstance(value, int | float) for value in moments):
            raise TypeError("Reduced critic moments must be numeric scalars")
        count, return_sum, return_square_sum, error_sum, error_square_sum = [
            float(value) for value in moments if isinstance(value, int | float)
        ]
        if count > 0:
            return_variance = max(0.0, return_square_sum / count - (return_sum / count) ** 2)
            error_variance = max(0.0, error_square_sum / count - (error_sum / count) ** 2)
        else:
            return_variance, error_variance = 0.0, 0.0
        result[f"{prefix}return_variance"] = return_variance
        result[f"{prefix}explained_variance_valid"] = float(return_variance > 0)
        if return_variance > 0:
            result[f"{prefix}explained_variance"] = 1.0 - error_variance / return_variance
    return result
