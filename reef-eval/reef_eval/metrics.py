"""Metrics: pure functions from DataFrame to DataFrame or Series.

Every function documents the columns it expects; the store never fixes a
schema. Raw scores stay raw in the store, and anything
normalization-shaped is applied here at query time.

Autoresearch curves, over trace rows:

- :func:`anytime`: best-so-far score over time
- :func:`auc`: area under the anytime curve, the anytime score
- :func:`time_to`: when the score first reached a threshold
- :func:`improvements`: how often a submission raised the best score

Budgets, over episode rows:

- :func:`scaling`: score as a function of budget, on any axis
- :func:`efficiency`: reward per unit actually spent

Streams, over episode rows tagged ``stream`` and ``position``:

- :func:`learning_curve`: score over stream position
- :func:`transfer`: what carrying state was worth, against a stateless run
- :func:`forgetting`: how much revisited tasks degraded

Normalizers:

- :func:`rescale_linear`: map raw scores to 0-100
- :func:`rescale_anchored`: 0-100, stretching past 100 instead of clipping

Conventions: ``score`` names the value column (``"reward"`` for episode
rows, ``"score"`` for trace rows; every function takes it as a
parameter), and aggregation over repeats is the mean unless a function
says otherwise.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "anytime",
    "auc",
    "time_to",
    "improvements",
    "scaling",
    "efficiency",
    "learning_curve",
    "transfer",
    "forgetting",
    "rescale_linear",
    "rescale_anchored",
]

# ------------------------------------------------------------------ curves


def anytime(
    df: pd.DataFrame,
    *,
    time: str = "t",
    score: str = "score",
    by: list[str] | None = None,
) -> pd.DataFrame:
    """Best-so-far curve over time: the autoresearch progress curve.

    Expects columns: *time*, *score*, plus any *by* group columns
    (typically task or version). Returns the input sorted by time with a
    ``best_so_far`` column (cumulative max within each group).
    """
    out = df.sort_values((by or []) + [time]).copy()
    grouped = out.groupby(by)[score] if by else out[score]
    out["best_so_far"] = grouped.cummax()
    return out


def auc(
    curve: pd.DataFrame,
    *,
    time: str = "t",
    value: str = "best_so_far",
    start: float | None = None,
    end: float | None = None,
) -> float:
    """Time-averaged best-so-far: the anytime score. Expects the output of
    :func:`anytime` (one group).

    By default the average runs from the first submission to the last, so
    it says how good the run was while it was submitting. That window
    differs per run, which makes the number comparable only across runs
    that submitted at the same times. Pass *start* and *end* (normally 0
    and the budget, in the same unit as *time*) to average over a fixed
    window instead: before the first submission the best-so-far counts as
    0, and the last one is held to *end*. Two runs scored over the same
    window are comparable, and reaching a score earlier scores higher.
    """
    c = curve.sort_values(time)
    if c.empty:
        return 0.0
    times = c[time].astype(float).tolist()
    values = c[value].astype(float).tolist()

    if start is not None:
        # Nothing was submitted yet, so there is no best-so-far to credit.
        times.insert(0, float(start))
        values.insert(0, 0.0)
    if end is not None:
        times.append(float(end))
        values.append(values[-1])

    if len(times) < 2:
        return values[-1]
    span = times[-1] - times[0]
    if span <= 0:
        return values[-1]
    # Left Riemann: each best-so-far is held until the next point.
    area = sum(values[i] * (times[i + 1] - times[i]) for i in range(len(times) - 1))
    return float(area / span)


def scaling(
    df: pd.DataFrame,
    *,
    budget: str = "budget",
    score: str = "reward",
    by: list[str] | None = None,
) -> pd.DataFrame:
    """Score as a function of budget (EdgeBench's 2 to 12 hour curves).

    Expects columns: *budget*, *score*, plus optional *by* groups (model,
    category). Returns mean and count per (group, budget).
    """
    keys = (by or []) + [budget]
    return (
        df.groupby(keys)[score]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": score})
    )


def time_to(
    df: pd.DataFrame,
    threshold: float,
    *,
    time: str = "t",
    score: str = "score",
    by: list[str] | None = None,
) -> pd.DataFrame:
    """When did the score first reach *threshold*?

    Expects columns *time*, *score* (+ *by* groups). Returns one row per
    group with ``time_to``, the earliest *time* whose score meets the
    threshold, or NaN if the group never reached it.
    """
    hits = df[df[score] >= threshold]
    if not by:
        value = hits[time].min() if len(hits) else float("nan")
        return pd.DataFrame([{"time_to": value}])
    reached = hits.groupby(by)[time].min().rename("time_to").reset_index()
    every_group = df[by].drop_duplicates()
    return every_group.merge(reached, on=by, how="left")


def improvements(
    df: pd.DataFrame,
    *,
    time: str = "t",
    score: str = "score",
    by: list[str] | None = None,
) -> pd.DataFrame:
    """How often self-evaluation actually raised the score.

    Expects columns: *time*, *score* (+ *by* groups). Per group:
    ``evals`` (total points), ``improvements`` (points strictly above the
    best so far; the first point counts), and their ratio
    ``improvement_rate``.

    The submission log records every submission, so the rate is a real property of
    the method: how often spending a submission actually helped.
    """
    out = df.sort_values((by or []) + [time])
    if by:
        cummax = out.groupby(by)[score].cummax()
        prev_best = cummax.groupby([out[k] for k in by]).shift()
    else:
        prev_best = out[score].cummax().shift()
    out = out.assign(_improved=prev_best.isna() | (out[score] > prev_best))
    if not by:
        n = len(out)
        k = int(out["_improved"].sum())
        return pd.DataFrame(
            [{"evals": n, "improvements": k, "improvement_rate": k / n if n else 0.0}]
        )
    agg = out.groupby(by).agg(evals=(score, "size"), improvements=("_improved", "sum"))
    agg["improvement_rate"] = agg["improvements"] / agg["evals"]
    return agg.reset_index()


def efficiency(
    df: pd.DataFrame,
    *,
    spend: str = "used_n_total_tokens",
    score: str = "reward",
    by: list[str] | None = None,
    per: float = 1.0,
) -> pd.DataFrame:
    """Reward per unit of what the run actually spent, the budget's other side.

    ``spend`` is a ``used_*`` column (``used_n_total_tokens``,
    ``used_n_submissions``); ``per`` scales the denominator (e.g. ``1000``
    for reward per 1k tokens). Returns mean reward, mean spend, and
    ``reward_per_unit`` per *by* group (or overall).

    Rows that recorded no spend are dropped, since not every harness
    reports usage. A missing *spend* column is a different thing and
    raises: asking for a column nobody recorded, or misspelling one, would
    otherwise come back as an empty frame that reads like a real answer.
    """
    if spend not in df:
        recorded = [c for c in df.columns if c.startswith("used_")]
        raise KeyError(
            f"no {spend!r} column. Recorded usage columns: "
            f"{', '.join(recorded) or 'none, so no run reported its spend'}"
        )
    have = df[df[spend].notna() & (df[spend] > 0)]
    if have.empty:
        return pd.DataFrame(columns=(by or []) + [score, spend, "reward_per_unit"])

    def _one(g: pd.DataFrame) -> dict:
        s, c = g[score].mean(), g[spend].mean()
        return {
            score: s,
            spend: c,
            "reward_per_unit": s / (c / per) if c else float("nan"),
        }

    if not by:
        return pd.DataFrame([_one(have)])
    rows = [
        {**dict(zip(by, k if isinstance(k, tuple) else (k,), strict=False)), **_one(g)}
        for k, g in have.groupby(by)
    ]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ streams


def learning_curve(
    df: pd.DataFrame,
    *,
    position: str = "position",
    score: str = "reward",
    by: list[str] | None = None,
    window: int | None = None,
) -> pd.DataFrame:
    """Score over stream position: the continual-learning progress curve.

    Expects columns: *position*, *score*, plus any *by* group columns
    (typically ``stream``). Returns the input sorted by position with a
    ``cum_mean`` column (expanding mean within each group) and, when
    *window* is given, a ``rolling_mean`` over that many positions.
    """
    out = df.sort_values((by or []) + [position]).copy()
    if by:
        grouped = out.groupby(by)[score]
        out["cum_mean"] = grouped.transform(lambda s: s.expanding().mean())
        if window is not None:
            out["rolling_mean"] = grouped.transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            )
    else:
        out["cum_mean"] = out[score].expanding().mean()
        if window is not None:
            out["rolling_mean"] = out[score].rolling(window, min_periods=1).mean()
    return out


def transfer(
    stream_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    *,
    score: str = "reward",
    on: str = "task",
) -> pd.DataFrame:
    """What carrying state was worth: stream score minus isolated score.

    Expects columns *on* and *score* in both frames: *stream_df* holds
    episodes run inside a stream, *baseline_df* the same tasks run
    isolated (a plain ``lab.run`` sweep). Returns one row per stream task
    with mean ``stream`` and ``isolated`` scores and their difference
    ``transfer``; tasks with no baseline get NaN.

    This is CL-Bench's gain metric, not the forward transfer of the
    continual-learning literature: that one scores a task the agent has
    not worked on yet, while both frames here score a task the agent did
    work on, differing only in whether earlier tasks came first.
    """
    s = stream_df.groupby(on)[score].mean().rename("stream").reset_index()
    b = baseline_df.groupby(on)[score].mean().rename("isolated").reset_index()
    out = s.merge(b, on=on, how="left")
    out["transfer"] = out["stream"] - out["isolated"]
    return out


def forgetting(
    df: pd.DataFrame,
    *,
    position: str = "position",
    score: str = "reward",
    task: str = "task",
) -> pd.DataFrame:
    """How much revisited tasks degraded over a stream.

    Expects columns: *task*, *position*, *score*. For each task seen at
    two or more positions: the best score among the earlier visits minus
    the score at the last visit (positive means the agent forgot). Tasks
    visited once are excluded. Returns one row per revisited task with
    its ``first``/``last`` positions and ``forgetting``.
    """
    rows = []
    for name, g in df.sort_values(position).groupby(task):
        if len(g) < 2:
            continue
        rows.append(
            {
                task: name,
                "first": g[position].iloc[0],
                "last": g[position].iloc[-1],
                "forgetting": g[score].iloc[:-1].max() - g[score].iloc[-1],
            }
        )
    return pd.DataFrame(rows, columns=[task, "first", "last", "forgetting"])


# -------------------------------------------------------------- normalizers


def rescale_linear(s: pd.Series, *, lo: float, hi: float) -> pd.Series:
    """Map raw scores to 0-100 linearly, clipped: ``lo`` becomes 0 and
    ``hi`` becomes 100."""
    if hi == lo:
        raise ValueError("rescale_linear needs hi != lo")
    return ((s - lo) / (hi - lo) * 100).clip(0, 100)


def rescale_anchored(
    s: pd.Series,
    *,
    baseline: float,
    top: float,
    super_anchor: float | None = None,
) -> pd.Series:
    """Two-segment rescale: ``baseline`` becomes 0 and ``top`` becomes 100.
    With ``super_anchor``, scores above ``top`` stretch linearly instead of
    clipping, so beating the best known result stays visible; a score at
    ``super_anchor`` lands on 200.

    EdgeBench's own normalization is a different, four-anchor curve capped
    at 100. Its converted tasks carry all four values (``rescale_baseline``,
    ``rescale_rank30``, ``rescale_rank1``, ``rescale_super_anchor``) in
    ``task.toml``, so reproducing it exactly is a task-side calculation, not
    this function."""
    scaled = rescale_linear(s, lo=baseline, hi=top)
    if super_anchor is not None:
        over = s > top
        scaled[over] = 100 + (s[over] - top) / (super_anchor - top) * 100
    return scaled
