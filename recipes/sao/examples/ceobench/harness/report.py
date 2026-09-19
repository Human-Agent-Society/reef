"""The reward: each finished week of a CEO-Bench episode, credited and posted to Reef.

The harness (``harness.harbor_agent``) plays the benchmark's agent turn by
turn and knows the engine's day at every model call, so :class:`WeekRecords`
files each captured turn under its simulated week with the state the week
started in (:class:`WeekStart`, read from the engine's dashboard and books).
When enough later weeks have opened the week is scored, and
:func:`post_week_reports` turns it into one Reef report per decision turn:
the week's credit as the score, that turn's receipt as the only reference.
One reference per report is what the ``sao`` recipe trains on.
:class:`TrainingPacer` holds the game until the trainer has consumed the
turns reported so far.

A week's opening state is valued as its cash plus its subscription run-rate
(the engine's own monthly recurring revenue, or the dashboard's estimate
when the books cannot be read), counted over the weeks left in the episode
and at most ``horizon_weeks`` of them (:func:`valuation`). Cash alone made
every purchase a loss and inaction the safest week; the run-rate term is what
pays for growth inside the horizon.

A week's credit is the discounted sum of the value changes of the next
``credit_weeks`` weeks (:func:`week_credit`): the week that spends on
acquisition is credited with the subscribers that arrive over the weeks
after it. The last weeks of an episode end with the final cash the engine
reports, valued as cash alone, and their sums are cut short there.

Credits are posted through :class:`ScoreScale`, which clips one week's
outliers (the benchmark's six-figure R&D purchases) and divides by the
running median magnitude, so an ordinary week's difference from the one
before it keeps a gradient instead of vanishing next to the outliers.

Only a week's decision turns are reported, the ones whose tool call changed
the company; a turn that only read is recorded and left out, so the week's
outcome lands on the decisions in it. A turn longer than the trainer's
window (``max_tokens``, prompt and completion together) is skipped too: the
engine served it and Reef recorded it, but the trainer could not hold it,
so it stays evaluation-only.
"""

import json
import logging
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypedDict

from reef_client import ReefClient

#: The benchmark's starting balance (the runner's ``--cash`` default).
INITIAL_CASH = 1_000_000.0
#: Weeks of subscription run-rate a week's opening state is valued at, at most.
DEFAULT_VALUE_HORIZON_WEEKS = 26
#: Subscriptions bill every 30 days; a simulated week is this share of a bill.
DAYS_PER_WEEK, DAYS_PER_BILLING_MONTH = 7, 30
#: Weeks of value change a week is credited with, and the discount per week.
DEFAULT_CREDIT_WEEKS = 4
DEFAULT_CREDIT_DISCOUNT = 0.8
#: A week's credit is clipped to this magnitude (in units of the starting
#: balance) before scaling, and the scale never drops below the floor.
DEFAULT_SCORE_CLIP = 0.05
DEFAULT_SCORE_FLOOR = 0.003
#: Scaled scores stay within this magnitude.
SCORE_CAP = 3.0


def valuation(cash: float, run_rate: float, remaining_weeks: int, horizon_weeks: int) -> float:
    """Cash plus the monthly ``run_rate`` over the weeks left, at most ``horizon_weeks`` of them."""
    weeks = max(0, min(remaining_weeks, horizon_weeks))
    return cash + run_rate * DAYS_PER_WEEK * weeks / DAYS_PER_BILLING_MONTH


def week_credit(deltas: Sequence[float], discount: float, initial_cash: float = INITIAL_CASH) -> float:
    """The discounted sum of the value changes in ``deltas``, in units of the starting balance.

    ``deltas[0]`` is the week's own change, ``deltas[1]`` the next week's, and
    so on; a shorter sequence is an episode that ended inside the window.
    """
    return sum(delta * discount**position for position, delta in enumerate(deltas)) / initial_cash


class ScoreScale:
    """Scale each week's credit by the running median magnitude of the credits so far.

    The credit is clipped to ``clip`` first, so one six-figure purchase does
    not set the scale for the rest of the episode; the scale never drops
    below ``floor``, so a run of near-zero weeks does not blow small noise up
    to the cap. Scaling keeps the sign: a week that grew the company scores
    positive whatever the weeks around it did.
    """

    def __init__(self, clip: float = DEFAULT_SCORE_CLIP, floor: float = DEFAULT_SCORE_FLOOR) -> None:
        if clip <= 0 or floor <= 0:
            raise ValueError("score clip and floor must be positive")
        self.clip = float(clip)
        self.floor = float(floor)
        self.magnitudes: list[float] = []

    def scale(self, credit: float) -> float:
        clipped = max(-self.clip, min(self.clip, credit))
        self.magnitudes.append(abs(clipped))
        scale = max(statistics.median(self.magnitudes), self.floor)
        return max(-SCORE_CAP, min(SCORE_CAP, clipped / scale))


def post_week_reports(
    client: ReefClient,
    scenario: str,
    *,
    week: int,
    day: int,
    cash_start: float,
    cash_end: float,
    value_start: float,
    value_end: float,
    credit: float,
    score: float,
    turns: Sequence[tuple[str, int, str | None]],
    max_tokens: int = 0,
) -> list[dict]:
    """Report one finished week against the receipts of its decision turns.

    ``turns`` are ``(receipt, tokens, decision)`` triples in call order, the
    decision being the company-changing call the turn made or ``None`` when
    it only read; ``score`` is the scaled ``credit`` and is what Reef trains
    on, the rest travels along for the record.
    """
    decisions = sum(1 for receipt, tokens, decision in turns if decision is not None)
    feedback = (
        f"ceobench week {week} (from day {day}): credit {credit:.4f}, score {score:.2f};"
        f" value {value_start:.0f} -> {value_end:.0f} (cash {cash_start:.0f} -> {cash_end:.0f})"
        f" over {decisions} decision turns of {len(turns)}"
    )
    posted = []
    for index, (receipt, tokens, decision) in enumerate(turns):
        if decision is None or (max_tokens and tokens > max_tokens):
            continue
        payload = {
            # A report id derived from the receipt makes a duplicate post a
            # no-op on Reef's side, not a second report about the same turn.
            "agent_record_id": uuid.uuid5(uuid.NAMESPACE_URL, f"reef:ceobench:{receipt}").hex,
            "score": score,
            "feedback": feedback,
            "references": [receipt],
            "metadata": {
                "ceobench": {
                    "week": week,
                    "day": day,
                    "cash_start": cash_start,
                    "cash_end": cash_end,
                    "value_start": value_start,
                    "value_end": value_end,
                    "credit": credit,
                    "decision": decision,
                    "turn": index,
                    "turns": len(turns),
                    "decisions": decisions,
                }
            },
        }
        posted.append(client.report(scenario, payload))
    return posted


#: How often the pacer re-reads the scenario's releases while it holds the game.
PACE_POLL_S = 10.0
#: The weekly dashboard the engine returns: the header, then the week's
#: opening cash, individual subscribers, and enterprise seats.
DASHBOARD_RE = re.compile(
    r"=== Week (\d+) Dashboard \(Day (\d+)\) ===\s*\n\s*\n"
    r"Cash: (-?)\$(-?[\d,]+)\n"
    r"Individual Subscribers: (\d+)\n"
    r"Enterprise Subscribed Seats: (\d+)"
)
#: The listed plan prices in the same dashboard's configuration block.
PRICES_RE = re.compile(r"--- Current Config ---\s*\nPrices: A=\$(\d+), B=\$(\d+), C=\$(\d+)")
#: SDK calls and CLI commands that change the company: money spent, prices,
#: targeting, research, deals, posts, and the week advanced. Everything else
#: the agent can do (queries, status, reading docs, files in its workspace)
#: only reads, and such turns are recorded but not trained on: a week's
#: outcome is credited to the decisions in it, not to looking at the books.
DECISION_CALLS = (
    "next-week",
    "next_week",
    "set_prices",
    "set_promotion",
    "set_lead_promotion",
    "set_model_tiers",
    "set_usage_quotas",
    "set_capacity_tier",
    "set_daily_spend",
    "set_targeted_ad_spend",
    "set_targeted_dev_spend",
    "set_targeted_ops_spend",
    "set_ads_strength",
    "start_research_project",
    "research_market",
    "research_group",
    "send_enterprise_deal",
    "reject_enterprise_deal",
    "post_social_media",
)
DECISION_RE = re.compile(r"\b(" + "|".join(re.escape(call) for call in DECISION_CALLS) + r")\b")
#: A bash command that runs a Python script file (not inline ``python -c`` code).
SCRIPT_RUN_RE = re.compile(r"\bpython3?\s+(?!-c\b)(\S+\.py)\b")


@dataclass(frozen=True)
class WeekStart:
    """A week's opening state as its dashboard shows it, plus the engine's own MRR when read."""

    week: int
    day: int
    cash: float
    subscribers: int
    seats: int
    prices: tuple[float, float, float]  # plans A, B, C as listed, monthly
    mrr: float | None = None

    @property
    def run_rate(self) -> float:
        """Monthly subscription revenue: the engine's MRR when read, else an estimate from the dashboard.

        The estimate counts each individual subscriber at the lowest nonzero
        listed price and each enterprise seat at plan C's; the dashboard shows
        neither the plan mix nor negotiated seat prices, so it is a floor.
        """
        if self.mrr is not None:
            return self.mrr
        listed = [price for price in self.prices if price > 0]
        return self.subscribers * (min(listed) if listed else 0.0) + self.seats * self.prices[2]


@dataclass
class WeekRecord:
    """The opening state and captured decision turns of one simulated week."""

    start: WeekStart
    turns: list[tuple[str, int, str | None]] = field(default_factory=list)


class TurnRecord(TypedDict):
    """A successfully served turn linked to its receipt and simulated week."""

    receipt: str
    tokens: int
    week: int
    decision: str | None


def dashboards(content: str) -> list[WeekStart]:
    """Every dashboard in ``content``, in order of appearance.

    The prices are looked for between a dashboard's header and the next
    one's; a dashboard without its configuration block keeps its week and
    cash and lists no prices.
    """
    starts = []
    matches = list(DASHBOARD_RE.finditer(content))
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(content)
        priced = PRICES_RE.search(content, match.end(), end)
        prices = tuple(float(price) for price in priced.groups()) if priced else (0.0, 0.0, 0.0)
        sign = -1.0 if match.group(3) == "-" else 1.0
        starts.append(
            WeekStart(
                week=int(match.group(1)),
                day=int(match.group(2)),
                cash=sign * float(match.group(4).replace(",", "")),
                subscribers=int(match.group(5)),
                seats=int(match.group(6)),
                prices=(prices[0], prices[1], prices[2]),
            )
        )
    return starts


def tool_calls(turn: dict) -> list[tuple[str, dict]]:
    """``(tool name, arguments)`` of every tool call in a captured turn's response."""
    calls = []
    for choice in (turn.get("response") or {}).get("choices") or []:
        message = choice.get("message") if isinstance(choice, dict) else None
        for call in (message or {}).get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = {"raw": arguments}
            calls.append((str(function.get("name") or ""), arguments if isinstance(arguments, dict) else {}))
    return calls


def turn_decision(turn: dict, scripts: dict[str, str]) -> str | None:
    """The company-changing call a turn made, or ``None`` for a turn that only read.

    ``scripts`` holds the files the agent has written so far (path to
    content), updated here as ``write_file``/``edit_file`` calls pass, so a
    bash command that runs one of them is judged by what it contains; a
    script this episode did not write is taken to act.
    """
    for name, arguments in tool_calls(turn):
        if name in ("write_file", "edit_file"):
            path = str(arguments.get("path") or "")
            content = str(arguments.get("content") or arguments.get("new_string") or "")
            if path:
                scripts[path] = content if name == "write_file" else scripts.get(path, "") + "\n" + content
            continue
        if name != "bash":
            continue
        command = str(arguments.get("command") or "")
        found = DECISION_RE.search(command)
        if found:
            return found.group(1).replace("-", "_")
        for match in SCRIPT_RUN_RE.finditer(command):
            script = match.group(1)
            basename = script.rsplit("/", 1)[-1]
            content = next((body for path, body in scripts.items() if path.rsplit("/", 1)[-1] == basename), None)
            if content is None:
                return "script"
            found = DECISION_RE.search(content)
            if found:
                return found.group(1).replace("-", "_")
    return None


def turn_tokens(turn: dict) -> int:
    """Prompt plus completion tokens of one captured turn (0 when unreported)."""
    usage = (turn.get("response") or {}).get("usage") or {}
    return int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)


class WeekRecords:
    """The episode's weeks: each one's opening state and its turns, in call order."""

    def __init__(
        self,
        total_weeks: int = 0,
        horizon_weeks: int = DEFAULT_VALUE_HORIZON_WEEKS,
        credit_weeks: int = DEFAULT_CREDIT_WEEKS,
        discount: float = DEFAULT_CREDIT_DISCOUNT,
    ) -> None:
        if credit_weeks < 1:
            raise ValueError("credit_weeks must be at least 1")
        self.total_weeks = total_weeks
        self.horizon_weeks = horizon_weeks
        self.credit_weeks = credit_weeks
        self.discount = discount
        self.weeks: dict[int, WeekRecord] = {}
        self.turns: list[TurnRecord] = []
        self.posted: set[int] = set()
        self.scores: dict[int, float] = {}  # week -> the scaled score it was posted with
        self.scripts: dict[str, str] = {}

    def open_week(self, start: WeekStart) -> None:
        self.weeks.setdefault(start.week, WeekRecord(start))

    def add_turns(self, week: int, captured: list[dict]) -> list[TurnRecord]:
        """File captured turns under ``week``; return the records of the ones Reef served."""
        records = []
        for turn in captured:
            if turn.get("status") != 200 or not turn.get("receipt"):
                continue
            record: TurnRecord = {
                "receipt": turn["receipt"],
                "tokens": turn_tokens(turn),
                "week": week,
                "decision": turn_decision(turn, self.scripts),
            }
            self.turns.append(record)
            self.weeks[week].turns.append((record["receipt"], record["tokens"], record["decision"]))
            records.append(record)
        return records

    def value(self, start: WeekStart) -> float:
        """What a week's opening state is worth, given the weeks left after it."""
        return valuation(start.cash, start.run_rate, self.total_weeks - start.week, self.horizon_weeks)

    def closing(
        self, ordered: list[int], position: int, final_cash: float | None
    ) -> tuple[float, float, float] | None:
        """``(cash_end, value_end, credit)`` of the week at ``position``, or ``None`` while it is open.

        The week closes with the opening state of the next week seen and is
        credited with the value changes of the ``credit_weeks`` weeks from it
        on, so it stays open until that many later weeks have opened. With
        ``final_cash`` the episode is over: the last week seen ends there,
        valued as cash alone, and a window that reaches the end is cut short.
        """
        deltas: list[float] = []
        closing: tuple[float, float] | None = None
        for offset in range(self.credit_weeks):
            index = position + offset
            value = self.value(self.weeks[ordered[index]].start)
            if index + 1 < len(ordered):
                start = self.weeks[ordered[index + 1]].start
                cash_end, value_end = start.cash, self.value(start)
            elif final_cash is not None:
                cash_end, value_end = final_cash, final_cash
            else:
                return None
            deltas.append(value_end - value)
            closing = closing or (cash_end, value_end)
            if index + 1 >= len(ordered):
                break  # the episode ended inside the window
        if closing is None:
            return None
        return (*closing, week_credit(deltas, self.discount))

    def finished_weeks(self, final_cash: float | None = None) -> list[tuple[int, float, float, float]]:
        """Unreported weeks whose credit is known, as ``(week, cash_end, value_end, credit)``."""
        ordered = sorted(self.weeks)
        finished = []
        for position, week in enumerate(ordered):
            if week in self.posted:
                continue
            closing = self.closing(ordered, position, final_cash)
            if closing is not None:
                finished.append((week, *closing))
        return finished

    def summary(self, final_cash: float | None = None) -> list[dict]:
        ordered = sorted(self.weeks)
        rows = []
        for position, week in enumerate(ordered):
            entry = self.weeks[week]
            start: WeekStart = entry.start
            closing = self.closing(ordered, position, final_cash) or (None, None, None)
            rows.append(
                {
                    "week": week,
                    "day": start.day,
                    "cash_start": start.cash,
                    "cash_end": closing[0],
                    "subscribers": start.subscribers,
                    "seats": start.seats,
                    "run_rate": start.run_rate,
                    "value_start": self.value(start),
                    "value_end": closing[1],
                    "credit": closing[2],
                    "score": self.scores.get(week),
                    "turns": len(entry.turns),
                    "decisions": sum(1 for receipt, tokens, decision in entry.turns if decision is not None),
                    "reported": week in self.posted,
                }
            )
        return rows


class ReleaseCount(ABC):
    """Where the pacer reads how many training releases the scenario has committed."""

    @abstractmethod
    def training_releases(self) -> int | None:
        """The count so far, or ``None`` when the service could not answer just now."""


class ScenarioReleases(ReleaseCount):
    """The scenario's release list on the Reef service."""

    def __init__(self, service_url: str, scenario: str, token: str) -> None:
        self.url = f"{service_url}/reef/scenarios/{scenario}/releases"
        self.token = token

    def training_releases(self) -> int | None:
        request = urllib.request.Request(self.url, headers={"Authorization": f"Bearer {self.token}"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, ValueError):
            return None
        return sum(1 for row in payload.get("releases", []) if row.get("operation") == "training")


class TrainingPacer:
    """Hold the game until the trainer has consumed the turns reported so far.

    The reported turns fill batches of ``batch_size``; each full batch must
    have produced one training release. ``wait`` blocks until the scenario
    has that many releases beyond the count at episode start, or the timeout
    passes, in which case the shortfall is forgiven so later weeks do not
    wait for a batch the recipe declined.
    """

    def __init__(self, batch_size: int, timeout_s: float, releases: ReleaseCount, logger: logging.Logger) -> None:
        self.batch_size = int(batch_size)
        self.timeout_s = float(timeout_s)
        self.releases = releases
        self.logger = logger
        self.base: int | None = None
        self.forgiven = 0
        self.lock = threading.Lock()

    def expected_releases(self, posted_turns: int) -> int:
        return posted_turns // self.batch_size - self.forgiven

    def wait(self, week: int, posted_turns: int) -> float:
        """Block until the trainer caught up; return the seconds spent waiting."""
        started = time.time()
        with self.lock:
            if self.base is None:
                self.base = self.releases.training_releases() or 0
            expected = self.expected_releases(posted_turns)
            while True:
                observed = (self.releases.training_releases() or 0) - self.base
                if observed >= expected:
                    break
                if time.time() - started >= self.timeout_s:
                    self.forgiven += expected - observed
                    self.logger.warning(
                        "week %d: trainer committed %d of %d expected releases after %.0fs; going on",
                        week,
                        observed,
                        expected,
                        self.timeout_s,
                    )
                    break
                time.sleep(PACE_POLL_S)
        return time.time() - started
