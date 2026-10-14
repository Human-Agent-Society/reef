"""Aligned live check results shared by the harness wrapper and tutorial driver."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text


class CheckDisplay:
    """Refresh a TTY in place; print bounded state changes and one final table to a pipe."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self.spinners: dict[str, Spinner] = {}
        self.checks: Sequence[Mapping[str, object]] = ()
        self.phase = "queued"
        self.started_at = time.monotonic()
        self.activity = ""
        self.last_status: dict[str, str] = {}
        self.running_since: dict[str, float] = {}
        self.live = Live(console=self.console, refresh_per_second=8, transient=False)

    def __enter__(self) -> CheckDisplay:
        if self.console.is_terminal:
            self.live.start()
        return self

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:
        if self.console.is_terminal:
            self.live.stop()
        elif self.checks:
            self.console.print(self.render())

    def update(self, phase: str, checks: object, activity: object = None) -> None:
        self.phase = phase
        if isinstance(activity, (tuple, list)) and activity and isinstance(activity[-1], Mapping):
            self.activity = " ".join(str(activity[-1].get("text", "")).split())[:160]
        if isinstance(checks, (tuple, list)):
            self.checks = [check for check in checks if isinstance(check, Mapping)]
        for check in self.checks:
            name = str(check.get("id", "check"))
            if check.get("status") == "running":
                self.running_since.setdefault(name, time.monotonic())
            else:
                self.running_since.pop(name, None)
        if self.console.is_terminal:
            self.live.update(self.render(), refresh=True)
        else:
            if not self.checks:
                return
            rows = self.checks or [{"id": "step", "status": phase, "reason": ""}]
            for check in rows:
                name = str(check.get("id", "check"))
                status = str(check.get("status", "pending"))
                if self.last_status.get(name) != status:
                    status_line = f"{name}: {status} {check.get('reason') or check.get('observed') or ''}"
                    self.console.print(Text(" ".join(status_line.split())[:300]))
                    self.last_status[name] = status

    def render(self) -> Table:
        elapsed = int(time.monotonic() - self.started_at)
        table = Table(
            title=f"Harness evolution | {self.phase} | {elapsed // 60:02d}:{elapsed % 60:02d}",
            caption=Text(self.activity),
            expand=True,
            show_lines=False,
        )
        table.add_column("Check", ratio=2, no_wrap=True, overflow="ellipsis")
        table.add_column("Status", width=10)
        table.add_column("Current", width=7, justify="right")
        table.add_column("Candidate", width=9, justify="right")
        if self.console.width >= 90:
            table.add_column("Expected", ratio=3, overflow="ellipsis", no_wrap=True)
        table.add_column("Observed / reason", ratio=4, overflow="ellipsis", no_wrap=True)
        if self.console.width >= 90:
            table.add_column("Seconds", width=7, justify="right")
        checks = self.checks or [
            {"id": "proposal", "group": "proposal", "status": "running" if self.phase == "proposing" else self.phase}
        ]
        for check in checks:
            name = str(check.get("id", "check"))
            status = str(check.get("status", "pending"))
            status_cell: Spinner | Text
            if status == "running":
                spinner = self.spinners.get(name)
                if spinner is None:
                    style = "dots" if "utf" in self.console.encoding.lower() else "line"
                    spinner = Spinner(style, text="running")
                    self.spinners[name] = spinner
                status_cell = spinner
            else:
                status_cell = Text(status)
            cells: list[Spinner | Text] = [
                Text(name),
                status_cell,
                Text(score_text(check.get("current_score"))),
                Text(score_text(check.get("score"))),
            ]
            if self.console.width >= 90:
                cells.append(Text(" ".join(str(check.get("expected", "")).split())))
            cells.append(Text(" ".join(str(check.get("reason") or check.get("observed") or "").split())))
            if self.console.width >= 90:
                seconds = check.get("seconds")
                if status == "running":
                    seconds = time.monotonic() - self.running_since.get(name, time.monotonic())
                cells.append(Text(f"{seconds:.1f}" if isinstance(seconds, (int, float)) else "—"))
            table.add_row(*cells)
        return table


def score_text(value: object) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) else "—"
