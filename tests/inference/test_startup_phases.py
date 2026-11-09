import pytest

from reef.inference.startup import StartupStep, StartupSteps, report_startup, reported_steps


def entry(artifact_id: str, phase: str, at: float, detail: str | None = None, estimate: float | None = None) -> dict:
    return {
        "startup": {
            "artifact_id": artifact_id,
            "phase": phase,
            "detail": detail,
            "estimate_seconds": estimate,
            "at": at,
        }
    }


@pytest.mark.unit
def test_reports_become_the_phases_of_their_artifact() -> None:
    progress = [
        entry("a", "prepare", 11.0, "Quantizing to NVFP4"),
        entry("b", "start", 12.0, "another artifact"),
        entry("a", "start", 20.0, "Waiting for a B200"),
        entry("a", "start", 25.0, "Downloading weights", 400.0),
        entry("a", "prepare", 26.0, "late"),
        {"other": 1},
    ]

    running = reported_steps(progress, "a", 10.0, None)
    assert running == StartupSteps(
        (StartupStep("prepare", 10.0, 20.0, "Quantizing to NVFP4"), StartupStep("start", 20.0, None, "late")),
        None,
    )
    ended = reported_steps(progress[:4], "a", 10.0, 30.0)
    assert ended.steps[-1] == StartupStep("start", 20.0, 30.0, "Downloading weights")
    assert ended.estimate_seconds == 400.0
    assert reported_steps(progress, "c", 10.0, None).steps == (StartupStep("prepare", 10.0),)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("steps", "estimate", "message"),
    [
        ((("idle", 1.0, None),), None, "one of"),
        ((("prepare", 2.0, 1.0),), None, "ends at or after"),
        ((("start", 1.0, 2.0), ("prepare", 2.0, None)), None, "distinct phases"),
        ((("prepare", 2.0, 3.0), ("start", 1.0, None)), None, "start in the order"),
        ((("prepare", 1.0, None), ("start", 2.0, None)), None, "only the last"),
        ((("prepare", 1.0, None),), -1.0, "at least 0"),
    ],
)
def test_startup_steps_refuse_a_value_outside_their_definition(steps, estimate, message) -> None:
    with pytest.raises(ValueError, match=message):
        StartupSteps(tuple(StartupStep(phase, start, end) for phase, start, end in steps), estimate)


@pytest.mark.unit
def test_a_report_outside_a_serving_work_does_nothing_and_a_bad_phase_raises() -> None:
    report_startup("load", "Compiling kernels")
    with pytest.raises(ValueError, match="one of"):
        report_startup("compile")
