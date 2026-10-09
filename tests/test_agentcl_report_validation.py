"""Reject changed persisted training reports before resume or independent verification."""

import asyncio
import importlib
from pathlib import Path

import pytest

from tests import test_opd_agentcl as opd_workflow
from tests import test_sdft_agentcl as sdft_workflow
from tests import test_sdpo_agentcl as sdpo_workflow


@pytest.fixture(params=[opd_workflow, sdft_workflow, sdpo_workflow], ids=["opd", "sdft", "sdpo"])
def workflow(request):
    return request.param


def setup_task(workflow, tmp_path: Path):
    arguments = workflow.arguments_fixture(tmp_path)
    manifest = workflow.manifest_fixture(arguments.data_root)
    if workflow is opd_workflow:
        api = workflow.FakeApi()
        backend = workflow.FakeEpisodes(api)
    else:
        api = workflow.FakeApi(attempts=arguments.attempts)
        backend = workflow.FakeEpisodes()
    campaign = workflow.run.Campaign(arguments, api, backend, manifest)
    positions = workflow.run.selected_training_positions(arguments, manifest)
    task = {**manifest["training"][positions[0]], "campaign_position": 0}
    state = {"started_episodes": [], "report_attempted": [], "complete": False}
    episodes = [
        asyncio.run(campaign.episode(task, "train", attempt, api.current_release(), state))
        for attempt in range(arguments.attempts)
    ]
    return arguments, manifest, api, backend, campaign, task, state, episodes


def report_payload(workflow, arguments, task, episode):
    if workflow is opd_workflow:
        context = ""
    elif workflow is sdft_workflow:
        context = (arguments.data_root / task["reference_path"]).read_text()
    else:
        context = "The submitted solution passed the verifier."
    return workflow.report.terminal_report(workflow.run.METHOD, episode, context, 1)


@pytest.mark.parametrize("field", ["score", "teacher_context", "report_id", "grid"])
def test_changed_persisted_report_stops_before_submission(workflow, tmp_path, field):
    if field == "grid" and workflow is not sdpo_workflow:
        pytest.skip("Only SDPO reports carry grid coordinates")
    arguments, _, api, backend, campaign, task, state, episodes = setup_task(workflow, tmp_path)
    payload = report_payload(workflow, arguments, task, episodes[0])
    if field == "score":
        payload["score"] = 1.0 - float(payload["score"])
    elif field == "teacher_context":
        payload["metadata"]["teacher_context"] = "unverified replacement"
    elif field == "report_id":
        payload["agent_record_id"] = "different-id"
    else:
        payload["metadata"]["rollout"] = 99
    workflow.report.write_object(arguments.run_root / "reports" / (episodes[0]["report_id"] + ".json"), payload)
    calls = len(backend.calls)
    with pytest.raises(RuntimeError, match="persisted report differs"):
        asyncio.run(campaign.commit_task(task, episodes, api.current_release(), state))
    assert not api.reports
    assert not api.history
    assert len(backend.calls) == calls


def test_changed_report_after_commit_is_rejected_without_inference_replay(workflow, tmp_path):
    arguments, _, api, backend, campaign, task, state, episodes = setup_task(workflow, tmp_path)
    initial = api.current_release()
    asyncio.run(campaign.commit_task(task, episodes, initial, state))
    path = arguments.run_root / "reports" / (episodes[0]["report_id"] + ".json")
    payload = workflow.report.read_object(path)
    payload["score"] = 1.0 - float(payload["score"])
    workflow.report.write_object(path, payload)
    calls = len(backend.calls)
    campaign.cursor["training_step"] = 0
    with pytest.raises(RuntimeError, match="persisted report differs"):
        asyncio.run(campaign.commit_task(task, episodes, initial, state))
    assert len(api.history) == 1
    assert len(backend.calls) == calls


@pytest.mark.parametrize("field", ["score", "teacher_context"])
def test_independent_verification_rejects_report_content_substitution(workflow, tmp_path, field):
    arguments = workflow.arguments_fixture(tmp_path)
    manifest = workflow.manifest_fixture(arguments.data_root)
    if workflow is opd_workflow:
        api = workflow.FakeApi()
        backend = workflow.FakeEpisodes(api)
    else:
        api = workflow.FakeApi(attempts=arguments.attempts)
        backend = workflow.FakeEpisodes()
    campaign = workflow.run.Campaign(arguments, api, backend, manifest)
    for phase in ("baseline", "train", "frozen-repeat", "independent"):
        asyncio.run(campaign.execute_phase(phase))
    path = next((arguments.run_root / "reports").glob("*.json"))
    payload = workflow.report.read_object(path)
    if field == "score":
        payload["score"] = 1.0 - float(payload["score"])
    else:
        payload["metadata"]["teacher_context"] = "unverified replacement"
    workflow.report.write_object(path, payload)
    verification = importlib.import_module(f"recipes.{workflow.run.METHOD}.examples.agentcl.verification")
    result = verification.verify_run(arguments.run_root)
    assert not result["local_checks_passed"]
    assert any("terminal report differs" in failure for failure in result["failures"])
