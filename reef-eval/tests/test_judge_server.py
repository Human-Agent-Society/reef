"""The judge: scoring, the submission budget, and the final grade.

Exercises the template's generic ``judge_server.py`` directly (no HTTP:
the HTTP path is covered by the LocalExecutor tests, which run the same
file as a real server).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TEMPLATE_JUDGE = Path(__file__).parent.parent / "tasks" / "_template" / "environment"


def make_judge(monkeypatch, tmp_path, judge_dir=TEMPLATE_JUDGE):
    monkeypatch.setenv("JUDGE_DIR", str(judge_dir))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    spec = importlib.util.spec_from_file_location(
        f"judge_server_{tmp_path.name}", TEMPLATE_JUDGE / "judge_server.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Judge()


def submit(judge, payload: dict):
    return judge.submit(json.dumps(payload).encode())


def test_submissions_are_scored_and_best_tracked(monkeypatch, tmp_path):
    judge = make_judge(monkeypatch, tmp_path)
    code, first = submit(judge, {"x": 0.5})
    assert (code, first["score"]) == (200, 0.5)
    code, second = submit(judge, {"x": 0.3})
    assert (code, second["score"], second["best"]) == (200, 0.3, 0.5)

    final = judge.final_result()
    assert final["reward"] == 0.5  # no final.py → the best session score
    assert final["best_n"] == 0
    assert [e["score"] for e in final["submissions"]] == [0.5, 0.3]


def test_invalid_submissions_score_zero_not_crash(monkeypatch, tmp_path):
    judge = make_judge(monkeypatch, tmp_path)
    code, result = submit(judge, {"nonsense": True})
    assert (code, result["score"]) == (200, 0.0)
    assert result["reason"]


def test_submission_limit_is_enforced(monkeypatch, tmp_path):
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    (judge_dir / "score.py").write_text((TEMPLATE_JUDGE / "score.py").read_text())
    (judge_dir / "judge_config.json").write_text(json.dumps({"max_submissions": 2}))
    judge = make_judge(monkeypatch, tmp_path, judge_dir)

    assert submit(judge, {"x": 0.1})[0] == 200
    assert submit(judge, {"x": 0.2})[0] == 200
    code, refused = submit(judge, {"x": 0.9})
    assert code == 429 and "limit" in refused["error"]
    assert judge.final_result()["reward"] == 0.2  # the refused 0.9 never counted


def test_final_judge_overrides_session_score(monkeypatch, tmp_path):
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    (judge_dir / "score.py").write_text((TEMPLATE_JUDGE / "score.py").read_text())
    # A final judge with a "hidden test": it also rejects x > 0.8.
    (judge_dir / "final.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "def grade(artifact):\n"
        "    x = float(json.loads(Path(artifact).read_text())['x'])\n"
        "    if x > 0.8:\n"
        "        return {'reward': 0.0, 'reason': 'hidden test: x too large'}\n"
        "    return {'reward': x, 'reason': 'ok'}\n"
    )
    judge = make_judge(monkeypatch, tmp_path, judge_dir)
    submit(judge, {"x": 0.9})  # session says 0.9, best by session score
    submit(judge, {"x": 0.6})

    final = judge.final_result()
    assert final["reward"] == 0.0  # the hidden test caught the best submission
    assert "hidden test" in final["reason"]


def test_no_submissions_means_zero(monkeypatch, tmp_path):
    judge = make_judge(monkeypatch, tmp_path)
    final = judge.final_result()
    assert final["reward"] == 0.0 and final["submissions"] == []


@pytest.mark.skipif(sys.platform == "win32", reason="posix paths in template")
def test_min_interval_is_enforced(monkeypatch, tmp_path):
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    (judge_dir / "score.py").write_text((TEMPLATE_JUDGE / "score.py").read_text())
    (judge_dir / "judge_config.json").write_text(json.dumps({"min_interval_sec": 3600}))
    judge = make_judge(monkeypatch, tmp_path, judge_dir)
    assert submit(judge, {"x": 0.1})[0] == 200
    code, refused = submit(judge, {"x": 0.2})
    assert code == 429 and "retry_after_sec" in refused


def test_finalize_is_terminal_and_idempotent(monkeypatch, tmp_path):
    judge = make_judge(monkeypatch, tmp_path)
    submit(judge, {"x": 0.7})
    first = judge.final_result()
    assert first["reward"] == 0.7

    # peeking again returns the cached grade, not a re-run
    assert judge.final_result() is first
    # and the session is over: no more submissions
    code, refused = submit(judge, {"x": 0.9})
    assert code == 429 and "finalized" in refused["error"]
    assert judge.final_result()["reward"] == 0.7
