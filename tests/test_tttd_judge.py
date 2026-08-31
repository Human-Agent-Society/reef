from __future__ import annotations

import importlib.util
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
JUDGE_SERVERS = (
    REPO_ROOT / "recipes/tttd/examples/tttd/harbor/erdos_min_overlap/environment/judge_server.py",
    REPO_ROOT / "recipes/tttd/examples/tttd/harbor/circle_packing_26/environment/judge_server.py",
    REPO_ROOT / "recipes/tttd/examples/tttd/harbor/circle_packing_32/environment/judge_server.py",
)


def _load_judge_server(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("server_path", JUDGE_SERVERS)
def test_judge_scores_independent_submissions_concurrently(monkeypatch, tmp_path, server_path):
    data_dir = tmp_path / server_path.parents[1].name
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    module = _load_judge_server(server_path, f"test_{server_path.parents[1].name}_judge")
    judge = module.Judge()

    active = 0
    max_active = 0
    active_lock = threading.Lock()
    all_scorers_entered = threading.Barrier(4)

    def score(path):
        nonlocal active, max_active
        with active_lock:
            active += 1
            max_active = max(max_active, active)
        all_scorers_entered.wait(timeout=2)
        time.sleep(0.01)
        with active_lock:
            active -= 1
        value = float(path.read_text())
        return {"reward": value, "reason": "ok"}

    judge.score = score
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda value: judge.submit(str(value).encode()), range(1, 5)))

    assert max_active == 4
    assert sorted(payload["n"] for status, payload in responses if status == 200) == [0, 1, 2, 3]
    expected_remaining = None if judge.max_submissions is None else judge.max_submissions - 4
    assert judge.status() == {"used": 4, "remaining": expected_remaining, "best": 4.0, "pending": 0}
    assert judge.final_result()["reward"] == 4.0
