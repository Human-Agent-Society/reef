"""EdgeBench conversion tests against a real published spec.

The fixtures are unmodified files from the EdgeBench HuggingFace dataset
(``ann_vector_search_qps.json`` + ``BENCHMARK.yaml``), so these tests pin the
conversion to the actual published format, not our idea of it.
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML not installed")

from tasks.autoresearch.edgebench.convert import convert_task  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "edgebench"


@pytest.fixture()
def task_dir(tmp_path):
    return convert_task(
        FIXTURES / "ann_vector_search_qps.json",
        FIXTURES / "BENCHMARK.yaml",
        tmp_path,
    )


def test_layout_is_a_complete_harbor_task(task_dir):
    assert (task_dir / "task.toml").exists()
    assert (task_dir / "instruction.md").exists()
    assert (task_dir / "environment" / "Dockerfile").exists()
    assert (task_dir / "tests" / "Dockerfile").exists()
    assert (task_dir / "tests" / "test.sh").exists()
    assert (task_dir / "tests" / "parse_score.py").exists()


def test_instruction_is_the_agent_query(task_dir):
    text = (task_dir / "instruction.md").read_text()
    assert "QPS" in text and "Recall@10" in text


def test_prebuilt_images_reference_the_registry(task_dir):
    env = (task_dir / "environment" / "Dockerfile").read_text()
    judge = (task_dir / "tests" / "Dockerfile").read_text()
    assert env.startswith(
        "FROM seededge/edgebench.work.ann_vector_search_qps:74bf3ba9a919"
    )
    assert judge.startswith(
        "FROM seededge/edgebench.judge.ann_vector_search_qps:9171f6b6de41"
    )
    assert "WORKDIR /home/workspace/ann-benchmarks" in env


def test_task_toml_maps_spec_faithfully(task_dir):
    import tomllib

    cfg = tomllib.loads((task_dir / "task.toml").read_text())
    assert cfg["artifacts"] == [
        "/home/workspace/ann-benchmarks/ann_benchmarks/algorithms/custom"
    ]
    assert cfg["verifier"]["environment_mode"] == "separate"
    assert cfg["verifier"]["timeout_sec"] == 7800.0
    assert cfg["environment"]["network_mode"] == "no-network"  # internet: false
    assert cfg["metadata"]["rescale_kind"] == "log_max"
    assert cfg["metadata"]["rescale_baseline"] == 3108.0


def test_task_validates_under_stock_harbor(task_dir):
    pytest.importorskip("harbor")
    import tomllib

    from harbor.models.task.config import TaskConfig

    TaskConfig.model_validate(tomllib.loads((task_dir / "task.toml").read_text()))


def test_test_sh_runs_eval_cmd_and_checks_submission(task_dir):
    sh = (task_dir / "tests" / "test.sh").read_text()
    assert "python3 /tmp/eval_ann_search.py" in sh
    assert "submission missing" in sh
    assert "parse_score.py" in sh


def test_parse_score_extracts_both_formats(task_dir):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "parse_score", task_dir / "tests" / "parse_score.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    score, _ = mod.extract("blah\nTOTAL_SCORE 21472.5\n")
    assert score == 21472.5
    score, _ = mod.extract(
        '>>>>> Start Structured Result\n{"score": 12.5, "valid": true}\n'
        ">>>>> End Structured Result\n"
    )
    assert score == 12.5
    score, reason = mod.extract("no score here")
    assert score is None and "no structured score" in reason
