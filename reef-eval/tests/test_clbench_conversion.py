"""CL-Bench conversion + scorer: real published fixtures, hand-computed IoU."""

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pytest

CLBENCH = Path(__file__).parent.parent / "tasks" / "continual-learning" / "cl-bench"
FIXTURES = Path(__file__).parent / "fixtures" / "clbench"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"clbench_{name}", CLBENCH / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


convert = _load("convert_bsm")
scorer = _load("score_bsm")
score_sales = _load("score_sales")
score_cohort = _load("score_cohort")
score_codebase = _load("score_codebase")
convert_dbx = _load("convert_dbx")
# dbx_server is loaded lazily inside the fixture below: importing it reads
# question.json, which only exists once the fixture has built the database.

METADATA = json.loads((FIXTURES / "mixed_grid_lifecycle_metadata.json").read_text())
SCANS = [
    json.loads(line)
    for line in (FIXTURES / "bsm_sample.jsonl").read_text().splitlines()
]


def _convert(scan, tmp_path):
    stage = convert.stage_for_scan(scan["scan_idx"], METADATA["stages"])
    return convert.convert_scan(scan, stage, tmp_path, total_scans=90)


# ---------------------------------------------------------------- converter


def test_convert_writes_a_complete_task(tmp_path):
    task_dir = _convert(SCANS[0], tmp_path)
    assert task_dir.name == "bsm-s01"
    for piece in (
        "task.toml",
        "instruction.md",
        "environment/Dockerfile",
        "tests/test.sh",
        "tests/score.py",
        "tests/truth.json",
        "solution/solve.sh",
    ):
        assert (task_dir / piece).exists(), piece
    config = tomllib.loads((task_dir / "task.toml").read_text())
    assert config["metadata"]["domain"] == "blind_spectrum_monitoring"
    assert config["metadata"]["scan_idx"] == 0


def test_instruction_matches_the_upstream_prompt_shape(tmp_path):
    scan = SCANS[0]
    text = (_convert(scan, tmp_path) / "instruction.md").read_text()
    assert "spectrum monitoring analyst" in text  # the system preamble
    assert f"--- Scan {scan['scan_idx'] + 1}/90 ---" in text
    for peak in scan["detected_peaks"]:
        assert peak["peak_id"] in text
    assert "Band: 0.0-168.0 MHz" in text  # stage 0 band width
    assert '"transmitters"' in text  # the exact answer schema
    assert "/app/report.json" in text


def test_stage_lookup_follows_metadata_ranges(tmp_path):
    later = SCANS[1]  # scan 40: stage 1, a different variant
    task_dir = _convert(later, tmp_path)
    assert task_dir.name == "bsm-s41"
    config = tomllib.loads((task_dir / "task.toml").read_text())
    assert config["metadata"]["stage_idx"] == 1
    assert config["metadata"]["variant_id"] == "five_plus_four_mixed"
    with pytest.raises(ValueError, match="outside"):
        convert.stage_for_scan(999, METADATA["stages"])


def test_convert_is_valid_stock_harbor(tmp_path):
    pytest.importorskip("harbor")
    from harbor.models.task.config import TaskConfig

    task_dir = _convert(SCANS[0], tmp_path)
    TaskConfig.model_validate(tomllib.loads((task_dir / "task.toml").read_text()))


def test_truth_includes_dormant_channels(tmp_path):
    """Upstream grades against every persistent channel, active or not."""
    scan = SCANS[0]
    truth = json.loads((_convert(scan, tmp_path) / "tests" / "truth.json").read_text())
    assert len(truth["channels"]) == len(scan["ground_truth"])
    assert any(not ch["active_this_scan"] for ch in truth["channels"])


# ------------------------------------------------------------------- scorer


def test_scorer_hand_computed_iou():
    truth = {"band_width": 100.0, "channels": [{"center_freq": 50, "bandwidth": 20}]}
    perfect = [{"center_freq": 50, "bandwidth": 20}]
    assert scorer.score_report(perfect, truth)["score"] == 1.0
    # Empty report: available = whole band (100); truth available = 80.
    # IoU = 80 / 100 = 0.8.
    assert scorer.score_report([], truth)["score"] == 0.8
    # Claiming the whole band occupied: no available overlap at all.
    whole = [{"center_freq": 50, "bandwidth": 100}]
    assert scorer.score_report(whole, truth)["score"] == 0.0


def test_reference_solution_scores_exactly_one(tmp_path):
    """The oracle contract: the truth-derived report must earn IoU 1.0."""
    for scan in SCANS:
        task_dir = _convert(scan, tmp_path)
        truth = json.loads((task_dir / "tests" / "truth.json").read_text())
        solve = (task_dir / "solution" / "solve.sh").read_text()
        report = json.loads(solve.split("<<'EOF'\n")[1].split("\nEOF")[0])
        assert scorer.score_report(report["transmitters"], truth)["score"] == 1.0


def test_malformed_reports_degrade_to_empty(tmp_path):
    bad = tmp_path / "report.json"
    bad.write_text("{not json")
    transmitters, reason = scorer.load_transmitters(str(bad))
    assert transmitters == [] and "malformed" in reason
    transmitters, reason = scorer.load_transmitters(str(tmp_path / "missing.json"))
    assert transmitters == [] and "no report" in reason


# ------------------------------------------------------------------- sales


SALES_TRUTH = [
    {"locality": "SF", "furniture_name": "Chair", "year": 2027, "items_sold": 100.0},
    {"locality": "NY", "furniture_name": "Chair", "year": 2027, "items_sold": 50.0},
]


def _sales_report(entries):
    return json.dumps({"predictions": entries})


def test_sales_wape_skill_hand_cases():
    perfect = [
        {"locality": "SF", "furniture_name": "Chair", "year": 2027, "items_sold": 100},
        {"locality": "NY", "furniture_name": "Chair", "year": 2027, "items_sold": 50},
    ]
    assert score_sales.score(_sales_report(perfect), SALES_TRUTH)["score"] == 1.0
    # Missing entries count as predicting zero: error = 50, volume = 150.
    partial = perfect[:1]
    result = score_sales.score(_sales_report(partial), SALES_TRUTH)
    assert result["score"] == pytest.approx(1 - 50 / 150)
    assert result["n_missing_counted_as_zero"] == 1
    # Catastrophic over-prediction goes negative (unclipped, as upstream).
    huge = [{**perfect[0], "items_sold": 1000.0}, perfect[1]]
    assert score_sales.score(_sales_report(huge), SALES_TRUTH)["score"] < 0
    # Duplicates: first wins.
    dup = [perfect[0], {**perfect[0], "items_sold": 0.0}, perfect[1]]
    assert score_sales.score(_sales_report(dup), SALES_TRUTH)["score"] == 1.0
    # Malformed JSON scores 0 with format_valid False.
    bad = score_sales.score("{nope", SALES_TRUTH)
    assert bad["score"] == 0.0 and not bad["format_valid"]


# ------------------------------------------------------------------ cohort


COHORT_TRUTH = {
    "reference_survival": [0.7, 0.4, 0.2],
    "cohorts": [
        {
            "cohort_id": "a",
            "layer": 1,
            "survival_12m": 0.9,
            "survival_24m": 0.6,
            "survival_36m": 0.3,
        },
        {
            "cohort_id": "b",
            "layer": 2,
            "survival_12m": 0.5,
            "survival_24m": 0.2,
            "survival_36m": 0.1,
        },
    ],
}


def test_cohort_perfect_scores_the_ceiling():
    perfect = {
        "estimates": [
            {
                "cohort_id": c["cohort_id"],
                "estimated_survival_12m": c["survival_12m"],
                "estimated_survival_24m": c["survival_24m"],
                "estimated_survival_36m": c["survival_36m"],
            }
            for c in COHORT_TRUTH["cohorts"]
        ]
    }
    result = score_cohort.score(json.dumps(perfect), COHORT_TRUTH)
    assert result["score"] == result["mean_reference_kl"] > 0
    assert result["mean_kl_divergence"] == 0.0


def test_cohort_reference_and_missing_score_zero():
    reference = {
        "estimates": [
            {
                "cohort_id": c["cohort_id"],
                "estimated_survival_12m": 0.7,
                "estimated_survival_24m": 0.4,
                "estimated_survival_36m": 0.2,
            }
            for c in COHORT_TRUTH["cohorts"]
        ]
    }
    assert score_cohort.score(json.dumps(reference), COHORT_TRUTH)["score"] == 0.0
    # Omitting everything backfills to the reference: also exactly zero.
    empty = score_cohort.score("{}", COHORT_TRUTH)
    assert empty["score"] == 0.0 and empty["n_cohorts_estimated"] == 0


# ---------------------------------------------------------------- codebase


def test_codebase_strip_patch_paths_blocks_test_sections():
    test_patch = (
        "diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n"
        "+++ b/tests/test_x.py\n@@ -1 +1 @@\n-old\n+new\n"
    )
    model_patch = (
        "diff --git a/src/x.py b/src/x.py\n--- a/src/x.py\n+++ b/src/x.py\n"
        "@@ -1 +1 @@\n-bug\n+fix\n"
        "diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n"
        "+++ b/tests/test_x.py\n@@ -1 +1 @@\n-assert False\n+assert True\n"
    )
    blocked = score_codebase.test_patch_paths(test_patch)
    assert blocked == ["tests/test_x.py"]
    stripped = score_codebase.strip_patch_paths(model_patch, blocked)
    assert "src/x.py" in stripped and "tests/test_x.py" not in stripped


# --------------------------------------------------------------------- dbx


def _load_dbx(tmp_path, question, monkeypatch):
    monkeypatch.setenv("JUDGE_DIR", str(tmp_path))
    (tmp_path / "question.json").write_text(json.dumps(question))
    import sqlite3

    conn = sqlite3.connect(tmp_path / "question.db")
    conn.execute("CREATE TABLE items (id INTEGER, name TEXT)")
    conn.executemany(
        "INSERT INTO items VALUES (?, ?)", [(i, f"item{i}") for i in range(60)]
    )
    conn.commit()
    conn.close()
    return _load("dbx_server")


def test_dbx_answer_checking_matches_upstream(tmp_path, monkeypatch):
    server = _load_dbx(
        tmp_path,
        {"answer": 58, "answer_type": "integer", "tolerance": 0.0},
        monkeypatch,
    )
    assert server.evaluate_answer(
        "The answer is 58.", {"answer": 58, "answer_type": "integer"}
    )
    assert not server.evaluate_answer(
        "57 or 58", {"answer": 58, "answer_type": "integer"}
    )
    q_float = {"answer": 4.11, "answer_type": "float", "tolerance": 0.01}
    assert server.evaluate_answer("4.10", q_float)  # within max(0.01, 1%)
    assert not server.evaluate_answer("4.30", q_float)
    q_text = {"answer": "Acme Corp", "answer_type": "text"}
    assert server.evaluate_answer("  acme corp ", q_text)
    assert not server.evaluate_answer("Acme", q_text)


def test_dbx_query_semantics(tmp_path, monkeypatch):
    server = _load_dbx(
        tmp_path, {"answer": 1, "answer_type": "integer", "tolerance": 0.0}, monkeypatch
    )
    assert "ERROR: Only SELECT" in server.execute_sql("DROP TABLE items")
    assert "items" in server.execute_sql(".tables")
    capped = server.execute_sql("SELECT * FROM items")
    assert "showing first 50 rows" in capped
    assert capped.count("\n") <= 54  # header + separator + 50 rows + notice


def test_dbx_reward_formula(tmp_path, monkeypatch):
    server = _load_dbx(
        tmp_path,
        {"answer": 60, "answer_type": "integer", "tolerance": 0.0},
        monkeypatch,
    )
    server._state.update({"queries": 3, "done": False, "verdict": None})
    verdict = server._finalize(True, "60")
    assert verdict["reward"] == pytest.approx(1 - 3 / 15)
    server._state.update({"queries": 3, "done": False, "verdict": None})
    assert server._finalize(False, "59")["reward"] == 0.0


def test_dbx_canonical_order_is_deterministic():
    pre = [{"question_id": i} for i in range(1, 31)]
    post = [{"question_id": 100 + i} for i in range(1, 21)]
    a = convert_dbx.canonical_order(pre, post)
    b = convert_dbx.canonical_order(pre, post)
    assert a == b and len(a) == 40
    assert {q["question_id"] for q in a[:20]} <= {q["question_id"] for q in pre[:20]}
    assert all(q["question_id"] > 100 for q in a[20:])


# ------------------------------------------------------------------- poker


def test_poker_hand_specs_cover_the_default_schedule():
    convert_poker = _load("convert_poker")
    specs = convert_poker.hand_specs()
    assert len(specs) == 120
    assert [s["variant"] for s in specs[:1]] == ["calling_station"]
    assert specs[20]["variant"] == "loose_aggressive" and specs[20]["stage_start"]
    assert specs[119] == {
        "hand_number": 120,
        "stage_idx": 4,
        "variant": "loose_aggressive",
        "opponent_name": "Alex",
        "stage_seed": 46,
        "burn": 24,
        "stage_start": False,
    }


def test_poker_deal_matches_the_verified_upstream_table():
    pytest.importorskip("texasholdem")
    server = _load("poker_server")
    hand = server.Hand(
        {
            "variant": "calling_station",
            "stage_seed": 42,
            "burn": 3,  # hand 4 of stage seed 42
            "opponent_name": "Tom",
            "hand_number": 4,
        }
    )
    assert [str(c) for c in hand.game.hands[0]] == ["Ac", "9s"]
    assert [str(c) for c in hand.game.hands[1]] == ["3h", "5h"]
    assert hand.game.btn_loc == 1
    # Invalid actions leave the state untouched and report a reason.
    error = hand.act("RAISE", None)
    assert error and "Invalid poker action" in error
    # Check/call always finishes a hand; reward is chips over the big blind.
    for _ in range(60):
        if hand.done:
            break
        if hand.act("CHECK", None) is not None:
            assert hand.act("CALL", None) is None
    assert hand.done
    result = hand.result()
    assert result["reward"] == pytest.approx(result["net_chips"] / 10)
