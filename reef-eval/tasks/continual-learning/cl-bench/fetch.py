"""Fetch CL-Bench and convert it to Harbor tasks, pinned.

    cd tasks/continual-learning/cl-bench
    python fetch.py                 # every domain
    python fetch.py bsm sales       # just these
    python fetch.py bsm --limit 10  # first N per domain

All 301 tasks are committed to the repo; run this only to regenerate
them. Sources download from the pinned commit of
pgasawa/continual-learning-bench (Apache-2.0) and are verified against
their published sha256 where one exists. Each domain's converter lives
beside this script as ``convert_<domain>.py``.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

# The pinned upstream commit (main at conversion time).
COMMIT = "5f8c50eb1e84b2eda2ef4faff757dfc812a0ea26"
RAW = f"https://raw.githubusercontent.com/pgasawa/continual-learning-bench/{COMMIT}"
HERE = Path(__file__).parent
CACHE = HERE / ".data"


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _download(rel_path: str) -> Path:
    path = CACHE / rel_path.replace("/", "__")
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"downloading {rel_path} ...", flush=True)
        urllib.request.urlretrieve(f"{RAW}/{rel_path}", path)
    return path


def _verify_sha256(path: Path, expected: str, label: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(
            f"{label} integrity check failed: sha256 {digest[:16]}... does not "
            f"match the declared {expected[:16]}..."
        )


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


# ------------------------------------------------------------------ domains


def fetch_bsm(limit: int | None) -> int:
    """Blind spectrum monitoring: 90 scans, IoU-scored."""
    convert = _module("convert_bsm")
    base = "data/blind_spectrum_monitoring"
    metadata = json.loads(
        _download(f"{base}/mixed_grid_lifecycle_metadata.json").read_text()
    )
    corpus = _download(f"{base}/mixed_grid_lifecycle.jsonl")
    _verify_sha256(corpus, metadata["jsonl_sha256"], "bsm corpus")

    scans = sorted(_load_jsonl(corpus), key=lambda s: s["scan_idx"])
    if limit is not None:
        scans = scans[:limit]
    for scan in scans:
        stage = convert.stage_for_scan(scan["scan_idx"], metadata["stages"])
        convert.convert_scan(scan, stage, HERE, total_scans=metadata["total_scans"])
    return len(scans)


def fetch_sales(limit: int | None) -> int:
    """Sales prediction: 12 yearly forecasting rounds, WAPE-skill-scored."""
    convert = _module("convert_sales")
    base = "data/sales_prediction"
    metadata = json.loads(
        _download(f"{base}/sales_lifecycle_metadata.json").read_text()
    )
    panel_path = _download(f"{base}/sales_lifecycle_panel.jsonl")
    _verify_sha256(panel_path, metadata["panel_sha256"], "sales panel")

    panel = _load_jsonl(panel_path)
    furniture = json.loads(_download(f"{base}/furniture.json").read_text())
    locations = json.loads(_download(f"{base}/locations.json").read_text())

    instances = metadata["instances"]
    if limit is not None:
        instances = instances[:limit]
    for instance in instances:
        convert.convert_instance(
            instance, HERE, panel=panel, furniture=furniture, locations=locations
        )
    return len(instances)


def fetch_cohort(limit: int | None) -> int:
    """Cohort studies: 20 study releases, information-gain-scored."""
    convert = _module("convert_cohort")
    base = "data/cohort_studies/default"
    metadata = json.loads(_download(f"{base}/metadata.json").read_text())
    ground_truth = json.loads(_download(f"{base}/ground_truth.json").read_text())
    definitions = json.loads(_download(f"{base}/cohort_definitions.json").read_text())
    references = json.loads(_download(f"{base}/instance_references.json").read_text())

    instances = metadata["instances"]
    if limit is not None:
        instances = instances[:limit]
    for instance in instances:
        db_path = _download(f"{base}/dbs/{instance['db_filename']}")
        _verify_sha256(db_path, instance["db_sha256"], instance["db_filename"])
        convert.convert_instance(
            instance,
            HERE,
            db_path=db_path,
            ground_truth=ground_truth,
            cohort_definitions=definitions,
            reference_survival=references[instance["variant_id"]],
            total_instances=metadata["n_instances"],
        )
    return len(instances)


# The upstream default schedule's blocked issue order: tablib, then tenacity.
_TABLIB_PRS = [534, 540, 547, 579, 584, 594, 595, 596, 613]
_TENACITY_PRS = [597, 603, 604, 606, 609, 610, 611, 614, 615, 628]
CODEBASE_ORDER = [f"jazzband__tablib-{n}" for n in _TABLIB_PRS] + [
    f"jd__tenacity-{n}" for n in _TENACITY_PRS
]


def fetch_codebase(limit: int | None) -> int:
    """Codebase adaptation: 19 sequential real-PR issues, pass/fail."""
    convert = _module("convert_codebase")
    rows = _load_jsonl(_download("data/codebase_adaptation/final-dataset.jsonl"))
    by_id = {row["instance_id"]: row for row in rows}
    ordered = [by_id[i] for i in CODEBASE_ORDER if i in by_id]
    if limit is not None:
        ordered = ordered[:limit]
    for idx, row in enumerate(ordered):
        convert.convert_row(row, idx, HERE, total=len(CODEBASE_ORDER))
    return len(ordered)


def fetch_dbx(limit: int | None) -> int:
    """Database exploration: 40 metered questions, efficiency-scored."""
    convert = _module("convert_dbx")
    pre = json.loads(_download("data/database_exploration/questions.json").read_text())
    post = json.loads(
        _download("data/database_exploration/questions_post_drift.json").read_text()
    )
    ordered = convert.canonical_order(pre, post)
    if limit is not None:
        ordered = ordered[:limit]
    for position, question in enumerate(ordered):
        convert.convert_question(
            question,
            position,
            HERE,
            total=40,
            previous=ordered[position - 1] if position else None,
            drifted=position >= convert.DRIFT_BOUNDARY,
        )
    return len(ordered)


def fetch_poker(limit: int | None) -> int:
    """Exploitable poker: 120 deterministic hands, profit-in-BB-scored."""
    try:
        import texasholdem  # noqa: F401 (conversion simulates the oracle policy)
    except ImportError:
        raise SystemExit(
            "the poker domain needs texasholdem for the conversion-time oracle "
            "simulation: pip install texasholdem==0.11.0"
        ) from None
    convert = _module("convert_poker")
    specs = convert.hand_specs()
    if limit is not None:
        specs = specs[:limit]
    for spec in specs:
        convert.convert_hand(spec, HERE)
    return len(specs)


DOMAINS: dict[str, tuple] = {
    "bsm": (fetch_bsm, "blind spectrum monitoring (90 scans)"),
    "sales": (fetch_sales, "sales prediction (12 rounds)"),
    "cohort": (fetch_cohort, "cohort studies (20 study releases)"),
    "codebase": (fetch_codebase, "codebase adaptation (19 PR issues)"),
    "dbx": (fetch_dbx, "database exploration (40 metered questions)"),
    "poker": (fetch_poker, "exploitable poker (120 hands)"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="fetch + convert CL-Bench domains (pinned)"
    )
    parser.add_argument(
        "domains",
        nargs="*",
        choices=[*DOMAINS, []],
        help=f"domains to convert (default: all of {', '.join(DOMAINS)})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="only the first N instances of each domain",
    )
    args = parser.parse_args()

    total = 0
    for name in args.domains or list(DOMAINS):
        fetch, description = DOMAINS[name]
        count = fetch(args.limit)
        total += count
        print(f"  {name}: converted {count} task(s), {description}")
    print(f"{total} CL-Bench task(s) -> {HERE}")
    print("stream a domain: reef-eval stream cl-bench/sales-* --agent <a>")


if __name__ == "__main__":
    main()
