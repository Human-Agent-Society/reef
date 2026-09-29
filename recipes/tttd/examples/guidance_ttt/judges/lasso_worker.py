"""Run the pinned Lasso evaluator in a disposable evaluator container."""

import importlib.util
import json
import sys
from pathlib import Path


def main() -> None:
    root = Path("/opt/SimpleTES")
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(
        "lasso_evaluator", root / "datasets/numerical_tasks/lasso_path/evaluator.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned Lasso evaluator")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    metrics = evaluator.evaluate("/candidate/solution.py")
    Path("/output/result.json").write_text(json.dumps(metrics))


if __name__ == "__main__":
    main()
