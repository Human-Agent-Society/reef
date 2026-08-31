from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_reef_never_wires_the_internal_rollout_stack() -> None:
    # Reef supplies batches externally and never calls the runtime's rollout
    # generation package.
    offenders = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "reef").rglob("*.py")
        if "slime.rollout" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, offenders


def test_rollout_manager_only_accepts_external_training_batches() -> None:
    source = (ROOT / "reef" / "train" / "slime_backend" / "reef_adapters" / "rollout" / "manager.py").read_text(
        encoding="utf-8"
    )

    assert "def prepare_external_train_data(self, data):" in source
    for removed_contract in (
        "def generate(self, rollout_id):",
        "def eval(self, rollout_id):",
        "data_source_path",
        "rollout_function_path",
        "custom_reward_post_process_path",
    ):
        assert removed_contract not in source


def test_slime_pin_replaces_the_submodule() -> None:
    project_metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pin = "41014d1f29e201137fdffce737bb8bac65bc5219"

    assert f"github.com/THUDM/slime.git@{pin}" in project_metadata
    assert "third_party/slime" not in (ROOT / ".gitmodules").read_text(encoding="utf-8")
    assert not (ROOT / "third_party" / "slime").exists()
