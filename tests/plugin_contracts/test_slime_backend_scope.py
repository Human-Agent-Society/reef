from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_reef_never_wires_the_internal_rollout_stack() -> None:
    # Reef supplies the batches of trajectory jobs externally and never calls the
    # runtime's rollout generation package. The rollouts of a dataset job and a
    # task job are Slime's own, through its rollout manager (RFC #1041). The task
    # job's check names Slime's default rollout function, to refuse another one.
    offenders = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "reef").rglob("*.py")
        if "slime.rollout" in path.read_text(encoding="utf-8")
    ]
    assert offenders in ([], ["reef/train/slime_backend/harbor_rollouts.py"]), offenders
    source = (ROOT / "reef/train/slime_backend/harbor_rollouts.py").read_text(encoding="utf-8")
    assert source.count("slime.rollout") == 1 and "import slime" not in source and "from slime" not in source


def test_batch_processor_only_accepts_external_training_batches() -> None:
    source = (ROOT / "reef" / "train" / "slime_backend" / "reef_adapters" / "batches.py").read_text(encoding="utf-8")

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
    pin = "8a53b680e82932dabb48cbfcd34f749854400857"

    assert f"github.com/THUDM/slime.git@{pin}" in project_metadata
    assert "third_party/slime" not in (ROOT / ".gitmodules").read_text(encoding="utf-8")
    assert not (ROOT / "third_party" / "slime").exists()
