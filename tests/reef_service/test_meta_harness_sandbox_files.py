"""No model or E2B calls: validate the executable-candidate filesystem boundary."""

import io
import tarfile

import pytest

from recipes.meta_harness.examples.terminal_bench.sandbox_files import collect_outputs, episode_archive
from reef.harness.executor import EpisodeLaunchError


def test_episode_archive_keeps_rendered_inputs_root_owned(tmp_path):
    (tmp_path / "source.py").write_text("raise RuntimeError('candidate')")
    (tmp_path / "workspace").mkdir()
    (tmp_path / "results").mkdir()
    stream = io.BytesIO(episode_archive(tmp_path, [tmp_path / "results"]))
    with tarfile.open(fileobj=stream) as archive:
        items = {item.name: item for item in archive.getmembers()}
    assert items["source.py"].uid == 0
    assert items["source.py"].mode == 0o444
    assert items["."].uid == 0
    assert items["workspace"].uid == 1001
    assert items["results"].uid == 1001


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escaped", tarfile.REGTYPE),
        ("/escaped", tarfile.REGTYPE),
        ("workspace/link", tarfile.SYMTYPE),
        ("workspace/pipe", tarfile.FIFOTYPE),
    ],
)
def test_container_outputs_cannot_escape_or_restore_links(tmp_path, name, kind):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        item = tarfile.TarInfo(name)
        item.type = kind
        item.linkname = "/etc/passwd" if kind == tarfile.SYMTYPE else ""
        archive.addfile(item)
    stream.seek(0)
    with pytest.raises(EpisodeLaunchError):
        collect_outputs(stream, tmp_path, [])
    assert not list(tmp_path.iterdir())


def test_collection_cannot_replace_frozen_inputs(tmp_path):
    (tmp_path / "source.py").write_text("frozen")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, text in (("source.py", b"tampered"), ("workspace/result.txt", b"observed")):
            item = tarfile.TarInfo(name)
            item.size = len(text)
            archive.addfile(item, io.BytesIO(text))
    stream.seek(0)
    collect_outputs(stream, tmp_path, [])
    assert (tmp_path / "source.py").read_text() == "frozen"
    assert (tmp_path / "workspace/result.txt").read_text() == "observed"
