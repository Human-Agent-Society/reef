import os
from pathlib import Path

import pytest


@pytest.fixture
def fake_git_lfs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executable = tmp_path / "bin" / "git-lfs"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{executable.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)


@pytest.fixture
def postgres_config():
    """Use only an explicitly configured test server and a disposable schema."""
    import uuid

    from sqlalchemy import create_engine
    from sqlalchemy.schema import DropSchema

    from reef.storage.postgres import postgres_url

    url = os.environ.get("REEF_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("set REEF_TEST_POSTGRES_URL to run PostgreSQL integration tests")
    schema = f"reef_test_{uuid.uuid4().hex}"
    try:
        yield url, schema
    finally:
        engine = create_engine(postgres_url(url), hide_parameters=True)
        try:
            with engine.begin() as connection:
                connection.execute(DropSchema(schema, cascade=True, if_exists=True))
        finally:
            engine.dispose()


@pytest.fixture
def postgres_database(postgres_config):
    from reef.storage.postgres import PostgresRecordDatabase

    url, schema = postgres_config
    database = PostgresRecordDatabase(url, schema=schema)
    try:
        yield database
    finally:
        database.close()
