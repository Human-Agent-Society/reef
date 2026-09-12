"""Storage interfaces and implementations for records and scenario commits.

``records`` defines the record interface without importing concrete adapters.
This package initializer also loads no adapters. Record and commit implementations
depend on ``reef.storage.records`` and ``reef.scenario.store``.
Application assembly selects a storage service and
injects it into scenario coordination. The scenario registry directly calls
``model_config`` functions for its fixed local JSON settings; those functions
hold no runtime state and do not depend on scenario or database code.

``sql_records`` implements shared SQLAlchemy record operations. ``sqlite``
supplies SQLite connections, schema upgrades, and file retention. ``postgres``
supplies PostgreSQL tables, a shared connection pool, and SQL retention. ``commit_log``
implements JSONL commits over any record store; ``scenario`` assembles the
default SQLite records and commit log layout. Database-specific code stays here.
"""
