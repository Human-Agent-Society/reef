"""Storage interfaces and implementations for records and scenario commits.

``commits`` owns CommitRecord, RecordProgress, and registration/checkpoint metadata
encoding. ``records`` and ``scenario`` define storage interfaces without importing
concrete adapters. All three depend only on shared core values and each other.
Nothing in this package imports scenario coordination, recipes, runtimes, or training.
This package initializer also loads no adapters. Record and commit implementations
depend on ``reef.storage.records`` and ``reef.storage.scenario``.
Application assembly selects a storage service and
injects it into scenario coordination. The scenario registry directly calls
``model_config`` functions for its fixed local JSON settings; those functions
hold no runtime state and do not depend on scenario or database code.

``sql_records`` implements shared SQLAlchemy record operations. ``sqlite``
supplies SQLite connections, schema upgrades, and file retention. ``postgres``
supplies PostgreSQL tables, a shared connection pool, and SQL retention. ``commit_log``
implements JSONL commits over any record store. ``sqlite`` and ``postgres``
also assemble their scenario storage services. Database-specific code stays here.
"""
