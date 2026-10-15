"""Storage interfaces and implementations for records and scenario commits.

``commits`` owns CommitRecord, RecordProgress, and registration/checkpoint metadata
encoding. ``records`` and ``scenario`` define storage interfaces without importing
concrete adapters. All three depend only on shared core values and each other.
Nothing in this package imports scenario coordination, recipes, runtimes, or training.
This package initializer also loads no adapters. Record and commit implementations
depend on ``reef.storage.records`` and ``reef.storage.scenario``.
Application assembly selects a storage service and
injects it into scenario coordination. Records, commits and scenario settings
all go through that service.

``sql_records`` implements shared SQLAlchemy record operations. ``sqlite``
supplies SQLite connections, schema upgrades, and file retention. ``postgres``
supplies PostgreSQL tables, a commit table, a shared connection pool, and SQL
retention. ``commit_log`` defines the commit history contract, implements JSONL
commits, and combines a commit history with any record store. ``migrations``
holds one-time upgrades of older schemas and files. ``sqlite`` and ``postgres``
also assemble their scenario storage services. Database-specific code stays here.
"""
