"""Storage interfaces and implementations for records, scenario commits, and generated tasks.

``commits`` owns CommitRecord, RecordProgress, and registration/checkpoint metadata
encoding. ``records``, ``tasks``, ``scenario`` and ``commit_history`` define storage
interfaces without importing concrete adapters. ``tasks`` also holds the local task
file store. These modules depend only on shared core values and each other.
Nothing in this package imports scenario coordination, recipes, runtimes, or training.
This package initializer also loads no adapters. Record and commit implementations
depend on ``reef.storage.records`` and ``reef.storage.scenario``.
Application assembly selects a storage service and
injects it into scenario coordination. Records, commits, generated tasks and
scenario settings all go through that service.

``sql_records`` implements shared SQLAlchemy record operations. ``sql_tasks``,
``sql_trajectories`` and ``sql_commits`` define the task, trajectory and commit
tables and implement their operations in the same database. ``sqlite``
supplies SQLite connections, schema upgrades, and file retention. ``postgres``
supplies PostgreSQL tables, a shared connection pool, and SQL retention.
``commit_history`` defines the commit history contract and combines a commit
history with any record store. ``migrations``
holds one-time upgrades of older schemas and files. ``sqlite`` and ``postgres``
also assemble their scenario storage services. Database-specific code stays here.
"""
