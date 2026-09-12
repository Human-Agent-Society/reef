"""Storage implementations for record and scenario contracts.

The dependency direction is from this infrastructure package to
``reef.records`` and ``reef.scenario.store``, never the reverse. Application
assembly selects a storage factory and injects it into scenario coordination.

``sql_records`` implements shared SQLAlchemy record operations. ``sqlite``
supplies SQLite connections, schema upgrades, and file retention. ``postgres``
supplies PostgreSQL tables, a shared connection pool, and SQL retention. ``commit_log``
implements JSONL commits over any record store; ``factory`` assembles the
default SQLite records and commit log layout. Database-specific code stays here.
"""
