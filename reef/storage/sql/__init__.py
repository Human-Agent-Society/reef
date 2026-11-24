"""The SQL implementations of the stores, for SQLite and PostgreSQL.

- ``setup`` holds the connection setup that the SQL stores share, and ``schema_upgrades`` the version of a store's
  tables and the steps that upgrade them.
- ``records`` holds the shared SQLAlchemy operations of a scenario's state: the trajectory nodes of record items
  and the database of one scenario (``SQLScenarioDatabase``).
- ``commits``, ``samples``, ``tasks`` and ``evaluations`` define the commit, training state, task and evaluation
  tables in the record database of a scenario, and implement their operations.
- ``data`` implements the data store (``SQLiteDataStore``, ``PostgresDataStore``).
- ``coordination`` implements the coordination tables.
- ``sqlite`` and ``postgres`` supply the connections, the table creation and the retention of each database, and
  assemble its scenario storage (``SQLiteScenarioStorage``, ``PostgresScenarioStorage``).

Import each module directly; this package exports nothing.
"""
