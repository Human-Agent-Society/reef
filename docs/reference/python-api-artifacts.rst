Artifact API
============

The contracts that store, import and describe artifacts: the artifact backend and its optional capabilities, the layout and the source readers of an import, base roots, and the components that a recipe declares.

`Choose the artifact backend <../user-guide/storage.rst#choose-the-artifact-backend>`__ shows how a deployment selects a backend.

Artifact backends
-----------------

An artifact backend stores the snapshots of each scenario. By default, Reef uses Git LFS (``GitLFSRepositoryBackendFactory``). A deployment selects a different backend in ``storage.artifact_backend``.

A backend plugin is a subclass of ``reef.artifact.ConfigurableRepositoryBackendFactory``. Calling the factory with a scenario name returns the ``RepositoryBackend`` of that scenario. The plugin implements one class method:

.. code:: python

   @classmethod
   def from_config(cls, config, *, context): ...

- ``config`` is the plugin's own ``config`` mapping. Raise ``ValueError`` for an invalid value.
- ``context`` is a ``RepositoryBackendContext``. ``work_dir`` and ``cache_dir`` are ``storage.artifact_work_dir`` and ``storage.artifact_cache_dir``.

Reef does not call a close method on a backend factory.

A snapshot's revision and checksum are private locators; the public identity belongs to each component's ``artifact_id``. Activating or serving an existing artifact updates scenario selections without calling ``publish``. Every scenario owns its root snapshot, which Reef writes when it creates the scenario (`Starting artifacts <../user-guide/operate.rst#starting-artifacts>`__). A backend shares no snapshot between scenarios.

``tests/artifact/test_reef_artifacts.py`` and ``tests/artifact/test_reef_git_lfs.py`` show the publication and recovery behavior that a backend must have.

Snapshots and artifact IDs
~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``RepositoryBackend`` implements these members and keeps each component entry's ``artifact_id`` in the manifests that it stores:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Member
     - Contract
   * - ``register(root, *, metadata)``
     - ``root`` is an unpublished ``Artifact`` whose ``local_path`` is a directory. When the scenario has no head, store the files of ``root`` as a snapshot with no parent, preserve its artifact IDs and ``metadata``, and make it the head in one atomic step. When the scenario has a head, also one that another caller registered first, store nothing and return that head.
   * - ``artifact_snapshot(artifact_id)``
     - Return the ``SnapshotRef`` of the stored snapshot that holds the artifact, or raise ``ArtifactNotFoundError``. The ID must keep naming the same files and metadata across publications and restarts. Reject an attempt to bind it to different content. The inherited ``resolve_artifact(artifact_id)`` materializes that snapshot and returns the artifact's component.
   * - ``bind_artifact_id(artifact_id, ref)``
     - Bind an unbound ID to a stored snapshot and return any earlier binding. A binding never changes.
   * - ``artifact_content(ref, artifact_id)``
     - Return a value that is equal for two snapshots exactly when they hold the same files and metadata for the artifact. The inherited ``bind_artifact_ids(snapshot)`` uses this and ``bind_artifact_id``, and raises ``ArtifactConflictError`` for an ID that names other content. Call it before a head moves.
   * - ``resolve_snapshot(revision)``
     - Return a ``SnapshotRef`` for a stored snapshot, or raise ``ArtifactNotFoundError``. Its ``revision`` and ``parent_revision`` are backend locators, separate from component IDs.

Stored content
~~~~~~~~~~~~~~

Stored content is a named file tree of one scenario with no parent, no metadata and no artifact ID, and no head moves to it. The training service stores each checkpoint as content, and control commits it by name without reading a file. A backend implements five operations on it:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Member
     - Contract
   * - ``store_content(name, directory)``
     - Store the files of ``directory`` as the content ``name``: exactly those files, whatever their names. Storing a name again replaces its files in one step for a reader, so a repeated call after a failure is safe.
   * - ``materialize_content(name)``
     - Return a local directory with exactly those files, which stays readable when the name is stored again.
   * - ``discard_content(name)``
     - Remove the content. A name that the backend does not hold changes nothing, and every snapshot keeps its files.
   * - ``discard_all_content()``
     - Remove every content of the scenario. Control's sweep calls it after the scenario was deleted and its registration archived, so the call must find the content of a scenario that has no registration. The sweep repeats a call that failed.
   * - ``publish_parts(parts, *, metadata, expected_parent, advance_head)``
     - Store a snapshot of files that the backend holds already, and move the head as ``publish`` does. ``parts`` maps the directory of each component to its ``StoredFiles``, or the one key ``None`` for a flat snapshot. A ``StoredFiles`` names stored content, or an artifact ID whose files an earlier snapshot holds. Carry a component's files byte for byte. ``metadata`` is complete: it holds the component manifest and the artifact IDs. Bind them with ``bind_artifact_ids`` before a head moves. Raise ``ArtifactNotFoundError`` for content or an artifact that the backend does not hold. A failure can leave the snapshot stored with its IDs bound and no head on it. Make the snapshot inside the storage, for example with a server-side copy, so that the calling process downloads no file.

Callers use the inherited ``publish_stored``, which checks first that the head is ``expected_parent``, also for ``advance_head=False``. A backend that moves the head checks it again in the same step as the move.

The factory has two cleanup hooks, each with a default that does nothing:

- ``reclaim_discarded()``: frees the files of content whose discard deleted only the name. Control's sweep of ended work calls it once per sweep.
- ``forget_scenario(scenario)``: drops what the process keeps of a deleted scenario beside the store, such as a cached backend, a clone or the cached files of its snapshots. It changes nothing for a scenario that was not deleted. Each service calls it when it learns of the delete. ``CachedRepositoryBackendFactory`` checks ``has_archived_registration``, drops its cached backend, and calls ``forget_local_state(scenario)``, which a subclass overrides to remove its own files.

Optional capabilities
~~~~~~~~~~~~~~~~~~~~~

Inherit only the base classes that the backend supports:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Base class
     - Capability
   * - ``RegistrationAwareRepositoryBackendFactory``
     - ``has_registration(scenario)``: find a stored scenario without making its backend.
   * - ``EnumerableRepositoryBackendFactory``
     - ``list_registrations()``: list the stored scenarios. The service loads them when it starts.
   * - ``CachedRepositoryBackendFactory``
     - Both capabilities, and one cached backend for each scenario. A subclass must implement ``has_archived_registration(scenario)``: True exactly when ``archive_registration(scenario)`` has completed in any process that shares the store, and never False again after that. Reef refuses to create a scenario for which it is true, because a deleted scenario's ID is never used again.
   * - ``StagedSnapshotRepositoryBackend``
     - The backend side: ``publish(advance_head=False)`` stores a snapshot without moving the head, and ``commit_snapshot`` moves the head after the scenario commit. A durable scenario store requires it.

A backend overrides these ``RepositoryBackend`` members when its storage allows it:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Member
     - Contract
   * - ``snapshot_metadata(ref)``
     - Read a snapshot's metadata without its files. The default materializes the whole snapshot.
   * - ``materialize_component(ref, name)``
     - Make one component of a snapshot local without the files of the others. Control reads a harness tree through it, so that it fetches no weights. The default materializes the whole snapshot.
   * - ``file_downloads(ref, expires_in_seconds)``
     - For storage that can give clients a URL for each file, such as an object store. Return every file of the snapshot as a ``FileDownload``: ``path`` in the snapshot, ``byte_count``, and a ``url`` that opens the file without a Reef credential until ``expires_at``. It must not materialize the snapshot. The artifact file list of the HTTP API gives these URLs to clients. The default returns ``None``, and Reef then serves the files itself.
   * - ``snapshot_uri(ref)``
     - The URI of the snapshot's directory, ending with ``/``, in a scheme that ``ArtifactSources`` reads. The `artifact items <http-api-data-items.rst#artifact-items>`__ of the data store carry it, so that a scenario, also in another deployment, imports the snapshot from there. The default returns ``None``.

Artifact imports
----------------

``POST /reef/scenarios/{scenario}/artifacts`` imports a directory as a new artifact of one component (`Import an artifact <../user-guide/operate.rst#import-an-artifact>`__). The steps are:

1. A source reader opens the files that the URI names.
2. The component's layout checks them where the source keeps them.
3. Reef stores a copy, or a reference to a Hugging Face revision that cannot change.
4. The component's admission check runs on what Reef stores (`Admission checks <../developer-guide/components.rst#admission-checks>`__).

Layouts
~~~~~~~

A component's layout is the ``layout`` of its ``ComponentSpec``, a ``reef.artifact.components.ArtifactLayout``. ``None``, the default, means that the component takes no import.

``check(files)`` returns when ``files`` are an artifact in the layout of the component's snapshots, and the remote weights that they name exist. Otherwise it raises ``ValueError``, and Reef answers HTTP 400 before it copies anything. ``files`` is a ``reef.artifact.ArtifactFiles``. It gives ``paths()``, and ``sha256(path)`` and ``read_bytes(path)`` for each path. Read only the files that the check needs: on an S3 prefix or a Hugging Face revision, each read downloads that file.

The built-in recipes bind one:

- A weight recipe's component gets ``TrainingArtifactLayout(facts, component)`` (``scenario_components``), which calls the facts' ``check_artifact`` (`Training components <python-api-training.rst#training-components>`__). For full weights and adapters it reads only file names and ``adapter/adapter_config.json``.
- A harness recipe binds ``HarnessLayout``. Every file must be UTF-8 text, and the files must form a ``HarnessTree`` that passes ``validate_tree`` for the adapter: the source tree that a harness artifact stores.

Source readers
~~~~~~~~~~~~~~

A source reader is a subclass of ``reef.artifact.ArtifactSourceReader`` with two members:

.. code:: python

   @property
   def schemes(self) -> tuple[str, ...]: ...

   def open(self, uri: str) -> SourceFiles: ...

``open`` returns the ``reef.artifact.SourceFiles`` that ``uri`` names, and reads at most a listing. Raise ``ValueError`` for a URI that names nothing the reader can read, and ``ArtifactSourceError`` when the source fails. ``SourceFiles`` adds three members to ``ArtifactFiles``:

- ``version``: the version that the source served, or ``""`` when it has none.
- ``reference``: ``None`` when Reef stores a copy. Otherwise, the files, as paths and bytes, that Reef stores in place of a copy, for a source that keeps the files fixed.
- ``copy_into(directory)``: write the files into the empty ``directory``.

Reef ships ``HuggingFaceReader`` (``hf``), ``FileReader`` (``file``) and ``S3Reader`` (``s3``), whose files are ``HuggingFaceRevisionFiles``, ``DirectorySource`` and ``S3PrefixFiles``. A deployment adds readers in ``storage.artifact_sources``. A configured reader replaces the built-in reader of a scheme that it reads.

``HuggingFaceRevisionFiles`` has a reference when the URI names a full 40-digit commit SHA:

- Reef stores the reference in place of a copy only when the component's layout says that its readers resolve one (``ArtifactLayout.resolves_references``). ``TrainingArtifactLayout`` does for snapshots that hold weight files.
- Reef then stores only ``hf-reference.json``, ``{"uri": "hf://org/model", "version": "<commit>"}``. ``reef.artifact.read_reference(directory)`` reads it back.
- The built-in serving runtimes load such a snapshot from the revision's snapshot, which Hugging Face downloads once into its cache. A runtime that loads models from the Hub can read the reference and download nothing.

Base roots
~~~~~~~~~~

An artifact of a weight component has one more form, the base root. It holds one file, ``base.json``, and no weight files. The artifact stands for the deployment's base model with no adapter:

.. code:: json

   {"format": "reef-base-root-1",
    "inputs": {"model": "/models/qwen", "lora_rank": "8"}}

``inputs`` are the inputs of the training backend's default start (``TrainingBackendFacts.default_start_inputs``). ``model`` is always there. Slime with LoRA adds ``lora_rank``, ``lora_alpha`` and ``lora_target_modules``. Tinker adds ``lora_rank``, and ``service`` when the deployment names a base URL.

``reef.artifact.BaseRoot`` is the value:

- ``read_base_root(directory)`` reads it from an artifact directory and returns ``None`` for an artifact without the file.
- ``base_root_of(files)`` reads it from ``ArtifactFiles`` and refuses an artifact that holds other files beside it.
- A reader refuses a file with another ``format``.

The default start of the Slime and the Tinker backends is a base root: ``SlimeFacts.default_start()`` and ``TinkerFacts.default_start()`` write it, and a new scenario's root holds it. The ``spindle_full`` backend has no base root. Its default start is a reference to the base revision, which ``SpindleFullFacts.default_start()`` writes.

`Saved-weight runtimes <python-api-inference.rst#saved-weight-runtimes>`__ says how each runtime serves a base root.

Scenario components
-------------------

.. code:: python

   from reef.artifact.components import ComponentSpec, ScenarioComponents, TextFileTree

``Recipe.build_components(scenario)`` returns the components that every snapshot of the scenario binds, as the recipe's configuration declares them. `Scenario components <../developer-guide/components.rst>`__ lists what the built-in recipes declare, so most methods never override it.

``ScenarioComponents`` is a frozen dataclass that maps each component name to a ``ComponentSpec``:

- Bare ``ScenarioComponents()`` is the complete record-only configuration.
- A snapshot of one component is flat. A snapshot of several components has one directory per component.
- At most one component has a weight kind, and at most one has ``files``.
- The field ``validator`` of ``ScenarioComponents`` is the snapshot's own check. A snapshot of several components has none, so there it stays ``AcceptAnyArtifact()``.

A ``ComponentSpec`` has these fields:

.. list-table::
   :header-rows: 1
   :widths: 18 28 54

   * - Field
     - Type
     - Contract
   * - ``validator``
     - ``ArtifactValidator``
     - Admit an artifact of the component before it is stored. The default accepts any artifact.
   * - ``kind``
     - ``ArtifactKind | None``
     - The kind of every artifact of the component. ``model`` and ``model_adapter`` load into the inference runtime, with ``InferenceRuntime.activate_checkpoint`` (`Runtime responsibilities <python-api-inference.rst#runtime-responsibilities>`__).
   * - ``files``
     - ``FileTree | None``
     - Back client pulls.
   * - ``layout``
     - ``ArtifactLayout | None``
     - Check an imported artifact. ``None`` takes no import.

Components do not decide which records train, compute candidates, execute a training job, or mutate the snapshot chain (`What components do not own <../developer-guide/components.rst#what-components-do-not-own>`__).
