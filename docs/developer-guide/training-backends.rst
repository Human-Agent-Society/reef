Training backends
=================

A training backend is the integration that trains weights for a deployment:
Slime, Tinker, or one of your own. A recipe does not implement it. This page
defines what a new backend declares so that ``training.backend`` can select
it, and the job data directory that its training component reads.

:For: authors of a training integration beside Slime and Tinker
:Before you start: the component contract in `Training components <training-components.rst>`__
:You will have: a ``TrainingDeployment`` class, its registration, and the format of a job's data

A recipe author does not need this page: a weight method plugs into an
existing backend with an objective (see `Training service
<recipe-training-service.rst>`__).

The deployment definition
-------------------------

``training.backend`` selects one ``TrainingDeployment``
(``reef.train.deployment``). The built-in names are ``slime`` (the default),
``tinker`` and ``spindle_full``. A definition describes the deployment. It
must not download models, allocate devices or build a component.

- ``prepare(config, settings)``: required. Validate the backend's
  combinations and bind derived values in ``config``. Return the process
  definitions that ``reef serve`` waits for, such as Slime's optional
  teacher engine; an empty tuple for an in-process backend.
- ``component_config(settings, *, default_parameters)``: required. Return
  the config of the training component. Its ``type`` is the dotted
  reference of a ``TrainingComponentFactory``. ``default_parameters`` is the
  recipe's job parameters by sample kind, or empty before a recipe is
  selected.
- ``http_environment(settings)``: variables for the service process. A value
  can name ``${endpoints.<service>}`` of a service that ``prepare``
  returned. Default: none.
- ``trainer_serving_config(settings)``: the runtime config of the backend's
  own serving (``inference.serving: trainer``), such as Tinker's sampler.
  Default: ``None``.
- ``engine_weights(settings)``: how Reef's engines get weights for snapshots
  that hold none. Default: ``None``.

Class attributes:

- ``requires_local_model`` (default ``True``): a hosted integration sets
  ``False``, so that deployment resolution keeps a remote model identifier.
- ``supports_teacher_engine`` (default ``False``): whether the backend
  accepts a ``teacher`` section.
- ``default_serving`` (default ``"engines"``): the serving of a deployment
  that sets no ``inference.serving`` and no engine setting; ``engines`` or
  ``trainer``.

Only the training service builds the component that the factory returns
(see `Facts without a component
<training-components.rst#backend-facts>`__). The backend does not build
serving: the ``inference`` section selects it, and the component's
``artifact_format`` tells the serving runtime what each snapshot holds (see
`Serving saved weights <serving-runtimes.rst#serving-saved-weights>`__).

An in-process backend
~~~~~~~~~~~~~~~~~~~~~

An integration whose component trains in the service's process reuses the
supplied definition:

.. code:: python

   from reef.train.deployment import InProcessTrainingDeployment


   class Deployment(InProcessTrainingDeployment):
       component_type = "my_backend.runtime:factory"

``factory`` is a ``TrainingComponentFactory`` instance. Its ``config_type()``
declarations validate ``training.options``. Import execution libraries only
when the factory builds the component. The base ``prepare`` refuses Ray and
training executor settings, and ``training.options`` cannot override the
managed fields (``type``, ``model_path``, the timeouts, ``connect``).

Select the backend
~~~~~~~~~~~~~~~~~~

Select the class directly with ``--training.backend
my_backend.launch:Deployment``, or register an installed name:

.. code:: toml

   [project.entry-points."reef.training_backends"]
   my-backend = "my_backend.launch:Deployment"

This enables ``--training.backend my-backend``. Reef imports only the
selected definition. A missing or ambiguous name fails; Reef does not fall
back to Slime.

The job data directory
----------------------

A training component does not get a job's data as an object. Before Reef
submits a job, it writes the job's data as a directory. The job holds a
reference to it (``JobDataRef``), and the component reads the directory.

.. code:: text

   job.json        {"format": 2, "key", "sample_count", "kind"}
   samples.jsonl   one sample per line, in job order

- ``job.json`` gives the directory's key, its number of samples, and the
  kind of its samples: ``trajectory``, ``task``, or ``null`` without
  samples.
- Each line of ``samples.jsonl`` is the entry of ``JobData.to_dict``, with
  one difference: the line of a task has no ``task_path``. Its ``files``
  names the task's directory in the task file store
  (``storage.task_file_backend``): ``partition``, ``digest`` and ``name``.
- The directory holds no file of a task and no training request.

How a component reads it:

- A component that trains on Reef's host reads ``job.read_data()``, or the
  directory at ``job.data_dir``. The read makes the local copy. For a job
  of tasks it also fetches each task's directory, which is the sample's
  ``task_path``.
- A component that runs the job elsewhere, such as in a container, gives
  that place ``job.data_uri`` and mounts or copies the directory there.
  That place reads ``files`` from each task line and gets the directory
  from the task file store. Such a component must not read
  ``job.data_dir`` or ``job.read_data()``.

Rules:

- A directory does not change after Reef writes it, and several jobs can
  read it. A component must not write in it, and must not change a task's
  directory.
- A missing directory or task fails the read.
- Every task of a job must be in the task file store. Reef refuses job data
  with a task that has no ``files``.
- The key of a directory is the SHA-256 of its lines. A job of the same
  samples reuses the directory that exists.
- Reef deletes a directory that no job used for
  ``storage.job_directory_ttl_s`` (7 days by default). Keep this time above
  the longest job: a component can read the directory again when it resumes
  the job.
