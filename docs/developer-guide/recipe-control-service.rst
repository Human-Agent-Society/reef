Control service: reports, jobs and activation
=============================================

The control service calls three recipe plug-ins. The report type checks each
report before Reef stores it. The sample-to-job stage sets the size of a
training job and makes its data. The activation rule decides whether a new
artifact serves. This page defines each contract.

:For: authors of a weight method, and deployers who write an activation rule
:Before you start: the plug-in map in `Write a recipe <write-a-recipe.rst>`__
:You will have: a report type, a sample-to-job stage with its job size, and an activation rule

Report type
-----------

``Recipe.report_type`` is the typed contract of the reports that the recipe
trains on. Control parses each ``POST /reef/report`` with it before it stores
the report.

**Contract.** Return a frozen dataclass that inherits
``reef.core.reports.ReportBase``:

.. code:: python

   from dataclasses import dataclass

   from reef.core.reports import ReportBase


   @dataclass(frozen=True)
   class MyReport(ReportBase):
       score: float
       difficulty: int | None = None


   class MyMethodRecipe(WeightTrainingRecipe):
       @property
       def report_type(self) -> type[ReportBase]:
           return MyReport

- ``score`` reads the payload's top-level ``score``. Every other field reads
  ``metadata.<field>``.
- Field types: ``float`` (finite), ``int``, ``str``, ``bool``,
  ``Mapping[str, str]``, each optionally ``| None``. A field without a
  default is required.
- Override ``validate()`` for rules between fields. Raise
  ``ReportValidationError``.
- Reef ships ``ScoredRolloutReport`` (a required ``score``) and
  ``TeacherContextReport`` (``teacher_context`` and an optional ``score``).

**Default.** ``None``: Reef accepts every well-formed report. A
computed-feedback recipe keeps the default.

**Failure behavior.** A report that the type refuses returns HTTP 400, and
the body names the field. Reef stores nothing. For every recipe, control
also refuses a report that:

- references a record that is not an inference of the same scenario;
- repeats a reference, or references a call inside another referenced
  trajectory. A report references only the last call of each trajectory;
- sets ``metadata.training.eligible``. A report cannot opt out of training.

An identical retry of a stored report returns the original receipt.

In a scenario of several components, control accepts a report that any
component's type accepts. Each trainer releases a report that has the shape
of another component.

Sample-to-job stage
-------------------

The sample-to-job stage (``reef.train.sample2job.Sample2Job``) turns the stored
samples of the next job into one ``JobData``, and gives the job its
parameters. Each trainer cycle on control calls it when enough samples wait
(``auto``) or a training request is queued (``manual``). Then control
submits the job to the training service.

- ``from_config(config, *, context, record2sample, environ)``: class
  method. Build the stage of one scenario. ``record2sample`` is the
  scenario's record-to-sample stage, so the stage can read a value that the
  other stage owns.
- ``__init__(context, buffer_size, job_samples, *, max_job_samples=None)``:
  the base constructor. Pass the parsed sizes to it.
- ``make_job_data(samples, data_id) -> JobData``: required. Build the job's
  data from ``samples``, in store order, with the ID ``data_id``. The job's
  commit consumes every sample of the tuple, also the samples that the
  stage leaves out. Do not call models or train here.
- ``JobData.delayed``: optional. Data whose samples are all tasks can also
  carry delayed samples (``DelayedSample``): trajectories that train beside
  the job's task rollouts, each with its advantage. Keep them within
  ``context.delayed_sample_limit``, the limit of the training backend (see
  `Backend facts <training-backends.rst#backend-facts>`__); the component
  refuses data outside it. Control builds the data against what the scenario
  serves then, and the pipeline leaves out each delayed sample that the
  record-to-sample stage's ``is_stale`` finds stale (see `Staleness
  <recipe-samples-service.rst#staleness>`__). The generated-task recipe
  carries `Designer replies <task-generators.rst#designer-replies>`__ this
  way.
- ``job_parameters(data)``: the job's parameters. The default is the
  recipe's parameters for the kind of the samples, with the request's
  ``parameters`` merged over them (see `Job parameters
  <recipe-training-service.rst#job-parameters>`__).
- ``status()`` and ``close()``: state for ``GET /reef/status``, and release
  of the stage's resources.

Class attributes tell Reef what the stage accepts:

- ``supported_training_modes`` (default ``auto`` and ``manual``): the
  `training modes`_ that the job data supports.
- ``reads_requests`` (default ``False``): in ``manual``, also run a request
  that names no sample.
- ``required_request_types`` (default inference and report): the kinds of
  record that the trainer gives the pipeline.
- ``accepts_trajectories`` (default ``False``): train `uploaded
  trajectories`_.
- ``accepts_imported_tasks`` (default ``False``): train the scenario's
  imported ``train`` tasks.

Built-in stages (``reef.train.sample2job``):

- ``NoJobData``: the default. The pipeline never becomes ready.
- ``DistillJobData``: the samples as they come, in store order. The
  distilling recipes and SFT use it.
- ``TraceJobData`` and ``TrainTaskJobData``: harness evolution's stages.

A minimal stage:

.. code:: python

   from reef.core.job_data import JobData
   from reef.train.sample2job import Sample2Job, Sample2JobSettings


   class MyJobData(Sample2Job):
       @classmethod
       def from_config(cls, config, *, context, record2sample, environ):
           settings = Sample2JobSettings.parse(config, plugin_class=cls)
           return cls(context, settings.buffer_size, settings.job_samples)

       def make_job_data(self, samples, data_id):
           return JobData(data_id, samples)

The recipe names the class in ``pipeline_spec()``. The deployment configures
it in ``pipeline.sample2job`` (see `the pipeline section
<../reference/configuration-recipe.rst#the-pipeline-section>`__).

Job size
~~~~~~~~

Reef submits one training job for each job's data.

- ``pipeline.sample2job.config.job_samples``: the fewest samples that one
  automatic job trains; a positive int.
- ``pipeline.sample2job.config.buffer_size``: the most samples that one
  automatic job takes, beside the rest of a group. It is at least
  ``job_samples``, and ``job_samples`` when absent.
- A weight recipe has no default ``job_samples``. Without it, building a
  scenario's trainer raises ``PluginConfigError``.
- The harness stages default ``job_samples`` to 1.
- A stage whose job data is one fixed unit, such as the TTTD grid, passes
  ``max_job_samples`` to ``Sample2Job.__init__``. A larger ``job_samples``
  or ``buffer_size`` raises ``PluginConfigError``.

The training component cuts the job's data into optimizer steps by its own
batch size: Slime's ``--global-batch-size`` or Tinker's ``batch_size``.
Usually, set ``job_samples`` to that batch size, so that one job is one
optimizer step. The two can differ:

- A job takes whole groups, so its data can hold more than ``job_samples``
  samples. The component then trains more than one step.
- Job data smaller than the batch size trains as one partial step. The
  ``remainder`` field of the ``scheduling`` parameter decides what happens
  to a shorter last step.
- With ``scheduling.packing_length``, a step trains batch-size packs of
  tokens. ``job_samples`` still counts samples; set it to the samples of
  many steps.

The next job
~~~~~~~~~~~~

In ``auto``, the pipeline selects the samples of the next job from the
scenario's stored samples:

1. It takes the samples that the trainer can train and did not train yet,
   in store order: the samples that the trainer's stage made, uploaded
   trajectories, and imported ``train`` tasks, when the stage accepts them.
   Imported ``eval`` tasks never train.
2. It keeps the samples of one kind, trajectory or task: the kind of the
   first sample. A kind without job parameters does not count.
3. It leaves out each sample that the samples service marked stale (see
   `Staleness <recipe-samples-service.rst#staleness>`__).
4. It takes the first ``buffer_size`` samples that remain, and the rest of
   each group that those samples touch.

The job is ready when it holds at least ``job_samples`` samples. Reef then
reserves the samples, calls ``make_job_data``, and submits the job. The
commit of the submission counts each sample as trained. A job that later
fails does not give its samples back.

The other samples wait in the store for later jobs. ``GET /reef/status``
reports ``ready_samples``, ``buffer_size`` and ``job_samples``.

Training modes
~~~~~~~~~~~~~~

``training_mode`` is ``auto`` (the default) or ``manual``. The recipe's
``training_mode`` field sets the initial value.

- ``auto`` makes job data when the next job holds ``job_samples`` samples.
- ``manual`` makes job data only for a queued training request. The job
  takes the samples that the request names, in store order, whatever
  ``job_samples``. A sample that trained before trains again, and staleness
  does not apply.

When Reef accepts a request, a trainer must be able to take its samples: the
store holds each one, the trainer can train each one, they are of one kind,
and they are at most ``max_job_samples``. Otherwise the request returns an
error. A request that names no sample is refused, unless the stage sets
``reads_requests``, as the harness stages do.

The pipeline runs the oldest request first. Automatic job data has the ID
``<scenario>:data:<number>``, and a request's data
``<scenario>:request:<request id>``. The request is in ``JobData.request``.

``POST /reef/scenarios/{scenario}/update`` changes the mode of a running
scenario. Reserved job data stays unchanged. Reef stores the mode in the
scenario's settings, so a restart applies it again. The samples of records
are stored in both modes, so a change to ``auto`` trains the data that was
collected in ``manual``. A mode that the stage does not support raises
``NotImplementedError``; an unknown mode name raises ``ValueError``.

Uploaded trajectories
~~~~~~~~~~~~~~~~~~~~~

A client can give a scenario complete ATIF trajectories as data items (see
the `HTTP API <../reference/http-api-data-items.rst#http-api-data-items>`__). A trajectory item
is not a record: no stage forms a record set of it.

- A recipe takes uploads when its ``Sample2Job`` sets
  ``accepts_trajectories = True``. Reef refuses with HTTP 400 an upload to a
  scenario with no trainer that accepts them.
- Control turns each upload into its training sample with
  ``Record2Sample.uploaded_sample(trajectory)`` when it reads the upload. The
  default returns the trajectory unchanged. Override it to check the
  structure or add metadata, as ``SFTSamples`` does. The method must be
  pure.
- A trajectory that ``uploaded_sample`` refuses with ``ValueError`` is
  skipped with a warning and stays untrained.
- An upload counts as one sample toward ``job_samples`` and
  ``buffer_size``, and it is never stale. In a scenario of several
  components, each trainer that accepts uploads trains it once.
- An objective that needs a reward needs a finite ``extra.reef.reward`` in
  the trajectory. An SFT sample needs none.

Activation rule
---------------

A recipe trains; it does not decide what serves. Each artifact that a
training job saves goes onto the scenario's snapshot chain, and the next job
starts from it. The activation rule decides whether the artifact also
serves. The rule applies to every recipe.

Control asks the rule at two moments:

- ``activates(made) -> bool``: when a training step publishes an artifact.
  ``made`` is a ``MadeArtifact``: the ``component`` whose trainer made it,
  and ``changed_kinds``, the harness node kinds that it changes (empty for
  weights).
- ``activates_evaluated(artifact_id, evaluation) -> bool``: when an
  evaluation of the artifact finishes. The default returns ``False``.

Name the rule in the top-level ``activation`` section:

.. code:: yaml

   activation:
     module: evaluated
     config:
       min_mean_score: 0.8

.. list-table::
   :header-rows: 1
   :widths: 16 30 54

   * - Rule
     - Config
     - Activates
   * - ``always``
     - ``review_kinds`` (optional list)
     - each new artifact, except a harness tree that changes a part in
       ``review_kinds``. The default rule
   * - ``never``
     - none
     - nothing: a person activates each artifact
   * - ``evaluated``
     - ``min_mean_score`` (required)
     - an artifact whose evaluation ends with a mean score of at least
       ``min_mean_score``

A rule of your own is an ``ActivationRule`` subclass, named by
``package.module:ClassName``. Reef builds it with the class method
``from_config(config)``:

.. code:: python

   from reef import ActivationRule


   class ComponentRule(ActivationRule):
       def __init__(self, components):
           self.components = frozenset(components)

       @classmethod
       def from_config(cls, config):
           return cls(config["components"])

       def activates(self, made):
           return made.component in self.components

Both methods must decide from their arguments alone.

**Failure behavior.** Reef imports the class and builds the rule once when
it loads the deployment. An unknown name or a config that the rule refuses
raises ``RecipeConfigError`` then. A scenario can select a built-in rule for
itself in its settings; it cannot name a class.

A harness job's own evaluation (``evolution.evaluation``) decides only
which candidate tree the job keeps (see `Evaluation
<write-a-harness-method.rst#evaluation>`__). The activation rule then
decides whether the kept tree serves. `Configuration
<../reference/configuration-recipe.rst#the-activation-section>`__ lists the rule
settings.
