Control service: reports, jobs and activation
=============================================

Extend control to validate reports, build training job data or choose which
artifacts serve. These are separate contracts; implement only what your
recipe needs.

Report type
-----------

``Recipe.report_type`` selects a frozen ``ReportBase`` dataclass.
Control validates each ``POST /reef/report`` before storing it:

.. code:: python

   from dataclasses import dataclass

   from reef.core.reports import ReportBase
   from reef.recipe import WeightTrainingRecipe


   @dataclass(frozen=True)
   class MyReport(ReportBase):
       score: float
       difficulty: int | None = None


   class MyMethodRecipe(WeightTrainingRecipe):
       @property
       def report_type(self) -> type[ReportBase]:
           return MyReport

``score`` reads the payload's top-level field. Other fields read
``metadata.<field>``. Supported types are finite ``float``, ``int``, ``str``,
``bool`` and ``Mapping[str, str]``, including optional forms.
Fields without defaults are required. Override ``validate()`` and raise
``ReportValidationError`` for cross-field checks.

Use ``ScoredRolloutReport`` for a required score, or ``TeacherContextReport``
for teacher context and an optional score. The default ``report_type=None``
accepts any well-formed report.

Invalid reports return HTTP 400 without storage. Reef also rejects:

- References to records outside the scenario or to non-inference records.
- Duplicate references, including an earlier call in an already referenced
  trajectory. Reference only each trajectory's last call.
- ``metadata.training.eligible``. Reports cannot opt out of training.

An identical retry returns the original receipt. With multiple components,
any component's report type can admit the report. Other trainers release
reports that do not match their own type.

Sample-to-job stage
-------------------

Subclass ``reef.train.sample2job.Sample2Job`` to turn stored samples into
``JobData``. Control calls it when an automatic job is ready or a request
is queued.

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

Name the class in ``Recipe.pipeline_spec()``. Configure it through
``pipeline.sample2job``.

- ``from_config`` receives scenario context, the record stage and environment.
  Parse settings and pass sizes to the base constructor.
- ``make_job_data(samples, data_id)`` receives samples in store order.
  Return ``JobData`` with the supplied ID. Do not call models or train here.
  The submission consumes every supplied sample, including any your stage omits.
- ``job_parameters(data)`` returns training parameters. By default, request
  parameters override recipe defaults for the sample kind.
- ``status()`` reports stage state; ``close()`` releases its resources.

Class attributes declare supported inputs:

.. list-table::
   :header-rows: 1
   :widths: 40 35 25

   * - Attribute
     - Meaning
     - Default
   * - ``supported_training_modes``
     - Accepted training modes
     - ``auto``, ``manual``
   * - ``reads_requests``
     - Allow manual requests without samples
     - ``False``
   * - ``required_request_types``
     - Record kinds the trainer supplies
     - Inference and report
   * - ``accepts_trajectories``
     - Accept uploaded trajectories
     - ``False``
   * - ``accepts_imported_tasks``
     - Accept imported ``train`` tasks
     - ``False``

Reuse ``DistillJobData`` for samples in store order. Harness recipes use
``TraceJobData`` or ``TrainTaskJobData``. The default ``NoJobData`` never
becomes ready.

Job size
~~~~~~~~

Set these fields under ``pipeline.sample2job.config``:

- ``job_samples``: minimum samples for an automatic job; a positive integer.
- ``buffer_size``: maximum samples before completing any touched groups.
  The base default is ``job_samples``; the value cannot be smaller.

Stages without their own size default require ``job_samples``.
Harness and generated-task stages default it to 1; generated tasks default
the buffer to at least 4. Missing required sizes raise ``PluginConfigError``.
Pass ``max_job_samples`` to the base constructor for a stage with a fixed
capacity. Configured sizes above that limit raise ``PluginConfigError``.

Job size and optimizer batch size are independent. Whole groups can exceed
the configured job size. The backend divides job data into optimizer steps;
``scheduling.remainder`` controls a shorter final step.
With ``scheduling.packing_length``, batches count token packs while
``job_samples`` still counts samples.

The next job
~~~~~~~~~~~~

Automatic jobs select untrained samples in store order:

1. Keep accepted inputs of the first available sample kind.
2. Exclude kinds without job parameters and samples marked stale.
3. Take ``buffer_size`` samples, then complete every group they touch.
4. Submit when at least ``job_samples`` samples remain.

Imported ``eval`` tasks never train. The submission commit counts all reserved
samples as trained. A later job failure does not return them.
Other samples stay available for subsequent jobs.

Training modes
~~~~~~~~~~~~~~

``auto`` is the default. ``manual`` runs queued training requests and uses
the samples they name, regardless of ``job_samples`` or previous training.
Manual requests bypass staleness checks.

A request's samples must exist, have one kind, fit ``max_job_samples`` and
be accepted by the trainer. Empty requests require ``reads_requests=True``.
Invalid requests fail. Reef processes the oldest request first and exposes
it as ``JobData.request``.

``POST /reef/scenarios/{scenario}/update`` changes the mode persistently.
Reserved job data stays unchanged. Both modes collect samples; switching
to ``auto`` can train samples collected in ``manual``.
Unsupported modes raise ``NotImplementedError``; unknown names raise
``ValueError``.

Uploaded trajectories
~~~~~~~~~~~~~~~~~~~~~

Uploaded ATIF trajectories are data items, not records. No record stage
forms sets from them. See the `data API
<../reference/http-api-data-items.rst#http-api-data-items>`__.

Set ``accepts_trajectories=True`` to accept them. Without an accepting trainer,
the scenario refuses uploads with HTTP 400.

Control calls ``Record2Sample.uploaded_sample(trajectory)`` when reading an
upload. Keep this method pure. The default returns the trajectory unchanged.
A ``ValueError`` skips the upload with a warning and leaves it untrained.

Each upload counts as one sample and is never stale. Each accepting component
can train it once. Reward-based objectives require finite
``extra.reef.reward``; SFT does not.

Activation rule
---------------

Training appends artifacts to the snapshot chain. The activation rule decides
whether they also serve; subsequent training can use an inactive artifact.

Subclass ``ActivationRule`` and implement:

- ``from_config(config)``: build the rule from deployment configuration.
- ``activates(made) -> bool``: decide from ``MadeArtifact.component`` and
  ``changed_kinds``. Weight artifacts have no changed harness kinds.
- ``activates_evaluated(artifact_id, evaluation) -> bool``: decide after an
  artifact evaluation. The default returns ``False``.

Both decision methods must depend only on their arguments.
Select a built-in rule or ``package.module:ClassName``:

.. code:: yaml

   activation:
     module: evaluated
     config:
       min_mean_score: 0.8

Built-in rules:

- ``always`` activates new artifacts unless they change a harness kind in
  ``review_kinds``. This is the default.
- ``never`` requires manual activation.
- ``evaluated`` requires a finished evaluation with mean score at least
  ``min_mean_score``.

Unknown rules and invalid configuration raise ``RecipeConfigError`` at load.
Scenario settings can select built-in rules, but cannot name custom classes.

Harness job evaluation chooses the candidate tree to keep. The activation
rule separately decides whether that tree serves. See
`harness evaluation <write-a-harness-method.rst#evaluation>`__ and
`activation configuration <../reference/configuration-recipe.rst#the-activation-section>`__.
