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

Invalid reports return HTTP 400 without storage; identical retries return
the original receipt. Follow the `report API's reference rules
<../reference/http-api-records.rst>`__.
With multiple components, any report type can admit a report.
Other trainers release reports that do not match their type.

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

Declare accepted inputs with ``required_request_types``, ``accepts_trajectories``
and ``accepts_imported_tasks``. Reuse ``DistillJobData`` for store-order samples
or ``TraceJobData`` for harness traces.
See the `stage contract <../../reef/train/sample2job/base.py>`__ for all hooks
and capability flags.

Job size
~~~~~~~~

Set these fields under ``pipeline.sample2job.config``:

- ``job_samples``: minimum samples for an automatic job; a positive integer.
- ``buffer_size``: maximum samples before completing any touched groups.
  The base default is ``job_samples``; the value cannot be smaller.

Stages without a size default require ``job_samples``.
Harness and generated-task stages default to 1; generated tasks use a buffer
of at least 4. Invalid sizes raise ``PluginConfigError``.
Set ``max_job_samples`` for a fixed-capacity stage.

Job size counts samples, independently of optimizer batch size or token packing.
Whole groups can exceed the configured buffer.

The next job
~~~~~~~~~~~~

Automatic jobs take untrained, non-stale samples of one accepted kind in
store order. The kind must have job parameters.
Selection takes ``buffer_size`` samples and completes touched groups.
At least ``job_samples`` must remain.

Imported eval tasks never train. Submission consumes all reserved samples,
even when the job later fails.

Training modes
~~~~~~~~~~~~~~

``auto`` is the default. ``manual`` runs queued training requests and uses
the samples they name, regardless of ``job_samples`` or previous training.
Manual requests bypass staleness checks.

Requested samples must exist, have one accepted kind and fit ``max_job_samples``.
Empty requests require ``reads_requests=True``.
Reef processes requests oldest first and exposes ``JobData.request``.
Both modes collect samples; switching modes preserves reserved job data.
See `Manual training <../reference/http-api-training.rst#manual-training>`__.

Uploaded trajectories
~~~~~~~~~~~~~~~~~~~~~

Set ``accepts_trajectories=True`` to accept uploaded ATIF data items.
Without an accepting trainer, uploads return HTTP 400.
They bypass record-set formation.

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

Decision methods depend only on their arguments.
Select ``package.module:ClassName`` in ``activation.module``;
invalid configuration raises ``RecipeConfigError`` at load.
Scenario overrides accept built-in rules only.
See `activation configuration <../reference/configuration-recipe.rst#the-activation-section>`__
for the built-in ``always``, ``never`` and ``evaluated`` rules.

Harness `evaluation <write-a-harness-method.rst#evaluation>`__ chooses which
tree to keep; activation separately decides whether it serves.
