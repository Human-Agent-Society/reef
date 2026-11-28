Control service: reports, jobs and activation
=============================================

Extend control to validate reports, build job data or choose which artifacts serve.

Report type
-----------

``Recipe.report_type`` selects a frozen ``ReportBase`` dataclass:

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

``score`` reads the top-level field; other fields read ``metadata.<field>``.
Supported types are finite float, int, str, bool and ``Mapping[str, str]``, including optional forms.
Fields without defaults are required.
For cross-field checks, override ``validate()`` and raise ``ReportValidationError``.
Invalid reports return 400 without storage.
See the `report API <../reference/http-api-records.rst>`__ for references and retry behavior.

Sample-to-job stage
-------------------

Subclass ``Sample2Job`` and select it in ``Recipe.pipeline_spec()``:

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

``make_job_data(samples, data_id)`` receives store-ordered samples and returns
``JobData`` with the supplied ID. Do not call models or train here.
Submission consumes every supplied sample, including omitted ones;
job failure does not return them.

``job_parameters(data)`` supplies backend parameters.
Declare accepted request types, trajectories and tasks through the
`stage's capability flags <../../reef/train/sample2job/base.py>`__.
Release resources in ``close``.

Job size
~~~~~~~~

Set ``pipeline.sample2job.config``:

- ``job_samples``: positive minimum for automatic jobs.
- ``buffer_size``: selection limit before completing touched groups; defaults to ``job_samples`` and cannot be smaller.

Stages without a job-size default require ``job_samples``.
Harness and generated-task stages default to one sample.
Generated tasks default the buffer to ``max(4, job_samples)``.
Invalid sizes raise ``PluginConfigError``.
Job size is independent of optimizer batches and token packing.

Automatic jobs select untrained, non-stale samples of one accepted kind.
Manual requests can retrain samples and bypass staleness.
See `Manual training <../reference/http-api-training.rst#manual-training>`__.
Uploaded trajectories require ``accepts_trajectories=True`` and bypass record sets.
Keep ``Record2Sample.uploaded_sample`` pure; ``ValueError`` skips the upload without consuming it.

Activation rule
---------------

Training publishes artifacts; activation decides whether they serve.
Subclass ``ActivationRule``:

- ``from_config(config)`` constructs the rule.
- ``activates(made)`` decides from component and changed part kinds.
- ``activates_evaluated(artifact_id, evaluation)`` decides after evaluation; default false.

Decision methods depend only on their arguments.
Select ``package.module:ClassName`` in deployment ``activation.module``.
Scenario overrides accept built-in rules only.
See `activation configuration <../reference/configuration-recipe.rst#the-activation-section>`__.
Harness evaluation chooses the tree to keep before activation decides whether it serves.
