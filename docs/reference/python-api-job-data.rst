Job data API
============

The values that a sample-to-job stage returns and a training component reads: ``JobData``, trajectory samples, task samples, the ATIF helpers and the shared chat rendering.

Job data
--------

.. code:: python

   from pathlib import Path
   from reef.train.types import (
       TaskSample, Sample, JobData, TrajectorySample,
       trajectories, trajectory_groups,
   )

``Sample = TrajectorySample | TaskSample``. Every sample-to-job stage returns a frozen ``JobData(data_id, samples, request=None)`` with an ordered tuple of samples:

- One job's data holds samples of one kind. Both readers, ``trajectories(data)`` and ``trajectory_groups(data)``, reject an unsupported sample kind. They do not filter mixed job data.
- ``request`` is keyword-only and carries an optional explicit training request. Request-only job data can be empty.
- The pipeline returns the same pending job data until its ID is acknowledged. Reef submits one training job for each value. The training component cuts the samples into its own optimizer batches.
- The component reads job data as a job data directory, so only the fields of ``JobData`` reach it. Keep job data serializable, with no handles to services, files, threads or models.
- Both sample kinds can carry ``source_agent_record_ids``. These record the derivation inputs. The pipeline's acknowledgement remains authoritative for consumption.

.. code:: python

   data = JobData(
       "scenario:data:1",
       (
           TrajectorySample(atif_document, source_agent_record_ids=("inference-1", "report-1")),
           TrajectorySample(other_atif_document, source_agent_record_ids=("inference-2", "report-2")),
       ),
   )

Trajectory samples
~~~~~~~~~~~~~~~~~~

A ``TrajectorySample.trajectory`` is an `ATIF document <https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md>`__ as a JSON-compatible mapping. The item checks the envelope; consumers validate the fields that their algorithms require. Standard steps describe messages, tool calls and observations. Captured records use ATIF-v1.7, matching the pinned Harbor version.

Reef-specific fields live inside ``trajectory.extra.reef``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Value
   * - ``reward``, ``feedback``
     - The supplied or computed training signal.
   * - ``source_agent_record_id``
     - The primary source.
   * - ``records``
     - The ordered source IDs, timestamps and original provider payloads, including provider-specific fields.
   * - ``teacher``
     - A distillation stage's teacher request: ``messages`` (null for a teacher that reads the student's own sequence), ``tools``, ``max_prompt_tokens`` and ``max_tokens``. OpenClaw-RL writes one request for each hint in ``teacher_candidates``.
   * - ``max_tokens``
     - The longest sample, in tokens, that a stage lets train.
   * - ``training``
     - What a recipe sets for its objective, such as ``rollout_created_at`` or ``distill_sample_weight``.

Which samples hold tokens:

- A sample that the record-to-sample stage makes holds the provider exchange. Its ``training`` has no ``tokens`` and no log-probabilities.
- The training component renders it (`Chat rendering`_). The rendered sample holds ``tokens`` and ``loss_mask`` in ``training``, and ``teacher_tokens`` or ``teacher_candidates`` when it has a teacher request.
- A task rollout of a training component carries its exact ``tokens``, ``loss_mask`` and ``rollout_log_probs``, and ``artifact_id`` (the artifact that sampled it). When the component records them, it also carries the sampler's ``topk_indices`` and ``topk_log_probs``.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Member
     - Contract
   * - ``item.metadata``, ``item.training``
     - Read the extension objects. Training arrays are JSON lists.
   * - ``with_metadata``, ``with_training``
     - Return updated documents and preserve the other extensions.
   * - ``item.artifact_id``
     - The artifact that produced the trajectory: ``training.artifact_id``, else the ``artifact_id`` of the last recorded call (``records[-1].payload``), or ``None`` for an upload. The samples service's staleness check reads it.
   * - ``group_id``
     - An optional string, scoped to the job data. ``None`` means an independent trajectory. Group-relative policy algorithms require an explicit ID on every trajectory and contiguous samples for each group, so advantages and rows stay aligned. ``trajectory_groups(data)`` reads these comparison groups.

TTTD and CORAL keep their group barriers and ordering. SAO and OpenClaw-RL schedule each trajectory independently, even when group metadata is present.

The ``reef.core.trajectories`` helpers construct and consume ATIF:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Helper
     - Contract
   * - ``make_trajectory(records, reward=None, feedback=None)``
     - Converts the exchanges with ``reef.core.atif_payloads``, as the record stores do, and writes the last record's ``tools`` as the agent's ``tool_definitions``. It keeps a summary of each call in ``extra.reef.records`` (``call_summary``): its ``agent_record_id``, ``created_at`` and ``artifact_id``, and its ``payload`` without the conversation, the tools and the response (``CONVERSATION_FIELDS``), such as ``model`` and ``chat_template_kwargs``.
   * - ``recorded_payloads(item)``
     - Each call's stored payload.
   * - ``recorded_payload(item)``
     - The last call as served, with the conversation that the steps project. For harness methods and for external ATIF without captured records.
   * - ``trajectory_reward(item)``
     - A finite policy reward.

``SampleAssembly.build`` (``reef.samples.assembly``) makes a report's sample: the trajectory's last call, or every call with ``accept_multi_turn_policy_samples``. A task rollout of a training component uses ``make_policy_trajectory`` or ``make_multi_turn_policy_trajectory`` from ``reef.samples.samples`` to keep the tensors that its engine returned.

Completed text-only streaming responses contribute their aggregated ``response.message`` to the ATIF agent step. The original SSE body remains in the captured record. Matching assistant messages in later request history are reused as context without duplicating those steps. An explicit ``training.response_message`` takes precedence over the stream summary.

Task samples
~~~~~~~~~~~~

A ``TaskSample`` names a `Harbor task directory <https://www.harborframework.com/docs/tasks>`__:

- ``task_path: Path`` is the directory with the instruction, environment and verifier configuration and files. The consuming algorithm's execution environment interprets the path.
- ``metadata`` is optional.
- ``files`` is a ``TaskFiles(partition, digest)`` that names the directory in the task file store, or ``None``. Only a task with ``files`` can go into a training job (`The job data directory <../developer-guide/training-backends.rst#the-job-data-directory>`__).

A ``TaskSample`` performs no file I/O or rollout. Algorithms that support tasks own validation, environment execution and conversion to trajectories. The Tinker component, and the Slime component with its own rollout engines, roll task samples out in task jobs. The harness backends refuse them. Accepting ``JobData`` does not imply that an algorithm supports both sample kinds.

Chat rendering
~~~~~~~~~~~~~~

Serving returns no tokens and no log-probabilities. A training component renders each trajectory sample with its model's chat template, and computes the log-probabilities that its objective needs with the weights of the job's start. `Rendering <../developer-guide/recipe-training-service.rst#rendering>`__ explains segments and the samples that a component leaves out. ``reef.core.chat_rendering`` holds the shared rendering:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Name
     - Contract
   * - ``ChatTemplate``, ``TokenizerChatTemplate(tokenizer, template_kwargs)``
     - Render messages and tools to token IDs. The second is the chat template of a Hugging Face tokenizer. Its ``template_kwargs`` must equal the chat template arguments that the serving engine renders with.
   * - ``render_trajectory(template, trajectory)``
     - One ``RenderedTrajectory`` for each segment. The agent's ``tool_definitions`` are the tools. Each agent step that is not ``is_copied_context`` trains: the loss mask is 1 on its tokens. A turn whose call's prompt extends the tokens of the turn before it continues that segment; otherwise it starts a new one. Raises ``ValueError`` when the trajectory has no trained step, or when the template does not render a turn after its own generation prompt.
   * - ``TeacherRequest``
     - The ``extra.reef.teacher`` value. Its ``teacher_tokens`` is the rendered teacher prompt followed by the student's reply tokens verbatim, or the student's own tokens when it names no messages.
   * - ``rendered_job_data(data, template)``
     - Renders each trajectory sample that has no ``tokens``. A trajectory of several segments becomes one sample for each segment, with ``segment``: ``{"index": i, "count": n}`` in its ``extra.reef.training``. Returns the job data and the positions of the samples that it leaves out. The Slime and Tinker components count them in the job metric ``unrenderable_samples``.
