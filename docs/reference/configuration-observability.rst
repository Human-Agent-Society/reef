Observability settings
======================

Enable W&B to track training or OpenTelemetry to trace records. Configure either independently.
Use underscores in this section. Find all keys in the `settings reference <configuration-settings.rst#observability>`__.

Experiment tracking
-------------------

Install ``reef-infra[wandb]`` and provide W&B credentials to the service.

.. code:: yaml

   observability:
     wandb:
       enabled: true
       project: reef

Open ``experiment.url`` in the job response to inspect training metrics.
W&B records ``operations/*`` metrics every ten seconds for each loaded scenario.
Read ``operations/processor/ready_samples`` to see how many samples are ready for training.

Restarts resume the same run. Changing the project, entity, group prefix or name prefix starts a new run.
For offline runs, set ``entity`` and run ``wandb sync`` to use the run URL.

Record tracing
--------------

Install ``reef-infra[opentelemetry]`` and configure an OTLP/HTTP endpoint.

.. code:: yaml

   observability:
     tracing:
       enabled: true
       endpoint: ${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:?}
       authorization: ${REEF_TRACING_AUTHORIZATION:?}

Traces connect inference, feedback and the commits that used each record.
They include request messages, replies and feedback. Use a backend trusted to store that content.
Reef logs export failures and continues accepting records.
