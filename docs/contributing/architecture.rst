Architecture
============

This page describes the layout that the packages under ``reef/`` move to. For each package, it says what the package holds and what the package does not decide.

The code moves to this layout in steps. `Codebase structure <codebase-structure.rst>`__ maps the packages as they are today, and `What moves <architecture.rst#what-moves>`__ lists the steps that remain.

Layers
------

Reef has four layers. A package imports from its own layer and from the layers below it. It imports from no layer above it.

.. code:: mermaid

   flowchart TD
       accTitle: The four layers of Reef and the direction of imports

       Methods("<b>Method packages</b><br/><code>recipes/*</code>")

       subgraph Entry["Entry layer: how a request or a command reaches Reef"]
           direction LR
           Deploy("<b>deploy</b><br/><code>reef serve</code>")
           Api("<b>api</b><br/>HTTP process of each service")
       end

       subgraph ControlLayer["Control layer: the loaded scenarios"]
           Control("<b>control</b><br/>scenarios · workers · operations")
       end

       subgraph Domain["Domain layer: one package for each capability"]
           direction LR
           Recipe("<b>recipe</b><br/>method contract")
           Train("<b>train</b><br/>training jobs")
           Samples("<b>samples</b><br/>records to samples")
           Evaluation("<b>evaluation</b><br/>candidate evaluation")
           Harness("<b>harness</b><br/>harness tree and runs")
           Routing("<b>routing</b><br/>Semantic Router")
           Inference("<b>inference</b><br/>model requests")
       end

       subgraph Foundation["Foundation layer: shared mechanisms"]
           direction LR
           Observability("<b>observability</b>")
           Execution("<b>execution</b>")
           Storage("<b>storage</b>")
           Artifact("<b>artifact</b>")
           Tinker("<b>tinker</b>")
           Core("<b>core</b>")
       end

       Entry --> ControlLayer --> Domain --> Foundation
       Methods -. "implement the recipe contract" .-> Domain

The tree lists the packages from the top layer to the bottom layer. Within a layer, a package is above the packages that it imports.

.. code:: text

   reef/
   │
   │   # Entry layer: how a request or a command reaches Reef.
   ├── deploy/          # `reef serve`: read the configuration, start and stop the processes.
   ├── api/             # The HTTP process of each service: routes, authentication, streaming, pages.
   │
   │   # Control layer: the loaded scenarios.
   ├── control/         # Scenario state and commits, background workers, client operations.
   │
   │   # Domain layer: one package for each capability.
   ├── recipe/          # The contract that a method implements.
   ├── train/           # The trainer, the training backends, and job data from samples.
   ├── samples/         # Records become samples.
   ├── evaluation/      # Candidate evaluation.
   ├── harness/         # The harness file tree, the agent adapters, and runs.
   ├── routing/         # The Semantic Router integration.
   ├── inference/       # Inference contracts, serving, and engines.
   │
   │   # Foundation layer: shared mechanisms.
   ├── observability/   # Experiment logging and record tracing.
   ├── execution/       # Processes, workers, and sandboxes.
   ├── storage/         # The database and the work queues.
   ├── artifact/        # The files of each published version.
   ├── tinker/          # The Tinker SDK pieces that training and inference share.
   └── core/            # Shared value types and errors.

   recipes/             # One package for each method. Nothing under reef/ imports them.

Entry layer
-----------

deploy: ``reef serve``
~~~~~~~~~~~~~~~~~~~~~~

- **It reads the configuration.** It merges the deployment file with the command-line overrides, validates the values, and makes the settings of each process.
- **It starts and stops the processes.** It starts each service, waits until the service is ready, watches for exits, and cleans up.
- **It does not handle requests.** It starts the HTTP processes of ``api``, and they handle the requests.

api: the HTTP process of each service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **It holds everything that is HTTP.** Routes, authentication, streaming, the table that maps errors to status codes, and the harness pages are here.
- **A route is a thin adapter.** It reads the request, calls an operation in ``control``, and writes the response.
- **It builds the app of each service.** It makes the dispatcher and the app from the settings that ``deploy`` gives it.
- **It holds no logic that works without HTTP.** That logic is an operation in ``control``.

Control layer
-------------

control: the loaded scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **It holds each scenario.** ``control/scenario/`` holds the scenario's state, its bound components, its creation and its recovery. Its committer decides when a new version becomes the head.
- **It owns every scenario of a process.** The dispatcher is that owner. ``control/workers/`` holds its background workers: the training threads, the sync with the shared store, and the cleanup.
- **It holds each operation that a client can ask for.** ``control/operations/`` holds them: a model request, a change of data items, an artifact evaluation, an import. They contain no HTTP code.
- **Operations call the dispatcher.** The dispatcher does not call operations.

Domain layer
------------

recipe: the contract that a method implements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **A recipe is configuration only.** It says how records and their feedback become a new version of the weights or the harness.
- **Methods are outside this package.** ``recipes/`` holds one package for each method. Only the SFT and dataset recipes ship with Reef.
- **It does not run training.** ``control`` binds a recipe to a scenario, and ``train`` runs the jobs.

train: training jobs
~~~~~~~~~~~~~~~~~~~~

- **The trainer builds each job.** It takes the samples of the next job from the store, shapes them into the job's data, and prepares the commit.
- **A training backend runs the job.** Slime trains weights, Tinker trains LoRA weights on a remote service, and the harness backend evolves a harness tree.
- **Control adds jobs, and the training service runs them.** The ``jobs`` queue connects the two.
- **It does not make samples.** ``samples`` makes them.

samples: records become samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **It decides which records train, and turns them into samples.** It groups each report with its records, waits for judgments, and marks stale samples.
- **It runs in the samples service, apart from training.** Slow work, such as judges, generated tasks and rollouts, does not hold the training path.
- **It stores the samples as data items.** The trainer reads them from the store.
- **It does not build jobs.** ``train`` decides which samples make the next job.

evaluation: candidate evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **It runs and scores the rollouts of an evaluation.** The evaluation service takes each rollout from a queue and writes its score.
- **It holds the activation rules.** A rule decides whether an artifact that training makes serves requests.
- **It does not make candidates, and it does not publish them.** ``train`` makes a candidate, and the committer in ``control`` publishes it.

harness: the harness as a file tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **A harness is a tree of files.** The package defines the tree, the patches that change it, and the checks of its files.
- **An adapter maps the tree to one agent.** It says which files Reef may evolve and where the agent reads each file.
- **It runs a harness and reads the result back.** It starts the agent on a rendered tree, binds the agent's model, and reads the trajectory.
- **It holds no recipe policy.** A method decides how the tree changes.

routing: the Semantic Router integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **It validates the external-model routing of a scenario.** A client names its scenario and asks for the automatic model or for one target of the scenario.
- **It asks the router which target a request selects.** The router runs unmodified, and this package calls its routing preview.
- **It manages the router container of a managed deployment.** One owner starts the container, updates its policy, and stops it.

inference: model requests
~~~~~~~~~~~~~~~~~~~~~~~~~

- **It defines the inference contracts.** ``control`` and the recipes use them, and each integration implements them.
- **It serves saved weights on engines.** SGLang, vLLM, an engine on another host, and the hosted sampler of Tinker are the integrations.
- **It adapts requests.** An OpenAI or Anthropic chat request becomes a request to the engine.
- **It does not import training code.** ``train`` and ``inference`` do not import each other.

Foundation layer
----------------

The packages of this layer are shared mechanisms. They do not decide when Reef trains, evaluates or publishes.

observability: reports to external tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Experiment logging.** It turns each commit of a scenario into experiment events. W&B is the built-in provider.
- **Record tracing.** It exports records to OpenTelemetry.
- **Both are optional.** A deployment selects them in its configuration.

execution: how Reef runs things
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Processes and background jobs.** It owns child processes and cleans them up, and it runs asynchronous jobs off the trainer thread.
- **Workers.** A queue worker takes items from a work queue and runs them. An executor starts the workers of the training and inference components, in one process, in several processes, or on Ray.
- **Sandboxes.** A sandbox runs a program in isolation: bwrap or seatbelt on the local host, e2b or Modal on a remote one.
- **It does not interpret the work.** The caller builds the command and reads the result.

storage: the database and the work queues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **SQLite or PostgreSQL holds the state.** The state of each scenario is its records, commit history, samples, tasks and evaluations. The data store holds the data items of the whole deployment.
- **The coordination store connects the services.** No service calls another. Each one reads and writes the work queues, ``served_models`` and the control lease.
- **It does not decide how Reef uses the data.** The rule for which records train is in ``samples``.

artifact: version files
~~~~~~~~~~~~~~~~~~~~~~~

- **It stores the content of each publication.** The content is model weights or a harness file tree. Git LFS is the built-in backend.
- **Each publication is an immutable snapshot in a chain.** The head of a chain moves only by compare-and-swap, so two concurrent publications conflict with an error and none is lost.
- **It does not decide when the head moves.** The committer in ``control`` decides.

tinker: the shared Tinker SDK pieces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Tinker is an optional remote service that trains LoRA weights.** Its training backend is in ``train``, and its serving runtime is in ``inference``.
- **This package holds what both sides need.** The checkpoint manifest, the SDK session and the sampling are here.
- **It keeps the two sides apart.** ``train`` and ``inference`` import this package and do not import each other.

core: shared value types
~~~~~~~~~~~~~~~~~~~~~~~~

- **It holds types only.** Records, reports, snapshots, job data and the root errors are here. Nothing here does I/O.
- **The admission rule is concrete.** A type belongs here only when at least two packages that do not depend on each other need it.
- **Every package imports it.** It imports no other package of Reef.

What moves
----------

A new top-level package needs an accepted RFC (`Changes that need an RFC <../../CONTRIBUTING.md#changes-that-need-an-rfc>`__). The table lists each step from today's layout. The packages that the table does not name stay as they are.

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Today
     - Target
   * - The Semantic Router modules of ``reef/service/`` and ``reef/service/deploy/routing/``
     - ``routing`` (`RFC #1570 <https://github.com/Human-Agent-Society/reef/issues/1570>`__)
   * - The modules of ``reef/service/`` that contain no HTTP code: requests, data, artifact evaluations, imports
     - ``control/operations/``
   * - ``reef/dispatcher/`` and ``reef/scenario/``
     - ``control``
   * - ``reef/service/deploy/``
     - ``deploy``
   * - The rest of ``reef/service/``
     - ``api``
   * - ``reef/record2sample/``, ``reef/pipeline_context.py``, and the part of ``TrainingPipeline`` that makes samples
     - ``samples``
   * - ``reef/sample2job/`` and the part of ``TrainingPipeline`` that builds job data
     - ``train``
   * - ``reef/recipe/generated_tasks/``
     - ``recipes/``
   * - The modules of ``reef/storage/``, in one directory
     - Subpackages of ``storage``: the scenario stores, the data store, the coordination store, and the SQL backends

Two groups of imports point upward today, and they change before the files move:

- The Semantic Router modules import ``reef.service.wire`` and the deployment settings. RFC #1570 changes these imports first.
- ``reef/service/deploy/`` and the HTTP modules of ``reef/service/`` import each other. After the move, ``deploy`` imports ``api``, and ``api`` does not import ``deploy``.
