The core loop: a closer look
================================

The core loop
-------------

.. code:: mermaid

   sequenceDiagram
       accTitle: How Reef serves, records, trains, evaluates, and publishes
       autonumber
       participant H as Harness
       participant S as Scenario
       participant I as Inference
       participant T as Trainer
       participant G as Training component
       participant E as Artifact evaluation

       opt Harness recipe: pull the active release's tree
         H->>S: GET /reef/harness for scenario
         S-->>H: Harness tree and release
         Note over H: Agent runs on that tree
       end
       Note over H,I: Serve and record each request
       H->>S: Inference request for scenario
       S->>S: Freeze active release
       S->>I: Provider-native request
       I-->>S: Provider response
       S->>S: Validate frozen release and store record
       S-->>H: Response and receipt
       H->>S: Feedback quotes the receipt
       S->>T: Eligible record
       opt Pipeline has job data
         Note over S,E: Produce, evaluate, and select a candidate
         T->>G: Submit a training job
         G-->>T: Saved checkpoint
         T->>E: evaluate(candidate)
         E-->>T: Evaluation result
         T->>E: decide(candidate, result)
         E-->>T: Select or reject
         alt Candidate selected
           T->>S: Commit new release
         else Candidate rejected
           Note over S,I: Previous release keeps serving
         end
       end

For a weight recipe, the trainer does not run optimizer steps. It submits a
bounded training job to a training component, such as Slime or Tinker. The
component trains the job's data with its own training loop and saves checkpoints.
Each saved checkpoint is a candidate. Serving loads only a release that Reef
published, and the next job starts from the newest saved checkpoint. A
harness recipe uses its own backend in place of the training component: the
backend prepares a candidate tree for each step. See `Training components
<../developer-guide/executors.rst#training-components>`__ and `Serving saved
weights <../developer-guide/executors.rst#serving-saved-weights>`__.

Learn more
----------

This page stays at the level of the loop. Each step it shows is specified in
full elsewhere:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Topic
     - Page
   * - Inference
     - `Inference HTTP API <../reference/http-api.rst#inference>`__
   * - Feedback report
     - `Report HTTP API <../reference/http-api.rst#report>`__
   * - Scenario
     - `Scenarios <../reference/http-api.rst#scenarios>`__
   * - Learning / training in general
     - `Write a recipe <../developer-guide/write-a-recipe.rst>`__
   * - Record eligibility
     - `Training pipeline <../developer-guide/processors.rst>`__
   * - Activation
     - `Activate an artifact
       <../developer-guide/write-a-recipe.rst#activate-an-artifact>`__
   * - Release lifecycle
     - `The release chain
       <../advanced_topics/state-model.rst#the-release-chain>`__
