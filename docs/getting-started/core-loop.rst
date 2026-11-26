The core loop: a closer look
============================

Reef connects model inference, feedback and versioned updates.
The `quickstart <quickstart.rst>`__ covers requests and reports.
A learning recipe can produce and evaluate new artifacts.

The core loop
-------------

.. code:: mermaid

   sequenceDiagram
       accTitle: From an agent request to a new version
       autonumber
       participant H as Agent harness
       participant R as Reef
       participant M as Model
       participant L as Learning method

       opt Harness updates enabled
         H->>R: Pull the active harness tree
         R-->>H: Files and artifact ID
       end
       H->>R: Request for a scenario
       R->>M: Model request
       M-->>R: Model response
       R->>R: Store the exchange and its served version
       R-->>H: Response and receipt
       H->>R: Feedback report with the receipt
       opt Recipe has enough eligible data
         R->>L: Data for the next job
         L-->>R: Training result
         opt Result produces an artifact
           R->>R: Save the artifact
           alt Activation rule selects the artifact
             R->>R: Update the active selection
           else Keep current selection
             Note over R: The existing artifact stays active
           end
         end
       end

A weight recipe sends training jobs to a backend such as Slime or Tinker.
The backend saves checkpoints. Reef selects which published weights serve.
By default, the next training job starts from the newest saved checkpoint.

A harness recipe creates candidate file trees and compares them on tasks.
The agent uses an updated tree after its next pull.
Each scenario keeps its own records, training state and snapshot history.

Learn more
----------

Use these pages for the contracts behind each step:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Topic
     - Page
   * - Inference
     - `Inference HTTP API <../reference/http-api-inference.rst#http-api-inference-and-feedback>`__
   * - Feedback report
     - `Report HTTP API <../reference/http-api-inference.rst#report>`__
   * - Scenario
     - `Scenarios <../reference/http-api-scenarios.rst#http-api-scenarios-and-their-settings>`__
   * - Learning / training in general
     - `Write a recipe <../developer-guide/write-a-recipe.rst>`__
   * - Record eligibility
     - `Samples service <../developer-guide/recipe-samples-service.rst>`__
   * - Activation
     - `Activation rule
       <../developer-guide/recipe-control-service.rst#activation-rule>`__
   * - Snapshot lifecycle
     - `The snapshot chain
       <../advanced_topics/state-model.rst#the-snapshot-chain>`__
