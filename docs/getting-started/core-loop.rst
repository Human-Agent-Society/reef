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
       participant G as Training*
       participant E as Artifact evaluation

       opt Harness recipe: pull the served tree
         H->>S: GET /reef/harness for scenario
         S-->>H: Harness tree and release
         Note over H: Agent runs on that tree
       end
       Note over H,I: Serve and record each request
       H->>S: Inference request for scenario
       S->>S: Freeze current release
       S->>I: Provider-native request
       I-->>S: Provider response
       S->>S: Validate frozen release and store record
       S-->>H: Response and receipt
       H->>S: Feedback quotes the receipt
       S->>T: Eligible record
       opt Processor has a batch
         Note over S,E: Produce, evaluate, and select a candidate
         T->>G: Prepared step
         G-->>T: Candidate artifact ready
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

Training runtimes call a backend-neutral group handle. Worker launch and RPC
sit behind a configurable ``Executor``; the Slime backend uses the same
interface for its model worker groups. See
`Worker executors <../developer-guide/executors.rst>`__ for that boundary.

Learn more
----------------------

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
     - `Processors <../developer-guide/processors.rst>`__
   * - Evaluation
     - `Gate a candidate
       <../developer-guide/write-a-recipe.rst#gate-a-candidate>`__
   * - Release lifecycle
     - `The release chain
       <../advanced_topics/state-model.rst#the-release-chain>`__
