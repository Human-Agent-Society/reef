The core loop: a closer look
============================

Each scenario has its own records, training state and artifact history.
A receipt links feedback to a model exchange.

The core loop
-------------

.. code:: mermaid

   sequenceDiagram
       accTitle: From an agent request to a new version
       autonumber
       participant H as Agent harness
       participant R as Reef
       participant M as Model
       participant L as Learning recipe
       H->>R: Request for a scenario
       R->>M: Model request
       M-->>R: Model response
       R->>R: Record the exchange and artifact ID
       R-->>H: Response and receipt
       H->>R: Feedback report with the receipt
       opt Recipe has enough eligible data
         R->>L: Data for the next job
         L-->>R: Training result
         R->>R: Save any new artifact
         opt Activation rule selects it
           R->>R: Update the active selection
         end
       end

Weight recipes save checkpoints. By default, the next job starts from the newest checkpoint, even if an older one still serves.
Harness recipes compare file trees on tasks. Agents receive selected changes on their next pull.

Learn more
----------

- `Send a request and report feedback <quickstart.rst>`__.
- `Choose which artifacts serve <../developer-guide/recipe-control-service.rst#activation-rule>`__.
- `Inspect the snapshot chain <../advanced_topics/state-model.rst#the-snapshot-chain>`__.
