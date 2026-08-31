Codebase structure
==================

This page says which package should own a change.


Choose a destination
--------------------

Start with the change's scope. Reef infrastructure belongs under ``reef/``;
Cookbook methods belong in that method's package under
``recipes/``:

.. diagram::

   <div class="ownership-map">
     <div class="ownership-step-label">1. Who should reuse the change?</div>
     <div class="ownership-scope">
       <div class="fig-node">One cookbook method<span class="fig-node-caption"><code>recipes/&lt;name&gt;</code><br/>policy, processor, preparer, examples</span></div>
       <div class="fig-node fig-emphasis">Reef or multiple methods<span class="fig-node-caption">shared behavior; choose a responsibility below</span></div>
       <div class="fig-node">Repository support<span class="fig-node-caption"><code>tests/</code>, <code>docker/</code>, or <code>pyproject.toml</code></span></div>
     </div>
     <div class="fig-edge ownership-continue">
       <span>if shared</span>
       <svg class="fig-arrow fig-arrow-next" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 5v13"/><path d="m7 13 5 5 5-5"/></svg>
     </div>
     <div class="ownership-step-label">2. What responsibility changes?</div>
     <div class="ownership-groups">
       <div class="ownership-group">
         <div class="ownership-group-title">Request and scenario control</div>
         <div class="ownership-packages">
           <div class="fig-node"><code>reef/service</code><span class="fig-node-caption">HTTP and process lifecycle</span></div>
           <div class="fig-node"><code>reef/scenario</code><span class="fig-node-caption">state, commit, recovery</span></div>
           <div class="fig-node"><code>reef/recipe</code><span class="fig-node-caption">method contract and binding</span></div>
         </div>
       </div>
       <div class="ownership-group">
         <div class="ownership-group-title">Execution and integrations</div>
         <div class="ownership-packages">
           <div class="fig-node"><code>reef/runtime</code><span class="fig-node-caption">backend-neutral model contracts</span></div>
           <div class="fig-node"><code>reef/train</code><span class="fig-node-caption">training loop, processors, backends</span></div>
           <div class="fig-node"><code>reef/harness</code><span class="fig-node-caption">descriptors, episodes, trajectories</span></div>
         </div>
       </div>
       <div class="ownership-group">
         <div class="ownership-group-title">Artifact lifecycle</div>
         <div class="ownership-packages">
           <div class="fig-node"><code>reef/artifact</code><span class="fig-node-caption">store and version bytes</span></div>
           <div class="fig-node"><code>reef/surface</code><span class="fig-node-caption">deliver or activate artifacts</span></div>
         </div>
       </div>
       <div class="ownership-group">
         <div class="ownership-group-title">Shared vocabulary</div>
         <div class="ownership-packages">
           <div class="fig-node"><code>reef/core</code><span class="fig-node-caption">shared values, wire shapes, errors</span></div>
         </div>
       </div>
     </div>
   </div>

``reef/`` should be self-contained, whereas ``recipes`` depends on ``reef/``
for building customized methods for deployment. For developing the reef infrastructure,
refer to the following table for which package owns which responsibility:

+----------------------+----------------------------------------------------------+--------------------------------------------+
| Package              | Owns                                                     | Does not own                               |
+======================+==========================================================+============================================+
| ``reef/core/``       | shared value types, wire shapes, artifact                | storage, I/O, runtime behavior             |
|                      | identity, root errors                                    |                                            |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/service/``    | HTTP routes, auth, streaming, process                    | training methods or domain logic           |
|                      | lifecycle                                                | tied to aiohttp                            |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/scenario/``   | scenario binding, commit ordering,                       | training algorithms, repository            |
|                      | recovery, checkpoint policy                              | implementations                            |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/recipe/``     | the contract a method implements, dotted                 | any particular method                      |
|                      | class resolution, and runtime instance binding           |                                            |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/train/``      | the trainer loop, processor engines, batch               | HTTP endpoints, deployment                 |
|                      | types, backend integrations                              | configuration parsing                      |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/runtime/``    | backend-neutral inference and training                   | a concrete training stack                  |
|                      | contracts                                                | integration                                |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/surface/``   | delivering a published artifact to the                   | proposing, evaluating, or                  |
|                      | process or client that uses it                           | selecting updates                          |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/artifact/``  | artifact bytes, repositories,                            | commit policy or delivery                  |
|                      | materialization, version heads                           | behavior                                   |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/harness/``    | harness descriptors, tree rendering,                     | recipe policy, the version chain           |
|                      | episodes, trajectories                                   |                                            |
+----------------------+----------------------------------------------------------+--------------------------------------------+

The extension points those packages expose are in `Python API
<../reference/python-api.rst>`__.


Adding a new subpackage under ``reef/`` or a new method under ``recipes/``
requires an RFC that states which layer owns the behavior.
