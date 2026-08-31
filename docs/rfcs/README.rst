RFC process
===========

Reef uses requests for comments (RFCs) to decide substantial changes before
implementation consumes contributor and reviewer time. An RFC records the
problem, constraints, proposed design, alternatives, and decision. It is a
design artifact, not a promise of staffing or a substitute for code review.

When an RFC is required
-----------------------

Start an RFC for a change that:

- adds a new top-level package under ``reef``;
- adds a bundled recipe, learning method, or training backend;
- makes a significant or backwards-incompatible public interface change;
- changes a persisted format, wire contract, trust boundary, or service
  topology; or
- establishes a project-wide policy that will constrain later work.

An RFC is usually unnecessary for a focused bug fix, documentation or test
improvement, behavior-preserving internal refactor, or implementation of an
already accepted RFC. When the boundary is unclear, open an issue describing
the intended change before drafting a full RFC.

Before writing
--------------

Search existing issues, pull requests, and RFCs. Discuss the problem with the
maintainers of the affected areas and identify the smallest decision the RFC
needs to make. Early discussion is not acceptance, but it can reveal conflicts
with current work or project direction before the proposal becomes expensive.

Submitting an RFC
-----------------

1. Copy ``0000-template.rst`` to a short descriptive filename in this
   directory.
2. Set the status to ``Draft`` and fill in the metadata and all relevant
   sections. Write ``Not applicable`` rather than silently omitting a required
   consideration.
3. Open a pull request containing the RFC only. Link the discussion issue, if
   one exists, and explain which maintainers or subsystems are affected.
4. Revise the RFC in response to review. Record important alternatives and
   tradeoffs in the document so that the reasoning survives the pull request.
5. A maintainer records the final decision and rationale. Update the RFC status
   before merging or closing the pull request.

Do not combine an RFC with its implementation. Small prototypes used to test
feasibility may be linked from the RFC, but they do not create a compatibility
commitment and are not merged as part of the design decision.

Lifecycle
---------

Every new RFC uses one of these statuses:

``Draft``
   The proposal is under discussion and may change substantially.

``Accepted``
   The affected maintainers agree with the direction and scope. Acceptance
   authorizes implementation work but does not guarantee staffing, priority,
   release timing, or final merge.

``Rejected``
   The proposal will not be pursued in its current form. The decision record
   must explain why.

``Implemented``
   The accepted design has shipped. The RFC links to the implementation and
   documents any material differences from the accepted proposal.

``Withdrawn``
   The author or maintainers ended consideration without a project decision.

``Superseded``
   A later RFC replaces this decision. Both RFCs link to each other.

Existing design records may predate this process and metadata format. New RFCs
and substantial amendments to existing RFCs follow this guide.

Decision process
----------------

RFC decisions are made by the maintainers accountable for every materially
affected area. They evaluate consistency with Reef's architecture, user value,
compatibility, security, operational cost, implementation feasibility, and
long-term maintenance burden.

Silence is not acceptance. An RFC becomes ``Accepted`` only after the affected
maintainers explicitly record acceptance in the pull request and the decision
record. If a disagreement cannot be resolved, the RFC remains ``Draft`` or is
closed as ``Rejected`` with the competing considerations documented.

After acceptance
----------------

Implementation uses normal pull requests and review requirements. Each pull
request links the RFC and states which part of the design it implements. If
implementation reveals a material change to public behavior, compatibility,
security, persistence, or topology, amend the RFC before merging that change.

When implementation is complete, update the RFC to ``Implemented``, link the
relevant pull requests, and record any differences between the accepted design
and what shipped. A stale accepted RFC can be withdrawn or superseded rather
than left as misleading project direction.
