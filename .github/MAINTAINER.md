# Reef Code Maintenance Model

This document defines how Reef maintainers route, review, and merge community
contributions. Its goals are to give contributors a clear path to a decision,
keep review ownership explicit, and protect the compatibility and operability
of the project as it grows.

Roles describe work, not status. One person may hold more than one role for a
pull request.

## Current Merge Oncall

The current Merge Oncall is configured in
[`merge-oncall.json`](merge-oncall.json): **@BobbyZhouZijian**.

When a pull request is opened, reopened, or marked ready for review, the
`assign merge oncall` workflow assigns the configured account and posts or
updates a single on-call comment. The workflow uses `pull_request_target`
without checking out or executing pull-request code, so assignment also works
for contributions from forks without exposing a write token to their code.

## Roles

### Maintainer

Maintainers are accountable for project direction and the repository as a
whole. They can merge and close pull requests and are responsible for:

- deciding whether a contribution fits Reef's direction and maintenance
  budget;
- identifying when an RFC is required;
- selecting reviewers for affected areas;
- resolving cross-area or project-level disagreements;
- merging changes when the acceptance conditions are met; and
- reverting or coordinating a fix when a merged change causes a regression.

An approval from a maintainer accepts responsibility for the change, not only
its local implementation.

### Merge Oncall

The Merge Oncall is the maintainer actively responsible for moving one pull
request toward a clear outcome. The on-call:

- performs the initial scope and RFC check;
- identifies affected areas and requests the right reviews;
- distinguishes blocking feedback from optional follow-up work;
- keeps the pull request state clear when review stalls or changes direction;
- confirms that acceptance conditions are met; and
- merges, closes, or hands the pull request to another maintainer.

The Merge Oncall is an active coordination role. An area reviewer may protect
a specific subsystem without being responsible for driving the entire pull
request.

### Area reviewer

Area reviewers have demonstrated expertise in a subsystem, interface, or
operational concern. They are responsible for reviewing changes in that area
for correctness, compatibility, tests, documentation, and maintainability.

Area reviewers do not need merge permission. Their approval is a technical
signal to the Merge Oncall; the final repository decision remains with a
maintainer. An area reviewer must have repository write permission before they
can be listed in [`CODEOWNERS`](CODEOWNERS), but qualified reviewers may still
be requested manually without being code owners.

Current primary area ownership is:

- **@Benjamin-eecs** — harness composition, recipes, surfaces, and their
  focused examples and documentation;
- **@hanfeiyu** — training execution, runtime integration, artifact handling,
  pipeline reliability, and their focused documentation; and
- **@BobbyZhouZijian** — service and core boundaries, scenario lifecycle,
  repository automation, packaging, and cross-cutting changes.

## Issue process

Structured issue forms are the default intake path. Blank issues are disabled;
security reports go through private vulnerability reporting. A workflow applies
the primary `area:*` label selected by the author, but maintainers remain
responsible for correcting the routing and adding secondary areas.

Issue titles use exactly one of `[Bug]`, `[Feature]`, `[Performance]`,
`[Task]`, `[Experiment]`, `[Example]`, `[RFC]`, `[Roadmap]`, or `[Question]`.
The prefix describes the work type only. Areas, methods, projects, lifecycle
states, and priorities belong in labels or the issue body. During triage,
maintainers correct non-standard or multiple prefixes and keep the remaining
title concise and outcome-oriented. `[Roadmap]` is reserved for a time-bounded
coordination index; its linked outcomes retain their actual work type.

Use `type: experiment` for work whose primary deliverable is empirical evidence
and `type: example` for a maintained, runnable user-facing reference. Use
`type: task` for maintainer-approved internal work. Feature, bug, performance,
question, RFC, and roadmap forms apply their corresponding existing labels.
These type labels do not replace area or lifecycle labels.

Every new public issue starts with `status: needs-triage`. During triage, a
maintainer should choose a clear next state:

- `status: needs-reproduction` when the behavior needs a runnable reproducer
  or independent confirmation;
- `status: waiting-author` when the maintainer has asked the author a concrete
  question that blocks progress;
- `status: confirmed` when the problem has been reproduced or otherwise
  validated, but the solution or ownership is not yet agreed;
- `status: ready` when the scope is agreed and implementation can begin;
- `status: blocked` when an external decision or dependency prevents progress;
  or
- close with `duplicate`, `invalid`, or `wontfix`, linking the canonical issue
  or explaining the decision.

When an issue author replies to an issue marked `status: waiting-author`, the
automation removes that label and restores `status: needs-triage`. Maintainers
apply the intended lifecycle label when moving an issue forward; automation
removes the previous lifecycle label. `status: keep-open`, area, priority, and
lifecycle labels describe different dimensions and may coexist.

Use `help wanted` only after the problem and contribution boundary are clear.
Use `good first issue` only when the expected files, acceptance criteria, and
validation are specific enough that a new contributor is unlikely to need an
architecture decision. Ask contributors to describe their approach before
assignment when duplicate work is likely.

Priority describes maintainer intent rather than intrinsic value:

- `priority: p0` — an active incident, severe regression, or release blocker;
- `priority: p1` — high-impact work maintainers intend to prioritize;
- `priority: p2` — normal-priority accepted work; and
- `priority: p3` — useful backlog work without a delivery commitment.

Do not apply a public priority to an undisclosed vulnerability. Use
`ci: gpu-required` when acceptance needs evidence from a supported accelerator
environment that hosted CI cannot provide. The pull request records the exact
environment, command, revision, and result.

## Pull request process

### 1. Submission

The author follows [CONTRIBUTING.md](../CONTRIBUTING.md), fills in the pull
request template, links any relevant issue or RFC, and reports the checks that
actually ran. Work that is not ready for an acceptance decision should be a
draft pull request.

### 2. Initial triage

A maintainer checks whether the pull request:

- duplicates existing work;
- has a clear problem and focused scope;
- needs prior discussion or an accepted RFC;
- belongs in this repository and architectural layer; and
- has enough information to begin review.

The maintainer may ask the author to narrow the change, move design discussion
to an issue or RFC, or close work that does not fit project direction. The
reason should be stated clearly.

### 3. On-call assignment and review routing

The assignment workflow makes the configured Merge Oncall the pull request
assignee. The Merge Oncall requests reviews for every materially affected area.
GitHub also requests reviewers using [`CODEOWNERS`](CODEOWNERS). Its current
area-specific entries route harness-facing work to @Benjamin-eecs and training
and reliability work to @hanfeiyu, while cross-cutting work stays with the
Merge Oncall. The pull-request labeler applies the corresponding `area:*`,
`dependencies`, and `type: tests` labels from
[`labeler.yml`](labeler.yml), including when the changed-file set is updated.
Labels are routing hints rather than approvals. The on-call remains responsible
for checking that every affected area has a qualified reviewer and requesting
additional reviews when path rules are insufficient.

### 4. Technical review

Reviewers evaluate the change at the appropriate level:

- local correctness and failure behavior;
- public contracts and backwards compatibility;
- package and trust boundaries;
- tests, documentation, and migration needs;
- operational cost and observability; and
- long-term maintenance burden.

Review feedback should be actionable and identify whether it is blocking,
optional, or suitable for a follow-up. Authors should respond to substantive
comments and may respectfully challenge a request with technical reasoning.

### 5. Continuous integration

Required checks must pass on the reviewed revision. CI is a gate, not a
substitute for review. The Merge Oncall may request focused or environment-
specific validation when the standard checks do not cover the affected path.
Dependabot groups weekly GitHub Actions, Python, documentation-site, and Docker
updates into bounded pull-request queues. Dependency pull requests follow the
same ownership, review, and required-check rules as author-submitted changes;
they are not merged automatically.

### 6. Acceptance

A pull request is ready to merge when:

- an RFC is accepted if the change requires one;
- at least one maintainer approves the pull request;
- every materially affected area has an approval from a maintainer or an area
  reviewer selected by the Merge Oncall;
- all required checks pass;
- no blocking review or design discussion remains;
- tests cover the behavior change; and
- affected user and developer documentation is updated.

Passing these gates does not create an obligation to merge. A maintainer may
still decline a contribution because of project direction, compatibility,
scope, maintenance cost, or insufficient ownership after merge. The decision
and rationale should be recorded on the pull request.

### 7. Merge and follow-up

A maintainer merges the accepted revision and ensures linked issues are closed
or updated. Deferred work should be captured in explicit follow-up issues
rather than left only in review comments.

If a merge causes a serious regression, restoring a known-good state takes
priority over preserving the change. A maintainer may revert first and resume
design or debugging in a follow-up pull request.

## Fast-track changes

Small documentation fixes, clear test-only changes, reversions, release
blockers, and fixes for active regressions may receive an expedited review. The
Merge Oncall must state why the change is being fast-tracked and which review
or validation remains required. Any exception is still subject to repository
permissions and branch protections.

## Stalled pull requests

When a pull request is waiting on the author, the Merge Oncall should identify
the remaining work and leave a clear prompt before closing it. When it is
waiting on review, the on-call should request another qualified reviewer or
hand the role to another maintainer.

The stale workflow marks a non-draft pull request after 60 days without
activity and closes it after another 21 days. Drafts and work labeled `KIV`,
`RFC`, `status: blocked`, or `status: keep-open` are exempt. Closure is
housekeeping, not a judgment on the contributor, and does not prevent a current
version from being reopened or resubmitted.

## Inactive issues

The stale workflow marks an issue after 90 days without activity and closes it
after another 30 days. Issues labeled `KIV`, `RFC`, `help wanted`, `roadmap`,
`status: blocked`, `status: confirmed`, `status: keep-open`, or `status: ready`
are exempt. Maintainers should use an exemption only when the issue remains
intentional work or a durable decision record; ordinary backlog items should
return to triage when someone is ready to pursue them.

[`labels.json`](labels.json) is the canonical catalog for managed labels. Its
workflow creates or updates catalog entries after changes reach `main` and
reapplies the current routing rules to all open pull requests. It deliberately
does not delete labels that are absent from the catalog.

## Disagreements and escalation

Reviewers should first seek a decision through concrete technical discussion
on the issue, RFC, or pull request. The Merge Oncall may request additional
area review when expertise or ownership is disputed. Maintainers make the final
repository-level decision and record the rationale when consensus is not
possible.

Substantial changes to this maintenance model follow the
[RFC process](../docs/rfcs/README.rst). Routine clarifications may be proposed
directly as a pull request to this document.
