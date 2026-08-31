# Security Policy

## Reporting a vulnerability

Do not report suspected vulnerabilities in an issue, pull request, discussion,
shared chat, or social-media post. If GitHub shows a **Report a vulnerability**
button to you on the repository Security page, use it. Otherwise, contact a
Reef maintainer through an existing private organizational channel and ask the
maintainer to open a draft GitHub Security Advisory before sending exploit
details. Include:

- the affected Reef version or commit;
- the affected deployment or trust boundary;
- reproduction steps or a proof of concept;
- the realistic impact and required attacker capabilities; and
- any known mitigations or workarounds.

Remove credentials, private data, provider tokens, and sensitive prompts from
the report. If reproducing the problem requires sensitive material, describe
how maintainers can create equivalent test data instead of attaching it.

Maintainers will acknowledge the report, assess its impact, and coordinate a
fix and disclosure through the private advisory. Please do not impose or
publish a disclosure deadline before maintainers have confirmed the issue and
agreed on a release plan.

## Supported versions

Security fixes target the latest released version and the `main` branch.
Maintainers may provide a workaround instead of a patch for older releases.
Users should upgrade to the latest release before reporting a vulnerability
that may already have been fixed.

## Public security work

Hardening, dependency updates, and follow-up work that does not reveal an
unpatched vulnerability may use the normal issue and pull-request process.
Maintainers decide when a private report can be referenced publicly.
