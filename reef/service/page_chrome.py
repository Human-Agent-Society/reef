"""The chrome both harness pages share: the document wrapper, the design tokens and the status vocabulary.

``GET /reef/harness/releases/{step}/page`` and
``GET /reef/harness/requests/{record_id}/page`` are two views of one thing, a
request becoming a release, and a person moves between them by the link each
carries. They therefore share this module's header, palette, card shapes and
status wording, and each adds only the CSS its own sections need.

Both pages are self-contained: no asset request, no script, the logo inlined
so an installed wheel needs no docs checkout, and pure ASCII out so any
transport carries them. ``document`` is the only place the outer HTML is
written.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

# Inline the README logo so installed wheels need no docs checkout or external asset request.
LOGO = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 268 80" role="img" aria-labelledby="title">
  <title id="title">Reef</title>
  <g fill="none" stroke="#a03729" stroke-linecap="round" stroke-linejoin="round" stroke-width="3.5">
    <path d="M14 23.5c2.5 0 4 2.5 7 2.5s5.5-4.5 8.5-4.5S35.5 26 38.5 26s6-4.5 9-4.5c2.5 0 4 2 6.5 2"/>
    <path d="M14 36.5c2.5 0 4 2.5 7 2.5s5.5-4.5 8.5-4.5S35.5 39 38.5 39s6-4.5 9-4.5c2.5 0 4 2 6.5 2"/>
    <path d="M14 49.5c2.5 0 4 2.5 7 2.5s5.5-4.5 8.5-4.5S35.5 52 38.5 52s6-4.5 9-4.5c2.5 0 4 2 6.5 2"/>
  </g>
  <text x="78" y="53" fill="#14110e" font-family="Arial, Helvetica, sans-serif" font-size="38" font-weight="700" letter-spacing="7">REEF</text>
</svg>
"""

#: Palette, typography, header, hero, cards, tables and footer: what both pages draw with.
BASE_STYLE = """
:root{color-scheme:light;--bg:#f8f7f5;--card:#fff;--ink:#25221e;--mute:#716b64;--line:#e6e2dc;
--accent:#a03729;--soft:#f8eee9;--good:#24715a;--good-bg:#edf6f1;--warn:#926319;--warn-bg:#fcf4e5;
--status:#a03729;--status-bg:#f8eee9;--bad:#ab3e45;--bad-bg:#fceff0;--code:#f5f3f0;--shadow:0 2px 4px #25221e03,0 12px 32px #25221e03}
@media(prefers-color-scheme:dark){.brand svg g{stroke:#d99183}.brand svg text{fill:#f7f4f0}:root{color-scheme:dark;--bg:#141310;--card:#1c1a17;--ink:#eeeae4;
--mute:#b1a99f;--line:#37322d;--accent:#e8a092;--soft:#33231f;--good:#9ed3bc;--good-bg:#1f3028;
--warn:#e6c084;--warn-bg:#352b1d;--bad:#f0a3a8;--bad-bg:#392327;--code:#24211d;--shadow:none}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
a:focus-visible,summary:focus-visible,.table-scroll:focus-visible{outline:2px solid var(--accent);outline-offset:5px;border-radius:4px}
header{background:var(--card);border-bottom:1px solid var(--line)}
.topbar{max-width:1200px;min-height:72px;margin:auto;padding:16px 40px;display:flex;align-items:center;gap:28px}
.brand{display:flex;align-items:center;gap:10px;font-size:15px;font-weight:750;letter-spacing:.16em}
.brand svg{display:block;width:107.2px;height:32px}.divider{height:20px;width:1px;background:var(--line)}
.breadcrumb{display:flex;gap:12px;align-items:center;color:var(--mute);font-size:13px}.breadcrumb b{font-weight:500;color:var(--ink)}
.breadcrumb a{color:var(--mute)}.breadcrumb a:hover{color:var(--accent)}
.topbar .context{margin-left:auto;color:var(--mute);font-size:12px;max-width:240px;overflow-wrap:anywhere}
h1,h2,h3,p{margin-top:0}h1{font-size:34px;line-height:1.2;letter-spacing:-1.2px;font-weight:650;margin-bottom:12px}
h2{font-size:15px;letter-spacing:-.2px;margin:0 0 22px;font-weight:650}h3{font-size:12px;font-weight:600;color:var(--mute)}
.eyebrow{font-size:11px;letter-spacing:.15em;text-transform:uppercase;font-weight:650;color:var(--mute);margin-bottom:12px}
.hero{display:flex;align-items:center;justify-content:space-between;gap:24px;margin-bottom:30px}
.hero>div{min-width:0}.subtitle{color:var(--mute);font-size:14px;margin:0}
.status{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:100px;
padding:6px 12px;font-size:12px;background:var(--card);white-space:nowrap}
.status:before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}
.tone-selected,.tone-promoted{--status:var(--good);--status-bg:var(--good-bg)}
.tone-pending,.tone-skipped{--status:var(--warn);--status-bg:var(--warn-bg)}
.tone-rejected{--status:var(--bad);--status-bg:var(--bad-bg)}
.tone-queued,.tone-proposing,.tone-evaluating,.tone-running,.tone-settling,.tone-creation,.tone-promote,
.tone-rollback,.tone-recovery,.tone-unknown{--status:var(--accent);--status-bg:var(--soft)}
.status{color:var(--status);background:var(--status-bg);border-color:transparent}
.selected,.promoted,.complete{color:var(--good)}
.pending,.skipped,.partial,.queued,.proposing,.evaluating,.running,.settling{color:var(--warn)}
.rejected{color:var(--bad)}.creation,.promote,.rollback,.recovery,.unknown{color:var(--accent)}
/* A status word inside a pill takes the pill's tone, which the hero and the summary set. */
.status span{color:inherit}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:28px;min-width:0;box-shadow:var(--shadow)}
.text{white-space:pre-wrap;overflow-wrap:anywhere}
dt{font-size:11px;color:var(--mute);margin-bottom:5px}dd{margin:0;overflow-wrap:anywhere;font-size:12px}
.id{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;overflow-wrap:anywhere}
.tag{display:inline-block;font-size:11px;border:1px solid var(--line);padding:2px 7px;border-radius:4px;
color:var(--mute);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;overflow-wrap:anywhere;max-width:100%}
.operation-create{color:var(--good);background:var(--good-bg);border-color:transparent}
.operation-update{color:var(--warn);background:var(--warn-bg);border-color:transparent}
.operation-delete,.operation-remove{color:var(--bad);background:var(--bad-bg);border-color:transparent}
.empty{font-size:13px;color:var(--mute);margin:0}
.failure{border-left:2px solid var(--bad);padding-left:12px;margin:20px 0}
.failure h3{color:var(--bad);margin-bottom:5px}.failure p{font-size:12px;overflow-wrap:anywhere;margin-bottom:0}
summary{cursor:pointer;font-size:12px;font-weight:550}summary::marker{color:var(--mute)}
.table-scroll{overflow-x:auto;margin-top:16px}table{border-collapse:collapse;width:100%;font-size:12px}
th,td{text-align:left;vertical-align:top;padding:10px 8px;border-bottom:1px solid var(--line);overflow-wrap:anywhere}
th{font-size:11px;color:var(--mute);font-weight:550}td{min-width:80px;max-width:220px}
footer{display:flex;justify-content:space-between;gap:16px;margin-top:36px;padding-top:18px;border-top:1px solid var(--line);
font-size:11px;color:var(--mute)}footer p{margin:0}.footer-brand{letter-spacing:.1em;font-weight:600}
@media(max-width:800px){.topbar{padding:16px 24px;gap:18px}}
@media(max-width:480px){.topbar{padding:14px 20px;min-height:62px;gap:16px}
.topbar .context,.breadcrumb b,.breadcrumb .slash{display:none}
.hero{align-items:flex-start;flex-direction:column;gap:16px}h1{font-size:27px;letter-spacing:-.8px}
.subtitle{font-size:12px}.hero .status{font-size:11px;padding:5px 8px}.card{padding:22px}footer{font-size:11px}}
"""

#: What each state is called where a person reads it, so both pages name one state the same way.
STATUS_LABELS = {
    "queued": "Queued",
    "proposing": "Designing the change",
    "evaluating": "Checking the harness",
    "running": "In progress",
    "settling": "Saving the result",
    "selected": "Published",
    "pending": "Ready for review",
    "rejected": "Not selected",
    "skipped": "No changes",
    "complete": "Complete",
    "partial": "Partial",
    "creation": "Starting point",
    "promote": "Promoted by a person",
    "rollback": "Rolled back by a person",
    "recovery": "Recovered at boot",
}


def escape(value: Any) -> str:
    """``value`` as page text: None as the empty string, every angle bracket and quote as an entity."""
    return html.escape("" if value is None else str(value), quote=True)


def tone(state: str) -> str:
    """The first word of a state, which is its CSS class: ``promoted at step 5`` tones as ``promoted``."""
    return state.split(" ")[0]


def status_label(state: str) -> str:
    """The state in the words a person reads; an unlisted state stands as written, capitalized."""
    label = STATUS_LABELS.get(state)
    if label is not None:
        return label
    # A composed state, "promoted at step 5", keeps its tail: only the leading word is renamed.
    first, separator, rest = state.partition(" ")
    named = STATUS_LABELS.get(first)
    if named is not None and separator:
        return f"{named}{separator}{rest}"
    return state.capitalize()


def status_span(state: str) -> str:
    """The state as a toned span, the class its tone and the text its label."""
    return f'<span class="{escape(tone(state))}">{escape(status_label(state))}</span>'


def stamp(seconds: float) -> str:
    """A commit time as a person reads it, UTC, inside a ``<time>`` element carrying the machine form."""
    moment = datetime.fromtimestamp(seconds, UTC)
    return f'<time datetime="{moment.isoformat()}">{moment.strftime("%d %b %Y, %H:%M:%S UTC")}</time>'


def requires_table(items: list[Mapping[str, Any]]) -> str:
    """The ``requires`` items as setup reads them: name, kind, the check as written and the prompt setup shows."""
    rows = "".join(
        f"<tr><td>{escape(item.get('name'))}</td><td>{escape(item.get('kind'))}</td>"
        f"<td class=\"id\">{escape(item.get('check') or '')}</td><td>{escape(item.get('prompt') or '')}</td></tr>"
        for item in items
    )
    return (
        "<table><thead><tr><th>name</th><th>kind</th><th>check</th><th>prompt</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def document(
    *,
    title: str,
    style: str,
    breadcrumb: str,
    context: str,
    state: str,
    eyebrow: str,
    heading: str,
    subtitle: str,
    body: str,
    home: str = "",
    head: str = "",
    tail: str = "",
) -> str:
    """One page: the head, the branded header, the hero carrying ``state``, then ``body`` and the footer.

    ``style`` is appended to :data:`BASE_STYLE`, so a page declares only what
    its own sections need. ``breadcrumb`` is the section the page sits under
    and ``context`` the scenario shown beside it. ``home`` is where the logo
    and the "Harness" crumb lead, the served head's page; both stay plain
    text when it is empty, since the only pages a browser can open are the
    two ``/page`` routes and a catalog may have no served head to point at.
    ``head`` adds elements to
    the document head, a reload for instance, and ``tail`` follows the footer,
    where the release page puts its inline data. ``subtitle``, ``body``,
    ``head`` and ``tail`` are HTML the caller has already escaped; every other
    argument is escaped here. Pure ASCII out: other characters leave as
    numeric references.
    """
    if home:
        brand = f'<a class="brand" href="{escape(home)}" aria-label="Harness home">{LOGO}</a>'
        harness = f'<a href="{escape(home)}">Harness</a>'
    else:
        brand = f'<div class="brand">{LOGO}</div>'
        harness = "<span>Harness</span>"
    page = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="referrer" content="no-referrer">\n'
        f"{head}<title>{escape(title)}</title>\n<style>{BASE_STYLE}{style}</style>\n</head>\n<body>\n"
        f'<header><div class="topbar">{brand}'
        '<span class="divider" aria-hidden="true"></span>'
        f'<div class="breadcrumb">{harness}<span class="slash" aria-hidden="true">/</span>'
        f"<b>{escape(breadcrumb)}</b></div>"
        f'<span class="context">{escape(context)}</span></div></header>\n'
        f'<main class="tone-{escape(tone(state))}">\n'
        f'<div class="hero"><div><p class="eyebrow">{escape(eyebrow)}</p><h1>{escape(heading)}</h1>'
        f'<p class="subtitle">{subtitle}</p></div>'
        f'<div class="status" role="status">{status_span(state)}</div></div>\n'
        f"{body}"
        '<footer><p><span class="footer-brand">REEF</span> &nbsp; / &nbsp; Harness evolution</p>'
        "<p>Built with Reef</p></footer>\n"
        f"</main>\n{tail}</body>\n</html>\n"
    )
    return page.encode("ascii", "xmlcharrefreplace").decode("ascii")


__all__ = [
    "BASE_STYLE",
    "LOGO",
    "STATUS_LABELS",
    "document",
    "escape",
    "requires_table",
    "stamp",
    "status_label",
    "status_span",
    "tone",
]
