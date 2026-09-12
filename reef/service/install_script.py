"""Server-side rendering of the one-command harness install script.

``GET /reef/harness/install`` answers with a self-contained POSIX sh
script: the composition files ride inline as quoted heredocs, so running
the script makes no reef callback and carries no token. The binary's bytes
never come from reef: the script checks the locally installed binary
against the descriptor's pinned version and, only on absence or mismatch,
runs the vendor's own install command. Before writing, the script removes
the files a previous install's release file recorded that the new composition
lacks, exactly like the stdlib client pull, so installing an older version
never leaves a newer version's files behind. After writing, the script
verifies a sha256 over the sorted relative paths, byte lengths, and file
bytes against the value baked in at render time, and records the pulled
version in the same release file the stdlib client pull writes, plus what the
release requires of the person (``requires``) and the check offs
``reef-<adapter> setup`` recorded (``setup``, carried over from the previous
release file). A release with an item the release file on disk does not check off is
refused first of all, before the binary is installed or a directory is
made, with the setup list and the release that installs on a machine with
nothing set up as the message; the refusal needs python3 only. Rerunning when everything already matches
writes nothing at all, not even the release file, and says "already current".
The interpreter is decided once: the python3 the installing shell resolves,
followed through to the interpreter behind it and pinned by absolute path
into the wrapper, so a later shell with another python3 on PATH runs the one
that passed the import check here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from reef.core.requirements import parse_requires
from reef.harness.adapters.descriptor import AdapterDescriptor, DescriptorError, InstallSpec
from reef.harness.episodes.vendor_install import DEFAULT_PREFIX_ROOT, PREFIX_ENV

#: The script's install-prefix root in shell spelling, the same root reef's
#: own server-side vendor install uses, honouring the same environment
#: override: a server and a client on one machine share the installed binary
#: instead of each fetching the pin into its own tree.
_SHELL_PREFIX_ROOT = DEFAULT_PREFIX_ROOT.replace("~", "$HOME", 1)

#: Client-side bookkeeping file, byte-identical to what the stdlib client
#: pull writes: ``.reef-harness-release`` contains the release id and file list.
HARNESS_RELEASE_FILE = ".reef-harness-release"


def composition_checksum(files: Mapping[str, str]) -> str:
    """sha256 over the sorted relative paths, byte lengths, and file bytes.

    The stream is, for each path in sorted order, the utf-8 path plus one
    newline plus the decimal byte length plus one newline plus the file
    bytes. The length frame makes the stream injective: without it, moving
    bytes across a file boundary could leave the concatenation unchanged.
    The script's ``compose_stream`` function reproduces exactly this stream
    with ``printf``, ``wc -c``, and ``cat``.
    """
    digest = hashlib.sha256()
    for relative in sorted(files):
        content = files[relative].encode("utf-8")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\n")
        digest.update(str(len(content)).encode("ascii"))
        digest.update(b"\n")
        digest.update(content)
    return digest.hexdigest()


def _heredoc_delimiter(content: str) -> str:
    """A heredoc delimiter that provably never occurs in ``content``.

    Derived from the content's own hash and checked by substring search:
    the candidate lengthens while it still occurs, and when even the full
    digest occurs the digest is re-hashed until a free candidate exists.
    ``content`` is finite, so the search terminates.
    """
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    while True:
        for length in range(12, len(digest) + 1):
            candidate = f"REEF_EOF_{digest[:length]}"
            if candidate not in content:
                return candidate
        digest = hashlib.sha256(digest.encode("ascii")).hexdigest()


def _double_quoted(path: str) -> str:
    """Escape ``path`` for interpolation inside a double-quoted string."""
    return path.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")


def _wrapper_quoted(text: str) -> str:
    """Escape ``text`` for the wrapper body, which travels through an unquoted heredoc.

    The wrapper body is emitted by ``cat <<REEF_WRAPPER_EOF`` in the install script, so it is
    expanded once by the installing shell before it becomes the wrapper. Double the escaping
    ``_double_quoted`` adds for the wrapper's own double-quoted context, so the heredoc pass
    emits the quoted form and the wrapper receives the value byte exact.
    """
    quoted = _double_quoted(text)
    return quoted.replace("\\", "\\\\").replace("$", "\\$").replace("`", "\\`")


def _single_quoted(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def _write_file_block(target: str, content: str) -> str:
    """Shell that writes ``content`` byte-exact to ``$DEST/target``.

    Single-quoted heredocs expand nothing, so hostile composition text
    (backticks, dollars, quotes, naive EOF lines) lands verbatim. A heredoc
    body always ends with a newline; content without a trailing newline
    goes through command substitution, which strips exactly that one added
    newline (the content itself then has no trailing newline to lose), and
    ``printf '%s'`` writes the rest untouched.
    """
    delimiter = _heredoc_delimiter(content)
    opener = f"<<{_single_quoted(delimiter)}"
    redirect = f'"$DEST/{_double_quoted(target)}"'
    if content.endswith("\n"):
        return f"cat > {redirect} {opener}\n{content}{delimiter}\n"
    return f"printf '%s' \"$(cat {opener}\n{content}\n{delimiter}\n)\" > {redirect}\n"


def _compose_env_var(descriptor: AdapterDescriptor) -> tuple[str, str]:
    """The env var and compose subdirectory that point the binary at the composition.

    The compose directory is the deepest directory above the primary config
    target that an env entry relocates with a ``{root}/<dir>`` value: the
    target's own parent for pi and opencode, the home two levels up for dsh,
    whose config file sits inside a profile. That entry relocates the
    binary's whole composition at the episode root, and it is the only env
    entry the user-facing wrapper needs (session/state dirs use the binary's
    own defaults outside episodes).
    """
    return descriptor.compose_relocation()


def _wrapper_lines(
    descriptor: AdapterDescriptor,
    env_var: str,
    compose_dir: str,
    release_id: str,
    scenario: str,
) -> list[str]:
    """The reef-<adapter> wrapper: a capture proxy + report command.

    Written after the composition on every run, whenever its text differs from
    the one on disk: the wrapper depends on this machine (the binary, the
    interpreter), not on the composition, so a rerun on a current tree still
    picks up a moved binary or interpreter, and a rerun that changes nothing
    writes nothing. The wrapper calls ``reef.harness.client.wrapper`` through
    ``$PYTHON``, the interpreter ``_python_lines`` resolved by absolute path
    (with ``-P`` where it exists), so the shell that runs it later needs
    neither that interpreter nor the checkout on its PATH.
    """
    wrapper_name = f"reef-{descriptor.name}"
    wrapper = f'"$DEST/{_double_quoted(wrapper_name)}"'
    return [
        f"# The {wrapper_name} wrapper: capture proxy + report command. Rewritten whenever its text",
        "# differs: it depends on this machine (binary, interpreter), not on the composition.",
        'BINARY_ABS="$(cd "$(dirname "$BINARY")" && pwd)/$(basename "$BINARY")"',
        f'COMPOSE_ABS="$(mkdir -p "$DEST/{_double_quoted(compose_dir)}" && cd "$DEST/{_double_quoted(compose_dir)}" && pwd)"',
        "wrapper_text() {",
        "    cat <<REEF_WRAPPER_EOF",
        "#!/bin/sh",
        f"# {wrapper_name}: run {descriptor.binary} with the reef-evolved composition.",
        f"# Generated by reef harness install (adapter {descriptor.name}, release {release_id}).",
        f'# Usage: {wrapper_name} -p "fix the bug"     # run the agent (receipts captured)',
        f'#        {wrapper_name} report --score 0 --feedback "..."  # report last run\'s receipts',
        f'#        {wrapper_name} harness "what the harness should do"  # ask reef for a change',
        f"#        {wrapper_name} doctor  # check the install: interpreter, service, binary, tools, release",
        "# Runs the python3 the install resolved; rerun the install from another shell to change it.",
        'export REEF_HARNESS_BINARY="$BINARY_ABS"',
        'export REEF_HARNESS_COMPOSE="$COMPOSE_ABS"',
        f'export REEF_HARNESS_SCENARIO="{_wrapper_quoted(scenario)}"',
        f'export REEF_HARNESS_ADAPTER="{_double_quoted(descriptor.name)}"',
        f'export REEF_HARNESS_ENV_VAR="{_double_quoted(env_var)}"',
        'exec "$PYTHON"${SAFE_PATH:+ $SAFE_PATH} -m reef.harness.client.wrapper "\\$@"',
        "REEF_WRAPPER_EOF",
        "}",
        f'if [ ! -x {wrapper} ] || [ "$(wrapper_text)" != "$(cat {wrapper})" ]; then',
        f"    wrapper_text > {wrapper}",
        f"    chmod +x {wrapper}",
        "fi",
        f"# Symlink into ~/.local/bin so {wrapper_name} is on PATH, on every run: the link may have been",
        "# pointed elsewhere since the wrapper was written (an install into another directory), and",
        "# ln -sf costs nothing. The link target must be absolute: DEST defaults to the relative",
        "# ./reef-harness, and a relative target resolves against the link's own directory, so the",
        f"# link dangles and {wrapper_name} is not runnable from anywhere.",
        'DEST_ABS="$(cd "$DEST" && pwd)"',
        'mkdir -p "$HOME/.local/bin"',
        f'ln -sf "$DEST_ABS/{_double_quoted(wrapper_name)}" "$HOME/.local/bin/{_double_quoted(wrapper_name)}"',
        'case ":$PATH:" in',
        '    *":$HOME/.local/bin:"*) ;;',
        f"    *) echo \"reef: add '$HOME/.local/bin' to your PATH to run {wrapper_name} from anywhere\" >&2 ;;",
        "esac",
    ]


def _import_check_lines(wrapper_name: str) -> list[str]:
    """Refuse, before anything is installed or written, an interpreter that cannot import what the wrapper runs.

    The check takes ``$SAFE_PATH`` so the working directory cannot stand in
    for an installed package: run from a checkout, it would pass for an
    interpreter that has no reef at all. Nothing is installed on the person's
    behalf: the one line that would is printed for them to run. The
    distribution is ``reef-infra``; the import package is ``reef``.
    """
    return [
        f"# reef-client (the capture proxy) and reef (the wrapper) must import in $PYTHON, which {wrapper_name} runs.",
        "if ! \"$PYTHON\" $SAFE_PATH -c 'import reef_client.serve, reef.harness.client.wrapper' 2>/dev/null; then",
        f'    echo "reef: reef-client and reef-infra are not importable by $PYTHON, which {wrapper_name} runs; install them there:" >&2',
        '    echo "    \\"$PYTHON\\" -m pip install reef-client \\"reef-infra @ git+https://github.com/Human-Agent-Society/reef.git\\"" >&2',
        '    echo "or rerun this script from a shell whose python3 has them" >&2',
        "    exit 1",
        "fi",
    ]


def _ensure_binary_lines(descriptor: AdapterDescriptor, install: InstallSpec) -> list[str]:
    """The vendor-delegating install step: check the pin, else install through the vendor's channel."""
    prelude: list[str] = []
    # Extra condition the "already installed" gate ands onto the binary check.
    gate = ""
    if install.kind == "git":
        # A checkout installed editable into a venv; the checkout's .git goes so
        # the agent's own startup update check has nothing to fetch.
        pin = f"{install.repository} at {install.ref}"
        # ``--version`` reports the package version, which a git ref moves
        # independently of (hermes pins date tags but reports 0.21.0), so the
        # version match alone would leave a ref-only bump on the old checkout.
        # The installed ref is recorded beside the prefix and gates too, and is
        # cleared before installing so an interrupted install never reads back
        # as current.
        prelude = [
            f"PIN={_single_quoted(f'{install.repository}@{install.ref}')}",
            'PIN_FILE="$PREFIX/.reef-install-pin"',
        ]
        gate = ' && [ "$(cat "$PIN_FILE" 2>/dev/null || true)" = "$PIN" ]'
        steps = [
            '        rm -f "$PIN_FILE"',
            # git clone refuses a non-empty target, so the checkout (and the
            # venv installed editable off it) is cleared first: without this a
            # failed install or a pin bump wedges every rerun on "destination
            # path already exists". The npm branch is idempotent the same way.
            '        rm -rf "$PREFIX/src" "$PREFIX/venv"',
            f"        git clone --quiet --depth 1 --branch {_single_quoted(install.ref)} "
            f'{_single_quoted(install.repository)} "$PREFIX/src"',
            '        rm -rf "$PREFIX/src/.git"',
            '        "$PYTHON" -m venv "$PREFIX/venv"',
            '        "$PREFIX/venv/bin/python" -m pip install --quiet -e "$PREFIX/src"',
            '        printf \'%s\\n\' "$PIN" > "$PIN_FILE"',
        ]
        # A Python CLI prints its version inside a label ("Hermes Agent v0.21.0"), so the match is a substring.
        pattern = f"    *{install.version}*)"
    else:
        pin = f"{install.package}@{install.version}"
        steps = [f'        npm install --prefix "$PREFIX" {_single_quoted(pin)}']
        pattern = f'    *" {install.version} "*)'
    probe_env = " ".join(
        f"{key}={_single_quoted(value)}" for key, value in descriptor.env.items() if "{root}" not in value
    )
    probe = f'{probe_env} "$BINARY"'.lstrip()
    return [
        f"# Ensure the pinned binary ({pin}) via the vendor's channel.",
        *prelude,
        "vendor_install() {",
        *[line.replace("        ", "    ", 1) for line in steps],
        "}",
        'installed=""',
        f'if [ -x "$BINARY" ]{gate}; then',
        f'    installed="$({probe} --version 2>/dev/null || true)"',
        "fi",
        'case " $installed " in',
        pattern,
        f'        echo "reef: {descriptor.binary} {install.version} already installed"',
        "        ;;",
        "    *)",
        '        mkdir -p "$PREFIX"',
        f'        spin "installing {descriptor.binary} {install.version} ({pin}) into $PREFIX, about a minute" vendor_install',
        f'        echo "reef: {descriptor.binary} {install.version} installed"',
        "        ;;",
        "esac",
        "",
        *(
            f'command -v {command} >/dev/null 2>&1 || echo "reef: warning: {descriptor.binary} wants {package} '
            f"({command}) on PATH and otherwise downloads it from GitHub at first start, which GitHub rate-limits; "
            f'install {package} with your package manager" >&2'
            for command, package in descriptor.client_tools
        ),
    ]


def _binding_lines(bindings: Mapping[str, str]) -> list[str]:
    """Shell that writes the model binding files over the pulled tree, the token filled from the environment.

    The binding is written after the checksum and on every run, so a rerun
    re-points an installed tree at the Reef the script came from; the
    checksum still covers the served composition alone."""
    if not bindings:
        return []
    lines = [
        "",
        "# The model binding: the adapter's config pointed at the Reef this script was",
        "# fetched from, with the client's own token; written on every run, after the",
        "# checksum, so the served composition stays what the release file records.",
        'if [ -z "${REEF_TOKEN:-}" ]; then',
        '    echo "reef: REEF_TOKEN is not set; the harness will reach Reef without a token" >&2',
        "fi",
    ]
    for relative in sorted(bindings):
        lines.append(_write_file_block(relative, bindings[relative]).rstrip("\n"))
        lines.extend(
            [
                f'"$PYTHON" - "$DEST/{_double_quoted(relative)}" <<\'REEF_BIND_EOF\'',
                "import os, sys",
                "path = sys.argv[1]",
                'text = open(path, encoding="utf-8").read()',
                f'open(path, "w", encoding="utf-8").write(text.replace({TOKEN_PLACEHOLDER!r}, os.environ.get("REEF_TOKEN", "")))',
                "REEF_BIND_EOF",
            ]
        )
    return lines


#: The literal the model binding overlay carries where the client's own token goes; the script swaps in
#: ``$REEF_TOKEN`` at install time, so the served script itself never holds a credential.
TOKEN_PLACEHOLDER = "__REEF_TOKEN__"


def _spinner_lines() -> list[str]:
    """``spin LABEL CMD...``: run a slow step with a spinner on a terminal, or one static line elsewhere.

    The step's output goes to a temporary log that is printed only on
    failure, so npm and pip cannot smear the spinner; on a terminal the line
    is erased once the step ends, so a successful install leaves the
    announcements alone. ``set -e`` does not see the background job's exit
    status, so it is read explicitly and returned.
    """
    return [
        "# Run a slow step behind a spinner on a terminal (a static line elsewhere); its output shows only on failure.",
        "spin() {",
        '    label="$1"; shift',
        '    log="$(mktemp)"',
        "    if [ -t 1 ]; then",
        '        "$@" >"$log" 2>&1 &',
        "        pid=$!",
        "        i=0",
        '        while kill -0 "$pid" 2>/dev/null; do',
        "            case $i in 0) c='|' ;; 1) c='/' ;; 2) c='-' ;; *) c='\\' ;; esac",
        "            i=$(( (i + 1) % 4 ))",
        '            printf \'\\r%s reef: %s\' "$c" "$label"',
        "            sleep 0.2",
        "        done",
        '        wait "$pid" && status=0 || status=$?',
        "        printf '\\r\\033[K'",
        "    else",
        '        echo "reef: $label"',
        '        "$@" >"$log" 2>&1 && status=0 || status=$?',
        "    fi",
        '    if [ "$status" -ne 0 ]; then',
        '        cat "$log" >&2',
        '        rm -f "$log"',
        '        return "$status"',
        "    fi",
        '    rm -f "$log"',
        "}",
    ]


def _python_lines() -> list[str]:
    """Resolve the interpreter once, for the script's own python steps and the wrapper it writes.

    The python3 the installing shell resolves is followed through to the
    interpreter behind it (``sys.executable``: a version manager's shim on
    PATH would otherwise re-decide the interpreter at every run, and a venv's
    python reports the venv's own path) and pinned by absolute path, so a
    later shell with another python3 on PATH (a virtualenv no longer active,
    an IDE's terminal) runs the wrapper with the interpreter that passed the
    import check here. ``-P`` (Python 3.11 and newer) keeps the working
    directory off ``sys.path``, so a directory named ``reef`` beside the
    caller, a checkout's parent for one, never shadows the package. It is a
    flag rather than ``PYTHONSAFEPATH`` because an environment variable would
    reach the agent's own ``python3`` runs through the wrapper's environment.
    """
    return [
        "# One interpreter for the install and the wrapper it writes: the python3 this shell resolves,",
        "# followed through to the interpreter behind it (a version manager's shim would re-decide it at",
        "# every run), by absolute path. -P (Python 3.11 and newer) keeps the working directory off sys.path.",
        'PYTHON="$(command -v python3 || true)"',
        'if [ -z "$PYTHON" ]; then',
        "    echo 'reef: python3 not found on PATH' >&2",
        "    exit 1",
        "fi",
        'PYTHON="$("$PYTHON" -c \'import sys; print(sys.executable)\')"',
        "# A python3 that prints at startup (a sitecustomize, a banner) would name nothing runnable.",
        'if [ ! -x "$PYTHON" ]; then',
        "    echo \"reef: python3 did not name its interpreter (sys.executable read '$PYTHON'); rerun from a shell whose python3 prints nothing at startup\" >&2",
        "    exit 1",
        "fi",
        'SAFE_PATH=""',
        "if \"$PYTHON\" -P -c '' 2>/dev/null; then",
        '    SAFE_PATH="-P"',
        "fi",
    ]


def _release_info_tool_lines(wrapper_name: str) -> list[str]:
    """The ``release_info_tool`` shell function: the release file's ``requires`` bookkeeping in python.

    ``static`` hashes the release file on disk without its check offs, the text
    ``RELEASE_FILE_CHECKSUM`` was baked from; ``gate`` refuses, naming the setup
    list and ``FALLBACK``, the release that installs on a machine with
    nothing set up, when an item of ``REQUIRES`` is not checked off there (a
    check off meets an item when it names it and the check it recorded is
    the item's; one without a recorded check, from an older release file, counts
    by name); ``carry`` prints the check offs to carry over and ``merge``
    writes them into the new release file. JSON is no job for sed, and python3
    runs the wrapper anyway."""
    return [
        f"# The release file's requires bookkeeping ({wrapper_name} setup's check offs): JSON is no job for sed.",
        "release_info_tool() {",
        '    "$PYTHON" - "$@" <<\'REEF_RELEASE_INFO_TOOL_EOF\'',
        "import hashlib, json, sys",
        "mode, path = sys.argv[1], sys.argv[2]",
        "try:",
        '    with open(path, encoding="utf-8") as handle:',
        "        record = json.load(handle)",
        "except (OSError, ValueError):",
        "    record = {}",
        "if not isinstance(record, dict):",
        "    record = {}",
        'setup = [item for item in record.get("setup") or [] if isinstance(item, dict) and item.get("name")]',
        'if mode == "static":',
        "    # The record without the check offs is what RELEASE_FILE_CHECKSUM was baked from.",
        '    record.pop("setup", None)',
        '    print(hashlib.sha256((json.dumps(record, indent=2) + "\\n").encode("utf-8")).hexdigest())',
        'elif mode == "gate":',
        '    checked = {item["name"]: item for item in setup}',
        "    def met(item):",
        "        # A check off records the check it stood for; one without it (an older release file) counts by name.",
        '        record = checked.get(item["name"])',
        '        return record is not None and ("check" not in record or record.get("check") == item.get("check"))',
        "    unmet = [item for item in json.loads(sys.argv[3]) if not met(item)]",
        "    if unmet:",
        '        print("reef: this release requires:", file=sys.stderr)',
        "        for item in unmet:",
        '            check = item.get("check")',
        '            print("    " + item["name"] + " (" + item["kind"] + ")" + (": " + check if check else ""), file=sys.stderr)',
        '        fallback = "; with nothing set up yet, install ?release_id=" + sys.argv[4] + " first: it requires nothing" if sys.argv[4] else ""',
        f'        print("reef: run {wrapper_name} setup, then install again" + fallback, file=sys.stderr)',
        "        sys.exit(1)",
        'elif mode == "carry":',
        "    print(json.dumps(setup))",
        'elif mode == "merge":',
        '    record["setup"] = json.loads(sys.argv[3])',
        '    with open(path, "w", encoding="utf-8") as handle:',
        '        handle.write(json.dumps(record, indent=2) + "\\n")',
        "REEF_RELEASE_INFO_TOOL_EOF",
        "}",
    ]


def render_install_script(
    *,
    descriptor: AdapterDescriptor,
    files: Mapping[str, str],
    release_id: str,
    content_id: str,
    scenario: str = "",
    binding_files: Mapping[str, str] | None = None,
    requires: Sequence[Mapping[str, Any]] = (),
    fallback_release_id: str | None = None,
) -> str:
    """The complete install script for one adapter and one served manifest.

    The manifest side (``files``, ``release_id``) is adapter-agnostic;
    the descriptor contributes the binary's vendor install path. ``scenario``
    is baked into the wrapper so ``reef-<adapter> report`` knows which
    scenario to report to. ``binding_files`` are the adapter's config targets
    re-rendered with the model binding that points the harness at Reef; they
    carry ``TOKEN_PLACEHOLDER`` where the token goes, and the script writes
    them over the pulled files after the checksum, filling the placeholder
    from ``$REEF_TOKEN``. ``requires`` is the manifest's list of what the
    release needs from the person over its chain: the script refuses,
    before it installs or writes anything, while one item is not checked
    off in the release file on disk, naming ``fallback_release_id`` (the newest
    release in the chain that requires nothing) as the one to install on a
    machine with nothing set up, and records the list in the release file it
    writes. Raises ``DescriptorError`` when the descriptor declares no
    install section and ``ValueError`` when a composition path is absolute
    or escapes the destination through a ``..`` part, the same rule the
    stdlib client pull applies to served paths, or when an item of
    ``requires`` is not of the shape the training route admits (the union
    of a chain may exceed one request's cap).
    """
    if not release_id:
        raise ValueError("release_id must be a non-empty string")
    if not content_id:
        raise ValueError("content_id must be a non-empty string")
    install = descriptor.install
    if install is None:
        raise DescriptorError(f"adapter {descriptor.name!r} declares no install section")
    env_var, compose_dir = _compose_env_var(descriptor)
    wrapper_name = f"reef-{descriptor.name}"
    bindings = dict(binding_files or {})
    for relative in (*files, *bindings):
        if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
            raise ValueError(f"composition path {relative!r} escapes the destination")
    items = parse_requires(list(requires), limit=None)
    ordered = sorted(files)
    checksum = composition_checksum(files)
    release_info_text = (
        json.dumps(
            {
                "release_id": release_id,
                "content_id": content_id,
                "files": ordered,
                "requires": items,
            },
            indent=2,
        )
        + "\n"
    )
    release_info_checksum = hashlib.sha256(release_info_text.encode("utf-8")).hexdigest()
    # The composition paths a prune run keeps, as one case alternation; the
    # render charset contains no glob or quote characters, so each quoted
    # path is a literal case pattern. An empty composition keeps nothing.
    keep = "|".join(_single_quoted(relative) for relative in ordered) or "''"
    directories = sorted(
        {str(parent) for relative in ordered if (parent := PurePosixPath(relative).parent) != PurePosixPath(".")}
    )
    lines = [
        "#!/bin/sh",
        f"# Reef harness install: adapter {descriptor.name}, release {release_id}.",
        "# Self contained: the composition files ride inline below and the harness",
        "# binary comes from the vendor's own channel; running this script calls no",
        "# reef route and carries no token. Inspect freely, then run:",
        "#     sh install.sh [DEST] [PREFIX]",
        "set -eu",
        "",
        'DEST="${1:-./reef-harness}"',
        f'PREFIX="${{2:-${{{PREFIX_ENV}:-{_SHELL_PREFIX_ROOT}}}/{descriptor.name}}}"',
        f'BINARY="$PREFIX/{_double_quoted(install.binary_path)}"',
        f'CHECKSUM="{checksum}"',
        f'RELEASE_FILE_CHECKSUM="{release_info_checksum}"',
        f"REQUIRES={_single_quoted(json.dumps(items))}",
        f"FALLBACK={_single_quoted(fallback_release_id or '')}",
        "",
        "if command -v sha256sum >/dev/null 2>&1; then",
        "    sha256() { sha256sum | cut -d' ' -f1; }",
        "elif command -v shasum >/dev/null 2>&1; then",
        "    sha256() { shasum -a 256 | cut -d' ' -f1; }",
        "else",
        "    echo 'reef: neither sha256sum nor shasum found' >&2",
        "    exit 1",
        "fi",
        "",
        *_python_lines(),
        "",
        *_import_check_lines(wrapper_name),
        "",
        *_release_info_tool_lines(wrapper_name),
        "",
        *_spinner_lines(),
        "",
        f'echo "reef: harness release {release_id} for {descriptor.name}"',
        f"# The gate runs first of all: nothing is installed or written while an item is not checked off ({wrapper_name} setup).",
        f'[ "$REQUIRES" = "[]" ] || release_info_tool gate "$DEST/{HARNESS_RELEASE_FILE}" "$REQUIRES" "$FALLBACK" || exit 1',
        "",
        *_ensure_binary_lines(descriptor, install),
        "",
        "# The checksum stream, as baked into CHECKSUM: each sorted relative path,",
        "# its byte length, then its bytes, newline separated. The unquoted wc",
        "# substitution word-splits away the padding BSD wc prints.",
        "compose_stream() {",
        "    :",
        *(
            line
            for relative in ordered
            for line in (
                f"    printf '%s\\n' {_single_quoted(relative)}",
                f"    printf '%s\\n' $(wc -c < \"$DEST/{_double_quoted(relative)}\")",
                f'    cat "$DEST/{_double_quoted(relative)}"',
            )
        ),
        "}",
        "",
        'mkdir -p "$DEST"',
        *(f'mkdir -p "$DEST/{_double_quoted(directory)}"' for directory in directories),
        "",
        "# A rerun on a current machine writes nothing at all, not even the release file.",
        'current=""',
        'current_release_checksum=""',
        "if "
        + " && ".join(f'[ -f "$DEST/{_double_quoted(relative)}" ]' for relative in (HARNESS_RELEASE_FILE, *ordered))
        + "; then",
        '    current="$(compose_stream | sha256)"',
        f'    current_release_checksum="$(release_info_tool static "$DEST/{HARNESS_RELEASE_FILE}")"',
        "fi",
        'if [ "$current" = "$CHECKSUM" ] && [ "$current_release_checksum" = "$RELEASE_FILE_CHECKSUM" ]; then',
        '    echo "reef: composition already current"',
        "else",
        f'    echo "reef: writing the harness tree ({len(ordered)} file{"" if len(ordered) == 1 else "s"}) to $DEST"',
        "    # The check offs the release file on disk holds, carried into the new release file below.",
        f'    SETUP="$(release_info_tool carry "$DEST/{HARNESS_RELEASE_FILE}")"',
        "    # Prune the files a previous install's release file recorded that this",
        "    # composition lacks, exactly like the stdlib client pull. The release file",
        "    # is json.dumps at indent 2, so every file entry is one four-space",
        "    # indented quoted line.",
        f'    if [ -f "$DEST/{HARNESS_RELEASE_FILE}" ]; then',
        '        sed -n \'s/^    "\\(.*\\)",\\{0,1\\}$/\\1/p\' "$DEST/' + HARNESS_RELEASE_FILE + '" |',
        "            while IFS= read -r old; do",
        '                case "$old" in',
        f"                    {keep}) ;;",
        '                    *) rm -f "$DEST/$old" ;;',
        "                esac",
        "            done",
        "    fi",
        *(_write_file_block(relative, files[relative]).rstrip("\n") for relative in ordered),
        '    written="$(compose_stream | sha256)"',
        '    if [ "$written" != "$CHECKSUM" ]; then',
        '        echo "reef: composition checksum mismatch: $written != $CHECKSUM" >&2',
        "        exit 1",
        "    fi",
        "    # The same release file the stdlib client pull writes, plus requires and the check offs carried over.",
        _write_file_block(HARNESS_RELEASE_FILE, release_info_text).rstrip("\n"),
        f'    release_info_tool merge "$DEST/{HARNESS_RELEASE_FILE}" "$SETUP"',
        "fi",
        "",
        *_wrapper_lines(descriptor, env_var, compose_dir, release_id, scenario),
        *_binding_lines(bindings),
        "",
        'echo "reef: done"',
        f'echo "run:     $DEST/{wrapper_name}"',
        'echo "binary:  $BINARY"',
        'echo "harness: $DEST"',
        "",
    ]
    return "\n".join(lines)
