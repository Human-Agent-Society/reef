"""Server-side rendering of the one-command harness install script.

``GET /reef/harness/install`` answers with a self-contained POSIX sh
script: the composition files ride inline as quoted heredocs, so running
the script makes no reef callback and carries no token. The binary's bytes
never come from reef: the script checks the locally installed binary
against the descriptor's pinned version and, only on absence or mismatch,
runs the vendor's own install command. Before writing, the script removes
the files a previous install's sidecar recorded that the new composition
lacks, exactly like the stdlib client pull, so installing an older version
never leaves a newer version's files behind. After writing, the script
verifies a sha256 over the sorted relative paths, byte lengths, and file
bytes against the value baked in at render time, and records the pulled
version in the same sidecar the stdlib client pull writes. Rerunning when
everything already matches writes nothing at all, not even the sidecar,
and says "already current".
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import PurePosixPath

from reef.harness.descriptor import AdapterDescriptor, DescriptorError, InstallSpec
from reef.harness.vendor_install import DEFAULT_PREFIX_ROOT, PREFIX_ENV

#: The script's install-prefix root in shell spelling, the same root reef's
#: own server-side vendor install uses, honouring the same environment
#: override: a server and a client on one machine share the installed binary
#: instead of each fetching the pin into its own tree.
_SHELL_PREFIX_ROOT = DEFAULT_PREFIX_ROOT.replace("~", "$HOME", 1)

#: Client-side bookkeeping file, byte-identical to what the stdlib client
#: pull writes; must match ``reef_client.client.HARNESS_RELEASE_SIDECAR``.
HARNESS_RELEASE_SIDECAR = ".reef-harness-release"


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

    Written inside the install script's ``else`` branch (only when the
    composition changed), after the checksum verifies and before the sidecar.
    The wrapper calls ``reef.harness.harness_wrapper``, which starts a local
    proxy between the agent binary and Reef — capturing receipts so
    ``reef-<adapter> report`` can report without manual receipt handling.
    """
    wrapper_name = f"reef-{descriptor.name}"
    return [
        f"    # Write the {wrapper_name} wrapper: capture proxy + report command.",
        '    BINARY_ABS="$(cd "$(dirname "$BINARY")" && pwd)/$(basename "$BINARY")"',
        f'    COMPOSE_ABS="$(mkdir -p "$DEST/{_double_quoted(compose_dir)}" && cd "$DEST/{_double_quoted(compose_dir)}" && pwd)"',
        f'    cat > "$DEST/{_double_quoted(wrapper_name)}" <<REEF_WRAPPER_EOF',
        "#!/bin/sh",
        f"# {wrapper_name}: run {descriptor.binary} with the reef-evolved composition.",
        f"# Generated by reef harness install (adapter {descriptor.name}, release {release_id}).",
        f'# Usage: {wrapper_name} -p "fix the bug"     # run the agent (receipts captured)',
        f'#        {wrapper_name} report --score 0 --feedback "..."  # report last run\'s receipts',
        'export REEF_HARNESS_BINARY="$BINARY_ABS"',
        'export REEF_HARNESS_COMPOSE="$COMPOSE_ABS"',
        f'export REEF_HARNESS_SCENARIO="{_double_quoted(scenario)}"',
        f'export REEF_HARNESS_ADAPTER="{_double_quoted(descriptor.name)}"',
        f'export REEF_HARNESS_ENV_VAR="{_double_quoted(env_var)}"',
        'exec python3 -m reef.harness.harness_wrapper "\\$@"',
        "REEF_WRAPPER_EOF",
        f'    chmod +x "$DEST/{_double_quoted(wrapper_name)}"',
        f"    # Symlink into ~/.local/bin so {wrapper_name} is on PATH.",
        '    mkdir -p "$HOME/.local/bin"',
        f'    ln -sf "$DEST/{_double_quoted(wrapper_name)}" "$HOME/.local/bin/{_double_quoted(wrapper_name)}"',
        '    case ":$PATH:" in',
        '        *":$HOME/.local/bin:"*) ;;',
        f"        *) echo \"reef: add '$HOME/.local/bin' to your PATH to run {wrapper_name} from anywhere\" >&2 ;;",
        "    esac",
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
            '        python3 -m venv "$PREFIX/venv"',
            '        "$PREFIX/venv/bin/python" -m pip install --quiet -e "$PREFIX/src"',
            '        printf \'%s\\n\' "$PIN" > "$PIN_FILE"',
        ]
        # A Python CLI prints its version inside a label ("Hermes Agent v0.21.0"), so the match is a substring.
        pattern = f"    *{install.version}*)"
    else:
        pin = f"{install.package}@{install.version}"
        steps = [f'        npm install --prefix "$PREFIX" {_single_quoted(pin)}']
        pattern = f'    *" {install.version} "*)'
    return [
        f"# Ensure the pinned binary ({pin}) via the vendor's channel.",
        *prelude,
        'installed=""',
        f'if [ -x "$BINARY" ]{gate}; then',
        '    installed="$("$BINARY" --version 2>/dev/null || true)"',
        "fi",
        'case " $installed " in',
        pattern,
        f'        echo "reef: {descriptor.binary} {install.version} already installed"',
        "        ;;",
        "    *)",
        '        mkdir -p "$PREFIX"',
        *steps,
        "        ;;",
        "esac",
        "",
        "# Ensure reef-client (capture proxy) and reef (harness wrapper) are installed.",
        (
            "python3 -c 'import reef_client.serve, reef.harness.harness_wrapper' 2>/dev/null || "
            'python3 -m pip install --quiet --user reef-client "reef @ git+https://github.com/Human-Agent-Society/reef.git" 2>/dev/null || true'
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
        "# checksum, so the served composition stays what the sidecar records.",
        'if [ -z "${REEF_TOKEN:-}" ]; then',
        '    echo "reef: REEF_TOKEN is not set; the harness will reach Reef without a token" >&2',
        "fi",
    ]
    for relative in sorted(bindings):
        lines.append(_write_file_block(relative, bindings[relative]).rstrip("\n"))
        lines.extend(
            [
                f"python3 - \"$DEST/{_double_quoted(relative)}\" <<'REEF_BIND_EOF'",
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


def render_install_script(
    *,
    descriptor: AdapterDescriptor,
    files: Mapping[str, str],
    release_id: str,
    content_id: str,
    scenario: str = "",
    binding_files: Mapping[str, str] | None = None,
) -> str:
    """The complete install script for one adapter and one served manifest.

    The manifest side (``files``, ``release_id``) is adapter-agnostic;
    the descriptor contributes the binary's vendor install path. ``scenario``
    is baked into the wrapper so ``reef-<adapter> report`` knows which
    scenario to report to. ``binding_files`` are the adapter's config targets
    re-rendered with the model binding that points the harness at Reef; they
    carry ``TOKEN_PLACEHOLDER`` where the token goes, and the script writes
    them over the pulled files after the checksum, filling the placeholder
    from ``$REEF_TOKEN``. Raises ``DescriptorError`` when the descriptor
    declares no install section and ``ValueError`` when a composition path is
    absolute or escapes the destination through a ``..`` part, the same rule
    the stdlib client pull applies to served paths.
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
    ordered = sorted(files)
    checksum = composition_checksum(files)
    sidecar_text = (
        json.dumps(
            {
                "release_id": release_id,
                "content_id": content_id,
                "files": ordered,
            },
            indent=2,
        )
        + "\n"
    )
    sidecar_checksum = hashlib.sha256(sidecar_text.encode("utf-8")).hexdigest()
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
        f'SIDECAR_CHECKSUM="{sidecar_checksum}"',
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
        "# A rerun on a current machine writes nothing at all, not even the sidecar.",
        'current=""',
        'sidecar=""',
        "if "
        + " && ".join(f'[ -f "$DEST/{_double_quoted(relative)}" ]' for relative in (HARNESS_RELEASE_SIDECAR, *ordered))
        + "; then",
        '    current="$(compose_stream | sha256)"',
        f'    sidecar="$(sha256 < "$DEST/{HARNESS_RELEASE_SIDECAR}")"',
        "fi",
        'if [ "$current" = "$CHECKSUM" ] && [ "$sidecar" = "$SIDECAR_CHECKSUM" ]; then',
        '    echo "reef: composition already current"',
        "else",
        "    # Prune the files a previous install's sidecar recorded that this",
        "    # composition lacks, exactly like the stdlib client pull. The sidecar",
        "    # is json.dumps at indent 2, so every file entry is one four-space",
        "    # indented quoted line.",
        f'    if [ -f "$DEST/{HARNESS_RELEASE_SIDECAR}" ]; then',
        '        sed -n \'s/^    "\\(.*\\)",\\{0,1\\}$/\\1/p\' "$DEST/' + HARNESS_RELEASE_SIDECAR + '" |',
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
        *_wrapper_lines(descriptor, env_var, compose_dir, release_id, scenario),
        "    # The same sidecar the stdlib client pull writes: pulled version and file list.",
        _write_file_block(HARNESS_RELEASE_SIDECAR, sidecar_text).rstrip("\n"),
        "fi",
        *_binding_lines(bindings),
        "",
        f'echo "run:     $DEST/{wrapper_name}"',
        'echo "binary:  $BINARY"',
        'echo "harness: $DEST"',
        "",
    ]
    return "\n".join(lines)
