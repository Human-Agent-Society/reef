from pathlib import Path


def run(args, workdir):
    root = Path(workdir).resolve()
    path = (root / str(args.get("path", ""))).resolve()
    if root not in path.parents and path != root:
        return "refused: path escapes the workspace"
    if not path.is_file():
        return f"no such file: {args.get('path', '')}"
    return path.read_text(encoding="utf-8", errors="replace")


NAME = "read_file"
DESCRIPTION = "Read a text file in the workspace."
PARAMETERS = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
CAPABILITIES = ["read"]
