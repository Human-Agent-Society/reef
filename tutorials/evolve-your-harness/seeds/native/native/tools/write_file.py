from pathlib import Path


def run(args, workdir):
    root = Path(workdir).resolve()
    path = (root / str(args.get("path", ""))).resolve()
    if root not in path.parents:
        return "refused: path escapes the workspace"
    content = str(args.get("content", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} characters to {args.get('path', '')}"


NAME = "write_file"
DESCRIPTION = "Write a text file in the workspace, creating parent directories."
PARAMETERS = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
    "required": ["path", "content"],
}
CAPABILITIES = ["write"]
