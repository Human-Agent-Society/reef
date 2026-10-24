import subprocess
import sys
from pathlib import Path


def run(args, workdir):
    code = str(args.get("code", ""))
    if not code.strip():
        return "refused: empty code"
    # The other tools are plain modules beside this one: the block imports one by name and calls run(args, WORKDIR).
    tools = str(Path(__file__).resolve().parent)
    prelude = f"import sys\nsys.path.insert(0, {tools!r})\nWORKDIR = {str(workdir)!r}\n"
    try:
        done = subprocess.run(
            [sys.executable, "-c", prelude + code],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "timed out after 60s"
    out = (done.stdout or "") + (("\n" + done.stderr) if done.stderr else "")
    return f"exit {done.returncode}\n{out}"


NAME = "execute"
DESCRIPTION = "Run a Python code block in the workspace; the other tools are importable by name and WORKDIR is set."
PARAMETERS = {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}
CAPABILITIES = ["exec", "write", "network"]
