import subprocess


def run(args, workdir):
    command = str(args.get("command", ""))
    if not command.strip():
        return "refused: empty command"
    try:
        done = subprocess.run(
            ["bash", "-lc", command], cwd=workdir, capture_output=True, text=True, timeout=60, check=False
        )
    except subprocess.TimeoutExpired:
        return "timed out after 60s"
    out = (done.stdout or "") + (("\n" + done.stderr) if done.stderr else "")
    return f"exit {done.returncode}\n{out}"


NAME = "run_bash"
DESCRIPTION = "Run a bash command in the workspace and return its exit code and output."
PARAMETERS = {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
CAPABILITIES = ["exec", "write", "network"]
