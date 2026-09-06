"""Recover a completed runner from a protected receipt, not a missing PID.

The root wrapper drops privileges only for the original runner. It records its
exit code and logs outside the candidate-writable episode tree before exiting.
The host never retries the runner command to recover a lost end event.
"""

import hashlib
import json
import re
import shlex
import uuid
from datetime import datetime
from types import SimpleNamespace

from reef.harness.executor import EpisodeLaunchError

PROTOCOL = "e2b-protected-completion-v1"
MAX_LOG_BYTES = 131072

# Only standard-library code crosses into the existing pinned sandbox runtime.
# Secrets remain in the command environment, never in this script or payload.
WRAPPER = r"""
import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
if os.getuid()!=0:raise RuntimeError('completion wrapper must own its receipt')
payload=json.loads(sys.argv[1]);token=payload['token']
if len(token)!=32 or any(c not in '0123456789abcdef' for c in token):raise ValueError('invalid token')
if hashlib.sha256(json.dumps(payload['argv'],sort_keys=True).encode()).hexdigest()!=payload['command_sha256']:
 raise ValueError('command identity differs from executable arguments')
runner_umask=os.umask(0o077)
root=Path('/run')/('reef-command-'+token);root.mkdir(mode=0o700)
def write(value):
 path=root/'state.tmp'
 with path.open('w') as stream:
  json.dump(value,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
 os.replace(path,root/'state.json')
 fd=os.open(root,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def unprivileged():
 os.setgroups([]);os.setgid(1001);os.setuid(1001);os.umask(runner_umask)
state={'protocol':'e2b-protected-completion-v1','token':token,'wrapper_pid':os.getpid(),
 'command_sha256':payload['command_sha256'],'input_sha256':payload['input_sha256'],
 'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'running'}
with (root/'stdout').open('wb') as out,(root/'stderr').open('wb') as err:
 child_env=dict(os.environ);child_env.update(USER='reef',LOGNAME='reef')
 child=subprocess.Popen(payload['argv'],cwd=payload['cwd'],env=child_env,
  stdout=out,stderr=err,preexec_fn=unprivileged)
 state['child_pid']=child.pid;write(state)
 code=child.wait();out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
state.update(status='complete',child_return_code=code,exit_code=code if code>=0 else 128-code,
 finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
write(state)
raise SystemExit(state['exit_code'])
"""

READ_RECEIPT = r"""
import json,os,sys
from pathlib import Path
if os.getuid()!=0:raise RuntimeError('completion receipt requires its owner')
token=sys.argv[1]
if len(token)!=32 or any(c not in '0123456789abcdef' for c in token):raise ValueError('invalid token')
root=Path('/run')/('reef-command-'+token);path=root/'state.json'
if not path.exists():print(json.dumps({'status':'missing'}));raise SystemExit
if path.stat().st_size>8192:raise ValueError('completion receipt exceeds limit')
state=json.loads(path.read_text())
reply={'status':state['status'],'receipt':state}
if state['status']=='complete':
 for name in ('stdout','stderr'):
  with (root/name).open('rb') as stream:
   size=os.fstat(stream.fileno()).st_size;stream.seek(max(0,size-131072));raw=stream.read(131072)
  reply[name]=raw.decode('utf-8',errors='replace');reply[name+'_truncated']=size>131072
print(json.dumps(reply))
"""


class RemoteCompletion:
    def __init__(self, sandbox, python, argv, cwd, input_sha256, remaining):
        if not re.fullmatch(r"[0-9a-f]{64}", input_sha256):
            raise ValueError("invalid episode input identity")
        self.sandbox, self.python, self.remaining = sandbox, python, remaining
        self.token = uuid.uuid4().hex
        self.command_sha256 = hashlib.sha256(json.dumps(list(argv), sort_keys=True).encode()).hexdigest()
        self.input_sha256 = input_sha256
        self.payload = {
            "token": self.token,
            "argv": list(argv),
            "cwd": cwd,
            "command_sha256": self.command_sha256,
            "input_sha256": input_sha256,
        }
        self.receipt = None
        self.result = None

    def command(self):
        # E2B identifies the launched shell process. Replace it explicitly so
        # the protected wrapper records that exact PID on every shell runtime.
        return "exec " + shlex.join([self.python, "-I", "-S", "-B", "-c", WRAPPER, json.dumps(self.payload)])

    def read_finished(self, pid):
        if self.result is not None:
            if self.receipt["wrapper_pid"] != pid:
                raise EpisodeLaunchError("completion receipt identifies another command")
            return self.result
        response = self.sandbox.commands.run(
            shlex.join([self.python, "-I", "-S", "-B", "-c", READ_RECEIPT, self.token]),
            user="root",
            timeout=min(20, self.remaining()),
            request_timeout=min(20, self.remaining()),
        )
        if len(response.stdout) > 12 * MAX_LOG_BYTES + 16384:
            raise EpisodeLaunchError("completion receipt response exceeds its limit")
        try:
            value = json.loads(response.stdout)
            if value.get("status") == "missing":
                return None
            receipt = value["receipt"]
            if (
                receipt["protocol"] != PROTOCOL
                or receipt["token"] != self.token
                or receipt["wrapper_pid"] != pid
                or type(receipt["wrapper_pid"]) is not int
                or receipt["command_sha256"] != self.command_sha256
                or receipt["input_sha256"] != self.input_sha256
                or type(receipt["child_pid"]) is not int
                or receipt["child_pid"] <= 0
                or receipt["status"] != value["status"]
            ):
                raise ValueError("completion identity mismatch")
            if value["status"] == "running":
                return None
            if value["status"] != "complete":
                raise ValueError("unexpected completion state")
            code, child_code = receipt["exit_code"], receipt["child_return_code"]
            if (
                type(code) is not int
                or not 0 <= code <= 255
                or type(child_code) is not int
                or code != (child_code if child_code >= 0 else 128 - child_code)
            ):
                raise ValueError("invalid recorded exit code")
            start = datetime.fromisoformat(receipt["started_at"])
            finish = datetime.fromisoformat(receipt["finished_at"])
            if start.tzinfo is None or finish.tzinfo is None or finish < start:
                raise ValueError("invalid completion timeline")
            if any(
                not isinstance(value[name], str)
                or len(value[name]) > MAX_LOG_BYTES
                or type(value[name + "_truncated"]) is not bool
                for name in ("stdout", "stderr")
            ):
                raise ValueError("invalid retained command output")
        except (KeyError, TypeError, ValueError) as error:
            raise EpisodeLaunchError("invalid protected command completion receipt") from error
        self.receipt = {
            **receipt,
            "stdout_truncated": value["stdout_truncated"],
            "stderr_truncated": value["stderr_truncated"],
        }
        self.result = SimpleNamespace(exit_code=code, stdout=value["stdout"], stderr=value["stderr"])
        return self.result

    def evidence(self, *, end_received):
        if self.receipt is None:
            raise EpisodeLaunchError("no completed receipt is available")
        return {
            **self.receipt,
            "command_end_received": bool(end_received),
            "receipt_sha256": hashlib.sha256(json.dumps(self.receipt, sort_keys=True).encode()).hexdigest(),
        }
