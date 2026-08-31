# Kernel verifier boundary

`service.py` exposes one infrastructure-neutral endpoint:

```http
POST /evaluate
Authorization: Bearer <token>
Content-Type: application/json

{"solution": "<complete candidate source>", "runner_timeout_s": 520}
```

Start it with `python -m examples.guidance_ttt.verifier.service --task ...`.
VLIW invokes the pinned official Docker judge; TriMul invokes the vendored
official evaluator and therefore must run on an H100 with its pinned software
environment. Each response contains the official validity flag, raw metric,
and per-case artifacts expected by the corresponding Reef adapter.

The service has no cluster/cloud SDK and does not launch hardware. Bind to
localhost or a private interface, require bearer authentication, bound
concurrency, and place it behind the cluster's normal scheduler or load
balancer. Use one concurrency-1 TriMul worker per H100 so concurrent candidates
cannot contaminate timing.

Candidate programs are untrusted. VLIW runs in a networkless, resource-bounded
judge container. TriMul candidate subprocesses receive only an allowlisted
compiler/CUDA environment and never inherit verifier or model credentials, but
the service still must run in a disposable container or VM with outbound
network and unrelated filesystem access disabled.

See the [direct-node deployment guide](../deployments/qwen3_8b_lora/README.md)
for complete VLIW and TriMul commands.
