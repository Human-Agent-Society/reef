# Route requests between a student and a teacher per Scenario

Clients call a record-only Reef service directly. `reef serve` runs one
unmodified Semantic Router (SR) container next to Reef and asks it which model
an automatic request should use; Reef then calls that model and returns its
normal JSON or SSE response with a receipt. Each Scenario has its own targets,
default target and keyword rules:

- `model: auto` goes to the Scenario's default target (the student), unless
  the request's latest user message contains one of a rule's keywords; then
  it goes to that rule's target (the teacher). Reef asks SR through its HTTP
  routing preview and checks the answer against the manifest and the SR
  runtime it verified.
- `model: <target ID>` pins one of the Scenario's own targets. A pin never
  calls SR.

```text
client ─token, x-reef-scenario─▶ Reef (127.0.0.1:8900) ─▶ selected student or teacher service
                                   │
                                   └─ model: auto only: POST /api/v1/routing/preview ─▶ SR
```

Reef stores each request's Scenario, target, matched rule, manifest version
and SR runtime hash (`null` for a pin) in its record, and links feedback sent
to `POST /reef/report` to that record.

What this covers, and what it does not:

- Reef listens on `127.0.0.1` only, runs with the local `uni` executor, and
  calls existing OpenAI-compatible Chat Completions services. It does not start
  or load models.
- Routing uses a default target, keyword rules (SR's static selection) and
  explicit pins. Other SR policies, such as classifiers, confidence escalation,
  fallback or shadow requests, are not used.
- The service records inference and feedback. It does not train models.
- Any holder of a Reef service token may name any configured Scenario. Binding
  callers to Scenarios belongs to a gateway in front of Reef.

## Requirements

- Linux, with `docker`, `docker compose` (v2) and `docker buildx` usable by the
  user who runs `reef serve`.
- Reef installed from this checkout in a Python 3.12 environment, with the
  `third_party/semantic-router` submodule checked out.
- Free loopback ports: 8900 (Reef), 50053 (SR's HTTP API) and one per model
  service (8101-8104 below).

Run every command from the repository root.

## 1. Build the Semantic Router image

`third_party/semantic-router` pins the upstream source at commit
`c89d5d4c15117b0a7b1e09c87c6c863dd1efcb59`, unmodified.

The example manifest pins the local image used for the real-inference test:
`reef-direct-sr@sha256:090b81606403497c9b3cd48eb576ad1f4406024ee0f242cb714ba8fe3962dfe1`.
This is not a published image. On another host, build the pinned source below
and replace the example reference with your image's repository digest.

Build the pinned source into a local image and note its digest:

```bash
PIN=c89d5d4c15117b0a7b1e09c87c6c863dd1efcb59
git submodule update --init third_party/semantic-router
test "$(git -C third_party/semantic-router rev-parse HEAD)" = "$PIN"
test -z "$(git -C third_party/semantic-router status --porcelain)"
docker buildx build --load \
  -f third_party/semantic-router/src/vllm-sr/Dockerfile \
  --label org.opencontainers.image.revision="$PIN" \
  -t reef-semantic-router:"${PIN:0:12}" third_party/semantic-router
docker image inspect --format '{{json .RepoDigests}}' reef-semantic-router:"${PIN:0:12}"
```

The last command prints a list such as
`["reef-semantic-router@sha256:<64 hex digits>"]`; use that entry as the image
reference. The digest names the image you built. It is not published, so
another host must build its own image, which can get a different digest, or
pull it from a registry you control. An empty list (`[]`) means your Docker
keeps no repository digest for local builds. Then push the image to a registry
you control, for example a local `registry:2`, and use the digest that the push
reports. `reef serve` never pulls or builds the image; it must already be
present on the host.

Before creating a container, including during recovery, Reef checks that the
image's `org.opencontainers.image.revision` label exactly matches the pinned
source commit above. A missing or different label prevents startup. Keep the
label when building your image. This checks the image's declared source version;
it does not prove that the binaries were built from that source.

## 2. Start the model services

Each target needs an OpenAI-compatible service whose `GET /v1/models` lists its
model ID and whose `POST /v1/chat/completions` answers both ordinary and
`"stream": true` requests. Reef checks every target's `/v1/models` before it
accepts requests.

To try the flow without GPUs, start the deterministic CPU stand-in four times,
once per target of [`manifest.example.yaml`](manifest.example.yaml). It runs no
model; every answer is `"<model ID> answered."`, which shows which target was
used:

```bash
for target in 8101:code-student 8102:code-teacher 8103:math-student 8104:math-teacher; do
  python tutorials/semantic-router/cpu_provider.py --port "${target%%:*}" --model "${target#*:}" &
done
curl -sS http://127.0.0.1:8102/v1/models
```

They run in the background of this shell; stop them with `kill %1 %2 %3 %4`
when you are done.

## 3. Write the manifest

Copy [`manifest.example.yaml`](manifest.example.yaml) to a private location and
edit it:

- `targets` lists every model service: its `provider_model` (the model ID its
  API accepts), an immutable `revision` you choose to identify the deployment,
  its `url` without `/v1`, and `role`. Add `api_key_env` to name the
  environment variable that holds a service's API key.
- `scenarios.<name>` lists the Scenario's `targets`, its `default_target`, and
  `rules`. Each rule has a `name`, `keywords` and the `target` an automatic
  request matching a keyword uses. Scenarios may share a target.
- `deployment.semantic_router_image` is the `repository@sha256:...` reference
  from step 1. Image tags without a digest are rejected.

```bash
mkdir -p ~/reef-routing && chmod 700 ~/reef-routing
cp tutorials/semantic-router/manifest.example.yaml ~/reef-routing/manifest.yaml
```

Reef records the declared `revision`; it does not check which weights a
service actually loaded. `role` describes a target and does not start
training.

## 4. Configure and start Reef

Write `~/reef-routing/reef.yaml`:

```yaml
reef:
  host: 127.0.0.1                # required: routing serves loopback only
  port: 8900
  tokens: ["${REEF_TOKEN:?}"]
  routing_manifest: /home/you/reef-routing/manifest.yaml
  routing_deployment:
    deployment_id: opd-routing   # names the SR container and its saved state
    restart_policy: fail_stop    # or bounded: up to max_restarts SR recoveries
recipe: {implementation: recipe}
inference:
  # Reef's launcher needs a reachable default; routed requests use the manifest's targets.
  upstream_url: http://127.0.0.1:8101
  upstream_model: code-student
```

Start it in its own terminal:

```bash
export REEF_TOKEN="$(openssl rand -hex 32)"
reef serve -c ~/reef-routing/reef.yaml
```

`reef serve` renders the SR configuration, starts the SR container and the Reef
service, waits until SR runs exactly that configuration, publishes SR's runtime
hash to Reef, checks every model service and one preview per Scenario, and only
then accepts requests. It is ready when its log shows a line like
`[reef-routing] {..., "state": "ready"}`. Until then, and whenever it is not
accepting requests, inference returns 503. `GET /healthz` reports only that the
Reef process is alive.

SR's HTTP API is published on `127.0.0.1:50053` only, with two credentials that
`reef serve` generates at each start: one allows only previews, the other only
configuration and readiness. Reef's own lifecycle control uses a private Unix
socket, never `reef.port`. See the
[configuration reference](../../docs/reference/configuration.rst#managed-routing-lifecycle)
for every `routing_deployment` setting.

## 5. Send requests

In another terminal with the same `REEF_TOKEN`:

```bash
chat() {  # chat <scenario> <JSON body>
  curl -sS -i http://127.0.0.1:8900/v1/chat/completions \
    -H "Authorization: Bearer $REEF_TOKEN" -H "x-reef-scenario: $1" \
    -H 'Content-Type: application/json' -d "$2"
}

# Default: the student answers.
chat code-opd '{"model":"auto","messages":[{"role":"user","content":"Fix this loop."}]}'
# A rule keyword: the teacher answers. The keyword is part of the prompt the model sees.
chat code-opd '{"model":"auto","messages":[{"role":"user","content":"reef_teacher_required Fix this loop."}]}'
# A pin: this target answers, and SR is not asked.
chat code-opd '{"model":"code-teacher","messages":[{"role":"user","content":"Fix this loop."}]}'
# Another Scenario's target: 403 target_not_allowed, and no model is called.
chat code-opd '{"model":"math-teacher","messages":[{"role":"user","content":"Fix this loop."}]}'
# SSE: the same selection, streamed.
chat code-opd '{"model":"auto","stream":true,"messages":[{"role":"user","content":"Fix this loop."}]}'
```

With the CPU stand-in, the first and last answers contain
`"code-student answered."` and the second and third `"code-teacher answered."`.
An ordinary response carries its receipt in the `x-reef-agent-record-id`
header. A stream carries it in a `data: {"reef": {"agent_record_id": ...}}`
event just before `data: [DONE]`; Reef sends that event only after the record
is saved.

Send feedback for a receipt in the same Scenario:

```bash
curl -sS http://127.0.0.1:8900/reef/report \
  -H "Authorization: Bearer $REEF_TOKEN" -H 'x-reef-scenario: code-opd' \
  -H 'Content-Type: application/json' \
  -d '{"score": 1, "references": ["<receipt>"]}'
```

A routed Reef serves only `POST /v1/chat/completions`, `POST /reef/report` and
`GET /healthz`; other routes return 404. Apart from `model`, which Reef replaces
with the selected target's model ID, the model service receives the client's
JSON body unchanged.

## How SR checks keywords

Rules apply only to `model: auto`. Following the pinned SR source, confirmed
against the running image for the cases in parentheses:

- SR reads the latest user message that has text: string content, or the
  `text` of content parts whose type is `text`, `input_text` or missing. Earlier
  user messages are ignored (an earlier user message with the keyword selects
  the default).
- While any user message has text, system, developer and assistant messages
  and tool results are ignored (a system or developer message with the keyword
  selects the default).
- If no user message has text and none has an inline image, SR checks the
  system, developer and assistant texts together instead (a keyword in a
  system or assistant message then selects the rule). Tool results are never
  checked.
- Keywords match regardless of case (an uppercase keyword selects the rule). A
  keyword made of letters, digits or underscores matches only as a whole word.
- If several rules match, the rule listed first in the manifest is used. No
  match means the default target.

## Errors

Reef never retries a failed selection and never switches to another model.
Error bodies are plain text starting with a code, such as
`target_not_allowed: routing target is not available to the Scenario`.

| Status | Meaning |
| --- | --- |
| 400 | Missing `x-reef-scenario` or `model`; an unknown target; a client-sent `x-reef-routing-*`, `x-reef-sr-decision` or `x-selected-model` header; a repeated `x-reef-*` header; JSON with repeated keys or non-finite numbers; for `auto`, messages SR cannot read (`routing_input_unrepresentable`, `routing_input_rejected`). |
| 401 | Missing or wrong Reef token. |
| 403 | An unknown Scenario, or a pin to a target outside the Scenario. |
| 404 | A route a routed Reef does not serve. |
| 408 | The request body did not finish uploading within 30 seconds. |
| 413 | The body is over 64 MiB; for `auto`, the preview (Scenario entrypoint plus messages) is over SR's 10 MiB limit. The same body can still be pinned. |
| 429 | SR is already running its maximum number of previews (`preview_max_concurrency`). |
| 502 | SR answered with something Reef cannot use: a malformed or cut-off reply, or a selection that does not match the manifest and runtime. |
| 503 | Reef is not accepting requests (starting, updating, recovering or stopping); SR refused, reset or closed the connection before answering, as when it stops or restarts (`routing_unavailable`); SR runs a different runtime than Reef published (`routing_runtime_changed`); or the target is disabled. |
| 504 | SR did not answer the preview in time (`selection_timeout_seconds` plus 5 seconds). |

## Change the routing policy

To change rules, a Scenario's default target or its target membership, edit
the manifest named by `reef.routing_manifest` and send `SIGHUP` to `reef serve`:

```bash
kill -HUP <reef serve PID>
```

Reef stops admitting requests, waits for admitted requests and record writes,
updates SR's configuration, waits until SR runs it, publishes the new manifest
and runtime hash to Reef together, and reopens. Requests admitted earlier keep
the policy they started with; new requests get 503 meanwhile. An invalid or
unchanged manifest is logged (`policy_rejected`, `policy_unchanged`) and the
current policy keeps serving. If SR does not accept the update, Reef restores
the previous policy and reopens with it; if Reef cannot confirm what SR runs,
the deployment stops instead of reopening. A `SIGHUP` sent while `reef serve` is
starting is applied once it is ready; one sent while it is stopping is ignored.

Changes to targets, endpoints, revisions, Scenario names, the image or the
preview settings need a restart. Check a candidate configuration first; this
uses no Docker and changes no deployment, but cannot tell whether the image
exists or will become ready:

```bash
python -m reef.service.deploy.routing.routing_ops validate --config /path/to/candidate-reef.yaml
```

## Stop, inspect and clean up

Ctrl-C or `SIGTERM` closes admission, waits for admitted requests and record
writes (30 and 5 seconds by default), then removes the SR container and its
anonymous volumes. Model services and records are kept. If `reef serve` was
killed, the SR container can remain until the next start or an explicit
cleanup:

```bash
python -m reef.service.deploy.routing.routing_ops status --deployment-id opd-routing
python -m reef.service.deploy.routing.routing_ops cleanup --deployment-id opd-routing
```

Both use the same per-user ownership registry as `reef serve`
(`${XDG_STATE_HOME:-~/.local/state}/reef/routing/`). Cleanup refuses while an
owner is running and never touches containers it does not own. If a Docker
operation was interrupted, its result is unknown and the next start refuses to
proceed; cleanup keeps the operation ID that `status` reports. Only after
checking that Docker has finished that operation, acknowledge it:

```bash
python -m reef.service.deploy.routing.routing_ops cleanup --deployment-id opd-routing \
  --acknowledge-unknown "<operation-id-from-status>"
```

Do not delete the registry to get past a lock or an ownership error.

## Tests without Docker

The routing, preview client, lifecycle and policy-update tests use a fake SR
preview listener and local providers, so they need neither Docker nor SR:

```bash
python -m pytest tests/reef_service/routing -q
```

They check what Reef sends and records, not how the real SR answers.
