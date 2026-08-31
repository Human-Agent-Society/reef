diff --git a/vllm/v1/core/sched/request_queue.py b/vllm/v1/core/sched/request_queue.py
index fc2bc30..a5aae28 100644
--- a/vllm/v1/core/sched/request_queue.py
+++ b/vllm/v1/core/sched/request_queue.py
@@ -214,11 +214,96 @@ class PriorityRequestQueue(RequestQueue):
         return reversed(list(self))
 
 
+def _request_job_id(request: Request) -> "str | None":
+    """The conversation/job id carried by a request, if any.
+
+    Forwarded by the client via ``vllm_xargs`` -> ``sampling_params.extra_args``
+    (vanilla vLLM ignores it). Used for job-level FCFS ordering below.
+    """
+    sampling_params = getattr(request, "sampling_params", None)
+    extra_args = getattr(sampling_params, "extra_args", None) if sampling_params else None
+    if extra_args:
+        job_id = extra_args.get("job_id")
+        if job_id is not None:
+            return str(job_id)
+    return None
+
+
+class JobFCFSRequestQueue(FCFSRequestQueue):
+    """First-come-first-served at the *job* granularity.
+
+    A request is ordered by the arrival time of the FIRST request seen for its
+    ``job_id`` (the conversation it belongs to), not by its own arrival time, so
+    a later turn of an in-flight conversation is admitted ahead of a brand-new
+    conversation's first prefill -- keeping ongoing multi-turn work moving and
+    its (already cache-hot) prefix reused. This is the core of the "continuum"
+    job-aware scheduling idea. Requests without a ``job_id`` fall back to plain
+    per-request FCFS, so this is a safe drop-in default. Ordering uses
+    ``request.arrival_time`` only (never wall-clock), so it is deterministic and
+    changes only admission order, never generated tokens.
+    """
+
+    def __init__(self) -> None:
+        super().__init__()
+        self.job_id_first_entry_time: dict[str, float] = {}
+
+    def _note_job(self, request: Request) -> None:
+        job_id = _request_job_id(request)
+        if job_id is not None and job_id not in self.job_id_first_entry_time:
+            self.job_id_first_entry_time[job_id] = request.arrival_time
+
+    def _order_key(self, request: Request) -> "tuple[float, float]":
+        job_id = _request_job_id(request)
+        if job_id is None:
+            return (request.arrival_time, request.arrival_time)
+        first = self.job_id_first_entry_time.get(job_id, request.arrival_time)
+        return (first, request.arrival_time)
+
+    def add_request(self, request: Request) -> None:
+        self._note_job(request)
+        self.append(request)
+
+    def prepend_request(self, request: Request) -> None:
+        self._note_job(request)
+        self.appendleft(request)
+
+    def prepend_requests(self, requests: RequestQueue) -> None:
+        materialized = list(requests)
+        for request in materialized:
+            self._note_job(request)
+        self.extendleft(reversed(materialized))
+
+    def _select_index(self) -> int:
+        best_index = 0
+        best_key = None
+        for index, request in enumerate(self):
+            key = self._order_key(request)
+            if best_key is None or key < best_key:
+                best_key = key
+                best_index = index
+        return best_index
+
+    def peek_request(self) -> Request:
+        if not self:
+            raise IndexError("peek from an empty queue")
+        return self[self._select_index()]
+
+    def pop_request(self) -> Request:
+        if not self:
+            raise IndexError("pop from an empty queue")
+        index = self._select_index()
+        request = self[index]
+        del self[index]
+        return request
+
+
 def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
     """Create request queue based on scheduling policy."""
     if policy == SchedulingPolicy.PRIORITY:
         return PriorityRequestQueue()
     elif policy == SchedulingPolicy.FCFS:
-        return FCFSRequestQueue()
+        # Job-aware FCFS (continuum-style); identical to plain FCFS when requests
+        # carry no job_id.
+        return JobFCFSRequestQueue()
     else:
         raise ValueError(f"Unknown scheduling policy: {policy}")
diff --git a/vllm/v1/core/sched/scheduler.py b/vllm/v1/core/sched/scheduler.py
index 2b2cd63..fc137ef 100644
--- a/vllm/v1/core/sched/scheduler.py
+++ b/vllm/v1/core/sched/scheduler.py
@@ -331,6 +331,16 @@ class Scheduler(SchedulerInterface):
         # skipped and put back at the head of the waiting queue later
         skipped_waiting_requests = create_request_queue(self.policy)
 
+        # [reference] Per-step cap on fresh, *uncached* long prefills so a burst
+        # of newly-arriving long prompts cannot head-of-line block the decode of
+        # in-flight conversations (prefill/decode interference). A deferred
+        # prefill is re-queued at the head and retried next step, so it is never
+        # starved and the generated tokens are unchanged.
+        uncached_long_prefills_scheduled = 0
+        long_prefill_defer_threshold = (
+            self.scheduler_config.long_prefill_token_threshold or 2048)
+        max_uncached_long_prefills_per_step = 1
+
         # Next, schedule the WAITING requests.
         if not preempted_reqs:
             while self.waiting and token_budget > 0:
@@ -437,6 +447,24 @@ class Scheduler(SchedulerInterface):
                     num_new_tokens = min(num_new_tokens, token_budget)
                     assert num_new_tokens > 0
 
+                    # [reference] Defer a fresh, uncached long prefill when there
+                    # is in-flight decode work and the per-step cap is used.
+                    # num_computed_tokens == 0 means no prefix-cache hit (a brand
+                    # new prompt); ongoing conversations keep a cached prefix and
+                    # are never deferred. Only admission order changes.
+                    is_cold_long_prefill = (
+                        num_computed_tokens == 0
+                        and (request.num_tokens - num_computed_tokens)
+                        > long_prefill_defer_threshold)
+                    if (is_cold_long_prefill and self.running
+                            and uncached_long_prefills_scheduled
+                            >= max_uncached_long_prefills_per_step):
+                        self.waiting.pop_request()
+                        skipped_waiting_requests.prepend_request(request)
+                        continue
+                    if is_cold_long_prefill:
+                        uncached_long_prefills_scheduled += 1
+
                     # Schedule encoder inputs.
                     if request.has_encoder_inputs:
                         (encoder_inputs_to_schedule, num_new_tokens,
