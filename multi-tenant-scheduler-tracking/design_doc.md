# Multi-tenant-scheduler Design Doc

```text

Note: This document is written by Zhikai, except for state diagram refining, the unit-test coverage subsection, and the performance evaluation section, which received Codex assistance.
```

## Overview

Related RFC: 

In this document, we described the multi-tenant scheduling design decisions, error handling and test suites. This document also inlcude known bugs.

## Scenario Lifecycle

<img src="reef_scenario_lifecycle.png" alt="reeflifecycle">

Since the multi-tenant scheduler (referred to as "scheduler" in the upcoming paragraphs) runs separately from the reef services process, the scheduler and services need to sync up on the scenario status.

The whole lifecycle management and scenario state transition is managed by the `reconcile` function of the scheduler. The reconcile function is triggered only in 3 places:
1. (Scheduler) actively trigger reconcile with a given frequencies to ensure that the system resources is correctly partitioned based on updated scenario status
2. (Scheduler, tenant) *submit*: when a new scenario asks to launch, after the scenario is added to the queue, scheduler triggers a reconcile. The scenario will get scheduled to run (starting->admitted) if there is enough quota.
3. (User) *cancel*: the user will initiate a cancel operation to remove the scenario

Additionally, the reef scenario state listed here is different from the running/blocked state managed by the dispatcher inside each service. In other words, the running/blocked state should all belongs to the admitted state.

It is worth noting that the reclaim here is quite "conservative". Even if the scheduler initiate the reclaim, if the service end happens to have active evolve requests joining between the check time and the reclaim time, service will turn down the reclaim process and enforce the scheduler to keep the admitted state of the scenario.

## Design Decisions

### System Resources/scenario resource calligraph

The system/scenario resources is characterized by the "execution units" for simplicity. For each scenario, the execution units is evaluated by the recipe configuration file. Each tenant is 

Note: the current tenant id is registered actively from the user space instead of assigned by the scheduler.

### Admitting new scenarios when queueing happens

The scenario admission when queuing happened is two-phase. We will first create a `ScenarioAdmission` as the token. This scenario admission object will contain the metadata of the scenario including the scenario id, tenant id, etc,. However, we will not 

### Tenant waiter queue operation frequencies

**Enqueue operation.** This only happens whenever the incoming scenario request does not match the quota.

**Dequeue operation.** This only happens when the tenant quota increases. When a scenario is terminated, it will notify the scheduler and the scheduler will tries to dequeue from the front of the queue. It will dequeue as much as possible until the front of the queue cannot get scheduled to run with the tenant quota. (This design guarantees good fairness among scenarios requiring varying size of quotas. However, it may impair the overall reef local harness evolve throughput. In addition, this will cause head-of-line blocking if the front of the queue waits for a large quota)

Optional dequeue choice: everytime we have one scenario deleted, we will traverse through the whole queue and try to fit as much scenarios as possible. This will lead to higher throughput, but will likely to starve the scenarios waiting for large quota.

### Dynamic Quota Recycling

When the user submit a scenario, the scenario may get scheduled to run. After the scenario do not recieve an event, it will get blocked. Then the actual active resources will be available. If the job has been waiting for long (longer than the `idle TTL`), it should be put back to the waiting queue and release its resources and allow others to run.

After the service is reclaimed, it does not directly get enqueued back to the tenant's waiter queue. Instead, it will get inserted in a dict that stores the scenarios

### Responsibility between multi-tenant scheduler & Dispatcher
The scheduler is only in charge of granting new requests and making sure the fairness, throughput is guaranteed in multi-tenant environment. However, the exact local harness evolve is controlled by the dispatcher. Each service will have 1 dispatcher in charge of the lifecycles of multiple scenarios.

Though the dispatcher are directly in charge of the scenario lifecycle, the fairness requirement also pushes scheduler to regularly investigate on the idle state of the scenarios that occupied the resources. If exceeded the idle TTL, this scenarios will get evicted and joining the end of the waiter queue of the current scheduler. Other than that, the two key abstractions are segregated.

### Identification

2 types of tokens, scheduler tokens hold by every service and control tokens hold by the scheduler. Scheduler tokens is used when a tenant is registering a scenario of specific type of service. Control tokens is used by the scheduler to manage the launch/recycle operations of the specific scenario.

### Execution unit definition

The execution unit is the standardized resource unit used in the scheduler. Each tenant will be assigned a limited number of units to serve scenarios and each scenarios will request different number units based on the workloads they will be running.

At startup, the scheduler calculates the global budget as
`floor(logical_thread_count * max_logical_thread_percentage / 100)`, with the
percentage defaulting to 70%. Each service reports a fixed per-scenario cost
based on its recipe's resolved evaluation worker count, summed across supported
components when combined. These units control admission and quota accounting
without reserving physical resources.


## Scheduler Overhead evaluation

(Note: the performance experiment is LLM driven, but reviewed by human)

We measured actual Cordis/Pi evolves with a fixed-response local model, quota 3
per tenant and idle TTL 1 s. The reference used the same execution capacity and
TTL, with immediate idle observation. Times below are medians of three runs,
from submission until all evolves settled, including queue waiting.

| Workload | Reference time | Scheduler time | Maximum concurrent evolves |
| --- | --- | --- | --- |
| 1 tenant, 12 scenarios | 11.87 s | 14.50 s | 3 |
| 1 tenant, 36 scenarios | 36.27 s | 45.06 s | 3 |
| 3 tenants, 18 scenarios | 7.83 s | 8.65 s | 9 |
| 6 tenants, 36 scenarios | 15.05 s | 14.64 s | 18 |

For single-tenant queue rotation, median claim/start/reclaim transitions each took
roughly 10–20 ms, while detecting idle after settlement took about **0.82 s**
before the configured TTL began. This maintenance delay explains much of the
extra waiting. With 6 tenants, throughput reached **2.46 evolves/s**; the lower
median time than the reference reflects run variation, not proven acceleration.

With the real Qwen3-8B model and TTL 10 s, the 3-tenant/6-scenario case took
**87.65 s**, versus **87.32 s** in the reference (two-run medians). None of the
live-model pairs exceeded 5% overhead, but these small samples and the shared
Scheduler/Service test process do not establish a production bound.

Environment: M3 MBP (18GB) + Intel Core Ultra 185H (32GB, use llama.cpp to serve qwen8-3b model)

## Error Handling

### Scheduler

1. **Exceeding quota**: 2 levels of quota exceed. Tenant quota exceed happens when a request come in with existing tenant. Global quota exceed happens when a request comes from a tenant for the first time and tries to reserve the quota from global quota for the whole tenant.
   -> Solution: for tenant quota exceed, enqueue the scenario to the waiter queue. For global quota exceed, raise quota exceed error and reject the tenant initialization.
2. **Who is in charge of the end scenario scheduling, Multi-tenant scheduler OR Dispatcher?**: Dispatcher


## Tests

### Unit tests

The unit and component tests cover tenant/global quota accounting, weighted FIFO admission across services, and concurrent admissions without exceeding quotas. They check tenant isolation, duplicate creation, scenario ID reuse prevention, and stale control messages. Lifecycle tests verify idle TTL, busy-scenario protection, confirmed resource release, sleep, explicit wake, and cleanup failures. Persistence tests check restart recovery, retained records and training state, schema migration, cold instruction storage, delivery deduplication, and cancellation. Configuration and control tests cover static execution costs, service registration, snapshot failures, and shutdown of in-flight operations. Logging tests verify queue changes and exclusion of private settings.

### End-to-end tests

Test1: Single tenant, single service, 4 scenarios
* Short Description: hardcoding the quota to 3 execution units and the standard execution units for service (CordisRecipe) to 1. Scenario 4 will get into the queue and wait until scenario 1/2/3 get evicted due to idle TTL.
* Target: evaluate the queuing and reclaim functionality

Test2: 3 tenant, single service, 5 scenarios
* Short Description: similar to test1, but propagated to 3 tenants
* Target: evaluate the multi-tenant functionality

Test3: 3 tenant, 3 services, each tenant serving 5 scenarios of different services (close to production use)
* Short Description: similar to test2, but each tenant will contains scenarios of different services
* Target: test the sophisticated waiting and admitting routines

## Known Bugs

1. When the scenario number exceed the maximum number of scenarios, there is a problem of duplicate scenario id?


## Future implementation

1. Current only support default quota for every tenant. In the future, we should support tenant quota autoscaling when there is hotspots happening.
2. For simplicity, the execution units size for certain scenario is simply defined by the related service. If two scenarios belong to the same service, they should consume same number of execution units.
3. (Scalability) When the number of tenant increases, the current per scheduler lock will become the bottleneck of system throughput. The lock could be per tenant in the future for better performance.
