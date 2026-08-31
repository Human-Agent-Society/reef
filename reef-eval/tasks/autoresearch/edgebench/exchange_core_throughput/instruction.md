## Role

You are a performance engineer optimizing **exchange-core**, an LMAX Disruptor–based open-source financial matching engine (Java 8, Maven). Your goal is to maximize the throughput (MT/s — millions of transactions per second) of `PerfThroughput#testThroughputPeak`.

---

## Repository Layout

- `src/main/java/...` — engine source (OrderBook, matching service, risk engine, Disruptor wiring, etc.)
- `src/main/resources/` — logging etc.
- `pom.xml` — Maven build (Java 1.8 source/target)
- Tests have been removed from the work container — the judge will re-apply the official throughput test.

---

## Benchmark

The judge runs (with Java 17 runtime, compiled to 1.8 bytecode):

```bash
mvn -B test -Dtest=PerfThroughput#testThroughputPeak
```

The test calls `PerformanceConfiguration.throughputPerformanceBuilder().ringBufferSize(32 * 1024).build()` — only ringBufferSize is overridden; ME/RE, msgsInGroupLimit and other parameters use the builder defaults (4ME+2RE, msgsInGroupLimit=4096, BUSY_SPIN wait, DirectImpl orderbook, AffinityThreadFactory). **You can change these defaults by editing `throughputPerformanceBuilder()` in `PerformanceConfiguration.java`.** The test uses 100 symbols, 10K accounts, 10K orders, 3M transactions — a multi-symbol peak throughput scenario where thread topology changes (e.g. 4ME+2RE → 2ME+1RE) can have dramatic impact.

The test prints rounds like `"<i>. <X.XXX> MT/s"` and a final `"Average: <Z.ZZZ> MT/s"`. Higher is better.

---

## Optimization Directions (suggestions)

- Thread topology: matching engine (ME) / risk engine (RE) thread counts, pinning via `CoreWaitStrategy`
- Disruptor wait strategy, ring-buffer size, producer type
- OrderBook implementation (array-based, naive, direct)
- Message grouping / batching (message-group parameters)
- Java GC choice (G1 / ZGC) and heap sizing via `MAVEN_OPTS`
- Reduce allocations in hot path

Kimi K2.6 reference: 0.43 → 1.24 MT/s (+185%) by dropping from 4ME+2RE to 2ME+1RE.

---

## Rules

- Do NOT modify test files (the judge re-applies them)
- Do NOT change behavior: `mvn test` must still pass functional assertions
- You may freely edit `src/main/java/**`, `pom.xml`, and add `MAVEN_OPTS` via a `.mvn/jvm.config` or `.mvn/maven.config` file
- The score is `int(round(Average_MT_s * 1000))` (e.g. 1.243 → 1243)
- `CASES_OK = 1` only when `mvn test` exits 0 AND an MT/s line was parsed
