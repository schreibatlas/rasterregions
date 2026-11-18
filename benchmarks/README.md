# Opt-in benchmarks

Run the deterministic benchmark workloads from the repository root after installing the project:

```bash
python -m benchmarks.benchmark_workloads --repeat 5 --warmup 1
```

The command prints JSON with Python and NumPy versions, workload dimensions, foreground fraction, component or cut counts, individual samples, and the median time. Fixture construction is deterministic. Timings depend on the machine and should be compared only under controlled conditions.

The workloads are:

- Dense connected components: a fully foreground `512 × 512` Boolean mask.
- Sparse connected components: a `512 × 512` mask containing a regular grid of 1,024 separated `3 × 3` components.
- Partition rows: deterministic synthetic rows of height 96 and width 2,000 or 4,000, each partitioned into eight segments.
- Seven-row handwritten sheet: a generated `1081 × 1198` page with 63 slanted body anchors, variable widths, overlapping boxes, detached marks, and tiny noise. The JSON reports segmentation samples and median, per-row and total anchors, repairs, partition states visited, truncated local searches, and returned regions. Anchor-only execution reports zero partition states.

Warmups are excluded from the reported samples. The benchmark is intentionally absent from CI, and there are no timing thresholds. Synthetic workloads provide repeatable probes, not a substitute for representative application data. Before changing or replacing an algorithm, capture representative workloads, input dimensions, environment details, and before/after timings. Do not infer a performance improvement from a single synthetic case.
