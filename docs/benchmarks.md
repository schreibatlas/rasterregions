# Opt-in benchmarks

Run the deterministic benchmark workloads from the repository root:

```bash
python -m benchmarks.benchmark_workloads --repeat 5 --warmup 1
```

The command prints JSON with Python and NumPy versions, workload dimensions, foreground fraction, component or cut counts, individual samples, and median times. Foreground extraction workloads report mask bytes and decision-field bytes. Mask-only workloads report `decision_field_allocated: false` and zero decision-field bytes. Fixture construction is deterministic. Timings depend on the machine and should be compared only under controlled conditions.

The workloads are:
- Dense connected components: a fully foreground `512 × 512` boolean mask.
- Sparse connected components: a `512 × 512` mask containing a regular grid of 1,024 separated `3 × 3` components.
- Partition rows: deterministic synthetic rows of height 96 and width 2,000 or 4,000, each partitioned into eight segments.
- Seven-row handwritten sheet: a generated `1081 × 1198` page with 63 slanted body anchors, variable widths, overlapping boxes, detached marks, and tiny noise. The JSON reports runtime, per-row and total anchors, repairs, partition states visited, truncated local searches, and returned regions. This anchor-only workload should report zero partition states.
- Foreground mask and evidence: both APIs in `auto` mode on a deterministic `2200 × 1650` full-page raster and a `128 × 128` glyph crop. Each result reports sample times, median time, mask bytes, decision-field bytes, and whether a field is present. Mask-only output reports zero field bytes; a float32 evidence field reports four bytes per pixel.


A representative recorded run with Python 3.13.17 and NumPy 2.5.4 (`--repeat 3 --warmup 1`) produced:

| Input | Mask-only median | Evidence median | Decision-field bytes |
| --- | ---: | ---: | ---: |
| Full page, `2200 × 1650` | 0.246 s | 0.254 s | 14,520,000 |
| Glyph crop, `128 × 128` | 1.19 ms | 1.16 ms | 65,536 |

The mask-only output reported no decision field. These timings are one machine-specific sample, not performance guarantees.

Warmups are excluded from the reported samples. The benchmark is intentionally absent from CI, and there are no timing thresholds. Synthetic workloads provide repeatable probes, not a substitute for representative application data. Before changing or replacing an algorithm, capture representative workloads, input dimensions, environment details, and before/after timings. Do not infer a performance improvement from a single synthetic case.
