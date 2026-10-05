# Paper and experiment handoff

This bundle makes the available paper and result artifacts fetchable from
`codex/paper-handoff-2026-10-04`. It was prepared on October 4, 2026 from
main commit `0bdb020c61745cadddd596072979597caee9e308`.

**The current paper and the July highway/warehouse raw results are included.
The original hospital and nonlinear Quad3D raw archives were not found in the
locations searched. Their documentation and packaged configuration provenance
are included, but are not substitutes for those raw archives.**

No simulations or tuning studies were run to assemble this bundle. All 35
source artifacts are copied unchanged. [MANIFEST.json](MANIFEST.json) records
their origins, sizes, and SHA-256 hashes. This README and the manifest are the
only newly authored files.

## Available files

| Material | Location | What it contains |
|---|---|---|
| Current paper | [2026_XXX__Policy_Library_CBF.pdf](paper/2026_XXX__Policy_Library_CBF.pdf) | The exact PDF provided for this revision, originally in the user's Downloads folder. |
| Additional-baseline outcomes | [Publication report](additional_baselines/projection_audited_baseline_only_2026-07-14/publication_report.md) | Raw CSV/JSON and reports for the two added baselines: 50 highway worlds and 100 warehouse worlds per method. |
| Latest archived additional-baseline timing | [Timing report](additional_baselines/single_trial_timing_2026-07-14/README.md) | Separate single-trial, single-worker measurements and raw data. These replace the earlier tentative timing cells, not the multi-trial outcome counts. |
| Historical highway summary | [50-trial report](historical_highway/benchmark_black_ice_results_50_tuned.md) | Existing local summary for the historical methods; not a new run or a full raw archive. |
| Historical highway speed protocol | [Speed-search summary](historical_highway/benchmark_black_ice_speed_search_final_summary.md) | Original 13-variant search settings, results, and special success/initialization rules. The two added baselines are absent. |
| Hospital documentation and results | [README snapshot](snapshots/hospital/README.md) | Final eight-method aggregate table and fixed-story publication protocol. |
| Hospital tuned configuration | [Packaged winner](snapshots/hospital/configs/plcbf_optuna_best.yaml) | Trial 42 provenance, 86 successes/12 collisions/2 timeouts on 100 worlds, and the original raw-archive path/hash. |
| Nonlinear Quad3D documentation | [README snapshot](snapshots/nl_quad3d/README.md) | Eight-method benchmark implementation and stress-scenario protocol. |
| Nonlinear Quad3D tuned configuration | [Packaged winner](snapshots/nl_quad3d/configs/plcbf_optuna_best.yaml) | Trial 44 provenance, PL-CBF tuning and held-out summaries, and the recorded original study location. |

The additional-baseline files were recovered byte-for-byte from commit
`de4f4d7bc583ae7b0a862271c8c135c2b32d913c`. Cleanup commit `7f3a63a` deleted
them from the current tree. The outcome archive's own manifest identifies
`4298440db247747105f587b73bbb9ed5c1e4a3f2` as its implementation revision.
Its 14 artifact hashes, paired scenario geometry, and per-trial failure
identity were verified during recovery.

These July archives concern highway and the **linear warehouse quadrotor**.
They are not the later nonlinear Quad3D benchmark results.

## Original archive locations to check on the development computer

The following paths are evidence copied from the tracked configuration
provenance. They are leads for the other computer, not verified accessible
locations on this Mac or Drive.

Hospital, relative to the original repository root:

```text
results/hospital_optuna_v11_100_final.db
results/hospital_optuna_v11_100_final_trials_4bb4ff8d0d879ae4/trial-0042.json
results/hospital_optuna_v11_100_final.json
results/hospital_optuna_v11_100_final.csv
results/hospital_optuna_v11_100_final.md
```

The exact selected trial archive is recorded as 3,612,471 bytes, 100 results,
SHA-256 `8b9375ae80a088e9960d2b585c1c07f3b2356c486926a2a6844a3f2313d088f2`.
The provenance above is for PL-CBF. The final hospital README also reports
seven baselines, but does not identify their raw archive filenames. Search
the original results directory for those reports and isolated timing runs.

Nonlinear Quad3D, absolute path recorded in the Optuna storage URI:

```text
/home/tk/Desktop/mycode/plcbf/results/new_benchmark/nl_quad3d_stress/optuna_corrected/study.db
```

The study is `nl_quad3d_plcbf_stress_v2_prefix_median`, with selected trial 44.
Search the enclosing `results/new_benchmark/nl_quad3d_stress/` directory for
the original validation and comparison reports. The recorded path identifies
a Linux checkout; this handoff does not establish its host or accessibility.

## What remains missing

- Hospital: the original per-world raw reports for all eight methods and raw
  isolated timing runs. The full aggregate table is already in the snapshot.
- Nonlinear Quad3D: original raw PL-CBF tuning/held-out reports and the complete
  seven-baseline comparison. The packaged PL-CBF provenance records 100/100
  successes on tuning seeds 1-100 and 100/100 on held-out seeds 101-200.
- Highway additions: maximum passing reference speed for both new methods.
  The existing speed driver needs adapters for these two algorithm keys.
- Comparable timing for the old paper tables if measured on a new computer.
  The July timings are explicitly tentative and are not directly comparable
  to the frozen paper timings.

If no matching raw archive can be recovered, run the missing cases and label
the output as a new run with its own commit, configuration, world, and machine
provenance. Do not present a rerun as the original selected-trial archive.
Use the packaged winners without starting new tuning by default.

For nonlinear Quad3D, a complete comparison on held-out seeds 101-200 is a
recommended collection target: all eight methods, `playground_stress`,
`balanced_six_axis_streams_v2`, 48 moving spheres, 800 plant steps. This is a
proposed reporting grid, not evidence that all eight methods were previously
run on it. The benchmark CLI defaults to **seed 0 only**.

For hospital, retain the established `hospital_fixed_refuge_v4` grid: five
canonical stories, seeds 0-19, 50 humans, fixed convoys, full policy library,
and 3000 plant steps (180 seconds). The hospital selected trial was tuned and
reported on the same 100 worlds; it is not a held-out result.

## Paper details to reconcile before appending rows

1. Table I/Fig. 2 describe highway friction as 0.25; the archived fixed-speed
   benchmark uses 0.30. The speed script separately uses 0.25 for an analytical
   braking estimate. Establish the source protocol before changing either
   table values or simulation settings.
2. Table II describes the warehouse library as `P+1`; the implementation uses
   `P` directions plus stop and nominal (`P+2`, hence 66 when `P=64`).
3. The paper calls maximum reference speed an offline worst-case-clearance
   analysis. The available search evaluates closed-loop simulations; preserve
   and explain its actual success conditions and tested speed grid.
4. The paper attributes all experiments to an M4 MacBook Air. Update that
   statement and timing provenance if additional runs use another computer.
5. Multi-Backup-CBF-MI uses sampled terminal proxies; describe it as the
   implemented benchmark adaptation without inheriting an unproved invariant-
   set guarantee. Library-PCBF-MI is the implemented per-policy QP selector
   comparison, not an unchanged reproduction of a separate prior method.

## Search scope and limitations

The audit covered the current repository and available Git history, local
projects under Desktop/mycode, Downloads, Documents, managed Codex worktrees,
mounted Google Drive/iCloud filename listings, and a targeted connected
Google Drive name search. No matching original hospital/nonlinear Quad3D raw
archive was found. The Drive connector's search covers supported file types;
this is not an exhaustive search of every cloud archive or remote machine.
An internal mounted Drive temporary directory was inaccessible and was not
bypassed. Related COMB-PLCBF hospital files in a separate local project use a
different experiment and are not included as evidence for this paper.

## Retrieve and verify

To inspect the handoff separately from ongoing work in another checkout:

```bash
git fetch origin codex/paper-handoff-2026-10-04
git worktree add --detach ../plcbf-paper-handoff origin/codex/paper-handoff-2026-10-04
```

From the root of that checkout, verify the copied source artifacts:

```bash
python3 - <<'PY'
from pathlib import Path
import hashlib
import json

root = Path('paper_handoff')
manifest = json.loads((root / 'MANIFEST.json').read_text())
for entry in manifest['files']:
    data = (root / entry['path']).read_bytes()
    assert len(data) == entry['bytes'], entry['path']
    assert hashlib.sha256(data).hexdigest() == entry['sha256'], entry['path']
print(f"Verified {len(manifest['files'])} original artifacts.")
PY
```

The original `results/` and `output/` paths are ignored by Git. This bundle
uses the tracked `paper_handoff/` directory so a normal branch fetch transfers
the available evidence.
