# Reproducible Q3 comparison reports

`experiments/research_v1_report.py` reads explicitly named, completed evaluation directories. It does not run a simulator or read sealed scenario definitions. The original evaluation harness and frozen protocol remain unchanged.

```text
python experiments/research_v1_report.py --baseline <rollout-directory> --candidate state=<state-directory> --candidate rl=<rl-directory> --candidate geometric=<geometric-directory> --output <report-directory> --figures
```

The input must contain the complete seed partition declared by its manifest, matching protocol hash, unique case identities, consistent success flags and the declared failure penalty. Each paired contrast checks identical scenario hashes. Source/configuration/checkpoint identities and input row-file hashes are retained in `comparison.json`. Means, P95, paired savings and intervals all retain failed cases with their penalty; the component figure adds the penalty separately. No failed or slow case is discarded.

The report writes machine-readable JSON, Markdown, and optionally standalone PNG/SVG figures. A positive paired saving means the candidate is faster. Each curve sorts its own case differences, so equal ranks across curves need not refer to the same scene. Per-contrast intervals do not provide simultaneous coverage for a winner selected from several candidates. Recorded runtime can include concurrent training load; a separate serial benchmark is needed to claim computational speedup.

A development or single-partition report never establishes final acceptance. After all selected code, configurations and weights have been frozen, run the two final partitions through the original guarded evaluation harness, generate their separate reports, and then use:

```text
python experiments/research_v1_report.py --final-random-report <random/comparison.json> --final-stress-report <stress/comparison.json> --output <final-summary-directory>
```

The final summary requires complete random and stress suites, the same frozen method identity in both, and a complete successful reference in both. Candidate failures or failed clearances prevent acceptance. Random-scene mean savings and their confidence interval use the protocol thresholds. Stress scenes are a separate robustness check, not IID draws to pool into the random confidence interval; the P95 degradation guard is checked on each partition. This is the conservative interpretation fixed before either final partition was opened.

The first-version practical target is evidence on the declared research distribution and stress families. It does not establish an optimal online policy or transfer to the official simulator. Official practice remains a separate, presently deferred verification.

Independent tests use only synthetic miniature protocols and records, with actual simulator generation disabled. They cover incomplete records, identity/protocol changes, failure penalties, configurable acceptance thresholds, both final suites and the final-summary CLI.
