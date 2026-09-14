# ICLR verification workbench

User scope: update GitHub research code, provide a working Gradio interface on server 212, and synchronize verified implementation/evidence into the existing Overleaf paper.

Server working tree has extensive uncommitted research. Develop from public commit d692223 and deploy a separate checkout; do not reset, clean, or overwrite the original server tree.

Deliverables: immutable three-state evaluation packets; strict versioned parsing; full-population and conditional metrics; matched direct/rule-guided/read-then-verify interventions with actual call accounting; audit download; Gradio live interface; CLI and regression tests; real GPU integration smoke test; paper appendix and reproducibility notes. Retain legacy search and historical scores.

Inspect the completed 128-source rule-explicit confirmation separately; only include its numbers after raw replay. It is a different protocol and not proof of general superiority. No invented human validation, closed-model results, or acceptance claims.
