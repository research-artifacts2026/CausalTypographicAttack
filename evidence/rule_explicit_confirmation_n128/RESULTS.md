# Fixed-reference content confirmation study

Status: complete. 128-item reserved-source confirmation of the rule-explicit variant, with new nominal inputs; unseen only relative to the frozen source-use audit.

Actual victim calls: 1280 (256 fresh clean + 1024 attacked); no planner or reading calls.

| Model | Clean correct | Simple wrong target | Rule wrong target | Simple valid correct | Rule valid correct |
|---|---:|---:|---:|---:|---:|
| qwen7 | 82/128 | 58/82 | 73/82 | 95/128 | 118/128 |
| qwen3vl8 | 97/128 | 58/97 | 58/97 | 122/128 | 125/128 |

Wrong-result columns use the same clean-correct subset within each model. Valid-result columns use all registered items. All-scene outcomes, parse failures, family strata, and paired vectors remain in analysis.json.

- qwen7: rule minus simple = 18.29 percentage points; discordants 15 versus 0; exact p=6.10352e-05, Holm p=0.00012207; 95% paired interval=[0.0975609756097561, 0.2682926829268293] (fraction units).
- qwen3vl8: rule minus simple = 0.00 percentage points; discordants 9 versus 9; exact p=1, Holm p=1; 95% paired interval=[-0.08247422680412371, 0.08247422680412371] (fraction units).

Interaction estimates and their paired intervals are secondary, descriptive all-scene results. No interaction significance test is claimed.

## Limits

- Controlled composites with synthetic task inputs; photographs supply context only.
- The content contrast bundles rule-explicit wording and residual character/ink differences.
- Target choices do not establish a within-query neural reasoning mechanism.
- This evaluates the fixed-reference rule-explicit variant under a confirmation protocol, not the original full ContraLedger three-state protocol.
- Independent human validation, physical transfer, and SceneTAP superiority remain unestablished.
