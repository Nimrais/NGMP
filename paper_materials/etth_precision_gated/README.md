# ETTh precision-gated ensemble study

This directory is the canonical artifact root for the ETTh1/ETTh2 calibration
benchmark in standardized OT units. The benchmark compares frozen-expert neural
gates trained by Adam or IVON, the historical 2026 Dynamic PGE result, and one
pre-existing affine NGMP precision-gated configuration.

The historical projective-VMP and affine NGMP rows are not a controlled
inference-only comparison: their model configurations differ. The former is
included for continuity with the precision-gated model family of
Lukashchuk et al. (2026).

## Reported NGMP result

The paper reports the existing affine independent-expert `linear2_ngmp` arm
from `results/etth_ct_table`. It uses one fixed model and hyperparameter setting
across both datasets and all four horizons, fitted on each complete original
validation partition. The test results were inspected during earlier
exploratory work, so the paper labels them retrospective rather than claiming
a historically unseen test set. The validation-selection sweep remains in this
directory as an archived sensitivity study but is not used in the paper table.

## Reproduction

Rebuild the canonical summary, tables, and figure from the pre-existing
artifacts:

```sh
julia --startup-file=no --project=. \
  scripts/assemble_etth_precision_study.jl
```

## Canonical outputs

- `config.toml`: frozen protocol;
- `selection.csv` and `selected_configurations.csv`: archived validation-sweep
  sensitivity results, not used by the reported benchmark;
- `summary.csv`: all six methods and all eight test cells, including RMSE, MSE, NLL,
  coverage, interval width, descriptive CIs, counts, configuration IDs,
  provenance, and target/expert/VAE hashes;
- `ngmp_vmp_comparison.csv`: paired affine-NGMP-minus-projective-VMP RMSE and
  NLL differences with approximate normal 95% intervals on identical origins;
- `main_table.tex` and `appendix_table.tex`: a compact IVON/VMP/NGMP main table
  and full RMSE/NLL results, including Adam gates, in the appendix; both use
  approximate normal 95% intervals;
- `etth1_h192_ngmp_vmp.{pdf,png}` and `figure_metadata.csv`: the direct NGMP--VMP
  comparison over the fixed first 168 ETTh1 test predictions at horizon 192, where
  NGMP has lower full-cell RMSE and NLL;
- `etth1_h192_ngmp_vmp_caption.tex`: normal LaTeX caption generated from the
  full-cell metrics and paired differences.

The assembler normalizes legacy metric semantics by negating the PGE/Adam
artifact field called `nll`; IVON already stores positive NLL. One legacy Adam
cell overflowed its Float32 exponential during its original metric pass, so the
assembler evaluates the saved finite Float32 logits with a Float64 exponential.
Every interval
uses the actual test count: 3446, 3426, 3398, and 3321 for horizons 96, 192,
336, and 720.
