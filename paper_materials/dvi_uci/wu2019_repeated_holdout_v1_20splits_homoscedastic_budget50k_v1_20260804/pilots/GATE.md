# Homoscedastic 50k/50k pilot gate

Date: 2026-08-04  
Device: NVIDIA GeForce RTX 3090  
Backend: Reactant/CUDA

The candidate protocol caps selection and refit at 50,000 optimizer steps,
checked at epoch boundaries. It retains the 14,000-step KL warmup and
1,000-step anneal. All six recorded pilot configurations completed without a
failure or numerical clamp.

## Power split-1 quality gate

| Run | Best epoch | Selection steps | Refit steps | Validation LPD | Test LPD | RMSE | Recorded time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full-budget dDVI reference | 2670 | 403260 | 232290 | -3.787527 | -3.793396 | 4.419283 | 2284.7 s |
| Budgeted dDVI | 540 | 50076 | 46980 | -3.787667 | -3.793528 | 4.428807 | 352.4 s |
| Budgeted DVI | 540 | 50076 | 46980 | -3.787667 | -3.793528 | 4.428859 | 419.2 s |

The budgeted dDVI validation LPD differs from the full-budget reference by
0.000140, passing the predeclared 0.01 tolerance.

## Four-process timing gate

Power splits 2--5 ran concurrently with four 22%-memory XLA clients. The gate
completed in 1,265 seconds. Individual recorded times were 946.9--1,066.2
seconds; all selections used 50,076 steps and refits used 31,755--50,025
steps.

Using the observed wall time as one four-configuration wave gives a
conservative projection of 10 h 32 min for one 120-configuration method and
21 h 05 min for both dDVI and DVI. Even excluding the approximately 199-second
shared startup overhead projects about 17 h 46 min for both methods.

**Decision: runtime gate failed.** The full 240-configuration campaign was not
started. The budget must not be reduced again without an explicit protocol
decision.

