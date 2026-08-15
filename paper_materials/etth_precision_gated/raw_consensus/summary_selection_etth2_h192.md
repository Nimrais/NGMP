# ETTh2 h192 — consensus-x precision hierarchy

Reference (paper Dynamic): NLL 0.92366; uncoupled direct port best: NLL 1.169 (lower is better)

| setup | L | carrier | τy | status | nll | ll | rmse | mae | cov95 | pinball | E[1/τy] | Eβ | min vx | iters | Δ | l2σ | secs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| linear_all | 2 | beta | scalar | ok | 1.2349±0.0824 | -1.2349±0.0824 | 0.6810±0.0258 | 0.5720 | 0.7726±0.0314 | 0.1350 | 0.187±0.000 | 0.292 | 0.0095 | 60 | 0.0032 | 0.0674 | 5 |
| linear_all | 2 | beta | scalar | ok | 1.2182±0.0807 | -1.2182±0.0807 | 0.6785±0.0258 | 0.5697 | 0.7770±0.0312 | 0.1337 | 0.183±0.000 | 0.601 | 0.013 | 60 | 0.0039 | 0.0621 | 5 |
| linear_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | nobeta | scalar | ok | 1.1046±0.0736 | -1.1046±0.0736 | 0.6276±0.0231 | 0.5428 | 0.8499±0.0267 | 0.1147 | 0.185±0.000 | 0 | 0.0041 | 60 | 0.0018 | 0.114 | 6 |
| linear_all | 2 | nobeta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | nobeta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| rff_all | 2 | beta | scalar | ok | 1.2361±0.0823 | -1.2361±0.0823 | 0.6835±0.0260 | 0.5731 | 0.7682±0.0316 | 0.1358 | 0.184±0.000 | 0.309 | 0.011 | 60 | 0.0039 | 0.181 | 55 |
| rff_all | 2 | beta | scalar | ok | 1.2145±0.0792 | -1.2145±0.0792 | 0.6834±0.0260 | 0.5728 | 0.7770±0.0312 | 0.1345 | 0.176±0.000 | 0.713 | 0.017 | 60 | 0.0047 | 0.18 | 55 |
| rff_all | 2 | beta | scalar | ok | 1.2385±0.0819 | -1.2385±0.0819 | 0.6874±0.0262 | 0.5756 | 0.7668±0.0316 | 0.1367 | 0.182±0.000 | 0.323 | 0.012 | 60 | 0.0052 | 0.162 | 55 |
| rff_all | 2 | beta | scalar | ok | 1.2107±0.0775 | -1.2107±0.0775 | 0.6886±0.0263 | 0.5765 | 0.7828±0.0309 | 0.1350 | 0.172±0.000 | 0.796 | 0.02 | 60 | 0.0062 | 0.159 | 55 |
| rff_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| rff_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| rff_all | 2 | nobeta | scalar | ok | 1.0848±0.0722 | -1.0848±0.0722 | 0.6220±0.0230 | 0.5361 | 0.8528±0.0265 | 0.1134 | 0.184±0.000 | 0 | 0.0038 | 60 | 0.0038 | 0.239 | 57 |
| rff_all | 2 | nobeta | scalar | ok | 1.0717±0.0708 | -1.0717±0.0708 | 0.6187±0.0229 | 0.5319 | 0.8571±0.0262 | 0.1126 | 0.182±0.000 | 0 | 0.0035 | 60 | 0.0044 | 0.205 | 55 |
| rff_all | 2 | nobeta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
