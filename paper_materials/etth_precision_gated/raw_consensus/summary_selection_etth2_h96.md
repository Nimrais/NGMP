# ETTh2 h96 — consensus-x precision hierarchy

Reference (paper Dynamic): NLL 0.93418; uncoupled direct port best: NLL 1.589 (lower is better)

| setup | L | carrier | τy | status | nll | ll | rmse | mae | cov95 | pinball | E[1/τy] | Eβ | min vx | iters | Δ | l2σ | secs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| linear_all | 2 | beta | scalar | ok | 0.7917±0.0645 | -0.7917±0.0645 | 0.5138±0.0230 | 0.4166 | 0.8884±0.0235 | 0.0930 | 0.160±0.000 | 0.412 | 0.0098 | 60 | 0.0039 | 0.0433 | 5 |
| linear_all | 2 | beta | scalar | ok | 0.7824±0.0624 | -0.7824±0.0624 | 0.5124±0.0229 | 0.4155 | 0.8971±0.0227 | 0.0923 | 0.154±0.000 | 1.17 | 0.015 | 60 | 0.0048 | 0.0404 | 5 |
| linear_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | nobeta | scalar | ok | 0.7584±0.0678 | -0.7584±0.0678 | 0.4961±0.0231 | 0.3988 | 0.8841±0.0239 | 0.0900 | 0.160±0.000 | 0 | 0.0039 | 60 | 0.0018 | 0.0658 | 7 |
| linear_all | 2 | nobeta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | nobeta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| rff_all | 2 | beta | scalar | ok | 0.7940±0.0639 | -0.7940±0.0639 | 0.5155±0.0231 | 0.4185 | 0.8942±0.0229 | 0.0932 | 0.157±0.000 | 0.434 | 0.011 | 60 | 0.0047 | 0.174 | 58 |
| rff_all | 2 | beta | scalar | ok | 0.7854±0.0605 | -0.7854±0.0605 | 0.5159±0.0230 | 0.4193 | 0.9043±0.0219 | 0.0925 | 0.146±0.000 | 1.39 | 0.019 | 60 | 0.0054 | 0.17 | 57 |
| rff_all | 2 | beta | scalar | ok | 0.8025±0.0634 | -0.8025±0.0634 | 0.5186±0.0232 | 0.4214 | 0.8971±0.0227 | 0.0943 | 0.154±0.000 | 0.448 | 0.013 | 60 | 0.006 | 0.154 | 57 |
| rff_all | 2 | beta | scalar | ok | 0.7945±0.0588 | -0.7945±0.0588 | 0.5200±0.0231 | 0.4232 | 0.9116±0.0212 | 0.0938 | 0.140±0.000 | 1.55 | 0.022 | 60 | 0.0068 | 0.152 | 56 |
| rff_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| rff_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| rff_all | 2 | nobeta | scalar | ok | 0.7553±0.0671 | -0.7553±0.0671 | 0.4942±0.0228 | 0.3984 | 0.8913±0.0232 | 0.0896 | 0.158±0.000 | 0 | 0.0038 | 60 | 0.0044 | 0.228 | 56 |
| rff_all | 2 | nobeta | scalar | ok | 0.7635±0.0672 | -0.7635±0.0672 | 0.4942±0.0226 | 0.3992 | 0.8942±0.0229 | 0.0912 | 0.156±0.000 | 0 | 0.0037 | 60 | 0.0059 | 0.208 | 57 |
| rff_all | 2 | nobeta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
