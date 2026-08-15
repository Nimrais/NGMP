# ETTh2 h720 — consensus-x precision hierarchy

Reference (paper Dynamic): NLL 0.86993; uncoupled direct port best: NLL 1.365 (lower is better)

| setup | L | carrier | τy | status | nll | ll | rmse | mae | cov95 | pinball | E[1/τy] | Eβ | min vx | iters | Δ | l2σ | secs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| linear_all | 2 | beta | scalar | ok | 1.2724±0.0516 | -1.2724±0.0516 | 0.8381±0.0315 | 0.7104 | 0.9353±0.0187 | 0.1446 | 0.461±0.000 | 0.36 | 0.015 | 60 | 0.0024 | 0.0661 | 5 |
| linear_all | 2 | beta | scalar | ok | 1.2565±0.0508 | -1.2565±0.0508 | 0.8283±0.0313 | 0.7005 | 0.9414±0.0179 | 0.1424 | 0.447±0.000 | 0.842 | 0.021 | 60 | 0.0031 | 0.0606 | 5 |
| linear_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | nobeta | scalar | ok | 0.8985±0.0369 | -0.8985±0.0369 | 0.5930±0.0250 | 0.4979 | 0.9910±0.0072 | 0.0997 | 0.385±0.000 | 0 | 0.0052 | 60 | 0.0027 | 0.0841 | 7 |
| linear_all | 2 | nobeta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | nobeta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| rff_all | 2 | beta | scalar | ok | 1.2515±0.0510 | -1.2515±0.0510 | 0.8250±0.0315 | 0.6962 | 0.9398±0.0181 | 0.1424 | 0.453±0.000 | 0.384 | 0.017 | 60 | 0.0033 | 0.207 | 56 |
| rff_all | 2 | beta | scalar | ok | 1.2407±0.0497 | -1.2407±0.0497 | 0.8197±0.0313 | 0.6915 | 0.9459±0.0172 | 0.1405 | 0.437±0.000 | 1.02 | 0.026 | 60 | 0.0042 | 0.207 | 56 |
| rff_all | 2 | beta | scalar | ok | 1.2527±0.0504 | -1.2527±0.0504 | 0.8270±0.0314 | 0.6985 | 0.9429±0.0176 | 0.1423 | 0.453±0.000 | 0.4 | 0.019 | 60 | 0.0048 | 0.108 | 57 |
| rff_all | 2 | beta | scalar | ok | 1.2391±0.0485 | -1.2391±0.0485 | 0.8208±0.0313 | 0.6927 | 0.9504±0.0165 | 0.1399 | 0.432±0.000 | 1.15 | 0.032 | 60 | 0.006 | 0.107 | 55 |
| rff_all | 2 | beta | scalar | ok | 1.2524±0.0502 | -1.2524±0.0502 | 0.8271±0.0314 | 0.6988 | 0.9459±0.0172 | 0.1421 | 0.452±0.000 | 0.407 | 0.019 | 60 | 0.0044 | 0.0292 | 56 |
| rff_all | 2 | beta | scalar | ok | 1.2390±0.0478 | -1.2390±0.0478 | 0.8216±0.0312 | 0.6937 | 0.9564±0.0155 | 0.1396 | 0.428±0.000 | 1.22 | 0.031 | 60 | 0.0058 | 0.0308 | 58 |
| rff_all | 2 | nobeta | scalar | ok | 0.9030±0.0384 | -0.9030±0.0384 | 0.5958±0.0256 | 0.4974 | 0.9880±0.0083 | 0.1012 | 0.383±0.000 | 0 | 0.0045 | 60 | 0.0046 | 0.28 | 56 |
| rff_all | 2 | nobeta | scalar | ok | 0.9073±0.0388 | -0.9073±0.0388 | 0.5982±0.0258 | 0.4987 | 0.9880±0.0083 | 0.1019 | 0.382±0.000 | 0 | 0.0044 | 60 | 0.0055 | 0.138 | 55 |
| rff_all | 2 | nobeta | scalar | ok | 0.9107±0.0391 | -0.9107±0.0391 | 0.6000±0.0260 | 0.4998 | 0.9880±0.0083 | 0.1024 | 0.382±0.000 | 0 | 0.0043 | 60 | 0.0055 | 0.0218 | 59 |
