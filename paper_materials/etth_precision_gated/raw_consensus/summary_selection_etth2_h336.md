# ETTh2 h336 — consensus-x precision hierarchy

Reference (paper Dynamic): NLL 0.96119; uncoupled direct port best: NLL 1.098 (lower is better)

| setup | L | carrier | τy | status | nll | ll | rmse | mae | cov95 | pinball | E[1/τy] | Eβ | min vx | iters | Δ | l2σ | secs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| linear_all | 2 | beta | scalar | ok | 1.2398±0.0667 | -1.2398±0.0667 | 0.7423±0.0266 | 0.6313 | 0.8456±0.0272 | 0.1355 | 0.260±0.000 | 0.27 | 0.01 | 60 | 0.0024 | 0.0603 | 5 |
| linear_all | 2 | beta | scalar | ok | 1.2350±0.0665 | -1.2350±0.0665 | 0.7412±0.0267 | 0.6296 | 0.8426±0.0274 | 0.1352 | 0.253±0.000 | 0.551 | 0.014 | 60 | 0.0029 | 0.0581 | 5 |
| linear_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | nobeta | scalar | ok | 0.8025±0.0473 | -0.8025±0.0473 | 0.5368±0.0218 | 0.4655 | 0.9441±0.0173 | 0.0882 | 0.232±0.000 | 0 | 0.0038 | 60 | 0.0021 | 0.111 | 7 |
| linear_all | 2 | nobeta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | nobeta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| rff_all | 2 | beta | scalar | ok | 1.2200±0.0660 | -1.2200±0.0660 | 0.7352±0.0268 | 0.6243 | 0.8559±0.0264 | 0.1336 | 0.254±0.000 | 0.29 | 0.012 | 60 | 0.003 | 0.192 | 59 |
| rff_all | 2 | beta | scalar | ok | 1.2126±0.0649 | -1.2126±0.0649 | 0.7354±0.0269 | 0.6239 | 0.8574±0.0263 | 0.1331 | 0.245±0.000 | 0.661 | 0.018 | 60 | 0.0039 | 0.191 | 58 |
| rff_all | 2 | beta | scalar | ok | 1.2132±0.0654 | -1.2132±0.0654 | 0.7343±0.0268 | 0.6234 | 0.8588±0.0262 | 0.1330 | 0.252±0.000 | 0.303 | 0.013 | 60 | 0.0046 | 0.155 | 62 |
| rff_all | 2 | beta | scalar | ok | 1.2042±0.0636 | -1.2042±0.0636 | 0.7358±0.0270 | 0.6241 | 0.8662±0.0256 | 0.1323 | 0.240±0.000 | 0.741 | 0.021 | 60 | 0.0058 | 0.153 | 56 |
| rff_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| rff_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| rff_all | 2 | nobeta | scalar | ok | 0.8010±0.0476 | -0.8010±0.0476 | 0.5358±0.0218 | 0.4634 | 0.9456±0.0170 | 0.0881 | 0.229±0.000 | 0 | 0.0041 | 60 | 0.0042 | 0.26 | 61 |
| rff_all | 2 | nobeta | scalar | ok | 0.8051±0.0493 | -0.8051±0.0493 | 0.5375±0.0225 | 0.4635 | 0.9426±0.0175 | 0.0885 | 0.227±0.000 | 0 | 0.004 | 60 | 0.0051 | 0.196 | 57 |
| rff_all | 2 | nobeta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
