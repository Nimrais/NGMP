# ETTh1 h720 — consensus-x precision hierarchy

Reference (paper Dynamic): NLL 0.37633; uncoupled direct port best: NLL NaN (lower is better)

| setup | L | carrier | τy | status | nll | ll | rmse | mae | cov95 | pinball | E[1/τy] | Eβ | min vx | iters | Δ | l2σ | secs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| linear_all | 2 | beta | scalar | ok | 0.8553±0.0413 | -0.8553±0.0413 | 0.5658±0.0271 | 0.4653 | 0.9729±0.0123 | 0.1002 | 0.320±0.000 | 0.252 | 0.021 | 60 | 0.0011 | 0.158 | 5 |
| linear_all | 2 | beta | scalar | ok | 0.7542±0.0398 | -0.7542±0.0398 | 0.5041±0.0270 | 0.3959 | 0.9759±0.0116 | 0.0932 | 0.279±0.000 | 0.404 | 0.026 | 60 | 0.0015 | 0.15 | 5 |
| linear_all | 2 | beta | scalar | ok | 0.8503±0.0413 | -0.8503±0.0413 | 0.5628±0.0272 | 0.4618 | 0.9729±0.0123 | 0.0999 | 0.316±0.000 | 0.258 | 0.022 | 60 | 0.0019 | 0.102 | 5 |
| linear_all | 2 | beta | scalar | ok | 0.7519±0.0396 | -0.7519±0.0396 | 0.5024±0.0269 | 0.3940 | 0.9759±0.0116 | 0.0931 | 0.276±0.000 | 0.423 | 0.026 | 60 | 0.0022 | 0.0952 | 5 |
| linear_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| linear_all | 2 | nobeta | scalar | ok | 0.5891±0.0421 | -0.5891±0.0421 | 0.4207±0.0260 | 0.3050 | 0.9729±0.0123 | 0.0818 | 0.235±0.000 | 0 | 0.0097 | 60 | 0.0014 | 0.209 | 8 |
| linear_all | 2 | nobeta | scalar | ok | 0.5752±0.0418 | -0.5752±0.0418 | 0.4143±0.0257 | 0.3002 | 0.9729±0.0123 | 0.0807 | 0.230±0.000 | 0 | 0.0099 | 60 | 0.0026 | 0.144 | 5 |
| linear_all | 2 | nobeta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| rff_all | 2 | beta | scalar | ok | 0.7577±0.0412 | -0.7577±0.0412 | 0.5081±0.0270 | 0.4001 | 0.9759±0.0116 | 0.0935 | 0.277±0.000 | 0.261 | 0.019 | 60 | 0.0025 | 0.101 | 55 |
| rff_all | 2 | beta | scalar | ok | 0.7195±0.0387 | -0.7195±0.0387 | 0.4822±0.0268 | 0.3715 | 0.9820±0.0101 | 0.0908 | 0.258±0.000 | 0.459 | 0.026 | 60 | 0.0034 | 0.1 | 55 |
| rff_all | 2 | beta | scalar | ok | 0.7652±0.0409 | -0.7652±0.0409 | 0.5117±0.0271 | 0.4037 | 0.9759±0.0116 | 0.0941 | 0.277±0.000 | 0.274 | 0.019 | 60 | 0.0038 | 0.124 | 55 |
| rff_all | 2 | beta | scalar | ok | 0.7114±0.0380 | -0.7114±0.0380 | 0.4760±0.0268 | 0.3647 | 0.9835±0.0097 | 0.0903 | 0.248±0.000 | 0.515 | 0.028 | 60 | 0.005 | 0.126 | 55 |
| rff_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| rff_all | 2 | beta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
| rff_all | 2 | nobeta | scalar | ok | 0.4850±0.0340 | -0.4850±0.0340 | 0.3702±0.0214 | 0.2860 | 0.9865±0.0088 | 0.0717 | 0.216±0.000 | 0 | 0.0046 | 60 | 0.0046 | 0.124 | 58 |
| rff_all | 2 | nobeta | scalar | ok | 0.4831±0.0324 | -0.4831±0.0324 | 0.3683±0.0208 | 0.2869 | 0.9880±0.0083 | 0.0709 | 0.218±0.000 | 0 | 0.0039 | 60 | 0.005 | 0.182 | 55 |
| rff_all | 2 | nobeta | scalar | unstable | non-finite posterior in hierarchy sweep |||||||||||||
