| scheme | bits/weight | mean return | Δreturn vs fp32 (95% CI) | failure rate (Wilson 95%) | open-loop RMS | T_div median | A |
|---|---|---|---|---|---|---|---|
| `fp32` | 32.00 | 31.65 | +0.000% [+0.000%, +0.000%] | 0.2% [0.0%, 1.1%] | 0.00e+00 | 1000 / 1000 | n/a (no measurable loss) |
| `fp16` | 16.00 | 31.64 | -0.033% [-0.066%, -0.000%] | 0.2% [0.0%, 1.1%] | 5.59e-05 | 5 / 1000 | 5.8 |
| `int8-tensor` | 8.00 | 31.49 | -0.483% [-0.648%, -0.198%] | 0.0% [0.0%, 0.8%] | 1.18e-02 | 1 / 1000 | 0.4 |
| `int8-channel` | 8.00 | 31.44 | -0.649% [-1.062%, -0.274%] | 0.2% [0.0%, 1.1%] | 1.16e-02 | 1 / 1000 | 0.6 |
| `int4-channel` | 4.00 | **collapsed: NaN actions** (25 of the normalization scales quantized to zero) | — | — | — | — | — |
| `int4-group32` | 4.00 | **collapsed: NaN actions** (25 of the normalization scales quantized to zero) | — | — | — | — | — |
| `ternary` | 1.58 | **collapsed: NaN actions** (34 of the normalization scales quantized to zero) | — | — | — | — | — |
| `mixed-head-fp16` | 4.19 | **collapsed: NaN actions** (25 of the normalization scales quantized to zero) | — | — | — | — | — |
| `int8-act` | 8.00 | 31.45 | -0.623% [-0.797%, -0.335%] | 0.2% [0.0%, 1.1%] | 1.35e-02 | 1 / 1000 | 0.5 |
