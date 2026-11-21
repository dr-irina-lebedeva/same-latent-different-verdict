# E4/E10 features and round-trip control (add-on)

Feature gate: {'n': 3700, 'median_abs_diff_sd': 6.908401041254171e-17, 'spearman': 1.0, 'passed': True}

Flag rate on the 300 untouched evaluation photos after each treatment, at the paper's thresholds (1,000 independent calibration photos).

| detector | FPR | untouched | shift4 | downup2 | noise2 | VAE round-trip (orig) |
|---|---|---|---|---|---|---|
| aeroblade | 1% | 0.003 | 0.003 | 0.110 | 0.000 | 0.967 |
| aeroblade | 5% | 0.027 | 0.027 | 0.577 | 0.010 | 1.000 |
| npr | 1% | 0.007 | 0.007 | 0.843 | 0.723 | 0.897 |
| npr | 5% | 0.033 | 0.053 | 0.937 | 0.980 | 0.987 |
| univfd | 1% | 0.017 | 0.020 | 0.027 | 0.027 | 0.250 |
| univfd | 5% | 0.033 | 0.053 | 0.083 | 0.077 | 0.417 |
| probe_clip | 1% | 0.003 | 0.007 | 0.193 | 0.010 | 0.473 |
| probe_clip | 5% | 0.023 | 0.027 | 0.757 | 0.113 | 0.827 |
| probe_dino | 1% | 0.003 | 0.003 | 0.020 | 0.000 | 0.010 |
| probe_dino | 5% | 0.023 | 0.020 | 0.057 | 0.013 | 0.033 |
| commfor | 1% | 0.007 | 0.000 | 0.113 | 0.010 | 0.997 |
| commfor | 5% | 0.037 | 0.033 | 0.407 | 0.067 | 1.000 |
| bfree | 1% | 0.013 | 0.010 | 0.007 | 0.020 | 0.997 |
| bfree | 5% | 0.037 | 0.073 | 0.050 | 0.043 | 1.000 |