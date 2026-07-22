| config                     | window   | count_err | matched | coverage | purity | staffFP | dwellMAE |
|----------------------------|----------|-----------|---------|----------|--------|---------|----------|
| baseline (main, as-is)     | sparse   |        +0 |   0/1   |    18.5% |  1.000 |       0 |     0.00 |
| baseline (main, as-is)     | sliceb   |        +0 |   1/2   |    67.6% |  1.000 |       0 |    35.67 |
| baseline (main, as-is)     | crowded  |        +2 |   2/4   |    95.1% |  0.979 |       0 |    60.39 |
| baseline + ROI fix         | sparse   |        +0 |   1/1   |   100.0% |  1.000 |       0 |     0.03 |
| baseline + ROI fix         | sliceb   |        +0 |   1/2   |    67.9% |  1.000 |       0 |    34.54 |
| baseline + ROI fix         | crowded  |        +3 |   3/4   |    98.4% |  0.991 |       0 |    12.01 |
| v2 (RF-DETR+BoT-SORT)      | sparse   |        +0 |   1/1   |   100.0% |  1.000 |       0 |     0.03 |
| v2 (RF-DETR+BoT-SORT)      | sliceb   |        +1 |   2/2   |    99.8% |  1.000 |       0 |     0.28 |
| v2 (RF-DETR+BoT-SORT)      | crowded  |        +3 |   4/4   |   100.0% |  0.975 |       0 |    10.01 |
