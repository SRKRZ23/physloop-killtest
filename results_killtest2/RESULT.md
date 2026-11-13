# Validation tests for ideas 11, 12, 13 (DROID-100 wrist camera, gripper-state task)

## Idea 11: does a cheap score choose a better half of the data?

Predict-the-mean baseline MAE: 0.3752. Every model gets the same 640 gradient steps.

| training subset | episodes | frames | seed | test gripper MAE |
|---|---|---|---|---|
| all episodes | 68 | 10554 | 0 | 0.0738 |
| all episodes | 68 | 10554 | 1 | 0.0728 |
| all episodes | 68 | 10554 | 2 | 0.0694 |
| least-idle half (score) | 34 | 6022 | 0 | 0.0875 |
| least-idle half (score) | 34 | 6022 | 1 | 0.0897 |
| least-idle half (score) | 34 | 6022 | 2 | 0.0885 |
| most-idle half (anti-score) | 34 | 4532 | 0 | 0.0828 |
| most-idle half (anti-score) | 34 | 4532 | 1 | 0.0890 |
| most-idle half (anti-score) | 34 | 4532 | 2 | 0.0884 |
| successful only (reward=1) | 60 | 9868 | 0 | 0.0733 |
| successful only (reward=1) | 60 | 9868 | 1 | 0.0735 |
| successful only (reward=1) | 60 | 9868 | 2 | 0.0725 |
| random half | 34 | 4544 | 0 | 0.0953 |
| random half | 34 | 5278 | 1 | 0.0974 |
| random half | 34 | 5372 | 2 | 0.0942 |
| random half | 34 | 5621 | 3 | 0.0837 |
| random half | 34 | 5853 | 4 | 0.0955 |
| random half | 34 | 5048 | 5 | 0.0940 |

Random-half spread: min 0.0837, max 0.0974. A score only counts if it lands outside this range.

## Ideas 12 and 13: rate versus task error by coding choice

Raw video is 20736 kbit/s; model MAE on raw frames 0.0738; predict-the-mean 0.3752.

**x264 native**: 18.2 kbps -> 0.2226, 20.8 kbps -> 0.2047, 26.6 kbps -> 0.1706, 35.5 kbps -> 0.1367, 49.3 kbps -> 0.1103, 72.6 kbps -> 0.0946, 114.1 kbps -> 0.0852, 189.5 kbps -> 0.0792, 338.2 kbps -> 0.0773

**x264 half-res**: 7.9 kbps -> 0.2621, 10.4 kbps -> 0.2158, 14.3 kbps -> 0.1725, 20.5 kbps -> 0.1402, 31.7 kbps -> 0.1172, 52.6 kbps -> 0.1023, 89.6 kbps -> 0.0914, 153.4 kbps -> 0.0862, 263.0 kbps -> 0.0836

**x264 quarter-res**: 8.0 kbps -> 0.2374, 11.4 kbps -> 0.1991, 17.3 kbps -> 0.1677, 27.3 kbps -> 0.1474, 43.3 kbps -> 0.1333, 67.7 kbps -> 0.1244, 104.2 kbps -> 0.1187, 152.9 kbps -> 0.1161

**JPEG per frame**: 217.6 kbps -> 0.2566, 236.1 kbps -> 0.2151, 275.3 kbps -> 0.1757, 334.5 kbps -> 0.1357, 416.8 kbps -> 0.1131, 549.7 kbps -> 0.0940, 658.9 kbps -> 0.0868, 940.0 kbps -> 0.0797, 2188.1 kbps -> 0.0745

**x264 + strong ROI gripper cell**: 21.0 kbps -> 0.2065, 31.2 kbps -> 0.1907, 71.1 kbps -> 0.1805, 116.0 kbps -> 0.1542

**x264 + strong ROI bottom row**: 19.3 kbps -> 0.1930, 26.2 kbps -> 0.1382, 61.1 kbps -> 0.0955, 239.3 kbps -> 0.0867

### Gripper MAE at equal bitrate (blank = that coding choice cannot reach this bitrate)

| kbit/s | x264 native | x264 half-res | x264 quarter-res | JPEG per frame | x264 + strong ROI gripper cell | x264 + strong ROI bottom row |
|---|---|---|---|---|---|---|
| 8 |  | 0.2603 |  |  |  |  |
| 12 |  | 0.1965 | 0.1950 |  |  |  |
| 16 |  | 0.1623 | 0.1734 |  |  |  |
| 20 | 0.2098 | 0.1425 | 0.1612 |  |  | 0.1867 |
| 30 | 0.1565 | 0.1201 | 0.1446 |  | 0.1922 | 0.1314 |
| 50 | 0.1097 | 0.1038 | 0.1305 |  | 0.1849 | 0.1056 |
| 100 | 0.0880 | 0.0903 | 0.1193 |  | 0.1622 | 0.0923 |
| 200 | 0.0790 | 0.0849 |  |  |  | 0.0878 |
| 400 |  |  |  | 0.1173 |  |  |

### Difference versus native x264 at the same bitrate, 95% bootstrap CI (negative = better)

| kbit/s | x264 half-res | x264 quarter-res | JPEG per frame | x264 + strong ROI gripper cell | x264 + strong ROI bottom row |
|---|---|---|---|---|---|
| 8 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 16 |  |  |  |  |  |
| 20 | -0.0673 [-0.0820, -0.0531] | -0.0486 [-0.0602, -0.0371] |  |  |  |
| 30 | -0.0364 [-0.0456, -0.0274] | -0.0120 [-0.0188, -0.0041] |  | +0.0357 [+0.0277, +0.0450] | -0.0251 [-0.0347, -0.0157] |
| 50 | -0.0059 [-0.0101, -0.0021] | +0.0207 [+0.0146, +0.0273] |  | +0.0751 [+0.0632, +0.0881] | -0.0041 [-0.0133, +0.0042] |
| 100 | +0.0023 [-0.0004, +0.0058] | +0.0313 [+0.0229, +0.0392] |  | +0.0742 [+0.0580, +0.0914] | +0.0043 [-0.0020, +0.0112] |
| 200 | +0.0059 [+0.0032, +0.0088] |  |  |  | +0.0088 [+0.0041, +0.0146] |
| 400 |  |  |  |  |  |
