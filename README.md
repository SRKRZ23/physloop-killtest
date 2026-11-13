# physloop-killtest

A fair test of "task-aware" pre-blur against plain codecs on real robot video. Result: no benefit.

Earlier PhysLoop figures published on 2026-11-09 (bandwidth savings, comparison with H.265, cost figures)
are withdrawn. They are not reproduced here and nothing in this repository supports them.

## Setup

- Data: DROID-100 (public, `gs://gresearch/robotics/droid_100`), wrist camera, 320x180, 15 fps.
- Split: 5 labs held out for test (31 episodes, 11,126 frames); the model never sees those labs in training.
- Model: ResNet18, pretrained backbone, trained on raw frames only. Task: gripper state per frame.
  A second task (xyz delta) failed the validity gate and is not used.
- Conditions at equal bitrate (CRF sweep): uniform x264, x264 + codec region of interest (centre; gripper cell),
  `physloop.py` pre-blur x2 and x4 before x264. Episode-level bootstrap 95% confidence intervals.

## Result (`results_killtest/RESULT.md`)

- The pre-blur is worse than plain x264 at every bitrate from 21 to 170 kbit/s; every interval excludes zero.
  It keeps the frame centre sharp, but on a wrist camera the gripper is at the bottom edge.
- Codec ROI on the gripper cell is no better than plain x264 (slightly worse at the lowest bitrates).
- Plain x264 is robust: 285x compression (20.7 Mbit/s to 73 kbit/s) raised gripper error by 12.5%.

Follow-up (`results_killtest2/RESULT.md`, same data and task, a separately trained model): per-frame JPEG needs
about 5-8x the bitrate of x264 for the same error; below about 50 kbit/s half-resolution x264 beats native.

## Limits

One dataset, one camera, low resolution, one small model, one easy task, one fixed ROI strength.
This says "no benefit here", not "impossible". It does not test fine-contact tasks, high resolution,
policy rollouts, or models trained on compressed video.

## Run

Needs Python with `tensorflow`, `tensorflow-datasets`, `torch`, `torchvision`, `opencv-python`, `pandas`, `numpy`,
ffmpeg with libx264, and the DROID-100 TFDS folder at `data/droid_100/1.0.0`.
`python killtest/killtest.py` and `python killtest/killtest2.py` (`--smoke` for a quick run).
Scripts were run on a Mac; expect some work to adapt paths.

Code: MIT. DROID data has its own licence (CC BY 4.0, see the DROID project).
