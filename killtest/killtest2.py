"""Validation tests for ideas 11, 12, 13 on DROID-100 wrist video (gripper-state task only).

Idea 11: does a cheap, label-free episode score pick a better half of the training data than chance?
Idea 12: below ~20 kbit/s, does a strong codec ROI on the gripper help?
Idea 13: at equal bitrate, how do native x264, half/quarter-resolution x264 and per-frame JPEG compare?
Same split and model family as killtest.py (lab-held-out, ResNet18, raw-frame training).
"""
import os, sys, json, subprocess, tempfile, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path, PurePosixPath
import numpy as np, cv2

ROOT = Path(__file__).resolve().parent.parent
SMOKE = "--smoke" in sys.argv
OUT = ROOT / ("results_killtest2_smoke" if SMOKE else "results_killtest2"); OUT.mkdir(exist_ok=True)
CAM, W, H, FPS, LAG, SEED = "wrist_image_left", 320, 180, 15, 10, 20261112
RAW_KBPS = W * H * 3 * 8 * FPS / 1000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


import tensorflow as tf, tensorflow_datasets as tfds
tf.config.set_visible_devices([], "GPU")
catalog = json.load(open(ROOT / "results_2026-11-12/r2d2_episode_catalog.json"))
source_of = {r["episode_id"]: PurePosixPath(r["source_path"]).parts[-5] for r in catalog}
sources = sorted(set(source_of.values())); test_sources = set(sources[::3])
ds = tfds.builder_from_directory(str(ROOT / "data/droid_100/1.0.0")).as_dataset(split="train", shuffle_files=False)

train, test = [], []
for eid, ep in enumerate(ds):
    is_test = source_of[eid] in test_sources
    if SMOKE and len(test if is_test else train) >= (3 if is_test else 8):
        if len(test) >= 3 and len(train) >= 8:
            break
        continue
    fr, grip, cart, rew = [], [], [], 0.0
    for st in ep["steps"].as_numpy_iterator():
        o = st["observation"]
        fr.append(np.asarray(o[CAM], np.uint8)); grip.append(float(np.ravel(o["gripper_position"])[0]))
        cart.append(np.asarray(o["cartesian_position"], np.float32)[:3]); rew = float(st["reward"])
    if len(fr) <= LAG + 5:
        continue
    fr = np.stack(fr); grip = np.asarray(grip, np.float32); cart = np.stack(cart)
    if is_test:
        if SMOKE or len(test) < 14:
            test.append(dict(eid=eid, frames=fr, grip=grip))
    else:
        move = np.linalg.norm(np.diff(cart, axis=0), axis=1) + np.abs(np.diff(grip))
        idle = float(np.mean(move < 1e-3))
        idx = np.arange(0, len(fr), 3)
        train.append(dict(eid=eid, frames=fr[idx].copy(), grip=grip[idx], idle=idle, reward=rew, n=len(fr)))
    del fr
# one shared training array; subsets are index arrays, so no per-subset copies
XTR = np.concatenate([t["frames"] for t in train]); YTR = np.concatenate([t["grip"] for t in train])
off = 0
for t in train:
    t["idx"] = np.arange(off, off + len(t["grip"])); off += len(t["grip"]); del t["frames"]
del ds
import gc; gc.collect()
log("train eps", len(train), "train frames", len(XTR), "test eps", len(test), "test frames", sum(len(t["grip"]) for t in test))

import torch, torch.nn as nn, torchvision
dev = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def prep(x):
    return (torch.from_numpy(np.ascontiguousarray(x)).to(dev).permute(0, 3, 1, 2).float() / 255 - MEAN) / STD


def fit(eps, seed, steps):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    sel = np.concatenate([t["idx"] for t in eps]); X, y = XTR, YTR
    net = torchvision.models.resnet18(weights="IMAGENET1K_V1"); net.fc = nn.Linear(512, 1); net.to(dev).train()
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    for s in range(steps):
        i = np.sort(rng.choice(sel, 64, replace=False))
        loss = nn.functional.smooth_l1_loss(net(prep(X[i])).squeeze(1), torch.from_numpy(y[i]).to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval(); return net, len(sel)


@torch.no_grad()
def ep_err(net, frames, grip):
    p = np.concatenate([net(prep(frames[i:i + 128])).squeeze(1).cpu().numpy() for i in range(0, len(frames), 128)])
    return float(np.abs(p - grip).mean())


def test_mae(net):
    n = sum(len(t["grip"]) for t in test)
    return sum(ep_err(net, t["frames"], t["grip"]) * len(t["grip"]) for t in test) / n


lines = ["# Validation tests for ideas 11, 12, 13 (DROID-100 wrist camera, gripper-state task)", ""]

# ---------------------------------------------------------------- idea 11
STEPS = 60 if SMOKE else 320          # same number of gradient steps for every subset
half = len(train) // 2
order_idle = sorted(train, key=lambda t: t["idle"])
subsets = {"all episodes": train, "least-idle half (score)": order_idle[:half], "most-idle half (anti-score)": order_idle[half:],
           "successful only (reward=1)": [t for t in train if t["reward"] > 0.5]}
r11 = []
full_net = None
for name, eps in subsets.items():
    for seed in [0]:
        net, nfr = fit(eps, 100 + seed, STEPS); m = test_mae(net); r11.append((name, len(eps), nfr, seed, m))
        log("idea11", name, "seed", seed, "MAE", round(m, 4))
        if name == "all episodes" and seed == 0:
            full_net = net
rng = np.random.default_rng(SEED)
for seed in range(1 if SMOKE else 3):
    eps = [train[i] for i in rng.choice(len(train), half, replace=False)]
    net, nfr = fit(eps, 200 + seed, STEPS); m = test_mae(net); r11.append(("random half", half, nfr, seed, m))
    log("idea11 random half seed", seed, "MAE", round(m, 4))
mean_g = float(YTR.mean())
base = sum(np.abs(mean_g - t["grip"]).sum() for t in test) / sum(len(t["grip"]) for t in test)
lines += ["## Idea 11: does a cheap score choose a better half of the data?", "",
          f"Predict-the-mean baseline MAE: {base:.4f}. Every model gets the same {STEPS} gradient steps.", "",
          "| training subset | episodes | frames | seed | test gripper MAE |", "|---|---|---|---|---|"]
lines += [f"| {a} | {b} | {c} | {d} | {e:.4f} |" for a, b, c, d, e in r11]
rand = [e for a, *_, e in r11 if a == "random half"]
lines += ["", f"Random-half spread: min {min(rand):.4f}, max {max(rand):.4f}. A score only counts if it lands outside this range.", ""]

# ---------------------------------------------------------------- ideas 12, 13
TMP = Path(tempfile.mkdtemp(prefix="kt2_", dir=os.environ.get("TMPDIR", "/tmp")))


def x264(frames, crf, w=W, h=H, roi=None, qoff=-0.75):
    src = frames if (w, h) == (W, H) else np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames])
    vf = [] if roi is None else ["-vf", "addroi=x=%d:y=%d:w=%d:h=%d:qoffset=%s" % (*roi, qoff)]
    p = TMP / "x.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
                    "-i", "-", *vf, "-c:v", "libx264", "-preset", "medium", "-tune", "zerolatency", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", str(p)], input=src.tobytes(), check=True)
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(p), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    dec = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    if (w, h) != (W, H):
        dec = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR) for f in dec])
    return dec, p.stat().st_size


def jpeg(frames, q):
    out, size = [], 0
    for f in frames:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        size += len(buf); out.append(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    return np.stack(out), size


R3C2, BOTTOM = (107, 135, 53, 45), (0, 134, 320, 46)
FAM = {"x264 native": [(lambda f, c=c: x264(f, c)) for c in ([32, 51] if SMOKE else [22, 30, 38, 44, 48, 51])],
       "x264 half-res": [(lambda f, c=c: x264(f, c, 160, 90)) for c in ([24, 44] if SMOKE else [18, 26, 34, 40, 46])],
       "x264 quarter-res": [(lambda f, c=c: x264(f, c, 80, 46)) for c in ([20, 36] if SMOKE else [12, 20, 28, 36])],
       "JPEG per frame": [(lambda f, q=q: jpeg(f, q)) for q in ([5, 50] if SMOKE else [3, 8, 20, 50, 90])],
       "x264 + strong ROI gripper cell": [(lambda f, c=c: x264(f, c, roi=R3C2)) for c in ([44, 51] if SMOKE else [40, 44, 48, 51])],
       "x264 + strong ROI bottom row": [(lambda f, c=c: x264(f, c, roi=BOTTOM)) for c in ([44, 51] if SMOKE else [40, 44, 48, 51])]}
rec = []                                   # (family, point, eid, frames, bytes, err)
for n, t in enumerate(test, 1):
    for fam, fns in FAM.items():
        for j, fn in enumerate(fns):
            dec, size = fn(t["frames"]); rec.append((fam, j, t["eid"], len(t["grip"]), size, ep_err(full_net, dec, t["grip"])))
    log(f"codec test episode {n}/{len(test)}")
import pandas as pd
df = pd.DataFrame(rec, columns=["fam", "pt", "eid", "frames", "bytes", "err"]); df.to_csv(OUT / "codec_points.csv", index=False)
raw_mae = test_mae(full_net)


def curve(sub, fam):
    g = sub[sub.fam == fam].groupby("pt").apply(lambda x: pd.Series(dict(
        kbps=x.bytes.sum() * 8 / (x.frames.sum() / FPS) / 1000, mae=(x.err * x.frames).sum() / x.frames.sum())), include_groups=False)
    return g.sort_values("kbps")


def at(sub, fam, k):
    c = curve(sub, fam)
    if not (c.kbps.min() <= k <= c.kbps.max()):
        return np.nan
    return float(np.interp(np.log(k), np.log(c.kbps.values), c.mae.values))


lines += ["## Ideas 12 and 13: rate versus task error by coding choice", "",
          f"Raw video is {RAW_KBPS:.0f} kbit/s; model MAE on raw frames {raw_mae:.4f}; predict-the-mean {base:.4f}.", ""]
for fam in FAM:
    c = curve(df, fam)
    lines += [f"**{fam}**: " + ", ".join(f"{k:.1f} kbps -> {m:.4f}" for k, m in zip(c.kbps, c.mae)), ""]
targets = [8, 12, 16, 20, 30, 50, 100, 200, 400]
eids = sorted(df.eid.unique()); by = {e: df[df.eid == e] for e in eids}
B = 100 if SMOKE else 500
boot = {(f, k): [] for f in FAM for k in targets}
for _ in range(B):
    sub = pd.concat([by[e] for e in rng.choice(eids, len(eids))])
    ref = {k: at(sub, "x264 native", k) for k in targets}
    for f in FAM:
        for k in targets:
            boot[(f, k)].append(at(sub, f, k) - ref[k])
lines += ["### Gripper MAE at equal bitrate (blank = that coding choice cannot reach this bitrate)", "",
          "| kbit/s | " + " | ".join(FAM) + " |", "|---|" + "---|" * len(FAM)]
for k in targets:
    lines.append(f"| {k} | " + " | ".join("" if np.isnan(at(df, f, k)) else f"{at(df, f, k):.4f}" for f in FAM) + " |")
lines += ["", "### Difference versus native x264 at the same bitrate, 95% bootstrap CI (negative = better)", "",
          "| kbit/s | " + " | ".join(f for f in FAM if f != "x264 native") + " |", "|---|" + "---|" * (len(FAM) - 1)]
for k in targets:
    cells = []
    for f in FAM:
        if f == "x264 native":
            continue
        v = np.array(boot[(f, k)]); v = v[~np.isnan(v)]
        d = at(df, f, k) - at(df, "x264 native", k)
        cells.append("" if np.isnan(d) or len(v) < B * 0.8 else f"{d:+.4f} [{np.percentile(v, 2.5):+.4f}, {np.percentile(v, 97.5):+.4f}]")
    lines.append(f"| {k} | " + " | ".join(cells) + " |")
(OUT / "RESULT.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
for p in TMP.glob("*"):
    p.unlink()
TMP.rmdir(); log("saved", OUT / "RESULT.md")
