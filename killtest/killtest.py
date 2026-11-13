"""PhysLoop kill-test: does ROI prioritisation beat a plain codec at matched bytes
on real robot video, measured by a downstream task model?

Data      : DROID-100 (TFDS name r2d2_faceblur), wrist_image_left, 320x180 @ 15 Hz
Split     : by source lab (no lab appears in both train and test)
Model     : ResNet18 trained on RAW train frames only, two heads
              T1 gripper_position            (visible in the wrist frame)
              T2 10-step cartesian xyz delta  (needs scene context, z-scored)
Conditions: uniform x264 | x264+ROI(center) | x264+ROI(frozen r3c2)
            PhysLoop pre-blur x2 + x264 | PhysLoop pre-blur x4 + x264
Rate      : CRF sweep -> (bytes, error) curve per condition; conditions are compared
            to uniform x264 at EQUAL BYTES by log-linear interpolation of the uniform
            curve, with an episode-level bootstrap CI.
"""
import os, sys, json, subprocess, tempfile, time, argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path, PurePosixPath
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from physloop import apply_physloop  # the actual product code

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--out", default=str(ROOT / "results_killtest"))
args = ap.parse_args()

OUT = Path(args.out + ("_smoke" if args.smoke else "")); OUT.mkdir(parents=True, exist_ok=True)
CAM, W, H, FPS, LAG, SEED = "wrist_image_left", 320, 180, 15, 10, 20261112
CRFS = [24, 28, 32, 36, 40, 44, 48] if not args.smoke else [32, 44]
UCRFS = sorted(set(CRFS + [20, 51]))    # uniform curve brackets the others
QOFF = -0.25  # fixed a priori, not tuned
ROI_CENTER = (80, 45, 160, 90)                     # central 50% box (PhysLoop default prior)
ROI_R3C2 = (107, 135, 53, 45)                      # frozen cell from LOCKED_PROTOCOL.md
CONDS = ["uniform", "roi_center", "roi_r3c2", "physloop_x2", "physloop_x4"]
rng = np.random.default_rng(SEED)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- data
import tensorflow as tf
import tensorflow_datasets as tfds
tf.config.set_visible_devices([], "GPU")

catalog = json.load(open(ROOT / "results_2026-11-12/r2d2_episode_catalog.json"))
source_of = {r["episode_id"]: PurePosixPath(r["source_path"]).parts[-5] for r in catalog}
sources = sorted(set(source_of.values()))
test_sources = set(sources[::3])                   # deterministic, chosen before seeing results
log("sources", len(sources), "test sources", sorted(test_sources))

ds = tfds.builder_from_directory(str(ROOT / "data/droid_100/1.0.0")).as_dataset(
    split="train", shuffle_files=False)

train, test = [], []
for eid, ep in enumerate(ds):
    is_test = source_of[eid] in test_sources
    if args.smoke and ((is_test and len(test) >= 3) or (not is_test and len(train) >= 6)):
        if len(test) >= 3 and len(train) >= 6:
            break
        continue
    fr, grip, cart = [], [], []
    for st in ep["steps"].as_numpy_iterator():
        o = st["observation"]
        fr.append(np.asarray(o[CAM], np.uint8))
        grip.append(float(np.ravel(o["gripper_position"])[0]))
        cart.append(np.asarray(o["cartesian_position"], np.float32)[:3])
    n = len(fr)
    if n <= LAG + 5:
        continue
    fr = np.stack(fr); grip = np.asarray(grip, np.float32); cart = np.stack(cart)
    assert fr.shape[1:] == (H, W, 3), fr.shape
    dxyz = np.zeros((n, 3), np.float32); dxyz[:-LAG] = cart[LAG:] - cart[:-LAG]
    valid = np.zeros(n, bool); valid[:-LAG] = True
    rec = dict(eid=eid, source=source_of[eid], frames=fr, grip=grip, dxyz=dxyz, valid=valid)
    if is_test:
        test.append(rec)
    else:                                          # keep every 2nd frame to bound RAM
        idx = np.where(valid)[0][::2]
        train.append(dict(frames=fr[idx], grip=grip[idx], dxyz=dxyz[idx]))
log("train episodes", len(train), "frames", sum(len(t["grip"]) for t in train),
    "| test episodes", len(test), "frames", sum(len(t["grip"]) for t in test))

# ---------------------------------------------------------------- model
import torch, torch.nn as nn, torchvision
torch.manual_seed(SEED)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
Xtr = np.concatenate([t["frames"] for t in train])
gtr = np.concatenate([t["grip"] for t in train])
dtr = np.concatenate([t["dxyz"] for t in train])
d_mu, d_sd = dtr.mean(0), dtr.std(0) + 1e-8
Ytr = np.concatenate([gtr[:, None], (dtr - d_mu) / d_sd], 1).astype(np.float32)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

try:
    net = torchvision.models.resnet18(weights="IMAGENET1K_V1"); pretrained = True
except Exception as e:                              # offline: train from scratch
    log("pretrained weights unavailable:", e); net = torchvision.models.resnet18(); pretrained = False
net.fc = nn.Linear(512, 4); net.to(dev)


def prep(x):
    return (torch.from_numpy(x).to(dev).permute(0, 3, 1, 2).float() / 255 - MEAN) / STD


opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
epochs = 1 if args.smoke else args.epochs
steps = epochs * (len(Xtr) // 64); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
net.train(); k = 0
for ep_i in range(epochs):
    perm = rng.permutation(len(Xtr)); tot = 0
    for b in range(len(Xtr) // 64):
        i = np.sort(perm[b * 64:(b + 1) * 64])
        loss = nn.functional.smooth_l1_loss(net(prep(Xtr[i])), torch.from_numpy(Ytr[i]).to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item(); k += 1
    log(f"epoch {ep_i + 1}/{epochs} loss {tot / max(1, len(Xtr) // 64):.4f}")
del Xtr
net.eval()


@torch.no_grad()
def predict(frames):
    return np.concatenate([net(prep(frames[i:i + 128])).cpu().numpy() for i in range(0, len(frames), 128)])


def errors(rec, frames):
    """per-episode MAE for T1 (gripper, raw units) and T2 (xyz delta, z-scored)."""
    p = predict(frames); v = rec["valid"]
    e1 = np.abs(p[v, 0] - rec["grip"][v]).mean()
    e2 = np.abs(p[v, 1:] - (rec["dxyz"][v] - d_mu) / d_sd).mean()
    return float(e1), float(e2)


# ---------------------------------------------------------------- codec
TMP = Path(tempfile.mkdtemp(prefix="physloop_kt_", dir=os.environ.get("TMPDIR", "/tmp")))


def encode_decode(frames, crf, roi=None):
    path = TMP / "x.mp4"
    vf = [] if roi is None else ["-vf", "addroi=x=%d:y=%d:w=%d:h=%d:qoffset=%s" % (*roi, QOFF)]
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", *vf, "-c:v", "libx264",
           "-preset", "medium", "-tune", "zerolatency", "-crf", str(crf),
           "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, input=frames.tobytes(), check=True)
    size = path.stat().st_size
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    dec = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    assert len(dec) == len(frames), (len(dec), len(frames))
    return dec, size


def variant(frames, cond, crf):
    if cond == "uniform":
        return encode_decode(frames, crf)
    if cond == "roi_center":
        return encode_decode(frames, crf, ROI_CENTER)
    if cond == "roi_r3c2":
        return encode_decode(frames, crf, ROI_R3C2)
    bf = 2 if cond == "physloop_x2" else 4
    pre = np.stack([apply_physloop(f, blur_factor=bf, center_frac=0.5) for f in frames])
    return encode_decode(pre, crf)


def psnr(a, b):
    m = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float(10 * np.log10(255 ** 2 / max(m, 1e-9)))


rows = []
for n, rec in enumerate(test, 1):
    f = rec["frames"]; e1, e2 = errors(rec, f)
    base = dict(eid=rec["eid"], source=rec["source"], frames=len(f))
    rows.append(dict(base, cond="raw", crf=-1, bytes=f.nbytes, e_grip=e1, e_dxyz=e2, psnr=99.0, psnr_center=99.0))
    mean_g = float(np.abs(gtr.mean() - rec["grip"][rec["valid"]]).mean())
    mean_d = float(np.abs((rec["dxyz"][rec["valid"]] - d_mu) / d_sd).mean())
    rows.append(dict(base, cond="mean_baseline", crf=-1, bytes=0, e_grip=mean_g, e_dxyz=mean_d, psnr=0, psnr_center=0))
    for cond in CONDS:
        for crf in (UCRFS if cond == "uniform" else CRFS):
            dec, size = variant(f, cond, crf)
            e1, e2 = errors(rec, dec)
            x, y, w, h = ROI_CENTER
            rows.append(dict(base, cond=cond, crf=crf, bytes=size, e_grip=e1, e_dxyz=e2,
                             psnr=psnr(dec, f), psnr_center=psnr(dec[:, y:y + h, x:x + w], f[:, y:y + h, x:x + w])))
    log(f"test episode {n}/{len(test)} done")

import pandas as pd
df = pd.DataFrame(rows); df.to_csv(OUT / "per_episode.csv", index=False)

# ---------------------------------------------------------------- analysis
eids = sorted(df.eid.unique())


def curves(sub):
    """frame-weighted mean error and kbps per (cond, crf)."""
    out = {}
    for (c, q), g in sub.groupby(["cond", "crf"]):
        fr = g.frames.sum()
        out[(c, q)] = (g.bytes.sum() * 8 / (fr / FPS) / 1000,
                       (g.e_grip * g.frames).sum() / fr, (g.e_dxyz * g.frames).sum() / fr)
    return out


def deltas(cv):
    """error(cond) - error(uniform interpolated to the same bitrate); negative = cond better."""
    u = sorted(cv[("uniform", q)] for q in UCRFS)
    lb = np.log([p[0] for p in u]); res = {}
    for c in CONDS[1:]:
        for q in CRFS:
            kb, a, b = cv[(c, q)]
            inside = lb[0] <= np.log(kb) <= lb[-1]
            res[(c, q)] = (kb, a - np.interp(np.log(kb), lb, [p[1] for p in u]),
                           b - np.interp(np.log(kb), lb, [p[2] for p in u]), inside)
    return res


cv = curves(df); point = deltas(cv)
B = 200 if args.smoke else 4000
boot = {k: [] for k in point}
by_ep = {e: df[df.eid == e] for e in eids}
for _ in range(B):
    samp = pd.concat([by_ep[e] for e in rng.choice(eids, len(eids))])
    for k_, v in deltas(curves(samp)).items():
        boot[k_].append(v[1:3])

lines = ["# PhysLoop kill-test result", "",
         f"Test episodes: {len(eids)} from held-out labs {sorted(test_sources)}; "
         f"frames: {int(df[df.cond == 'raw'].frames.sum())}; pretrained backbone: {pretrained}; "
         f"epochs: {epochs}; ROI qoffset {QOFF}; smoke={args.smoke}", "",
         "## Model validity (lower is better)", "",
         "| input | gripper MAE | xyz-delta MAE (z) |", "|---|---|---|"]
for c in ["mean_baseline", "raw"]:
    g = df[df.cond == c]; fr = g.frames.sum()
    lines.append(f"| {c} | {(g.e_grip * g.frames).sum() / fr:.4f} | {(g.e_dxyz * g.frames).sum() / fr:.4f} |")
lines += ["", "## Rate-error curves", "", "| condition | CRF | kbps | gripper MAE | xyz MAE | PSNR | PSNR center |", "|---|---|---|---|---|---|---|"]
for c in CONDS:
    for q in (UCRFS if c == "uniform" else CRFS):
        g = df[(df.cond == c) & (df.crf == q)]; kb, a, b = cv[(c, q)]
        lines.append(f"| {c} | {q} | {kb:.1f} | {a:.4f} | {b:.4f} | {g.psnr.mean():.2f} | {g.psnr_center.mean():.2f} |")
lines += ["", "## Versus uniform x264 at EQUAL bitrate (negative = better than uniform)", "",
          "| condition | CRF | kbps | d gripper MAE [95% CI] | d xyz MAE [95% CI] | in range |", "|---|---|---|---|---|---|"]
for (c, q), (kb, da, db, inside) in point.items():
    arr = np.asarray(boot[(c, q)]); lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
    lines.append(f"| {c} | {q} | {kb:.1f} | {da:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}] | "
                 f"{db:+.4f} [{lo[1]:+.4f}, {hi[1]:+.4f}] | {'yes' if inside else 'EXTRAPOLATED'} |")
(OUT / "RESULT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
for p in TMP.glob("*"):
    p.unlink()
TMP.rmdir()
log("saved", OUT / "RESULT.md")
