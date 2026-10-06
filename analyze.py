"""Step 3: dwell, journeys, heatmap, bottleneck/dead-zone flags."""
import json, itertools, cv2, numpy as np, pandas as pd
from collections import Counter
from matplotlib.path import Path

DWELL_MIN = 5.0   # seconds in a zone to count as "browsing"
O = "out/"
meta = json.load(open(O + "meta.json")); fps, W, H = meta["fps"], meta["W"], meta["H"]
zones = json.load(open("zones.json"))
df = pd.read_csv(O + "tracks.csv")
df = df[df.groupby("track_id").frame.transform("count") >= fps].copy()   # drop <1s tracks (noise)
df = df.sort_values(["track_id", "frame"])

# zone assignment
df["zone"] = ""
for name, pts in zones.items():
    m = Path(np.array(pts)).contains_points(df[["x", "y"]].values) & (df.zone == "")
    df.loc[m, "zone"] = name

# speed in screen-widths/sec (relative units, no calibration)
g = df.groupby("track_id")
df["speed"] = np.hypot(g.x.diff(), g.y.diff()) / g.t.diff() / W
df["speed"] = df.groupby("track_id").speed.transform(lambda s: s.rolling(int(fps), min_periods=1, center=True).median())

z = df[df.zone != ""]
td = z.groupby(["track_id", "zone"]).agg(dwell=("frame", "count"), speed=("speed", "median")).reset_index()
td["dwell"] /= fps
occ = z.groupby(["frame", "zone"]).track_id.nunique().reset_index(name="n")

rows = []
for name in zones:
    t, o = td[td.zone == name], occ[occ.zone == name].n
    rows.append(dict(zone=name, visitors=len(t),
        avg_dwell_s=round(t.dwell.mean(), 1) if len(t) else 0,
        browsers=int((t.dwell >= DWELL_MIN).sum()),
        avg_occupancy=round(o.mean(), 2) if len(o) else 0,
        p90_occupancy=float(o.quantile(.9)) if len(o) else 0,
        peak_occupancy=int(o.max()) if len(o) else 0,
        median_speed=round(t.speed.median(), 3) if len(t) else 0))
m = pd.DataFrame(rows); m["browse_rate"] = (m.browsers / m.visitors.replace(0, np.nan)).fillna(0).round(2)
m.to_csv(O + "zone_metrics.csv", index=False)

# flags
flags, gs = [], df.speed.median()
for r in m.itertuples():
    if r.visitors and r.p90_occupancy >= max(3, m.p90_occupancy.median()) and r.median_speed < 0.6 * gs:
        flags.append(dict(zone=r.zone, type="BOTTLENECK", reason=
            f"{r.zone}: 90% of the time >= {r.p90_occupancy:.0f} people present (peak {r.peak_occupancy}) "
            f"and median speed {r.median_speed:.3f} vs store median {gs:.3f} widths/s -> congestion"))
    if r.visitors < 0.3 * m.visitors.mean():
        flags.append(dict(zone=r.zone, type="DEAD ZONE", reason=
            f"{r.zone}: only {r.visitors} visitors vs zone average {m.visitors.mean():.0f} -> ignored area"))
    elif r.visitors >= m.visitors.mean() and r.browse_rate < 0.15:
        flags.append(dict(zone=r.zone, type="PASS-THROUGH", reason=
            f"{r.zone}: {r.visitors} visitors but only {r.browse_rate:.0%} stay >= {DWELL_MIN:.0f}s -> traffic without engagement"))
json.dump(flags, open(O + "flags.json", "w"), indent=1)

# journeys
paths = []
for tid, s in df.groupby("track_id"):
    seq = [k for k, v in itertools.groupby(s.zone) if k and len(list(v)) >= fps]
    seq = [k for i, k in enumerate(seq) if i == 0 or k != seq[i - 1]]
    if seq: paths.append(dict(track_id=tid, path=" > ".join(seq), duration_s=round(s.t.max() - s.t.min(), 1)))
pd.DataFrame(paths).to_csv(O + "journeys.csv", index=False)

# heatmap overlay
h = np.zeros((H, W), np.float32)
np.add.at(h, (df.y.clip(0, H - 1).astype(int), df.x.clip(0, W - 1).astype(int)), 1)
h = cv2.GaussianBlur(h, (0, 0), max(W, H) / 50)
h8 = (255 * h / max(h.max(), 1)).astype(np.uint8)
bg = cv2.imread(O + "bg.jpg"); blend = cv2.addWeighted(bg, .5, cv2.applyColorMap(h8, cv2.COLORMAP_JET), .5, 0)
out = np.where((h8 > 10)[..., None], blend, bg).astype(np.uint8)
for n, p in zones.items():
    cv2.polylines(out, [np.array(p)], True, (255, 255, 255), 2); cv2.putText(out, n, tuple(p[0]), 0, 0.8, (255, 255, 255), 2)
cv2.imwrite(O + "heatmap.png", out)
print(m.to_string(index=False)); print(*[f["type"] + ": " + f["reason"] for f in flags], sep="\n")
