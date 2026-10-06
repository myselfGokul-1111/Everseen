"""Step 1: detect + track people on GPU. Usage: python track.py store.mp4"""
import sys, json, cv2, pandas as pd
from ultralytics import YOLO

src = sys.argv[1]
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
W, H = int(cap.get(3)), int(cap.get(4))
ok, first = cap.read(); cap.release()
cv2.imwrite("out/bg.jpg", first)
json.dump({"fps": fps, "W": W, "H": H}, open("out/meta.json", "w"))

model = YOLO("yolov8s.pt")
writer = cv2.VideoWriter("out/annotated.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
rows = []
for i, r in enumerate(model.track(source=src, stream=True, classes=[0], conf=0.3,
                                  tracker="bytetrack.yaml", device=0, half=True, verbose=False)):
    writer.write(r.plot())
    if r.boxes.id is not None:
        for b, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist()):
            rows.append((i, tid, float((b[0] + b[2]) / 2), float(b[3]), i / fps))  # foot point
    if i % 300 == 0: print("frame", i)
writer.release()
pd.DataFrame(rows, columns=["frame", "track_id", "x", "y", "t"]).to_csv("out/tracks.csv", index=False)
print("done ->", len(rows), "detections")
