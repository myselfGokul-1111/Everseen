"""Step 2: click store sections. Click corners, ENTER to name zone, q to save."""
import cv2, json, numpy as np
img = cv2.imread("out/bg.jpg"); zones, cur = {}, []
def click(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN: cur.append([x, y])
cv2.namedWindow("zones"); cv2.setMouseCallback("zones", click)
print("Click polygon corners. ENTER = finish zone (then type name in terminal). q = save & quit")
while True:
    v = img.copy()
    for n, p in zones.items():
        cv2.polylines(v, [np.array(p)], True, (0, 255, 0), 2)
        cv2.putText(v, n, tuple(p[0]), 0, 0.8, (0, 255, 0), 2)
    for p in cur: cv2.circle(v, tuple(p), 4, (0, 0, 255), -1)
    if len(cur) > 1: cv2.polylines(v, [np.array(cur)], False, (0, 0, 255), 2)
    cv2.imshow("zones", v); k = cv2.waitKey(20) & 255
    if k == 13 and len(cur) >= 3:
        zones[input("Zone name: ").strip()] = cur[:]; cur.clear()
    if k == ord("q"): break
json.dump(zones, open("zones.json", "w"), indent=1); print("saved zones.json")
