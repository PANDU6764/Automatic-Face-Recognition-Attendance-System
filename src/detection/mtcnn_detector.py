from mtcnn import MTCNN

def load_mtcnn():
    return MTCNN()

def detect_faces(detector, frame_bgr, min_confidence=0.90):
    # MTCNN expects RGB and returns dicts with box/keypoints/confidence. [web:75]
    rgb = frame_bgr[:, :, ::-1]
    results = detector.detect_faces(rgb)

    out = []
    for r in results:
        conf = float(r.get("confidence", 0.0))
        if conf < min_confidence:
            continue

        x, y, w, h = r["box"]
        x, y = max(0, x), max(0, y)

        out.append({
            "box": (x, y, w, h),
            "keypoints": r.get("keypoints", {}),
            "confidence": conf
        })

    out.sort(key=lambda d: d["box"][2] * d["box"][3], reverse=True)
    return out
