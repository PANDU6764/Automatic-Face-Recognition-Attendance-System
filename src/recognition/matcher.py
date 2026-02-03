import numpy as np

def l2_distance(a, b):
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    return float(np.linalg.norm(a - b))

def match_face(embedding, persons, threshold):
    # FaceNet uses a Euclidean embedding space where distance reflects similarity. [web:111]
    best = (None, None, 1e9)
    for pid, name, emb in persons:
        d = l2_distance(embedding, emb)
        if d < best[2]:
            best = (pid, name, d)

    pid, name, dist = best
    if pid is None or dist > threshold:
        return None, None, dist
    return pid, name, dist
