import cv2
import numpy as np

FACENET_SIZE = (160, 160)

def to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def prewhiten(x_rgb_uint8):
    x = x_rgb_uint8.astype("float32")
    mean, std = x.mean(), x.std()
    std_adj = max(std, 1.0 / np.sqrt(x.size))
    return (x - mean) / std_adj

def l2_normalize(v):
    v = np.asarray(v, dtype="float32")
    return v / (np.linalg.norm(v) + 1e-10)
