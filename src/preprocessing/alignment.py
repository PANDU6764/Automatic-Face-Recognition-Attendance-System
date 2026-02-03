import cv2
import numpy as np

def crop_with_offset(frame_bgr, box, pad=20):
    x, y, w, h = box
    H, W = frame_bgr.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    return frame_bgr[y1:y2, x1:x2].copy(), (x1, y1)

def align_by_eyes(face_bgr, left_eye_xy, right_eye_xy, output_size=(160, 160)):
    lx, ly = left_eye_xy
    rx, ry = right_eye_xy

    dx, dy = (rx - lx), (ry - ly)
    angle = np.degrees(np.arctan2(dy, dx))

    desired_left_x = 0.35
    desired_left_y = 0.35
    desired_dist = (1.0 - 2.0 * desired_left_x) * output_size[0]

    dist = np.sqrt(dx * dx + dy * dy) + 1e-6
    scale = desired_dist / dist

    eyes_center = ((lx + rx) / 2.0, (ly + ry) / 2.0)
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    tx = output_size[0] * 0.5
    ty = output_size[1] * desired_left_y
    M[0, 2] += (tx - eyes_center[0])
    M[1, 2] += (ty - eyes_center[1])

    return cv2.warpAffine(face_bgr, M, output_size, flags=cv2.INTER_CUBIC)
