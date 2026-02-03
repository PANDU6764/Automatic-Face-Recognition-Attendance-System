from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Paths
MODEL_PATH = ROOT / "models" / "facenet_keras.h5"
DB_PATH = ROOT / "data" / "attendance.db"
ENROLLMENT_DIR = ROOT / "data" / "enrollment"
SNAPSHOTS_DIR = ROOT / "data" / "snapshots"

# MTCNN
MIN_CONFIDENCE = 0.95

# Face matching (Euclidean / L2) - tune for your dataset
L2_THRESHOLD = 0.65

# Camera
CAMERA_INDEX = 0
FRAME_W = 640
FRAME_H = 480
