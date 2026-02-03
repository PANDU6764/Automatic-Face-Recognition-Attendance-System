from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from src.detection.mtcnn_detector import load_mtcnn, detect_faces
from src.recognition.embedder import Embedder
from src.storage.sqlite_db import connect, init_db, upsert_person

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def iter_images(folder: Path):
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            yield p


def enroll_all(enrollment_dir: Path, model_path: Path, db_path: Path, min_confidence=0.60):
    enrollment_dir = Path(enrollment_dir)
    if not enrollment_dir.exists():
        raise RuntimeError(f"Enrollment dir not found: {enrollment_dir}")

    detector = load_mtcnn()
    embedder = Embedder(model_path)

    conn = connect(db_path)
    init_db(conn)

    images = list(iter_images(enrollment_dir))
    if not images:
        raise RuntimeError(f"No enrollment images found in {enrollment_dir}")

    for img_path in tqdm(images, desc="Bulk enrolling students"):
        roll = img_path.stem  # filename without extension (ROLL NUMBER)
        name = roll           # name = roll (you can map later)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Cannot read image {img_path.name}")
            continue

        dets = detect_faces(detector, img, min_confidence=min_confidence)
        if not dets:
            print(f"[WARN] No face detected in {img_path.name}")
            continue

        # Take the most confident face
        dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
        embedding = embedder.embed_from_detection(img, dets[0])

        # Normalize embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

        upsert_person(conn, roll, name, embedding)
        print(f"[OK] Enrolled {roll}")

    conn.close()
    print("🎉 Enrollment completed for all images")
