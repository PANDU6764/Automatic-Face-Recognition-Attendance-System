from keras_facenet import FaceNet
import numpy as np
import cv2
import logging

logging.basicConfig(level=logging.INFO)


class Embedder:
    def __init__(self, model_path=None):
        """
        Initialize FaceNet embedder.
        keras-facenet automatically downloads weights on first run.
        """
        self.embedder = FaceNet()
        logging.info("FaceNet embedder initialized successfully")

    def preprocess_face(self, face_img):
        """
        Resize and convert face image for FaceNet input.
        """
        if isinstance(face_img, str):
            face_img = cv2.imread(face_img)

        face_img = cv2.resize(face_img, (160, 160))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        return face_img

    def get_embedding(self, face_img):
        """
        Generate L2-normalized FaceNet embedding.
        """
        face_img = self.preprocess_face(face_img)
        face_img = np.expand_dims(face_img, axis=0)

        embedding = self.embedder.embeddings(face_img)[0]

        # L2 normalization
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
        return embedding

    def embed_from_detection(self, frame_bgr, det):
        """
        Generate embedding from MTCNN detection result.
        """
        from src.preprocessing.alignment import align_by_eyes
        from src.preprocessing.clahe import clahe_on_l_channel

        # 1. Bounding box from MTCNN
        x, y, w, h = det["box"]

        # Safety checks
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)

        # 2. Crop face
        face = frame_bgr[y:y + h, x:x + w]
        if face.size == 0:
            raise ValueError("Empty face crop encountered")

        # 3. Extract eye keypoints (ABSOLUTE COORDS)
        kps = det["keypoints"]
        left_eye_xy = tuple(kps["left_eye"])
        right_eye_xy = tuple(kps["right_eye"])

        # Convert to FACE-LOCAL coordinates
        left_eye_xy = (left_eye_xy[0] - x, left_eye_xy[1] - y)
        right_eye_xy = (right_eye_xy[0] - x, right_eye_xy[1] - y)

        # 4. Align face using eye landmarks
        face = align_by_eyes(face, left_eye_xy, right_eye_xy)

        # 5. Illumination normalization (CLAHE)
        face = clahe_on_l_channel(face)

        # 6. Generate FaceNet embedding
        embedding = self.get_embedding(face)
        return embedding
