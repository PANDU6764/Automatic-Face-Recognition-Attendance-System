import cv2

def clahe_on_l_channel(face_bgr, clipLimit=2.0, tileGridSize=(8, 8)):
    # Using createCLAHE + apply is the standard OpenCV flow. [web:58]
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    l2 = clahe.apply(l)

    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
