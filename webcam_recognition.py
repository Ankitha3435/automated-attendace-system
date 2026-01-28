import cv2
import pickle
import numpy as np
from deepface import DeepFace

with open("embeddings.pkl", "rb") as f:
    db = pickle.load(f)

MODEL = "Facenet512"
DETECTOR = "opencv"
THRESHOLD = 0.7

cap = cv2.VideoCapture(0)

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        rep = DeepFace.represent(
            img_path=frame,
            model_name=MODEL,
            detector_backend=DETECTOR,
            enforce_detection=False
        )

        emb = rep[0]["embedding"]

        best_match = None
        best_score = -1

        for item in db:
            score = cosine(emb, item["embedding"])
            if score > best_score:
                best_score = score
                best_match = item["name"]

        if best_score > THRESHOLD:
            name = best_match
        else:
            name = "Unknown"

        cv2.putText(frame, name, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    except Exception:
        pass

    cv2.imshow("Face Recognition (Cached)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
