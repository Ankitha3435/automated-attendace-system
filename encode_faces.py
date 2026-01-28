import os
import pickle
from deepface import DeepFace

DATASET = "dataset"
MODEL = "Facenet512"
DETECTOR = "opencv"

embeddings = []

print("Encoding dataset...")

for person in os.listdir(DATASET):
    person_path = os.path.join(DATASET, person)

    for img in os.listdir(person_path):
        img_path = os.path.join(person_path, img)

        try:
            rep = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL,
                detector_backend=DETECTOR,
                enforce_detection=True
            )

            embeddings.append({
                "name": person,
                "embedding": rep[0]["embedding"]
            })

        except Exception:
            pass

with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("Encoding complete. Saved embeddings.pkl")
