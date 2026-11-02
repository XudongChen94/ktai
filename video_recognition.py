import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
import pandas as pd
import crop_faces
import model_utils

import get_embeddings_facenet
from facenet_pytorch import InceptionResnetV1

model = InceptionResnetV1(pretrained="vggface2").eval()
VIDEO_PATH = "video/compilation_final_uniform.mp4"
SAMPLES_PER_SECOND = 2
THRESHOLD = 0.9
results = []

video = cv2.VideoCapture(VIDEO_PATH)
frames_per_second = video.get(cv2.CAP_PROP_FPS) # Get frames per second from vid
# calculate the steps between frames to capture
frame_step = round(frames_per_second / SAMPLES_PER_SECOND)

frame_number = 0
# Crop faces from the video
detector = crop_faces.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Load embeddings from training data
references = model_utils.load_all_embeddings()

while True:
    success, frame = video.read()
    if not success:
        break

    if frame_number % frame_step == 0:
        time_in_seconds = frame_number / frames_per_second
        faces = crop_faces.crop_faces(frame, detector)
        print(round(time_in_seconds, 1), "seconden:", len(faces),
              "gezicht(en)")

        for face_index, face in enumerate(faces):
            embedding = get_embeddings_facenet.get_embedding(face, model)
            closest_per_person = {}
            for person in model_utils.PEOPLE:
                distances = np.linalg.norm(references[person] - embedding, axis=1)
                closest_per_person[person] = distances.min()

            best_person = min(closest_per_person,key=closest_per_person.get)
            best_distance = closest_per_person[best_person]
            if best_distance > THRESHOLD:
                best_person = "unknown"
            results.append(
                [frame_number, round(time_in_seconds, 2), face_index,
                 best_person, round(best_distance, 3)])
    frame_number += 1

video.release()
detector.close()


table = pd.DataFrame(results, columns=["frame", "time", "face", "person", "distance"])
table.to_csv("results/facenet/video_results.csv", index=False)
print(table["person"].value_counts())