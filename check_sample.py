from fileinput import filename

import pandas as pd
import os
import cv2
import crop_faces

# Kies een sample size om te controleren hoeveel goed/fout is
SAMPLE_SIZE = 30
table = pd.read_csv("results/facenet/video_results.csv")
samples = []
VIDEO_PATH = "video/compilation_final_uniform.mp4"
OUTPUT_FOLDER = "results/facenet/check"

# groepeer de data per persoon uit de kolom perons en loop erdoor. [dustin, mike, elf, unkown]
for person_name, person_group in table.groupby(table["person"]):

    # number of samples, nooit groter dan de lengte van de groep
    n = min(SAMPLE_SIZE, len(person_group))
    # pak random een n aantal samples en stop die in de list
    samples.append(person_group.sample(n=n, random_state=0))

samples_table = pd.concat(samples)
print(samples_table["person"].value_counts())

# Extraheer de gezichten uit de video
frames_needed = set(samples_table["frame"].astype(int))


video = cv2.VideoCapture(VIDEO_PATH)
detector = crop_faces.FaceDetection(model_selection=1, min_detection_confidence=0.5)

frame_number = 0
while True:
    success, frame = video.read()
    if not success:
        break

    if frame_number in frames_needed:
        faces = crop_faces.crop_faces(frame, detector)
        rows_of_this_frame = samples_table[samples_table["frame"] == frame_number]

        for _, row in rows_of_this_frame.iterrows():
            if int(row["face"]) >= len(faces):
                print("Could not find face:", row["frame"], row["face"])
                continue

            face = faces[int(row["face"])]

            folder = os.path.join(OUTPUT_FOLDER, row["person"])
            os.makedirs(folder, exist_ok=True)
            file_name = f"{row["time"]}_s_face{int(row["face"])}_{row["distance"]}.jpg"
            cv2.imwrite(os.path.join(folder, file_name), face)

    frame_number += 1

video.release()
detector.close()

