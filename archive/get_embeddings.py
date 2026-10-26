import os
import cv2
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
ImageEmbedder = mp.tasks.vision.ImageEmbedder
ImageEmbedderOptions = mp.tasks.vision.ImageEmbedderOptions
VisionRunningMode = mp.tasks.vision.RunningMode

detector_model_path = '../models/blaze_face_short_range.tflite'
embedder_model_path = '../models/mobilenet_v3_small.tflite'
TRAIN_DIR = '../dataset/train'
EMBEDDING_DIR = '../embeddings'


def get_embeddings(char_name, images, detector, embedder):
    embeddings = []

    for file in images:
        img = cv2.imread(file)
        if img is None:
            print(f"Kan bestand {file} niet lezen")
            continue
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        detection_result = detector.detect(mp_image)
        if len(detection_result.detections) == 0:
            print(f"Geen gezicht gevonden bij bestand {file}")
            continue

        # Gezicht uitknippen
        bbox = detection_result.detections[0].bounding_box
        h, w, _ = image.shape
        x1, y1 = max(bbox.origin_x, 0), max(bbox.origin_y, 0)
        x2, y2 = min(x1 + bbox.width, w), min(y1 + bbox.height, h)
        face_crop = np.ascontiguousarray(image[y1:y2, x1:x2])

        face_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_crop)
        embedding_result = embedder.embed(face_mp_image)
        embeddings.append(np.array(embedding_result.embeddings[0].embedding))

    embeddings = np.array(embeddings)
    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    np.save(os.path.join(EMBEDDING_DIR, f"{char_name}.npy"), embeddings)
    print(f"{char_name}: {embeddings.shape} opgeslagen")


def main():
    detector_options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=detector_model_path),
        running_mode=VisionRunningMode.IMAGE)
    embedder_options = ImageEmbedderOptions(
        base_options=BaseOptions(model_asset_path=embedder_model_path),
        l2_normalize=True,
        quantize=False,
        running_mode=VisionRunningMode.IMAGE)

    with FaceDetector.create_from_options(detector_options) as detector, \
         ImageEmbedder.create_from_options(embedder_options) as embedder:

        # Elke submap in dataset/train is een personage
        for char_name in os.listdir(TRAIN_DIR):
            char_dir = os.path.join(TRAIN_DIR, char_name)
            if not os.path.isdir(char_dir):
                continue

            IMAGE_FILES = []
            for x in os.listdir(char_dir):
                if x.lower().endswith(('.jpg', '.jpeg', '.png')):
                    IMAGE_FILES.append(os.path.join(char_dir, x))

            get_embeddings(char_name, IMAGE_FILES, detector, embedder)
            # get_embeddings('dustin', IMAGE_FILES, detector, embedder)
        # embeddings = np.load('embeddings/dustin.npy')
        # print(embeddings.shape)   
        # print(embeddings[0])      # embedding van de eerste foto

if __name__ == "__main__":
    main()