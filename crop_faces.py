import glob
import os
import shutil
from xml.etree.ElementTree import tostring

import mediapipe as mp
import cv2

model_path = 'models/blaze_face_full_range.tflite'

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetection = mp.solutions.face_detection.FaceDetection
VisionRunningMode = mp.tasks.vision.RunningMode

input_folder = "dataset/train/mike"
output_folder = "frames/faces/mike/train"
margin = 0.2

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

def main():
        list_dir = []
        for file_name in sorted(os.listdir(input_folder)):
            if file_name.lower().endswith(IMAGE_EXTENSIONS):
                list_dir.append(os.path.join(input_folder, file_name))

        with FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        ) as detector:
            for img in list_dir:
                image = cv2.imread(img)
                image_name = os.path.splitext(os.path.basename(img))[0]
                image_height, image_width = image.shape[:2]

                if image_width <= 128:
                    shutil.copyfile(img, os.path.join(output_folder, image_name+"_face0.jpg"))
                    continue

                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                detection_result = detector.process(rgb_image)

                if not detection_result.detections:
                    print("Could not find a face in: " + image_name)
                    continue

                for index, detection in enumerate(detection_result.detections):
                    bbox = detection.location_data.relative_bounding_box

                    face_x = bbox.xmin * image_width
                    face_y = bbox.ymin * image_height
                    face_width = bbox.width * image_width
                    face_height = bbox.height * image_height

                    margin_x = margin * face_width
                    margin_y = margin * face_height

                    left = max(0, int(face_x - margin_x))
                    top = max(0, int(face_y - margin_y))
                    right = min(image_width, int(face_x + face_width + margin_x))
                    bottom = min(image_height, int(face_y + face_height + margin_y))

                    crop_face = image[top:bottom, left:right]
                    if crop_face.size == 0:
                        print(
                            f"Lege uitsnede overgeslagen: {image_name}_face{index}")

                    cv2.imwrite(
                        os.path.join(output_folder, f"{image_name}_face{index}.jpg"),
                        crop_face
                    )

if __name__ == "__main__": main()