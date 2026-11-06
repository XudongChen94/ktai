import os
import shutil
import mediapipe as mp
import cv2

FaceDetection = mp.solutions.face_detection.FaceDetection
PERSON = "dustin"
SPLIT = "test"

input_folder = f"dataset/{SPLIT}/{PERSON}"
output_folder = f"frames/faces/{PERSON}/{SPLIT}"

margin = 0.2

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def crop_faces(image, detector):
    image_height, image_width = image.shape[:2]

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detection_result = detector.process(rgb_image)

    cropped_faces = []

    if not detection_result.detections:
        return cropped_faces

    for detection in detection_result.detections:
        bbox = detection.location_data.relative_bounding_box

        face_x = bbox.xmin * image_width
        face_y = bbox.ymin * image_height
        face_width = bbox.width * image_width
        face_height = bbox.height * image_height

        margin_x = margin * face_width
        margin_y = margin * face_height

        left = max(0, int(face_x - margin_x))
        top = max(0, int(face_y - margin_y))
        right = min(
            image_width,
            int(face_x + face_width + margin_x)
        )
        bottom = min(
            image_height,
            int(face_y + face_height + margin_y)
        )

        crop_face = image[top:bottom, left:right]

        if crop_face.size != 0:
            cropped_faces.append(crop_face)

    return cropped_faces


def main():
    os.makedirs(output_folder, exist_ok=True)

    list_dir = []

    for file_name in sorted(os.listdir(input_folder)):
        if file_name.lower().endswith(IMAGE_EXTENSIONS):
            list_dir.append(
                os.path.join(input_folder, file_name)
            )

    with FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ) as detector:

        for img in list_dir:
            image = cv2.imread(img)
            image_name = os.path.splitext(
                os.path.basename(img)
            )[0]

            if image is None:
                print("Could not read: " + image_name)
                continue

            cropped_faces = crop_faces(image, detector)

            if not cropped_faces:
                if min(image.shape[:2]) < 200:  # onder welke grootte neem je aan dat het al een crop is?
                    cropped_faces = [image]  # gebruik het hele plaatje
                    print("Already a crop, used as is: " + image_name)
                else:
                    print("Could not find a face in: " + image_name)
                    continue

            for index, crop_face in enumerate(cropped_faces):
                cv2.imwrite(
                    os.path.join(
                        output_folder,
                        f"{image_name}_face{index}.jpg"
                    ),
                    crop_face
                )


if __name__ == "__main__":
    main()