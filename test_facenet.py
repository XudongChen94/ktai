import glob
import os
import cv2
import numpy as np
import model_utils
import crop_faces
import get_embeddings_facenet
from facenet_pytorch import InceptionResnetV1

PEOPLE = ["dustin", "elf", "mike"]
TEST_DATASET_FOLDER = "dataset/test"
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")
SCALES = [0.50, 1.00, 1.50]
TRANSFORMATIONS = ["roll", "yaw", "pitch"]
ANGLES = {
    "roll": [-90, 0, 90],
    "yaw": [-40, 0, 40],
    "pitch": [-40, 0, 40],
}
THRESHOLD = 0.9


"""
Add different datasets for stresstesting the model
"""


def change_scale(image, scale):
    height, width = image.shape[:2]

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    scaled_image = cv2.resize(image,(new_width, new_height))
    result = np.zeros_like(image)

    source_x1 = max(0, (new_width - width) // 2)
    source_y1 = max(0, (new_height - height) // 2)

    source_x2 = min(new_width, source_x1 + width)
    source_y2 = min(new_height, source_y1 + height)

    used_width = source_x2 - source_x1
    used_height = source_y2 - source_y1

    target_x1 = max(0, (width - used_width) // 2)
    target_y1 = max(0, (height - used_height) // 2)

    target_x2 = target_x1 + used_width
    target_y2 = target_y1 + used_height

    result[target_y1:target_y2, target_x1:target_x2] = (
        scaled_image[source_y1:source_y2, source_x1:source_x2]
    )

    return result


def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(image, rotation_matrix, (width, height), 
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def change_perspective(image, angle, axis):
    if angle == 0:
        return image.copy()

    height, width = image.shape[:2]
    factor = np.sin(np.radians(abs(angle))) * 0.3
    source_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    if axis == "yaw":
        shrink = factor * height
        squeeze = factor * width

        if angle > 0:
            target_points = np.float32([[0, 0], [width - squeeze, shrink], 
                                        [width - squeeze, height - shrink], [0, height]])
        else:
            target_points = np.float32([[squeeze, shrink], [width, 0], [width, height],
                                        [squeeze, height - shrink]])

    elif axis == "pitch":
        shrink = factor * width
        squeeze = factor * height

        if angle > 0:
            target_points = np.float32([[shrink, squeeze], [width - shrink, squeeze],
                                        [width, height], [0, height]])
        else:
            target_points = np.float32([[0, 0], [width, 0], [width - shrink, height - squeeze],
                                        [shrink, height - squeeze]])

    else:
        raise ValueError(f"Unknown axis: {axis}")

    matrix = cv2.getPerspectiveTransform(source_points, target_points)
    return cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0))


def change_angle(image, angle_type, angle):
    if angle_type == "roll":
        return rotate_image(image, angle)

    return change_perspective(image, angle, angle_type)


def create_transformed_dataset(transformation, value):
    output_dataset_folder = (f"dataset/test_{transformation}_{value}")

    for person in PEOPLE:
        input_person_folder = os.path.join(TEST_DATASET_FOLDER, person)
        output_person_folder = os.path.join(output_dataset_folder, person)
        os.makedirs(output_person_folder, exist_ok=True)
        image_paths = []

        for extension in IMAGE_EXTENSIONS:
            image_paths.extend(glob.glob(os.path.join(input_person_folder, extension)))

        for image_path in sorted(image_paths):
            image = model_utils.create_input(image_path)
            if image is None:
                print(f"Could not read: {image_path}")
                continue

            if transformation == "scale":
                transformed_image = change_scale(image, value)

            elif transformation in TRANSFORMATIONS:
                transformed_image = change_angle(image, transformation, value)

            else:
                raise ValueError(f"Unknown transformation: {transformation}")

            output_path = os.path.join(output_person_folder,os.path.basename(image_path))
            cv2.imwrite(output_path, transformed_image)
    print(f"Created dataset: {output_dataset_folder}")


def create_all_test_datasets():
    # Scales
    for scale in SCALES:
        create_transformed_dataset("scale", scale)

    # Roll, yaw en pitch
    for angle_type, angles in ANGLES.items():
        for angle in angles:
            create_transformed_dataset(angle_type,angle)


def test_dataset(model, reference_embeddings, detector,
                 people_to_test, test_dataset_folder, result_file):
    
    results = []

    for true_label in people_to_test:
        person_folder = os.path.join(test_dataset_folder, true_label)
        image_paths = []

        for extension in IMAGE_EXTENSIONS:
            image_paths.extend(glob.glob(os.path.join(person_folder, extension)))

        print(f"Testing {true_label}: " f"{len(image_paths)} images")

        for image_path in sorted(image_paths):
            image_bgr = model_utils.create_input(image_path)
            if image_bgr is None:
                print(f"Could not read: {image_path}")
                continue

            faces = crop_faces.crop_faces(image_bgr,detector)
            if not faces:
                print(f"No face detected: {image_path}")
                continue

            for face_index, face in enumerate(faces, start=1):
                test_embedding = (get_embeddings_facenet.get_embedding(face, model))
                distances = {
                    person: model_utils.calculate_distance(
                        test_embedding,
                        reference_embeddings[person]
                    )for person in PEOPLE}

                prediction = model_utils.predict_identity(distances, THRESHOLD)

                results.append([os.path.basename(image_path), face_index,
                                true_label,THRESHOLD]
                    + [distances[person] for person in PEOPLE]
                    + [prediction]
                )

    model_utils.save_results(result_file,["image", "face", "true_label", "threshold"]
                             + PEOPLE + ["prediction"], results)

if __name__ == "__main__":
    create_all_test_datasets()
    detector = crop_faces.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    model = InceptionResnetV1(pretrained="vggface2").eval()
    reference_embeddings = model_utils.load_reference_embeddings()
    test_dataset(model, reference_embeddings, detector,
                PEOPLE, "dataset/test_roll_-90", "roll_-90_results.csv")
    
    test_dataset(model, reference_embeddings, detector,
                PEOPLE, "dataset/test_roll_0", "roll_0_results.csv")

    test_dataset(model, reference_embeddings, detector,
                PEOPLE, "dataset/test_roll_90", "roll_90_results.csv")
    
    test_dataset(model, reference_embeddings, detector,
                PEOPLE, "dataset/test_yaw_-40", "yaw_-40_results.csv")

    test_dataset(model, reference_embeddings, detector,
                PEOPLE, "dataset/test_yaw_0", "yaw_0_results.csv")

    test_dataset(model, reference_embeddings, detector,
                PEOPLE, "dataset/test_yaw_40", "yaw_40_results.csv")

    test_dataset(model, reference_embeddings, detector,
                PEOPLE, "dataset/test_pitch_-40", "pitch_-40_results.csv")

    test_dataset(model, reference_embeddings, detector,
                PEOPLE, "dataset/test_pitch_0", "pitch_0_results.csv")

    test_dataset(model, reference_embeddings, detector,
                PEOPLE, "dataset/test_pitch_40", "pitch_40_results.csv")
    
    test_dataset(model, reference_embeddings, detector,
                PEOPLE,"dataset/test_roll_90","roll_90_results.csv")
    
    test_dataset(model, reference_embeddings, detector,
                PEOPLE,"dataset/test_scale_0.5","test_scale_0.5_results.csv")
    
    test_dataset(model, reference_embeddings, detector,
                PEOPLE,"dataset/test_scale_1.0","test_scale_1.0_results.csv")
    
    test_dataset(model, reference_embeddings, detector,
                PEOPLE,"dataset/test_scale_1.5","test_scale_1.5_results.csv")