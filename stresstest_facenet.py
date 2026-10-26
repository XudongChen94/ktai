import glob
import os
import csv
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
import get_embeddings_facenet
import crop_faces


PEOPLE = ["dustin", "elf", "mike"]
EMBEDDINGS_FOLDER = f"embeddings/facenet/"
# Eén testafbeelding voor een vergelijking met de gemiddelde embedding van elf
# TEST_IMAGE = f"dataset/test/{PERSON}/afstand v camera 3.png"
# TEST_IMAGE = f"dataset/test/dustin/stills_dustin_stranger_things_001.jpg"
TEST_IMAGE = f"dataset/test/mike/mike (7).jpeg"

# Voor de resultaten
RESULTS_FOLDER = "results/facenet"

# Voor het testen van het model op test
TEST_DATASET_FOLDER = "dataset/test/"
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")

# Standaard threshold, pas aan om te experimeteren
THRESHOLD = 0.9

# Threshold voor de voorspelling in test_angle
ANGLE_THRESHOLD = 0.8  

# Hoeken voor het draaien
ANGLES = {
    "roll": [-90, -45, 0, 45, 90], # Hoofd draaien
    "yaw": [-40, -20, 0, 20, 40], # Nee schudden
    "pitch": [-40, -20, 0, 20, 40], # Ja knikken
}


def get_embedding_from_image(image_bgr, model):
    """
    Maakt een FaceNet-embedding van een OpenCV-afbeelding.
    """

    image_tensor = get_embeddings_facenet.create_input_from_image(image_bgr)

    with torch.no_grad():
        batch = image_tensor.unsqueeze(0)
        output = model(batch)
        embedding = output[0].cpu().numpy()

    return embedding

def load_reference_embeddings():
    """
    Laadt alle embeddings uit alle .npy-bestanden in EMBEDDINGS_FOLDER.
    Geeft een lijst met namen en een lijst met embeddings terug.
    """
    reference_embeddings = {}

    for person in PEOPLE:
        embeddings_file = os.path.join(EMBEDDINGS_FOLDER, f"{person}_facenet.npy")

        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

        embeddings = np.load(embeddings_file)
        if len(embeddings) == 0:
            raise ValueError(f"No embeddings found in: {embeddings_file}")

        mean_embedding = np.mean(embeddings, axis=0)
        reference_embeddings[person] = mean_embedding / np.linalg.norm(mean_embedding)

    return reference_embeddings


# Albert: volgens mij hoeft dit niet, en geeft FaceNet zelf al een genormaliseerde vector terug.
# Xudong: Klopt, FaceNet geeft inderdaad een genormaliseerde vector terug. Functie aangepast, zodat 
# alleen de Euclidische afstand tussen genormaliseerde vectoren wordt berekend.
def calculate_distance(embedding_1, embedding_2):
    """
    Berekent de genormaliseerde Euclidische afstand tussen twee embeddings.
    """

    return np.linalg.norm(embedding_1 - embedding_2)


def predict_identity(distances, threshold):
    """
    Geeft de naam terug als precies één persoon onder de threshold zit,
    anders "Unknown".
    """
    matches = [name for name, distance in distances.items() if distance < threshold]

    if len(matches) == 1:
        return matches[0]

    return "Unknown"


def test_thresholds(model, reference_embeddings, test_image, detector):

    faces = crop_faces.crop_faces(test_image, detector)

    if not faces:
        print("No faces detected")
        return

    print(f"Number of detected faces: {len(faces)}\n")

    thresholds = np.arange(0.2, 2.0, 0.2)

    for face_index, test_face in enumerate(faces):
        test_embedding = get_embedding_from_image(test_face, model)
        distances = {
            name: calculate_distance(reference_embeddings[name], test_embedding)
            for name in PEOPLE
        }

        print(f"--- Face {face_index + 1} ---")

        header = (f"{'Threshold':<10}| "
                  + " | ".join(f"{name.capitalize():<7}" for name in PEOPLE)
                  + " | Prediction")
        print(header)
        print("-" * len(header))

        for threshold in thresholds:
            prediction = predict_identity(distances, threshold)

            print(f"{threshold:<10.1f}| "
                  + " | ".join(f"{distances[name]:<7.3f}" for name in PEOPLE)
                  + f" | {prediction.capitalize()}")
        print()
        cv2.imshow(f"Cropped face {face_index + 1}", test_face)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_results(file_name, headers, rows):
    """
    Slaat experimentresultaten op als CSV-bestand.
    """

    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    output_path = os.path.join(RESULTS_FOLDER, file_name)

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Results saved to: {output_path}")


def change_scale(image, scale):
    """
    Maakt het gezicht kleiner of groter binnen dezelfde afbeelding.

    scale < 1:
        De afbeelding wordt kleiner en krijgt zwarte ruimte eromheen.

    scale > 1:
        De afbeelding wordt groter en wordt aan de randen afgesneden.
    """

    height, width = image.shape[:2]

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    scaled_image = cv2.resize(image,(new_width, new_height))

    # Zwart canvas met dezelfde grootte als de originele afbeelding
    result = np.zeros_like(image)

    # Bepalen welk gedeelte van de geschaalde afbeelding wordt gebruikt
    source_x1 = max(0, (new_width - width) // 2)
    source_y1 = max(0, (new_height - height) // 2)

    source_x2 = min(new_width, source_x1 + width)
    source_y2 = min(new_height, source_y1 + height)

    # Bepalen waar het gedeelte op het canvas komt
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


def test_scale(model, reference_embeddings, detector):
    """
    Test per gedetecteerd gezicht hoe de embedding-distances en de voorspelling
    veranderen wanneer het gecropte gezicht kleiner of groter wordt gemaakt.
    """

    image_bgr = get_embeddings_facenet.create_input(TEST_IMAGE)
    faces = crop_faces.crop_faces(image_bgr, detector)

    if not faces:
        print(f"No faces detected in: {TEST_IMAGE}")
        return

    print(f"Number of detected faces: {len(faces)}")
    scale_output_folder = os.path.join(RESULTS_FOLDER, "scale_images")
    os.makedirs(scale_output_folder, exist_ok=True)
    results = []
    scales = [0.50, 0.75, 1.00, 1.25, 1.50]

    # Alle gevonden gezichten testen
    for face_index, face in enumerate(faces):

        print(f"\n--- Face {face_index + 1} ---")

        header = (f"{'Scale':<7}| {'Threshold':<10}| "
                  + " | ".join(f"{name.capitalize():<7}" for name in PEOPLE)
                  + " | Prediction")
        print(header)
        print("-" * len(header))

        # Originele face crop bewaren
        # original_face_path = os.path.join(scale_output_folder, f"face_{face_index + 1}_original.jpg")
        # cv2.imwrite(original_face_path, face)

        for scale in scales:
            scaled_face = change_scale(face, scale)
            test_embedding = get_embedding_from_image(scaled_face, model)
            distances = {
                name: calculate_distance(test_embedding, reference_embeddings[name])
                for name in PEOPLE
            }

            output_image_path = os.path.join(scale_output_folder,
                                             f"face_{face_index + 1}_scale_{scale:.2f}.jpg")
            # cv2.imwrite(output_image_path, scaled_face)

            prediction = predict_identity(distances, THRESHOLD)
            print(f"{scale:<7.2f}| {THRESHOLD:<10.1f}| "
                    + " | ".join(f"{distances[name]:<7.3f}" for name in PEOPLE)
                    + f" | {prediction.capitalize()}")
            
            results.append([face_index + 1, scale, round(THRESHOLD, 1)]
                            + [distances[name] for name in PEOPLE]
                            + [prediction, output_image_path])
            # Geschaalde afbeelding tonen
            cv2.imshow(f"Face {face_index + 1} - Scale {scale:.2f}", scaled_face)

    save_results("scale_results.csv",
                 ["face", "scale", "threshold"] + PEOPLE + ["prediction", "image_path"],
                 results)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def test_dataset(model, reference_embeddings, detector, people_to_test):
    """
    Test alle foto's in dataset/test/<persoon> voor elke persoon in people_to_test.
    Per foto, per gezicht en per threshold wordt een rij naar de CSV geschreven.
    """

    results = []
    for true_label in people_to_test:
        person_folder = os.path.join(TEST_DATASET_FOLDER, true_label)

        image_paths = []
        for extension in IMAGE_EXTENSIONS:
            image_paths.extend(glob.glob(os.path.join(person_folder, extension)))

        if not image_paths:
            print(f"No images found in: {person_folder}")
            continue

        print(f"\nTesting {true_label}: {len(image_paths)} images")

        for image_path in sorted(image_paths):
            image_bgr = get_embeddings_facenet.create_input(image_path)

            if image_bgr is None:
                print(f"Could not read: {image_path}")
                continue
            faces = crop_faces.crop_faces(image_bgr, detector)
            
            if not faces:
                print(f"No faces detected in: {image_path}")
                continue

            for face_index, face in enumerate(faces):
                test_embedding = get_embedding_from_image(face, model)
                distances = {
                    name: calculate_distance(test_embedding, reference_embeddings[name])
                    for name in PEOPLE
                }
                prediction = predict_identity(distances, THRESHOLD)
                results.append(
                    [os.path.basename(image_path), face_index + 1, true_label, round(THRESHOLD, 1)]
                    + [distances[name] for name in PEOPLE]
                    + [prediction]
                )

    save_results(
        "dataset_results.csv",
        ["image", "face", "true_label", "threshold"] + PEOPLE + ["prediction"],
        results
    )


def rotate_image(image, angle):
    """
    Draait de afbeelding om het midden (roll).
    Positieve hoek = tegen de klok in. Lege hoeken worden zwart.
    """

    height, width = image.shape[:2]
    # Middelste punt van een afbeelding vinden
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def change_perspective(image, angle, axis):
    """
    Simuleert een gedraaid hoofd met een perspectiefvervorming.

    axis = "yaw":   links/rechts draaien (één zijkant wordt smaller)
    axis = "pitch": van boven/onder kijken (boven- of onderkant wordt smaller)
    """

    if angle == 0:
        return image.copy()

    height, width = image.shape[:2]

    # Hoe groter de hoek, hoe sterker de vervorming
    factor = np.sin(np.radians(abs(angle))) * 0.3
    source_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    if axis == "yaw":
        shrink = factor * height
        squeeze = factor * width

        if angle > 0:
            # Rechterkant verder weg
            target_points = np.float32([[0, 0], [width - squeeze, shrink],
                                        [width - squeeze, height - shrink], [0, height]])
        else:
            # Linkerkant verder weg
            target_points = np.float32([[squeeze, shrink], [width, 0],
                                        [width, height], [squeeze, height - shrink]])

    elif axis == "pitch":
        shrink = factor * width
        squeeze = factor * height

        if angle > 0:
            # Bovenkant verder weg (camera van onderen)
            target_points = np.float32([[shrink, squeeze], [width - shrink, squeeze],
                                        [width, height], [0, height]])
        else:
            # Onderkant verder weg (camera van boven)
            target_points = np.float32([[0, 0], [width, 0],
                                        [width - shrink, height - squeeze], [shrink, height - squeeze]])
    else:
        raise ValueError(f"Unknown axis: {axis}")

    matrix = cv2.getPerspectiveTransform(source_points, target_points)
    return cv2.warpPerspective(image, matrix, (width, height),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def change_angle(image, angle_type, angle):
    """
    Kiest de juiste vervorming op basis van het type hoek.
    """

    if angle_type == "roll":
        return rotate_image(image, angle)

    return change_perspective(image, angle, angle_type)



def test_angle(model, reference_embeddings, detector, people_to_test):
    """
    Test per gedetecteerd gezicht hoe de embedding-distances en de voorspelling
    veranderen bij rotatie (roll) en perspectiefvervorming (yaw, pitch).
    """
    
    image_paths = []
    for person in people_to_test:
        person_folder = os.path.join(TEST_DATASET_FOLDER, person)
        for extension in IMAGE_EXTENSIONS:
            for image_path in glob.glob(os.path.join(person_folder, extension)):
                image_paths.append((image_path, person))

    if not image_paths:
        print("No images found")
        return

    results = []
    for image_path, true_label in sorted(image_paths):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_bgr = get_embeddings_facenet.create_input(image_path)
        faces = crop_faces.crop_faces(image_bgr, detector)

        if not faces:
            print(f"No faces detected in: {image_name}")
            continue

        print(f"Number of detected faces: {len(faces)}")
        print(f"Threshold: {ANGLE_THRESHOLD}")

        for face_index, face in enumerate(faces):
            print(f"\n--- Face {face_index + 1} ---")

            header = (f"{'Type':<7}| {'Angle':<7}| "
                    + " | ".join(f"{name.capitalize():<7}" for name in PEOPLE)
                    + " | Prediction")
            print(header)
            print("-" * len(header))

            for angle_type, angles in ANGLES.items():
                print("hoi")
                for angle in angles:
                    angled_face = change_angle(face, angle_type, angle)
                    test_embedding = get_embedding_from_image(angled_face, model)

                    distances = {
                        name: calculate_distance(test_embedding, reference_embeddings[name])
                        for name in PEOPLE
                    }

                    prediction = predict_identity(distances, ANGLE_THRESHOLD)
                    print(f"{angle_type:<7}| {angle:<7}| "
                        + " | ".join(f"{distances[name]:<7.3f}" for name in PEOPLE)
                        + f" | {prediction.capitalize()}")

                    results.append([image_name, face_index + 1, true_label, angle_type, angle]
                                    + [distances[name] for name in PEOPLE]
                                    + [prediction])

                print()

    save_results("angle_results.csv",
                ["image", "face", "true_label", "angle_type", "angle"]
                + PEOPLE + ["prediction"],
                results)

def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    print("Loading FaceNet model...")

    model = InceptionResnetV1(pretrained="vggface2").eval()
    reference_embeddings = load_reference_embeddings()
    print(f"Reference embeddings loaded: {len(reference_embeddings)}")
    print("Reference embedding loaded.")

    with crop_faces.FaceDetection(model_selection=1, min_detection_confidence=0.5
        ) as detector:
        # test_image = get_embeddings_facenet.create_input(TEST_IMAGE)
        # test_thresholds(model, reference_embeddings, test_image, detector)
        # test_dataset(model, reference_embeddings, detector, ["mike"])
        # test_scale(model,reference_embeddings, detector)
        test_angle(model, reference_embeddings, detector, ["mike"])


if __name__ == "__main__":
    main()