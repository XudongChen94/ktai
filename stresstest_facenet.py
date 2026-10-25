import glob
import os
import csv
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
import get_embeddings_facenet
import crop_faces


PERSON = "elf"
EMBEDDINGS_FILE = f"embeddings/facenet/{PERSON}_facenet.npy"
# Eén testafbeelding voor een vergelijking met de gemiddelde embedding van elf
# TEST_IMAGE = f"dataset/test/{PERSON}/afstand v camera 3.png"
# TEST_IMAGE = f"dataset/test/dustin/stills_dustin_stranger_things_001.jpg"
TEST_IMAGE = f"dataset/test/mike/mike (7).jpeg"

# Voor de threshold-test
SAME_PERSON_FOLDER = f"frames/faces/{PERSON}/test/same"
DIFFERENT_PERSON_FOLDER = f"frames/faces/{PERSON}/test/different"
RESULTS_FOLDER = "results/facenet"
INPUT_SIZE = 160


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

def load_reference_embedding():
    """
    Laadt alle eerder gemaakte embeddings en berekent
    daarvan één gemiddelde reference-embedding.
    """

    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_FILE}")

    embeddings = np.load(EMBEDDINGS_FILE)
    if len(embeddings) == 0:
        raise ValueError(f"No embeddings found in: {EMBEDDINGS_FILE}")

    reference_embedding = np.mean(embeddings, axis=0)
    return reference_embedding


def calculate_distance(embedding_1, embedding_2):
    """
    Berekent de genormaliseerde Euclidische afstand tussen twee embeddings.
    """

    embedding_1 = embedding_1 / np.linalg.norm(embedding_1)
    embedding_2 = embedding_2 / np.linalg.norm(embedding_2)

    return np.linalg.norm(embedding_1 - embedding_2)


def test_thresholds(model, reference_embedding, test_image, detector):

    faces = crop_faces.crop_faces(test_image, detector)

    if not faces:
        print("No faces detected")
        return

    print(f"Number of detected faces: {len(faces)}\n")

    for face_index, test_face in enumerate(faces):
        test_embedding = get_embedding_from_image(test_face,model)
        distance = calculate_distance(reference_embedding,test_embedding)

        print(f"--- Face {face_index + 1} ---")
        print(f"Distance: {distance:.3f}")

        thresholds = np.arange(0.2, 2.0, 0.2)

        for threshold in thresholds:
            prediction = ("Same identity"if distance < threshold else "Different identity")

            print(f"Threshold: {threshold:.1f} | "f"Distance: {distance:.3f} | "
                f"{prediction}"
            )
        print()
        cv2.imshow(f"Cropped face {face_index + 1}",test_face)

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


def test_scale(model, reference_embedding, detector):
    """
    Test per gedetecteerd gezicht hoe de embedding-distance verandert
    wanneer het gecropte gezicht kleiner of groter wordt gemaakt.
    """

    image_bgr = get_embeddings_facenet.create_input(TEST_IMAGE)
    faces = crop_faces.crop_faces(image_bgr,detector)

    if not faces:
        print(f"No faces detected in: {TEST_IMAGE}")
        return

    print(f"Number of detected faces: {len(faces)}")

    scale_output_folder = os.path.join(RESULTS_FOLDER,"scale_images")
    os.makedirs(scale_output_folder, exist_ok=True)
    results = []

    # Alle gevonden gezichten testen
    for face_index, face in enumerate(faces):
        print(f"\n--- Face {face_index + 1} ---")

        # Originele face crop bewaren
        original_face_path = os.path.join(scale_output_folder,f"face_{face_index + 1}_original.jpg")
        cv2.imwrite(original_face_path, face)

        scales = [0.50, 0.75, 1.00, 1.25, 1.50]
        for scale in scales:
            scaled_face = change_scale(face,scale)
            embedding = get_embedding_from_image(scaled_face,model)
            distance = calculate_distance(embedding,reference_embedding)
            output_image_path = os.path.join(scale_output_folder,
                                             f"face_{face_index + 1}_scale_{scale:.2f}.jpg")
            cv2.imwrite(output_image_path,scaled_face)
            results.append([face_index + 1,scale,distance,output_image_path])
            print(f"Face: {face_index + 1} | "f"Scale: {scale:.2f} | "
                  f"Distance: {distance:.4f}")

            # Geschaalde afbeelding tonen
            cv2.imshow(f"Face {face_index + 1} - Scale {scale:.2f}",scaled_face)

    save_results("scale_results.csv",["face", "scale", "distance", "image_path"], results)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

def test_angle():
    return

def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    print("Loading FaceNet model...")

    model = InceptionResnetV1(pretrained="vggface2").eval()
    reference_embedding = load_reference_embedding()
    print("Reference embedding loaded.")
    print(f"Reference shape: {reference_embedding.shape}")

    with crop_faces.FaceDetection(model_selection=1, min_detection_confidence=0.5
        ) as detector:
    #     test_image = get_embeddings_facenet.create_input(TEST_IMAGE)
    #     test_thresholds(model, reference_embedding, test_image, detector)
        test_scale(model,reference_embedding, detector)

    # test_angle(model,reference_embedding)
    # test_threshold( model,reference_embedding)


if __name__ == "__main__":
    main()