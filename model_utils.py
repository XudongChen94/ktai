import os
import cv2
import numpy as np
import csv


EMBEDDINGS_FOLDER = "embeddings/facenet/"
PEOPLE = ["dustin", "elf", "mike"]
RESULTS_FOLDER = "results/facenet"

def create_input(image_path):
    """Leest een pad en geeft een afbeelding terug."""
    image_bgr = cv2.imread(image_path)
    
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
   
    return image_bgr

def load_all_embeddings():
    """Returns a dict: person -> array with shape (number_of_faces, 512)."""
    embeddings = {}
    for person in PEOPLE:
        embeddings[person] = np.load(f"{EMBEDDINGS_FOLDER}/{person}_facenet.npy")
    return embeddings

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

def calculate_distance(embedding_1, embedding_2):
    """
    Berekent de genormaliseerde Euclidische afstand tussen twee embeddings.
    """

    return np.linalg.norm(embedding_1 - embedding_2)


def predict_identity(distances, threshold):
    """
    Geeft de persoon met de kleinste afstand terug.
    Is die afstand groter dan de drempel, dan "Unknown".
    """

    best_person = min(distances, key=distances.get)
    best_distance = distances[best_person]

    if best_distance > threshold:
        return "Unknown"

    return best_person


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
