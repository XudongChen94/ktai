import get_embeddings_facenet
import torch
import os
import numpy as np

EMBEDDINGS_FOLDER = f"embeddings/facenet/"
PEOPLE = ["dustin", "elf", "mike"]


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