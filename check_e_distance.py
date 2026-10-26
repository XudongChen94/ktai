import numpy as np
import pandas as pd

PEOPLE = ["dustin", "elf", "mike"]
EMBEDDINGS_FOLDER = "embeddings/facenet"


def load_embeddings():
    """Returns a dict: person -> array with shape (number_of_faces, 512)."""
    embeddings = {}
    for person in PEOPLE:
        embeddings[person] = np.load(f"{EMBEDDINGS_FOLDER}/{person}_facenet.npy")
    return embeddings

def distance_matrix(group_a, group_b):
    """Distance between every face in group_a and every face in group_b."""
    rows = []
    for face_a in group_a:
        row = []
        for face_b in group_b:
            row.append(np.linalg.norm(face_a - face_b))
        rows.append(row)
    return np.array(rows)

def summarize(distances, same_group):
    """Returns (mean, min, max) of the distances."""
    if same_group:
        mask = ~np.eye(len(distances), dtype=bool)
        values = distances[mask]
    else:
        values = distances.ravel()

    return values.mean(), values.min(), values.max()

def main():
    embeddings = load_embeddings()
    for index, first_person in enumerate(PEOPLE):
        for second_person in PEOPLE[index:]:
            distances = distance_matrix(embeddings[first_person], embeddings[second_person])
            same_group = first_person == second_person
            mean, smallest, largest = summarize(distances, same_group)
            print(
                f"{first_person} - {second_person}: mean {mean:.2f}  min {smallest:.2f}  max {largest:.2f}")

    dustin_test = np.load(f"{EMBEDDINGS_FOLDER}/dustin_facenet_test.npy")
    for person in PEOPLE:
        distances = distance_matrix(dustin_test, embeddings[person])
        mean, smallest, largest = summarize(distances, same_group=False)
        print(
            f"dustin test - {person} train: mean {mean:.2f}  min {smallest:.2f}  max {largest:.2f}")

    for person in PEOPLE:
        distances = distance_matrix(dustin_test, embeddings[person])
        closest_per_photo = distances.min(axis=1)
        print(person, closest_per_photo.shape)
        print(np.round(closest_per_photo, 2))

    closest_per_person = {}

    for person in PEOPLE:
        distances = distance_matrix(dustin_test, embeddings[person])
        closest_per_person[person] = distances.min(axis=1)

    all_distances = np.stack([
        closest_per_person["dustin"],
        closest_per_person["elf"],
        closest_per_person["mike"],
    ])# vorm (3, 16)
    predicted_index = np.argmin(all_distances, axis=0)
    correct = predicted_index == 0  # index 0 =  Dustin
    print(correct.sum(), "van", len(correct))

    photo_names = open(
        f"{EMBEDDINGS_FOLDER}/dustin_facenet_test_names.txt").read().splitlines()

    table = pd.DataFrame(all_distances.T, columns=PEOPLE, index=photo_names)
    table["predicted"] = [PEOPLE[i] for i in predicted_index]
    print(table.round(2))

if __name__ == "__main__":
    main()