import numpy as np
import pandas as pd

METHOD = "1nn"   # "1nn" of "centroid"
TEST_PERSON = "mike"
PEOPLE = ["dustin", "elf", "mike"]
EMBEDDINGS_FOLDER = "embeddings/facenet"
THRESHOLD = 0.9

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

    test_embeddings = np.load(f"{EMBEDDINGS_FOLDER}/{TEST_PERSON}_facenet_test.npy")
    true_index = PEOPLE.index(TEST_PERSON)
    for person in PEOPLE:
        distances = distance_matrix(test_embeddings, embeddings[person])
        mean, smallest, largest = summarize(distances, same_group=False)
        print(
            f"{TEST_PERSON} test - {person} train: mean {mean:.2f}  min {smallest:.2f}  max {largest:.2f}")

    closest_per_person = {}

    for person in PEOPLE:
        distances = distance_matrix(test_embeddings, embeddings[person])
        closest_per_person[person] = distances.min(axis=1)

    all_distances = np.stack([
        closest_per_person["dustin"],
        closest_per_person["elf"],
        closest_per_person["mike"],
    ])# vorm (3, 16)
    predicted_index = np.argmin(all_distances, axis=0)
    correct = predicted_index == true_index
    print(correct.sum(), "van", len(correct))

    photo_names = open(
        f"{EMBEDDINGS_FOLDER}/{TEST_PERSON}_facenet_test_namen.txt").read().splitlines()

    table = pd.DataFrame(all_distances.T, columns=PEOPLE, index=photo_names)
    table["predicted"] = [PEOPLE[i] for i in predicted_index]


    smallest = all_distances.min(axis=0)  # kleinste afstand per foto
    table["with_threshold"] = [
        "unknown" if distance > THRESHOLD else person
        for person, distance in zip(table["predicted"], smallest)
    ]
    print(table.round(2))
    table.round(2).to_csv(f"results/facenet/{TEST_PERSON}_test_predictions.csv")

if __name__ == "__main__":
    main()