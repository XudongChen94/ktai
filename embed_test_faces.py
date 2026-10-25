import glob
import os
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
import get_embeddings_facenet

PERSON = "dustin"
SPLIT = "test"
INPUT_FOLDER = f"frames/faces/{PERSON}/{SPLIT}"
OUTPUT_FILE = f"embeddings/facenet/{PERSON}_facenet_{SPLIT}.npy"
NAMES_FILE = f"embeddings/facenet/{PERSON}_facenet_{SPLIT}_names.txt"


def main():

    model = InceptionResnetV1(pretrained='vggface2').eval()

    image_paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")))
    embeddings = []
    file_names = []

    for image_path in image_paths:
        image_bgr = get_embeddings_facenet.create_input(image_path)
        image_tensor = get_embeddings_facenet.create_input_from_image(image_bgr)

        with torch.no_grad():
            batch = image_tensor.unsqueeze(0)
            output = model(batch)
            embedding = output[0].numpy()

        embeddings.append(embedding)
        file_names.append(os.path.basename(image_path))

    embeddings_array = np.array(embeddings)
    np.save(OUTPUT_FILE, embeddings_array)

    with open(NAMES_FILE, "w") as names_file:
        names_file.write("\n".join(file_names))

    print("shape:", np.array(embeddings).shape)


if __name__ == "__main__":
    main()