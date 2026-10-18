import glob
import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

PERSON = "elf"

INPUT_FOLDER = f"frames/faces/{PERSON}/train"
OUTPUT_FOLDER = f"embeddings/facenet/{PERSON}_facenet.npy"
NAMES_TXT = f"embeddings/facenet/{PERSON}_facenet_namen.txt"
INPUT_SIZE = 160  # FaceNet wil 160 x 160 pixels


def create_input(image_path):
    """Leest een gezichtsplaatje en maakt er een tensor van voor FaceNet."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resize = cv2.resize(image_rgb, (INPUT_SIZE,INPUT_SIZE))
    image_tensor = torch.from_numpy(image_resize).float()
    image_tensor = image_tensor.permute(2,0,1)
    image_tensor = (image_tensor - 127.5) / 128.0 # standaardiseer de pixelwaarden zoals FaceNet verwacht

    return image_tensor


def main():
    os.makedirs("embeddings", exist_ok=True)

    model = InceptionResnetV1(pretrained='vggface2').eval()

    image_paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")))
    embeddings = []
    file_names = []

    for image_path in image_paths:
        image_tensor = create_input(image_path)

        with torch.no_grad():
            batch = image_tensor.unsqueeze(0)
            output = model(batch)
            embedding = output[0].numpy()

        embeddings.append(embedding)
        file_names.append(os.path.basename(image_path))

    embeddings_array = np.array(embeddings)
    np.save(OUTPUT_FOLDER, embeddings_array)

    with open(NAMES_TXT, "w") as names_file:
        names_file.write("\n".join(file_names))



if __name__ == "__main__":
    main()