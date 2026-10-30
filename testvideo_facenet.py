import os
import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN


def load_reference_embeddings():
    """
    Load data from the embeddings into dict
    """
    reference_embeddings = {}
    for x in os.listdir("embeddings/facenet"):
        if x.endswith(".npy"):
            identity = x.split("_")[0] #take name out of the filename
            print(identity)
            x_path = os.path.join("embeddings/facenet", x)
            reference_embeddings[identity] = np.load(x_path)

            print(identity, reference_embeddings[identity].shape)

    #elf_embeddings = reference_embeddings["elf"] # this way you can thus get the embeddings of each person
    #print(elf_embeddings)
    print("klaar load_reference()")
    return reference_embeddings, identity

def detect_faces(testvideo, mtcnn):
    """
    Detect faces in video frames and save frame containing face
    """
    os.makedirs("detect_faces", exist_ok=True)
    for filename in os.listdir("detect_faces"):
        file_path = os.path.join("detected_faces", filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    video = cv2.VideoCapture(testvideo)
    frame_number = 0
    saved_frames=0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame_number +=1
        #MTCNN verwacht een RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        boxes, probabilities = mtcnn.detect(image)

        if boxes is not None:
            valid_boxes = [box for box, probability in zip(boxes, probabilities)
                           if probability is not None and probability >=0.90]
            if valid_boxes: #sla het op zonder rechthoeken
                filename = os.path.join("detected_faces", f"frame{frame_number:05d}.jpg")
                cv2.imwrite(filename, frame)
                saved_frames +=1
                display_frame = frame.copy()
                for box in valid_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(display_frame, (x1, y1), (x2,y2), (0,0,225), 2)
                cv2.imshow("Video", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break


    video.release()
    cv2.destroyAllWindows()

    print("klaar. opgeslagenframes:", saved_frames)
    print("klaar video_embedding()")

def embedding_videoframe(mtcnn, facenet):
    """
    Extract facenet embeddings from video frames
    """
    video_embeddings = {}
    os.makedirs("video_embeddings", exist_ok=True)

    for filename in os.listdir("video_embeddings"):
        file_path = os.path.join("video_embeddings", filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    for filename in sorted(os.listdir("detect_faces")):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        file_path = os.path.join("detected_faces", filename)
        img = cv2.imread(file_path)
        if img is None:
            continue
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        #MTCNN detect, cropt, lines faces
        face_tensors = mtcnn(pil_image)
        if face_tensors is None:
            print(f"geen gezicht gevonden: {filename}")

            continue
        if face_tensors.ndim == 3:
            face_tensors = face_tensors.unsqueeze(0)

        for i, face_tensor in enumerate(face_tensors, start=1):
            face_tensor = face_tensor.unsqueeze(0)
            with torch.no_grad():
                embedding = facenet(face_tensor)
            embedding = embedding.cpu().numpy().flatten()
            name = f"{os.path.splitext(filename)[0]}_face_{i}"
            video_embeddings[name] = embedding

            np.save(os.path.join("video_embeddings", name + ".npy"), embedding)

            print(f"Embedding saved: {name}, shape: {embedding.shape}")
    print("klaar embedding_videoframe()")
    return video_embeddings

def calculate(video_embeddings, reference_embeddings,characters, threshold=0.8):

    """
    calculate the normalized euclidean distance between the new face embedding and the reference face embeddings
    If the distance is smaller than a specified threshold then you can consider it as the same identity. for just
    the three personages
    """
    print("start")
    results = {}

    for face_name, video_embedding in video_embeddings.items():
        best_char = "onbekend"
        best_dist = float("inf")
        for character in characters:
            if character not in reference_embeddings:
                continue
            refs = np.asarray(reference_embeddings[character], dtype=np.float32)
            if refs.ndim==1:
                refs = refs.reshape(1, -1)
            distances = np.linalg.norm(refs - video_embedding.reshape(1, -1), axis=1)

            distance = float(np.min(distances))
            if distance < best_dist:
                best_dist = distance
                best_char = character
        if best_dist > threshold:
            best_char = "onbekend"
        results[face_name] = {"character": best_char, "distance": best_dist}

        print(f"{face_name} -  {best_char}")
        print(f"afstand:{best_dist:.4f}")

    print("klaar calculate()")
    return results

def main():
    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        keep_all=True)
    facenet = InceptionResnetV1(pretrained="vggface2").eval()
    characters = ["elf", "dustin", "mike"]

    testvideo = "Schermopname 2026-10-16 om 20.11.32.mov"

    reference_embeddings, identity = load_reference_embeddings()

    missing = [name for name in characters if name not in reference_embeddings]
    if missing:
        raise ValueError(f"geen referentie embedding gevonden voor: {missing}")
    reference_embeddings = {name: reference_embeddings[name] for name in characters}


    detect_faces(testvideo, mtcnn)
    video_embeddings = embedding_videoframe(mtcnn, facenet)
    results = calculate(video_embeddings,reference_embeddings,characters,threshold=0.8)


    return results


main()