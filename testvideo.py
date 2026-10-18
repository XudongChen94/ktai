import os
import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode
ImageEmbedder = mp.tasks.vision.ImageEmbedder
ImageEmbedderOptions = mp.tasks.vision.ImageEmbedderOptions
detector_model_path = 'models/blaze_face_short_range.tflite'
embedder_model_path = 'models/mobilenet_v3_small.tflite'
#embeddings = np.load("embeddings/dustin.npy")
#print(embeddings.shape) # laat zien hoeveel embeddings je hebt
#print(embeddings) # toont opgeslagen waarden
#print("Aantal embeddings;", len(embeddings))


def load_reference_embeddings():
    """
    Load data from the embeddings into dict
    """
    reference_embeddings = {}
    for x in os.listdir("embeddings"):
        if x.endswith(".npy"):
            identity = x[:-4] #take name out of the filename
            x_path = os.path.join("embeddings", x)
            reference_embeddings[identity] = np.load(x_path)

            #print(identity, reference_embeddings[identity].shape)

    #elf_embeddings = reference_embeddings["elf"] # this way you can thus get the embeddings of each person
    #print(elf_embeddings)
    return reference_embeddings, identity

def stap_twee(testvideo, model_path):
    """
    detect faces at each frame from the testvideo and saves them in a map
    """


    saved_frames = 0
    frame_number = 0
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE, min_detection_confidence=0.7) #confidence because he also saw other
    # stuff as a face

    # remove the files in detected_faces before it checks video
    for filename in os.listdir('detected_faces'):
        file_path = os.path.join('detected_faces', filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


    video = cv2.VideoCapture(testvideo)
    with FaceDetector.create_from_options(options) as detector:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame_number += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format= mp.ImageFormat.SRGB, data=rgb_frame)
            result = detector.detect(mp_image)
            # detect faces in frame
            for detection in result.detections:
                box = detection.bounding_box
                left = max(0, box.origin_x)
                top = max(0, box.origin_y)
                right = min(frame.shape[1], left + box.width)
                bottom = min(frame.shape[0], top + box.height)

                cv2.rectangle(frame, (left, top), (right, bottom), (0,0,225), 2)

            if result.detections:
                filename = os.path.join('detected_faces', f"frame_{frame_number:05d}.jpg")
                cv2.imwrite(filename, frame)
                saved_frames +=1
                print(saved_frames)

                cv2.imshow("video", frame)

                if cv2.waitKey(25) & 0xFF == ord('q'): # press q to exit
                    break

    video.release()
    cv2.destroyAllWindows()
    print("we are done now and saved it all", saved_frames)

def stap_drie():
    """
    extract new face embeddings and makes the picture in detected_faces smalle
    """
    video_embeddings = {}
    detector_options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=detector_model_path),
        running_mode=VisionRunningMode.IMAGE)
    embedder_options = ImageEmbedderOptions(
        base_options=BaseOptions(model_asset_path=embedder_model_path),
        l2_normalize=True,
        quantize=False,
        running_mode=VisionRunningMode.IMAGE)
    os.makedirs("video_embeddings", exist_ok=True)
    # remove the files in detected_faces before it checks video
    for filename in os.listdir('video_embeddings'):
        file_path = os.path.join('video_embeddings', filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    with FaceDetector.create_from_options(detector_options) as detector, \
        ImageEmbedder.create_from_options(embedder_options) as embedder:
        for filename in sorted(os.listdir("detected_faces")):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            file_path = os.path.join("detected_faces", filename)
            img = cv2.imread(file_path)
            if img is None:
                continue
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data=image)
            detection_result = detector.detect(mp_image)
            if not detection_result.detections:
                print(f"Geen gezicht gevonden: {filename}")
                continue
            for i, detection in enumerate(detection_result.detections, start=1):
                box = detection.bounding_box
                h, w, _ = image.shape
                x1 = max(box.origin_x, 0)
                y1 = max(box.origin_y, 0)
                x2 = min(box.origin_x + box.width, w)
                y2 = min(box.origin_y + box.height, h)

                if x2 <= x1 or y2 <= y1:
                    continue
                face_crop = np.ascontiguousarray(image[y1:y2, x1:x2])
                face_mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data=face_crop)

                embedding_rsult = embedder.embed(face_mp_image)
                embedding = np.array(embedding_rsult.embeddings[0].embedding)
                name = f"{os.path.splitext(filename)[0]}_face_{i}"
                video_embeddings[name] = embedding

                np.save(os.path.join("video_embeddings", name + ".npy"), embedding)


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


    return results
def main():
    characters = ["elf", "dustin", "mike"]
    model_path = 'models/blaze_face_short_range.tflite'
    testvideo = "Schermopname 2026-10-16 om 20.11.32.mov"
    reference_embeddings, identity = load_reference_embeddings()

    missing = [name for name in characters if name not in reference_embeddings]
    if missing:
        raise ValueError(f"geen referentie embedding gevonden voor: {missing}")
    reference_embeddings = { name: reference_embeddings[name] for name in characters}

    stap_twee(testvideo, model_path)
    video_embeddings = stap_drie()

    results = calculate(video_embeddings, reference_embeddings,characters,)


main()
