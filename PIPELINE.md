# Face recognition pipeline (Stranger Things: Dustin, Elf, Mike)

Run everything from the `ktai` folder with the Python of the `venv`
(Python 3.12, `numpy<2`, `pandas<2.3`).

## Overview

```
photos -> crop faces -> reference embeddings -> video frames -> match -> results -> plots / checks
```

| Step                                   | Script | Input | Output |
|----------------------------------------|---|---|---|
| 1. Crop faces                          | `crop_faces.py` | `dataset/train/<person>`, `dataset/test/<person>` | `frames/faces/<person>/train`, `frames/faces/<person>/test` |
| 2. Reference embeddings                | `get_embeddings_facenet.py` (set `PERSON`) | `frames/faces/<person>/train` | `embeddings/facenet/<person>_facenet.npy` |
| 3. Recognize faces in the video        | `video_recognition.py` | `video/<test video>`, reference embeddings | `results/facenet/video_results.csv` |
| 4. Plots                               | `plot_video_results.py` | `video_results.csv` | `frequency_plot.png`, `timeline.png` |
| 5. Manual check of a sample (optional) | `check_sample.py`, then `create_check_labels.py` | `video_results.csv`, same video | `results/facenet/check/<group>/`, `check_labels.csv` |
| 6. Stress tests                        | `stresstest_facenet.py` | reference embeddings, test images | `results/facenet/*.csv`, `scale_images/` |
| 7. Metrics                             | `count_results.py` | the CSV files of step 6 | TP, FP, FN, precision, recall and F1 (printed), threshold experiment |
`model_utils.py` is a helper module (loading the reference embeddings, distance, prediction, saving
results); it is not run directly.
## Settings used

| Setting | Value |
|---|---|
| Face detector | MediaPipe FaceDetection, `model_selection=1`, `min_detection_confidence=0.5` |
| Crop margin | 0.2 of the face width and height |
| Embedding model | FaceNet (`InceptionResnetV1`, `pretrained="vggface2"`, `facenet-pytorch`), not fine-tuned |
| Model input | RGB, 160 x 160, `(x - 127.5) / 128` |
| Embedding | 512 numbers, already L2-normalized |
| Distance | Euclidean distance between embeddings |
| Matching, baseline | nearest centroid: mean of the reference embeddings per person, L2-normalized |
| Matching, experiment | 1-nearest neighbour: smallest distance to any reference photo of that person |
| Threshold | 0.9: if the smallest distance is larger, the face is `unknown` (threshold experiment: 0.5 to 1.3) |
| Small images | images with a shorter side below 200 px are assumed to be a crop already and are not cropped again |
| Stress tests | scale 0.5 and 1.5; roll -90 and 90; yaw -40 and 40; pitch -40 and 40 |
| Video sampling | 2 frames per second |
| Check sample | up to 30 faces per predicted group (all faces if the group is smaller), `random_state=0` |

## Step by step

1. **Crop faces.** Set `input_folder` and `output_folder` in `crop_faces.py` and run it once per
   person and per split (train, test). Remove crops that are too small or that show the wrong person.
2. **Reference embeddings.** Set `PERSON` in `get_embeddings_facenet.py` to `dustin`, `elf` or `mike`
   and run it for each person. The training images are only used to compute reference embeddings.
3. **Video.** Set `VIDEO_PATH` in `video_recognition.py` and set `METHOD` to
   `"centroid"` (baseline) or `"1nn"` (experiment). For every sampled frame it detects and crops
   the faces, computes an embedding per face and compares it with the reference embeddings.
   Output columns: `frame, time, face, person, distance`.
4. **Plots.** Run `plot_video_results.py` for the appearance frequency (seconds on screen per
   character) and the timeline.
5. **Manual check.** Run `check_sample.py` with the same `VIDEO_PATH` as in step 3. Look at the
   saved crops per predicted group and note where the real person differs in
   `real_label_exceptions` in `create_check_labels.py`, then run it to create `check_labels.csv`.
6.  **Metrics.** Run `count_results.py`. It reads the CSV files of step 6 and prints TP,
   FP, FN, precision, recall and F1 per person and in total, for both methods. It also
   repeats this for different thresholds (0.5 to 1.3) as a threshold experiment.