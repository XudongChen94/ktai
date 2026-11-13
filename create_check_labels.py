import csv
import os

CHECK_FOLDER = "results/facenet/check"
OUTPUT_FILE = "archive/check_labels.csv"
PREDICTED_GROUPS = ["dustin", "elf", "mike", "unknown"]

# Stills die afwijken
real_label_exceptions = {
    ("mike", "49.87_s_face1_0.875.jpg"): "other",
    ("unknown", "182.84_s_face0_0.968.jpg"): "elf",
    ("unknown", "183.33_s_face0_0.981.jpg"): "elf",
    ("unknown", "142.26_s_face0_1.03.jpg"): "elf",
}

rows = []
for predicted in PREDICTED_GROUPS:
    folder = os.path.join(CHECK_FOLDER, predicted)
    for file_name in sorted(os.listdir(folder)):
        if not file_name.endswith(".jpg"):
            continue

        if (predicted, file_name) in real_label_exceptions:
            real = real_label_exceptions[(predicted, file_name)]
        elif predicted == "unknown":
            real = "other"
        else:
            real = predicted

        rows.append([f"{predicted}/{file_name}", predicted, real])

with open(OUTPUT_FILE, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["file", "person_predicted", "person_real"])
    writer.writerows(rows)

print(len(rows), "rows saved to", OUTPUT_FILE)