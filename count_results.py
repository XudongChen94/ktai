import pandas as pd

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
PEOPLE = ["dustin", "elf", "mike"]
FOLDER = "results/facenet"
TEST_SETS = ["normal_test", "pitch_-40", "pitch_40", "roll_-90", "roll_90",
             "test_scale_0.5", "test_scale_1.5", "yaw_-40", "yaw_40"]

def count(file_name):
    table = pd.read_csv(f"{FOLDER}/{file_name}")
    rows = []
    for person in PEOPLE:
        tp = ((table["true_label"] == person) & (table["prediction"] == person)).sum()
        fn = ((table["true_label"] == person) & (table["prediction"] != person)).sum()   # echt person, maar voorspeld iets anders (ook "Unknown")
        fp = ((table["true_label"] != person) & (table["prediction"] == person)).sum()    # voorspeld person, maar echt iemand anders
        rows.append([person, tp, fp, fn])
    return rows

def totals(file_name):
    rows = count(file_name)
    tp = sum(row[1] for row in rows)
    fp = sum(row[2] for row in rows)
    fn = sum(row[3] for row in rows)
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn)
    return tp, fp, fn, precision, recall

def totals_at_threshold(file_name, threshold):
    table = pd.read_csv(f"{FOLDER}/{file_name}")
    distances = table[PEOPLE]                      # de kolommen dustin, elf, mike
    nearest = distances.idxmin(axis=1)             # naam van de dichtstbijzijnde persoon
    smallest = distances.min(axis=1)               # de kleinste afstand
    prediction = nearest.where(smallest <= threshold, "Unknown")
    tp = (prediction == table["true_label"]).sum()
    fp = ((prediction != table["true_label"]) & (prediction != "Unknown")).sum()
    fn = len(table) - tp
    return tp, fp, fn


if __name__ == "__main__":
    for method in ["centroid", "1nn"]:
        print(method)
        total_tp, total_fp, total_fn = 0, 0, 0
        for person, tp, fp, fn in count(f"{method}_normal_test_results.csv"):
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            print(person, tp, fp, fn, round(precision, 2), round(recall, 2))
            total_tp += tp
            total_fp += fp
            total_fn += fn
        total_precision = total_tp / (total_tp + total_fp)
        total_recall = total_tp / (total_tp + total_fn)
        print("total", total_tp, total_fp, total_fn, total_precision, total_recall)

    for method in ["centroid", "1nn"]:
        print(method)
        for test_set in TEST_SETS:
            tp, fp, fn, precision, recall = totals(
                f"{method}_{test_set}_results.csv")
            print(test_set, tp, fp, fn, precision, round(recall, 2))

    for method in ["centroid", "1nn"]:
        print(method)
        for threshold in THRESHOLDS:
            tp, fp, fn = totals_at_threshold(
                f"{method}_normal_test_results.csv", threshold)
            precision = tp / (tp + fp) if (tp + fp) > 0 else None
            recall = tp / (tp + fn)
            print(threshold, tp, fp, fn, precision, round(recall, 2))
