import pandas as pd
import matplotlib.pyplot as plt

SAMPLES_PER_SECOND = 2
# zorg dat de order overal gelijk is
ORDER = ["dustin", "elf", "mike", "unknown"]

table = pd.read_csv("results/facenet/video_results.csv")

# Herschik de tabel op basis van de order
counts = table["person"].value_counts().reindex(ORDER, fill_value=0)
seconds_on_screen = counts / SAMPLES_PER_SECOND

plt.figure(figsize=(12, 3))

# elke naam omzetten naar een getal: dustin 0, elf 1, mike 2, unknown 3
y_positions = table["person"].map(lambda person: ORDER.index(person))

plt.scatter(table["time"], y_positions, marker="|", s=200)
plt.yticks(range(len(ORDER)), ORDER)
plt.xlabel("Time (seconds)")
plt.title("Timeline per character")
plt.tight_layout()
plt.savefig("results/facenet/timeline.png", dpi=150)
plt.show()