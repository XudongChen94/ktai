import pandas as pd
import matplotlib.pyplot as plt

SAMPLES_PER_SECOND = 2
# zorg dat de order overal gelijk is
ORDER = ["dustin", "elf", "mike", "unknown"]

table = pd.read_csv("results/facenet/video_results.csv")


# Herschik de tabel op basis van de order
counts = table["person"].value_counts().reindex(ORDER, fill_value=0)
seconds_on_screen = counts / SAMPLES_PER_SECOND

print(seconds_on_screen.index)

plt.bar(seconds_on_screen.index, height=seconds_on_screen.values, align="center")
plt.title("Appearance frequency per character")
plt.ylabel("Seconds on Screen")
plt.savefig("results/facenet/frequency_plot.png", dpi=150)
plt.show()
