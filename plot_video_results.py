import pandas as pd
import matplotlib.pyplot as plt

SAMPLES_PER_SECOND = 2
RESULTS_FOLDER = "results/facenet"
# zorg dat de order overal gelijk is
ORDER = ["dustin", "elf", "mike", "unknown"]


def plot_frequency(table):
    """Staafdiagram: hoeveel seconden elk personage in beeld is."""
    counts = table["person"].value_counts().reindex(ORDER, fill_value=0)
    seconds_on_screen = counts / SAMPLES_PER_SECOND

    plt.figure()
    plt.bar(seconds_on_screen.index, seconds_on_screen.values)
    plt.title("Appearance frequency per character")
    plt.ylabel("Seconds on Screen")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_FOLDER}/frequency_plot.png", dpi=150)
    plt.show()


def plot_timeline(table):
    """Tijdlijn: wanneer elk personage in beeld is."""
    # elke naam omzetten naar een getal: dustin 0, elf 1, mike 2, unknown 3
    y_positions = table["person"].map(ORDER.index)

    plt.figure(figsize=(12, 3))
    plt.scatter(table["time"], y_positions, marker="|", s=200)
    plt.yticks(range(len(ORDER)), ORDER)
    plt.xlabel("Time (seconds)")
    plt.title("Timeline per character")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_FOLDER}/timeline.png", dpi=150)
    plt.show()


def main():
    table = pd.read_csv(f"{RESULTS_FOLDER}/video_results.csv")
    plot_frequency(table)
    plot_timeline(table)


if __name__ == "__main__":
    main()