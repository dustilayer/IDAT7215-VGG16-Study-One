"""Generate the README figure from the recorded per-seed metrics."""

import csv
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
with (root / "study_one/results/per_seed.csv").open() as stream:
    rows = list(csv.DictReader(stream))

variants = ("paper", "keras", "keras_gap")
labels = ("Paper input\nLocal pool", "Keras input\nLocal pool", "Keras input\nGlobal pool")
colors = ("#8495a8", "#5681ad", "#176b57")
losses = [[float(r["validation_loss"]) for r in rows if r["variant"] == name]
          for name in variants]
misses = [[int(r["missed_defects"]) for r in rows if r["variant"] == name]
          for name in variants]
fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.8), layout="constrained")
values = list(map(mean, losses))
left.bar(range(3), values, color=colors, width=0.58, yerr=list(map(stdev, losses)), capsize=4)
for index, value in enumerate(values):
    left.text(index, value + stdev(losses[index]) + 0.003, f"{value:.5f}", ha="center")
left.set(title="Validation loss across three seeds", ylabel="Mean final binary cross-entropy", ylim=(0, 0.112))
for index, (color, group) in enumerate(zip(colors, misses, strict=True)):
    for seed, value, offset in zip((42, 43, 44), group, (-0.15, 0, 0.15), strict=True):
        right.scatter(index + offset, value, color=color, s=46, zorder=3)
        right.annotate(str(seed), (index + offset, value), xytext=(0, 8),
                       textcoords="offset points", ha="center", fontsize=7)
right.set(title="Missed defects per training seed", ylabel="False negatives out of 28 defects",
          yticks=(0, 1, 2), ylim=(-0.23, 2.15))
for axis in (left, right):
    axis.set_xticks(range(3), labels)
    axis.set_xlim(-0.55, 2.55)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
fig.savefig(root / "figures/vgg16-study-one.png", dpi=180, facecolor="white")
