import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nets = [json.load(open(f"decode_{m:03d}.json")) for m in range(5)]
san = json.load(open("sanity_results.json"))
rows = [("Photoreceptors", "brightness"), ("Lamina", "brightness"), ("Medulla", "brightness"),
        ("Motion detectors", "brightness"), ("Motion detectors", "change"),
        ("Shuffled control", "brightness"), ("Mean-brightness baseline", "brightness")]
labels = ["Photoreceptors\n(R1–R6)", "Lamina\n(L1–L5)", "Medulla\n(Mi, Tm)", "T4/T5:\nbrightness",
          "T4/T5:\nchange", "Shuffled\ncontrol", "Mean-brightness\nbaseline"]
table = []
for (g, k), lab in zip(rows, labels):
    for split in ["train", "natural", "synthetic"]:
        vals = [n[split][g][k] for n in nets]
        table.append(dict(group=g, target=k, split=split, net000=vals[0], min=min(vals), max=max(vals)))
json.dump(table, open("score_table.json", "w"), indent=1)
for r in table:
    if r["split"] == "natural":
        print(f"{r['group']:26s} {r['target']:10s} net000 {r['net000']:6.3f}   range over 000-004 {r['min']:6.3f} to {r['max']:6.3f}")
still = [n["still"]["Motion detectors"]["brightness"] for n in nets]
print("still scenes, T4/T5 brightness corr per network:", [round(s, 2) for s in still])

fig, ax = plt.subplots(figsize=(10, 4.6))
xs = np.arange(len(rows))
for i, (g, k) in enumerate(rows):
    vals = [n["natural"][g][k] for n in nets]
    ax.plot([i, i], [min(vals), max(vals)], color="#9aa4b2", lw=6, solid_capstyle="round", zorder=1)
    ax.scatter([i] * 4, vals[1:], s=28, color="#5b6676", zorder=2, label="networks 001–004" if i == 0 else None)
    ax.scatter([i], [vals[0]], s=80, color="#e4572e", zorder=3, label="network 000 (best)" if i == 0 else None)
ax.axhline(0, color="#cccccc", lw=0.8)
ax.set_xticks(xs, labels, fontsize=9)
ax.set_ylabel("Correlation with the truth\n(per frame, averaged)", fontsize=9)
ax.set_ylim(-0.2, 1.05)
ax.set_title("What can be decoded from each part of the simulated fly visual system\n"
             "48 test clips from photos never used in training", fontsize=11)
ax.axvspan(4.5, 6.5, color="#f2f2f2", zorder=0)
ax.text(5.5, 1.0, "controls", ha="center", va="top", fontsize=9, color="#666666")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="lower left")
fig.tight_layout()
fig.savefig("decoder_scores.png", dpi=150)
print("saved decoder_scores.png")
