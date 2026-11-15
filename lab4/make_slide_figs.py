import json, glob, os
import numpy as np, matplotlib.pyplot as plt
import cw4_experiments as X
LAB = {"base": "базова", "lr_1e-4": "lr=1e-4", "lr_3e-3": "lr=3e-3", "target_10": "target=10",
       "target_2000": "target=2000", "net_16": "[16]", "net_64x64": "[64,64]",
       "explore_0.5": "expl.frac=0.5", "eps_end_0.01": "ε_final=0.01"}
order = ["base", "lr_1e-4", "lr_3e-3", "target_10", "target_2000", "net_64x64", "net_16", "explore_0.5", "eps_end_0.01"]
fin = {}; ev = {}
for p in glob.glob("results/*/final.json"):
    f = json.load(open(p)); fin[(f["config"], f["seed"])] = np.mean(f["final_returns"])
    e = np.load(os.path.join(os.path.dirname(p), "evaluations.npz")); ev[(f["config"], f["seed"])] = (e["timesteps"], e["results"].mean(1))
plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(figsize=(5, 6.2))
for i, c in enumerate(order[::-1]):
    vals = [fin[(c, s)] for s in X.SEEDS]
    col = "#2a9d8f" if np.mean(vals) > -100 else ("#e9c46a" if np.mean(vals) > -300 else "#e76f51")
    ax.barh(i, np.mean(vals) + 500, left=-500, color=col, alpha=0.85)
    ax.scatter(vals, [i] * 3, color="k", s=14, zorder=3)
ax.set_yticks(range(len(order))); ax.set_yticklabels([LAB[c] for c in order[::-1]])
ax.axvline(-100, color="k", ls=":", lw=1); ax.set_xlim(-510, -50)
ax.set_xlabel("фінальний дохід (50 еп.)"); ax.set_title("Середнє за 3 seed (стовпчик)\nта окремі seed (точки)", fontsize=12)
ax.grid(axis="x", alpha=.3); plt.tight_layout(); plt.savefig("fig_slide_bars.png", dpi=170); plt.close()

fig, axes = plt.subplots(2, 1, figsize=(5, 6.4), sharex=True)
for ax, cfgs, title in [(axes[0], ["target_10", "base", "target_2000"], "Період оновлення цільової мережі"),
                        (axes[1], ["net_16", "net_64x64", "base"], "Архітектура мережі")]:
    for i, c in enumerate(cfgs):
        t = ev[(c, 0)][0]; R = np.array([ev[(c, s)][1] for s in X.SEEDS])
        ax.plot(t / 1000, R.mean(0), lw=2, color=f"C{i}", label=LAB[c]); ax.fill_between(t / 1000, R.min(0), R.max(0), color=f"C{i}", alpha=.15)
    ax.axhline(-100, color="k", ls=":", lw=1); ax.set_ylim(-510, -50); ax.set_title(title, fontsize=12); ax.grid(alpha=.3)
    ax.legend(fontsize=9, loc="lower right"); ax.set_ylabel("оцінка")
axes[1].set_xlabel("тис. кроків"); plt.tight_layout(); plt.savefig("fig_slide_curves.png", dpi=170); plt.close()
print("ok")
