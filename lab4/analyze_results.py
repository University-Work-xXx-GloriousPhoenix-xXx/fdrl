"""
КП4: аналіз результатів експериментів без Jupyter (дублює відповідні комірки ноутбука).

Читає results/<config>_s<seed>/{final.json, evaluations.npz, monitor.monitor.csv} і створює:
  cw4_summary.csv, fig_eval_curves.png, fig_train_curves.png, fig_final.png

Запуск (з папки lab4):  python analyze_results.py
Потрібні лише numpy, pandas, matplotlib (torch/SB3 не потрібні).
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")  # без вікон: підходить для запуску з IDE і з консолі
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cw4_experiments as X  # лише константи CONFIGS, SEEDS, N_TIMESTEPS (важкі імпорти — всередині run_one)

ROOT = "results"
GROUPS = {
    "Крок навчання": ["lr_1e-4", "base", "lr_3e-3"],
    "Цільова мережа": ["target_10", "base", "target_2000"],
    "Архітектура": ["net_16", "net_64x64", "base"],
    "Дослідження": ["base", "explore_0.5", "eps_end_0.01"],
}
LABELS = {"base": "базова", "lr_1e-4": "lr=1e-4", "lr_3e-3": "lr=3e-3", "target_10": "target=10",
          "target_2000": "target=2000", "net_16": "[16]", "net_64x64": "[64,64]",
          "explore_0.5": "expl.frac=0.5", "eps_end_0.01": "ε_final=0.01"}


def load_runs():
    runs = {}
    for path in sorted(glob.glob(f"{ROOT}/*/final.json")):
        d = os.path.dirname(path)
        with open(path) as f:
            fin = json.load(f)
        ev = np.load(os.path.join(d, "evaluations.npz"))
        mon = pd.read_csv(os.path.join(d, "monitor.monitor.csv"), skiprows=1)
        runs[(fin["config"], fin["seed"])] = dict(final=fin, eval_t=ev["timesteps"],
                                                  eval_r=ev["results"].mean(1), mon=mon)
    return runs


def make_summary(runs):
    rows = []
    for cfg in X.CONFIGS:
        seeds = [s for s in X.SEEDS if (cfg, s) in runs]
        if not seeds:
            continue
        finals = [np.mean(runs[(cfg, s)]["final"]["final_returns"]) for s in seeds]
        succ = [np.mean(np.array(runs[(cfg, s)]["final"]["final_lengths"]) < 500) for s in seeds]
        t_solve = []  # перший крок, на якому оцінка перевищила -100
        for s in seeds:
            r = runs[(cfg, s)]
            idx = np.flatnonzero(r["eval_r"] > -100)
            t_solve.append(r["eval_t"][idx[0]] if len(idx) else np.nan)
        tail = [runs[(cfg, s)]["eval_r"][-5:] for s in seeds]  # остання 15 000 кроків
        rows.append({"конфігурація": LABELS[cfg], "cfg": cfg, "запусків": len(seeds),
                     "фін. дохід (сер. за seed)": np.mean(finals), "std між seed": np.std(finals),
                     "найгірший seed": np.min(finals), "успішних епізодів": np.mean(succ),
                     "крок досягнення −100": np.nanmean(t_solve) if not np.all(np.isnan(t_solve)) else np.nan,
                     "seed, що досягли −100": int(np.sum(~np.isnan(t_solve))),
                     "коливання оцінки (ост. 15k)": np.mean([np.std(t) for t in tail]),
                     "час, с": np.mean([runs[(cfg, s)]["final"]["train_time_s"] for s in seeds]),
                     "параметрів": runs[(cfg, seeds[0])]["final"]["q_net_params"]})
    summary = pd.DataFrame(rows)
    summary.to_csv("cw4_summary.csv", index=False)
    return summary


def plot_eval_curves(runs):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (title, cfgs) in zip(axes.ravel(), GROUPS.items()):
        for i, cfg in enumerate(cfgs):
            seeds = [s for s in X.SEEDS if (cfg, s) in runs]
            if not seeds:
                continue
            t = runs[(cfg, seeds[0])]["eval_t"]
            R = np.array([runs[(cfg, s)]["eval_r"][:len(t)] for s in seeds])
            ax.plot(t, R.mean(0), lw=2, color=f"C{i}", label=LABELS[cfg])
            ax.fill_between(t, R.min(0), R.max(0), color=f"C{i}", alpha=0.15)
        ax.axhline(-100, color="k", ls=":", lw=1)
        ax.set_title(title); ax.set_xlabel("кроки середовища"); ax.set_ylabel("оцінка: сер. дохід (10 еп.)")
        ax.set_ylim(-510, -50); ax.grid(alpha=.3); ax.legend(fontsize=9, loc="lower right")
    plt.suptitle("Криві детермінованого оцінювання: середнє за 3 seed (лінія) та мін–макс (тінь); пунктир — поріг −100",
                 fontweight="bold")
    plt.tight_layout(); plt.savefig("fig_eval_curves.png", dpi=130, bbox_inches="tight"); plt.close()


def smooth(x, w=20):
    x = np.asarray(x, float); c = np.r_[0, np.cumsum(x)]; k = np.arange(1, len(x) + 1)
    return (c[k] - c[np.maximum(k - w, 0)]) / (k - np.maximum(k - w, 0))


def plot_train_curves(runs):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    for ax, (g, cfgs) in zip(axes.ravel(), GROUPS.items()):
        for i, cfg in enumerate(cfgs):
            for s in X.SEEDS:
                if (cfg, s) not in runs:
                    continue
                mon = runs[(cfg, s)]["mon"]
                ax.plot(np.cumsum(mon.l), smooth(mon.r), color=f"C{i}", alpha=0.7, lw=1,
                        label=LABELS[cfg] if s == 0 else None)
        ax.set_title(f"{g}: навчальні епізоди (ε-жадібна поведінка)"); ax.set_xlabel("кроки середовища")
        ax.set_ylabel("дохід епізоду (ковзне сер. 20)"); ax.set_ylim(-510, -50)
        ax.grid(alpha=.3); ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout(); plt.savefig("fig_train_curves.png", dpi=130, bbox_inches="tight"); plt.close()


def plot_final(runs):
    fig, ax = plt.subplots(figsize=(13, 4.2))
    order = list(dict.fromkeys(c for g in GROUPS.values() for c in g))
    for i, cfg in enumerate(order):
        for s in X.SEEDS:
            if (cfg, s) in runs:
                rr = runs[(cfg, s)]["final"]["final_returns"]
                ax.scatter(i + (s - 1) * 0.18, np.mean(rr), color=f"C{s}", s=40, label=f"seed {s}" if i == 0 else None)
                ax.vlines(i + (s - 1) * 0.18, np.percentile(rr, 10), np.percentile(rr, 90), color=f"C{s}", alpha=.5)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([LABELS[c] for c in order])
    ax.axhline(-100, color="k", ls=":", lw=1); ax.set_ylabel("фінальний дохід (50 еп.)")
    ax.set_title("Фінальне оцінювання кожного запуску: середнє (точка) та 10–90 перцентилі (лінія)")
    ax.grid(alpha=.3); ax.legend(); plt.tight_layout(); plt.savefig("fig_final.png", dpi=130, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    runs = load_runs()
    print("Завантажено запусків:", len(runs))
    if not runs:
        raise SystemExit("Немає results/*/final.json — спочатку запустіть: python cw4_experiments.py")
    summary = make_summary(runs)
    print(summary.drop(columns="cfg").round(2).to_string(index=False))
    plot_eval_curves(runs); plot_train_curves(runs); plot_final(runs)
    print("Збережено: cw4_summary.csv, fig_eval_curves.png, fig_train_curves.png, fig_final.png")
