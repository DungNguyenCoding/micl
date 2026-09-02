from pathlib import Path
import pandas as pd

runs = {
    "proposed_pyro": Path("results/dense_priority_pyro_proposed_seed12025/metrics.csv"),
    "fedavg": Path("results/dense_priority_fedavg_seed12025/metrics.csv"),
}
rows = []
for name, path in runs.items():
    if not path.exists():
        print(f"MISSING: {name}: {path}")
        continue
    df = pd.read_csv(path)
    best = df.loc[df["accuracy"].idxmax()]
    final = df.iloc[-1]
    rows.append({
        "method": name,
        "best_accuracy": float(best["accuracy"]),
        "best_round": int(best["round"]),
        "final_accuracy": float(final["accuracy"]),
        "final_nll": float(final["nll"]),
        "final_ece": float(final["ece"]),
        "final_lr": float(final["learning_rate"]),
        "wall_time_sec": float(final["wall_time_sec"]),
        "posterior_mean_accuracy": (
            float(final["posterior_mean_accuracy"])
            if "posterior_mean_accuracy" in df.columns
            else float(final["accuracy"])
        ),
    })

if rows:
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
