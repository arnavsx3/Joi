"""Plot saved training metrics (optional: uv run --with matplotlib)."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("checkpoints/v1/metrics.csv"))
    parser.add_argument("--out", type=Path, default=Path("runs/v1-loss.png"))
    parser.add_argument("--title", default="Joi · learning from random weights")
    parser.add_argument("--reference-step", type=int, help="Mark a previous experiment's final step")
    parser.add_argument("--min-step", type=int, default=0, help="Zoom in on later training updates")
    parser.add_argument("--unit",choices=("character","token"),default="character")
    args = parser.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        parser.error("Run with: uv run --with matplotlib python -m joi.plot")
    with args.metrics.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["step"]) >= args.min_step]
    if not rows:
        parser.error("Metrics file contains no evaluation rows in the selected range")
    steps = [int(row["step"]) for row in rows]
    train = [float(row["train_loss"]) for row in rows]
    validation = [float(row["val_loss"]) for row in rows]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        figure, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
        figure.set_facecolor("#faf9f6")
        axis.set_facecolor("#faf9f6")
        axis.plot(steps, train, color="#aa6526", label="Training", linewidth=2)
        axis.plot(steps, validation, color="#19766b", label="Validation", linewidth=2)
        if args.reference_step is not None:
            axis.axvline(args.reference_step, color="#777777", linestyle="--", linewidth=1,
                         label=f"Reference at {args.reference_step:,} updates")
        axis.set_title(args.title, loc="left", fontsize=15, pad=14)
        axis.set_xlabel("Training updates")
        axis.set_ylabel(f"Next-{args.unit} loss (nats; lower is better)")
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.99, 0.84))
        axis.text(0.98, 0.94, f"Validation: {validation[0]:.2f} → {validation[-1]:.2f}",
                  transform=axis.transAxes, ha="right", va="top", color="#19766b")
        figure.supxlabel("Fixed sampled windows from separate data splits · CPU",
                         fontsize=9, color="#555555")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.out, dpi=160)
        plt.close(figure)
    print(f"Saved loss curve to {args.out}")


if __name__ == "__main__":
    main()
