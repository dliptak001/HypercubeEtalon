"""Plot a few Raman training spectra + baselines from RamanSpectra."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"C:\HypercubeEtalon\RamanSpectra")
SPLIT = ROOT / "Training"
OUT = Path(__file__).with_name("sample_spectra.png")
INDICES = (0, 42, 250)


def load_row(path: Path) -> np.ndarray:
    return np.fromstring(path.read_text(encoding="ascii").strip(), sep=",", dtype=np.float64)


def main() -> None:
    x = load_row(SPLIT / "xaxis.txt")
    fig, axes = plt.subplots(len(INDICES), 1, sharex=True, figsize=(10, 8))
    fig.suptitle("Raman training patterns (spectrum + baseline)")

    for ax, idx in zip(axes, INDICES):
        spec = load_row(SPLIT / f"{idx}.data.txt")
        base = load_row(SPLIT / f"{idx}.label.txt")
        if spec.size != x.size or base.size != x.size:
            raise SystemExit(f"length mismatch at {idx}: x={x.size} data={spec.size} label={base.size}")
        ax.plot(x, spec, color="0.25", lw=0.9, label="spectrum")
        ax.plot(x, base, color="C3", lw=1.2, label="baseline")
        ax.set_ylabel("amplitude")
        ax.set_title(f"Training/{idx}")
        ax.legend(loc="upper right", frameon=False)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("wavenumber")
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
