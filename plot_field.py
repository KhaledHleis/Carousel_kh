"""
plot_field_circulation.py
=========================

Static quiver where, at every grid point, the boat heading is set to the
PERFECT CIRCULATION direction (the analytic phi field), the arrow points along
that circulation, and the colour is the norm of the model's commanded output
||(u1, u2)|| when queried at that matched heading.

Interpretation: "if the boat were perfectly oriented along the ideal circle at
every point, how much thrust does the learned controller command, and where?"
With a good controller the heading correction is ~0, so this is essentially a
map of commanded effort over the plane.

Usage
-----
    python plot_field_circulation.py --model models/network_rl_torch.csv --R 10
    python plot_field_circulation.py --norm l1 --out effort.png
"""

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

from neural_net import NeuralNetwork


def phi_0(a, b):
    return -a**3 - a*b**2 + a - b, -b**3 - b*a**2 + a + b


def phi(x1, x2, R):
    """Perfect circulation field (same as the simulator's)."""
    z1, z2 = x1 / R, x2 / R
    w1, w2 = phi_0(z1, z2)
    return R * w1, R * w2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/network_rl_torch.csv")
    ap.add_argument("--R", type=float, default=10.0, help="orbit radius defining the circulation")
    ap.add_argument("--lim", type=float, default=25.0, help="plot half-extent")
    ap.add_argument("--grid", type=int, default=30, help="arrows per axis")
    ap.add_argument("--norm", choices=("l2", "l1"), default="l2",
                    help="how to combine (u1,u2): l2 = sqrt(u1^2+u2^2), l1 = |u1|+|u2|")
    ap.add_argument("--out", default="field_circulation.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    model = NeuralNetwork(args.model)
    L, n = args.lim, args.grid
    X, Y = np.meshgrid(np.linspace(-L, L, n), np.linspace(-L, L, n))

    Ux = np.zeros_like(X)        # arrow direction = perfect circulation (unit)
    Uy = np.zeros_like(X)
    C = np.zeros_like(X)         # colour = ||model output||
    for i in range(n):
        for j in range(n):
            x, y = float(X[i, j]), float(Y[i, j])
            vx, vy = phi(x, y, args.R)
            mag = math.hypot(vx, vy) + 1e-9
            theta = math.atan2(vy, vx)                 # perfect-circulation heading
            Ux[i, j], Uy[i, j] = vx / mag, vy / mag
            u1, u2 = model.forward([x, y, math.cos(theta), math.sin(theta)])
            C[i, j] = (math.hypot(u1, u2) if args.norm == "l2"
                       else abs(u1) + abs(u2))

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.set_aspect("equal")
    ax.set_xlim(-L, L); ax.set_ylim(-L, L)

    q = ax.quiver(X, Y, Ux, Uy, C, cmap="viridis", pivot="mid", width=0.004)
    cb = fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04)
    label = r"$\sqrt{u_1^2+u_2^2}$" if args.norm == "l2" else r"$|u_1|+|u_2|$"
    cb.set_label(f"model output norm  {label}")

    ax.add_artist(plt.Circle((0, 0), args.R, fill=False, ls="--", color="white", lw=1.5))
    ax.plot(0, 0, "w*", ms=14, markeredgecolor="black")

    ax.set_title(f"Perfect-circulation field, coloured by model output norm\n{args.model}")
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")
    print(f"output norm: min {C.min():.2f}  max {C.max():.2f}  mean {C.mean():.2f}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()