"""
plot_field_streamlines.py
=========================

Streamlines of the perfect circulation field (which spiral onto the ideal
orbit), with a dashed circle marking the target radius R. The streamlines are
coloured by the model's commanded output at each point, evaluated at the
perfect-circulation heading -- forward U = (u1+u2)/2 by default.

Usage
-----
    python plot_field_streamlines.py --model models/network_rl_torch.csv --R 10
    python plot_field_streamlines.py --color turn --density 1.4 --out stream.png
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
    ap.add_argument("--R", type=float, default=10.0, help="ideal orbit radius (dashed circle)")
    ap.add_argument("--lim", type=float, default=25.0, help="plot half-extent")
    ap.add_argument("--grid", type=int, default=200, help="grid resolution for streamlines")
    ap.add_argument("--density", type=float, default=1.3, help="streamline density")
    ap.add_argument("--color", choices=("forward", "turn", "l2", "l1", "speed"), default="forward",
                    help="line colour: forward=(u1+u2)/2, turn=(u1-u2)/2, "
                         "l2=sqrt(u1^2+u2^2), l1=|u1|+|u2|, speed=|phi| field magnitude")
    ap.add_argument("--out", default="field_streamlines.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    model = NeuralNetwork(args.model)
    L, n = args.lim, args.grid
    xs = np.linspace(-L, L, n)
    ys = np.linspace(-L, L, n)
    X, Y = np.meshgrid(xs, ys)

    # circulation field (drives the streamlines)
    Ux, Uy = phi(X, Y, args.R)

    # scalar field for colouring
    if args.color == "speed":
        C = np.hypot(Ux, Uy)
    else:
        # query the model on a coarser grid (forward() is a python loop) and
        # interpolate up to the streamline grid, to keep this fast
        m = min(n, 60)
        cx = np.linspace(-L, L, m)
        cy = np.linspace(-L, L, m)
        Cc = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                vx, vy = phi(cx[j], cy[i], args.R)
                th = math.atan2(vy, vx)
                u1, u2 = model.forward([float(cx[j]), float(cy[i]),
                                        math.cos(th), math.sin(th)])
                if args.color == "forward":
                    Cc[i, j] = (u1 + u2) / 2.0
                elif args.color == "turn":
                    Cc[i, j] = (u1 - u2) / 2.0
                elif args.color == "l2":
                    Cc[i, j] = math.hypot(u1, u2)
                else:  # l1
                    Cc[i, j] = abs(u1) + abs(u2)
        # bilinear interpolation onto the fine grid
        ix = np.clip(np.searchsorted(cx, X) - 1, 0, m - 2)
        iy = np.clip(np.searchsorted(cy, Y) - 1, 0, m - 2)
        tx = (X - cx[ix]) / (cx[ix + 1] - cx[ix])
        ty = (Y - cy[iy]) / (cy[iy + 1] - cy[iy])
        C = ((1-tx)*(1-ty)*Cc[iy, ix] + tx*(1-ty)*Cc[iy, ix+1]
             + (1-tx)*ty*Cc[iy+1, ix] + tx*ty*Cc[iy+1, ix+1])

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.set_aspect("equal")
    ax.set_xlim(-L, L); ax.set_ylim(-L, L)

    diverging = args.color == "turn"
    kw = dict(density=args.density, linewidth=1.1, arrowsize=1.0)
    if diverging:
        vmax = float(np.abs(C).max())
        strm = ax.streamplot(X, Y, Ux, Uy, color=C, cmap="coolwarm",
                             norm=plt.Normalize(-vmax, vmax), **kw)
    else:
        strm = ax.streamplot(X, Y, Ux, Uy, color=C, cmap="viridis", **kw)

    labels = {"forward": r"forward $U=(u_1+u_2)/2$", "turn": r"turn $(u_1-u_2)/2$",
              "l2": r"$\sqrt{u_1^2+u_2^2}$", "l1": r"$|u_1|+|u_2|$",
              "speed": r"field speed $\|\varphi\|$"}
    cb = fig.colorbar(strm.lines, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(labels[args.color])

    # dashed circle at the ideal orbit radius + buoy
    ax.add_artist(plt.Circle((0, 0), args.R, fill=False, ls="--",
                             color="crimson", lw=2.0, zorder=5))
    ax.plot(0, 0, "*", color="crimson", ms=15, zorder=6)

    ax.set_title(f"Circulation streamlines, coloured by {args.color}\n{args.model}")
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()