"""
train_rl_pytorch.py
===================

Reinforcement learning with PyTorch for the DDBoat "carousel" task: make the
boat orbit a buoy (at the origin) at a fixed distance R.

Method: differentiable simulation / analytic policy gradient (BPTT).
--------------------------------------------------------------------
The boat dynamics `f` and the Euler integration are smooth, differentiable
operations, and so is the orbit cost. So we don't need a black-box optimiser
(CMA-ES) or a high-variance estimator (REINFORCE/PPO): we re-implement the
rollout in torch, run a *batch* of boats in parallel, accumulate the cost, and
let autograd backprop the episode cost straight through the simulator into the
network weights. Adam then does the rest. This is far more sample-efficient
than CMA-ES *because* we have an analytic differentiable model of the boat.

(If the real boat dynamics were unknown or non-differentiable you'd switch to a
model-free method like PPO; here the model is known and smooth, so BPTT wins.)

* Policy: the repo's `neural_net_pytorch_.NeuralNetwork` (4->8->8->2, no bias),
  architecture UNCHANGED (ReLU hidden, linear output). The output is clamped to
  the action range a in [0, 1] and the simulation scales it to physical thrust
  u = u_scale * a. The clamp + scale are environment logic (mirror them in
  Unity); the network's layers and activations are identical to the original.
* Warm start: load the imitation-trained CSV
  (`models/network_v1_100000_4_20_10_100000.csv`) into the torch model.
* Export: `model.register_to_csv(out)` -> drops straight into
  `simple_simulator_with_neural_net.py` and `neural_net.NeuralNetwork`.

Usage
-----
    python train_rl_pytorch.py
    python train_rl_pytorch.py --R 10 --iters 400 --horizon 400 --lr 1e-3
    python train_rl_pytorch.py --scratch          # random init instead of warm start
"""

import argparse
import csv
import os
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from neural_net_pytorch_ import NeuralNetwork

# both NumPy (MKL) and PyTorch ship their own libiomp5md.dll, and Windows refuses to load it twice. It's not a bug in your code or the simulator.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Boat physical parameters (identical to simple_simulator.py's p1..p7, which are
# only defined under __main__ there; redefined here so the module is self-contained).
P1, P2, P3, P4, P5, P6, P7 = 0.07, 2200.0, 3.0e-05, 15.0e-05, 0.4, 5.0, 5.0


# --------------------------------------------------------------------------- #
# Differentiable dynamics: batched version of simple_simulator.f
# x: (B, 8) = [px, py, theta, vx, vy, w, w1, w2]   u: (B, 2) = [u1, u2]
# --------------------------------------------------------------------------- #
def boat_deriv(x, u):
    theta, vx, vy, w = x[:, 2], x[:, 3], x[:, 4], x[:, 5]
    w1, w2 = x[:, 6], x[:, 7]
    u1, u2 = u[:, 0], u[:, 1]

    dx = vx * torch.cos(theta) - vy * torch.sin(theta)     # world-frame velocity
    dy = vx * torch.sin(theta) + vy * torch.cos(theta)
    dtheta = w

    dvx = w * vy - P5 * vx * torch.abs(vx) + P3 * (w1 * torch.abs(w1) + w2 * torch.abs(w2))
    dvy = -w * vx - P6 * vy * torch.abs(vy)
    dw = -P7 * w * torch.abs(w) + P4 * (w2 * torch.abs(w2) - w1 * torch.abs(w1))

    dw1 = -P1 * w1 * torch.abs(w1) + P2 * u1
    dw2 = -P1 * w2 * torch.abs(w2) + P2 * u2

    return torch.stack([dx, dy, dtheta, dvx, dvy, dw, dw1, dw2], dim=1)


def net_input(x):
    """[px, py, cos(theta), sin(theta)] for the policy, batched: (B, 4)."""
    return torch.stack([x[:, 0], x[:, 1], torch.cos(x[:, 2]), torch.sin(x[:, 2])], dim=1)


# --------------------------------------------------------------------------- #
# Orbit cost (see notes): radius-keeping - tangential-progress + radial-damping
# (+ optional control effort). Returns mean over the batch at one step.
# --------------------------------------------------------------------------- #
def step_cost(x, xdot, a, R, direction, w_radius, w_tang, w_radial, w_ctrl):
    px, py = x[:, 0], x[:, 1]
    r = torch.sqrt(px * px + py * py + 1e-9)
    dx, dy = xdot[:, 0], xdot[:, 1]
    v_r = (px * dx + py * dy) / r                       # radial speed
    v_t = (px * dy - py * dx) / r                       # tangential speed (CCW +)

    c = w_radius * (r - R) ** 2
    c = c - w_tang * direction * v_t
    c = c + w_radial * v_r ** 2
    if w_ctrl:
        c = c + w_ctrl * ((a - 0.5) ** 2).sum(dim=1)   # keep actions mid-range (off the clamp)
    return c.mean()


# --------------------------------------------------------------------------- #
# Differentiable rollout. The network is UNCHANGED (ReLU hidden, linear output);
# we only clamp its output to the action range a in [0, 1] (the value Unity
# sees) and the SIMULATION scales it to physical thrust u = u_scale * a. The
# clamp + scale are environment logic, not network layers, so the model's
# layers and activations stay identical to the original. Returns mean per-step cost.
# --------------------------------------------------------------------------- #
def rollout_cost(model, x0, R, dt, horizon, direction,
                 w_radius, w_tang, w_radial, w_ctrl, u_scale):
    x = x0
    total = x.new_zeros(())
    for _ in range(horizon):
        a = model(net_input(x)).clamp(0.0, 1.0)        # action in [0, 1]
        u = u_scale * a                                # scale to physical thrust
        xdot = boat_deriv(x, u)
        x = x + dt * xdot                              # Euler, same as the simulator
        total = total + step_cost(x, xdot, a, R, direction,
                                  w_radius, w_tang, w_radial, w_ctrl)
    return total / horizon


def sample_init_states(batch, R, device, generator):
    ang = torch.rand(batch, generator=generator, device=device) * 2 * np.pi
    rad = (0.4 + 1.2 * torch.rand(batch, generator=generator, device=device)) * R  # [0.4R,1.6R]
    px, py = rad * torch.cos(ang), rad * torch.sin(ang)
    heading = torch.rand(batch, generator=generator, device=device) * 2 * np.pi
    vx0 = 5.0 + 10.0 * torch.rand(batch, generator=generator, device=device)        # ~10 like the demo
    z = torch.zeros(batch, device=device)
    o = torch.ones(batch, device=device)
    return torch.stack([px, py, heading, vx0, z, z, o, o], dim=1)


# --------------------------------------------------------------------------- #
# Warm-start: load the repo CSV (transposed flat vector) into the torch model.
# Inverse of NeuralNetwork.register_to_csv.
# --------------------------------------------------------------------------- #
def load_params_from_csv(model, path, out_rescale=1.0):
    """Load repo CSV weights. If out_rescale != 1, divide the final layer's
    weights by it -- used to convert an old raw-thrust model (output ~u_bar)
    into the new bounded-action convention (output ~u_bar/u_scale, near tanh's
    linear region) so the warm start reproduces roughly the same thrust."""
    with open(path, "r") as fh:
        flat = np.array([float(v) for v in next(csv.reader(fh))], dtype=np.float32)
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    idx = 0
    with torch.no_grad():
        for k, layer in enumerate(linears):
            out_f, in_f = layer.weight.shape                # torch stores (out, in)
            n = in_f * out_f
            block = flat[idx:idx + n].reshape(in_f, out_f)  # CSV stored (in, out)
            w = torch.from_numpy(block.T.copy())
            if k == len(linears) - 1 and out_rescale != 1.0:
                w = w / out_rescale                         # final layer only
            layer.weight.copy_(w)
            idx += n
    if idx != flat.size:
        raise ValueError(f"CSV had {flat.size} params, model used {idx}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warm-start", default="models/network_v1_100000_4_20_10_100000.csv",
                    help="CSV of the imitation-trained network to start from (default)")
    ap.add_argument("--resume", nargs="?", const="__OUT__", default=None, metavar="CSV",
                    help="continue training from already-trained weights. With no value, "
                         "resumes from --out; or give a CSV path. Overrides --warm-start.")
    ap.add_argument("--scratch", action="store_true", help="random init instead of warm start")
    ap.add_argument("--out", default="models/network_rl_torch.csv")
    ap.add_argument("--R", type=float, default=10.0, help="target orbit radius")
    ap.add_argument("--direction", type=float, default=+1.0, choices=(+1.0, -1.0),
                    help="+1 CCW (matches classic controller), -1 CW")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64, help="boats simulated in parallel")
    ap.add_argument("--horizon", type=int, default=400, help="Euler steps per episode")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--w-radius", type=float, default=1.0)
    ap.add_argument("--w-tangential", type=float, default=2.0)
    ap.add_argument("--w-radial", type=float, default=0.5)
    ap.add_argument("--w-ctrl", type=float, default=0.0,
                    help="penalty on (a-0.5)^2 (keeps the action mid-range, off the clamp)")
    ap.add_argument("--u-scale", type=float, default=40.0,
                    help="physical thrust = u_scale * action, action in [0,1]; "
                         "default 40 puts cruise (~20) at action 0.5")
    ap.add_argument("--no-warm-rescale", action="store_true",
                    help="do NOT divide the warm-start output layer by u_scale "
                         "(use if the warm-start model is already in the [-1,1] convention)")
    ap.add_argument("--eval-every", type=int, default=5,
                    help="evaluate on a fixed held-out batch every N iters for checkpointing")
    ap.add_argument("--score-plot", default="training_score.png",
                    help="where to save the score-vs-iteration plot")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(args.seed)

    model = NeuralNetwork().to(device)
    if args.scratch:
        print("Random initialisation.")
    elif args.resume is not None:
        # continue training from existing (already-trained) weights -- already in
        # the bounded-action convention, so no output rescale.
        resume_path = args.out if args.resume == "__OUT__" else args.resume
        if not os.path.exists(resume_path):
            raise SystemExit(f"--resume: weights file not found: {resume_path}")
        load_params_from_csv(model, resume_path)
        print(f"Resuming from existing weights: {resume_path}")
    else:
        # imitation model outputs raw thrust (~u_bar); convert to the [0,1]
        # action convention by dividing the output layer by u_scale (so cruise
        # ~u_bar maps to action ~u_bar/u_scale, mid-range).
        rescale = 1.0 if args.no_warm_rescale else args.u_scale
        load_params_from_csv(model, args.warm_start, out_rescale=rescale)
        print(f"Warm-started from {args.warm_start} (output rescaled by 1/{rescale:g})")

    cost_kw = dict(R=args.R, dt=args.dt, horizon=args.horizon, direction=args.direction,
                   w_radius=args.w_radius, w_tang=args.w_tangential,
                   w_radial=args.w_radial, w_ctrl=args.w_ctrl, u_scale=args.u_scale)

    # fixed held-out batch of starts, used to score the policy for checkpointing
    eval_gen = torch.Generator(device=device).manual_seed(args.seed + 999)
    x_eval = sample_init_states(args.batch, args.R, device, eval_gen)

    def evaluate():
        with torch.no_grad():
            return rollout_cost(model, x_eval, **cost_kw).item()

    base = evaluate()
    print(f"Loaded-policy baseline cost (held-out): {base:.4f}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # start best at the loaded policy's score: we only overwrite --out if we
    # actually beat the weights we started from (safe to resume repeatedly).
    best = base
    hist_it, hist_train, hist_eval = [], [], []
    for it in range(args.iters):
        x0 = sample_init_states(args.batch, args.R, device, gen)
        loss = rollout_cost(model, x0, **cost_kw)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if it % args.eval_every == 0 or it == args.iters - 1:
            val = evaluate()                               # held-out score
            improved = val < best
            if improved:
                best = val
                model.register_to_csv(args.out)            # checkpoint best
            hist_it.append(it); hist_train.append(loss.item()); hist_eval.append(val)
            print(f"iter {it:4d} | train {loss.item():10.4f} | eval {val:10.4f} "
                  f"| best {best:10.4f}{'  <- saved' if improved else ''}")

    print(f"\nDone. Best held-out cost {best:.4f} (started from {base:.4f}).")
    print(f"Saved best policy to {args.out}")

    # --- plot the score after training ------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist_it, hist_train, color="tab:gray", alpha=0.6, label="train (random batch)")
    ax.plot(hist_it, hist_eval, color="tab:blue", lw=2, marker="o", ms=3,
            label="eval (held-out)")
    ax.axhline(base, color="tab:red", ls="--", lw=1, label=f"start ({base:.2f})")
    ax.axhline(best, color="tab:green", ls="--", lw=1, label=f"best ({best:.2f})")
    ax.set_xlabel("iteration"); ax.set_ylabel("cost (lower is better)")
    ax.set_title("Training score")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(args.score_plot, dpi=150, bbox_inches="tight")
    print(f"Saved score plot to {args.score_plot}")

    print("Visualise with simple_simulator_with_neural_net.py "
          f"(remember: action a = clamp(output, 0, 1), physical thrust = {args.u_scale:g} * a).")


if __name__ == "__main__":
    main()