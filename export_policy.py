"""
Export a PPO checkpoint (state dict) to a deployable TorchScript policy.

The training checkpoint stores raw module state dicts:

    {"cfg", "actor", "critic", "actor_opt", "critic_opt",
     "actor_rms", "critic_rms"}

torch.jit.load cannot open that. Deployment needs a single traced module that
takes a RAW observation vector and returns the DETERMINISTIC action, with
observation normalization baked in — at deploy time there is no
RunningMeanStd to call, and feeding unnormalized observations to a policy
trained on normalized ones produces confident nonsense.

This script also writes a spec file recording exactly which observation terms
in which order the deployment must reproduce, and the joint index map.
Checkpoints whose configuration has to be recovered by inspecting
mean_net.0.weight.shape are how you end up deploying the wrong thing.

The joint map is NOT hardcoded. Every deploy.yaml in unitree_rl_mjlab
(velocity/v0, mimic/dance1_subject2, mimic/multiclip) has
joint_ids_map = [0..28], i.e. identity: mjlab joint order already matches the
Unitree SDK motor order for the 29-DoF G1. A non-identity map belongs to
unitree_rl_lab, which is IsaacLab-based and orders joints differently. Pass
--deploy-yaml to take the map from the file that ships with the policy.

Usage:
    python export_policy.py ckpt.pt policy.pt --deploy-yaml deploy.yaml \
        --obs-terms motion_command:58 motion_anchor_ori_b:6 base_ang_vel:3 \
        joint_pos_rel:29 joint_vel_rel:29 last_action:29
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml


def make_mlp(in_dim: int, hidden: tuple, out_dim: int, activation: str = "elu"):
    """Same topology as training. Init doesn't matter — weights get overwritten
    — but the module structure must match for load_state_dict to succeed."""
    act_cls = {"elu": nn.ELU, "tanh": nn.Tanh}[activation.lower()]
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), act_cls()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class DeployPolicy(nn.Module):
    """Raw obs in, deterministic action out.

    Two things are folded in that live outside the actor during training:
      - observation normalization (was RunningMeanStd, called by PPOModel)
      - taking the distribution mean instead of sampling (was Actor.act)
    """

    def __init__(self, mean_net: nn.Sequential, mean: torch.Tensor,
                 var: torch.Tensor):
        super().__init__()
        self.mean_net = mean_net
        # float32 buffers: RunningMeanStd accumulates in float64, which
        # TorchScript will happily keep and then silently upcast the input.
        self.register_buffer("obs_mean", mean.float())
        self.register_buffer("obs_std", torch.sqrt(var.float() + 1e-8))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean_net((obs - self.obs_mean) / self.obs_std)


def infer_hidden(actor_sd: dict) -> tuple:
    """Recover hidden widths from the state dict rather than trusting a flag."""
    dims = []
    i = 0
    while f"mean_net.{i}.weight" in actor_sd:
        dims.append(actor_sd[f"mean_net.{i}.weight"].shape[0])
        i += 2  # Linear, activation, Linear, ...
    return tuple(dims[:-1])  # last entry is the output head


def resolve_joint_map(deploy_yaml: str | None, act_dim: int) -> list:
    """Take the map from the yaml that ships with the policy, or fall back to
    identity. Validate either way — a truncated or wrong-robot map produces a
    spec that indexes out of range at deploy time, with no error until then."""
    if not deploy_yaml:
        print("no --deploy-yaml given; assuming identity joint order (mjlab). "
              "IsaacLab-trained policies need a permutation here.")
        return list(range(act_dim))

    cfg = yaml.safe_load(open(deploy_yaml))
    jmap = cfg.get("joint_ids_map")
    if jmap is None:
        raise SystemExit(f"{deploy_yaml} has no joint_ids_map")
    if sorted(jmap) != list(range(act_dim)):
        raise SystemExit(
            f"joint_ids_map from {deploy_yaml} has {len(jmap)} entries and is "
            f"not a permutation of 0..{act_dim - 1} — wrong robot or a "
            f"truncated file")
    if jmap != list(range(act_dim)):
        print(f"non-identity joint_ids_map from {deploy_yaml} — correct only "
              f"if this policy was NOT trained in mjlab")
    return list(jmap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("output")
    ap.add_argument("--obs-terms", nargs="*", default=[],
                    help="name:width pairs, in observation order")
    ap.add_argument("--step-dt", type=float, default=0.02)
    ap.add_argument("--action-scale", type=float, default=0.25)
    ap.add_argument("--deploy-yaml",
                    help="read joint_ids_map, step_dt and per-joint action "
                         "scale/offset from the yaml that ships with the policy")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor_sd = ck["actor"]

    obs_dim = actor_sd["mean_net.0.weight"].shape[1]
    act_dim = actor_sd["log_std"].shape[0]
    hidden = infer_hidden(actor_sd)
    print(f"obs_dim={obs_dim}  act_dim={act_dim}  hidden={hidden}")

    mean_net = make_mlp(obs_dim, hidden, act_dim)
    mean_net.load_state_dict(
        {k[len("mean_net."):]: v for k, v in actor_sd.items()
         if k.startswith("mean_net.")}
    )
    mean_net.eval()

    rms = ck.get("actor_rms")
    if rms is None:
        print("WARNING: no actor_rms in checkpoint — assuming identity "
              "normalization. If the run used normalize_obs=True this is wrong.")
        mean = torch.zeros(obs_dim, dtype=torch.float64)
        var = torch.ones(obs_dim, dtype=torch.float64)
    else:
        mean, var = rms["mean"], rms["var"]
        print(f"obs_rms count={float(rms['count']):.0f}  "
              f"mean|max|={mean.abs().max():.3f}  var range="
              f"[{var.min():.4f}, {var.max():.4f}]")
        if mean.shape[0] != obs_dim:
            raise SystemExit(
                f"actor_rms is {mean.shape[0]}-dim but the actor takes "
                f"{obs_dim} — mismatched checkpoint pieces")

    # Resolve and validate everything the spec needs BEFORE writing any
    # file. A late SystemExit would otherwise leave a policy.pt on disk
    # with no spec beside it — or overwrite a known-good export.
    jmap = resolve_joint_map(args.deploy_yaml, act_dim)

    step_dt, action_scale, action_offset = args.step_dt, args.action_scale, None
    if args.deploy_yaml:
        cfg = yaml.safe_load(open(args.deploy_yaml))
        step_dt = float(cfg.get("step_dt", args.step_dt))
        act = cfg.get("actions", {}).get("JointPositionAction", {})
        if act.get("scale") is not None:
            action_scale = list(act["scale"])
        if act.get("offset") is not None:
            action_offset = list(act["offset"])
        for name, v in (("scale", action_scale), ("offset", action_offset)):
            if isinstance(v, list) and len(v) != act_dim:
                raise SystemExit(
                    f"action {name} in {args.deploy_yaml} has {len(v)} entries, "
                    f"policy outputs {act_dim}")

    policy = DeployPolicy(mean_net, mean, var).eval()

    # Verify the traced module against the eager one before writing it.
    example = torch.randn(1, obs_dim)
    with torch.no_grad():
        eager = policy(example)
    traced = torch.jit.trace(policy, example)
    traced = torch.jit.freeze(traced)
    with torch.no_grad():
        got = traced(example)
    if not torch.allclose(eager, got, atol=1e-6):
        raise SystemExit("traced output diverges from eager — refusing to write")

    # Determinism check: the whole point of exporting the mean.
    with torch.no_grad():
        if not torch.allclose(traced(example), traced(example)):
            raise SystemExit("traced policy is not deterministic")

    traced.save(args.output)
    print(f"wrote {args.output}")

    # Spec: what the deployment must reproduce.
    terms = []
    total = 0
    for t in args.obs_terms:
        name, width = t.rsplit(":", 1)
        terms.append({"name": name, "width": int(width)})
        total += int(width)
    if terms and total != obs_dim:
        print(f"WARNING: --obs-terms sum to {total} but the actor takes "
              f"{obs_dim}. The spec is wrong, the policy is not.")

    spec = {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden": list(hidden),
        "obs_terms": terms,
        "obs_normalization": "baked into the exported module",
        "step_dt": step_dt,
        "action_scale": action_scale,
        "action_offset": action_offset,
        "joint_ids_map": jmap,
        "source_deploy_yaml": args.deploy_yaml,
        "note": "action is in mjlab joint order; motor_cmd[joint_ids_map[j]] "
                "is the SDK slot for mjlab joint j",
    }
    spec_path = Path(args.output).with_suffix(".spec.json")
    spec_path.write_text(json.dumps(spec, indent=2))
    print(f"wrote {spec_path}")


if __name__ == "__main__":
    main()