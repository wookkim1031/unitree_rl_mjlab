"""
Multi-clip tracking env configs 

Mirrors build_env_mulit from the notebook, but returns a CONFIG rather than a built env
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .mdp import(
    MultiMotionCommandCfg,
    clip_timeout,
    obs_ref_lookahead,
    obs_ref_root_vel,
)

from src.tasks.tracking.config.g1.env_cfgs import (
    unitree_g1_flat_tracking_env_cfg as BASE_CFG_FN,
)


@dataclass
class MultiClipSettings:
  """Everything the notebook set by hand, in one place so it lands in the
  env.yaml that train.py dumps beside every checkpoint."""
  motion_file: str = ""
  split_file: str = ""
  split: str = "train"
  episode_length_s: float = 10.0
  njmax: int = 512
  use_ref_vel: bool = True
  use_lookahead: bool = True
  lookahead: tuple[int, ...] = (5, 10, 20)
  lookahead_frame: str = "ref"          # "ref" deployable, "root" as trained
  w_root_pos: float = 1.0
  w_action_rate: float = -0.1
  ee_body_names: tuple[str, ...] = (
    "left_ankle_roll_link", "right_ankle_roll_link")
  eval: bool = False
  eval_disturbed: bool = False
 
 
def make_multiclip_cfg(
  s: MultiClipSettings = MultiClipSettings(),
  has_state_estimation: bool = False,
  play: bool = False,
):
  """Base tracking cfg -> multi-clip tracking cfg.
 
  has_state_estimation=False drops the root-state observation terms, which
  is what makes a checkpoint deployable. Combine with
  MultiClipSettings.lookahead_frame="ref" so the lookahead does not
  reintroduce the same dependency.
  """
  cfg = BASE_CFG_FN(has_state_estimation=has_state_estimation, play=play)
 
  cfg.episode_length_s = s.episode_length_s
  cfg.sim.njmax = s.njmax
 
  # Clip end terminates the episode; the command holds at the last frame.
  TermCfg = type(cfg.terminations["time_out"])
  cfg.terminations["clip_timeout"] = TermCfg(func=clip_timeout, time_out=True)
 
  # Wrists stay out of the end-effector termination: the retargeting emits no
  # wrist motion, so reference wrist poses sit far off on many resets and
  # ee_body_pos then dominates every termination.
  if "ee_body_pos" in cfg.terminations:
    cfg.terminations["ee_body_pos"].params["body_names"] = s.ee_body_names
 
  if s.eval:
    # A fall must not truncate the measurement; only clip_timeout ends it.
    for t in ("anchor_pos", "anchor_ori", "ee_body_pos"):
      cfg.terminations.pop(t, None)
    cfg.episode_length_s = 1e9
    if not s.eval_disturbed:
      for ev in ("push_robot", "base_com", "encoder_bias", "foot_friction"):
        cfg.events.pop(ev, None)
 
  base = cfg.commands["motion"]
  cfg.commands["motion"] = MultiMotionCommandCfg(
    **{**vars(base), "sampling_mode": "uniform"},
    split_file=s.split_file or None,
    split=s.split,
    min_remaining_frames=1 if s.eval else 100,
    random_start_phase=not s.eval,
    deterministic_clips=s.eval,
    lookahead=s.lookahead,
    lookahead_frame=s.lookahead_frame,
  )
  cfg.commands["motion"].motion_file = s.motion_file
  cfg.commands["motion"].resampling_time_range = (1e9, 1e9)
 
  cfg.rewards["motion_global_root_pos"].weight = s.w_root_pos
  cfg.rewards["action_rate_l2"].weight = s.w_action_rate
 
  if s.use_ref_vel or s.use_lookahead:
    ObsTerm = type(next(t for t in cfg.observations["actor"].terms.values()
                        if t is not None))
    for group in (cfg.observations["actor"], cfg.observations["critic"]):
      if s.use_ref_vel:
        group.terms["ref_root_vel"] = ObsTerm(func=obs_ref_root_vel)
      if s.use_lookahead:
        group.terms["ref_lookahead"] = ObsTerm(func=obs_ref_lookahead)
 
  return cfg
 