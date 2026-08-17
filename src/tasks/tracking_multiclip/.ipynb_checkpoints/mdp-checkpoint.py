"""
Multi-clip motion tracking MDP terms

Ported from the training notebook into a registrable mjlab task.
"""


from __future__ import annotations
 
import json
from dataclasses import dataclass
from pathlib import Path
 
import numpy as np
import torch
 
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
# mjlab 1.2.0 renamed quat_rotate_inv -> quat_apply_inverse.
# Same signature (quat, vec); aliased so the call sites below are unchanged.
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse as quat_rotate_inv,
  quat_inv,
  quat_mul,
  yaw_quat,
)
 
 
def quat_to_6d(q: torch.Tensor) -> torch.Tensor:
  """(N, 4) wxyz -> (N, 6): the first two columns of the rotation matrix.
 
  Flattened ROW-major, so identity reads [1, 0, 0, 1, 0, 0]. Verified against
  the checkpoint normalizer, whose mean peaks at indices 0 and 3.
  """
  w, x, y, z = q.unbind(-1)
  return torch.stack([
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
    2 * (x * z - w * y), 2 * (y * z + w * x),
  ], dim=-1)


# termination
def clip_timeout(env, command_name: str="motion") -> torch.Tensor: 
  """
  End the episode when the current clip runs out. 

  The command holds at the clip's last frame rather than resampling, 
  so without this the reference would freeze while the episode continued. 

  critic learns V(s) and trained on consecutive transitions and when there's no done. It bootstraps
  """
  cmd = env.command_manager.get_term(command_name)
  return cmd.time_steps >= cmd.clip_end

# Observation terms
def obs_ref_root_vel(env, command_name: str="motion") -> torch.Tensor: 
  return env.command_manager.get_term(command_name).ref_root_vel_b()

def obs_ref_lookahead(env, command_name: str="motion") -> torch.Tensor: 
  return env.command_manager.get_term(command_name).lookahead_obs()

# command

class MultiMotionCommand(MotionCommand):
  """
  MotionCommand over a merged multi-clip buffer.
  """

  cfg: "MultiMotionCommandCfg"

  def __init__(self, cfg, env):
    super().__init__(cfg, env)

    with np.load(cfg.motion_file) as d:
      if "clip_starts" not in d:
        raise ValueError(
          f"{cfg.motion_file} has no clip_starts -- run merge_motions first")
      starts = np.asarray(d["clip_starts"], dtype=np.int64)
      lengths = np.asarray(d["clip_lengths"], dtype=np.int64)
      names = [str(n) for n in np.asarray(d["clip_names"])]

    keep = np.arange(len(lengths))
    if cfg.split_file is not None:
      split = json.loads(Path(cfg.split_file).read_text())
      keep = np.asarray(split[cfg.split], dtype=np.int64)
      if keep.size == 0:
        raise ValueError(f"split '{cfg.split}' is empty")

    # keep only clips long enough to be worth sampling
    keep = keep[lengths[keep] >= max(2, cfg.min_clip_frames)]
    if keep.size == 0:
      raise ValueError(f"no clip has >= {cfg.min_clip_frames} frames")

    self.clip_starts = torch.as_tensor(starts[keep], device=self.device)
    self.clip_lengths = torch.as_tensor(lengths[keep], device=self.device)
    self.clip_names = [names[i] for i in keep]
    self.num_clips = int(keep.size)      # used by _clip_weights below

    # Length-proportional by default; uniform over clips over-weights short ones
    self._clip_weights = (
      torch.ones(self.num_clips, device=self.device)
      if cfg.sample_uniform_over_clips
      else self.clip_lengths.float())

    self.env_clip = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._det_ptr = 0

    # Envs sampled during super().__init__ took the fallback path; redo them
    # now that the clip tables exist.
    self._uniform_sampling(torch.arange(self.num_envs, device=self.device))

    print(f"[MultiMotionCommand] {self.num_clips} clips, "
          f"{int(self.clip_lengths.sum()):,} frames, split={cfg.split}")

  # views

  @property
  def clip_begin(self):
    return self.clip_starts[self.env_clip]

  @property
  def clip_end(self):
    """Last valid row of the clip each env is on."""
    return self.clip_begin + self.clip_lengths[self.env_clip] - 1

  @property
  def clip_phase(self):
    """Progress within the current clip, in [0, 1]"""
    span = (self.clip_lengths[self.env_clip] - 1).clamp(min=1)
    return (self.time_steps - self.clip_begin).float() / span.float()

  # Sampling

  def _uniform_sampling(self, env_ids):
    """
    Draw a (clip, phase) per env. Called by the base _resample_command
    BEFORE it reads time_steps back to teleport the robot onto the reference.
    """
    if not hasattr(self, "clip_starts"):
      return super()._uniform_sampling(env_ids)  # during super().__init__

    n = int(env_ids.numel())
    if n == 0:
      return

    # deterministic clips for exhaustive evaluation
    if self.cfg.deterministic_clips:
      clip = (self._det_ptr + torch.arange(n, device=self.device)) % self.num_clips
      self._det_ptr = int((self._det_ptr + n) % self.num_clips)
    else:
      clip = torch.multinomial(self._clip_weights, n, replacement=True)
    self.env_clip[env_ids] = clip

    if self.cfg.random_start_phase:
      span = (self.clip_lengths[clip] - self.cfg.min_remaining_frames).clamp(min=1)
      phase = (torch.rand(n, device=self.device) * span.float()).long()
    else:
      phase = torch.zeros(n, dtype=torch.long, device=self.device)

    self.time_steps[env_ids] = self.clip_starts[clip] + phase

    self.metrics["sampling_entropy"][:] = 1.0
    self.metrics["sampling_top1_prob"][:] = 1.0 / max(self.num_clips, 1)
    self.metrics["sampling_top1_bin"][:] = 0.5

  # Update

  def _update_command(self, env_ids=None):
    """Advance time, then hold at the clip's last frame.

    The base resamples when time_steps >= motion.time_step_total, which on a
    merged buffer is the END OF THE WHOLE CONCATENATION, not of the current
    clip. Clamping per clip is what keeps one env on one motion; the
    clip_timeout termination then ends the episode cleanly, so the critic
    gets a done instead of a teleporting reference.

    mjlab 1.2.0 inlined update_relative_body_poses() into the base
    _update_command, so the anchor-relative pose block below is reproduced
    here. It must run AFTER the clamp: computing it from an unclamped
    time_steps would read the next clip's frames for one step at every clip
    boundary.
    """
    if env_ids is None:
      self.time_steps += 1
    else:
      self.time_steps[env_ids] += 1
    # In place: the base class and the viewer scrubber both write through
    # self.time_steps, so keep the same tensor object.
    self.time_steps.clamp_(max=self.clip_end)

    n_bodies = len(self.cfg.body_names)
    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, n_bodies, 1)
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, n_bodies, 1)
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, n_bodies, 1)
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, n_bodies, 1)

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

  # extra observations

  def ref_root_vel_b(self) -> torch.Tensor:
    """(N, 6) reference anchor lin+ang velocity, in the ROBOT's base frame.

    Without it the policy only sees accumulated position error, i.e. it learns
    the reference moved AFTER falling behind. Base frame keeps it
    heading-invariant. Uses only the root QUATERNION, which the IMU provides
    on hardware.
    """
    q = self.robot.data.root_link_quat_w
    return torch.cat([
      quat_rotate_inv(q, self.anchor_lin_vel_w),
      quat_rotate_inv(q, self.anchor_ang_vel_w),
    ], dim=-1)

  def lookahead_obs(self) -> torch.Tensor:
    """
    (N, 38 * len(cfg.lookahead)) upcoming reference frames.

    Per offset k: anchor position offset (3) + anchor orientation 6D (6)
                + reference joint targets (29) = 38

    cfg.lookahead_frame decides what the position offset is measured from:

      "root" -- from the robot's current root position. Needs root_link_pos_w,
                which NO G1 SENSOR PROVIDES. Fine in simulation, blocks
                deployment.
      "ref"  -- from the reference anchor at the CURRENT frame. Pure reference
                data, no robot state, so it survives to hardware and carries
                the same information.

    time_steps + k is clamped at clip_end, so the last frame repeats instead
    of bleeding into the next clip in the merged buffer.
    """
    if not self.cfg.lookahead:
      return torch.zeros(self.num_envs, 0, device=self.device)

    ai = self.motion_anchor_body_index
    q = self.robot.data.root_link_quat_w

    if self.cfg.lookahead_frame == "root":
      origin = self.robot.data.root_link_pos_w
    elif self.cfg.lookahead_frame == "ref":
      origin = self.motion.body_pos_w[self.time_steps, ai]
    else:
      raise ValueError(f"lookahead_frame must be 'root' or 'ref', "
                       f"got {self.cfg.lookahead_frame!r}")

    out = []
    for k in self.cfg.lookahead:
      t = torch.minimum(self.time_steps + k, self.clip_end)
      out += [
        quat_rotate_inv(q, self.motion.body_pos_w[t, ai] - origin),
        quat_to_6d(self.motion.body_quat_w[t, ai]),
        self.motion.joint_pos[t],
      ]
    return torch.cat(out, dim=-1)


@dataclass(kw_only=True)
class MultiMotionCommandCfg(MotionCommandCfg):
  split_file: str | None = None      # .split.json from merge_motions; None = all
  split: str = "train"
  min_clip_frames: int = 50          # 1 s at 50 Hz
  min_remaining_frames: int = 25     # frames left after the sampled start phase
  sample_uniform_over_clips: bool = False
  random_start_phase: bool = True
  deterministic_clips: bool = False  # env e -> clip (ptr+e) % K; exhaustive eval
  lookahead: tuple[int, ...] = (5, 10, 20)   # frames ahead: 0.1 / 0.2 / 0.4 s
  lookahead_frame: str = "ref"       # "ref" is deployable, "root" is not
 
  def build(self, env):
    # The base cfg hardcodes MotionCommand here -- this override is required.
    if self.sampling_mode != "uniform":
      raise ValueError(
        f"sampling_mode must be 'uniform' for multi-clip, got "
        f"{self.sampling_mode!r}. Adaptive bins span clip boundaries.")
    return MultiMotionCommand(self, env)