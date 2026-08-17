from mjlab.tasks.registry import register_mjlab_task
from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfg import MultiClipSettings, make_multiclip_cfg

# Reuse tuned PPO hyperparameters from the existing G1 tracking task 
from src.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

# Where the merged clip buffer and its split file live. Override on the CLI
# with --env.commands.motion.motion-file / .split-file.
MOTION_FILE = "/opt/nb/johan/data/motion_file/phuma_track_v2.npz"
SPLIT_FILE = ""

# binds kw = {"split": "train", "lookahead_frame":"ref"}
# base start with two paths 
# update merges you overrieds in 
# MultiClipSettings unpacks the dict back out into keyword arguments
def _settings(**kw) -> MultiClipSettings:
    base = dict(motion_file=MOTION_FILE, split_file=SPLIT_FILE)
    base.update(kw)
    return MultiClipSettings(**base)

# No root state terms in actor group, and the lookahead measured from the reference anchor rather than the robot's world position
# This is the one that ca reach hardware
register_mjlab_task(
    task_id="Unitree-G1-Tracking-MultiClip-No-State-Estimation",
    # motion_file: merged npz
    # split_file -> a JSON of clip indices 
    # split="train" -> which key to read 
    env_cfg=make_multiclip_cfg(
        _settings(split="train", lookahead_frame="ref"), 
        has_state_estimation=False, 
    ),
    play_env_cfg=make_multiclip_cfg(
        _settings(split="test", eval=True, lookahead_frame="ref"), 
        has_state_estimation=False, 
        play=True,
    ),
    rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# This is then with privileged root state in the actor group and the lookahead measured from the robot's root
register_mjlab_task(
  task_id="Unitree-G1-Tracking-MultiClip",
  env_cfg=make_multiclip_cfg(
    _settings(split="train", lookahead_frame="root"),
    has_state_estimation=True,
  ),
  play_env_cfg=make_multiclip_cfg(
    _settings(split="test", eval=True, lookahead_frame="root"),
    has_state_estimation=True,
    play=True,
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)


# 154 dims: drops ref_root_vel and ref_lookahead so the observation matches
# the six terms their C++ controller already implements. No custom C++ needed.
register_mjlab_task(
  task_id="Unitree-G1-Tracking-MultiClip-Deploy",
  env_cfg=make_multiclip_cfg(
    _settings(split="train", use_ref_vel=False, use_lookahead=False),
    has_state_estimation=False,
  ),
  play_env_cfg=make_multiclip_cfg(
    _settings(split="test", eval=True, use_ref_vel=False, use_lookahead=False),
    has_state_estimation=False, play=True,
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
