#!/usr/bin/env python3
"""
record_video.py — Headless video recording for clean legged_gym baseline.

Supports two modes:
  A. --zero_action : feed torch.zeros without loading a checkpoint
  B. default (policy) : load a PPO checkpoint and run policy(obs)

Usage:
  python legged_gym/scripts/record_video.py --task=g1_stand --zero_action --record_video --video_length=300
  python legged_gym/scripts/record_video.py --task=g1_stand --record_video --video_length=300 --checkpoint=-1
"""

import os
import sys
import subprocess
import tempfile
import numpy as np

# NOTE: isaacgym MUST be imported before torch
import isaacgym
from isaacgym import gymtorch, gymapi, gymutil
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, get_load_path
from legged_gym.utils.helpers import update_cfg_from_args
from legged_gym.envs.base.base_task import BaseTask


# ── monkey-patch: allow GPU rendering without viewer in headless mode ──
_original_base_init = BaseTask.__init__


def _patched_base_init(self, cfg, sim_params, physics_engine, sim_device, headless):
    """Same as BaseTask.__init__ but keeps GPU graphics context even when headless."""
    self.gym = gymapi.acquire_gym()
    self.sim_params = sim_params
    self.physics_engine = physics_engine
    self.sim_device = sim_device
    sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)

    if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
        self.device = self.sim_device
    else:
        self.device = 'cpu'

    # ── key patch: always use GPU for graphics when recording ──
    if headless:
        self.graphics_device_id = self.sim_device_id  # GPU rendering available
        self.headless = True
    else:
        self.graphics_device_id = self.sim_device_id
        self.headless = False

    self.num_envs = cfg.env.num_envs
    self.num_obs = cfg.env.num_observations
    self.num_privileged_obs = cfg.env.num_privileged_obs
    self.num_actions = cfg.env.num_actions
    self.num_one_step_obs = cfg.env.num_one_step_observations

    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)

    self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
    self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
    self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
    self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    if self.num_privileged_obs is not None:
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
    else:
        self.privileged_obs_buf = None

    self.extras = {}

    self.create_sim()
    self.gym.prepare_sim(self.sim)

    self.enable_viewer_sync = True
    self.viewer = None
    # NEVER create viewer in headless mode


# Apply the monkey-patch
BaseTask.__init__ = _patched_base_init


# ── Camera helpers ──

def setup_camera(env, env_handle, width=1280, height=720,
                 cam_pos=None, cam_target=None, ref_env_id=0, env_id=None):
    """Create an Isaac Gym camera sensor attached to env_handle."""
    camera_props = gymapi.CameraProperties()
    camera_props.width = width
    camera_props.height = height
    camera_props.enable_tensors = True

    cam_handle = env.gym.create_camera_sensor(env_handle, camera_props)

    if cam_pos is None:
        cam_pos = (2.5, 2.5, 1.2)
    if cam_target is None:
        cam_target = (0.0, 0.0, 0.8)

    if env_id is not None and ref_env_id != env_id:
        ref_origin = env.env_origins[ref_env_id]
        cur_origin = env.env_origins[env_id]
        dx = (cur_origin[0] - ref_origin[0]).item()
        dy = (cur_origin[1] - ref_origin[1]).item()
        cam_pos = (cam_pos[0] + dx, cam_pos[1] + dy, cam_pos[2])
        cam_target = (cam_target[0] + dx, cam_target[1] + dy, cam_target[2])

    env.gym.set_camera_location(
        cam_handle, env_handle,
        gymapi.Vec3(*cam_pos), gymapi.Vec3(*cam_target),
    )
    return cam_handle, width, height


def setup_overview_camera(env, env_handle, width=1280, height=720):
    """Position camera high + far back to show all envs in one frame."""
    origins = env.env_origins.cpu().numpy()
    centre_x = origins[:, 0].mean()
    centre_y = origins[:, 1].mean()
    extent_x = origins[:, 0].ptp()
    extent_y = origins[:, 1].ptp()

    print(f"[INFO] Overview camera: {env.num_envs} envs")
    print(f"       centre=({centre_x:.1f}, {centre_y:.1f})")
    for i in range(env.num_envs):
        print(f"       env {i}: ({origins[i, 0]:.1f}, {origins[i, 1]:.1f}, {origins[i, 2]:.1f})")

    grid_diag = np.sqrt(extent_x**2 + extent_y**2) + 3.0
    cam_z = grid_diag * 0.7 + 1.0

    cam_pos = (centre_x, centre_y - 1.0, cam_z)
    cam_target = (centre_x, centre_y, 0.4)

    print(f"       camera_pos=({cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f})")
    print(f"       camera_target=({cam_target[0]:.1f}, {cam_target[1]:.1f}, {cam_target[2]:.1f})")

    return setup_camera(env, env_handle, width, height, cam_pos, cam_target)


def capture_frame(env, cam_handle, cam_width, cam_height, camera_env_id=0):
    """Capture one RGB frame from the camera sensor. Returns numpy (H, W, 3) uint8."""
    env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    env.gym.start_access_image_tensors(env.sim)

    img = env.gym.get_camera_image(
        env.sim, env.envs[camera_env_id], cam_handle, gymapi.IMAGE_COLOR
    )
    img = np.frombuffer(img, dtype=np.uint8).copy()
    img = img.reshape(cam_height, cam_width, -1)
    img = img[..., :3]
    return img


def frames_to_mp4(frame_dir, video_path, fps=30):
    """Compose PNG frames into MP4 using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        video_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[INFO] Video saved to: {video_path}")


# ── Main play function ──

def play(args):
    print("[INFO] Starting record_video.py ...")
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # ── env config: minimal, quiet, clean ──
    env_cfg.env.num_envs = min(getattr(args, "num_envs", None) or 4, 16)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_initial_joint_pos = False
    env_cfg.domain_rand.push_robots = False

    headless = getattr(args, "headless", True)
    camera_env_id = getattr(args, "camera_env_id", 0)
    zero_action = getattr(args, "zero_action", False)

    # ── create env ──
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # ── load policy (skip if --zero_action) ──
    policy = None
    if not zero_action:
        train_cfg.runner.resume = True
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg
        )
        policy = ppo_runner.get_inference_policy(device=env.device)
        print("[INFO] Policy loaded from checkpoint")
    else:
        print("[INFO] Zero-action mode: no policy loaded")

    # ── video recording setup ──
    record_video = getattr(args, "record_video", False)
    overview_video = getattr(args, "overview_video", False)
    video_path = getattr(args, "video_path", None)
    video_length = getattr(args, "video_length", 1000)
    video_interval = getattr(args, "video_interval", 1)

    cam_handle = None
    frame_dir = None

    if record_video:
        mode = "overview" if overview_video else "single"
        print(f"[INFO] Recording {mode} video: {video_length} frames, env_id={camera_env_id}")
        frame_dir = tempfile.mkdtemp(prefix="legyd_video_frames_")
        print(f"[INFO] Frame dir: {frame_dir}")

        if overview_video:
            cam_handle, cam_width, cam_height = setup_overview_camera(env, env.envs[0])
        else:
            cam_handle, cam_width, cam_height = setup_camera(
                env, env.envs[camera_env_id], env_id=camera_env_id
            )

        # Step once to initialise rendering pipeline
        zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        obs, _, _, _, _ = env.step(zero_actions)

    # ── main play loop ──
    max_steps = video_length if record_video else 3000
    frame_idx = 0

    for step_count in range(max_steps):
        if zero_action:
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        else:
            actions = policy(obs.detach())

        obs, privileged_obs, rews, dones, infos = env.step(actions.detach())

        if record_video and step_count % video_interval == 0:
            try:
                img = capture_frame(env, cam_handle, cam_width, cam_height, camera_env_id)
                from PIL import Image
                Image.fromarray(img).save(
                    os.path.join(frame_dir, f"frame_{frame_idx:06d}.png")
                )
                frame_idx += 1
            except Exception as e:
                print(f"[WARN] Frame {frame_idx} capture failed: {e}")

    # ── compose MP4 ──
    if record_video and frame_idx > 0:
        if video_path is None:
            task_name = args.task
            mode = "zero" if zero_action else "policy"
            video_path = os.path.join(
                LEGGED_GYM_ROOT_DIR, "videos",
                f"{task_name}_{mode}_{frame_idx}f.mp4"
            )
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        frames_to_mp4(frame_dir, video_path, fps=50)
        print(f"[INFO] Total frames captured: {frame_idx}")
    elif record_video:
        print("[WARN] No frames captured!")


if __name__ == "__main__":
    custom_parameters = [
        {"name": "--task", "type": str, "default": "g1_stand", "help": "Task name."},
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume from checkpoint."},
        {"name": "--experiment_name", "type": str, "help": "Experiment name."},
        {"name": "--run_name", "type": str, "help": "Run name."},
        {"name": "--load_run", "type": str, "help": "Run to load when resume=True."},
        {"name": "--checkpoint", "type": int, "help": "Checkpoint number. -1 = latest."},
        {"name": "--headless", "action": "store_true", "default": True, "help": "Headless mode."},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": "RL device."},
        {"name": "--num_envs", "type": int, "help": "Number of environments (clamped to [1, 16])."},
        {"name": "--seed", "type": int, "help": "Random seed."},
        {"name": "--max_iterations", "type": int, "help": "Max training iterations."},
        # ── mode ──
        {"name": "--zero_action", "action": "store_true", "default": False,
         "help": "Feed zero actions instead of loading a policy checkpoint."},
        # ── video recording ──
        {"name": "--record_video", "action": "store_true", "default": False,
         "help": "Record video using offscreen camera."},
        {"name": "--overview_video", "action": "store_true", "default": False,
         "help": "Wide-angle overview showing all envs (overrides single close-up)."},
        {"name": "--camera_env_id", "type": int, "default": 0,
         "help": "Which env index to attach the camera to (default: 0)."},
        {"name": "--video_path", "type": str, "default": None,
         "help": "Output MP4 path."},
        {"name": "--video_length", "type": int, "default": 1000,
         "help": "Number of steps to record."},
        {"name": "--video_interval", "type": int, "default": 1,
         "help": "Record every N steps."},
    ]

    args = gymutil.parse_arguments(description="RecordVideo", custom_parameters=custom_parameters)
    args.sim_device = args.rl_device

    play(args)
