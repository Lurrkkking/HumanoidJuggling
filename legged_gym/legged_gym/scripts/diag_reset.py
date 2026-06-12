#!/usr/bin/env python3
"""Diagnose WHY g1_stand falls — track base orientation and DOF tracking error."""

import isaacgym
from isaacgym import gymtorch, gymapi, gymutil
import torch
import numpy as np
from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args

args = get_args()
args.task = "g1_stand"
args.headless = True

env_cfg, _ = task_registry.get_cfgs(name=args.task)
env_cfg.env.num_envs = 1
env_cfg.noise.add_noise = False
env_cfg.domain_rand.randomize_initial_joint_pos = False
env_cfg.domain_rand.push_robots = False

env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

print(f"gravity thresh: 0.8")
print(f"default_joint_angles (legs only):")
for name in ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll']:
    for side in ['left', 'right']:
        j = f'{side}_{name}_joint'
        if j in env.cfg.init_state.default_joint_angles:
            print(f"  {j}: {env.cfg.init_state.default_joint_angles[j]:.2f}")
print(f"action_scale: {env.cfg.control.action_scale}")
print(f"dt: {env.dt:.4f}s")
print()

# Track over time
STEPS = 120
rolls = np.zeros(STEPS)
pitches = np.zeros(STEPS)
gxy = np.zeros(STEPS)
dof_pos_err = np.zeros((STEPS, env.num_dof))

obs = env.get_observations()
for step in range(STEPS):
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    obs, _, _, _, _ = env.step(actions)

    rolls[step] = env.roll[0].item()
    pitches[step] = env.pitch[0].item()
    gxy[step] = torch.norm(env.projected_gravity[0, :2]).item()
    dof_pos_err[step] = (env.dof_pos[0] - env.default_dof_pos[0]).abs().cpu().numpy()

    if env.reset_buf[0]:
        print(f"[RESET] step {step}: roll={np.degrees(rolls[step]):.1f}° pitch={np.degrees(pitches[step]):.1f}° gxy={gxy[step]:.3f}")

# Find which DOFs drifted most before reset
print(f"\nBase orientation over time (first 40 steps):")
for step in [0, 5, 10, 15, 20, 25, 30, 35, 40]:
    if step < STEPS:
        print(f"  step {step:3d}: roll={np.degrees(rolls[step]):6.1f}°  pitch={np.degrees(pitches[step]):6.1f}°  gxy={gxy[step]:.4f}")

print(f"\nDOF position tracking error (|dof_pos - default|), average over first 40 steps:")
print(f"  (joints with largest drift are likely the source of instability)")
avg_err = dof_pos_err[:40].mean(axis=0)
top10 = np.argsort(-avg_err)[:10]
for i, idx in enumerate(top10):
    name = env.dof_names[idx]
    print(f"  {i+1:2d}. {name:35s}  avg_err={avg_err[idx]:.3f} rad")

# Check: which direction is the fall?
if rolls[STEPS-1] > 0.1:
    print(f"\n⚠ Robot is falling to the RIGHT (roll positive, {np.degrees(rolls[STEPS-1]):.1f}°)")
elif rolls[STEPS-1] < -0.1:
    print(f"\n⚠ Robot is falling to the LEFT (roll negative, {np.degrees(rolls[STEPS-1]):.1f}°)")
if pitches[STEPS-1] > 0.1:
    print(f"⚠ Robot is falling FORWARD (pitch positive, {np.degrees(pitches[STEPS-1]):.1f}°)")
elif pitches[STEPS-1] < -0.1:
    print(f"⚠ Robot is falling BACKWARD (pitch negative, {np.degrees(pitches[STEPS-1]):.1f}°)")
