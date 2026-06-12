#!/usr/bin/env python3
"""
check_g1_default_pose.py — Verify G1 reset correctness and foot-ground contact.

Key questions:
1. After reset_idx + refresh, is root_z == init_state.pos[2]?
2. Which bodies are actually at ground level (z ~ 0)?
3. Do we have stale tensors?
"""

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

# ── Don't use make_env (it calls env.reset() → step() internally) ──
# Build env manually to inspect pre-reset state
from legged_gym.utils.helpers import class_to_dict, set_seed, parse_sim_params
set_seed(env_cfg.seed)
sim_params = {"sim": class_to_dict(env_cfg.sim)}
sim_params = parse_sim_params(args, sim_params)

env = LeggedRobot(cfg=env_cfg, sim_params=sim_params, physics_engine=args.physics_engine,
                  sim_device=args.sim_device, headless=args.headless)

# env.__init__ creates sim, creates envs, calls _init_buffers().
# After __init__: sim is prepared, tensors are wrapped, but NO reset has been called yet.
# The actor was created at env_origin (z=0).

print("=" * 70)
print("A. POST-__INIT__ STATE (actor created, no reset, no step)")
print("=" * 70)

env.gym.refresh_actor_root_state_tensor(env.sim)
env.gym.refresh_rigid_body_state_tensor(env.sim)
env.gym.refresh_dof_state_tensor(env.sim)
env.gym.refresh_net_contact_force_tensor(env.sim)

eid = 0
print(f"root_states[{eid}] = {env.root_states[eid].tolist()}")
print(f"  root z = {env.root_states[eid, 2].item():.4f}")
print(f"init_state.pos[2] = {env_cfg.init_state.pos[2]}")
print(f"base_init_state = {env.base_init_state.tolist()}")
print(f"env_origins[{eid}] = {env.env_origins[eid].tolist()}")

# Get body names
robot_asset = env.gym.get_actor_asset(env.envs[eid], env.actor_handles[eid])
body_names = env.gym.get_asset_rigid_body_names(robot_asset)
body_idx = {n: i for i, n in enumerate(body_names)}

# Print all body z positions
body_z = [(n, env.rigid_body_states[eid, i, 2].item()) for i, n in enumerate(body_names)]
body_z.sort(key=lambda x: x[1])
print(f"\nBody z positions (lowest first) — BEFORE reset:")
for n, z in body_z[:15]:
    marker = " ← FOOT" if 'ankle_roll' in n.lower() else ""
    marker = " ← FEET_INDEX" if body_idx[n] in env.feet_indices.tolist() else marker
    print(f"  {n:35s}  z={z:7.4f}{marker}")

# Check foot indices
print(f"\nfeet_indices: {env.feet_indices.tolist()}")
for fi in env.feet_indices:
    bname = body_names[fi.item()]
    bz = env.rigid_body_states[eid, fi.item(), 2].item()
    print(f"  feet_indices[{fi.item()}] = '{bname}'  z={bz:.4f}")

# ── Now call reset_idx manually ──
print("\n" + "=" * 70)
print("B. AFTER reset_idx([0]) — before any simulate")
print("=" * 70)

env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
env._reset_root_states(env_ids)
env._reset_dofs(env_ids)

# IMMEDIATELY refresh tensors
env.gym.refresh_actor_root_state_tensor(env.sim)
env.gym.refresh_rigid_body_state_tensor(env.sim)
env.gym.refresh_dof_state_tensor(env.sim)

print(f"root_states[{eid}] after reset: {env.root_states[eid].tolist()}")
print(f"  root z = {env.root_states[eid, 2].item():.4f}")
print(f"  expected root z = {env_cfg.init_state.pos[2]} + {env.env_origins[eid, 2].item():.3f} = {env_cfg.init_state.pos[2] + env.env_origins[eid, 2].item():.3f}")

body_z = [(n, env.rigid_body_states[eid, i, 2].item()) for i, n in enumerate(body_names)]
body_z.sort(key=lambda x: x[1])
print(f"\nBody z positions (lowest first) — AFTER reset, BEFORE simulate:")
for n, z in body_z[:15]:
    marker = " ← FOOT" if 'ankle_roll' in n.lower() else ""
    marker = " ← FEET_INDEX" if body_idx[n] in env.feet_indices.tolist() else marker
    print(f"  {n:35s}  z={z:7.4f}{marker}")

# Check: does root_z match expectation?
actual_rz = env.root_states[eid, 2].item()
expected_rz = env_cfg.init_state.pos[2] + env.env_origins[eid, 2].item()
if abs(actual_rz - expected_rz) > 0.01:
    print(f"\n⚠ MISMATCH: root_z={actual_rz:.4f} but expected={expected_rz:.4f}")
else:
    print(f"\n✅ root_z matches expected: {actual_rz:.4f}")

# ── Now simulate 1 policy step (= 4 physics steps) ──
print("\n" + "=" * 70)
print("C. AFTER 1 simulate step (4 physics substeps, dt=0.005 each)")
print("=" * 70)

zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
for _ in range(env.cfg.control.decimation):
    torques = env._compute_torques(zero_actions).view(env.torques.shape)
    env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(torques))
    env.gym.simulate(env.sim)
    if env.device == 'cpu':
        env.gym.fetch_results(env.sim, True)
    env.gym.refresh_dof_state_tensor(env.sim)

# Refresh all tensors after simulate
env.gym.refresh_actor_root_state_tensor(env.sim)
env.gym.refresh_rigid_body_state_tensor(env.sim)
env.gym.refresh_net_contact_force_tensor(env.sim)

print(f"root_states[{eid}] after 1 step: {env.root_states[eid].tolist()}")
print(f"  root z = {env.root_states[eid, 2].item():.4f}")
print(f"  root lin_vel = {env.root_states[eid, 7:10].tolist()}")
print(f"  root ang_vel = {env.root_states[eid, 10:13].tolist()}")

body_z = [(n, env.rigid_body_states[eid, i, 2].item()) for i, n in enumerate(body_names)]
body_z.sort(key=lambda x: x[1])
print(f"\nBody z positions (lowest first) — after 1 simulate step:")
for n, z in body_z[:15]:
    f_norm = torch.norm(env.contact_forces[eid, body_idx[n]]).item()
    marker = ""
    if 'ankle_roll' in n.lower():
        marker = f" ← FOOT F={f_norm:.1f}N"
    elif 'ankle_pitch' in n.lower():
        marker = f" F={f_norm:.1f}N"
    elif f_norm > 0.1:
        marker = f" F={f_norm:.1f}N"
    print(f"  {n:35s}  z={z:7.4f}{marker}")

min_z = body_z[0][1]
min_body = body_z[0][0]
print(f"\nMin body z: {min_body} at z={min_z:.4f}")
if min_z < -0.01:
    print(f"  ⚠ PENETRATION: {min_body} is {abs(min_z):.3f}m below ground")
elif min_z < 0.01:
    print(f"  ✅ {min_body} touches ground (z≈{min_z:.4f})")
else:
    print(f"  ⚠ NO GROUND CONTACT — lowest body at z={min_z:.4f}")

# ── Simulate more steps, track root_z ──
print("\n" + "=" * 70)
print("D. TRACKING root_z over 100 more steps")
print("=" * 70)

for step in range(100):
    for _ in range(env.cfg.control.decimation):
        torques = env._compute_torques(zero_actions).view(env.torques.shape)
        env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(torques))
        env.gym.simulate(env.sim)
        if env.device == 'cpu':
            env.gym.fetch_results(env.sim, True)
        env.gym.refresh_dof_state_tensor(env.sim)

    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)

    if step % 10 == 0 or step == 99:
        rz = env.root_states[eid, 2].item()
        # compute pitch from base quat
        base_quat = env.root_states[eid, 3:7]
        qw = base_quat[3]; qx = base_quat[0]; qy = base_quat[1]; qz = base_quat[2]
        # pitch = asin(2*(qw*qy - qz*qx))
        sinp = 2 * (qw * qy - qz * qx)
        pitch = np.degrees(np.arcsin(torch.clamp(sinp, -1, 1).item()))
        # projected gravity xy: from quat
        # g_vec = (0,0,-1) rotated by inverse quat
        # projected_gravity = R^T * (0,0,-1)
        gx = 2 * (qx * qz - qw * qy)
        gy = 2 * (qy * qz + qw * qx)
        gxy_val = np.sqrt(gx.item()**2 + gy.item()**2)

        foot_z_vals = [env.rigid_body_states[eid, fi.item(), 2].item() for fi in env.feet_indices]
        foot_names = [body_names[fi.item()] for fi in env.feet_indices]
        foot_str = " ".join([f"{n}={z:.3f}" for n, z in zip(foot_names, foot_z_vals)])

        min_body_z = min(env.rigid_body_states[eid, :, 2]).item()
        min_body_idx = env.rigid_body_states[eid, :, 2].argmin().item()
        min_body_name = body_names[min_body_idx]

        print(f"  step {step+1:3d}: root_z={rz:.4f}  pitch={pitch:6.1f}°  gxy={gxy_val:.4f}  feet=({foot_str})  min_z={min_body_z:.4f} ({min_body_name})")

# ── Summary ──
print("\n" + "=" * 70)
print("E. SUMMARY")
print("=" * 70)
rz_final = env.root_states[eid, 2].item()
print(f"Final root_z = {rz_final:.4f} (expected ~0.78-0.80 for stable stand)")
print(f"Init pos z   = {env_cfg.init_state.pos[2]}")

if abs(rz_final - env_cfg.init_state.pos[2]) > 0.1:
    print(f"⚠ Robot settled {abs(rz_final - env_cfg.init_state.pos[2]):.2f}m below spawn height")
    print(f"  → This is a GROUND PENETRATION / SETTLING issue, not a sensor error")
    print(f"  → G1 default pose sinks {env_cfg.init_state.pos[2] - rz_final:.2f}m into ground")
