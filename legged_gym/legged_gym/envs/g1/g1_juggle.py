"""
G1 Juggle task skeleton — G1 robot + ball actor + phase manager + contact classifier + AMP stub.

Inherits from the CLEAN LeggedRobot base (not the Goalkeeper one).
All juggle logic lives here in the subclass.
"""

import os
import copy
import numpy as np

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot, euler_from_quaternion
from legged_gym.envs.g1.g1_juggle_config import G1JuggleCfg
from legged_gym.utils.math import *
from legged_gym.utils.helpers import class_to_dict

# ── Phase constants ──
WAIT = 0
KICK_RIGHT = 1
KICK_LEFT = 2
RECOVER = 3
NUM_PHASES = 4


class G1Juggle(LeggedRobot):
    """G1 juggling environment skeleton.

    Extends the clean LeggedRobot with:
      - Ball actor (sphere, passively simulated)
      - 4-phase state machine (WAIT → KICK_RIGHT/KICK_LEFT → RECOVER → WAIT)
      - Contact classifier (right/left foot, hands, wrong bodies)
      - AMP observation interface (stub, not wired to PPO)
    """

    def __init__(self, cfg: G1JuggleCfg, sim_params, physics_engine, sim_device, headless):
        self._ball_asset_loaded = False
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # NOTE: cfg.ball.radius/mass are used for reset/contact logic only.
        # Actual ball physics (radius, mass, inertia) come from ball.urdf.
        # To change ball physics, edit ball.urdf AND update cfg.ball accordingly.
        if self._ball_asset_loaded:
            print(f"[G1Juggle] Ball URDF loaded. cfg.ball.radius={cfg.ball.radius:.3f}m, "
                  f"cfg.ball.mass={cfg.ball.mass:.3f}kg "
                  f"(verify ball.urdf matches these values)")

    # ═══════════════════════════════════════════════════════════════════════
    # Simulation setup — override _create_envs to add ball actor
    # ═══════════════════════════════════════════════════════════════════════

    def _create_envs(self):
        """Load robot + ball assets, create envs with two actors each."""
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # ── Shared asset options ──
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        # ── Load robot asset ──
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # ── Load ball asset (gravity enabled) ──
        ball_options = gymapi.AssetOptions()
        ball_options.disable_gravity = False
        ball_options.density = 100.0  # ignored when mass is set in URDF
        ball_path = self.cfg.asset.ballfile.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        ball_root = os.path.dirname(ball_path)
        ball_file = os.path.basename(ball_path)
        ball_asset = self.gym.load_asset(self.sim, ball_root, ball_file, ball_options)
        self._ball_asset_loaded = True

        # ── Body / joint name discovery ──
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dof = len(self.dof_names)

        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device)

        base_init_state_list = (
            self.cfg.init_state.pos + self.cfg.init_state.rot +
            self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel)
        self.base_init_state = to_torch(base_init_state_list, device=self.device)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.ball_handles = []
        self.envs = []

        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper,
                                             int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-0.3, 0.3, (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            # Robot actor
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
                self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)

            if i == 0:
                self.default_com = copy.deepcopy(body_props[0].com)
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass

            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True)
            self.actor_handles.append(actor_handle)

            # Ball actor — placed in front of robot
            bp = pos.clone()
            bp[0] += self.cfg.ball.init_pos_base[0]
            bp[1] += self.cfg.ball.init_pos_base[1]
            bp[2] = self.cfg.ball.init_pos_base[2]
            start_pose.p = gymapi.Vec3(*bp)
            ball_handle = self.gym.create_actor(
                env_handle, ball_asset, start_pose, "ball", i, 0, 1)
            # Color the ball orange
            color = gymapi.Vec3(1.0, 0.5, 0.0)
            self.gym.set_rigid_body_color(
                env_handle, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            self.ball_handles.append(ball_handle)

            self.envs.append(env_handle)

        # ── Standard body indices (feet, penalized, termination) ──
        self._prepare_joint_indices(
            body_names, feet_names, penalized_contact_names, termination_contact_names)

        # ── Contact classifier body indices ──
        self._prepare_contact_classifier_indices(body_names)

    def _prepare_contact_classifier_indices(self, body_names):
        """Build body ID tensors for semantic ball-body contact detection."""
        env0 = self.envs[0]
        actor0 = self.actor_handles[0]

        def _find_body(name):
            return self.gym.find_actor_rigid_body_handle(env0, actor0, name)

        # Contact foot links (ankle_roll_link)
        contact_foot_names = [s for s in body_names if self.cfg.asset.contact_foot_names in s]
        self.contact_feet_indices = torch.zeros(
            len(contact_foot_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(contact_foot_names):
            self.contact_feet_indices[i] = _find_body(name)

        # Right foot bodies
        right_foot_names = [s for s in body_names if "right" in s and self.cfg.asset.foot_name in s]
        self.right_foot_body_ids = torch.zeros(
            len(right_foot_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(right_foot_names):
            self.right_foot_body_ids[i] = _find_body(name)

        # Left foot bodies
        left_foot_names = [s for s in body_names if "left" in s and self.cfg.asset.foot_name in s]
        self.left_foot_body_ids = torch.zeros(
            len(left_foot_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(left_foot_names):
            self.left_foot_body_ids[i] = _find_body(name)

        # Hand bodies: rubber_hand + wrist yaw/pitch/roll links
        hand_name = getattr(self.cfg.asset, 'hand_name', 'hand')
        hand_names = [s for s in body_names if hand_name in s or 'wrist' in s]
        self.hand_body_ids = torch.zeros(
            len(hand_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(hand_names):
            self.hand_body_ids[i] = _find_body(name)

        # Wrong contact bodies — only links with real collision geometry.
        # Exclude sensors, cameras, LiDAR, cosmetic/visual links.
        _exclude_wrong = {"imu_", "d435_", "mid360_", "logo_", "pelvis_contour"}
        wrong_keywords = ["torso", "pelvis", "head", "hip"]
        wrong_names = []
        for kw in wrong_keywords:
            for s in body_names:
                if kw in s and s not in contact_foot_names:
                    if not any(ex in s for ex in _exclude_wrong):
                        if s not in wrong_names:
                            wrong_names.append(s)
        self.wrong_body_ids = torch.zeros(
            len(wrong_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(wrong_names):
            self.wrong_body_ids[i] = _find_body(name)

        # Ball body index: ball is the last body in the concatenated contact forces tensor
        self.ball_body_id = self.num_bodies  # index in (num_bodies + 1) flat tensor

        print(f"[ContactClassifier] right_foot_bodies={right_foot_names}")
        print(f"[ContactClassifier] left_foot_bodies={left_foot_names}")
        print(f"[ContactClassifier] hand_bodies={hand_names}")
        print(f"[ContactClassifier] wrong_bodies={wrong_names}")
        print(f"[ContactClassifier] ball_body_id={self.ball_body_id} (in flat contact tensor)")

    # ═══════════════════════════════════════════════════════════════════════
    # Buffer init — override for multi-actor tensors
    # ═══════════════════════════════════════════════════════════════════════

    def _init_buffers(self):
        """Acquire multi-actor tensors: robot (idx 2*i) + ball (idx 2*i+1)."""
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # ── Multi-actor root states: flat tensor [r0, b0, r1, b1, ...] (2N, 13) ──
        # Use stride indexing (0::2, 1::2) to avoid .view() which can cause CUDA issues
        # with Isaac Gym wrapped tensors. root_states/ball_states are clones for safe writes.
        self._all_root_states = gymtorch.wrap_tensor(actor_root_state)  # flat (2N, 13)
        self.root_states = self._all_root_states[0::2, :].clone()
        self.ball_states = self._all_root_states[1::2, :].clone()

        # ── Rigid body states: robot bodies + ball body ──
        all_body_states = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies + 1, 13)
        self.rigid_body_states = all_body_states[:, :-1, :]

        # ── Contact forces ──
        all_contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, self.num_bodies + 1, 3)
        self.contact_forces = all_contact_forces[:, :-1, :]
        self.ball_contact_forces = all_contact_forces[:, -1:, :]

        # ── DOF state (ball has no DOFs — unchanged) ──
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        # ── Standard counters and flags ──
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(
            get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

        # ── Torques / actions / history ──
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # ── Noise ──
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

        # ── Default joint positions and PD gains ──
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # Action scale
        self.action_scale_vec = (
            torch.ones(self.num_dof, dtype=torch.float, device=self.device) *
            self.cfg.control.action_scale)

        # Command ranges
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands,
                                     dtype=torch.float, device=self.device)
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device, requires_grad=False)

        # DR factors
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)

        # Feet air time
        self.feet_air_time = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)

        # ── Observation buffer (125 dims) ──
        self.obs_buf = torch.zeros(self.num_envs, self.num_one_step_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = None

        # ═══════════════════════════════════════════════════════
        # Ball buffers
        # ═══════════════════════════════════════════════════════
        self.ball_pos = self.ball_states[:, :3].clone()
        self.ball_vel = self.ball_states[:, 7:10].clone()
        self.ball_pos_base = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_vel_base = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

        # ═══════════════════════════════════════════════════════
        # Foot position buffers (in base frame, for observation)
        # ═══════════════════════════════════════════════════════
        self.right_foot_pos_base = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.right_foot_vel_base = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.left_foot_pos_base = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.left_foot_vel_base = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

        # ═══════════════════════════════════════════════════════
        # Phase manager
        # ═══════════════════════════════════════════════════════
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.kick_duration = self.cfg.juggle.kick_duration_s
        self.recover_duration = self.cfg.juggle.recover_duration_s
        self.phase_onehot = torch.zeros(self.num_envs, NUM_PHASES, dtype=torch.float, device=self.device)
        self.target_foot_onehot = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)

        # ═══════════════════════════════════════════════════════
        # Contact classifier — semantic ball-body contact flags
        # ═══════════════════════════════════════════════════════
        self.ball_any_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_ground_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.right_foot_ball_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.left_foot_ball_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hand_ball_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.wrong_ball_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.target_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.wrong_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_ball_contact_flag = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # ═══════════════════════════════════════════════════════
        # AMP interface (stub)
        # ═══════════════════════════════════════════════════════
        self.amp_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.amp_motion_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # DR log
        dr = self.cfg.domain_rand
        print(f"[DR] randomize_friction={dr.randomize_friction}")
        print(f"[DR] randomize_payload_mass={dr.randomize_payload_mass}")
        print(f"[DR] push_robots={dr.push_robots}")

    # ═══════════════════════════════════════════════════════════════════════
    # Step & post-physics
    # ═══════════════════════════════════════════════════════════════════════

    def post_physics_step(self):
        """Refresh multi-actor tensors, update ball/foot/phase/contact states."""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Sync root/ball states from flat tensor (clones — must copy manually)
        self.root_states[:] = self._all_root_states[0::2, :]
        self.ball_states[:] = self._all_root_states[1::2, :]

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # ── Robot base state ──
        self.base_quat[:] = self.root_states[:, 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # ── Ball state (world + base frame) ──
        self.ball_pos[:] = self.ball_states[:, :3]
        self.ball_vel[:] = self.ball_states[:, 7:10]
        self.ball_pos_base[:] = quat_rotate_inverse(
            self.base_quat, self.ball_pos - self.root_states[:, :3])
        self.ball_vel_base[:] = quat_rotate_inverse(
            self.base_quat, self.ball_vel - self.root_states[:, 7:10])

        # ── Foot states in base frame ──
        if len(self.right_foot_body_ids) > 0:
            rfoot_idx = self.right_foot_body_ids[0]
            rfoot_pos = self.rigid_body_states[:, rfoot_idx, :3]
            rfoot_vel = self.rigid_body_states[:, rfoot_idx, 7:10]
            self.right_foot_pos_base[:] = quat_rotate_inverse(
                self.base_quat, rfoot_pos - self.root_states[:, :3])
            self.right_foot_vel_base[:] = quat_rotate_inverse(
                self.base_quat, rfoot_vel - self.root_states[:, 7:10])

        if len(self.left_foot_body_ids) > 0:
            lfoot_idx = self.left_foot_body_ids[0]
            lfoot_pos = self.rigid_body_states[:, lfoot_idx, :3]
            lfoot_vel = self.rigid_body_states[:, lfoot_idx, 7:10]
            self.left_foot_pos_base[:] = quat_rotate_inverse(
                self.base_quat, lfoot_pos - self.root_states[:, :3])
            self.left_foot_vel_base[:] = quat_rotate_inverse(
                self.base_quat, lfoot_vel - self.root_states[:, 7:10])

        # ── Phase manager ──
        self._update_phase()

        # ── Contact classifier ──
        self._update_contact_flags()

        # ── Standard post-physics ──
        self._post_physics_step_callback()

        self.compute_observations()
        self.compute_reward()
        self.check_termination()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        # AMP extras (stub — not wired to PPO)
        self.extras["amp_obs"] = self.get_amp_observations()
        self.extras["amp_active"] = self.amp_active
        self.extras["amp_motion_id"] = self.amp_motion_id

    # ═══════════════════════════════════════════════════════════════════════
    # Phase manager
    # ═══════════════════════════════════════════════════════════════════════

    def _update_phase(self):
        """4-phase state machine: WAIT → KICK_RIGHT/KICK_LEFT → RECOVER → WAIT."""
        self.phase_time += self.dt

        if self.cfg.juggle.enable_rule_trigger:
            # Trigger: ball falling + within z window + within xy radius + robot alive
            ball_vz = self.ball_vel_base[:, 2]
            ball_z = self.ball_pos_base[:, 2]
            ball_xy = torch.norm(self.ball_pos_base[:, :2], dim=1)

            trigger = (
                (ball_vz < self.cfg.juggle.trigger_ball_vz_max) &
                (ball_z > self.cfg.juggle.trigger_z_range[0]) &
                (ball_z < self.cfg.juggle.trigger_z_range[1]) &
                (ball_xy < self.cfg.juggle.trigger_xy_radius) &
                (~self.reset_buf)
            )

            # WAIT → KICK_RIGHT (hardcoded right foot for skeleton)
            entering_kick = trigger & (self.phase == WAIT)
            self.phase[entering_kick] = KICK_RIGHT
            self.phase_time[entering_kick] = 0.0

        # KICK_RIGHT → RECOVER
        kick_done = (self.phase == KICK_RIGHT) & (self.phase_time >= self.kick_duration)
        self.phase[kick_done] = RECOVER
        self.phase_time[kick_done] = 0.0

        # KICK_LEFT → RECOVER
        kick_done_l = (self.phase == KICK_LEFT) & (self.phase_time >= self.kick_duration)
        self.phase[kick_done_l] = RECOVER
        self.phase_time[kick_done_l] = 0.0

        # RECOVER → WAIT
        recover_done = (self.phase == RECOVER) & (self.phase_time >= self.recover_duration)
        self.phase[recover_done] = WAIT
        self.phase_time[recover_done] = 0.0

        # Build one-hot and target-foot tensors
        self.phase_onehot.zero_()
        for p in range(NUM_PHASES):
            self.phase_onehot[:, p] = (self.phase == p).float()

        self.target_foot_onehot.zero_()
        self.target_foot_onehot[:, 0] = (self.phase == KICK_RIGHT).float()
        self.target_foot_onehot[:, 1] = (self.phase == KICK_LEFT).float()

        # AMP motion_id: WAIT=0, KICK_RIGHT=1, KICK_LEFT=2, RECOVER=0 (stand)
        self.amp_motion_id[:] = 0
        self.amp_motion_id[self.phase == KICK_RIGHT] = 1
        self.amp_motion_id[self.phase == KICK_LEFT] = 2
        self.amp_active[:] = self.cfg.juggle.use_amp

    # ═══════════════════════════════════════════════════════════════════════
    # Contact classifier — semantic ball-body contact detection
    # ═══════════════════════════════════════════════════════════════════════

    def _update_contact_flags(self):
        """Semantic ball-body contact via force + proximity heuristic.

        Isaac Gym's net_contact_force_tensor is per-rigid-body net force (not
        pairwise).  We approximate ball-body contact as:
            ball has force  AND  body has force  AND  ball centre near body centre.
        """
        cfg_c = self.cfg.contact
        ball_pos = self.ball_pos  # (N, 3)

        # ── Ball force ──
        ball_force = torch.norm(self.ball_contact_forces[:, 0, :], dim=-1)  # (N,)
        self.ball_any_contact[:] = ball_force > cfg_c.ball_force_threshold

        # ── Ball-ground contact (ball touching ground, not robot) ──
        ball_on_ground = ball_pos[:, 2] < (self.cfg.ball.radius + cfg_c.ground_margin)
        self.ball_ground_contact[:] = self.ball_any_contact & ball_on_ground

        # ── Helper: ball-body contact for a group of body ids ──
        def _ball_body_contact(body_ids, distance_threshold):
            """Return (N,) bool: ball in contact with any body in body_ids."""
            if len(body_ids) == 0:
                return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            body_force = torch.norm(self.contact_forces[:, body_ids, :], dim=-1)  # (N, K)
            body_has_contact = body_force > cfg_c.body_force_threshold              # (N, K)
            body_pos = self.rigid_body_states[:, body_ids, 0:3]                     # (N, K, 3)
            dist = torch.norm(body_pos - ball_pos.unsqueeze(1), dim=-1)             # (N, K)
            near = dist < distance_threshold
            contact = self.ball_any_contact.unsqueeze(1) & body_has_contact & near
            return contact.any(dim=1)  # (N,)

        # ── Per-group ball-body contacts ──
        self.right_foot_ball_contact[:] = _ball_body_contact(
            self.right_foot_body_ids, cfg_c.foot_contact_distance)
        self.left_foot_ball_contact[:] = _ball_body_contact(
            self.left_foot_body_ids, cfg_c.foot_contact_distance)
        self.hand_ball_contact[:] = _ball_body_contact(
            self.hand_body_ids, cfg_c.hand_contact_distance)
        self.wrong_ball_contact[:] = _ball_body_contact(
            self.wrong_body_ids, cfg_c.wrong_contact_distance)

        # ── Target contact: right foot + ball NOT on ground (aerial juggle/kick only) ──
        self.target_contact[:] = self.right_foot_ball_contact & (~self.ball_ground_contact)

        # ── Composite flags ──
        # "ball touched robot" = ball contacted any robot body
        self.last_ball_contact_flag[:] = (
            self.right_foot_ball_contact
            | self.left_foot_ball_contact
            | self.hand_ball_contact
            | self.wrong_ball_contact
        ).float()

        # Legacy wrong_contact (for compatibility): keep the semantic version
        self.wrong_contact[:] = self.wrong_ball_contact

    # ═══════════════════════════════════════════════════════════════════════
    # Observations
    # ═══════════════════════════════════════════════════════════════════════

    def compute_observations(self):
        """Base (99) + juggle-specific (26) = 125 dims."""
        # Base observation: lin_vel(3) + ang_vel(3) + projected_gravity(3) + commands(3) +
        #   dof_pos(29) + dof_vel(29) + actions(29) = 99
        base_obs = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)

        # Juggle extension (26 dims)
        phase_time_norm = (self.phase_time / max(self.kick_duration, 0.01)).unsqueeze(1)

        juggle_obs = torch.cat((
            self.ball_pos_base,
            self.ball_vel_base,
            self.right_foot_pos_base,
            self.right_foot_vel_base,
            self.left_foot_pos_base,
            self.left_foot_vel_base,
            self.phase_onehot,
            phase_time_norm,
            self.target_foot_onehot,
            self.last_ball_contact_flag.unsqueeze(1),
        ), dim=-1)

        self.obs_buf = torch.cat((base_obs, juggle_obs), dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = self.obs_buf.clone()

    # ═══════════════════════════════════════════════════════════════════════
    # Noise scale vector — extended to match 125-dim obs
    # ═══════════════════════════════════════════════════════════════════════

    def _get_noise_scale_vec(self, cfg):
        """Noise vector for 125-dim observation."""
        noise_vec = torch.zeros(
            3 + 3 + 3 + 3 + self.num_dof + self.num_dof + self.num_actions + 26,
            device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # Base obs noise
        noise_vec[0:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.  # commands
        noise_vec[12:12 + self.num_dof] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos)
        start = 12 + self.num_dof
        noise_vec[start:start + self.num_dof] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel)
        start += self.num_dof
        noise_vec[start:start + self.num_actions] = 0.  # previous actions
        # Juggle extension: no noise (phase/contact signals should be clean)
        return noise_vec

    # ═══════════════════════════════════════════════════════════════════════
    # Rewards (all stubs — return zeros)
    # ═══════════════════════════════════════════════════════════════════════

    def compute_reward(self):
        """Sum reward functions (all scales=0 during skeleton phase)."""
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)

        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def _reward_ball_alive(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_ball_in_cylinder(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_target_contact(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_ball_vz_after_contact(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_wrong_contact(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_alive(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_upright(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_base_height(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_torques(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_action_rate(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_termination(self):
        return self.reset_buf.float()

    # ═══════════════════════════════════════════════════════════════════════
    # Termination
    # ═══════════════════════════════════════════════════════════════════════

    def check_termination(self):
        """Standard gravity termination + ball out-of-bounds."""
        # Contact termination (from base — empty by default for juggle)
        self.reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
            dim=1
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.gravity_termination_buf = torch.norm(self.projected_gravity[:, :2], dim=1) > 0.999
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.gravity_termination_buf

        # Ball termination: below ground or too far (world frame)
        if hasattr(self.cfg, 'termination'):
            ball_z = self.ball_states[:, 2]
            ball_xy = torch.norm(self.ball_states[:, :2] - self.env_origins[:, :2], dim=1)
            ball_below = ball_z < self.cfg.termination.ball_min_z
            ball_far = ball_xy > self.cfg.termination.ball_max_xy_dist
            self.reset_buf |= ball_below
            self.reset_buf |= ball_far

    # ═══════════════════════════════════════════════════════════════════════
    # Reset
    # ═══════════════════════════════════════════════════════════════════════

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        # Robot DOF reset
        self._reset_dofs(env_ids)

        # Reset root states (robot + ball, via indexed actor API)
        self._reset_root_states(env_ids)

        # Clear history buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.reset_buf[env_ids] = 1

        # Phase reset
        self.phase[env_ids] = WAIT
        self.phase_time[env_ids] = 0.0

        # Contact classifier reset
        self.ball_any_contact[env_ids] = False
        self.ball_ground_contact[env_ids] = False
        self.right_foot_ball_contact[env_ids] = False
        self.left_foot_ball_contact[env_ids] = False
        self.hand_ball_contact[env_ids] = False
        self.wrong_ball_contact[env_ids] = False
        self.target_contact[env_ids] = False
        self.wrong_contact[env_ids] = False
        self.last_ball_contact_flag[env_ids] = 0.0

        # AMP reset
        self.amp_motion_id[env_ids] = 0
        self.amp_active[env_ids] = self.cfg.juggle.use_amp

        # DR factors reset
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(
                self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1],
                (len(env_ids), self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(
                self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1],
                (len(env_ids), self.num_dof), device=self.device)

        # Episode extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids] /
                torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            self.episode_sums[key][env_ids] = 0.

        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0

    def _reset_dofs(self, env_ids):
        """DOF reset — multi-actor aware: robot actor idx = 2*env_idx (ball is 2*env_idx+1)."""
        dof_lower = self.dof_pos_limits[:, 0].view(1, -1)
        dof_upper = self.dof_pos_limits[:, 1].view(1, -1)

        if self.cfg.domain_rand.randomize_initial_joint_pos:
            init_dof_pos = self.default_dof_pos.repeat(len(env_ids), 1) * torch_rand_float(
                self.cfg.domain_rand.initial_joint_pos_scale[0],
                self.cfg.domain_rand.initial_joint_pos_scale[1],
                (len(env_ids), self.num_dof), device=self.device)
            init_dof_pos += torch_rand_float(
                self.cfg.domain_rand.initial_joint_pos_offset[0],
                self.cfg.domain_rand.initial_joint_pos_offset[1],
                (len(env_ids), self.num_dof), device=self.device)
            self.dof_pos[env_ids] = torch.clip(init_dof_pos, dof_lower, dof_upper)
        else:
            self.dof_pos[env_ids] = self.default_dof_pos.repeat(len(env_ids), 1)

        self.dof_vel[env_ids] = 0.
        # Multi-actor: robot has global actor index 2*env_idx, ball is 2*env_idx+1 (no DOFs)
        robot_actor_ids = (2 * env_ids).to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(robot_actor_ids), len(robot_actor_ids))

    def _reset_root_states(self, env_ids):
        """Reset robot to default stand pose + ball to random position in front."""
        # Robot root state
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = 0.0

        # Ball root state
        self.ball_states[env_ids] = self.base_init_state
        self.ball_states[env_ids, :3] = self.env_origins[env_ids, :3]
        # Offset: cfg.ball.init_pos_base + noise
        noise = torch.stack([
            torch_rand_float(-self.cfg.ball.init_pos_noise[0],
                             self.cfg.ball.init_pos_noise[0],
                             (len(env_ids),), device=self.device),
            torch_rand_float(-self.cfg.ball.init_pos_noise[1],
                             self.cfg.ball.init_pos_noise[1],
                             (len(env_ids),), device=self.device),
            torch_rand_float(-self.cfg.ball.init_pos_noise[2],
                             self.cfg.ball.init_pos_noise[2],
                             (len(env_ids),), device=self.device),
        ], dim=1)
        self.ball_states[env_ids, 0] += self.cfg.ball.init_pos_base[0] + noise[:, 0]
        self.ball_states[env_ids, 1] += self.cfg.ball.init_pos_base[1] + noise[:, 1]
        self.ball_states[env_ids, 2] = self.cfg.ball.init_pos_base[2] + noise[:, 2]
        # Velocity noise
        vel_noise = torch.stack([
            torch_rand_float(-self.cfg.ball.init_vel_noise[0],
                             self.cfg.ball.init_vel_noise[0],
                             (len(env_ids),), device=self.device),
            torch_rand_float(-self.cfg.ball.init_vel_noise[1],
                             self.cfg.ball.init_vel_noise[1],
                             (len(env_ids),), device=self.device),
            torch_rand_float(-self.cfg.ball.init_vel_noise[2],
                             self.cfg.ball.init_vel_noise[2],
                             (len(env_ids),), device=self.device),
        ], dim=1)
        self.ball_states[env_ids, 7:10] = (
            torch.tensor(self.cfg.ball.init_vel_base, device=self.device) + vel_noise)
        self.ball_states[env_ids, 10:13] = 0.0

        # Combined indexed set: robot(idx=2*i) + ball(idx=2*i+1)
        all_states = torch.empty(2 * self.num_envs, 13, dtype=torch.float, device=self.device)
        all_states[0::2, :] = self.root_states
        all_states[1::2, :] = self.ball_states
        env_ids_int32 = torch.cat(
            (2 * env_ids, 2 * env_ids + 1)).to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(all_states),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # ═══════════════════════════════════════════════════════════════════════
    # _push_robots — override for multi-actor
    # ═══════════════════════════════════════════════════════════════════════

    def _push_robots(self):
        """Push robot base velocity (multi-actor aware)."""
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)
        all_states = torch.empty(2 * self.num_envs, 13, dtype=torch.float, device=self.device)
        all_states[0::2, :] = self.root_states
        all_states[1::2, :] = self.ball_states
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(all_states))

    # ═══════════════════════════════════════════════════════════════════════
    # AMP interface (stub — not wired to PPO)
    # ═══════════════════════════════════════════════════════════════════════

    def get_amp_observations(self):
        """Return AMP observation: concatenated [last_dof_pos, dof_pos] (58 dims)."""
        return torch.cat([self.last_dof_pos, self.dof_pos], dim=-1)

    def _prepare_reward_function(self):
        """Overridden to only register juggle reward stubs."""
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for name in self.reward_scales.keys()
        }
