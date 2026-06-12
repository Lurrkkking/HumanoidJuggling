# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class AnymalCFlatCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 48
        num_actions = 12

    class terrain(LeggedRobotCfg.terrain):
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.6]  # x,y,z [m]
        default_joint_angles = {
            'LF_HAA': 0.0,
            'LF_HFE': 0.4,
            'LF_KFE': -0.8,
            'LH_HAA': 0.0,
            'LH_HFE': 1.0,
            'LH_KFE': -1.2,
            'RF_HAA': 0.0,
            'RF_HFE': 0.4,
            'RF_KFE': -0.8,
            'RH_HAA': 0.0,
            'RH_HFE': 1.0,
            'RH_KFE': -1.2,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'HAA': 40., 'HFE': 40., 'KFE': 40.}
        damping = {'HAA': 0.5, 'HFE': 0.5, 'KFE': 0.5}
        action_scale = 0.5
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_c/urdf/anymal_c.urdf'
        name = "anymal_c"
        foot_name = "FOOT"
        penalize_contacts_on = ["SHANK", "THIGH"]
        terminate_after_contacts_on = ["BASE"]
        self_collisions = 1

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.5
        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            torques = -0.00001
            dof_pos_limits = -10.0


class AnymalCFlatCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'flat_anymal_c'
        max_iterations = 1500


class AnymalCRoughCfg(AnymalCFlatCfg):
    class env(AnymalCFlatCfg.env):
        num_envs = 4096
        num_one_step_observations = 50
        num_observations = 50 * AnymalCFlatCfg.env.num_actor_history
    class terrain(AnymalCFlatCfg.terrain):
        pass
    class rewards(AnymalCFlatCfg.rewards):
        class scales(AnymalCFlatCfg.rewards.scales):
            feet_air_time = 1.0


class AnymalCRoughCfgPPO(AnymalCFlatCfgPPO):
    class runner(AnymalCFlatCfgPPO.runner):
        experiment_name = 'rough_anymal_c'
