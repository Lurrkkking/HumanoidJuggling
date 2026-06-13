from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.g1.g1_stand_config import G1StandCfg, G1StandCfgPPO


class G1JuggleCfg(G1StandCfg):
    """G1 juggling task config skeleton — inherits stand defaults, overrides minimally."""

    class env(G1StandCfg.env):
        num_envs = 256
        # 99 base + 26 juggle-specific = 125
        num_one_step_observations = 125
        num_observations = 125  # no history stacking
        episode_length_s = 15

    class asset(G1StandCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/urdf/g1_29_juggle.urdf"
        ballfile = "{LEGGED_GYM_ROOT_DIR}/resources/objects/ball/ball.urdf"
        name = "g1"
        foot_name = "ankle_roll"
        contact_foot_names = "ankle_roll_link"
        terminate_after_contacts_on = []
        penalize_contacts_on = []
        collapse_fixed_joints = False
        # Keep hand_name / waist_joints / etc. from G1StandCfg for contact classifier

    class ball:
        # NOTE: ball physical radius/mass come from ball.urdf.
        # cfg.ball.radius is used for reset positioning and contact distance checks.
        # To change actual physics, edit ball.urdf AND update cfg.ball.radius/mass.
        radius = 0.11
        mass = 0.43
        init_pos_base = [0.28, 0.0, 0.75]     # ball init pos in robot base frame
        init_pos_noise = [0.08, 0.08, 0.15]
        init_vel_base = [0.0, 0.0, 0.0]
        init_vel_noise = [0.1, 0.1, 0.2]

    class contact:
        """Ball-body semantic contact thresholds — conservative first pass."""
        ball_force_threshold = 1.0
        body_force_threshold = 1.0
        foot_contact_distance = 0.18      # ball center to ankle_roll_link center
        hand_contact_distance = 0.16
        wrong_contact_distance = 0.20
        ground_margin = 0.02             # ball z below this → ground contact

    class juggle:
        task_mode = "skeleton"   # skeleton / single_touch / juggle
        use_amp = False
        amp_coef = 0.0

        phase_names = ["WAIT", "KICK_RIGHT", "KICK_LEFT", "RECOVER"]
        kick_duration_s = 0.7
        recover_duration_s = 0.4

        enable_rule_trigger = True
        trigger_ball_vz_max = -0.05   # ball must be falling
        trigger_z_range = [0.35, 0.85]  # ball z window in base frame
        trigger_xy_radius = 0.35        # ball xy must be within this radius

    class amp:
        use_amp = False
        obs_type = "dof"
        num_obs = 29 * 2
        amp_coef = 0.0
        motion_names = ["stand", "right_kickup", "left_kickup"]
        dataset_folder = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/juggle"

    class rewards(G1StandCfg.rewards):
        only_positive_rewards = False

        class scales:
            # skeleton stage: all zero — no reward training
            ball_alive = 0.0
            ball_in_cylinder = 0.0
            target_contact = 0.0
            ball_vz_after_contact = 0.0
            wrong_contact = 0.0

            alive = 0.0
            upright = 0.0
            base_height = 0.0
            torques = 0.0
            action_rate = 0.0
            termination = 0.0

    class termination(G1StandCfg.termination if hasattr(G1StandCfg, 'termination') else type('_', (), {})):
        """Ball termination thresholds (world frame)."""
        ball_min_z = -2.0       # ball below ground → reset
        ball_max_xy_dist = 5.0  # ball too far horizontally → reset


class G1JuggleCfgPPO(G1StandCfgPPO):
    class runner(G1StandCfgPPO.runner):
        run_name = 'g1_juggle'
        experiment_name = 'g1'
        max_iterations = 2000
        save_interval = 100
