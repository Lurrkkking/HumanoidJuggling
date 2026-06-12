from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class G1StandCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 256
        num_actor_history = 1  # no history stacking for minimal smoke test
        num_actions = 29
        num_dofs = 29
        num_one_step_observations = 3 + 3 + 3 + 3 + num_dofs + num_dofs + num_actions  # = 99
        num_privileged_obs = None
        num_observations = num_one_step_observations  # no history

        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 15  # seconds

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4  # lin_vel_x, lin_vel_y, ang_vel_yaw, heading
        resampling_time = 10.
        heading_command = True

        class ranges:
            lin_vel_x = [0.0, 0.0]  # zero command = stand still
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [-0.01, 0.01]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.8]  # x, y, z [m]

        default_joint_angles = {
            # Legs (12)
            'left_hip_pitch_joint': -0.1,
            'left_hip_roll_joint': 0.2,
            'left_hip_yaw_joint': 0.0,
            'left_knee_joint': 0.3,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': -0.2,
            'right_hip_pitch_joint': -0.1,
            'right_hip_roll_joint': -0.2,
            'right_hip_yaw_joint': 0.0,
            'right_knee_joint': 0.3,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.2,
            # Waist (3)
            'waist_yaw_joint': 0.0,
            'waist_roll_joint': 0.0,
            'waist_pitch_joint': 0.0,
            # Left arm (7)
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.5,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 1.2,
            'left_wrist_roll_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            # Right arm (7)
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': -0.5,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 1.2,
            'right_wrist_roll_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {
            'hip_yaw': 150,
            'hip_roll': 150,
            'hip_pitch': 150,
            'knee': 300,
            'ankle': 40,
            'shoulder': 150,
            'elbow': 150,
            'waist': 150,
            'wrist': 20,
        }
        damping = {
            'hip_yaw': 2,
            'hip_roll': 2,
            'hip_pitch': 2,
            'knee': 4,
            'ankle': 2,
            'shoulder': 2,
            'elbow': 2,
            'waist': 2,
            'wrist': 0.5,
        }
        action_scale = 0.25
        decimation = 4
        curriculum_joints = []
        # G1-specific joint name lists (kept in config, not consumed by base LeggedRobot)
        left_leg_joints = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint',
                           'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint']
        right_leg_joints = ['right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
                            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']
        knee_joints = ['left_knee_joint', 'right_knee_joint']
        left_arm_joints = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
                           'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint']
        right_arm_joints = ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
                            'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
        elbow_joints = ['left_elbow_joint', 'right_elbow_joint']
        wrist_joints = ['left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
                        'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
        upper_body_link = "pelvis"
        torso_link = "torso_link"
        left_hip_joints = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint']
        right_hip_joints = ['right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint']

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/urdf/g1_29.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee", "torso", "shoulder", "elbow", "pelvis", "hand", "head"]
        terminate_after_contacts_on = []
        # G1-specific asset fields (kept in config, not consumed by base LeggedRobot)
        hand_name = "hand"
        contact_foot_names = "ankle_roll_link"
        waist_joints = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
        ankle_joints = ["left_ankle_pitch_joint", "right_ankle_pitch_joint"]
        imu_link = "imu_link"
        knee_names = ["left_knee_link", "right_knee_link"]
        keyframe_name = "keyframe"
        disable_gravity = False
        collapse_fixed_joints = False  # keep fixed joints (sensors)
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False
        density = 0.001
        angular_damping = 0.01
        linear_damping = 0.01
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.01
        thickness = 0.01

    class terrain(LeggedRobotCfg.terrain):
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        randomize_restitution = False
        randomize_payload_mass = False
        randomize_com_displacement = False
        randomize_link_mass = False
        randomize_kp = False
        randomize_kd = False
        randomize_initial_joint_pos = False
        push_robots = False
        delay = False

    class rewards(LeggedRobotCfg.rewards):
        class scales:
            # positive rewards
            alive = 1.0
            upright = 2.0
            base_height = 1.0
            feet_contact = 0.5
            # penalties
            base_lin_vel = -0.2
            base_ang_vel = -0.1
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -2.5e-7
            action_rate = -0.01
            dof_pos_limits = -1.0
            termination = -2.0

        only_positive_rewards = False
        base_height_target = 0.76  # root z when feet on ground
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.95
        max_contact_force = 1000.

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
        clip_observations = 100.
        clip_actions = 100.

    class noise(LeggedRobotCfg.noise):
        add_noise = False
        noise_level = 1.0
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05

    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1
        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 8
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class G1StandCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        runner_class_name = 'OnPolicyRunner'
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration (small for smoke test)
        max_iterations = 2000

        # logging
        save_interval = 100
        run_name = 'g1_stand'
        experiment_name = 'g1'

        # load and resume
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
