from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg23( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'waist_yaw_joint' : 0.,
           'left_shoulder_pitch_joint': 0.3,
           'left_shoulder_roll_joint': 0.3,
           'left_shoulder_yaw_joint': 0.0,
           'left_elbow_joint': 0.9,
           'left_wrist_roll_joint': 0.0,
           'right_shoulder_pitch_joint': 0.3,
           'right_shoulder_roll_joint': 0.3,
           'right_shoulder_yaw_joint': 0.0,
           'right_elbow_joint': 0.9,
           'right_wrist_roll_joint': 0.0
        }
        
    
    class env(LeggedRobotCfg.env):
        num_observations = 78
        num_privileged_obs = 81
        num_actions = 23
    
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     'waist_yaw_joint': 100,
                     'shoulder_pitch': 50,
                     'shoulder_roll': 50,
                     'shoulder_yaw': 50,
                     'elbow': 50,
                     'wrist_roll': 30,
                    }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'waist_yaw_joint': 2,
                     'shoulder_pitch': 2,
                     'shoulder_roll': 2,
                     'shoulder_yaw': 2,
                     'elbow': 2,
                     'wrist_roll': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_23dof_rev_1_0.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis", "torso_link"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

        hip_dof_name = ["hip_roll", "hip_yaw"]
        hip_knee_dof_name = ["hip", "knee"]
        ankle_dof_name = ["ankle_roll", "ankle_pitch"]
        
        arm_dof_name = ["shoulder", "elbow", "wrist", ]
        waist_dof_name = ["waist", ]
        
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        feet_dist_min = 0.2
        feet_dist_max = 0.6
        only_positive_rewards = True
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -5.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 2.0
            collision = 0.0
            action_rate = -0.05
            # upper_action_rate = -0.01
            dof_pos_limits = -2.0
            # alive = 2.0
            hip_pos = -1.0
            contact_no_vel = -0.2
            # feet_swing_height = -20.0
            contact = 0.25
            # target_height = 5.0
            # feet_contact_forces = -1

            feet_slip = -0.1
            ankle_dof_pos_limits = -0.2
            hip_dof_deviation = -0.2
            arm_dof_deviation = -1.0
            waist_dof_deviation = -1.0
            hip_knee_dof_acc = -1.25e-7
            hip_knee_dof_torques = -2.0e-6
            # termination = -200.0

            no_movement_when_stationary = -0.1
            large_stride = 5.5


# class G1RoughCfgPPO23( LeggedRobotCfgPPO ):
#     class policy:
#         init_noise_std = 1.0
#         actor_hidden_dims = [512, 256, 128]
#         critic_hidden_dims = [512, 256, 128]
#         activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
#         # only for 'ActorCriticRecurrent':
#         rnn_type = 'gru'
#         rnn_hidden_size = 64
#         rnn_num_layers = 1
        
#     class algorithm( LeggedRobotCfgPPO.algorithm ):
#         entropy_coef = 0.01
#     class runner( LeggedRobotCfgPPO.runner ):
#         policy_class_name = "ActorCriticRecurrent"
#         max_iterations = 20000
#         run_name = ''
#         experiment_name = 'g1_23'

class G1RoughCfgPPO23( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'gru'
        # rnn_hidden_size = 64
        # rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1e-4
        num_learning_epochs = 2
        gamma = 0.994 #0.994 聚焦当下奖励
        lam = 0.9
        num_mini_batches = 4
        
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCritic"
        algorithm_class_name = 'PPO'
        max_iterations = 5000
        run_name = ''
        experiment_name = 'g1_23'
  
