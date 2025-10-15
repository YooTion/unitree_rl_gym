
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from legged_gym.utils.terrain import Terrain

@torch.jit.script
def copysign(a:float, b:torch.Tensor):
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)

class G1Robot23(LeggedRobot):
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        self.rpy = get_euler_xyz(self.base_quat)

        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()
        self.joint_names = self.gym.get_actor_joint_names(self.envs[0], self.actor_handles[0])
        print("所有关节名称:", self.joint_names)


    def check_termination(self):
        super().check_termination()
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        self.rpy[:] = get_euler_xyz(self.base_quat[:])
        
        return super()._post_physics_step_callback()
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.5 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
            self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
            self.command_ranges["ang_vel_yaw"][0] = np.clip(self.command_ranges["ang_vel_yaw"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["ang_vel_yaw"][1] = np.clip(self.command_ranges["ang_vel_yaw"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    
    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    # sin_phase,
                                    # cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel, #3
                                    self.base_ang_vel  * self.obs_scales.ang_vel, #3
                                    self.projected_gravity, #3
                                    self.commands[:, :3] * self.commands_scale, #3
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, #23
                                    self.dof_vel * self.obs_scales.dof_vel, #23
                                    self.actions, #23
                                    # sin_phase,
                                    # cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type == 'trimesh':
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _create_envs(self):
        super()._create_envs()
        
        # find all hip dofs, obeying order in self.dof_names
        self.hip_dof_indices = []
        for i in range(len(self.dof_names)):
            if any([s in self.dof_names[i] for s in self.cfg.asset.hip_dof_name]):
                self.hip_dof_indices.append(i)
        self.hip_dof_indices = torch.tensor(self.hip_dof_indices, dtype=torch.long, device=self.device, requires_grad=False)
        
        self.arm_dof_indices = []
        for i in range(len(self.dof_names)):
            if any([s in self.dof_names[i] for s in self.cfg.asset.arm_dof_name]):
                self.arm_dof_indices.append(i)
        self.arm_dof_indices = torch.tensor(self.arm_dof_indices, dtype=torch.long, device=self.device, requires_grad=False)
        
        self.waist_dof_indices = []
        for i in range(len(self.dof_names)):
            if any([s in self.dof_names[i] for s in self.cfg.asset.waist_dof_name]):
                self.waist_dof_indices.append(i)
        self.waist_dof_indices = torch.tensor(self.waist_dof_indices, dtype=torch.long, device=self.device, requires_grad=False)

        self.ankle_dof_indices = []
        for i in range(len(self.dof_names)):
            if any([s in self.dof_names[i] for s in self.cfg.asset.ankle_dof_name]):
                self.ankle_dof_indices.append(i)
        self.ankle_dof_indices = torch.tensor(self.ankle_dof_indices, dtype=torch.long, device=self.device, requires_grad=False)

        self.hip_knee_dof_indices = []
        for i in range(len(self.dof_names)):
            if any([s in self.dof_names[i] for s in self.cfg.asset.hip_knee_dof_name]):
                self.hip_knee_dof_indices.append(i)
        self.hip_knee_dof_indices = torch.tensor(self.hip_knee_dof_indices, dtype=torch.long, device=self.device, requires_grad=False)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
    
    # ================================================ Rewards ================================================== #
    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)
    
    def _reward_target_height(self):
        # Reward base height
        target_height = self.root_states[:, 2]
        return target_height
    
    def _reward_arm_dof_deviation(self):
        arm_dof_err = torch.sum(torch.abs(self.dof_pos[:, self.arm_dof_indices] - self.default_dof_pos[:, self.arm_dof_indices]), dim=1)
        return arm_dof_err
    
    def _reward_waist_dof_deviation(self):
        waist_dof_err = torch.sum(torch.abs(self.dof_pos[:, self.waist_dof_indices] - self.default_dof_pos[:, self.waist_dof_indices]), dim=1)
        return waist_dof_err
    
    def _reward_hip_dof_deviation(self):
        hip_dof_err = torch.sum(torch.abs(self.dof_pos[:, self.hip_dof_indices] - self.default_dof_pos[:, self.hip_dof_indices]), dim=1)
        return hip_dof_err
    
    def _reward_ankle_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos[:, self.ankle_dof_indices] - self.dof_pos_limits[self.ankle_dof_indices, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos[:, self.ankle_dof_indices] - self.dof_pos_limits[self.ankle_dof_indices, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_hip_knee_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel[:, self.hip_knee_dof_indices] - self.dof_vel[:, self.hip_knee_dof_indices]) / self.dt), dim=1)

    def _reward_hip_knee_dof_torques(self):
        return torch.sum(torch.square(self.torques[:, self.hip_knee_dof_indices]), dim=1)
    
    def _reward_feet_slip(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        # contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        # penalize = torch.square(contact_feet_vel[:, :, :2])
        # return torch.sum(penalize, dim=(1,2))
        contact_vel_xy = torch.norm(self.feet_vel[:, :, :2], dim=2) * contact
        return torch.sum(contact_vel_xy, dim=1)
    
    # ================================================ Rewards unique ================================================== #
    def _reward_no_movement_when_stationary(self):
        """
        Reward for minimizing movement when the robot's current velocity is zero.
        Penalizes joint velocities and deviations from default positions.
        """
        # Check if the robot's linear and angular velocities are close to zero
        is_stationary = torch.norm(self.base_lin_vel, dim=1) < 0.1  # Linear velocity threshold
        # is_stationary &= torch.norm(self.base_ang_vel, dim=1) < 0.1  # Angular velocity threshold

        # Penalize joint velocities when stationary
        joint_vel_penalty = torch.sum(torch.square(self.dof_vel), dim=1)

        # Penalize deviations from default joint positions when stationary
        joint_pos_penalty = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

        # Combine penalties and apply only when stationary
        total_penalty = (joint_vel_penalty + joint_pos_penalty) * is_stationary
        # print("total_penalty----------------",total_penalty)

        # Return the negative penalty as a reward
        return total_penalty
    
    def _reward_large_stride(self):
        """
        Reward for increasing stride length without target limit.
        Encourages the robot to take longer steps by measuring the horizontal distance traveled by the feet.
        """
        # Calculate the horizontal displacement (x, y) of each foot since the last step
        foot_displacement = torch.norm(self.feet_pos[:, :, :2] - self.last_feet_pos[:, :, :2], dim=2)
        
        # Use the maximum displacement among all feet as the stride length
        stride_length = torch.max(foot_displacement, dim=1)[0]
        
        # Normalize stride length by a reference value and clip
        normalized_stride = torch.clamp(stride_length / 0.8, 0, 1.5)  # 0.8m as reference
        
        # Quadratic reward to favor larger strides
        reward_stride = normalized_stride ** 2
        
        # Update the last foot positions for the next step
        self.last_feet_pos = self.feet_pos.clone()
        
        return reward_stride