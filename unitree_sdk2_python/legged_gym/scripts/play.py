import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from pynput import keyboard
import threading

import numpy as np
import torch

# 全局变量存储速度命令
current_lin_vel = 0.0
current_lat_vel = 0.0
lock = threading.Lock()  # 线程锁确保数据安全

def on_press(key):
    global current_lin_vel, current_lat_vel
    try:
        with lock:
            if key.char == 'w':
                current_lin_vel = 1.0
            elif key.char == 's':
                current_lin_vel = -1.0
            elif key.char == 'a':
                current_lat_vel = 1.0
            elif key.char == 'd':
                current_lat_vel = -1.0
    except AttributeError:
        pass

def on_release(key):
    global current_lin_vel, current_lat_vel
    try:
        with lock:
            if key.char in ('w', 's'):
                current_lin_vel = 0.0
            elif key.char in ('a', 'd'):
                current_lat_vel = 0.0
    except AttributeError:
        pass


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        # 使用线程安全的数值
        with lock:
            env.commands[:, 0] = current_lin_vel  # 线速度
            env.commands[:, 1] = current_lat_vel  # 侧向速度
            env.commands[:, 2] = 0.0
            env.commands[:, 3] = 0.0
        obs, _, rews, dones, infos = env.step(actions.detach())

        # 更新相机位置和方向，使其跟随机器人并保持侧面视角
        root_positions = env.root_states[:, :3].cpu().numpy()
        root_orientations = env.root_states[:, 3:7].cpu().numpy()
        
        # 获取第一个机器人的位置和方向
        robot_pos = root_positions[0]
        robot_orn = root_orientations[0]
        
        # 计算相机位置（机器人侧面2米，高度1.5米）
        camera_distance = 3.0
        camera_height = 0.8
        camera_pos = np.array([
            robot_pos[0],  # 侧面视角
            robot_pos[1] - camera_distance,
            robot_pos[2] + camera_height
        ])
        
        # 相机看向机器人
        look_at = np.array([
            robot_pos[0],
            robot_pos[1],
            robot_pos[2]
        ])
        
        # 更新相机位置
        env.set_camera(camera_pos, look_at)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    args = get_args()
    play(args)
