import numpy as np
import cv2
from collections import deque

frame_stacks = {}

def preprocess(obs):
    obs = np.array(obs)

    obs = cv2.resize(obs, (84, 84))

    if len(obs.shape) == 3:
        obs = np.mean(obs, axis=2, keepdims=True)

    return obs.astype(np.float32) / 255.0

frame_stacks = {}

def stack_frames(agent, obs, k=4):
    if agent not in frame_stacks:
        frame_stacks[agent] = deque(maxlen=k)

    frame_stacks[agent].append(obs)

    while len(frame_stacks[agent]) < k:
        frame_stacks[agent].append(obs)

    return np.concatenate(frame_stacks[agent], axis=-1)

def preprocess_agent(agent, obs):
    obs = preprocess(obs)
    obs = stack_frames(agent, obs)
    return obs

def shape_rewards(rewards, infos):
    shaped = {}
    for agent, r in rewards.items():
        r += 0.01 ## reward if survives
        if infos[agent].get("died", False): ## penalize if dies
            r -= 1.0
        ## other to do? Reward for killing?? 
        shaped[agent] = r
    return shaped