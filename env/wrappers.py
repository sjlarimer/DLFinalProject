
import numpy as np


def shape_rewards(rewards, infos):
    """
    Define custom rewards based on surviving, killing or hitting zombie, penalty for dying. 
    """
    pass


def resize(obs):
    """
    Resize observation (e.g., to 84x84)
    """
    pass


def to_gray(obs):
    """
    Convert RGB observation to grayscale
    """
    pass


def normalize(obs):
    """
    Normalize pixel values (e.g., /255.0)
    """
    pass


def preprocess(obs):
    """
    Full preprocessing pipeline
    """
    pass


def init_frame_stack(agent):
    """
    Initialize frame buffer for a given agent
    """
    pass


def stack_frames(agent, obs):
    """
    Maintain last N frames per agent and return stacked obs
    """
    pass


def preprocess_agent(agent, obs):
    """
    Full per-agent pipeline:
    preprocess + frame stacking
    """
    pass
