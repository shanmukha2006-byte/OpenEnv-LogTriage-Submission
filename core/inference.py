# core/inference.py
# RL agent inference logic — runs one episode and returns a summary.

import numpy as np
from core.env import LogTriageEnvironment, Action


def _select_action(observation: np.ndarray) -> Action:
    """
    Simple rule-based policy (replace with your trained model weights).
    
    High mean activation  → ESCALATE
    Medium mean activation → MONITOR
    Low mean activation   → IGNORE
    """
    mean_val = float(observation.mean())

    if mean_val > 0.66:
        return Action.ESCALATE
    elif mean_val > 0.33:
        return Action.MONITOR
    else:
        return Action.IGNORE


def run_inference(max_steps: int = 10) -> dict:
    """
    Run one inference episode and return a structured summary dict.
    
    Returns:
        {
            "total_reward"   : float,
            "steps"          : int,
            "accuracy"       : float,
            "episode_log"    : list[dict],
        }
    """
    env    = LogTriageEnvironment()
    obs, _ = env.reset()

    total_reward = 0.0
    episode_log  = []

    for step in range(max_steps):
        action              = _select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_log.append({
            "step"         : step + 1,
            "action"       : info["agent_action"],
            "true_label"   : info["true_label"],
            "reward"       : reward,
        })

        if terminated or truncated:
            break

    steps    = len(episode_log)
    accuracy = round(total_reward / steps, 4) if steps > 0 else 0.0

    return {
        "total_reward": round(total_reward, 4),
        "steps"       : steps,
        "accuracy"    : accuracy,
        "episode_log" : episode_log,
    }
