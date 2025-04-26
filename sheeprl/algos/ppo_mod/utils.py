from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, Sequence

import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor

from sheeprl.algos.ppo.agent import PPOPlayer, build_agent
from sheeprl.utils.env import make_env
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE
from sheeprl.utils.utils import unwrap_fabric

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo

AGGREGATOR_KEYS = {
    "Rewards/rew_avg", 
    "Game/ep_len_avg", 
    "Loss/value_loss", 
    "Loss/policy_loss", 
    "Loss/entropy_loss",
    "Metrics/observed_ratio",
    "Metrics/area_ratio",
    "Metrics/reward",
    "Metrics/total_power",
    "Metrics/served_people",
    "Metrics/service_ratio",
    "Test/cumulative_reward",
    "Test/episode_length",
    "Test/observed_ratio_mean",
    "Test/observed_area_ratio_mean",
    "Test/service_ratio_mean",
    "Test/reward_mean",
    "Test/reward_std",
    "Test/total_power_mean",
    "Test/served_people_mean",
    "Test_Steps/reward",
    "Test_Steps/observed_ratio",
    "Test_Steps/observed_area_ratio",
    "Test_Steps/service_ratio",
    "Test_Steps/total_power",
    "Test_Steps/served_people"
}
MODELS_TO_REGISTER = {"agent"}


def normalize_density(density_matrix, max_value=20.0):
    """对密度矩阵进行特殊归一化处理
    
    Args:
        density_matrix: 输入的密度矩阵
        max_value: 预设的密度矩阵最大值，默认为20.0
    
    Returns:
        归一化后的密度矩阵，范围为[-0.5, 0.5]
    """
    return density_matrix / max_value - 0.5


def prepare_obs(
    fabric: Fabric, obs: Dict[str, np.ndarray], *, cnn_keys: Sequence[str] = [], num_envs: int = 1, **kwargs
) -> Dict[str, Tensor]:
    torch_obs = {}
    for k in obs.keys():
        torch_obs[k] = torch.from_numpy(obs[k].copy()).to(fabric.device).float()
        if k in cnn_keys:
            if k == "density_matrix":
                torch_obs[k] = torch_obs[k].reshape(num_envs, -1, *torch_obs[k].shape[-2:])
                torch_obs[k] = normalize_density(torch_obs[k])
            else:
                torch_obs[k] = torch_obs[k].reshape(num_envs, -1, *torch_obs[k].shape[-2:])
        else:
            torch_obs[k] = torch_obs[k].reshape(num_envs, -1)
    return normalize_obs(torch_obs, cnn_keys, obs.keys())


@torch.no_grad()
def test(agent: PPOPlayer, fabric: Fabric, cfg: Dict[str, Any], log_dir: str):
    env = make_env(cfg, None, 0, log_dir, "test", vector_env_idx=0)()
    agent.eval()
    done = False
    cumulative_rew = 0
    obs, info = env.reset(seed=cfg.seed)
    
    episode_length = 0
    step_metrics = []
    
    step_data = {"step": 0}
    if "observed_ratio" in info and info["observed_ratio"] is not None:
        step_data["observed_ratio"] = info["observed_ratio"]
    if "observed_area_ratio" in info and info["observed_area_ratio"] is not None:
        step_data["observed_area_ratio"] = info["observed_area_ratio"]
    if "service_ratio" in info and info["service_ratio"] is not None:
        step_data["service_ratio"] = info["service_ratio"]
    if "total_power" in info and info["total_power"] is not None:
        step_data["total_power"] = info["total_power"]
    if "served_people" in info and info["served_people"] is not None:
        step_data["served_people"] = info["served_people"]
    
    step_metrics.append(step_data)

    while not done:
        torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder)

        # Act greedly through the environment
        actions = agent.get_actions(torch_obs, greedy=True)
        if agent.actor.is_continuous:
            actions = torch.cat(actions, dim=-1)
        else:
            actions = torch.cat([act.argmax(dim=-1) for act in actions], dim=-1)

        obs, reward, done, truncated, info = env.step(actions.cpu().numpy().reshape(env.action_space.shape))
        done = done or truncated
        cumulative_rew += reward
        episode_length += 1
        
        step_data = {"step": episode_length, "reward": reward}
        if "observed_ratio" in info and info["observed_ratio"] is not None:
            step_data["observed_ratio"] = info["observed_ratio"]
        if "observed_area_ratio" in info and info["observed_area_ratio"] is not None:
            step_data["observed_area_ratio"] = info["observed_area_ratio"]
        if "service_ratio" in info and info["service_ratio"] is not None:
            step_data["service_ratio"] = info["service_ratio"]
        if "total_power" in info and info["total_power"] is not None:
            step_data["total_power"] = info["total_power"]
        if "served_people" in info and info["served_people"] is not None:
            step_data["served_people"] = info["served_people"]
        
        step_metrics.append(step_data)

        if cfg.dry_run:
            done = True
            
    fabric.print("Test - Reward:", cumulative_rew)
    fabric.print("Test - Episode Length:", episode_length)
    
    if cfg.metric.log_level > 0:
        metrics_dict = {"Test/cumulative_reward": cumulative_rew, "Test/episode_length": episode_length}
        
        metrics_avg = {}
        for key in ["observed_ratio", "observed_area_ratio", "service_ratio", "reward", "total_power", "served_people"]:
            values = [step[key] for step in step_metrics if key in step]
            if values:
                metrics_avg[f"Test/{key}_mean"] = np.mean(values)
                if key == "reward" and len(values) > 1:
                    metrics_avg[f"Test/{key}_std"] = np.std(values)
        
        metrics_dict.update(metrics_avg)
        fabric.log_dict(metrics_dict, 0)
        
        for step_idx, step_data in enumerate(step_metrics):
            step_metrics_dict = {}
            for key, value in step_data.items():
                if key != "step":
                    step_metrics_dict[f"Test_Steps/{key}"] = value
            
            if step_metrics_dict:
                fabric.log_dict(step_metrics_dict, step_idx)
    
    env.close()


def normalize_obs(
    obs: Dict[str, np.ndarray | Tensor], cnn_keys: Sequence[str], obs_keys: Sequence[str]
) -> Dict[str, np.ndarray | Tensor]:
    normalized_obs = {}
    for k in obs_keys:
        if k in cnn_keys:
            if k == "density_matrix":
                normalized_obs[k] = obs[k]
            else:
                normalized_obs[k] = obs[k] / 255.0 - 0.5
        else:
            normalized_obs[k] = obs[k]
    return normalized_obs


def log_models(
    cfg: Dict[str, Any],
    models_to_log: Dict[str, torch.nn.Module | _FabricModule],
    run_id: str,
    experiment_id: str | None = None,
    run_name: str | None = None,
) -> Dict[str, "ModelInfo"]:
    if not _IS_MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_IS_MLFLOW_AVAILABLE))
    import mlflow  # noqa

    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id, run_name=run_name, nested=True) as _:
        model_info = {}
        unwrapped_models = {}
        for k in cfg.model_manager.models.keys():
            if k not in models_to_log:
                warnings.warn(f"Model {k} not found in models_to_log, skipping.", category=UserWarning)
                continue
            unwrapped_models[k] = unwrap_fabric(models_to_log[k])
            model_info[k] = mlflow.pytorch.log_model(unwrapped_models[k], artifact_path=k)
        mlflow.log_dict(cfg, "config.json")
    return model_info


def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence["ModelInfo"]:
    if not _IS_MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_IS_MLFLOW_AVAILABLE))
    import mlflow  # noqa

    # Create the models
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        env.action_space.shape
        if is_continuous
        else (env.action_space.nvec.tolist() if is_multidiscrete else [env.action_space.n])
    )
    agent = build_agent(fabric, actions_dim, is_continuous, cfg, env.observation_space, state["agent"])

    # Log the model, create a new run if `cfg.run_id` is None.
    model_info = {}
    with mlflow.start_run(run_id=cfg.run.id, experiment_id=cfg.experiment.id, run_name=cfg.run.name, nested=True) as _:
        model_info["agent"] = mlflow.pytorch.log_model(unwrap_fabric(agent), artifact_path="agent")
        mlflow.log_dict(cfg.to_log, "config.json")
    return model_info
