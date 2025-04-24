from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric
from torch import Tensor, nn
from lightning.fabric.wrappers import _FabricModule
from torch.distributions import Categorical

from sheeprl.utils.env import make_env
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE
from sheeprl.utils.utils import unwrap_fabric

# 导入函数前，定义一个前向声明以防止循环导入
normalize_density = None
from sheeprl.algos.dreamer_v3_mod import dreamer_v3_mod
normalize_density = dreamer_v3_mod.normalize_density

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo

    from sheeprl.algos.dreamer_v3_mod.agent import PlayerDV3

AGGREGATOR_KEYS = {
    "Rewards/rew_avg",
    "Game/ep_len_avg",
    "Loss/world_model_loss",
    "Loss/value_loss",
    "Loss/policy_loss",
    "Loss/observation_loss",
    "Loss/reward_loss",
    "Loss/state_loss",
    "Loss/continue_loss",
    "State/kl",
    "State/post_entropy",
    "State/prior_entropy",
    "Grads/world_model",
    "Grads/actor",
    "Grads/critic",
    # 训练过程中的环境指标
    "Metrics/observed_ratio",
    "Metrics/area_ratio",
    "Metrics/reward",
    "Metrics/total_power",
    "Metrics/served_people",
    "Metrics/service_ratio",
    # 测试过程中的环境指标（汇总值）
    "Test/cumulative_reward",
    "Test/episode_length",
    "Test/observed_ratio_mean",
    "Test/observed_area_ratio_mean",
    "Test/service_ratio_mean",
    "Test/reward_mean",
    "Test/reward_std",
    "Test/total_power_mean",
    "Test/served_people_mean",
    # 测试过程中的环境指标（按步骤记录）
    "Test_Steps/reward",
    "Test_Steps/observed_ratio",
    "Test_Steps/observed_area_ratio",
    "Test_Steps/service_ratio",
    "Test_Steps/total_power",
    "Test_Steps/served_people"
}
MODELS_TO_REGISTER = {"world_model", "actor", "critic", "target_critic", "moments"}


class Moments(nn.Module):
    def __init__(
        self,
        decay: float = 0.99,
        max_: float = 1e8,
        percentile_low: float = 0.05,
        percentile_high: float = 0.95,
    ) -> None:
        super().__init__()
        self._decay = decay
        self._max = torch.tensor(max_)
        self._percentile_low = percentile_low
        self._percentile_high = percentile_high
        self.register_buffer("low", torch.zeros((), dtype=torch.float32))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32))

    def forward(self, x: Tensor, fabric: Fabric) -> Any:
        gathered_x = fabric.all_gather(x).float().detach()
        low = torch.quantile(gathered_x, self._percentile_low)
        high = torch.quantile(gathered_x, self._percentile_high)
        self.low = self._decay * self.low + (1 - self._decay) * low
        self.high = self._decay * self.high + (1 - self._decay) * high
        invscale = torch.max(1 / self._max, self.high - self.low)
        return self.low.detach(), invscale.detach()


def compute_lambda_values(
    rewards: Tensor,
    values: Tensor,
    continues: Tensor,
    lmbda: float = 0.95,
):
    vals = [values[-1:]]
    interm = rewards + continues * values * (1 - lmbda)
    for t in reversed(range(len(continues))):
        vals.append(interm[t] + continues[t] * lmbda * vals[-1])
    ret = torch.cat(list(reversed(vals))[:-1])
    return ret


def prepare_obs(
    fabric: Fabric, obs: Dict[str, np.ndarray], *, cnn_keys: Sequence[str] = [], num_envs: int = 1, **kwargs
) -> Dict[str, Tensor]:
    torch_obs = {}
    for k, v in obs.items():
        torch_obs[k] = torch.from_numpy(v.copy()).to(fabric.device).float()
        if k in cnn_keys:
            if k == "density_matrix":  # 对密度矩阵使用特殊的归一化
                torch_obs[k] = normalize_density(torch_obs[k].view(1, num_envs, -1, *v.shape[-2:]))
            else:  # 对其他CNN输入使用标准图像归一化
                torch_obs[k] = torch_obs[k].view(1, num_envs, -1, *v.shape[-2:]) / 255.0 - 0.5
        else:
            torch_obs[k] = torch_obs[k].view(1, num_envs, -1)
    return torch_obs


@torch.no_grad()
def test(
    player: "PlayerDV3",
    fabric: Fabric,
    cfg: Dict[str, Any],
    log_dir: str,
    test_name: str = "",
    greedy: bool = True,
):
    """Test the model on the environment with the frozen model.

    Args:
        player (PlayerDV3): the agent which contains all the models needed to play.
        fabric (Fabric): the fabric instance.
        cfg (DictConfig): the hyper-parameters.
        log_dir (str): the logging directory.
        test_name (str): the name of the test.
            Default to "".
        greedy (bool): whether or not to sample the actions.
            Default to True.
    """
    env: gym.Env = make_env(cfg, cfg.seed, 0, log_dir, "test" + (f"_{test_name}" if test_name != "" else ""))()
    done = False
    cumulative_rew = 0
    obs, info = env.reset(seed=cfg.seed)
    player.num_envs = 1
    player.init_states()
    
    # 初始化新增的监控指标
    episode_length = 0
    step_metrics = []  # 存储每步的指标
    
    # 记录初始info中的数据
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
    
    # 添加初始步骤的指标
    step_metrics.append(step_data)
    
    while not done:
        # Act greedly through the environment
        torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder)
        real_actions = player.get_actions(
            torch_obs, greedy, {k: v for k, v in torch_obs.items() if k.startswith("mask")}
        )
        if player.actor.is_continuous:
            real_actions = torch.stack(real_actions, -1).cpu().numpy()
        else:
            real_actions = torch.stack([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()

        # Single environment step
        obs, reward, done, truncated, info = env.step(real_actions.reshape(env.action_space.shape))
        done = done or truncated or cfg.dry_run
        cumulative_rew += reward
        episode_length += 1
        
        # 记录每步的环境信息
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
        
        # 添加当前步骤的指标
        step_metrics.append(step_data)
    
    fabric.print("Test - Reward:", cumulative_rew)
    fabric.print("Test - Episode Length:", episode_length)
    
    if cfg.metric.log_level > 0 and len(fabric.loggers) > 0:
        # 记录基本评估指标
        metrics_dict = {"Test/cumulative_reward": cumulative_rew, "Test/episode_length": episode_length}
        
        # 计算整体平均值作为汇总指标
        metrics_avg = {}
        for key in ["observed_ratio", "observed_area_ratio", "service_ratio", "reward", "total_power", "served_people"]:
            values = [step[key] for step in step_metrics if key in step]
            if values:
                metrics_avg[f"Test/{key}_mean"] = np.mean(values)
                if key == "reward" and len(values) > 1:
                    metrics_avg[f"Test/{key}_std"] = np.std(values)
        
        # 记录汇总指标
        metrics_dict.update(metrics_avg)
        fabric.logger.log_metrics(metrics_dict, 0)
        
        # 记录每步的指标，这样在TensorBoard中可以看到随时间变化的曲线
        for step_idx, step_data in enumerate(step_metrics):
            step_metrics_dict = {}
            for key, value in step_data.items():
                if key != "step":  # 排除step本身
                    step_metrics_dict[f"Test_Steps/{key}"] = value
            
            if step_metrics_dict:
                # 使用step_idx作为x轴
                fabric.logger.log_metrics(step_metrics_dict, step_idx)
    
    env.close()


# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L929
def init_weights(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0, b=2.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L957
def uniform_init_weights(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence["ModelInfo"]:
    if not _IS_MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_IS_MLFLOW_AVAILABLE))
    import mlflow  # noqa

    from sheeprl.algos.dreamer_v3.agent import build_agent

    # Create the models
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        env.action_space.shape
        if is_continuous
        else (env.action_space.nvec.tolist() if is_multidiscrete else [env.action_space.n])
    )
    world_model, actor, critic, target_critic = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        env.observation_space,
        state["world_model"],
        state["actor"],
        state["critic"],
        state["target_critic"],
    )
    moments = Moments(
        fabric,
        cfg.algo.actor.moments.decay,
        cfg.algo.actor.moments.max,
        cfg.algo.actor.moments.percentile.low,
        cfg.algo.actor.moments.percentile.high,
    )
    moments.load_state_dict(state["moments"])

    # Log the model, create a new run if `cfg.run_id` is None.
    model_info = {}
    with mlflow.start_run(run_id=cfg.run.id, experiment_id=cfg.experiment.id, run_name=cfg.run.name, nested=True) as _:
        model_info["world_model"] = mlflow.pytorch.log_model(unwrap_fabric(world_model), artifact_path="world_model")
        model_info["actor"] = mlflow.pytorch.log_model(unwrap_fabric(actor), artifact_path="actor")
        model_info["critic"] = mlflow.pytorch.log_model(unwrap_fabric(critic), artifact_path="critic")
        model_info["target_critic"] = mlflow.pytorch.log_model(target_critic, artifact_path="target_critic")
        model_info["moments"] = mlflow.pytorch.log_model(moments, artifact_path="moments")
        mlflow.log_dict(cfg.to_log, "config.json")
    return model_info
